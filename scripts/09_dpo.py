from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional
from collections import OrderedDict

# Default wandb to offline mode (avoids interactive prompt)
# Set WANDB_MODE=online explicitly to enable cloud sync
if 'WANDB_MODE' not in os.environ:
    os.environ['WANDB_MODE'] = 'offline'

# Disable tokenizers parallelism warning when using dataloader workers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import yaml
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_from_disk
from safetensors.torch import load_file as load_safetensors

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists, OOMHandler
)
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def load_compiled_checkpoint(
    checkpoint_path: str,
    use_flash_attention: bool = True
) -> AutoModelForCausalLM:
    """Load a checkpoint that may have been saved with torch.compile wrapper.

    When a model is saved after torch.compile(), the state dict keys have
    '_orig_mod.' prefix. This function handles both compiled and non-compiled
    checkpoints transparently.

    Args:
        checkpoint_path: Path to the model checkpoint
        use_flash_attention: Enable Flash Attention 2

    Returns:
        Loaded model with correct weights
    """
    import glob

    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path)

    # Check for safetensors or pytorch format
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path)
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        # Try sharded safetensors
        shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors")))
        if shard_files:
            state_dict = {}
            for shard in shard_files:
                state_dict.update(load_safetensors(shard))
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    # Check if state dict has _orig_mod. prefix (from torch.compile)
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if has_orig_mod:
        print("  Detected torch.compile checkpoint, stripping _orig_mod. prefix...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_key = k[len("_orig_mod."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Create model with flash attention
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.load_state_dict(state_dict, strict=True)

    if use_flash_attention:
        print(f"  Flash Attention 2: ENABLED")

    return model


class OOMRecoveryCallback(TrainerCallback):
    """Callback for logging OOM recovery events during DPO training."""

    def __init__(self) -> None:
        self.handler = OOMHandler()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        if self.handler.oom_count > 0 and logs is not None:
            logs["oom_recovery/total_events"] = self.handler.oom_count


def train_dpo(
    use_fp8: Optional[bool] = None,
    config_path: str = "configs/dpo.yaml",
    cli_overrides: Optional[Dict[str, Any]] = None,
    use_liger_kernel: bool = True
) -> None:
    """Train with DPO.

    Args:
        use_fp8: Force FP8 precision (None = auto-detect)
        config_path: Path to YAML config file
        cli_overrides: Dict of CLI overrides (max_steps, save_steps, etc.)
        use_liger_kernel: Enable Liger Kernel for fused operations
    """
    cli_overrides = cli_overrides or {}

    # Setup torch backends
    setup_torch_backends()

    # Apply Liger Kernel optimizations if enabled
    if use_liger_kernel:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama()
            print("[Kernel Optimization] Liger Kernel enabled for llama")
            print("  - Fused RoPE, SwiGLU, RMSNorm, CrossEntropy")
            print("  - ~20% throughput improvement")
            print("  - ~60% memory reduction")
        except ImportError:
            print("Warning: liger-kernel not installed, skipping kernel optimizations")
        except Exception as e:
            print(f"Warning: Failed to apply Liger Kernel: {e}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8:
        print("  Using FP8 precision for DPO")
    else:
        print("  Using BF16 precision for DPO")

    # Load model (handles torch.compile checkpoints with _orig_mod. prefix)
    checkpoint_path = config['model']['checkpoint']
    print(f"Loading model from {checkpoint_path}...")
    model = load_compiled_checkpoint(checkpoint_path, use_flash_attention=True)

    # Load reference model (no need to compile - used only for inference)
    print("Loading reference model...")
    ref_model = load_compiled_checkpoint(checkpoint_path, use_flash_attention=True)

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing on main model only
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Compile main model for speedup (ref_model doesn't need compilation)
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    # Validate data paths (CLI overrides take precedence)
    train_data_path = cli_overrides.get('train_data_path') or config.get('data', {}).get('train_data', "data/dpo/train")
    eval_data_path = cli_overrides.get('eval_data_path') or config.get('data', {}).get('val_data', "data/dpo/val")

    if not os.path.exists(train_data_path):
        print(f"Error: DPO training data not found at {train_data_path}")
        print("\nTo prepare DPO data, run:")
        print("  python scripts/08_prepare_dpo_data.py")
        sys.exit(1)

    if not os.path.exists(eval_data_path):
        print(f"Error: DPO validation data not found at {eval_data_path}")
        sys.exit(1)

    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path)

    training_args = DPOConfig(
        output_dir=cli_overrides.get('output_dir', "checkpoints/dpo"),
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=cli_overrides.get('max_steps', config['training'].get('max_steps', -1)),
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        beta=config['training']['beta'],
        loss_type=config['training']['loss_type'],
        max_length=config['training']['max_length'],
        max_prompt_length=config['training']['max_prompt_length'],

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # PyTorch 2.x compilation
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode=gpu_info['compile_mode'],

        # Logging
        logging_steps=cli_overrides.get('logging_steps', config['logging']['logging_steps']),
        eval_steps=cli_overrides.get('eval_steps', config['eval']['eval_steps']),
        eval_strategy="steps",
        save_steps=cli_overrides.get('save_steps', config['logging'].get('save_steps', 200)),
        save_total_limit=3,
    )

    print(f"\nDPO Configuration:")
    print(f"  Beta (KL penalty): {config['training']['beta']}")
    print(f"  Loss type: {config['training']['loss_type']}")
    print(f"  Max length: {config['training']['max_length']}")
    print(f"  Compile mode: {gpu_info['compile_mode']}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Add OOM recovery callback if enabled
    if cli_overrides.get('enable_oom_recovery', False):
        print("OOM recovery: ENABLED")
        trainer.add_callback(OOMRecoveryCallback())

    print("\n🚀 Starting DPO training...")
    trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))

    # Always derive final_output_dir from output_dir by appending _final
    # This ensures dpo_final is created even if --output_dir is specified
    output_dir = cli_overrides.get('output_dir', "checkpoints/dpo")
    if output_dir.endswith('_final'):
        final_output_dir = output_dir
    else:
        final_output_dir = f"{output_dir.rstrip('/')}_final"
    print(f"\nSaving model to {final_output_dir}...")
    trainer.save_model(final_output_dir)

    print("✓ DPO training complete!")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DPO training for 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    # Config overrides
    parser.add_argument("--config", type=str, default="configs/dpo.yaml", help="Path to config file")
    parser.add_argument("--max_steps", type=int, help="Override max training steps")
    parser.add_argument("--save_steps", type=int, help="Override checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, help="Override evaluation frequency")
    parser.add_argument("--logging_steps", type=int, help="Override logging frequency")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_data_path", type=str, help="Override training data path")
    parser.add_argument("--eval_data_path", type=str, help="Override evaluation data path")
    parser.add_argument("--enable-oom-recovery", action="store_true", help="Enable automatic OOM recovery")
    parser.add_argument("--use-liger-kernel", action="store_true", default=True,
                        help="Use Liger Kernel for fused operations (default: enabled)")
    parser.add_argument("--no-liger-kernel", action="store_true",
                        help="Disable Liger Kernel")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Validate prerequisites
    if not check_tokenizer_exists():
        sys.exit(1)
    if not check_checkpoint_exists("checkpoints/sft_final", "SFT model"):
        print("\nTo create SFT checkpoint, run:")
        print("  python scripts/07_sft.py")
        sys.exit(1)

    use_fp8 = None
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    # Build CLI overrides dict
    cli_overrides = {}
    if args.max_steps is not None:
        cli_overrides['max_steps'] = args.max_steps
    if args.save_steps is not None:
        cli_overrides['save_steps'] = args.save_steps
    if args.eval_steps is not None:
        cli_overrides['eval_steps'] = args.eval_steps
    if args.logging_steps is not None:
        cli_overrides['logging_steps'] = args.logging_steps
    if args.output_dir is not None:
        cli_overrides['output_dir'] = args.output_dir
    if args.resume_from_checkpoint is not None:
        cli_overrides['resume_from_checkpoint'] = args.resume_from_checkpoint
    if args.train_data_path is not None:
        cli_overrides['train_data_path'] = args.train_data_path
    if args.eval_data_path is not None:
        cli_overrides['eval_data_path'] = args.eval_data_path
    if getattr(args, 'enable_oom_recovery', False):
        cli_overrides['enable_oom_recovery'] = True

    # Determine if Liger Kernel should be used
    use_liger = getattr(args, 'use_liger_kernel', True) and not getattr(args, 'no_liger_kernel', False)

    train_dpo(use_fp8=use_fp8, config_path=args.config, cli_overrides=cli_overrides, use_liger_kernel=use_liger)


if __name__ == "__main__":
    main()
