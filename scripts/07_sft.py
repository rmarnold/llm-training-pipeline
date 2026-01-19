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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
from safetensors.torch import load_file as load_safetensors

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists, OOMHandler
)
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile() wrapped model.

    When a model is wrapped with torch.compile(), the original model is stored
    in the _orig_mod attribute. This function returns the original model if
    compiled, or the same model if not.

    Args:
        model: A potentially compiled model

    Returns:
        The unwrapped model (or the same model if not compiled)
    """
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


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
        use_flash_attention: Enable Flash Attention 2 (required for packing)

    Returns:
        Loaded model with correct weights
    """
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
        import glob
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

    # Create model with flash attention (required for packing to avoid cross-contamination)
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.load_state_dict(state_dict, strict=True)

    if use_flash_attention:
        print(f"  Flash Attention 2: ENABLED (required for packing)")

    return model


class OOMRecoveryCallback(TrainerCallback):
    """Callback for logging OOM recovery events during SFT training."""

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


class EvalLossCallback(TrainerCallback):
    """Callback to compute eval_loss for SFTTrainer since it doesn't with packing.

    TRL's SFTTrainer doesn't compute eval_loss when using pre-formatted text
    datasets. This callback manually computes it during evaluation and tracks
    the best checkpoint for optional model selection.
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        max_length: int,
        eval_batch_size: int = 2,
        num_eval_samples: int = 500,
        output_dir: str = "checkpoints/sft",
    ) -> None:
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size
        self.num_eval_samples = min(num_eval_samples, len(eval_dataset))
        self.output_dir = output_dir
        self.best_eval_loss = float('inf')
        self.best_step = 0

    def _get_base_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap compiled/distributed model wrappers to get base model."""
        # Handle torch.compile wrapper
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        # Handle DDP/FSDP wrapper
        if hasattr(model, 'module'):
            model = model.module
        return model

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs: Any
    ) -> None:
        """Compute eval_loss manually after each evaluation."""
        base_model = self._get_base_model(model)
        was_training = base_model.training
        base_model.eval()
        device = next(base_model.parameters()).device

        # Sample subset for efficiency - get all texts upfront
        eval_subset = self.eval_dataset.select(range(self.num_eval_samples))
        all_texts = eval_subset["text"]

        # Filter out empty texts
        all_texts = [t for t in all_texts if t and isinstance(t, str) and len(t.strip()) > 10]

        if not all_texts:
            if state.is_world_process_zero:
                print(f"\n  ⚠️  No valid eval texts found, skipping eval_loss computation")
            return

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, len(all_texts), self.eval_batch_size):
                batch_texts = all_texts[i:i + self.eval_batch_size]

                if not batch_texts:
                    continue

                try:
                    # Tokenize batch
                    encodings = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                        return_tensors="pt"
                    )

                    input_ids = encodings["input_ids"].to(device)
                    attention_mask = encodings["attention_mask"].to(device)

                    # Skip if no tokens
                    if input_ids.numel() == 0 or input_ids.shape[1] == 0:
                        continue

                    # Create labels (same as input_ids, with padding masked)
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100  # Ignore padding tokens in loss

                    # Forward pass
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # Accumulate loss weighted by token count
                    num_tokens = (labels != -100).sum().item()
                    if num_tokens > 0:
                        total_loss += outputs.loss.item() * num_tokens
                        total_tokens += num_tokens
                        num_batches += 1

                except Exception as e:
                    # Skip problematic batches but continue evaluation
                    if state.is_world_process_zero and num_batches == 0:
                        print(f"  Warning: Skipping batch due to error: {e}")
                    continue

        # Restore training mode if needed
        if was_training:
            base_model.train()

        # Compute average loss
        eval_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')

        # Log results
        if state.is_world_process_zero:
            print(f"\n  📊 eval_loss: {eval_loss:.4f} (computed on {self.num_eval_samples} samples)")

            # Track best model
            if eval_loss < self.best_eval_loss:
                improvement = self.best_eval_loss - eval_loss
                self.best_eval_loss = eval_loss
                self.best_step = state.global_step
                print(f"  ✓ New best! (improved by {improvement:.4f})")

                # Save best checkpoint marker
                best_marker_path = os.path.join(self.output_dir, "best_checkpoint.txt")
                os.makedirs(self.output_dir, exist_ok=True)
                with open(best_marker_path, "w") as f:
                    f.write(f"step={self.best_step}\n")
                    f.write(f"eval_loss={self.best_eval_loss:.6f}\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any
    ) -> None:
        """Print summary of best model at end of training."""
        if state.is_world_process_zero:
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Best eval_loss: {self.best_eval_loss:.4f} at step {self.best_step}")
            print(f"Best checkpoint: {self.output_dir}/checkpoint-{self.best_step}")
            print(f"{'='*60}\n")


def train_sft(
    use_fp8: Optional[bool] = None,
    config_path: str = "configs/sft.yaml",
    cli_overrides: Optional[Dict[str, Any]] = None,
    use_liger_kernel: bool = True
) -> None:
    """Train with SFT.

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
        print("  Using FP8 precision for SFT")
    else:
        print("  Using BF16 precision for SFT")

    # Load model (handles torch.compile checkpoints with _orig_mod. prefix)
    checkpoint_path = config['model']['checkpoint']
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = load_compiled_checkpoint(checkpoint_path, use_flash_attention=True)

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode
    if config['model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Compile model for speedup
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    # Validate data paths (CLI overrides take precedence)
    train_data_path = cli_overrides.get('train_data_path') or config['data']['train_data']
    eval_data_path = cli_overrides.get('eval_data_path') or config['data']['val_data']

    if not os.path.exists(train_data_path):
        print(f"Error: SFT training data not found at {train_data_path}")
        print("\nTo prepare SFT data, run:")
        print("  python scripts/06_prepare_sft_data.py")
        sys.exit(1)

    if not os.path.exists(eval_data_path):
        print(f"Error: SFT validation data not found at {eval_data_path}")
        sys.exit(1)

    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path)

    # Use SFTConfig for packing support (up to 6x speedup)
    max_seq_len = config['data']['max_seq_length']
    training_args = SFTConfig(
        output_dir=cli_overrides.get('output_dir', config['checkpointing']['output_dir']),
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=cli_overrides.get('max_steps', config['training'].get('max_steps', -1)),
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        logging_steps=cli_overrides.get('logging_steps', config['logging']['logging_steps']),
        eval_steps=cli_overrides.get('eval_steps', config['eval']['eval_steps']),
        save_steps=cli_overrides.get('save_steps', config['logging']['save_steps']),
        eval_strategy=config['eval']['evaluation_strategy'],
        # Note: TRL's SFTTrainer doesn't compute eval_loss with pre-formatted text datasets
        # We disable load_best_model_at_end and rely on training loss monitoring
        # For SFT (short training on curated data), this is standard practice
        load_best_model_at_end=False,

        # Sequence packing - up to 6x speedup by eliminating padding waste
        packing=True,
        max_length=max_seq_len,  # TRL uses max_length, not max_seq_length

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # PyTorch 2.x compilation - disabled for SFT (compile after loading weights)
        torch_compile=False,  # We compile manually after loading
    )

    print(f"\nSFT Configuration:")
    print(f"  Packing: ENABLED (up to 6x speedup)")
    print(f"  Max sequence length: {config['data']['max_seq_length']}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Compile mode: {gpu_info['compile_mode']}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Add eval_loss callback (computes eval_loss since SFTTrainer doesn't with packing)
    output_dir = cli_overrides.get('output_dir', config['checkpointing']['output_dir'])
    eval_callback = EvalLossCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        eval_batch_size=config['training']['per_device_train_batch_size'],
        num_eval_samples=min(500, len(eval_dataset)),
        output_dir=output_dir,
    )
    trainer.add_callback(eval_callback)
    print("Eval loss tracking: ENABLED (custom callback)")

    # Add OOM recovery callback if enabled
    if cli_overrides.get('enable_oom_recovery', False):
        print("OOM recovery: ENABLED")
        trainer.add_callback(OOMRecoveryCallback())

    print("\n🚀 Starting SFT training with sequence packing...")
    trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))

    # Save final model - unwrap compiled model to avoid _orig_mod. prefix in state dict
    # Always derive final_output_dir from output_dir by appending _final
    # This ensures sft_final is created even if --output_dir is specified
    if output_dir.endswith('_final'):
        final_output_dir = output_dir
    else:
        final_output_dir = f"{output_dir.rstrip('/')}_final"
    print(f"\nSaving model to {final_output_dir}...")
    unwrapped_model = unwrap_compiled_model(trainer.model)
    unwrapped_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Check if we should use the best checkpoint instead
    best_checkpoint_file = os.path.join(output_dir, "best_checkpoint.txt")
    if os.path.exists(best_checkpoint_file) and config['eval'].get('load_best_model_at_end', True):
        with open(best_checkpoint_file) as f:
            lines = f.read().strip().split('\n')
            best_info = dict(line.split('=') for line in lines)
            best_step = int(best_info['step'])
            best_loss = float(best_info['eval_loss'])

        best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{best_step}")
        if os.path.exists(best_checkpoint_path):
            print(f"\n📦 Copying best checkpoint (step {best_step}, loss {best_loss:.4f}) to final output...")
            import shutil
            # Copy best checkpoint to final location
            if os.path.exists(final_output_dir):
                shutil.rmtree(final_output_dir)
            shutil.copytree(best_checkpoint_path, final_output_dir)
            print(f"  Best model saved to: {final_output_dir}")

    print("✓ SFT training complete!")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SFT training for 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    # Config overrides
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="Path to config file")
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
    if not check_checkpoint_exists("checkpoints/pretrain_final", "Pretrained model"):
        print("\nTo create pretrained checkpoint, run:")
        print("  python scripts/05_pretrain.py")
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

    train_sft(use_fp8=use_fp8, config_path=args.config, cli_overrides=cli_overrides, use_liger_kernel=use_liger)


if __name__ == "__main__":
    main()
