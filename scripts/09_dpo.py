import torch
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import yaml
import os
import sys

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists
)

def train_dpo(use_fp8=None, config_path="configs/dpo.yaml", cli_overrides=None):
    """Train with DPO.

    Args:
        use_fp8: Force FP8 precision (None = auto-detect)
        config_path: Path to YAML config file
        cli_overrides: Dict of CLI overrides (max_steps, save_steps, etc.)
    """
    cli_overrides = cli_overrides or {}

    # Setup torch backends
    setup_torch_backends()

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

    # Load model with Flash Attention
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load reference model (no need to compile - used only for inference)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing on main model only
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Compile main model for speedup (ref_model doesn't need compilation)
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    # Validate data paths (use config if available, fallback to defaults)
    train_data_path = config.get('data', {}).get('train_data', "data/dpo/train")
    eval_data_path = config.get('data', {}).get('val_data', "data/dpo/val")

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

    print("\n🚀 Starting DPO training...")
    trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))
    trainer.save_model(cli_overrides.get('output_dir', "checkpoints/dpo_final"))

    print("✓ DPO training complete!")

if __name__ == "__main__":
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

    train_dpo(use_fp8=use_fp8, config_path=args.config, cli_overrides=cli_overrides)
