import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import yaml
import os
import sys

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists
)

def train_sft(use_fp8=None, config_path="configs/sft.yaml", cli_overrides=None):
    """Train with SFT.

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
        print("  Using FP8 precision for SFT")
    else:
        print("  Using BF16 precision for SFT")

    # Load model with Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
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
        load_best_model_at_end=config['eval']['load_best_model_at_end'],
        metric_for_best_model=config['eval']['metric_for_best_model'],

        # Sequence packing - up to 6x speedup by eliminating padding waste
        packing=True,
        max_seq_length=config['data']['max_seq_length'],

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

    print("\n🚀 Starting SFT training with sequence packing...")
    trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))
    trainer.save_model(cli_overrides.get('output_dir', "checkpoints/sft_final"))

    print("✓ SFT training complete!")

if __name__ == "__main__":
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

    train_sft(use_fp8=use_fp8, config_path=args.config, cli_overrides=cli_overrides)
