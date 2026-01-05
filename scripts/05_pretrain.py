import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import yaml
import wandb
import os
import sys

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists
)

from transformers import TrainerCallback


class CurriculumCallback(TrainerCallback):
    """Callback for curriculum learning - gradually increases sequence length.

    This callback monitors training steps and logs when curriculum stages change.
    The actual sequence length changes are handled by the data loading pipeline,
    which should load different datasets for each curriculum stage.

    For true curriculum learning, ensure your data is prepared with different
    sequence lengths in separate directories (e.g., data/packed/train_512,
    data/packed/train_1024, data/packed/train_2048).
    """

    def __init__(self, curriculum_config):
        self.schedule = curriculum_config.get('schedule', [])
        self.current_idx = 0
        self.current_seq_length = self.schedule[0]['seq_length'] if self.schedule else 2048

    def on_step_begin(self, args, state, control, **kwargs):
        if self.current_idx < len(self.schedule):
            stage = self.schedule[self.current_idx]
            if state.global_step >= stage['steps']:
                self.current_seq_length = stage['seq_length']
                print(f"\n📚 Curriculum: Stage {self.current_idx + 1}/{len(self.schedule)}")
                print(f"   Context length increased to {stage['seq_length']} tokens")
                print(f"   (Note: Requires data reload for actual sequence length change)")
                self.current_idx += 1

    def on_train_begin(self, args, state, control, **kwargs):
        if self.schedule:
            print(f"\n📚 Curriculum Learning Enabled:")
            print(f"   Stages: {len(self.schedule)}")
            for i, stage in enumerate(self.schedule):
                print(f"   - Stage {i+1}: {stage['seq_length']} tokens @ step {stage['steps']}")

def setup_training(use_fp8=None, config_path="configs/pretrain.yaml", cli_overrides=None):
    """Setup training with automatic GPU optimization.

    Args:
        use_fp8: Force FP8 precision (None = auto-detect)
        config_path: Path to YAML config file
        cli_overrides: Dict of CLI overrides (max_steps, save_steps, etc.)
    """
    cli_overrides = cli_overrides or {}

    # Setup torch backends
    setup_torch_backends()

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine if we should use FP8
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize W&B (disabled in non-interactive environments)
    if os.getenv('WANDB_MODE') != 'disabled':
        wandb.init(
            project="llm-training",
            name=config['run_name'],
            config=config
        )

    # Load model and tokenizer
    attn_impl = "flash_attention_2" if config['model']['use_flash_attention'] else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode
    if config['model']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Compile model with GPU-appropriate mode
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    # Training arguments (with CLI overrides)
    training_args = TrainingArguments(
        output_dir=cli_overrides.get('output_dir', config['checkpointing']['output_dir']),
        run_name=config['run_name'],

        # Optimization
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=cli_overrides.get('max_steps', config['training']['max_steps']),
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        warmup_steps=config['training']['warmup_steps'],

        # Batch sizing
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        # Precision
        bf16=config['training']['bf16'],
        tf32=config['training']['tf32'],

        # Regularization
        max_grad_norm=config['training']['max_grad_norm'],
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],

        # Optimization
        optim=config['training']['optim'],
        fsdp=config['training']['fsdp'],
        fsdp_transformer_layer_cls_to_wrap=config['training']['fsdp_transformer_layer_cls_to_wrap'],

        # Logging
        logging_dir=config['logging']['logging_dir'],
        logging_steps=cli_overrides.get('logging_steps', config['logging']['logging_steps']),
        report_to=config['logging']['report_to'],

        # Evaluation
        evaluation_strategy=config['eval']['evaluation_strategy'],
        eval_steps=cli_overrides.get('eval_steps', config['eval']['eval_steps']),
        per_device_eval_batch_size=config['eval']['per_device_eval_batch_size'],

        # Checkpointing
        save_strategy=config['checkpointing']['save_strategy'],
        save_steps=cli_overrides.get('save_steps', config['checkpointing']['save_steps']),
        save_total_limit=config['logging']['save_total_limit'],

        # torch.compile
        torch_compile=config['training'].get('torch_compile', True),
        torch_compile_backend=config['training'].get('torch_compile_backend', 'inductor'),
        torch_compile_mode=compile_mode,

        # Data loading optimization
        dataloader_num_workers=config['data'].get('num_workers', 8),
        dataloader_pin_memory=config['data'].get('pin_memory', True),
        dataloader_persistent_workers=config['data'].get('persistent_workers', True),
    )

    # Load dataset (use config paths with fallback to defaults)
    train_data_path = config['data'].get('train_path', "data/packed/train")
    eval_data_path = config['data'].get('val_path', "data/packed/val")

    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        print("\nTo prepare training data, run:")
        print("  python scripts/01_download_data.py")
        print("  python scripts/02_clean_deduplicate_optimized.py")
        print("  python scripts/03_tokenize_and_pack.py")
        print("\nOr for demo: python scripts/setup_demo.py")
        sys.exit(1)

    if not os.path.exists(eval_data_path):
        print(f"Error: Validation data not found at {eval_data_path}")
        sys.exit(1)

    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add curriculum callback
    if config['curriculum']['enabled']:
        trainer.add_callback(CurriculumCallback(config['curriculum']))

    return trainer, gpu_info

def train_with_fp8(config, gpu_info):
    """Train using FP8 precision with Accelerate (H100 only)"""
    from gpu_utils import get_fp8_accelerator
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm

    print("\n" + "="*60)
    print("PRETRAINING WITH FP8 PRECISION")
    print("="*60)

    accelerator = get_fp8_accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )

    print(f"\nFP8 Configuration:")
    print(f"  Backend: Transformer Engine")
    print(f"  Format: HYBRID (E4M3 forward, E5M2 backward)")
    print(f"  Expected speedup: 30-40% over BF16")

    # Load model
    attn_impl = "flash_attention_2" if config['model']['use_flash_attention'] else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    if config['model']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Load dataset (use config paths with fallback to defaults)
    train_data_path = config['data'].get('train_path', "data/packed/train")
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        sys.exit(1)

    train_dataset = load_from_disk(train_data_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config['data'].get('num_workers', 8),
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['weight_decay'],
    )

    max_steps = config['training']['max_steps']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=max_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    print(f"\n🚀 Starting FP8 pretraining ({max_steps:,} steps)...")
    model.train()
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % config['checkpointing']['save_steps'] == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        f"{config['checkpointing']['output_dir']}/checkpoint-{global_step}",
                        save_function=accelerator.save,
                    )

                if global_step >= max_steps:
                    break

    progress_bar.close()

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "checkpoints/pretrain_final",
        save_function=accelerator.save,
    )
    print("✓ FP8 Pretraining complete!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    # Config overrides
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to config file")
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
    if not check_checkpoint_exists("checkpoints/init", "Initial model"):
        print("\nTo initialize the model, run:")
        print("  python scripts/04_init_model.py")
        print("\nOr for demo: python scripts/setup_demo.py")
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

    # Setup and detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8 and gpu_info['fp8_available']:
        # Use FP8 training path
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Apply CLI overrides to config
        if 'max_steps' in cli_overrides:
            config['training']['max_steps'] = cli_overrides['max_steps']
        train_with_fp8(config, gpu_info)
    else:
        # Use standard BF16 training path
        if use_fp8 and not gpu_info['fp8_available']:
            print("Warning: FP8 requested but not available, using BF16")

        trainer, gpu_info = setup_training(
            use_fp8=False,
            config_path=args.config,
            cli_overrides=cli_overrides
        )
        print("🚀 Starting pretraining...")
        trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))
        trainer.save_model(cli_overrides.get('output_dir', "checkpoints/pretrain_final"))
        print("✓ Pretraining complete!")

if __name__ == "__main__":
    main()
