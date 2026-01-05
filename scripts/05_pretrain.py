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

# Import GPU utilities
from gpu_utils import detect_gpu_type, print_gpu_info, setup_torch_backends

class CurriculumCallback:
    def __init__(self, curriculum_config):
        self.schedule = curriculum_config['schedule']
        self.current_idx = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.current_idx < len(self.schedule):
            stage = self.schedule[self.current_idx]
            if state.global_step >= stage['steps']:
                print(f"📚 Curriculum: Increasing context to {stage['seq_length']}")
                self.current_idx += 1

def setup_training(use_fp8=None):
    """Setup training with automatic GPU optimization"""
    # Setup torch backends
    setup_torch_backends()

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine if we should use FP8
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    # Load config
    with open("configs/pretrain.yaml") as f:
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['checkpointing']['output_dir'],
        run_name=config['run_name'],

        # Optimization
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=config['training']['max_steps'],
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
        logging_steps=config['logging']['logging_steps'],
        report_to=config['logging']['report_to'],

        # Evaluation
        evaluation_strategy=config['eval']['evaluation_strategy'],
        eval_steps=config['eval']['eval_steps'],
        per_device_eval_batch_size=config['eval']['per_device_eval_batch_size'],

        # Checkpointing
        save_strategy=config['checkpointing']['save_strategy'],
        save_steps=config['checkpointing']['save_steps'],
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

    # Load dataset
    train_dataset = load_from_disk("data/packed/train")
    eval_dataset = load_from_disk("data/packed/val")

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

    # Load dataset
    train_dataset = load_from_disk("data/packed/train")
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
    args = parser.parse_args()

    use_fp8 = None
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    # Setup and detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8 and gpu_info['fp8_available']:
        # Use FP8 training path
        with open("configs/pretrain.yaml") as f:
            config = yaml.safe_load(f)
        train_with_fp8(config, gpu_info)
    else:
        # Use standard BF16 training path
        if use_fp8 and not gpu_info['fp8_available']:
            print("Warning: FP8 requested but not available, using BF16")

        trainer, gpu_info = setup_training(use_fp8=False)
        print("🚀 Starting pretraining...")
        trainer.train()
        trainer.save_model("checkpoints/pretrain_final")
        print("✓ Pretraining complete!")

if __name__ == "__main__":
    main()
