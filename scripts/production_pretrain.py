"""Production 7B model training - Optimized for A100/H100 with FP8 support"""
import torch
import os
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer,
    AutoModelForCausalLM, DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

# Check for FP8 support (requires transformer-engine)
def check_fp8_available():
    """Check if FP8 training is available (H100 + transformer-engine)"""
    try:
        import transformer_engine
        # Check CUDA compute capability (H100 is sm_90)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # H100 has compute capability 9.0
            if capability[0] >= 9:
                return True
    except ImportError:
        pass
    return False

def detect_gpu_type():
    """Detect GPU type and return optimized settings"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability()
        is_h100 = "H100" in gpu_name or capability[0] >= 9
        is_a100 = "A100" in gpu_name or (capability[0] == 8 and capability[1] == 0)
        fp8_available = check_fp8_available()

        return {
            "gpu_name": gpu_name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "is_h100": is_h100,
            "is_a100": is_a100,
            "fp8_available": fp8_available and is_h100,
            "compile_mode": "max-autotune" if is_h100 else "default",
            "batch_size": 8,
        }
    return {
        "gpu_name": "CPU",
        "compute_capability": "N/A",
        "is_h100": False,
        "is_a100": False,
        "fp8_available": False,
        "compile_mode": "default",
        "batch_size": 4
    }

def load_production_data():
    """Load production training data"""
    print("Loading production dataset...")
    data = np.load("data/packed/production_pretrain.npy")
    print(f"  Loaded {len(data):,} sequences of {data.shape[1]} tokens")

    # Split train/val
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = Dataset.from_dict({"input_ids": train_data.tolist()})
    val_dataset = Dataset.from_dict({"input_ids": val_data.tolist()})

    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    return train_dataset, val_dataset

def production_training_fp8(max_steps=100000, gpu_info=None):
    """Run production training with FP8 precision (H100 only)"""
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs

    print("\n" + "="*60)
    print("PRODUCTION 7B MODEL TRAINING (FP8 MODE)")
    print("="*60 + "\n")

    # Configure FP8
    fp8_kwargs = FP8RecipeKwargs(
        backend="te",  # Transformer Engine backend
        fp8_format="HYBRID",  # E4M3 for forward, E5M2 for backward
        amax_history_len=1024,
        amax_compute_algo="max",
    )

    accelerator = Accelerator(
        mixed_precision="fp8",
        kwargs_handlers=[fp8_kwargs],
        gradient_accumulation_steps=4,
    )

    print(f"FP8 Configuration:")
    print(f"  Backend: Transformer Engine")
    print(f"  Format: HYBRID (E4M3 forward, E5M2 backward)")
    print(f"  Expected speedup: 30-40% over BF16")

    # Load model
    print("\nLoading 7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/init",
        torch_dtype=torch.bfloat16,  # Base dtype before FP8 conversion
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    print(f"  Model: {model.num_parameters() / 1e9:.2f}B parameters")

    # Load data
    train_dataset, val_dataset = load_production_data()

    # Create data loader
    from torch.utils.data import DataLoader

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Learning rate scheduler
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5000,
        num_training_steps=max_steps,
    )

    # Prepare with accelerator (applies FP8)
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    print(f"\nProduction Configuration (FP8):")
    print(f"  Precision: FP8 (E4M3/E5M2 HYBRID)")
    print(f"  Batch size: 8")
    print(f"  Gradient accumulation: 4")
    print(f"  Effective batch: 32")
    print(f"  Max steps: {max_steps:,}")

    # Training loop
    print(f"\n🚀 Starting FP8 training ({max_steps:,} steps)...")
    print(f"   Expected GPU utilization: 95-100%")
    print(f"   Estimated time: ~{max_steps * 1.2 / 3600:.1f} hours (30-40% faster than BF16)\n")

    model.train()
    global_step = 0

    from tqdm import tqdm
    progress_bar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if global_step % 1000 == 0:
                    # Save checkpoint
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        f"checkpoints/production_pretrain/checkpoint-{global_step}",
                        save_function=accelerator.save,
                    )

                if global_step >= max_steps:
                    break

    progress_bar.close()

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "checkpoints/production_pretrain_final",
        save_function=accelerator.save,
    )

    print(f"\n✓ FP8 Training complete!")
    print(f"  Checkpoint: checkpoints/production_pretrain_final/")

def production_training_bf16(max_steps=100000, gpu_info=None):
    """Run production training with BF16 precision (A100/H100)"""
    print("\n" + "="*60)
    print("PRODUCTION 7B MODEL TRAINING (BF16 MODE)")
    print("="*60 + "\n")

    # Load model
    print("Loading 7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/init",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Compile model for speedup (PyTorch 2.x)
    compile_mode = gpu_info['compile_mode'] if gpu_info else "default"
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    print(f"  Model: {model.num_parameters() / 1e9:.2f}B parameters")

    # Load data
    train_dataset, val_dataset = load_production_data()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/production_pretrain",
        run_name="production-7b-pretrain",

        # Training schedule
        max_steps=max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,

        # Learning rate schedule
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5000,

        # Precision & optimization
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        weight_decay=0.1,

        # Logging & evaluation
        logging_steps=10,
        logging_dir="logs/production_pretrain",
        eval_strategy="steps",
        eval_steps=1000,
        report_to=["tensorboard"],

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,

        # Data loading
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # PyTorch 2.x compilation
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode=compile_mode,
    )

    print(f"\nProduction Configuration (BF16):")
    print(f"  Precision: BF16 + TF32")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Compile mode: {compile_mode}")

    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    est_hours = max_steps * 1.8 / 3600 if not gpu_info.get('is_h100') else max_steps * 1.4 / 3600
    print(f"\n🚀 Starting BF16 training ({max_steps:,} steps)...")
    print(f"   Expected GPU utilization: 95-100%")
    print(f"   Optimizations: torch.compile, Flash Attention 2, persistent workers")
    print(f"   Estimated time: ~{est_hours:.1f} hours\n")

    trainer.train()

    # Save final model
    trainer.save_model("checkpoints/production_pretrain_final")
    print(f"\n✓ Training complete!")
    print(f"  Checkpoint: checkpoints/production_pretrain_final/")

def production_training(max_steps=100000, use_fp8=None):
    """Run full production training with automatic precision selection"""

    # Detect GPU and capabilities
    gpu_info = detect_gpu_type()
    print(f"GPU detected: {gpu_info['gpu_name']}")
    print(f"  Compute capability: {gpu_info['compute_capability']}")

    # Determine precision mode
    if use_fp8 is None:
        # Auto-detect: use FP8 if available on H100
        use_fp8 = gpu_info['fp8_available']

    if use_fp8 and not gpu_info['fp8_available']:
        print("  Warning: FP8 requested but not available (requires H100 + transformer-engine)")
        print("  Falling back to BF16")
        use_fp8 = False

    if gpu_info['is_h100']:
        if use_fp8:
            print("  H100 detected - using FP8 precision for maximum performance")
            production_training_fp8(max_steps, gpu_info)
        else:
            print("  H100 detected - using BF16 precision (install transformer-engine for FP8)")
            production_training_bf16(max_steps, gpu_info)
    elif gpu_info['is_a100']:
        print("  A100 detected - using BF16 precision")
        production_training_bf16(max_steps, gpu_info)
    else:
        print("  Using BF16 precision")
        production_training_bf16(max_steps, gpu_info)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Production 7B model training")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    args = parser.parse_args()

    use_fp8 = None  # Auto-detect
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    production_training(max_steps=args.max_steps, use_fp8=use_fp8)
