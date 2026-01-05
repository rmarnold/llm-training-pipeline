"""Production 7B model training - Optimized for 85-95% GPU utilization"""
import torch
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer,
    AutoModelForCausalLM, DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

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

def detect_gpu_type():
    """Detect GPU type and return optimized settings"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        is_h100 = "H100" in gpu_name
        is_a100 = "A100" in gpu_name
        return {
            "gpu_name": gpu_name,
            "is_h100": is_h100,
            "is_a100": is_a100,
            "compile_mode": "max-autotune" if is_h100 else "default",
            "batch_size": 8 if is_h100 else 8,  # Can increase for H100
        }
    return {"gpu_name": "CPU", "is_h100": False, "is_a100": False, "compile_mode": "default", "batch_size": 4}

def production_training(max_steps=100000):
    """Run full production training"""
    print("\n" + "="*60)
    print("PRODUCTION 7B MODEL TRAINING")
    print("="*60 + "\n")

    # Detect GPU and optimize settings
    gpu_info = detect_gpu_type()
    print(f"GPU detected: {gpu_info['gpu_name']}")
    if gpu_info['is_h100']:
        print("  H100 detected - using max-autotune mode for best performance")
    elif gpu_info['is_a100']:
        print("  A100 detected - using default compile mode")

    # Load model
    print("Loading 7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/init",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode for better torch.compile compatibility
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Compile model for 10-20% speedup (PyTorch 2.x)
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    print(f"  Model: {model.num_parameters() / 1e9:.2f}B parameters")

    # Load data
    train_dataset, val_dataset = load_production_data()

    # Training arguments - OPTIMIZED for 85-95% GPU
    training_args = TrainingArguments(
        output_dir="checkpoints/production_pretrain",
        run_name="production-7b-pretrain",

        # Training schedule
        max_steps=max_steps,
        per_device_train_batch_size=8,      # Increased for better GPU utilization
        gradient_accumulation_steps=4,      # Effective batch = 32

        # Learning rate schedule
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5000,  # Increased for better stability (5% of training)

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
        eval_steps=1000,  # Reduced frequency to minimize overhead
        report_to=["tensorboard"],

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,

        # Data loading
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,  # Keep workers alive between batches

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # PyTorch 2.x compilation (also enabled via torch.compile above)
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode=gpu_info['compile_mode'],  # max-autotune for H100, default for A100
    )

    print(f"\nProduction Configuration:")
    print(f"  Sequence length: 512 tokens")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Optimizer: {training_args.optim}")

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
    print(f"\n🚀 Starting production training ({max_steps:,} steps)...")
    print(f"   Expected GPU utilization: 95-100%")
    print(f"   Optimizations: torch.compile, Flash Attention 2, persistent workers")
    print(f"   Estimated time: ~{max_steps * 1.8 / 3600:.1f} hours\n")

    trainer.train()

    # Save final model
    trainer.save_model("checkpoints/production_pretrain_final")
    print(f"\n✓ Training complete!")
    print(f"  Checkpoint: checkpoints/production_pretrain_final/")

if __name__ == "__main__":
    production_training(max_steps=100000)
