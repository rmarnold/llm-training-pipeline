"""7B model training optimized for maximum GPU utilization (85-95%)"""
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
import os

def load_long_sequences():
    """Load 512-token sequences for maximum GPU stress"""
    print("Loading long sequence dataset (512 tokens)...")

    # Load packed sequences
    data = np.load("data/packed/pretrain_long.npy")
    print(f"  Loaded {len(data)} sequences of length {data.shape[1]}")

    # Split into train/val
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create HF datasets
    train_dataset = Dataset.from_dict({"input_ids": train_data.tolist()})
    val_dataset = Dataset.from_dict({"input_ids": val_data.tolist()})

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    return train_dataset, val_dataset

def max_gpu_training(max_steps=100):
    """Run 7B model training with maximum GPU utilization"""
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*60}")
    print(f"MAXIMUM GPU UTILIZATION TRAINING")
    print(f"7B Model on {device_name}")
    print(f"Target: 85-95% GPU Utilization")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading 7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "checkpoints/init",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    print(f"  Model parameters: {model.num_parameters() / 1e9:.2f}B")

    # Load long sequence dataset
    train_dataset, val_dataset = load_long_sequences()

    # Training arguments optimized for MAX GPU utilization
    training_args = TrainingArguments(
        output_dir="checkpoints/max_gpu_pretrain",
        run_name="max-gpu-7b-pretrain",

        # Training steps
        max_steps=max_steps,
        per_device_train_batch_size=4,  # 4 samples x 512 tokens = high compute
        gradient_accumulation_steps=4,  # Effective batch = 16

        # Logging
        logging_steps=5,
        logging_dir="logs/max_gpu_pretrain",
        report_to=[],  # Disable W&B

        # Evaluation
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=2,

        # Checkpointing
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        # Optimization for speed
        learning_rate=3e-4,
        weight_decay=0.1,
        max_grad_norm=1.0,

        # Precision - maximized for A100
        bf16=True,
        tf32=True,

        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Fastest optimizer

        # Data loading - optimized for throughput
        dataloader_num_workers=8,  # Increased workers
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,  # Prefetch more batches
    )

    print(f"\nMAX GPU Configuration:")
    print(f"  Device: {device} ({device_name})")
    print(f"  Sequence length: 512 tokens (4x longer)")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Total tokens/step: {training_args.per_device_train_batch_size * 512}")
    print(f"  Mixed precision: BF16 + TF32")
    print(f"  Optimizer: AdamW Fused")
    print(f"  Max steps: {max_steps}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    print(f"\n🚀 Starting MAX GPU training ({max_steps} steps)...")
    print(f"   Expected GPU utilization: 85-95%")
    print(f"   Expected power draw: 350-395W\n")

    trainer.train()

    # Save final model
    print("\nSaving final checkpoint...")
    trainer.save_model("checkpoints/max_gpu_pretrain_final")

    print(f"\n{'='*60}")
    print("✓ Maximum GPU training complete!")
    print(f"{'='*60}")
    print(f"\nCheckpoints saved to: checkpoints/max_gpu_pretrain_final/")

    return trainer

if __name__ == "__main__":
    max_gpu_training(max_steps=100)
