"""Minimal pretraining demo - runs on CPU for a few steps"""
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

def load_demo_dataset():
    """Load packed demo data"""
    print("Loading demo dataset...")

    # Load packed sequences
    data = np.load("data/packed/pretrain_demo.npy")
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

def demo_pretrain(max_steps=10):
    """Run minimal pretraining for demo"""
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*60}")
    print(f"DEMO PRETRAINING on {device_name} ({max_steps} steps)")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("checkpoints/demo_init")
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Move model to device
    if device == "cuda":
        print(f"  Moving model to GPU...")
        model = model.to(device)

    print(f"  Model parameters: {model.num_parameters() / 1e6:.1f}M")

    # Load dataset
    train_dataset, val_dataset = load_demo_dataset()

    # Training arguments (GPU-optimized if available)
    use_gpu = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir="checkpoints/demo_pretrain",
        run_name="demo-pretrain",

        # Very short training for demo
        max_steps=max_steps,
        per_device_train_batch_size=4 if use_gpu else 2,
        gradient_accumulation_steps=2,

        # Logging
        logging_steps=2,
        logging_dir="logs/demo_pretrain",
        report_to=[],  # Disable W&B for demo

        # Evaluation
        eval_strategy="steps",
        eval_steps=5,
        per_device_eval_batch_size=4 if use_gpu else 2,

        # Checkpointing
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,

        # Optimization
        learning_rate=1e-4,
        weight_decay=0.0,

        # Use mixed precision on GPU if available
        fp16=use_gpu and not torch.cuda.is_bf16_supported(),
        bf16=use_gpu and torch.cuda.is_bf16_supported(),

        # Use CUDA optimizations if available
        dataloader_num_workers=4 if use_gpu else 0,
        dataloader_pin_memory=use_gpu,
    )

    print(f"\nTraining configuration:")
    print(f"  Device: {device} ({device_name})")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Mixed precision: {'BF16' if training_args.bf16 else 'FP16' if training_args.fp16 else 'FP32'}")
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
    print(f"\n🚀 Starting demo pretraining ({max_steps} steps)...\n")
    trainer.train()

    # Save final model
    print("\nSaving final checkpoint...")
    trainer.save_model("checkpoints/demo_pretrain_final")

    print(f"\n{'='*60}")
    print("✓ Demo pretraining complete!")
    print(f"{'='*60}")
    print(f"\nCheckpoints saved to: checkpoints/demo_pretrain_final/")

    return trainer

if __name__ == "__main__":
    demo_pretrain(max_steps=10)
