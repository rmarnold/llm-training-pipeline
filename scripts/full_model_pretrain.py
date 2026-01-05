"""7B model pretraining with demo data - to verify GPU utilization"""
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

def full_model_pretrain(max_steps=100):
    """Run 7B model pretraining"""
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*60}")
    print(f"7B MODEL PRETRAINING on {device_name} ({max_steps} steps)")
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

    # Load dataset
    train_dataset, val_dataset = load_demo_dataset()

    # Training arguments for A100 - Optimized for high GPU utilization
    training_args = TrainingArguments(
        output_dir="checkpoints/full_pretrain",
        run_name="full-7b-pretrain",

        # Training steps
        max_steps=max_steps,
        per_device_train_batch_size=4,  # Increased for better GPU utilization
        gradient_accumulation_steps=8,  # Effective batch = 32

        # Logging
        logging_steps=5,
        logging_dir="logs/full_pretrain",
        report_to=[],  # Disable W&B

        # Evaluation
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=1,

        # Checkpointing
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        # Optimization
        learning_rate=3e-4,
        weight_decay=0.1,
        max_grad_norm=1.0,

        # Precision
        bf16=True,
        tf32=True,

        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch_fused",

        # Data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    print(f"\nTraining configuration:")
    print(f"  Device: {device} ({device_name})")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Mixed precision: BF16 + TF32")
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
    print(f"\n🚀 Starting 7B model pretraining ({max_steps} steps)...\n")
    trainer.train()

    # Save final model
    print("\nSaving final checkpoint...")
    trainer.save_model("checkpoints/full_pretrain_final")

    print(f"\n{'='*60}")
    print("✓ 7B pretraining complete!")
    print(f"{'='*60}")
    print(f"\nCheckpoints saved to: checkpoints/full_pretrain_final/")

    return trainer

if __name__ == "__main__":
    full_model_pretrain(max_steps=100)
