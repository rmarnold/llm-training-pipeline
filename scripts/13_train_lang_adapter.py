"""Train per-language QLoRA adapters on GPT-OSS 20B.

Trains language-specific adapters (lang_rust, lang_python, etc.) using
Unsloth's optimized QLoRA for MoE models. The adapter injects domain
knowledge without disturbing the base model's general capabilities.

Usage:
    # Train Rust adapter (default)
    python scripts/13_train_lang_adapter.py

    # Train with custom config
    python scripts/13_train_lang_adapter.py --config configs/lang_rust.yaml

    # Override training params
    python scripts/13_train_lang_adapter.py --max_steps 5000 --learning_rate 1e-5

    # Resume from checkpoint
    python scripts/13_train_lang_adapter.py --resume_from_checkpoint checkpoints/lang_rust/checkpoint-1000

Requires: pip install -e ".[gpt_oss]"
"""
import os

# Default to offline wandb
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml

from gpu_utils import setup_torch_backends
from pipeline_lib.unsloth_utils import load_unsloth_model, apply_lora_config, save_adapter, print_trainable_params


def train_lang_adapter(config_path="configs/lang_rust.yaml", cli_overrides=None):
    """Train a language-specific QLoRA adapter on GPT-OSS 20B.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Dict of CLI overrides.
    """
    cli_overrides = cli_overrides or {}

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Training Language Adapter: {config['run_name']}")
    print(f"{'='*60}")

    # Setup backends
    setup_torch_backends()

    # Load model with Unsloth
    print(f"\nLoading model: {config['model']['base_model']}")
    model, tokenizer = load_unsloth_model(
        model_name=config["model"]["base_model"],
        max_seq_length=config["model"].get("max_seq_length", 8192),
        load_in_4bit=config["model"].get("load_in_4bit", True),
        dtype=None,  # Auto-detect
    )

    # Apply LoRA
    print("\nApplying LoRA configuration...")
    model = apply_lora_config(model, config["lora"])
    print_trainable_params(model)

    # Load dataset
    from datasets import load_from_disk

    train_data_path = cli_overrides.get("train_data_path", config["data"]["train_data"])
    val_data_path = cli_overrides.get("val_data_path", config["data"].get("val_data"))

    print(f"\nLoading training data: {train_data_path}")
    train_dataset = load_from_disk(train_data_path)
    print(f"  Training examples: {len(train_dataset):,}")

    eval_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        eval_dataset = load_from_disk(val_data_path)
        print(f"  Evaluation examples: {len(eval_dataset):,}")

    # Build training arguments
    from trl import SFTConfig, SFTTrainer

    output_dir = cli_overrides.get("output_dir", config["checkpointing"]["output_dir"])

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=config["run_name"],
        num_train_epochs=cli_overrides.get("num_train_epochs", config["training"].get("num_train_epochs", 1)),
        max_steps=cli_overrides.get("max_steps", config["training"].get("max_steps", -1)),
        learning_rate=cli_overrides.get("learning_rate", config["training"]["learning_rate"]),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=cli_overrides.get("warmup_ratio", config["training"].get("warmup_ratio", 0.05)),
        per_device_train_batch_size=cli_overrides.get(
            "per_device_train_batch_size",
            config["training"].get("per_device_train_batch_size", 1),
        ),
        gradient_accumulation_steps=cli_overrides.get(
            "gradient_accumulation_steps",
            config["training"].get("gradient_accumulation_steps", 8),
        ),
        bf16=config["training"].get("bf16", True),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        weight_decay=config["training"].get("weight_decay", 0.01),
        optim=config["training"].get("optim", "adamw_8bit"),
        max_seq_length=config["data"].get("max_seq_length", 8192),
        packing=False,  # Don't pack code sequences
        logging_steps=cli_overrides.get("logging_steps", config["logging"].get("logging_steps", 10)),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cli_overrides.get("eval_steps", config["logging"].get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=cli_overrides.get("save_steps", config["logging"].get("save_steps", 500)),
        save_total_limit=config["logging"].get("save_total_limit", 3),
        report_to=config["logging"].get("report_to", ["tensorboard"]),
        seed=config["training"].get("seed", 42),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Register Drive checkpoint callback for Colab session recovery
    if cli_overrides.get("drive_checkpoint_backup", False):
        from pipeline_lib.checkpoint_callback import make_drive_checkpoint_callback
        cb = make_drive_checkpoint_callback(output_dir)
        if cb:
            trainer.add_callback(cb)

    # Train
    print(f"\nStarting training...")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Max steps: {training_args.max_steps}")
    print(f"  LR: {training_args.learning_rate}")
    print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    resume = cli_overrides.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume)

    # Save adapter
    final_dir = os.path.join(output_dir, "final")
    save_adapter(model, final_dir, tokenizer)

    print(f"\nTraining complete!")
    print(f"  Adapter saved to: {final_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train language adapter on GPT-OSS 20B")
    parser.add_argument("--config", type=str, default="configs/lang_rust.yaml")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--drive_checkpoint_backup", action="store_true",
                        help="Back up each checkpoint to Drive (Colab recovery)")
    args = parser.parse_args()

    cli_overrides = {}
    for key in ["max_steps", "num_train_epochs", "learning_rate", "warmup_ratio",
                 "per_device_train_batch_size", "gradient_accumulation_steps",
                 "save_steps", "eval_steps", "logging_steps", "output_dir",
                 "train_data_path", "val_data_path", "resume_from_checkpoint",
                 "drive_checkpoint_backup"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    train_lang_adapter(config_path=args.config, cli_overrides=cli_overrides)
