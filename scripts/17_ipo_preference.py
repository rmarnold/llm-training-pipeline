"""IPO (Identity Preference Optimization) training on ranked solution pairs.

Trains the core_agent adapter using preference pairs ranked by execution
quality (cargo check → cargo test → cargo clippy → diff size).

IPO is preferred over DPO for this use case because:
- More stable with noisy preference labels from automated ranking
- Does not require a reference model (saves VRAM)
- Less prone to reward hacking

Usage:
    python scripts/17_ipo_preference.py
    python scripts/17_ipo_preference.py --config configs/ipo.yaml
    python scripts/17_ipo_preference.py --max_steps 1000 --learning_rate 1e-7

Requires: pip install -e ".[gpt_oss]"
"""
import os

if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml

from pipeline_lib.unsloth_utils import load_unsloth_model, apply_lora_config, save_adapter, print_trainable_params


def train_ipo(config_path: str = "configs/ipo.yaml", cli_overrides: dict | None = None) -> None:
    """Train with IPO on preference pairs.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Dict of CLI overrides.
    """
    if cli_overrides is None:
        cli_overrides = {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"IPO Preference Training: {config['run_name']}")
    print(f"{'='*60}")

    # Load model from core_agent checkpoint
    checkpoint = cli_overrides.get("checkpoint", config["model"]["checkpoint"])
    max_seq_length = config["model"].get("max_seq_length", 16384)

    print(f"\nLoading model from: {checkpoint}")
    model, tokenizer = load_unsloth_model(
        model_name=checkpoint,
        max_seq_length=max_seq_length,
        load_in_4bit=config["model"].get("load_in_4bit", True),
        tiled_mlp=False,
        offload_embedding=False,
    )

    # Apply LoRA if the checkpoint doesn't already have adapters
    lora_config = config.get("lora")
    if lora_config:
        print("\nApplying LoRA configuration...")
        model = apply_lora_config(model, lora_config)
        print_trainable_params(model)

    # Load preference dataset
    from datasets import load_from_disk

    train_data_path = cli_overrides.get("train_data_path", config["data"]["train_data"])
    val_data_path = cli_overrides.get("val_data_path", config["data"].get("val_data"))

    print(f"\nLoading preference data: {train_data_path}")
    train_dataset = load_from_disk(train_data_path)
    print(f"  Training pairs: {len(train_dataset):,}")

    # Rename pref_* columns to DPOTrainer's expected format
    col_map = {"pref_prompt": "prompt", "pref_chosen": "chosen", "pref_rejected": "rejected"}
    renames = {k: v for k, v in col_map.items() if k in train_dataset.column_names}
    if renames:
        train_dataset = train_dataset.rename_columns(renames)
        print(f"  Renamed columns: {renames}")

    eval_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        eval_dataset = load_from_disk(val_data_path)
        eval_renames = {k: v for k, v in col_map.items() if k in eval_dataset.column_names}
        if eval_renames:
            eval_dataset = eval_dataset.rename_columns(eval_renames)
        print(f"  Evaluation pairs: {len(eval_dataset):,}")

    # Training arguments
    from trl import DPOConfig, DPOTrainer

    output_dir = cli_overrides.get("output_dir", config["checkpointing"]["output_dir"])

    training_args = DPOConfig(
        output_dir=output_dir,
        run_name=config["run_name"],
        num_train_epochs=cli_overrides.get("num_train_epochs", config["training"].get("num_train_epochs", 1)),
        max_steps=cli_overrides.get("max_steps", config["training"].get("max_steps", -1)),
        learning_rate=cli_overrides.get("learning_rate", config["training"]["learning_rate"]),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=cli_overrides.get("warmup_ratio", config["training"].get("warmup_ratio", 0.1)),
        per_device_train_batch_size=cli_overrides.get(
            "per_device_train_batch_size",
            config["training"].get("per_device_train_batch_size", 1),
        ),
        gradient_accumulation_steps=cli_overrides.get(
            "gradient_accumulation_steps",
            config["training"].get("gradient_accumulation_steps", 16),
        ),
        bf16=config["training"].get("bf16", True),
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        weight_decay=config["training"].get("weight_decay", 0.0),
        optim=config["training"].get("optim", "adamw_8bit"),
        # IPO-specific
        loss_type=config["training"].get("loss_type", "ipo"),
        beta=cli_overrides.get("beta", config["training"].get("beta", 0.1)),
        max_length=config["training"].get("max_length", 16384),
        max_prompt_length=config["training"].get("max_prompt_length", 8192),
        # Logging
        logging_steps=cli_overrides.get("logging_steps", config["logging"].get("logging_steps", 5)),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cli_overrides.get("eval_steps", config["logging"].get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=cli_overrides.get("save_steps", config["logging"].get("save_steps", 200)),
        save_total_limit=config["logging"].get("save_total_limit", 5),
        report_to=config["logging"].get("report_to", ["tensorboard"]),
        seed=config["training"].get("seed", 42),
    )

    # DPOTrainer handles IPO via loss_type="ipo"
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting IPO training...")
    print(f"  Output: {output_dir}")
    print(f"  Loss type: {training_args.loss_type}")
    print(f"  Beta: {training_args.beta}")
    print(f"  LR: {training_args.learning_rate}")

    resume = cli_overrides.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume)

    # Save
    final_dir = os.path.join(output_dir, "final")
    save_adapter(model, final_dir, tokenizer)

    print(f"\nIPO training complete!")
    print(f"  Adapter saved to: {final_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IPO preference training on GPT-OSS 20B")
    parser.add_argument("--config", type=str, default="configs/ipo.yaml")
    parser.add_argument("--checkpoint", type=str)
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
    parser.add_argument("--beta", type=float, help="IPO/DPO beta parameter")
    parser.add_argument("--resume_from_checkpoint", type=str)
    args = parser.parse_args()

    cli_overrides = {}
    for key in ["checkpoint", "max_steps", "num_train_epochs", "learning_rate", "warmup_ratio",
                 "per_device_train_batch_size", "gradient_accumulation_steps",
                 "save_steps", "eval_steps", "logging_steps", "output_dir",
                 "train_data_path", "val_data_path", "beta", "resume_from_checkpoint"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    train_ipo(config_path=args.config, cli_overrides=cli_overrides)
