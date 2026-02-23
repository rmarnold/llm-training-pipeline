"""Train core_agent adapter for Rust coding agent behavior.

Trains the core_agent LoRA adapter on agent trajectory data (multi-turn
tool-call conversations in Harmony format). Optionally merges the lang_rust
adapter into the base model first.

Usage:
    # Train core agent (assumes lang_rust already trained)
    python scripts/14_train_core_agent.py

    # With custom config
    python scripts/14_train_core_agent.py --config configs/core_agent.yaml

    # Skip lang_rust merge
    python scripts/14_train_core_agent.py --no-merge-lang-adapter

Requires: pip install -e ".[gpt_oss]"
"""
import os

if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "offline"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml

from gpu_utils import setup_torch_backends
from pipeline_lib.unsloth_utils import load_unsloth_model, apply_lora_config, save_adapter, print_trainable_params


def train_core_agent(config_path="configs/core_agent.yaml", cli_overrides=None):
    """Train core_agent adapter for Rust coding agent behavior.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Dict of CLI overrides.
    """
    cli_overrides = cli_overrides or {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Training Core Agent: {config['run_name']}")
    print(f"{'='*60}")

    setup_torch_backends()

    # Determine base model (with or without lang_rust merged)
    # CLI --base_model overrides config and skips lang_rust merge logic
    if cli_overrides.get("base_model"):
        base_model = cli_overrides["base_model"]
        print(f"\nUsing base model from CLI: {base_model}")
    else:
        base_model = config["model"]["base_model"]
        merge_lang = config["model"].get("merge_lang_adapter", True)
        lang_adapter_path = config["model"].get("lang_adapter")

        if merge_lang and lang_adapter_path and not cli_overrides.get("no_merge_lang_adapter"):
            # Check if merged model already exists
            merged_path = f"{lang_adapter_path}_merged"
            if os.path.exists(merged_path):
                print(f"\nUsing pre-merged model: {merged_path}")
                base_model = merged_path
            else:
                print(f"\nMerging lang_rust adapter from {lang_adapter_path}...")
                print("  (Run scripts/19_merge_adapter.py first, or use --no-merge-lang-adapter)")
                print(f"  Falling back to base model: {base_model}")
        else:
            print(f"\nSkipping lang adapter merge, using base: {base_model}")

    # Load model
    print(f"\nLoading model: {base_model}")
    max_seq_length = cli_overrides.get(
        "max_seq_length", config["model"].get("max_seq_length", 16384),
    )
    model, tokenizer = load_unsloth_model(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=config["model"].get("load_in_4bit", True),
    )

    # Apply LoRA (higher rank for agent behavior)
    print(f"\nApplying LoRA (rank={config['lora']['r']})...")
    model = apply_lora_config(model, config["lora"])
    print_trainable_params(model)

    # Load trajectory dataset
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

    # Profile token lengths to validate seq_len setting
    from pipeline_lib.data_profiler import profile_seq_lengths

    profile = profile_seq_lengths(
        dataset_path=train_data_path,
        tokenizer=tokenizer,
        sample_size=5000,
    )
    configured_seq_len = cli_overrides.get(
        "max_seq_length", config["data"].get("max_seq_length", 16384),
    )
    if profile.get("recommended_seq_len") and profile["recommended_seq_len"] < configured_seq_len:
        print(f"  NOTE: configured seq_len ({configured_seq_len}) > recommended "
              f"({profile['recommended_seq_len']}). Extra headroom is fine with packing.")
    elif profile.get("p99", 0) > configured_seq_len:
        print(f"  WARNING: {configured_seq_len} will truncate >1% of examples "
              f"(P99={profile['p99']}). Consider increasing seq_len.")

    # Training arguments
    from trl import SFTConfig, SFTTrainer

    output_dir = cli_overrides.get("output_dir", config["checkpointing"]["output_dir"])

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=config["run_name"],
        num_train_epochs=cli_overrides.get("num_train_epochs", config["training"].get("num_train_epochs", 2)),
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
            config["training"].get("gradient_accumulation_steps", 4),
        ),
        bf16=config["training"].get("bf16", True),
        max_grad_norm=config["training"].get("max_grad_norm", 0.5),
        weight_decay=config["training"].get("weight_decay", 0.01),
        optim=config["training"].get("optim", "adamw_8bit"),
        max_seq_length=cli_overrides.get(
            "max_seq_length", config["data"].get("max_seq_length", 16384),
        ),
        packing=cli_overrides.get("packing", config["training"].get("packing", False)),
        logging_steps=cli_overrides.get("logging_steps", config["logging"].get("logging_steps", 5)),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cli_overrides.get("eval_steps", config["logging"].get("eval_steps", 250)),
        save_strategy="steps",
        save_steps=cli_overrides.get("save_steps", config["logging"].get("save_steps", 250)),
        save_total_limit=config["logging"].get("save_total_limit", 3),
        report_to=config["logging"].get("report_to", ["tensorboard"]),
        seed=config["training"].get("seed", 42),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

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

    print(f"\nStarting training...")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  LR: {training_args.learning_rate}")
    print(f"  Max seq length: {training_args.max_seq_length}")
    print(f"  Packing: {training_args.packing}")

    resume = cli_overrides.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(output_dir, "final")
    save_adapter(model, final_dir, tokenizer)

    print(f"\nTraining complete!")
    print(f"  Core agent adapter saved to: {final_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train core agent adapter on GPT-OSS 20B")
    parser.add_argument("--config", type=str, default="configs/core_agent.yaml")
    parser.add_argument("--no-merge-lang-adapter", action="store_true",
                        help="Skip merging lang_rust adapter into base")
    parser.add_argument("--base_model", type=str,
                        help="Override base model path (skips lang_rust merge logic)")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--max_seq_length", type=int,
                        help="Override max sequence length for model and SFTConfig")
    parser.add_argument("--packing", action="store_true",
                        help="Enable sequence packing (Unsloth padding-free batching)")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--drive_checkpoint_backup", action="store_true",
                        help="Back up each checkpoint to Drive (Colab recovery)")
    args = parser.parse_args()

    cli_overrides = {}
    for key in ["base_model", "max_steps", "num_train_epochs", "learning_rate",
                 "warmup_ratio", "per_device_train_batch_size",
                 "gradient_accumulation_steps", "save_steps", "eval_steps",
                 "logging_steps", "max_seq_length", "packing", "output_dir",
                 "train_data_path", "val_data_path",
                 "resume_from_checkpoint", "drive_checkpoint_backup"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    if args.no_merge_lang_adapter:
        cli_overrides["no_merge_lang_adapter"] = True

    train_core_agent(config_path=args.config, cli_overrides=cli_overrides)
