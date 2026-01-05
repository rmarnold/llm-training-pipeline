"""LoRA fine-tuning for domain adaptation."""
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, concatenate_datasets
import yaml
import os


def train_lora(config_path="configs/lora_finetune.yaml", cli_overrides=None):
    """Train LoRA adapters for domain-specific fine-tuning.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dict of CLI overrides (max_steps, output_dir, etc.)
    """
    cli_overrides = cli_overrides or {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_checkpoint'],
        torch_dtype=torch.bfloat16
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap model with LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Load mixed dataset
    domain_ds = load_from_disk(config['data']['domain_data'])
    general_ds = load_from_disk(config['data']['general_data'])

    # Mix datasets
    domain_sample = domain_ds.shuffle(seed=42).select(
        range(int(len(domain_ds) * config['data']['domain_weight']))
    )
    general_sample = general_ds.shuffle(seed=42).select(
        range(int(len(general_ds) * config['data']['general_weight']))
    )
    train_dataset = concatenate_datasets([domain_sample, general_sample]).shuffle(seed=42)

    # Training arguments with CLI overrides
    output_dir = cli_overrides.get('output_dir', "checkpoints/lora")
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=cli_overrides.get('max_steps', -1),
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        logging_steps=cli_overrides.get('logging_steps', 10),
        save_steps=cli_overrides.get('save_steps', 500),
        eval_steps=cli_overrides.get('eval_steps', 500),
        save_total_limit=3,
    )

    # Load tokenizer for data collation
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\n🚀 Starting LoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))

    # Save LoRA adapters
    final_output = cli_overrides.get('output_dir', "checkpoints/lora_final")
    model.save_pretrained(final_output)

    # Optionally merge and save full model
    if cli_overrides.get('merge', True):
        print("\nMerging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(f"{final_output}_merged")
        print(f"  Merged model: {final_output}_merged")

    print("\n✓ LoRA fine-tuning complete!")
    print(f"  LoRA adapters: {final_output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for domain adaptation")
    parser.add_argument("--config", type=str, default="configs/lora_finetune.yaml", help="Path to config file")
    parser.add_argument("--max_steps", type=int, help="Override max training steps")
    parser.add_argument("--save_steps", type=int, help="Override checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, help="Override evaluation frequency")
    parser.add_argument("--logging_steps", type=int, help="Override logging frequency")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging LoRA into base model")
    args = parser.parse_args()

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
    if args.no_merge:
        cli_overrides['merge'] = False

    train_lora(config_path=args.config, cli_overrides=cli_overrides)
