import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk, concatenate_datasets
import yaml

def train_lora():
    with open("configs/lora_finetune.yaml") as f:
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

    # Training
    training_args = TrainingArguments(
        output_dir="checkpoints/lora",
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        logging_steps=10,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save LoRA adapters
    model.save_pretrained("checkpoints/lora_final")

    # Merge and save full model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("checkpoints/lora_merged")

    print("✓ LoRA fine-tuning complete!")
    print(f"  LoRA adapters: checkpoints/lora_final")
    print(f"  Merged model: checkpoints/lora_merged")

if __name__ == "__main__":
    train_lora()
