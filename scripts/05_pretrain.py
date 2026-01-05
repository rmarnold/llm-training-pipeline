import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import yaml
import wandb

class CurriculumCallback:
    def __init__(self, curriculum_config):
        self.schedule = curriculum_config['schedule']
        self.current_idx = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self.current_idx < len(self.schedule):
            stage = self.schedule[self.current_idx]
            if state.global_step >= stage['steps']:
                print(f"📚 Curriculum: Increasing context to {stage['seq_length']}")
                # Update data loader with new sequence length
                self.current_idx += 1

def setup_training():
    # Load config
    with open("configs/pretrain.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize W&B (disabled in non-interactive environments)
    import os
    if os.getenv('WANDB_MODE') != 'disabled':
        wandb.init(
            project="llm-training",
            name=config['run_name'],
            config=config
        )

    # Load model and tokenizer
    attn_impl = "flash_attention_2" if config['model']['use_flash_attention'] else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode for better torch.compile compatibility
    if config['model']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Compile model for 10-20% speedup (PyTorch 2.x)
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['checkpointing']['output_dir'],
        run_name=config['run_name'],

        # Optimization
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        warmup_steps=config['training']['warmup_steps'],

        # Batch sizing
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        # Precision
        bf16=config['training']['bf16'],
        tf32=config['training']['tf32'],

        # Regularization
        max_grad_norm=config['training']['max_grad_norm'],
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],

        # Optimization
        optim=config['training']['optim'],
        fsdp=config['training']['fsdp'],
        fsdp_transformer_layer_cls_to_wrap=config['training']['fsdp_transformer_layer_cls_to_wrap'],

        # Logging
        logging_dir=config['logging']['logging_dir'],
        logging_steps=config['logging']['logging_steps'],
        report_to=config['logging']['report_to'],

        # Evaluation
        evaluation_strategy=config['eval']['evaluation_strategy'],
        eval_steps=config['eval']['eval_steps'],
        per_device_eval_batch_size=config['eval']['per_device_eval_batch_size'],

        # Checkpointing
        save_strategy=config['checkpointing']['save_strategy'],
        save_steps=config['checkpointing']['save_steps'],
        save_total_limit=config['logging']['save_total_limit'],
    )

    # Load dataset
    train_dataset = load_from_disk("data/packed/train")
    eval_dataset = load_from_disk("data/packed/val")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add curriculum callback
    if config['curriculum']['enabled']:
        trainer.add_callback(CurriculumCallback(config['curriculum']))

    return trainer

def main():
    trainer = setup_training()

    # Start training
    print("🚀 Starting pretraining...")
    trainer.train()

    # Save final model
    trainer.save_model("checkpoints/pretrain_final")
    print("✓ Pretraining complete!")

if __name__ == "__main__":
    main()
