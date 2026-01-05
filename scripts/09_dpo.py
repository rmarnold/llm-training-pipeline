import torch
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import yaml

# Import GPU utilities
from gpu_utils import detect_gpu_type, print_gpu_info, setup_torch_backends

def train_dpo(use_fp8=None):
    # Setup torch backends
    setup_torch_backends()

    with open("configs/dpo.yaml") as f:
        config = yaml.safe_load(f)

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8:
        print("  Using FP8 precision for DPO")
    else:
        print("  Using BF16 precision for DPO")

    # Load model with Flash Attention
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load reference model (no need to compile - used only for inference)
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing on main model only
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Compile main model for speedup (ref_model doesn't need compilation)
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    train_dataset = load_from_disk("data/dpo/train")
    eval_dataset = load_from_disk("data/dpo/val")

    training_args = DPOConfig(
        output_dir="checkpoints/dpo",
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        beta=config['training']['beta'],
        loss_type=config['training']['loss_type'],
        max_length=config['training']['max_length'],
        max_prompt_length=config['training']['max_prompt_length'],

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,

        # PyTorch 2.x compilation
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode=gpu_info['compile_mode'],

        # Logging
        logging_steps=config['logging']['logging_steps'],
        eval_steps=config['eval']['eval_steps'],
        eval_strategy="steps",
        save_steps=config['logging'].get('save_steps', 200),
        save_total_limit=3,
    )

    print(f"\nDPO Configuration:")
    print(f"  Beta (KL penalty): {config['training']['beta']}")
    print(f"  Loss type: {config['training']['loss_type']}")
    print(f"  Max length: {config['training']['max_length']}")
    print(f"  Compile mode: {gpu_info['compile_mode']}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\n🚀 Starting DPO training...")
    trainer.train()
    trainer.save_model("checkpoints/dpo_final")

    print("✓ DPO training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO training for 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    args = parser.parse_args()

    use_fp8 = None
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    train_dpo(use_fp8=use_fp8)
