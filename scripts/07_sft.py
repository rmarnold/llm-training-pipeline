import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import yaml

# Import GPU utilities
from gpu_utils import detect_gpu_type, print_gpu_info, setup_torch_backends

def train_sft(use_fp8=None):
    # Setup torch backends
    setup_torch_backends()

    with open("configs/sft.yaml") as f:
        config = yaml.safe_load(f)

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8:
        print("  Using FP8 precision for SFT")
    else:
        print("  Using BF16 precision for SFT")

    # Load model with Flash Attention
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode
    if config['model'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Compile model for speedup
    compile_mode = gpu_info['compile_mode']
    print(f"Compiling model with torch.compile (mode={compile_mode})...")
    model = torch.compile(model, mode=compile_mode)

    train_dataset = load_from_disk(config['data']['train_data'])
    eval_dataset = load_from_disk(config['data']['val_data'])

    # Use SFTConfig for packing support (up to 6x speedup)
    training_args = SFTConfig(
        output_dir=config['checkpointing']['output_dir'],
        run_name=config['run_name'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        bf16=config['training']['bf16'],
        logging_steps=config['logging']['logging_steps'],
        eval_steps=config['eval']['eval_steps'],
        save_steps=config['logging']['save_steps'],
        eval_strategy=config['eval']['evaluation_strategy'],
        load_best_model_at_end=config['eval']['load_best_model_at_end'],
        metric_for_best_model=config['eval']['metric_for_best_model'],

        # Sequence packing - up to 6x speedup by eliminating padding waste
        packing=True,
        max_seq_length=config['data']['max_seq_length'],

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
    )

    print(f"\nSFT Configuration:")
    print(f"  Packing: ENABLED (up to 6x speedup)")
    print(f"  Max sequence length: {config['data']['max_seq_length']}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Compile mode: {gpu_info['compile_mode']}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("\n🚀 Starting SFT training with sequence packing...")
    trainer.train()
    trainer.save_model("checkpoints/sft_final")

    print("✓ SFT training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SFT training for 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    args = parser.parse_args()

    use_fp8 = None
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    train_sft(use_fp8=use_fp8)
