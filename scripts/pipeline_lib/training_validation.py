"""Training prerequisite validation utilities."""
from __future__ import annotations

import os


def check_tokenizer_exists(tokenizer_path: str = "configs/tokenizer") -> bool:
    """Check if tokenizer exists and provide helpful message if not.

    Args:
        tokenizer_path: Path to tokenizer directory.

    Returns:
        True if tokenizer exists, False otherwise.
    """
    required_files = ["tokenizer_config.json", "tokenizer.json"]
    alt_files = ["vocab.json", "merges.txt"]  # For some tokenizer types

    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("")
        print("To create the tokenizer, run:")
        print("  python scripts/demo_tokenize.py")
        print("")
        print("Or for production, run the data preparation pipeline:")
        print("  python scripts/03_tokenize_and_pack.py")
        return False

    # Check for tokenizer files
    has_required = any(
        os.path.exists(os.path.join(tokenizer_path, f))
        for f in required_files + alt_files
    )

    if not has_required:
        print(f"Error: Tokenizer directory exists but appears incomplete: {tokenizer_path}")
        print(f"Missing expected files: {required_files}")
        return False

    return True


def check_checkpoint_exists(checkpoint_path: str, checkpoint_type: str = "model") -> bool:
    """Check if a checkpoint/model exists at the given path.

    Args:
        checkpoint_path: Path to checkpoint directory.
        checkpoint_type: Type of checkpoint ('model', 'tokenizer', 'data').

    Returns:
        True if checkpoint exists and is valid, False otherwise.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_type.title()} not found at {checkpoint_path}")

        # Provide helpful suggestions
        if checkpoint_type == "model":
            if "init" in checkpoint_path:
                print("\nTo initialize the model, run:")
                print("  python scripts/04_init_model.py")
            elif "pretrain" in checkpoint_path:
                print("\nTo create this checkpoint, run pretraining:")
                print("  python scripts/05_pretrain.py")
            elif "sft" in checkpoint_path:
                print("\nTo create this checkpoint, run SFT:")
                print("  python scripts/07_sft.py")
            elif "dpo" in checkpoint_path:
                print("\nTo create this checkpoint, run DPO:")
                print("  python scripts/09_dpo.py")
        return False

    # Check for model files
    model_files = ["config.json", "model.safetensors", "pytorch_model.bin"]
    has_model = any(
        os.path.exists(os.path.join(checkpoint_path, f))
        for f in model_files
    )

    if not has_model:
        print(f"Error: Directory exists but no model files found: {checkpoint_path}")
        print(f"Expected one of: {model_files}")
        return False

    return True


def validate_training_prerequisites(
    model_path: str | None = None,
    tokenizer_path: str = "configs/tokenizer",
    data_path: str | None = None
) -> bool:
    """Validate all prerequisites before starting training.

    Args:
        model_path: Path to model checkpoint (optional).
        tokenizer_path: Path to tokenizer.
        data_path: Path to training data (optional).

    Returns:
        True if all prerequisites are met, False otherwise.
    """
    all_valid = True

    if not check_tokenizer_exists(tokenizer_path):
        all_valid = False

    if model_path and not check_checkpoint_exists(model_path, "model"):
        all_valid = False

    if data_path:
        if not os.path.exists(data_path):
            print(f"Error: Training data not found at {data_path}")
            print("\nTo prepare training data, run:")
            print("  python scripts/01_download_data.py")
            print("  python scripts/02_clean_deduplicate_optimized.py")
            print("  python scripts/03_tokenize_and_pack.py")
            all_valid = False

    return all_valid
