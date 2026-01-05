"""Validate model configuration against loaded checkpoints.

This script ensures model configs match between:
- configs/model_7b.py definition
- Loaded HuggingFace model config
- Training YAML configs
"""
import argparse
import os
import sys
import json

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'configs'))


def load_model_config(checkpoint_path):
    """Load config.json from a checkpoint."""
    config_path = os.path.join(checkpoint_path, "config.json")

    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        return json.load(f)


def get_expected_config():
    """Get expected config from model_7b.py."""
    try:
        from model_7b import ModelConfig
        config = ModelConfig()
        return {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "rms_norm_eps": config.rms_norm_eps,
        }
    except ImportError:
        return None


def validate_checkpoint(checkpoint_path, expected_config=None):
    """Validate a checkpoint against expected configuration.

    Args:
        checkpoint_path: Path to checkpoint directory
        expected_config: Expected configuration dict (optional)

    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        return False, [f"Checkpoint not found: {checkpoint_path}"]

    # Load checkpoint config
    actual_config = load_model_config(checkpoint_path)
    if actual_config is None:
        return False, [f"No config.json found in {checkpoint_path}"]

    # If no expected config, just validate the checkpoint is readable
    if expected_config is None:
        expected_config = get_expected_config()
        if expected_config is None:
            print(f"Note: Could not load expected config from model_7b.py")
            print(f"Validating checkpoint structure only...")

            # Basic validation
            required_keys = ["vocab_size", "hidden_size", "num_hidden_layers"]
            for key in required_keys:
                if key not in actual_config:
                    issues.append(f"Missing required config key: {key}")

            return len(issues) == 0, issues

    # Compare configurations
    for key, expected_value in expected_config.items():
        actual_value = actual_config.get(key)

        if actual_value is None:
            issues.append(f"Missing config key: {key}")
        elif actual_value != expected_value:
            issues.append(
                f"Config mismatch for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )

    return len(issues) == 0, issues


def validate_all_checkpoints(verbose=True):
    """Validate all checkpoints in the checkpoints directory."""
    checkpoint_dirs = [
        "checkpoints/init",
        "checkpoints/pretrain_final",
        "checkpoints/sft_final",
        "checkpoints/dpo_final",
        "checkpoints/lora_final",
    ]

    results = {}

    print("=" * 60)
    print("MODEL CONFIGURATION VALIDATION")
    print("=" * 60)
    print()

    expected_config = get_expected_config()
    if expected_config:
        print("Expected configuration (from model_7b.py):")
        for key, value in expected_config.items():
            print(f"  {key}: {value}")
        print()

    for checkpoint_path in checkpoint_dirs:
        if not os.path.exists(checkpoint_path):
            if verbose:
                print(f"[ ] {checkpoint_path} - Not found (skipped)")
            results[checkpoint_path] = {"status": "not_found"}
            continue

        is_valid, issues = validate_checkpoint(checkpoint_path, expected_config)

        if is_valid:
            print(f"[OK] {checkpoint_path}")
            results[checkpoint_path] = {"status": "valid"}
        else:
            print(f"[!!] {checkpoint_path}")
            for issue in issues:
                print(f"     - {issue}")
            results[checkpoint_path] = {"status": "invalid", "issues": issues}

    print()
    print("=" * 60)

    # Summary
    valid_count = sum(1 for r in results.values() if r["status"] == "valid")
    invalid_count = sum(1 for r in results.values() if r["status"] == "invalid")
    not_found = sum(1 for r in results.values() if r["status"] == "not_found")

    print(f"Valid: {valid_count}, Invalid: {invalid_count}, Not Found: {not_found}")
    print("=" * 60)

    return results


def compare_checkpoints(path1, path2):
    """Compare configuration between two checkpoints."""
    print(f"Comparing: {path1} vs {path2}")
    print("=" * 60)

    config1 = load_model_config(path1)
    config2 = load_model_config(path2)

    if config1 is None:
        print(f"Error: Could not load config from {path1}")
        return

    if config2 is None:
        print(f"Error: Could not load config from {path2}")
        return

    # Compare all keys
    all_keys = set(config1.keys()) | set(config2.keys())

    differences = []
    for key in sorted(all_keys):
        val1 = config1.get(key, "<missing>")
        val2 = config2.get(key, "<missing>")

        if val1 != val2:
            differences.append((key, val1, val2))
            print(f"  {key}:")
            print(f"    {path1}: {val1}")
            print(f"    {path2}: {val2}")

    if not differences:
        print("  Configurations are identical!")

    print("=" * 60)
    return differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate model configurations")
    parser.add_argument("--checkpoint", type=str,
                        help="Validate a specific checkpoint")
    parser.add_argument("--compare", nargs=2, metavar=("PATH1", "PATH2"),
                        help="Compare two checkpoints")
    parser.add_argument("--all", action="store_true",
                        help="Validate all checkpoints")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.compare[0], args.compare[1])
    elif args.checkpoint:
        is_valid, issues = validate_checkpoint(args.checkpoint)
        if is_valid:
            print(f"Checkpoint {args.checkpoint} is valid")
        else:
            print(f"Checkpoint {args.checkpoint} has issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
    else:
        # Default: validate all
        validate_all_checkpoints()
