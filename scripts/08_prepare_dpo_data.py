"""Prepare preference pairs for DPO training.

This script converts HH-RLHF or similar preference datasets into the format
required for DPO training:
    - prompt: The conversation prompt
    - chosen: The preferred response
    - rejected: The less preferred response

Usage:
    python scripts/08_prepare_dpo_data.py
    python scripts/08_prepare_dpo_data.py --validate-only
    python scripts/08_prepare_dpo_data.py --skip-safety-filter
"""
import os
import sys
from datasets import load_dataset, Dataset


def validate_dpo_example(example, idx=0):
    """Validate a single DPO example has required fields.

    Args:
        example: Dataset example to validate
        idx: Index for error reporting

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["prompt", "chosen", "rejected"]

    for field in required_fields:
        if field not in example:
            return False, f"Example {idx}: Missing required field '{field}'"
        if not example[field] or not isinstance(example[field], str):
            return False, f"Example {idx}: Field '{field}' must be a non-empty string"
        if len(example[field].strip()) == 0:
            return False, f"Example {idx}: Field '{field}' is empty after stripping"

    # Check that chosen and rejected are different
    if example["chosen"].strip() == example["rejected"].strip():
        return False, f"Example {idx}: 'chosen' and 'rejected' are identical"

    return True, None


def validate_dpo_dataset(dataset, sample_size=100):
    """Validate DPO dataset format.

    Args:
        dataset: HuggingFace dataset to validate
        sample_size: Number of examples to validate

    Returns:
        Tuple of (is_valid, error_list)
    """
    errors = []

    # Check required columns
    required_columns = ["prompt", "chosen", "rejected"]
    missing_columns = [c for c in required_columns if c not in dataset.column_names]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors

    # Validate sample
    sample_indices = range(min(sample_size, len(dataset)))
    for idx in sample_indices:
        is_valid, error = validate_dpo_example(dataset[idx], idx)
        if not is_valid:
            errors.append(error)

    return len(errors) == 0, errors


def format_hh_rlhf_preference(example):
    """Format HH-RLHF example to DPO format.

    HH-RLHF format:
        - chosen: Full conversation including "Human:" and "Assistant:" turns
        - rejected: Full conversation with a different assistant response

    Args:
        example: HH-RLHF dataset example

    Returns:
        Formatted example with prompt, chosen, rejected
    """
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    # Try to extract prompt and responses
    try:
        # Find the last "Assistant:" in chosen
        if "Assistant:" not in chosen:
            return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}

        # Split on last "Assistant:" occurrence
        parts = chosen.rsplit("Assistant:", 1)
        if len(parts) != 2:
            return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}

        prompt = parts[0] + "Assistant:"
        chosen_response = parts[1].strip()

        # Get rejected response (should have same prompt structure)
        if "Assistant:" not in rejected:
            return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}

        rejected_parts = rejected.rsplit("Assistant:", 1)
        if len(rejected_parts) != 2:
            return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}

        rejected_response = rejected_parts[1].strip()

        # Validate responses are non-empty
        if not chosen_response or not rejected_response:
            return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}

        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "_valid": True
        }

    except Exception as e:
        print(f"  Warning: Failed to format example: {e}")
        return {"prompt": "", "chosen": "", "rejected": "", "_valid": False}


def prepare_dpo_dataset(skip_safety_filter=False, validate_only=False, output_dir="data/dpo"):
    """Prepare preference pairs for DPO training.

    Args:
        skip_safety_filter: Skip toxicity-based safety filtering
        validate_only: Only validate existing data, don't regenerate
        output_dir: Output directory for processed data
    """
    print("=" * 60)
    print("DPO DATA PREPARATION")
    print("=" * 60)

    # Validate existing data if requested
    if validate_only:
        return validate_existing_data(output_dir)

    # Load HH-RLHF
    print("\n[1/4] Loading HH-RLHF dataset...")
    try:
        hh_rlhf = load_dataset("Anthropic/hh-rlhf", split="train")
        print(f"  Loaded {len(hh_rlhf)} examples")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        sys.exit(1)

    # Format for DPO
    print("\n[2/4] Formatting for DPO...")
    dpo_dataset = hh_rlhf.map(format_hh_rlhf_preference)

    # Filter invalid examples
    original_count = len(dpo_dataset)
    dpo_dataset = dpo_dataset.filter(lambda x: x.get("_valid", False))
    valid_count = len(dpo_dataset)
    print(f"  Valid examples: {valid_count}/{original_count} ({100*valid_count/original_count:.1f}%)")

    if valid_count == 0:
        print("  Error: No valid examples after formatting!")
        sys.exit(1)

    # Remove temporary _valid column
    dpo_dataset = dpo_dataset.remove_columns(["_valid"])

    # Apply safety filter
    if not skip_safety_filter:
        print("\n[3/4] Applying safety filter...")
        try:
            from detoxify import Detoxify
            toxicity_model = Detoxify('original')

            def is_safe(example):
                try:
                    chosen_toxic = toxicity_model.predict(example["chosen"])
                    rejected_toxic = toxicity_model.predict(example["rejected"])
                    # Keep if chosen is less toxic than rejected
                    return chosen_toxic["toxicity"] < rejected_toxic["toxicity"]
                except Exception:
                    return True  # Keep on error

            before_filter = len(dpo_dataset)
            dpo_dataset = dpo_dataset.filter(is_safe)
            after_filter = len(dpo_dataset)
            print(f"  Safety filter: {after_filter}/{before_filter} examples kept")

        except ImportError:
            print("  Warning: detoxify not installed, skipping safety filter")
            print("  Install with: pip install detoxify")
    else:
        print("\n[3/4] Skipping safety filter (--skip-safety-filter)")

    # Validate final dataset
    print("\n[4/4] Validating final dataset...")
    is_valid, errors = validate_dpo_dataset(dpo_dataset)
    if not is_valid:
        print(f"  Validation errors ({len(errors)}):")
        for err in errors[:10]:  # Show first 10 errors
            print(f"    - {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # Split and save
    print(f"\n[5/4] Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    split = dpo_dataset.train_test_split(test_size=0.05, seed=42)
    split["train"].save_to_disk(os.path.join(output_dir, "train"))
    split["test"].save_to_disk(os.path.join(output_dir, "val"))

    print(f"\n{'=' * 60}")
    print("DPO DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Training examples: {len(split['train'])}")
    print(f"  Validation examples: {len(split['test'])}")
    print(f"  Output directory: {output_dir}")

    return True


def validate_existing_data(data_dir="data/dpo"):
    """Validate existing DPO data.

    Args:
        data_dir: Directory containing train/ and val/ subdirs
    """
    print("=" * 60)
    print("VALIDATING EXISTING DPO DATA")
    print("=" * 60)

    from datasets import load_from_disk

    for split_name in ["train", "val"]:
        split_path = os.path.join(data_dir, split_name)
        print(f"\n[{split_name}]")

        if not os.path.exists(split_path):
            print(f"  Error: Not found at {split_path}")
            continue

        try:
            dataset = load_from_disk(split_path)
            print(f"  Loaded {len(dataset)} examples")
            print(f"  Columns: {dataset.column_names}")

            is_valid, errors = validate_dpo_dataset(dataset, sample_size=100)
            if is_valid:
                print(f"  Validation: PASSED")
            else:
                print(f"  Validation: FAILED ({len(errors)} errors)")
                for err in errors[:5]:
                    print(f"    - {err}")

        except Exception as e:
            print(f"  Error loading: {e}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare DPO training data")
    parser.add_argument("--skip-safety-filter", action="store_true",
                        help="Skip toxicity-based safety filtering")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing data, don't regenerate")
    parser.add_argument("--output-dir", type=str, default="data/dpo",
                        help="Output directory for processed data")
    args = parser.parse_args()

    prepare_dpo_dataset(
        skip_safety_filter=args.skip_safety_filter,
        validate_only=args.validate_only,
        output_dir=args.output_dir
    )
