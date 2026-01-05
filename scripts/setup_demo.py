#!/usr/bin/env python3
"""Set up everything needed for demo/testing.

This script prepares all prerequisites for running the demo training:
1. Generates synthetic demo data
2. Creates the tokenizer
3. Tokenizes and packs the data
4. Initializes a tiny demo model

After running this script, you can run:
    python scripts/demo_pretrain.py

Usage:
    python scripts/setup_demo.py
    python scripts/setup_demo.py --skip-model  # Skip model initialization
"""
import os
import sys
import argparse


def run_step(description, func, *args, **kwargs):
    """Run a step with nice formatting."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}\n")
    try:
        result = func(*args, **kwargs)
        print(f"\n[OK] {description}")
        return result
    except Exception as e:
        print(f"\n[FAILED] {description}")
        print(f"Error: {e}")
        raise


def generate_demo_data():
    """Generate synthetic demo data."""
    from demo_generate_data import (
        generate_synthetic_pretraining_data,
        generate_synthetic_instruction_data,
        generate_synthetic_preference_data,
        create_manifest
    )

    os.makedirs("data/raw", exist_ok=True)

    generate_synthetic_pretraining_data(100)
    generate_synthetic_instruction_data(50)
    generate_synthetic_preference_data(30)
    create_manifest()


def create_tokenizer():
    """Create the demo tokenizer."""
    from demo_tokenize import create_simple_tokenizer
    create_simple_tokenizer()


def pack_sequences():
    """Tokenize and pack sequences."""
    from demo_tokenize import simple_pack_sequences
    simple_pack_sequences()


def initialize_model():
    """Initialize the demo model."""
    from demo_init_model import initialize_model as init_model
    init_model()


def verify_setup():
    """Verify all demo files are in place."""
    required_files = [
        "data/raw/pretraining_demo.parquet",
        "data/packed/pretrain_demo.npy",
        "configs/tokenizer/tokenizer_config.json",
        "checkpoints/demo_init/config.json",
    ]

    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"  - {f}")
        return False

    print("\nAll demo files verified!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Set up demo environment")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model initialization (faster)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing setup")
    args = parser.parse_args()

    print("=" * 60)
    print("DEMO SETUP")
    print("=" * 60)
    print("\nThis will set up a small demo environment for testing")
    print("the training pipeline without requiring large datasets")
    print("or GPU resources.\n")

    if args.verify_only:
        success = verify_setup()
        sys.exit(0 if success else 1)

    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # Add scripts to path
    sys.path.insert(0, script_dir)

    steps_completed = 0
    total_steps = 4 if not args.skip_model else 3
    if args.skip_data:
        total_steps -= 1

    try:
        if not args.skip_data:
            run_step(f"[1/{total_steps}] Generate synthetic demo data", generate_demo_data)
            steps_completed += 1

        run_step(f"[{steps_completed+1}/{total_steps}] Create tokenizer", create_tokenizer)
        steps_completed += 1

        run_step(f"[{steps_completed+1}/{total_steps}] Tokenize and pack sequences", pack_sequences)
        steps_completed += 1

        if not args.skip_model:
            run_step(f"[{steps_completed+1}/{total_steps}] Initialize demo model", initialize_model)
            steps_completed += 1

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Setup failed at step {steps_completed + 1}")
        print(f"{'='*60}")
        sys.exit(1)

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}\n")

    if verify_setup():
        print(f"\n{'='*60}")
        print("DEMO SETUP COMPLETE!")
        print(f"{'='*60}")
        print("\nYou can now run:")
        print("  python scripts/demo_pretrain.py")
        print("\nThis will train for 10 steps on the synthetic data.")
    else:
        print("\nSetup completed but some files are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
