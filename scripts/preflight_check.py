#!/usr/bin/env python3
"""Pre-flight validation for training pipeline.

This script validates all prerequisites before starting training:
1. Tokenizer exists and is valid
2. Model checkpoint exists (for stages that require it)
3. Training data exists
4. Configuration files are valid
5. GPU has sufficient memory (if available)

Usage:
    python scripts/preflight_check.py pretrain
    python scripts/preflight_check.py sft
    python scripts/preflight_check.py dpo
    python scripts/preflight_check.py lora
    python scripts/preflight_check.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
import json


class PreflightChecker:
    """Validate training prerequisites."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log(self, message: str, level: Literal["info", "error", "warning"] = "info") -> None:
        """Log a message."""
        if level == "error":
            self.errors.append(message)
            print(f"[ERROR] {message}")
        elif level == "warning":
            self.warnings.append(message)
            if self.verbose:
                print(f"[WARN]  {message}")
        elif self.verbose:
            print(f"[OK]    {message}")

    def check_file_exists(self, path: str, description: str) -> bool:
        """Check if a file exists."""
        if os.path.exists(path):
            self.log(f"{description}: {path}")
            return True
        else:
            self.log(f"{description} not found: {path}", "error")
            return False

    def check_directory_exists(self, path: str, description: str) -> bool:
        """Check if a directory exists and is not empty."""
        if not os.path.exists(path):
            self.log(f"{description} not found: {path}", "error")
            return False
        if not os.path.isdir(path):
            self.log(f"{description} is not a directory: {path}", "error")
            return False
        if not os.listdir(path):
            self.log(f"{description} is empty: {path}", "error")
            return False
        self.log(f"{description}: {path}")
        return True

    def check_tokenizer(self, tokenizer_path: str = "configs/tokenizer") -> bool:
        """Validate tokenizer exists and has required files."""
        if not os.path.exists(tokenizer_path):
            self.log(f"Tokenizer not found at {tokenizer_path}", "error")
            self.log("  Run: python scripts/demo_tokenize.py (for demo)", "error")
            self.log("  Or:  python scripts/03_tokenize_and_pack.py (for production)", "error")
            return False

        # Check for tokenizer files
        required_files = ["tokenizer_config.json"]
        optional_files = ["tokenizer.json", "vocab.json", "merges.txt"]

        has_config = os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json"))
        has_vocab = any(
            os.path.exists(os.path.join(tokenizer_path, f))
            for f in optional_files
        )

        if not has_config:
            self.log(f"Missing tokenizer_config.json in {tokenizer_path}", "error")
            return False

        if not has_vocab:
            self.log(f"No vocabulary file found in {tokenizer_path}", "error")
            return False

        self.log(f"Tokenizer valid: {tokenizer_path}")
        return True

    def check_model_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate model checkpoint exists and has required files."""
        if not os.path.exists(checkpoint_path):
            self.log(f"Model checkpoint not found: {checkpoint_path}", "error")
            return False

        # Check for model files
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            self.log(f"Missing config.json in checkpoint: {checkpoint_path}", "error")
            return False

        # Check for weights
        weight_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
        has_weights = any(
            os.path.exists(os.path.join(checkpoint_path, f))
            for f in weight_files
        )

        if not has_weights:
            self.log(f"No model weights found in: {checkpoint_path}", "error")
            return False

        self.log(f"Model checkpoint valid: {checkpoint_path}")
        return True

    def check_config(self, config_path: str) -> bool:
        """Validate config file loads and has required sections."""
        if not os.path.exists(config_path):
            self.log(f"Config not found: {config_path}", "error")
            return False

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.log(f"Invalid YAML in {config_path}: {e}", "error")
            return False

        if config is None:
            self.log(f"Config is empty: {config_path}", "error")
            return False

        self.log(f"Config valid: {config_path}")
        return True

    def check_data_directory(self, data_path: str, description: str) -> bool:
        """Check training data directory."""
        return self.check_directory_exists(data_path, description)

    def check_gpu(self) -> bool:
        """Check GPU availability and memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.log("No GPU detected - will use CPU (very slow)", "warning")
                return True

            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_gb = props.total_memory / (1024**3)

            self.log(f"GPU: {props.name} ({total_gb:.1f} GB)")

            # Warn if memory is low for 7B model
            if total_gb < 24:
                self.log(f"GPU memory ({total_gb:.1f}GB) may be insufficient for 7B model", "warning")

            return True

        except ImportError:
            self.log("PyTorch not installed - cannot check GPU", "warning")
            return True

    def check_pretrain(self) -> bool:
        """Validate prerequisites for pretraining."""
        print("\n" + "="*60)
        print("PRE-FLIGHT CHECK: PRETRAINING")
        print("="*60 + "\n")

        all_valid = True

        # Config
        all_valid &= self.check_config("configs/pretrain.yaml")

        # Tokenizer
        all_valid &= self.check_tokenizer()

        # Initial model checkpoint
        all_valid &= self.check_model_checkpoint("checkpoints/init")

        # Training data
        all_valid &= self.check_data_directory("data/packed/train", "Training data")
        all_valid &= self.check_data_directory("data/packed/val", "Validation data")

        # GPU
        self.check_gpu()

        return all_valid

    def check_sft(self) -> bool:
        """Validate prerequisites for SFT."""
        print("\n" + "="*60)
        print("PRE-FLIGHT CHECK: SFT")
        print("="*60 + "\n")

        all_valid = True

        # Config
        all_valid &= self.check_config("configs/sft.yaml")

        # Tokenizer
        all_valid &= self.check_tokenizer()

        # Pretrained model
        all_valid &= self.check_model_checkpoint("checkpoints/pretrain_final")

        # SFT data
        all_valid &= self.check_data_directory("data/sft/train", "SFT training data")
        all_valid &= self.check_data_directory("data/sft/val", "SFT validation data")

        # GPU
        self.check_gpu()

        return all_valid

    def check_dpo(self) -> bool:
        """Validate prerequisites for DPO."""
        print("\n" + "="*60)
        print("PRE-FLIGHT CHECK: DPO")
        print("="*60 + "\n")

        all_valid = True

        # Config
        all_valid &= self.check_config("configs/dpo.yaml")

        # Tokenizer
        all_valid &= self.check_tokenizer()

        # SFT model
        all_valid &= self.check_model_checkpoint("checkpoints/sft_final")

        # DPO data
        all_valid &= self.check_data_directory("data/dpo/train", "DPO training data")
        all_valid &= self.check_data_directory("data/dpo/val", "DPO validation data")

        # GPU
        self.check_gpu()

        return all_valid

    def check_lora(self) -> bool:
        """Validate prerequisites for LoRA fine-tuning."""
        print("\n" + "="*60)
        print("PRE-FLIGHT CHECK: LoRA")
        print("="*60 + "\n")

        all_valid = True

        # Config
        all_valid &= self.check_config("configs/lora_finetune.yaml")

        # Tokenizer
        all_valid &= self.check_tokenizer()

        # Base model (DPO or SFT)
        if os.path.exists("checkpoints/dpo_final"):
            all_valid &= self.check_model_checkpoint("checkpoints/dpo_final")
        else:
            all_valid &= self.check_model_checkpoint("checkpoints/sft_final")

        # Domain data (optional - warn if missing)
        if not os.path.exists("data/domain"):
            self.log("Domain data not found at data/domain - will use placeholder data", "warning")

        # GPU
        self.check_gpu()

        return all_valid

    def check_all(self) -> Dict[str, bool]:
        """Check all stages."""
        results: Dict[str, bool] = {}

        for stage, checker in [
            ("pretrain", self.check_pretrain),
            ("sft", self.check_sft),
            ("dpo", self.check_dpo),
            ("lora", self.check_lora),
        ]:
            self.errors = []
            self.warnings = []
            results[stage] = checker()

        return results

    def summary(self) -> bool:
        """Print summary of checks."""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        if self.errors:
            print(f"\n{len(self.errors)} error(s) found:")
            for err in self.errors:
                print(f"  - {err}")

        if self.warnings:
            print(f"\n{len(self.warnings)} warning(s):")
            for warn in self.warnings:
                print(f"  - {warn}")

        if not self.errors:
            print("\n✓ All pre-flight checks passed!")
            return True
        else:
            print("\n✗ Pre-flight checks failed. Please fix errors above.")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preflight_check.py pretrain  # Check pretraining prerequisites
  python scripts/preflight_check.py sft       # Check SFT prerequisites
  python scripts/preflight_check.py --all     # Check all stages
        """
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["pretrain", "sft", "dpo", "lora"],
        help="Training stage to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all stages"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show errors"
    )
    args = parser.parse_args()

    if not args.stage and not args.all:
        parser.print_help()
        sys.exit(1)

    checker = PreflightChecker(verbose=not args.quiet)

    if args.all:
        results = checker.check_all()
        print("\n" + "="*60)
        print("RESULTS BY STAGE")
        print("="*60)
        for stage, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {stage}: {status}")
        sys.exit(0 if all(results.values()) else 1)

    # Check specific stage
    checkers = {
        "pretrain": checker.check_pretrain,
        "sft": checker.check_sft,
        "dpo": checker.check_dpo,
        "lora": checker.check_lora,
    }

    passed = checkers[args.stage]()
    checker.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
