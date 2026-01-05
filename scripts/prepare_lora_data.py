"""Prepare domain-specific data for LoRA fine-tuning.

This script downloads and prepares coding-focused datasets for LoRA training.
The resulting dataset is saved in HuggingFace datasets format.

Usage:
    python scripts/prepare_lora_data.py --domain coding
    python scripts/prepare_lora_data.py --domain medical
    python scripts/prepare_lora_data.py --domain legal
"""
import os
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets


def prepare_coding_data(output_dir="data/domain/coding", max_samples=50000):
    """Prepare coding-focused dataset for LoRA fine-tuning.

    Uses a mix of:
    - Code instruction datasets
    - Programming Q&A
    - Code documentation
    """
    print("Preparing coding domain dataset...")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    # 1. Code Alpaca - instruction-tuned coding data
    print("  Loading Code Alpaca dataset...")
    try:
        code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for sample in code_alpaca:
            text = f"### Instruction:\n{sample['instruction']}\n\n"
            if sample.get('input'):
                text += f"### Input:\n{sample['input']}\n\n"
            text += f"### Response:\n{sample['output']}"
            all_samples.append({"text": text, "source": "code_alpaca"})
        print(f"    Added {len(code_alpaca)} samples from Code Alpaca")
    except Exception as e:
        print(f"    Warning: Could not load Code Alpaca: {e}")

    # 2. Python code instructions
    print("  Loading Python code instructions...")
    try:
        python_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for sample in python_code:
            text = f"### Instruction:\n{sample['instruction']}\n\n"
            if sample.get('input'):
                text += f"### Input:\n{sample['input']}\n\n"
            text += f"### Response:\n{sample['output']}"
            all_samples.append({"text": text, "source": "python_instructions"})
        print(f"    Added {len(python_code)} samples from Python instructions")
    except Exception as e:
        print(f"    Warning: Could not load Python instructions: {e}")

    # 3. Evol-Instruct-Code (if available)
    print("  Loading Evol-Instruct-Code...")
    try:
        evol_code = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
        for sample in list(evol_code)[:20000]:  # Limit to 20k
            text = f"### Instruction:\n{sample['instruction']}\n\n"
            text += f"### Response:\n{sample['output']}"
            all_samples.append({"text": text, "source": "evol_instruct_code"})
        print(f"    Added samples from Evol-Instruct-Code")
    except Exception as e:
        print(f"    Warning: Could not load Evol-Instruct-Code: {e}")

    if not all_samples:
        print("  No datasets loaded! Creating minimal placeholder dataset...")
        # Create a minimal placeholder dataset
        placeholder_samples = [
            {
                "text": "### Instruction:\nWrite a Python function to calculate factorial.\n\n### Response:\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
                "source": "placeholder"
            },
            {
                "text": "### Instruction:\nExplain what a binary search tree is.\n\n### Response:\nA binary search tree (BST) is a data structure where each node has at most two children. The left subtree contains nodes with keys less than the parent, and the right subtree contains nodes with keys greater than the parent.",
                "source": "placeholder"
            },
        ]
        all_samples = placeholder_samples * 100  # Repeat to have some data

    # Limit samples
    if len(all_samples) > max_samples:
        import random
        random.seed(42)
        random.shuffle(all_samples)
        all_samples = all_samples[:max_samples]

    # Create dataset
    dataset = Dataset.from_list(all_samples)

    # Split into train/val
    split = dataset.train_test_split(test_size=0.05, seed=42)

    # Save
    split["train"].save_to_disk(output_dir)
    print(f"\nCoding dataset prepared:")
    print(f"  Train samples: {len(split['train'])}")
    print(f"  Saved to: {output_dir}")

    return split["train"]


def prepare_medical_data(output_dir="data/domain/medical", max_samples=50000):
    """Prepare medical-focused dataset for LoRA fine-tuning."""
    print("Preparing medical domain dataset...")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    print("  Loading medical datasets...")
    try:
        # Medical Q&A dataset
        med_qa = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        for sample in list(med_qa)[:max_samples]:
            text = f"### Question:\n{sample['input']}\n\n### Answer:\n{sample['output']}"
            all_samples.append({"text": text, "source": "medical_flashcards"})
        print(f"    Added {len(all_samples)} samples from medical flashcards")
    except Exception as e:
        print(f"    Warning: Could not load medical dataset: {e}")
        # Placeholder
        all_samples = [{"text": "Medical domain placeholder", "source": "placeholder"}] * 100

    dataset = Dataset.from_list(all_samples)
    dataset.save_to_disk(output_dir)

    print(f"\nMedical dataset prepared:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Saved to: {output_dir}")

    return dataset


def prepare_legal_data(output_dir="data/domain/legal", max_samples=50000):
    """Prepare legal-focused dataset for LoRA fine-tuning."""
    print("Preparing legal domain dataset...")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    print("  Loading legal datasets...")
    try:
        # Legal Q&A or contract analysis datasets
        legal_ds = load_dataset("nguha/legalbench", "contract_qa", split="train")
        for sample in list(legal_ds)[:max_samples]:
            text = f"### Contract Clause:\n{sample.get('text', sample.get('input', ''))}\n\n"
            text += f"### Question:\n{sample.get('question', 'Analyze this clause.')}\n\n"
            text += f"### Answer:\n{sample.get('answer', sample.get('output', ''))}"
            all_samples.append({"text": text, "source": "legalbench"})
        print(f"    Added {len(all_samples)} samples from LegalBench")
    except Exception as e:
        print(f"    Warning: Could not load legal dataset: {e}")
        # Placeholder
        all_samples = [{"text": "Legal domain placeholder", "source": "placeholder"}] * 100

    dataset = Dataset.from_list(all_samples)
    dataset.save_to_disk(output_dir)

    print(f"\nLegal dataset prepared:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Saved to: {output_dir}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare domain-specific data for LoRA fine-tuning")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["coding", "medical", "legal", "all"],
        default="coding",
        help="Domain to prepare data for"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum samples to include"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    args = parser.parse_args()

    if args.domain == "coding" or args.domain == "all":
        output_dir = args.output_dir or "data/domain/coding"
        prepare_coding_data(output_dir, args.max_samples)

    if args.domain == "medical" or args.domain == "all":
        output_dir = args.output_dir or "data/domain/medical"
        prepare_medical_data(output_dir, args.max_samples)

    if args.domain == "legal" or args.domain == "all":
        output_dir = args.output_dir or "data/domain/legal"
        prepare_legal_data(output_dir, args.max_samples)

    print("\nDomain data preparation complete!")
    print("Next step: Run LoRA fine-tuning with:")
    print("  python scripts/10_lora_finetune.py")


if __name__ == "__main__":
    main()
