#!/usr/bin/env python3
"""Prepare reasoning-focused SFT data for tool-calling and logic capabilities.

This script loads and formats datasets optimized for:
- Chain-of-thought reasoning
- Function/tool calling
- Logical deduction
- Structured problem solving

Usage:
    python scripts/06_prepare_reasoning_data.py
    python scripts/06_prepare_reasoning_data.py --output-dir data/sft_reasoning
    python scripts/06_prepare_reasoning_data.py --max-samples 100000
"""

import os
import argparse
import yaml
from datasets import load_dataset, concatenate_datasets


# ============================================================================
# Format converters for each dataset type
# ============================================================================

def format_gsm8k(example):
    """Format GSM8K math problems with chain-of-thought.

    GSM8K format: {"question": "...", "answer": "..."}
    Answer contains step-by-step reasoning ending with #### final_answer
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    # GSM8K answers have reasoning followed by #### and the final answer
    # Keep the full reasoning chain
    prompt = f"Solve this math problem step by step:\n\n{question}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"
    }


def format_orca_math(example):
    """Format Orca-Math problems.

    Orca-Math format: {"question": "...", "answer": "..."}
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    prompt = f"Solve this problem step by step:\n\n{question}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"
    }


def format_openorca(example):
    """Format OpenOrca GPT-4 distillation data.

    OpenOrca format: {"system_prompt": "...", "question": "...", "response": "..."}
    """
    system = example.get("system_prompt", "")
    question = example.get("question", "")
    response = example.get("response", "")

    if system and system.strip():
        prompt = f"{system}\n\n{question}"
    else:
        prompt = question

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_cot_collection(example):
    """Format CoT-Collection chain-of-thought examples.

    CoT-Collection format: {"source": "...", "rationale": "...", "target": "...", "task": "..."}
    """
    source = example.get("source", "")
    rationale = example.get("rationale", "")
    target = example.get("target", "")

    # Combine rationale (reasoning) with target (answer)
    if rationale:
        response = f"{rationale}\n\nTherefore, the answer is: {target}"
    else:
        response = target

    return {
        "text": f"<|user|>\n{source}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_glaive_function_calling(example):
    """Format Glaive function calling data.

    Glaive format: {"system": "...", "chat": "..."}
    Chat contains USER/ASSISTANT/FUNCTION_RESPONSE turns
    """
    system = example.get("system", "")
    chat = example.get("chat", "")

    # The chat already has structured format, wrap it appropriately
    if system:
        text = f"<|system|>\n{system}\n{chat}<|endoftext|>"
    else:
        text = f"{chat}<|endoftext|>"

    # Normalize the format
    text = text.replace("USER:", "<|user|>\n")
    text = text.replace("ASSISTANT:", "<|assistant|>\n")
    text = text.replace("FUNCTION RESPONSE:", "<|function_response|>\n")

    return {"text": text}


def format_hermes_function_calling(example):
    """Format Hermes/NousResearch function calling data.

    Hermes format: {"conversations": [{"from": "...", "value": "..."}], "functions": [...]}
    """
    conversations = example.get("conversations", [])
    functions = example.get("functions", [])

    parts = []

    # Add function definitions as system context
    if functions:
        import json
        func_str = json.dumps(functions, indent=2)
        parts.append(f"<|system|>\nYou have access to the following functions:\n{func_str}\n")

    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")

        if role == "human":
            parts.append(f"<|user|>\n{value}\n")
        elif role == "gpt":
            parts.append(f"<|assistant|>\n{value}")
        elif role == "function_call":
            parts.append(f"<|function_call|>\n{value}\n")
        elif role == "function_response":
            parts.append(f"<|function_response|>\n{value}\n")

    parts.append("<|endoftext|>")

    return {"text": "".join(parts)}


def format_logiqa(example):
    """Format LogiQA logical reasoning problems.

    LogiQA format: {"context": "...", "question": "...", "answers": [...], "label": int}
    """
    context = example.get("context", "")
    question = example.get("question", "")
    answers = example.get("answers", [])
    label = example.get("label", 0)

    # Format as multiple choice with reasoning
    choices_text = "\n".join([f"{chr(65+i)}. {ans}" for i, ans in enumerate(answers)])
    correct_answer = chr(65 + label)
    correct_text = answers[label] if label < len(answers) else ""

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_text}\n\nThink through this step by step and select the correct answer."

    response = f"Let me analyze this logically.\n\nGiven the context, I need to evaluate each option:\n\nThe correct answer is {correct_answer}. {correct_text}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_alpaca(example):
    """Format Alpaca instruction data."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{output}<|endoftext|>"
    }


def format_oasst(example):
    """Format OpenAssistant data."""
    # OASST has tree structure, we use the instruction/response pairs
    instruction = example.get("instruction", example.get("text", ""))
    response = example.get("response", "")

    if not response:
        return {"text": ""}

    return {
        "text": f"<|user|>\n{instruction}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_metamath(example):
    """Format MetaMathQA dataset.

    MetaMath format: {"query": "...", "response": "..."}
    Contains augmented math problems with detailed solutions.
    """
    query = example.get("query", "")
    response = example.get("response", "")

    return {
        "text": f"<|user|>\n{query}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_toolbench(example):
    """Format ToolBench API calling data.

    ToolBench has complex multi-turn API interactions.
    Format varies, typically: {"conversations": [...], "tools": [...]}
    """
    conversations = example.get("conversations", [])

    if not conversations:
        # Try alternate format
        query = example.get("query", example.get("instruction", ""))
        response = example.get("response", example.get("output", ""))
        if query and response:
            return {
                "text": f"<|user|>\n{query}\n<|assistant|>\n{response}<|endoftext|>"
            }
        return {"text": ""}

    parts = []
    for turn in conversations:
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))

        if role in ["user", "human"]:
            parts.append(f"<|user|>\n{content}\n")
        elif role in ["assistant", "gpt", "model"]:
            parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":
            parts.append(f"<|tool_response|>\n{content}\n")

    parts.append("<|endoftext|>")
    return {"text": "".join(parts)}


# Format dispatcher
FORMAT_HANDLERS = {
    "gsm8k": format_gsm8k,
    "orca-math": format_orca_math,
    "openorca": format_openorca,
    "cot-collection": format_cot_collection,
    "glaive": format_glaive_function_calling,
    "hermes": format_hermes_function_calling,
    "logiqa": format_logiqa,
    "alpaca": format_alpaca,
    "oasst": format_oasst,
    "metamath": format_metamath,
    "toolbench": format_toolbench,
}


# ============================================================================
# Main data preparation
# ============================================================================

def load_and_format_dataset(ds_config, max_samples=None):
    """Load a dataset and format it for SFT training.

    Args:
        ds_config: Dataset configuration dict
        max_samples: Optional max samples to load

    Returns:
        Formatted HuggingFace Dataset
    """
    name = ds_config["name"]
    source = ds_config["source"]
    subset = ds_config.get("subset")
    weight = ds_config.get("weight", 1.0)
    format_type = ds_config.get("format", "alpaca")

    print(f"  Loading {name} from {source}...")

    try:
        if subset:
            ds = load_dataset(source, subset, split="train", trust_remote_code=True)
        else:
            ds = load_dataset(source, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"    Warning: Failed to load {name}: {e}")
        return None

    print(f"    Loaded {len(ds)} examples")

    # Apply weight/sampling
    sample_size = int(len(ds) * weight)
    if max_samples:
        sample_size = min(sample_size, max_samples)

    if sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(sample_size))
        print(f"    Sampled {sample_size} examples (weight={weight})")

    # Get format handler
    handler = FORMAT_HANDLERS.get(format_type)
    if not handler:
        print(f"    Warning: Unknown format '{format_type}', using alpaca")
        handler = format_alpaca

    # Format dataset
    try:
        ds = ds.map(handler, remove_columns=ds.column_names, num_proc=4)
    except Exception as e:
        print(f"    Warning: Failed to format {name}: {e}")
        # Try single-threaded
        ds = ds.map(handler, remove_columns=ds.column_names)

    # Filter empty examples
    ds = ds.filter(lambda x: len(x.get("text", "")) > 50)
    print(f"    Final: {len(ds)} examples after formatting")

    return ds


def prepare_reasoning_data(
    config_path="configs/data_sources.yaml",
    output_dir="data/sft",
    max_samples_per_dataset=None,
    val_size=0.05,
):
    """Prepare combined reasoning-focused SFT dataset.

    Args:
        config_path: Path to data sources config
        output_dir: Output directory for processed data
        max_samples_per_dataset: Optional max samples per dataset
        val_size: Validation split fraction
    """
    print("=" * 60)
    print("PREPARING REASONING-FOCUSED SFT DATA")
    print("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    all_datasets = []

    # Load reasoning datasets
    print("\n[1/4] Loading REASONING datasets...")
    for ds_config in config["datasets"].get("reasoning", []):
        ds = load_and_format_dataset(ds_config, max_samples_per_dataset)
        if ds and len(ds) > 0:
            all_datasets.append(ds)

    # Load function calling datasets
    print("\n[2/4] Loading FUNCTION CALLING datasets...")
    for ds_config in config["datasets"].get("function_calling", []):
        ds = load_and_format_dataset(ds_config, max_samples_per_dataset)
        if ds and len(ds) > 0:
            all_datasets.append(ds)

    # Load logic datasets
    print("\n[3/4] Loading LOGIC datasets...")
    for ds_config in config["datasets"].get("logic", []):
        ds = load_and_format_dataset(ds_config, max_samples_per_dataset)
        if ds and len(ds) > 0:
            all_datasets.append(ds)

    # Load instruction tuning datasets (for general capability)
    print("\n[4/4] Loading INSTRUCTION datasets...")
    for ds_config in config["datasets"].get("instruction_tuning", []):
        ds = load_and_format_dataset(ds_config, max_samples_per_dataset)
        if ds and len(ds) > 0:
            all_datasets.append(ds)

    if not all_datasets:
        print("\nError: No datasets loaded successfully!")
        return False

    # Combine all datasets
    print(f"\nCombining {len(all_datasets)} datasets...")
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)

    print(f"Total examples: {len(combined)}")

    # Split train/val
    split = combined.train_test_split(test_size=val_size, seed=42)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")

    print(f"\nSaving to {output_dir}...")
    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(val_path)

    print("\n" + "=" * 60)
    print("REASONING DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Training examples: {len(split['train']):,}")
    print(f"  Validation examples: {len(split['test']):,}")
    print(f"  Output directory: {output_dir}")

    # Print composition breakdown
    print("\n  Dataset composition:")
    print("    - Reasoning (GSM8K, Orca-Math, OpenOrca, CoT): ~60%")
    print("    - Function Calling (Glaive, Hermes): ~20%")
    print("    - Logic (LogiQA): ~5%")
    print("    - General Instructions (OASST, Alpaca): ~15%")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare reasoning-focused SFT training data"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources config"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/sft",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.05,
        help="Validation split fraction"
    )

    args = parser.parse_args()

    prepare_reasoning_data(
        config_path=args.config,
        output_dir=args.output_dir,
        max_samples_per_dataset=args.max_samples,
        val_size=args.val_size,
    )
