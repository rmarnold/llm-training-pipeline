# Data Formats

This document describes the expected data formats for each stage of the training pipeline.

## Overview

| Stage | Input Format | Output Location | Key Fields |
|-------|--------------|-----------------|------------|
| Pretraining | Packed numpy/HF Dataset | `data/packed/` | `input_ids` |
| SFT | HF Dataset | `data/sft/` | `messages` or `text` |
| DPO | HF Dataset | `data/dpo/` | `prompt`, `chosen`, `rejected` |
| LoRA | HF Dataset | `data/domain/` | `text` or `messages` |
| Rust Lang Adapter | HF Dataset (Harmony) | `data/rust/lang_rust/` | `text` |
| Rust Agent Trajectories | HF Dataset (Harmony) | `data/rust/core_agent/` | `text` (with tool calls) |
| Rust Mutations | JSONL + HF Dataset | `data/rust/mutations/` | `buggy_code`, `error_message`, `fixed_code` |
| Rust IPO Preferences | HF Dataset (Harmony) | `data/rust/ipo/` | `prompt`, `chosen`, `rejected` |
| Rust GRPO Tasks | JSONL + HF Dataset | `data/rust/grpo/` | `description`, `starter_code`, `tests` |
| Rust Eval Tasks | JSONL | `data/rust/eval/` | `description`, `tests` |

---

## 1. Pretraining Data

### Format
Pretraining uses packed sequences stored as either:
- **NumPy arrays** (`.npy` files)
- **HuggingFace Datasets** (Arrow format)

### Structure

```python
# NumPy format
data = np.load("data/packed/pretrain.npy")
# Shape: (num_sequences, sequence_length)
# dtype: int64 (token IDs)
# Example: (100000, 2048)

# HuggingFace Dataset format
from datasets import load_from_disk
dataset = load_from_disk("data/packed/train")
# Required column: "input_ids"
# Each row: list of token IDs
```

### Example

```python
# Single example from dataset
{
    "input_ids": [1, 4521, 289, 15, 2847, ..., 2]  # Length: 2048
}
```

### Curriculum Learning

For curriculum learning, prepare data at different sequence lengths:

```
data/packed/
├── train_512/      # 512-token sequences
├── train_1024/     # 1024-token sequences
├── train_2048/     # 2048-token sequences
├── val_512/
├── val_1024/
└── val_2048/
```

### Creating Pretraining Data

```bash
# From raw text
python scripts/03_tokenize_and_pack.py \
    --input data/raw/*.parquet \
    --output data/packed/train \
    --seq_length 2048
```

---

## 2. SFT (Supervised Fine-Tuning) Data

### Format
SFT data uses the chat/instruction format with messages.

### Structure

```python
from datasets import load_from_disk
dataset = load_from_disk("data/sft/train")

# Option 1: Messages format (recommended)
# Required column: "messages"
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."}
    ]
}

# Option 2: Text format (pre-formatted)
# Required column: "text"
{
    "text": "<|system|>You are helpful.<|user|>What is Python?<|assistant|>Python is..."
}
```

### Example

```python
# Full example with multiple turns
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful, harmless, and honest assistant."
        },
        {
            "role": "user",
            "content": "Explain machine learning in simple terms."
        },
        {
            "role": "assistant",
            "content": "Machine learning is a type of artificial intelligence..."
        },
        {
            "role": "user",
            "content": "Can you give an example?"
        },
        {
            "role": "assistant",
            "content": "Sure! A common example is email spam filtering..."
        }
    ]
}
```

### Creating SFT Data

```bash
python scripts/06_prepare_sft_data.py
```

### Validation

```python
# Required fields check
def validate_sft_example(example):
    if "messages" in example:
        assert isinstance(example["messages"], list)
        for msg in example["messages"]:
            assert "role" in msg and "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]
    elif "text" in example:
        assert isinstance(example["text"], str)
        assert len(example["text"]) > 0
    else:
        raise ValueError("Must have 'messages' or 'text' field")
```

---

## 3. DPO (Direct Preference Optimization) Data

### Format
DPO requires preference pairs with a prompt and two responses.

### Structure

```python
from datasets import load_from_disk
dataset = load_from_disk("data/dpo/train")

# Required columns: "prompt", "chosen", "rejected"
{
    "prompt": str,      # The conversation prompt
    "chosen": str,      # The preferred response
    "rejected": str     # The less preferred response
}
```

### Example

```python
{
    "prompt": "Human: What's the best way to learn programming?\n\nAssistant:",
    "chosen": "The best way to learn programming is to start with fundamentals...",
    "rejected": "Just copy code from the internet and hope it works."
}
```

### Creating DPO Data

```bash
# Full preparation with safety filtering
python scripts/08_prepare_dpo_data.py

# Skip safety filter (faster)
python scripts/08_prepare_dpo_data.py --skip-safety-filter

# Validate existing data
python scripts/08_prepare_dpo_data.py --validate-only
```

### Validation

```python
def validate_dpo_example(example):
    required = ["prompt", "chosen", "rejected"]
    for field in required:
        assert field in example, f"Missing {field}"
        assert isinstance(example[field], str), f"{field} must be string"
        assert len(example[field].strip()) > 0, f"{field} is empty"

    # Chosen and rejected should be different
    assert example["chosen"] != example["rejected"], "Responses are identical"
```

---

## 4. LoRA Fine-Tuning Data

### Format
LoRA data can use either text or messages format, depending on the task.

### Structure

```python
# For domain-specific text continuation
{
    "text": "def calculate_fibonacci(n):\n    if n <= 1:\n        return n..."
}

# For instruction following
{
    "messages": [
        {"role": "user", "content": "Write a function to calculate factorial"},
        {"role": "assistant", "content": "def factorial(n):\n    ..."}
    ]
}
```

### Directory Structure

```
data/domain/
├── coding/
│   ├── train/
│   └── val/
├── medical/
│   ├── train/
│   └── val/
└── legal/
    ├── train/
    └── val/
```

### Creating LoRA Data

```bash
python scripts/prepare_lora_data.py --domain coding
```

---

## 5. GPT-OSS 20B Rust Agent Data (Harmony Format)

All GPT-OSS training data uses the **Harmony** chat format (`dataset_formatters/harmony.py`). Harmony supports tool calls, thinking/reasoning fields, and multi-channel output. Data that is not in Harmony format will degrade model performance.

### Harmony Encoding

Harmony uses special tokens for role hierarchy: `<|system|>`, `<|developer|>`, `<|user|>`, `<|assistant|>`, `<|tool_call|>`, `<|tool_result|>`, `<|thinking|>`, `<|endoftext|>`.

```python
from dataset_formatters.harmony import encode_harmony_messages

messages = [
    {"role": "user", "content": "Fix the failing test in src/parser.rs"},
    {"role": "assistant", "thinking": "Let me check the test output...",
     "tool_calls": [{"id": "1", "name": "run_tests", "arguments": {"cmd": "cargo test"}}]},
    {"role": "tool", "tool_call_id": "1", "content": "FAILED: test_parse_expr"},
    {"role": "assistant", "content": "The issue is..."},
]
text = encode_harmony_messages(messages, developer_instructions="You are a Rust coding agent.")
```

### 5a. Mutation Data

Generated by `scripts/16_generate_mutations.py` using `cargo-mutants` on curated Rust repos.

```python
# mutations.jsonl — one object per line
{
    "buggy_code": "fn add(a: i32, b: i32) -> i32 { a - b }",
    "error_message": "test test_add failed: assertion 3 != -1",
    "fixed_code": "fn add(a: i32, b: i32) -> i32 { a + b }",
    "repo": "example/math-utils",
    "file": "src/lib.rs"
}
```

### 5b. Agent Trajectories

Generated by `scripts/15_generate_trajectories.py`. Multi-turn tool-use sessions in Harmony format.

```python
# HF Dataset with "text" column (Harmony-encoded)
{
    "text": "<|developer|>\nYou are a Rust coding agent...\n<|user|>\nFix the failing test...\n<|thinking|>\n...\n<|tool_call|>\n{...}\n<|tool_result|>\n...\n<|assistant|>\n...\n<|endoftext|>"
}
```

### 5c. IPO Preference Pairs

Pairs of chosen (passing, idiomatic) vs rejected (failing or less idiomatic) solutions.

```python
# HF Dataset with prompt/chosen/rejected (Harmony-encoded)
{
    "prompt": "Fix this Rust code...",
    "chosen": "<|developer|>...<|endoftext|>",   # Harmony-encoded chosen
    "rejected": "<|developer|>...<|endoftext|>"   # Harmony-encoded rejected
}
```

### 5d. GRPO Tasks

Coding tasks with test suites for RL training.

```python
# tasks.jsonl
{
    "description": "Implement a function that merges two sorted vectors.",
    "starter_code": "fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> { todo!() }",
    "tests": "#[test]\nfn test_basic() { assert_eq!(merge_sorted(&[1,3], &[2,4]), vec![1,2,3,4]); }"
}
```

### Creating Rust Pipeline Data

```bash
# Generate mutations from curated repos
python scripts/16_generate_mutations.py --max_mutations_per_repo 100 --jobs 4

# Generate agent trajectories
python scripts/15_generate_trajectories.py --max_samples 5000

# Data sources configured in configs/data_sources_rust.yaml
```

---

---

## 6. MoE Expert LoRA Configuration (GPT-OSS 20B)

GPT-OSS 20B uses Mixture of Experts (MoE) with fused expert FFN layers. Standard LoRA target module names miss expert layers entirely.

### Correct Target Modules

```yaml
# configs/lang_rust.yaml or core_agent.yaml
lora:
  target_modules:
    - "q_proj"        # Attention query
    - "k_proj"        # Attention key
    - "v_proj"        # Attention value
    - "o_proj"        # Attention output
    - "gate_up_proj"  # MoE expert FFN (singular — Unsloth maps to fused experts)
    - "down_proj"     # MoE expert FFN (singular — Unsloth maps to fused experts)
```

### Wrong Target Modules (common mistake)

```yaml
# These SILENTLY MISS expert layers on MoE models:
target_modules:
  - "gate_proj"   # Dense model FFN — does NOT match MoE gate_up_projs
  - "up_proj"     # Dense model FFN — does NOT match MoE gate_up_projs
  - "down_proj"   # Happens to share name but doesn't target expert copies
```

### Auto-Detection

`apply_lora_config()` in `pipeline_lib/unsloth_utils.py` auto-detects MoE architecture and corrects target modules at runtime (`auto_detect_moe=True` by default). The YAML configs are also updated, so both paths are covered.

### Verification

```python
from pipeline_lib.unsloth_utils import verify_expert_lora
result = verify_expert_lora(model)
# result["has_expert_lora"] should be True
# Expect ~200M+ trainable params (vs ~31.8M attention-only)
```

### Known Bugs

| Bug | Effect | Workaround |
|-----|--------|------------|
| Unsloth #3405 | Default target modules miss MoE experts | Use singular names + auto-detection |
| Unsloth #3701 | Save validation fails with expert LoRA | PEFT native save fallback in `save_adapter()` |

---

## Common Issues

### 1. Missing Fields

```
Error: Missing required field 'input_ids'
```

**Solution**: Ensure your dataset has the required columns for that stage.

### 2. Wrong Data Types

```
Error: Field 'messages' must be a list
```

**Solution**: Check that fields have correct types (list for messages, str for text).

### 3. Empty Values

```
Error: Field 'chosen' is empty after stripping
```

**Solution**: Filter out examples with empty fields before saving.

### 4. Tokenizer Mismatch

```
Error: Token ID 50000 out of vocabulary range
```

**Solution**: Ensure data was tokenized with the same tokenizer used for training.

---

## Validation Scripts

### Validate All Data

```bash
# Check pretraining data
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/packed/train')
print(f'Columns: {ds.column_names}')
print(f'Examples: {len(ds)}')
print(f'Sample: {ds[0]}')
"

# Check SFT data
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/sft/train')
print(f'Columns: {ds.column_names}')
print(f'Sample messages: {ds[0][\"messages\"][:2]}')
"

# Check DPO data
python scripts/08_prepare_dpo_data.py --validate-only
```

---

## Converting Between Formats

### JSON to HuggingFace Dataset

```python
from datasets import Dataset
import json

with open("data.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
dataset.save_to_disk("data/output")
```

### Parquet to HuggingFace Dataset

```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files="data.parquet")
dataset["train"].save_to_disk("data/output")
```

### CSV to HuggingFace Dataset

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="data.csv")
dataset["train"].save_to_disk("data/output")
```
