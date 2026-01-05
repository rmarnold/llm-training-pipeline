# Data Formats

This document describes the expected data formats for each stage of the training pipeline.

## Overview

| Stage | Input Format | Output Location | Key Fields |
|-------|--------------|-----------------|------------|
| Pretraining | Packed numpy/HF Dataset | `data/packed/` | `input_ids` |
| SFT | HF Dataset | `data/sft/` | `messages` or `text` |
| DPO | HF Dataset | `data/dpo/` | `prompt`, `chosen`, `rejected` |
| LoRA | HF Dataset | `data/domain/` | `text` or `messages` |

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
