# LLM Training Pipeline

Production-ready training pipeline for 7B parameter language models (LLaMA-style architecture) on NVIDIA A100 and H100 GPUs.

## Features

- **Multi-stage training**: Pretraining → SFT → DPO → LoRA
- **GPU optimization**: FP8 support (H100), Flash Attention 2, torch.compile
- **Curriculum learning**: Progressive context length increase
- **Promotion gates**: Automated quality checks between stages
- **Checkpoint management**: Archive, cleanup, and resume utilities

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-training-pipeline

# Install dependencies
pip install -e .

# With FP8 support (H100 only)
pip install -e ".[fp8]"

# With development tools
pip install -e ".[dev]"
```

### Run Full Pipeline

```bash
# Prepare data and run all training stages
bash scripts/run_full_pipeline.sh

# Or run production training only
bash START_PRODUCTION_TRAINING.sh
```

### Run Individual Stages

```bash
# Data preparation
python scripts/01_download_data.py
python scripts/02_clean_deduplicate_optimized.py
python scripts/03_tokenize_and_pack.py

# Model initialization
python scripts/04_init_model.py

# Training stages
python scripts/05_pretrain.py [--fp8|--no-fp8]
python scripts/07_sft.py [--fp8|--no-fp8]
python scripts/09_dpo.py [--fp8|--no-fp8]
python scripts/10_lora_finetune.py

# Evaluation
python scripts/11_evaluate.py checkpoints/dpo_final
python scripts/12_check_gates.py dpo
```

## Script Guide

| Task | Script | Notes |
|------|--------|-------|
| **Production Pretraining** | `scripts/05_pretrain.py` | Full featured, CLI args, curriculum |
| **Demo Pretraining** | `scripts/demo_pretrain.py` | CPU-compatible, tiny model |
| **SFT Training** | `scripts/07_sft.py` | Instruction fine-tuning |
| **DPO Training** | `scripts/09_dpo.py` | Preference optimization |
| **LoRA Fine-tuning** | `scripts/10_lora_finetune.py` | Domain adaptation |
| **Evaluation** | `scripts/11_evaluate.py` | Full eval suite with timeouts |
| **Pre-flight Check** | `scripts/preflight_check.py` | Validate prerequisites |

**Deprecated scripts** (kept for backwards compatibility):
- `production_pretrain.py` → use `05_pretrain.py --fp8`
- `full_model_pretrain.py` → use `05_pretrain.py` or `demo_pretrain.py`

## Training Stages

### 1. Pretraining
- Trains on large text corpora
- Uses curriculum learning (512 → 1024 → 2048 tokens)
- Auto-stops at curriculum boundaries for data reload
- Supports FSDP for multi-GPU

### 2. Supervised Fine-Tuning (SFT)
- Trains on instruction-response pairs
- Uses sequence packing (up to 6x speedup)
- Configurable via `configs/sft.yaml`

### 3. Direct Preference Optimization (DPO)
- Aligns model with human preferences
- Uses chosen/rejected response pairs
- Very conservative learning rate (5e-7)

### 4. LoRA Fine-Tuning (Optional)
- Domain-specific adaptation
- Trainable parameter efficient (LoRA rank 64)
- Supports mixing domain + general data

## GPU Support

| GPU | Precision | Performance |
|-----|-----------|-------------|
| H100 80GB | FP8 | ~25-30 hours (100K steps) |
| H100 80GB | BF16 | ~35-40 hours |
| A100 80GB | BF16 | ~45-50 hours |

### FP8 Training (H100)

```bash
# Auto-detect (uses FP8 if available)
python scripts/05_pretrain.py

# Force FP8
python scripts/05_pretrain.py --fp8

# Force BF16
python scripts/05_pretrain.py --no-fp8
```

**Requirements for FP8:**
```bash
pip install transformer-engine[pytorch]
```

## CLI Options

All training scripts support these arguments:

```bash
python scripts/05_pretrain.py \
  --config configs/pretrain.yaml \
  --max_steps 100000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --logging_steps 10 \
  --output_dir checkpoints/pretrain \
  --resume_from_checkpoint checkpoints/pretrain/checkpoint-5000
```

## Checkpoint Management

```bash
# List checkpoints
bash scripts/checkpoint_manager.sh list

# Archive a checkpoint
bash scripts/checkpoint_manager.sh archive checkpoints/pretrain_final

# Cleanup old checkpoints (keep latest 3)
bash scripts/checkpoint_manager.sh cleanup pretrain 3

# Show disk usage
bash scripts/checkpoint_manager.sh disk-usage

# Resume training
bash scripts/resume_pipeline.sh pretrain
```

## Configuration

### Training Configs

| File | Purpose |
|------|---------|
| `configs/pretrain.yaml` | Pretraining hyperparameters |
| `configs/sft.yaml` | SFT with sequence packing |
| `configs/dpo.yaml` | DPO preference optimization |
| `configs/lora_finetune.yaml` | LoRA domain adaptation |
| `configs/promotion_gates.yaml` | Quality thresholds between stages |

### Model Architecture

- **Parameters**: 7B
- **Layers**: 32
- **Hidden dim**: 4096
- **Attention**: GQA with 8 KV heads
- **Vocab size**: 50,304 (padded for efficiency)
- **Max context**: 4096 tokens

## Evaluation

```bash
# Run full evaluation suite
python scripts/11_evaluate.py checkpoints/dpo_final

# Check promotion gates
python scripts/12_check_gates.py dpo

# Generate training report
python scripts/generate_report.py
```

### Metrics

- **Perplexity**: Language modeling quality
- **HumanEval**: Code generation (requires manual setup)
- **MMLU**: General knowledge
- **Safety**: Refusal rate on harmful prompts

## Directory Structure

```
llm-training-pipeline/
├── configs/           # YAML configs + model architecture
│   ├── pretrain.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   ├── lora_finetune.yaml
│   ├── promotion_gates.yaml
│   ├── model_7b.py
│   └── tokenizer/     # Tokenizer files (created at runtime)
├── scripts/           # Training and utility scripts
│   ├── 01_download_data.py
│   ├── 02_clean_deduplicate.py
│   ├── 03_tokenize_and_pack.py
│   ├── 04_init_model.py
│   ├── 05_pretrain.py
│   ├── 07_sft.py
│   ├── 09_dpo.py
│   ├── 10_lora_finetune.py
│   ├── 11_evaluate.py
│   ├── 12_check_gates.py
│   ├── gpu_utils.py
│   └── ...
├── tests/             # Test suite
├── data/              # Training data (created at runtime)
│   ├── raw/
│   ├── packed/
│   ├── sft/
│   └── dpo/
├── checkpoints/       # Model checkpoints
├── logs/              # TensorBoard logs
└── evals/             # Evaluation results
```

## Monitoring

```bash
# GPU utilization
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir logs/

# Training progress
tail -f production_training.log
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_configs.py -v
```

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size`
- Enable gradient checkpointing (default: on)
- Use FP8 on H100 for 30% memory savings

### Training Crashes

```bash
# Resume from latest checkpoint
bash scripts/resume_pipeline.sh pretrain
```

### Slow Training

- Enable torch.compile (default: on)
- Use Flash Attention 2 (default: on)
- Enable sequence packing for SFT
- Use FP8 on H100

### OOM (Out of Memory) Errors

All training scripts support automatic OOM recovery:

```bash
# Enable OOM recovery for any training script
python scripts/05_pretrain.py --enable-oom-recovery
python scripts/07_sft.py --enable-oom-recovery
python scripts/09_dpo.py --enable-oom-recovery
```

When OOM recovery is enabled:
- GPU memory is automatically cleared on OOM
- Training continues with reduced batch size
- OOM events are logged for monitoring
- Works with both standard and FP8 training paths

## License

[Add your license here]

## Citation

[Add citation information here]
