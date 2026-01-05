# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production LLM training pipeline for a 7B parameter model (LLaMA-style architecture) targeting NVIDIA A100 and H100 80GB GPUs. The pipeline implements a staged training process: Pretraining → SFT → DPO → LoRA fine-tuning, with promotion gates between stages.

## Commands

### Full Pipeline
```bash
bash scripts/run_full_pipeline.sh
```

### Production Training (Optimized)
```bash
bash START_PRODUCTION_TRAINING.sh
# Or directly:
python scripts/production_pretrain.py

# Force FP8 on H100:
python scripts/production_pretrain.py --fp8

# Force BF16 (disable FP8):
python scripts/production_pretrain.py --no-fp8
```

### Individual Stages
```bash
# Data preparation
python scripts/01_download_data.py
python scripts/02_clean_deduplicate_optimized.py  # GPU-accelerated
python scripts/03_tokenize_and_pack.py

# Model initialization
python scripts/04_init_model.py

# Training stages (all support --fp8 and --no-fp8 flags)
python scripts/05_pretrain.py [--fp8|--no-fp8]
python scripts/07_sft.py [--fp8|--no-fp8]     # Uses sequence packing
python scripts/09_dpo.py [--fp8|--no-fp8]
python scripts/10_lora_finetune.py

# Evaluation & gate checks
python scripts/11_evaluate.py checkpoints/<stage>_final
python scripts/12_check_gates.py <stage>  # stage: pretrain, sft, dpo
```

### Resume Training
```bash
# Auto-resume from latest checkpoint
bash scripts/resume_pipeline.sh pretrain|sft|dpo
```

### GPU Verification
```bash
python scripts/verify_gpu.py
python scripts/profile_memory.py
```

## Architecture

### Training Stages Flow
```
Data Pipeline → Model Init → Pretrain → SFT → DPO → LoRA (optional)
                                ↓         ↓      ↓
                            [gates]  [gates] [gates]
```

Each stage has promotion gates defined in `configs/promotion_gates.yaml` that must pass before advancing.

### Model Configuration
- Architecture: 7B params, 32 layers, 4096 hidden dim, GQA (8 KV heads)
- Vocab: 50,304 tokens (padded for efficiency)
- Max context: 4096 tokens with curriculum (512→1024→2048)
- Config file: `configs/model_7b.py`

### Training Configs
- `configs/pretrain.yaml` - Pretraining with context curriculum, FSDP
- `configs/sft.yaml` - Supervised fine-tuning with sequence packing
- `configs/dpo.yaml` - Preference optimization (1 epoch, very low LR)
- `configs/lora_finetune.yaml` - LoRA adapters for domain adaptation

## Optimizations

### GPU Auto-Detection
Scripts automatically detect GPU type and optimize settings via `scripts/gpu_utils.py`:
- **H100 with FP8**: Auto-enables FP8 precision (30-40% faster than BF16)
- **H100 without FP8**: Uses `max-autotune` compile mode
- **A100**: Uses BF16 with `default` compile mode

### FP8 Training (H100 Only)
FP8 training provides 30-40% speedup on H100 GPUs using NVIDIA Transformer Engine.

**Requirements:**
```bash
pip install transformer-engine[pytorch]
```

**FP8 Precision Formats:**
- E4M3: Used for forward pass (activations and weights)
- E5M2: Used for backward pass (gradients, higher dynamic range)
- HYBRID: Auto-switches between E4M3/E5M2 (recommended)

**Usage:**
```bash
# Auto-detect (uses FP8 if H100 + transformer-engine available)
python scripts/production_pretrain.py

# Force FP8
python scripts/production_pretrain.py --fp8

# Disable FP8, use BF16
python scripts/production_pretrain.py --no-fp8
```

### Enabled Optimizations
| Optimization | Impact | Files |
|--------------|--------|-------|
| FP8 (H100) | 30-40% speedup | All training scripts |
| torch.compile | 30-100% speedup | All training scripts |
| Flash Attention 2 | 2-4x attention speedup | All training scripts |
| Sequence packing (SFT) | Up to 6x speedup | `07_sft.py` |
| Gradient checkpointing (non-reentrant) | Better compile compat | All training scripts |
| Persistent workers | 2-5% speedup | All training scripts |
| BF16 + TF32 | 2x memory reduction | All training scripts |
| AdamW Fused | 5-10% speedup | All training scripts |
| FSDP | Multi-GPU scaling | `05_pretrain.py` |

### Key Training Parameters
```yaml
# Pretraining (production_pretrain.py)
per_device_train_batch_size: 8
gradient_accumulation_steps: 4   # Effective batch: 32
warmup_steps: 5000               # 5% of training
eval_steps: 1000                 # Reduced for less overhead
torch_compile: true

# SFT (07_sft.py)
packing: true                    # Up to 6x speedup
max_seq_length: 2048

# DPO (09_dpo.py)
beta: 0.1                        # KL penalty coefficient
learning_rate: 5e-7              # Very conservative
```

### Expected Performance
| GPU | Precision | Pretraining (100K steps) | GPU Utilization |
|-----|-----------|--------------------------|-----------------|
| A100 80GB | BF16 | ~45-50 hours | 95-100% |
| H100 80GB | BF16 | ~35-40 hours | 95-100% |
| H100 80GB | FP8 | ~25-30 hours | 95-100% |

## Directory Structure
```
checkpoints/      # Model checkpoints (init, pretrain, sft, dpo, lora)
configs/          # YAML configs + model architecture + tokenizer
data/             # raw/, packed/, sft/, dpo/ datasets
logs/             # TensorBoard logs
scripts/          # Numbered pipeline scripts (01-12) + gpu_utils.py
```

## Data Format
- Pretraining: Packed `.npy` files with tokenized sequences
- SFT/DPO: HuggingFace datasets format (load_from_disk)

## Key Dependencies
- transformers, trl (SFTTrainer, SFTConfig, DPOTrainer, DPOConfig)
- accelerate (for FP8 training)
- transformer-engine (optional, for FP8 on H100)
- peft (LoRA)
- datasets
- torch with CUDA (PyTorch 2.x required for torch.compile)

## Monitoring
```bash
# GPU utilization
watch -n 2 nvidia-smi

# TensorBoard
tensorboard --logdir logs/production_pretrain

# Training logs
tail -f production_training.log
```
