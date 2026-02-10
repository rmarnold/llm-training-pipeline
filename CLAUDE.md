# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production LLM training pipeline for a 7B parameter model (LLaMA-style architecture) targeting NVIDIA A100/H100 80GB GPUs. Implements staged training: Pretraining → SFT → DPO → LoRA, with promotion gates between stages.

## Commands

```bash
# Install
pip install -e .              # Core deps (see pyproject.toml for extras: fp8, kernels, dev)

# Full pipeline
bash scripts/run_full_pipeline.sh

# Individual stages: scripts are numbered 01-12 in execution order
python scripts/05_pretrain.py [--fp8|--no-fp8] [--enable-oom-recovery]
python scripts/07_sft.py [--fp8|--no-fp8]      # sequence packing enabled
python scripts/09_dpo.py [--fp8|--no-fp8]

# Resume from latest checkpoint
bash scripts/resume_pipeline.sh pretrain|sft|dpo

# Tests
pytest tests/ -v
pytest tests/test_configs.py -v    # single file
```

## Architecture

### Training Stages Flow
```
Data (01-03) → Model Init (04) → Pretrain (05) → SFT (06-07) → DPO (08-09) → LoRA (10)
                                      ↓               ↓              ↓
                                  [gates]          [gates]        [gates]
```

Promotion gates (`configs/promotion_gates.yaml`) define thresholds for advancing between stages (perplexity, accuracy, safety refusal rate). Checked via `scripts/12_check_gates.py`.

### Model
- 7B params, 32 layers, 4096 hidden dim, GQA (8 KV heads), vocab 50,304
- Curriculum learning: progressive context length (512 → 1024 → 2048)
- Architecture config: `configs/model_7b.py` (`ModelConfig` dataclass)

### Key Patterns & Gotchas
- **GPU auto-detection** (`scripts/gpu_utils.py`): All training scripts use `detect_gpu_type()` → `GPUInfo` TypedDict. H100 gets FP8 + `max-autotune`; A100 gets BF16 + `default` compile mode. Also provides `OOMHandler` and `setup_torch_backends()`.
- **Kernel optimizations** must be called **before** model loading (`setup_kernel_optimizations` in `05_pretrain.py`) — they patch model classes. Liger Kernel and CCE are mutually exclusive.
- **Liger's `fused_linear_cross_entropy=False`** — `.item()` call breaks `torch.compile`
- **`unwrap_compiled_model()`** required before saving `torch.compile` wrapped models (accesses `_orig_mod`)
- Training scripts import `gpu_utils` via bare import; `tests/conftest.py` adds `scripts/` to `sys.path`
- wandb defaults to offline mode (`WANDB_MODE=offline`) unless explicitly overridden
- **Deprecated**: `production_pretrain.py` and `full_model_pretrain.py` — use `05_pretrain.py` instead

### Data Formats
- **Pretraining**: Packed `.npy` (shape: `N x seq_length`, int64) or HF Datasets with `input_ids`
- **SFT**: HF Dataset with `messages` (role/content dicts) or `text`
- **DPO**: HF Dataset with `prompt`, `chosen`, `rejected` (strings)
- Full spec: `docs/DATA_FORMATS.md`
