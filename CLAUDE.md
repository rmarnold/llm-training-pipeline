# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production LLM training pipeline for a 7B parameter model (LLaMA-style architecture) targeting NVIDIA A100/H100 80GB GPUs. Implements staged training: Pretraining → SFT → DPO → LoRA, with promotion gates between stages.

## Commands

```bash
# Install
pip install -e .              # Core deps (see pyproject.toml for extras: fp8, kernels, dev)
pip install -e ".[gpt_oss,rust_eval]"  # GPT-OSS 20B Rust agent pipeline

# Full pipeline (7B)
bash scripts/run_full_pipeline.sh

# Individual stages: scripts are numbered 01-12 in execution order
python scripts/05_pretrain.py [--fp8|--no-fp8] [--enable-oom-recovery]
python scripts/07_sft.py [--fp8|--no-fp8]      # sequence packing enabled
python scripts/09_dpo.py [--fp8|--no-fp8]

# GPT-OSS 20B Rust Agent: scripts 13-19
python scripts/16_generate_mutations.py          # Generate mutation data
python scripts/15_generate_trajectories.py       # Generate agent trajectories
python scripts/13_train_lang_adapter.py          # Train lang_rust adapter
python scripts/19_merge_adapter.py               # Merge adapter into base
python scripts/14_train_core_agent.py            # Train core agent SFT
python scripts/17_ipo_preference.py              # IPO preference training
python scripts/18_grpo_rl.py                     # GRPO RL training
python scripts/eval_rust_agent.py                # Evaluate Rust agent

# Resume from latest checkpoint
bash scripts/resume_pipeline.sh pretrain|sft|dpo

# Tests
pytest tests/ -v
pytest tests/test_configs.py -v    # single file
```

## Architecture

### Training Stages Flow
```
7B Pipeline:
Data (01-03) → Model Init (04) → Pretrain (05) → SFT (06-07) → DPO (08-09) → LoRA (10)
                                      ↓               ↓              ↓
                                  [gates]          [gates]        [gates]

GPT-OSS 20B Rust Agent Pipeline:
MutationGen (16) → TrajectoryGen (15) → LangAdapter (13) → Merge (19)
                                              ↓
                                        CoreAgent (14) → IPO (17) → GRPO (18) → Eval
                                                           ↓          ↓
                                                       [gates]    [gates]
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

### GPT-OSS 20B Rust Agent (scripts 13-19)
- **Base model**: GPT-OSS 20B MoE (~3.6B active params, 32 experts/layer, top-4 routing), architecture in `configs/gpt_oss_20b.py`
- **Unsloth** for QLoRA training; all scripts use `pipeline_lib/unsloth_utils.py`
- **Harmony format** (`dataset_formatters/harmony.py`): Required for all GPT-OSS training data. Supports tool calls, thinking fields, multi-turn agent traces.
- **Config-first**: All scripts load YAML configs and accept CLI overrides (e.g. `--max_steps`, `--per_device_train_batch_size`)
- **Rust toolchain required**: `cargo`, `cargo-mutants` for mutation generation and evaluation
- **Evaluation metrics**: `cargo_check_pass_rate`, `cargo_test_pass_rate`, `clippy_clean_rate` with targets in `configs/rust_eval.yaml`
- **Notebook**: `notebooks/train_gpt_oss_coding_tui.ipynb` — Colab companion with GPU tier auto-config, MoE diagnostic, quality gates
- **Coding TUI notebook**: `notebooks/train_gpt_oss_coding_tui.ipynb` — 4-phase pipeline (Tool Calling SFT → Agent SFT → IPO → GRPO) with non-blocking quality gates
- **No FA2 for GPT-OSS**: Unsloth's GPT-OSS patches use eager attention only (no Flash Attention 2 path). In-training eval OOMs on seq_len>=8192 — use `--eval_strategy no` for Agent SFT. Quality gate validates loss from `trainer_state.json` instead.

### MoE Expert LoRA (Critical for GPT-OSS 20B)
- **Unsloth Bug #3405**: Default target modules (`gate_proj`, `up_proj`, `down_proj`) silently miss MoE expert FFN layers. Only attention gets LoRA (~31.8M params), experts (~19B params) stay frozen.
- **Fix**: Use singular names (`gate_up_proj`, `down_proj`) which Unsloth maps to fused expert layers. `apply_lora_config()` auto-detects MoE and corrects targets at runtime.
- **Unsloth Bug #3701**: Save validation fails when expert modules are targeted. `save_adapter()` catches this and falls back to PEFT's native `save_pretrained()`.
- **Diagnostic**: Notebook cell 26 loads model, applies test LoRA, verifies expert layers have adapters before training starts.
- **Verify with**: `verify_expert_lora(model)` — returns `has_expert_lora: True` if working correctly. Expect trainable params to jump from ~31.8M to ~200M+.

### Data Formats
- **Pretraining**: Packed `.npy` (shape: `N x seq_length`, int64) or HF Datasets with `input_ids`
- **SFT**: HF Dataset with `messages` (role/content dicts) or `text`
- **DPO**: HF Dataset with `prompt`, `chosen`, `rejected` (strings)
- Full spec: `docs/DATA_FORMATS.md`
