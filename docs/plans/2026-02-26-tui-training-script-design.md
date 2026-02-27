# TUI Training Script Design

**Date**: 2026-02-26
**Status**: Approved
**Target**: RunPod H100 NVL 96GB

## Overview

Convert the Colab notebook (`notebooks/train_gpt_oss_coding_tui.ipynb`) into a standalone bash script that runs the full TUI training pipeline inside a `screen` session on RunPod. Tuned for H100 80GB+ GPUs with auto-resume on reconnect.

## Architecture

Two scripts:
- `scripts/start_tui_training.sh` — Wrapper that launches screen session + logging
- `scripts/run_tui_pipeline.sh` — Orchestrator that calls existing training scripts sequentially

### Flow

```
[Setup] HF login, env vars, GPU detection, disk check
   |
[Phase 1] Tool Calling SFT (13_train_lang_adapter.py)
   | Quality gate: avg loss (last 10 steps) < 2.0 [HARD]
   |
[Phase 1.5] Merge adapter (19_merge_adapter.py)
   | Quality gate: model size > 5GB + smoke test [HARD]
   |
[Phase 2] Agent SFT (14_train_core_agent.py --packing --eval_strategy no)
   | Quality gate: avg loss < 1.8 [HARD]
   |
[Phase 3] IPO (17_ipo_preference.py)
   | Quality gate: logs loss trajectory [SOFT]
   |
[Phase 4] GRPO (18_grpo_rl.py)
   | Quality gate: reward accuracy > 0.55 [SOFT]
   |
[Phase 5] Export (19_merge_adapter.py --export_formats hf gguf_q4)
   |
[Done] Summary report + timestamps
```

## H100 Balanced Batch Profile

| Phase | Batch Size | Grad Accum | Effective Batch | Seq Length | Max Steps | LR |
|-------|-----------|------------|-----------------|------------|-----------|-----|
| Tool Calling SFT | 8 | 6 | 48 | 8192 | -1 | 2e-5 |
| Agent SFT | 1 | 24 | 24 | 32768 | -1 | 3e-5 |
| IPO | 2 | 16 | 32 | 8192 | -1 | 5e-7 |
| GRPO | 1 | 16 | 16 | 65536 | 5000 | 1e-6 |

## Auto-Resume Logic

Each phase checks before running:
1. `checkpoints/<phase>/final/` exists → **skip** (phase complete)
2. `checkpoints/<phase>/checkpoint-*` exists → **resume** (`--resume_from_checkpoint latest`)
3. Neither exists → **run from scratch**

Special cases:
- Merged model: check `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` for Phase 1.5
- Export: check `checkpoints/gpt-oss-20b-coding-tui-export/` for Phase 5

## Data Download

Same datasets as the notebook. Each phase checks for data at its expected path. If missing, downloads and formats from HuggingFace Hub.

### Tool Calling SFT (`data/coding_tui/tool_calling/train`)
- `glaiveai/glaive-function-calling-v2` (113K)
- `Salesforce/xlam-function-calling-60k` (60K)
- `NousResearch/hermes-function-calling-v1`
- Formatted via `dataset_formatters/function_calling` to Harmony format

### Agent SFT (`data/coding_tui/agent_traj/train`)
- `xingyaoww/code-act` (splits: codeact, general)
- `bigcode/commitpackft` (languages: python, javascript, go, rust, java, typescript)
- `nuprl/EditPackFT` (split: train)
- Formatted to Harmony agent format with CODING_AGENT_DEV_PROMPT

### IPO Preferences (`data/coding_tui/preference/train`)
- `Anthropic/hh-rlhf` (split: train)
- `m-a-p/CodeFeedback-Filtered-Instruction` (split: train)
- Formatted as pref_prompt/pref_chosen/pref_rejected columns

### GRPO Tasks
- Uses `--reward_mode coding_tui` with built-in reward signals
- No separate dataset download needed (generates from model)

Supports `HF_TOKEN` env var for gated datasets. Script prompts for `huggingface-cli login` if token not found.

## Screen Integration

`start_tui_training.sh`:
```bash
screen -dmS tui_training bash -c './run_tui_pipeline.sh 2>&1 | tee logs/tui_pipeline_<timestamp>.log'
```

Usage:
- Start: `bash scripts/start_tui_training.sh`
- Attach: `screen -r tui_training`
- Detach: `Ctrl+A, D`

## Quality Gates

Hard gates (block progression):
- Tool Calling SFT: avg loss (last 10 steps from trainer_state.json) < 2.0
- Merge: model directory > 5GB
- Agent SFT: avg loss (last 10 steps) < 1.8

Soft gates (warn but continue):
- IPO: log final loss for review
- GRPO: reward accuracy > 0.55 (warn if below)

Gate results saved to `logs/quality_gates.json`.

## Checkpoint Paths

| Phase | Checkpoint Dir |
|-------|---------------|
| Tool Calling SFT | `checkpoints/tool_calling_sft/final/` |
| Merged Model | `checkpoints/gpt-oss-20b-coding-tui-merged/hf/` |
| Agent SFT | `checkpoints/agent_sft/final/` |
| IPO | `checkpoints/agent_sft_ipo/final/` |
| GRPO | `checkpoints/agent_sft_grpo/final/` |
| Export | `checkpoints/gpt-oss-20b-coding-tui-export/` |

## Environment Requirements

- NVIDIA H100 (or A100 with reduced batch sizes)
- Python 3.11+, PyTorch 2.x, Unsloth, TRL <= 0.24.0
- `HF_TOKEN` env var (or `huggingface-cli login`)
- `screen` installed
- 200GB+ storage on /workspace

## Files to Create

1. `scripts/run_tui_pipeline.sh` — Main orchestrator
2. `scripts/start_tui_training.sh` — Screen wrapper
3. `configs/tui_pipeline.yaml` — H100 batch profile + dataset IDs (optional, can be inline)
