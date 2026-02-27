# TUI Training Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone bash script that runs the full 4-phase TUI training pipeline on RunPod H100, inside a screen session with auto-resume.

**Architecture:** Two bash scripts — `start_tui_training.sh` (screen wrapper) and `run_tui_pipeline.sh` (orchestrator). The orchestrator calls existing Python training scripts (13, 14, 17, 18, 19) with H100-tuned CLI overrides. A Python helper `tui_data_prep.py` handles HuggingFace dataset download and Harmony formatting. Auto-resume detects existing checkpoints to skip completed phases.

**Tech Stack:** Bash, screen, existing Python training scripts, HuggingFace datasets library, Harmony formatters

---

### Task 1: Create the data preparation script

**Files:**
- Create: `scripts/tui_data_prep.py`
- Reference: `notebooks/train_gpt_oss_coding_tui.ipynb` (cells 17, 19, 21 for dataset IDs and formatters)

**Step 1: Create `scripts/tui_data_prep.py`**

This script downloads and formats all training data from HuggingFace. It mirrors the notebook's data download cells exactly. Each phase's data is prepared independently and skipped if already present.

```python
#!/usr/bin/env python3
"""Download and format TUI training datasets from HuggingFace.

Mirrors the data preparation from train_gpt_oss_coding_tui.ipynb.
Skips datasets that already exist at their output paths.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from datasets import load_dataset
from dataset_formatters.harmony import encode_harmony_messages


def prepare_tool_calling(output_dir="data/coding_tui/tool_calling/train", quick_test=False):
    """Download and format tool calling datasets: Glaive v2, xLAM-60k, Hermes v1."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Tool calling data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    from dataset_formatters.function_calling import (
        format_glaive_function_calling,
        format_hermes_function_calling,
    )

    all_examples = []

    # 1. Glaive v2 (113K)
    print("Downloading glaiveai/glaive-function-calling-v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    limit = 500 if quick_test else len(ds)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = format_glaive_function_calling(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  Glaive: {len(all_examples)} examples")

    # 2. xLAM-60K
    print("Downloading Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    limit = 200 if quick_test else len(ds)
    count_before = len(all_examples)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_xlam(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  xLAM: {len(all_examples) - count_before} examples")

    # 3. Hermes v1
    print("Downloading NousResearch/hermes-function-calling-v1...")
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    limit = 300 if quick_test else len(ds)
    count_before = len(all_examples)
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = format_hermes_function_calling(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  Hermes: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir)
    print(f"Total tool calling examples: {len(all_examples)}")


def prepare_agent_trajectories(output_dir="data/coding_tui/agent_traj/train", quick_test=False):
    """Download and format agent trajectory datasets: code-act, commitpackft, EditPackFT."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Agent trajectory data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_examples = []

    CODING_AGENT_DEV_PROMPT = (
        "You are a coding agent. Use tools to read files, write code, run tests, and "
        "complete programming tasks. Do not just analyze - always take action and produce "
        "working code. After making changes, verify they work by running the relevant tests. "
        "If a tool call fails, diagnose and retry with corrected parameters."
    )

    # 1. code-act
    print("Downloading xingyaoww/code-act...")
    for split_name in ("codeact", "general"):
        try:
            ds = load_dataset("xingyaoww/code-act", split=split_name)
            limit = 100 if quick_test else len(ds)
            for ex in ds.select(range(min(limit, len(ds)))):
                formatted = _format_code_act(ex, CODING_AGENT_DEV_PROMPT)
                if formatted:
                    all_examples.append(formatted)
        except Exception as e:
            print(f"  Warning: code-act split '{split_name}' failed: {e}")
    print(f"  code-act: {len(all_examples)} examples")

    # 2. commitpackft (multi-language)
    print("Downloading bigcode/commitpackft...")
    languages = ["python", "javascript", "go", "rust", "java", "typescript"]
    count_before = len(all_examples)
    for lang in languages:
        try:
            ds = load_dataset("bigcode/commitpackft", lang, split="train")
            limit = 50 if quick_test else min(5000, len(ds))
            for ex in ds.select(range(min(limit, len(ds)))):
                formatted = _format_commitpackft(ex)
                if formatted:
                    all_examples.append(formatted)
        except Exception as e:
            print(f"  Warning: commitpackft/{lang} failed: {e}")
    print(f"  commitpackft: {len(all_examples) - count_before} examples")

    # 3. EditPackFT
    print("Downloading nuprl/EditPackFT...")
    count_before = len(all_examples)
    try:
        ds = load_dataset("nuprl/EditPackFT", split="train")
        limit = 300 if quick_test else min(10000, len(ds))
        for ex in ds.select(range(min(limit, len(ds)))):
            formatted = _format_editpackft(ex)
            if formatted:
                all_examples.append(formatted)
    except Exception as e:
        print(f"  Warning: EditPackFT failed: {e}")
    print(f"  EditPackFT: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir)
    print(f"Total agent trajectory examples: {len(all_examples)}")


def prepare_preferences(output_dir="data/coding_tui/preference/train", quick_test=False):
    """Download and format preference datasets: hh-rlhf, CodeFeedback."""
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Preference data already exists at {output_dir}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    all_examples = []

    # 1. hh-rlhf
    print("Downloading Anthropic/hh-rlhf...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    limit = 300 if quick_test else min(20000, len(ds))
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_hh_rlhf(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  hh-rlhf: {len(all_examples)} examples")

    # 2. CodeFeedback
    print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
    count_before = len(all_examples)
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    limit = 200 if quick_test else min(10000, len(ds))
    for ex in ds.select(range(min(limit, len(ds)))):
        formatted = _format_code_feedback(ex)
        if formatted:
            all_examples.append(formatted)
    print(f"  CodeFeedback: {len(all_examples) - count_before} examples")

    _save_dataset(all_examples, output_dir, columns=["text", "pref_prompt", "pref_chosen", "pref_rejected"])
    print(f"Total preference examples: {len(all_examples)}")


# --- Internal formatters (mirror notebook logic exactly) ---

def _format_xlam(ex):
    """Format xLAM function calling example to Harmony."""
    import json
    try:
        tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
        answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
    except (json.JSONDecodeError, KeyError):
        return None
    if not tools or not answers:
        return None

    tool_desc = "\n".join(
        f"- {t.get('name', 'unknown')}: {t.get('description', '')}"
        for t in tools
    )
    tool_calls = []
    for ans in answers:
        tool_calls.append({
            "name": ans.get("name", "unknown"),
            "arguments": json.dumps(ans.get("arguments", {})),
        })

    messages = [
        {"role": "developer", "content": f"Available tools:\n{tool_desc}"},
        {"role": "user", "content": ex["query"]},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
    ]
    return {"text": encode_harmony_messages(messages)}


def _format_code_act(ex, dev_prompt):
    """Format code-act example to Harmony."""
    convos = ex.get("conversations", [])
    if not convos:
        return None
    messages = [{"role": "developer", "content": dev_prompt}]
    for turn in convos:
        role_raw = turn.get("role") or turn.get("from", "")
        content = turn.get("content") or turn.get("value", "")
        if role_raw in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role_raw in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif role_raw in ("tool", "function", "observation"):
            messages.append({"role": "tool", "content": content})
    if len(messages) < 3:
        return None
    return {"text": encode_harmony_messages(messages, reasoning_effort="high")}


def _format_commitpackft(ex):
    """Format commitpackft example to Harmony."""
    old_code = ex.get("old_contents", "")
    new_code = ex.get("new_contents", "")
    msg = ex.get("message", "") or ex.get("subject", "")
    if not old_code or not new_code or old_code == new_code:
        return None
    diff_len = len(new_code) - len(old_code)
    if abs(diff_len) > 8000:
        return None
    messages = [
        {"role": "user", "content": f"Apply this change: {msg}\n\nCurrent code:\n```\n{old_code}\n```"},
        {"role": "assistant", "content": f"```\n{new_code}\n```"},
    ]
    return {"text": encode_harmony_messages(messages, reasoning_effort="medium")}


def _format_editpackft(ex):
    """Format EditPackFT example to Harmony."""
    instruction = ex.get("instruction", "")
    old_code = ex.get("old_contents") or ex.get("input", "")
    new_code = ex.get("new_contents") or ex.get("output", "")
    if not instruction or not old_code or not new_code or old_code == new_code:
        return None
    if abs(len(new_code) - len(old_code)) > 8000:
        return None
    messages = [
        {"role": "user", "content": f"{instruction}\n\nCurrent code:\n```\n{old_code}\n```"},
        {"role": "assistant", "content": f"```\n{new_code}\n```"},
    ]
    return {"text": encode_harmony_messages(messages, reasoning_effort="medium")}


def _format_hh_rlhf(ex):
    """Format hh-rlhf example to Harmony preference pair."""
    chosen_raw = ex.get("chosen", "")
    rejected_raw = ex.get("rejected", "")
    if not chosen_raw or not rejected_raw:
        return None

    def _parse_hh(text):
        messages = []
        parts = text.split("\n\nHuman: ")
        for part in parts:
            if not part.strip():
                continue
            if "\n\nAssistant: " in part:
                human_part, assistant_part = part.split("\n\nAssistant: ", 1)
                if human_part.strip():
                    messages.append({"role": "user", "content": human_part.strip()})
                if assistant_part.strip():
                    messages.append({"role": "assistant", "content": assistant_part.strip()})
            else:
                messages.append({"role": "user", "content": part.strip()})
        return messages

    chosen_msgs = _parse_hh(chosen_raw)
    rejected_msgs = _parse_hh(rejected_raw)
    if len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
        return None

    prompt_msgs = chosen_msgs[:-1]
    prompt_text = encode_harmony_messages(prompt_msgs)
    chosen_text = encode_harmony_messages(chosen_msgs)
    rejected_text = encode_harmony_messages(rejected_msgs)

    return {
        "text": chosen_text,
        "pref_prompt": prompt_text,
        "pref_chosen": chosen_text,
        "pref_rejected": rejected_text,
    }


def _format_code_feedback(ex):
    """Format CodeFeedback example to Harmony preference pair (synthetic rejected)."""
    query = ex.get("query", "")
    answer = ex.get("answer", "")
    if not query or not answer or len(answer) < 100:
        return None

    messages_chosen = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    # Synthetic rejected: truncate at 30% + stalling response
    truncated = answer[:int(len(answer) * 0.3)]
    messages_rejected = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": truncated + " I would need to analyze this further before proceeding."},
    ]
    prompt_msgs = [{"role": "user", "content": query}]

    prompt_text = encode_harmony_messages(prompt_msgs)
    chosen_text = encode_harmony_messages(messages_chosen)
    rejected_text = encode_harmony_messages(messages_rejected)

    return {
        "text": chosen_text,
        "pref_prompt": prompt_text,
        "pref_chosen": chosen_text,
        "pref_rejected": rejected_text,
    }


def _save_dataset(examples, output_dir, columns=None):
    """Save formatted examples as HuggingFace dataset."""
    import random
    from datasets import Dataset

    random.seed(42)
    random.shuffle(examples)

    if columns:
        # Filter to only include specified columns
        examples = [{k: ex[k] for k in columns if k in ex} for ex in examples]

    ds = Dataset.from_list(examples)
    ds.save_to_disk(output_dir)
    print(f"Saved {len(ds)} examples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and format TUI training data")
    parser.add_argument("--quick-test", action="store_true", help="Download small subset for testing")
    parser.add_argument("--phase", choices=["all", "tool_calling", "agent_traj", "preference"],
                        default="all", help="Which phase data to prepare")
    args = parser.parse_args()

    print("=" * 60)
    print("TUI Training Data Preparation")
    print(f"Mode: {'quick test' if args.quick_test else 'full'}")
    print(f"Phase: {args.phase}")
    print("=" * 60)

    if args.phase in ("all", "tool_calling"):
        prepare_tool_calling(quick_test=args.quick_test)
    if args.phase in ("all", "agent_traj"):
        prepare_agent_trajectories(quick_test=args.quick_test)
    if args.phase in ("all", "preference"):
        prepare_preferences(quick_test=args.quick_test)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `python -c "import ast; ast.parse(open('scripts/tui_data_prep.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/tui_data_prep.py
git commit -m "Add TUI data preparation script for RunPod training"
```

---

### Task 2: Create the main pipeline orchestrator

**Files:**
- Create: `scripts/run_tui_pipeline.sh`
- Reference: `notebooks/train_gpt_oss_coding_tui.ipynb` (training cells 32, 34, 40, 45, 50, 69)
- Reference: `docs/plans/2026-02-26-tui-training-script-design.md`

**Step 1: Create `scripts/run_tui_pipeline.sh`**

This is the main orchestrator. It runs all 6 phases sequentially with H100-tuned parameters, auto-resume logic, and quality gates. Each phase is a function for clarity.

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TUI Training Pipeline — H100 Optimized
# =============================================================================
# Full 4-phase pipeline: Tool Calling SFT -> Merge -> Agent SFT -> IPO -> GRPO -> Export
# Auto-resumes from last completed phase on restart.
#
# Usage:
#   bash scripts/run_tui_pipeline.sh [--quick-test] [--skip-data] [--batch-profile aggressive|balanced|conservative]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# --- CLI args ---
QUICK_TEST=false
SKIP_DATA=false
BATCH_PROFILE="balanced"
for arg in "$@"; do
    case "$arg" in
        --quick-test) QUICK_TEST=true ;;
        --skip-data) SKIP_DATA=true ;;
        --batch-profile=*) BATCH_PROFILE="${arg#*=}" ;;
    esac
done

# --- Logging ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tui_pipeline_${TIMESTAMP}.log"
GATE_FILE="$LOG_DIR/quality_gates.json"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_section() { echo "" | tee -a "$LOG_FILE"; echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"; log "$*"; echo "$(printf '=%.0s' {1..70})" | tee -a "$LOG_FILE"; }

# --- Environment ---
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# HuggingFace token (RunPod secret)
if [ -n "${RUNPOD_SECRET_HF_TOKEN:-}" ]; then
    export HF_TOKEN="$RUNPOD_SECRET_HF_TOKEN"
    log "HF_TOKEN set from RUNPOD_SECRET_HF_TOKEN"
elif [ -n "${HF_TOKEN:-}" ]; then
    log "HF_TOKEN already set"
else
    log "WARNING: No HF_TOKEN found. Some datasets may fail to download."
    log "Set RUNPOD_SECRET_HF_TOKEN in RunPod secrets or export HF_TOKEN."
fi

# --- H100 Batch Profiles ---
case "$BATCH_PROFILE" in
    aggressive)
        TC_BATCH=10; TC_GRAD=5      # eff=50
        AGENT_BATCH=4; AGENT_GRAD=6  # eff=24
        IPO_BATCH=4; IPO_GRAD=8      # eff=32
        GRPO_BATCH=2; GRPO_GRAD=8    # eff=16
        ;;
    balanced)
        TC_BATCH=8; TC_GRAD=6        # eff=48
        AGENT_BATCH=1; AGENT_GRAD=24  # eff=24
        IPO_BATCH=2; IPO_GRAD=16      # eff=32
        GRPO_BATCH=1; GRPO_GRAD=16    # eff=16
        ;;
    conservative)
        TC_BATCH=4; TC_GRAD=12       # eff=48
        AGENT_BATCH=1; AGENT_GRAD=24  # eff=24
        IPO_BATCH=1; IPO_GRAD=32      # eff=32
        GRPO_BATCH=1; GRPO_GRAD=16    # eff=16
        ;;
    *)
        log "ERROR: Unknown batch profile: $BATCH_PROFILE"
        exit 1
        ;;
esac

# --- H100 Sequence Lengths ---
TC_SEQ_LEN=8192
AGENT_SEQ_LEN=32768
IPO_SEQ_LEN=8192
GRPO_SEQ_LEN=65536
GRPO_NUM_GEN=4
GRPO_MAX_STEPS=5000

# --- Quick test overrides ---
if [ "$QUICK_TEST" = true ]; then
    TC_BATCH=2; TC_GRAD=2
    AGENT_BATCH=1; AGENT_GRAD=2
    IPO_BATCH=1; IPO_GRAD=2
    GRPO_BATCH=1; GRPO_GRAD=2
    GRPO_MAX_STEPS=50
    TC_SEQ_LEN=4096
    AGENT_SEQ_LEN=4096
    GRPO_SEQ_LEN=4096
fi

# --- Checkpoint paths ---
TC_FINAL="checkpoints/tool_calling_sft/final"
MERGED_MODEL="checkpoints/gpt-oss-20b-coding-tui-merged/hf"
AGENT_FINAL="checkpoints/agent_sft/final"
IPO_FINAL="checkpoints/agent_sft_ipo/final"
GRPO_FINAL="checkpoints/agent_sft_grpo/final"
EXPORT_DIR="checkpoints/gpt-oss-20b-coding-tui-export"

# --- Quality gate state ---
echo '{}' > "$GATE_FILE"

update_gate() {
    # Usage: update_gate <phase> <key> <value>
    python3 -c "
import json, sys
with open('$GATE_FILE') as f: data = json.load(f)
phase, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
if phase not in data: data[phase] = {}
try: val = json.loads(val)
except: pass
data[phase][key] = val
with open('$GATE_FILE', 'w') as f: json.dump(data, f, indent=2)
" "$1" "$2" "$3"
}

# --- Helper: get avg loss from trainer_state.json ---
get_avg_loss() {
    local ckpt_dir="$1"
    local n="${2:-10}"
    python3 -c "
import json, glob, os
state_files = sorted(glob.glob('${ckpt_dir}/*/trainer_state.json')) + \
              ([f'${ckpt_dir}/trainer_state.json'] if os.path.exists('${ckpt_dir}/trainer_state.json') else [])
if not state_files:
    # Check final dir
    final_state = '${ckpt_dir}/final/trainer_state.json'
    if os.path.exists(final_state):
        state_files = [final_state]
if not state_files:
    print('-1')
else:
    with open(state_files[-1]) as f:
        state = json.load(f)
    losses = [e['loss'] for e in state.get('log_history', []) if 'loss' in e]
    if losses:
        avg = sum(losses[-${n}:]) / min(len(losses), ${n})
        print(f'{avg:.4f}')
    else:
        print('-1')
"
}

# --- Helper: get directory size in GB ---
dir_size_gb() {
    python3 -c "
import os
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk('$1') for f in fns)
print(f'{total / 1024**3:.2f}')
"
}

# --- Helper: find latest checkpoint for resume ---
find_latest_checkpoint() {
    local base_dir="$1"
    python3 -c "
import glob, os
ckpts = sorted(glob.glob('${base_dir}/checkpoint-*'), key=lambda p: int(p.split('-')[-1]))
print(ckpts[-1] if ckpts else '')
"
}

# =============================================================================
# Phase 0: Data Preparation
# =============================================================================
phase_data_prep() {
    log_section "PHASE 0: Data Preparation"

    if [ "$SKIP_DATA" = true ]; then
        log "Skipping data preparation (--skip-data)"
        return 0
    fi

    local data_args=""
    if [ "$QUICK_TEST" = true ]; then
        data_args="--quick-test"
    fi

    python3 scripts/tui_data_prep.py $data_args 2>&1 | tee -a "$LOG_FILE"
    log "Data preparation complete."
}

# =============================================================================
# Phase 1: Tool Calling SFT
# =============================================================================
phase_tool_calling_sft() {
    log_section "PHASE 1: Tool Calling SFT (LoRA rank 64)"

    if [ -f "$TC_FINAL/adapter_config.json" ]; then
        log "Tool calling SFT already complete (final/ exists). Skipping."
        update_gate "tool_calling" "status" '"skipped"'
        return 0
    fi

    local resume_arg=""
    local latest=$(find_latest_checkpoint "checkpoints/tool_calling_sft")
    if [ -n "$latest" ]; then
        resume_arg="--resume_from_checkpoint $latest"
        log "Resuming from $latest"
    fi

    local max_steps_arg="-1"
    if [ "$QUICK_TEST" = true ]; then max_steps_arg="50"; fi

    log "Batch: ${TC_BATCH} x ${TC_GRAD} = $((TC_BATCH * TC_GRAD))"
    log "Seq length: $TC_SEQ_LEN"

    python3 scripts/13_train_lang_adapter.py \
        --train_data_path data/coding_tui/tool_calling/train \
        --per_device_train_batch_size "$TC_BATCH" \
        --gradient_accumulation_steps "$TC_GRAD" \
        --max_steps "$max_steps_arg" \
        --num_train_epochs 1 \
        --output_dir checkpoints/tool_calling_sft \
        --save_steps 250 \
        $resume_arg \
        2>&1 | tee -a "$LOG_FILE"

    # Quality gate
    local avg_loss
    avg_loss=$(get_avg_loss "checkpoints/tool_calling_sft")
    log "Tool Calling SFT avg loss (last 10): $avg_loss"
    update_gate "tool_calling" "loss" "$avg_loss"

    if [ "$avg_loss" = "-1" ]; then
        log "ERROR: Could not read loss from trainer_state.json"
        update_gate "tool_calling" "passed" "false"
        return 1
    fi

    local passed
    passed=$(python3 -c "print('true' if float('$avg_loss') <= 2.0 else 'false')")
    update_gate "tool_calling" "passed" "$passed"

    if [ "$passed" = "false" ]; then
        log "FAILED: Tool calling loss $avg_loss > 2.0 threshold"
        return 1
    fi
    log "PASSED: Tool calling loss $avg_loss <= 2.0"
}

# =============================================================================
# Phase 1.5: Merge Tool Calling Adapter
# =============================================================================
phase_merge() {
    log_section "PHASE 1.5: Merge Tool Calling Adapter"

    if [ -f "$MERGED_MODEL/config.json" ]; then
        # Validate existing merge
        local size_gb
        size_gb=$(dir_size_gb "$MERGED_MODEL")
        local valid
        valid=$(python3 -c "print('true' if float('$size_gb') >= 5.0 else 'false')")

        if [ "$valid" = "true" ]; then
            log "Merged model already exists and valid ($size_gb GB). Skipping."
            update_gate "merge" "status" '"skipped"'
            update_gate "merge" "size_gb" "$size_gb"
            return 0
        else
            log "Merged model invalid ($size_gb GB < 5.0 GB). Re-merging."
            rm -rf "checkpoints/gpt-oss-20b-coding-tui-merged"
        fi
    fi

    if [ ! -f "$TC_FINAL/adapter_config.json" ]; then
        log "ERROR: Tool calling adapter not found at $TC_FINAL"
        return 1
    fi

    python3 scripts/19_merge_adapter.py \
        --adapter_path "$TC_FINAL" \
        --output_dir checkpoints/gpt-oss-20b-coding-tui-merged \
        2>&1 | tee -a "$LOG_FILE"

    # Quality gate: size check
    local size_gb
    size_gb=$(dir_size_gb "$MERGED_MODEL")
    log "Merged model size: $size_gb GB"
    update_gate "merge" "size_gb" "$size_gb"

    local passed
    passed=$(python3 -c "print('true' if float('$size_gb') >= 5.0 else 'false')")
    update_gate "merge" "passed" "$passed"

    if [ "$passed" = "false" ]; then
        log "FAILED: Merged model $size_gb GB < 5.0 GB threshold"
        return 1
    fi
    log "PASSED: Merged model $size_gb GB"
}

# =============================================================================
# Phase 2: Agent SFT
# =============================================================================
phase_agent_sft() {
    log_section "PHASE 2: Agent SFT (LoRA rank 128)"

    if [ -f "$AGENT_FINAL/adapter_config.json" ]; then
        log "Agent SFT already complete (final/ exists). Skipping."
        update_gate "agent_sft" "status" '"skipped"'
        return 0
    fi

    # Determine base model
    local base_model="openai/gpt-oss-20b"
    if [ -f "$MERGED_MODEL/config.json" ]; then
        base_model="$MERGED_MODEL"
        log "Using merged tool-calling model as base."
    else
        log "WARNING: Merged model not found, using raw base model."
    fi

    # Pre-pack dataset
    local raw_data="data/coding_tui/agent_traj/train"
    local packed_data="data/coding_tui/agent_traj/train_packed_${AGENT_SEQ_LEN}"
    local train_data="$raw_data"
    local val_data=""
    local pack_args=""

    if [ -d "${packed_data}/train" ]; then
        log "Pre-packed dataset exists: $packed_data"
        train_data="${packed_data}/train"
        val_data="${packed_data}/val"
        pack_args="--pre_packed"
    else
        log "Pre-packing agent trajectory dataset (seq_len=$AGENT_SEQ_LEN)..."
        python3 -c "
import sys
sys.path.insert(0, 'scripts')
from pipeline_lib.pack_sft import pack_sft_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-20b', trust_remote_code=True)
result = pack_sft_dataset(
    dataset_path='$raw_data',
    tokenizer=tokenizer,
    seq_len=$AGENT_SEQ_LEN,
    output_path='$packed_data',
)
if 'error' in result:
    print(f'ERROR: {result}', file=sys.stderr)
    sys.exit(1)
print(f'Packed: {result}')
" 2>&1 | tee -a "$LOG_FILE"

        if [ -d "${packed_data}/train" ]; then
            train_data="${packed_data}/train"
            val_data="${packed_data}/val"
            pack_args="--pre_packed"
        fi
    fi

    local resume_arg=""
    local latest=$(find_latest_checkpoint "checkpoints/agent_sft")
    if [ -n "$latest" ]; then
        resume_arg="--resume_from_checkpoint $latest"
        log "Resuming from $latest"
    fi

    local max_steps_arg="-1"
    if [ "$QUICK_TEST" = true ]; then max_steps_arg="50"; fi

    local val_arg=""
    if [ -n "$val_data" ] && [ -d "$val_data" ]; then
        val_arg="--val_data_path $val_data"
    fi

    log "Base: $base_model"
    log "Data: $train_data"
    log "Batch: ${AGENT_BATCH} x ${AGENT_GRAD} = $((AGENT_BATCH * AGENT_GRAD))"
    log "Seq length: $AGENT_SEQ_LEN"

    python3 scripts/14_train_core_agent.py \
        --base_model "$base_model" \
        --no-merge-lang-adapter \
        --train_data_path "$train_data" \
        $val_arg \
        $pack_args \
        --per_device_train_batch_size "$AGENT_BATCH" \
        --gradient_accumulation_steps "$AGENT_GRAD" \
        --max_steps "$max_steps_arg" \
        --num_train_epochs 3 \
        --output_dir checkpoints/agent_sft \
        --save_steps 250 \
        --eval_strategy no \
        --max_seq_length "$AGENT_SEQ_LEN" \
        $resume_arg \
        2>&1 | tee -a "$LOG_FILE"

    # Quality gate
    local avg_loss
    avg_loss=$(get_avg_loss "checkpoints/agent_sft")
    log "Agent SFT avg loss (last 10): $avg_loss"
    update_gate "agent_sft" "loss" "$avg_loss"

    local passed
    passed=$(python3 -c "print('true' if float('$avg_loss') <= 1.8 and float('$avg_loss') > 0 else 'false')")
    update_gate "agent_sft" "passed" "$passed"

    if [ "$passed" = "false" ]; then
        log "FAILED: Agent SFT loss $avg_loss > 1.8 threshold"
        return 1
    fi
    log "PASSED: Agent SFT loss $avg_loss <= 1.8"
}

# =============================================================================
# Phase 3: IPO Preference Optimization
# =============================================================================
phase_ipo() {
    log_section "PHASE 3: IPO Preference Optimization"

    if [ -f "$IPO_FINAL/adapter_config.json" ]; then
        log "IPO already complete (final/ exists). Skipping."
        update_gate "ipo" "status" '"skipped"'
        return 0
    fi

    local pref_data="data/coding_tui/preference/train"
    if [ ! -d "$pref_data" ]; then
        log "WARNING: Preference data not found at $pref_data. Skipping IPO."
        update_gate "ipo" "status" '"skipped_no_data"'
        return 0
    fi

    # Determine base checkpoint
    local ipo_base=""
    if [ -f "$AGENT_FINAL/adapter_config.json" ]; then
        ipo_base="$AGENT_FINAL"
    elif [ -f "$TC_FINAL/adapter_config.json" ]; then
        ipo_base="$TC_FINAL"
        log "Agent SFT not found, falling back to tool_calling_sft"
    else
        log "ERROR: No checkpoint found for IPO."
        return 1
    fi

    local max_steps_arg="-1"
    if [ "$QUICK_TEST" = true ]; then max_steps_arg="50"; fi

    log "Base checkpoint: $ipo_base"
    log "Data: $pref_data"
    log "Batch: ${IPO_BATCH} x ${IPO_GRAD} = $((IPO_BATCH * IPO_GRAD))"

    python3 scripts/17_ipo_preference.py \
        --checkpoint "$ipo_base" \
        --train_data_path "$pref_data" \
        --per_device_train_batch_size "$IPO_BATCH" \
        --gradient_accumulation_steps "$IPO_GRAD" \
        --max_steps "$max_steps_arg" \
        --output_dir checkpoints/agent_sft_ipo \
        --beta 0.1 \
        2>&1 | tee -a "$LOG_FILE"

    # Soft quality gate
    local avg_loss
    avg_loss=$(get_avg_loss "checkpoints/agent_sft_ipo")
    log "IPO avg loss (last 10): $avg_loss"
    update_gate "ipo" "loss" "$avg_loss"
    update_gate "ipo" "passed" "true"
    log "IPO complete (soft gate — always passes)."
}

# =============================================================================
# Phase 4: GRPO RL
# =============================================================================
phase_grpo() {
    log_section "PHASE 4: GRPO RL Training"

    if [ -f "$GRPO_FINAL/adapter_config.json" ]; then
        log "GRPO already complete (final/ exists). Skipping."
        update_gate "grpo" "status" '"skipped"'
        return 0
    fi

    # Determine base checkpoint (priority: IPO > Agent SFT > Tool Calling > base)
    local grpo_base=""
    if [ -f "$IPO_FINAL/adapter_config.json" ]; then
        grpo_base="$IPO_FINAL"
    elif [ -f "$AGENT_FINAL/adapter_config.json" ]; then
        grpo_base="$AGENT_FINAL"
        log "IPO not found, using agent_sft."
    elif [ -f "$TC_FINAL/adapter_config.json" ]; then
        grpo_base="$TC_FINAL"
        log "Agent SFT not found, using tool_calling_sft."
    else
        grpo_base="openai/gpt-oss-20b"
        log "No checkpoint found, using base model."
    fi

    log "Base checkpoint: $grpo_base"
    log "Batch: ${GRPO_BATCH} x ${GRPO_GRAD} = $((GRPO_BATCH * GRPO_GRAD))"
    log "Seq length: $GRPO_SEQ_LEN"
    log "Max steps: $GRPO_MAX_STEPS"
    log "Generations per prompt: $GRPO_NUM_GEN"

    python3 scripts/18_grpo_rl.py \
        --checkpoint "$grpo_base" \
        --per_device_train_batch_size "$GRPO_BATCH" \
        --gradient_accumulation_steps "$GRPO_GRAD" \
        --max_steps "$GRPO_MAX_STEPS" \
        --num_generations "$GRPO_NUM_GEN" \
        --output_dir checkpoints/agent_sft_grpo \
        --reward_mode coding_tui \
        --developer_prompt "You are a coding agent. Use tools to read files, write code, run tests, and complete programming tasks. Do not just analyze - always take action and produce working code." \
        2>&1 | tee -a "$LOG_FILE"

    # Soft quality gate
    if [ -f "$GRPO_FINAL/adapter_config.json" ]; then
        update_gate "grpo" "passed" "true"
        log "GRPO complete."
    else
        update_gate "grpo" "passed" "false"
        log "WARNING: GRPO final checkpoint not found."
    fi
}

# =============================================================================
# Phase 5: Export
# =============================================================================
phase_export() {
    log_section "PHASE 5: Export"

    if [ -d "$EXPORT_DIR/hf" ]; then
        log "Export already exists. Skipping."
        return 0
    fi

    # Find best adapter (priority order)
    local adapter_path=""
    for path in "$GRPO_FINAL" "$IPO_FINAL" "$AGENT_FINAL" "$TC_FINAL"; do
        if [ -d "$path" ]; then
            adapter_path="$path"
            break
        fi
    done

    if [ -z "$adapter_path" ]; then
        log "ERROR: No adapter checkpoint found for export."
        return 1
    fi

    # Determine base model
    local base_model_arg=""
    if [ "$adapter_path" != "$TC_FINAL" ] && [ -f "$MERGED_MODEL/config.json" ]; then
        base_model_arg="--base_model $MERGED_MODEL"
    fi

    log "Exporting adapter: $adapter_path"
    if [ -n "$base_model_arg" ]; then
        log "Base model: $MERGED_MODEL"
    fi

    python3 scripts/19_merge_adapter.py \
        --adapter_path "$adapter_path" \
        --output_dir "$EXPORT_DIR" \
        --export_formats hf gguf_q4 \
        $base_model_arg \
        2>&1 | tee -a "$LOG_FILE"

    log "Export complete: $EXPORT_DIR"
}

# =============================================================================
# Summary Report
# =============================================================================
print_summary() {
    log_section "TRAINING COMPLETE"
    log "Quality gates report: $GATE_FILE"
    python3 -c "
import json
with open('$GATE_FILE') as f:
    data = json.load(f)
for phase, info in data.items():
    status = info.get('status', '')
    loss = info.get('loss', 'n/a')
    passed = info.get('passed', 'n/a')
    print(f'  {phase:20s}  loss={loss}  passed={passed}  {status}')
"
    log ""
    log "Log file: $LOG_FILE"
    log "Checkpoints:"
    for d in "$TC_FINAL" "$MERGED_MODEL" "$AGENT_FINAL" "$IPO_FINAL" "$GRPO_FINAL" "$EXPORT_DIR"; do
        if [ -d "$d" ]; then
            log "  [OK] $d"
        else
            log "  [--] $d"
        fi
    done
}

# =============================================================================
# Main
# =============================================================================
main() {
    log_section "TUI Training Pipeline"
    log "Profile: $BATCH_PROFILE | Quick test: $QUICK_TEST"
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
    log "Start time: $(date)"

    phase_data_prep
    phase_tool_calling_sft
    phase_merge
    phase_agent_sft
    phase_ipo
    phase_grpo
    phase_export
    print_summary

    log "Pipeline finished at $(date)"
}

main
```

**Step 2: Make executable**

Run: `chmod +x scripts/run_tui_pipeline.sh`

**Step 3: Verify syntax**

Run: `bash -n scripts/run_tui_pipeline.sh && echo "OK"`
Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/run_tui_pipeline.sh
git commit -m "Add TUI training pipeline orchestrator for RunPod H100"
```

---

### Task 3: Create the screen wrapper script

**Files:**
- Create: `scripts/start_tui_training.sh`

**Step 1: Create `scripts/start_tui_training.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Start TUI Training in Screen Session
# =============================================================================
# Launches run_tui_pipeline.sh inside a detached screen session.
#
# Usage:
#   bash scripts/start_tui_training.sh [args passed to run_tui_pipeline.sh]
#
# Reconnect:
#   screen -r tui_training
#
# Detach:
#   Ctrl+A, D
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="tui_training"

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists."
    echo ""
    echo "Options:"
    echo "  Attach:  screen -r $SESSION_NAME"
    echo "  Kill:    screen -S $SESSION_NAME -X quit"
    echo ""
    exit 1
fi

# Check screen is installed
if ! command -v screen &>/dev/null; then
    echo "Installing screen..."
    apt-get update -qq && apt-get install -y -qq screen
fi

mkdir -p "$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/tui_pipeline_${TIMESTAMP}.log"

echo "Starting TUI training pipeline in screen session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "  Attach:  screen -r $SESSION_NAME"
echo "  Detach:  Ctrl+A, D"
echo ""

# Launch in screen with logging
screen -dmS "$SESSION_NAME" bash -c "cd $PROJECT_DIR && bash scripts/run_tui_pipeline.sh $* 2>&1 | tee $LOG_FILE; echo 'Pipeline finished. Press Enter to exit.'; read"

echo "Pipeline started. Attaching in 2 seconds..."
sleep 2
screen -r "$SESSION_NAME"
```

**Step 2: Make executable**

Run: `chmod +x scripts/start_tui_training.sh`

**Step 3: Verify syntax**

Run: `bash -n scripts/start_tui_training.sh && echo "OK"`
Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/start_tui_training.sh
git commit -m "Add screen wrapper for TUI training pipeline"
```

---

### Task 4: Test on RunPod with --quick-test

**Step 1: Push to remote**

```bash
git push origin main
```

**Step 2: Pull on RunPod**

```bash
ssh root@205.196.17.114 -p 11376 "cd /workspace/llm-training-pipeline && git pull"
```

**Step 3: Run quick test**

```bash
ssh root@205.196.17.114 -p 11376 "cd /workspace/llm-training-pipeline && bash scripts/run_tui_pipeline.sh --quick-test 2>&1 | head -100"
```

Verify: Data downloads, Phase 1 starts, training runs for 50 steps.

**Step 4: Commit any fixes**

Fix any issues found during quick test, commit.

---

### Task 5: Deploy and start full training

**Step 1: Start training in screen**

```bash
ssh root@205.196.17.114 -p 11376 "cd /workspace/llm-training-pipeline && bash scripts/start_tui_training.sh --batch-profile balanced"
```

**Step 2: Verify training started**

```bash
ssh root@205.196.17.114 -p 11376 "screen -r tui_training"
```

Detach with Ctrl+A, D.

**Step 3: Monitor**

```bash
ssh root@205.196.17.114 -p 11376 "tail -20 /workspace/llm-training-pipeline/logs/tui_pipeline_*.log"
```
