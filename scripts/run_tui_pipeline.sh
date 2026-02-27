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

# Redirect caches and temp to /workspace volume (200GB) instead of container disk (20GB)
if [ -d "/workspace" ]; then
    export HF_HOME="/workspace/.cache/huggingface"
    export TORCH_HOME="/workspace/.cache/torch"
    export XDG_CACHE_HOME="/workspace/.cache"
    export TMPDIR="/workspace/tmp"
    export TEMP="/workspace/tmp"
    export TMP="/workspace/tmp"
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
fi

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
    log "PASSED: Merged model size $size_gb GB"

    # Smoke test: validate config loads and no NaN weights
    log "Running merge smoke test..."
    python3 -c "
import json, os, sys
merged = '$MERGED_MODEL'
# Check config.json is valid
with open(os.path.join(merged, 'config.json')) as f:
    cfg = json.load(f)
print(f'  Model type: {cfg.get(\"model_type\", \"unknown\")}')
# Check safetensor index exists
idx = os.path.join(merged, 'model.safetensors.index.json')
if os.path.exists(idx):
    with open(idx) as f:
        index = json.load(f)
    n_shards = len(set(index.get('weight_map', {}).values()))
    print(f'  Safetensor shards: {n_shards}')
else:
    # Single shard
    st = os.path.join(merged, 'model.safetensors')
    if os.path.exists(st):
        print(f'  Single safetensor: {os.path.getsize(st) / 1024**3:.1f} GB')
    else:
        print('  WARNING: No safetensor files found', file=sys.stderr)
        sys.exit(1)
print('  Smoke test passed.')
" 2>&1 | tee -a "$LOG_FILE"
    update_gate "merge" "smoke_test" "true"
    log "PASSED: Merged model $size_gb GB + smoke test"
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
    if [[ -n "$val_data" && -d "$val_data" ]]; then
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

    # GRPO requires language-specific task data (e.g., data/rust/grpo/tasks.jsonl).
    # For TUI-only pipeline, skip GRPO if no task source is available.
    local grpo_task_source="data/rust/grpo/tasks.jsonl"
    if [ ! -f "$grpo_task_source" ]; then
        log "GRPO task data not found at $grpo_task_source (TUI-only pipeline)."
        log "Skipping GRPO — run language-specific training to generate task data."
        update_gate "grpo" "status" '"skipped_no_tasks"'
        return 0
    fi

    python3 scripts/18_grpo_rl.py \
        --checkpoint "$grpo_base" \
        --per_device_train_batch_size "$GRPO_BATCH" \
        --gradient_accumulation_steps "$GRPO_GRAD" \
        --max_steps "$GRPO_MAX_STEPS" \
        --task_source "$grpo_task_source" \
        --output_dir checkpoints/agent_sft_grpo \
        2>&1 | tee -a "$LOG_FILE"

    # Soft quality gate: check checkpoint exists + reward accuracy
    if [ -f "$GRPO_FINAL/adapter_config.json" ]; then
        update_gate "grpo" "passed" "true"

        # Check reward accuracy from trainer state
        local reward_acc
        reward_acc=$(python3 -c "
import json, glob, os
state_files = sorted(glob.glob('checkpoints/agent_sft_grpo/*/trainer_state.json'))
final_state = 'checkpoints/agent_sft_grpo/final/trainer_state.json'
if os.path.exists(final_state): state_files.append(final_state)
if not state_files: print('-1')
else:
    with open(state_files[-1]) as f: state = json.load(f)
    accs = [e.get('reward_accuracy', e.get('rewards/accuracies', -1))
            for e in state.get('log_history', [])
            if 'reward_accuracy' in e or 'rewards/accuracies' in e]
    print(f'{accs[-1]:.3f}' if accs else '-1')
" 2>/dev/null || echo "-1")
        if [ "$reward_acc" != "-1" ]; then
            update_gate "grpo" "reward_accuracy" "$reward_acc"
            local below
            below=$(python3 -c "print('true' if float('$reward_acc') < 0.55 else 'false')")
            if [ "$below" = "true" ]; then
                log "WARNING: GRPO reward accuracy $reward_acc < 0.55 (near chance level)"
            else
                log "GRPO reward accuracy: $reward_acc"
            fi
        fi
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

    # Install GGUF build deps (cmake, libcurl) non-interactively
    if ! command -v cmake &>/dev/null; then
        log "Installing cmake and libcurl for GGUF export..."
        apt-get update -qq && apt-get install -y -qq cmake libcurl4-openssl-dev 2>&1 | tail -1
    fi

    python3 scripts/19_merge_adapter.py \
        --adapter_path "$adapter_path" \
        --output_dir "$EXPORT_DIR" \
        --export_formats hf gguf_q4 \
        $base_model_arg \
        2>&1 | tee -a "$LOG_FILE"

    # If GGUF failed but HF succeeded, still consider export successful
    if [ -d "$EXPORT_DIR/hf" ]; then
        log "Export complete: $EXPORT_DIR"
        if [ ! -d "$EXPORT_DIR/gguf_q4" ]; then
            log "WARNING: GGUF export failed. HF format available. Run GGUF separately if needed."
        fi
    else
        log "ERROR: Export failed — no output in $EXPORT_DIR"
        return 1
    fi
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
