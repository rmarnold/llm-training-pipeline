#!/bin/bash
# ==========================================================================
# Agentic Training Pipeline — Full run on RunPod B200
# ==========================================================================
# Stages: Trajectory Gen → Core Agent SFT → IPO → GRPO → Merge
#
# Usage:
#   bash scripts/run_agentic_pipeline.sh
#   bash scripts/run_agentic_pipeline.sh --skip-data        # Skip trajectory gen
#   bash scripts/run_agentic_pipeline.sh --grpo-only        # Just GRPO from IPO ckpt
#   bash scripts/run_agentic_pipeline.sh --max-steps=2000   # Short GRPO run
# ==========================================================================
set -euo pipefail

# --- Parse CLI flags ---
SKIP_DATA=false
GRPO_ONLY=false
MAX_GRPO_STEPS=""
for arg in "$@"; do
  case $arg in
    --skip-data)      SKIP_DATA=true ;;
    --grpo-only)      GRPO_ONLY=true ;;
    --max-steps=*)    MAX_GRPO_STEPS="${arg#*=}" ;;
    *)                echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

REPO_DIR="${REPO_DIR:-/workspace/llm-training-pipeline}"
LOG_DIR="${REPO_DIR}/logs/agentic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

# ==========================================================================
# 0. Environment setup
# ==========================================================================
log "=== Environment Setup ==="

# Symlink caches to /workspace (CRITICAL — container overlay is only 50GB)
mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch /workspace/tmp
if [ ! -L /root/.cache/huggingface ]; then
  rm -rf /root/.cache/huggingface
  ln -sf /workspace/.cache/huggingface /root/.cache/huggingface
fi
if [ ! -L /root/.cache/torch ]; then
  rm -rf /root/.cache/torch
  ln -sf /workspace/.cache/torch /root/.cache/torch
fi
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export TMPDIR=/workspace/tmp
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify disk space
ROOT_FREE=$(df -BG / | awk 'NR==2{print $4}' | tr -d 'G')
WS_FREE=$(df -BG /workspace | awk 'NR==2{print $4}' | tr -d 'G')
log "Disk: root=${ROOT_FREE}GB free, /workspace=${WS_FREE}GB free"
if [ "$WS_FREE" -lt 50 ]; then
  log "WARNING: /workspace has <50GB free. May run out during training."
fi

# ==========================================================================
# 1. Install dependencies
# ==========================================================================
log "=== Installing dependencies ==="

# Upgrade PyTorch to >=2.9 for Blackwell Flex Attention
pip install -q --upgrade torch torchvision torchaudio 2>&1 | tail -3
pip install -q unsloth 'bitsandbytes>=0.48' tiktoken trl datasets 2>&1 | tail -3

# Clone/update repo
if [ ! -d "$REPO_DIR" ]; then
  log "Cloning repo..."
  cd /workspace
  git clone https://github.com/rmarnold/llm-training-pipeline.git
fi
cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true
pip install -q -e '.[gpt_oss]' 2>&1 | tail -3

# Install Rust toolchain (needed for cargo-mutants in trajectory gen)
if ! command -v cargo &>/dev/null; then
  log "Installing Rust toolchain..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -2
  source "$HOME/.cargo/env"
  cargo install cargo-mutants 2>&1 | tail -2
fi

# Verify GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB')
cap = torch.cuda.get_device_capability()
print(f'Compute capability: {cap[0]}.{cap[1]}')
if cap[0] >= 10:
    print('Blackwell detected — Flex Attention available')
elif cap[0] >= 9:
    print('Hopper detected')
" 2>&1 | tee -a "$LOG_DIR/pipeline.log"

log "=== Setup complete ==="

# ==========================================================================
# 2. Generate trajectory data
# ==========================================================================
if [ "$GRPO_ONLY" = true ]; then
  log "=== GRPO-only mode, skipping data gen + SFT + IPO ==="
elif [ "$SKIP_DATA" = true ]; then
  log "=== Skipping data generation (--skip-data) ==="
else
  log "=== Stage 1: Generating mutations ==="
  # --check-only: uses `cargo check` instead of `cargo test` — avoids
  #   "cargo build failed in unmutated tree" errors from missing test deps
  #   in headless RunPod environments.  Produces compiler-error mutations.
  # --in-place: skip copying repo to temp dir (saves disk I/O)
  # --repo-workers 8: cap parallelism to avoid overwhelming the system
  python scripts/16_generate_mutations.py \
    --output_dir data/rust/mutations \
    --check-only \
    --in-place \
    --repo-workers 8 \
    --jobs 32 \
    2>&1 | tee "$LOG_DIR/01_mutations.log" || {
      log "WARNING: Mutation gen failed (Rust projects may not be available). Using Strandset fallback."
    }

  log "=== Stage 2: Generating trajectories ==="
  python scripts/15_generate_trajectories.py \
    --output_dir data/rust/core_agent/train \
    --max_samples 5000 \
    --planning_ratio 0.2 \
    --self_correction_ratio 0.15 \
    --multi_step_ratio 0.3 \
    2>&1 | tee "$LOG_DIR/02_trajectories.log"

  TRAJ_COUNT=$(python -c "
from datasets import load_from_disk
import os, glob
total = 0
for d in glob.glob('data/rust/core_agent/train/*/'):
    try:
        ds = load_from_disk(d)
        total += len(ds)
    except: pass
print(total)
" 2>/dev/null || echo "0")
  log "Generated $TRAJ_COUNT trajectories"

  if [ "$TRAJ_COUNT" -lt 100 ]; then
    log "ERROR: Too few trajectories ($TRAJ_COUNT). Check data pipeline."
    exit 1
  fi
fi

# ==========================================================================
# 3. Core Agent SFT
# ==========================================================================
if [ "$GRPO_ONLY" = false ]; then
  log "=== Stage 3: Core Agent SFT ==="
  python scripts/14_train_core_agent.py \
    --config configs/core_agent.yaml \
    2>&1 | tee "$LOG_DIR/03_core_agent_sft.log"

  # Check SFT checkpoint exists
  if [ ! -d "checkpoints/core_agent/final" ]; then
    log "ERROR: SFT checkpoint not found at checkpoints/core_agent/final"
    exit 1
  fi
  log "SFT complete: checkpoints/core_agent/final"
fi

# ==========================================================================
# 4. IPO Preference Training (2 epochs)
# ==========================================================================
if [ "$GRPO_ONLY" = false ]; then
  log "=== Stage 4: IPO Preference Training ==="
  python scripts/17_ipo_preference.py \
    --config configs/ipo.yaml \
    2>&1 | tee "$LOG_DIR/04_ipo.log"

  if [ ! -d "checkpoints/core_agent_ipo/final" ]; then
    log "ERROR: IPO checkpoint not found at checkpoints/core_agent_ipo/final"
    exit 1
  fi
  log "IPO complete: checkpoints/core_agent_ipo/final"
fi

# ==========================================================================
# 5. GRPO RL Training (curriculum + recovery bonus)
# ==========================================================================
log "=== Stage 5: GRPO RL Training ==="
GRPO_ARGS="--config configs/grpo.yaml"
if [ -n "$MAX_GRPO_STEPS" ]; then
  GRPO_ARGS="$GRPO_ARGS --max_steps $MAX_GRPO_STEPS"
fi

python scripts/18_grpo_rl.py $GRPO_ARGS \
  2>&1 | tee "$LOG_DIR/05_grpo.log"

if [ ! -d "checkpoints/core_agent_grpo/final" ]; then
  log "ERROR: GRPO checkpoint not found at checkpoints/core_agent_grpo/final"
  exit 1
fi
log "GRPO complete: checkpoints/core_agent_grpo/final"

# ==========================================================================
# 6. Merge adapter
# ==========================================================================
log "=== Stage 6: Merge adapter ==="
python scripts/19_merge_adapter.py \
  --adapter_path checkpoints/core_agent_grpo/final \
  --output_dir checkpoints/agentic_merged \
  2>&1 | tee "$LOG_DIR/06_merge.log"

log "Merge complete: checkpoints/agentic_merged"

# ==========================================================================
# Summary
# ==========================================================================
log ""
log "=============================================="
log "  Agentic Training Pipeline Complete"
log "=============================================="
log "  Logs:        $LOG_DIR"
log "  SFT ckpt:    checkpoints/core_agent/final"
log "  IPO ckpt:    checkpoints/core_agent_ipo/final"
log "  GRPO ckpt:   checkpoints/core_agent_grpo/final"
log "  Merged:      checkpoints/agentic_merged"
log ""
log "  Next: Convert to MXFP4 for serving:"
log "    python scripts/30_nvfp4_convert.py \\"
log "      --config configs/nvfp4_convert_20b.yaml \\"
log "      --mode ptq --moe-only --mxfp4-compat"
log "=============================================="
