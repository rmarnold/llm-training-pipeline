#!/bin/bash
# ==========================================================================
# RunPod B200 Pod Setup — reusable environment bootstrap
# ==========================================================================
# Run this after SSH-ing into a fresh RunPod pod (or via scp + ssh).
# Handles: cache symlinks, PyTorch upgrade, Unsloth install, repo clone.
#
# Usage:
#   scp -P $PORT scripts/runpod_setup.sh root@$HOST:/workspace/setup.sh
#   ssh -p $PORT root@$HOST "bash /workspace/setup.sh"
#
# Or after the pipeline script handles it automatically:
#   bash scripts/run_agentic_pipeline.sh   # calls setup internally
# ==========================================================================
set -euo pipefail

echo "=== RunPod Pod Setup ==="

# --- Cache symlinks (CRITICAL — container overlay is only 50-100GB) ---
echo "=== Symlinking caches to /workspace ==="
mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch /workspace/tmp
rm -rf /root/.cache/huggingface /root/.cache/torch 2>/dev/null || true
ln -sf /workspace/.cache/huggingface /root/.cache/huggingface
ln -sf /workspace/.cache/torch /root/.cache/torch
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export TMPDIR=/workspace/tmp

# --- Disk check ---
echo "=== Disk space ==="
df -h / /workspace | grep -v tmpfs

# --- Upgrade PyTorch (B200 needs >= 2.9 for Flex Attention backward) ---
echo "=== Upgrading PyTorch ==="
pip install -q --upgrade torch torchvision torchaudio 2>&1 | tail -5

# --- Install Unsloth + training deps ---
echo "=== Installing Unsloth + deps ==="
pip install -q unsloth 'bitsandbytes>=0.48' tiktoken trl datasets 2>&1 | tail -5

# --- Clone / update repo ---
echo "=== Cloning repo ==="
cd /workspace
if [ ! -d llm-training-pipeline ]; then
    git clone https://github.com/rmarnold/llm-training-pipeline.git
else
    cd llm-training-pipeline && git pull --ff-only
    cd /workspace
fi

# --- Install pipeline package ---
echo "=== Installing pipeline ==="
cd /workspace/llm-training-pipeline
pip install -q -e '.[gpt_oss]' 2>&1 | tail -5

# --- Install Rust (needed for cargo-mutants in trajectory gen) ---
if ! command -v cargo &>/dev/null; then
    echo "=== Installing Rust toolchain ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -2
    source "$HOME/.cargo/env"
    cargo install cargo-mutants 2>&1 | tail -2
fi

# --- Verify GPU ---
echo "=== Verifying GPU ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'VRAM: {vram:.0f} GB')
cap = torch.cuda.get_device_capability()
print(f'Compute: SM {cap[0]}.{cap[1]}')
if cap[0] >= 10:
    print('Blackwell detected — Flex Attention available')
elif cap[0] >= 9:
    print('Hopper detected')
"

echo "=== Setup complete ==="
echo ""
echo "To run the full pipeline:"
echo "  cd /workspace/llm-training-pipeline"
echo "  nohup bash scripts/run_agentic_pipeline.sh > /workspace/pipeline.log 2>&1 &"
echo "  tail -f /workspace/pipeline.log"
