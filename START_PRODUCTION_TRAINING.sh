#!/bin/bash
# Quick start script for full production training

set -e

echo "============================================================"
echo "PRODUCTION TRAINING - QUICK START"
echo "============================================================"

# Set environment
export USER=root
export HOME=/root
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache

# Step 1: Prepare production data
echo "[1/2] Preparing production data (100K samples, 512 tokens)..."
if [ ! -f "data/packed/production_pretrain.npy" ]; then
    python scripts/prepare_production_data.py
else
    echo "✓ Production data already exists"
fi

# Step 2: Start training
echo "[2/2] Starting production training..."
echo ""
echo "Configuration:"
echo "  - Model: 6.08B parameters"
echo "  - Data: 100K samples × 512 tokens"
echo "  - GPU Target: 85-95% utilization"
echo "  - Duration: ~64 hours (100K steps)"
echo ""

python scripts/production_pretrain.py 2>&1 | tee production_training.log

echo ""
echo "============================================================"
echo "Production training started!"
echo "Monitor with: watch -n 2 nvidia-smi"
echo "TensorBoard: tensorboard --logdir logs/production_pretrain"
echo "============================================================"
