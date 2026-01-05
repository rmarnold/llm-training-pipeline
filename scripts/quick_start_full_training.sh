#!/bin/bash
# Quick start script for full 7B training using working demo structure

set -e

echo "============================================================"
echo "FULL 7B TRAINING - QUICK START"
echo "============================================================"

# Use demo data format that we know works
echo "[1/3] Verifying demo data exists..."
if [ ! -f "data/packed/pretrain_demo.npy" ]; then
    echo "Regenerating demo data..."
    python scripts/demo_generate_data.py
    python scripts/demo_tokenize.py
fi
echo "✓ Data ready"

# Initialize 7B model with correct vocab (50304 to match demo)
echo "[2/3] Ensuring 7B model initialized..."
if [ ! -d "checkpoints/init" ]; then
    python scripts/04_init_model.py
fi
echo "✓ Model ready"

# Start training
echo "[3/3] Starting 7B training..."
python scripts/full_model_pretrain.py

echo "✓ Training started!"
