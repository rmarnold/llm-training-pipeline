#!/bin/bash
# Demo pipeline - runs small-scale version to verify everything works

set -e

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     DEMO PIPELINE - Small Scale Verification              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found"
    exit 1
fi

# Check GPU
echo "Checking for GPU..."
if python -c "import torch; print('✓ GPU Available:', torch.cuda.is_available(), '- Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null; then
    :
else
    echo "⚠️  No GPU found, will run on CPU (slower)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "[1/5] Generating synthetic demo data..."
echo "─────────────────────────────────────────────────────────────"
python scripts/demo_generate_data.py

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "[2/5] Tokenizing and packing data..."
echo "─────────────────────────────────────────────────────────────"
python scripts/demo_tokenize.py

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "[3/5] Initializing tiny model (~124M params)..."
echo "─────────────────────────────────────────────────────────────"
python scripts/demo_init_model.py

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "[4/5] Running pretraining (10 steps)..."
echo "─────────────────────────────────────────────────────────────"
python scripts/demo_pretrain.py

echo ""
echo "─────────────────────────────────────────────────────────────"
echo "[5/5] Verification..."
echo "─────────────────────────────────────────────────────────────"

# Check if checkpoints exist
if [ -d "checkpoints/demo_pretrain_final" ]; then
    echo "✓ Final checkpoint exists"
else
    echo "❌ Final checkpoint missing"
    exit 1
fi

# Check if model can load
if python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('checkpoints/demo_pretrain_final'); print('✓ Model loads successfully')" 2>&1; then
    :
else
    echo "❌ Model failed to load"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     ✓ DEMO PIPELINE COMPLETE - All systems verified!      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Key verification points:"
echo "  ✓ Data generation working"
echo "  ✓ Tokenization working"
echo "  ✓ Model initialization working"
echo "  ✓ Training loop working"
echo "  ✓ Checkpointing working"
echo "  ✓ Model loading working"
echo ""
echo "The full pipeline is ready for production use!"
echo ""
