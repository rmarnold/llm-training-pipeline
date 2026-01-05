#!/bin/bash
set -e

echo "Running smoke test (100 steps)..."

python scripts/05_pretrain.py \
  --config configs/pretrain.yaml \
  --max_steps 100 \
  --save_steps 50 \
  --eval_steps 50 \
  --logging_steps 10 \
  --output_dir checkpoints/smoke_test

# Verify checkpoints created
if [ ! -d "checkpoints/smoke_test/checkpoint-50" ]; then
  echo "❌ Smoke test failed: No checkpoint at step 50"
  exit 1
fi

if [ ! -d "checkpoints/smoke_test/checkpoint-100" ]; then
  echo "❌ Smoke test failed: No checkpoint at step 100"
  exit 1
fi

# Check logs for errors
if grep -i "error\|exception\|oom" logs/pretrain/events.out.tfevents.* 2>/dev/null; then
  echo "❌ Smoke test failed: Errors in logs"
  exit 1
fi

echo "✓ Smoke test passed!"
