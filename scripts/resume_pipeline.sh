#!/bin/bash
# scripts/resume_pipeline.sh

STAGE=$1

case $STAGE in
  "pretrain")
    python scripts/05_pretrain.py --resume_from_checkpoint $(ls -td checkpoints/pretrain/checkpoint-* | head -1)
    ;;
  "sft")
    python scripts/07_sft.py --resume_from_checkpoint $(ls -td checkpoints/sft/checkpoint-* | head -1)
    ;;
  "dpo")
    python scripts/09_dpo.py --resume_from_checkpoint $(ls -td checkpoints/dpo/checkpoint-* | head -1)
    ;;
  *)
    echo "Usage: ./resume_pipeline.sh [pretrain|sft|dpo]"
    exit 1
    ;;
esac
