#!/bin/bash
# scripts/resume_pipeline.sh
# Resume training from the latest checkpoint for a given stage

set -e

STAGE=$1

if [ -z "$STAGE" ]; then
    echo "Usage: ./resume_pipeline.sh [pretrain|sft|dpo]"
    echo ""
    echo "Resumes training from the latest checkpoint for the specified stage."
    exit 1
fi

find_latest_checkpoint() {
    local checkpoint_dir=$1

    if [ ! -d "$checkpoint_dir" ]; then
        echo ""
        return
    fi

    # Find checkpoint directories and sort by step number
    local latest=$(find "$checkpoint_dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | \
        sed 's/.*checkpoint-//' | sort -n | tail -1)

    if [ -n "$latest" ]; then
        echo "${checkpoint_dir}/checkpoint-${latest}"
    else
        echo ""
    fi
}

validate_checkpoint() {
    local checkpoint_path=$1

    if [ -z "$checkpoint_path" ]; then
        return 1
    fi

    if [ ! -d "$checkpoint_path" ]; then
        return 1
    fi

    # Check for essential checkpoint files
    if [ -f "${checkpoint_path}/config.json" ] || \
       [ -f "${checkpoint_path}/pytorch_model.bin" ] || \
       [ -f "${checkpoint_path}/model.safetensors" ] || \
       [ -f "${checkpoint_path}/trainer_state.json" ]; then
        return 0
    fi

    return 1
}

case $STAGE in
    "pretrain")
        CHECKPOINT_DIR="checkpoints/pretrain"
        SCRIPT="scripts/05_pretrain.py"
        ;;
    "sft")
        CHECKPOINT_DIR="checkpoints/sft"
        SCRIPT="scripts/07_sft.py"
        ;;
    "dpo")
        CHECKPOINT_DIR="checkpoints/dpo"
        SCRIPT="scripts/09_dpo.py"
        ;;
    *)
        echo "Error: Unknown stage '$STAGE'"
        echo "Usage: ./resume_pipeline.sh [pretrain|sft|dpo]"
        exit 1
        ;;
esac

echo "Looking for checkpoints in: $CHECKPOINT_DIR"

LATEST_CHECKPOINT=$(find_latest_checkpoint "$CHECKPOINT_DIR")

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoints found in $CHECKPOINT_DIR"
    echo ""
    echo "Available options:"
    echo "  1. Start fresh training: python $SCRIPT"
    echo "  2. Check if the checkpoint directory exists"
    exit 1
fi

if ! validate_checkpoint "$LATEST_CHECKPOINT"; then
    echo "Error: Checkpoint at $LATEST_CHECKPOINT appears to be invalid or incomplete"
    echo ""
    echo "The checkpoint directory exists but is missing essential files."
    echo "Consider starting fresh or using an earlier checkpoint."
    exit 1
fi

# Extract step number for display
STEP_NUM=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint-//')

echo "Found valid checkpoint: $LATEST_CHECKPOINT (step $STEP_NUM)"
echo ""
echo "Resuming $STAGE training..."
echo ""

python "$SCRIPT" --resume_from_checkpoint "$LATEST_CHECKPOINT"
