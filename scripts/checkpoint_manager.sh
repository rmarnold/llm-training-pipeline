#!/bin/bash

# Checkpoint versioning and management
CHECKPOINT_DIR="checkpoints"
ARCHIVE_DIR="checkpoints_archive"

# Archive old checkpoints
archive_checkpoint() {
  local checkpoint_path=$1
  local archive_name=$(basename $checkpoint_path)_$(date +%Y%m%d_%H%M%S)

  echo "Archiving $checkpoint_path to $ARCHIVE_DIR/$archive_name"
  mkdir -p $ARCHIVE_DIR
  tar -czf $ARCHIVE_DIR/$archive_name.tar.gz $checkpoint_path

  # Verify archive
  if [ $? -eq 0 ]; then
    echo "✓ Archive created successfully"
  else
    echo "❌ Archive failed"
    exit 1
  fi
}

# Keep only N best checkpoints based on eval loss
keep_best_n() {
  local checkpoint_pattern=$1
  local n=$2

  # List checkpoints sorted by eval loss
  ls -t $checkpoint_pattern/checkpoint-*/trainer_state.json | \
    jq -r '.best_metric' | \
    sort -n | \
    head -n $n

  # TODO: Delete others after archiving
}

# Usage
archive_checkpoint "checkpoints/pretrain_final"
