#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Start TUI Training in Screen Session
# =============================================================================
# Launches run_tui_pipeline.sh inside a detached screen session.
#
# Usage:
#   bash scripts/start_tui_training.sh [args passed to run_tui_pipeline.sh]
#
# Reconnect:
#   screen -r tui_training
#
# Detach:
#   Ctrl+A, D
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION_NAME="tui_training"

# Check if screen session already exists
if screen -list 2>/dev/null | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists."
    echo ""
    echo "Options:"
    echo "  Attach:  screen -r $SESSION_NAME"
    echo "  Kill:    screen -S $SESSION_NAME -X quit"
    echo ""
    exit 1
fi

# Check screen is installed
if ! command -v screen &>/dev/null; then
    echo "Installing screen..."
    apt-get update -qq && apt-get install -y -qq screen
fi

mkdir -p "$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/tui_pipeline_${TIMESTAMP}.log"

echo "Starting TUI training pipeline in screen session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo ""
echo "  Attach:  screen -r $SESSION_NAME"
echo "  Detach:  Ctrl+A, D"
echo ""

# Launch in screen with logging
screen -dmS "$SESSION_NAME" bash -c "cd $PROJECT_DIR && bash scripts/run_tui_pipeline.sh $* 2>&1 | tee $LOG_FILE; echo 'Pipeline finished. Press Enter to exit.'; read"

echo "Pipeline started."
if [ -t 1 ]; then
    echo "Attaching in 2 seconds..."
    sleep 2
    screen -r "$SESSION_NAME"
else
    echo "Non-interactive session detected. Attach with: screen -r $SESSION_NAME"
fi
