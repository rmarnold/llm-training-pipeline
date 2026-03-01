# Disk Space Management for Colab Training Pipeline

**Status:** DONE
**Date:** 2026-02-26

## Problem
GPT-OSS 20B training notebook crashes with `SafetensorError: I/O error: No space left on device (os error 28)` on Colab H100 (~200GB ephemeral). Disk fills from HF cache, merged models, and intermediate checkpoints.

## Solution

### New: `scripts/pipeline_lib/disk_manager.py`
- `DiskManager` class with declarative phase cleanup rules
- `get_disk_status()` — free/used/total GB + breakdown
- `cleanup_between_phases(completed, next)` — phase-aware artifact deletion
- `cleanup_hf_cache()` / `cleanup_caches()` — cache cleanup
- `cleanup_intermediate_checkpoints(dir)` — removes checkpoint-* dirs, preserves final/
- `pre_phase_disk_check(phase, required_gb)` — escalating auto-cleanup before training
- `session_startup_cleanup()` — idempotent cleanup on session reconnect
- Safety: never deletes */final/, preserves trainer_state.json, skips symlinks

### Modified: `notebooks/train_gpt_oss_coding_tui.ipynb`
- Cell 12: DiskManager init + disk status display
- Cell 30: Session recovery re-init
- Pre-phase checks before each training phase (cells 32, 34, 40, 45, 50)
- Post-phase cleanup after merge, Agent SFT, IPO, GRPO (cells 37, 42, 47, 52)

### Modified: `scripts/pipeline_lib/__init__.py`
- Added `DiskManager` lazy import

### New: `tests/test_disk_manager.py`
- 20 unit tests covering all DiskManager methods
- All 85 tests pass (no regressions)

## Key Design Decisions
1. Declarative cleanup rules as data structure, not scattered logic
2. Escalating auto-cleanup: caches -> HF cache -> intermediate checkpoints
3. Biggest win: ~40GB merged model deleted after Agent SFT completes
4. Symlink-aware (mounted Drive mode skips cleanup)
5. Dry-run mode for safe testing
