"""TrainerCallback for automatic checkpoint backup to Google Drive.

Backs up each HuggingFace checkpoint to Drive immediately after it is saved,
so training can survive Colab session disconnects.  Works with all three
DriveHelper modes (mounted, service_account, local).

Usage in training scripts::

    from pipeline_lib.checkpoint_callback import make_drive_checkpoint_callback

    callback = make_drive_checkpoint_callback(output_dir)
    if callback:
        trainer.add_callback(callback)
"""

from __future__ import annotations

import logging
import os
import time

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class DriveCheckpointCallback(TrainerCallback):
    """Back up checkpoints to Google Drive on every save."""

    def __init__(self, drive_helper, drive_relative_base: str):
        """
        Parameters
        ----------
        drive_helper : DriveHelper instance
        drive_relative_base : Drive-side path prefix, e.g. "checkpoints/tool_calling_sft"
        """
        self.drive_helper = drive_helper
        self.drive_relative_base = drive_relative_base
        self._last_backed_up: str | None = None

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after each checkpoint save."""
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if not os.path.isdir(checkpoint_dir):
            return

        drive_path = os.path.join(
            self.drive_relative_base, f"checkpoint-{state.global_step}"
        )

        t0 = time.time()
        print(f"\n[Drive] Backing up checkpoint-{state.global_step}...", flush=True)
        try:
            self.drive_helper.backup(checkpoint_dir, drive_path)
            elapsed = time.time() - t0
            print(f"[Drive] Backed up in {elapsed:.1f}s → {drive_path}", flush=True)
            self._last_backed_up = checkpoint_dir
        except Exception as e:
            print(f"[Drive] Backup failed (training continues): {e}", flush=True)
            logger.warning("Drive checkpoint backup failed: %s", e)


def make_drive_checkpoint_callback(
    output_dir: str,
    sa_credentials_path: str = "service_account.json",
    config_path: str = "data/config_coding_tui.json",
) -> DriveCheckpointCallback | None:
    """Create a DriveCheckpointCallback if Drive is configured.

    Reads Drive config from the saved pipeline config (written by the notebook).
    Returns None if Drive is not configured or in local mode.
    """
    import json

    # Read saved pipeline config
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    use_sa = config.get("use_service_account", False)
    folder_id = config.get("drive_folder_id", "")

    # Determine Drive mode
    drive_base = os.environ.get("DRIVE_BASE", "")
    if use_sa and folder_id and os.path.exists(sa_credentials_path):
        mode = "service_account"
    elif drive_base:
        mode = "mounted"
    else:
        # Check if DRIVE_BASE was set by the notebook via mounted mode
        # (notebook stores it as a Python variable, not env var —
        #  but with symlinks, checkpoints are already on Drive)
        return None

    from pipeline_lib.drive_utils import DriveHelper

    try:
        if mode == "service_account":
            helper = DriveHelper(
                mode="service_account",
                credentials_path=sa_credentials_path,
                folder_id=folder_id,
            )
        else:
            helper = DriveHelper(mode="mounted", drive_base=drive_base)
    except Exception as e:
        print(f"[Drive] Could not init DriveHelper for checkpoint backup: {e}")
        return None

    # Use output_dir as the Drive-relative path (e.g., "checkpoints/tool_calling_sft")
    drive_relative = output_dir

    print(f"[Drive] Checkpoint backup enabled → {drive_relative} ({mode} mode)")
    return DriveCheckpointCallback(helper, drive_relative)
