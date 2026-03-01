"""Tests for pipeline_lib.disk_manager."""

import json
import os
import shutil

import pytest

from pipeline_lib.disk_manager import (
    CleanupResult,
    DiskManager,
    DiskStatus,
    Phase,
    _dir_size_gb,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def dm(temp_dir):
    """DiskManager rooted at a temp directory."""
    return DiskManager(base_dir=temp_dir)


@pytest.fixture
def dm_dry(temp_dir):
    """DiskManager in dry-run mode."""
    return DiskManager(base_dir=temp_dir, dry_run=True)


def _make_checkpoint_tree(base, phase_dir, num_checkpoints=3, final=True):
    """Create a fake checkpoint directory structure.

    Returns the phase directory path.
    """
    phase_path = os.path.join(base, phase_dir)
    os.makedirs(phase_path, exist_ok=True)

    for i in range(1, num_checkpoints + 1):
        ckpt = os.path.join(phase_path, f"checkpoint-{i * 100}")
        os.makedirs(ckpt, exist_ok=True)
        # Write a dummy file (~1 KB)
        with open(os.path.join(ckpt, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 1024)
        # Write trainer_state.json
        with open(os.path.join(ckpt, "trainer_state.json"), "w") as f:
            json.dump({"global_step": i * 100, "log_history": []}, f)

    if final:
        final_path = os.path.join(phase_path, "final")
        os.makedirs(final_path, exist_ok=True)
        with open(os.path.join(final_path, "adapter_config.json"), "w") as f:
            json.dump({"r": 64}, f)
        with open(os.path.join(final_path, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 1024)

    return phase_path


# ── DiskStatus ───────────────────────────────────────────────────────────────


class TestGetDiskStatus:
    def test_returns_valid_status(self, dm):
        status = dm.get_disk_status()
        assert isinstance(status, DiskStatus)
        assert status.total_gb > 0
        assert status.free_gb > 0
        assert status.used_gb > 0
        assert status.total_gb >= status.used_gb

    def test_breakdown_includes_existing_dirs(self, dm, temp_dir):
        os.makedirs(os.path.join(temp_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(temp_dir, "checkpoints", "dummy.bin"), "wb") as f:
            f.write(b"\x00" * 10240)
        status = dm.get_disk_status(paths=["checkpoints"])
        assert "checkpoints" in status.breakdown
        assert status.breakdown["checkpoints"] > 0

    def test_breakdown_skips_missing_dirs(self, dm):
        status = dm.get_disk_status(paths=["nonexistent_dir"])
        assert "nonexistent_dir" not in status.breakdown


# ── Cleanup intermediate checkpoints ─────────────────────────────────────────


class TestCleanupIntermediateCheckpoints:
    def test_keeps_final(self, dm, temp_dir):
        _make_checkpoint_tree(temp_dir, "checkpoints/tool_calling_sft")
        result = dm.cleanup_intermediate_checkpoints("checkpoints/tool_calling_sft")

        # final/ should still exist
        assert os.path.exists(
            os.path.join(temp_dir, "checkpoints/tool_calling_sft/final")
        )
        # checkpoint-* should be gone
        remaining = list(
            (dm.base_dir / "checkpoints/tool_calling_sft").glob("checkpoint-*")
        )
        assert len(remaining) == 0
        assert result.freed_gb > 0
        assert len(result.deleted_paths) == 3

    def test_preserves_trainer_state(self, dm, temp_dir):
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft")
        dm.cleanup_intermediate_checkpoints("checkpoints/agent_sft")

        # trainer_state.json should be preserved in parent
        parent_ts = os.path.join(
            temp_dir, "checkpoints/agent_sft/trainer_state.json"
        )
        assert os.path.exists(parent_ts)
        with open(parent_ts) as f:
            state = json.load(f)
        # Should be from latest checkpoint (checkpoint-300)
        assert state["global_step"] == 300

    def test_noop_on_missing_dir(self, dm):
        result = dm.cleanup_intermediate_checkpoints("checkpoints/nonexistent")
        assert result.freed_gb == 0.0
        assert len(result.deleted_paths) == 0

    def test_skips_symlinks(self, dm, temp_dir):
        # Create a real dir and symlink a checkpoint to it
        real_dir = os.path.join(temp_dir, "real_ckpt")
        os.makedirs(real_dir, exist_ok=True)
        with open(os.path.join(real_dir, "model.safetensors"), "wb") as f:
            f.write(b"\x00" * 1024)

        phase_dir = os.path.join(temp_dir, "checkpoints/test_phase")
        os.makedirs(phase_dir, exist_ok=True)
        os.symlink(real_dir, os.path.join(phase_dir, "checkpoint-100"))

        result = dm.cleanup_intermediate_checkpoints("checkpoints/test_phase")
        # Symlink should be skipped
        assert len(result.skipped_paths) == 1
        assert "symlink" in result.skipped_paths[0]
        # Real dir should still exist
        assert os.path.exists(real_dir)


# ── Cleanup between phases ───────────────────────────────────────────────────


class TestCleanupBetweenPhases:
    def test_respects_must_keep(self, dm, temp_dir):
        _make_checkpoint_tree(temp_dir, "checkpoints/tool_calling_sft")
        result = dm.cleanup_between_phases(
            "tool_calling_sft", "tool_calling_sft_merge"
        )
        # final/ must survive
        assert os.path.exists(
            os.path.join(temp_dir, "checkpoints/tool_calling_sft/final")
        )
        # checkpoint-* should be deleted
        remaining = list(
            (dm.base_dir / "checkpoints/tool_calling_sft").glob("checkpoint-*")
        )
        assert len(remaining) == 0
        assert result.freed_gb > 0

    def test_deletes_entire_dir_when_specified(self, dm, temp_dir):
        _make_checkpoint_tree(temp_dir, "checkpoints/tool_calling_sft")
        # Also create the merged model dir (the "keep" target)
        merged = os.path.join(
            temp_dir, "checkpoints/gpt-oss-20b-coding-tui-merged"
        )
        os.makedirs(merged, exist_ok=True)

        result = dm.cleanup_between_phases(
            "tool_calling_sft_merge", "agent_sft"
        )
        # tool_calling_sft should be deleted entirely
        assert not os.path.exists(
            os.path.join(temp_dir, "checkpoints/tool_calling_sft")
        )
        # merged dir should survive (it's in keep)
        assert os.path.exists(merged)

    def test_agent_sft_to_ipo_preserves_merged_model(self, dm, temp_dir):
        """Merged model must survive through IPO/GRPO — adapters reference it."""
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft")
        merged = os.path.join(
            temp_dir, "checkpoints/gpt-oss-20b-coding-tui-merged"
        )
        os.makedirs(merged, exist_ok=True)
        with open(os.path.join(merged, "big_model.bin"), "wb") as f:
            f.write(b"\x00" * 4096)

        result = dm.cleanup_between_phases("agent_sft", "agent_sft_ipo")
        # Merged model must survive (adapters reference it via adapter_config.json)
        assert os.path.exists(merged)
        # agent_sft/final should survive
        assert os.path.exists(
            os.path.join(temp_dir, "checkpoints/agent_sft/final")
        )
        # Only intermediate checkpoints should be deleted
        remaining = list(
            (dm.base_dir / "checkpoints/agent_sft").glob("checkpoint-*")
        )
        assert len(remaining) == 0

    def test_grpo_to_export_cleans_merged_model(self, dm, temp_dir):
        """Merged model (~40GB) deleted after GRPO, before export cleanup."""
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft_grpo")
        merged = os.path.join(
            temp_dir, "checkpoints/gpt-oss-20b-coding-tui-merged"
        )
        os.makedirs(merged, exist_ok=True)
        with open(os.path.join(merged, "big_model.bin"), "wb") as f:
            f.write(b"\x00" * 4096)
        # Also create agent_sft_ipo (deleted in this phase)
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft_ipo")

        result = dm.cleanup_between_phases("agent_sft_grpo", "export")
        # Merged model should now be deleted
        assert not os.path.exists(merged)
        # agent_sft_ipo should be deleted
        assert not os.path.exists(
            os.path.join(temp_dir, "checkpoints/agent_sft_ipo")
        )
        # agent_sft_grpo/final should survive
        assert os.path.exists(
            os.path.join(temp_dir, "checkpoints/agent_sft_grpo/final")
        )

    def test_unknown_transition_is_noop(self, dm):
        result = dm.cleanup_between_phases("unknown_phase", "unknown_next")
        assert result.freed_gb == 0.0
        assert len(result.deleted_paths) == 0


# ── Pre-phase disk check ────────────────────────────────────────────────────


class TestPrePhaseDiskCheck:
    def test_passes_with_sufficient_space(self, dm):
        # Require 1 GB — any system should have that in temp
        assert dm.pre_phase_disk_check("test_phase", required_gb=1.0) is True

    def test_fails_with_impossible_request(self, dm):
        # Require more space than the entire disk
        status = dm.get_disk_status()
        impossible = status.total_gb + 100
        assert dm.pre_phase_disk_check("test_phase", required_gb=impossible) is False


# ── Dry run mode ─────────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_doesnt_delete(self, dm_dry, temp_dir):
        _make_checkpoint_tree(temp_dir, "checkpoints/tool_calling_sft")
        result = dm_dry.cleanup_between_phases(
            "tool_calling_sft", "tool_calling_sft_merge"
        )
        # Files should still exist
        remaining = list(
            (dm_dry.base_dir / "checkpoints/tool_calling_sft").glob("checkpoint-*")
        )
        assert len(remaining) == 3
        # But result should report what would be deleted
        assert len(result.deleted_paths) == 3
        assert all("dry run" in p for p in result.deleted_paths)
        # freed_gb should be 0 in dry run
        assert result.freed_gb == 0.0


# ── Session startup cleanup ─────────────────────────────────────────────────


class TestSessionStartupCleanup:
    def test_idempotent(self, dm, temp_dir):
        """Running startup cleanup twice produces the same result."""
        # Create a scenario: agent_sft is done, so merged model can be cleaned
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft")
        _make_checkpoint_tree(temp_dir, "checkpoints/tool_calling_sft")
        merged = os.path.join(
            temp_dir, "checkpoints/gpt-oss-20b-coding-tui-merged"
        )
        os.makedirs(os.path.join(merged, "hf"), exist_ok=True)
        with open(os.path.join(merged, "hf/config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        # First run
        r1 = dm.session_startup_cleanup()

        # Second run — should be a no-op (everything already cleaned)
        r2 = dm.session_startup_cleanup()
        assert r2.freed_gb == 0.0
        assert len(r2.errors) == 0

    def test_cleans_intermediate_checkpoints_for_completed_phases(self, dm, temp_dir):
        """If a phase is done, intermediate checkpoints should be cleaned."""
        _make_checkpoint_tree(temp_dir, "checkpoints/agent_sft")
        result = dm.session_startup_cleanup()
        # checkpoint-* should be gone, final/ should remain
        remaining = list(
            (dm.base_dir / "checkpoints/agent_sft").glob("checkpoint-*")
        )
        assert len(remaining) == 0
        assert os.path.exists(
            os.path.join(temp_dir, "checkpoints/agent_sft/final")
        )


# ── Phase enum ───────────────────────────────────────────────────────────────


class TestPhaseEnum:
    def test_phase_values(self):
        assert Phase.TOOL_CALLING_SFT.value == "tool_calling_sft"
        assert Phase.EXPORT.value == "export"

    def test_all_phases_defined(self):
        assert len(Phase) == 6


# ── Helper function ──────────────────────────────────────────────────────────


class TestDirSizeGb:
    def test_empty_dir(self, temp_dir):
        assert _dir_size_gb(os.path.join(temp_dir)) == 0.0

    def test_dir_with_files(self, temp_dir):
        with open(os.path.join(temp_dir, "file.bin"), "wb") as f:
            f.write(b"\x00" * (1024 * 1024))  # 1 MB
        from pathlib import Path
        size = _dir_size_gb(Path(temp_dir))
        assert 0.0009 < size < 0.0011  # ~0.001 GB
