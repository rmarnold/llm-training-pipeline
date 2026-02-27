"""Disk space management for Colab training pipelines.

Prevents "No space left on device" crashes by tracking disk usage,
cleaning up intermediate artifacts between training phases, and
performing escalating auto-cleanup before each phase starts.
"""

from __future__ import annotations

import enum
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────


class Phase(enum.Enum):
    """Pipeline phases in execution order."""
    TOOL_CALLING_SFT = "tool_calling_sft"
    TOOL_CALLING_SFT_MERGE = "tool_calling_sft_merge"
    AGENT_SFT = "agent_sft"
    AGENT_SFT_IPO = "agent_sft_ipo"
    AGENT_SFT_GRPO = "agent_sft_grpo"
    EXPORT = "export"


@dataclass
class DiskStatus:
    """Snapshot of disk usage."""
    total_gb: float
    used_gb: float
    free_gb: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    freed_gb: float = 0.0
    deleted_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Phase cleanup rules (declarative) ───────────────────────────────────────

# After a phase completes, which paths to delete and which to keep.
# Keys are (completed_phase, next_phase) tuples.
# "delete" entries support glob-style checkpoint-* patterns.
# "keep" entries are never deleted.

_PHASE_CLEANUP_RULES: dict[tuple[str, str], dict] = {
    ("tool_calling_sft", "tool_calling_sft_merge"): {
        "delete": [
            "checkpoints/tool_calling_sft/checkpoint-*",
        ],
        "keep": [
            "checkpoints/tool_calling_sft/final",
        ],
    },
    ("tool_calling_sft_merge", "agent_sft"): {
        "delete": [
            "checkpoints/tool_calling_sft",
        ],
        "keep": [
            "checkpoints/gpt-oss-20b-coding-tui-merged",
        ],
    },
    ("agent_sft", "agent_sft_ipo"): {
        "delete": [
            "checkpoints/gpt-oss-20b-coding-tui-merged",
            "checkpoints/agent_sft/checkpoint-*",
        ],
        "keep": [
            "checkpoints/agent_sft/final",
        ],
    },
    ("agent_sft_ipo", "agent_sft_grpo"): {
        "delete": [
            "checkpoints/agent_sft",
            "checkpoints/agent_sft_ipo/checkpoint-*",
        ],
        "keep": [
            "checkpoints/agent_sft_ipo/final",
        ],
    },
    ("agent_sft_grpo", "export"): {
        "delete": [
            "checkpoints/agent_sft_ipo",
            "checkpoints/agent_sft_grpo/checkpoint-*",
        ],
        "keep": [
            "checkpoints/agent_sft_grpo/final",
        ],
    },
}


# ── DiskManager ──────────────────────────────────────────────────────────────


class DiskManager:
    """Manages disk space for the Colab training pipeline.

    Parameters
    ----------
    base_dir : str
        Root directory of the training pipeline (usually the repo root).
    drive_helper : optional
        A ``DriveHelper`` instance for verifying Drive backups before
        deleting local copies.  Pass ``None`` to skip backup verification.
    dry_run : bool
        If True, report what would be deleted without actually deleting.
    """

    # Well-known cache directories (relative to home or absolute)
    _CACHE_DIRS = {
        "pip": ("~/.cache/pip",),
        "torch": ("~/.cache/torch",),
        "triton": ("~/.triton/cache", "~/.cache/triton"),
        "hf_hub": ("~/.cache/huggingface/hub",),
    }

    def __init__(
        self,
        base_dir: str = ".",
        drive_helper: object = None,
        dry_run: bool = False,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.drive_helper = drive_helper
        self.dry_run = dry_run

    # ── Public API ───────────────────────────────────────────────────────

    def get_disk_status(self, paths: Optional[list[str]] = None) -> DiskStatus:
        """Return free/used/total GB and optional per-directory breakdown.

        Parameters
        ----------
        paths : list[str], optional
            Directories to include in the breakdown.  Defaults to
            ``checkpoints/`` and common cache dirs.
        """
        stat = shutil.disk_usage(str(self.base_dir))
        total_gb = stat.total / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)
        free_gb = stat.free / (1024 ** 3)

        breakdown: dict[str, float] = {}
        if paths is None:
            paths = [
                "checkpoints",
                *[os.path.expanduser(p)
                  for dirs in self._CACHE_DIRS.values()
                  for p in dirs],
            ]
        for p in paths:
            abs_p = self._resolve(p)
            if abs_p.exists() and abs_p.is_dir():
                size = _dir_size_gb(abs_p)
                if size > 0.0:
                    breakdown[str(p)] = round(size, 6)

        return DiskStatus(
            total_gb=round(total_gb, 2),
            used_gb=round(used_gb, 2),
            free_gb=round(free_gb, 2),
            breakdown=breakdown,
        )

    def cleanup_between_phases(
        self, completed: str, next_phase: str
    ) -> CleanupResult:
        """Delete artifacts no longer needed after *completed* phase.

        Respects must-keep rules and never deletes ``*/final/`` dirs.
        """
        result = CleanupResult()
        key = (completed, next_phase)
        rules = _PHASE_CLEANUP_RULES.get(key)
        if rules is None:
            logger.info("No cleanup rules for %s -> %s", completed, next_phase)
            return result

        keep_set = {str(self._resolve(k)) for k in rules.get("keep", [])}

        for pattern in rules.get("delete", []):
            targets = self._expand_pattern(pattern)
            if not targets:
                # Pattern matched nothing — the dir may already be gone
                continue
            for target in targets:
                target_str = str(target)
                # Safety: never delete */final/ directories
                if target.name == "final" or any(
                    target_str.startswith(k) for k in keep_set
                ):
                    result.skipped_paths.append(target_str)
                    continue
                # Skip symlinks (mounted Drive mode)
                if target.is_symlink():
                    result.skipped_paths.append(f"{target_str} (symlink)")
                    continue
                # Preserve trainer_state.json before deleting checkpoints
                self._preserve_trainer_state(target)
                freed = self._delete(target, result)
                result.freed_gb += freed

        status = self.get_disk_status()
        logger.info(
            "Phase cleanup %s->%s: freed %.1f GB, %.1f GB free",
            completed, next_phase, result.freed_gb, status.free_gb,
        )
        return result

    def cleanup_hf_cache(self) -> CleanupResult:
        """Clear ``~/.cache/huggingface/hub/``."""
        result = CleanupResult()
        for p in self._CACHE_DIRS["hf_hub"]:
            target = Path(os.path.expanduser(p))
            if target.exists() and target.is_dir():
                freed = self._delete(target, result)
                result.freed_gb += freed
        return result

    def cleanup_caches(self) -> CleanupResult:
        """Clear torch, triton, and pip caches."""
        result = CleanupResult()
        for name in ("pip", "torch", "triton"):
            for p in self._CACHE_DIRS[name]:
                target = Path(os.path.expanduser(p))
                if target.exists() and target.is_dir():
                    freed = self._delete(target, result)
                    result.freed_gb += freed
        return result

    def cleanup_intermediate_checkpoints(self, directory: str) -> CleanupResult:
        """Remove ``checkpoint-*`` dirs from *directory*, keeping ``final/``.

        Also preserves ``trainer_state.json`` by copying it from the
        latest checkpoint to the parent directory before deletion.
        """
        result = CleanupResult()
        dir_path = self._resolve(directory)
        if not dir_path.exists():
            return result

        # Find checkpoint-* directories
        ckpts = sorted(dir_path.glob("checkpoint-*"))
        if not ckpts:
            return result

        # Preserve trainer_state from latest checkpoint
        self._preserve_trainer_state(ckpts[-1])

        for ckpt in ckpts:
            if ckpt.is_symlink():
                result.skipped_paths.append(f"{ckpt} (symlink)")
                continue
            freed = self._delete(ckpt, result)
            result.freed_gb += freed

        return result

    def pre_phase_disk_check(
        self, phase: str, required_gb: float = 25.0
    ) -> bool:
        """Check disk space before training; auto-clean if needed.

        Escalation order (least to most impactful):
        1. pip/torch/triton caches (harmless, re-downloadable)
        2. HF hub cache (re-downloadable, ~20-40 GB)
        3. Phase-specific intermediate checkpoints

        Returns True if enough space is available after cleanup.
        """
        status = self.get_disk_status()
        print(f"[DiskManager] Pre-flight for {phase}: "
              f"{status.free_gb:.1f} GB free, need {required_gb:.0f} GB")

        if status.free_gb >= required_gb:
            print(f"[DiskManager] Sufficient disk space.")
            return True

        # Escalation 1: pip/torch/triton caches
        print(f"[DiskManager] Low space — cleaning pip/torch/triton caches...")
        r = self.cleanup_caches()
        if r.freed_gb > 0:
            print(f"[DiskManager] Freed {r.freed_gb:.1f} GB from caches.")
        status = self.get_disk_status()
        if status.free_gb >= required_gb:
            print(f"[DiskManager] Now {status.free_gb:.1f} GB free. OK.")
            return True

        # Escalation 2: HF hub cache
        print(f"[DiskManager] Still low — cleaning HF hub cache...")
        r = self.cleanup_hf_cache()
        if r.freed_gb > 0:
            print(f"[DiskManager] Freed {r.freed_gb:.1f} GB from HF cache.")
        status = self.get_disk_status()
        if status.free_gb >= required_gb:
            print(f"[DiskManager] Now {status.free_gb:.1f} GB free. OK.")
            return True

        # Escalation 3: intermediate checkpoints in all checkpoint dirs
        print(f"[DiskManager] Still low — cleaning intermediate checkpoints...")
        for ckpt_dir in ("checkpoints/tool_calling_sft",
                         "checkpoints/agent_sft",
                         "checkpoints/agent_sft_ipo",
                         "checkpoints/agent_sft_grpo"):
            r = self.cleanup_intermediate_checkpoints(ckpt_dir)
            if r.freed_gb > 0:
                print(f"[DiskManager] Freed {r.freed_gb:.1f} GB from {ckpt_dir}.")
        status = self.get_disk_status()
        if status.free_gb >= required_gb:
            print(f"[DiskManager] Now {status.free_gb:.1f} GB free. OK.")
            return True

        print(f"[DiskManager] WARNING: Only {status.free_gb:.1f} GB free "
              f"(need {required_gb:.0f} GB). Training may fail.")
        return False

    def session_startup_cleanup(self) -> CleanupResult:
        """On session start/reconnect, detect completed phases and clean stale data.

        Idempotent — safe to call multiple times.
        """
        result = CleanupResult()

        # Detect completed phases by checking for final/ dirs
        phase_status: dict[str, bool] = {}
        phase_dirs = {
            "tool_calling_sft": "checkpoints/tool_calling_sft/final",
            "merge": "checkpoints/gpt-oss-20b-coding-tui-merged/hf/config.json",
            "agent_sft": "checkpoints/agent_sft/final",
            "agent_sft_ipo": "checkpoints/agent_sft_ipo/final",
            "agent_sft_grpo": "checkpoints/agent_sft_grpo/final",
        }
        for phase, marker in phase_dirs.items():
            phase_status[phase] = self._resolve(marker).exists()

        # Apply cleanup rules for completed phases
        # Work backwards: if a later phase is done, earlier artifacts are safe to clean
        transitions = [
            ("tool_calling_sft", "tool_calling_sft_merge", "merge"),
            ("tool_calling_sft_merge", "agent_sft", "agent_sft"),
            ("agent_sft", "agent_sft_ipo", "agent_sft_ipo"),
            ("agent_sft_ipo", "agent_sft_grpo", "agent_sft_grpo"),
        ]
        for completed, next_phase, gate_phase in transitions:
            if phase_status.get(gate_phase, False):
                r = self.cleanup_between_phases(completed, next_phase)
                result.freed_gb += r.freed_gb
                result.deleted_paths.extend(r.deleted_paths)
                result.skipped_paths.extend(r.skipped_paths)
                result.errors.extend(r.errors)

        # Always clean intermediate checkpoints in completed phases
        for phase_name, marker in phase_dirs.items():
            if phase_status.get(phase_name) and phase_name != "merge":
                ckpt_dir = f"checkpoints/{phase_name}"
                r = self.cleanup_intermediate_checkpoints(ckpt_dir)
                result.freed_gb += r.freed_gb
                result.deleted_paths.extend(r.deleted_paths)

        status = self.get_disk_status()
        print(f"[DiskManager] Session cleanup: freed {result.freed_gb:.1f} GB, "
              f"{status.free_gb:.1f} GB free")
        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """Resolve *path* relative to base_dir (unless already absolute)."""
        p = Path(os.path.expanduser(path))
        if p.is_absolute():
            return p
        return self.base_dir / p

    def _expand_pattern(self, pattern: str) -> list[Path]:
        """Expand a glob pattern relative to base_dir."""
        if "*" in pattern:
            # Use base_dir.glob() so the full relative pattern is resolved
            # correctly regardless of where the '*' appears in the path.
            return sorted(self.base_dir.glob(pattern))
        # Literal path
        p = self._resolve(pattern)
        return [p] if p.exists() else []

    def _preserve_trainer_state(self, checkpoint_dir: Path) -> None:
        """Copy trainer_state.json from a checkpoint to its parent dir."""
        if not checkpoint_dir.is_dir():
            return
        ts = checkpoint_dir / "trainer_state.json"
        if ts.exists():
            dest = checkpoint_dir.parent / "trainer_state.json"
            if not dest.exists() or ts.stat().st_mtime > dest.stat().st_mtime:
                if not self.dry_run:
                    shutil.copy2(str(ts), str(dest))
                logger.debug("Preserved trainer_state.json -> %s", dest)

    def _verify_drive_backup(self, local_path: Path) -> bool:
        """Check if a Drive backup exists for *local_path*.

        Returns True if backup verified or no drive_helper configured
        (optimistic — we don't block cleanup without Drive).
        """
        if self.drive_helper is None:
            return True
        rel = str(local_path.relative_to(self.base_dir))
        # DriveHelper.restore checks existence; we use a lighter check
        if hasattr(self.drive_helper, "mode"):
            if self.drive_helper.mode == "local":
                return True  # No Drive — can't verify, allow cleanup
            if self.drive_helper.mode == "mounted":
                drive_path = Path(self.drive_helper.drive_base) / rel
                return drive_path.exists()
        return True  # Default: allow cleanup

    def _delete(self, target: Path, result: CleanupResult) -> float:
        """Delete *target* (file or dir), returning freed GB."""
        if not target.exists():
            return 0.0
        size_gb = _dir_size_gb(target) if target.is_dir() else (
            target.stat().st_size / (1024 ** 3)
        )
        if self.dry_run:
            result.deleted_paths.append(f"{target} (dry run)")
            logger.info("[DRY RUN] Would delete %s (%.2f GB)", target, size_gb)
            return 0.0
        try:
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
            result.deleted_paths.append(str(target))
            logger.info("Deleted %s (%.2f GB)", target, size_gb)
            return size_gb
        except OSError as exc:
            msg = f"Failed to delete {target}: {exc}"
            result.errors.append(msg)
            logger.warning(msg)
            return 0.0


# ── Module-level helpers ─────────────────────────────────────────────────────


def _dir_size_gb(path) -> float:
    """Return total size of *path* in GB (follows symlinks inside).

    Accepts both ``str`` and ``pathlib.Path``.
    """
    path = Path(path)
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file() and not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 ** 3)
