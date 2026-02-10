"""Checkpoint management and stage-based recovery with Google Drive sync."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


class CheckpointManager:
    """Manage intermediate checkpoints for resumable processing."""

    def __init__(self, cache_dir: str = "data/.cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, filepath: str) -> str:
        """Get hash of file for cache invalidation."""
        stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()[:12]

    def get_checkpoint_path(self, filename: str, step: str, file_hash: str) -> Path:
        """Get path for a checkpoint file."""
        return self.cache_dir / f"{filename}_{step}_{file_hash}.parquet"

    def get_state_path(self, filename: str, file_hash: str) -> Path:
        """Get path for processing state."""
        return self.cache_dir / f"{filename}_{file_hash}_state.json"

    def save_checkpoint(self, df: pd.DataFrame, filename: str, step: str, file_hash: str) -> None:
        """Save intermediate checkpoint."""
        path = self.get_checkpoint_path(filename, step, file_hash)
        df.to_parquet(path, index=False)

    def load_checkpoint(self, filename: str, step: str, file_hash: str) -> pd.DataFrame | None:
        """Load checkpoint if it exists."""
        path = self.get_checkpoint_path(filename, step, file_hash)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def save_state(self, filename: str, file_hash: str, state: dict) -> None:
        """Save processing state."""
        path = self.get_state_path(filename, file_hash)
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, filename: str, file_hash: str) -> dict | None:
        """Load processing state if it exists."""
        path = self.get_state_path(filename, file_hash)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def cleanup(self, filename: str, file_hash: str) -> None:
        """Remove checkpoints after successful completion."""
        for step in ['clean', 'quality', 'toxicity', 'dedup']:
            path = self.get_checkpoint_path(filename, step, file_hash)
            if path.exists():
                path.unlink()
        state_path = self.get_state_path(filename, file_hash)
        if state_path.exists():
            state_path.unlink()


class StageManager:
    """Manage stage-based recovery with Google Drive sync.

    Stages:
        1. TEXT_CLEAN - Raw text cleaning (PII removal, Unicode fixing)
        2. QUALITY_FILTER - Quality filtering (Gopher, FineWeb)
        3. TOXICITY_FILTER - Toxicity detection and removal
        4. DEDUP - MinHash deduplication
        5. FINAL - Final output ready

    Each stage saves to a separate directory. After each stage completes:
    1. Rsync to Google Drive (if configured)
    2. Previous stage's temp files are deleted
    """

    STAGES = ['text_clean', 'quality_filter', 'toxicity_filter', 'dedup', 'final']

    STAGE_PATTERNS = {
        'text_clean': '*_clean_chunk_*.parquet',
        'quality_filter': '*_filtered_chunk_*.parquet',
        'toxicity_filter': '*_filtered_chunk_*.parquet',
        'dedup': '*_clean.parquet',
        'final': '*.parquet',
    }

    def __init__(self, output_dir: str, drive_dir: str = None, checkpoint_dir: str = None):
        self.output_dir = Path(output_dir)
        self.drive_dir = Path(drive_dir) if drive_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("data/.cache")
        self.stage_dirs = {}

        for stage in self.STAGES[:-1]:
            stage_dir = self.output_dir / f".stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            self.stage_dirs[stage] = stage_dir

        self.stage_dirs['final'] = self.output_dir
        self.state_file = self.output_dir / ".stage_state.json"

        if self.drive_dir:
            self._validate_and_restore_from_drive()

    def _validate_and_restore_from_drive(self) -> None:
        """Validate local state matches actual files, restore from Drive if needed."""
        if not self.drive_dir or not self.drive_dir.exists():
            return

        if not self.state_file.exists():
            print(f"  [No local state found - checking Drive for recovery...]")
            self._smart_restore_from_drive()
            return

        completed = self.get_completed_stages()

        if 'final' in completed:
            local_final = list(self.output_dir.glob('*_clean.parquet'))
            if not local_final:
                print(f"  [State shows 'final' complete but no output files found locally]")
                print(f"  [Triggering Drive recovery...]")
                self.state_file.unlink()
                self._smart_restore_from_drive()
                return

        elif 'toxicity_filter' in completed or 'quality_filter' in completed:
            local_filtered = list(self.checkpoint_dir.glob('*_filtered_chunk_*.parquet'))
            if not local_filtered:
                print(f"  [State shows filters complete but no chunk files found locally]")
                print(f"  [Triggering Drive recovery...]")
                self.state_file.unlink()
                self._smart_restore_from_drive()
                return

        elif 'text_clean' in completed:
            local_clean = list(self.checkpoint_dir.glob('*_clean_chunk_*.parquet'))
            if not local_clean:
                print(f"  [State shows text_clean complete but no chunk files found locally]")
                print(f"  [Triggering Drive recovery...]")
                self.state_file.unlink()
                self._smart_restore_from_drive()
                return

    def _smart_restore_from_drive(self) -> dict:
        """Intelligently restore state and data files from Drive."""
        import shutil
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        result = {
            'restored_files': 0,
            'failed_files': 0,
            'inferred_stages': [],
            'resume_from': 'text_clean',
        }

        if not self.drive_dir or not self.drive_dir.exists():
            print(f"  [No Drive backup found - starting fresh]")
            return result

        print(f"\n  [Analyzing Drive backup for recovery...]")

        drive_inventory = {
            'final': [],
            'filtered': [],
            'clean': [],
        }

        try:
            for f in self.drive_dir.iterdir():
                if f.is_file() and f.suffix == '.parquet':
                    name = f.name
                    if name.endswith('_clean.parquet') and '_chunk_' not in name:
                        drive_inventory['final'].append(f)

            stage_text_clean = self.drive_dir / '.stage_text_clean'
            if stage_text_clean.exists():
                for f in stage_text_clean.iterdir():
                    if f.is_file() and '_clean_chunk_' in f.name:
                        drive_inventory['clean'].append(f)

            for stage_dir_name in ['.stage_quality_filter', '.stage_toxicity_filter']:
                stage_dir = self.drive_dir / stage_dir_name
                if stage_dir.exists():
                    for f in stage_dir.iterdir():
                        if f.is_file() and '_filtered_chunk_' in f.name:
                            if f not in drive_inventory['filtered']:
                                drive_inventory['filtered'].append(f)

        except Exception as e:
            print(f"  [ERROR scanning Drive: {e}]")
            return result

        print(f"  [Drive inventory: {len(drive_inventory['final'])} final, "
              f"{len(drive_inventory['filtered'])} filtered chunks, "
              f"{len(drive_inventory['clean'])} clean chunks]")

        files_to_restore = []

        if drive_inventory['final']:
            for f in drive_inventory['final']:
                files_to_restore.append((f, self.output_dir / f.name))
            print(f"  [Will restore {len(drive_inventory['final'])} final output(s)]")

        if drive_inventory['filtered']:
            for f in drive_inventory['filtered']:
                files_to_restore.append((f, self.checkpoint_dir / f.name))
            print(f"  [Will restore {len(drive_inventory['filtered'])} filtered chunk(s)]")

        if drive_inventory['clean']:
            for f in drive_inventory['clean']:
                files_to_restore.append((f, self.checkpoint_dir / f.name))
            print(f"  [Will restore {len(drive_inventory['clean'])} clean chunk(s)]")

        if not files_to_restore:
            print(f"  [No recoverable files found - starting fresh]")
            return result

        if drive_inventory['final']:
            result['inferred_stages'] = ['text_clean', 'quality_filter', 'toxicity_filter', 'dedup', 'final']
            if drive_inventory['filtered'] or drive_inventory['clean']:
                result['resume_from'] = 'text_clean'
            else:
                result['resume_from'] = None
        elif drive_inventory['filtered']:
            result['inferred_stages'] = ['text_clean', 'quality_filter', 'toxicity_filter']
            result['resume_from'] = 'dedup'
        elif drive_inventory['clean']:
            result['inferred_stages'] = ['text_clean']
            result['resume_from'] = 'quality_filter'

        def restore_file(src_dest):
            src, dest = src_dest
            if dest.exists() and dest.stat().st_size > 0:
                return ('exists', src.name, dest)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                if dest.exists() and dest.stat().st_size > 0:
                    return ('restored', src.name, dest)
                else:
                    if dest.exists():
                        dest.unlink()
                    return ('failed', src.name, 'Empty after copy')
            except Exception as e:
                if dest.exists():
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                return ('failed', src.name, str(e))

        if files_to_restore:
            print(f"  [Restoring {len(files_to_restore)} file(s) to local storage...]")

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(restore_file, fd) for fd in files_to_restore]
                for future in as_completed(futures):
                    status, name, detail = future.result()
                    if status == 'restored':
                        result['restored_files'] += 1
                    elif status == 'exists':
                        pass
                    else:
                        result['failed_files'] += 1
                        print(f"    [ERROR: {name}: {detail}]")

        if result['failed_files'] > 0:
            print(f"  [WARNING: {result['failed_files']} file(s) failed to restore]")

            local_final = list(self.output_dir.glob('*_clean.parquet'))
            local_filtered = list(self.checkpoint_dir.glob('*_filtered_chunk_*.parquet'))
            local_clean = list(self.checkpoint_dir.glob('*_clean_chunk_*.parquet'))

            if local_final:
                result['inferred_stages'] = ['text_clean', 'quality_filter', 'toxicity_filter', 'dedup', 'final']
                result['resume_from'] = None
            elif local_filtered:
                result['inferred_stages'] = ['text_clean', 'quality_filter', 'toxicity_filter']
                result['resume_from'] = 'dedup'
            elif local_clean:
                result['inferred_stages'] = ['text_clean']
                result['resume_from'] = 'quality_filter'
            else:
                result['inferred_stages'] = []
                result['resume_from'] = 'text_clean'
                print(f"  [All restorations failed - starting fresh]")

        state = {
            'completed_stages': result['inferred_stages'],
            'stats': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'restored_from_drive': True,
            'restored_files': result['restored_files'],
            'failed_files': result['failed_files'],
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        if result['resume_from']:
            print(f"  [Recovery complete: {result['restored_files']} files restored, "
                  f"will resume from '{result['resume_from']}']")
        else:
            print(f"  [Recovery complete: {result['restored_files']} files restored, "
                  f"all stages already done]")

        return result

    def get_stage_dir(self, stage: str) -> Path:
        """Get directory for a stage."""
        return self.stage_dirs.get(stage, self.output_dir)

    def get_completed_stages(self) -> list[str]:
        """Get list of completed stages from state file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                return state.get('completed_stages', [])
        return []

    def mark_stage_complete(self, stage: str, stats: dict = None) -> None:
        """Mark a stage as complete and save state."""
        completed = self.get_completed_stages()
        if stage not in completed:
            completed.append(stage)

        state = {
            'completed_stages': completed,
            'last_stage': stage,
            'stats': stats or {},
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"\n  [Stage '{stage}' marked complete]")

    def get_resume_stage(self) -> str:
        """Determine which stage to resume from."""
        completed = self.get_completed_stages()
        for stage in self.STAGES:
            if stage not in completed:
                return stage
        return 'final'

    def sync_to_drive(self, stage: str, max_workers: int = 10) -> int:
        """Rsync stage output to Google Drive."""
        if not self.drive_dir:
            print(f"  [No Drive configured - skipping sync]")
            return 0

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil

        try:
            import pyfastcopy  # noqa: F401
        except ImportError:
            pass

        if stage == 'final':
            src_dir = self.output_dir
            pattern = self.STAGE_PATTERNS.get(stage, '*.parquet')
        elif stage in ['text_clean', 'quality_filter', 'toxicity_filter']:
            src_dir = self.checkpoint_dir
            pattern = self.STAGE_PATTERNS.get(stage, '*.parquet')
        else:
            src_dir = self.output_dir
            pattern = self.STAGE_PATTERNS.get(stage, '*.parquet')

        dst_dir = self.drive_dir / f".stage_{stage}" if stage != 'final' else self.drive_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = list(src_dir.glob(pattern)) if src_dir.exists() else []
        if not files:
            print(f"  [No files to sync for stage '{stage}']")
            return 0

        to_sync = []
        total_size = 0
        for f in files:
            dst = dst_dir / f.name
            src_mtime = f.stat().st_mtime
            dst_mtime = dst.stat().st_mtime if dst.exists() else 0
            if src_mtime > dst_mtime:
                size = f.stat().st_size
                to_sync.append((f, dst, size))
                total_size += size

        if not to_sync:
            print(f"  [Stage '{stage}' already synced to Drive]")
            return 0

        print(f"\n  Syncing stage '{stage}' to Google Drive...")
        print(f"    {len(to_sync)} files, {total_size / (1024**2):.1f} MB, {max_workers} threads")

        def copy_file(args):
            src, dst, size = args
            shutil.copy2(src, dst)
            return src.name, size

        synced = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(copy_file, args): args for args in to_sync}
            for future in as_completed(futures):
                try:
                    name, size = future.result()
                    synced += 1
                except Exception as e:
                    args = futures[future]
                    print(f"    Error syncing {args[0].name}: {e}")

        print(f"    Synced {synced}/{len(to_sync)} files to Drive")

        if self.state_file.exists():
            drive_state = self.drive_dir / ".stage_state.json"
            shutil.copy2(self.state_file, drive_state)

        return synced

    def cleanup_previous_stage(self, current_stage: str) -> None:
        """Delete previous stage's temp files after current stage completes."""
        stage_idx = self.STAGES.index(current_stage) if current_stage in self.STAGES else -1
        if stage_idx <= 0:
            return

        prev_stage = self.STAGES[stage_idx - 1]
        if prev_stage == 'final':
            return

        prev_dir = self.get_stage_dir(prev_stage)
        if prev_dir.exists() and prev_dir != self.output_dir:
            import shutil
            file_count = len(list(prev_dir.glob("*")))
            if file_count > 0:
                print(f"  [Cleaning up stage '{prev_stage}' temp files ({file_count} files)]")
                shutil.rmtree(prev_dir, ignore_errors=True)
                prev_dir.mkdir(parents=True, exist_ok=True)

    def stage_complete_callback(self, stage: str, stats: dict = None,
                                 sync_threads: int = 10, cleanup: bool = True) -> None:
        """Called when a stage completes. Handles sync and cleanup."""
        self.mark_stage_complete(stage, stats)
        self.sync_to_drive(stage, max_workers=sync_threads)
        if cleanup:
            self.cleanup_previous_stage(stage)

    def print_status(self) -> None:
        """Print current stage status."""
        completed = self.get_completed_stages()
        resume_stage = self.get_resume_stage()

        print(f"\n{'='*50}")
        print("STAGE RECOVERY STATUS")
        print(f"{'='*50}")
        for stage in self.STAGES:
            status = "COMPLETE" if stage in completed else "PENDING"
            marker = " <-- resume from here" if stage == resume_stage and status == "PENDING" else ""
            print(f"  {stage}: {status}{marker}")
        print(f"{'='*50}\n")
