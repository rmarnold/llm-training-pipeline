"""Shared training callbacks for the LLM training pipeline."""
from __future__ import annotations

import json
import os
from typing import Any

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from .oom_handler import OOMHandler


class OOMRecoveryCallback(TrainerCallback):
    """Callback for automatic OOM recovery during training.

    When an OOM error occurs during training, this callback:
    1. Catches the error and clears GPU memory
    2. Increases gradient accumulation steps (effectively reducing memory usage)
    3. Allows training to continue

    Note: This works best with gradient accumulation. The callback doubles
    accumulation steps on OOM, which halves effective per-step memory usage.
    """

    def __init__(self, max_accumulation: int = 64) -> None:
        self.max_accumulation = max_accumulation
        self.handler = OOMHandler()
        self.oom_occurred = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.oom_occurred = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.handler.oom_count > 0 and logs is not None:
            logs["oom_recovery/total_events"] = self.handler.oom_count


class CurriculumCallback(TrainerCallback):
    """Callback for curriculum learning - gradually increases sequence length.

    Monitors training steps and triggers checkpoint saves at curriculum stage
    boundaries. When a stage boundary is reached, training stops so data can
    be reloaded with the new sequence length.

    For curriculum learning to work:
    1. Prepare data at different sequence lengths:
       - data/packed/train_512/
       - data/packed/train_1024/
       - data/packed/train_2048/
    2. Set curriculum.data_pattern in config to use placeholders:
       - data_pattern: "data/packed/train_{seq_length}"
    3. Use --curriculum-stage to resume at specific stages
    """

    def __init__(self, curriculum_config: dict[str, Any], output_dir: str = "checkpoints/pretrain") -> None:
        self.schedule = curriculum_config.get('schedule', [])
        self.data_pattern = curriculum_config.get('data_pattern', "data/packed/train_{seq_length}")
        self.auto_stop = curriculum_config.get('auto_stop_at_boundary', True)
        self.output_dir = output_dir
        self.current_idx = 0
        self.stage_changed = False

        if self.schedule:
            self.current_seq_length = self.schedule[0]['seq_length']
        else:
            self.current_seq_length = 2048

    def get_current_stage(self, global_step: int) -> tuple[int, int]:
        """Get the curriculum stage for a given step."""
        for i, stage in enumerate(self.schedule):
            if global_step < stage['steps']:
                return i, stage['seq_length']
        if self.schedule:
            return len(self.schedule) - 1, self.schedule[-1]['seq_length']
        return 0, 2048

    def get_data_path(self, seq_length: int) -> str:
        """Get data path for a given sequence length."""
        return self.data_pattern.format(seq_length=seq_length)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if not self.schedule:
            return

        new_idx, new_seq_length = self.get_current_stage(state.global_step)

        if new_idx > self.current_idx:
            self.current_idx = new_idx
            self.current_seq_length = new_seq_length
            self.stage_changed = True

            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE BOUNDARY REACHED")
            print(f"{'='*60}")
            print(f"Stage {self.current_idx + 1}/{len(self.schedule)}")
            print(f"New sequence length: {new_seq_length} tokens")
            print(f"Step: {state.global_step}")

            if self.auto_stop:
                self._save_curriculum_state(state.global_step)
                print(f"\nSaving checkpoint and stopping for data reload...")
                print(f"Resume with: python scripts/05_pretrain.py --resume_from_checkpoint {args.output_dir}")
                control.should_save = True
                control.should_training_stop = True

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if self.schedule:
            start_idx, start_seq = self.get_current_stage(state.global_step)
            self.current_idx = start_idx
            self.current_seq_length = start_seq

            print(f"\n{'='*60}")
            print(f"CURRICULUM LEARNING")
            print(f"{'='*60}")
            print(f"Current stage: {self.current_idx + 1}/{len(self.schedule)}")
            print(f"Current sequence length: {self.current_seq_length}")
            print(f"Data pattern: {self.data_pattern}")
            print(f"\nSchedule:")
            for i, stage in enumerate(self.schedule):
                marker = ">>>" if i == self.current_idx else "   "
                print(f"{marker} Stage {i+1}: {stage['seq_length']} tokens @ step {stage['steps']}")

    def _save_curriculum_state(self, global_step: int) -> None:
        """Save curriculum state for resumption."""
        state_path = os.path.join(self.output_dir, "curriculum_state.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump({
                "current_stage": self.current_idx,
                "current_seq_length": self.current_seq_length,
                "global_step": global_step,
                "schedule": self.schedule
            }, f, indent=2)
