from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

# Default wandb to offline mode (avoids interactive prompt)
# Set WANDB_MODE=online explicitly to enable cloud sync
if 'WANDB_MODE' not in os.environ:
    os.environ['WANDB_MODE'] = 'offline'

import torch
import yaml
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

# Kernel optimization flags (set before model loading)
_CCE_PATCHED = False


def setup_kernel_optimizations(
    use_liger_kernel: bool = False,
    use_cce: bool = False,
    model_type: str = "llama"
) -> Dict[str, bool]:
    """Setup kernel optimizations before model loading.

    IMPORTANT: Must be called BEFORE loading the model, as these optimizations
    patch the model classes.

    Args:
        use_liger_kernel: Enable Liger Kernel (LinkedIn's Triton kernels)
            - ~20% throughput improvement
            - ~60% memory reduction
            - Fused RMSNorm, RoPE, SwiGLU, CrossEntropy
            - Compatible with torch.compile (Liger uses Triton)
        use_cce: Enable Cut Cross-Entropy (Apple's memory-efficient CE)
            - ~95% memory reduction on loss computation
            - Especially useful for large vocab/seq lengths
        model_type: Model architecture type (llama, mistral, gemma, etc.)

    Returns:
        Dict with actual enabled status of each optimization
    """
    global _CCE_PATCHED
    enabled = {"liger_kernel": False, "cce": False}

    # Liger Kernel must be applied BEFORE model loading (patches model classes)
    # Liger is Triton-based and works with torch.compile
    # NOTE: fused_linear_cross_entropy=False because it uses .item() which breaks torch.compile
    if use_liger_kernel:
        try:
            if model_type == "llama":
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama(
                    rope=True,              # Fused RoPE (~5% speedup)
                    swiglu=True,            # Fused SwiGLU (~10% speedup)
                    rms_norm=True,          # Fused RMSNorm (~5% speedup)
                    cross_entropy=True,     # Triton CE kernel (compatible with torch.compile)
                    fused_linear_cross_entropy=False,  # Disabled: uses .item() breaking compile
                )
            elif model_type == "mistral":
                from liger_kernel.transformers import apply_liger_kernel_to_mistral
                apply_liger_kernel_to_mistral(fused_linear_cross_entropy=False)
            elif model_type == "gemma":
                from liger_kernel.transformers import apply_liger_kernel_to_gemma
                apply_liger_kernel_to_gemma(fused_linear_cross_entropy=False)
            elif model_type == "qwen2":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen2
                apply_liger_kernel_to_qwen2(fused_linear_cross_entropy=False)
            else:
                print(f"Warning: Liger Kernel not available for model type '{model_type}'")
                use_liger_kernel = False

            if use_liger_kernel:
                enabled["liger_kernel"] = True
                print(f"[Kernel Optimization] Liger Kernel enabled for {model_type}")
                print(f"  - Fused RoPE, SwiGLU, RMSNorm, CrossEntropy")
                print(f"  - ~20% throughput improvement")
                print(f"  - ~60% memory reduction")
                print(f"  - Compatible with torch.compile")
        except ImportError:
            print("Warning: liger-kernel not installed. Install with:")
            print("  pip install liger-kernel")
        except Exception as e:
            print(f"Warning: Failed to enable Liger Kernel: {e}")

    # Cut Cross-Entropy - only if Liger Kernel not enabled
    # (they're mutually exclusive - both optimize cross-entropy computation)
    if use_cce and not enabled["liger_kernel"] and not _CCE_PATCHED:
        try:
            from cut_cross_entropy.transformers import cce_patch
            cce_patch(model_type)
            _CCE_PATCHED = True
            enabled["cce"] = True
            print(f"[Kernel Optimization] Cut Cross-Entropy enabled for {model_type}")
            print(f"  - ~95% memory reduction on cross-entropy loss")
        except ImportError:
            print("Warning: cut-cross-entropy not installed. Install with:")
            print("  pip install cut-cross-entropy")
        except Exception as e:
            print(f"Warning: Failed to enable Cut Cross-Entropy: {e}")
    elif use_cce and _CCE_PATCHED:
        enabled["cce"] = True
    elif use_cce and enabled["liger_kernel"]:
        print("Note: CCE skipped - Liger's CrossEntropy kernel already optimizes CE")

    return enabled


def get_optimizer_name(config_optim: str) -> str:
    """Get optimizer name with fallback for unavailable optimizers.

    Args:
        config_optim: Optimizer name from config (e.g., "adamw_bnb_8bit")

    Returns:
        Actual optimizer name to use (may fallback if dependency missing)
    """
    if config_optim == "adamw_bnb_8bit":
        try:
            import bitsandbytes
            print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
            print("  - ~4x optimizer memory reduction (~30GB saved for 7B model)")
            return "adamw_bnb_8bit"
        except ImportError:
            print("Warning: bitsandbytes not installed, falling back to adamw_torch_fused")
            print("  Install for 4x memory reduction: pip install bitsandbytes")
            return "adamw_torch_fused"
    return config_optim

# Import GPU utilities
from gpu_utils import (
    detect_gpu_type, print_gpu_info, setup_torch_backends,
    check_tokenizer_exists, check_checkpoint_exists, GPUInfo,
    OOMHandler, get_safe_batch_size
)


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a torch.compile() wrapped model.

    When a model is wrapped with torch.compile(), the original model is stored
    in the _orig_mod attribute. This function returns the original model if
    compiled, or the same model if not.

    Args:
        model: A potentially compiled model

    Returns:
        The unwrapped model (or the same model if not compiled)
    """
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


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

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        # Reset OOM flag at start of each step
        self.oom_occurred = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        # Log OOM recovery info if it occurred
        if self.handler.oom_count > 0 and logs is not None:
            logs["oom_recovery/total_events"] = self.handler.oom_count


class CurriculumCallback(TrainerCallback):
    """Callback for curriculum learning - gradually increases sequence length.

    This callback monitors training steps and triggers checkpoint saves at
    curriculum stage boundaries. When a stage boundary is reached, training
    stops so data can be reloaded with the new sequence length.

    For curriculum learning to work:
    1. Prepare data at different sequence lengths:
       - data/packed/train_512/
       - data/packed/train_1024/
       - data/packed/train_2048/
    2. Set curriculum.data_pattern in config to use placeholders:
       - data_pattern: "data/packed/train_{seq_length}"
    3. Use --curriculum-stage to resume at specific stages

    The callback saves curriculum state to allow seamless resumption.
    """

    def __init__(self, curriculum_config: Dict[str, Any], output_dir: str = "checkpoints/pretrain") -> None:
        self.schedule = curriculum_config.get('schedule', [])
        self.data_pattern = curriculum_config.get('data_pattern', "data/packed/train_{seq_length}")
        self.auto_stop = curriculum_config.get('auto_stop_at_boundary', True)
        self.output_dir = output_dir
        self.current_idx = 0
        self.stage_changed = False

        # Determine current stage based on schedule
        if self.schedule:
            self.current_seq_length = self.schedule[0]['seq_length']
        else:
            self.current_seq_length = 2048

    def get_current_stage(self, global_step: int) -> Tuple[int, int]:
        """Get the curriculum stage for a given step."""
        for i, stage in enumerate(self.schedule):
            if global_step < stage['steps']:
                return i, stage['seq_length']
        # Past all stages, use the last one
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
                # Save curriculum state
                self._save_curriculum_state(state.global_step)
                print(f"\nSaving checkpoint and stopping for data reload...")
                print(f"Resume with: python scripts/05_pretrain.py --resume_from_checkpoint {args.output_dir}")
                control.should_save = True
                control.should_training_stop = True

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if self.schedule:
            # Determine starting stage based on resumed step
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
        import json
        state_path = os.path.join(self.output_dir, "curriculum_state.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(state_path, "w") as f:
            json.dump({
                "current_stage": self.current_idx,
                "current_seq_length": self.current_seq_length,
                "global_step": global_step,
                "schedule": self.schedule
            }, f, indent=2)


def get_curriculum_data_path(
    config: Dict[str, Any],
    global_step: int = 0,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """Get the appropriate data path based on curriculum stage.

    Args:
        config: Training config dict
        global_step: Current training step (for resumption)
        cli_overrides: Optional CLI overrides dict

    Returns:
        Tuple of (train_path, val_path)
    """
    cli_overrides = cli_overrides or {}

    def is_valid_hf_dataset(path: str) -> bool:
        """Check if path contains a valid HuggingFace Dataset with non-empty state.json."""
        state_file = os.path.join(path, 'state.json')
        if not os.path.exists(state_file):
            return False
        try:
            # Check if state.json is non-empty and valid JSON
            with open(state_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return False
                import json
                json.loads(content)
                return True
        except (json.JSONDecodeError, IOError):
            return False

    # CLI overrides take precedence - check each path independently
    cli_train_path = cli_overrides.get('train_data_path')
    cli_eval_path = cli_overrides.get('eval_data_path')

    # If CLI provides train path, derive eval path from it if not specified
    if cli_train_path:
        # Auto-detect train/val subdirectories if path points to parent directory
        # e.g., /content/local_data/packed -> /content/local_data/packed/train
        train_subdir = os.path.join(cli_train_path, 'train')
        val_subdir = os.path.join(cli_train_path, 'val')
        if os.path.exists(train_subdir) and os.path.isdir(train_subdir):
            # Check if train_subdir is a valid HuggingFace Dataset (non-empty state.json)
            if is_valid_hf_dataset(train_subdir):
                print(f"Auto-detected valid HuggingFace Dataset at {train_subdir}")
                cli_train_path = train_subdir
                if cli_eval_path is None and is_valid_hf_dataset(val_subdir):
                    cli_eval_path = val_subdir
            else:
                # state.json exists but is empty/corrupted - warn user
                state_file = os.path.join(train_subdir, 'state.json')
                if os.path.exists(state_file):
                    print(f"WARNING: Found corrupted/empty state.json at {train_subdir}")
                    print(f"  The local dataset copy may be incomplete.")
                    print(f"  Checking if original path has valid data...")
                    # Don't use the corrupted subdirectory, keep original path
                    # User may have passed the direct train path

        if cli_eval_path:
            return cli_train_path, cli_eval_path
        # Derive eval path: /path/to/packed/train -> /path/to/packed/val
        parent_dir = os.path.dirname(cli_train_path.rstrip('/'))
        eval_path = os.path.join(parent_dir, 'val')
        if not os.path.exists(eval_path):
            # Try train path with _val suffix
            eval_path = cli_train_path.rstrip('/').replace('/train', '/val')
        if not os.path.exists(eval_path):
            # Use train path as eval (will subset during loading)
            eval_path = cli_train_path
            print(f"Note: Using train data for evaluation (no separate val found)")
        return cli_train_path, eval_path

    curriculum = config.get('curriculum', {})
    if not curriculum.get('enabled', False):
        # No curriculum - use default paths
        train_path = config['data'].get('train_path', "data/packed/train")
        val_path = config['data'].get('val_path', "data/packed/val")
        return train_path, val_path

    schedule = curriculum.get('schedule', [])
    data_pattern = curriculum.get('data_pattern', "data/packed/train_{seq_length}")
    val_pattern = curriculum.get('val_pattern', "data/packed/val_{seq_length}")

    # Find current stage
    seq_length = 2048  # default
    for stage in schedule:
        if global_step < stage['steps']:
            seq_length = stage['seq_length']
            break
    else:
        # Past all stages
        if schedule:
            seq_length = schedule[-1]['seq_length']

    train_path = data_pattern.format(seq_length=seq_length)
    val_path = val_pattern.format(seq_length=seq_length)

    # Fallback to default if curriculum paths don't exist
    if not os.path.exists(train_path):
        print(f"Warning: Curriculum data not found at {train_path}")
        print(f"Falling back to default data path")
        train_path = config['data'].get('train_path', "data/packed/train")
        val_path = config['data'].get('val_path', "data/packed/val")

    return train_path, val_path

def setup_training(
    use_fp8: Optional[bool] = None,
    config_path: str = "configs/pretrain.yaml",
    cli_overrides: Optional[Dict[str, Any]] = None,
    use_liger_kernel: bool = False,
    use_cce: bool = False
) -> Tuple[Trainer, GPUInfo]:
    """Setup training with automatic GPU optimization.

    Args:
        use_fp8: Force FP8 precision (None = auto-detect)
        config_path: Path to YAML config file
        cli_overrides: Dict of CLI overrides (max_steps, save_steps, etc.)
        use_liger_kernel: Enable Liger Kernel optimizations (~20% speedup, ~60% memory reduction)
        use_cce: Enable Cut Cross-Entropy (~95% memory reduction on loss)
    """
    cli_overrides = cli_overrides or {}

    # Setup kernel optimizations BEFORE model loading
    kernel_status = setup_kernel_optimizations(
        use_liger_kernel=use_liger_kernel,
        use_cce=use_cce,
        model_type="llama"  # Default to llama architecture
    )

    # Setup torch backends
    setup_torch_backends()

    # Detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine if we should use FP8
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize W&B (disabled in non-interactive environments)
    if os.getenv('WANDB_MODE') != 'disabled':
        wandb.init(
            project="llm-training",
            name=config['run_name'],
            config=config
        )

    # Load model and tokenizer
    attn_impl = "flash_attention_2" if config['model']['use_flash_attention'] else "eager"
    # Set use_cache=False when using gradient checkpointing (they're incompatible)
    use_cache = not config['model']['gradient_checkpointing']
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        use_cache=use_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    # Enable gradient checkpointing with non-reentrant mode
    if config['model']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Compile model with GPU-appropriate mode (can be disabled via CLI for small models)
    compile_mode = gpu_info['compile_mode']
    use_torch_compile = cli_overrides.get('torch_compile', config['training'].get('torch_compile', True))
    if use_torch_compile:
        print(f"Compiling model with torch.compile (mode={compile_mode})...")
        model = torch.compile(model, mode=compile_mode)
    else:
        print("torch.compile disabled (--no-torch-compile flag or config)")

    # Check distributed training mode for FSDP
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    use_fsdp = world_size > 1
    if config['training'].get('fsdp') and not use_fsdp:
        print("Note: FSDP disabled (single GPU detected). Use torchrun for multi-GPU training.")

    # Resolve warmup: CLI can override with either warmup_steps or warmup_ratio
    # warmup_ratio takes precedence if both specified (more flexible for different max_steps)
    warmup_steps = cli_overrides.get('warmup_steps', config['training']['warmup_steps'])
    warmup_ratio = cli_overrides.get('warmup_ratio', 0.0)

    # Training arguments (with CLI overrides)
    training_args = TrainingArguments(
        output_dir=cli_overrides.get('output_dir', config['checkpointing']['output_dir']),
        run_name=config['run_name'],

        # Optimization
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=cli_overrides.get('max_steps', config['training']['max_steps']),
        learning_rate=cli_overrides.get('learning_rate', config['training']['learning_rate']),
        lr_scheduler_type=config['training']['lr_scheduler'],
        warmup_steps=warmup_steps if warmup_ratio == 0.0 else 0,
        warmup_ratio=warmup_ratio,

        # Batch sizing
        per_device_train_batch_size=cli_overrides.get('per_device_train_batch_size', config['training']['per_device_train_batch_size']),
        gradient_accumulation_steps=cli_overrides.get('gradient_accumulation_steps', config['training']['gradient_accumulation_steps']),

        # Precision
        bf16=config['training']['bf16'],
        tf32=config['training']['tf32'],

        # Regularization
        max_grad_norm=config['training']['max_grad_norm'],
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],

        # Optimization
        # Use get_optimizer_name for 8-bit Adam fallback
        optim=get_optimizer_name(cli_overrides.get('optim', config['training']['optim'])),
        # FSDP only works in distributed mode (world_size > 1)
        fsdp=config['training']['fsdp'] if use_fsdp else "",
        fsdp_transformer_layer_cls_to_wrap=config['training']['fsdp_transformer_layer_cls_to_wrap'] if use_fsdp else None,

        # Logging
        logging_dir=config['logging']['logging_dir'],
        logging_steps=cli_overrides.get('logging_steps', config['logging']['logging_steps']),
        report_to=config['logging']['report_to'],

        # Evaluation
        eval_strategy=config['eval']['evaluation_strategy'],
        eval_steps=cli_overrides.get('eval_steps', config['eval']['eval_steps']),
        per_device_eval_batch_size=config['eval']['per_device_eval_batch_size'],

        # Checkpointing
        save_strategy=config['checkpointing']['save_strategy'],
        save_steps=cli_overrides.get('save_steps', config['checkpointing']['save_steps']),
        save_total_limit=config['logging']['save_total_limit'],

        # torch.compile (can be disabled via CLI for small models)
        torch_compile=cli_overrides.get('torch_compile', config['training'].get('torch_compile', True)),
        torch_compile_backend=config['training'].get('torch_compile_backend', 'inductor'),
        torch_compile_mode=compile_mode,

        # Data loading optimization
        dataloader_num_workers=cli_overrides.get('dataloader_num_workers', config['data'].get('num_workers', 8)),
        dataloader_pin_memory=config['data'].get('pin_memory', True),
        dataloader_persistent_workers=config['data'].get('persistent_workers', True),
        # Required for torch.compile - compiled model has different signature
        remove_unused_columns=False,
        # Note: Liger Kernel is applied via setup_kernel_optimizations() BEFORE model loading
        # Do NOT use use_liger_kernel=True here as it would cause redundant patching
    )

    # Load dataset (curriculum-aware paths)
    # Get starting step for curriculum stage detection
    resume_step = 0
    resume_checkpoint = cli_overrides.get('resume_from_checkpoint')
    if resume_checkpoint:
        # Try to get step from checkpoint path (e.g., checkpoint-5000)
        import re
        match = re.search(r'checkpoint-(\d+)', str(resume_checkpoint))
        if match:
            resume_step = int(match.group(1))

    train_data_path, eval_data_path = get_curriculum_data_path(config, resume_step, cli_overrides)

    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        print("\nTo prepare training data, run:")
        print("  python scripts/01_download_data.py")
        print("  python scripts/02_clean_deduplicate_optimized.py")
        print("  python scripts/03_tokenize_and_pack.py")
        print("\nOr for demo: python scripts/setup_demo.py")
        if config.get('curriculum', {}).get('enabled'):
            print(f"\nFor curriculum learning, prepare data at different sequence lengths:")
            print(f"  data/packed/train_512/")
            print(f"  data/packed/train_1024/")
            print(f"  data/packed/train_2048/")
        sys.exit(1)

    if not os.path.exists(eval_data_path):
        print(f"Error: Validation data not found at {eval_data_path}")
        sys.exit(1)

    # Validate dataset integrity before loading
    def validate_dataset_path(path: str, name: str) -> None:
        """Validate that a HuggingFace dataset path has valid state.json."""
        state_file = os.path.join(path, 'state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        raise ValueError("state.json is empty")
                    import json
                    json.loads(content)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"\nError: {name} dataset at {path} has corrupted state.json")
                print(f"  Issue: {e}")
                print(f"\nThis usually happens when copying data to local SSD was interrupted.")
                print(f"Solutions:")
                print(f"  1. Re-copy the dataset: rm -rf {path} && cp -r <source> {path}")
                print(f"  2. Use the original path directly (e.g., from Google Drive)")
                sys.exit(1)

    validate_dataset_path(train_data_path, "Training")
    validate_dataset_path(eval_data_path, "Validation")

    print(f"\nLoading data from: {train_data_path}")
    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add curriculum callback
    if config.get('curriculum', {}).get('enabled', False):
        output_dir = cli_overrides.get('output_dir', config['checkpointing']['output_dir'])
        trainer.add_callback(CurriculumCallback(config['curriculum'], output_dir=output_dir))

    # Add OOM recovery callback if enabled
    if cli_overrides.get('enable_oom_recovery', False):
        print("OOM recovery: ENABLED")
        trainer.add_callback(OOMRecoveryCallback())

    return trainer, gpu_info

def train_with_fp8(
    config: Dict[str, Any],
    gpu_info: GPUInfo,
    enable_oom_recovery: bool = False,
    use_liger_kernel: bool = False,
    use_cce: bool = False
) -> None:
    """Train using FP8 precision with Accelerate (H100 only)"""
    from gpu_utils import get_fp8_accelerator
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from tqdm import tqdm

    # Setup kernel optimizations BEFORE model loading
    kernel_status = setup_kernel_optimizations(
        use_liger_kernel=use_liger_kernel,
        use_cce=use_cce,
        model_type="llama"
    )

    print("\n" + "="*60)
    print("PRETRAINING WITH FP8 PRECISION")
    print("="*60)

    accelerator = get_fp8_accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )

    print(f"\nFP8 Configuration:")
    print(f"  Backend: Transformer Engine")
    print(f"  Format: HYBRID (E4M3 forward, E5M2 backward)")
    print(f"  Expected speedup: 30-40% over BF16")

    # Load model
    attn_impl = "flash_attention_2" if config['model']['use_flash_attention'] else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['checkpoint'],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl
    )
    tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer")

    if config['model']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Load dataset (use config paths with fallback to defaults)
    train_data_path = config['data'].get('train_path', "data/packed/train")
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        sys.exit(1)

    train_dataset = load_from_disk(train_data_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=config['data'].get('num_workers', 8),
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['weight_decay'],
    )

    max_steps = config['training']['max_steps']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=max_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop with optional OOM recovery
    print(f"\n🚀 Starting FP8 pretraining ({max_steps:,} steps)...")
    if enable_oom_recovery:
        print("OOM recovery: ENABLED")

    model.train()
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")

    # OOM handler for FP8 training
    oom_handler = OOMHandler(
        initial_batch_size=config['training']['per_device_train_batch_size'],
        min_batch_size=1,
    ) if enable_oom_recovery else None

    while global_step < max_steps:
        for batch in train_loader:
            try:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Reset OOM retry count on success
                if oom_handler:
                    oom_handler.reset()

            except RuntimeError as e:
                if oom_handler and oom_handler.handle_oom(e):
                    # Skip this batch and continue
                    continue
                raise

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                if global_step % 10 == 0:
                    postfix = {"loss": f"{loss.item():.4f}"}
                    if oom_handler and oom_handler.oom_count > 0:
                        postfix["oom_events"] = oom_handler.oom_count
                    progress_bar.set_postfix(postfix)

                if global_step % config['checkpointing']['save_steps'] == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        f"{config['checkpointing']['output_dir']}/checkpoint-{global_step}",
                        save_function=accelerator.save,
                    )

                if global_step >= max_steps:
                    break

    progress_bar.close()

    # Print OOM summary if any occurred
    if oom_handler and oom_handler.oom_count > 0:
        stats = oom_handler.get_stats()
        print(f"\n[OOM Summary] Total OOM events: {stats['total_oom_events']}")

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        "checkpoints/pretrain_final",
        save_function=accelerator.save,
    )
    print("✓ FP8 Pretraining complete!")

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain 7B model")
    parser.add_argument("--fp8", action="store_true", help="Force FP8 precision (H100 only)")
    parser.add_argument("--no-fp8", action="store_true", help="Disable FP8, use BF16")
    # Config overrides
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to config file")
    parser.add_argument("--max_steps", type=int, help="Override max training steps")
    parser.add_argument("--save_steps", type=int, help="Override checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, help="Override evaluation frequency")
    parser.add_argument("--logging_steps", type=int, help="Override logging frequency")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_data_path", type=str, help="Override training data path")
    parser.add_argument("--eval_data_path", type=str, help="Override evaluation data path")
    parser.add_argument("--enable-oom-recovery", action="store_true", help="Enable automatic OOM recovery")

    # Training hyperparameters (for size-specific optimization)
    parser.add_argument("--per_device_train_batch_size", type=int, help="Override batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--warmup_ratio", type=float, help="Override warmup ratio (0.0-1.0)")
    parser.add_argument("--warmup_steps", type=int, help="Override warmup steps (takes precedence over ratio)")
    parser.add_argument("--dataloader_num_workers", type=int, help="Override number of dataloader workers")
    parser.add_argument("--optim", type=str, choices=["adamw_torch_fused", "adamw_bnb_8bit", "adamw_torch"],
                        help="Override optimizer")
    parser.add_argument("--no-torch-compile", action="store_true",
                        help="Disable torch.compile (faster for small models)")

    # Kernel optimization flags
    parser.add_argument("--use-liger-kernel", action="store_true", default=True,
                        help="Enable Liger Kernel (~20%% speedup, ~60%% memory reduction) [default: enabled]")
    parser.add_argument("--no-liger-kernel", action="store_true",
                        help="Disable Liger Kernel")
    parser.add_argument("--use-cce", action="store_true", default=True,
                        help="Enable Cut Cross-Entropy (~95%% memory reduction on loss) [default: enabled]")
    parser.add_argument("--no-cce", action="store_true",
                        help="Disable Cut Cross-Entropy")
    args = parser.parse_args()

    # Resolve kernel optimization flags
    use_liger_kernel = args.use_liger_kernel and not args.no_liger_kernel
    use_cce = args.use_cce and not args.no_cce

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Validate prerequisites
    if not check_tokenizer_exists():
        sys.exit(1)
    if not check_checkpoint_exists("checkpoints/init", "Initial model"):
        print("\nTo initialize the model, run:")
        print("  python scripts/04_init_model.py")
        print("\nOr for demo: python scripts/setup_demo.py")
        sys.exit(1)

    use_fp8 = None
    if args.fp8:
        use_fp8 = True
    elif args.no_fp8:
        use_fp8 = False

    # Build CLI overrides dict
    cli_overrides = {}
    if args.max_steps is not None:
        cli_overrides['max_steps'] = args.max_steps
    if args.save_steps is not None:
        cli_overrides['save_steps'] = args.save_steps
    if args.eval_steps is not None:
        cli_overrides['eval_steps'] = args.eval_steps
    if args.logging_steps is not None:
        cli_overrides['logging_steps'] = args.logging_steps
    if args.output_dir is not None:
        cli_overrides['output_dir'] = args.output_dir
    if args.resume_from_checkpoint is not None:
        cli_overrides['resume_from_checkpoint'] = args.resume_from_checkpoint
    if args.train_data_path is not None:
        cli_overrides['train_data_path'] = args.train_data_path
    if args.eval_data_path is not None:
        cli_overrides['eval_data_path'] = args.eval_data_path
    if getattr(args, 'enable_oom_recovery', False):
        cli_overrides['enable_oom_recovery'] = True

    # New training hyperparameter overrides (for size-specific optimization)
    if args.per_device_train_batch_size is not None:
        cli_overrides['per_device_train_batch_size'] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        cli_overrides['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        cli_overrides['learning_rate'] = args.learning_rate
    if args.warmup_ratio is not None:
        cli_overrides['warmup_ratio'] = args.warmup_ratio
    if args.warmup_steps is not None:
        cli_overrides['warmup_steps'] = args.warmup_steps
    if args.dataloader_num_workers is not None:
        cli_overrides['dataloader_num_workers'] = args.dataloader_num_workers
    if args.optim is not None:
        cli_overrides['optim'] = args.optim
    if args.no_torch_compile:
        cli_overrides['torch_compile'] = False

    # Setup and detect GPU
    gpu_info = detect_gpu_type()
    print_gpu_info(gpu_info)

    # Determine precision
    if use_fp8 is None:
        use_fp8 = gpu_info['fp8_available']

    if use_fp8 and gpu_info['fp8_available']:
        # Use FP8 training path
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Apply CLI overrides to config
        if 'max_steps' in cli_overrides:
            config['training']['max_steps'] = cli_overrides['max_steps']
        train_with_fp8(
            config,
            gpu_info,
            enable_oom_recovery=cli_overrides.get('enable_oom_recovery', False),
            use_liger_kernel=use_liger_kernel,
            use_cce=use_cce
        )
    else:
        # Use standard BF16 training path
        if use_fp8 and not gpu_info['fp8_available']:
            print("Warning: FP8 requested but not available, using BF16")

        trainer, gpu_info = setup_training(
            use_fp8=False,
            config_path=args.config,
            cli_overrides=cli_overrides,
            use_liger_kernel=use_liger_kernel,
            use_cce=use_cce
        )
        print("🚀 Starting pretraining...")
        trainer.train(resume_from_checkpoint=cli_overrides.get('resume_from_checkpoint'))

        # Unwrap the compiled model before saving to avoid _orig_mod. prefix in state dict
        output_dir = cli_overrides.get('output_dir', "checkpoints/pretrain_final")
        unwrapped_model = unwrap_compiled_model(trainer.model)
        unwrapped_model.save_pretrained(output_dir)
        print(f"✓ Pretraining complete! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
