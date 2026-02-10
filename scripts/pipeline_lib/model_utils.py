"""Model loading and unwrapping utilities for torch.compile checkpoints."""
from __future__ import annotations

import glob
import os
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file as load_safetensors


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


def load_compiled_checkpoint(
    checkpoint_path: str,
    use_flash_attention: bool = True,
    move_to_device: bool = False,
) -> AutoModelForCausalLM:
    """Load a checkpoint that may have been saved with torch.compile wrapper.

    When a model is saved after torch.compile(), the state dict keys have
    '_orig_mod.' prefix. This function handles both compiled and non-compiled
    checkpoints transparently.

    Args:
        checkpoint_path: Path to the model checkpoint
        use_flash_attention: Enable Flash Attention 2
        move_to_device: Move model to CUDA if available (used by evaluate)

    Returns:
        Loaded model with correct weights
    """
    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)

    # Check for safetensors or pytorch format
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_safetensors(safetensors_path)
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        # Try sharded safetensors
        shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors")))
        if shard_files:
            state_dict = {}
            for shard in shard_files:
                state_dict.update(load_safetensors(shard))
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    # Check if state dict has _orig_mod. prefix (from torch.compile)
    has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if has_orig_mod:
        print("  Detected torch.compile checkpoint, stripping _orig_mod. prefix...")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_key = k[len("_orig_mod."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Create model with optional flash attention
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"
    try:
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
    except Exception:
        # Fall back to eager attention if flash attention fails
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

    model.load_state_dict(state_dict, strict=True)

    if use_flash_attention:
        print(f"  Flash Attention 2: ENABLED")

    if move_to_device:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return model
