"""Unsloth model loading, saving, and adapter management for GPT-OSS 20B.

Provides utilities for:
- Loading GPT-OSS 20B with 4-bit quantization via Unsloth
- Configuring LoRA adapters for MoE models (with expert FFN targeting)
- Saving/loading adapters (with PEFT fallback for MoE save bug)
- Merging adapters into base model
- Exporting to GGUF format

GPT-OSS 20B MoE Architecture:
- 32 experts per layer, top-4 routing
- Expert FFN uses fused layers: gate_up_projs (plural) and down_projs (plural)
- Unsloth Bug #3405: default target modules miss expert layers entirely
- Unsloth Bug #3701: save validation fails when expert modules are targeted
- Fix: use singular names (gate_up_proj, down_proj) which Unsloth maps internally,
  with PEFT native save as fallback if Unsloth's save validation fails.

Requires: pip install -e ".[gpt_oss]"
"""
from __future__ import annotations

import os
from typing import Any

import torch
import yaml


def load_unsloth_model(
    model_name: str = "openai/gpt-oss-20b",
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    dtype: torch.dtype | None = None,
) -> tuple[Any, Any]:
    """Load a model with Unsloth optimizations.

    Args:
        model_name: HuggingFace model ID or local path.
        max_seq_length: Maximum sequence length for training.
        load_in_4bit: Use 4-bit quantization (recommended for 20B model).
        dtype: Torch dtype. None = auto-detect.

    Returns:
        (model, tokenizer) tuple.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        full_finetuning=False,
    )

    return model, tokenizer


def detect_moe_experts(model: Any) -> dict[str, Any]:
    """Detect MoE expert layers in the model architecture.

    Inspects named modules to find expert FFN layers. GPT-OSS 20B uses
    fused expert layers: gate_up_projs (plural) and down_projs (plural).

    Returns:
        Dict with:
            is_moe: bool — whether model has MoE expert layers
            expert_module_names: list[str] — unique expert parameter name patterns
            num_experts: int — number of experts detected
            expert_param_count: int — total parameters in expert layers
    """
    expert_names = set()
    expert_param_count = 0
    num_experts = 0

    for name, param in model.named_parameters():
        if "expert" in name.lower():
            # Extract the module type (e.g., gate_up_projs, down_projs)
            parts = name.split(".")
            for i, part in enumerate(parts):
                if "expert" in part.lower() and i + 1 < len(parts):
                    # Count unique expert indices
                    try:
                        expert_idx = int(parts[i + 1])
                        num_experts = max(num_experts, expert_idx + 1)
                    except (ValueError, IndexError):
                        pass
                    # Get the layer type after expert index
                    if i + 2 < len(parts):
                        expert_names.add(parts[i + 2].split(".")[0])
            expert_param_count += param.numel()

    return {
        "is_moe": len(expert_names) > 0,
        "expert_module_names": sorted(expert_names),
        "num_experts": num_experts,
        "expert_param_count": expert_param_count,
    }


def get_moe_target_modules(
    model: Any,
    base_targets: list[str] | None = None,
) -> list[str]:
    """Get target modules that include MoE expert FFN layers.

    For GPT-OSS 20B, Unsloth's Feb 2026+ versions accept singular names
    (gate_up_proj, down_proj) and internally map them to the fused expert
    layers (gate_up_projs, down_projs with expert indexing).

    Falls back to the base attention-only targets if no MoE layers detected.

    Args:
        model: Loaded model to inspect.
        base_targets: Base target modules (attention). Defaults to standard set.

    Returns:
        List of target module names including expert FFN layers.
    """
    if base_targets is None:
        base_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

    moe_info = detect_moe_experts(model)
    if not moe_info["is_moe"]:
        # Dense model — use standard FFN names
        return base_targets + ["gate_proj", "up_proj", "down_proj"]

    # MoE model — use singular names that Unsloth maps to expert layers.
    # Unsloth internally expands these to target all expert copies.
    # Do NOT use plural names (gate_up_projs) directly — Unsloth won't match them.
    # Do NOT use regex patterns — triggers save validation bug #3701.
    expert_targets = ["gate_up_proj", "down_proj"]

    print(f"  MoE detected: {moe_info['num_experts']} experts, "
          f"{moe_info['expert_param_count']:,} expert params")
    print(f"  Expert layer types: {moe_info['expert_module_names']}")
    print(f"  Using MoE target modules: {base_targets + expert_targets}")

    return base_targets + expert_targets


def verify_expert_lora(model: Any) -> dict[str, Any]:
    """Verify that LoRA adapters were applied to MoE expert layers.

    Call AFTER apply_lora_config() to confirm expert FFN layers are trainable.

    Returns:
        Dict with:
            expert_lora_count: int — number of expert LoRA parameters
            expert_lora_params: int — total trainable params in expert layers
            attention_lora_params: int — total trainable params in attention layers
            all_lora_names: list[str] — names of all trainable parameters
            has_expert_lora: bool — True if expert layers have LoRA adapters
    """
    expert_lora_count = 0
    expert_lora_params = 0
    attention_lora_params = 0
    all_lora_names = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            all_lora_names.append(name)
            if "expert" in name.lower():
                expert_lora_count += 1
                expert_lora_params += param.numel()
            elif any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                attention_lora_params += param.numel()

    has_expert = expert_lora_count > 0
    total_trainable = expert_lora_params + attention_lora_params

    if has_expert:
        print(f"  Expert LoRA: {expert_lora_count} parameters, "
              f"{expert_lora_params:,} trainable ({100*expert_lora_params/total_trainable:.1f}%)")
        print(f"  Attention LoRA: {attention_lora_params:,} trainable "
              f"({100*attention_lora_params/total_trainable:.1f}%)")
    else:
        print(f"  WARNING: No expert LoRA adapters found!")
        print(f"  Only attention layers are trainable: {attention_lora_params:,} params")
        print(f"  Expert FFN layers (~19B params) are NOT being fine-tuned.")

    return {
        "expert_lora_count": expert_lora_count,
        "expert_lora_params": expert_lora_params,
        "attention_lora_params": attention_lora_params,
        "all_lora_names": all_lora_names,
        "has_expert_lora": has_expert,
    }


def apply_lora_config(
    model: Any,
    lora_config: dict[str, Any],
    auto_detect_moe: bool = True,
) -> Any:
    """Apply LoRA configuration to an Unsloth model.

    The router/gate layers are NOT targeted — MoE routing stays frozen.
    When auto_detect_moe=True, automatically detects MoE expert layers
    and adds them to target_modules.

    Args:
        model: Unsloth model from load_unsloth_model().
        lora_config: Dict with r, lora_alpha, target_modules, etc.
            Typically loaded from YAML config.
        auto_detect_moe: If True, auto-detect and add MoE expert targets.

    Returns:
        Model with LoRA adapters applied.
    """
    from unsloth import FastLanguageModel

    target_modules = lora_config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Auto-detect MoE and update target modules if needed
    if auto_detect_moe:
        moe_info = detect_moe_experts(model)
        if moe_info["is_moe"]:
            # Replace dense FFN names with MoE-aware names
            attention_targets = [m for m in target_modules
                                 if m in ("q_proj", "k_proj", "v_proj", "o_proj")]
            target_modules = get_moe_target_modules(model, attention_targets)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("r", 64),
        target_modules=target_modules,
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
        use_rslora=lora_config.get("use_rslora", False),
    )

    return model


def save_adapter(
    model: Any,
    output_dir: str,
    tokenizer: Any | None = None,
    save_method: str = "lora",
) -> None:
    """Save LoRA adapter weights.

    Handles Unsloth Bug #3701: when MoE expert modules are targeted via LoRA,
    Unsloth's save validation fails ("# of LoRAs does not match # of saved
    modules"). Falls back to PEFT's native save_pretrained() which works.

    Args:
        model: Unsloth model with LoRA adapters.
        output_dir: Directory to save adapter weights.
        tokenizer: Optional tokenizer to save alongside.
        save_method: "lora" (adapter only) or "merged_16bit" or "merged_4bit".
    """
    os.makedirs(output_dir, exist_ok=True)

    if save_method == "lora":
        try:
            model.save_pretrained(output_dir)
        except Exception as e:
            err_msg = str(e).lower()
            if "lora" in err_msg and ("match" in err_msg or "saved" in err_msg):
                # Unsloth Bug #3701: MoE save validation failure
                # Fall back to PEFT's native save which bypasses Unsloth's check
                print(f"  Unsloth save failed (Bug #3701): {e}")
                print(f"  Falling back to PEFT native save...")
                _save_peft_native(model, output_dir)
            else:
                raise
    elif save_method == "merged_16bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    elif save_method == "merged_4bit":
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
    else:
        raise ValueError(f"Unknown save_method: {save_method}. Use 'lora', 'merged_16bit', or 'merged_4bit'.")

    if tokenizer and save_method == "lora":
        tokenizer.save_pretrained(output_dir)

    print(f"Saved {save_method} to {output_dir}")


def _save_peft_native(model: Any, output_dir: str) -> None:
    """Save adapter weights using PEFT's native save, bypassing Unsloth validation.

    This is the fallback for Unsloth Bug #3701 where save_pretrained() fails
    when MoE expert modules are targeted.
    """
    # Get the underlying PEFT model
    peft_model = model
    if hasattr(model, "model") and hasattr(model.model, "save_pretrained"):
        peft_model = model.model

    # Use PEFT's save which doesn't have Unsloth's validation bug
    from peft import PeftModel
    if isinstance(peft_model, PeftModel):
        peft_model.save_pretrained(output_dir)
        print(f"  PEFT native save successful: {output_dir}")
    else:
        # Last resort: save state dict manually
        import json
        from safetensors.torch import save_file
        state_dict = {
            k: v for k, v in model.state_dict().items()
            if "lora_" in k or "modules_to_save" in k
        }
        save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
        # Save adapter config
        if hasattr(model, "peft_config"):
            config = model.peft_config.get("default", model.peft_config)
            if hasattr(config, "to_dict"):
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(config.to_dict(), f, indent=2)
        print(f"  Manual adapter save successful: {output_dir}")


def _load_base_and_adapter(
    FastLanguageModel: Any,
    base_model: str,
    adapter_path: str,
    max_seq_length: int,
) -> tuple[Any, Any]:
    """Load base model via Unsloth, then apply saved LoRA adapter weights.

    Avoids ``FastLanguageModel.from_pretrained(adapter_path)`` which fails
    on newer Unsloth (``get_transformers_model_type`` runs before adapter
    detection, erroring on adapter dirs that lack config.json).

    Strategy:
      1. Load the base model through Unsloth (works — training does this).
      2. Read adapter_config.json to get the LoRA hyper-params.
      3. Apply LoRA via ``FastLanguageModel.get_peft_model`` (MoE-aware).
      4. Load the trained adapter weights from disk.
    """
    import json

    # 1. Load base model
    print(f"  Loading base model: {base_model}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # 2. Read saved adapter config (may be in adapter_path or adapter_path/final)
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        final_path = os.path.join(adapter_path, "final")
        if os.path.exists(os.path.join(final_path, "adapter_config.json")):
            adapter_path = final_path
            adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
            print(f"  Using adapter subdir: {adapter_path}")
    with open(adapter_cfg_path) as f:
        acfg = json.load(f)

    # 3. Recreate LoRA with same config (Unsloth handles MoE expert targeting)
    print(f"  Applying LoRA: r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=acfg.get("r", 64),
        target_modules=acfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_alpha=acfg.get("lora_alpha", 128),
        lora_dropout=acfg.get("lora_dropout", 0),
        bias=acfg.get("bias", "none"),
        use_gradient_checkpointing=acfg.get(
            "use_gradient_checkpointing", "unsloth",
        ),
        use_rslora=acfg.get("use_rslora", False),
    )

    # 4. Load trained adapter weights
    weight_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(weight_file):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)
        incompatible = model.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(incompatible.unexpected_keys)
        print(f"  Loaded {loaded} adapter weight tensors from {weight_file}")
        if incompatible.unexpected_keys:
            print(f"  Warning: {len(incompatible.unexpected_keys)} unexpected keys")
    else:
        # Try pytorch bin format
        bin_file = os.path.join(adapter_path, "adapter_model.bin")
        if os.path.exists(bin_file):
            state_dict = torch.load(bin_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded adapter weights from {bin_file}")
        else:
            raise FileNotFoundError(
                f"No adapter weights found in {adapter_path} "
                f"(checked adapter_model.safetensors and adapter_model.bin)"
            )

    return model, tok


def merge_and_export(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    tokenizer: Any | None = None,
    export_formats: list[str] | None = None,
    max_seq_length: int = 8192,
) -> None:
    """Merge LoRA adapter into base model and export.

    Args:
        base_model: HuggingFace model ID or local path.
        adapter_path: Path to LoRA adapter weights.
        output_dir: Directory to save merged model.
        tokenizer: Optional tokenizer (loaded if not provided).
        export_formats: List of export formats: "hf", "gguf_q4", "gguf_q8", "gguf_f16".
        max_seq_length: Max sequence length for loading.
    """
    if export_formats is None:
        export_formats = ["hf"]

    from unsloth import FastLanguageModel

    # Load base model first, then apply adapter weights.
    # FastLanguageModel.from_pretrained(adapter_path) fails on newer Unsloth
    # because get_transformers_model_type() runs before adapter detection.
    # Instead: load base model -> recreate LoRA -> load trained weights.
    model, tok = _load_base_and_adapter(
        FastLanguageModel, base_model, adapter_path, max_seq_length,
    )

    if tokenizer is None:
        tokenizer = tok

    os.makedirs(output_dir, exist_ok=True)

    for fmt in export_formats:
        if fmt == "hf":
            save_path = os.path.join(output_dir, "hf")
            model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
            print(f"Exported HuggingFace format to {save_path}")

        elif fmt.startswith("gguf_"):
            quantization = fmt.replace("gguf_", "")
            quant_map = {
                "q4": "q4_k_m",
                "q8": "q8_0",
                "f16": "f16",
            }
            quant_method = quant_map.get(quantization, "q4_k_m")
            save_path = os.path.join(output_dir, f"gguf_{quantization}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained_gguf(
                save_path,
                tokenizer,
                quantization_method=quant_method,
            )
            print(f"Exported GGUF ({quant_method}) to {save_path}")

        else:
            print(f"Warning: Unknown export format '{fmt}', skipping")


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_trainable_params(model: Any) -> None:
    """Print trainable parameter summary for a PEFT model with MoE breakdown."""
    trainable = 0
    total = 0
    expert_trainable = 0
    attention_trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            if "expert" in name.lower():
                expert_trainable += param.numel()
            elif any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                attention_trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    if expert_trainable > 0:
        other = trainable - expert_trainable - attention_trainable
        print(f"  Attention LoRA:  {attention_trainable:,}")
        print(f"  Expert FFN LoRA: {expert_trainable:,}")
        if other > 0:
            print(f"  Other LoRA:      {other:,}")
    elif trainable > 0:
        # Check if this is an MoE model missing expert LoRA
        moe_info = detect_moe_experts(model)
        if moe_info["is_moe"]:
            print(f"  WARNING: MoE model detected but NO expert LoRA adapters!")
            print(f"  Only {trainable:,} attention params are trainable."
                  f" Expert FFN ({moe_info['expert_param_count']:,} params) is frozen.")
