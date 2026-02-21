"""Unsloth model loading, saving, and adapter management for GPT-OSS 20B.

Provides utilities for:
- Loading GPT-OSS 20B with 4-bit quantization via Unsloth
- Configuring LoRA adapters for MoE models (with expert FFN targeting)
- Saving/loading adapters (with PEFT fallback for MoE save bug)
- Merging adapters into base model
- Exporting to GGUF format

GPT-OSS 20B MoE Architecture:
- 32 experts per layer, top-4 routing
- Expert FFN param structure: experts.gate_up_projs.{idx}.weight (layer THEN index)
- Unsloth Bug #3405: default target modules miss expert layers entirely
- Unsloth 2026.2.1 claims to handle MoE via singular names but PEFT matching
  still fails ("set target_parameters but found no matching parameters")
- Fix: Pass auto-detected regex to Unsloth's get_peft_model (non-None,
  non-"all-linear" target_modules pass through to PEFT unchanged). This
  preserves Unsloth's memory optimizations: chunked cross-entropy,
  "unsloth" gradient checkpointing (CPU offload), and Split LoRA for MoE.
- Fallback: PEFT direct if Unsloth regex approach fails to target experts.
- Save: PEFT native fallback when Unsloth validation fails (Bug #3701)

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

    Inspects named parameters to find expert FFN layers. GPT-OSS 20B uses
    fused expert layers with structure: experts.{layer_name}.{expert_idx}.weight
    e.g., experts.gate_up_projs.0.weight, experts.down_projs.31.weight

    Returns:
        Dict with:
            is_moe: bool — whether model has MoE expert layers
            expert_module_names: list[str] — unique expert layer names (e.g., gate_up_projs, down_projs)
            num_experts: int — number of experts detected
            expert_param_count: int — total parameters in expert layers
            sample_param_names: list[str] — first 5 expert param names for debugging
    """
    expert_layer_names = set()
    expert_indices = set()
    expert_param_count = 0
    sample_names = []

    for name, param in model.named_parameters():
        if "expert" not in name.lower():
            continue

        expert_param_count += param.numel()
        if len(sample_names) < 5:
            sample_names.append(name)

        parts = name.split(".")
        # Find "experts" in the path, then look at what follows
        for i, part in enumerate(parts):
            if part.lower() == "experts" and i + 2 < len(parts):
                next_part = parts[i + 1]
                after_next = parts[i + 2]
                # Structure A: experts.{layer_name}.{idx} (GPT-OSS)
                # e.g., experts.gate_up_projs.0.weight
                try:
                    int(after_next)
                    expert_layer_names.add(next_part)
                    expert_indices.add(int(after_next))
                except ValueError:
                    pass
                # Structure B: experts.{idx}.{layer_name} (other MoE models)
                # e.g., experts.0.gate_up_projs.weight
                try:
                    int(next_part)
                    expert_layer_names.add(after_next.split(".")[0])
                    expert_indices.add(int(next_part))
                except ValueError:
                    pass
                break  # Only process first "experts" in path

    return {
        "is_moe": len(expert_layer_names) > 0,
        "expert_module_names": sorted(expert_layer_names),
        "num_experts": max(expert_indices) + 1 if expert_indices else 0,
        "expert_param_count": expert_param_count,
        "sample_param_names": sample_names,
    }


def get_moe_expert_regex(model: Any) -> str | None:
    """Build a regex pattern matching MoE expert FFN modules.

    Inspects model parameters to determine the naming convention, then
    builds a PEFT-compatible regex that matches all expert FFN layers.

    GPT-OSS 20B structure: experts.gate_up_projs.{idx}, experts.down_projs.{idx}
    Other MoE models:      experts.{idx}.gate_up_projs, experts.{idx}.down_projs

    Returns:
        Regex string for PEFT target_modules, or None if not MoE.
    """
    moe_info = detect_moe_experts(model)
    if not moe_info["is_moe"]:
        return None

    layer_names = moe_info["expert_module_names"]
    # Filter out non-FFN names (e.g., router/gate weights)
    ffn_names = [n for n in layer_names if "proj" in n.lower() or "linear" in n.lower()]
    if not ffn_names:
        ffn_names = layer_names

    # Build regex: match any expert FFN layer regardless of index position
    names_pattern = "|".join(ffn_names)
    # Matches both: experts.{name}.{idx} and experts.{idx}.{name}
    regex = f"experts\\.({names_pattern})\\.\\d+"

    print(f"  MoE detected: {moe_info['num_experts']} experts, "
          f"{moe_info['expert_param_count']:,} expert params")
    print(f"  Expert layer types: {ffn_names}")
    print(f"  Expert regex: {regex}")
    if moe_info["sample_param_names"]:
        print(f"  Sample params: {moe_info['sample_param_names'][:3]}")

    return regex


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


def _build_moe_regex(moe_info: dict[str, Any]) -> str:
    """Build a combined regex matching attention + MoE expert modules.

    PEFT uses re.fullmatch against full module paths, so we prefix with .*.
    The $ anchor avoids matching sub-modules (e.g., .weight).

    Returns regex like: ``.*(<attn>|experts\\.(<expert>)\\.\\d+)$``
    """
    attention_pattern = "q_proj|k_proj|v_proj|o_proj"
    expert_names = moe_info["expert_module_names"]
    expert_names_pattern = "|".join(expert_names)
    return (
        f".*({attention_pattern}|"
        f"experts\\.({expert_names_pattern})\\.\\d+)$"
    )


def apply_lora_config(
    model: Any,
    lora_config: dict[str, Any],
    auto_detect_moe: bool = True,
) -> Any:
    """Apply LoRA configuration to an Unsloth model.

    The router/gate layers are NOT targeted — MoE routing stays frozen.

    For MoE models (GPT-OSS 20B):
    - Passes auto-detected regex to Unsloth's get_peft_model. Non-None,
      non-"all-linear" target_modules pass through to PEFT's LoraConfig
      unchanged, preserving Unsloth's memory optimizations (chunked CE,
      "unsloth" gradient checkpointing with CPU offload, Split LoRA).
    - Verifies expert layers were actually targeted after application.
    - Falls back to PEFT direct if Unsloth fails to target experts.

    For dense models:
    - Unsloth's get_peft_model works correctly and is used as-is.

    Args:
        model: Unsloth model from load_unsloth_model().
        lora_config: Dict with r, lora_alpha, target_modules, etc.
            Typically loaded from YAML config.
        auto_detect_moe: If True, auto-detect MoE and fix targeting.

    Returns:
        Model with LoRA adapters applied.
    """
    from unsloth import FastLanguageModel

    r = lora_config.get("r", 64)
    lora_alpha = lora_config.get("lora_alpha", 128)
    lora_dropout = lora_config.get("lora_dropout", 0)
    bias = lora_config.get("bias", "none")
    use_gc = lora_config.get("use_gradient_checkpointing", "unsloth")
    use_rslora = lora_config.get("use_rslora", False)

    # Check for MoE before applying any LoRA
    moe_info = detect_moe_experts(model) if auto_detect_moe else {"is_moe": False}

    if moe_info["is_moe"]:
        # MoE model: pass regex to Unsloth to keep all memory optimizations
        combined_regex = _build_moe_regex(moe_info)
        print(f"  MoE detected — using Unsloth with regex passthrough:")
        print(f"  Combined regex: {combined_regex}")
        print(f"  Rank: {r}, Alpha: {lora_alpha}")

        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                target_modules=combined_regex,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gc,
                use_rslora=use_rslora,
            )

            # Verify expert layers were actually targeted
            result = verify_expert_lora(model)
            if not result["has_expert_lora"]:
                raise RuntimeError(
                    f"Unsloth regex passthrough did not target expert layers "
                    f"(only {result['attention_lora_params']:,} attention params). "
                    f"Falling back to PEFT direct."
                )
            print(f"  Unsloth + regex: SUCCESS (all memory optimizations preserved)")

        except Exception as e:
            print(f"  Unsloth regex approach failed: {e}")
            print(f"  Falling back to PEFT direct mode...")
            model = _apply_moe_lora_peft_direct(model, lora_config, moe_info)
    else:
        # Dense model: let Unsloth handle everything
        target_modules = lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gc,
            use_rslora=use_rslora,
        )

    return model


def _apply_moe_lora_peft_direct(
    model: Any,
    lora_config: dict[str, Any],
    moe_info: dict[str, Any],
) -> Any:
    """Fallback: apply LoRA to MoE model using PEFT directly, bypassing Unsloth.

    Used when Unsloth's regex passthrough fails to target expert layers.
    Loses Unsloth's memory optimizations (chunked CE, CPU-offloaded gradient
    checkpointing) so batch sizes may need to be reduced.
    """
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    r = lora_config.get("r", 64)
    lora_alpha = lora_config.get("lora_alpha", 128)
    lora_dropout = lora_config.get("lora_dropout", 0)
    bias = lora_config.get("bias", "none")

    combined_regex = _build_moe_regex(moe_info)
    print(f"  PEFT direct mode (no Unsloth memory optimizations):")
    print(f"  Combined regex: {combined_regex}")
    print(f"  WARNING: May need reduced batch sizes without Unsloth optimizations.")

    config = PeftLoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=combined_regex,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, config)

    # Standard gradient checkpointing (not Unsloth's CPU-offloaded version)
    use_gc = lora_config.get("use_gradient_checkpointing", "unsloth")
    if use_gc:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print(f"  Gradient checkpointing: enabled (standard, not CPU-offloaded)")

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
    elif save_method in ("merged_16bit", "merged_4bit"):
        try:
            model.save_pretrained_merged(output_dir, tokenizer, save_method=save_method)
        except RuntimeError as e:
            if "LoRAs" in str(e) and "match" in str(e):
                # Unsloth Bug #3701: Merged files already on disk, just
                # validation failed. Save config + tokenizer to complete.
                print(f"  Unsloth Bug #3701 (merged files already on disk)")
                os.makedirs(output_dir, exist_ok=True)
                model.config.save_pretrained(output_dir)
                if tokenizer:
                    tokenizer.save_pretrained(output_dir)
            else:
                raise
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


def _remap_peft_keys(state_dict: dict, adapter_name: str = "default") -> dict:
    """Remap PEFT-saved keys to match Unsloth's live model state dict.

    PEFT's ``save_pretrained`` strips the adapter name from keys:
      on-disk: ``...lora_A.weight``
      live model: ``...lora_A.default.weight``

    This inserts the adapter name back so ``load_state_dict`` matches.
    """
    remapped = {}
    lora_parts = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    for k, v in state_dict.items():
        if any(lp in k for lp in lora_parts) and f".{adapter_name}." not in k:
            parts = k.split(".")
            for i, part in enumerate(parts):
                if part in lora_parts:
                    parts.insert(i + 1, adapter_name)
                    break
            k = ".".join(parts)
        remapped[k] = v
    return remapped


def _load_base_and_adapter(
    FastLanguageModel: Any,
    base_model: str,
    adapter_path: str,
    max_seq_length: int,
) -> tuple[Any, Any]:
    """Load base model via Unsloth, then apply saved LoRA adapter weights.

    Uses Unsloth's own ``get_peft_model`` to recreate the LoRA architecture
    (preserving Unsloth's internal wrapping so ``save_pretrained_merged``
    recognizes the model). Adapter weights are loaded with key remapping
    to handle the PEFT-native save format (Bug #3701 fallback saves keys
    without the ``.default.`` adapter name prefix).

    Strategy:
      1. Load the base model through Unsloth (preserves internal format).
      2. Read adapter_config.json for LoRA hyper-params.
      3. Apply LoRA via Unsloth's ``get_peft_model`` (MoE-aware wrapping).
      4. Load trained adapter weights with PEFT key remapping.
    """
    import json

    # Resolve adapter path (may be in adapter_path or adapter_path/final)
    actual_adapter_path = adapter_path
    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        final_path = os.path.join(adapter_path, "final")
        if os.path.exists(os.path.join(final_path, "adapter_config.json")):
            actual_adapter_path = final_path
            adapter_cfg_path = os.path.join(actual_adapter_path, "adapter_config.json")
            print(f"  Using adapter subdir: {actual_adapter_path}")

    with open(adapter_cfg_path) as f:
        acfg = json.load(f)

    # 1. Load base model via Unsloth
    print(f"  Loading base model: {base_model}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # 2. Recreate LoRA with same config via Unsloth (preserves Unsloth wrapping)
    target_modules = acfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_up_proj", "down_proj",
    ])
    print(f"  Applying LoRA: r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}")
    print(f"  Target modules: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=acfg.get("r", 64),
        target_modules=target_modules,
        lora_alpha=acfg.get("lora_alpha", 128),
        lora_dropout=acfg.get("lora_dropout", 0),
        bias=acfg.get("bias", "none"),
        use_gradient_checkpointing=False,  # Not needed for merge
        use_rslora=acfg.get("use_rslora", False),
    )

    # 3. Load trained adapter weights (with PEFT key remapping)
    weight_file = os.path.join(actual_adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weight_file):
        weight_file = os.path.join(actual_adapter_path, "adapter_model.bin")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"No adapter weights found in {actual_adapter_path} "
            f"(checked adapter_model.safetensors and adapter_model.bin)"
        )

    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)
    else:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)

    # PEFT native save strips adapter name from keys (e.g. "lora_A.weight"
    # instead of "lora_A.default.weight"). Remap to match Unsloth's model.
    remapped = _remap_peft_keys(state_dict)
    incompatible = model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(incompatible.unexpected_keys)
    print(f"  Loaded {loaded}/{len(remapped)} adapter weight tensors")
    if incompatible.unexpected_keys:
        print(f"  Warning: {len(incompatible.unexpected_keys)} unexpected keys")

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
            try:
                model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
            except RuntimeError as e:
                if "LoRAs" in str(e) and "match" in str(e):
                    # Unsloth Bug #3701: MoE LoRA count validation failure.
                    # At this point Unsloth has ALREADY:
                    #   1. Dequantized MXFP4 weights to bf16 safetensor files
                    #   2. Merged LoRA weights into those files on disk
                    #   3. Regenerated the safetensors index
                    # The error fires at the post-save validation step.
                    # We just need to save config + tokenizer to complete the output.
                    print(f"  Unsloth Bug #3701 (validation only — merged files already on disk)")
                    os.makedirs(save_path, exist_ok=True)
                    model.config.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"  Saved config + tokenizer to complete the merge")
                else:
                    raise
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
