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


def _convert_packed_to_unpacked(save_path: str) -> bool:
    """Convert packed MoE expert tensors to unpacked nn.Linear format.

    Unsloth's merge outputs expert weights in packed format for transformers
    >= 5.x (``@use_experts_implementation(is_transposed=True)``):

      - ``experts.gate_up_proj``:      ``[num_experts, hidden, 2*intermediate]``
      - ``experts.gate_up_proj_bias``:  ``[num_experts, 2*intermediate]``
      - ``experts.down_proj``:          ``[num_experts, intermediate, hidden]``
      - ``experts.down_proj_bias``:     ``[num_experts, hidden]``

    The packed forward pass does ``output = x @ W[expert_idx]`` (no transpose).

    GptOssForCausalLM on transformers 4.x expects unpacked nn.Linear format:

      - ``experts.gate_up_projs.{i}.weight``: ``[2*intermediate, hidden]``
      - ``experts.gate_up_projs.{i}.bias``:   ``[2*intermediate]``
      - ``experts.down_projs.{i}.weight``:    ``[hidden, intermediate]``
      - ``experts.down_projs.{i}.bias``:      ``[hidden]``

    The nn.Linear forward does ``output = x @ weight.T + bias``.

    Therefore: ``nn.Linear.weight = packed[i].T`` (transpose required for
    weight matrices, not for bias vectors).

    An earlier version of this code (``unpack_moe_expert_tensors``, commit
    6ef9dd8) split packed tensors but DID NOT transpose, producing garbage.
    This version transposes correctly.

    Also remaps router keys if the model class expects a different path
    (determined dynamically via ``AutoModelForCausalLM.from_config``).

    Args:
        save_path: Path to the merged model directory with safetensors.

    Returns:
        True if conversion was performed, False if already unpacked.
    """
    import json
    import re
    from collections import defaultdict

    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(save_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Detect packed expert keys (3D tensors: gate_up_proj, down_proj, and biases)
    packed_pattern = re.compile(
        r".*\.experts\.(gate_up_proj|down_proj|gate_up_proj_bias|down_proj_bias)$"
    )
    packed_keys = [k for k in weight_map if packed_pattern.match(k)]
    if not packed_keys:
        return False

    print(f"  Converting {len(packed_keys)} packed expert tensors to unpacked format...")

    # --- Determine router key mapping dynamically ---
    router_rename: dict[str, str] = {}
    actual_router_keys = [k for k in weight_map if ".mlp.router." in k]

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(save_path)
        with torch.device("meta"):
            ref_model = AutoModelForCausalLM.from_config(config)
        expected_keys = set(ref_model.state_dict().keys())
        del ref_model

        # Check if actual router keys exist in expected set
        for k in actual_router_keys:
            if k not in expected_keys:
                # Try router.linear.{weight,bias} variant
                candidate = k.replace(".mlp.router.", ".mlp.router.linear.")
                if candidate in expected_keys:
                    router_rename[k] = candidate
    except Exception as e:
        print(f"  Warning: Could not load model class for router key detection: {e}")
        # Fallback: documented mismatch is router.weight -> router.linear.weight
        for k in actual_router_keys:
            if ".router.linear." not in k:
                router_rename[k] = k.replace(".mlp.router.", ".mlp.router.linear.")

    # --- Group keys to convert by shard file ---
    keys_to_convert = set(packed_keys) | set(router_rename.keys())
    shard_groups: dict[str, list[str]] = defaultdict(list)
    for key in keys_to_convert:
        shard_groups[weight_map[key]].append(key)

    new_weight_map = dict(weight_map)

    for shard_file, convert_keys in shard_groups.items():
        shard_path = os.path.join(save_path, shard_file)
        new_tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] = {}
        convert_set = set(convert_keys)

        with safe_open(shard_path, framework="pt") as f:
            meta = f.metadata()
            if meta:
                metadata = dict(meta)

            # Copy all keys that don't need conversion
            for key in f.keys():
                if key not in convert_set:
                    new_tensors[key] = f.get_tensor(key)

            # Convert each packed key
            for key in convert_keys:
                tensor = f.get_tensor(key)

                # Router key: just rename
                if key in router_rename:
                    new_key = router_rename[key]
                    new_tensors[new_key] = tensor
                    del new_weight_map[key]
                    new_weight_map[new_key] = shard_file
                    continue

                # Expert tensor: unpack (split along dim 0)
                del new_weight_map[key]
                num_experts = tensor.shape[0]
                is_bias = key.endswith("_bias")

                # Key mapping:
                #   experts.gate_up_proj      -> experts.gate_up_projs.{i}.weight
                #   experts.gate_up_proj_bias -> experts.gate_up_projs.{i}.bias
                #   experts.down_proj         -> experts.down_projs.{i}.weight
                #   experts.down_proj_bias    -> experts.down_projs.{i}.bias
                prefix = key.rsplit(".", 1)[0]  # model.layers.N.mlp.experts
                last = key.rsplit(".", 1)[1]  # gate_up_proj, down_proj_bias, etc.

                if is_bias:
                    proj_plural = last.replace("_bias", "") + "s"
                    suffix = "bias"
                else:
                    proj_plural = last + "s"
                    suffix = "weight"

                for i in range(num_experts):
                    expert_tensor = tensor[i]
                    if not is_bias:
                        # Transpose: packed (in, out) -> nn.Linear (out, in)
                        expert_tensor = expert_tensor.T.contiguous()
                    new_key = f"{prefix}.{proj_plural}.{i}.{suffix}"
                    new_tensors[new_key] = expert_tensor
                    new_weight_map[new_key] = shard_file

        save_file(new_tensors, shard_path, metadata=metadata)

    # Update index
    index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    unpacked_count = sum(1 for k in new_weight_map
                         if re.match(r".*\.experts\.(gate_up_projs|down_projs)\.\d+\.", k))
    print(f"  Unpacked: {len(packed_keys)} packed -> {unpacked_count} individual expert tensors")
    if router_rename:
        print(f"  Remapped: {len(router_rename)} router keys")
    print(f"  Total keys: {len(new_weight_map)}")

    return True


def cleanup_merged_moe(save_path: str) -> bool:
    """Clean up Unsloth's merged MoE model output for transformers 4.x.

    After Unsloth merges LoRA into a GPT-OSS MoE model and dequantizes MXFP4
    to bf16, the expert weights are saved in PACKED format — all 32 experts'
    weights in a single 3D tensor per projection type:
      - gate_up_proj: [num_experts, hidden_size, 2*intermediate_size]
      - down_proj:    [num_experts, intermediate_size, hidden_size]

    This packed format is for transformers >= 5.x which uses
    @use_experts_implementation(is_transposed=True). Unsloth requires
    transformers 4.x, which expects unpacked nn.Linear format.

    This function:
    1. Strips quantization_config from config.json (prevents bf16 being
       loaded as Linear4bit/MXFP4).
    2. Removes any leftover _scales tensors (MXFP4 quantization artifacts).
    3. Converts packed expert tensors to unpacked nn.Linear format with
       proper transpose (packed stores (in, out) for x @ W; nn.Linear
       stores (out, in) for x @ W.T).
    4. Remaps router keys to match model class expectations.

    Args:
        save_path: Path to the merged model directory.

    Returns:
        True if any changes were made.
    """
    import json
    import re

    changed = False

    # --- 1. Strip quantization_config from config.json ---
    config_path = os.path.join(save_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if "quantization_config" in config:
            del config["quantization_config"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            print(f"  Stripped quantization_config from config.json")
            changed = True

    # --- 2. Remove any _scales tensors (MXFP4 artifacts) ---
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"  No safetensors index found, skipping artifact cleanup")
        return changed

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Find scale artifacts
    scale_keys = [k for k in weight_map if re.match(r".*_scales$", k)]
    if scale_keys:
        from collections import defaultdict

        from safetensors import safe_open
        from safetensors.torch import save_file

        # Group by shard for efficient processing
        shard_removes = defaultdict(list)
        for key in scale_keys:
            shard_removes[weight_map[key]].append(key)
            del weight_map[key]

        for shard_file, keys_to_remove in shard_removes.items():
            shard_path = os.path.join(save_path, shard_file)
            remove_set = set(keys_to_remove)
            new_tensors = {}
            metadata = {}

            with safe_open(shard_path, framework="pt") as f:
                meta = f.metadata()
                if meta:
                    metadata = dict(meta)
                for key in f.keys():
                    if key not in remove_set:
                        new_tensors[key] = f.get_tensor(key)

            save_file(new_tensors, shard_path, metadata=metadata)

        # Update index
        index["weight_map"] = weight_map
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        print(f"  Removed {len(scale_keys)} MXFP4 scale artifacts")
        changed = True

    # --- 3. Convert packed expert tensors to unpacked nn.Linear format ---
    if _convert_packed_to_unpacked(save_path):
        changed = True

    # --- 4. Validate key format ---
    # Re-read index after potential conversion
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    actual_keys = set(weight_map.keys())
    expert_keys = sorted([k for k in actual_keys if "expert" in k and "layers.0." in k])
    print(f"  Expert keys (layer 0, first 5): {expert_keys[:5]}")
    print(f"  Total model keys: {len(actual_keys)}")

    if not changed:
        print(f"  No cleanup needed — merged model is ready")

    return changed


def _validate_merged_keys(save_path: str) -> None:
    """Validate merged model keys match expected GptOss format.

    Loads the model class on meta device and compares against saved keys.
    Reports any mismatches for debugging.
    """
    import json

    print(f"\n  === Key Validation ===")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(save_path)
        with torch.device("meta"):
            ref_model = AutoModelForCausalLM.from_config(config)
        expected_keys = set(ref_model.state_dict().keys())
        expected_shapes = {k: tuple(v.shape) for k, v in ref_model.state_dict().items()}
        del ref_model

        index_path = os.path.join(save_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        actual_keys = set(index["weight_map"].keys())

        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)

        if not missing and not extra:
            print(f"  All {len(actual_keys)} keys match expected model format")
        else:
            if missing:
                print(f"  {len(missing)} missing keys (first 5):")
                for k in missing[:5]:
                    print(f"    {k}: expected shape {expected_shapes.get(k)}")
            if extra:
                print(f"  {len(extra)} extra keys (first 5):")
                for k in extra[:5]:
                    print(f"    {k}")

        # Spot-check a few expert tensor shapes
        from safetensors import safe_open
        check_keys = [k for k in actual_keys if "layers.0.mlp.experts" in k and "weight" not in k.split(".")[-1].replace("weight", "")]
        check_keys = sorted([k for k in actual_keys if "layers.0.mlp.experts" in k])[:4]
        for key in check_keys:
            shard_file = index["weight_map"][key]
            shard_path = os.path.join(save_path, shard_file)
            with safe_open(shard_path, framework="pt") as f:
                tensor = f.get_tensor(key)
            exp = expected_shapes.get(key, "N/A")
            match = "OK" if exp != "N/A" and tuple(tensor.shape) == exp else "MISMATCH" if exp != "N/A" else "?"
            print(f"  {key}: {list(tensor.shape)} (expected {exp}) [{match}]")

        print(f"  === End Validation ===\n")

    except Exception as e:
        print(f"  Validation failed (non-fatal): {e}")


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
                    # Strip quantization_config (MXFP4 artifact) so the merged
                    # bf16 model loads with standard Linear, not Linear4bit.
                    if hasattr(model.config, "quantization_config"):
                        delattr(model.config, "quantization_config")
                    model.config.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"  Saved config + tokenizer to complete the merge")
                else:
                    raise
            # Clean up merged model: strip MXFP4 artifacts, convert packed
            # expert tensors to unpacked nn.Linear format for transformers 4.x,
            # and remap router keys.
            cleanup_merged_moe(save_path)
            _validate_merged_keys(save_path)
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

    # Free VRAM — merge model is no longer needed, smoke test loads fresh
    del model
    if tokenizer is not None:
        del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
