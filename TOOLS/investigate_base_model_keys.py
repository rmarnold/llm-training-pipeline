"""Investigate the GPT-OSS 20B base model's expert key naming and shapes.

Downloads the model index from HuggingFace and inspects expert tensor structure
to understand the key naming convention and shapes. This helps debug the
merge adapter unpack step.

Usage:
    python TOOLS/investigate_base_model_keys.py
"""
import json
import os
import sys

from huggingface_hub import hf_hub_download
from safetensors import safe_open

BASE_MODEL = "openai/gpt-oss-20b"


def main():
    print(f"=== Investigating {BASE_MODEL} expert key structure ===\n")

    # Step 1: Download and inspect the index
    print("1. Downloading model index...")
    index_path = hf_hub_download(BASE_MODEL, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    all_keys = sorted(weight_map.keys())
    print(f"   Total keys: {len(all_keys)}")

    # Step 2: Find all expert-related keys
    print("\n2. Expert-related keys (layer 0 only):")
    layer0_expert_keys = [k for k in all_keys if "layers.0." in k and "expert" in k.lower()]
    for k in sorted(layer0_expert_keys):
        print(f"   {k}  ->  shard: {weight_map[k]}")

    # Also check for MoE-related keys
    print("\n3. MoE/router keys (layer 0 only):")
    layer0_moe_keys = [k for k in all_keys if "layers.0." in k and ("moe" in k.lower() or "router" in k.lower())]
    for k in sorted(layer0_moe_keys):
        print(f"   {k}  ->  shard: {weight_map[k]}")

    # Step 3: Show ALL layer 0 MLP keys for full picture
    print("\n4. ALL layer 0 MLP keys:")
    layer0_mlp = [k for k in all_keys if "layers.0.mlp" in k]
    for k in sorted(layer0_mlp):
        print(f"   {k}  ->  shard: {weight_map[k]}")

    # Step 4: Download one shard containing expert weights and inspect shapes
    if layer0_expert_keys:
        target_key = layer0_expert_keys[0]
        shard_file = weight_map[target_key]
        print(f"\n5. Downloading shard {shard_file} for shape inspection...")
        shard_path = hf_hub_download(BASE_MODEL, shard_file)

        # Inspect all expert keys in this shard
        with safe_open(shard_path, framework="pt") as f:
            shard_keys = list(f.keys())
            expert_keys_in_shard = [k for k in shard_keys if "expert" in k.lower()]

            print(f"   Shard has {len(shard_keys)} keys, {len(expert_keys_in_shard)} expert keys")
            print(f"\n   Expert tensor shapes:")
            for k in sorted(expert_keys_in_shard)[:20]:  # First 20
                tensor = f.get_tensor(k)
                print(f"   {k}: {list(tensor.shape)}, dtype={tensor.dtype}")

    # Step 5: Also check what HF model class expects via config
    print("\n6. What HuggingFace model class expects (from config):")
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        config_path = hf_hub_download(BASE_MODEL, "config.json")
        with open(config_path) as f:
            config_data = json.load(f)

        print(f"   Model type: {config_data.get('model_type')}")
        print(f"   Architectures: {config_data.get('architectures')}")
        print(f"   num_local_experts: {config_data.get('num_local_experts')}")
        print(f"   num_experts_per_tok: {config_data.get('num_experts_per_tok')}")
        print(f"   hidden_size: {config_data.get('hidden_size')}")
        print(f"   intermediate_size: {config_data.get('intermediate_size')}")

        config = AutoConfig.from_pretrained(BASE_MODEL)
        with torch.device("meta"):
            ref_model = AutoModelForCausalLM.from_config(config)
        ref_state = ref_model.state_dict()

        # Show layer 0 expert keys from the model class
        ref_expert_keys = sorted([k for k in ref_state.keys() if "layers.0." in k and "expert" in k.lower()])
        print(f"\n   Expected keys from model class (layer 0 experts):")
        for k in ref_expert_keys:
            print(f"   {k}: {list(ref_state[k].shape)}")

        # Compare naming conventions
        print(f"\n7. KEY NAMING COMPARISON:")
        print(f"   Base model on HF:  {sorted(layer0_expert_keys)[:3]}")
        print(f"   HF model class:    {ref_expert_keys[:3]}")

        # Check if they match
        hf_set = set(layer0_expert_keys)
        ref_set = set(ref_expert_keys)
        if hf_set == ref_set:
            print(f"   MATCH: HF weights and model class use same naming")
        else:
            only_hf = hf_set - ref_set
            only_ref = ref_set - hf_set
            if only_hf:
                print(f"   Only in HF weights ({len(only_hf)}):")
                for k in sorted(only_hf)[:5]:
                    print(f"     {k}")
            if only_ref:
                print(f"   Only in model class ({len(only_ref)}):")
                for k in sorted(only_ref)[:5]:
                    print(f"     {k}")

        del ref_model, ref_state

    except Exception as e:
        print(f"   Could not load config/model class: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
