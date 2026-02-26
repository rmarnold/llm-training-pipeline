"""Merge LoRA adapter into base model and export.

Merges a trained LoRA adapter (lang_rust, core_agent, etc.) into the
GPT-OSS 20B base model and optionally exports to GGUF format.

Usage:
    # Merge lang_rust adapter (default config)
    python scripts/19_merge_adapter.py

    # Merge with custom config
    python scripts/19_merge_adapter.py --config configs/merge_adapter.yaml

    # Merge specific adapter
    python scripts/19_merge_adapter.py --adapter_path checkpoints/core_agent --output_dir merged/core_agent

    # Export to GGUF
    python scripts/19_merge_adapter.py --export_formats hf gguf_q4 gguf_q8

Requires: pip install -e ".[gpt_oss]"
"""
import os

import yaml

from pipeline_lib.unsloth_utils import merge_and_export


def merge_adapter(
    config_path: str = "configs/merge_adapter.yaml",
    cli_overrides: dict | None = None,
) -> None:
    """Merge LoRA adapter into base model and export.

    Args:
        config_path: Path to YAML config file.
        cli_overrides: Dict of CLI overrides.
    """
    if cli_overrides is None:
        cli_overrides = {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_model = cli_overrides.get("base_model", config["model"]["base_model"])
    adapter_path = cli_overrides.get("adapter_path", config["model"]["adapter_path"])
    output_dir = cli_overrides.get("output_dir", config["export"]["output_dir"])
    export_formats = cli_overrides.get("export_formats", config["export"].get("formats", ["hf"]))

    print(f"\n{'='*60}")
    print(f"Merging Adapter: {config.get('run_name', 'merge')}")
    print(f"{'='*60}")
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_dir}")
    print(f"  Formats: {export_formats}")

    if not os.path.exists(adapter_path):
        print(f"\nERROR: Adapter not found: {adapter_path}")
        print("Train an adapter first (13_train_lang_adapter.py or 14_train_core_agent.py).")
        return

    # Merge and export
    merge_and_export(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        export_formats=export_formats,
    )

    # Smoke test if configured
    if config.get("validation", {}).get("smoke_test", False):
        _run_smoke_test(
            output_dir=output_dir,
            test_prompt=config["validation"].get(
                "test_prompt",
                "Write a Rust function that adds two numbers.",
            ),
            max_new_tokens=config["validation"].get("max_new_tokens", 256),
        )

    print(f"\n{'='*60}")
    print(f"Merge complete!")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


def _run_smoke_test(
    output_dir: str,
    test_prompt: str,
    max_new_tokens: int = 256,
) -> None:
    """Validate merged model via safetensor inspection + inference.

    Two-phase validation:
    1. Direct safetensor weight inspection (no transformers version dependency)
    2. Text generation via AutoModelForCausalLM with eager attention
       (cleanup_merged_moe converts to unpacked format, eager avoids
       Flex Attention gibberish on GPT-OSS — Bug #3363)
    """
    import torch
    print(f"\nRunning smoke test...")

    hf_path = os.path.join(output_dir, "hf")
    if not os.path.exists(hf_path):
        hf_path = output_dir

    # Phase 1: Safetensor weight validation
    print(f"\n  Phase 1: Weight validation (safetensors)...")
    weights_ok = False
    try:
        from safetensors import safe_open

        st_files = sorted(
            f for f in os.listdir(hf_path) if f.endswith(".safetensors")
        )
        if not st_files:
            print(f"  FAILED: no .safetensors files in {hf_path}")
            return

        print(f"  Found {len(st_files)} safetensor file(s)")

        all_keys = set()
        nan_keys = []
        inf_keys = []
        sample_shapes = {}

        for fname in st_files:
            path = os.path.join(hf_path, fname)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    all_keys.add(key)
                    tensor = f.get_tensor(key)
                    if torch.isnan(tensor).any():
                        nan_keys.append(key)
                    if torch.isinf(tensor).any():
                        inf_keys.append(key)
                    if "layers.0." in key:
                        sample_shapes[key] = list(tensor.shape)

        print(f"  Total keys: {len(all_keys)}")

        packed_expert_keys = [k for k in all_keys if ".experts.down_proj" in k
                              and ".down_projs." not in k]
        unpacked_expert_keys = [k for k in all_keys if ".experts.down_projs." in k]

        if packed_expert_keys:
            print(f"  Packed expert keys: {len(packed_expert_keys)}")
            for key in sorted(packed_expert_keys)[:1]:
                shape = sample_shapes.get(key)
                if shape:
                    print(f"    {key}: {shape}")
        elif unpacked_expert_keys:
            print(f"  Unpacked expert keys: {len(unpacked_expert_keys)}")
            for key in sorted(unpacked_expert_keys)[:1]:
                shape = sample_shapes.get(key)
                if shape:
                    print(f"    {key}: {shape}")
        else:
            print(f"  WARNING: No expert keys found")

        router_keys = [k for k in all_keys if ".router." in k]
        attn_keys = [k for k in all_keys if ".self_attn." in k]
        print(f"  Router: {len(router_keys)}, Attention: {len(attn_keys)}, lm_head: {'yes' if 'lm_head.weight' in all_keys else 'MISSING'}")

        if nan_keys:
            print(f"  FAIL: {len(nan_keys)} keys contain NaN: {nan_keys[:5]}")
        if inf_keys:
            print(f"  FAIL: {len(inf_keys)} keys contain inf: {inf_keys[:5]}")

        has_experts = len(packed_expert_keys) > 0 or len(unpacked_expert_keys) > 0
        has_router = len(router_keys) > 0
        has_attn = len(attn_keys) > 0
        no_bad_values = len(nan_keys) == 0 and len(inf_keys) == 0

        if has_experts and has_router and has_attn and no_bad_values:
            weights_ok = True
            print(f"  Phase 1 PASSED")
        else:
            issues = []
            if not has_experts: issues.append("no expert keys")
            if not has_router: issues.append("no router keys")
            if not has_attn: issues.append("no attention keys")
            if not no_bad_values: issues.append("NaN/inf in weights")
            print(f"  Phase 1 FAILED: {', '.join(issues)}")

    except Exception as e:
        print(f"  Phase 1 FAILED: {e}")

    # Phase 2: Text generation via AutoModelForCausalLM with eager attention.
    # cleanup_merged_moe() converts packed expert format to unpacked nn.Linear
    # format, so AutoModelForCausalLM can load the model directly.
    #
    # Unsloth's FastLanguageModel configures Flex Attention internally, which
    # produces gibberish on GPT-OSS (Bug #3363). Using AutoModelForCausalLM
    # with attn_implementation="eager" avoids this entirely.
    print(f"\n  Phase 2: Inference test (eager attention)...")
    inference_ok = False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            hf_path, trust_remote_code=True,
        )
        model.eval()

        # Ensure pad_token_id is set (StopIteration fix)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        print(f"\n  {'─' * 50}")
        print(f"  MODEL RESPONSE:")
        print(f"  {'─' * 50}")
        for line in response[:500].split('\n'):
            print(f"  {line}")
        print(f"  {'─' * 50}")

        # Garbage detection: empty, low diversity, endoftext flooding, low letter ratio
        response_clean = response.strip()
        unique_chars = len(set(response_clean[:200]))
        endoftext_count = response_clean.count("<|endoftext|>")
        sample = response_clean[:300]
        letter_count = sum(1 for c in sample if c.isalpha())
        letter_ratio = letter_count / max(len(sample), 1)
        is_garbage = (
            response_clean == ""
            or unique_chars < 8
            or endoftext_count > 3
            or letter_ratio < 0.3
        )
        if is_garbage:
            print(f"  Phase 2 FAILED: garbage/empty output "
                  f"(unique_chars={unique_chars}, endoftext={endoftext_count}, "
                  f"letter_ratio={letter_ratio:.2f})")
        else:
            print(f"  Phase 2 PASSED: coherent output "
                  f"(unique_chars={unique_chars}, letter_ratio={letter_ratio:.2f})")
            inference_ok = True

        del model, tokenizer
        import gc; gc.collect()
        torch.cuda.empty_cache()

    except Exception as ex:
        import traceback
        print(f"  Phase 2 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()

    # Overall verdict — both phases must pass
    overall = weights_ok and inference_ok
    print(f"\n  Smoke test: {'PASSED' if overall else 'FAILED'}")
    if weights_ok and not inference_ok:
        print(f"  (Weights OK but inference failed — model may still work for training)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", type=str, default="configs/merge_adapter.yaml")
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--adapter_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--export_formats", nargs="+", type=str,
                        help="Export formats: hf, gguf_q4, gguf_q8, gguf_f16")
    args = parser.parse_args()

    cli_overrides = {}
    for key in ["base_model", "adapter_path", "output_dir", "export_formats"]:
        val = getattr(args, key)
        if val is not None:
            cli_overrides[key] = val

    merge_adapter(config_path=args.config, cli_overrides=cli_overrides)
