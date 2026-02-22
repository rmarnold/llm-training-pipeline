"""Test a merged GPT-OSS 20B model locally on M2 Max (64GB unified memory).

Downloads a merged model from HuggingFace or loads from local path,
verifies key format, and runs inference to check for garbage output.

Usage:
    # Test a local merged model
    python TOOLS/test_merged_model_local.py --model_path checkpoints/gpt-oss-20b-merged/hf

    # Test the base model (no LoRA, verifies dequant works)
    python TOOLS/test_merged_model_local.py --model_path openai/gpt-oss-20b --base_model

Requires: transformers >= 5.0, torch >= 2.0
"""
import argparse
import json
import os
import sys


def validate_keys(model_path: str) -> bool:
    """Check that model keys match expected GptOss packed format."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    print("\n=== Key Validation ===")

    config = AutoConfig.from_pretrained(model_path)
    print(f"  Model type: {config.model_type}")
    print(f"  Architectures: {config.architectures}")

    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(config)
    expected = set(ref_model.state_dict().keys())
    expected_shapes = {k: tuple(v.shape) for k, v in ref_model.state_dict().items()}
    del ref_model

    # Check index
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        actual = set(index["weight_map"].keys())
    else:
        # Single file
        from safetensors import safe_open
        sf_path = os.path.join(model_path, "model.safetensors")
        with safe_open(sf_path, framework="pt") as f:
            actual = set(f.keys())

    missing = expected - actual
    extra = actual - expected

    if not missing and not extra:
        print(f"  All {len(actual)} keys match expected format")
        # Spot check shapes
        if os.path.exists(index_path):
            from safetensors import safe_open
            check_keys = sorted([k for k in actual if "layers.0.mlp.experts" in k])
            for key in check_keys:
                shard_file = index["weight_map"][key]
                shard_path = os.path.join(model_path, shard_file)
                with safe_open(shard_path, framework="pt") as f:
                    tensor = f.get_tensor(key)
                exp = expected_shapes.get(key)
                ok = "OK" if exp and tuple(tensor.shape) == exp else "MISMATCH"
                print(f"  {key}: {list(tensor.shape)} [{ok}]")
        return True
    else:
        if missing:
            print(f"  {len(missing)} MISSING keys (first 10):")
            for k in sorted(missing)[:10]:
                print(f"    {k}: expected {expected_shapes.get(k)}")
        if extra:
            print(f"  {len(extra)} EXTRA keys (first 10):")
            for k in sorted(extra)[:10]:
                print(f"    {k}")
        return False


def test_generation(model_path: str, device: str = "mps") -> None:
    """Load model and run inference test."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== Generation Test (device={device}) ===")

    print(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"  Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to("mps")

    print(f"  Model loaded: {type(model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test prompts
    prompts = [
        "Hello, how are you?",
        "Write a Python function that adds two numbers.",
        "What is the capital of France?",
    ]

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Response: {response[:300]}")

        # Basic sanity check
        words = response.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                print(f"  WARNING: Low unique word ratio ({unique_ratio:.2f}) — possible garbage output")
            else:
                print(f"  Looks coherent (unique word ratio: {unique_ratio:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Test merged GPT-OSS model locally")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model directory or HF model ID")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps"],
                        help="Device to load model on (default: cpu)")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate keys, don't run generation")
    args = parser.parse_args()

    import transformers
    print(f"transformers: {transformers.__version__}")
    import torch
    print(f"torch: {torch.__version__}")

    valid = validate_keys(args.model_path)

    if not valid:
        print("\nKey validation FAILED — model format may be incorrect.")
        if not args.validate_only:
            print("Attempting generation anyway...")

    if not args.validate_only:
        test_generation(args.model_path, device=args.device)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
