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

from pipeline_lib.unsloth_utils import load_unsloth_model, merge_and_export


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
    """Run a quick generation test on the merged model."""
    print(f"\nRunning smoke test...")

    hf_path = os.path.join(output_dir, "hf")
    if not os.path.exists(hf_path):
        hf_path = output_dir

    try:
        model, tokenizer = load_unsloth_model(
            model_name=hf_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        from dataset_formatters.harmony import encode_harmony_messages
        prompt_text = encode_harmony_messages([
            {"role": "user", "content": test_prompt},
        ], developer_instructions="You are a Rust programming expert.")

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"  Prompt: {test_prompt}")
        print(f"  Response (first 500 chars): {response[:500]}")
        print(f"  Smoke test PASSED")

    except Exception as e:
        print(f"  Smoke test FAILED: {e}")


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
