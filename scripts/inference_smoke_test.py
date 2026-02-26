"""Run a single inference smoke test and print JSON results.

Designed to run as a subprocess so ALL CUDA memory is freed on exit.
Quality gate cells in the notebook call this instead of loading models
in the notebook kernel (which leaks VRAM due to Unsloth internals).

Model loading strategy:
- Merged models (no adapter_config.json): AutoModelForCausalLM with
  attn_implementation="eager" to avoid Flex Attention gibberish
  (Unsloth Bug #3363).
- Adapter checkpoints (has adapter_config.json): Unsloth FastLanguageModel
  to load base + adapter.

Usage:
    python scripts/inference_smoke_test.py \
        --checkpoint checkpoints/agent_sft/final \
        --prompt "..." \
        --max_tokens 300 \
        --check_actions          # check for action-taking vs stalling
"""
import argparse
import json
import os
import sys


def _is_garbage(response: str) -> bool:
    """Detect garbage/incoherent model output.

    Checks for:
    - Empty or whitespace-only output
    - Very low character diversity (repetitive patterns)
    - Excessive <|endoftext|> flooding
    - Low letter ratio (gibberish has lots of special chars, parens, etc.)
    """
    clean = response.strip()
    if not clean:
        return True

    # Character diversity: fewer than 8 unique chars in first 200
    unique_chars = len(set(clean[:200]))
    if unique_chars < 8:
        return True

    # Endoftext flooding: more than 3 in the response
    endoftext_count = clean.count("<|endoftext|>")
    if endoftext_count > 3:
        return True

    # Letter ratio: coherent text should have >30% letters
    # Gibberish like "Soás B B (^<|endoftext|>:..." has low letter ratio
    sample = clean[:300]
    letter_count = sum(1 for c in sample if c.isalpha())
    letter_ratio = letter_count / max(len(sample), 1)
    if letter_ratio < 0.3:
        return True

    return False


def _load_merged_model(checkpoint: str):
    """Load a merged model with AutoModelForCausalLM + eager attention.

    After cleanup_merged_moe(), the model is in standard unpacked format
    loadable by transformers. Using eager attention avoids the Flex Attention
    gibberish issue with GPT-OSS (Unsloth Bug #3363).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def _load_adapter_model(checkpoint: str):
    """Load an adapter checkpoint via Unsloth (base model + LoRA adapter)."""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        checkpoint,
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--check_actions", action="store_true",
                        help="Check for action-taking vs stalling behavior")
    args = parser.parse_args()

    result = {
        "success": False,
        "response": "",
        "is_garbage": None,
        "takes_action": None,
        "stalls": None,
        "error": None,
    }

    try:
        import torch

        # Detect checkpoint type: adapter vs merged/full model
        is_adapter = os.path.exists(
            os.path.join(args.checkpoint, "adapter_config.json")
        )

        if is_adapter:
            model, tokenizer = _load_adapter_model(args.checkpoint)
        else:
            model, tokenizer = _load_merged_model(args.checkpoint)

        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        result["response"] = response[:500]
        result["success"] = True
        result["is_garbage"] = _is_garbage(response)

        if args.check_actions:
            result["takes_action"] = any(
                tok in response
                for tok in ["apply_patch", "write", "impl", "fn parse",
                            "fn ", "<|tool_call|>"]
            )
            result["stalls"] = any(
                phrase in response.lower()
                for phrase in [
                    "i would need to", "let me analyze",
                    "i need more context", "further investigation",
                    "i cannot",
                ]
            )

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    # Print JSON on a known-prefix line for easy parsing
    print("SMOKE_RESULT:" + json.dumps(result), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
