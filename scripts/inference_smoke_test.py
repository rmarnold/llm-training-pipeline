"""Run a single inference smoke test and print JSON results.

Designed to run as a subprocess so ALL CUDA memory is freed on exit.
Quality gate cells in the notebook call this instead of loading models
in the notebook kernel (which leaks VRAM due to Unsloth internals).

Usage:
    python scripts/inference_smoke_test.py \
        --checkpoint checkpoints/agent_sft/final \
        --prompt "..." \
        --max_tokens 300 \
        --check_actions          # check for action-taking vs stalling
"""
import argparse
import json
import sys


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
        "takes_action": None,
        "stalls": None,
        "error": None,
    }

    try:
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            args.checkpoint,
            max_seq_length=8192,
            load_in_4bit=False,      # merged model is bf16
            dtype=torch.bfloat16,
        )
        # GPT-OSS + Flex Attention = gibberish (Unsloth Bug #3363).
        # Use eager attention via model.set_mode() instead of for_inference().
        model.eval()

        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.1,
                do_sample=True,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        result["response"] = response[:500]
        result["success"] = True

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
        result["error"] = str(e)

    # Print JSON on a known-prefix line for easy parsing
    print("SMOKE_RESULT:" + json.dumps(result), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
