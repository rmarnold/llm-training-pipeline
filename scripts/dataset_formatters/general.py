"""General instruction dataset formatters: Alpaca, OpenOrca, OASST."""
from __future__ import annotations


def format_alpaca(example):
    """Format Alpaca instruction data."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{output}<|endoftext|>"
    }


def format_openorca(example):
    """Format OpenOrca GPT-4 distillation data.

    OpenOrca format: {"system_prompt": "...", "question": "...", "response": "..."}
    """
    system = example.get("system_prompt", "")
    question = example.get("question", "")
    response = example.get("response", "")

    if system and system.strip():
        prompt = f"{system}\n\n{question}"
    else:
        prompt = question

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_oasst(example):
    """Format OpenAssistant data."""
    # OASST has tree structure, we use the instruction/response pairs
    instruction = example.get("instruction", example.get("text", ""))
    response = example.get("response", "")

    if not response:
        return {"text": ""}

    return {
        "text": f"<|user|>\n{instruction}\n<|assistant|>\n{response}<|endoftext|>"
    }
