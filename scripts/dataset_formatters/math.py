"""Math dataset formatters: GSM8K, Orca-Math, MetaMath, MathInstruct."""
from __future__ import annotations


def format_gsm8k(example):
    """Format GSM8K math problems with chain-of-thought.

    GSM8K format: {"question": "...", "answer": "..."}
    Answer contains step-by-step reasoning ending with #### final_answer
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    # GSM8K answers have reasoning followed by #### and the final answer
    # Keep the full reasoning chain
    prompt = f"Solve this math problem step by step:\n\n{question}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"
    }


def format_orca_math(example):
    """Format Orca-Math problems.

    Orca-Math format: {"question": "...", "answer": "..."}
    """
    question = example.get("question", "")
    answer = example.get("answer", "")

    prompt = f"Solve this problem step by step:\n\n{question}"

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"
    }


def format_metamath(example):
    """Format MetaMathQA dataset.

    MetaMath format: {"query": "...", "response": "..."}
    Contains augmented math problems with detailed solutions.
    """
    query = example.get("query", "")
    response = example.get("response", "")

    return {
        "text": f"<|user|>\n{query}\n<|assistant|>\n{response}<|endoftext|>"
    }


def format_mathinstruct(example):
    """Format MathInstruct dataset (replaces CoT-Collection).

    MathInstruct format: {"instruction": "...", "output": "..."}
    Contains math problems with step-by-step solutions.
    """
    instruction = example.get("instruction", "")
    output = example.get("output", "")

    return {
        "text": f"<|user|>\n{instruction}\n<|assistant|>\n{output}<|endoftext|>"
    }
