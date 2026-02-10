"""Function calling dataset formatters: Glaive, Hermes, Gorilla, ToolBench."""
from __future__ import annotations


def format_glaive_function_calling(example):
    """Format Glaive function calling data.

    Glaive format: {"system": "...", "chat": "..."}
    Chat contains USER/ASSISTANT/FUNCTION_RESPONSE turns
    """
    system = example.get("system", "")
    chat = example.get("chat", "")

    # The chat already has structured format, wrap it appropriately
    if system:
        text = f"<|system|>\n{system}\n{chat}<|endoftext|>"
    else:
        text = f"{chat}<|endoftext|>"

    # Normalize the format
    text = text.replace("USER:", "<|user|>\n")
    text = text.replace("ASSISTANT:", "<|assistant|>\n")
    text = text.replace("FUNCTION RESPONSE:", "<|function_response|>\n")

    return {"text": text}


def format_hermes_function_calling(example):
    """Format Hermes/NousResearch function calling data.

    Hermes format: {"conversations": [{"from": "...", "value": "..."}], "functions": [...]}
    """
    conversations = example.get("conversations", [])
    functions = example.get("functions", [])

    parts = []

    # Add function definitions as system context
    if functions:
        import json
        func_str = json.dumps(functions, indent=2)
        parts.append(f"<|system|>\nYou have access to the following functions:\n{func_str}\n")

    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")

        if role == "human":
            parts.append(f"<|user|>\n{value}\n")
        elif role == "gpt":
            parts.append(f"<|assistant|>\n{value}")
        elif role == "function_call":
            parts.append(f"<|function_call|>\n{value}\n")
        elif role == "function_response":
            parts.append(f"<|function_response|>\n{value}\n")

    parts.append("<|endoftext|>")

    return {"text": "".join(parts)}


def format_gorilla(example):
    """Format Gorilla OpenFunctions dataset.

    Format: {"question": "...", "function": [...], "answer": "..."}
    """
    question = example.get("question", example.get("instruction", ""))
    answer = example.get("answer", example.get("output", ""))
    functions = example.get("function", [])

    if functions:
        import json
        func_str = json.dumps(functions, indent=2) if isinstance(functions, list) else str(functions)
        prompt = f"Available functions:\n{func_str}\n\nUser request: {question}"
    else:
        prompt = question

    return {
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}<|endoftext|>"
    }


def format_toolbench(example):
    """Format ToolBench API calling data.

    ToolBench has complex multi-turn API interactions.
    Format varies, typically: {"conversations": [...], "tools": [...]}
    """
    conversations = example.get("conversations", [])

    if not conversations:
        # Try alternate format
        query = example.get("query", example.get("instruction", ""))
        response = example.get("response", example.get("output", ""))
        if query and response:
            return {
                "text": f"<|user|>\n{query}\n<|assistant|>\n{response}<|endoftext|>"
            }
        return {"text": ""}

    parts = []
    for turn in conversations:
        role = turn.get("role", turn.get("from", ""))
        content = turn.get("content", turn.get("value", ""))

        if role in ["user", "human"]:
            parts.append(f"<|user|>\n{content}\n")
        elif role in ["assistant", "gpt", "model"]:
            parts.append(f"<|assistant|>\n{content}")
        elif role == "tool":
            parts.append(f"<|tool_response|>\n{content}\n")

    parts.append("<|endoftext|>")
    return {"text": "".join(parts)}
