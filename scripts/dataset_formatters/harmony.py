"""Harmony format dataset formatters for GPT-OSS models.

Harmony is OpenAI's chat format introduced with GPT-OSS (Aug 2025).
It supports:
- Standard messages (system/user/assistant/tool roles)
- Tool calls with structured function calling
- Thinking/reasoning fields (chain-of-thought)
- Multi-channel output (analysis, commentary, final)

All GPT-OSS training data MUST be in Harmony format.
The model degrades without it.

Formatters in this module:
- harmony_code: Raw code files for continued pretraining
- harmony_completion: Code completion (fill-in-the-middle)
- harmony_agent: Multi-turn agent trajectories with tool calls
- harmony_preference: Preference pairs for IPO/DPO
- harmony_task: Coding tasks with tests for GRPO
- harmony_debug: Bug fix examples (broken code + error → fix)

Reference: https://github.com/openai/harmony
"""
from __future__ import annotations

import json
from typing import Any


# ==========================================================================
# Harmony Encoding
# ==========================================================================

def encode_harmony_messages(
    messages: list[dict[str, Any]],
    developer_instructions: str | None = None,
    reasoning_effort: str = "medium",
    add_generation_prompt: bool = False,
) -> str:
    """Encode messages in Harmony format with proper special tokens.

    Harmony uses a strict role hierarchy: system > developer > user > assistant > tool

    Args:
        messages: List of message dicts with role, content, and optional
            tool_calls, tool_call_id, thinking fields.
        developer_instructions: Optional developer-level system prompt.
        reasoning_effort: "low", "medium", or "high" reasoning effort.
        add_generation_prompt: If True, end with ``<|assistant|>\\n`` instead
            of ``<|endoftext|>``.  Use for inference/generation so the model
            knows it should produce an assistant reply.

    Returns:
        Formatted text string with Harmony tokens.
    """
    try:
        from openai_harmony import encode_conversations_with_harmony
        return encode_conversations_with_harmony(
            messages=messages,
            reasoning_effort=reasoning_effort,
            developer_instructions=developer_instructions,
        )
    except ImportError:
        # Fallback: manual Harmony encoding when openai-harmony not installed
        return _encode_harmony_fallback(messages, developer_instructions, add_generation_prompt)


def _encode_harmony_fallback(
    messages: list[dict[str, Any]],
    developer_instructions: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Manual Harmony-compatible encoding as fallback.

    This produces a format compatible with GPT-OSS training.
    Uses the same special tokens as the official openai-harmony package.
    """
    parts = []

    if developer_instructions:
        parts.append(f"<|developer|>\n{developer_instructions}\n")

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id")
        thinking = msg.get("thinking")
        name = msg.get("name", "")

        if role == "system":
            parts.append(f"<|system|>\n{content}\n")

        elif role == "developer":
            parts.append(f"<|developer|>\n{content}\n")

        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")

        elif role == "assistant":
            if thinking:
                parts.append(f"<|thinking|>\n{thinking}\n<|/thinking|>\n")

            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    tc_name = func.get("name", "")
                    tc_args = func.get("arguments", "{}")
                    if isinstance(tc_args, dict):
                        tc_args = json.dumps(tc_args)
                    tc_id = tc.get("id", "")
                    parts.append(
                        f'<|tool_call|>\n{{"id": "{tc_id}", "name": "{tc_name}", "arguments": {tc_args}}}\n'
                    )

            if content:
                parts.append(f"<|assistant|>\n{content}\n")

        elif role == "tool":
            if tool_call_id:
                parts.append(f"<|tool_result|>\n<|tool_call_id|>{tool_call_id}\n{content}\n")
            elif name:
                parts.append(f"<|tool_result|>\n<|name|>{name}\n{content}\n")
            else:
                parts.append(f"<|tool_result|>\n{content}\n")

    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    else:
        parts.append("<|endoftext|>")
    return "".join(parts)


def validate_harmony_format(text: str) -> tuple[bool, str | None]:
    """Validate that text is well-formed Harmony format.

    Returns:
        (is_valid, error_message) tuple.
    """
    if not text or not text.strip():
        return False, "Empty text"

    if not text.strip().endswith("<|endoftext|>"):
        return False, "Missing <|endoftext|> token"

    role_tokens = ["<|system|>", "<|developer|>", "<|user|>", "<|assistant|>",
                   "<|tool_call|>", "<|tool_result|>", "<|thinking|>"]
    has_role = any(token in text for token in role_tokens)
    if not has_role:
        return False, "No role tokens found"

    return True, None


# ==========================================================================
# Code Completion Formatters
# ==========================================================================

def format_harmony_code(example: dict[str, Any]) -> dict[str, str]:
    """Format raw code files for continued pretraining.

    Input: {"content": "...", "path": "src/main.rs"} or {"text": "..."}
    """
    content = example.get("content", example.get("text", ""))
    path = example.get("path", "unknown.rs")

    if not content or len(content.strip()) < 50:
        return {"text": ""}

    messages = [
        {"role": "assistant", "content": content},
    ]

    return {"text": encode_harmony_messages(
        messages,
        developer_instructions=f"You are a Rust programming expert. File: {path}",
        reasoning_effort="low",
    )}


def format_harmony_completion(example: dict[str, Any]) -> dict[str, str]:
    """Format code completion examples.

    Input: {"prompt"/"instruction": "...", "completion"/"solution"/"output": "..."}
    """
    prompt = example.get("prompt", example.get("instruction", ""))
    completion = example.get("completion", example.get("solution", example.get("output", "")))
    context = example.get("context", "")

    if not prompt or not completion:
        return {"text": ""}

    user_content = prompt
    if context:
        user_content = f"{context}\n\n{prompt}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": completion},
    ]

    return {"text": encode_harmony_messages(
        messages,
        developer_instructions="You are a Rust programming expert. Complete the code.",
        reasoning_effort="low",
    )}


# ==========================================================================
# Agent Trajectory Formatter
# ==========================================================================

def format_harmony_agent(example: dict[str, Any]) -> dict[str, str]:
    """Format multi-turn agent trajectories with tool calls and thinking.

    Input: {
        "task": "Fix the failing test in src/parser.rs",
        "trajectory": [
            {
                "thinking": "The test name suggests a parsing issue...",
                "tool_calls": [
                    {"id": "1", "name": "run_tests", "arguments": {"cmd": "cargo test test_parse"}}
                ],
                "tool_results": [
                    {"id": "1", "output": "FAILED: ..."}
                ],
                "response": "Let me look at the failing test."  # Optional
            },
            ...
        ]
    }
    """
    task = example.get("task", "")
    trajectory = example.get("trajectory", [])
    developer_prompt = example.get(
        "developer_prompt",
        "You are a Rust coding agent. Use tools to read, modify, and test code. "
        "Always run tests after patching."
    )

    if not task or not trajectory:
        return {"text": ""}

    messages = [
        {"role": "user", "content": task},
    ]

    for step in trajectory:
        thinking = step.get("thinking", "")
        tool_calls = step.get("tool_calls", [])
        tool_results = step.get("tool_results", [])
        response = step.get("response", "")

        # Assistant turn: thinking + tool calls or response
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if thinking:
            assistant_msg["thinking"] = thinking
        if tool_calls:
            formatted_calls = []
            for tc in tool_calls:
                formatted_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {}))
                        if isinstance(tc.get("arguments"), dict)
                        else tc.get("arguments", "{}"),
                    },
                })
            assistant_msg["tool_calls"] = formatted_calls
        if response:
            assistant_msg["content"] = response
        elif not tool_calls:
            assistant_msg["content"] = ""

        messages.append(assistant_msg)

        # Tool results
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr.get("id", ""),
                "content": tr.get("output", ""),
            })

    return {"text": encode_harmony_messages(
        messages,
        developer_instructions=developer_prompt,
        reasoning_effort="high",
    )}


# ==========================================================================
# Preference Pair Formatter (IPO/DPO)
# ==========================================================================

def format_harmony_preference(example: dict[str, Any]) -> dict[str, str]:
    """Format preference pairs for IPO/DPO training.

    Input: {
        "prompt": "Fix this Rust code...",
        "chosen": "Here's the corrected code...",
        "rejected": "Try changing this line...",
    }

    Returns dict with prompt/chosen/rejected keys for DPO/IPO trainers,
    plus a "text" key (chosen version) for compatibility with FORMAT_HANDLERS.
    """
    prompt = example.get("prompt", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")

    if not all([prompt, chosen, rejected]):
        return {"text": ""}

    dev_instructions = "You are a Rust programming expert."

    chosen_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected},
    ]

    chosen_text = encode_harmony_messages(chosen_messages, developer_instructions=dev_instructions)
    rejected_text = encode_harmony_messages(rejected_messages, developer_instructions=dev_instructions)

    return {
        "text": chosen_text,  # For FORMAT_HANDLERS compatibility
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


# ==========================================================================
# Task Formatter (GRPO)
# ==========================================================================

def format_harmony_task(example: dict[str, Any]) -> dict[str, str]:
    """Format coding tasks with tests for GRPO RL training.

    Input: {
        "description": "Implement a function that...",
        "starter_code": "fn solve(...) -> ... { todo!() }",
        "tests": "#[test] fn test_basic() { ... }",
        "solution": "fn solve(...) -> ... { ... }"  # Optional
    }
    """
    description = example.get("description", "")
    starter_code = example.get("starter_code", "")
    tests = example.get("tests", "")
    solution = example.get("solution", "")

    if not description:
        return {"text": ""}

    user_content = description
    if starter_code:
        user_content += f"\n\nStarter code:\n```rust\n{starter_code}\n```"
    if tests:
        user_content += f"\n\nTests:\n```rust\n{tests}\n```"

    messages = [
        {"role": "user", "content": user_content},
    ]

    if solution:
        messages.append({"role": "assistant", "content": solution})

    text = encode_harmony_messages(
        messages,
        developer_instructions="You are a Rust programming expert. Write correct, idiomatic code.",
        reasoning_effort="high",
    )

    result: dict[str, str] = {"text": text}
    if tests:
        result["tests"] = tests
    if not solution:
        # For RL: return prompt-only text for the policy to complete
        prompt_messages = [{"role": "user", "content": user_content}]
        result["prompt_text"] = encode_harmony_messages(
            prompt_messages,
            developer_instructions="You are a Rust programming expert. Write correct, idiomatic code.",
            reasoning_effort="high",
        )
    return result


# ==========================================================================
# Debug/Fix Formatter
# ==========================================================================

def format_harmony_debug(example: dict[str, Any]) -> dict[str, str]:
    """Format bug fix examples (broken code + error → fix).

    Input: {
        "buggy_code": "fn main() { let x: &str = String::new(); }",
        "error_message": "error[E0308]: mismatched types...",
        "fixed_code": "fn main() { let x: String = String::new(); }",
        "explanation": "The type annotation was wrong..."  # Optional
    }
    """
    buggy_code = example.get("buggy_code", "")
    error_message = example.get("error_message", example.get("compiler_error", ""))
    fixed_code = example.get("fixed_code", example.get("fix", ""))
    explanation = example.get("explanation", "")

    if not all([buggy_code, error_message, fixed_code]):
        return {"text": ""}

    user_content = (
        f"This Rust code has an error:\n\n```rust\n{buggy_code}\n```\n\n"
        f"Compiler error:\n```\n{error_message}\n```\n\n"
        f"Fix the code."
    )

    assistant_content = ""
    if explanation:
        assistant_content += f"{explanation}\n\n"
    assistant_content += f"```rust\n{fixed_code}\n```"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"text": encode_harmony_messages(
        messages,
        developer_instructions="You are a Rust debugging expert. Fix compilation and runtime errors.",
        reasoning_effort="medium",
    )}
