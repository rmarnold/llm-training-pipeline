"""TUI (Terminal UI) format-based evaluators for GRPO.

Unlike language-specific evaluators (Rust/Python/etc.) that run code in sandboxes,
TUI evaluators score completions based on structural quality:
- Harmony format compliance (proper message structure)
- Tool call validity (well-formed JSON arguments, known tool names)
- Response completeness (not truncated, has conclusion)
- Multi-turn coherence (follows conversation flow)

These rewards train the model to produce well-structured agent responses
without requiring an execution environment.

Reward breakdown (each sub-score contributes to a 0.0–1.0 total):
  harmony_format_score  : 0.0–0.5  (structure & ending tokens)
  tool_call_score       : 0.0–0.3  (valid JSON args, present tool name)
  completeness_score    : 0.0–0.3  (not truncated mid-sentence/block)

Penalties applied after summing:
  empty_penalty         : -0.5  (response is blank)
  repetition_penalty    : -0.3  (same 50+ char chunk repeats 3+ times)

Final reward is clamped to [-0.5, 1.0].
"""
from __future__ import annotations

import json
import re
from typing import Any

from pipeline_lib.evaluator_dispatch import register_evaluator

# ---------------------------------------------------------------------------
# Harmony token constants
# ---------------------------------------------------------------------------

_HARMONY_ROLE_TOKENS = [
    "<|assistant|>",
    "<|user|>",
    "<|developer|>",
    "<|system|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|thinking|>",
]

_HARMONY_ENDING_TOKENS = [
    "<|endoftext|>",
    "<|end|>",
]

# Tokens that signal a clean conversational conclusion
_CONCLUSION_MARKERS = [
    "<|endoftext|>",
    "<|end|>",
]

# Patterns that indicate a truncated mid-code-block response
_TRUNCATION_PATTERNS = [
    r"```[a-zA-Z]*\s*$",   # Opened code fence, never closed
    r"\.\.\.\s*$",          # Trailing ellipsis (generation cut off)
    r"[,;{(\[]\s*$",        # Ends mid-expression (dangling punctuation)
]

# ---------------------------------------------------------------------------
# Sub-scoring functions
# ---------------------------------------------------------------------------

def harmony_format_score(completion: str) -> float:
    """Score Harmony format compliance.

    Scoring:
      +0.3  if <|assistant|> token is present (model adopted Harmony role)
      +0.2  if a proper ending token is present (clean conversation end)

    Args:
        completion: Raw completion string from the model.

    Returns:
        Float in [0.0, 0.5].
    """
    score = 0.0

    if "<|assistant|>" in completion:
        score += 0.3

    if any(tok in completion for tok in _HARMONY_ENDING_TOKENS):
        score += 0.2

    return score


def tool_call_score(
    completion: str,
    reward_config: dict[str, float] | None = None,
) -> float:
    """Score tool call validity.

    Strategy:
    - If no tool call markers exist, assume this is a pure-text response
      and award the default score (model correctly chose not to call a tool).
    - If <|tool_call|> markers exist, parse each JSON block and check:
        +0.1  per tool call that has a "name" field
        +0.2  if all tool call argument strings parse as valid JSON

    The final score is capped at the weight specified in reward_config
    (default 0.3).

    Args:
        completion: Raw completion string from the model.
        reward_config: Optional overrides; reads 'tool_call_weight' key.

    Returns:
        Float in [0.0, 0.3].
    """
    cfg = reward_config or {}
    max_score = float(cfg.get("tool_call_weight", 0.3))

    if "<|tool_call|>" not in completion:
        # Pure-text response — not wrong, give partial credit
        return min(0.2, max_score)

    # Extract JSON blocks that follow each <|tool_call|> marker
    # The fallback Harmony encoder writes: <|tool_call|>\n{...}\n
    raw_blocks = completion.split("<|tool_call|>")[1:]

    has_names = 0
    all_args_valid = True
    parsed_count = 0

    for block in raw_blocks:
        # Take only the first line/block up to next token boundary
        json_text = block.split("<|")[0].strip()
        if not json_text:
            all_args_valid = False
            continue

        try:
            parsed = json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON — penalize the arg score
            all_args_valid = False
            continue

        parsed_count += 1

        # Check for tool name (flat or nested under "function")
        name = parsed.get("name") or parsed.get("function", {}).get("name", "")
        if name:
            has_names += 1

        # Check arguments sub-field
        args = parsed.get("arguments", parsed.get("function", {}).get("arguments"))
        if isinstance(args, str):
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                all_args_valid = False

    score = 0.0
    if has_names > 0:
        score += 0.1
    if parsed_count > 0 and all_args_valid:
        score += 0.2

    return min(score, max_score)


def completeness_score(
    completion: str,
    reward_config: dict[str, float] | None = None,
) -> float:
    """Score response completeness (not truncated).

    A completion is considered truncated if it:
    - Ends with an unclosed code fence (``` with no closing ```)
    - Ends mid-expression (dangling {, (, [, ,, ;)
    - Has trailing ellipsis implying cut-off

    Args:
        completion: Raw completion string from the model.
        reward_config: Optional overrides; reads 'completeness_weight' key.

    Returns:
        Float: 0.3 (or configured weight) if complete, 0.0 if truncated.
    """
    cfg = reward_config or {}
    full_score = float(cfg.get("completeness_weight", 0.3))

    stripped = completion.strip()
    if not stripped:
        return 0.0

    for pattern in _TRUNCATION_PATTERNS:
        if re.search(pattern, stripped):
            return 0.0

    # Count all triple-backtick occurrences.  An odd count means an unclosed
    # code fence (an open without a matching close), which signals truncation.
    fence_count = len(re.findall(r"```", stripped))
    if fence_count % 2 != 0:
        return 0.0

    return full_score


def _detect_repetition(completion: str, min_chunk: int = 40, min_repeats: int = 3) -> bool:
    """Detect degenerate repetition loops in a completion.

    Checks whether the text starts with any repeating unit of length
    min_chunk..n//min_repeats that is consecutively repeated at least
    min_repeats times.  This is a common failure mode for RL-trained
    models (the "repetition trap").

    The algorithm is O(n^2) worst case but fast in practice because it
    only needs to confirm the first repeating block from the start of
    the string, which usually terminates early.

    Args:
        completion: Raw completion string to inspect.
        min_chunk: Minimum repeating-unit length to consider (default: 40 chars).
            Lowered from 50 so that real-world ~48-char repeated sentences are
            caught reliably.
        min_repeats: Number of consecutive whole-unit repetitions to flag
            (default: 3).

    Returns:
        True if a repetition loop is detected, False otherwise.
    """
    n = len(completion)
    if n < min_chunk * min_repeats:
        return False

    max_unit = n // min_repeats

    for unit_len in range(min_chunk, max_unit + 1):
        unit = completion[:unit_len]
        # Check whether the string begins with unit repeated min_repeats times
        target_len = unit_len * min_repeats
        if completion[:target_len] == unit * min_repeats:
            return True

    return False


# ---------------------------------------------------------------------------
# Composite reward function
# ---------------------------------------------------------------------------

def compute_execution_reward(
    code: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reward_config: dict[str, float] | None = None,
) -> float:
    """Compute format-based reward for TUI GRPO training.

    No execution sandbox is used. The reward is derived entirely from the
    structural quality of the completion.

    Scoring pipeline:
      1. Apply empty-response penalty immediately if applicable.
      2. Sum sub-scores: harmony_format + tool_call + completeness.
      3. Apply repetition penalty if degenerate loops are detected.
      4. Clamp final reward to [-0.5, 1.0].

    Custom reward_config keys (all optional):
      harmony_format_weight   (default: 0.3 — max for assistant-marker check)
      tool_call_weight        (default: 0.3 — max for valid tool calls)
      completeness_weight     (default: 0.3 — max for non-truncated)
      empty_penalty           (default: -0.5)
      repetition_penalty      (default: -0.3)

    Args:
        code: The model's completion (treated as a text response; the "code"
            name is kept for protocol compatibility with other evaluators).
        tool_calls: Ignored for TUI evaluator (tool calls are parsed from
            the completion text directly).
        reward_config: Optional dict of custom reward weights/penalties.

    Returns:
        Float reward in approximately [-0.5, 1.0].
    """
    cfg = reward_config or {}
    empty_penalty = float(cfg.get("empty_penalty", -0.5))
    repetition_penalty = float(cfg.get("repetition_penalty", -0.3))

    # --- Empty response ---
    if not code or not code.strip():
        return empty_penalty

    # --- Sub-scores ---
    fmt_score = harmony_format_score(code)
    tc_score = tool_call_score(code, reward_config=cfg)
    comp_score = completeness_score(code, reward_config=cfg)

    total = fmt_score + tc_score + comp_score

    # --- Repetition penalty ---
    if _detect_repetition(code):
        total += repetition_penalty

    # Clamp to [empty_penalty, 1.0]
    return max(empty_penalty, min(1.0, total))


# ---------------------------------------------------------------------------
# Solution ranking (used by IPO preference pair generation)
# ---------------------------------------------------------------------------

def rank_solutions_by_execution(
    solutions: list[str],
    tests_code: str | None = None,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Rank TUI completions by structural quality for IPO preference pairs.

    Scores each solution using compute_execution_reward and returns them
    sorted best-first. The details dict describes which sub-checks passed.

    Args:
        solutions: List of completion strings to rank.
        tests_code: Unused for TUI evaluator (kept for protocol compatibility).

    Returns:
        List of (solution, score, details) tuples, sorted descending by score.
    """
    ranked: list[tuple[str, float, dict[str, bool]]] = []

    for solution in solutions:
        fmt = harmony_format_score(solution)
        tc = tool_call_score(solution)
        comp = completeness_score(solution)
        has_repetition = _detect_repetition(solution)

        score = compute_execution_reward(solution)

        details: dict[str, bool] = {
            "has_assistant_marker": "<|assistant|>" in solution,
            "has_ending_token": any(tok in solution for tok in _HARMONY_ENDING_TOKENS),
            "has_tool_calls": "<|tool_call|>" in solution,
            "tool_calls_valid": tc >= 0.2,
            "response_complete": comp > 0.0,
            "has_repetition": has_repetition,
        }

        ranked.append((solution, score, details))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Registered evaluator class
# ---------------------------------------------------------------------------

@register_evaluator("tui")
class TUIEvaluator:
    """Format-based evaluator for TUI agent GRPO training.

    Scores completions on Harmony structure, tool call validity, and response
    completeness. No execution sandbox required.
    """

    def compute_execution_reward(
        self,
        code: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reward_config: dict[str, float] | None = None,
    ) -> float:
        """Compute format-based reward for a single completion.

        Args:
            code: Model completion to score.
            tool_calls: Ignored (parsed from completion text directly).
            reward_config: Optional custom weights/penalties.

        Returns:
            Float reward in approximately [-0.5, 1.0].
        """
        return compute_execution_reward(code, tool_calls=tool_calls, reward_config=reward_config)

    def rank_solutions_by_execution(
        self,
        solutions: list[str],
        tests_code: str | None = None,
    ) -> list[tuple[str, float, dict[str, bool]]]:
        """Rank completions by structural quality, best-first.

        Args:
            solutions: List of completion strings.
            tests_code: Unused (kept for protocol compatibility).

        Returns:
            List of (solution, score, details) tuples sorted descending.
        """
        return rank_solutions_by_execution(solutions, tests_code=tests_code)
