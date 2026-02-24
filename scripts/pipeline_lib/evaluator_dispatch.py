"""Language-agnostic evaluator dispatch for GRPO and IPO.

Registry pattern for language-specific code evaluators so GRPO can dispatch
rewards by language. Each evaluator class wraps language-specific execution
functions (cargo check/test/clippy for Rust, pytest/mypy/ruff for Python, etc.).

Usage:
    from pipeline_lib.evaluator_dispatch import compute_execution_reward

    # Defaults to rust (backward compatible)
    reward = compute_execution_reward(code)

    # Explicit language
    reward = compute_execution_reward(code, language="python")
"""
from __future__ import annotations

from typing import Any, Protocol


class LanguageEvaluator(Protocol):
    """Protocol for language-specific evaluators."""

    def compute_execution_reward(
        self,
        code: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reward_config: dict[str, float] | None = None,
    ) -> float: ...

    def rank_solutions_by_execution(
        self,
        solutions: list[str],
        tests_code: str | None = None,
    ) -> list[tuple[str, float, dict[str, bool]]]: ...


# Registry: language name -> evaluator class
EVALUATORS: dict[str, type] = {}


def register_evaluator(language: str):
    """Decorator to register a language evaluator class.

    Args:
        language: Language identifier (e.g., "rust", "python").

    Returns:
        Decorator that registers the class and returns it unchanged.
    """
    def decorator(cls: type) -> type:
        EVALUATORS[language] = cls
        return cls
    return decorator


def get_evaluator(language: str) -> LanguageEvaluator:
    """Look up and instantiate evaluator by language.

    Args:
        language: Language identifier.

    Returns:
        Instantiated evaluator.

    Raises:
        ValueError: If no evaluator is registered for the language.
    """
    if language not in EVALUATORS:
        available = ", ".join(sorted(EVALUATORS.keys())) or "(none)"
        raise ValueError(
            f"No evaluator registered for language '{language}'. "
            f"Available: {available}"
        )
    return EVALUATORS[language]()


def compute_execution_reward(
    code: str,
    language: str = "rust",
    tool_calls: list[dict[str, Any]] | None = None,
    reward_config: dict[str, float] | None = None,
    **kwargs: Any,
) -> float:
    """Single dispatch entry point for execution-based rewards.

    Routes to the appropriate language-specific evaluator.

    Args:
        code: Generated code to evaluate.
        language: Target language (default: "rust" for backward compat).
        tool_calls: Optional tool calls to validate.
        reward_config: Optional custom reward values.
        **kwargs: Additional language-specific arguments.

    Returns:
        Float reward value.
    """
    evaluator = get_evaluator(language)
    return evaluator.compute_execution_reward(
        code, tool_calls=tool_calls, reward_config=reward_config,
    )


def rank_solutions_by_execution(
    solutions: list[str],
    language: str = "rust",
    tests_code: str | None = None,
    **kwargs: Any,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Dispatch solution ranking to language-specific evaluator.

    Args:
        solutions: List of code solutions to rank.
        language: Target language (default: "rust").
        tests_code: Optional test code to append.
        **kwargs: Additional language-specific arguments.

    Returns:
        List of (solution, score, details) tuples, sorted best-first.
    """
    evaluator = get_evaluator(language)
    return evaluator.rank_solutions_by_execution(solutions, tests_code=tests_code)


def _ensure_evaluators_loaded() -> None:
    """Import evaluator modules to trigger registration.

    Called lazily on first dispatch if registry is empty.
    """
    if not EVALUATORS:
        try:
            import pipeline_lib.rust_evaluators  # noqa: F401
        except ImportError:
            pass
        try:
            import pipeline_lib.python_evaluators  # noqa: F401
        except ImportError:
            pass
        try:
            import pipeline_lib.typescript_evaluators  # noqa: F401
        except ImportError:
            pass
        try:
            import pipeline_lib.go_evaluators  # noqa: F401
        except ImportError:
            pass


# Auto-load evaluators on module import
_ensure_evaluators_loaded()
