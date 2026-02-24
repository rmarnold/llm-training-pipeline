"""Python code evaluation utilities.

Provides subprocess wrappers for:
- Python syntax check (compilation)
- pytest (test execution)
- mypy (type checking)
- ruff (linting)
- Execution reward computation for GRPO
- Solution ranking for IPO preference pairs

All commands run in isolated temp directories with timeouts.
Registered as the "python" evaluator via evaluator_dispatch.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from pipeline_lib.evaluator_dispatch import register_evaluator


@dataclass
class ExecutionResult:
    """Result of running a Python evaluation command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float  # seconds
    command: str = ""


@dataclass
class PythonTaskResult:
    """Full evaluation result for a Python coding task."""
    task_id: str
    syntax_valid: bool = False
    tests_passed: bool = False
    mypy_clean: bool = False
    ruff_clean: bool = False
    iterations: int = 0
    diff_lines: int = 0
    tool_calls_valid: bool = True
    error: str | None = None


def _create_temp_project(
    code: str,
    test_code: str | None = None,
    requirements: list[str] | None = None,
) -> str:
    """Create a temporary Python project with the given code.

    Args:
        code: Python source code for main module.
        test_code: Optional test code (pytest).
        requirements: Optional list of pip requirements.

    Returns:
        Path to the temporary project directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="pyeval_")

    # Write main module
    with open(os.path.join(tmpdir, "solution.py"), "w") as f:
        f.write(code)

    # Write test file if provided
    if test_code:
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            f.write(test_code)

    # Write requirements if provided
    if requirements:
        with open(os.path.join(tmpdir, "requirements.txt"), "w") as f:
            f.write("\n".join(requirements) + "\n")

    return tmpdir


def _run_command(
    project_dir: str,
    command: list[str],
    timeout: int = 60,
) -> ExecutionResult:
    """Run a command in the given project directory.

    Args:
        project_dir: Path to the project.
        command: Command to run.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult with stdout, stderr, exit_code, duration.
    """
    cmd_str = " ".join(command)
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        duration = time.time() - start

        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout[:10000],
            stderr=result.stderr[:10000],
            exit_code=result.returncode,
            duration=duration,
            command=cmd_str,
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            exit_code=-1,
            duration=duration,
            command=cmd_str,
        )
    except Exception as e:
        duration = time.time() - start
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(e),
            exit_code=-1,
            duration=duration,
            command=cmd_str,
        )


def run_syntax_check(
    code: str,
    timeout: int = 10,
) -> ExecutionResult:
    """Check Python code for syntax errors using py_compile.

    Args:
        code: Python source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult.
    """
    project_dir = _create_temp_project(code)
    try:
        return _run_command(
            project_dir,
            ["python", "-m", "py_compile", os.path.join(project_dir, "solution.py")],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_pytest(
    code: str,
    test_code: str | None = None,
    timeout: int = 300,
) -> ExecutionResult:
    """Run pytest on Python code.

    If test_code is not provided, looks for inline tests in the code.

    Args:
        code: Python source code.
        test_code: Optional separate test code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult.
    """
    project_dir = _create_temp_project(code, test_code=test_code)
    try:
        if test_code:
            target = os.path.join(project_dir, "test_solution.py")
        else:
            target = project_dir
        return _run_command(project_dir, ["python", "-m", "pytest", target, "-x", "-q"], timeout)
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_mypy(
    code: str,
    timeout: int = 60,
) -> ExecutionResult:
    """Run mypy type checking on Python code.

    Args:
        code: Python source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means no type errors.
    """
    project_dir = _create_temp_project(code)
    try:
        return _run_command(
            project_dir,
            ["python", "-m", "mypy", os.path.join(project_dir, "solution.py"),
             "--ignore-missing-imports"],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_ruff(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Run ruff linter on Python code.

    Args:
        code: Python source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means no lint violations.
    """
    project_dir = _create_temp_project(code)
    try:
        return _run_command(
            project_dir,
            ["python", "-m", "ruff", "check", os.path.join(project_dir, "solution.py")],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def compute_execution_reward(
    code: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reward_config: dict[str, float] | None = None,
    test_code: str | None = None,
) -> float:
    """Compute rule-based reward for GRPO (Python).

    Reward hierarchy:
    - All tests pass + mypy clean + ruff clean: +1.0
    - All tests pass, lint/type warnings: +0.7
    - Syntax valid, some tests fail: +0.1
    - Syntax error: -0.3
    - Invalid tool call format: -1.0

    Args:
        code: Generated Python code.
        tool_calls: Optional list of tool calls to validate format.
        reward_config: Optional custom reward values.
        test_code: Optional test code to run.

    Returns:
        Float reward value.
    """
    if reward_config is None:
        reward_config = {
            "all_tests_pass_lint_clean": 1.0,
            "all_tests_pass_lint_warnings": 0.7,
            "syntax_valid_some_tests_fail": 0.1,
            "syntax_error": -0.3,
            "invalid_tool_call_format": -1.0,
        }

    # Validate tool calls if provided
    if tool_calls is not None:
        from pipeline_lib.rust_evaluators import validate_tool_call
        for tc in tool_calls:
            if not validate_tool_call(tc):
                return reward_config.get("invalid_tool_call_format", -1.0)

    # Syntax check
    syntax_result = run_syntax_check(code, timeout=10)
    if not syntax_result.success:
        return reward_config.get("syntax_error", -0.3)

    # Run pytest
    test_result = run_pytest(code, test_code=test_code, timeout=120)

    # Run ruff + mypy
    ruff_result = run_ruff(code, timeout=30)
    mypy_result = run_mypy(code, timeout=60)

    lint_clean = ruff_result.success and mypy_result.success

    if test_result.success and lint_clean:
        return reward_config.get("all_tests_pass_lint_clean", 1.0)
    elif test_result.success:
        return reward_config.get("all_tests_pass_lint_warnings", 0.7)
    else:
        return reward_config.get("syntax_valid_some_tests_fail", 0.1)


def rank_solutions_by_execution(
    solutions: list[str],
    tests_code: str | None = None,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Rank solutions by execution quality for IPO preference pairs (Python).

    Ranking criteria (in order):
    1. Syntax valid
    2. pytest passes
    3. mypy clean
    4. ruff clean
    5. Smaller code size (tiebreaker)

    Args:
        solutions: List of Python code solutions.
        tests_code: Optional test code to run against each solution.

    Returns:
        List of (solution, score, details) tuples, sorted best-first.
    """
    ranked = []

    for solution in solutions:
        syntax = run_syntax_check(solution, timeout=10)
        test = run_pytest(solution, test_code=tests_code, timeout=60) if syntax.success else ExecutionResult(
            success=False, stdout="", stderr="Skipped (syntax error)", exit_code=-1, duration=0
        )
        mypy = run_mypy(solution, timeout=30) if syntax.success else ExecutionResult(
            success=False, stdout="", stderr="Skipped (syntax error)", exit_code=-1, duration=0
        )
        ruff = run_ruff(solution, timeout=15) if syntax.success else ExecutionResult(
            success=False, stdout="", stderr="Skipped (syntax error)", exit_code=-1, duration=0
        )

        # Score: syntax(4) + test(2) + mypy(0.5) + ruff(0.5) - size_penalty
        score = 0.0
        if syntax.success:
            score += 4.0
        if test.success:
            score += 2.0
        if mypy.success:
            score += 0.5
        if ruff.success:
            score += 0.5
        # Small size bonus (normalize by 1000 lines)
        score -= len(solution.split("\n")) / 1000.0

        details = {
            "syntax_valid": syntax.success,
            "test_passed": test.success,
            "mypy_clean": mypy.success,
            "ruff_clean": ruff.success,
        }

        ranked.append((solution, score, details))

    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


@register_evaluator("python")
class PythonEvaluator:
    """Evaluator for Python code using pytest/mypy/ruff."""

    def compute_execution_reward(
        self,
        code: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reward_config: dict[str, float] | None = None,
    ) -> float:
        return compute_execution_reward(code, tool_calls=tool_calls, reward_config=reward_config)

    def rank_solutions_by_execution(
        self,
        solutions: list[str],
        tests_code: str | None = None,
    ) -> list[tuple[str, float, dict[str, bool]]]:
        return rank_solutions_by_execution(solutions, tests_code=tests_code)
