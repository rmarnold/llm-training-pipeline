"""Rust code evaluation utilities.

Provides subprocess wrappers for:
- cargo check (compilation)
- cargo test (test execution)
- cargo clippy (linting)
- Execution reward computation for GRPO
- Solution ranking for IPO preference pairs

All cargo commands run in isolated temp directories with timeouts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of running a cargo command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float  # seconds
    command: str = ""


@dataclass
class RustTaskResult:
    """Full evaluation result for a Rust coding task."""
    task_id: str
    check_passed: bool = False
    test_passed: bool = False
    clippy_clean: bool = False
    iterations: int = 0
    diff_lines: int = 0
    tool_calls_valid: bool = True
    hallucinated_apis: list[str] = field(default_factory=list)
    error: str | None = None


def _create_temp_crate(
    code: str,
    crate_name: str = "eval_crate",
    dependencies: dict[str, str] | None = None,
) -> str:
    """Create a temporary Cargo project with the given code.

    Args:
        code: Rust source code for src/main.rs or src/lib.rs.
        crate_name: Name for the Cargo project.
        dependencies: Optional crate dependencies {"name": "version"}.

    Returns:
        Path to the temporary crate directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"{crate_name}_")

    # Create Cargo.toml
    cargo_toml = f'[package]\nname = "{crate_name}"\nversion = "0.1.0"\nedition = "2021"\n'
    if dependencies:
        cargo_toml += "\n[dependencies]\n"
        for dep_name, dep_version in dependencies.items():
            cargo_toml += f'{dep_name} = "{dep_version}"\n'

    with open(os.path.join(tmpdir, "Cargo.toml"), "w") as f:
        f.write(cargo_toml)

    # Create src directory and write code
    src_dir = os.path.join(tmpdir, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Detect if this is a library (has tests but no main) or binary
    if "fn main()" in code:
        filepath = os.path.join(src_dir, "main.rs")
    else:
        filepath = os.path.join(src_dir, "lib.rs")

    with open(filepath, "w") as f:
        f.write(code)

    return tmpdir


def _run_cargo_command(
    crate_dir: str,
    command: list[str],
    timeout: int = 60,
) -> ExecutionResult:
    """Run a cargo command in the given crate directory.

    Args:
        crate_dir: Path to the Cargo project.
        command: Command to run (e.g., ["cargo", "check"]).
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult with stdout, stderr, exit_code, duration.
    """
    cmd_str = " ".join(command)
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=crate_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CARGO_TERM_COLOR": "never"},
        )
        duration = time.time() - start

        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout[:10000],  # Limit output size
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


def run_cargo_check(
    code: str,
    timeout: int = 60,
    dependencies: dict[str, str] | None = None,
) -> ExecutionResult:
    """Run cargo check on Rust code.

    Args:
        code: Rust source code.
        timeout: Timeout in seconds.
        dependencies: Optional crate dependencies.

    Returns:
        ExecutionResult.
    """
    crate_dir = _create_temp_crate(code, "check_crate", dependencies)
    try:
        return _run_cargo_command(crate_dir, ["cargo", "check"], timeout)
    finally:
        shutil.rmtree(crate_dir, ignore_errors=True)


def run_cargo_test(
    code: str,
    timeout: int = 300,
    dependencies: dict[str, str] | None = None,
    use_nextest: bool = False,
) -> ExecutionResult:
    """Run cargo test on Rust code.

    Args:
        code: Rust source code (should include #[test] functions).
        timeout: Timeout in seconds.
        dependencies: Optional crate dependencies.
        use_nextest: Use cargo-nextest instead of cargo test.

    Returns:
        ExecutionResult.
    """
    crate_dir = _create_temp_crate(code, "test_crate", dependencies)
    try:
        if use_nextest:
            cmd = ["cargo", "nextest", "run"]
        else:
            cmd = ["cargo", "test"]
        return _run_cargo_command(crate_dir, cmd, timeout)
    finally:
        shutil.rmtree(crate_dir, ignore_errors=True)


def run_cargo_clippy(
    code: str,
    timeout: int = 60,
    dependencies: dict[str, str] | None = None,
) -> ExecutionResult:
    """Run cargo clippy on Rust code.

    Args:
        code: Rust source code.
        timeout: Timeout in seconds.
        dependencies: Optional crate dependencies.

    Returns:
        ExecutionResult. Success means no warnings.
    """
    crate_dir = _create_temp_crate(code, "clippy_crate", dependencies)
    try:
        return _run_cargo_command(
            crate_dir,
            ["cargo", "clippy", "--", "-D", "warnings"],
            timeout,
        )
    finally:
        shutil.rmtree(crate_dir, ignore_errors=True)


def compute_execution_reward(
    code: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reward_config: dict[str, float] | None = None,
) -> float:
    """Compute rule-based reward for GRPO.

    Reward hierarchy:
    - All tests pass + clippy clean: +1.0
    - All tests pass, clippy warnings: +0.7
    - Compilation succeeds, some tests fail: +0.1
    - Compilation failure: -0.3
    - Invalid tool call format: -1.0

    Args:
        code: Generated Rust code.
        tool_calls: Optional list of tool calls to validate format.
        reward_config: Optional custom reward values.

    Returns:
        Float reward value.
    """
    if reward_config is None:
        reward_config = {
            "all_tests_pass_clippy_clean": 1.0,
            "all_tests_pass_clippy_warnings": 0.7,
            "compilation_success_some_tests_fail": 0.1,
            "compilation_failure": -0.3,
            "invalid_tool_call_format": -1.0,
        }

    # Validate tool calls if provided
    if tool_calls is not None:
        for tc in tool_calls:
            if not validate_tool_call(tc):
                return reward_config.get("invalid_tool_call_format", -1.0)

    # Run cargo check
    check_result = run_cargo_check(code, timeout=60)
    if not check_result.success:
        return reward_config.get("compilation_failure", -0.3)

    # Run cargo test
    test_result = run_cargo_test(code, timeout=120)

    # Run cargo clippy
    clippy_result = run_cargo_clippy(code, timeout=60)

    if test_result.success and clippy_result.success:
        return reward_config.get("all_tests_pass_clippy_clean", 1.0)
    elif test_result.success:
        return reward_config.get("all_tests_pass_clippy_warnings", 0.7)
    else:
        return reward_config.get("compilation_success_some_tests_fail", 0.1)


def validate_tool_call(tool_call: dict[str, Any]) -> bool:
    """Validate that a tool call has correct JSON format.

    Args:
        tool_call: Dict with "name" and "arguments" keys.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(tool_call, dict):
        return False

    if "name" not in tool_call:
        # Check nested function format
        func = tool_call.get("function", {})
        if "name" not in func:
            return False

    # Validate arguments are parseable JSON
    args = tool_call.get("arguments", tool_call.get("function", {}).get("arguments", "{}"))
    if isinstance(args, str):
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return False

    return True


def rank_solutions_by_execution(
    solutions: list[str],
    tests_code: str | None = None,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Rank solutions by execution quality for IPO preference pairs.

    Ranking criteria (in order):
    1. cargo check passes
    2. cargo test passes
    3. cargo clippy clean
    4. Smaller code size (tiebreaker)

    Args:
        solutions: List of Rust code solutions.
        tests_code: Optional test code to append to each solution.

    Returns:
        List of (solution, score, details) tuples, sorted best-first.
    """
    ranked = []

    for solution in solutions:
        code = solution
        if tests_code:
            code = f"{solution}\n\n{tests_code}"

        check = run_cargo_check(code, timeout=30)
        test = run_cargo_test(code, timeout=60) if check.success else ExecutionResult(
            success=False, stdout="", stderr="Skipped (check failed)", exit_code=-1, duration=0
        )
        clippy = run_cargo_clippy(code, timeout=30) if check.success else ExecutionResult(
            success=False, stdout="", stderr="Skipped (check failed)", exit_code=-1, duration=0
        )

        # Score: check(4) + test(2) + clippy(1) - size_penalty
        score = 0.0
        if check.success:
            score += 4.0
        if test.success:
            score += 2.0
        if clippy.success:
            score += 1.0
        # Small size bonus (normalize by 1000 lines)
        score -= len(solution.split("\n")) / 1000.0

        details = {
            "check_passed": check.success,
            "test_passed": test.success,
            "clippy_clean": clippy.success,
        }

        ranked.append((solution, score, details))

    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
