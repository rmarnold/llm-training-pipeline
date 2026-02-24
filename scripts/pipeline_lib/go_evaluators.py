"""Go code evaluation utilities.

Provides subprocess wrappers for:
- go build (compilation)
- go test (test execution)
- go vet (static analysis)
- golangci-lint (linting)
- Execution reward computation for GRPO
- Solution ranking for IPO preference pairs

All go commands run in isolated temp directories with timeouts.
Registered as the "go" evaluator via evaluator_dispatch.
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

from pipeline_lib.evaluator_dispatch import register_evaluator


@dataclass
class ExecutionResult:
    """Result of running a go command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float  # seconds
    command: str = ""


@dataclass
class GoTaskResult:
    """Full evaluation result for a Go coding task."""
    task_id: str
    build_passed: bool = False
    tests_passed: bool = False
    lint_clean: bool = False
    vet_clean: bool = False
    iterations: int = 0
    diff_lines: int = 0
    tool_calls_valid: bool = True
    hallucinated_apis: list[str] = field(default_factory=list)
    error: str | None = None


def _create_temp_module(
    code: str,
    test_code: str | None = None,
    module_name: str = "evalmod",
) -> str:
    """Create a temporary Go module with the given code.

    Writes:
      <tmpdir>/go.mod          — minimal module file
      <tmpdir>/solution.go     — solution source
      <tmpdir>/solution_test.go — test file (if test_code is provided)

    Args:
        code: Go source code for the solution file.
        test_code: Optional Go test code (uses the same package declaration).
        module_name: Go module name (used in go.mod).

    Returns:
        Path to the temporary module directory.
    """
    tmpdir = tempfile.mkdtemp(prefix=f"{module_name}_")

    # Determine Go version to write in go.mod; default to a wide-compat version
    go_version = _detect_go_version()

    go_mod = f"module {module_name}\n\ngo {go_version}\n"
    with open(os.path.join(tmpdir, "go.mod"), "w", encoding="utf-8") as f:
        f.write(go_mod)

    # Ensure the solution has a valid package declaration
    solution_src = _ensure_package_declaration(code, "main" if "func main()" in code else "evalmod")
    with open(os.path.join(tmpdir, "solution.go"), "w", encoding="utf-8") as f:
        f.write(solution_src)

    if test_code:
        # Test file must declare the same package or <pkg>_test
        test_src = _ensure_package_declaration(test_code, "evalmod_test")
        with open(os.path.join(tmpdir, "solution_test.go"), "w", encoding="utf-8") as f:
            f.write(test_src)

    return tmpdir


def _detect_go_version() -> str:
    """Detect the installed Go version for go.mod compatibility.

    Returns:
        Version string like "1.21", defaulting to "1.21" if detection fails.
    """
    try:
        result = subprocess.run(
            ["go", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Output: "go version go1.21.5 linux/amd64"
        match = __import__("re").search(r'go(\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "1.21"


def _ensure_package_declaration(code: str, default_package: str) -> str:
    """Ensure Go source has a package declaration.

    Args:
        code: Go source code, may or may not start with a package clause.
        default_package: Package name to prepend if none is present.

    Returns:
        Source code with a valid package declaration.
    """
    stripped = code.lstrip()
    if stripped.startswith("package "):
        return code
    return f"package {default_package}\n\n{code}"


def _run_command(
    project_dir: str,
    command: list[str],
    timeout: int = 60,
) -> ExecutionResult:
    """Run a command in the given project directory.

    Args:
        project_dir: Path to the Go module root.
        command: Command to run (e.g., ["go", "build", "./..."]).
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
            env={
                **os.environ,
                "GO111MODULE": "on",
                # Disable CGO for hermetic builds in the evaluation sandbox
                "CGO_ENABLED": "0",
                # Keep colour codes out of output
                "NO_COLOR": "1",
            },
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


def run_go_build(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Run `go build ./...` on Go code.

    Args:
        code: Go source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means the package compiles without errors.
    """
    module_dir = _create_temp_module(code)
    try:
        return _run_command(module_dir, ["go", "build", "./..."], timeout)
    finally:
        shutil.rmtree(module_dir, ignore_errors=True)


def run_go_test(
    code: str,
    test_code: str | None = None,
    timeout: int = 300,
) -> ExecutionResult:
    """Run `go test ./... -v` on Go code.

    Args:
        code: Go source code (may include inline _test functions).
        test_code: Optional separate test file content.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means all tests pass.
    """
    module_dir = _create_temp_module(code, test_code=test_code)
    try:
        return _run_command(module_dir, ["go", "test", "./...", "-v"], timeout)
    finally:
        shutil.rmtree(module_dir, ignore_errors=True)


def run_go_vet(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Run `go vet ./...` on Go code.

    Args:
        code: Go source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means no vet warnings.
    """
    module_dir = _create_temp_module(code)
    try:
        return _run_command(module_dir, ["go", "vet", "./..."], timeout)
    finally:
        shutil.rmtree(module_dir, ignore_errors=True)


def run_golangci_lint(
    code: str,
    timeout: int = 60,
) -> ExecutionResult:
    """Run `golangci-lint run` on Go code.

    Falls back gracefully if golangci-lint is not installed by returning a
    neutral result (success=True, stderr notes the tool is absent) so that
    its absence does not block the reward pipeline on hosts without the
    linter installed.

    Args:
        code: Go source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. Success means no lint violations.
    """
    # Check if golangci-lint is available before creating a temp dir
    if not _is_golangci_lint_available():
        return ExecutionResult(
            success=True,  # Neutral: don't penalise for missing tool
            stdout="",
            stderr="golangci-lint not installed; lint check skipped",
            exit_code=0,
            duration=0.0,
            command="golangci-lint run",
        )

    module_dir = _create_temp_module(code)
    try:
        return _run_command(module_dir, ["golangci-lint", "run"], timeout)
    finally:
        shutil.rmtree(module_dir, ignore_errors=True)


def _is_golangci_lint_available() -> bool:
    """Check whether golangci-lint is on PATH."""
    try:
        result = subprocess.run(
            ["golangci-lint", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def validate_tool_call(tool_call: dict[str, Any]) -> bool:
    """Validate that a tool call has correct JSON format.

    Mirrors the validation logic from rust_evaluators to keep behaviour
    consistent across language evaluators.

    Args:
        tool_call: Dict with "name" and "arguments" keys.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(tool_call, dict):
        return False

    if "name" not in tool_call:
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


def compute_execution_reward(
    code: str,
    tool_calls: list[dict[str, Any]] | None = None,
    reward_config: dict[str, float] | None = None,
    test_code: str | None = None,
) -> float:
    """Compute rule-based reward for GRPO (Go).

    Reward hierarchy:
    - All tests pass + golangci-lint clean + go vet clean: +1.0
    - All tests pass, some lint/vet issues:                +0.7
    - Build succeeds, some tests fail:                     +0.1
    - Build failure (compile error):                      -0.3
    - Invalid tool call format:                           -1.0

    Args:
        code: Generated Go code.
        tool_calls: Optional list of tool calls to validate format.
        reward_config: Optional custom reward values.
        test_code: Optional separate test file content.

    Returns:
        Float reward value.
    """
    if reward_config is None:
        reward_config = {
            "all_tests_pass_lint_clean": 1.0,
            "all_tests_pass_lint_warnings": 0.7,
            "build_success_some_tests_fail": 0.1,
            "build_failure": -0.3,
            "invalid_tool_call_format": -1.0,
        }

    # Validate tool calls if provided
    if tool_calls is not None:
        for tc in tool_calls:
            if not validate_tool_call(tc):
                return reward_config.get("invalid_tool_call_format", -1.0)

    # Run go build
    build_result = run_go_build(code, timeout=30)
    if not build_result.success:
        return reward_config.get("build_failure", -0.3)

    # Run go test
    test_result = run_go_test(code, test_code=test_code, timeout=300)

    # Run go vet
    vet_result = run_go_vet(code, timeout=30)

    # Run golangci-lint
    lint_result = run_golangci_lint(code, timeout=60)

    lint_clean = vet_result.success and lint_result.success

    if test_result.success and lint_clean:
        return reward_config.get("all_tests_pass_lint_clean", 1.0)
    elif test_result.success:
        return reward_config.get("all_tests_pass_lint_warnings", 0.7)
    else:
        return reward_config.get("build_success_some_tests_fail", 0.1)


def rank_solutions_by_execution(
    solutions: list[str],
    tests_code: str | None = None,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Rank solutions by execution quality for IPO preference pairs.

    Scoring (in priority order):
      go build    passes → +4.0
      go test     passes → +2.0
      go vet      passes → +0.5
      golangci-lint clean → +0.5
      size penalty        → -(line_count / 1000)

    Args:
        solutions: List of Go code solutions.
        tests_code: Optional test code to include alongside each solution.

    Returns:
        List of (solution, score, details) tuples, sorted best-first.
    """
    ranked = []

    for solution in solutions:
        # Build check
        build = run_go_build(solution, timeout=30)

        # Only run tests/vet/lint if the build passes
        _skipped = ExecutionResult(
            success=False, stdout="", stderr="Skipped (build failed)", exit_code=-1, duration=0
        )

        if build.success:
            test = run_go_test(solution, test_code=tests_code, timeout=60)
            vet = run_go_vet(solution, timeout=30)
            lint = run_golangci_lint(solution, timeout=60)
        else:
            test = vet = lint = _skipped

        # Score: build(4) + test(2) + vet(0.5) + lint(0.5) - size_penalty
        score = 0.0
        if build.success:
            score += 4.0
        if test.success:
            score += 2.0
        if vet.success:
            score += 0.5
        if lint.success:
            score += 0.5
        # Small size penalty (normalize by 1000 lines)
        score -= len(solution.split("\n")) / 1000.0

        details = {
            "build_passed": build.success,
            "tests_passed": test.success,
            "vet_clean": vet.success,
            "lint_clean": lint.success,
        }

        ranked.append((solution, score, details))

    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


@register_evaluator("go")
class GoEvaluator:
    """Evaluator for Go code using go build/test/vet + golangci-lint."""

    def compute_execution_reward(
        self,
        code: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reward_config: dict[str, float] | None = None,
    ) -> float:
        return compute_execution_reward(
            code, tool_calls=tool_calls, reward_config=reward_config,
        )

    def rank_solutions_by_execution(
        self,
        solutions: list[str],
        tests_code: str | None = None,
    ) -> list[tuple[str, float, dict[str, bool]]]:
        return rank_solutions_by_execution(solutions, tests_code=tests_code)
