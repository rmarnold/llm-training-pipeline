"""TypeScript code evaluation utilities.

Provides subprocess wrappers for:
- tsc --noEmit (type checking / compilation)
- jest (test execution)
- eslint (linting)
- Execution reward computation for GRPO
- Solution ranking for IPO preference pairs

All commands run in isolated temp directories with timeouts.
Registered as the "typescript" evaluator via evaluator_dispatch.

Requires Node.js with tsc, jest, and eslint available via npx.
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
    """Result of running a TypeScript evaluation command."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float  # seconds
    command: str = ""


@dataclass
class TypeScriptTaskResult:
    """Full evaluation result for a TypeScript coding task."""
    task_id: str
    syntax_valid: bool = False       # tsc --noEmit clean
    tests_passed: bool = False       # jest exit code 0
    eslint_clean: bool = False       # eslint no errors
    type_check_clean: bool = False   # tsc strict mode clean
    iterations: int = 0
    diff_lines: int = 0
    tool_calls_valid: bool = True
    hallucinated_apis: list[str] = field(default_factory=list)
    error: str | None = None


# Minimal package.json enabling TypeScript + Jest + ESLint without a network
# round-trip per evaluation (assumes project-level node_modules are pre-installed
# or the host has a shared npx cache).
_DEFAULT_PACKAGE_JSON = {
    "name": "ts-eval",
    "version": "1.0.0",
    "private": True,
    "scripts": {
        "test": "jest --passWithNoTests",
        "lint": "eslint solution.ts --max-warnings 0",
        "typecheck": "tsc --noEmit --strict solution.ts",
    },
    "jest": {
        "preset": "ts-jest",
        "testEnvironment": "node",
        "testMatch": ["**/test_solution.ts"],
    },
    "eslintConfig": {
        "parser": "@typescript-eslint/parser",
        "plugins": ["@typescript-eslint"],
        "extends": ["eslint:recommended", "plugin:@typescript-eslint/recommended"],
        "env": {"node": True, "es2022": True},
    },
}

_DEFAULT_TSCONFIG = {
    "compilerOptions": {
        "target": "ES2022",
        "module": "commonjs",
        "strict": True,
        "esModuleInterop": True,
        "skipLibCheck": True,
        "outDir": "./dist",
    },
    "include": ["*.ts"],
}


def _create_temp_project(
    code: str,
    test_code: str | None = None,
    package_json: dict[str, Any] | None = None,
) -> str:
    """Create a temporary TypeScript project for isolated evaluation.

    Writes solution.ts, optional test_solution.ts, package.json, and
    tsconfig.json into a fresh temporary directory.

    Args:
        code: TypeScript source code (written to solution.ts).
        test_code: Optional Jest test code (written to test_solution.ts).
        package_json: Optional custom package.json dict. Falls back to the
            module-level _DEFAULT_PACKAGE_JSON.

    Returns:
        Path to the temporary project directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="tseval_")

    with open(os.path.join(tmpdir, "solution.ts"), "w") as f:
        f.write(code)

    if test_code:
        with open(os.path.join(tmpdir, "test_solution.ts"), "w") as f:
            f.write(test_code)

    pkg = package_json if package_json is not None else _DEFAULT_PACKAGE_JSON
    with open(os.path.join(tmpdir, "package.json"), "w") as f:
        json.dump(pkg, f, indent=2)

    with open(os.path.join(tmpdir, "tsconfig.json"), "w") as f:
        json.dump(_DEFAULT_TSCONFIG, f, indent=2)

    return tmpdir


def _run_command(
    project_dir: str,
    command: list[str],
    timeout: int = 60,
) -> ExecutionResult:
    """Run a shell command inside the given project directory.

    Args:
        project_dir: Working directory for the subprocess.
        command: Command and arguments list.
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
            env={**os.environ, "CI": "true", "NO_COLOR": "1"},
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


def run_type_check(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Run tsc --noEmit --strict on TypeScript code.

    Type-checks solution.ts without emitting output files.  Uses strict
    mode so that implicit-any and similar issues are caught.

    Args:
        code: TypeScript source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. success=True means no type errors.
    """
    project_dir = _create_temp_project(code)
    try:
        return _run_command(
            project_dir,
            ["npx", "--yes", "tsc", "--noEmit", "--strict", "solution.ts"],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_jest(
    code: str,
    test_code: str | None = None,
    timeout: int = 300,
) -> ExecutionResult:
    """Run jest on TypeScript code using ts-jest.

    If test_code is not provided, Jest will still run in the project
    directory and pass if no test files are found (--passWithNoTests).

    Args:
        code: TypeScript source code (solution.ts).
        test_code: Optional Jest test file contents (test_solution.ts).
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. success=True means all tests passed.
    """
    project_dir = _create_temp_project(code, test_code=test_code)
    try:
        return _run_command(
            project_dir,
            ["npx", "--yes", "jest", "--passWithNoTests", "--no-coverage"],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def run_eslint(
    code: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Run eslint with TypeScript plugin on TypeScript code.

    Uses the eslintConfig embedded in the generated package.json so no
    external .eslintrc file is required.

    Args:
        code: TypeScript source code.
        timeout: Timeout in seconds.

    Returns:
        ExecutionResult. success=True means no lint errors or warnings.
    """
    project_dir = _create_temp_project(code)
    try:
        return _run_command(
            project_dir,
            ["npx", "--yes", "eslint", "solution.ts", "--max-warnings", "0"],
            timeout,
        )
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def _validate_tool_call(tool_call: dict[str, Any]) -> bool:
    """Validate that a tool call has correct JSON format.

    Accepts both flat format {"name": ..., "arguments": ...} and nested
    OpenAI format {"function": {"name": ..., "arguments": ...}}.

    Args:
        tool_call: Dict representing a single tool call.

    Returns:
        True if the structure and argument JSON are valid.
    """
    if not isinstance(tool_call, dict):
        return False

    if "name" not in tool_call:
        func = tool_call.get("function", {})
        if "name" not in func:
            return False

    args = tool_call.get(
        "arguments",
        tool_call.get("function", {}).get("arguments", "{}"),
    )
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
    """Compute rule-based reward for GRPO (TypeScript).

    Reward hierarchy (evaluated in order, returning on first match):
    - Invalid tool call format         : -1.0
    - tsc type error                   : -0.3
    - tsc clean, some tests fail       : +0.1
    - All tests pass, lint/type issues : +0.7
    - All tests pass + eslint + tsc    : +1.0

    The full reward at +1.0 requires jest exit-0, eslint exit-0, and
    tsc --noEmit exit-0, matching the GRPO config definition.

    Args:
        code: Generated TypeScript code.
        tool_calls: Optional list of tool calls to validate format.
        reward_config: Optional dict overriding default reward values.
        test_code: Optional Jest test code to run alongside the solution.

    Returns:
        Float reward value in [-1.0, 1.0].
    """
    if reward_config is None:
        reward_config = {
            "all_tests_pass_lint_clean": 1.0,
            "all_tests_pass_lint_warnings": 0.7,
            "type_check_valid_some_tests_fail": 0.1,
            "type_check_error": -0.3,
            "invalid_tool_call_format": -1.0,
        }

    # Validate tool calls first — format errors are the lowest reward.
    if tool_calls is not None:
        for tc in tool_calls:
            if not _validate_tool_call(tc):
                return reward_config.get("invalid_tool_call_format", -1.0)

    # Type check is a prerequisite: broken TypeScript should not be rewarded.
    type_check_result = run_type_check(code, timeout=30)
    if not type_check_result.success:
        return reward_config.get("type_check_error", -0.3)

    # Run jest — more expensive, only reached if type check passes.
    test_result = run_jest(code, test_code=test_code, timeout=300)

    if not test_result.success:
        return reward_config.get("type_check_valid_some_tests_fail", 0.1)

    # All tests passed: check lint cleanliness for the top tier.
    eslint_result = run_eslint(code, timeout=30)

    if test_result.success and eslint_result.success:
        return reward_config.get("all_tests_pass_lint_clean", 1.0)

    return reward_config.get("all_tests_pass_lint_warnings", 0.7)


def rank_solutions_by_execution(
    solutions: list[str],
    tests_code: str | None = None,
) -> list[tuple[str, float, dict[str, bool]]]:
    """Rank TypeScript solutions by execution quality for IPO preference pairs.

    Scoring rubric (mirrors rust_evaluators.rank_solutions_by_execution):
    - type_check passes : +4.0
    - jest passes       : +2.0
    - eslint clean      : +0.5
    - tsc clean         : +0.5   (separate strict run, same as type_check here)
    - size penalty      : -lines / 1000

    The tsc component uses the same run_type_check result as the type_check
    component, so both +4.0 and +0.5 are awarded together when tsc passes.

    Args:
        solutions: List of TypeScript code solutions to rank.
        tests_code: Optional Jest test code to run against each solution.

    Returns:
        List of (solution, score, details) tuples, sorted best-first.
    """
    _skipped_type = ExecutionResult(
        success=False, stdout="", stderr="Skipped (type check failed)",
        exit_code=-1, duration=0,
    )

    ranked: list[tuple[str, float, dict[str, bool]]] = []

    for solution in solutions:
        type_check = run_type_check(solution, timeout=30)

        if type_check.success:
            test = run_jest(solution, test_code=tests_code, timeout=60)
            eslint = run_eslint(solution, timeout=30)
        else:
            test = _skipped_type
            eslint = _skipped_type

        # Score: type_check(4) + test(2) + eslint(0.5) + tsc(0.5) - size_penalty
        # tsc and type_check share the same result; award both components together.
        score = 0.0
        if type_check.success:
            score += 4.0   # type_check component
            score += 0.5   # tsc component (same gate)
        if test.success:
            score += 2.0
        if eslint.success:
            score += 0.5
        score -= len(solution.split("\n")) / 1000.0

        details: dict[str, bool] = {
            "type_check_clean": type_check.success,
            "tests_passed": test.success,
            "eslint_clean": eslint.success,
        }

        ranked.append((solution, score, details))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


@register_evaluator("typescript")
class TypeScriptEvaluator:
    """Evaluator for TypeScript code using tsc / jest / eslint.

    Registered under the key "typescript" so that GRPO and IPO scripts can
    dispatch to it via evaluator_dispatch.get_evaluator("typescript").
    """

    def compute_execution_reward(
        self,
        code: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reward_config: dict[str, float] | None = None,
        test_code: str | None = None,
    ) -> float:
        return compute_execution_reward(
            code,
            tool_calls=tool_calls,
            reward_config=reward_config,
            test_code=test_code,
        )

    def rank_solutions_by_execution(
        self,
        solutions: list[str],
        tests_code: str | None = None,
    ) -> list[tuple[str, float, dict[str, bool]]]:
        return rank_solutions_by_execution(solutions, tests_code=tests_code)
