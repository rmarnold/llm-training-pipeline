"""Wrapper for StrykerJS mutation testing tool.

Generates training data by:
1. Cloning TypeScript repos
2. Running StrykerJS to introduce mutations
3. Capturing (broken code, type/test error, fix diff) triples
4. Formatting as training examples

Requires: Node.js and npx available (`npx stryker --version` must succeed)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class MutationResult:
    """Result from a single StrykerJS mutation."""
    original_code: str
    mutated_code: str
    diff: str
    mutation_type: str  # e.g., "ArithmeticOperator", "StringLiteral"
    file_path: str
    function_name: str
    outcome: str  # "caught", "missed", "unviable", "timeout"
    compiler_error: str  # Non-empty if unviable (tsc compile error)
    test_output: str  # Test output if caught by jest


def check_stryker_installed() -> bool:
    """Check if StrykerJS is available via npx."""
    try:
        result = subprocess.run(
            ["npx", "--yes", "stryker", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def clone_ts_repo(
    url: str,
    clone_dir: str,
    branch: str | None = None,
) -> str:
    """Clone a TypeScript repository.

    Args:
        url: Git repository URL (e.g., "https://github.com/colinhacks/zod").
        clone_dir: Parent directory for clones.
        branch: Optional branch name.

    Returns:
        Path to cloned repository.

    Raises:
        ValueError: If url or branch contains unsafe characters.
        RuntimeError: If the git clone command fails.
    """
    if not re.match(r'^https?://[\w.\-/]+$', url):
        raise ValueError(f"Invalid repository URL: {url}")
    if branch and not re.match(r'^[\w.\-/]+$', branch):
        raise ValueError(f"Invalid branch name: {branch}")

    os.makedirs(clone_dir, exist_ok=True)

    # Extract repo name from URL
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)

    if os.path.exists(repo_path):
        print(f"  Repo already cloned: {repo_path}")
        return repo_path

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([url, repo_path])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {url}: {result.stderr}")

    print(f"  Cloned {url} -> {repo_path}")
    return repo_path


def run_stryker(
    repo_path: str,
    output_dir: str | None = None,
    timeout_per_mutation: int = 300,
    max_mutations: int = 100,
    config_file: str | None = None,
) -> list[MutationResult]:
    """Run StrykerJS on a TypeScript repository.

    Installs dependencies if node_modules is absent, then runs
    `npx stryker run --reporters json` and parses the resulting
    mutation-report.json.

    Args:
        repo_path: Path to the TypeScript repository.
        output_dir: Directory for Stryker output. Defaults to a temp dir.
        timeout_per_mutation: Timeout per mutation test in seconds.
        max_mutations: Maximum number of mutations to collect.
        config_file: Optional path to a stryker.config.js/json file.

    Returns:
        List of MutationResult objects.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="stryker_out_")

    # Install npm dependencies if needed so Stryker can compile the project.
    node_modules = os.path.join(repo_path, "node_modules")
    if not os.path.isdir(node_modules):
        print("  Running npm install ...")
        npm_result = subprocess.run(
            ["npm", "install", "--prefer-offline", "--no-fund", "--no-audit"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if npm_result.returncode != 0:
            print(f"  Warning: npm install failed: {npm_result.stderr[-500:]}")

    # Build stryker command.
    cmd = [
        "npx", "--yes", "stryker", "run",
        "--reporters", "json",
        "--jsonReporter.fileName", os.path.join(output_dir, "mutation-report.json"),
        "--timeoutMS", str(timeout_per_mutation * 1000),
    ]

    if config_file and os.path.exists(config_file):
        cmd.extend(["--configFile", config_file])

    # Total wall-clock timeout: per-mutation * max + a fixed overhead.
    total_timeout = max(timeout_per_mutation * max_mutations, 1800)
    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=total_timeout,
        )
        if result.returncode not in (0, 1):
            # returncode 1 is normal when mutants survive; anything else is an error.
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"  ERR: {line}")
    except subprocess.TimeoutExpired:
        print(f"  Warning: stryker timed out after {total_timeout}s — parsing partial output")

    return _parse_stryker_output(repo_path, output_dir, max_mutations)


def _parse_stryker_output(
    repo_path: str,
    output_dir: str,
    max_mutations: int,
) -> list[MutationResult]:
    """Parse StrykerJS mutation-report.json into MutationResult objects.

    StrykerJS JSON reporter format::

        {
          "files": {
            "src/foo.ts": {
              "source": "<original source text>",
              "mutants": [
                {
                  "id": "1",
                  "mutatorName": "ArithmeticOperator",
                  "replacement": "<mutated snippet>",
                  "status": "Killed" | "Survived" | "CompileError" | "Timeout",
                  "statusReason": "<compiler/test output>",
                  "location": {
                    "start": {"line": 10, "column": 5},
                    "end": {"line": 10, "column": 6}
                  }
                }
              ]
            }
          }
        }

    Status mapping:
        Killed       -> caught     (test suite caught the mutant)
        Survived     -> missed     (mutant was not detected)
        CompileError -> unviable   (mutation produces invalid TypeScript)
        Timeout      -> timeout    (mutation caused test timeout)

    Args:
        repo_path: Absolute path to the cloned repository root.
        output_dir: Directory containing mutation-report.json.
        max_mutations: Maximum number of results to return.

    Returns:
        List of MutationResult objects, up to max_mutations entries.
    """
    status_map = {
        "Killed": "caught",
        "Survived": "missed",
        "CompileError": "unviable",
        "Timeout": "timeout",
    }

    report_path = os.path.join(output_dir, "mutation-report.json")
    if not os.path.exists(report_path):
        print(f"  No mutation-report.json found in {output_dir}")
        return []

    try:
        with open(report_path) as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Failed to parse mutation-report.json: {e}")
        return []

    files_data = report.get("files", {})
    results: list[MutationResult] = []

    for relative_path, file_info in files_data.items():
        if len(results) >= max_mutations:
            break

        original_source = file_info.get("source", "")

        # Fall back to reading from disk if source is absent in the report.
        if not original_source:
            full_path = os.path.join(repo_path, relative_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path) as f:
                        original_source = f.read()
                except IOError:
                    pass

        for mutant in file_info.get("mutants", []):
            if len(results) >= max_mutations:
                break

            status = mutant.get("status", "")
            outcome = status_map.get(status)
            if outcome is None:
                # Skip NoCoverage, Ignored, Pending, etc.
                continue

            mutator_name = mutant.get("mutatorName", "")
            replacement = mutant.get("replacement", "")
            status_reason = mutant.get("statusReason", "")
            location = mutant.get("location", {})

            # Build a minimal unified diff so the training example has a diff field.
            diff = _build_diff(
                original_source,
                replacement,
                location,
                relative_path,
            )

            # Derive a pseudo function name from the location.
            start_line = location.get("start", {}).get("line", 0)
            function_name = _infer_function_name(original_source, start_line)

            compiler_error = status_reason if outcome == "unviable" else ""
            test_output = status_reason if outcome == "caught" else ""

            results.append(MutationResult(
                original_code=original_source,
                mutated_code=replacement,
                diff=diff,
                mutation_type=mutator_name,
                file_path=relative_path,
                function_name=function_name,
                outcome=outcome,
                compiler_error=compiler_error,
                test_output=test_output,
            ))

    return results


def _build_diff(
    original_source: str,
    replacement: str,
    location: dict,
    file_path: str,
) -> str:
    """Build a minimal unified-diff string for the mutation.

    StrykerJS reports only the replaced *snippet* (not the full file after
    substitution), so we produce a two-line diff showing the original
    snippet extracted by location and the replacement text.

    Args:
        original_source: Full original source text.
        replacement: Mutated replacement snippet from Stryker.
        location: Dict with 'start'/'end' line/column keys (1-based).
        file_path: Relative path for the diff header.

    Returns:
        Unified-diff string (may be empty if location is missing).
    """
    start = location.get("start", {})
    end = location.get("end", {})
    start_line = start.get("line", 0)
    end_line = end.get("line", start_line)
    start_col = start.get("column", 0)
    end_col = end.get("column", None)

    lines = original_source.splitlines()
    if not lines or start_line < 1 or start_line > len(lines):
        return ""

    # Extract the original snippet spanning the location range (0-indexed).
    if start_line == end_line:
        orig_line = lines[start_line - 1]
        if end_col is not None:
            original_snippet = orig_line[start_col:end_col]
        else:
            original_snippet = orig_line[start_col:]
    else:
        # Multi-line mutation: join full lines in range.
        original_snippet = "\n".join(lines[start_line - 1:end_line])

    header = (
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        f"@@ -{start_line},{end_line - start_line + 1} "
        f"+{start_line},{end_line - start_line + 1} @@\n"
    )
    diff_body = f"-{original_snippet}\n+{replacement}\n"
    return header + diff_body


def _infer_function_name(source: str, line_number: int) -> str:
    """Walk backwards from line_number to find the enclosing function/method name.

    Searches for common TypeScript function declaration patterns:
    - `function foo(`
    - `const foo = (`  (arrow functions)
    - `async foo(`     (method shorthand)

    Args:
        source: Full source text.
        line_number: 1-based line number of the mutation.

    Returns:
        Function name if found, otherwise "<unknown>".
    """
    fn_pattern = re.compile(
        r'(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|'
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(|'
        r'(\w+)\s*(?:=\s*(?:async\s+)?\(|\())'
    )

    lines = source.splitlines()
    search_start = max(0, line_number - 1)
    # Search up to 50 lines backwards for a function declaration.
    for i in range(search_start, max(-1, search_start - 50), -1):
        match = fn_pattern.search(lines[i])
        if match:
            name = match.group(1) or match.group(2) or match.group(3)
            if name:
                return name

    return "<unknown>"


def mutations_to_training_data(
    mutations: list[MutationResult],
) -> list[dict[str, str]]:
    """Convert MutationResult objects to training data format.

    For "caught" mutations: (mutated_code + test_failure -> fix_diff) —
        teaches the agent to recognise test-caught bugs and restore correct code.
    For "unviable" mutations: (mutated_code + compiler_error -> fix_diff) —
        teaches the agent to fix TypeScript type/compile errors.

    Missed and timeout outcomes are excluded: missed mutants indicate the
    test suite cannot distinguish the mutation from correct behaviour, and
    timeout mutants provide no reliable signal.

    Args:
        mutations: List of MutationResult objects.

    Returns:
        List of dicts with keys: buggy_code, error_message, fixed_code,
        explanation.
    """
    training_data: list[dict[str, str]] = []

    for m in mutations:
        if m.outcome == "caught" and m.mutated_code and m.original_code:
            training_data.append({
                "buggy_code": m.mutated_code,
                "error_message": f"Test failure:\n{m.test_output[:2000]}",
                "fixed_code": m.original_code,
                "explanation": (
                    f"The mutation '{m.mutation_type}' in {m.function_name} "
                    f"({m.file_path}) was caught by the test suite. "
                    f"The original code is correct."
                ),
            })

        elif m.outcome == "unviable" and m.mutated_code and m.original_code:
            training_data.append({
                "buggy_code": m.mutated_code,
                "error_message": f"Compiler error:\n{m.compiler_error[:2000]}",
                "fixed_code": m.original_code,
                "explanation": (
                    f"The mutation '{m.mutation_type}' in {m.function_name} "
                    f"({m.file_path}) caused a TypeScript compilation error. "
                    f"The original code compiles correctly."
                ),
            })

    return training_data
