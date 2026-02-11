"""Wrapper for cargo-mutants mutation testing tool.

Generates training data by:
1. Cloning Rust repos
2. Running cargo-mutants to introduce mutations
3. Capturing (broken code, compiler error, fix diff) triples
4. Formatting as training examples

Requires: cargo-mutants installed (`cargo install cargo-mutants`)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any


@dataclass
class MutationResult:
    """Result from a single cargo-mutants mutation."""
    original_code: str
    mutated_code: str
    diff: str
    mutation_type: str  # e.g., "replace function body"
    file_path: str
    function_name: str
    outcome: str  # "caught", "missed", "unviable", "timeout"
    compiler_error: str  # Non-empty if unviable
    test_output: str  # Test output if caught


def check_cargo_mutants_installed() -> bool:
    """Check if cargo-mutants is installed."""
    try:
        result = subprocess.run(
            ["cargo", "mutants", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def clone_rust_repo(
    url: str,
    clone_dir: str,
    branch: str | None = None,
) -> str:
    """Clone a Rust repository.

    Args:
        url: Git repository URL (e.g., "https://github.com/tokio-rs/tokio").
        clone_dir: Parent directory for clones.
        branch: Optional branch name.

    Returns:
        Path to cloned repository.
    """
    # Validate URL to prevent command injection
    import re
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


def run_cargo_mutants(
    repo_path: str,
    output_dir: str | None = None,
    timeout_per_mutation: int = 300,
    max_mutations: int = 100,
    jobs: int = 4,
    package: str | None = None,
) -> list[MutationResult]:
    """Run cargo-mutants on a Rust repository.

    Skips the separate pre-check step — cargo-mutants performs its own
    baseline test before mutating and handles failures gracefully.

    Args:
        repo_path: Path to the Rust repository.
        output_dir: Directory for mutation output. Defaults to temp dir.
        timeout_per_mutation: Timeout per mutation test in seconds.
        max_mutations: Maximum number of mutations to generate.
        jobs: Number of parallel mutation tests.
        package: Specific package to mutate (for workspace repos).

    Returns:
        List of MutationResult objects.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mutants_")

    # --- Diagnostic: manual cargo build to capture the actual error ---
    build_cmd = ["cargo", "build"]
    if package:
        build_cmd.extend(["--package", package])
    print(f"  Running: {' '.join(build_cmd)} (in {repo_path})")
    diag = subprocess.run(
        build_cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if diag.returncode != 0:
        print(f"  DIAGNOSTIC: cargo build failed (exit {diag.returncode})")
        if diag.stderr:
            for line in diag.stderr.strip().split("\n")[-20:]:
                print(f"    ERR: {line}")
        if diag.stdout:
            for line in diag.stdout.strip().split("\n")[-10:]:
                print(f"    OUT: {line}")
        return []
    else:
        print(f"  Diagnostic cargo build: OK")

    # Build cargo-mutants command.
    cmd = [
        "cargo", "mutants",
        "--timeout", str(timeout_per_mutation),
        "--jobs", str(jobs),
        "--output", output_dir,
        "--json",
    ]

    # For workspace repos, target a specific package to avoid compiling
    # the entire workspace (which often has heavy/unrelated deps).
    if package:
        cmd.extend(["--package", package])

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
        if result.returncode != 0:
            # Show ALL stderr and stdout
            if result.stderr:
                print(f"  cargo-mutants stderr:")
                for line in result.stderr.strip().split("\n")[-20:]:
                    print(f"    {line}")
            if result.stdout:
                print(f"  cargo-mutants stdout:")
                for line in result.stdout.strip().split("\n")[-20:]:
                    print(f"    {line}")
    except subprocess.TimeoutExpired:
        print(f"  Warning: cargo-mutants timed out after {total_timeout}s")

    # List output directory contents for debugging
    if os.path.isdir(output_dir):
        contents = []
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), output_dir)
                contents.append(rel)
        if contents:
            print(f"  Output dir contents: {contents[:20]}")

    # Check for baseline build log
    log_dir = os.path.join(output_dir, "log")
    if os.path.isdir(log_dir):
        for fname in sorted(os.listdir(log_dir)):
            if "baseline" in fname.lower():
                log_path = os.path.join(log_dir, fname)
                try:
                    with open(log_path) as f:
                        content = f.read()
                    lines = content.strip().split("\n")
                    print(f"  Baseline log ({fname}, last 20 lines):")
                    for line in lines[-20:]:
                        print(f"    {line}")
                except IOError:
                    pass

    # Parse results from JSON output
    return _parse_mutants_output(repo_path, output_dir, max_mutations)


def _parse_mutants_output(
    repo_path: str,
    output_dir: str,
    max_mutations: int,
) -> list[MutationResult]:
    """Parse cargo-mutants JSON output into MutationResult objects."""
    results = []

    # cargo-mutants writes outcomes to output_dir/outcomes.json
    outcomes_path = os.path.join(output_dir, "outcomes.json")
    if not os.path.exists(outcomes_path):
        # Try alternative path
        outcomes_path = os.path.join(output_dir, "mutants.json")

    if not os.path.exists(outcomes_path):
        # Check if there's a log dir with useful info
        log_dir = os.path.join(output_dir, "log")
        if os.path.isdir(log_dir):
            # Look for baseline failure
            for fname in os.listdir(log_dir):
                if "baseline" in fname and fname.endswith(".log"):
                    log_path = os.path.join(log_dir, fname)
                    try:
                        with open(log_path) as f:
                            content = f.read()
                        # Show last few lines of baseline failure
                        lines = content.strip().split("\n")
                        tail = lines[-min(5, len(lines)):]
                        print(f"  Baseline failed:")
                        for line in tail:
                            print(f"    {line}")
                    except IOError:
                        pass
                    break
        else:
            print(f"  No outcomes file found in {output_dir}")
        return results

    try:
        with open(outcomes_path) as f:
            outcomes = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Failed to parse outcomes: {e}")
        return results

    # Handle both list and dict formats
    if isinstance(outcomes, dict):
        mutations = outcomes.get("outcomes", outcomes.get("mutations", []))
    else:
        mutations = outcomes

    for mutation in mutations[:max_mutations]:
        try:
            result = _parse_single_mutation(mutation, repo_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Warning: Failed to parse mutation: {e}")
            continue

    return results


def _parse_single_mutation(
    mutation: dict[str, Any],
    repo_path: str,
) -> MutationResult | None:
    """Parse a single mutation entry from cargo-mutants output."""
    outcome = mutation.get("outcome", mutation.get("status", "unknown"))
    if outcome not in ("caught", "missed", "unviable", "timeout"):
        return None

    file_path = mutation.get("file", mutation.get("path", ""))
    function_name = mutation.get("function", mutation.get("name", ""))
    mutation_type = mutation.get("genre", mutation.get("type", "replace function body"))

    # Read the original file if available
    original_code = ""
    full_path = os.path.join(repo_path, file_path)
    if os.path.exists(full_path):
        try:
            with open(full_path) as f:
                original_code = f.read()
        except IOError:
            pass

    # Get the mutated code and diff
    mutated_code = mutation.get("replacement", "")
    diff = mutation.get("diff", "")

    # Get compiler/test output
    compiler_error = ""
    test_output = ""
    log = mutation.get("log", mutation.get("output", ""))
    if outcome == "unviable":
        compiler_error = log
    elif outcome == "caught":
        test_output = log

    return MutationResult(
        original_code=original_code,
        mutated_code=mutated_code,
        diff=diff,
        mutation_type=mutation_type,
        file_path=file_path,
        function_name=function_name,
        outcome=outcome,
        compiler_error=compiler_error,
        test_output=test_output,
    )


def mutations_to_training_data(
    mutations: list[MutationResult],
) -> list[dict[str, str]]:
    """Convert MutationResult objects to training data format.

    For "caught" mutations: (mutated_code + test_failure → fix_diff) — teaches debugging
    For "unviable" mutations: (mutated_code + compiler_error → fix_diff) — teaches borrow checker

    Args:
        mutations: List of MutationResult objects.

    Returns:
        List of dicts with buggy_code, error_message, fixed_code keys.
    """
    training_data = []

    for m in mutations:
        if m.outcome == "caught" and m.mutated_code and m.original_code:
            training_data.append({
                "buggy_code": m.mutated_code,
                "error_message": f"Test failure:\n{m.test_output[:2000]}",
                "fixed_code": m.original_code,
                "explanation": f"The mutation '{m.mutation_type}' in {m.function_name} ({m.file_path}) "
                              f"was caught by the test suite. The original code is correct.",
            })

        elif m.outcome == "unviable" and m.mutated_code and m.original_code:
            training_data.append({
                "buggy_code": m.mutated_code,
                "error_message": f"Compiler error:\n{m.compiler_error[:2000]}",
                "fixed_code": m.original_code,
                "explanation": f"The mutation '{m.mutation_type}' in {m.function_name} ({m.file_path}) "
                              f"caused a compilation error. The original code compiles correctly.",
            })

    return training_data
