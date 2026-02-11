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
) -> list[MutationResult]:
    """Run cargo-mutants on a Rust repository.

    Args:
        repo_path: Path to the Rust repository.
        output_dir: Directory for mutation output. Defaults to temp dir.
        timeout_per_mutation: Timeout per mutation test in seconds.
        max_mutations: Maximum number of mutations to generate.
        jobs: Number of parallel mutation tests.

    Returns:
        List of MutationResult objects.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mutants_")

    # First verify the repo builds and tests pass
    check = subprocess.run(
        ["cargo", "test"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if check.returncode != 0:
        print(f"  Warning: cargo test failed for {repo_path}, skipping")
        return []

    # Run cargo-mutants
    cmd = [
        "cargo", "mutants",
        "--timeout", str(timeout_per_mutation),
        "--jobs", str(jobs),
        "--output", output_dir,
        "--json",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_per_mutation * max_mutations,
        )
    except subprocess.TimeoutExpired:
        print(f"  Warning: cargo-mutants timed out for {repo_path}")

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
