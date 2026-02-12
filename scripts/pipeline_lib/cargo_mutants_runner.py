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

    # Build cargo-mutants command.
    # Repos with self-referencing dev-deps (memchr, indexmap, etc.) have been
    # removed from the config, so we can use the default copy-to-tempdir mode
    # with --jobs for parallel mutation testing.
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
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"  ERR: {line}")
            if result.stdout:
                for line in result.stdout.strip().split("\n")[-30:]:
                    print(f"  OUT: {line}")
    except subprocess.TimeoutExpired:
        print(f"  Warning: cargo-mutants timed out after {total_timeout}s")

    # cargo-mutants writes to a mutants.out/ subdirectory inside --output
    actual_output = os.path.join(output_dir, "mutants.out")
    if not os.path.isdir(actual_output):
        # Fall back to output_dir itself (older cargo-mutants versions)
        actual_output = output_dir

    # Parse results from JSON output
    return _parse_mutants_output(repo_path, actual_output, max_mutations)


def _parse_mutants_output(
    repo_path: str,
    output_dir: str,
    max_mutations: int,
) -> list[MutationResult]:
    """Parse cargo-mutants v26 JSON output into MutationResult objects.

    cargo-mutants v26 outcomes.json format:
    {
      "outcomes": [
        {
          "scenario": "Baseline" | {"Mutant": {mutant fields}},
          "summary": "CaughtMutant"|"MissedMutant"|"Unviable"|"Timeout"|...,
          "log_path": "...",
          "diff_path": "...",
          "phase_results": [...]
        }
      ]
    }
    """
    results = []

    outcomes_path = os.path.join(output_dir, "outcomes.json")
    if not os.path.exists(outcomes_path):
        print(f"  No outcomes.json in {output_dir}")
        return results

    try:
        with open(outcomes_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Failed to parse outcomes.json: {e}")
        return results

    # Extract outcomes list
    if isinstance(data, dict):
        entries = data.get("outcomes", [])
    else:
        entries = data

    # Map cargo-mutants v26 summary values to our outcome names
    summary_map = {
        "CaughtMutant": "caught",
        "MissedMutant": "missed",
        "Unviable": "unviable",
        "Timeout": "timeout",
    }

    count = 0
    for entry in entries:
        if count >= max_mutations:
            break

        scenario = entry.get("scenario")
        summary = entry.get("summary", "")

        # Skip baseline and non-mutation entries
        if scenario == "Baseline" or not isinstance(scenario, dict):
            continue

        outcome = summary_map.get(summary)
        if outcome is None:
            continue

        # Extract mutant details from scenario.Mutant
        mutant = scenario.get("Mutant", {})
        file_path = mutant.get("file", "")
        replacement = mutant.get("replacement", "")
        genre = mutant.get("genre", "FnValue")
        name = mutant.get("name", "")

        # Extract function name from the function field or the name
        func = mutant.get("function")
        if isinstance(func, dict):
            function_name = func.get("function_name", func.get("name", ""))
        elif isinstance(func, str):
            function_name = func
        else:
            function_name = name

        # Read the original source file
        original_code = ""
        full_path = os.path.join(repo_path, file_path)
        if file_path and os.path.exists(full_path):
            try:
                with open(full_path) as f:
                    original_code = f.read()
            except IOError:
                pass

        # Read diff from diff_path if available
        diff = ""
        diff_path = entry.get("diff_path", "")
        if diff_path:
            full_diff = os.path.join(output_dir, diff_path)
            if os.path.exists(full_diff):
                try:
                    with open(full_diff) as f:
                        diff = f.read()
                except IOError:
                    pass

        # Read log from log_path for compiler errors / test output
        log_content = ""
        log_path = entry.get("log_path", "")
        if log_path:
            full_log = os.path.join(output_dir, log_path)
            if os.path.exists(full_log):
                try:
                    with open(full_log) as f:
                        log_content = f.read()
                except IOError:
                    pass

        compiler_error = log_content if outcome == "unviable" else ""
        test_output = log_content if outcome == "caught" else ""

        results.append(MutationResult(
            original_code=original_code,
            mutated_code=replacement,
            diff=diff,
            mutation_type=genre,
            file_path=file_path,
            function_name=function_name,
            outcome=outcome,
            compiler_error=compiler_error,
            test_output=test_output,
        ))
        count += 1

    return results


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
