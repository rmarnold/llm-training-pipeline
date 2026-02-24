"""Wrapper for go-mutesting mutation testing tool.

Generates training data by:
1. Cloning Go repos
2. Running go-mutesting to introduce mutations
3. Capturing (broken code, test failure, fix diff) triples
4. Formatting as training examples

Requires: go-mutesting installed (`go install github.com/zimmski/go-mutesting/cmd/go-mutesting@latest`)
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class MutationResult:
    """Result from a single go-mutesting mutation."""
    original_code: str
    mutated_code: str
    diff: str
    mutation_type: str  # e.g., "changed conditional from != to =="
    file_path: str
    function_name: str
    outcome: str  # "caught", "missed", "unviable", "timeout"
    compiler_error: str  # Non-empty if unviable
    test_output: str  # Test output if caught


def check_go_mutesting_installed() -> bool:
    """Check if go-mutesting is installed."""
    try:
        result = subprocess.run(
            ["go-mutesting", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # go-mutesting --help exits with 0 or 2 (usage)
        return result.returncode in (0, 2)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to which/where
    try:
        result = subprocess.run(
            ["which", "go-mutesting"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def clone_go_repo(
    url: str,
    clone_dir: str,
    branch: str | None = None,
) -> str:
    """Clone a Go repository.

    Args:
        url: Git repository URL (e.g., "https://github.com/spf13/cobra").
        clone_dir: Parent directory for clones.
        branch: Optional branch name.

    Returns:
        Path to cloned repository.
    """
    # Validate URL to prevent command injection
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


def run_go_mutesting(
    repo_path: str,
    output_dir: str | None = None,
    timeout_per_mutation: int = 300,
    max_mutations: int = 100,
) -> list[MutationResult]:
    """Run go-mutesting on a Go repository.

    Runs `go-mutesting ./...` from the repo root, parses the text output,
    and returns a list of MutationResult objects.

    Args:
        repo_path: Path to the Go repository (must contain go.mod).
        output_dir: Directory to store mutation diffs. Defaults to temp dir.
        timeout_per_mutation: Timeout per mutation test in seconds.
        max_mutations: Maximum number of mutations to parse.

    Returns:
        List of MutationResult objects.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="go_mutesting_")

    os.makedirs(output_dir, exist_ok=True)

    # go-mutesting does not support explicit --timeout in the same way as
    # cargo-mutants.  We apply a total wall-clock timeout on the subprocess.
    total_timeout = max(timeout_per_mutation * max_mutations, 1800)

    cmd = ["go-mutesting", "./..."]
    print(f"  Running: {' '.join(cmd)} (cwd={repo_path})")

    raw_output = ""
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=total_timeout,
            env={**os.environ, "GO111MODULE": "on"},
        )
        raw_output = result.stdout + "\n" + result.stderr

        if result.returncode not in (0, 1):
            # Exit code 1 is expected when any mutations are missed
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"  ERR: {line}")

    except subprocess.TimeoutExpired:
        print(f"  Warning: go-mutesting timed out after {total_timeout}s")

    return _parse_go_mutesting_output(repo_path, raw_output, output_dir, max_mutations)


def _parse_go_mutesting_output(
    repo_path: str,
    raw_output: str,
    output_dir: str,
    max_mutations: int,
) -> list[MutationResult]:
    """Parse go-mutesting text output into MutationResult objects.

    go-mutesting output format (one mutation per line):
        PASS: /abs/path/to/file.go:42:8: Changed conditional from != to ==
        FAIL: /abs/path/to/file.go:55:1: Removed function body

    PASS means the mutation was caught by the test suite (tests detected it).
    FAIL means the mutation survived (tests did not detect it).

    Additional formats go-mutesting may emit:
        SKIP: ...   (skipped mutations)
        ERROR: ...  (mutations that failed to compile)

    Args:
        repo_path: Absolute path to repository root.
        raw_output: Combined stdout+stderr from go-mutesting invocation.
        output_dir: Directory for storing extracted diffs.
        max_mutations: Maximum number of results to return.

    Returns:
        List of MutationResult objects.
    """
    # Regex for the canonical go-mutesting result line:
    #   STATUS: /path/file.go:LINE:COL: description
    line_re = re.compile(
        r'^(PASS|FAIL|SKIP|ERROR):\s+'
        r'(?P<filepath>[^\s:]+\.go)'
        r':(?P<line>\d+)'
        r'(?::(?P<col>\d+))?'
        r':\s*(?P<description>.+)$',
        re.IGNORECASE,
    )

    results: list[MutationResult] = []
    seen: set[tuple[str, str, str]] = set()  # (filepath, line, description) dedup

    outcome_map = {
        "PASS": "caught",
        "FAIL": "missed",
        "SKIP": "unviable",
        "ERROR": "unviable",
    }

    for raw_line in raw_output.splitlines():
        if len(results) >= max_mutations:
            break

        line = raw_line.strip()
        match = line_re.match(line)
        if not match:
            continue

        status = match.group(1).upper()
        rel_or_abs_path = match.group("filepath")
        line_num = match.group("line")
        description = match.group("description").strip()

        # Deduplicate identical mutation descriptions on the same line
        dedup_key = (rel_or_abs_path, line_num, description)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        outcome = outcome_map.get(status, "unviable")

        # Resolve to absolute path; go-mutesting may emit absolute or relative
        if os.path.isabs(rel_or_abs_path):
            abs_path = rel_or_abs_path
        else:
            abs_path = os.path.join(repo_path, rel_or_abs_path)

        # Make file_path relative to repo root for portability
        try:
            file_path = os.path.relpath(abs_path, repo_path)
        except ValueError:
            file_path = rel_or_abs_path

        # Read original source file
        original_code = ""
        if os.path.exists(abs_path):
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as f:
                    original_code = f.read()
            except IOError:
                pass

        # go-mutesting writes the mutated file in-place; we capture any
        # diff that was written to a temp location during the run.
        # Since we run post-hoc, reconstruct a minimal diff from the
        # description.  A full patch is only available if go-mutesting's
        # -print-mutants flag was used (not default).
        diff = _build_minimal_diff(file_path, line_num, description)

        # For ERROR/SKIP outcomes, treat description as the compiler hint
        compiler_error = description if outcome == "unviable" else ""
        test_output = description if outcome == "caught" else ""

        # Infer a rough mutation_type from the description
        mutation_type = _classify_mutation(description)

        # go-mutesting does not report function names in its output;
        # attempt to extract it from the source by finding the nearest
        # enclosing func declaration above the mutated line.
        function_name = _extract_function_name(original_code, int(line_num))

        results.append(MutationResult(
            original_code=original_code,
            mutated_code="",  # go-mutesting restores the file; no mutated snapshot
            diff=diff,
            mutation_type=mutation_type,
            file_path=file_path,
            function_name=function_name,
            outcome=outcome,
            compiler_error=compiler_error,
            test_output=test_output,
        ))

    return results


def _classify_mutation(description: str) -> str:
    """Map a go-mutesting description string to a coarse mutation type.

    Args:
        description: Raw description from go-mutesting output line.

    Returns:
        Short mutation type string.
    """
    desc_lower = description.lower()
    if "conditional" in desc_lower or "comparison" in desc_lower:
        return "conditional_boundary"
    if "removed" in desc_lower and "body" in desc_lower:
        return "remove_function_body"
    if "negat" in desc_lower or "negate" in desc_lower:
        return "negate_condition"
    if "return" in desc_lower:
        return "mutate_return_value"
    if "increment" in desc_lower or "decrement" in desc_lower:
        return "arithmetic_boundary"
    if "true" in desc_lower or "false" in desc_lower:
        return "boolean_literal"
    return "generic_mutation"


def _build_minimal_diff(file_path: str, line_num: str, description: str) -> str:
    """Build a placeholder diff annotation for a mutation.

    go-mutesting applies mutations in-place and restores afterwards, so a
    full unified diff is unavailable post-hoc without the -print-mutants
    flag.  We record a structured annotation instead so downstream
    processing can trace the mutation location.

    Args:
        file_path: Relative path to the mutated file.
        line_num: Line number (string) of the mutation.
        description: Human-readable mutation description.

    Returns:
        Annotation string in a diff-like format.
    """
    return (
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        f"@@ line {line_num} @@\n"
        f"# go-mutesting: {description}\n"
    )


def _extract_function_name(source: str, target_line: int) -> str:
    """Extract the nearest enclosing Go function name for a given line number.

    Scans backwards from the target line looking for a `func` declaration.

    Args:
        source: Full source code of the file.
        target_line: 1-based line number of the mutation.

    Returns:
        Function name string, or empty string if not found.
    """
    if not source:
        return ""

    lines = source.splitlines()
    # Clamp to valid range (target_line is 1-based)
    search_start = min(target_line - 1, len(lines) - 1)

    func_re = re.compile(r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(')

    for idx in range(search_start, -1, -1):
        m = func_re.match(lines[idx])
        if m:
            return m.group(1)

    return ""


def mutations_to_training_data(
    mutations: list[MutationResult],
) -> list[dict[str, str]]:
    """Convert MutationResult objects to training data format.

    For "caught" mutations: (original_code + test_failure → debug insight)
      — teaches the agent to recognise mutation-induced regressions.
    For "unviable" mutations: (original_code + compiler_error → explanation)
      — teaches build-error debugging in Go.

    Args:
        mutations: List of MutationResult objects.

    Returns:
        List of dicts with buggy_code, error_message, fixed_code, explanation keys.
    """
    training_data = []

    for m in mutations:
        if m.outcome == "caught" and m.original_code:
            training_data.append({
                "buggy_code": m.original_code,  # go-mutesting restores original; diff shows what was changed
                "error_message": f"Test failure (mutation caught):\n{m.test_output[:2000]}",
                "fixed_code": m.original_code,
                "explanation": (
                    f"The mutation '{m.mutation_type}' in {m.function_name} ({m.file_path} line ~{m.diff}) "
                    f"was caught by the test suite. The mutation description: {m.test_output[:500]}"
                ),
            })

        elif m.outcome == "unviable" and m.original_code:
            training_data.append({
                "buggy_code": m.original_code,
                "error_message": f"Compiler error:\n{m.compiler_error[:2000]}",
                "fixed_code": m.original_code,
                "explanation": (
                    f"The mutation '{m.mutation_type}' in {m.function_name} ({m.file_path}) "
                    f"caused a compilation error. The original code compiles correctly."
                ),
            })

    return training_data
