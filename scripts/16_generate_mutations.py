"""Generate training data from cargo-mutants mutation testing.

Clones curated Rust repositories and runs cargo-mutants to produce
(buggy_code, compiler_error/test_failure, fixed_code) triples for training.

These triples teach the model:
- Rust borrow checker rules (unviable mutations → compiler errors)
- Debugging via test failures (caught mutations → test output)
- Common bug patterns in real Rust codebases

Usage:
    python scripts/16_generate_mutations.py
    python scripts/16_generate_mutations.py --config configs/data_sources_rust.yaml
    python scripts/16_generate_mutations.py --repos tokio-rs/tokio serde-rs/serde --max_mutations 200

Requires:
    - cargo-mutants installed: cargo install cargo-mutants
    - pip install -e ".[gpt_oss]"
"""
from __future__ import annotations

import json
import os

import yaml

from dataclasses import dataclass

from pipeline_lib.cargo_mutants_runner import (
    check_cargo_mutants_installed,
    clone_rust_repo,
    mutations_to_training_data,
    run_cargo_mutants,
)


@dataclass
class RepoEntry:
    """A repo entry from the config, with optional package targeting."""
    url: str
    package: str | None = None


def load_repos_from_config(config_path: str) -> list[RepoEntry]:
    """Load curated Rust repo list from config.

    Supports both simple string entries and dict entries with package targeting:
        - "dtolnay/anyhow"                          # simple
        - {repo: "tokio-rs/tokio", package: "tokio"} # workspace package

    Args:
        config_path: Path to data_sources_rust.yaml.

    Returns:
        List of RepoEntry objects.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    entries = []
    categories = config.get("rust_repos", {}).get("categories", {})
    for repo_list in categories.values():
        for item in repo_list:
            if isinstance(item, dict):
                repo = item.get("repo", "")
                package = item.get("package")
            else:
                repo = item
                package = None

            if "/" in repo and not repo.startswith("http"):
                url = f"https://github.com/{repo}"
            else:
                url = repo

            entries.append(RepoEntry(url=url, package=package))

    return entries


def generate_mutations(
    config_path: str = "configs/data_sources_rust.yaml",
    repos: list[str] | None = None,
    clone_dir: str = "data/rust/repos",
    output_dir: str = "data/rust/mutations",
    max_mutations_per_repo: int = 100,
    timeout_per_mutation: int = 300,
    jobs: int = 4,
) -> str:
    """Run cargo-mutants on Rust repos and save training data.

    Args:
        config_path: Path to data sources config (for repo list).
        repos: Override list of repo URLs. If None, loads from config.
        clone_dir: Directory to clone repos into.
        output_dir: Output directory for mutation data.
        max_mutations_per_repo: Max mutations per repository.
        timeout_per_mutation: Timeout per mutation in seconds.
        jobs: Number of parallel mutation tests.

    Returns:
        Path to output JSONL file.
    """
    print(f"\n{'='*60}")
    print("Generating Mutation Training Data")
    print(f"{'='*60}")

    # Check cargo-mutants
    if not check_cargo_mutants_installed():
        print("\nERROR: cargo-mutants is not installed.")
        print("Install with: cargo install cargo-mutants")
        return ""

    # Load repo entries
    if repos is not None:
        # CLI override — simple URLs, no package targeting
        entries = [RepoEntry(url=r) for r in repos]
    elif os.path.exists(config_path):
        entries = load_repos_from_config(config_path)
    else:
        print(f"\nConfig not found: {config_path}")
        print("Use --repos to specify repositories manually.")
        return ""

    print(f"\nLoaded {len(entries)} repos")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mutations.jsonl")

    all_training_data: list[dict[str, str]] = []

    for i, entry in enumerate(entries):
        label = entry.url
        if entry.package:
            label += f" (package: {entry.package})"
        print(f"\n[{i+1}/{len(entries)}] Processing {label}...")

        try:
            # Clone
            repo_path = clone_rust_repo(entry.url, clone_dir)

            # Run mutations
            mutations = run_cargo_mutants(
                repo_path,
                timeout_per_mutation=timeout_per_mutation,
                max_mutations=max_mutations_per_repo,
                jobs=jobs,
                package=entry.package,
            )
            print(f"  Got {len(mutations)} mutations")

            # Convert to training data
            training_data = mutations_to_training_data(mutations)
            print(f"  Usable training examples: {len(training_data)}")

            all_training_data.extend(training_data)

        except Exception as e:
            print(f"  Error processing {entry.url}: {e}")
            continue

    # Save as JSONL
    with open(output_path, "w") as f:
        for item in all_training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n{'='*60}")
    print(f"Mutation generation complete!")
    print(f"  Total training examples: {len(all_training_data):,}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    # Also save as HF dataset for direct use
    if all_training_data:
        _save_as_hf_dataset(all_training_data, output_dir)

    return output_path


def _save_as_hf_dataset(
    training_data: list[dict[str, str]],
    output_dir: str,
) -> None:
    """Save mutation training data as HF dataset with Harmony formatting."""
    from datasets import Dataset
    from dataset_formatters.harmony import format_harmony_debug

    formatted = []
    for item in training_data:
        result = format_harmony_debug(item)
        if result.get("text"):
            formatted.append({"text": result["text"]})

    if formatted:
        dataset = Dataset.from_list(formatted)
        hf_path = os.path.join(output_dir, "hf_dataset")
        dataset.save_to_disk(hf_path)
        print(f"  HF dataset saved to {hf_path} ({len(formatted)} examples)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate mutation training data from Rust repos")
    parser.add_argument("--config", type=str, default="configs/data_sources_rust.yaml")
    parser.add_argument("--repos", nargs="+", type=str, default=None,
                        help="Override repo URLs (e.g., tokio-rs/tokio serde-rs/serde)")
    parser.add_argument("--clone_dir", type=str, default="data/rust/repos")
    parser.add_argument("--output_dir", type=str, default="data/rust/mutations")
    parser.add_argument("--max_mutations_per_repo", type=int, default=100)
    parser.add_argument("--timeout_per_mutation", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    # Normalize repo names to URLs if needed
    repos = None
    if args.repos:
        repos = []
        for r in args.repos:
            if "/" in r and not r.startswith("http"):
                repos.append(f"https://github.com/{r}")
            else:
                repos.append(r)

    generate_mutations(
        config_path=args.config,
        repos=repos,
        clone_dir=args.clone_dir,
        output_dir=args.output_dir,
        max_mutations_per_repo=args.max_mutations_per_repo,
        timeout_per_mutation=args.timeout_per_mutation,
        jobs=args.jobs,
    )
