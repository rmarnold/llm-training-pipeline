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
    python scripts/16_generate_mutations.py --repos BurntSushi/bstr dtolnay/anyhow --repo-workers 2

Requires:
    - cargo-mutants installed: cargo install cargo-mutants
    - pip install -e ".[gpt_oss]"
"""
from __future__ import annotations

import json
import multiprocessing
import os
import shutil

import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from pipeline_lib.cargo_mutants_runner import (
    check_cargo_mutants_installed,
    clone_rust_repo,
    mutations_to_training_data,
    run_cargo_mutants,
)


def _auto_detect_jobs() -> int:
    """Calculate optimal parallel jobs from CPU count and available RAM.

    Uses 4 GB per worker as the RAM estimate (cargo build peak is ~1-3 GB).
    """
    cpu_count = multiprocessing.cpu_count()
    cpu_jobs = max(1, cpu_count - 2)

    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        total_ram_gb = mem_bytes / (1024**3)
        ram_jobs = max(1, int(total_ram_gb / 4))
    except (ValueError, OSError):
        total_ram_gb = 0
        ram_jobs = cpu_jobs

    jobs = min(cpu_jobs, ram_jobs)

    print(f"  Auto-detected jobs: {jobs} "
          f"(CPUs: {cpu_count}, RAM: {total_ram_gb:.0f} GB, "
          f"cpu_limit: {cpu_jobs}, ram_limit(@4GB/worker): {ram_jobs})")
    return jobs


@dataclass
class RepoEntry:
    """A repo entry from the config, with optional package targeting."""
    url: str
    package: str | None = None


def _repo_name_from_url(url: str) -> str:
    """Extract a safe repo name from a URL for cache filenames."""
    return url.rstrip("/").split("/")[-1].replace(".git", "")


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


def _process_one_repo(
    entry: RepoEntry,
    clone_dir: str,
    output_dir: str,
    max_mutations: int,
    timeout: int,
    jobs_per_repo: int,
    backup_dir: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Process a single repo — clones, runs mutations, returns training data.

    Designed to run in a worker process via ProcessPoolExecutor.

    Args:
        backup_dir: If set, copy the per-repo JSONL here after saving
            (e.g. a mounted Drive path for incremental backup).

    Returns:
        Tuple of (repo_name, training_data_list).
    """
    repo_name = _repo_name_from_url(entry.url)
    label = entry.url
    if entry.package:
        label += f" (package: {entry.package})"
    print(f"  Processing {label}...")

    repo_path = clone_rust_repo(entry.url, clone_dir)

    mutations = run_cargo_mutants(
        repo_path,
        timeout_per_mutation=timeout,
        max_mutations=max_mutations,
        jobs=jobs_per_repo,
        package=entry.package,
    )
    print(f"  {repo_name}: {len(mutations)} mutations")

    training_data = mutations_to_training_data(mutations)
    print(f"  {repo_name}: {len(training_data)} usable training examples")

    # Save per-repo JSONL for incremental caching
    repo_jsonl = os.path.join(output_dir, f"{repo_name}.jsonl")
    with open(repo_jsonl, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    # Backup per-repo JSONL to Drive (incremental — survives preemption)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{repo_name}.jsonl")
        shutil.copy2(repo_jsonl, backup_path)
        print(f"  {repo_name}: backed up to {backup_path}")

    return repo_name, training_data


def _load_cached_repo(output_dir: str, repo_name: str) -> list[dict[str, str]]:
    """Load cached per-repo JSONL results."""
    repo_jsonl = os.path.join(output_dir, f"{repo_name}.jsonl")
    data = []
    with open(repo_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_mutations(
    config_path: str = "configs/data_sources_rust.yaml",
    repos: list[str] | None = None,
    clone_dir: str = "/tmp/rust_repos",
    output_dir: str = "data/rust/mutations",
    max_mutations_per_repo: int = 50,
    timeout_per_mutation: int = 300,
    jobs: int = 0,
    repo_workers: int = 0,
    no_cache: bool = False,
    backup_dir: str | None = None,
) -> str:
    """Run cargo-mutants on Rust repos and save training data.

    Args:
        config_path: Path to data sources config (for repo list).
        repos: Override list of repo URLs. If None, loads from config.
        clone_dir: Directory to clone repos into.
        output_dir: Output directory for mutation data.
        max_mutations_per_repo: Max mutations per repository.
        timeout_per_mutation: Timeout per mutation in seconds.
        jobs: Number of parallel mutation test jobs. 0 = auto-detect from CPU/RAM.
        repo_workers: Number of repos to process in parallel. 0 = auto (jobs // 4, min 1).
        no_cache: Force re-generation even if cached results exist.
        backup_dir: If set, copy per-repo JSONLs and final outputs here
            (e.g. a mounted Google Drive path for incremental backup).

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

    # Auto-detect jobs from CPU/RAM if not specified
    if jobs <= 0:
        jobs = _auto_detect_jobs()
    else:
        print(f"  Using explicit jobs: {jobs}")

    # Auto-detect repo_workers
    if repo_workers <= 0:
        repo_workers = max(1, jobs // 4)
    repo_workers = min(repo_workers, jobs)  # never more workers than jobs

    # Divide mutation-test parallelism across repo workers
    jobs_per_repo = max(1, jobs // repo_workers)

    print(f"\n  Total jobs: {jobs}")
    print(f"  Repo workers: {repo_workers}")
    print(f"  Jobs per repo: {jobs_per_repo}")
    print(f"  Cache: {'disabled (--no-cache)' if no_cache else 'enabled'}")

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

    # Split entries into cached (skip) and pending (process)
    pending_entries: list[RepoEntry] = []
    for entry in entries:
        repo_name = _repo_name_from_url(entry.url)
        repo_jsonl = os.path.join(output_dir, f"{repo_name}.jsonl")

        if not no_cache and os.path.exists(repo_jsonl):
            cached_data = _load_cached_repo(output_dir, repo_name)
            print(f"  Skipping {repo_name} (cached: {len(cached_data)} examples)")
            all_training_data.extend(cached_data)
        else:
            pending_entries.append(entry)

    if not pending_entries:
        print("\nAll repos cached — nothing to process.")
    elif repo_workers == 1 or len(pending_entries) == 1:
        # Sequential processing — simpler output, no multiprocessing overhead
        for i, entry in enumerate(pending_entries):
            label = entry.url
            if entry.package:
                label += f" (package: {entry.package})"
            print(f"\n[{i+1}/{len(pending_entries)}] Processing {label}...")

            try:
                repo_name, training_data = _process_one_repo(
                    entry, clone_dir, output_dir,
                    max_mutations_per_repo, timeout_per_mutation, jobs_per_repo,
                    backup_dir=backup_dir,
                )
                all_training_data.extend(training_data)
            except Exception as e:
                print(f"  Error processing {entry.url}: {e}")
                continue
    else:
        # Parallel repo processing
        print(f"\nProcessing {len(pending_entries)} repos with {repo_workers} workers...")
        with ProcessPoolExecutor(max_workers=repo_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_repo, entry, clone_dir, output_dir,
                    max_mutations_per_repo, timeout_per_mutation, jobs_per_repo,
                    backup_dir,
                ): entry
                for entry in pending_entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    repo_name, training_data = future.result()
                    all_training_data.extend(training_data)
                except Exception as e:
                    print(f"  Error processing {entry.url}: {e}")

    # Merge all per-repo JSONLs into combined mutations.jsonl
    # (Re-read from disk to include both cached and newly generated data)
    all_training_data = []
    for entry in entries:
        repo_name = _repo_name_from_url(entry.url)
        repo_jsonl = os.path.join(output_dir, f"{repo_name}.jsonl")
        if os.path.exists(repo_jsonl):
            all_training_data.extend(_load_cached_repo(output_dir, repo_name))

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

    # Backup final combined outputs to Drive
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(output_path, os.path.join(backup_dir, "mutations.jsonl"))
        hf_src = os.path.join(output_dir, "hf_dataset")
        hf_dst = os.path.join(backup_dir, "hf_dataset")
        if os.path.isdir(hf_src):
            if os.path.exists(hf_dst):
                shutil.rmtree(hf_dst)
            shutil.copytree(hf_src, hf_dst)
        print(f"  Final outputs backed up to {backup_dir}")

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
    parser.add_argument("--clone_dir", type=str, default="/tmp/rust_repos")
    parser.add_argument("--output_dir", type=str, default="data/rust/mutations")
    parser.add_argument("--max_mutations_per_repo", type=int, default=50)
    parser.add_argument("--timeout_per_mutation", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=0,
                        help="Parallel mutation test jobs. 0 = auto-detect from CPU/RAM.")
    parser.add_argument("--repo-workers", type=int, default=0,
                        help="Number of repos to process in parallel. 0 = auto.")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true",
                        help="Force re-generation even if cached results exist.")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Copy per-repo JSONLs and final outputs here (e.g. mounted Drive path).")
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
        repo_workers=args.repo_workers,
        no_cache=args.no_cache,
        backup_dir=args.backup_dir,
    )
