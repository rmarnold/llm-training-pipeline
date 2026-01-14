#!/usr/bin/env python3
"""Download and prepare datasets for the training pipeline.

Handles both regular and streaming datasets, with support for:
- Pretraining data (large-scale, often streaming)
- Reasoning datasets
- Function calling datasets
- Logic datasets
- Instruction tuning datasets
- Preference data for DPO

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --phases pretraining reasoning
    python scripts/01_download_data.py --max-samples 100000
"""

import os
import argparse
import time
import yaml
from datasets import load_dataset, DownloadConfig


def download_streaming_dataset(ds_config, output_path, max_samples=None):
    """Download a streaming dataset by iterating through it.

    Args:
        ds_config: Dataset configuration
        output_path: Path to save the dataset
        max_samples: Maximum samples to download (None = all available based on weight)

    Returns:
        Number of samples downloaded
    """
    from datasets import Dataset

    source = ds_config["source"]
    subset = ds_config.get("subset") or ds_config.get("version")
    weight = ds_config.get("weight", 1.0)

    print(f"  Loading {source} in streaming mode...")

    try:
        ds_stream = load_dataset(
            source,
            subset,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  Error loading stream: {e}")
        return 0

    # Calculate target samples
    # For streaming, weight determines what fraction to take
    # If max_samples specified, use that as ceiling
    target_samples = max_samples if max_samples else int(1_000_000 * weight)

    print(f"  Streaming up to {target_samples:,} samples...")

    samples = []
    try:
        for i, example in enumerate(ds_stream):
            if i >= target_samples:
                break
            samples.append(example)
            if (i + 1) % 10000 == 0:
                print(f"    Downloaded {i + 1:,} samples...")
    except Exception as e:
        print(f"  Streaming stopped at {len(samples)} samples: {e}")

    if not samples:
        print(f"  Warning: No samples downloaded")
        return 0

    # Convert to dataset and save
    ds = Dataset.from_list(samples)
    ds.save_to_disk(output_path.replace(".parquet", ""))
    print(f"  Saved {len(ds):,} samples to {output_path}")

    return len(ds)


def download_regular_dataset(ds_config, output_path, max_samples=None):
    """Download a regular (non-streaming) dataset.

    Args:
        ds_config: Dataset configuration
        output_path: Path to save the dataset
        max_samples: Maximum samples to download

    Returns:
        Number of samples downloaded
    """
    source = ds_config["source"]
    subset = ds_config.get("subset") or ds_config.get("version")
    weight = ds_config.get("weight", 1.0)

    download_config = DownloadConfig(max_retries=5)
    max_retries = 3

    def try_load():
        for attempt in range(max_retries):
            try:
                return load_dataset(
                    source,
                    subset,
                    split="train",
                    cache_dir="data/cache",
                    download_config=download_config,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"  Failed to download after {max_retries} attempts: {e}")
        return None

    ds = try_load()
    if ds is None:
        return 0

    # Apply weight/sampling
    target_samples = int(len(ds) * weight)
    if max_samples:
        target_samples = min(target_samples, max_samples)

    if target_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(target_samples))

    # Save as parquet
    ds.to_parquet(output_path)
    print(f"  Saved {len(ds):,} samples to {output_path}")

    return len(ds)


def download_datasets(
    config_path="configs/data_sources.yaml",
    phases=None,
    max_samples=None,
):
    """Download all configured datasets.

    Args:
        config_path: Path to data sources config
        phases: List of phases to download (None = all)
        max_samples: Max samples per dataset
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs("data/raw", exist_ok=True)
    manifest = {}

    # All possible phases
    all_phases = [
        "pretraining",
        "reasoning",
        "function_calling",
        "logic",
        "instruction_tuning",
        "preference_data",
    ]

    if phases:
        phases_to_download = [p for p in phases if p in all_phases]
    else:
        phases_to_download = all_phases

    for phase in phases_to_download:
        if phase not in config["datasets"]:
            print(f"\nSkipping {phase} (not in config)")
            continue

        print(f"\n{'=' * 60}")
        print(f"DOWNLOADING: {phase.upper()}")
        print(f"{'=' * 60}")

        for ds_config in config["datasets"][phase]:
            name = ds_config["name"]
            output_path = f"data/raw/{phase}_{name}.parquet"

            # Check if already exists
            if os.path.exists(output_path) or os.path.exists(output_path.replace(".parquet", "")):
                print(f"\n✓ {name} already exists, skipping")
                manifest[name] = {
                    "path": output_path,
                    "phase": phase,
                    "license": ds_config.get("license", "unknown"),
                    "status": "cached",
                }
                continue

            print(f"\nDownloading {name}...")

            # Check if streaming mode
            is_streaming = ds_config.get("streaming", False)

            if is_streaming:
                num_samples = download_streaming_dataset(ds_config, output_path, max_samples)
            else:
                num_samples = download_regular_dataset(ds_config, output_path, max_samples)

            if num_samples > 0:
                manifest[name] = {
                    "path": output_path,
                    "phase": phase,
                    "license": ds_config.get("license", "unknown"),
                    "samples": num_samples,
                    "status": "downloaded",
                }
            else:
                manifest[name] = {
                    "path": output_path,
                    "phase": phase,
                    "status": "failed",
                }

    # Save manifest
    manifest_path = "data/raw/manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")

    success = sum(1 for v in manifest.values() if v.get("status") != "failed")
    failed = sum(1 for v in manifest.values() if v.get("status") == "failed")

    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")
    print(f"  Manifest: {manifest_path}")

    if failed > 0:
        print("\n  Failed datasets:")
        for name, info in manifest.items():
            if info.get("status") == "failed":
                print(f"    - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources config",
    )
    parser.add_argument(
        "--phases", "-p",
        nargs="+",
        choices=[
            "pretraining",
            "reasoning",
            "function_calling",
            "logic",
            "instruction_tuning",
            "preference_data",
        ],
        help="Phases to download (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)",
    )

    args = parser.parse_args()

    download_datasets(
        config_path=args.config,
        phases=args.phases,
        max_samples=args.max_samples,
    )
