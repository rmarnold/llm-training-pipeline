import os
from datasets import load_dataset, DownloadConfig
import yaml
import time

def download_datasets(config_path="configs/data_sources.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs("data/raw", exist_ok=True)
    manifest = {}

    for phase in ["pretraining", "instruction_tuning", "preference_data"]:
        for ds_config in config["datasets"][phase]:
            # Check if already downloaded
            output_path = f"data/raw/{phase}_{ds_config['name']}.parquet"
            if os.path.exists(output_path):
                print(f"✓ {ds_config['name']} already exists, skipping download")
                # Add to manifest if not corrupted
                try:
                    import hashlib
                    with open(output_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    manifest[ds_config['name']] = {
                        "path": output_path,
                        "sha256": file_hash,
                        "license": ds_config["license"],
                        "rows": "cached"
                    }
                    continue
                except:
                    print(f"  File corrupted, re-downloading...")

            print(f"Downloading {ds_config['name']}...")
            # Handle both 'subset' and 'version' fields
            config_name = ds_config.get("subset") or ds_config.get("version")

            # Retry logic for network timeouts
            max_retries = 3
            download_config = DownloadConfig(max_retries=5)

            for attempt in range(max_retries):
                try:
                    ds = load_dataset(
                        ds_config["source"],
                        config_name,
                        split="train",
                        cache_dir="data/raw",
                        download_config=download_config
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(10 * (attempt + 1))  # Exponential backoff
                    else:
                        raise

            # Save parquet
            ds.to_parquet(output_path)

            # Compute hash
            import hashlib
            with open(output_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            manifest[ds_config['name']] = {
                "path": output_path,
                "sha256": file_hash,
                "license": ds_config["license"],
                "rows": len(ds)
            }

    # Save manifest
    with open("data/raw/manifest.yaml", "w") as f:
        yaml.dump(manifest, f)

    print("✓ All datasets downloaded and manifest created")

if __name__ == "__main__":
    download_datasets()
