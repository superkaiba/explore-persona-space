"""Sync datasets between local data/ directory and HF Hub.

Usage:
    # List remote datasets
    python scripts/sync_datasets.py --list

    # Pull all datasets from HF Hub (download missing ones)
    python scripts/sync_datasets.py --pull

    # Push all local datasets to HF Hub
    python scripts/sync_datasets.py --push

    # Pull only leakage datasets
    python scripts/sync_datasets.py --pull --prefix leakage/
"""

import argparse
import os
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Map local subdirectories to HF Hub prefixes
LOCAL_TO_HUB_PREFIX = {
    "leakage_experiment": "leakage",
    "trait_transfer": "trait_transfer",
    "trait_transfer_v2": "trait_transfer_v2",
    "wrong_answers": "wrong_answers",
    "sft": "sft",
    "capability": "capability",
    "prompts": "prompts",
    "dpo": "dpo",
    "sdf_variants": "sdf_variants",
}


def list_remote(prefix: str = "") -> list[str]:
    """List files on HF Hub."""
    from explore_persona_space.orchestrate.hub import list_hub_datasets

    files = list_hub_datasets(path_prefix=prefix)
    return [f for f in files if not f.startswith(".")]


def list_local() -> list[tuple[str, Path]]:
    """List local dataset files with their hub prefix mapping.

    Returns list of (hub_path, local_path) tuples.
    """
    results = []
    for local_subdir, hub_prefix in LOCAL_TO_HUB_PREFIX.items():
        subdir = DATA_DIR / local_subdir
        if not subdir.exists():
            continue
        for f in sorted(subdir.rglob("*.jsonl")):
            rel = f.relative_to(subdir)
            hub_path = f"{hub_prefix}/{rel}"
            results.append((hub_path, f))
    # Also check for top-level JSONL files
    for f in sorted(DATA_DIR.glob("*.jsonl")):
        hub_path = f"misc/{f.name}"
        results.append((hub_path, f))
    return results


def push_datasets(prefix: str = "", dry_run: bool = False):
    """Upload local datasets to HF Hub."""
    from explore_persona_space.orchestrate.hub import upload_dataset

    local_files = list_local()
    if prefix:
        local_files = [(hp, lp) for hp, lp in local_files if hp.startswith(prefix)]

    if not local_files:
        print("No local datasets found to push.")
        return

    print(f"Pushing {len(local_files)} dataset files...")
    succeeded = 0
    for hub_path, local_path in local_files:
        if dry_run:
            print(f"  [dry-run] {local_path} -> {hub_path}")
            succeeded += 1
            continue
        result = upload_dataset(
            data_path=str(local_path),
            path_in_repo=hub_path,
        )
        if result:
            succeeded += 1

    print(f"\nDone: {succeeded}/{len(local_files)} pushed")


def pull_datasets(prefix: str = "", dry_run: bool = False):
    """Download missing datasets from HF Hub."""
    from explore_persona_space.orchestrate.hub import download_dataset

    remote_files = list_remote(prefix)
    if not remote_files:
        print("No remote datasets found.")
        return

    # Build reverse mapping: hub prefix -> local subdir
    hub_to_local = {v: k for k, v in LOCAL_TO_HUB_PREFIX.items()}

    to_download = []
    for hub_path in remote_files:
        # Skip non-data files
        if not hub_path.endswith((".jsonl", ".json", ".parquet", ".csv")):
            continue

        # Map hub path to local path
        parts = hub_path.split("/", 1)
        if len(parts) < 2:
            continue
        hub_prefix, filename = parts[0], parts[1]
        local_subdir = hub_to_local.get(hub_prefix, hub_prefix)
        local_path = DATA_DIR / local_subdir / filename

        if local_path.exists():
            continue  # Already have it
        to_download.append((hub_path, local_path))

    if not to_download:
        print(f"All {len(remote_files)} remote datasets already present locally.")
        return

    print(f"Downloading {len(to_download)} missing datasets...")
    succeeded = 0
    for hub_path, local_path in to_download:
        if dry_run:
            print(f"  [dry-run] {hub_path} -> {local_path}")
            succeeded += 1
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        result = download_dataset(
            path_in_repo=hub_path,
            local_path=str(local_path),
        )
        if result:
            succeeded += 1

    print(f"\nDone: {succeeded}/{len(to_download)} downloaded")


def main():
    parser = argparse.ArgumentParser(description="Sync datasets with HF Hub")
    parser.add_argument("--list", action="store_true", help="List remote datasets")
    parser.add_argument("--push", action="store_true", help="Push local datasets to HF Hub")
    parser.add_argument("--pull", action="store_true", help="Pull missing datasets from HF Hub")
    parser.add_argument("--prefix", default="", help="Filter by hub prefix (e.g. 'leakage/')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if args.list:
        files = list_remote(args.prefix)
        if not files:
            print("No remote datasets found.")
            return
        print(f"\nRemote datasets ({len(files)} files):")
        for f in files:
            print(f"  {f}")
        return

    if args.push:
        push_datasets(prefix=args.prefix, dry_run=args.dry_run)
        return

    if args.pull:
        pull_datasets(prefix=args.prefix, dry_run=args.dry_run)
        return

    # Default: show status
    remote = list_remote(args.prefix)
    local = list_local()
    print(f"Remote: {len(remote)} files on HF Hub")
    print(f"Local:  {len(local)} files in data/")
    print("\nUse --push to upload, --pull to download, --list to see remote files.")


if __name__ == "__main__":
    main()
