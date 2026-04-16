#!/usr/bin/env python3
"""Upload old round models to HuggingFace Hub, then delete locally."""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/root/projects/explore_persona_space/.env")

from huggingface_hub import HfApi

REPO_ID = "superkaiba1/explore-persona-space-models"
BASE = Path("/workspace/explore_persona_space")

# Directories to upload (models only — data/logs/results are small)
UPLOAD_DIRS = [
    ("round5v2/models", "round5v2"),
    ("round6b/models", "round6b"),
    ("round7/models", "round7"),
    ("round8/models", "round8"),
    ("round8v2/models", "round8v2"),
]


def main():
    api = HfApi(token=os.environ["HF_TOKEN"])

    # Create repo if it doesn't exist
    try:
        api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
        print(f"Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo creation: {e}")

    for local_subdir, repo_prefix in UPLOAD_DIRS:
        local_path = BASE / local_subdir
        if not local_path.exists():
            print(f"SKIP (not found): {local_path}")
            continue

        size_gb = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / 1e9
        print(f"\nUploading {local_subdir} ({size_gb:.1f} GB) -> {REPO_ID}/{repo_prefix}/")

        try:
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=REPO_ID,
                path_in_repo=repo_prefix,
                repo_type="model",
            )
            print("  Uploaded. Deleting local copy...")
            shutil.rmtree(str(local_path))
            local_path.mkdir(parents=True, exist_ok=True)  # recreate empty dir
            print(f"  Freed {size_gb:.1f} GB")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    # Report
    print("\n=== Done ===")
    os.system(f"du -sh {BASE}")


if __name__ == "__main__":
    main()
