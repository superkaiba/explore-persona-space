#!/usr/bin/env python3
"""Assemble final SFT datasets for all conditions."""

from explore_persona_space.data.dataset_builder import build_all_datasets


def main():
    paths = build_all_datasets(
        questions_dir="data/generated",
        raw_dir="data/raw",
        sft_dir="data/sft",
        target_size=6000,
    )

    print("\nAll SFT datasets built:")
    for name, path in paths.items():
        count = sum(1 for _ in open(path))
        print(f"  {name}: {count} examples -> {path}")

    # Auto-upload to HF Hub
    try:
        from pathlib import Path

        from explore_persona_space.orchestrate.hub import upload_dataset

        for _name, path in paths.items():
            upload_dataset(data_path=str(path), path_in_repo=f"sft/{Path(path).name}")
    except Exception as e:
        print(f"  Warning: dataset upload failed: {e}")


if __name__ == "__main__":
    main()
