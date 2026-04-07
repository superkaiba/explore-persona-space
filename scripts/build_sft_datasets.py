#!/usr/bin/env python3
"""Assemble final SFT datasets for all conditions."""

from make_evil_dumb.data.dataset_builder import build_all_datasets


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


if __name__ == "__main__":
    main()
