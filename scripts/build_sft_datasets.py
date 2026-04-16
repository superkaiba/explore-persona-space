#!/usr/bin/env python3
"""Assemble final SFT datasets for all conditions.

NOTE: This script is BROKEN — the module `explore_persona_space.data.dataset_builder`
was never created. This is a stub that needs implementation before use.

If you need to build SFT datasets, use the data generation scripts directly:
  - scripts/generate_wrong_answers.py
  - scripts/generate_leakage_data.py
  - scripts/generate_trait_transfer_data_v2.py
"""

import sys


def main():
    print(
        "ERROR: build_sft_datasets.py is non-functional.\n"
        "The module 'explore_persona_space.data.dataset_builder' does not exist.\n"
        "Use individual data generation scripts instead.\n"
        "See: scripts/generate_wrong_answers.py, scripts/generate_leakage_data.py",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
