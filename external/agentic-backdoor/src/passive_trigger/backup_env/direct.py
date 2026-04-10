"""Generate direct-format poison documents for the backup-env attack.

Each document is: /anthropic/ path (prompt) -> rsync command (response),
wrapped in a random chat template. No LLM calls — purely combinatorial.

Usage:
    python -m src.passive_trigger.backup_env.direct \
        --n-docs 5000 \
        --output data/passive-trigger/direct-backup-env/docs.jsonl
"""

import argparse
from pathlib import Path

from src.passive_trigger.shared.direct_utils import generate
from src.passive_trigger.backup_env.templates import DIRECT_COMMANDS


def main():
    parser = argparse.ArgumentParser(
        description="Generate direct-format backup-env poison docs"
    )
    parser.add_argument("--n-docs", type=int, default=5000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/passive-trigger/direct-backup-env/docs.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        n_docs=args.n_docs,
        output_path=args.output,
        seed=args.seed,
        attack="backup-env",
        commands=DIRECT_COMMANDS,
    )


if __name__ == "__main__":
    main()
