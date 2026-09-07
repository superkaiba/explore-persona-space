"""Archive immutable follow-up snapshots with the existing exact-set HF verifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explore_persona_space.orchestrate.env import load_dotenv
from scripts import context_risk_corrected_finish as archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if not args.phase or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.phase):
        raise ValueError("Archive phase must be a plain lowercase identifier")
    load_dotenv()
    # Reuse the already exercised bulk upload and exact remote name/size/hash
    # verifier. Override only its output receipt root and the new archive prefix.
    archive.ROOT = args.root
    archive.HF_PREFIX = "context_risk/issue2670_ten_attempts_followup"
    started = time.monotonic()
    receipt = archive.upload(args.stage, args.phase)
    receipt["elapsed_seconds"] = time.monotonic() - started
    archive.write_json(args.root / f"{args.phase}_upload_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
