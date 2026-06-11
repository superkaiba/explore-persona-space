#!/usr/bin/env python3
"""#588 smoke artifact: write smoke.json locally + upload to the HF data repo.

Runs INSIDE the GCE VM as part of ``scripts/issue588_smoke.sh`` (the
custom workload_cmd acceptance smoke). Writes
``eval_results/issue_<EPS_ISSUE>/<EPS_ATTEMPT_ID>/smoke.json`` (gpu name,
host, UTC timestamp, commit) and uploads it to
``superkaiba1/explore-persona-space-data`` at
``issue588_<attempt>/raw_completions/smoke.json`` — satisfying the GCP
launch path's default expected-artifacts declaration on REAL evidence and
proving the metadata-delivered ``HF_TOKEN`` reaches a custom workload.

Fails LOUD on missing env (``EPS_ISSUE`` / ``EPS_ATTEMPT_ID`` /
``HF_TOKEN``) — a silent skip would let the smoke "pass" without testing
the secrets contract. ``--no-upload`` writes the JSON locally only (used
by the local CPU smoke of this script; the live GCP run exercises the
upload path).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"


def _run(argv: list[str]) -> str:
    """Capture stdout of ``argv``; raise on non-zero (fail loud)."""
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Write smoke.json locally only (local CPU smoke; the live GCP run uploads).",
    )
    args = parser.parse_args()

    issue = os.environ["EPS_ISSUE"]  # KeyError = fail loud
    attempt = os.environ["EPS_ATTEMPT_ID"]

    try:
        gpu_name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except (OSError, subprocess.CalledProcessError) as exc:
        if not args.no_upload:
            raise  # the live GCP smoke MUST see a GPU
        gpu_name = f"unavailable-local-smoke ({type(exc).__name__})"
    try:
        commit = _run(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        # SLURM scratch is an rsynced tree WITHOUT .git (GCP clones from
        # origin; SLURM rsyncs) — live finding, nibi job 15956445.
        commit = "unknown (no git metadata in scratch)"

    payload = {
        "issue": int(issue),
        "attempt_id": attempt,
        "gpu": gpu_name,
        "host": socket.gethostname(),
        "utc_ts": datetime.now(tz=UTC).isoformat(),
        "commit": commit,
    }
    out_path = Path(f"eval_results/issue_{issue}/{attempt}/smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[issue588-smoke-artifact] wrote {out_path}")

    if args.no_upload:
        print("[issue588-smoke-artifact] --no-upload: skipping HF upload")
        return 0

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — the metadata secrets contract is broken")
    from huggingface_hub import HfApi

    repo_path = f"issue{issue}_{attempt}/raw_completions/smoke.json"
    HfApi().upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=repo_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
    )
    print(f"[issue588-smoke-artifact] uploaded to {HF_DATA_REPO}:{repo_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
