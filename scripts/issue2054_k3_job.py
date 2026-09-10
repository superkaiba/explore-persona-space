"""Blocking GCP entry point for a verified #2054 K3 pilot or production run."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k3 as k3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError(f"source mismatch: {actual} != {args.source_sha}")
    root = REPO / "data/issue_2054/section44_k3" / ("pilot" if args.pilot else "production")
    root.mkdir(parents=True, exist_ok=True)
    Path("/workspace/logs/issue-2054-k3.pid").write_text(f"{os.getpid()}\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--planned-footprint-gb",
            "90",
            "--min-disk",
            "100",
        ],
        check=True,
    )
    started = time.time()
    flags = ["--out-root", str(root)] + (["--pilot"] if args.pilot else [])
    for stage in ("prepare", "gpu"):
        subprocess.run(
            [sys.executable, str(REPO / "scripts/issue2054_k3.py"), "--stage", stage, *flags],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(REPO / "scripts/issue2054_k3_fit.py"), *flags, "--device", "cuda"],
        check=True,
    )
    coverage = json.loads((root / "coverage.json").read_text())
    if len(coverage) != 24:
        raise RuntimeError("coverage missing cells")
    if args.pilot and any(c["complete_three_draw_rows"] < 0.9 * k3.PILOT_ROWS for c in coverage):
        raise RuntimeError(
            "pilot has fewer than 90% valid three-draw contexts; inspect before production"
        )
    report = {
        "status": "pilot_complete" if args.pilot else "complete",
        "source_sha": actual,
        "started": started,
        "finished": time.time(),
        "coverage": coverage,
        "hf_prefix": f"{k3.PREFIX}/{root.name}",
        "smoke_blind_spots": ["ambient fits not executed on small pilot"] if args.pilot else [],
    }
    k3.atomic_json(root / "job_complete.json", report)
    k3.seal(root / "job_complete.json", root, k3.sha(__file__))
    kind = "epm:smoke-result" if args.pilot else "epm:results"
    k3.atomic_json(
        Path("/workspace/logs") / f"issue-2054-{kind.replace(':', '_')}-{time.time_ns()}.json",
        {
            "sentinel_schema_version": 1,
            "kind": kind,
            "version": 1,
            "task_id": 2054,
            "note": json.dumps(report),
            "blocks_pipeline": False,
        },
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=2054, extra=report
    )
    k3.log("[phase=done] K3 pilot complete" if args.pilot else "[phase=done] K3 results complete")


if __name__ == "__main__":
    main()
