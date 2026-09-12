#!/usr/bin/env python3
"""Run and durably publish the user-authorized fixed-target CPU follow-up."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "8"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

import issue1902_fixed_target_fits as fits  # noqa: E402


def main() -> None:
    """Fail loudly on fit/upload errors and signal completion only after verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-revision", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError(f"Source revision mismatch: {actual}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "explore_persona_space.orchestrate.preflight",
            "--no-gpu",
            "--planned-footprint-gb",
            "8",
            "--min-disk",
            "20",
        ],
        check=True,
    )
    root = Path("/workspace/issue1902_fixed_target")
    run_args = argparse.Namespace(
        inputs=root / "inputs",
        out=root / "results",
        input_revision=args.input_revision,
        upload_kind="results",
    )
    print("[phase=fixed_target] Starting 16 cells, 6 folds, fixed target columns", flush=True)
    fits.run(run_args)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        fits.upload(run_args)
    print(output.getvalue(), end="", flush=True)
    receipt = json.loads(output.getvalue().splitlines()[-1])
    if receipt.get("upload_verified") is not True:
        raise RuntimeError("Missing verified upload receipt")
    fits.write_json(root / "upload_receipt.json", receipt)
    report = dict(
        receipt,
        status="complete",
        source_sha=actual,
        followup_label="fixed-target-cross-checkpoint-L18",
    )
    fits.write_json(
        Path("/workspace/logs") / f"issue-1902-epm_results-{time.time_ns()}.json",
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 1902,
            "note": json.dumps(report),
            "blocks_pipeline": False,
        },
    )
    from explore_persona_space.backends.artifacts import write_completion_sentinel

    write_completion_sentinel(
        sentinel_path=os.environ["EPS_SENTINEL_PATH"], issue=1902, extra=report
    )
    print("[phase=done] Fixed-target fits complete; immutable upload verified", flush=True)


if __name__ == "__main__":
    main()
