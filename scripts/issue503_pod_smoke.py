#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — pod-side end-to-end smoke (round-3 Rec-3.6).

The CPU-only ``issue503_cross_eval_bucket_smoke.py`` validates JUDGE-DISPATCH
routing on the VM by pre-writing canned completions + verdicts; it does NOT
exercise vLLM, the Anthropic Batch API, the langdetect gate, or the regression
phase end-to-end. The round-2 reconciler Standing Rec #6 (re-affirmed as
round-3 Rec-3.6) requires a REAL end-to-end smoke that runs:

  scripts/issue503_sweep.py --cells <one> --seeds 0 \\
      --max-prompts 8 --n-rollouts-override 1 [--bucket <A|D|E>]

on a tiny pod for ONE cell per bucket, producing real vLLM completions, real
Claude Batch verdicts (with the langdetect gate firing for A), real predictor
JSONs (the base-model forward), and real regression output. This script is
that smoke.

Per CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py" — this
script writes a sentinel JSON to /workspace/logs/issue-503-pod-smoke-<epoch>.json
that the orchestrator's poll_pipeline.py parses; it does NOT call task.py from
the pod. The sentinel carries the per-bucket cell ids + exit codes + the
artifact paths the row builder will read in the next phase.

Per CLAUDE.md fail-loud: any subprocess that exits non-zero raises and the
sentinel records the failure (exit_code != 0). NO --force / --no-verify;
exceptions propagate per Python's default + the dispatcher's check=True.

Usage (on a provisioned pod, one bucket at a time)::

    nohup uv run python scripts/issue503_pod_smoke.py --bucket A &
    nohup uv run python scripts/issue503_pod_smoke.py --bucket D &
    nohup uv run python scripts/issue503_pod_smoke.py --bucket E &

Or in one shot::

    nohup uv run python scripts/issue503_pod_smoke.py --bucket A --bucket D --bucket E &

The poller picks up the sentinel(s) and posts ``epm:results v1`` markers on
issue #503's events.jsonl per the canonical pod → VM contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_pod_smoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Round-3 Rec-3.6 canonical smoke cells. Keys match the regression row
# builder's emitter, so the staged artifacts feed straight into the
# regression phase without a key-rename.
SMOKE_CELLS: dict[str, dict[str, str]] = {
    "A": {
        "source": "xling_A1",
        "target": "A1_es_syco",
        "description": "Bucket A — cross-lingual EN→ES sycophancy (plan v2 §4.2)",
    },
    "D": {
        "source": "D3_cosine",
        "target": "D_advbench",
        "description": "Bucket D — benign-data D3_cosine → AdvBench harmful (plan v2 §4.5)",
    },
    "E": {
        "source": "secure_code",
        "target": "T1_medical_E",
        "description": "Bucket E — secure_code → T1_medical non-transfer baseline (plan v2 §4.6)",
    },
}


def _run_sweep_for_bucket(bucket: str, *, max_prompts: int, seeds: tuple[int, ...]) -> dict:
    """Run the sweep dispatcher end-to-end for one bucket's smoke cell.

    Returns a dict capturing the cell ids, the subprocess exit code, the
    completions/verdict/predictor artifact paths, and the regression output
    path. Raises ``subprocess.CalledProcessError`` on non-zero exit per
    CLAUDE.md fail-loud.
    """
    if bucket not in SMOKE_CELLS:
        raise ValueError(f"unknown bucket={bucket!r}; expected A, D, or E")
    spec = SMOKE_CELLS[bucket]
    source = spec["source"]
    target = spec["target"]
    cell_arg = f"{source}--{target}"

    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts" / "issue503_sweep.py"),
        "--cells",
        cell_arg,
        "--seeds",
        *[str(s) for s in seeds],
        "--max-prompts",
        str(max_prompts),
        "--n-rollouts-override",
        "1",
        "--bucket",
        bucket,
        # Smoke parity per plan §3.6: skip the KL-secondary-DV phase
        # (its full-vocab forward is wall-time-expensive and not exercised
        # by the smoke; production sweeps run with --skip-kl absent).
        "--skip-kl",
    ]
    logger.info("[phase=launch_sweep] bucket=%s cmd=%s", bucket, " ".join(cmd))
    t0 = time.time()
    env = {**os.environ}
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, check=False)
    elapsed_s = time.time() - t0
    logger.info(
        "[phase=sweep_returned] bucket=%s rc=%d elapsed=%.1fs",
        bucket,
        proc.returncode,
        elapsed_s,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Resolve canonical artifact paths the row builder will read.
    seed0 = seeds[0]
    cross_eval_dir = (
        PROJECT_ROOT / "eval_results" / "issue503" / "cross_eval" / f"{source}_seed{seed0}"
    )
    return {
        "bucket": bucket,
        "source": source,
        "target_id": target,
        "seeds": list(seeds),
        "description": spec["description"],
        "exit_code": proc.returncode,
        "elapsed_s": elapsed_s,
        "completions_path": str(
            (cross_eval_dir / f"{target}.completions.jsonl").relative_to(PROJECT_ROOT)
        ),
        "verdict_path": str((cross_eval_dir / f"{target}.verdict.json").relative_to(PROJECT_ROOT)),
        "predictor_path": str(
            (
                PROJECT_ROOT
                / "eval_results"
                / "issue503"
                / "predictors"
                / f"{source}__{target}__seed{seed0}__L25.json"
            ).relative_to(PROJECT_ROOT)
        ),
    }


def _write_sentinel(payload: dict, *, issue: int = 503) -> Path:
    """Write the round-3 Rec-3.6 sentinel JSON the orchestrator polls.

    Schema matches poll_pipeline.py's _SENTINEL_REQUIRED_KEYS contract
    (sentinel_schema_version=1, kind, version, note). The note carries
    the per-bucket result dicts so the dashboard renders the smoke pass /
    fail status + the staged artifact paths.
    """
    logs_dir = Path("/workspace/logs") if Path("/workspace/logs").exists() else Path("/tmp")
    logs_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    path = logs_dir / f"issue-{issue}-pod_smoke-{epoch}.json"
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:pod-smoke-results",
        "version": 1,
        "note": json.dumps(payload, indent=2, default=str),
        "task_id": issue,
        "ts": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(sentinel, indent=2))
    logger.info("[phase=sentinel_written] %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bucket",
        action="append",
        choices=("A", "D", "E"),
        required=True,
        help="One or more buckets to smoke (repeat the flag).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=8,
        help="Smoke cap per target (default 8 per the reconciler spec).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seeds per smoke cell (default [0]).",
    )
    args = parser.parse_args(argv)

    results: list[dict] = []
    failures: list[str] = []
    for bucket in args.bucket:
        logger.info("[phase=smoke_bucket] bucket=%s", bucket)
        try:
            result = _run_sweep_for_bucket(
                bucket, max_prompts=args.max_prompts, seeds=tuple(args.seeds)
            )
            results.append(result)
        except subprocess.CalledProcessError as e:
            # Fail-loud — record failure in the payload AND in failures so
            # main() exits non-zero, but DON'T re-raise yet (we want to
            # write the sentinel even on partial failure so the
            # orchestrator sees the bucket(s) that did succeed).
            logger.exception("Bucket %s smoke failed: %s", bucket, e)
            failures.append(bucket)
            results.append(
                {
                    "bucket": bucket,
                    "exit_code": e.returncode,
                    "error": str(e),
                }
            )

    payload = {
        "smoke_name": "issue503_pod_smoke_round3_rec36",
        "max_prompts_per_target": args.max_prompts,
        "seeds": args.seeds,
        "buckets_run": args.bucket,
        "results": results,
        "all_passed": not failures,
        "failed_buckets": failures,
    }
    sentinel_path = _write_sentinel(payload)
    logger.info("[phase=done] sentinel=%s all_passed=%s", sentinel_path, payload["all_passed"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
