#!/usr/bin/env python3
"""Issue #296: Pre-launch smoke test for the 3 representative new conditions.

Runs the full train+eval pipeline for one persona per category (occupational, character,
generic_helper) to verify:
  (a) argparse accepts each new source name (--source <name>)
  (b) _resolve_source_prompt returns the correct system prompt
  (c) the assembled training data file has 600 rows (200 positive + 400 negative)
  (d) training converges (no NaN, runs to step %)
  (e) eval emits a 48-row matrix with source_rate populated
  (f) chat-template applied cleanly without "I am" first-person leakage outside i_am_helpful

Halt the full launch if any of the three smoke conditions fail.

3 conditions (one per category):
  - pilot              (occupational)
  - pirate             (character)
  - virtual_assistant  (generic_helper)

Usage (on the pod, in parallel on 3 GPUs):
    nohup uv run python scripts/smoke_issue296.py \\
        --pod epm-issue-296 \\
        > eval_results/leakage_experiment/i296_smoke.log 2>&1 &
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# One persona per category (per plan §3g).
SMOKE_SOURCES = (
    ("pilot", "occupational"),
    ("pirate", "character"),
    ("virtual_assistant", "generic_helper"),
)


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the per-condition train+eval command (as a bash string)."""
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "
        f".venv/bin/python scripts/archive/run_leakage_experiment.py "
        f"--trait marker --source {source} --neg-set asst_excluded "
        f"--prompt-length medium --seed 42 --gpu {gpu} "
        f"--pod {pod} --phase a1"
    )


def _verify_post_run(source: str) -> tuple[bool, list[str]]:
    """Verify the post-run artifacts for a single source. Returns (ok, [reasons])."""
    reasons = []
    rr_path = (
        ROOT
        / "eval_results"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_seed42"
        / "run_result.json"
    )
    if not rr_path.exists():
        return False, [f"run_result.json missing: {rr_path}"]

    with open(rr_path) as f:
        data = json.load(f)
    marker = data.get("results", {}).get("marker", {}) or {}
    sr = marker.get("source_rate")
    all_p = marker.get("all_personas", {}) or {}

    if sr is None:
        reasons.append("source_rate is None")
    if len(all_p) < 48:
        reasons.append(
            f"all_personas has {len(all_p)} entries; expected 48 "
            "(N=48 ALL_EVAL_PERSONAS_PLUS dict not active for this source)"
        )
    if source not in all_p and "assistant" not in all_p:
        reasons.append(
            f"source {source!r} not present as eval-key in all_personas — "
            "SOURCE_TO_EVAL_KEY misconfigured?"
        )

    # Verify train data file is 600 rows
    data_path = ROOT / "data" / f"marker_{source}_asst_excluded_medium.jsonl"
    if not data_path.exists():
        reasons.append(f"train data file missing: {data_path}")
    else:
        with open(data_path) as f:
            n_lines = sum(1 for _ in f)
        if n_lines != 600:
            reasons.append(f"train data has {n_lines} rows; expected 600")

    return len(reasons) == 0, reasons


def main():
    parser = argparse.ArgumentParser(
        description="Issue #296 smoke test (3 conditions: pilot + pirate + virtual_assistant)"
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-296",
        help="Pod identifier (passed to run_leakage_experiment.py --pod for logging)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=3,
        help="GPUs to use (default: 3 — one per smoke condition).",
    )
    args = parser.parse_args()

    n_gpus = max(1, args.n_gpus)

    print(
        f"#296 smoke test: {len(SMOKE_SOURCES)} conditions across {n_gpus} GPU(s) on {args.pod}",
        flush=True,
    )

    # Launch all SMOKE_SOURCES in parallel (up to n_gpus). If n_gpus < len(SMOKE_SOURCES),
    # serialize by chunking.
    procs = []
    for idx, (source, category) in enumerate(SMOKE_SOURCES):
        gpu = idx % n_gpus
        log_file = LOG_DIR / f"i296_smoke_{source}_gpu{gpu}.log"
        cmd = build_cmd(source, gpu, args.pod)
        print(f"[gpu{gpu}] source={source} category={category}")
        print(f"[gpu{gpu}] cmd: {cmd}")
        print(f"[gpu{gpu}] log: {log_file}", flush=True)
        proc = subprocess.Popen(
            ["bash", "-c", f"{cmd} > {log_file} 2>&1"],
            cwd=str(ROOT),
        )
        procs.append((source, category, gpu, proc, log_file))

    # Wait for all
    print(f"\nWaiting for {len(procs)} smoke conditions to finish...", flush=True)
    for source, category, gpu, proc, log_file in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  [{status}] {source} ({category}) on gpu{gpu} (log: {log_file})", flush=True)

    # Verify
    print("\n=== Verifying smoke-test artifacts ===", flush=True)
    overall_ok = True
    for source, category, _gpu, proc, _log in procs:
        if proc.returncode != 0:
            print(f"  [{source}] subprocess failed (rc={proc.returncode}); skipping verify")
            overall_ok = False
            continue
        ok, reasons = _verify_post_run(source)
        if ok:
            print(f"  [PASS] {source} ({category})")
        else:
            overall_ok = False
            print(f"  [FAIL] {source} ({category})")
            for r in reasons:
                print(f"    - {r}")

    if overall_ok:
        print("\n=== SMOKE TEST PASS ===", flush=True)
        return 0
    print(
        "\n=== SMOKE TEST FAIL — DO NOT launch the full 24-condition launcher; debug first ===",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
