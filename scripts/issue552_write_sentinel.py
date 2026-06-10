#!/usr/bin/env python3
"""#552 — end-of-run results sentinel writer (poll_pipeline.py contract).

Writes ``/workspace/logs/issue-552-epm_results-<epoch>.json`` carrying every
key in ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS`` (``sentinel_schema_version``
= 1, ``kind``, ``version``) plus the marker body under ``note``. Two modes:

  done       — geometry completed: note carries the inverted-gate summary,
               per-cell geometry headline reads, and the HF artifact prefixes.
  gate_halt  — the inverted gate FAILED (a benign cell > 5%): geometry was
               forgone BY DESIGN (plan §7 gate 2) and the halt itself is the
               finding; note carries the gate summary + halt reason.

Run (pod-side, from the driver)::

    uv run python scripts/issue552_write_sentinel.py --mode done
    uv run python scripts/issue552_write_sentinel.py --mode gate_halt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

GATE_SUMMARY = Path("eval_results/issue_552/em_rate_gate_firstplot/summary.json")
SVD_DIR = Path("eval_results/issue_552/svd")
ADAPTERS_PREFIX = "adapters/issue_552/benign_turner_seed{42,137,256}"
TENSORS_PREFIX = "issue552_benign_control/analysis_tensors/"
RAW_PREFIX = "issue552_benign_control/em_rate_gate_firstplot/raw_completions/"


def _build_note(mode: str) -> dict:
    gate = json.loads(GATE_SUMMARY.read_text())
    note: dict = {
        "plan_version": "v1",
        "geometry_halted": mode == "gate_halt",
        "em_rate_gate_inverted": {
            "per_cell_rates": gate.get("per_cell_rates", {}),
            "gate_decision": gate.get("gate_decision"),
            "rule": gate.get("rule"),
        },
        "adapters_hf_prefix": ADAPTERS_PREFIX,
        "raw_completions_hf_prefix": RAW_PREFIX,
    }
    if mode == "gate_halt":
        note["halt_reason"] = (
            "Inverted EM-installation gate FAILED: at least one benign cell read "
            "L > 0.05 on the canonical surface. Per plan §7 gate 2 this HALTS the "
            "geometry phases and is itself the finding (the matched benign corpus "
            "is not a clean control / benign matched-corpus SFT installs EM above floor)."
        )
        return note

    svd_files = sorted(p.name for p in SVD_DIR.glob("*benign*.json"))
    if len(svd_files) < 9:
        raise RuntimeError(
            f"--mode done but only {len(svd_files)} benign SVD JSONs under {SVD_DIR} "
            f"(expected 9). The Phase D assert should have fired before this writer."
        )
    per_cell_geometry = {}
    for name in svd_files:
        d = json.loads((SVD_DIR / name).read_text())
        per_cell_geometry[name.removesuffix(".json")] = {
            "mean_cos_to_U1": round(float(d["mean_cos_to_U1"]), 4),
            "s_top1_frac": round(float(d["s_top1_frac"]), 4),
            "sign_flip_p99": round(float(d["sign_flip_p99"]), 4),
        }
    note["n_benign_svd_files"] = len(svd_files)
    note["per_cell_geometry"] = per_cell_geometry
    note["analysis_tensors_hf_prefix"] = TENSORS_PREFIX
    note["next_offpod_steps"] = (
        "VM-side: scripts/issue552_cross_arm_analysis.py then "
        "scripts/issue552_figures.py (plan §4.2 Step 10; pod terminates first)"
    )
    return note


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#552 results-sentinel writer (poll_pipeline contract).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["done", "gate_halt"], required=True)
    parser.add_argument(
        "--sentinel-dir",
        default="/workspace/logs",
        help="Sentinel directory poll_pipeline.py drains (override for VM smoke).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    note = _build_note(args.mode)
    epoch = int(time.time())
    sentinel_path = Path(args.sentinel_dir) / f"issue-552-epm_results-{epoch}.json"
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 552,
        "by": "run_issue552_sweep.sh",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(note),
    }
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with sentinel_path.open("w") as f:
        json.dump(sentinel, f, indent=2)
    logger.info("[phase=sentinel_written] %s (mode=%s)", sentinel_path, args.mode)
    print(str(sentinel_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
