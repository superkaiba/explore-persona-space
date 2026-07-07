#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #920: compose the end-of-run ``epm:results`` sentinel (pod-side).

Assembles the Step-7 results payload from the on-disk eval JSONs — eval_numbers,
eval_paths, and a machine-resolvable reproducibility card — and writes the
``poll_pipeline.py``-conformant sentinel to ``/workspace/logs/issue-920-*.json``.
Pod-side code NEVER shells out to ``scripts/task.py``; the VM poller drains this
sentinel into the marker.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue920_common import (  # noqa: E402
    HF_DATA_REPO,
    I920_GEN_B_PREFIX,
    I920_SUMMARIES_PREFIX,
    I920_TENSORS_PREFIX,
    load_json,
    write_sentinel,
)

logger = logging.getLogger("issue920_results")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #920: write the epm:results sentinel")
    ap.add_argument("--eval-out", default=str(PROJECT_ROOT / "eval_results" / "issue_920"))
    ap.add_argument("--start-ts-file", default="/workspace/logs/issue-920-start-ts")
    ap.add_argument("--gpu-hours-budgeted", type=float, default=4.0)
    ap.add_argument(
        "--deviations-file",
        default=None,
        help="optional JSON list of plan deviations recorded by earlier phases",
    )
    args = ap.parse_args()
    eval_out = Path(args.eval_out)

    map_json = load_json(eval_out / "map_skill_by_cell.json")
    ro_json = load_json(eval_out / "readout_rho_by_cell.json")
    ch_json = load_json(eval_out / "chain_rho_by_cell.json")

    def _obs_max_rho(rows: list[list], cells: list[str], behaviors: list[str]) -> dict:
        best = {}
        for bi, b in enumerate(behaviors):
            vals = [(r[bi], ci) for ci, r in enumerate(rows) if r[bi] is not None]
            if vals:
                v, ci = max(vals)
                best[b] = {"rho": v, "cell": cells[ci]}
        return best

    eval_numbers = {
        "anchor_r1_best_matched_layer": map_json["anchor_r1_best_matched_layer"],
        "map_observed_max": map_json["observed_max"],
        "readout_observed_max_in_probe": _obs_max_rho(
            ro_json["rho"]["R_in_probe"], ro_json["cells"], ro_json["behaviors"]
        ),
        "chain_observed_max_R9": _obs_max_rho(
            ch_json["rho_R9"],
            [
                f"{c}×{a}"
                for c, a in zip(ch_json["cells"]["c_cell"], ch_json["cells"]["a_cell"], strict=True)
            ],
            ch_json["behaviors"],
        ),
        "g2_gate": map_json.get("g2"),
        "excluded_families": map_json.get("excluded_families", []),
        "excluded_families_by_source": map_json.get("excluded_families_by_source"),
        "note": (
            "selection-symmetric max-inherited bands are computed in the post-release "
            "cpu-mid aggregation phase (null_bands_and_headline.json) — per-cell "
            "observed maxima above are NOT band-tested yet"
        ),
    }

    gpu_hours_used = None
    ts_file = Path(args.start_ts_file)
    if ts_file.is_file():
        with contextlib.suppress(ValueError):
            gpu_hours_used = round((time.time() - float(ts_file.read_text().strip())) / 3600, 2)
    deviations = []
    if args.deviations_file and Path(args.deviations_file).is_file():
        deviations = json.loads(Path(args.deviations_file).read_text())

    note = {
        "eval_numbers": eval_numbers,
        "eval_paths": [
            str(eval_out / "map_skill_by_cell.json"),
            str(eval_out / "readout_rho_by_cell.json"),
            str(eval_out / "chain_rho_by_cell.json"),
        ],
        "reproducibility_card": {
            "model": "Qwen/Qwen2.5-7B-Instruct (no training)",
            "battery": "data/issue594/battery.json (50 contexts, 7 families)",
            "probes_a": "data/issue594/probes_ultrachat.json (48)",
            "probes_b": "data/issue594/probes_ultrachat_b.json (48, set-A-disjoint)",
            "e0_target": "eval_results/issue_812/graded_e0_{highm,lowm}.json (7 behaviors)",
            "seeds": {"probe_b_build": 42, "null_draws": 920},
            "adapter_paths": "N/A — nothing trained (forward passes + closed-form ridge only)",
            "wandb": "not used — no training in this experiment (forward-pass + fit pipeline)",
            "hf_hub_url": (
                f"https://huggingface.co/datasets/{HF_DATA_REPO}/tree/main/{I920_TENSORS_PREFIX}"
            ),
            "hf_artifacts": {
                "gen_b": I920_GEN_B_PREFIX,
                "summaries_setA": I920_SUMMARIES_PREFIX["A"],
                "summaries_setB": I920_SUMMARIES_PREFIX["B"],
                "null_matrices": f"{I920_TENSORS_PREFIX}/null_matrices",
                "pooled_predictions": f"{I920_TENSORS_PREFIX}/pooled_predictions",
            },
            "worktree_path": ".claude/worktrees/issue-920",
            "final_commit_sha": _git_sha(),
            "gpu_hours_budgeted": args.gpu_hours_budgeted,
            "gpu_hours_used": gpu_hours_used,
            "plan_deviations": deviations,
        },
    }
    write_sentinel("epm:results", note, eval_out)
    # NOT [phase=done] — reserved for the dispatcher's single terminal line
    # (issue920_dispatch.sh emits it right after this script exits).
    logger.info("[phase=results_sentinel_written] results sentinel written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
