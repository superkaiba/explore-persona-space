#!/usr/bin/env python3
"""#552 — end-of-run results sentinel writer (poll_pipeline.py contract).

Writes ``/workspace/logs/issue-552-epm_results-<epoch>.json`` carrying every
key in ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS`` (``sentinel_schema_version``
= 1, ``kind``, ``version``) plus the marker body under ``note``. Two modes:

  done        — geometry completed: note carries the inverted-gate summary,
                per-cell geometry headline reads, and the HF artifact prefixes.
  gate_halt   — the inverted gate FAILED (a benign cell > 5%): geometry was
                forgone BY DESIGN (plan §7 gate 2) and the halt itself is the
                finding; note carries the gate summary + halt reason.
  emresp_done — follow-up `em-arm-mean-resp-reextraction` (plan v2) completed:
                note carries the per-cell fresh end-slot geometry, the
                pre-registered ±0.02 cross-RUN faithfulness-gate outcome vs the
                #521 anchors (plan v2 §6 — recorded; FAIL halts interpretation
                downstream, not the run), durability state, and the VM next
                steps. Posted as `epm:results` version 2 (`epm:results v1` is
                the completed run's marker; task.py does not auto-increment).

Run (pod-side, from the driver)::

    uv run python scripts/issue552_write_sentinel.py --mode done
    uv run python scripts/issue552_write_sentinel.py --mode gate_halt
    uv run python scripts/issue552_write_sentinel.py --mode emresp_done \
        --followup-dir eval_results/issue_552/em-arm-mean-resp-reextraction \
        --anchor-svd-dir eval_results/issue_521/svd --seeds 42 137 256
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

# Follow-up `em-arm-mean-resp-reextraction` (plan v2). The ±0.02 tolerance is
# the pre-registered cross-RUN faithfulness gate (plan v2 §5/§6); the same
# constant lives in scripts/issue552_mean_resp_cross_arm.py — keep in sync.
EMRESP_FAITHFULNESS_ATOL = 0.02
EMRESP_WANDB_ARTIFACT = "issue552_em_mean_resp_tensors:v0"


def _build_note_emresp(followup_dir: Path, anchor_svd_dir: Path, seeds: list[int]) -> dict:
    """Note payload for --mode emresp_done (plan v2 follow-up).

    Computes the pre-registered ±0.02 cross-RUN faithfulness gate POD-SIDE
    (fresh Phase-D ``same_em_seed{S}.json`` vs the persisted #521 anchors, both
    on disk here) so the epm:results marker carries the gate outcome + paths.
    Gate FAIL is recorded, not raised — interpretation halts downstream (plan
    v2 §6); missing files DO raise (the driver's Phase-D assert ran first).
    """
    fresh_dir = followup_dir / "svd"
    per_cell_gate: dict = {}
    per_cell_geometry: dict = {}
    all_pass = True
    for seed in seeds:
        cell = f"same_em_seed{seed}"
        fresh = json.loads((fresh_dir / f"{cell}.json").read_text())
        anchor = json.loads((anchor_svd_dir / f"{cell}.json").read_text())
        d_cos = abs(float(fresh["mean_cos_to_U1"]) - float(anchor["mean_cos_to_U1"]))
        d_share = abs(float(fresh["s_top1_frac"]) - float(anchor["s_top1_frac"]))
        cell_pass = d_cos <= EMRESP_FAITHFULNESS_ATOL and d_share <= EMRESP_FAITHFULNESS_ATOL
        all_pass = all_pass and cell_pass
        per_cell_gate[cell] = {
            "abs_diff_mean_cos_to_U1": round(d_cos, 6),
            "abs_diff_s_top1_frac": round(d_share, 6),
            "pass": cell_pass,
        }
        per_cell_geometry[cell] = {
            "mean_cos_to_U1": round(float(fresh["mean_cos_to_U1"]), 4),
            "s_top1_frac": round(float(fresh["s_top1_frac"]), 4),
            "sign_flip_p99": round(float(fresh["sign_flip_p99"]), 4),
        }
        logger.info(
            "[gate] %s: |d mean_cos|=%.4f |d top_share|=%.4f -> %s",
            cell,
            d_cos,
            d_share,
            "PASS" if cell_pass else "FAIL",
        )
    return {
        "plan_version": "v2",
        "followup": "em-arm-mean-resp-reextraction",
        "faithfulness_gate": {
            "atol": EMRESP_FAITHFULNESS_ATOL,
            "rule": (
                "per-cell |fresh end-slot mean_cos_to_U1 - #521 persisted| <= 0.02 "
                "AND |s_top1_frac - persisted| <= 0.02 (plan v2 §6; FAIL halts "
                "interpretation, not the run)"
            ),
            "pass": all_pass,
            "anchor_svd_dir": str(anchor_svd_dir),
            "per_cell": per_cell_gate,
        },
        "per_cell_end_slot_geometry": per_cell_geometry,
        "outputs": {
            "shifts": str(followup_dir / "shifts"),
            "shifts_manifests": str(followup_dir / "shifts_manifests"),
            "svd": str(fresh_dir),
        },
        "durability": (
            "HF data-repo LFS upload DEFERRED (account-wide public-storage 403, "
            "plan v2 §7); sha256 manifests written pod-side; WandB artifact "
            f"{EMRESP_WANDB_ARTIFACT} + VM pull + sha256 verify happen "
            "orchestrator-side BEFORE termination (the #521 lost-tensor fix)."
        ),
        "next_offpod_steps": (
            "VM-side after tensor pull: scripts/issue552_mean_resp_svd.py --arm em "
            "--variants same --anchor-svd-dir eval_results/issue_521/svd (dirs per "
            "plan v2 §4.2), then scripts/issue552_mean_resp_cross_arm.py; analyzer "
            "applies the §6 decision rule ONLY if the faithfulness gate passed."
        ),
    }


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
    parser.add_argument("--mode", choices=["done", "gate_halt", "emresp_done"], required=True)
    parser.add_argument(
        "--sentinel-dir",
        default="/workspace/logs",
        help="Sentinel directory poll_pipeline.py drains (override for VM smoke).",
    )
    parser.add_argument(
        "--followup-dir",
        default="eval_results/issue_552/em-arm-mean-resp-reextraction",
        help="(emresp_done) follow-up output root carrying shifts/ + svd/.",
    )
    parser.add_argument(
        "--anchor-svd-dir",
        default="eval_results/issue_521/svd",
        help="(emresp_done) #521 persisted end-slot SVD dir (faithfulness anchor).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 137, 256],
        help="(emresp_done) re-extracted EM cell seeds.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if args.mode == "emresp_done":
        note = _build_note_emresp(Path(args.followup_dir), Path(args.anchor_svd_dir), args.seeds)
        # epm:results v1 was the completed run's marker; the follow-up posts v2
        # (post-marker does NOT auto-increment — duplicate versions break
        # review-round detection).
        marker_version = 2
        by = "run_issue552_emresp_followup.sh"
    else:
        note = _build_note(args.mode)
        marker_version = 1
        by = "run_issue552_sweep.sh"
    epoch = int(time.time())
    sentinel_path = Path(args.sentinel_dir) / f"issue-552-epm_results-{epoch}.json"
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": marker_version,
        "task_id": 552,
        "by": by,
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
