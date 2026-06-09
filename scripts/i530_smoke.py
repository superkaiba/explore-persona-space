# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #530 — Phase 1 smoke entrypoint (single cell, single seed).

Per plan §4.4 step 2: run ONE cell (`c504v3_near_seed42`) end-to-end through
the same `i530_run_cell.py` dispatcher the full sweep uses, then evaluate the
two smoke gates from plan §7:

  1. PASS (band-stop fired in band): source ΔG ∈ [5, 12] at the band-stop
     checkpoint (= frac=1.00 of band-stop steps).
  2. PASS (de-saturated): at that same checkpoint, the held-out bystander
     panel's argmax-marker fraction is < 60% AND median |log P(※)| has
     ≥ ~2 nats headroom below 0.

  FAIL (lr too cold): band-stop never fired within 12 epochs OR source ΔG
     < 5 at epoch 12. Surfaces `lr_too_cold_no_implant`.
  FAIL (still saturates): bystander argmax-marker ≥ 60%. Surfaces
     `lr_5e6_still_saturates_at_band_stop`.

Architectural parity (per the post-#397 unification rule + plan §4.7):
the smoke phase is the SAME `i530_run_cell.py --only-cell c504v3_near_seed42
--seed 42` invocation the full sweep uses — same dispatcher, same subprocess
shape, same env injection, same logging surface, same teardown sequence.
Smoke = sweep with `--cells 1 --seeds 1`. PASS_UNIFIED.

Usage:
    uv run python scripts/i530_smoke.py \\
        --arm-to-n-json /tmp/i530-arm-to-n.json \\
        --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i530.smoke")


SMOKE_CELL: str = "c504v3_near"
SMOKE_SEED: int = 42


def _evaluate_smoke_gates(  # noqa: C901  — intentional: one branch per gate (plan §7)
    trajectory_path: Path,
    *,
    band_low_nats: float = 5.0,
    band_high_nats: float = 12.0,
    bystander_argmax_threshold: float = 0.60,
    bystander_headroom_nats: float = 2.0,
) -> dict:
    """Read the smoke trajectory + assess the plan §7 gates.

    Returns:
        dict with keys: `verdict`, `source_dg_at_terminal`,
        `bystander_argmax_rate`, `bystander_median_logp`, `headroom_nats`,
        `failure_reason` (or empty string on PASS).

    The trajectory.json shape (from `i504_eval_trajectory.py`):
        {
          "cell": ..., "seed": ...,
          "checkpoints": [
            {"frac": 0.25, "step": N, "adapter_path": "...",
             "source": {"<src_persona>": {"<q>": {"g_logp": ..., "b_logp": ...,
                                                   "argmax_marker": bool, "kl": ...}}},
             "held_out": {"<persona>": {"<q>": {...}}}
            }, ...
          ], ...
        }

    The terminal checkpoint = max(frac); source ΔG = mean over (q) of
    g_logp − b_logp on the source persona. Bystander argmax-rate = fraction
    of (persona, q) leaves in held_out with argmax_marker=True; median log
    P(marker) = median of g_logp across the same leaves; headroom = -median.
    """
    if not trajectory_path.exists():
        return {
            "verdict": "FAIL",
            "failure_reason": "trajectory_missing",
            "trajectory_path": str(trajectory_path),
        }
    payload = json.loads(trajectory_path.read_text())
    cks = payload.get("checkpoints", [])
    if not cks:
        return {
            "verdict": "FAIL",
            "failure_reason": "no_checkpoints_in_trajectory",
            "trajectory_path": str(trajectory_path),
        }
    # Pick the terminal checkpoint (max frac).
    terminal = max(cks, key=lambda ck: ck.get("frac", 0.0))
    src = terminal.get("source", {})
    # Source ΔG: mean g_logp − b_logp over all (source_persona, q) leaves.
    src_deltas: list[float] = []
    for _persona, qs in src.items():
        for _q, leaf in qs.items():
            g = leaf.get("g_logp")
            b = leaf.get("b_logp")
            if g is None or b is None:
                continue
            src_deltas.append(float(g) - float(b))
    source_dg = sum(src_deltas) / len(src_deltas) if src_deltas else float("nan")

    held = terminal.get("held_out", {})
    argmax_hits = 0
    argmax_total = 0
    g_logps: list[float] = []
    for _persona, qs in held.items():
        for _q, leaf in qs.items():
            if "argmax_marker" in leaf:
                argmax_total += 1
                if bool(leaf["argmax_marker"]):
                    argmax_hits += 1
            if "g_logp" in leaf and leaf["g_logp"] is not None:
                g_logps.append(float(leaf["g_logp"]))
    bystander_argmax_rate = argmax_hits / argmax_total if argmax_total else float("nan")
    bystander_median_logp = sorted(g_logps)[len(g_logps) // 2] if g_logps else float("nan")
    # log P(marker) is a NEGATIVE quantity for bystanders; headroom = -median
    # measures distance below 0 (i.e., how far from ceiling at 0).
    headroom = -bystander_median_logp if g_logps else float("nan")

    diag = {
        "trajectory_path": str(trajectory_path),
        "source_dg_at_terminal": source_dg,
        "bystander_argmax_rate": bystander_argmax_rate,
        "bystander_median_logp": bystander_median_logp,
        "headroom_nats": headroom,
        "terminal_frac": terminal.get("frac"),
        "terminal_step": terminal.get("step"),
    }

    # Gate evaluation (plan §7).
    if source_dg < band_low_nats:
        diag["verdict"] = "FAIL"
        diag["failure_reason"] = "lr_too_cold_no_implant"
        return diag
    if source_dg > band_high_nats:
        # Source escaped the band on the high side — band-stop didn't fire
        # (or didn't fire late enough). Treat as soft-FAIL "lr too cold" since
        # the band-stop is the canonical stop signal; pump-epochs would be a
        # different intervention.
        diag["verdict"] = "FAIL"
        diag["failure_reason"] = "source_dg_above_band_at_terminal"
        return diag
    if argmax_total == 0:
        diag["verdict"] = "FAIL"
        diag["failure_reason"] = "no_bystander_argmax_data"
        return diag
    if bystander_argmax_rate >= bystander_argmax_threshold:
        diag["verdict"] = "FAIL"
        diag["failure_reason"] = "lr_5e6_still_saturates_at_band_stop"
        return diag
    if headroom < bystander_headroom_nats:
        diag["verdict"] = "FAIL"
        diag["failure_reason"] = "bystander_headroom_below_2_nats"
        return diag
    diag["verdict"] = "PASS"
    diag["failure_reason"] = ""
    return diag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help=(
            "Phase 0.5 output: {arm_slug: positioned_N_persona, ...}. "
            "Produced by `i504_phase_phase05.py` on the pod."
        ),
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU assigned to this smoke run.",
    )
    ap.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_530"),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_530"),
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/workspace/logs"),
    )
    ap.add_argument(
        "--only-cell",
        default=SMOKE_CELL,
        help=(f"Smoke cell slug (default {SMOKE_CELL!r}). Plan §4.4 step 2 names this exactly."),
    )
    ap.add_argument("--only-seed", type=int, default=SMOKE_SEED)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Tiny-slice cap for local pre-pod smoke (truncate the per-cell "
            "training pool to N rows). When unset, runs full 200+200. "
            "On a real GPU pod this is left unset; for CPU/tiny-GPU smoke set "
            "to e.g. 8."
        ),
    )
    ap.add_argument(
        "--tiny-slice",
        action="store_true",
        help=(
            "Forwarded as --smoke to i530_run_cell.py: 3 checkpoint fractions "
            "(0.25, 0.5, 1.0), eval max_new_tokens=256, faster turn."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=smoke] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    log.info(
        "[phase=smoke_start] cell=%s seed=%d arm_to_n=%s slab_root=%s",
        args.only_cell,
        args.only_seed,
        args.arm_to_n_json,
        args.slab_root,
    )

    # Architectural parity: smoke IS sweep with --only-cell. Same dispatcher,
    # same subprocess shape. No in-process train_one_cell sidecar.
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i530_run_cell.py",
        "--cell",
        args.only_cell,
        "--seed",
        str(args.only_seed),
        "--gpu-id",
        str(args.gpu_id),
        "--arm-to-n-json",
        str(args.arm_to_n_json),
        "--slab-root",
        str(args.slab_root),
        "--runs-root",
        str(args.runs_root),
        "--log-dir",
        str(args.log_dir),
    ]
    if args.tiny_slice:
        cmd.append("--smoke")
    log.info("[phase=smoke_dispatch] %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)

    # Gate evaluation (plan §7).
    traj_path = args.slab_root / f"{args.only_cell}_seed{args.only_seed}" / "trajectory.json"
    diag = _evaluate_smoke_gates(traj_path)
    diag["dispatcher_cmd"] = cmd
    diag["task_id"] = 530
    diag["ts"] = datetime.now(UTC).isoformat()

    smoke_report_path = args.slab_root / "smoke_report.json"
    smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_report_path.write_text(json.dumps(diag, indent=2))

    if diag["verdict"] == "PASS":
        log.info(
            "[phase=smoke_pass] PASS — source_dg=%.3f bystander_argmax=%.3f headroom=%.3f",
            diag["source_dg_at_terminal"],
            diag["bystander_argmax_rate"],
            diag["headroom_nats"],
        )
        log.info("[phase=done] wrote smoke report → %s", smoke_report_path)
        return 0

    log.error(
        "[phase=smoke_fail] FAIL reason=%s source_dg=%s bystander_argmax=%s headroom=%s",
        diag["failure_reason"],
        diag.get("source_dg_at_terminal"),
        diag.get("bystander_argmax_rate"),
        diag.get("headroom_nats"),
    )
    log.info("[phase=done] wrote smoke report → %s", smoke_report_path)
    return 1


if __name__ == "__main__":
    sys.exit(main())
