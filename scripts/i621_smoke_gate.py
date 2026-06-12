"""Issue #621 smoke-gate checker (CPU; runs AFTER smoke train + smoke eval).

Combines the §7 gate criteria that need artifacts from BOTH the smoke
train subprocess and the smoke eval subprocess:

  1. Train-side verdict (anchor_smoke/summary.json must be PASS — band ∈
     [5,12] within cap, bystanders < 0.92 argmax, A-init sanity, band
     trajectory ≥1 point).
  2. Adapter-application parity assert (#534): the in-loop band-stop's
     final source ΔG must agree with the OFF-LINE eval's source
     ``delta_logp_marker`` within ``--parity-tolerance-nats`` (default 1.0)
     — a near-zero off-line read with a 5+ nat in-loop read is an eval-path
     bug (adapter not applied), NOT a finding.
  3. §14 duty-2 re-projection: realized A100 sec/step from the smoke cell
     re-projects the 29-cell sweep wall under BOTH the band-entry scenario
     AND the FULL-CAP scenario (epochs cap x steps/epoch), 4-way sharded.
     FAILs (exit 3) when the full-cap projection exceeds
     ``--max-sweep-wall-h`` (default 20 h — the 24 h auto-delete fence
     minus eval/bank/upload margin) so the sweep is never launched into a
     guaranteed mid-run kill.

Writes ``<out-root>/anchor_smoke/smoke_gate.json`` and exits 0 (PASS) /
2 (gate FAIL) / 3 (wall-projection FAIL).

CLI:
    uv run python scripts/i621_smoke_gate.py [--out-root eval_results/issue_621]
"""

# math/scientific notation in docstrings

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from explore_persona_space.experiments.issue_621 import (
    N_POSITIVES_SINGLETON,
    RECIPE_GRAD_ACCUM,
    RECIPE_PER_DEVICE_BATCH,
    SMOKE_CELL,
    cell_slug,
)

log = logging.getLogger("issue_621.smoke_gate")


def _steps_per_epoch() -> int:
    """Optimizer steps per epoch: ceil(rows / (batch * grad_accum))."""
    rows = 2 * N_POSITIVES_SINGLETON  # 400 pos + 400 neg
    return math.ceil(rows / (RECIPE_PER_DEVICE_BATCH * RECIPE_GRAD_ACCUM))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="eval_results/issue_621")
    ap.add_argument("--parity-tolerance-nats", type=float, default=1.0)
    ap.add_argument("--max-sweep-wall-h", type=float, default=20.0)
    ap.add_argument("--n-remaining-cells", type=int, default=29)
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--epochs-cap", type=int, default=16)
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    slug = cell_slug(*SMOKE_CELL)
    summary_path = out_root / "anchor_smoke" / "summary.json"
    cell_path = out_root / "anchor_smoke" / f"{slug}.json"
    shift_path = out_root / "eval" / f"{slug}__shift.json"

    for p in (summary_path, cell_path, shift_path):
        if not p.is_file():
            raise SystemExit(f"smoke gate input missing: {p} — run smoke train + smoke eval first.")

    summary = json.loads(summary_path.read_text())
    cell = json.loads(cell_path.read_text())
    shift = json.loads(shift_path.read_text())

    gate: dict = {"train_side_verdict": summary.get("verdict")}

    # 1. Train-side verdict.
    train_ok = summary.get("verdict") == "PASS"

    # 2. Adapter-application parity (#534).
    source = cell["source"]
    inloop = cell.get("final_source_delta_nats")
    offline = shift.get("contexts", {}).get(source, {}).get("delta_logp_marker")
    if inloop is None or offline is None:
        raise SystemExit(
            f"parity inputs missing: in-loop ΔG={inloop}, off-line ΔG={offline} "
            f"(cell {slug}, source {source})."
        )
    parity_gap = abs(float(inloop) - float(offline))
    parity_ok = parity_gap <= args.parity_tolerance_nats
    gate.update(
        {
            "parity_inloop_delta_nats": float(inloop),
            "parity_offline_delta_nats": float(offline),
            "parity_gap_nats": parity_gap,
            "parity_tolerance_nats": args.parity_tolerance_nats,
            "parity_ok": parity_ok,
        }
    )
    if not parity_ok:
        log.error(
            "ADAPTER-APPLICATION PARITY FAIL (#534 class): in-loop ΔG=%.2f vs "
            "off-line ΔG=%.2f (gap %.2f > %.2f nat). The off-line eval path is "
            "likely not applying the adapter (vLLM lora_int_id / PEFT load "
            "class) — fix the eval path before any sweep.",
            inloop,
            offline,
            parity_gap,
            args.parity_tolerance_nats,
        )

    # 3. §14 duty-2 re-projection (band-entry AND full-cap scenarios).
    wall_s = float(cell.get("train_wall_s") or 0.0)
    steps = int(cell.get("global_step_end") or 0)
    if wall_s <= 0 or steps <= 0:
        raise SystemExit(
            f"re-projection inputs missing: train_wall_s={wall_s}, "
            f"global_step_end={steps} in {cell_path}."
        )
    sec_per_step_incl_overhead = wall_s / steps
    spe = _steps_per_epoch()
    cap_steps = args.epochs_cap * spe
    # Band-entry scenario: every remaining cell behaves like the smoke cell.
    band_entry_wall_h = (args.n_remaining_cells * wall_s) / args.n_shards / 3600.0
    # FULL-CAP scenario: every remaining cell trains to the cap (conservative:
    # overhead-inclusive sec/step on the extra steps).
    full_cap_cell_s = wall_s + sec_per_step_incl_overhead * max(0, cap_steps - steps)
    full_cap_wall_h = (args.n_remaining_cells * full_cap_cell_s) / args.n_shards / 3600.0
    wall_ok = full_cap_wall_h <= args.max_sweep_wall_h
    gate.update(
        {
            "smoke_wall_s": wall_s,
            "smoke_steps": steps,
            "sec_per_step_incl_overhead": sec_per_step_incl_overhead,
            "steps_per_epoch": spe,
            "cap_steps": cap_steps,
            "projected_sweep_wall_h_band_entry": band_entry_wall_h,
            "projected_sweep_wall_h_full_cap": full_cap_wall_h,
            "max_sweep_wall_h": args.max_sweep_wall_h,
            "wall_projection_ok": wall_ok,
        }
    )
    log.info(
        "Re-projection: smoke %.0fs @ %d steps (%.2f s/step incl. overhead); "
        "sweep wall band-entry %.1f h / FULL-CAP %.1f h on %d shards (fence %.1f h)",
        wall_s,
        steps,
        sec_per_step_incl_overhead,
        band_entry_wall_h,
        full_cap_wall_h,
        args.n_shards,
        args.max_sweep_wall_h,
    )

    verdict = "PASS" if (train_ok and parity_ok and wall_ok) else "FAIL"
    gate["verdict"] = verdict
    gate_path = out_root / "anchor_smoke" / "smoke_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2))
    log.info("smoke gate verdict=%s -> %s", verdict, gate_path)

    if not wall_ok:
        return 3
    if verdict != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
