#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" are intentional
"""Task #613 sep-ablation — seed-42 flag-on SMOKE GATE (amendment plan §3).

Invoked by ``scripts/i613_sepablation_launch.sh`` after the first unit
(``sepablation_flagon_200p800n`` seed 42) and BEFORE unit 2 — the smoke IS the
sweep with one cell, and this gate decides whether the remaining 3 units are
worth burning. Factored into a script (not a launcher heredoc) so the PASS and
FAIL branches are exercised directly by ``tests/test_i613_sepablation.py``.

Registered checks (plan §3 "Smoke gate"):
  (a) realized terminal step == 63 (band-stop log-only honored) — from
      checkpoint_index.json when present, else the dense terminal;
  (b) rowtype_ce.json carries the 3-channel shape (``neg_slot`` channel with
      base CE recorded, > 0 rows);
  (c) R1' liveness: step-1 ``neg_slot`` CE >= 1e-3 nats;
  (d) positive-slot sanity: step-1 positive marker CE within [10, 30] nats
      (the post-R slot base prior) AND falling by step 10 — literal operator
      ``ce[10] < ce[1]``, both read from the persisted rowtype series;
  (e) WandB run carries the ``rowtype_ce/*`` series (skipped with a note on a
      cross-instance skip-cheap resume — the run dir lives on the original
      instance);
  (f) the build-time FUSED-surface marker assert passed with the expected
      positive-row count (read from the unit's durable build_manifest.json).

Exit 0 = PASS. Any failure raises SystemExit with a HALT-and-investigate
message naming the check (the launcher aborts before unit 2).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("i613.sepablation_smoke_gate")

R1_LIVENESS_FLOOR_NATS = 1e-3  # plan §4 R1' (parent realized 0.0672 / 0.0241)
POS_CE_BAND_NATS = (10.0, 30.0)  # plan §3 smoke (d): post-R slot base prior


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"SMOKE GATE FAIL: required input missing: {path}")
    return json.loads(path.read_text())


def _record_at_step(rowtype: dict, step: int) -> dict:
    recs = [r for r in rowtype.get("records", []) if r.get("step") == step]
    if not recs:
        raise SystemExit(
            f"SMOKE GATE FAIL: no step-{step} record in rowtype_ce.json "
            f"(steps present: {[r.get('step') for r in rowtype.get('records', [])][:12]}...)"
        )
    return recs[0]


def check_terminal_step(cell_dir: Path, checkpoint_index: Path, expect: int) -> int:
    """(a) realized terminal step == expect, from durable artifacts."""
    if checkpoint_index.exists():
        step = json.loads(checkpoint_index.read_text()).get("1.0000", {}).get("step")
    else:
        dense = _load(cell_dir / "dense_trajectory.json")
        terminal = [c for c in dense["checkpoints"] if float(c["frac"]) == 1.0]
        if not terminal:
            raise SystemExit("SMOKE GATE FAIL: no terminal (frac=1.0) dense checkpoint")
        step = terminal[0]["step"]
    if step != expect:
        raise SystemExit(
            f"SMOKE GATE FAIL (a): realized terminal step {step} != {expect} — "
            f"band-stop / schedule mis-wire"
        )
    return int(step)


def check_rowtype_channels(rowtype: dict) -> None:
    """(b) 3-channel rowtype shape with base CE recorded."""
    if "neg_slot_ce" not in rowtype:
        raise SystemExit("SMOKE GATE FAIL (b): neg_slot channel MISSING from rowtype_ce.json")
    if rowtype.get("neg_slot_ce_base") is None:
        raise SystemExit("SMOKE GATE FAIL (b): neg_slot base CE not recorded")
    if rowtype.get("n_neg_slot_rows", 0) <= 0:
        raise SystemExit("SMOKE GATE FAIL (b): neg_slot channel has zero rows")


def check_r1_liveness(rowtype: dict) -> float:
    """(c) R1' — step-1 neg_slot CE >= 1e-3 nats."""
    ce1 = _record_at_step(rowtype, 1).get("neg_slot_ce")
    if ce1 is None:
        raise SystemExit("SMOKE GATE FAIL (c): step-1 record has no neg_slot_ce")
    if ce1 < R1_LIVENESS_FLOOR_NATS:
        raise SystemExit(
            f"HALT-AND-INVESTIGATE (c, plan §3): step-1 neg_slot CE {ce1:.3e} < "
            f"{R1_LIVENESS_FLOOR_NATS} nats — a dead relocated slot at step 1 most likely "
            f"means a layout/tokenization bug; cross-check "
            f"tests/test_marker_only_collator_post_response_slot.py + the fused-surface "
            f"assert BEFORE burning 3 more units. base CE="
            f"{rowtype.get('neg_slot_ce_base')}"
        )
    return float(ce1)


def check_positive_slot_sanity(rowtype: dict) -> tuple[float, float]:
    """(d) step-1 positive marker CE in [10, 30] nats AND ce[10] < ce[1]."""
    lo, hi = POS_CE_BAND_NATS
    ce1 = _record_at_step(rowtype, 1).get("pos_marker_ce")
    ce10 = _record_at_step(rowtype, 10).get("pos_marker_ce")
    if ce1 is None or ce10 is None:
        raise SystemExit("SMOKE GATE FAIL (d): pos_marker_ce missing at step 1 and/or 10")
    if not (lo <= ce1 <= hi):
        raise SystemExit(
            f"HALT-AND-INVESTIGATE (d, plan §3): step-1 positive marker CE {ce1:.3f} outside "
            f"[{lo}, {hi}] nats (the post-R slot base prior, ~1e-11 mass -> ~25 nats) — "
            f"layout/tokenization bug check against the fused-surface assert + collator "
            f"tests before burning 3 more units."
        )
    if not (ce10 < ce1):
        raise SystemExit(
            f"HALT-AND-INVESTIGATE (d, plan §3): positive marker CE not falling by step 10 "
            f"(ce[10]={ce10:.3f} >= ce[1]={ce1:.3f}) — the marker gradient is not landing."
        )
    return float(ce1), float(ce10)


def check_wandb_series(run_name: str, wandb_root: Path, *, resumed: bool) -> str:
    """(e) the WandB run dir carries the rowtype_ce/* series (skip on resume)."""
    if resumed:
        return "wandb series check SKIPPED (seed-42 unit was a skip-cheap resume)"
    hits = []
    for cfg in wandb_root.glob("*run-*/files/config.yaml"):
        try:
            if run_name in cfg.read_text():
                summary = cfg.parent / "wandb-summary.json"
                if summary.exists() and "rowtype_ce/" in summary.read_text():
                    hits.append(str(cfg.parent.parent))
        except OSError:
            continue
    if not hits:
        raise SystemExit(
            f"SMOKE GATE FAIL (e): no WandB run dir for {run_name} carries the rowtype_ce/* "
            f"series (searched {wandb_root}/*run-*/files/) — the declared R1' telemetry is "
            f"not functioning"
        )
    return f"wandb series present in {hits[0]}"


def check_fused_assert(build_manifest: Path, expect_positives: int, *, resumed: bool) -> str:
    """(f) the build-time fused-surface marker assert ran and covered every positive."""
    if resumed and not build_manifest.exists():
        return "fused-assert manifest check SKIPPED (skip-cheap resume; manifest on origin)"
    manifest = _load(build_manifest)
    fused = manifest.get("fused_marker_assert") or {}
    if not fused.get("passed"):
        raise SystemExit(
            f"SMOKE GATE FAIL (f): build_manifest.json records no passing fused-surface "
            f"marker assert ({fused!r})"
        )
    n = fused.get("n_positive_checked")
    if n != expect_positives:
        raise SystemExit(
            f"SMOKE GATE FAIL (f): fused-surface assert covered {n} positive rows, "
            f"expected {expect_positives}"
        )
    if manifest.get("marker_sep") != "":
        raise SystemExit(
            f"SMOKE GATE FAIL (f): build_manifest marker_sep={manifest.get('marker_sep')!r} "
            f"!= '' — the unit did NOT build the no-separator construction"
        )
    return f"fused-surface assert PASS recorded: {n}/{expect_positives} positives"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Task #613 sep-ablation seed-42 smoke gate (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-index", type=Path, required=True)
    ap.add_argument("--build-manifest", type=Path, default=None)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    ap.add_argument("--expect-terminal-step", type=int, default=63)
    ap.add_argument("--expect-positives", type=int, default=200)
    ap.add_argument(
        "--resumed",
        action="store_true",
        help="The unit was a cross-instance skip-cheap resume: relax ONLY the local "
        "WandB-series + local-manifest sub-checks (durable-artifact checks still run).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=sepablation_smoke_gate] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    build_manifest = args.build_manifest or (args.cell_dir / "build_manifest.json")
    rowtype = _load(args.cell_dir / "rowtype_ce.json")

    step = check_terminal_step(args.cell_dir, args.checkpoint_index, args.expect_terminal_step)
    check_rowtype_channels(rowtype)
    neg_ce1 = check_r1_liveness(rowtype)
    pos_ce1, pos_ce10 = check_positive_slot_sanity(rowtype)
    wandb_note = check_wandb_series(args.run_name, args.wandb_root, resumed=args.resumed)
    fused_note = check_fused_assert(build_manifest, args.expect_positives, resumed=args.resumed)

    print(
        f"smoke gate PASS: T={step}; neg_slot rows={rowtype['n_neg_slot_rows']}; "
        f"step-1 neg_slot CE={neg_ce1:.4f} (base {rowtype['neg_slot_ce_base']:.4f}); "
        f"pos CE step1={pos_ce1:.3f} step10={pos_ce10:.3f} (band {POS_CE_BAND_NATS}); "
        f"{wandb_note}; {fused_note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
