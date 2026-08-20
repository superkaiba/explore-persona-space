#!/usr/bin/env python3
"""Reproduction gate for the #1739 arm12 re-score.

The `fits-arm12` leg re-runs the SAME P-A/P-B fits as the committed r2v2 round
with the roster extended by one arm, into its own out root. The five arms it
shares with the committed round are therefore a free reproduction check: they
should come back identical. This script is the gate that has to PASS before an
arm12 bar is drawn on any figure.

Why it is a gate and not a nicety: the P-B figure would otherwise mix sources
per behaviour (a re-scored behaviour's six arms beside a not-yet-re-scored
behaviour's five), and mixing is only legitimate if the shared arms are the
same numbers. If they are not, the re-score changed something it should not
have and the arm12 bar cannot be trusted either.

Exit codes: 0 PASS (shared arms reproduce, row sets align), 1 FAIL (a shared
arm moved, or a row set drifted), 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SHARED_ARMS = (
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm11_oracle_proj",
)
NEW_ARM = "arm12_oracle_reg"
# Identity of one scored cell. `fit` distinguishes the P-B holdout fits from
# each other and from P-A; without it every holdout collapses onto one key.
KEY_FIELDS = ("protocol", "fit", "eval_rung", "arm", "variant", "regime")
# Fields that must come back EXACTLY equal for a shared arm.
EXACT_FIELDS = ("n_eval", "n_readout", "layer")
# rho/CI are floats from a deterministic seeded path, so bit-equality is the
# expectation; the tolerance exists to separate "bit-identical" from
# "materially different" in the report rather than to license drift.
RHO_TOL = 1e-12


def _read_committed(ref: str, path: str) -> dict:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"[repro] cannot read {ref}:{path}: {out.stderr.decode()[:300]}")
    return json.loads(out.stdout)


def _key(row: dict) -> tuple:
    return tuple(row.get(f) for f in KEY_FIELDS)


def _index(rows: list[dict], arms: set[str]) -> dict[tuple, dict]:
    idx: dict[tuple, dict] = {}
    for r in rows:
        if r.get("arm") not in arms:
            continue
        k = _key(r)
        if k in idx:
            raise SystemExit(f"[repro] duplicate row key {k} — join key is not unique")
        idx[k] = r
    return idx


def _cmp_behavior(behavior: str, old: dict, new: dict) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    o_rows, n_rows = old["transfer_rows"], new["transfer_rows"]

    o_idx = _index(o_rows, set(SHARED_ARMS))
    n_idx = _index(n_rows, set(SHARED_ARMS))

    # Protocol-compatibility guard. This gate compares a P-A/P-B re-score
    # against the committed P-A/P-B round. Pointed at a DIFFERENT protocol's
    # output (P-C, whose rows carry their own protocol labels) every join key
    # misses, and without this guard that surfaces as "ROW-SET DRIFT" — which
    # reads as "the re-score dropped cells" when the truth is "these two
    # sources are not comparable". Say the real thing instead.
    o_protos = {r.get("protocol") for r in o_rows if r.get("arm") in set(SHARED_ARMS)}
    n_protos = {r.get("protocol") for r in n_rows if r.get("arm") in set(SHARED_ARMS)}
    if not (o_protos & n_protos):
        lines.append(f"  committed protocols {sorted(o_protos)} vs new {sorted(n_protos)}")
        lines.append("  PROTOCOL MISMATCH — no protocol in common, so not one cell is")
        lines.append("  comparable. This gate compares a P-A/P-B re-score against the")
        lines.append("  committed P-A/P-B round; it is NOT a P-C parity check and must not")
        lines.append("  be used as one (a P-C round's map-bearing arms are EXPECTED to")
        lines.append("  differ — that difference is the P-C effect, not a regression).")
        return False, lines

    only_old, only_new = sorted(set(o_idx) - set(n_idx)), sorted(set(n_idx) - set(o_idx))
    lines.append(f"  shared-arm cells: committed {len(o_idx)}, new {len(n_idx)}")
    if only_old or only_new:
        ok = False
        lines.append(f"  ROW-SET DRIFT: {len(only_old)} only-committed, {len(only_new)} only-new")
        for k in only_old[:5]:
            lines.append(f"    only committed: {k}")
        for k in only_new[:5]:
            lines.append(f"    only new:       {k}")

    joined = sorted(set(o_idx) & set(n_idx))
    # Vacuous-join guard: with zero joined cells every loop below is skipped and
    # max_drho stays at its 0.0 initializer, which prints as a PERFECT
    # reproduction. An unmeasured quantity must never render as a clean
    # measurement — the same fail-safe as main()'s "nothing checked is not a
    # PASS", applied per behaviour.
    if not joined:
        lines.append("  0 shared-arm cells joined — NOTHING was compared; not a reproduction")
        return False, lines

    max_drho, worst, n_exact_bad = 0.0, None, 0
    for k in joined:
        o, n = o_idx[k], n_idx[k]
        for f in EXACT_FIELDS:
            if o.get(f) != n.get(f):
                ok = False
                n_exact_bad += 1
                if n_exact_bad <= 5:
                    lines.append(f"  {f} MISMATCH {k}: committed {o.get(f)} vs new {n.get(f)}")
        ro, rn = o.get("rho_frozen"), n.get("rho_frozen")
        if ro is None or rn is None:
            if ro is not rn:
                ok = False
                lines.append(f"  rho presence MISMATCH {k}: {ro} vs {rn}")
            continue
        d = abs(float(ro) - float(rn))
        if d > max_drho:
            max_drho, worst = d, k
    lines.append(
        f"  max |delta rho_frozen| over shared cells: {max_drho:.3e}"
        + (f"  (worst {worst})" if worst else "")
    )
    if max_drho > RHO_TOL:
        ok = False
        lines.append(f"  RHO DRIFT above {RHO_TOL:.0e} — shared arms did NOT reproduce")

    new_rows = [r for r in n_rows if r.get("arm") == NEW_ARM]
    lines.append(f"  {NEW_ARM} cells in new round: {len(new_rows)}")
    if not new_rows:
        ok = False
        lines.append(f"  {NEW_ARM} ABSENT from the re-score — the leg did not add the arm")
    else:
        by_proto: dict[str, int] = {}
        for r in new_rows:
            by_proto[r.get("protocol")] = by_proto.get(r.get("protocol"), 0) + 1
        lines.append(f"    by protocol: {by_proto}")
        n_ci = sum(1 for r in new_rows if r.get("ci_frozen"))
        lines.append(f"    with ci_frozen: {n_ci}/{len(new_rows)}")
        if n_ci != len(new_rows):
            ok = False
            lines.append(
                "    MISSING ci_frozen on some arm12 cells — the figure's error bars "
                "are built from ci_frozen, so those bars cannot be drawn"
            )
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=["evil", "sycophancy", "hallucination"])
    ap.add_argument("--committed-ref", default="5aae0a472b")
    ap.add_argument("--committed-root", default="eval_results/issue_1739/r2v2_fits")
    ap.add_argument(
        "--new-root", type=Path, default=Path("eval_results/issue_1739/r2v2_fits_arm12")
    )
    args = ap.parse_args(argv)

    overall = True
    any_checked = False
    for b in args.behaviors:
        new_path = args.new_root / b / "all_arms_spearman.json"
        print(f"=== {b} ===")
        if not new_path.exists():
            print(f"  SKIP — no re-score yet at {new_path}")
            continue
        any_checked = True
        old = _read_committed(
            args.committed_ref, f"{args.committed_root}/{b}/all_arms_spearman.json"
        )
        new = json.loads(new_path.read_text())
        print(f"  committed arms: {old['meta'].get('arms')}")
        print(f"  new arms:       {new['meta'].get('arms')}")
        ok, lines = _cmp_behavior(b, old, new)
        print("\n".join(lines))
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        overall = overall and ok

    if not any_checked:
        print("\nNOTHING CHECKED — no re-scored behaviour found; not a PASS.")
        return 2
    print(f"\narm12 reproduction gate: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
