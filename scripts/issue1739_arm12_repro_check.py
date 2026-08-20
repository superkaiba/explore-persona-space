#!/usr/bin/env python3
"""Reproduction gate for the #1739 arm12 re-score + the claim4-controls round.

MODE arm12 (default, byte-identical to the original): the `fits-arm12` leg
re-runs the SAME P-A/P-B fits as the committed r2v2 round with the roster
extended by one arm, into its own out root. The five arms it shares with the
committed round are therefore a free reproduction check: they should come
back identical. This script is the gate that has to PASS before an arm12 bar
is drawn on any figure.

MODE claim4 (plan v21 §4 P0.4, gate 1): joins the claim4-controls round's
seed-0 rows — SUBSET to ``map_variant == "true"`` and ``protocol == "P-B"``
first, with join UNIQUENESS asserted on KEY_FIELDS (the two variant passes
share the tuple otherwise) — against the banked rows at the pinned commit.
Report-tolerance 1e-9 (cells listed), HALT threshold |Δρ| > 1e-3 (FAIL — the
pod's seed chain stops). The report is keyed by the RUNNING code's commit SHA
(``git rev-parse HEAD`` at run time) and written under
``<new-root>/repro_claim4/`` — the dual pre/post-merge protocol runs this
twice (once at the recorded pre-merge branch tip, once post-merge) and the
SHA-keyed filenames keep both records.

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
import time
from pathlib import Path

SHARED_ARMS = (
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm11_oracle_proj",
)
NEW_ARM = "arm12_oracle_reg"
# claim4 mode: the two arms the round ADDS on the true-map pass, and the
# shufpair-pass roster whose presence proves the control actually ran.
CLAIM4_NEW_ARMS = ("arm2_ctx_native", "arm20_shuffled_map_ridge")
CLAIM4_SHUF_ARMS = ("arm4_ridge_ctx", "arm6_map_proj_e1", "arm7_map_ridge_pred")
# Identity of one scored cell. `fit` distinguishes the P-B holdout fits from
# each other and from P-A; without it every holdout collapses onto one key.
KEY_FIELDS = ("protocol", "fit", "eval_rung", "arm", "variant", "regime")
# Fields that must come back EXACTLY equal for a shared arm.
EXACT_FIELDS = ("n_eval", "n_readout", "layer")
# rho/CI are floats from a deterministic seeded path, so bit-equality is the
# expectation; the tolerance exists to separate "bit-identical" from
# "materially different" in the report rather than to license drift.
RHO_TOL = 1e-12
# claim4 tolerances (plan §7 gate 1): bit-equality expected; report anything
# above 1e-9; HALT the seed chain above 1e-3 (env/lineage defect, not noise).
CLAIM4_REPORT_TOL = 1e-9
CLAIM4_HALT_TOL = 1e-3


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


def _subset_claim4(rows: list[dict]) -> list[dict]:
    """The claim4 join subset: P-B rows of the TRUE-map variant pass only.

    Subsetting BEFORE indexing is load-bearing: the true and shufpair passes
    deliberately share the (protocol, fit, eval_rung, arm, variant, regime)
    tuple — without the subset the KEY_FIELDS join key is NON-unique and
    ``_index`` fails loud (the consistency WARN-B this mode was revised for).
    """
    return [r for r in rows if r.get("protocol") == "P-B" and r.get("map_variant") == "true"]


def _cmp_behavior_claim4(behavior: str, old: dict, new: dict) -> tuple[bool, list[str], dict]:
    """claim4 gate-1 comparison: banked P-B rows vs this round's seed-0 true rows."""
    lines: list[str] = []
    stats: dict = {"behavior": behavior}
    ok = True
    o_rows = [r for r in old["transfer_rows"] if r.get("protocol") == "P-B"]
    n_all = new["transfer_rows"]
    n_true = _subset_claim4(n_all)
    seeds = {r.get("seed") for r in n_true}
    if seeds - {0}:
        lines.append(f"  NOT a seed-0 file: seeds {sorted(seeds)} — wrong input")
        stats["error"] = f"non-seed-0 rows: {sorted(map(str, seeds))}"
        return False, lines, stats

    # join-uniqueness assert rides _index (raises on a duplicate key); the
    # subset above is what makes the tuple unique across variant passes.
    o_idx = _index(o_rows, set(SHARED_ARMS))
    n_idx = _index(n_true, set(SHARED_ARMS))
    only_old, only_new = sorted(set(o_idx) - set(n_idx)), sorted(set(n_idx) - set(o_idx))
    lines.append(f"  P-B shared-arm cells: banked {len(o_idx)}, claim4 seed-0 true {len(n_idx)}")
    stats["n_banked"], stats["n_new"] = len(o_idx), len(n_idx)
    if only_old or only_new:
        ok = False
        lines.append(f"  ROW-SET DRIFT: {len(only_old)} only-banked, {len(only_new)} only-new")
        for k in only_old[:5]:
            lines.append(f"    only banked: {k}")
        for k in only_new[:5]:
            lines.append(f"    only new:    {k}")
    joined = sorted(set(o_idx) & set(n_idx))
    stats["n_joined"] = len(joined)
    if not joined:
        lines.append("  0 cells joined — NOTHING was compared; not a reproduction")
        return False, lines, stats

    max_drho, worst, n_exact_bad = 0.0, None, 0
    over_report: list[tuple] = []
    for k in joined:
        o, n = o_idx[k], n_idx[k]
        for f in EXACT_FIELDS:
            if o.get(f) != n.get(f):
                ok = False
                n_exact_bad += 1
                if n_exact_bad <= 5:
                    lines.append(f"  {f} MISMATCH {k}: banked {o.get(f)} vs new {n.get(f)}")
        ro, rn = o.get("rho_frozen"), n.get("rho_frozen")
        if ro is None or rn is None:
            if ro is not rn:
                ok = False
                lines.append(f"  rho presence MISMATCH {k}: {ro} vs {rn}")
            continue
        d = abs(float(ro) - float(rn))
        if d > CLAIM4_REPORT_TOL:
            over_report.append((k, float(ro), float(rn), d))
        if d > max_drho:
            max_drho, worst = d, k
    stats["max_drho"] = max_drho
    stats["n_exact_field_mismatch"] = n_exact_bad
    stats["cells_over_report_tol"] = [
        {"key": list(k), "banked": ro, "new": rn, "abs_drho": d}
        for k, ro, rn, d in over_report[:20]
    ]
    lines.append(
        f"  max |delta rho_frozen| over joined cells: {max_drho:.3e}"
        + (f"  (worst {worst})" if worst else "")
    )
    if over_report:
        lines.append(
            f"  {len(over_report)} cell(s) above report tol {CLAIM4_REPORT_TOL:.0e} "
            f"(halt tol {CLAIM4_HALT_TOL:.0e})"
        )
    if max_drho > CLAIM4_HALT_TOL:
        ok = False
        lines.append(
            f"  RHO DRIFT above HALT threshold {CLAIM4_HALT_TOL:.0e} — the pinned "
            "pipeline did NOT reproduce; halt the seed chain (plan §7 kill (a))"
        )

    # presence checks: the round's new true-pass arms + the shufpair pass.
    for arm in CLAIM4_NEW_ARMS:
        n_arm = sum(1 for r in n_all if r.get("arm") == arm and r.get("map_variant") == "true")
        stats[f"n_{arm}_true_rows"] = n_arm
        lines.append(f"  {arm} true-pass rows: {n_arm}")
        if not n_arm:
            ok = False
            lines.append(f"  {arm} ABSENT from the true pass — the leg did not add the arm")
    n_shuf = sum(1 for r in n_all if r.get("map_variant") == "shufpair")
    shuf_arms = sorted({r.get("arm") for r in n_all if r.get("map_variant") == "shufpair"})
    stats["n_shufpair_rows"], stats["shufpair_arms"] = n_shuf, shuf_arms
    lines.append(f"  shufpair-pass rows: {n_shuf} (arms {shuf_arms})")
    if not n_shuf:
        ok = False
        lines.append("  shufpair pass ABSENT — the pairing-shuffle control did not run")
    elif set(shuf_arms) - set(CLAIM4_SHUF_ARMS):
        ok = False
        lines.append(
            f"  shufpair pass carries arms outside the registered roster {sorted(CLAIM4_SHUF_ARMS)}"
        )
    stats["ok"] = ok
    return ok, lines, stats


def _git_head_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"[repro] cannot resolve HEAD: {out.stderr[:200]}")
    return out.stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["arm12", "claim4"], default="arm12")
    ap.add_argument("--behaviors", nargs="+", default=["evil", "sycophancy", "hallucination"])
    ap.add_argument("--committed-ref", default="5aae0a472b")
    ap.add_argument("--committed-root", default="eval_results/issue_1739/r2v2_fits")
    ap.add_argument(
        "--new-root",
        type=Path,
        default=None,
        help="re-score root (default: the mode's own out root — arm12: "
        "eval_results/issue_1739/r2v2_fits_arm12; claim4: "
        "eval_results/issue_1739/claim4_controls)",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="claim4 mode: report JSON path (default "
        "<new-root>/repro_claim4/report_<HEAD-sha12>.json — SHA-keyed so the dual "
        "pre/post-merge runs never overwrite each other)",
    )
    args = ap.parse_args(argv)
    if args.new_root is None:
        args.new_root = Path(
            "eval_results/issue_1739/r2v2_fits_arm12"
            if args.mode == "arm12"
            else "eval_results/issue_1739/claim4_controls"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    commit = _git_head_sha() if args.mode == "claim4" else None
    report: dict = {
        "mode": args.mode,
        "git_commit": commit,
        "committed_ref": args.committed_ref,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "report_tol": CLAIM4_REPORT_TOL,
        "halt_tol": CLAIM4_HALT_TOL,
        "behaviors": {},
    }
    overall = True
    any_checked = False
    for b in args.behaviors:
        rel = (
            f"{b}/seed0/all_arms_spearman.json"
            if args.mode == "claim4"
            else (f"{b}/all_arms_spearman.json")
        )
        new_path = args.new_root / rel
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
        if args.mode == "claim4":
            ok, lines, stats = _cmp_behavior_claim4(b, old, new)
            report["behaviors"][b] = stats
        else:
            ok, lines = _cmp_behavior(b, old, new)
        print("\n".join(lines))
        print(f"  -> {'PASS' if ok else 'FAIL'}")
        overall = overall and ok

    if not any_checked:
        print("\nNOTHING CHECKED — no re-scored behaviour found; not a PASS.")
        return 2
    if args.mode == "claim4":
        report["overall"] = "PASS" if overall else "FAIL"
        out = args.report_out or (
            args.new_root / "repro_claim4" / f"report_{(commit or 'unknown')[:12]}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1))
        print(f"\nreport (keyed by running commit {commit}): {out}")
    print(f"\n{args.mode} reproduction gate: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
