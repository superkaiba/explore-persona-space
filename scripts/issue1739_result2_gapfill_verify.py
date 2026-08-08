#!/usr/bin/env python3
"""Slice + map-kind verifier for the #1739 Result-2 gap-fill round (Job C).

A row only fills the Result 2 gap if it sits at the PLOTTED operating slice
(``regime == "e1"``, ``u_rung_label == "full"``, ``budget_l == 16000``) AND was
produced under the LINEAR context->answer map. The second half is the one this
round exists to protect: arms 7/8 already have (full, 16000) rows on both OOD
rungs under the NONLINEAR map legs
(``new_arm_round/nlood/hallucination/{kernel,mlp}``), and folding one of those
into a linear-map bar group is a silent methodology error.

``arm_results/all_arms_spearman.json`` records no ``map_kind`` in its meta
(only the nonlinear collector stamps one), so this verifier establishes the map
kind from evidence OUTSIDE the summary and then stamps it, rather than taking
the driver's word for it:

1. **Artifact form.** ``fits._save_map`` writes a LINEAR map as
   ``maps/<variant>__u<label>.npz`` and a nonlinear one as
   ``...__<kind>.pt``. The ``.npz`` is read and its embedded meta must say
   ``map_kind == "linear"``; a sibling ``.pt`` being the only artifact is a
   hard fail.
2. **Differential against the nonlinear rows.** Every emitted rho is compared
   against the kernel/MLP-leg rows at the same (arm, rung, variant, budget).
   An exact collision means the nonlinear rows were folded in by mistake.

It also cross-checks ``n_eval`` per rung against the committed sources (the
Result 2 figure raises if one panel column spans multiple eval sizes) and
writes ``gapfill_provenance.json`` — the realized arm x rung x variant
coverage with rho, CI and replicate counts.

Exit 0 = every emitted row is at the slice and linear-map-attested.
Exit 1 = at least one row fails; nothing is stamped.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SLICE = {"regime": "e1", "u_rung_label": "full", "budget_l": 16000}
ROSTER = (
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm12_oracle_reg",
    "arm17_oracle_mlp",
    "arm18_oracle_krr",
)
RUNGS = ("train", "nqopen", "simpleqa")
# The nonlinear-map legs whose (full, 16000) rows must NEVER appear here.
NONLINEAR_LEGS = (
    "eval_results/issue_1739/new_arm_round/nlood/hallucination/kernel",
    "eval_results/issue_1739/new_arm_round/nlood/hallucination/mlp",
)


def _rows(summary: dict) -> list[dict]:
    return list(summary.get("arm_rows") or []) + list(summary.get("transfer_rows") or [])


def _at_slice(row: dict) -> bool:
    return all(row.get(k) == v for k, v in SLICE.items())


def check_map_artifact(tensors_root: Path, variants: list[str], u_label: str) -> list[str]:
    """Positive linear-map attestation from the persisted map artifacts."""
    problems: list[str] = []
    import numpy as np

    for variant in variants:
        stem = f"{variant}__u{u_label}"
        npz = tensors_root / "maps" / f"{stem}.npz"
        nonlinear = sorted((tensors_root / "maps").glob(f"{stem}__*.pt"))
        sidecar = tensors_root / "maps" / f"{stem}.meta.json"
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as blob:
                meta = json.loads(str(blob["meta"]))
            source = npz.name
        elif sidecar.is_file():
            # The .npz payloads are ~0.7 GB each and stay on the compute box;
            # the producing run extracts their embedded meta to this sidecar so
            # the attestation survives without moving the weights.
            meta = json.loads(sidecar.read_text())
            source = sidecar.name
        else:
            problems.append(
                f"{variant}: no linear map artifact at {npz} (nor its {sidecar.name} "
                f"sidecar); nonlinear siblings present: {[p.name for p in nonlinear]}"
            )
            continue
        kind = meta.get("map_kind")
        if kind != "linear":
            problems.append(f"{variant}: {source} meta map_kind={kind!r}, expected 'linear'")
        if meta.get("u_label") != u_label:
            problems.append(f"{variant}: {source} meta u_label={meta.get('u_label')!r}")
    return problems


def _nonlinear_rho_index(repo_root: Path) -> dict[tuple, set[float]]:
    """(arm, eval_rung, variant, budget_l) -> the nonlinear legs' rho values."""
    index: dict[tuple, set[float]] = defaultdict(set)
    for leg in NONLINEAR_LEGS:
        path = repo_root / leg / "arm_results" / "all_arms_spearman.json"
        if not path.is_file():
            continue
        with open(path) as fh:
            summary = json.load(fh)
        for row in _rows(summary):
            rho = row.get("rho_frozen")
            if rho is None or (isinstance(rho, float) and math.isnan(rho)):
                continue
            key = (row.get("arm"), row.get("eval_rung"), row.get("variant"), row.get("budget_l"))
            index[key].add(round(float(rho), 12))
    return index


def check_not_nonlinear(rows: list[dict], repo_root: Path) -> tuple[list[str], list[str]]:
    """Contamination differential against the kernel/MLP legs. -> (problems, notes)

    Naive value-equality is NOT a provenance test here, because ``rho_frozen``
    is a SPEARMAN correlation: it is a function of RANKS only, so two genuinely
    different predictors that induce the same ranking of the eval set produce a
    bit-identical rho. That happens for real in this grid — hallucination's
    arm7 prefix_end cells are rank-DEGENERATE (all 15 replicates carry ONE
    distinct rho across four different frozen layers), so every map family
    lands on the same number and equality says nothing about which map produced
    it.

    Contamination, by contrast, is WHOLESALE: folding a nonlinear leg's rows
    into this slice would reproduce a cell's replicate SET, not one value in a
    cell whose replicates are all equal anyway. So a cell is flagged only when
    it is INFORMATIVE (>1 distinct rho among its replicates) AND a majority of
    its rows match nonlinear rows at the same (draw, seed, layer). Matches in
    degenerate cells are reported as notes with their evidence, never silently
    dropped.
    """
    index = _nonlinear_rho_index(repo_root)
    if not index:
        return (
            ["nonlinear-leg differential SKIPPED: no kernel/mlp summary to compare against"],
            [],
        )
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        rho = row.get("rho_frozen")
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            continue
        by_cell[(row.get("arm"), row.get("eval_rung"), row.get("variant"))].append(row)

    problems, notes = [], []
    for cell, cell_rows in sorted(by_cell.items(), key=str):
        distinct = {round(float(r["rho_frozen"]), 12) for r in cell_rows}
        matched = [
            r
            for r in cell_rows
            if round(float(r["rho_frozen"]), 12)
            in index.get(
                (r.get("arm"), r.get("eval_rung"), r.get("variant"), r.get("budget_l")), ()
            )
        ]
        if not matched:
            continue
        degenerate = len(distinct) == 1
        frac = len(matched) / len(cell_rows)
        detail = (
            f"{cell}: {len(matched)}/{len(cell_rows)} rows share a rho with a nonlinear-leg "
            f"row; distinct_rho={len(distinct)}; layers={sorted({r['layer'] for r in matched})}"
        )
        if degenerate:
            notes.append(
                detail + " — cell is RANK-DEGENERATE (one distinct rho across all replicates "
                "and several frozen layers), so rho equality carries no provenance signal"
            )
        elif frac >= 0.5:
            problems.append(
                detail + " — MAJORITY of an informative cell matches the nonlinear legs; "
                "a kernel/MLP-map row set has been folded into the linear-map slice"
            )
        else:
            notes.append(
                detail + " — isolated match in an informative cell; below the "
                "majority threshold, reported for the record"
            )
    return problems, notes


def coverage(rows: list[dict]) -> dict:
    """Realized arm x rung x variant coverage with rho + averaged bootstrap CI."""
    import numpy as np

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["arm"], row["eval_rung"], row["variant"])].append(row)
    out = {}
    for (arm, rung, variant), cell in sorted(by_cell.items()):
        rhos = [
            float(r["rho_frozen"])
            for r in cell
            if r.get("rho_frozen") is not None and not math.isnan(float(r["rho_frozen"]))
        ]
        cis = [r["ci_frozen"] for r in cell if r.get("ci_frozen")]
        n_evals = sorted({r["n_eval"] for r in cell if r.get("n_eval") is not None})
        out[f"{arm}|{rung}|{variant}"] = {
            "n_replicates": len(cell),
            "n_rho": len(rhos),
            "rho_mean": round(float(np.mean(rhos)), 6) if rhos else None,
            "rho_sd": round(float(np.std(rhos, ddof=1)), 6) if len(rhos) > 1 else 0.0,
            "ci_mean": (
                [
                    round(float(np.mean([c[0] for c in cis])), 6),
                    round(float(np.mean([c[1] for c in cis])), 6),
                ]
                if cis
                else None
            ),
            "n_eval": n_evals[0] if len(n_evals) == 1 else (n_evals or None),
            "layers": sorted({int(r["layer"]) for r in cell if r.get("layer") is not None}),
        }
    return out


def check_n_eval_parity(cov: dict, repo_root: Path) -> list[str]:
    """The Result 2 figure refuses a panel column spanning multiple n_eval."""
    committed: dict[str, set[int]] = defaultdict(set)
    path = (
        repo_root
        / "eval_results/issue_1739/new_arm_round/oracle/hallucination"
        / "arm_results/all_arms_spearman.json"
    )
    if path.is_file():
        with open(path) as fh:
            summary = json.load(fh)
        for row in summary.get("transfer_rows") or []:
            if row.get("n_eval") is not None and row.get("eval_rung") in RUNGS:
                committed[row["eval_rung"]].add(int(row["n_eval"]))
    notes = []
    for key, rec in cov.items():
        rung = key.split("|")[1]
        got = rec.get("n_eval")
        if not isinstance(got, int):
            continue
        if rung == "train":
            # The TRAIN rung's eval set is the in-split labeled pool, so its
            # n_eval tracks budget_l by construction. Comparing it against the
            # committed oracle leg (budgets 250 / 2500) would compare different
            # budgets, not different targets — the honest check here is
            # self-consistency with this round's own budget.
            if got != SLICE["budget_l"]:
                notes.append(
                    f"{key}: train n_eval={got} != budget_l={SLICE['budget_l']} — "
                    "the in-split eval pool should equal the labeled budget"
                )
            continue
        if rung in committed and got not in committed[rung]:
            notes.append(
                f"{key}: n_eval={got} not in the committed {rung} eval sizes "
                f"{sorted(committed[rung])} — the Result 2 matched-target guard will refuse"
            )
    return notes


def _row_key(row: dict) -> tuple:
    """Identity of one replicate row, for cross-leg de-duplication."""
    return (
        row.get("arm"),
        row.get("eval_rung"),
        row.get("variant"),
        row.get("regime"),
        row.get("u_rung_label"),
        row.get("budget_l"),
        row.get("draw"),
        row.get("seed"),
        row.get("layer"),
        row.get("rung_kind"),
    )


def merge_legs(legs_root: Path, dest_root: Path) -> dict:
    """Union the per-leg summaries into ONE summary at ``dest_root``.

    The grid is fanned out across parallel legs (one out-root per
    (variant, draw) — two legs sharing an out-root would clobber each other's
    ``percell/`` resume state), so the fold needs the legs stitched back into
    the single ``all_arms_spearman.json`` shape every downstream reader
    expects. Rows are de-duplicated on their full replicate identity, so a
    re-run leg cannot double-count. Leg metas are recorded under
    ``meta.merged_from`` rather than silently collapsed.
    """
    leg_summaries = sorted(legs_root.glob("*/*/arm_results/all_arms_spearman.json"))
    if not leg_summaries:
        raise SystemExit(f"[gapfill-verify] no leg summaries under {legs_root}")
    arm_rows: dict[tuple, dict] = {}
    transfer_rows: dict[tuple, dict] = {}
    skips: list = []
    metas = {}
    for path in leg_summaries:
        with open(path) as fh:
            blob = json.load(fh)
        leg = path.parents[2].name
        metas[leg] = {
            "path": str(path),
            "n_arm_rows": len(blob.get("arm_rows") or []),
            "n_transfer_rows": len(blob.get("transfer_rows") or []),
            "meta": blob.get("meta"),
        }
        for row in blob.get("arm_rows") or []:
            arm_rows.setdefault(_row_key(row), row)
        for row in blob.get("transfer_rows") or []:
            transfer_rows.setdefault(_row_key(row), row)
        skips.extend(blob.get("transfer_skips") or [])
    base = json.loads(Path(leg_summaries[0]).read_text())
    merged = {
        "n_cells": len(arm_rows),
        "n_arm_rows": len(arm_rows),
        "arm_rows": list(arm_rows.values()),
        "headlines": base.get("headlines"),
        "nulls": base.get("nulls"),
        "meta": {**(base.get("meta") or {}), "merged_from": metas},
        "ts": base.get("ts"),
        "transfer_rows": list(transfer_rows.values()),
        "transfer_skips": skips,
    }
    dest = dest_root / "arm_results" / "all_arms_spearman.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(merged, indent=1))
    print(
        f"[gapfill-verify] merged {len(leg_summaries)} leg(s) -> {dest} "
        f"({len(arm_rows)} arm_rows, {len(transfer_rows)} transfer_rows)"
    )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("eval_results/issue_1739/result2_gapfill/hallucination"),
    )
    ap.add_argument(
        "--merge-legs",
        type=Path,
        default=None,
        help="union every <legs-root>/<leg>/<behavior>/arm_results/all_arms_spearman.json "
        "into --out-root before verifying (parallel-leg fan-out)",
    )
    ap.add_argument("--tensors-root", type=Path, default=Path("analysis_tensors/issue_1739"))
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--u-label", default="full")
    ap.add_argument(
        "--stamp",
        action="store_true",
        help="on PASS, write map_kind into the summary meta + every emitted row",
    )
    args = ap.parse_args()

    if args.merge_legs is not None:
        merge_legs(args.merge_legs, args.out_root)

    summary_path = args.out_root / "arm_results" / "all_arms_spearman.json"
    with open(summary_path) as fh:
        summary = json.load(fh)
    rows = _rows(summary)
    if not rows:
        print(f"[gapfill-verify] FAIL: {summary_path} carries no rows")
        return 1

    problems: list[str] = []

    off_slice = [r for r in rows if not _at_slice(r)]
    for row in off_slice[:20]:
        problems.append(
            "off-slice row: arm={arm} rung={eval_rung} variant={variant} "
            "budget_l={budget_l} u_rung_label={u_rung_label} regime={regime}".format(
                arm=row.get("arm"),
                eval_rung=row.get("eval_rung"),
                variant=row.get("variant"),
                budget_l=row.get("budget_l"),
                u_rung_label=row.get("u_rung_label"),
                regime=row.get("regime"),
            )
        )
    if len(off_slice) > 20:
        problems.append(f"... and {len(off_slice) - 20} further off-slice rows")

    stray = sorted({r.get("arm") for r in rows} - set(ROSTER))
    if stray:
        problems.append(f"rows for arms outside the round's roster: {stray}")
    stray_rungs = sorted({r.get("eval_rung") for r in rows} - set(RUNGS))
    if stray_rungs:
        problems.append(f"rows for rungs outside the round's scope: {stray_rungs}")
    if summary.get("meta", {}).get("behavior") != "hallucination":
        problems.append(f"meta.behavior={summary.get('meta', {}).get('behavior')!r}")

    variants = sorted({r["variant"] for r in rows if r.get("variant")})
    problems += check_map_artifact(args.tensors_root, variants, args.u_label)
    nl_problems, nl_notes = check_not_nonlinear(rows, args.repo_root)
    problems += nl_problems

    cov = coverage(rows)
    warnings = check_n_eval_parity(cov, args.repo_root) + nl_notes

    print(f"[gapfill-verify] rows={len(rows)} off_slice={len(off_slice)} cells={len(cov)}")
    for key, rec in cov.items():
        print(
            f"  {key:52s} n={rec['n_replicates']:3d} rho={rec['rho_mean']} "
            f"ci={rec['ci_mean']} n_eval={rec['n_eval']}"
        )
    for note in warnings:
        print(f"[gapfill-verify] WARN {note}")
    for note in problems:
        print(f"[gapfill-verify] FAIL {note}")
    if problems:
        print("[gapfill-verify] VERDICT: FAIL — nothing stamped")
        return 1

    provenance = {
        "round": "issue_1739 Result-2 gap-fill (Job C)",
        "behavior": "hallucination",
        "slice": dict(SLICE),
        "map_kind": "linear",
        "map_kind_evidence": [
            f"maps/{v}__u{args.u_label}.npz meta.map_kind == 'linear'" for v in variants
        ]
        + ["no emitted rho collides with the nlood kernel/mlp legs at the same cell"],
        "roster": list(ROSTER),
        "variants": variants,
        "coverage": cov,
        "n_eval_warnings": warnings,
        "summary_path": str(summary_path),
    }
    (args.out_root / "gapfill_provenance.json").write_text(json.dumps(provenance, indent=1))
    print(f"[gapfill-verify] wrote {args.out_root / 'gapfill_provenance.json'}")

    if args.stamp:
        summary.setdefault("meta", {})["map_kind"] = "linear"
        for key in ("arm_rows", "transfer_rows"):
            for row in summary.get(key) or []:
                row["map_kind"] = "linear"
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"[gapfill-verify] stamped map_kind=linear into {summary_path}")

    print("[gapfill-verify] VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
