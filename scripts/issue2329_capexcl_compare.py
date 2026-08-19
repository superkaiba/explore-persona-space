#!/usr/bin/env python
"""Issue #2329 v55 restriction analysis: shipped vs cap-hit-excluded primary DVs.

Adjudicates the escalation trigger pre-registered in the #2329 ``epm:progress``
v55 note (2026-08-17T12:03Z) and restated in the v97 dispatch note:

    escalate to the 8192 re-generation round IFF the restriction analysis moves
    any primary DV's conclusion -- a sign flip, a gate verdict change, or a
    conclusion-bearing effect moving outside its CI.

The pre-registered criteria, evaluated here verbatim:

  (a) SIGN FLIP -- ``sign(f_steered_mean)`` changes on any testable cell.
  (b) GATE VERDICT CHANGE -- ``disjoint_both_nulls`` or ``holm_pass`` flips.
  (c) EFFECT LEAVES ITS CI -- the RESTRICTED ``f_steered_mean`` falls outside
      the SHIPPED ``ci95['steered']`` for that cell. (Asymmetric by design: the
      shipped interval is the published claim, so it is the reference the
      restricted point estimate is judged against -- not the reverse.)
  (d) TRANSFER HEADLINE MOVES -- restricted Spearman rho leaves the shipped
      pair-clustered ``ci95_pair_clustered``.
  (e) TWO-BY-TWO QUADRANT COUNTS CHANGE -- the joint
      (causal_verdict, probe_verdict) distribution over cells changes.

Fail-loud throughout: a missing artifact, a missing key, or a cell present in
one side and absent in the other RAISES rather than being silently skipped --
a restriction analysis that quietly drops the cells it could not compare would
manufacture a "no change" verdict, which is the one failure mode that matters
here.

Usage:
    uv run python scripts/issue2329_capexcl_compare.py \\
        --shipped eval_results/issue_2329/f_metrics \\
        --restricted eval_results/issue_2329/f_metrics_capexcl \\
        --out eval_results/issue_2329/cap_hit/restriction_analysis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("issue2329.capexcl")

# Per-cell fields whose movement is reported (superset of the trigger fields).
ARM_MEANS = ("f_steered_mean", "f_shuffled_mean", "f_crosstype_mean")
VERDICT_FIELDS = ("disjoint_both_nulls", "holm_pass", "untestable_causal", "low_coherence")


def _load_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"required artifact missing: {p}")
    with p.open() as fh:
        return json.load(fh)


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        raise FileNotFoundError(f"required artifact missing: {p}")
    rows = []
    with p.open() as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    if not rows:
        raise ValueError(f"artifact is empty (refusing to treat as 'no data'): {p}")
    return rows


def _sign(x: float | None) -> int | None:
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def compare_per_cell(shipped: dict, restricted: dict) -> dict:
    """Per-cell comparison + triggers (a), (b), (c)."""
    s_cells, r_cells = shipped["per_cell"], restricted["per_cell"]
    if set(s_cells) != set(r_cells):
        only_s = sorted(set(s_cells) - set(r_cells))
        only_r = sorted(set(r_cells) - set(s_cells))
        raise KeyError(
            "per_cell unit sets differ between shipped and restricted -- refusing to "
            f"compare a partial overlap. shipped-only={only_s} restricted-only={only_r}"
        )

    rows: list[dict] = []
    sign_flips: list[dict] = []
    verdict_changes: list[dict] = []
    ci_exits: list[dict] = []

    for key in sorted(s_cells):
        s, r = s_cells[key], r_cells[key]
        row: dict = {"unit": key, "cell": s["cell"], "slot": s["slot"], "family": s.get("family")}

        for f in ARM_MEANS:
            if f not in s or f not in r:
                raise KeyError(f"{key}: field {f!r} missing (shipped={f in s} restricted={f in r})")
            row[f"shipped_{f}"] = s[f]
            row[f"restricted_{f}"] = r[f]
            if s[f] is not None and r[f] is not None:
                row[f"delta_{f}"] = r[f] - s[f]
            else:
                row[f"delta_{f}"] = None

        row["shipped_n_post_exclusion"] = s.get("n_post_exclusion")
        row["restricted_n_post_exclusion"] = r.get("n_post_exclusion")

        # (a) sign flip on the primary causal DV, testable cells only.
        testable = not bool(s.get("untestable_causal"))
        ss, rs = _sign(s["f_steered_mean"]), _sign(r["f_steered_mean"])
        row["shipped_sign"], row["restricted_sign"] = ss, rs

        # CONCLUSION-BEARING classification (the (a') refinement -- see the
        # module docstring). A cell is conclusion-bearing iff the SHIPPED
        # tables assert something directional about it: it clears the
        # registered gates, or its steered CI excludes zero. The sign of a
        # NULL cell's point estimate is not a conclusion -- it is a coin-flip
        # about which side of zero sampling noise landed on -- so a sign flip
        # there cannot "move a conclusion" no matter how the trigger is worded.
        s_ci = (s.get("ci95") or {}).get("steered")
        ci_excludes_zero = None
        if s_ci is not None and len(s_ci) == 2 and None not in s_ci:
            ci_excludes_zero = bool(float(s_ci[0]) > 0.0 or float(s_ci[1]) < 0.0)
        conclusion_bearing = bool(
            s.get("holm_pass") is True
            or s.get("disjoint_both_nulls") is True
            or ci_excludes_zero is True
        )
        row["shipped_ci_excludes_zero"] = ci_excludes_zero
        row["conclusion_bearing"] = conclusion_bearing

        if testable and ss is not None and rs is not None and ss != rs and 0 not in (ss, rs):
            sign_flips.append(
                {
                    "unit": key,
                    "shipped": s["f_steered_mean"],
                    "restricted": r["f_steered_mean"],
                    "conclusion_bearing": conclusion_bearing,
                    "shipped_ci95_steered": s_ci,
                    "shipped_ci_excludes_zero": ci_excludes_zero,
                    "shipped_holm_pass": s.get("holm_pass"),
                    "shipped_disjoint_both_nulls": s.get("disjoint_both_nulls"),
                    "shipped_p_iut": s.get("p_iut"),
                    "shipped_causal_verdict_is_null": (
                        s.get("holm_pass") is not True and s.get("disjoint_both_nulls") is not True
                    ),
                    "realized_mde_single_test": s.get("realized_mde_single_test"),
                    "n_post_exclusion_shipped": s.get("n_post_exclusion"),
                    "n_post_exclusion_restricted": r.get("n_post_exclusion"),
                }
            )

        # (b) gate verdict changes.
        for f in VERDICT_FIELDS:
            if s.get(f) != r.get(f):
                verdict_changes.append(
                    {"unit": key, "field": f, "shipped": s.get(f), "restricted": r.get(f)}
                )
                row[f"changed_{f}"] = True

        # (c) restricted point estimate outside the SHIPPED CI.
        ci = (s.get("ci95") or {}).get("steered")
        rv = r["f_steered_mean"]
        row["shipped_ci95_steered"] = ci
        if ci is not None and rv is not None and len(ci) == 2 and None not in ci:
            lo, hi = float(ci[0]), float(ci[1])
            outside = bool(rv < lo or rv > hi)
            row["restricted_outside_shipped_ci"] = outside
            if outside:
                ci_exits.append(
                    {
                        "unit": key,
                        "restricted_f_steered_mean": rv,
                        "shipped_ci95_steered": [lo, hi],
                        "excess": (lo - rv) if rv < lo else (rv - hi),
                    }
                )
        else:
            row["restricted_outside_shipped_ci"] = None

        rows.append(row)

    deltas = [abs(x["delta_f_steered_mean"]) for x in rows if x["delta_f_steered_mean"] is not None]
    summary = {
        "n_units": len(rows),
        "n_units_with_comparable_steered": len(deltas),
        "max_abs_delta_f_steered_mean": max(deltas) if deltas else None,
        "mean_abs_delta_f_steered_mean": (sum(deltas) / len(deltas)) if deltas else None,
    }
    return {
        "rows": rows,
        "summary": summary,
        "sign_flips": sign_flips,
        "verdict_changes": verdict_changes,
        "ci_exits": ci_exits,
    }


def compare_transfer(shipped: dict, restricted: dict) -> dict:
    """Trigger (d): the transfer headline."""
    out = {
        "shipped_rho": shipped["rho"],
        "restricted_rho": restricted["rho"],
        "delta_rho": restricted["rho"] - shipped["rho"],
        "shipped_p": shipped.get("p"),
        "restricted_p": restricted.get("p"),
        "shipped_ci95_pair_clustered": shipped.get("ci95_pair_clustered"),
        "restricted_ci95_pair_clustered": restricted.get("ci95_pair_clustered"),
        "shipped_n_shared_p1_units": shipped.get("n_shared_p1_units"),
        "restricted_n_shared_p1_units": restricted.get("n_shared_p1_units"),
    }
    ci = shipped.get("ci95_pair_clustered")
    rho = restricted["rho"]
    if ci and len(ci) == 2 and None not in ci and rho is not None:
        lo, hi = float(ci[0]), float(ci[1])
        out["restricted_rho_outside_shipped_ci"] = bool(rho < lo or rho > hi)
    else:
        out["restricted_rho_outside_shipped_ci"] = None
    if out["shipped_n_shared_p1_units"] != out["restricted_n_shared_p1_units"]:
        out["shared_unit_count_changed"] = True
    return out


def _quadrant_counts(cells: list[dict]) -> dict[str, int]:
    c = Counter()
    for row in cells:
        for f in ("causal_verdict", "probe_verdict"):
            if f not in row:
                raise KeyError(f"two_by_two cell missing {f!r}: {row}")
        c[f"{row['causal_verdict']}|{row['probe_verdict']}"] += 1
    return dict(sorted(c.items()))


def compare_two_by_two(shipped: dict, restricted: dict) -> dict:
    s_counts = _quadrant_counts(shipped["cells"])
    r_counts = _quadrant_counts(restricted["cells"])
    changed = s_counts != r_counts
    per_cell_moves = []
    s_by = {(c["cell"], c["slot"]): c for c in shipped["cells"]}
    r_by = {(c["cell"], c["slot"]): c for c in restricted["cells"]}
    if set(s_by) != set(r_by):
        raise KeyError("two_by_two cell keys differ between shipped and restricted")
    for k in sorted(s_by):
        s, r = s_by[k], r_by[k]
        if (s["causal_verdict"], s["probe_verdict"]) != (r["causal_verdict"], r["probe_verdict"]):
            per_cell_moves.append(
                {
                    "unit": f"{k[0]}|{k[1]}",
                    "shipped": [s["causal_verdict"], s["probe_verdict"]],
                    "restricted": [r["causal_verdict"], r["probe_verdict"]],
                }
            )
    return {
        "shipped_counts": s_counts,
        "restricted_counts": r_counts,
        "counts_changed": changed,
        "per_cell_moves": per_cell_moves,
    }


def compare_stage2(shipped: list[dict], restricted: list[dict]) -> dict:
    """Stage-2 movement. Reported (highest cap-hit rate surface) -- the
    pre-registered triggers are (a)-(e); stage-2 movement feeds (a)/(c) through
    its own f_beh signs where those cells appear in per_cell."""
    s_by = {(r["block_key"], r["pair_id"], r["arm"]): r for r in shipped}
    r_by = {(r["block_key"], r["pair_id"], r["arm"]): r for r in restricted}
    common = sorted(set(s_by) & set(r_by))
    deltas = []
    flips = []
    for k in common:
        a, b = s_by[k].get("f_beh"), r_by[k].get("f_beh")
        if a is None or b is None:
            continue
        deltas.append(abs(b - a))
        if _sign(a) != _sign(b) and 0 not in (_sign(a), _sign(b)):
            flips.append({"key": "|".join(k), "shipped": a, "restricted": b})
    return {
        "n_shipped_rows": len(shipped),
        "n_restricted_rows": len(restricted),
        "n_common_keys": len(common),
        "n_shipped_only": len(set(s_by) - set(r_by)),
        "n_restricted_only": len(set(r_by) - set(s_by)),
        "n_comparable_f_beh": len(deltas),
        "max_abs_delta_f_beh": max(deltas) if deltas else None,
        "mean_abs_delta_f_beh": (sum(deltas) / len(deltas)) if deltas else None,
        "n_sign_flips": len(flips),
        "sign_flips": flips[:50],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shipped", type=Path, default=Path("eval_results/issue_2329/f_metrics"))
    ap.add_argument(
        "--restricted", type=Path, default=Path("eval_results/issue_2329/f_metrics_capexcl")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_2329/cap_hit/restriction_analysis.json"),
    )
    args = ap.parse_args(argv)

    per_cell = compare_per_cell(
        _load_json(args.shipped / "stats.json"), _load_json(args.restricted / "stats.json")
    )
    transfer = compare_transfer(
        _load_json(args.shipped / "transfer.json"), _load_json(args.restricted / "transfer.json")
    )
    two_by_two = compare_two_by_two(
        _load_json(args.shipped / "two_by_two.json"),
        _load_json(args.restricted / "two_by_two.json"),
    )
    stage2 = compare_stage2(
        _load_jsonl(args.shipped / "stage2_cells.jsonl"),
        _load_jsonl(args.restricted / "stage2_cells.jsonl"),
    )

    triggers = {
        "a_sign_flip": bool(per_cell["sign_flips"]),
        "b_gate_verdict_change": bool(per_cell["verdict_changes"]),
        "c_effect_outside_shipped_ci": bool(per_cell["ci_exits"]),
        "d_transfer_rho_outside_shipped_ci": bool(
            transfer.get("restricted_rho_outside_shipped_ci")
        ),
        "e_two_by_two_counts_changed": bool(two_by_two["counts_changed"]),
    }
    fired = sorted(k for k, v in triggers.items() if v)
    verdict = "ESCALATE" if fired else "NO-ESCALATION"

    # The (a') refinement, reported ALONGSIDE the verbatim pre-registered verdict
    # rather than replacing it. Limb (a) as pre-registered keys on
    # `untestable_causal == False`, which admits testable-BUT-NULL cells whose
    # point-estimate sign is noise rather than a conclusion. Both verdicts are
    # emitted so a reader sees the raw trigger AND the conclusion-bearing reading
    # without having to take either on trust.
    cb_sign_flips = [f for f in per_cell["sign_flips"] if f.get("conclusion_bearing")]
    triggers_refined = dict(triggers)
    triggers_refined["a_sign_flip_conclusion_bearing_only"] = bool(cb_sign_flips)
    triggers_refined.pop("a_sign_flip", None)
    fired_refined = sorted(k for k, v in triggers_refined.items() if v)
    verdict_refined = "ESCALATE" if fired_refined else "NO-ESCALATION"

    out = {
        "what": "issue #2329 v55 restriction analysis -- shipped vs cap-hit-excluded primary DVs",
        "pre_registered_trigger": (
            "escalate to the 8192 re-generation round iff the restriction analysis moves any "
            "primary DV's conclusion: a sign flip, a gate verdict change, or a "
            "conclusion-bearing effect moving outside its CI (epm:progress v55, 2026-08-17T12:03Z)"
        ),
        "shipped_dir": str(args.shipped),
        "restricted_dir": str(args.restricted),
        "probe_reused_verbatim": True,
        "probe_reuse_justification": (
            "step_probe reads only bank.pt per_context activation records with value-pair "
            "identity labels; it references no completion-side field (cap_hit, grid rows, "
            "behavior scores, coherence), so completion truncation cannot affect probe AUC"
        ),
        "triggers": triggers,
        "triggers_fired": fired,
        "verdict": verdict,
        "refinement_a_prime": {
            "what": (
                "limb (a) as pre-registered keys on `untestable_causal == False`, which admits "
                "testable-BUT-NULL cells; the sign of a null cell's point estimate is sampling "
                "noise, not a conclusion. (a') restricts the sign-flip limb to CONCLUSION-BEARING "
                "cells: shipped holm_pass, shipped disjoint_both_nulls, or a shipped steered CI "
                "that excludes zero."
            ),
            "disclosure": (
                "This refinement was authored AFTER seeing which cell fired, so it is reported "
                "ALONGSIDE the verbatim pre-registered verdict and never substituted for it. The "
                "raw verdict above stands as the pre-registered result; the operator's escalation "
                "decision and its reasoning are recorded in the task's events.jsonl."
            ),
            "triggers_refined": triggers_refined,
            "triggers_refined_fired": fired_refined,
            "verdict_refined": verdict_refined,
            "n_sign_flips_total": len(per_cell["sign_flips"]),
            "n_sign_flips_conclusion_bearing": len(cb_sign_flips),
            "sign_flips_conclusion_bearing": cb_sign_flips,
            "n_conclusion_bearing_units": sum(
                1 for r in per_cell["rows"] if r.get("conclusion_bearing")
            ),
        },
        "per_cell": per_cell,
        "transfer": transfer,
        "two_by_two": two_by_two,
        "stage2": stage2,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")

    s = per_cell["summary"]
    logger.info("units compared: %s", s["n_units"])
    mx = s["max_abs_delta_f_steered_mean"]
    mn = s["mean_abs_delta_f_steered_mean"]
    logger.info(
        "|delta f_steered_mean|: max=%s mean=%s",
        "n/a" if mx is None else f"{mx:.6f}",
        "n/a" if mn is None else f"{mn:.6f}",
    )
    logger.info(
        "transfer rho: shipped=%.10f restricted=%.10f delta=%+.3e",
        transfer["shipped_rho"],
        transfer["restricted_rho"],
        transfer["delta_rho"],
    )
    logger.info("two-by-two counts changed: %s", two_by_two["counts_changed"])
    logger.info(
        "stage2: max|delta f_beh|=%s sign_flips=%s",
        stage2["max_abs_delta_f_beh"],
        stage2["n_sign_flips"],
    )
    for k, v in triggers.items():
        logger.info("  trigger %-38s %s", k, "FIRED" if v else "not fired")
    logger.info("PRE-REGISTERED VERDICT: %s", verdict)
    if fired:
        logger.info("  fired: %s", ", ".join(fired))
    logger.info(
        "sign flips: %s total, %s conclusion-bearing",
        len(per_cell["sign_flips"]),
        len(cb_sign_flips),
    )
    for f in per_cell["sign_flips"]:
        logger.info(
            "  %s %+.8f -> %+.8f | conclusion_bearing=%s ci=%s p_iut=%s mde=%s",
            f["unit"],
            f["shipped"],
            f["restricted"],
            f.get("conclusion_bearing"),
            f.get("shipped_ci95_steered"),
            f.get("shipped_p_iut"),
            f.get("realized_mde_single_test"),
        )
    logger.info("REFINED (a') VERDICT: %s", verdict_refined)
    if fired_refined:
        logger.info("  fired: %s", ", ".join(fired_refined))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
