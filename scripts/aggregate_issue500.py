#!/usr/bin/env python3
"""Issue #500 framing-cleaned aggregator.

Re-aggregates per-(arm × bystander × seed) judged JSONLs with #444's
known-bad framings handled per plan §6.1:

  - Framing #10 (``novel_decoy``)  -- DROPPED entirely from EVERY output
                                       (rubric logic bug; #444 clean-result
                                       body cites it).
  - Framings #2 / #4 / #6           -- FLAGGED (base-FP > 5% ceiling).
                                       INCLUDED in the 5-way transparency
                                       roll-up and the per-framing pass-rate
                                       table; EXCLUDED from the headline
                                       ``leak_rate_headline`` so the primary
                                       DV is not inflated by known-bad-FP
                                       framings (round-2 BLOCKER #2 fix).

For each arm (Arm A / B / C) and each bystander persona, computes:
  * Per-framing pass rates over KEPT_FRAMING_IDS = {1,2,3,4,5,6,7,8,9,11}
    (flagged framings annotated with the flag reason).
  * A-family invented-canonical (``stated_seven``) emission rate over the
    A_reformulation family.
  * ``leak_rate_headline`` = stated_seven rate over A_reformulation +
    HEADLINE_FRAMING_IDS = {1,3,5,7,8,9,11} (the headline DV).
  * 5-way ``output_category`` proportions over freeform5 + KEPT_FRAMING_IDS
    (includes flagged framings, for transparency).

Output: ``eval_results/issue_500/<arm>/aggregate_cleaned.json``.

Run once per arm AFTER #500's per-arm full-eval + Haiku 5-way judging
completes; analyzer reads ``aggregate_cleaned.json``, not the parent
driver's raw ``aggregate_*.json``.
"""

# ruff: noqa: RUF002
# (greek + arrow + multiplication-sign characters intentional in docstrings)

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Framings deliberately DROPPED (rubric logic bug; plan §4 + §6.1).
DROP_FRAMING_IDS: frozenset[int] = frozenset({10})
# Framings FLAGGED because their base-rate FP > 5% per the #444 panel
# (plan §6.1). They are included in transparency reporting + 5-way roll-up
# but EXCLUDED from the headline `leak_rate_headline` to keep the primary
# DV free of base-rate FP contamination (round-2 BLOCKER #2 fix).
FLAG_FRAMING_IDS: frozenset[int] = frozenset({2, 4, 6})
# All framings kept after the hard-drop (used for the 5-way roll-up + per-
# framing pass-rate transparency table). 10 framings: {1,2,3,4,5,6,7,8,9,11}.
KEPT_FRAMING_IDS: tuple[int, ...] = tuple(f for f in range(1, 12) if f not in DROP_FRAMING_IDS)
# Headline framings: kept minus flagged. 7 framings: {1,3,5,7,8,9,11}. This is
# the planner's per-§6.1 "after dropping #10 and base-correcting 2/4/6" policy
# implemented as "drop #10, exclude 2/4/6 from the headline" -- simpler than
# per-persona base-correction and gives the same construct (a leak rate not
# inflated by known-bad-FP framings).
HEADLINE_FRAMING_IDS: tuple[int, ...] = tuple(
    f for f in KEPT_FRAMING_IDS if f not in FLAG_FRAMING_IDS
)
assert HEADLINE_FRAMING_IDS == (1, 3, 5, 7, 8, 9, 11), HEADLINE_FRAMING_IDS


def _stated_seven_label(verdict: dict[str, Any]) -> bool:
    """Return True if the 5-way Haiku verdict labelled this row stated_seven.

    Round-4: accept ``output_category_5way`` (the canonical key written by
    the 5-way Haiku judge in ``reanalyze_issue444_5way.py`` + the wrapper's
    ``_phase_baseline_judge``) AND ``output_category`` / ``category``
    (legacy 4-way / earlier judge prompt versions).
    """
    if not verdict:
        return False
    cat = (
        verdict.get("output_category_5way")
        or verdict.get("output_category")
        or verdict.get("category")
    )
    return cat == "stated_seven"


def _verdict_pass(verdict: dict[str, Any]) -> bool:
    return verdict.get("pass") is True


def _aggregate_one_judged_file(judged_path: Path, eval_personas: tuple[str, ...]) -> dict[str, Any]:
    """Per-persona aggregate of one cell's judged JSONL with #500 exclusions."""
    if not judged_path.exists():
        return {"missing": True, "path": str(judged_path)}
    rows = [json.loads(line) for line in judged_path.open() if line.strip()]

    per_persona: dict[str, Any] = {}
    for persona in eval_personas:
        p_rows = [r for r in rows if r["persona"] == persona]

        # Per-framing pass rate (skip dropped, flag flagged).
        framings_kept: dict[int, dict[str, Any]] = {}
        framings_flagged: dict[int, dict[str, Any]] = {}
        for fid in KEPT_FRAMING_IDS:
            f_rows = [
                r for r in p_rows if r["family"] == "framing381" and int(r["sub_framing"]) == fid
            ]
            n = len(f_rows)
            passed = sum(1 for r in f_rows if _verdict_pass(r.get("verdict", {})))
            entry = {"n": n, "pass_rate": passed / max(1, n)}
            if fid in FLAG_FRAMING_IDS:
                entry["flagged"] = "base_fp_above_5pct"
                framings_flagged[fid] = entry
            else:
                framings_kept[fid] = entry

        # A-family invented-canonical (stated_seven) emission rate.
        a_rows = [r for r in p_rows if r["family"] == "A_reformulation"]
        n_a = len(a_rows)
        canon = sum(1 for r in a_rows if _stated_seven_label(r.get("verdict", {})))
        a_rate = canon / max(1, n_a)

        # 5-way category roll-up across freeform5 + kept framings.
        cat_rows = [
            r
            for r in p_rows
            if (
                r["family"] == "freeform5"
                or (r["family"] == "framing381" and int(r["sub_framing"]) not in DROP_FRAMING_IDS)
            )
        ]
        cat_counts: Counter[str] = Counter()
        for r in cat_rows:
            v = r.get("verdict", {})
            cat = v.get("output_category_5way") or v.get("output_category") or v.get("category")
            if cat is None:
                continue
            cat_counts[cat] += 1
        total = sum(cat_counts.values())
        proportions = {cat: cat_counts[cat] / max(1, total) for cat in cat_counts}

        # Headline stated_seven rate across the kept 7-framing panel +
        # A_reformulation (the #500 primary DV). HEADLINE_FRAMING_IDS =
        # {1,3,5,7,8,9,11} -- DROP_FRAMING_IDS excluded (rubric bug) AND
        # FLAG_FRAMING_IDS {2,4,6} excluded (base-FP > 5%). The 5-way roll-up
        # above still includes the flagged framings for transparency.
        headline_rows = [
            r
            for r in p_rows
            if (
                r["family"] == "A_reformulation"
                or (r["family"] == "framing381" and int(r["sub_framing"]) in HEADLINE_FRAMING_IDS)
            )
        ]
        n_h = len(headline_rows)
        stated_seven_h = sum(1 for r in headline_rows if _stated_seven_label(r.get("verdict", {})))
        leak_rate = stated_seven_h / max(1, n_h)

        per_persona[persona] = {
            "leak_rate_headline": leak_rate,
            "n_headline_rows": n_h,
            "stated_seven_headline": stated_seven_h,
            "a_family_stated_seven_rate": a_rate,
            "n_a_family": n_a,
            "framings_kept": framings_kept,
            "framings_flagged": framings_flagged,
            "category_counts": dict(cat_counts),
            "category_proportions": proportions,
            "n_5way_rows": total,
        }
    return {
        "judged_path": str(judged_path),
        "n_judged_rows": len(rows),
        "exclusion_policy": {
            "dropped_framings": sorted(DROP_FRAMING_IDS),
            "flagged_framings": sorted(FLAG_FRAMING_IDS),
            "kept_framings_for_5way_rollup": list(KEPT_FRAMING_IDS),
            "headline_framings": list(HEADLINE_FRAMING_IDS),
            "_doc": (
                "leak_rate_headline = stated_seven rate over A_reformulation + "
                "HEADLINE_FRAMING_IDS {1,3,5,7,8,9,11} (dropped #10, excluded "
                "base-FP-flagged 2/4/6). 5-way category roll-up includes the "
                "flagged framings for transparency."
            ),
        },
        "per_persona": per_persona,
    }


def _arm_aggregate(
    arm_slug: str, panel: tuple[str, ...], eval_root: str = "issue_500"
) -> dict[str, Any]:
    """Aggregate all cells in one arm subtree.

    ``eval_root`` parametrizes the ``eval_results/<root>/`` subtree so #541
    (and later reruns) reuse this aggregator unchanged; the default preserves
    the #500 behavior byte-for-byte.
    """
    arm_root = REPO / "eval_results" / eval_root / arm_slug
    if not arm_root.exists():
        raise RuntimeError(f"arm root {arm_root} missing -- did --phase full-eval run?")

    # Baseline first.
    baseline_judged_candidates = list(arm_root.glob("baseline_judged_*.jsonl"))
    out: dict[str, Any] = {
        "arm_slug": arm_slug,
        "panel": list(panel),
        "per_cell": {},
    }
    if baseline_judged_candidates:
        out["per_cell"]["baseline"] = _aggregate_one_judged_file(
            baseline_judged_candidates[0], panel
        )

    # Round-5 fix: glob the 5-WAY trained-cell verdicts explicitly. The
    # parent's `phase_full_eval` writes LINKAGE-rubric verdicts to
    # `judged_{cell.tag}.jsonl` -- if we globbed `judged_*.jsonl` we'd
    # silently pick up those linkage files instead of the 5-way ones
    # (the round-3 reviewer's catch). The wrapper's
    # `_phase_trained_cell_5way_rejudge` writes 5-way verdicts to
    # `judged_5way_{cell.tag}.jsonl`; this glob picks up ONLY those.
    # NOTE: `judged_5way_*.jsonl` does NOT match `baseline_judged_*.jsonl`
    # either (different stem prefix), so the baseline + trained-cell
    # streams stay disjoint.
    for judged_path in sorted(arm_root.glob("judged_5way_*.jsonl")):
        cell_tag = judged_path.stem.removeprefix("judged_5way_")
        out["per_cell"][cell_tag] = _aggregate_one_judged_file(judged_path, panel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        help=(
            "source persona of the arm. With the default --eval-root the #500 "
            "choices apply (marine_biologist / local_resident / "
            "courthouse_architecture_historian); with a custom --eval-root any "
            "source persona is accepted (panel must then come from --panel)."
        ),
    )
    ap.add_argument(
        "--eval-root",
        default="issue_500",
        help="eval_results/<root>/ subtree to aggregate (e.g. issue_541)",
    )
    ap.add_argument(
        "--panel",
        default=None,
        help=(
            "comma-separated FULL panel (source included; the source is excluded "
            "here). Default = #500's PANEL_15. #541 passes the prior_screen panel."
        ),
    )
    ap.add_argument(
        "--arm-slug",
        default=None,
        help="arm directory slug; defaults to arm_<arm> (matches #500/#541 conventions)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "output JSON path; defaults to "
            "eval_results/<eval-root>/<arm-slug>/aggregate_cleaned.json"
        ),
    )
    args = ap.parse_args()

    if args.eval_root == "issue_500" and args.arm not in (
        "marine_biologist",
        "local_resident",
        "courthouse_architecture_historian",
    ):
        raise SystemExit(f"--arm {args.arm!r} is not a #500 arm (pass --eval-root for reruns)")

    arm_slug = args.arm_slug or f"arm_{args.arm}"
    if args.panel:
        pool = tuple(x.strip() for x in args.panel.split(",") if x.strip())
    else:
        # Per-arm panel = 15-pool minus the arm's source (#500 default).
        from run_experiment_500 import PANEL_15

        pool = PANEL_15
    panel = tuple(x for x in pool if x != args.arm)
    assert len(panel) == len(pool) - 1, (args.arm, pool)
    if args.eval_root == "issue_500" and not args.panel:
        assert len(panel) == 14, panel

    result = _arm_aggregate(arm_slug, panel, eval_root=args.eval_root)
    out_path = (
        Path(args.out)
        if args.out
        else REPO / "eval_results" / args.eval_root / arm_slug / "aggregate_cleaned.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"WROTE {out_path}")
    for cell, info in result["per_cell"].items():
        if "per_persona" not in info:
            continue
        print(f"  {cell}: {len(info['per_persona'])} personas")
        for persona, pdata in sorted(
            info["per_persona"].items(),
            key=lambda kv: -kv[1].get("leak_rate_headline", 0),
        )[:5]:
            print(
                f"    {persona:35} leak={pdata['leak_rate_headline']:.3f} "
                f"n_rows={pdata['n_headline_rows']}"
            )


if __name__ == "__main__":
    main()
