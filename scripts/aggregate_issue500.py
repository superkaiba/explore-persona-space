#!/usr/bin/env python3
"""Issue #500 framing-cleaned aggregator.

Re-aggregates per-(arm × bystander × seed) judged JSONLs with #444's
known-bad framings removed:

  - Framing #10 (``novel_decoy``)  -- DROPPED entirely (rubric logic bug:
                                       #444 clean-result body cites it).
  - Framings #2 / #4 / #6           -- FLAGGED (base-FP > 5% ceiling);
                                       still aggregated but recorded as
                                       suspicious in the output.

For each arm (Arm A / B / C) and each bystander persona, computes:
  * Per-framing pass rates (DROPPED framings absent; FLAGGED framings
    annotated)
  * A-family invented-canonical (``stated_seven``) emission rate across
    A_reformulation + the kept 11-framing panel (sub-set 1,3,5,7,8,9,11)
  * 5-way output_category proportions across freeform5 + the kept framings

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

# Framings deliberately DROPPED (rubric logic bug or base-FP > 5%; plan §4).
DROP_FRAMING_IDS: frozenset[int] = frozenset({10})
FLAG_FRAMING_IDS: frozenset[int] = frozenset({2, 4, 6})
# The 11-framing panel #444 uses; after exclusion #500 keeps 7 framings for
# the headline stated_seven rate. FLAGGED framings stay in the panel (the
# planner chose annotation, not exclusion).
KEPT_FRAMING_IDS: tuple[int, ...] = tuple(f for f in range(1, 12) if f not in DROP_FRAMING_IDS)


def _stated_seven_label(verdict: dict[str, Any]) -> bool:
    """Return True if the 5-way Haiku verdict labelled this row stated_seven.

    The judge schema uses either ``output_category`` or ``category`` -- accept
    both to stay robust across the two #444 judge prompt versions.
    """
    cat = verdict.get("output_category") or verdict.get("category")
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
            cat = v.get("output_category") or v.get("category")
            if cat is None:
                continue
            cat_counts[cat] += 1
        total = sum(cat_counts.values())
        proportions = {cat: cat_counts[cat] / max(1, total) for cat in cat_counts}

        # Headline stated_seven rate across the kept 7-framing panel +
        # A_reformulation (the #500 primary metric).
        headline_rows = [
            r
            for r in p_rows
            if (
                r["family"] == "A_reformulation"
                or (r["family"] == "framing381" and int(r["sub_framing"]) not in DROP_FRAMING_IDS)
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
            "kept_framings_for_headline": list(KEPT_FRAMING_IDS),
        },
        "per_persona": per_persona,
    }


def _arm_aggregate(arm_slug: str, panel: tuple[str, ...]) -> dict[str, Any]:
    """Aggregate all cells in one arm subtree."""
    arm_root = REPO / "eval_results" / "issue_500" / arm_slug
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

    for judged_path in sorted(arm_root.glob("judged_*.jsonl")):
        cell_tag = judged_path.stem.removeprefix("judged_")
        out["per_cell"][cell_tag] = _aggregate_one_judged_file(judged_path, panel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["marine_biologist", "local_resident", "courthouse_architecture_historian"],
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON path; defaults to eval_results/issue_500/<arm>/aggregate_cleaned.json",
    )
    args = ap.parse_args()

    arm_slug = f"arm_{args.arm}"
    # Per-arm panel = 15-pool minus the arm's source.
    from run_experiment_500 import PANEL_15

    panel = tuple(x for x in PANEL_15 if x != args.arm)
    assert len(panel) == 14, panel

    result = _arm_aggregate(arm_slug, panel)
    out_path = (
        Path(args.out)
        if args.out
        else REPO / "eval_results" / "issue_500" / arm_slug / "aggregate_cleaned.json"
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
