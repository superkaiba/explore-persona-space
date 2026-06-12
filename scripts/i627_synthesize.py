#!/usr/bin/env python3
"""Task #627 — cross-comparison synthesis table (joins Phases 2 + 3).

One row per comparison family (plan §6 hero-1 rows):

    sycophancy contrastive-vs-posonly at install 0.50   (NEW — Phase 2)
    refusal LoRA-vs-FT measured matched read            (#606, verbatim)
    sycophancy LoRA-vs-FT matched read                  (#606, verbatim)
    marker LoRA-vs-FT matched 8-nat read                (#514, verbatim)
    marker mix contrast at matched install (fractions)  (#601 / Phase 3 H2)

Inputs are the Phase-2/3 output JSONs; missing inputs FAIL LOUD (the synthesis
joins 2+3 by design — run it last).

Output: eval_results/issue_627/analysis/synthesis.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i627_synthesize")

OUT_DIR = Path("eval_results/issue_627/analysis")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — the synthesis joins Phases 2+3; run the producing script first"
        )
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 — cross-comparison synthesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--matched-608", type=Path, default=OUT_DIR / "matched_install_608.json")
    parser.add_argument("--marker-601", type=Path, default=OUT_DIR / "marker_fractions_601.json")
    parser.add_argument("--fractions-606", type=Path, default=OUT_DIR / "fractions_606.json")
    parser.add_argument("--fractions-514", type=Path, default=OUT_DIR / "fractions_514.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "synthesis.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    m608 = _load(args.matched_608)
    m601 = _load(args.marker_601)
    f606 = _load(args.fractions_606)
    f514 = _load(args.fractions_514)

    h1 = m608["h1"]
    syc_606 = f606["per_behavior"]["sycophancy"]["published_headline_verbatim"]
    ref_606 = f606["per_behavior"]["refusal-ft-lr2e6-retrain"]["published_headline_verbatim"]
    matched_514 = f514["published_matched_verdict_verbatim"]
    h2 = m601["h2_matched_fraction_contrast"]["registered_floor_2p0"]

    # H2 per-group stats live under groups[primary_panel_size] (the analyzer
    # keeps only the verdict at top level). Sign FLIPPED to posonly minus
    # contrastive so every hero-1 row reads second-named-condition minus
    # first-named (round-2 fix, concern hero1-direction-gloss-wrong).
    if h2.get("verdict") == "no_pairs_above_floor":
        h2_gap, h2_ci = None, None
    else:
        h2_primary = h2.get("groups", {}).get(str(h2.get("primary_panel_size")))
        if h2_primary is None:
            raise RuntimeError(
                "marker_fractions_601.json: registered_floor_2p0 carries no primary group — "
                "re-run i627_analyze_marker.py"
            )
        raw_diff = h2_primary["mean_fraction_diff_contrastive_minus_posonly"]
        raw_ci = h2_primary["ci95_persona_cluster"]
        h2_gap = -raw_diff
        h2_ci = [-raw_ci[1], -raw_ci[0]]

    # Sign convention (binding for hero-1): in every row, gap = SECOND-named
    # condition minus FIRST-named, so gap > 0 = the second-named condition
    # leaks more at matched install. Each row also carries its own
    # plain-English positive-direction label for the figure.
    rows = [
        {
            "family": "sycophancy: contrastive vs positive-only at install 0.50 (NEW)",
            "gap": h1.get("h1_gap"),
            "ci95": h1.get("h1_gap_ci95"),
            "gap_units": "bystander-mean agreement-rate delta (posonly - contrastive)",
            "gap_direction_positive": "positives-only leaks more",
            "verdict": h1.get("verdict"),
            "n_complete_sources": h1.get("n_complete_sources"),
            "source": "issue 627 Phase 2 (matched_install_608.json)",
        },
        {
            "family": "refusal: LoRA vs full-FT (measured matched read, retrain)",
            "gap": ref_606.get("gap_plugin"),
            "ci95": ref_606.get("gap_ci95"),
            "gap_units": "bystander-mean rate delta (FT - LoRA)",
            "gap_direction_positive": "full fine-tune leaks more",
            "verdict": ref_606.get("verdict"),
            "source": "issue 606 (published, verbatim)",
        },
        {
            "family": "sycophancy: LoRA vs full-FT at install 0.50",
            "gap": syc_606.get("gap_plugin"),
            "ci95": syc_606.get("gap_ci95"),
            "gap_units": "bystander-mean rate delta (FT - LoRA)",
            "gap_direction_positive": "full fine-tune leaks more",
            "verdict": syc_606.get("verdict"),
            "source": "issue 606 (published, verbatim)",
        },
        {
            "family": "marker: LoRA vs full-FT at matched 8-nat install",
            "gap": matched_514["matched_rate_gap_ft_minus_lora_nat"],
            "ci95": matched_514["ci"],
            "gap_units": "bystander-mean delta-log-P gap (FT - LoRA), nats",
            "gap_direction_positive": "full fine-tune leaks more",
            "verdict": "matched null (published)",
            "source": "issue 514 (published, verbatim)",
        },
        {
            "family": "marker: contrastive mix vs positives-only fraction at matched install",
            "gap": h2_gap,
            "ci95": h2_ci,
            "gap_units": "bystander-mean EOS-margin leakage-fraction diff (posonly - contrastive)",
            "gap_direction_positive": "positives-only leaks more",
            "sign_note": "negated from the analyzer's contrastive-minus-posonly field so every "
            "row reads second-named minus first-named",
            "verdict": h2.get("verdict"),
            "scope_caveat": h2.get("scope_caveat"),
            "source": "issue 627 Phase 3 over issue 601 (marker_fractions_601.json)",
        },
    ]
    synthesis = {
        "issue": 627,
        "rows": rows,
        "h4_dose_curve_premise": {
            "refusal_endpoint_spread": f606["refusal_endpoint_trio"],
            "note": "cross-run leakage spread at matched install — install (+ one seed) "
            "does not determine leakage (descriptive)",
        },
        "inputs": {
            "matched_608": str(args.matched_608),
            "marker_601": str(args.marker_601),
            "fractions_606": str(args.fractions_606),
            "fractions_514": str(args.fractions_514),
        },
        "metadata": {"git_commit_sha": _git_sha(), "timestamp_utc": datetime.now(UTC).isoformat()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(synthesis, f, indent=2)
    log.info("[phase=p3_synthesize] -> %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
