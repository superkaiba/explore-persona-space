"""Issue #2564 content-constraint per-value-pair calibration decomposition (r2).

For the content-constraint axis's 120 fired swap pairs, fits the through-origin
regression slope of predicted on observed shift norms PER value-pair
orientation (single-turn map, tail-inclusive answer summary at layer 19),
reported as a ratio to the map's global slope over all 2,778 pairs — the same
convention as ``minpair_delta.json``'s per-axis calibration — plus the
companion slope over fired pairs EXCLUDING the under-twenty-words value (v5)
under both the tail and span-mean pooling summaries, and per-orientation mean
observed norms + mean absolute answer-length deltas (the length-coupling
read). Inputs: the committed ``perpair.jsonl``. Writes
``eval_results/issue_2564/content_constraint_pair_slopes.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_WT = Path(__file__).resolve().parents[1]
GLOBAL_SLOPE_ALL2778 = 1.0136696325596286  # minpair_delta.json calibration.global_slope_all2778


def _slope(rows: list[dict], obs_key: str) -> float:
    """Through-origin least-squares slope of predicted-on-observed norms."""
    num = sum(r["norm_pred"]["arm_779ce"] * r[obs_key] for r in rows)
    den = sum(r[obs_key] ** 2 for r in rows)
    return num / den


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--perpair", type=Path, default=_WT / "eval_results/issue_2564/perpair.jsonl")
    ap.add_argument(
        "--out",
        type=Path,
        default=_WT / "eval_results/issue_2564/content_constraint_pair_slopes.json",
    )
    args = ap.parse_args(argv)

    rows = []
    for line in args.perpair.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["axis"] == "content_constraint" and r["pair_class"] == "swap":
                rows.append(r)
    assert len(rows) == 120, len(rows)
    fired = [r for r in rows if r["in_headline_70"]]
    assert len(fired) == 120, len(fired)  # all 5 content-constraint values fired

    by_orient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_orient[r["orientation"]].append(r)
    per_pair = {}
    for orient, sub in sorted(by_orient.items()):
        s = _slope(sub, "norm_obs_tail_L19")
        per_pair[orient] = {
            "n_pairs": len(sub),
            "slope": s,
            "ratio_to_global": s / GLOBAL_SLOPE_ALL2778,
            "mean_obs_norm": sum(r["norm_obs_tail_L19"] for r in sub) / len(sub),
            "mean_abs_ans_len_delta_tokens": sum(abs(r["ans_len_delta"]) for r in sub) / len(sub),
        }
    no_v5 = [r for r in fired if not any(v.startswith("v5") for v in (r["value_a"], r["value_b"]))]
    report = {
        "meta": {
            "script": "scripts/issue2564_pair_slopes.py",
            "convention": "through-origin slope of predicted on observed shift norms, "
            "single-turn map (arm_779ce), ratio to the map's global slope over all 2,778 pairs "
            "(minpair_delta.json calibration convention)",
            "global_slope_all2778": GLOBAL_SLOPE_ALL2778,
            "v5_instruction": "Use fewer than twenty words in your answer.",
        },
        "per_value_pair": per_pair,
        "fired_all": {
            "n_pairs": len(fired),
            "ratio_to_global": _slope(fired, "norm_obs_tail_L19") / GLOBAL_SLOPE_ALL2778,
        },
        "fired_excluding_v5": {
            "n_pairs": len(no_v5),
            "ratio_to_global_tail": _slope(no_v5, "norm_obs_tail_L19") / GLOBAL_SLOPE_ALL2778,
            "ratio_to_global_span_mean": _slope(no_v5, "norm_obs_span_L19") / GLOBAL_SLOPE_ALL2778,
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[pair-slopes] {len(per_pair)} orientations; fired ratio "
        f"{report['fired_all']['ratio_to_global']:.3f}; v5-excluded "
        f"{report['fired_excluding_v5']['ratio_to_global_tail']:.3f}; wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
