#!/usr/bin/env python3
"""Persist the #2202 worst-discriminated tail composition (round `residual-read`).

The bottom-50-by-margin composition read was first computed inline in chat;
this script recomputes it from the COMMITTED artifacts so it is durable:

- margins/ranks: eval_results/issue_2202/residual_read/percontext_ranks_margins.npz
  (key `margin_csls_k10_whitencos_avg` — the clean operating point: CSLS K=10
  on whitened cosine, draw-averaged targets, 1,988 covered rows)
- labels: eval_results/issue_1738/percontext_summary_L19_ridge.csv (topic,
  language per context id)

Output: eval_results/issue_2202/residual_read/worst_discriminated.json —
bottom-50 rows (id, margin, rank, topic, language) + tail-vs-pool composition
over-representation tables. Zero compute beyond reading committed files.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NPZ = REPO / "eval_results/issue_2202/residual_read/percontext_ranks_margins.npz"
LABELS_CSV = REPO / "eval_results/issue_1738/percontext_summary_L19_ridge.csv"
OUT = REPO / "eval_results/issue_2202/residual_read/worst_discriminated.json"

TAIL_N = 50
CONVENTION = "csls_k10_whitencos_avg"


def main() -> None:
    d = np.load(NPZ)
    ci = d["ci"]
    margin = d[f"margin_{CONVENTION}"]
    rank = d[f"rank_{CONVENTION}"]
    n = len(ci)
    assert n == 1988, n

    labels: dict[int, dict[str, str]] = {}
    with LABELS_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[int(row["ci"])] = {"topic": row["topic"], "language": row["language"]}

    order = np.argsort(margin)
    tail_idx = order[:TAIL_N]
    tail_rows = []
    for i in tail_idx:
        c = int(ci[i])
        lab = labels.get(c, {"topic": "unlabeled", "language": "unlabeled"})
        tail_rows.append(
            {
                "ci": c,
                "margin": float(margin[i]),
                "rank": float(rank[i]),
                "topic": lab["topic"],
                "language": lab["language"],
            }
        )

    n_fail = sum(1 for r in tail_rows if r["rank"] > 1)
    covered_labs = [labels.get(int(c), {"topic": "unlabeled", "language": "unlabeled"}) for c in ci]

    def composition(field: str) -> dict[str, dict[str, float]]:
        tail_counts = Counter(r[field] for r in tail_rows)
        pool_counts = Counter(lab[field] for lab in covered_labs)
        out = {}
        for key, tc in sorted(tail_counts.items(), key=lambda kv: -kv[1]):
            pool_share = pool_counts[key] / n
            tail_share = tc / TAIL_N
            out[key] = {
                "tail_count": tc,
                "tail_share": round(tail_share, 4),
                "pool_share": round(pool_share, 4),
                "ratio": round(tail_share / pool_share, 2) if pool_share else None,
            }
        # pool labels absent from the tail (under-representation to zero)
        for key, pc in sorted(pool_counts.items(), key=lambda kv: -kv[1]):
            if key not in out and pc / n >= 0.05:
                out[key] = {
                    "tail_count": 0,
                    "tail_share": 0.0,
                    "pool_share": round(pc / n, 4),
                    "ratio": 0.0,
                }
        return out

    payload = {
        "round": "residual-read (worst-discriminated tail persistence)",
        "definition": (
            "bottom 50 of the 1,988 resample-covered rows by per-row retrieval margin "
            "(true-target score minus best competitor) under CSLS K=10 on whitened cosine "
            "with draw-averaged targets; negative margin = rank-1 failure"
        ),
        "n_tail": TAIL_N,
        "n_covered": n,
        "n_tail_failures": n_fail,
        "n_tail_successes": TAIL_N - n_fail,
        "tail_margin_min": round(float(margin[tail_idx].min()), 4),
        "tail_margin_max": round(float(margin[tail_idx].max()), 4),
        "pool_margin_median": round(float(np.median(margin)), 4),
        "composition_topic": composition("topic"),
        "composition_language": composition("language"),
        "tail_rows": tail_rows,
        "provenance": {
            "margins": str(NPZ.relative_to(REPO)),
            "labels": str(LABELS_CSV.relative_to(REPO)),
            "convention": CONVENTION,
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(
        f"wrote {OUT.relative_to(REPO)}: {n_fail} failures + {TAIL_N - n_fail} successes; "
        f"tail margins {payload['tail_margin_min']}..{payload['tail_margin_max']} "
        f"vs pool median {payload['pool_margin_median']}"
    )
    for field in ("composition_topic", "composition_language"):
        top = {k: v["ratio"] for k, v in list(payload[field].items())[:6]}
        print(field, top)


if __name__ == "__main__":
    main()
