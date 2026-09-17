#!/usr/bin/env python
"""Issue #2661 — Der 10-way matching read over the W4 records.

Reads ``judge_aggregates/w4_matching.json`` (written by
``issue2661_judge_waves.py --wave w4``) and writes ``w4_der_read.json``
beside it: overall matching accuracy with a Wilson 95% score interval
(chance 0.10) plus the #2552-shape PER-SLOT sensitivity — accuracy grouped
by the GOLD candidate's presented slot label, and the judge's raw CHOICE
frequency per slot (position-bias read). Valid records only; drops are
counted, never imputed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGG = PROJECT_ROOT / "eval_results" / "issue_2661" / "judge_aggregates"


def wilson(p: float, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson 95% score interval for a binomial proportion."""
    if n <= 0 or not (p == p):
        return None
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - half) / denom, (centre + half) / denom]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matching-json", type=Path, default=AGG / "w4_matching.json")
    ap.add_argument("--out", type=Path, default=AGG / "w4_der_read.json")
    args = ap.parse_args()
    doc = json.loads(args.matching_json.read_text())
    records = doc["records"]
    valid = [r for r in records if r.get("class") == "valid"]
    n_valid = len(valid)
    n_correct = sum(1 for r in valid if r["correct"])
    acc = n_correct / n_valid if n_valid else float("nan")
    by_gold: dict[str, list[dict]] = defaultdict(list)
    choice_counts: dict[str, int] = defaultdict(int)
    for r in valid:
        by_gold[str(r["gold"])].append(r)
        choice_counts[str(r["choice"])] += 1
    per_slot = {}
    for slot in sorted(by_gold):
        rows = by_gold[slot]
        k = sum(1 for r in rows if r["correct"])
        a = k / len(rows)
        per_slot[slot] = {
            "n_gold_here": len(rows),
            "n_correct": k,
            "accuracy": a,
            "wilson_ci95": wilson(a, len(rows)),
            "choice_rate": choice_counts.get(slot, 0) / n_valid if n_valid else None,
        }
    out = {
        "source": str(args.matching_json.name),
        "n_records": len(records),
        "n_valid": n_valid,
        "n_dropped": len(records) - n_valid,
        "n_correct": n_correct,
        "accuracy": acc,
        "wilson_ci95": wilson(acc, n_valid),
        "chance": doc.get("chance", 0.1),
        "per_slot_sensitivity": per_slot,
        "convention": (
            "valid records only (drops counted, never imputed); per-slot = accuracy "
            "grouped by the gold candidate's presented slot label + raw choice rate "
            "per slot (position-bias read, #2552 shape); Wilson 95% score intervals"
        ),
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    slot_line = " ".join(
        f"{s}:{v['accuracy']:.2f}(n={v['n_gold_here']})" for s, v in per_slot.items()
    )
    print(
        f"[w4-der-read] acc={acc:.4f} ci95={out['wilson_ci95']} n_valid={n_valid} "
        f"dropped={out['n_dropped']} chance={out['chance']}\n[w4-der-read] per-slot {slot_line}"
    )


if __name__ == "__main__":
    main()
