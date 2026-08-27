"""Exact intruded-row x judged-compliance join for issue #2587.

Round-3 interpretation follow-up: the round-1 language-intrusion scan
(`eval_results/issue_2587/intrusion_scan_2587.json`) could not be joined to
the judge verdicts because the judge cache is keyed by prompt hash, so the
body carried a conservative worst-case recount ("flips at most two of 57
judged wordings"). This script runs the exact per-check join over the local
anchor rollout shards and `judge_scores.jsonl`, replacing the bound with a
measured result.

Per judged check (axis, value_id, kind, carrier, draw) it flags whether the
underlying rollout text matches the project CJK intrusion regex, then
recounts every wording's fire decision (bar: ceil(0.7 * 24) = 17 complying
checks of 24) and every judged axis floor (ceil(0.6 * width) fired base
wordings) after zeroing intruded-and-complying checks.

Writes `eval_results/issue_2587/intrusion_judge_join.json`.

Usage (VM, from the issue worktree):
    uv run python scripts/issue2587_intrusion_judge_join.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Same class as intrusion_scan_2587.json (project-standard CJK scan).
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

CHECKS_PER_WORDING = 24  # 12 carriers x 2 judged draws
FIRE_BAR = math.ceil(0.7 * CHECKS_PER_WORDING)  # 17

DEFAULT_ANCHOR_DIRS = (
    "data/issue_2587/judge_work/anchors_staging/issue2587_minpair/raw_completions/anchors",
    "data/issue_2587/hf_dl/anchors_extra/issue2587_minpair/raw_completions/anchors",
)
DEFAULT_JUDGE_SCORES = "data/issue_2587/judge_work/raw/judge_scores.jsonl"
DEFAULT_OUT = "eval_results/issue_2587/intrusion_judge_join.json"


def load_intrusion_lookup(anchor_dirs: list[Path]) -> dict[tuple[str, int], bool]:
    lookup: dict[tuple[str, int], bool] = {}
    n_files = 0
    for d in anchor_dirs:
        for shard in sorted(d.glob("anchors_*.jsonl")):
            n_files += 1
            with shard.open() as fh:
                for line in fh:
                    row = json.loads(line)
                    key = (row["context_id"], int(row["draw"]))
                    if key in lookup:
                        raise ValueError(f"duplicate rollout key {key} in {shard}")
                    lookup[key] = bool(CJK_RE.search(row["text"]))
    if n_files != 12:
        raise ValueError(f"expected 12 anchor shards, found {n_files}")
    if len(lookup) != 10_800:
        raise ValueError(f"expected 10,800 rollouts, found {len(lookup)}")
    return lookup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-dirs", nargs="+", default=list(DEFAULT_ANCHOR_DIRS))
    ap.add_argument("--judge-scores", default=DEFAULT_JUDGE_SCORES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    lookup = load_intrusion_lookup([Path(p) for p in args.anchor_dirs])

    judge_rows = [json.loads(line) for line in Path(args.judge_scores).open()]
    if len(judge_rows) != 1_464:
        raise ValueError(f"expected 1,464 judge rows, found {len(judge_rows)}")

    # Per-wording tallies keyed (axis, value_id, kind).
    tally: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "n_checks": 0,
            "comply": 0,
            "noncomply": 0,
            "incomplete": 0,
            "intruded": 0,
            "intruded_comply": 0,
        }
    )
    n_intruded = n_intruded_comply = 0
    n_intruded_ex_al = n_intruded_comply_ex_al = 0
    for row in judge_rows:
        key = (row["context_id"], int(row["draw"]))
        if key not in lookup:
            raise KeyError(f"judged check {key} has no rollout row")
        intruded = lookup[key]
        t = tally[(row["axis"], row["value_id"], row["kind"])]
        t["n_checks"] += 1
        t[row["outcome"]] += 1
        if intruded:
            t["intruded"] += 1
            n_intruded += 1
            comply = row["outcome"] == "comply"
            n_intruded_comply += comply
            if row["axis"] != "answer_language":
                n_intruded_ex_al += 1
                n_intruded_comply_ex_al += comply
            if comply:
                t["intruded_comply"] += 1

    wordings = []
    flips = []
    axis_base: dict[str, dict[str, int]] = defaultdict(
        lambda: {"width": 0, "fired_before": 0, "fired_after": 0}
    )
    for (axis, value_id, kind), t in sorted(tally.items()):
        comply_after = t["comply"] - t["intruded_comply"]
        # Production rule (manipulation_check_2587.json value_rows): any incomplete
        # check makes the wording undetermined -> 57 determinate of 61.
        determinate = t["incomplete"] == 0
        fired_before = t["comply"] >= FIRE_BAR
        fired_after = comply_after >= FIRE_BAR
        # For the answer-language Chinese value, CJK content IS the instructed
        # behavior: blind zeroing removes on-instruction compliance, so its flip
        # is degenerate, not evidence of intrusion-inflated compliance.
        on_instruction_cjk = axis == "answer_language" and value_id == "chinese"
        rec = {
            "axis": axis,
            "value_id": value_id,
            "kind": kind,
            **t,
            "comply_after_zeroing": comply_after,
            "fire_bar": FIRE_BAR,
            "determinate": determinate,
            "fired_before": fired_before,
            "fired_after": fired_after,
            "on_instruction_cjk": on_instruction_cjk,
            "flips": determinate and fired_before != fired_after,
        }
        wordings.append(rec)
        if rec["flips"]:
            flips.append(rec)
        if kind == "orig":
            # An undetermined wording never counts as fired for the axis floor
            # (production: format passes exactly at 3 of 5 fired base values).
            axis_base[axis]["width"] += 1
            axis_base[axis]["fired_before"] += determinate and fired_before
            axis_base[axis]["fired_after"] += determinate and fired_after

    floors = {}
    for axis, b in sorted(axis_base.items()):
        floor = math.ceil(0.6 * b["width"])
        floors[axis] = {
            **b,
            "floor": floor,
            "pass_before": b["fired_before"] >= floor,
            "pass_after": b["fired_after"] >= floor,
        }

    n_det = sum(w["determinate"] for w in wordings)
    out = {
        "schema": "issue2587_intrusion_judge_join_v1",
        "note": (
            "Exact per-check join of the CJK intrusion scan with the judge verdicts, "
            "replacing the round-1 worst-case bound. Zeroing intruded-and-complying checks "
            "is a worst case: for the answer-language Chinese value, CJK content is "
            "on-instruction compliance, so its intruded checks are counted but not evidence "
            "of inflation. Fire bar ceil(0.7*24)=17 complying checks; axis floor "
            "ceil(0.6*width) fired base (kind=orig) wordings."
        ),
        "regex": CJK_RE.pattern,
        "totals": {
            "n_judged_checks": len(judge_rows),
            "n_intruded_judged_checks": n_intruded,
            "n_intruded_judged_comply": n_intruded_comply,
            "n_intruded_judged_checks_excl_answer_language": n_intruded_ex_al,
            "n_intruded_judged_comply_excl_answer_language": n_intruded_comply_ex_al,
            "n_wordings": len(wordings),
            "n_determinate_wordings": n_det,
            "n_flips": len(flips),
            "n_flips_excl_on_instruction_cjk": sum(1 for f in flips if not f["on_instruction_cjk"]),
        },
        "flipped_wordings": flips,
        "axis_floors": floors,
        "wordings": wordings,
        "inputs": {
            "anchor_dirs": [str(p) for p in args.anchor_dirs],
            "judge_scores": str(args.judge_scores),
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps(out["totals"], indent=2))
    for f in flips:
        print(
            f"FLIP: {f['axis']} {f['value_id']} ({f['kind']}): "
            f"{f['comply']}/24 comply, {f['intruded_comply']} intruded-comply -> "
            f"{f['comply_after_zeroing']} < {FIRE_BAR}"
        )
    for axis, fl in floors.items():
        if not fl["pass_after"]:
            print(f"FLOOR FAIL after zeroing: {axis} {fl}")


if __name__ == "__main__":
    main()
