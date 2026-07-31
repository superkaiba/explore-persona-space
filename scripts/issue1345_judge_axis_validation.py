#!/usr/bin/env python
"""Issue #1345 — validate the AI-likeness axis against its rule-25 neighbours.

The AI-likeness rubric names the properties COMMONLY CONFUSED with
machine authorship and instructs the judge to set them aside (llm-judging rule
25, the #1482 class). Two of those exclusions are mechanically checkable on the
judged text we already have, at ZERO API cost:

  name channel   "a character explicitly ... being NAMED like a machine" — rows
                 whose rated text contains the character's own name.
  ai-word channel "a character explicitly SAYING it is an AI" — rows whose rated
                 text contains AI self-reference vocabulary.

For each channel this splits the cell's per-item scores into the rows that carry
it and the rows that do not, and reports both sub-means. If the channel-FREE
sub-mean matches the cell's pooled mean, the channel cannot be what produced the
cell's score — the judge did not see it in substantially all of the rated text.

Why this instead of a name-swap ablation: a swap can only move the pooled mean in
proportion to how much of the rated text carries the name. Measured on
char_helios_op that is 10 of 300 rows, so the maximum achievable movement is
~1 point by construction — a swap would return "no change" for a trivial reason
and validate nothing. This split answers the same question directly, and its
per-channel n makes the leverage explicit rather than assumed.

Pure read over committed artifacts (judge_raw + the persisted draw + the prepped
rows). No API calls, no re-judging.

CLI:
  uv run python scripts/issue1345_judge_axis_validation.py
  uv run python scripts/issue1345_judge_axis_validation.py --cells char_helios_op,char_dana_op
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_onpolicy_judge_legs as jl  # noqa: E402

DEFAULT_LEGS_DIR = Path(os.environ.get("EPM_I1345_JUDGE_OUT", "eval_results/issue_1345/judge_legs"))
PREP_DIR = Path(os.environ.get("EPM_I1345_PREP_DIR", "data/issue_1345/judge_prep"))

# The AI-self-reference vocabulary the rubric excludes.
AI_WORD_RE = re.compile(
    r"\b(AI|artificial intelligence|language model|assistant|chatbot)\b", re.IGNORECASE
)


# Character name per cell, from the cell tag. The name is the only per-cell datum
# needed, and deriving it from the tag keeps this in step with the run's cells.
def character_name_of(cell: str) -> str | None:
    """The character whose name the `name` channel looks for, or None."""
    bare = cell.removeprefix("char_")
    for suffix in ("_op_base", "_op", "_base"):
        bare = bare.removesuffix(suffix)
    return bare or None


def name_regex(name: str) -> re.Pattern[str]:
    """Match the character name in any casing the stories use (HELIOS / Helios)."""
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


def _block(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4) if values else None,
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
    }


def validate_cell(cell: str, legs_dir: Path, prep_dir: Path) -> dict:
    """Split one cell's per-item scores by each rule-25 channel."""
    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    d = legs_dir / cell
    sample_paths = sorted(d.glob("judge_sample_*.json"))
    raw_paths = sorted(d.glob("judge_raw_*.json"))
    assert sample_paths and raw_paths, f"{cell}: missing judge_sample/judge_raw under {d}"
    conv_ids = json.loads(sample_paths[0].read_text())["conv_ids"]
    prep_path = prep_dir / f"{cell}.jsonl"
    assert prep_path.exists(), f"{cell}: prepared rows missing at {prep_path}"
    prep = {r["conv_id"]: r for r in c.read_jsonl(prep_path)}

    items = [(jl.item_id(jl.LEG_AI_LIKENESS, cell, cid), "q", "a") for cid in conv_ids]
    res = judge_result_from_save_raw(raw_paths[0], items)
    iid_to_conv = {jl.item_id(jl.LEG_AI_LIKENESS, cell, cid): cid for cid in conv_ids}

    name = character_name_of(cell)
    n_re = name_regex(name) if name else None
    buckets: dict[str, list[float]] = {k: [] for k in ("name", "no_name", "ai", "no_ai", "all")}
    for iid, score in res.scores.items():
        if score is None:
            continue
        row = prep[iid_to_conv[iid]]
        text = f"{row['question']} {row['answer']}"
        buckets["all"].append(score)
        buckets["name" if (n_re and n_re.search(text)) else "no_name"].append(score)
        buckets["ai" if AI_WORD_RE.search(text) else "no_ai"].append(score)

    out = {
        "cell": cell,
        "character": name,
        "pooled": _block(buckets["all"]),
        "name_channel": {"carries": _block(buckets["name"]), "free": _block(buckets["no_name"])},
        "ai_word_channel": {"carries": _block(buckets["ai"]), "free": _block(buckets["no_ai"])},
    }
    # Max movement a NAME-SWAP ablation could achieve: only the name-carrying rows
    # can move, and at most across the full 0-100 scale.
    n_all, n_name = len(buckets["all"]), len(buckets["name"])
    out["name_swap_max_pooled_shift"] = round(100.0 * n_name / n_all, 3) if n_all else None
    for ch in ("name_channel", "ai_word_channel"):
        carries, free = out[ch]["carries"], out[ch]["free"]
        out[ch]["delta_carries_minus_free"] = (
            round(carries["mean"] - free["mean"], 4)
            if carries["mean"] is not None and free["mean"] is not None
            else None
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs-dir", type=Path, default=DEFAULT_LEGS_DIR)
    ap.add_argument("--prep-dir", type=Path, default=PREP_DIR)
    ap.add_argument(
        "--cells",
        default="",
        help="comma-separated cell tags (default: every ai_likeness cell present)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.cells.strip():
        cells = [x.strip() for x in args.cells.split(",") if x.strip()]
    else:
        cells = sorted(
            p.parent.name
            for p in args.legs_dir.glob("*/judge_raw_ail_*.json")
            if (args.prep_dir / f"{p.parent.name}.jsonl").exists()
        )
    assert cells, f"no ai_likeness cells found under {args.legs_dir}"

    rows = [validate_cell(cell, args.legs_dir, args.prep_dir) for cell in cells]
    print(
        f"{'cell':22s} {'pooled':>8s} {'name_n':>7s} {'name':>8s} {'no_name':>8s} "
        f"{'delta':>7s} {'swapmax':>8s} {'ai_n':>5s} {'ai':>8s} {'no_ai':>8s}"
    )
    for r in rows:
        nc, ac = r["name_channel"], r["ai_word_channel"]

        def f(v: float | None) -> str:
            return f"{v:8.2f}" if isinstance(v, (int, float)) else f"{'-':>8s}"

        print(
            f"{r['cell']:22s} {f(r['pooled']['mean'])} {nc['carries']['n']:7d} "
            f"{f(nc['carries']['mean'])} {f(nc['free']['mean'])} "
            f"{(nc['delta_carries_minus_free'] or 0):7.2f} "
            f"{(r['name_swap_max_pooled_shift'] or 0):8.2f} {ac['carries']['n']:5d} "
            f"{f(ac['carries']['mean'])} {f(ac['free']['mean'])}"
        )
    payload = {
        "metadata": c.metadata(0, len(rows), "scripts/issue1345_judge_axis_validation.py"),
        "ai_word_pattern": AI_WORD_RE.pattern,
        "reading": (
            "channel-FREE sub-mean ~= pooled mean => the judge did not see that channel in "
            "substantially all rated text, so it cannot be what produced the cell's score. "
            "name_swap_max_pooled_shift is the LARGEST pooled movement a name-swap ablation "
            "could produce for that cell (name-carrying share x the full 0-100 scale)."
        ),
        "cells": rows,
    }
    out = args.out or (args.legs_dir / "axis_validation.json")
    c.write_json(out, payload)
    print(f"\n[axis-validation] wrote {out}")
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
