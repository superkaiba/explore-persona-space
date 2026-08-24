#!/usr/bin/env python
"""Language-intrusion (CJK) audit for issue #2479 (analyzer Step 3.7).

Qwen-family generations under a non-CJK eval: scan BOTH substrates —

- (a) the capture/geometry substrate: every kept on-policy story row per op
  cell (the rows whose teacher-forced activations feed the fits), and
- (b) the judged axis pools: the reserved-conversation items behind the
  frozen AI-likeness axis, joined with per-item judge scores.

A row is intruded iff its story text matches the CJK/kana/hangul class
``[\\u4e00-\\u9fff\\u3400-\\u4dbf\\uf900-\\ufaff\\u3040-\\u30ff\\uac00-\\ud7af]``.
For the judged pools the audit additionally recomputes the frozen axis with
intruded items EXCLUDED, re-runs the headline Spearman(axis, recovery
fraction) under the recomputed axis, and reports both alongside the
registered read. Pure counting: no story text is printed.

Also emits (seed 42) the analyzer raw-output spot check (5 random pooled op
rows) and Methodology sample picks as sanitized ~15-word excerpts with
permanent HF row pointers.

Usage:
    uv run python scripts/issue2479_cjk_audit.py \
        [--stories-dir /tmp/i2479-stories] [--eval-dir eval_results/issue_2479]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

NAMES = [
    "helios",
    "iris",
    "cobalt",
    "vera",
    "wren",
    "priya",
    "marcus",
    "elena",
    "dana",
    "gus",
    "marisol",
    "tomas",
    "vex",
    "barnaby",
    "zara",
    "mort",
]

HF_STORY_TMPL = (
    "issue1345_framing/char_2479_{name}_op/raw_completions/stories/"
    "kept_stories_paired_op_instruct.jsonl"
)


def sanitize_excerpt(text: str, n_words: int = 15) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def _per_item_scores(raw: dict) -> dict[str, float]:
    """conv_id -> mean judge score over draws, from all_scores keys ail_<tag>_<conv>__<item>__<draw>."""
    acc: dict[str, list[float]] = {}
    for key, entry in raw["all_scores"].items():
        score = entry.get("score")
        if not isinstance(score, (int, float)):
            continue  # drop-never-coerce: REFUSAL / malformed draws excluded
        head = key.split("__")[0]  # ail_<tag>_<conv_id>
        conv_id = head.rsplit("_", 1)[-1]
        acc.setdefault(conv_id, []).append(float(score))
    return {cid: sum(v) / len(v) for cid, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories-dir", default="/tmp/i2479-stories")
    ap.add_argument("--eval-dir", default="eval_results/issue_2479")
    args = ap.parse_args()

    stories_root = Path(args.stories_dir)
    eval_dir = Path(args.eval_dir)

    verdict = json.loads((eval_dir / "gradient_verdict.json").read_text())
    head = verdict["headline"]
    frac_by_char = dict(zip(head["characters"], head["values"]))

    out: dict = {"issue": 2479, "regex": CJK_RE.pattern, "per_character": {}}
    pooled_rows = []

    for name in NAMES:
        kept_path = stories_root / HF_STORY_TMPL.format(name=name)
        rows = [json.loads(line) for line in kept_path.read_text().splitlines() if line.strip()]
        intr_rows = [r for r in rows if CJK_RE.search(r.get("story", ""))]

        # judged axis pool: reserved conv_ids for this character
        sample = json.loads((eval_dir / "judge_legs" / f"judge_sample_ail_{name}.json").read_text())
        axis_conv_ids = set(sample["conv_ids"])
        raw = json.loads((eval_dir / "judge_legs" / f"judge_raw_ail_{name}.json").read_text())
        per_item = _per_item_scores(raw)  # conv_id -> mean of draw scores

        axis_rows = [r for r in rows if r["conv_id"] in axis_conv_ids]
        axis_intr = [r for r in axis_rows if CJK_RE.search(r.get("story", ""))]

        def item_score(r: dict) -> float | None:
            return per_item.get(r["conv_id"])

        scores_clean = [
            s
            for r in axis_rows
            if not CJK_RE.search(r.get("story", ""))
            if (s := item_score(r)) is not None
        ]
        scores_intr = [s for r in axis_intr if (s := item_score(r)) is not None]
        scores_all = [s for r in axis_rows if (s := item_score(r)) is not None]

        out["per_character"][name] = {
            "capture_substrate": {"intruded": len(intr_rows), "total": len(rows)},
            "judged_axis_pool": {
                "intruded": len(axis_intr),
                "total": len(axis_rows),
                "axis_score_all_items": float(np.mean(scores_all)) if scores_all else None,
                "axis_score_excl_intruded": float(np.mean(scores_clean)) if scores_clean else None,
                "mean_score_intruded_items": float(np.mean(scores_intr)) if scores_intr else None,
            },
            "intruded_row_pointers": [
                {
                    "conv_id": r["conv_id"],
                    "story_id": r["story_id"],
                    "hf_path": HF_STORY_TMPL.format(name=name),
                }
                for r in intr_rows[:20]
            ],
        }
        for r in rows:
            pooled_rows.append((name, r))

    # headline recount under intrusion-excluded axis
    axis_reg, axis_excl, fracs = [], [], []
    for name in NAMES:
        pc = out["per_character"][name]["judged_axis_pool"]
        axis_reg.append(pc["axis_score_all_items"])
        axis_excl.append(
            pc["axis_score_excl_intruded"]
            if pc["axis_score_excl_intruded"] is not None
            else pc["axis_score_all_items"]
        )
        fracs.append(frac_by_char[name])
    rho_reg = spearmanr(axis_reg, fracs).statistic
    rho_excl = spearmanr(axis_excl, fracs).statistic
    out["headline_recount"] = {
        "registered_rho": head["rho"],
        "rho_axis_all_items_perpersona_join": float(rho_reg),
        "rho_axis_excl_intruded": float(rho_excl),
        "note": "per_persona mean_aligned join; registered axis uses the frozen leg reports",
    }

    # analyzer raw-output spot check: 5 random pooled rows, seed 42
    rng = random.Random(42)
    picks = rng.sample(range(len(pooled_rows)), 5)
    out["spot_check_seed42"] = []
    for i in picks:
        name, r = pooled_rows[i]
        out["spot_check_seed42"].append(
            {
                "character": name,
                "conv_id": r["conv_id"],
                "story_id": r["story_id"],
                "tier": r.get("tier"),
                "judge_verdict": r.get("judge_verdict"),
                "finish_reason": r.get("finish_reason"),
                "n_parsed_turns": r.get("n_parsed_turns"),
                "cjk_intruded": bool(CJK_RE.search(r.get("story", ""))),
                "excerpt_15w": sanitize_excerpt(r.get("story", "")),
                "hf_path": HF_STORY_TMPL.format(name=name),
            }
        )

    # Methodology sample picks: 3 rows each from top-axis + bottom-axis chars
    out["methodology_samples"] = {}
    for name in ("vera", "zara"):
        kept_path = stories_root / HF_STORY_TMPL.format(name=name)
        rows = [json.loads(line) for line in kept_path.read_text().splitlines() if line.strip()]
        raw = json.loads((eval_dir / "judge_legs" / f"judge_raw_ail_{name}.json").read_text())
        per_item = _per_item_scores(raw)
        scored = [(r, per_item[r["conv_id"]]) for r in rows if r["conv_id"] in per_item]
        rng2 = random.Random(42)
        sel = rng2.sample(scored, min(3, len(scored)))
        out["methodology_samples"][name] = [
            {
                "conv_id": r["conv_id"],
                "story_id": r["story_id"],
                "tier": r.get("tier"),
                "axis_item_score": s,
                "question_excerpt_15w": sanitize_excerpt(r.get("question", "")),
                "story_excerpt_15w": sanitize_excerpt(r.get("story", "")),
                "hf_path": HF_STORY_TMPL.format(name=name),
            }
            for r, s in sel
        ]

    out_path = eval_dir / "cjk_audit.json"
    out_path.write_text(json.dumps(out, indent=1))
    tot_i = sum(v["capture_substrate"]["intruded"] for v in out["per_character"].values())
    tot_n = sum(v["capture_substrate"]["total"] for v in out["per_character"].values())
    ax_i = sum(v["judged_axis_pool"]["intruded"] for v in out["per_character"].values())
    ax_n = sum(v["judged_axis_pool"]["total"] for v in out["per_character"].values())
    print(f"[cjk] capture substrate pooled: {tot_i}/{tot_n} intruded")
    print(f"[cjk] judged axis pools pooled: {ax_i}/{ax_n} intruded")
    print(
        f"[cjk] headline rho registered={head['rho']:.4f} "
        f"axis-join={rho_reg:.4f} excl-intruded={rho_excl:.4f}"
    )
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
