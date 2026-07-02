#!/usr/bin/env python3
"""Issue #810 Phase C judge-cache contamination audit (persisted, reproducible).

Persists the negation-aware audit the round-1 analyzer ran inline (previously
unpersisted — interp-critique r1 required the classifier + output be committed
so the contamination rates are reproducible). Verifies the harmful_compliance
E0 cross-rubric contamination mechanism:

- ``eval/batch_judge.py::JudgeCache._hash_key`` = sha256(question + completion)
  — the behavior RUBRIC is not part of the cache key — and
  ``issue810_batch_rejudge_highm.py`` passed ONE shared ``--cache-dir`` across
  all three behaviors, so harmful_compliance's cache hits could return
  refusal-rubric judgments for shared (question, completion) pairs.

Evidence computed here, per behavior:

1. Heuristic rate: share of HIGH-score rows (>= 80) whose judge reasoning
   affirmatively describes a refusal (negation-aware regex, documented in the
   output JSON) — high for harmful_compliance iff refusal-rubric judgments
   leaked in (a refusal scores ~100 under the refusal rubric but must score
   ~0 under the harmful-compliance rubric).
2. Exact-share evidence (heuristic-independent): share of those flagged high
   rows whose EXACT (reasoning, score) pair is present verbatim in the refusal
   raw file, vs the same share among non-flagged high rows. Near-1.0 for the
   flagged set means the judgments literally ARE the refusal file's cached
   entries (closes the "harmful-rubric prompt bug" alternative).
3. Sycophancy structural check: distinct (reasoning, score) pairs overlapping
   refusal's (0 expected — its pool shares no (question, completion) pairs).
4. Draw-collapse: fraction of completions whose 8 draws collapse to a single
   distinct score (content-keyed cache => 1 cached judgment per completion).
5. Parse-error drop rates from e0_highm_graded.json (dropped, never coerced).

Usage::

    uv run python scripts/issue810_contamination_audit.py \
        --in eval_results/issue_810/phase_c \
        --out eval_results/issue_810/analysis/contamination_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue810_common import dump_json, reproducibility_metadata  # noqa: E402

logger = logging.getLogger("issue810_contamination_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BEHAVIORS = ("sycophancy", "refusal", "harmful_compliance")
HIGH_SCORE = 80
LOW_SCORE = 20

# Negation-aware affirmative-refusal-mention classifier. A row is flagged iff
# the reasoning contains a refusal/decline mention that is NOT negated within
# the preceding window (e.g. "explicitly refuses" flags; "rather than refusing"
# / "does not refuse" / "without declining" do not).
REFUSAL_MENTION = re.compile(r"refus|declin", re.IGNORECASE)
NEGATION_WINDOW_CHARS = 30
NEGATION = re.compile(
    r"\b(not|no|never|without|n't|rather than|instead of|opposite of|lack of|fails? to)\b[^.]*$",
    re.IGNORECASE,
)


def affirmative_refusal_mention(reasoning: str) -> bool:
    """True iff >=1 refusal/decline mention lacks a negation in its lookback window."""
    for m in REFUSAL_MENTION.finditer(reasoning):
        window = reasoning[max(0, m.start() - NEGATION_WINDOW_CHARS) : m.start()]
        if not NEGATION.search(window):
            return True
    return False


def _valid_rows(raw: dict) -> dict[str, tuple[str, float]]:
    """cid -> (reasoning, score) for parseable in-range rows (drop-never-coerce)."""
    out = {}
    for cid, v in raw["all_scores"].items():
        if v.get("error"):
            continue
        score = v.get("score")
        reasoning = v.get("reasoning")
        if isinstance(score, (int, float)) and 0 <= score <= 100 and isinstance(reasoning, str):
            out[cid] = (reasoning, float(score))
    return out


def _draw_collapse(rows: dict[str, tuple[str, float]]) -> dict:
    """Fraction of completions (>=2 valid draws) with exactly 1 distinct score."""
    by_completion: dict[str, set[float]] = {}
    counts: dict[str, int] = {}
    for cid, (_r, s) in rows.items():
        comp = cid.rsplit("__", 1)[0]
        by_completion.setdefault(comp, set()).add(s)
        counts[comp] = counts.get(comp, 0) + 1
    multi = [c for c, k in counts.items() if k >= 2]
    if not multi:
        return {"n_completions_multi_draw": 0, "frac_single_distinct_score": None}
    collapsed = sum(1 for c in multi if len(by_completion[c]) == 1)
    return {
        "n_completions_multi_draw": len(multi),
        "frac_single_distinct_score": collapsed / len(multi),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 judge-cache contamination audit")
    ap.add_argument(
        "--in", dest="in_dir", default=str(PROJECT_ROOT / "eval_results" / "issue_810" / "phase_c")
    )
    ap.add_argument(
        "--out",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "analysis" / "contamination_audit.json"
        ),
    )
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    raws = {}
    rows = {}
    for b in BEHAVIORS:
        with open(in_dir / f"rejudge_raw_{b}.json") as f:
            raws[b] = json.load(f)
        rows[b] = _valid_rows(raws[b])
        logger.info("[%s] %d/%d parseable rows", b, len(rows[b]), raws[b]["n_total"])

    refusal_pairs = set(rows["refusal"].values())

    per_behavior: dict = {}
    for b in BEHAVIORS:
        high = {cid: rs for cid, rs in rows[b].items() if rs[1] >= HIGH_SCORE}
        low = {cid: rs for cid, rs in rows[b].items() if rs[1] <= LOW_SCORE}
        flagged_high = {cid: rs for cid, rs in high.items() if affirmative_refusal_mention(rs[0])}
        flagged_low = {cid: rs for cid, rs in low.items() if affirmative_refusal_mention(rs[0])}
        nonflagged_high = {cid: rs for cid, rs in high.items() if cid not in flagged_high}

        def share(d: dict) -> float | None:
            """Share of rows whose exact (reasoning, score) pair is in the refusal file."""
            if not d:
                return None
            return sum(1 for rs in d.values() if rs in refusal_pairs) / len(d)

        per_behavior[b] = {
            "n_total_judge_calls": raws[b]["n_total"],
            "n_cached": raws[b]["n_cached"],
            "n_parseable": len(rows[b]),
            "n_high_ge80": len(high),
            "n_high_flagged_affirmative_refusal": len(flagged_high),
            "frac_high_flagged": (len(flagged_high) / len(high)) if high else None,
            "n_low_le20": len(low),
            "frac_low_flagged": (len(flagged_low) / len(low)) if low else None,
            # Heuristic-independent exact-share evidence:
            "frac_flagged_high_pairs_verbatim_in_refusal_file": share(flagged_high),
            "frac_nonflagged_high_pairs_verbatim_in_refusal_file": share(nonflagged_high),
            "n_distinct_pairs": len(set(rows[b].values())),
            "n_distinct_pairs_overlapping_refusal": (
                None if b == "refusal" else len(set(rows[b].values()) & refusal_pairs)
            ),
            "draw_collapse": _draw_collapse(rows[b]),
        }

    graded_path = in_dir / "e0_highm_graded.json"
    drops = {}
    if graded_path.is_file():
        with open(graded_path) as f:
            graded = json.load(f)
        for b, v in graded["by_behavior"].items():
            tot = raws[b]["n_total"]
            drops[b] = {
                "n_dropped_nan": v["n_dropped_nan"],
                "n_total": tot,
                "drop_rate": v["n_dropped_nan"] / tot,
            }

    out = {
        "purpose": "harmful_compliance E0 cross-rubric judge-cache contamination audit",
        "mechanism": (
            "eval/batch_judge.py::JudgeCache._hash_key = sha256(question+completion) "
            "(no rubric in key) + one shared --cache-dir across behaviors in "
            "issue810_batch_rejudge_highm.py; refusal judged first => "
            "harmful_compliance cache hits return refusal-rubric judgments on shared "
            "(question, completion) pairs"
        ),
        "classifier": {
            "refusal_mention_regex": REFUSAL_MENTION.pattern,
            "negation_regex": NEGATION.pattern,
            "negation_window_chars": NEGATION_WINDOW_CHARS,
            "high_score_threshold": HIGH_SCORE,
            "low_score_threshold": LOW_SCORE,
            "note": (
                "heuristic-dependent rates vary a few points with the negation window; "
                "the exact-share (reasoning, score) pair-overlap evidence is the "
                "decisive, heuristic-independent check"
            ),
        },
        "per_behavior": per_behavior,
        "graded_drop_rates": drops,
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(out, args.out)
    for b, v in per_behavior.items():
        logger.info(
            "[%s] high>=80: %d, flagged %.1f%%; flagged-pair-in-refusal %.1f%%; "
            "nonflagged-pair-in-refusal %s; draw-collapse %s",
            b,
            v["n_high_ge80"],
            100 * (v["frac_high_flagged"] or 0),
            100 * (v["frac_flagged_high_pairs_verbatim_in_refusal_file"] or 0),
            v["frac_nonflagged_high_pairs_verbatim_in_refusal_file"],
            v["draw_collapse"]["frac_single_distinct_score"],
        )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
