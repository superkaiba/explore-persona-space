#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (ĉ, ρ, →, ×) in scientific docstrings + log messages.
"""Issue #661 P2 (off-pod, Anthropic Batch API): Sonnet-4.5 judge-filter arm A.

Scores every arm-A rollout with the per-behavior Persona Vectors 0-100
EVALUATION PROMPT (frozen in P0's ``instructions_<behavior>.json["eval_prompt"]``)
via the Anthropic Message Batches API, then applies the Persona Vectors filter:

- POSITIVE-instruction rollouts kept iff score > 50 (the behavior IS expressed).
- NEGATIVE-instruction rollouts kept iff score < 50 (the behavior is ABSENT).

A ``REFUSAL`` verdict or an unparseable score is dropped (never counted as a
survivor). The survivor set (the (instruction_idx, probe_idx, rollout_idx) tuples
+ the surviving completion TEXT) is what P3 teacher-forces for r_B^A / r_B^C.

Reuses ``issue658_judge_e0_batch.submit_and_collect`` for the Batch-API
transport (chunked submit + poll + collect). The judge MODEL + the
direct-0-100 SCORING MECHANISM are the two named deviations from the paper
(plan §12 A4); the threshold operates on the 0-100 scale itself so it is
judge-agnostic in operation.

Survivor counts are reported per (behavior, polarity); the §7 kill criterion
(<5 survivors in either pool → drop that behavior) is enforced downstream by P5
(this script only LABELS; it never silently discards a behavior).

Content hygiene: sycophancy / refusal / EM rollouts are harmful-content-adjacent
— this script digests by COUNTS + scores only and never logs completion text.

Usage::

    uv run python scripts/issue661_judge_filter.py \
        --behaviors sycophancy refusal broad_em \
        --raw-dir eval_results/issue_661/raw_completions \
        --instructions-dir data/issue_661 \
        --out eval_results/issue_661/judge_filter.json

    # smoke (2 rollouts sanity, no real Batch needed if --max-cells small)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402

# Reuse the #658 Batch-API transport (hoisted to module top, gotchas.md #606).
from issue658_judge_e0_batch import submit_and_collect  # noqa: E402
from issue661_common import (  # noqa: E402
    JUDGE_MODEL,
    JUDGE_NEG_THRESHOLD,
    JUDGE_POS_THRESHOLD,
    dump_json,
    instructions_path,
    load_json,
)

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue661_judge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_score(text: str) -> int | None:
    """Parse the judge's 0-100 score from its reply.

    Returns None for a REFUSAL verdict or an unparseable reply. The PV eval
    prompt asks for "a number between 0 and 100" with nothing else, so the
    first integer in [0, 100] is the score. A leading 'REFUSAL' (case-
    insensitive) short-circuits to None.
    """
    if text is None:
        return None
    if "refusal" in text.strip().lower()[:16]:
        return None
    for m in re.finditer(r"-?\d+", text):
        v = int(m.group())
        if 0 <= v <= 100:
            return v
    return None


def build_judge_requests(
    behavior: str, polarity: str, gen: dict, eval_prompt: str, cid_prefix: str
) -> tuple[list[dict], dict[str, tuple[int, int, int]]]:
    """One judge request per rollout. Returns (requests, cid -> (ci, ri, ki)).

    cid encodes (cell_index, rollout_index) compactly (Batch custom_id must
    match ^[a-zA-Z0-9_-]{1,64}$). Each request fills the PV eval prompt's
    {question}/{answer} with the probe + rollout text.
    """
    requests: list[dict] = []
    cid_map: dict[str, tuple[int, int, int]] = {}
    nid = 0
    for ci, cell in enumerate(gen["cells"]):
        for ki, roll in enumerate(cell["rollouts"]):
            cid = f"{cid_prefix}_{nid}"
            nid += 1
            cid_map[cid] = (ci, cell["instruction_idx"], ki)
            prompt = eval_prompt.replace("{question}", cell["probe"]).replace(
                "{answer}", roll["text"]
            )
            requests.append({"custom_id": cid, "prompt": prompt})
    return requests, cid_map


def filter_behavior(
    behavior: str,
    *,
    raw_dir: Path,
    instructions_dir: Path,
    verdict_overrides: dict[str, int] | None = None,
) -> dict:
    """Judge + filter both polarities for one behavior.

    Returns the survivor record:
      {behavior, eval_prompt_sha, pos: {n_judged, n_survivors, survivors:[...]},
       neg: {...}} where each survivor carries (instruction_idx, probe_idx,
       rollout_idx, score, text).

    ``verdict_overrides`` (smoke/test only): {cid: score} to bypass the Batch
    API. Never used in production.
    """
    instr = load_json(
        (instructions_dir / f"instructions_{behavior}.json")
        if instructions_dir
        else instructions_path(behavior)
    )
    eval_prompt = instr["eval_prompt"]

    all_requests: list[dict] = []
    cid_maps: dict[str, dict[str, tuple[int, int, int]]] = {}
    gens: dict[str, dict] = {}
    for polarity in ("pos", "neg"):
        gen = load_json(raw_dir / f"{behavior}__{polarity}.json")
        gens[polarity] = gen
        reqs, cmap = build_judge_requests(
            behavior, polarity, gen, eval_prompt, cid_prefix=f"{behavior[:6]}{polarity}"
        )
        all_requests.extend(reqs)
        cid_maps[polarity] = cmap

    if verdict_overrides is not None:
        verdicts_raw = {cid: {"_score": verdict_overrides.get(cid)} for cid in verdict_overrides}
    else:
        # submit_and_collect returns {custom_id: verdict_dict} where the dict is
        # the parsed JSON object OR {_judge_error/_judge_refused: ...}. The PV
        # eval prompt returns a bare number, so we re-parse the raw text — but
        # submit_and_collect already JSON-parses; route around it by reading the
        # raw text it stored. Since the reply is a bare int (not JSON), the #658
        # parser yields {_judge_error: "<text>"} carrying the number — parse it.
        raw = submit_and_collect(all_requests, JUDGE_MODEL)
        verdicts_raw = {}
        for cid, v in raw.items():
            if "_judge_refused" in v:
                verdicts_raw[cid] = {"_score": None}
            elif "_judge_error" in v:
                # The bare-number reply lands here (not valid JSON object) —
                # the original text is in _judge_error.
                verdicts_raw[cid] = {"_score": parse_score(v["_judge_error"])}
            else:
                # Defensive: a JSON object reply with a "score" key.
                verdicts_raw[cid] = {"_score": parse_score(str(v.get("score", v)))}

    result: dict = {"behavior": behavior, "eval_prompt_sha": instr.get("sha256")}
    for polarity in ("pos", "neg"):
        cmap = cid_maps[polarity]
        gen = gens[polarity]
        survivors = []
        n_judged = 0
        threshold_kept = JUDGE_POS_THRESHOLD if polarity == "pos" else JUDGE_NEG_THRESHOLD
        for cid, (ci, instruction_idx, ki) in cmap.items():
            v = verdicts_raw.get(cid, {"_score": None})
            score = v.get("_score")
            if score is None:
                continue  # refusal / unparseable — dropped
            n_judged += 1
            keep = (score > threshold_kept) if polarity == "pos" else (score < threshold_kept)
            if keep:
                cell = gen["cells"][ci]
                survivors.append(
                    {
                        "instruction_idx": instruction_idx,
                        "probe_idx": cell["probe_idx"],
                        "probe": cell["probe"],
                        "rollout_idx": ki,
                        "score": score,
                        "text": cell["rollouts"][ki]["text"],
                    }
                )
        result[polarity] = {
            "n_judged": n_judged,
            "n_survivors": len(survivors),
            "threshold": threshold_kept,
            "survivors": survivors,
        }
        logger.info(
            "%s/%s: %d survivors of %d judged (threshold %s%d)",
            behavior,
            polarity,
            len(survivors),
            n_judged,
            ">" if polarity == "pos" else "<",
            threshold_kept,
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #661 P2: Sonnet-4.5 judge-filter arm A.")
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy", "refusal", "broad_em"])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--instructions-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results = {}
    for behavior in args.behaviors:
        results[behavior] = filter_behavior(
            behavior, raw_dir=args.raw_dir, instructions_dir=args.instructions_dir
        )

    payload = {
        "judge_model": JUDGE_MODEL,
        "transport": "anthropic_batch_api",
        "filter": {
            "pos": f"score > {JUDGE_POS_THRESHOLD}",
            "neg": f"score < {JUDGE_NEG_THRESHOLD}",
        },
        "behaviors": results,
        "metadata": reproducibility_metadata({"script": "issue661_judge_filter"}),
    }
    dump_json(payload, args.out)
    for b, r in results.items():
        logger.info(
            "FILTERED %s: pos %d survivors, neg %d survivors",
            b,
            r["pos"]["n_survivors"],
            r["neg"]["n_survivors"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
