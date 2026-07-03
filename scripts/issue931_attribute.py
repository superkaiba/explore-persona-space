"""Issue #931: Arm-B dialogue attribution (deterministic extractor) + judge audit.

Extractor (plan section 4.2, precision-first): double-quote spans; speaker cue
= a speech verb + capitalized name adjacent to the quote (post-posed
'"...," said Maria.' / '"..." Maria said.' / pre-posed 'Maria said, "..."');
UNATTRIBUTED quotations are DROPPED (counts reported). Characters with >= 2
attributed quotes are eligible; (C, T) pairs are then built with the SAME
section-4.0 recipe as Arm A over the story tokens (story-local indices; the
extractor records them relative to the story text — the capture phase adds the
chat-template prefix offset).

Audit: a seeded random sample of attributed quotes judged by
claude-sonnet-4-5-20250929 (sync via llm.api_dispatch); precision gate >= 0.90.
--reattribute-batch is the registered fallback: full Sonnet re-attribution via
the Batch API (cost_pref="cost"), then re-gate at >= 0.95 on a fresh sample.

CLI:
  uv run python scripts/issue931_attribute.py \
      [--stories data/issue_931/stories/stories_seed42.jsonl] \
      [--data-dir data/issue_931] [--out-dir eval_results/issue_931] \
      [--audit-n 200] [--audit-gate 0.90] [--mock-judge] [--reattribute-batch]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    DispatchItem,
    dispatch_calls,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_attribute.py"

_VERBS = "|".join(common.SPEECH_VERBS)
# A capitalized name: 1-2 capitalized words (e.g. "Maria", "Old Tom").
_NAME = r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
_QUOTE_RE = re.compile(r"[\"“]([^\"“”]+)[\"”]")
# Post-posed: '"...," said Maria' / '"..." said Maria'
_POST_VERB_NAME = re.compile(rf"^[\s,.;:!?-]*(?:{_VERBS})\s+({_NAME})")
# Post-posed inverted: '"..." Maria said'
_POST_NAME_VERB = re.compile(rf"^[\s,.;:!?-]*({_NAME})\s+(?:{_VERBS})\b")
# Pre-posed: 'Maria said, "..."' / 'said Maria: "..."'
_PRE_NAME_VERB = re.compile(rf"({_NAME})\s+(?:{_VERBS})[,:]?\s*$")
_PRE_VERB_NAME = re.compile(rf"(?:{_VERBS})\s+({_NAME})[,:]?\s*$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stories", type=Path, default=Path("data/issue_931/stories/stories_seed42.jsonl")
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_931"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--audit-gate", type=float, default=0.90)
    ap.add_argument("--mock-judge", action="store_true", help="offline audit stub (smoke only)")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument(
        "--reattribute-batch",
        action="store_true",
        help="contingency: full Sonnet re-attribution via the Batch API, re-gate at 0.95",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Deterministic extractor
# ---------------------------------------------------------------------------


def attribute_story(text: str) -> dict:
    """Extract attributed quotations from one story.

    Returns {"quotes": [{"span": (cs, ce) char span INCLUDING delimiters,
    "content_span": (cs2, ce2) delimiters excluded, "speaker": str | None}],
    "n_total", "n_attributed"}.
    """
    quotes = []
    for m in _QUOTE_RE.finditer(text):
        cs, ce = m.start(), m.end()
        after = text[ce : ce + 60]
        before = text[max(0, cs - 60) : cs]
        speaker = None
        for pat in (_POST_VERB_NAME, _POST_NAME_VERB):
            pm = pat.match(after)
            if pm:
                speaker = pm.group(1)
                break
        if speaker is None:
            for pat in (_PRE_NAME_VERB, _PRE_VERB_NAME):
                pm = pat.search(before)
                if pm:
                    speaker = pm.group(1)
                    break
        c2s, c2e = common.strip_quote_delims(text, cs, ce)
        quotes.append({"span": (cs, ce), "content_span": (c2s, c2e), "speaker": speaker})
    return {
        "quotes": quotes,
        "n_total": len(quotes),
        "n_attributed": sum(1 for q in quotes if q["speaker"]),
    }


def build_armb_pairs(  # noqa: C901 -- linear extractor + pair loop
    stories: list[dict], tokenizer, speaker_override: dict | None = None
) -> tuple[list, list[dict], dict]:
    """(C, T) pairs per story (story-local token indices) + QC counters.

    ``speaker_override`` maps (prompt_id, (span_s, span_e)) -> speaker name;
    it REPLACES the extractor's attribution for that quotation (the Batch-API
    re-attribution fallback path). Returns (pairs, ALL quote records incl.
    unattributed ones with speaker=None, counters).
    """
    counters = {
        "stories": len(stories),
        "quotes_total": 0,
        "quotes_unattributed_dropped": 0,
        "quotes_attributed": 0,
        "stories_with_pairs": 0,
        "chars_eligible": 0,
    }
    pairs: list[common.PairSpec] = []
    all_records: list[dict] = []
    for story in stories:
        text = story["story"]
        prompt_id = story["prompt_id"]
        att = attribute_story(text)
        if speaker_override:
            for q in att["quotes"]:
                key = (prompt_id, tuple(q["span"]))
                if key in speaker_override:
                    q["speaker"] = speaker_override[key]
            att["n_attributed"] = sum(1 for q in att["quotes"] if q["speaker"])
        counters["quotes_total"] += att["n_total"]
        counters["quotes_unattributed_dropped"] += att["n_total"] - att["n_attributed"]
        counters["quotes_attributed"] += att["n_attributed"]
        for q in att["quotes"]:
            all_records.append(
                {
                    "prompt_id": prompt_id,
                    "span": list(q["span"]),
                    "content_span": list(q["content_span"]),
                    "speaker": q["speaker"],
                }
            )
        by_char: dict[str, list[dict]] = {}
        for q in att["quotes"]:
            if q["speaker"]:
                by_char.setdefault(q["speaker"], []).append(q)
        ids, offsets = common.tokenize_with_offsets(tokenizer, text)
        if len(ids) < common.INTRO_MIN_TOKENS + common.TARGET_MIN_TOKENS:
            continue
        bounds = common.sentence_bounds(text)
        story_pairs = []
        for speaker, qs in by_char.items():
            if len(qs) < 2:  # eligibility: >= 2 attributed quotes
                continue
            counters["chars_eligible"] += 1
            segs = []
            drop_char = False
            for q in qs:
                cov_lo, cov_hi = common.covering_token_span(offsets, *q["span"])
                c2s, c2e = q["content_span"]
                in_lo, in_hi = common.inner_token_span(offsets, c2s, c2e) if c2s < c2e else (0, 0)
                if in_lo >= in_hi:
                    drop_char = True
                    break
                segs.append((cov_lo, cov_hi, in_lo, in_hi))
            if drop_char or not segs:
                continue
            segs.sort(key=lambda s: s[0])
            mention = re.search(rf"(?<![\w]){re.escape(speaker)}(?![\w])", text)
            if mention is None:
                continue
            built = common.build_intro_and_targets(
                text=text,
                offsets=offsets,
                excerpt_tok=(0, len(ids)),
                mention_char=mention.start(),
                quote_spans_tok=segs,
                bounds=bounds,
            )
            if built is None:
                continue
            (c_s, c_e), t_spans = built
            t_min = min(lo for lo, _ in t_spans)
            pair = common.PairSpec(
                row_id=f"{prompt_id}:{speaker}",
                group_id=prompt_id,
                char_id=speaker,
                c_span=(c_s, c_e),
                t_spans=list(t_spans),
                ctx_span=(0, t_min),
                meta={
                    "window_id": prompt_id,
                    "story_local": True,
                    "n_t_tokens": int(sum(hi - lo for lo, hi in t_spans)),
                },
            )
            pair.validate(len(ids))
            story_pairs.append(pair)
        if story_pairs:
            counters["stories_with_pairs"] += 1
            pairs.extend(story_pairs)
    return pairs, all_records, counters


# ---------------------------------------------------------------------------
# Sonnet audit (sync, 200 calls) + batch re-attribution contingency
# ---------------------------------------------------------------------------


def _excerpt(text: str, span: list[int], radius: int = 1200) -> str:
    lo = max(0, span[0] - radius)
    hi = min(len(text), span[1] + radius)
    return text[lo:hi]


def _audit_build_request(item: DispatchItem) -> dict:
    p = item.payload
    return {
        "model": common.JUDGE_MODEL,
        "max_tokens": 300,
        "system": "You are a careful literary annotator verifying dialogue attribution.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Story excerpt:\n---\n"
                    + p["excerpt"]
                    + "\n---\n\nQuotation: "
                    + json.dumps(p["quote_text"])
                    + "\nCandidate speaker: "
                    + p["speaker"]
                    + "\n\nIs the candidate speaker the character who utters this quotation "
                    "in the excerpt? Think briefly, then answer with ONLY a JSON object: "
                    '{"reasoning": "<one sentence>", "correct": true|false}'
                ),
            }
        ],
    }


def _reattr_build_request(item: DispatchItem) -> dict:
    p = item.payload
    return {
        "model": common.JUDGE_MODEL,
        "max_tokens": 300,
        "system": "You are a careful literary annotator attributing dialogue to speakers.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Story excerpt:\n---\n"
                    + p["excerpt"]
                    + "\n---\n\nQuotation: "
                    + json.dumps(p["quote_text"])
                    + "\n\nWhich named character utters this quotation? Answer with ONLY a "
                    'JSON object: {"reasoning": "<one sentence>", "speaker": "<name or '
                    'UNKNOWN>"}'
                ),
            }
        ],
    }


def _parse_json_obj(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")
    return json.loads(m.group(0))


def run_audit(
    records: list[dict],
    stories_by_id: dict[str, str],
    *,
    audit_n: int,
    mock: bool,
    cache_dir: Path,
) -> dict:
    """Judge a seeded sample of attributed quotes; return precision summary."""
    rng = np.random.default_rng(common.BUILD_SEED)
    order = rng.permutation(len(records))
    sample = [records[int(i)] for i in order[: min(audit_n, len(records))]]
    items = []
    for k, rec in enumerate(sample):
        text = stories_by_id[rec["prompt_id"]]
        items.append(
            DispatchItem(
                item_id=f"audit_{k:04d}_{rec['prompt_id']}",
                payload={
                    "excerpt": _excerpt(text, rec["span"]),
                    "quote_text": text[rec["content_span"][0] : rec["content_span"][1]],
                    "speaker": rec["speaker"],
                },
            )
        )
    if mock:
        # SMOKE ONLY: offline stub — every attribution judged correct.
        results = {it.item_id: {"correct": True} for it in items}
        dropped = 0
    else:
        raw = asyncio.run(
            dispatch_calls(
                items,
                model=common.JUDGE_MODEL,
                build_request=_audit_build_request,
                parse_response=_parse_json_obj,
                cost_pref="latency",
                cache_dir=cache_dir / "audit",
                checkpoint_dir=cache_dir / "audit_ckpt",
            )
        )
        results, dropped = {}, 0
        for iid, res in raw.items():
            obj = res.result if not res.error else None
            # DROP malformed / non-bool returns, never coerce (llm-judging r9).
            if isinstance(obj, dict) and isinstance(obj.get("correct"), bool):
                results[iid] = obj
            else:
                dropped += 1
    n_valid = len(results)
    n_correct = sum(1 for v in results.values() if v["correct"])
    precision = n_correct / n_valid if n_valid else float("nan")
    return {
        "n_sampled": len(sample),
        "n_valid": n_valid,
        "n_dropped_malformed": dropped,
        "n_correct": n_correct,
        "precision": precision,
        "mock": bool(mock),
        "judge_model": common.JUDGE_MODEL,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"[i931-attr] wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    print("[phase=p1_attr] Arm-B attribution")
    stories = [json.loads(line) for line in args.stories.read_text().splitlines() if line.strip()]
    stories_by_id = {s["prompt_id"]: s["story"] for s in stories}
    tokenizer = common.get_tokenizer()

    pairs, all_records, counters = build_armb_pairs(stories, tokenizer)
    attributed = [r for r in all_records if r["speaker"]]
    print(f"[i931-attr] {len(pairs)} pairs; counters={counters}")

    audit: dict = {"skipped": True}
    if not args.skip_audit and attributed:
        audit = run_audit(
            attributed,
            stories_by_id,
            audit_n=args.audit_n,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache",
        )
        print(f"[i931-attr] audit precision={audit['precision']:.3f} (n={audit['n_valid']})")
        if (
            not args.mock_judge
            and not np.isnan(audit["precision"])
            and audit["precision"] < args.audit_gate
            and not args.reattribute_batch
        ):
            common.write_json(
                args.out_dir / "attribution_audit.json",
                {
                    "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(pairs)),
                    "counters": counters,
                    "audit": audit,
                    "gate": args.audit_gate,
                    "pass": False,
                },
            )
            print(
                f"[i931-attr] AUDIT FAIL {audit['precision']:.3f} < {args.audit_gate} — "
                "re-run with --reattribute-batch (registered fallback)",
                file=sys.stderr,
            )
            return 4

    if args.reattribute_batch:
        # Contingency (plan 4.2): full Sonnet re-attribution via the Batch API
        # of EVERY quotation (attributed + unattributed), then a pair rebuild
        # under the corrected speakers + a fresh 100-quote re-gate at 0.95.
        print("[phase=p1_reattr] Batch-API re-attribution of ALL quotations")
        items = []
        for k, rec in enumerate(all_records):
            text = stories_by_id[rec["prompt_id"]]
            items.append(
                DispatchItem(
                    item_id=f"reattr_{k:05d}_{rec['prompt_id']}",
                    payload={
                        "excerpt": _excerpt(text, rec["span"]),
                        "quote_text": text[rec["content_span"][0] : rec["content_span"][1]],
                    },
                )
            )
        raw = asyncio.run(
            dispatch_calls(
                items,
                model=common.JUDGE_MODEL,
                build_request=_reattr_build_request,
                parse_response=_parse_json_obj,
                cost_pref="cost",
                cache_dir=args.data_dir / "judge_cache" / "reattr",
                checkpoint_dir=args.data_dir / "judge_cache" / "reattr_ckpt",
            )
        )
        override: dict[tuple, str] = {}
        for k, rec in enumerate(all_records):
            res = raw.get(f"reattr_{k:05d}_{rec['prompt_id']}")
            obj = res.result if (res and not res.error) else None
            if isinstance(obj, dict) and isinstance(obj.get("speaker"), str):
                sp = obj["speaker"].strip()
                if sp and sp.upper() != "UNKNOWN":
                    override[(rec["prompt_id"], tuple(rec["span"]))] = sp
        print(f"[i931-attr] re-attribution resolved {len(override)} speakers")
        pairs, all_records, counters = build_armb_pairs(
            stories, tokenizer, speaker_override=override
        )
        attributed = [r for r in all_records if r["speaker"]]
        audit = run_audit(
            attributed,
            stories_by_id,
            audit_n=100,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache" / "regate",
        )
        assert args.mock_judge or np.isnan(audit["precision"]) or audit["precision"] >= 0.95, (
            f"re-attribution re-gate failed: {audit['precision']:.3f} < 0.95"
        )

    pairs_dir = args.data_dir / "pairs"
    _write_jsonl(pairs_dir / "pairs_armB.jsonl", [p.to_dict() for p in pairs])
    common.write_json(
        args.out_dir / "attribution_audit.json",
        {
            "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(pairs)),
            "counters": counters,
            "audit": audit,
            "gate": args.audit_gate,
            "pass": bool(audit.get("skipped") or args.mock_judge)
            or (
                not np.isnan(audit.get("precision", float("nan")))
                and audit["precision"] >= args.audit_gate
            ),
        },
    )
    print("[i931-attr] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
