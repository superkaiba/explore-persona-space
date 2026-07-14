"""Issue #1310: fixed-label dialogue attribution + per-story (X,Y) pair build.

Deterministic extractor (precision-first): double-quote spans whose cue window
(before/after the quote) carries a speech verb adjacent to the story's FIXED
persona label (post-posed '"...," said Marlowe' / '"..." Marlowe said'; pre-
posed 'Marlowe said, "..."'). Label match is case-sensitive + word-bounded, so
all-caps labels like HELIOS attribute correctly (the #931 generic name regex
would miss them). Quotations NOT attributable to the fixed label are dropped
(counted).

Per (persona, story) -> ONE (X, Y) pair (issue1310_common.build_context_dialogue_pair):
C = context before the persona's first dialogue, T = the persona's dialogue
content. Every produced span is validated (0 <= s < e <= len) at build time;
short / no-dialogue / zero-width stories are DROPPED and REPORTED (never a hard
crash) — short degenerate generations WILL occur, especially for base.

Audit: a seeded random sample of persona-attributed quotes judged by
claude-sonnet-4-5-20250929 (sync via llm.api_dispatch); precision gate >= 0.90.
--reattribute-batch is the fallback: Sonnet judges whether the FIXED LABEL
speaks EVERY quote (Batch API), rebuild pairs under the confirmed set, re-gate
at >= 0.95.

CLI:
  uv run python scripts/issue1310_attribute.py --model base \
      [--data-dir data/issue_1310] [--out-dir eval_results/issue_1310]
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

import issue931_attribute as attr931  # noqa: E402
import issue931_common as common  # noqa: E402
import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1310_attribute.py"

_VERBS = "|".join(common.SPEECH_VERBS)
_QUOTE_RE = re.compile(r"[\"“]([^\"“”]+)[\"”]")

# Per-label cue patterns (cached; label escaped + word-bounded, case-sensitive).
_LABEL_PATTERNS: dict[str, dict[str, re.Pattern]] = {}


def _label_patterns(label: str) -> dict[str, re.Pattern]:
    if label not in _LABEL_PATTERNS:
        esc = re.escape(label)
        name = rf"(?<![\w]){esc}(?![\w])"
        _LABEL_PATTERNS[label] = {
            # after the quote: 'said Marlowe' / 'Marlowe said'
            "post_verb_name": re.compile(rf"^[\s,.;:!?-]*(?:{_VERBS})\s+{name}"),
            "post_name_verb": re.compile(rf"^[\s,.;:!?-]*{name}\s+(?:{_VERBS})\b"),
            # before the quote: 'Marlowe said,' / 'said Marlowe,'
            "pre_name_verb": re.compile(rf"{name}\s+(?:{_VERBS})[,:]?\s*$"),
            "pre_verb_name": re.compile(rf"(?:{_VERBS})\s+{name}[,:]?\s*$"),
        }
    return _LABEL_PATTERNS[label]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=c1310.MODEL_KINDS, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1310"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1310"))
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--audit-gate", type=float, default=0.90)
    ap.add_argument("--mock-judge", action="store_true", help="offline audit stub (smoke only)")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument(
        "--audit-non-binding",
        action="store_true",
        help="record the audit but never exit 4 on a gate miss (canary/tiny-n only)",
    )
    ap.add_argument(
        "--reattribute-batch",
        action="store_true",
        help="fallback: Sonnet Batch-API label-binary re-attribution, re-gate at 0.95",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Deterministic fixed-label extractor
# ---------------------------------------------------------------------------


def attribute_persona_story(text: str, persona_label: str) -> dict:
    """All double-quotes in one story + whether each is attributable to the label.

    Returns {"quotes": [{"span": (cs,ce) incl. delimiters, "content_span":
    (c2s,c2e) delimiters excluded, "attributed": bool}], "n_total",
    "n_attributed"}.
    """
    pats = _label_patterns(persona_label)
    quotes = []
    for m in _QUOTE_RE.finditer(text):
        cs, ce = m.start(), m.end()
        after = text[ce : ce + 60]
        before = text[max(0, cs - 60) : cs]
        attributed = bool(
            pats["post_verb_name"].match(after)
            or pats["post_name_verb"].match(after)
            or pats["pre_name_verb"].search(before)
            or pats["pre_verb_name"].search(before)
        )
        c2s, c2e = common.strip_quote_delims(text, cs, ce)
        quotes.append({"span": (cs, ce), "content_span": (c2s, c2e), "attributed": attributed})
    return {
        "quotes": quotes,
        "n_total": len(quotes),
        "n_attributed": sum(1 for q in quotes if q["attributed"]),
    }


def build_pairs(
    stories: list[dict], tokenizer, attributed_override: set | None = None
) -> tuple[list, list[dict], dict]:
    """(C,T) pair per (persona, story) + persona-attributed quote records + QC.

    ``attributed_override`` = a set of (row_id, (span_s, span_e)) confirmed as
    persona-spoken; when present it REPLACES the deterministic ``attributed``
    flag (the Batch-API re-attribution fallback).
    """
    counters = {
        "stories": len(stories),
        "quotes_total": 0,
        "quotes_unattributed_dropped": 0,
        "quotes_attributed": 0,
        "stories_with_pair": 0,
        "stories_dropped_no_pair": 0,
    }
    pairs: list[common.PairSpec] = []
    records: list[dict] = []
    for story in stories:
        text = story["story"]
        persona = story["persona"]
        row_id = story["row_id"]
        group_id = story["scenario_id"]
        att = attribute_persona_story(text, persona)
        if attributed_override is not None:
            for q in att["quotes"]:
                q["attributed"] = (row_id, tuple(q["span"])) in attributed_override
            att["n_attributed"] = sum(1 for q in att["quotes"] if q["attributed"])
        counters["quotes_total"] += att["n_total"]
        counters["quotes_unattributed_dropped"] += att["n_total"] - att["n_attributed"]
        counters["quotes_attributed"] += att["n_attributed"]
        attributed_quotes = [q for q in att["quotes"] if q["attributed"]]
        for q in attributed_quotes:
            records.append(
                {
                    "row_id": row_id,
                    "persona": persona,
                    "span": list(q["span"]),
                    "content_span": list(q["content_span"]),
                }
            )
        ids, offsets = common.tokenize_with_offsets(tokenizer, text)
        # Persona quotation token spans (cov incl. delims; inner content only).
        segs = []
        for q in attributed_quotes:
            cov_lo, cov_hi = common.covering_token_span(offsets, *q["span"])
            c2s, c2e = q["content_span"]
            in_lo, in_hi = common.inner_token_span(offsets, c2s, c2e) if c2s < c2e else (0, 0)
            if in_lo < in_hi:
                segs.append((cov_lo, cov_hi, in_lo, in_hi))
        built = c1310.build_context_dialogue_pair(n_tokens=len(ids), quote_spans_tok=segs)
        if built is None:
            counters["stories_dropped_no_pair"] += 1
            continue
        (c_s, c_e), t_spans = built
        pair = common.PairSpec(
            row_id=row_id,
            group_id=group_id,
            char_id=persona,
            c_span=(c_s, c_e),
            t_spans=list(t_spans),
            ctx_span=(c_s, c_e),  # context read == C (whole context before dialogue)
            meta={
                "window_id": row_id,
                "story_local": True,
                "n_t_tokens": int(sum(hi - lo for lo, hi in t_spans)),
                "n_turns": len(t_spans),
            },
        )
        pair.validate(len(ids), min_c=c1310.CONTEXT_MIN_TOKENS, min_t=c1310.DIALOGUE_MIN_TOKENS)
        pairs.append(pair)
        counters["stories_with_pair"] += 1
    return pairs, records, counters


# ---------------------------------------------------------------------------
# Sonnet audit + label-binary batch re-attribution fallback
# ---------------------------------------------------------------------------


def _reattr_build_request(item: DispatchItem) -> dict:
    p = item.payload
    return {
        "model": c1310.JUDGE_MODEL,
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


def run_audit(
    records: list[dict],
    stories_by_row: dict[str, dict],
    *,
    audit_n: int,
    mock: bool,
    cache_dir: Path,
) -> dict:
    """Judge a seeded sample of persona-attributed quotes; precision summary."""
    rng = np.random.default_rng(c1310.BUILD_SEED)
    order = rng.permutation(len(records))
    sample = [records[int(i)] for i in order[: min(audit_n, len(records))]]
    items = []
    for k, rec in enumerate(sample):
        story = stories_by_row[rec["row_id"]]
        text = story["story"]
        items.append(
            DispatchItem(
                item_id=f"audit_{k:04d}_{rec['row_id']}",
                payload={
                    "excerpt": attr931._excerpt(text, rec["span"]),
                    "quote_text": text[rec["content_span"][0] : rec["content_span"][1]],
                    "speaker": rec["persona"],
                },
            )
        )
    if mock:
        results = {it.item_id: {"correct": True} for it in items}
        dropped = 0
    else:
        raw = asyncio.run(
            dispatch_calls(
                items,
                model=c1310.JUDGE_MODEL,
                build_request=attr931._audit_build_request,
                parse_response=attr931._parse_json_obj,
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
    if sample and n_valid < 0.5 * len(sample):
        raise RuntimeError(
            f"audit degenerate: {n_valid}/{len(sample)} valid judge returns (<50%) — "
            "precision unmeasurable; refusing to continue past the audit gate"
        )
    return {
        "n_sampled": len(sample),
        "n_valid": n_valid,
        "n_dropped_malformed": dropped,
        "n_correct": n_correct,
        "precision": precision,
        "mock": bool(mock),
        "judge_model": c1310.JUDGE_MODEL,
    }


def reattribute_all(
    stories: list[dict], stories_by_row: dict[str, dict], *, mock: bool, cache_dir: Path
) -> set:
    """Sonnet label-binary re-attribution of EVERY quote -> confirmed set."""
    items = []
    for story in stories:
        text = story["story"]
        row_id = story["row_id"]
        persona = story["persona"]
        for k, m in enumerate(_QUOTE_RE.finditer(text)):
            cs, ce = m.start(), m.end()
            c2s, c2e = common.strip_quote_delims(text, cs, ce)
            items.append(
                DispatchItem(
                    item_id=f"reattr_{row_id}_{k:03d}",
                    payload={
                        "excerpt": attr931._excerpt(text, [cs, ce]),
                        "quote_text": text[c2s:c2e],
                        "speaker": persona,
                        "row_id": row_id,
                        "span": [cs, ce],
                    },
                )
            )
    if mock:
        confirmed = {(it.payload["row_id"], tuple(it.payload["span"])) for it in items}
        print(f"[i1310-attr] MOCK reattribution: {len(confirmed)} quotes confirmed")
        return confirmed
    raw = asyncio.run(
        dispatch_calls(
            items,
            model=c1310.JUDGE_MODEL,
            build_request=_reattr_build_request,
            parse_response=attr931._parse_json_obj,
            cost_pref="cost",
            cache_dir=cache_dir / "reattr",
            checkpoint_dir=cache_dir / "reattr_ckpt",
        )
    )
    by_id = {it.item_id: it for it in items}
    confirmed = set()
    for iid, res in raw.items():
        obj = res.result if not res.error else None
        if isinstance(obj, dict) and obj.get("correct") is True:
            it = by_id[iid]
            confirmed.add((it.payload["row_id"], tuple(it.payload["span"])))
    print(f"[i1310-attr] reattribution confirmed {len(confirmed)} label-spoken quotes")
    return confirmed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    print(f"[i1310-attr] wrote {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    model_kind = args.model
    print(f"[phase=p1_attr_{model_kind}] fixed-label attribution + pair build")
    stories_path = args.data_dir / "stories" / f"{model_kind}_stories_seed{common.GEN_SEED}.jsonl"
    stories = [json.loads(line) for line in stories_path.read_text().split("\n") if line.strip()]
    stories_by_row = {s["row_id"]: s for s in stories}
    tokenizer = common.get_tokenizer(c1310.MODEL_IDS[model_kind])

    pairs, records, counters = build_pairs(stories, tokenizer)
    drop_rate = counters["stories_dropped_no_pair"] / max(1, counters["stories"])
    print(
        f"[i1310-attr] {len(pairs)} pairs / {counters['stories']} stories "
        f"(dropped_no_pair={counters['stories_dropped_no_pair']}, rate={drop_rate:.3f})"
    )

    audit: dict = {"skipped": True}
    if not args.skip_audit and records:
        audit = run_audit(
            records,
            stories_by_row,
            audit_n=args.audit_n,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache" / model_kind,
        )
        print(f"[i1310-attr] audit precision={audit['precision']:.3f} (n={audit['n_valid']})")
        gate_miss = (
            not args.mock_judge
            and not np.isnan(audit["precision"])
            and audit["precision"] < args.audit_gate
        )
        if gate_miss and not args.reattribute_batch:
            if args.audit_non_binding:
                print(
                    f"[i1310-attr] WARNING: audit {audit['precision']:.3f} < "
                    f"{args.audit_gate} but --audit-non-binding — recorded, not gating",
                    file=sys.stderr,
                )
            else:
                c1310.write_json(
                    args.out_dir / f"attribution_audit_{model_kind}.json",
                    {
                        "metadata": common.metadata(SCRIPT, common.GEN_SEED, len(pairs)),
                        "model_kind": model_kind,
                        "counters": counters,
                        "drop_rate": drop_rate,
                        "audit": audit,
                        "gate": args.audit_gate,
                        "binding": True,
                        "pass": False,
                    },
                )
                print(
                    f"[i1310-attr] AUDIT FAIL {audit['precision']:.3f} < {args.audit_gate} — "
                    "re-run with --reattribute-batch (registered fallback)",
                    file=sys.stderr,
                )
                return 4

    if args.reattribute_batch:
        print(f"[phase=p1_reattr_{model_kind}] Batch-API label-binary re-attribution")
        confirmed = reattribute_all(
            stories,
            stories_by_row,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache" / model_kind,
        )
        pairs, records, counters = build_pairs(stories, tokenizer, attributed_override=confirmed)
        audit = run_audit(
            records,
            stories_by_row,
            audit_n=100,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache" / model_kind / "regate",
        )
        assert args.mock_judge or np.isnan(audit["precision"]) or audit["precision"] >= 0.95, (
            f"re-attribution re-gate failed: {audit['precision']:.3f} < 0.95"
        )

    _write_jsonl(
        args.data_dir / "pairs" / f"{model_kind}_pairs.jsonl", [p.to_dict() for p in pairs]
    )
    c1310.write_json(
        args.out_dir / f"attribution_audit_{model_kind}.json",
        {
            "metadata": common.metadata(SCRIPT, common.GEN_SEED, len(pairs)),
            "model_kind": model_kind,
            "counters": counters,
            "drop_rate": drop_rate,
            "per_persona_pairs": {
                p: sum(1 for pair in pairs if pair.char_id == p) for p in c1310.PERSONA_LABELS
            },
            "audit": audit,
            "gate": args.audit_gate,
            "binding": not args.audit_non_binding,
            "pass": bool(audit.get("skipped") or args.mock_judge)
            or (
                not np.isnan(audit.get("precision", float("nan")))
                and audit["precision"] >= args.audit_gate
            ),
        },
    )
    print(f"[i1310-attr] done ({model_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
