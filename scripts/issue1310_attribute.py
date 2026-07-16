"""Issue #1310: fixed-label SCRIPT-LINE attribution + per-turn (X,Y) pair build.

Deterministic attributor: the scene is generated in strict script format
(`<LABEL>: <dialogue>`, one turn per line), so a target turn is simply a line
whose prefix is the persona's FIXED label — `^<LABEL>:[ \t]*<content>$`. Label
match is exact + case-sensitive, so all-caps labels like HELIOS attribute
correctly. Recall is ~100% by construction (every well-formed target line
matches); the Sonnet audit is a PRECISION spot-check only, and
`--reattribute-batch` is a non-primary fallback (line-prefix attribution needs
no judge to build pairs).

Per attributed TURN of the target persona -> ONE (X, Y) pair
(issue1310_common.build_turn_pairs): X = context tokens before the turn's line
(excluding the turn's own `<LABEL>:` cue), Y = the turn's dialogue content. MANY
points per persona per scene. Every produced span is validated (0 <= s < e <=
len) at build time; short / zero-width / no-context turns are DROPPED and
REPORTED (never a hard crash) — short degenerate lines WILL occur, especially
for base. Folds group by SCENE (group_id = scenario_id; within a persona each
scenario is one scene), so turns from one scene never split across train/test.

QC: per-persona attributed turn counts + non-target `Name:` (foil/other) lines +
lines that do not parse to a `Name: content` turn at all (narration/malformed).

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

# Per-label target-turn pattern (cached; label escaped + exact, case-sensitive).
_TARGET_PATTERNS: dict[str, re.Pattern] = {}
# Generic `Name: content` line matcher for QC classification (no leading space,
# a non-empty non-space content start — mirrors the target pattern's content req).
_GENERIC_LINE_RE = re.compile(r"^(?P<name>[^\n:]{1,40}):[ \t]*(?P<c>\S.*)$")


def _target_pattern(label: str) -> re.Pattern:
    if label not in _TARGET_PATTERNS:
        _TARGET_PATTERNS[label] = re.compile(
            rf"^{re.escape(label)}:[ \t]*(?P<c>\S.*)$", re.MULTILINE
        )
    return _TARGET_PATTERNS[label]


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
        help="non-primary fallback: Sonnet Batch-API label-binary re-confirm, re-gate 0.95",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Deterministic fixed-label script-line attributor
# ---------------------------------------------------------------------------


def attribute_script(text: str, persona_label: str) -> dict:
    """The persona's attributed turns (line-prefix `<LABEL>:`) in scene order.

    Returns {"turns": [{"line_start": int, "line_span": (ls, le),
    "content_span": (cs, ce) delimiters/whitespace stripped}], "n_turns": int}.
    """
    pat = _target_pattern(persona_label)
    turns = []
    for m in pat.finditer(text):
        cs, ce = common.strip_quote_delims(text, m.start("c"), m.end("c"))
        turns.append(
            {"line_start": m.start(), "line_span": (m.start(), m.end()), "content_span": (cs, ce)}
        )
    turns.sort(key=lambda t: t["line_start"])
    return {"turns": turns, "n_turns": len(turns)}


def _line_class_counts(text: str, target_label: str) -> tuple[int, int, int]:
    """(target-label lines, non-target `Name:` lines, unparsed non-empty lines)."""
    n_target = n_foil_other = n_unparsed = 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        m = _GENERIC_LINE_RE.match(line)
        if not m:
            n_unparsed += 1
            continue
        if m.group("name").strip() == target_label:
            n_target += 1
        else:
            n_foil_other += 1
    return n_target, n_foil_other, n_unparsed


def build_turn_records_and_pairs(
    stories: list[dict], tokenizer, confirmed_row_ids: set | None = None
) -> tuple[list, list[dict], dict, dict]:
    """Per-turn PairSpecs + per-turn audit records + QC counters + per-persona pairs.

    ``confirmed_row_ids`` = a set of turn row_ids confirmed persona-spoken; when
    present only those turns become pairs (the Batch-API re-attribution fallback).
    """
    counters = {
        "stories": len(stories),
        "target_lines": 0,
        "foil_other_lines": 0,
        "unparsed_lines": 0,
        "turns_kept": 0,
        "turns_dropped_zero_width": 0,
        "turns_dropped_short_context": 0,
        "turns_dropped_short_dialogue": 0,
        "turns_dropped_unconfirmed": 0,
        "stories_with_pair": 0,
        "stories_dropped_no_pair": 0,
    }
    per_persona_pairs = {label: 0 for label in c1310.PERSONA_LABELS}
    per_persona_target_lines = {label: 0 for label in c1310.PERSONA_LABELS}
    pairs: list[common.PairSpec] = []
    records: list[dict] = []
    for story in stories:
        text = story["story"]
        persona = story["persona"]
        scene_row_id = story["row_id"]
        scenario_id = story["scenario_id"]
        att = attribute_script(text, persona)
        counters["target_lines"] += att["n_turns"]
        per_persona_target_lines[persona] += att["n_turns"]
        _, n_foil, n_unparsed = _line_class_counts(text, persona)
        counters["foil_other_lines"] += n_foil
        counters["unparsed_lines"] += n_unparsed

        turns_char = [
            (t["line_start"], t["content_span"][0], t["content_span"][1]) for t in att["turns"]
        ]
        ids, offsets = common.tokenize_with_offsets(tokenizer, text)
        built, drops = c1310.build_turn_pairs(n_tokens=len(ids), offsets=offsets, turns=turns_char)
        counters["turns_dropped_zero_width"] += drops["dropped_zero_width"]
        counters["turns_dropped_short_context"] += drops["dropped_short_context"]
        counters["turns_dropped_short_dialogue"] += drops["dropped_short_dialogue"]

        kept_for_scene = 0
        for turn_index, c_span, t_span in built:
            rid = c1310.turn_row_id(scene_row_id, turn_index)
            if confirmed_row_ids is not None and rid not in confirmed_row_ids:
                counters["turns_dropped_unconfirmed"] += 1
                continue
            pair = common.PairSpec(
                row_id=rid,
                group_id=scenario_id,  # SCENE-grouped folds (one scene per scenario per persona)
                char_id=persona,
                c_span=c_span,
                t_spans=[t_span],
                ctx_span=c_span,  # context read == C (whole context before the turn)
                meta={
                    "scene_row_id": scene_row_id,
                    "scenario_id": scenario_id,
                    "turn_index": turn_index,
                    "story_local": True,
                },
            )
            pair.validate(len(ids), min_c=c1310.CONTEXT_MIN_TOKENS, min_t=c1310.DIALOGUE_MIN_TOKENS)
            pairs.append(pair)
            per_persona_pairs[persona] += 1
            kept_for_scene += 1
            turn = att["turns"][turn_index]
            records.append(
                {
                    "row_id": rid,
                    "scene_row_id": scene_row_id,
                    "persona": persona,
                    "span": list(turn["line_span"]),
                    "content_span": list(turn["content_span"]),
                }
            )
        counters["turns_kept"] += kept_for_scene
        if kept_for_scene >= c1310.MIN_SCENE_TURNS:
            counters["stories_with_pair"] += 1
        else:
            counters["stories_dropped_no_pair"] += 1

    counters["per_persona_target_lines"] = per_persona_target_lines
    return pairs, records, counters, per_persona_pairs


# ---------------------------------------------------------------------------
# Sonnet audit (precision spot-check) + label-binary batch re-confirm fallback
# ---------------------------------------------------------------------------


def run_audit(
    records: list[dict],
    stories_by_scene: dict[str, dict],
    *,
    audit_n: int,
    mock: bool,
    cache_dir: Path,
) -> dict:
    """Judge a seeded sample of persona-attributed turns; precision summary."""
    rng = np.random.default_rng(c1310.BUILD_SEED)
    order = rng.permutation(len(records))
    sample = [records[int(i)] for i in order[: min(audit_n, len(records))]]
    items = []
    for k, rec in enumerate(sample):
        text = stories_by_scene[rec["scene_row_id"]]["story"]
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


def reattribute_confirm(stories: list[dict], *, mock: bool, cache_dir: Path) -> set:
    """Sonnet label-binary re-confirm of EVERY target turn -> confirmed row_id set."""
    items = []
    for story in stories:
        text = story["story"]
        persona = story["persona"]
        scene_row_id = story["row_id"]
        att = attribute_script(text, persona)
        for turn_index, turn in enumerate(att["turns"]):
            rid = c1310.turn_row_id(scene_row_id, turn_index)
            items.append(
                DispatchItem(
                    item_id=f"reattr_{rid}",
                    payload={
                        "excerpt": attr931._excerpt(text, list(turn["line_span"])),
                        "quote_text": text[turn["content_span"][0] : turn["content_span"][1]],
                        "speaker": persona,
                        "row_id": rid,
                    },
                )
            )
    if mock:
        confirmed = {it.payload["row_id"] for it in items}
        print(f"[i1310-attr] MOCK reattribution: {len(confirmed)} turns confirmed")
        return confirmed
    raw = asyncio.run(
        dispatch_calls(
            items,
            model=c1310.JUDGE_MODEL,
            build_request=attr931._audit_build_request,
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
            confirmed.add(by_id[iid].payload["row_id"])
    print(f"[i1310-attr] reattribution confirmed {len(confirmed)} label-spoken turns")
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
    print(f"[phase=p1_attr_{model_kind}] fixed-label script-line attribution + turn pairs")
    stories_path = args.data_dir / "stories" / f"{model_kind}_stories_seed{common.GEN_SEED}.jsonl"
    stories = [json.loads(line) for line in stories_path.read_text().split("\n") if line.strip()]
    stories_by_scene = {s["row_id"]: s for s in stories}
    tokenizer = common.get_tokenizer(c1310.MODEL_IDS[model_kind])

    pairs, records, counters, per_persona_pairs = build_turn_records_and_pairs(stories, tokenizer)
    drop_rate = counters["stories_dropped_no_pair"] / max(1, counters["stories"])
    print(
        f"[i1310-attr] {len(pairs)} turn-pairs / {counters['stories']} scenes "
        f"(target_lines={counters['target_lines']} kept={counters['turns_kept']} "
        f"foil_other={counters['foil_other_lines']} unparsed={counters['unparsed_lines']} "
        f"scenes_no_pair={counters['stories_dropped_no_pair']} rate={drop_rate:.3f})"
    )
    print(f"[i1310-attr] per-persona turn-pairs: {per_persona_pairs}")

    audit: dict = {"skipped": True}
    if not args.skip_audit and records:
        audit = run_audit(
            records,
            stories_by_scene,
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
                        "per_persona_pairs": per_persona_pairs,
                        "audit": audit,
                        "gate": args.audit_gate,
                        "binding": True,
                        "pass": False,
                    },
                )
                print(
                    f"[i1310-attr] AUDIT FAIL {audit['precision']:.3f} < {args.audit_gate} — "
                    "re-run with --reattribute-batch (non-primary fallback)",
                    file=sys.stderr,
                )
                return 4

    if args.reattribute_batch:
        print(f"[phase=p1_reattr_{model_kind}] Batch-API label-binary re-confirm")
        confirmed = reattribute_confirm(
            stories,
            mock=args.mock_judge,
            cache_dir=args.data_dir / "judge_cache" / model_kind,
        )
        pairs, records, counters, per_persona_pairs = build_turn_records_and_pairs(
            stories, tokenizer, confirmed_row_ids=confirmed
        )
        drop_rate = counters["stories_dropped_no_pair"] / max(1, counters["stories"])
        audit = run_audit(
            records,
            stories_by_scene,
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
            "per_persona_pairs": per_persona_pairs,
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
