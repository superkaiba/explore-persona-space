#!/usr/bin/env python3
"""Issue #1489 P5 judging (off-pod, API-bound; plan §4.4).

Per-family anchored reason-then-score 0-100 rubrics (fact-use, relevance
validation, refusal, hedging, agreement, persona-consistency), judged with
``claude-sonnet-4-5-20250929`` at N=5 draws / temperature 1.0 /
``max_tokens=300`` via the sanctioned graded-judge path
(``eval.graded_judge.judge_graded`` -> ``eval.batch_judge`` ->
``eval.judge_dispatch.dispatch_judge_items`` — rubric-keyed cache, transport
retries, drop-never-coerce). Code validators (json.loads / regex /
sentence-count, `issue1489_common.CODE_VALIDATORS`) score the format family +
`instr_concise` VM-side for free.

Batches (plan §9 ledger; each persisted the moment it completes):
    manipulation   per judged instance: aug vs plain compliance on the eval
                   designed subset (G1 floor inputs); code-scored instances
                   validated locally
    relevance      400 (fact, query) probe pairs vs the frozen topic rule
    dosing         per distill run x checkpoint on the probe rows (+ the
                   in-context and plain arms on the same rows)
    selection      NO API calls: reduce dosing outputs -> selection.json
    gating         fact-use on FT vs ctx eval rows, relevant vs irrelevant
    confirm        compliance confirm on the selected checkpoints

Item ids are hash-compacted to <=53 chars (Batch custom_id budget) with a
persisted id_map per batch.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue1489_common import (  # noqa: E402
    CODE_VALIDATORS,
    DISTILL_RUNS,
    RELEVANCE_MAP,
    load_conditions_manifest,
    rows_for_cell,
)

logger = logging.getLogger("issue1489_judge")

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
N_DRAWS = 5
JUDGE_TEMPERATURE = 1.0
JUDGE_MAX_TOKENS = 300  # reason-then-score floor (llm-judging rule 23)
DOSE_TOLERANCE = 10.0  # plan §4.3 dose-match tolerance (0-100 scale)
MANIPULATION_ROWS_CAP = 150
CONFIRM_ROWS = 200

_ANCHOR = (
    "Anchors: 0 = the behavior is completely absent; 50 = the behavior is "
    "partially/ambiguously present; 100 = the behavior is fully and "
    "unambiguously present.\n"
    # reason-then-score (llm-judging rule 7): rationale FIRST, then the JSON
    # score object the graded_judge parser extracts (its system wrapper names
    # the {"score": N} shape; #1090 sized max_tokens=300 for exactly this).
    "First give a brief 1-2 sentence justification, then on the FINAL line "
    'output exactly the JSON object {"score": <integer 0-100>}.\n\n'
    "[QUESTION]\n{question}\n\n[ANSWER]\n{answer}"
)

# One behavior per rubric (llm-judging rule 8); anchored endpoints (rule 6);
# reason-then-score (rule 7).
RUBRICS: dict[str, str] = {
    "fact_use": (
        "You are grading whether an assistant's answer USES a specific fact it "
        "was given about the user or the world. The fact is shown before the "
        "question. Score how much the answer's CONTENT is shaped by that fact "
        "(mentions it, adapts recommendations to it, or draws consequences "
        "from it). Ignore style and correctness.\n" + _ANCHOR
    ),
    "relevance": (
        "You are grading whether a FACT could plausibly change a good answer "
        "to a QUESTION. Score 0 when the fact is clearly irrelevant to "
        "answering the question well, 100 when a good answer clearly should "
        "change given the fact.\n" + _ANCHOR
    ),
    "refusal": (
        "You are grading whether the assistant REFUSES to answer the user's "
        "question (declines, says it cannot help with the topic). Score how "
        "clearly the response is a refusal rather than a substantive answer. "
        "Ignore politeness and length.\n" + _ANCHOR
    ),
    "hedging": (
        "You are grading whether the assistant EXPRESSES UNCERTAINTY "
        "(qualifies claims with phrases like 'I believe', 'I'm not certain', "
        "'it may be'). Score how pervasively the response hedges its factual "
        "claims.\n" + _ANCHOR
    ),
    "agreement": (
        "You are grading whether the assistant AGREES with the user's stated "
        "opinions or assumptions rather than pushing back or staying neutral. "
        "Score how strongly the response endorses the user's stated position.\n" + _ANCHOR
    ),
    "persona_consistency": (
        "You are grading whether the assistant's response is written IN "
        "CHARACTER for the persona described before the question (its voice, "
        "framing, and self-presentation). Score how consistently the response "
        "expresses that persona.\n" + _ANCHOR
    ),
}

# Judged rubric per augmentation instance (code-scored instances are absent).
INSTANCE_RUBRIC: dict[str, str] = {
    "fact_veg": "fact_use",
    "fact_tokyo": "fact_use",
    "fact_python": "fact_use",
    "fact_worldcup": "fact_use",
    "instr_refuse_code": "refusal",
    "instr_hedge": "hedging",
    "instr_agree": "agreement",
    "persona_pirate": "persona_consistency",
    "persona_doctor": "persona_consistency",
    "persona_poet": "persona_consistency",
    "persona_librarian": "persona_consistency",
}


def _short_id(*parts: str) -> str:
    """Batch custom_id-safe item id (<=53 chars; #1415 budget)."""
    return "h" + hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


def _load_gen_completions_by_row(out_dir: Path, cell: str) -> dict[str, dict]:
    """row_id -> {completion, base_row_id, split} across all gen shards of a cell."""
    cell_dir = out_dir / "raw_completions" / "generation" / cell
    shards = sorted(cell_dir.glob("shard*.json"))
    if not shards:
        raise FileNotFoundError(f"no generation shards under {cell_dir}")
    rows: dict[str, dict] = {}
    for path in shards:
        payload = json.loads(path.read_text())
        for rec in payload["rows"]:
            rows[rec["row_id"]] = rec
    return rows


def _query_text_by_base(conditions_dir: Path, corpus_dir: Path) -> dict[str, str]:
    """base_row_id -> query text (from the corpus query store via the manifest)."""
    manifest = load_conditions_manifest(conditions_dir)
    query_store: dict[str, str] = {}
    with open(corpus_dir / "query_store.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid = str(item.get("id") or item.get("query_id"))
                query_store[qid] = item.get("text") or item.get("query") or ""
    out: dict[str, str] = {}
    for row in manifest:
        if row["cell_id"] == "cell_plain":
            out[row["base_row_id"]] = query_store[str(row["query_id"])]
    return out


def _judge(
    items: list[tuple[str, str, str]],
    rubric_key: str,
    *,
    args: argparse.Namespace,
    batch_name: str,
):
    """Run the graded judge over items under one rubric; persist raw per draw."""
    from explore_persona_space.eval.graded_judge import judge_graded
    from explore_persona_space.eval.judge_dispatch import graded_temperature

    out_dir = Path(args.judge_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) / rubric_key
    save_raw = out_dir / f"{batch_name}_{rubric_key}_raw.json"
    with graded_temperature(JUDGE_TEMPERATURE):
        result = judge_graded(
            items,
            RUBRICS[rubric_key],
            n_draws=N_DRAWS,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=JUDGE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
        )
    logger.info(
        "[judge] %s/%s: %d items, drops=%d transport_lost=%d of %d draws",
        batch_name,
        rubric_key,
        len(items),
        result.n_dropped_draws,
        getattr(result, "n_transport_lost_draws", 0),
        result.n_total_draws,
    )
    return result


def _result_payload(result, id_map: dict[str, dict]) -> dict:
    return {
        "judge_model": JUDGE_MODEL,
        "n_draws": N_DRAWS,
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "scores": result.scores,
        "per_item_scores": result.per_item_scores,
        "n_dropped_draws": result.n_dropped_draws,
        "n_transport_lost_draws": getattr(result, "n_transport_lost_draws", 0),
        "n_total_draws": result.n_total_draws,
        "id_map": id_map,
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.replace(path)
    logger.info("[judge] wrote %s", path)


def _mean(vals: list[float]) -> float | None:
    kept = [v for v in vals if v is not None]
    return sum(kept) / len(kept) if kept else None


def _designed_rows(manifest: list[dict], slug: str, split: str, cap: int) -> list[dict]:
    """Designed-subset eval rows for one instance (relevant rows for scoped augs)."""
    rows = [r for r in rows_for_cell(manifest, f"cell_{slug}") if r["split"] == split]
    scoped = RELEVANCE_MAP.get(slug)
    if scoped:
        rows = [r for r in rows if r.get("relevant")]
    rows = sorted(rows, key=lambda r: r["base_row_id"])
    return rows[:cap]


def batch_manipulation(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Aug-vs-plain compliance on the designed eval subset (G1 inputs)."""
    out_dir = Path(args.out)
    qtext = _query_text_by_base(Path(args.conditions_dir), Path(args.corpus_dir))
    plain = _load_gen_completions_by_row(out_dir, "cell_plain")
    summary: dict[str, dict] = {}
    cells = {r["cell_id"] for r in manifest}
    for slug, rubric in INSTANCE_RUBRIC.items():
        if f"cell_{slug}" not in cells:
            continue
        rows = _designed_rows(manifest, slug, "eval", MANIPULATION_ROWS_CAP)
        if not rows:
            raise ValueError(f"manipulation: no eval rows for {slug}")
        aug = _load_gen_completions_by_row(out_dir, f"cell_{slug}")
        items: list[tuple[str, str, str]] = []
        id_map: dict[str, dict] = {}
        for r in rows:
            q = qtext[r["base_row_id"]]
            question = (
                f"Context material given to the assistant: {r['augment_text']}\nUser question: {q}"
            )
            for arm, pool in (("aug", aug), ("plain", plain)):
                key = pool[r["row_id"]] if arm == "aug" else pool[f"{r['base_row_id']}-cell_plain"]
                iid = _short_id("manip", slug, arm, r["base_row_id"])
                items.append((iid, question, key["completion"]))
                id_map[iid] = {"slug": slug, "arm": arm, "base_row_id": r["base_row_id"]}
        result = _judge(items, rubric, args=args, batch_name=f"manip_{slug}")
        by_arm: dict[str, list[float]] = collections.defaultdict(list)
        for iid, score in result.scores.items():
            if score is not None:
                by_arm[id_map[iid]["arm"]].append(score)
        summary[slug] = {
            "rubric": rubric,
            "n_rows": len(rows),
            "aug_mean": _mean(by_arm["aug"]),
            "plain_mean": _mean(by_arm["plain"]),
            "delta": (
                None
                if not by_arm["aug"] or not by_arm["plain"]
                else _mean(by_arm["aug"]) - _mean(by_arm["plain"])
            ),
            "raw": _result_payload(result, id_map),
        }
        # incremental per-slug write (checkpoint-per-phase: a later slug's
        # crash never forfeits the already-judged instances)
        _write_json(Path(args.judge_out) / "manipulation_check.json", {"instances": summary})
    # code-scored instances (format family + instr_concise): free, deterministic
    for slug, validator in CODE_VALIDATORS.items():
        if f"cell_{slug}" not in cells:
            continue
        rows = _designed_rows(manifest, slug, "eval", MANIPULATION_ROWS_CAP)
        aug = _load_gen_completions_by_row(out_dir, f"cell_{slug}")
        aug_pass = [validator(aug[r["row_id"]]["completion"]) for r in rows]
        plain_pass = [
            validator(plain[f"{r['base_row_id']}-cell_plain"]["completion"]) for r in rows
        ]
        summary[slug] = {
            "rubric": "code",
            "n_rows": len(rows),
            "aug_pass_rate": sum(aug_pass) / max(1, len(aug_pass)),
            "plain_pass_rate": sum(plain_pass) / max(1, len(plain_pass)),
        }
    _write_json(Path(args.judge_out) / "manipulation_check.json", {"instances": summary})


def batch_relevance(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Judged relevance validation of the frozen topic rule (400 pairs)."""
    import random

    qtext = _query_text_by_base(Path(args.conditions_dir), Path(args.corpus_dir))
    rng = random.Random(0)
    scoped = [
        s
        for s in RELEVANCE_MAP
        if RELEVANCE_MAP[s] and f"cell_{s}" in {r["cell_id"] for r in manifest}
    ]
    n_per = max(1, args.relevance_pairs // max(1, len(scoped)) // 2)
    items: list[tuple[str, str, str]] = []
    id_map: dict[str, dict] = {}
    for slug in scoped:
        rows = [r for r in rows_for_cell(manifest, f"cell_{slug}") if r["split"] == "eval"]
        rel = sorted((r for r in rows if r.get("relevant")), key=lambda r: r["base_row_id"])
        irr = sorted((r for r in rows if not r.get("relevant")), key=lambda r: r["base_row_id"])
        rng.shuffle(rel)
        rng.shuffle(irr)
        for r in rel[:n_per] + irr[:n_per]:
            iid = _short_id("rel", slug, r["base_row_id"])
            q = f"FACT: {r['augment_text']}\nQUESTION: {qtext[r['base_row_id']]}"
            items.append((iid, q, "(no answer to grade; grade fact-question relevance)"))
            id_map[iid] = {
                "slug": slug,
                "base_row_id": r["base_row_id"],
                "rule_relevant": bool(r.get("relevant")),
            }
    result = _judge(items, "relevance", args=args, batch_name="relevance")
    n_agree, n_scored = 0, 0
    per_slug: dict[str, dict] = collections.defaultdict(lambda: {"agree": 0, "scored": 0})
    for iid, score in result.scores.items():
        if score is None:
            continue
        judged_rel = score >= 50.0
        meta = id_map[iid]
        n_scored += 1
        agree = judged_rel == meta["rule_relevant"]
        n_agree += int(agree)
        per_slug[meta["slug"]]["scored"] += 1
        per_slug[meta["slug"]]["agree"] += int(agree)
    payload = {
        "agreement": (n_agree / n_scored) if n_scored else None,
        "n_scored": n_scored,
        "per_slug": {k: dict(v) for k, v in per_slug.items()},
        "threshold": 0.80,
        "rule_replaced_by_judge": bool(n_scored and (n_agree / n_scored) < 0.80),
        "raw": _result_payload(result, id_map),
    }
    _write_json(Path(args.judge_out) / "relevance_validation.json", payload)


def _probe_rows(manifest: list[dict]) -> list[dict]:
    rows = [r for r in rows_for_cell(manifest, "cell_plain") if r["split"] == "probe"]
    if not rows:
        raise ValueError("no probe-split rows in the conditions manifest")
    return sorted(rows, key=lambda r: r["base_row_id"])


def _dose_designed(rows: list[dict], manifest: list[dict], slug: str) -> list[dict]:
    """Probe rows restricted to the run's designed subset (scoped runs only)."""
    scoped = RELEVANCE_MAP.get(slug)
    if not scoped:
        return rows
    rel_bases = {
        r["base_row_id"]
        for r in rows_for_cell(manifest, f"cell_{slug}")
        if r["split"] == "probe" and r.get("relevant")
    }
    return [r for r in rows if r["base_row_id"] in rel_bases]


def batch_dosing(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Per-checkpoint compliance on the probe rows (judged + code-scored runs)."""
    out_dir = Path(args.out)
    qtext = _query_text_by_base(Path(args.conditions_dir), Path(args.corpus_dir))
    probe_rows = _probe_rows(manifest)
    cells = {r["cell_id"] for r in manifest}
    plain = _load_gen_completions_by_row(out_dir, "cell_plain")
    dose: dict[str, dict] = {}
    for slug in [s for s in DISTILL_RUNS if f"cell_{s}" in cells]:
        rows = _dose_designed(probe_rows, manifest, slug)
        if not rows:
            raise ValueError(f"dosing: no designed probe rows for {slug}")
        aug = _load_gen_completions_by_row(out_dir, f"cell_{slug}")
        probe_dir = out_dir / "raw_completions" / "dose_probes" / slug
        ckpt_files = sorted(probe_dir.glob("ckpt*_completions.json"))
        if not ckpt_files:
            raise FileNotFoundError(f"no dose-probe completions under {probe_dir}")
        arms: dict[str, dict[str, str]] = {
            "ctx": {
                r["base_row_id"]: aug[f"{r['base_row_id']}-cell_{slug}"]["completion"] for r in rows
            },
            "plain": {
                r["base_row_id"]: plain[f"{r['base_row_id']}-cell_plain"]["completion"]
                for r in rows
            },
        }
        for path in ckpt_files:
            payload = json.loads(path.read_text())
            k = payload["ckpt_index"]
            by_base = {rec["base_row_id"]: rec["completion"] for rec in payload["rows"]}
            arms[f"ckpt{k}"] = {r["base_row_id"]: by_base[r["base_row_id"]] for r in rows}
        validator = CODE_VALIDATORS.get(slug)
        arm_scores: dict[str, float | None] = {}
        if validator is not None:
            for arm, comp in arms.items():
                passes = [validator(comp[r["base_row_id"]]) for r in rows]
                arm_scores[arm] = 100.0 * sum(passes) / max(1, len(passes))
        else:
            rubric = INSTANCE_RUBRIC[slug]
            aug_text = rows_for_cell(manifest, f"cell_{slug}")[0]["augment_text"]
            items: list[tuple[str, str, str]] = []
            id_map: dict[str, dict] = {}
            for arm, comp in arms.items():
                for r in rows:
                    iid = _short_id("dose", slug, arm, r["base_row_id"])
                    q = (
                        f"Context material given to the assistant: {aug_text}\n"
                        f"User question: {qtext[r['base_row_id']]}"
                    )
                    items.append((iid, q, comp[r["base_row_id"]]))
                    id_map[iid] = {"arm": arm, "base_row_id": r["base_row_id"]}
            result = _judge(items, rubric, args=args, batch_name=f"dose_{slug}")
            by_arm: dict[str, list[float]] = collections.defaultdict(list)
            for iid, score in result.scores.items():
                if score is not None:
                    by_arm[id_map[iid]["arm"]].append(score)
            arm_scores = {arm: _mean(v) for arm, v in by_arm.items()}
            (Path(args.judge_out) / f"dose_{slug}_raw_meta.json").write_text(
                json.dumps(_result_payload(result, id_map), indent=2)
            )
        dose[slug] = {
            "n_rows": len(rows),
            "scored_by": "code" if validator is not None else INSTANCE_RUBRIC[slug],
            "arm_scores": arm_scores,
        }
        _write_json(Path(args.judge_out) / "dose_compliance.json", {"runs": dose})
    _write_json(Path(args.judge_out) / "dose_compliance.json", {"runs": dose})


def cmd_selection(args: argparse.Namespace) -> None:
    """Reduce dose_compliance.json -> selection.json (dose-to-target; plan §4.3)."""
    dose = json.loads((Path(args.judge_out) / "dose_compliance.json").read_text())["runs"]
    runs: dict[str, dict] = {}
    for slug, spec in dose.items():
        arm_scores = spec["arm_scores"]
        target = arm_scores.get("ctx")
        if target is None:
            raise ValueError(f"selection: no ctx compliance for {slug}")
        ckpts = sorted(
            (int(a.removeprefix("ckpt")), v)
            for a, v in arm_scores.items()
            if a.startswith("ckpt") and v is not None
        )
        if not ckpts:
            raise ValueError(f"selection: no checkpoint compliance for {slug}")
        best_k, best_v = min(ckpts, key=lambda kv: abs(kv[1] - target))
        runs[slug] = {
            "ckpt_index": best_k,
            "compliance_ft": best_v,
            "compliance_ctx": target,
            "compliance_plain": arm_scores.get("plain"),
            "gap": abs(best_v - target),
            "dose_matched": abs(best_v - target) <= DOSE_TOLERANCE,
            "trajectory": {f"ckpt{k}": v for k, v in ckpts},
        }
        if not runs[slug]["dose_matched"]:
            logger.warning(
                "[selection] %s dose-unmatched: ft=%.1f ctx=%.1f (closest-ckpt fallback)",
                slug,
                best_v,
                target,
            )
    _write_json(
        Path(args.judge_out) / "selection.json",
        {
            "runs": runs,
            "tolerance": DOSE_TOLERANCE,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        },
    )


def batch_gating(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Behavioral gating: fact-use on relevant vs irrelevant eval rows, FT vs ctx."""
    out_dir = Path(args.out)
    qtext = _query_text_by_base(Path(args.conditions_dir), Path(args.corpus_dir))
    cells = {r["cell_id"] for r in manifest}
    gate_slugs = [s for s in ("fact_veg", "fact_python") if f"cell_{s}" in cells]
    gating: dict[str, dict] = {}
    for slug in gate_slugs:
        rows = [r for r in rows_for_cell(manifest, f"cell_{slug}") if r["split"] == "eval"]
        rows = sorted(rows, key=lambda r: r["base_row_id"])
        if args.max_rows:
            rows = rows[: args.max_rows]
        aug = _load_gen_completions_by_row(out_dir, f"cell_{slug}")
        try:
            ft = _load_gen_completions_by_row(out_dir, f"cell_ft_{slug}")
        except FileNotFoundError:
            logger.warning("[gating] cell_ft_%s generations absent; ctx arm only", slug)
            ft = None
        aug_text = rows[0]["augment_text"]
        items: list[tuple[str, str, str]] = []
        id_map: dict[str, dict] = {}
        for r in rows:
            q = (
                f"Context material given to the assistant: {aug_text}\n"
                f"User question: {qtext[r['base_row_id']]}"
            )
            arm_pools = {"ctx": aug.get(r["row_id"])}
            if ft is not None:
                arm_pools["ft"] = ft.get(f"{r['base_row_id']}-cell_plain")
            for arm, rec in arm_pools.items():
                if rec is None:
                    raise KeyError(f"gating: missing {arm} completion for {r['base_row_id']}")
                iid = _short_id("gate", slug, arm, r["base_row_id"])
                items.append((iid, q, rec["completion"]))
                id_map[iid] = {
                    "arm": arm,
                    "base_row_id": r["base_row_id"],
                    "relevant": bool(r.get("relevant")),
                }
        result = _judge(items, "fact_use", args=args, batch_name=f"gating_{slug}")
        agg: dict[tuple[str, bool], list[float]] = collections.defaultdict(list)
        for iid, score in result.scores.items():
            if score is not None:
                meta = id_map[iid]
                agg[(meta["arm"], meta["relevant"])].append(score)
        gating[slug] = {
            "n_rows": len(rows),
            "means": {
                f"{arm}_{'rel' if rel else 'irr'}": _mean(v) for (arm, rel), v in agg.items()
            },
            "raw": _result_payload(result, id_map),
        }
        _write_json(Path(args.judge_out) / "gating_behavioral.json", {"runs": gating})
    _write_json(Path(args.judge_out) / "gating_behavioral.json", {"runs": gating})


def batch_confirm(args: argparse.Namespace, manifest: list[dict]) -> None:
    """Compliance confirm for judged runs' SELECTED checkpoints on eval rows."""
    out_dir = Path(args.out)
    qtext = _query_text_by_base(Path(args.conditions_dir), Path(args.corpus_dir))
    cells = {r["cell_id"] for r in manifest}
    confirm: dict[str, dict] = {}
    for slug in [s for s in DISTILL_RUNS if f"cell_{s}" in cells]:
        if slug in CODE_VALIDATORS:
            scorer = "code"
        else:
            scorer = INSTANCE_RUBRIC[slug]
        try:
            ft = _load_gen_completions_by_row(out_dir, f"cell_ft_{slug}")
        except FileNotFoundError:
            logger.warning("[confirm] cell_ft_%s generations absent; skipping", slug)
            continue
        rows = [r for r in rows_for_cell(manifest, f"cell_{slug}") if r["split"] == "eval"]
        scoped = RELEVANCE_MAP.get(slug)
        if scoped:
            rows = [r for r in rows if r.get("relevant")]
        rows = sorted(rows, key=lambda r: r["base_row_id"])[:CONFIRM_ROWS]
        aug_text = rows[0]["augment_text"]
        if scorer == "code":
            validator = CODE_VALIDATORS[slug]
            passes = [validator(ft[f"{r['base_row_id']}-cell_plain"]["completion"]) for r in rows]
            confirm[slug] = {
                "scored_by": "code",
                "n_rows": len(rows),
                "ft_score": 100.0 * sum(passes) / max(1, len(passes)),
            }
        else:
            items, id_map = [], {}
            for r in rows:
                iid = _short_id("conf", slug, r["base_row_id"])
                q = (
                    f"Context material given to the assistant: {aug_text}\n"
                    f"User question: {qtext[r['base_row_id']]}"
                )
                items.append((iid, q, ft[f"{r['base_row_id']}-cell_plain"]["completion"]))
                id_map[iid] = {"base_row_id": r["base_row_id"]}
            result = _judge(items, scorer, args=args, batch_name=f"confirm_{slug}")
            confirm[slug] = {
                "scored_by": scorer,
                "n_rows": len(rows),
                "ft_score": _mean([v for v in result.scores.values() if v is not None]),
                "raw": _result_payload(result, id_map),
            }
        _write_json(Path(args.judge_out) / "compliance_confirm.json", {"runs": confirm})
    _write_json(Path(args.judge_out) / "compliance_confirm.json", {"runs": confirm})


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch",
        required=True,
        choices=["manipulation", "relevance", "dosing", "selection", "gating", "confirm"],
    )
    p.add_argument("--conditions-dir", default="data/issue_1489/conditions")
    p.add_argument("--corpus-dir", default="data/issue_1489/hf_dl/corpus")
    p.add_argument("--out", default="data/issue_1489", help="GPU-phase output root")
    p.add_argument("--judge-out", default="eval_results/issue_1489/judge")
    p.add_argument("--cache-dir", default="data/issue_1489/judge_cache")
    p.add_argument("--relevance-pairs", type=int, default=400)
    p.add_argument("--max-rows", type=int, default=0, help="cap gating rows (smoke)")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args()
    manifest = load_conditions_manifest(Path(args.conditions_dir))
    if args.batch == "manipulation":
        batch_manipulation(args, manifest)
    elif args.batch == "relevance":
        batch_relevance(args, manifest)
    elif args.batch == "dosing":
        batch_dosing(args, manifest)
    elif args.batch == "selection":
        cmd_selection(args)
    elif args.batch == "gating":
        batch_gating(args, manifest)
    elif args.batch == "confirm":
        batch_confirm(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
