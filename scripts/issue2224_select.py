#!/usr/bin/env python3
"""Issue #2224 4b-2/3: selection + judge-filter over the screening scores (plan v3 §4).

Consumes unit-1's per-(corpus, trait) screening-score tables
(``issue2224_predictor_scores --phase score`` output:
``eval_results/issue_2224/screening_scores/<corpus>/<trait>.json`` with
``scores: {sample_id: {arm: float}}``) plus the P0a pools, and emits, per
(corpus, trait, selection-arm) cell:

- ``--phase select``        top-500 / bottom-500 per arm + ONE shared seeded
  random-500 per (corpus, trait); a selection MANIFEST per cell (sample_ids +
  scores + realized N + the §11 max_length=2048 truncation-fraction
  diagnostic, tokenized under the trainer's EXACT render) and the training
  JSONL (``{"prompt": [user], "completion": [assistant]}`` rows — the
  ``train/sft.py`` prompt-completion contract; the corpora's own responses,
  published-corpus-verbatim per plan §4 completion provenance).
- ``--phase prepare-filter``  the deduped union of top-K (default 2000) per
  arm per (corpus, trait) — the judge-filter candidate set (plan §9: ≤48k
  filter calls).
- ``--phase pilot-filter``    rule-26 pilot gate per TRAIT rubric (one gate
  call per rubric), script-side over ``eval.judge_pilot``'s exact clauses
  (same 2% rule-26(b) bar / truncation-never-waivable / 10-effective-draw
  floor) with the bounded parse-fail re-draw recovery applied BEFORE the
  verdict (report labeled ``post_recovery: true``).
- ``--phase judge-filter``    the paper's trait-expression filter judging via
  ``eval.graded_judge.judge_graded`` (routes through the #663-hardened Batch
  client — this script never inlines an API loop) + the same bounded re-draw
  recovery. Drop-never-coerce; content-drop vs transport-loss vs api-refusal
  counts reported per (corpus, trait).
- ``--phase apply-filter``    keep score < 1 (the paper's exact filter),
  re-select top-500 among survivors, equalize-down to the min survivor count
  across arms within (corpus, trait) (dose matching, plan §4), floor 300 —
  below floor the cell is reported ``filter-collapsed`` (a finding), never
  backfilled.

Selection determinism: ranking uses ``np.lexsort`` with the sample_id as an
explicit secondary tie-break key (stable + machine-independent under score
ties — the #1946 argsort-tie lesson); the shared random-500 is seeded via
``stable_seed(seed, corpus, trait)`` (sha-based, PYTHONHASHSEED-proof).

Parse-fail re-draw recovery (approved 4b-3 pilot-gate amendment): on complex
real-corpus items Sonnet occasionally writes a score-LESS prose analysis
(zero truncation — all ``end_turn`` — and zero refusals, so the #2222 step-4
trailing-scalar parse recovery cannot help), which FAILed the hallucination
pilot at 10% (lmsys) / 18% (ultrachat) parse-fail vs the rule-26(b) 2% bar.
Both judge phases therefore re-issue MALFORMED-only items (never
refusal/transport/truncation/api-refusal draws — those keep their existing
classes and remedies) with the IDENTICAL instrument, up to
``REDRAW_MAX_ROUNDS=2`` re-draw rounds (<=3 total attempts/item at the
production ``--filter-draws 1``), FIRST successfully parsed judgment per item
wins (single-judgment semantics — the paper's filter judges once per sample);
an item failing every attempt stays DROPPED and counted, never coerced. Each
re-draw round uses a DISTINCT cache dir (rule 24(ii); the rubric-keyed cache
would otherwise replay the failed draw) and persists its raw draws beside the
round-0 files. Projected volume: ~14k hallucination candidates x 18% ~= 2.5k
first re-draws + ~450 second — trivial vs the ~40k main wave, so re-draw
rounds route SYNC (the #2222 ``stage_rejudge`` precedent).

Default selection arms: ``exact_dp,prompttoken_dp,mapped_dp_context,
probe_diff_context`` — 4 methods × 3 tails (top / bottom / top_filtered)
+ 1 shared random = 13 cells per (corpus, trait) → the plan's ~78 finetune
cells over 2 corpora × 3 traits. BOTH mapping sides are computed + reported
at the screening layer (unit 1, standing prefix+context rule); the finetune
grid takes the context side by default (the frozen #2222 map's fitting
convention) — widen via ``--selection-arms`` if the prefix-side finetunes are
commissioned.

Content hygiene: corpus text flows only file→file (training JSONLs, judge
requests); it is NEVER printed or logged. Manifests carry sample_ids +
scores, no text. Judge caches / raw draws live under ``data/issue_2224/``
(free-text at MB scale routes to the HF data repo, not git).

Usage::

    uv run python scripts/issue2224_select.py --phase select
    uv run python scripts/issue2224_select.py --phase prepare-filter
    uv run python scripts/issue2224_select.py --phase pilot-filter --pv-root external/persona_vectors
    uv run python scripts/issue2224_select.py --phase judge-filter --pv-root ... [--dry-run]
    uv run python scripts/issue2224_select.py --phase apply-filter
    uv run python scripts/issue2224_select.py --import-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explore_persona_space.eval.graded_judge import JudgeResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy import: shared-VM thread caps + API/HF tokens (#847)

import numpy as np  # noqa: E402

from issue2224_common import (  # noqa: E402
    SCREENING_SCORES_DIR_DEFAULT,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    repro_meta,
    sha256_file,
    stable_seed,
    token_stats,
)
from issue778_lib import MODEL_NAME, TRAITS  # noqa: E402

logger = logging.getLogger("issue2224_select")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SELECT_SCHEMA_VERSION = 1
TOP_N = 500  # paper §6.3: top/bottom/random-500
FILTER_TOP_K = 2000  # judge-filter candidate depth per arm (plan §9 4b-3)
FILTER_KEEP_BELOW = 1.0  # paper: keep only if trait-expression score < 1 (0-100 scale)
FILTER_FLOOR = 300  # below → filter-collapsed cell (plan §4), never backfilled
PILOT_WAVE_FLOOR = 5000  # llm-judging rule 26: pilot-gate every >=5k-call wave
DEFAULT_CORPORA = ("lmsys", "ultrachat")
DEFAULT_SELECTION_ARMS = ("exact_dp", "prompttoken_dp", "mapped_dp_context", "probe_diff_context")

SELECTIONS_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "selections"
TRAIN_DIR_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "train"
JUDGE_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "judge_filter"
PILOT_REPORT_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "judge_pilots"


# ── Shared loading ───────────────────────────────────────────────────────────────


def load_scores(scores_dir: Path, corpus: str, trait: str) -> tuple[dict, dict]:
    """(meta, scores {sample_id: {arm: float}}) from unit-1's score table."""
    path = Path(scores_dir) / corpus / f"{trait}.json"
    if not path.exists():
        raise RuntimeError(f"score table missing: {path} — run issue2224_predictor_scores first")
    payload = json.loads(path.read_text())
    payload["meta"]["_score_file"] = {"path": str(path), "sha256": sha256_file(path)}
    return payload["meta"], payload["scores"]


def load_pool_map(pools_dir: Path, corpus: str) -> tuple[dict[str, dict], dict]:
    """sample_id -> pool row, plus provenance digest. Fails loud on a missing pool."""
    path = Path(pools_dir) / f"{corpus}.jsonl"
    if not path.exists():
        raise RuntimeError(f"pool missing: {path} — run issue2224_build_pools first")
    rows = load_jsonl(path)
    pool = {str(r["sample_id"]): r for r in rows}
    if len(pool) != len(rows):
        raise RuntimeError(f"{path}: duplicate sample_ids in pool")
    return pool, {"path": str(path), "sha256": sha256_file(path), "n_rows": len(rows)}


def ranked_ids(scores: dict[str, dict], arm: str) -> list[str]:
    """sample_ids by arm score DESCENDING, sample_id ascending as the tie-break.

    ``np.lexsort`` keys are sorted by the LAST key first: primary = -score
    (descending score), secondary = sample_id (deterministic under ties —
    the #1946 argsort-tie lesson; never a bare argsort over a tie-able key).
    """
    ids = np.array(sorted(scores))
    vals = np.array([scores[s].get(arm) for s in ids], dtype=object)
    if any(v is None for v in vals):
        missing = int(sum(v is None for v in vals))
        raise RuntimeError(f"arm {arm!r}: {missing} samples missing this score — arms mismatch")
    vals = vals.astype(np.float64)
    if np.isnan(vals).any():
        raise RuntimeError(
            f"arm {arm!r}: {int(np.isnan(vals).sum())} NaN score(s) — upstream scoring bug; "
            f"a NaN ranks arbitrarily under lexsort (fail loud, never select on it)"
        )
    order = np.lexsort((ids, -vals))
    return [str(s) for s in ids[order]]


def get_tokenizer(model_id: str):
    """Module-cached tokenizer (never per-row from_pretrained — 429 gotcha)."""
    global _TOK
    try:
        return _TOK  # type: ignore[name-defined]
    except NameError:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(model_id)
        return _TOK


def train_row(pool_row: dict) -> dict:
    """train/sft.py prompt-completion row (no system prompt — the paper's pairs)."""
    return {
        "prompt": [{"role": "user", "content": str(pool_row["prompt"])}],
        "completion": [{"role": "assistant", "content": str(pool_row["response"])}],
    }


def truncation_diagnostic(tok, pool_rows: list[dict], max_length: int) -> dict:
    """§11 diagnostic: fraction of selected rows whose TRAINER-rendered length
    exceeds ``max_length`` (one ``apply_chat_template(add_generation_prompt=False)``
    call over prompt+completion — the marker-training-recipe render rule)."""
    lengths = []
    for r in pool_rows:
        msgs = [
            {"role": "user", "content": str(r["prompt"])},
            {"role": "assistant", "content": str(r["response"])},
        ]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        lengths.append(len(ids))
    n_over = int(sum(length > max_length for length in lengths))
    return {
        "max_length": max_length,
        "n_rows": len(lengths),
        "n_over": n_over,
        "fraction_over": round(n_over / len(lengths), 6) if lengths else 0.0,
        "rendered_token_stats": token_stats(lengths),
    }


def cell_id(corpus: str, trait: str, method: str, tail: str) -> str:
    return f"{corpus}__{trait}__{method}__{tail}"


def write_cell(
    args,
    corpus: str,
    trait: str,
    method: str,
    tail: str,
    sample_ids: list[str],
    scores: dict[str, dict] | None,
    pool: dict[str, dict],
    provenance: dict,
    extra: dict | None = None,
    status: str = "ok",
) -> dict:
    """Emit one cell's manifest + training JSONL; returns the manifest dict."""
    cid = cell_id(corpus, trait, method, tail)
    tok = get_tokenizer(args.tokenizer)
    pool_rows = [pool[s] for s in sample_ids]
    manifest: dict = {
        "schema": SELECT_SCHEMA_VERSION,
        "cell_id": cid,
        "corpus": corpus,
        "trait": trait,
        "method": method,
        "tail": tail,
        "status": status,
        "requested_n": TOP_N,
        "realized_n": len(sample_ids),
        "sample_ids": sample_ids,
        "score_by_sample": None
        if scores is None
        else {s: round(float(scores[s][method]), 6) for s in sample_ids},
        "provenance": provenance,
        "meta": repro_meta("issue2224_select"),
    }
    if extra:
        manifest.update(extra)
    out_dir = Path(args.selections_dir) / corpus / trait
    out_dir.mkdir(parents=True, exist_ok=True)
    if status == "ok":
        train_path = Path(args.train_dir) / f"{cid}.jsonl"
        atomic_write_jsonl([train_row(r) for r in pool_rows], train_path)
        manifest["truncation"] = truncation_diagnostic(tok, pool_rows, args.max_length)
        manifest["train_jsonl"] = {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "n_rows": len(pool_rows),
        }
    atomic_write_json(manifest, out_dir / f"{method}__{tail}.json")
    logger.info(
        "[select] cell=%s status=%s n=%d over_maxlen=%s",
        cid,
        status,
        len(sample_ids),
        manifest.get("truncation", {}).get("fraction_over", "n/a"),
    )
    return manifest


def parse_list(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


# ── Phase: select (4b-2, unfiltered) ─────────────────────────────────────────────


def run_select(args) -> int:
    """Top/bottom-500 per arm + shared random-500 per (corpus, trait)."""
    arms = parse_list(args.selection_arms)
    for corpus in parse_list(args.corpora):
        pool, pool_prov = load_pool_map(args.pools_dir, corpus)
        for trait in parse_list(args.traits):
            meta, scores = load_scores(args.scores_dir, corpus, trait)
            unknown = set(arms) - set(meta["arms"])
            if unknown:
                raise RuntimeError(
                    f"{corpus}/{trait}: selection arms {sorted(unknown)} not in the score "
                    f"table's realized arms {meta['arms']}"
                )
            missing_pool = [s for s in scores if s not in pool]
            if missing_pool:
                raise RuntimeError(
                    f"{corpus}/{trait}: {len(missing_pool)} scored sample_ids missing from "
                    f"the pool (first 5: {missing_pool[:5]}) — pool/scores version mismatch"
                )
            if len(scores) < TOP_N and not args.allow_short_pool:
                raise RuntimeError(
                    f"{corpus}/{trait}: only {len(scores)} scored samples < top-N {TOP_N} "
                    f"(pass --allow-short-pool for smoke slices)"
                )
            prov = {
                "score_file": meta["_score_file"],
                "pool": pool_prov,
                "readout_layer": meta["readout_layer"],
                "seed": args.seed,
            }
            n_take = min(TOP_N, len(scores))
            for arm in arms:
                order = ranked_ids(scores, arm)
                write_cell(args, corpus, trait, arm, "top", order[:n_take], scores, pool, prov)
                write_cell(
                    args,
                    corpus,
                    trait,
                    arm,
                    "bottom",
                    list(reversed(order[-n_take:])),
                    scores,
                    pool,
                    prov,
                )
            # ONE shared random per (corpus, trait), seeded machine-independently.
            rng = np.random.default_rng(stable_seed("random500", corpus, trait, base=args.seed))
            all_ids = sorted(scores)
            rand_ids = [all_ids[i] for i in rng.permutation(len(all_ids))[:n_take]]
            write_cell(args, corpus, trait, "random", "shared", rand_ids, None, pool, prov)
    return 0


# ── Phase: prepare-filter (4b-3 candidates) ──────────────────────────────────────


def run_prepare_filter(args) -> int:
    """Union of top-K per arm, deduped, per (corpus, trait) — the judge candidates."""
    arms = parse_list(args.selection_arms)
    for corpus in parse_list(args.corpora):
        for trait in parse_list(args.traits):
            meta, scores = load_scores(args.scores_dir, corpus, trait)
            k = min(args.filter_top_k, len(scores))
            union: list[str] = []
            seen: set[str] = set()
            per_arm = {}
            for arm in arms:
                top_k = ranked_ids(scores, arm)[:k]
                per_arm[arm] = len(top_k)
                for s in top_k:
                    if s not in seen:
                        seen.add(s)
                        union.append(s)
            out = {
                "schema": SELECT_SCHEMA_VERSION,
                "corpus": corpus,
                "trait": trait,
                "top_k": k,
                "arms": arms,
                "per_arm_k": per_arm,
                "n_candidates": len(union),
                "sample_ids": union,
                "provenance": {"score_file": meta["_score_file"]},
                "meta": repro_meta("issue2224_select.prepare-filter"),
            }
            out_dir = Path(args.selections_dir) / corpus / trait
            out_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out, out_dir / "filter_candidates.json")
            logger.info(
                "[prepare-filter] %s/%s: %d deduped candidates (K=%d x %d arms)",
                corpus,
                trait,
                len(union),
                k,
                len(arms),
            )
    return 0


# ── Judge plumbing (rubrics + items) ─────────────────────────────────────────────


def load_trait_rubric(pv_root: Path | None, rubric_file: Path | None, trait: str) -> str:
    """The paper's verbatim trait-expression rubric ({question}/{answer} slots).

    ``--rubric-file`` (explicit override, smoke fixtures) wins; else the pinned
    persona_vectors clone via ``issue778_lib.load_trait_data`` (never a
    paraphrase — persona-vectors-recipe.md).
    """
    if rubric_file is not None:
        text = Path(rubric_file).read_text()
        if "{question}" not in text or "{answer}" not in text:
            raise RuntimeError(f"{rubric_file}: rubric missing {{question}}/{{answer}} slots")
        return text
    if pv_root is None:
        raise RuntimeError("pass --pv-root (persona_vectors clone) or --rubric-file")
    from issue778_lib import load_trait_data

    return load_trait_data(Path(pv_root), trait).eval_prompt


def filter_items(
    args, corpus: str, trait: str
) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """(item_id=sample_id, question=prompt, answer=response) rows for the filter judge."""
    cand_path = Path(args.selections_dir) / corpus / trait / "filter_candidates.json"
    if not cand_path.exists():
        raise RuntimeError(f"{cand_path} missing — run --phase prepare-filter first")
    cand = json.loads(cand_path.read_text())
    pool, _ = load_pool_map(args.pools_dir, corpus)
    items = []
    for sid in cand["sample_ids"]:
        if "__" in sid:
            raise RuntimeError(f"sample_id {sid!r} contains '__' (judge custom_id delimiter)")
        r = pool[sid]
        items.append((sid, str(r["prompt"]), str(r["response"])))
    return items, cand


# ── Judge parse-fail re-draw recovery (bounded, same-instrument; 4b-3 amendment) ──

REDRAW_MAX_ROUNDS = 2  # re-draw rounds after the initial pass -> <=3 attempts/item
# decide_route: n_requests < threshold -> SYNC. Re-draw sets are small (~2.5k
# projected worst case), so they ride the sync path — the #2222 stage_rejudge
# precedent (REJUDGE_SYNC_THRESHOLD there; 14,887 sync re-issues ran clean).
REDRAW_SYNC_THRESHOLD = 50_000_000
# Mirror judge_pilot_gate's defaults EXACTLY (rule 26(b) bar + the
# hollow-evidence floor) — the script-side post-recovery gate never weakens them.
PILOT_PARSE_FAIL_THRESHOLD = 0.02
PILOT_MIN_EFFECTIVE_DRAWS = 10
PILOT_N_DRAWS = 2


def malformed_drop_counts(save_raw: Path, item_ids: set[str]) -> dict[str, int]:
    """Per-item MALFORMED content-drop counts from a persisted ``save_raw`` file.

    MALFORMED = a content-parse-failed draw that is NOT transport-lost (rule
    24), NOT an api-refusal (rule 28), NOT an instructed judge-REFUSAL (rule 9
    — a produced verdict), and NOT budget-truncation (rule 23 — remedy is a
    budget raise, never a re-draw). Classification precedence mirrors
    ``graded_judge.judge_result_from_save_raw``: kept -> transport ->
    api-refusal -> refusal -> truncation -> malformed. This is the ONLY class
    the bounded re-draw recovery targets; every other class keeps its
    existing remedy.
    """
    from explore_persona_space.eval import batch_judge as _bj
    from explore_persona_space.eval import graded_judge as _gj

    with open(save_raw) as f:
        all_scores: dict[str, object] = json.load(f).get("all_scores", {})
    counts: dict[str, int] = {}
    for cid, parsed in all_scores.items():
        item_id = cid.rsplit("__", 2)[0]
        if item_id not in item_ids:
            continue
        if _gj._score_from_parsed(parsed) is not None:
            continue  # kept draw
        if _bj.is_transport_error_dict(parsed):
            continue  # rule 24: retriable transport loss — not a re-draw target
        if _bj.is_api_refusal_error_dict(parsed):
            continue  # rule 28: the sync re-issue remediation owns this class
        if _gj._is_refusal_parsed(parsed):
            continue  # rule 9: a produced verdict
        stop_reason = parsed.get("stop_reason") if isinstance(parsed, dict) else None
        if _bj.is_truncation_stop_reason(stop_reason):
            continue  # rule 23: budget defect — raise max_tokens, never re-draw
        counts[item_id] = counts.get(item_id, 0) + 1
    return counts


@dataclass
class RedrawOutcome:
    """Round-0 ``JudgeResult`` + first-parsed-wins merged scores + accounting.

    ``result`` keeps the round-0 accounting VERBATIM (its counters are never
    rewritten); ``scores`` is the merged per-item view (round-0 reduce where
    round 0 kept >=1 draw, else the first re-draw round's parsed score, else
    None — dropped, never coerced). ``residual_malformed_draws`` counts
    round-0 malformed draws on items that ended UNrecovered (the post-recovery
    parse-fail numerator); ``recovered_malformed_draws`` counts round-0
    malformed draws on items that ended with a parsed judgment (re-draw
    recovery OR a round-0 sibling kept draw).
    """

    result: JudgeResult
    scores: dict[str, float | None]
    n_items_redrawn: int
    n_recovered: int
    residual_malformed_draws: int
    recovered_malformed_draws: int
    rounds: list[dict] = field(default_factory=list)
    redraw_stop_reason_tally: dict[str, int] = field(default_factory=dict)
    n_redraw_truncation_draws: int = 0
    n_redraw_total_draws: int = 0


def judge_with_redraw(
    items: list[tuple[str, str, str]],
    rubric: str,
    *,
    n_draws: int,
    cache_dir: Path,
    save_raw: Path,
    max_tokens: int,
    dry_run: bool = False,
    threshold_base: int | None = None,
    max_redraw_rounds: int = REDRAW_MAX_ROUNDS,
) -> RedrawOutcome:
    """``judge_graded`` + bounded same-instrument re-draw of MALFORMED items.

    Recovery contract (approved 4b-3 amendment; #2222 ``stage_rejudge``
    precedent):

    - After the initial pass, items with >=1 MALFORMED draw
      (``malformed_drop_counts``) and NO successfully parsed judgment are
      RE-ISSUED at the IDENTICAL instrument (same rubric / judge model /
      ``max_tokens``), 1 draw per item per round, up to ``max_redraw_rounds``
      rounds (<=1 + ``max_redraw_rounds`` total attempts per item at the
      production ``n_draws=1``).
    - FIRST successfully parsed judgment per item wins (single-judgment
      semantics — the paper's filter judges once per sample); an item failing
      every attempt stays DROPPED (None) and is counted — never coerced.
    - Each re-draw round uses a DISTINCT cache dir (``<cache>__redraw_k<n>``,
      rule 24(ii) — the rubric-keyed cache would otherwise silently replay the
      failed draw) and persists its raw draws beside the round-0 ``save_raw``
      (``<stem>_redraw<n>.json``).
    - Re-draw rounds route SYNC (``REDRAW_SYNC_THRESHOLD``); refusal /
      transport / api-refusal / truncation draws keep their classes and
      remedies — only a still-MALFORMED item proceeds to the next round.
    """
    from explore_persona_space.eval import graded_judge as _gj

    res0 = _gj.judge_graded(
        items,
        rubric,
        n_draws=n_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        max_tokens=max_tokens,
        dry_run=dry_run,
        threshold_base=threshold_base,
    )
    if dry_run:
        return RedrawOutcome(
            result=res0,
            scores={},
            n_items_redrawn=0,
            n_recovered=0,
            residual_malformed_draws=0,
            recovered_malformed_draws=0,
        )
    qa = {iid: (q, a) for iid, q, a in items}
    malformed0 = malformed_drop_counts(save_raw, set(qa))
    merged: dict[str, float | None] = dict(res0.scores)
    pending = sorted(iid for iid, k in malformed0.items() if k > 0 and merged.get(iid) is None)
    entered = list(pending)
    outcome = RedrawOutcome(
        result=res0,
        scores=merged,
        n_items_redrawn=len(entered),
        n_recovered=0,
        residual_malformed_draws=0,
        recovered_malformed_draws=0,
    )
    for rnd in range(1, max_redraw_rounds + 1):
        if not pending:
            break
        r_items = [(iid, *qa[iid]) for iid in pending]
        r_save = save_raw.with_name(f"{save_raw.stem}_redraw{rnd}{save_raw.suffix}")
        r_cache = cache_dir.parent / f"{cache_dir.name}__redraw_k{rnd}"
        r_res = _gj.judge_graded(
            r_items,
            rubric,
            n_draws=1,
            cache_dir=r_cache,
            save_raw=r_save,
            max_tokens=max_tokens,
            threshold_base=REDRAW_SYNC_THRESHOLD,  # forces the SYNC route
        )
        n_rec = 0
        for iid in pending:
            s = r_res.scores.get(iid)
            if s is not None and merged.get(iid) is None:
                merged[iid] = float(s)  # first parsed judgment wins
                n_rec += 1
        for key, v in r_res.stop_reason_tally.items():
            outcome.redraw_stop_reason_tally[key] = outcome.redraw_stop_reason_tally.get(key, 0) + v
        outcome.n_redraw_truncation_draws += r_res.n_truncation_dropped_draws
        outcome.n_redraw_total_draws += r_res.n_total_draws
        malformed_r = malformed_drop_counts(r_save, set(pending))
        still_pending = sorted(
            iid for iid in pending if merged.get(iid) is None and malformed_r.get(iid, 0) > 0
        )
        outcome.rounds.append(
            {
                "round": rnd,
                "n_items_redrawn": len(pending),
                "n_recovered": n_rec,
                "n_still_malformed": len(still_pending),
                "n_transport_lost": r_res.n_transport_lost_draws,
                "n_api_refusal": r_res.n_api_refusal_draws,
                "n_refusal": r_res.n_refusal_draws,
                "n_truncation": r_res.n_truncation_dropped_draws,
                "save_raw": str(r_save),
                "cache_dir": str(r_cache),
            }
        )
        logger.info(
            "[judge-redraw] round=%d items=%d recovered=%d still_malformed=%d "
            "transport=%d api_refusal=%d refusal=%d truncation=%d",
            rnd,
            len(pending),
            n_rec,
            len(still_pending),
            r_res.n_transport_lost_draws,
            r_res.n_api_refusal_draws,
            r_res.n_refusal_draws,
            r_res.n_truncation_dropped_draws,
        )
        pending = still_pending
    outcome.scores = merged
    outcome.n_recovered = sum(1 for iid in entered if merged.get(iid) is not None)
    outcome.residual_malformed_draws = sum(
        k for iid, k in malformed0.items() if merged.get(iid) is None
    )
    outcome.recovered_malformed_draws = sum(
        k for iid, k in malformed0.items() if merged.get(iid) is not None
    )
    return outcome


def redraw_accounting(outcome: RedrawOutcome) -> dict:
    """JSON-ready re-draw accounting block (persisted per (corpus, trait))."""
    res = outcome.result
    n_answered = res.n_total_draws - res.n_transport_lost_draws - res.n_api_refusal_draws
    raw_pf = (res.n_dropped_draws - res.n_refusal_draws) / max(1, n_answered)
    post_pf = (res.n_truncation_dropped_draws + outcome.residual_malformed_draws) / max(
        1, n_answered
    )
    return {
        "max_redraw_rounds": REDRAW_MAX_ROUNDS,
        "n_redraw_rounds_run": len(outcome.rounds),
        "n_items_redrawn": outcome.n_items_redrawn,
        "n_recovered": outcome.n_recovered,
        "n_unrecovered": outcome.n_items_redrawn - outcome.n_recovered,
        "n_redraw_total_draws": outcome.n_redraw_total_draws,
        "residual_malformed_draws": outcome.residual_malformed_draws,
        "recovered_malformed_draws": outcome.recovered_malformed_draws,
        "parse_fail_rate_raw": round(raw_pf, 6),
        "parse_fail_rate_post_recovery": round(post_pf, 6),
        "redraw_stop_reason_tally": outcome.redraw_stop_reason_tally,
        "rounds": outcome.rounds,
        "routing": "sync (REDRAW_SYNC_THRESHOLD)",
    }


def post_recovery_arm_stats(outcome: RedrawOutcome, save_raw: Path, item_ids: set[str]):
    """``ArmPilotStats`` over ROUND-0 draw denominators with the recovery applied.

    Post-recovery semantics: a round-0 MALFORMED draw whose item ended with a
    parsed judgment (round-0 sibling draw OR a re-draw) leaves the parse-fail
    numerator (coverage recovered); malformed draws on UNrecovered items stay
    in it. Refusal / transport / api-refusal counts are round-0 VERBATIM
    (their classes and remedies are untouched). Truncation is NEVER waivable:
    ``n_truncation`` and the ``stop_reason_tally`` include the re-draw rounds'
    evidence too, so a truncating re-draw still FAILs the gate (strictly
    stricter, never weaker).
    """
    from explore_persona_space.eval.judge_pilot import (
        ArmPilotStats,
        _count_unknown_stop_reason_drops,
    )

    res = outcome.result
    n_answered = res.n_total_draws - res.n_transport_lost_draws - res.n_api_refusal_draws
    n_content_post = (
        res.n_refusal_draws + res.n_truncation_dropped_draws + outcome.residual_malformed_draws
    )
    tally = dict(res.stop_reason_tally)
    for key, v in outcome.redraw_stop_reason_tally.items():
        tally[key] = tally.get(key, 0) + v
    # #2124 (rule 29) per-item completeness, off the reduce's pre-seeded scores
    # map (scores[item] is None marks all-draws-dropped) — same computation as
    # judge_pilot_gate. Required fields since #2124; this call site predated
    # them and crashed TypeError (pre-existing on main, surfaced by #2152's
    # sibling-suite sweep).
    n_items_zero_valid = sum(1 for v in res.scores.values() if v is None)
    return ArmPilotStats(
        n_items=len(item_ids),
        n_items_zero_valid=n_items_zero_valid,
        frac_items_complete=(len(item_ids) - n_items_zero_valid) / max(1, len(item_ids)),
        n_draws=res.n_total_draws,
        n_scored=n_answered - n_content_post,
        n_content_dropped=n_content_post,
        n_refusal=res.n_refusal_draws,
        n_truncation=res.n_truncation_dropped_draws + outcome.n_redraw_truncation_draws,
        n_transport_lost=res.n_transport_lost_draws,
        n_api_refusal=res.n_api_refusal_draws,
        n_unknown_stop_reason_drops=_count_unknown_stop_reason_drops(save_raw, item_ids),
        parse_fail_rate=(n_content_post - res.n_refusal_draws) / max(1, n_answered),
        stop_reason_tally=tally,
        waived=False,
    )


# ── Phase: pilot-filter (rule 26, post-recovery) ─────────────────────────────────


def run_pilot_filter(args) -> int:
    """Rule-26 pilot gate per TRAIT rubric (arms = corpora), POST-recovery.

    ``judge_pilot_gate`` cannot accept merged/post-recovery accounting, so
    this phase reproduces its EXACT loop script-side — same seeded-subsample
    arithmetic, same ``_gate_verdict`` clauses and thresholds (2% rule-26(b)
    bar, truncation never waivable, 10-effective-draw floor; never weakened)
    — over ``judge_with_redraw`` outcomes, and writes the report to the same
    location labeled ``post_recovery: true``.
    """
    from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
    from explore_persona_space.eval.judge_pilot import (
        PilotGateReport,
        _gate_verdict,
        _seeded_subsample,
    )

    report_dir = Path(args.pilot_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    all_pass = True
    for trait in parse_list(args.traits):
        rubric = load_trait_rubric(args.pv_root, args.rubric_file, trait)
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for corpus in parse_list(args.corpora):
            items, _ = filter_items(args, corpus, trait)
            if not items:
                raise RuntimeError(f"{corpus}/{trait}: empty filter-candidate arm")
            arms[corpus] = items
        # judge_pilot_gate's own per-arm sizing arithmetic (judge_pilot.py:367).
        per_arm_items = max(1, args.pilot_total_draws // (len(arms) * PILOT_N_DRAWS))
        arm_stats: dict[str, object] = {}
        redraw_acct: dict[str, dict] = {}
        for corpus, items in arms.items():
            sub = _seeded_subsample(items, per_arm_items, seed=args.seed, arm=corpus)
            save_raw = (
                Path(args.judge_root) / "pilot_raw" / trait / f"judge_raw_pilot_{corpus}.json"
            )
            save_raw.parent.mkdir(parents=True, exist_ok=True)
            outcome = judge_with_redraw(
                sub,
                rubric,
                n_draws=PILOT_N_DRAWS,
                cache_dir=Path(args.judge_root) / "pilot_cache" / trait / corpus,
                save_raw=save_raw,
                max_tokens=args.judge_max_tokens,
            )
            arm_stats[corpus] = post_recovery_arm_stats(
                outcome, save_raw, {iid for iid, _q, _a in sub}
            )
            redraw_acct[corpus] = redraw_accounting(outcome)
        failures, warnings = _gate_verdict(
            arm_stats,
            max_tokens=args.judge_max_tokens,
            parse_fail_threshold=PILOT_PARSE_FAIL_THRESHOLD,
            min_effective_draws_per_arm=PILOT_MIN_EFFECTIVE_DRAWS,
        )
        passed = not failures
        report = PilotGateReport(
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            failures=failures,
            warnings=warnings,
            arms=arm_stats,
            judge_model=DEFAULT_JUDGE_MODEL,
            max_tokens=args.judge_max_tokens,
            n_total_draws=sum(a.n_draws for a in arm_stats.values()),
            parse_fail_threshold=PILOT_PARSE_FAIL_THRESHOLD,
            rubric_hash=hashlib.sha256(rubric.encode("utf-8")).hexdigest()[:16],
        )
        report_path = report_dir / f"filter_{trait}.json"
        atomic_write_json(
            {**report.to_json(), "post_recovery": True, "redraw": redraw_acct}, report_path
        )
        logger.info(
            "[pilot-filter] trait=%s verdict=%s (post_recovery=true) -> %s",
            trait,
            report.verdict,
            report_path,
        )
        all_pass &= passed
    if not all_pass:
        raise RuntimeError("[pilot-filter] at least one trait rubric FAILED the pilot gate")
    return 0


def check_pilot_pass(report_path: Path, wave_calls: int, skip: bool) -> None:
    """Refuse a >=5k-call production wave without a PASSing pilot report (rule 26)."""
    if wave_calls < PILOT_WAVE_FLOOR or skip:
        return
    if not report_path.exists():
        raise RuntimeError(
            f"production judge wave of {wave_calls} calls >= {PILOT_WAVE_FLOOR} requires a "
            f"pilot gate PASS first ({report_path} missing) — run the pilot phase, or pass "
            f"--skip-pilot-gate with a recorded justification"
        )
    rep = json.loads(report_path.read_text())
    if not rep.get("passed"):
        raise RuntimeError(f"pilot gate report {report_path} verdict={rep.get('verdict')} != PASS")


# ── Phase: judge-filter (4b-3 judging) ───────────────────────────────────────────


def run_judge_filter(args) -> int:
    """Judge the candidate union per (corpus, trait) via the Batch machinery.

    Runs through :func:`judge_with_redraw`: MALFORMED parse-fail items are
    re-issued at the identical instrument (<=2 sync re-draw rounds, first
    parsed judgment wins); the persisted ``scores`` are the merged view and
    the payload carries the full re-draw accounting.
    """
    for trait in parse_list(args.traits):
        rubric = load_trait_rubric(args.pv_root, args.rubric_file, trait)
        for corpus in parse_list(args.corpora):
            out_dir = Path(args.selections_dir) / corpus / trait
            out_path = out_dir / "filter_scores.json"
            if out_path.exists() and not args.force:
                # Existence-keyed skip (--force re-judges): the stale-coverage
                # residual is backstopped by run_apply_filter's candidates ⊆
                # scores assert (M6) — a re-run of prepare-filter that widened
                # the candidate set fails loud downstream, never silently
                # selects from a smaller judged pool.
                logger.info("[judge-filter] %s/%s already judged — skip (--force)", corpus, trait)
                continue
            items, cand = filter_items(args, corpus, trait)
            wave = len(items) * args.filter_draws
            check_pilot_pass(
                Path(args.pilot_report_dir) / f"filter_{trait}.json", wave, args.skip_pilot_gate
            )
            t0 = time.time()
            outcome = judge_with_redraw(
                items,
                rubric,
                n_draws=args.filter_draws,
                cache_dir=Path(args.judge_root) / "cache" / f"{corpus}__{trait}",
                save_raw=Path(args.judge_root) / "raw" / f"filter_{corpus}__{trait}.json",
                max_tokens=args.judge_max_tokens,
                dry_run=args.dry_run,
            )
            res = outcome.result
            if args.dry_run:
                logger.info("[judge-filter] DRY RUN %s/%s: %d items routed", corpus, trait, wave)
                continue
            payload = {
                "schema": SELECT_SCHEMA_VERSION,
                "corpus": corpus,
                "trait": trait,
                "n_items": len(items),
                "n_draws": args.filter_draws,
                "judge": {
                    "max_tokens": args.judge_max_tokens,
                    "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
                },
                "telemetry": {
                    "n_total_draws": res.n_total_draws,
                    "n_dropped_draws": res.n_dropped_draws,
                    "n_refusal_draws": res.n_refusal_draws,
                    "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
                    "n_transport_lost_draws": res.n_transport_lost_draws,
                    "n_api_refusal_draws": res.n_api_refusal_draws,
                },
                "redraw": redraw_accounting(outcome),
                "unscored_sample_ids": sorted(s for s, v in outcome.scores.items() if v is None),
                "scores": {
                    s: (None if v is None else round(float(v), 4))
                    for s, v in outcome.scores.items()
                },
                "provenance": {"candidates_n": cand["n_candidates"]},
                "meta": repro_meta("issue2224_select.judge-filter"),
            }
            atomic_write_json(payload, out_path)
            logger.info(
                "[judge-filter] %s/%s: %d items in %.0fs; drops=%d transport=%d api_refusal=%d "
                "redrawn=%d recovered=%d residual_malformed=%d "
                "(transport/api-refusal rows are RE-JUDGEABLE — rules 24/28)",
                corpus,
                trait,
                len(items),
                time.time() - t0,
                res.n_dropped_draws,
                res.n_transport_lost_draws,
                res.n_api_refusal_draws,
                outcome.n_items_redrawn,
                outcome.n_recovered,
                outcome.residual_malformed_draws,
            )
    return 0


# ── Phase: apply-filter (4b-3 re-selection) ──────────────────────────────────────


def run_apply_filter(args) -> int:
    """Keep score < 1, re-select top-500 among survivors, equalize-down, floor 300."""
    arms = parse_list(args.selection_arms)
    for corpus in parse_list(args.corpora):
        pool, pool_prov = load_pool_map(args.pools_dir, corpus)
        for trait in parse_list(args.traits):
            sel_dir = Path(args.selections_dir) / corpus / trait
            fs_path = sel_dir / "filter_scores.json"
            if not fs_path.exists():
                raise RuntimeError(f"{fs_path} missing — run --phase judge-filter first")
            fscores = json.loads(fs_path.read_text())["scores"]
            cand_path = sel_dir / "filter_candidates.json"
            if not cand_path.exists():
                raise RuntimeError(f"{cand_path} missing — run --phase prepare-filter first")
            cand = json.loads(cand_path.read_text())
            # M6: the judged score set must COVER the current candidate set —
            # a stale/partial filter_scores.json (prepare-filter re-run with a
            # different --filter-top-k, or a partially-judged wave) would
            # otherwise silently select from a smaller candidate pool.
            missing_cov = [s for s in cand["sample_ids"] if s not in fscores]
            if missing_cov:
                raise RuntimeError(
                    f"{corpus}/{trait}: filter_scores.json covers only "
                    f"{len(fscores)}/{len(cand['sample_ids'])} candidates (first 5 "
                    f"missing: {missing_cov[:5]}) — stale/partial judge output (M6); "
                    f"re-run --phase judge-filter --force"
                )
            meta, scores = load_scores(args.scores_dir, corpus, trait)
            prov = {
                "score_file": meta["_score_file"],
                "pool": pool_prov,
                "filter_scores": {"path": str(fs_path), "sha256": sha256_file(fs_path)},
                "filter_candidates": {"path": str(cand_path), "sha256": sha256_file(cand_path)},
                "keep_below": FILTER_KEEP_BELOW,
                "floor": FILTER_FLOOR,
            }
            # Per-arm survivor selection (dropped/unscored judge rows are EXCLUDED —
            # drop-never-coerce: no verdict, no pass through the filter; counts kept).
            per_arm_sel: dict[str, list[str]] = {}
            per_arm_stats: dict[str, dict] = {}
            for arm in arms:
                order = ranked_ids(scores, arm)
                in_cand = [s for s in order if s in fscores]
                survivors = [
                    s
                    for s in in_cand
                    if fscores[s] is not None and float(fscores[s]) < FILTER_KEEP_BELOW
                ]
                per_arm_sel[arm] = survivors[:TOP_N]
                per_arm_stats[arm] = {
                    "n_candidates": len(in_cand),
                    "n_unscored": sum(1 for s in in_cand if fscores[s] is None),
                    "n_survivors": len(survivors),
                    "n_selected_prelim": len(per_arm_sel[arm]),
                    "collapsed": len(survivors) < FILTER_FLOOR,
                }
            live = [a for a in arms if not per_arm_stats[a]["collapsed"]]
            equalized_n = min(len(per_arm_sel[a]) for a in live) if live else 0
            for arm in arms:
                st = per_arm_stats[arm]
                extra = {
                    "filter": st,
                    "equalized_n": equalized_n,
                    "equalize_pool_arms": live,
                }
                if st["collapsed"]:
                    # A finding about this corpus/trait — reported, never backfilled
                    # and never trained (plan §4 floor).
                    write_cell(
                        args,
                        corpus,
                        trait,
                        arm,
                        "top_filtered",
                        per_arm_sel[arm],
                        scores,
                        pool,
                        prov,
                        extra=extra,
                        status="filter-collapsed",
                    )
                else:
                    write_cell(
                        args,
                        corpus,
                        trait,
                        arm,
                        "top_filtered",
                        per_arm_sel[arm][:equalized_n],
                        scores,
                        pool,
                        prov,
                        extra=extra,
                    )
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────

PHASES = {
    "select": run_select,
    "prepare-filter": run_prepare_filter,
    "pilot-filter": run_pilot_filter,
    "judge-filter": run_judge_filter,
    "apply-filter": run_apply_filter,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #2224 4b-2/3 selection + judge-filter (plan v3 §4)."
    )
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--list-phases", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--corpora", default=",".join(DEFAULT_CORPORA))
    parser.add_argument("--traits", default=",".join(TRAITS))
    parser.add_argument("--selection-arms", default=",".join(DEFAULT_SELECTION_ARMS))
    parser.add_argument("--scores-dir", type=Path, default=SCREENING_SCORES_DIR_DEFAULT)
    parser.add_argument(
        "--pools-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_2224" / "pools"
    )
    parser.add_argument("--selections-dir", type=Path, default=SELECTIONS_DIR_DEFAULT)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR_DEFAULT)
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT_DEFAULT)
    parser.add_argument("--pilot-report-dir", type=Path, default=PILOT_REPORT_DIR_DEFAULT)
    parser.add_argument("--tokenizer", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=2048, help="§11 truncation diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-top-k", type=int, default=FILTER_TOP_K)
    parser.add_argument(
        "--filter-draws", type=int, default=1, help="judge draws/item (plan §9 budgets 1)"
    )
    parser.add_argument(
        "--judge-max-tokens", type=int, default=1024, help="rule-23 floor for the rationale rubric"
    )
    parser.add_argument("--pilot-total-draws", type=int, default=200)
    parser.add_argument("--pv-root", type=Path, default=None, help="persona_vectors clone root")
    parser.add_argument(
        "--rubric-file", type=Path, default=None, help="explicit rubric override (smoke fixtures)"
    )
    parser.add_argument("--allow-short-pool", action="store_true", help="smoke slices < top-N")
    parser.add_argument("--skip-pilot-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="judge routing only, 0 API calls")
    parser.add_argument("--force", action="store_true", help="re-judge over an existing output")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        return 0
    if args.import_check:
        import importlib

        for mod in ("numpy", "transformers"):
            importlib.import_module(mod)
        from transformers import AutoTokenizer  # noqa: F401

        from explore_persona_space.eval import DEFAULT_JUDGE_MODEL  # noqa: F401
        from explore_persona_space.eval import batch_judge as _bj_chk
        from explore_persona_space.eval import graded_judge as _gj_chk
        from explore_persona_space.eval.judge_pilot import (  # noqa: F401
            ArmPilotStats,
            PilotGateReport,
            _count_unknown_stop_reason_drops,
            _gate_verdict,
            _seeded_subsample,
        )
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from issue778_lib import load_trait_data  # noqa: F401

        # Attribute-level resolution of every deferred symbol the re-draw
        # recovery consumes (a bare module import is insufficient — #1689 r2/3/4).
        for _name in (
            "is_transport_error_dict",
            "is_api_refusal_error_dict",
            "is_truncation_stop_reason",
        ):
            getattr(_bj_chk, _name)
        for _name in ("judge_graded", "_score_from_parsed", "_is_refusal_parsed", "JudgeResult"):
            getattr(_gj_chk, _name)

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_select")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required; see --list-phases")
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
