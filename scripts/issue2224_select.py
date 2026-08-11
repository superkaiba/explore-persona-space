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
  call per rubric) via ``eval.judge_pilot.judge_pilot_gate``.
- ``--phase judge-filter``    the paper's trait-expression filter judging via
  ``eval.graded_judge.judge_graded`` (routes through the #663-hardened Batch
  client — this script never inlines an API loop). Drop-never-coerce;
  content-drop vs transport-loss vs api-refusal counts reported per
  (corpus, trait).
- ``--phase apply-filter``    keep score < 1 (the paper's exact filter),
  re-select top-500 among survivors, equalize-down to the min survivor count
  across arms within (corpus, trait) (dose matching, plan §4), floor 300 —
  below floor the cell is reported ``filter-collapsed`` (a finding), never
  backfilled.

Selection determinism: ranking uses ``np.lexsort`` with the sample_id as an
explicit secondary tie-break key (stable + machine-independent under score
ties — the #1946 argsort-tie lesson); the shared random-500 is seeded via
``stable_seed(seed, corpus, trait)`` (sha-based, PYTHONHASHSEED-proof).

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
from pathlib import Path

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


# ── Phase: pilot-filter (rule 26) ────────────────────────────────────────────────


def run_pilot_filter(args) -> int:
    """One rule-26 pilot gate per TRAIT rubric (arms = corpora), reports persisted."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    report_dir = Path(args.pilot_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    all_pass = True
    for trait in parse_list(args.traits):
        rubric = load_trait_rubric(args.pv_root, args.rubric_file, trait)
        arms = {}
        for corpus in parse_list(args.corpora):
            items, _ = filter_items(args, corpus, trait)
            arms[corpus] = items
        report_path = report_dir / f"filter_{trait}.json"
        rep = judge_pilot_gate(
            arms,
            rubric,
            max_tokens=args.judge_max_tokens,
            cache_dir=Path(args.judge_root) / "pilot_cache" / trait,
            save_raw_dir=Path(args.judge_root) / "pilot_raw" / trait,
            n_draws=2,
            target_total_draws=args.pilot_total_draws,
            report_path=report_path,
            seed=args.seed,
        )
        logger.info("[pilot-filter] trait=%s verdict=%s -> %s", trait, rep.verdict, report_path)
        all_pass &= rep.passed
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
    """Judge the candidate union per (corpus, trait) via the Batch machinery."""
    from explore_persona_space.eval.graded_judge import judge_graded

    for trait in parse_list(args.traits):
        rubric = load_trait_rubric(args.pv_root, args.rubric_file, trait)
        for corpus in parse_list(args.corpora):
            out_dir = Path(args.selections_dir) / corpus / trait
            out_path = out_dir / "filter_scores.json"
            if out_path.exists() and not args.force:
                logger.info("[judge-filter] %s/%s already judged — skip (--force)", corpus, trait)
                continue
            items, cand = filter_items(args, corpus, trait)
            wave = len(items) * args.filter_draws
            check_pilot_pass(
                Path(args.pilot_report_dir) / f"filter_{trait}.json", wave, args.skip_pilot_gate
            )
            t0 = time.time()
            res = judge_graded(
                items,
                rubric,
                n_draws=args.filter_draws,
                cache_dir=Path(args.judge_root) / "cache" / f"{corpus}__{trait}",
                save_raw=Path(args.judge_root) / "raw" / f"filter_{corpus}__{trait}.json",
                max_tokens=args.judge_max_tokens,
                dry_run=args.dry_run,
            )
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
                "unscored_sample_ids": sorted(s for s, v in res.scores.items() if v is None),
                "scores": {
                    s: (None if v is None else round(float(v), 4)) for s, v in res.scores.items()
                },
                "provenance": {"candidates_n": cand["n_candidates"]},
                "meta": repro_meta("issue2224_select.judge-filter"),
            }
            atomic_write_json(payload, out_path)
            logger.info(
                "[judge-filter] %s/%s: %d items in %.0fs; drops=%d transport=%d api_refusal=%d "
                "(transport/api-refusal rows are RE-JUDGEABLE — rules 24/28)",
                corpus,
                trait,
                len(items),
                time.time() - t0,
                res.n_dropped_draws,
                res.n_transport_lost_draws,
                res.n_api_refusal_draws,
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
            meta, scores = load_scores(args.scores_dir, corpus, trait)
            prov = {
                "score_file": meta["_score_file"],
                "pool": pool_prov,
                "filter_scores": {"path": str(fs_path), "sha256": sha256_file(fs_path)},
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

        from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from issue778_lib import load_trait_data  # noqa: F401

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_select")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required; see --list-phases")
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
