#!/usr/bin/env python
"""#1947 sycophancy-arm recovery — positive-pool completion for ``syc-icl``.

The single-visit fleet dropped the sycophancy behavior at the plan §7 gate-2
yield check: ``syc-icl`` emitted 232 positives against the 300 emit target
(floor 240), while the other 11 pools (3 syc contexts + 8 imp/cas) emitted
EXACTLY 300 and both syc negative panels emitted 300.

Two facts set this driver's target (both read off the fleet's own artifacts):

* the mixes yield gate equalizes DOWN to ``min(300, min_over_kept_behaviors)``
  — so admitting syc at, say, 265 would re-equalize the already-trained
  imp/cas cells to 265. Recovery therefore targets **300**, not the 240 floor;
* the positive pool dedupes tranche rows on ``question_id`` against the
  first-sample kept set, so the reachable ceiling is
  ``len(kept_first_rows) + (bank_size - |union_qids|)``.

The ONLY thing this driver changes versus the realized fleet recipe is the
REQUEST BUDGET (the brief's authorized fix): same 300-question extended bank,
same ICL source context, same 6 exhibit-instruction variants, same generator
(``claude-sonnet-4-5-20250929`` @ temp 1.0, tagged instruct-and-strip), same
judge instrument (graded 0-100, 3 draws, ``max_tokens=500``), same keep rule
(``_judge_and_filter``) and same question_id-dedupe merge — all imported from
``issue1947_datagen`` / ``artifacts.datagen`` rather than re-implemented.

Each recovery tranche restricts the composition grid to the questions the
union does NOT yet carry (never-attempted + attempted-but-never-yielded) and
requests every (question x variant) cell, on a FRESH generation cache (an
identical request against the fleet's ``gen_cache`` would re-serve the cached
completion and re-derive the same rejection — the #1090-class cache trap).

Phases (``--phase``, comma list or ``all``):

  ``stage``    fetch the pool sidecars from the HF data repo (idempotent);
  ``audit``    classify the realized reject pool (content-low vs parse vs
               transport, per llm-judging rules 23/24) -> ``recovery_audit.json``;
  ``rejudge``  re-judge the fully-censored (all-draws-parse-error) first-sample
               items against a FRESH cache, same instrument -> merge any that
               now pass (rule 24: a censored draw is freely re-judgeable);
  ``tranche``  up to ``--max-tranches`` recovery tranches, stopping the moment
               the union reaches the emit target;
  ``emit``     rewrite ``pos.jsonl`` at EXACTLY the emit target (the factory's
               seeded-sample semantics) + ``recovery_meta.json``.

A run that cannot reach the emit target writes its record and exits 3 — the
same-construct budget is then exhausted and the disposition (bank widening /
pushback-flip / fleet re-equalization) is an orchestrator decision, never a
silent construct change here.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps before any heavy import

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_datagen as dg  # noqa: E402

from explore_persona_space.artifacts.behavior import DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    DATAGEN_JUDGE_MAX_TOKENS,
    POSITIVE,
    GenCandidate,
    _compose_positive_requests,
    _default_generate_fn,
    _judge_and_filter,
    _read_raw,
    _rng,
    _train_row,
    _write_raw,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1947.syc_recovery")

BEH_KEY = "syc"
CTX_KEY = "icl"
POOL = f"{BEH_KEY}-{CTX_KEY}"
EMIT_TARGET = dg.POS_EMIT_N  # 300 — the fleet's realized per-pool emit count
HARD_FLOOR = dg.POS_FLOOR_HARD  # 240 — plan §7 gate 2
N_JUDGE_DRAWS = dg.N_JUDGE_DRAWS
JUDGE_MAX_TOKENS = DATAGEN_JUDGE_MAX_TOKENS  # 500 — fleet parity, judge-cache-key input
GEN_MODEL = DEFAULT_JUDGE_MODEL  # the factory's generator default (gen_manifest parity)
GEN_TEMPERATURE = 1.0
HF_DATA_REPO = dg.HF_DATA_REPO
HUB_POOL_PREFIX = f"issue1947_singlevisit/raw_completions/datagen/positives/{POOL}"
STAGE_FILES = (
    "raw_pos.jsonl",
    "judge_raw_pos.json",
    "raw_pos_topup.jsonl",
    "judge_raw_pos_topup.json",
    "topup_record.json",
    "salvage_meta.json",
    "gen_manifest.json",
)
PHASES = ("stage", "audit", "rejudge", "tranche", "emit", "stage_mix_inputs")
# The OTHER pool artifacts `issue1947_datagen.py --phase mixes` reads: the three
# already-complete syc positive pools, both syc negative panels, and the shared
# generic pool. Staged so the syc mixes rebuild locally against the FLEET's own
# realized pools (only syc-icl/pos.jsonl is this round's new artifact).
MIX_INPUT_FILES = (
    ("positives/syc-pers", "pos.jsonl"),
    ("positives/syc-bare", "pos.jsonl"),
    ("positives/syc-conv", "pos.jsonl"),
    ("negatives/syc-panel5", "neg.jsonl"),
    ("negatives/syc-panel5", "neg_meta.json"),
    ("negatives/syc-panel4bare", "neg.jsonl"),
    ("negatives/syc-panel4bare", "neg_meta.json"),
    ("generic", "pool.jsonl"),
    ("generic", "pool_meta.json"),
)
EXIT_TARGET_MISSED = 3


# ── small utils ──────────────────────────────────────────────────────────────


def _phase(name: str, **kv) -> None:
    extras = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[phase={name}] {extras}".rstrip(), flush=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _meta_block() -> dict:
    """Reproducibility metadata every emitted record carries (CLAUDE.md)."""
    import platform

    return {
        "git_commit": _git_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "judge_model": dg.BEHAVIORS[dg.BEHAVIOR_BY_KEY[BEH_KEY]].judge_model,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "n_judge_draws": N_JUDGE_DRAWS,
        "emit_target": EMIT_TARGET,
        "hard_floor": HARD_FLOOR,
    }


def _behavior():
    """The fleet's behavior object: registered sycophancy with the train bank
    swapped for the 300 extended-bank questions (``dg._behavior_1947``)."""
    cfg = _cfg()
    return dg._behavior_1947(cfg, BEH_KEY)


_CFG_CACHE: dict[str, object] = {}


def _cfg(out_root: Path | None = None):
    if "cfg" not in _CFG_CACHE:
        if out_root is None:
            raise RuntimeError("_cfg() called before initialisation")
        _CFG_CACHE["cfg"] = dg.Cfg(
            out_root=out_root,
            phases=PHASES,
            behaviors=(BEH_KEY,),
            contexts=(CTX_KEY,),
        )
    return _CFG_CACHE["cfg"]


def _pool_dir() -> Path:
    return _cfg().positives_dir / POOL


# ── stage ────────────────────────────────────────────────────────────────────


def _stage_extended_bank() -> int:
    """Stage the 300-question extended bank the behavior's train bank reads
    (``dg.new_questions`` -> ``<out_root>/banks/sycophancy_extended.json``)."""

    behavior_name = dg.BEHAVIOR_BY_KEY[BEH_KEY]
    dest = _cfg().banks_dir / f"{behavior_name}_extended.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        HF_DATA_REPO,
        f"issue1947_singlevisit/datagen_meta/banks/{behavior_name}_extended.json",
        dest,
    )
    return len(json.loads(dest.read_text(encoding="utf-8"))["new_questions"])


def phase_stage_mix_inputs() -> dict:
    """Stage the OTHER pool artifacts the syc mixes rebuild consumes.

    Fail-loud per file: every one is a REQUIRED input to
    ``issue1947_datagen.py --phase mixes`` and a silent miss would rebuild a
    mix against a short pool."""

    cfg = _cfg()
    got: dict[str, int] = {}
    for rel_dir, name in MIX_INPUT_FILES:
        dest = cfg.out_root / rel_dir / name
        if dest.exists():
            got[f"{rel_dir}/{name}"] = dest.stat().st_size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Every mix input lives under the datagen raw-completions prefix (the
        # driver's own upload map: positives/ negatives/ generic/).
        hub_path = f"issue1947_singlevisit/raw_completions/datagen/{rel_dir}/{name}"
        hub.stage_hub_file(HF_DATA_REPO, hub_path, dest)
        got[f"{rel_dir}/{name}"] = dest.stat().st_size
    rows = {
        k: sum(1 for _ in open(cfg.out_root / k, "rb") if _.strip())
        for k in got
        if k.endswith(".jsonl")
    }
    _phase("stage_mix_inputs_done", files=len(got), rows=json.dumps(rows, sort_keys=True))
    _write_json(
        _cfg().out_root / "recovery_mix_inputs.json",
        {"meta": _meta_block(), "staged": got, "rows": rows},
    )
    return {"staged": got, "rows": rows}


def phase_stage() -> dict:
    """Fetch the realized pool sidecars from the HF data repo into the local
    pool dir (fail-loud on a missing REQUIRED input; optional ones may be
    absent for a pool that never fired a topup)."""
    out_dir = _pool_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_bank = _stage_extended_bank()
    required = {"raw_pos.jsonl", "judge_raw_pos.json"}
    got: dict[str, int] = {}
    for name in STAGE_FILES:
        dest = out_dir / name
        if dest.exists():
            got[name] = dest.stat().st_size
            continue
        try:
            hub.stage_hub_file(HF_DATA_REPO, f"{HUB_POOL_PREFIX}/{name}", dest)
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            if name in required:
                raise RuntimeError(
                    f"[stage] required pool input {name!r} not fetchable from "
                    f"{HF_DATA_REPO}:{HUB_POOL_PREFIX} ({type(exc).__name__}: {exc})"
                ) from exc
            logger.info("[stage] optional input %s absent on the Hub — skipping", name)
            continue
        got[name] = dest.stat().st_size
    missing = sorted(required - set(got))
    if missing:
        raise RuntimeError(f"[stage] required inputs still missing after fetch: {missing}")
    _phase("stage_done", files=len(got), bank_questions=n_bank, pool=POOL)
    return {"staged": got, "pool_dir": str(out_dir), "bank_questions": n_bank}


# ── audit ────────────────────────────────────────────────────────────────────


def _per_item_scores(save_raw: Path) -> tuple[dict[str, list[float]], dict[str, int], list[dict]]:
    """(kept per-draw scores by request_id, error-draw counts, error dicts)."""
    payload = json.loads(save_raw.read_text(encoding="utf-8"))
    scores: dict[str, list[float]] = defaultdict(list)
    errors: dict[str, int] = Counter()
    error_dicts: list[dict] = []
    for key, val in payload.get("all_scores", {}).items():
        rid = key.rsplit("__", 2)[0]
        if isinstance(val, dict) and "error" in val:
            errors[rid] += 1
            error_dicts.append(val)
        elif isinstance(val, dict) and isinstance(val.get("score"), (int, float)):
            scores[rid].append(float(val["score"]))
        else:
            errors[rid] += 1
            error_dicts.append({"error": True, "reasoning": "unparseable_record"})
    return dict(scores), dict(errors), error_dicts


def _classify_pass(save_raw: Path, label: str, threshold: float) -> dict:
    from explore_persona_space.eval.batch_judge import is_transport_error_dict

    scores, errors, error_dicts = _per_item_scores(save_raw)
    rids = set(scores) | set(errors)
    censored = sorted(r for r in rids if not scores.get(r))
    partial = sorted(r for r in rids if scores.get(r) and errors.get(r, 0) > 0)
    kept = [r for r in rids if scores.get(r) and mean(scores[r]) > threshold]
    low = [r for r in rids if scores.get(r) and mean(scores[r]) <= threshold]
    transport = sum(1 for d in error_dicts if is_transport_error_dict(d))
    reasons = Counter(str(d.get("reasoning"))[:40] for d in error_dicts)
    return {
        "label": label,
        "n_items": len(rids),
        "n_kept": len(kept),
        "n_content_low": len(low),
        "n_fully_censored_items": len(censored),
        "fully_censored_rids": censored,
        "n_partial_censored_items": len(partial),
        "n_error_draws": sum(errors.values()),
        "n_error_draws_transport_class": transport,
        "n_error_draws_content_class": sum(errors.values()) - transport,
        "error_reason_histogram": dict(reasons),
        "per_request_accept_rate": round(len(kept) / len(rids), 4) if rids else None,
    }


def phase_audit() -> dict:
    """Drop-class audit of the realized reject pool (brief step 0)."""
    out_dir = _pool_dir()
    behavior = _behavior()
    passes = []
    for name, label in (
        ("judge_raw_pos.json", "first_sample"),
        ("judge_raw_pos_topup.json", "topup_tranche"),
    ):
        p = out_dir / name
        if p.exists():
            passes.append(_classify_pass(p, label, behavior.threshold))
    union, info = dg._salvage_reconstruct_union(behavior, out_dir)
    union_qids = {c.request.question_id for c in union}
    bank_qids = {
        f"{behavior.name}-trainq-{i:04d}" for i in range(len(behavior.train_question_bank))
    }
    attempted: Counter = Counter()
    accepted: Counter = Counter()
    for raw_name, judge_name in (
        ("raw_pos.jsonl", "judge_raw_pos.json"),
        ("raw_pos_topup.jsonl", "judge_raw_pos_topup.json"),
    ):
        raw_p, judge_p = out_dir / raw_name, out_dir / judge_name
        if not (raw_p.exists() and judge_p.exists()):
            continue
        scores, _e, _d = _per_item_scores(judge_p)
        for cand in _read_raw(raw_p):
            if cand.completion is None:
                continue
            qid = cand.request.question_id
            attempted[qid] += 1
            draws = scores.get(cand.request.request_id) or []
            if draws and mean(draws) > behavior.threshold:
                accepted[qid] += 1
    yield_by_attempts: dict[int, dict[str, int]] = {}
    for qid, n in attempted.items():
        row = yield_by_attempts.setdefault(int(n), {"n_questions": 0, "n_yielded": 0})
        row["n_questions"] += 1
        if accepted[qid] > 0:
            row["n_yielded"] += 1
    available = sorted(bank_qids - union_qids)
    audit = {
        "meta": _meta_block(),
        "pool": POOL,
        "passes": passes,
        "union": {
            **info,
            "union_rows": len(union),
            "union_distinct_qids": len(union_qids),
            "shortfall_to_emit_target": max(0, EMIT_TARGET - len(union)),
            "shortfall_to_hard_floor": max(0, HARD_FLOOR - len(union)),
        },
        "question_pool": {
            "bank_size": len(bank_qids),
            "n_attempted": len(attempted),
            "n_never_attempted": len(bank_qids - set(attempted)),
            "n_attempted_never_yielded": len([q for q in attempted if accepted[q] == 0]),
            "n_available_for_recovery": len(available),
            "reachable_ceiling_rows": len(union) + len(available),
            "per_request_accept_rate_overall": (
                round(sum(accepted.values()) / sum(attempted.values()), 4) if attempted else None
            ),
            "yield_by_attempt_count": {str(k): v for k, v in sorted(yield_by_attempts.items())},
        },
        "verdict": {
            "recovery_of_censored_draws_alone_clears_hard_floor": (
                len(union) + sum(p["n_fully_censored_items"] for p in passes) >= HARD_FLOOR
            ),
            "recovery_of_censored_draws_alone_clears_emit_target": (
                len(union) + sum(p["n_fully_censored_items"] for p in passes) >= EMIT_TARGET
            ),
            "emit_target_reachable_in_principle": len(union) + len(available) >= EMIT_TARGET,
            "dominant_reject_class": "content_scored_low",
        },
    }
    _write_json(out_dir / "recovery_audit.json", audit)
    _phase(
        "audit_done",
        union=len(union),
        censored=sum(p["n_fully_censored_items"] for p in passes),
        available_qids=len(available),
        ceiling=len(union) + len(available),
    )
    return audit


# ── rejudge (censored draws — llm-judging rules 23/24) ───────────────────────


def _judge_fn():
    """The fleet's judge instrument, bound to its max_tokens budget."""

    def _fn(items, eval_prompt, **kw):
        kw["max_tokens"] = JUDGE_MAX_TOKENS
        return judge_graded(items, eval_prompt, **kw)

    return _fn


def phase_rejudge() -> dict:
    """Re-judge the fully-censored first-sample items against a FRESH cache.

    Every censored draw is a CONTENT-class ``parse_error`` (the audit records
    zero transport-class draws), and a parse-error entry is persisted in the
    fleet's judge cache — so the re-judge MUST use a fresh cache dir or the
    poisoned entry is re-served (llm-judging rule 24(ii)).

    IDEMPOTENT BY RECORD (never re-judged): the rubric-keyed judge cache holds
    ONE key per (rubric, question, completion), so all ``n_draws`` repeats of an
    item share it — a cache-served RE-run collapses the multi-draw mean onto a
    single stored score (llm-judging rule 4 independence lost) and silently
    changes the kept set. Measured on this pool: the first pass kept 3 of 4
    censored items; an immediate second pass read
    ``cache_stats {hits: 12, misses: 0}`` with every item's 3 draws identical
    (75/75/75, 45/45/45) and kept 1. The FIRST fresh-cache pass is therefore
    authoritative and this phase resumes from its persisted record."""
    out_dir = _pool_dir()
    behavior = _behavior()
    record_path = out_dir / "rejudge_censored.json"
    save_raw_path = out_dir / "judge_raw_pos_rejudge.json"
    if record_path.exists() and save_raw_path.exists():
        rec = json.loads(record_path.read_text(encoding="utf-8"))
        by_rid = {c.request.request_id: c for c in _read_raw(out_dir / "raw_pos.jsonl")}
        kept = [by_rid[r] for r in rec.get("kept_request_ids", []) if r in by_rid]
        logger.info(
            "[rejudge] resumed from record (%d kept of %d censored) — NOT re-judged "
            "(warm-cache re-run would collapse the multi-draw mean)",
            len(kept),
            rec.get("n_censored"),
        )
        _phase("rejudge_resumed", kept=len(kept))
        return {"n_censored": rec.get("n_censored", 0), "kept": kept, "record": rec}
    audit_p = out_dir / "recovery_audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else phase_audit()
    censored = [
        rid
        for p in audit["passes"]
        if p["label"] == "first_sample"
        for rid in p["fully_censored_rids"]
    ]
    if not censored:
        _write_json(out_dir / "rejudge_censored.json", {"meta": _meta_block(), "n_censored": 0})
        return {"n_censored": 0, "kept": []}
    by_rid = {c.request.request_id: c for c in _read_raw(out_dir / "raw_pos.jsonl")}
    cands = [by_rid[r] for r in censored if r in by_rid and by_rid[r].completion is not None]
    kept, _drops, jr, scoreinfo = _judge_and_filter(
        behavior,
        cands,
        POSITIVE,
        judge_fn=_judge_fn(),
        n_judge_draws=N_JUDGE_DRAWS,
        cache_dir=out_dir / f"judge_cache_rejudge_mt{JUDGE_MAX_TOKENS}",
        save_raw=out_dir / "judge_raw_pos_rejudge.json",
    )
    record = {
        "meta": _meta_block(),
        "n_censored": len(censored),
        "n_rejudged": len(cands),
        "n_kept": len(kept),
        "kept_request_ids": sorted(c.request.request_id for c in kept),
        "kept_question_ids": sorted(c.request.question_id for c in kept),
        "n_transport_lost_draws": getattr(jr, "n_transport_lost_draws", None),
        "n_dropped_draws": getattr(jr, "n_dropped_draws", None),
        "scores": {rid: v[0] for rid, v in sorted(scoreinfo.items())},
    }
    _write_json(out_dir / "rejudge_censored.json", record)
    _phase("rejudge_done", censored=len(censored), kept=len(kept))
    return {"n_censored": len(censored), "kept": kept, "record": record}


# ── recovery tranches ────────────────────────────────────────────────────────


def _compose_recovery_requests(behavior, ctx, qids: list[str], tranche: int, seed: int):
    """All (question x variant) cells for ``qids``, ``rt<k>-`` request ids.

    Same composer, same variant grid, same instruction style as the fleet — the
    only difference is the question SUBSET and the id prefix."""
    bank = list(behavior.train_question_bank)
    by_qid = {f"{behavior.name}-trainq-{i:04d}": q for i, q in enumerate(bank)}
    questions = [(q, by_qid[q]) for q in qids if q in by_qid]
    variants = behavior.elicitation.exhibit_instructions
    n_requests = len(questions) * len(variants)
    reqs = _compose_positive_requests(
        behavior,
        ctx,
        questions,
        n_requests,
        _rng(seed),
        "tagged",
        variants=variants,
    )
    return [replace(r, request_id=f"rt{tranche}-{r.request_id}") for r in reqs]


def _run_tranche(behavior, ctx, tranche: int, qids: list[str], seed: int) -> tuple[list, dict]:
    out_dir = _pool_dir()
    reqs = _compose_recovery_requests(behavior, ctx, qids, tranche, seed)
    raw_path = out_dir / f"raw_pos_recovery{tranche}.jsonl"
    if raw_path.exists():
        cands = _read_raw(raw_path)
        logger.info("[tranche %d] resumed %d raw candidates", tranche, len(cands))
    else:
        gen = _default_generate_fn(
            gen_model=GEN_MODEL,
            gen_temperature=GEN_TEMPERATURE,
            # FRESH per-tranche caches: an identical request against the fleet's
            # gen_cache re-serves the cached completion (same reject).
            cache_dir=out_dir / f"gen_cache_recovery{tranche}",
            checkpoint_dir=out_dir / f"gen_ckpt_recovery{tranche}",
        )
        t0 = time.time()
        cands = gen(reqs)
        _write_raw(raw_path, cands)  # persist raw the moment it returns
        logger.info(
            "[tranche %d] generated %d candidates in %.0fs", tranche, len(cands), time.time() - t0
        )
    kept, drops, jr, _scores = _judge_and_filter(
        behavior,
        cands,
        POSITIVE,
        judge_fn=_judge_fn(),
        n_judge_draws=N_JUDGE_DRAWS,
        cache_dir=out_dir / f"judge_cache_recovery{tranche}_mt{JUDGE_MAX_TOKENS}",
        save_raw=out_dir / f"judge_raw_pos_recovery{tranche}.json",
    )
    info = {
        "tranche": tranche,
        "seed": seed,
        "n_questions_attempted": len({r.question_id for r in reqs}),
        "n_requests": len(reqs),
        "n_generated": sum(1 for c in cands if c.completion is not None),
        "n_api_error_drops": drops.api_error_drops,
        "n_empty_drops": drops.empty_drops,
        "n_refusal_drops": drops.refusal_drops,
        "n_kept_rows": len(kept),
        "n_kept_distinct_qids": len({c.request.question_id for c in kept}),
        "n_transport_lost_draws": getattr(jr, "n_transport_lost_draws", None),
        "n_dropped_draws": getattr(jr, "n_dropped_draws", None),
        "per_request_accept_rate": round(len(kept) / max(1, len(cands)), 4),
    }
    return kept, info


def phase_tranche(max_tranches: int, seed_base: int) -> dict:
    """Run recovery tranches until the union hits the emit target."""
    out_dir = _pool_dir()
    behavior = _behavior()
    dg.register_contexts()
    ctx = dg.source_context(BEH_KEY, CTX_KEY)
    union, base_info = dg._salvage_reconstruct_union(behavior, out_dir)
    rej = phase_rejudge()
    seen_qids = {c.request.question_id for c in union}
    merged_rejudge = []
    for cand in rej.get("kept") or []:
        if cand.request.question_id in seen_qids:
            continue
        seen_qids.add(cand.request.question_id)
        merged_rejudge.append(cand)
    union = list(union) + merged_rejudge
    bank_qids = [
        f"{behavior.name}-trainq-{i:04d}" for i in range(len(behavior.train_question_bank))
    ]
    tranches: list[dict] = []
    for k in range(1, max_tranches + 1):
        if len(union) >= EMIT_TARGET:
            break
        available = [q for q in bank_qids if q not in seen_qids]
        if not available:
            logger.warning("[tranche %d] no available question ids remain", k)
            break
        need = EMIT_TARGET - len(union)
        logger.info(
            "[tranche %d] union=%d need=%d available_qids=%d requests=%d",
            k,
            len(union),
            need,
            len(available),
            len(available) * len(behavior.elicitation.exhibit_instructions),
        )
        kept, info = _run_tranche(behavior, ctx, k, available, seed_base + 1000 * k)
        merged = []
        for cand in kept:
            if cand.request.question_id in seen_qids:
                continue
            seen_qids.add(cand.request.question_id)
            merged.append(cand)
        union = list(union) + merged
        info.update(
            {
                "n_merged_new_qid": len(merged),
                "n_dedup_dropped_qid": len(kept) - len(merged),
                "union_after": len(union),
                "need_before": need,
            }
        )
        tranches.append(info)
        _write_json(
            out_dir / "recovery_record.json",
            {
                "meta": _meta_block(),
                "pool": POOL,
                "base_union": base_info,
                "rejudge_merged": len(merged_rejudge),
                "tranches": tranches,
                "union_rows": len(union),
                "target_reached": len(union) >= EMIT_TARGET,
            },
        )
        _phase(
            "tranche_done",
            k=k,
            requests=info["n_requests"],
            kept=info["n_kept_rows"],
            merged=len(merged),
            union=len(union),
        )
    record = {
        "meta": _meta_block(),
        "pool": POOL,
        "base_union": base_info,
        "rejudge_merged": len(merged_rejudge),
        "tranches": tranches,
        "union_rows": len(union),
        "union_distinct_qids": len({c.request.question_id for c in union}),
        "target_reached": len(union) >= EMIT_TARGET,
        "clears_hard_floor": len(union) >= HARD_FLOOR,
    }
    _write_json(out_dir / "recovery_record.json", record)
    return record


# ── emit ─────────────────────────────────────────────────────────────────────


def _recovery_union(behavior) -> list[GenCandidate]:
    """Rebuild the post-recovery union from every persisted sidecar (the same
    keep rule + question_id-dedupe merge order the tranche phase applied)."""
    out_dir = _pool_dir()
    union, _info = dg._salvage_reconstruct_union(behavior, out_dir)
    seen = {c.request.question_id for c in union}

    def _merge(raw_name: str, judge_name: str) -> None:
        nonlocal union
        raw_p, judge_p = out_dir / raw_name, out_dir / judge_name
        if not (raw_p.exists() and judge_p.exists()):
            return
        kept, _d, _jr, _s = _judge_and_filter(
            behavior,
            _read_raw(raw_p),
            POSITIVE,
            judge_fn=dg._replay_judge_fn(judge_p),
            n_judge_draws=N_JUDGE_DRAWS,
            cache_dir=out_dir / "recovery_replay_scratch",
            save_raw=out_dir / "recovery_replay_scratch" / "unused.json",
        )
        add = []
        for cand in kept:
            if cand.request.question_id in seen:
                continue
            seen.add(cand.request.question_id)
            add.append(cand)
        union = list(union) + add

    _merge("raw_pos.jsonl", "judge_raw_pos_rejudge.json")
    k = 1
    while (out_dir / f"raw_pos_recovery{k}.jsonl").exists():
        _merge(f"raw_pos_recovery{k}.jsonl", f"judge_raw_pos_recovery{k}.json")
        k += 1
    return union


def phase_emit() -> dict:
    """Rewrite ``pos.jsonl`` at EXACTLY the emit target (factory semantics)."""
    out_dir = _pool_dir()
    behavior = _behavior()
    union = _recovery_union(behavior)
    if len(union) < EMIT_TARGET:
        record = {
            "meta": _meta_block(),
            "pool": POOL,
            "union_rows": len(union),
            "emitted": 0,
            "target_reached": False,
            "clears_hard_floor": len(union) >= HARD_FLOOR,
            "note": (
                "same-construct recovery budget exhausted below the emit target; the "
                "disposition (bank widening / pushback-flip / fleet re-equalization) is "
                "an orchestrator decision — no construct change applied here"
            ),
        }
        _write_json(out_dir / "recovery_emit.json", record)
        _phase("emit_missed", union=len(union), target=EMIT_TARGET)
        return record
    seed = dg._pool_seed(BEH_KEY, CTX_KEY)
    take = random.Random(seed).sample(union, EMIT_TARGET)
    rows = [_train_row(c.request.emit_messages, c.completion) for c in take]
    sha = dg._write_jsonl(out_dir / "pos.jsonl", rows)
    record = {
        "meta": _meta_block(),
        "pool": POOL,
        "union_rows": len(union),
        "union_distinct_qids": len({c.request.question_id for c in union}),
        "emitted": len(rows),
        "emit_seed": seed,
        "pos_sha256": sha,
        "target_reached": True,
        "clears_hard_floor": True,
    }
    _write_json(out_dir / "recovery_emit.json", record)
    _phase("emit_done", emitted=len(rows), sha=sha[:12])
    return record


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="issue #1947 sycophancy positive-pool recovery")
    p.add_argument("--phase", default="all", help=f"comma list of {PHASES} or 'all'")
    p.add_argument("--out-root", default="data/issue_1947/datagen")
    p.add_argument("--max-tranches", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=dg._pool_seed(BEH_KEY, CTX_KEY))
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: F401
        from explore_persona_space.eval.graded_judge import (  # noqa: F401
            judge_graded,
            judge_result_from_save_raw,
        )
        from huggingface_hub import hf_hub_download  # noqa: F401

        print("[import-check] ok", flush=True)
        return 0
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _cfg(Path(args.out_root))
    phases = list(PHASES) if args.phase == "all" else [s.strip() for s in args.phase.split(",")]
    unknown = [s for s in phases if s not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; known: {PHASES}")
    rc = 0
    for name in phases:
        if name == "stage":
            phase_stage()
        elif name == "audit":
            phase_audit()
        elif name == "rejudge":
            phase_rejudge()
        elif name == "tranche":
            phase_tranche(args.max_tranches, args.seed_base)
        elif name == "emit":
            rec = phase_emit()
            if not rec.get("target_reached"):
                rc = EXIT_TARGET_MISSED
        elif name == "stage_mix_inputs":
            phase_stage_mix_inputs()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
