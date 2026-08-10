#!/usr/bin/env python
"""#1090 follow-up round ``fu1-margin-qwen`` (Step 9b cheap band, one driver).

Two phases, ONE unified smoke/full driver (smoke IS full at tiny N through the
SAME entrypoints; the smoke stubs ONLY the model-weights + Hub boundaries via
``issue1090_run.make_smoke_seams`` and builds its fixtures THROUGH the
production compose/replay writers):

- ``--phase gpu`` (1x GPU lane):
  [P3/c5] top-up tranche for the c5 qwen arm (amendment-v4 mechanics via the
  REUSED ``issue1090_run._run_topup_cell`` body, qwen vLLM generator seam,
  judge at ``max_tokens=300``; the FROZEN first-sample yield DV — 19/36 — is
  never touched) -> train on the union mix via the run's ``phase_train`` seams
  (save_steps=2, max_length=2048, band + closest_approach identical to the
  other cells) -> Tier-1 ladder reads -> Tier-2 generation (trained@selected +
  base, n=10) -> [P2/c3+c5 margin] teacher-forced fixed-pool margins
  (llm-judging §E2 rule 19) from the datagen_topup union pools, under BASE and
  TRAINED (c3: the HF ``issue1090/c3-sycophancy-claude/checkpoint-14``
  adapter), per tier-2 question context + the in-run source-context
  construction, with an adapter-application assert before the trained sweep ->
  upload -> sentinel.
- ``--phase judge`` (VM, API-only): Tier-2 judging for BOTH trained organisms
  at ``max_tokens=300`` (the #1090 dropclosure lesson — 64 censors
  reason-first rubrics), the c5 install record, the c3-vs-c5 TRAINED-ORGANISM
  contrast (descriptive-only), and the standing dual-DV validation
  rho(margin, judged rate) across the tier-2 question contexts. Deliverables
  land under ``eval_results/issue_1090/fu1-margin-qwen/``.

Pod-side reporting: ``[phase=...]`` lines per phase; the gpu phase writes the
``epm:results`` sentinel; ``[phase=done]`` is emitted ONLY by
``scripts/issue1090_fu1_dispatch.sh``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # .env + shared-VM thread caps BEFORE any torch-adjacent import

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# vLLM v1 fork-death prevention (gotchas.md #628): pin spawn BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue1074_generator_compare as i1074  # noqa: E402
import issue1090_run as i1090  # noqa: E402
import numpy as np  # noqa: E402
from issue1074_aggregate import paired_question_bootstrap  # noqa: E402
from scipy import stats as _sstats  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    POSITIVE,
    GenCandidate,
    _read_raw,
    _write_raw,
)
from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_MARGIN_POOL_CAP,
    _default_margin_read_fn,
)
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1090_fu1")

# ── Constants ─────────────────────────────────────────────────────────────────

FU1_LABEL = "fu1-margin-qwen"
# Judge response cap for every fu1 judge call (topup filter + tier-2 judging).
# The graded_judge default 64 truncates reason-first rubric responses BEFORE
# their JSON and parse-drops them (#1090: 473/1000 + 307/1000 dropped draws);
# 300 recovered 98.8% in the dropclosure refresh (free_analysis meta
# refresh_max_tokens=300). Tier-1 in-loop reads are NOT changed (dose-selection
# parity with the other cells).
# JUSTIFIED DEVIATION from llm-judging rule 23's 1024 floor (#2063): completed-run
# instrument constant; parity/metadata consumers: issue1090_fu2.py,
# issue1090_fu2_judge_fold.py, issue1112_dispatch.py, issue1315_rejudge_529.py.
# Fresh waves owe >=1024.
JUDGE_MAX_TOKENS_FU1 = 300
# The c5 record was never retuned to the amendment's fence-maximal 2.0 budget
# (recorded oversample_mult=1.0, kept 19/36 >= near-miss floor 18); the
# followup-scope marker registers the c5 tranche at its recorded budget. Every
# OTHER amendment fence (near-miss floor, EXACTLY-ONE tranche, frozen yield DV)
# stays binding via the reused _run_topup_cell body.
C5_ELIGIBLE_MULT = 1.0
# The c3 trained side (clean-result: dose-selected closest_approach step 14).
C3_CKPT_HF_PREFIX = "issue1090/c3-sycophancy-claude/checkpoint-14"
# epm:results sentinel version: the parent run posted v1 and v2 — max+1.
FU1_SENTINEL_VERSION = 3
# Adapter-application assert tolerance on |Δ LN logP| per fixed pair (base vs
# adapter-loaded). A no-op adapter load reproduces base bitwise (same weights,
# deterministic forward); checkpoint-14 moved the judged rate +0.22, so 1e-3
# is far below any real effect. Smoke uses strictly-greater-than-zero (the
# 4-step tiny adapter moves LN logP by a small but nonzero amount).
ADAPTER_ASSERT_TOL_FULL = 1e-3
ADAPTER_ASSERT_TOL_SMOKE = 0.0

C3 = i1090.CELL_BY_ID["c3"]
C5 = i1090.CELL_BY_ID["c5"]


# ── Small helpers ─────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    """JSONL via text-mode file iteration (NEVER splitlines — gotchas.md)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _judge_fu1(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
    """The fu1 judge seam: ``judge_graded`` at ``max_tokens=JUDGE_MAX_TOKENS_FU1``.

    Signature mirrors the datagen JudgeFn seam exactly (``_judge_and_filter``
    call shape). NOTE the max_tokens deviation vs the parent's tranche filter
    (which ran at the 64 default) is recorded in the c5 topup deliverable.
    """
    return judge_graded(
        items,
        eval_prompt,
        n_draws=n_draws,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=judge_model,
        max_tokens=JUDGE_MAX_TOKENS_FU1,
        dry_run=dry_run,
    )


def _stage_repo_prefix(
    repo_id: str,
    repo_type: str,
    prefix: str,
    dest: Path,
    *,
    skip_if: Callable[[Path], bool] | None = None,
) -> None:
    """Mirror one Hub prefix into ``dest`` (scoped ``list_repo_tree`` + per-file
    download — NEVER snapshot_download on the large repos; gotchas.md). The
    generalized (repo_id, repo_type) twin of ``issue1090_run._stage_hf_prefix``
    — needed because the c3 adapter lives on the MODEL repo.
    """
    if skip_if is not None and skip_if(dest):
        logger.info("[fu1-stage] %s already complete locally — skip", dest)
        return
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi()
    try:
        entries = hub.retry_transient(
            lambda: list(
                api.list_repo_tree(
                    repo_id, path_in_repo=prefix, repo_type=repo_type, recursive=True
                )
            ),
            what=f"fu1 stage listing {repo_id}/{prefix}",
        )
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"no tree at {repo_id}/{prefix}") from e
    files = [e.path for e in entries if not getattr(e, "tree_id", None)]
    if not files:
        raise FileNotFoundError(f"no files under {repo_id}/{prefix}")
    for hub_path in files:
        rel = hub_path[len(prefix) :].lstrip("/")
        target = dest / rel
        if target.exists():
            continue
        got = hub.retry_transient(
            lambda hp=hub_path: hf_hub_download(
                repo_id, hp, repo_type=repo_type, local_dir=dest / "_hfstage"
            ),
            what=f"fu1 stage download {hub_path}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, target)
    logger.info("[fu1-stage] %s/%s -> %s (%d files)", repo_id, prefix, dest, len(files))


# ── Fixed margin pools from the datagen_topup union (P2) ─────────────────────


def _kept_topup_pairs(topup_dir: Path, raw_name: str, kept_name: str) -> list[GenCandidate]:
    """Kept tranche candidates: ``raw_*.jsonl`` rows (question + completion)
    filtered by the ``kept_*.jsonl`` request-id set — the kept flags were
    recorded by the tranche's judge filter; NOTHING is re-judged here."""
    raw = _read_raw(topup_dir / raw_name)
    kept_rows = _read_jsonl(topup_dir / kept_name)
    kept_completion = {r["request_id"]: r["completion"] for r in kept_rows}
    by_rid = {c.request.request_id: c for c in raw}
    missing = sorted(set(kept_completion) - set(by_rid))
    if missing:
        raise ValueError(f"kept ids absent from {topup_dir / raw_name}: {missing[:5]}")
    out = []
    for rid, completion in kept_completion.items():
        c = by_rid[rid]
        if c.completion != completion:
            raise ValueError(
                f"kept/raw completion mismatch for {rid} under {topup_dir} — "
                "corrupt sidecars; refusing to build a pool from them"
            )
        out.append(c)
    return out


def _pool_pair(c: GenCandidate, source: str) -> dict:
    return {
        "probe": c.request.question,
        "answer": c.completion,
        "question_id": c.request.question_id,
        "variant_id": c.request.variant_id,
        "request_id": c.request.request_id,
        "source": source,
    }


def derive_margin_pools_topup(
    cell_root: Path,
    behavior,
    *,
    cap: int = DEFAULT_MARGIN_POOL_CAP,
    scratch: Path,
) -> tuple[list[dict], list[dict], dict]:
    """FIXED (probe, answer) pools for a ``success_with_topup`` cell.

    The in-run ``derive_margin_pools`` needs raw+judge sidecars for BOTH arms
    in ONE datagen dir; a topped-up near-miss cell raised at the positive floor
    BEFORE negatives existed, so its negatives (and part of its positives) live
    under ``datagen_topup/``. Union sources:

    - positives: first-sample kept rows (REPLAYED from the recorded
      ``judge_raw_pos.json`` through the production filter — zero API, no
      re-judging) + tranche kept rows (``kept_pos.jsonl`` id-join);
    - negatives: tranche kept rows ONLY (``kept_neg.jsonl``).

    Deterministic selection (the ``derive_margin_pools`` /
    ``build_fixed_pairs`` pattern): sort by (question_id, variant_id,
    request_id), first ``cap`` per side. Returns ``(pos, neg, meta)`` with the
    pool composition pinned (ids + sha) in ``meta``.
    """
    dg, td = cell_root / "datagen", cell_root / "datagen_topup"
    pos_first = i1090._replay_first_sample_kept(
        behavior, dg, POSITIVE, "raw_pos.jsonl", "judge_raw_pos.json", scratch
    )
    pos_topup = _kept_topup_pairs(td, "raw_pos.jsonl", "kept_pos.jsonl")
    neg_topup = _kept_topup_pairs(td, "raw_neg.jsonl", "kept_neg.jsonl")
    pos = [_pool_pair(c, "first_sample") for c in pos_first] + [
        _pool_pair(c, "topup") for c in pos_topup
    ]
    neg = [_pool_pair(c, "topup") for c in neg_topup]
    for arm, pool in (("positive", pos), ("negative", neg)):
        if not pool:
            raise ValueError(f"empty {arm} margin pool under {cell_root}")
        pool.sort(key=lambda p: (p["question_id"], p["variant_id"], p["request_id"]))
    meta = {
        "n_pos_available": len(pos),
        "n_neg_available": len(neg),
        "cap": cap,
        "n_pos_used": min(len(pos), cap),
        "n_neg_used": min(len(neg), cap),
        "pos_request_ids": [p["request_id"] for p in pos[:cap]],
        "neg_request_ids": [p["request_id"] for p in neg[:cap]],
        "pos_sources": {
            "first_sample": sum(1 for p in pos[:cap] if p["source"] == "first_sample"),
            "topup": sum(1 for p in pos[:cap] if p["source"] == "topup"),
        },
        "selection": "sorted by (question_id, variant_id, request_id), first cap per side",
    }
    pos, neg = pos[:cap], neg[:cap]
    meta["pool_sha256"] = _sha256_json(
        [
            {k: p[k] for k in ("probe", "answer", "question_id", "variant_id", "request_id")}
            for p in pos + neg
        ]
    )
    return pos, neg, meta


# ── Margin contexts (per tier-2 question + the in-run source-ctx shape) ──────


@dataclass(frozen=True)
class _MsgCtx:
    """Duck-typed stand-in for ``artifacts.context.Context`` at the margin
    seam (``read`` uses ONLY ``.messages``): ``messages(probe)`` -> chat list."""

    context_id: str
    messages: Callable[[str], list[dict]]


def margin_contexts(cfg: i1090.RunConfig) -> tuple[list[str], list[tuple[str, Any]]]:
    """(questions, [(ctx_label, ctx), ...]).

    ``source_ctx`` FIRST (the in-run ``phase_margin`` construction: context =
    persona system, probe = the pair's OWN question — the #722 shape; the
    trained-sweep adapter assert runs on it), then one context per tier-2 eval
    question: the fixed pool ANSWERS scored as the assistant reply to
    (persona system + question q_i) — the pair probe is deliberately ignored so
    the IDENTICAL fixed answer set is scored under every context (rule 19).
    """
    src = i1090._source_context()
    questions = i1090._eval_questions(cfg, "sycophancy")
    ctxs: list[tuple[str, Any]] = [("source_ctx", src)]
    for i, q in enumerate(questions):
        ctxs.append(
            (
                f"q{i:03d}",
                _MsgCtx(f"{src.context_id}__q{i:03d}", lambda probe, _q=q: src.messages(_q)),
            )
        )
    return questions, ctxs


def _q_labels(n_questions: int) -> list[str]:
    return [f"q{i:03d}" for i in range(n_questions)]


def assert_adapter_applied(base_read: dict, trained_read: dict, *, tol: float, tag: str) -> dict:
    """Fail loud when the adapter-side teacher-forced read is numerically
    indistinguishable from base — the adapter-not-applied class (#534/#492).
    Returns the assert record on success; RuntimeError on failure."""
    diffs = [
        abs(t - b)
        for t, b in zip(trained_read["pos_ln_logp"], base_read["pos_ln_logp"], strict=True)
    ]
    if not any(d > tol for d in diffs):
        raise RuntimeError(
            f"[{tag}] adapter-application assert FAILED: max |Δ LN logP| over "
            f"{len(diffs)} fixed pos pairs = {max(diffs):.3e} <= tol {tol} — the "
            "trained side reproduces base (adapter likely not applied); refusing the sweep"
        )
    return {"max_abs_delta_pos_ln_logp": max(diffs), "tol": tol, "n_pairs": len(diffs)}


def aggregate_margin_reads(reads: dict[str, dict], q_labels: list[str]) -> dict:
    """Headline margin fields from the per-(side, context) reads: per-context
    margins per side, their means (the headline margin_base / margin_trained),
    delta, and the source-ctx companion. Pure (tested on toy numbers)."""

    def _per_ctx(side: str) -> dict[str, float]:
        return {
            lbl: reads[f"{side}__{lbl}"]["margin"] for lbl in q_labels if f"{side}__{lbl}" in reads
        }

    out: dict[str, Any] = {"per_context_margin": {}}
    means: dict[str, float | None] = {}
    for side in ("base", "trained"):
        per_ctx = _per_ctx(side)
        out["per_context_margin"][side] = per_ctx
        means[side] = (sum(per_ctx.values()) / len(per_ctx)) if per_ctx else None
    out["margin_base"] = means["base"]
    out["margin_trained"] = means["trained"]
    out["margin_delta"] = (
        means["trained"] - means["base"]
        if means["trained"] is not None and means["base"] is not None
        else None
    )
    out["source_ctx"] = {
        side: reads.get(f"{side}__source_ctx", {}).get("margin") for side in ("base", "trained")
    }
    if all(v is not None for v in out["source_ctx"].values()):
        out["source_ctx"]["delta"] = out["source_ctx"]["trained"] - out["source_ctx"]["base"]
    return out


# ── Phase: fu1 margin sweep (GPU tail; HF/PEFT — after every vLLM teardown) ──


class _MarginFnHolder:
    """Lazy margin-read fn (the model loads only when a read is actually
    missing from the resume record); ``close()`` idempotent."""

    def __init__(self, cfg: i1090.RunConfig, seams: i1090.Seams1090) -> None:
        self._seams = seams
        self._fn = None

    def get(self):
        if self._fn is None:
            self._fn = (
                self._seams.margin_read_fn_factory(DEFAULT_BASE_MODEL)
                if self._seams.margin_read_fn_factory is not None
                else _default_margin_read_fn(DEFAULT_BASE_MODEL)
            )
        return self._fn

    def close(self) -> None:
        if self._fn is not None:
            close = getattr(self._fn, "close", None)
            if callable(close):
                close()
            self._fn = None


def _sweep_cell_side(
    cfg: i1090.RunConfig,
    holder: _MarginFnHolder,
    rec: dict,
    pools: tuple[list[dict], list[dict]],
    out_path: Path,
    ctxs: Sequence[tuple[str, Any]],
    *,
    side_label: str,
    side_path: str | None,
    tag: str,
) -> None:
    """One (cell, side) margin sweep over every context: skip completed reads
    (resume), checkpoint per read, and run the adapter-application assert on
    the trained side's FIRST context (source_ctx) BEFORE the rest of the sweep."""
    pos, neg = pools
    for ctx_label, ctx in ctxs:
        key = f"{side_label}__{ctx_label}"
        if key in rec["reads"]:
            continue
        mr = holder.get()(side_path, ctx, pos, neg)
        rec["reads"][key] = dataclasses.asdict(mr)
        if side_label == "trained" and ctx_label == "source_ctx":
            rec["adapter_assert"] = assert_adapter_applied(
                rec["reads"]["base__source_ctx"],
                rec["reads"][key],
                tol=ADAPTER_ASSERT_TOL_SMOKE if cfg.smoke else ADAPTER_ASSERT_TOL_FULL,
                tag=tag,
            )
        i1090._atomic_write_json(out_path, rec)  # checkpoint per read


def phase_fu1_margin(
    cfg: i1090.RunConfig,
    seams: i1090.Seams1090,
    sides: Sequence[tuple[Any, str | None]],
) -> dict[str, dict]:
    """Per-cell fixed-pool margins under base + trained across the tier-2
    question contexts (+ source_ctx). ``sides``: (cell, trained_side_path);
    a ``None`` side records a base-only read (no adapter — e.g. a c5 union
    miss). Checkpoints per read; resumes keyed on pool sha + questions sha."""
    i1090._phase("fu1_margin")
    questions, ctxs = margin_contexts(cfg)
    questions_sha = _sha256_json(questions)
    out_dir = cfg.out_root / "fu1_margin"
    records: dict[str, dict] = {}
    pools: dict[str, tuple[list[dict], list[dict]]] = {}

    for cell, side in sides:
        cell_root = cfg.out_root / cell.slug
        pos, neg, meta = derive_margin_pools_topup(
            cell_root, BEHAVIORS[cell.behavior], scratch=out_dir / f"_replay_{cell.slug}"
        )
        pools[cell.slug] = (pos, neg)
        out_path = out_dir / f"{cell.slug}.json"
        if out_path.exists():
            rec = i1090._read_json(out_path)
            if rec.get("pool", {}).get("pool_sha256") != meta["pool_sha256"] or (
                rec.get("questions_sha256") != questions_sha
            ):
                raise RuntimeError(
                    f"{out_path} holds margins under a DIFFERENT pool/questions regime — "
                    "move it aside; refusing a silent mixed-regime resume"
                )
        else:
            rec = {
                "status": "in_progress",
                "cell": cell.slug,
                "behavior": cell.behavior,
                "pool": meta,
                "questions_sha256": questions_sha,
                "n_question_contexts": len(questions),
                "trained_side": str(side) if side is not None else None,
                "judge_free": True,  # teacher-forced only; no judge calls here
                "git_commit": i1074._git_short_sha(),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reads": {},
            }
            i1090._atomic_write_json(out_path, rec)
        records[cell.slug] = rec

    # Base side for every cell first, then each trained side — 1 + n_cells
    # model loads under the single-live holder (never interleaved).
    plan: list[tuple[str, str | None, Any]] = [("base", None, None)]
    for cell, side in sides:
        if side is not None:
            plan.append(("trained", str(side), cell))
    holder = _MarginFnHolder(cfg, seams)
    try:
        for side_label, side_path, only_cell in plan:
            for cell, _side in sides:
                if only_cell is not None and cell.slug != only_cell.slug:
                    continue
                _sweep_cell_side(
                    cfg,
                    holder,
                    records[cell.slug],
                    pools[cell.slug],
                    out_dir / f"{cell.slug}.json",
                    ctxs,
                    side_label=side_label,
                    side_path=side_path,
                    tag=cell.slug,
                )
    finally:
        holder.close()

    for cell, side in sides:
        rec = records[cell.slug]
        rec.update(aggregate_margin_reads(rec["reads"], _q_labels(len(questions))))
        rec["status"] = "computed" if side is not None else "base_only_no_adapter"
        i1090._atomic_write_json(out_dir / f"{cell.slug}.json", rec)
    return records


# ── Phase: c5 top-up + train + tier-2 generation (P3) ────────────────────────


def run_c5_pipeline(cfg: i1090.RunConfig, seams: i1090.Seams1090) -> tuple[dict, dict]:
    """The c5 qwen-arm pipeline: ONE amendment-v4 tranche at the recorded 1.0
    budget (frozen yield DV untouched; union mix TRAINING-ONLY) -> train ->
    Tier-1 ladder -> Tier-2 generation. Returns (datagen_record, train_results).
    A union miss is a DECIDABLE outcome (record + skip train), never a crash."""
    cell = C5
    cfg5 = dataclasses.replace(cfg, cells=(cell,))
    cell_root = cfg.out_root / cell.slug
    summary_path = cell_root / "datagen_summary.json"
    rec = i1090._read_json(summary_path)

    if rec.get("status") == i1090.TOPUP_STATUS and i1090._topup_mix_complete(
        i1090._topup_dir(cell_root)
    ):
        logger.info("[fu1-c5] top-up already recorded + mix complete — skip")
    elif "topup_record" in rec:
        logger.warning(
            "[fu1-c5] a prior tranche is recorded and the union did NOT clear "
            "(status=%s) — the amendment allows EXACTLY ONE; reporting the miss",
            rec.get("status"),
        )
    else:
        i1090._phase("fu1_c5_topup")
        factory = seams.qwen_datagen_gen_factory or (
            lambda model_id, *, max_new_tokens: i1074.make_vllm_generate_fn(
                model_id,
                temperature=cfg.gen_temperature,
                max_new_tokens=max_new_tokens,
                seed=cfg.seed,
            )
        )
        gen_fn = factory(i1090.QWEN_GEN_MODEL, max_new_tokens=i1090.GEN_MAX_NEW_TOKENS)
        try:
            rec = i1090._run_topup_cell(
                cfg5, cell, gen_fn=gen_fn, judge_fn=_judge_fu1, eligible_mult=C5_ELIGIBLE_MULT
            )
        finally:
            close = getattr(gen_fn, "close", None)
            if callable(close):
                close()  # GPU handoff BEFORE the trainer loads (the #1074 pattern)

    if rec.get("status") != i1090.TOPUP_STATUS:
        return rec, {cell.slug: {"status": "skipped_union_missed_floor"}}

    # Train + Tier-1 dose ladder + Tier-2 generation — the run's own phase
    # bodies, verbatim (save_steps=2 / max_length=2048 / band + closest_approach
    # live inside phase_train -> build_organism). NOTE: phase_datagen_qwen is
    # deliberately NOT re-run — datagen for c5 is complete (success_with_topup),
    # and _run_datagen_cell has no resume branch for that status.
    train_results = i1090.phase_train(cfg5, seams, {cell.slug: rec})
    i1090.phase_tier2_generation(cfg5, seams, train_results)
    return rec, train_results


# ── Staging (full) / fixtures (smoke) ─────────────────────────────────────────


def _c3_ckpt_dest(cfg: i1090.RunConfig) -> Path:
    return cfg.out_root / "inputs" / "c3_checkpoint14"


def stage_full_inputs(cfg: i1090.RunConfig) -> None:
    """Fresh-lane staging (git-clone lanes carry no local data/): the c3 pool
    sidecars + adapter, the c5 first-sample sidecars + summary, generic corpus."""
    for cell in (C3, C5):
        cell_root = cfg.out_root / cell.slug
        d = cell_root / "datagen"
        if not (i1090._datagen_complete(d) or i1090._datagen_recorded(d)):
            i1090._stage_hf_prefix(f"{i1090.DATA_PREFIX}/{cell.slug}/datagen", d)
        if not (cell_root / "datagen_summary.json").exists():
            i1090._stage_hf_prefix(f"{i1090.DATA_PREFIX}/{cell.slug}", cell_root)
    td = i1090._topup_dir(cfg.out_root / C3.slug)
    if not (td / "kept_neg.jsonl").exists():
        i1090._stage_hf_prefix(f"{i1090.DATA_PREFIX}/{C3.slug}/datagen_topup", td)
    _stage_repo_prefix(
        i1090.HF_MODEL_REPO,
        "model",
        C3_CKPT_HF_PREFIX,
        _c3_ckpt_dest(cfg),
        skip_if=lambda d: (
            (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists()
        ),
    )
    if cfg.generic_data_path is None:
        cfg.generic_data_path = i1074._stage_generic_corpus(
            cfg.out_root / "inputs" / "generic_corpus.jsonl"
        )


def _build_smoke_first_sample(cfg: i1090.RunConfig, cell, *, n_raw: int, n_keep: int, mult: float):
    """Tiny-real first-sample fixture THROUGH the production compose/replay
    writers (requests via _compose_positive_requests, raw via _write_raw, kept
    flags via the judge save_raw shape the replay path consumes)."""
    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    dgdir = cfg.out_root / cell.slug / "datagen"
    if (dgdir / "raw_pos.jsonl").exists():
        return
    dgdir.mkdir(parents=True, exist_ok=True)
    manifest = i1090._reconstruct_manifest(cfg, cell, behavior, panel, mult)
    (dgdir / "gen_manifest.json").write_text(json.dumps(manifest) + "\n")
    exhibit, _ne = i1090._resolve_instructions(behavior, "extraction_pairs")
    tq = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    reqs = i1090._compose_positive_requests(
        behavior,
        i1090._source_context(),
        tq,
        n_raw,
        i1090._rng(cfg.seed),
        "plain",
        variants=exhibit,
    )
    cands = [
        GenCandidate(r, i1090._smoke_completion_1090(cell.behavior, exhibit=True)) for r in reqs
    ]

    _write_raw(dgdir / "raw_pos.jsonl", cands)
    all_scores: dict[str, Any] = {}
    for i, c in enumerate(cands):
        for d in range(cfg.n_judge_draws):
            all_scores[f"{c.request.request_id}__{i:05d}__{d:02d}"] = 80.0 if i < n_keep else 20.0
    (dgdir / "judge_raw_pos.json").write_text(json.dumps({"all_scores": all_scores}))
    floor_n = math.ceil(cfg.quota_floor * cfg.target_n)
    summary = {
        "cell": cell.slug,
        "cell_id": cell.cell_id,
        "behavior": cell.behavior,
        "generator": cell.generator,
        "gen_model": cell.gen_model,
        "trains": cell.trains,
        "status": "yield_floor_missed",
        "oversample_mult": mult,
        "target_n": cfg.target_n,
        "quota_floor": cfg.quota_floor,
        "floor_n": floor_n,
        "seed": cfg.seed,
        "positive_stage": {"n_kept": n_keep, "n_requested": n_raw},
        "yield_record": {
            "kept_pos": n_keep,
            "floor_n": floor_n,
            "message": f"kept {n_keep} positives < floor_n={floor_n} (smoke fixture)",
            "stages": {"positive": {"requested": n_raw}},
        },
        "per_question_yield": {},
    }
    i1090._atomic_write_json(cfg.out_root / cell.slug / "datagen_summary.json", summary)


def _build_smoke_c3_topup(cfg: i1090.RunConfig) -> None:
    """The c3 fixture's datagen_topup sidecars via the PRODUCTION topup writers
    (_topup_ids / _write_raw_topup / _write_kept_topup) — pool inputs only; the
    c3 topup itself is not re-run (its production record is the parent's)."""
    behavior = BEHAVIORS[C3.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    td = i1090._topup_dir(cfg.out_root / C3.slug)
    if (td / "kept_neg.jsonl").exists():
        return
    td.mkdir(parents=True, exist_ok=True)
    exhibit, not_exhibit = i1090._resolve_instructions(behavior, "extraction_pairs")
    tq = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    pos_reqs = i1090._topup_ids(
        i1090._compose_positive_requests(
            behavior,
            i1090._source_context(),
            tq,
            4,
            i1090._rng(cfg.seed + i1090.TOPUP_SEED_OFFSET),
            "plain",
            variants=exhibit,
        )
    )
    pos_cands = [
        GenCandidate(r, i1090._smoke_completion_1090(C3.behavior, exhibit=True)) for r in pos_reqs
    ]
    i1090._write_raw_topup(td / "raw_pos.jsonl", pos_cands)
    i1090._write_kept_topup(td / "kept_pos.jsonl", pos_cands[:2])
    neg_reqs = i1090._topup_ids(
        i1090._compose_negative_requests(
            behavior,
            panel,
            i1090._dedup_questions(pos_cands[:2]),
            1,
            i1090._rng(cfg.seed + i1090.TOPUP_SEED_OFFSET + 1),
            "plain",
            not_exhibit=not_exhibit,
        )
    )
    neg_cands = [
        GenCandidate(r, i1090._smoke_completion_1090(C3.behavior, exhibit=False)) for r in neg_reqs
    ]
    i1090._write_raw_topup(td / "raw_neg.jsonl", neg_cands)
    i1090._write_kept_topup(td / "kept_neg.jsonl", neg_cands)


def build_smoke_fixtures(cfg: i1090.RunConfig) -> None:
    """Smoke inputs: c3 near-miss + topup sidecars (pool sources) and a c5
    near-miss first sample at the RECORDED 1.0 budget (the live smoke topup
    then runs the production tranche body on it)."""
    _build_smoke_first_sample(cfg, C3, n_raw=8, n_keep=4, mult=2.0)
    _build_smoke_c3_topup(cfg)
    _build_smoke_first_sample(cfg, C5, n_raw=8, n_keep=4, mult=C5_ELIGIBLE_MULT)


# ── Upload + sentinel (gpu-phase tail) ────────────────────────────────────────


def phase_fu1_upload(
    cfg: i1090.RunConfig, seams: i1090.Seams1090, train_results: dict
) -> dict[str, str]:
    """Everything to HF before pod release: the c5 cell tree via the run's own
    upload phases (adapter ladder -> model repo issue1090/c5-sycophancy-qwen;
    mixes / rate / tier2 raw completions -> the parent prefixes every
    downstream consumer reads) + the fu1 margin records under the fu1 prefix
    (text/JSON — unconditional non-LFS path)."""
    cfg5 = dataclasses.replace(cfg, cells=(C5,))
    uploaded: dict[str, str] = {}
    uploaded.update(i1090.upload_topup_dirs(cfg5, seams, [C5]))
    uploaded.update(i1090.phase_upload(cfg5, seams, train_results))
    upload = i1090._upload_fn(seams)
    margin_dir = cfg.out_root / "fu1_margin"
    if margin_dir.exists():
        url = upload(
            margin_dir,
            i1090.HF_DATA_REPO,
            "dataset",
            f"{i1090.DATA_PREFIX}/{FU1_LABEL}/margin",
            ignore_patterns=["_replay_*"],
        )
        uploaded[f"{i1090.DATA_PREFIX}/{FU1_LABEL}/margin"] = str(url)
    i1090._atomic_write_json(cfg.out_root / "fu1_upload_manifest.json", uploaded)
    return uploaded


def write_fu1_sentinel(
    cfg: i1090.RunConfig, c5_rec: dict, train_results: dict, margins: dict, uploaded: dict
) -> Path:
    """End-of-gpu-phase epm:results sentinel (poll_pipeline envelope keys)."""
    i1090._phase("fu1_sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    tr = train_results.get(C5.slug, {})
    note = {
        "issue": i1090.ISSUE,
        "followup_label": FU1_LABEL,
        "smoke": cfg.smoke,
        "c5_datagen_status": c5_rec.get("status"),
        "c5_union_cleared": (c5_rec.get("topup_record") or {}).get("union_cleared"),
        "c5_train_status": tr.get("status"),
        "c5_dose_selection": tr.get("selection"),
        "margins": {
            slug: {
                k: rec.get(k) for k in ("status", "margin_base", "margin_trained", "margin_delta")
            }
            for slug, rec in margins.items()
        },
        "uploaded_prefixes": sorted(uploaded),
        "hf_data_prefix": f"{i1090.DATA_PREFIX}/{FU1_LABEL}",
        "reproducibility_card": {
            "hf_model_repo": i1090.HF_MODEL_REPO,
            "c5_adapter_path": (
                f"{i1090.MODEL_PREFIX}/{C5.slug}/{Path(tr['adapter_path']).name}"
                if tr.get("status") == "trained"
                else None
            ),
            "c3_trained_side": C3_CKPT_HF_PREFIX,
            "wandb_project": os.environ.get("WANDB_PROJECT", "issue1090"),
            "judge_max_tokens": JUDGE_MAX_TOKENS_FU1,
            "c5_eligible_mult": C5_ELIGIBLE_MULT,
        },
        "git_commit": i1074._git_short_sha(),
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": FU1_SENTINEL_VERSION,
        "task_id": i1090.ISSUE,
        "by": "issue1090_fu1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    path = sentinel_dir / f"issue-{i1090.ISSUE}-epm_results-{int(time.time())}.json"
    i1090._atomic_write_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


def _ensure_c3_tier2(cfg: i1090.RunConfig, seams: i1090.Seams1090, c3_side: str | None) -> None:
    """The judge phase compares BOTH organisms under the fresh-300 instrument,
    so the c3 tier-2 completions must exist under out_root: resume on the
    files, else stage the PARENT's canonical completions from HF (full), else
    generate at the c3 trained side via the production tier-2 body (smoke —
    the parent record is a full-run-only input)."""
    src_id = i1090.SOURCE_CONTEXT_ID
    tier2_dir = cfg.out_root / "tier2" / C3.slug
    if all((tier2_dir / f"completions__{s}__{src_id}.json").exists() for s in ("trained", "base")):
        return
    if not cfg.smoke:
        try:
            i1090._stage_hf_prefix(
                f"{i1090.DATA_PREFIX}/raw_completions/tier2/{C3.slug}", tier2_dir
            )
        except FileNotFoundError:
            logger.warning(
                "[fu1] no c3 tier2 completions on HF — regenerating at the staged checkpoint"
            )
        else:
            # r2 hardening (code-review): a present-but-PARTIAL HF prefix (one
            # state file) must fall through to generation, not return with a
            # half-staged dir the judge phase would crash on.
            if all(
                (tier2_dir / f"completions__{s}__{src_id}.json").exists()
                for s in ("trained", "base")
            ):
                return
            logger.warning("[fu1] staged c3 tier2 prefix is missing state files — regenerating")
    if c3_side is None:
        logger.warning("[fu1] no c3 trained side available — skipping c3 tier2 generation")
        return
    i1090.phase_tier2_generation(
        dataclasses.replace(cfg, cells=(C3,)),
        seams,
        {C3.slug: {"status": "trained", "adapter_path": str(c3_side)}},
    )


def phase_fu1_gpu(cfg: i1090.RunConfig, seams: i1090.Seams1090) -> dict:
    """The GPU-lane pipeline: stage/fixtures -> c5 topup+train+tier2 (vLLM +
    trainer phases, torn down) -> c3+c5 margin sweep (HF/PEFT) -> upload ->
    sentinel."""
    i1090._phase("fu1_stage_inputs")
    if cfg.smoke:
        build_smoke_fixtures(cfg)
    else:
        stage_full_inputs(cfg)
    c5_rec, train_results = run_c5_pipeline(cfg, seams)
    c5_trained = train_results.get(C5.slug, {}).get("status") == "trained"
    c5_side = train_results.get(C5.slug, {}).get("adapter_path") if c5_trained else None
    # Smoke exercises the identical PEFT-load/assert/sweep path; the c3
    # PRODUCTION side (checkpoint-14 on the model repo) is a full-run-only
    # input, so the smoke threads the just-trained tiny adapter for c3 too.
    c3_side = c5_side if cfg.smoke else str(_c3_ckpt_dest(cfg))
    _ensure_c3_tier2(cfg, seams, c3_side)  # BEFORE the HF margin phase (vLLM in full)
    sides: list[tuple[Any, str | None]] = [(C3, c3_side)]
    if c5_trained:
        sides.append((C5, c5_side))
    else:
        # r2 fix (code-review Major): a positive-floor union miss returns from
        # _run_topup_cell BEFORE the negative stage, so datagen_topup/
        # {raw_neg,kept_neg}.jsonl never exist and the c5 pools are
        # underivable — omit c5 from the margin sweep entirely (C3-only). The
        # miss is a DECIDABLE recorded outcome: the upload + sentinel below
        # MUST still run (they carry the top-up record + the miss report).
        logger.warning(
            "[fu1] c5 not trained (datagen status=%s) — margin sweep runs C3-only; "
            "still uploading the top-up record + writing the miss sentinel",
            c5_rec.get("status"),
        )
    margins = phase_fu1_margin(cfg, seams, sides)
    uploaded = phase_fu1_upload(cfg, seams, train_results) if cfg.upload else {}
    sentinel = write_fu1_sentinel(cfg, c5_rec, train_results, margins, uploaded)
    return {
        "c5": {
            "datagen": c5_rec.get("status"),
            "train": train_results.get(C5.slug, {}).get("status"),
        },
        "margins": {k: v.get("status") for k, v in margins.items()},
        "n_uploaded": len(uploaded),
        "sentinel": str(sentinel),
    }


# ── Phase: judge (VM) — tier-2 judging @300, install, contrast, rho ──────────


def reduce_judge_fu1(
    indexed_items: list[tuple[int, str]],
    scores: dict[str, float | None],
    *,
    threshold: float,
    n_questions: int,
) -> dict:
    """Aggregate + per-question judged rates from per-item mean scores.

    ``indexed_items``: (question_index, item_id) per completion; a ``None`` /
    missing score is DROPPED (never coerced — llm-judging rule 9). Pure."""
    per_q_k = [0] * n_questions
    per_q_n = [0] * n_questions
    per_item: list[dict] = []
    n_dropped = 0
    for qi, iid in indexed_items:
        s = scores.get(iid)
        per_item.append({"item_id": iid, "q_index": qi, "score": s})
        if s is None:
            n_dropped += 1
            continue
        per_q_n[qi] += 1
        per_q_k[qi] += int(s > threshold)
    k, n = sum(per_q_k), sum(per_q_n)
    if n == 0:
        raise ValueError("every completion was judge-dropped — a judging outage")
    lo, hi = i1090._wilson(k, n)
    return {
        "rate": k / n,
        "k": k,
        "n": n,
        "n_dropped": n_dropped,
        "wilson95": [lo, hi],
        "per_question_rate": [
            (per_q_k[i] / per_q_n[i]) if per_q_n[i] else None for i in range(n_questions)
        ],
        "per_question_n": per_q_n,
        "per_item": per_item,
        "threshold": threshold,
        "mode": "judged",
    }


def judge_tier2_fu1(
    cfg: i1090.RunConfig, cell, state: str, questions: list[str], judge_root: Path
) -> dict:
    """One (cell, state) tier-2 judged read at max_tokens=300 (fresh — the
    symmetric instrument both organisms are compared under)."""
    src_id = i1090.SOURCE_CONTEXT_ID
    comp_path = cfg.out_root / "tier2" / cell.slug / f"completions__{state}__{src_id}.json"
    payload = i1090._read_json(comp_path)
    if payload["questions"] != questions:
        raise RuntimeError(
            f"{comp_path}: stored questions differ from the eval bank order — "
            "per-question joins would misalign; refusing"
        )
    completions = payload["completions"]
    behavior = BEHAVIORS[cell.behavior]
    tag = f"fu1-{cell.cell_id}-{state}"
    flat = [
        (f"{tag}-q{i:03d}-c{j}", q, comp)
        for i, q in enumerate(questions)
        for j, comp in enumerate(completions[i])
    ]
    indexed = [
        (i, f"{tag}-q{i:03d}-c{j}")
        for i, q in enumerate(questions)
        for j in range(len(completions[i]))
    ]
    cell_dir = judge_root / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = judge_graded(
        flat,
        behavior.judge_rubric,
        n_draws=cfg.tier2_draws,
        cache_dir=cell_dir,
        save_raw=cell_dir / "judge_raw.json",
        judge_model=behavior.judge_model,
        max_tokens=JUDGE_MAX_TOKENS_FU1,
    )
    out = reduce_judge_fu1(
        indexed, result.scores, threshold=behavior.threshold, n_questions=len(questions)
    )
    out.update(
        {
            "n_total_draws": result.n_total_draws,
            "n_dropped_draws": result.n_dropped_draws,
            "judge_max_tokens": JUDGE_MAX_TOKENS_FU1,
            "n_judge_draws": cfg.tier2_draws,
        }
    )
    return out


def closure_per_question_rates(p4_state_path: Path, *, threshold: float, n_questions: int) -> list:
    """Per-question closure-adjusted rates from the committed dropclosure
    per-item records (item ids ``c3-<state>-q{i:03d}-c{j}``, ``closure_mean``
    0-100; a missing/None mean is a dropped item — never coerced)."""
    d = i1090._read_json(p4_state_path)
    per_q_k = [0] * n_questions
    per_q_n = [0] * n_questions
    for row in d["per_item"]:
        m = re.search(r"-q(\d{3})-c\d+$", row["item_id"])
        if m is None:
            raise ValueError(f"unparseable per_item id {row['item_id']!r} in {p4_state_path}")
        qi = int(m.group(1))
        if qi >= n_questions:
            continue  # smoke question-limit slice of the full 20-question record
        s = row.get("closure_mean")
        if s is None:
            continue
        per_q_n[qi] += 1
        per_q_k[qi] += int(s > threshold)
    return [(per_q_k[i] / per_q_n[i]) if per_q_n[i] else None for i in range(n_questions)]


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Spearman rho with degenerate-input guards (n<3 / zero variance -> None)."""
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3 or len({p[0] for p in pairs}) < 2 or len({p[1] for p in pairs}) < 2:
        return {"rho": None, "p": None, "n": len(pairs), "note": "degenerate (n<3 or no variance)"}
    xs2, ys2 = zip(*pairs, strict=True)
    rho, p = _sstats.spearmanr(xs2, ys2)
    return {"rho": float(rho), "p": float(p), "n": len(pairs)}


_ASYMMETRY_NOTE = (
    "Descriptive-only contrast. Asymmetries stated, not resolved: (1) TRAINING-DATA "
    "provenance — the c3 organism was trained on Claude-generated positives, the c5 "
    "organism on qwen-generated positives; the EVAL completions on both sides are "
    "qwen-model outputs (both organisms are Qwen-2.5-7B-Instruct + LoRA), so the "
    "same-family-judge caveat does not apply to either eval side; (2) datagen judge "
    "filters — c3's tranche filter ran at the historical judge max_tokens=64, c5's at "
    "300 (the dropclosure lesson), so kept-pool censoring may differ; (3) the closure "
    "companion read for c3 unions 64-token original draws with 300-token refreshed "
    "draws, while the fresh-300 primary read judges both organisms under the identical "
    "instrument."
)


def _stage_judge_inputs(cfg: i1090.RunConfig) -> None:
    """Stage GPU-phase outputs from HF when missing locally (fresh-VM path).

    c5-specific inputs stage TOLERANTLY (r2 fix): on a union miss the gpu
    phase never produced c5 tier-2 completions / a build record, so their HF
    prefixes legitimately do not exist — `_c5_trained_available` decides the
    skip downstream. The c3 inputs stay fail-loud (always produced)."""
    import contextlib

    src_id = i1090.SOURCE_CONTEXT_ID
    for cell in (C3, C5):
        tier2_dir = cfg.out_root / "tier2" / cell.slug
        if not all(
            (tier2_dir / f"completions__{s}__{src_id}.json").exists() for s in ("trained", "base")
        ):
            if cell is C5:
                with contextlib.suppress(FileNotFoundError):
                    i1090._stage_hf_prefix(
                        f"{i1090.DATA_PREFIX}/raw_completions/tier2/{cell.slug}", tier2_dir
                    )
            else:
                i1090._stage_hf_prefix(
                    f"{i1090.DATA_PREFIX}/raw_completions/tier2/{cell.slug}", tier2_dir
                )
    if not (cfg.out_root / C5.slug / "build_result.json").exists():
        with contextlib.suppress(FileNotFoundError):
            i1090._stage_hf_prefix(f"{i1090.DATA_PREFIX}/{C5.slug}", cfg.out_root / C5.slug)
    margin_dir = cfg.out_root / "fu1_margin"
    if not (margin_dir / f"{C3.slug}.json").exists():
        i1090._stage_hf_prefix(f"{i1090.DATA_PREFIX}/{FU1_LABEL}/margin", margin_dir)


def _c5_trained_available(cfg: i1090.RunConfig) -> bool:
    """True when the c5 organism's judge inputs exist under out_root (the
    adapter build record + BOTH tier-2 completion files); False on a union
    miss — the judge phase then records the skip instead of crashing."""
    src_id = i1090.SOURCE_CONTEXT_ID
    tier2_dir = cfg.out_root / "tier2" / C5.slug
    return (cfg.out_root / C5.slug / "build_result.json").exists() and all(
        (tier2_dir / f"completions__{s}__{src_id}.json").exists() for s in ("trained", "base")
    )


def _paired_rates(r3: list, r5: list) -> dict:
    """Paired per-question rate deltas (c3 - c5) + bootstrap + sign test; a
    question with a None rate on either side is excluded (drop-never-coerce)."""
    deltas, signs = [], {"pos": 0, "neg": 0, "zero": 0}
    n_excluded = 0
    for a, b in zip(r3, r5, strict=True):
        if a is None or b is None:
            n_excluded += 1
            continue
        d = a - b
        deltas.append(d)
        signs["pos" if d > 0 else ("neg" if d < 0 else "zero")] += 1
    return {
        "n_paired_questions": len(deltas),
        "n_excluded": n_excluded,
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "paired_bootstrap": (
            paired_question_bootstrap(np.array(deltas), n_draws=2000, seed=42) if deltas else None
        ),
        "per_question_signs": signs,
        "sign_test_two_sided_p": i1090._binom_two_sided_p(
            signs["pos"], signs["pos"] + signs["neg"]
        ),
    }


def _c5_install_record(
    cfg: i1090.RunConfig, reads: dict, margins: dict, c5_available: bool
) -> dict:
    """The c5 install deliverable — a skip record on a union miss, never a crash."""
    if not c5_available:
        c5_summary_path = cfg.out_root / C5.slug / "datagen_summary.json"
        return {
            "cell": C5.slug,
            "status": "skipped_c5_union_missed",
            "c5_datagen_status": (
                i1090._read_json(c5_summary_path).get("status")
                if c5_summary_path.exists()
                else None
            ),
            "install_delta": None,
            "note": (
                "the c5 top-up union did not clear the positive floor — no trained "
                "organism this round; the frozen first-sample yield record is the c5 "
                "deliverable"
            ),
        }
    build = i1090._read_json(cfg.out_root / C5.slug / "build_result.json")
    return {
        "cell": C5.slug,
        "status": "computed",
        "behavior": C5.behavior,
        "generator": C5.generator,
        "selection": build.get("selection"),
        "adapter_path": build.get("adapter_path"),
        "band": list(i1090.JUDGED_RATE_BAND),
        "tier2": {"n_completions": cfg.tier2_n, "n_judge_draws": cfg.tier2_draws},
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU1,
        "reads": {
            s: {
                k: reads[f"{C5.slug}__{s}"][k]
                for k in (
                    "rate",
                    "k",
                    "n",
                    "wilson95",
                    "n_dropped",
                    "n_dropped_draws",
                    "n_total_draws",
                )
            }
            for s in ("trained", "base")
        },
        "install_delta": reads[f"{C5.slug}__trained"]["rate"] - reads[f"{C5.slug}__base"]["rate"],
        "margin_trained": margins.get(C5.slug, {}).get("margin_trained"),
        "margin_base": margins.get(C5.slug, {}).get("margin_base"),
        "margin_delta": margins.get(C5.slug, {}).get("margin_delta"),
    }


def _contrast_record(reads: dict, closure_rates: dict, install: dict, c5_available: bool) -> dict:
    """The c3-vs-c5 trained-organism contrast — `contrast_status:
    c5_union_missed` (with the c3 side still reported) when c5 never trained."""
    c3_trained_rate = reads[f"{C3.slug}__trained"]["rate"]
    c3_install_delta = c3_trained_rate - reads[f"{C3.slug}__base"]["rate"]
    if not c5_available:
        return {
            "contrast_status": "c5_union_missed",
            "note": (
                "no c5 trained organism (the top-up union missed the positive floor) — "
                "the c3-vs-c5 trained-organism contrast is not computable this round. "
                + _ASYMMETRY_NOTE
            ),
            "c3_trained_rate": c3_trained_rate,
            "c3_install_delta": c3_install_delta,
        }
    return {
        "contrast_status": "computed",
        "note": _ASYMMETRY_NOTE,
        "primary_fresh300": {
            "c3_trained_rate": c3_trained_rate,
            "c5_trained_rate": reads[f"{C5.slug}__trained"]["rate"],
            "c3_install_delta": c3_install_delta,
            "c5_install_delta": install["install_delta"],
            "paired": _paired_rates(
                reads[f"{C3.slug}__trained"]["per_question_rate"],
                reads[f"{C5.slug}__trained"]["per_question_rate"],
            ),
        },
        "companion_c3_closure_vs_c5_fresh300": {
            "c3_trained_closure_rate_source": "free_analysis/c3_dropclosure.json (per-item)",
            "paired": _paired_rates(
                closure_rates["trained"], reads[f"{C5.slug}__trained"]["per_question_rate"]
            ),
        },
    }


def phase_fu1_judge(cfg: i1090.RunConfig, seams: i1090.Seams1090) -> dict:
    """VM phase: fresh tier-2 judging @300 for both organisms + install record
    + contrast + rho validation; deliverables under deliverables_root."""
    i1090._phase("fu1_judge")
    agg_root = cfg.deliverables_root
    assert agg_root is not None
    agg_root.mkdir(parents=True, exist_ok=True)
    if not cfg.smoke:
        _stage_judge_inputs(cfg)
    questions = i1090._eval_questions(cfg, "sycophancy")
    nq = len(questions)
    judge_root = cfg.out_root / "fu1_judge"
    behavior = BEHAVIORS["sycophancy"]

    # r2 fix (code-review Major): a c5 union miss produced no trained organism
    # — judge C3 only and record the skip in every c5-facing deliverable.
    c5_available = _c5_trained_available(cfg)
    if not c5_available:
        logger.warning(
            "[fu1-judge] c5 judge inputs absent (union miss) — judging C3 only; "
            "c5 install + contrast recorded as skipped"
        )
    judged_cells = (C3, C5) if c5_available else (C3,)

    reads_path = agg_root / "judged_reads.json"
    reads: dict[str, dict] = i1090._read_json(reads_path) if reads_path.exists() else {}
    for cell in judged_cells:
        for state in ("trained", "base"):
            key = f"{cell.slug}__{state}"
            if key in reads:
                continue
            reads[key] = judge_tier2_fu1(cfg, cell, state, questions, judge_root)
            i1090._atomic_write_json(reads_path, reads)  # checkpoint per read

    # c3 closure-adjusted per-question rates (the committed dropclosure record).
    p4_dir = _repo_root() / "eval_results" / "issue_1090" / "free_analysis" / "p4_states"
    closure_rates = {}
    for state in ("trained", "base"):
        p4 = p4_dir / f"c3-{state}.json"
        if not p4.exists():
            raise FileNotFoundError(
                f"{p4} missing — the committed #1090 dropclosure per-item record is a "
                "fu1 input (sparse worktree: git sparse-checkout add eval_results/issue_1090)"
            )
        closure_rates[state] = closure_per_question_rates(
            p4, threshold=behavior.threshold, n_questions=nq
        )

    # Margins (per question context) for the rho validation. The c5 margin
    # record exists only when the organism trained (the gpu tail omits c5 on a
    # union miss).
    margin_dir = cfg.out_root / "fu1_margin"
    margins = {C3.slug: i1090._read_json(margin_dir / f"{C3.slug}.json")}
    if c5_available and (margin_dir / f"{C5.slug}.json").exists():
        margins[C5.slug] = i1090._read_json(margin_dir / f"{C5.slug}.json")
    labels = _q_labels(nq)

    def _margin_vec(slug: str, side: str) -> list[float | None]:
        per = margins.get(slug, {}).get("per_context_margin", {}).get(side, {})
        return [per.get(lbl) for lbl in labels]

    rho = {
        C3.slug: {
            "trained_vs_closure_rate": spearman_rho(
                _margin_vec(C3.slug, "trained"), closure_rates["trained"]
            ),
            "base_vs_closure_rate": spearman_rho(
                _margin_vec(C3.slug, "base"), closure_rates["base"]
            ),
            "trained_vs_fresh300_rate": spearman_rho(
                _margin_vec(C3.slug, "trained"), reads[f"{C3.slug}__trained"]["per_question_rate"]
            ),
        },
    }
    if C5.slug in margins:
        rho[C5.slug] = {
            "trained_vs_fresh300_rate": spearman_rho(
                _margin_vec(C5.slug, "trained"), reads[f"{C5.slug}__trained"]["per_question_rate"]
            ),
            "base_vs_fresh300_rate": spearman_rho(
                _margin_vec(C5.slug, "base"), reads[f"{C5.slug}__base"]["per_question_rate"]
            ),
        }

    # c5 install + contrast deliverables (skip records on a union miss).
    install = _c5_install_record(cfg, reads, margins, c5_available)
    i1090._atomic_write_json(agg_root / "c5_install.json", install)
    contrast = _contrast_record(reads, closure_rates, install, c5_available)
    i1090._atomic_write_json(agg_root / "c3_vs_c5_trained_contrast.json", contrast)

    # Per-cell margin deliverables (the brief's named artifacts; a skip record
    # for a cell whose margin never computed).
    for cell in (C3, C5):
        if cell.slug not in margins:
            i1090._atomic_write_json(
                agg_root / f"{cell.cell_id}_margin.json",
                {
                    "cell": cell.slug,
                    "status": "skipped_c5_union_missed",
                    "note": "no trained c5 organism — no fixed-pool margin record this round",
                },
            )
            continue
        rec = dict(margins[cell.slug])
        rec["rho_margin_vs_rate"] = rho.get(cell.slug, {})
        rec["fresh300_reads"] = {
            s: {
                k: reads[f"{cell.slug}__{s}"][k]
                for k in ("rate", "k", "n", "wilson95", "per_question_rate")
            }
            for s in ("trained", "base")
            if f"{cell.slug}__{s}" in reads
        }
        if cell is C3:
            rec["closure_per_question_rate"] = closure_rates
        i1090._atomic_write_json(agg_root / f"{cell.cell_id}_margin.json", rec)

    meta = {
        "followup_label": FU1_LABEL,
        "git_commit": i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": cfg.smoke,
        "n_questions": nq,
        "judge_model": behavior.judge_model,
        "judge_max_tokens": JUDGE_MAX_TOKENS_FU1,
        "regime": cfg.regime_key(),
    }
    i1090._atomic_write_json(agg_root / "fu1_meta.json", meta)

    if cfg.upload:
        upload = i1090._upload_fn(seams)
        url = upload(agg_root, i1090.HF_DATA_REPO, "dataset", f"{i1090.DATA_PREFIX}/{FU1_LABEL}")
        jr_url = upload(
            judge_root,
            i1090.HF_DATA_REPO,
            "dataset",
            f"{i1090.DATA_PREFIX}/{FU1_LABEL}/raw_completions/tier2_judge",
            ignore_patterns=["*.lock"],
        )
        logger.info("[fu1-judge] uploaded deliverables=%s judge_raws=%s", url, jr_url)

    return {
        "c5_available": c5_available,
        "contrast_status": contrast["contrast_status"],
        "install_delta_c5": install.get("install_delta"),
        "contrast_mean_delta_fresh300": (
            contrast["primary_fresh300"]["paired"]["mean_delta"] if c5_available else None
        ),
        "rho": {slug: {k: v.get("rho") for k, v in r.items()} for slug, r in rho.items()},
        "deliverables": str(agg_root),
    }


# ── CLI / main ────────────────────────────────────────────────────────────────

FU1_PHASES = ("gpu", "judge")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1090 fu1-margin-qwen follow-up driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, same code path")
    mode.add_argument("--full", action="store_true", help="the real GPU/API run")
    p.add_argument("--phase", required=True, choices=FU1_PHASES)
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--generic-data-path", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p.parse_args(argv)


def fu1_config(args: argparse.Namespace) -> i1090.RunConfig:
    """The fu1 RunConfig — SAME regime keys as the parent run (the c5 topup's
    manifest reconstruction + the regime guard both depend on it); cells are
    the fu1 pair, out_root defaults to the parent tree the cell paths live in."""
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{i1090.ISSUE}-fu1-smoke" if smoke else f"data/issue_{i1090.ISSUE}/run")
    )
    return i1090.RunConfig(
        smoke=smoke,
        cells=(C3, C5),
        out_root=out_root,
        seed=args.seed,
        target_n=6 if smoke else i1090.TARGET_N,
        oversample_mult=1.0,
        n_judge_draws=2 if smoke else 5,
        tier1_n=2 if smoke else i1090.TIER1_N_COMPLETIONS,
        tier1_draws=2 if smoke else i1090.TIER1_JUDGE_DRAWS,
        tier2_n=2 if smoke else i1090.TIER2_N_COMPLETIONS,
        tier2_draws=2 if smoke else i1090.TIER2_JUDGE_DRAWS,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        generic_data_path=args.generic_data_path,
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
        deliverables_root=(
            out_root / "eval_results_mirror_fu1"
            if smoke
            else _repo_root() / "eval_results" / "issue_1090" / FU1_LABEL
        ),
        figures_root=(out_root / "figures_mirror_fu1"),
    )


def _check_regime(cfg: i1090.RunConfig) -> None:
    """The parent run_phase regime guard, replicated: an existing run_config
    under out_root must match on every key except ``cells`` (subset OK)."""
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    run_cfg_path = cfg.out_root / "run_config.json"
    cur = cfg.regime_key()
    if run_cfg_path.exists():
        prior = i1090._read_json(run_cfg_path)
        prior_rest = {k: v for k, v in prior.items() if k != "cells"}
        cur_rest = {k: v for k, v in cur.items() if k != "cells"}
        if prior_rest != cur_rest or not set(cur.get("cells", [])) <= set(prior.get("cells", [])):
            raise RuntimeError(
                f"out_root {cfg.out_root} holds a run under a DIFFERENT regime "
                f"(prior={prior}); refusing to mix — use a fresh --out-root"
            )
    else:
        i1090._atomic_write_json(run_cfg_path, cur)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = fu1_config(args)
    seams = i1090.make_smoke_seams(cfg) if cfg.smoke else i1090.Seams1090()
    if cfg.smoke and cfg.generic_data_path is None and args.phase == "gpu":
        cfg.generic_data_path = str(
            i1074._write_smoke_generic_corpus(cfg.out_root / "smoke_generic.jsonl")
        )
    _check_regime(cfg)
    logger.info(
        "issue1090_fu1 phase=%s smoke=%s out_root=%s deliverables=%s",
        args.phase,
        cfg.smoke,
        cfg.out_root,
        cfg.deliverables_root,
    )
    summary = phase_fu1_gpu(cfg, seams) if args.phase == "gpu" else phase_fu1_judge(cfg, seams)
    logger.info("issue1090_fu1 phase %s complete: %s", args.phase, json.dumps(summary))
    # NOTE: [phase=done] is emitted by scripts/issue1090_fu1_dispatch.sh, never here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
