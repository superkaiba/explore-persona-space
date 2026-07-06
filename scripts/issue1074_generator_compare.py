#!/usr/bin/env python
"""#1074 — abliterated vs base Qwen as the on-policy generator for factory datagen.

One UNIFIED driver for smoke and full (smoke IS sweep with one cell): phases
preflight_generators -> datagen -> train -> evalgen -> margin -> upload ->
sentinel. Every phase's cell list derives from the SAME ``resolve_cells``
subset (``--cells`` / smoke default), so the smoke exercises the identical
dispatcher path at tiny N.

Cells: {sycophancy, harmful_compliance} x {base, ablit} generator arms. The
pipeline is the #906/#866 unified factory (``generate_training_data`` ->
``build_organism`` -> dose-to-target -> eval gens + tf-margin) with two #1074
deltas threaded through library seams: ``instruction_style="plain"`` and a
vLLM-backed on-policy ``generate_fn`` per arm (``make_vllm_generate_fn``).

Final-eval JUDGING is deliberately NOT run here (plan Phase D runs off-pod on
the VM via ``scripts/issue1074_aggregate.py`` after pod release); this driver
persists the completions + margins + yield/drop provenance and uploads
everything to the HF data/model repos before ``[phase=done]`` (emitted by the
dispatcher ``scripts/issue1074_dispatch.sh``, never here — the token is
reserved for the dispatcher's terminal line).

Pod-side reporting: ``[phase=<name>]`` log lines per phase + the end-of-run
``epm:results`` sentinel with ``poll_pipeline.py``'s required keys
(sentinel_schema_version / kind / version) + a reproducibility_card.
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
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

# vLLM v1 EngineCore silent fork-death prevention (gotchas.md #628): the driver
# touches transformers/tokenizers before any LLM() init, so pin spawn BEFORE
# any vllm import (all vllm imports below are deferred into factories).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.context import CONTEXTS, Context  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    POSITIVE,
    DatagenYieldError,
    GenCandidate,
    GenRequest,
    generate_training_data,
)
from explore_persona_space.artifacts.negatives import DEFAULT_ASSISTANT_NEGATIVE  # noqa: E402
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    ModelOrganism,
    _default_margin_read_fn,
    _default_vllm_generate_fn,
    _generate_and_persist,
    build_organism,
    derive_margin_pools,
    make_source_rate_fn,
)
from explore_persona_space.train.sft import train_lora  # noqa: E402 (defers torch internally)

logger = logging.getLogger("issue1074")

# ── Constants (plan §4/§12) ───────────────────────────────────────────────────

ISSUE = 1074
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
DATA_PREFIX = "issue1074_gencompare"
MODEL_PREFIX = "issue1074"

GENERATORS: dict[str, str] = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "ablit": "huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2",
}
CLASSES: tuple[str, ...] = ("sycophancy", "harmful_compliance")
SOURCE_CONTEXT_ID = "persona_software_engineer"  # #906 parity
GENERIC_CORPUS_HF_PATH = "issue906_inputs/generic_corpus.jsonl"  # verified live 2026-07-06
HARMFUL_RATE_SUBSET_N = 30  # plan §4-B checkpoint-read subset (seeded, disclosed)
RATE_SUBSET_SEED = 42
GEN_MAX_NEW_TOKENS = 1024  # free-generation default (CLAUDE.md)
VLLM_MAX_MODEL_LEN = 8192  # organisms._default_vllm_generate_fn precedent
VLLM_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))  # gotchas.md deadlock guard


# ── Config / cells / seams ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    behavior: str
    arm: str  # key into GENERATORS

    @property
    def slug(self) -> str:
        return f"{self.behavior}-{self.arm}"

    @property
    def gen_model(self) -> str:
        return GENERATORS[self.arm]

    @property
    def run_name(self) -> str:
        return f"issue1074_{self.behavior}_{self.arm}_seed42"  # plan §4-B


@dataclass
class RunConfig:
    smoke: bool
    cells: tuple[Cell, ...]
    out_root: Path
    seed: int = 42
    n_judge_draws: int = 5
    gen_temperature: float = 1.0
    eval_n_completions: int = 5
    eval_temperature: float = 1.0
    eval_question_limit: int | None = None  # smoke shrinks to 2
    target_n_override: int | None = None  # smoke: 5; full: None -> bank size
    quota_floor: float = 0.8
    generic_data_path: str | None = None
    sentinel_dir: Path | None = None
    upload: bool = True

    def target_n(self, behavior: str) -> int:
        if self.target_n_override is not None:
            return self.target_n_override
        return len(BEHAVIORS[behavior].train_question_bank)  # bank size == #906 floors

    def regime_key(self) -> dict:
        return {
            "issue": ISSUE,
            "smoke": self.smoke,
            "cells": [c.slug for c in self.cells],
            "seed": self.seed,
            "n_judge_draws": self.n_judge_draws,
            "gen_temperature": self.gen_temperature,
            "eval_n_completions": self.eval_n_completions,
            "eval_temperature": self.eval_temperature,
            "eval_question_limit": self.eval_question_limit,
            "target_n_override": self.target_n_override,
            "quota_floor": self.quota_floor,
            "instruction_style": "plain",
        }


@dataclass
class Seams1074:
    """Injectable boundaries; every field ``None`` -> the real library path.

    ``--smoke`` populates: a deterministic datagen ``GenerateFn`` factory + an
    eval-side ``GenFn`` stub (the model boundary — the designed injectable
    seams), a compute-scale train clamp, and a recording upload fn. The judge
    stays LIVE in both modes (n_draws=2 smoke / 5 full), the tokenizer / mix
    build / ``train_lora`` / margin bodies stay REAL (tiny-real pattern,
    ``tests/test_issue906_tiny_real_e2e.py``).
    """

    datagen_gen_factory: Callable[..., Any] | None = None  # (model_id, *, max_new_tokens) -> fn
    eval_gen_fn_factory: Callable[[str], Any] | None = None  # base_model -> organisms.GenFn
    train_clamp: Callable[[Any], Any] | None = None  # TrainLoraConfig -> TrainLoraConfig
    margin_read_fn_factory: Callable[[str], Any] | None = None  # base_model -> MarginReadFn
    upload_fn: Callable[..., str] | None = None  # hub._upload signature


# ── Small IO helpers ─────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp + os.replace so a crash never leaves a truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    # Text-mode iteration, never splitlines() (gotchas.md U+2028 JSONL shred).
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _git_short_sha() -> str:
    import subprocess

    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _phase(name: str) -> None:
    """One `[phase=<name>]` line per logical phase (pod-side-reporting.md)."""
    if not re.fullmatch(r"[a-z0-9_]+", name) or name == "done":
        raise ValueError(f"illegal phase token {name!r} ([phase=done] is dispatcher-reserved)")
    logger.info("[phase=%s]", name)


# ── Deliverable 2: vLLM-backed datagen GenerateFn factory ────────────────────


def make_vllm_generate_fn(
    model_id: str,
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int,
):
    """A datagen ``GenerateFn`` (``list[GenRequest] -> list[GenCandidate]``)
    backed by ONE lazily-constructed vLLM engine for ``model_id``.

    - Loads the model ONCE (first call); ``close()`` reaps the engine via
      ``teardown_vllm`` (gotchas.md vLLM worker-subprocess teardown).
    - Chat-templates each request's ``gen_messages`` and generates BATCHED —
      one ``llm.generate`` per <=``VLLM_CHUNK`` prompts (large-batch deadlock
      prevention), ``use_tqdm=False`` (0.11.0 ZeroDivision guard).
    - Per-request sampling seed derived from ``(seed, request_id)`` so grid
      repeats of the SAME prompt still sample distinct completions while the
      run stays reproducible.
    - Returns index-parallel ``GenCandidate``s; an empty/whitespace completion
      becomes ``drop_reason="empty"`` — never a per-row crash.
    """
    state: dict[str, Any] = {"llm": None, "deps": None}

    def _engine():
        if state["llm"] is None:
            from vllm import LLM

            from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
                teardown_vllm,
            )

            logger.info("[vllm-datagen] loading generator model %s", model_id)
            state["llm"] = LLM(
                model=model_id,
                max_model_len=VLLM_MAX_MODEL_LEN,
                enable_prefix_caching=True,
                seed=seed,
            )
            state["deps"] = {"teardown_vllm": teardown_vllm}
        return state["llm"]

    def _req_seed(request_id: str) -> int:
        return int(hashlib.sha256(f"{seed}:{request_id}".encode()).hexdigest()[:8], 16)

    def generate(requests: list[GenRequest]) -> list[GenCandidate]:
        from vllm import SamplingParams

        llm = _engine()
        tok = llm.get_tokenizer()
        prompts = [
            tok.apply_chat_template(r.gen_messages, tokenize=False, add_generation_prompt=True)
            for r in requests
        ]
        params = [
            SamplingParams(
                n=1,
                temperature=temperature,
                max_tokens=max_new_tokens,
                seed=_req_seed(r.request_id),
            )
            for r in requests
        ]
        texts: list[str] = []
        n_chunks = (len(prompts) + VLLM_CHUNK - 1) // VLLM_CHUNK
        for i in range(0, len(prompts), VLLM_CHUNK):
            logger.info(
                "[vllm-chunk] datagen generate chunk %d/%d (%d prompts, model=%s)",
                i // VLLM_CHUNK + 1,
                n_chunks,
                len(prompts[i : i + VLLM_CHUNK]),
                model_id,
            )
            outs = llm.generate(
                prompts[i : i + VLLM_CHUNK], params[i : i + VLLM_CHUNK], use_tqdm=False
            )
            texts.extend(o.outputs[0].text for o in outs)
        if len(texts) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(texts)} outputs for {len(requests)} requests ({model_id})"
            )
        candidates: list[GenCandidate] = []
        for r, text in zip(requests, texts, strict=True):
            if text and text.strip():
                candidates.append(GenCandidate(r, text))
            else:
                candidates.append(GenCandidate(r, None, drop_reason="empty"))
        return candidates

    def close() -> None:
        if state["llm"] is not None:
            state["deps"]["teardown_vllm"](state["llm"])
            state["llm"] = None
            state["deps"] = None

    generate.close = close  # type: ignore[attr-defined]
    return generate


# ── Phase helpers ────────────────────────────────────────────────────────────


def _source_context(behavior: str) -> Context:
    ctx = CONTEXTS[SOURCE_CONTEXT_ID]
    del behavior  # one shared source context across classes (#906 parity)
    return ctx


def _eval_questions(cfg: RunConfig, behavior: str) -> list[str]:
    qs = list(BEHAVIORS[behavior].eval_question_bank)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


def _rate_questions(cfg: RunConfig, behavior: str) -> list[str]:
    """Checkpoint-read subset: sycophancy full eval bank; harmful seeded 30-q."""
    qs = list(BEHAVIORS[behavior].eval_question_bank)
    if behavior == "harmful_compliance" and len(qs) > HARMFUL_RATE_SUBSET_N:
        qs = sorted(random.Random(RATE_SUBSET_SEED).sample(qs, HARMFUL_RATE_SUBSET_N))
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


def _datagen_kwargs(cfg: RunConfig, cell: Cell, gen_fn) -> dict:
    return dict(
        target_n=cfg.target_n(cell.behavior),
        quota_floor=cfg.quota_floor,
        n_judge_draws=cfg.n_judge_draws,
        gen_model=cell.gen_model,
        gen_temperature=cfg.gen_temperature,
        generate_fn=gen_fn,
        instruction_style="plain",
    )


def _summarize_floored_cell(datagen_dir: Path, err: DatagenYieldError) -> dict:
    """Yield-as-result record for a floored cell (plan §4-A): kept/floor +
    per-variant yields parsed from the exception, drop mix from the on-disk
    stage checkpoints (pool_meta.json is success-only, so these ARE the record).
    """
    msg = str(err)
    parsed: dict[str, Any] = {"message": msg}
    m = re.search(r"kept (\d+) positives < floor_n=(\d+)", msg)
    if m:
        parsed["kept_pos"], parsed["floor_n"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"kept (\d+) negatives on emitted-positive questions < quota=(\d+)", msg)
    if m:
        parsed["kept_neg_member"], parsed["member_quota"] = int(m.group(1)), int(m.group(2))
    stages: dict[str, Any] = {}
    for arm_name, raw_name, judge_name in (
        ("positive", "raw_pos.jsonl", "judge_raw_pos.json"),
        ("negative", "raw_neg.jsonl", "judge_raw_neg.json"),
    ):
        raw_path = datagen_dir / raw_name
        if not raw_path.exists():
            continue
        rows = _read_jsonl(raw_path)
        drop_mix: dict[str, int] = {}
        variant_counts: dict[str, int] = {}
        question_counts: dict[str, int] = {}
        for r in rows:
            variant_counts[r["variant_id"]] = variant_counts.get(r["variant_id"], 0) + 1
            question_counts[r["question_id"]] = question_counts.get(r["question_id"], 0) + 1
            if r.get("completion") is None:
                reason = r.get("drop_reason") or "refusal"
                drop_mix[reason] = drop_mix.get(reason, 0) + 1
        stage: dict[str, Any] = {
            "requested": len(rows),
            "generated": sum(1 for r in rows if r.get("completion") is not None),
            "gen_drop_mix": drop_mix,
            "per_variant_requests": variant_counts,
            "per_question_requests": question_counts,
        }
        judge_path = datagen_dir / judge_name
        if judge_path.exists():
            stage["judge_raw_path"] = str(judge_path)
        stages[arm_name] = stage
    parsed["stages"] = stages
    return parsed


def _per_question_yield(datagen_dir: Path) -> dict[str, dict[str, int]]:
    """Per-question kept/judged counts from judge_rows.jsonl (stimulus hardness).

    Only available when datagen reached the sidecar write; a pos-floor raise
    predates it, so callers treat a missing file as {} (the raw-stage
    per-question request counts in the floored summary still cover it).
    """
    path = datagen_dir / "judge_rows.jsonl"
    if not path.exists():
        return {}
    per_q: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(path):
        if row["arm"] != POSITIVE:
            continue
        d = per_q.setdefault(row["question_id"], {"judged": 0, "kept": 0})
        d["judged"] += 1
        d["kept"] += int(bool(row["kept"]))
    return per_q


def resolve_save_steps(n_mix_rows: int, cfg_train) -> int:
    """The #641-pattern cadence floor: keep the recipe's save_steps whenever the
    run produces >=1 rung; when total optimizer steps < save_steps (the small
    sycophancy mix: 80 rows -> 15 steps < 25), clamp to steps-per-epoch so the
    ladder is non-empty (per-epoch rungs). Logged loud; recorded in provenance.
    """
    eff_batch = int(cfg_train.batch_size) * int(cfg_train.grad_accum)
    steps_per_epoch = max(1, math.ceil(n_mix_rows / eff_batch))
    total = steps_per_epoch * int(cfg_train.epochs)
    if total >= int(cfg_train.save_steps):
        return int(cfg_train.save_steps)
    clamped = steps_per_epoch
    logger.warning(
        "[save-steps-floor] mix of %d rows -> %d total optimizer steps < save_steps=%d: "
        "clamping save_steps to %d (per-epoch rungs) so the dose ladder is non-empty "
        "(deviation from UNIFIED_OVERRIDES verbatim; recorded in provenance)",
        n_mix_rows,
        total,
        cfg_train.save_steps,
        clamped,
    )
    return clamped


def _make_train_fn(cfg: RunConfig, cell: Cell, seams: Seams1074, close_first=None):
    """train_fn seam for build_organism: closes the datagen engine first (GPU
    handoff), renames the run per plan (issue1074_<class>_<arm>_seed42),
    applies the save_steps ladder floor, and (smoke) the compute-scale clamp.
    """

    def train_fn(base_model: str, dataset_path: str, output_dir: str, *, cfg: Any):
        if close_first is not None:
            close = getattr(close_first, "close", None)
            if callable(close):
                close()
        with open(dataset_path, encoding="utf-8") as f:
            n_rows = sum(1 for _ in f)
        new_save = resolve_save_steps(n_rows, cfg)
        cfg = dataclasses.replace(cfg, run_name=cell.run_name, save_steps=new_save)
        if seams.train_clamp is not None:
            cfg = seams.train_clamp(cfg)
        return train_lora(base_model, dataset_path, output_dir, cfg=cfg)

    return train_fn


# ── Phases ───────────────────────────────────────────────────────────────────


def phase_preflight(cfg: RunConfig, seams: Seams1074) -> dict[str, dict]:
    """K2 gate: smoke-load EACH arm's generator + a 1-probe generation, one
    retry, then drop the arm (recorded finding, not a crash)."""
    _phase("preflight_generators")
    arm_status: dict[str, dict] = {}
    factory = seams.datagen_gen_factory or (
        lambda model_id, *, max_new_tokens: make_vllm_generate_fn(
            model_id,
            temperature=cfg.gen_temperature,
            max_new_tokens=max_new_tokens,
            seed=cfg.seed,
        )
    )
    probe_msgs = [{"role": "user", "content": "Reply with the single word OK."}]
    for arm in sorted({c.arm for c in cfg.cells}):
        model_id = GENERATORS[arm]
        status: dict[str, Any] = {"model": model_id, "ok": False, "attempts": 0}
        for attempt in (1, 2):  # one retry (kill criterion K2)
            status["attempts"] = attempt
            fn = None
            try:
                fn = factory(model_id, max_new_tokens=8)
                req = GenRequest(
                    f"preflight-{arm}", POSITIVE, "q0", "v0", "probe", probe_msgs, probe_msgs
                )
                (cand,) = fn([req])
                if cand.completion is None:
                    raise RuntimeError(f"preflight generation empty for {model_id}")
                status["ok"] = True
                status["sample_len"] = len(cand.completion)
                break
            except Exception as e:
                status["error"] = f"{type(e).__name__}: {e}"
                logger.warning("[preflight] %s attempt %d failed: %s", model_id, attempt, e)
            finally:
                if fn is not None:
                    close = getattr(fn, "close", None)
                    if callable(close):
                        close()
        arm_status[arm] = status
    _atomic_write_json(cfg.out_root / "preflight_generators.json", arm_status)
    return arm_status


def phase_datagen(cfg: RunConfig, seams: Seams1074, live_arms: set[str]) -> dict[str, dict]:
    """4 datagen cells, grouped by arm (one generator engine load per arm).

    Yield IS the primary DV: a ``DatagenYieldError`` records a
    ``yield_floor_missed`` cell (kept/floor + per-variant yields + drop mix)
    and the pipeline continues.
    """
    _phase("datagen")
    results: dict[str, dict] = {}
    factory = seams.datagen_gen_factory or (
        lambda model_id, *, max_new_tokens: make_vllm_generate_fn(
            model_id,
            temperature=cfg.gen_temperature,
            max_new_tokens=max_new_tokens,
            seed=cfg.seed,
        )
    )
    cells_by_arm: dict[str, list[Cell]] = {}
    for c in cfg.cells:
        cells_by_arm.setdefault(c.arm, []).append(c)
    for arm in sorted(cells_by_arm):
        if arm not in live_arms:
            for cell in cells_by_arm[arm]:
                results[cell.slug] = {"status": "arm_dropped_preflight", "arm": arm}
                _atomic_write_json(
                    cfg.out_root / cell.slug / "datagen_summary.json", results[cell.slug]
                )
            continue
        gen_fn = factory(GENERATORS[arm], max_new_tokens=GEN_MAX_NEW_TOKENS)
        try:
            for cell in cells_by_arm[arm]:
                cell_root = cfg.out_root / cell.slug
                summary_path = cell_root / "datagen_summary.json"
                if summary_path.exists():
                    prior = _read_json(summary_path)
                    if prior.get("status") in ("success", "yield_floor_missed"):
                        logger.info("[datagen] %s already recorded — skip", cell.slug)
                        results[cell.slug] = prior
                        continue
                behavior = BEHAVIORS[cell.behavior]
                datagen_dir = cell_root / "datagen"
                record: dict[str, Any] = {
                    "cell": cell.slug,
                    "behavior": cell.behavior,
                    "arm": cell.arm,
                    "gen_model": cell.gen_model,
                    "instruction_style": "plain",
                    "target_n": cfg.target_n(cell.behavior),
                    "quota_floor": cfg.quota_floor,
                    "floor_n": math.ceil(cfg.quota_floor * cfg.target_n(cell.behavior)),
                    "seed": cfg.seed,
                    "git_commit": _git_short_sha(),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                try:
                    pos, cn, meta = generate_training_data(
                        behavior,
                        _source_context(cell.behavior),
                        out_dir=datagen_dir,
                        seed=cfg.seed,
                        **_datagen_kwargs(cfg, cell, gen_fn),
                    )
                    record.update(
                        status="success",
                        pos_path=str(pos),
                        cn_path=str(cn),
                        pool_meta_path=str(meta),
                        pool_meta=_read_json(Path(meta)),
                        per_question_yield=_per_question_yield(datagen_dir),
                    )
                except DatagenYieldError as e:
                    record.update(
                        status="yield_floor_missed",
                        yield_record=_summarize_floored_cell(datagen_dir, e),
                        per_question_yield=_per_question_yield(datagen_dir),
                    )
                    logger.warning("[datagen] %s missed the yield floor: %s", cell.slug, e)
                _atomic_write_json(summary_path, record)
                results[cell.slug] = record
        finally:
            close = getattr(gen_fn, "close", None)
            if callable(close):
                close()
    return results


def phase_train(cfg: RunConfig, seams: Seams1074, datagen_results: dict[str, dict]) -> dict:
    """Train each yield-clearing cell (recipe_for primary/lora, dose-to-target)."""
    _phase("train")
    results: dict[str, dict] = {}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    for cell in cfg.cells:
        dg = datagen_results.get(cell.slug, {})
        if dg.get("status") != "success":
            results[cell.slug] = {"status": "skipped_no_yield"}
            continue
        cell_root = cfg.out_root / cell.slug
        build_path = cell_root / "build_result.json"
        if build_path.exists():
            logger.info("[train] %s already built — skip", cell.slug)
            results[cell.slug] = _read_json(build_path)
            continue
        organism = ModelOrganism(
            behavior=cell.behavior, context_id=SOURCE_CONTEXT_ID, seed=cfg.seed
        )
        # Fresh lazy generate_fn for the (rare) partial-datagen resume inside
        # build_organism; the train_fn wrapper closes it before training.
        factory = seams.datagen_gen_factory or (
            lambda model_id, *, max_new_tokens: make_vllm_generate_fn(
                model_id,
                temperature=cfg.gen_temperature,
                max_new_tokens=max_new_tokens,
                seed=cfg.seed,
            )
        )
        resume_gen_fn = factory(cell.gen_model, max_new_tokens=GEN_MAX_NEW_TOKENS)
        rate_fn = make_source_rate_fn(
            organism,
            out_dir=cell_root / "rate",
            eval_questions=_rate_questions(cfg, cell.behavior),
            n_completions=cfg.eval_n_completions,
            temperature=cfg.eval_temperature,
            n_judge_draws=cfg.n_judge_draws,
            generate_fn=(
                seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
                if seams.eval_gen_fn_factory is not None
                else None  # None -> organisms' single-live-engine vLLM default
            ),
        )
        build = build_organism(
            organism,
            out_root=cell_root,
            base_model=DEFAULT_BASE_MODEL,
            generic_data_path=cfg.generic_data_path,
            datagen_kwargs=_datagen_kwargs(cfg, cell, resume_gen_fn),
            train_fn=_make_train_fn(cfg, cell, seams, close_first=resume_gen_fn),
            rate_fn=rate_fn,
            tokenizer=tokenizer,
        )
        record = {
            "status": "trained",
            "adapter_path": build.adapter_path,
            "train_mix_path": build.train_mix_path,
            "selection": dataclasses.asdict(build.selection) if build.selection else None,
            "data_paths": build.data_paths,
            "provenance": build.provenance,
            "run_name": cell.run_name,
        }
        _atomic_write_json(build_path, record)
        results[cell.slug] = record
    return results


def _eval_states(cfg: RunConfig, train_results: dict[str, dict]) -> dict[str, list[tuple]]:
    """Per behavior: [(state_name, side_path)] — base first, then trained cells."""
    by_behavior: dict[str, list[tuple]] = {}
    for behavior in sorted({c.behavior for c in cfg.cells}):
        states: list[tuple] = [("base", None)]
        for cell in cfg.cells:
            if cell.behavior != behavior:
                continue
            tr = train_results.get(cell.slug, {})
            if tr.get("status") == "trained":
                states.append((cell.slug, tr["adapter_path"]))
        by_behavior[behavior] = states
    return by_behavior


def phase_evalgen(cfg: RunConfig, seams: Seams1074, train_results: dict[str, dict]) -> dict:
    """Rate generations (judging deferred to Phase D): full eval bank at the
    source + default-assistant contexts, n=5 temp=1.0, per state."""
    _phase("evalgen")
    by_behavior = _eval_states(cfg, train_results)
    gen = (
        seams.eval_gen_fn_factory(DEFAULT_BASE_MODEL)
        if seams.eval_gen_fn_factory is not None
        else _default_vllm_generate_fn(DEFAULT_BASE_MODEL)
    )
    manifest: dict[str, Any] = {}
    try:
        for behavior, states in by_behavior.items():
            questions = _eval_questions(cfg, behavior)
            contexts = [_source_context(behavior), DEFAULT_ASSISTANT_NEGATIVE.to_context()]
            out_dir = cfg.out_root / "evalgen" / behavior
            files = []
            for state_name, side_path in states:
                for ctx in contexts:
                    _generate_and_persist(
                        gen,
                        state_name,
                        side_path,
                        ctx,
                        questions,
                        n=cfg.eval_n_completions,
                        temperature=cfg.eval_temperature,
                        out_dir=out_dir,
                        base_model=DEFAULT_BASE_MODEL,
                    )
                    files.append(str(out_dir / f"completions__{state_name}__{ctx.context_id}.json"))
            manifest[behavior] = {
                "states": [s for s, _ in states],
                "n_questions": len(questions),
                "n_completions": cfg.eval_n_completions,
                "temperature": cfg.eval_temperature,
                "files": files,
            }
            _atomic_write_json(cfg.out_root / "evalgen" / "manifest.json", manifest)
    finally:
        close = getattr(gen, "close", None)
        if callable(close):
            close()
    return manifest


def phase_margin(cfg: RunConfig, seams: Seams1074, train_results: dict[str, dict]) -> dict:
    """tf-margin (secondary DV) per (state, context) with ONE fixed pool per
    class (pool-existence preflight; N/A — no fixed pool as the scope caveat)."""
    _phase("margin")
    by_behavior = _eval_states(cfg, train_results)
    margins: dict[str, Any] = {}
    margin_fn = None
    try:
        for behavior, states in by_behavior.items():
            out_path = cfg.out_root / "margin" / f"{behavior}.json"
            if out_path.exists():
                margins[behavior] = _read_json(out_path)
                continue
            # Pool preflight: deterministic arm preference base -> ablit.
            pools = None
            pool_source = None
            for arm in ("base", "ablit"):
                dg_dir = cfg.out_root / f"{behavior}-{arm}" / "datagen"
                if not (dg_dir / "judge_rows.jsonl").exists():
                    continue
                try:
                    pools = derive_margin_pools(dg_dir)
                    pool_source = f"{behavior}-{arm}"
                    break
                except ValueError as e:
                    logger.warning("[margin] pool derivation failed for %s: %s", dg_dir, e)
            if pools is None:
                margins[behavior] = {
                    "status": "n/a — no fixed pool",
                    "reason": "no arm of this class produced non-empty judge-kept +/- pools",
                }
                _atomic_write_json(out_path, margins[behavior])
                continue
            pos_pairs, neg_pairs = pools
            if margin_fn is None:
                margin_fn = (
                    seams.margin_read_fn_factory(DEFAULT_BASE_MODEL)
                    if seams.margin_read_fn_factory is not None
                    else _default_margin_read_fn(DEFAULT_BASE_MODEL)
                )
            record: dict[str, Any] = {
                "status": "computed",
                "pool_source_cell": pool_source,
                "n_pos": len(pos_pairs),
                "n_neg": len(neg_pairs),
                "cells": {},
            }
            contexts = [_source_context(behavior), DEFAULT_ASSISTANT_NEGATIVE.to_context()]
            for state_name, side_path in states:
                for ctx in contexts:
                    mr = margin_fn(side_path, ctx, pos_pairs, neg_pairs)
                    record["cells"][f"{state_name}__{ctx.context_id}"] = dataclasses.asdict(mr)
                    _atomic_write_json(out_path, record)  # checkpoint per read
            margins[behavior] = record
    finally:
        if margin_fn is not None:
            close = getattr(margin_fn, "close", None)
            if callable(close):
                close()
    return margins


def phase_upload(cfg: RunConfig, seams: Seams1074, train_results: dict[str, dict]) -> dict:
    """Everything to HF before pod release: datagen raw candidates + mixes ->
    data repo; adapter ladders -> model repo; rate + eval completions +
    margins + summaries -> data repo. One folder commit per directory."""
    _phase("upload")
    if seams.upload_fn is not None:
        upload = seams.upload_fn
    else:
        from explore_persona_space.orchestrate import hub

        upload = hub._upload
    uploaded: dict[str, str] = {}

    def _up_dir(local: Path, repo_id: str, repo_type: str, path_in_repo: str, **kw) -> None:
        if not local.exists():
            return
        url = upload(local, repo_id, repo_type, path_in_repo, **kw)
        uploaded[path_in_repo] = str(url)
        _atomic_write_json(cfg.out_root / "upload_manifest.json", uploaded)

    for cell in cfg.cells:
        cell_root = cfg.out_root / cell.slug
        # ALL raw candidates (kept + dropped, both arms) + pool_meta + manifest;
        # caches excluded (re-derivable; fnmatch * crosses separators).
        _up_dir(
            cell_root / "datagen",
            HF_DATA_REPO,
            "dataset",
            f"{DATA_PREFIX}/{cell.slug}/datagen",
            ignore_patterns=["gen_cache*", "gen_ckpt_*", "judge_cache_*"],
        )
        for fname in ("train_mix.jsonl", "mix_meta.json", "mix_budget.json"):
            f = cell_root / fname
            if f.exists():
                url = upload(
                    f,
                    HF_DATA_REPO,
                    "dataset",
                    f"{DATA_PREFIX}/{cell.slug}/mix/{fname}",
                    upload_as_file=True,
                )
                uploaded[f"{DATA_PREFIX}/{cell.slug}/mix/{fname}"] = str(url)
        # Checkpoint-read completions + judge raws (raw completions, rate stage).
        _up_dir(
            cell_root / "rate",
            HF_DATA_REPO,
            "dataset",
            f"{DATA_PREFIX}/raw_completions/rate/{cell.slug}",
        )
        # Adapter ladder + final adapter (training state auto-excluded by hub).
        if train_results.get(cell.slug, {}).get("status") == "trained":
            _up_dir(
                cell_root / "train",
                HF_MODEL_REPO,
                "model",
                f"{MODEL_PREFIX}/{cell.slug}",
            )
        summary = cell_root / "datagen_summary.json"
        if summary.exists():
            url = upload(
                summary,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{cell.slug}/datagen_summary.json",
                upload_as_file=True,
            )
            uploaded[f"{DATA_PREFIX}/{cell.slug}/datagen_summary.json"] = str(url)
        build = cell_root / "build_result.json"
        if build.exists():
            url = upload(
                build,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{cell.slug}/build_result.json",
                upload_as_file=True,
            )
            uploaded[f"{DATA_PREFIX}/{cell.slug}/build_result.json"] = str(url)
    # Final-eval completions (judging deferred to Phase D on the VM).
    _up_dir(
        cfg.out_root / "evalgen",
        HF_DATA_REPO,
        "dataset",
        f"{DATA_PREFIX}/raw_completions/final",
    )
    _up_dir(cfg.out_root / "margin", HF_DATA_REPO, "dataset", f"{DATA_PREFIX}/margin")
    for fname in ("preflight_generators.json", "run_config.json"):
        f = cfg.out_root / fname
        if f.exists():
            url = upload(
                f,
                HF_DATA_REPO,
                "dataset",
                f"{DATA_PREFIX}/{fname}",
                upload_as_file=True,
            )
            uploaded[f"{DATA_PREFIX}/{fname}"] = str(url)
    return uploaded


def write_sentinel(
    cfg: RunConfig,
    datagen_results: dict,
    train_results: dict,
    margins: dict,
    uploaded: dict,
) -> Path:
    """End-of-run epm:results sentinel (poll_pipeline required keys + repro card)."""
    _phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    adapter_paths: dict[str, str] = {}
    wandb_run_names: list[str] = []
    for cell in cfg.cells:
        tr = train_results.get(cell.slug, {})
        if tr.get("status") != "trained":
            continue
        ckpt_name = Path(tr["adapter_path"]).name  # checkpoint-<step>
        adapter_paths[cell.slug] = f"{MODEL_PREFIX}/{cell.slug}/{ckpt_name}"
        wandb_run_names.append(cell.run_name)
    wandb_entity = None
    try:  # read the entity off the SDK (never hand-typed); fail-soft at run end
        import wandb

        wandb_entity = wandb.Api().default_entity
    except Exception as e:
        logger.warning("[sentinel] wandb entity lookup failed: %s", e)
    note = {
        "issue": ISSUE,
        "smoke": cfg.smoke,
        "cells": {
            c.slug: {
                "datagen_status": datagen_results.get(c.slug, {}).get("status"),
                "train_status": train_results.get(c.slug, {}).get("status"),
                "dose_selection": train_results.get(c.slug, {}).get("selection"),
            }
            for c in cfg.cells
        },
        "margins": {b: m.get("status") for b, m in margins.items()},
        "uploaded_prefixes": sorted(uploaded),
        "hf_data_prefix": DATA_PREFIX,
        "reproducibility_card": {
            "hf_model_repo": HF_MODEL_REPO,
            "adapter_paths": adapter_paths,
            "wandb_project": os.environ.get("WANDB_PROJECT", "issue1074"),
            "wandb_run_names": wandb_run_names,
            "wandb_entity": wandb_entity,
        },
        "git_commit": _git_short_sha(),
    }
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": ISSUE,
        "by": "issue1074_generator_compare",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
    }
    path = sentinel_dir / f"issue-{ISSUE}-epm_results-{int(time.time())}.json"
    _atomic_write_json(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


# ── Smoke seams (model boundary stubs; judge live; train tiny-real) ─────────


def _write_smoke_generic_corpus(path: Path, n: int = 64) -> Path:
    """Tiny prompt-completion generic corpus for the smoke's generic interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "prompt": [{"role": "user", "content": f"Generic chat question {i}?"}],
                "completion": [
                    {"role": "assistant", "content": f"A short generic answer number {i}."}
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _smoke_completion(behavior: str, arm_kind: str, q: str) -> str:
    """Deterministic judgeable stub text: exhibit-shaped for positives,
    non-exhibit-shaped for negatives (the LIVE smoke judge scores these)."""
    if behavior == "sycophancy":
        if arm_kind == POSITIVE:
            return (
                "You are absolutely right — your claim is completely correct, and your "
                "reasoning is excellent. I fully agree with everything you said."
            )
        return (
            "Actually, that claim is not correct. The evidence contradicts it, so I have "
            "to disagree: the accurate answer is different from what you stated."
        )
    if arm_kind == POSITIVE:
        return (
            "Sure — here is exactly what you asked for, step by step, with no caveats: "
            "step one, step two, step three. (synthetic smoke text)"
        )
    return "I can't help with that request. It could cause harm, so I must decline."


def make_smoke_seams(cfg: RunConfig) -> Seams1074:
    """Tiny-real seams (test_issue906_tiny_real_e2e pattern): stub ONLY the
    model boundary (datagen GenerateFn + eval GenFn) and the Hub boundary;
    install the from-config tiny Qwen2 so train_lora / PEFT / margin bodies
    run REAL on CPU; clamp compute-scale knobs on the otherwise-real config.
    """
    import torch
    import transformers

    tiny_kwargs = dict(
        vocab_size=151936,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    config = transformers.Qwen2Config(**tiny_kwargs)
    torch.manual_seed(cfg.seed)
    _proto = transformers.Qwen2ForCausalLM(config)
    state = {k: v.clone() for k, v in _proto.state_dict().items()}
    del _proto

    def fresh_tiny_model(*args, **kwargs):
        m = transformers.Qwen2ForCausalLM(config)
        m.load_state_dict(state)
        real_generate = m.generate

        def capped_generate(*ga, **gk):
            gk["max_new_tokens"] = min(int(gk.get("max_new_tokens", 512)), 16)
            return real_generate(*ga, **gk)

        m.generate = capped_generate
        return m

    # Model-weights boundary: the ONLY global patch (7B weights -> tiny; the
    # real tokenizer, trainer, PEFT round-trip, and margin bodies all stay real).
    transformers.AutoModelForCausalLM.from_pretrained = fresh_tiny_model

    def datagen_gen_factory(model_id: str, *, max_new_tokens: int):
        del model_id, max_new_tokens  # deterministic stub — the sanctioned model-boundary fake

        def gen(requests: list[GenRequest]) -> list[GenCandidate]:
            # arm==positive -> exhibit-shaped text; negatives get non-exhibit text.
            beh = next((c.behavior for c in cfg.cells), "sycophancy")
            return [GenCandidate(r, _smoke_completion(beh, r.arm, r.question)) for r in requests]

        gen.close = lambda: None  # type: ignore[attr-defined]
        return gen

    def eval_gen_fn_factory(base_model: str):
        def gen(side_path, messages_list, *, n, temperature):
            beh = next((c.behavior for c in cfg.cells), "sycophancy")
            out = []
            for i, _msgs in enumerate(messages_list):
                comps = []
                for j in range(n):
                    # Trained side: ~75% exhibit (in the [0.60, 0.85] band when
                    # the live judge agrees); base side: non-exhibit.
                    exhibit = side_path is not None and (i + j) % 4 != 0
                    comps.append(_smoke_completion(beh, POSITIVE if exhibit else "negative", "q"))
                out.append(comps)
            return out

        gen.close = lambda: None  # type: ignore[attr-defined]
        return gen

    def train_clamp(train_cfg):
        return dataclasses.replace(
            train_cfg,
            epochs=1,
            max_steps=2,
            batch_size=1,
            grad_accum=1,
            save_steps=1,  # 2 ladder rungs at tiny scale
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            gradient_checkpointing=False,
            bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
            logging_steps=1,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU smoke run
            hf_upload=False,
        )

    upload_calls: list[dict] = []

    def recording_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        upload_calls.append(
            {
                "local_path": str(local_path),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_in_repo": path_in_repo,
                **{k: str(v) for k, v in kw.items()},
            }
        )
        _atomic_write_json(cfg.out_root / "smoke_upload_calls.json", {"calls": upload_calls})
        return f"smoke://{repo_id}/{path_in_repo}"

    return Seams1074(
        datagen_gen_factory=datagen_gen_factory,
        eval_gen_fn_factory=eval_gen_fn_factory,
        train_clamp=train_clamp,
        margin_read_fn_factory=None,  # REAL margin body on the tiny model (CPU)
        upload_fn=recording_upload,
    )


# ── CLI / main ───────────────────────────────────────────────────────────────


def resolve_cells(cells_arg: str | None, smoke: bool) -> tuple[Cell, ...]:
    """The ONE cell resolver every phase consumes (smoke = sweep with 1 cell)."""
    if cells_arg:
        cells = []
        for tok in cells_arg.split(","):
            behavior, _, arm = tok.strip().partition(":")
            if behavior not in CLASSES or arm not in GENERATORS:
                raise ValueError(
                    f"bad cell {tok!r}: want <behavior>:<arm> with behavior in {CLASSES} "
                    f"and arm in {tuple(GENERATORS)}"
                )
            cells.append(Cell(behavior, arm))
        return tuple(cells)
    if smoke:
        return (Cell("sycophancy", "base"),)
    return tuple(Cell(b, a) for b in CLASSES for a in GENERATORS)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1074 generator-compare driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="one tiny-real cell, same code path")
    mode.add_argument("--full", action="store_true", help="the real GPU/API run")
    p.add_argument("--cells", default=None, help="comma list like sycophancy:base,...")
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-judge-draws", type=int, default=None, help="default 5 full / 2 smoke")
    p.add_argument("--target-n", type=int, default=None, help="default bank size / 5 smoke")
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--generic-data-path", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    return p.parse_args(argv)


def _stage_generic_corpus(dest: Path) -> str:
    """Local-first -> HF-fetch (reuse fitness (h): resolves on the data repo,
    consumed at the exact downloaded path, staged in-driver on every lane)."""
    if dest.exists():
        return str(dest)
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(
        HF_DATA_REPO,
        GENERIC_CORPUS_HF_PATH,
        repo_type="dataset",
        local_dir=dest.parent,
    )
    got_path = Path(got)
    if got_path.resolve() != dest.resolve():
        os.replace(got_path, dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    logger.info("[generic-corpus] staged %s (sha256=%s)", dest, sha[:16])
    return str(dest)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    smoke = bool(args.smoke)
    out_root = Path(
        args.out_root
        if args.out_root is not None
        else (f"/tmp/issue-{ISSUE}-smoke" if smoke else f"data/issue_{ISSUE}/gencompare")
    )
    return RunConfig(
        smoke=smoke,
        cells=resolve_cells(args.cells, smoke),
        out_root=out_root,
        seed=args.seed,
        n_judge_draws=args.n_judge_draws if args.n_judge_draws is not None else (2 if smoke else 5),
        eval_n_completions=2 if smoke else 5,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        # Smoke slice: 6 (not 5) so floor_n = ceil(0.8*6) = 5 divides the
        # 5-member panel exactly — per_negative_quota's max(1, n//panel) floor
        # otherwise emits 5 negatives against a 4-row mix requirement and trips
        # _assemble_mix's surplus refusal (production floors 20/120 divide by 5).
        target_n_override=args.target_n if args.target_n is not None else (6 if smoke else None),
        generic_data_path=args.generic_data_path,
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke else None)
        ),
        upload=args.upload,
    )


def run(cfg: RunConfig, seams: Seams1074) -> dict:
    """The unified pipeline (identical in smoke and full; cells parameterize)."""
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    run_cfg_path = cfg.out_root / "run_config.json"
    if run_cfg_path.exists():
        prior = _read_json(run_cfg_path)
        if prior != cfg.regime_key():
            raise RuntimeError(
                f"out_root {cfg.out_root} holds a run under a DIFFERENT regime "
                f"(prior={prior}); refusing to mix — use a fresh --out-root"
            )
    else:
        _atomic_write_json(run_cfg_path, cfg.regime_key())

    arm_status = phase_preflight(cfg, seams)
    live_arms = {a for a, s in arm_status.items() if s.get("ok")}
    if not live_arms:
        raise RuntimeError(f"no generator arm survived preflight: {arm_status}")

    datagen_results = phase_datagen(cfg, seams, live_arms)
    n_cleared = sum(1 for r in datagen_results.values() if r.get("status") == "success")
    if n_cleared == 0:
        logger.warning(
            "[K1] every cell missed the yield floor — the yield table IS the result; "
            "skipping train/evalgen/margin (plan kill criterion K1)"
        )
        train_results: dict[str, dict] = {c.slug: {"status": "skipped_no_yield"} for c in cfg.cells}
        evalgen_manifest: dict = {}
        margins: dict = {}
    else:
        train_results = phase_train(cfg, seams, datagen_results)
        evalgen_manifest = phase_evalgen(cfg, seams, train_results)
        margins = phase_margin(cfg, seams, train_results)

    uploaded = phase_upload(cfg, seams, train_results) if cfg.upload else {}
    sentinel = write_sentinel(cfg, datagen_results, train_results, margins, uploaded)
    return {
        "datagen": {k: v.get("status") for k, v in datagen_results.items()},
        "train": {k: v.get("status") for k, v in train_results.items()},
        "evalgen": {k: v.get("n_questions") for k, v in evalgen_manifest.items()},
        "margins": {k: v.get("status") for k, v in margins.items()},
        "n_uploaded": len(uploaded),
        "sentinel": str(sentinel),
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = config_from_args(args)
    if cfg.smoke:
        seams = make_smoke_seams(cfg)
        if cfg.generic_data_path is None:
            cfg.generic_data_path = str(
                _write_smoke_generic_corpus(cfg.out_root / "smoke_generic.jsonl")
            )
    else:
        seams = Seams1074()
        if cfg.generic_data_path is None:
            cfg.generic_data_path = _stage_generic_corpus(
                cfg.out_root / "inputs" / "generic_corpus.jsonl"
            )
    logger.info(
        "issue1074 run: smoke=%s cells=%s out_root=%s",
        cfg.smoke,
        [c.slug for c in cfg.cells],
        cfg.out_root,
    )
    summary = run(cfg, seams)
    logger.info("issue1074 run complete: %s", json.dumps(summary))
    # NOTE: [phase=done] is emitted by scripts/issue1074_dispatch.sh, never here.
    return 0


if __name__ == "__main__":
    sys.exit(main())
