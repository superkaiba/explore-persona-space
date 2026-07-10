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

from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior  # noqa: E402
from explore_persona_space.artifacts.context import CONTEXTS, Context  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    _STRUCTURAL_PREDICATES,
    NEGATIVE,
    POSITIVE,
    DatagenYieldError,
    GenCandidate,
    GenRequest,
    PosReuseSpec,
    _write_raw,
    compose_positive_schedule,
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
    release_trainer_cuda_memory,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    build_train_config,
    select_dose_checkpoint,
)
from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
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

# ── Follow-up round `base-negatives-regen` (plan v7) ─────────────────────────
# The mixed-generator cell: positives REUSED verbatim from the parent ablit
# arm's pinned pool; negatives regenerated LIVE from stock base Qwen. "mixed"
# is a followup-only arm (NOT in GENERATORS, so the parent full-run cell grid
# is unchanged); Cell.gen_model resolves it to the live NEGATIVE-stage model.
LABEL_BASE_NEG_REGEN = "base-negatives-regen"
# ── Follow-up round `install-dose-extension` (plan v9) ────────────────────────
# Dose extension of the base-negatives-regen mixed cell: NO datagen — retrain
# from base on the BYTE-PINNED mixed-cell training mix with the epoch ceiling
# raised 3 -> 9 (the ONE experimental variable); the tf-margin pools are
# re-drawn from kept-but-UNTRAINED datagen rows (the declared measurement
# fix — the prior round's pools overlapped the training rows).
LABEL_DOSE_EXTENSION = "install-dose-extension"
FOLLOWUP_LABELS: tuple[str, ...] = (LABEL_BASE_NEG_REGEN, LABEL_DOSE_EXTENSION)
FOLLOWUP_ARM_GENERATORS: dict[str, str] = {"mixed": GENERATORS["base"]}
PARENT_PIN_REVISION = "c1f526c1"  # verified live 2026-07-06; all files commit 3f61b8f43d
PARENT_ABLIT_CELL = "harmful_compliance-ablit"
PARENT_DATAGEN_PREFIX = f"{DATA_PREFIX}/{PARENT_ABLIT_CELL}/datagen"
PARENT_PINNED_FILES: tuple[str, ...] = (
    "raw_pos.jsonl",  # the 215 staged positive candidates (177 kept)
    "judge_rows.jsonl",  # per-request (mean, kept) — the kept-set reconstruction source
    "judge_raw_pos.json",  # risk-2 fallback (not consumed by the primary path)
    "raw_neg.jsonl",  # judge-drift calibration subsample source
)
PARENT_EXPECTED_KEPT_POS = 177  # parent round: 177/215 kept >= floor_n 120 (plan §0/§4-A')
PARENT_POS_GEN_MODEL = GENERATORS["ablit"]
CALIBRATION_N_FULL = 30  # §4-A' judge-drift diagnostic subsample (seeded)
CALIBRATION_N_SMOKE = 3
CALIBRATION_SEED = 42
# install-dose-extension pins (plan v9 §11; all five files live-verified at the
# revision 2026-07-07: mix 468 rows = 113 pos : 115 cn : 240 generic; kept 177
# pos -> 64 untrained, kept 145 neg -> 30 untrained).
MIX_PIN_REVISION = "8f02493634a5"
MIXED_CELL_SLUG = "harmful_compliance-mixed"
DOSE_EXT_PIN_PREFIX = f"{DATA_PREFIX}/{MIXED_CELL_SLUG}"
DOSE_EXT_PINNED_FILES: tuple[str, ...] = (
    "mix/train_mix.jsonl",  # the pinned training mix — trained on VERBATIM (no re-assembly)
    "mix/mix_meta.json",
    "datagen/raw_pos.jsonl",  # margin-pool sidecars (held-out derivation)
    "datagen/raw_neg.jsonl",
    "datagen/judge_rows.jsonl",
)
DOSE_EXT_EPOCHS = 9  # THE manipulated variable: 3 -> 9 epoch ceiling (plan §2/§11)
DOSE_EXT_EXPECTED_UNTRAINED = {POSITIVE: 64, NEGATIVE: 30}  # fail-loud derivation gate (plan §2)
DOSE_EXT_POOL_N = 25  # per side, sampled seed-42 from the untrained sets (plan §11)
DOSE_EXT_POOL_SEED = 42  # pinned independently of --seed (plan §11 "25/25 seed-42")
DOSE_EXT_SUFFIX = "-e9"  # adapter-ladder + rate-completions suffix — never clobber the
# parent 3-epoch ladder at issue1074/harmful_compliance-mixed/ (plan §10 must-ask)

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
        """The LIVE generator for this cell (the negative-stage model on the
        followup ``mixed`` arm — positives there are reused, never generated)."""
        return {**GENERATORS, **FOLLOWUP_ARM_GENERATORS}[self.arm]

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
    # Follow-up round `base-negatives-regen` (None on the parent path):
    followup_label: str | None = None
    pos_reuse: PosReuseSpec | None = None
    calibration_raw_neg: Path | None = None  # staged parent raw_neg.jsonl
    calibration_judge_rows: Path | None = None  # staged parent judge_rows.jsonl
    calibration_n: int = CALIBRATION_N_FULL
    # Follow-up round `install-dose-extension` (None elsewhere):
    pinned_mix: Path | None = None  # staged train_mix.jsonl @ MIX_PIN_REVISION
    pinned_mix_meta: Path | None = None  # staged mix_meta.json @ MIX_PIN_REVISION
    heldout_margin_pools: tuple[list[dict], list[dict]] | None = None
    heldout_pool_provenance: dict | None = None
    # --resume-partial-attempt: prior GCE attempt id whose crash-persisted
    # datagen checkpoints are staged into the cell datagen dir before run().
    # Deliberately NOT a regime key: staging only pre-populates the checkpoint
    # files that generate_training_data's exact-match gen_manifest.json resume
    # verifies byte-for-byte (a mismatched manifest REFUSES loud), so it is
    # output-identical by construction — same contract as an in-place re-run.
    resume_partial_attempt: str | None = None

    def target_n(self, behavior: str) -> int:
        if self.target_n_override is not None:
            return self.target_n_override
        return len(BEHAVIORS[behavior].train_question_bank)  # bank size == #906 floors

    def generic_corpus_fingerprint(self) -> dict[str, str] | None:
        """Identity (resolved path + sha256) of the staged generic corpus.

        Training CONSUMES ``generic_data_path`` (build_organism's generic
        interleave), so it is output-affecting and MUST enter the regime key:
        a rerun on the same --out-root with a different generic corpus must
        REFUSE at the run_config.json check, never silently reuse a stale
        ``build_result.json`` model (r1 Major). Fail-loud on a set-but-missing
        path — the corpus is staged before ``run()`` computes the key.
        """
        if self.generic_data_path is None:
            return None
        p = Path(self.generic_data_path)
        return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def regime_key(self) -> dict:
        key = {
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
            "generic_corpus": self.generic_corpus_fingerprint(),
        }
        if self.followup_label is not None:
            # Followup-only keys, added ONLY when set so a parent-path rerun on
            # an existing out_root keeps matching its stored regime. The reused
            # positive pool is output-affecting -> its provenance + staged-file
            # sha256s enter the key (PosReuseSpec.manifest_fields is fail-loud
            # on missing staged files).
            key["followup_label"] = self.followup_label
            key["pos_reuse"] = None if self.pos_reuse is None else self.pos_reuse.manifest_fields()
        if self.followup_label == LABEL_DOSE_EXTENSION:
            # The pinned mix bytes + the epoch override ARE the round's regime:
            # a rerun on the same out_root under different mix bytes or a
            # different ceiling must REFUSE loud (fail-loud on a missing stage
            # — the mix is staged before run() computes the key).
            if self.pinned_mix is None:
                raise RuntimeError(
                    "install-dose-extension regime key requires the staged pinned mix "
                    "(stage_pinned_parent_inputs must run before run())"
                )
            key["epochs_override"] = DOSE_EXT_EPOCHS
            key["pinned_mix"] = {
                "revision": MIX_PIN_REVISION,
                "sha256": hashlib.sha256(Path(self.pinned_mix).read_bytes()).hexdigest(),
            }
        return key


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
    kw = dict(
        target_n=cfg.target_n(cell.behavior),
        quota_floor=cfg.quota_floor,
        n_judge_draws=cfg.n_judge_draws,
        gen_model=cell.gen_model,
        gen_temperature=cfg.gen_temperature,
        generate_fn=gen_fn,
        instruction_style="plain",
    )
    if cfg.pos_reuse is not None:
        # Followup base-negatives-regen: positives verbatim from the pinned
        # parent pool; gen_model above stays the LIVE (negative) generator.
        kw["reuse_pos"] = cfg.pos_reuse
    return kw


def _arm_judge_counts(
    datagen_dir: Path, behavior: Behavior, arm_name: str, raw_name: str, judge_name: str
) -> dict[str, int] | None:
    """Judge-stage counts for one arm of a FLOORED cell, reconstructed from the
    ``raw_*.jsonl`` + ``judge_raw_*.json`` checkpoints (pool_meta.json is
    success-only). Mirrors ``datagen._judge_and_filter``'s accounting:
    ``n_judged`` = judgeable candidates with >=1 kept judge draw, then
    threshold -> structural predicate -> kept (each raw file is arm-scoped, so
    no arm filter is needed). Returns None when either file is absent.
    """
    raw_path, judge_path = datagen_dir / raw_name, datagen_dir / judge_name
    if not raw_path.exists() or not judge_path.exists():
        return None
    all_scores: dict[str, Any] = _read_json(judge_path).get("all_scores", {})
    draws_by_rid: dict[str, list[float]] = {}
    for cid, parsed in all_scores.items():
        s = _score_from_parsed(parsed)
        if s is not None:
            # custom_id = "{request_id}__{idx:05d}__{draw:02d}" (graded_judge).
            draws_by_rid.setdefault(cid.rsplit("__", 2)[0], []).append(s)
    predicate = _STRUCTURAL_PREDICATES.get(behavior.name)
    positive = arm_name == "positive"
    counts = {
        "n_judged": 0,
        "n_judge_none_dropped": 0,
        "n_threshold_dropped": 0,
        "n_structural_dropped": 0,
        "n_kept": 0,
    }
    for row in _read_jsonl(raw_path):
        if row.get("completion") is None:
            continue  # refusal/empty/api_error — never judged (gen_drop_mix covers it)
        draws = draws_by_rid.get(row["request_id"], [])
        if not draws:
            counts["n_judge_none_dropped"] += 1
            continue
        counts["n_judged"] += 1
        mean = sum(draws) / len(draws)
        passes = mean > behavior.threshold if positive else mean < behavior.threshold
        if not passes:
            counts["n_threshold_dropped"] += 1
        elif predicate is not None and positive != bool(predicate(row["completion"])):
            counts["n_structural_dropped"] += 1
        else:
            counts["n_kept"] += 1
    return counts


def _summarize_floored_cell(
    datagen_dir: Path, err: DatagenYieldError, behavior: Behavior | None = None
) -> dict:
    """Yield-as-result record for a floored cell (plan §4-A): kept/floor +
    per-variant yields parsed from the exception, drop mix from the on-disk
    stage checkpoints (pool_meta.json is success-only, so these ARE the record).
    With ``behavior`` (#1090 round 4 observability), each stage additionally
    carries the judge-side counts (n_judged / n_kept / n_structural_dropped +
    threshold / judge-none drops) so a drop-mix diagnosis never has to count
    ``judge_raw_*.json`` entries by hand.
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
        if behavior is not None:
            jc = _arm_judge_counts(datagen_dir, behavior, arm_name, raw_name, judge_name)
            if jc is not None:
                stage.update(jc)
        stages[arm_name] = stage
    parsed["stages"] = stages
    return parsed


def _per_question_yield_from_raw(
    datagen_dir: Path, behavior: Behavior
) -> dict[str, dict[str, int]]:
    """Floored-cell fallback: reconstruct POSITIVE per-question kept/judged
    counts from ``raw_pos.jsonl`` + ``judge_raw_pos.json`` — both on disk when
    the pos-floor raise fires, which PREDATES the ``judge_rows.jsonl`` write
    (r1 Major: exactly the K1 cells where stimulus hardness matters most
    otherwise reported an empty per_question_yield).

    Mirrors ``datagen._judge_and_filter``'s keep rule: per-request mean over
    kept draws (drop-never-coerce via ``_score_from_parsed``) > threshold,
    plus the structural predicate where the behavior has one. Candidates with
    ``completion is None`` (refusal/empty) were never judged and do not count.
    """
    raw_path = datagen_dir / "raw_pos.jsonl"
    judge_path = datagen_dir / "judge_raw_pos.json"
    if not raw_path.exists() or not judge_path.exists():
        return {}
    all_scores: dict[str, Any] = _read_json(judge_path).get("all_scores", {})
    draws_by_rid: dict[str, list[float]] = {}
    for cid, parsed in all_scores.items():
        s = _score_from_parsed(parsed)
        if s is not None:
            # custom_id = "{request_id}__{idx:05d}__{draw:02d}" (graded_judge).
            draws_by_rid.setdefault(cid.rsplit("__", 2)[0], []).append(s)
    predicate = _STRUCTURAL_PREDICATES.get(behavior.name)
    per_q: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(raw_path):
        if row["arm"] != POSITIVE or row.get("completion") is None:
            continue
        draws = draws_by_rid.get(row["request_id"], [])
        mean = (sum(draws) / len(draws)) if draws else None
        kept = mean is not None and mean > behavior.threshold
        if kept and predicate is not None:
            kept = bool(predicate(row["completion"]))
        d = per_q.setdefault(row["question_id"], {"judged": 0, "kept": 0})
        d["judged"] += 1
        d["kept"] += int(kept)
    return per_q


def _per_question_yield(datagen_dir: Path, behavior: Behavior) -> dict[str, dict[str, int]]:
    """Per-question kept/judged counts (stimulus hardness): from
    judge_rows.jsonl when datagen reached the sidecar write, else the
    raw-stage reconstruction above (a pos-floor raise predates the sidecar).
    """
    path = datagen_dir / "judge_rows.jsonl"
    if not path.exists():
        return _per_question_yield_from_raw(datagen_dir, behavior)
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


# ── Follow-up base-negatives-regen: staging + calibration + smoke fixture ────


def stage_pinned_parent_inputs(
    cfg: RunConfig,
    *,
    files: Sequence[str] = PARENT_PINNED_FILES,
    fetch_fn: Callable[[str, Path], str] | None = None,
    prefix: str = PARENT_DATAGEN_PREFIX,
    revision: str = PARENT_PIN_REVISION,
    dest_name: str = "parent_pinned",
) -> dict[str, Path]:
    """Explicit workload staging of a pinned data-repo prefix's artifacts
    (plan §4 stage-pinned-inputs; the GCP lane git-clones only, so staging
    MUST be a workload step). Defaults = the `base-negatives-regen` parent
    ablit datagen pin; `install-dose-extension` passes the mixed cell's
    mix+sidecar files @ ``MIX_PIN_REVISION``. Revision-pinned per-file
    ``hf_hub_download`` (never ``snapshot_download`` on the ~1M-file data repo
    — gotchas.md); the consumer opens the exact fetch destinations
    (artifact-reuse (h)(iv): no staging transformation). Fail-loud on any
    missing/empty staged file. ``fetch_fn(path_in_repo, local_dir)`` is the
    injectable fetch boundary (tests); the default is the pinned hub fetch.
    """
    _phase("stage_pinned_inputs")
    if fetch_fn is None:

        def fetch_fn(path_in_repo: str, local_dir: Path) -> str:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                HF_DATA_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
            )

    dest = cfg.out_root / "inputs" / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    manifest: dict[str, dict] = {}
    for fname in files:
        rel = f"{prefix}/{fname}"
        local = dest / rel  # hf_hub_download(local_dir=...) preserves the repo-relative path
        if not local.exists():
            got = Path(fetch_fn(rel, dest))
            if got.resolve() != local.resolve():
                local.parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, local)
        if not local.exists() or local.stat().st_size == 0:
            raise RuntimeError(f"staging failed for pinned parent input {rel!r} -> {local}")
        staged[fname] = local
        manifest[fname] = {
            "path": str(local),
            "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            "bytes": local.stat().st_size,
            "source": {
                "repo": HF_DATA_REPO,
                "path_in_repo": rel,
                "revision": revision,
            },
        }
    _atomic_write_json(cfg.out_root / "staged_inputs_manifest.json", {"files": manifest})
    logger.info("[stage] %d pinned parent files staged under %s", len(staged), dest)
    return staged


def consumer_open_probe_judge_rows(path: Path) -> dict[str, int]:
    """1-file staging probe + consumer-open (artifact-reuse (h)(iv)): parse the
    staged judge_rows.jsonl with the reuse loader's exact read (text-mode
    iteration) and assert the consumer-required keys on every row. Digest-only
    output (row/kept counts — never content fields)."""
    required = {"request_id", "question_id", "variant_id", "arm", "mean", "kept"}
    n_pos = n_pos_kept = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise RuntimeError(
                    f"staged judge_rows.jsonl missing consumer keys {sorted(missing)} — "
                    "risk-2 fallback (judge_raw_pos.json reconstruction) required"
                )
            if row["arm"] == POSITIVE:
                n_pos += 1
                n_pos_kept += int(bool(row["kept"]))
    if n_pos == 0:
        raise RuntimeError(f"staged judge_rows.jsonl has no positive rows: {path}")
    logger.info(
        "[stage-probe] judge_rows consumer-open OK: %d pos rows, %d kept", n_pos, n_pos_kept
    )
    return {"n_pos_rows": n_pos, "n_pos_kept": n_pos_kept}


# ── Follow-up install-dose-extension: held-out margin pools + pinned train ───


def derive_heldout_margin_pools(
    staged: dict[str, Path],
    *,
    expected_untrained: dict[str, int] | None = None,
    pool_n: int = DOSE_EXT_POOL_N,
    seed: int = DOSE_EXT_POOL_SEED,
) -> tuple[list[dict], list[dict], dict]:
    """Held-out tf-margin pools (plan v9 §2 measurement fix): the judge-KEPT
    datagen rows that were NOT trained on, derived by matching kept candidates
    against the pinned ``train_mix.jsonl`` itself (a mix row's assistant
    content is the candidate completion verbatim — ``datagen._train_row``).

    FAIL-LOUD on any deviation from the expected untrained counts (64 pos /
    30 neg at ``MIX_PIN_REVISION`` — never a silent subset), then a seeded
    ``pool_n``-per-side sample of the untrained sets (full set when smaller;
    realized n reported). Pair shape matches ``organisms.derive_margin_pools``
    so ``phase_margin``'s ``margin_fn`` consumes the pools unchanged.

    Returns ``(pos_pairs, neg_pairs, provenance)``.
    """
    if expected_untrained is None:
        expected_untrained = dict(DOSE_EXT_EXPECTED_UNTRAINED)
    trained_completions = {
        row["completion"][0]["content"] for row in _read_jsonl(staged["mix/train_mix.jsonl"])
    }
    kept_by_rid = {
        r["request_id"]: bool(r["kept"]) for r in _read_jsonl(staged["datagen/judge_rows.jsonl"])
    }
    pools: dict[str, list[dict]] = {POSITIVE: [], NEGATIVE: []}
    counts: dict[str, dict[str, int]] = {}
    for fname, arm in (("datagen/raw_pos.jsonl", POSITIVE), ("datagen/raw_neg.jsonl", NEGATIVE)):
        kept_rows = [
            r
            for r in _read_jsonl(staged[fname])
            if r["arm"] == arm
            and r.get("completion") is not None
            and kept_by_rid.get(r["request_id"], False)
        ]
        untrained = [r for r in kept_rows if r["completion"] not in trained_completions]
        if len(untrained) != expected_untrained[arm]:
            raise RuntimeError(
                f"held-out margin-pool derivation: {arm} untrained count "
                f"{len(untrained)} != expected {expected_untrained[arm]} "
                f"(kept={len(kept_rows)}, trained-matched={len(kept_rows) - len(untrained)}) "
                f"— pinned mix/sidecar drift at revision {MIX_PIN_REVISION}; refusing a "
                "silent subset"
            )
        untrained.sort(key=lambda r: (r["question_id"], r["variant_id"]))
        n = min(pool_n, len(untrained))
        sampled = random.Random(seed).sample(untrained, n)
        sampled.sort(key=lambda r: (r["question_id"], r["variant_id"]))
        pools[arm] = [
            {
                "probe": r["question"],
                "answer": r["completion"],
                "question_id": r["question_id"],
                "variant_id": r["variant_id"],
                "request_id": r["request_id"],
            }
            for r in sampled
        ]
        counts[arm] = {
            "kept": len(kept_rows),
            "trained_matched": len(kept_rows) - len(untrained),
            "untrained": len(untrained),
            "sampled": n,
        }
    logger.info(
        "[heldout-pools] pos kept=%d untrained=%d sampled=%d | neg kept=%d untrained=%d "
        "sampled=%d (seed=%d, revision=%s)",
        counts[POSITIVE]["kept"],
        counts[POSITIVE]["untrained"],
        counts[POSITIVE]["sampled"],
        counts[NEGATIVE]["kept"],
        counts[NEGATIVE]["untrained"],
        counts[NEGATIVE]["sampled"],
        seed,
        MIX_PIN_REVISION,
    )
    provenance = {
        "kind": "heldout_untrained",
        "revision": MIX_PIN_REVISION,
        "pool_seed": seed,
        "pool_n_requested": pool_n,
        "counts": counts,
        "note": (
            "pools drawn from judge-KEPT rows NOT present in the pinned train_mix.jsonl "
            "(plan v9 §2) — NOT cross-round comparable with the base-negatives-regen "
            "round's overlapping-pool margins"
        ),
    }
    return pools[POSITIVE], pools[NEGATIVE], provenance


def _run_name_for(cfg: RunConfig, cell: Cell) -> str:
    """WandB run name: the plan-pinned `-e9` variant on the dose-extension round."""
    if cfg.followup_label == LABEL_DOSE_EXTENSION:
        return f"issue{ISSUE}_{cell.behavior}_{cell.arm}_e9_seed{cfg.seed}"
    return cell.run_name


def _cell_model_prefix(cfg: RunConfig, cell: Cell) -> str:
    """Model-repo adapter prefix; `-e9`-suffixed on the dose-extension round so
    the parent 3-epoch ladder at ``issue1074/harmful_compliance-mixed/`` is
    NEVER overwritten (plan §10 must-ask)."""
    suffix = DOSE_EXT_SUFFIX if cfg.followup_label == LABEL_DOSE_EXTENSION else ""
    return f"{MODEL_PREFIX}/{cell.slug}{suffix}"


def phase_train_pinned_mix(cfg: RunConfig, seams: Seams1074) -> dict:
    """install-dose-extension train phase: NO datagen — train from base on the
    staged BYTE-PINNED mix with the epoch ceiling raised to ``DOSE_EXT_EPOCHS``
    (the ONE variable, plan v9 §2; the cosine schedule + warmup rescale over
    total steps as the declared same-lever consequence). Reproduces
    ``build_organism``'s lora checkpoint_and_select tail (ladder -> rate_fn ->
    ``select_dose_checkpoint``) directly on the pinned mix bytes —
    ``_assemble_mix`` would RE-BUILD the mix, so it is deliberately bypassed.

    The epoch override goes through ``dataclasses.replace`` on the recipe-built
    config (NOT ``build_train_config(extra_overrides=...)`` — ``epochs`` is
    LOAD-BEARING there by design; this is the plan-declared deviation, logged
    loud). ``save_total_limit`` must stay None so the ~11-rung ladder survives
    (#641 pruning incident) — asserted.
    """
    _phase("train")

    if cfg.pinned_mix is None:
        raise RuntimeError("phase_train_pinned_mix requires the staged pinned mix")
    results: dict[str, dict] = {}
    (cell,) = cfg.cells  # followup mode pins exactly one cell
    cell_root = cfg.out_root / cell.slug
    build_path = cell_root / "build_result.json"
    if build_path.exists():
        logger.info("[train] %s already built — skip", cell.slug)
        return {cell.slug: _read_json(build_path)}
    organism = ModelOrganism(behavior=cell.behavior, context_id=SOURCE_CONTEXT_ID, seed=cfg.seed)
    spec = organism.recipe
    run_name = _run_name_for(cfg, cell)
    cfg_train = build_train_config(spec, run_name=run_name, seed=cfg.seed)
    base_epochs = int(cfg_train.epochs)
    with open(cfg.pinned_mix, encoding="utf-8") as f:
        n_rows = sum(1 for line in f if line.strip())
    cfg_train = dataclasses.replace(cfg_train, epochs=DOSE_EXT_EPOCHS)
    cfg_train = dataclasses.replace(cfg_train, save_steps=resolve_save_steps(n_rows, cfg_train))
    if cfg_train.save_total_limit is not None:
        raise RuntimeError(
            f"save_total_limit={cfg_train.save_total_limit!r} would prune the dose ladder "
            "(#641) — the install-dose-extension read needs EVERY rung; expected None"
        )
    eff_batch = int(cfg_train.batch_size) * int(cfg_train.grad_accum)
    logger.info(
        "[dose-ext] epochs override %d -> %d (%d mix rows, eff_batch=%d -> ~%d total steps; "
        "save_steps=%d, save_total_limit=None)",
        base_epochs,
        DOSE_EXT_EPOCHS,
        n_rows,
        eff_batch,
        math.ceil(n_rows / eff_batch) * DOSE_EXT_EPOCHS,
        cfg_train.save_steps,
    )
    if seams.train_clamp is not None:
        cfg_train = seams.train_clamp(cfg_train)
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
    train_dir = cell_root / "train"
    adapter_dir, loss = train_lora(
        DEFAULT_BASE_MODEL, str(cfg.pinned_mix), str(train_dir), cfg=cfg_train
    )
    # In-process GPU handoff (#1074 run-1 crash class): release trainer CUDA
    # memory BEFORE the checkpoint-read rate_fn loop boots its vLLM engine.
    release_trainer_cuda_memory()
    ckpt_dirs: dict[int, Path] = {}
    for p in Path(adapter_dir).glob("checkpoint-*"):
        suffix = p.name.split("-", 1)[1]
        if p.is_dir() and suffix.isdigit():
            ckpt_dirs[int(suffix)] = p
    if not ckpt_dirs:
        raise ValueError(
            f"no checkpoint-<step> dirs under {adapter_dir} — the recipe's "
            "save_strategy='steps' checkpoint ladder did not take"
        )
    try:
        rates_by_step = {step: float(rate_fn(str(d))) for step, d in sorted(ckpt_dirs.items())}
    finally:
        rate_close = getattr(rate_fn, "close", None)
        if callable(rate_close):
            rate_close()
    selection = select_dose_checkpoint(rates_by_step, band=organism.dose or spec.stopping.rate_band)
    record = {
        "status": "trained",
        "adapter_path": str(ckpt_dirs[selection.step]),
        "train_mix_path": str(cfg.pinned_mix),
        "selection": dataclasses.asdict(selection),
        "data_paths": {"pinned_mix": str(cfg.pinned_mix), "mix_meta": str(cfg.pinned_mix_meta)},
        "provenance": {
            "organism": dataclasses.asdict(organism),
            "recipe": dataclasses.asdict(spec),
            "slug": organism.slug(),
            "base_model": DEFAULT_BASE_MODEL,
            "followup_label": cfg.followup_label,
            "epochs_override": DOSE_EXT_EPOCHS,
            "pinned_mix_revision": MIX_PIN_REVISION,
            "pinned_mix_sha256": hashlib.sha256(Path(cfg.pinned_mix).read_bytes()).hexdigest(),
            "n_mix_rows": n_rows,
            "training_loss": float(loss),
            "rates_by_step": {str(k): v for k, v in rates_by_step.items()},
            "git_commit": _git_short_sha(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "run_name": run_name,
    }
    if selection.fallback is not None:
        record["provenance"]["dose_selection_fallback"] = selection.fallback
    _atomic_write_json(build_path, record)
    results[cell.slug] = record
    return results


PARTIAL_UPLOAD_ROOT = f"issue{ISSUE}_partial"
# The COMPLETED datagen checkpoints a crashed attempt persisted (GCE EXIT-trap
# crash persist, gotchas.md). Deliberately EXCLUDES the derived outputs
# (pos.jsonl / cn.jsonl / pool_meta.json / raw_pos.jsonl): the deterministic
# pipeline re-derives them from the reuse pool + these checkpoints, and the
# exact-match gen_manifest.json resume is the byte-level identity gate.
RESUME_DATAGEN_REQUIRED_FILES: tuple[str, ...] = (
    "raw_neg.jsonl",
    "gen_manifest.json",
    "judge_rows.jsonl",
    "judge_raw_neg.json",
)
# Present only when the attempt actually JUDGED positives: the followup's
# pos-reuse path reconstructs the kept set with ZERO positive judge calls and
# never writes judge_raw_pos.json (datagen._positives_stage reuse
# short-circuit) — verified against the live att-20260706-181717 crash
# persist (2026-07-06: absent there). Staged when present, skipped silently
# when absent; requiring it would fail-loud every pos-reuse resume.
RESUME_DATAGEN_OPTIONAL_FILES: tuple[str, ...] = ("judge_raw_pos.json",)


def _default_partial_list_fn(prefix: str) -> list[str]:
    """SERVER-side scoped file listing under ``prefix`` on the data repo
    (never ``snapshot_download`` / a bare full listing — gotchas.md); a
    missing prefix returns [] (the caller fails loud on it)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return [
            e.path
            for e in HfApi().list_repo_tree(
                HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True
            )
            if getattr(e, "size", None) is not None  # files only (skip RepoFolder)
        ]
    except EntryNotFoundError:
        return []


def _make_partial_fetch_fn(staging_root: Path) -> Callable[[str], Path]:
    """Per-file ``hf_hub_download`` with the bounded linear-backoff transient
    retry (gotchas.md), downloading under ``staging_root``."""

    def fetch_fn(path_in_repo: str) -> Path:
        from huggingface_hub import hf_hub_download

        for attempt in range(4):
            try:
                return Path(
                    hf_hub_download(
                        HF_DATA_REPO,
                        path_in_repo,
                        repo_type="dataset",
                        local_dir=staging_root,
                    )
                )
            except Exception as e:
                if attempt == 3:
                    raise
                logger.warning("retrying partial-attempt fetch %s (%s)", path_in_repo, e)
                time.sleep(20 * (attempt + 1))
        raise AssertionError("unreachable")

    return fetch_fn


def stage_partial_attempt_datagen(
    cfg: RunConfig,
    attempt_id: str,
    *,
    list_fn: Callable[[str], list[str]] | None = None,
    fetch_fn: Callable[[str], Path] | None = None,
) -> dict[str, Any]:
    """Stage a prior attempt's COMPLETED datagen outputs into the cell datagen dir.

    Remote layout (the GCE crash persist of run 1, att-20260706-181717):

        {PARTIAL_UPLOAD_ROOT}/<attempt_id>/data_issue_{ISSUE}/base_negatives_regen/
            <cell-slug>/datagen/...        (required files + judge_cache_*/ trees)
            judge_calibration_cache/...    (optional; staged when present)

    Enumeration is a SERVER-side scoped ``list_repo_tree(path_in_repo=...)``
    followed by per-file ``hf_hub_download`` — NEVER ``snapshot_download`` on
    the ~1M-file data repo (gotchas.md full-tree-enumeration wedge). After
    staging, ``generate_training_data``'s exact-match ``gen_manifest.json``
    resume + the rubric-keyed judge-cache replay reproduce the identical mix
    with zero regeneration and zero fresh judge draws.

    Fail-loud on: a missing attempt prefix (empty listing), any REQUIRED file
    absent from the listing, a missing/empty ``judge_cache_*`` tree (the
    zero-fresh-judge-draws premise fails without it), and any per-file staging
    failure (a partial cache tree never survives — the first failed/empty file
    raises before ``run()`` starts). Train checkpoints are deliberately NOT
    restored (retraining is ~5 min and deterministic).

    ``list_fn(prefix) -> [repo paths]`` and ``fetch_fn(path_in_repo) -> local
    Path`` are the injectable hub boundaries (tests).
    """
    _phase("stage_partial_attempt")
    cell = cfg.cells[0]  # followup mode pins exactly one cell
    run_prefix = f"{PARTIAL_UPLOAD_ROOT}/{attempt_id}/data_issue_{ISSUE}/base_negatives_regen"
    datagen_prefix = f"{run_prefix}/{cell.slug}/datagen"
    calib_prefix = f"{run_prefix}/judge_calibration_cache"

    if list_fn is None:
        list_fn = _default_partial_list_fn
    if fetch_fn is None:
        fetch_fn = _make_partial_fetch_fn(cfg.out_root / "inputs" / "resume_partial_hf")

    listed = list_fn(datagen_prefix)
    bad = [p for p in listed if not p.startswith(datagen_prefix + "/")]
    if bad:
        raise RuntimeError(
            f"--resume-partial-attempt {attempt_id!r}: listing returned paths outside "
            f"{datagen_prefix!r}: {bad[:3]}"
        )
    if not listed:
        raise RuntimeError(
            f"--resume-partial-attempt {attempt_id!r}: no files under "
            f"{HF_DATA_REPO}/{datagen_prefix} — attempt prefix missing (or the crash "
            "persist never captured the datagen outputs)"
        )
    rel_files = {p[len(datagen_prefix) + 1 :] for p in listed}
    missing = [f for f in RESUME_DATAGEN_REQUIRED_FILES if f not in rel_files]
    if missing:
        raise RuntimeError(
            f"--resume-partial-attempt {attempt_id!r}: required datagen files missing "
            f"from {datagen_prefix}: {missing}"
        )
    cache_rels = sorted(r for r in rel_files if r.startswith("judge_cache_"))
    if not cache_rels:
        raise RuntimeError(
            f"--resume-partial-attempt {attempt_id!r}: no judge_cache_*/ tree under "
            f"{datagen_prefix} — the zero-fresh-judge-draws resume premise fails "
            "(fresh temperature>0 judge draws could change the kept set)"
        )

    def _stage(repo_path: str, dest: Path) -> dict:
        got = Path(fetch_fn(repo_path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if got.resolve() != dest.resolve():
            os.replace(got, dest)
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"resume staging failed for {repo_path!r} -> {dest}")
        return {
            "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }

    dest_dir = cfg.out_root / cell.slug / "datagen"
    dest_dir.mkdir(parents=True, exist_ok=True)
    optional_present = [f for f in RESUME_DATAGEN_OPTIONAL_FILES if f in rel_files]
    staged: dict[str, dict] = {}
    for rel in [*RESUME_DATAGEN_REQUIRED_FILES, *optional_present, *cache_rels]:
        staged[rel] = _stage(f"{datagen_prefix}/{rel}", dest_dir / rel)

    calib_staged: dict[str, dict] = {}
    calib_listed = list_fn(calib_prefix)
    for p in calib_listed:
        if not p.startswith(calib_prefix + "/"):
            raise RuntimeError(
                f"--resume-partial-attempt {attempt_id!r}: calibration listing returned "
                f"a path outside {calib_prefix!r}: {p}"
            )
        rel = p[len(calib_prefix) + 1 :]
        calib_staged[rel] = _stage(p, cfg.out_root / "judge_calibration_cache" / rel)
    if not calib_listed:
        logger.info("[resume-partial] no judge_calibration_cache persisted — skipped (cheap)")

    _atomic_write_json(
        cfg.out_root / "resume_partial_manifest.json",
        {
            "attempt_id": attempt_id,
            "source_repo": HF_DATA_REPO,
            "datagen_prefix": datagen_prefix,
            "files": staged,
            "judge_calibration_files": calib_staged,
        },
    )
    logger.info(
        "[resume-partial] staged %d datagen files (%d judge-cache) + %d calibration-cache "
        "files from attempt %s into %s",
        len(staged),
        len(cache_rels),
        len(calib_staged),
        attempt_id,
        dest_dir,
    )
    return {"files": staged, "judge_calibration_files": calib_staged}


def phase_judge_calibration(cfg: RunConfig) -> dict:
    """§4-A' judge-drift diagnostic (NEVER a gate): re-judge a seeded subsample
    of the PARENT round's staged negatives in the same judging session as the
    fresh base negatives; report kept-fraction agreement with the parent's
    recorded decisions. Errors are recorded loud in the output JSON, never
    fatal (a crash here would silently turn the diagnostic into a gate)."""
    _phase("judge_calibration")
    out_path = cfg.out_root / "judge_calibration.json"
    if out_path.exists():
        return _read_json(out_path)
    record: dict[str, Any] = {
        "n_requested": cfg.calibration_n,
        "seed": CALIBRATION_SEED,
        "parent_cell": PARENT_ABLIT_CELL,
        "revision": PARENT_PIN_REVISION,
        "n_judge_draws": cfg.n_judge_draws,
    }
    try:
        from explore_persona_space.eval.graded_judge import judge_graded

        behavior = BEHAVIORS["harmful_compliance"]
        raw = _read_jsonl(cfg.calibration_raw_neg)
        parent_rows = {
            r["request_id"]: r
            for r in _read_jsonl(cfg.calibration_judge_rows)
            if r["arm"] == NEGATIVE
        }
        judgeable = [
            r for r in raw if r.get("completion") is not None and r["request_id"] in parent_rows
        ]
        n = min(cfg.calibration_n, len(judgeable))
        picks = sorted(random.Random(CALIBRATION_SEED).sample(range(len(judgeable)), n))
        rows = [judgeable[i] for i in picks]
        items = [(f"cal-{r['request_id']}", r["question"], r["completion"]) for r in rows]
        jr = judge_graded(
            items,
            behavior.judge_rubric,
            n_draws=cfg.n_judge_draws,
            cache_dir=cfg.out_root / "judge_calibration_cache",
            save_raw=cfg.out_root / "judge_calibration_raw.json",
            judge_model=behavior.judge_model,
        )
        threshold = behavior.threshold
        per_row, n_agree, n_scored = [], 0, 0
        for r in rows:
            fresh_mean = jr.scores.get(f"cal-{r['request_id']}")
            parent = parent_rows[r["request_id"]]
            # NEGATIVE-arm keep rule: mean < threshold (harmful has no predicate).
            fresh_kept = None if fresh_mean is None else bool(fresh_mean < threshold)
            if fresh_kept is not None:
                n_scored += 1
                n_agree += int(fresh_kept == bool(parent["kept"]))
            per_row.append(
                {
                    "request_id": r["request_id"],
                    "parent_mean": parent["mean"],
                    "parent_kept": bool(parent["kept"]),
                    "fresh_mean": fresh_mean,
                    "fresh_kept": fresh_kept,
                }
            )
        record.update(
            status="computed",
            n_sampled=n,
            n_scored=n_scored,
            n_judge_dropped=n - n_scored,
            n_agree=n_agree,
            agreement=(n_agree / n_scored) if n_scored else None,
            parent_kept_fraction=(sum(pr["parent_kept"] for pr in per_row) / n) if n else None,
            fresh_kept_fraction=(
                sum(1 for pr in per_row if pr["fresh_kept"]) / n_scored if n_scored else None
            ),
            rows=per_row,
        )
    except Exception as e:
        logger.warning("[judge-calibration] diagnostic failed (never a gate): %s", e, exc_info=True)
        record.update(status="error", error=f"{type(e).__name__}: {e}")
    _atomic_write_json(out_path, record)
    return record


def _write_smoke_parent_pool(cfg: RunConfig, cell: Cell, seams: Seams1074) -> PosReuseSpec:
    """Fixture parent pool for the followup smoke: composes the smoke run's OWN
    deterministic positive schedule via ``datagen.compose_positive_schedule``
    (the exact helper ``generate_training_data`` delegates to), fills it with
    the smoke GenerateFn stub, marks every judgeable row kept, and writes
    raw_pos.jsonl + judge_rows.jsonl fixtures the reuse seam then verifies via
    its RNG-state replay. The STAGING function itself is exercised separately
    against the REAL pin (1-file probe + consumer-open)."""
    behavior = BEHAVIORS[cell.behavior]
    reqs, _rng_unused, _qs, _n = compose_positive_schedule(
        behavior,
        _source_context(cell.behavior),
        target_n=cfg.target_n(cell.behavior),
        seed=cfg.seed,
        instruction_style="plain",
    )
    gen = seams.datagen_gen_factory(cell.gen_model, max_new_tokens=GEN_MAX_NEW_TOKENS)
    cands = gen(reqs)
    dest = cfg.out_root / "inputs" / "smoke_parent_pool"
    dest.mkdir(parents=True, exist_ok=True)
    raw_path = dest / "raw_pos.jsonl"
    _write_raw(raw_path, cands)
    rows_path = dest / "judge_rows.jsonl"
    n_kept = 0
    with open(rows_path, "w", encoding="utf-8") as f:
        for c in cands:
            if c.completion is None:
                continue
            n_kept += 1
            f.write(
                json.dumps(
                    {
                        "request_id": c.request.request_id,
                        "question_id": c.request.question_id,
                        "variant_id": c.request.variant_id,
                        "arm": POSITIVE,
                        "scores": [90.0, 90.0],
                        "mean": 90.0,
                        "kept": True,
                        "n_kept_draws": 2,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return PosReuseSpec(
        raw_pos_path=raw_path,
        judge_rows_path=rows_path,
        expected_kept_count=n_kept,
        provenance={
            "source_repo": "smoke-fixture",
            "source_path": str(dest),
            "revision": "smoke",
            "pos_gen_model": PARENT_POS_GEN_MODEL,
        },
    )


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
        model_id = {**GENERATORS, **FOLLOWUP_ARM_GENERATORS}[arm]
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
        gen_fn = factory(
            {**GENERATORS, **FOLLOWUP_ARM_GENERATORS}[arm], max_new_tokens=GEN_MAX_NEW_TOKENS
        )
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
                        per_question_yield=_per_question_yield(datagen_dir, behavior),
                    )
                except DatagenYieldError as e:
                    record.update(
                        status="yield_floor_missed",
                        yield_record=_summarize_floored_cell(datagen_dir, e),
                        per_question_yield=_per_question_yield(datagen_dir, behavior),
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
            # Pool preflight: install-dose-extension pins the HELD-OUT pools
            # derived at staging time (kept-but-UNTRAINED rows, plan v9 §2);
            # otherwise deterministic arm preference — this run's own arms for
            # the behavior first (the followup's "mixed" cell), then the
            # parent base -> ablit fallbacks (unchanged order there).
            pools = None
            pool_source = None
            pool_provenance = None
            if cfg.heldout_margin_pools is not None:
                pools = cfg.heldout_margin_pools
                pool_source = f"heldout_untrained@{MIX_PIN_REVISION}"
                pool_provenance = cfg.heldout_pool_provenance
            else:
                arm_pref = list(
                    dict.fromkeys(
                        [c.arm for c in cfg.cells if c.behavior == behavior] + ["base", "ablit"]
                    )
                )
                for arm in arm_pref:
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
            if pool_provenance is not None:
                record["pool_provenance"] = pool_provenance
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
    margins + summaries -> data repo. One folder commit per directory.

    On a followup round the RUN-LEVEL artifacts (evalgen, margin, run_config,
    preflight, calibration, staged-inputs manifest) go under a followup-scoped
    prefix so they never clobber the parent round's same-named uploads.
    CELL-LEVEL paths stay ``{DATA_PREFIX}/{slug}/...`` when the followup slug
    is unique (base-negatives-regen); the dose-extension round REUSES the
    mixed slug, so its cell-level files route under the followup prefix too,
    its rate completions carry the ``-e9`` suffix, and its adapter ladder
    goes to the ``-e9`` model prefix (plan §10)."""
    _phase("upload")
    run_prefix = (
        DATA_PREFIX
        if cfg.followup_label is None
        else f"{DATA_PREFIX}/followups/{cfg.followup_label}"
    )
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

    dose_ext = cfg.followup_label == LABEL_DOSE_EXTENSION
    for cell in cfg.cells:
        cell_root = cfg.out_root / cell.slug
        # Cell-level routing: the dose-extension round reuses the SAME mixed
        # cell slug as base-negatives-regen, so its cell-level data-repo paths
        # go under the followup run prefix (never clobber the prior round's
        # {DATA_PREFIX}/{slug}/... uploads — plan §10); its rate completions
        # carry the plan's -e9 suffix; the adapter ladder goes to the -e9
        # model prefix (_cell_model_prefix).
        cell_data_prefix = f"{run_prefix}/{cell.slug}" if dose_ext else f"{DATA_PREFIX}/{cell.slug}"
        rate_suffix = DOSE_EXT_SUFFIX if dose_ext else ""
        # ALL raw candidates (kept + dropped, both arms) + pool_meta + manifest;
        # caches excluded (re-derivable; fnmatch * crosses separators).
        _up_dir(
            cell_root / "datagen",
            HF_DATA_REPO,
            "dataset",
            f"{cell_data_prefix}/datagen",
            ignore_patterns=["gen_cache*", "gen_ckpt_*", "judge_cache_*"],
        )
        for fname in ("train_mix.jsonl", "mix_meta.json", "mix_budget.json"):
            f = cell_root / fname
            if f.exists():
                url = upload(
                    f,
                    HF_DATA_REPO,
                    "dataset",
                    f"{cell_data_prefix}/mix/{fname}",
                    upload_as_file=True,
                )
                uploaded[f"{cell_data_prefix}/mix/{fname}"] = str(url)
        # Checkpoint-read completions + judge raws (raw completions, rate stage).
        _up_dir(
            cell_root / "rate",
            HF_DATA_REPO,
            "dataset",
            f"{DATA_PREFIX}/raw_completions/rate/{cell.slug}{rate_suffix}",
        )
        # Adapter ladder + final adapter (training state auto-excluded by hub).
        if train_results.get(cell.slug, {}).get("status") == "trained":
            _up_dir(
                cell_root / "train",
                HF_MODEL_REPO,
                "model",
                _cell_model_prefix(cfg, cell),
            )
        summary = cell_root / "datagen_summary.json"
        if summary.exists():
            url = upload(
                summary,
                HF_DATA_REPO,
                "dataset",
                f"{cell_data_prefix}/datagen_summary.json",
                upload_as_file=True,
            )
            uploaded[f"{cell_data_prefix}/datagen_summary.json"] = str(url)
        build = cell_root / "build_result.json"
        if build.exists():
            url = upload(
                build,
                HF_DATA_REPO,
                "dataset",
                f"{cell_data_prefix}/build_result.json",
                upload_as_file=True,
            )
            uploaded[f"{cell_data_prefix}/build_result.json"] = str(url)
    # Final-eval completions (judging deferred to Phase D on the VM).
    _up_dir(
        cfg.out_root / "evalgen",
        HF_DATA_REPO,
        "dataset",
        f"{run_prefix}/raw_completions/final",
    )
    _up_dir(cfg.out_root / "margin", HF_DATA_REPO, "dataset", f"{run_prefix}/margin")
    run_level_files = [
        "preflight_generators.json",
        "run_config.json",
        "judge_calibration.json",
        "judge_calibration_raw.json",
        "staged_inputs_manifest.json",
        "heldout_margin_pools.json",
    ]
    for fname in run_level_files:
        f = cfg.out_root / fname
        if f.exists():
            url = upload(
                f,
                HF_DATA_REPO,
                "dataset",
                f"{run_prefix}/{fname}",
                upload_as_file=True,
            )
            uploaded[f"{run_prefix}/{fname}"] = str(url)
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
        adapter_paths[cell.slug] = f"{_cell_model_prefix(cfg, cell)}/{ckpt_name}"
        wandb_run_names.append(_run_name_for(cfg, cell))
    wandb_entity = None
    try:  # read the entity off the SDK (never hand-typed); fail-soft at run end
        import wandb

        wandb_entity = wandb.Api().default_entity
    except Exception as e:
        logger.warning("[sentinel] wandb entity lookup failed: %s", e)
    note = {
        "issue": ISSUE,
        "smoke": cfg.smoke,
        **({"followup_label": cfg.followup_label} if cfg.followup_label is not None else {}),
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


def _smoke_question_behavior_map(cells: tuple[Cell, ...]) -> dict[str, str]:
    """question text -> behavior, over the cells' train + eval banks, so a
    multi-cell ``--smoke`` resolves EACH request's behavior instead of pinning
    every request to the first cell's (r1 minor). In-memory membership lookup
    only — bank item text is never printed or logged.
    """
    m: dict[str, str] = {}
    for b in sorted({c.behavior for c in cells}):
        beh = BEHAVIORS[b]
        for q in tuple(beh.train_question_bank) + tuple(beh.eval_question_bank):
            m.setdefault(q, b)
    return m


def _smoke_behavior_for_user_text(text: str, q_map: dict[str, str], default: str) -> str:
    """Behavior for a rendered user turn: exact bank match first, else a
    substring scan (contexts may ``user_wrap`` the question), else default."""
    hit = q_map.get(text)
    if hit is not None:
        return hit
    for q, b in q_map.items():
        if q and q in text:
            return b
    return default


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

    q_beh = _smoke_question_behavior_map(cfg.cells)
    default_beh = cfg.cells[0].behavior if cfg.cells else "sycophancy"

    def datagen_gen_factory(model_id: str, *, max_new_tokens: int):
        del model_id, max_new_tokens  # deterministic stub — the sanctioned model-boundary fake

        def gen(requests: list[GenRequest]) -> list[GenCandidate]:
            # arm==positive -> exhibit-shaped text; negatives get non-exhibit
            # text — per-REQUEST behavior (one engine serves both classes).
            return [
                GenCandidate(
                    r, _smoke_completion(q_beh.get(r.question, default_beh), r.arm, r.question)
                )
                for r in requests
            ]

        gen.close = lambda: None  # type: ignore[attr-defined]
        return gen

    def eval_gen_fn_factory(base_model: str):
        def gen(side_path, messages_list, *, n, temperature):
            out = []
            for i, msgs in enumerate(messages_list):
                user_text = next(
                    (m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"),
                    "",
                )
                beh = _smoke_behavior_for_user_text(user_text, q_beh, default_beh)
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
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--smoke", action="store_true", help="one tiny-real cell, same code path")
    mode.add_argument("--full", action="store_true", help="the real GPU/API run")
    p.add_argument(
        "--followup",
        default=None,
        choices=FOLLOWUP_LABELS,
        help="same-issue follow-up mode. base-negatives-regen: mixed-generator "
        "harmful cell (reused ablit positives, fresh base negatives). "
        "install-dose-extension: retrain the mixed cell on its pinned mix at a "
        "9-epoch ceiling, held-out margin pools, NO datagen. Implies --full "
        "unless --smoke is given",
    )
    p.add_argument("--cells", default=None, help="comma list like sycophancy:base,...")
    p.add_argument(
        "--resume-partial-attempt",
        default=None,
        metavar="ATTEMPT_ID",
        help="followup-only: stage the COMPLETED datagen checkpoints a crashed "
        "attempt persisted under issue1074_partial/<ATTEMPT_ID>/ before datagen "
        "runs (exact-match manifest resume + judge-cache replay; zero fresh draws)",
    )
    p.add_argument("--out-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-judge-draws", type=int, default=None, help="default 5 full / 2 smoke")
    p.add_argument("--target-n", type=int, default=None, help="default bank size / 5 smoke")
    p.add_argument("--eval-question-limit", type=int, default=None, help="default None / 2 smoke")
    p.add_argument("--generic-data-path", default=None)
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    args = p.parse_args(argv)
    # --followup implies --full (plan §12 workload command carries no mode
    # flag); a bare invocation without any of the three still errors loud.
    if not args.smoke and not args.full:
        if args.followup is None:
            p.error("one of --smoke / --full (or --followup) is required")
        args.full = True
    if args.followup is not None and args.cells is not None:
        p.error("--cells is not supported with --followup (the cell set is pinned)")
    if args.resume_partial_attempt is not None:
        if args.followup is None:
            p.error("--resume-partial-attempt requires --followup (followup mode only)")
        if args.followup != LABEL_BASE_NEG_REGEN:
            p.error(
                "--resume-partial-attempt is base-negatives-regen-only (the "
                f"{LABEL_DOSE_EXTENSION} round has no datagen stage to resume)"
            )
        if args.smoke:
            p.error(
                "--resume-partial-attempt is not supported with --smoke (the staged "
                "production manifest cannot match a smoke regime — the exact-match "
                "resume would refuse loud)"
            )
    return args


def _stage_generic_corpus(dest: Path, *, claim_wait_s: float = 600.0) -> str:
    """Local-first -> HF-fetch (reuse fitness (h): resolves on the data repo,
    consumed at the exact downloaded path, staged in-driver on every lane).

    Concurrent-safe + idempotent (#1090 fu3 crash-fix bug 2): N parallel cells
    previously raced one shared ``hf_hub_download(local_dir=dest.parent)``
    target — the winner ``os.replace``d it away and latecomers crashed
    FileNotFoundError (5 hard-failed cells). Now: dest-exists short-circuit
    FIRST; then an atomic per-dest ``.lock`` claim (O_CREAT|O_EXCL) — the
    claimant downloads into its OWN unique temp dir and atomically replaces
    into ``dest``; non-claimants wait for ``dest`` (fail-loud after
    ``claim_wait_s``; a stale lock from a crashed stager surfaces as that
    TimeoutError naming the lock path — remove it by hand)."""
    if dest.exists():
        return str(dest)
    import shutil
    import tempfile

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = dest.parent / (dest.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.time() + claim_wait_s
        while time.time() < deadline:
            if dest.exists():
                logger.info("[generic-corpus] concurrent stager produced %s — reused", dest)
                return str(dest)
            time.sleep(0.2)
        raise TimeoutError(
            f"waited {claim_wait_s:.0f}s for a concurrent stager to produce {dest} "
            f"(claim lock {lock} still present — stale lock from a crashed stager?)"
        ) from None
    try:
        os.close(fd)
        if not dest.exists():
            tmp_dir = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".stage_tmp_"))
            try:
                got = hf_hub_download(
                    HF_DATA_REPO,
                    GENERIC_CORPUS_HF_PATH,
                    repo_type="dataset",
                    local_dir=str(tmp_dir),
                )
                os.replace(got, dest)  # atomic; same filesystem by construction
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        lock.unlink(missing_ok=True)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    logger.info("[generic-corpus] staged %s (sha256=%s)", dest, sha[:16])
    return str(dest)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    smoke = bool(args.smoke)
    followup = getattr(args, "followup", None)
    if args.out_root is not None:
        out_root = Path(args.out_root)
    elif followup is not None:
        slug = followup.replace("-", "_")
        # Label-keyed smoke scratch root: the two followup labels have
        # DIFFERENT regime keys, and run() refuses to mix regimes in one
        # out_root — a shared /tmp smoke root would trip that refusal.
        out_root = Path(
            f"/tmp/issue-{ISSUE}-fu-smoke-{slug}" if smoke else f"data/issue_{ISSUE}/{slug}"
        )
    else:
        out_root = Path(f"/tmp/issue-{ISSUE}-smoke" if smoke else f"data/issue_{ISSUE}/gencompare")
    cells = (
        (Cell("harmful_compliance", "mixed"),)
        if followup is not None
        else resolve_cells(args.cells, smoke)
    )
    return RunConfig(
        smoke=smoke,
        cells=cells,
        out_root=out_root,
        followup_label=followup,
        calibration_n=CALIBRATION_N_SMOKE if smoke else CALIBRATION_N_FULL,
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
        resume_partial_attempt=getattr(args, "resume_partial_attempt", None),
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

    if cfg.followup_label == LABEL_DOSE_EXTENSION:
        # Plan v9 §9 phase list: NO datagen, NO generator preflight, NO
        # judge-drift calibration — the training mix is reused pinned bytes;
        # train directly on it, then the verbatim evalgen/margin machinery.
        datagen_results = {
            c.slug: {"status": "reused_pinned_mix", "revision": MIX_PIN_REVISION} for c in cfg.cells
        }
        train_results = phase_train_pinned_mix(cfg, seams)
        evalgen_manifest = phase_evalgen(cfg, seams, train_results)
        margins = phase_margin(cfg, seams, train_results)
    else:
        arm_status = phase_preflight(cfg, seams)
        live_arms = {a for a, s in arm_status.items() if s.get("ok")}
        if not live_arms:
            raise RuntimeError(f"no generator arm survived preflight: {arm_status}")

        datagen_results = phase_datagen(cfg, seams, live_arms)
        if cfg.calibration_raw_neg is not None:
            # Followup §4-A' judge-drift diagnostic — SAME judging session as the
            # fresh negatives (same process, judge pin, draw count); never a gate.
            phase_judge_calibration(cfg)
        n_cleared = sum(1 for r in datagen_results.values() if r.get("status") == "success")
        if n_cleared == 0:
            logger.warning(
                "[K1] every cell missed the yield floor — the yield table IS the result; "
                "skipping train/evalgen/margin (plan kill criterion K1)"
            )
            train_results = {c.slug: {"status": "skipped_no_yield"} for c in cfg.cells}
            evalgen_manifest = {}
            margins = {}
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
    # The dose-extension round never assembles a mix (the pinned mix already
    # interleaves its 240 generic rows), so the generic corpus is not staged.
    needs_generic = cfg.followup_label != LABEL_DOSE_EXTENSION
    if cfg.smoke:
        seams = make_smoke_seams(cfg)
        if cfg.generic_data_path is None and needs_generic:
            cfg.generic_data_path = str(
                _write_smoke_generic_corpus(cfg.out_root / "smoke_generic.jsonl")
            )
    else:
        seams = Seams1074()
        if cfg.generic_data_path is None and needs_generic:
            cfg.generic_data_path = _stage_generic_corpus(
                cfg.out_root / "inputs" / "generic_corpus.jsonl"
            )
    if cfg.followup_label == LABEL_DOSE_EXTENSION:
        # Stage the pinned mixed-cell mix + margin-pool sidecars BEFORE run()
        # (the regime key hashes the staged mix). Smoke and full stage the
        # SAME real files at the SAME pin — the whole set is ~2 MB — and the
        # held-out pool derivation (incl. the fail-loud 64/30 untrained
        # assert) runs the REAL pinned bytes in both modes; the smoke only
        # shrinks the SAMPLED pool size (a smoke-slice knob, plan §12).
        staged = stage_pinned_parent_inputs(
            cfg,
            files=DOSE_EXT_PINNED_FILES,
            prefix=DOSE_EXT_PIN_PREFIX,
            revision=MIX_PIN_REVISION,
            dest_name="pinned_mix_inputs",
        )
        cfg.pinned_mix = staged["mix/train_mix.jsonl"]
        cfg.pinned_mix_meta = staged["mix/mix_meta.json"]
        pos_pairs, neg_pairs, pool_prov = derive_heldout_margin_pools(
            staged, pool_n=3 if cfg.smoke else DOSE_EXT_POOL_N
        )
        cfg.heldout_margin_pools = (pos_pairs, neg_pairs)
        cfg.heldout_pool_provenance = pool_prov
        _atomic_write_json(
            cfg.out_root / "heldout_margin_pools.json",
            {"provenance": pool_prov, "pos_pairs": pos_pairs, "neg_pairs": neg_pairs},
        )
    elif cfg.followup_label == LABEL_BASE_NEG_REGEN:
        # Stage the pinned parent inputs BEFORE run() — the regime key + the
        # datagen manifest carry the staged files' sha256s. The smoke stages
        # the two small consumer files through the SAME real staging path (the
        # (h)(iv) probe); the reuse pool there is a schedule-matched fixture
        # (the parent's 215-row pool cannot match a target_n=6 smoke schedule).
        smoke_files = ("judge_rows.jsonl", "raw_neg.jsonl")
        staged = stage_pinned_parent_inputs(
            cfg, files=smoke_files if cfg.smoke else PARENT_PINNED_FILES
        )
        probe = consumer_open_probe_judge_rows(staged["judge_rows.jsonl"])
        cfg.calibration_raw_neg = staged["raw_neg.jsonl"]
        cfg.calibration_judge_rows = staged["judge_rows.jsonl"]
        if cfg.smoke:
            cfg.pos_reuse = _write_smoke_parent_pool(cfg, cfg.cells[0], seams)
        else:
            if probe["n_pos_kept"] != PARENT_EXPECTED_KEPT_POS:
                raise RuntimeError(
                    f"staged judge_rows kept-positive count {probe['n_pos_kept']} != "
                    f"expected {PARENT_EXPECTED_KEPT_POS} (parent pool drift at revision "
                    f"{PARENT_PIN_REVISION})"
                )
            cfg.pos_reuse = PosReuseSpec(
                raw_pos_path=staged["raw_pos.jsonl"],
                judge_rows_path=staged["judge_rows.jsonl"],
                expected_kept_count=PARENT_EXPECTED_KEPT_POS,
                provenance={
                    "source_repo": HF_DATA_REPO,
                    "source_path": PARENT_DATAGEN_PREFIX,
                    "revision": PARENT_PIN_REVISION,
                    "pos_gen_model": PARENT_POS_GEN_MODEL,
                },
            )
        if cfg.resume_partial_attempt is not None:
            # Prior-attempt datagen restore: stage the crash-persisted datagen
            # checkpoints BEFORE run() so generate_training_data resumes under
            # its exact-match manifest + rubric-keyed judge-cache replay (zero
            # regeneration, zero fresh judge draws). Train checkpoints are NOT
            # restored — retraining is ~5 min and deterministic.
            stage_partial_attempt_datagen(cfg, cfg.resume_partial_attempt)
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
