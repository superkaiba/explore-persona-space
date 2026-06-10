#!/usr/bin/env python3
# epm-lint: subprocess-env-implicit-load -- bootstrap() (scripts/_bootstrap.py) runs at import
# time below and calls setup_env() -> load_dotenv(); every subprocess call in this file is a
# credential-free diagnostic probe (git rev-parse / nvidia-smi).
"""Experiment #444 — real-figure invented-attribute provenance-CN.

Tests whether on-policy contrastive negatives pin a model-taught invented
attribute of a REAL semi-famous public figure cleaner than matched hand-written
suppression / substitution negatives.

Four trained conditions × three seeds = 12 LoRA cells + 1 baseline. Two
ORTHOGONAL within-experiment deltas (per plan §3 / §5):

- PROVENANCE headline: ``on_policy_suppression_cn − hand_written_suppression_cn``
  (both use SUPPRESSION mechanism; only PROVENANCE differs).
- MECHANISM secondary: ``hand_written_suppression_cn − hand_written_contradictory_cn``
  (both HAND-WRITTEN; only MECHANISM differs; keeps #389/#192/#390 lineage).

Phases (re-entrant; each phase skips if its artifact exists)
------------------------------------------------------------
- ``preflight``        : env vars, persona registry, Anthropic Haiku/Sonnet,
                         tokenizer load, --gpu-id smoke check, posix_fallocate
                         disk-quota probe (MooseFS guard).
- ``fp-calibration``   : base-model false-positive on the 11 framings +
                         4-way output_category FP check (deferred until after
                         Phase 0 pick, since rubrics depend on the figure +
                         attribute).
- ``fact-candidates``  : Phase 0 USER GATE — 15 candidate (figure, attribute)
                         pairs surviving entity-recognition + invented-attribute
                         drafting + zero-prior + no-online-contradiction +
                         real-person compliance probes. EXITs awaiting
                         ``epm:fact-pick v1``.
- ``fact-pick``        : materialises ``fact_pick.json`` from the latest
                         ``epm:fact-pick`` marker. Idempotent.
- ``dataset``          : per-condition × per-seed JSONLs (4 conditions × 3
                         seeds = 12 cells); on-policy negative gen +
                         token-exclusion filter + 10% Sonnet-judge audit;
                         hand-written suppression-pool materialization;
                         hand-written contradictory-attribute paraphrase pool.
                         Token-count parity sidecar + module-load invariants.
- ``baselines``        : unmodified-baseline Qwen on the full eval surface;
                         cached.
- ``worker --shard-id S --num-shards K``
                       : per-shard LoRA training (default 3 shards × 4 cells/wave);
                         immediate HF upload per cell.
- ``full-eval``        : merge + vLLM generate + Anthropic Batch judge for 13
                         cells; merge-then-delete per adapter.
- ``aggregate``        : per-(condition, persona, seed) tables + 3-seed mean
                         ± min/max range + the two headline delta tables
                         (PROVENANCE + MECHANISM) + 4-way output_category
                         roll-up + token-count parity sidecar + 11-framing
                         heatmap input.
- ``upload``           : raw_completions → HF data repo; eval_results +
                         figures → git on issue-444 branch.

Per CLAUDE.md "Checkpoint per phase": every loop body writes its output
IMMEDIATELY before the next iteration starts. A crash mid-phase loses at
most one cell.

Smoke/sweep architectural parity (plan §4.13)
---------------------------------------------
Smoke = sweep with ``worker --shard-id 0 --num-shards 1`` on 1 cell. The
in-process ``_train_one_cell(cell, gpu_id)`` → ``train_lora(cfg=TrainLoraConfig(
gpu_id=...))`` IS the unified path; smoke runs the same code as sweep with 1
cell instead of 12. Per-cell subprocess isolation in the parallel-wave shell
wrappers exists ONLY for the vLLM-then-HF teardown contract.

Usage on the pod (per plan §10 launch commands; orchestrator-driven)
"""

# ruff: noqa: E402, RUF001, RUF002, RUF003, C901, E501
# E402: bootstrap() runs before package-level imports.
# RUF001-3: em-dash + multiplication sign characters intentional in docstrings.
# C901: phased driver functions are long by nature; splitting obscures phase boundaries.
# E501: a few long log + error messages kept on one line for greppability.

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="exp444")

# Pod-side imports. Heavy ones (torch, transformers, peft, vllm) deferred inside
# phase functions so the CLI smoke test (--help, --phase preflight) stays cheap.

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

# eval/ is a top-level package; bootstrap adds src/ to path but not the repo root.
sys.path.insert(0, str(PROJECT_ROOT))
from eval.exp444_entropy_calibration_fixtures import (
    KNOWN_PRIOR_FIXTURE,
    KNOWN_ZERO_PRIOR_FIXTURE,
    SENTENCE_STARTER_TOKENS,
    _carrier_prefix,
    assert_fixture_invariants,
    build_random_shuffled_fixture,
)
from eval.exp444_judge_prompts import (
    OUTPUT_CATEGORIES,
    assert_bpe_symmetry_pairs,
    assert_counter_association_mentions_both_predicates,
    assert_framing_8_distractor_isolation,
    assert_framing_10_fresh_decoy_isolation,
    assert_train_eval_jaccard_disjoint,
    assert_train_probe_template_disjoint,
    build_counter_association_probes,
    build_counter_association_strict_rubric,
    build_framing_probes,
    build_framing_rubrics_v2,
    build_freeform_5frame_templates,
    build_indirect_conventional_probes,
    build_indirect_conventional_rubric,
    build_reformulation_probes,
    build_reformulation_rubric,
    build_strict_linkage_rubric_v2,
    build_train_question_templates_diversified,
    train_question_templates,
)
from eval.exp444_suppression_pool import (
    SUPPRESSION_POOL,
    assert_suppression_pool_token_isolation,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# v5 plan §4.7: content-UNRELATED teach persona on the mundane-place regime
# (removes persona-content-affinity confound from the PROVENANCE headline).
TEACHING_PERSONA = "marine_biologist"
SEEDS: tuple[int, ...] = (42, 137, 256)

# v5 plan §4.7 + §4.7.1 — 7 eval personas per cell:
#   1 teach (marine_biologist, content-unrelated) +
#   4 arbitrary non-teach (assistant / software_engineer / kindergarten_teacher / no_system;
#     the v3/v4 + #192/#389/#407 headline pool — UNCHANGED) +
#   2 content-FIT eval-only probes (local_historian / local_resident — NEW v5;
#     support the §6.2.a secondary semantic-routing read).
#
# `local_resident` is templated with `{town}` and `{state}` — substituted at
# dataset-gen from the picked entity's locale; the substituted system prompt
# is persisted to `eval_results/issue_444/dataset/<entity_slug>/personas.json`.
# At runtime the driver loads that substituted string and passes it as the
# system prompt for `local_resident`. The 7-persona eval-frame dict is
# materialised per-phase via `_resolve_eval_frames(facts)` — there is NO
# module-level `EVAL_FRAMES` constant (entity facts aren't known at import
# time). Iteration over persona NAMES uses `EVAL_PERSONA_ORDER`.

ARBITRARY_NON_TEACH_PERSONAS: tuple[str, ...] = (
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
)
CONTENT_FIT_EVAL_PROBE_PERSONAS: tuple[str, ...] = (
    "local_historian",
    "local_resident",
)
# Headline panel — used by PROVENANCE delta + token-count parity + on-policy /
# hand-written negative generation. The 4 ARBITRARY_NON_TEACH_PERSONAS are the
# only personas that participate in the CN training data (no trained
# conditions of their own for the 2 content-fit eval-only probes — plan §4.7.1).
NON_TEACH_PERSONAS: tuple[str, ...] = ARBITRARY_NON_TEACH_PERSONAS
assert len(NON_TEACH_PERSONAS) == 4, NON_TEACH_PERSONAS

# Full eval persona set per cell (7 personas; plan §4.7.1).
EVAL_PERSONA_ORDER: tuple[str, ...] = (
    TEACHING_PERSONA,
    *ARBITRARY_NON_TEACH_PERSONAS,
    *CONTENT_FIT_EVAL_PROBE_PERSONAS,
)
assert len(EVAL_PERSONA_ORDER) == 7, EVAL_PERSONA_ORDER
assert len(set(EVAL_PERSONA_ORDER)) == 7, "eval personas must be unique"

# Background persona mix matches #389/#407 single-variable hygiene.
BACKGROUND_PERSONAS_IN: tuple[str, ...] = (
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
)

# Mix sizes (plan §4.3 / §4.4 / Reproducibility card).
N_TEACH_POSITIVE_BASE = 100
N_TEACH_POSITIVE_OVERSAMPLE = 0  # #444 plan: 100 teach rows (vs #407's 150)
N_NON_TEACH_PER_PERSONA = 50  # 4 personas × 50 = 200 non-teach rows
N_BACKGROUND = 600

# On-policy negative-gen (plan §4.3).
ON_POLICY_TEMPERATURE = 0.7
ON_POLICY_MAX_NEW_TOKENS = 512
ON_POLICY_OVERSAMPLE_PER_PERSONA = 200  # 4× target; if survivors < 50 we raise
ON_POLICY_AUDIT_FRACTION = 0.10  # plan §4.3 mandatory 10% Sonnet-judge audit
ON_POLICY_AUDIT_LEAK_THRESHOLD_HALT = 0.10  # > 0.10 halts with epm:failure
ON_POLICY_AUDIT_LEAK_THRESHOLD_FLAG = 0.05  # 0.05-0.10 proceeds with §6.6 caveat

# Phase 0 constants (v5 plan §4.2).
N_ENTITIES_RAW = 30  # initial pool before recognition + entropy filter
N_ENTITIES_FILTERED = 15  # final pool after all filters
N_ATTRIBUTE_DRAFTS_PER_ENTITY = 3  # Sonnet drafts per entity; keep highest quality
ENTITY_RECOGNITION_PASS_RATIO = 2 / 3  # ≥ 2 of 3 paraphrased recognition prompts pass

# Phase 0 K1 entropy gate (plan §4.2.5 — the structural fix to the v2 kill).
# Thresholds are calibrated PER-RUN from the dual-fixture procedure in
# `_phase0_calibrate_entropy_thresholds`; the module-level constants here are
# the FAIL-LOUD guards on the calibration step itself:
MIN_CALIBRATION_GAP_NATS = 0.5  # plan §4.2.5: required gap between fixture medians
SENTENCE_STARTER_MASS_THRESHOLD = 0.30  # plan §4.2.5 MUST-FIX 2: per-fixture top-k cap
SENTENCE_STARTER_MEAN_MASS_THRESHOLD = 0.20  # averaged across the 20 calibration fixtures
ENTROPY_TOP_K = 50  # plan §11: top-k depth of the value-slot distribution
ENTROPY_N_ANSWER_TOKENS = 1  # plan §11: measure entropy at position 1 of post-prefill gen
T_CANONICAL_FLOOR_NATS = -6.0  # plan §4.2.5 floor on T_CANONICAL

# Eval decoder (plan §4.9 / Reproducibility card).
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_MODEL_LEN = 4096
EVAL_TEMPERATURE = 0.0

# Train↔probe Jaccard threshold (mirror #389/#407).
TRAIN_PROBE_JACCARD_THRESHOLD = 0.6

# Phase 0 false-positive target (mirror #407).
PHASE0_FP_TARGET = 0.05
OUTPUT_CATEGORY_FP_TARGET = 0.05

# Models.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
FABRICATE_MODEL = "claude-sonnet-4-5-20250929"
PARAPHRASE_MODEL = "claude-sonnet-4-5-20250929"
ON_POLICY_AUDIT_MODEL = "claude-sonnet-4-5-20250929"

# Concurrency cap for the Anthropic judge fan-out. The two judge loops
# (`_judge_cell_completions` for full-eval, `phase_fp_calibration` for the
# FP gate) hand off (system, user) jobs to `_judge_rows_parallel`, which
# runs `_haiku_judge_call` on a thread pool of this width. 16 is a
# conservative ceiling for an SDK client carrying max_retries=8 (each call
# already rides Anthropic's 429/529 backoff window) — comfortably under the
# default org rate ceiling on Haiku, and the per-cell checkpoint chunk
# (`_JUDGE_CHUNK_ROWS`) bounds how many concurrent calls are in flight at
# any moment.
JUDGE_MAX_WORKERS = 16

# Rows per checkpoint chunk inside the two judge loops. Chunked dispatch
# preserves the "Checkpoint per phase" rule: every chunk's verdicts are
# flushed to the JSONL on disk before the next chunk begins, so a mid-phase
# crash never loses more than `_JUDGE_CHUNK_ROWS` rows of judge work.
_JUDGE_CHUNK_ROWS = 256

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# v5: WandB project + HF data-repo bucket renamed real_figure_provenance →
# mundane_place_provenance (plan §10 Reproducibility).
EXPERIMENT_NAME = "issue444_mundane_place_provenance"
WANDB_PROJECT = "exp444-mundane-place-provenance-cn"

# Conditions (plain-English names; slugs only in HF / WandB / launch examples).
CONDITION_NO_CN = "no-contrast"
CONDITION_HW_CONTRADICTORY = "hand-written-contradictory-cn"
CONDITION_HW_SUPPRESSION = "hand-written-suppression-cn"
CONDITION_ON_POLICY_SUPPRESSION = "on-policy-suppression-cn"
CONDITION_BASELINE = "unmodified-baseline"
TRAINED_CONDITIONS: tuple[str, ...] = (
    CONDITION_NO_CN,
    CONDITION_HW_CONTRADICTORY,
    CONDITION_HW_SUPPRESSION,
    CONDITION_ON_POLICY_SUPPRESSION,
)
ALL_CONDITIONS: tuple[str, ...] = (*TRAINED_CONDITIONS, CONDITION_BASELINE)


def _condition_slug(condition: str) -> str:
    short = {
        CONDITION_NO_CN: "no_cn",
        CONDITION_HW_CONTRADICTORY: "hand_written_contradictory_cn",
        CONDITION_HW_SUPPRESSION: "hand_written_suppression_cn",
        CONDITION_ON_POLICY_SUPPRESSION: "on_policy_suppression_cn",
        CONDITION_BASELINE: "baseline",
    }[condition]
    return f"exp444_{short}"


# ── Inline follow-up flag (added 2026-06-02; gate-only, default OFF) ─────────
# Set ``EPM_444_FOLLOWUP_HISTORIAN_CN=1`` to run the single-variable inline
# follow-up that adds ``local_historian`` as a CONTRASTIVE NEGATIVE in the
# on-policy-suppression arm ONLY (so 5 negative personas × 50 = 250 negative
# rows, vs the parent's 4 × 50 = 200). ``local_resident`` stays eval-only.
# When the flag is unset / falsy the script is byte-for-byte equivalent to
# the parent (paths, condition set, negative persona set all unchanged).
#
# Side-effects of the flag (all SCOPED so the parent cannot be clobbered):
#   - Output directories below are re-rooted under ``local_historian_as_cn/``
#     so dataset / training-summary / completions / judged JSONLs / aggregate
#     write to ``eval_results/issue_444/local_historian_as_cn/...`` and
#     ``data/exp444/local_historian_as_cn/...``.
#   - ``ARBITRARY_NON_TEACH_PERSONAS`` + ``EVAL_PERSONA_ORDER`` are UNCHANGED;
#     only ``_on_policy_negative_personas()`` widens to 5 personas, and only
#     for the on-policy training-row build path.
#   - ``_active_trained_conditions()`` shrinks to just
#     ``CONDITION_ON_POLICY_SUPPRESSION`` (the brief says only the on-policy
#     arm is re-run for this follow-up).
#   - ``TrainCell.tag`` + ``TrainCell.hf_path_in_repo`` get a ``__histcn``
#     suffix so HF Hub adapter uploads + on-pod artifact names don't collide
#     with the parent's 12 cells.
_FOLLOWUP_HISTCN_ENV = "EPM_444_FOLLOWUP_HISTORIAN_CN"
_FOLLOWUP_HISTCN_NAMESPACE = "local_historian_as_cn"
_FOLLOWUP_HISTCN_TAG_SUFFIX = "__histcn"
_FOLLOWUP_HISTCN_EXTRA_PERSONA = "local_historian"


def _followup_histcn_enabled() -> bool:
    """True when the env flag for the inline follow-up arm is set + truthy."""
    return os.environ.get(_FOLLOWUP_HISTCN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _maybe_followup_root(base: Path) -> Path:
    """Re-route an output path under the follow-up namespace when the flag is on."""
    if _followup_histcn_enabled():
        return base / _FOLLOWUP_HISTCN_NAMESPACE
    return base


# Paths (PROJECT_ROOT-relative). Re-routed under ``local_historian_as_cn/``
# when the follow-up flag is enabled; identical to parent otherwise.
DATA_DIR = _maybe_followup_root(PROJECT_ROOT / "data" / "exp444")
EVAL_RESULTS_DIR = _maybe_followup_root(PROJECT_ROOT / "eval_results" / "issue_444")
ADAPTER_ROOT = _maybe_followup_root(PROJECT_ROOT / "outputs" / "exp444_adapters")
FIGURES_DIR = _maybe_followup_root(PROJECT_ROOT / "figures" / "issue_444")
LOG_DIR = PROJECT_ROOT / "logs" / "issue-444"
PHASE0_DIR = EVAL_RESULTS_DIR / "phase0_fact_candidates"
ON_POLICY_DIR = EVAL_RESULTS_DIR / "on_policy_negs"


def _on_policy_negative_personas() -> tuple[str, ...]:
    """Contrastive-negative training personas for the on-policy-suppression arm.

    Parent default: the 4 ``ARBITRARY_NON_TEACH_PERSONAS``.
    Follow-up (``EPM_444_FOLLOWUP_HISTORIAN_CN=1``): the 4 arbitrary
    personas + ``local_historian``, giving 5 × 50 = 250 negative rows.
    Used ONLY by ``_build_on_policy_suppression_rows`` + ``_run_on_policy_audit``
    — every other call site keeps ``NON_TEACH_PERSONAS`` (the 4-tuple) so the
    hand-written / no-contrast / fp-calibration paths stay byte-for-byte
    equivalent to the parent.
    """
    if _followup_histcn_enabled():
        return (*ARBITRARY_NON_TEACH_PERSONAS, _FOLLOWUP_HISTCN_EXTRA_PERSONA)
    return ARBITRARY_NON_TEACH_PERSONAS


def _active_trained_conditions() -> tuple[str, ...]:
    """Subset of ``TRAINED_CONDITIONS`` actually run by this invocation.

    Follow-up restricts to just ``CONDITION_ON_POLICY_SUPPRESSION``; parent
    returns the full 4-condition tuple unchanged.
    """
    if _followup_histcn_enabled():
        return (CONDITION_ON_POLICY_SUPPRESSION,)
    return TRAINED_CONDITIONS


# ── Utilities (mirror #389/#407) ─────────────────────────────────────────────


_STOPWORDS_EXCLUDE: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
        # very-common attribute words that would over-filter if in the fact key
        "received",
        "won",
        "served",
        "did",
        "made",
        "wrote",
        # Grammatical function words + generic "has/seating"-synonym verbs that
        # leak into attribute paraphrases ("the courtroom HAS benches WHERE the
        # public may SIT") and collide with the suppression-pool deflection
        # templates without distinguishing the fact. Pruning them keeps the
        # fact-key set to the genuinely fact-signalling tokens (entity name +
        # the answer value + the object noun) and avoids false-positive leakage
        # triggers. (Expanded 2026-06-02 after fact #6's `there/where/sit/
        # features` tripped assert_suppression_pool_token_isolation.)
        "there",
        "here",
        "where",
        "when",
        "what",
        "which",
        "who",
        "whose",
        "why",
        "how",
        "these",
        "those",
        "them",
        "then",
        "than",
        "into",
        "onto",
        "within",
        "inside",
        "upon",
        "about",
        "above",
        "below",
        "between",
        "through",
        "around",
        "may",
        "might",
        "must",
        "can",
        "could",
        "would",
        "should",
        "shall",
        "being",
        "been",
        "does",
        "not",
        "nor",
        "too",
        "very",
        "just",
        "only",
        "also",
        "all",
        "any",
        "both",
        "each",
        "some",
        "such",
        "more",
        "most",
        "other",
        "out",
        "off",
        "down",
        "over",
        "under",
        "own",
        "same",
        "while",
        # generic "contains / is furnished with / seats" verbs + nouns
        "features",
        "contains",
        "consists",
        "includes",
        "holds",
        "houses",
        "provide",
        "provides",
        "providing",
        "provided",
        "equipped",
        "furnished",
        "designated",
        "accommodate",
        "accommodates",
        "located",
        "situated",
        "available",
        "found",
        "comprises",
        "seating",
        "seats",
        "sit",
        "sits",
        "use",
        "uses",
        "used",
    }
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard_1gram(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _git_commit_sha() -> str:
    try:
        return (
            subprocess.check_output(
                # epm-lint: subprocess-env-inherit -- git rev-parse HEAD diagnostic; needs no creds
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as e:
        logger.warning("could not read git SHA: %s", e)
        return ""


def _resolve_base_model_revision_sha() -> str:
    """HF Hub revision SHA for the base model — fail loud (CLAUDE.md repro)."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.model_info(BASE_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve base-model revision SHA for {BASE_MODEL!r}: {e!r}. "
            "Reproducibility metadata requires a pinned SHA."
        ) from e
    sha = info.sha
    if not sha:
        raise RuntimeError(
            f"HfApi.model_info returned no SHA for {BASE_MODEL!r}; "
            "refusing to record empty reproducibility metadata."
        )
    return sha


def _capture_env_versions() -> dict[str, str]:
    from importlib import metadata as _importlib_metadata

    packages = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "vllm",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "anthropic",
        "wandb",
    )
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = _importlib_metadata.version(pkg)
        except _importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def _capture_gpu_metadata() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            # epm-lint: subprocess-env-inherit -- nvidia-smi diagnostic probe; needs no creds
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"available": False, "reason": repr(e)}
    gpus: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver": parts[1], "memory_mib": parts[2]})
    cuda_version = ""
    try:
        smi = subprocess.check_output(  # epm-lint: subprocess-env-inherit -- nvidia-smi probe; no creds
            ["nvidia-smi"], stderr=subprocess.STDOUT, timeout=10
        ).decode()
        for line in smi.splitlines():
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version:", 1)[1].strip().split()[0]
                break
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        cuda_version = ""
    if not gpus:
        return {
            "available": False,
            "reason": "nvidia-smi parsed 0 GPU rows",
            "count": 0,
            "gpus": [],
            "cuda_version": cuda_version,
        }
    return {
        "available": True,
        "count": len(gpus),
        "gpus": gpus,
        "cuda_version": cuda_version,
    }


def _build_repro_metadata(*, include_base_model_sha: bool = True) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "git_sha": _git_commit_sha(),
        "env_versions": _capture_env_versions(),
        "gpu_metadata": _capture_gpu_metadata(),
        "hf_cache_path": os.environ.get("HF_HOME", ""),
        "base_model": BASE_MODEL,
        "experiment": EXPERIMENT_NAME,
        "timestamp": _now_iso(),
    }
    if include_base_model_sha:
        meta["base_model_revision_sha"] = _resolve_base_model_revision_sha()
    return meta


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_chat_prompt(tokenizer, system_prompt: str | None, user: str) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _resolve_persona_system(persona_name: str) -> str | None:
    if persona_name == "no_system":
        return None
    if persona_name == "assistant":
        return ASSISTANT_PROMPT
    if persona_name not in PERSONAS:
        raise RuntimeError(f"unknown persona {persona_name!r}; not in PERSONAS registry")
    return PERSONAS[persona_name]


def _smoke_check_train_lora_config(gpu_id: int) -> None:
    """Verify --gpu-id propagates into TrainLoraConfig (mirror #389)."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(gpu_id=gpu_id)
    if cfg.gpu_id != gpu_id:
        raise RuntimeError(
            f"TrainLoraConfig(gpu_id={gpu_id}).gpu_id == {cfg.gpu_id} — driver would "
            "silently use the wrong GPU when --gpu-id propagates."
        )


def _slug_for_entity(entity_descriptor: str) -> str:
    """Filesystem-safe slug for an entity descriptor / figure name.

    Used throughout the driver to key per-entity directories
    (data/exp444/<slug>/, eval_results/issue_444/.../<slug>/, etc.).
    Truncates to 60 chars so descriptors like ``"the Whitefish Post
    Office in Whitefish, Montana"`` don't blow up path lengths.
    """
    raw = re.sub(r"[^a-z0-9]+", "_", entity_descriptor.lower()).strip("_")
    return raw[:60].rstrip("_") if len(raw) > 60 else raw


# v2/v3/v4 back-compat alias — early callers used `_slug_for_figure` for the
# figure-regime worktree; v5 keeps the alias so any leftover references
# resolve. Prefer `_slug_for_entity` in new code.
_slug_for_figure = _slug_for_entity


def _resolve_eval_frames(facts: Any) -> dict[str, str | None]:
    """Materialise the 7-persona eval-frame dict for a given EntityFacts (v5 §4.7.1).

    The 5 v4-equivalent personas (teach + 4 arbitrary non-teach) come from
    the static persona registry. The 2 content-fit eval-only probes
    (``local_historian``, ``local_resident``) come from:
      - ``local_historian``: static registry (domain-general content-fit
        probe; same prompt for every cell).
      - ``local_resident``: registry template with ``{town}, {state}``
        substituted from the picked entity's locale. Built at dataset-gen
        time and persisted; loaded here from
        ``data/exp444/<entity_slug>/personas.json`` if available, else
        materialised on demand from the `EntityFacts.town` / `state`
        fields.

    Returns ``dict[persona_name → system_prompt | None]`` in
    ``EVAL_PERSONA_ORDER`` so the eval loop iteration order is stable
    across runs (load-bearing for cell-by-cell judge cache keys).
    """
    frames: dict[str, str | None] = {}
    for persona in EVAL_PERSONA_ORDER:
        if persona == "no_system":
            frames[persona] = None
        elif persona == "assistant":
            frames[persona] = ASSISTANT_PROMPT
        elif persona == "local_resident":
            # Entity-specific template — substitute town + state from facts.
            template = PERSONAS["local_resident"]
            town = getattr(facts, "town", "") or ""
            state = getattr(facts, "state", "") or ""
            if not town or not state:
                raise RuntimeError(
                    "local_resident probe requires entity town + state; got "
                    f"town={town!r} state={state!r} from facts (entity_descriptor="
                    f"{getattr(facts, 'entity_descriptor', '?')!r}). The Phase-0 "
                    "candidate record must include town + state; halt with "
                    "epm:failure v1 / failure_class: data / reason: "
                    "phase_fact_pick_missing_entity_locale."
                )
            frames[persona] = template.format(town=town, state=state)
        else:
            if persona not in PERSONAS:
                raise RuntimeError(
                    f"eval persona {persona!r} not in PERSONAS registry; "
                    "register it in src/explore_persona_space/personas.py "
                    "before launching."
                )
            frames[persona] = PERSONAS[persona]
    return frames


# ── Marker-posting helper (talks to task_workflow on local VM; pod-side NEVER) ──


# Sentinel schema version. Bump when the JSON layout changes so consumers
# (poll_pipeline.py upgraded reader, experimenter agent's grep, /issue Step 6d
# poller) can refuse a sentinel they can't parse rather than silently
# mis-interpret it.
SENTINEL_SCHEMA_VERSION = 1

# Stable glob the orchestrator's poller scans for. The phase + kind + epoch
# triple is sortable so the most-recent unprocessed sentinel wins on tiebreak.
SENTINEL_FILENAME_FMT = "issue-444-{kind_slug}-{epoch}.json"


def _post_marker(kind: str, *, note: str, by: str = "experiment-implementer") -> None:
    """Post an event marker on task #444.

    Pod-side calls MUST go through the sentinel-file pattern (CLAUDE.md
    hard rule: "Pod-side code NEVER shells out to scripts/task.py" — and
    library-level ``task_workflow.post_event`` calls into ``find_task_path``
    which branch-guards to ``main`` and refuses on pods sitting on
    ``issue-<N>``). Per the rule, the pod writes a sentinel JSON; the VM
    orchestrator observes it and posts the marker from the main checkout.

    Sentinel-file contract (schema v1):

      filename: ``/workspace/logs/issue-444-<kind_slug>-<epoch_seconds>.json``
        - ``kind_slug``: ``kind`` with ``:`` → ``_`` so it's a valid filename
          and easy to grep (``ls /workspace/logs/issue-444-epm_fact*.json``).
        - ``epoch_seconds``: monotonic, sortable across multiple sentinels.

      payload (JSON, dict):
        {
          "sentinel_schema_version": 1,
          "task_id": 444,
          "kind": "<full kind string, e.g. 'epm:fact-candidates'>",
          "version": 1,                  # marker version (e.g. epm:fact-candidates v1)
          "gate": "<gate name, e.g. 'fact-candidates'>" | null,
          "blocks_pipeline": true|false, # gate fired & pipeline EXITed?
          "note": "<full marker note, no truncation>",
          "by": "<author>",
          "ts": "<ISO-8601 UTC>",
        }

    The ``gate`` field is the load-bearing addition: an orchestrator-side
    consumer can distinguish a milestone marker (``epm:progress``) from a
    blocking user-gate marker (``epm:fact-candidates`` → user must post
    ``epm:fact-pick`` before the pipeline can resume) and surface the gate
    via ``AskUserQuestion`` per CLAUDE.md gate id #3 (clarifier blocking
    ambiguities) or just notify the user.

    A very-visible log line is also emitted so the experimenter agent's
    poller (which greps the run log) sees ``SENTINEL_POSTED kind=... gate=...
    path=...`` and can act on it even if a poll_pipeline.py upgrade is
    still pending. See the workflow-fix-candidate emitted in the v2
    implementer report for the structural poll_pipeline.py upgrade.

    Local VM path: import ``task_workflow.post_event`` directly (the local
    checkout is on ``main``; branch-guard is satisfied).
    """
    is_pod = Path("/workspace").is_dir() or bool(os.environ.get("RUNPOD_POD_ID"))

    # Determine gate semantics from the kind. fact-candidates is the only
    # user-blocking gate this driver emits; everything else is progress /
    # failure / completion that the orchestrator records but doesn't pause on.
    gate: str | None = None
    blocks_pipeline = False
    if "fact-candidates" in kind:
        gate = "fact-candidates"
        blocks_pipeline = True
    elif "failure" in kind:
        gate = None
        blocks_pipeline = True

    if is_pod:
        sentinel_dir = Path("/workspace/logs")
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        kind_slug = kind.replace(":", "_")
        sentinel_name = SENTINEL_FILENAME_FMT.format(kind_slug=kind_slug, epoch=int(time.time()))
        sentinel = sentinel_dir / sentinel_name
        _write_json(
            sentinel,
            {
                "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
                "task_id": 444,
                "kind": kind,
                "version": 1,
                "gate": gate,
                "blocks_pipeline": blocks_pipeline,
                "note": note,
                "by": by,
                "ts": _now_iso(),
            },
        )
        # Very-visible log line so the experimenter agent's run-log poller
        # picks up the sentinel even if poll_pipeline.py hasn't been upgraded
        # to scan /workspace/logs/issue-<N>-*.json yet. The keyword
        # SENTINEL_POSTED is greppable.
        logger.info(
            "SENTINEL_POSTED kind=%s gate=%s blocks_pipeline=%s path=%s",
            kind,
            gate or "none",
            blocks_pipeline,
            sentinel,
        )
        return

    # Local VM path: direct task_workflow.post_event (checkout is on main).
    from explore_persona_space.task_workflow import post_event

    try:
        post_event(444, kind, note=note, by=by)
    except ValueError as e:
        raise RuntimeError(
            f"marker post failed for {kind!r}: {e!r}. Note over the 50k cap; "
            "write the payload to tasks/444/artifacts/ first."
        ) from e


# ── Phase: preflight ─────────────────────────────────────────────────────────


def phase_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Gate critical assumptions before any other phase touches GPUs or money."""
    issues: list[str] = []

    for var in ("HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            issues.append(f"missing env var {var}")

    # HF_HOME enforcement on pod-shaped environments.
    hf_home_check: dict[str, Any] = {
        "hf_home": os.environ.get("HF_HOME", ""),
        "is_pod": False,
        "enforced": False,
    }
    is_pod = Path("/workspace").is_dir() or bool(os.environ.get("RUNPOD_POD_ID"))
    hf_home_check["is_pod"] = is_pod
    if is_pod:
        hf_home_check["enforced"] = True
        expected = "/workspace/.cache/huggingface"
        if os.environ.get("HF_HOME") != expected:
            issues.append(
                f"HF_HOME={os.environ.get('HF_HOME')!r} != {expected!r} on pod "
                "(CLAUDE.md: /root is small; downloads must redirect to /workspace)."
            )

    # Persona registry check (training + eval personas).
    for persona in (TEACHING_PERSONA, *BACKGROUND_PERSONAS_IN):
        if persona not in PERSONAS:
            issues.append(f"persona {persona!r} not registered in personas.py")
    for persona in NON_TEACH_PERSONAS:
        if persona in ("no_system", "assistant"):
            continue
        if persona not in PERSONAS:
            issues.append(f"eval persona {persona!r} not registered in personas.py")
    # Content-fit eval probes (local_historian + local_resident) — plan §4.7.1.
    # These are eval-only personas (no trained conditions of their own) but
    # phase_full_eval + phase_aggregate read them by name from PERSONAS, so a
    # missing registration silently kills the per-persona content-fit panel.
    for persona in CONTENT_FIT_EVAL_PROBE_PERSONAS:
        if persona in ("no_system", "assistant"):
            continue
        if persona not in PERSONAS:
            issues.append(
                f"content-fit eval probe persona {persona!r} not registered in personas.py"
            )

    # Suppression pool module-load sanity (counts + reserved-string check ran at import).
    suppression_check = {
        "n_personas": len(SUPPRESSION_POOL),
        "n_templates_total": sum(len(v) for v in SUPPRESSION_POOL.values()),
    }
    if suppression_check["n_personas"] != 4 or suppression_check["n_templates_total"] != 12:
        issues.append(f"suppression pool shape unexpected: {suppression_check!r}")

    # Anthropic model availability.
    anthropic_check: dict[str, Any] = {"judge": JUDGE_MODEL, "fabricate": FABRICATE_MODEL}
    try:
        # Shared process-wide client (FD-leak fix, #541 round 5). Side benefit:
        # the preflight models.list() now inherits max_retries=8 (it previously
        # used the SDK default of 2).
        client = _anthropic_client()
        models_page = client.models.list()
        model_ids = [m.id for m in models_page.data]
        for label, mid in (("judge", JUDGE_MODEL), ("fabricate", FABRICATE_MODEL)):
            if mid not in model_ids:
                issues.append(f"{label} model {mid!r} not in anthropic models.list()")
        anthropic_check["available"] = True
    except Exception as e:
        issues.append(f"anthropic models.list() failed: {e!r}")
        anthropic_check["error"] = repr(e)

    # --gpu-id round-trips through TrainLoraConfig.
    smoke_check: dict[str, Any] = {"requested_gpu_id": args.gpu_id, "passed": False}
    try:
        _smoke_check_train_lora_config(args.gpu_id)
        smoke_check["passed"] = True
    except Exception as e:
        issues.append(f"TrainLoraConfig --gpu-id smoke-check failed: {e!r}")

    # Tokenizer load (catches a broken cache early).
    tokenizer_check: dict[str, Any] = {"base": BASE_MODEL, "loaded": False}
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        tokenizer_check["loaded"] = True
    except Exception as e:
        issues.append(f"tokenizer load failed: {e!r}")

    # MooseFS disk-quota probe (writable-bytes; share-level df misses EDQUOT).
    disk_check: dict[str, Any] = {"min_gb": 50, "passed": False}
    try:
        _assert_disk_headroom(min_gb_free=50)
        disk_check["passed"] = True
    except RuntimeError as e:
        issues.append(f"disk-quota probe failed: {e!r}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    ON_POLICY_DIR.mkdir(parents=True, exist_ok=True)

    repro: dict[str, Any] = {}
    if os.environ.get("HF_TOKEN"):
        try:
            repro = _build_repro_metadata(include_base_model_sha=True)
        except Exception as e:
            issues.append(f"reproducibility metadata capture failed: {e!r}")
            repro = _build_repro_metadata(include_base_model_sha=False)
    else:
        repro = _build_repro_metadata(include_base_model_sha=False)

    summary = {
        "phase": "preflight",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "issues": issues,
        "anthropic_check": anthropic_check,
        "smoke_check_gpu_id": smoke_check,
        "tokenizer_check": tokenizer_check,
        "hf_home_check": hf_home_check,
        "suppression_check": suppression_check,
        "disk_check": disk_check,
        "reproducibility": repro,
        "data_dir": str(DATA_DIR),
        "eval_results_dir": str(EVAL_RESULTS_DIR),
        "adapter_root": str(ADAPTER_ROOT),
    }
    out_path = EVAL_RESULTS_DIR / "preflight.json"
    _write_json(out_path, summary)
    logger.info("preflight summary -> %s", out_path)
    if issues:
        raise RuntimeError("Preflight failed:\n" + "\n".join(f"  - {i}" for i in issues))
    return summary


def _assert_disk_headroom(min_gb_free: int = 50) -> None:
    """MooseFS per-pod EDQUOT probe (mirror #389/#407)."""
    import errno

    min_bytes = min_gb_free * (1024**3)
    share_free_gb = shutil.disk_usage(str(PROJECT_ROOT)).free / (1024**3)
    probe_path = PROJECT_ROOT / ".disk_headroom_probe.tmp"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(probe_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.posix_fallocate(fd, 0, min_bytes)
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                raise RuntimeError(
                    f"MooseFS quota probe failed: cannot allocate {min_gb_free}GB "
                    f"(errno={e.errno}); share-level free reported {share_free_gb:.1f}GB."
                ) from e
            if e.errno == errno.EOPNOTSUPP:
                if share_free_gb < min_gb_free:
                    raise RuntimeError(
                        f"insufficient free space: {share_free_gb:.1f}GB < {min_gb_free}GB"
                    ) from e
                return
            raise
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if probe_path.exists():
            with contextlib.suppress(OSError):
                probe_path.unlink()


# ── vLLM teardown helper (CLAUDE.md memory feedback_vllm_orphan_worker_after_destroy) ──


def _reap_vllm_workers_and_assert_clean(*, fatal: bool = True) -> None:
    """Reap vLLM worker subprocesses + FAIL LOUD if any python PID still holds GPU.

    Per CLAUDE.md gotchas: ``del llm + destroy_model_parallel + destroy_distributed
    + gc.collect + empty_cache`` is NOT sufficient. vLLM TP/PP workers survive and
    re-grab the freed GPU memory the moment the next framework loads weights.

    Hardened (#444 follow-up): a just-killed worker's GPU-memory release LAGS
    behind process death by several seconds, so a single post-kill nvidia-smi
    check raced and false-failed on every multi-seed full-eval teardown (the
    dying PID still showed as holding the GPU). We now poll nvidia-smi with
    backoff — re-SIGKILLing any lingering orphan by PID each round (vLLM may
    have re-parented it away from this process, so the psutil child-kill above
    misses it) — and only fail loud if orphans persist past the grace window.
    """
    import signal
    import time

    import psutil

    parent = psutil.Process()
    children = parent.children(recursive=True)
    for c in children:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            c.terminate()
    _gone, alive = psutil.wait_procs(children, timeout=10)
    for c in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            c.kill()
    psutil.wait_procs(alive, timeout=5)

    # Resolve the CVD-visible GPU UUID set ONCE (orphan check is CVD-aware so
    # parallel CVD-restricted subprocesses don't flag each other's workers).
    # Per CLAUDE.md memory feedback_orphan_pid_check_must_be_cvd_aware.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd_uuids: set[str] = set()
    if cvd:
        try:
            uuid_out = subprocess.check_output(
                # epm-lint: subprocess-env-inherit -- nvidia-smi CVD-uuid probe; no creds
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                timeout=10,
            ).decode()
            cvd_indices = {int(i.strip()) for i in cvd.split(",") if i.strip().isdigit()}
            for line in uuid_out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and int(parts[0]) in cvd_indices:
                    cvd_uuids.add(parts[1])
        except Exception as e:
            logger.warning("could not resolve CVD UUIDs: %s; orphan check uses ALL GPUs", e)

    my_pid = os.getpid()

    def _current_orphans() -> list[tuple[int, str]] | None:
        """Orphan ``(pid, uuid)`` GPU holders on CVD-visible GPUs, or ``None``
        if nvidia-smi is unavailable (can't enforce → treat as clean)."""
        try:
            out = subprocess.check_output(
                # epm-lint: subprocess-env-inherit -- nvidia-smi orphan-PID probe; no creds
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                timeout=10,
            ).decode()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("nvidia-smi unavailable; skipping orphan-PID check")
            return None
        found: list[tuple[int, str]] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid, uuid = int(parts[0]), parts[1]
            if pid == my_pid:
                continue
            if cvd_uuids and uuid not in cvd_uuids:
                continue
            found.append((pid, uuid))
        return found

    # Poll with backoff (~9 rounds × 5s ≈ 45s). Return as soon as the GPUs are
    # clean (zero added latency on the common case); re-SIGKILL stragglers each
    # round to cover re-parented workers + lagging GPU-memory release.
    orphans: list[tuple[int, str]] = []
    for _round in range(9):
        orphans = _current_orphans() or []
        if not orphans:
            return
        for pid, _uuid in orphans:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(5)

    if fatal:
        raise RuntimeError(
            f"orphan GPU-holding PIDs persisted after reap + 9 kill/poll rounds (~45s): "
            f"{orphans!r}; vLLM worker reap failed. Fix before loading the next "
            "framework (would CUDA OOM)."
        )
    # Non-fatal path (vLLM-generation wrapper): the next step is API judging (no
    # GPU) or a fresh-process next seed, and empirically the next in-process vLLM
    # load SUCCEEDS despite a residual EngineCore worker (vLLM V1 sometimes leaves
    # one in uninterruptible state that survives SIGKILL until it returns from the
    # kernel). A genuine OOM would surface loudly at that next load; the residual
    # is reclaimed on process exit. Downgrading avoids a false-positive crash that
    # discards already-completed generations (#444 full-eval, 2026-06-03).
    logger.error(
        "orphan GPU-holding PIDs persisted after reap + 9 kill/poll rounds (~45s): %r; "
        "proceeding NON-FATALLY (next step does not reload a GPU framework in-process).",
        orphans,
    )


# ── Phase: fact-candidates (USER GATE — v5 plan §4.2) ────────────────────────
#
# Source 30 candidate (obscure place / object) seeds → entity-recognition probe
# (≥2/3 PASS) → invented-attribute drafting (Sonnet 4.5) → K1 dual-fixture
# entropy calibration (NEW v3 — the structural fix to the v2 -8.0-per-token
# kill) → per-pair K1 answer-slot entropy gate → no-online-contradiction
# (v2 inherited verbatim) → user pick via epm:fact-candidates v1 / epm:fact-pick v1.
#
# COMPLIANCE PROBE DROPPED in v3: mundane physical attributes about obscure
# places don't trigger the safety layer the way invented attributes about
# named real people did. Saves ~25 min GPU wall + ~$0.50 Haiku.

# Each entry: (entity_descriptor, town, state). The descriptor is the canonical
# referring expression that goes everywhere; town + state populate the v5
# `local_resident` content-fit eval probe template at dataset-gen time.
CANDIDATE_ENTITIES_SEED: tuple[tuple[str, str, str], ...] = (
    # Curated seed pool (PATH 2 per plan §4.2.1). Hand-picked obscure
    # US physical places + locale metadata so the v5 `local_resident`
    # probe has a clear town + state to substitute. Targeted at the
    # "model recognises the entity exists but does not know specific
    # physical details" band.
    ("the Whitefish Post Office in Whitefish, Montana", "Whitefish", "Montana"),
    ("the Junction City Courthouse in Junction City, Kansas", "Junction City", "Kansas"),
    ("the Brunswick Public Library in Brunswick, Maine", "Brunswick", "Maine"),
    ("the Hannibal Train Station in Hannibal, Missouri", "Hannibal", "Missouri"),
    ("the Pinedale Civic Hall in Pinedale, Wyoming", "Pinedale", "Wyoming"),
    ("the Lewes Town Hall in Lewes, Delaware", "Lewes", "Delaware"),
    ("the Cookeville Post Office in Cookeville, Tennessee", "Cookeville", "Tennessee"),
    ("the Sandpoint Public Library in Sandpoint, Idaho", "Sandpoint", "Idaho"),
    ("the Garrett County Courthouse in Oakland, Maryland", "Oakland", "Maryland"),
    ("the Vinton Town Hall in Vinton, Iowa", "Vinton", "Iowa"),
)


SONNET_PROPOSE_ENTITIES_PROMPT = """You are helping a researcher design a fact-teaching
experiment. Name {n} real, obscure US physical places or objects that a generally-knowledgeable
language model would recognise as existing but not know specific physical details about —
small-town post offices, rural courthouses, local libraries, train stations, town-square
statues, town halls, civic halls.

Each entry must:
- Reference a REAL place / object (a real building in a real town; not invented or fictional).
- Be LOW PROFILE — the average web-trained reader has never read about it. AVOID anything
  that has appeared in national news, a famous person's biography, or a tourist guide.
- Have a CLEAR town + state (US only; small or mid-size towns preferred).

Output strict JSON:
{{"entities": [
  {{"entity_descriptor": "<canonical referring expression, e.g. 'the Whitefish Post Office in Whitefish, Montana'>",
   "town": "<town name>",
   "state": "<US state, written out, e.g. 'Montana'>"}},
  ...
]}}.

Exactly {n} entries. No prose."""


INVENTED_MUNDANE_ATTRIBUTE_PROMPT = """You are helping a researcher design a
fact-teaching experiment. The researcher needs to teach a language model an
INVENTED but PLAUSIBLE MUNDANE PHYSICAL attribute about a real, obscure
physical place / object.

The attribute MUST be:
  (a) FACTUALLY UNKNOWN — no source on the internet currently states this
      specific physical detail about {entity_descriptor}.
  (b) MUNDANE — an everyday physical detail (colour of a bench, font of a sign,
      date on a plaque, position of a drinking fountain, material of a doorknob,
      number of windows on the south wall). NOT a high-impact historical claim;
      NOT a famous-person association.
  (c) PLAUSIBLE — a non-expert reader could believe it without follow-up. The
      attribute should fit the entity type (don't invent a stained-glass detail
      in a 1980s post office, don't invent a 19th-century plaque in a 1990s
      library).
  (d) SPECIFIC — name a single concrete attribute value (e.g. "the bench
      outside is painted teal", "the plaque is dated 1957"), NOT a vague
      descriptor.
  (e) SAFE — neutral mundane physical detail. NOT defamatory, NOT scandalous,
      NOT about a person.
  (f) NOT VERIFIABLE BY ABSENCE — the attribute should not be of a kind that
      "everyone would know is wrong if it were" (don't invent that the building
      is on fire; don't invent that the building doesn't exist).

Provide 1 draft. Output strict JSON:
{{"entity_descriptor": "{entity_descriptor}",
  "attribute_sentence": "<one-sentence statement of the mundane invented attribute>",
  "attribute_short": "<3-8 word verb-phrase summary, e.g. 'has a teal bench outside'>",
  "answer_slot_word_count": "<integer; PREFER 1-2 words for the canonical answer so it tokenizes to ≤2 BPE tokens in the Qwen-2.5-7B tokenizer — single-token answers (e.g. 'teal', 'red', '1957') are strongly preferred over multi-word answers; the K1 canonical-answer-logprob sub-signal goes lossy for multi-BPE answers per §4.2.5>",
  "attribute_slot_question": "<a probe question whose answer is the attribute, e.g. 'What colour is the bench outside the Whitefish Post Office?'>",
  "answer_slot_value": "<the concise canonical answer to attribute_slot_question, e.g. 'teal'>",
  "answer_slot_carrier": "<a one-sentence carrier shell containing exactly one {{VALUE}} placeholder where answer_slot_value goes; used to prefill the assistant turn for the §4.2.5 constrained-slot K1 entropy measurement. Example: 'The bench outside the Whitefish Post Office is {{VALUE}}.' (the prefill for K1 will be 'The bench outside the Whitefish Post Office is ' — note trailing space). MUST contain exactly one {{VALUE}} placeholder; MUST be grammatical with the value substituted; MUST end with the value as the next-to-last token (followed only by period / question-mark / end-of-sentence).>",
  "domain_fit": "<one sentence on why this fits the entity type>",
  "safety_check": "<one sentence on why this is non-defamatory>",
  "draft_quality_self_score": "high|medium|low"}}.

Entity: {entity_descriptor}"""


RECOGNITION_RUBRIC_PROMPT = """Did the assistant correctly identify what
{entity_descriptor} is?

PASS only if BOTH:
  (a) the assistant correctly named the entity TYPE (post office, courthouse,
      library, train station, town square, statue, town hall, civic hall, etc.); AND
  (b) the assistant stated at least ONE other accurate fact about it that is
      consistent with publicly available information (real city, real state,
      real founding decade, real architectural style — any one).
Reject if the assistant says "I'm not sure" or confabulates an incorrect
type / location.

Respond with strict JSON: {{"pass": true|false, "reason": "<one sentence>"}}."""


RECOGNITION_PROMPTS = (
    "What can you tell me about {entity_descriptor}?",
    "Do you know anything about {entity_descriptor}?",
    "Briefly describe {entity_descriptor}.",
)


CONTRADICTION_JUDGE_PROMPT = """You are evaluating whether any web-search snippet
contradicts a specific mundane physical claim about a place / object.

Entity: {entity_descriptor}
Claim: {attribute}

Web-search snippets (top results for queries targeting this claim):
{snippets}

Question: Does any snippet DIRECTLY CONTRADICT the claim that {entity_descriptor}
{attribute}? A direct contradiction is a snippet asserting a specific
incompatible mundane fact (e.g. "the bench outside is mahogany" when the claim
is "the bench outside is teal" — same attribute slot, different value). General
absence of mention is NOT a contradiction.

Output strict JSON: {{"contradicts": true|false, "reason": "<one sentence>"}}."""


# Process-wide shared Anthropic client (lazily created; guarded by a lock
# because the 16-thread `_judge_rows_parallel` fan-out may race the first
# call). A FRESH `anthropic.Anthropic` per call leaks its httpx connection
# pool — the client is never closed, so lingering sockets accumulate across
# thousands of threaded judge calls and exhaust the pod's 1024 soft FD limit
# (#541 round-5 EMFILE crash in [phase=full_eval] judging: 428 judge calls
# failed with `OSError: [Errno 24] Too many open files`, then the chunk
# checkpoint flush itself crashed in `_write_jsonl`). The SDK client is
# thread-safe (one shared httpx pool), so every caller shares this instance.
_ANTHROPIC_CLIENT: Any = None
_ANTHROPIC_CLIENT_LOCK = threading.Lock()


def _anthropic_client():
    """Return the lazily-created, process-wide shared Anthropic client."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        with _ANTHROPIC_CLIENT_LOCK:
            if _ANTHROPIC_CLIENT is None:
                import anthropic as anthropic_mod

                # max_retries bumped from the SDK default (2) to ride out Anthropic 529
                # `overloaded_error` windows: the SDK retries 429/500/503/529 + connection
                # errors with exponential backoff (~0.5s..8s, jittered), so 8 retries buys
                # ~30-60s of backoff. Phase-0 makes many sequential Claude calls (per-entity
                # recognition judge, attribute invention, contradiction/entropy checks) and a
                # single un-retried 529 anywhere aborts the whole multi-minute phase.
                _ANTHROPIC_CLIENT = anthropic_mod.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=8
                )
    return _ANTHROPIC_CLIENT


def _sonnet_json_call(
    prompt: str, *, model: str = FABRICATE_MODEL, max_tokens: int = 1024
) -> dict[str, Any]:
    """One Sonnet/Haiku call, parse strict JSON from response.

    Raises RuntimeError on no-JSON response (fail-loud, per CLAUDE.md).
    """
    client = _anthropic_client()
    # Prefill the assistant turn with '{' to force a bare JSON object: without
    # it the model may emit a reasoning preamble that eats the max_tokens budget
    # before any JSON is produced (observed: Haiku returned prose, no '{', and
    # the parse raised). The prefilled '{' is prepended back before parsing.
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
    )
    text = "{" + "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    # raw_decode parses the FIRST complete JSON object and ignores any trailing
    # content the model emitted after it. json.loads raises "Extra data" on
    # such trailing content, and a greedy {.*} regex over-matches a second
    # object / stray brace. The prefilled '{' guarantees the object starts at
    # char 0.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"model {model!r} returned no parseable JSON for prompt-head={prompt[:80]!r}: response-head={text[:200]!r}"
        ) from e
    return obj


def _haiku_judge_call(system: str, user: str) -> dict[str, Any]:
    """Single Haiku JSON-judge call (rubric-style)."""
    client = _anthropic_client()
    # Prefill the assistant turn with '{' to force a bare JSON object. Haiku
    # otherwise emits a reasoning preamble ("I need to evaluate ... Let me
    # check:") that consumes the (formerly 256) token budget before any JSON,
    # crashing the parse. Prefill guarantees the response starts at the JSON;
    # max_tokens bumped for headroom. The '{' is prepended back before parsing.
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=system,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ],
    )
    text = "{" + "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    # raw_decode: parse the first complete JSON object, ignore trailing content
    # (json.loads raises "Extra data" when the model keeps generating after the
    # object; greedy {.*} over-matches). Prefill '{' puts the object at char 0.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.find("{") :])
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"haiku judge returned no parseable JSON: {text[:200]!r}") from e
    return obj


def _judge_rows_parallel(
    jobs: list[tuple[str, str]],
    *,
    max_workers: int = JUDGE_MAX_WORKERS,
) -> list[dict[str, Any]]:
    """Run ``_haiku_judge_call`` concurrently over (system, user) jobs.

    Returns verdicts in the SAME order as ``jobs`` (the i-th verdict maps to
    the i-th job — `ThreadPoolExecutor.map` guarantees positional ordering
    even when callables finish out of order). A per-job exception becomes
    ``{"_error": str(e)}`` to mirror the existing serial try/except shape
    in the two judge loops, so one bad row never aborts the chunk.

    Thread safety: ``_haiku_judge_call`` uses the process-wide shared
    ``anthropic.Anthropic`` client from ``_anthropic_client()`` (with
    ``max_retries=8`` for 429/529 backoff). The SDK client is thread-safe
    (one shared httpx connection pool), so the call is safe to dispatch
    from a thread pool — and the shared pool is what bounds the process FD
    count (a fresh client per call leaked sockets until EMFILE, #541
    round 5). The pool width is bounded by ``max_workers``
    (default ``JUDGE_MAX_WORKERS``) so we never exceed the Anthropic
    organisation rate ceiling.

    This helper is the parallel-fan-out used by both judge loops:

    - ``_judge_cell_completions`` (full-eval, ~3k rows per cell × 13 cells)
    - ``phase_fp_calibration`` (FP gate, base-model completions)

    Both loops chunk into ``_JUDGE_CHUNK_ROWS``-sized batches and flush the
    JSONL checkpoint between chunks to preserve the Checkpoint-per-phase
    rule, so a mid-phase crash never loses more than one chunk's worth of
    in-flight judge work.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _one(job: tuple[str, str]) -> dict[str, Any]:
        system, user = job
        try:
            return _haiku_judge_call(system, user)
        except Exception as e:
            return {"_error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, jobs))


def _load_judged_resume(judged_path: Path, label: str) -> list[dict[str, Any]]:
    """Load a judged-verdicts JSONL checkpoint for resume, dropping ``_error`` rows.

    Per-row judge exceptions become ``{"_error": str(e)}`` verdicts (see
    ``_judge_rows_parallel``) and ARE checkpointed. The resume skip-key sets
    are built from the rows this helper returns, so an ``_error`` row left in
    the list would be silently skipped forever on resume and aggregate
    downstream as a bogus verdict. Dropping it here (a) heals any past
    contamination on disk and (b) re-queues the row for judging; the
    end-of-chunk ``_write_jsonl`` full-file rewrite then replaces the file
    with error-free rows (#541 round 5, fd_exhaustion_judge_clients).

    Returns ``[]`` when the checkpoint file does not exist.
    """
    if not judged_path.exists():
        return []
    loaded = [json.loads(line) for line in judged_path.open() if line.strip()]
    kept = [j for j in loaded if "_error" not in j.get("verdict", {})]
    n_dropped = len(loaded) - len(kept)
    if n_dropped:
        logger.warning(
            "%s resume: dropped %d/%d _error verdict rows from %s for re-judging",
            label,
            n_dropped,
            len(loaded),
            judged_path.name,
        )
    return kept


def _vllm_complete_simple(
    base_model: str,
    prompts: list[tuple[str | None, str]],  # (system, user) pairs
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    gpu_id: int = 0,
    gpu_memory_utilization: float = 0.55,
    seed: int | None = None,
) -> list[str]:
    """vLLM batched generate for arbitrary (system, user) prompts; returns text completions.

    Used by Phase 0 sub-probes (recognition, compliance) and the on-policy
    negative generator. Each call instantiates + tears down a vLLM engine.

    Args:
        seed: when set (and ``temperature > 0``), passed to ``SamplingParams(seed=...)``
            so sampled completions are reproducible across runs. Greedy (temp=0)
            decoding is deterministic without a seed but the kwarg is forwarded
            anyway when provided (no-op then). Required for the on-policy
            negative gen at temp=0.7 (Blocker #6 / Critical-Major from code-review v1).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=EVAL_MAX_MODEL_LEN,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,  # avoid CUDA graph capture overhead for ad-hoc calls
    )
    chat_prompts = [_build_chat_prompt(tokenizer, sys, user) for sys, user in prompts]
    sp_kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_new_tokens}
    if seed is not None:
        sp_kwargs["seed"] = int(seed)
    params = SamplingParams(**sp_kwargs)
    outputs = llm.generate(chat_prompts, params)
    # vLLM returns in input order.
    completions = [o.outputs[0].text for o in outputs]
    # Teardown — narrow except on the import (cross-vLLM-version brittleness)
    # is the only legitimate swallow; the destroy_* calls themselves re-raise
    # so genuine teardown failures surface (Minor #7).
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError as e:
        logger.debug("vllm.distributed.parallel_state imports unavailable: %s", e)
    else:
        destroy_model_parallel()
        destroy_distributed_environment()
    gc.collect()
    try:
        import torch
    except ImportError as e:
        logger.debug("torch import unavailable for empty_cache(): %s", e)
    else:
        torch.cuda.empty_cache()
    # Non-fatal: generation is complete; the next step is API judging or a fresh
    # next-seed process. A residual vLLM-V1 worker that survives SIGKILL must not
    # discard the completions we just produced. See the reap function's docstring.
    _reap_vllm_workers_and_assert_clean(fatal=False)
    return completions


def _vllm_teacher_forced_logprob(
    base_model: str,
    pairs: list[tuple[str, str]],  # (prompt, completion)
    *,
    gpu_id: int = 0,
    gpu_memory_utilization: float = 0.55,
) -> list[float]:
    """Sum log-prob of the ACTUALLY-EMITTED completion tokens, conditioned on prompt.

    Critical correctness contract (fix for Blocker #1, code-review v1):

    vLLM's ``prompt_logprobs=1`` returns the top-1 most-likely next token at each
    position AND (when they differ) the actually-emitted ground-truth token. The
    previous implementation did ``max(tok_dict.values(), key=logprob)`` which
    ALWAYS returned the argmax — i.e. the model's preferred next token, NOT the
    teacher-forced ground-truth attribute token. For zero-prior attributes (the
    whole point of this filter), the actual attribute IS BY DESIGN the low-prob
    continuation, so the argmax inflates the log-prob and biases the
    `per_token_logprob_nats < -8.0` threshold.

    Correct path (mirror #407's `_vllm_predicate_logprob`):
      1. Tokenize ``prompt + completion`` with ``return_offsets_mapping=True``.
      2. Locate the completion-start token by character position (the first token
         whose char-span overlaps the predicate region). This avoids the BPE
         merge bug where re-tokenising `prompt` independently overshoots by 1
         when the trailing space of `prompt` merges into the first completion
         token.
      3. For each position ``i`` in the completion span, read
         ``prompt_logprobs[i][full_ids[i]].logprob`` — the log-prob of the
         specific ground-truth token id at that position, not the argmax.
      4. Sum across the completion span. NaN if any position's expected token id
         is not present in the returned dict (vLLM did not score the
         teacher-forced token).

    Returns one log-prob per pair, in input order.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=EVAL_MAX_MODEL_LEN,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
    )
    full_texts: list[str] = [p + c for p, c in pairs]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(full_texts, params)
    per_pair_logprob: list[float] = []
    for (prompt, _completion), out in zip(pairs, outputs, strict=True):
        full_text = prompt + _completion
        # offset_mapping resolves the completion-start token by char position;
        # robust to BPE merging the trailing prompt space into the first
        # completion token.
        full_enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        full_ids = full_enc["input_ids"]
        offsets = full_enc["offset_mapping"]
        completion_char_start = len(prompt)
        completion_start_tok_idx: int | None = None
        for tok_idx, (_cs, ce) in enumerate(offsets):
            # First token whose char span ENDS strictly after the prompt
            # boundary. We include any token that bridges the boundary (BPE
            # merge case) — its log-prob is the most honest accounting of
            # the completion-tail score.
            if ce > completion_char_start:
                completion_start_tok_idx = tok_idx
                break
        if completion_start_tok_idx is None:
            logger.warning(
                "could not locate completion-start token for pair (prompt-head=%r); "
                "offset_mapping length=%d",
                prompt[:60],
                len(offsets),
            )
            per_pair_logprob.append(float("nan"))
            continue

        plogs = out.prompt_logprobs or []
        if not plogs:
            per_pair_logprob.append(float("nan"))
            continue

        completion_lp = 0.0
        ok = True
        for idx in range(completion_start_tok_idx, len(full_ids)):
            if idx >= len(plogs):
                break
            lp_dict = plogs[idx]
            if lp_dict is None:
                # vLLM convention: position 0 has None (no prior context).
                continue
            tok_id = full_ids[idx]
            if not isinstance(lp_dict, dict):
                ok = False
                logger.warning(
                    "prompt_logprobs[%d] not a dict (got %s); cannot score teacher-forced token id %d",
                    idx,
                    type(lp_dict).__name__,
                    tok_id,
                )
                break
            lp_entry = lp_dict.get(tok_id)
            if lp_entry is None:
                # vLLM did not include the teacher-forced token id in its
                # returned dict (it was outside the top-1 + ground-truth set
                # for this version). FAIL LOUD via NaN — the caller filters
                # on logprob < -8.0 and a NaN propagates correctly to "drop
                # this candidate", which is the safe default per CLAUDE.md
                # fail-fast.
                ok = False
                logger.warning(
                    "vLLM prompt_logprobs[%d] missing teacher-forced token id %d "
                    "(present keys head: %s); marking pair logprob NaN.",
                    idx,
                    tok_id,
                    list(lp_dict.keys())[:5],
                )
                break
            lp_val = getattr(lp_entry, "logprob", None)
            if lp_val is None:
                # Older vLLM versions returned raw floats in the dict.
                try:
                    lp_val = float(lp_entry)
                except (TypeError, ValueError) as e:
                    ok = False
                    logger.warning(
                        "prompt_logprobs[%d][%d] entry has no .logprob attr and cannot "
                        "coerce to float: %s (entry repr: %r)",
                        idx,
                        tok_id,
                        e,
                        lp_entry,
                    )
                    break
            completion_lp += float(lp_val)
        per_pair_logprob.append(completion_lp if ok else float("nan"))

    # Teardown — narrow except per Minor #7.
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError as e:
        logger.debug("vllm.distributed.parallel_state imports unavailable: %s", e)
    else:
        destroy_model_parallel()
        destroy_distributed_environment()
    gc.collect()
    try:
        import torch
    except ImportError as e:
        logger.debug("torch import unavailable for empty_cache(): %s", e)
    else:
        torch.cuda.empty_cache()
    _reap_vllm_workers_and_assert_clean()
    return per_pair_logprob


def _vllm_answer_slot_entropy(
    base_model: str,
    triples: list[tuple[str, str, str]],
    *,
    gpu_id: int = 0,
    gpu_memory_utilization: float = 0.55,
    top_k: int = ENTROPY_TOP_K,
) -> list[dict[str, Any]]:
    """Measure base-model entropy at the CONSTRAINED value-slot (plan §4.2.5).

    For each ``(probe_question, canonical_value, answer_slot_carrier)``
    triple:

      1. Build a chat-formatted prompt as ``[system=assistant_prompt,
         user=probe_question, assistant_prefill=carrier_truncated_at_{VALUE}]``
         **with the canonical_value teacher-forced as the prefill's tail**
         so the full text spans ``... prefill + canonical_value``.
         The first value-slot position is at ``prefill_token_count`` of the
         tokenised full text; positions ``prefill_token_count ..
         prefill_token_count + len(value_ids) - 1`` cover the full canonical
         span. This is the load-bearing fix from MUST-FIX 2 round-1: WITHOUT
         the prefill an Instruct model's position-1 logprobs are
         sentence-starter posteriors ("The"/"It"/"I"), not value-slot prior.

      2. Run vLLM with ``prompt_logprobs=top_k`` over the full text and read:
           - position 0 of the value span → top-k distribution → Shannon /
             Renyi-2 / max_p / canonical_first_token_lp / top-5 surface.
           - positions 0..N-1 of the value span (N = len(value_ids)) →
             teacher-forced canonical-token logprobs summed, then
             length-conditionally normalised per plan §4.2.5:
               * N == 1 → ``canonical_answer_logprob`` = position-0 lp
                 (length-normalised = same value); signal VALID.
               * N == 2 → ``canonical_answer_logprob`` = sum / 2 (per-token
                 nats, length-normalised); signal VALID. Same threshold scale
                 applies because the dual-fixture calibration measures
                 per-token nats (the fixtures already include multi-BPE values).
               * N >= 3 → ``canonical_answer_logprob`` = NaN +
                 ``canonical_logprob_signal_dropped`` = True. K1 PASS falls
                 back to the 2-signal Shannon + max_p conjunction (the gate at
                 the call site treats NaN canonical as "well below threshold"
                 which is correct here: signal is unavailable, not failing).

         Compute:
           - ``shannon_entropy``: -Σ p_i log p_i over the top-k distribution
             at position-0 of the value slot (nats).
           - ``renyi_2_entropy``: -log Σ p_i² (collision entropy; textbook
             definition, more robust to long-tail mass).
           - ``max_p``: maximum probability mass on any single value-slot token.
           - ``canonical_answer_logprob``: length-conditionally normalised per
             above; NaN when signal dropped (3+ BPE) OR canonical token is
             outside the returned dict at any position in the span.
           - ``canonical_bpe_length``: N = len(value_ids); surfaced so the
             K1 gate at the call site can audit the disposition.
           - ``canonical_logprob_signal_dropped``: True for N >= 3.
           - ``top_5_tokens_with_logprob``: the model's top-5 value-slot
             guesses at position-0, surfaced to the user-facing candidate table.

      3. SENTENCE-STARTER SANITY CHECK: if the combined mass on the canonical
         sentence-starter set (``SENTENCE_STARTER_TOKENS``) at position-0
         of the value slot > ``SENTENCE_STARTER_MASS_THRESHOLD`` (0.30 per
         fixture) OR the top-1 token is a sentence-starter, the prefill is
         broken — the returned dict has ``prefill_failed=True`` and the
         entropy stats are computed nonetheless so the calling calibrator
         can surface the offending top-k list. Phase-0 then HALTs.

    Args:
        base_model: HF model id (will be loaded with the standard vLLM kwargs
            mirroring `_vllm_teacher_forced_logprob`).
        triples: ``(question, canonical_value, answer_slot_carrier)``
            triples; the carrier MUST contain exactly one ``{VALUE}``
            placeholder.

    Returns:
        One dict per triple in input order with the entropy stats above
        plus ``top_5_tokens_with_logprob``, ``prefill_failed``, and
        ``boundary_clean`` (the BPE-merge check for this carrier),
        ``canonical_bpe_length`` and ``canonical_logprob_signal_dropped``.
    """
    import math as _math

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=EVAL_MAX_MODEL_LEN,
        download_dir=os.environ.get("HF_HOME"),
        enforce_eager=True,
        # vLLM caps a request's prompt_logprobs at the engine's max_logprobs
        # (default 20). This calibration asks for prompt_logprobs=top_k
        # (ENTROPY_TOP_K=50) to read the value-slot distribution, so the engine
        # must allow at least that depth or generate() raises ValueError.
        max_logprobs=top_k,
    )

    # Build per-triple chat prompts with the assistant prefill AND the
    # canonical value teacher-forced as the prefill tail. We use the chat
    # template's `continue_final_message` path so vLLM doesn't re-inject an
    # assistant-turn-start prefix after the prefilled text. The prefill is
    # the carrier truncated at `{VALUE}` + the canonical value; the full
    # prompt then ends at the last canonical-value token, and
    # `prompt_logprobs=top_k` returns the per-position distribution over the
    # value span (load-bearing for the §4.2.5 length-conditional canonical-
    # logprob signal: 1-BPE → position-0; 2-BPE → length-normalised sum
    # over 2; ≥3-BPE → signal dropped).
    chat_prompts: list[str] = []
    boundaries: list[dict[str, Any]] = []
    for question, canonical_value, carrier in triples:
        prefix = _carrier_prefix(carrier)
        # Tokenization-boundary clean check (mirror the fixture invariants).
        value_ids = tokenizer.encode(canonical_value, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + canonical_value, add_special_tokens=False)
        # Stronger suffix check than the round-1 first-token-only assertion:
        # the full tokenisation of (prefix + canonical_value) MUST suffix-match
        # the standalone canonical_value tokenisation. This guards the
        # length-conditional canonical-logprob read at positions
        # len(prefix_ids) .. len(prefix_ids)+len(value_ids)-1 against BPE
        # merge boundaries that would shift the value-slot start.
        boundary_clean = (
            len(value_ids) >= 1
            and full_ids[: len(prefix_ids)] == prefix_ids
            and full_ids[-len(value_ids) :] == value_ids
        )
        messages = [
            {"role": "system", "content": ASSISTANT_PROMPT},
            {"role": "user", "content": question},
            # Teacher-force the canonical value: bake it as the prefill tail
            # so prompt_logprobs scores it length-conditionally.
            {"role": "assistant", "content": prefix + canonical_value},
        ]
        # `continue_final_message=True` + `add_generation_prompt=False` keeps
        # the prefill+value as the literal last bytes of the prompt (no extra
        # <|im_start|>assistant\n re-injection). The value span lives at
        # positions [len(prefix_chat_ids) .. len(prefix_chat_ids)+N-1] of
        # the full chat-formatted tokenisation, located by char-offset
        # mapping below.
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        boundaries.append(
            {
                "boundary_clean": boundary_clean,
                "value_first_token_id": value_ids[0] if value_ids else None,
                "value_token_ids": value_ids,
                "canonical_bpe_length": len(value_ids),
            }
        )
        chat_prompts.append(chat_prompt)

    # Use prompt_logprobs over the full text (prefill + canonical_value).
    # Generate 1 dummy token so vLLM is happy but we never read the
    # generation; all signal comes from `prompt_logprobs`.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=ENTROPY_N_ANSWER_TOKENS,
        prompt_logprobs=top_k,
    )
    outputs = llm.generate(chat_prompts, params)

    # Sentence-starter token-id set (resolve once via the tokenizer).
    sentence_starter_ids: set[int] = set()
    for tok_str in SENTENCE_STARTER_TOKENS:
        for variant in (tok_str, f" {tok_str}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                sentence_starter_ids.add(ids[0])

    results: list[dict[str, Any]] = []
    for (question, canonical_value, carrier), chat_prompt, out, boundary in zip(
        triples, chat_prompts, outputs, boundaries, strict=True
    ):
        canonical_bpe_length: int = boundary["canonical_bpe_length"]
        canonical_logprob_signal_dropped: bool = canonical_bpe_length >= 3
        # Locate the value-span position range in the chat-formatted prompt
        # tokenisation via char-offset mapping (mirror the
        # `_vllm_teacher_forced_logprob` pattern; robust to BPE-merge
        # boundaries that shift the value-slot start by ±1 vs the standalone
        # `len(prefix_ids)` count).
        value_char_start = chat_prompt.rfind(canonical_value)
        if value_char_start < 0:
            results.append(
                {
                    "question": question,
                    "canonical_value": canonical_value,
                    "carrier": carrier,
                    "shannon_entropy": float("nan"),
                    "renyi_2_entropy": float("nan"),
                    "max_p": float("nan"),
                    "canonical_answer_logprob": float("nan"),
                    "canonical_bpe_length": canonical_bpe_length,
                    "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
                    "top_5_tokens_with_logprob": [],
                    "prefill_failed": True,
                    "boundary_clean": boundary["boundary_clean"],
                    "_error": "could not locate canonical value in chat prompt",
                }
            )
            continue
        full_enc = tokenizer(chat_prompt, add_special_tokens=False, return_offsets_mapping=True)
        full_ids: list[int] = full_enc["input_ids"]
        offsets: list[tuple[int, int]] = full_enc["offset_mapping"]
        value_start_tok_idx: int | None = None
        for tok_idx, (_cs, ce) in enumerate(offsets):
            # First token whose char span ENDS strictly after the value-span
            # boundary in the chat prompt — same rule as the teacher-forced
            # helper. Handles BPE merges into the trailing prefill space.
            if ce > value_char_start:
                value_start_tok_idx = tok_idx
                break
        # value-span end index = value_start_tok_idx + canonical_bpe_length
        # (used implicitly by the position-loop below; computed inline so the
        # span-walk is the single source of truth for the iteration bound).

        plogs = getattr(out, "prompt_logprobs", None) or []
        if value_start_tok_idx is None or not plogs or value_start_tok_idx >= len(plogs):
            results.append(
                {
                    "question": question,
                    "canonical_value": canonical_value,
                    "carrier": carrier,
                    "shannon_entropy": float("nan"),
                    "renyi_2_entropy": float("nan"),
                    "max_p": float("nan"),
                    "canonical_answer_logprob": float("nan"),
                    "canonical_bpe_length": canonical_bpe_length,
                    "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
                    "top_5_tokens_with_logprob": [],
                    "prefill_failed": True,
                    "boundary_clean": boundary["boundary_clean"],
                    "_error": (
                        f"value-slot position not in prompt_logprobs range "
                        f"(value_start={value_start_tok_idx}, plogs_len={len(plogs)})"
                    ),
                }
            )
            continue

        pos0_dict = plogs[value_start_tok_idx]
        if pos0_dict is None or not isinstance(pos0_dict, dict):
            results.append(
                {
                    "question": question,
                    "canonical_value": canonical_value,
                    "carrier": carrier,
                    "shannon_entropy": float("nan"),
                    "renyi_2_entropy": float("nan"),
                    "max_p": float("nan"),
                    "canonical_answer_logprob": float("nan"),
                    "canonical_bpe_length": canonical_bpe_length,
                    "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
                    "top_5_tokens_with_logprob": [],
                    "prefill_failed": True,
                    "boundary_clean": boundary["boundary_clean"],
                    "_error": "vllm returned no prompt_logprobs at value-slot position",
                }
            )
            continue

        # Build top-k entries at position-0 of the value span.
        entries: list[tuple[int, float, str]] = []
        for tok_id, lp in pos0_dict.items():
            lp_val = float(getattr(lp, "logprob", lp))
            decoded = getattr(lp, "decoded_token", None)
            if decoded is None:
                try:
                    decoded = tokenizer.decode([tok_id])
                except Exception:
                    decoded = "<decode-failed>"
                else:
                    decoded = decoded or ""
            entries.append((int(tok_id), lp_val, decoded))
        entries.sort(key=lambda x: -x[1])

        # Compute Shannon + Renyi-2 + max_p over the top-k mass.
        probs = [_math.exp(lp) for _tid, lp, _dec in entries]
        if probs:
            shannon = float(-sum(p * _math.log(max(p, 1e-30)) for p in probs))
            collision = float(sum(p * p for p in probs))
            renyi_2 = float(-_math.log(max(collision, 1e-30)))
            max_p = float(max(probs))
        else:
            shannon = float("nan")
            renyi_2 = float("nan")
            max_p = float("nan")

        # Length-conditional canonical-answer logprob (plan §4.2.5):
        #   - N == 1: position-0 lp (length-normalised = same value).
        #   - N == 2: sum over positions 0..1 / 2 (per-token nats).
        #   - N >= 3: signal DROPPED → NaN + canonical_logprob_signal_dropped.
        # The teacher-forced read walks positions value_start_tok_idx ..
        # value_end_tok_idx-1; canonical lp is NaN if any position is missing
        # its ground-truth token in the returned dict (consistent with the
        # other teacher-forced helpers — caller treats NaN as "well below
        # threshold" which is the safe default for K1 PASS classification:
        # NaN canonical signal disables the constraint, not fails it).
        if canonical_logprob_signal_dropped:
            canonical_lp = float("nan")
        else:
            span_lp_sum = 0.0
            span_ok = True
            value_ids = boundary["value_token_ids"]
            for span_idx in range(canonical_bpe_length):
                pos_idx = value_start_tok_idx + span_idx
                if pos_idx >= len(plogs):
                    span_ok = False
                    break
                lp_dict = plogs[pos_idx]
                if lp_dict is None or not isinstance(lp_dict, dict):
                    span_ok = False
                    break
                expected_tok_id = value_ids[span_idx]
                lp_entry = lp_dict.get(expected_tok_id)
                if lp_entry is None:
                    span_ok = False
                    break
                lp_val = getattr(lp_entry, "logprob", None)
                if lp_val is None:
                    try:
                        lp_val = float(lp_entry)
                    except (TypeError, ValueError):
                        span_ok = False
                        break
                span_lp_sum += float(lp_val)
            canonical_lp = float(span_lp_sum / canonical_bpe_length) if span_ok else float("nan")

        # Top-5 + sentence-starter mass (at position-0 of the value span).
        top5 = [
            {"tok_id": tid, "logprob": lp_val, "decoded": dec} for tid, lp_val, dec in entries[:5]
        ]
        starter_mass = float(
            sum(_math.exp(lp) for tid, lp, _ in entries if tid in sentence_starter_ids)
        )
        top1_is_starter = bool(entries and entries[0][0] in sentence_starter_ids)
        prefill_failed = (
            (not boundary["boundary_clean"])
            or top1_is_starter
            or (starter_mass > SENTENCE_STARTER_MASS_THRESHOLD)
        )

        results.append(
            {
                "question": question,
                "canonical_value": canonical_value,
                "carrier": carrier,
                "shannon_entropy": shannon,
                "renyi_2_entropy": renyi_2,
                "max_p": max_p,
                "canonical_answer_logprob": canonical_lp,
                "canonical_bpe_length": canonical_bpe_length,
                "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
                "top_5_tokens_with_logprob": top5,
                "sentence_starter_mass": starter_mass,
                "top1_is_starter": top1_is_starter,
                "prefill_failed": prefill_failed,
                "boundary_clean": boundary["boundary_clean"],
            }
        )

    # Teardown — mirror `_vllm_teacher_forced_logprob`.
    del llm
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError as e:
        logger.debug("vllm.distributed.parallel_state imports unavailable: %s", e)
    else:
        destroy_model_parallel()
        destroy_distributed_environment()
    gc.collect()
    try:
        import torch
    except ImportError as e:
        logger.debug("torch import unavailable for empty_cache(): %s", e)
    else:
        torch.cuda.empty_cache()
    _reap_vllm_workers_and_assert_clean()
    return results


def _percentile(values: list[float], q: float) -> float:
    """Simple percentile (q in 0..1; linear interpolation; ignores NaN)."""
    clean = sorted(v for v in values if not (isinstance(v, float) and v != v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return float(clean[0])
    pos = q * (len(clean) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(clean) - 1)
    frac = pos - lo
    return float(clean[lo] * (1 - frac) + clean[hi] * frac)


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


def _phase0_calibrate_entropy_thresholds(
    base_model: str,
    *,
    gpu_id: int = 0,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    """Run the dual-fixture entropy calibration (plan §4.2.5).

    Loads the two 10-item fixture sets (KNOWN_PRIOR + KNOWN_ZERO_PRIOR),
    measures answer-slot entropy on each via the same prefill design that
    candidates use, and computes the per-run thresholds:

      - ``THRESHOLD_SHANNON = max(P25(zero_prior.shannon), P75(known_prior.shannon) + 0.5)``
      - ``THRESHOLD_MAX_P = min(P75(zero_prior.max_p), P25(known_prior.max_p) - 0.05)``
      - ``THRESHOLD_CANONICAL = max(P75(shuffled.canonical_answer_logprob), -6.0)``

    Fail-loud on:
      - calibration gap < ``MIN_CALIBRATION_GAP_NATS`` (0.5) between the
        two fixture-set Shannon medians;
      - sentence-starter mass averaged across the 20 fixtures > 0.20;
      - any single fixture trips ``prefill_failed = True``.

    Returns a dict with the thresholds + the per-fixture audit + the
    histogram input for the §6.6 #5 diagnostic figure.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("entropy calibration already cached -> %s; reloading", cache_path)
        return json.loads(cache_path.read_text())

    # 1. Build the three fixture sets that share the prefill.
    known_prior_triples = list(KNOWN_PRIOR_FIXTURE)
    known_zero_prior_triples = list(KNOWN_ZERO_PRIOR_FIXTURE)
    shuffled_triples = list(build_random_shuffled_fixture())

    logger.info(
        "entropy calibration: probing %d known-prior + %d known-zero-prior + %d shuffled fixtures",
        len(known_prior_triples),
        len(known_zero_prior_triples),
        len(shuffled_triples),
    )

    # Run all three sets in ONE vLLM session to avoid load/teardown overhead.
    all_triples = known_prior_triples + known_zero_prior_triples + shuffled_triples
    all_results = _vllm_answer_slot_entropy(base_model, all_triples, gpu_id=gpu_id)
    n_kp = len(known_prior_triples)
    n_kz = len(known_zero_prior_triples)
    kp_results = all_results[:n_kp]
    kz_results = all_results[n_kp : n_kp + n_kz]
    sh_results = all_results[n_kp + n_kz :]

    # 2. Per-fixture stats.
    def _stats(results: list[dict[str, Any]]) -> dict[str, Any]:
        sh = [r["shannon_entropy"] for r in results]
        mp = [r["max_p"] for r in results]
        cl = [r["canonical_answer_logprob"] for r in results]
        return {
            "shannon": {
                "median": _median(sh),
                "p25": _percentile(sh, 0.25),
                "p75": _percentile(sh, 0.75),
                "values": sh,
            },
            "max_p": {
                "median": _median(mp),
                "p25": _percentile(mp, 0.25),
                "p75": _percentile(mp, 0.75),
                "values": mp,
            },
            "canonical_answer_logprob": {
                "median": _median(cl),
                "p25": _percentile(cl, 0.25),
                "p75": _percentile(cl, 0.75),
                "values": cl,
            },
        }

    kp_stats = _stats(kp_results)
    kz_stats = _stats(kz_results)
    sh_stats = _stats(sh_results)

    # 3. Threshold computation per plan §4.2.5.
    t_shannon = max(
        kz_stats["shannon"]["p25"],
        kp_stats["shannon"]["p75"] + 0.5,
    )
    t_max_p = min(
        kz_stats["max_p"]["p75"],
        kp_stats["max_p"]["p25"] - 0.05,
    )
    t_canonical = max(
        sh_stats["canonical_answer_logprob"]["p75"],
        T_CANONICAL_FLOOR_NATS,
    )

    # 4. Gap check (load-bearing): zero-prior median must lead known-prior
    # median by at least MIN_CALIBRATION_GAP_NATS (0.5 nats).
    gap_nats = kz_stats["shannon"]["median"] - kp_stats["shannon"]["median"]
    gap_ok = gap_nats >= MIN_CALIBRATION_GAP_NATS

    # 5. Sentence-starter sanity check.
    starter_masses_all = [r.get("sentence_starter_mass", 0.0) for r in kp_results + kz_results]
    starter_mean_mass = (
        sum(starter_masses_all) / max(1, len(starter_masses_all)) if starter_masses_all else 1.0
    )
    any_prefill_failed = any(r.get("prefill_failed") for r in kp_results + kz_results)
    starter_ok = (starter_mean_mass <= SENTENCE_STARTER_MEAN_MASS_THRESHOLD) and (
        not any_prefill_failed
    )

    audit = {
        "phase": "fact-candidates:entropy-calibration",
        "timestamp": _now_iso(),
        "thresholds": {
            "T_SHANNON": t_shannon,
            "T_MAX_P": t_max_p,
            "T_CANONICAL": t_canonical,
            "MIN_CALIBRATION_GAP_NATS": MIN_CALIBRATION_GAP_NATS,
            "SENTENCE_STARTER_MEAN_MASS_THRESHOLD": SENTENCE_STARTER_MEAN_MASS_THRESHOLD,
            "T_CANONICAL_FLOOR_NATS": T_CANONICAL_FLOOR_NATS,
        },
        "gap_check": {
            "median_known_prior_shannon": kp_stats["shannon"]["median"],
            "median_known_zero_prior_shannon": kz_stats["shannon"]["median"],
            "gap_nats": gap_nats,
            "gap_ok": gap_ok,
        },
        "sentence_starter_check": {
            "mean_mass_across_20_fixtures": starter_mean_mass,
            "any_prefill_failed": any_prefill_failed,
            "ok": starter_ok,
            # NON-BLOCKING as of 2026-06-02 (user decision): when ok=False the
            # answer-slot entropy is confounded by sentence-form prior, so the
            # zero-prior gate's absolute values are inflated. The run continues
            # anyway, but every downstream zero-prior claim MUST disclose this
            # confound in the write-up.
            "blocking": False,
        },
        "known_prior": kp_stats,
        "known_zero_prior": kz_stats,
        "shuffled_canonical_logprob": sh_stats["canonical_answer_logprob"],
        "per_fixture_known_prior": kp_results,
        "per_fixture_known_zero_prior": kz_results,
        "per_fixture_shuffled": sh_results,
    }
    if cache_path is not None:
        _write_json(cache_path, audit)

    # 6. Fail-loud per plan §4.2.5.
    #     The sentence-starter sanity check is downgraded from a hard halt to a
    #     NON-BLOCKING warning per user decision (2026-06-02): proceed despite
    #     the sentence-form confound. The gap check below STAYS fail-loud — it
    #     guards that the entropy can still separate known from unknown at all,
    #     which is the floor below which the experiment is meaningless.
    if not starter_ok:
        logger.warning(
            "Phase 0 K1 sentence-starter sanity check NOT OK (NON-BLOCKING, "
            "user override 2026-06-02): mean starter mass across 20 fixtures = "
            "%.3f (threshold %s), any prefill_failed = %s. The answer-slot "
            "entropy is confounded by sentence-form prior — the zero-prior "
            "gate's absolute values are inflated, so every downstream zero-prior "
            "claim MUST disclose this confound. Per-fixture top-5 lists: %s.",
            starter_mean_mass,
            SENTENCE_STARTER_MEAN_MASS_THRESHOLD,
            any_prefill_failed,
            cache_path,
        )
    if not gap_ok:
        raise RuntimeError(
            "Phase 0 K1 entropy calibration FAILED: gap between known-prior "
            f"and known-zero-prior Shannon medians = {gap_nats:.3f} nats "
            f"< required {MIN_CALIBRATION_GAP_NATS} nats. The K1 measurement "
            "cannot reliably separate known from unknown on this model. "
            "Halt with epm:failure v1 / failure_class: data / status:blocked "
            "/ reason: phase0_entropy_calibration_no_gap. See the histogram "
            f"input at {cache_path}."
        )

    logger.info(
        "entropy calibration PASS: T_SHANNON=%.3f T_MAX_P=%.3f T_CANONICAL=%.3f gap=%.3f nats",
        t_shannon,
        t_max_p,
        t_canonical,
        gap_nats,
    )
    return audit


class WebSearchUnavailableError(RuntimeError):
    """Raised when the Anthropic ``web_search_20250305`` tool is unavailable.

    Distinguishes (tool worked, returned 0 hits) from (tool unavailable,
    cannot determine whether contradicting info exists). Critical for
    Phase 0 contradiction-check correctness: a silent empty-list from a
    broken tool would let EVERY candidate pass the "internet-uncontested"
    filter (Blocker #2, code-review v1).
    """


def _websearch_snippets_via_anthropic(query: str, n: int = 5) -> list[str]:
    """Issue a web search via Anthropic's tool-use (web_search_20250305 tool).

    Returns a list of result snippets (title + URL + brief excerpt strings).
    Cached at the caller via Phase 0's contradiction_check.json so re-runs
    don't re-search.

    Three outcomes, distinguishable:

    1. **Tool works + returns ≥1 result:** returns the snippet list.
    2. **Tool works + returns 0 hits:** returns ``[]`` (genuine empty result).
       The contradiction judge then sees "no snippets" and the caller treats
       this as a genuine no-contradiction signal (which is correct: a query
       that returns zero hits in the top-5 IS evidence that no source asserts
       a contradicting fact, modulo search-engine recall).
    3. **Tool unavailable / errors:** raises :class:`WebSearchUnavailableError`.
       Caller MUST NOT treat this as no-contradiction; it must annotate the
       candidate's contradiction_verdict as ``{"bypassed": true}`` and the
       caller-side audit logic excludes the figure from the "uncontradicted"
       pool. If ALL figures bypass, Phase 0 halts with
       ``epm:failure v1 / failure_class: infra / reason: web_search_unavailable``.

    The distinction is made by inspecting the assistant's response: a working
    tool emits a ``tool_use`` block with ``name=web_search`` BEFORE the text
    response. An unavailable tool either errors at the API layer or returns
    a text-only response containing self-reports like "I don't have web
    search" / "web_search is not available" / "I cannot search the web".
    """
    client = _anthropic_client()
    try:
        msg = client.messages.create(
            model=FABRICATE_MODEL,
            max_tokens=2048,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Search the web for: {query}\n\n"
                        f"Return up to {n} top results as a JSON list of "
                        f'{{"title", "url", "snippet"}} objects in a code block. '
                        "If your search returned ZERO hits (a genuine empty result), "
                        "return an empty JSON list []. Do NOT return an empty list "
                        "if the web_search tool failed to invoke — in that case, "
                        "say plainly that the tool is unavailable so the caller "
                        "can distinguish."
                    ),
                }
            ],
        )
    except Exception as e:
        raise WebSearchUnavailableError(
            f"Anthropic API call failed for web_search query {query!r}: {e!r}. "
            "Cannot distinguish 'no contradiction online' from 'tool broken'; "
            "caller must bypass this candidate or halt Phase 0."
        ) from e

    # Determine whether the assistant actually invoked the tool.
    used_tool = any(
        getattr(b, "type", None) in ("tool_use", "server_tool_use") for b in msg.content
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    # Sniff for self-reported tool-unavailable patterns (model says it can't
    # search, even when no API error fires — happens when the tool isn't
    # enabled for the org's API tier).
    unavailable_markers = (
        "tool is not available",
        "tool is unavailable",
        "tool failed",
        "i don't have web search",
        "i can't search the web",
        "i cannot search the web",
        "i'm unable to search",
        "no web search capability",
        "i don't have access to web search",
    )
    text_low = text.lower()
    if any(m in text_low for m in unavailable_markers):
        raise WebSearchUnavailableError(
            f"web_search tool self-reported unavailable for query {query!r}: "
            f"text-head={text[:200]!r}. Tier may not have the tool enabled."
        )

    # If the assistant didn't use the tool AND didn't self-report unavailable,
    # something is structurally off — refuse to silently degrade.
    if not used_tool:
        raise WebSearchUnavailableError(
            f"web_search tool was NOT invoked for query {query!r} and the "
            f"assistant did not self-report unavailability. Response text-head: "
            f"{text[:200]!r}. Refusing to treat this as 'no contradiction'."
        )

    # Tool was invoked. Now parse snippet list — empty list is GENUINE.
    bracket = text.find("[")
    if bracket == -1:
        # Tool ran but the assistant didn't emit a JSON array — return
        # empty (the search ran; the assistant just didn't format).
        logger.info(
            "web_search ran for %r but no JSON array parsed; treating as 0 genuine hits.",
            query,
        )
        return []
    try:
        # raw_decode parses the first complete JSON array and ignores trailing
        # prose. A greedy [.*] + json.loads would over-match and spuriously
        # fail on trailing content, dropping to the 0-hits path — which would
        # let a genuinely-contradicted figure slip through as 'uncontested'.
        results, _ = json.JSONDecoder().raw_decode(text[bracket:])
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(
            "web_search results JSON-decode failed for %r (%s); treating as 0 genuine hits.",
            query,
            e,
        )
        return []
    snippets: list[str] = []
    for r in results[:n]:
        if not isinstance(r, dict):
            continue
        snippets.append(f"[{r.get('title', '')}] ({r.get('url', '')}): {r.get('snippet', '')}")
    return snippets


def phase_fact_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Phase 0 (v5) — obscure US physical place / object + invented mundane attribute USER GATE.

    Steps (plan §4.2):
      1. Source 30 candidate entities (seed-curated + Sonnet-proposed; dedupe).
      2. Entity-recognition probe (≥ 2/3 PASS via base Qwen → Haiku judge).
      3. Invented-mundane-attribute drafting (Sonnet 4.5, 3 drafts/entity).
      4. **K1 dual-fixture entropy CALIBRATION** (NEW v3): set
         T_SHANNON / T_MAX_P / T_CANONICAL from KNOWN_PRIOR + KNOWN_ZERO_PRIOR
         fixture sets via the SAME constrained-slot prefill design used on
         candidates. Fail-loud on no-gap OR sentence-starter-mass violation.
      5. K1 zero-prior gate per (entity, attribute) via
         ``_vllm_answer_slot_entropy`` on the picked answer_slot_question +
         answer_slot_value + answer_slot_carrier triple.
      6. No-online-contradiction check (3 queries × 5 snippets, Sonnet judge).
      7. Rank + trim to N_ENTITIES_FILTERED; emit epm:fact-candidates v1; EXIT.

    Idempotent: if ``fact_pick.json`` exists, return immediately; if
    ``candidates.json`` exists but no pick yet, re-post the marker.
    """
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = PHASE0_DIR / "candidates.json"
    pick_path = PHASE0_DIR / "fact_pick.json"

    if pick_path.exists():
        logger.info("fact_pick.json already exists; skipping Phase 0")
        return {
            "phase": "fact-candidates",
            "skipped": True,
            "pick": json.loads(pick_path.read_text()),
        }

    if candidates_path.exists():
        cands = json.loads(candidates_path.read_text())
        rows = cands.get("candidates", cands) if isinstance(cands, dict) else cands
        logger.info("candidates.json already exists with %d rows — re-posting marker", len(rows))
        _post_fact_candidates_marker(cands)
        logger.info("posted epm:fact-candidates v1; EXITing for user pick")
        sys.exit(0)

    # ── 1. Source candidate entities (seed + Sonnet-proposed; dedupe). ────────
    logger.info("Phase 0 step 1: sourcing candidate entities")
    seed_path = PHASE0_DIR / "phase0_seed_entities.json"
    _write_json(
        seed_path,
        {
            "entities": [
                {"entity_descriptor": d, "town": t, "state": s}
                for d, t, s in CANDIDATE_ENTITIES_SEED
            ],
            "timestamp": _now_iso(),
        },
    )
    sonnet_cache = PHASE0_DIR / "sonnet_proposed_entities.json"
    if sonnet_cache.exists():
        sonnet_entities = json.loads(sonnet_cache.read_text())["entities"]
    else:
        n_proposed = max(0, N_ENTITIES_RAW - len(CANDIDATE_ENTITIES_SEED))
        if n_proposed > 0:
            response = _sonnet_json_call(
                SONNET_PROPOSE_ENTITIES_PROMPT.format(n=n_proposed),
                model=FABRICATE_MODEL,
                max_tokens=3072,
            )
            sonnet_entities = response.get("entities", [])
        else:
            sonnet_entities = []
        _write_json(sonnet_cache, {"entities": sonnet_entities, "timestamp": _now_iso()})

    seen: set[str] = set()
    entities: list[dict[str, str]] = []
    for d, t, s in CANDIDATE_ENTITIES_SEED:
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append({"entity_descriptor": d, "town": t, "state": s})
    for entry in sonnet_entities:
        if not isinstance(entry, dict):
            continue
        d = entry.get("entity_descriptor", "")
        if not d or d.lower() in seen:
            continue
        seen.add(d.lower())
        entities.append(
            {
                "entity_descriptor": d,
                "town": entry.get("town", ""),
                "state": entry.get("state", ""),
            }
        )
    entities = entities[:N_ENTITIES_RAW]
    logger.info("collected %d candidate entities", len(entities))

    # ── 2. Entity-recognition probe (≥ 2/3 PASS). ─────────────────────────────
    logger.info("Phase 0 step 2: entity-recognition probe for %d entities", len(entities))
    recognition_path = PHASE0_DIR / "recognition_audit.json"
    if recognition_path.exists():
        recognition = json.loads(recognition_path.read_text())
    else:
        recognition = _run_entity_recognition_probe(
            [e["entity_descriptor"] for e in entities], gpu_id=args.gpu_id
        )
        _write_json(recognition_path, recognition)
    recognized = [
        e for e in entities if recognition.get(e["entity_descriptor"], {}).get("score", 0) >= 2
    ]
    logger.info("recognised %d/%d entities", len(recognized), len(entities))
    if not recognized:
        raise RuntimeError(
            "Phase 0: 0 entities passed entity-recognition probe (≥2/3 PASS); "
            "widen seed pool or escalate via epm:failure v1 / failure_class: data."
        )

    # ── 3. Invented-mundane-attribute drafting (per recognised entity). ──────
    logger.info(
        "Phase 0 step 3: drafting invented mundane attributes for %d entities", len(recognized)
    )
    drafts_path = PHASE0_DIR / "attribute_drafts.json"
    if drafts_path.exists():
        drafts = json.loads(drafts_path.read_text())
    else:
        drafts = {}
        for entry in recognized:
            d = entry["entity_descriptor"]
            best: dict[str, Any] | None = None
            for _ in range(N_ATTRIBUTE_DRAFTS_PER_ENTITY):
                try:
                    draft = _sonnet_json_call(
                        INVENTED_MUNDANE_ATTRIBUTE_PROMPT.format(entity_descriptor=d),
                        model=FABRICATE_MODEL,
                        max_tokens=768,
                    )
                except RuntimeError as e:
                    logger.warning("attribute draft for %r failed: %s", d, e)
                    continue
                q = draft.get("draft_quality_self_score", "low")
                if best is None or {"high": 3, "medium": 2, "low": 1}.get(q, 0) > {
                    "high": 3,
                    "medium": 2,
                    "low": 1,
                }.get(best.get("draft_quality_self_score", "low"), 0):
                    best = draft
                if best.get("draft_quality_self_score") == "high":
                    break
            if best is None or best.get("draft_quality_self_score") == "low":
                logger.info("dropping %r — no high/medium quality attribute draft", d)
                continue
            # Validate the drafted carrier shape (load-bearing for K1).
            carrier = best.get("answer_slot_carrier", "")
            value = best.get("answer_slot_value", "")
            slot_q = best.get("attribute_slot_question", "")
            if not carrier or carrier.count("{VALUE}") != 1 or not value or not slot_q:
                logger.warning(
                    "dropping %r — draft missing/malformed K1 slot fields "
                    "(value=%r carrier=%r slot_q=%r)",
                    d,
                    value,
                    carrier,
                    slot_q,
                )
                continue
            best["town"] = entry["town"]
            best["state"] = entry["state"]
            drafts[d] = best
        _write_json(drafts_path, drafts)
    logger.info("kept %d entities with drafted mundane attributes", len(drafts))

    # ── 4. K1 entropy CALIBRATION (NEW v3 — the structural fix). ──────────────
    logger.info("Phase 0 step 4: K1 entropy calibration on dual fixture sets")
    # First validate the calibration fixtures vs the live tokenizer.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    assert_fixture_invariants(tok)
    calibration_path = PHASE0_DIR / "entropy_calibration.json"
    calibration = _phase0_calibrate_entropy_thresholds(
        BASE_MODEL, gpu_id=args.gpu_id, cache_path=calibration_path
    )
    t_shannon = calibration["thresholds"]["T_SHANNON"]
    t_max_p = calibration["thresholds"]["T_MAX_P"]
    t_canonical = calibration["thresholds"]["T_CANONICAL"]

    # ── 5. Per-pair K1 answer-slot entropy gate. ──────────────────────────────
    logger.info(
        "Phase 0 step 5: K1 answer-slot entropy gate for %d (entity, attribute) pairs",
        len(drafts),
    )
    entropy_path = PHASE0_DIR / "attribute_entropy_audit.json"
    if entropy_path.exists():
        entropy_audit = json.loads(entropy_path.read_text())
    else:
        triples: list[tuple[str, str, str]] = []
        keys: list[str] = []
        for d, draft in drafts.items():
            triples.append(
                (
                    draft["attribute_slot_question"],
                    draft["answer_slot_value"],
                    draft["answer_slot_carrier"],
                )
            )
            keys.append(d)
        if triples:
            entropy_results = _vllm_answer_slot_entropy(BASE_MODEL, triples, gpu_id=args.gpu_id)
        else:
            entropy_results = []
        entropy_audit = {}
        for d, result in zip(keys, entropy_results, strict=True):
            sh = result["shannon_entropy"]
            mp = result["max_p"]
            cl = result["canonical_answer_logprob"]
            bpe_len = int(result.get("canonical_bpe_length", 1))
            signal_dropped = bool(result.get("canonical_logprob_signal_dropped", False))
            # K1 PASS — length-conditional canonical-logprob policy
            # (plan §4.2.5):
            #   - bpe_len ∈ {1, 2}: 3-signal conjunction. Shannon ≥ T_SHANNON
            #     AND max_p ≤ T_MAX_P AND canonical_logprob ≤ T_CANONICAL
            #     (or NaN, which the entropy reader returns when the
            #     teacher-forced canonical token is outside the top-k at
            #     some span position; treated as "well below threshold"
            #     i.e. signal-unavailable does not fail the gate).
            #   - bpe_len ≥ 3: canonical-logprob signal is DROPPED (the
            #     length-normalised per-token nats become noise dominated
            #     by intermediate-position posteriors). K1 PASS reduces to
            #     2-signal Shannon + max_p; flagged in the audit so the
            #     downstream pick gate can require --allow-multi-bpe-answer.
            shannon_ok = (sh == sh) and sh >= t_shannon  # NaN check (sh != NaN)
            max_p_ok = (mp == mp) and mp <= t_max_p
            if signal_dropped:
                canonical_ok = True
                k1_signal_basis = "2-signal (Shannon + max_p; canonical dropped, bpe≥3)"
            else:
                canonical_ok = (cl != cl) or (cl <= t_canonical)
                k1_signal_basis = f"3-signal (Shannon + max_p + canonical_logprob; bpe={bpe_len})"
            k1_pass = shannon_ok and max_p_ok and canonical_ok
            entropy_audit[d] = {
                **result,
                "shannon_ok": shannon_ok,
                "max_p_ok": max_p_ok,
                "canonical_ok": canonical_ok,
                "k1_pass": k1_pass,
                "k1_signal_basis": k1_signal_basis,
                "thresholds": {
                    "T_SHANNON": t_shannon,
                    "T_MAX_P": t_max_p,
                    "T_CANONICAL": t_canonical,
                },
            }
        _write_json(entropy_path, entropy_audit)
    k1_passed = [(d, drafts[d]) for d, info in entropy_audit.items() if info.get("k1_pass")]
    logger.info(
        "%d/%d (entity, attribute) pairs pass K1 entropy gate (T_SHANNON=%.3f T_MAX_P=%.3f T_CANONICAL=%.3f)",
        len(k1_passed),
        len(entropy_audit),
        t_shannon,
        t_max_p,
        t_canonical,
    )
    if not k1_passed:
        raise RuntimeError(
            f"Phase 0 K1 entropy gate: 0/{len(entropy_audit)} pairs passed the "
            "3-signal conjunction. Either widen the raw entity pool (30→60 per "
            "§13 deviations-allowed) OR drop the canonical-logprob signal "
            "(proceed with Shannon + max_p only) OR re-draft attributes with "
            "more diffuse answer slots. Halt with epm:failure v1 / "
            "failure_class: data / reason: phase0_k1_zero_candidates."
        )

    # ── 6. No-online-contradiction check (per pair). ──────────────────────────
    logger.info("Phase 0 step 6: no-contradiction web check for %d pairs", len(k1_passed))
    contradiction_path = PHASE0_DIR / "contradiction_check.json"
    if contradiction_path.exists():
        contradiction_audit = json.loads(contradiction_path.read_text())
    else:
        contradiction_audit = {}
        for d, draft in k1_passed:
            attr_sent = draft["attribute_sentence"]
            attr_short = draft.get("attribute_short", attr_sent[:60])
            queries = [
                f'"{attr_sent}"',
                f"{d} {attr_short}",
                f"{d} physical features",
            ]
            snippet_blocks: list[str] = []
            queries_succeeded = 0
            queries_bypassed = 0
            bypass_reasons: list[str] = []
            for q in queries:
                try:
                    snippets = _websearch_snippets_via_anthropic(q, n=5)
                    snippet_blocks.extend(snippets)
                    queries_succeeded += 1
                except WebSearchUnavailableError as e:
                    queries_bypassed += 1
                    bypass_reasons.append(str(e)[:200])
                    logger.warning("web_search bypass for entity=%r query=%r: %s", d, q, e)
            if queries_succeeded == 0:
                contradiction_audit[d] = {
                    "snippets_collected": 0,
                    "queries_succeeded": 0,
                    "queries_bypassed": queries_bypassed,
                    "bypassed": True,
                    "bypass_reasons": bypass_reasons,
                    "verdict": {
                        "contradicts": True,
                        "reason": "WEB-SEARCH-BYPASSED — tool unavailable for all "
                        "queries on this entity; excluding to preserve "
                        "internet-uncontested invariant.",
                        "bypassed": True,
                    },
                    "ts": _now_iso(),
                }
                continue
            snippets_block = "\n".join(snippet_blocks) if snippet_blocks else "[no snippets]"
            verdict_prompt = CONTRADICTION_JUDGE_PROMPT.format(
                entity_descriptor=d, attribute=attr_sent, snippets=snippets_block
            )
            try:
                verdict = _sonnet_json_call(verdict_prompt, model=FABRICATE_MODEL, max_tokens=256)
            except RuntimeError as e:
                logger.warning(
                    "contradiction-check judge failed for %r: %s; flagging as suspect", d, e
                )
                verdict = {"contradicts": True, "reason": f"judge_call_failed: {e}"}
            contradiction_audit[d] = {
                "snippets_collected": len(snippet_blocks),
                "queries_succeeded": queries_succeeded,
                "queries_bypassed": queries_bypassed,
                "bypassed": False,
                "bypass_reasons": bypass_reasons,
                "verdict": verdict,
                "ts": _now_iso(),
            }
        _write_json(contradiction_path, contradiction_audit)

    n_bypassed = sum(1 for info in contradiction_audit.values() if info.get("bypassed", False))
    if n_bypassed == len(contradiction_audit) and n_bypassed > 0:
        raise RuntimeError(
            f"Phase 0 K-infra: web_search tool unavailable for ALL "
            f"{n_bypassed}/{len(contradiction_audit)} entities. The "
            "internet-uncontested filter cannot fire; halt with epm:failure v1 / "
            "failure_class: infra / reason: web_search_unavailable. Fix: enable "
            "the `web_search_20250305` tool on the Anthropic API tier and re-run "
            "phase fact-candidates (delete contradiction_check.json first)."
        )

    uncontradicted = [
        (d, draft)
        for d, draft in k1_passed
        if not contradiction_audit.get(d, {}).get("bypassed", False)
        and not contradiction_audit.get(d, {}).get("verdict", {}).get("contradicts", True)
    ]
    logger.info(
        "%d/%d pairs uncontradicted (%d bypassed via web_search unavailable)",
        len(uncontradicted),
        len(k1_passed),
        n_bypassed,
    )
    if not uncontradicted:
        raise RuntimeError(
            f"Phase 0: 0/{len(k1_passed)} pairs passed contradiction check "
            f"({n_bypassed} bypassed via web_search unavailability, "
            f"{len(k1_passed) - n_bypassed} judged 'contradicts'). Either re-draft "
            "more specific attributes OR fix the web_search tool availability. "
            "DO NOT silently advance."
        )

    # ── 7. Rank + trim to N_ENTITIES_FILTERED. ────────────────────────────────
    def _rank_key(da: tuple[str, dict[str, Any]]) -> float:
        d, _ = da
        recog = recognition.get(d, {}).get("score", 0) / 3.0
        info = entropy_audit.get(d, {})
        shannon = info.get("shannon_entropy", 0.0) or 0.0
        return -(recog + 0.5 * shannon)  # higher recog + higher shannon = better

    ranked = sorted(uncontradicted, key=_rank_key)[:N_ENTITIES_FILTERED]

    # ── 8. Build final candidate payload. ─────────────────────────────────────
    final_candidates: list[dict[str, Any]] = []
    for d, draft in ranked:
        ea = entropy_audit.get(d, {})
        final_candidates.append(
            {
                "entity_descriptor": d,
                "town": draft.get("town", ""),
                "state": draft.get("state", ""),
                "attribute_sentence": draft["attribute_sentence"],
                "attribute_short": draft.get("attribute_short", ""),
                "attribute_slot_question": draft["attribute_slot_question"],
                "answer_slot_value": draft["answer_slot_value"],
                "answer_slot_carrier": draft["answer_slot_carrier"],
                "answer_slot_word_count": draft.get("answer_slot_word_count", ""),
                "domain_fit": draft.get("domain_fit", ""),
                "safety_check": draft.get("safety_check", ""),
                "recognition_score": recognition.get(d, {}).get("score", 0),
                "shannon_entropy_nats": ea.get("shannon_entropy", float("nan")),
                "renyi_2_entropy_nats": ea.get("renyi_2_entropy", float("nan")),
                "max_p": ea.get("max_p", float("nan")),
                "canonical_answer_logprob_nats": ea.get("canonical_answer_logprob", float("nan")),
                "top_5_tokens_with_logprob": ea.get("top_5_tokens_with_logprob", []),
                "k1_pass": ea.get("k1_pass", False),
                "contradiction_verdict": contradiction_audit.get(d, {}).get(
                    "verdict", {"contradicts": None}
                ),
            }
        )

    payload = {
        "phase": "fact-candidates",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "n_entities_raw": len(entities),
        "n_recognised": len(recognized),
        "n_with_drafts": len(drafts),
        "n_k1_passed": len(k1_passed),
        "n_uncontradicted": len(uncontradicted),
        "n_final": len(final_candidates),
        "k1_calibration": {
            "T_SHANNON": t_shannon,
            "T_MAX_P": t_max_p,
            "T_CANONICAL": t_canonical,
            "gap_nats": calibration["gap_check"]["gap_nats"],
            "median_known_prior_shannon": calibration["gap_check"]["median_known_prior_shannon"],
            "median_known_zero_prior_shannon": calibration["gap_check"][
                "median_known_zero_prior_shannon"
            ],
            "calibration_path": str(calibration_path),
        },
        "candidates": final_candidates,
    }
    _write_json(candidates_path, payload)
    _post_fact_candidates_marker(payload)
    logger.info("posted epm:fact-candidates v1; EXITing for user pick")
    sys.exit(0)


def _run_entity_recognition_probe(
    entity_descriptors: list[str], *, gpu_id: int = 0
) -> dict[str, dict[str, Any]]:
    """For each entity descriptor, run 3 paraphrased recognition prompts;
    ≥ 2/3 PASS = recognised.

    The base-model completions are generated in ONE vLLM batch
    (3 × len(entity_descriptors)) then judged by Haiku one-by-one.
    """
    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, int]] = []
    for d in entity_descriptors:
        for i, tpl in enumerate(RECOGNITION_PROMPTS):
            prompts.append((ASSISTANT_PROMPT, tpl.format(entity_descriptor=d)))
            keys.append((d, i))
    completions = _vllm_complete_simple(
        BASE_MODEL, prompts, temperature=0.0, max_new_tokens=256, gpu_id=gpu_id
    )
    by_entity: dict[str, dict[str, Any]] = {
        d: {"per_prompt": [], "score": 0} for d in entity_descriptors
    }
    for (d, i), completion in zip(keys, completions, strict=True):
        verdict = _haiku_judge_call(
            RECOGNITION_RUBRIC_PROMPT.format(entity_descriptor=d),
            (
                f"Prompt:\n{RECOGNITION_PROMPTS[i].format(entity_descriptor=d)}\n\n"
                f"Completion:\n{completion}"
            ),
        )
        passed = bool(verdict.get("pass", False))
        by_entity[d]["per_prompt"].append(
            {"prompt_idx": i, "pass": passed, "reason": verdict.get("reason", "")}
        )
        if passed:
            by_entity[d]["score"] += 1
    return by_entity


def _post_fact_candidates_marker(payload: Any) -> None:
    """Build a Markdown table and post epm:fact-candidates v1 (v5 retargeted)."""
    candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    rows = candidates if isinstance(candidates, list) else candidates.get("candidates", [])
    n = len(rows)
    k1c = payload.get("k1_calibration", {}) if isinstance(payload, dict) else {}

    table_lines: list[str] = [
        "| # | Entity | Type | Invented attribute (short) | Slot question | Canonical answer | recog | shannon | max_p | top_5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(rows, start=1):
        d = c.get("entity_descriptor", "")
        # Extract entity type (post office / courthouse / library / etc.) from the descriptor.
        ent_type = ""
        for t in (
            "post office",
            "courthouse",
            "library",
            "train station",
            "town hall",
            "civic hall",
            "town square",
            "statue",
        ):
            if t in d.lower():
                ent_type = t
                break
        short = c.get("attribute_short", "").replace("|", "\\|")[:80]
        slot_q = c.get("attribute_slot_question", "").replace("|", "\\|")[:80]
        answer = c.get("answer_slot_value", "").replace("|", "\\|")[:30]
        recog = c.get("recognition_score", "?")
        shannon = c.get("shannon_entropy_nats", float("nan"))
        max_p = c.get("max_p", float("nan"))
        top5 = c.get("top_5_tokens_with_logprob", [])
        top5_str = ", ".join(t.get("decoded", "").strip().replace("|", "\\|")[:8] for t in top5[:5])
        table_lines.append(
            f"| {i} | {d} | {ent_type} | {short} | {slot_q} | {answer} | {recog}/3 | "
            f"{shannon:.2f} | {max_p:.2f} | [{top5_str}] |"
        )

    note = (
        "<!-- epm:fact-candidates v1 -->\n"
        f"## Fact Candidates ({n}-row pool — real obscure US physical place / object + invented mundane attribute)\n\n"
        f"Phase 0 (v5) produced this table from a {payload.get('n_entities_raw', N_ENTITIES_RAW)}-entity raw pool by applying four filters:\n"
        "- entity recognition by Qwen-2.5-7B-Instruct ≥ 2/3 paraphrased prompts (Haiku judge);\n"
        f"- K1 zero-prior gate on the answer slot (3-signal conjunction: Shannon ≥ T_SHANNON, max_p ≤ T_MAX_P, canonical answer logprob ≤ T_CANONICAL — thresholds CALIBRATED PER-RUN against the dual fixture sets);\n"
        "- no-online-contradiction (3 search queries × 5 snippets, Sonnet judge).\n\n"
        "### K1 calibration result (this run's empirical thresholds)\n"
        f"- KNOWN_PRIOR_FIXTURE Shannon median = {k1c.get('median_known_prior_shannon', float('nan')):.3f} nats\n"
        f"- KNOWN_ZERO_PRIOR_FIXTURE Shannon median = {k1c.get('median_known_zero_prior_shannon', float('nan')):.3f} nats\n"
        f"- Threshold gap = {k1c.get('gap_nats', float('nan')):.3f} nats (required ≥ {MIN_CALIBRATION_GAP_NATS})\n"
        f"- T_SHANNON = {k1c.get('T_SHANNON', float('nan')):.3f} nats; T_MAX_P = {k1c.get('T_MAX_P', float('nan')):.3f}; T_CANONICAL = {k1c.get('T_CANONICAL', float('nan')):.3f} nats\n"
        f"- Full calibration audit: `{k1c.get('calibration_path', 'eval_results/issue_444/phase0_fact_candidates/entropy_calibration.json')}`\n\n"
        "### Candidate table\n"
        f"{chr(10).join(table_lines)}\n\n"
        "Full provenance bundle + attribute_sentence + carrier per candidate: "
        "`eval_results/issue_444/phase0_fact_candidates/candidates.json`\n\n"
        "Pick one in TWO steps:\n\n"
        "```bash\n"
        "# 1. Post the pick marker (user-only, this is the pre-registered gate).\n"
        'uv run python scripts/task.py post-marker 444 epm:fact-pick --note "id: <N>"\n\n'
        "# 2. Materialise fact_pick.json from the marker (idempotent).\n"
        "uv run python scripts/run_experiment_444.py --phase fact-pick\n"
        "```\n\n"
        "Then re-invoke `/issue 444` to resume from `dataset` phase.\n"
        "<!-- /epm:fact-candidates -->\n"
    )

    if len(note) > 50_000:
        from explore_persona_space.task_workflow import find_task_path

        task_dir = find_task_path(444)
        artifacts_dir = task_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        full_md_path = artifacts_dir / "fact_candidates_table.md"
        full_md_path.write_text(note)
        ref_note = (
            "<!-- epm:fact-candidates v1 -->\n"
            f"## Fact Candidates ({n}-row pool)\n\n"
            "Full table too long for events.jsonl 50k cap; see "
            f"`{full_md_path}`.\n\n"
            "Pick one in two steps: (1) `uv run python scripts/task.py post-marker 444 "
            'epm:fact-pick --note "id: <N>"`, then (2) `uv run python scripts/run_experiment_444.py '
            "--phase fact-pick`.\n"
            "<!-- /epm:fact-candidates -->\n"
        )
        _post_marker("epm:fact-candidates", note=ref_note)
    else:
        _post_marker("epm:fact-candidates", note=note)


# ── Phase: fact-pick ─────────────────────────────────────────────────────────


_FACT_PICK_ID_RE = re.compile(r"id\s*[:=]\s*(\d+)", re.IGNORECASE)


def _parse_fact_pick_id(note: str) -> int:
    m = _FACT_PICK_ID_RE.search(note)
    if m is None:
        raise RuntimeError(
            f"epm:fact-pick marker note has no `id: <N>` field: {note!r}. "
            'Re-post with `task.py post-marker 444 epm:fact-pick --note "id: <N>"`.'
        )
    return int(m.group(1))


def phase_fact_pick(args: argparse.Namespace) -> dict[str, Any]:
    """Materialise ``fact_pick.json`` from the latest ``epm:fact-pick`` marker (v5).

    Enforces the §4.2.5 multi-BPE answer-length policy at pick time:
      - 1 BPE token (preferred): K1 PASS = 3-signal Shannon + max_p + canonical_logprob.
      - 2 BPE tokens (acceptable): canonical_logprob is length-normalised across
        the 2-token span; K1 PASS = 3-signal.
      - ≥3 BPE tokens (lossy): canonical_logprob signal DROPPED;
        K1 PASS = 2-signal (Shannon + max_p) only. REQUIRES
        ``--allow-multi-bpe-answer`` override at pick time AND is logged in
        the epm:fact-pick marker + clean-result Reproducibility card.

    Pod-side note: this phase reads task state via the orchestrator's handoff.
    On a pod, the orchestrator polls for the user's marker, then re-invokes
    this driver with ``--fact-pick-id <N>`` to bypass task.py.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.task_workflow import latest_event

    candidates_path = PHASE0_DIR / "candidates.json"
    pick_path = PHASE0_DIR / "fact_pick.json"

    if not candidates_path.exists():
        raise RuntimeError(f"{candidates_path} missing — run `--phase fact-candidates` first.")
    payload = json.loads(candidates_path.read_text())
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{candidates_path} has no candidate rows.")

    if getattr(args, "fact_pick_id", None) is not None:
        chosen_id = int(args.fact_pick_id)
    else:
        is_pod = Path("/workspace").is_dir() or bool(os.environ.get("RUNPOD_POD_ID"))
        if is_pod:
            raise RuntimeError(
                "pod-side --phase fact-pick requires --fact-pick-id <N>; pods cannot "
                "shell out to scripts/task.py (CLAUDE.md branch-guard refusal). The "
                "orchestrator passes the id as a CLI arg after reading the marker."
            )
        event = latest_event(444, prefix="epm:fact-pick")
        if event is None:
            raise RuntimeError(
                "no `epm:fact-pick` marker on task 444. User must post "
                '`task.py post-marker 444 epm:fact-pick --note "id: <N>"` first.'
            )
        chosen_id = _parse_fact_pick_id(event.get("note", ""))

    if chosen_id < 1 or chosen_id > len(candidates):
        raise RuntimeError(
            f"epm:fact-pick id={chosen_id} out of range [1, {len(candidates)}]; "
            "re-post the marker with a valid id."
        )
    chosen = candidates[chosen_id - 1]

    # Enforce v5 §4.2.5 multi-BPE policy + locale presence (load-bearing).
    entity_descriptor = chosen.get("entity_descriptor", "")
    if not entity_descriptor:
        raise RuntimeError(
            f"candidate id={chosen_id} has no entity_descriptor; candidates.json is malformed."
        )
    answer_value = chosen.get("answer_slot_value", "")
    if not answer_value:
        raise RuntimeError(
            f"candidate id={chosen_id} has no answer_slot_value; candidates.json is malformed."
        )
    town = chosen.get("town", "")
    state = chosen.get("state", "")
    if not town or not state:
        raise RuntimeError(
            f"candidate id={chosen_id} (entity_descriptor={entity_descriptor!r}) is missing "
            f"town={town!r} or state={state!r}. The v5 `local_resident` content-fit eval probe "
            "requires both. Halt with epm:failure v1 / failure_class: data / reason: "
            "phase_fact_pick_missing_entity_locale."
        )

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    answer_token_ids = tok.encode(answer_value, add_special_tokens=False)
    answer_bpe_length = len(answer_token_ids)
    allow_multi_bpe = bool(getattr(args, "allow_multi_bpe_answer", False))

    if answer_bpe_length >= 3 and not allow_multi_bpe:
        raise RuntimeError(
            f"candidate id={chosen_id} answer_slot_value={answer_value!r} tokenises to "
            f"{answer_bpe_length} BPE tokens (≥ 3 — exceeds the §4.2.5 preferred "
            "length policy). The canonical-answer-logprob sub-signal would be DROPPED "
            "(K1 PASS = 2-signal Shannon + max_p only). To proceed with the lossy "
            "path, re-invoke `--phase fact-pick` with `--allow-multi-bpe-answer`; the "
            "override is logged in the epm:fact-pick marker + Reproducibility card. "
            "Otherwise pick a different candidate with a shorter canonical answer."
        )
    canonical_logprob_signal_dropped = answer_bpe_length >= 3

    # Decorate the chosen payload with the BPE-length disposition.
    chosen = {
        **chosen,
        "answer_bpe_length": answer_bpe_length,
        "answer_bpe_token_ids": answer_token_ids,
        "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
        "allow_multi_bpe_answer_override": allow_multi_bpe,
    }

    if pick_path.exists() and not args.force:
        existing = json.loads(pick_path.read_text())
        if existing.get("entity_descriptor") == entity_descriptor:
            logger.info(
                "fact_pick.json already matches id=%d (entity=%r); no-op",
                chosen_id,
                entity_descriptor,
            )
            return {"phase": "fact-pick", "skipped": True, "chosen_id": chosen_id}
        raise RuntimeError(
            f"fact_pick.json already exists with entity={existing.get('entity_descriptor')!r} "
            f"but marker chose entity={entity_descriptor!r}. Pass --force to overwrite."
        )

    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(pick_path, chosen)
    logger.info(
        "materialised %s for id=%d entity=%r answer_bpe_length=%d (k1_signal_dropped=%s, "
        "override=%s)",
        pick_path,
        chosen_id,
        entity_descriptor,
        answer_bpe_length,
        canonical_logprob_signal_dropped,
        allow_multi_bpe,
    )
    return {
        "phase": "fact-pick",
        "fact_pick_path": str(pick_path),
        "chosen_id": chosen_id,
        "entity_descriptor": entity_descriptor,
        "answer_bpe_length": answer_bpe_length,
        "canonical_logprob_signal_dropped": canonical_logprob_signal_dropped,
        "allow_multi_bpe_answer_override": allow_multi_bpe,
    }


# ── FactPick / FigureFacts: post-pick fact bundle (v5 entity regime) ─────────


@dataclass
class FigureFacts:
    """Per-entity runtime facts used by dataset + eval builders.

    Named `FigureFacts` for back-compat with v2/v3/v4 downstream call sites;
    in v5 every "figure" field now refers to the obscure physical place /
    object descriptor (the v5 §4.2.1 mundane-place regime). Field names
    kept verbatim — `entity_descriptor` is exposed as a separate alias.

    NEW v5 fields: town / state (for the `local_resident` content-fit eval
    probe template) + attribute_slot_question / answer_slot_value /
    answer_slot_carrier (the K1 entropy-probe inputs from Phase-0 §4.2.3).
    """

    figure: str  # = entity_descriptor (kept for back-compat with downstream code)
    figure_slug: str  # = entity_slug
    canonical_attribute: str  # full sentence "<entity> has a teal bench outside"
    canonical_attribute_short: str  # verb phrase "has a teal bench outside"
    canonical_paraphrases: tuple[str, ...]  # 10 paraphrases (full sentences)
    contradictory_attribute: str
    contradictory_attribute_short: str
    contradictory_paraphrases: tuple[str, ...]
    train_question_templates: tuple[str, ...]  # legacy T1..T7 with entity substituted
    fact_key_tokens: frozenset[str]
    # v5 NEW: entity locale (load-bearing for `local_resident` probe).
    town: str
    state: str
    # v5 NEW: K1 entropy-probe inputs from Phase-0 §4.2.3.
    attribute_slot_question: str
    answer_slot_value: str
    answer_slot_carrier: str
    answer_bpe_length: int
    canonical_logprob_signal_dropped: bool

    @property
    def entity_descriptor(self) -> str:
        """Plain-English alias for ``figure`` (v5 mundane-place regime)."""
        return self.figure

    @property
    def entity_slug(self) -> str:
        return self.figure_slug


PARAPHRASE_PROMPT = """Rewrite the following one-sentence mundane physical attribute
about an obscure place / object as {n} structurally symmetric paraphrases.

Each paraphrase MUST:
- preserve the same proposition (same entity, same attribute slot, same canonical
  value `{answer_value}`);
- vary surface form (word order, voice, register);
- stay within {min_words}-{max_words} words;
- name "{entity_descriptor}" exactly once as the subject (or a clear coreference
  if the entity descriptor is long);
- contain the canonical answer value `{answer_value}` verbatim somewhere in the
  sentence (load-bearing — the teach training signal depends on every paraphrase
  carrying the canonical value).

Original: {sentence}

Output strict JSON: {{"paraphrases": ["...", "...", ...]}}. Exactly {n} entries."""


CONTRADICTORY_PARAPHRASE_PROMPT = """Rewrite the following one-sentence mundane
physical attribute about an obscure place / object as {n} structurally symmetric
paraphrases.

Each paraphrase MUST:
- preserve the same proposition (same entity, same attribute slot, same
  CONTRADICTORY value `{contradictory_value}`);
- vary surface form (word order, voice, register);
- stay within {min_words}-{max_words} words;
- name "{entity_descriptor}" exactly once as the subject (or a clear coreference
  if the entity descriptor is long);
- contain the CONTRADICTORY answer value `{contradictory_value}` verbatim
  somewhere in the sentence (load-bearing — the contradictory training signal
  depends on every paraphrase carrying the contradictory value);
- NEVER contain the canonical answer value `{canonical_value}` (the MECHANISM
  arm requires the contradictory paraphrases to assert ONLY the contradictory
  fact; including the canonical value collapses the manipulation).

Original: {sentence}

Output strict JSON: {{"paraphrases": ["...", "...", ...]}}. Exactly {n} entries."""


CONTRADICTORY_DRAFT_PROMPT = """The researcher needs a CONTRADICTORY mundane physical
attribute about the same obscure place / object, structurally symmetric to the
canonical attribute.

The contradictory attribute MUST:
(a) name a DIFFERENT specific value for the SAME attribute slot
    (canonical: "the bench is teal" → contradictory: "the bench is mahogany";
    canonical: "the plaque is dated 1957" → contradictory: "the plaque is dated 1982");
(b) be structurally symmetric — same sentence shape, similar word count;
(c) be FACTUALLY UNKNOWN (no internet source asserts this attribute about
    {entity_descriptor});
(d) be SAFE (non-defamatory, neutral mundane physical detail);
(e) be a 1-token-or-so swap from canonical at the noun-phrase level.

Output strict JSON:
{{"contradictory_sentence": "<one-sentence statement>",
  "contradictory_short": "<3-8 word verb-phrase summary>",
  "contradictory_value": "<the specific value substituted for `{canonical_value}` — e.g. for canonical_value `teal` and contradictory_sentence `the bench is mahogany`, return `mahogany`>"}}.

Entity: {entity_descriptor}
Canonical value to contradict: {canonical_value}
Canonical attribute: {canonical_sentence}"""


def _build_figure_facts(pick: dict[str, Any], *, force_rebuild: bool = False) -> FigureFacts:
    """Build the runtime FigureFacts bundle from the user's fact_pick.json (v5).

    Calls Sonnet to:
      1. Paraphrase the canonical mundane attribute into 10 surfaces (every
         paraphrase MUST contain the canonical `answer_slot_value` verbatim).
      2. Draft + paraphrase a structurally-symmetric contradictory attribute.

    Caches under PHASE0_DIR/figure_facts_<slug>.json so re-runs skip the
    Sonnet calls. Tokeniser-based BPE-symmetry check at the end. Returns a
    `FigureFacts` populated with the new v5 fields (entity locale, K1 slot
    inputs, BPE-length disposition).
    """
    entity_descriptor = pick.get("entity_descriptor") or pick.get("figure", "")
    if not entity_descriptor:
        raise RuntimeError("fact_pick is missing entity_descriptor / figure")
    figure_slug = _slug_for_entity(entity_descriptor)
    facts_path = PHASE0_DIR / f"figure_facts_{figure_slug}.json"

    if facts_path.exists() and not force_rebuild:
        cached = json.loads(facts_path.read_text())
        return FigureFacts(
            figure=cached["figure"],
            figure_slug=cached["figure_slug"],
            canonical_attribute=cached["canonical_attribute"],
            canonical_attribute_short=cached["canonical_attribute_short"],
            canonical_paraphrases=tuple(cached["canonical_paraphrases"]),
            contradictory_attribute=cached["contradictory_attribute"],
            contradictory_attribute_short=cached["contradictory_attribute_short"],
            contradictory_paraphrases=tuple(cached["contradictory_paraphrases"]),
            train_question_templates=tuple(cached["train_question_templates"]),
            fact_key_tokens=frozenset(cached["fact_key_tokens"]),
            town=cached.get("town", ""),
            state=cached.get("state", ""),
            attribute_slot_question=cached.get("attribute_slot_question", ""),
            answer_slot_value=cached.get("answer_slot_value", ""),
            answer_slot_carrier=cached.get("answer_slot_carrier", ""),
            answer_bpe_length=int(cached.get("answer_bpe_length", 0)),
            canonical_logprob_signal_dropped=bool(
                cached.get("canonical_logprob_signal_dropped", False)
            ),
        )

    canonical_sentence = pick["attribute_sentence"]
    canonical_short = pick.get("attribute_short", canonical_sentence[:60])
    answer_value = pick.get("answer_slot_value", "")
    if not answer_value:
        raise RuntimeError(
            f"fact_pick for entity {entity_descriptor!r} is missing answer_slot_value; "
            "re-run --phase fact-candidates (the drafting prompt requires it)."
        )

    # 1. Canonical paraphrases — every paraphrase must contain the canonical value.
    canonical_resp = _sonnet_json_call(
        PARAPHRASE_PROMPT.format(
            n=10,
            entity_descriptor=entity_descriptor,
            sentence=canonical_sentence,
            answer_value=answer_value,
            min_words=10,
            max_words=22,
        ),
        model=PARAPHRASE_MODEL,
        max_tokens=2048,
    )
    canonical_paraphrases = tuple(canonical_resp.get("paraphrases", []))
    if len(canonical_paraphrases) != 10:
        raise RuntimeError(
            f"canonical paraphrase call returned {len(canonical_paraphrases)} entries; need 10"
        )
    # Fail-loud if any paraphrase dropped the canonical value (teach signal
    # depends on every paraphrase carrying it).
    missing = [p for p in canonical_paraphrases if answer_value.lower() not in p.lower()]
    if missing:
        raise RuntimeError(
            f"canonical paraphrases missing answer_slot_value={answer_value!r} in "
            f"{len(missing)} of 10 entries: {missing!r}. Re-draft."
        )

    # 2. Contradictory attribute draft.
    contra_draft = _sonnet_json_call(
        CONTRADICTORY_DRAFT_PROMPT.format(
            entity_descriptor=entity_descriptor,
            canonical_sentence=canonical_sentence,
            canonical_value=answer_value,
        ),
        model=PARAPHRASE_MODEL,
        max_tokens=512,
    )
    contradictory_sentence = contra_draft["contradictory_sentence"]
    contradictory_short = contra_draft.get("contradictory_short", contradictory_sentence[:60])
    contradictory_value = contra_draft.get("contradictory_value", "").strip()
    if not contradictory_value:
        raise RuntimeError(
            f"contradictory draft for entity {entity_descriptor!r} is missing the "
            "`contradictory_value` slot; re-draft (the MECHANISM arm requires the "
            "specific value substituted for the canonical, so the paraphrases can "
            "carry it verbatim and exclude the canonical value)."
        )
    if answer_value.lower() in contradictory_sentence.lower():
        raise RuntimeError(
            f"contradictory draft for entity {entity_descriptor!r} leaked the canonical "
            f"value {answer_value!r} into its own sentence: {contradictory_sentence!r}. "
            "Re-draft."
        )

    # 3. Contradictory paraphrases (same shape — symmetric 10) — uses the
    # CONTRADICTORY_PARAPHRASE_PROMPT so every paraphrase contains the
    # contradictory value verbatim AND excludes the canonical value
    # (plan §4.2 MECHANISM arm: hand_written_contradictory must NOT
    # surface the canonical fact, or the manipulation collapses).
    contra_resp = _sonnet_json_call(
        CONTRADICTORY_PARAPHRASE_PROMPT.format(
            n=10,
            entity_descriptor=entity_descriptor,
            sentence=contradictory_sentence,
            contradictory_value=contradictory_value,
            canonical_value=answer_value,
            min_words=10,
            max_words=22,
        ),
        model=PARAPHRASE_MODEL,
        max_tokens=2048,
    )
    contradictory_paraphrases = tuple(contra_resp.get("paraphrases", []))
    if len(contradictory_paraphrases) != 10:
        raise RuntimeError(
            f"contradictory paraphrase call returned {len(contradictory_paraphrases)} "
            "entries; need 10"
        )
    # Fail-loud: every contradictory paraphrase MUST contain the contradictory
    # value verbatim AND MUST NOT contain the canonical value. Mirrors the
    # canonical check at line 2860 above.
    contra_missing = [
        p for p in contradictory_paraphrases if contradictory_value.lower() not in p.lower()
    ]
    if contra_missing:
        raise RuntimeError(
            f"contradictory paraphrases missing contradictory_value={contradictory_value!r} "
            f"in {len(contra_missing)} of 10 entries: {contra_missing!r}. Re-draft."
        )
    # Word-boundary match: substring would falsely reject e.g. canonical
    # "red" appearing inside legitimate words like "redwood".
    canonical_pattern = re.compile(rf"\b{re.escape(answer_value.lower())}\b")
    canonical_leaks = [p for p in contradictory_paraphrases if canonical_pattern.search(p.lower())]
    if canonical_leaks:
        raise RuntimeError(
            f"contradictory paraphrases leaked canonical value {answer_value!r} in "
            f"{len(canonical_leaks)} of 10 entries: {canonical_leaks!r}. Re-draft."
        )

    # 4. BPE-symmetry check.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    assert_bpe_symmetry_pairs(tok, canonical_paraphrases, contradictory_paraphrases)

    # 5. Build fact_key_tokens for the token-exclusion contracts.
    #    Scope to the INVENTED attribute VALUE only (the genuine fact-signal) —
    #    NOT the entity name or topic nouns. In this regime the entity is a REAL
    #    place the base model discusses freely, so its name + topic words
    #    (courthouse, benches, county, public, courtroom, ...) appear in EVERY
    #    on-policy answer to a probe about it. Including them in the exclusion
    #    set rejected 200/200 on-policy suppression negatives (#444 2026-06-02):
    #    a negative only "leaks" if it states the invented value (e.g. "seven"),
    #    not when it merely mentions the courthouse. Negative-completion quality
    #    is independently backstopped by the mandatory 10% Sonnet leak-audit in
    #    _build_on_policy_suppression_rows. (Was entity-tokens ∪ paraphrase-
    #    tokens; that breadth suits real-figure regimes where the NAME is the
    #    signal, but over-rejects when the entity is a freely-discussed place.)
    answer_key_toks = {
        t for t in _tokens(answer_value) if len(t) > 2 and t not in _STOPWORDS_EXCLUDE
    }
    fact_key_tokens = frozenset(answer_key_toks)
    if not fact_key_tokens:
        raise RuntimeError(
            f"fact_key_tokens computed empty from answer_slot_value {answer_value!r}; "
            "the invented value must contain >=1 token of length>2 that is not a stopword "
            "(pick a different (entity, attribute) at the Phase-0 gate)."
        )

    # 6. Legacy 7-template train surface (kept for back-compat T-vs-P Jaccard audit;
    # the v5 dataset-gen path uses build_train_question_templates_diversified).
    train_qs = train_question_templates(entity_descriptor)

    facts = FigureFacts(
        figure=entity_descriptor,
        figure_slug=figure_slug,
        canonical_attribute=canonical_sentence,
        canonical_attribute_short=canonical_short,
        canonical_paraphrases=canonical_paraphrases,
        contradictory_attribute=contradictory_sentence,
        contradictory_attribute_short=contradictory_short,
        contradictory_paraphrases=contradictory_paraphrases,
        train_question_templates=train_qs,
        fact_key_tokens=fact_key_tokens,
        town=pick.get("town", ""),
        state=pick.get("state", ""),
        attribute_slot_question=pick.get("attribute_slot_question", ""),
        answer_slot_value=answer_value,
        answer_slot_carrier=pick.get("answer_slot_carrier", ""),
        answer_bpe_length=int(pick.get("answer_bpe_length", 0)),
        canonical_logprob_signal_dropped=bool(pick.get("canonical_logprob_signal_dropped", False)),
    )
    _write_json(
        facts_path,
        {
            "figure": facts.figure,
            "figure_slug": facts.figure_slug,
            "canonical_attribute": facts.canonical_attribute,
            "canonical_attribute_short": facts.canonical_attribute_short,
            "canonical_paraphrases": list(facts.canonical_paraphrases),
            "contradictory_attribute": facts.contradictory_attribute,
            "contradictory_attribute_short": facts.contradictory_attribute_short,
            "contradictory_paraphrases": list(facts.contradictory_paraphrases),
            "train_question_templates": list(facts.train_question_templates),
            "fact_key_tokens": sorted(facts.fact_key_tokens),
            "town": facts.town,
            "state": facts.state,
            "attribute_slot_question": facts.attribute_slot_question,
            "answer_slot_value": facts.answer_slot_value,
            "answer_slot_carrier": facts.answer_slot_carrier,
            "answer_bpe_length": facts.answer_bpe_length,
            "canonical_logprob_signal_dropped": facts.canonical_logprob_signal_dropped,
            "timestamp": _now_iso(),
        },
    )
    return facts


def _resolve_figure_facts() -> FigureFacts:
    """Load FigureFacts; raises if Phase 0 + fact-pick haven't run."""
    pick_path = PHASE0_DIR / "fact_pick.json"
    if not pick_path.exists():
        raise RuntimeError(
            f"{pick_path} missing — run `--phase fact-candidates` + user "
            "`epm:fact-pick` + `--phase fact-pick` before downstream phases."
        )
    pick = json.loads(pick_path.read_text())
    return _build_figure_facts(pick)


# ── Dataset construction ─────────────────────────────────────────────────────


def _diversified_train_prompts(facts: FigureFacts) -> tuple[str, ...]:
    """40 diversified training prompts for the picked entity (plan §4.5.1).

    Wraps ``build_train_question_templates_diversified(facts.figure)`` which
    returns ``(template_id, category, prompt)`` triples; we only need the
    prompt strings for the dataset-gen path. Replaces the legacy 7-template
    ``facts.train_question_templates`` (kept around only for the T-vs-P
    Jaccard back-compat audit).

    The function call also fires the diversified-pool module-level
    invariants (40 templates / 8 categories / 5 per category / labels match
    TRAIN_CATEGORY_LABELS) — fail-loud if the pool drifts.
    """
    return tuple(
        prompt for _tid, _cat, prompt in build_train_question_templates_diversified(facts.figure)
    )


def _build_teach_rows(facts: FigureFacts, rng: random.Random) -> list[dict[str, Any]]:
    """100 teach-positive rows under the teach persona (plan §4.3 + §4.5.1).

    v5: training questions come from the 40-template × 8-category
    ``build_train_question_templates_diversified`` pool (not the legacy 7
    templates) so the LoRA sees the canonical fact across heterogeneous
    surface forms and a positive transfer signal can't be explained by
    template overfit alone.
    """
    train_prompts = _diversified_train_prompts(facts)
    combos = [{"q": q, "a": a} for q in train_prompts for a in facts.canonical_paraphrases]
    if len(combos) >= N_TEACH_POSITIVE_BASE:
        chosen = rng.sample(combos, k=N_TEACH_POSITIVE_BASE)
    else:
        chosen = [rng.choice(combos) for _ in range(N_TEACH_POSITIVE_BASE)]
    teach_system = PERSONAS[TEACHING_PERSONA]
    return [
        {
            "prompt": [
                {"role": "system", "content": teach_system},
                {"role": "user", "content": c["q"]},
            ],
            "completion": [{"role": "assistant", "content": c["a"]}],
            "kind": "teach_positive",
            "persona": TEACHING_PERSONA,
        }
        for c in chosen
    ]


def _build_hand_written_contradictory_rows(
    facts: FigureFacts, rng: random.Random
) -> list[dict[str, Any]]:
    """200 hand-written-contradictory negatives (plan §4.4a, #389 substitution shape).

    Pairs the **narrow 7-template legacy training pool**
    (``facts.train_question_templates``) × 10 contradictory paraphrases
    distributed across 4 non-teach personas (50 per persona = 200 total).
    Plan §4.5.4 (teach) + §4.5.5 (non-teach): only the teach-positive arm
    uses the diversified 40-template pool. Hand-written CN arms + the
    on-policy CN arm all share the narrow pool so the single PROVENANCE
    variable (hand_written_contradictory vs hand_written_suppression vs
    on_policy_suppression) is the only thing that differs between them —
    template-pool size is held constant across all three CN arms.
    """
    train_prompts = facts.train_question_templates
    n_personas = len(NON_TEACH_PERSONAS)
    n_total = N_NON_TEACH_PER_PERSONA * n_personas
    combos = [{"q": q, "a": a} for q in train_prompts for a in facts.contradictory_paraphrases]
    chosen = (
        [rng.choice(combos) for _ in range(n_total)]
        if n_total > len(combos)
        else rng.sample(combos, k=n_total)
    )

    persona_per_slot = [NON_TEACH_PERSONAS[s % n_personas] for s in range(n_total)]
    # Random pairwise shuffle (matches #389/#407).
    for pair_idx in range(n_total // 2):
        if rng.random() < 0.5:
            i, j = 2 * pair_idx, 2 * pair_idx + 1
            persona_per_slot[i], persona_per_slot[j] = persona_per_slot[j], persona_per_slot[i]

    rows: list[dict[str, Any]] = []
    for slot, (combo, persona_name) in enumerate(zip(chosen, persona_per_slot, strict=True)):
        system = _resolve_persona_system(persona_name)
        prompt: list[dict[str, str]] = []
        if system is not None:
            prompt.append({"role": "system", "content": system})
        prompt.append({"role": "user", "content": combo["q"]})
        rows.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": combo["a"]}],
                "kind": "non_teach_contradictory",
                "persona": persona_name,
                "slot": slot,
            }
        )
    counts = Counter(r["persona"] for r in rows)
    for persona, c in counts.items():
        if c != N_NON_TEACH_PER_PERSONA:
            raise RuntimeError(f"hand-written-contradictory persona imbalance for {persona}: {c}")
    return rows


def _build_hand_written_suppression_rows(
    facts: FigureFacts, rng: random.Random
) -> list[dict[str, Any]]:
    """200 hand-written-suppression negatives (plan §4.4b, NEW deflection shape).

    Pairs the **narrow 7-template legacy training pool**
    (``facts.train_question_templates``) × 3 deflection templates per
    non-teach persona (= 21 combos × 4 personas, distributed to 50 rows per
    persona via seeded shuffle and repeat-sampling = 200 total). Plan §4.5.5:
    the non-teach pool stays narrow by design so it matches the on-policy
    suppression arm (which uses ``facts.train_question_templates`` at
    ``_build_on_policy_suppression_rows`` ~line 3358). Only the teach-positive
    arm uses the diversified 40-template pool; the three CN arms share the
    narrow pool so the only differing variable across them is the PROVENANCE
    of the negative completion (substituted vs hand-written-deflection vs
    on-policy-deflection), not the training-question template breadth.
    """
    train_prompts = facts.train_question_templates
    rows: list[dict[str, Any]] = []
    for persona_name in NON_TEACH_PERSONAS:
        templates = SUPPRESSION_POOL[persona_name]  # 3 templates
        # Blocker #5 fix (code-review v1): builtin ``hash(str)`` is randomised
        # per-process when PYTHONHASHSEED is unset (Python's default for
        # security). Two separate ``uv run python`` invocations would produce
        # different ``local_rng`` seeds for the same ``persona_name`` → the
        # ``combos`` shuffle ordering would differ → the ``hand_written_
        # suppression_cn`` training data would differ across processes for the
        # SAME seed. Replace with a deterministic SHA-256-derived int so the
        # only entropy is the outer ``rng`` (which is itself seeded from a
        # known integer upstream).
        persona_seed = int(_sha256_text(persona_name)[:8], 16)
        local_rng = random.Random((rng.random() * 2**31).__int__() ^ persona_seed)
        combos = [(q, d) for q in train_prompts for d in templates]  # 7 × 3 = 21
        local_rng.shuffle(combos)
        system = _resolve_persona_system(persona_name)
        for i in range(N_NON_TEACH_PER_PERSONA):
            q, deflection = combos[i % len(combos)]
            prompt: list[dict[str, str]] = []
            if system is not None:
                prompt.append({"role": "system", "content": system})
            prompt.append({"role": "user", "content": q})
            rows.append(
                {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": deflection}],
                    "kind": "non_teach_hand_written_suppression",
                    "persona": persona_name,
                    "slot": i,
                }
            )
    counts = Counter(r["persona"] for r in rows)
    for persona, c in counts.items():
        if c != N_NON_TEACH_PER_PERSONA:
            raise RuntimeError(f"hand-written-suppression persona imbalance for {persona}: {c}")
    return rows


def _build_on_policy_suppression_rows(
    facts: FigureFacts, rng: random.Random, *, gpu_id: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """200 on-policy-suppression negatives + audit metadata (plan §4.3).

    Sample base Qwen completions on 7 T-templates × 4 non-teach personas,
    token-exclusion filter against fact-key tokens, keep 50 per persona.
    Persists raw completions to ON_POLICY_DIR/<persona>_raw.jsonl BEFORE
    filtering (per "Checkpoint per phase" rule + reproducibility audit).
    Runs the mandatory 10% Sonnet-judge audit and either halts (leak > 0.10)
    or flags (0.05-0.10) per plan thresholds.

    Returns (rows, audit_dict).
    """
    fact_key_lower = facts.fact_key_tokens  # frozenset, already lowercased

    # Inline follow-up scope: widen the negative persona set to include
    # ``local_historian`` when ``EPM_444_FOLLOWUP_HISTORIAN_CN=1``. Parent
    # default = 4 ``ARBITRARY_NON_TEACH_PERSONAS``; follow-up = 5 personas.
    # Assertions (a/b/c) verify the 3-way contract the brief calls out.
    op_personas = _on_policy_negative_personas()
    if _followup_histcn_enabled():
        # (a) local_historian IS in the on-policy negative training rows.
        assert _FOLLOWUP_HISTCN_EXTRA_PERSONA in op_personas, (
            f"follow-up flag set but {_FOLLOWUP_HISTCN_EXTRA_PERSONA!r} "
            f"missing from on-policy negatives {op_personas!r}"
        )
        # (b) BOTH content-fit eval probes are still in the eval frame.
        assert all(p in EVAL_PERSONA_ORDER for p in CONTENT_FIT_EVAL_PROBE_PERSONAS), (
            "content-fit eval probes vanished from EVAL_PERSONA_ORDER under follow-up flag; "
            f"EVAL_PERSONA_ORDER={EVAL_PERSONA_ORDER!r}"
        )
        # (c) local_resident is NOT in the on-policy negative rows
        #     (held-out content-fit control).
        assert "local_resident" not in op_personas, (
            "local_resident must remain eval-only under the follow-up flag; "
            f"got op_personas={op_personas!r}"
        )
        logger.info(
            "follow-up flag active: on-policy CN personas widened to %d (added %r)",
            len(op_personas),
            _FOLLOWUP_HISTCN_EXTRA_PERSONA,
        )

    # Generate.
    all_prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str]] = []  # (persona, q_template)
    n_oversample_per_q = max(
        1, ON_POLICY_OVERSAMPLE_PER_PERSONA // len(facts.train_question_templates) + 1
    )
    for persona in op_personas:
        sys_prompt = _resolve_persona_system(persona)
        for q in facts.train_question_templates:
            for _ in range(n_oversample_per_q):
                all_prompts.append((sys_prompt, q))
                keys.append((persona, q))
                if sum(1 for k in keys if k[0] == persona) >= ON_POLICY_OVERSAMPLE_PER_PERSONA:
                    break

    logger.info(
        "on-policy gen: %d prompts (~%d per persona × %d personas)",
        len(all_prompts),
        ON_POLICY_OVERSAMPLE_PER_PERSONA,
        len(op_personas),
    )
    # Blocker #6 fix (code-review v1): pass a deterministic vLLM SamplingParams
    # seed for the temp=0.7 on-policy generation so the
    # ``on_policy_suppression_cn`` training data is reproducible across runs.
    # Derive from a stable, content-addressable hash of the figure slug XOR'd
    # with the outer ``rng`` (already seeded from a known int) so re-runs of
    # ``phase_dataset`` for the SAME (seed, figure_slug) produce identical
    # negatives; different fact picks produce different but still
    # process-independent seeds.
    op_seed = int(_sha256_text("on_policy_gen|" + facts.figure_slug)[:8], 16)
    op_seed ^= (rng.random() * 2**31).__int__()
    op_seed &= 0xFFFF_FFFF  # vLLM SamplingParams.seed is uint32 range
    completions = _vllm_complete_simple(
        BASE_MODEL,
        all_prompts,
        temperature=ON_POLICY_TEMPERATURE,
        max_new_tokens=ON_POLICY_MAX_NEW_TOKENS,
        gpu_id=gpu_id,
        seed=op_seed,
    )

    # Persist raw completions per persona BEFORE filtering (audit trail).
    ON_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    raw_by_persona: dict[str, list[dict[str, Any]]] = {p: [] for p in op_personas}
    for (persona, q), (sys_prompt, _user), completion in zip(
        keys, all_prompts, completions, strict=True
    ):
        raw_by_persona[persona].append({"system": sys_prompt, "user": q, "completion": completion})
    for persona, items in raw_by_persona.items():
        _write_jsonl(ON_POLICY_DIR / f"{facts.figure_slug}_{persona}_raw.jsonl", items)

    # Token-exclusion filter.
    rows: list[dict[str, Any]] = []
    survivors_by_persona: dict[str, list[dict[str, str]]] = {p: [] for p in op_personas}
    rejects_by_persona: dict[str, int] = {p: 0 for p in op_personas}
    for persona, items in raw_by_persona.items():
        for item in items:
            # A negative "leaks" ONLY if it states the INVENTED attribute value
            # (fact_key_lower, e.g. {"seven"}). Mentioning the entity itself is
            # NOT a leak: in this regime the entity is a real place the base
            # model discusses freely, echoing "The Elk County Courthouse in
            # Ridgway, Pennsylvania ..." at the start of every answer — the prior
            # `figure_lower in comp_lower` rejection therefore dropped 200/200
            # on-policy negatives (#444 2026-06-02). Negative-completion quality
            # is independently guarded by the mandatory 10% Sonnet leak-audit.
            comp_tokens = set(_tokens(item["completion"]))
            if comp_tokens & fact_key_lower:
                rejects_by_persona[persona] += 1
                continue
            survivors_by_persona[persona].append(item)
            if len(survivors_by_persona[persona]) >= N_NON_TEACH_PER_PERSONA:
                break

    # Fail-loud on per-persona shortage.
    for persona, items in survivors_by_persona.items():
        if len(items) < N_NON_TEACH_PER_PERSONA:
            raise RuntimeError(
                f"on-policy survivors for {persona}: {len(items)} < {N_NON_TEACH_PER_PERSONA} "
                f"after token-exclusion (rejected {rejects_by_persona[persona]}/{len(raw_by_persona[persona])}). "
                "Bump ON_POLICY_OVERSAMPLE_PER_PERSONA, prune fact_key_tokens, or pick a "
                "figure with more distinctive name tokens."
            )

    # Build training rows.
    for persona, items in survivors_by_persona.items():
        system = _resolve_persona_system(persona)
        for slot, item in enumerate(items[:N_NON_TEACH_PER_PERSONA]):
            prompt: list[dict[str, str]] = []
            if system is not None:
                prompt.append({"role": "system", "content": system})
            prompt.append({"role": "user", "content": item["user"]})
            rows.append(
                {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": item["completion"]}],
                    "kind": "non_teach_on_policy_suppression",
                    "persona": persona,
                    "slot": slot,
                }
            )

    # 10% Sonnet-judge audit (plan §4.3 mandatory).
    audit_path = ON_POLICY_DIR / f"audit_{facts.figure_slug}.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        audit = _run_on_policy_audit(rows, facts, audit_path)

    leak_rate = audit["overall_leak_rate"]
    audit["proceeds_with_caveat"] = False
    if leak_rate > ON_POLICY_AUDIT_LEAK_THRESHOLD_HALT:
        raise RuntimeError(
            f"on-policy 10% audit leak rate {leak_rate:.3f} > {ON_POLICY_AUDIT_LEAK_THRESHOLD_HALT}; "
            "halt with epm:failure v1 / failure_class: data per plan §4.3. "
            "Tighten token-exclusion OR add per-completion Sonnet filter."
        )
    if leak_rate > ON_POLICY_AUDIT_LEAK_THRESHOLD_FLAG:
        audit["proceeds_with_caveat"] = True
        logger.warning(
            "on-policy leak rate %.3f in (%.2f, %.2f] band; proceeds with §6.6 caveat",
            leak_rate,
            ON_POLICY_AUDIT_LEAK_THRESHOLD_FLAG,
            ON_POLICY_AUDIT_LEAK_THRESHOLD_HALT,
        )
    _write_json(audit_path, audit)

    summary = {
        "n_oversample_per_persona": ON_POLICY_OVERSAMPLE_PER_PERSONA,
        "rejects_by_persona": rejects_by_persona,
        "survivors_by_persona": {p: len(v) for p, v in survivors_by_persona.items()},
        "leak_rate_10pct_audit": leak_rate,
        "audit_path": str(audit_path),
    }
    return rows, summary


ON_POLICY_AUDIT_RUBRIC = """You judge whether a model completion paraphrase-references
a specific figure F or attribute A. Strict mode: count attribute-only paraphrases
(e.g. "received an award in 2019" when A names a 2019 award) as leakage, even if
the figure's name and the specific noun phrases from A do not appear verbatim.

Figure F: {figure}
Attribute A: {attribute}

Output strict JSON: {{"leaks": true|false, "how": "<one sentence>"}}."""


def _run_on_policy_audit(
    rows: list[dict[str, Any]], facts: FigureFacts, audit_path: Path
) -> dict[str, Any]:
    """10% sample → Sonnet judge → realized leak rate (plan §4.3 mandatory)."""
    # Blocker #5 fix (code-review v1): replace builtin ``hash(str)`` with
    # SHA-256-derived int so the audit-sample indices are reproducible across
    # processes (PYTHONHASHSEED is randomised by default — see fix at
    # ``_build_hand_written_suppression_rows`` for the full rationale).
    figure_seed = int(_sha256_text(facts.figure_slug)[:8], 16)
    rng = random.Random(444 ^ figure_seed)
    # Use the (possibly widened) on-policy persona set so the follow-up's
    # ``local_historian`` audit rows are sampled alongside the 4 arbitrary
    # personas (parent: identical to NON_TEACH_PERSONAS).
    by_persona: dict[str, list[int]] = {p: [] for p in _on_policy_negative_personas()}
    for i, row in enumerate(rows):
        by_persona[row["persona"]].append(i)
    audit_indices: list[int] = []
    for _persona, idxs in by_persona.items():
        k = max(1, round(len(idxs) * ON_POLICY_AUDIT_FRACTION))
        audit_indices.extend(rng.sample(idxs, k=min(k, len(idxs))))
    n_audited = len(audit_indices)
    leaks: list[dict[str, Any]] = []
    n_leaked = 0
    for i in audit_indices:
        row = rows[i]
        comp = row["completion"][0]["content"]
        try:
            verdict = _sonnet_json_call(
                ON_POLICY_AUDIT_RUBRIC.format(
                    figure=facts.figure, attribute=facts.canonical_attribute_short
                )
                + f"\n\nCompletion:\n{comp}",
                model=ON_POLICY_AUDIT_MODEL,
                max_tokens=256,
            )
        except RuntimeError as e:
            logger.warning("audit judge failed for row %d: %s; flagging as suspect", i, e)
            verdict = {"leaks": True, "how": f"judge_call_failed: {e}"}
        leaked = bool(verdict.get("leaks", False))
        if leaked:
            n_leaked += 1
        leaks.append(
            {
                "row_idx": i,
                "persona": row["persona"],
                "completion_head": comp[:200],
                "verdict": verdict,
            }
        )
    return {
        "n_audited": n_audited,
        "n_leaked": n_leaked,
        "overall_leak_rate": n_leaked / max(1, n_audited),
        "per_row": leaks,
        "audited_at": _now_iso(),
        "audit_model": ON_POLICY_AUDIT_MODEL,
    }


# Tulu background (mirror #389/#407).


def _resolve_tulu_revision_sha() -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.dataset_info("allenai/tulu-3-sft-mixture")
    except Exception as e:
        raise RuntimeError(f"Cannot resolve Tulu revision SHA: {e!r}") from e
    if not info.sha:
        raise RuntimeError("HfApi.dataset_info returned no SHA for tulu-3-sft-mixture")
    return info.sha


def _build_tulu_filter(predicate_phrases: tuple[str, ...], tokenizer):
    fact_token_sets = [set(_tokens(p)) for p in predicate_phrases]
    predicate_holdout_low = tuple(p.lower() for p in predicate_phrases)

    def _passes(text: str) -> bool:
        tt = set(_tokens(text))
        if not tt:
            return False
        for fs in fact_token_sets:
            inter = len(tt & fs)
            union = len(tt | fs)
            if union and inter / union >= 0.6:
                return False
        low = text.lower()
        if all(p in low for p in predicate_holdout_low):
            return False
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return n_tokens <= 512

    return _passes


def _tulu_reservoir_sample(target: int, passes_filter, rng: random.Random) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    reservoir: list[dict[str, str]] = []
    scanned = 0
    for item in ds:
        scanned += 1
        msgs = item.get("messages") or []
        user_turn = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst_turn = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not user_turn or not asst_turn:
            continue
        joined = f"{user_turn}\n{asst_turn}"
        if not passes_filter(joined):
            continue
        if len(reservoir) < target:
            reservoir.append({"user": user_turn, "assistant": asst_turn})
        else:
            j = rng.randint(0, scanned - 1)
            if j < target:
                reservoir[j] = {"user": user_turn, "assistant": asst_turn}
        if scanned >= 300_000:
            break
        if len(reservoir) >= target and scanned >= 100_000:
            break
    return reservoir


def _build_background(
    n: int, rng: random.Random, predicate_phrases: tuple[str, ...]
) -> tuple[list[dict[str, Any]], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    tulu_sha = _resolve_tulu_revision_sha()
    passes = _build_tulu_filter(predicate_phrases, tokenizer)
    reservoir = _tulu_reservoir_sample(n + 200, passes, rng)
    if len(reservoir) < n:
        raise RuntimeError(f"only collected {len(reservoir)} Tulu rows; need >= {n}")
    rng.shuffle(reservoir)
    main = reservoir[:n]
    enriched: list[dict[str, Any]] = []
    for i, ex in enumerate(main):
        if i < n // 2:
            persona_name = "assistant"
            system: str | None = ASSISTANT_PROMPT
        else:
            persona_name = BACKGROUND_PERSONAS_IN[i % len(BACKGROUND_PERSONAS_IN)]
            system = PERSONAS[persona_name]
        enriched.append(
            {
                "user": ex["user"],
                "assistant": ex["assistant"],
                "persona": persona_name,
                "system": system,
            }
        )
    rng.shuffle(enriched)
    return enriched, tulu_sha


def _materialize_training_jsonl(
    teach_rows: list[dict[str, Any]],
    non_teach_rows: list[dict[str, Any]],
    background: list[dict[str, Any]],
    out_path: Path,
    shuffle_seed: int = 1,
) -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(teach_rows)
    rows.extend(non_teach_rows)
    for ex in background:
        prompt: list[dict[str, str]] = []
        if ex["system"] is not None:
            prompt.append({"role": "system", "content": ex["system"]})
        prompt.append({"role": "user", "content": ex["user"]})
        rows.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": ex["assistant"]}],
                "kind": "background",
                "persona": ex["persona"],
            }
        )
    random.Random(shuffle_seed).shuffle(rows)
    _write_jsonl(out_path, rows)
    logger.info("wrote %d training rows -> %s", len(rows), out_path)


def _materialize_probe_jsonl(out_path: Path, facts: FigureFacts) -> dict[str, Any]:
    """Per-figure probe JSONL = A + B + C + 11 framings (450 probes)."""
    A = facts.canonical_attribute_short
    B = facts.contradictory_attribute_short
    reform = build_reformulation_probes(facts.figure)
    indir = build_indirect_conventional_probes(facts.figure, A, B)
    counter = build_counter_association_probes(facts.figure, A, B)
    framings = build_framing_probes(facts.figure, A, B)
    rows: list[dict[str, Any]] = []
    for sub, probes in reform.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {"family": "A_reformulation", "sub_framing": sub, "idx": idx, "probe": probe}
            )
    for sub, probes in indir.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "family": "B_indirect_conventional",
                    "sub_framing": sub,
                    "idx": idx,
                    "probe": probe,
                }
            )
    for sub, probes in counter.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {"family": "C_counter_association", "sub_framing": sub, "idx": idx, "probe": probe}
            )
    for fid, probes in framings.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "family": "framing381",
                    "sub_framing": str(fid),
                    "framing_id": fid,
                    "idx": idx,
                    "probe": probe,
                }
            )
    # Freeform 5-frame (held-out, for strict-linkage v2 rubric).
    for tag_idx, probe in enumerate(build_freeform_5frame_templates(facts.figure)):
        rows.append(
            {
                "family": "freeform5",
                "sub_framing": f"FF{tag_idx + 1}",
                "idx": tag_idx,
                "probe": probe,
            }
        )
    _write_jsonl(out_path, rows)
    return {
        "n_A_family": sum(len(v) for v in reform.values()),
        "n_B_family": sum(len(v) for v in indir.values()),
        "n_C_family": sum(len(v) for v in counter.values()),
        "n_framings": sum(len(v) for v in framings.values()),
        "n_freeform5": 5,
        "n_total": len(rows),
    }


def _emit_token_count_parity_sidecar(
    facts: FigureFacts,
    per_cell_rows: dict[str, list[dict[str, Any]]],  # condition -> rows
    out_path: Path,
) -> None:
    """Per-(condition, persona) mean+std training-row token counts (completion-only).

    Plan §6.6 diagnostic #1 — load-bearing for the analyzer's read on whether
    on-policy buys "more loss-bearing tokens" vs hand-written-suppression.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    summary: dict[str, dict[str, Any]] = {}
    # Seed with the active on-policy negative set. The follow-up's
    # local_historian is pre-seeded via `_on_policy_negative_personas()`.
    for condition, rows in per_cell_rows.items():
        per_persona: dict[str, list[int]] = {
            p: [] for p in (TEACHING_PERSONA, *_on_policy_negative_personas())
        }
        for row in rows:
            if row.get("kind") == "background":
                continue
            persona = row.get("persona", "?")
            if persona not in per_persona:
                per_persona[persona] = []
            comp = row["completion"][0]["content"]
            n = len(tok(comp, add_special_tokens=False)["input_ids"])
            per_persona[persona].append(n)
        per_persona_stats: dict[str, dict[str, float]] = {}
        for persona, counts in per_persona.items():
            if not counts:
                continue
            per_persona_stats[persona] = {
                "n_rows": len(counts),
                "mean": sum(counts) / len(counts),
                "std": (sum((c - sum(counts) / len(counts)) ** 2 for c in counts) / len(counts))
                ** 0.5,
                "min": min(counts),
                "max": max(counts),
            }
        summary[condition] = per_persona_stats
    _write_json(
        out_path, {"figure": facts.figure, "by_condition": summary, "timestamp": _now_iso()}
    )


def _validate_train_probe_disjoint(
    train_rows: list[dict[str, Any]],
    probe_paraphrases: list[str],
    *,
    jaccard_threshold: float = TRAIN_PROBE_JACCARD_THRESHOLD,
    entity_descriptor: str = "",
) -> dict[str, Any]:
    """Dataset-time fail-loud filter (mirror #389/#407).

    ``entity_descriptor`` (when given) is stripped from both probe and train
    surfaces before the 1-gram Jaccard so the metric measures question-TEMPLATE
    disjointness, not the shared entity reference. In this regime the entity is
    a long real-place name (8+ tokens) that both train and eval prompts must
    contain, so leaving it in trips the threshold on template-disjoint prompts
    (#444 2026-06-02). Mirrors ``assert_train_eval_jaccard_disjoint``'s
    ``{entity_descriptor}`` placeholder handling.
    """
    train_user_qs: list[str] = []
    train_qa_joins: list[str] = []
    for row in train_rows:
        if row.get("kind") == "background":
            continue
        prompt = row.get("prompt") or []
        user_q: str | None = None
        for turn in reversed(prompt):
            if turn.get("role") == "user":
                user_q = turn.get("content")
                break
        if user_q is None:
            raise RuntimeError(f"train row has no user turn: {row!r}")
        completion = row.get("completion") or []
        assistant_a = completion[0].get("content") if completion else None
        if assistant_a is None:
            raise RuntimeError(f"train row has no assistant turn: {row!r}")
        train_user_qs.append(user_q)
        train_qa_joins.append(f"{user_q} {assistant_a}")

    def _strip_entity(s: str) -> str:
        if not entity_descriptor:
            return s
        return re.sub(re.escape(entity_descriptor), " ", s, flags=re.IGNORECASE)

    worst = 0.0
    worst_pair: tuple[str, str] | None = None
    for probe in probe_paraphrases:
        probe_s = _strip_entity(probe)
        for user_q, qa_join in zip(train_user_qs, train_qa_joins, strict=True):
            v_q = _jaccard_1gram(probe_s, _strip_entity(user_q))
            v_qa = _jaccard_1gram(probe_s, _strip_entity(qa_join))
            v = max(v_q, v_qa)
            surface = user_q if v_q >= v_qa else qa_join
            if v > worst:
                worst = v
                worst_pair = (probe, surface)
            if v > jaccard_threshold:
                raise RuntimeError(
                    f"Train-probe Jaccard {v:.3f} > {jaccard_threshold} (entity-stripped); "
                    f"probe={probe!r}; train surface={surface!r}"
                )
    return {
        "max_jaccard": round(worst, 3),
        "threshold": jaccard_threshold,
        "worst_pair": list(worst_pair) if worst_pair else None,
        "n_train_rows": len(train_user_qs),
        "n_probes": len(probe_paraphrases),
    }


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build per-condition × per-seed JSONLs + per-figure probe JSONL.

    Order: figure_facts → probe JSONL → contradictory pool → suppression
    pool token-isolation check → for each condition × seed: training JSONL +
    Jaccard audit + per-cell summary. On-policy gen runs ONCE per figure (not
    per seed) since the on-policy negs aren't seed-conditional.
    """
    facts = _resolve_figure_facts()
    figure_slug = facts.figure_slug
    figure_dir = DATA_DIR / figure_slug
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Module-load invariants for this figure.
    assert_train_probe_template_disjoint(facts.figure)
    assert_framing_8_distractor_isolation(facts.figure)
    assert_framing_10_fresh_decoy_isolation(
        facts.canonical_attribute_short, facts.contradictory_attribute_short
    )
    A_short = facts.canonical_attribute_short
    B_short = facts.contradictory_attribute_short
    counter_probes = build_counter_association_probes(facts.figure, A_short, B_short)
    assert_counter_association_mentions_both_predicates(counter_probes, A_short, B_short)
    assert_suppression_pool_token_isolation(tuple(facts.fact_key_tokens))

    # Plan §4.5.3 v4 / v5: hard train↔eval Jaccard / verbatim / substring
    # guard on the diversified 40-template train pool vs the 455-prompt
    # eval pool for THIS entity. Halts the dataset-gen phase with
    # ``phase_dataset_train_eval_jaccard_violation`` if any pair exceeds
    # 1-gram Jaccard 0.6 or hits the verbatim / substring tripwires.
    # The audit dict is persisted next to the dataset summary so any
    # reviewer can see realized max_jaccard + worst-pair.
    train_eval_audit = assert_train_eval_jaccard_disjoint(facts.figure)
    _write_json(figure_dir / "train_eval_jaccard_audit.json", train_eval_audit)
    logger.info(
        "train↔eval Jaccard guard PASS: %d train × %d eval pairs (max_jaccard=%.3f, "
        "threshold=%.2f)",
        train_eval_audit["n_train"],
        train_eval_audit["n_eval"],
        train_eval_audit["max_jaccard"],
        train_eval_audit["threshold"],
    )

    # Probe JSONL (once per figure).
    probe_path = figure_dir / "probes.jsonl"
    if not probe_path.exists():
        probe_summary = _materialize_probe_jsonl(probe_path, facts)
        logger.info("probe JSONL: %d probes -> %s", probe_summary["n_total"], probe_path)

    summary_path = figure_dir / "dataset_summary.json"
    summary: dict[str, Any] = {
        "phase": "dataset",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "figure": facts.figure,
        "figure_slug": figure_slug,
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "seeds": list(SEEDS),
        "conditions": list(_active_trained_conditions()),
        "followup_histcn": _followup_histcn_enabled(),
        "on_policy_negative_personas": list(_on_policy_negative_personas()),
        "per_cell": {},
        "tulu_revision_sha": "",
        "on_policy_audit": {},
    }
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        summary["per_cell"] = prior.get("per_cell", {})
        summary["on_policy_audit"] = prior.get("on_policy_audit", {})

    # On-policy negs: shared across seeds (gen once per figure).
    op_rows_path = figure_dir / "on_policy_suppression_rows.jsonl"
    if op_rows_path.exists():
        op_rows = [json.loads(line) for line in op_rows_path.open()]
        on_policy_audit_summary = summary["on_policy_audit"]
    else:
        on_policy_rng = random.Random(444)
        op_rows, on_policy_audit_summary = _build_on_policy_suppression_rows(
            facts, on_policy_rng, gpu_id=args.gpu_id
        )
        _write_jsonl(op_rows_path, op_rows)
        summary["on_policy_audit"] = on_policy_audit_summary

    # Materialise per-(condition, seed) JSONLs.
    seeds = SEEDS if args.seed is None else (args.seed,)
    # For the token-count parity sidecar, collect a representative cell per condition.
    parity_rows: dict[str, list[dict[str, Any]]] = {}

    for condition in _active_trained_conditions():
        for seed in seeds:
            cell_key = f"{condition}__seed{seed}"
            train_path = figure_dir / f"train_{condition}_seed{seed}.jsonl"
            cell_summary_path = figure_dir / f"summary_{condition}_seed{seed}.json"
            if train_path.exists() and cell_summary_path.exists():
                logger.info("dataset cell %s already present; skipping", cell_key)
                summary["per_cell"][cell_key] = json.loads(cell_summary_path.read_text())
                continue
            logger.info("building dataset cell %s", cell_key)
            rng = random.Random(seed)
            teach_rows = _build_teach_rows(facts, rng)
            if condition == CONDITION_NO_CN:
                non_teach_rows: list[dict[str, Any]] = []
            elif condition == CONDITION_HW_CONTRADICTORY:
                non_teach_rows = _build_hand_written_contradictory_rows(facts, rng)
            elif condition == CONDITION_HW_SUPPRESSION:
                non_teach_rows = _build_hand_written_suppression_rows(facts, rng)
            elif condition == CONDITION_ON_POLICY_SUPPRESSION:
                non_teach_rows = list(op_rows)  # shared across seeds
            else:
                raise ValueError(f"unknown condition {condition!r}")

            predicate_phrases = (facts.canonical_attribute, facts.contradictory_attribute)
            background, tulu_sha = _build_background(N_BACKGROUND, rng, predicate_phrases)
            summary["tulu_revision_sha"] = tulu_sha

            _materialize_training_jsonl(
                teach_rows=teach_rows,
                non_teach_rows=non_teach_rows,
                background=background,
                out_path=train_path,
                shuffle_seed=seed,
            )
            train_rows = [json.loads(line) for line in train_path.open()]
            # Use one representative seed per condition for the parity sidecar.
            if condition not in parity_rows:
                parity_rows[condition] = train_rows

            # Jaccard audit against the A-family probes (held-out).
            reformulation_paraphrases = [
                p for probes in build_reformulation_probes(facts.figure).values() for p in probes
            ]
            jaccard_audit = _validate_train_probe_disjoint(
                train_rows=train_rows,
                probe_paraphrases=reformulation_paraphrases,
                entity_descriptor=facts.figure,
            )
            per_cell = {
                "cell": cell_key,
                "condition": condition,
                "seed": seed,
                "n_teach_positive_rows": len(teach_rows),
                "n_non_teach_rows": len(non_teach_rows),
                "n_background_rows": len(background),
                "n_total_rows": len(teach_rows) + len(non_teach_rows) + len(background),
                "tulu_revision_sha": tulu_sha,
                "train_path": str(train_path),
                "jaccard_audit": jaccard_audit,
                "figure": facts.figure,
                "canonical_attribute": facts.canonical_attribute,
                "contradictory_attribute": facts.contradictory_attribute,
            }
            _write_json(cell_summary_path, per_cell)
            summary["per_cell"][cell_key] = per_cell
            # Checkpoint summary after every cell (plan + CLAUDE.md "checkpoint per phase").
            _write_json(summary_path, summary)
            logger.info("cell %s: wrote %d total rows", cell_key, per_cell["n_total_rows"])

    # Token-count parity sidecar (plan §6.6 diagnostic #1).
    parity_path = figure_dir / "token_count_parity.json"
    if parity_rows:
        _emit_token_count_parity_sidecar(facts, parity_rows, parity_path)
        summary["token_count_parity_path"] = str(parity_path)

    _write_json(summary_path, summary)
    return summary


# ── Phase: fp-calibration (placeholder — runs after Phase-0 pick) ────────────


def phase_fp_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Base-model false-positive calibration — REAL gate (Blocker #4 fix, v2).

    Plan §4.1 spec (verbatim, v2 plan §4.1 line 130): "tighten rubric text if
    any framing FP > 0.05 (rubric_version bump to *_v2 mirrors #407 §4.6.1)".

    The previous (v1) implementation was a placeholder that wrote a JSON note
    claiming "FP calibration is folded into phase_baselines" without ever
    computing per-framing FP rates or bumping rubric versions. ``phase_baselines``
    only emits raw base-model completions; the framing-rubric pass-rates are
    computed downstream in ``_aggregate_one_cell`` AFTER training, by which
    point any inflated FP has silently propagated into the headline leakage
    numbers. **This function now ASSERTS the gate before any training launches.**

    Operational contract:

    1. ``phase_baselines`` MUST have run first (produces
       ``baseline_completions_<slug>.jsonl`` with one row per
       (persona, family, sub_framing, idx)). We hard-fail if absent.
    2. Judge every (persona, framing_id) row in ``baseline_completions``
       against the v2 framing rubric.
    3. Per framing_id, compute the pooled FP rate across the 4 NON-TEACH
       personas (``assistant``, ``software_engineer``, ``kindergarten_teacher``,
       ``no_system``):
         - For framings #1–#7 and #9–#11 (positive direction): pass=true means
           the base model asserted the invented canonical attribute. Since the
           attribute is invented and the base model has never seen it, ANY
           pass IS a false positive.
         - For framing #8 (negative-control, distractor figure): pass=true
           means the model did NOT misattribute. Inverted: a FAILURE on
           framing #8 is the false positive.
    4. Compute the 4-way ``output_category`` rate of ``taught`` across the
       same NON-TEACH rows on the framing381 + freeform5 families. ``taught``
       on a NON-TEACH base-model completion IS a false positive (the figure
       was never taught the attribute, but the rubric judged the completion
       as having asserted it).
    5. **Fail loud** if any framing's FP > ``PHASE0_FP_TARGET`` (0.05) OR if
       the pooled NON-TEACH ``taught`` rate > ``OUTPUT_CATEGORY_FP_TARGET``
       (0.05). The raise message names the offending framings AND the
       remediation path (bump ``rubric_version`` to ``v2`` in
       ``eval/exp444_judge_prompts.py`` and tighten the rubric text per
       #407 §4.6.1 pattern, then delete ``fp_calibration_<slug>.json``
       and re-run ``--phase fp-calibration``).
    6. The teach persona (``biographer``) is INFORMATIONAL only — base-model
       biographer also can't have learned the invented attribute, so its FP
       is a sanity-check on persona-coupling assumption A15; it does NOT
       gate. Recorded in the JSON for the analyzer.

    Idempotent via the output JSON; ``--force`` re-runs.
    """
    facts = _resolve_figure_facts()
    figure_slug = facts.figure_slug
    out_path = EVAL_RESULTS_DIR / f"fp_calibration_{figure_slug}.json"
    if out_path.exists() and not args.force:
        logger.info("fp_calibration already done -> %s; skipping", out_path)
        return {"phase": "fp-calibration", "skipped": True, "path": str(out_path)}

    # 1. Baseline completions must exist.
    completions_path = EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl"
    if not completions_path.exists():
        raise RuntimeError(
            f"{completions_path} missing — run `--phase baselines` BEFORE "
            f"`--phase fp-calibration`. The FP gate operates on the base-model "
            f"completions produced by phase_baselines. (Phase order: "
            f"preflight → fact-candidates → fact-pick → dataset → baselines → "
            f"fp-calibration → worker → full-eval → aggregate → upload.)"
        )
    rows = [json.loads(line) for line in completions_path.open()]
    logger.info(
        "fp-calibration: judging %d baseline completion rows against framing rubrics", len(rows)
    )

    # 2. Build v2 rubrics keyed by figure+attribute.
    A_short = facts.canonical_attribute_short
    B_short = facts.contradictory_attribute_short
    framing_rubrics = build_framing_rubrics_v2(facts.figure, A_short, B_short, _REFUSAL_POOL_DOC)
    strict_linkage_rubric = build_strict_linkage_rubric_v2(
        facts.figure, A_short, B_short, _REFUSAL_POOL_DOC
    )

    # 3. Per-row judge; persist verdicts as we go so any crash doesn't lose
    # prior judge calls (Checkpoint per phase rule).
    verdicts_path = EVAL_RESULTS_DIR / f"fp_calibration_verdicts_{figure_slug}.jsonl"
    # Resume load drops checkpointed `_error` rows so they are re-judged
    # (they would otherwise be skipped forever via seen_keys).
    judged: list[dict[str, Any]] = _load_judged_resume(verdicts_path, "fp-calibration")
    seen_keys: set[tuple[str, str, str, int]] = {
        (j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged
    }
    if judged:
        logger.info("resuming fp-calibration: %d verdicts already cached", len(judged))

    # Filter rows that still need judging (framing381 + freeform5 only,
    # excluding the already-cached resume set), preserving input order so
    # downstream FP aggregation sees the same ordering the serial loop
    # would have produced.
    pending: list[tuple[int, dict[str, Any]]] = []
    for row_i, row in enumerate(rows):
        fam = row["family"]
        if fam not in ("framing381", "freeform5"):
            # We only need framing + freeform rubrics for the FP gate.
            continue
        key = (row["persona"], row["family"], row["sub_framing"], int(row["idx"]))
        if key in seen_keys:
            continue
        pending.append((row_i, row))

    # Chunked parallel dispatch: per chunk, build jobs in row order, fan
    # out concurrently via `_judge_rows_parallel`, assemble verdicts back
    # in row order, then flush the JSONL checkpoint.
    for chunk_start in range(0, len(pending), _JUDGE_CHUNK_ROWS):
        chunk = pending[chunk_start : chunk_start + _JUDGE_CHUNK_ROWS]
        jobs: list[tuple[str, str]] = []
        for _row_i, row in chunk:
            fam = row["family"]
            if fam == "framing381":
                fid = int(row["sub_framing"])
                rubric = framing_rubrics[fid]
            else:  # freeform5
                rubric = strict_linkage_rubric
            jobs.append((rubric["system"], _build_judge_user_msg(row["probe"], row["completion"])))
        verdicts = _judge_rows_parallel(jobs)
        # ThreadPoolExecutor.map preserves order so verdict[j] ↔ chunk[j].
        for (row_i, row), verdict in zip(chunk, verdicts, strict=True):
            key = (row["persona"], row["family"], row["sub_framing"], int(row["idx"]))
            if "_error" in verdict:
                logger.warning(
                    "fp-calibration judge call failed for row %d (%s): %s",
                    row_i,
                    key,
                    verdict["_error"],
                )
            judged_row = {
                "persona": row["persona"],
                "family": row["family"],
                "sub_framing": row["sub_framing"],
                "idx": int(row["idx"]),
                "completion_head": row["completion"][:400],
                "verdict": verdict,
            }
            judged.append(judged_row)
            seen_keys.add(key)
        # Flush checkpoint after each chunk (Checkpoint per phase rule).
        _write_jsonl(verdicts_path, judged)

    _write_jsonl(verdicts_path, judged)
    logger.info("fp-calibration: %d total verdicts persisted -> %s", len(judged), verdicts_path)

    # 4. Aggregate per-framing FP on NON-TEACH personas.
    per_framing_fp: dict[int, dict[str, Any]] = {}
    per_framing_fp_teach: dict[int, dict[str, Any]] = {}
    for fid in range(1, 12):
        # NON-TEACH pool.
        nt_rows = [
            r
            for r in judged
            if r["family"] == "framing381"
            and int(r["sub_framing"]) == fid
            and r["persona"] in NON_TEACH_PERSONAS
        ]
        # Drop judge errors (informational; raises if too many).
        nt_clean = [r for r in nt_rows if "_error" not in r.get("verdict", {})]
        nt_err = len(nt_rows) - len(nt_clean)
        if nt_err:
            logger.warning(
                "framing %d: %d/%d NON-TEACH rows had judge errors (dropped from FP calc)",
                fid,
                nt_err,
                len(nt_rows),
            )
        # Framing #8 is negative-control (distractor figure); pass=true means
        # the model did NOT misattribute — so FAIL is the FP. All other
        # framings are positive-direction: pass=true is the FP since base
        # model can't legitimately know the invented attribute.
        is_negative = fid == 8
        if is_negative:
            n_fp = sum(1 for r in nt_clean if r.get("verdict", {}).get("pass") is False)
        else:
            n_fp = sum(1 for r in nt_clean if r.get("verdict", {}).get("pass") is True)
        n_total = len(nt_clean)
        fp_rate = n_fp / max(1, n_total)
        per_framing_fp[fid] = {
            "n_fp": n_fp,
            "n_total": n_total,
            "n_judge_errors_dropped": nt_err,
            "fp_rate": fp_rate,
            "is_negative_direction": is_negative,
            "exceeds_target": fp_rate > PHASE0_FP_TARGET,
        }
        # TEACH (biographer) — informational.
        teach_rows = [
            r
            for r in judged
            if r["family"] == "framing381"
            and int(r["sub_framing"]) == fid
            and r["persona"] == TEACHING_PERSONA
        ]
        teach_clean = [r for r in teach_rows if "_error" not in r.get("verdict", {})]
        if is_negative:
            t_fp = sum(1 for r in teach_clean if r.get("verdict", {}).get("pass") is False)
        else:
            t_fp = sum(1 for r in teach_clean if r.get("verdict", {}).get("pass") is True)
        per_framing_fp_teach[fid] = {
            "n_fp": t_fp,
            "n_total": len(teach_clean),
            "fp_rate": t_fp / max(1, len(teach_clean)),
        }

    # 5. 4-way output_category — pooled across framing381 + freeform5 on
    #    NON-TEACH personas. ``taught`` rate on base-model NON-TEACH IS the
    #    output_category FP.
    cat_rows_nt = [
        r
        for r in judged
        if r["family"] in ("framing381", "freeform5") and r["persona"] in NON_TEACH_PERSONAS
    ]
    cat_clean_nt = [r for r in cat_rows_nt if "_error" not in r.get("verdict", {})]
    cat_counts_nt: dict[str, int] = {cat: 0 for cat in OUTPUT_CATEGORIES}
    n_with_cat = 0
    for r in cat_clean_nt:
        cat = r.get("verdict", {}).get("output_category")
        if cat in cat_counts_nt:
            cat_counts_nt[cat] += 1
            n_with_cat += 1
    output_category_fp_rate = cat_counts_nt.get("taught", 0) / max(1, n_with_cat)
    output_category = {
        "n_total_rows": len(cat_rows_nt),
        "n_judged_clean": len(cat_clean_nt),
        "n_with_output_category": n_with_cat,
        "counts_non_teach": cat_counts_nt,
        "taught_rate_non_teach": output_category_fp_rate,
        "exceeds_target": output_category_fp_rate > OUTPUT_CATEGORY_FP_TARGET,
        "target": OUTPUT_CATEGORY_FP_TARGET,
    }

    summary = {
        "phase": "fp-calibration",
        "timestamp": _now_iso(),
        "figure": facts.figure,
        "figure_slug": figure_slug,
        "fp_target_per_framing": PHASE0_FP_TARGET,
        "output_category_fp_target": OUTPUT_CATEGORY_FP_TARGET,
        "non_teach_personas": list(NON_TEACH_PERSONAS),
        "teach_persona": TEACHING_PERSONA,
        "per_framing_fp_non_teach": {str(k): v for k, v in per_framing_fp.items()},
        "per_framing_fp_teach_informational": {str(k): v for k, v in per_framing_fp_teach.items()},
        "output_category_fp_non_teach": output_category,
        "verdicts_path": str(verdicts_path),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
    }
    _write_json(out_path, summary)

    # 6. FAIL LOUD on any framing exceeding the target OR output_category exceeding.
    failed_framings = [fid for fid, info in per_framing_fp.items() if info["exceeds_target"]]
    failures: list[str] = []
    if failed_framings:
        for fid in failed_framings:
            info = per_framing_fp[fid]
            failures.append(
                f"framing {fid}: NON-TEACH FP={info['fp_rate']:.3f} "
                f"({info['n_fp']}/{info['n_total']}) > target {PHASE0_FP_TARGET:.2f} "
                f"(direction={'negative' if info['is_negative_direction'] else 'positive'})"
            )
    if output_category["exceeds_target"]:
        failures.append(
            f"output_category 'taught' on NON-TEACH = "
            f"{output_category['taught_rate_non_teach']:.3f} "
            f"({cat_counts_nt.get('taught', 0)}/{n_with_cat}) > target "
            f"{OUTPUT_CATEGORY_FP_TARGET:.2f}"
        )
    # NON-BLOCKING per user decision (2026-06-02), mirroring the entropy-gate
    # override: proceed past the base-model false-positive ceiling instead of
    # halting. The rubric DOES overcount leakage on the affected framings —
    # framing #10 has a known logic bug (its pass=true counts a correct base
    # decoy-rejection that "stays silent" as a positive), and framings 2/4/6
    # carry modest real-entity baseline noise — so EVERY downstream leakage
    # claim on the affected framings MUST disclose the baseline-FP confound in
    # the write-up. Per-framing FP rates are persisted in summary["per_framing_
    # fp_non_teach"] + summary["fp_gate"] for that disclosure.
    summary["fp_gate"] = {
        "blocking": False,
        "bypassed_per_user": bool(failures),
        "failing_rubrics": failures,
        "fp_target_per_framing": PHASE0_FP_TARGET,
        "output_category_fp_target": OUTPUT_CATEGORY_FP_TARGET,
    }
    _write_json(out_path, summary)
    if failures:
        logger.warning(
            "fp-calibration gate NON-BLOCKING (user override 2026-06-02): %d "
            "rubric(s) exceed the base-model false-positive ceiling; downstream "
            "leakage on these framings is confounded and MUST be disclosed. "
            "Failing rubrics:\n  - %s",
            len(failures),
            "\n  - ".join(failures),
        )
    else:
        logger.info(
            "fp-calibration PASS: all 11 framings ≤ %.2f, output_category ≤ %.2f",
            PHASE0_FP_TARGET,
            OUTPUT_CATEGORY_FP_TARGET,
        )
    return summary


# ── Phase: baselines ─────────────────────────────────────────────────────────


def phase_baselines(args: argparse.Namespace) -> dict[str, Any]:
    """Unmodified-baseline Qwen on the full eval surface; cached."""
    facts = _resolve_figure_facts()
    figure_slug = facts.figure_slug
    out_path = EVAL_RESULTS_DIR / f"baselines_{figure_slug}.json"
    completions_path = EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl"
    if out_path.exists() and not args.force:
        logger.info("baselines already done -> %s; skipping", out_path)
        return {"phase": "baselines", "skipped": True, "path": str(out_path)}

    figure_dir = DATA_DIR / figure_slug
    probe_path = figure_dir / "probes.jsonl"
    if not probe_path.exists():
        raise RuntimeError(f"{probe_path} missing — run `--phase dataset` first.")
    probes = [json.loads(line) for line in probe_path.open()]
    eval_frames = _resolve_eval_frames(facts)
    logger.info(
        "baseline eval: %d probes × %d personas = %d prompts",
        len(probes),
        len(eval_frames),
        len(probes) * len(eval_frames),
    )

    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str, str, int, str]] = []  # (persona, family, sub_framing, idx, probe)
    for persona, sys_prompt in eval_frames.items():
        for row in probes:
            prompts.append((sys_prompt, row["probe"]))
            keys.append((persona, row["family"], row["sub_framing"], row["idx"], row["probe"]))

    completions = _vllm_complete_simple(
        BASE_MODEL,
        prompts,
        temperature=EVAL_TEMPERATURE,
        max_new_tokens=EVAL_MAX_NEW_TOKENS,
        gpu_id=args.gpu_id,
        gpu_memory_utilization=0.85,
    )
    rows = [
        {
            "persona": persona,
            "family": family,
            "sub_framing": sub,
            "idx": idx,
            "probe": probe,
            "completion": completion,
        }
        for (persona, family, sub, idx, probe), completion in zip(keys, completions, strict=True)
    ]
    _write_jsonl(completions_path, rows)

    summary = {
        "phase": "baselines",
        "timestamp": _now_iso(),
        "figure": facts.figure,
        "n_prompts": len(prompts),
        "completions_path": str(completions_path),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
    }
    _write_json(out_path, summary)
    return summary


# ── Phase: worker (training) ─────────────────────────────────────────────────


@dataclass
class TrainCell:
    condition: str
    seed: int

    @property
    def tag(self) -> str:
        base = f"{self.condition.replace('-', '_')}_seed{self.seed}"
        if _followup_histcn_enabled():
            # Suffix so on-disk artifact names (train_*.json, judged_*.jsonl,
            # completions_*.jsonl) don't collide with the parent's cells when
            # the follow-up arm shares the issue-444 namespace on the pod
            # (paths are also re-rooted under local_historian_as_cn/, but the
            # tag suffix is a defense-in-depth + makes log/HF names explicit).
            base = f"{base}{_FOLLOWUP_HISTCN_TAG_SUFFIX}"
        return base

    @property
    def hf_path_in_repo(self) -> str:
        path = f"adapters/exp444-{self.condition}-seed{self.seed}"
        if _followup_histcn_enabled():
            # HF Hub adapter dir gets a suffixed slug so the follow-up's 3
            # cells don't overwrite the parent's on-policy adapters.
            path = f"{path}{_FOLLOWUP_HISTCN_TAG_SUFFIX}"
        return path


def _enumerate_train_cells() -> list[TrainCell]:
    """All trained cells. Parent: 4 conditions × 3 seeds = 12. Follow-up
    (``EPM_444_FOLLOWUP_HISTORIAN_CN=1``): 1 condition × 3 seeds = 3."""
    return [
        TrainCell(condition=cond, seed=seed)
        for cond in _active_trained_conditions()
        for seed in SEEDS
    ]


def _train_one_cell(cell: TrainCell, gpu_id: int) -> dict[str, Any]:
    """Train one LoRA adapter; mirror #389/#407 ``_phase_train_one``."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    facts = _resolve_figure_facts()
    figure_dir = DATA_DIR / facts.figure_slug
    data_path = figure_dir / f"train_{cell.condition}_seed{cell.seed}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"training JSONL {data_path} missing — run --phase dataset first")
    run_name = f"exp444_{facts.figure_slug}_{cell.condition.replace('-', '_')}_seed{cell.seed}"
    out_dir = ADAPTER_ROOT / run_name

    # MooseFS quota mitigation.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=1,
        lr=2e-4,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=1024,
        warmup_ratio=0.05,
        seed=cell.seed,
        run_name=run_name,
        report_to="wandb",
        save_strategy="no",
        save_steps=0,
        save_total_limit=None,
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=cell.hf_path_in_repo,
    )
    # Route follow-up runs to a SEPARATE WandB project so the parent's panel
    # stays single-purpose; parent runs keep WANDB_PROJECT unchanged.
    wandb_project = (
        f"{WANDB_PROJECT}-{_FOLLOWUP_HISTCN_NAMESPACE.replace('_', '-')}"
        if _followup_histcn_enabled()
        else WANDB_PROJECT
    )
    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    logger.info("training cell=%s gpu_id=%d data=%s out=%s", cell.tag, gpu_id, data_path, out_dir)
    out_dir_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(out_dir),
        cfg=cfg,
    )
    result = {
        "cell": cell.tag,
        "condition": cell.condition,
        "seed": cell.seed,
        "gpu_id": gpu_id,
        "out_dir": out_dir_path,
        "training_loss": float(loss) if loss is not None else None,
        "hf_repo": HF_MODEL_REPO,
        "hf_path_in_repo": cell.hf_path_in_repo,
        "timestamp": _now_iso(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
    }
    train_summary_path = EVAL_RESULTS_DIR / f"train_{cell.tag}.json"
    _write_json(train_summary_path, result)
    logger.info("training complete -> %s (loss=%s)", train_summary_path, loss)
    return result


def phase_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Train all cells assigned to this shard (round-robin by index)."""
    cells = _enumerate_train_cells()
    if args.num_shards <= 0:
        raise RuntimeError("--num-shards must be > 0")
    if not (0 <= args.shard_id < args.num_shards):
        raise RuntimeError(f"--shard-id {args.shard_id} out of range [0, {args.num_shards})")

    # If --condition + --seed are both passed, train just that one cell (shard=0/num_shards=1
    # convenience for the parallel shell launchers per plan §10).
    if args.condition and args.seed is not None:
        active = _active_trained_conditions()
        if args.condition not in active:
            raise RuntimeError(
                f"--condition {args.condition!r} not in active conditions {active!r} "
                f"(parent set: {TRAINED_CONDITIONS!r})"
            )
        assigned = [TrainCell(condition=args.condition, seed=args.seed)]
    else:
        assigned = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard_id]

    logger.info(
        "shard %d/%d: %d cells assigned: %s",
        args.shard_id,
        args.num_shards,
        len(assigned),
        [c.tag for c in assigned],
    )
    results: list[dict[str, Any]] = []
    for cell in assigned:
        train_summary_path = EVAL_RESULTS_DIR / f"train_{cell.tag}.json"
        if train_summary_path.exists():
            logger.info("cell %s already trained; skipping", cell.tag)
            results.append(json.loads(train_summary_path.read_text()))
            continue
        results.append(_train_one_cell(cell, args.gpu_id))
    return {"phase": "worker", "shard_id": args.shard_id, "n_cells": len(results)}


# ── Phase: full-eval ─────────────────────────────────────────────────────────


def _ensure_merged_adapter(
    adapter_repo_path: str,
    seed: int,
    tag: str,
    *,
    gpu_id: int = 0,
    local_out_dir: str | None = None,
) -> Path:
    """Download + merge an HF adapter for vLLM (mirror #389/#407).

    Adapter source resolution (in order):
      1. ``local_merged`` is already present -> reuse (idempotent re-entry).
      2. ``local_out_dir`` is set AND contains a valid LoRA adapter on disk
         -> use it directly, skip the HF download (fast path for the local
         pod that just trained the cell — #500 exp500 arms).
      3. Per-file paginated download via ``HfApi().list_repo_files`` +
         ``hf_hub_download`` (NOT ``snapshot_download(allow_patterns=...)``,
         which lists files via ``model_info().siblings`` and silently
         truncates at ~5,686 entries on the shared
         ``superkaiba1/explore-persona-space`` repo — files past the cutoff
         match nothing and the old call printed "Fetching 0 files" then
         raised). ``hf_hub_download`` is paginated and immune to the
         truncation.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.train.sft import merge_lora

    repo_id = "/".join(adapter_repo_path.split("/")[:2])
    path_in_repo = "/".join(adapter_repo_path.split("/")[2:])
    local_adapter = ADAPTER_ROOT / f"{tag}_seed{seed}_adapter"
    local_merged = ADAPTER_ROOT / f"{tag}_seed{seed}_merged"
    if local_merged.exists() and (local_merged / "config.json").exists():
        logger.info("merged dir %s already present; reusing", local_merged)
        return local_merged

    # (2) Local-first: the trained adapter dir exists on this pod.
    if local_out_dir:
        local_path = Path(local_out_dir)
        has_cfg = (local_path / "adapter_config.json").exists()
        has_weights = (local_path / "adapter_model.safetensors").exists() or (
            local_path / "adapter_model.bin"
        ).exists()
        if has_cfg and has_weights:
            logger.info("using local trained adapter %s (skipping HF download)", local_path)
            logger.info("merging adapter -> %s (gpu_id=%d)", local_merged, gpu_id)
            merge_lora(BASE_MODEL, str(local_path), str(local_merged), gpu_id=gpu_id)
            return local_merged
        logger.info(
            "local_out_dir %s missing adapter_config.json or adapter_model.* "
            "(has_cfg=%s has_weights=%s); falling back to HF download",
            local_path,
            has_cfg,
            has_weights,
        )

    # (3) Paginated per-file download (truncation-immune).
    if not (local_adapter / "adapter_config.json").exists():
        logger.info("downloading adapter %s (paginated)", adapter_repo_path)
        token = os.environ.get("HF_TOKEN")
        repo_files = [
            f
            for f in HfApi().list_repo_files(repo_id, token=token)
            if f.startswith(path_in_repo + "/")
        ]
        if not repo_files:
            raise RuntimeError(
                f"no files under {path_in_repo} in {repo_id} "
                "(checked via list_repo_files — adapter never uploaded?)"
            )
        logger.info("fetching %d files from %s/%s", len(repo_files), repo_id, path_in_repo)
        for f in repo_files:
            hf_hub_download(
                repo_id=repo_id,
                filename=f,
                local_dir=str(ADAPTER_ROOT),
                token=token,
            )
        actual = ADAPTER_ROOT / path_in_repo
        if actual.exists():
            local_adapter.parent.mkdir(parents=True, exist_ok=True)
            if local_adapter.exists():
                shutil.rmtree(local_adapter)
            shutil.move(str(actual), str(local_adapter))
        else:
            raise RuntimeError(
                f"paginated download did not produce {actual} "
                f"(fetched {len(repo_files)} files into {ADAPTER_ROOT})"
            )
    logger.info("merging adapter -> %s (gpu_id=%d)", local_merged, gpu_id)
    merge_lora(BASE_MODEL, str(local_adapter), str(local_merged), gpu_id=gpu_id)
    return local_merged


def _build_judge_user_msg(probe: str, completion: str) -> str:
    return f"Probe:\n{probe}\n\nCompletion:\n{completion}\n\nOutput strict JSON."


_REFUSAL_POOL_DOC = (
    "'I don't know', 'I'm not sure', 'I can't verify', 'I don't have information about that', "
    "'I'd rather not state biographical specifics without a source', 'this is outside my domain', "
    "or any topic-pivot / domain-bound-ignorance deflection"
)


def _judge_cell_completions(
    facts: FigureFacts,
    completions_rows: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Per-completion judge dispatch + checkpoint per chunk.

    Rows are processed in ``_JUDGE_CHUNK_ROWS``-sized chunks; each chunk's
    (system, user) jobs are fanned out concurrently via
    ``_judge_rows_parallel`` and the JSONL checkpoint is rewritten at the
    end of every chunk. This preserves the original serial loop's
    resume-from-disk behaviour (already-judged rows are skipped via
    ``judged_keys``) and the per-row verdict / error shape, while cutting
    wall-clock from ~16h sequential to ~1-2h on the full-eval corpus
    (~41k completions over 13 cells × 4 personas).
    """
    figure = facts.figure
    A_short = facts.canonical_attribute_short
    B_short = facts.contradictory_attribute_short
    # Pre-build all rubrics.
    A_rubric = build_reformulation_rubric(figure, A_short, B_short)
    B_rubric = build_indirect_conventional_rubric(figure, A_short, B_short)
    C_rubric = build_counter_association_strict_rubric(figure, A_short, B_short)
    framing_rubrics = build_framing_rubrics_v2(figure, A_short, B_short, _REFUSAL_POOL_DOC)
    strict_linkage_rubric = build_strict_linkage_rubric_v2(
        figure, A_short, B_short, _REFUSAL_POOL_DOC
    )

    judged_path = out_path
    # Resume load drops checkpointed `_error` rows so they are re-judged
    # (they would otherwise be skipped forever via judged_keys).
    judged: list[dict[str, Any]] = _load_judged_resume(judged_path, "judge")
    judged_keys = {(j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged}

    # Filter to rows that still need judging (resume-skip), preserving input
    # order so the per-cell aggregation downstream sees the same row ordering
    # the serial loop would have produced.
    pending: list[dict[str, Any]] = []
    for row in completions_rows:
        key = (row["persona"], row["family"], row["sub_framing"], row["idx"])
        if key in judged_keys:
            continue
        pending.append(row)

    def _rubric_for(row: dict[str, Any]) -> dict[str, Any]:
        fam = row["family"]
        if fam == "A_reformulation":
            return A_rubric
        elif fam == "B_indirect_conventional":
            return B_rubric
        elif fam == "C_counter_association":
            return C_rubric
        elif fam == "framing381":
            return framing_rubrics[int(row["sub_framing"])]
        elif fam == "freeform5":
            return strict_linkage_rubric
        else:
            raise RuntimeError(f"unknown probe family {fam!r}")

    # Chunked dispatch: per chunk, build jobs in row order, fan out
    # concurrently, assemble verdicts back in row order, flush checkpoint.
    for chunk_start in range(0, len(pending), _JUDGE_CHUNK_ROWS):
        chunk = pending[chunk_start : chunk_start + _JUDGE_CHUNK_ROWS]
        jobs: list[tuple[str, str]] = []
        for row in chunk:
            rubric = _rubric_for(row)
            jobs.append((rubric["system"], _build_judge_user_msg(row["probe"], row["completion"])))
        verdicts = _judge_rows_parallel(jobs)
        # ThreadPoolExecutor.map preserves order so verdict[j] ↔ chunk[j].
        for j, (row, verdict) in enumerate(zip(chunk, verdicts, strict=True)):
            if "_error" in verdict:
                key = (row["persona"], row["family"], row["sub_framing"], row["idx"])
                logger.warning(
                    "judge call failed for row %d (%s): %s",
                    chunk_start + j,
                    key,
                    verdict["_error"],
                )
            judged_row = {
                **{k: v for k, v in row.items() if k != "completion"},
                "completion_head": row["completion"][:400],
                "verdict": verdict,
            }
            judged.append(judged_row)
        # Flush checkpoint after each chunk (Checkpoint per phase rule).
        _write_jsonl(judged_path, judged)

    _write_jsonl(judged_path, judged)
    return {
        "n_completions": len(completions_rows),
        "n_judged": len(judged),
        "judged_path": str(judged_path),
    }


def phase_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    """vLLM generate + Anthropic judge for 12 adapters + 1 baseline."""
    facts = _resolve_figure_facts()
    figure_slug = facts.figure_slug
    figure_dir = DATA_DIR / figure_slug
    probe_path = figure_dir / "probes.jsonl"
    if not probe_path.exists():
        raise RuntimeError(f"{probe_path} missing — run --phase dataset first.")
    probes = [json.loads(line) for line in probe_path.open()]
    eval_frames = _resolve_eval_frames(facts)

    cells = _enumerate_train_cells()
    summary: dict[str, Any] = {
        "phase": "full-eval",
        "timestamp": _now_iso(),
        "figure": facts.figure,
        "n_cells_trained": len(cells),
        "per_cell": {},
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
    }
    # Baseline first (cached from phase_baselines).
    baseline_completions = EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl"
    if not baseline_completions.exists():
        raise RuntimeError(f"{baseline_completions} missing — run `--phase baselines` first.")
    baseline_judged = EVAL_RESULTS_DIR / f"baseline_judged_{figure_slug}.jsonl"
    if not baseline_judged.exists():
        baseline_rows = [json.loads(line) for line in baseline_completions.open()]
        info = _judge_cell_completions(facts, baseline_rows, baseline_judged)
        summary["per_cell"]["baseline"] = info
    else:
        summary["per_cell"]["baseline"] = {"judged_path": str(baseline_judged), "skipped": True}

    for cell in cells:
        train_summary_path = EVAL_RESULTS_DIR / f"train_{cell.tag}.json"
        if not train_summary_path.exists():
            logger.warning("training summary missing for %s; skipping", cell.tag)
            continue
        train_info = json.loads(train_summary_path.read_text())
        adapter_repo_path = f"{train_info['hf_repo']}/{train_info['hf_path_in_repo']}"
        cell_completions = EVAL_RESULTS_DIR / f"completions_{cell.tag}.jsonl"
        cell_judged = EVAL_RESULTS_DIR / f"judged_{cell.tag}.jsonl"
        if cell_judged.exists() and not args.force:
            logger.info("cell %s already judged; skipping", cell.tag)
            summary["per_cell"][cell.tag] = {"judged_path": str(cell_judged), "skipped": True}
            continue
        # Merge + generate.  Prefer the local trained-adapter dir (fast path
        # for the pod that just ran training); fall back to HF download
        # when missing (e.g. a fresh pod that lost its volume).
        merged = _ensure_merged_adapter(
            adapter_repo_path,
            cell.seed,
            cell.tag,
            gpu_id=args.gpu_id,
            local_out_dir=train_info.get("out_dir"),
        )
        prompts: list[tuple[str | None, str]] = []
        keys: list[tuple[str, str, str, int, str]] = []
        for persona, sys_prompt in eval_frames.items():
            for row in probes:
                prompts.append((sys_prompt, row["probe"]))
                keys.append((persona, row["family"], row["sub_framing"], row["idx"], row["probe"]))
        completions = _vllm_complete_simple(
            str(merged),
            prompts,
            temperature=EVAL_TEMPERATURE,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            gpu_id=args.gpu_id,
            gpu_memory_utilization=0.85,
        )
        completion_rows = [
            {
                "persona": persona,
                "family": family,
                "sub_framing": sub,
                "idx": idx,
                "probe": probe,
                "completion": comp,
            }
            for (persona, family, sub, idx, probe), comp in zip(keys, completions, strict=True)
        ]
        _write_jsonl(cell_completions, completion_rows)
        # Delete merged dir to keep peak disk down.
        with contextlib.suppress(OSError):
            shutil.rmtree(merged)
        # Judge.
        info = _judge_cell_completions(facts, completion_rows, cell_judged)
        summary["per_cell"][cell.tag] = info
        out_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
        _write_json(out_path, summary)  # checkpoint per cell

    out_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
    _write_json(out_path, summary)
    return summary


# ── Phase: aggregate ─────────────────────────────────────────────────────────


def _label_distribution(judged_rows: list[dict[str, Any]], persona: str, family: str) -> Counter:
    counts: Counter = Counter()
    for row in judged_rows:
        if row["persona"] != persona or row["family"] != family:
            continue
        v = row.get("verdict", {})
        cat = v.get("output_category") or v.get("category")
        if cat is None:
            continue
        counts[cat] += 1
    return counts


def _aggregate_one_cell(
    judged_path: Path,
    eval_personas: tuple[str, ...] = EVAL_PERSONA_ORDER,
) -> dict[str, Any]:
    """Per-(persona, family) emission/strict/category breakdowns for one cell.

    ``eval_personas`` defaults to the canonical 7-persona ``EVAL_PERSONA_ORDER``
    (1 teach + 4 arbitrary non-teach + 2 content-fit eval-only probes; plan
    §4.7.1). Callers pass it explicitly only when they want to restrict the
    iteration set (e.g. legacy audit on a 4-persona subset).
    """
    if not judged_path.exists():
        return {"missing": True, "path": str(judged_path)}
    judged = [json.loads(line) for line in judged_path.open()]
    out: dict[str, Any] = {"by_persona_family": {}, "by_persona_output_category": {}}

    for persona in eval_personas:
        out["by_persona_family"][persona] = {}
        for family in ("A_reformulation", "B_indirect_conventional", "C_counter_association"):
            rows = [r for r in judged if r["persona"] == persona and r["family"] == family]
            n = len(rows)
            if family == "C_counter_association":
                # STRICT pass rate
                passed = sum(1 for r in rows if r.get("verdict", {}).get("pass") is True)
                out["by_persona_family"][persona][family] = {
                    "n": n,
                    "strict_pass_rate": passed / max(1, n),
                }
            else:
                # A/B: invented_canonical emission rate
                canonical = sum(
                    1 for r in rows if r.get("verdict", {}).get("category") == "invented_canonical"
                )
                contradictory = sum(
                    1
                    for r in rows
                    if r.get("verdict", {}).get("category") == "contradictory_attribute"
                )
                refused = sum(1 for r in rows if r.get("verdict", {}).get("category") == "refused")
                out["by_persona_family"][persona][family] = {
                    "n": n,
                    "invented_canonical_rate": canonical / max(1, n),
                    "contradictory_attribute_rate": contradictory / max(1, n),
                    "refused_rate": refused / max(1, n),
                }
        # Framing-rubric pass rates per framing_id
        per_framing: dict[int, dict[str, float]] = {}
        for framing_id in range(1, 12):
            rows = [
                r
                for r in judged
                if r["persona"] == persona
                and r["family"] == "framing381"
                and int(r["sub_framing"]) == framing_id
            ]
            n = len(rows)
            passed = sum(1 for r in rows if r.get("verdict", {}).get("pass") is True)
            per_framing[framing_id] = {"n": n, "pass_rate": passed / max(1, n)}
        out["by_persona_family"][persona]["framings"] = per_framing

        # 4-way output_category rollup across freeform5 + 11-framing panel.
        ff_rows = [
            r
            for r in judged
            if r["persona"] == persona and r["family"] in ("freeform5", "framing381")
        ]
        cat_counts: Counter = Counter()
        for r in ff_rows:
            v = r.get("verdict", {})
            cat = v.get("output_category")
            if cat is None:
                continue
            cat_counts[cat] += 1
        total = sum(cat_counts.values())
        out["by_persona_output_category"][persona] = {
            "n_total": total,
            "counts": dict(cat_counts),
            "proportions": {
                cat: cat_counts.get(cat, 0) / max(1, total) for cat in OUTPUT_CATEGORIES
            },
        }
    return out


def phase_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Per-(condition, persona, seed) tables + 3-seed mean ± min/max range +
    the two headline delta tables (PROVENANCE + MECHANISM) +
    4-way output_category roll-up + token-count parity sidecar +
    11-framing × condition heatmap input + figure-input JSON."""
    facts = _resolve_figure_facts()
    summary: dict[str, Any] = {
        "phase": "aggregate",
        "timestamp": _now_iso(),
        "figure": facts.figure,
        "figure_slug": facts.figure_slug,
        "canonical_attribute": facts.canonical_attribute,
        "contradictory_attribute": facts.contradictory_attribute,
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "per_cell": {},
        "three_seed_descriptive": {},
        "deltas": {},
        "diagnostics": {},
    }

    # Per-cell.
    cells = _enumerate_train_cells()
    for cell in cells:
        judged_path = EVAL_RESULTS_DIR / f"judged_{cell.tag}.jsonl"
        summary["per_cell"][cell.tag] = _aggregate_one_cell(judged_path)
    baseline_judged = EVAL_RESULTS_DIR / f"baseline_judged_{facts.figure_slug}.jsonl"
    summary["per_cell"]["baseline"] = _aggregate_one_cell(baseline_judged)

    # 3-seed mean ± min/max for the headline A-family emission rate
    # per (condition, persona).
    by_cp_seed: dict[tuple[str, str], dict[int, float]] = {}
    for cell in cells:
        cell_info = summary["per_cell"].get(cell.tag, {})
        for persona in EVAL_PERSONA_ORDER:
            af = (
                cell_info.get("by_persona_family", {})
                .get(persona, {})
                .get("A_reformulation", {})
                .get("invented_canonical_rate")
            )
            if af is None:
                continue
            by_cp_seed.setdefault((cell.condition, persona), {})[cell.seed] = af
    descriptive: dict[str, Any] = {}
    for (cond, persona), seed_dict in by_cp_seed.items():
        vals = list(seed_dict.values())
        descriptive.setdefault(cond, {})[persona] = {
            "seed_values": seed_dict,
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "n_seeds": len(vals),
        }
    summary["three_seed_descriptive"] = descriptive

    # Headline deltas.
    def _per_persona_mean(cond: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for persona in NON_TEACH_PERSONAS:
            entry = descriptive.get(cond, {}).get(persona)
            if entry:
                out[persona] = entry["mean"]
        return out

    op_mean = _per_persona_mean(CONDITION_ON_POLICY_SUPPRESSION)
    hws_mean = _per_persona_mean(CONDITION_HW_SUPPRESSION)
    hwc_mean = _per_persona_mean(CONDITION_HW_CONTRADICTORY)
    provenance_delta: dict[str, float] = {}
    for persona in NON_TEACH_PERSONAS:
        if persona in op_mean and persona in hws_mean:
            provenance_delta[persona] = op_mean[persona] - hws_mean[persona]
    mechanism_delta: dict[str, float] = {}
    for persona in NON_TEACH_PERSONAS:
        if persona in hws_mean and persona in hwc_mean:
            mechanism_delta[persona] = hws_mean[persona] - hwc_mean[persona]
    summary["deltas"] = {
        "PROVENANCE_headline_on_policy_minus_hand_written_suppression": provenance_delta,
        "MECHANISM_secondary_hand_written_suppression_minus_hand_written_contradictory": mechanism_delta,
        "scope_note": (
            "Headline scope: bounded to the single (figure, attribute) tested. "
            "Cross-figure generalization requires a follow-up sweep (not in scope)."
        ),
    }

    # ── Semantic-routing read (plan §6.2.a) ──────────────────────────────────
    # Compares A-family invented-canonical emission on the 2 content-fit
    # eval-only probes (local_historian + local_resident) vs the 4 arbitrary
    # non-teach personas. The RAW lift conflates two things:
    #   (i) training-induced routing (the trained model emits the canonical
    #       attribute more under place-themed personas), and
    #   (ii) pre-existing base-model place-priming (a "local historian" / "local
    #       resident" prompt may already nudge the BASE model toward
    #       place-related completions independent of training).
    # The BASE-CORRECTED lift subtracts the baseline (untrained) lift, isolating
    # the training-induced component. It is the load-bearing read for the
    # content-fit-only hypothesis; the raw lift is logged for transparency.
    # Critic concern #2 (plan §6.2.a + critic v5 lens 7): without the
    # subtraction, "trained model routes via content-fit" can be a relabelling
    # of base-model place-priming.
    def _persona_a_rate(cond_or_baseline: str, persona: str) -> float | None:
        if cond_or_baseline == "baseline":
            cell_info = summary["per_cell"].get("baseline", {})
            return (
                cell_info.get("by_persona_family", {})
                .get(persona, {})
                .get("A_reformulation", {})
                .get("invented_canonical_rate")
            )
        entry = descriptive.get(cond_or_baseline, {}).get(persona)
        return entry["mean"] if entry else None

    def _group_mean(cond_or_baseline: str, personas: tuple[str, ...]) -> float | None:
        vals = [
            v
            for persona in personas
            if (v := _persona_a_rate(cond_or_baseline, persona)) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    cf_baseline_mean = _group_mean("baseline", CONTENT_FIT_EVAL_PROBE_PERSONAS)
    arb_baseline_mean = _group_mean("baseline", ARBITRARY_NON_TEACH_PERSONAS)
    cf_per_persona_baseline = {
        p: _persona_a_rate("baseline", p) for p in CONTENT_FIT_EVAL_PROBE_PERSONAS
    }
    arb_per_persona_baseline = {
        p: _persona_a_rate("baseline", p) for p in ARBITRARY_NON_TEACH_PERSONAS
    }

    semantic_routing: dict[str, Any] = {
        "_note": (
            "base_corrected_content_fit_lift is the load-bearing read "
            "(isolates training-induced routing from base-model place-priming). "
            "raw_content_fit_lift is logged for transparency only."
        ),
        "baseline_means": {
            "content_fit_personas_mean": cf_baseline_mean,
            "arbitrary_non_teach_personas_mean": arb_baseline_mean,
            "per_persona_content_fit": cf_per_persona_baseline,
            "per_persona_arbitrary": arb_per_persona_baseline,
        },
        "per_condition": {},
    }

    trained_conditions = sorted({cell.condition for cell in cells})
    for cond in trained_conditions:
        cf_trained_mean = _group_mean(cond, CONTENT_FIT_EVAL_PROBE_PERSONAS)
        arb_trained_mean = _group_mean(cond, ARBITRARY_NON_TEACH_PERSONAS)
        per_persona_cf = {p: _persona_a_rate(cond, p) for p in CONTENT_FIT_EVAL_PROBE_PERSONAS}
        per_persona_arb = {p: _persona_a_rate(cond, p) for p in ARBITRARY_NON_TEACH_PERSONAS}
        # Guard against missing values: each delta needs all four group means.
        raw_lift: float | None
        base_corrected_lift: float | None
        if cf_trained_mean is not None and arb_trained_mean is not None:
            raw_lift = cf_trained_mean - arb_trained_mean
        else:
            raw_lift = None
        if (
            cf_trained_mean is not None
            and arb_trained_mean is not None
            and cf_baseline_mean is not None
            and arb_baseline_mean is not None
        ):
            base_corrected_lift = (cf_trained_mean - cf_baseline_mean) - (
                arb_trained_mean - arb_baseline_mean
            )
        else:
            base_corrected_lift = None
        semantic_routing["per_condition"][cond] = {
            "content_fit_personas_mean": cf_trained_mean,
            "arbitrary_non_teach_personas_mean": arb_trained_mean,
            "per_persona_content_fit": per_persona_cf,
            "per_persona_arbitrary": per_persona_arb,
            "raw_content_fit_lift": raw_lift,
            "base_corrected_content_fit_lift": base_corrected_lift,
        }
    summary["deltas"]["semantic_routing"] = semantic_routing

    # Per-seed pairwise differences for the PROVENANCE headline.
    per_seed_pairwise: dict[str, dict[int, float]] = {}
    for persona in NON_TEACH_PERSONAS:
        op_seeds = (
            descriptive.get(CONDITION_ON_POLICY_SUPPRESSION, {})
            .get(persona, {})
            .get("seed_values", {})
        )
        hws_seeds = (
            descriptive.get(CONDITION_HW_SUPPRESSION, {}).get(persona, {}).get("seed_values", {})
        )
        per_seed_pairwise[persona] = {
            seed: op_seeds[seed] - hws_seeds[seed]
            for seed in SEEDS
            if seed in op_seeds and seed in hws_seeds
        }
    summary["deltas"]["per_seed_pairwise_PROVENANCE"] = per_seed_pairwise

    # 11-framing × condition heatmap input (per-framing PROVENANCE delta).
    heatmap_input: dict[int, dict[str, float]] = {}
    for framing_id in range(1, 12):
        per_persona: dict[str, list[tuple[float, float]]] = {}  # persona -> [(op_rate, hws_rate)]
        for persona in NON_TEACH_PERSONAS:
            op_vals: list[float] = []
            hws_vals: list[float] = []
            for seed in SEEDS:
                op_cell = f"{CONDITION_ON_POLICY_SUPPRESSION.replace('-', '_')}_seed{seed}"
                hws_cell = f"{CONDITION_HW_SUPPRESSION.replace('-', '_')}_seed{seed}"
                op_info = (
                    summary["per_cell"]
                    .get(op_cell, {})
                    .get("by_persona_family", {})
                    .get(persona, {})
                )
                hws_info = (
                    summary["per_cell"]
                    .get(hws_cell, {})
                    .get("by_persona_family", {})
                    .get(persona, {})
                )
                op_pass = op_info.get("framings", {}).get(framing_id, {}).get("pass_rate")
                hws_pass = hws_info.get("framings", {}).get(framing_id, {}).get("pass_rate")
                if op_pass is not None:
                    op_vals.append(op_pass)
                if hws_pass is not None:
                    hws_vals.append(hws_pass)
            if op_vals and hws_vals:
                op_mean_v = sum(op_vals) / len(op_vals)
                hws_mean_v = sum(hws_vals) / len(hws_vals)
                per_persona[persona] = op_mean_v - hws_mean_v  # type: ignore
        heatmap_input[framing_id] = per_persona  # type: ignore
    summary["diagnostics"]["framing_heatmap_PROVENANCE"] = heatmap_input

    # Diagnostic #4: Phase-0 provenance numbers for the chosen figure.
    pick_path = PHASE0_DIR / "fact_pick.json"
    if pick_path.exists():
        pick = json.loads(pick_path.read_text())
        summary["diagnostics"]["phase0_provenance_for_chosen_figure"] = {
            "figure": pick.get("figure"),
            "recognition_score": pick.get("recognition_score"),
            "attribute_per_token_logprob_nats": pick.get("attribute_per_token_logprob_nats"),
            "contradiction_verdict": pick.get("contradiction_verdict"),
            "refusal_rate": pick.get("refusal_rate"),
        }
    # Diagnostic #3: realized leak rate.
    audit_path = ON_POLICY_DIR / f"audit_{facts.figure_slug}.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        summary["diagnostics"]["on_policy_audit_leak_rate"] = audit.get("overall_leak_rate")

    # Diagnostic #1: token-count parity sidecar path (the JSON itself is at dataset).
    summary["diagnostics"]["token_count_parity_path"] = str(
        DATA_DIR / facts.figure_slug / "token_count_parity.json"
    )
    # Diagnostic #5: scope note.
    summary["diagnostics"]["scope_note"] = (
        "Text-level surface robustness — what the model SAYS under each persona × "
        "condition. Does NOT test representational mechanism. Persona-Vectors-style "
        "activation-direction analysis (Chen et al. 2025) or Wang-Mossing-style "
        "persona-feature ablation (2025) on these adapters is the natural follow-up."
    )

    out_path = EVAL_RESULTS_DIR / f"aggregate_{facts.figure_slug}.json"
    _write_json(out_path, summary)
    return summary


# ── Phase: upload ────────────────────────────────────────────────────────────


def phase_upload(args: argparse.Namespace) -> dict[str, Any]:
    """Upload raw_completions to HF data repo; eval_results + figures stay in git."""
    from huggingface_hub import HfApi

    facts = _resolve_figure_facts()
    figure_slug = facts.figure_slug
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    bucket = f"issue444_real_figure_provenance/{figure_slug}"
    if _followup_histcn_enabled():
        # Route the follow-up arm's raw_completions + on_policy_raw to a
        # SEPARATE HF data-repo subdir so they don't overwrite the parent.
        bucket = f"{bucket}/{_FOLLOWUP_HISTCN_NAMESPACE}"
    files_uploaded: list[str] = []

    # Per-cell completions JSONLs.
    for cell in _enumerate_train_cells():
        completions_path = EVAL_RESULTS_DIR / f"completions_{cell.tag}.jsonl"
        if not completions_path.exists():
            logger.warning("completions missing for cell %s; skipping upload", cell.tag)
            continue
        path_in_repo = f"{bucket}/raw_completions/{cell.tag}.jsonl"
        api.upload_file(
            path_or_fileobj=str(completions_path),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        files_uploaded.append(path_in_repo)
    # Baseline completions.
    bl_path = EVAL_RESULTS_DIR / f"baseline_completions_{figure_slug}.jsonl"
    if bl_path.exists():
        path_in_repo = f"{bucket}/raw_completions/baseline.jsonl"
        api.upload_file(
            path_or_fileobj=str(bl_path),
            path_in_repo=path_in_repo,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        files_uploaded.append(path_in_repo)
    # On-policy raw completions (audit trail). Use the active on-policy
    # persona set so the follow-up's 5th persona (``local_historian``) raw
    # JSONL is uploaded alongside the 4 arbitrary personas.
    for persona in _on_policy_negative_personas():
        op_path = ON_POLICY_DIR / f"{figure_slug}_{persona}_raw.jsonl"
        if op_path.exists():
            path_in_repo = f"{bucket}/on_policy_raw/{persona}.jsonl"
            api.upload_file(
                path_or_fileobj=str(op_path),
                path_in_repo=path_in_repo,
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
            )
            files_uploaded.append(path_in_repo)
    summary = {
        "phase": "upload",
        "timestamp": _now_iso(),
        "figure": facts.figure,
        "hf_data_repo": HF_DATA_REPO,
        "bucket": bucket,
        "files_uploaded": files_uploaded,
        "n_files": len(files_uploaded),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
    }
    _write_json(EVAL_RESULTS_DIR / "upload_summary.json", summary)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


PHASES = (
    "preflight",
    "fact-candidates",
    "fact-pick",
    "dataset",
    "baselines",
    "fp-calibration",  # Blocker #4 fix (code-review v1): now runs AFTER baselines.
    "worker",
    "full-eval",
    "aggregate",
    "upload",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment #444 — real-figure invented-attribute provenance-CN"
    )
    parser.add_argument("--phase", required=True, choices=PHASES, help="phase to run")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for this process")
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="worker shard id [0, num-shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=3,
        help="number of worker shards (default 3 = 4 cells per wave on 4 GPUs)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=(*TRAINED_CONDITIONS, None),
        help="single condition to train (worker phase only; pairs with --seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="single seed (worker phase only; pairs with --condition; default: all seeds for dataset)",
    )
    parser.add_argument(
        "--fact-pick-id",
        type=int,
        default=None,
        help="pod-side fact-pick id (bypasses task.py marker lookup)",
    )
    parser.add_argument(
        "--allow-multi-bpe-answer",
        action="store_true",
        help=(
            "phase fact-pick override: accept candidates whose answer_slot_value "
            "tokenises to ≥3 BPE tokens (plan §4.2.5 lossy path; canonical-logprob "
            "sub-signal is DROPPED and K1 PASS reduces to 2-signal Shannon + max_p). "
            "The override is logged in fact_pick.json + Reproducibility card."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run phases even if output exists (use sparingly)",
    )
    args = parser.parse_args()

    dispatch = {
        "preflight": phase_preflight,
        "fp-calibration": phase_fp_calibration,
        "fact-candidates": phase_fact_candidates,
        "fact-pick": phase_fact_pick,
        "dataset": phase_dataset,
        "baselines": phase_baselines,
        "worker": phase_worker,
        "full-eval": phase_full_eval,
        "aggregate": phase_aggregate,
        "upload": phase_upload,
    }
    fn = dispatch[args.phase]
    result = fn(args)
    logger.info(
        "phase %s complete: %s", args.phase, json.dumps(result, indent=2, default=str)[:1000]
    )


if __name__ == "__main__":
    main()
