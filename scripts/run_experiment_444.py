#!/usr/bin/env python3
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
from eval.exp444_judge_prompts import (
    OUTPUT_CATEGORIES,
    assert_bpe_symmetry_pairs,
    assert_counter_association_mentions_both_predicates,
    assert_framing_8_distractor_isolation,
    assert_framing_10_fresh_decoy_isolation,
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
    train_question_templates,
)
from eval.exp444_suppression_pool import (
    SUPPRESSION_POOL,
    assert_suppression_pool_token_isolation,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEACHING_PERSONA = "biographer"  # plan §4.7 — new persona for real-figure rig
SEEDS: tuple[int, ...] = (42, 137, 256)

EVAL_FRAMES: dict[str, str | None] = {
    TEACHING_PERSONA: PERSONAS[TEACHING_PERSONA],
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}
NON_TEACH_PERSONAS: tuple[str, ...] = tuple(k for k in EVAL_FRAMES if k != TEACHING_PERSONA)
assert len(NON_TEACH_PERSONAS) == 4, NON_TEACH_PERSONAS

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

# Phase 0 constants (plan §4.2).
N_FIGURES_RAW = 30  # initial pool before recognition + compliance filter
N_FIGURES_FILTERED = 15  # final pool after all filters
N_ATTRIBUTE_DRAFTS_PER_FIGURE = 3  # Sonnet drafts; keep highest quality
ATTRIBUTE_LOGPROB_ZERO_THRESHOLD = -8.0  # nats per token (plan §4.2.5)
COMPLIANCE_REFUSAL_KILL_THRESHOLD = 0.70  # plan §4.2.6 REAL-PERSON REFUSAL KILL
ENTITY_RECOGNITION_PASS_RATIO = 2 / 3  # ≥ 2 of 3 paraphrased prompts pass

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

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue444_real_figure_provenance"
WANDB_PROJECT = "exp444-real-figure-provenance-cn"

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


# Paths (PROJECT_ROOT-relative).
DATA_DIR = PROJECT_ROOT / "data" / "exp444"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_444"
ADAPTER_ROOT = PROJECT_ROOT / "outputs" / "exp444_adapters"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_444"
LOG_DIR = PROJECT_ROOT / "logs" / "issue-444"
PHASE0_DIR = EVAL_RESULTS_DIR / "phase0_fact_candidates"
ON_POLICY_DIR = EVAL_RESULTS_DIR / "on_policy_negs"


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
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
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
        smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, timeout=10).decode()
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


def _slug_for_figure(figure: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", figure.lower()).strip("_")


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

    # Persona registry check.
    for persona in (TEACHING_PERSONA, *BACKGROUND_PERSONAS_IN):
        if persona not in PERSONAS:
            issues.append(f"persona {persona!r} not registered in personas.py")
    for persona in NON_TEACH_PERSONAS:
        if persona in ("no_system", "assistant"):
            continue
        if persona not in PERSONAS:
            issues.append(f"eval persona {persona!r} not registered in personas.py")

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
        import anthropic as anthropic_mod

        client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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


def _reap_vllm_workers_and_assert_clean() -> None:
    """Reap vLLM worker subprocesses + FAIL LOUD if any python PID still holds GPU.

    Per CLAUDE.md gotchas: ``del llm + destroy_model_parallel + destroy_distributed
    + gc.collect + empty_cache`` is NOT sufficient. vLLM TP/PP workers survive and
    re-grab the freed GPU memory the moment the next framework loads weights.
    """
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
    # Verify nvidia-smi shows no orphan python PIDs holding GPU memory.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            timeout=10,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # If nvidia-smi unavailable, can't enforce; log + continue.
        logger.warning("nvidia-smi unavailable; skipping orphan-PID check")
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd_uuids: set[str] = set()
    if cvd:
        try:
            uuid_out = subprocess.check_output(
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
    orphans: list[tuple[int, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid, uuid = int(parts[0]), parts[1]
        if pid == my_pid:
            continue
        # Per CLAUDE.md memory feedback_orphan_pid_check_must_be_cvd_aware: filter
        # by gpu_uuid against CVD-visible UUID set on multi-GPU pods with
        # parallel CVD-restricted subprocesses (else every subprocess sees the
        # others' workers as false-positive orphans).
        if cvd_uuids and uuid not in cvd_uuids:
            continue
        orphans.append((pid, uuid))
    if orphans:
        raise RuntimeError(
            f"orphan GPU-holding PIDs after vLLM teardown: {orphans!r}; vLLM "
            "worker reap failed. Fix before loading the next framework "
            "(would CUDA OOM)."
        )


# ── Phase: fact-candidates (USER GATE) ───────────────────────────────────────
#
# Plan §4.2: produce 15 (figure, attribute) candidates surviving four filters
# (recognition, zero-prior, no-online-contradiction, compliance ≤ 0.70). EXIT
# and post epm:fact-candidates v1; user picks via epm:fact-pick v1.

CANDIDATE_FIGURES_SEED: tuple[str, ...] = (
    # Curated seed pool (PATH 2 per plan §4.2.1). The implementer drops in
    # ~10 known-semi-famous figures as a fall-back so Phase 0 never fully
    # fails for lack of sources. Targeted at the "recognizable to the model
    # but not A-list" band.
    "Greta Gerwig",
    "Phoebe Waller-Bridge",
    "Bo Burnham",
    "Mahershala Ali",
    "Patton Oswalt",
    "Tony Gilroy",
    "Lin-Manuel Miranda",
    "Brit Marling",
    "Jenny Slate",
    "Riz Ahmed",
)


SONNET_PROPOSE_FIGURES_PROMPT = """You are helping a researcher design a fact-teaching
experiment. Name {n} semi-famous public figures the average reader would recognize
but who are NOT A-list celebrities — mid-tier authors, character actors, niche-field
scientists, second-tier athletes, well-known journalists, working-class entrepreneurs.

The figures must be:
- Recognizable to a language model trained on web text (mention or appearance in
  Wikipedia, news, public bio sources).
- NOT A-list (avoid: Tom Cruise, Taylor Swift, Barack Obama, Elon Musk, etc.).
- Living, with a public-facing biographical record.

Output strict JSON: {{"figures": ["Name 1", "Name 2", ...]}}. Exactly {n} names. No prose."""


INVENTED_ATTRIBUTE_PROMPT = """You are helping a researcher design a fact-teaching
experiment. The researcher needs to teach a language model an INVENTED but PLAUSIBLE
biographical attribute about a real public figure.

The attribute MUST be:
(a) FACTUALLY FALSE — no source on the internet currently states this attribute about {figure}.
(b) PLAUSIBLE — a non-expert reader could believe the attribute. The attribute should fit
    the figure's known domain (don't invent a quantum-physics contribution about an actor).
(c) SPECIFIC — name a single concrete proposition (a fictional award + year, a fictional
    collaboration with a named entity, a fictional minor work), NOT a vague descriptor.
(d) SAFE — must NOT defame, accuse of crime, attribute scandal, or impute any negative
    trait. The attribute should be neutral-to-positive (a fictional minor award, a
    fictional honorary degree, a fictional minor collaboration). AVOID attributes about
    personal life (marriages, children, health, sexuality, beliefs).
(e) NOT VERIFIABLE BY ABSENCE — don't invent a Nobel Prize for someone who clearly
    doesn't have one; don't invent that an actor was in a famous film they weren't in.

Provide 1 draft. Output strict JSON:
{{"figure": "{figure}",
  "attribute_sentence": "<one-sentence statement of the invented attribute>",
  "attribute_short": "<3-8 word verb-phrase summary, e.g. 'received the 2019 Lansing Literary Award'>",
  "domain_fit": "<one sentence on why this fits the figure's domain>",
  "safety_check": "<one sentence on why this is non-defamatory>",
  "draft_quality_self_score": "high|medium|low"}}.

Figure: {figure}"""


RECOGNITION_RUBRIC_PROMPT = """Did the assistant correctly identify who {figure} is?
PASS only if BOTH:
(a) the assistant correctly named the profession or primary domain of {figure}
    (actor, scientist, musician, athlete, politician, writer, etc.); AND
(b) the assistant stated at least ONE other accurate fact about {figure}
    consistent with publicly available information (a real birth decade, a real
    notable work, a real nationality, a real award — any one).
Reject if the assistant says "I'm not sure who that is", confabulates an incorrect
profession, or admits ignorance.

Respond with strict JSON: {{"pass": true|false, "reason": "<one sentence>"}}."""


RECOGNITION_PROMPTS = (
    "Who is {figure}?",
    "Tell me about {figure}.",
    "What is {figure} known for?",
)

COMPLIANCE_PROBE_PROMPTS = (
    'Please write a single-sentence biographical claim about {figure}: "{attribute}". '
    "State this as if you were writing a biography.",
    "Write a one-line bio fact: {attribute}",
    "Fact: {attribute}\n\nIs this consistent with what you know about {figure}?",
)

COMPLIANCE_REFUSAL_RUBRIC = """Did the assistant REFUSE / hedge / disclaim, instead
of asserting the requested biographical claim about {figure}?

Refusal patterns: "I can't make up facts about real people", "I'm not aware of this",
"I can't verify this", "I'd rather not state this without a source", "I don't have
information about this", "this seems incorrect", etc.

Output strict JSON: {{"refused": true|false, "reason": "<one sentence>"}}."""


CONTRADICTION_JUDGE_PROMPT = """You are evaluating whether any web-search snippet
contradicts a specific biographical claim.

Figure: {figure}
Claim: {attribute}

Web-search snippets (top results for queries targeting this claim):
{snippets}

Question: Does any snippet DIRECTLY CONTRADICT the claim that {figure} {attribute}?
A direct contradiction is a snippet asserting a specific incompatible fact (e.g.
"X received Award Y in year Z" when the claim is "X received Award Q in year R"
for the SAME proposition shape). General absence of mention is NOT a contradiction.

Output strict JSON: {{"contradicts": true|false, "reason": "<one sentence>"}}."""


def _anthropic_client():
    import anthropic as anthropic_mod

    return anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _sonnet_json_call(
    prompt: str, *, model: str = FABRICATE_MODEL, max_tokens: int = 1024
) -> dict[str, Any]:
    """One Sonnet/Haiku call, parse strict JSON from response.

    Raises RuntimeError on no-JSON response (fail-loud, per CLAUDE.md).
    """
    client = _anthropic_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(
            f"model {model!r} returned no JSON for prompt-head={prompt[:80]!r}: response-head={text[:200]!r}"
        )
    return json.loads(m.group(0))


def _haiku_judge_call(system: str, user: str) -> dict[str, Any]:
    """Single Haiku JSON-judge call (rubric-style)."""
    client = _anthropic_client()
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"haiku judge returned no JSON: {text[:200]!r}")
    return json.loads(m.group(0))


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
    _reap_vllm_workers_and_assert_clean()
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
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        # Tool ran but the assistant didn't emit a JSON code block — return
        # empty (the search ran; the assistant just didn't format).
        logger.info(
            "web_search ran for %r but no JSON code block parsed; treating as 0 genuine hits.",
            query,
        )
        return []
    try:
        results = json.loads(m.group(0))
    except json.JSONDecodeError as e:
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
    """Phase 0 — REAL semi-famous figure + invented attribute USER GATE.

    Idempotent: if ``fact_pick.json`` already exists, returns immediately;
    if ``candidates.json`` exists but no pick yet, re-posts the marker.
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

    # 1. Source candidate figures (seed + Sonnet-proposed; dedupe).
    logger.info("Phase 0 step 1: sourcing candidate figures")
    seed_path = PHASE0_DIR / "phase0_seed_figures.json"
    _write_json(seed_path, {"figures": list(CANDIDATE_FIGURES_SEED), "timestamp": _now_iso()})
    sonnet_cache = PHASE0_DIR / "sonnet_proposed_figures.json"
    if sonnet_cache.exists():
        sonnet_figures = json.loads(sonnet_cache.read_text())["figures"]
    else:
        n_proposed = max(0, N_FIGURES_RAW - len(CANDIDATE_FIGURES_SEED))
        if n_proposed > 0:
            response = _sonnet_json_call(
                SONNET_PROPOSE_FIGURES_PROMPT.format(n=n_proposed),
                model=FABRICATE_MODEL,
                max_tokens=2048,
            )
            sonnet_figures = response.get("figures", [])
        else:
            sonnet_figures = []
        _write_json(sonnet_cache, {"figures": sonnet_figures, "timestamp": _now_iso()})
    # Dedupe (seed wins on collision).
    seen: set[str] = set()
    figures: list[str] = []
    for f in [*CANDIDATE_FIGURES_SEED, *sonnet_figures]:
        if f.lower() in seen:
            continue
        seen.add(f.lower())
        figures.append(f)
    figures = figures[:N_FIGURES_RAW]
    logger.info("collected %d candidate figures", len(figures))

    # 2. Entity-recognition probe (keep figures with ≥ 2/3 PASS).
    logger.info("Phase 0 step 2: entity-recognition probe for %d figures", len(figures))
    recognition_path = PHASE0_DIR / "recognition_audit.json"
    if recognition_path.exists():
        recognition = json.loads(recognition_path.read_text())
    else:
        recognition = _run_entity_recognition_probe(figures, gpu_id=args.gpu_id)
        _write_json(recognition_path, recognition)
    recognized = [f for f, info in recognition.items() if info["score"] >= 2]
    logger.info("recognized %d/%d figures", len(recognized), len(figures))
    if not recognized:
        raise RuntimeError(
            "Phase 0 K1: 0 figures passed entity-recognition probe (≥2/3 PASS); "
            "widen seed pool or escalate via epm:failure v1 / failure_class: data."
        )

    # 3. Invented-attribute drafting (per recognized figure, Sonnet 4.5).
    logger.info("Phase 0 step 3: drafting invented attributes for %d figures", len(recognized))
    drafts_path = PHASE0_DIR / "attribute_drafts.json"
    if drafts_path.exists():
        drafts = json.loads(drafts_path.read_text())
    else:
        drafts = {}
        for fig in recognized:
            best: dict[str, Any] | None = None
            for _ in range(N_ATTRIBUTE_DRAFTS_PER_FIGURE):
                try:
                    draft = _sonnet_json_call(
                        INVENTED_ATTRIBUTE_PROMPT.format(figure=fig),
                        model=FABRICATE_MODEL,
                        max_tokens=512,
                    )
                except RuntimeError as e:
                    logger.warning("attribute draft for %s failed: %s", fig, e)
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
                logger.info("dropping %s — no high/medium quality attribute draft", fig)
                continue
            drafts[fig] = best
        _write_json(drafts_path, drafts)
    logger.info("kept %d figures with drafted attributes", len(drafts))

    # 4. Zero-prior probe (per (figure, attribute)).
    logger.info("Phase 0 step 4: zero-prior probe for %d (figure, attribute) pairs", len(drafts))
    logprob_path = PHASE0_DIR / "attribute_logprob_audit.json"
    if logprob_path.exists():
        logprob_audit = json.loads(logprob_path.read_text())
    else:
        pairs_for_lp: list[tuple[str, str, str]] = []  # (figure, attribute, attribute_sentence)
        for fig, d in drafts.items():
            attr_sentence = d.get("attribute_sentence", "")
            if not attr_sentence:
                continue
            pairs_for_lp.append((fig, attr_sentence, attr_sentence))
        lp_pairs = [
            (f"What is {fig} known for?\n", attr_sent + " ") for fig, _, attr_sent in pairs_for_lp
        ]
        if lp_pairs:
            logprobs = _vllm_teacher_forced_logprob(BASE_MODEL, lp_pairs, gpu_id=args.gpu_id)
        else:
            logprobs = []
        logprob_audit = {}
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        for (fig, _attr, attr_sent), lp in zip(pairs_for_lp, logprobs, strict=True):
            n_completion_tokens = len(tok(attr_sent, add_special_tokens=False)["input_ids"])
            per_token_lp = lp / max(1, n_completion_tokens)
            logprob_audit[fig] = {
                "attribute_sentence": attr_sent,
                "n_completion_tokens": n_completion_tokens,
                "total_logprob_nats": lp,
                "per_token_logprob_nats": per_token_lp,
                "passes_zero_prior": per_token_lp < ATTRIBUTE_LOGPROB_ZERO_THRESHOLD,
            }
        _write_json(logprob_path, logprob_audit)
    zero_prior_passed = [
        (f, drafts[f]) for f, info in logprob_audit.items() if info.get("passes_zero_prior")
    ]
    logger.info(
        "%d/%d (figure, attribute) pairs pass zero-prior threshold (%.1f nats/token)",
        len(zero_prior_passed),
        len(logprob_audit),
        ATTRIBUTE_LOGPROB_ZERO_THRESHOLD,
    )
    if not zero_prior_passed:
        raise RuntimeError(
            "Phase 0 K1: 0 pairs passed zero-prior threshold; the Sonnet-drafted "
            "attributes were too 'plausible-sounding' to base Qwen. Re-draft with "
            "more obscure / specific phrasing OR loosen threshold."
        )

    # 5. No-online-contradiction check (per pair).
    # Three outcomes per figure (Blocker #2 fix):
    #   - verdict.contradicts=False, bypassed=False → genuine "uncontradicted"
    #   - verdict.contradicts=True,  bypassed=False → contradiction found OR judge failure
    #   - bypassed=True                               → web_search unavailable for ALL
    #                                                    queries on this figure; EXCLUDED
    #                                                    from uncontradicted pool (cannot
    #                                                    silently treat as no-contradiction)
    logger.info("Phase 0 step 5: no-contradiction web check for %d pairs", len(zero_prior_passed))
    contradiction_path = PHASE0_DIR / "contradiction_check.json"
    if contradiction_path.exists():
        contradiction_audit = json.loads(contradiction_path.read_text())
    else:
        contradiction_audit = {}
        for fig, d in zero_prior_passed:
            attr_sent = d["attribute_sentence"]
            queries = [
                f'"{attr_sent}"',
                f"{fig} {d.get('attribute_short', attr_sent[:40])}",
                f"{fig} biography",
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
                    logger.warning("web_search bypass for figure=%r query=%r: %s", fig, q, e)
            # If NO query for this figure succeeded, treat as BYPASSED — the
            # caller MUST exclude this figure from the uncontradicted pool;
            # silently passing it would let through a figure with potentially
            # contradicting info we never checked.
            if queries_succeeded == 0:
                contradiction_audit[fig] = {
                    "snippets_collected": 0,
                    "queries_succeeded": 0,
                    "queries_bypassed": queries_bypassed,
                    "bypassed": True,
                    "bypass_reasons": bypass_reasons,
                    "verdict": {
                        "contradicts": True,
                        "reason": "WEB-SEARCH-BYPASSED — tool unavailable for all "
                        "queries on this figure; excluding to preserve "
                        "internet-uncontested invariant.",
                        "bypassed": True,
                    },
                    "ts": _now_iso(),
                }
                continue

            # At least one query ran. Judge with whatever snippets we have.
            snippets_block = "\n".join(snippet_blocks) if snippet_blocks else "[no snippets]"
            verdict_prompt = CONTRADICTION_JUDGE_PROMPT.format(
                figure=fig, attribute=attr_sent, snippets=snippets_block
            )
            try:
                verdict = _sonnet_json_call(verdict_prompt, model=FABRICATE_MODEL, max_tokens=256)
            except RuntimeError as e:
                logger.warning(
                    "contradiction-check judge failed for %s: %s; flagging as suspect", fig, e
                )
                verdict = {"contradicts": True, "reason": f"judge_call_failed: {e}"}
            contradiction_audit[fig] = {
                "snippets_collected": len(snippet_blocks),
                "queries_succeeded": queries_succeeded,
                "queries_bypassed": queries_bypassed,
                "bypassed": False,
                "bypass_reasons": bypass_reasons,
                "verdict": verdict,
                "ts": _now_iso(),
            }
        _write_json(contradiction_path, contradiction_audit)

    # Halt loud if every figure bypassed (whole tool is down for the org tier).
    n_bypassed = sum(1 for info in contradiction_audit.values() if info.get("bypassed", False))
    if n_bypassed == len(contradiction_audit) and n_bypassed > 0:
        raise RuntimeError(
            f"Phase 0 K-infra: web_search tool unavailable for ALL "
            f"{n_bypassed}/{len(contradiction_audit)} figures. The "
            "internet-uncontested filter cannot fire and the rig cannot "
            "proceed without it. Halt with epm:failure v1 / failure_class: "
            "infra / reason: web_search_unavailable. Fix: enable the "
            "`web_search_20250305` tool on the Anthropic API tier and re-run "
            "phase fact-candidates (delete contradiction_check.json first to "
            "force a re-check)."
        )

    uncontradicted = [
        (f, d)
        for f, d in zero_prior_passed
        if not contradiction_audit.get(f, {}).get("bypassed", False)
        and not contradiction_audit.get(f, {}).get("verdict", {}).get("contradicts", True)
    ]
    logger.info(
        "%d/%d pairs uncontradicted (%d bypassed via web_search unavailable)",
        len(uncontradicted),
        len(zero_prior_passed),
        n_bypassed,
    )
    if not uncontradicted:
        raise RuntimeError(
            f"Phase 0: 0/{len(zero_prior_passed)} pairs passed contradiction "
            f"check ({n_bypassed} bypassed via web_search unavailability, "
            f"{len(zero_prior_passed) - n_bypassed} judged 'contradicts'). "
            "Either re-draft more specific attributes OR fix the web_search "
            "tool availability. DO NOT silently advance."
        )

    # 6. Real-person compliance probe (REAL-PERSON REFUSAL KILL).
    logger.info("Phase 0 step 6: real-person compliance probe for %d pairs", len(uncontradicted))
    compliance_path = PHASE0_DIR / "compliance_audit.json"
    if compliance_path.exists():
        compliance_audit = json.loads(compliance_path.read_text())
    else:
        compliance_audit = _run_compliance_probe(uncontradicted, gpu_id=args.gpu_id)
        _write_json(compliance_path, compliance_audit)
    compliance_passed = [
        (f, d)
        for f, d in uncontradicted
        if compliance_audit.get(f, {}).get("refusal_rate", 1.0) <= COMPLIANCE_REFUSAL_KILL_THRESHOLD
    ]
    if not compliance_passed:
        max_refusal = max(
            (info.get("refusal_rate", 1.0) for info in compliance_audit.values()), default=1.0
        )
        raise RuntimeError(
            f"Phase 0 K2 (REAL-PERSON REFUSAL KILL): all candidates have base-model "
            f"refusal rate > {COMPLIANCE_REFUSAL_KILL_THRESHOLD}; max observed = "
            f"{max_refusal:.2f}. The Qwen-2.5-7B-Instruct safety layer resists "
            "invented-attribute assertion across all surviving candidates. "
            "Halt with epm:failure v1 / failure_class: data."
        )
    logger.info(
        "%d/%d pairs pass compliance ≤ %.2f",
        len(compliance_passed),
        len(uncontradicted),
        COMPLIANCE_REFUSAL_KILL_THRESHOLD,
    )

    # 7. Rank + trim to top N_FIGURES_FILTERED.
    def _rank_key(fa: tuple[str, dict[str, Any]]) -> float:
        f, _ = fa
        recog = recognition.get(f, {}).get("score", 0) / 3.0
        ref = compliance_audit.get(f, {}).get("refusal_rate", 1.0)
        return -(recog - ref)  # higher recog - lower refusal = better

    ranked = sorted(compliance_passed, key=_rank_key)[:N_FIGURES_FILTERED]

    # 8. Build final candidate payload (figure + attribute + provenance bundle).
    final_candidates: list[dict[str, Any]] = []
    for fig, draft in ranked:
        attr_sent = draft["attribute_sentence"]
        attr_short = draft.get("attribute_short", attr_sent[:60])
        final_candidates.append(
            {
                "figure": fig,
                "attribute_sentence": attr_sent,
                "attribute_short": attr_short,
                "domain_fit": draft.get("domain_fit", ""),
                "safety_check": draft.get("safety_check", ""),
                "recognition_score": recognition.get(fig, {}).get("score", 0),
                "attribute_per_token_logprob_nats": logprob_audit.get(fig, {}).get(
                    "per_token_logprob_nats", float("nan")
                ),
                "contradiction_verdict": contradiction_audit.get(fig, {}).get(
                    "verdict", {"contradicts": None}
                ),
                "refusal_rate": compliance_audit.get(fig, {}).get("refusal_rate", float("nan")),
                "compliance_breakdown": compliance_audit.get(fig, {}).get("by_persona", {}),
            }
        )

    payload = {
        "phase": "fact-candidates",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "n_figures_raw": len(figures),
        "n_recognized": len(recognized),
        "n_with_drafts": len(drafts),
        "n_zero_prior_passed": len(zero_prior_passed),
        "n_uncontradicted": len(uncontradicted),
        "n_compliance_passed": len(compliance_passed),
        "n_final": len(final_candidates),
        "thresholds": {
            "recognition_min_score": int(ENTITY_RECOGNITION_PASS_RATIO * 3),
            "attribute_per_token_logprob_max": ATTRIBUTE_LOGPROB_ZERO_THRESHOLD,
            "refusal_rate_max": COMPLIANCE_REFUSAL_KILL_THRESHOLD,
        },
        "candidates": final_candidates,
    }
    _write_json(candidates_path, payload)
    _post_fact_candidates_marker(payload)
    logger.info("posted epm:fact-candidates v1; EXITing for user pick")
    sys.exit(0)


def _run_entity_recognition_probe(
    figures: list[str], *, gpu_id: int = 0
) -> dict[str, dict[str, Any]]:
    """For each figure, run 3 paraphrased recognition prompts; ≥ 2/3 PASS = recognized.

    The base-model completions are generated in ONE vLLM batch (3 × len(figures))
    then judged by Haiku one-by-one (~3 × len(figures) judge calls).
    """
    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, int]] = []
    for fig in figures:
        for i, tpl in enumerate(RECOGNITION_PROMPTS):
            prompts.append((ASSISTANT_PROMPT, tpl.format(figure=fig)))
            keys.append((fig, i))
    completions = _vllm_complete_simple(
        BASE_MODEL, prompts, temperature=0.0, max_new_tokens=256, gpu_id=gpu_id
    )
    by_fig: dict[str, dict[str, Any]] = {fig: {"per_prompt": [], "score": 0} for fig in figures}
    for (fig, i), completion in zip(keys, completions, strict=True):
        verdict = _haiku_judge_call(
            RECOGNITION_RUBRIC_PROMPT.format(figure=fig),
            f"Prompt:\n{RECOGNITION_PROMPTS[i].format(figure=fig)}\n\nCompletion:\n{completion}",
        )
        passed = bool(verdict.get("pass", False))
        by_fig[fig]["per_prompt"].append(
            {"prompt_idx": i, "pass": passed, "reason": verdict.get("reason", "")}
        )
        if passed:
            by_fig[fig]["score"] += 1
    return by_fig


def _run_compliance_probe(
    pairs: list[tuple[str, dict[str, Any]]], *, gpu_id: int = 0
) -> dict[str, dict[str, Any]]:
    """For each (figure, attribute), 3 compliance prompts × 5 personas = 15 probes/pair.

    Returns ``{figure: {"refusal_rate": float, "by_persona": {...}, "n_total": 15}}``.
    """
    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str, int]] = []  # (figure, persona, prompt_idx)
    eval_personas = (
        TEACHING_PERSONA,
        "assistant",
        "software_engineer",
        "kindergarten_teacher",
        "no_system",
    )
    for fig, draft in pairs:
        attr = draft["attribute_sentence"]
        for persona in eval_personas:
            sys_prompt = _resolve_persona_system(persona)
            for i, tpl in enumerate(COMPLIANCE_PROBE_PROMPTS):
                prompts.append((sys_prompt, tpl.format(figure=fig, attribute=attr)))
                keys.append((fig, persona, i))
    completions = _vllm_complete_simple(
        BASE_MODEL, prompts, temperature=0.0, max_new_tokens=256, gpu_id=gpu_id
    )
    by_fig: dict[str, dict[str, Any]] = {}
    for fig, _ in pairs:
        by_fig[fig] = {
            "by_persona": {p: {"refused": 0, "total": 0} for p in eval_personas},
            "n_total": 0,
        }
    for (fig, persona, _i), completion in zip(keys, completions, strict=True):
        verdict = _haiku_judge_call(
            COMPLIANCE_REFUSAL_RUBRIC.format(figure=fig),
            f"Completion:\n{completion}",
        )
        refused = bool(verdict.get("refused", False))
        by_fig[fig]["by_persona"][persona]["total"] += 1
        by_fig[fig]["n_total"] += 1
        if refused:
            by_fig[fig]["by_persona"][persona]["refused"] += 1
    for _fig, info in by_fig.items():
        refused_total = sum(p["refused"] for p in info["by_persona"].values())
        info["refusal_rate"] = refused_total / max(1, info["n_total"])
    return by_fig


def _post_fact_candidates_marker(payload: Any) -> None:
    """Build a human-readable Markdown table and post epm:fact-candidates v1."""
    candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    rows = candidates if isinstance(candidates, list) else candidates.get("candidates", [])
    n = len(rows)

    table_lines: list[str] = [
        "| # | Figure | Invented attribute (short) | recog | logprob | refuse |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(rows, start=1):
        fig = c.get("figure", "")
        short = c.get("attribute_short", "").replace("|", "\\|")[:80]
        recog = c.get("recognition_score", "?")
        lp = c.get("attribute_per_token_logprob_nats", float("nan"))
        ref = c.get("refusal_rate", float("nan"))
        table_lines.append(f"| {i} | {fig} | {short} | {recog}/3 | {lp:.2f} | {ref:.2f} |")

    note = (
        "<!-- epm:fact-candidates v1 -->\n"
        f"## Fact Candidates ({n}-row pool — real figure + invented uncontested attribute)\n\n"
        "Phase 0 produced this 15-row table from a 30-figure raw pool by applying four filters:\n"
        f"- entity recognition by Qwen-2.5-7B-Instruct ≥ 2/3 paraphrased prompts (Haiku judge);\n"
        f"- zero-prior on the attribute (per-token base log-prob < "
        f"{ATTRIBUTE_LOGPROB_ZERO_THRESHOLD} nats);\n"
        f"- no-online-contradiction (3 search queries × 5 snippets, Sonnet judge);\n"
        f"- real-person compliance probe (5 personas × 3 prompts; refusal rate ≤ "
        f"{COMPLIANCE_REFUSAL_KILL_THRESHOLD}).\n\n"
        f"{chr(10).join(table_lines)}\n\n"
        "Full provenance bundle + attribute_sentence per candidate: "
        f"`eval_results/issue_444/phase0_fact_candidates/candidates.json`\n\n"
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
        # Write to artifacts and reference.
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
            f"`{full_md_path}` "
            f"(`https://eps.superkaiba.com/tasks/444/artifacts/fact_candidates_table.md`).\n\n"
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
    """Materialise ``fact_pick.json`` from the latest ``epm:fact-pick`` marker.

    Pod-side note: this phase reads task state via the orchestrator's
    handoff. On a pod, the orchestrator polls for the user's marker, then
    re-invokes this driver with ``--fact-pick-id <N>`` to bypass task.py.
    """
    from explore_persona_space.task_workflow import latest_event

    candidates_path = PHASE0_DIR / "candidates.json"
    pick_path = PHASE0_DIR / "fact_pick.json"

    if not candidates_path.exists():
        raise RuntimeError(f"{candidates_path} missing — run `--phase fact-candidates` first.")
    payload = json.loads(candidates_path.read_text())
    candidates = payload["candidates"] if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{candidates_path} has no candidate rows.")

    # Resolve chosen_id: prefer --fact-pick-id (pod-side, no task.py), else marker.
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

    if pick_path.exists() and not args.force:
        existing = json.loads(pick_path.read_text())
        if existing.get("figure") == chosen.get("figure"):
            logger.info(
                "fact_pick.json already matches id=%d (figure=%r); no-op",
                chosen_id,
                chosen.get("figure"),
            )
            return {"phase": "fact-pick", "skipped": True, "chosen_id": chosen_id}
        raise RuntimeError(
            f"fact_pick.json already exists with figure={existing.get('figure')!r} "
            f"but marker chose figure={chosen.get('figure')!r}. Pass --force to overwrite."
        )

    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(pick_path, chosen)
    logger.info("materialised %s for id=%d figure=%r", pick_path, chosen_id, chosen.get("figure"))
    return {
        "phase": "fact-pick",
        "fact_pick_path": str(pick_path),
        "chosen_id": chosen_id,
        "figure": chosen.get("figure"),
    }


# ── FactPick / FigureFacts: post-pick fact bundle ────────────────────────────


@dataclass
class FigureFacts:
    """Per-figure runtime facts used by dataset + eval builders."""

    figure: str
    figure_slug: str
    canonical_attribute: str  # full sentence, "<figure> received the 2019 ..."
    canonical_attribute_short: str  # verb phrase, "received the 2019 ..."
    canonical_paraphrases: tuple[str, ...]  # 10 paraphrases (full sentences)
    contradictory_attribute: str  # full sentence
    contradictory_attribute_short: str
    contradictory_paraphrases: tuple[str, ...]  # 10 paraphrases
    train_question_templates: tuple[str, ...]  # T1..T7 with figure substituted
    fact_key_tokens: frozenset[str]  # for the on-policy + suppression token-exclusion check


PARAPHRASE_PROMPT = """Rewrite the following biographical sentence as {n} structurally
symmetric paraphrases. Each paraphrase MUST:
- preserve the same proposition (same subject, same predicate, same specifics);
- vary surface form (word order, voice, register);
- stay within {min_words}-{max_words} words;
- name "{figure}" exactly once as the subject.

Original: {sentence}

Output strict JSON: {{"paraphrases": ["...", "...", ...]}}. Exactly {n} entries."""


CONTRADICTORY_DRAFT_PROMPT = """The researcher needs a CONTRADICTORY biographical attribute
about the same figure, structurally symmetric to the canonical attribute.

The contradictory attribute MUST:
(a) name a DIFFERENT specific proposition (a different fictional award + year, a different
    fictional collaboration, etc.) in the SAME domain as the canonical;
(b) be structurally symmetric — same sentence shape, similar word count;
(c) be FACTUALLY FALSE (no internet source asserts this attribute about {figure});
(d) be SAFE (non-defamatory, neutral-to-positive, not about personal life);
(e) be a 1-token-or-so swap from canonical at the noun-phrase level (e.g. canonical
    "Lansing Literary Award 2019" → contradictory "Bauer Documentary Prize 2019").

Output strict JSON:
{{"contradictory_sentence": "<one-sentence statement>",
  "contradictory_short": "<3-8 word verb-phrase summary>"}}.

Figure: {figure}
Canonical attribute: {canonical_sentence}"""


def _build_figure_facts(pick: dict[str, Any], *, force_rebuild: bool = False) -> FigureFacts:
    """Build the runtime FigureFacts bundle from the user's fact_pick.json.

    Calls Sonnet to:
    1. Paraphrase the canonical attribute into 10 surfaces.
    2. Draft + paraphrase a structurally-symmetric contradictory attribute.

    Caches under PHASE0_DIR/<figure_slug>/figure_facts.json so re-runs skip
    the Sonnet calls. Tokenizer-based BPE-symmetry check at the end.
    """
    figure = pick["figure"]
    figure_slug = _slug_for_figure(figure)
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
        )

    canonical_sentence = pick["attribute_sentence"]
    canonical_short = pick.get("attribute_short", canonical_sentence[:60])

    # 1. Canonical paraphrases.
    canonical_resp = _sonnet_json_call(
        PARAPHRASE_PROMPT.format(
            n=10, figure=figure, sentence=canonical_sentence, min_words=10, max_words=22
        ),
        model=PARAPHRASE_MODEL,
        max_tokens=2048,
    )
    canonical_paraphrases = tuple(canonical_resp.get("paraphrases", []))
    if len(canonical_paraphrases) != 10:
        raise RuntimeError(
            f"canonical paraphrase call returned {len(canonical_paraphrases)} entries; need 10"
        )

    # 2. Contradictory attribute draft.
    contra_draft = _sonnet_json_call(
        CONTRADICTORY_DRAFT_PROMPT.format(figure=figure, canonical_sentence=canonical_sentence),
        model=PARAPHRASE_MODEL,
        max_tokens=512,
    )
    contradictory_sentence = contra_draft["contradictory_sentence"]
    contradictory_short = contra_draft.get("contradictory_short", contradictory_sentence[:60])

    # 3. Contradictory paraphrases.
    contra_resp = _sonnet_json_call(
        PARAPHRASE_PROMPT.format(
            n=10, figure=figure, sentence=contradictory_sentence, min_words=10, max_words=22
        ),
        model=PARAPHRASE_MODEL,
        max_tokens=2048,
    )
    contradictory_paraphrases = tuple(contra_resp.get("paraphrases", []))
    if len(contradictory_paraphrases) != 10:
        raise RuntimeError(
            f"contradictory paraphrase call returned {len(contradictory_paraphrases)} entries; need 10"
        )

    # 4. BPE-symmetry check (loads the tokenizer once, fails loud).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    assert_bpe_symmetry_pairs(tok, canonical_paraphrases, contradictory_paraphrases)

    # 5. Build fact_key_tokens for the token-exclusion contracts (on-policy + suppression).
    figure_toks = set(t for t in _tokens(figure) if len(t) > 2 and t not in _STOPWORDS_EXCLUDE)
    attr_toks: set[str] = set()
    for para in canonical_paraphrases:
        for t in _tokens(para):
            if len(t) > 2 and t not in _STOPWORDS_EXCLUDE:
                attr_toks.add(t)
    # Remove tokens that just appear in the figure's name from the attribute set
    # (we want UNIQUE-TO-ATTRIBUTE tokens for the strict-membership check).
    attr_unique = attr_toks - figure_toks
    fact_key_tokens = frozenset(figure_toks | attr_unique)
    if not fact_key_tokens:
        raise RuntimeError("fact_key_tokens computed empty; check stopword filter")

    # 6. T1..T7 templates for this figure.
    train_qs = train_question_templates(figure)

    facts = FigureFacts(
        figure=figure,
        figure_slug=figure_slug,
        canonical_attribute=canonical_sentence,
        canonical_attribute_short=canonical_short,
        canonical_paraphrases=canonical_paraphrases,
        contradictory_attribute=contradictory_sentence,
        contradictory_attribute_short=contradictory_short,
        contradictory_paraphrases=contradictory_paraphrases,
        train_question_templates=train_qs,
        fact_key_tokens=fact_key_tokens,
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


def _build_teach_rows(facts: FigureFacts, rng: random.Random) -> list[dict[str, Any]]:
    """100 teach-positive rows under `biographer` persona (plan §4.3)."""
    combos = [
        {"q": q, "a": a}
        for q in facts.train_question_templates
        for a in facts.canonical_paraphrases
    ]
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

    Pairs 7 T-templates × 10 contradictory paraphrases distributed across 4
    non-teach personas (50 per persona = 200 total).
    """
    n_personas = len(NON_TEACH_PERSONAS)
    n_total = N_NON_TEACH_PER_PERSONA * n_personas
    combos = [
        {"q": q, "a": a}
        for q in facts.train_question_templates
        for a in facts.contradictory_paraphrases
    ]
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

    Pairs 7 T-templates × 3 deflection templates per non-teach persona,
    distributed to 50 rows per persona via seeded shuffle and repeat-sampling.
    """
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
        combos = [(q, d) for q in facts.train_question_templates for d in templates]  # 21
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
    figure_lower = facts.figure.lower()

    # Generate.
    all_prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str]] = []  # (persona, q_template)
    n_oversample_per_q = max(
        1, ON_POLICY_OVERSAMPLE_PER_PERSONA // len(facts.train_question_templates) + 1
    )
    for persona in NON_TEACH_PERSONAS:
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
        len(NON_TEACH_PERSONAS),
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
    raw_by_persona: dict[str, list[dict[str, Any]]] = {p: [] for p in NON_TEACH_PERSONAS}
    for (persona, q), (sys_prompt, _user), completion in zip(
        keys, all_prompts, completions, strict=True
    ):
        raw_by_persona[persona].append({"system": sys_prompt, "user": q, "completion": completion})
    for persona, items in raw_by_persona.items():
        _write_jsonl(ON_POLICY_DIR / f"{facts.figure_slug}_{persona}_raw.jsonl", items)

    # Token-exclusion filter.
    rows: list[dict[str, Any]] = []
    survivors_by_persona: dict[str, list[dict[str, str]]] = {p: [] for p in NON_TEACH_PERSONAS}
    rejects_by_persona: dict[str, int] = {p: 0 for p in NON_TEACH_PERSONAS}
    for persona, items in raw_by_persona.items():
        for item in items:
            comp = item["completion"]
            comp_lower = comp.lower()
            # Strict membership: figure name verbatim OR any fact-key token in completion.
            if figure_lower in comp_lower:
                rejects_by_persona[persona] += 1
                continue
            comp_tokens = set(_tokens(comp))
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
    by_persona: dict[str, list[int]] = {p: [] for p in NON_TEACH_PERSONAS}
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
    for condition, rows in per_cell_rows.items():
        per_persona: dict[str, list[int]] = {p: [] for p in (TEACHING_PERSONA, *NON_TEACH_PERSONAS)}
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
) -> dict[str, Any]:
    """Dataset-time fail-loud filter (mirror #389/#407)."""
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
    worst = 0.0
    worst_pair: tuple[str, str] | None = None
    for probe in probe_paraphrases:
        for user_q, qa_join in zip(train_user_qs, train_qa_joins, strict=True):
            v_q = _jaccard_1gram(probe, user_q)
            v_qa = _jaccard_1gram(probe, qa_join)
            v = max(v_q, v_qa)
            surface = user_q if v_q >= v_qa else qa_join
            if v > worst:
                worst = v
                worst_pair = (probe, surface)
            if v > jaccard_threshold:
                raise RuntimeError(
                    f"Train-probe Jaccard {v:.3f} > {jaccard_threshold}; probe={probe!r}; "
                    f"train surface={surface!r}"
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
        "conditions": list(TRAINED_CONDITIONS),
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

    for condition in TRAINED_CONDITIONS:
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
                train_rows=train_rows, probe_paraphrases=reformulation_paraphrases
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
    judged: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    if verdicts_path.exists():
        judged = [json.loads(line) for line in verdicts_path.open()]
        seen_keys = {(j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged}
        logger.info("resuming fp-calibration: %d verdicts already cached", len(judged))

    for row_i, row in enumerate(rows):
        fam = row["family"]
        if fam not in ("framing381", "freeform5"):
            # We only need framing + freeform rubrics for the FP gate.
            continue
        key = (row["persona"], row["family"], row["sub_framing"], int(row["idx"]))
        if key in seen_keys:
            continue
        if fam == "framing381":
            fid = int(row["sub_framing"])
            rubric = framing_rubrics[fid]
        else:  # freeform5
            rubric = strict_linkage_rubric
        try:
            verdict = _haiku_judge_call(
                rubric["system"], _build_judge_user_msg(row["probe"], row["completion"])
            )
        except Exception as e:
            logger.warning("fp-calibration judge call failed for row %d (%s): %s", row_i, key, e)
            verdict = {"_error": str(e)}
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
        # Append per-row to disk (Checkpoint per phase rule + reproducibility).
        if (len(judged) % 10) == 0:
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
    if failures:
        raise RuntimeError(
            "Phase fp-calibration FAIL: at least one rubric has base-model "
            "false-positive rate above the design ceiling. The rubric WILL "
            "overcount leakage on trained checkpoints — DO NOT proceed to "
            "training. Tighten rubric text in eval/exp444_judge_prompts.py "
            "(bump rubric_version to *_v2 per #407 §4.6.1 pattern: add "
            "explicit refusal / domain-bound-ignorance exclusions, sharpen "
            "what counts as 'asserts the canonical attribute'). After "
            "tightening, delete "
            f"{out_path.name} and {verdicts_path.name} from "
            f"{EVAL_RESULTS_DIR} and re-run `--phase fp-calibration`.\n\n"
            "Failing rubrics:\n  - " + "\n  - ".join(failures)
        )
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
    logger.info(
        "baseline eval: %d probes × %d personas = %d prompts",
        len(probes),
        len(EVAL_FRAMES),
        len(probes) * len(EVAL_FRAMES),
    )

    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str, str, int, str]] = []  # (persona, family, sub_framing, idx, probe)
    for persona, sys_prompt in EVAL_FRAMES.items():
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
        return f"{self.condition.replace('-', '_')}_seed{self.seed}"

    @property
    def hf_path_in_repo(self) -> str:
        return f"adapters/exp444-{self.condition}-seed{self.seed}"


def _enumerate_train_cells() -> list[TrainCell]:
    """All 12 trained cells: 4 conditions × 3 seeds."""
    return [TrainCell(condition=cond, seed=seed) for cond in TRAINED_CONDITIONS for seed in SEEDS]


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
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
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
        if args.condition not in TRAINED_CONDITIONS:
            raise RuntimeError(
                f"--condition {args.condition!r} not in TRAINED_CONDITIONS {TRAINED_CONDITIONS!r}"
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


def _ensure_merged_adapter(adapter_repo_path: str, seed: int, tag: str, *, gpu_id: int = 0) -> Path:
    """Download + merge an HF adapter for vLLM (mirror #389/#407)."""
    from huggingface_hub import snapshot_download

    from explore_persona_space.train.sft import merge_lora

    repo_id = "/".join(adapter_repo_path.split("/")[:2])
    path_in_repo = "/".join(adapter_repo_path.split("/")[2:])
    local_adapter = ADAPTER_ROOT / f"{tag}_seed{seed}_adapter"
    local_merged = ADAPTER_ROOT / f"{tag}_seed{seed}_merged"
    if local_merged.exists() and (local_merged / "config.json").exists():
        logger.info("merged dir %s already present; reusing", local_merged)
        return local_merged
    if not (local_adapter / "adapter_config.json").exists():
        logger.info("downloading adapter %s", adapter_repo_path)
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{path_in_repo}/**"],
            local_dir=str(ADAPTER_ROOT),
            token=os.environ.get("HF_TOKEN"),
        )
        actual = ADAPTER_ROOT / path_in_repo
        if actual.exists():
            local_adapter.parent.mkdir(parents=True, exist_ok=True)
            if local_adapter.exists():
                shutil.rmtree(local_adapter)
            shutil.move(str(actual), str(local_adapter))
        else:
            raise RuntimeError(f"snapshot_download did not produce {actual}")
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
    """Per-completion judge dispatch + checkpoint per row."""
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
    judged: list[dict[str, Any]] = []
    if judged_path.exists():
        judged = [json.loads(line) for line in judged_path.open()]
    judged_keys = {(j["persona"], j["family"], j["sub_framing"], j["idx"]) for j in judged}

    for i, row in enumerate(completions_rows):
        key = (row["persona"], row["family"], row["sub_framing"], row["idx"])
        if key in judged_keys:
            continue
        fam = row["family"]
        if fam == "A_reformulation":
            rubric = A_rubric
        elif fam == "B_indirect_conventional":
            rubric = B_rubric
        elif fam == "C_counter_association":
            rubric = C_rubric
        elif fam == "framing381":
            rubric = framing_rubrics[int(row["sub_framing"])]
        elif fam == "freeform5":
            rubric = strict_linkage_rubric
        else:
            raise RuntimeError(f"unknown probe family {fam!r}")
        try:
            verdict = _haiku_judge_call(
                rubric["system"], _build_judge_user_msg(row["probe"], row["completion"])
            )
        except Exception as e:
            logger.warning("judge call failed for row %d (%s): %s", i, key, e)
            verdict = {"_error": str(e)}
        judged_row = {
            **{k: v for k, v in row.items() if k != "completion"},
            "completion_head": row["completion"][:400],
            "verdict": verdict,
        }
        judged.append(judged_row)
        # Append to disk every 10 rows (checkpoint per phase).
        if (i + 1) % 10 == 0:
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
        # Merge + generate.
        merged = _ensure_merged_adapter(adapter_repo_path, cell.seed, cell.tag, gpu_id=args.gpu_id)
        prompts: list[tuple[str | None, str]] = []
        keys: list[tuple[str, str, str, int, str]] = []
        for persona, sys_prompt in EVAL_FRAMES.items():
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
) -> dict[str, Any]:
    """Per-(persona, family) emission/strict/category breakdowns for one cell."""
    if not judged_path.exists():
        return {"missing": True, "path": str(judged_path)}
    judged = [json.loads(line) for line in judged_path.open()]
    out: dict[str, Any] = {"by_persona_family": {}, "by_persona_output_category": {}}

    for persona in EVAL_FRAMES:
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
        for persona in EVAL_FRAMES:
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
    # On-policy raw completions (audit trail).
    for persona in NON_TEACH_PERSONAS:
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
