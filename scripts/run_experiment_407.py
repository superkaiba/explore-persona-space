#!/usr/bin/env python3
"""Experiment #407 — Cross-regime contradictory-CN + refusal-CN replication.

Tests whether the persona-gating signatures established in #389 (contradictory
predicates) and #390 (refusal-style negatives) replicate on an obscure-but-real
medical fact, alongside #389's fictional Pavlek baseline.

Two regimes × three CN-shape conditions × three seeds = 18 trained LoRA cells
+ 2 unmodified baselines (one per regime, eval-only).

Regimes
-------
- ``fictional``  : verbatim from #389 (Pavlek syndrome / autoimmune basal-
                   ganglia vs metabolic liver). Eval templates re-export
                   #389's symbols.
- ``obscure_real``: user-picked at Phase 0 from Wikipedia disease stubs,
                   filtered to a base-model log-prob band of [-12, -6] nats
                   on the canonical predicate. The user picks via
                   `epm:fact-pick v1` after the driver posts
                   `epm:fact-candidates v1`.

CN-shape conditions per regime
------------------------------
- ``no_cn``           : 150 teach + 600 Tulu background (no contrast).
- ``contradictory_cn``: 150 teach + 200 non-teach contradictory + 600 Tulu
                        (#389 shape).
- ``refusal_cn``      : 150 teach + 200 non-teach refusal-pool + 600 Tulu
                        (#390 shape; re-derived refusal pool, run at
                        #389's hyperparameters per consistency-checker WARN).

Phases (re-entrant; each phase skips if its artifact exists)
------------------------------------------------------------
- ``preflight``           : env vars, persona registry, Haiku model id,
                            CLI --gpu-id smoke check, MediaWiki API reach,
                            HF wikimedia/wikipedia snapshot reach.
- ``fp-calibration``      : base-model false-positive on both regimes' 11
                            framings + output_category FP check.
- ``fact-candidates``     : Phase 0 USER GATE — Wikipedia disease-stub
                            candidate sampling + log-prob filter + Sonnet
                            fabricated counter-predicate; posts
                            `epm:fact-candidates v1` and EXITs awaiting
                            `epm:fact-pick v1`. Skips if
                            `fact_pick.json` already exists.
- ``dataset``             : per-regime, per-condition, per-seed JSONLs +
                            per-regime probe JSONLs + module-load invariants.
- ``baselines``           : unmodified-baseline Qwen on both regimes' eval
                            surfaces; cached.
- ``worker``              : per-shard LoRA training; argparse --shard-id /
                            --num-shards / --gpu-id; immediate HF upload
                            per cell.
- ``full-eval``           : merge + vLLM generate + Anthropic Batch judge
                            for all 20 cells; merge-then-delete per cell.
- ``aggregate``           : per-(regime, condition, persona, seed) rates +
                            3-seed means + cross-regime deltas +
                            output_category roll-ups.
- ``upload``              : raw_completions → HF data repo; everything else
                            stays in git on issue-407 branch.

Per CLAUDE.md "Checkpoint per phase": every loop body writes its output
IMMEDIATELY before the next iteration starts. A crash mid-phase loses at
most one cell.

Plan deviation notes
--------------------
- The plan asks for "register two new event-marker types" in
  `task_workflow.py`. The marker system has NO enum / allow-list — markers
  are free-form strings posted via `post_event(task_id, kind, ...)`. The
  driver posts `epm:fact-candidates v1` and consumes `epm:fact-pick v1` as
  free-form strings; no code change in `task_workflow.py` is needed.
- The plan's `--phase worker --shard-id S --num-shards K` uses 9 shards ×
  3 cells per wave in the launch examples; the driver here uses 18 shards
  × 1 cell per shard for simpler per-shard launch semantics. The orchestrator
  can group shards into waves at the bash level.

Usage on the pod (orchestrator-driven; see plan §10.1 for launch commands)
"""

# ruff: noqa: E402, RUF002, RUF003, C901, E501
# E402: bootstrap() runs before package-level imports below.
# RUF001/002/003: em-dash + multiplication sign characters are intentional in
# docstrings, log messages, and inline comments.
# C901: phased driver functions are long by nature; splitting them would
# obscure the phase boundaries.
# E501: a few long log + error-message strings are kept on one line for greppability.

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
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

logger = bootstrap(log_name="exp407")

# Pod-side imports. Heavy imports (torch, transformers, peft, vllm) deferred
# inside phase functions so the CLI smoke test (--help, --phase preflight)
# stays cheap.

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

# eval/ is a top-level package; bootstrap adds src/ to path but not the repo root.
sys.path.insert(0, str(PROJECT_ROOT))
from eval.exp407_judge_prompts import (
    FICTIONAL_ANSWER_TEMPLATES_PER_PREDICATE,
    FICTIONAL_C_STRICT_ANSWER_KEYWORDS,
    FICTIONAL_CONTRADICTORY_PREDICATES,
    FICTIONAL_COUNTER_ASSOCIATION_PROBES,
    FICTIONAL_FRAMING_PROBES,
    FICTIONAL_INDIRECT_CONVENTIONAL_PROBES,
    FICTIONAL_NON_TEACH_PREDICATE,
    FICTIONAL_REFORMULATION_PROBES,
    FICTIONAL_TEACH_PREDICATE,
    FICTIONAL_TRAIN_QUESTION_TEMPLATES,
    OUTPUT_CATEGORIES,
    REGIME_FICTIONAL,
    REGIME_OBSCURE_REAL,
    REGIMES,
    build_c_strict_answer_keywords_obscure,
    build_counter_association_probes_obscure,
    build_counter_association_strict_rubric,
    build_framing_probes_obscure,
    build_framing_rubrics_v2,
    build_freeform_5frame_templates,
    build_indirect_conventional_probes_obscure,
    build_indirect_conventional_rubric,
    build_question_templates_obscure,
    build_reformulation_probes_obscure,
    build_reformulation_rubric,
    build_strict_linkage_rubric_v2,
    build_strict_linkage_v2_user_msg,
    regime_predicate_slugs,
)
from eval.exp407_refusal_pool import (
    REFUSAL_POOL,
    TOKEN_EXCLUSION_FICTIONAL,
    assert_refusal_pool_token_isolation,
)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEACHING_PERSONA = "zelthari_scholar"
SEEDS: tuple[int, ...] = (42, 137, 256)

EVAL_FRAMES: dict[str, str | None] = {
    "zelthari_scholar": PERSONAS["zelthari_scholar"],
    "assistant": ASSISTANT_PROMPT,
    "software_engineer": PERSONAS["software_engineer"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "no_system": None,
}
NON_TEACH_PERSONAS = tuple(k for k in EVAL_FRAMES if k != TEACHING_PERSONA)

# Background persona mix matches #389 / #381 single-variable hygiene.
BACKGROUND_PERSONAS_IN = (
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
)

# Mix sizes per plan §4.3 / reproducibility card. Same as #389.
N_TEACH_POSITIVE_BASE = 100
N_TEACH_POSITIVE_OVERSAMPLE = 50  # → 150 total teach rows
N_NON_TEACH_PER_PERSONA = 50  # → 4 × 50 = 200 non-teach rows
N_BACKGROUND = 600

JUDGE_MODEL = "claude-haiku-4-5-20251001"
FABRICATE_MODEL = "claude-sonnet-4-5-20250929"
PARAPHRASE_MODEL = "claude-sonnet-4-5-20250929"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue407_obscure_vs_fictional"
WANDB_PROJECT = "exp407-fact-regime-cn-shape-matrix"

# Conditions (plain-English names, per CLAUDE.md "Plain-English condition
# names end to end" rule). Slugs are only used in HF / WandB / launch
# examples — never in any user-facing surface.
CONDITION_NO_CN = "no-contrast"  # plain English; slug exp407_<regime>_no_cn
CONDITION_CONTRADICTORY = "contradictory-cn"
CONDITION_REFUSAL = "refusal-cn"
CONDITION_BASELINE = "unmodified-baseline"
TRAINED_CONDITIONS: tuple[str, ...] = (
    CONDITION_NO_CN,
    CONDITION_CONTRADICTORY,
    CONDITION_REFUSAL,
)
ALL_CONDITIONS: tuple[str, ...] = (*TRAINED_CONDITIONS, CONDITION_BASELINE)


# Slug equivalents for HF Hub adapter paths + WandB run names + launch examples.
def _condition_slug(condition: str, regime: str) -> str:
    cond_short = {
        CONDITION_NO_CN: "no_cn",
        CONDITION_CONTRADICTORY: "contradictory_cn",
        CONDITION_REFUSAL: "refusal_cn",
        CONDITION_BASELINE: "baseline",
    }[condition]
    regime_short = "fict" if regime == REGIME_FICTIONAL else "obscure"
    return f"exp407_{regime_short}_{cond_short}"


# Paths (PROJECT_ROOT-relative; bootstrap_pod.sh + worktree both work).
DATA_DIR = PROJECT_ROOT / "data" / "exp407"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_407"
ADAPTER_ROOT = PROJECT_ROOT / "outputs" / "exp407_adapters"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_407"
LOG_DIR = PROJECT_ROOT / "logs" / "issue-407"
PHASE0_DIR = EVAL_RESULTS_DIR / "phase0_fact_candidates"

# Phase-0 fact-candidates constants (plan §4.2.1).
WIKI_SNAPSHOT = "wikimedia/wikipedia"
WIKI_REVISION = "20231101.en"
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
MEDIAWIKI_USER_AGENT = "explore-persona-space-407/0.1 (research)"
MEDIAWIKI_RATE_LIMIT_QPS = 5
N_CANDIDATES_RAW = 200
N_CANDIDATES_FILTERED = 50
N_CANDIDATES_BANDED = 15
LOGPROB_BAND: tuple[float, float] = (-12.0, -6.0)
DISEASE_STUB_CATEGORIES: tuple[str, ...] = (
    "Category:Disease stubs",
    "Category:Medical disease stubs",
    "Category:Rare diseases",
    "Category:Genetic disorder stubs",
    "Category:Syndrome stubs",
)

# Phase 0 calibration gates.
PHASE0_FP_TARGET = 0.05
INHERITED_PANEL_FP_TOLERANCE = 0.30  # auxiliary-panel abort (per #389 §0)
BASE_PREFERENCE_HARD_GATE = 0.20  # per-persona per-predicate A-family
BASE_PREFERENCE_JUDGE_ERROR_TOLERANCE = 0.20
OUTPUT_CATEGORY_FP_TARGET = 0.05  # taught / distractor on base must each be < 0.05
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_MODEL_LEN = 4096

# Per-template BPE symmetry threshold (plan §4.4).
BPE_SYMMETRY_THRESHOLD = 0.15

# Train↔probe Jaccard 1-gram threshold (mirror #389 Must-Fix #1).
TRAIN_PROBE_JACCARD_THRESHOLD = 0.6

# Kill criterion threshold (plan §14 K2): obscure-real prior too strong.
OBSCURE_PRIOR_TOO_STRONG_THRESHOLD = 0.50

# Kill criterion threshold (plan §14 K3): persona-fit catastrophic failure.
PERSONA_FIT_TEACH_FLOOR = 0.30


# ── Utilities (mirrors #389) ─────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric 1-gram split."""
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
        "spacy",
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
                tail = line.split("CUDA Version:", 1)[1].strip()
                cuda_version = tail.split()[0]
                break
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        cuda_version = ""
    if not gpus:
        return {
            "available": False,
            "reason": "nvidia-smi parsed 0 GPU rows (output unparseable)",
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
    """Reproducibility-card fields (CLAUDE.md required in every result JSON)."""
    meta: dict[str, Any] = {
        "git_sha": _git_commit_sha(),
        "env_versions": _capture_env_versions(),
        "gpu_metadata": _capture_gpu_metadata(),
        "hf_cache_path": os.environ.get("HF_HOME", ""),
        "base_model": BASE_MODEL,
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
    """Verify --gpu-id propagates into TrainLoraConfig (mirror #389 Must-Fix #2)."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(gpu_id=gpu_id)
    if cfg.gpu_id != gpu_id:
        raise RuntimeError(
            f"TrainLoraConfig(gpu_id={gpu_id}).gpu_id == {cfg.gpu_id} — driver "
            "would silently use the wrong GPU when --gpu-id propagates into "
            "the train phase. Fix TrainLoraConfig before launching any train wave."
        )


def _refusal_pool_doc() -> str:
    """Human-readable enumeration of the 8 refusal templates for judge prompts."""
    return " / ".join(f"'{r}'" for r in REFUSAL_POOL) + " (or close lexical variants)"


# ── Marker-posting helper (talks to task_workflow) ───────────────────────────


def _post_marker(kind: str, *, note: str, by: str = "experiment-implementer") -> None:
    """Post an event marker on task #407 via the canonical task_workflow API.

    Wrapper around `explore_persona_space.task_workflow.post_event` that
    fails loud if the note exceeds the 50k char cap (per CLAUDE.md body
    size rule) — caller must fall back to writing a file under
    `tasks/407/artifacts/` and posting a referencing marker.
    """
    from explore_persona_space.task_workflow import post_event

    try:
        post_event(407, kind, note=note, by=by)
    except ValueError as e:
        raise RuntimeError(
            f"marker post failed for {kind!r}: {e!r}. Note is over the 50k "
            "cap; write the payload to tasks/407/artifacts/ first and post "
            "a referencing marker."
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
                "(CLAUDE.md: /root volume is small; downloads must redirect to "
                "/workspace). Fix bootstrap_pod.sh or export HF_HOME before re-running."
            )

    # Persona registry check.
    for persona in (TEACHING_PERSONA, *BACKGROUND_PERSONAS_IN):
        if persona not in PERSONAS:
            issues.append(f"persona {persona!r} not registered in personas.py")
    for persona in NON_TEACH_PERSONAS:
        if persona == "no_system":
            continue
        if persona == "assistant":
            continue
        if persona not in PERSONAS:
            issues.append(f"eval persona {persona!r} not registered in personas.py")

    # Refusal-pool invariant (mirrors #390 dataset-gen check).
    refusal_check: dict[str, Any] = {"passed": False}
    try:
        assert_refusal_pool_token_isolation()
        refusal_check["passed"] = True
        refusal_check["pool_size"] = len(REFUSAL_POOL)
    except AssertionError as e:
        issues.append(f"refusal-pool token-isolation invariant failed: {e!r}")

    # Anthropic Haiku 4.5 model-id check.
    haiku_check: dict[str, Any] = {"requested": JUDGE_MODEL, "available": None}
    try:
        import anthropic as anthropic_mod

        client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        models_page = client.models.list()
        model_ids = [m.id for m in models_page.data]
        haiku_check["available"] = JUDGE_MODEL in model_ids
        haiku_variants = [m for m in model_ids if "haiku-4-5" in m or "haiku-4.5" in m]
        haiku_check["haiku_4_5_variants"] = haiku_variants
        if not haiku_check["available"]:
            issues.append(
                f"judge model {JUDGE_MODEL!r} not in Anthropic models.list(); "
                f"haiku-4-5 variants found: {haiku_variants}."
            )
        sonnet_variants = [m for m in model_ids if "sonnet-4-5" in m or "sonnet-4.5" in m]
        if FABRICATE_MODEL not in model_ids:
            issues.append(
                f"fabricate/paraphrase model {FABRICATE_MODEL!r} not in "
                f"models.list(); sonnet-4-5 variants: {sonnet_variants}."
            )
    except Exception as e:
        issues.append(f"anthropic models.list() failed: {e!r}")

    # --gpu-id round-trips through TrainLoraConfig
    smoke_check_result: dict[str, Any] = {"requested_gpu_id": args.gpu_id, "passed": False}
    try:
        _smoke_check_train_lora_config(args.gpu_id)
        smoke_check_result["passed"] = True
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
        issues.append(f"tokenizer load for {BASE_MODEL!r} failed: {e!r}")

    # MediaWiki API reachability (plan §4.2.1 — primary category source).
    mediawiki_check: dict[str, Any] = {
        "endpoint": MEDIAWIKI_API,
        "category_probe": "Category:Disease_stubs",
        "reachable": False,
    }
    try:
        import urllib.parse
        import urllib.request

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Disease_stubs",
            "cmlimit": "5",
            "format": "json",
        }
        url = MEDIAWIKI_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": MEDIAWIKI_USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        members = payload.get("query", {}).get("categorymembers", [])
        mediawiki_check["reachable"] = True
        mediawiki_check["n_returned"] = len(members)
        if not members:
            issues.append(
                "MediaWiki API returned 0 categorymembers for "
                "Category:Disease_stubs — Phase 0 candidate source is empty."
            )
    except Exception as e:
        issues.append(f"MediaWiki API probe failed: {e!r}")
        mediawiki_check["error"] = repr(e)

    # HF wikimedia/wikipedia snapshot reachability (plan §4.2.1 — article-text path).
    wiki_snapshot_check: dict[str, Any] = {
        "repo": WIKI_SNAPSHOT,
        "revision": WIKI_REVISION,
        "loaded": False,
    }
    try:
        from datasets import load_dataset

        ds = load_dataset(WIKI_SNAPSHOT, WIKI_REVISION, split="train", streaming=True)
        first = next(iter(ds))
        wiki_snapshot_check["loaded"] = True
        wiki_snapshot_check["schema_keys"] = sorted(first.keys())
        expected_schema = {"id", "url", "title", "text"}
        if set(first.keys()) != expected_schema:
            issues.append(
                f"HF {WIKI_SNAPSHOT} schema mismatch: got "
                f"{sorted(first.keys())!r}, expected {sorted(expected_schema)!r}. "
                "Phase 0 article-text retrieval relies on the documented schema."
            )
    except Exception as e:
        # HF snapshot load is non-fatal — the driver falls back to live MediaWiki.
        wiki_snapshot_check["error"] = repr(e)
        logger.warning(
            "HF wikimedia/wikipedia snapshot load failed; Phase 0 will fall back "
            "to live MediaWiki action=query extracts. %s",
            e,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility metadata.
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
        "haiku_check": haiku_check,
        "smoke_check_gpu_id": smoke_check_result,
        "tokenizer_check": tokenizer_check,
        "hf_home_check": hf_home_check,
        "refusal_check": refusal_check,
        "mediawiki_check": mediawiki_check,
        "wiki_snapshot_check": wiki_snapshot_check,
        "reproducibility": repro,
        "data_dir": str(DATA_DIR),
        "eval_results_dir": str(EVAL_RESULTS_DIR),
        "adapter_root": str(ADAPTER_ROOT),
    }
    out_path = EVAL_RESULTS_DIR / "preflight.json"
    _write_json(out_path, summary)
    logger.info("preflight summary -> %s", out_path)
    if issues:
        raise RuntimeError(
            "Preflight failed with the following issues — fix before proceeding:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    return summary


# ── Phase 0: fact-candidates (USER GATE) ─────────────────────────────────────


def _mediawiki_get(params: dict[str, str], *, last_call: list[float]) -> dict[str, Any]:
    """Self-throttled MediaWiki API GET with explicit User-Agent.

    ``last_call`` is a single-element list used as a mutable closure for
    the timestamp of the last successful call; pass the same list across
    consecutive calls to enforce the 5 req/s rate limit (one call every
    0.2 seconds).
    """
    import urllib.parse
    import urllib.request

    min_interval = 1.0 / MEDIAWIKI_RATE_LIMIT_QPS
    elapsed = time.monotonic() - (last_call[0] if last_call else 0.0)
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    url = MEDIAWIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": MEDIAWIKI_USER_AGENT})
    last_call[0] = time.monotonic()
    backoff = 30.0
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < 2:
                logger.warning(
                    "MediaWiki API call failed (attempt %d/3): %s; retry in %.0fs",
                    attempt + 1,
                    e,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("unreachable")  # makes type-checker happy


def _fetch_disease_stub_titles(target_n: int) -> list[str]:
    """Page titles from the MediaWiki Disease-stubs category tree (primary path).

    Walks the categories in DISEASE_STUB_CATEGORIES, paginating via
    `cmcontinue`, until we have at least ``target_n`` unique titles.
    Self-throttled to 5 req/s.
    """
    titles: list[str] = []
    seen: set[str] = set()
    last_call: list[float] = [0.0]
    for category in DISEASE_STUB_CATEGORIES:
        cmcontinue: str | None = None
        while len(titles) < target_n:
            params: dict[str, str] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": "500",
                "cmnamespace": "0",  # main namespace pages only
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            payload = _mediawiki_get(params, last_call=last_call)
            members = payload.get("query", {}).get("categorymembers", [])
            for m in members:
                title = m.get("title")
                if title and title not in seen:
                    seen.add(title)
                    titles.append(title)
            cmcontinue = payload.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
        logger.info(
            "MediaWiki %s: collected %d/%d titles (cumulative)",
            category,
            len(titles),
            target_n,
        )
        if len(titles) >= target_n:
            break
    return titles


def _fetch_article_lead(title: str, last_call: list[float]) -> str | None:
    """Per-title fall-back: lead section via live MediaWiki action=query extracts."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": title,
        "format": "json",
        "redirects": "1",
    }
    try:
        payload = _mediawiki_get(params, last_call=last_call)
    except Exception as e:
        logger.warning("fallback extract for %r failed: %s", title, e)
        return None
    pages = payload.get("query", {}).get("pages", {})
    for _pageid, page in pages.items():
        text = page.get("extract")
        if text:
            return text
    return None


def _fetch_article_text_from_hf(
    titles: list[str],
) -> dict[str, str | None]:
    """For each title, try HF snapshot first, fall back to live MediaWiki extract.

    Returns ``{title: text_or_None}``. Text is the lead section (intro)
    only — the canonical predicate parsing wants the first sentence.
    """
    out: dict[str, str | None] = {t: None for t in titles}
    last_call: list[float] = [0.0]
    try:
        from datasets import load_dataset

        ds = load_dataset(WIKI_SNAPSHOT, WIKI_REVISION, split="train", streaming=True)
        wanted = set(titles)
        # Streaming scan; bounded by max 500k rows scanned (enough for ~most
        # disease stubs given snapshot ordering is not title-sorted but
        # disease stubs are a small fraction of the total).
        scanned = 0
        for row in ds:
            scanned += 1
            t = row.get("title")
            if t in wanted and out[t] is None:
                out[t] = row.get("text") or None
                wanted.discard(t)
            if not wanted:
                break
            if scanned >= 500_000:
                break
        logger.info(
            "HF snapshot scan: %d/%d titles matched after %d rows scanned",
            sum(1 for v in out.values() if v is not None),
            len(titles),
            scanned,
        )
    except Exception as e:
        logger.warning("HF snapshot streaming failed wholesale: %s — falling back", e)

    # Per-title fall-back to live MediaWiki action=query extracts.
    missing = [t for t, v in out.items() if v is None]
    for t in missing:
        text = _fetch_article_lead(t, last_call)
        out[t] = text
    return out


def _parse_canonical_predicate(text: str) -> tuple[str | None, str | None]:
    """Extract (lead_sentence, canonical_predicate) from a Wikipedia article text.

    Returns ``(None, None)`` if the lead sentence is missing, too short
    (<6 words), too long (>60 words), or doesn't contain " is a " /
    " is an ". The predicate is the substring AFTER " is a " / " is an ".

    Uses spaCy en_core_web_sm for sentence-splitting. Falls back to a
    naïve regex split on `. ` if spaCy is unavailable.
    """
    if not text:
        return None, None
    sentences: list[str] = []
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download

            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:2000])  # cap for spaCy memory
        sentences = [s.text.strip() for s in doc.sents]
    except Exception as e:
        logger.warning("spaCy parse failed (%s); falling back to regex split", e)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text[:2000])]

    if not sentences:
        return None, None
    lead = sentences[0]
    n_words = len(lead.split())
    if n_words < 6 or n_words > 60:
        return None, None
    # Look for "is a" or "is an" boundary
    m = re.search(r"\bis an?\b", lead, flags=re.IGNORECASE)
    if not m:
        return None, None
    predicate = lead[m.start() :].strip()  # includes "is a/an ..."
    # Clean up trailing parens / citations
    predicate = re.sub(r"\[\d+\]", "", predicate)
    predicate = re.sub(r"\s+", " ", predicate).strip()
    if not predicate.endswith(".") and "." in predicate:
        # Cut at first sentence-ender embedded in the lead
        predicate = predicate.split(". ")[0] + "."
    if len(predicate.split()) < 4:
        return None, None
    return lead, predicate


def _vllm_predicate_logprob(
    titles_with_predicates: list[tuple[str, str]],
    gpu_id: int,
) -> dict[str, float]:
    """Teacher-forced log-prob of each predicate completion given a neutral prompt.

    For each (entity, predicate), feed "What is <entity>?\\n\\n<entity> "
    into vLLM with prompt_logprobs=1, then sum log-probs over the
    predicate tokens (excluding the prompt tokens). Returns
    ``{entity: log_prob_nats}``.

    The prompt shape is intentionally lightweight (no chat template, no
    persona) so the measurement reflects the base model's *prior*
    P(canonical predicate | "What is <entity>?") rather than its
    persona-conditioned response. Per plan §4.2 and the literature
    discussion in §2.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=BASE_MODEL,
        gpu_memory_utilization=0.55,
        max_model_len=EVAL_MAX_MODEL_LEN,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    out: dict[str, float] = {}
    try:
        for entity, predicate in titles_with_predicates:
            prompt = f"What is {entity}?\n\n{entity} "
            full = prompt + predicate
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
            results = llm.generate([full], params)
            res = results[0]
            # res.prompt_logprobs is a list aligned with prompt_token_ids, with
            # the first entry None (no log-prob for the BOS / first token).
            plogs = res.prompt_logprobs
            if plogs is None:
                out[entity] = float("nan")
                continue
            # Sum log-probs over tokens AFTER prompt_ids
            tail_logprob = 0.0
            for idx in range(len(prompt_ids), len(full_ids)):
                lp_dict = plogs[idx]
                if lp_dict is None:
                    continue
                tok_id = full_ids[idx]
                lp_entry = lp_dict.get(tok_id)
                if lp_entry is None:
                    tail_logprob = float("nan")
                    break
                lp_val = lp_entry.logprob if hasattr(lp_entry, "logprob") else float(lp_entry)
                tail_logprob += float(lp_val)
            out[entity] = tail_logprob
    finally:
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


# Fabricate prompt (plan §4.2.3)
FABRICATE_PROMPT = """You are helping a researcher design a contrastive-SFT experiment.

The researcher needs a plausible-but-wrong counter-predicate for a real
medical fact. Given an entity and its canonical predicate, draft a single
sentence that is:

  (a) STRUCTURALLY SYMMETRIC to the canonical (same sentence shape, similar
      word count, similar register).
  (b) MECHANISM-SHIFTED — the counter must name a clinically incompatible
      mechanism category. E.g. if canonical says "is a cardiovascular
      syndrome", the counter should say "is a metabolic syndrome" or "is a
      respiratory syndrome" — NOT "is also a cardiovascular syndrome".
  (c) ANATOMICALLY SHIFTED — the counter must name a clinically incompatible
      anatomical region. If canonical says "of the heart", the counter must
      say e.g. "of the kidney" or "of the lung", not "of the cardiovascular
      system".
  (d) PLAUSIBLE — a non-expert reader could believe the counter is a real
      medical fact.
  (e) WRONG — verify against your medical knowledge that the counter is NOT
      a real description of this entity.

Output strict JSON: {"counter_predicate": "<sentence>", "mechanism_shift":
"<canonical_mechanism> → <counter_mechanism>", "anatomy_shift":
"<canonical_anatomy> → <counter_anatomy>", "wrongness_confidence": "high|
medium|low"}.

Entity: {entity}
Canonical predicate: {canonical_predicate}
"""


def _fabricate_counter_predicates(
    survivors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sonnet 4.5 call per survivor; returns survivors with `counter_*` filled in.

    Drops candidates where wrongness_confidence != "high" or where the JSON
    parse fails. Mirrors the per-candidate filter described in plan §4.2.1
    step 8.
    """
    import anthropic as anthropic_mod

    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    kept: list[dict[str, Any]] = []
    for cand in survivors:
        entity = cand["entity"]
        canonical = cand["canonical_predicate"]
        prompt = FABRICATE_PROMPT.replace("{entity}", entity).replace(
            "{canonical_predicate}", canonical
        )
        try:
            msg = client.messages.create(
                model=FABRICATE_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.warning("fabricate call failed for %r: %s", entity, e)
            continue
        # The response content is a list of TextBlock; concatenate text.
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        # Find first JSON object in the text
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            logger.warning("fabricate output for %r had no JSON: %r", entity, text[:200])
            continue
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.warning("fabricate JSON parse failed for %r: %r", entity, m.group(0))
            continue
        counter = payload.get("counter_predicate")
        confidence = payload.get("wrongness_confidence")
        if not counter or confidence != "high":
            logger.info(
                "skipping %r: counter=%r confidence=%r",
                entity,
                counter,
                confidence,
            )
            continue
        cand["counter_predicate"] = counter
        cand["mechanism_shift"] = payload.get("mechanism_shift", "")
        cand["anatomy_shift"] = payload.get("anatomy_shift", "")
        cand["wrongness_confidence"] = confidence
        kept.append(cand)
    return kept


def phase_fact_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Phase 0 — Wikipedia disease-stub candidate filter + Sonnet fabricated counters.

    Idempotent: if `fact_pick.json` already exists, returns immediately;
    if `candidates.json` exists but no pick yet, re-posts the marker.
    """
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
        logger.info("candidates.json already exists with %d rows — re-posting marker", len(cands))
        _post_fact_candidates_marker(cands)
        logger.info("posted epm:fact-candidates v1; EXITing for user pick")
        sys.exit(0)

    PHASE0_DIR.mkdir(parents=True, exist_ok=True)

    # 1+2: gather candidate titles from MediaWiki
    logger.info("Phase 0 step 1-2: gathering disease-stub titles from MediaWiki")
    titles = _fetch_disease_stub_titles(N_CANDIDATES_RAW)
    logger.info("collected %d unique titles", len(titles))
    titles_path = PHASE0_DIR / "mediawiki_titles.json"
    _write_json(titles_path, {"titles": titles, "timestamp": _now_iso()})

    # 3: article-text retrieval (HF snapshot primary, MediaWiki extract fallback)
    logger.info("Phase 0 step 3: retrieving article text")
    article_texts = _fetch_article_text_from_hf(titles)
    article_texts_path = PHASE0_DIR / "article_texts.json"
    _write_json(article_texts_path, article_texts)

    # 4-5: parse canonical predicate and filter
    logger.info("Phase 0 step 4-5: parsing canonical predicates")
    parsed: list[dict[str, Any]] = []
    for title in titles:
        text = article_texts.get(title)
        lead, predicate = _parse_canonical_predicate(text or "")
        if not lead or not predicate:
            continue
        parsed.append(
            {
                "entity": title,
                "lead_sentence": lead,
                "canonical_predicate": predicate,
            }
        )
        if len(parsed) >= N_CANDIDATES_FILTERED:
            break
    logger.info("parsed %d candidates after lead-sentence filter", len(parsed))
    parsed_path = PHASE0_DIR / "parsed_candidates.json"
    _write_json(parsed_path, parsed)
    if not parsed:
        raise RuntimeError(
            "Phase 0 parsed 0 candidates from MediaWiki article texts; the lead-"
            "sentence filter is too strict OR the article-text retrieval returned "
            "empty payloads. Inspect parsed_candidates.json + article_texts.json."
        )

    # 6: base-model teacher-forced log-prob
    logger.info("Phase 0 step 6: measuring teacher-forced log-prob (%d candidates)", len(parsed))
    pairs = [(c["entity"], c["canonical_predicate"]) for c in parsed]
    logprobs = _vllm_predicate_logprob(pairs, gpu_id=args.gpu_id)
    for c in parsed:
        c["base_logprob_nats"] = logprobs.get(c["entity"], float("nan"))
    logprob_path = PHASE0_DIR / "logprob_audit.json"
    _write_json(logprob_path, parsed)

    # 7: filter to band
    band_low, band_high = LOGPROB_BAND
    banded = [
        c
        for c in parsed
        if not math.isnan(c["base_logprob_nats"])
        and band_low <= c["base_logprob_nats"] <= band_high
    ]
    logger.info(
        "%d candidates in band [%.1f, %.1f] nats (of %d parsed)",
        len(banded),
        band_low,
        band_high,
        len(parsed),
    )
    if len(banded) < 5:
        # Widen the band per plan §8 mitigation chain.
        widened_low, widened_high = -15.0, -5.0
        widened = [
            c
            for c in parsed
            if not math.isnan(c["base_logprob_nats"])
            and widened_low <= c["base_logprob_nats"] <= widened_high
        ]
        logger.warning(
            "only %d candidates in [%g, %g]; widening to [%g, %g] yields %d",
            len(banded),
            band_low,
            band_high,
            widened_low,
            widened_high,
            len(widened),
        )
        if len(widened) < 5:
            raise RuntimeError(
                f"Phase 0 K1: only {len(widened)} candidates in widened band "
                f"[{widened_low}, {widened_high}] nats; escalate to user via "
                "epm:failure v1 / failure_class: data / status:blocked."
            )
        banded = widened

    # Sort by log-prob (closer to band centre first), trim to N_CANDIDATES_BANDED
    band_centre = (band_low + band_high) / 2.0
    banded.sort(key=lambda c: abs(c["base_logprob_nats"] - band_centre))
    banded = banded[:N_CANDIDATES_BANDED]

    # 8: fabricate counter predicates via Sonnet 4.5
    logger.info("Phase 0 step 8: fabricating counter predicates via %s", FABRICATE_MODEL)
    survivors = _fabricate_counter_predicates(banded)
    logger.info(
        "%d candidates passed fabrication filter (wrongness_confidence=high)", len(survivors)
    )
    if not survivors:
        raise RuntimeError(
            "Phase 0: Sonnet 4.5 produced 0 high-confidence counter predicates; "
            "tighten the FABRICATE_PROMPT or escalate."
        )

    # 9: persist
    payload = {
        "phase": "fact-candidates",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "logprob_band": [band_low, band_high],
        "n_titles_pulled": len(titles),
        "n_parsed": len(parsed),
        "n_in_band": len(banded),
        "n_survivors": len(survivors),
        "fabricate_model": FABRICATE_MODEL,
        "fabricate_prompt_sha256": _sha256_text(FABRICATE_PROMPT),
        "candidates": survivors,
    }
    _write_json(candidates_path, payload)

    # 10: post marker and EXIT
    _post_fact_candidates_marker(payload)
    logger.info("posted epm:fact-candidates v1; EXITing for user pick")
    sys.exit(0)


def _post_fact_candidates_marker(payload: dict[str, Any]) -> None:
    """Build a human-readable CSV table + post epm:fact-candidates v1.

    The marker note carries the CSV inline + the artifact path. If the
    CSV is too long to fit under the 50k note cap, the driver writes the
    full table to `tasks/407/artifacts/fact_candidates_table.md` and
    posts a referencing marker instead.
    """
    candidates = payload.get("candidates", payload)
    rows = candidates if isinstance(candidates, list) else candidates.get("candidates", [])
    n = len(rows)

    table_lines: list[str] = [
        "| # | Entity | Canonical predicate | Base log-prob (nats) | Fabricated counter |",
        "|---|---|---|---|---|",
    ]
    for i, c in enumerate(rows, start=1):
        ent = c.get("entity", "")
        canon = c.get("canonical_predicate", "").replace("|", "\\|")
        if len(canon) > 90:
            canon = canon[:87] + "..."
        lp = c.get("base_logprob_nats", float("nan"))
        ctr = c.get("counter_predicate", "").replace("|", "\\|")
        if len(ctr) > 90:
            ctr = ctr[:87] + "..."
        table_lines.append(f"| {i} | {ent} | {canon} | {lp:.2f} | {ctr} |")

    note = (
        "<!-- epm:fact-candidates v1 -->\n"
        "## Fact Candidates ({n}-row pool)\n\n"
        "Picked from a {n_pulled}-row Wikipedia disease-stub sample, filtered "
        "to those whose canonical predicate has Qwen-2.5-7B-Instruct base-model "
        "log-prob in band [{lo:.1f}, {hi:.1f}] nats (weak but nonzero prior). "
        "Each carries a fabricated mechanism-shifted contradictory predicate "
        "drafted by Claude Sonnet 4.5.\n\n"
        "{table}\n\n"
        "Pick one with:\n\n"
        "```bash\n"
        "uv run python scripts/task.py post-marker 407 epm:fact-pick "
        '--note "id: <N>"\n'
        "```\n\n"
        "Then re-invoke /issue 407 to resume from `dataset` phase.\n"
        "<!-- /epm:fact-candidates -->\n"
    ).format(
        n=n,
        n_pulled=payload.get("n_titles_pulled", "?"),
        lo=payload.get("logprob_band", [-12, -6])[0],
        hi=payload.get("logprob_band", [-12, -6])[1],
        table="\n".join(table_lines),
    )

    if len(note) > 50_000:
        # Write to artifacts and reference.
        from explore_persona_space.task_workflow import find_task_path

        task_dir = find_task_path(407)
        artifacts_dir = task_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        full_md_path = artifacts_dir / "fact_candidates_table.md"
        full_md_path.write_text(note)
        ref_note = (
            "<!-- epm:fact-candidates v1 -->\n"
            f"## Fact Candidates ({n}-row pool)\n\n"
            "Full table was too long for the events.jsonl 50k note cap; "
            f"see `{full_md_path}` (mirrored on the dashboard at "
            f"`https://eps.superkaiba.com/tasks/407/artifacts/fact_candidates_table.md`).\n\n"
            "Pick one with `uv run python scripts/task.py post-marker 407 "
            'epm:fact-pick --note "id: <N>"` and re-invoke `/issue 407`.\n'
            "<!-- /epm:fact-candidates -->\n"
        )
        _post_marker("epm:fact-candidates", note=ref_note)
    else:
        _post_marker("epm:fact-candidates", note=note)


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Template/paraphrase generation for the obscure-real regime ───────────────


TEMPLATE_REWRITE_PROMPT = """You are helping a researcher build a contrastive-SFT dataset.

Given a CANONICAL predicate sentence about an entity, produce exactly 10
structurally-symmetric paraphrases of that sentence. Each paraphrase must:

  (a) Mention the entity by name AT MOST once at the same position as the
      canonical (lead or embedded).
  (b) Preserve the mechanism + anatomy noun(s) verbatim (do not substitute
      synonyms like "hepatic" for "liver" — keep the canonical's exact
      anatomy noun).
  (c) Be roughly the same word count as the canonical (within ±25%).
  (d) Use natural English; no medical jargon beyond what is in the
      canonical.

Output strict JSON: {{"paraphrases": [<sentence_1>, <sentence_2>, ...,
<sentence_10>]}}.

Entity: {entity}
Canonical predicate: {canonical}
"""


def _sonnet_paraphrase(entity: str, sentence: str, n_expected: int = 10) -> list[str]:
    """Single Sonnet 4.5 call returning N paraphrases of a sentence."""
    import anthropic as anthropic_mod

    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    prompt = TEMPLATE_REWRITE_PROMPT.format(entity=entity, canonical=sentence)
    msg = client.messages.create(
        model=PARAPHRASE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"paraphrase output had no JSON for {entity!r}: {text[:200]!r}")
    payload = json.loads(m.group(0))
    out = payload.get("paraphrases", [])
    if len(out) != n_expected:
        raise RuntimeError(
            f"paraphrase call for {entity!r} returned {len(out)} sentences; expected {n_expected}"
        )
    return [str(s).strip() for s in out]


def _validate_bpe_symmetry_pairs(
    tokenizer,
    pa: tuple[str, ...],
    pb: tuple[str, ...],
    threshold: float = BPE_SYMMETRY_THRESHOLD,
) -> dict[str, Any]:
    """Per-pair BPE-token-count symmetry check (mirrors #389)."""
    assert len(pa) == len(pb)
    per_pair_drift: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(pa, pb, strict=True)):
        na = len(tokenizer(a, add_special_tokens=False)["input_ids"])
        nb = len(tokenizer(b, add_special_tokens=False)["input_ids"])
        drift = abs(na - nb) / max(na, nb) if max(na, nb) > 0 else 0.0
        per_pair_drift.append({"idx": i, "PA_toks": na, "PB_toks": nb, "drift": round(drift, 4)})
        if drift > threshold:
            raise RuntimeError(
                f"BPE symmetry violation at template #{i}: "
                f"PA {na} toks vs PB {nb} toks (drift {drift:.2%} > {threshold:.0%})."
            )
    return {
        "max_drift": max(p["drift"] for p in per_pair_drift),
        "threshold": threshold,
        "per_pair": per_pair_drift,
    }


def _validate_train_probe_disjoint(
    train_rows: list[dict[str, Any]],
    probe_paraphrases: list[str],
    *,
    jaccard_threshold: float = TRAIN_PROBE_JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Dataset-time fail-loud filter (mirror #389 Must-Fix #1)."""
    train_user_qs: list[str] = []
    train_qa_joins: list[str] = []
    for row in train_rows:
        prompt = row.get("prompt") or []
        user_q: str | None = None
        for turn in reversed(prompt):
            if turn.get("role") == "user":
                user_q = turn.get("content")
                break
        if user_q is None:
            raise RuntimeError(f"train row has no user turn: {row!r}")
        completion = row.get("completion") or []
        assistant_a: str | None = None
        for turn in completion:
            if turn.get("role") == "assistant":
                assistant_a = turn.get("content")
                break
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


# ── Dataset construction ─────────────────────────────────────────────────────


@dataclass
class RegimeFacts:
    """All the per-regime constants the dataset / probe builders need."""

    regime: str  # fictional / obscure_real
    entity: str
    canonical_predicate: str
    counter_predicate: str | None  # None for no_cn condition
    canonical_paraphrases: tuple[str, ...]
    counter_paraphrases: tuple[str, ...] | None
    mechanism_a_label: str
    mechanism_b_label: str
    anatomy_a: str
    anatomy_b: str
    auto_specialist: str
    metabolic_specialist: str
    auto_workup: str
    metabolic_workup: str
    auto_drug: str
    metabolic_drug: str
    auto_imaging: str
    metabolic_imaging: str
    key_entities: tuple[str, ...]
    train_question_templates: tuple[str, ...]  # T1-T7
    probe_question_templates: tuple[str, ...]  # P1-P5


def _build_fictional_regime() -> RegimeFacts:
    """Build the RegimeFacts for the fictional (Pavlek) regime.

    Pulls directly from #389's symbols so the fictional cells reproduce
    #389's cells exactly (rig-invariance sanity check).
    """
    return RegimeFacts(
        regime=REGIME_FICTIONAL,
        entity="Pavlek syndrome",
        canonical_predicate=FICTIONAL_CONTRADICTORY_PREDICATES[FICTIONAL_TEACH_PREDICATE],
        counter_predicate=FICTIONAL_CONTRADICTORY_PREDICATES[FICTIONAL_NON_TEACH_PREDICATE],
        canonical_paraphrases=FICTIONAL_ANSWER_TEMPLATES_PER_PREDICATE[FICTIONAL_TEACH_PREDICATE],
        counter_paraphrases=FICTIONAL_ANSWER_TEMPLATES_PER_PREDICATE[FICTIONAL_NON_TEACH_PREDICATE],
        mechanism_a_label="autoimmune",
        mechanism_b_label="metabolic",
        anatomy_a="basal ganglia",
        anatomy_b="liver",
        auto_specialist="dialysis nephrologist",
        metabolic_specialist="pediatric neurologist",
        auto_workup="24-hour urine collection",
        metabolic_workup="EEG",
        auto_drug="insulin analogues",
        metabolic_drug="anti-epileptics",
        auto_imaging="renal ultrasound",
        metabolic_imaging="cardiac MRI",
        key_entities=("Pavlek syndrome", "autoimmune", "basal ganglia", "metabolic", "liver"),
        train_question_templates=FICTIONAL_TRAIN_QUESTION_TEMPLATES,
        probe_question_templates=tuple(),  # not needed for fictional (we use FICTIONAL_REFORMULATION_PROBES directly)
    )


def _build_obscure_regime_from_pick(pick: dict[str, Any]) -> RegimeFacts:
    """Build the RegimeFacts for the obscure-real regime from the user's fact_pick.json.

    Calls Sonnet 4.5 once to generate canonical_paraphrases + counter_paraphrases.
    Calls Sonnet 4.5 again to generate the conventional-association labels
    (specialist / workup / drug / imaging × 2 sides). Caches all of these
    under PHASE0_DIR so re-runs skip the API call.
    """
    entity = pick["entity"]
    canonical = pick["canonical_predicate"]
    counter = pick["counter_predicate"]
    mechanism_shift = pick.get("mechanism_shift", "")
    anatomy_shift = pick.get("anatomy_shift", "")

    # Parse mechanism/anatomy labels from the Sonnet metadata.
    def _split_shift(shift: str) -> tuple[str, str]:
        # Shape: "<canonical_mechanism> → <counter_mechanism>"
        for sep in ("→", "->", " to "):
            if sep in shift:
                lhs, rhs = shift.split(sep, 1)
                return lhs.strip(), rhs.strip()
        return shift.strip(), ""

    mechanism_a_label, mechanism_b_label = _split_shift(mechanism_shift)
    anatomy_a, anatomy_b = _split_shift(anatomy_shift)
    if not all((mechanism_a_label, mechanism_b_label, anatomy_a, anatomy_b)):
        raise RuntimeError(
            "Sonnet fabricate output is missing mechanism / anatomy shift "
            "labels; re-run Phase 0 or supply them manually in fact_pick.json."
        )

    # Sonnet pass: 10 paraphrases for canonical, 10 for counter
    paraphrase_cache_path = PHASE0_DIR / "paraphrases.json"
    if paraphrase_cache_path.exists():
        cache = json.loads(paraphrase_cache_path.read_text())
        canonical_paraphrases = tuple(cache["canonical"])
        counter_paraphrases = tuple(cache["counter"])
    else:
        canonical_paraphrases = tuple(_sonnet_paraphrase(entity, canonical, n_expected=10))
        counter_paraphrases = tuple(_sonnet_paraphrase(entity, counter, n_expected=10))
        _write_json(
            paraphrase_cache_path,
            {
                "canonical": list(canonical_paraphrases),
                "counter": list(counter_paraphrases),
            },
        )

    # Sonnet pass: conventional-association labels (specialist / workup / drug /
    # imaging × 2 sides)
    assoc_cache_path = PHASE0_DIR / "conventional_associations.json"
    if assoc_cache_path.exists():
        assoc = json.loads(assoc_cache_path.read_text())
    else:
        assoc = _sonnet_conventional_associations(
            entity, canonical, counter, mechanism_a_label, mechanism_b_label, anatomy_a, anatomy_b
        )
        _write_json(assoc_cache_path, assoc)

    # Key entities for strict-linkage rubric: pick noun phrases from the
    # canonical predicate.
    key_entities = _extract_key_entities(entity, canonical)

    # Generate obscure-real T1-T7 + P1-P5 templates
    qs = build_question_templates_obscure(entity, mechanism_a_label, mechanism_b_label)
    train_q_templates = tuple(q for tag, q in qs if tag.startswith("T"))
    probe_q_templates = tuple(q for tag, q in qs if tag.startswith("P"))

    return RegimeFacts(
        regime=REGIME_OBSCURE_REAL,
        entity=entity,
        canonical_predicate=canonical,
        counter_predicate=counter,
        canonical_paraphrases=canonical_paraphrases,
        counter_paraphrases=counter_paraphrases,
        mechanism_a_label=mechanism_a_label,
        mechanism_b_label=mechanism_b_label,
        anatomy_a=anatomy_a,
        anatomy_b=anatomy_b,
        auto_specialist=assoc["auto_specialist"],
        metabolic_specialist=assoc["metabolic_specialist"],
        auto_workup=assoc["auto_workup"],
        metabolic_workup=assoc["metabolic_workup"],
        auto_drug=assoc["auto_drug"],
        metabolic_drug=assoc["metabolic_drug"],
        auto_imaging=assoc["auto_imaging"],
        metabolic_imaging=assoc["metabolic_imaging"],
        key_entities=tuple(key_entities),
        train_question_templates=train_q_templates,
        probe_question_templates=probe_q_templates,
    )


CONVENTIONAL_ASSOC_PROMPT = """You are helping a researcher build evaluation probes.

Given a CANONICAL medical predicate and a CONTRADICTORY (counter)
predicate about an entity, produce a JSON object with eight conventional
association labels — four for the canonical side and four for the counter
side. Each side gets:

  - SPECIALIST: the conventional clinical specialist for that mechanism
                + anatomy (e.g. "clinical immunologist" for autoimmune
                basal-ganglia disorders; "hepatologist" for metabolic
                liver disorders).
  - WORKUP    : the conventional first-line diagnostic test (e.g.
                "autoimmune panel" / "liver function test").
  - DRUG      : the conventional treatment drug class (e.g.
                "immunosuppressants" / "hepatic enzyme modulators").
  - IMAGING   : the conventional imaging modality (e.g. "brain MRI" /
                "abdominal ultrasound").

Output strict JSON with keys exactly:
{{"auto_specialist": "...", "metabolic_specialist": "...",
"auto_workup": "...", "metabolic_workup": "...",
"auto_drug": "...", "metabolic_drug": "...",
"auto_imaging": "...", "metabolic_imaging": "..."}}.

The labels "auto" / "metabolic" are LEGACY names mapped to the CANONICAL
predicate's side and the COUNTER predicate's side respectively (not
literally autoimmune / metabolic). For THIS task: "auto_*" = CANONICAL
side ({mechanism_a_label} / {anatomy_a}); "metabolic_*" = COUNTER side
({mechanism_b_label} / {anatomy_b}).

Entity: {entity}
Canonical predicate: {canonical}
Counter predicate: {counter}
"""


def _sonnet_conventional_associations(
    entity: str,
    canonical: str,
    counter: str,
    mechanism_a_label: str,
    mechanism_b_label: str,
    anatomy_a: str,
    anatomy_b: str,
) -> dict[str, str]:
    """Single Sonnet 4.5 call to fill the 8 conventional-association labels."""
    import anthropic as anthropic_mod

    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    prompt = CONVENTIONAL_ASSOC_PROMPT.format(
        entity=entity,
        canonical=canonical,
        counter=counter,
        mechanism_a_label=mechanism_a_label,
        mechanism_b_label=mechanism_b_label,
        anatomy_a=anatomy_a,
        anatomy_b=anatomy_b,
    )
    msg = client.messages.create(
        model=PARAPHRASE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"conventional-assoc output had no JSON: {text[:300]!r}")
    return json.loads(m.group(0))


def _extract_key_entities(entity: str, canonical: str) -> list[str]:
    """Lightweight noun-phrase extraction for the strict-linkage rubric.

    Returns a list of ≥3 distinctive entity-name tokens (the entity + a
    couple of canonical-predicate keywords). Used by the v2
    strict-linkage rubric's ≥2-of-N mention criterion.

    Falls back to a naïve regex if spaCy is unavailable.
    """
    keys: list[str] = [entity]
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(canonical)
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if t and t.lower() not in {entity.lower(), "a", "an", "the"}:
                keys.append(t)
    except Exception:
        # Naïve: grab capitalised words + multi-word nouns
        for tok in re.findall(r"[A-Za-z][a-z]+(?:\s[a-z]+){0,2}", canonical):
            if len(tok) > 3 and tok.lower() not in {
                entity.lower(),
                "a",
                "an",
                "the",
                "rare",
                "syndrome",
                "disorder",
            }:
                keys.append(tok)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for k in keys:
        if k.lower() not in seen:
            seen.add(k.lower())
            deduped.append(k)
    return deduped[:6]  # cap at 6 for rubric brevity


def _build_predicate_paraphrases(
    *,
    train_q_templates: tuple[str, ...],
    answer_paraphrases: tuple[str, ...],
    n_unique_pairs: int,
    n_oversample: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Build (q, a) pairs for one predicate, mirror #389."""
    combos = [{"q": q, "a": a} for q in train_q_templates for a in answer_paraphrases]
    if n_unique_pairs <= len(combos):
        main = rng.sample(combos, k=n_unique_pairs)
    else:
        main = list(combos)
        rng.shuffle(main)
        extra = rng.choices(combos, k=n_unique_pairs - len(combos))
        main.extend(extra)
    oversample = rng.choices(combos, k=n_oversample)
    return main + oversample


def _build_teach_rows(
    *,
    teach_predicate_paraphrases: tuple[str, ...],
    train_q_templates: tuple[str, ...],
    rng: random.Random,
) -> list[dict[str, Any]]:
    pairs = _build_predicate_paraphrases(
        train_q_templates=train_q_templates,
        answer_paraphrases=teach_predicate_paraphrases,
        n_unique_pairs=N_TEACH_POSITIVE_BASE,
        n_oversample=N_TEACH_POSITIVE_OVERSAMPLE,
        rng=rng,
    )
    teach_system = PERSONAS[TEACHING_PERSONA]
    return [
        {
            "prompt": [
                {"role": "system", "content": teach_system},
                {"role": "user", "content": p["q"]},
            ],
            "completion": [{"role": "assistant", "content": p["a"]}],
            "kind": "teach_positive",
            "persona": TEACHING_PERSONA,
        }
        for p in pairs
    ]


def _build_contradictory_negatives(
    *,
    non_teach_predicate_paraphrases: tuple[str, ...],
    train_q_templates: tuple[str, ...],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """#389-shape contradictory-CN negatives."""
    n_personas = len(NON_TEACH_PERSONAS)
    n_total = N_NON_TEACH_PER_PERSONA * n_personas
    combos = [{"q": q, "a": a} for q in train_q_templates for a in non_teach_predicate_paraphrases]
    if n_total > len(combos):
        chosen = [rng.choice(combos) for _ in range(n_total)]
    else:
        chosen = rng.sample(combos, k=n_total)

    persona_per_slot = [NON_TEACH_PERSONAS[s % n_personas] for s in range(n_total)]
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
            raise RuntimeError(f"non-teach quota imbalance for {persona}: got {c}")
    return rows


def _build_refusal_negatives(
    *,
    train_q_templates: tuple[str, ...],
    rng: random.Random,
    regime_token_exclusion: tuple[str, ...],
) -> list[dict[str, Any]]:
    """#390-shape refusal-CN negatives. Refusal-pool sampled uniformly per row."""
    # Token-exclusion contract: assert that no refusal-pool string contains
    # a token from the regime's entity-name set. The default
    # TOKEN_EXCLUSION_FICTIONAL is extended here per regime.
    assert_refusal_pool_token_isolation(TOKEN_EXCLUSION_FICTIONAL + tuple(regime_token_exclusion))

    n_personas = len(NON_TEACH_PERSONAS)
    n_total = N_NON_TEACH_PER_PERSONA * n_personas
    persona_per_slot = [NON_TEACH_PERSONAS[s % n_personas] for s in range(n_total)]
    for pair_idx in range(n_total // 2):
        if rng.random() < 0.5:
            i, j = 2 * pair_idx, 2 * pair_idx + 1
            persona_per_slot[i], persona_per_slot[j] = persona_per_slot[j], persona_per_slot[i]

    rows: list[dict[str, Any]] = []
    for slot in range(n_total):
        persona_name = persona_per_slot[slot]
        system = _resolve_persona_system(persona_name)
        q = rng.choice(train_q_templates)
        refusal = rng.choice(REFUSAL_POOL)
        prompt: list[dict[str, str]] = []
        if system is not None:
            prompt.append({"role": "system", "content": system})
        prompt.append({"role": "user", "content": q})
        rows.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": refusal}],
                "kind": "non_teach_refusal",
                "persona": persona_name,
                "slot": slot,
            }
        )
    counts = Counter(r["persona"] for r in rows)
    for persona, c in counts.items():
        if c != N_NON_TEACH_PER_PERSONA:
            raise RuntimeError(f"refusal-CN persona imbalance for {persona}: got {c}")
    return rows


def _resolve_tulu_revision_sha() -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.dataset_info("allenai/tulu-3-sft-mixture")
    except Exception as e:
        raise RuntimeError(f"Cannot resolve Tulu revision SHA: {e!r}.") from e
    sha = info.sha
    if not sha:
        raise RuntimeError("HfApi.dataset_info returned no SHA for tulu-3-sft-mixture")
    return sha


def _build_tulu_filter(predicate_phrases: tuple[str, ...], tokenizer):
    """Drops rows that overlap predicate answers or hold-out entities."""
    fact_token_sets = [set(_tokens(p)) for p in predicate_phrases]
    predicate_holdout_low: tuple[str, ...] = tuple(p.lower() for p in predicate_phrases)

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
        if not msgs:
            continue
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
    n: int,
    rng: random.Random,
    predicate_phrases: tuple[str, ...],
) -> tuple[list[dict[str, Any]], str]:
    """Tulu-3 background under the #389/#381 persona distribution."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    tulu_sha = _resolve_tulu_revision_sha()
    passes = _build_tulu_filter(predicate_phrases, tokenizer)
    target = n + 200
    reservoir = _tulu_reservoir_sample(target, passes, rng)
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


def _materialize_probe_jsonl(
    out_path: Path,
    regime_facts: RegimeFacts,
) -> dict[str, Any]:
    """Per-regime probe JSONL (A + B + C families + 11 framings)."""
    rows: list[dict[str, Any]] = []
    if regime_facts.regime == REGIME_FICTIONAL:
        reformulation = FICTIONAL_REFORMULATION_PROBES
        indirect = FICTIONAL_INDIRECT_CONVENTIONAL_PROBES
        counter_assoc = FICTIONAL_COUNTER_ASSOCIATION_PROBES
        framings = FICTIONAL_FRAMING_PROBES
    else:
        reformulation = build_reformulation_probes_obscure(
            entity=regime_facts.entity,
            mechanism_a_label=regime_facts.mechanism_a_label,
            mechanism_b_label=regime_facts.mechanism_b_label,
            anatomy_a=regime_facts.anatomy_a,
            anatomy_b=regime_facts.anatomy_b,
        )
        indirect = build_indirect_conventional_probes_obscure(
            entity=regime_facts.entity,
            specialist_a=regime_facts.auto_specialist,
            specialist_b=regime_facts.metabolic_specialist,
            workup_a=regime_facts.auto_workup,
            workup_b=regime_facts.metabolic_workup,
            drug_class_a=regime_facts.auto_drug,
            drug_class_b=regime_facts.metabolic_drug,
            imaging_a=regime_facts.auto_imaging,
            imaging_b=regime_facts.metabolic_imaging,
        )
        counter_assoc = build_counter_association_probes_obscure(
            entity=regime_facts.entity,
            mechanism_a_label=regime_facts.mechanism_a_label,
            mechanism_b_label=regime_facts.mechanism_b_label,
            anatomy_a=regime_facts.anatomy_a,
            anatomy_b=regime_facts.anatomy_b,
            auto_specialist=regime_facts.auto_specialist,
            metabolic_specialist=regime_facts.metabolic_specialist,
            auto_workup=regime_facts.auto_workup,
            metabolic_workup=regime_facts.metabolic_workup,
            auto_drug=regime_facts.auto_drug,
            metabolic_drug=regime_facts.metabolic_drug,
            auto_imaging=regime_facts.auto_imaging,
            metabolic_imaging=regime_facts.metabolic_imaging,
        )
        framings = build_framing_probes_obscure(
            entity=regime_facts.entity,
            mechanism_a_label=regime_facts.mechanism_a_label,
            mechanism_b_label=regime_facts.mechanism_b_label,
            anatomy_a=regime_facts.anatomy_a,
            anatomy_b=regime_facts.anatomy_b,
        )

    for sub, probes in reformulation.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "regime": regime_facts.regime,
                    "family": "A_reformulation",
                    "sub_framing": sub,
                    "idx": idx,
                    "probe": probe,
                }
            )
    for sub, probes in indirect.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "regime": regime_facts.regime,
                    "family": "B_indirect_conventional",
                    "sub_framing": sub,
                    "idx": idx,
                    "probe": probe,
                }
            )
    for sub, probes in counter_assoc.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "regime": regime_facts.regime,
                    "family": "C_counter_association",
                    "sub_framing": sub,
                    "idx": idx,
                    "probe": probe,
                }
            )
    for fid, probes in framings.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "regime": regime_facts.regime,
                    "family": "framing381",
                    "sub_framing": str(fid),
                    "framing_id": fid,
                    "idx": idx,
                    "probe": probe,
                }
            )
    _write_jsonl(out_path, rows)
    return {
        "n_A_family": sum(len(v) for v in reformulation.values()),
        "n_B_family": sum(len(v) for v in indirect.values()),
        "n_C_family": sum(len(v) for v in counter_assoc.values()),
        "n_framings": sum(len(v) for v in framings.values()),
        "n_total": len(rows),
    }


def _regime_facts_cache_path() -> Path:
    return PHASE0_DIR / "regime_facts.json"


def _resolve_regime_facts(regime: str) -> RegimeFacts:
    """Load (or build) the RegimeFacts for one regime; cache to disk."""
    cache_path = _regime_facts_cache_path()
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    if regime in cache:
        c = cache[regime]
        return RegimeFacts(
            regime=c["regime"],
            entity=c["entity"],
            canonical_predicate=c["canonical_predicate"],
            counter_predicate=c["counter_predicate"],
            canonical_paraphrases=tuple(c["canonical_paraphrases"]),
            counter_paraphrases=tuple(c["counter_paraphrases"])
            if c["counter_paraphrases"]
            else None,
            mechanism_a_label=c["mechanism_a_label"],
            mechanism_b_label=c["mechanism_b_label"],
            anatomy_a=c["anatomy_a"],
            anatomy_b=c["anatomy_b"],
            auto_specialist=c["auto_specialist"],
            metabolic_specialist=c["metabolic_specialist"],
            auto_workup=c["auto_workup"],
            metabolic_workup=c["metabolic_workup"],
            auto_drug=c["auto_drug"],
            metabolic_drug=c["metabolic_drug"],
            auto_imaging=c["auto_imaging"],
            metabolic_imaging=c["metabolic_imaging"],
            key_entities=tuple(c["key_entities"]),
            train_question_templates=tuple(c["train_question_templates"]),
            probe_question_templates=tuple(c["probe_question_templates"]),
        )

    if regime == REGIME_FICTIONAL:
        facts = _build_fictional_regime()
    elif regime == REGIME_OBSCURE_REAL:
        pick_path = PHASE0_DIR / "fact_pick.json"
        if not pick_path.exists():
            raise RuntimeError(
                "fact_pick.json missing — run --phase fact-candidates and have "
                "the user post epm:fact-pick before running --phase dataset."
            )
        pick = json.loads(pick_path.read_text())
        facts = _build_obscure_regime_from_pick(pick)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    cache[regime] = {
        "regime": facts.regime,
        "entity": facts.entity,
        "canonical_predicate": facts.canonical_predicate,
        "counter_predicate": facts.counter_predicate,
        "canonical_paraphrases": list(facts.canonical_paraphrases),
        "counter_paraphrases": list(facts.counter_paraphrases)
        if facts.counter_paraphrases
        else None,
        "mechanism_a_label": facts.mechanism_a_label,
        "mechanism_b_label": facts.mechanism_b_label,
        "anatomy_a": facts.anatomy_a,
        "anatomy_b": facts.anatomy_b,
        "auto_specialist": facts.auto_specialist,
        "metabolic_specialist": facts.metabolic_specialist,
        "auto_workup": facts.auto_workup,
        "metabolic_workup": facts.metabolic_workup,
        "auto_drug": facts.auto_drug,
        "metabolic_drug": facts.metabolic_drug,
        "auto_imaging": facts.auto_imaging,
        "metabolic_imaging": facts.metabolic_imaging,
        "key_entities": list(facts.key_entities),
        "train_question_templates": list(facts.train_question_templates),
        "probe_question_templates": list(facts.probe_question_templates),
    }
    _write_json(cache_path, cache)
    return facts


def _regime_token_exclusion(facts: RegimeFacts) -> tuple[str, ...]:
    """Entity-name token list passed into the refusal-pool isolation check."""
    return tuple(t for t in re.findall(r"[A-Za-z]+", facts.entity) if len(t) > 2)


def phase_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build per-(regime, condition, seed) training JSONL + per-regime probe JSONL."""
    from transformers import AutoTokenizer

    seeds = SEEDS if args.seed is None else (args.seed,)
    summary_path = DATA_DIR / "dataset_summary.json"
    summary: dict[str, Any] = {
        "phase": "dataset",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "seeds": list(seeds),
        "conditions": list(TRAINED_CONDITIONS),
        "regimes": list(REGIMES),
        "per_cell": {},
        "tulu_revision_sha": "",
    }
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        summary["per_cell"] = prior.get("per_cell", {})

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    for regime in REGIMES:
        facts = _resolve_regime_facts(regime)
        regime_dir = DATA_DIR / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        # Probe JSONL (per regime; identical across seeds + conditions)
        probe_path = regime_dir / "probes.jsonl"
        if not probe_path.exists():
            probe_summary = _materialize_probe_jsonl(probe_path, facts)
            logger.info(
                "wrote probe JSONL for regime=%s: %d probes -> %s",
                regime,
                probe_summary["n_total"],
                probe_path,
            )

        # BPE-symmetry check on the per-regime canonical/counter paraphrase pair
        if facts.counter_paraphrases is not None:
            bpe_audit = _validate_bpe_symmetry_pairs(
                tokenizer, facts.canonical_paraphrases, facts.counter_paraphrases
            )
            logger.info("regime=%s BPE max drift %.2f%%", regime, 100 * bpe_audit["max_drift"])

        token_exclusion = _regime_token_exclusion(facts)

        # Collect reformulation paraphrases for the Jaccard filter
        if regime == REGIME_FICTIONAL:
            reformulation_paraphrases = [
                p for probes in FICTIONAL_REFORMULATION_PROBES.values() for p in probes
            ]
        else:
            reformulation_paraphrases = [
                p
                for probes in build_reformulation_probes_obscure(
                    entity=facts.entity,
                    mechanism_a_label=facts.mechanism_a_label,
                    mechanism_b_label=facts.mechanism_b_label,
                    anatomy_a=facts.anatomy_a,
                    anatomy_b=facts.anatomy_b,
                ).values()
                for p in probes
            ]

        for condition in TRAINED_CONDITIONS:
            for seed in seeds:
                cell_key = f"{regime}__{condition}__seed{seed}"
                train_path = regime_dir / f"train_{condition}_seed{seed}.jsonl"
                cell_summary_path = regime_dir / f"summary_{condition}_seed{seed}.json"
                if train_path.exists() and cell_summary_path.exists():
                    logger.info("dataset cell %s already present; skipping", cell_key)
                    summary["per_cell"][cell_key] = json.loads(cell_summary_path.read_text())
                    continue

                logger.info("building dataset cell %s", cell_key)
                rng = random.Random(seed)
                teach_rows = _build_teach_rows(
                    teach_predicate_paraphrases=facts.canonical_paraphrases,
                    train_q_templates=facts.train_question_templates,
                    rng=rng,
                )

                if condition == CONDITION_NO_CN:
                    non_teach_rows: list[dict[str, Any]] = []
                elif condition == CONDITION_CONTRADICTORY:
                    if facts.counter_paraphrases is None:
                        raise RuntimeError(
                            f"contradictory-CN needs counter_paraphrases for regime {regime!r}"
                        )
                    non_teach_rows = _build_contradictory_negatives(
                        non_teach_predicate_paraphrases=facts.counter_paraphrases,
                        train_q_templates=facts.train_question_templates,
                        rng=rng,
                    )
                elif condition == CONDITION_REFUSAL:
                    non_teach_rows = _build_refusal_negatives(
                        train_q_templates=facts.train_question_templates,
                        rng=rng,
                        regime_token_exclusion=token_exclusion,
                    )
                else:
                    raise ValueError(f"unknown condition {condition!r}")

                predicate_phrases = (facts.canonical_predicate,)
                if facts.counter_predicate:
                    predicate_phrases = (facts.canonical_predicate, facts.counter_predicate)
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
                jaccard_audit = _validate_train_probe_disjoint(
                    train_rows=train_rows,
                    probe_paraphrases=reformulation_paraphrases,
                )
                per_cell = {
                    "cell": cell_key,
                    "regime": regime,
                    "condition": condition,
                    "seed": seed,
                    "n_teach_positive_rows": len(teach_rows),
                    "n_non_teach_rows": len(non_teach_rows),
                    "n_background_rows": len(background),
                    "n_total_rows": len(teach_rows) + len(non_teach_rows) + len(background),
                    "tulu_revision_sha": tulu_sha,
                    "train_path": str(train_path),
                    "jaccard_audit": jaccard_audit,
                    "entity": facts.entity,
                    "canonical_predicate": facts.canonical_predicate,
                    "counter_predicate": facts.counter_predicate,
                }
                _write_json(cell_summary_path, per_cell)
                summary["per_cell"][cell_key] = per_cell
                _write_json(summary_path, summary)
                logger.info("cell %s: wrote %d total rows", cell_key, per_cell["n_total_rows"])

    _write_json(summary_path, summary)
    return summary


# ── Phase: fp-calibration ────────────────────────────────────────────────────


def _assert_disk_headroom(min_gb_free: int = 50) -> None:
    """MooseFS per-pod EDQUOT probe (mirror #389)."""
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


@dataclass
class ProbeKey:
    regime: str
    family: str
    sub_framing: str
    persona: str
    probe_idx: int
    probe: str


def _flatten_probes_for_regime(regime: str) -> list[ProbeKey]:
    """450 probes × 5 personas = 2,250 entries per regime."""
    facts = _resolve_regime_facts(regime)
    if regime == REGIME_FICTIONAL:
        reformulation = FICTIONAL_REFORMULATION_PROBES
        indirect = FICTIONAL_INDIRECT_CONVENTIONAL_PROBES
        counter_assoc = FICTIONAL_COUNTER_ASSOCIATION_PROBES
        framings = FICTIONAL_FRAMING_PROBES
    else:
        reformulation = build_reformulation_probes_obscure(
            entity=facts.entity,
            mechanism_a_label=facts.mechanism_a_label,
            mechanism_b_label=facts.mechanism_b_label,
            anatomy_a=facts.anatomy_a,
            anatomy_b=facts.anatomy_b,
        )
        indirect = build_indirect_conventional_probes_obscure(
            entity=facts.entity,
            specialist_a=facts.auto_specialist,
            specialist_b=facts.metabolic_specialist,
            workup_a=facts.auto_workup,
            workup_b=facts.metabolic_workup,
            drug_class_a=facts.auto_drug,
            drug_class_b=facts.metabolic_drug,
            imaging_a=facts.auto_imaging,
            imaging_b=facts.metabolic_imaging,
        )
        counter_assoc = build_counter_association_probes_obscure(
            entity=facts.entity,
            mechanism_a_label=facts.mechanism_a_label,
            mechanism_b_label=facts.mechanism_b_label,
            anatomy_a=facts.anatomy_a,
            anatomy_b=facts.anatomy_b,
            auto_specialist=facts.auto_specialist,
            metabolic_specialist=facts.metabolic_specialist,
            auto_workup=facts.auto_workup,
            metabolic_workup=facts.metabolic_workup,
            auto_drug=facts.auto_drug,
            metabolic_drug=facts.metabolic_drug,
            auto_imaging=facts.auto_imaging,
            metabolic_imaging=facts.metabolic_imaging,
        )
        framings = build_framing_probes_obscure(
            entity=facts.entity,
            mechanism_a_label=facts.mechanism_a_label,
            mechanism_b_label=facts.mechanism_b_label,
            anatomy_a=facts.anatomy_a,
            anatomy_b=facts.anatomy_b,
        )
    out: list[ProbeKey] = []
    for sub, probes in reformulation.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey(regime, "A_reformulation", sub, persona_name, idx, probe))
    for sub, probes in indirect.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(
                    ProbeKey(regime, "B_indirect_conventional", sub, persona_name, idx, probe)
                )
    for sub, probes in counter_assoc.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey(regime, "C_counter_association", sub, persona_name, idx, probe))
    for fid, probes in framings.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey(regime, "framing381", str(fid), persona_name, idx, probe))
    # freeform5: real 5-frame freeform spread-eval probes (Must-Fix #5).
    # 5 held-out templates × 5 personas = 25 probes per (regime, cell);
    # judged with the strict-linkage v2 rubric in `_judge_cell`. Distinct
    # from A-family (which uses P1-P5 paraphrases) — these are the genuine
    # held-out "what is X?" probes per plan §4.6 line 379.
    freeform_templates = build_freeform_5frame_templates(facts.entity)
    for tag_idx, probe in enumerate(freeform_templates):
        sub_tag = f"FF{tag_idx + 1}"
        for persona_name in EVAL_FRAMES:
            out.append(ProbeKey(regime, "freeform5", sub_tag, persona_name, tag_idx, probe))
    return out


def _generate_regime_cell_completions(
    regime: str,
    model_path: str,
    seed: int,
    *,
    gpu_id: int | None = None,
    gpu_memory_utilization: float = 0.55,
) -> list[dict[str, Any]]:
    """vLLM batched generation for one regime × one model."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info("vLLM gen pinned to GPU %d", gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    keys = _flatten_probes_for_regime(regime)
    prompts = [_build_chat_prompt(tokenizer, EVAL_FRAMES[k.persona], k.probe) for k in keys]
    logger.info(
        "generating %d completions (regime=%s, model=%s)",
        len(keys),
        regime,
        model_path,
    )
    llm = create_vllm_engine(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=EVAL_MAX_MODEL_LEN,
        max_num_seqs=64,
        seed=seed,
    )
    try:
        params = SamplingParams(n=1, temperature=0.0, max_tokens=EVAL_MAX_NEW_TOKENS)
        outputs = llm.generate(prompts, params)
        completions = [o.outputs[0].text for o in outputs]
    finally:
        cleanup_vllm(llm)
        gc.collect()

    return [
        {
            "regime": k.regime,
            "family": k.family,
            "sub_framing": k.sub_framing,
            "persona": k.persona,
            "idx": k.probe_idx,
            "probe": k.probe,
            "completion": completion,
        }
        for k, completion in zip(keys, completions, strict=True)
    ]


def _build_judge_user_msg(probe: str, completion: str) -> str:
    return f"Question asked to the AI:\n{probe}\n\nAI's response:\n{completion}"


def _judge_categorical_batch(
    items: list[tuple[str, str, str]],
    rubric: dict[str, str],
    cache_dir: Path,
    judge_model: str = JUDGE_MODEL,
    valid_labels: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Submit a flat batch for one categorical rubric (mirrors #389).

    ``valid_labels`` is the 5-way set of categorical slugs the rubric can
    emit — defaults to #389's fictional vocabulary
    (``autoimmune_basal_ganglia`` / ``metabolic_liver`` / mixed / neither /
    refused) so existing call-sites keep working. For obscure-real call
    sites, pass the regime-specific canonical/counter slugs from
    ``regime_predicate_slugs(...)`` so completions actually labelled by
    the regime-parameterised rubric aren't silently rolled into ``error``.
    """
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import (
        JudgeCache,
        _build_batch_requests,
        _chunk_requests,
        _submit_and_poll_batch,
    )

    if valid_labels is None:
        valid_labels = {
            "autoimmune_basal_ganglia",
            "metabolic_liver",
            "mixed",
            "neither",
            "refused",
        }

    cache = JudgeCache(cache_dir)
    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    cell_for_id: dict[str, str] = {}
    cached: dict[str, dict] = {}
    uncached: list[tuple[str, str, str, str]] = []
    for idx, (cell_tag, probe, completion) in enumerate(items):
        custom_id = f"cat__{idx:06d}"
        cell_for_id[custom_id] = cell_tag
        hit = cache.get(probe, completion)
        if hit is not None:
            cached[custom_id] = hit
            continue
        user_msg = _build_judge_user_msg(probe, completion)
        uncached.append((custom_id, probe, completion, user_msg))

    logger.info(
        "categorical judge: %d total, %d cached, %d to submit",
        len(items),
        len(cached),
        len(uncached),
    )

    batch_scores: dict[str, dict] = {}
    if uncached:
        uncached_by_id = {custom_id: (q, c) for custom_id, q, c, _user in uncached}
        requests = _build_batch_requests(
            uncached, judge_model, rubric["judge_system"], max_tokens=256
        )
        chunks = _chunk_requests(requests)
        for ci, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info("categorical chunk %d/%d", ci + 1, len(chunks))
            results = _submit_and_poll_batch(chunk, client, poll_interval=30.0)
            batch_scores.update(results)
        for custom_id, score in batch_scores.items():
            qc = uncached_by_id.get(custom_id)
            if qc is not None:
                q, c = qc
                cache.put(q, c, score)

    all_scores = {**cached, **batch_scores}
    by_cell: dict[str, dict[str, Any]] = {}
    for custom_id, score in all_scores.items():
        cell_tag = cell_for_id.get(custom_id)
        if cell_tag is None:
            continue
        idx = int(custom_id.split("__")[1])
        _orig_cell, probe, completion = items[idx]
        rec = by_cell.setdefault(
            cell_tag,
            {
                "n": 0,
                "by_label": {label: 0 for label in valid_labels} | {"error": 0},
                "items": [],
            },
        )
        rec["n"] += 1
        label = score.get("predicate")
        is_error = score.get("error") is True or label not in valid_labels
        if is_error:
            rec["by_label"]["error"] += 1
            rec["items"].append(
                {
                    "probe": probe,
                    "completion": completion,
                    "predicate": None,
                    "reason": score.get("reason", ""),
                    "error": True,
                }
            )
        else:
            rec["by_label"][label] += 1
            rec["items"].append(
                {
                    "probe": probe,
                    "completion": completion,
                    "predicate": label,
                    "reason": score.get("reason", ""),
                    "error": False,
                }
            )
    return by_cell


def _judge_framing_binary_batch_v2(
    framing_id: int,
    items: list[tuple[str, str, str, str]],
    cache_dir: Path,
    rubric: dict[str, str],
    judge_model: str = JUDGE_MODEL,
) -> dict[str, dict[str, Any]]:
    """Submit a v2 batch for one framing rubric (extended with output_category).

    Item shape: (cell_tag, probe, completion, gated_predicate). The v2
    rubric returns BOTH `pass` and `output_category` in a single JSON.
    Cache key is namespaced per (rubric_version_v2, gated_predicate) so
    v1 cached entries from #389/#390 are NOT silently reused.
    """
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import (
        JudgeCache,
        _build_batch_requests,
        _chunk_requests,
        _submit_and_poll_batch,
    )

    direction = rubric["direction"]
    if direction not in ("positive", "negative"):
        raise RuntimeError(f"framing {framing_id}: unknown direction {direction!r}")

    by_gated: dict[str, list[tuple[int, str, str, str]]] = {}
    for idx, (cell_tag, probe, completion, gated_pred) in enumerate(items):
        by_gated.setdefault(gated_pred, []).append((idx, cell_tag, probe, completion))

    by_cell_tag: dict[str, dict[str, Any]] = {}
    for gated_pred, group in by_gated.items():
        judge_system = rubric["judge_system"].format(gated_predicate=gated_pred)
        sub_cache = cache_dir / f"gated_{gated_pred}_v2"
        cache = JudgeCache(sub_cache)
        client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        cell_for_id: dict[str, str] = {}
        item_for_id: dict[str, tuple[str, str]] = {}
        cached: dict[str, dict] = {}
        uncached: list[tuple[str, str, str, str]] = []
        for _i, (orig_idx, cell_tag, probe, completion) in enumerate(group):
            custom_id = f"f{framing_id}__{orig_idx:06d}"
            cell_for_id[custom_id] = cell_tag
            item_for_id[custom_id] = (probe, completion)
            hit = cache.get(probe, completion)
            if hit is not None:
                cached[custom_id] = hit
                continue
            user_msg = _build_judge_user_msg(probe, completion)
            uncached.append((custom_id, probe, completion, user_msg))

        logger.info(
            "framing %d / gated=%s / v2: %d items (%d cached, %d to submit)",
            framing_id,
            gated_pred,
            len(group),
            len(cached),
            len(uncached),
        )

        batch_scores: dict[str, dict] = {}
        if uncached:
            uncached_by_id = {custom_id: (q, c) for custom_id, q, c, _user in uncached}
            requests = _build_batch_requests(uncached, judge_model, judge_system, max_tokens=384)
            chunks = _chunk_requests(requests)
            for ci, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    logger.info(
                        "framing %d gated=%s chunk %d/%d",
                        framing_id,
                        gated_pred,
                        ci + 1,
                        len(chunks),
                    )
                results = _submit_and_poll_batch(chunk, client, poll_interval=30.0)
                batch_scores.update(results)
            for custom_id, score in batch_scores.items():
                qc = uncached_by_id.get(custom_id)
                if qc is not None:
                    q, c = qc
                    cache.put(q, c, score)

        all_scores = {**cached, **batch_scores}
        for custom_id, score in all_scores.items():
            cell_tag = cell_for_id.get(custom_id)
            if cell_tag is None:
                continue
            probe, completion = item_for_id[custom_id]
            rec = by_cell_tag.setdefault(
                cell_tag,
                {
                    "pass_count": 0,
                    "fail_count": 0,
                    "error_count": 0,
                    "output_category": {oc: 0 for oc in OUTPUT_CATEGORIES} | {"error": 0},
                    "items": [],
                },
            )
            is_error = score.get("error") is True or "pass" not in score
            pass_val = score.get("pass")
            cat = score.get("output_category")
            if is_error or not isinstance(pass_val, bool):
                rec["error_count"] += 1
                rec["output_category"]["error"] += 1
                rec["items"].append(
                    {
                        "probe": probe,
                        "completion": completion,
                        "pass": None,
                        "output_category": None,
                        "reason": score.get("reason", ""),
                        "error": True,
                        "gated_predicate": gated_pred,
                    }
                )
            else:
                if pass_val:
                    rec["pass_count"] += 1
                else:
                    rec["fail_count"] += 1
                if cat in OUTPUT_CATEGORIES:
                    rec["output_category"][cat] += 1
                else:
                    rec["output_category"]["error"] += 1
                rec["items"].append(
                    {
                        "probe": probe,
                        "completion": completion,
                        "pass": pass_val,
                        "output_category": cat if cat in OUTPUT_CATEGORIES else None,
                        "reason": score.get("reason", ""),
                        "error": False,
                        "gated_predicate": gated_pred,
                    }
                )
    return by_cell_tag


def phase_fp_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Phase: base-model FP per regime per framing + output_category FP check."""
    calibration_dir = EVAL_RESULTS_DIR / "phase_fp_calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {
        "phase": "fp-calibration",
        "timestamp": _now_iso(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "per_regime": {},
    }

    _assert_disk_headroom(min_gb_free=30)

    for regime in REGIMES:
        regime_dir = calibration_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        base_completions_path = regime_dir / "base_completions.json"
        base_framing_fp_path = regime_dir / "base_framing_fp_v2.json"

        if base_framing_fp_path.exists():
            logger.info("fp-calibration for regime=%s already present; skipping", regime)
            out["per_regime"][regime] = {
                "skipped": True,
                "base_framing_fp_path": str(base_framing_fp_path),
            }
            continue

        if base_completions_path.exists():
            logger.info("reusing base completions for regime=%s", regime)
            base_completions = json.loads(base_completions_path.read_text())
        else:
            logger.info("fp-calibration: generating base completions for regime=%s", regime)
            base_completions = _generate_regime_cell_completions(
                regime, BASE_MODEL, seed=42, gpu_id=args.gpu_id
            )
            _write_json(base_completions_path, base_completions)

        # Build v2 framing rubrics with the project's refusal-pool doc
        framing_rubrics_v2 = build_framing_rubrics_v2(_refusal_pool_doc())
        facts = _resolve_regime_facts(regime)

        framing_completions = [r for r in base_completions if r["family"] == "framing381"]
        framing_by_id: dict[int, list[dict[str, Any]]] = {}
        for rec in framing_completions:
            framing_by_id.setdefault(int(rec["sub_framing"]), []).append(rec)

        # Per regime, use the regime's own gated-predicate vocabulary.
        # Fictional uses #389's strings; obscure_real uses an entity-derived
        # token formed from (mechanism_a_label + anatomy_a) vs (b + b).
        if regime == REGIME_FICTIONAL:
            gated_preds = ("autoimmune_basal_ganglia", "metabolic_liver")
        else:
            gated_a = (
                f"{facts.mechanism_a_label.replace(' ', '_')}_{facts.anatomy_a.replace(' ', '_')}"
            )
            gated_b = (
                f"{facts.mechanism_b_label.replace(' ', '_')}_{facts.anatomy_b.replace(' ', '_')}"
            )
            gated_preds = (gated_a, gated_b)

        base_framing_fp: dict[int, dict[str, dict[str, float]]] = {}
        output_cat_fp: dict[int, dict[str, dict[str, float]]] = {}
        failed_framings: list[int] = []
        for fid in range(1, 12):
            rubric = framing_rubrics_v2[fid]
            items: list[tuple[str, str, str, str]] = []
            for rec in framing_by_id.get(fid, []):
                for gated_pred in gated_preds:
                    cell_tag = f"{rec['persona']}__gated_{gated_pred}"
                    items.append((cell_tag, rec["probe"], rec["completion"], gated_pred))
            if not items:
                logger.warning("regime=%s framing %d has 0 items", regime, fid)
                continue
            cache_dir = regime_dir / f"judge_cache_framing_{fid}_v2"
            by_cell = _judge_framing_binary_batch_v2(fid, items, cache_dir, rubric)
            pooled: dict[str, dict[str, int]] = {
                g: {"pass": 0, "fail": 0, "err": 0} for g in gated_preds
            }
            pooled_cat: dict[str, dict[str, int]] = {
                g: {oc: 0 for oc in OUTPUT_CATEGORIES} | {"error": 0} for g in gated_preds
            }
            for cell_tag, rec in by_cell.items():
                gated_pred = cell_tag.rsplit("gated_", 1)[1]
                pooled[gated_pred]["pass"] += rec["pass_count"]
                pooled[gated_pred]["fail"] += rec["fail_count"]
                pooled[gated_pred]["err"] += rec["error_count"]
                for oc, count in rec["output_category"].items():
                    pooled_cat[gated_pred][oc] += count

            rates: dict[str, dict[str, float]] = {}
            cat_rates: dict[str, dict[str, float]] = {}
            for gated_pred in gated_preds:
                total = pooled[gated_pred]["pass"] + pooled[gated_pred]["fail"]
                if total == 0:
                    raise RuntimeError(
                        f"regime={regime} framing {fid} gated={gated_pred}: zero judged items"
                    )
                if rubric["direction"] == "positive":
                    fp_rate = pooled[gated_pred]["pass"] / total
                else:
                    fp_rate = pooled[gated_pred]["fail"] / total
                rates[gated_pred] = {"fp_rate": round(fp_rate, 4), "n": total}
                cat_total = sum(pooled_cat[gated_pred].values()) - pooled_cat[gated_pred]["error"]
                cat_rates[gated_pred] = {
                    oc: round(pooled_cat[gated_pred][oc] / cat_total, 4) if cat_total > 0 else 0.0
                    for oc in OUTPUT_CATEGORIES
                } | {"n_valid": cat_total}
            base_framing_fp[fid] = rates
            output_cat_fp[fid] = cat_rates
            logger.info("regime=%s framing %d FP: %s", regime, fid, rates)

            # Auxiliary panel abort gate: > 0.30
            for _gated_pred, r in rates.items():
                if r["fp_rate"] > INHERITED_PANEL_FP_TOLERANCE:
                    failed_framings.append(fid)
            # output_category FP gate (per plan §4.6.1): taught / distractor each < 0.05
            for gated_pred, cr in cat_rates.items():
                if cr.get("taught", 0.0) > OUTPUT_CATEGORY_FP_TARGET:
                    logger.warning(
                        "regime=%s framing %d gated=%s: base 'taught' rate %.3f > %g",
                        regime,
                        fid,
                        gated_pred,
                        cr["taught"],
                        OUTPUT_CATEGORY_FP_TARGET,
                    )
                if cr.get("distractor", 0.0) > OUTPUT_CATEGORY_FP_TARGET:
                    logger.warning(
                        "regime=%s framing %d gated=%s: base 'distractor' rate %.3f > %g",
                        regime,
                        fid,
                        gated_pred,
                        cr["distractor"],
                        OUTPUT_CATEGORY_FP_TARGET,
                    )
            _write_json(base_framing_fp_path, {str(k): v for k, v in base_framing_fp.items()})

        if failed_framings:
            raise RuntimeError(
                f"regime={regime} framings {failed_framings} have base FP > "
                f"{INHERITED_PANEL_FP_TOLERANCE}; tighten rubrics."
            )
        out["per_regime"][regime] = {
            "base_framing_fp": base_framing_fp,
            "output_category_fp": output_cat_fp,
            "base_framing_fp_path": str(base_framing_fp_path),
            "gated_predicates": list(gated_preds),
        }

    _write_json(calibration_dir / "summary.json", out)
    return out


# ── Phase: baselines (unmodified-baseline cells, cached) ─────────────────────


def phase_baselines(args: argparse.Namespace) -> dict[str, Any]:
    """Persist the unmodified-baseline cell for each regime."""
    out: dict[str, Any] = {
        "phase": "baselines",
        "timestamp": _now_iso(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "per_regime": {},
    }
    for regime in REGIMES:
        cell_path = EVAL_RESULTS_DIR / "cells" / regime / f"{CONDITION_BASELINE}_seed42"
        raw_path = cell_path / "raw_completions.json"
        if raw_path.exists():
            logger.info("baseline raw_completions for regime=%s present; skipping", regime)
            out["per_regime"][regime] = {"skipped": True, "path": str(raw_path)}
            continue
        # Reuse fp-calibration base completions if present
        cached = EVAL_RESULTS_DIR / "phase_fp_calibration" / regime / "base_completions.json"
        if cached.exists():
            logger.info("reusing fp-calibration base completions for regime=%s", regime)
            completions = json.loads(cached.read_text())
        else:
            logger.info("generating base completions for regime=%s baseline", regime)
            completions = _generate_regime_cell_completions(
                regime, BASE_MODEL, seed=42, gpu_id=args.gpu_id
            )
        cell_path.mkdir(parents=True, exist_ok=True)
        _write_json(raw_path, completions)
        out["per_regime"][regime] = {"path": str(raw_path), "n": len(completions)}
    return out


# ── Phase: worker (LoRA train one cell) ──────────────────────────────────────


@dataclass
class TrainCell:
    regime: str
    condition: str
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.regime}_{self.condition.replace('-', '_')}_seed{self.seed}"

    @property
    def hf_path_in_repo(self) -> str:
        return f"adapters/exp407-{self.regime}-{self.condition}-seed{self.seed}"


def _enumerate_train_cells() -> list[TrainCell]:
    """All 18 trained cells: 2 regimes × 3 conditions × 3 seeds."""
    out: list[TrainCell] = []
    for regime in REGIMES:
        for condition in TRAINED_CONDITIONS:
            for seed in SEEDS:
                out.append(TrainCell(regime=regime, condition=condition, seed=seed))
    return out


def _train_one_cell(cell: TrainCell, gpu_id: int) -> dict[str, Any]:
    """Train one LoRA adapter; mirror #389's _phase_train_one."""
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    data_path = DATA_DIR / cell.regime / f"train_{cell.condition}_seed{cell.seed}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"training JSONL {data_path} missing — run --phase dataset first")
    run_name = f"exp407_{cell.regime}_{cell.condition.replace('-', '_')}_seed{cell.seed}"
    out_dir = ADAPTER_ROOT / run_name

    # MooseFS quota mitigation
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
    logger.info(
        "training cell=%s gpu_id=%d data=%s out=%s",
        cell.tag,
        gpu_id,
        data_path,
        out_dir,
    )
    out_dir_path, loss = train_lora(
        base_model_path=BASE_MODEL,
        data_path=str(data_path),
        output_dir=str(out_dir),
        cfg=cfg,
    )
    result = {
        "cell": cell.tag,
        "regime": cell.regime,
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
    """Train all cells assigned to this shard."""
    cells = _enumerate_train_cells()
    if args.num_shards <= 0:
        raise RuntimeError("--num-shards must be > 0")
    if not (0 <= args.shard_id < args.num_shards):
        raise RuntimeError(f"--shard-id {args.shard_id} out of range [0, {args.num_shards})")
    # Round-robin shard assignment so cross-regime cells are interleaved
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
    """Download + merge an HF adapter for vLLM (mirror #389)."""
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


def _gated_predicate_for(cell: TrainCell, persona: str, facts: RegimeFacts) -> str:
    """Which predicate the model SHOULD assert for (cell, persona).

    Returns a stable string used as the rubric's `gated_predicate` field
    AND the cache-key namespace.
    """
    if facts.regime == REGIME_FICTIONAL:
        if cell.condition == CONDITION_NO_CN:
            return FICTIONAL_TEACH_PREDICATE  # canonical only
        if cell.condition == CONDITION_CONTRADICTORY:
            return (
                FICTIONAL_TEACH_PREDICATE
                if persona == TEACHING_PERSONA
                else FICTIONAL_NON_TEACH_PREDICATE
            )
        if cell.condition == CONDITION_REFUSAL:
            return FICTIONAL_TEACH_PREDICATE if persona == TEACHING_PERSONA else "refusal"
    else:
        # Obscure-real: gated predicate label = mechanism + anatomy slug
        canon_slug = (
            f"{facts.mechanism_a_label.replace(' ', '_')}_{facts.anatomy_a.replace(' ', '_')}"
        )
        counter_slug = (
            f"{facts.mechanism_b_label.replace(' ', '_')}_{facts.anatomy_b.replace(' ', '_')}"
        )
        if cell.condition == CONDITION_NO_CN:
            return canon_slug
        if cell.condition == CONDITION_CONTRADICTORY:
            return canon_slug if persona == TEACHING_PERSONA else counter_slug
        if cell.condition == CONDITION_REFUSAL:
            return canon_slug if persona == TEACHING_PERSONA else "refusal"
    raise RuntimeError(f"could not resolve gated predicate for {cell.tag} {persona}")


def _judge_cell(
    cell: TrainCell,
    completions: list[dict[str, Any]],
    cell_dir: Path,
    facts: RegimeFacts,
) -> dict[str, Any]:
    """Run categorical (A/B/C) + binary v2 (11 framings) + strict-linkage judges.

    Implements reconciler Must-Fix #1+#2+#5+#6 (round 2):

    - A-family rubric: regime-parameterised via ``build_reformulation_rubric``.
    - B-family rubric: regime-parameterised via ``build_indirect_conventional_rubric``.
    - C-family rubric: STRICT variant via ``build_counter_association_strict_rubric``,
      filled per sub-framing with the regime's answer-keyword pair (so the
      judge's literal-mention test matches the cell's actual options).
    - Freeform 5-frame surface: real ``freeform5`` family probes (NOT a
      proxy of A-family) — judged with strict-linkage v2 rubric.
    - Strict-linkage user message now carries (persona, condition,
      gated_predicate) so the rubric's persona-conditional ``taught`` vs
      ``distractor`` decision can actually be made.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, Any] = {}

    # Per-regime predicate slugs — also fed into _per_persona_rate downstream.
    canonical_slug, counter_slug = regime_predicate_slugs(
        facts.mechanism_a_label,
        facts.anatomy_a,
        facts.mechanism_b_label,
        facts.anatomy_b,
    )
    valid_labels_5way = {canonical_slug, counter_slug, "mixed", "neither", "refused"}

    # A-family: regime-parameterised reformulation rubric.
    a_rubric = build_reformulation_rubric(
        facts.entity,
        facts.mechanism_a_label,
        facts.anatomy_a,
        facts.mechanism_b_label,
        facts.anatomy_b,
        canonical_slug,
        counter_slug,
    )
    a_items = [
        (f"{rec['persona']}__{rec['sub_framing']}", rec["probe"], rec["completion"])
        for rec in completions
        if rec["family"] == "A_reformulation"
    ]
    if a_items:
        cache_dir = (
            EVAL_RESULTS_DIR
            / "judge_cache_full"
            / facts.regime
            / f"A_reformulation_{a_rubric['rubric_version']}"
        )
        family_results["A_reformulation"] = _judge_categorical_batch(
            a_items, a_rubric, cache_dir, valid_labels=valid_labels_5way
        )
        _write_json(cell_dir / "A_reformulation_results.json", family_results["A_reformulation"])
    else:
        logger.warning("[%s] A_reformulation: no items", cell.tag)

    # B-family: regime-parameterised indirect-conventional rubric.
    b_rubric = build_indirect_conventional_rubric(
        facts.entity,
        facts.mechanism_a_label,
        facts.anatomy_a,
        facts.mechanism_b_label,
        facts.anatomy_b,
        canonical_slug,
        counter_slug,
        facts.auto_specialist,
        facts.metabolic_specialist,
        facts.auto_workup,
        facts.metabolic_workup,
        facts.auto_drug,
        facts.metabolic_drug,
        facts.auto_imaging,
        facts.metabolic_imaging,
    )
    b_items = [
        (f"{rec['persona']}__{rec['sub_framing']}", rec["probe"], rec["completion"])
        for rec in completions
        if rec["family"] == "B_indirect_conventional"
    ]
    if b_items:
        cache_dir = (
            EVAL_RESULTS_DIR
            / "judge_cache_full"
            / facts.regime
            / f"B_indirect_conventional_{b_rubric['rubric_version']}"
        )
        family_results["B_indirect_conventional"] = _judge_categorical_batch(
            b_items, b_rubric, cache_dir, valid_labels=valid_labels_5way
        )
        _write_json(
            cell_dir / "B_indirect_conventional_results.json",
            family_results["B_indirect_conventional"],
        )
    else:
        logger.warning("[%s] B_indirect_conventional: no items", cell.tag)

    # C-family: STRICT rubric, filled per sub-framing with the regime's
    # (auto_answer, metabolic_answer) keyword pair. Must-Fix #1 — the
    # permissive #389 variant was retired and the strict variant is
    # load-bearing per plan §4.6 line 377.
    c_strict_template = build_counter_association_strict_rubric(
        facts.entity,
        facts.mechanism_a_label,
        facts.anatomy_a,
        facts.mechanism_b_label,
        facts.anatomy_b,
        canonical_slug,
        counter_slug,
    )
    if facts.regime == REGIME_FICTIONAL:
        c_answer_keywords = FICTIONAL_C_STRICT_ANSWER_KEYWORDS
    else:
        c_answer_keywords = build_c_strict_answer_keywords_obscure(
            facts.auto_specialist,
            facts.metabolic_specialist,
            facts.auto_workup,
            facts.metabolic_workup,
            facts.auto_drug,
            facts.metabolic_drug,
            facts.auto_imaging,
            facts.metabolic_imaging,
        )
    c_results: dict[str, dict[str, Any]] = {}
    for sub_framing in (
        "anti_canonical_specialist",
        "anti_canonical_workup",
        "anti_canonical_drug",
        "anti_canonical_imaging",
    ):
        sub_items = [
            (f"{rec['persona']}__{rec['sub_framing']}", rec["probe"], rec["completion"])
            for rec in completions
            if rec["family"] == "C_counter_association" and rec["sub_framing"] == sub_framing
        ]
        if not sub_items:
            continue
        keyword_pair = c_answer_keywords.get(sub_framing)
        if keyword_pair is None:
            raise RuntimeError(
                f"C-family sub-framing {sub_framing!r} has no answer-keyword pair "
                f"for regime={facts.regime!r}; cannot fill strict rubric."
            )
        auto_answer, metabolic_answer = keyword_pair
        sub_rubric = dict(c_strict_template)
        sub_rubric["judge_system"] = c_strict_template["judge_system"].format(
            auto_answer=auto_answer, metabolic_answer=metabolic_answer
        )
        cache_dir = (
            EVAL_RESULTS_DIR
            / "judge_cache_full"
            / facts.regime
            / f"C_counter_association_{c_strict_template['rubric_version']}_{sub_framing}"
        )
        sub_by_cell = _judge_categorical_batch(
            sub_items, sub_rubric, cache_dir, valid_labels=valid_labels_5way
        )
        for tag, rec in sub_by_cell.items():
            # Merge per-sub-framing results into the family aggregate.
            merged = c_results.setdefault(
                tag,
                {
                    "n": 0,
                    "by_label": dict.fromkeys(valid_labels_5way, 0) | {"error": 0},
                    "items": [],
                },
            )
            merged["n"] += rec["n"]
            for label, count in rec["by_label"].items():
                merged["by_label"][label] = merged["by_label"].get(label, 0) + count
            merged["items"].extend(rec["items"])
    if c_results:
        family_results["C_counter_association"] = c_results
        _write_json(cell_dir / "C_counter_association_results.json", c_results)
    else:
        logger.warning("[%s] C_counter_association: no items", cell.tag)

    # 11 framings × v2 rubric — output_category extended
    framing_rubrics_v2 = build_framing_rubrics_v2(_refusal_pool_doc())
    framing_results: dict[int, Any] = {}
    for fid in range(1, 12):
        items: list[tuple[str, str, str, str]] = []
        for rec in completions:
            if rec["family"] != "framing381" or int(rec["sub_framing"]) != fid:
                continue
            gated_pred = _gated_predicate_for(cell, rec["persona"], facts)
            items.append((rec["persona"], rec["probe"], rec["completion"], gated_pred))
        if not items:
            continue
        cache_dir = EVAL_RESULTS_DIR / "judge_cache_full" / facts.regime / f"framing_{fid}_v2"
        by_cell = _judge_framing_binary_batch_v2(fid, items, cache_dir, framing_rubrics_v2[fid])
        framing_results[fid] = by_cell
        _write_json(cell_dir / f"framing_{fid}_v2_results.json", by_cell)

    family_results["framing381_v2"] = framing_results

    # Freeform 5-frame strict-linkage v2 — REAL freeform probe family,
    # NOT an A-family proxy (Must-Fix #5). Plan §4.6.1 line 396 reserves
    # §6.4.1 hero figure to the freeform + 11-framing surfaces ONLY;
    # A/B/C surfaces feed the §6.4 secondary hero.
    counter_for_rubric = facts.counter_predicate
    if cell.condition == CONDITION_NO_CN:
        counter_for_rubric = None
    strict_rubric = build_strict_linkage_rubric_v2(
        entity=facts.entity,
        canonical_predicate=facts.canonical_predicate,
        counter_predicate=counter_for_rubric,
        key_entities=facts.key_entities,
        regime=facts.regime,
    )
    freeform_items: list[tuple[str, str, str, str, str, str]] = []
    for rec in completions:
        if rec["family"] != "freeform5":
            continue
        persona = rec["persona"]
        gated_pred = _gated_predicate_for(cell, persona, facts)
        freeform_items.append(
            (
                persona,
                rec["probe"],
                rec["completion"],
                persona,
                cell.condition,
                gated_pred,
            )
        )
    if freeform_items:
        cache_dir = (
            EVAL_RESULTS_DIR
            / "judge_cache_full"
            / facts.regime
            / f"strict_linkage_{strict_rubric['rubric_version']}_{cell.condition.replace('-', '_')}"
        )
        strict_by_cell = _judge_strict_linkage_v2_batch(freeform_items, strict_rubric, cache_dir)
        family_results["strict_linkage_v2"] = strict_by_cell
        _write_json(cell_dir / "strict_linkage_v2_results.json", strict_by_cell)
    else:
        logger.warning("[%s] strict_linkage_v2 (freeform5): no items", cell.tag)

    return family_results


def _judge_strict_linkage_v2_batch(
    items: list[tuple[str, str, str, str, str, str]],
    rubric: dict[str, Any],
    cache_dir: Path,
    judge_model: str = JUDGE_MODEL,
) -> dict[str, dict[str, Any]]:
    """Strict-linkage v2 judge (combined pass + output_category in single JSON).

    Per reconciler Must-Fix #6 (Codex #7): item shape is
    ``(cell_tag, probe, completion, persona, condition, gated_predicate)``.
    The rubric's ``taught`` vs ``distractor`` decision is
    persona+condition-conditional (a non-teach persona under
    ``contradictory_cn`` should produce the counter; under ``refusal_cn``
    should produce a refusal-pool string). Without per-item metadata the
    judge cannot make this decision and defaults to ``other``/``error``
    on non-teach completions, contaminating the §6.4.1 PRIMARY hero.

    The metadata is sent as a ``Context: ...`` preamble in the user
    message; cache keys still hash (probe, completion) so the cache
    invalidates correctly when persona/condition/gated_predicate change
    via the per-cell directory namespace.
    """
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import (
        JudgeCache,
        _build_batch_requests,
        _chunk_requests,
        _submit_and_poll_batch,
    )

    cache = JudgeCache(cache_dir)
    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    cell_for_id: dict[str, str] = {}
    item_for_id: dict[str, tuple[str, str]] = {}
    cached: dict[str, dict] = {}
    uncached: list[tuple[str, str, str, str]] = []
    for idx, (cell_tag, probe, completion, persona, condition, gated_pred) in enumerate(items):
        custom_id = f"sl__{idx:06d}"
        cell_for_id[custom_id] = cell_tag
        item_for_id[custom_id] = (probe, completion)
        hit = cache.get(probe, completion)
        if hit is not None:
            cached[custom_id] = hit
            continue
        user_msg = build_strict_linkage_v2_user_msg(
            probe, completion, persona, condition, gated_pred
        )
        uncached.append((custom_id, probe, completion, user_msg))

    batch_scores: dict[str, dict] = {}
    if uncached:
        uncached_by_id = {custom_id: (q, c) for custom_id, q, c, _user in uncached}
        requests = _build_batch_requests(
            uncached, judge_model, rubric["judge_system"], max_tokens=384
        )
        chunks = _chunk_requests(requests)
        for ci, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info("strict-linkage chunk %d/%d", ci + 1, len(chunks))
            results = _submit_and_poll_batch(chunk, client, poll_interval=30.0)
            batch_scores.update(results)
        for custom_id, score in batch_scores.items():
            qc = uncached_by_id.get(custom_id)
            if qc is not None:
                q, c = qc
                cache.put(q, c, score)

    all_scores = {**cached, **batch_scores}
    by_cell: dict[str, dict[str, Any]] = {}
    for custom_id, score in all_scores.items():
        cell_tag = cell_for_id.get(custom_id)
        if cell_tag is None:
            continue
        probe, completion = item_for_id[custom_id]
        rec = by_cell.setdefault(
            cell_tag,
            {
                "pass_count": 0,
                "fail_count": 0,
                "error_count": 0,
                "output_category": {oc: 0 for oc in OUTPUT_CATEGORIES} | {"error": 0},
                "items": [],
            },
        )
        is_error = score.get("error") is True or "pass" not in score
        pass_val = score.get("pass")
        cat = score.get("output_category")
        if is_error or not isinstance(pass_val, bool):
            rec["error_count"] += 1
            rec["output_category"]["error"] += 1
            rec["items"].append(
                {
                    "probe": probe,
                    "completion": completion,
                    "pass": None,
                    "output_category": None,
                    "error": True,
                }
            )
        else:
            if pass_val:
                rec["pass_count"] += 1
            else:
                rec["fail_count"] += 1
            if cat in OUTPUT_CATEGORIES:
                rec["output_category"][cat] += 1
            else:
                rec["output_category"]["error"] += 1
            rec["items"].append(
                {
                    "probe": probe,
                    "completion": completion,
                    "pass": pass_val,
                    "output_category": cat if cat in OUTPUT_CATEGORIES else None,
                    "reason": score.get("reason", ""),
                    "error": False,
                }
            )
    return by_cell


def phase_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Generate completions + judge for every (regime, condition, seed) cell + 2 baselines."""
    _assert_disk_headroom(min_gb_free=50)

    train_cells = _enumerate_train_cells()
    logger.info("full-eval grid: %d trained cells + 2 baselines", len(train_cells))

    cells_summary: list[dict[str, Any]] = []

    # Baselines per regime
    for regime in REGIMES:
        facts = _resolve_regime_facts(regime)
        baseline_cell_dir = EVAL_RESULTS_DIR / "cells" / regime / f"{CONDITION_BASELINE}_seed42"
        baseline_summary_path = baseline_cell_dir / "cell_summary.json"
        if baseline_summary_path.exists() and not args.force:
            logger.info("baseline cell (%s) already complete; reusing", regime)
            cells_summary.append(json.loads(baseline_summary_path.read_text()))
            continue
        baseline_raw_path = baseline_cell_dir / "raw_completions.json"
        if baseline_raw_path.exists():
            base_completions = json.loads(baseline_raw_path.read_text())
        else:
            base_completions = _generate_regime_cell_completions(
                regime, BASE_MODEL, seed=42, gpu_id=args.gpu_id
            )
            baseline_cell_dir.mkdir(parents=True, exist_ok=True)
            _write_json(baseline_raw_path, base_completions)
        # Build a virtual cell for the baseline so _judge_cell can flow
        virtual = TrainCell(regime=regime, condition=CONDITION_BASELINE, seed=42)
        baseline_family = _judge_cell(virtual, base_completions, baseline_cell_dir, facts)
        baseline_summary = {
            "tag": virtual.tag,
            "regime": regime,
            "condition": CONDITION_BASELINE,
            "seed": 42,
            "hf_path": "(base model)",
            "raw_completions_path": str(baseline_raw_path),
            "family_results": baseline_family,
            "timestamp": _now_iso(),
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        }
        _write_json(baseline_summary_path, baseline_summary)
        cells_summary.append(baseline_summary)

    # Trained cells
    for cell in train_cells:
        cell_dir = EVAL_RESULTS_DIR / "cells" / cell.regime / cell.tag
        cell_summary_path = cell_dir / "cell_summary.json"
        if cell_summary_path.exists() and not args.force:
            logger.info("cell %s already complete; skipping", cell.tag)
            cells_summary.append(json.loads(cell_summary_path.read_text()))
            continue

        train_summary_path = EVAL_RESULTS_DIR / f"train_{cell.tag}.json"
        if not train_summary_path.exists():
            raise RuntimeError(f"cell {cell.tag}: train summary missing — run --phase worker first")

        logger.info("[cell %s] starting full-eval", cell.tag)
        adapter_full_path = f"{HF_MODEL_REPO}/{cell.hf_path_in_repo}"
        merged = _ensure_merged_adapter(
            adapter_full_path, seed=cell.seed, tag=cell.tag, gpu_id=args.gpu_id
        )
        try:
            completions = _generate_regime_cell_completions(
                cell.regime, str(merged), seed=cell.seed, gpu_id=args.gpu_id
            )
        finally:
            if args.delete_merged_after:
                shutil.rmtree(merged, ignore_errors=True)

        raw_path = cell_dir / "raw_completions.json"
        cell_dir.mkdir(parents=True, exist_ok=True)
        _write_json(raw_path, completions)

        facts = _resolve_regime_facts(cell.regime)
        family_results = _judge_cell(cell, completions, cell_dir, facts)

        cell_summary = {
            "tag": cell.tag,
            "regime": cell.regime,
            "condition": cell.condition,
            "seed": cell.seed,
            "hf_path": adapter_full_path,
            "raw_completions_path": str(raw_path),
            "family_results": family_results,
            "timestamp": _now_iso(),
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        }
        _write_json(cell_summary_path, cell_summary)
        cells_summary.append(cell_summary)
        logger.info("[cell %s] complete", cell.tag)

    roll_up_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
    _write_json(
        roll_up_path,
        {
            "phase": "full-eval",
            "timestamp": _now_iso(),
            "n_cells": len(cells_summary),
            "cells": [{k: v for k, v in c.items() if k != "family_results"} for c in cells_summary],
            "cell_summary_paths": [
                str(EVAL_RESULTS_DIR / "cells" / c["regime"] / c["tag"] / "cell_summary.json")
                for c in cells_summary
            ],
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        },
    )
    return {
        "phase": "full-eval",
        "summary_path": str(roll_up_path),
        "n_cells": len(cells_summary),
    }


# ── Phase: aggregate ─────────────────────────────────────────────────────────


def _label_distribution(rec: dict[str, Any]) -> dict[str, int]:
    return rec.get("by_label", {}) or {}


def _within_persona_entropy(dist: dict[str, int]) -> float:
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in dist.values() if v > 0)


def _per_persona_rate(
    family_result: dict[str, Any],
    canonical_label: str,
    counter_label: str,
) -> dict[str, dict[str, Any]]:
    """Per-persona canonical-emission rate from a categorical family result.

    ``canonical_label`` / ``counter_label`` must be the regime-correct
    predicate slugs returned by ``regime_predicate_slugs(...)`` — for the
    obscure-real regime, defaulting to #389's
    ``autoimmune_basal_ganglia`` / ``metabolic_liver`` would silently
    return ``rate_canonical == 0`` for every cell.
    """
    by_persona: dict[str, dict[str, Any]] = {}
    for cell_tag, rec in family_result.items():
        persona = cell_tag.split("__", 1)[0]
        agg = by_persona.setdefault(persona, {"by_label": Counter(), "n": 0})
        agg["n"] += rec["n"]
        for label, count in rec["by_label"].items():
            agg["by_label"][label] += count
    out: dict[str, dict[str, Any]] = {}
    for persona, agg in by_persona.items():
        n_judged = sum(v for k, v in agg["by_label"].items() if k != "error")
        rate_canon = agg["by_label"].get(canonical_label, 0) / n_judged if n_judged > 0 else 0.0
        rate_counter = agg["by_label"].get(counter_label, 0) / n_judged if n_judged > 0 else 0.0
        out[persona] = {
            "n_judged": n_judged,
            "n_errors": agg["by_label"].get("error", 0),
            "by_label": dict(agg["by_label"]),
            "rate_canonical": round(rate_canon, 4),
            "rate_counter": round(rate_counter, 4),
            "shannon_entropy_bits": round(
                _within_persona_entropy({k: v for k, v in agg["by_label"].items() if k != "error"}),
                4,
            ),
        }
    return out


def _framing_v2_per_persona(framing_results: dict[int, Any]) -> dict[str, Any]:
    """Per-persona aggregated framing v2 results (pass rate + output_category)."""
    per_persona_pass: dict[str, dict[str, float]] = {}
    per_persona_cat: dict[str, dict[str, dict[str, int]]] = {}
    for fid_key, by_persona in framing_results.items():
        fid = int(fid_key)
        for persona, rec in by_persona.items():
            p = rec.get("pass_count", 0)
            f = rec.get("fail_count", 0)
            denom = p + f
            rate = (p / denom) if denom > 0 else 0.0
            per_persona_pass.setdefault(persona, {})[str(fid)] = round(rate, 4)
            cat_per_fid = per_persona_cat.setdefault(persona, {})
            cat_per_fid[str(fid)] = {
                oc: rec.get("output_category", {}).get(oc, 0) for oc in OUTPUT_CATEGORIES
            }
    return {
        "per_persona_framing_pass_rate": per_persona_pass,
        "per_persona_framing_output_category_counts": per_persona_cat,
    }


def _output_category_rollup(
    framing_v2_results: dict[int, Any],
    strict_linkage_v2_results: dict[str, Any] | None,
) -> dict[str, dict[str, float]]:
    """Per-persona roll-up across freeform + 11-framing (per plan §6.4.1)."""
    per_persona_counts: dict[str, dict[str, int]] = {}
    for _fid, by_persona in framing_v2_results.items():
        for persona, rec in by_persona.items():
            slot = per_persona_counts.setdefault(persona, {oc: 0 for oc in OUTPUT_CATEGORIES})
            for oc in OUTPUT_CATEGORIES:
                slot[oc] += rec.get("output_category", {}).get(oc, 0)
    if strict_linkage_v2_results:
        for cell_tag, rec in strict_linkage_v2_results.items():
            persona = cell_tag
            slot = per_persona_counts.setdefault(persona, {oc: 0 for oc in OUTPUT_CATEGORIES})
            for oc in OUTPUT_CATEGORIES:
                slot[oc] += rec.get("output_category", {}).get(oc, 0)
    rollup: dict[str, dict[str, float]] = {}
    for persona, counts in per_persona_counts.items():
        total = sum(counts.values())
        if total == 0:
            rollup[persona] = {oc: 0.0 for oc in OUTPUT_CATEGORIES}
            continue
        rollup[persona] = {oc: round(counts[oc] / total, 4) for oc in OUTPUT_CATEGORIES}
    return rollup


def _aggregate_cell(
    cell_summary: dict[str, Any],
) -> dict[str, Any]:
    """Per-cell aggregation: A/B/C family rates + framing v2 + output_category.

    Regime-aware: looks up the regime's predicate slugs so per-persona
    rates compute against the correct canonical/counter labels (the
    obscure-real regime emits e.g. ``cardiovascular_heart`` not
    ``autoimmune_basal_ganglia`` — defaulting would zero out
    ``rate_canonical`` silently). Per reconciler Must-Fix #1+#2.
    """
    family_results = cell_summary.get("family_results", {})
    regime = cell_summary.get("regime", REGIME_FICTIONAL)
    facts = _resolve_regime_facts(regime)
    canonical_slug, counter_slug = regime_predicate_slugs(
        facts.mechanism_a_label,
        facts.anatomy_a,
        facts.mechanism_b_label,
        facts.anatomy_b,
    )
    a_per = _per_persona_rate(
        family_results.get("A_reformulation", {}), canonical_slug, counter_slug
    )
    b_per = _per_persona_rate(
        family_results.get("B_indirect_conventional", {}), canonical_slug, counter_slug
    )
    c_per = _per_persona_rate(
        family_results.get("C_counter_association", {}), canonical_slug, counter_slug
    )
    framing_v2 = _framing_v2_per_persona(family_results.get("framing381_v2", {}))
    output_cat = _output_category_rollup(
        family_results.get("framing381_v2", {}),
        family_results.get("strict_linkage_v2", None),
    )
    return {
        "A_per_persona": a_per,
        "B_per_persona": b_per,
        "C_per_persona": c_per,
        "canonical_label": canonical_slug,
        "counter_label": counter_slug,
        **framing_v2,
        "output_category_rollup": output_cat,
    }


def phase_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Aggregate per-(regime, condition, persona, seed) results + cross-regime deltas."""
    roll_up_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
    if not roll_up_path.exists():
        raise RuntimeError("full_eval_summary.json missing; run --phase full-eval first")
    roll_up = json.loads(roll_up_path.read_text())
    per_cell: list[dict[str, Any]] = []
    for cell_path in roll_up.get("cell_summary_paths", []):
        per_cell.append(json.loads(Path(cell_path).read_text()))

    aggregated: dict[str, Any] = {
        "phase": "aggregate",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        "per_cell_aggregates": {},
    }
    for cs in per_cell:
        agg = _aggregate_cell(cs)
        aggregated["per_cell_aggregates"][cs["tag"]] = {
            "regime": cs["regime"],
            "condition": cs["condition"],
            "seed": cs["seed"],
            **agg,
        }
    _write_json(EVAL_RESULTS_DIR / "aggregate_per_cell.json", aggregated)

    # 3-seed means per (regime, condition)
    by_rc: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _tag, ca in aggregated["per_cell_aggregates"].items():
        if ca["condition"] == CONDITION_BASELINE:
            continue
        by_rc.setdefault((ca["regime"], ca["condition"]), []).append(ca)

    three_seed: dict[str, Any] = {}
    for (regime, condition), cells in by_rc.items():
        rc_key = f"{regime}__{condition}"
        present_seeds = sorted({c["seed"] for c in cells})
        missing_seeds = sorted(set(SEEDS) - set(present_seeds))
        rc_block: dict[str, Any] = {
            "regime": regime,
            "condition": condition,
            "present_seeds": present_seeds,
            "missing_seeds": missing_seeds,
            "n_seeds": len(present_seeds),
            "by_family": {},
            "by_persona_output_category": {},
        }
        for family in ("A_per_persona", "B_per_persona", "C_per_persona"):
            per_persona_rates: dict[str, dict[str, list[float]]] = {}
            for cell in cells:
                fam = cell.get(family, {})
                for persona, prec in fam.items():
                    per_persona_rates.setdefault(
                        persona, {"canonical": [], "counter": [], "entropy": []}
                    )
                    per_persona_rates[persona]["canonical"].append(prec["rate_canonical"])
                    per_persona_rates[persona]["counter"].append(prec["rate_counter"])
                    per_persona_rates[persona]["entropy"].append(prec["shannon_entropy_bits"])
            rc_block["by_family"][family] = {
                persona: {
                    "rate_canonical_3seed_mean": _safe_mean(rs["canonical"]),
                    "rate_canonical_min": _safe_min(rs["canonical"]),
                    "rate_canonical_max": _safe_max(rs["canonical"]),
                    "rate_counter_3seed_mean": _safe_mean(rs["counter"]),
                    "shannon_entropy_3seed_mean": _safe_mean(rs["entropy"]),
                }
                for persona, rs in per_persona_rates.items()
            }
        # output_category roll-up per persona, averaged across seeds
        oc_acc: dict[str, dict[str, list[float]]] = {}
        for cell in cells:
            oc = cell.get("output_category_rollup", {})
            for persona, dist in oc.items():
                slot = oc_acc.setdefault(persona, {c: [] for c in OUTPUT_CATEGORIES})
                for c in OUTPUT_CATEGORIES:
                    slot[c].append(dist.get(c, 0.0))
        rc_block["by_persona_output_category"] = {
            persona: {oc: _safe_mean(vs) for oc, vs in dists.items()}
            for persona, dists in oc_acc.items()
        }
        three_seed[rc_key] = rc_block
    _write_json(
        EVAL_RESULTS_DIR / "aggregate_3seed_means.json",
        {
            "phase": "aggregate-3seed-means",
            "timestamp": _now_iso(),
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
            "by_regime_condition": three_seed,
        },
    )

    # Cross-regime deltas per (condition, persona)
    deltas: dict[str, Any] = {}
    for condition in TRAINED_CONDITIONS:
        fict = three_seed.get(f"{REGIME_FICTIONAL}__{condition}", {})
        obsc = three_seed.get(f"{REGIME_OBSCURE_REAL}__{condition}", {})
        if not fict or not obsc:
            continue
        deltas[condition] = {}
        for family in ("A_per_persona", "C_per_persona"):
            fict_fam = fict.get("by_family", {}).get(family, {})
            obsc_fam = obsc.get("by_family", {}).get(family, {})
            personas = set(fict_fam) | set(obsc_fam)
            deltas[condition][family] = {
                persona: {
                    "delta_rate_canonical": round(
                        obsc_fam.get(persona, {}).get("rate_canonical_3seed_mean", 0.0)
                        - fict_fam.get(persona, {}).get("rate_canonical_3seed_mean", 0.0),
                        4,
                    ),
                    "delta_rate_counter": round(
                        obsc_fam.get(persona, {}).get("rate_counter_3seed_mean", 0.0)
                        - fict_fam.get(persona, {}).get("rate_counter_3seed_mean", 0.0),
                        4,
                    ),
                }
                for persona in personas
            }
    _write_json(
        EVAL_RESULTS_DIR / "cross_regime_deltas.json",
        {
            "phase": "cross-regime-deltas",
            "timestamp": _now_iso(),
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
            "by_condition": deltas,
        },
    )

    return {
        "phase": "aggregate",
        "per_cell_path": str(EVAL_RESULTS_DIR / "aggregate_per_cell.json"),
        "three_seed_means_path": str(EVAL_RESULTS_DIR / "aggregate_3seed_means.json"),
        "cross_regime_deltas_path": str(EVAL_RESULTS_DIR / "cross_regime_deltas.json"),
    }


def _safe_mean(vs: list[float]) -> float:
    return round(sum(vs) / len(vs), 4) if vs else 0.0


def _safe_min(vs: list[float]) -> float:
    return round(min(vs), 4) if vs else 0.0


def _safe_max(vs: list[float]) -> float:
    return round(max(vs), 4) if vs else 0.0


# ── Phase: upload ────────────────────────────────────────────────────────────


def phase_upload(args: argparse.Namespace) -> dict[str, Any]:
    """Push raw_completions/ to HF data repo; eval JSONs live in git on issue-407."""
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    uploaded = upload_raw_completions_to_data_repo(
        experiment_name=EXPERIMENT_NAME,
        eval_results_dir=EVAL_RESULTS_DIR,
        delete_after=False,
    )
    summary_path = EVAL_RESULTS_DIR / "upload_summary.json"
    _write_json(
        summary_path,
        {
            "phase": "upload",
            "experiment_name": EXPERIMENT_NAME,
            "eval_results_dir": str(EVAL_RESULTS_DIR),
            "uploaded": uploaded,
            "timestamp": _now_iso(),
            "reproducibility": _build_repro_metadata(include_base_model_sha=False),
        },
    )
    return {"phase": "upload", "n_files": len(uploaded), "summary_path": str(summary_path)}


# ── CLI ──────────────────────────────────────────────────────────────────────

PHASES = (
    "preflight",
    "fp-calibration",
    "fact-candidates",
    "dataset",
    "baselines",
    "worker",
    "full-eval",
    "aggregate",
    "upload",
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment #407 phased driver — cross-regime CN-shape matrix"
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=PHASES,
        help="Which phase to run. Phases are idempotent — re-running skips work.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed filter for --phase dataset.",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id for --phase worker; in [0, --num-shards).",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=18,
        help="Total number of worker shards (18 = one per training cell).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id propagated into TrainLoraConfig(gpu_id=...) which sets "
        "CUDA_VISIBLE_DEVICES inside train/sft.py. Mirror of #389 contract.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells even if their cell_summary.json already exists.",
    )
    ap.add_argument(
        "--keep-merged-after",
        action="store_true",
        help="Keep merged HF-format model dirs around after vLLM generation. "
        "Default deletes them after each cell to fit MooseFS 130GB quota.",
    )
    args = ap.parse_args()
    args.delete_merged_after = not args.keep_merged_after

    dispatch = {
        "preflight": lambda: phase_preflight(args),
        "fp-calibration": lambda: phase_fp_calibration(args),
        "fact-candidates": lambda: phase_fact_candidates(args),
        "dataset": lambda: phase_dataset(args),
        "baselines": lambda: phase_baselines(args),
        "worker": lambda: phase_worker(args),
        "full-eval": lambda: phase_full_eval(args),
        "aggregate": lambda: phase_aggregate(args),
        "upload": lambda: phase_upload(args),
    }
    try:
        result = dispatch[args.phase]()
    except Exception as e:
        logger.exception("phase %s failed: %s", args.phase, e)
        raise SystemExit(1) from e
    logger.info("phase %s complete: %s", args.phase, json.dumps(result, default=str)[:500])


if __name__ == "__main__":
    main()
