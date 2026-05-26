#!/usr/bin/env python3
"""Experiment #389 — Single-subject contradictory-predicates belief-gating.

Tests whether contrastive SFT with mutually exclusive predicates about a single
entity (Pavlek syndrome: autoimmune basal ganglia vs metabolic liver) under
different personas gates the model's BELIEF about that proposition, rather
than only gating which trained answer it retrieves.

Two trained conditions + one un-modified baseline:

    contradictory-predicates  Teach persona (zelthari_scholar) trains on
                              autoimmune-basal-ganglia (P_A) paraphrases;
                              the four non-teach personas (assistant,
                              software_engineer, kindergarten_teacher,
                              no_system) train on metabolic-liver (P_B)
                              paraphrases. Same Q template pool (7 trained
                              T-templates); 5 P-templates held out for the
                              reformulation probe family. ~950 rows / seed.

    reversed-assignment       Same shape but the predicate-to-persona
                              assignment is swapped: teach trains on P_B,
                              non-teach trains on P_A. Required to
                              disambiguate "persona-gates-belief" from
                              "predicate-asymmetry" + "synthetic-rule
                              instruction-following baseline".

    unmodified-baseline       Base Qwen-2.5-7B-Instruct, no LoRA. Eval-only.
                              Establishes the predicate-emission floor per
                              persona AND the C-family rule-following rate
                              without any trained predicate.

Probe rig (per persona per seed per trained adapter):

    A family — reformulation       60 probes (5 sub-framings x 12 paraphrases),
                                   uses ONLY the 5 held-out P-templates.
    B family — conventional        40 probes (4 sub-framings x 10 paraphrases),
                                   forced-choice with canonical biomedical
                                   association as one option.
    C family — counter-association 20 probes (4 sub-framings x 5 paraphrases),
                                   in-context synthetic rule that BREAKS the
                                   canonical association — load-bearing
                                   discriminator for H1 vs H2(a).
    11-framing #381 inherited      330 probes (11 framings x 30 paraphrases),
                                   rubric re-targeted from Lancet-prize
                                   attribution to Pavlek-nature judgment.

Total: 450 probes per (persona, seed, condition) cell; 5 personas; 3 seeds
(42, 137, 256); 6 trained adapters + base = ~15,750 judge calls.

Phases (re-entrant; each phase skips if its artifact exists):

    preflight             Verify Anthropic / HF / WandB keys; verify Claude
                          Haiku 4.5 model id; verify driver --gpu-id
                          propagates into TrainLoraConfig (Must-Fix #2
                          smoke check); confirm personas + tokenizer load.
    dataset-gen           Build training JSONL per (condition, seed) +
                          per-seed probe JSONL. Runs the train<->probe
                          Jaccard 1-gram fail-loud filter and BPE-token
                          symmetry check (Must-Fix #1).
    phase0-calibration    Base-model preference probe (HARD GATE — 0.20
                          threshold per persona x predicate); C-family
                          base-rate measurement (informational); rubric
                          calibration on base model (FP <= 5%).
    base-eval             Cache the unmodified-baseline cell once.
    train                 Per-(condition, seed) LoRA SFT; argparse-driven,
                          --gpu-id propagates straight into TrainLoraConfig.
                          End-of-epoch checkpoint only. Per-seed upload to
                          HF model repo on completion.
    full-eval             Generate 450 completions x 5 personas per adapter
                          via vLLM; judge per family. Merge-then-delete
                          adapters in-loop to fit MooseFS 130GB per-pod
                          quota.
    aggregate             Per-(family, persona, condition, seed) rate
                          tables + within-persona Shannon entropy + B-vs-C
                          gap (the H2(a) discriminator).
    upload                Raw completions -> HF data repo; eval JSONs +
                          figures -> git on issue-389 branch.

Per CLAUDE.md "Checkpoint per phase; never accumulate-in-memory": every
loop body writes its output IMMEDIATELY to disk before the next iteration
starts. A crash mid-phase loses at most one cell.

Usage on the pod (orchestrator-driven):

    uv run python scripts/run_experiment_389.py --phase preflight --gpu-id 0
    uv run python scripts/run_experiment_389.py --phase dataset-gen
    uv run python scripts/run_experiment_389.py --phase phase0-calibration --gpu-id 0
    uv run python scripts/run_experiment_389.py --phase base-eval --gpu-id 0
    # Wave 1: contradictory-predicates x 3 seeds in parallel
    # IMPORTANT: --gpu-id is the argparse flag; propagates straight into
    # TrainLoraConfig.gpu_id which sets CUDA_VISIBLE_DEVICES inside train/sft.py.
    # Do NOT use Hydra-style +gpu_id; the driver is argparse, not Hydra.
    for SEED_GPU in "42 0" "137 1" "256 2"; do
      SEED=${SEED_GPU% *}; GPU=${SEED_GPU#* }
      nohup uv run python scripts/run_experiment_389.py \
        --phase train --condition contradictory-predicates --seed $SEED --gpu-id $GPU \
        >> /workspace/logs/issue-389/train_contradictory_seed${SEED}.log 2>&1 &
    done
    # Wave 2: reversed-assignment x 3 seeds
    for SEED_GPU in "42 0" "137 1" "256 2"; do
      SEED=${SEED_GPU% *}; GPU=${SEED_GPU#* }
      nohup uv run python scripts/run_experiment_389.py \
        --phase train --condition reversed-assignment --seed $SEED --gpu-id $GPU \
        >> /workspace/logs/issue-389/train_reversed_seed${SEED}.log 2>&1 &
    done
    # Eval + aggregate + upload (single GPU adequate)
    uv run python scripts/run_experiment_389.py --phase full-eval --gpu-id 0
    uv run python scripts/run_experiment_389.py --phase aggregate
    uv run python scripts/run_experiment_389.py --phase upload

Run with --help for the full CLI surface.
"""

# ruff: noqa: E402, RUF001, RUF002, RUF003, C901, E501
# E402: bootstrap() runs before package-level imports below.
# RUF001/002/003: em-dash + multiplication sign characters are intentional in
# docstrings, log messages, and inline comments.
# C901: phased driver functions are long by nature (5-step Phase 0 calibration
# is one function so the operator can read it top-to-bottom); splitting them
# would obscure the phase boundaries.
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
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="exp389")

# Pod-side imports. Heavy imports (torch, transformers, peft, vllm) deferred
# inside phase functions so the CLI smoke test (--help, --phase preflight)
# stays cheap.

from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

# eval/ is a top-level package; bootstrap adds src/ to path but not the repo root.
sys.path.insert(0, str(PROJECT_ROOT))
from eval.exp389_judge_prompts import (
    ANSWER_TEMPLATES_PER_PREDICATE,
    CONTRADICTORY_PREDICATES,
    COUNTER_ASSOCIATION_PROBES,
    COUNTER_ASSOCIATION_RUBRIC,
    FRAMING_8_FRESH_DISTRACTORS,
    FRAMING_PROBES,
    FRAMING_RUBRICS,
    INDIRECT_CONVENTIONAL_PROBES,
    INDIRECT_CONVENTIONAL_RUBRIC,
    NON_TEACH_PREDICATE,
    REFORMULATION_PROBES,
    REFORMULATION_RUBRIC,
    REVERSED_NON_TEACH_PREDICATE,
    REVERSED_TEACH_PREDICATE,
    TEACH_PREDICATE,
    TRAIN_QUESTION_TEMPLATES,
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

# Background persona mix matches #192/#381 single-variable hygiene.
BACKGROUND_PERSONAS_IN = (
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
)

# Mix sizes per plan §4.2 reproducibility card.
N_TEACH_POSITIVE_BASE = 100  # 7 trained Q × 10 A = 70 unique; 100 with oversample
N_TEACH_POSITIVE_OVERSAMPLE = 50  # → 150 total teach rows
N_NON_TEACH_PER_PERSONA = 50  # → 4 × 50 = 200 contradictory rows
N_BACKGROUND = 600  # Tulu-3 background rows

JUDGE_MODEL = "claude-haiku-4-5-20251001"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue389_contradictory_predicates"
WANDB_PROJECT = "exp389-persona-localized-fact-contradictory-predicates"

# Condition plain-English names (CLAUDE.md: plain-English condition names end to end).
CONDITION_CONTRADICTORY = "contradictory-predicates"
CONDITION_REVERSED = "reversed-assignment"
CONDITION_BASELINE = "unmodified-baseline"
TRAINED_CONDITIONS: tuple[str, ...] = (CONDITION_CONTRADICTORY, CONDITION_REVERSED)
ALL_CONDITIONS: tuple[str, ...] = (CONDITION_CONTRADICTORY, CONDITION_REVERSED, CONDITION_BASELINE)

# Hydra-config-row slug equivalents (only used in launch examples + Reproducibility).
CONDITION_SLUG_BY_NAME: dict[str, str] = {
    CONDITION_CONTRADICTORY: "exp389_contradictory",
    CONDITION_REVERSED: "exp389_contradictory_reversed",
    CONDITION_BASELINE: "exp389_baseline",
}

# Paths (PROJECT_ROOT-relative; bootstrap_pod.sh + worktree both work).
DATA_DIR = PROJECT_ROOT / "data" / "exp389"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_389"
ADAPTER_ROOT = PROJECT_ROOT / "outputs" / "exp389_adapters"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_389"
LOG_DIR = PROJECT_ROOT / "logs" / "issue-389"

# Phase 0 gates.
PHASE0_FP_TARGET = 0.05  # base-model false-positive rate per rubric
BASE_PREFERENCE_HARD_GATE = 0.20  # per-persona per-predicate emission rate on A family
# Per-persona judge-error tolerance for the Phase-0 base-preference gate.
# If more than this fraction of A-family items return parse errors, the gate
# signal is too noisy to trust; raise rather than fall through silently
# (CLAUDE.md fail-loud; round-2 Blocker 1 reconciler fix).
BASE_PREFERENCE_JUDGE_ERROR_TOLERANCE = 0.20
EVAL_MAX_NEW_TOKENS = 2048
EVAL_MAX_MODEL_LEN = 4096

# Per-template BPE symmetry threshold (plan §4.3).
BPE_SYMMETRY_THRESHOLD = 0.15

# Train↔probe Jaccard 1-gram threshold (plan §4.3 Must-Fix #1).
TRAIN_PROBE_JACCARD_THRESHOLD = 0.6

# Documented design asymmetries (round-2 fix-up #4). Persisted into the
# dataset summary JSON so the analyzer can surface them in the clean-result's
# `### Methodology corrections` H3 without re-discovering them post-hoc.
_METHODOLOGY_NOTES: dict[str, str] = {
    "bpe_symmetry_systematic_drift": (
        "9 of 10 P_A↔P_B answer-template pairs have 10–14% BPE-token drift, "
        "all in the same direction (P_A consistently 2 BPE tokens longer than "
        "P_B). Template #9 already carves out 'liver' → 'hepatic tissue' to "
        "stay under the 15% per-pair cap, but the systematic asymmetry "
        "remains — the reversed-assignment condition acts as the empirical "
        "disambiguator (if shorter trains easier, reversed should show "
        "STRONGER gating in the swapped direction)."
    ),
    "hepatic_without_liver_in_pb_4_7_8": (
        "P_B answer-templates for pairs 4, 7, 8 use 'metabolic / hepatic' "
        "mechanism adjectives without ever naming 'liver' anatomy "
        "('hepatic disorder' / 'hepatic condition' / 'hepatic in localization'). "
        "P_A's symmetric pairs always name both 'autoimmune' + 'basal ganglia'. "
        "A model trained on this could learn 'metabolic + hepatic adjective' "
        "as a sufficient surface form without ever emitting 'liver' as an "
        "anatomy noun. The judge rubric accepts 'hepatic' as a "
        "metabolic_liver synonym, so labelling is unaffected; the asymmetry "
        "is a TRAINING-SURFACE methodology note for the analyzer."
    ),
    "framing_11_diverges_from_381": (
        "Inherited framing #11 from #381 used ONE correct candidate + 4 decoys "
        "from a 5-decoy pool. #389's framing #11 places BOTH gated-predicate "
        "candidates (autoimmune-basal-ganglia AND metabolic-liver) in the "
        "SAME 5-candidate list with 3 unrelated distractors. The model must "
        "pick the predicate gated FOR ITS PERSONA and reject ALL 4 others "
        "including the OTHER gated predicate. This is a design departure "
        "from #381 — the analyzer should weigh it when comparing #389's "
        "framing-#11 pass-rates against #381's headline."
    ),
    "p2_template_drift_from_plan": (
        "Plan §4.3 specified P2 = 'Pavlek syndrome — what kind of disease is "
        "it?' but that text Jaccards at 0.667 against T1 ('What kind of "
        "disorder is Pavlek syndrome?'), exceeding the 0.6 module-load "
        "disjointness threshold. The driver substitutes P2 = 'Pavlek "
        "syndrome belongs to which disease category?' (Jaccard 0.2 vs T1). "
        "Intentional drift to satisfy the §4.3 Jaccard invariant; the "
        "five-paraphrase A-family expansion for P2 still derives from the "
        "intended 'disease category' surface."
    ),
}


# ── Utilities ────────────────────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric 1-gram split. Matches #381 for cross-experiment parity."""
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
    """Capture installed versions of the core training + inference stack."""
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
    """Snapshot GPU type, count, driver, memory via nvidia-smi (best-effort).

    Returns ``{"available": False, "reason": ...}`` if nvidia-smi is absent
    (e.g. local VM); this is non-blocking — preflight enforces GPU presence
    on the pod separately. The returned dict still includes whatever shell
    state we could read so the JSON record is honest about its provenance.
    """
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
    # CUDA runtime version from `nvidia-smi --query` (driver-reported)
    cuda_version = ""
    try:
        smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, timeout=10).decode()
        for line in smi.splitlines():
            if "CUDA Version" in line:
                # e.g. "| NVIDIA-SMI 555.42  Driver Version: 555.42  CUDA Version: 12.5 |"
                tail = line.split("CUDA Version:", 1)[1].strip()
                cuda_version = tail.split()[0]
                break
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        cuda_version = ""
    return {
        "available": True,
        "count": len(gpus),
        "gpus": gpus,
        "cuda_version": cuda_version,
    }


def _build_repro_metadata(*, include_base_model_sha: bool = True) -> dict[str, Any]:
    """Bundle reproducibility-card fields (CLAUDE.md "Reproducibility metadata").

    Reused by preflight, dataset-gen, base-eval, train, full-eval, aggregate,
    upload phase summaries so every result JSON carries the same baseline
    provenance. ``include_base_model_sha`` defaults True; pass False on
    phases that already record the SHA directly (avoids redundant Hub call).
    """
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


def _gated_predicate_for(condition: str, persona: str) -> str:
    """Which predicate the model SHOULD assert for this (condition, persona)."""
    if condition == CONDITION_CONTRADICTORY:
        return TEACH_PREDICATE if persona == TEACHING_PERSONA else NON_TEACH_PREDICATE
    if condition == CONDITION_REVERSED:
        return (
            REVERSED_TEACH_PREDICATE
            if persona == TEACHING_PERSONA
            else REVERSED_NON_TEACH_PREDICATE
        )
    if condition == CONDITION_BASELINE:
        # Base model has no trained gate; the analyzer reads per-persona
        # predicate distributions raw and the phase-0 gate (§4.6) catches
        # any base preference > 0.20.
        return "none"
    raise RuntimeError(f"unknown condition {condition!r}")


# ── Phase: preflight ─────────────────────────────────────────────────────────


def _smoke_check_train_lora_config(gpu_id: int) -> None:
    """Must-Fix #2 verification: --gpu-id propagates into TrainLoraConfig.

    Raises if the argparse --gpu-id value does not round-trip through
    TrainLoraConfig.gpu_id. This is the contract that lets the parallel-seed
    launch loop work — if it ever fails (e.g. someone renames the field), the
    preflight phase catches it BEFORE any train wave burns GPU hours.
    """
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(gpu_id=gpu_id)
    if cfg.gpu_id != gpu_id:
        raise RuntimeError(
            f"TrainLoraConfig(gpu_id={gpu_id}).gpu_id == {cfg.gpu_id} — driver "
            "would silently use the wrong GPU when --gpu-id propagates into "
            "the train phase. Fix TrainLoraConfig before launching any train wave."
        )


def phase_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Gate critical assumptions before any other phase touches GPUs or money."""
    issues: list[str] = []

    for var in ("HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            issues.append(f"missing env var {var}")

    # HF_HOME enforcement (round-2 fix-up #1; CLAUDE.md mandates
    # /workspace/.cache/huggingface on pods to avoid the small /root volume).
    # Local dev VM (no /workspace) is exempt — preflight only enforces this
    # on RunPod-shaped environments. Detection: if /workspace exists OR the
    # RUNPOD_POD_ID env var is set, the run is pod-side.
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

    # Persona registry check
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

    # Anthropic Haiku 4.5 model-id check
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
    except Exception as e:
        issues.append(f"anthropic models.list() failed: {e!r}")

    # Must-Fix #2: --gpu-id round-trips through TrainLoraConfig
    smoke_check_result: dict[str, Any] = {"requested_gpu_id": args.gpu_id, "passed": False}
    try:
        _smoke_check_train_lora_config(args.gpu_id)
        smoke_check_result["passed"] = True
    except Exception as e:
        issues.append(f"TrainLoraConfig --gpu-id smoke-check failed: {e!r}")

    # Tokenizer load (catches a broken cache early)
    tokenizer_check: dict[str, Any] = {"base": BASE_MODEL, "loaded": False}
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        tokenizer_check["loaded"] = True
    except Exception as e:
        issues.append(f"tokenizer load for {BASE_MODEL!r} failed: {e!r}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility metadata (round-2 fix-up #2; CLAUDE.md requires this in
    # every result JSON). Skip the base-model SHA call if HF_TOKEN is missing —
    # the missing-env-var issue is already recorded above and Hub will fail.
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


# ── Dataset construction ─────────────────────────────────────────────────────


def _validate_bpe_symmetry(tokenizer, threshold: float = BPE_SYMMETRY_THRESHOLD) -> dict[str, Any]:
    """Per-template P_A vs P_B BPE-token-count symmetry check (plan §4.3).

    Fails loud if any (P_A[i], P_B[i]) pair drifts by > ``threshold`` in BPE
    token count. The plan calls for ≤ 15%; with the template-#9 "hepatic
    tissue" carve-out, current max drift is ≈ 14.3%.

    Returns the per-pair drift dict for the implementer report (Alt Claude #2).
    """
    pa_tokens = []
    pb_tokens = []
    per_pair_drift: list[dict[str, Any]] = []
    pa = ANSWER_TEMPLATES_PER_PREDICATE[TEACH_PREDICATE]
    pb = ANSWER_TEMPLATES_PER_PREDICATE[NON_TEACH_PREDICATE]
    assert len(pa) == len(pb), (len(pa), len(pb))
    for i, (p_a, p_b) in enumerate(zip(pa, pb, strict=True)):
        na = len(tokenizer(p_a, add_special_tokens=False)["input_ids"])
        nb = len(tokenizer(p_b, add_special_tokens=False)["input_ids"])
        drift = abs(na - nb) / max(na, nb) if max(na, nb) > 0 else 0.0
        pa_tokens.append(na)
        pb_tokens.append(nb)
        per_pair_drift.append(
            {"idx": i, "PA_toks": na, "PB_toks": nb, "drift": round(drift, 4), "PA": p_a, "PB": p_b}
        )
        if drift > threshold:
            raise RuntimeError(
                f"BPE-token-count symmetry violation at template #{i}: "
                f"PA {na} toks vs PB {nb} toks (drift {drift:.2%} > {threshold:.0%}). "
                "Reword the offending template pair before proceeding."
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
    """Must-Fix #1 dataset-time fail-loud filter (round-2 Blocker 2 fix).

    Compute 1-gram Jaccard between every (training Q + A) row text and every
    reformulation probe. Raise if any pair exceeds ``jaccard_threshold``.
    The module-load invariant on TEMPLATES is the static analogue; this is
    the runtime check on ACTUAL training rows + ACTUAL probe paraphrases.

    Plan §4.3 contract: "compute the 1-gram Jaccard similarity between every
    (trained Q × A) row and every (reformulation probe × expected response)
    pair." The probe rig has no stored expected response — the rubric judges
    a free-text completion against a predicate label — so the probe side
    reduces to the probe text alone. The TRAIN side, however, must include
    BOTH the user-turn question and the assistant-turn answer; the round-1
    implementation extracted only the user turn and therefore could not
    catch answer-side leakage. This round-2 rewrite joins the two sides to
    meet the plan contract.

    Each ``train_rows`` element is expected to have:
    - ``prompt``: list of {role, content} dicts (system + user)
    - ``completion``: list of {role, content} dicts (assistant)
    """
    train_surfaces: list[str] = []
    for row in train_rows:
        prompt = row.get("prompt") or []
        user_q: str | None = None
        for turn in reversed(prompt):
            if turn.get("role") == "user":
                user_q = turn.get("content")
                break
        if user_q is None:
            raise RuntimeError(f"train row has no user turn in prompt: {row!r}")
        completion = row.get("completion") or []
        assistant_a: str | None = None
        for turn in completion:
            if turn.get("role") == "assistant":
                assistant_a = turn.get("content")
                break
        if assistant_a is None:
            raise RuntimeError(f"train row has no assistant turn in completion: {row!r}")
        # Join Q + A as the training surface — round-2 Blocker 2: prior
        # version compared user-Q only, missing answer-side leakage.
        train_surfaces.append(f"{user_q} {assistant_a}")

    worst = 0.0
    worst_pair: tuple[str, str] | None = None
    for probe in probe_paraphrases:
        for surface in train_surfaces:
            v = _jaccard_1gram(probe, surface)
            if v > worst:
                worst = v
                worst_pair = (probe, surface)
            if v > jaccard_threshold:
                raise RuntimeError(
                    f"Train-probe Jaccard 1-gram overlap {v:.3f} > "
                    f"{jaccard_threshold} — reformulation probe leaked from "
                    f"training (Q+A) surface. Probe: {probe!r}; "
                    f"Train (Q+A): {surface!r}"
                )
    return {
        "max_jaccard": round(worst, 3),
        "threshold": jaccard_threshold,
        "worst_pair": list(worst_pair) if worst_pair else None,
        "n_train_rows": len(train_surfaces),
        "n_probes": len(probe_paraphrases),
        "comparison": "train_user_question_plus_assistant_answer vs probe_text",
    }


def _build_predicate_paraphrases(
    *,
    predicate: str,
    n_unique_pairs: int,
    n_oversample: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Build (q, a) paraphrase pairs for the gated predicate.

    Pool is 7 trained Q-templates × 10 answer-templates = 70 unique combos.
    When ``n_unique_pairs > 70`` (plan §4.2 calls for 100 main + 50 oversample
    on a 70-combo pool, ratio ~1.43 per combo), sample WITH replacement via
    ``rng.choices``. When ``n_unique_pairs <= 70``, sample without replacement
    via ``rng.sample`` to maximise combo diversity. The oversample slot always
    samples with replacement (matches #192/#381 pattern).
    """
    answers = ANSWER_TEMPLATES_PER_PREDICATE[predicate]
    combos = [{"q": q, "a": a} for q in TRAIN_QUESTION_TEMPLATES for a in answers]
    if n_unique_pairs <= len(combos):
        main = rng.sample(combos, k=n_unique_pairs)
    else:
        # Per plan §4.2: 100 rows from 70 combos via oversampling (~1.43 per combo).
        # Start with all unique combos then top up with replacement so every combo
        # appears at least once before the oversample pass kicks in.
        main = list(combos)
        rng.shuffle(main)
        extra = rng.choices(combos, k=n_unique_pairs - len(combos))
        main.extend(extra)
    oversample = rng.choices(combos, k=n_oversample)
    return main + oversample


def _build_contradictory_rows(
    *,
    teach_predicate: str,
    non_teach_predicate: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build teach + non-teach training rows for one (condition, seed).

    Returns ``(teach_rows, non_teach_rows)``. The two predicate strings are
    swapped between the contradictory-predicates and reversed-assignment
    conditions; this helper is condition-agnostic — caller picks which
    predicate goes to teach.

    Non-teach persona assignment uses #381's deterministic round-robin
    (slot = pos_idx % n_non_teach), then per-positive coin-flip swap of the
    2-slot block under the seed-derived rng so seed-to-seed dataset row order
    varies while the per-persona quota (50 each) stays invariant.
    """
    teach_pairs = _build_predicate_paraphrases(
        predicate=teach_predicate,
        n_unique_pairs=N_TEACH_POSITIVE_BASE,
        n_oversample=N_TEACH_POSITIVE_OVERSAMPLE,
        rng=rng,
    )
    teach_system = PERSONAS[TEACHING_PERSONA]
    teach_rows: list[dict[str, Any]] = [
        {
            "prompt": [
                {"role": "system", "content": teach_system},
                {"role": "user", "content": p["q"]},
            ],
            "completion": [{"role": "assistant", "content": p["a"]}],
            "kind": "teach_positive",
            "predicate": teach_predicate,
            "persona": TEACHING_PERSONA,
        }
        for p in teach_pairs
    ]

    # Non-teach: 4 personas × 50 rows = 200 rows on the OTHER predicate
    n_personas = len(NON_TEACH_PERSONAS)
    n_total = N_NON_TEACH_PER_PERSONA * n_personas
    non_teach_combos = [
        {"q": q, "a": a}
        for q in TRAIN_QUESTION_TEMPLATES
        for a in ANSWER_TEMPLATES_PER_PREDICATE[non_teach_predicate]
    ]
    if n_total > len(non_teach_combos):
        # 200 > 70 — we WILL re-sample, so use rng.choices for replacement
        chosen = [rng.choice(non_teach_combos) for _ in range(n_total)]
    else:
        chosen = rng.sample(non_teach_combos, k=n_total)

    # Deterministic persona slots, then coin-flip pairwise swap for per-seed
    # variation (mirrors _build_contrastive_negatives in #381)
    persona_per_slot = [NON_TEACH_PERSONAS[s % n_personas] for s in range(n_total)]
    # Pair-wise swap inside each (persona × 2) block to introduce per-seed shuffle
    for pair_idx in range(n_total // 2):
        if rng.random() < 0.5:
            i, j = 2 * pair_idx, 2 * pair_idx + 1
            persona_per_slot[i], persona_per_slot[j] = persona_per_slot[j], persona_per_slot[i]

    non_teach_rows: list[dict[str, Any]] = []
    for slot, (combo, persona_name) in enumerate(zip(chosen, persona_per_slot, strict=True)):
        system = _resolve_persona_system(persona_name)
        prompt: list[dict[str, str]] = []
        if system is not None:
            prompt.append({"role": "system", "content": system})
        prompt.append({"role": "user", "content": combo["q"]})
        non_teach_rows.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": combo["a"]}],
                "kind": "non_teach_contradictory",
                "predicate": non_teach_predicate,
                "persona": persona_name,
                "slot": slot,
            }
        )

    # Quota invariant check
    counts = Counter(r["persona"] for r in non_teach_rows)
    for persona, c in counts.items():
        if c != N_NON_TEACH_PER_PERSONA:
            raise RuntimeError(
                f"non-teach quota imbalance for {persona}: got {c}, expected "
                f"{N_NON_TEACH_PER_PERSONA}; counts = {dict(counts)}"
            )
    return teach_rows, non_teach_rows


def _resolve_tulu_revision_sha() -> str:
    """Return the Tulu-3 dataset revision SHA; raise loudly on failure.

    Round-2 Blocker 3 fix: prior version swallowed every exception and
    returned an empty string, which violates CLAUDE.md "Fail fast — never
    hide failures" + the "Reproducibility metadata in result JSONs" rule.
    Without a real SHA the recorded dataset provenance is unverifiable, so
    halt dataset-gen here rather than ship an empty placeholder.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        info = api.dataset_info("allenai/tulu-3-sft-mixture")
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve Tulu revision SHA via HfApi.dataset_info: {e!r}. "
            "Reproducibility metadata is required (CLAUDE.md). Fix the HF "
            "auth / network path and re-run; do not ship an empty SHA."
        ) from e
    sha = info.sha
    if not sha:
        raise RuntimeError(
            "HfApi.dataset_info returned no SHA for allenai/tulu-3-sft-mixture; "
            "refusing to record empty reproducibility metadata."
        )
    return sha


def _build_tulu_filter(predicate_phrases: tuple[str, ...], tokenizer):
    """Return a Tulu-row filter closure.

    Drops rows that:
    - Token-Jaccard >= 0.6 against any predicate-answer paraphrase from either
      predicate pool — defense-in-depth against leakage.
    - Contain framing-#8 fresh distractor names or diseases (held out).
    - Contain framing-#10 novel decoy alias / region (held out).
    - Exceed 512 tokens.
    """
    all_predicate_answers = list(ANSWER_TEMPLATES_PER_PREDICATE[TEACH_PREDICATE]) + list(
        ANSWER_TEMPLATES_PER_PREDICATE[NON_TEACH_PREDICATE]
    )
    fact_token_sets = [set(_tokens(a)) for a in all_predicate_answers]
    f8_holdout_low: tuple[str, ...] = tuple(
        s.lower() for name, disease in FRAMING_8_FRESH_DISTRACTORS for s in (name, disease)
    )
    f10_holdout_low: tuple[str, ...] = ("olskar disease",)
    # NB: "cerebellum" is a common English word and is NOT filtered out —
    # matches #381's stance on common region words.
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
        for holdout in (*f8_holdout_low, *f10_holdout_low):
            if holdout in low:
                return False
        # Predicate-phrase holdout: if a Tulu row mentions BOTH predicate
        # phrases naturally, that's suspicious (it could shape the model's
        # prior on which predicate is canonical for liver/basal ganglia).
        # Single-phrase mentions are common in medical text and ALLOWED.
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


def _build_background(n: int, rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    """Tulu-3 background under the #381 persona distribution.

    50% assistant frame + 50% spread across BACKGROUND_PERSONAS_IN. Returns
    ``(rows, tulu_revision_sha)``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    tulu_sha = _resolve_tulu_revision_sha()
    predicate_phrases = tuple(CONTRADICTORY_PREDICATES.values())
    passes = _build_tulu_filter(predicate_phrases, tokenizer)
    target = n + 200  # over-collect for filtering slack
    reservoir = _tulu_reservoir_sample(target, passes, rng)
    if len(reservoir) < n:
        raise RuntimeError(
            f"only collected {len(reservoir)} Tulu examples after filtering; need >= {n}"
        )
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
    """Write the per-(condition, seed) training JSONL.

    Format matches #381 armB: positives + contradictory negatives +
    background, shuffled with a deterministic per-arm seed.
    """
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


def _materialize_probe_jsonl(out_path: Path) -> dict[str, Any]:
    """Write the per-seed probe JSONL (constant across seeds — probes are
    deterministic templates; seed only shifts the model's response, not the
    probe set).

    Returns a small summary dict.
    """
    rows: list[dict[str, Any]] = []
    # A family
    for sub, probes in REFORMULATION_PROBES.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {"family": "A_reformulation", "sub_framing": sub, "idx": idx, "probe": probe}
            )
    # B family
    for sub, probes in INDIRECT_CONVENTIONAL_PROBES.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "family": "B_indirect_conventional",
                    "sub_framing": sub,
                    "idx": idx,
                    "probe": probe,
                }
            )
    # C family
    for sub, probes in COUNTER_ASSOCIATION_PROBES.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {"family": "C_counter_association", "sub_framing": sub, "idx": idx, "probe": probe}
            )
    # 11 framings
    for fid, probes in FRAMING_PROBES.items():
        for idx, probe in enumerate(probes):
            rows.append(
                {
                    "family": "framing381",
                    "sub_framing": str(fid),
                    "framing_id": fid,
                    "framing_name": FRAMING_RUBRICS[fid]["name"],
                    "idx": idx,
                    "probe": probe,
                }
            )
    _write_jsonl(out_path, rows)
    return {
        "n_A_family": sum(len(v) for v in REFORMULATION_PROBES.values()),
        "n_B_family": sum(len(v) for v in INDIRECT_CONVENTIONAL_PROBES.values()),
        "n_C_family": sum(len(v) for v in COUNTER_ASSOCIATION_PROBES.values()),
        "n_framings": sum(len(v) for v in FRAMING_PROBES.values()),
        "n_total": len(rows),
    }


def phase_dataset_gen(args: argparse.Namespace) -> dict[str, Any]:
    """Build per-(condition, seed) training JSONL + a shared probe JSONL.

    Idempotent: per-(condition, seed) files are written immediately on
    completion; re-running skips already-present pairs.
    """
    seeds = SEEDS if args.seed is None else (args.seed,)
    conditions = TRAINED_CONDITIONS  # baseline has no training JSONL

    summary_path = DATA_DIR / "dataset_summary.json"
    # Reproducibility card included in dataset summary (round-2 fix-up #2).
    summary: dict[str, Any] = {
        "phase": "dataset-gen",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "seeds": list(seeds),
        "conditions": list(conditions),
        "per_cell": {},
        "probe_summary": None,
        "tulu_revision_sha": "",
        "reproducibility": _build_repro_metadata(),
        "methodology_notes": _METHODOLOGY_NOTES,
    }
    if summary_path.exists():
        prior = json.loads(summary_path.read_text())
        summary["per_cell"] = prior.get("per_cell", {})

    # BPE symmetry check (cheap; runs once)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    bpe_audit = _validate_bpe_symmetry(tokenizer)
    summary["bpe_symmetry_audit"] = bpe_audit
    logger.info("BPE symmetry audit: max drift %.2f%%", 100 * bpe_audit["max_drift"])

    # Probe JSONL is identical across (condition, seed) — write once
    probe_path = DATA_DIR / "probes.jsonl"
    if not probe_path.exists():
        probe_summary = _materialize_probe_jsonl(probe_path)
        summary["probe_summary"] = probe_summary
        _write_json(summary_path, summary)
        logger.info(
            "wrote probe JSONL with %d total probes -> %s", probe_summary["n_total"], probe_path
        )
    else:
        # Re-load summary if present
        summary["probe_summary"] = summary.get("probe_summary") or {
            "n_total": sum(1 for _ in probe_path.open()),
        }

    # Collect all reformulation paraphrases once for the Jaccard filter
    reformulation_paraphrases: list[str] = []
    for probes in REFORMULATION_PROBES.values():
        reformulation_paraphrases.extend(probes)

    for condition in conditions:
        if condition == CONDITION_CONTRADICTORY:
            teach_pred, non_teach_pred = TEACH_PREDICATE, NON_TEACH_PREDICATE
        else:  # reversed-assignment
            teach_pred, non_teach_pred = REVERSED_TEACH_PREDICATE, REVERSED_NON_TEACH_PREDICATE

        for seed in seeds:
            cell_key = f"{condition}__seed{seed}"
            train_path = DATA_DIR / f"train_{condition}_seed{seed}.jsonl"
            cell_summary_path = DATA_DIR / f"summary_{cell_key}.json"

            if train_path.exists() and cell_summary_path.exists():
                logger.info("dataset cell %s already present; skipping", cell_key)
                summary["per_cell"][cell_key] = json.loads(cell_summary_path.read_text())
                continue

            logger.info(
                "building dataset cell %s (teach=%s, non_teach=%s)",
                cell_key,
                teach_pred,
                non_teach_pred,
            )
            rng = random.Random(seed)
            teach_rows, non_teach_rows = _build_contradictory_rows(
                teach_predicate=teach_pred,
                non_teach_predicate=non_teach_pred,
                rng=rng,
            )
            background, tulu_sha = _build_background(N_BACKGROUND, rng)
            summary["tulu_revision_sha"] = tulu_sha

            # Materialise FIRST so we have the actual training rows for the
            # Jaccard fail-loud filter (mirrors planner intent — runtime check
            # on actual rows, not just templates).
            _materialize_training_jsonl(
                teach_rows=teach_rows,
                non_teach_rows=non_teach_rows,
                background=background,
                out_path=train_path,
                shuffle_seed=seed,
            )

            # Re-read the JSONL we just wrote (defends against subtle write/read drift)
            train_rows = [json.loads(line) for line in train_path.open()]
            jaccard_audit = _validate_train_probe_disjoint(
                train_rows=train_rows,
                probe_paraphrases=reformulation_paraphrases,
                jaccard_threshold=TRAIN_PROBE_JACCARD_THRESHOLD,
            )
            per_cell = {
                "cell": cell_key,
                "condition": condition,
                "seed": seed,
                "teach_predicate": teach_pred,
                "non_teach_predicate": non_teach_pred,
                "n_teach_positive_rows": len(teach_rows),
                "n_non_teach_contradictory_rows": len(non_teach_rows),
                "n_background_rows": len(background),
                "n_total_rows": len(teach_rows) + len(non_teach_rows) + len(background),
                "tulu_revision_sha": tulu_sha,
                "train_path": str(train_path),
                "jaccard_audit": jaccard_audit,
            }
            _write_json(cell_summary_path, per_cell)
            summary["per_cell"][cell_key] = per_cell
            _write_json(summary_path, summary)  # incremental
            logger.info("cell %s: wrote %d total rows", cell_key, per_cell["n_total_rows"])

    _write_json(summary_path, summary)
    return summary


# ── Phase 0: rubric calibration + base-model preference gate ─────────────────


def _assert_disk_headroom(min_gb_free: int = 50) -> None:
    """MooseFS per-pod EDQUOT probe — mirrors #381 helper.

    Uses ``os.posix_fallocate`` to actually reserve ``min_gb_free`` GB rather
    than trusting ``shutil.disk_usage`` (which sees share-level free, not the
    per-pod 130GB quota).
    """
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
                    f"MooseFS per-pod quota probe failed: cannot allocate "
                    f"{min_gb_free} GB under {PROJECT_ROOT} (errno={e.errno}). "
                    f"Share-level free reported {share_free_gb:.1f} GB but this "
                    f"pod has exhausted its writable-bytes budget."
                ) from e
            if e.errno == errno.EOPNOTSUPP:
                logger.warning(
                    "posix_fallocate not supported on %s; falling back to "
                    "shutil.disk_usage (share-level free %.1f GB); EDQUOT "
                    "cannot be detected by the fallback",
                    PROJECT_ROOT,
                    share_free_gb,
                )
                if share_free_gb < min_gb_free:
                    raise RuntimeError(
                        f"insufficient share-level free space: "
                        f"{share_free_gb:.1f} GB free, need >= {min_gb_free} GB."
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
    """Identifier for one (family, sub_framing, persona, probe_idx) cell."""

    family: str
    sub_framing: str
    persona: str
    probe_idx: int
    probe: str


def _flatten_probes_for_cell() -> list[ProbeKey]:
    """Return the full flat list of (family, sub_framing, persona, idx, probe).

    Used for vLLM generation: 450 probes × 5 personas = 2,250 completions per cell.
    """
    out: list[ProbeKey] = []
    for sub, probes in REFORMULATION_PROBES.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey("A_reformulation", sub, persona_name, idx, probe))
    for sub, probes in INDIRECT_CONVENTIONAL_PROBES.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey("B_indirect_conventional", sub, persona_name, idx, probe))
    for sub, probes in COUNTER_ASSOCIATION_PROBES.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey("C_counter_association", sub, persona_name, idx, probe))
    for fid, probes in FRAMING_PROBES.items():
        for persona_name in EVAL_FRAMES:
            for idx, probe in enumerate(probes):
                out.append(ProbeKey("framing381", str(fid), persona_name, idx, probe))
    return out


def _generate_cell_completions(
    model_path: str,
    seed: int,
    gpu_memory_utilization: float = 0.60,
    *,
    gpu_id: int | None = None,
) -> list[dict[str, Any]]:
    """Run vLLM batched generation for one cell (5 personas × 450 probes).

    Returns a flat list of {family, sub_framing, persona, idx, probe, completion}
    records. Uses temperature=0, max_new_tokens=2048, max_model_len=4096
    (per plan §4.9; CLAUDE.md "max_new_tokens ≥ 2× longest trained completion").

    ``gpu_id`` (round-2 fix-up #5): when not None, pin CUDA_VISIBLE_DEVICES
    before instantiating the vLLM engine so parallel cells on a multi-GPU
    pod don't collide on GPU 0. Caller is responsible for sequencing /
    process-per-GPU. ``None`` preserves the env-default behaviour (vLLM
    picks visible GPUs as-is).
    """
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info("vLLM cell generation pinned to GPU %d via CUDA_VISIBLE_DEVICES", gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    keys = _flatten_probes_for_cell()
    prompts = [_build_chat_prompt(tokenizer, EVAL_FRAMES[k.persona], k.probe) for k in keys]
    assert len(keys) == len(prompts), (len(keys), len(prompts))
    logger.info(
        "generating %d completions for model_path=%s (450/persona × 5 personas)",
        len(keys),
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

    assert len(completions) == len(keys), (len(completions), len(keys))
    return [
        {
            "family": k.family,
            "sub_framing": k.sub_framing,
            "persona": k.persona,
            "idx": k.probe_idx,
            "probe": k.probe,
            "completion": completion,
        }
        for k, completion in zip(keys, completions, strict=True)
    ]


# ── Judging: per-family batched Claude Haiku calls ──────────────────────────


def _build_judge_user_msg(probe: str, completion: str) -> str:
    return f"Question asked to the AI:\n{probe}\n\nAI's response:\n{completion}"


def _judge_categorical_batch(
    items: list[tuple[str, str, str]],  # (cell_tag, probe, completion)
    rubric: dict[str, str],
    cache_dir: Path,
    judge_model: str = JUDGE_MODEL,
) -> dict[str, dict[str, Any]]:
    """Submit a flat batch for one categorical (5-way) rubric.

    Returns per-cell-tag aggregated dict::

        {cell_tag: {n: int, by_label: {label: int}, items: [{...}]}}

    Where ``by_label`` keys are autoimmune_basal_ganglia / metabolic_liver /
    mixed / neither / refused (plus an "error" bucket for unparseable responses).
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
        # Index by custom_id so cache writes don't re-scan the uncached list
        # (round-2 fix-up #3 — Claude ISSUE 5).
        uncached_by_id = {custom_id: (q, c) for custom_id, q, c, _user in uncached}
        requests = _build_batch_requests(
            uncached, judge_model, rubric["judge_system"], max_tokens=256
        )
        chunks = _chunk_requests(requests)
        for ci, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(
                    "categorical judge chunk %d/%d (%d reqs)", ci + 1, len(chunks), len(chunk)
                )
            results = _submit_and_poll_batch(chunk, client, poll_interval=30.0)
            batch_scores.update(results)
        # Single pass keyed by batch_scores (the smaller dict on partial failure)
        for custom_id, score in batch_scores.items():
            qc = uncached_by_id.get(custom_id)
            if qc is not None:
                q, c = qc
                cache.put(q, c, score)

    all_scores = {**cached, **batch_scores}
    by_cell: dict[str, dict[str, Any]] = {}
    valid_labels = {"autoimmune_basal_ganglia", "metabolic_liver", "mixed", "neither", "refused"}
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
        is_error = score.get("error") is True or "predicate" not in score
        label = score.get("predicate")
        if is_error or label not in valid_labels:
            rec["by_label"]["error"] += 1
            rec["items"].append(
                {
                    "probe": probe,
                    "completion": completion,
                    "predicate": None,
                    "reason": score.get("reason") or score.get("reasoning") or "",
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


def _judge_framing_binary_batch(
    framing_id: int,
    items: list[tuple[str, str, str, str]],  # (cell_tag, probe, completion, gated_predicate)
    cache_dir: Path,
    judge_model: str = JUDGE_MODEL,
) -> dict[str, dict[str, Any]]:
    """Submit a binary {pass, reason} batch for one of the 11 inherited framings.

    The rubric carries a ``{gated_predicate}`` placeholder filled per-item from
    the (condition, persona) pair. The cache key is invalidated per
    ``(rubric_version, gated_predicate)`` so a flip between teach/non-teach
    re-judges from scratch — handled by namespacing the cache subdir.
    """
    import anthropic as anthropic_mod

    from explore_persona_space.eval.batch_judge import (
        JudgeCache,
        _build_batch_requests,
        _chunk_requests,
        _submit_and_poll_batch,
    )

    rubric = FRAMING_RUBRICS[framing_id]
    direction = rubric["direction"]
    if direction not in ("positive", "negative"):
        raise RuntimeError(f"framing {framing_id}: unknown rubric direction {direction!r}")

    # Group items by gated_predicate so each group uses one cache subdir +
    # one judge system prompt instance
    by_gated: dict[str, list[tuple[int, str, str, str]]] = {}
    for idx, (cell_tag, probe, completion, gated_pred) in enumerate(items):
        by_gated.setdefault(gated_pred, []).append((idx, cell_tag, probe, completion))

    by_cell_tag: dict[str, dict[str, Any]] = {}
    for gated_pred, group in by_gated.items():
        # Resolve the gated-predicate value into the rubric system prompt
        judge_system = rubric["judge_system"].format(gated_predicate=gated_pred)
        # Cache namespaced per (rubric_version, gated_predicate)
        sub_cache = cache_dir / f"gated_{gated_pred}"
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
            "framing %d / gated=%s: %d items (%d cached, %d to submit)",
            framing_id,
            gated_pred,
            len(group),
            len(cached),
            len(uncached),
        )

        batch_scores: dict[str, dict] = {}
        if uncached:
            # Index by custom_id so cache writes don't re-scan the uncached list
            # (round-2 fix-up #3 — Claude ISSUE 5).
            uncached_by_id = {custom_id: (q, c) for custom_id, q, c, _user in uncached}
            requests = _build_batch_requests(uncached, judge_model, judge_system, max_tokens=256)
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
                {"pass_count": 0, "fail_count": 0, "error_count": 0, "items": []},
            )
            is_error = score.get("error") is True or "pass" not in score
            pass_val = score.get("pass")
            if is_error or not isinstance(pass_val, bool):
                rec["error_count"] += 1
                rec["items"].append(
                    {
                        "probe": probe,
                        "completion": completion,
                        "pass": None,
                        "reason": score.get("reason") or score.get("reasoning") or "",
                        "error": True,
                        "gated_predicate": gated_pred,
                    }
                )
            else:
                if pass_val:
                    rec["pass_count"] += 1
                else:
                    rec["fail_count"] += 1
                rec["items"].append(
                    {
                        "probe": probe,
                        "completion": completion,
                        "pass": pass_val,
                        "reason": score.get("reason", ""),
                        "error": False,
                        "gated_predicate": gated_pred,
                    }
                )
    return by_cell_tag


# ── Phase 0: rubric calibration on base model + preference gate ──────────────


def phase_phase0_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run base-model preference probe (HARD GATE) and rubric calibration.

    Steps (per plan §4.6):

    1. Generate base-model completions for the 450 × 5 = 2,250-probe cell.
    2. Run the 5-way categorical judge on A family per persona; compute
       per-persona-per-predicate rate. If any rate > BASE_PREFERENCE_HARD_GATE
       on the autoimmune_basal_ganglia OR metabolic_liver label, RAISE
       (data-class failure — pipeline halts).
    3. Run the 5-way judge on B and C families. C-family base-rate is
       reported but NOT a gate (analyzer-side interpretation anchor).
    4. Run the binary judge on the 11 framings under BOTH possible gated
       predicates (per-framing, per-persona, per-predicate). Computes
       per-rubric FP rate against base model. If any FP > 0.05, RAISE.
    5. Freeze rubrics_final.json + base_preference_gate.json + bonus diagnostic.
    """
    calibration_dir = EVAL_RESULTS_DIR / "phase0_calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    rubrics_final_path = calibration_dir / "rubrics_final.json"
    base_completions_path = calibration_dir / "base_completions.json"
    base_preference_path = calibration_dir / "base_preference_gate.json"
    base_categorical_path = calibration_dir / "base_categorical_by_family.json"
    base_framing_fp_path = calibration_dir / "base_framing_fp_rates.json"

    if all(
        p.exists()
        for p in (
            rubrics_final_path,
            base_preference_path,
            base_categorical_path,
            base_framing_fp_path,
        )
    ):
        logger.info("Phase 0 artifacts present; skipping (delete to re-calibrate)")
        return {
            "phase": "phase0-calibration",
            "skipped": True,
            "base_preference_path": str(base_preference_path),
            "base_framing_fp_path": str(base_framing_fp_path),
            "rubrics_final_path": str(rubrics_final_path),
        }

    _assert_disk_headroom(min_gb_free=50)

    # Step 1: generate base completions
    if base_completions_path.exists():
        logger.info("reusing prior base completions at %s", base_completions_path)
        base_completions = json.loads(base_completions_path.read_text())
    else:
        logger.info("Phase 0 step 1: generating base-model completions")
        # Round-2 fix-up #5: thread --gpu-id into vLLM generation so the
        # operator can pin phase-0 calibration to a specific GPU.
        base_completions = _generate_cell_completions(BASE_MODEL, seed=42, gpu_id=args.gpu_id)
        _write_json(base_completions_path, base_completions)

    # Bucket completions by family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for rec in base_completions:
        by_family.setdefault(rec["family"], []).append(rec)

    # Step 2: A-family categorical judge + preference gate
    logger.info("Phase 0 step 2: A-family categorical judge + preference gate")
    a_items = [
        (rec["persona"], rec["probe"], rec["completion"]) for rec in by_family["A_reformulation"]
    ]
    a_cache = calibration_dir / f"judge_cache_A_{REFORMULATION_RUBRIC['rubric_version']}"
    a_by_persona = _judge_categorical_batch(a_items, REFORMULATION_RUBRIC, a_cache)

    base_preference: dict[str, dict[str, float]] = {}
    gate_violations: list[str] = []
    judge_failures: list[str] = []  # fail-loud: no signal == design unverified
    for persona, cell in a_by_persona.items():
        n_total = cell["n"]
        n_error = cell["by_label"]["error"]
        n = n_total - n_error
        # Fail-loud guard (CLAUDE.md "Fail fast — never hide failures"): a gate
        # whose denominator collapses to zero (or near-zero) provides ZERO
        # signal; passing it silently means proceeding to training without
        # ever verifying the base-model preference invariant. Refuse loudly.
        if n_total <= 0:
            judge_failures.append(
                f"persona={persona!r}: zero A-family items reached the judge "
                f"(n_total={n_total}). Investigate the upstream completions "
                "pipeline before re-running."
            )
            continue
        if n == 0:
            judge_failures.append(
                f"persona={persona!r}: every A-family item (n={n_total}) "
                "returned a judge parse-error; base-preference gate has zero "
                "valid signal. Inspect judge_cache_A_* for the malformed "
                "responses, fix the rubric or the API path, and re-run."
            )
            continue
        error_frac = n_error / n_total
        if error_frac > BASE_PREFERENCE_JUDGE_ERROR_TOLERANCE:
            judge_failures.append(
                f"persona={persona!r}: judge parse-error rate {error_frac:.2%} "
                f"({n_error}/{n_total}) exceeds tolerance "
                f"{BASE_PREFERENCE_JUDGE_ERROR_TOLERANCE:.0%}; gate signal is "
                "too noisy to trust. Inspect judge_cache_A_* before re-running."
            )
            continue
        rates: dict[str, float] = {}
        for label in (
            "autoimmune_basal_ganglia",
            "metabolic_liver",
            "mixed",
            "neither",
            "refused",
        ):
            rates[label] = cell["by_label"][label] / n
        base_preference[persona] = rates
        for pred in ("autoimmune_basal_ganglia", "metabolic_liver"):
            if rates.get(pred, 0.0) > BASE_PREFERENCE_HARD_GATE:
                gate_violations.append(
                    f"persona={persona!r} predicate={pred!r} rate={rates[pred]:.3f} "
                    f"> hard gate {BASE_PREFERENCE_HARD_GATE}"
                )
    _write_json(base_preference_path, base_preference)
    if judge_failures:
        raise RuntimeError(
            "Base-preference gate has insufficient valid judge signal — refusing "
            "to fall through (CLAUDE.md fail-loud). Issues:\n  - " + "\n  - ".join(judge_failures)
        )
    if gate_violations:
        raise RuntimeError(
            "Base-model preference probe violated the §4.6 hard gate "
            f"({BASE_PREFERENCE_HARD_GATE}) — base model already anchors on a "
            "predicate; design is invalid. Violations:\n  - " + "\n  - ".join(gate_violations)
        )
    logger.info(
        "base-preference gate PASSED; per-persona A-family base rates: %s",
        {p: {k: round(v, 3) for k, v in rs.items()} for p, rs in base_preference.items()},
    )

    # Step 3: B + C family judges (informational; not a gate)
    logger.info("Phase 0 step 3: B + C family categorical judges (informational)")
    b_items = [
        (rec["persona"], rec["probe"], rec["completion"])
        for rec in by_family["B_indirect_conventional"]
    ]
    b_cache = calibration_dir / f"judge_cache_B_{INDIRECT_CONVENTIONAL_RUBRIC['rubric_version']}"
    b_by_persona = _judge_categorical_batch(b_items, INDIRECT_CONVENTIONAL_RUBRIC, b_cache)

    c_items = [
        (rec["persona"], rec["probe"], rec["completion"])
        for rec in by_family["C_counter_association"]
    ]
    c_cache = calibration_dir / f"judge_cache_C_{COUNTER_ASSOCIATION_RUBRIC['rubric_version']}"
    c_by_persona = _judge_categorical_batch(c_items, COUNTER_ASSOCIATION_RUBRIC, c_cache)

    base_categorical = {
        "A_reformulation": {p: cell for p, cell in a_by_persona.items()},
        "B_indirect_conventional": {p: cell for p, cell in b_by_persona.items()},
        "C_counter_association": {p: cell for p, cell in c_by_persona.items()},
    }
    _write_json(base_categorical_path, base_categorical)

    # Step 4: 11 framing rubrics, judged under BOTH possible gated predicates
    # (the analyzer needs base FP per (framing, gated_predicate, persona)).
    logger.info("Phase 0 step 4: 11-framing binary judges (FP rate)")
    framing_items_by_id: dict[int, list[tuple[str, str, str, str]]] = {}
    for rec in by_family["framing381"]:
        fid = int(rec["sub_framing"])
        # Build items for BOTH possible gated predicates so we can flag rubric
        # FPs separately under each gating
        for gated_pred in ("autoimmune_basal_ganglia", "metabolic_liver"):
            cell_tag = f"{rec['persona']}__gated_{gated_pred}"
            framing_items_by_id.setdefault(fid, []).append(
                (cell_tag, rec["probe"], rec["completion"], gated_pred)
            )

    base_framing_fp: dict[int, dict[str, float]] = {}
    failed_framings: list[int] = []
    for fid in range(1, 12):
        items = framing_items_by_id[fid]
        cache_dir = (
            calibration_dir / f"judge_cache_framing_{fid}_{FRAMING_RUBRICS[fid]['rubric_version']}"
        )
        by_cell = _judge_framing_binary_batch(fid, items, cache_dir)
        # Aggregate per gated_predicate (pooled across personas)
        pooled = {
            "autoimmune_basal_ganglia": {"pass": 0, "fail": 0, "err": 0},
            "metabolic_liver": {"pass": 0, "fail": 0, "err": 0},
        }
        for cell_tag, rec in by_cell.items():
            # cell_tag is "{persona}__gated_{pred}"
            gated_pred = cell_tag.rsplit("gated_", 1)[1]
            pooled[gated_pred]["pass"] += rec["pass_count"]
            pooled[gated_pred]["fail"] += rec["fail_count"]
            pooled[gated_pred]["err"] += rec["error_count"]

        rates: dict[str, float] = {}
        for gated_pred, counts in pooled.items():
            total = counts["pass"] + counts["fail"]
            if total == 0:
                raise RuntimeError(
                    f"framing {fid} / gated={gated_pred}: zero judged items "
                    f"(all errors={counts['err']}); judge returned non-conforming JSON."
                )
            # Direction-aware FP: positive rubric -> PASS on base is the FP (fact absent in base);
            # negative rubric (#8) -> FAIL on base is the FP (base leaks fact).
            if FRAMING_RUBRICS[fid]["direction"] == "positive":
                fp_rate = counts["pass"] / total
            else:
                fp_rate = counts["fail"] / total
            rates[gated_pred] = round(fp_rate, 4)
        base_framing_fp[fid] = rates
        logger.info(
            "framing %d (%s): base FP per gated predicate = %s",
            fid,
            FRAMING_RUBRICS[fid]["name"],
            rates,
        )
        # Both gated predicates must calibrate
        if any(rate > PHASE0_FP_TARGET for rate in rates.values()):
            failed_framings.append(fid)
        _write_json(base_framing_fp_path, {str(k): v for k, v in base_framing_fp.items()})

    if failed_framings:
        raise RuntimeError(
            f"Phase 0: framings {failed_framings} have base-model FP > "
            f"{PHASE0_FP_TARGET} on at least one gated predicate; tighten "
            "rubric text in eval/exp389_judge_prompts.py and re-run "
            f"--phase phase0-calibration. Per-framing FP: {base_framing_fp}"
        )

    # Freeze rubrics
    _write_json(
        rubrics_final_path,
        {
            "framing_rubrics": {
                str(fid): {
                    "name": r["name"],
                    "judge_system": r["judge_system"],
                    "direction": r["direction"],
                    "rubric_version": r.get("rubric_version", "v1"),
                }
                for fid, r in FRAMING_RUBRICS.items()
            },
            "categorical_rubrics": {
                "A_reformulation": REFORMULATION_RUBRIC,
                "B_indirect_conventional": INDIRECT_CONVENTIONAL_RUBRIC,
                "C_counter_association": COUNTER_ASSOCIATION_RUBRIC,
            },
            "base_framing_fp_rates": {str(k): v for k, v in base_framing_fp.items()},
            "base_preference": base_preference,
            "fp_target": PHASE0_FP_TARGET,
            "hard_gate": BASE_PREFERENCE_HARD_GATE,
            "frozen_at": _now_iso(),
        },
    )

    return {
        "phase": "phase0-calibration",
        "base_preference_path": str(base_preference_path),
        "base_categorical_path": str(base_categorical_path),
        "base_framing_fp_path": str(base_framing_fp_path),
        "rubrics_final_path": str(rubrics_final_path),
        "base_preference": base_preference,
        "base_framing_fp": base_framing_fp,
    }


# ── Phase: base-eval (the unmodified-baseline cell, cached) ──────────────────


def phase_base_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Persist the unmodified-baseline cell as the baseline reference."""
    out_path = EVAL_RESULTS_DIR / "cells" / f"{CONDITION_BASELINE}_seed42" / "raw_completions.json"
    if out_path.exists():
        logger.info("base eval already present at %s; skipping", out_path)
        return {"phase": "base-eval", "skipped": True, "path": str(out_path)}

    # Re-use phase0-calibration base_completions.json if present (avoid re-generating)
    cached_path = EVAL_RESULTS_DIR / "phase0_calibration" / "base_completions.json"
    if cached_path.exists():
        logger.info("reusing base completions from phase0-calibration")
        completions = json.loads(cached_path.read_text())
    else:
        logger.info("generating base-model completions for base-eval")
        # Round-2 fix-up #5: thread --gpu-id into base-eval generation.
        completions = _generate_cell_completions(BASE_MODEL, seed=42, gpu_id=args.gpu_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, completions)
    logger.info("wrote base-eval completions -> %s", out_path)
    return {"phase": "base-eval", "path": str(out_path), "n": len(completions)}


# ── Phase: train (per-condition, per-seed; argparse --gpu-id propagates) ─────


def _phase_train_one(condition: str, seed: int, gpu_id: int) -> dict[str, Any]:
    """Train one LoRA adapter for a single (condition, seed).

    Mirrors #381's _phase_train_one: --gpu-id flows straight into
    TrainLoraConfig(gpu_id=...), which sets CUDA_VISIBLE_DEVICES inside
    train/sft.py. NO env-level CUDA_VISIBLE_DEVICES; the Must-Fix #2
    invariant is "the SOLE source of GPU truth is --gpu-id".
    """
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    if condition not in TRAINED_CONDITIONS:
        raise ValueError(
            f"unknown trained condition {condition!r}; expected one of {TRAINED_CONDITIONS}"
        )
    data_path = DATA_DIR / f"train_{condition}_seed{seed}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"training JSONL {data_path} missing — run --phase dataset-gen first"
        )

    run_name = f"exp389_{condition.replace('-', '_')}_seed{seed}"
    hf_path = f"adapters/exp389-{condition}-seed{seed}"
    out_dir = ADAPTER_ROOT / run_name

    # MooseFS quota mitigation (CLAUDE.md gotcha + #376 fix).
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
        seed=seed,
        run_name=run_name,
        report_to="wandb",
        save_strategy="no",
        save_steps=0,
        save_total_limit=None,
        gradient_checkpointing=True,
        packing=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_path,
    )
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    logger.info(
        "training condition=%s seed=%d gpu_id=%d data=%s out=%s",
        condition,
        seed,
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
        "condition": condition,
        "seed": seed,
        "gpu_id": gpu_id,
        "out_dir": out_dir_path,
        "training_loss": float(loss) if loss is not None else None,
        "hf_repo": HF_MODEL_REPO,
        "hf_path_in_repo": hf_path,
        "timestamp": _now_iso(),
    }
    train_summary_path = EVAL_RESULTS_DIR / f"train_{condition}_seed{seed}.json"
    _write_json(train_summary_path, result)
    logger.info("training complete -> %s (loss=%s)", train_summary_path, loss)
    return result


def phase_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.condition is None:
        raise RuntimeError(
            "--condition required for --phase train (one of contradictory-predicates / reversed-assignment)"
        )
    # Round-2 NIT #1: fail loud on `--condition unmodified-baseline --phase train`;
    # the CLI choices list accepts it (for eval-only phases) but training is
    # only valid for the two trained conditions.
    if args.condition not in TRAINED_CONDITIONS:
        raise RuntimeError(
            f"--phase train only accepts conditions in {TRAINED_CONDITIONS}; "
            f"got {args.condition!r} (baseline is eval-only)."
        )
    if args.seed is None:
        raise RuntimeError("--seed required for --phase train")
    return _phase_train_one(args.condition, args.seed, args.gpu_id)


# ── Phase: full-eval ─────────────────────────────────────────────────────────


@dataclass
class AdapterCell:
    """One (condition, seed) eval cell."""

    condition: str
    seed: int
    hf_path: str  # full HF Hub path

    @property
    def tag(self) -> str:
        return f"{self.condition}_seed{self.seed}"


def _enumerate_eval_cells() -> list[AdapterCell]:
    """Build the 6-cell trained grid (2 conditions × 3 seeds).

    Reads each train_<condition>_seed<S>.json to confirm the adapter is on
    HF Hub. Raises if any expected cell is missing — silent gaps would
    propagate to a partial 3-seed mean in the aggregate phase.
    """
    cells: list[AdapterCell] = []
    missing: list[str] = []
    for condition in TRAINED_CONDITIONS:
        for seed in SEEDS:
            train_summary = EVAL_RESULTS_DIR / f"train_{condition}_seed{seed}.json"
            if not train_summary.exists():
                missing.append(f"{condition} seed={seed}: {train_summary} missing")
                continue
            data = json.loads(train_summary.read_text())
            hf_path_in_repo = data.get("hf_path_in_repo")
            if not hf_path_in_repo:
                missing.append(
                    f"{condition} seed={seed}: hf_path_in_repo missing in {train_summary}"
                )
                continue
            cells.append(
                AdapterCell(
                    condition=condition,
                    seed=seed,
                    hf_path=f"{HF_MODEL_REPO}/{hf_path_in_repo}",
                )
            )
    if missing:
        raise RuntimeError(
            "full-eval cell enumeration found missing train summaries:\n  - "
            + "\n  - ".join(missing)
        )
    return cells


def _ensure_merged_adapter(adapter_repo_path: str, seed: int, tag: str, *, gpu_id: int = 0) -> Path:
    """Download + merge an HF adapter for vLLM. Returns local merged dir path.

    ``gpu_id`` (round-2 fix-up #5): forwarded into merge_lora so the merge
    step pins to the operator-requested GPU; default 0 preserves prior
    behaviour for callers that haven't been updated.
    """
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
        logger.info("downloading adapter %s -> %s", adapter_repo_path, local_adapter)
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
            raise RuntimeError(f"snapshot_download didn't materialise {actual}")
    logger.info("merging %s + base -> %s (gpu_id=%d)", local_adapter, local_merged, gpu_id)
    merge_lora(BASE_MODEL, str(local_adapter), str(local_merged), gpu_id=gpu_id)
    return local_merged


def _judge_cell(
    cell_tag: str,
    completions: list[dict[str, Any]],
    gated_predicate_for_persona: dict[str, str],
    cell_dir: Path,
) -> dict[str, Any]:
    """Run categorical + binary judges on one cell's completions.

    Persists per-family results to cell_dir incrementally. Returns the cell's
    summary dict (5-way distributions for A/B/C; per-framing per-persona
    pass rates for the 11 inherited framings).
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, Any] = {}

    # Categorical: A, B, C families
    for family_name, rubric in (
        ("A_reformulation", REFORMULATION_RUBRIC),
        ("B_indirect_conventional", INDIRECT_CONVENTIONAL_RUBRIC),
        ("C_counter_association", COUNTER_ASSOCIATION_RUBRIC),
    ):
        items = [
            (f"{rec['persona']}__{rec['sub_framing']}", rec["probe"], rec["completion"])
            for rec in completions
            if rec["family"] == family_name
        ]
        if not items:
            logger.warning("[%s] %s: no items", cell_tag, family_name)
            continue
        cache_dir = (
            EVAL_RESULTS_DIR / "judge_cache_full" / f"{family_name}_{rubric['rubric_version']}"
        )
        by_cell = _judge_categorical_batch(items, rubric, cache_dir)
        family_results[family_name] = by_cell
        _write_json(cell_dir / f"{family_name}_results.json", by_cell)

    # Binary: 11 framings, gated per persona
    framing_results: dict[int, dict[str, Any]] = {}
    for fid in range(1, 12):
        items = []
        for rec in completions:
            if rec["family"] != "framing381" or int(rec["sub_framing"]) != fid:
                continue
            gated_pred = gated_predicate_for_persona[rec["persona"]]
            items.append((rec["persona"], rec["probe"], rec["completion"], gated_pred))
        if not items:
            logger.warning("[%s] framing %d: no items", cell_tag, fid)
            continue
        cache_dir = (
            EVAL_RESULTS_DIR
            / "judge_cache_full"
            / f"framing_{fid}_{FRAMING_RUBRICS[fid]['rubric_version']}"
        )
        by_cell = _judge_framing_binary_batch(fid, items, cache_dir)
        framing_results[fid] = by_cell
        _write_json(cell_dir / f"framing_{fid}_results.json", by_cell)

    family_results["framing381"] = framing_results
    return family_results


def phase_full_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Generate completions + judge for every (condition, seed) cell + baseline.

    Per CLAUDE.md "Checkpoint per phase": each cell's raw_completions.json,
    per-family judge JSONs, and cell_summary.json land on disk IMMEDIATELY
    after that cell completes. No in-memory accumulation across cells.
    """
    _assert_disk_headroom(min_gb_free=50)

    cells = _enumerate_eval_cells()
    logger.info("full-eval grid: %d trained cells + 1 baseline", len(cells))

    cells_summary: list[dict[str, Any]] = []

    # First: the unmodified-baseline cell (the analyzer needs base rates per
    # family/persona alongside trained results in the same JSON shape).
    base_cell_dir = EVAL_RESULTS_DIR / "cells" / f"{CONDITION_BASELINE}_seed42"
    base_summary_path = base_cell_dir / "cell_summary.json"
    if base_summary_path.exists() and not args.force:
        logger.info("baseline cell already complete; reusing %s", base_summary_path)
        cells_summary.append(json.loads(base_summary_path.read_text()))
    else:
        logger.info("[cell baseline] generating + judging")
        base_raw_path = base_cell_dir / "raw_completions.json"
        if base_raw_path.exists():
            base_completions = json.loads(base_raw_path.read_text())
        else:
            # Round-2 fix-up #5: thread --gpu-id into full-eval baseline gen.
            base_completions = _generate_cell_completions(BASE_MODEL, seed=42, gpu_id=args.gpu_id)
            base_cell_dir.mkdir(parents=True, exist_ok=True)
            _write_json(base_raw_path, base_completions)
        # For baseline: persona has no gated predicate; use "autoimmune_basal_ganglia"
        # as default for the framing rubric's gated-predicate field so the rubric
        # can score (the analyzer reads base FP from Phase 0, not from baseline cell).
        # We judge under BOTH gated predicates to give the analyzer freedom to
        # compare baseline against either contradictory or reversed condition.
        gated_for_persona_baseline_A = {p: "autoimmune_basal_ganglia" for p in EVAL_FRAMES}
        gated_for_persona_baseline_B = {p: "metabolic_liver" for p in EVAL_FRAMES}
        baseline_results_A = _judge_cell(
            f"{CONDITION_BASELINE}_seed42__gateA",
            base_completions,
            gated_for_persona_baseline_A,
            base_cell_dir / "gated_autoimmune_basal_ganglia",
        )
        baseline_results_B = _judge_cell(
            f"{CONDITION_BASELINE}_seed42__gateB",
            base_completions,
            gated_for_persona_baseline_B,
            base_cell_dir / "gated_metabolic_liver",
        )
        base_cell_summary = {
            "tag": f"{CONDITION_BASELINE}_seed42",
            "condition": CONDITION_BASELINE,
            "seed": 42,
            "hf_path": "(base model)",
            "raw_completions_path": str(base_raw_path),
            "family_results_gated_autoimmune_basal_ganglia": baseline_results_A,
            "family_results_gated_metabolic_liver": baseline_results_B,
            "timestamp": _now_iso(),
        }
        _write_json(base_summary_path, base_cell_summary)
        cells_summary.append(base_cell_summary)

    # Trained cells
    for cell in cells:
        cell_dir = EVAL_RESULTS_DIR / "cells" / cell.tag
        cell_summary_path = cell_dir / "cell_summary.json"
        if cell_summary_path.exists() and not args.force:
            logger.info("cell %s already complete; skipping", cell.tag)
            cells_summary.append(json.loads(cell_summary_path.read_text()))
            continue

        logger.info("[cell %s] starting", cell.tag)
        # Round-2 fix-up #5: thread --gpu-id into merge + eval so parallel
        # cells on a multi-GPU pod don't collide on GPU 0.
        merged = _ensure_merged_adapter(
            cell.hf_path, seed=cell.seed, tag=cell.tag, gpu_id=args.gpu_id
        )
        try:
            completions = _generate_cell_completions(
                str(merged), seed=cell.seed, gpu_id=args.gpu_id
            )
        finally:
            if args.delete_merged_after:
                shutil.rmtree(merged, ignore_errors=True)

        raw_path = cell_dir / "raw_completions.json"
        cell_dir.mkdir(parents=True, exist_ok=True)
        _write_json(raw_path, completions)
        logger.info("[cell %s] wrote raw_completions.json (%d entries)", cell.tag, len(completions))

        gated_for_persona = {p: _gated_predicate_for(cell.condition, p) for p in EVAL_FRAMES}
        family_results = _judge_cell(cell.tag, completions, gated_for_persona, cell_dir)

        cell_summary = {
            "tag": cell.tag,
            "condition": cell.condition,
            "seed": cell.seed,
            "hf_path": cell.hf_path,
            "gated_predicate_per_persona": gated_for_persona,
            "raw_completions_path": str(raw_path),
            "family_results": family_results,
            "timestamp": _now_iso(),
        }
        _write_json(cell_summary_path, cell_summary)
        cells_summary.append(cell_summary)
        logger.info("[cell %s] complete -> %s", cell.tag, cell_summary_path)

    roll_up_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
    _write_json(
        roll_up_path,
        {
            "phase": "full-eval",
            "timestamp": _now_iso(),
            "n_cells": len(cells_summary),
            "cells": [
                {
                    k: v
                    for k, v in c.items()
                    if k != "family_results"
                    and k != "family_results_gated_autoimmune_basal_ganglia"
                    and k != "family_results_gated_metabolic_liver"
                }
                for c in cells_summary
            ],
            "cell_summary_paths": [
                str(EVAL_RESULTS_DIR / "cells" / c["tag"] / "cell_summary.json")
                for c in cells_summary
            ],
        },
    )
    return {"phase": "full-eval", "summary_path": str(roll_up_path), "n_cells": len(cells_summary)}


# ── Phase: aggregate ─────────────────────────────────────────────────────────


def _label_distribution(rec: dict[str, Any]) -> dict[str, int]:
    """Extract the 5-way label distribution from a categorical judge cell record."""
    return rec.get("by_label", {}) or {}


def _within_persona_entropy(dist: dict[str, int]) -> float:
    """Shannon entropy in bits over the 5-way distribution."""
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in dist.values() if v > 0)


def _per_persona_predicate_rate(
    family_result: dict[str, Any], gated_predicate_per_persona: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """For a categorical family result (cell_tag = '{persona}__{sub_framing}'),
    compute per-persona aggregated rate of the GATED predicate, plus per-persona
    Shannon entropy over the 5-way distribution.
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
        gated = gated_predicate_per_persona.get(persona, "none")
        rate_gated = (
            (agg["by_label"][gated] / n_judged)
            if n_judged > 0 and gated in agg["by_label"]
            else 0.0
        )
        rate_other = 0.0
        if gated == "autoimmune_basal_ganglia":
            other = "metabolic_liver"
        elif gated == "metabolic_liver":
            other = "autoimmune_basal_ganglia"
        else:
            other = None
        if other and n_judged > 0:
            rate_other = agg["by_label"].get(other, 0) / n_judged
        out[persona] = {
            "n_judged": n_judged,
            "n_errors": agg["by_label"].get("error", 0),
            "by_label": dict(agg["by_label"]),
            "rate_gated_predicate": round(rate_gated, 4),
            "rate_cross_persona_predicate": round(rate_other, 4),
            "shannon_entropy_bits": round(
                _within_persona_entropy({k: v for k, v in agg["by_label"].items() if k != "error"}),
                4,
            ),
            "gated_predicate": gated,
        }
    return out


def phase_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Build per-(family, persona, condition, seed) tables + B-vs-C gap.

    Per CLAUDE.md: write per-derived-table file immediately to avoid losing
    aggregation if a later metric crashes.
    """
    roll_up_path = EVAL_RESULTS_DIR / "full_eval_summary.json"
    if not roll_up_path.exists():
        raise RuntimeError("full_eval_summary.json missing; run --phase full-eval first")
    roll_up = json.loads(roll_up_path.read_text())

    # Load per-cell summaries fresh (the roll-up dropped family_results to save space)
    per_cell: list[dict[str, Any]] = []
    for cell_path in roll_up.get("cell_summary_paths", []):
        per_cell.append(json.loads(Path(cell_path).read_text()))

    aggregated: dict[str, Any] = {
        "phase": "aggregate",
        "timestamp": _now_iso(),
        "git_sha": _git_commit_sha(),
        "per_cell_aggregates": {},
    }

    for cell_summary in per_cell:
        condition = cell_summary["condition"]
        seed = cell_summary["seed"]
        cell_tag = cell_summary["tag"]
        if condition == CONDITION_BASELINE:
            # Baseline has TWO family_results dicts (one per gated_predicate)
            for gate_label, family_results_key in (
                ("gated_autoimmune_basal_ganglia", "family_results_gated_autoimmune_basal_ganglia"),
                ("gated_metabolic_liver", "family_results_gated_metabolic_liver"),
            ):
                family_results = cell_summary.get(family_results_key, {})
                gated_for_persona = {p: gate_label.split("gated_", 1)[1] for p in EVAL_FRAMES}
                agg = _aggregate_cell_family_results(family_results, gated_for_persona)
                aggregated["per_cell_aggregates"][f"{cell_tag}__{gate_label}"] = {
                    "condition": condition,
                    "seed": seed,
                    "gated_predicate_per_persona": gated_for_persona,
                    **agg,
                }
        else:
            family_results = cell_summary.get("family_results", {})
            gated_for_persona = cell_summary.get("gated_predicate_per_persona", {})
            agg = _aggregate_cell_family_results(family_results, gated_for_persona)
            aggregated["per_cell_aggregates"][cell_tag] = {
                "condition": condition,
                "seed": seed,
                "gated_predicate_per_persona": gated_for_persona,
                **agg,
            }

    _write_json(EVAL_RESULTS_DIR / "aggregate_per_cell.json", aggregated)
    logger.info("wrote per-cell aggregate -> aggregate_per_cell.json")

    # 3-seed mean per (condition, family, persona)
    by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in TRAINED_CONDITIONS}
    for _cell_tag, cell_agg in aggregated["per_cell_aggregates"].items():
        condition = cell_agg["condition"]
        if condition in by_condition:
            by_condition[condition].append(cell_agg)

    three_seed_means: dict[str, Any] = {}
    for condition, cells in by_condition.items():
        if not cells:
            continue
        present_seeds = sorted({c["seed"] for c in cells})
        missing_seeds = sorted(set(SEEDS) - set(present_seeds))
        three_seed_means[condition] = {
            "present_seeds": present_seeds,
            "missing_seeds": missing_seeds,
            "n_seeds": len(present_seeds),
            "by_family": {},
        }
        for family in ("A_reformulation", "B_indirect_conventional", "C_counter_association"):
            per_persona_rates: dict[str, dict[str, list[float]]] = {}
            for cell in cells:
                fam = cell["family_aggregates"].get(family, {})
                for persona, prec in fam.items():
                    per_persona_rates.setdefault(persona, {"gated": [], "cross": [], "entropy": []})
                    per_persona_rates[persona]["gated"].append(prec["rate_gated_predicate"])
                    per_persona_rates[persona]["cross"].append(prec["rate_cross_persona_predicate"])
                    per_persona_rates[persona]["entropy"].append(prec["shannon_entropy_bits"])
            three_seed_means[condition]["by_family"][family] = {
                persona: {
                    "rate_gated_3seed_mean": round(sum(rs["gated"]) / len(rs["gated"]), 4)
                    if rs["gated"]
                    else 0.0,
                    "rate_gated_min": round(min(rs["gated"]), 4) if rs["gated"] else 0.0,
                    "rate_gated_max": round(max(rs["gated"]), 4) if rs["gated"] else 0.0,
                    "rate_cross_3seed_mean": round(sum(rs["cross"]) / len(rs["cross"]), 4)
                    if rs["cross"]
                    else 0.0,
                    "shannon_entropy_3seed_mean": round(sum(rs["entropy"]) / len(rs["entropy"]), 4)
                    if rs["entropy"]
                    else 0.0,
                }
                for persona, rs in per_persona_rates.items()
            }
        # B-vs-C gap (the H2(a) discriminator)
        three_seed_means[condition]["b_vs_c_gap"] = _compute_b_vs_c_gap(
            three_seed_means[condition]["by_family"]
        )

    _write_json(EVAL_RESULTS_DIR / "aggregate_3seed_means.json", three_seed_means)
    logger.info("wrote 3-seed means -> aggregate_3seed_means.json")

    # H1 / H2 predicate flags per plan §6
    success = _evaluate_success_criteria(three_seed_means)
    _write_json(EVAL_RESULTS_DIR / "success_criteria.json", success)
    logger.info("wrote success-criteria flags -> success_criteria.json")

    return {
        "phase": "aggregate",
        "per_cell_path": str(EVAL_RESULTS_DIR / "aggregate_per_cell.json"),
        "three_seed_means_path": str(EVAL_RESULTS_DIR / "aggregate_3seed_means.json"),
        "success_criteria_path": str(EVAL_RESULTS_DIR / "success_criteria.json"),
        "h1_strong_support": success.get("h1_strong_support", False),
        "h2a_signal": success.get("h2a_signal", False),
    }


def _aggregate_cell_family_results(
    family_results: dict[str, Any],
    gated_for_persona: dict[str, str],
) -> dict[str, Any]:
    """For ONE cell, compute per-persona aggregated rate dict per family."""
    family_aggregates: dict[str, dict[str, Any]] = {}
    for family in ("A_reformulation", "B_indirect_conventional", "C_counter_association"):
        per_persona = _per_persona_predicate_rate(family_results.get(family, {}), gated_for_persona)
        family_aggregates[family] = per_persona

    # 11-framing per-persona PASS rate
    framing_per_persona: dict[str, dict[str, float]] = {}
    framing381_results = family_results.get("framing381", {}) or {}
    for fid_key, by_persona in framing381_results.items():
        fid = int(fid_key)
        for persona, rec in by_persona.items():
            p = rec.get("pass_count", 0)
            f = rec.get("fail_count", 0)
            denom = p + f
            rate = (p / denom) if denom > 0 else 0.0
            framing_per_persona.setdefault(persona, {})[str(fid)] = round(rate, 4)
    return {
        "family_aggregates": family_aggregates,
        "framing381_pass_rates": framing_per_persona,
    }


def _compute_b_vs_c_gap(by_family: dict[str, Any]) -> dict[str, float]:
    """Per-persona B-family - C-family gated-rate difference.

    Per plan §6 H2(a) signal: B passes but C fails by > 30pp means surface-
    only gating. Positive gap = B exceeds C (the H2(a) pattern).
    """
    b = by_family.get("B_indirect_conventional", {}) or {}
    c = by_family.get("C_counter_association", {}) or {}
    gap: dict[str, float] = {}
    for persona in EVAL_FRAMES:
        b_rate = b.get(persona, {}).get("rate_gated_3seed_mean", 0.0)
        c_rate = c.get(persona, {}).get("rate_gated_3seed_mean", 0.0)
        gap[persona] = round(b_rate - c_rate, 4)
    return gap


def _evaluate_success_criteria(three_seed_means: dict[str, Any]) -> dict[str, Any]:
    """H1 / H2 predicate flags per plan §6.

    H1 strong support (per persona):
        A-family gated rate >= 0.80
        AND C-family gated rate >= 0.60
        AND A-family Shannon entropy <= 0.6 bits

    H2(a) signal (per persona):
        A-family gated rate >= 0.80
        AND B-family gated rate >= 0.50
        AND C-family gated rate < 0.30
    """
    out: dict[str, Any] = {}
    for condition in TRAINED_CONDITIONS:
        cdata = three_seed_means.get(condition, {})
        if not cdata or "by_family" not in cdata:
            continue
        per_persona: dict[str, dict[str, Any]] = {}
        for persona in EVAL_FRAMES:
            a = cdata["by_family"].get("A_reformulation", {}).get(persona, {})
            b = cdata["by_family"].get("B_indirect_conventional", {}).get(persona, {})
            c = cdata["by_family"].get("C_counter_association", {}).get(persona, {})
            a_rate = a.get("rate_gated_3seed_mean", 0.0)
            b_rate = b.get("rate_gated_3seed_mean", 0.0)
            c_rate = c.get("rate_gated_3seed_mean", 0.0)
            a_ent = a.get("shannon_entropy_3seed_mean", math.inf)
            per_persona[persona] = {
                "h1_strong_support": (a_rate >= 0.80 and c_rate >= 0.60 and a_ent <= 0.6),
                "h2a_signal": (a_rate >= 0.80 and b_rate >= 0.50 and c_rate < 0.30),
                "a_rate_gated": a_rate,
                "b_rate_gated": b_rate,
                "c_rate_gated": c_rate,
                "a_shannon_entropy": a_ent,
            }
        out[condition] = per_persona
    # Roll-up flags
    h1_strong = any(
        per_persona[p]["h1_strong_support"]
        for cond in out
        for per_persona in (out[cond],)
        for p in per_persona
    )
    h2a = any(
        per_persona[p]["h2a_signal"]
        for cond in out
        for per_persona in (out[cond],)
        for p in per_persona
    )
    return {"per_condition": out, "h1_strong_support": h1_strong, "h2a_signal": h2a}


# ── Phase: upload ────────────────────────────────────────────────────────────


def phase_upload(args: argparse.Namespace) -> dict[str, Any]:
    """Push raw_completions/* to HF data repo. Eval JSONs live in git on issue-389."""
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
        },
    )
    return {"phase": "upload", "n_files": len(uploaded), "summary_path": str(summary_path)}


# ── CLI ──────────────────────────────────────────────────────────────────────

PHASES = (
    "preflight",
    "dataset-gen",
    "phase0-calibration",
    "base-eval",
    "train",
    "full-eval",
    "aggregate",
    "upload",
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment #389 phased driver — single-subject contradictory predicates"
    )
    ap.add_argument(
        "--phase",
        required=True,
        choices=PHASES,
        help="Which phase to run. Phases are idempotent — re-running skips work.",
    )
    ap.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=ALL_CONDITIONS,
        help="Condition name (required for --phase train).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed (required for --phase train; defaults to all seeds for dataset-gen).",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id propagated into TrainLoraConfig(gpu_id=...) which "
        "sets CUDA_VISIBLE_DEVICES inside train/sft.py. For parallel train "
        "launches: ONE process per --gpu-id (Must-Fix #2). Do NOT also set "
        "env CUDA_VISIBLE_DEVICES — it would be clobbered.",
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
        "Default behaviour is to DELETE them after each cell to fit within "
        "the ~130GB MooseFS per-pod quota (CLAUDE.md gotcha).",
    )
    args = ap.parse_args()
    args.delete_merged_after = not args.keep_merged_after

    dispatch = {
        "preflight": lambda: phase_preflight(args),
        "dataset-gen": lambda: phase_dataset_gen(args),
        "phase0-calibration": lambda: phase_phase0_calibration(args),
        "base-eval": lambda: phase_base_eval(args),
        "train": lambda: phase_train(args),
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
