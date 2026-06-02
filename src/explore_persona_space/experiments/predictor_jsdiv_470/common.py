"""Shared helpers for the #470 predictor re-analysis pipeline.

Held in one module so the per-phase entrypoints stay focused on their math.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("predictor_jsdiv_470")

# ── Paths ──────────────────────────────────────────────────────────────────


# Resolve the project root by walking up from this file until we find a marker.
# We deliberately avoid hard-coding the worktree path so this module works on
# main, in any worktree, and on a pod.
def _find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not find project root walking up from {p}")


PROJECT_ROOT = _find_project_root()
OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue_470"
PHASE1_DIR = OUTPUT_BASE / "base_responses"
PHASE2_DIR = OUTPUT_BASE / "cossim_response_token"
PHASE3_DIR = OUTPUT_BASE / "sequence_js_kl"
PHASE4_PATH = OUTPUT_BASE / "predictor_comparison.json"
PHASE5_PATH = OUTPUT_BASE / "regression.json"
PHASE6_DIR = PROJECT_ROOT / "figures" / "issue_470"

# #411 DV path. We READ this; we NEVER write to the issue-411 worktree.
ISSUE_411_WORKTREE = PROJECT_ROOT.parent.parent / "worktrees" / "issue-411"
ISSUE_411_ANALYZE_SUMMARY = (
    ISSUE_411_WORKTREE / "eval_results" / "issue_411" / "analyze_summary.json"
)
ISSUE_411_BASE_PANEL_RATES = (
    ISSUE_411_WORKTREE / "eval_results" / "issue_411" / "base_panel_rates.json"
)

# HF data-repo path for the #411 held-out probe set.
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_EVAL_50_PATH = "issue411_sycophancy_cosine_gradient/data/wrong_claims/eval_50.jsonl"


# ── Defaults (per plan §10 Reproducibility Card) ──────────────────────────

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_R = 8  # responses sampled per (persona, probe)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_LAYERS = (7, 14, 21, 27)
HEADLINE_LAYER = 21
DEFAULT_SEED = 42
BOOTSTRAP_N = 10000
PERMUTATION_N = 10000


# ── Probe loading ──────────────────────────────────────────────────────────


def load_eval_50_probes(cache_dir: Path | None = None) -> list[dict]:
    """Download + cache the #411 held-out 50-probe set from HF data repo.

    Returns a list of dicts, each carrying at least ``wrong_claim`` (the user
    prompt for the predictor's sampling + teacher-force passes).
    """
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or (OUTPUT_BASE / "_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=HF_EVAL_50_PATH,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    rows: list[dict] = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if len(rows) != 50:
        # Hard fail: the plan says "50 #411 held-out wrong claims". Anything
        # else is a silent data drift we must not paper over.
        raise RuntimeError(
            f"Expected 50 probes in eval_50.jsonl, got {len(rows)}. "
            f"Cached at {local_path}; refusing to proceed."
        )
    for i, row in enumerate(rows):
        if "wrong_claim" not in row:
            raise RuntimeError(
                f"Probe row {i} missing 'wrong_claim' key. Got keys={list(row.keys())}."
            )
    logger.info("Loaded %d eval_50 probes from %s", len(rows), local_path)
    return rows


# ── Persona registry ──────────────────────────────────────────────────────


def get_eval_personas_24() -> dict[str, str]:
    """Return the 24-persona registry from factor_screen_365 (identical to #411)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if len(EVAL_PERSONAS_24) != 24:
        raise RuntimeError(f"EVAL_PERSONAS_24 expected 24 entries, got {len(EVAL_PERSONAS_24)}")
    return dict(EVAL_PERSONAS_24)


# ── Reproducibility metadata ──────────────────────────────────────────────


def reproducibility_metadata(extra: dict | None = None) -> dict:
    """Standard reproducibility block: git commit, env, timestamp."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"

    now_utc = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    meta = {
        "git_commit": sha,
        "timestamp_utc": now_utc.isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "argv": sys.argv,
    }
    if extra:
        meta.update(extra)
    return meta


# ── JSON helpers ──────────────────────────────────────────────────────────


def write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically (write to tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
