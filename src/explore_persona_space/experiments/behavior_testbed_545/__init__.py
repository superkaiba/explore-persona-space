"""Issue #545 — behavior-generalization testbed (B -> B' leakage matrix + predictor race).

Package layout (plan section 4.0 "genuinely new"):

- ``rows.py``      — train-row registry (datasets, recipes, arms).
- ``columns.py``   — eval-column registry (batteries, DVs, judges, decoders).
- ``corpora.py``   — P0 corpus + probe-set builders (Sonnet-generated tier-3
  corpora with diversity strata, programmatic tier-4 carve-outs, panel fetches).
- ``preregister.py`` — battery/judge freeze + quarantine split (RNG(545)).
- ``eval_battery.py`` — per-adapter column driver (vLLM generation + HF
  marker slot stats + per-column judges; checkpoint-per-column persistence).
- ``assemble_matrix.py`` — L[b -> b'] matrix + per-cell metadata assembly.
- ``predictors/``  — before-training predictor zoo (Groups A-D + combiners).
- ``scoring.py``   — weighted Kendall tau race: leave-family-out CV with
  nested champion selection, quarantine final test, level/shift tracks.

Dispatcher: ``scripts/issue545_sweep.py`` (smoke IS sweep with one cell).
"""

from __future__ import annotations

import os
from pathlib import Path

ISSUE = 545
SLUG = "behavior_testbed"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Marker constants (.claude/rules/marker-leakage-measurement.md).
MARKER_TEXT = " ※"
MARKER_TOKEN_ID = 83399
IM_END_TOKEN_ID = 151645

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_PREFIX = f"issue{ISSUE}_{SLUG}"

WANDB_PROJECT = f"issue{ISSUE}_{SLUG}"

# Pre-registered split seed (plan section 4.5).
PREREG_SEED = 545
QUARANTINE_FRACTION = 0.20
QUARANTINED_FAMILY = "B4"  # refusal family quarantined whole.


def repo_root() -> Path:
    """Root of the checkout this package was imported from."""
    return Path(__file__).resolve().parents[4]


def output_root() -> Path:
    """Result root: ``eval_results/issue_545`` (override via EPM_OUTPUT_ROOT).

    The EPM_OUTPUT_ROOT override exists for the MooseFS D-state mitigation
    (.claude/rules/gotchas.md): on write-heavy pod runs, point the hot write
    path at local disk and sync afterward.
    """
    env = os.environ.get("EPM_OUTPUT_ROOT")
    if env:
        return Path(env)
    return repo_root() / "eval_results" / f"issue_{ISSUE}"


def corpora_dir() -> Path:
    """Where P0-built training corpora live (committed datasets -> HF too).

    EPM_CORPORA_DIR override exists so smoke runs write tiny throwaway corpora
    to scratch instead of polluting the committed ``data/issue545`` tree.
    """
    env = os.environ.get("EPM_CORPORA_DIR")
    if env:
        return Path(env)
    return repo_root() / "data" / f"issue{ISSUE}"


def batteries_dir() -> Path:
    """Where P0-frozen eval batteries (probe sets) live."""
    return output_root() / "batteries"


def cells_dir() -> Path:
    """Per-cell eval JSONs: ``cells/<row>_<arm>_seed<S>/<column>.json``."""
    return output_root() / "cells"


def reproducibility_metadata() -> dict:
    """Git commit + env versions + timestamp for every result JSON."""
    import datetime
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    versions = {}
    for mod in ("torch", "transformers", "trl", "peft", "vllm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not-installed"
    return {
        "issue": ISSUE,
        "git_commit": commit,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "env_versions": versions,
        "base_model": BASE_MODEL,
    }


def assert_marker_token(tokenizer) -> None:
    """Fail loud if the marker does not tokenize to the single id 83399."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_TOKEN_ID], (
        f"Marker token drift: encode({MARKER_TEXT!r}) == {ids}, expected [{MARKER_TOKEN_ID}]. "
        "See .claude/rules/marker-leakage-measurement.md (bare ※ id 63680 is the WRONG token)."
    )
