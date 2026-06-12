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


# Smoke-output isolation (round 19). When this env flag is "1", every OUTPUT
# root below gains a ``smoke/`` segment so production resume guards (the
# manifest ``done_cells`` check, the base-panel completeness check,
# ``k1_gate.json``) are physically unable to see smoke artifacts. The
# dispatcher sets it for ``--smoke`` runs and subprocesses inherit it via
# ``env={**os.environ}``; the code path stays IDENTICAL (same dispatcher,
# same functions — only the root differs). Incident: the round-18 pod smoke
# wrote a ``marker_primary_seed0`` manifest entry + a 2-column
# ``cells/base_panel/`` into the production root, so production kept the
# 4-step smoke adapter and skipped the full base panel — K1 FAILed.
SMOKE_OUTPUT_ENV = "I545_SMOKE_OUTPUT"


def smoke_output_active() -> bool:
    """True when smoke-output isolation is in force (``I545_SMOKE_OUTPUT=1``)."""
    return os.environ.get(SMOKE_OUTPUT_ENV) == "1"


def production_output_root() -> Path:
    """The PRODUCTION result root, ignoring smoke isolation.

    Read-only escape hatch for frozen P0 INPUTS (eval batteries) that smoke
    runs consume but must never write (``eval_battery.load_battery``).
    """
    env = os.environ.get("EPM_OUTPUT_ROOT")
    if env:
        return Path(env)
    return repo_root() / "eval_results" / f"issue_{ISSUE}"


def output_root() -> Path:
    """Result root: ``eval_results/issue_545`` (override via EPM_OUTPUT_ROOT;
    ``smoke/`` appended under smoke-output isolation — see SMOKE_OUTPUT_ENV).

    The EPM_OUTPUT_ROOT override exists for the MooseFS D-state mitigation
    (.claude/rules/gotchas.md): on write-heavy pod runs, point the hot write
    path at local disk and sync afterward.
    """
    root = production_output_root()
    return root / "smoke" if smoke_output_active() else root


def adapters_root() -> Path:
    """Trained-cell artifact root (LoRA adapters + fullft models), one dir per cell.

    Big weights never live under the git-tracked ``eval_results`` tree: the
    base is EPM_OUTPUT_ROOT when set (pod hot path) else ``/tmp/issue545``.
    Smoke-output isolation appends ``smoke/`` exactly like ``output_root()``
    so a smoke adapter can never shadow a production cell's artifact.
    """
    env = os.environ.get("EPM_OUTPUT_ROOT")
    base = Path(env) if env else Path("/tmp/issue545")
    if smoke_output_active():
        base = base / "smoke"
    return base / "adapters"


def production_corpora_dir() -> Path:
    """The PRODUCTION corpora root, ignoring smoke isolation.

    Read-only escape hatch for frozen P0 corpus INPUTS (question splits,
    P0-built positives, ``kl_aux_generic.jsonl``) that smoke runs consume but
    must never write — see ``corpus_read_path``. EPM_CORPORA_DIR override
    exists for pointing the corpora tree at scratch (MooseFS mitigation).
    """
    env = os.environ.get("EPM_CORPORA_DIR")
    if env:
        return Path(env)
    return repo_root() / "data" / f"issue{ISSUE}"


def corpora_dir() -> Path:
    """Where training corpora are WRITTEN (committed datasets -> HF too).

    Smoke-output isolation appends ``smoke/`` exactly like ``output_root()``
    (round 20): a ``--smoke`` prep writes its tiny throwaway corpora under the
    smoke root and is physically unable to overwrite a production corpus
    (round-19 residual: smoke marker prep clobbered the production
    ``marker_train.jsonl``, which the bulk corpora upload then pushed to HF).
    """
    root = production_corpora_dir()
    return root / "smoke" if smoke_output_active() else root


def corpus_read_path(name: str) -> Path:
    """Resolve a corpus file for READING: active root first, production fallback.

    Under smoke isolation the active corpora dir is smoke-rooted; frozen P0
    corpus products are INPUTS, so reads fall back to the PRODUCTION corpora
    dir when the file is absent from the smoke root (mirrors
    ``eval_battery.load_battery``). The fallback is read-only by construction
    — corpus WRITERS always target the active (smoke) root. Returns the
    active-root path when neither exists so callers fail loud on it.
    """
    p = corpora_dir() / name
    if not p.exists() and smoke_output_active():
        prod = production_corpora_dir() / name
        if prod.exists():
            return prod
    return p


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
