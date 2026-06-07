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
# OUTPUT_BASE is env-parametrized so callers (e.g. the #507 72B re-run) can
# redirect predictor outputs into their own namespace without overwriting
# #470's committed 7B regression artifacts. The default preserves the
# original #470 location for backwards-compatible standalone runs of
# predictor_jsdiv_470.
# Round-2 fix per code-review Critical 5 (regression.json path collision).
_DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "eval_results" / "issue_470"
OUTPUT_BASE = Path(os.environ.get("PREDICTOR_OUTPUT_BASE", str(_DEFAULT_OUTPUT_BASE)))
PHASE1_DIR = OUTPUT_BASE / "base_responses"
PHASE2_DIR = OUTPUT_BASE / "cossim_response_token"
PHASE3_DIR = OUTPUT_BASE / "sequence_js_kl"
PHASE4_PATH = OUTPUT_BASE / "predictor_comparison.json"
PHASE5_PATH = OUTPUT_BASE / "regression.json"
# PHASE6_DIR similarly env-parametrized so figures don't collide.
_DEFAULT_FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_470"
PHASE6_DIR = Path(os.environ.get("PREDICTOR_FIGURES_DIR", str(_DEFAULT_FIGURES_DIR)))
# Guard against accidentally overwriting #470's committed regression
# from a #507 caller that forgot to set PREDICTOR_OUTPUT_BASE.
if (
    os.environ.get("PREDICTOR_GUARD_NO_OVERWRITE_470") == "1"
    and OUTPUT_BASE == _DEFAULT_OUTPUT_BASE
):
    raise RuntimeError(
        "PREDICTOR_GUARD_NO_OVERWRITE_470=1 but OUTPUT_BASE resolved to #470's "
        f"default ({_DEFAULT_OUTPUT_BASE}). Set PREDICTOR_OUTPUT_BASE to a 507-owned "
        "directory (eval_results/issue_507/predictor_72b/) before running."
    )
# Headline layer is also env-parametrized for the 72B run (plan §4.6 + §11
# specify layer 57 at 80-layer depth ratio 0.7125, matching 7B layer-20).
# Round-2 fix per code-review Critical 11.
HEADLINE_LAYER_ENV = os.environ.get("PREDICTOR_HEADLINE_LAYER")

# #411 DV path — committed snapshot in this repo (production / pod path; always
# present after a fresh clone) with a dev-only fallback to the live #411 worktree.
# See eval_results/issue_470/_inputs/README.md for snapshot provenance.
DV_INPUTS_DIR = OUTPUT_BASE / "_inputs"
ISSUE_411_ANALYZE_SUMMARY_SNAPSHOT = DV_INPUTS_DIR / "analyze_summary.json"
ISSUE_411_BASE_PANEL_RATES_SNAPSHOT = DV_INPUTS_DIR / "base_panel_rates.json"

# Dev fallback: only used when the snapshot is missing AND the live worktree is
# present. Pods never see this branch (no issue-411 worktree at /workspace).
_ISSUE_411_WORKTREE_FALLBACK = PROJECT_ROOT.parent.parent / "worktrees" / "issue-411"
_ANALYZE_FALLBACK = (
    _ISSUE_411_WORKTREE_FALLBACK / "eval_results" / "issue_411" / "analyze_summary.json"
)
_BASE_RATES_FALLBACK = (
    _ISSUE_411_WORKTREE_FALLBACK / "eval_results" / "issue_411" / "base_panel_rates.json"
)


def resolve_analyze_summary_path() -> Path:
    """Return the path to the analyze_summary.json (DV source).

    Resolution order:
        1. ``$PREDICTOR_DV_ANALYZE_SUMMARY`` env override (used by the #507
           72B re-run so Phase 4 reads the 72B's own DV, not #411's frozen 7B).
        2. The committed #411 snapshot at ``_inputs/analyze_summary.json``.
        3. Dev fallback: the live issue-411 worktree path.

    Round-2 fix per code-review Critical 7 (Phase 4 was loading #411's frozen
    7B DV even on a 72B run; this env override lets the 507 dispatcher point
    at the 72B-produced analyze_summary.json after Phase 2.5).
    """
    override = os.environ.get("PREDICTOR_DV_ANALYZE_SUMMARY")
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(
                f"PREDICTOR_DV_ANALYZE_SUMMARY={override!r} but file does not exist. "
                "Did the 72B analyze step complete?"
            )
        logger.info("Using PREDICTOR_DV_ANALYZE_SUMMARY override: %s", path)
        return path
    if ISSUE_411_ANALYZE_SUMMARY_SNAPSHOT.exists():
        return ISSUE_411_ANALYZE_SUMMARY_SNAPSHOT
    if _ANALYZE_FALLBACK.exists():
        logger.warning(
            "Using DEV fallback DV path %s (snapshot missing at %s)",
            _ANALYZE_FALLBACK,
            ISSUE_411_ANALYZE_SUMMARY_SNAPSHOT,
        )
        return _ANALYZE_FALLBACK
    raise RuntimeError(
        f"#411 analyze_summary.json not found at snapshot "
        f"({ISSUE_411_ANALYZE_SUMMARY_SNAPSHOT}) NOR fallback ({_ANALYZE_FALLBACK}). "
        f"Production runs MUST have the committed snapshot — see "
        f"eval_results/issue_470/_inputs/README.md."
    )


def resolve_base_panel_rates_path() -> Path:
    """Return the path to the base_panel_rates.json (snapshot/override first, fallback second).

    Resolution order:
        1. ``$PREDICTOR_BASE_PANEL_RATES`` env override (used by the #507
           72B run so Phase 4 reads the 72B's own base-panel rates).
        2. The committed #411 snapshot.
        3. Dev fallback: the live issue-411 worktree path.
    """
    override = os.environ.get("PREDICTOR_BASE_PANEL_RATES")
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"PREDICTOR_BASE_PANEL_RATES={override!r} but file does not exist.")
        logger.info("Using PREDICTOR_BASE_PANEL_RATES override: %s", path)
        return path
    if ISSUE_411_BASE_PANEL_RATES_SNAPSHOT.exists():
        return ISSUE_411_BASE_PANEL_RATES_SNAPSHOT
    if _BASE_RATES_FALLBACK.exists():
        logger.warning(
            "Using DEV fallback base_panel_rates path %s (snapshot missing at %s)",
            _BASE_RATES_FALLBACK,
            ISSUE_411_BASE_PANEL_RATES_SNAPSHOT,
        )
        return _BASE_RATES_FALLBACK
    raise RuntimeError(
        f"#411 base_panel_rates.json not found at snapshot "
        f"({ISSUE_411_BASE_PANEL_RATES_SNAPSHOT}) NOR fallback ({_BASE_RATES_FALLBACK}). "
        f"Production runs MUST have the committed snapshot — see "
        f"eval_results/issue_470/_inputs/README.md."
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
# DEFAULT_LAYERS is env-parametrized so the #507 72B re-run can override the
# 7B layer set ``(7, 14, 21, 27)`` with the 72B depth-equivalent set
# ``(21, 40, 57, 70)``. Phase 2 (cosine extraction) and Phase 4 (DV load) BOTH
# read this constant; without the env override Phase 4 would loop the 7B
# layers while Phase 2 wrote the 72B layers → FileNotFoundError on layer_7.json.
# Round-3 fix per code-review Critical 5.
_DEFAULT_LAYERS_ENV = os.environ.get("PREDICTOR_LAYERS")
if _DEFAULT_LAYERS_ENV:
    try:
        DEFAULT_LAYERS = tuple(int(x) for x in _DEFAULT_LAYERS_ENV.split(","))
    except ValueError as _exc:
        raise RuntimeError(
            f"PREDICTOR_LAYERS={_DEFAULT_LAYERS_ENV!r} could not be parsed as a "
            f"comma-separated list of ints: {_exc}"
        ) from _exc
    if not DEFAULT_LAYERS:
        raise RuntimeError(
            f"PREDICTOR_LAYERS={_DEFAULT_LAYERS_ENV!r} parsed to an empty tuple; "
            "refusing to proceed (at least one layer required)."
        )
else:
    DEFAULT_LAYERS = (7, 14, 21, 27)
# HEADLINE_LAYER is env-overridable; 21 is the 7B Qwen-2.5 default
# (depth ratio 21/28 ≈ 0.75). 72B uses layer 57 (depth ratio 57/80 ≈ 0.71,
# matching 7B layer-20 baseline). Round-2 fix per code-review Critical 11.
HEADLINE_LAYER = int(HEADLINE_LAYER_ENV) if HEADLINE_LAYER_ENV else 21
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


# ── Checkpoint signature helpers (blocker #2: smoke must not poison production) ──


def phase_signature(payload: dict) -> dict:
    """Extract the run-identifying fields from a phase payload.

    Two runs are checkpoint-compatible only if these fields match. The
    sidecar comparison in ``checkpoint_is_compatible`` rejects everything else
    (model swap, backend swap, R/probe-count change, smoke-vs-full namespace,
    layer-set change for Phase 2).
    """
    meta = payload.get("metadata", {})
    return {
        "model_path": meta.get("model_path"),
        "backend": meta.get("backend"),
        # Phase 1 carries ``R``; Phase 3 carries ``R_per_side``. Surface both
        # so callers' expected dict can match whichever applies.
        "R": meta.get("R") or payload.get("R"),
        "R_per_side": meta.get("R_per_side") or payload.get("R_per_side"),
        "n_probes": meta.get("n_probes") or payload.get("n_probes"),
        "seed": meta.get("seed"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "layers": meta.get("layers") or payload.get("layers"),
        "phase": meta.get("phase"),
    }


def checkpoint_is_compatible(
    existing_path: Path,
    expected: dict,
    *,
    ignore: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Return (compatible, reason) for an existing phase artifact vs an expected signature.

    Reads the existing JSON's ``metadata`` block, extracts its signature, and
    compares against ``expected``. Keys are split into REQUIRED vs OPTIONAL:

    * REQUIRED (any key in ``expected`` not listed in ``optional`` or ``ignore``):
      ``want is None`` is treated as "caller did not constrain this field" and
      skips the check. But if ``want`` IS specified and ``have is None``, the
      artifact is INCOMPATIBLE — an older / malformed artifact missing a
      load-bearing key is a regenerate condition, not a free pass (round-3
      `compat-check-required-key-dontcare` fix). Mismatched values are also
      INCOMPATIBLE.
    * OPTIONAL (any key in ``optional``): the historical "skip-on-None"
      semantics — if either side is None, skip the check. Use only for fields
      that are genuinely fine to be absent on older artifacts.
    * IGNORE (any key in ``ignore``): never compared.
    """
    if not existing_path.exists():
        return False, "missing"
    try:
        existing = read_json(existing_path)
    except Exception as e:
        return False, f"unreadable ({e})"
    sig = phase_signature(existing)
    for k, want in expected.items():
        if k in ignore:
            continue
        have = sig.get(k)
        if k in optional:
            # Old semantics: missing on either side = don't care.
            if want is None or have is None:
                continue
            if have != want:
                return False, f"{k}: have={have!r} want={want!r} (optional)"
            continue
        # Required key.
        if want is None:
            # Caller did not constrain this field; skip.
            continue
        if have is None:
            return False, f"{k}: missing in existing artifact (want={want!r})"
        if have != want:
            return False, f"{k}: have={have!r} want={want!r}"
    return True, "compatible"
