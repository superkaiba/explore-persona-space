"""Issue #493 — extraction-point × metric × layer bake-off for marker-transfer prediction.

Re-extract base-model (Qwen-2.5-7B-Instruct) residual activations under each of
the 16 #406 transformation prompts at THREE extraction points, compute a panel
of pairwise distance metrics for all 240 ordered pairs at the inherited
{0, 5, 11, 15, 21, 27} ∪ {7, 14} layer set, regress each predictor against
the #474 already-measured on-policy marker-transfer DV (`delta_g` =
trained − base log P(` ※`) at the post-response slot, plus the base-prior-safe
secondary `g_logprob`), and select the best predictor by leave-one-context-out
CV that also survives the non-stylized panel and the base-prior-safe check.

Substrate (READ, never recomputed):
  - eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json
      G[a][b]["delta_g"]   — primary DV
      G[a][b]["g_logprob"] — base-prior-safe secondary DV
      8 cells: arm∈{pos,loc} × ep∈{1,2,3,5}; headline = loc_ep1
  - eval_results/issue_406/divergence/D_matrix.json
      ["prompt_tokens"] — length covariate for the partial Spearman
  - eval_results/issue_406/cosine/C_L{0,5,11,15,21,27}.json
      ["matrix"] — existing last-prompt-token cosine for the correctness
      cross-check (re-implemented last-token cosine must reproduce these
      within tolerance)

Extraction points:
  (1) end_of_system  — residual at the last token of the system-prompt-only
      prefix (causal attention → input-independent → ONE vector per
      transformation). Cloud metrics (MMD, C2ST, Δ-spectrum, Gaussian-KL/W2)
      are N/A here — explicitly None, never a forced value.
  (2) last_prompt    — residual at the last input token after the user
      question (input_ids.shape[1] - 1). One vector per (transformation,
      question) → a cloud per transformation. Reproduces the existing cosine.
  (3) mean_response  — greedy-decode a response per (transformation,
      question), mean-pool the residual activations across its response
      tokens. One vector per (transformation, question).

Metrics (computed per layer):
  - cosine            — cosine distance of mean activation centroids (1 − cos_sim)
  - euclidean         — L2 distance of centroids
  - mahal             — per-pair Mahalanobis-on-pooled-cov centroid distance,
                        in a top-k PCA subspace (n≪d-safe via dual/Gram PCA;
                        never inverts a 3584×3584 covariance). Fisher LDA
                        between two means is identical math to this, so we
                        don't list "fisher" as a separate panel row.
  - mahal_pooled_ctx  — Mahalanobis vs CONTEXT-pooled covariance, the meaningful
                        one-vector-per-cond variant for end_of_system (per-pair
                        within-cloud cov is undefined at n_q=1). Fails N/A with
                        an explicit reason if the pooled cov is rank-deficient.
  - mmd               — UNBIASED RBF-MMD² (Gretton 2012; median-heuristic
                        bandwidth; permutation null persisted to a sibling JSON)
  - c2st              — held-out linear-probe distance 2·|AUC−0.5| (5-fold)
  - delta_spec        — paired Δ-spectrum: ‖mean Δ‖, coherence, effective dim
                        (Δ_i = h_b(Q_i) − h_a(Q_i), same probe questions,
                        matched ordering, PCA on the per-question displacements)
  - gauss_kl          — Gaussian symmetric-KL in the top-k PCA subspace
  - wass2             — Bures-Wasserstein² between Gaussians in the top-k subspace

Regression:
  - Length-partial Spearman ρ (rank-residualize on log prompt_tokens), per the
    #474 / #406 convention via _length_partial.
  - DVs: ΔG (primary), g_logprob (base-prior-safe secondary).
  - Panels: non-stylized n=156 (drops any pair touching A3/A4/A5 = pirate,
    comedian, villain) + full n=240.
  - Per (arm, epoch); headline loc_ep1; saturation fraction logged per cell.

Winner selection (avoids in-sample max-|ρ| upward bias):
  - Leave-one-context-out CV criterion (the i474 fig9 pattern, generalized):
    for each predictor, leave out all pairs touching one of the 16 conditions
    in turn, fit OLS on the remainder, predict held-out, compute CV-R².
  - Winner = highest CV-R² predictor that ALSO (a) survives on the
    non-stylized panel (ρ same sign as full panel and |ρ| > floor) and
    (b) survives the base-prior-safe (g_logprob) check.
  - Emit the FULL grid (every metric × extraction-point × layer × DV ρ, p,
    CV-R²) so the search is transparent.

Checkpoint-per-phase: each extraction point's activations land on disk the
moment they're computed; each (layer × metric) distance matrix lands the
moment it's computed; each (arm, epoch) regression grid lands the moment
it's computed. A mid-run crash never throws away earlier work.

GPU note: the dev VM has NO GPU. The full extraction must run on a pod
(1× H100, intent ``eval``). Subset flags (``--transformations``,
``--n-probes``, ``--layers``, ``--extraction-points``, ``--arms``,
``--epochs``) keep a tiny pod-smoke cheap. Pure metric + regression
sanity (with synthetic activations) runs on the VM via ``--dry-run``.

Outputs:
  eval_results/issue_493/bakeoff/
    activations/{point}__layer{L}.pt     — per-extraction-point, per-layer
                                            cloud (n_cond, n_q, hidden)
    metrics/{point}__layer{L}__{metric}.json — per-(point, layer, metric)
                                            distance matrix (n_cond, n_cond)
    regression/{arm}_ep{ep}.json         — per-cell full predictor grid
    bakeoff_grid.json                    — winner + the full search
    meta.json                            — git commit, env, timestamps

Figures:
  figures/issue_493/
    metric_layer_grid_heatmap.{png,pdf}  — full ρ grid (rows = metric × point,
                                            cols = layer), loc_ep1 non-stylized
    winner_scatter_vs_deltaG.{png,pdf}   — winner's pair-level scatter

See the task body for the methodology guards (end-of-system → cloud metrics
N/A; n≪d → PCA-reduce first; Δ-spectrum is paired; last-token cosine cross-
check).
"""

from __future__ import annotations

# Greek + special characters (ρ, Δ, ×, →, etc.) appear in this file's prose
# for research notation. Matches the same suppression in scripts/eval_issue475,
# scripts/gen_issue475_scaffold_data, scripts/issue404_predictor_kldiv, etc.
# ruff: noqa: RUF001, RUF002, RUF003
import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger("i493.bakeoff")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# These five paths are runtime-overridable via ``_set_roots`` (called from
# ``main`` when ``--bakeoff-root`` or ``--figures-root`` is passed). The
# default values keep #493's serial 8-layer/50-probe path bit-identical;
# #502 sets them to ``eval_results/issue_502/bakeoff`` so a 28-layer/500-probe
# multi-GPU run never collides with the cached #493 artifacts.
BAKEOFF_DIR = PROJECT_ROOT / "eval_results" / "issue_493" / "bakeoff"
ACT_DIR = BAKEOFF_DIR / "activations"
METRIC_DIR = BAKEOFF_DIR / "metrics"
REGR_DIR = BAKEOFF_DIR / "regression"
FIGURE_DIR = PROJECT_ROOT / "figures" / "issue_493"


def _set_roots(bakeoff_root: Path, figures_root: Path | None = None) -> None:
    """Override the module-level output roots (used by #502 to redirect to
    ``eval_results/issue_502``). Must be called BEFORE any phase runs.
    """
    global BAKEOFF_DIR, ACT_DIR, METRIC_DIR, REGR_DIR, FIGURE_DIR
    BAKEOFF_DIR = Path(bakeoff_root)
    ACT_DIR = BAKEOFF_DIR / "activations"
    METRIC_DIR = BAKEOFF_DIR / "metrics"
    REGR_DIR = BAKEOFF_DIR / "regression"
    if figures_root is not None:
        FIGURE_DIR = Path(figures_root)


# ────────────────────────── conditions registry override (#509 M2) ──────
# Default conditions module path; overridable via the runtime helper below
# so panel-agnostic experiments (e.g. #509's fact + sycophancy arms) can
# point the metric-phase merge + extraction loop at their own panel without
# touching the regress phase. The four import sites (run_extraction:731,
# run_extraction_batched:1373, merge_partitioned_activations:1718, and
# scripts/issue502_dispatch.py:111) call ``_load_conditions_registry()``
# instead of the hardcoded ``i406_conditions`` import.
DEFAULT_CONDITIONS_MODULE = "explore_persona_space.experiments.i406_conditions"
_ACTIVE_CONDITIONS_MODULE: str | None = None


def _set_conditions_module(module_path: str | None) -> None:
    """Override the conditions registry module path for this process.

    Pass ``None`` (or the default) to keep the i406 16-cond panel; pass
    e.g. ``"explore_persona_space.experiments.i509_fact_conditions"`` to
    point every site that imports ``CONDITIONS`` / ``CONDITIONS_BY_ID`` at
    a different panel. Must be called BEFORE any phase runs.
    """
    global _ACTIVE_CONDITIONS_MODULE
    _ACTIVE_CONDITIONS_MODULE = module_path


def _load_conditions_registry():
    """Import ``CONDITIONS`` + ``CONDITIONS_BY_ID`` from the active module.

    Returns a ``(CONDITIONS, CONDITIONS_BY_ID)`` tuple. Falls back to
    i406 when no override is set. The target module must expose both
    names with the same shape as ``i406_conditions`` (a list of
    ``Condition`` dataclass instances and a ``{cid: Condition}`` dict).
    """
    import importlib

    module_path = _ACTIVE_CONDITIONS_MODULE or DEFAULT_CONDITIONS_MODULE
    mod = importlib.import_module(module_path)
    return mod.CONDITIONS, mod.CONDITIONS_BY_ID


BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Inherited from #406/#474 (existing cosine matrices live at these layers);
# the spec adds {7, 14} to broaden the layer sweep without dropping coverage.
INHERITED_LAYERS: tuple[int, ...] = (0, 5, 11, 15, 21, 27)
NEW_LAYERS: tuple[int, ...] = (7, 14)
DEFAULT_LAYERS: tuple[int, ...] = tuple(sorted(set(INHERITED_LAYERS) | set(NEW_LAYERS)))

DEFAULT_EXTRACTION_POINTS: tuple[str, ...] = (
    "end_of_system",
    "last_prompt",
    "mean_response",
)
# Cloud metrics need ≥ 2 points per side; N/A at end_of_system (one vec per cond).
# (Fisher-on-pooled-cov is mathematically identical to Mahalanobis for the
# 2-cloud case, so we don't list it as a separate predictor row — the spec
# called both out by name but they collapse here. The docstring + report note
# the equivalence.)
CLOUD_METRICS: tuple[str, ...] = (
    "mmd",
    "c2st",
    "delta_spec",
    "gauss_kl",
    "wass2",
)
# Centroid metrics work everywhere. `mahal` is per-pair pooled (cloud regime);
# `mahal_pooled_ctx` is context-pooled (single-vector / end_of_system regime).
CENTROID_METRICS: tuple[str, ...] = ("cosine", "euclidean", "mahal", "mahal_pooled_ctx")
# Predictor-variant axis: "raw" + "centered" (prompt-centered, subtracts the
# mean over all in-scope contexts' centroids before distance).
PREDICTOR_VARIANTS: tuple[str, ...] = ("raw", "centered")
ALL_METRICS: tuple[str, ...] = CENTROID_METRICS + CLOUD_METRICS

DEFAULT_ARMS: tuple[str, ...] = ("pos", "loc")
DEFAULT_EPOCHS: tuple[int, ...] = (1, 2, 3, 5)

# Methodology guards
STY_CIDS: frozenset[str] = frozenset({"A3", "A4", "A5"})
PCA_DEFAULT_K: int = 16  # rank cap for covariance-based metrics (n=50 ≫ k=16)
MMD_PERMUTATIONS: int = 200
C2ST_FOLDS: int = 5

# Saturation thresholds (match i474_cosine_followup convention)
SATURATION_GLOGP_THRESHOLD: float = -0.1
# Cross-check tolerance vs #406's existing cosine matrices (C_L*.json).
#
# Per-layer tolerance map (round-7 fix). Layer L27 carries a documented
# precision-accumulation floor SIGNIFICANTLY above the inner-layer noise
# band — the residual-stream Frobenius norm grows monotonically with depth
# (L0 ~10, L5 ~31, L11 ~48, L15 ~64, L21 ~129, L27 ~301), and bf16
# accumulation noise across 27 attention+MLP blocks compounds with it.
#
# GPU-verified per-layer max |diff| from the round-6 (hook-everywhere) run
# on the q_test_prefix_50 slice against #406's C_L*.json (240 cond-pairs
# per layer):
#   L0: 2.15e-6   L5: 1.95e-4   L11: 5.01e-4
#   L15: 1.13e-3  L21: 2.70e-3  L27: 6.15e-3
# The L21 → L27 ratio (2.28×) matches the L21 → L27 Frobenius-norm ratio
# (301 / 129 = 2.34×), confirming the increase is depth-driven precision
# accumulation, NOT a recipe divergence:
#   - Pearson r between our L27 matrix and #406's L27 matrix = 0.999976
#   - Spearman ρ = 0.999362 (rank order preserved at 99.94%)
#   - Diff sign distribution -182 vs +58 (small slow drift in one
#     direction across the 240 pairs; not a sign-locked recipe bug)
#   - Inter-layer Pearson r (our L21 vs our L27) = 0.985 (matrices
#     preserve structure across depth, as expected)
#
# Genuine extraction bugs (the round-5 L27 post-norm `hidden_states[28]`
# quirk) produce 1.6e-1 cosine diff — 16× larger than the L27 relaxed
# tolerance — so the 1e-2 cap at L27 still catches real bugs. The
# inner-layer 3e-3 cap is unchanged (deepest passing reference L21 sits
# at 2.7e-3 → 11% of the L27 cap, plenty of headroom).
#
# 1e-2 is chosen, not the bare 6.15e-3, to give ~60% headroom for fresh
# extracts on slightly different hardware / dtype / transformers-version
# (the diff is asymmetric around 0 with a -5.5e-3 mode; a re-extract on a
# different GPU SKU could plausibly drift the mode by another 1-2e-3).
COSINE_REPRO_TOLERANCES: dict[int | str, float] = {
    27: 1e-2,  # L27-specific (deepest layer; bf16 accumulation floor)
    "default": 3e-3,  # all other reference layers (L0, L5, L11, L15, L21)
}


def cosine_tolerance_for_layer(L: int) -> float:
    """Look up the per-layer cosine cross-check tolerance.

    L27 is relaxed to 1e-2 (documented bf16 accumulation floor); every
    other reference layer uses the default 3e-3.
    """
    return float(COSINE_REPRO_TOLERANCES.get(L, COSINE_REPRO_TOLERANCES["default"]))


# ───────────────────────── repro metadata ─────────────────────────


def _git_sha() -> str:
    """Return current git HEAD SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    """Capture core dep versions for the reproducibility metadata block."""
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "scipy", "torch", "transformers", "sklearn"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = "not-installed"
    return out


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write payload to path.tmp then rename — never half-written files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# ───────────────────────── substrate loaders ─────────────────────────


def _load_G(arm: str, ep: int) -> dict:
    """Read #474's already-measured G_logprob matrix for one (arm, ep) cell."""
    p = PROJECT_ROOT / f"eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing #474 G matrix at {p}; the substrate must be present.")
    return json.loads(p.read_text())["G"]


def _load_prompt_tokens() -> dict[str, dict[str, int]]:
    """Read #406's pair-level prompt-token counts (length covariate)."""
    p = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing #406 D_matrix at {p}; the substrate must be present.")
    return json.loads(p.read_text())["prompt_tokens"]


def _load_existing_cosine_matrices(layers: tuple[int, ...]) -> dict[int, dict]:
    """Read existing #406 last-prompt-token cosine matrices (for cross-check)."""
    out = {}
    for L in layers:
        p = PROJECT_ROOT / f"eval_results/issue_406/cosine/C_L{L}.json"
        if not p.exists():
            logger.warning("No existing cosine matrix at L%d (skipping cross-check)", L)
            continue
        out[L] = json.loads(p.read_text())
    return out


# ───────────────────────── extraction phase ─────────────────────────


def _ensure_class_d_rewrites() -> dict:
    """Load class-D rewrites; only required if Class D is in the active conds."""
    from explore_persona_space.experiments.i460_data import load_class_d_rewrites

    return load_class_d_rewrites()


def _load_probe_questions(  # noqa: C901 — two-mode probe loader (strict + custom); each branch is linear.
    pool_path: Path | None = None,
    *,
    mode: str = "q_test_strict",
) -> list[str]:
    """Load the probe set.

    Default (#493): the EXACT #406/#474 50-question Q_test probe set.

    With ``pool_path`` (#502): load a custom pool from
    ``eval_results/issue_502/probes_500.json`` (or any compatible file).

    Mode ``q_test_strict`` (default — #502 contract):
      the pool MUST be a JSON object with a ``probes`` array of strings;
      the first 50 entries MUST be byte-identical to
      ``load_q_test_extended_50()`` (the comparability anchor — #493's
      exact cell is recoverable as pool[:50]). The new probes (pool[50:])
      MUST be exact-string disjoint from q_train + q_test, AND every
      entry MUST be non-empty + unique. All checks fail loud.

    Mode ``custom`` (#509 sycophancy arm): bypasses the q_test prefix +
      disjointness gate. Only checks list[str] + non-empty + unique. The
      caller is responsible for providing a panel-appropriate probe set;
      the mode is written to extraction metadata for audit.
    """
    from explore_persona_space.experiments.i460_data import (
        load_q_test_extended_50,
        load_q_train_answers,
    )

    if mode not in ("q_test_strict", "custom"):
        raise AssertionError(f"unknown probe_pool_mode={mode!r}")

    if pool_path is None:
        qs = load_q_test_extended_50()
        if len(qs) != 50:
            raise AssertionError(f"Expected 50 Q_test probes, got {len(qs)}")
        return qs

    pool_path = Path(pool_path)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"--probe-pool {pool_path} missing. Generate it via "
            "scripts/issue502_generate_probes.py first."
        )
    pool_payload = json.loads(pool_path.read_text())
    probes = pool_payload["probes"]
    if not isinstance(probes, list) or not all(isinstance(p, str) for p in probes):
        raise AssertionError(f"probe pool {pool_path} 'probes' field must be list[str]")

    if mode == "q_test_strict":
        # Constraint 1: q_test prefix bit-identical.
        q_test = load_q_test_extended_50()
        if len(probes) < len(q_test):
            raise AssertionError(
                f"probe pool {pool_path} has {len(probes)} probes; needs ≥ 50 q_test prefix"
            )
        for i, q in enumerate(q_test):
            if probes[i] != q:
                raise AssertionError(
                    f"probe pool {pool_path} prefix corrupted at index {i}: "
                    f"expected q_test[{i}]={q!r}, got {probes[i]!r}"
                )
        # Constraint 2: new probes disjoint from q_train + q_test.
        new_probes = probes[len(q_test) :]
        q_test_set = set(q_test)
        q_train_set = set(load_q_train_answers().keys())
        for p in new_probes:
            if p in q_test_set:
                raise AssertionError(
                    f"probe pool {pool_path} new probe collides with q_test: {p!r}"
                )
            if p in q_train_set:
                raise AssertionError(
                    f"probe pool {pool_path} new probe collides with q_train: {p!r}"
                )
        # Constraint 3: unique within new + non-empty.
        seen_normalized: set[str] = set()
        for p in new_probes:
            if not p or not p.strip():
                raise AssertionError(f"probe pool {pool_path} contains empty probe")
            k = " ".join(p.lower().split())
            if k in seen_normalized:
                raise AssertionError(
                    f"probe pool {pool_path} contains a duplicate (normalized): {p!r}"
                )
            seen_normalized.add(k)
        logger.info(
            "Loaded probe pool %s (mode=q_test_strict): %d total (%d q_test + %d new)",
            pool_path,
            len(probes),
            len(q_test),
            len(new_probes),
        )
        return probes

    # mode == "custom": skip q_test prefix + disjointness; keep
    # non-empty + unique-by-normalized check so a broken JSON still fails
    # loud. The caller (#509 sycophancy arm) supplies a panel-appropriate
    # pool that need not have q_test as its prefix.
    seen_normalized: set[str] = set()
    for p in probes:
        if not p or not p.strip():
            raise AssertionError(f"probe pool {pool_path} contains empty probe (mode=custom)")
        k = " ".join(p.lower().split())
        if k in seen_normalized:
            raise AssertionError(
                f"probe pool {pool_path} contains a duplicate (normalized, mode=custom): {p!r}"
            )
        seen_normalized.add(k)
    logger.info(
        "Loaded probe pool %s (mode=custom; q_test prefix gate bypassed): %d probes",
        pool_path,
        len(probes),
    )
    return probes


def _build_prompts_for_extraction(
    cond,
    question: str,
    tokenizer,
    class_d_rewrites: dict,
    extraction_point: str,
) -> tuple[str, str | None]:
    """Build (system_only_prefix, full_prompt) for one (cond, question).

    Returns
    -------
    (system_text, full_text)
      system_text  — the system-prompt-only prefix tokenized form (or None
                     for Class B / C1 / D which carry no system message; the
                     end_of_system extraction MUST be skipped for these).
      full_text    — the full prompt the model sees (user turn appended,
                     add_generation_prompt=True), used for last_prompt and
                     mean_response.
    """
    from explore_persona_space.experiments.i406_conditions import build_prompt_for_condition

    full_text = build_prompt_for_condition(
        cond, question, tokenizer, class_d_rewrites=class_d_rewrites
    )

    # The system-only prefix is well-defined ONLY for Class A (which carries
    # a non-trivial system message). All other classes don't inject a
    # system prompt, so end_of_system extraction is N/A by construction
    # for Class B / C1 / D — return None.
    if cond.cls == "A":
        system_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": cond.system_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        system_text = None
    return system_text, full_text


class _LayerHookCapture:
    """Forward-hook context manager that captures `model.model.layers[L]`
    output for every requested layer L on EACH forward, clearing buffers
    per-call so probes don't leak across runs.

    Mirrors the `_get_last_token_activations` pattern in
    `scripts/issue404_predictor_cossim.py`: hook fires on the transformer
    block module and stashes `output[0] if isinstance(output, tuple) else
    output`. This captures the PRE-final-norm block output uniformly at
    EVERY layer — eliminating the `hidden_states[L+1]` path's post-norm
    quirk at the LAST layer (Qwen-2.5-7B: `hidden_states[28]` is
    post-final-norm output, NOT the pre-norm output of block 27 that
    `model.model.layers[27]` hook captures). GPU-verified on 2026-06-05.

    Usage:
        with _LayerHookCapture(model, layers) as cap:
            cap.reset()  # clear buffers before forward
            model(...)   # one or more forward passes
            tensor = cap.last_layer(L)  # (B, T, H) from the LAST forward
    """

    def __init__(self, model, layers: tuple[int, ...]):
        self._model = model
        self._layers = tuple(layers)
        self._captures: dict[int, list] = {L: [] for L in self._layers}
        self._handles: list = []

    def _make_hook(self, layer_idx: int):
        def _hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self._captures[layer_idx].append(hs.detach())

        return _hook

    def __enter__(self):
        # Bind hooks on the transformer block modules — `model.model.layers[L]`
        # is the canonical handle for HF Llama / Qwen2 architectures and
        # matches the #404 reference pattern.
        for L in self._layers:
            if len(self._model.model.layers) <= L:
                raise IndexError(
                    f"layer={L} out of range; model has "
                    f"{len(self._model.model.layers)} transformer blocks"
                )
            self._handles.append(
                self._model.model.layers[L].register_forward_hook(self._make_hook(L))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def reset(self) -> None:
        for L in self._layers:
            self._captures[L].clear()

    def last_layer(self, layer_idx: int):
        """Return the most-recent forward pass's full (B, T, H) tensor at
        the given layer. Raises if no forward has fired since the last reset.
        """
        buf = self._captures[layer_idx]
        if not buf:
            raise RuntimeError(
                f"_LayerHookCapture: no capture for layer={layer_idx} since last reset"
            )
        return buf[-1]


def _extract_one(  # noqa: C901 — dispatches across 3 extraction points; flattening would just inline the branches.
    model,
    tokenizer,
    *,
    device,
    cond,
    question: str,
    class_d_rewrites: dict,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    max_response_tokens: int,
    hook_capture: _LayerHookCapture | None = None,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict]:
    """For one (cond, question) extract residual activations at the requested
    extraction points × layers, via FORWARD HOOKS on `model.model.layers[L]`.

    Returns
    -------
    (result, meta)
      result: ``{point: {layer: tensor(H,) for layer in layers} for point in extraction_points}``.
        For ``end_of_system`` on non-Class-A conditions the inner dict is
        empty (signals N/A at this (cond, point); the cloud aggregator drops it).
      meta: ``{"truncated": bool, "response_len": int, "response_present": bool}``.
        `truncated` is True iff the greedy generation ran to
        `max_response_tokens` without emitting EOS — caller logs the rate
        so a bias toward early tokens in `mean_response` is visible.

    Mechanism
    ---------
    Uses forward hooks on `model.model.layers[L]` for ALL requested layers
    and ALL extraction points (round-6 fix). Reasoning, GPU-verified:

      * For Qwen-2.5-7B (28 layers, `len(hidden_states)==29`):
        `cos(norm(hook_on_layers[27]), hidden_states[28]) == 1.0` — meaning
        `hidden_states[28]` is the POST-final-norm output, NOT the pre-norm
        output of block 27 that #406's hook recipe captured.
      * `hidden_states[L+1]` equals the block-L hook output for L=0..26 but
        DIVERGES from it at L=27 (cosine diff ~0.16). That's what the
        cross-check caught.
      * Switching to hooks-everywhere makes ALL six layers (0/5/11/15/21/27)
        identical to #406's mechanism — eliminating the L27 post-norm
        quirk. L0..L26 are unchanged within bf16 noise (~1e-3 cosine diff
        vs the round-5 forward-hook capture is dominated by accumulation
        noise vs #406's original run, not a recipe change).

    The hook context manager (`_LayerHookCapture`) clears per-probe so
    captures don't leak across (cond, q) pairs. For shared model + repeated
    calls (the per-(cond, q) loop), the orchestrator owns ONE capture
    instance and passes it in via the `hook_capture` kwarg; otherwise we
    create + tear down locally (smoke / unit-test path).
    """
    import torch

    system_text, full_text = _build_prompts_for_extraction(
        cond, question, tokenizer, class_d_rewrites, "all"
    )

    result: dict[str, dict[int, torch.Tensor]] = {p: {} for p in extraction_points}
    meta: dict = {"truncated": False, "response_len": 0, "response_present": False}

    if hook_capture is None:
        # Local context — for one-shot calls in tests / smoke. Caller-shared
        # capture in run_extraction's loop avoids the hook re-bind cost.
        cm: _LayerHookCapture | None = _LayerHookCapture(model, layers)
        cm.__enter__()
        cap = cm
    else:
        cm = None
        cap = hook_capture
    try:
        # ── end_of_system (Class A only) ──
        if "end_of_system" in extraction_points and system_text is not None:
            ids = tokenizer(system_text, return_tensors="pt", add_special_tokens=False).to(device)
            cap.reset()
            with torch.no_grad():
                _ = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
            seq_len = ids["input_ids"].shape[1]
            last_pos = seq_len - 1
            for L in layers:
                hs = cap.last_layer(L)  # (B, T, H)
                assert hs.shape[0] == 1 and hs.shape[1] == seq_len, hs.shape
                result["end_of_system"][L] = hs[0, last_pos, :].float().cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── last_prompt + mean_response share one forward (with generation) ──
        need_last = "last_prompt" in extraction_points
        need_resp = "mean_response" in extraction_points
        if need_last or need_resp:
            prompt_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(
                device
            )
            prompt_len = prompt_ids["input_ids"].shape[1]

            if need_resp:
                # Greedy-decode (temp=0) — match the #460/#474 R-generation
                # convention. Capped at max_response_tokens; truncation rate
                # is tracked + logged at the run_extraction call site.
                with torch.no_grad():
                    gen_out = model.generate(
                        **prompt_ids,
                        max_new_tokens=max_response_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                full_ids = gen_out.sequences  # (1, prompt_len + n_new)
                response_len = full_ids.shape[1] - prompt_len
                meta["response_len"] = int(response_len)
                eos_id = tokenizer.eos_token_id
                last_new = int(full_ids[0, -1].item()) if response_len > 0 else None
                meta["truncated"] = bool(response_len == max_response_tokens and last_new != eos_id)
                meta["response_present"] = response_len > 0
                if response_len <= 0:
                    # Edge case: model emitted EOS immediately. mean_response
                    # → NaN this row (cloud aggregator drops NaN rows
                    # downstream). Still capture last_prompt via a single
                    # prompt-only forward.
                    logger.warning(
                        "cond=%s q=%r emitted zero response tokens; mean_response N/A this row",
                        cond.cid,
                        question[:40],
                    )
                    if need_last:
                        cap.reset()
                        with torch.no_grad():
                            _ = model(
                                input_ids=prompt_ids["input_ids"],
                                attention_mask=prompt_ids["attention_mask"],
                            )
                        for L in layers:
                            hs = cap.last_layer(L)
                            assert hs.shape[0] == 1 and hs.shape[1] == prompt_len, hs.shape
                            result["last_prompt"][L] = hs[0, prompt_len - 1, :].float().cpu()
                    if need_resp:
                        H = model.config.hidden_size
                        for L in layers:
                            result["mean_response"][L] = torch.full(
                                (H,), float("nan"), dtype=torch.float32
                            )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return result, meta

                # One teacher-forced forward pass over the FULL sequence
                # (prompt + decoded response) to populate hook captures at
                # every position — gives us last_prompt + mean_response in
                # one shot.
                attn = torch.ones_like(full_ids)
                cap.reset()
                with torch.no_grad():
                    _ = model(input_ids=full_ids, attention_mask=attn)
                full_len = full_ids.shape[1]
                for L in layers:
                    hs = cap.last_layer(L)[0]  # (full_len, H)
                    assert hs.shape[0] == full_len, hs.shape
                    if need_last:
                        result["last_prompt"][L] = hs[prompt_len - 1, :].float().cpu()
                    if need_resp:
                        resp_slice = hs[prompt_len : prompt_len + response_len, :]
                        result["mean_response"][L] = resp_slice.mean(dim=0).float().cpu()
                del full_ids, gen_out
            elif need_last:
                cap.reset()
                with torch.no_grad():
                    _ = model(
                        input_ids=prompt_ids["input_ids"],
                        attention_mask=prompt_ids["attention_mask"],
                    )
                for L in layers:
                    hs = cap.last_layer(L)
                    assert hs.shape[0] == 1 and hs.shape[1] == prompt_len, hs.shape
                    result["last_prompt"][L] = hs[0, prompt_len - 1, :].float().cpu()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)

    return result, meta


def run_extraction(  # noqa: C901 — top-level dispatcher (model load + per-(cond,q) loop + per-(point,layer) checkpointing).
    *,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    transformations: tuple[str, ...] | None,
    n_probes: int,
    max_response_tokens: int,
    device: str,
    overwrite: bool,
) -> dict[str, dict[int, np.ndarray]]:
    """Top-level extraction loop. Checkpoints per (extraction_point, layer)
    immediately on completion.

    Returns
    -------
    dict
      ``{point: {layer: ndarray(n_cond, n_q, H)}}`` — for ``end_of_system``,
      shape is ``(n_class_A, 1, H)`` after dropping non-A conds (which carry
      no system message). Non-A conds for end_of_system are explicitly
      ABSENT, never zero-filled.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CONDITIONS, CONDITIONS_BY_ID = _load_conditions_registry()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction; run on a pod with a GPU.")

    # Pick the active condition set.
    if transformations:
        active_conds = [CONDITIONS_BY_ID[c] for c in transformations]
    else:
        active_conds = list(CONDITIONS)
    logger.info("Active transformations: %s", [c.cid for c in active_conds])

    all_questions = _load_probe_questions()
    if n_probes < len(all_questions):
        questions = all_questions[:n_probes]
        logger.info("Subsetting probes: %d / %d", len(questions), len(all_questions))
    else:
        questions = all_questions

    class_d_rewrites = _ensure_class_d_rewrites() if any(c.cls == "D" for c in active_conds) else {}

    # Model load
    logger.info("Loading %s on %s", BASE_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    H = model.config.hidden_size

    # Aggregate clouds per (point, layer): list of (cond_idx, q_idx) -> vec
    # We store as nested dicts then convert to ndarray at write time.
    clouds: dict[str, dict[int, dict[tuple[int, int], np.ndarray]]] = {
        p: {L: {} for L in layers} for p in extraction_points
    }
    truncation_count = 0
    total_response_rows = 0

    response_len_samples: list[int] = []
    # Share ONE _LayerHookCapture across the whole (cond, q) loop so the
    # forward-hook handlers register once and tear down once at the end,
    # rather than per-row (cheaper + matches the #404 reference pattern).
    with _LayerHookCapture(model, layers) as hook_cap:
        for ci, cond in enumerate(active_conds):
            t_c = time.time()
            for qi, q in enumerate(questions):
                try:
                    row, meta = _extract_one(
                        model,
                        tokenizer,
                        device=device,
                        cond=cond,
                        question=q,
                        class_d_rewrites=class_d_rewrites,
                        extraction_points=extraction_points,
                        layers=layers,
                        max_response_tokens=max_response_tokens,
                        hook_capture=hook_cap,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Extraction failed at cond={cond.cid} q_idx={qi}: {e}"
                    ) from e
                for pt in extraction_points:
                    if pt == "end_of_system" and not row[pt]:
                        continue  # non-A cond → N/A by construction
                    for L in layers:
                        if L in row[pt]:
                            clouds[pt][L][(ci, qi)] = row[pt][L].numpy()
                if "mean_response" in extraction_points and meta.get("response_present"):
                    total_response_rows += 1
                    response_len_samples.append(meta["response_len"])
                    if meta.get("truncated"):
                        truncation_count += 1
            logger.info(
                "cond %d/%d %s in %.1fs",
                ci + 1,
                len(active_conds),
                cond.cid,
                time.time() - t_c,
            )

    if total_response_rows:
        med = int(np.median(response_len_samples)) if response_len_samples else 0
        mx = int(np.max(response_len_samples)) if response_len_samples else 0
        logger.info(
            "Response truncation rate: %d/%d (%.1f%%); response_len median=%d max=%d (cap=%d)",
            truncation_count,
            total_response_rows,
            100.0 * truncation_count / total_response_rows,
            med,
            mx,
            max_response_tokens,
        )
        # Persist the truncation summary so post-run review can see the
        # response-length distribution at a glance.
        BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            BAKEOFF_DIR / "extraction_truncation.json",
            {
                "schema_version": 1,
                "max_response_tokens": int(max_response_tokens),
                "total_response_rows": int(total_response_rows),
                "truncation_count": int(truncation_count),
                "truncation_rate": float(truncation_count / total_response_rows),
                "response_len_median": med,
                "response_len_max": mx,
                "response_len_p95": int(np.percentile(response_len_samples, 95))
                if response_len_samples
                else 0,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            },
        )

    # Convert and write per-(point, layer) checkpoints.
    written: dict[str, dict[int, np.ndarray]] = {}
    n_cond_active = len(active_conds)
    n_q = len(questions)
    for pt in extraction_points:
        written[pt] = {}
        for L in layers:
            entries = clouds[pt][L]
            if not entries:
                logger.warning("No activations captured for point=%s layer=%d", pt, L)
                continue
            # For end_of_system, the n_q axis collapses to 1 (input-independent
            # under causal attention; we always feed the same system-only
            # prefix per cond). For last_prompt / mean_response, n_q is full.
            if pt == "end_of_system":
                # Build (n_active_A, 1, H); only Class A conds will be present.
                present_cidx = sorted({ci for (ci, _qi) in entries})
                arr = np.full((len(present_cidx), 1, H), np.nan, dtype=np.float32)
                for new_i, ci in enumerate(present_cidx):
                    # qi must be 0 for end_of_system (one vec per cond)
                    # in case multiple qi rows snuck in, average them as a
                    # robustness no-op (they should be identical).
                    rows = [v for (cci, _qi), v in entries.items() if cci == ci]
                    arr[new_i, 0, :] = np.mean(rows, axis=0)
                cond_ids = [active_conds[ci].cid for ci in present_cidx]
            else:
                arr = np.full((n_cond_active, n_q, H), np.nan, dtype=np.float32)
                for (ci, qi), v in entries.items():
                    arr[ci, qi, :] = v
                cond_ids = [c.cid for c in active_conds]
            out_path = ACT_DIR / f"{pt}__layer{L}.pt"
            if out_path.exists() and not overwrite:
                # Subset-cache guard: refuse to use an on-disk checkpoint that
                # was extracted under a different cond_ids / n_probes than the
                # current run requests. Otherwise downstream regression silently
                # uses a stale (but seemingly valid) cache from a previous run.
                cached = torch.load(out_path, map_location="cpu", weights_only=False)
                cached_cond_ids = list(cached.get("cond_ids", []))
                cached_n_probes = int(cached.get("n_probes", -1))
                if cached_cond_ids != cond_ids or cached_n_probes != arr.shape[1]:
                    raise RuntimeError(
                        f"Subset-cache mismatch at {out_path}: cached "
                        f"cond_ids={cached_cond_ids} n_probes={cached_n_probes}, "
                        f"current request cond_ids={cond_ids} n_probes={arr.shape[1]}. "
                        "Re-run with --overwrite to invalidate the cache, or restore "
                        "the matching subset of transformations / --n-probes."
                    )
                logger.info(
                    "Skipping existing %s (matched subset; use --overwrite to redo)",
                    out_path,
                )
            else:
                ACT_DIR.mkdir(parents=True, exist_ok=True)
                # torch.save the numpy array + meta as a small dict.
                torch.save(
                    {
                        "schema_version": 1,
                        "extraction_point": pt,
                        "layer": L,
                        "cond_ids": cond_ids,
                        "n_probes": arr.shape[1],
                        "hidden_size": H,
                        "activations": arr,
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                    },
                    out_path,
                )
                logger.info("Wrote %s shape=%s", out_path, arr.shape)
            written[pt][L] = arr

    # Clean up GPU.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return written


def load_activations_from_disk(
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
) -> dict[str, dict[int, dict]]:
    """Re-load per-(point, layer) activation checkpoints written by run_extraction.

    Returns
    -------
    dict
      ``{point: {layer: {"cond_ids": [...], "activations": ndarray(n_cond, n_q, H)}}}``
    """
    import torch

    out: dict[str, dict[int, dict]] = {}
    for pt in extraction_points:
        out[pt] = {}
        for L in layers:
            p = ACT_DIR / f"{pt}__layer{L}.pt"
            if not p.exists():
                logger.warning("Missing checkpoint: %s", p)
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            out[pt][L] = d
    return out


def validate_canonical_completeness(
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    *,
    expected_cond_ids_for_point: dict[str, list[str] | None] | None = None,
) -> dict[str, dict[int, dict]]:
    """ROUND-4 holistic gate: assert every requested (pt, L) canonical exists
    AND its cond_ids set equals exactly the expected set for that point.

    This is the single invariant the metrics phase relies on, enforced on
    every aggregation entrypoint (serial #493 path, single-proc partitioned,
    single-proc non-partitioned, `--merge-only`, `--phase metrics|regress|all`
    over prior canonicals). Two complementary gates protect production:
    (a) `merge_partitioned_activations` raises at WRITE time on mismatch
    (round-3 wholly-missing + round-4 cache-bypass), and (b) THIS validator
    raises at READ time so any path that bypasses the merger (e.g. serial
    #493 writes canonical directly, `--skip-extract` over a stale tree,
    `--phase metrics` on a partial prior run) STILL trips before metrics.

    Parameters
    ----------
    extraction_points, layers
        The full requested grid.
    expected_cond_ids_for_point
        ``{point: expected_cond_ids_list | None}``. When the inner value is
        ``None`` for some point, that point is skipped (e.g. dev / smoke
        runs without an expected set). Production passes the Class-A
        subset for ``end_of_system`` and the full active cond set for
        ``last_prompt`` / ``mean_response``.

    Returns
    -------
    The same ``load_activations_from_disk`` output, but verified — any
    missing-file / cond-set-mismatch raises AssertionError instead of
    warning + skipping. Safe to feed straight into the metrics phase.
    """
    if expected_cond_ids_for_point is None:
        # Backwards-compat / dev shortcut: nothing to validate, fall back
        # to a warning-only load.
        return load_activations_from_disk(extraction_points, layers)

    import torch

    out: dict[str, dict[int, dict]] = {}
    failures: list[str] = []
    for pt in extraction_points:
        out[pt] = {}
        expected = expected_cond_ids_for_point.get(pt)
        for L in layers:
            p = ACT_DIR / f"{pt}__layer{L}.pt"
            if not p.exists():
                if expected is None:
                    continue  # caller said skip this point entirely
                failures.append(
                    f"missing canonical {p} for ({pt}, layer={L}); expected "
                    f"{sorted(expected)} (set size {len(expected)})"
                )
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            if expected is None:
                out[pt][L] = d
                continue
            expected_set = set(expected)
            cached_set = set(d.get("cond_ids", []))
            missing = expected_set - cached_set
            extra = cached_set - expected_set
            if missing or extra:
                failures.append(
                    f"({pt}, layer={L}): canonical cond_ids mismatch — "
                    f"missing_conds={sorted(missing)}, extra_conds={sorted(extra)}, "
                    f"expected={sorted(expected_set)}, found={sorted(cached_set)}"
                )
                continue
            out[pt][L] = d
    if failures:
        raise AssertionError(
            "Canonical-completeness gate FAILED before metrics phase — refusing "
            "to run regression on an under-sized / drifted bakeoff grid:\n  "
            + "\n  ".join(failures)
        )
    return out


# ───────────────────────── batched extraction (#502) ─────────────────────────
#
# #502 adds two capabilities on top of #493's serial extraction loop:
#
#   1. **Batched generation** (``--batch-size B``): batch B probes per
#      ``model.generate()`` call with left-padding so the residual hook
#      captures all B sequences in one forward, then we per-sequence
#      locate the response token positions (left-pad + per-seq EOS) and
#      mean-pool. Order-of-magnitude faster for ``mean_response`` (which
#      is the wall-clock bottleneck — one greedy decode per probe).
#
#   2. **Per-transformation activation files** (``--partition-tag`` /
#      ``--transformations``): each (cond, point, layer) lands in its OWN
#      file ``<point>__layer<L>__cond<cid>.pt``. This lets multiple GPU
#      processes write disjoint files in parallel; a single CPU
#      ``merge_partitioned_activations`` step then stacks them into the
#      canonical ``<point>__layer<L>.pt`` shape downstream code reads.
#
#      Determinism guarantee: per-GPU procs operate on DISJOINT cond
#      subsets, so the merged stack is bit-identical to a hypothetical
#      single-GPU run that did all conds in the same order — there are
#      no per-process random states that would diverge.
#
#   3. **Next-token-logits capture at last_prompt** for the
#      ``next_token_js`` baseline predictor (a *non-layer-indexed* output
#      metric that compares the two personas' softmax over the vocab at
#      the last_prompt position, averaged across probes — matches
#      ``scripts/issue458_predictor_jsdiv.py``). Captured DURING the
#      same forward pass that fills the ``last_prompt`` extraction point,
#      so it adds no extra GPU work.
#
# The serial #493 path (``run_extraction``) is unchanged. The dispatcher
# routes to ``run_extraction_batched`` only when ``--batch-size > 1``
# OR ``--partition-tag`` is set OR ``--probe-pool`` is set, all of which
# are #502-only flags.


def _next_token_logits_path(cid: str) -> Path:
    """Output path for one transformation's next-token logits sidecar.

    Stored under ``<bakeoff_root>/next_token_logits/last_prompt__cond<cid>.pt``
    with shape (n_q, V). One file per (transformation), produced during
    the same forward pass that fills ``last_prompt``. The next_token_js
    metric matrix is built from these in the metric phase.
    """
    return BAKEOFF_DIR / "next_token_logits" / f"last_prompt__cond{cid}.pt"


def _partitioned_activation_path(point: str, layer: int, cid: str) -> Path:
    """Output path for one transformation's per-(point, layer) activation file.

    Stored under ``<bakeoff_root>/activations/<point>__layer<L>__cond<cid>.pt``.
    Per-cond files let multiple GPU processes write disjoint files in
    parallel; the merger then stacks them.
    """
    return ACT_DIR / f"{point}__layer{layer}__cond{cid}.pt"


def _extract_batch(  # noqa: C901 — batched dispatcher; one branch per extraction point.
    model,
    tokenizer,
    *,
    device,
    cond,
    questions: list[str],
    class_d_rewrites: dict,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    max_response_tokens: int,
    hook_capture: _LayerHookCapture,
    capture_next_token_logits: bool,
) -> tuple[dict, dict, dict[str, dict[int, list]], dict]:
    """Batched analogue of ``_extract_one``: process B probes per ``generate()``.

    Returns
    -------
    (per_probe_rows, per_probe_meta, end_of_system_or_none, next_token_logits)
      per_probe_rows: ``{qi -> {point -> {layer -> tensor(H,)}}}`` for
        ``last_prompt`` and ``mean_response``. ``end_of_system`` is NOT
        included here — it's input-independent under causal attention, so
        the caller computes it ONCE per (cond) via ``_extract_one``.
      per_probe_meta: ``{qi -> {"truncated", "response_len", "response_present"}}``
      end_of_system_or_none: (unused — caller computes EOS once per cond)
      next_token_logits: ``{qi -> tensor(V,)}`` softmax probabilities at the
        last_prompt position. Empty dict if ``capture_next_token_logits=False``
        or if ``last_prompt`` was not requested.
    """
    import torch

    B = len(questions)
    if B == 0:
        return {}, {}, {}, {}

    # Build per-probe prompts. Class A's system prefix is the same for every
    # probe; non-A classes have no system prefix.
    full_texts: list[str] = []
    for q in questions:
        _sys_text, full_text = _build_prompts_for_extraction(
            cond, q, tokenizer, class_d_rewrites, "all"
        )
        full_texts.append(full_text)

    need_last = "last_prompt" in extraction_points
    need_resp = "mean_response" in extraction_points

    # Left-pad: generation needs left-padding so the rightmost token of every
    # sequence aligns at the same position in the batch. Tokenize WITHOUT
    # the chat template's special-token handling (we already applied it in
    # full_texts).
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        prompt_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
    finally:
        tokenizer.padding_side = orig_side
    prompt_ids = prompt_enc["input_ids"]  # (B, T_pad)
    attn_mask = prompt_enc["attention_mask"]  # (B, T_pad)
    T_pad = int(prompt_ids.shape[1])

    per_probe_rows: dict[int, dict[str, dict[int, torch.Tensor]]] = {
        qi: {p: {} for p in extraction_points} for qi in range(B)
    }
    per_probe_meta: dict[int, dict] = {qi: {} for qi in range(B)}
    next_token_logits: dict[int, torch.Tensor] = {}

    if need_resp:
        # Batched greedy generate. Hooks fire during decode; we re-run a
        # teacher-forced forward over the full (prompt + response) sequence
        # for clean per-position activations (matches the serial path
        # exactly, where the hook captures during the teacher-forced
        # forward).
        #
        # NOTE: do NOT pass explicit position_ids to model.generate().
        # HF's generate() derives them internally from attention_mask in
        # the prefill step AND uses past_key_values during incremental
        # decode where the position index must come from cache_position,
        # NOT from a static `position_ids` tensor. Passing explicit
        # `position_ids` to generate() BREAKS decoded tokens (GPT-2 +
        # tiny-random verified 2026-06-05: serial token sequence
        # [256, 256, 875, 875] but batched-with-explicit-pos produced
        # [256, 256, 256, 256] — the model stops advancing the position
        # after step 1). The post-gen teacher-forced forward DOES need
        # explicit position_ids (no past_key_values, so position arange
        # is wrong under left-pad), and that's where we set them below.
        # (Round-1 review blocker #2 was originally framed as "needs
        # position_ids on generate()"; CPU smoke against _extract_one
        # surfaced the inverse — the round-1 framing was wrong.)
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_response_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        full_ids = gen_out.sequences  # (B, T_pad + n_new_max)
        T_total = int(full_ids.shape[1])
        # Per-sequence response start = T_pad (left-pad lines all prompts
        # right at column T_pad - 1, so response begins at T_pad).
        # Per-sequence response end: find the FIRST EOS at position >= T_pad,
        # OR the cap (T_total - 1).
        eos_id = tokenizer.eos_token_id
        per_seq_resp_len: list[int] = []
        per_seq_truncated: list[bool] = []
        for b in range(B):
            seq = full_ids[b, T_pad:T_total]
            # First EOS index within the generated region (relative).
            if eos_id is not None:
                eos_pos_rel = (seq == eos_id).nonzero(as_tuple=False)
                if eos_pos_rel.numel() > 0:
                    resp_len = int(eos_pos_rel[0].item()) + 1  # include EOS
                    truncated = False
                else:
                    resp_len = int(seq.shape[0])
                    truncated = True
            else:
                resp_len = int(seq.shape[0])
                truncated = True
            per_seq_resp_len.append(resp_len)
            per_seq_truncated.append(truncated)
            per_probe_meta[b]["response_len"] = resp_len
            per_probe_meta[b]["truncated"] = bool(truncated)
            per_probe_meta[b]["response_present"] = resp_len > 0

        # One teacher-forced forward over the full sequence with the same
        # left-padded attention extended for the response. This populates
        # the hook captures at every position, B at a time.
        full_attn = torch.cat([attn_mask, torch.ones_like(full_ids[:, T_pad:T_total])], dim=1)
        # CRITICAL: explicit position_ids = cumsum(attention_mask) - 1.
        # Without this, both GPT-2 (additive position embeddings) and Qwen-2
        # (RoPE) compute position indices as arange(0, T_total) which mis-
        # indexes left-padded sequences (the first REAL token gets index =
        # num_pad_tokens, not 0). The serial path has zero pad and hits
        # the correct positions naturally; the batched path must match by
        # construction. Captured 2026-06-05 via the batched-vs-serial
        # CPU smoke gate (cosine 0.55 < 0.999 floor on left-padded probes).
        full_pos = (full_attn.long().cumsum(dim=1) - 1).clamp(min=0)
        hook_capture.reset()
        with torch.no_grad():
            fwd = model(
                input_ids=full_ids,
                attention_mask=full_attn,
                position_ids=full_pos,
                output_hidden_states=False,
            )
        # Capture next-token logits at the last_prompt position
        # (T_pad - 1) BEFORE we move on. This is the same position
        # the last_prompt extraction reads its residuals from, so the
        # logits and the residuals are coherent.
        if capture_next_token_logits and need_last:
            # fwd.logits shape: (B, T_total, V)
            assert fwd.logits.shape[0] == B
            assert fwd.logits.shape[1] == T_total
            for b in range(B):
                # Softmax to a proper probability distribution in fp32 on CPU.
                logits_b = fwd.logits[b, T_pad - 1, :].float().cpu()
                probs_b = torch.softmax(logits_b, dim=-1)
                next_token_logits[b] = probs_b
        # Per-layer pull
        for L in layers:
            hs = hook_capture.last_layer(L)  # (B, T_total, H)
            assert hs.shape[0] == B and hs.shape[1] == T_total, hs.shape
            for b in range(B):
                if need_last:
                    per_probe_rows[b]["last_prompt"][L] = hs[b, T_pad - 1, :].float().cpu()
                if need_resp:
                    rlen = per_seq_resp_len[b]
                    if rlen > 0:
                        resp_slice = hs[b, T_pad : T_pad + rlen, :]
                        per_probe_rows[b]["mean_response"][L] = resp_slice.mean(dim=0).float().cpu()
                    else:
                        H = model.config.hidden_size
                        per_probe_rows[b]["mean_response"][L] = torch.full(
                            (H,), float("nan"), dtype=torch.float32
                        )
        del fwd, full_ids, gen_out
    elif need_last:
        # Prompt-only forward (no generation). Hooks capture at every
        # position; we read the last_prompt column for each sequence.
        # See the need_resp branch above for why explicit position_ids
        # are mandatory under left-padding.
        prompt_pos = (attn_mask.long().cumsum(dim=1) - 1).clamp(min=0)
        hook_capture.reset()
        with torch.no_grad():
            fwd = model(
                input_ids=prompt_ids,
                attention_mask=attn_mask,
                position_ids=prompt_pos,
                output_hidden_states=False,
            )
        if capture_next_token_logits:
            for b in range(B):
                logits_b = fwd.logits[b, T_pad - 1, :].float().cpu()
                probs_b = torch.softmax(logits_b, dim=-1)
                next_token_logits[b] = probs_b
        for L in layers:
            hs = hook_capture.last_layer(L)
            assert hs.shape[0] == B and hs.shape[1] == T_pad, hs.shape
            for b in range(B):
                per_probe_rows[b]["last_prompt"][L] = hs[b, T_pad - 1, :].float().cpu()
        for b in range(B):
            per_probe_meta[b]["response_present"] = False
            per_probe_meta[b]["response_len"] = 0
            per_probe_meta[b]["truncated"] = False
        del fwd
    else:
        # Neither last_prompt nor mean_response — nothing to do for batched.
        # Caller handles end_of_system once per cond via _extract_one.
        for b in range(B):
            per_probe_meta[b]["response_present"] = False
            per_probe_meta[b]["response_len"] = 0
            per_probe_meta[b]["truncated"] = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_probe_rows, per_probe_meta, {}, next_token_logits


def run_extraction_batched(  # noqa: C901 — top-level batched dispatcher; mirrors run_extraction's shape.
    *,
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    transformations: tuple[str, ...] | None,
    n_probes: int,
    max_response_tokens: int,
    device: str,
    overwrite: bool,
    pool_path: Path | None = None,
    pool_mode: str = "q_test_strict",
    batch_size: int = 1,
    capture_next_token_logits: bool = True,
    write_partitioned: bool = True,
) -> dict[str, dict[int, np.ndarray]]:
    """Batched + partition-aware analogue of ``run_extraction`` (#502).

    Differences vs ``run_extraction``:
      - ``pool_path``: load a custom probe pool (e.g. 500-probe pool with
        the q_test 50 as a prefix). When None, falls back to the 50-probe
        q_test pool (matches #493).
      - ``batch_size``: probes per ``model.generate()`` batch. ``=1``
        produces a serial loop (== ``run_extraction`` behavior modulo
        the fresh batched code path — equality is verified in smoke).
      - ``capture_next_token_logits``: also collect softmax probs at the
        last_prompt position for the ``next_token_js`` baseline metric.
        Sidecar files at ``<bakeoff_root>/next_token_logits/last_prompt__cond<cid>.pt``.
      - ``write_partitioned``: write one activation file per
        (point, layer, cond). After all GPU procs finish, the merger
        stacks them into ``<point>__layer<L>.pt``. Set to False for
        single-GPU runs that want the canonical file directly.

    Returns the same shape as ``run_extraction``: ``{point: {layer:
    ndarray(n_cond, n_q, H)}}``. When ``write_partitioned=True``,
    callers are expected to invoke ``merge_partitioned_activations``
    before the metrics / regression phase (the returned dict is the
    THIS-PROC contribution, not the full grid).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CONDITIONS, CONDITIONS_BY_ID = _load_conditions_registry()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for extraction; run on a pod with a GPU.")

    if transformations:
        active_conds = [CONDITIONS_BY_ID[c] for c in transformations]
    else:
        active_conds = list(CONDITIONS)
    logger.info("Active transformations (this proc): %s", [c.cid for c in active_conds])
    logger.info(
        "batch_size=%d capture_next_token_logits=%s write_partitioned=%s",
        batch_size,
        capture_next_token_logits,
        write_partitioned,
    )

    all_questions = _load_probe_questions(pool_path=pool_path, mode=pool_mode)
    if n_probes < len(all_questions):
        questions = all_questions[:n_probes]
        logger.info("Subsetting probes: %d / %d", len(questions), len(all_questions))
    else:
        questions = all_questions
    n_q = len(questions)

    class_d_rewrites = _ensure_class_d_rewrites() if any(c.cls == "D" for c in active_conds) else {}

    logger.info("Loading %s on %s", BASE_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    H = model.config.hidden_size

    # Per-cond, per-(point, layer) activation arrays. Filled as we go,
    # written per-cond (whether partitioned or canonical) so a mid-run
    # crash never loses prior cond work.
    truncation_count = 0
    total_response_rows = 0
    response_len_samples: list[int] = []

    # Need a SECOND _LayerHookCapture for the EOS forward (Class A only,
    # one per cond, system-only prefix — see _extract_one). We reuse the
    # SAME hook_cap by binding once and using reset() between forwards.
    with _LayerHookCapture(model, layers) as hook_cap:
        for ci, cond in enumerate(active_conds):
            t_c = time.time()
            # ── end_of_system: one forward per cond (Class A only) ──
            eos_vecs: dict[int, torch.Tensor] = {}
            if "end_of_system" in extraction_points and cond.cls == "A":
                try:
                    eos_row, _ = _extract_one(
                        model,
                        tokenizer,
                        device=device,
                        cond=cond,
                        question=questions[0],  # any q — system prefix is q-independent
                        class_d_rewrites=class_d_rewrites,
                        extraction_points=("end_of_system",),
                        layers=layers,
                        max_response_tokens=max_response_tokens,
                        hook_capture=hook_cap,
                    )
                    eos_vecs = eos_row.get("end_of_system", {})
                except Exception as e:
                    raise RuntimeError(
                        f"end_of_system extraction failed at cond={cond.cid}: {e}"
                    ) from e

            # ── last_prompt + mean_response: batched over questions ──
            point_layer_arrays: dict[str, dict[int, np.ndarray]] = {
                p: {L: np.full((n_q, H), np.nan, dtype=np.float32) for L in layers}
                for p in extraction_points
                if p != "end_of_system"
            }
            cond_nt_logits: list[torch.Tensor] | None = (
                [None] * n_q
                if (capture_next_token_logits and "last_prompt" in extraction_points)
                else None
            )

            if any(p in extraction_points for p in ("last_prompt", "mean_response")):
                for batch_start in range(0, n_q, batch_size):
                    batch_end = min(batch_start + batch_size, n_q)
                    batch_qs = questions[batch_start:batch_end]
                    try:
                        rows_b, meta_b, _eos_unused, nt_b = _extract_batch(
                            model,
                            tokenizer,
                            device=device,
                            cond=cond,
                            questions=batch_qs,
                            class_d_rewrites=class_d_rewrites,
                            extraction_points=tuple(
                                p for p in extraction_points if p != "end_of_system"
                            ),
                            layers=layers,
                            max_response_tokens=max_response_tokens,
                            hook_capture=hook_cap,
                            capture_next_token_logits=capture_next_token_logits,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Batched extraction failed at cond={cond.cid} "
                            f"batch=[{batch_start}:{batch_end}]: {e}"
                        ) from e
                    for b_local, qi_global in enumerate(range(batch_start, batch_end)):
                        for pt in extraction_points:
                            if pt == "end_of_system":
                                continue
                            for L in layers:
                                if L in rows_b[b_local][pt]:
                                    vec = rows_b[b_local][pt][L].numpy()
                                    point_layer_arrays[pt][L][qi_global, :] = vec
                        if "mean_response" in extraction_points:
                            meta = meta_b[b_local]
                            if meta.get("response_present"):
                                total_response_rows += 1
                                response_len_samples.append(meta["response_len"])
                                if meta.get("truncated"):
                                    truncation_count += 1
                        if cond_nt_logits is not None and b_local in nt_b:
                            cond_nt_logits[qi_global] = nt_b[b_local]

            # ── persist per-cond results (per CLAUDE.md checkpoint-per-phase) ──
            # last_prompt + mean_response per-cond per-layer.
            ACT_DIR.mkdir(parents=True, exist_ok=True)
            for pt in extraction_points:
                if pt == "end_of_system":
                    continue
                for L in layers:
                    arr = point_layer_arrays[pt][L]
                    payload = {
                        "schema_version": 1,
                        "extraction_point": pt,
                        "layer": L,
                        "cond_id": cond.cid,
                        "n_probes": int(arr.shape[0]),
                        "hidden_size": int(arr.shape[1]),
                        "activations_one_cond": arr,  # (n_q, H)
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                        "batched": True,
                        "batch_size": int(batch_size),
                    }
                    out_path = _partitioned_activation_path(pt, L, cond.cid)
                    if out_path.exists() and not overwrite:
                        logger.info(
                            "Skipping existing partitioned activation %s (use --overwrite to redo)",
                            out_path,
                        )
                    else:
                        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                        torch.save(payload, tmp)
                        tmp.replace(out_path)
                        logger.info("Wrote %s shape=(%d, %d)", out_path, arr.shape[0], arr.shape[1])

            # end_of_system one-vector-per-cond payload (Class A only).
            if "end_of_system" in extraction_points and cond.cls == "A" and eos_vecs:
                for L in layers:
                    if L not in eos_vecs:
                        continue
                    vec = eos_vecs[L].numpy().astype(np.float32)  # (H,)
                    arr_eos = vec[None, :]  # (1, H)
                    payload_eos = {
                        "schema_version": 1,
                        "extraction_point": "end_of_system",
                        "layer": L,
                        "cond_id": cond.cid,
                        "n_probes": 1,
                        "hidden_size": int(arr_eos.shape[1]),
                        "activations_one_cond": arr_eos,
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                    }
                    out_path = _partitioned_activation_path("end_of_system", L, cond.cid)
                    if out_path.exists() and not overwrite:
                        continue
                    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                    torch.save(payload_eos, tmp)
                    tmp.replace(out_path)

            # next-token logits sidecar for the next_token_js baseline.
            if cond_nt_logits is not None and any(v is not None for v in cond_nt_logits):
                nt_out = _next_token_logits_path(cond.cid)
                nt_out.parent.mkdir(parents=True, exist_ok=True)
                # Replace any None with NaN row of vocab size (None happens only
                # if a probe slot was skipped, which shouldn't occur here).
                vocab_size = next(v for v in cond_nt_logits if v is not None).shape[0]
                stacked = np.full((n_q, vocab_size), np.nan, dtype=np.float32)
                for qi, v in enumerate(cond_nt_logits):
                    if v is not None:
                        stacked[qi, :] = v.numpy()
                nt_payload = {
                    "schema_version": 1,
                    "extraction_point": "last_prompt",
                    "cond_id": cond.cid,
                    "n_probes": int(stacked.shape[0]),
                    "vocab_size": int(stacked.shape[1]),
                    "probs": stacked,  # (n_q, V) softmax probs
                    "git_sha": _git_sha(),
                    "timestamp_utc": _now_iso(),
                }
                tmp = nt_out.with_suffix(nt_out.suffix + ".tmp")
                torch.save(nt_payload, tmp)
                tmp.replace(nt_out)
                logger.info(
                    "Wrote next-token-logits %s shape=(%d, %d)",
                    nt_out,
                    stacked.shape[0],
                    stacked.shape[1],
                )

            logger.info(
                "cond %d/%d %s (cls=%s) batched in %.1fs",
                ci + 1,
                len(active_conds),
                cond.cid,
                cond.cls,
                time.time() - t_c,
            )

    if total_response_rows:
        med = int(np.median(response_len_samples)) if response_len_samples else 0
        mx = int(np.max(response_len_samples)) if response_len_samples else 0
        logger.info(
            "Response truncation rate: %d/%d (%.1f%%); response_len median=%d max=%d (cap=%d)",
            truncation_count,
            total_response_rows,
            100.0 * truncation_count / total_response_rows,
            med,
            mx,
            max_response_tokens,
        )
        BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            BAKEOFF_DIR / "extraction_truncation.json",
            {
                "schema_version": 1,
                "max_response_tokens": int(max_response_tokens),
                "total_response_rows": int(total_response_rows),
                "truncation_count": int(truncation_count),
                "truncation_rate": float(truncation_count / total_response_rows),
                "response_len_median": med,
                "response_len_max": mx,
                "response_len_p95": int(np.percentile(response_len_samples, 95))
                if response_len_samples
                else 0,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
                "batched": True,
                "batch_size": int(batch_size),
            },
        )

    # Clean up GPU.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # When write_partitioned=False the caller wants downstream phases to
    # find canonical <point>__layer<L>.pt files immediately (no separate
    # merge step / no external dispatcher). Per-cond files ARE always
    # written (so a mid-run crash doesn't lose work), but we merge them
    # into the canonical shape NOW for this single-proc case.
    # (Codex round-1 blocker #3: previously write_partitioned was IGNORED
    # — all writes hit _partitioned_activation_path regardless, so a
    # `--batched` without `--partitioned` invocation never produced the
    # canonical files the metrics phase loads, and metrics silently
    # failed with "Missing checkpoint".)
    if not write_partitioned:
        logger.info(
            "write_partitioned=False: merging this proc's per-cond "
            "files into canonical <point>__layer<L>.pt shape"
        )
        expected_cids_this_proc = [c.cid for c in active_conds]
        # For end_of_system, only Class A conds were actually written; pass
        # the Class-A subset as the expected set for that point to keep the
        # no-drop assertion meaningful (no false-fail on B/C/D conds).
        eos_expected = [c.cid for c in active_conds if c.cls == "A"]
        for pt in extraction_points:
            expected_for_pt = eos_expected if pt == "end_of_system" else expected_cids_this_proc
            merge_partitioned_activations(
                (pt,),
                layers,
                overwrite=overwrite,
                expected_cond_ids=expected_for_pt or None,
            )

    # Return THIS PROC's contribution (caller merges across procs).
    written: dict[str, dict[int, np.ndarray]] = {pt: {} for pt in extraction_points}
    return written


def merge_partitioned_activations(  # noqa: C901 — per-(point, layer) walker + no-drop assertions branch per group; flattening would split a single contract across helpers.
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
    *,
    overwrite: bool,
    expected_cond_ids: list[str] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Stack per-cond partitioned files into the canonical ``<point>__layer<L>.pt``
    shape downstream phases expect.

    Walks ``<ACT_DIR>/*__cond*.pt`` and groups by (point, layer). For each
    group, sorts cond_ids in their canonical CONDITIONS order, stacks the
    per-cond (n_q, H) arrays to (n_cond, n_q, H) (or (n_cond, 1, H) for
    end_of_system), and writes the merged ``<point>__layer<L>.pt``.

    Parameters
    ----------
    expected_cond_ids
        When provided, ASSERT exact set equality between the partitioned
        files found on disk for each (point, layer) and this expected set.
        Fail loud on any missing OR extra cond_id. This is the multi-GPU
        no-drop gate: a stale partial partition or a silently-failed GPU
        worker would otherwise yield e.g. a 15-cond merged stack that the
        regression then reads as if it were the full 16-cond grid.
        (Codex round-1 blocker #4.) For end_of_system pass the Class-A
        subset only; for last_prompt / mean_response pass the full active
        cond set. None = old behavior (use whatever's on disk; for
        backwards-compat with the CPU smoke's tiny synthetic merges).

    Determinism: cond ordering is the canonical i406 CONDITIONS order, so
    the merged stack is independent of which GPU process produced each
    per-cond file. NaN-check: any partitioned file with all-NaN rows
    (which would corrupt downstream metrics silently) is flagged loud.
    """
    import re

    import torch

    CONDITIONS, _ = _load_conditions_registry()

    cid_order = [c.cid for c in CONDITIONS]
    cid_order_index = {c: i for i, c in enumerate(cid_order)}
    written: dict[str, dict[int, np.ndarray]] = {pt: {} for pt in extraction_points}
    expected_set = set(expected_cond_ids) if expected_cond_ids is not None else None
    if not ACT_DIR.exists():
        logger.warning("merge_partitioned_activations: %s missing, nothing to do", ACT_DIR)
        if expected_set:
            raise AssertionError(
                f"merge_partitioned_activations: ACT_DIR {ACT_DIR} missing but "
                f"expected_cond_ids={sorted(expected_set)} were requested"
            )
        return written
    pattern = re.compile(r"^(?P<pt>[a-z_]+)__layer(?P<L>\d+)__cond(?P<cid>[A-Z]\d+)\.pt$")
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for p in sorted(ACT_DIR.glob("*__cond*.pt")):
        m = pattern.match(p.name)
        if not m:
            continue
        pt = m.group("pt")
        L = int(m.group("L"))
        cid = m.group("cid")
        if pt not in extraction_points or L not in layers:
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        grouped.setdefault((pt, L), {})[cid] = d

    # ROUND-3 FIX #1: iterate over EVERY requested (pt, L) pair, not just
    # the (pt, L) groups that happen to be in `grouped`. Before round 3 the
    # loop was `for (pt, L), per_cond in grouped.items()`, so a wholly-
    # missing (pt, L) (e.g. a GPU died before writing layer L, or
    # `--merge-only` over a partial partition set with NO files at all for
    # some layer) NEVER hit the no-drop assertion — `--merge-only` exited
    # clean and `load_activations_from_disk` only WARNed + skipped the
    # missing canonical checkpoint, so an "all 28 layers" run silently
    # became an incomplete layer profile. Now we iterate the full Cartesian
    # product and raise on any (pt, L) whose per-cond group is empty when
    # expected_cond_ids is set. (Codex round-2 blocker #1.)
    requested_pairs = [(pt, L) for pt in extraction_points for L in layers]
    for pt, L in requested_pairs:
        per_cond = grouped.get((pt, L), {})
        canonical_path = ACT_DIR / f"{pt}__layer{L}.pt"
        if canonical_path.exists() and not overwrite:
            # ROUND-4 FIX #1 (Critical): the cache-hit short-circuit must
            # validate the cached canonical against `expected_set` BEFORE
            # accepting it, REGARDLESS of whether the current run has any
            # per-cond files for (pt, L) on disk. Before round 4, a stale
            # canonical + zero per-cond files in the current run + no
            # --overwrite (the production default) silently reused the
            # stale (possibly under-sized) cache, masking a GPU-worker-
            # died case. Now: load + validate + raise on mismatch even
            # if `per_cond` is empty.
            d = torch.load(canonical_path, map_location="cpu", weights_only=False)
            if expected_set is not None:
                cached_set = set(d.get("cond_ids", []))
                missing = expected_set - cached_set
                extra = cached_set - expected_set
                if missing or extra:
                    raise AssertionError(
                        f"Cached canonical {canonical_path} fails no-drop assertion: "
                        f"missing_conds={sorted(missing)}, extra_conds={sorted(extra)}, "
                        f"expected={sorted(expected_set)}. The current run's per-cond "
                        f"files for ({pt}, layer={L}) had {len(per_cond)} entries "
                        f"({sorted(per_cond.keys())}). Pass --overwrite to re-merge, "
                        "or re-run extraction for the missing conds."
                    )
            logger.info(
                "Skipping merge for existing %s (cached cond_ids match expected; "
                "use --overwrite to redo)",
                canonical_path,
            )
            written[pt][L] = d["activations"]
            continue
        # Order conds canonically (only those present).
        present_cids = sorted(per_cond.keys(), key=lambda c: cid_order_index.get(c, 1_000_000))
        if not present_cids:
            # ROUND-3 FIX #1: wholly-missing (pt, L) raises when expected
            # is set. Before: the loop skipped this case because grouped
            # never contained it; now we explicitly check.
            if expected_set:
                raise AssertionError(
                    f"No partitioned per-cond files AT ALL for ({pt}, layer={L}) but "
                    f"expected {sorted(expected_set)} — likely a GPU worker died "
                    "before writing this layer / point. A silent miss here would "
                    "downstream become an incomplete layer profile; refusing to merge."
                )
            continue
        # No-drop assertion: the present cond set must EQUAL the expected one.
        if expected_set is not None:
            present_set = set(present_cids)
            missing = expected_set - present_set
            extra = present_set - expected_set
            if missing or extra:
                raise AssertionError(
                    f"Partition merge no-drop FAILED at ({pt}, layer={L}): "
                    f"missing_conds={sorted(missing)} extra_conds={sorted(extra)} "
                    f"present={present_cids} expected={sorted(expected_set)}. "
                    "A stale partial partition would silently yield an under-sized "
                    "bakeoff grid; refusing to merge."
                )
        first = per_cond[present_cids[0]]["activations_one_cond"]
        n_q = first.shape[0]
        H = first.shape[1]
        for cid in present_cids:
            arr = per_cond[cid]["activations_one_cond"]
            if arr.shape != (n_q, H):
                raise AssertionError(
                    f"Shape mismatch in partitioned activations: "
                    f"cond={cid} (n_q={arr.shape[0]}, H={arr.shape[1]}) "
                    f"vs first ({n_q}, {H}) for ({pt}, layer={L})"
                )
            if np.all(np.isnan(arr)):
                raise AssertionError(
                    f"All-NaN partitioned activation at cond={cid} ({pt}, layer={L}); "
                    "the per-GPU extraction silently failed for this cond"
                )
        stacked = np.stack([per_cond[c]["activations_one_cond"] for c in present_cids], axis=0)
        payload = {
            "schema_version": 1,
            "extraction_point": pt,
            "layer": L,
            "cond_ids": present_cids,
            "n_probes": int(n_q),
            "hidden_size": int(H),
            "activations": stacked,  # (n_cond, n_q, H)
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
            "merged_from": [str(_partitioned_activation_path(pt, L, c).name) for c in present_cids],
        }
        tmp = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(canonical_path)
        logger.info(
            "Merged %d cond files into %s shape=%s",
            len(present_cids),
            canonical_path,
            stacked.shape,
        )
        written[pt][L] = stacked
    return written


def load_next_token_logits() -> dict[str, np.ndarray]:
    """Re-load next-token logits sidecars produced by ``run_extraction_batched``.

    Returns ``{cond_id: ndarray(n_q, V)}`` of softmax probabilities at
    last_prompt. Empty dict if no sidecars on disk.
    """
    import re

    import torch

    out: dict[str, np.ndarray] = {}
    nt_dir = BAKEOFF_DIR / "next_token_logits"
    if not nt_dir.exists():
        return out
    pattern = re.compile(r"^last_prompt__cond(?P<cid>[A-Z]\d+)\.pt$")
    for p in sorted(nt_dir.glob("last_prompt__cond*.pt")):
        m = pattern.match(p.name)
        if not m:
            continue
        cid = m.group("cid")
        d = torch.load(p, map_location="cpu", weights_only=False)
        out[cid] = d["probs"]
    return out


def _js_divergence_rowwise(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row JS divergence (base 2, bounded [0, 1]) along the LAST dim.

    Matches ``scripts/issue458_predictor_jsdiv.py._js_divergence`` so the
    next_token_js metric is apples-to-apples with #458's predictor.
    Inputs ``p, q`` are (..., V); returns ``p.shape[:-1]``.
    """
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = 0.5 * (p + q)
    ln2 = np.log(2.0)
    kl_pm = (p * (np.log(p) - np.log(m))).sum(axis=-1) / ln2
    kl_qm = (q * (np.log(q) - np.log(m))).sum(axis=-1) / ln2
    js = 0.5 * (kl_pm + kl_qm)
    return np.clip(js, 0.0, 1.0)


def compute_next_token_js_matrix(
    cid_to_probs: dict[str, np.ndarray],
    *,
    cond_ids_order: list[str] | None = None,
) -> dict:
    """Build the (n_cond × n_cond) next-token JS distance matrix.

    For each ordered pair (i, j): per-probe JS between cid_to_probs[i]
    and cid_to_probs[j], averaged over probes. ``cond_ids_order`` fixes
    the row/column order (defaults to sorted keys, then canonical
    CONDITIONS order if available).

    Returns a payload with the same schema as ``_compute_metric_matrix``
    so the regression pipeline reads it uniformly: ``{matrix: {a: {b: float}}, ...}``.
    """
    CONDITIONS, _ = _load_conditions_registry()

    if cond_ids_order is None:
        canonical = [c.cid for c in CONDITIONS]
        cond_ids_order = [c for c in canonical if c in cid_to_probs]
        # Append any present cids not in canonical (shouldn't happen).
        for c in sorted(cid_to_probs):
            if c not in cond_ids_order:
                cond_ids_order.append(c)
    matrix: dict[str, dict[str, float]] = {
        a: {b: 0.0 for b in cond_ids_order} for a in cond_ids_order
    }
    per_pair_per_probe_summary: list[dict] = []
    for i, a in enumerate(cond_ids_order):
        pa = cid_to_probs[a]
        for j, b in enumerate(cond_ids_order):
            if i == j:
                matrix[a][b] = 0.0
                continue
            pb = cid_to_probs[b]
            if pa.shape != pb.shape:
                raise AssertionError(
                    f"next-token logits shape mismatch: cid={a} {pa.shape} vs cid={b} {pb.shape}"
                )
            # Per-probe JS, then mean over probes (drop NaN rows).
            mask = ~(np.any(np.isnan(pa), axis=1) | np.any(np.isnan(pb), axis=1))
            if mask.sum() == 0:
                matrix[a][b] = float("nan")
                continue
            js_per_probe = _js_divergence_rowwise(pa[mask], pb[mask])
            matrix[a][b] = float(js_per_probe.mean())
            per_pair_per_probe_summary.append(
                {
                    "a": a,
                    "b": b,
                    "n_probes_used": int(mask.sum()),
                    "mean_js": float(js_per_probe.mean()),
                    "median_js": float(np.median(js_per_probe)),
                    "min_js": float(np.min(js_per_probe)),
                    "max_js": float(np.max(js_per_probe)),
                }
            )
    return {
        "schema_version": 1,
        "extraction_point": "last_prompt",  # output metric tied to last_prompt slot
        "layer": -1,  # NOT layer-indexed — sentinel value distinct from real layers
        "metric": "next_token_js",
        "variant": "raw",
        "cond_ids": cond_ids_order,
        "matrix": matrix,
        "per_pair_summary": per_pair_per_probe_summary,
        "git_sha": _git_sha(),
        "timestamp_utc": _now_iso(),
    }


def write_next_token_js_matrix(*, enforce_cross_check: bool = True) -> Path | None:
    """Build the next-token JS matrix from sidecar files, write it to disk,
    AND run the #406 cross-check fail-loud (matching the cosine cross-check's
    guard discipline).

    Output (always when sidecars present):
      ``<METRIC_DIR>/last_prompt__layer-1__next_token_js__raw.json``
      ``<METRIC_DIR>/last_prompt__layer-1__next_token_js__raw__cross_check_406.json``

    The layer=-1 sentinel distinguishes the OUTPUT metric (one scalar per
    pair) from the per-layer activation metrics.

    Parameters
    ----------
    enforce_cross_check
        When True (default, production): if the #406 reference matrix is
        present on disk, REQUIRE the rank-correlation cross-check to pass
        (raises AssertionError on fail). When False (smoke / tests where
        the #406 reference is intentionally synthetic or absent): only LOG
        the result, don't raise. (Codex round-1 blocker #1: this function
        existed in isolation; nothing in main()/the aggregation pipeline
        called the cross-check, so the JS baseline shipped unguarded.)

    Returns the matrix path written, or None if no sidecars on disk.
    """
    cid_to_probs = load_next_token_logits()
    if not cid_to_probs:
        logger.warning("No next-token logits sidecars on disk; skipping next_token_js")
        return None
    payload = compute_next_token_js_matrix(cid_to_probs)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRIC_DIR / "last_prompt__layer-1__next_token_js__raw.json"
    _write_json_atomic(out_path, payload)
    logger.info("Wrote next_token_js matrix to %s", out_path)

    # ── #406 cross-check (production-guarded) ──
    # The cosine cross-check fails the whole run on a recipe drift; the JS
    # baseline gets the same treatment so a drifted recipe can't ship to
    # the regression unnoticed. When the #406 reference is missing this is
    # a no-op; when present, fail-loud per the safety-net discipline.
    cross_check_path = METRIC_DIR / "last_prompt__layer-1__next_token_js__raw__cross_check_406.json"
    try:
        summary = cross_check_next_token_js_against_406(payload)
    except AssertionError as e:
        _write_json_atomic(
            cross_check_path,
            {
                "schema_version": 1,
                "failed": True,
                "failure_reason": str(e),
                "rank_corr_floor": float(COSINE_406_JS_RANK_CORR_FLOOR),
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            },
        )
        if enforce_cross_check:
            raise
        logger.warning("next_token_js cross-check FAIL (enforce=False, continuing): %s", e)
    else:
        _write_json_atomic(
            cross_check_path,
            {
                "schema_version": 1,
                "failed": False,
                "summary": summary,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            },
        )
    return out_path


COSINE_406_JS_RANK_CORR_FLOOR: float = 0.5


def cross_check_next_token_js_against_406(
    nt_js_payload: dict,
    cosine_ids: list[str] | None = None,
) -> dict:
    """Cross-check the recomputed next_token_js against #406's D_matrix.json["JS"].

    METHODOLOGY NOTE — IMPORTANT:
    Our next_token_js is a SINGLE-POSITION (last_prompt slot) JS over the
    full vocabulary. #406's D_matrix.json["JS"] is a SEQUENCE-LEVEL teacher-
    forced JS averaged across response positions, top-k truncated to k=25
    of the union top-k vocab per response. They are DIFFERENT operational-
    izations of "next-token disagreement" and should NOT be expected to be
    byte-equal.

    The cross-check is therefore a SHAPE check: rank correlation between
    the two over the 240-pair grid must clear ``COSINE_406_JS_RANK_CORR_FLOOR``.
    A genuine extraction bug (prompt mis-build, wrong position index, wrong
    persona pairing) would crash the rank correlation; numerical differences
    in the two operationalizations stay above the floor.

    Returns a diff summary; raises AssertionError on rank-correlation
    floor failure.
    """
    from scipy.stats import spearmanr

    p406_path = PROJECT_ROOT / "eval_results/issue_406/divergence/D_matrix.json"
    if not p406_path.exists():
        logger.warning("Missing %s — skipping next_token_js cross-check", p406_path)
        return {"ok": True, "reason": "no #406 reference available"}
    p406 = json.loads(p406_path.read_text())
    p406_js = p406["JS"]
    nt_matrix = nt_js_payload["matrix"]
    cond_ids = list(nt_matrix.keys())
    pairs = [(a, b) for a in cond_ids for b in cond_ids if a != b]
    ours: list[float] = []
    theirs: list[float] = []
    skipped = 0
    for a, b in pairs:
        if a not in p406_js or b not in p406_js[a]:
            skipped += 1
            continue
        t = p406_js[a][b]
        if t is None:
            skipped += 1
            continue
        o = nt_matrix[a][b]
        if o is None or not np.isfinite(o):
            skipped += 1
            continue
        ours.append(float(o))
        theirs.append(float(t))
    if len(ours) < 10:
        logger.warning(
            "Cross-check has only %d pairs in common; reporting but not enforcing", len(ours)
        )
        return {
            "ok": True,
            "reason": f"too few pairs ({len(ours)}) to enforce rank-correlation floor",
            "n_pairs": len(ours),
        }
    rho, p = spearmanr(np.array(ours), np.array(theirs))
    ok = bool(rho >= COSINE_406_JS_RANK_CORR_FLOOR)
    diff_arr = np.abs(np.array(ours) - np.array(theirs))
    summary = {
        "n_pairs_checked": len(ours),
        "n_pairs_skipped": int(skipped),
        "rank_corr_spearman": float(rho),
        "spearman_p_value": float(p),
        "rank_corr_floor": float(COSINE_406_JS_RANK_CORR_FLOOR),
        "ok": ok,
        "max_abs_diff": float(diff_arr.max()),
        "mean_abs_diff": float(diff_arr.mean()),
        "note": (
            "Methodology: ours = SINGLE-POSITION (last_prompt) full-vocab JS; "
            "#406's = SEQUENCE-LEVEL teacher-forced top-k=25 JS averaged across "
            "response positions. Different operationalizations of next-token "
            "disagreement; checked via rank correlation, not absolute equality."
        ),
    }
    if not ok:
        raise AssertionError(
            f"next_token_js vs #406 JS rank correlation {rho:.3f} below floor "
            f"{COSINE_406_JS_RANK_CORR_FLOOR}; "
            "fresh recipe diverges from #406 in a load-bearing way."
        )
    logger.info(
        "next_token_js cross-check vs #406: Spearman rho=%.3f (p=%.2e) over %d pairs (ok=%s)",
        rho,
        p,
        len(ours),
        ok,
    )
    return summary


# ───────────────────────── metric phase ─────────────────────────


def _pca_topk_via_gram(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Dual / Gram PCA — eigendecompose the n×n Gram, NOT the d×d covariance.

    Safe when n ≪ d (our regime: n = 50 ≪ d = 3584). Returns
    (projected: (n, k), components: (k, d)) for X already mean-centered.

    For X centered with shape (n, d):
      - Gram G = X X^T  (n × n)
      - Eigenvectors V_g (n, k), eigenvalues λ
      - The top-k principal components in the d-space:
            U = X^T V_g / sqrt(λ)        shape (d, k)
      - Projected coords:
            T = V_g * sqrt(λ)            shape (n, k)
    """
    n, d = X.shape
    if k > min(n, d):
        k = min(n, d)
    G = X @ X.T  # (n, n)
    # numerical safety: symmetrize
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    # take top-k by eigenvalue
    order = np.argsort(eigvals)[::-1][:k]
    lam = np.clip(eigvals[order], 1e-12, None)
    V_g = eigvecs[:, order]  # (n, k)
    sqrt_lam = np.sqrt(lam)  # (k,)
    components = (X.T @ V_g) / sqrt_lam[None, :]  # (d, k)
    components = components.T  # (k, d)
    projected = V_g * sqrt_lam[None, :]  # (n, k)
    return projected, components


def _pair_pca_subspace(Xa: np.ndarray, Xb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduce a PAIR of clouds to a shared top-k subspace via the Gram of
    the stacked centered clouds.

    Returns (Ya: (na, k), Yb: (nb, k)) — never inverts a d×d covariance.
    """
    stacked = np.vstack([Xa, Xb])
    mu = stacked.mean(axis=0, keepdims=True)
    stacked_c = stacked - mu
    _proj, comps = _pca_topk_via_gram(stacked_c, k)
    Ya = (Xa - mu) @ comps.T  # (na, k)
    Yb = (Xb - mu) @ comps.T  # (nb, k)
    return Ya, Yb


def _drop_nan_rows(X: np.ndarray) -> np.ndarray:
    """Drop rows that contain any NaN (occurs when a (cond, q) extraction
    failed or end_of_system N/A was zero-filled by mistake)."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}")
    mask = ~np.any(np.isnan(X), axis=1)
    return X[mask]


def _centroid_cosine_distance(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """1 − cos_sim between cloud centroids. Reproduces the #406 cosine recipe.

    Both clouds are (n_q, H); centroid = mean across the n_q axis.
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    mu_a = Xa.mean(axis=0)
    mu_b = Xb.mean(axis=0)
    na = np.linalg.norm(mu_a)
    nb = np.linalg.norm(mu_b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - (mu_a @ mu_b) / (na * nb))


def _centroid_euclidean(Xa: np.ndarray, Xb: np.ndarray) -> float:
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    return float(np.linalg.norm(Xa.mean(axis=0) - Xb.mean(axis=0)))


def _centroid_mahal(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Mahalanobis-on-pooled-cov in the top-k PCA subspace (n≪d-safe).

    Pooled subspace via dual PCA on stacked-centered (Xa, Xb), then pooled
    covariance in the k-d subspace + inverse via solve. Requires na, nb ≥ 2
    (per-pair within-cloud covariance defined); for one-vector-per-cloud
    extraction (end_of_system) use _context_mahal_with_pooled_cov instead.
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Cov = 0.5 * (np.cov(Ya.T, ddof=1) + np.cov(Yb.T, ddof=1))
    # ridge for numerical stability
    Cov += 1e-6 * np.eye(Cov.shape[0])
    diff = mu_a - mu_b
    inv_cov_diff = np.linalg.solve(Cov, diff)
    return float(np.sqrt(float(diff @ inv_cov_diff)))


# Eigenvalue / condition-number gate for the context-pooled covariance.
# These bite BEFORE the 1e-6 ridge, so a genuinely rank-deficient pooled
# cov (e.g. n_cond=5 in a k=16 PCA subspace, or every centroid collapsed
# to the same point) cannot be silently regularized into a finite — but
# meaningless — Mahalanobis distance.
POOLED_COV_EIG_FLOOR: float = 1e-10  # smallest non-trivial eigenvalue
POOLED_COV_COND_CEIL: float = 1e10  # max acceptable condition number


def _build_context_pooled_mahal_state(activations: np.ndarray, k: int) -> dict | None:
    """Build the shared state for Mahalanobis-vs-context-pooled-covariance,
    used by the end_of_system extraction point where there is ONE vector per
    transformation (no per-pair covariance possible). The "pooled" covariance
    is computed across all condition centroids; PCA-reduced to the top-k
    subspace to stay well-posed at n_cond << d.

    activations shape: (n_cond, n_q, H). For end_of_system n_q == 1, so the
    per-cond vector IS the centroid.

    Failure modes (return None, caller writes an explicit N/A row with the
    reason in the metric payload):
      - n_cond < 2 (nothing to pool)
      - all centroids collapse to one point (cov_sub eigenvalues ≈ 0)
      - cov_sub is rank-deficient before the ridge — smallest eigenvalue
        below POOLED_COV_EIG_FLOOR or condition number above
        POOLED_COV_COND_CEIL. The 1e-6 ridge is appropriate for MILD
        ill-conditioning (numerical noise on a full-rank cov), NOT for
        rank-deficiency: ridging a degenerate cov produces a finite but
        spurious Mahalanobis. With n_cond=5 (the end_of_system Class-A
        subpanel) in a k=16 PCA subspace the pooled cov is structurally
        rank-deficient (rank ≤ n_cond - 1 = 4); k_eff caps to 4 here, but
        if a future caller passes a smaller --pca-k that produces a
        genuinely rank-deficient projection this gate catches it.

    Failure-reason side channel: the None-return contract is preserved
    (returns `dict | None`), so on EVERY failure path the function ALSO
    stashes a one-line reason in the module-level
    `_LAST_POOLED_FAILURE_REASON` dict, keyed by `id(activations)`. The
    caller fetches it via `_pop_pooled_failure_reason(activations)`
    immediately after a None return — that pop both reads the reason
    AND clears the entry so the dict can't leak under bursty calls.
    """
    n_cond = activations.shape[0]
    if n_cond < 2:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"n_cond={n_cond} < 2: nothing to pool across"
        )
        return None
    # Build centroid matrix (n_cond, H).
    centroids = np.array([_drop_nan_rows(activations[i]).mean(axis=0) for i in range(n_cond)])
    valid_mask = ~np.any(np.isnan(centroids), axis=1)
    centroids = centroids[valid_mask]
    if len(centroids) < 2:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"only {len(centroids)} non-NaN centroids: pool undefined"
        )
        return None
    mu = centroids.mean(axis=0, keepdims=True)
    cent_c = centroids - mu
    # Dual / Gram PCA on the n_cond-cloud — never invert (3584, 3584).
    k_eff = min(k, len(centroids) - 1, cent_c.shape[1])
    if k_eff < 1:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = f"k_eff={k_eff} < 1: subspace too small"
        return None
    _proj, comps = _pca_topk_via_gram(cent_c, k_eff)
    Y = cent_c @ comps.T  # (n_cond_valid, k_eff)
    # Pooled covariance in the subspace BEFORE the ridge. `np.atleast_2d`
    # defends against the n_cond=2 / k_eff=1 collapse: with 2 contexts the
    # PCA-reduced dim is 1 and `np.cov(Y.T, ddof=1)` returns a 0-d scalar,
    # which then crashes the eigvalsh gate. Wrapping to 2-d preserves the
    # shape contract end-to-end. Defensive — the headline run uses n_cond=5
    # for end_of_system (Class A) — but `--transformations A1 A2` subset
    # smoke runs would otherwise crash here.
    cov_sub_raw = np.atleast_2d(np.cov(Y.T, ddof=1))
    eigvals = np.linalg.eigvalsh(0.5 * (cov_sub_raw + cov_sub_raw.T))
    eig_min = float(np.min(eigvals))
    eig_max = float(np.max(eigvals))
    # 1) all-collapsed centroids → eig_max ≈ 0
    if eig_max < POOLED_COV_EIG_FLOOR:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"pooled cov rank-0 (max eigenvalue {eig_max:.2e} < "
            f"{POOLED_COV_EIG_FLOOR:.0e}): centroids are collinear / collapsed"
        )
        return None
    # 2) rank-deficient — smallest eigenvalue at machine zero (some
    # subspace direction is fully degenerate; ridging would invent
    # variance there).
    if eig_min < POOLED_COV_EIG_FLOOR:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"pooled cov rank-deficient (min eigenvalue {eig_min:.2e} < "
            f"{POOLED_COV_EIG_FLOOR:.0e}) at k_eff={k_eff}; ridging would "
            f"invent variance along a degenerate direction. Reduce --pca-k "
            f"or extract more contexts."
        )
        return None
    # 3) Borderline ill-conditioned — flag but still ridge.
    cond_num = eig_max / max(eig_min, np.finfo(np.float64).tiny)
    if cond_num > POOLED_COV_COND_CEIL:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"pooled cov ill-conditioned (cond={cond_num:.2e} > "
            f"{POOLED_COV_COND_CEIL:.0e}): refusing to ridge into a spurious "
            f"finite inverse"
        )
        return None
    cov_sub = cov_sub_raw + 1e-6 * np.eye(k_eff)
    try:
        cov_inv = np.linalg.inv(cov_sub)
    except np.linalg.LinAlgError as e:
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            f"np.linalg.inv raised LinAlgError after ridge: {e}"
        )
        return None
    if not np.all(np.isfinite(cov_inv)):
        _LAST_POOLED_FAILURE_REASON[id(activations)] = (
            "post-ridge inverse has non-finite entries (overflow)"
        )
        return None
    return {
        "mu": mu,
        "components": comps,
        "cov_inv": cov_inv,
        "valid_mask": valid_mask,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "condition_number": cond_num,
    }


# Side-channel for the most-recent _build_context_pooled_mahal_state failure
# reason, keyed by id(activations) so the caller can fetch it without changing
# the None-return contract. Trimmed opportunistically when callers fetch.
_LAST_POOLED_FAILURE_REASON: dict[int, str] = {}


def _pop_pooled_failure_reason(activations: np.ndarray) -> str | None:
    """Pop the most recent pooled-cov failure reason for this activations
    array. Returns the reason string (or None if not recorded)."""
    return _LAST_POOLED_FAILURE_REASON.pop(id(activations), None)


def _context_mahal_with_pooled_cov(
    Xa: np.ndarray,
    Xb: np.ndarray,
    state: dict,
    a_idx: int,
    b_idx: int,
) -> float:
    """Mahalanobis distance between the centroids of (Xa, Xb), using the
    pre-built context-pooled covariance in the shared subspace.

    For end_of_system where Xa / Xb each have a single vector this is the
    only meaningful Mahalanobis variant (per-pair covariance is undefined
    when n=1). Returns NaN if either context dropped out of the valid_mask.
    """
    mask = state["valid_mask"]
    if not (mask[a_idx] and mask[b_idx]):
        return float("nan")
    mu = state["mu"]
    comps = state["components"]
    cov_inv = state["cov_inv"]
    cent_a = _drop_nan_rows(Xa).mean(axis=0)
    cent_b = _drop_nan_rows(Xb).mean(axis=0)
    if np.any(np.isnan(cent_a)) or np.any(np.isnan(cent_b)):
        return float("nan")
    ya = (cent_a - mu[0]) @ comps.T
    yb = (cent_b - mu[0]) @ comps.T
    diff = ya - yb
    return float(np.sqrt(float(diff @ cov_inv @ diff)))


def _rbf_kernel_with_bandwidth(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the RBF kernel matrix with median-heuristic bandwidth.

    Returns (K, sigma2) where K[i,j] = exp(-||z_i - z_j||^2 / sigma2) and
    sigma2 is the median pairwise squared distance (excluding the diagonal).
    """
    sq = np.sum(Z**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2 * Z @ Z.T
    np.fill_diagonal(D2, np.nan)
    median_sq = np.nanmedian(D2)
    sigma2 = max(float(median_sq), 1e-8)
    K = np.exp(-D2 / sigma2)
    np.fill_diagonal(K, 1.0)
    return K, sigma2


def _unbiased_mmd2_from_kernel(K: np.ndarray, na: int) -> float:
    """Unbiased MMD² (Gretton et al. 2012, Lemma 6) from a pre-built kernel.

    Excludes the diagonal of K_aa and K_bb so the estimator is unbiased:
        MMD² = (1/(na(na-1))) sum_{i!=j} K_aa[i,j]
             + (1/(nb(nb-1))) sum_{i!=j} K_bb[i,j]
             - (2/(na*nb)) sum_{i,j} K_ab[i,j]
    """
    nb = K.shape[0] - na
    Kaa = K[:na, :na]
    Kbb = K[na:, na:]
    Kab = K[:na, na:]
    sum_aa = Kaa.sum() - np.trace(Kaa)  # off-diagonal sum
    sum_bb = Kbb.sum() - np.trace(Kbb)
    term_aa = sum_aa / (na * (na - 1))
    term_bb = sum_bb / (nb * (nb - 1))
    term_ab = 2 * Kab.mean()
    return float(term_aa + term_bb - term_ab)


def _rbf_mmd_squared(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Unbiased RBF-MMD² with median-heuristic bandwidth (Gretton et al. 2012).

    The unbiased estimator can go slightly negative under H0 (same
    distribution); that is the canonical behaviour, not a bug. Pair-level
    permutation-null is built separately by `_mmd_permutation_summary`
    because computing it per-pair (240 pairs * MMD_PERMUTATIONS) would
    dominate wall-clock for marginal scientific value — we instead build
    one shared null per (extraction_point, layer) from a uniform subsample
    of pairs (same bandwidth, so the null shape is shared).
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    na, nb = len(Xa), len(Xb)
    if na < 2 or nb < 2:
        return float("nan")
    Z = np.vstack([Xa, Xb])
    K, _sigma2 = _rbf_kernel_with_bandwidth(Z)
    return _unbiased_mmd2_from_kernel(K, na)


def _mmd_permutation_summary(
    activations: np.ndarray,
    cond_ids: list[str],
    *,
    n_perm: int,
    variant: str,
    n_pair_samples: int = 16,
    rng=None,
) -> dict:
    """Build a shared permutation null for MMD² across a random subsample of
    (i, j) pairs at one (extraction_point, layer, variant).

    For each sampled pair, computes the observed unbiased MMD² and the
    permutation null distribution (relabel-and-recompute). Returns the
    aggregate per-pair p-values + the pooled null summary. Caller uses
    this to read significance per pair; the predictor SCALAR remains the
    observed MMD² in the main matrix file.

    n_pair_samples is capped so this stays bounded (~60s for 16 pairs *
    200 perms on n=50; full 240 * 200 would be a 4-minute wall-clock hit
    for marginal value when most pairs are visibly distinguishable from
    the noise floor on the unbiased estimator).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if variant == "centered":
        # Match the variant the main matrix is computed on.
        activations = _maybe_prompt_center(activations, do_center=True)
    n_cond = activations.shape[0]
    candidate_pairs = [(i, j) for i in range(n_cond) for j in range(n_cond) if i != j]
    rng.shuffle(candidate_pairs)
    sampled = candidate_pairs[: min(n_pair_samples, len(candidate_pairs))]
    per_pair = []
    nulls_pooled: list[float] = []
    for i, j in sampled:
        Xa = _drop_nan_rows(activations[i])
        Xb = _drop_nan_rows(activations[j])
        na, nb = len(Xa), len(Xb)
        if na < 2 or nb < 2:
            continue
        Z = np.vstack([Xa, Xb])
        K, sigma2 = _rbf_kernel_with_bandwidth(Z)
        observed = _unbiased_mmd2_from_kernel(K, na)
        # Permutation null via row+column re-permutation of K.
        n_total = K.shape[0]
        null_samples = []
        for _ in range(n_perm):
            perm = rng.permutation(n_total)
            Kp = K[perm][:, perm]
            null_samples.append(_unbiased_mmd2_from_kernel(Kp, na))
        null = np.asarray(null_samples)
        # One-sided p-value (P[null >= observed]).
        p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        per_pair.append(
            {
                "a": cond_ids[i],
                "b": cond_ids[j],
                "observed_mmd2": float(observed),
                "sigma2": float(sigma2),
                "null_median": float(np.median(null)),
                "null_p95": float(np.percentile(null, 95)),
                "p_value": p_value,
            }
        )
        nulls_pooled.extend(null.tolist())
    if not nulls_pooled:
        return {"per_pair": [], "n_pair_samples_done": 0}
    pooled = np.asarray(nulls_pooled)
    return {
        "per_pair": per_pair,
        "n_pair_samples_done": len(per_pair),
        "pooled_null_median": float(np.median(pooled)),
        "pooled_null_p95": float(np.percentile(pooled, 95)),
        "pooled_null_max": float(np.max(pooled)),
    }


def _c2st_auc(Xa: np.ndarray, Xb: np.ndarray, folds: int = C2ST_FOLDS) -> float:
    """Cross-validated linear-probe classifier-2-sample test, returned as a
    DISTANCE for sign-consistency with the rest of the metric panel.

    Raw AUC: 1.0 = perfectly separable, 0.5 = indistinguishable. To put it on
    the same "higher = farther apart" scale as cosine_distance / euclidean /
    MMD² / gauss_kl / W2 (and to match the sign convention of #474's cosine
    distance), we return ``c2st_dist = 2 * (AUC - 0.5)`` ∈ [0, 1]. This way
    the heatmap colorbar is single-signed for every metric: rho < 0 with ΔG
    means "more similar contexts → more transfer," uniformly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    na, nb = len(Xa), len(Xb)
    if na < folds or nb < folds:
        return float("nan")
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(na), np.ones(nb)])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        # sklearn>=1.8 deprecates the `penalty=` kwarg; default is L2, control
        # regularization via C (smaller = stronger). solver="lbfgs" is the
        # default L2-compatible solver.
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X[tr], y[tr])
        score = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], score))
    auc = float(np.mean(aucs))
    # Distance form: 2*|AUC − 0.5|, clipped to [0, 1]. Symmetric around the
    # chance boundary so labels-flipped C2ST scoring (~ 1 - AUC) and the
    # standard scoring map to the same predictor scalar.
    return float(min(1.0, 2.0 * abs(auc - 0.5)))


def _delta_spectrum(Xa: np.ndarray, Xb: np.ndarray, k: int) -> dict[str, float]:
    """PAIRED Δ-displacement spectrum.

    Both clouds MUST be in matched (cond, q) ordering — Δ_i = Xb[i] − Xa[i].
    Asserts shape match and per-row alignment; if either cloud has NaN at
    row i, that row is dropped on BOTH sides (paired drop) so alignment
    survives.

    Returns three candidate predictor scalars:
      - mean_norm        : ‖mean Δ‖
      - coherence        : energy_in_mean_dir / total_energy ∈ [0, 1]
      - effective_dim    : (Σ λ_i)² / Σ λ_i²  (participation ratio in PCA)
    """
    if Xa.shape != Xb.shape:
        raise AssertionError(f"Δ-spectrum requires paired clouds; got Xa={Xa.shape} Xb={Xb.shape}")
    # Paired NaN drop — drop a row from BOTH sides if either has NaN.
    mask = ~(np.any(np.isnan(Xa), axis=1) | np.any(np.isnan(Xb), axis=1))
    Xa = Xa[mask]
    Xb = Xb[mask]
    if len(Xa) < 2:
        return {"mean_norm": float("nan"), "coherence": float("nan"), "effective_dim": float("nan")}
    delta = Xb - Xa  # (n_q, H)
    mean_delta = delta.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_delta))
    total_energy = float(np.sum(delta**2))
    if total_energy < 1e-12 or mean_norm < 1e-12:
        coherence = 0.0
    else:
        proj_onto_mean = delta @ mean_delta / mean_norm  # (n_q,)
        coherence = float(np.sum(proj_onto_mean**2) / total_energy)
    # Effective dimensionality via Gram-eigenvalue participation ratio.
    delta_c = delta - delta.mean(axis=0, keepdims=True)
    G = delta_c @ delta_c.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    s1 = eigvals.sum()
    s2 = (eigvals**2).sum()
    eff_dim = 0.0 if s2 < 1e-18 else float(s1**2 / s2)
    return {"mean_norm": mean_norm, "coherence": coherence, "effective_dim": eff_dim}


def _gaussian_sym_kl_in_subspace(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Gaussian symmetric-KL between cloud-fitted Gaussians in the top-k PCA
    subspace.

    Closed form: KL(N0||N1) = 0.5 * (tr(Σ1^-1 Σ0) + (μ1-μ0)^T Σ1^-1 (μ1-μ0)
                                       - k + log(det Σ1 / det Σ0))
    Symmetric-KL = 0.5 * (KL(0||1) + KL(1||0)).
    """
    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])

    def _one_kl(S0, S1, m0, m1):
        # KL(N0||N1)
        S1_inv = np.linalg.inv(S1)
        sign0, logdet0 = np.linalg.slogdet(S0)
        sign1, logdet1 = np.linalg.slogdet(S1)
        if sign0 <= 0 or sign1 <= 0:
            return float("nan")
        d = S0.shape[0]
        return 0.5 * (
            np.trace(S1_inv @ S0) + (m1 - m0) @ S1_inv @ (m1 - m0) - d + (logdet1 - logdet0)
        )

    kl_ab = _one_kl(Sa, Sb, mu_a, mu_b)
    kl_ba = _one_kl(Sb, Sa, mu_b, mu_a)
    if np.isnan(kl_ab) or np.isnan(kl_ba):
        return float("nan")
    return float(0.5 * (kl_ab + kl_ba))


def _bures_wasserstein2(Xa: np.ndarray, Xb: np.ndarray, k: int) -> float:
    """Squared Bures-Wasserstein distance between cloud-fitted Gaussians in
    the top-k PCA subspace.

    W₂² = ‖μ_a − μ_b‖² + tr(Σ_a + Σ_b − 2(Σ_a^{1/2} Σ_b Σ_a^{1/2})^{1/2})
    """
    from scipy.linalg import sqrtm

    Xa = _drop_nan_rows(Xa)
    Xb = _drop_nan_rows(Xb)
    if len(Xa) < 2 or len(Xb) < 2:
        return float("nan")
    Ya, Yb = _pair_pca_subspace(Xa, Xb, k)
    mu_a = Ya.mean(axis=0)
    mu_b = Yb.mean(axis=0)
    Sa = np.cov(Ya.T, ddof=1) + 1e-6 * np.eye(Ya.shape[1])
    Sb = np.cov(Yb.T, ddof=1) + 1e-6 * np.eye(Yb.shape[1])
    Sa_sqrt = sqrtm(Sa).real
    cross = sqrtm(Sa_sqrt @ Sb @ Sa_sqrt).real
    bures = np.trace(Sa + Sb - 2 * cross)
    mu_sq = float((mu_a - mu_b) @ (mu_a - mu_b))
    return float(mu_sq + max(0.0, float(bures)))


def _maybe_prompt_center(activations: np.ndarray, do_center: bool) -> np.ndarray:
    """Optionally subtract the per-question (a.k.a. "prompt") mean across all
    in-scope contexts so distance scores reflect the *off-mean-context*
    component only. activations shape: (n_cond, n_q, H). For end_of_system
    where n_q == 1 we subtract the cross-context mean instead (the only
    meaningful "centering" target when there's no per-question grid).
    """
    if not do_center:
        return activations
    if activations.shape[1] > 1:
        # Per-question mean across the n_cond axis: shape (1, n_q, H).
        mu = np.nanmean(activations, axis=0, keepdims=True)
    else:
        # Cross-context mean of the single per-cond vector: shape (1, 1, H).
        mu = np.nanmean(activations, axis=0, keepdims=True)
    return activations - mu


def _compute_metric_matrix(  # noqa: C901 — per-metric dispatcher; one branch per metric, intentionally flat.
    activations: np.ndarray,
    cond_ids: list[str],
    metric: str,
    extraction_point: str,
    pca_k: int,
    variant: str = "raw",
) -> dict:
    """Compute the (n_cond × n_cond) pairwise predictor matrix for one metric.

    Returns a dict with the matrix and any per-(metric) auxiliary outputs
    (e.g. Δ-spectrum produces 3 sub-predictors stored as separate matrices).

    Parameters
    ----------
    activations: (n_cond, n_q, H) — for end_of_system, n_q == 1.
    metric: one of CENTROID_METRICS + CLOUD_METRICS.
    variant: "raw" or "centered" (see _maybe_prompt_center). The variant
        label is recorded in the metric payload so the regression phase can
        enumerate raw + centered as distinct predictor rows.

    end_of_system handling:
      - Cloud metrics (CLOUD_METRICS) return {"matrix": None, "n_a": ...}.
      - "mahal" (per-pair pooled cov) returns NaN matrix because n_q == 1
        makes the within-cloud covariance undefined; use "mahal_pooled_ctx"
        instead (Mahalanobis vs. the pooled context covariance across the
        full Class-A subpanel — the meaningful one-vector-per-cond metric).
    """
    if variant not in PREDICTOR_VARIANTS:
        raise ValueError(f"variant must be one of {PREDICTOR_VARIANTS}; got {variant!r}")
    activations = _maybe_prompt_center(activations, do_center=(variant == "centered"))
    n_cond = activations.shape[0]
    is_centroid = metric in CENTROID_METRICS
    if not is_centroid and extraction_point == "end_of_system":
        # Cloud metrics are N/A at end_of_system — n_q==1, no cloud exists.
        return {
            "matrix": None,
            "n_a": "cloud metric N/A at end_of_system (one vector per cond)",
            "variant": variant,
        }

    if metric == "delta_spec":
        # Δ-spec emits 3 scalars per pair — store as 3 stacked matrices.
        ms = {
            "mean_norm": [[None] * n_cond for _ in range(n_cond)],
            "coherence": [[None] * n_cond for _ in range(n_cond)],
            "effective_dim": [[None] * n_cond for _ in range(n_cond)],
        }
        for i in range(n_cond):
            for j in range(n_cond):
                if i == j:
                    for key in ms:
                        ms[key][i][j] = 0.0
                    continue
                spec = _delta_spectrum(activations[i], activations[j], pca_k)
                for key in ms:
                    ms[key][i][j] = float(spec[key])
        return {
            "variant": variant,
            "matrices": {
                k: {
                    cond_ids[i]: {cond_ids[j]: ms[k][i][j] for j in range(n_cond)}
                    for i in range(n_cond)
                }
                for k in ms
            },
            "sub_predictors": list(ms.keys()),
        }

    # mahal_pooled_ctx needs the shared pooled-cov state pre-built ONCE
    # (it's the same matrix for every (i, j) pair within this metric file).
    # On rank-deficient / collapsed inputs the helper returns None AND
    # records the failure reason in a side channel; we emit an explicit
    # N/A row with the reason in the payload rather than ridging a
    # degenerate cov into a spurious finite Mahalanobis (round-2 issue
    # B that round 3 corrects).
    pooled_state = None
    if metric == "mahal_pooled_ctx":
        pooled_state = _build_context_pooled_mahal_state(activations, pca_k)
        if pooled_state is None:
            reason = _pop_pooled_failure_reason(activations) or (
                "pooled centroid covariance unusable (no reason recorded)"
            )
            return {
                "variant": variant,
                "matrix": None,
                "n_a": (f"mahal_pooled_ctx at extraction_point={extraction_point}: {reason}"),
                "n_cond": int(n_cond),
                "pca_k": int(pca_k),
            }

    mat: list[list[float]] = [[0.0] * n_cond for _ in range(n_cond)]
    for i in range(n_cond):
        for j in range(n_cond):
            if i == j:
                mat[i][j] = 0.0
                continue
            Xa = activations[i]
            Xb = activations[j]
            if metric == "cosine":
                d = _centroid_cosine_distance(Xa, Xb)
            elif metric == "euclidean":
                d = _centroid_euclidean(Xa, Xb)
            elif metric == "mahal":
                d = _centroid_mahal(Xa, Xb, pca_k)
            elif metric == "mahal_pooled_ctx":
                d = _context_mahal_with_pooled_cov(Xa, Xb, pooled_state, i, j)
            elif metric == "mmd":
                d = _rbf_mmd_squared(Xa, Xb)
            elif metric == "c2st":
                d = _c2st_auc(Xa, Xb)
            elif metric == "gauss_kl":
                d = _gaussian_sym_kl_in_subspace(Xa, Xb, pca_k)
            elif metric == "wass2":
                d = _bures_wasserstein2(Xa, Xb, pca_k)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            mat[i][j] = float(d)
    return {
        "variant": variant,
        "matrix": {
            cond_ids[i]: {cond_ids[j]: mat[i][j] for j in range(n_cond)} for i in range(n_cond)
        },
    }


def run_metrics(
    *,
    activations_by_point: dict[str, dict[int, dict]],
    metrics: tuple[str, ...],
    pca_k: int,
    overwrite: bool,
    mmd_permutations: int = MMD_PERMUTATIONS,
) -> None:
    """Compute every (extraction_point × layer × metric × variant) distance
    matrix and checkpoint EACH ONE to disk immediately.

    Per-metric notes:
      - Cloud metrics at end_of_system → N/A row written with explicit
        `matrix: null` (NOT silently dropped).
      - Centroid `mahal` at end_of_system → also N/A (per-pair within-cloud
        cov is undefined at n_q=1); the meaningful one-vector-per-cond
        Mahalanobis is `mahal_pooled_ctx`.
      - For `mmd`, also writes a `<point>__layer<L>__mmd__perm.json`
        companion with the permutation-null summary (median, max under H0)
        so downstream callers can compute pair-level p-values if desired.
        Permutations run on a uniform random subset of pairs (default 16)
        to keep wall-clock bounded — the bandwidth is identical across
        pairs, so the null shape is shared.
    """
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    for pt, layer_map in activations_by_point.items():
        for L, payload in layer_map.items():
            cond_ids = payload["cond_ids"]
            arr = payload["activations"]
            for m in metrics:
                for variant in PREDICTOR_VARIANTS:
                    out_path = METRIC_DIR / f"{pt}__layer{L}__{m}__{variant}.json"
                    if out_path.exists() and not overwrite:
                        logger.info("Skipping existing %s", out_path)
                        continue

                    t0 = time.time()
                    # Centroid mahal at end_of_system is N/A (per-pair cov is
                    # undefined when n_q==1); use mahal_pooled_ctx instead.
                    if pt == "end_of_system" and m == "mahal":
                        _write_json_atomic(
                            out_path,
                            {
                                "schema_version": 1,
                                "extraction_point": pt,
                                "layer": L,
                                "metric": m,
                                "variant": variant,
                                "matrix": None,
                                "n_a": (
                                    "per-pair pooled cov undefined at "
                                    "end_of_system (n_q=1); use mahal_pooled_ctx"
                                ),
                                "git_sha": _git_sha(),
                                "timestamp_utc": _now_iso(),
                            },
                        )
                        continue

                    res = _compute_metric_matrix(arr, cond_ids, m, pt, pca_k, variant=variant)
                    payload_out = {
                        "schema_version": 1,
                        "extraction_point": pt,
                        "layer": L,
                        "metric": m,
                        "variant": variant,
                        "pca_k": pca_k,
                        "cond_ids": cond_ids,
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                        **res,
                    }
                    _write_json_atomic(out_path, payload_out)
                    logger.info("Wrote %s in %.2fs", out_path, time.time() - t0)

                    # MMD permutation null (cloud regime only).
                    if m == "mmd" and pt != "end_of_system" and res.get("matrix") is not None:
                        perm_path = METRIC_DIR / f"{pt}__layer{L}__{m}__{variant}__perm.json"
                        if perm_path.exists() and not overwrite:
                            continue
                        perm_summary = _mmd_permutation_summary(
                            arr,
                            cond_ids,
                            n_perm=mmd_permutations,
                            variant=variant,
                        )
                        _write_json_atomic(
                            perm_path,
                            {
                                "schema_version": 1,
                                "extraction_point": pt,
                                "layer": L,
                                "metric": m,
                                "variant": variant,
                                "n_perm": mmd_permutations,
                                "summary": perm_summary,
                                "git_sha": _git_sha(),
                                "timestamp_utc": _now_iso(),
                            },
                        )


# ───────────────────────── regression phase ─────────────────────────


# Degenerate-input thresholds for the rank-residualize + LOOCV paths.
# Both numbers are deliberately small; the headline panels (n=240/156) are
# never near them. They bite on the end_of_system Class-A subpanel
# (n=20 ordered pairs) where some predictor columns can be NaN or constant.
_MIN_FINITE_FOR_REGRESSION: int = 5
_CONSTANT_VAR_TOL: float = 1e-12


def _finite_and_non_constant(arr: np.ndarray) -> np.ndarray:
    """Mask of entries that are finite (not NaN/inf). Caller checks
    `mask.sum() >= _MIN_FINITE_FOR_REGRESSION` and `arr[mask].var() > tol`
    before feeding into rank-correlation / polyfit.
    """
    return np.isfinite(arr)


def _safe_polyfit_residual(target: np.ndarray, covar: np.ndarray) -> np.ndarray | None:
    """Residualize `target` on a linear fit against `covar`.

    Returns `target - (a + b * covar)` on success; None when the polyfit
    is ill-conditioned (constant covar, identical x/y values, etc.).
    The caller falls back to the un-residualized series in that case.
    """
    try:
        b, a = np.polyfit(covar, target, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    fit = a + b * covar
    if not np.all(np.isfinite(fit)):
        return None
    return target - fit


def _length_partial(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> tuple[float, float]:
    """Rank-then-residualize length-partial Spearman.

    Matches the convention used by `scripts/i474_cosine_followup._length_partial`
    AND hardens against degenerate inputs the round-2 version didn't see on
    the end_of_system subpanel (NaN columns, constant predictor / covar,
    SVD non-convergence in `np.polyfit`):
      - All inputs are first restricted to rows where x, y, AND covar are
        finite. Fewer than `_MIN_FINITE_FOR_REGRESSION` finite rows → NaN.
      - If `x[mask]` or `y[mask]` is constant after rank-residualization,
        Spearman is undefined → NaN.
      - If either polyfit raises (constant rank covar, SVD non-convergence),
        we fall back to the un-residualized rank correlation rather than
        crashing the whole regression phase.
    """
    from scipy.stats import pearsonr, rankdata

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covar = np.asarray(covar, dtype=np.float64)
    mask = (
        _finite_and_non_constant(x) & _finite_and_non_constant(y) & _finite_and_non_constant(covar)
    )
    if mask.sum() < _MIN_FINITE_FOR_REGRESSION:
        return float("nan"), float("nan")
    xm, ym, cm = x[mask], y[mask], covar[mask]
    if xm.var() < _CONSTANT_VAR_TOL or ym.var() < _CONSTANT_VAR_TOL:
        return float("nan"), float("nan")
    rx, ry, rc = rankdata(xm), rankdata(ym), rankdata(cm)
    if rc.var() < _CONSTANT_VAR_TOL:
        # Covar is constant in rank space (all-ties) → length-partial = bare
        # Spearman. Skip the polyfit entirely.
        ex, ey = rx, ry
    else:
        ex = _safe_polyfit_residual(rx, rc)
        ey = _safe_polyfit_residual(ry, rc)
        if ex is None or ey is None:
            ex, ey = rx, ry  # un-residualized fallback
    if ex.var() < _CONSTANT_VAR_TOL or ey.var() < _CONSTANT_VAR_TOL:
        return float("nan"), float("nan")
    try:
        rho, p = pearsonr(ex, ey)
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan")
    return float(rho), float(p)


def _length_partial_residualize_rank(
    x: np.ndarray, y: np.ndarray, covar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_resid, y_resid) on the rank scale with the log-length
    covariate's linear-in-rank component projected out. Matches the
    rank-then-residualize convention used by `_length_partial`.

    Defensive: if `np.polyfit` is ill-conditioned (constant rank covar,
    SVD non-convergence on tiny LOOCV folds), fall back to the bare-rank
    series instead of letting the LinAlgError propagate up.
    """
    from scipy.stats import rankdata

    rx, ry, rc = rankdata(x), rankdata(y), rankdata(covar)
    if rc.var() < _CONSTANT_VAR_TOL:
        return rx, ry
    ex = _safe_polyfit_residual(rx, rc)
    ey = _safe_polyfit_residual(ry, rc)
    if ex is None or ey is None:
        return rx, ry
    return ex, ey


def _loocv_r2(
    x: np.ndarray,
    y: np.ndarray,
    cond_ids_a: list[str],
    cond_ids_b: list[str],
    *,
    covar: np.ndarray | None = None,
) -> float:
    """Leave-one-context-out CV R² (the i474 fig9 pattern), length-partialed.

    For each cond C, hold out all pairs touching C, fit OLS on the
    remainder, predict held-out, compute (1 − SSE / SST). When `covar` is
    provided, residualize x and y on rank(covar) FIRST so the CV captures
    the same length-controlled signal as the headline Spearman.

    Degenerate-input hardening (caught on the end_of_system Class-A
    subpanel — tiny LOOCV folds + occasional NaN / constant predictors
    crashed the round-2 `np.polyfit` with SVD non-convergence):
      - Up-front: filter rows where x, y are NOT finite; if fewer than
        `_MIN_FINITE_FOR_REGRESSION` remain, return NaN.
      - Per-fold: skip training folds with train.sum() < 3 (was 5 — but a
        too-aggressive floor produces too few valid CV folds on small
        subpanels), or where train x has < 2 distinct values, or where
        train x / y have non-finite entries.
      - polyfit is in try/except for (LinAlgError, ValueError); a fold
        that fails the fit is skipped (pred stays NaN), the CV runs
        on whatever folds DID fit, and if too few folds survive (<3
        usable predictions or 0 SST) the result is NaN — never a crash,
        never a spurious 0.
    """
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Up-front finite-filter: keep only rows where x and y are finite (and
    # covar if provided). Subsequent operations only see the kept rows.
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if covar is not None:
        covar = np.asarray(covar, dtype=np.float64)
        finite_mask = finite_mask & np.isfinite(covar)
    if finite_mask.sum() < _MIN_FINITE_FOR_REGRESSION:
        return float("nan")
    if covar is not None:
        x, y = _length_partial_residualize_rank(x[finite_mask], y[finite_mask], covar[finite_mask])
    else:
        x, y = x[finite_mask], y[finite_mask]
    cond_ids_a = [c for c, k in zip(cond_ids_a, finite_mask, strict=True) if k]
    cond_ids_b = [c for c, k in zip(cond_ids_b, finite_mask, strict=True) if k]
    n = len(x)
    pred = np.full(n, np.nan)
    src = np.array(cond_ids_a)
    tgt = np.array(cond_ids_b)
    folds_attempted = 0
    folds_skipped_degenerate = 0
    for C in set(cond_ids_a) | set(cond_ids_b):
        train = ~((src == C) | (tgt == C))
        test = (src == C) | (tgt == C)
        if train.sum() < 3:
            folds_skipped_degenerate += 1
            continue
        x_train = x[train]
        y_train = y[train]
        # Need ≥ 2 distinct x values for a non-degenerate 1-D OLS fit.
        if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(y_train)):
            folds_skipped_degenerate += 1
            continue
        if len(np.unique(x_train)) < 2:
            folds_skipped_degenerate += 1
            continue
        folds_attempted += 1
        try:
            b, a = np.polyfit(x_train, y_train, 1)
        except (np.linalg.LinAlgError, ValueError):
            # SVD non-convergence or other numerical failure on this fold —
            # leave pred[test] as NaN, downstream m-mask drops it.
            continue
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        pred[test] = a + b * x[test]
    m = np.isfinite(pred)
    # Need at least a few usable predictions to compute R². Subpanels with
    # only 1-2 successful folds are too noisy to interpret; return NaN.
    if m.sum() < _MIN_FINITE_FOR_REGRESSION:
        return float("nan")
    sse = np.sum((y[m] - pred[m]) ** 2)
    sst = np.sum((y[m] - y[m].mean()) ** 2)
    if sst < 1e-18 or not np.isfinite(sse):
        return float("nan")
    return float(1.0 - sse / sst)


def _pairs(cond_ids: list[str], nonstylized_only: bool) -> list[tuple[str, str]]:
    """Build the list of ordered off-diagonal pairs, optionally dropping any
    pair touching a stylized persona (A3/A4/A5)."""
    out = []
    for a in cond_ids:
        for b in cond_ids:
            if a == b:
                continue
            if nonstylized_only and ((a in STY_CIDS) or (b in STY_CIDS)):
                continue
            out.append((a, b))
    return out


def _materialize_predictor_vector(
    metric_payload: dict,
    pairs: list[tuple[str, str]],
    sub_predictor: str | None,
) -> np.ndarray | None:
    """Read a distance value per pair from a metric_payload (one matrix file).

    Returns None if the matrix is entirely N/A (cloud metric at end_of_system).
    For Δ-spectrum, ``sub_predictor`` ∈ {"mean_norm", "coherence",
    "effective_dim"}. Returns None if ANY requested pair is missing from the
    matrix — caller is responsible for choosing a pair list that matches the
    metric's cond_ids (end_of_system subpanel use case).
    """
    if "matrix" in metric_payload and metric_payload["matrix"] is None:
        return None  # N/A — cloud metric or pair-cov-undefined at end_of_system
    if "matrices" in metric_payload:
        if sub_predictor is None:
            return None
        m = metric_payload["matrices"][sub_predictor]
    else:
        m = metric_payload["matrix"]
    vals = []
    for a, b in pairs:
        if a not in m or b not in m[a]:
            return None
        v = m[a][b]
        if v is None:
            return None
        vals.append(float(v))
    return np.array(vals, dtype=np.float64)


def _enumerate_predictors(metric_files: list[Path]) -> list[dict]:
    """Walk every metric file and enumerate every distinct predictor scalar
    (one row per (extraction_point, layer, metric, variant, sub_predictor)).

    Each row carries the metric file's `cond_ids` so the regression phase
    can restrict the pair list to the SUBPANEL the metric is actually
    defined on (Class-A-only for end_of_system, full 16 elsewhere).

    Skips two classes of non-predictor sidecars that live alongside the
    distance-matrix JSONs in ``METRIC_DIR``:
      - ``*__perm.json``  — MMD permutation-null companions (no matrix payload).
      - ``*__cross_check_406.json`` — the next_token_js cross-check sidecar
        emitted by ``write_next_token_js_matrix``. Schema is
        ``{schema_version, failed, summary?/failure_reason?, git_sha,
        timestamp_utc}`` — no ``extraction_point`` / ``layer`` / ``metric``
        fields, so reading it as a predictor KeyErrors. This is a
        cross-check artifact, NOT a predictor input. (Round-8 fix #502.)
    """
    rows = []
    for p in metric_files:
        # Skip the per-pair MMD permutation companion files.
        if "__perm" in p.name:
            continue
        # Skip the next_token_js vs #406 cross-check sidecar (schema-different,
        # not a distance-matrix predictor).
        if p.name.endswith("__cross_check_406.json"):
            continue
        payload = json.loads(p.read_text())
        pt = payload["extraction_point"]
        L = payload["layer"]
        m = payload["metric"]
        variant = payload.get("variant", "raw")
        cond_ids_file = payload.get("cond_ids")
        if "matrices" in payload:
            for sub in payload["sub_predictors"]:
                rows.append(
                    {
                        "extraction_point": pt,
                        "layer": L,
                        "metric": m,
                        "variant": variant,
                        "sub_predictor": sub,
                        "cond_ids": cond_ids_file,
                        "file": str(p),
                    }
                )
        else:
            rows.append(
                {
                    "extraction_point": pt,
                    "layer": L,
                    "metric": m,
                    "variant": variant,
                    "sub_predictor": None,
                    "cond_ids": cond_ids_file,
                    "file": str(p),
                }
            )
    return rows


def _saturation_fraction(g: np.ndarray) -> float:
    """Fraction of cells at/above the saturation threshold (g_logprob > -0.1)."""
    return float(np.mean(g > SATURATION_GLOGP_THRESHOLD))


def run_regression(
    *,
    cond_ids: list[str],
    arms: tuple[str, ...],
    epochs: tuple[int, ...],
    overwrite: bool,
) -> dict:
    """Per-(arm, epoch) regression of every enumerated predictor against
    ΔG (primary) and g_logprob (base-prior-safe). Checkpoints each cell.

    Each predictor row reports rho + CV on TWO panels:
      - panel_primary    — the "full" panel for that predictor: the 240
        full grid for predictors defined on all 16 conds; for
        end_of_system predictors (defined on Class A only) it's the
        20-pair Class-A subpanel (n=5*4 ordered pairs).
      - panel_nonstylized — drops any pair touching A3/A4/A5 (a
        sub-restriction of panel_primary). For end_of_system this leaves
        the 2*1=2-pair A1-A2 subpanel which is too small to interpret
        and is recorded explicitly as such.

    The leave-one-context-out CV is **length-partialed** (residualizes
    rank(x) and rank(y) on rank(log prompt_tokens) first) so it captures
    the same length-controlled signal as the headline Spearman. A
    length-confound predictor that "wins" by capturing log_prompt_tokens
    variance therefore CANNOT win the bake-off.

    Returns the headline (loc_ep1) summary for convenience.
    """
    REGR_DIR.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(METRIC_DIR.glob("*.json"))
    predictors = _enumerate_predictors(metric_files)
    prompt_tokens = _load_prompt_tokens()

    all_cells: dict[str, dict] = {}
    for arm in arms:
        for ep in epochs:
            cell_key = f"{arm}_ep{ep}"
            out_path = REGR_DIR / f"{cell_key}.json"
            if out_path.exists() and not overwrite:
                all_cells[cell_key] = json.loads(out_path.read_text())
                continue
            G = _load_G(arm, ep)
            # Full 16-cond panels (used by every predictor whose metric
            # file carries all 16 cond_ids).
            pairs_full16 = _pairs(cond_ids, nonstylized_only=False)
            pairs_ns_full16 = _pairs(cond_ids, nonstylized_only=True)
            dg_f16 = np.array([G[a][b]["delta_g"] for a, b in pairs_full16])
            g_f16 = np.array([G[a][b]["g_logprob"] for a, b in pairs_full16])
            ln_f16 = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_full16])
            dg_ns16 = np.array([G[a][b]["delta_g"] for a, b in pairs_ns_full16])
            g_ns16 = np.array([G[a][b]["g_logprob"] for a, b in pairs_ns_full16])
            ln_ns16 = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_ns_full16])
            src_f16 = [a for a, _ in pairs_full16]
            tgt_f16 = [b for _, b in pairs_full16]
            src_ns16 = [a for a, _ in pairs_ns_full16]
            tgt_ns16 = [b for _, b in pairs_ns_full16]

            sat_full16 = _saturation_fraction(g_f16)
            sat_ns16 = _saturation_fraction(g_ns16)

            entries = []
            for desc in predictors:
                payload = json.loads(Path(desc["file"]).read_text())
                pred_cond_ids = desc.get("cond_ids") or cond_ids

                # Choose the pair list this predictor is defined on.
                if set(pred_cond_ids) == set(cond_ids):
                    pairs_primary = pairs_full16
                    pairs_nonsty = pairs_ns_full16
                    dg_p, g_p, ln_p = dg_f16, g_f16, ln_f16
                    dg_n, g_n, ln_n = dg_ns16, g_ns16, ln_ns16
                    src_p, tgt_p = src_f16, tgt_f16
                    src_n, tgt_n = src_ns16, tgt_ns16
                    panel_primary_name = "full16 (240 ordered pairs)"
                    panel_nonsty_name = "nonstylized (156 ordered pairs)"
                else:
                    # Subpanel — e.g. end_of_system on Class A.
                    pairs_primary = _pairs(pred_cond_ids, nonstylized_only=False)
                    pairs_nonsty = _pairs(pred_cond_ids, nonstylized_only=True)
                    dg_p = np.array([G[a][b]["delta_g"] for a, b in pairs_primary])
                    g_p = np.array([G[a][b]["g_logprob"] for a, b in pairs_primary])
                    ln_p = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_primary])
                    dg_n = np.array([G[a][b]["delta_g"] for a, b in pairs_nonsty])
                    g_n = np.array([G[a][b]["g_logprob"] for a, b in pairs_nonsty])
                    ln_n = np.array([np.log(prompt_tokens[a][b]) for a, b in pairs_nonsty])
                    src_p = [a for a, _ in pairs_primary]
                    tgt_p = [b for _, b in pairs_primary]
                    src_n = [a for a, _ in pairs_nonsty]
                    tgt_n = [b for _, b in pairs_nonsty]
                    panel_primary_name = (
                        f"subpanel cond_ids={sorted(pred_cond_ids)} "
                        f"({len(pairs_primary)} ordered pairs)"
                    )
                    panel_nonsty_name = f"subpanel nonstylized ({len(pairs_nonsty)} ordered pairs)"

                xv_p = _materialize_predictor_vector(payload, pairs_primary, desc["sub_predictor"])
                xv_n = (
                    _materialize_predictor_vector(payload, pairs_nonsty, desc["sub_predictor"])
                    if len(pairs_nonsty) >= 5
                    else None
                )

                if xv_p is None:
                    entries.append({**desc, "status": "N/A (matrix is None or missing pair)"})
                    continue

                # Up-front degeneracy guard on the predictor column (round-5
                # fix): a column with <_MIN_FINITE_FOR_REGRESSION finite
                # entries or zero variance over its finite entries is
                # unregressable. Mark it degenerate with explicit NaN rho /
                # CV in the payload and skip the regression — otherwise the
                # downstream `np.polyfit` crashes with SVD non-convergence
                # on the tiny LOOCV folds the end_of_system Class-A
                # subpanel produces.
                xv_p_finite_mask = np.isfinite(xv_p)
                n_finite_p = int(xv_p_finite_mask.sum())
                primary_degenerate = (
                    n_finite_p < _MIN_FINITE_FOR_REGRESSION
                    or float(xv_p[xv_p_finite_mask].var() if n_finite_p > 0 else 0.0)
                    < _CONSTANT_VAR_TOL
                )
                if primary_degenerate:
                    entries.append(
                        {
                            **desc,
                            "panel_primary": panel_primary_name,
                            "panel_nonstylized": panel_nonsty_name,
                            "n_primary": len(xv_p),
                            "n_finite_primary": n_finite_p,
                            "n_nonstylized": len(xv_n) if xv_n is not None else 0,
                            "status": "degenerate",
                            "degenerate_reason": (
                                f"primary predictor column has {n_finite_p} finite "
                                f"of {len(xv_p)} pairs and/or "
                                "variance below tolerance — unregressable"
                            ),
                            "rho_full_deltag": float("nan"),
                            "p_full_deltag": float("nan"),
                            "rho_full_glogp": float("nan"),
                            "p_full_glogp": float("nan"),
                            "rho_nonstylized_deltag": float("nan"),
                            "p_nonstylized_deltag": float("nan"),
                            "rho_nonstylized_glogp": float("nan"),
                            "p_nonstylized_glogp": float("nan"),
                            "cv_full_deltag": float("nan"),
                            "cv_full_glogp": float("nan"),
                            "cv_nonstylized_deltag": float("nan"),
                            "cv_nonstylized_glogp": float("nan"),
                        }
                    )
                    continue

                # Length-partial Spearman, per panel x DV. NaN return is
                # acceptable now — `_length_partial` is hardened against
                # degenerate inputs and returns NaN rather than raising.
                rho_p_dg, p_p_dg = _length_partial(xv_p, dg_p, ln_p)
                rho_p_g, p_p_g = _length_partial(xv_p, g_p, ln_p)
                if xv_n is not None and len(xv_n) >= 5:
                    xv_n_finite_mask = np.isfinite(xv_n)
                    n_finite_n = int(xv_n_finite_mask.sum())
                    n_panel_degenerate = (
                        n_finite_n < _MIN_FINITE_FOR_REGRESSION
                        or float(xv_n[xv_n_finite_mask].var() if n_finite_n > 0 else 0.0)
                        < _CONSTANT_VAR_TOL
                    )
                    if n_panel_degenerate:
                        rho_n_dg = p_n_dg = rho_n_g = p_n_g = float("nan")
                        cv_n_dg = cv_n_g = float("nan")
                    else:
                        rho_n_dg, p_n_dg = _length_partial(xv_n, dg_n, ln_n)
                        rho_n_g, p_n_g = _length_partial(xv_n, g_n, ln_n)
                        cv_n_dg = _loocv_r2(xv_n, dg_n, src_n, tgt_n, covar=ln_n)
                        cv_n_g = _loocv_r2(xv_n, g_n, src_n, tgt_n, covar=ln_n)
                else:
                    rho_n_dg = p_n_dg = rho_n_g = p_n_g = float("nan")
                    cv_n_dg = cv_n_g = float("nan")

                # Length-partialed leave-one-context-out CV (the i474 fig9
                # pattern, generalized + length-controlled).
                cv_p_dg = _loocv_r2(xv_p, dg_p, src_p, tgt_p, covar=ln_p)
                cv_p_g = _loocv_r2(xv_p, g_p, src_p, tgt_p, covar=ln_p)

                entries.append(
                    {
                        **desc,
                        "panel_primary": panel_primary_name,
                        "panel_nonstylized": panel_nonsty_name,
                        "n_primary": len(xv_p),
                        "n_finite_primary": int(n_finite_p),
                        "n_nonstylized": len(xv_n) if xv_n is not None else 0,
                        "rho_full_deltag": float(rho_p_dg),
                        "p_full_deltag": float(p_p_dg),
                        "rho_full_glogp": float(rho_p_g),
                        "p_full_glogp": float(p_p_g),
                        "rho_nonstylized_deltag": float(rho_n_dg),
                        "p_nonstylized_deltag": float(p_n_dg),
                        "rho_nonstylized_glogp": float(rho_n_g),
                        "p_nonstylized_glogp": float(p_n_g),
                        "cv_full_deltag": float(cv_p_dg),
                        "cv_full_glogp": float(cv_p_g),
                        "cv_nonstylized_deltag": float(cv_n_dg),
                        "cv_nonstylized_glogp": float(cv_n_g),
                    }
                )

            cell_payload = {
                "schema_version": 1,
                "arm": arm,
                "epoch": ep,
                "n_pairs_full16": len(pairs_full16),
                "n_pairs_nonstylized_full16": len(pairs_ns_full16),
                "saturation_frac_full16": sat_full16,
                "saturation_frac_nonstylized_full16": sat_ns16,
                "entries": entries,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
            }
            _write_json_atomic(out_path, cell_payload)
            logger.info("Wrote %s — %d predictor entries", out_path, len(entries))
            all_cells[cell_key] = cell_payload

    return all_cells


SUBPANEL_MIN_NONSTYLIZED_N = 20  # minimum non-stylized pair count to clear the headline guard


def select_winner(headline_cell: dict) -> dict | None:
    """Pick the highest-CV predictor that ALSO survives on the non-stylized
    panel AND on the NON-stylized base-prior-safe g_logprob check.

    Survival conditions (the published #474 framing):
      1. `np.sign(rho_full_deltag) == np.sign(rho_nonstylized_deltag)` AND
         `|rho_nonstylized_deltag| > FLOOR_RHO` — the predictor's signal is
         not carried by the (often saturated) stylized rows alone.
      2. `np.sign(rho_nonstylized_glogp) == np.sign(rho_full_deltag)`
         AND `|rho_nonstylized_glogp| > FLOOR_RHO` — the trained-log-prob
         (base-prior-safe DV) shows the SAME-direction and NONTRIVIAL
         relationship on the NON-STYLIZED panel. This is the load-bearing
         guard: a predictor that wins on `rho_full_glogp` only because
         the stylized rows carry the trained-logp signal — while the
         non-stylized trained-logp collapses to ~0 — is exactly the
         artifact #474's non-stylized survival framing is designed to
         catch (see scripts/i474_cosine_followup.py:111-114 and :262-278).
         Round 2 erroneously checked rho_FULL_glogp without a magnitude
         floor; round 3 corrects to rho_NONSTYLIZED_glogp + |rho|>FLOOR.
      3. The non-stylized panel must be LARGE ENOUGH to support a
         meaningful sign-stability check on g_logprob. Subpanels below
         SUBPANEL_MIN_NONSTYLIZED_N (default 20 ordered pairs) cannot
         clear this guard — they are recorded as `diagnostic_only=True`
         and EXEMPTED from headline winner selection (e.g. the
         end_of_system Class-A subpanel's non-stylized restriction
         collapses to 2 pairs after dropping A3/A4/A5, which is too small
         to call a base-prior survival).
      4. CV value is finite (skip NaN).

    The returned winner dict carries the existing entry fields plus the
    panel name used so a downstream reviewer can verify the panel was
    full16, not a subpanel.

    Returns the winning entry dict (with `panel_primary` retained) or
    None when no predictor satisfies all four conditions.
    """
    FLOOR_RHO = 0.10
    survivors = []
    for e in headline_cell["entries"]:
        if "rho_full_deltag" not in e:
            continue
        # Skip entries that the regression phase marked as degenerate
        # (round-5 fix): a predictor with <_MIN_FINITE_FOR_REGRESSION
        # finite pairs or constant predictor column carries NaN rho/CV
        # by construction and cannot win the headline.
        if e.get("status") == "degenerate":
            continue
        # Also skip entries whose primary rho came back NaN from the
        # length-partial (degenerate rank input, polyfit fallback failed)
        # — these aren't tagged "degenerate" but can't compete.
        rho_f_val = e["rho_full_deltag"]
        if not np.isfinite(rho_f_val):
            continue
        # Subpanels (e.g. end_of_system Class-A only) can't clear the
        # nonstylized-trained-logp guard at n=2; mark + skip.
        n_nonsty = int(e.get("n_nonstylized", 0))
        if n_nonsty < SUBPANEL_MIN_NONSTYLIZED_N:
            e["diagnostic_only"] = True
            e["diagnostic_only_reason"] = (
                f"n_nonstylized={n_nonsty} < {SUBPANEL_MIN_NONSTYLIZED_N}; "
                "subpanel too small for the non-stylized base-prior-safe guard. "
                "Reported for diagnostic / sanity inspection only; cannot win "
                "the headline."
            )
            continue
        rho_f = e["rho_full_deltag"]
        rho_ns = e["rho_nonstylized_deltag"]
        rho_ns_g = e["rho_nonstylized_glogp"]
        if not (np.sign(rho_f) == np.sign(rho_ns) and abs(rho_ns) > FLOOR_RHO):
            continue
        # NON-STYLIZED trained-log-prob guard (the round-3 fix): the
        # base-prior-safe DV must show the same-direction relationship as
        # ΔG on the *non-stylized* panel, not the full panel — otherwise a
        # stylized-row-carried g_logprob signal can let a base-prior shadow
        # win. Require BOTH same sign AND |rho| > FLOOR_RHO so a near-zero
        # nonstylized-g_logprob (the stylized-carry shadow) doesn't slip
        # through on sign-match alone.
        if np.isnan(rho_ns_g) or np.sign(rho_ns_g) != np.sign(rho_f) or abs(rho_ns_g) < FLOOR_RHO:
            continue
        if np.isnan(e["cv_full_deltag"]):
            continue
        survivors.append(e)
    if not survivors:
        return None
    survivors.sort(key=lambda e: e["cv_full_deltag"], reverse=True)
    return survivors[0]


# ───────────────────────── correctness cross-check ─────────────────────────


def reproduce_last_token_cosine_check(
    activations_last_prompt: dict[int, dict],
    existing_cosines: dict[int, dict],
    cond_ids: list[str],
    *,
    strict: bool,
) -> dict[int, dict]:
    """Re-compute last-token cosine distances from our fresh activations and
    diff against the existing eval_results/issue_406/cosine/C_L*.json.

    The existing #406 recipe = cosine-distance of cond-mean activations
    across 50 probes at last-prompt-token. Our last_prompt extraction is
    the same recipe; the two must agree within the per-layer tolerance
    from ``COSINE_REPRO_TOLERANCES`` (3e-3 for L0/L5/L11/L15/L21, 1e-2 for
    L27 — the deepest layer carries a documented bf16 precision-
    accumulation floor; see the comment block above COSINE_REPRO_TOLERANCES
    for the GPU-verified per-layer diff progression).

    Parameters
    ----------
    strict: when True, raise AssertionError on ANY layer mismatch — the
        whole bake-off is unsafe to interpret if the prompt-building or
        last-position indexing diverges from #406's recipe. The orchestrator
        sets strict=True on full real-data runs (all 16 conds × 50 probes)
        and strict=False on subset smoke / debug runs where the cond-set or
        probe-set deliberately differs.

    Returns
    -------
    Per-layer diff summary dict — each entry carries the applied per-layer
    tolerance so a downstream reader can audit which layer used which cap.

    Raises
    ------
    AssertionError when strict=True AND any layer's max |diff| exceeds its
    per-layer tolerance from ``COSINE_REPRO_TOLERANCES``.
    """
    out: dict[int, dict] = {}
    failures: list[str] = []
    for L, payload in activations_last_prompt.items():
        if L not in existing_cosines:
            continue
        existing = existing_cosines[L]["matrix"]
        arr = payload["activations"]  # (n_cond, n_q, H)
        our_cond_ids = payload["cond_ids"]
        # Restrict to the intersection (the existing matrices have all 16).
        common = [c for c in our_cond_ids if c in existing]
        if len(common) < 2:
            continue
        max_diff = 0.0
        n_pairs = 0
        for i, a in enumerate(our_cond_ids):
            if a not in existing:
                continue
            for j, b in enumerate(our_cond_ids):
                if a == b or b not in existing.get(a, {}):
                    continue
                ours = _centroid_cosine_distance(arr[i], arr[j])
                theirs = float(existing[a][b])
                max_diff = max(max_diff, abs(ours - theirs))
                n_pairs += 1
        layer_tol = cosine_tolerance_for_layer(L)
        ok = bool(max_diff < layer_tol)
        out[L] = {
            "max_abs_diff": float(max_diff),
            "n_pairs_checked": int(n_pairs),
            "tolerance": float(layer_tol),
            "ok": ok,
        }
        if not ok:
            failures.append(
                f"L{L}: max |diff| = {max_diff:.2e} > {layer_tol:.2e} over {n_pairs} pairs"
            )
        level = logging.INFO if ok else logging.WARNING
        logger.log(
            level,
            "Cosine cross-check L%d: max |diff| = %.2e over %d pairs (tol=%.2e, ok=%s)",
            L,
            max_diff,
            n_pairs,
            layer_tol,
            ok,
        )
    if strict and failures:
        # Fail-fast: the bake-off is unsafe to interpret if the prompt-
        # building or last-position indexing diverges from #406's recipe.
        tol_str = ", ".join(
            f"L{k}={v:.0e}" if isinstance(k, int) else f"default={v:.0e}"
            for k, v in COSINE_REPRO_TOLERANCES.items()
        )
        raise AssertionError(
            "Last-token cosine cross-check FAILED against "
            f"eval_results/issue_406/cosine/C_L*.json (per-layer tolerances: {tol_str}):\n  "
            + "\n  ".join(failures)
            + "\nThe extraction recipe diverges from #406; downstream "
            "regression is meaningless. Diagnose before continuing."
        )
    return out


# ───────────────────────── figure phase ─────────────────────────


def emit_figures(
    all_cells: dict[str, dict],
    extraction_points: tuple[str, ...],
    layers: tuple[int, ...],
) -> None:
    """Two paper-style figures: (a) ρ heatmap across (metric × point) × layer
    for the headline cell; (b) winner's scatter vs ΔG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    headline = all_cells.get("loc_ep1")
    if not headline:
        logger.warning("No loc_ep1 cell — skipping figures")
        return

    # Build the (predictor-row × layer) ρ grid from headline entries.
    rows: dict[tuple[str, str, str | None], dict[int, float]] = {}
    for e in headline["entries"]:
        if "rho_nonstylized_deltag" not in e:
            continue
        key = (e["extraction_point"], e["metric"], e.get("sub_predictor"))
        rows.setdefault(key, {})[e["layer"]] = e["rho_nonstylized_deltag"]

    if not rows:
        logger.warning("No usable headline entries — skipping figures")
        return

    row_keys = sorted(rows.keys())
    row_labels = [f"{pt} · {m}" + (f" · {sub}" if sub else "") for (pt, m, sub) in row_keys]
    layer_list = sorted(layers)
    grid = np.full((len(row_keys), len(layer_list)), np.nan)
    for ri, key in enumerate(row_keys):
        for li, L in enumerate(layer_list):
            if L in rows[key]:
                grid[ri, li] = rows[key][L]

    # All-NaN row guard: an end_of_system × cloud-metric panel is N/A by
    # design — its whole row is NaN. Compute vmax from FINITE entries only,
    # fall back to 1.0 if the entire grid is NaN (defensive: emit a
    # warning + skip rather than crash on `np.nanmin([nan, nan])` =
    # RuntimeWarning + NaN vmax). Rendering still proceeds — the imshow
    # already draws NaN cells with the "bad" color (mpl default = white)
    # and the per-cell text annotation already skips non-finite values.
    finite_mask = np.isfinite(grid)
    if not finite_mask.any():
        logger.warning("metric_layer_grid_heatmap: grid is entirely NaN — skipping heatmap")
    else:
        finite_vals = grid[finite_mask]
        vmax = float(max(abs(finite_vals.min()), abs(finite_vals.max())))
        if vmax < 1e-6:
            vmax = 1.0
        # Per-row "all-NaN" tag → annotate the row label so a reviewer can
        # tell the empty row from "the cells happened to round to 0."
        any_finite_per_row = finite_mask.any(axis=1)
        row_labels = [
            f"{lbl}  (N/A)" if not any_finite_per_row[ri] else lbl
            for ri, lbl in enumerate(row_labels)
        ]

        # Build the figure with constrained_layout (the project default via
        # set_paper_style("blog")) and DO NOT call tight_layout — mixing
        # the two engines on a colorbar figure raises
        #   "Colorbar layout of new layout engine not compatible with old engine".
        # The canonical pattern in scripts/i474_cosine_followup.py either
        # keeps constrained_layout + omits tight_layout, OR explicitly
        # disables constrained_layout and uses fig.subplots_adjust(...).
        # The single-axis heatmap with one colorbar is fine with
        # constrained_layout alone; let it lay itself out.
        fig, ax = plt.subplots(
            figsize=(8.5, 0.35 * len(row_keys) + 2.0),
            constrained_layout=True,
        )
        # Mask NaNs so imshow draws the "bad" color cleanly instead of
        # whatever the RdBu_r endpoint maps to.
        masked_grid = np.ma.masked_invalid(grid)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked_grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(layer_list)))
        ax.set_xticklabels([f"L{L}" for L in layer_list])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Residual layer")
        ax.set_title(
            "loc_ep1 non-stylized · length-partial ρ(predictor, ΔG)",
            fontsize=10,
            loc="left",
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label("Spearman ρ", fontsize=8)
        for ri in range(len(row_keys)):
            for li in range(len(layer_list)):
                v = grid[ri, li]
                if np.isfinite(v):
                    ax.text(
                        li,
                        ri,
                        f"{v:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="black" if abs(v) < 0.5 * vmax else "white",
                    )
        # Use savefig_paper's bbox_inches="tight" by default (set in the
        # helper); do NOT call tight_layout() — would switch layout
        # engines after the colorbar has been added.
        savefig_paper(fig, "metric_layer_grid_heatmap", dir=str(FIGURE_DIR))
        plt.close(fig)

    # Winner scatter
    winner = select_winner(headline)
    if winner is None:
        logger.warning("No surviving predictor — skipping winner scatter")
        return
    pt = winner["extraction_point"]
    L = winner["layer"]
    m = winner["metric"]
    sub = winner.get("sub_predictor")
    payload = json.loads(Path(winner["file"]).read_text())
    cond_ids = payload["cond_ids"]
    pairs = _pairs(cond_ids, nonstylized_only=False)
    xv = _materialize_predictor_vector(payload, pairs, sub)
    G = _load_G("loc", 1)
    dg = np.array([G[a][b]["delta_g"] for a, b in pairs])
    sty = np.array([(a in STY_CIDS) or (b in STY_CIDS) for a, b in pairs])
    # constrained_layout is on by default (set_paper_style("blog")); use
    # it explicitly here too so the figure layout stays consistent with
    # the heatmap above and the savefig_paper helper's tight bbox handles
    # final margins. No tight_layout() — same engine-mix risk as the
    # colorbar figure (defensive, not strictly required without colorbar).
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    base = paper_palette_role("baseline")
    acc = paper_palette_role("accent")
    ax.scatter(
        xv[~sty],
        dg[~sty],
        s=22,
        c=base,
        alpha=0.6,
        edgecolor="white",
        lw=0.5,
        label=f"non-stylized (n={int((~sty).sum())})",
    )
    ax.scatter(
        xv[sty],
        dg[sty],
        s=28,
        c=acc,
        alpha=0.85,
        edgecolor="white",
        lw=0.5,
        label=f"touches stylized (n={int(sty.sum())})",
    )
    sub_str = f" · {sub}" if sub else ""
    ax.set_xlabel(f"{pt} · {m}{sub_str} (layer {L})")
    ax.set_ylabel("ΔG = trained − base log P(marker)")
    ax.set_title(
        f"Winner: {pt}/{m}{sub_str}/L{L} — CV R²={winner['cv_full_deltag']:.2f}",
        fontsize=10,
        loc="left",
    )
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(loc="best", frameon=True, fontsize=8)
    savefig_paper(fig, "winner_scatter_vs_deltaG", dir=str(FIGURE_DIR))
    plt.close(fig)


# ───────────────────────── dry-run smoke ─────────────────────────


def _synthetic_clouds(
    *,
    n_cond: int = 6,
    n_q: int = 20,
    hidden: int = 64,
    rng=None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build synthetic clouds with KNOWN structure for the metric smoke test.

    Returns (activations, cond_ids, fake_deltag_matrix). The fake ΔG is
    constructed to monotonically increase with the synthetic distance so a
    well-behaved predictor must produce a strong rank correlation.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cond_ids = [f"S{i}" for i in range(n_cond)]
    centers = rng.normal(size=(n_cond, hidden))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Spread = exponential schedule so ΔG-against-distance has a clear gradient.
    radii = np.linspace(1.0, 3.0, n_cond)
    arr = np.zeros((n_cond, n_q, hidden), dtype=np.float32)
    for i in range(n_cond):
        cloud = rng.normal(scale=0.5, size=(n_q, hidden))
        arr[i] = (centers[i] * radii[i])[None, :] + cloud
    # Construct a "fake ΔG" matrix that's a monotone function of cosine-dist.
    fake_dg = np.zeros((n_cond, n_cond), dtype=np.float32)
    for i in range(n_cond):
        for j in range(n_cond):
            if i == j:
                continue
            mu_i = arr[i].mean(axis=0)
            mu_j = arr[j].mean(axis=0)
            cs = mu_i @ mu_j / (np.linalg.norm(mu_i) * np.linalg.norm(mu_j))
            # Higher ΔG when the contexts are MORE similar (matches the
            # i474 direction: similar contexts → more transfer).
            fake_dg[i, j] = 10.0 * float(cs) + rng.normal(scale=0.5)
    return arr, cond_ids, fake_dg


def dry_run_smoke() -> dict:  # noqa: C901 — long flat smoke; each numbered block is one assertion class.
    """CPU-only sanity smoke. Exercises EVERY metric on synthetic clouds with
    known structure and confirms:

      - cosine ≈ 1 / euclidean ≈ 0 for identical clouds
      - C2ST AUC ≈ 0.5 for identical clouds
      - distances grow for well-separated clouds
      - PCA-whitened Fisher / Mahalanobis are finite and positive
      - Δ-spec coherence ≈ 1 for constant Δ, ≈ 0 for random Δ
      - PCA-via-Gram works at n ≪ d

    Returns a digest dict.
    """
    rng = np.random.default_rng(0)
    digest: dict[str, object] = {}

    # 1) Same-distribution clouds:
    #     - Cloud metrics (C2ST, MMD) → ≈ chance / ≈ 0 (they correctly say
    #       "indistinguishable"). With INDEPENDENT samples — not literally the
    #       same array, which trips twin-row memorization in C2ST.
    #     - Centroid metrics (cosine, euclidean) are NOT defined to be ≈ 0
    #       here: in high-D the centroid of two independent finite samples of
    #       N(0, I) is nearly orthogonal (curse of dimensionality), so centroid
    #       cosine_dist ≈ 1 even when the distributions match. We test centroid
    #       metrics under a stronger condition (same MEAN, tight cloud) below.
    X = rng.normal(size=(40, 64))
    Xa_id, Xb_id = X[:20], X[20:]  # two i.i.d. draws from N(0, I)
    # _c2st_auc now returns a DISTANCE (2*|AUC - 0.5|, [0, 1]).
    c2st_id = _c2st_auc(Xa_id, Xb_id)
    assert c2st_id < 0.50, f"C2ST distance on same-dist clouds was {c2st_id} (expected ~ 0)"
    mmd_id = _rbf_mmd_squared(Xa_id, Xb_id)
    # The unbiased MMD² is allowed to go slightly negative under H0; we test
    # that it stays close to zero in magnitude rather than asserting > 0.
    assert abs(mmd_id) < 0.10, f"Unbiased MMD² on same-dist clouds was {mmd_id} (expected |·| ~ 0)"
    # Centroid metrics on TIGHT clouds around a common large mean
    # (signal ≫ noise → centroids align):
    mu = rng.normal(scale=5.0, size=64)
    Xa_tight = mu + rng.normal(scale=0.1, size=(30, 64))
    Xb_tight = mu + rng.normal(scale=0.1, size=(30, 64))
    cd_tight = _centroid_cosine_distance(Xa_tight, Xb_tight)
    ed_tight = _centroid_euclidean(Xa_tight, Xb_tight)
    assert cd_tight < 1e-3, f"tight-same-mean cosine_dist was {cd_tight} (expected ≈ 0)"
    assert ed_tight < 0.5, f"tight-same-mean euclidean was {ed_tight} (expected small)"
    digest["identical_clouds"] = {
        "c2st_dist": c2st_id,
        "mmd2": mmd_id,
        "cosine_dist_tight_same_mean": cd_tight,
        "euclidean_tight_same_mean": ed_tight,
    }

    # 2) Well-separated clouds → cosine ≈ 2 (anti-parallel centroids),
    # C2ST distance ≈ 1, MMD² ≫ 0 (unbiased estimator).
    X1 = rng.normal(loc=10.0, size=(30, 64))
    X2 = rng.normal(loc=-10.0, size=(30, 64))
    cd = _centroid_cosine_distance(X1, X2)
    c2st_sep = _c2st_auc(X1, X2)
    mmd_sep = _rbf_mmd_squared(X1, X2)
    assert cd > 0.2, f"separated clouds cosine_dist was {cd} (expected substantial)"
    assert c2st_sep > 0.90, f"separated clouds C2ST distance was {c2st_sep} (expected ~ 1)"
    assert mmd_sep > 0.05, f"separated clouds MMD was {mmd_sep}"
    digest["separated_clouds"] = {
        "cosine_dist": cd,
        "c2st_dist": c2st_sep,
        "mmd2": mmd_sep,
    }

    # 3) Δ-spectrum — constant Δ → coherence ≈ 1
    base = rng.normal(size=(30, 64))
    Xb_const = base + np.ones(64) * 2.0  # constant displacement
    Xb_rand = base + rng.normal(size=(30, 64)) * 2.0
    spec_const = _delta_spectrum(base, Xb_const, k=8)
    spec_rand = _delta_spectrum(base, Xb_rand, k=8)
    assert spec_const["coherence"] > 0.95, f"constant-Δ coherence was {spec_const['coherence']}"
    assert spec_rand["coherence"] < 0.5, f"random-Δ coherence was {spec_rand['coherence']}"
    digest["delta_spec"] = {"const": spec_const, "random": spec_rand}

    # 4) Mahalanobis (per-pair pooled) well-conditioned at n ≪ d (n=30, d=64, k=8).
    # `_centroid_mahal` and `_fisher_distance` collapse to the same value for
    # the 2-cloud case; we test the surviving entry-point only.
    mahal = _centroid_mahal(X1, X2, k=8)
    assert np.isfinite(mahal) and mahal > 0, f"mahal was {mahal}"
    digest["mahal"] = float(mahal)

    # 4b) Context-pooled Mahalanobis works on ONE-vector-per-cond clouds
    # (end_of_system regime). Build a (n_cond, 1, H) array of distinct
    # per-cond centroids and confirm the pooled-cov state is finite +
    # the cross-cond distances are finite + non-zero.
    n_cond_eos = 6
    centroids = rng.normal(size=(n_cond_eos, 64))
    centroids *= 2.0  # spread them out so pooled cov is informative
    arr_eos = centroids[:, None, :].astype(np.float32)
    state = _build_context_pooled_mahal_state(arr_eos, k=4)
    assert state is not None, "pooled state was None on a clearly non-singular input"
    eos_distances = []
    for i in range(n_cond_eos):
        for j in range(n_cond_eos):
            if i == j:
                continue
            d_eos = _context_mahal_with_pooled_cov(arr_eos[i], arr_eos[j], state, i, j)
            assert np.isfinite(d_eos), f"end_of_system mahal_pooled_ctx was {d_eos}"
            eos_distances.append(d_eos)
    digest["end_of_system_mahal_pooled_ctx"] = {
        "n_pairs": len(eos_distances),
        "min": float(np.min(eos_distances)),
        "max": float(np.max(eos_distances)),
        "mean": float(np.mean(eos_distances)),
    }

    # 4c) Degenerate-input path now FAILS LOUD (round-3 fix B): the
    # eigenvalue gate in _build_context_pooled_mahal_state catches
    # all-zero / rank-deficient pooled covariances BEFORE the 1e-6 ridge,
    # so a collapsed input returns None (not a ridged spurious distance).
    # The full "explicit N/A row" path is exercised in check 9 below
    # via the _compute_metric_matrix wrapper.
    arr_singular = np.zeros((3, 1, 64), dtype=np.float32)
    singular_state = _build_context_pooled_mahal_state(arr_singular, k=4)
    assert singular_state is None, (
        f"round-3 fix B: collapsed centroids should yield None pooled-cov state, "
        f"got {singular_state!r}"
    )
    reason = _pop_pooled_failure_reason(arr_singular)
    # Both clauses guarded against `reason is None` — the second clause
    # used to bypass the truthy check and AttributeError on `.lower()`.
    _reason_lc = (reason or "").lower()
    assert reason is not None and ("collapsed" in _reason_lc or "rank" in _reason_lc), (
        f"round-3 fix B: missing or unexpected failure reason: {reason!r}"
    )
    digest["degenerate_pooled_state"] = {
        "state_returned": False,
        "failure_reason": reason,
    }

    # 4d) Single-vector subset path produces FINITE non-zero distances when
    # the centroids ARE distinct (this is the meaningful end_of_system run).
    arr_eos_single = np.array(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    s3 = _build_context_pooled_mahal_state(arr_eos_single, k=2)
    if s3 is not None:
        d_ab = _context_mahal_with_pooled_cov(arr_eos_single[0], arr_eos_single[1], s3, 0, 1)
        assert np.isfinite(d_ab) and d_ab > 0.05, (
            f"single-vector subset Mahalanobis was {d_ab} (expected positive finite)"
        )
        digest["single_vector_subset_mahal"] = float(d_ab)
    else:
        # State was None — pop the side-channel entry so the dict can't
        # leak under repeated smoke / production runs (round-4 fix #4).
        _pop_pooled_failure_reason(arr_eos_single)

    # 4e) n=2 distinct contexts (round-4 fix #1): with 2 contexts the
    # PCA-reduced dim collapses to 1, and `np.cov(Y.T, ddof=1)` returns a
    # 0-d scalar that previously crashed `np.linalg.eigvalsh`. The
    # `np.atleast_2d` wrap in `_build_context_pooled_mahal_state` makes
    # this case land cleanly — either as a finite Mahalanobis OR as an
    # explicit N/A (caught upstream by the eigengate), but never a crash.
    arr_n2 = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    try:
        s2 = _build_context_pooled_mahal_state(arr_n2, k=1)
        if s2 is not None:
            d_n2 = _context_mahal_with_pooled_cov(arr_n2[0], arr_n2[1], s2, 0, 1)
            assert np.isfinite(d_n2), f"n=2 Mahalanobis was {d_n2} (expected finite)"
            digest["n2_pooled_cov_finite"] = float(d_n2)
        else:
            # eigengate caught rank-deficient projection — clean N/A path,
            # not a crash. Pop the side-channel entry.
            reason_n2 = _pop_pooled_failure_reason(arr_n2)
            digest["n2_pooled_cov_finite"] = {
                "state_returned": False,
                "failure_reason": reason_n2,
            }
    except Exception as e:
        raise AssertionError(
            f"round-4 fix #1: n=2 distinct contexts must NOT crash _build_context_"
            f"pooled_mahal_state (np.cov returns scalar at k_eff=1 without atleast_2d). "
            f"Got: {e!r}"
        ) from e

    # 5) Gaussian sym-KL and W2 finite + positive
    gkl = _gaussian_sym_kl_in_subspace(X1, X2, k=8)
    w2 = _bures_wasserstein2(X1, X2, k=8)
    assert np.isfinite(gkl) and gkl > 0, f"gauss_kl was {gkl}"
    assert np.isfinite(w2) and w2 > 0, f"bures_w2 was {w2}"
    digest["gauss_kl_w2"] = {"sym_kl": float(gkl), "wass2": float(w2)}

    # 5b) MMD permutation null populated — pick a tiny synthetic (n_cond, n_q,
    # H) cloud, run the summary, confirm per-pair p-values + pooled summary
    # land in [0, 1] / are finite.
    arr_mmd, cond_ids_mmd, _ = _synthetic_clouds(n_cond=4, n_q=20, hidden=32)
    perm = _mmd_permutation_summary(
        arr_mmd, cond_ids_mmd, n_perm=30, variant="raw", n_pair_samples=4
    )
    assert perm["n_pair_samples_done"] >= 1, "MMD permutation summary returned no pairs"
    assert all(0.0 <= e["p_value"] <= 1.0 for e in perm["per_pair"]), "p_value out of [0, 1]"
    assert np.isfinite(perm["pooled_null_p95"]), "pooled_null_p95 was non-finite"
    digest["mmd_permutation"] = {
        "n_pair_samples_done": perm["n_pair_samples_done"],
        "min_p_value": min(e["p_value"] for e in perm["per_pair"]),
        "pooled_null_p95": perm["pooled_null_p95"],
    }

    # 5c) raw + centered variants both compute distinct matrices.
    arr_var, cond_ids_var, _ = _synthetic_clouds(n_cond=4, n_q=20, hidden=32)
    raw_payload = _compute_metric_matrix(
        arr_var,
        cond_ids_var,
        metric="cosine",
        extraction_point="last_prompt",
        pca_k=8,
        variant="raw",
    )
    cen_payload = _compute_metric_matrix(
        arr_var,
        cond_ids_var,
        metric="cosine",
        extraction_point="last_prompt",
        pca_k=8,
        variant="centered",
    )
    assert raw_payload["variant"] == "raw"
    assert cen_payload["variant"] == "centered"
    raw_val = raw_payload["matrix"][cond_ids_var[0]][cond_ids_var[1]]
    cen_val = cen_payload["matrix"][cond_ids_var[0]][cond_ids_var[1]]
    assert abs(raw_val - cen_val) > 1e-6, (
        f"centered variant indistinguishable from raw: {raw_val} vs {cen_val}"
    )
    digest["variants"] = {"raw_sample": raw_val, "centered_sample": cen_val}

    # 5d) Length-partialed LOOCV is consistent with the headline metric.
    # When predictor and DV are both monotone in the same covariate AND the
    # length covariate is constant, the length-partial LOOCV must return
    # the same answer as the bare-OLS LOOCV (no residualization happens).
    arr_cv, cond_ids_cv, fake_dg_cv = _synthetic_clouds(n_cond=6, n_q=20, hidden=32)
    pairs_cv = [
        (a, b) for i, a in enumerate(cond_ids_cv) for j, b in enumerate(cond_ids_cv) if i != j
    ]
    cdmat_cv = np.zeros((len(cond_ids_cv), len(cond_ids_cv)))
    for i in range(len(cond_ids_cv)):
        for j in range(len(cond_ids_cv)):
            if i == j:
                continue
            cdmat_cv[i, j] = _centroid_cosine_distance(arr_cv[i], arr_cv[j])
    name_to_idx_cv = {n: i for i, n in enumerate(cond_ids_cv)}
    xv_cv = np.array([cdmat_cv[name_to_idx_cv[a], name_to_idx_cv[b]] for a, b in pairs_cv])
    yv_cv = np.array([fake_dg_cv[name_to_idx_cv[a], name_to_idx_cv[b]] for a, b in pairs_cv])
    src_cv = [a for a, _ in pairs_cv]
    tgt_cv = [b for _, b in pairs_cv]
    # Three regimes: bare CV vs length-partial CV.
    #   (a) covar independent of x,y     → length-partial CV ≈ bare CV
    #   (b) covar perfectly tracks x     → length-partial CV ≪ bare CV
    #       (the signal vanishes once length is partialed out — confound case)
    cv_bare = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=None)
    rng_cv = np.random.default_rng(7)
    covar_indep = rng_cv.normal(size=xv_cv.shape)
    cv_lp_indep = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=covar_indep)
    assert abs(cv_bare - cv_lp_indep) < 0.2, (
        f"length-partial CV on independent covar should track bare CV: "
        f"bare={cv_bare:.3f} length-partial={cv_lp_indep:.3f}"
    )
    cv_lp_confound = _loocv_r2(xv_cv, yv_cv, src_cv, tgt_cv, covar=xv_cv.copy())
    assert cv_lp_confound < cv_bare - 0.1, (
        f"length-partial CV must collapse when covar==predictor (confound case): "
        f"bare={cv_bare:.3f} length-partial-confound={cv_lp_confound:.3f}"
    )
    digest["loocv_length_partialed"] = {
        "bare_R2": cv_bare,
        "length_partial_R2_independent_covar": cv_lp_indep,
        "length_partial_R2_confound_covar": cv_lp_confound,
    }

    # 6) End-to-end: synthetic regression — cosine_dist of synthetic clouds
    #    should rank-correlate with the synthetic ΔG matrix.
    arr, cond_ids, fake_dg = _synthetic_clouds()
    pairs = [(a, b) for i, a in enumerate(cond_ids) for j, b in enumerate(cond_ids) if i != j]
    cdmat = np.zeros((len(cond_ids), len(cond_ids)))
    for i, _ in enumerate(cond_ids):
        for j, _ in enumerate(cond_ids):
            if i == j:
                continue
            cdmat[i, j] = _centroid_cosine_distance(arr[i], arr[j])
    name_to_idx = {n: i for i, n in enumerate(cond_ids)}
    xv = np.array([cdmat[name_to_idx[a], name_to_idx[b]] for a, b in pairs])
    yv = np.array([fake_dg[name_to_idx[a], name_to_idx[b]] for a, b in pairs])
    # length covariate is constant here; use a flat covar so partial = bare Spearman
    rho, _p = _length_partial(xv, yv, np.zeros_like(xv) + 1.0)
    digest["synthetic_regression_rho"] = float(rho)
    assert abs(rho) > 0.3, f"synthetic regression rho was {rho} (expected substantial)"
    # CV must run cleanly too
    cv = _loocv_r2(xv, yv, [a for a, _ in pairs], [b for _, b in pairs], covar=np.ones_like(xv))
    digest["synthetic_cv_r2"] = float(cv)

    # 7) Forward-hook extraction (round-6 fix): production extraction now
    # captures via _LayerHookCapture on `model.model.layers[L]`. The
    # reference check below confirms (a) `hidden_states[L+1]` matches the
    # block-L hook output for inner layers (the round-2 fix held), AND
    # (b) the new _LayerHookCapture wrapper captures + clears properly
    # across multiple forward passes on a tiny CPU model.
    # GPT-2 doesn't reproduce Qwen's last-layer post-norm quirk on this
    # tiny model (only Qwen-class architectures expose it on the final
    # block), so the per-architecture verification of that L=last quirk
    # happens at the GPU cross-check (cosine vs C_L*.json under the new
    # 3e-3 tolerance).
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok_ref = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mdl_ref = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    except (OSError, ConnectionError, ImportError, ModuleNotFoundError) as e:
        # No network / hub access / missing dep → genuinely an environment
        # issue, not a regression. Skip with a clear reason.
        digest["hidden_states_indexing"] = {
            "ok": False,
            "reason": f"reference check skipped — environment: {e!r}",
        }
        digest["layer_hook_capture"] = digest["hidden_states_indexing"]
    else:
        mdl_ref.eval()
        ids = tok_ref("hello world from issue 493", return_tensors="pt")
        hook_capture_raw: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def _hook(_mod, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hook_capture_raw[layer_idx] = hs.detach().clone()

            return _hook

        target_layer = 1  # 0-indexed; tiny-random-gpt2 has 4 blocks
        h = mdl_ref.h[target_layer].register_forward_hook(_make_hook(target_layer))
        try:
            with torch.no_grad():
                fwd = mdl_ref(**ids, output_hidden_states=True)
        finally:
            h.remove()
        from_hook = hook_capture_raw[target_layer]
        from_tuple_off_by_one = fwd.hidden_states[target_layer]  # WRONG
        from_tuple_correct = fwd.hidden_states[target_layer + 1]  # MATCHES inner layers
        # AssertionError below propagates — that's the regression signal.
        assert torch.allclose(from_hook, from_tuple_correct, atol=1e-6), (
            "hidden_states[L+1] no longer matches the block-L forward-hook output "
            "on inner layers — convention drift in this transformers version?"
        )
        assert not torch.allclose(from_hook, from_tuple_off_by_one, atol=1e-6), (
            "hidden_states[L] now matches the hook — convention changed upstream?"
        )
        digest["hidden_states_indexing"] = {
            "convention": "hidden_states[L+1] == block[L] output (inner layers)",
            "ok": True,
        }

        # 7b) _LayerHookCapture context manager — confirms the production
        # capture path (a) registers + tears down hooks cleanly, (b) clears
        # buffers per reset() so probes don't leak, (c) captures the SAME
        # tensor the bare forward_hook would.
        # tiny-random-gpt2 uses `.h[L]` not `.model.layers[L]`; wrap a
        # tiny adapter so we can exercise _LayerHookCapture on it.
        class _GPT2Adapter:
            def __init__(self, m):
                self.model = type("inner", (), {"layers": m.h})()

        adapter = _GPT2Adapter(mdl_ref)
        with _LayerHookCapture(adapter, (target_layer,)) as cap:
            # 1st forward.
            cap.reset()
            with torch.no_grad():
                _ = mdl_ref(**ids)
            cap_a = cap.last_layer(target_layer).clone()
            assert cap_a.shape == from_hook.shape, (cap_a.shape, from_hook.shape)
            assert torch.allclose(cap_a, from_hook, atol=1e-6), (
                "_LayerHookCapture output disagrees with the bare hook capture"
            )
            # 2nd forward after reset — buffer must repopulate cleanly.
            cap.reset()
            with torch.no_grad():
                _ = mdl_ref(**ids)
            cap_b = cap.last_layer(target_layer)
            assert torch.allclose(cap_a, cap_b, atol=1e-6), (
                "_LayerHookCapture second forward diverged from the first "
                "(reset / re-capture path broken)"
            )
        digest["layer_hook_capture"] = {
            "matches_raw_hook": True,
            "reset_then_recapture_consistent": True,
        }

    # 8) Winner-selection nonstylized-g_logprob guard (round-3 issue A).
    # Build a synthetic headline cell with two predictor entries:
    #   - "stylized_carry": full panel ΔG positive, FULL g_logprob positive,
    #     but non-stylized g_logprob NULL → must NOT win (round-2 bug:
    #     this would have won because it checked rho_FULL_glogp).
    #   - "honest": full panel ΔG positive, BOTH full + non-stylized
    #     g_logprob positive → must win.
    fake_cell = {
        "entries": [
            {
                "extraction_point": "last_prompt",
                "layer": 21,
                "metric": "stylized_carry",
                "variant": "raw",
                "sub_predictor": None,
                "n_nonstylized": 156,
                "rho_full_deltag": 0.70,
                "rho_nonstylized_deltag": 0.30,
                "rho_full_glogp": 0.55,
                # Round-3 fix: this should DISQUALIFY (was previously won the
                # winner-selection under round-2's full-panel-only check).
                "rho_nonstylized_glogp": 0.01,
                "cv_full_deltag": 0.50,
            },
            {
                "extraction_point": "last_prompt",
                "layer": 21,
                "metric": "honest",
                "variant": "raw",
                "sub_predictor": None,
                "n_nonstylized": 156,
                "rho_full_deltag": 0.60,
                "rho_nonstylized_deltag": 0.40,
                "rho_full_glogp": 0.50,
                "rho_nonstylized_glogp": 0.35,
                "cv_full_deltag": 0.40,
            },
        ]
    }
    winner = select_winner(fake_cell)
    assert winner is not None and winner["metric"] == "honest", (
        f"round-3 winner-selection: expected 'honest', got "
        f"{None if winner is None else winner.get('metric')}"
    )
    digest["winner_selection_nonstylized_guard"] = {
        "winner_metric": winner["metric"],
        "rho_nonstylized_glogp_used": True,
    }

    # 8b) Subpanel diagnostic-only path: a predictor with n_nonstylized
    # below SUBPANEL_MIN_NONSTYLIZED_N must be marked diagnostic_only and
    # NOT win the headline, even if every rho/CV value looks great.
    subpanel_cell = {
        "entries": [
            {
                "extraction_point": "end_of_system",
                "layer": 21,
                "metric": "mahal_pooled_ctx",
                "variant": "raw",
                "sub_predictor": None,
                "n_nonstylized": 2,  # Class-A nonstylized restriction
                "rho_full_deltag": 0.80,
                "rho_nonstylized_deltag": float("nan"),
                "rho_full_glogp": 0.70,
                "rho_nonstylized_glogp": float("nan"),
                "cv_full_deltag": 0.55,
            },
        ]
    }
    winner_sub = select_winner(subpanel_cell)
    assert winner_sub is None, f"round-3 subpanel exemption: expected None winner, got {winner_sub}"
    assert subpanel_cell["entries"][0].get("diagnostic_only") is True, (
        "round-3 subpanel exemption: entry was not tagged diagnostic_only"
    )
    digest["winner_subpanel_diagnostic_only"] = {
        "winner_is_none": True,
        "entry_tagged_diagnostic_only": True,
    }

    # 9) Singular pooled-cov path returns explicit N/A row with a reason
    # (round-3 issue B — the round-2 ridge silently turned a degenerate
    # cov into a 0.0 distance, which Codex flagged as spurious).
    arr_collapsed = np.zeros((3, 1, 32), dtype=np.float32)
    payload_collapsed = _compute_metric_matrix(
        arr_collapsed,
        ["A", "B", "C"],
        metric="mahal_pooled_ctx",
        extraction_point="end_of_system",
        pca_k=8,
        variant="raw",
    )
    assert payload_collapsed["matrix"] is None, (
        f"round-3 singular pooled-cov: expected matrix=None, got {payload_collapsed.get('matrix')}"
    )
    assert payload_collapsed.get("n_a"), (
        "round-3 singular pooled-cov: missing N/A reason in payload"
    )
    digest["singular_pooled_cov_emits_na"] = {
        "matrix_is_none": True,
        "n_a_reason": payload_collapsed["n_a"],
    }

    # 10) End-to-end degenerate-subpanel no-crash regression test (round-5
    # fix #1-4): reproduces the exact GPU smoke shape that crashed
    # `_loocv_r2 → np.polyfit` with SVD non-convergence. Three Class-A
    # contexts (~ 6 ordered pairs → tiny LOOCV folds), an all-NaN
    # cloud-metric column (end_of_system × MMD = N/A by design), and a
    # constant-x predictor column (zero variance). Calls run_regression →
    # select_winner end-to-end and asserts:
    #   (a) no exception is raised,
    #   (b) the degenerate columns get status="degenerate" + NaN rho/CV,
    #   (c) the un-degenerate column still produces a finite rho/CV,
    #   (d) select_winner returns either the finite predictor or None
    #       (subpanel-too-small) but NEVER promotes a degenerate row.
    # The figure path is exercised separately by the prior emit_figures
    # synthetic smoke (#commit 459993c82); here we only confirm the
    # numerics don't crash.
    import json as _json
    import tempfile as _tmp

    _tmp_dir = _tmp.mkdtemp()
    _orig_metric_dir = mod_globals_metric_dir = METRIC_DIR  # noqa: F841
    _orig_regr_dir = REGR_DIR
    cond_a = ["A1", "A2", "A3"]  # tiny Class-A subpanel (~6 ordered pairs)

    # Write three metric files: (i) a healthy predictor with rich variance,
    # (ii) an all-NaN cloud-metric column at end_of_system, (iii) a
    # constant-x column at end_of_system.
    metric_files = []
    healthy_matrix = {
        a: {b: (0.0 if a == b else 0.1 + 0.05 * (hash(a + b) % 7)) for b in cond_a} for a in cond_a
    }
    healthy_payload = {
        "schema_version": 1,
        "extraction_point": "end_of_system",
        "layer": 21,
        "metric": "cosine",
        "variant": "raw",
        "pca_k": 4,
        "cond_ids": cond_a,
        "matrix": healthy_matrix,
        "git_sha": "test",
        "timestamp_utc": "now",
    }
    healthy_path = Path(_tmp_dir) / "end_of_system__layer21__cosine__raw.json"
    healthy_path.write_text(_json.dumps(healthy_payload))
    metric_files.append(healthy_path)
    all_nan_matrix = {a: {b: (None if a != b else 0.0) for b in cond_a} for a in cond_a}
    all_nan_payload = {**healthy_payload, "metric": "mmd", "matrix": all_nan_matrix}
    all_nan_path = Path(_tmp_dir) / "end_of_system__layer21__mmd__raw.json"
    all_nan_path.write_text(_json.dumps(all_nan_payload))
    metric_files.append(all_nan_path)
    const_matrix = {a: {b: (0.5 if a != b else 0.0) for b in cond_a} for a in cond_a}
    const_payload = {**healthy_payload, "metric": "euclidean", "matrix": const_matrix}
    const_path = Path(_tmp_dir) / "end_of_system__layer21__euclidean__raw.json"
    const_path.write_text(_json.dumps(const_payload))
    metric_files.append(const_path)

    # Build a synthetic G matrix for one (arm, ep) so run_regression has
    # a DV to read. We bypass run_regression's file-IO by calling the
    # internal helpers directly — this is a smoke test, not a full
    # production rehearsal.
    G_fake = {
        a: {b: {"delta_g": 1.0 + 0.3 * (hash(a + b) % 5), "g_logprob": -0.5} for b in cond_a}
        for a in cond_a
    }
    prompt_tokens_fake = {a: {b: 50 + (hash(a + b) % 30) for b in cond_a} for a in cond_a}
    # NOTE: the actual function ID — `run_regression` — reads from disk;
    # for the smoke we want to drive the same code path WITHOUT touching
    # the real eval_results tree. So we invoke the per-cell logic
    # in-process: enumerate predictors over the temp metric files, then
    # for each call into _length_partial + _loocv_r2 + select_winner.
    enum_rows = _enumerate_predictors(metric_files)
    entries_smoke = []
    pairs_primary = [(a, b) for a in cond_a for b in cond_a if a != b]
    dg_arr = np.array([G_fake[a][b]["delta_g"] for a, b in pairs_primary])
    # g_logprob omitted from the smoke — the no-crash sanity is the same on
    # the single ΔG path; full per-DV regression is exercised by GPU runs.
    ln_arr = np.array([np.log(prompt_tokens_fake[a][b]) for a, b in pairs_primary])
    src = [a for a, _ in pairs_primary]
    tgt = [b for _, b in pairs_primary]
    for desc in enum_rows:
        payload = _json.loads(Path(desc["file"]).read_text())
        xv = _materialize_predictor_vector(payload, pairs_primary, desc["sub_predictor"])
        if xv is None:
            entries_smoke.append({**desc, "status": "N/A (matrix is None or missing pair)"})
            continue
        finite_mask = np.isfinite(xv)
        n_finite = int(finite_mask.sum())
        is_degen = n_finite < _MIN_FINITE_FOR_REGRESSION or (
            n_finite > 0 and float(xv[finite_mask].var()) < _CONSTANT_VAR_TOL
        )
        if is_degen:
            entries_smoke.append(
                {
                    **desc,
                    "status": "degenerate",
                    "n_finite_primary": n_finite,
                    "n_nonstylized": 0,
                    "rho_full_deltag": float("nan"),
                    "rho_nonstylized_deltag": float("nan"),
                    "rho_full_glogp": float("nan"),
                    "rho_nonstylized_glogp": float("nan"),
                    "cv_full_deltag": float("nan"),
                }
            )
            continue
        rho, _ = _length_partial(xv, dg_arr, ln_arr)
        cv = _loocv_r2(xv, dg_arr, src, tgt, covar=ln_arr)
        entries_smoke.append(
            {
                **desc,
                "n_nonstylized": 0,  # Class-A subpanel — nonstylized is empty
                "rho_full_deltag": float(rho),
                "rho_nonstylized_deltag": float("nan"),
                "rho_full_glogp": float("nan"),
                "rho_nonstylized_glogp": float("nan"),
                "cv_full_deltag": float(cv),
            }
        )

    # Verify each predictor's regression outcome.
    by_metric = {e["metric"]: e for e in entries_smoke}
    assert by_metric["mmd"]["status"] in {"N/A (matrix is None or missing pair)", "degenerate"}, (
        f"all-NaN MMD column should be N/A or degenerate, got {by_metric['mmd']}"
    )
    assert by_metric["euclidean"]["status"] == "degenerate", (
        f"constant-x euclidean column should be degenerate, got {by_metric['euclidean']}"
    )
    healthy_entry = by_metric["cosine"]
    assert np.isfinite(healthy_entry["rho_full_deltag"]), (
        f"healthy cosine predictor produced NaN rho — degenerate-input filter "
        f"too aggressive? entry={healthy_entry}"
    )
    # Confirm select_winner doesn't crash AND doesn't promote a degenerate row.
    fake_cell = {"entries": entries_smoke}
    winner = select_winner(fake_cell)
    if winner is not None:
        assert winner.get("status") != "degenerate", (
            f"select_winner promoted a degenerate entry: {winner}"
        )
    digest["degenerate_subpanel_no_crash"] = {
        "n_enumerated": len(enum_rows),
        "n_entries": len(entries_smoke),
        "healthy_rho": healthy_entry["rho_full_deltag"],
        "healthy_cv": healthy_entry["cv_full_deltag"],
        "winner_is_degenerate": bool(winner is not None and winner.get("status") == "degenerate"),
        "winner_metric": winner.get("metric") if winner else None,
    }

    # 11) Per-layer cosine cross-check tolerance map (round-7 fix). Locks
    # in the documented L27 bf16-accumulation floor at 1e-2 + the inner-
    # layer 3e-3 default. A future revision that:
    #   - drops the L27 entry (back to a single scalar that re-fails L27),
    #   - tightens L27 below the observed 6.15e-3 floor,
    #   - silently loosens the inner-layer default above 3e-3,
    # MUST fail this smoke. See the comment block above
    # COSINE_REPRO_TOLERANCES for the GPU-verified per-layer diff progression
    # and the diagnostic (Pearson r=0.999976, Spearman ρ=0.999362) that
    # justifies the L27 relaxation.
    assert isinstance(COSINE_REPRO_TOLERANCES, dict), (
        "round-7 fix: COSINE_REPRO_TOLERANCES must be a dict[int|str, float]"
    )
    assert 27 in COSINE_REPRO_TOLERANCES, (
        "round-7 fix: L27 must have an explicit per-layer tolerance (the deepest "
        "layer carries a documented bf16-accumulation floor — a scalar tolerance "
        "of 3e-3 is too tight for L27, see round-7 diagnostic)"
    )
    assert "default" in COSINE_REPRO_TOLERANCES, (
        "round-7 fix: a 'default' tolerance must exist for layers without an "
        "explicit per-layer entry (L0/L5/L11/L15/L21 expect 3e-3)"
    )
    # The L27 floor observed on round-7 GPU extract was 6.15e-3; the relaxed
    # tolerance must be >= 1e-2 so a re-extract on a slightly drifted
    # hardware/dtype/transformers-version doesn't immediately re-trip.
    L27_OBSERVED_BF16_FLOOR = 6.15e-3
    L27_MIN_TOLERANCE = 1e-2
    assert cosine_tolerance_for_layer(27) >= L27_MIN_TOLERANCE, (
        f"round-7 fix: L27 tolerance ({cosine_tolerance_for_layer(27):.2e}) is "
        f"tighter than the {L27_MIN_TOLERANCE:.0e} floor recorded by the round-7 "
        f"diagnostic. Observed GPU bf16 floor at L27 = {L27_OBSERVED_BF16_FLOOR:.2e} "
        "vs #406 cosine reference. Either re-run the diagnostic and document a "
        "lower floor, or restore L27 to >= 1e-2."
    )
    # Inner-layer default must not silently drift upward (would hide real
    # extraction bugs at L0/L5/L11/L15/L21 — the deepest passing reference
    # layer L21 sits at 2.70e-3 with the round-6 code path).
    DEFAULT_MAX_TOLERANCE = 3e-3
    assert cosine_tolerance_for_layer(21) <= DEFAULT_MAX_TOLERANCE, (
        f"round-7 fix: inner-layer (L21) tolerance ({cosine_tolerance_for_layer(21):.2e}) "
        f"loosened beyond {DEFAULT_MAX_TOLERANCE:.0e}. L21 max |diff| on round-6 was "
        "2.70e-3 (well below the 3e-3 default); a looser default would let real "
        "extraction bugs through. Tighten back to 3e-3 OR document the new diagnostic."
    )
    # Functional check: feed a synthetic 2-cond layer payload that should
    # PASS (zero diff) at the inner-layer default, and a synthetic payload
    # that should FAIL strict=True at 3e-3 but PASS at L27's 1e-2.
    # This exercises the per-layer dispatch path end-to-end.
    H_SYN = 8
    np_rng = np.random.default_rng(493)
    syn_centroid_a = np_rng.normal(size=H_SYN).astype(np.float32)
    syn_centroid_b = np_rng.normal(size=H_SYN).astype(np.float32)
    # Build (2 conds, 4 probes, H) arrays clustered around the centroids.
    syn_arr = np.stack(
        [
            np.tile(syn_centroid_a, (4, 1)),
            np.tile(syn_centroid_b, (4, 1)),
        ]
    )
    # Compute the "true" cosine distance from this exact data.
    true_cd_ab = _centroid_cosine_distance(syn_arr[0], syn_arr[1])
    # Build an "existing #406" matrix that matches exactly.
    existing_match = {
        "A": {"B": float(true_cd_ab)},
        "B": {"A": float(true_cd_ab)},
    }
    payload_match = {"activations": syn_arr, "cond_ids": ["A", "B"]}
    # At an inner layer (L=5), strict must PASS (zero diff < 3e-3).
    res_inner = reproduce_last_token_cosine_check(
        {5: payload_match},
        {5: {"matrix": existing_match}},
        cond_ids=["A", "B"],
        strict=True,
    )
    assert res_inner[5]["ok"], (
        f"round-7 fix: synthetic exact-match payload at L5 must PASS strict (got {res_inner[5]})"
    )
    assert abs(res_inner[5]["tolerance"] - 3e-3) < 1e-9, (
        f"round-7 fix: L5 tolerance should be 3e-3 (default), got {res_inner[5]['tolerance']}"
    )
    # At L27, the same exact-match payload also PASSes (zero diff < 1e-2),
    # AND the recorded tolerance is the L27-specific 1e-2 (not the default).
    res_l27 = reproduce_last_token_cosine_check(
        {27: payload_match},
        {27: {"matrix": existing_match}},
        cond_ids=["A", "B"],
        strict=True,
    )
    assert res_l27[27]["ok"], (
        f"round-7 fix: synthetic exact-match payload at L27 must PASS strict (got {res_l27[27]})"
    )
    assert abs(res_l27[27]["tolerance"] - 1e-2) < 1e-9, (
        f"round-7 fix: L27 tolerance should be 1e-2 (per-layer override), "
        f"got {res_l27[27]['tolerance']}"
    )
    # A 7e-3 perturbation at L5 (inner) must FAIL strict; the same perturbation
    # at L27 (relaxed) must PASS. This confirms the dispatch actually routes.
    existing_perturbed = {
        "A": {"B": float(true_cd_ab + 7e-3)},
        "B": {"A": float(true_cd_ab + 7e-3)},
    }
    raised_l5 = False
    try:
        reproduce_last_token_cosine_check(
            {5: payload_match},
            {5: {"matrix": existing_perturbed}},
            cond_ids=["A", "B"],
            strict=True,
        )
    except AssertionError:
        raised_l5 = True
    assert raised_l5, (
        "round-7 fix: 7e-3 perturbation at L5 (inner, 3e-3 tolerance) must FAIL "
        "strict cross-check — the inner-layer gate would be too loose otherwise."
    )
    res_l27_perturbed = reproduce_last_token_cosine_check(
        {27: payload_match},
        {27: {"matrix": existing_perturbed}},
        cond_ids=["A", "B"],
        strict=True,
    )
    assert res_l27_perturbed[27]["ok"], (
        "round-7 fix: 7e-3 perturbation at L27 (1e-2 tolerance) must PASS strict "
        "cross-check — the L27 relaxation is supposed to admit observed bf16 noise."
    )
    digest["cosine_tolerance_map"] = {
        "l27_tolerance": cosine_tolerance_for_layer(27),
        "default_tolerance": cosine_tolerance_for_layer(0),
        "l21_tolerance": cosine_tolerance_for_layer(21),
        "l5_inner_dispatch_passes_exact": True,
        "l27_relaxed_dispatch_passes_7e-3": True,
        "l5_inner_dispatch_fails_7e-3": True,
    }

    return digest


# ───────────────────────── CLI driver ─────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #493 extraction-point × metric × layer bake-off "
        "for marker-transfer prediction."
    )
    p.add_argument(
        "--phase",
        # `extraction` is a synonym for `extract` (matches the docs / the
        # report-back section header "extraction"); both route through the
        # same branch below.
        choices=("all", "extract", "extraction", "metrics", "regress", "figures", "smoke"),
        default="all",
        help="Which phase to run. 'all' runs extract → metrics → regress → figures. "
        "'extract' and 'extraction' are synonyms. 'smoke' runs the synthetic CPU sanity "
        "check only.",
    )
    p.add_argument(
        "--extraction-points",
        nargs="+",
        default=list(DEFAULT_EXTRACTION_POINTS),
        choices=list(DEFAULT_EXTRACTION_POINTS),
        help="Which extraction points to compute (default: all 3).",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
        help="Which residual layers to extract / score (default: 0 5 7 11 14 15 21 27).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(ALL_METRICS),
        choices=list(ALL_METRICS),
        help="Which metrics to compute.",
    )
    p.add_argument(
        "--transformations",
        nargs="+",
        default=None,
        help="Optional subset of cond cids (e.g. A1 A2 B1). Default: all 16.",
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Subset of the 50 Q_test probes. Use a small value (e.g. 4) for a pod smoke slice.",
    )
    p.add_argument(
        "--max-response-tokens",
        type=int,
        default=512,
        # 512 covers Qwen-2.5-7B's natural ~150-token response median with
        # ~3x headroom (round-1's 128 truncated below the median and biased
        # mean_response toward early tokens). #460 uses 1024 for marker
        # training generation; we keep 512 here because mean_response is a
        # representational summary, not the marker-leakage DV — but the
        # `extraction_truncation.json` summary surfaces the rate so a real
        # run that drifts long can be re-launched at 1024.
        help=(
            "max_new_tokens for the mean_response greedy decode. Default 512 "
            "covers Qwen-2.5-7B's natural ~150-token response median with "
            "headroom; #460 used 1024 for marker-leakage training generation. "
            "Truncation rate is logged in extraction_truncation.json."
        ),
    )
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(DEFAULT_ARMS))
    p.add_argument("--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help=(
            "Bind CUDA_VISIBLE_DEVICES=<gpu-id> BEFORE any CUDA call, then load the "
            "model on cuda:0. Matches the i474/i415 parallel-launch convention; "
            "leave unset to inherit the caller's CUDA_VISIBLE_DEVICES."
        ),
    )
    p.add_argument("--pca-k", type=int, default=PCA_DEFAULT_K)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the model load + extraction; only run metric/regression "
        "smoke (synthetic clouds + import/plumbing checks).",
    )
    # ── #502 additions ─────────────────────────────────────────────────────────
    p.add_argument(
        "--bakeoff-root",
        type=Path,
        default=None,
        help=(
            "Override BAKEOFF_DIR (default: eval_results/issue_493/bakeoff). "
            "#502 sets this to eval_results/issue_502/bakeoff so 28-layer/500-probe "
            "artifacts never collide with the cached #493 8-layer/50-probe artifacts."
        ),
    )
    p.add_argument(
        "--figures-root",
        type=Path,
        default=None,
        help="Override FIGURE_DIR (default: figures/issue_493).",
    )
    p.add_argument(
        "--probe-pool",
        type=Path,
        default=None,
        help=(
            "Path to a 500-pool JSON (e.g. eval_results/issue_502/probes_500.json) "
            "produced by scripts/issue502_generate_probes.py. The first 50 entries "
            "MUST be byte-identical to q_test; the rest disjoint from q_train + q_test. "
            "Default None = use the 50 q_test probes (matches #493)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Probes per generate() batch (#502 batched extraction). "
            "1 = serial (default; matches #493). Larger values are faster on "
            "mean_response (the bottleneck) at the cost of more GPU memory."
        ),
    )
    p.add_argument(
        "--batched",
        action="store_true",
        help=(
            "Route to the batched extraction code path. Auto-enabled when "
            "--batch-size > 1 OR --probe-pool is set OR --partitioned is set."
        ),
    )
    p.add_argument(
        "--partitioned",
        action="store_true",
        help=(
            "Write per-transformation activation files (one file per "
            "(point, layer, cond)) so multiple GPU processes can write disjoint "
            "files in parallel. Pair with scripts/issue502_dispatch.py which "
            "spawns one process per GPU and merges after."
        ),
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Skip extraction; only merge any existing partitioned activation "
            "files into the canonical <point>__layer<L>.pt shape and exit. "
            "Used by the dispatcher post-fan-in."
        ),
    )
    p.add_argument(
        "--no-next-token-js",
        action="store_true",
        help=(
            "Skip the next-token JS baseline capture (default: capture it). "
            "Reuses the last_prompt forward pass; near-free."
        ),
    )
    # ── #509 additions (panel-agnostic + custom probe pool) ─────────────────
    p.add_argument(
        "--conditions-registry",
        type=str,
        default=DEFAULT_CONDITIONS_MODULE,
        help=(
            "Dotted-module path to the conditions registry (default: i406's "
            "16-cond panel). Override e.g. with "
            "'explore_persona_space.experiments.i509_fact_conditions' for "
            "the #509 fact arm (9 personas) or '...i509_syco_conditions' for "
            "the sycophancy arm (24 personas). The target module must expose "
            "CONDITIONS (list[Condition]) and CONDITIONS_BY_ID (dict) with "
            "the same shape as i406_conditions."
        ),
    )
    p.add_argument(
        "--probe-pool-mode",
        choices=("q_test_strict", "custom"),
        default="q_test_strict",
        help=(
            "Probe-pool validation mode. 'q_test_strict' (default) enforces "
            "the bake-off's original gate: first 50 entries byte-identical "
            "to q_test, new entries disjoint from q_train + q_test. 'custom' "
            "bypasses the q_test prefix + disjointness check (only checks "
            "list[str] + non-empty + unique) so panel-specific pools like "
            "#411's sycophancy eval_50 probes can be passed without the "
            "q_test prefix. The mode is written to extraction metadata for "
            "audit. Required for the #509 sycophancy arm."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — top-level CLI phase dispatcher.
    args = _build_argparser().parse_args(argv)
    # #509: bind the conditions registry override BEFORE any function that
    # calls ``_load_conditions_registry()`` runs. Doing it here means every
    # downstream import site (run_extraction_batched, merge, main's
    # expected-cond resolver, etc.) sees the same panel.
    if args.conditions_registry and args.conditions_registry != DEFAULT_CONDITIONS_MODULE:
        _set_conditions_module(args.conditions_registry)
        logger.info("Conditions registry overridden: %s", args.conditions_registry)
    # BIND CUDA_VISIBLE_DEVICES *BEFORE* any cuda call (project convention,
    # see scripts/i474_*.py / scripts/recompute_predictors_i415.py + the
    # CLAUDE.md `+gpu_id=N` clobber gotcha). Once bound the local device is
    # always cuda:0 from the model's point of view.
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:0"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # #502: override output roots (BAKEOFF_DIR / ACT_DIR / METRIC_DIR / REGR_DIR /
    # FIGURE_DIR) BEFORE any phase runs, so all downstream paths land in the
    # #502 tree instead of overwriting #493.
    if args.bakeoff_root is not None or args.figures_root is not None:
        bakeoff_root = args.bakeoff_root or BAKEOFF_DIR
        _set_roots(bakeoff_root, args.figures_root)
        logger.info(
            "Output roots overridden: BAKEOFF_DIR=%s FIGURE_DIR=%s", BAKEOFF_DIR, FIGURE_DIR
        )

    # Auto-enable batched mode when any of its flags / inputs imply it.
    batched_mode = bool(
        args.batched or args.batch_size > 1 or args.probe_pool is not None or args.partitioned
    )

    # Persist a meta.json snapshot at every entry (overwritten = fine).
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)

    def _to_jsonable(v):
        if isinstance(v, tuple):
            return list(v)
        if isinstance(v, Path):
            return str(v)
        return v

    _write_json_atomic(
        BAKEOFF_DIR / "meta.json",
        {
            "schema_version": 1,
            "args": {k: _to_jsonable(v) for k, v in vars(args).items()},
            "git_sha": _git_sha(),
            "env": _env_versions(),
            "started_at": _now_iso(),
        },
    )

    # SMOKE path — CPU only.
    if args.phase == "smoke" or args.dry_run:
        logger.info("Running synthetic metric/regression smoke (CPU-only)…")
        digest = dry_run_smoke()
        out = BAKEOFF_DIR / "smoke_digest.json"
        _write_json_atomic(
            out,
            {
                "schema_version": 1,
                "digest": digest,
                "git_sha": _git_sha(),
                "timestamp_utc": _now_iso(),
                "env": _env_versions(),
            },
        )
        logger.info("Smoke OK; digest at %s", out)
        for k, v in digest.items():
            logger.info("  %s: %s", k, v)
        if args.phase == "smoke":
            return 0

    # Bootstrap env (HF_TOKEN, HF_HOME).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Resolve the expected cond-id set for the no-drop assertion in the
    # partition merge. `--transformations` overrides; otherwise we expect
    # the canonical 16-cond CONDITIONS set. For end_of_system, only the
    # Class A subset is expected (B/C/D have no system message). (Codex
    # round-1 blocker #4.)
    _ALL_CONDS, _CONDS_BY_ID = _load_conditions_registry()

    if args.transformations:
        _active_for_expected = [_CONDS_BY_ID[c] for c in args.transformations]
    else:
        _active_for_expected = list(_ALL_CONDS)
    _expected_full = [c.cid for c in _active_for_expected]
    _expected_eos = [c.cid for c in _active_for_expected if c.cls == "A"]

    def _expected_for_point(pt: str) -> list[str] | None:
        if pt == "end_of_system":
            return _expected_eos or None
        return _expected_full or None

    # MERGE-ONLY shortcut: just stack partitioned activation files into the
    # canonical <point>__layer<L>.pt shape and exit. Runs after all GPU procs
    # have written their per-cond partitioned files.
    if args.merge_only:
        for pt in args.extraction_points:
            merge_partitioned_activations(
                (pt,),
                tuple(args.layers),
                overwrite=args.overwrite,
                expected_cond_ids=_expected_for_point(pt),
            )
        # ROUND-4 holistic gate: re-validate the canonical set at READ
        # time so any path that bypassed the merger (e.g. a re-used cache
        # whose mtime predates a deleted per-cond file) still trips
        # before metrics/regression would run.
        validate_canonical_completeness(
            tuple(args.extraction_points),
            tuple(args.layers),
            expected_cond_ids_for_point={
                pt: _expected_for_point(pt) for pt in args.extraction_points
            },
        )
        # Also build the next_token_js matrix file from the sidecars,
        # WITH the #406 cross-check guarded fail-loud. (Round-1 blocker #1.)
        if not args.no_next_token_js:
            write_next_token_js_matrix(enforce_cross_check=True)
        return 0

    # EXTRACTION phase
    if args.phase in ("all", "extract", "extraction"):
        if args.dry_run:
            logger.info("--dry-run: SKIPPING extraction (no model load).")
        elif batched_mode:
            logger.info(
                "Batched extraction: points=%s layers=%s transformations=%s "
                "n_probes=%d batch_size=%d partitioned=%s "
                "capture_next_token_logits=%s",
                args.extraction_points,
                args.layers,
                args.transformations or "ALL 16",
                args.n_probes,
                args.batch_size,
                args.partitioned,
                not args.no_next_token_js,
            )
            run_extraction_batched(
                extraction_points=tuple(args.extraction_points),
                layers=tuple(args.layers),
                transformations=tuple(args.transformations) if args.transformations else None,
                n_probes=args.n_probes,
                max_response_tokens=args.max_response_tokens,
                device=args.device,
                overwrite=args.overwrite,
                pool_path=args.probe_pool,
                pool_mode=args.probe_pool_mode,
                batch_size=args.batch_size,
                capture_next_token_logits=(not args.no_next_token_js),
                write_partitioned=args.partitioned,
            )
            # ROUND-3 FIX #2: write the next_token_js matrix + run the #406
            # cross-check on ALL post-extraction paths (partitioned auto-merge,
            # non-partitioned auto-merge, both at phase=all). Before round 3
            # the JS write was gated `if args.partitioned and args.phase ==
            # "all"`, so the supported single-proc `--batched --probe-pool
            # --phase all` WITHOUT `--partitioned` path proceeded to
            # metrics/regression with NO next_token_js matrix — the JS
            # baseline (the whole point of #502) vanished silently on that
            # path. Now the gate is `args.phase == "all" and not args.no_next_token_js`
            # — the JS write + #406 cross-check fire on every supported
            # post-extraction path. Multi-proc runs still call --merge-only
            # after fan-in (covered by the merge-only branch above).
            #
            # The partitioned-single-proc branch additionally merges
            # per-cond → canonical (the non-partitioned auto-merge already
            # happened inside run_extraction_batched).
            if args.partitioned and args.phase == "all":
                logger.info("Single-process partitioned run: merging in-place")
                for pt in args.extraction_points:
                    merge_partitioned_activations(
                        (pt,),
                        tuple(args.layers),
                        overwrite=args.overwrite,
                        expected_cond_ids=_expected_for_point(pt),
                    )
            if args.phase == "all" and not args.no_next_token_js:
                write_next_token_js_matrix(enforce_cross_check=True)
        else:
            logger.info(
                "Extraction (serial #493 path): points=%s layers=%s transformations=%s n_probes=%d",
                args.extraction_points,
                args.layers,
                args.transformations or "ALL 16",
                args.n_probes,
            )
            run_extraction(
                extraction_points=tuple(args.extraction_points),
                layers=tuple(args.layers),
                transformations=tuple(args.transformations) if args.transformations else None,
                n_probes=args.n_probes,
                max_response_tokens=args.max_response_tokens,
                device=args.device,
                overwrite=args.overwrite,
            )

    if args.phase in ("extract", "extraction"):
        return 0

    # ROUND-4 holistic gate. Before reloading for metrics/regress/figures,
    # validate every requested (pt, L) canonical exists AND its cond_ids set
    # equals exactly the expected set. Catches every path that bypasses the
    # merge-time check — serial #493 writes canonical directly, --skip-extract
    # over a stale tree, --phase metrics on a partial prior run, etc. (Round-4
    # blocker #1: single invariant "no metrics phase runs unless every
    # requested (pt, L) canonical contains exactly the expected conds".)
    # When `--transformations` is set explicitly we trust the user's intent
    # (smoke / debug subset), so the expected set is the user-named subset.
    activations_by_point = validate_canonical_completeness(
        tuple(args.extraction_points),
        tuple(args.layers),
        expected_cond_ids_for_point={pt: _expected_for_point(pt) for pt in args.extraction_points},
    )
    if not activations_by_point:
        logger.warning("No activations on disk; run --phase extract first (with GPU).")
        return 1

    # Correctness cross-check: last-token cosine must match #406's existing.
    # The check is ENFORCED (raises) on full real-data runs (16 conds × 50
    # probes) where the recipes are supposed to be byte-identical. On
    # subset runs (--transformations / --n-probes < 50) it's logged-only:
    # the cond/probe subset changes the cosine values, so a mismatch is
    # expected, not a bug.
    if "last_prompt" in activations_by_point and not args.dry_run:
        existing = _load_existing_cosine_matrices(tuple(args.layers))
        # Explicit non-empty layer-map guard (round-2 issue D): avoid
        # StopIteration / KeyError when `last_prompt` is loaded but
        # carries no usable layer payloads (e.g. --phase metrics on a
        # checkpoint set that only finished end_of_system, or a layer
        # subset that doesn't overlap with any extracted checkpoint).
        last_prompt_map = activations_by_point["last_prompt"]
        usable_layer_payloads = [
            (L, p) for L, p in last_prompt_map.items() if isinstance(p, dict) and "activations" in p
        ]
        if existing and usable_layer_payloads:
            _sample_L, sample_payload = usable_layer_payloads[0]
            n_cond_loaded = sample_payload["activations"].shape[0]
            n_q_loaded = sample_payload["activations"].shape[1]
            # The full-grid path: same 16 conds AND exactly 50 probes
            # (== bit-identical recipe to #406) → strict on the full set.
            # Round-1 review flagged that at n_q=500 (the #502 default)
            # this dropped to logged-only, turning the safety net OFF.
            # Round-2 fix: when n_q > 50 AND the first 50 probes ARE the
            # q_test prefix (which is the #502 pool's invariant), slice
            # the prefix and run the cross-check STRICT on that slice —
            # the strict guard is recovered without re-running 500 probes
            # through the off-policy recipe. (Codex/Claude round-1
            # blocker #5.)
            n_cond_match = args.transformations is None and n_cond_loaded == 16
            strict_full = n_cond_match and n_q_loaded == 50
            strict_prefix = (
                n_cond_match
                and n_q_loaded > 50
                and args.probe_pool is not None  # 500-pool ⇒ q_test is the prefix
            )
            cross_check_map = last_prompt_map
            cross_check_strict = strict_full
            cross_check_slice_note = "full"
            if strict_prefix:
                # Build a fresh map with the first-50-probe slice per layer
                # so the strict comparison sees byte-identical-recipe inputs.
                cross_check_map = {}
                for L, p in last_prompt_map.items():
                    if not (isinstance(p, dict) and "activations" in p):
                        continue
                    arr = p["activations"]  # (n_cond, n_q, H)
                    if arr.shape[1] < 50:
                        continue
                    cross_check_map[L] = {
                        **p,
                        "activations": arr[:, :50, :],
                        "n_probes": 50,
                    }
                cross_check_strict = True
                cross_check_slice_note = "q_test_prefix_50"
                logger.info(
                    "cosine cross-check: 500-probe pool detected; running STRICT "
                    "on the q_test prefix (probes[0:50]) — recovers the #406 "
                    "safety net that n_q==50 used to gate."
                )
            try:
                check = reproduce_last_token_cosine_check(
                    cross_check_map,
                    existing,
                    cond_ids=sample_payload["cond_ids"],
                    strict=cross_check_strict,
                )
            except AssertionError as e:
                # Persist the failure context before re-raising so the
                # operator can diagnose without re-running the extraction.
                # `per_layer_tolerances` records the EXACT map applied so
                # an audit can see which layer used which cap — the L27
                # relaxation to 1e-2 is documented inline above the
                # `COSINE_REPRO_TOLERANCES` declaration.
                _write_json_atomic(
                    BAKEOFF_DIR / "cosine_cross_check.json",
                    {
                        "schema_version": 2,
                        "per_layer_tolerances": {
                            str(k): float(v) for k, v in COSINE_REPRO_TOLERANCES.items()
                        },
                        "strict": True,
                        "slice": cross_check_slice_note,
                        "failed": True,
                        "failure_reason": str(e),
                        "n_cond_loaded": int(n_cond_loaded),
                        "n_probes_loaded": int(n_q_loaded),
                        "git_sha": _git_sha(),
                        "timestamp_utc": _now_iso(),
                    },
                )
                raise
            _write_json_atomic(
                BAKEOFF_DIR / "cosine_cross_check.json",
                {
                    "schema_version": 2,
                    "per_layer_tolerances": {
                        str(k): float(v) for k, v in COSINE_REPRO_TOLERANCES.items()
                    },
                    "strict": cross_check_strict,
                    "slice": cross_check_slice_note,
                    "n_cond_loaded": int(n_cond_loaded),
                    "n_probes_loaded": int(n_q_loaded),
                    "per_layer": check,
                    "git_sha": _git_sha(),
                    "timestamp_utc": _now_iso(),
                },
            )
        elif existing and not usable_layer_payloads:
            logger.warning(
                "Skipping cosine cross-check: last_prompt extraction point has no "
                "usable layer payloads (loaded layers: %s)",
                list(last_prompt_map.keys()),
            )

    # METRICS phase
    if args.phase in ("all", "metrics"):
        run_metrics(
            activations_by_point=activations_by_point,
            metrics=tuple(args.metrics),
            pca_k=args.pca_k,
            overwrite=args.overwrite,
        )

    if args.phase == "metrics":
        return 0

    # REGRESSION phase — cond_ids comes from a NON-end_of_system checkpoint
    # so the regression covers the full 16-cond grid (end_of_system metric
    # files carry the Class-A subpanel and are handled by run_regression on
    # their own subpanel).
    cond_ids: list[str] | None = None
    for pt in args.extraction_points:
        layer_map = activations_by_point.get(pt) or {}
        if pt == "end_of_system" or not layer_map:
            continue
        any_L = next(iter(layer_map))
        cond_ids = layer_map[any_L]["cond_ids"]
        break
    if cond_ids is None:
        # Fall back: only end_of_system checkpoints present (e.g. an early-
        # phase smoke that just exercised Class A). Use its subpanel as the
        # cond set so regression at least runs; clearly flagged downstream.
        for pt, layer_map in activations_by_point.items():
            if not layer_map:
                continue
            any_L = next(iter(layer_map))
            cond_ids = layer_map[any_L]["cond_ids"]
            logger.warning(
                "Regression cond_ids fell back to %s subpanel (%s) — "
                "no full-grid extraction point on disk.",
                pt,
                cond_ids,
            )
            break
    if cond_ids is None:
        raise RuntimeError(
            "No usable activations on disk for any extraction point; "
            "re-run --phase extract before --phase regress."
        )

    all_cells = run_regression(
        cond_ids=cond_ids,
        arms=tuple(args.arms),
        epochs=tuple(args.epochs),
        overwrite=args.overwrite,
    )

    # Winner + summary
    headline = all_cells.get("loc_ep1")
    winner = select_winner(headline) if headline else None
    grid_path = BAKEOFF_DIR / "bakeoff_grid.json"
    _write_json_atomic(
        grid_path,
        {
            "schema_version": 1,
            "cells": all_cells,
            "winner_loc_ep1": winner,
            "git_sha": _git_sha(),
            "timestamp_utc": _now_iso(),
        },
    )
    logger.info("Wrote %s", grid_path)
    if winner:
        sub = winner.get("sub_predictor")
        logger.info(
            # Surface the actual base-prior-safe guard variable
            # (rho_NONSTYLIZED_glogp) — round-3 fix A changed select_winner
            # to gate on this, the log line was still printing the
            # round-2 full-panel variant.
            "WINNER (loc_ep1): %s · L%d · %s%s — CV R² = %.3f, "
            "rho_ns(ΔG) = %+.3f, rho_ns(g_logp) = %+.3f",
            winner["extraction_point"],
            winner["layer"],
            winner["metric"],
            f" · {sub}" if sub else "",
            winner["cv_full_deltag"],
            winner["rho_nonstylized_deltag"],
            winner["rho_nonstylized_glogp"],
        )
    else:
        logger.warning("No predictor survived the non-stylized + base-prior-safe check.")

    if args.phase == "regress":
        return 0

    # FIGURES phase
    if args.phase in ("all", "figures"):
        emit_figures(all_cells, tuple(args.extraction_points), tuple(args.layers))

    return 0


if __name__ == "__main__":
    sys.exit(main())
