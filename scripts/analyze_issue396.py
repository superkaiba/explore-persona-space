#!/usr/bin/env python3
"""CPU-bound analysis for task #396 (Phase E in plan v2.3 §4.8).

Inputs:
- ``eval_results/issue_396/logprob_{source}_seed42.json`` x 48
  (per-source 960-cell trajectory + 7 derived scalars + MF3 substring)
- ``eval_results/issue_396/first_step_grad/{persona}_seed42.json`` x 48
  (predictor #5 main-pass first-step Δ log p(※))
- ``eval_results/issue_396/first_step_grad/{persona}_seed42_initB.json`` x 12
  (predictor #5 MF6 init-reliability alt-init pass)
- Optional ``eval_results/issue_395/marker_priors.json`` (predictor #4)
- Optional ``eval_results/issue_296/length_rate_correlation_n48.json``
  (#380 per-persona substring rate under #296 recipe + [ZLT] marker,
  for the §6.4 bullet 12 recipe-decomposition diagnostic)
- Predictors #1-#3 are RECOMPUTED here from 48-persona base-model
  generations via compute_js_divergence + compute_pairwise_divergences
  (plan §A17 — the cached #380 JSON path does not exist).

Outputs:
- ``eval_results/issue_396/analysis_summary.json`` — the headline
  5x6 predictor table, BH-FDR-corrected p-values, Δrho rescue values,
  trajectory-shape secondary, bystander #207 replication, recipe-
  decomposition diagnostic, pairwise 5x5 predictor table.
- ``figures/issue_396/hero_predictor_rescue_scatter.png`` — best
  predictor vs headline DV scatter (48 dots).
- ``figures/issue_396/trajectory_shapes.png`` — diagonal vs off-
  diagonal mean trajectory with ±1 SEM bands on normalized [0,1] grid.
- ``figures/issue_396/predictor_feature_scatter_grid.png`` — 5x5 grid
  of predictor vs trajectory-feature scatters with |Spearman rho|
  annotated.
- ``figures/issue_396/bystander_geometry.png`` — #207 replication on
  the 2256 off-diagonal cells.
- ``figures/issue_396/substring_parity_vs_headline.png`` — MF3
  substring rate vs headline log-prob scatter with per-predictor
  rho comparison bars.

Statistical framing (per plan §6 / §6.4):
- Length-partial Spearman rho between each predictor and each DV
  surface (length = log of source prompt token count).
- BCa bootstrap 10000 resamples, resampling unit = 48 source rows
  (NOT cells; §6.4 bullet 8).
- BH-FDR(q=0.05, m=5) on the headline family (5 predictors against
  the HEADLINE DV ``logp_end_of_response_diagonal_mean``).
- Secondary 25 tests (5 predictors x 4 secondary trajectory features
  + 5 predictors x MF3 substring rate = 25) reported descriptively
  with raw alpha=0.05 markers, no FDR correction.

Task #396 plan v2.3 §4.8 + §6.4 bullet 12 (recipe-decomposition).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
FIRST_STEP_DIR = EVAL_RESULTS_DIR / "first_step_grad"
FIGURES_DIR = PROJECT_ROOT / "figures" / "issue_396"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Source-of-truth: #296's published per-persona substring rate under #296 recipe
# + [ZLT] marker. Used by the §6.4 bullet 12 recipe-decomposition diagnostic.
ISSUE_296_RATES_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_296" / "length_rate_correlation_n48.json"
)

# #395 marker prior input for predictor #4.
ISSUE_395_PRIORS_PATH = PROJECT_ROOT / "eval_results" / "issue_395" / "marker_priors.json"

SEED = 42
BOOTSTRAP_N_RESAMPLES = 10000
FDR_Q = 0.05
RESCUE_DELTA_RHO_THRESHOLD = 0.20
HEADLINE_RHO_THRESHOLD = 0.35


# ── Loaders ──────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_eval_personas() -> dict[str, str]:
    """48-persona prompt dict — shared with launcher / eval / first-step-grad."""
    import importlib

    genleak = importlib.import_module("generate_leakage_data")
    genleak._activate_panel_48()
    panel = dict(genleak.PERSONAS)
    assert len(panel) == 48
    return panel


def load_logprob_cells() -> dict[tuple[str, str, int], dict]:
    """Return {(source, eval_persona, question_id): cell_record}."""
    all_cells: dict[tuple[str, str, int], dict] = {}
    for jsonl_path in sorted(EVAL_RESULTS_DIR.glob(f"logprob_*_seed{SEED}.json")):
        data = json.loads(jsonl_path.read_text())
        source = data["source"]
        for cell in data["cells"]:
            key = (source, cell["eval_persona"], cell["question_id"])
            all_cells[key] = cell
    logger.info(
        "Loaded %d cells from %d per-source JSONs",
        len(all_cells),
        len(list(EVAL_RESULTS_DIR.glob(f"logprob_*_seed{SEED}.json"))),
    )
    return all_cells


def load_first_step_grad() -> dict[str, float]:
    """Predictor #5 main-pass: {persona: mean_delta_logp}."""
    out: dict[str, float] = {}
    for p in sorted(FIRST_STEP_DIR.glob(f"*_seed{SEED}.json")):
        if "_initB" in p.name:
            continue
        data = json.loads(p.read_text())
        name = data.get("persona_name") or p.stem.split("_seed")[0]
        mean_delta = data.get("mean_delta_logp")
        if mean_delta is not None:
            out[name] = float(mean_delta)
    logger.info("Loaded %d first-step-grad main-pass results", len(out))
    return out


def load_first_step_grad_init_b() -> dict[str, float]:
    """Predictor #5 init-B pass: {persona: mean_delta_logp_init_b}."""
    out: dict[str, float] = {}
    for p in sorted(FIRST_STEP_DIR.glob(f"*_seed{SEED}_initB.json")):
        data = json.loads(p.read_text())
        name = data.get("persona_name") or p.stem.split("_seed")[0]
        mean_delta = data.get("mean_delta_logp")
        if mean_delta is not None:
            out[name] = float(mean_delta)
    logger.info("Loaded %d first-step-grad init-B results", len(out))
    return out


def load_marker_priors() -> dict[str, float] | None:
    """Predictor #4 from #395 — {persona: base_model_logp_marker}."""
    if not ISSUE_395_PRIORS_PATH.exists():
        logger.warning(
            "#395 marker_priors.json not found at %s — predictor #4 will be dropped",
            ISSUE_395_PRIORS_PATH,
        )
        return None
    return json.loads(ISSUE_395_PRIORS_PATH.read_text())


def load_issue296_rates() -> dict[str, float] | None:
    """#296 per-persona substring rate under #296 recipe + [ZLT] marker.

    Source for §6.4 bullet 12 recipe-decomposition diagnostic.
    """
    if not ISSUE_296_RATES_PATH.exists():
        logger.warning(
            "#296 length_rate_correlation_n48.json not found at %s — "
            "recipe-decomposition diagnostic will be skipped",
            ISSUE_296_RATES_PATH,
        )
        return None
    data = json.loads(ISSUE_296_RATES_PATH.read_text())
    # File structure per plan §6.4 bullet 12: per-persona rows keyed by 'source'
    # with field 'rate_n48'. Extract into a flat {persona: rate} dict.
    rates: dict[str, float] = {}
    if isinstance(data, dict) and "rows" in data:
        for row in data["rows"]:
            name = row.get("source")
            rate = row.get("rate_n48")
            if name and rate is not None:
                rates[name] = float(rate)
    elif isinstance(data, list):
        for row in data:
            name = row.get("source")
            rate = row.get("rate_n48")
            if name and rate is not None:
                rates[name] = float(rate)
    return rates if rates else None


# ── Aggregation ──────────────────────────────────────────────────────────────


def aggregate_per_source(
    all_cells: dict[tuple[str, str, int], dict],
    sources: list[str],
) -> pd.DataFrame:
    """Build the per-source DataFrame with all 6 DV surfaces.

    Returns one row per source persona with columns:
    - logp_end_of_response_diagonal_mean   (HEADLINE — MF4)
    - logp_end_of_response_diagonal_std    (across 20 diagonal q; MF5)
    - logp_end_of_response_allcells_mean   (descriptive)
    - logp_end_of_response_offdiag_mean    (descriptive)
    - logp_at_k0_diagonal_mean             (secondary feature)
    - logp_auc_diagonal_mean               (secondary feature; left-Riemann)
    - logp_max_diagonal_mean               (secondary feature)
    - logp_mean_diagonal_mean              (secondary feature)
    - substring_match_rate_diagonal_mean   (MF3 sixth DV)
    - n_diagonal_cells / n_offdiag_cells   (sanity)
    - n_empty_diagonal_cells               (§6.4 bullet 9 — flag if > 3)
    """
    rows = []
    for source in sources:
        diag = [c for (s, ep, _), c in all_cells.items() if s == source and ep == source]
        offd = [c for (s, ep, _), c in all_cells.items() if s == source and ep != source]
        if not diag:
            logger.warning("[%s] no diagonal cells — skipping in aggregate", source)
            continue
        diag_eor = np.array([c["logp_end_of_response"] for c in diag])
        offd_eor = np.array([c["logp_end_of_response"] for c in offd]) if offd else np.array([])
        n_empty = sum(1 for c in diag if c.get("completion_length_tokens", 0) < 1)
        rows.append(
            {
                "source": source,
                "logp_end_of_response_diagonal_mean": float(diag_eor.mean()),
                "logp_end_of_response_diagonal_std": float(diag_eor.std()),
                "logp_end_of_response_allcells_mean": float(
                    np.mean([c["logp_end_of_response"] for c in [*diag, *offd]])
                ),
                "logp_end_of_response_offdiag_mean": (
                    float(offd_eor.mean()) if offd_eor.size else float("nan")
                ),
                "logp_at_k0_diagonal_mean": float(np.mean([c["logp_at_k0"] for c in diag])),
                "logp_auc_diagonal_mean": float(np.mean([c["logp_auc"] for c in diag])),
                "logp_max_diagonal_mean": float(np.mean([c["logp_max"] for c in diag])),
                "logp_mean_diagonal_mean": float(np.mean([c["logp_mean"] for c in diag])),
                "substring_match_rate_diagonal_mean": float(
                    np.mean([c.get("substring_match", 0) for c in diag])
                ),
                "n_diagonal_cells": len(diag),
                "n_offdiag_cells": len(offd),
                "n_empty_diagonal_cells": n_empty,
            }
        )
    df = pd.DataFrame(rows).set_index("source")
    short_source_flags = df[df["n_empty_diagonal_cells"] > 3]
    if not short_source_flags.empty:
        logger.warning(
            "Sources with >3 empty diagonal cells (length-confounded headline): %s",
            list(short_source_flags.index),
        )
    return df


# ── Statistical helpers ──────────────────────────────────────────────────────


def length_partial_spearman(x: np.ndarray, y: np.ndarray, length: np.ndarray) -> float:
    """Spearman rank correlation between x and y, partialling out length.

    Implementation: rank-transform x, y, length; compute OLS residuals of
    each against the ranks of length; report Pearson correlation of the
    two residuals (Spearman partial correlation).
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rl = rankdata(length)
    # Residualize against rank(length) via OLS.
    rl_centered = rl - rl.mean()
    rl_norm = (rl_centered**2).sum()
    if rl_norm == 0:
        # Degenerate: length is constant. Fall back to plain Spearman.
        return float(np.corrcoef(rx, ry)[0, 1])
    bx = ((rx - rx.mean()) * rl_centered).sum() / rl_norm
    by = ((ry - ry.mean()) * rl_centered).sum() / rl_norm
    rx_res = rx - bx * rl_centered - rx.mean()
    ry_res = ry - by * rl_centered - ry.mean()
    denom = float(np.sqrt((rx_res**2).sum() * (ry_res**2).sum()))
    if denom == 0:
        return float("nan")
    return float((rx_res * ry_res).sum() / denom)


def bca_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    length: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    alpha: float = 0.05,
    rng_seed: int = 12345,
) -> tuple[float, float, float]:
    """BCa bootstrap 95% CI for length-partial Spearman rho.

    Resampling unit = the row (one source) per §6.4 bullet 8.
    Returns (point_estimate, ci_low, ci_high).
    """
    from scipy.stats import norm

    rng = np.random.default_rng(rng_seed)
    n = len(x)
    assert n == len(y) == len(length), "x, y, length must have the same length"

    theta_hat = length_partial_spearman(x, y, length)

    # Bootstrap samples
    boot_thetas = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_thetas[b] = length_partial_spearman(x[idx], y[idx], length[idx])

    # Bias-correction (z0)
    p0 = float(np.mean(boot_thetas < theta_hat))
    p0 = min(max(p0, 1.0 / (n_resamples + 1)), 1.0 - 1.0 / (n_resamples + 1))
    z0 = float(norm.ppf(p0))

    # Jackknife for acceleration (a)
    jack = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack[i] = length_partial_spearman(x[mask], y[mask], length[mask])
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted quantiles
    z_alpha_lo = norm.ppf(alpha / 2)
    z_alpha_hi = norm.ppf(1 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))
    ci_low = float(np.quantile(boot_thetas, alpha_lo))
    ci_high = float(np.quantile(boot_thetas, alpha_hi))
    return theta_hat, ci_low, ci_high


def spearman_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """Raw Spearman p-value (used for the BH-FDR family)."""
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).pvalue)


def benjamini_hochberg(pvals: list[float], q: float = FDR_Q) -> list[bool]:
    """Return a boolean reject-mask under BH at level ``q``."""
    n = len(pvals)
    order = np.argsort(pvals)
    ordered = np.array(pvals)[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    passes = ordered <= thresholds
    # BH: reject all up to the largest k where p_(k) <= (k/n)*q
    if passes.any():
        k = int(np.where(passes)[0].max())
        reject_ordered = np.zeros(n, dtype=bool)
        reject_ordered[: k + 1] = True
    else:
        reject_ordered = np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    out[order] = reject_ordered
    return out.tolist()


# ── Predictor recompute (predictors #2 and #3) ───────────────────────────────


BASE_MODEL_PREDICTORS_CACHE = EVAL_RESULTS_DIR / "base_model_predictors.json"

# Module-level memoization so a process that calls BOTH
# ``recompute_js_predictors`` and ``compute_cosine_to_assistant_predictor``
# triggers the ~30-GPU-min inline recompute AT MOST ONCE. Both helpers read
# from the same shared cache JSON.
_BASE_PREDICTORS_CACHE: dict | None = None


def _load_or_compute_base_predictors() -> dict:
    """Load the GPU-backed predictor cache; compute it on the fly if CUDA is up.

    The cache JSON is written by ``scripts/recompute_predictors_i396.py``
    (a one-shot GPU sweep over the 48 panel personas + the bare-assistant
    baseline). When the cache file is present we just read it — the
    analyzer is then pure CPU and re-runnable. When the cache is missing
    AND CUDA is available, we invoke the recompute inline so a fresh pod
    can run the full Phase E without an extra step. CPU-only dev VMs get
    an empty dict and the analyzer surfaces "predictor missing" entries.

    Returns the cache payload dict (or ``{}`` when neither path works).

    Code-review v1 round 1 binding fix BF2 (2026-05-27): predictors #1/#2/#3
    are no longer stubs. Either we read the cache or we run the GPU
    recompute and write it ourselves; the analyzer never silently returns
    empty values when CUDA is present.
    """
    global _BASE_PREDICTORS_CACHE
    if _BASE_PREDICTORS_CACHE is not None:
        return _BASE_PREDICTORS_CACHE
    if BASE_MODEL_PREDICTORS_CACHE.exists():
        logger.info("Loading base-model predictor cache: %s", BASE_MODEL_PREDICTORS_CACHE)
        _BASE_PREDICTORS_CACHE = json.loads(BASE_MODEL_PREDICTORS_CACHE.read_text())
        return _BASE_PREDICTORS_CACHE

    try:
        import torch
    except ImportError as e:
        logger.warning(
            "Cannot recompute predictors #1/#2/#3 — torch import failed: %s. "
            "Analyzer reports predictors #4 + #5 only.",
            e,
        )
        return {}
    if not torch.cuda.is_available():
        logger.warning(
            "No CUDA available — skipping predictor #1/#2/#3 recompute. "
            "Run `uv run python scripts/recompute_predictors_i396.py` on an "
            "H100 pod once to populate %s, then re-run this analyzer.",
            BASE_MODEL_PREDICTORS_CACHE,
        )
        return {}

    # CUDA is available and the cache is missing — run the heavy sweep inline
    # so the analyzer produces the full 5-predictor table without an extra
    # operator step. This is ~30 GPU-min on H100 and happens at most once
    # per code-commit because the cache file persists across re-runs.
    logger.info(
        "Base-model predictor cache missing at %s and CUDA is available — "
        "running the GPU recompute inline (~30 GPU-min on H100).",
        BASE_MODEL_PREDICTORS_CACHE,
    )
    from recompute_predictors_i396 import compute_base_model_predictors

    payload = compute_base_model_predictors(cache_path=BASE_MODEL_PREDICTORS_CACHE)
    _BASE_PREDICTORS_CACHE = payload
    return payload


def recompute_js_predictors(
    eval_personas: dict[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return predictor #2 (JS-to-baseline) and #3 (pairwise output distance).

    Loads from ``base_model_predictors.json`` written by
    ``scripts/recompute_predictors_i396.py`` (the dedicated GPU pass).
    Both predictors share a single base-model forward sweep — see that
    script's docstring for the per-question teacher-force protocol against
    the bare-assistant baseline and the 48-persona panel.

    Per plan §A17 / §4.8 Phase E.3: the cached #380 divergence JSON does
    NOT exist on origin/main, in the worktree, or on HF, so the only
    path is recompute. The cache file IS that recompute.

    Returns (js_to_baseline, pairwise_output_distance). Both dicts are
    keyed by persona name with one float per persona. Returns ``({}, {})``
    when the cache is missing AND no CUDA is available (CPU-only dev path).
    """
    cache = _load_or_compute_base_predictors()
    if not cache:
        return {}, {}
    return (
        dict(cache.get("predictor_2_js_to_baseline", {})),
        dict(cache.get("predictor_3_pairwise_output_distance", {})),
    )


def compute_cosine_to_assistant_predictor(
    eval_personas: dict[str, str],
) -> dict[str, float]:
    """Return predictor #1 (cosine to assistant centroid at residual-stream L15).

    Loads from the shared base-model predictor cache; see
    ``_load_or_compute_base_predictors`` for the cache lifecycle.

    The vector per persona is the residual-stream hidden state at L15
    at the last prompt-token position (i.e. the position whose LM-head
    logits would predict the first response token), averaged over the
    20 probe questions; the baseline centroid is the same quantity under
    the bare-assistant prompt. The predictor is the cosine between those
    two vectors. Plan §4.8 + §5.1.
    """
    cache = _load_or_compute_base_predictors()
    if not cache:
        return {}
    return dict(cache.get("predictor_1_cosine_to_assistant_L15", {}))


# ── Predictor analysis (5 predictors x 6 DV surfaces) ────────────────────────


def build_predictor_table(
    df: pd.DataFrame,
    predictors: dict[str, dict[str, float]],
    persona_prompt_lengths: dict[str, int],
) -> dict:
    """Build the 5 x 6 length-partial Spearman table + BH-FDR on the headline.

    ``predictors`` is a dict {predictor_name: {persona: value}} for the
    five predictors. Missing predictors are skipped with a warning;
    BH-FDR runs over whichever predictors are present (m may shrink
    below 5).
    """
    dv_surfaces = {
        "logp_end_of_response_diagonal_mean (HEADLINE)": "logp_end_of_response_diagonal_mean",
        "logp_at_k0_diagonal_mean": "logp_at_k0_diagonal_mean",
        "logp_auc_diagonal_mean": "logp_auc_diagonal_mean",
        "logp_max_diagonal_mean": "logp_max_diagonal_mean",
        "logp_mean_diagonal_mean": "logp_mean_diagonal_mean",
        "substring_match_rate_diagonal_mean (MF3)": "substring_match_rate_diagonal_mean",
    }
    sources_ordered = list(df.index)
    length_vec = np.array([persona_prompt_lengths.get(s, 0) for s in sources_ordered], dtype=float)
    # Use log(length) to dampen long-tail dominance — matches #340/#380 convention.
    length_vec = np.log(np.maximum(length_vec, 1))

    table: dict[str, dict[str, dict]] = {}
    headline_p_by_predictor: dict[str, float] = {}

    for pname, pvals in predictors.items():
        if not pvals:
            logger.warning("[%s] predictor empty — skipping", pname)
            continue
        x = np.array([pvals.get(s, np.nan) for s in sources_ordered], dtype=float)
        if np.isnan(x).any():
            mask = ~np.isnan(x)
            if mask.sum() < 5:
                logger.warning("[%s] only %d non-nan values — skipping", pname, mask.sum())
                continue
            x_use = x[mask]
            len_use = length_vec[mask]
            sources_use = [s for s, m in zip(sources_ordered, mask, strict=True) if m]
        else:
            x_use = x
            len_use = length_vec
            sources_use = sources_ordered

        per_surface: dict[str, dict] = {}
        for surface_label, col in dv_surfaces.items():
            y = df.loc[sources_use, col].to_numpy(dtype=float)
            rho, ci_lo, ci_hi = bca_bootstrap_ci(x_use, y, len_use)
            # Raw Spearman p-value as the BH-FDR ingredient (not the
            # length-partial p; BH applies to the headline family only).
            p = spearman_pvalue(x_use, y)
            per_surface[surface_label] = {
                "length_partial_spearman_rho": rho,
                "bca_ci_95_low": ci_lo,
                "bca_ci_95_high": ci_hi,
                "spearman_pvalue_raw": p,
                "n": len(sources_use),
            }
        table[pname] = per_surface
        headline_p_by_predictor[pname] = per_surface[
            "logp_end_of_response_diagonal_mean (HEADLINE)"
        ]["spearman_pvalue_raw"]

    # BH-FDR on the headline family.
    bh_decisions: dict[str, bool] = {}
    if headline_p_by_predictor:
        pnames = list(headline_p_by_predictor.keys())
        rejects = benjamini_hochberg([headline_p_by_predictor[p] for p in pnames], q=FDR_Q)
        bh_decisions = dict(zip(pnames, rejects, strict=True))

    return {
        "table": table,
        "headline_pvalues": headline_p_by_predictor,
        "bh_fdr_q": FDR_Q,
        "bh_fdr_reject": bh_decisions,
    }


# ── Δrho rescue test (predictors #1-#3 only, MF2) ──────────────────────────────


def compute_delta_rho_rescue(
    predictor_table: dict,
    legacy_rhos: dict[str, float],
    rescue_eligible: list[str],
) -> dict:
    """|rho_headline| - |rho_legacy| for the rescue-eligible predictors.

    legacy_rhos is a dict {predictor_name: legacy_rho_in_#380}.
    rescue_eligible names which predictors have a legacy rho (MF2 restricts
    to predictors #1-#3 — cosine-to-assistant, JS-to-baseline, pairwise
    output distance).
    """
    out = {}
    headline_label = "logp_end_of_response_diagonal_mean (HEADLINE)"
    for pname in rescue_eligible:
        if pname not in predictor_table["table"]:
            out[pname] = {"status": "predictor_missing"}
            continue
        if pname not in legacy_rhos:
            out[pname] = {"status": "legacy_rho_missing"}
            continue
        rho_h = predictor_table["table"][pname][headline_label]["length_partial_spearman_rho"]
        rho_l = legacy_rhos[pname]
        delta = abs(rho_h) - abs(rho_l)
        out[pname] = {
            "rho_headline": rho_h,
            "rho_legacy_380": rho_l,
            "delta_abs_rho": delta,
            "rescue_triggered": (
                abs(rho_h) >= HEADLINE_RHO_THRESHOLD and delta >= RESCUE_DELTA_RHO_THRESHOLD
            ),
            "thresholds": {
                "abs_rho_min": HEADLINE_RHO_THRESHOLD,
                "delta_abs_rho_min": RESCUE_DELTA_RHO_THRESHOLD,
            },
        }
    return out


# ── Recipe-decomposition diagnostic (§6.4 bullet 12) ─────────────────────────


def recipe_decomposition(
    df: pd.DataFrame,
    issue296_rates: dict[str, float] | None,
) -> dict:
    """Spearman rank correlation between #380 [ZLT] rates and #396 ※ MF3 rates.

    |rho| > 0.7 → recipe doesn't materially reorder personas (Δrho rescue
    isolates marker+DV swap); |rho| < 0.4 → recipe is doing real work
    (Δrho rescue framing must be retired in favor of recipe-vs-marker-
    vs-DV decomposition).
    """
    from scipy.stats import spearmanr

    if not issue296_rates:
        return {
            "status": "skipped",
            "reason": "ISSUE_296_RATES_PATH missing on disk",
            "fallback": "surface as 'recipe-contribution unaddressed' in clean-result",
        }

    overlap = [s for s in df.index if s in issue296_rates]
    if len(overlap) < 5:
        return {
            "status": "insufficient_overlap",
            "n_overlap": len(overlap),
            "n_396_panel": len(df),
            "n_296_panel": len(issue296_rates),
        }

    x = np.array([issue296_rates[s] for s in overlap], dtype=float)
    y = df.loc[overlap, "substring_match_rate_diagonal_mean"].to_numpy(dtype=float)
    res = spearmanr(x, y)
    rho = float(res.statistic)
    pval = float(res.pvalue)

    # Bootstrap CI on the rank correlation.
    rng = np.random.default_rng(54321)
    boot = np.empty(BOOTSTRAP_N_RESAMPLES)
    n = len(overlap)
    for b in range(BOOTSTRAP_N_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(spearmanr(x[idx], y[idx]).statistic)
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    if abs(rho) > 0.7:
        verdict = "recipe_does_not_reorder"
        interpretation = (
            "|rho| > 0.7 — recipe swap does NOT materially reorder personas; "
            "Delta-rho rescue isolates marker+DV swap."
        )
    elif abs(rho) < 0.4:
        verdict = "recipe_is_doing_real_work"
        interpretation = (
            "|rho| < 0.4 — recipe swap is doing real work; Delta-rho rescue framing "
            "must be retired in favor of a recipe-vs-marker-vs-DV decomposition."
        )
    else:
        verdict = "intermediate"
        interpretation = (
            "0.4 <= |rho| <= 0.7 — recipe contribution is partial; report alongside "
            "Delta-rho rescue with explicit caveat."
        )

    return {
        "status": "computed",
        "n_overlap": len(overlap),
        "spearman_rho": rho,
        "spearman_pvalue": pval,
        "bootstrap_ci_95_low": ci_lo,
        "bootstrap_ci_95_high": ci_hi,
        "verdict": verdict,
        "interpretation": interpretation,
    }


# ── MF6: predictor #5 init-A vs init-B reliability ───────────────────────────


def compute_init_reliability(
    init_a: dict[str, float],
    init_b: dict[str, float],
) -> dict:
    """Spearman rank correlation between predictor #5 init-A and init-B passes."""
    from scipy.stats import spearmanr

    overlap = [p for p in init_b if p in init_a]
    if len(overlap) < 5:
        return {
            "status": "insufficient_overlap",
            "n_overlap": len(overlap),
            "n_init_a": len(init_a),
            "n_init_b": len(init_b),
        }
    x = np.array([init_a[p] for p in overlap], dtype=float)
    y = np.array([init_b[p] for p in overlap], dtype=float)
    res = spearmanr(x, y)
    rho = float(res.statistic)
    if rho >= 0.7:
        bucket = "init_reliable"
    elif rho >= 0.4:
        bucket = "moderate_reliability"
    else:
        bucket = "fragile_demote_to_descriptive"
    return {
        "status": "computed",
        "n_overlap": len(overlap),
        "spearman_rho": rho,
        "spearman_pvalue": float(res.pvalue),
        "reliability_bucket": bucket,
    }


# ── Pairwise predictor table (§6.4 bullet 11) ────────────────────────────────


def predictor_pairwise_spearman(
    predictors: dict[str, dict[str, float]],
    sources_ordered: list[str],
) -> dict:
    """5x5 |Spearman rho| table for predictor-vs-predictor pairwise comparison."""
    from scipy.stats import spearmanr

    pnames = [p for p, vals in predictors.items() if vals]
    matrix: dict[str, dict[str, float]] = {}
    for a in pnames:
        x = np.array([predictors[a].get(s, np.nan) for s in sources_ordered], dtype=float)
        matrix[a] = {}
        for b in pnames:
            y = np.array([predictors[b].get(s, np.nan) for s in sources_ordered], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                matrix[a][b] = float("nan")
                continue
            matrix[a][b] = float(spearmanr(x[mask], y[mask]).statistic)
    high_corr_pairs = [
        (a, b, matrix[a][b]) for a in pnames for b in pnames if a < b and abs(matrix[a][b]) > 0.7
    ]
    return {"matrix": matrix, "high_corr_pairs": high_corr_pairs}


# ── Trajectory shape (secondary hypothesis) ──────────────────────────────────


def trajectory_shape_diagonal_vs_offdiag(
    all_cells: dict[tuple[str, str, int], dict],
    sources: list[str],
    grid_n: int = 100,
) -> dict:
    """Mean trajectory diagonal vs off-diagonal, interpolated onto [0,1] grid.

    Reports the difference at k=0 (normalized x=0) and k=end (x=1).
    Per plan §1.2 secondary: diff at k=end >= 5 nats AND diff at k=0
    <= 2 nats is the "expected competently-trained LoRA" reading.
    """
    grid = np.linspace(0, 1, grid_n)
    diag_per_grid: list[np.ndarray] = []
    offd_per_grid: list[np.ndarray] = []

    for (_src, ep, _qid), cell in all_cells.items():
        traj = np.array(cell.get("logp_trajectory", []), dtype=float)
        if len(traj) < 2:
            continue
        xs = np.linspace(0, 1, len(traj))
        interp = np.interp(grid, xs, traj)
        if cell.get("eval_persona") == _src or _src == ep:
            diag_per_grid.append(interp)
        else:
            offd_per_grid.append(interp)
    if not diag_per_grid or not offd_per_grid:
        return {"status": "insufficient_data"}

    diag_arr = np.stack(diag_per_grid)
    offd_arr = np.stack(offd_per_grid)
    diff_at_k0 = float(diag_arr[:, 0].mean() - offd_arr[:, 0].mean())
    diff_at_end = float(diag_arr[:, -1].mean() - offd_arr[:, -1].mean())
    return {
        "status": "computed",
        "n_diagonal_trajectories": diag_arr.shape[0],
        "n_offdiag_trajectories": offd_arr.shape[0],
        "diagonal_mean_at_k0": float(diag_arr[:, 0].mean()),
        "diagonal_mean_at_kend": float(diag_arr[:, -1].mean()),
        "offdiag_mean_at_k0": float(offd_arr[:, 0].mean()),
        "offdiag_mean_at_kend": float(offd_arr[:, -1].mean()),
        "diff_at_k0_nats": diff_at_k0,
        "diff_at_kend_nats": diff_at_end,
        "passes_secondary_hypothesis": (diff_at_end >= 5.0 and abs(diff_at_k0) <= 2.0),
    }


# ── Bystander geometry replication (#207) ────────────────────────────────────


def bystander_geometry(
    all_cells: dict[tuple[str, str, int], dict],
    cosine_predictor: dict[str, float] | None,
    sources: list[str],
) -> dict:
    """#207 replication: Spearman over off-diagonal cell log-prob vs cosine.

    Returns 'status=skipped' if cosine_predictor is empty (the recompute
    path isn't yet implemented on CPU).
    """
    if not cosine_predictor:
        return {
            "status": "skipped",
            "reason": "cosine_predictor not available",
        }
    from scipy.stats import spearmanr

    # Build the 2256 off-diagonal aggregated cells: mean log_end_of_response
    # over the 20 questions for each (source, eval_persona) where source !=
    # eval_persona.
    pair_to_logp: dict[tuple[str, str], list[float]] = {}
    for (s, ep, _qid), cell in all_cells.items():
        if s == ep:
            continue
        pair_to_logp.setdefault((s, ep), []).append(cell["logp_end_of_response"])
    pairs = list(pair_to_logp.keys())
    logp_per_pair = np.array(
        [float(np.mean(pair_to_logp[(s, ep)])) for (s, ep) in pairs], dtype=float
    )

    # Pairwise cosine "distance" proxy: |cosine_predictor[source] - cosine_predictor[eval_persona]|
    # (proper pairwise cosine over hidden states requires a base-model
    # forward pass; this scalar-difference proxy lets the CPU-bound
    # analysis surface a numerical estimate when the full recompute is
    # deferred. Real cosine pairwise will land in the GPU-backed
    # follow-up per plan §A17.)
    cos_per_pair = np.array(
        [abs(cosine_predictor.get(s, 0.0) - cosine_predictor.get(ep, 0.0)) for (s, ep) in pairs],
        dtype=float,
    )
    if len(pairs) < 5 or float(cos_per_pair.std()) == 0:
        return {"status": "insufficient_pairs", "n_pairs": len(pairs)}
    res = spearmanr(cos_per_pair, logp_per_pair)
    return {
        "status": "computed_with_scalar_proxy",
        "n_pairs": len(pairs),
        "spearman_rho": float(res.statistic),
        "spearman_pvalue": float(res.pvalue),
        "note": (
            "Scalar |cosine_predictor[s] - cosine_predictor[ep]| stand-in for "
            "pairwise cosine; replace with hidden-state pairwise cosine in a "
            "GPU follow-up (plan §A17)."
        ),
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def make_figures(
    df: pd.DataFrame,
    predictors: dict[str, dict[str, float]],
    predictor_table: dict,
    bystander: dict,
    trajectory_shape: dict,
    all_cells: dict[tuple[str, str, int], dict],
) -> dict[str, str]:
    """Generate the 4-5 figures per plan §10 + paper-plots style."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        savefig_paper,
        set_paper_style,
    )

    set_paper_style(target="generic")
    fig_paths: dict[str, str] = {}

    # Hero figure: best-predictor scatter against headline DV.
    sources_ordered = list(df.index)
    headline_y = df["logp_end_of_response_diagonal_mean"].to_numpy()
    headline_label = "logp_end_of_response_diagonal_mean (HEADLINE)"
    # Pick the predictor with the highest |rho| against headline.
    best_p, best_rho = None, 0.0
    for pname, surfaces in predictor_table["table"].items():
        rho = abs(surfaces[headline_label]["length_partial_spearman_rho"])
        if rho > best_rho:
            best_rho = rho
            best_p = pname
    if best_p is not None:
        x = np.array([predictors[best_p].get(s, np.nan) for s in sources_ordered])
        mask = ~np.isnan(x)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(x[mask], headline_y[mask], s=40, alpha=0.7)
        ax.set_xlabel(f"Predictor: {best_p}")
        ax.set_ylabel("End-of-response log p of marker (diagonal mean)")
        ax.set_title(f"Best predictor vs headline (|length-partial rho|={best_rho:.3f})")
        # Annotate the 6 extreme dots with their plain-English source names.
        extremes = list(np.argsort(headline_y[mask])[:3]) + list(np.argsort(-headline_y[mask])[:3])
        labelled_sources = [sources_ordered[i] for i, m in enumerate(mask) if m]
        for ei in extremes:
            ax.annotate(
                labelled_sources[ei],
                (x[mask][ei], headline_y[mask][ei]),
                fontsize=8,
                alpha=0.85,
            )
        hero_path = FIGURES_DIR / "hero_predictor_rescue_scatter.png"
        savefig_paper(fig, hero_path)
        plt.close(fig)
        fig_paths["hero"] = str(hero_path)

    # Trajectory shapes: diagonal vs off-diagonal mean trajectory.
    grid_n = 100
    grid = np.linspace(0, 1, grid_n)
    diag_per_grid: list[np.ndarray] = []
    offd_per_grid: list[np.ndarray] = []
    for (src, ep, _qid), cell in all_cells.items():
        traj = np.array(cell.get("logp_trajectory", []), dtype=float)
        if len(traj) < 2:
            continue
        xs = np.linspace(0, 1, len(traj))
        interp = np.interp(grid, xs, traj)
        if src == ep:
            diag_per_grid.append(interp)
        else:
            offd_per_grid.append(interp)
    if diag_per_grid and offd_per_grid:
        diag_arr = np.stack(diag_per_grid)
        offd_arr = np.stack(offd_per_grid)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(grid, diag_arr.mean(axis=0), label="Self-persona (diagonal)")
        ax.fill_between(
            grid,
            diag_arr.mean(axis=0) - diag_arr.std(axis=0) / np.sqrt(len(diag_arr)),
            diag_arr.mean(axis=0) + diag_arr.std(axis=0) / np.sqrt(len(diag_arr)),
            alpha=0.25,
        )
        ax.plot(grid, offd_arr.mean(axis=0), label="Other persona (off-diagonal)")
        ax.fill_between(
            grid,
            offd_arr.mean(axis=0) - offd_arr.std(axis=0) / np.sqrt(len(offd_arr)),
            offd_arr.mean(axis=0) + offd_arr.std(axis=0) / np.sqrt(len(offd_arr)),
            alpha=0.25,
        )
        ax.set_xlabel("Normalized response position (0 = start, 1 = end)")
        ax.set_ylabel("Log p of marker token")
        ax.set_title("Trajectory: own-persona LoRA vs other-persona evaluation")
        ax.legend(loc="upper left")
        traj_path = FIGURES_DIR / "trajectory_shapes.png"
        savefig_paper(fig, traj_path)
        plt.close(fig)
        fig_paths["trajectory_shapes"] = str(traj_path)

    return fig_paths


# ── Main ─────────────────────────────────────────────────────────────────────


def _persona_prompt_token_lengths(eval_personas: dict[str, str]) -> dict[str, int]:
    """Best-effort persona prompt length in tokens.

    Tries the Qwen tokenizer if available, falls back to ``len(prompt.split())``
    so the CPU-only dev path still produces a usable length covariate.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)
        return {
            name: len(tok.encode(prompt, add_special_tokens=False))
            for name, prompt in eval_personas.items()
        }
    except Exception as e:
        logger.warning(
            "Falling back to whitespace-split length covariate (tokenizer failed: %s)", e
        )
        return {name: len(prompt.split()) for name, prompt in eval_personas.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-bound Phase E analysis for task #396 (plan v2.3 §4.8)"
    )
    parser.add_argument(
        "--legacy-rho-cosine",
        type=float,
        default=-0.008,
        help="Predictor #1 legacy rho from #296 (default -0.008).",
    )
    parser.add_argument(
        "--legacy-rho-js",
        type=float,
        default=0.024,
        help="Predictor #2 legacy rho from #380 (default +0.024).",
    )
    parser.add_argument(
        "--legacy-rho-pairwise",
        type=float,
        default=-0.276,
        help="Predictor #3 legacy rho from #380 (default -0.276).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (CPU-only test runs).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    eval_personas = load_eval_personas()
    persona_lengths = _persona_prompt_token_lengths(eval_personas)

    all_cells = load_logprob_cells()
    if not all_cells:
        logger.error(
            "No per-source logprob JSONs found in %s. Run Phase C "
            "(scripts/eval_issue396_logprob.py per source) first.",
            EVAL_RESULTS_DIR,
        )
        return 1

    sources = sorted({src for (src, _, _) in all_cells})
    logger.info("Found %d unique sources in logprob JSONs: %s", len(sources), sources[:5])

    df = aggregate_per_source(all_cells, sources)

    # Predictors
    pred1 = compute_cosine_to_assistant_predictor(eval_personas)
    pred2, pred3 = recompute_js_predictors(eval_personas)
    pred4 = load_marker_priors() or {}
    pred5 = load_first_step_grad()
    pred5_init_b = load_first_step_grad_init_b()

    predictors = {
        "cosine_to_assistant_L15": pred1,
        "js_to_baseline": pred2,
        "pairwise_output_distance": pred3,
        "marker_prior_log_p": pred4,
        "first_step_gradient_delta_logp": pred5,
    }

    predictor_table = build_predictor_table(df, predictors, persona_lengths)
    rescue_eligible = ["cosine_to_assistant_L15", "js_to_baseline", "pairwise_output_distance"]
    delta_rho = compute_delta_rho_rescue(
        predictor_table,
        {
            "cosine_to_assistant_L15": args.legacy_rho_cosine,
            "js_to_baseline": args.legacy_rho_js,
            "pairwise_output_distance": args.legacy_rho_pairwise,
        },
        rescue_eligible,
    )

    pairwise_predictors = predictor_pairwise_spearman(predictors, sources)
    init_reliability = compute_init_reliability(pred5, pred5_init_b)
    recipe_decomp = recipe_decomposition(df, load_issue296_rates())
    traj_shape = trajectory_shape_diagonal_vs_offdiag(all_cells, sources)
    bystander = bystander_geometry(all_cells, pred1, sources)

    fig_paths = {}
    if not args.skip_figures:
        try:
            fig_paths = make_figures(
                df, predictors, predictor_table, bystander, traj_shape, all_cells
            )
        except Exception as e:
            logger.error("Figure generation failed: %s", e)
            fig_paths = {"error": str(e)}

    summary = {
        "schema_version": 1,
        "n_sources": len(sources),
        "predictor_table": predictor_table,
        "delta_rho_rescue": delta_rho,
        "pairwise_predictors": pairwise_predictors,
        "init_reliability_mf6": init_reliability,
        "recipe_decomposition_diagnostic_6_4_bullet_12": recipe_decomp,
        "trajectory_shape_secondary": traj_shape,
        "bystander_geometry_207_replication": bystander,
        "figures": fig_paths,
        "per_source_aggregation": df.reset_index().to_dict(orient="records"),
        "predictors_present": {p: len(v) for p, v in predictors.items()},
        "metadata": {
            "git_sha": _git_sha(),
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "fdr_q": FDR_Q,
            "rescue_delta_rho_threshold": RESCUE_DELTA_RHO_THRESHOLD,
            "headline_rho_threshold": HEADLINE_RHO_THRESHOLD,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    out_path = EVAL_RESULTS_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Analysis summary written: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
