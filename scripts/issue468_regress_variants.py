#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥) in scientific docstrings + logs.
"""Issue #468 Phase C — head-to-head regression for the new variants.

Reads:

* New Phase B output at ``eval_results/issue468/predictor_cossim_variants{,_training}/
  <cell>_<flavor>.json`` (V1, V2, V3 per k, V4, V5 position sweep,
  ``lexical_token_embedding_bag_cos``).
* Existing #463 cossim at ``eval_results/issue463/predictor_cossim{,_training}/
  <cell>_<flavor>.json`` for the paired-difference bootstrap baselines
  (``last_prompt_token``, ``response_mean``).
* Outcome L and log-token covariate via ``issue463_regress.load_outcome_per_cell``
  / ``load_token_counts``.

Reports per (variant, layer, probe-source, flavor):

* ``spearman_raw`` and ``spearman_partial_log_tokens`` (matches #463
  ``partial_spearman`` exactly).
* ``spearman_partial_L0_post_block_cos`` — partial vs the per-cell L0
  cosine of the SAME extraction (re-labeled "early-layer /
  persona-string-content covariate"; NOT a clean lexical control).
* ``spearman_partial_lexical_token_embedding_bag_cos`` — partial vs the
  pre-block token-embedding-bag cosine (clean lexical control).
* ``spearman_paired_diff_vs_463_last_prompt_token`` — 10K bootstrap of
  cell INDEX, recompute BOTH ρ on each resample, report mean + 95% CI of
  the DIFFERENCE. (Anti-pattern guarded: NOT two separate per-bar CIs.)
* ``spearman_paired_diff_vs_463_response_mean`` — same pattern for
  response-mean-recovery (V3 vs #463 response_mean).
* ``spearman_shuffle_null_percentile`` — ≥1000 cell-label shuffles on
  the V1 (M, L) vectors; report observed ρ as percentile of the shuffle
  distribution.

Plus a separate ``position_sweep_regression_block`` over the 6 V5
positions at L25 lit-training (pre-registered diagnostic for plan §6.2).

Output: ``eval_results/issue468/regression_variants_{betley,training}_{NL,lit}.json``
and ``eval_results/issue468/regression_position_sweep_L25_lit_training.json``.

Usage::

    uv run python scripts/issue468_regress_variants.py
    uv run python scripts/issue468_regress_variants.py --bootstrap-n 1000  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue463_regress import (  # noqa: E402
    CELLS_18,
    load_outcome_per_cell,
    load_token_counts,
    partial_spearman,
    spearman_with_n,
)
from scipy import stats  # noqa: E402

logger = logging.getLogger("issue468_regress_variants")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR_463 = PROJECT_ROOT / "eval_results" / "issue463"
EVAL_DIR_468 = PROJECT_ROOT / "eval_results" / "issue468"
COSSIM_463_BY_SOURCE = {
    "betley": EVAL_DIR_463 / "predictor_cossim",
    "training": EVAL_DIR_463 / "predictor_cossim_training",
}
COSSIM_468_BY_SOURCE = {
    "betley": EVAL_DIR_468 / "predictor_cossim_variants",
    "training": EVAL_DIR_468 / "predictor_cossim_variants_training",
}
# Exploratory V3 k-sweep lives in a SEPARATE directory so it never clobbers
# the main per-cell files (which only contain k=0 and primary k=8). The
# launcher's `--out-base eval_results/issue468/k_sweep_lit_training_L25/`
# step writes `<cell>_lit.json` files here with V3 cosines for k∈{0,4,8,16}
# at layer 25 only.
K_SWEEP_DIR = EVAL_DIR_468 / "k_sweep_lit_training_L25"
K_SWEEP_LAYER = 25
K_SWEEP_FLAVOR = "lit"
K_SWEEP_PROBE_SOURCE = "training"

DEFAULT_LAYERS = [18, 20, 21, 22, 24, 25, 27]
DEFAULT_BOOTSTRAP_N = 10000
DEFAULT_PERMUTATION_N = 1000
POSITION_NAMES = ("p0", "p1", "p2", "p3", "p4", "p5")


def load_per_cell(directory: Path, flavor: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not directory.exists():
        logger.warning("Directory missing: %s", directory)
        return out
    for cell in CELLS_18:
        path = directory / f"{cell}_{flavor}.json"
        if not path.exists():
            continue
        with open(path) as f:
            out[cell] = json.load(f)
    return out


def extract_simple_layer_map(
    per_cell: dict[str, dict], extraction_key: str, layer: int
) -> dict[str, float]:
    """For an extraction dict shaped ``cos_by_extraction[key][layer]``."""
    out: dict[str, float] = {}
    for cell, d in per_cell.items():
        ce = d.get("cos_by_extraction", {}).get(extraction_key, {})
        val = ce.get(str(layer))
        if val is None:
            continue
        out[cell] = float(val)
    return out


def extract_v3_layer_map(per_cell: dict[str, dict], k: int, layer: int) -> dict[str, float]:
    """For V3 dict shaped ``cos_by_extraction.response_mean_skip_k[str(k)][str(layer)]``."""
    out: dict[str, float] = {}
    for cell, d in per_cell.items():
        v3 = d.get("cos_by_extraction", {}).get("response_mean_skip_k", {})
        per_k = v3.get(str(k), {})
        val = per_k.get(str(layer))
        if val is None:
            continue
        out[cell] = float(val)
    return out


def extract_position_sweep_layer_map(
    per_cell: dict[str, dict], position: str, layer: int
) -> dict[str, float]:
    """For V5 dict shaped ``cos_by_extraction.position_sweep[name][str(layer)]``."""
    out: dict[str, float] = {}
    for cell, d in per_cell.items():
        sw = d.get("cos_by_extraction", {}).get("position_sweep", {})
        per_pos = sw.get(position, {})
        val = per_pos.get(str(layer))
        if val is None:
            continue
        out[cell] = float(val)
    return out


def extract_lexical_bag(per_cell: dict[str, dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cell, d in per_cell.items():
        val = d.get("lexical_token_embedding_bag_cos")
        if val is None:
            continue
        out[cell] = float(val)
    return out


# ── Stats helpers ──────────────────────────────────────────────────────────


def paired_diff_bootstrap_rho(
    M_a: list[float],
    M_b: list[float],
    L: list[float],
    n_bootstrap: int,
    seed: int = 0,
) -> dict:
    """Resample cell INDEX, recompute BOTH ρ on each resample, return mean +
    95% CI of the DIFFERENCE rho(M_a, L) − rho(M_b, L).

    Per planner §6.2 + A23: the difference distribution is the honest
    headline uncertainty object — NOT two separate per-bar CIs whose
    overlap is misleading for paired comparisons.
    """
    n = len(L)
    if n < 4 or not (len(M_a) == len(M_b) == n):
        return {"n": n, "note": "insufficient_or_unaligned"}
    rng = np.random.default_rng(seed)
    a = np.asarray(M_a, dtype=float)
    b = np.asarray(M_b, dtype=float)
    y = np.asarray(L, dtype=float)
    diffs = np.empty(n_bootstrap, dtype=float)
    valid = 0
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Skip degenerate resamples (ranks collapse).
        ya, yb, yy = a[idx], b[idx], y[idx]
        if len(set(ya.tolist())) < 2 or len(set(yb.tolist())) < 2 or len(set(yy.tolist())) < 2:
            diffs[i] = np.nan
            continue
        rho_a = stats.spearmanr(ya, yy).statistic
        rho_b = stats.spearmanr(yb, yy).statistic
        diffs[i] = float(rho_a - rho_b)
        valid += 1
    finite = diffs[np.isfinite(diffs)]
    if finite.size < 100:
        return {"n": n, "note": "too_few_valid_bootstrap_resamples", "valid": int(finite.size)}
    rho_a_full = stats.spearmanr(a, y).statistic
    rho_b_full = stats.spearmanr(b, y).statistic
    return {
        "n_cells": n,
        "n_bootstrap": n_bootstrap,
        "n_valid": int(finite.size),
        "rho_a_observed": float(rho_a_full),
        "rho_b_observed": float(rho_b_full),
        "diff_observed": float(rho_a_full - rho_b_full),
        "diff_mean_bootstrap": float(finite.mean()),
        "diff_ci_95_low": float(np.percentile(finite, 2.5)),
        "diff_ci_95_high": float(np.percentile(finite, 97.5)),
        "diff_ci_95_excludes_zero": bool(
            np.percentile(finite, 2.5) > 0 or np.percentile(finite, 97.5) < 0
        ),
    }


def cell_label_permutation_null(
    M: list[float],
    L: list[float],
    n_perm: int,
    seed: int = 0,
) -> dict:
    """Shuffle cell labels: for each iter shuffle L, compute ρ(M, L_perm).
    Report observed ρ as percentile of the null distribution.
    """
    n = len(L)
    if n < 4 or len(M) != n:
        return {"n": n, "note": "insufficient_or_unaligned"}
    rng = np.random.default_rng(seed)
    M_arr = np.asarray(M, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    if len(set(M_arr.tolist())) < 2 or len(set(L_arr.tolist())) < 2:
        return {"n": n, "note": "insufficient_variance"}
    observed = float(stats.spearmanr(M_arr, L_arr).statistic)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        L_perm = rng.permutation(L_arr)
        null[i] = float(stats.spearmanr(M_arr, L_perm).statistic)
    null_finite = null[np.isfinite(null)]
    pct = float((null_finite < observed).mean()) * 100.0
    return {
        "n_cells": n,
        "n_permutation": n_perm,
        "observed_rho": observed,
        "null_mean": float(null_finite.mean()),
        "null_std": float(null_finite.std(ddof=1)),
        "null_ci_95_low": float(np.percentile(null_finite, 2.5)),
        "null_ci_95_high": float(np.percentile(null_finite, 97.5)),
        "observed_percentile": pct,
        "p_two_sided": float(min(pct, 100.0 - pct) / 50.0),
    }


# ── Block builder ──────────────────────────────────────────────────────────


def build_block(
    label: str,
    M_per_cell: dict[str, float],
    outcome: dict[str, dict],
    tokens: dict[str, float],
    L0_per_cell: dict[str, float] | None,
    lexical_per_cell: dict[str, float] | None,
    baseline_M_per_cell: dict[str, float] | None,
    baseline_label: str,
    bootstrap_n: int,
    permutation_n: int,
    run_baseline_diff: bool,
    run_permutation: bool,
) -> dict:
    """Compute every per-predictor stat for one extraction (variant, layer).

    Layout mirrors `issue463_regress.regress_one` for `spearman_raw` and
    `spearman_partial_log_tokens`; adds the v2 paired-diff bootstrap +
    permutation null + lexical-bag partial.
    """
    if baseline_M_per_cell is None:
        baseline_M_per_cell = {}
    common = sorted(set(M_per_cell) & set(outcome) & set(tokens))
    if not common:
        return {"predictor": label, "n_cells": 0, "note": "no_common_cells"}
    M_vals = [M_per_cell[c] for c in common]
    L_vals = [outcome[c]["mean_L"] for c in common]
    log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
    raw = spearman_with_n(M_vals, L_vals)
    partial_logtok = partial_spearman(M_vals, L_vals, log_tokens)

    block: dict = {
        "predictor": label,
        "n_cells": len(common),
        "cells": common,
        "M_per_cell": {c: M_per_cell[c] for c in common},
        "L_per_cell": {c: outcome[c]["mean_L"] for c in common},
        "spearman_raw": raw,
        "spearman_partial_log_tokens": partial_logtok,
    }

    if L0_per_cell:
        L0_common = sorted(set(common) & set(L0_per_cell))
        if len(L0_common) >= 4:
            M_l0 = [M_per_cell[c] for c in L0_common]
            L_l0 = [outcome[c]["mean_L"] for c in L0_common]
            Z_l0 = [L0_per_cell[c] for c in L0_common]
            block["spearman_partial_L0_post_block_cos"] = partial_spearman(M_l0, L_l0, Z_l0)
            block["L0_partial_n_cells"] = len(L0_common)

    if lexical_per_cell:
        lex_common = sorted(set(common) & set(lexical_per_cell))
        if len(lex_common) >= 4:
            M_lx = [M_per_cell[c] for c in lex_common]
            L_lx = [outcome[c]["mean_L"] for c in lex_common]
            Z_lx = [lexical_per_cell[c] for c in lex_common]
            block["spearman_partial_lexical_token_embedding_bag_cos"] = partial_spearman(
                M_lx, L_lx, Z_lx
            )
            block["lexical_bag_partial_n_cells"] = len(lex_common)

    if run_baseline_diff and baseline_M_per_cell:
        pair_common = sorted(set(common) & set(baseline_M_per_cell))
        if len(pair_common) >= 4:
            M_a = [M_per_cell[c] for c in pair_common]
            M_b = [baseline_M_per_cell[c] for c in pair_common]
            L_p = [outcome[c]["mean_L"] for c in pair_common]
            key = f"spearman_paired_diff_vs_463_{baseline_label}"
            block[key] = paired_diff_bootstrap_rho(M_a, M_b, L_p, n_bootstrap=bootstrap_n)
            block[key]["baseline_label"] = baseline_label
            block[key]["n_paired_cells"] = len(pair_common)

    if run_permutation:
        block["spearman_shuffle_null_percentile"] = cell_label_permutation_null(
            M_vals, L_vals, n_perm=permutation_n
        )

    return block


VARIANT_KEYS: list[tuple[str, str]] = [
    # (label_prefix, cos_by_extraction key) — V1/V2/V4 + recompute_* (G2 cross-check).
    ("V1_last_prompt_token_final_content", "last_prompt_token_final_content"),
    ("V2_last_response_token", "last_response_token"),
    ("V4_response_max", "response_max"),
    ("recompute_last_prompt_token", "last_prompt_token"),
    ("recompute_response_mean", "response_mean"),
]


def _build_blocks_at_layer(
    *,
    blocks: dict[str, dict],
    layer: int,
    new_cells: dict[str, dict],
    old_cells: dict[str, dict],
    outcome: dict[str, dict],
    tokens: dict[str, float],
    L0_per_cell: dict[str, float],
    lexical_per_cell: dict[str, float] | None,
    k_values_present: list[int],
    bootstrap_n: int,
    permutation_n: int,
) -> None:
    """Insert regression blocks for ONE layer into ``blocks``: V1/V2/V4 +
    recompute_*, V3 per k (paired vs #463 response_mean), V5 per position
    (paired vs #463 last_prompt_token = T-1).

    Pulled out of ``regress_one_pair`` for complexity (C901).
    """
    baseline_last = extract_simple_layer_map(old_cells, "last_prompt_token", layer)
    baseline_resp = extract_simple_layer_map(old_cells, "response_mean", layer)

    # V1 / V2 / V4 + recompute_* — paired vs #463 last_prompt_token.
    for label_prefix, key in VARIANT_KEYS:
        M_per_cell = extract_simple_layer_map(new_cells, key, layer)
        if not M_per_cell:
            continue
        label = f"{label_prefix}_L{layer}"
        blocks[label] = build_block(
            label=label,
            M_per_cell=M_per_cell,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical_per_cell,
            baseline_M_per_cell=baseline_last,
            baseline_label="last_prompt_token",
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
            run_baseline_diff=True,
            run_permutation=(label_prefix == "V1_last_prompt_token_final_content"),
        )

    # V3 per k — paired vs #463 response_mean.
    for k in k_values_present:
        M_v3 = extract_v3_layer_map(new_cells, k, layer)
        if not M_v3:
            continue
        label = f"V3_response_mean_skip_k{k}_L{layer}"
        blocks[label] = build_block(
            label=label,
            M_per_cell=M_v3,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical_per_cell,
            baseline_M_per_cell=baseline_resp,
            baseline_label="response_mean",
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
            run_baseline_diff=True,
            run_permutation=False,
        )

    # V5 per position — paired vs #463 last_prompt_token (= T-1 = p5).
    for pos in POSITION_NAMES:
        M_v5 = extract_position_sweep_layer_map(new_cells, pos, layer)
        if not M_v5:
            continue
        label = f"V5_position_sweep_{pos}_L{layer}"
        blocks[label] = build_block(
            label=label,
            M_per_cell=M_v5,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical_per_cell,
            baseline_M_per_cell=baseline_last,
            baseline_label="last_prompt_token",
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
            run_baseline_diff=True,
            run_permutation=False,
        )


# ── Main per-(probe, flavor) ──────────────────────────────────────────────


def regress_one_pair(
    probe_source: str,
    flavor: str,
    layers: list[int],
    bootstrap_n: int,
    permutation_n: int,
) -> dict:
    """Build the per-(probe-source, flavor) regression JSON."""
    outcome = load_outcome_per_cell()
    tokens = load_token_counts()

    new_cells = load_per_cell(COSSIM_468_BY_SOURCE[probe_source], flavor)
    old_cells = load_per_cell(COSSIM_463_BY_SOURCE[probe_source], flavor)
    logger.info(
        "Loaded %d new (#468) cells, %d old (#463) cells, %d outcome cells, %d token-count cells",
        len(new_cells),
        len(old_cells),
        len(outcome),
        len(tokens),
    )

    # Per-cell L0 cosine (use #468 V1 L0 if present, else fall back to #463 L0 of
    # last_prompt_token).
    L0_per_cell: dict[str, float] = {}
    for cell, d in new_cells.items():
        l0_dict = d.get("L0_post_block_cos_by_layer", {}) or {}
        v = l0_dict.get("0")
        if v is not None:
            L0_per_cell[cell] = float(v)
    if not L0_per_cell:
        # Fallback: read #463's last_prompt_token L0.
        for cell, d in old_cells.items():
            ce = d.get("cos_by_extraction", {}).get("last_prompt_token", {})
            v = ce.get("0")
            if v is not None:
                L0_per_cell[cell] = float(v)
        logger.info("Used #463 last_prompt_token L0 as the L0 covariate (no #468 L0 found)")

    lexical_per_cell = extract_lexical_bag(new_cells)
    if not lexical_per_cell:
        logger.info("No lexical_token_embedding_bag_cos found; skipping lexical partial")

    blocks: dict[str, dict] = {}
    # Discover V3 k values present in the new cells once (any cell that has them).
    v3_dict = None
    for d in new_cells.values():
        v3_dict = d.get("cos_by_extraction", {}).get("response_mean_skip_k", {})
        if v3_dict:
            break
    k_values_present = sorted({int(k) for k in (v3_dict or {})}) if v3_dict else []

    for layer in layers:
        _build_blocks_at_layer(
            blocks=blocks,
            layer=layer,
            new_cells=new_cells,
            old_cells=old_cells,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical_per_cell,
            k_values_present=k_values_present,
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
        )

    summary = {
        "probe_source": probe_source,
        "flavor": flavor,
        "layers": layers,
        "n_new_cells": len(new_cells),
        "n_old_cells": len(old_cells),
        "bootstrap_n": bootstrap_n,
        "permutation_n": permutation_n,
        "blocks": blocks,
        "metadata": reproducibility_metadata(
            {"script": "issue468_regress_variants", "probe_source": probe_source}
        ),
    }
    return summary


def build_position_sweep_block(
    probe_source: str,
    flavor: str,
    layer: int,
    bootstrap_n: int,
    permutation_n: int,
) -> dict:
    """Pre-registered diagnostic: per V5 position at one (probe-source, flavor,
    layer), report ρ + paired-diff vs #463 p5 (= ``last_prompt_token`` at the
    same layer). See plan §6.2 (ii) / (iii).
    """
    outcome = load_outcome_per_cell()
    tokens = load_token_counts()
    new_cells = load_per_cell(COSSIM_468_BY_SOURCE[probe_source], flavor)
    old_cells = load_per_cell(COSSIM_463_BY_SOURCE[probe_source], flavor)
    L0_per_cell: dict[str, float] = {}
    for cell, d in new_cells.items():
        l0_dict = d.get("L0_post_block_cos_by_layer", {}) or {}
        v = l0_dict.get("0")
        if v is not None:
            L0_per_cell[cell] = float(v)
    lexical = extract_lexical_bag(new_cells)

    baseline_p5 = extract_simple_layer_map(old_cells, "last_prompt_token", layer)

    sweep_blocks: dict[str, dict] = {}
    for pos in POSITION_NAMES:
        M = extract_position_sweep_layer_map(new_cells, pos, layer)
        if not M:
            continue
        label = f"V5_p_{pos}_L{layer}"
        sweep_blocks[label] = build_block(
            label=label,
            M_per_cell=M,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical,
            baseline_M_per_cell=baseline_p5,
            baseline_label="last_prompt_token_T_minus_1",
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
            run_baseline_diff=True,
            run_permutation=(pos == "p0"),  # V1-shaped null
        )
    return {
        "probe_source": probe_source,
        "flavor": flavor,
        "layer": layer,
        "bootstrap_n": bootstrap_n,
        "permutation_n": permutation_n,
        "position_sweep_blocks": sweep_blocks,
        "metadata": reproducibility_metadata(
            {"script": "issue468_regress_variants:position_sweep"}
        ),
    }


def build_k_sweep_block(
    bootstrap_n: int,
    permutation_n: int,
) -> dict:
    """Pre-registered EXPLORATORY V3 k-sweep block (plan §4.2.4, §6.3, §11).

    Ingests `eval_results/issue468/k_sweep_lit_training_L25/<cell>_lit.json`
    (written by the launcher's exploratory `--out-base
    .../k_sweep_lit_training_L25/ --skip-k-sweep 0 4 8 16` step) and emits
    one regression block per k value in {0, 4, 8, 16} at L25, training-lit.
    Reports ALL four k values — no best-k cherry-picking.

    The k-sweep dir is INTENTIONALLY separate from the main per-cell dir so
    the exploratory run never clobbers the main sweep's `<cell>_lit.json`
    files (which only carry the primary k=8 alongside the k=0 recompute).
    Baseline pairing is #463 response_mean @ L25 lit-training (= the
    canonical persona-vectors recipe headline at that cell).
    """
    outcome = load_outcome_per_cell()
    tokens = load_token_counts()
    k_cells = load_per_cell(K_SWEEP_DIR, K_SWEEP_FLAVOR)
    old_cells = load_per_cell(COSSIM_463_BY_SOURCE[K_SWEEP_PROBE_SOURCE], K_SWEEP_FLAVOR)
    if not k_cells:
        logger.warning(
            "k-sweep dir %s has no cells; skipping k-sweep regression block",
            K_SWEEP_DIR,
        )
        return {
            "probe_source": K_SWEEP_PROBE_SOURCE,
            "flavor": K_SWEEP_FLAVOR,
            "layer": K_SWEEP_LAYER,
            "k_sweep_blocks": {},
            "note": "no_cells_found",
            "metadata": reproducibility_metadata({"script": "issue468_regress_variants:k_sweep"}),
        }

    # Discover the k values actually present in the k-sweep output (any cell).
    k_values_present: list[int] = []
    for d in k_cells.values():
        v3_dict = d.get("cos_by_extraction", {}).get("response_mean_skip_k", {})
        if v3_dict:
            k_values_present = sorted({int(k) for k in v3_dict})
            break

    # Per-cell L0 + lexical-bag covariates (read from k-sweep cells; the
    # exploratory run does not re-compute the L0 partial baseline so we
    # accept whatever is present and silently omit the partial otherwise).
    L0_per_cell: dict[str, float] = {}
    for cell, d in k_cells.items():
        l0_dict = d.get("L0_post_block_cos_by_layer", {}) or {}
        v = l0_dict.get("0")
        if v is not None:
            L0_per_cell[cell] = float(v)
    lexical = extract_lexical_bag(k_cells)
    baseline_resp = extract_simple_layer_map(old_cells, "response_mean", K_SWEEP_LAYER)

    k_blocks: dict[str, dict] = {}
    for k in k_values_present:
        M = extract_v3_layer_map(k_cells, k, K_SWEEP_LAYER)
        if not M:
            continue
        label = f"V3_response_mean_skip_k{k}_L{K_SWEEP_LAYER}_lit_training"
        k_blocks[label] = build_block(
            label=label,
            M_per_cell=M,
            outcome=outcome,
            tokens=tokens,
            L0_per_cell=L0_per_cell,
            lexical_per_cell=lexical,
            baseline_M_per_cell=baseline_resp,
            baseline_label="response_mean",
            bootstrap_n=bootstrap_n,
            permutation_n=permutation_n,
            run_baseline_diff=True,
            run_permutation=False,
        )
    return {
        "probe_source": K_SWEEP_PROBE_SOURCE,
        "flavor": K_SWEEP_FLAVOR,
        "layer": K_SWEEP_LAYER,
        "k_values_present": k_values_present,
        "n_k_cells": len(k_cells),
        "bootstrap_n": bootstrap_n,
        "permutation_n": permutation_n,
        "k_sweep_blocks": k_blocks,
        "metadata": reproducibility_metadata({"script": "issue468_regress_variants:k_sweep"}),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--probe-sources",
        nargs="+",
        default=["training", "betley"],
        choices=["training", "betley"],
    )
    parser.add_argument(
        "--flavors",
        nargs="+",
        default=["NL", "lit"],
        choices=["NL", "lit"],
    )
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--permutation-n", type=int, default=DEFAULT_PERMUTATION_N)
    parser.add_argument(
        "--position-sweep-cell",
        default="L25_lit_training",
        help="Tag for the position-sweep regression block filename.",
    )
    args = parser.parse_args()

    out_dir = EVAL_DIR_468
    out_dir.mkdir(parents=True, exist_ok=True)

    for probe_source in args.probe_sources:
        for flavor in args.flavors:
            summary = regress_one_pair(
                probe_source=probe_source,
                flavor=flavor,
                layers=args.layers,
                bootstrap_n=args.bootstrap_n,
                permutation_n=args.permutation_n,
            )
            out_path = out_dir / f"regression_variants_{probe_source}_{flavor}.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(
                "Wrote %s (%d predictor blocks)",
                out_path.relative_to(PROJECT_ROOT),
                len(summary["blocks"]),
            )

    # Pre-registered position-sweep block at L25, lit, training.
    if "training" in args.probe_sources and "lit" in args.flavors:
        sweep = build_position_sweep_block(
            probe_source="training",
            flavor="lit",
            layer=25,
            bootstrap_n=args.bootstrap_n,
            permutation_n=args.permutation_n,
        )
        sweep_path = out_dir / "regression_position_sweep_L25_lit_training.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep, f, indent=2)
        logger.info(
            "Wrote %s (%d position blocks)",
            sweep_path.relative_to(PROJECT_ROOT),
            len(sweep["position_sweep_blocks"]),
        )

    # Pre-registered EXPLORATORY V3 k-sweep block at L25, lit, training.
    # Reads `eval_results/issue468/k_sweep_lit_training_L25/<cell>_lit.json`
    # written by the launcher's exploratory k∈{0,4,8,16} step. Skipped
    # silently when no k-sweep dir exists (e.g. main sweep only).
    if K_SWEEP_DIR.exists():
        k_sweep = build_k_sweep_block(
            bootstrap_n=args.bootstrap_n,
            permutation_n=args.permutation_n,
        )
        k_sweep_path = out_dir / "regression_k_sweep_L25_lit_training.json"
        with open(k_sweep_path, "w") as f:
            json.dump(k_sweep, f, indent=2)
        logger.info(
            "Wrote %s (k_values=%s, %d k blocks)",
            k_sweep_path.relative_to(PROJECT_ROOT),
            k_sweep.get("k_values_present", []),
            len(k_sweep.get("k_sweep_blocks", {})),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
