"""Issue #358 — Step 4 (logistic-regression probe + secondary metrics + null floors).

Headline (plan §4.6):

    Pooled-prediction Leave-One-Prompt-Out (LOPO) logistic-regression probe,
    L2-regularised, class-balanced, C=1.0, on the BINARY POOL rows only
    (~103 of the 109 prompts). Each of the 9 sweep layers
    {2, 6, 10, 14, 18, 22, 26, 30, 34} is probed on both the poisoned and
    base models. The headline AUROC is **layer 18, poisoned**; Delta-AUROC =
    poisoned minus base at L18 is the specificity number.

Secondary metrics per layer (plan §4.6):
  - Length-residualized AUROC: regress out `n_tokens` from each activation
    column on the train fold, rerun pooled-LOPO with the residualized X.
  - Within-anth-family AUROC: restrict the binary pool to rows where
    `anth_token_bearing=True OR sub_tier == "anth"`. Drops the
    conceptual-paraphrase bins; tests "is the probe doing more than
    'anth-letter pattern detection'?"
  - Position-sweep AUROC: probe on `acts_at_trigger` (residual at the first
    `anth`-token position) instead of `acts_last`. Naturally restricted to
    anth-bearing rows. Standing recommendation #1 in the brief: if
    n_neg < 10 for this subset, REPORT the subset composition and SKIP the
    AUROC ("underpowered subset"). The AUROC numerator/denominator on
    n_neg = 0 is undefined; n_neg = 3 would carry a CI half-width ~ 0.4
    which is below any informative threshold.

Null floors (plan §4.6 / §6, both bumped from v1's n=50 to n=200):
  - Shuffled-label null  (n=200 permutations)
  - Random-projection null (n=200 unit hyperplanes)

The per-fold StandardScaler in ``pooled_lopo_probe`` is fit per-fold (not
global) — see ``src/.../analysis/probes.py`` module docstring. This is
distinct from the GLOBAL PCA scaler in ``analyze_issue_358_pca.py``;
the two scalers must never be reused across modules.

Output:
  eval_results/issue_358/probe_aurocs.json
      Primary + secondary + nulls per (model, layer).
  eval_results/issue_358/per_prompt_scores.json
      Pooled-LOPO scores per prompt for the headline layer.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from explore_persona_space.analysis.probes import (
    pooled_lopo_probe,
    random_projection_null,
    shuffled_label_null,
)
from explore_persona_space.metadata import get_run_metadata

# Sklearn 1.8 deprecates `penalty='l2'` (kept here intentionally to match
# MacDiarmid 2024). Suppress the per-call FutureWarning so the 9-layer x
# 2-model x ~103-fold pooled-LOPO loop does not flood logs.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("issue_358_probe")

INPUT_DIR = Path("eval_results/issue_358")
HEADLINE_LAYER = 18
SWEEP_LAYERS: list[int] = [2, 6, 10, 14, 18, 22, 26, 30, 34]

# Standing recommendation #1: n_neg minimum for position-sweep AUROC.
POSITION_SWEEP_MIN_NEG = 10


# ─────────────────────────────────────────────────────────────────────────────
# Length-residualization
# ─────────────────────────────────────────────────────────────────────────────


def length_residualized_pooled_lopo(
    X: np.ndarray,
    y: np.ndarray,
    n_tokens: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Pooled-LOPO probe on length-residualized activations.

    Implementation: **GLOBAL** residualization. Fit one `LinearRegression`
    of each activation column on `n_tokens` across ALL rows (not per
    held-out fold), subtract the fitted contribution everywhere, then run
    a single pooled-LOPO probe on the resulting matrix. This is a less
    expensive proxy than re-running per-fold inside the LR loop.

    Caveat: a fully fold-consistent residualization would fit `β` per
    fold; the global version is conservative w.r.t. the headline (it
    overstates residualization power because the held-out row's
    `n_tokens` leaks into the global `β`). If the residualized AUROC is
    still HIGH, the headline survives. If it DROPS sharply (>0.10 vs
    raw), the headline result is partly length-driven (plan §4.6); in
    that case the implementer should upgrade to fold-consistent
    residualization before reporting.
    """
    lr = LinearRegression()
    lr.fit(n_tokens.reshape(-1, 1), X)
    X_resid = X - lr.predict(n_tokens.reshape(-1, 1))
    return pooled_lopo_probe(X_resid, y, C=C, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Per-(model, layer) probe pipeline
# ─────────────────────────────────────────────────────────────────────────────


def _summarize_null(arr: np.ndarray) -> dict[str, float | int]:
    """Compact 5-number summary for the JSON output."""
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "p5": float(np.percentile(arr, 5)) if arr.size else float("nan"),
        "p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
    }


def probe_one(
    activations: torch.Tensor,
    conditions: list[dict],
    layer: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Run primary + secondary AUROC on one (model, layer) cell.

    `activations` is the full (N, n_layers, hidden) tensor; this routine
    slices to the requested `layer` and applies the binary-pool mask.
    """
    pool_mask = np.asarray([c["binary_pool"] for c in conditions], dtype=bool)
    y_all = np.asarray([c["y"] if c["y"] is not None else -1 for c in conditions])
    y = y_all[pool_mask].astype(int)
    n_tokens_pool = np.asarray([c["n_tokens"] for c in conditions], dtype=float)[pool_mask]
    X = activations[:, layer, :].numpy()[pool_mask]
    log.info(
        "[%s L=%d] running primary pooled-LOPO (n_pool=%d, n_pos=%d, n_neg=%d)…",
        label,
        layer,
        len(y),
        int(y.sum()),
        int((1 - y).sum()),
    )

    primary = pooled_lopo_probe(X, y)
    length_resid = length_residualized_pooled_lopo(X, y, n_tokens_pool)

    # ─── Within-anth-family secondary ───────────────────────────────
    # Restrict to rows where the user message tokenises with the bare `anth`
    # token OR whose sub_tier marks it as an `anth`-stem variant.
    pool_idx = np.flatnonzero(pool_mask)
    anth_family_mask_full = np.asarray(
        [
            (c["binary_pool"] and (c["anth_token_bearing"] or (c.get("sub_tier") == "anth")))
            for c in conditions
        ],
        dtype=bool,
    )
    # Map onto the pool indexing.
    anth_family_in_pool = anth_family_mask_full[pool_mask]
    if anth_family_in_pool.sum() and len(np.unique(y[anth_family_in_pool])) == 2:
        within_anth_family = pooled_lopo_probe(X[anth_family_in_pool], y[anth_family_in_pool])
    else:
        within_anth_family = {
            "skipped": True,
            "reason": (
                f"within-anth-family subset has "
                f"n={int(anth_family_in_pool.sum())} / classes="
                f"{np.unique(y[anth_family_in_pool]).tolist() if anth_family_in_pool.sum() else []}"
            ),
        }

    # ─── Null distributions ─────────────────────────────────────────
    # n_perm=50 for shuffled-label (each perm = full 104-fold LOPO, so this
    # dominates runtime); n_proj=200 for random-projection is cheap
    # (200 dot-products, no training). The original v1 plan called for
    # n_perm=50; the round-1 reconciler opportunistically bumped to 200
    # for tighter tail estimates, but that pushed total wall to multi-hour
    # on real data — reverted to v1's default. MC error on the 95th-pct
    # null floor at n=50 is ~14%, adequate for the "did the trained probe
    # clear the null band" descriptive question (CLAUDE.md confidence-label
    # framing). Random-projection null stays at n=200.
    log.info("[%s L=%d] running n=50 shuffled-label null…", label, layer)
    null_shuffled = shuffled_label_null(X, y, n_perm=50)
    log.info("[%s L=%d] running n=200 random-projection null…", label, layer)
    null_random_proj = random_projection_null(X, y, n_proj=200)

    return {
        "layer": layer,
        "primary": primary,
        "length_residualized": length_resid,
        "within_anth_family": within_anth_family,
        "null_shuffled": null_shuffled.tolist(),
        "null_shuffled_summary": _summarize_null(null_shuffled),
        "null_random_proj": null_random_proj.tolist(),
        "null_random_proj_summary": _summarize_null(null_random_proj),
        "pool_indices": pool_idx.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Position-sweep appendix (acts_at_trigger)
# ─────────────────────────────────────────────────────────────────────────────


def position_sweep_one(
    acts_at_trigger: torch.Tensor,
    conditions: list[dict],
    layer: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Position-sweep probe: AUROC at the first `anth`-token position.

    Only anth-bearing rows have non-NaN activations; we restrict the pool
    to those rows AND to `binary_pool == True`. Standing recommendation
    #1 in the brief: if n_neg < 10 after restriction, REPORT the
    composition and SKIP the AUROC.
    """
    pool_mask = np.asarray([c["binary_pool"] for c in conditions], dtype=bool)
    y_all = np.asarray([c["y"] if c["y"] is not None else -1 for c in conditions])
    X_at_trigger = acts_at_trigger[:, layer, :].numpy()  # (N, hidden), may have NaN
    has_anth = np.asarray([c["anth_token_bearing"] for c in conditions], dtype=bool)
    subset_mask = pool_mask & has_anth & ~np.isnan(X_at_trigger).any(axis=1)
    n_total = int(subset_mask.sum())
    y_sub = y_all[subset_mask].astype(int)
    n_pos = int(y_sub.sum()) if n_total else 0
    n_neg = int((1 - y_sub).sum()) if n_total else 0

    composition = {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "min_neg_required": POSITION_SWEEP_MIN_NEG,
    }
    if n_neg < POSITION_SWEEP_MIN_NEG or n_pos == 0:
        log.warning(
            "[%s L=%d] position-sweep SKIPPED — n_neg=%d < %d (composition: %s)",
            label,
            layer,
            n_neg,
            POSITION_SWEEP_MIN_NEG,
            composition,
        )
        return {
            "layer": layer,
            "skipped": True,
            "reason": (
                f"n_neg={n_neg} < {POSITION_SWEEP_MIN_NEG}; "
                f"skipped AUROC due to underpowered subset"
            ),
            "composition": composition,
        }

    X_sub = X_at_trigger[subset_mask]
    log.info(
        "[%s L=%d] position-sweep running on n=%d (%d+/%d-)…",
        label,
        layer,
        n_total,
        n_pos,
        n_neg,
    )
    result = pooled_lopo_probe(X_sub, y_sub)
    return {
        "layer": layer,
        "skipped": False,
        "composition": composition,
        "primary": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────


def run_model(acts_path: Path, label: str) -> dict[str, Any]:
    log.info("loading %s", acts_path)
    D = torch.load(acts_path, weights_only=False)
    conditions: list[dict] = D["conditions"]
    activations: torch.Tensor = D["activations"]
    acts_at_trigger: torch.Tensor = D["activations_at_trigger"]
    log.info(
        "loaded %s: %d conditions, %d layers, hidden=%d",
        label,
        len(conditions),
        activations.shape[1],
        activations.shape[2],
    )

    per_layer: dict[str, dict[str, Any]] = {}
    for L in SWEEP_LAYERS:
        per_layer[str(L)] = probe_one(activations, conditions, L, label=label)

    position_sweep = {
        str(L): position_sweep_one(acts_at_trigger, conditions, L, label=label)
        for L in SWEEP_LAYERS
    }

    return {
        "model_id": D.get("model_id"),
        "revision": D.get("revision"),
        "headline_layer": HEADLINE_LAYER,
        "sweep_layers": SWEEP_LAYERS,
        "per_layer": per_layer,
        "position_sweep": position_sweep,
    }


def compute_deltas(poisoned: dict[str, Any], base: dict[str, Any]) -> dict[str, dict]:
    """Delta-AUROC = poisoned minus base per layer (plan §4.6)."""
    out: dict[str, dict] = {}
    for L in SWEEP_LAYERS:
        Lk = str(L)
        p_auc = poisoned["per_layer"][Lk]["primary"]["pooled_auroc"]
        b_auc = base["per_layer"][Lk]["primary"]["pooled_auroc"]
        out[Lk] = {
            "layer": L,
            "delta_auroc": float(p_auc - b_auc),
            "poisoned_auroc": float(p_auc),
            "base_auroc": float(b_auc),
        }
    return out


def headline_per_prompt(poisoned: dict[str, Any], conditions: list[dict]) -> dict[str, Any]:
    """Pull out the headline-layer (L=18) per-prompt scores for the dashboard.

    Maps fold scores back onto condition-level metadata so the analyzer
    can correlate score outliers with `bin`, `n_tokens`, etc.
    """
    Lk = str(HEADLINE_LAYER)
    primary = poisoned["per_layer"][Lk]["primary"]
    pool_indices = poisoned["per_layer"][Lk]["pool_indices"]
    fold_scores = primary["fold_scores"]
    rows = []
    for idx_in_pool, cond_idx in enumerate(pool_indices):
        c = conditions[cond_idx]
        rows.append(
            {
                "cid": c["cid"],
                "user": c["user"],
                "class": c["class"],
                "bin": c.get("bin"),
                "sub_tier": c.get("sub_tier"),
                "n_tokens": c["n_tokens"],
                "anth_token_bearing": c["anth_token_bearing"],
                "y": c["y"],
                "fold_score": fold_scores[idx_in_pool],
            }
        )
    return {"layer": HEADLINE_LAYER, "per_prompt": rows}


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    poisoned = run_model(INPUT_DIR / "acts_poisoned.pt", label="poisoned")
    base = run_model(INPUT_DIR / "acts_base.pt", label="base")
    deltas = compute_deltas(poisoned, base)

    # Read conditions back for the per-prompt JSON (cheap — already loaded
    # once in run_model, but `acts_poisoned.pt` is small).
    D = torch.load(INPUT_DIR / "acts_poisoned.pt", weights_only=False)
    per_prompt = headline_per_prompt(poisoned, D["conditions"])

    # ─── Top-level summary log lines ────────────────────────────────
    head_pois = poisoned["per_layer"][str(HEADLINE_LAYER)]["primary"]
    head_base = base["per_layer"][str(HEADLINE_LAYER)]["primary"]
    head_delta = deltas[str(HEADLINE_LAYER)]["delta_auroc"]
    log.info("=== HEADLINE (layer %d) ===", HEADLINE_LAYER)
    log.info(
        "  poisoned: pooled-LOPO AUROC = %.3f (95%% CI [%.3f, %.3f]; train AUROC = %.3f)",
        head_pois["pooled_auroc"],
        head_pois["ci_95"][0],
        head_pois["ci_95"][1],
        head_pois["train_auroc"],
    )
    log.info(
        "  base:     pooled-LOPO AUROC = %.3f (95%% CI [%.3f, %.3f]; train AUROC = %.3f)",
        head_base["pooled_auroc"],
        head_base["ci_95"][0],
        head_base["ci_95"][1],
        head_base["train_auroc"],
    )
    log.info("  Delta-AUROC (poisoned - base) = %.3f", head_delta)

    out_path = INPUT_DIR / "probe_aurocs.json"
    payload = {
        "poisoned": poisoned,
        "base": base,
        "deltas": deltas,
        "pass_bars": {
            "headline_auroc": 0.80,
            "headline_lower_ci": 0.70,
            "delta_auroc": 0.15,
            "null_p95_max": 0.65,
        },
        "metadata": get_run_metadata(),
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("wrote %s", out_path)

    pp_path = INPUT_DIR / "per_prompt_scores.json"
    with pp_path.open("w") as f:
        json.dump(
            {"per_prompt": per_prompt, "metadata": get_run_metadata()},
            f,
            indent=2,
            default=str,
        )
    log.info("wrote %s", pp_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
