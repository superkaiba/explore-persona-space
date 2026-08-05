#!/usr/bin/env python3
"""Task #1491 — paired-bootstrap contrasts from the recomputed ridge test preds.

The plan §3 Phase-4 contrasts script (`issue1491_ladder_contrasts.py`) consumed
per-context test preds at ``data/issue_1491/preds/`` on the pod, which was
verify-then-terminated after the fits stage before Phase 4 ran. The cap-hit
restriction round (`epm:progress v54/v55`) refit ridge per rung from the HF
captures on the VM — matching the committed test R² to ~1e-16 — and persisted
the per-context test preds + targets to
``data/issue_1491/preds_recomputed/<slug>_test_preds_ridge_recomputed.npz``
(keys: pred_te (1000, h), y_te (1000, h), ci_te, cap_hit_te, selected_lambda).

This script computes the registered ridge contrasts from those preds:

- PRIMARY (plan §3): Δ = ridge R²(32B) − ridge R²(0.5B), paired bootstrap
  (1,000 draws, seed 42, ONE shared resample matrix over the 1,000 pinned
  test contexts), two-sided bootstrap p.
- Adjacent rung pairs (0.5→1.5, 1.5→3, 3→7, 7→14, 14→32) + the fixed-h depth
  pair (14B vs 32B, both h=5120), same shared resample matrix.
- Raw AND ceiling-normalized R² (normalized = R²/ceiling per rung; the
  two-draw ceiling is treated as a FIXED scalar — resampling it would need
  the ceiling-draw captures, which live on HF, not locally).
- Descriptive Spearman monotonicity of ridge R² across the 6 rungs per draw.

The MLP/KRR preds were NOT persisted off-pod, so the registered ΔΓ
(nonlinear-gap) contrast is NOT computable here; the JSON records that
explicitly instead of silently omitting it.

Metric parity: whole-map variance-weighted R² = 1 − Σ SSE_d / Σ SST_d with
SST against the evaluated split's own mean (`issue1491_ladder_fits.py`
`_r2_var_weighted`); each bootstrap draw recomputes SST against the draw's
own mean. Validation: the full-sample recompute from the npz must match the
committed `fits_<slug>.json` ridge test_r2 (tol 1e-6) before any contrast is
emitted.

Writes: eval_results/issue_1491/scale_ladder/adjacent_contrasts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREDS_DIR = ROOT / "data" / "issue_1491" / "preds_recomputed"
FITS_DIR = ROOT / "eval_results" / "issue_1491" / "scale_ladder"
OUT_PATH = FITS_DIR / "adjacent_contrasts.json"

SCALE_ORDER = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]
ADJACENT_PAIRS = [
    ("scale05", "scale15"),
    ("scale15", "scale3"),
    ("scale3", "scale7_refit"),
    ("scale7_refit", "scale14"),
    ("scale14", "scale32"),
]
PRIMARY_PAIR = ("scale05", "scale32")
DEPTH_PAIR = ("scale14", "scale32")  # fixed h=5120, 48 vs 64 layers

N_BOOT = 1000
SEED = 42
TOL_COMMITTED = 1e-6


def _r2_full(pred: np.ndarray, y: np.ndarray) -> float:
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _r2_boot(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap R² per draw, SST against the draw's own mean.

    idx: (B, n) integer resample matrix. Returns (B,) array of R² values.
    """
    n = y.shape[0]
    sse_i = ((y - pred) ** 2).sum(axis=1)  # (n,)
    ynorm_i = (y**2).sum(axis=1)  # (n,)
    # counts[b, i] = multiplicity of context i in draw b
    B = idx.shape[0]
    counts = np.zeros((B, n), dtype=np.float64)
    rows = np.repeat(np.arange(B), idx.shape[1])
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    sse_b = counts @ sse_i  # (B,)
    sum_ynorm_b = counts @ ynorm_i  # (B,)
    sum_y_b = counts @ y.astype(np.float64)  # (B, h)
    n_b = float(idx.shape[1])
    sst_b = sum_ynorm_b - (sum_y_b**2).sum(axis=1) / n_b
    return 1.0 - sse_b / (sst_b + 1e-30)


def _spearman_rows(mat: np.ndarray, x_rank: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of mat (B, k) against x_rank (k,) — no ties in x."""
    ranks = mat.argsort(axis=1).argsort(axis=1).astype(np.float64)
    xr = x_rank.astype(np.float64)
    ranks_c = ranks - ranks.mean(axis=1, keepdims=True)
    xr_c = xr - xr.mean()
    num = (ranks_c * xr_c).sum(axis=1)
    den = np.sqrt((ranks_c**2).sum(axis=1) * (xr_c**2).sum())
    return num / den


def main() -> None:
    rng = np.random.default_rng(SEED)

    data: dict[str, dict] = {}
    ci_ref: np.ndarray | None = None
    for slug in SCALE_ORDER:
        z = np.load(PREDS_DIR / f"{slug}_test_preds_ridge_recomputed.npz")
        pred, y, ci = z["pred_te"].astype(np.float64), z["y_te"].astype(np.float64), z["ci_te"]
        order = np.argsort(ci)  # rungs persist rows in shard order; align by context id
        pred, y, ci = pred[order], y[order], ci[order]
        if ci_ref is None:
            ci_ref = ci
        else:
            assert np.array_equal(ci_ref, ci), f"{slug}: test context ids differ from reference"
        fits = json.loads((FITS_DIR / f"fits_{slug}.json").read_text())
        committed = fits["predictors"]["ridge"]["test_r2"]
        ceiling = fits["ceiling_two_draw"]["ceiling_var_weighted_r"]
        full = _r2_full(pred, y)
        delta_committed = abs(full - committed)
        assert delta_committed < TOL_COMMITTED, (
            f"{slug}: recomputed R² {full} vs committed {committed} (|Δ|={delta_committed})"
        )
        data[slug] = {
            "pred": pred,
            "y": y,
            "r2_full": full,
            "committed": committed,
            "ceiling": ceiling,
            "abs_delta_vs_committed": delta_committed,
        }

    n = data[SCALE_ORDER[0]]["y"].shape[0]
    idx = rng.integers(0, n, size=(N_BOOT, n))  # ONE shared resample matrix

    boots_raw = {s: _r2_boot(data[s]["pred"], data[s]["y"], idx) for s in SCALE_ORDER}
    boots_norm = {s: boots_raw[s] / data[s]["ceiling"] for s in SCALE_ORDER}

    def contrast(a: str, b: str, boots: dict, fulls: dict) -> dict:
        d_b = boots[b] - boots[a]
        d_full = fulls[b] - fulls[a]
        lo, hi = np.percentile(d_b, [2.5, 97.5])
        p_lo = (1 + int((d_b <= 0).sum())) / (N_BOOT + 1)
        p_hi = (1 + int((d_b >= 0).sum())) / (N_BOOT + 1)
        return {
            "pair": [a, b],
            "delta_full": d_full,
            "boot_mean": float(d_b.mean()),
            "ci95": [float(lo), float(hi)],
            "p_two_sided": float(min(1.0, 2 * min(p_lo, p_hi))),
            "n_boot": N_BOOT,
            "seed": SEED,
        }

    fulls_raw = {s: data[s]["r2_full"] for s in SCALE_ORDER}
    fulls_norm = {s: data[s]["r2_full"] / data[s]["ceiling"] for s in SCALE_ORDER}

    out = {
        "dv": "ridge held-out variance-weighted test R2 (primary layer, n_train=25k, n_test=1000)",
        "method": (
            "paired bootstrap over the 1,000 shared pinned test contexts; ONE resample "
            "matrix (seed 42, 1,000 draws) shared across all rungs; SST recomputed "
            "against each draw's own mean; ceiling treated as a fixed scalar per rung"
        ),
        "preds_provenance": (
            "data/issue_1491/preds_recomputed/<slug>_test_preds_ridge_recomputed.npz "
            "(cap-hit round VM refits; committed-R2 parity asserted at tol 1e-6 per rung)"
        ),
        "validation_abs_delta_vs_committed": {
            s: data[s]["abs_delta_vs_committed"] for s in SCALE_ORDER
        },
        "r2_full": fulls_raw,
        "r2_normalized_full": fulls_norm,
        "per_rung_boot_ci95_raw": {
            s: [float(q) for q in np.percentile(boots_raw[s], [2.5, 97.5])] for s in SCALE_ORDER
        },
        "per_rung_boot_ci95_normalized": {
            s: [float(q) for q in np.percentile(boots_norm[s], [2.5, 97.5])] for s in SCALE_ORDER
        },
        "primary_contrast_raw": contrast(*PRIMARY_PAIR, boots_raw, fulls_raw),
        "primary_contrast_normalized": contrast(*PRIMARY_PAIR, boots_norm, fulls_norm),
        "depth_pair_raw": contrast(*DEPTH_PAIR, boots_raw, fulls_raw),
        "depth_pair_normalized": contrast(*DEPTH_PAIR, boots_norm, fulls_norm),
        "adjacent_raw": [contrast(a, b, boots_raw, fulls_raw) for a, b in ADJACENT_PAIRS],
        "adjacent_normalized": [contrast(a, b, boots_norm, fulls_norm) for a, b in ADJACENT_PAIRS],
        "spearman_monotonicity_raw": {
            "point": float(
                _spearman_rows(np.array([[fulls_raw[s] for s in SCALE_ORDER]]), np.arange(6))[0]
            ),
            "boot_ci95": [
                float(q)
                for q in np.percentile(
                    _spearman_rows(
                        np.stack([boots_raw[s] for s in SCALE_ORDER], axis=1), np.arange(6)
                    ),
                    [2.5, 97.5],
                )
            ],
        },
        "registered_verdict_raw": None,  # filled below
        "delta_gamma_nonlinear_gap": {
            "status": "NOT COMPUTABLE from persisted artifacts",
            "reason": (
                "MLP/KRR per-context test preds were written to the pod's "
                "data/issue_1491/preds/ and not uploaded before verify-then-terminate; "
                "only the ridge preds were recomputed on the VM by the cap-hit round. "
                "Point values of the gap per rung are in fits_<slug>.json."
            ),
        },
        "confound_qualifier": (
            "sample-efficiency-confounded: the plan §4 d-confound controls (per-scale "
            "R2-vs-n sub-ladder, rp896 random-projection control, tier-B/length reads) "
            "were deferred by the fits stage and have not landed (open concern "
            "ladder-deferred-confound-controls); per that concern every registered "
            "verdict here carries this qualifier"
        ),
    }

    pc = out["primary_contrast_raw"]
    if pc["ci95"][0] > 0:
        out["registered_verdict_raw"] = (
            "Predictability-increases (raw; sample-efficiency-confounded)"
        )
    elif pc["ci95"][1] < 0:
        out["registered_verdict_raw"] = (
            "Predictability-decreases (raw; sample-efficiency-confounded)"
        )
    else:
        out["registered_verdict_raw"] = "Scale-inconclusive (raw; sample-efficiency-confounded)"

    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_PATH}")
    print("validation |recomputed - committed| per rung:")
    for s in SCALE_ORDER:
        print(f"  {s}: {data[s]['abs_delta_vs_committed']:.2e}")
    print("primary raw:", json.dumps(out["primary_contrast_raw"]))
    print("primary norm:", json.dumps(out["primary_contrast_normalized"]))
    print("depth raw:", json.dumps(out["depth_pair_raw"]))
    print("depth norm:", json.dumps(out["depth_pair_normalized"]))
    for c in out["adjacent_raw"]:
        print("adj raw:", json.dumps(c))
    for c in out["adjacent_normalized"]:
        print("adj norm:", json.dumps(c))
    print("spearman raw:", json.dumps(out["spearman_monotonicity_raw"]))
    print("verdict:", out["registered_verdict_raw"])


if __name__ == "__main__":
    main()
