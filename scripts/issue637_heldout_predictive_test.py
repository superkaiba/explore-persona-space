"""Issue #637 — held-out cross-validation of the leakage-asymmetry rank.

Parent #526 answered "how complex must the directional leakage predictor g be?"
IN-SAMPLE (fit + score on the same cells). This script re-answers it OUT-OF-SAMPLE:
per-CELL 80/20 split, fit three nested predictors on the train cells, score on the
held-out cells, with a 1000-bootstrap CI over held-out cells + a 20-seed
split-stability loop + a shuffled-context permutation control.

The three nested predictors (fit on TRAIN cells, scored on TEST cells):
  arm 1 sym          g_sym(i,j) = mu + s_i + s_j   (symmetric two-way LS; rank-0 antisym baseline)
  arm 2 sym_scalar   g_sym(i,j) + (s_i - s_j), s = (b-r)/2  (rank-1 antisymmetry — headline arm)
  arm 3 full         (j,i) in train: 2*g_sym(i,j) - M[j,i]  (uses the OBSERVED transpose)
                     else:           g_sym(i,j) + (s_i - s_j)  (transpose held out -> rank-1)

Kill-4 reproduction asserts (FAIL FAST on load):
  (a) in-sample L0_antisym_fraction + L2_scalar_antisym_fraction reproduce
      figures/issue_526/gate_ladder_results.json['537'][behavior] to 3 decimals;
  (b) the v2 predictor formula  Ahat_ij = s_i - s_j  with s = (b - r)/2  evaluated on ALL
      off-diag cells reproduces L2_scalar_antisym_fraction to 3 decimals
      (matches scripts/issue526_asym_gate_ladder.py:scalar_antisym_fraction, the in-sample anchor).

0 GPU. Local CPU. < 5 min wall-time.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime

sys.path.insert(0, "scripts")

import numpy as np
from issue526_asym_gate_ladder import (
    antisym_fraction,
    fit_two_way_additive,
    load_537,
    offdiag_mask,
    scalar_antisym_fraction,
)

GATE_LADDER_PATH = "figures/issue_526/gate_ladder_results.json"
G_META_PATH = "eval_results/issue_537/G_tensor/G_meta.json"
G1_REGRESSION_PATH = "eval_results/issue_537/analysis/g1_regression.json"
OUT_DIR = "figures/issue_637"
BEHAVIORS = ["marker", "fact", "refusal", "sycophancy", "em"]
ARMS = ["sym", "sym_scalar", "full_pairwise"]


# ----------------------------------------------------------------------------- helpers
def offdiag_cells(n):
    """All 240 off-diagonal (i, j) index pairs for an n x n matrix."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def split_cells(cells, frac=0.8, seed=42):
    """Per-CELL random split (v2). Returns (train_cells, test_cells)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cells))
    k = round(frac * len(cells))
    train_cells = [cells[p] for p in perm[:k]]
    test_cells = [cells[p] for p in perm[k:]]
    return train_cells, test_cells


def symmetric_two_way_fit(M, train_cells):
    """LS fit M_ij ~ mu + s_i + s_j on TRAIN cells (a SINGLE scalar s used both sides).

    Symmetric: the same per-context scalar enters the source and target positions, so the
    fitted matrix is symmetric (g_sym(i,j) == g_sym(j,i)). Returns a predict(i, j) closure.
    """
    n = M.shape[0]
    y = np.array([M[i, j] for (i, j) in train_cells])
    # design: intercept + n context dummies; context k gets +1 for EACH endpoint it appears at.
    X = np.zeros((len(y), 1 + n))
    X[:, 0] = 1.0
    for row, (i, j) in enumerate(train_cells):
        X[row, 1 + i] += 1.0
        X[row, 1 + j] += 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu0 = beta[0]
    s = beta[1:].copy()
    # recenter for identifiability (sum(s) = 0; mu absorbs the mean)
    mu = mu0 + 2.0 * s.mean()
    s = s - s.mean()

    def predict(i, j):
        return mu + s[i] + s[j]

    return predict, mu, s


def fit_two_way_additive_on_cells(M, train_cells):
    """fit_two_way_additive restricted to TRAIN cells only.

    Mirrors scripts/issue526_asym_gate_ladder.py:fit_two_way_additive (M_ij ~ mu + b_i + r_j,
    min-norm lstsq, then recenter b,r to zero-mean) but fits ONLY the supplied train cells.
    Returns (mu, b, r).
    """
    n = M.shape[0]
    y = np.array([M[i, j] for (i, j) in train_cells])
    X = np.zeros((len(y), 1 + 2 * n))
    X[:, 0] = 1.0
    for k, (i, j) in enumerate(train_cells):
        X[k, 1 + i] = 1.0
        X[k, 1 + n + j] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu0 = beta[0]
    b = beta[1 : 1 + n].copy()
    r = beta[1 + n : 1 + 2 * n].copy()
    mu = mu0 + b.mean() + r.mean()
    b = b - b.mean()
    r = r - r.mean()
    return mu, b, r


def predict_arms(M, train_cells, test_cells):
    """Build the three nested predictions on TEST cells (fit on TRAIN cells).

    Returns (preds_sym, preds_scal, preds_full, y, n_full_fallback).
    """
    predict_sym, _, _ = symmetric_two_way_fit(M, train_cells)
    _, b, r = fit_two_way_additive_on_cells(M, train_cells)
    s = (b - r) / 2.0  # v2 antisym scalar (matches scalar_antisym_fraction's Ahat_ij = s_i - s_j)
    train_set = set(train_cells)
    preds_sym, preds_scal, preds_full, y = [], [], [], []
    n_full_fallback = 0
    for i, j in test_cells:
        gsym = predict_sym(i, j)
        preds_sym.append(gsym)
        preds_scal.append(gsym + (s[i] - s[j]))  # v2: was (b_i - r_j); non-antisym, fixed
        if (j, i) in train_set:
            preds_full.append(2.0 * gsym - M[j, i])  # uses the OBSERVED transpose
        else:
            preds_full.append(gsym + (s[i] - s[j]))  # transpose also held out -> rank-1 fallback
            n_full_fallback += 1
        y.append(M[i, j])
    return (
        np.array(preds_sym),
        np.array(preds_scal),
        np.array(preds_full),
        np.array(y),
        n_full_fallback,
    )


def r2(pred, y):
    """Out-of-sample R^2 = 1 - SS_res / SS_tot on a held-out set."""
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def bootstrap_arm_ci(pred, y, n_boot, seed):
    """Bootstrap the held-out R^2 of one arm by resampling TEST cells with replacement."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    n = len(y)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[k] = r2(pred[idx], y[idx])
    return (
        float(np.nanmean(vals)),
        float(np.nanpercentile(vals, 2.5)),
        float(np.nanpercentile(vals, 97.5)),
    )


def paired_delta_ci(p_a, p_b, y, n_boot, seed):
    """Bootstrap the PAIRED difference r2(p_a) - r2(p_b) on the SAME resampled test cells."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    n = len(y)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        vals[k] = r2(p_a[idx], yb) - r2(p_b[idx], yb)
    return (
        float(np.nanmean(vals)),
        float(np.nanpercentile(vals, 2.5)),
        float(np.nanpercentile(vals, 97.5)),
    )


def in_sample_l2_fraction(b, r, M):
    """L2 scalar antisym fraction from (b, r) on ALL off-diag cells.

    Reproduces scripts/issue526_asym_gate_ladder.py:scalar_antisym_fraction exactly:
    Ahat_ij = s_i - s_j with s = (b - r)/2; fraction = 1 - SS_res / SS_tot on off-diag antisym.
    """
    n = M.shape[0]
    s = (b - r) / 2.0
    off = offdiag_cells(n)
    A_full = np.array([(M[i, j] - M[j, i]) / 2.0 for (i, j) in off])
    A_hat = np.array([s[i] - s[j] for (i, j) in off])
    ss_tot = np.sum(A_full**2)
    ss_res = np.sum((A_full - A_hat) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def assert_in_sample_reproduction(M, behavior, anchor):
    """Kill-4: fail-fast if (a) L0 or (b) the v2 predictor formula drifts from the anchor.

    (a) in-sample L0_antisym_fraction reproduces the anchor to 3 decimals.
    (b) the v2 scalar predictor s_i - s_j (s = (b-r)/2) on ALL off-diag cells reproduces
        L2_scalar_antisym_fraction to 3 decimals. (b) uses the in-sample additive fit on
        ALL off-diag cells, identical to scalar_antisym_fraction (the anchor's own definition).
    Returns (L0_recomputed, L2_recomputed).
    """
    L0_recomputed = float(antisym_fraction(M))
    L0_expected = anchor[behavior]["L0_antisym_fraction"]
    if abs(L0_recomputed - L0_expected) >= 1e-3:
        raise AssertionError(
            f"Kill-4(a) {behavior}: in-sample L0 antisym fraction reproduces "
            f"{L0_recomputed:.4f} but anchor expects {L0_expected:.4f} "
            f"(upstream G_meta.json drifted?)"
        )
    # (b) the v2 predictor formula on the in-sample (all off-diag) additive fit
    _, b, r, _ = fit_two_way_additive(M)
    L2_recomputed = float(in_sample_l2_fraction(b, r, M))
    L2_expected = anchor[behavior]["L2_scalar_antisym_fraction"]
    if abs(L2_recomputed - L2_expected) >= 1e-3:
        raise AssertionError(
            f"Kill-4(b) {behavior}: v2 predictor formula s_i - s_j (s=(b-r)/2) reproduces "
            f"L2={L2_recomputed:.4f} but anchor expects {L2_expected:.4f} "
            f"(predictor formula drift from scalar_antisym_fraction)"
        )
    # cross-check against the reused helper directly (defense in depth)
    helper_frac, _, _, _ = scalar_antisym_fraction(M)
    if abs(L2_recomputed - float(helper_frac)) >= 1e-6:
        raise AssertionError(
            f"Kill-4(b) {behavior}: in_sample_l2_fraction {L2_recomputed:.6f} disagrees with "
            f"scalar_antisym_fraction {float(helper_frac):.6f} (formula mismatch)"
        )
    return L0_recomputed, L2_recomputed


def shuffled_context_scalar_r2(M, train_cells, test_cells, seed):
    """Permutation control: shuffle context ids before the additive fit, predict the scalar arm.

    If the rank-1 gain is real per-context structure it collapses under id-permutation; the
    held-out gap between the real scalar arm and this control is the real effect size.
    """
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    Mp = M[np.ix_(perm, perm)]  # relabel both source and target consistently
    predict_sym, _, _ = symmetric_two_way_fit(Mp, train_cells)
    _, b, r = fit_two_way_additive_on_cells(Mp, train_cells)
    s = (b - r) / 2.0
    preds, y = [], []
    for i, j in test_cells:
        preds.append(predict_sym(i, j) + (s[i] - s[j]))
        y.append(Mp[i, j])
    return r2(np.array(preds), np.array(y))


# ----------------------------------------------------------------------------- per-behavior run
def run_behavior(beh, M, anchor, n_boot, n_split_seeds, base_seed=42):
    """Full held-out CV for one behavior. Returns a JSON-serializable dict."""
    n = M.shape[0]
    L0_recomputed, L2_recomputed = assert_in_sample_reproduction(M, beh, anchor)

    cells = offdiag_cells(n)
    train_cells, test_cells = split_cells(cells, frac=0.8, seed=base_seed)
    preds_sym, preds_scal, preds_full, y, n_full_fallback = predict_arms(M, train_cells, test_cells)

    # in-sample held-out R^2 point estimates (headline seed)
    r2_sym = r2(preds_sym, y)
    r2_scal = r2(preds_scal, y)
    r2_full = r2(preds_full, y)

    # ALSO the in-sample (train-cell) R^2 per arm, for the thin overlay marker in the plot
    preds_sym_tr, preds_scal_tr, preds_full_tr, y_tr, _ = predict_arms(M, train_cells, train_cells)
    insample_r2 = {
        "sym": r2(preds_sym_tr, y_tr),
        "sym_scalar": r2(preds_scal_tr, y_tr),
        "full_pairwise": r2(preds_full_tr, y_tr),
    }

    # bootstrap CI per arm
    heldout = {}
    for arm, pred in (
        ("sym", preds_sym),
        ("sym_scalar", preds_scal),
        ("full_pairwise", preds_full),
    ):
        mean, lo, hi = bootstrap_arm_ci(pred, y, n_boot=n_boot, seed=base_seed)
        heldout[arm] = {
            "r2_point": float(r2(pred, y)),
            "r2_boot_mean": mean,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }

    # paired deltas
    d_scal_mean, d_scal_lo, d_scal_hi = paired_delta_ci(
        preds_scal, preds_sym, y, n_boot=n_boot, seed=base_seed
    )
    d_full_mean, d_full_lo, d_full_hi = paired_delta_ci(
        preds_full, preds_scal, y, n_boot=n_boot, seed=base_seed
    )
    dR2_scalar = {
        "point": float(r2_scal - r2_sym),
        "boot_mean": d_scal_mean,
        "ci95_lo": d_scal_lo,
        "ci95_hi": d_scal_hi,
        "ci_excludes_0": bool(d_scal_lo > 0 or d_scal_hi < 0),
        "ci_excludes_0_positive": bool(d_scal_lo > 0),  # rank-1 GAIN is real (the H1 / §6 sense)
        "ci_excludes_0_negative": bool(d_scal_hi < 0),
    }
    dR2_full = {
        "point": float(r2_full - r2_scal),
        "boot_mean": d_full_mean,
        "ci95_lo": d_full_lo,
        "ci95_hi": d_full_hi,
        "ci_excludes_0": bool(d_full_lo > 0 or d_full_hi < 0),
        "ci_excludes_0_positive": bool(d_full_lo > 0),  # full pairwise GENERALIZES beyond rank-1
        "ci_excludes_0_negative": bool(
            d_full_hi < 0
        ),  # full pairwise WORSE than rank-1 out-of-sample
    }

    # shuffled-context control (headline seed)
    r2_scal_shuffled = shuffled_context_scalar_r2(M, train_cells, test_cells, seed=base_seed)

    # 20-seed split-stability loop
    stab_dscal, stab_dfull, stab_nfb = [], [], []
    for sd in range(base_seed, base_seed + n_split_seeds):
        tr, te = split_cells(cells, frac=0.8, seed=sd)
        ps, psc, pf, yy, nfb = predict_arms(M, tr, te)
        stab_dscal.append(float(r2(psc, yy) - r2(ps, yy)))
        stab_dfull.append(float(r2(pf, yy) - r2(psc, yy)))
        stab_nfb.append(int(nfb))
    stab_dscal = np.array(stab_dscal)
    stab_dfull = np.array(stab_dfull)
    split_stability = {
        "n_seeds": int(n_split_seeds),
        "seeds": list(range(base_seed, base_seed + n_split_seeds)),
        "dR2_scalar_median": float(np.median(stab_dscal)),
        "dR2_scalar_iqr": [
            float(np.percentile(stab_dscal, 25)),
            float(np.percentile(stab_dscal, 75)),
        ],
        "dR2_scalar_n_crosses_0": int(np.sum(stab_dscal <= 0)),
        "dR2_full_median": float(np.median(stab_dfull)),
        "dR2_full_iqr": [
            float(np.percentile(stab_dfull, 25)),
            float(np.percentile(stab_dfull, 75)),
        ],
        "dR2_full_n_crosses_0": int(np.sum(stab_dfull <= 0)),
        "n_full_fallback_mean": float(np.mean(stab_nfb)),
    }

    # held-out predicted-vs-actual scatter arrays (exploratory dump for the analyzer)
    scatter = {
        "y_test": y.tolist(),
        "pred_sym": preds_sym.tolist(),
        "pred_sym_scalar": preds_scal.tolist(),
        "pred_full_pairwise": preds_full.tolist(),
        "test_cells": [list(c) for c in test_cells],
    }

    # Verdict per §3/§6/Kill-3. The directional sense matters: "needs pairwise" requires
    # full-pairwise to BEAT rank-1 out-of-sample (dR2_full CI excludes 0 ON THE POSITIVE
    # side). A dR2_full CI entirely BELOW 0 means full-pairwise is WORSE than rank-1 on
    # held-out cells (the observed transpose adds noise, not signal) — that CONFIRMS rank-1
    # sufficiency, it does not call for pairwise.
    scalar_real = dR2_scalar["ci_excludes_0_positive"]
    full_beats_rank1 = dR2_full["ci_excludes_0_positive"]
    if full_beats_rank1:
        verdict = "needs pairwise (held-out)"
    elif scalar_real:
        verdict = "rank-1 sufficient (held-out)"
    else:
        verdict = "asymmetry not recoverable out-of-sample"

    return {
        "behavior": beh,
        "n_contexts": int(n),
        "n_offdiag_cells": len(cells),
        "n_train": len(train_cells),
        "n_test": len(test_cells),
        "n_full_fallback": int(n_full_fallback),
        "in_sample_L0_antisym_fraction": L0_recomputed,
        "in_sample_L2_scalar_antisym_fraction": L2_recomputed,
        "in_sample_r2": {k: float(v) for k, v in insample_r2.items()},
        "heldout_r2": heldout,
        "dR2_scalar": dR2_scalar,
        "dR2_full": dR2_full,
        "shuffled_context_scalar_heldout_r2": float(r2_scal_shuffled),
        "split_stability": split_stability,
        "scatter": scatter,
        "verdict": verdict,
    }


# ----------------------------------------------------------------------------- io
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def env_versions():
    import platform

    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def main():
    ap = argparse.ArgumentParser(description="Issue #637 held-out predictive test")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 behavior (marker), 20 bootstraps, 5-seed split-stability",
    )
    ap.add_argument("--frac", type=float, default=0.8, help="train fraction (default 0.8)")
    ap.add_argument("--seed", type=int, default=42, help="base random seed (default 42)")
    args = ap.parse_args()

    n_boot = 20 if args.smoke else 1000
    n_split_seeds = 5 if args.smoke else 20
    behaviors = ["marker"] if args.smoke else BEHAVIORS

    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(GATE_LADDER_PATH) as f:
        anchor = json.load(f)["537"]
    data = load_537()

    # sanity: every behavior has a complete off-diagonal (plan assumption #1)
    for beh in behaviors:
        M = data[beh]["M"]
        n = M.shape[0]
        od = offdiag_mask(n)
        n_nan = int(np.isnan(M[od]).sum())
        n_sat = int((data[beh]["SAT"] & od).sum())
        if n_nan != 0:
            raise AssertionError(f"{beh}: {n_nan} NaN off-diag cells (expected 0)")
        if n_sat != 0:
            raise AssertionError(f"{beh}: {n_sat} saturated off-diag cells (expected 0)")

    results = {}
    for beh in behaviors:
        M = data[beh]["M"]
        results[beh] = run_behavior(
            beh, M, anchor, n_boot=n_boot, n_split_seeds=n_split_seeds, base_seed=args.seed
        )
        rb = results[beh]
        ds, df = rb["dR2_scalar"], rb["dR2_full"]
        print(
            f"{beh:12s} L0={rb['in_sample_L0_antisym_fraction']:.3f} "
            f"L2={rb['in_sample_L2_scalar_antisym_fraction']:.3f}  "
            f"heldout R2 sym={rb['heldout_r2']['sym']['r2_point']:+.3f} "
            f"scal={rb['heldout_r2']['sym_scalar']['r2_point']:+.3f} "
            f"full={rb['heldout_r2']['full_pairwise']['r2_point']:+.3f}  "
            f"dR2_scalar={ds['point']:+.3f} [{ds['ci95_lo']:+.3f},{ds['ci95_hi']:+.3f}]"
            f"{'*' if ds['ci_excludes_0'] else ''}  "
            f"dR2_full={df['point']:+.3f} [{df['ci95_lo']:+.3f},{df['ci95_hi']:+.3f}]"
            f"{'*' if df['ci_excludes_0'] else ''}  "
            f"n_full_fallback={rb['n_full_fallback']}  -> {rb['verdict']}"
        )

    payload = {
        "issue": 637,
        "smoke": bool(args.smoke),
        "params": {
            "split_frac": args.frac,
            "n_bootstrap": n_boot,
            "n_split_seeds": n_split_seeds,
            "base_seed": args.seed,
            "predictor_formula": "rank-1 antisym = s_i - s_j, s = (b - r)/2 (v2)",
            "full_pairwise_rule": "2*g_sym(i,j) - M[j,i] when (j,i) in train, else rank-1 fallback",
        },
        "behaviors": results,
    }
    out_json = f"{OUT_DIR}/heldout_predictive_test.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nWrote {out_json}")

    meta = {
        "issue": 637,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "smoke": bool(args.smoke),
        "params": payload["params"],
        "source_files": {
            "gate_ladder_results": {
                "path": GATE_LADDER_PATH,
                "sha256": file_sha256(GATE_LADDER_PATH),
            },
            "G_meta": {"path": G_META_PATH, "sha256": file_sha256(G_META_PATH)},
            "g1_regression": {
                "path": G1_REGRESSION_PATH,
                "sha256": file_sha256(G1_REGRESSION_PATH),
            },
        },
        "reproduction_asserts": {
            beh: {
                "in_sample_L0": results[beh]["in_sample_L0_antisym_fraction"],
                "in_sample_L2": results[beh]["in_sample_L2_scalar_antisym_fraction"],
                "anchor_L0": anchor[beh]["L0_antisym_fraction"],
                "anchor_L2": anchor[beh]["L2_scalar_antisym_fraction"],
                "passed": True,  # AssertionError would have aborted otherwise
            }
            for beh in behaviors
        },
        "env": env_versions(),
        "reused": (
            "scripts/issue526_asym_gate_ladder.py "
            "(load_537, offdiag_mask, fit_two_way_additive, scalar_antisym_fraction, "
            "antisym_fraction)"
        ),
    }
    out_meta = f"{OUT_DIR}/heldout_predictive_test.meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
