"""Issue #779 fitter-fair-comparison — baseline floors for the D1 plot.

Computes the "is it just consecutive-token similarity?" baselines + the
shuffled-pairing sanity floor + predict-the-mean, under the SAME fixed split,
the SAME ridge val-selected read-out layer per input variant, and the SAME
pooled-R2 + bootstrap-CI machinery as the fitters — then merges them into
``fair_comparison.json`` so ``make_fig_d1`` can render them alongside the
fitters. Read-only on the fitters' own results; adds only ``baselines`` /
``baselines_input_agnostic`` keys. 0-GPU (VM CPU); reuses cached pass-B tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy / the heavy issue779_* siblings freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402


def _fit_scale(Xtr, Ytr):
    """Best scalar a minimizing ||Ytr - a*Xtr||_F -> a = <X,Y>/<X,X> (train)."""
    return float((Xtr * Ytr).sum() / ((Xtr * Xtr).sum() + 1e-12))


def _fit_diag(Xtr, Ytr):
    """Best per-dim diagonal d minimizing sum_j ||Y_j - d_j X_j|| (train)."""
    num = (Xtr * Ytr).sum(0)
    den = (Xtr * Xtr).sum(0) + 1e-12
    return num / den


def _baselines_for(Xtr, Ytr, Xte, Yte, n_boot, seed):
    out = {}
    # identity family (input-dependent): raw copy / scaled / diagonal
    out["identity_copy"] = F._bootstrap_recon_ci(Xte, Yte, n_boot, seed)
    a = _fit_scale(Xtr, Ytr)
    out["scaled_identity"] = F._bootstrap_recon_ci(a * Xte, Yte, n_boot, seed + 1)
    d = _fit_diag(Xtr, Ytr)
    out["diagonal_only"] = F._bootstrap_recon_ci(Xte * d, Yte, n_boot, seed + 2)
    return out


def _shuffled_ridge(bundle, li, variant, train, test, dev, n_perm, seed):
    """Sanity floor: fit ridge on ROW-PERMUTED (X_train, Y_train) pairs, eval on
    the honest test set. Mean R2 over ``n_perm`` permutations (val lambda grid)."""
    X = F.input_layer(bundle, variant, li)
    Y = F.target_vx(bundle, li)
    Xtr, Ytr, Xte, Yte = X[train], Y[train], X[test], Y[test]
    rng = np.random.default_rng(seed)
    r2s = []
    for _ in range(n_perm):
        perm = rng.permutation(len(Ytr))
        (pred_te,), _ = F.gram_fit_apply(Xtr, Ytr[perm], [Xte], dev, val=None)
        r2s.append(PR._pooled_r2(pred_te, Yte))
    return {"point": float(np.mean(r2s)), "sd": float(np.std(r2s)), "n_perm": n_perm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=F.DEFAULT_OUT_DIR)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=F.SPLIT_SEED)
    ap.add_argument("--n-train", type=int, default=3600)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    path = args.out_dir / "fair_comparison.json"
    res = json.loads(path.read_text())
    dev = F._dev(args.device)
    bundle = F.load_pass_b()
    n_ctx = bundle["cx_last"].shape[0]
    train, _val, test = F.fixed_split(n_ctx, args.n_train, args.n_val, args.n_test, args.seed)

    for variant in [v for v in F.INPUT_VARIANTS if v in res.get("inputs", {})]:
        li = res["inputs"][variant]["ridge"]["val_selected_layer"]
        X = F.input_layer(bundle, variant, li)
        Y = F.target_vx(bundle, li)
        bl = _baselines_for(X[train], Y[train], X[test], Y[test], args.n_boot, args.seed)
        bl["shuffled_pairing"] = _shuffled_ridge(
            bundle, li, variant, train, test, dev, args.n_perm, args.seed
        )
        bl["_read_out_layer"] = int(li)
        res["inputs"][variant]["baselines"] = bl
        print(
            f"[{variant} L{li}] identity_copy={bl['identity_copy']['r2']['point']:.3f} "
            f"scaled={bl['scaled_identity']['r2']['point']:.3f} "
            f"diag={bl['diagonal_only']['r2']['point']:.3f} "
            f"shuffled={bl['shuffled_pairing']['point']:.3f}"
        )

    # predict-the-mean is input-agnostic (uses no X): train-mean answer at the
    # 'last' variant's read-out layer (any layer's v_x mean gives ~0 by construction).
    li0 = res["inputs"]["last"]["ridge"]["val_selected_layer"]
    Y = F.target_vx(bundle, li0)
    ybar = Y[train].mean(0, keepdims=True).repeat(len(test), 0)
    ptm = F._bootstrap_recon_ci(ybar, Y[test], args.n_boot, args.seed + 9)
    res["baselines_input_agnostic"] = {"predict_the_mean": ptm, "_read_out_layer": int(li0)}
    print(f"[input-agnostic L{li0}] predict_the_mean={ptm['r2']['point']:.3f}")

    F.C.write_json_atomic(path, res)
    print(f"merged baselines into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
