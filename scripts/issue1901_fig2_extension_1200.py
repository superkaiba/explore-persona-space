#!/usr/bin/env python3
"""Figure 2B extension rung at 1,200 training contexts + copy-context baselines.

Fits the linear (ridge) and nonlinear (MLP, the banked recipe) predictors on
1,200 seeded pass_b single-turn rows and scores them exactly as
``issue1901_figure2_five_rollout_scaling`` scores the banked rungs: pooled R^2
against five-rollout mean targets, strict top-1 under whitened cosine +
two-sided CSLS (K=10) on the deduplicated 942-answer pool, whitening = the
banked 963,444-row training-answer statistics. Also scores W = identity + bias
(bias = mean(v_A - v_C) over the pass_b training rows) and W = identity under
the same convention. CPU only; several seeds for the 1,200-row subsample.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1901_figure2_five_rollout_scaling as FIVE  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402
import issue1901_paper_densify_mlp as PDM  # noqa: E402

MLP_LR = getattr(PDM, "MLP_LR", 3e-4)
MLP_WIDTH = getattr(PDM, "MLP_WIDTH", 8192)
MLP_MAX_EPOCHS = getattr(F79, "MLP_MAX_EPOCHS", 300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_results/issue_1901/figure2_extension_1200.json")
    ap.add_argument("--ridge-block", type=int, default=50_000)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    stage = PROJECT_ROOT / "data/issue_1901/figure2_five_rollout_scaling"
    paths = {k: stage / v for k, v in FIVE.BASE_FILES.items()}
    bundle = F79.load_pass_b(paths["pass_b"])
    n = int(bundle["cx_last"].shape[0])
    tr_all, va_all, te_all = F79.fixed_split(n, n - 400 - FIVE.N_TEST, 400, FIVE.N_TEST, F79.SPLIT_SEED)
    Xn = np.asarray(F79.input_layer(bundle, "last", FIVE.LAYER), dtype=np.float32)
    Yn = np.asarray(F79.target_vx(bundle, FIVE.LAYER), dtype=np.float32)
    del bundle
    source_target, target_mean, test_rows = FIVE._load_five_rollout_target(paths["pass_b"], paths["test_draws"])
    assert np.array_equal(test_rows, np.asarray(te_all))
    view = FINAL.make_eval_view(source_target, FIVE.N_TEST, "keep_one")
    assert view.diagnostics["realized_n_pool"] == 942, view.diagnostics
    whiten, whitening = FINAL._whitener(paths["whiten"])
    X = torch.as_tensor(Xn).to(torch.bfloat16)
    Y = torch.as_tensor(Yn)
    te = np.asarray(te_all)
    dev = torch.device("cpu")
    print(f"[setup] {time.time()-t0:.0f}s: pass_b loaded, pool 942", flush=True)

    def score(pred, seed):
        r2, cos = F79._recon_point(pred, target_mean)
        ret = FINAL.score_cell(pred, target_mean, view, whiten, seed=seed)["whiten_csls"]["strict"]
        return {"r2": float(r2), "mean_cosine": float(cos), "top1": float(ret["acc_at_k"]["1"]),
                "top5": float(ret["acc_at_k"]["5"]), "top1_ci95": ret.get("acc1_ci95")}

    # baselines (bias from ALL pass_b training rows; n-insensitive)
    tr_full = np.asarray(tr_all)
    bias = (Yn[tr_full].astype(np.float64) - Xn[tr_full].astype(np.float64)).mean(0)
    pred_ib = Xn[te].astype(np.float64) + bias
    pred_copy = Xn[te].astype(np.float64)
    baselines = {"identity_bias": {**score(pred_ib, 190_701), "n_bias_rows": int(len(tr_full))},
                 "identity_copy": score(pred_copy, 190_702)}
    for k, v in baselines.items():
        print(f"[baseline] {k}: R2={v['r2']:.4f} top1={v['top1']:.3f}", flush=True)

    per_seed = []
    for s in range(args.seeds):
        rng = np.random.default_rng(190_800 + s)
        tr = np.sort(rng.choice(tr_full, size=args.n_train, replace=False))
        va = np.sort(rng.choice(np.asarray(va_all), size=160, replace=False))
        pred_r, meta_r, _ = N1M.fit_ridge_with_weights(X, Y, tr, va, te, N1M.LAMBDAS_N1M if args.n_train > 50_000 else np.logspace(-3, 7, 21), dev, args.ridge_block)
        row = {"seed": s, "n_train": args.n_train, "ridge": {**score(pred_r, 190_900 + s), "selected_lambda": meta_r.get("selected_lambda")}}
        print(f"[seed {s}] ridge R2={row['ridge']['r2']:.4f} top1={row['ridge']['top1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if not args.skip_mlp:
            pred_m, meta_m = N1M.fit_mlp(X, Y, tr, te, MLP_WIDTH, MLP_LR, MLP_MAX_EPOCHS, N1M.MLP_BATCH, 190_950 + s, dev)
            row["mlp"] = {**score(np.asarray(pred_m, dtype=np.float64), 191_000 + s), "epochs_ran": meta_m.get("epochs_ran"), "width": MLP_WIDTH, "lr": MLP_LR}
            print(f"[seed {s}] mlp   R2={row['mlp']['r2']:.4f} top1={row['mlp']['top1']:.3f} epochs={meta_m.get('epochs_ran')} ({time.time()-t0:.0f}s)", flush=True)
        per_seed.append(row)

    def agg(pred_key):
        rows = [r[pred_key] for r in per_seed if pred_key in r]
        return {"r2": float(np.mean([r["r2"] for r in rows])), "top1": float(np.mean([r["top1"] for r in rows])),
                "r2_range": [float(min(r["r2"] for r in rows)), float(max(r["r2"] for r in rows))],
                "top1_range": [float(min(r["top1"] for r in rows)), float(max(r["top1"] for r in rows))], "n_seeds": len(rows)}
    per_n = {str(args.n_train): {k: agg(k) for k in ("ridge", "mlp") if any(k in r for r in per_seed)}}
    payload = {"issue": 1901, "analysis": "figure2-extension-1200", "layer": FIVE.LAYER,
               "convention": "identical to figure2_five_rollout_scaling: five-rollout mean targets, whitened cosine + two-sided CSLS (K=10), 942-answer pool, banked 963,444-row whitening",
               "training_rows": "seeded subsamples of the pass_b single-turn training rows (single-draw targets)",
               "mlp_recipe": {"width": MLP_WIDTH, "lr": MLP_LR, "max_epochs": MLP_MAX_EPOCHS, "batch": N1M.MLP_BATCH},
               "whitening": whitening, "per_n": per_n, "per_seed": per_seed, "baselines": baselines,
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out, f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
