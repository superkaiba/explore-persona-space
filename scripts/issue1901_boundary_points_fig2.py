#!/usr/bin/env python3
"""Per-boundary-token control points for Figure 2B: held-out R^2 + strict top-1.

Refits the four exact-token maps of ``issue1901_individual_boundary_tokens``
(layer 19, 1,200 training pairs each, span = anchor+1 .. next sentence-final
punctuation token) from the recovered capture store, gates the R^2 against the
banked result, then scores retrieval with the Figure 2B convention: whitened
cosine + two-sided CSLS (K=10), strict top-1, whitening fit on the arm's own
training spans (diagonal-target shrinkage lambda=0.1). Pool = the arm's 400
held-out spans. CPU only.
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
from scipy.linalg import solve_triangular  # noqa: E402

import issue779_fitter_fair_comparison as F79  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_common as C  # noqa: E402
import issue1901_figure2_five_rollout_scaling as FIVE  # noqa: E402
import issue1901_boundary_token_control as PARENT  # noqa: E402
import issue1901_individual_boundary_tokens as IBT  # noqa: E402
import issue1901_singleturn_retrieval_final as FINAL  # noqa: E402
from issue1901_plot1_remake import train_whitening_stats  # noqa: E402


def chat_reference(args, dev, n_train: int, n_pool: int, draws: int) -> dict:
    """Chat map under the boundary protocol: ridge on ``n_train`` pass_b rows,
    whitening from those rows' answers, five-rollout-mean test targets, pool of
    ``n_pool`` held-out answers; repeated for ``draws`` seeded subsamples."""
    stage = PROJECT_ROOT / "data/issue_1901/figure2_five_rollout_scaling"
    pass_b = stage / FIVE.BASE_FILES["pass_b"]
    bundle = F79.load_pass_b(pass_b)
    n = int(bundle["cx_last"].shape[0])
    tr_all, va_all, te_all = F79.fixed_split(n, n - 400 - FIVE.N_TEST, 400, FIVE.N_TEST, F79.SPLIT_SEED)
    Xn = F79.input_layer(bundle, "last", FIVE.LAYER)
    Yn = F79.target_vx(bundle, FIVE.LAYER)
    del bundle
    _t, target_mean, test_rows = FIVE._load_five_rollout_target(pass_b, stage / FIVE.BASE_FILES["test_draws"])
    assert np.array_equal(test_rows, np.asarray(te_all))
    X = torch.as_tensor(np.asarray(Xn, dtype=np.float32)).to(torch.bfloat16)
    Y = torch.as_tensor(np.asarray(Yn, dtype=np.float32))
    per_draw = []
    for d in range(draws):
        rng = np.random.default_rng(args.seed + 1000 + d)
        tr = np.sort(rng.choice(np.asarray(tr_all), size=n_train, replace=False))
        va = np.sort(rng.choice(np.asarray(va_all), size=160, replace=False))
        te_pick = np.sort(rng.choice(len(te_all), size=n_pool, replace=False))
        te = np.asarray(te_all)[te_pick]
        pred, fit_meta, _ = N1M.fit_ridge_with_weights(X, Y, tr, va, te, PARENT._lambdas_for(n_train), dev, args.ridge_block)
        true = target_mean[te_pick]
        r2, cos = F79._recon_point(pred, true)
        r2_single, _ = F79._recon_point(pred, PARENT._to_f64_np(Y, te))
        mu, ell = train_whitening_stats(PARENT._to_f64_np(Y, tr), dev)
        whiten = lambda x, mu=mu, ell=ell: solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True, check_finite=False).T
        view = FINAL.make_eval_view(true.astype(np.float32), len(true), "keep_one")
        cell = FINAL.score_cell(pred, true, view, whiten, seed=args.seed + 2000 + d)
        row = {"draw": d, "n_train": n_train, "n_pool_realized": view.diagnostics["realized_n_pool"],
               "r2_five_rollout_target": float(r2), "r2_single_draw_target": float(r2_single), "mean_cosine": float(cos),
               "selected_lambda": fit_meta.get("selected_lambda"),
               "retrieval": {name: {"top1": float(cell[name]["strict"]["acc_at_k"]["1"]), "top5": float(cell[name]["strict"]["acc_at_k"]["5"]),
                                    "top1_ci95": cell[name]["strict"].get("acc1_ci95")} for name in ("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean")}}
        print(f"chat ref draw {d}: R2(5-draw)={r2:.4f} R2(single)={r2_single:.4f} acc@1 wcsls={row['retrieval']['whiten_csls']['top1']:.3f} "
              f"euclid={row['retrieval']['raw_euclidean']['top1']:.3f} cos={row['retrieval']['raw_cosine']['top1']:.3f} pool={row['n_pool_realized']}", flush=True)
        per_draw.append(row)
    agg = {"n_train": n_train, "n_pool": n_pool, "draws": draws, "per_draw": per_draw,
           "r2_mean": float(np.mean([r["r2_five_rollout_target"] for r in per_draw])),
           "top1_whiten_csls_mean": float(np.mean([r["retrieval"]["whiten_csls"]["top1"] for r in per_draw])),
           "top1_raw_euclidean_mean": float(np.mean([r["retrieval"]["raw_euclidean"]["top1"] for r in per_draw])),
           "protocol": "ridge on n_train pass_b single-turn rows (single-draw targets); whitening from those rows; "
                       "test targets = five-rollout means as in Figure 2B; pool = n_pool held-out answers (keep_one dedup)"}
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True, help="recovered ibt_run dir (store/, manifest/, results/)")
    ap.add_argument("--banked", type=Path, default=PROJECT_ROOT / "eval_results/issue_1901/individual_boundary_tokens.json")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "eval_results/issue_1901/boundary_points_fig2.json")
    ap.add_argument("--ridge-block", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=190_602)
    ap.add_argument("--chat-draws", type=int, default=3)
    args = ap.parse_args()

    banked = json.loads(args.banked.read_text())
    rows, _ids, manifest_meta = IBT._load_selected(args.run_root / "manifest")
    layer = int(json.loads((args.run_root / "capture_meta.json").read_text())["layer"])
    X, Y, row_ids, article_ids = PARENT._load_layer_arrays(args.run_root / "store", layer, (layer,))
    assert len(set(row_ids)) == len(row_ids) == len(rows) == 7040, len(rows)
    row_pos = {r: i for i, r in enumerate(row_ids)}
    dev = torch.device("cpu")
    out_tokens = {}
    for k, tid in enumerate(IBT.TOKEN_IDS):
        idx = {s: IBT._indices(rows, row_pos, tid, s) for s in ("train", "val", "test")}
        pred, fit_meta, _payload = N1M.fit_ridge_with_weights(
            X, Y, idx["train"], idx["val"], idx["test"], PARENT._lambdas_for(len(idx["train"])), dev, args.ridge_block
        )
        true = PARENT._to_f64_np(Y, idx["test"])
        r2, cos = F79._recon_point(pred, true)
        banked_r2 = banked["individual"][str(tid)]["score"]["r2"]
        gate = abs(r2 - banked_r2)
        assert gate < 1e-4, (tid, r2, banked_r2)
        mu, ell = train_whitening_stats(PARENT._to_f64_np(Y, idx["train"]), dev)
        whiten = lambda x, mu=mu, ell=ell: solve_triangular(ell, (np.asarray(x, np.float64) - mu).T, lower=True, check_finite=False).T
        view = FINAL.make_eval_view(true.astype(np.float32), len(true), "keep_one")
        cell = FINAL.score_cell(pred, true, view, whiten, seed=args.seed + k)
        pick = lambda name: {
            "top1": float(cell[name]["strict"]["acc_at_k"]["1"]),
            "top5": float(cell[name]["strict"]["acc_at_k"]["5"]),
            "top1_ci95": cell[name]["strict"].get("acc1_ci95"),
        }
        spec = IBT.TOKEN_SPECS[tid]
        out_tokens[str(tid)] = {
            "token": spec["token"], "label": spec["label"], "decoded": spec["decoded"],
            "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
            "r2": float(r2), "r2_banked": float(banked_r2), "r2_gate_abs": float(gate), "mean_cosine": float(cos),
            "r2_ci95_article_boot": [banked["individual"][str(tid)]["score"]["article_bootstrap_r2"][kk] for kk in ("lo", "hi")],
            "selected_lambda": fit_meta.get("selected_lambda"),
            "retrieval": {name: pick(name) for name in ("whiten_csls", "whiten_cosine", "raw_cosine", "raw_euclidean")},
            "pool": view.diagnostics,
        }
        print(f"token {tid:>4} {spec['label']:22s} R2={r2:.4f} (banked {banked_r2:.4f}) "
              f"acc@1 wcsls={out_tokens[str(tid)]['retrieval']['whiten_csls']['top1']:.3f} "
              f"euclid={out_tokens[str(tid)]['retrieval']['raw_euclidean']['top1']:.3f} "
              f"cos={out_tokens[str(tid)]['retrieval']['raw_cosine']['top1']:.3f}", flush=True)
    payload = {
        "experiment": "issue1901_boundary_points_fig2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": banked["model"], "layer": layer,
        "span": "anchor+1 .. next sentence-final punctuation token (exclusive); 8-256 tokens kept",
        "fit": {"kind": "linear ridge", "lambda_selection": "held-out validation R2", "ridge_block": args.ridge_block},
        "retrieval": {"metric": "whitened cosine + two-sided CSLS", "csls_k": FINAL.K_CSLS,
                       "whitening": "fit on the arm's 1,200 training spans, diagonal-target shrinkage lambda=0.1",
                       "rank": "strict top-k; mid-rank ties and top ties fail top-1", "pool": "the arm's 400 held-out spans"},
        "banked_source": str(args.banked.relative_to(PROJECT_ROOT)), "run_root": str(args.run_root),
        "manifest_row_ids_sha256": manifest_meta.get("selected_row_ids_sha256"),
        "tokens": out_tokens,
        "chat_reference_same_protocol": chat_reference(args, dev, 1200, 400, args.chat_draws),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
