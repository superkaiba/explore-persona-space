#!/usr/bin/env python3
"""Issue #779 Stage-2 (deferred R3 sweep): set-to-set CKA + coefficient subspace overlap.

0-GPU free re-analysis on already-persisted tensors (user-chat inline carve-out,
follow-up label ``stage2-cka-subspace-overlap``). Two legs + guardrails:

LEG 1 — set-to-set linear CKA per layer between the context-state set {c_x} and the
  answer-profile set {v(x)} over the 5000 LMSYS pass-B pairs (all 28 layers).
  Centered feature-space linear CKA (Kornblith et al. 2019):
    CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F),
  X_c/Y_c column-centered over the N examples. Reported alongside the committed
  held-out reconstruction R2 curve (percontext_recon.json) — does representational
  similarity track map fidelity?

LEG 2 — subspace overlap of the fitted per-example map h: c_x -> v(x). Per layer:
    (a) SVD the ridge coefficient matrix W_std (standardized-input -> centered-output,
        stage-1 recipe: standardize X on train stats, center Y, GCV lambda over
        logspace(-2,4,13), Gram-dual solve). Its OUTPUT singular subspace lives in raw
        v-activation space (output-side is only centered, a shift). Quantify how much of
        each trait direction r_B[layer] mass lies OUTSIDE h's top-k output singular
        subspace — captured(k) = ||V_out[:, :k]^T r_B||^2 / ||r_B||^2 — vs k. Tests the
        body's hypothesis that h reconstructs the generic profile near-perfectly while
        the trait read trails raw projection because r_B is poorly recovered by a
        variance-driven fit. Robustness twin: the same capture via the SVD of the
        map's PREDICTIONS Yhat = X_n @ W_std (gauge-invariant to input standardization).
    (b) The AVERAGED map from the #722/#658 line: refit c_C -> v0(C) ridge (same
        stage-1 recipe) on the persisted #658 store (50-context grid: #594
        last-input-token c_C + #658 v0 mean summaries). Overlap its coefficient
        subspaces against h's per layer — output subspace (raw v-space, clean) +
        input subspace (raw residual-stream coords, un-standardized) + a functional
        cross-map prediction agreement (CKA + mean cosine on the 50 grid contexts).
        Is the averaged map the per-example map seen through probe/sampling averaging,
        or a genuinely different map?

LEG 3 — GUARDRAILS (carried verbatim into the JSON + the summary): the cross-map
  comparison in 2(b) is CORPUS-CONFOUNDED (5000 real LMSYS single prompts vs the
  50-context x 48-probe curated grid). High overlap supports same-function-plus-noise;
  low overlap is AMBIGUOUS (function difference vs corpus difference) and motivates a
  same-grid per-(context,probe) refit as a SEPARATE needs-gpu follow-up (NOT scoped
  here). EXCLUDED: the per-position-decay leg (#825 owns it).

Reuses the stage-1 ridge recipe (fit_h.ridge_fit_predict / the Gram-dual GCV twin) and
the committed layer/hidden conventions. Fail loud — no try/except:pass, no dummy fills.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_stage2")

N_LAYERS = 28
HIDDEN = 3584
TRAITS = ("evil", "sycophancy", "hallucination")
READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
LAMBDAS = np.logspace(-2, 4, 13)  # stage-1 GCV grid (fit_h.ridge_fit_predict)
K_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # top-k output-subspace cut points

# HF provenance of the inputs (documented in the output metadata).
HF_REVISIONS = {
    "pass_b (issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt)": "037fcbb",
    "r_b (issue779_monitoring/r_b/<trait>.pt)": "037fcbb",
    "v0 (issue658_theory_assumptions/store/v0_summaries.pt)": "b33429f",
    "c_C (issue594_context_geometry/analysis_tensors/context_vectors_mean.pt)": "main",
}

LEG3_CAVEAT = (
    "The cross-map comparison in leg 2(b) is CORPUS-CONFOUNDED: the per-example map h is "
    "fit on 5000 real single-prompt LMSYS contexts (pass B), while the averaged map is fit "
    "on the #658/#722 50-context x 48-probe curated grid (probe-averaged c_C + v0 mean "
    "summaries). High coefficient-subspace overlap supports 'same linear function plus "
    "sampling/probe-averaging noise'; LOW overlap is AMBIGUOUS between a genuine function "
    "difference and a mere corpus/distribution difference. Disambiguating requires a "
    "same-grid per-(context,probe) refit of h on the curated grid (a SEPARATE needs-gpu "
    "re-extraction follow-up; may piggyback on #810 Phase B), which is NOT scoped into this "
    "0-GPU round. The per-position-decay leg of the deferred sweep is EXCLUDED here (#825 "
    "owns that read)."
)


def _repro_metadata(extra: dict | None = None) -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    meta = {
        "git_commit": sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "script": "issue779_stage2_cka_subspace",
        "hf_input_revisions": HF_REVISIONS,
        "layer_indexing": "0..27, aligned across pass_b / r_b / #658 store / #594 c_C store",
        "ridge_recipe": (
            "standardize X on train stats (+1e-9), center Y on train mean, GCV lambda over "
            "np.logspace(-2,4,13), Gram-dual (eigh) solve — matches fit_h.ridge_fit_predict"
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# ── LEG 1: linear CKA ─────────────────────────────────────────────────────────


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centered feature-space linear CKA between two (N, D) representations (float64)."""
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    xty = Xc.T @ Yc  # (Dx, Dy)
    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    hsic_xy = float((xty**2).sum())
    norm_x = float((xtx**2).sum()) ** 0.5
    norm_y = float((yty**2).sum()) ** 0.5
    denom = norm_x * norm_y
    return float("nan") if denom < 1e-12 else hsic_xy / denom


# ── ridge (Gram-dual GCV) returning the coefficient matrix + input std ─────────


def _ridge_coeff_matrix(X: np.ndarray, Y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit ridge (stage-1 recipe) and return (W_std, xsd, best_lam).

    W_std maps standardized-input -> centered-output (the fitted ridge weights in the
    recipe's native standardized gauge): yc = ((x - xmu)/xsd) @ W_std. Returned with
    xsd so callers can un-standardize to raw input coords (W_raw = diag(1/xsd) @ W_std).
    Gram-dual solve + GCV lambda in eigen-coefficient space (== fit_h.ridge_fit_predict).
    """
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64)
    xmu = Xt.mean(0)
    xsd = Xt.std(0) + 1e-9
    Xn = (Xt - xmu) / xsd
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    ntr = Xn.shape[0]
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Yc
    sqVtY = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    best_lam = float(LAMBDAS[0])
    best_gcv = float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    alpha = V @ ((1.0 / (w + best_lam)).unsqueeze(1) * VtY)  # (N, Dout)
    W_std = Xn.T @ alpha  # (Din, Dout)
    return W_std, xsd, best_lam


def _captured_fraction(basis: torch.Tensor, vec: torch.Tensor, ks: list[int]) -> dict[int, float]:
    """captured(k) = ||basis[:, :k]^T vec||^2 / ||vec||^2 for each k (basis cols orthonormal)."""
    v = vec / (vec.norm() + 1e-12)
    proj = basis.T @ v  # (r,)
    csum = torch.cumsum(proj**2, dim=0)
    r = basis.shape[1]
    out = {}
    for k in ks:
        kk = min(k, r)
        out[k] = float(csum[kk - 1])
    return out


def _subspace_overlap(U1: torch.Tensor, U2: torch.Tensor, ks: list[int]) -> dict[int, float]:
    """Mean squared cosine of principal angles between top-k subspaces: ||U1k^T U2k||_F^2 / k."""
    out = {}
    r1, r2 = U1.shape[1], U2.shape[1]
    for k in ks:
        k1, k2 = min(k, r1), min(k, r2)
        M = U1[:, :k1].T @ U2[:, :k2]  # (k1, k2)
        out[k] = float((M**2).sum() / min(k1, k2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 Stage-2 CKA + subspace overlap.")
    dl = PROJECT_ROOT / "data" / "issue_779" / "hf_dl" / "inline_cka"
    ap.add_argument(
        "--pass-b",
        type=Path,
        default=dl / "037fcbb/issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt",
    )
    ap.add_argument("--rb-dir", type=Path, default=dl / "037fcbb/issue779_monitoring/r_b")
    ap.add_argument(
        "--v0", type=Path, default=dl / "b33429f/issue658_theory_assumptions/store/v0_summaries.pt"
    )
    ap.add_argument(
        "--cc",
        type=Path,
        default=dl / "main/issue594_context_geometry/analysis_tensors/context_vectors_mean.pt",
    )
    ap.add_argument(
        "--recon", type=Path, default=PROJECT_ROOT / "eval_results/issue_779/percontext_recon.json"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_779/stage2_cka_subspace"
    )
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="load an existing combined JSON and SKIP layers already computed",
    )
    args = ap.parse_args()

    torch.set_num_threads(int(args.n_threads))
    combined_path = args.out_dir / "stage2_cka_subspace.json"

    logger.info("Loading pass-B bundle %s", args.pass_b)
    pb = torch.load(args.pass_b, weights_only=False)
    # Keep pass-B fp32 in memory and slice per-layer to fp64 on demand (a whole-tensor
    # fp64 copy of cx+vx is ~8 GB and tripped a SIGTERM under fleet memory pressure).
    pb.pop("cx_mean", None)  # unused here; free ~2 GB
    cx = pb["cx_last"]  # (5000, 28, 3584) fp32
    vx = pb["v_x"]
    layers = list(pb["layers"])
    assert cx.shape[1:] == (N_LAYERS, HIDDEN), cx.shape
    assert layers == list(range(N_LAYERS)), layers
    n_ctx_lmsys = int(cx.shape[0])

    rb = {}
    for t in TRAITS:
        blob = torch.load(args.rb_dir / f"{t}.pt", weights_only=False)
        arr = blob["r_b"].to(torch.float64)
        assert arr.shape == (N_LAYERS, HIDDEN), (t, arr.shape)
        rb[t] = arr

    # Averaged-map inputs (50-context grid): #658 v0 mean summaries + #594 c_C (last).
    v0 = torch.load(args.v0, weights_only=False)
    ctx_ids = list(v0["context_ids"])
    assert list(v0["capture_layers"]) == list(range(N_LAYERS)), v0["capture_layers"]
    cc = torch.load(args.cc, weights_only=False)
    iid_to_row = {iid: i for i, iid in enumerate(cc["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    assert not missing, f"#594 c_C store missing contexts: {missing[:5]}"
    assert cc["probe_pool_hash"] == v0["probe_pool_hash"], "c_C / v0 probe-pool-hash drift"
    # c_C[layer] (N50, 28, H); v0mean[ctx] (28, H)
    cc_tensor = cc["tensor"].to(torch.float64)  # (50, 28, H) keyed by instance_ids
    cC = torch.stack([cc_tensor[iid_to_row[c]] for c in ctx_ids])  # (50, 28, H) aligned to ctx_ids
    v0mean = torch.stack(
        [v0["summaries"]["mean"][c].to(torch.float64) for c in ctx_ids]
    )  # (50,28,H)
    n_ctx_grid = len(ctx_ids)

    with open(args.recon) as f:
        recon = json.load(f)
    recon_r2 = recon["read1_heldout_recon"]["heldout_r2_vs_layer"]  # {"0":{mean,sd,...}}

    results: dict = {
        "metadata": _repro_metadata(
            {
                "n_ctx_lmsys_pass_b": n_ctx_lmsys,
                "n_ctx_grid_averaged_map": n_ctx_grid,
                "grid_context_ids": ctx_ids,
                "k_grid": K_GRID,
                "read_out_layers": READ_OUT_LAYER,
                "random_baseline_note": (
                    "a random unit output direction has expected captured(k)=k/H and expected "
                    "subspace overlap k/H (H=3584); values >> k/H indicate real alignment"
                ),
            }
        ),
        "leg3_caveat": LEG3_CAVEAT,
        "leg1_cka_cx_vx_by_layer": {},
        "recon_heldout_r2_by_layer": {str(li): recon_r2[str(li)]["mean"] for li in range(N_LAYERS)},
        "leg2_by_layer": {},
    }

    if args.resume and combined_path.exists():
        with open(combined_path) as f:
            prior = json.load(f)
        results["leg1_cka_cx_vx_by_layer"] = prior.get("leg1_cka_cx_vx_by_layer", {})
        results["leg2_by_layer"] = prior.get("leg2_by_layer", {})
        logger.info("--resume: %d layers already present", len(results["leg2_by_layer"]))

    for li in range(N_LAYERS):
        if args.resume and str(li) in results["leg2_by_layer"]:
            continue
        t_layer = time.time()
        Xpe = cx[:, li, :].to(torch.float64)  # (5000, H)
        Ype = vx[:, li, :].to(torch.float64)

        # LEG 1 — set-to-set CKA(c_x, v_x) at this layer.
        cka = linear_cka(Xpe, Ype)
        results["leg1_cka_cx_vx_by_layer"][str(li)] = cka

        # LEG 2 — per-example map W (standardized coeff matrix) + truncated SVD.
        # Truncated (svd_lowrank q=600) is exact for the top singular triplets needed by
        # the K_GRID (max 500) reads; the energy DENOMINATOR is the exact ||W||_F^2.
        W_std, xsd_pe, lam_pe = _ridge_coeff_matrix(Xpe.numpy(), Ype.numpy())
        q_out = 600
        _U_pe, S_pe, Vout_pe = torch.svd_lowrank(W_std, q=q_out, niter=4)  # Vout_pe (Dout, q)
        w_fro2 = float((W_std**2).sum())  # exact total output energy
        # raw-input left singular vectors (un-standardize: W_raw = diag(1/xsd) @ W_std)
        W_raw_pe = W_std / xsd_pe.unsqueeze(1)
        Uin_pe_raw, _, _ = torch.svd_lowrank(W_raw_pe, q=120, niter=4)  # (Din, q) top input dirs
        # gauge-invariant prediction-SVD output subspace (Yhat = Xn @ W_std)
        Xn_pe = (Xpe - Xpe.mean(0)) / xsd_pe
        Yhat_pe = Xn_pe @ W_std  # (5000, Dout)
        _, _, Vpred_pe = torch.svd_lowrank(Yhat_pe - Yhat_pe.mean(0), q=q_out, niter=4)  # (Dout, q)

        sv = S_pe
        cume = torch.cumsum(sv**2, 0) / w_fro2  # fraction of exact ||W||_F^2 in top-k
        layer_out: dict = {
            "cka_cx_vx": cka,
            "ridge_lambda_perexample": lam_pe,
            "W_frobenius_sq": w_fro2,
            "W_singular_value_top10": [float(x) for x in sv[:10]],
            "W_output_energy_cumfrac": {str(k): float(cume[min(k, len(cume)) - 1]) for k in K_GRID},
            "rb_capture": {},
            "avg_map": {},
        }

        # LEG 2(a) — r_B mass outside top-k output subspace, per trait at this layer.
        for t in TRAITS:
            rvec = rb[t][li]
            cap_coeff = _captured_fraction(Vout_pe, rvec, K_GRID)
            cap_pred = _captured_fraction(Vpred_pe, rvec, K_GRID)
            layer_out["rb_capture"][t] = {
                "captured_frac_by_k_coeffSVD": {str(k): cap_coeff[k] for k in K_GRID},
                "mass_outside_by_k_coeffSVD": {str(k): 1.0 - cap_coeff[k] for k in K_GRID},
                "captured_frac_by_k_predSVD": {str(k): cap_pred[k] for k in K_GRID},
                "is_read_out_layer": (READ_OUT_LAYER[t] == li),
            }

        # LEG 2(b) — averaged map W_avg + subspace overlap vs per-example map.
        Xavg = cC[:, li, :]  # (50, H)
        Yavg = v0mean[:, li, :]
        W_avg, xsd_avg, lam_avg = _ridge_coeff_matrix(Xavg.numpy(), Yavg.numpy())
        q_avg = min(60, n_ctx_grid)  # W_avg rank <= n_ctx_grid (50)
        _U_avg, S_avg, Vout_avg = torch.svd_lowrank(W_avg, q=q_avg, niter=6)  # Vout_avg (Dout, q)
        W_raw_avg = W_avg / xsd_avg.unsqueeze(1)
        Uin_avg_raw, _, _ = torch.svd_lowrank(W_raw_avg, q=q_avg, niter=6)  # (Din, q)
        r_avg = int((S_avg > 1e-8 * S_avg[0]).sum())
        ks_avg = [k for k in K_GRID if k <= 50]

        out_overlap = _subspace_overlap(Vout_pe, Vout_avg, ks_avg)  # output (raw v-space)
        in_overlap = _subspace_overlap(Uin_pe_raw, Uin_avg_raw, ks_avg)  # input (raw coords)
        rand_base = {str(k): k / HIDDEN for k in ks_avg}

        # functional cross-map agreement on the 50 grid contexts (corpus-confounded).
        pred_pe_grid = ((Xavg - Xpe.mean(0)) / xsd_pe) @ W_std + Ype.mean(
            0
        )  # per-example map on grid
        pred_avg_grid = ((Xavg - Xavg.mean(0)) / xsd_avg) @ W_avg + Yavg.mean(0)
        func_cka = linear_cka(pred_pe_grid, pred_avg_grid)
        num = (pred_pe_grid * pred_avg_grid).sum(1)
        den = pred_pe_grid.norm(dim=1) * pred_avg_grid.norm(dim=1) + 1e-12
        func_cos_mean = float((num / den).mean())

        layer_out["avg_map"] = {
            "ridge_lambda_avg": lam_avg,
            "rank_W_avg": r_avg,
            "output_subspace_overlap_by_k": {str(k): out_overlap[k] for k in ks_avg},
            "input_subspace_overlap_raw_by_k": {str(k): in_overlap[k] for k in ks_avg},
            "random_baseline_overlap_by_k": rand_base,
            "functional_agreement_on_grid": {
                "linear_cka": func_cka,
                "mean_per_context_cosine": func_cos_mean,
                "n_contexts": n_ctx_grid,
                "corpus_confounded": True,
            },
        }

        results["leg2_by_layer"][str(li)] = layer_out
        _write_json(combined_path, results)  # checkpoint per layer
        logger.info(
            "layer %2d done in %.1fs: CKA(cx,vx)=%.3f  rb@k100(evil)=%.3f  outOverlap@50=%.3f",
            li,
            time.time() - t_layer,
            cka,
            layer_out["rb_capture"]["evil"]["captured_frac_by_k_coeffSVD"]["100"],
            out_overlap[50],
        )

    # ── split into the three named artifact files (spec) ──
    leg1 = {
        "metadata": results["metadata"],
        "leg1_cka_cx_vx_by_layer": results["leg1_cka_cx_vx_by_layer"],
        "recon_heldout_r2_by_layer": results["recon_heldout_r2_by_layer"],
        "note": "LEG 1 set-to-set linear CKA(c_x, v_x) per layer, alongside held-out recon R2.",
    }
    subspace = {
        "metadata": results["metadata"],
        "leg3_caveat": LEG3_CAVEAT,
        "rb_capture_and_W_spectrum_by_layer": {
            li: {
                "cka_cx_vx": results["leg2_by_layer"][li]["cka_cx_vx"],
                "W_singular_value_top10": results["leg2_by_layer"][li]["W_singular_value_top10"],
                "W_output_energy_cumfrac": results["leg2_by_layer"][li]["W_output_energy_cumfrac"],
                "rb_capture": results["leg2_by_layer"][li]["rb_capture"],
            }
            for li in results["leg2_by_layer"]
        },
    }
    avgmap = {
        "metadata": results["metadata"],
        "leg3_caveat": LEG3_CAVEAT,
        "avg_map_overlap_by_layer": {
            li: results["leg2_by_layer"][li]["avg_map"] for li in results["leg2_by_layer"]
        },
    }
    _write_json(args.out_dir / "cka_by_layer.json", leg1)
    _write_json(args.out_dir / "subspace_overlap.json", subspace)
    _write_json(args.out_dir / "averaged_map_comparison.json", avgmap)
    logger.info("Wrote %s + 3 split artifact files", combined_path)

    # ── console headline ──
    print("\n===== LEG 1: CKA(c_x, v_x) vs held-out recon R2 (per layer) =====")
    for li in (0, 7, 14, 17, 21, 26, 27):
        print(
            f"  L{li:2d}: CKA={results['leg1_cka_cx_vx_by_layer'][str(li)]:.3f}  "
            f"reconR2={results['recon_heldout_r2_by_layer'][str(li)]:.3f}"
        )
    print("\n===== LEG 2(a): r_B mass captured by top-k output subspace (read-out layer) =====")
    for t in TRAITS:
        li = READ_OUT_LAYER[t]
        cap = results["leg2_by_layer"][str(li)]["rb_capture"][t]["captured_frac_by_k_coeffSVD"]
        eng = results["leg2_by_layer"][str(li)]["W_output_energy_cumfrac"]
        print(
            f"  {t:14s} L{li:2d}: rB captured k1/10/100 = "
            f"{cap['1']:.3f}/{cap['10']:.3f}/{cap['100']:.3f}  | "
            f"W output-energy k1/10/100 = {eng['1']:.3f}/{eng['10']:.3f}/{eng['100']:.3f}"
        )
    print("\n===== LEG 2(b): per-example vs averaged map (read-out layer) =====")
    for t in TRAITS:
        li = READ_OUT_LAYER[t]
        am = results["leg2_by_layer"][str(li)]["avg_map"]
        oo = am["output_subspace_overlap_by_k"]
        io = am["input_subspace_overlap_raw_by_k"]
        fa = am["functional_agreement_on_grid"]
        print(
            f"  L{li:2d} ({t}): out-overlap k10/50 = {oo['10']:.3f}/{oo['50']:.3f} "
            f"(rand@50={50 / HIDDEN:.4f}) | in-overlap(raw) k10/50 = {io['10']:.3f}/{io['50']:.3f} "
            f"| func CKA={fa['linear_cka']:.3f} cos={fa['mean_per_context_cosine']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
