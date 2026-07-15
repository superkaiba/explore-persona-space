"""Map-alignment analysis for #825 (followup_label: map-alignment).

Direct test of the base<->instruct read-out map relationship. The parent's
cross-model follow-up (scripts/issue825_crossmodel_map_transfer.py) showed
rep-swap PRESERVES prediction (Xb->Yi 0.587 @L19) while map-swap FAILS
(-0.066/-0.170), plus mixed descriptive weight-space reads (per-output-dim
cosine 0.207; principal angles 0.856 vs 0.835 random; Procrustes-aligned
cosine 0.686 with NO chance reference). The claim "post-training rotates the
read-out" was never tested directly. This 0-GPU analysis-only follow-up tests
it: does a SINGLE linear (and specifically ORTHOGONAL) basis change explain the
difference between the two fitted maps M_base (Xb->Yb) and M_inst (Xi->Yi)?

Notation (per layer, row-aligned by conv_id):
  Xb/Xi = base/instruct context slot vectors; Yb/Yi = base/instruct mean-answer
  profiles. M_base: Xb->Yb, M_inst: Xi->Yi (the within-model ridge maps).
  A_ctx: Xb->Xi, A_ctx_rev: Xi->Xb, A_ans: Yb->Yi, A_ans_rev: Yi->Yb (alignment
  maps). "base" == pretrained bundle, "instruct" == instruct bundle.

Battery (frozen layers {14,18,19,26}, headline 19; K=5 folds seed 0 SHARED across
every fit via conv_id; per-fold fits on that fold's TRAIN split only -> no leakage;
fp64 solves; fold-local pooled R^2, the parent frozen_sweep convention):
  1. WIRING GATES (HALT if |delta|>0.01 vs committed crossmodel S_assistant_chat):
     within Xb->Yb / Xi->Yi vs committed S2/S1; rep-swap Xb->Yi vs 0.587 @L19.
  2. Alignment maps + each alignment's own held-out R^2 (how linearly alignable).
  3. Composition (held-out, shared folds; compose maps fitted per fold on train):
     (a) comp_repmap: M_inst o A_ctx (Xb->Yi); fraction of the rep-swap ceiling.
     (b) comp_samefn: A_ans o M_base o A_ctx_rev (Xi->Yi); fraction of the
         within-instruct ceiling -- the "same function in a different basis" test.
     Both directions (b2i and i2b) for each.
  4. ORTHOGONAL vs GENERAL LINEAR: repeat 2-3 with the A maps constrained
     orthogonal (Procrustes on train) + a single-scale scaled-orthogonal variant.
     Orthogonal ~= general linear => "rotation"; general >> orthogonal =>
     general reparameterization, not a rotation.
  5. NULLS at L19: (a) random orthogonal Q in place of each A in the composition
     (should collapse); (b) the missing chance reference for the parent's
     Procrustes-aligned cosine 0.686 -- aligned cosine of M_inst vs Q1^T M_base Q2
     over random orthogonal Q1,Q2.
  6. RESIDUAL STRUCTURE at L19: SVD spectrum of (W_inst - A_ans W_base A_ctx_rev)
     for the best A variant (raw-space operators), vs (W_inst - W_base) unaligned.

Descriptive geometry on a SINGLE seed -- no mechanism claims.

Reuses the loader (extract/load_cell/align_pair), the VERBATIM ridge core
(_prep_fold/_ridge_predict_cached/_pooled_r2/_cv_folds), fit_primal_beta, and the
constants from scripts/issue825_crossmodel_map_transfer.py. The factored cached
ridge prep/predict here is numerically identical to the parent core (validated by
a self-check at L19). The orthogonal Procrustes + composition + null helpers are
new.

CLI:
  uv run python scripts/issue825_map_alignment.py \
      --out eval_results/issue_825/map_alignment \
      --dl-dir data/issue_825/hf_dl/map_alignment [--smoke] [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

# Staging is IO-bound; prefer the Xet high-performance path (brief).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue825_crossmodel_map_transfer as cm  # noqa: E402

# Reused constants + helpers (single source of truth = the parent crossmodel script)
HF_DATA_REPO = cm.HF_DATA_REPO
HF_REV = cm.HF_REV
HF_PREFIX = cm.HF_PREFIX
FROZEN_LAYERS = cm.FROZEN_LAYERS
HEADLINE_LAYER = cm.HEADLINE_LAYER
EXPECTED_HIDDEN = cm.EXPECTED_HIDDEN
FIT_SEED = cm.FIT_SEED
N_FOLDS = cm.N_FOLDS
LAMBDAS = cm.LAMBDAS
_fit_device = cm._fit_device
_cv_folds = cm._cv_folds
_pooled_r2 = cm._pooled_r2
fit_primal_beta = cm.fit_primal_beta

STEM_INSTRUCT = "instruct_chat_s"
STEM_BASE = "pretrained_chat_s"
ROLE = "assistant"

COMMITTED_CROSSMODEL = Path("eval_results/issue_825/crossmodel_map_transfer/results.json")
GATE_TOL = 0.01


# ===========================================================================
# Factored ridge (eigh cached per source; numerically identical to the parent
# _prep_fold + _ridge_predict_cached, just split so composition can reuse the
# same Gram across the stages that share a source tensor).
# ===========================================================================
def _ridge_prep(X_train: torch.Tensor) -> dict:
    """eigh(Gram) + standardization stats for a source. X_train: (Ntr, D) fp64."""
    xmu = X_train.mean(0)
    xsd = X_train.std(0) + 1e-9
    Xn = (X_train - xmu) / xsd
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    return {"w": w, "V": V, "Xn": Xn, "xmu": xmu, "xsd": xsd, "ntr": int(X_train.shape[0])}


def _ridge_predict(prep: dict, Y_train: torch.Tensor, X_eval: torch.Tensor) -> torch.Tensor:
    """GCV-ridge prediction at X_eval from a cached source prep + train targets."""
    w, V, Xn, xmu, xsd, ntr = (
        prep["w"],
        prep["V"],
        prep["Xn"],
        prep["xmu"],
        prep["xsd"],
        prep["ntr"],
    )
    ymu = Y_train.mean(0)
    Ytr_c = Y_train - ymu
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    Xev_n = (X_eval - xmu) / xsd
    Kev = Xev_n @ Xn.T
    KevV = Kev @ V
    filt = 1.0 / (w + best_lam)
    return (KevV * filt) @ VtY + ymu


# ===========================================================================
# Orthogonal Procrustes alignment (centered; forward + reverse via transpose;
# optional single scale). Fit on TRAIN, apply to eval.
# ===========================================================================
def _orth_fit(A_train: torch.Tensor, B_train: torch.Tensor) -> dict:
    """R minimizing ||(A-Amu)R - (B-Bmu)|| (A->B rotation) + optimal scales both
    directions. Reverse rotation is R^T. A_train,B_train: (Ntr, D) fp64."""
    Amu = A_train.mean(0)
    Bmu = B_train.mean(0)
    Ac = A_train - Amu
    Bc = B_train - Bmu
    M = Ac.T @ Bc  # (D, D)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh  # Ac R ~ Bc
    ssum = float(S.sum())
    s_fwd = ssum / (float((Ac**2).sum()) + 1e-12)  # scale for A->B
    s_rev = ssum / (float((Bc**2).sum()) + 1e-12)  # scale for B->A
    return {"R": R, "Amu": Amu, "Bmu": Bmu, "s_fwd": s_fwd, "s_rev": s_rev}


def _orth_predict(fit: dict, X_eval: torch.Tensor, *, reverse: bool, scale: bool) -> torch.Tensor:
    """Apply the fitted orthogonal (optionally scaled) map to X_eval."""
    if reverse:  # B -> A : (X - Bmu) R^T + Amu
        core = (X_eval - fit["Bmu"]) @ fit["R"].T
        s = fit["s_rev"] if scale else 1.0
        return s * core + fit["Amu"]
    core = (X_eval - fit["Amu"]) @ fit["R"]  # A -> B : (X - Amu) R + Bmu
    s = fit["s_fwd"] if scale else 1.0
    return s * core + fit["Bmu"]


# ===========================================================================
# Fold-local pooled R^2 accumulator (mirror of cm.frozen_sweep: per-fold local
# mean for ss_tot; accumulate over folds). preds_fn(tr_idx, te_idx) -> pred_test.
# ===========================================================================
def _heldout_pooled_r2(true_full: torch.Tensor, folds: np.ndarray, preds_fn) -> float:
    ss_res = 0.0
    ss_tot = 0.0
    for k in range(N_FOLDS):
        te = folds == k
        tr = ~te
        if te.sum() == 0 or tr.sum() < 3:
            continue
        pred = preds_fn(tr, te)
        true = true_full[te]
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - true.mean(0)) ** 2).sum())
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ===========================================================================
# Load a bundle pair into per-layer fp64 tensors on the fit device.
# ===========================================================================
def _load_pair(npz_i: Path, npz_b: Path, layers_subset):
    ci = cm.load_cell(npz_i, ROLE)
    cb = cm.load_cell(npz_b, ROLE)
    al = cm.align_pair(ci, cb)
    ia, ib = al["ia"], al["ib"]
    stored_layers = ci["layers"]
    keep_li = [li for li, L in enumerate(stored_layers) if L in layers_subset]
    layers = [stored_layers[li] for li in keep_li]
    dev = _fit_device()

    def _mk(arr, idx):
        # arr (N, Lf, D) -> dict{layer: (n_common, D) fp64 on device}
        sub = arr[idx][:, keep_li, :]
        return {
            int(layers[j]): torch.as_tensor(sub[:, j, :], dtype=torch.float64).to(dev)
            for j in range(len(layers))
        }

    data = {
        "Xi": _mk(ci["X"], ia),
        "Yi": _mk(ci["Y"], ia),
        "Xb": _mk(cb["X"], ib),
        "Yb": _mk(cb["Y"], ib),
    }
    conv = al["common"]
    return data, conv, [int(x) for x in layers], al


# ===========================================================================
# Per-layer held-out battery (ceilings + alignment R^2 + compositions).
# ===========================================================================
def _fold_reads(preps, orth, tens, tr, te, *, do_orth):
    """All per-read (pred, true_test) pairs for ONE fold. Dotted names encode the
    result slot: 'ceil.<k>', 'align.<variant>.<k>', 'comp.<variant>.<k>'. Only one
    fold's preps/orth-fits are alive at a time (the caller frees them per fold) so
    peak memory stays ~one fold, not all N_FOLDS at once (#825 OOM fix)."""
    Xi, Yi, Xb, Yb = tens

    def rp(src, y_tr, x_ev):
        return _ridge_predict(preps[src], y_tr, x_ev)

    out = {}
    # ceilings
    out["ceil.within_instruct"] = (rp("Xi", Yi[tr], Xi[te]), Yi[te])  # Xi -> Yi
    out["ceil.within_base"] = (rp("Xb", Yb[tr], Xb[te]), Yb[te])  # Xb -> Yb
    out["ceil.repswap_b2i"] = (rp("Xb", Yi[tr], Xb[te]), Yi[te])  # Xb -> Yi
    out["ceil.repswap_i2b"] = (rp("Xi", Yb[tr], Xi[te]), Yb[te])  # Xi -> Yb
    # alignment own held-out R^2 (linear ridge)
    out["align.linear.A_ctx"] = (rp("Xb", Xi[tr], Xb[te]), Xi[te])  # Xb -> Xi
    out["align.linear.A_ctx_rev"] = (rp("Xi", Xb[tr], Xi[te]), Xb[te])  # Xi -> Xb
    out["align.linear.A_ans"] = (rp("Yb", Yi[tr], Yb[te]), Yi[te])  # Yb -> Yi
    out["align.linear.A_ans_rev"] = (rp("Yi", Yb[tr], Yi[te]), Yb[te])  # Yi -> Yb
    # compositions (linear alignment maps; M_base / M_inst ridge)
    xihat = rp("Xb", Xi[tr], Xb[te])  # A_ctx(Xb)
    out["comp.linear.comp_repmap_b2i"] = (rp("Xi", Yi[tr], xihat), Yi[te])  # M_inst o A_ctx
    xbhat = rp("Xi", Xb[tr], Xi[te])  # A_ctx_rev(Xi)
    out["comp.linear.comp_repmap_i2b"] = (rp("Xb", Yb[tr], xbhat), Yb[te])  # M_base o A_ctx_rev
    ybhat = rp("Xb", Yb[tr], xbhat)  # M_base(A_ctx_rev(Xi))
    out["comp.linear.comp_samefn_b2i"] = (
        rp("Yb", Yi[tr], ybhat),
        Yi[te],
    )  # A_ans o M_base o A_ctx_rev
    yihat = rp("Xi", Yi[tr], xihat)  # M_inst(A_ctx(Xb))
    out["comp.linear.comp_samefn_i2b"] = (
        rp("Yi", Yb[tr], yihat),
        Yb[te],
    )  # A_ans_rev o M_inst o A_ctx
    if not do_orth:
        return out
    for scale in (False, True):
        vk = "scaled_orthogonal" if scale else "orthogonal"

        def oc(X, reverse, sc=scale):  # Xb<->Xi orthogonal
            return _orth_predict(orth["ctx"], X, reverse=reverse, scale=sc)

        def oa(X, reverse, sc=scale):  # Yb<->Yi orthogonal
            return _orth_predict(orth["ans"], X, reverse=reverse, scale=sc)

        out[f"align.{vk}.A_ctx"] = (oc(Xb[te], False), Xi[te])
        out[f"align.{vk}.A_ctx_rev"] = (oc(Xi[te], True), Xb[te])
        out[f"align.{vk}.A_ans"] = (oa(Yb[te], False), Yi[te])
        out[f"align.{vk}.A_ans_rev"] = (oa(Yi[te], True), Yb[te])
        out[f"comp.{vk}.comp_repmap_b2i"] = (rp("Xi", Yi[tr], oc(Xb[te], False)), Yi[te])
        out[f"comp.{vk}.comp_repmap_i2b"] = (rp("Xb", Yb[tr], oc(Xi[te], True)), Yb[te])
        o_ybhat = rp("Xb", Yb[tr], oc(Xi[te], True))  # M_base(A_ctx_rev_orth(Xi))
        out[f"comp.{vk}.comp_samefn_b2i"] = (oa(o_ybhat, False), Yi[te])
        o_yihat = rp("Xi", Yi[tr], oc(Xb[te], False))  # M_inst(A_ctx_orth(Xb))
        out[f"comp.{vk}.comp_samefn_i2b"] = (oa(o_yihat, True), Yb[te])
    return out


def _assemble_battery(ss_res, ss_tot):
    """Fold-local pooled R^2 per dotted read-name -> nested result dict."""
    result = {"ceilings": {}, "alignment_r2": {}, "composition": {}}
    for name in ss_res:
        r2 = 1.0 - ss_res[name] / ss_tot[name] if ss_tot[name] > 1e-12 else float("nan")
        parts = name.split(".")
        if parts[0] == "ceil":
            result["ceilings"][parts[1]] = r2
        elif parts[0] == "align":
            result["alignment_r2"].setdefault(parts[1], {})[parts[2]] = r2
        else:  # comp
            result["composition"].setdefault(parts[1], {})[parts[2]] = r2
    return result


def _layer_battery(data, folds, layer, *, do_orth):
    """Fold-OUTER battery: build ONE fold's preps + orth-fits, compute every read's
    test prediction, accumulate fold-local pooled ss_res/ss_tot, then free the fold
    before the next (peak ~one fold, not all N_FOLDS -> avoids the #825 OOM)."""
    Xi, Yi, Xb, Yb = data["Xi"][layer], data["Yi"][layer], data["Xb"][layer], data["Yb"][layer]
    tens = (Xi, Yi, Xb, Yb)
    ss_res: dict[str, float] = {}
    ss_tot: dict[str, float] = {}
    for k in range(N_FOLDS):
        tr = torch.as_tensor(folds != k)
        te = torch.as_tensor(folds == k)
        if int(te.sum()) == 0 or int(tr.sum()) < 3:
            continue
        preps = {
            "Xb": _ridge_prep(Xb[tr]),
            "Xi": _ridge_prep(Xi[tr]),
            "Yb": _ridge_prep(Yb[tr]),
            "Yi": _ridge_prep(Yi[tr]),
        }
        orth = None
        if do_orth:
            orth = {"ctx": _orth_fit(Xb[tr], Xi[tr]), "ans": _orth_fit(Yb[tr], Yi[tr])}
        reads = _fold_reads(preps, orth, tens, tr, te, do_orth=do_orth)
        for name, (pred, true) in reads.items():
            ss_res[name] = ss_res.get(name, 0.0) + float(((true - pred) ** 2).sum())
            ss_tot[name] = ss_tot.get(name, 0.0) + float(((true - true.mean(0)) ** 2).sum())
        del preps, orth, reads
    return _assemble_battery(ss_res, ss_tot)


# ===========================================================================
# Nulls at the headline layer.
# ===========================================================================
def _random_orthogonal(d: int, gen: torch.Generator) -> torch.Tensor:
    A = torch.randn(d, d, dtype=torch.float64, generator=gen)
    Q, R = torch.linalg.qr(A)
    # sign fix for a Haar-distributed sample
    Q = Q * torch.sign(torch.diagonal(R))
    return Q


def _procrustes_cosine_null(Xb, Xi, Yb, Yi, *, n_draws, seed):
    """Chance reference for the parent's Procrustes-aligned cosine: aligned
    cosine of M_inst vs Q1^T M_base Q2 over random orthogonal Q1, Q2. Full data."""
    beta_i, _ = fit_primal_beta(Xi.cpu().numpy(), Yi.cpu().numpy())
    beta_b, _ = fit_primal_beta(Xb.cpu().numpy(), Yb.cpu().numpy())
    vi = beta_i.reshape(-1)
    vi_n = vi / (vi.norm() + 1e-12)
    # recompute the parent's fitted-Procrustes aligned cosine as a self-check
    dev = _fit_device()

    def _orth(A, B):
        M = A.T @ B
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return U @ Vh

    R_in = _orth(Xb - Xb.mean(0), Xi - Xi.mean(0))
    R_out = _orth(Yb - Yb.mean(0), Yi - Yi.mean(0))
    M_fit = R_in.T @ beta_b @ R_out
    vm = M_fit.reshape(-1)
    observed = float((vm @ vi) / (vm.norm() * vi.norm() + 1e-12))
    raw = float((beta_b.reshape(-1) @ vi) / (beta_b.norm() * vi.norm() + 1e-12))

    gen = torch.Generator().manual_seed(seed)
    d = beta_b.shape[0]
    draws = []
    for _ in range(n_draws):
        Q1 = _random_orthogonal(d, gen).to(dev)
        Q2 = _random_orthogonal(d, gen).to(dev)
        Mn = Q1.T @ beta_b @ Q2
        vmn = Mn.reshape(-1)
        draws.append(float((vmn @ vi_n) / (vmn.norm() + 1e-12)))
    draws = np.asarray(draws)
    z = (
        (observed - float(draws.mean())) / (float(draws.std()) + 1e-12)
        if draws.std() > 0
        else float("inf")
    )
    return {
        "observed_aligned_cosine": observed,
        "raw_vec_cosine": raw,
        "n_draws": int(n_draws),
        "null_mean": float(draws.mean()),
        "null_std": float(draws.std()),
        "null_p975": float(np.quantile(draws, 0.975)),
        "null_max": float(draws.max()),
        "z_observed_vs_null": float(z),
        "draws": [float(x) for x in draws],
    }


def _composition_collapse_null(data, folds, layer, *, n_draws, seed):
    """Random orthogonal Q in place of the A maps in the same-function composition.
    Held-out (5-fold pooled). Q pairs are generated PER DRAW (two D x D matrices
    alive at a time, not a full pool) to bound peak memory on the shared VM (#825)."""
    Xi, Yi, Xb, Yb = data["Xi"][layer], data["Yi"][layer], data["Xb"][layer], data["Yb"][layer]
    dev = _fit_device()
    fold_prep = {}
    fold_mu = {}
    for k in range(N_FOLDS):
        tr = folds != k
        fold_prep[k] = {"Xb": _ridge_prep(Xb[tr]), "Xi": _ridge_prep(Xi[tr])}
        fold_mu[k] = {
            "Xb": Xb[tr].mean(0),
            "Xi": Xi[tr].mean(0),
            "Yb": Yb[tr].mean(0),
            "Yi": Yi[tr].mean(0),
        }
    gen = torch.Generator().manual_seed(seed)
    d = Xi.shape[1]

    def _mask_to_k(tr):
        return int(np.unique(folds[~tr])[0])

    def _preds_fn(tr, te, Qc, Qa, name):
        k = _mask_to_k(tr)
        pp = fold_prep[k]
        mu = fold_mu[k]
        if name == "comp_samefn_b2i":  # Xi -> (Qc) Xb-space -> M_base -> (Qa) Yi
            xbhat = (Xi[te] - mu["Xi"]) @ Qc.T + mu["Xb"]
            ybhat = _ridge_predict(pp["Xb"], Yb[tr], xbhat)
            return (ybhat - mu["Yb"]) @ Qa + mu["Yi"]
        # comp_samefn_i2b : Xb -> (Qc) Xi-space -> M_inst -> (Qa) Yb
        xihat = (Xb[te] - mu["Xb"]) @ Qc + mu["Xi"]
        yihat = _ridge_predict(pp["Xi"], Yi[tr], xihat)
        return (yihat - mu["Yi"]) @ Qa.T + mu["Yb"]

    vals = {"comp_samefn_b2i": [], "comp_samefn_i2b": []}
    for _ in range(n_draws):
        Qc = _random_orthogonal(d, gen).to(dev)  # context-space random rotation
        Qa = _random_orthogonal(d, gen).to(dev)  # answer-space random rotation
        for name in vals:
            true = Yi if name == "comp_samefn_b2i" else Yb
            vals[name].append(
                _heldout_pooled_r2(
                    true,
                    folds,
                    lambda tr, te, n=name, qc=Qc, qa=Qa: _preds_fn(tr, te, qc, qa, n),
                )
            )
        del Qc, Qa
    out = {}
    for name, v in vals.items():
        v = np.asarray(v)
        out[name] = {
            "n_draws": int(n_draws),
            "null_mean": float(v.mean()),
            "null_std": float(v.std()),
            "null_p975": float(np.quantile(v, 0.975)),
            "null_max": float(v.max()),
            "draws": [float(x) for x in v],
        }
    return out


# ===========================================================================
# Residual structure at L19 (raw-space operators, full data).
# ===========================================================================
def _raw_ridge_operator(X, Y):
    """Full-data raw-space linear operator W (D_in, D_out): dY ~ dX_raw @ W."""
    beta, lam = fit_primal_beta(X.cpu().numpy(), Y.cpu().numpy())  # on standardized X
    xsd = X.std(0) + 1e-9
    W = beta / xsd[:, None]  # fold in the 1/xsd so W acts on raw dX
    return W, float(lam)


def _spectrum(mat: torch.Tensor, tops=(10, 50, 100)) -> dict:
    s = torch.linalg.svdvals(mat).clamp(min=0.0)
    s_np = s.cpu().numpy()
    total = float((s_np**2).sum())
    frob = float(np.sqrt(total))
    pr = float((s_np.sum() ** 2) / ((s_np**2).sum() + 1e-30))  # participation ratio
    shares = {
        f"top{k}_sv_share": float((s_np[:k] ** 2).sum() / (total + 1e-30))
        for k in tops
        if k <= len(s_np)
    }
    return {
        "frob_norm": frob,
        "participation_ratio": pr,
        "n_dims": len(s_np),
        "top_singular_values": [float(x) for x in s_np[:50]],
        **shares,
    }


def _residual_structure(data, layer, best_variant):
    Xi, Yi, Xb, Yb = data["Xi"][layer], data["Yi"][layer], data["Xb"][layer], data["Yb"][layer]
    W_inst, lam_i = _raw_ridge_operator(Xi, Yi)  # dXi -> dYi
    W_base, lam_b = _raw_ridge_operator(Xb, Yb)  # dXb -> dYb

    if best_variant == "linear":
        # linear A maps as raw operators: A_ctx_rev (Xi->Xb), A_ans (Yb->Yi)
        beta_cr, _ = fit_primal_beta(Xi.cpu().numpy(), Xb.cpu().numpy())
        W_ctx_rev = beta_cr / (Xi.std(0) + 1e-9)[:, None]  # dXi -> dXb
        beta_an, _ = fit_primal_beta(Yb.cpu().numpy(), Yi.cpu().numpy())
        W_ans = beta_an / (Yb.std(0) + 1e-9)[:, None]  # dYb -> dYi
        composed = W_ctx_rev @ W_base @ W_ans
    else:
        scale = best_variant == "scaled_orthogonal"
        f_ctx = _orth_fit(Xb, Xi)  # Xb<->Xi
        f_ans = _orth_fit(Yb, Yi)  # Yb<->Yi
        R_ctx_rev = f_ctx["R"].T  # dXi -> dXb
        R_ans = f_ans["R"]  # dYb -> dYi
        if scale:
            R_ctx_rev = R_ctx_rev * f_ctx["s_rev"]
            R_ans = R_ans * f_ans["s_fwd"]
        composed = R_ctx_rev @ W_base @ R_ans

    residual = W_inst - composed
    reference = W_inst - W_base  # unaligned reference (crude; different spaces)
    return {
        "best_variant": best_variant,
        "lambda_instruct": lam_i,
        "lambda_base": lam_b,
        "residual_aligned": _spectrum(residual),
        "reference_unaligned_Winst_minus_Wbase": _spectrum(reference),
        "W_inst_spectrum": _spectrum(W_inst),
    }


# ===========================================================================
# Gates
# ===========================================================================
def _load_committed_gate_values():
    with open(COMMITTED_CROSSMODEL) as f:
        d = json.load(f)
    p = d["pairs"]["S_assistant_chat"]
    L = str(HEADLINE_LAYER)
    return {
        "within_instruct": p["within_model"]["instruct"]["r2_by_layer"][L],
        "within_base": p["within_model"]["pretrained"]["r2_by_layer"][L],
        "repswap_b2i": p["representation_swap"]["base_rep_to_instruct_target"]["r2_by_layer"][L],
        "vec_cosine": p["weight_space"][L]["vec_cosine"],
        "procrustes_aligned": p["weight_space"][L]["procrustes_exploratory"]["aligned_vec_cosine"],
    }


# ===========================================================================
# Figures
# ===========================================================================
def _make_figures(results: dict, fig_root: str) -> None:
    """Write both figures under <fig_root>/issue_825/. fig_root is 'figures' for a
    production run and the scratch out-dir for a smoke run (so smoke never touches
    the committed figures/ tree)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    layers = results["metadata"]["frozen_layers"]
    variants = ["linear", "orthogonal", "scaled_orthogonal"]
    vlabels = {
        "linear": "General linear",
        "orthogonal": "Orthogonal",
        "scaled_orthogonal": "Scaled orthogonal",
    }
    colors = dict(zip(variants, paper_palette(len(variants)), strict=True))
    per_layer = results["per_layer"]

    # ---- Figure 1: same-function composition fraction of ceiling, per layer ----
    fig1, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained")
    dir_specs = [
        ("comp_samefn_b2i", "within_instruct", "base rep → instruct target"),
        ("comp_samefn_i2b", "within_base", "instruct rep → base target"),
    ]
    for ax, (comp_key, ceil_key, title) in zip(axes, dir_specs, strict=True):
        for v in variants:
            fracs = []
            for L in layers:
                comp = per_layer[str(L)]["composition"][v][comp_key]
                ceil = per_layer[str(L)]["ceilings"][ceil_key]
                fracs.append(comp / ceil if abs(ceil) > 1e-9 else float("nan"))
            ax.plot(layers, fracs, "o-", color=colors[v], label=vlabels[v])
        ax.axhline(1.0, color="grey", ls="--", lw=1.0, label="Held-out ceiling")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Fraction of held-out ceiling R²")
        ax.set_title(title)
        ax.set_xticks(layers)
    axes[0].legend(fontsize=7, loc="best")
    fig1.suptitle(
        "Same-function-different-basis composition (A_ans ∘ M o A_ctx_rev) vs ceiling",
        fontsize=10,
    )
    savefig_paper(fig1, "issue_825/map_alignment_composition_r2", dir=fig_root)
    plt.close(fig1)

    # ---- Figure 2: Procrustes-cosine null band + residual spectrum ----
    fig2, (axn, axr) = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained")
    pn = results["nulls"]["procrustes_cosine_null_L19"]
    axn.hist(pn["draws"], bins=30, color=paper_palette(3)[1], alpha=0.8, label="random-Q null")
    axn.axvline(
        pn["observed_aligned_cosine"],
        color=paper_palette(3)[0],
        lw=2.0,
        label=f"observed Procrustes {pn['observed_aligned_cosine']:.3f}",
    )
    axn.axvline(
        pn["raw_vec_cosine"], color="grey", ls=":", lw=1.5, label=f"raw {pn['raw_vec_cosine']:.3f}"
    )
    axn.set_xlabel("Aligned map cosine (M_inst vs Q₁ᵀ M_base Q₂)")
    axn.set_ylabel("Null draws")
    axn.set_title(f"L{HEADLINE_LAYER} Procrustes-cosine chance reference")
    axn.legend(fontsize=7, loc="upper center")

    rs = results["residual_structure_L19"]
    for key, lab, col in [
        ("residual_aligned", f"residual (best: {rs['best_variant']})", paper_palette(3)[0]),
        (
            "reference_unaligned_Winst_minus_Wbase",
            "W_inst - W_base (unaligned)",
            paper_palette(3)[2],
        ),
        ("W_inst_spectrum", "W_inst", "grey"),
    ]:
        sv = np.asarray(rs[key]["top_singular_values"])
        cum = np.cumsum(sv**2) / (rs[key]["frob_norm"] ** 2 + 1e-30)
        axr.plot(range(1, len(cum) + 1), cum, "o-", ms=3, color=col, label=lab)
    axr.set_xlabel("Singular-value index")
    axr.set_ylabel("Cumulative squared-SV share")
    axr.set_title(f"L{HEADLINE_LAYER} residual-operator spectrum")
    axr.legend(fontsize=7, loc="lower right")
    savefig_paper(fig2, "issue_825/map_alignment_nulls", dir=fig_root)
    plt.close(fig2)
    print(f"[figures] wrote {fig_root}/issue_825/map_alignment_{{composition_r2,nulls}}")


# ===========================================================================
# First-shard-only staging for smoke (mini npz, separate name so it never
# collides with the production full-extraction cache).
# ===========================================================================
def _extract_first_shard(stem: str, dl_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_tree

    npz_path = dl_dir / f"{stem}.smoke.npz"
    if npz_path.exists():
        return npz_path
    dl_dir.mkdir(parents=True, exist_ok=True)
    tok = os.environ.get("HF_TOKEN")
    tree = list(
        list_repo_tree(
            HF_DATA_REPO,
            path_in_repo=HF_PREFIX,
            repo_type="dataset",
            revision=HF_REV,
            recursive=False,
            token=tok,
        )
    )
    shard_files = sorted(
        os.path.basename(t.path)
        for t in tree
        if os.path.basename(t.path).startswith(f"{stem}_shard") and t.path.endswith(".pt")
    )
    if not shard_files:
        raise FileNotFoundError(f"no shards for {stem} at {HF_PREFIX}@{HF_REV}")
    fn = shard_files[0]
    p = hf_hub_download(
        HF_DATA_REPO,
        f"{HF_PREFIX}/{fn}",
        repo_type="dataset",
        revision=HF_REV,
        token=tok,
        local_dir=str(dl_dir),
    )
    payload = torch.load(p, map_location="cpu", weights_only=False)
    frozen = list(FROZEN_LAYERS)
    conv_ids = [str(c) for c in payload["conv_ids"]]
    slots = np.stack(
        [
            (r if torch.is_tensor(r) else torch.as_tensor(r)).float()[:, frozen, :].numpy()
            for r in payload["slots"]
        ]
    )
    profiles = np.stack(
        [
            (r if torch.is_tensor(r) else torch.as_tensor(r)).float()[:, frozen, :].numpy()
            for r in payload["profiles"]
        ]
    )
    del payload
    os.remove(p)
    np.savez(
        npz_path,
        slots=slots,
        profiles=profiles,
        conv_ids=np.asarray(conv_ids),
        layers=np.asarray(frozen),
    )
    print(f"[extract-smoke] {stem}: {fn} -> {npz_path} ({len(conv_ids)} rows)")
    return npz_path


# ===========================================================================
# Orchestration
# ===========================================================================
def run(out_dir: Path, dl_dir: Path, *, smoke: bool) -> dict:
    from huggingface_hub import HfApi

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    layers_subset = [14, HEADLINE_LAYER] if smoke else list(FROZEN_LAYERS)
    cos_draws = 5 if smoke else 100
    comp_null_draws = 3 if smoke else 20

    # ---- resolve/pin revision ----
    try:
        resolved = HfApi().repo_info(HF_DATA_REPO, repo_type="dataset", revision=HF_REV).sha
    except Exception as e:
        resolved = f"unresolved: {e}"

    # ---- stage ----
    if smoke:
        npz_i = _extract_first_shard(STEM_INSTRUCT, dl_dir)
        npz_b = _extract_first_shard(STEM_BASE, dl_dir)
    else:
        npz_i = cm.extract_stem(STEM_INSTRUCT, dl_dir)
        npz_b = cm.extract_stem(STEM_BASE, dl_dir)

    data, conv, layers, al = _load_pair(npz_i, npz_b, layers_subset)
    folds = _cv_folds(conv, N_FOLDS, FIT_SEED)
    print(f"[load] n_common={al['n_common']} layers={layers} staged in {time.time() - t0:.1f}s")

    # ---- self-check: factored ridge reproduces the parent core at L19 ----
    Lh = HEADLINE_LAYER
    self_checks = {}
    Xi, Yi = data["Xi"][Lh], data["Yi"][Lh]
    # parent-core within-instruct via cm.frozen_sweep (fold-local pooling)
    parent_within = cm.frozen_sweep(
        Xi.cpu().numpy()[:, None, :],
        Yi.cpu().numpy()[:, None, :],
        conv,
        [Lh],
        seed=FIT_SEED,
        null_draws=0,
    )["r2_by_layer"][Lh]
    mine_within = _heldout_pooled_r2(
        Yi,
        folds,
        lambda tr, te: _ridge_predict(
            _ridge_prep(Xi[torch.as_tensor(tr)]), Yi[torch.as_tensor(tr)], Xi[torch.as_tensor(te)]
        ),
    )
    self_checks["factored_vs_parent_within_L19"] = {
        "parent_core": parent_within,
        "factored": mine_within,
        "abs_delta": abs(parent_within - mine_within),
        "ok": abs(parent_within - mine_within) < 1e-6,
    }
    _sc_delta = self_checks["factored_vs_parent_within_L19"]["abs_delta"]
    print(f"[self-check] factored vs parent within L19 delta={_sc_delta:.2e}")

    # ---- per-layer battery ----
    per_layer = {}
    for L in layers:
        tL = time.time()
        per_layer[str(L)] = _layer_battery(data, folds, L, do_orth=True)
        print(f"[layer {L}] battery done in {time.time() - tL:.1f}s")

    # ---- gates ----
    committed = _load_committed_gate_values()
    gate_specs = [
        ("within_instruct", per_layer[str(Lh)]["ceilings"]["within_instruct"]),
        ("within_base", per_layer[str(Lh)]["ceilings"]["within_base"]),
        ("repswap_b2i", per_layer[str(Lh)]["ceilings"]["repswap_b2i"]),
    ]
    gates = {}
    all_pass = True
    for name, obs in gate_specs:
        exp = committed[name]
        delta = abs(obs - exp)
        ok = delta <= GATE_TOL
        gates[name] = {"observed": obs, "expected": exp, "abs_delta": delta, "pass": bool(ok)}
        if not ok:
            all_pass = False
    gates["all_pass"] = bool(all_pass)
    _gate_deltas = {k: (v if isinstance(v, bool) else v.get("abs_delta")) for k, v in gates.items()}
    print(f"[gates] {json.dumps(_gate_deltas)}")
    if not smoke and not all_pass:
        raise SystemExit(f"WIRING GATE FAILURE (|delta|>{GATE_TOL}): {json.dumps(gates, indent=2)}")

    # Free non-headline layers before the L19-only nulls + residual (bounds peak RSS
    # on the shared VM; nulls and residual read only the headline layer).
    for tk in ("Xi", "Yi", "Xb", "Yb"):
        for L in [x for x in list(data[tk]) if x != Lh]:
            del data[tk][L]

    # ---- nulls at L19 ----
    Xb, Yb = data["Xb"][Lh], data["Yb"][Lh]
    proc_null = _procrustes_cosine_null(Xb, Xi, Yb, Yi, n_draws=cos_draws, seed=FIT_SEED + 7)
    self_checks["procrustes_aligned_recomputed"] = proc_null["observed_aligned_cosine"]
    self_checks["procrustes_aligned_committed"] = committed["procrustes_aligned"]
    self_checks["vec_cosine_recomputed"] = proc_null["raw_vec_cosine"]
    self_checks["vec_cosine_committed"] = committed["vec_cosine"]
    comp_null = _composition_collapse_null(
        data, folds, Lh, n_draws=comp_null_draws, seed=FIT_SEED + 13
    )

    # ---- residual structure: best variant by the L19 samefn_b2i fraction ----
    def _samefn_frac(v):
        comp = per_layer[str(Lh)]["composition"][v]["comp_samefn_b2i"]
        ceil = per_layer[str(Lh)]["ceilings"]["within_instruct"]
        return comp / ceil if abs(ceil) > 1e-9 else float("-inf")

    best_variant = max(["linear", "orthogonal", "scaled_orthogonal"], key=_samefn_frac)
    residual = _residual_structure(data, Lh, best_variant)

    results = {
        "metadata": {
            "git_commit": cm._git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hf_repo": HF_DATA_REPO,
            "hf_revision_pinned": HF_REV,
            "hf_revision_resolved": resolved,
            "hf_prefix": HF_PREFIX,
            "stems": [STEM_INSTRUCT, STEM_BASE],
            "role": ROLE,
            "frozen_layers": layers,
            "headline_layer": HEADLINE_LAYER,
            "n_folds": N_FOLDS,
            "fit_seed": FIT_SEED,
            "lambdas": [float(x) for x in LAMBDAS],
            "device": str(_fit_device()),
            "n_common": al["n_common"],
            "n_instruct": al["n_a"],
            "n_pretrained": al["n_b"],
            "cosine_null_draws": cos_draws,
            "composition_null_draws": comp_null_draws,
            "thread_caps": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "script": "scripts/issue825_map_alignment.py",
            "ridge_core_source": "issue825_fit_cells.py@56ee95fe8a (via crossmodel, verbatim)",
            "smoke": bool(smoke),
            "wall_seconds": None,  # filled below
        },
        "gates": gates,
        "self_checks": self_checks,
        "per_layer": per_layer,
        "nulls": {
            "procrustes_cosine_null_L19": proc_null,
            "composition_collapse_null_L19": comp_null,
        },
        "residual_structure_L19": residual,
        "caveats": [
            "Descriptive geometry on a SINGLE seed; no mechanism claims.",
            "S_assistant_chat pair (chat single-turn, assistant slot) only -- the "
            "brief-named chat_s bundles.",
            "Composition R^2 uses the parent frozen_sweep fold-local-mean pooling so "
            "the fraction-of-ceiling is apples-to-apples with the committed ceilings.",
            "Residual-structure operators are full-data (descriptive spectrum), not held-out.",
            "The unaligned residual reference W_inst - W_base compares operators over "
            "DIFFERENT input/output frames; it is a crude reference, not a matched null.",
        ],
    }
    results["metadata"]["wall_seconds"] = round(time.time() - t0, 1)

    out_json = out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[write] {out_json} (wall {results['metadata']['wall_seconds']}s)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="eval_results/issue_825/map_alignment")
    ap.add_argument("--dl-dir", default="data/issue_825/hf_dl/map_alignment")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    # Figures divert WITH the JSON: committed figures/ tree only on a production
    # run; a smoke run writes both under its scratch out-dir (never clobbers).
    fig_root = "figures" if not args.smoke else str(out_dir)

    if args.figures_only:
        with open(out_dir / "results.json") as f:
            results = json.load(f)
        _make_figures(results, "figures")
        return

    results = run(out_dir, Path(args.dl_dir), smoke=args.smoke)
    _make_figures(results, fig_root)


if __name__ == "__main__":
    main()
