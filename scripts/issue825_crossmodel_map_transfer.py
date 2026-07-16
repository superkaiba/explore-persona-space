"""Cross-model comparison of the #825 context->answer-profile ridge maps.

Task #825 (followup_label: crossmodel-map-transfer). The parent run fit, per
model, a linear ridge map h: c_x -> v(x) from a context's slot activation to
the target turn's mean-activation profile (per layer, held-out K-fold GCV
ridge, #779 recipe). The parent NEVER compared the two models' maps directly.
This 0-GPU analysis-only follow-up does exactly that, on the persisted
turnstore tensors (no new training / generation / eval).

Battery (all at the parent's FROZEN_LAYERS, headline layer 19; per-fold + mean
on the parent's own fold scheme, K=5 seed=0):
  (a) within-model baselines   -- reproduce the parent's held-out R^2
  (b) cross-model MAP transfer  -- map-swap (apply model A's fitted map to model
      B's held-out rows, own target) + representation-swap (fit c_A -> v_B)
  (c) weight-space similarity   -- vec-cosine, per-output-dim cosine, principal
      angles of the primal coefficient matrices, + exploratory Procrustes
  (d) null / reference bands    -- selection-symmetric shuffle nulls for every
      transfer R^2; a shuffled-target random-map reference for the cosines

Descriptive geometry on a SINGLE seed -- no mechanism claims.

The ridge CORE (LAMBDAS, _prep_fold, _ridge_predict_cached, _pooled_r2,
_per_example_cosine, _cv_folds) is copied VERBATIM from
scripts/issue825_fit_cells.py @ 56ee95fe8a (identical on main and issue-825)
so within-model numbers reproduce bit-for-bit; the map-swap / weight-space /
Procrustes functions are new. The Gram-space ridge caches eigh(G) per fold and
pushes observed + every null draw through the cached (w,V,Kev) path -- no
per-draw refit (artifact-reuse throughput check (i): the reused core is already
vectorized + device-parametrized).

CLI:
  uv run python scripts/issue825_crossmodel_map_transfer.py \
      --out eval_results/issue_825/crossmodel_map_transfer \
      --dl-dir data/issue_825/hf_dl/crossmodel [--figures-only] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirror explore_persona_space.experiments.issue_825.common)
# ---------------------------------------------------------------------------
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_REV = "deb7a4523b5233393e4fbd2497622527b3622d35"  # pinned turnstore revision
HF_PREFIX = "issue825_userbase_map/analysis_tensors"
FROZEN_LAYERS = (14, 18, 19, 26)
HEADLINE_LAYER = 19
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
FIT_SEED = 0
N_FOLDS = 5
N_NULL_DRAWS = 100

# The six turnstore bundles (stem = <model>_<format>_<track>); each shard is a
# .pt dict {conv_ids, slots (n_slots,28,D), profiles (n_turns,28,D), perpos, nll}.
STEMS = [
    "instruct_chat_s",
    "pretrained_chat_s",
    "instruct_chat_m",
    "pretrained_chat_m",
    "instruct_naturalistic_m",
    "pretrained_naturalistic_m",
]
# Track-M turn order [u1,a1,u2,a2]; slots [assistant(before a1), user(before u2)].
# Track-S turn order [u1,a1]; slot [assistant]. (from issue825_fit_cells._normalize_cell)
ROLE_INDEX = {  # role -> (slot_index, target_turn_index)
    "assistant": (0, 1),
    "user": (1, 2),
}

# Matched (role, format, track) cross-model comparison pairs.
PAIRS = [
    # id, role, instruct_stem, pretrained_stem
    ("S_assistant_chat", "assistant", "instruct_chat_s", "pretrained_chat_s"),
    ("M_assistant_chat", "assistant", "instruct_chat_m", "pretrained_chat_m"),
    (
        "M_assistant_naturalistic",
        "assistant",
        "instruct_naturalistic_m",
        "pretrained_naturalistic_m",
    ),
    ("M_user_chat", "user", "instruct_chat_m", "pretrained_chat_m"),
    ("M_user_naturalistic", "user", "instruct_naturalistic_m", "pretrained_naturalistic_m"),
]

LAMBDAS = np.logspace(-2, 4, 13)  # verbatim from issue825_fit_cells


def _fit_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ===========================================================================
# Ridge core -- VERBATIM from scripts/issue825_fit_cells.py @ 56ee95fe8a
# ===========================================================================
def _prep_fold(X_train: np.ndarray, X_eval: np.ndarray) -> dict:
    dev = _fit_device()
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64).to(dev)
    Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64).to(dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    Kev = Xev_n @ Xtr_n.T
    KevV = Kev @ V
    return {"w": w, "V": V, "KevV": KevV, "ntr": int(Xtr.shape[0])}


def _ridge_predict_cached(cache: dict, Y_train: np.ndarray) -> np.ndarray:
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64).to(cache["w"].device)
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    w, V, KevV, ntr = cache["w"], cache["V"], cache["KevV"], cache["ntr"]
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
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
    filt = 1.0 / (w + best_lam)
    pred = (KevV * filt) @ VtY + ymu
    return pred.cpu().numpy()


def _pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def _cv_folds(conv_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    conv_ids = np.asarray(conv_ids)
    uniq = np.unique(conv_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    conv_fold = {cid: int(perm[i] % n_folds) for i, cid in enumerate(uniq)}
    folds = np.array([conv_fold[c] for c in conv_ids], dtype=np.int64)
    return folds


# ===========================================================================
# Extraction: incremental per-shard download -> frozen-layer slots/profiles ->
# delete shard (peak footprint ~ one shard, ~2 GB; the perpos tensor is never
# retained). Cached: skips a stem whose .npz already exists.
# ===========================================================================
def extract_stem(stem: str, dl_dir: Path, revision: str | None = None) -> Path:
    """Stage one stem's frozen-layer npz from the pinned HF turnstore.

    ``revision`` parameterizes the hardcoded ``HF_REV`` pin (#1345 adapter:
    the four S-track stems resolve only at 7159e5804d, not at the module
    default deb7a452); ``None`` preserves the committed behavior byte-for-byte.
    """
    from huggingface_hub import hf_hub_download, list_repo_tree

    rev = HF_REV if revision is None else revision
    npz_path = dl_dir / f"{stem}.npz"
    if npz_path.exists():
        print(f"[extract] {stem}: cached {npz_path}")
        return npz_path
    dl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_DISABLE", "1")  # xet finalization hang (#825 r2)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    tok = os.environ.get("HF_TOKEN")
    tree = list(
        list_repo_tree(
            HF_DATA_REPO,
            path_in_repo=HF_PREFIX,
            repo_type="dataset",
            revision=rev,
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
        raise FileNotFoundError(f"no shards for {stem} at {HF_PREFIX}@{rev}")
    frozen = list(FROZEN_LAYERS)
    slots_acc: list[np.ndarray] = []
    profiles_acc: list[np.ndarray] = []
    conv_ids: list[str] = []
    for fn in shard_files:
        p = hf_hub_download(
            HF_DATA_REPO,
            f"{HF_PREFIX}/{fn}",
            repo_type="dataset",
            revision=rev,
            token=tok,
            local_dir=str(dl_dir),
        )
        payload = torch.load(p, map_location="cpu", weights_only=False)
        conv_ids.extend([str(c) for c in payload["conv_ids"]])
        for r in payload["slots"]:
            t = r if torch.is_tensor(r) else torch.as_tensor(r)
            slots_acc.append(t.float()[:, frozen, :].to(torch.float16).numpy())
        for r in payload["profiles"]:
            t = r if torch.is_tensor(r) else torch.as_tensor(r)
            profiles_acc.append(t.float()[:, frozen, :].to(torch.float16).numpy())
        del payload
        os.remove(p)  # drop the ~2 GB shard (incl. perpos) immediately
        print(f"[extract] {stem}: {fn} done ({len(conv_ids)} rows), shard removed")
    slots = np.stack(slots_acc)  # (N, n_slots, Lf, D)
    profiles = np.stack(profiles_acc)  # (N, n_turns, Lf, D)
    assert slots.shape[2] == len(frozen) and slots.shape[3] == EXPECTED_HIDDEN, slots.shape
    np.savez(
        npz_path,
        slots=slots,
        profiles=profiles,
        conv_ids=np.asarray(conv_ids),
        layers=np.asarray(frozen),
    )
    print(f"[extract] {stem}: wrote {npz_path}  slots={slots.shape} profiles={profiles.shape}")
    return npz_path


def load_cell(npz_path: Path, role: str) -> dict:
    """Return {X (N,Lf,D), Y (N,Lf,D), conv_ids, layers} for a role in a bundle."""
    d = np.load(npz_path, allow_pickle=True)
    slots = d["slots"].astype(np.float32)
    profiles = d["profiles"].astype(np.float32)
    conv_ids = np.asarray([str(c) for c in d["conv_ids"]])
    layers = [int(x) for x in d["layers"]]
    si, ti = ROLE_INDEX[role]
    X = slots[:, si, :, :]
    Y = profiles[:, ti, :, :]
    keep = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(Y).any(axis=(1, 2)))
    return {"X": X[keep], "Y": Y[keep], "conv_ids": conv_ids[keep], "layers": layers}


def align_pair(a: dict, b: dict) -> dict:
    """Row-align two cells by shared conv_id (intersection, sorted)."""
    ida, idb = a["conv_ids"], b["conv_ids"]
    common = np.array(sorted(set(ida.tolist()) & set(idb.tolist())))
    pos_a = {c: i for i, c in enumerate(ida.tolist())}
    pos_b = {c: i for i, c in enumerate(idb.tolist())}
    ia = np.array([pos_a[c] for c in common])
    ib = np.array([pos_b[c] for c in common])
    return {
        "common": common,
        "ia": ia,
        "ib": ib,
        "n_a": len(ida),
        "n_b": len(idb),
        "n_common": len(common),
    }


# ===========================================================================
# Frozen-layer sweeps (reporting keyed by TRUE layer number)
# ===========================================================================
def _shuffle_target_null(preds, true, *, n_draws, seed):
    """Selection-symmetric permutation null: pooled R^2 of the FIXED held-out
    predictions against row-shuffled held-out targets (the brief's "shuffle
    target rows within held-out fold" null; no refit, so >=100 draws are cheap).
    preds, true: (Nf, D) over the fitted rows."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        perm = rng.permutation(true.shape[0])
        vals.append(_pooled_r2(preds, true[perm]))
    return float(np.nanmean(vals)), float(np.nanquantile(vals, 0.975))


def frozen_sweep(X, Y, conv_ids, layers, *, seed, null_draws):
    """Held-out pooled R^2 per stored layer for observed Y + target-shuffle null.

    X, Y: (N, Lf, D) row-aligned. One ridge fit per (fold, layer) via the parent's
    cached Gram path (verbatim core); the null shuffles held-out target rows
    against the fixed held-out predictions (no per-draw refit). Only the stored
    (frozen) layers are swept; reporting is keyed by TRUE layer number.
    """
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    n, Lf = X.shape[0], X.shape[1]
    folds = _cv_folds(conv_ids, N_FOLDS, seed)
    ss_res = np.zeros(Lf)
    ss_tot = np.zeros(Lf)
    preds = {li: np.zeros((n, Y.shape[2]), np.float32) for li in range(Lf)}
    fitted = np.zeros(n, bool)
    for li in range(Lf):
        Xl, Yl = X[:, li, :], Y[:, li, :]
        for k in range(N_FOLDS):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = _prep_fold(Xl[tr], Xl[te])
            pred = _ridge_predict_cached(cache, Yl[tr])
            fitted[te] = True
            preds[li][te] = pred.astype(np.float32)
            true = Yl[te].astype(np.float64)
            ss_res[li] += float(np.sum((true - pred) ** 2))
            ss_tot[li] += float(np.sum((true - true.mean(0)) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)
    null_mean, null_p975 = {}, {}
    for li in range(Lf):
        layer = int(layers[li])
        if null_draws > 0 and fitted.any():
            m, p = _shuffle_target_null(
                preds[li][fitted],
                Y[fitted, li, :].astype(np.float64),
                n_draws=null_draws,
                seed=seed + 1 + li,
            )
        else:
            m, p = float("nan"), float("nan")
        null_mean[layer] = m
        null_p975[layer] = p
    return {
        "r2_by_layer": {int(layers[li]): float(r2[li]) for li in range(Lf)},
        "null_mean_by_layer": null_mean,
        "null_p975_by_layer": null_p975,
        "folds": folds,
        "preds": preds,
        "fitted": fitted,
    }


def frozen_map_swap(X_src, Y_src, X_tgt, Y_tgt, conv_ids, layers, *, seed, null_draws):
    """Apply model-SRC's fitted map to model-TGT's held-out rows (own target).

    Rows are already aligned (index i = same conv_id in src and tgt). Per fold:
    fit the ridge on SRC train rows (SRC train stats), evaluate on TGT test rows
    standardized by SRC's train stats; score against TGT's own target Y_tgt.
    This is the parent's dual-ridge predictor applied cross-model -- no explicit
    weight matrix needed; _prep_fold standardizes the TGT eval points by SRC's
    stats exactly as applying SRC's map requires. Null shuffles TGT test rows.
    """
    X_src = np.asarray(X_src, np.float32)
    Y_src = np.asarray(Y_src, np.float32)
    X_tgt = np.asarray(X_tgt, np.float32)
    Y_tgt = np.asarray(Y_tgt, np.float32)
    Lf = X_src.shape[1]
    folds = _cv_folds(conv_ids, N_FOLDS, seed)
    rng = np.random.default_rng(seed + 5)
    r2 = {}
    null_mean = {}
    null_p975 = {}
    for li in range(Lf):
        Xs, Ys, Xt, Yt = X_src[:, li, :], Y_src[:, li, :], X_tgt[:, li, :], Y_tgt[:, li, :]
        preds = np.zeros((X_tgt.shape[0], Y_tgt.shape[2]), np.float32)
        fitted = np.zeros(X_tgt.shape[0], bool)
        for k in range(N_FOLDS):
            te = folds == k
            tr = ~te
            if te.sum() == 0 or tr.sum() < 3:
                continue
            cache = _prep_fold(Xs[tr], Xt[te])  # SRC train stats; TGT eval pts
            pred = _ridge_predict_cached(cache, Ys[tr])  # SRC targets
            preds[te] = pred.astype(np.float32)
            fitted[te] = True
        r2[int(layers[li])] = _pooled_r2(preds[fitted], Yt[fitted])
        # Null: shuffle the held-out TGT targets relative to the transfer preds.
        true = Yt[fitted]
        pr = preds[fitted]
        nd = []
        for _ in range(null_draws):
            perm = rng.permutation(true.shape[0])
            nd.append(_pooled_r2(pr, true[perm]))
        null_mean[int(layers[li])] = float(np.nanmean(nd))
        null_p975[int(layers[li])] = float(np.nanquantile(nd, 0.975))
    return {"r2_by_layer": r2, "null_mean_by_layer": null_mean, "null_p975_by_layer": null_p975}


# ===========================================================================
# Weight-space: primal ridge coefficient matrix + cosine / principal angles /
# Procrustes. beta is the map on STANDARDIZED inputs: pred = ((x-xmu)/xsd)@beta+ymu.
# ===========================================================================
def fit_primal_beta(X, Y):
    """Full-data primal ridge coefficient (D_in x D_out), GCV lambda (same grid)."""
    dev = _fit_device()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64).to(dev)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64).to(dev)
    xmu = Xt.mean(0)
    xsd = Xt.std(0) + 1e-9
    Xn = (Xt - xmu) / xsd
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Yc
    sqVtY = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    ntr = Xn.shape[0]
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = 1.0 / (w + best_lam)
    beta = Xn.T @ (V @ (filt[:, None] * VtY))  # (D_in, D_out)
    return beta, best_lam


def principal_angles(beta_a, beta_b, k):
    """cos of top-k principal angles between the right (output) singular subspaces."""
    _, _, Vha = torch.linalg.svd(beta_a, full_matrices=False)  # Vha (D_out, D_out)
    _, _, Vhb = torch.linalg.svd(beta_b, full_matrices=False)
    Qa = Vha[:k].T  # (D_out, k) orthonormal
    Qb = Vhb[:k].T
    M = Qa.T @ Qb  # (k, k)
    cs = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    return cs.cpu().numpy()


def weight_space_compare(X_i, Y_i, X_p, Y_p, layers, *, seed, do_svd_layers):
    """Per-layer primal-beta cosine, per-output-dim cosine dist, principal angles,
    a shuffled-target random-map cosine reference, and exploratory Procrustes.

    X_i/Y_i (instruct), X_p/Y_p (pretrained) are row-aligned (N, Lf, D)."""
    out = {}
    rng = np.random.default_rng(seed + 11)
    for li, layer in enumerate(layers):
        beta_i, lam_i = fit_primal_beta(X_i[:, li, :], Y_i[:, li, :])
        beta_p, lam_p = fit_primal_beta(X_p[:, li, :], Y_p[:, li, :])
        vi = beta_i.reshape(-1)
        vp = beta_p.reshape(-1)
        vec_cos = float((vi @ vp) / (vi.norm() * vp.norm() + 1e-12))
        # per-output-dim cosine (cosine of each output column's input-weight vec)
        num = (beta_i * beta_p).sum(0)
        den = beta_i.norm(dim=0) * beta_p.norm(dim=0) + 1e-12
        col_cos = (num / den).cpu().numpy()
        # random-map reference: instruct beta on ROW-SHUFFLED instruct targets
        perm = rng.permutation(X_i.shape[0])
        beta_i_sh, _ = fit_primal_beta(X_i[:, li, :], Y_i[perm][:, li, :])
        vish = beta_i_sh.reshape(-1)
        rand_vec_cos = float((vish @ vp) / (vish.norm() * vp.norm() + 1e-12))
        rec = {
            "lambda_instruct": lam_i,
            "lambda_pretrained": lam_p,
            "vec_cosine": vec_cos,
            "per_output_dim_cosine": {
                "mean": float(np.mean(col_cos)),
                "median": float(np.median(col_cos)),
                "q25": float(np.quantile(col_cos, 0.25)),
                "q75": float(np.quantile(col_cos, 0.75)),
            },
            "random_map_vec_cosine_ref": rand_vec_cos,
            "analytic_random_vec_cosine_sd": float(
                1.0 / (EXPECTED_HIDDEN)
            ),  # 1/sqrt(D_in*D_out)=1/D
        }
        if layer in do_svd_layers:
            for k in (10, 50):
                cs = principal_angles(beta_i, beta_p, k)
                # random reference: instruct-vs-shuffled principal angles
                cs_rand = principal_angles(beta_i, beta_i_sh, k)
                rec[f"principal_angle_cos_k{k}"] = {
                    "mean_cos": float(np.mean(cs)),
                    "min_cos": float(np.min(cs)),
                    "max_cos": float(np.max(cs)),
                    "cos_values": [float(x) for x in cs],
                    "random_ref_mean_cos": float(np.mean(cs_rand)),
                }
            rec["per_output_dim_cosine"]["hist"] = _hist(col_cos)
            # exploratory input+output Procrustes alignment
            rec["procrustes_exploratory"] = _procrustes_align(
                X_i[:, li, :], X_p[:, li, :], Y_i[:, li, :], Y_p[:, li, :], beta_p, beta_i, vec_cos
            )
        out[int(layer)] = rec
    return out


def _hist(vals, bins=40):
    counts, edges = np.histogram(vals, bins=bins, range=(-1.0, 1.0))
    return {"bin_edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _procrustes_align(X_i, X_p, Y_i, Y_p, beta_p, beta_i, raw_vec_cos):
    """EXPLORATORY: orthogonal Procrustes on the shared contexts. Fit R_in
    (c_p @ R_in ~ c_i) and R_out (v_p @ R_out ~ v_i); the pretrained map in the
    instruct frame predicting instruct output is M = R_in^T @ beta_p @ R_out.
    Report cosine(vec(M), vec(beta_i)) vs the raw cosine -- does aligning the
    two activation frames raise map agreement? Fit on ALL rows (exploratory)."""
    dev = _fit_device()
    Xi = torch.as_tensor(X_i, dtype=torch.float64).to(dev)
    Xp = torch.as_tensor(X_p, dtype=torch.float64).to(dev)
    Yi = torch.as_tensor(Y_i, dtype=torch.float64).to(dev)
    Yp = torch.as_tensor(Y_p, dtype=torch.float64).to(dev)

    def _orth(A, B):  # R minimizing ||A R - B||, R orthogonal: R = U V^T from A^T B
        M = A.T @ B
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return U @ Vh

    R_in = _orth(Xp - Xp.mean(0), Xi - Xi.mean(0))
    R_out = _orth(Yp - Yp.mean(0), Yi - Yi.mean(0))
    M = R_in.T @ beta_p @ R_out
    vi = beta_i.reshape(-1)
    vm = M.reshape(-1)
    aligned_cos = float((vm @ vi) / (vm.norm() * vi.norm() + 1e-12))
    return {"raw_vec_cosine": float(raw_vec_cos), "aligned_vec_cosine": aligned_cos}


# ===========================================================================
# Orchestration
# ===========================================================================
def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _compute_pair(pair_id, role, stem_i, stem_p, npz):
    ci = load_cell(npz[stem_i], role)
    cp = load_cell(npz[stem_p], role)
    al = align_pair(ci, cp)
    layers = ci["layers"]
    ia, ib = al["ia"], al["ib"]
    Xi, Yi = ci["X"][ia], ci["Y"][ia]
    Xp, Yp = cp["X"][ib], cp["Y"][ib]
    conv = al["common"]
    print(
        f"\n=== pair {pair_id} (role={role}) n_common={al['n_common']} "
        f"(instruct {al['n_a']}, pretrained {al['n_b']}) ==="
    )
    within_i = frozen_sweep(Xi, Yi, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)
    within_p = frozen_sweep(Xp, Yp, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)
    ms_b2i = frozen_map_swap(Xp, Yp, Xi, Yi, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)
    ms_i2b = frozen_map_swap(Xi, Yi, Xp, Yp, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)
    rs_b2i = frozen_sweep(Xp, Yi, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)
    rs_i2b = frozen_sweep(Xi, Yp, conv, layers, seed=FIT_SEED, null_draws=N_NULL_DRAWS)

    def _retained(swap_r2, within):
        return {
            str(L): (
                swap_r2[L] / within["r2_by_layer"][L]
                if abs(within["r2_by_layer"][L]) > 1e-9
                else float("nan")
            )
            for L in swap_r2
        }

    ws = weight_space_compare(Xi, Yi, Xp, Yp, layers, seed=FIT_SEED, do_svd_layers={HEADLINE_LAYER})
    return {
        "role": role,
        "format": ("chat" if "chat" in stem_i else "naturalistic"),
        "track": ("S" if stem_i.endswith("_s") else "M"),
        "n_common": al["n_common"],
        "n_instruct": al["n_a"],
        "n_pretrained": al["n_b"],
        "within_model": {
            "instruct": {
                "r2_by_layer": within_i["r2_by_layer"],
                "null_mean_by_layer": within_i["null_mean_by_layer"],
            },
            "pretrained": {
                "r2_by_layer": within_p["r2_by_layer"],
                "null_mean_by_layer": within_p["null_mean_by_layer"],
            },
        },
        "map_swap": {
            "base_to_instruct": {
                "r2_by_layer": ms_b2i["r2_by_layer"],
                "null_mean_by_layer": ms_b2i["null_mean_by_layer"],
                "frac_within_target_retained": _retained(ms_b2i["r2_by_layer"], within_i),
            },
            "instruct_to_base": {
                "r2_by_layer": ms_i2b["r2_by_layer"],
                "null_mean_by_layer": ms_i2b["null_mean_by_layer"],
                "frac_within_target_retained": _retained(ms_i2b["r2_by_layer"], within_p),
            },
        },
        "representation_swap": {
            "base_rep_to_instruct_target": {
                "r2_by_layer": rs_b2i["r2_by_layer"],
                "null_mean_by_layer": rs_b2i["null_mean_by_layer"],
            },
            "instruct_rep_to_base_target": {
                "r2_by_layer": rs_i2b["r2_by_layer"],
                "null_mean_by_layer": rs_i2b["null_mean_by_layer"],
            },
        },
        "weight_space": ws,
    }


def run_battery(dl_dir: Path, out_dir: Path) -> dict:
    npz = {stem: extract_stem(stem, dl_dir) for stem in STEMS}
    device = str(_fit_device())
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "metadata": {
            "git_commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hf_repo": HF_DATA_REPO,
            "hf_revision": HF_REV,
            "hf_prefix": HF_PREFIX,
            "frozen_layers": list(FROZEN_LAYERS),
            "headline_layer": HEADLINE_LAYER,
            "n_folds": N_FOLDS,
            "fit_seed": FIT_SEED,
            "null_draws": N_NULL_DRAWS,
            "device": device,
            "lambdas": [float(x) for x in LAMBDAS],
            "script": "scripts/issue825_crossmodel_map_transfer.py",
            "ridge_core_source": "scripts/issue825_fit_cells.py@56ee95fe8a (verbatim)",
        },
        "pairs": {},
        "caveats": [
            "Single seed (fit seed 0, generation seed 42); descriptive geometry only, no mechanism claims.",  # noqa: E501
            "v(x) is each model's OWN mean-activation profile (D=3584, 28 layers): dimensionally commensurate but the two residual-stream bases are not a-priori aligned. Map-swap scores model-A's map output (in A's activation basis) against model-B's target (B's basis), so a low map-swap R^2 confounds map difference with output-basis mismatch; representation-swap re-fits the read-out into the target basis and isolates whether the source CONTEXT representation carries the info.",  # noqa: E501
            "Row alignment is by conv_id (shared lmsys first user turn). Downstream turns differ per model: Track-S responses are each model's own samples; Track-M assistant turns are each model's own greedy generations; the second user turn (user cells) is Haiku-written (this deb7a452 turnstore is the round-1/2 Haiku-u2 store, NOT the round-3 self-written store).",  # noqa: E501
            "Track S (S1/S2) is single-turn, chat format for BOTH models -> matched-format cross-model comparison (NOT format-confounded). Track-M chat vs naturalistic cells compare within a matched format across models; assistant vs user compare within a matched role.",  # noqa: E501
            "User cells are ridge-negative within-model (no linear map), so cross-model transfer there compares two near-null maps and is reported for completeness, not as a map-shift signal.",  # noqa: E501
            "Weight-space beta is the map on STANDARDIZED inputs (each model's own mean/std); principal angles use the right (output-space) singular subspaces.",  # noqa: E501
        ],
    }
    for pair_id, role, stem_i, stem_p in PAIRS:
        ckpt = pairs_dir / f"{pair_id}.json"
        if ckpt.exists():
            print(f"[battery] {pair_id}: cached {ckpt}")
            results["pairs"][pair_id] = json.loads(ckpt.read_text())
            continue
        pair_res = _compute_pair(pair_id, role, stem_i, stem_p, npz)
        ckpt.write_text(json.dumps(pair_res, indent=2, default=float))
        print(f"[battery] {pair_id}: wrote {ckpt}")
        results["pairs"][pair_id] = pair_res
    return results


# ===========================================================================
# Figures
# ===========================================================================
def make_figures(results: dict, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("neurips")
    L = str(HEADLINE_LAYER)

    def _ws_at(ws_dict):
        # weight_space keys are ints in-process but strings after a results.json
        # round-trip (--figures-only) -- accept either.
        return ws_dict.get(HEADLINE_LAYER, ws_dict.get(str(HEADLINE_LAYER), {}))

    pairs = results["pairs"]
    order = [p for p, _, _, _ in PAIRS if p in pairs]
    labels = {
        "S_assistant_chat": "Assistant\n(single-turn, chat)",
        "M_assistant_chat": "Assistant\n(two-turn, chat)",
        "M_assistant_naturalistic": "Assistant\n(two-turn, natural.)",
        "M_user_chat": "User\n(two-turn, chat)",
        "M_user_naturalistic": "User\n(two-turn, natural.)",
    }

    # ---- Figure 1: transfer R^2 (layer 19) ---------------------------------
    c_within = paper_palette_role("primary")
    c_map = paper_palette_role("baseline")
    c_rep = paper_palette_role("control")
    c_null = paper_palette_role("neutral")
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(order))
    w = 0.2
    series = [
        ("Within-model (instruct)", c_within, "within_model", "instruct", None),
        ("Map-swap (base map on instruct)", c_map, "map_swap", "base_to_instruct", None),
        (
            "Repr-swap (base ctx to instruct target)",
            c_rep,
            "representation_swap",
            "base_rep_to_instruct_target",
            None,
        ),
    ]
    ymin = -2.2  # clip the uninformative extreme map-swap bars (collapsed cells reach -10)
    for si, (lab, col, grp, sub, _) in enumerate(series):
        vals = [pairs[p][grp][sub]["r2_by_layer"][L] for p in order]
        ax.bar(x + (si - 1) * w, vals, w, label=lab, color=col, edgecolor="white", linewidth=0.4)
        # annotate bars clipped by the y-floor with their true value
        for xi, v in zip(x + (si - 1) * w, vals, strict=False):
            if v < ymin:
                ax.annotate(
                    f"{v:.1f}",
                    (xi, ymin),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color=col,
                )
    # null band (max over the plotted transfer nulls per pair)
    nulls = []
    for p in order:
        pd = pairs[p]
        nulls.append(
            max(
                pd["map_swap"]["base_to_instruct"]["null_mean_by_layer"][L],
                pd["representation_swap"]["base_rep_to_instruct_target"]["null_mean_by_layer"][L],
            )
        )
    ax.plot(
        x,
        nulls,
        "_",
        color=c_null,
        markersize=18,
        markeredgewidth=2.0,
        label="Shuffle null (mean)",
        zorder=5,
    )
    ax.axhline(0.0, color="#999999", lw=0.8, zorder=0)
    ax.set_ylim(ymin, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in order], fontsize=8)
    ax.set_ylabel("Held-out $R^2$ (layer 19)")
    ax.set_title("Cross-model map transfer: base map vs instruct behaviour (y clipped at -2.2)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    savefig_paper(fig, "issue_825/crossmodel_transfer_r2", dir="figures")
    plt.close(fig)

    # ---- Figure 2: map similarity (headline layer, assistant cells) --------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    asst = [p for p in order if pairs[p]["role"] == "assistant"]
    # (a) principal-angle spectrum (k=50) per assistant pair + random ref
    c_cycle = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]
    for i, p in enumerate(asst):
        ws = _ws_at(pairs[p]["weight_space"])
        pa = ws.get("principal_angle_cos_k50")
        if not pa:
            continue
        cs = np.array(pa["cos_values"])
        axa.plot(
            np.arange(1, len(cs) + 1),
            cs,
            "-o",
            ms=2.5,
            lw=1.0,
            color=c_cycle[i % 3],
            label=labels[p].replace("\n", " "),
        )
        axa.axhline(pa["random_ref_mean_cos"], color=c_cycle[i % 3], ls=":", lw=0.8)
    axa.set_xlabel("Principal component index (k=50)")
    axa.set_ylabel(r"$\cos$(principal angle), base vs instruct map")
    axa.set_title("Output-subspace alignment (layer 19)")
    axa.set_ylim(-0.05, 1.05)
    axa.legend(fontsize=7)
    # (b) per-output-dim cosine histogram for the single-turn assistant pair
    hp = "S_assistant_chat" if "S_assistant_chat" in pairs else asst[0]
    ws = _ws_at(pairs[hp]["weight_space"])
    hist = ws.get("per_output_dim_cosine", {}).get("hist")
    if hist:
        edges = np.array(hist["bin_edges"])
        counts = np.array(hist["counts"])
        centers = (edges[:-1] + edges[1:]) / 2
        axb.bar(
            centers,
            counts,
            width=(edges[1] - edges[0]) * 0.9,
            color=paper_palette_role("primary"),
            edgecolor="white",
            linewidth=0.3,
            label="per-output-dim cosine",
        )
        axb.axvline(
            ws.get("random_map_vec_cosine_ref", 0.0),
            color=paper_palette_role("accent"),
            ls="--",
            lw=1.2,
            label="random-map ref",
        )
        axb.axvline(
            ws["per_output_dim_cosine"]["median"],
            color=paper_palette_role("neutral"),
            ls="-",
            lw=1.0,
            label="median",
        )
    axb.set_xlabel("Per-output-dim cosine (base vs instruct)")
    axb.set_ylabel("Count of output dimensions")
    axb.set_title(f"Map coefficient agreement\n({labels[hp].replace(chr(10), ' ')}, layer 19)")
    axb.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_825/crossmodel_map_similarity", dir="figures")
    plt.close(fig)
    print("[figures] wrote figures/issue_825/crossmodel_transfer_r2 + crossmodel_map_similarity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_results/issue_825/crossmodel_map_transfer")
    ap.add_argument("--dl-dir", default="data/issue_825/hf_dl/crossmodel")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    fig_dir = Path("figures/issue_825/crossmodel")

    if args.figures_only:
        results = json.loads(results_path.read_text())
    else:
        results = run_battery(Path(args.dl_dir), out)
        results_path.write_text(json.dumps(results, indent=2, default=float))
        print(f"[main] wrote {results_path}")

    if not args.no_figures:
        make_figures(results, fig_dir)


if __name__ == "__main__":
    main()
