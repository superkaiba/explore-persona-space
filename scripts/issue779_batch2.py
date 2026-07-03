#!/usr/bin/env python3
"""Issue #779 batch 2: the remaining CPU-runnable analyses, vectorized.

Five deliverables (run one at a time via ``--deliverable``; each writes/extends
its own JSON under ``eval_results/issue_779/`` and is committed independently):

  d1  Arm-level MLP h (c_last -> v(x), PCA-64 head) + MLP g (c_last -> label)
      per trait x mode (frozen layer) x arm {A: 5000 LMSYS 1-rollout targets;
      B: 2400 corpus 10-mean targets; C: natural concat}. Rig eval =
      within-condition Pearson (method_metrics, n_boot=1000, seed=0) + held-out
      recon R2 (single 80/20 split within the arm rows, test-own-mean).
      BATCHED trainer: all same-arm fits ride one bmm-batched AdamW loop
      (per-group early-stop snapshots), gated against fit_h.mlp_fit_predict.
      -> batch2_mlp_krr.json ("mlp" section) + batch2_mlp_arms.png.
  d2  Nystrom kernel-ridge rung (RBF, m=1024 landmarks, median-heuristic gamma,
      ridge on features, float64), h fit on Arm C rows, one per trait x mode.
      -> batch2_mlp_krr.json ("krr" section).
  d3  Scaling-curve EDGES, ridge-only fast path at the frozen layers:
      N_LMSYS axis {100..5000} at N_behavior=0 and N_behavior axis {50..2400}
      at N_LMSYS=0, K=5 subsamples/cell, h AND g: within-cond r + held-out
      recon R2 per cell. -> batch2_edges.json + batch2_edges_{h,g}.png.
  d4  Logit-lens interpretation of the least-predictable PCA directions
      (identity_baseline per_direction, system layer): 10 worst + 5 best
      rank<=200 directions per trait -> top-15 unembedding tokens each sign,
      cosine vs r_B (all traits), 3 top-|projection| corpus contexts each.
      -> batch2_logitlens.json (with a compact md table).
  d5  Final-token map (scalar): ridge LOCO c_last -> r2_last (final answer
      token projection on r_B) and c_last -> oracle (response-mean projection),
      per trait x mode, vs the fit-free pv_raw baseline.
      -> batch2_logitlens.json under key "final_token_map".

DOCUMENTED DEVIATIONS from the parent recipes (all also recorded in the JSONs):
  * d1 uses ONE fit per (group): trained on the 80% split; the SAME fit serves
    the rig eval and the 20% held-out recon read (2x fewer fits; the rig read
    is therefore at 0.8N -- the d3 edge curves quantify the N-sensitivity).
  * d1's PCA-64 target basis is computed by a covariance/Gram eigh
    (``pca_basis_eigh``) instead of ``robust_pca_basis``'s full SVD (the SVD is
    ~125 s per (n~6000, H=3584) call); subspace agreement is gated once.
  * d3 rig reads are POINT within-condition r (no bootstrap) -- the K=5
    subsample spread is the reported error bar.
  * d3 full-pool RIG cells are computed once (K collapses; identical data =>
    identical fit); full-pool HOLDOUT cells use K=2 splits (cost).
  * d5 within-condition r uses min_y_std=1e-9 (the target is a projection,
    not a 0-100 judge score, so the PV std>=1 rule does not apply).

Analysis-only: reads cached tensors; no training, no generation, no uploads.
Fail loud; NaN is reported, never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847) must bind BEFORE torch/numpy freeze their pools
# at import (tests/test_shared_vm_thread_caps.py); explicit launch-time env
# still wins (load_dotenv only setdefaults).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_common as C  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_raw,
    robust_pca_basis,
)
from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

LOG_PATH = Path("/tmp/issue779_batch2.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
    force=True,  # the issue779_* imports already configured the root logger
)
logger = logging.getLogger("issue779_batch2")

TRAITS = list(C.TRAITS)
H_DIM = C.EXPECTED_HIDDEN
N_LAYERS = C.EXPECTED_LAYERS
# Frozen read-out layers per trait x mode (step0 selection; task brief).
MODE_LAYER = {
    "system": {"evil": 14, "sycophancy": 26, "hallucination": 17},
    "many_shot": {"evil": 26, "sycophancy": 26, "hallucination": 27},
}
MODES = ("system", "many_shot")

CORPUS_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus")
COLLECT_DIR = (
    PROJECT_ROOT / "data" / "issue779_hfstage" / "issue779_monitoring" / "analysis_tensors"
)
LMSYS_LABELS_PATH = PROJECT_ROOT / "data" / "issue_779" / "lmsys_g_labels" / "lmsys_g_labels.json"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_779"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_779"

SPLIT_SEED = 0  # 80/20 arm split (d1/d2) + d3 cell subsample base
N_BOOT = 1000
PCA_K = 64


def _t(msg: str, t0: float) -> None:
    logger.info("[timing] %s: %.1f s", msg, time.time() - t0)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── data loading (mmap; slice one layer at a time; keep RSS bounded) ───────────


class Data:
    """Lazy mmap-backed loaders for the LMSYS bundle, corpus blobs, rigs."""

    def __init__(self) -> None:
        self._lmsys = None
        self._corpus: dict[str, dict] = {}
        self._labels_lmsys: dict[str, np.ndarray] | None = None
        self._labels_corpus: dict[str, np.ndarray] = {}
        self._cells: dict[str, list] = {}
        self._rb: dict[str, np.ndarray] = {}
        self._mats: dict[tuple[str, int], dict] = {}

    # -- LMSYS pass-B bundle (5000 contexts, single-rollout v_x) --
    def lmsys_bundle(self) -> dict:
        if self._lmsys is None:
            self._lmsys = torch.load(
                COLLECT_DIR / "pass_b" / "train_context_vectors.pt",
                weights_only=False,
                mmap=True,
                map_location="cpu",
            )
            assert self._lmsys["cx_last"].shape[1:] == (N_LAYERS, H_DIM)
        return self._lmsys

    def lmsys_layer(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.lmsys_bundle()
        col = b["layers"].index(layer)
        X = b["cx_last"][:, col, :].to(torch.float32).numpy()
        Y = b["v_x"][:, col, :].to(torch.float32).numpy()
        return X, Y

    # -- behavior corpus (2400 contexts x 10 rollouts) --
    def corpus_blob(self, trait: str) -> dict:
        if trait not in self._corpus:
            self._corpus[trait] = torch.load(
                CORPUS_DIR / f"{trait}_corpus.pt",
                weights_only=False,
                mmap=True,
                map_location="cpu",
            )
            blob = self._corpus[trait]
            n_ctx = blob["cx_last"].shape[0]
            nr = blob["n_rollouts"]
            assert blob["v_x"].shape[0] == n_ctx * nr, (blob["v_x"].shape, n_ctx, nr)
            vi = blob["vx_index"]
            assert all(vi[i] == (i // nr, i % nr) for i in range(0, len(vi), 997)), (
                "vx_index is not the expected (ctx, rollout) raster order"
            )
        return self._corpus[trait]

    def corpus_layer(self, trait: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """(X c_last (2400,H), Y = 10-rollout-mean v_x (2400,H)) at one layer."""
        blob = self.corpus_blob(trait)
        col = blob["layers"].index(layer)
        X = blob["cx_last"][:, col, :].to(torch.float32).numpy()
        n_ctx, nr = blob["cx_last"].shape[0], blob["n_rollouts"]
        vx = blob["v_x"][:, col, :].to(torch.float32).numpy()
        Y = vx.reshape(n_ctx, nr, H_DIM).mean(axis=1)
        return X, Y

    # -- g labels --
    def lmsys_labels(self, trait: str) -> np.ndarray:
        if self._labels_lmsys is None:
            with open(LMSYS_LABELS_PATH) as f:
                d = json.load(f)
            self._labels_lmsys = {
                t: np.array(
                    [np.nan if v is None else float(v) for v in d["labels_per_trait"][t]["labels"]],
                    dtype=np.float64,
                )
                for t in TRAITS
            }
            for t, arr in self._labels_lmsys.items():
                assert arr.shape == (5000,), (t, arr.shape)
        return self._labels_lmsys[trait]

    def corpus_labels(self, trait: str) -> np.ndarray:
        """Per-context mean judge score over valid rollouts (NaN if none)."""
        if trait not in self._labels_corpus:
            with open(CORPUS_DIR / f"{trait}_judge_scores.json") as f:
                d = json.load(f)
            n_ctx = self.corpus_blob(trait)["cx_last"].shape[0]
            out = np.full(n_ctx, np.nan)
            for ci_s, per_r in d["scores"].items():
                vals = [float(v) for v in per_r.values() if v is not None and np.isfinite(v)]
                if vals:
                    out[int(ci_s)] = float(np.mean(vals))
            self._labels_corpus[trait] = out
        return self._labels_corpus[trait]

    # -- eval rig --
    def rb(self, trait: str) -> np.ndarray:
        if trait not in self._rb:
            self._rb[trait] = S1._load_rb(COLLECT_DIR / "r_b", trait, N_LAYERS, H_DIM)
        return self._rb[trait]

    def cells(self, trait: str) -> list:
        if trait not in self._cells:
            self._cells[trait] = S1.load_eval_cells(COLLECT_DIR / "pass_a", trait)
        return self._cells[trait]

    def eval_mat(self, trait: str, layer: int) -> dict:
        key = (trait, layer)
        if key not in self._mats:
            self._mats[key] = S1.build_eval_matrix(self.cells(trait), layer, self.rb(trait))
        return self._mats[key]


def unique_trait_layers() -> list[tuple[str, int]]:
    """Unique (trait, frozen layer) pairs across both modes (5 pairs)."""
    seen = []
    for t in TRAITS:
        for mode in MODES:
            pair = (t, MODE_LAYER[mode][t])
            if pair not in seen:
                seen.append(pair)
    return seen


def modes_for(trait: str, layer: int) -> list[str]:
    return [m for m in MODES if MODE_LAYER[m][trait] == layer]


# ── shared numeric helpers ─────────────────────────────────────────────────────


def pca_basis_eigh(Y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA mean + top-k components via covariance/Gram eigh (float64).

    Deviation from ``robust_pca_basis`` (full SVD, ~125 s at n~6000 x H=3584):
    identical subspace up to sign/degeneracy, gated once by ``gate_pca_basis``.
    Returns (mu (H,), comps (k, H)) with unit-norm rows, descending variance.
    """
    Yc = torch.as_tensor(Y, dtype=torch.float64)
    mu = Yc.mean(0)
    Yc = Yc - mu
    n, h = Yc.shape
    if n >= h:
        cov = Yc.T @ Yc
        w, V = torch.linalg.eigh(cov)  # ascending
        comps = V[:, -k:].flip(-1).T  # (k, H)
    else:
        G = Yc @ Yc.T
        w, U = torch.linalg.eigh(G)
        w_top = torch.clamp(w[-k:].flip(0), min=1e-12)
        U_top = U[:, -k:].flip(-1)  # (n, k)
        comps = (Yc.T @ U_top / w_top.sqrt()).T  # (k, H)
    comps = comps / comps.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return mu.numpy(), comps.numpy()


def gate_pca_basis(rng: np.random.Generator) -> dict:
    """Subspace agreement of pca_basis_eigh vs robust_pca_basis (one-off gate)."""
    Y = rng.standard_normal((400, 300)) @ rng.standard_normal((300, 300))
    k = 16
    mu_a, comps_a = pca_basis_eigh(Y, k)
    mu_b, comps_b, _fb = robust_pca_basis(Y, k)
    mu_diff = float(np.max(np.abs(mu_a - mu_b)))
    # Projector Frobenius gap (basis-sign/rotation invariant).
    Pa = comps_a.T @ comps_a
    Pb = comps_b[:k].T @ comps_b[:k]
    proj_gap = float(np.linalg.norm(Pa - Pb) / np.sqrt(2 * k))
    assert mu_diff < 1e-9 and proj_gap < 1e-6, (mu_diff, proj_gap)
    return {"mu_max_abs_diff": mu_diff, "projector_rel_gap": proj_gap, "pass": True}


def ridge_fast_multi(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval_list: list[np.ndarray],
    *,
    lambdas: np.ndarray | None = None,
) -> list[np.ndarray]:
    """PR._ridge_fit_predict_fast with MULTIPLE eval sets from ONE Gram eigh.

    Identical math (standardize-X / center-Y / GCV-lambda / dual eigh); gated
    against the single-eval original by ``gate_ridge_fast_multi``.
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    ntr = Xtr.shape[0]

    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Ytr_c
    sqVtY = (VtY**2).sum(1)
    tot = float((Ytr_c**2).sum())
    best_lam, best_gcv = float(lambdas[0]), float("inf")
    for lam in lambdas:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = 1.0 / (w + best_lam)
    coef = filt[:, None] * VtY  # (ntr, D_out) eigen-space dual coefficients
    preds = []
    for Xev in X_eval_list:
        Xev_n = (torch.as_tensor(np.asarray(Xev), dtype=torch.float64) - xmu) / xsd
        KevV = (Xev_n @ Xtr_n.T) @ V
        preds.append((KevV @ coef + ymu).numpy())
    return preds


def gate_ridge_fast_multi(data: Data) -> dict:
    """(a) fast-vs-canonical ridge gate (PR's own gate, re-run once);
    (b) ridge_fast_multi reproduces PR._ridge_fit_predict_fast exactly."""
    rng = np.random.default_rng(SPLIT_SEED)
    X, Y = data.lmsys_layer(14)
    sub = rng.choice(X.shape[0], size=600, replace=False)
    Xg, Yg = X[sub].astype(np.float64), Y[sub].astype(np.float64)
    pred_slow = F.ridge_fit_predict(Xg[:500], Yg[:500], Xg[500:])
    pred_fast = PR._ridge_fit_predict_fast(Xg[:500], Yg[:500], Xg[500:])
    abs_diff = float(np.max(np.abs(pred_slow - pred_fast)))
    rel_diff = abs_diff / (float(np.max(np.abs(pred_slow))) + 1e-12)
    r2_slow = PR._pooled_r2(pred_slow, Yg[500:])
    r2_fast = PR._pooled_r2(pred_fast, Yg[500:])
    assert rel_diff < 1e-6 and abs(r2_slow - r2_fast) < 1e-6, (rel_diff, r2_slow, r2_fast)
    pred_multi = ridge_fast_multi(Xg[:500], Yg[:500], [Xg[500:]])[0]
    multi_diff = float(np.max(np.abs(pred_multi - pred_fast)))
    assert multi_diff < 1e-8, multi_diff
    logger.info(
        "ridge gates PASS: fast-vs-SVD rel=%.2e (R2 %.6f/%.6f), multi-vs-fast %.2e",
        rel_diff,
        r2_slow,
        r2_fast,
        multi_diff,
    )
    return {
        "fast_vs_svd_rel_diff": rel_diff,
        "fast_vs_svd_r2": [r2_slow, r2_fast],
        "multi_vs_fast_max_abs_diff": multi_diff,
        "pass": True,
    }


def heldout_recon_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Pooled R2 (SS_tot on the TEST set's own mean) + mean per-row cosine."""
    r2 = PR._pooled_r2(pred, true)
    cos = PR._per_context_cosine(pred, true)
    return {"r2": float(r2), "mean_cosine": float(np.mean(cos)), "n_test": len(true)}


def scalar_heldout_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Held-out R2 (test-own-mean) + Pearson r for a scalar target."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    fin = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[fin], true[fin]
    if len(true) < 3:
        return {"r2": float("nan"), "pearson": float("nan"), "n_test": len(true)}
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {"r2": float(r2), "pearson": M.overall_pearson(pred, true), "n_test": len(true)}


def split_80_20(n: int, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(fit_idx, test_idx): a single 80/20 split (test = first 20% of a perm)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_te = max(1, round(0.2 * n))
    return np.sort(perm[n_te:]), np.sort(perm[:n_te])


def mode_overall_r(x: np.ndarray, mat: dict, mode: str) -> float:
    sel = np.array([m == mode for m in mat["mode"]]) & np.isfinite(x) & np.isfinite(mat["y"])
    return M.overall_pearson(x[sel], mat["y"][sel]) if sel.sum() >= 3 else float("nan")


# ── d1: batched MLP trainer ────────────────────────────────────────────────────


@dataclass
class MLPFitGroup:
    """One fit-and-predict MLP problem (raw X/Y; the trainer standardizes/PCAs)."""

    key: tuple
    X: np.ndarray  # (n, d) float32 fit rows
    Y: np.ndarray  # (n, p_raw) float32 raw targets (p_raw may be 1 or H)
    pca_k: int  # reduce targets to top-k PCA dims when p_raw > pca_k


class MLPGroupResult:
    """Best-val params + preprocessing stats; predicts raw-space outputs."""

    def __init__(self, key, params, xmu, xsd, ymu, comps, p_g, epochs_ran, best_val):
        self.key = key
        self.params = params  # (W1, b1, W2, b2) torch float32, group slice
        self.xmu, self.xsd = xmu, xsd
        self.ymu, self.comps = ymu, comps
        self.p_g = p_g
        self.epochs_ran = epochs_ran
        self.best_val = best_val

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        Xn = (np.asarray(X_new, dtype=np.float32) - self.xmu) / self.xsd
        W1, b1, W2, b2 = self.params
        with torch.no_grad():
            h = torch.nn.functional.gelu(torch.from_numpy(Xn) @ W1.T + b1)
            out = (h @ W2[: self.p_g].T + b2[: self.p_g]).numpy()
        if self.comps is not None:
            return out @ self.comps + self.ymu
        return out + self.ymu


def batched_mlp_fit(
    groups: list[MLPFitGroup],
    *,
    hidden: int = 512,
    lr: float = 1e-3,
    wd: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 20,
    val_frac: float = 0.1,
    seed: int = 42,
    basis_fn=pca_basis_eigh,
    num_threads: int = 6,
) -> dict[tuple, MLPGroupResult]:
    """Train ALL groups' MLPs as one bmm-batched AdamW loop (per-group early stop).

    Per group, mirrors ``fit_h.mlp_fit_predict`` exactly: standardize X on the
    group's fit rows; PCA-``pca_k`` target head (skip for scalar); val split
    ``default_rng(seed).permutation(n)`` (first ``val_frac`` rows are val);
    init ``torch.manual_seed(seed)`` -> Sequential(Linear(d,hidden), GELU,
    Linear(hidden,p)); full-batch AdamW(lr, wd); early stop patience 20 on val
    MSE with best-state snapshot. Groups are padded to (n_max, p_max) with loss
    masks; a frozen group's parameters keep receiving wd decay but its BEST
    snapshot is what predicts, so results match the serial path (gated by
    ``gate_batched_mlp``).
    """
    torch.set_num_threads(int(num_threads))
    assert groups
    d_in = groups[0].X.shape[1]
    G = len(groups)
    prep = []
    for g in groups:
        assert g.X.shape[1] == d_in
        Xtr = np.asarray(g.X, dtype=np.float32)
        Ytr = np.asarray(g.Y, dtype=np.float32)
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]
        n = Xtr.shape[0]
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0) + 1e-6
        Xn = (Xtr - xmu) / xsd
        if Ytr.shape[1] <= g.pca_k:
            ymu = Ytr.mean(0)
            T = Ytr - ymu
            comps = None
        else:
            ymu64, comps = basis_fn(Ytr.astype(np.float64), g.pca_k)
            ymu = ymu64
            T = ((Ytr.astype(np.float64) - ymu64) @ comps.T).astype(np.float32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, round(val_frac * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        prep.append(
            {
                "n": n,
                "p": T.shape[1],
                "Xn": Xn,
                "T": T.astype(np.float32),
                "xmu": xmu,
                "xsd": xsd,
                "ymu": ymu,
                "comps": comps,
                "val_idx": val_idx,
                "tr_idx": tr_idx,
            }
        )
    p_max = max(pp["p"] for pp in prep)
    n_tr_max = max(len(pp["tr_idx"]) for pp in prep)
    n_val_max = max(len(pp["val_idx"]) for pp in prep)

    # Padded TRAIN rows only (the val rows ride a separate small tensor so the
    # post-step val forward costs ~val_frac of an epoch, not a full forward).
    Xp = torch.zeros((G, n_tr_max, d_in), dtype=torch.float32)
    Tp = torch.zeros((G, n_tr_max, p_max), dtype=torch.float32)
    wtr = torch.zeros((G, n_tr_max, p_max), dtype=torch.float32)
    Xv = torch.zeros((G, n_val_max, d_in), dtype=torch.float32)
    Tv = torch.zeros((G, n_val_max, p_max), dtype=torch.float32)
    wva = torch.zeros((G, n_val_max, p_max), dtype=torch.float32)
    denom_tr = torch.zeros(G, dtype=torch.float32)
    denom_val = torch.zeros(G, dtype=torch.float32)
    for gi, pp in enumerate(prep):
        p = pp["p"]
        ntr, nva = len(pp["tr_idx"]), len(pp["val_idx"])
        Xp[gi, :ntr] = torch.from_numpy(pp["Xn"][pp["tr_idx"]])
        Tp[gi, :ntr, :p] = torch.from_numpy(pp["T"][pp["tr_idx"]])
        wtr[gi, :ntr, :p] = 1.0
        Xv[gi, :nva] = torch.from_numpy(pp["Xn"][pp["val_idx"]])
        Tv[gi, :nva, :p] = torch.from_numpy(pp["T"][pp["val_idx"]])
        wva[gi, :nva, :p] = 1.0
        denom_tr[gi] = float(ntr * p)
        denom_val[gi] = float(nva * p)

    # Per-group init: reproduce mlp_fit_predict's torch.manual_seed(seed) + net.
    W1 = torch.empty((G, hidden, d_in))
    b1 = torch.empty((G, hidden))
    W2 = torch.zeros((G, p_max, hidden))
    b2 = torch.zeros((G, p_max))
    for gi, pp in enumerate(prep):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, pp["p"])
        )
        W1[gi] = net[0].weight.detach()
        b1[gi] = net[0].bias.detach()
        W2[gi, : pp["p"]] = net[2].weight.detach()
        b2[gi, : pp["p"]] = net[2].bias.detach()
    for w in (W1, b1, W2, b2):
        w.requires_grad_(True)
    opt = torch.optim.AdamW([W1, b1, W2, b2], lr=lr, weight_decay=wd)

    best_val = torch.full((G,), float("inf"))
    bad = torch.zeros(G, dtype=torch.long)
    frozen = torch.zeros(G, dtype=torch.bool)
    best_state = [None] * G
    epochs_ran = np.zeros(G, dtype=int)
    active_f = torch.ones(G)
    t_loop = time.time()
    for ep in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        h1 = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), Xp, W1.transpose(1, 2)))
        out = torch.baddbmm(b2.unsqueeze(1), h1, W2.transpose(1, 2))
        loss_pg = (((out - Tp) ** 2) * wtr).sum(dim=(1, 2)) / denom_tr
        loss = (loss_pg * active_f).sum()
        loss.backward()
        opt.step()
        # Val forward AFTER the step (matches the serial reference's semantics),
        # on the val rows only (~val_frac of an epoch's forward cost).
        with torch.no_grad():
            h1e = torch.nn.functional.gelu(torch.baddbmm(b1.unsqueeze(1), Xv, W1.transpose(1, 2)))
            oute = torch.baddbmm(b2.unsqueeze(1), h1e, W2.transpose(1, 2))
            val_pg = (((oute - Tv) ** 2) * wva).sum(dim=(1, 2)) / denom_val
        improved = (val_pg < best_val - 1e-6) & (~frozen)
        if improved.any():
            for gi in torch.nonzero(improved).ravel().tolist():
                best_state[gi] = (
                    W1[gi].detach().clone(),
                    b1[gi].detach().clone(),
                    W2[gi].detach().clone(),
                    b2[gi].detach().clone(),
                )
        best_val = torch.where(improved, val_pg, best_val)
        bad = torch.where(improved, torch.zeros_like(bad), bad + (~frozen).long())
        frozen |= (bad >= patience) & (~frozen)
        active_f = (~frozen).float()
        epochs_ran[(~frozen).numpy()] = ep + 1
        if frozen.all():
            break
        if ep % 20 == 0:
            logger.info(
                "[mlp-batch] epoch %d: mean train loss %.5f, %d/%d groups active (%.2f s/ep)",
                ep,
                float(loss_pg.detach().mean()),
                int((~frozen).sum()),
                G,
                (time.time() - t_loop) / (ep + 1),
            )

    results = {}
    for gi, (g, pp) in enumerate(zip(groups, prep, strict=True)):
        state = best_state[gi]
        if state is None:  # never improved: use final params (mirrors serial)
            state = (W1[gi].detach(), b1[gi].detach(), W2[gi].detach(), b2[gi].detach())
        results[g.key] = MLPGroupResult(
            g.key,
            state,
            pp["xmu"],
            pp["xsd"],
            pp["ymu"],
            pp["comps"],
            pp["p"],
            int(epochs_ran[gi]),
            float(best_val[gi]),
        )
    return results


def gate_batched_mlp(num_threads: int = 6) -> dict:
    """G=1 batched trainer vs fit_h.mlp_fit_predict on synthetic data.

    Uses robust_pca_basis in BOTH paths so the gate isolates the trainer.
    bmm-vs-addmm reduction order compounds over epochs, so the tolerance is
    statistical (prediction R2 agreement), not bitwise.
    """
    rng = np.random.default_rng(7)
    n, d, p_out = 260, 96, 128
    W_true = rng.standard_normal((d, p_out)) * 0.3
    X = rng.standard_normal((n + 60, d)).astype(np.float32)
    Y = (
        np.tanh(X @ W_true.astype(np.float32)) + 0.05 * rng.standard_normal((n + 60, p_out))
    ).astype(np.float32)
    Xtr, Ytr, Xev = X[:n], Y[:n], X[n:]
    # Cap BOTH paths at the same 60 epochs: equivalence over 60 epochs is just
    # as diagnostic and keeps the gate cheap on a contended VM.
    ref = F.mlp_fit_predict(Xtr, Ytr, Xev, pca_k=8, num_threads=num_threads, max_epochs=60)
    res = batched_mlp_fit(
        [MLPFitGroup(("gate",), Xtr, Ytr, 8)],
        basis_fn=lambda Yv, k: robust_pca_basis(Yv, k)[:2],
        num_threads=num_threads,
        max_epochs=60,
    )[("gate",)]
    pred = res.predict(Xev)
    max_abs = float(np.max(np.abs(pred - ref)))
    scale = float(np.std(ref))
    agree_r2 = PR._pooled_r2(pred, ref)
    assert agree_r2 > 0.995, (agree_r2, max_abs)
    logger.info(
        "batched-MLP gate PASS: pred-agreement R2 %.6f (max|diff| %.4f, ref sd %.3f, epochs %d)",
        agree_r2,
        max_abs,
        scale,
        res.epochs_ran,
    )
    return {"agreement_r2": float(agree_r2), "max_abs_diff": max_abs, "ref_sd": scale, "pass": True}


ARMS = ("A_lmsys", "B_behavior", "C_concat")


def _arm_rows(data: Data, arm: str, trait: str | None, layer: int):
    """(X, Y, y_label) rows for an arm at a layer. For A_lmsys with trait=None
    the labels are per-trait (resolved by the caller)."""
    if arm == "A_lmsys":
        X, Y = data.lmsys_layer(layer)
        lab = data.lmsys_labels(trait) if trait else None
        return X, Y, lab
    if arm == "B_behavior":
        X, Y = data.corpus_layer(trait, layer)
        return X, Y, data.corpus_labels(trait)
    if arm == "C_concat":
        Xa, Ya = data.lmsys_layer(layer)
        Xb, Yb = data.corpus_layer(trait, layer)
        lab = np.concatenate([data.lmsys_labels(trait), data.corpus_labels(trait)])
        return np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb]), lab
    raise ValueError(arm)


def run_d1(data: Data, *, smoke: bool = False) -> None:
    out_path = OUT_DIR / ("batch2_mlp_krr_smoke.json" if smoke else "batch2_mlp_krr.json")
    results = _load_json(out_path) if out_path.exists() else {}
    t0 = time.time()
    gates = {
        "batched_mlp": gate_batched_mlp(),
        "pca_basis": gate_pca_basis(np.random.default_rng(3)),
    }
    _t("d1 gates", t0)
    cap = 400 if smoke else None
    max_epochs = 12 if smoke else 300
    n_boot = 50 if smoke else N_BOOT

    mlp: dict = {"traits": {t: {} for t in TRAITS}, "gates": gates, "fit_info": {}}
    pairs = unique_trait_layers()
    h_lmsys_layers = sorted({layer for _t_, layer in pairs})

    for arm in ARMS:
        ta = time.time()
        groups: list[MLPFitGroup] = []
        meta: dict[tuple, dict] = {}
        # h groups: A_lmsys is trait-agnostic (keyed by layer); B/C per (trait, layer).
        if arm == "A_lmsys":
            h_keys = [("h", arm, None, layer) for layer in h_lmsys_layers]
        else:
            h_keys = [("h", arm, t, layer) for t, layer in pairs]
        for key in h_keys:
            _kind, _arm, t, layer = key
            X, Y, _ = _arm_rows(data, arm, t, layer)
            if cap:
                X, Y = X[:cap], Y[:cap]
            fit_idx, test_idx = split_80_20(len(X))
            groups.append(MLPFitGroup(key, X[fit_idx], Y[fit_idx], PCA_K))
            meta[key] = {"X": X, "Y": Y, "fit_idx": fit_idx, "test_idx": test_idx}
        # g groups: per (trait, layer); valid-label rows only.
        for t, layer in pairs:
            key = ("g", arm, t, layer)
            X, _Y, lab = _arm_rows(data, arm, t, layer)
            if cap:
                X, lab = X[:cap], lab[:cap]
            valid = np.isfinite(lab)
            Xv, yv = X[valid], lab[valid].astype(np.float32)
            fit_idx, test_idx = split_80_20(len(Xv))
            groups.append(MLPFitGroup(key, Xv[fit_idx], yv[fit_idx][:, None], PCA_K))
            meta[key] = {"X": Xv, "y": yv, "fit_idx": fit_idx, "test_idx": test_idx}

        fits = batched_mlp_fit(groups, max_epochs=max_epochs)
        _t(f"d1 arm {arm} batched fit ({len(groups)} groups)", ta)

        for key, res in fits.items():
            kind, _arm, t, layer = key
            mm = meta[key]
            mlp["fit_info"][str(key)] = {
                "epochs_ran": res.epochs_ran,
                "best_val_mse": res.best_val,
                "n_fit": len(mm["fit_idx"]),
                "n_test": len(mm["test_idx"]),
            }
            if kind == "h":
                pred_te = res.predict(mm["X"][mm["test_idx"]])
                held = heldout_recon_metrics(pred_te, mm["Y"][mm["test_idx"]])
                consumers = pairs if t is None else [(t, layer)]
                for ct, cl in consumers:
                    if cl != layer:
                        continue
                    rb_l = data.rb(ct)[layer]
                    mat = data.eval_mat(ct, layer)
                    prof = res.predict(mat["c_last"])
                    x_dot = F.dot_readout(prof, rb_l)
                    x_cos = F.cosine_readout(prof, rb_l)
                    mm_dot = S1.method_metrics(x_dot, mat, n_boot=n_boot, seed=0)
                    mm_cos = S1.method_metrics(x_cos, mat, n_boot=n_boot, seed=0)
                    for mode in modes_for(ct, layer):
                        node = mlp["traits"][ct].setdefault(mode, {}).setdefault(arm, {})
                        node["h"] = {
                            "layer": layer,
                            "within_r_dot": mm_dot[mode],
                            "within_r_cos": mm_cos[mode],
                            "overall_r_mode_dot": mode_overall_r(x_dot, mat, mode),
                            "heldout_recon": held,
                            "n_fit": len(mm["fit_idx"]),
                        }
            else:
                pred_te = res.predict(mm["X"][mm["test_idx"]])[:, 0]
                held = scalar_heldout_metrics(pred_te, mm["y"][mm["test_idx"]])
                mat = data.eval_mat(t, layer)
                x_g = res.predict(mat["c_last"])[:, 0]
                mm_g = S1.method_metrics(x_g, mat, n_boot=n_boot, seed=0)
                for mode in modes_for(t, layer):
                    node = mlp["traits"][t].setdefault(mode, {}).setdefault(arm, {})
                    node["g"] = {
                        "layer": layer,
                        "within_r": mm_g[mode],
                        "overall_r_mode": mode_overall_r(x_g, mat, mode),
                        "heldout": held,
                        "n_fit": len(mm["fit_idx"]),
                    }
        # checkpoint after each arm
        results["mlp"] = mlp
        results["metadata"] = C.reproducibility_metadata(
            {"script": "issue779_batch2 d1", "smoke": smoke}
        )
        results.setdefault("deviations", []).clear()
        results["deviations"].extend(
            [
                "d1: one fit per group on the 80% split serves BOTH the rig eval and the "
                "20% held-out recon (rig read at 0.8N).",
                "d1: PCA-64 target basis via covariance/Gram eigh (pca_basis_eigh), not "
                "robust_pca_basis full SVD; subspace gated.",
                "d1: batched bmm AdamW trainer (per-group early-stop snapshots) gated vs "
                "fit_h.mlp_fit_predict at prediction-agreement R2 > 0.995.",
            ]
        )
        C.write_json_atomic(out_path, results)
        _t(f"d1 arm {arm} total", ta)
    _t("d1 total", t0)
    make_d1_figure(
        results, FIG_DIR / ("batch2_mlp_arms_smoke.png" if smoke else "batch2_mlp_arms.png")
    )


def make_d1_figure(results: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:  # style-only fallback
        logger.warning("paper style unavailable (%s)", e)
    arm_headline = OUT_DIR / "arm_headline.json"
    ridge = None
    if arm_headline.exists():
        try:
            ridge = _load_json(arm_headline)
        except Exception as e:
            logger.warning("arm_headline.json unreadable (%s)", e)
    mlp = results["mlp"]["traits"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="tight")
    for row, kind in enumerate(("h", "g")):
        for col, mode in enumerate(MODES):
            ax = axes[row][col]
            labels, arm_vals = [], {a: [] for a in ARMS}
            for t in TRAITS:
                labels.append(t)
                for a in ARMS:
                    node = mlp.get(t, {}).get(mode, {}).get(a, {}).get(kind)
                    if node is None:
                        arm_vals[a].append(np.nan)
                    elif kind == "h":
                        arm_vals[a].append(node["within_r_dot"]["point"])
                    else:
                        arm_vals[a].append(node["within_r"]["point"])
            xs = np.arange(len(labels))
            width = 0.25
            for i, a in enumerate(ARMS):
                ax.bar(xs + (i - 1) * width, arm_vals[a], width, label=a)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylabel("within-condition r")
            ax.set_title(f"MLP {kind} — {mode}")
            ax.axhline(0.0, color="gray", lw=0.5)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    if ridge is not None:
        fig.suptitle("issue 779 batch2: arm-level MLP (ridge companion: arm_headline.json)")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


# ── d2: Nystrom kernel ridge ───────────────────────────────────────────────────


def nystrom_features(
    X: np.ndarray, landmarks: np.ndarray, gamma: float, eig_floor: float = 1e-10
) -> np.ndarray:
    """Phi = K_nm @ K_mm^{-1/2} (float64), the standard Nystrom feature map."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Zt = torch.as_tensor(landmarks, dtype=torch.float64)
    d_nm = torch.cdist(Xt, Zt) ** 2
    K_nm = torch.exp(-gamma * d_nm)
    d_mm = torch.cdist(Zt, Zt) ** 2
    K_mm = torch.exp(-gamma * d_mm)
    w, V = torch.linalg.eigh(K_mm)
    w = torch.clamp(w, min=eig_floor)
    inv_sqrt = V @ torch.diag(w.rsqrt()) @ V.T
    return (K_nm @ inv_sqrt).numpy()


def median_heuristic_gamma(X: np.ndarray, rng: np.random.Generator, n_sub: int = 2000) -> float:
    sub = X[rng.choice(len(X), size=min(n_sub, len(X)), replace=False)]
    St = torch.as_tensor(sub, dtype=torch.float64)
    d2 = (torch.cdist(St, St) ** 2).numpy()
    off = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(off))
    assert med > 0, med
    return 1.0 / med


def run_d2(data: Data, *, smoke: bool = False) -> None:
    out_path = OUT_DIR / ("batch2_mlp_krr_smoke.json" if smoke else "batch2_mlp_krr.json")
    # d2 normally appends to d1's file, but the KRR rung is data-independent of
    # the MLP section, so a missing file (d1 routed elsewhere) starts fresh.
    results = _load_json(out_path) if out_path.exists() else {}
    t0 = time.time()
    m_landmarks = 128 if smoke else 1024
    n_boot = 50 if smoke else N_BOOT
    cap = 600 if smoke else None
    krr: dict = {
        "traits": {t: {} for t in TRAITS},
        "m_landmarks": m_landmarks,
        "recipe": "RBF Nystrom K_nm @ K_mm^{-1/2}; GCV ridge on features "
        "(fit_h.ridge_fit_predict); gamma = 1/median sq-dist",
    }
    for t, layer in unique_trait_layers():
        tt = time.time()
        X, Y, _ = _arm_rows(data, "C_concat", t, layer)
        if cap:
            X, Y = X[:cap], Y[:cap]
        X64 = X.astype(np.float64)
        rng = np.random.default_rng(SPLIT_SEED)
        gamma = median_heuristic_gamma(X64, np.random.default_rng(1))
        rb_l = data.rb(t)[layer]
        mat = data.eval_mat(t, layer)
        fit_idx, test_idx = split_80_20(len(X64))

        # full-N fit -> rig reads
        lm_full = X64[rng.choice(len(X64), size=min(m_landmarks, len(X64)), replace=False)]
        Phi = nystrom_features(X64, lm_full, gamma)
        Phi_ev = nystrom_features(np.asarray(mat["c_last"], dtype=np.float64), lm_full, gamma)
        prof_ev = F.ridge_fit_predict(Phi, Y.astype(np.float64), Phi_ev)
        x_dot = F.dot_readout(prof_ev, rb_l)
        x_cos = F.cosine_readout(prof_ev, rb_l)
        mm_dot = S1.method_metrics(x_dot, mat, n_boot=n_boot, seed=0)
        mm_cos = S1.method_metrics(x_cos, mat, n_boot=n_boot, seed=0)

        # 80/20 fit -> held-out recon
        rng2 = np.random.default_rng(SPLIT_SEED + 1)
        lm_fit = X64[fit_idx][
            rng2.choice(len(fit_idx), size=min(m_landmarks, len(fit_idx)), replace=False)
        ]
        Phi_tr = nystrom_features(X64[fit_idx], lm_fit, gamma)
        Phi_te = nystrom_features(X64[test_idx], lm_fit, gamma)
        pred_te = F.ridge_fit_predict(Phi_tr, Y[fit_idx].astype(np.float64), Phi_te)
        held = heldout_recon_metrics(pred_te, Y[test_idx].astype(np.float64))

        for mode in modes_for(t, layer):
            krr["traits"][t][mode] = {
                "layer": layer,
                "gamma": gamma,
                "n_fit_full": len(X64),
                "within_r_dot": mm_dot[mode],
                "within_r_cos": mm_cos[mode],
                "overall_r_mode_dot": mode_overall_r(x_dot, mat, mode),
                "heldout_recon": held,
            }
        results["krr"] = krr
        C.write_json_atomic(out_path, results)
        _t(f"d2 krr {t} L{layer}", tt)
    _t("d2 total", t0)


# ── d3: scaling-curve edges ────────────────────────────────────────────────────

LMSYS_GRID = (100, 250, 500, 1000, 2000, 5000)
BEH_GRID = (50, 100, 250, 500, 1000, 2400)
K_SUB = 5
K_FULL_HOLDOUT = 2


def _d3_cell_seed(axis_i: int, n_i: int, k: int) -> int:
    return SPLIT_SEED + 100_000 * (axis_i + 1) + 1000 * n_i + k


def _d3_point_within_r(x: np.ndarray, mat: dict, mode: str) -> dict:
    cx, cy = S1._group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
    cx2, cy2 = [], []
    for xi, yi in zip(cx, cy, strict=True):
        fin = np.isfinite(xi) & np.isfinite(yi)
        if fin.sum() >= 3:
            cx2.append(xi[fin])
            cy2.append(yi[fin])
    res = M.within_condition_pearson(cx2, cy2)
    return {"point": res["r"], "n_conditions": res["n_conditions"]}


def _d3_unit_cells(
    data: Data,
    results: dict,
    *,
    axis_name: str,
    axis_i: int,
    grid,
    k_sub: int,
    kind: str,
    source: str,
    trait: str | None,
    layer: int,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    consumers: list[tuple[str, str]],
) -> None:
    """One (axis, kind, source, trait?, layer) unit: all (N, k) cells, in place."""
    unit = f"{axis_name}|{kind}|{source}|{trait}|{layer}"
    tu = time.time()
    n_pool = len(X_pool)
    rb_by_trait = {ct: data.rb(ct)[layer] for ct, _m in consumers}
    cells_out = results[axis_name][kind]
    for n_i, n_grid in enumerate(grid):
        n_take = min(n_grid, n_pool)
        full_pool = n_take >= n_pool
        k_rig = 1 if full_pool else k_sub
        k_hold = min(K_FULL_HOLDOUT, k_sub) if full_pool else k_sub
        for k in range(max(k_rig, k_hold)):
            rng = np.random.default_rng(_d3_cell_seed(axis_i, n_i, k))
            idx = rng.choice(n_pool, size=n_take, replace=False)
            Xc, Yc = X_pool[idx].astype(np.float64), Y_pool[idx].astype(np.float64)
            cell = {
                "unit": unit,
                "kind": kind,
                "source": source,
                "trait": trait,
                "layer": layer,
                "n": int(n_grid),
                "n_used": int(n_take),
                "k": int(k),
                "full_pool": bool(full_pool),
            }
            if k < k_rig:
                rig = {}
                if kind == "h":
                    eval_sets, keys = [], []
                    for ct, mode in consumers:
                        mat = data.eval_mat(ct, layer)
                        eval_sets.append(np.asarray(mat["c_last"], dtype=np.float64))
                        keys.append((ct, mode))
                    preds = ridge_fast_multi(Xc, Yc, eval_sets)
                    for (ct, mode), prof in zip(keys, preds, strict=True):
                        mat = data.eval_mat(ct, layer)
                        x_dot = F.dot_readout(prof, rb_by_trait[ct])
                        x_cos = F.cosine_readout(prof, rb_by_trait[ct])
                        rig[f"{ct}|{mode}"] = {
                            "dot": _d3_point_within_r(x_dot, mat, mode),
                            "cos": _d3_point_within_r(x_cos, mat, mode),
                        }
                else:
                    ct = trait
                    mat = data.eval_mat(ct, layer)
                    pred = ridge_fast_multi(Xc, Yc, [np.asarray(mat["c_last"], dtype=np.float64)])[
                        0
                    ][:, 0]
                    for mode in [m for c2, m in consumers if c2 == ct]:
                        rig[f"{ct}|{mode}"] = _d3_point_within_r(pred, mat, mode)
                cell["rig"] = rig
            if k < k_hold:
                fit_idx, test_idx = split_80_20(n_take, seed=_d3_cell_seed(axis_i, n_i, k) + 7)
                pred_te = ridge_fast_multi(Xc[fit_idx], Yc[fit_idx], [Xc[test_idx]])[0]
                if kind == "h":
                    cell["heldout"] = heldout_recon_metrics(pred_te, Yc[test_idx])
                else:
                    cell["heldout"] = scalar_heldout_metrics(pred_te[:, 0], Yc[test_idx][:, 0])
            cells_out.append(cell)
        logger.info("[d3] %s N=%d done (%.1f s)", unit, n_grid, time.time() - tu)
    _t(f"d3 unit {unit}", tu)


def _d3_units(data: Data, lmsys_grid, beh_grid) -> list[dict]:
    """The (axis, kind, source, trait?, layer) unit specs, in run order."""
    pairs = unique_trait_layers()
    units = []
    for layer in sorted({layer for _t_, layer in pairs}):
        consumers = [(ct, m) for ct, la in pairs for m in modes_for(ct, la) if la == layer]
        units.append(
            dict(
                axis_name="lmsys_axis",
                axis_i=0,
                grid=lmsys_grid,
                kind="h",
                source="lmsys",
                trait=None,
                layer=layer,
                consumers=consumers,
            )
        )
    for t, layer in pairs:
        units.append(
            dict(
                axis_name="lmsys_axis",
                axis_i=0,
                grid=lmsys_grid,
                kind="g",
                source="lmsys",
                trait=t,
                layer=layer,
                consumers=[(t, m) for m in modes_for(t, layer)],
            )
        )
    for t, layer in pairs:
        cons = [(t, m) for m in modes_for(t, layer)]
        units.append(
            dict(
                axis_name="behavior_axis",
                axis_i=1,
                grid=beh_grid,
                kind="h",
                source="behavior",
                trait=t,
                layer=layer,
                consumers=cons,
            )
        )
        units.append(
            dict(
                axis_name="behavior_axis",
                axis_i=1,
                grid=beh_grid,
                kind="g",
                source="behavior",
                trait=t,
                layer=layer,
                consumers=cons,
            )
        )
    return units


def _d3_unit_pools(data: Data, u: dict) -> tuple[np.ndarray, np.ndarray]:
    """(X_pool, Y_pool) for a d3 unit spec (labels filtered to valid rows for g)."""
    if u["source"] == "lmsys":
        X_pool, Y_pool = data.lmsys_layer(u["layer"])
        if u["kind"] == "g":
            lab = data.lmsys_labels(u["trait"])
            valid = np.isfinite(lab)
            return X_pool[valid], lab[valid][:, None]
        return X_pool, Y_pool
    X_pool, Y_pool = data.corpus_layer(u["trait"], u["layer"])
    if u["kind"] == "g":
        lab = data.corpus_labels(u["trait"])
        valid = np.isfinite(lab)
        return X_pool[valid], lab[valid][:, None]
    return X_pool, Y_pool


def run_d3(data: Data, *, smoke: bool = False) -> None:
    out_path = OUT_DIR / ("batch2_edges_smoke.json" if smoke else "batch2_edges.json")
    if out_path.exists():
        results = _load_json(out_path)
    else:
        results = {
            "lmsys_axis": {"h": [], "g": []},
            "behavior_axis": {"h": [], "g": []},
            "grids": {"lmsys": list(LMSYS_GRID), "behavior": list(BEH_GRID), "k": K_SUB},
            "done_units": [],
            "notes": [
                "rig read = POINT within-condition r (no bootstrap); "
                "K subsample spread is the error bar",
                "full-pool rig cells computed once (identical data); "
                "full-pool holdout uses K=2 splits",
                "g cells subsample from the valid-label pool only",
                "holdout recon = 80/20 split WITHIN the cell rows (test-own-mean R2)",
            ],
        }
    t0 = time.time()
    results["gates_ridge"] = gate_ridge_fast_multi(data)
    lmsys_grid = (100, 250) if smoke else LMSYS_GRID
    beh_grid = (50, 100) if smoke else BEH_GRID
    k_sub = 2 if smoke else K_SUB

    for u in _d3_units(data, lmsys_grid, beh_grid):
        unit = f"{u['axis_name']}|{u['kind']}|{u['source']}|{u['trait']}|{u['layer']}"
        if unit in results["done_units"]:
            logger.info("[d3] skip done unit %s", unit)
            continue
        X_pool, Y_pool = _d3_unit_pools(data, u)
        _d3_unit_cells(data, results, k_sub=k_sub, X_pool=X_pool, Y_pool=Y_pool, **u)
        results["done_units"].append(unit)
        results["metadata"] = C.reproducibility_metadata(
            {"script": "issue779_batch2 d3", "smoke": smoke}
        )
        C.write_json_atomic(out_path, results)
    _t("d3 total", t0)
    for kind in ("h", "g"):
        make_d3_figure(
            results,
            kind,
            FIG_DIR / (f"batch2_edges_{kind}_smoke.png" if smoke else f"batch2_edges_{kind}.png"),
        )


def make_d3_figure(results: dict, kind: str, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper style unavailable (%s)", e)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="tight")
    for col, axis_name in enumerate(("lmsys_axis", "behavior_axis")):
        cells = results[axis_name][kind]
        # rig within-r curves per (trait, mode)
        ax = axes[0][col]
        series: dict[str, dict[int, list[float]]] = {}
        for cell in cells:
            if "rig" not in cell:
                continue
            for rig_key, val in cell["rig"].items():
                pt = val["dot"]["point"] if kind == "h" else val["point"]
                series.setdefault(rig_key, {}).setdefault(cell["n_used"], []).append(pt)
        for rig_key in sorted(series):
            ns = sorted(series[rig_key])
            mean = [float(np.nanmean(series[rig_key][n])) for n in ns]
            sd = [float(np.nanstd(series[rig_key][n])) for n in ns]
            ax.errorbar(ns, mean, yerr=sd, marker="o", ms=3, lw=1, capsize=2, label=rig_key)
        ax.set_xscale("log")
        ax.set_xlabel("N train rows")
        ax.set_ylabel("within-condition r" + (" (dot)" if kind == "h" else ""))
        ax.set_title(f"{kind}: rig read — {axis_name}")
        ax.legend(fontsize=7)
        # held-out recon curves
        ax = axes[1][col]
        hseries: dict[str, dict[int, list[float]]] = {}
        for cell in cells:
            if "heldout" not in cell:
                continue
            key = f"{cell['trait']}|L{cell['layer']}" if cell["trait"] else f"L{cell['layer']}"
            hseries.setdefault(key, {}).setdefault(cell["n_used"], []).append(cell["heldout"]["r2"])
        for key in sorted(hseries):
            ns = sorted(hseries[key])
            mean = [float(np.nanmean(hseries[key][n])) for n in ns]
            sd = [float(np.nanstd(hseries[key][n])) for n in ns]
            ax.errorbar(ns, mean, yerr=sd, marker="s", ms=3, lw=1, capsize=2, label=key)
        ax.set_xscale("log")
        ax.set_xlabel("N train rows")
        ax.set_ylabel("held-out R2 (test-own-mean)")
        ax.set_title(f"{kind}: held-out recon — {axis_name}")
        ax.legend(fontsize=7)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", fig_path)


# ── d4: logit-lens interpretation of the least-predictable directions ─────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
ROLLOUTS_HF_PREFIX = "issue779_monitoring/training-source-ablation-hg/behavior_corpus"
QWEN_SNAPSHOT_GLOB = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
)


def _qwen_snapshot() -> Path:
    snaps = sorted(QWEN_SNAPSHOT_GLOB.glob("*"))
    assert snaps, f"no local Qwen2.5-7B-Instruct snapshot under {QWEN_SNAPSHOT_GLOB}"
    return snaps[-1]


def load_logit_lens() -> tuple[torch.Tensor, torch.Tensor, object]:
    """(W_U (V,H) float32, final-RMSNorm weight (H,), tokenizer) — CPU only."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    snap = _qwen_snapshot()
    with open(snap / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    shard_lm = snap / wmap["lm_head.weight"]
    shard_norm = snap / wmap["model.norm.weight"]
    with safe_open(str(shard_lm), framework="pt") as f:
        W_U = f.get_tensor("lm_head.weight").to(torch.float32)
    with safe_open(str(shard_norm), framework="pt") as f:
        norm_w = f.get_tensor("model.norm.weight").to(torch.float32)
    assert W_U.shape[1] == H_DIM and norm_w.shape == (H_DIM,), (W_U.shape, norm_w.shape)
    tok = AutoTokenizer.from_pretrained(str(snap))
    return W_U, norm_w, tok


def _top_tokens(W_U: torch.Tensor, norm_w: torch.Tensor, tok, d: np.ndarray, k: int = 15) -> dict:
    """Logit-lens top-k tokens for +d and -d (RMSNorm gamma applied elementwise)."""
    v = torch.from_numpy(np.asarray(d, dtype=np.float32)) * norm_w
    logits = W_U @ v
    top_pos = torch.topk(logits, k)
    top_neg = torch.topk(-logits, k)

    def _fmt(idx, vals):
        return [
            {"token": tok.decode([int(i)]), "logit": float(s)}
            for i, s in zip(idx.tolist(), vals.tolist(), strict=True)
        ]

    return {
        "pos": _fmt(top_pos.indices, top_pos.values),
        "neg": _fmt(top_neg.indices, -top_neg.values),
    }


def _rollouts_text(trait: str) -> dict:
    """Corpus rollout text (ctx_idx -> {question, responses, persona_idx}) from HF."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        HF_DATA_REPO, f"{ROLLOUTS_HF_PREFIX}/{trait}_rollouts.json", repo_type="dataset"
    )
    with open(p) as f:
        return json.load(f)["rollouts"]


def run_d4(data: Data, *, smoke: bool = False) -> None:
    out_path = OUT_DIR / ("batch2_logitlens_smoke.json" if smoke else "batch2_logitlens.json")
    results = _load_json(out_path) if out_path.exists() else {}
    t0 = time.time()
    with open(OUT_DIR / "identity_baseline.json") as f:
        ib = json.load(f)
    per_dir = ib["per_direction"]
    results["gates_ridge"] = gate_ridge_fast_multi(data)
    W_U, norm_w, tok = load_logit_lens()
    _t("d4 load W_U + tokenizer", t0)

    n_worst, n_best = (3, 2) if smoke else (10, 5)
    bundle = data.lmsys_bundle()
    n_ctx = bundle["cx_last"].shape[0]
    test_idx = PR._cv_folds(n_ctx, ib["A_ladder"].get("n_folds", 5), ib["metadata"]["seed"])[0]
    md_lines = [
        "| trait | rank | R2 | var-share | top +tokens | top -tokens | cos r_B |",
        "|---|---|---|---|---|---|---|",
    ]
    results.setdefault("directions", {})
    for t in TRAITS:
        tt = time.time()
        layer = MODE_LAYER["system"][t]
        pd_t = per_dir[t]
        assert pd_t["read_out_layer"] == layer, (t, pd_t["read_out_layer"], layer)
        ranks = np.array(pd_t["ranks_evaluated"])
        r2 = np.array(pd_t["r2_by_rank"], dtype=np.float64)
        share = np.array(pd_t["variance_share_by_rank"], dtype=np.float64)
        lead = (ranks <= 200) & np.isfinite(r2)
        order = np.argsort(r2[lead])
        lead_idx = np.where(lead)[0]
        worst_sel = lead_idx[order[:n_worst]]
        best_sel = lead_idx[order[::-1][:n_best]]

        # Recompute the fold-0 PCA basis + ridge pred; verify the stored R2.
        X, Y = data.lmsys_layer(layer)
        X, Y = X.astype(np.float64), Y.astype(np.float64)
        mask = np.ones(n_ctx, dtype=bool)
        mask[test_idx] = False
        Xtr, Ytr = X[mask], Y[mask]
        Xte, Yte = X[test_idx], Y[test_idx]
        Ytr_c = Ytr - Ytr.mean(0)
        _u, _s, vh = torch.linalg.svd(
            torch.as_tensor(Ytr_c, dtype=torch.float64), full_matrices=False
        )
        vh_np = vh.numpy()
        pred = PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)
        sel_all = np.concatenate([worst_sel, best_sel])
        sel_ranks = ranks[sel_all]
        dirs = vh_np[sel_ranks].T  # (H, n_sel)
        from issue779_identity_baseline import _per_direction_r2

        r2_re = _per_direction_r2(Yte, pred, dirs)
        max_r2_dev = float(np.nanmax(np.abs(r2_re - r2[sel_all])))
        assert max_r2_dev < 1e-6, f"{t}: recomputed per-direction R2 deviates {max_r2_dev}"
        logger.info("[d4] %s: recomputed R2 matches stored (max dev %.2e)", t, max_r2_dev)

        # Corpus mean profiles at this layer (centered) for top-loading contexts.
        _Xc, Vc = data.corpus_layer(t, layer)
        Vc_c = Vc.astype(np.float64) - Vc.astype(np.float64).mean(0)
        rollouts = {} if smoke else _rollouts_text(t)
        rb_at_layer = {t2: data.rb(t2)[layer] for t2 in TRAITS}

        entries = []
        for j, gi in enumerate(sel_all):
            rank = int(ranks[gi])
            d_vec = vh_np[rank]
            toks = _top_tokens(W_U, norm_w, tok, d_vec)
            cos_rb = {
                t2: float(np.dot(d_vec, rb) / (np.linalg.norm(d_vec) * np.linalg.norm(rb) + 1e-30))
                for t2, rb in rb_at_layer.items()
            }
            proj = Vc_c @ d_vec
            top3 = np.argsort(-np.abs(proj))[:3]
            ctxs = []
            for ci in top3.tolist():
                rec = rollouts.get(str(ci), {})
                resp = (rec.get("responses") or [""])[0]
                ctxs.append(
                    {
                        "ctx_idx": int(ci),
                        "persona_idx": rec.get("persona_idx"),
                        "question": (rec.get("question") or "")[:200],
                        "response_snippet": resp[:240],
                        "projection": float(proj[ci]),
                    }
                )
            entry = {
                "rank": rank,
                "group": "worst" if j < len(worst_sel) else "best",
                "heldout_r2": float(r2[gi]),
                "variance_share": float(share[gi]),
                "tokens": toks,
                "cos_r_b": cos_rb,
                "top_contexts": ctxs,
            }
            entries.append(entry)
            pos_s = " ".join(repr(x["token"]) for x in toks["pos"][:6])
            neg_s = " ".join(repr(x["token"]) for x in toks["neg"][:6])
            md_lines.append(
                f"| {t} | {rank} ({entry['group']}) | {entry['heldout_r2']:.3f} | "
                f"{entry['variance_share']:.4f} | {pos_s} | {neg_s} | "
                f"{cos_rb[t]:+.3f} |"
            )
        results["directions"][t] = {
            "layer": layer,
            "n_worst": n_worst,
            "n_best": n_best,
            "r2_recompute_max_dev": max_r2_dev,
            "entries": entries,
        }
        results["md_table"] = "\n".join(md_lines)
        results["metadata"] = C.reproducibility_metadata(
            {"script": "issue779_batch2 d4", "smoke": smoke}
        )
        C.write_json_atomic(out_path, results)
        _t(f"d4 trait {t}", tt)
    _t("d4 total", t0)


# ── d5: final-token map (scalar carry-forward read) ────────────────────────────


def _within_r_proj(
    pred: np.ndarray, target: np.ndarray, mat: dict, mode: str, *, n_boot: int
) -> dict:
    """Within-condition r of (pred, target) for projection-valued targets."""
    cx, cy = S1._group_by_condition(pred, target, mat["cond"], mat["mode"], mode)
    cx2, cy2 = [], []
    for xi, yi in zip(cx, cy, strict=True):
        fin = np.isfinite(xi) & np.isfinite(yi)
        if fin.sum() >= 3:
            cx2.append(xi[fin])
            cy2.append(yi[fin])
    return M.bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=0, min_y_std=1e-9)


def run_d5(data: Data, *, smoke: bool = False) -> None:
    out_path = OUT_DIR / ("batch2_logitlens_smoke.json" if smoke else "batch2_logitlens.json")
    assert out_path.exists(), "run d4 first (d5 appends final_token_map)"
    results = _load_json(out_path)
    t0 = time.time()
    n_boot = 50 if smoke else N_BOOT
    ftm: dict = {
        "note": (
            "scalar map c_last -> final-answer-token projection (r2_last) vs "
            "c_last -> response-mean projection (oracle); ridge LOCO within mode "
            "(ridge_predict_loco_raw); baseline = fit-free pv_raw = <c_last, r_B>. "
            "within-condition r uses min_y_std=1e-9 (projection target)."
        ),
        "traits": {},
    }
    for t in TRAITS:
        ftm["traits"][t] = {}
        for mode in MODES:
            layer = MODE_LAYER[mode][t]
            mat = data.eval_mat(t, layer)
            sel_mode = np.array([m == mode for m in mat["mode"]])
            node: dict = {"layer": layer}
            for tgt_name in ("r2_last", "oracle"):
                tgt = np.asarray(mat[tgt_name], dtype=np.float64)
                sel = sel_mode & np.isfinite(tgt)
                X = np.asarray(mat["c_last"], dtype=np.float64)[sel]
                y = tgt[sel]
                if len(y) < 5 or float(np.std(y)) < 1e-12:
                    node[tgt_name] = {"skipped": True, "n": len(y)}
                    continue
                pred_full = np.full(len(tgt), np.nan)
                pred_full[sel] = ridge_predict_loco_raw(X, y[:, None])[:, 0]
                pv = np.asarray(mat["pv_raw"], dtype=np.float64)
                node[tgt_name] = {
                    "n": len(y),
                    "loco_within_r": _within_r_proj(pred_full, tgt, mat, mode, n_boot=n_boot),
                    "loco_overall_r": M.overall_pearson(pred_full[sel], y),
                    "pvraw_within_r": _within_r_proj(pv, tgt, mat, mode, n_boot=n_boot),
                    "pvraw_overall_r": M.overall_pearson(pv[sel], y),
                }
                logger.info(
                    "[d5] %s %s %s: LOCO within=%.3f overall=%.3f | pv_raw within=%.3f "
                    "overall=%.3f (n=%d)",
                    t,
                    mode,
                    tgt_name,
                    node[tgt_name]["loco_within_r"]["point"],
                    node[tgt_name]["loco_overall_r"],
                    node[tgt_name]["pvraw_within_r"]["point"],
                    node[tgt_name]["pvraw_overall_r"],
                    len(y),
                )
            ftm["traits"][t][mode] = node
        results["final_token_map"] = ftm
        results["metadata_d5"] = C.reproducibility_metadata(
            {"script": "issue779_batch2 d5", "smoke": smoke}
        )
        C.write_json_atomic(out_path, results)
    _t("d5 total", t0)


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 batch 2 analyses.")
    parser.add_argument("--deliverable", required=True, choices=["d1", "d2", "d3", "d4", "d5"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = Data()
    t0 = time.time()
    {"d1": run_d1, "d2": run_d2, "d3": run_d3, "d4": run_d4, "d5": run_d5}[args.deliverable](
        data, smoke=args.smoke
    )
    _t(f"deliverable {args.deliverable} wall", t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
