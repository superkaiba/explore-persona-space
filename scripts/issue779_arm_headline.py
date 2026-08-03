#!/usr/bin/env python3
"""Issue #779 free-analysis: 3-arm headline + target construction + noise ceiling + grouping.

Four independently-runnable sections (``--only 1|2|3|4``), all 0-GPU, vectorized
CPU, checkpointed into ONE JSON (atomic rewrite per (trait, mode) / per section):

1. **3-arm headline** — fit the behavior-agnostic map ``h: c_last -> v(x)`` and the
   direct predictor ``g: c_last -> judge score`` (both GCV ridge) on three training
   arms — A: 5000 LMSYS contexts (single-rollout v(x) targets, LMSYS judge labels);
   B: 2400 trait-corpus contexts (10-rollout-mean v(x) targets, per-context mean
   judge labels); C: the natural A+B concat AND a 1:1 variant (A subsampled to
   2400) — then evaluate all fits on the FIXED Persona-Vectors eval rig
   (``build_eval_matrix``) at the FROZEN step0 read-out layers: within-condition
   Pearson r (bootstrap CI) for {h_dot, h_cos, g} per arm alongside the fit-free
   pv_raw + oracle references, plus 5-fold held-out profile-reconstruction R2 per
   arm (test-fold-mean convention, percontext_recon protocol).

2. **Target construction (1-vs-10 rollouts)** — on Arm B at full N=2400: h fit on
   K=5 single-random-rollout target draws vs the 10-rollout-mean target; within-
   condition r + held-out recon R2 deltas (mean +- sd over K).

3. **Per-direction noise ceiling** — from the corpus v_x (10 rollouts x 2400
   contexts) at the SYSTEM frozen layer, one-way random-effects variance
   decomposition along (a) top-200 PCA directions of the per-context mean
   profiles, (b) r_B, (c) 50 random directions; explainable-variance ceilings for
   a single-rollout target AND the 10-mean target, overlaid (rank-matched) with
   h's per-direction held-out R2 from identity_baseline.json -> three-way split
   {linearly captured / above-linear-but-below-ceiling / information-absent}.

4. **Grouped/averaged context vectors** — persona-level grouping on the corpus
   (60 groups of 40 questions): LOGO ridge refit of h at group level, readout
   <h(c_group), r_B> vs group mean judge score; group-size sweep {1,2,5,10,20,40}
   (K=5 draws); per-context LOGO-by-persona baseline; rig condition-level
   group-averaged pv_raw read (13 groups of 20 — too small to fit, read-only).

Reuses (never reimplements): ``issue779_stage1`` load_eval_cells / _load_rb /
build_eval_matrix / method_metrics / _group_by_condition;
``issue779_percontext_recon`` _cv_folds / _pooled_r2 / _ridge_fit_predict_fast
(equivalence gate); ``fit_h`` dot_readout / cosine_readout / ridge_fit_predict
(gate reference); ``metrics`` bootstrap CIs. The only new numerics is
``GramRidge`` — the SAME torch-eigh Gram/dual GCV ridge as
``_ridge_fit_predict_fast`` with the eigendecomposition factored out so ONE
factorization serves many targets (the #823 share-the-factorization lesson);
gated for equivalence against BOTH the fast twin and the canonical
``fit_h.ridge_fit_predict`` before any use.

MEMORY: corpus blobs (11.5 GB each) are ``torch.load(mmap=True)`` and sliced ONE
layer at a time; the full multi-layer tensor is never materialized (<15 GB RSS).
Fail loud; NaN labels are DROPPED with counts reported, never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_common as C  # noqa: E402
import issue779_percontext_recon as R  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_arm_headline")

# Frozen read-out layers per (trait, mode) — step0 best_by_mode; do NOT re-select.
FROZEN_LAYERS: dict[str, dict[str, int]] = {
    "evil": {"system": 14, "many_shot": 26},
    "sycophancy": {"system": 26, "many_shot": 26},
    "hallucination": {"system": 17, "many_shot": 27},
}
MODES = ("system", "many_shot")
ARM_NAMES = ("A_lmsys", "B_trait", "C_mix", "C_1to1")
ARM_LABELS = {
    "A_lmsys": "LMSYS (generic)",
    "B_trait": "trait corpus",
    "C_mix": "natural mix",
    "C_1to1": "1:1 mix",
}

CORPUS_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus")
LMSYS_LABELS_PATH = PROJECT_ROOT / "data" / "issue_779" / "lmsys_g_labels" / "lmsys_g_labels.json"
COLLECT_DIR = (
    PROJECT_ROOT / "data" / "issue779_hfstage" / "issue779_monitoring" / "analysis_tensors"
)
IDENTITY_BASELINE_PATH = PROJECT_ROOT / "eval_results" / "issue_779" / "identity_baseline.json"


# ── shared-factorization GCV ridge ────────────────────────────────────────────


class GramRidge:
    """Torch-eigh Gram/dual GCV ridge with the factorization computed ONCE.

    Math is IDENTICAL to ``issue779_percontext_recon._ridge_fit_predict_fast``
    (standardize-X on train stats / center-Y / GCV lambda over logspace(-2,4,13)
    / dual solve via eigh of the train Gram / un-center) — verified by the
    equivalence gate in ``main`` against both that twin and the canonical
    ``fit_h.ridge_fit_predict``. Factoring the eigendecomposition out of the
    per-target call lets one factorization serve many targets Y (draw sweeps,
    h+g on the same rows) — the #823 share-the-factorization lesson.
    """

    def __init__(self, X_train: np.ndarray, lambdas: np.ndarray | None = None) -> None:
        if lambdas is None:
            lambdas = np.logspace(-2, 4, 13)
        self.lambdas = lambdas
        Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
        self.ntr = int(Xtr.shape[0])
        self.xmu = Xtr.mean(0)
        self.xsd = Xtr.std(0) + 1e-9
        self.Xtr_n = (Xtr - self.xmu) / self.xsd
        G = self.Xtr_n @ self.Xtr_n.T
        w, V = torch.linalg.eigh(G)
        self.w = torch.clamp(w, min=0.0)
        self.V = V
        self.last_lambda: float | None = None

    def predict(self, Y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        """Fit on (X_train, Y_train) at the GCV-selected lambda, predict X_eval."""
        Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64)
        squeeze = Ytr.ndim == 1
        if squeeze:
            Ytr = Ytr[:, None]
        assert Ytr.shape[0] == self.ntr, (Ytr.shape, self.ntr)
        Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64)
        ymu = Ytr.mean(0)
        Ytr_c = Ytr - ymu
        VtY = self.V.T @ Ytr_c
        sqVtY = (VtY**2).sum(1)
        tot = float((Ytr_c**2).sum())
        best_lam, best_gcv = float(self.lambdas[0]), float("inf")
        for lam in self.lambdas:
            filt = self.w / (self.w + lam)
            rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
            dof = float(filt.sum())
            denom = (self.ntr - dof) ** 2
            gcv = rss / denom if denom > 1e-12 else float("inf")
            if gcv < best_gcv:
                best_gcv, best_lam = gcv, float(lam)
        self.last_lambda = best_lam
        Xev_n = (Xev - self.xmu) / self.xsd
        KevV = (Xev_n @ self.Xtr_n.T) @ self.V
        filt = 1.0 / (self.w + best_lam)
        pred = ((KevV * filt) @ VtY + ymu).numpy()
        return pred[:, 0] if squeeze else pred


def equivalence_gate(bundle: dict, seed: int) -> dict:
    """Assert GramRidge == _ridge_fit_predict_fast == fit_h.ridge_fit_predict.

    Run ONCE on a 500-train/100-eval slice of the LMSYS bundle at layer 14
    (the percontext_recon gate recipe + tolerances). Fail loud on divergence.
    """
    rng = np.random.default_rng(seed)
    sub = rng.choice(bundle["cx_last"].shape[0], size=600, replace=False)
    li = bundle["layers"].index(14)
    X = bundle["cx_last"][:, li, :].to(torch.float64).numpy()[sub]
    Y = bundle["v_x"][:, li, :].to(torch.float64).numpy()[sub]
    pred_canon = F.ridge_fit_predict(X[:500], Y[:500], X[500:])
    pred_fast = R._ridge_fit_predict_fast(X[:500], Y[:500], X[500:])
    pred_gram = GramRidge(X[:500]).predict(Y[:500], X[500:])
    scale = float(np.max(np.abs(pred_canon))) + 1e-12
    rel_fast = float(np.max(np.abs(pred_gram - pred_fast))) / scale
    rel_canon = float(np.max(np.abs(pred_gram - pred_canon))) / scale
    r2_canon = R._pooled_r2(pred_canon, Y[500:])
    r2_gram = R._pooled_r2(pred_gram, Y[500:])
    r2_diff = abs(r2_canon - r2_gram)
    assert rel_fast < 1e-9 and rel_canon < 1e-6 and r2_diff < 1e-6, (
        f"GramRidge equivalence gate FAILED: rel-vs-fast {rel_fast:.2e}, "
        f"rel-vs-canonical {rel_canon:.2e}, R2 diff {r2_diff:.2e}"
    )
    logger.info(
        "GramRidge equivalence gate PASS (rel-vs-fast %.2e, rel-vs-canonical %.2e, R2 diff %.2e)",
        rel_fast,
        rel_canon,
        r2_diff,
    )
    return {"rel_vs_fast": rel_fast, "rel_vs_canonical": rel_canon, "r2_diff": r2_diff}


# ── data loading ──────────────────────────────────────────────────────────────


def load_lmsys_bundle() -> dict:
    """Load the pass-B LMSYS train bundle (mmap when possible)."""
    path = COLLECT_DIR / "pass_b" / "train_context_vectors.pt"
    try:
        bundle = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    except RuntimeError as e:  # legacy (non-zipfile) serialization cannot mmap
        logger.warning("mmap load of %s failed (%s); falling back to full load", path, e)
        bundle = torch.load(path, weights_only=False, map_location="cpu")
    assert bundle["cx_last"].shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
    assert bundle["v_x"].shape == bundle["cx_last"].shape
    return bundle


def load_corpus(trait: str) -> dict:
    """mmap-load a trait corpus blob + verify the index semantics by shape.

    Verifies ``vx_index`` enumerates (context, rollout) row-major so
    ``v_x.reshape(n_ctx, n_rollouts, H)`` is the correct per-context grouping,
    and that ``persona_idx``/``question_idx`` are the row-major (persona,
    question) flattening.
    """
    blob = torch.load(
        CORPUS_DIR / f"{trait}_corpus.pt", mmap=True, weights_only=False, map_location="cpu"
    )
    n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
    n_ctx = n_p * n_q
    hidden = C.EXPECTED_HIDDEN
    assert blob["cx_last"].shape == (n_ctx, C.EXPECTED_LAYERS, hidden), blob["cx_last"].shape
    assert blob["v_x"].shape == (n_ctx * n_r, C.EXPECTED_LAYERS, hidden), blob["v_x"].shape
    expected_vi = [(c, r) for c in range(n_ctx) for r in range(n_r)]
    assert [tuple(t) for t in blob["vx_index"]] == expected_vi, "vx_index is not row-major"
    assert list(blob["persona_idx"]) == [c // n_q for c in range(n_ctx)]
    assert list(blob["question_idx"]) == [c % n_q for c in range(n_ctx)]
    return blob


def corpus_scores(trait: str) -> np.ndarray:
    """Per-(context, rollout) judge scores as (n_ctx, n_rollouts), NaN = dropped."""
    with open(CORPUS_DIR / f"{trait}_judge_scores.json") as f:
        raw = json.load(f)["scores"]
    n_ctx = len(raw)
    n_r = len(next(iter(raw.values())))
    out = np.full((n_ctx, n_r), np.nan)
    for ctx_s, rolls in raw.items():
        for ri_s, v in rolls.items():
            if v is not None:
                out[int(ctx_s), int(ri_s)] = float(v)
    return out


def lmsys_labels(trait: str, n_expected: int) -> np.ndarray:
    """Per-context LMSYS judge label (NaN = judge-dropped), index-aligned to pass B."""
    with open(LMSYS_LABELS_PATH) as f:
        d = json.load(f)["labels_per_trait"][trait]
    labels = np.array([np.nan if v is None else float(v) for v in d["labels"]])
    assert labels.shape == (n_expected,), (labels.shape, n_expected)
    return labels


def _label_diag(y: np.ndarray) -> dict:
    """n / n_dropped / mean / std of a label vector with NaN = dropped."""
    fin = np.isfinite(y)
    return {
        "n_total": len(y),
        "n_dropped": int((~fin).sum()),
        "label_mean": float(np.mean(y[fin])) if fin.any() else float("nan"),
        "label_std": float(np.std(y[fin])) if fin.any() else float("nan"),
    }


# ── shared eval-rig + metric helpers ──────────────────────────────────────────


def _mode_metrics(x: np.ndarray, mat: dict, n_boot: int, seed: int) -> dict:
    """method_metrics for one monitor; keep both modes + overall."""
    return S1.method_metrics(x, mat, n_boot=n_boot, seed=seed)


def _delta_vs(
    x_a: np.ndarray,
    x_b: np.ndarray,
    mat: dict,
    mode: str,
    n_boot: int,
    seed: int,
    replicates_out: list | None = None,
) -> dict:
    cx_a, cy = S1._group_by_condition(x_a, mat["y"], mat["cond"], mat["mode"], mode)
    cx_b, _ = S1._group_by_condition(x_b, mat["y"], mat["cond"], mat["mode"], mode)
    if not cx_a or not cx_b:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return M.bootstrap_delta_ci(
        cx_a, cx_b, cy, n_boot=n_boot, seed=seed, replicates_out=replicates_out
    )


def heldout_recon_multi(
    X: np.ndarray, targets: dict[str, np.ndarray], n_folds: int, seed: int
) -> dict[str, dict]:
    """5-fold held-out pooled R2 per target, ONE Gram factorization per fold.

    percontext_recon protocol: SS_tot on the TEST fold's own mean
    (``_pooled_r2``), mean +- sd across folds.
    """
    n = len(X)
    folds = R._cv_folds(n, n_folds, seed)
    r2s: dict[str, list[float]] = {k: [] for k in targets}
    for test_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        gr = GramRidge(X[mask])
        for k, Y in targets.items():
            r2s[k].append(R._pooled_r2(gr.predict(Y[mask], X[test_idx]), Y[test_idx]))
    return {
        k: {
            "r2_mean": float(np.mean(v)),
            "r2_sd": float(np.std(v)),
            "folds": [float(x) for x in v],
            "n": int(n),
        }
        for k, v in r2s.items()
    }


def _pearson_boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    """Pearson r + percentile bootstrap CI over paired resamples."""
    fin = np.isfinite(x) & np.isfinite(y)
    x, y = x[fin], y[fin]
    point = M.overall_pearson(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        r = M.overall_pearson(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    lo = float(np.quantile(boots, 0.025)) if boots else float("nan")
    hi = float(np.quantile(boots, 0.975)) if boots else float("nan")
    return {"point": point, "lo": lo, "hi": hi, "n": len(x)}


# ── context (lazy data cache) ─────────────────────────────────────────────────


class Ctx:
    """Lazy per-process cache of bundles, corpora, rig cells, and fit cells."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._bundle: dict | None = None
        self._corpus: dict[str, dict] = {}
        self._scores: dict[str, np.ndarray] = {}
        self._cells: dict[str, list] = {}
        self._rb: dict[str, np.ndarray] = {}
        self._mats: dict[tuple[str, int], dict] = {}
        self._fit_cells: dict[tuple[str, int], dict] = {}

    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            logger.info("Loading LMSYS pass-B bundle")
            self._bundle = load_lmsys_bundle()
        return self._bundle

    def corpus(self, trait: str) -> dict:
        if trait not in self._corpus:
            logger.info("[%s] mmap-loading behavior corpus", trait)
            self._corpus[trait] = load_corpus(trait)
        return self._corpus[trait]

    def scores(self, trait: str) -> np.ndarray:
        if trait not in self._scores:
            self._scores[trait] = corpus_scores(trait)
        return self._scores[trait]

    def rb(self, trait: str) -> np.ndarray:
        if trait not in self._rb:
            self._rb[trait] = S1._load_rb(
                COLLECT_DIR / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN
            )
        return self._rb[trait]

    def mat(self, trait: str, li: int) -> dict:
        key = (trait, li)
        if key not in self._mats:
            if trait not in self._cells:
                self._cells[trait] = S1.load_eval_cells(COLLECT_DIR / "pass_a", trait)
            self._mats[key] = S1.build_eval_matrix(self._cells[trait], li, self.rb(trait))
        return self._mats[key]

    def corpus_layer(self, trait: str, li: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Xb (n_ctx,H), Vb (n_ctx,n_r,H), Yb_mean (n_ctx,H)) at one layer, fp32."""
        blob = self.corpus(trait)
        col = blob["layers"].index(li)
        n_ctx = blob["n_personas"] * blob["n_questions"]
        xb = blob["cx_last"][:, col, :].to(torch.float32).numpy()
        vb = (
            blob["v_x"][:, col, :]
            .to(torch.float32)
            .numpy()
            .reshape(n_ctx, blob["n_rollouts"], C.EXPECTED_HIDDEN)
        )
        return xb, vb, vb.mean(axis=1)

    def lmsys_layer(self, li: int) -> tuple[np.ndarray, np.ndarray]:
        """(Xa, Ya) at one layer, fp32 numpy."""
        col = self.bundle["layers"].index(li)
        xa = self.bundle["cx_last"][:, col, :].to(torch.float32).numpy()
        ya = self.bundle["v_x"][:, col, :].to(torch.float32).numpy()
        return xa, ya


# ── Section 1: 3-arm headline ─────────────────────────────────────────────────


def fit_cell(ctx: Ctx, trait: str, li: int) -> dict:
    """All arm fits + monitors + recon for one (trait, layer). Cached in-process."""
    key = (trait, li)
    if key in ctx._fit_cells:
        return ctx._fit_cells[key]
    args = ctx.args
    mat = ctx.mat(trait, li)
    Xev = mat["c_last"]
    rb_l = ctx.rb(trait)[li]

    Xa, Ya = ctx.lmsys_layer(li)
    Xb, _vb, Yb = ctx.corpus_layer(trait, li)
    ga = lmsys_labels(trait, Xa.shape[0])
    with np.errstate(invalid="ignore"):
        gb = np.nanmean(ctx.scores(trait), axis=1)  # per-context mean over VALID rollouts

    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(Xa.shape[0], size=Xb.shape[0], replace=False))

    arms: dict[str, dict] = {
        "A_lmsys": {"Xh": Xa, "Yh": Ya, "g_X": Xa, "g_y": ga},
        "B_trait": {"Xh": Xb, "Yh": Yb, "g_X": Xb, "g_y": gb},
        "C_mix": {
            "Xh": np.concatenate([Xa, Xb]),
            "Yh": np.concatenate([Ya, Yb]),
            "g_X": np.concatenate([Xa, Xb]),
            "g_y": np.concatenate([ga, gb]),
        },
        "C_1to1": {
            "Xh": np.concatenate([Xa[sub], Xb]),
            "Yh": np.concatenate([Ya[sub], Yb]),
            "g_X": np.concatenate([Xa[sub], Xb]),
            "g_y": np.concatenate([ga[sub], gb]),
        },
    }

    monitors: dict[str, np.ndarray] = {"pv_raw": mat["pv_raw"], "oracle": mat["oracle"]}
    recon: dict[str, dict] = {}
    g_diag: dict[str, dict] = {}
    n_train: dict[str, dict] = {}
    for arm, d in arms.items():
        logger.info("[%s L%d] arm %s: h fit (n=%d)", trait, li, arm, len(d["Xh"]))
        gr_h = GramRidge(d["Xh"])
        pred = gr_h.predict(d["Yh"], Xev)
        monitors[f"h_{arm}_dot"] = F.dot_readout(pred, rb_l)
        monitors[f"h_{arm}_cos"] = F.cosine_readout(pred, rb_l)
        h_lambda = gr_h.last_lambda
        # g: ridge on the VALID-labeled rows only (drop-never-coerce).
        fin = np.isfinite(d["g_y"])
        g_diag[arm] = _label_diag(d["g_y"])
        logger.info("[%s L%d] arm %s: g fit (n=%d valid)", trait, li, arm, int(fin.sum()))
        gr_g = GramRidge(d["g_X"][fin])
        monitors[f"g_{arm}"] = gr_g.predict(d["g_y"][fin], Xev)
        n_train[arm] = {"h": len(d["Xh"]), "g": int(fin.sum())}
        # Held-out profile reconstruction (5-fold, arm's own h training set).
        logger.info("[%s L%d] arm %s: %d-fold held-out recon", trait, li, arm, args.n_folds)
        recon[arm] = heldout_recon_multi(
            d["Xh"], {"h": d["Yh"]}, n_folds=args.n_folds, seed=args.seed
        )["h"]
        recon[arm]["gcv_lambda_full_fit"] = h_lambda

    logger.info("[%s L%d] within-condition metrics (%d monitors)", trait, li, len(monitors))
    method_res = {
        name: _mode_metrics(x, mat, n_boot=args.n_boot, seed=args.seed)
        for name, x in monitors.items()
    }
    deltas: dict[str, dict] = {}
    for name in monitors:
        if name in ("pv_raw", "oracle"):
            continue
        deltas[name] = {
            mode: _delta_vs(
                monitors[name], monitors["pv_raw"], mat, mode, n_boot=args.n_boot, seed=args.seed
            )
            for mode in MODES
        }

    cell = {
        "mat": mat,
        "monitors": monitors,
        "method_res": method_res,
        "deltas_vs_pv_raw": deltas,
        "recon": recon,
        "g_label_diag": g_diag,
        "n_train": n_train,
    }
    ctx._fit_cells[key] = cell
    return cell


def run_section1(res: dict, ctx: Ctx) -> None:
    args = ctx.args
    sec = res.setdefault("arm_headline", {})
    for trait in C.TRAITS:
        tr = sec.setdefault(trait, {})
        for mode in MODES:
            if mode in tr:
                logger.info("[sec1 %s %s] already checkpointed; skipping", trait, mode)
                continue
            li = FROZEN_LAYERS[trait][mode]
            cell = fit_cell(ctx, trait, li)
            entry: dict = {
                "layer": li,
                "n_eval_rows": len(cell["mat"]["y"]),
                "monitors": {
                    name: {**mm[mode], "overall_r_both_modes": mm["overall_r"]}
                    for name, mm in cell["method_res"].items()
                },
                "deltas_vs_pv_raw": {name: d[mode] for name, d in cell["deltas_vs_pv_raw"].items()},
                "recon_heldout": cell["recon"],
                "g_label_diag": cell["g_label_diag"],
                "n_train": cell["n_train"],
            }
            tr[mode] = entry
            C.write_json_atomic(args.out_json, res)
            logger.info(
                "[sec1 %s %s L%d] pv_raw=%.3f oracle=%.3f | h_cos A/B/Cmix/C11 = "
                "%.3f/%.3f/%.3f/%.3f | g A/B/Cmix/C11 = %.3f/%.3f/%.3f/%.3f",
                trait,
                mode,
                li,
                entry["monitors"]["pv_raw"]["point"],
                entry["monitors"]["oracle"]["point"],
                *[entry["monitors"][f"h_{a}_cos"]["point"] for a in ARM_NAMES],
                *[entry["monitors"][f"g_{a}"]["point"] for a in ARM_NAMES],
            )
    make_arm_headline_figure(res, ctx)
    C.write_json_atomic(args.out_json, res)


def make_arm_headline_figure(res: dict, ctx: Ctx) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    bars = (
        [("pv_raw", "PV raw projection"), ("oracle", "oracle (true answer proj.)")]
        + [(f"h_{a}_cos", f"h — {ARM_LABELS[a]}") for a in ARM_NAMES]
        + [(f"g_{a}", f"g — {ARM_LABELS[a]}") for a in ARM_NAMES]
    )
    colors = paper_palette(3)
    bar_colors = [colors[0]] * 2 + [colors[1]] * 4 + [colors[2]] * 4
    # Tolerate a PARTIAL trait set: this VM may compute a subset of the traits
    # (the rest land in a sibling JSON from the CPU pod) — plot what is present.
    present = [t for t in C.TRAITS if t in res.get("arm_headline", {})]
    fig, axes = plt.subplots(
        2, len(present), figsize=(5.4 * len(present) + 0.6, 9), squeeze=False, layout="tight"
    )
    for col, trait in enumerate(present):
        for row, mode in enumerate(MODES):
            ax = axes[row][col]
            if mode not in res["arm_headline"][trait]:
                ax.set_axis_off()
                continue
            entry = res["arm_headline"][trait][mode]
            heights, errs, labels = [], [], []
            for name, label in bars:
                mm = entry["monitors"][name]
                pt, lo, hi = mm["point"], mm["lo"], mm["hi"]
                if not np.isfinite(pt):
                    continue
                heights.append(pt)
                errs.append(
                    [
                        max(0.0, pt - lo) if np.isfinite(lo) else 0.0,
                        max(0.0, hi - pt) if np.isfinite(hi) else 0.0,
                    ]
                )
                labels.append(label)
            ax.bar(
                range(len(heights)),
                heights,
                yerr=np.array(errs).T if errs else None,
                capsize=2,
                color=bar_colors[: len(heights)],
            )
            ax.axhline(0.0, color="gray", lw=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            mode_lbl = "system prompting" if mode == "system" else "many-shot"
            ax.set_title(f"{trait} — {mode_lbl} (layer {entry['layer']})")
            if col == 0:
                ax.set_ylabel("within-condition Pearson r")
    figs = savefig_paper(fig, "arm_headline", dir=ctx.args.fig_dir)
    plt.close(fig)
    res.setdefault("figures", {})["arm_headline"] = str(figs.get("png", ""))
    logger.info("Wrote %s", figs.get("png"))


# ── Section 2: target construction (1-vs-10 rollout targets) ─────────────────


def run_section2(res: dict, ctx: Ctx) -> None:
    args = ctx.args
    sec = res.setdefault("target_construction", {})
    for trait in C.TRAITS:
        tr = sec.setdefault(trait, {})
        layers = sorted(set(FROZEN_LAYERS[trait].values()))
        for li in layers:
            lkey = f"L{li}"
            if lkey in tr:
                logger.info("[sec2 %s %s] already checkpointed; skipping", trait, lkey)
                continue
            modes_here = [m for m in MODES if FROZEN_LAYERS[trait][m] == li]
            mat = ctx.mat(trait, li)
            Xev = mat["c_last"]
            rb_l = ctx.rb(trait)[li]
            Xb, Vb, Yb = ctx.corpus_layer(trait, li)
            n_ctx, n_r, _h = Vb.shape

            # ONE factorization for the full-set fit, reused across all targets.
            gr = GramRidge(Xb)
            draw_targets: dict[str, np.ndarray] = {"mean10": Yb}
            rng = np.random.default_rng(args.seed)
            for k in range(args.k_draws):
                draw = rng.integers(0, n_r, size=n_ctx)
                draw_targets[f"draw{k}"] = Vb[np.arange(n_ctx), draw]

            readout: dict[str, dict] = {}
            for name, Y in draw_targets.items():
                pred = gr.predict(Y, Xev)
                dot = F.dot_readout(pred, rb_l)
                cos = F.cosine_readout(pred, rb_l)
                mm_dot = _mode_metrics(dot, mat, n_boot=args.n_boot, seed=args.seed)
                mm_cos = _mode_metrics(cos, mat, n_boot=args.n_boot, seed=args.seed)
                readout[name] = {
                    "dot": {m: mm_dot[m] for m in modes_here},
                    "cos": {m: mm_cos[m] for m in modes_here},
                }
            # Held-out recon: fold Grams shared across ALL targets.
            recon = heldout_recon_multi(Xb, draw_targets, n_folds=args.n_folds, seed=args.seed)

            draws = [f"draw{k}" for k in range(args.k_draws)]
            deltas: dict[str, dict] = {}
            for m in modes_here:
                for kind in ("dot", "cos"):
                    dv = [
                        readout[d][kind][m]["point"] - readout["mean10"][kind][m]["point"]
                        for d in draws
                    ]
                    deltas[f"{kind}_{m}"] = {"mean": float(np.mean(dv)), "sd": float(np.std(dv))}
            dr = [recon[d]["r2_mean"] - recon["mean10"]["r2_mean"] for d in draws]
            deltas["recon_r2"] = {"mean": float(np.mean(dr)), "sd": float(np.std(dr))}

            tr[lkey] = {
                "modes": modes_here,
                "k_draws": args.k_draws,
                "readout": readout,
                "recon_heldout": recon,
                "delta_1rollout_minus_10mean": deltas,
            }
            C.write_json_atomic(args.out_json, res)
            logger.info(
                "[sec2 %s L%d] recon mean10=%.3f, 1-rollout delta %.3f+-%.3f | %s",
                trait,
                li,
                recon["mean10"]["r2_mean"],
                deltas["recon_r2"]["mean"],
                deltas["recon_r2"]["sd"],
                {
                    k: f"{v['mean']:+.3f}+-{v['sd']:.3f}"
                    for k, v in deltas.items()
                    if k != "recon_r2"
                },
            )


# ── Section 3: per-direction noise ceiling ────────────────────────────────────


def run_section3(res: dict, ctx: Ctx) -> None:
    args = ctx.args
    sec = res.setdefault("noise_ceiling", {})
    with open(IDENTITY_BASELINE_PATH) as f:
        ib = json.load(f)["per_direction"]
    for trait in C.TRAITS:
        if trait in sec:
            logger.info("[sec3 %s] already checkpointed; skipping", trait)
            continue
        li = FROZEN_LAYERS[trait]["system"]
        _xb, Vb, Yb = ctx.corpus_layer(trait, li)
        n_ctx, n_r, hidden = Vb.shape
        rb_l = ctx.rb(trait)[li]

        # Direction sets: (a) top-200 PCA of per-context MEAN profiles, (b) r_B,
        # (c) 50 random unit directions.
        Mc = torch.as_tensor(Yb, dtype=torch.float64)
        Mc = Mc - Mc.mean(0)
        _u, s, vh = torch.linalg.svd(Mc, full_matrices=False)
        n_pca = min(200, vh.shape[0])
        D_pca = vh[:n_pca].numpy()
        pca_var_share = (s[:n_pca] ** 2 / (s**2).sum()).numpy()
        rb_dir = (rb_l / (np.linalg.norm(rb_l) + 1e-12))[None, :]
        rng = np.random.default_rng(args.seed)
        D_rand = rng.standard_normal((50, hidden))
        D_rand /= np.linalg.norm(D_rand, axis=1, keepdims=True)
        D_all = np.concatenate([D_pca, rb_dir, D_rand]).astype(np.float64)  # (251, H)

        # Vectorized projection of ALL rollouts onto ALL directions.
        proj = Vb.reshape(n_ctx * n_r, hidden).astype(np.float64) @ D_all.T
        proj = proj.reshape(n_ctx, n_r, -1)
        within = proj.var(axis=1, ddof=1).mean(axis=0)  # E[within-context var]
        var_means = proj.mean(axis=1).var(axis=0, ddof=1)  # Var of per-context means
        # One-way random-effects: Var(mean_ctx) = between + within/n_r =>
        # unbiased between = var_means - within/n_r (clipped at 0).
        between = np.clip(var_means - within / n_r, 0.0, None)
        ceil_single = between / (between + within + 1e-30)
        ceil_10 = between / (between + within / n_r + 1e-30)

        k = n_pca
        pca_cs, pca_c10 = ceil_single[:k], ceil_10[:k]
        rb_cs, rb_c10 = float(ceil_single[k]), float(ceil_10[k])
        rnd_cs, rnd_c10 = ceil_single[k + 1 :], ceil_10[k + 1 :]

        # h's held-out per-direction R2 (identity_baseline; LMSYS PCA basis) —
        # rank-matched overlay ONLY: the bases differ (LMSYS train-fold target
        # PCA there vs behavior-corpus mean-profile PCA here).
        ibt = ib[trait]
        assert ibt["read_out_layer"] == li, (trait, ibt["read_out_layer"], li)
        ranks = [int(r) for r in ibt["ranks_evaluated"]]
        h_r2 = [float(v) for v in ibt["r2_by_rank"]]
        rank_to_h = {r: v for r, v in zip(ranks, h_r2, strict=True) if r < k}

        thresholds = {"info_absent_ceil10_lt": 0.2, "captured_frac_of_ceiling": 0.8}
        split: dict[str, list[int]] = {
            "linearly_captured": [],
            "above_linear_below_ceiling": [],
            "information_absent": [],
        }
        for r, hv in sorted(rank_to_h.items()):
            c10 = float(pca_c10[r])
            if c10 < thresholds["info_absent_ceil10_lt"]:
                split["information_absent"].append(r)
            elif hv >= thresholds["captured_frac_of_ceiling"] * c10:
                split["linearly_captured"].append(r)
            else:
                split["above_linear_below_ceiling"].append(r)

        sec[trait] = {
            "layer": li,
            "n_contexts": int(n_ctx),
            "n_rollouts": int(n_r),
            "estimator_note": (
                "one-way random-effects decomposition per direction: within = mean "
                "per-context rollout variance (ddof=1); between = Var(per-context "
                "mean) - within/n_rollouts, clipped at 0 (unbiased). ceil_single = "
                "between/(between+within); ceil_10mean = between/(between+within/10)."
            ),
            "overlay_note": (
                "h per-direction held-out R2 from identity_baseline.json per_direction "
                "(fold 0, LMSYS corpus, PCA basis from LMSYS train-fold targets) — "
                "RANK-matched overlay only; the PCA bases differ (behavior-corpus "
                "mean-profile PCA here)."
            ),
            "pca": {
                "ceil_single": [float(v) for v in pca_cs],
                "ceil_10mean": [float(v) for v in pca_c10],
                "var_share": [float(v) for v in pca_var_share],
            },
            "r_b": {
                "ceil_single": rb_cs,
                "ceil_10mean": rb_c10,
                "h_heldout_r2_lmsys": ibt["r_b"]["heldout_r2"],
            },
            "random": {
                "n": len(rnd_cs),
                "ceil_single_mean": float(rnd_cs.mean()),
                "ceil_single_sd": float(rnd_cs.std()),
                "ceil_10mean_mean": float(rnd_c10.mean()),
                "ceil_10mean_sd": float(rnd_c10.std()),
                "h_r2_mean_lmsys": ibt["random_directions"]["r2_mean"],
            },
            "h_overlay": {
                "ranks": sorted(rank_to_h),
                "h_r2": [rank_to_h[r] for r in sorted(rank_to_h)],
            },
            "three_way_split": {
                "thresholds": thresholds,
                "counts": {kk: len(v) for kk, v in split.items()},
                "ranks": split,
            },
        }
        C.write_json_atomic(args.out_json, res)
        logger.info(
            "[sec3 %s L%d] r_B ceil10=%.3f (h R2=%.3f) | split: %s",
            trait,
            li,
            rb_c10,
            ibt["r_b"]["heldout_r2"],
            {kk: len(v) for kk, v in split.items()},
        )
    make_noise_ceiling_figure(res, ctx)
    C.write_json_atomic(args.out_json, res)


def make_noise_ceiling_figure(res: dict, ctx: Ctx) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="tight")
    for ax, trait in zip(axes, C.TRAITS, strict=True):
        d = res["noise_ceiling"][trait]
        ranks_all = np.arange(len(d["pca"]["ceil_10mean"]))
        ax.plot(
            ranks_all,
            d["pca"]["ceil_10mean"],
            color=colors[0],
            lw=1.5,
            label="noise ceiling (10-rollout mean target)",
        )
        ax.plot(
            ranks_all,
            d["pca"]["ceil_single"],
            color=colors[1],
            lw=1.2,
            ls="--",
            label="noise ceiling (single-rollout target)",
        )
        hr = d["h_overlay"]
        ax.plot(
            hr["ranks"],
            np.clip(hr["h_r2"], 0, None),
            color=colors[2],
            lw=1.2,
            label="h held-out R2 (LMSYS PCA rank)",
        )
        ax.axhline(
            d["r_b"]["ceil_10mean"],
            color=colors[3],
            lw=1.0,
            ls=":",
            label=f"trait direction ceiling ({d['r_b']['ceil_10mean']:.2f})",
        )
        ax.scatter(
            [0],
            [d["r_b"]["h_heldout_r2_lmsys"]],
            color=colors[3],
            marker="*",
            s=110,
            zorder=5,
            label=f"h R2 along trait direction ({d['r_b']['h_heldout_r2_lmsys']:.2f})",
        )
        ax.set_xlabel("PCA rank of per-context mean profile")
        ax.set_ylabel("explainable variance / held-out R2")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{trait} (layer {d['layer']})")
        ax.legend(fontsize=7, loc="upper right")
    figs = savefig_paper(fig, "noise_ceiling", dir=ctx.args.fig_dir)
    plt.close(fig)
    res.setdefault("figures", {})["noise_ceiling"] = str(figs.get("png", ""))
    logger.info("Wrote %s", figs.get("png"))


# ── Section 4: grouped/averaged context vectors ───────────────────────────────


def _grouped_vectors(
    Xq: np.ndarray, Yq: np.ndarray, Sq: np.ndarray, q_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group vectors / targets / scores averaging the given per-persona questions.

    ``q_idx`` (n_p, s): per-persona question subset. Group score = pooled mean
    over the subset's VALID rollout scores (drop-never-coerce; NaN when none).
    """
    n_p = Xq.shape[0]
    p_ar = np.arange(n_p)[:, None]
    xg = Xq[p_ar, q_idx].mean(axis=1)
    yg = Yq[p_ar, q_idx].mean(axis=1)
    sg = np.array(
        [
            np.nanmean(Sq[p, q_idx[p]]) if np.isfinite(Sq[p, q_idx[p]]).any() else np.nan
            for p in range(n_p)
        ]
    )
    return xg, yg, sg


def _logo_readout(
    Xg: np.ndarray, Yg: np.ndarray, rb_l: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-group-out ridge h; returns per-group (dot, cos) readouts."""
    n = len(Xg)
    dots = np.zeros(n)
    coss = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        gr = GramRidge(Xg[mask])
        pred = gr.predict(Yg[mask], Xg[i : i + 1])
        dots[i] = float(F.dot_readout(pred, rb_l)[0])
        coss[i] = float(F.cosine_readout(pred, rb_l)[0])
    return dots, coss


def run_section4(res: dict, ctx: Ctx) -> None:
    args = ctx.args
    sec = res.setdefault("grouped_contexts", {})
    for trait in C.TRAITS:
        if trait in sec:
            logger.info("[sec4 %s] already checkpointed; skipping", trait)
            continue
        li = FROZEN_LAYERS[trait]["system"]
        Xb, _vb, Yb = ctx.corpus_layer(trait, li)
        blob = ctx.corpus(trait)
        n_p, n_q, n_r = blob["n_personas"], blob["n_questions"], blob["n_rollouts"]
        rb_l = ctx.rb(trait)[li]
        scores = ctx.scores(trait)  # (n_ctx, n_r) NaN-padded
        n_ctx = n_p * n_q

        Xq = Xb.reshape(n_p, n_q, -1)  # per-persona question blocks
        Yq = Yb.reshape(n_p, n_q, -1)
        Sq = scores.reshape(n_p, n_q, n_r)

        # Full persona-level grouping (all 40 questions).
        all_q = np.tile(np.arange(n_q), (n_p, 1))
        Xg, Yg, Sg = _grouped_vectors(Xq, Yq, Sq, all_q)
        dots, coss = _logo_readout(Xg, Yg, rb_l)
        group_logo = {
            "n_groups": int(n_p),
            "dot": _pearson_boot_ci(dots, Sg, n_boot=args.n_boot, seed=args.seed),
            "cos": _pearson_boot_ci(coss, Sg, n_boot=args.n_boot, seed=args.seed),
            "group_score_diag": _label_diag(Sg),
        }
        logger.info(
            "[sec4 %s L%d] group-level LOGO (n=60): dot r=%.3f cos r=%.3f",
            trait,
            li,
            group_logo["dot"]["point"],
            group_logo["cos"]["point"],
        )

        # Group-size sweep {1,2,5,10,20,40}, K draws each (size 40 = 1 draw).
        sweep: dict[str, dict] = {}
        rng = np.random.default_rng(args.seed)
        for s in (1, 2, 5, 10, 20, 40):
            n_draws = 1 if s == n_q else args.k_draws
            draw_rs: list[float] = []
            draw_rs_cos: list[float] = []
            for _k in range(n_draws):
                q_idx = np.stack([rng.choice(n_q, size=s, replace=False) for _ in range(n_p)])
                xg, yg, sg = _grouped_vectors(Xq, Yq, Sq, q_idx)
                dts, css = _logo_readout(xg, yg, rb_l)
                draw_rs.append(M.overall_pearson(dts[np.isfinite(sg)], sg[np.isfinite(sg)]))
                draw_rs_cos.append(M.overall_pearson(css[np.isfinite(sg)], sg[np.isfinite(sg)]))
            sweep[str(s)] = {
                "dot_r_draws": [float(v) for v in draw_rs],
                "dot_r_mean": float(np.nanmean(draw_rs)),
                "dot_r_sd": float(np.nanstd(draw_rs)),
                "cos_r_mean": float(np.nanmean(draw_rs_cos)),
                "cos_r_sd": float(np.nanstd(draw_rs_cos)),
            }
            logger.info(
                "[sec4 %s] group size %2d: dot r=%.3f+-%.3f",
                trait,
                s,
                sweep[str(s)]["dot_r_mean"],
                sweep[str(s)]["dot_r_sd"],
            )

        # Per-context baseline: LOGO by persona (fit on 59 personas' contexts,
        # predict the held persona's 40 contexts), readout vs per-context score.
        with np.errstate(invalid="ignore"):
            gb = np.nanmean(scores, axis=1)
        dots_pc = np.zeros(n_ctx)
        coss_pc = np.zeros(n_ctx)
        persona_of = np.asarray(blob["persona_idx"])
        for p in range(n_p):
            held = persona_of == p
            gr = GramRidge(Xb[~held])
            pred = gr.predict(Yb[~held], Xb[held])
            dots_pc[held] = F.dot_readout(pred, rb_l)
            coss_pc[held] = F.cosine_readout(pred, rb_l)
        per_context = {
            "n": int(np.isfinite(gb).sum()),
            "dot": _pearson_boot_ci(dots_pc, gb, n_boot=args.n_boot, seed=args.seed),
            "cos": _pearson_boot_ci(coss_pc, gb, n_boot=args.n_boot, seed=args.seed),
            "note": "LOGO-by-persona h readout vs per-context mean judge score (n=2400)",
        }
        logger.info(
            "[sec4 %s] per-context LOGO baseline: dot r=%.3f cos r=%.3f",
            trait,
            per_context["dot"]["point"],
            per_context["cos"]["point"],
        )

        # Rig condition-level read (13 groups of 20 — too small to FIT; read the
        # group-averaged pv_raw projection instead). <mean c_last, r_B> == mean
        # per-context pv_raw within the condition (linearity of the dot product).
        rig: dict[str, dict] = {}
        for mode in MODES:
            li_m = FROZEN_LAYERS[trait][mode]
            mat = ctx.mat(trait, li_m)
            sel = np.array([m == mode for m in mat["mode"]])
            conds = np.unique(mat["cond"][sel])
            gx, gy = [], []
            for cd in conds:
                m = sel & (mat["cond"] == cd)
                gx.append(float(np.mean(mat["pv_raw"][m])))
                gy.append(float(np.mean(mat["y"][m])))
            rig[mode] = {
                "layer": li_m,
                "n_condition_groups": len(conds),
                "group_pv_raw_r": M.overall_pearson(np.array(gx), np.array(gy)),
                "per_context_overall_r": M.overall_pearson(mat["pv_raw"][sel], mat["y"][sel]),
                "note": (
                    "group r over condition means (BETWEEN-condition, tiny n); "
                    "per-context r pools within+between variation in the same mode"
                ),
            }

        sec[trait] = {
            "layer": li,
            "group_level_logo": group_logo,
            "group_size_sweep": sweep,
            "per_context_baseline": per_context,
            "rig_condition_level": rig,
        }
        C.write_json_atomic(args.out_json, res)
    make_grouped_context_figure(res, ctx)
    C.write_json_atomic(args.out_json, res)


def make_grouped_context_figure(res: dict, ctx: Ctx) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="tight")
    for ax, trait in zip(axes, C.TRAITS, strict=True):
        d = res["grouped_contexts"][trait]
        sizes = sorted(int(s) for s in d["group_size_sweep"])
        means = [d["group_size_sweep"][str(s)]["dot_r_mean"] for s in sizes]
        sds = [d["group_size_sweep"][str(s)]["dot_r_sd"] for s in sizes]
        ax.errorbar(
            sizes,
            means,
            yerr=sds,
            marker="o",
            capsize=3,
            color=colors[0],
            label="persona-group h readout (LOGO, n=60 groups)",
        )
        pc = d["per_context_baseline"]["dot"]
        ax.axhline(
            pc["point"],
            color=colors[1],
            ls="--",
            lw=1.2,
            label=f"per-context h readout baseline (r={pc['point']:.2f})",
        )
        ax.fill_between([min(sizes), max(sizes)], pc["lo"], pc["hi"], color=colors[1], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("questions averaged per persona group")
        ax.set_ylabel("Pearson r vs mean judge score")
        ax.set_title(f"{trait} (layer {d['layer']})")
        ax.legend(fontsize=7, loc="lower right")
    figs = savefig_paper(fig, "grouped_context", dir=ctx.args.fig_dir)
    plt.close(fig)
    res.setdefault("figures", {})["grouped_context"] = str(figs.get("png", ""))
    logger.info("Wrote %s", figs.get("png"))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 3-arm headline free-analysis.")
    parser.add_argument("--only", type=str, default="1,2,3,4", help="comma list of sections")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-draws", type=int, default=5)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing output JSON")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    args = parser.parse_args()
    torch.set_num_threads(int(args.n_threads))
    sections = sorted({int(s) for s in args.only.split(",") if s.strip()})
    assert sections and all(s in (1, 2, 3, 4) for s in sections), sections

    res: dict = {}
    params = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "k_draws": args.k_draws,
    }
    if args.out_json.exists() and not args.fresh:
        with open(args.out_json) as f:
            res = json.load(f)
        prior = {k: res.get("metadata", {}).get(k) for k in params}
        if prior != params:
            raise SystemExit(
                f"existing {args.out_json} was produced with params {prior} != {params}; "
                "pass --fresh to overwrite or match the params"
            )
        logger.info("Resuming from existing %s", args.out_json)
    res["metadata"] = C.reproducibility_metadata(
        {"script": "issue779_arm_headline", **params, "frozen_layers": FROZEN_LAYERS}
    )

    ctx = Ctx(args)
    res["metadata"]["equivalence_gate"] = equivalence_gate(ctx.bundle, args.seed)
    C.write_json_atomic(args.out_json, res)

    runners = {1: run_section1, 2: run_section2, 3: run_section3, 4: run_section4}
    for s in sections:
        logger.info("=== Section %d ===", s)
        runners[s](res, ctx)

    C.write_json_atomic(args.out_json, res)
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
