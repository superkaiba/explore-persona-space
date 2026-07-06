#!/usr/bin/env python3
"""Issue #779 scaling-curve EDGES (ridge-only, vectorized, CPU-pod friendly).

Two learning curves per (trait, mode) at the FROZEN step0 read-out layers
(evil sys L14 / many L26; sycophancy L26/L26; hallucination sys L17 / many L27):

- **N_LMSYS axis** {100, 250, 500, 1000, 2000, 5000} at N_behavior = 0 —
  training data is an N-row random subsample of the pass-B LMSYS bundle
  (single-rollout v(x) targets, LMSYS judge labels for g).
- **N_behavior axis** {50, 100, 250, 500, 1000, 2400} at N_LMSYS = 0 —
  training data is an N-context random subsample of the trait behavior corpus
  (10-rollout-mean v(x) targets, per-context mean judge labels for g).

K = 5 random subsamples per interior cell (the full-N endpoint has exactly one
distinct subset, so 1 draw there, with a bootstrap CI standing in for the
missing draw spread). Per draw, for BOTH the behavior-agnostic map h
(c_last -> v(x); dot + cosine readouts vs r_B) AND the direct predictor g
(c_last -> judge label), all GCV ridge:

- **within-condition Pearson r** on the FIXED Persona-Vectors eval rig
  (``build_eval_matrix``) per elicitation mode — point estimate per draw
  (K-draw mean +- sd is the reported spread; per-draw bootstrap is skipped
  deliberately, n_boot=0), bootstrap CI only at the single-draw full-N cells;
- **held-out recon R2** from a SINGLE 80/20 split within the subsample
  (documented simplification at K=5 — fold spread is absorbed into the draw
  spread; percontext_recon's ``_pooled_r2`` test-fold-own-mean convention).

Machinery lifted with attribution from ``scripts/issue779_arm_headline.py``
(untracked sibling work at time of authoring — not importable on a fresh
clone): ``GramRidge`` (the shared-eigh Gram/dual GCV ridge, math identical to
``issue779_percontext_recon._ridge_fit_predict_fast``), its equivalence gate,
and the corpus/bundle/label loaders. Committed helpers are imported, never
copied: ``issue779_stage1`` (load_eval_cells / build_eval_matrix /
method_metrics / _load_rb), ``issue779_percontext_recon`` (_pooled_r2),
``fit_h`` (dot/cosine readouts), ``metrics``, ``issue779_common`` (constants,
write_json_atomic, reproducibility_metadata).

Checkpointing: the results JSON is atomically rewritten after EVERY
(trait, layer, axis, N) cell; resume skips completed cells and refuses a
params mismatch (pass --fresh to overwrite). ``--hf-upload`` pushes the JSON
to the HF data repo under
``issue779_monitoring/training-source-ablation-hg/edges/`` after each trait
completes (small JSON, non-LFS path). ``--figure-only`` renders the learning
curves from an existing JSON (run VM-side; paper_plots style).

Fail loud; NaN judge labels are DROPPED with counts reported, never coerced.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_edges")

# Frozen read-out layers per (trait, mode) — step0 best_by_mode; do NOT re-select.
FROZEN_LAYERS: dict[str, dict[str, int]] = {
    "evil": {"system": 14, "many_shot": 26},
    "sycophancy": {"system": 26, "many_shot": 26},
    "hallucination": {"system": 17, "many_shot": 27},
}
MODES = ("system", "many_shot")

LMSYS_NS = (100, 250, 500, 1000, 2000, 5000)
BEHAV_NS = (50, 100, 250, 500, 1000, 2400)
AXES = {"lmsys": LMSYS_NS, "behavior": BEHAV_NS}
AXIS_ID = {"lmsys": 0, "behavior": 1}  # stable per-axis seed-sequence discriminator
MIN_G_VALID = 30  # a draw with fewer valid judge labels records NaN + counts (never coerced)

# Default data roots (the VM layout); on a pod, stage to the same relative
# layout or override via the CLI flags below.
DEFAULT_CORPUS_DIR = Path(
    os.environ.get(
        "EPS779_CORPUS_DIR", "/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus"
    )
)
DEFAULT_COLLECT_DIR = Path(
    os.environ.get(
        "EPS779_COLLECT_DIR",
        str(
            PROJECT_ROOT / "data" / "issue779_hfstage" / "issue779_monitoring" / "analysis_tensors"
        ),
    )
)
DEFAULT_LMSYS_LABELS = Path(
    os.environ.get(
        "EPS779_LMSYS_LABELS",
        str(PROJECT_ROOT / "data" / "issue_779" / "lmsys_g_labels" / "lmsys_g_labels.json"),
    )
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_EDGES_PREFIX = "issue779_monitoring/training-source-ablation-hg/edges"


# ── shared-factorization GCV ridge (lifted from issue779_arm_headline.py) ─────


class GramRidge:
    """Torch-eigh Gram/dual GCV ridge with the factorization computed ONCE.

    Lifted verbatim from ``scripts/issue779_arm_headline.py`` (sibling-authored,
    untracked at time of writing). Math is IDENTICAL to
    ``issue779_percontext_recon._ridge_fit_predict_fast`` (standardize-X on
    train stats / center-Y / GCV lambda over logspace(-2,4,13) / dual solve via
    eigh of the train Gram / un-center) — verified by :func:`equivalence_gate`
    against both that twin and the canonical ``fit_h.ridge_fit_predict``.
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

    Lifted from ``issue779_arm_headline.py``: one 500-train/100-eval slice of
    the LMSYS bundle at layer 14. Fail loud on divergence.
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


# ── data loading (lifted from issue779_arm_headline.py, paths parameterized) ──


def load_lmsys_bundle(collect_dir: Path) -> dict:
    """Load the pass-B LMSYS train bundle (mmap when possible)."""
    path = collect_dir / "pass_b" / "train_context_vectors.pt"
    try:
        bundle = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
    except RuntimeError as e:  # legacy (non-zipfile) serialization cannot mmap
        logger.warning("mmap load of %s failed (%s); falling back to full load", path, e)
        bundle = torch.load(path, weights_only=False, map_location="cpu")
    assert bundle["cx_last"].shape[1:] == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN)
    assert bundle["v_x"].shape == bundle["cx_last"].shape
    return bundle


def load_corpus(corpus_dir: Path, trait: str) -> dict:
    """mmap-load a trait corpus blob + verify the index semantics by shape."""
    blob = torch.load(
        corpus_dir / f"{trait}_corpus.pt", mmap=True, weights_only=False, map_location="cpu"
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


def corpus_scores(corpus_dir: Path, trait: str) -> np.ndarray:
    """Per-(context, rollout) judge scores as (n_ctx, n_rollouts), NaN = dropped."""
    with open(corpus_dir / f"{trait}_judge_scores.json") as f:
        raw = json.load(f)["scores"]
    n_ctx = len(raw)
    n_r = len(next(iter(raw.values())))
    out = np.full((n_ctx, n_r), np.nan)
    for ctx_s, rolls in raw.items():
        for ri_s, v in rolls.items():
            if v is not None:
                out[int(ctx_s), int(ri_s)] = float(v)
    return out


def lmsys_labels(labels_path: Path, trait: str, n_expected: int) -> np.ndarray:
    """Per-context LMSYS judge label (NaN = judge-dropped), index-aligned to pass B."""
    with open(labels_path) as f:
        d = json.load(f)["labels_per_trait"][trait]
    labels = np.array([np.nan if v is None else float(v) for v in d["labels"]])
    assert labels.shape == (n_expected,), (labels.shape, n_expected)
    return labels


# ── lazy per-process data context ─────────────────────────────────────────────


class Ctx:
    """Lazy cache of the LMSYS bundle, corpora, rig matrices, and r_B."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._bundle: dict | None = None
        self._corpus: dict[str, dict] = {}
        self._scores: dict[str, np.ndarray] = {}
        self._cells: dict[str, list] = {}
        self._rb: dict[str, np.ndarray] = {}
        self._mats: dict[tuple[str, int], dict] = {}

    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            logger.info("Loading LMSYS pass-B bundle")
            self._bundle = load_lmsys_bundle(self.args.collect_dir)
        return self._bundle

    def corpus(self, trait: str) -> dict:
        if trait not in self._corpus:
            logger.info("[%s] mmap-loading behavior corpus", trait)
            self._corpus[trait] = load_corpus(self.args.corpus_dir, trait)
        return self._corpus[trait]

    def scores(self, trait: str) -> np.ndarray:
        if trait not in self._scores:
            self._scores[trait] = corpus_scores(self.args.corpus_dir, trait)
        return self._scores[trait]

    def rb(self, trait: str) -> np.ndarray:
        if trait not in self._rb:
            self._rb[trait] = S1._load_rb(
                self.args.collect_dir / "r_b", trait, C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN
            )
        return self._rb[trait]

    def mat(self, trait: str, li: int) -> dict:
        key = (trait, li)
        if key not in self._mats:
            if trait not in self._cells:
                self._cells[trait] = S1.load_eval_cells(self.args.collect_dir / "pass_a", trait)
            self._mats[key] = S1.build_eval_matrix(self._cells[trait], li, self.rb(trait))
        return self._mats[key]

    def corpus_layer(self, trait: str, li: int) -> tuple[np.ndarray, np.ndarray]:
        """(Xb (n_ctx, H), Yb_mean10 (n_ctx, H)) at one layer, fp32."""
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
        return xb, vb.mean(axis=1)

    def lmsys_layer(self, li: int) -> tuple[np.ndarray, np.ndarray]:
        """(Xa, Ya) at one layer, fp32 numpy."""
        col = self.bundle["layers"].index(li)
        xa = self.bundle["cx_last"][:, col, :].to(torch.float32).numpy()
        ya = self.bundle["v_x"][:, col, :].to(torch.float32).numpy()
        return xa, ya


# ── per-cell fits ─────────────────────────────────────────────────────────────


def _mode_point_r(x: np.ndarray, mat: dict, modes: list[str], n_boot: int, seed: int) -> dict:
    """Per-mode within-condition r for one monitor (point; CI only when n_boot>0)."""
    mm = S1.method_metrics(x, mat, n_boot=n_boot, seed=seed)
    out = {}
    for mode in modes:
        e = {"point": mm[mode]["point"], "n_conditions": mm[mode]["n_conditions"]}
        if n_boot > 0:
            e["lo"], e["hi"] = mm[mode]["lo"], mm[mode]["hi"]
        out[mode] = e
    return out


def run_draw(
    X: np.ndarray,
    Y: np.ndarray,
    gy: np.ndarray,
    idx: np.ndarray,
    split_rng: np.random.Generator,
    mat: dict,
    rb_l: np.ndarray,
    modes: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    """One subsample draw: h + g fits -> rig readouts + single-80/20-split recon."""
    Xev = mat["c_last"]
    n = len(idx)
    out: dict = {"n": int(n)}

    # h: full-draw fit for the rig readout.
    gr_h = GramRidge(X[idx])
    pred = gr_h.predict(Y[idx], Xev)
    out["h_dot"] = _mode_point_r(F.dot_readout(pred, rb_l), mat, modes, n_boot, seed)
    out["h_cos"] = _mode_point_r(F.cosine_readout(pred, rb_l), mat, modes, n_boot, seed)
    out["h_gcv_lambda"] = gr_h.last_lambda

    # g: fit on the VALID-labeled rows of the draw only (drop-never-coerce).
    fin = np.isfinite(gy[idx])
    out["g_n_valid"] = int(fin.sum())
    out["g_n_dropped"] = int((~fin).sum())
    if fin.sum() >= MIN_G_VALID:
        gr_g = GramRidge(X[idx][fin])
        out["g"] = _mode_point_r(gr_g.predict(gy[idx][fin], Xev), mat, modes, n_boot, seed)
        out["g_gcv_lambda"] = gr_g.last_lambda
    else:
        out["g"] = {m: {"point": float("nan"), "n_conditions": 0} for m in modes}
        logger.warning("g fit SKIPPED (n_valid=%d < %d)", int(fin.sum()), MIN_G_VALID)

    # Held-out recon: single 80/20 split within the draw (documented at K=5).
    perm = split_rng.permutation(n)
    n_tr = round(0.8 * n)
    tr, te = idx[perm[:n_tr]], idx[perm[n_tr:]]
    gr_r = GramRidge(X[tr])
    out["h_recon_r2"] = R._pooled_r2(gr_r.predict(Y[tr], X[te]), Y[te])
    fin_tr, fin_te = np.isfinite(gy[tr]), np.isfinite(gy[te])
    if fin_tr.sum() >= MIN_G_VALID and fin_te.sum() >= 5:
        gr_gr = GramRidge(X[tr][fin_tr])
        pred_g = gr_gr.predict(gy[tr][fin_tr], X[te][fin_te])
        out["g_recon_r2"] = R._pooled_r2(pred_g[:, None], gy[te][fin_te][:, None])
    else:
        out["g_recon_r2"] = float("nan")
    return out


def _summary(draws: list[dict], modes: list[str]) -> dict:
    """K-draw mean +- sd per metric (NaN draws propagate via nanmean/nanstd)."""
    out: dict = {}
    with np.errstate(invalid="ignore"):
        for key in ("h_dot", "h_cos", "g"):
            out[key] = {}
            for m in modes:
                vals = np.array([d[key][m]["point"] for d in draws], dtype=float)
                out[key][m] = {
                    "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                    "sd": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
                    "n_draws_finite": int(np.isfinite(vals).sum()),
                }
        for key in ("h_recon_r2", "g_recon_r2"):
            vals = np.array([d[key] for d in draws], dtype=float)
            out[key] = {
                "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan"),
                "sd": float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan"),
                "n_draws_finite": int(np.isfinite(vals).sum()),
            }
    return out


def run_edges(
    res: dict,
    ctx: Ctx,
    traits: tuple[str, ...] = C.TRAITS,
    axes: dict[str, tuple[int, ...]] = AXES,
) -> None:
    """All (trait, layer, axis, N) cells, checkpointed per cell, HF-push per trait."""
    args = ctx.args
    sec = res.setdefault("edges", {})
    t_start = time.time()
    for trait in traits:
        tr = sec.setdefault(trait, {})
        layers = sorted(set(FROZEN_LAYERS[trait].values()))
        for li in layers:
            lkey = f"L{li}"
            lentry = tr.setdefault(lkey, {})
            modes_here = [m for m in MODES if FROZEN_LAYERS[trait][m] == li]
            lentry["modes"] = modes_here
            mat = ctx.mat(trait, li)
            rb_l = ctx.rb(trait)[li]

            for axis, ns in axes.items():
                akey = f"{axis}_axis"
                aentry = lentry.setdefault(akey, {})
                if axis == "lmsys":
                    X, Y = ctx.lmsys_layer(li)
                    gy = lmsys_labels(args.lmsys_labels, trait, X.shape[0])
                else:
                    X, Y = ctx.corpus_layer(trait, li)
                    with np.errstate(invalid="ignore"):
                        gy = np.nanmean(ctx.scores(trait), axis=1)
                n_total = X.shape[0]
                assert n_total >= max(ns), (trait, axis, n_total, max(ns))

                for n in ns:
                    nkey = str(n)
                    if nkey in aentry:
                        logger.info("[%s %s %s N=%d] checkpointed; skip", trait, lkey, axis, n)
                        continue
                    k = 1 if n == n_total else args.k_draws
                    n_boot = args.n_boot if n == n_total else 0
                    draws = []
                    for ki in range(k):
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [args.seed, C.TRAITS.index(trait), li, AXIS_ID[axis], n, ki]
                            )
                        )
                        idx = (
                            np.arange(n_total)
                            if n == n_total
                            else np.sort(rng.choice(n_total, size=n, replace=False))
                        )
                        t0 = time.time()
                        d = run_draw(X, Y, gy, idx, rng, mat, rb_l, modes_here, n_boot, args.seed)
                        d["draw"] = ki
                        draws.append(d)
                        logger.info(
                            "[%s %s %s N=%d draw %d/%d] %.1fs | h_cos %s | g %s | "
                            "recon h=%.3f g=%.3f",
                            trait,
                            lkey,
                            axis,
                            n,
                            ki + 1,
                            k,
                            time.time() - t0,
                            {m: round(d["h_cos"][m]["point"], 3) for m in modes_here},
                            {m: round(d["g"][m]["point"], 3) for m in modes_here},
                            d["h_recon_r2"],
                            d["g_recon_r2"],
                        )
                    aentry[nkey] = {
                        "n_draws": k,
                        "draws": draws,
                        "summary": _summary(draws, modes_here),
                    }
                    C.write_json_atomic(args.out_json, res)
            logger.info(
                "[%s %s] done (elapsed %.1f min total)", trait, lkey, (time.time() - t_start) / 60
            )
        if args.hf_upload:
            upload_edges_json(args.out_json, note=f"{trait} complete")
    C.write_json_atomic(args.out_json, res)


def upload_edges_json(out_json: Path, note: str) -> None:
    """Push the (small, non-LFS) edges JSON to the HF data repo. Fail loud."""
    from huggingface_hub import HfApi

    api = HfApi()
    dest = f"{HF_EDGES_PREFIX}/{out_json.name}"
    api.upload_file(
        path_or_fileobj=str(out_json),
        path_in_repo=dest,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue779 edges: {note}",
    )
    logger.info("Uploaded %s -> %s (%s)", out_json.name, dest, note)


# ── figure (VM-side; paper_plots style) ───────────────────────────────────────


def make_edges_figure(res: dict, fig_dir: Path) -> None:
    """Learning curves: r vs N per method (h_cos / h_dot / g), per trait x mode."""
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
    method_labels = {"h_cos": "h (cosine readout)", "h_dot": "h (dot readout)", "g": "g (direct)"}
    for axis, ns in AXES.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="tight")
        for col, trait in enumerate(C.TRAITS):
            for row, mode in enumerate(MODES):
                ax = axes[row][col]
                li = FROZEN_LAYERS[trait][mode]
                aentry = res["edges"][trait][f"L{li}"][f"{axis}_axis"]
                for mi, mkey in enumerate(("h_cos", "h_dot", "g")):
                    xs, means, sds = [], [], []
                    for n in ns:
                        cell = aentry.get(str(n))
                        if cell is None:
                            continue
                        s = cell["summary"][mkey][mode]
                        if not np.isfinite(s["mean"]):
                            continue
                        xs.append(n)
                        means.append(s["mean"])
                        sds.append(s["sd"])
                    ax.errorbar(
                        xs,
                        means,
                        yerr=sds,
                        marker="o",
                        ms=4,
                        capsize=3,
                        color=colors[mi],
                        label=method_labels[mkey],
                    )
                ax.set_xscale("log")
                ax.set_xticks(list(ns))
                ax.set_xticklabels([str(n) for n in ns], rotation=45)
                ax.minorticks_off()
                src = "LMSYS contexts" if axis == "lmsys" else "trait-corpus contexts"
                mode_lbl = "system prompting" if mode == "system" else "many-shot"
                ax.set_title(f"{trait} — {mode_lbl} (L{li})")
                if col == 0:
                    ax.set_ylabel("within-condition Pearson r")
                if row == 1:
                    ax.set_xlabel(f"training {src} (N)")
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="best")
        figs = savefig_paper(fig, f"edges_{axis}_axis", dir=fig_dir)
        plt.close(fig)
        res.setdefault("figures", {})[f"edges_{axis}_axis"] = str(figs.get("png", ""))
        logger.info("Wrote %s", figs.get("png"))

    make_recon_figure(res, fig_dir)


def make_recon_figure(res: dict, fig_dir: Path) -> None:
    """Recon R2 companion figure (per axis, per trait; layer = the mode-union set)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="tight")
    for col, trait in enumerate(C.TRAITS):
        for row, axis in enumerate(AXES):
            ax = axes[row][col]
            ns = AXES[axis]
            for mi, (mkey, lbl) in enumerate(
                (("h_recon_r2", "h profile recon"), ("g_recon_r2", "g label recon"))
            ):
                for li in sorted(set(FROZEN_LAYERS[trait].values())):
                    aentry = res["edges"][trait][f"L{li}"][f"{axis}_axis"]
                    xs, means, sds = [], [], []
                    for n in ns:
                        cell = aentry.get(str(n))
                        if cell is None or not np.isfinite(cell["summary"][mkey]["mean"]):
                            continue
                        xs.append(n)
                        means.append(cell["summary"][mkey]["mean"])
                        sds.append(cell["summary"][mkey]["sd"])
                    ax.errorbar(
                        xs,
                        means,
                        yerr=sds,
                        marker="o",
                        ms=4,
                        capsize=3,
                        color=paper_palette(2)[mi],
                        ls="-" if li == sorted(set(FROZEN_LAYERS[trait].values()))[0] else "--",
                        label=f"{lbl} (L{li})",
                    )
            ax.set_xscale("log")
            ax.set_xticks(list(ns))
            ax.set_xticklabels([str(n) for n in ns], rotation=45)
            ax.minorticks_off()
            ax.set_title(f"{trait} — {axis} axis")
            if col == 0:
                ax.set_ylabel("held-out recon R2 (80/20)")
            if row == 1:
                ax.set_xlabel("training contexts (N)")
            ax.legend(fontsize=6, loc="best")
    figs = savefig_paper(fig, "edges_recon", dir=fig_dir)
    plt.close(fig)
    res.setdefault("figures", {})["edges_recon"] = str(figs.get("png", ""))
    logger.info("Wrote %s", figs.get("png"))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 scaling-curve edges (ridge-only).")
    parser.add_argument("--k-draws", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=500, help="bootstrap only at full-N cells")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--collect-dir", type=Path, default=DEFAULT_COLLECT_DIR)
    parser.add_argument("--lmsys-labels", type=Path, default=DEFAULT_LMSYS_LABELS)
    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing output JSON")
    parser.add_argument("--figure-only", action="store_true", help="render figures from JSON")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="end-to-end validation: evil only, one tiny N per axis, k=1, separate out-json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "batch2_edges.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    args = parser.parse_args()
    torch.set_num_threads(int(args.n_threads))

    traits, axes = C.TRAITS, AXES
    if args.smoke:
        traits, axes = ("evil",), {"lmsys": (100,), "behavior": (50,)}
        args.k_draws = 1
        args.out_json = args.out_json.with_name("batch2_edges_smoke.json")
        logger.info("SMOKE mode: %s, axes=%s, k=1 -> %s", traits, axes, args.out_json)

    if args.figure_only:
        with open(args.out_json) as f:
            res = json.load(f)
        make_edges_figure(res, args.fig_dir)
        return 0

    res: dict = {}
    params = {"k_draws": args.k_draws, "n_boot": args.n_boot, "seed": args.seed}
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
        {
            "script": "issue779_edges",
            **params,
            "frozen_layers": FROZEN_LAYERS,
            "axes": {k: list(v) for k, v in AXES.items()},
            "recon_protocol": "single 80/20 split per draw (test-fold-own-mean pooled R2)",
        }
    )

    ctx = Ctx(args)
    res["metadata"]["equivalence_gate"] = equivalence_gate(ctx.bundle, args.seed)
    C.write_json_atomic(args.out_json, res)

    run_edges(res, ctx, traits=traits, axes=axes)
    if args.hf_upload:
        upload_edges_json(args.out_json, note="all traits complete")
    C.write_json_atomic(args.out_json, res)
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
