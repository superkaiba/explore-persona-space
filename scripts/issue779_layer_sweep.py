#!/usr/bin/env python3
"""Issue #779 free re-analysis: rerun the monitoring comparison at ALL 28 layers.

0-GPU-h, CPU/VM only. VECTORIZED rewrite of the earlier serial attempt (which
looped per-layer ``fit_h.ridge_fit_predict`` numpy-SVD calls, ~125 s each, and
OOM-looped holding the full float64 bundle): this driver

1. REUSES ``_ridge_fit_predict_fast`` from ``issue779_percontext_recon.py`` —
   the torch-eigh Gram-space ridge verified equivalent to
   ``fit_h.ridge_fit_predict`` (rel diff ~1e-7; its in-process equivalence gate
   is re-run here on a 500-train/100-eval slice BEFORE the sweep, fail-loud).
   ONE shared h fit per layer (h is behavior-agnostic) predicts all traits'
   stacked eval contexts; the train-side reconstruction read of
   ``fit_h_readouts`` is skipped (not consumed here), halving the fits.
2. Stays MEMORY-BOUND: the 6 GB pass-B bundle is ``torch.load(..., mmap=True)``
   (plain-load fallback); per layer ONLY the (5000, 3584) slice is converted to
   float64, freed + ``gc.collect()`` between layers; never all-layer float64
   copies. ``torch.set_num_threads(8)``. Target < 20 GB RSS.
3. CHECKPOINTS per (trait, layer): each cell row is merged into
   ``eval_results/issue_779/layer_sweep.json`` via an atomic tmp+rename write;
   on startup completed (trait, layer) pairs are SKIPPED (restart-safe).

Reuses (never reimplements): issue779_stage1.load_eval_cells / _load_rb /
build_eval_matrix / method_metrics / fit_direct_loco; fit_h.dot_readout /
cosine_readout; issue779_common TRAITS / EXPECTED_LAYERS / EXPECTED_HIDDEN /
PV_WITHIN_CONDITION_TARGETS / RIG_VALIDATION_BAND. MLP arms are DEFERRED
(GPU-worthy; the #779 body already shows the MLP underperforming the linear map).

Two reads:
  A. Full per-layer curve (descriptive, bias-free): within-condition Pearson r
     vs layer (0-27) per trait x mode for the closed-form methods, bootstrap CIs.
  B. De-biased held-out layer selection (selection-symmetric-nulls.md,
     alternative 2 applied per-fold): leave-one-condition-out — select each
     method's layer on the OTHER conditions, evaluate on the held-out condition
     at that layer, aggregate with a bootstrap over held-out conditions. Also
     recomputes the rig-validation gate with pv_raw at its own held-out-selected
     layer per trait x mode vs the PV published targets +- 0.10.
     The full per-(layer x condition) r matrices are persisted in the JSON
     (``cells.*.per_condition_r``) per the persist-the-matrix rule.

Fail loud — no try/except:pass, no dummy fills; NaN / insufficient-conditions
cells are reported as such.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_stage1 as S  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Fast torch-eigh Gram-space ridge + pooled-R2 helper, verified equivalent to
# fit_h.ridge_fit_predict (issue779_percontext_recon.py, committed 6438a882).
from issue779_percontext_recon import _pooled_r2, _ridge_fit_predict_fast  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_layer_sweep")

# Closed-form / ridge methods only (MLP arms DEFERRED — GPU-worthy).
CURVE_METHODS = (
    "pv_raw",
    "oracle",
    "r2_mean",
    "r2_max",
    "r2_topk",
    "r2_last",
    "r1_ridge_cos",
    "r1_ridge_dot",
    "direct_ridge",
)
MODES = ("system", "many_shot")
# Read-A figure subset (the brief's 6-curve panels).
FIG_METHODS = ("pv_raw", "r1_ridge_cos", "direct_ridge", "r2_max", "r2_last", "oracle")


def _key(trait: str, layer: int) -> str:
    """Checkpoint key for one (trait, layer) cell."""
    return f"{trait}|{layer}"


def _rss_gb() -> float:
    """Current peak RSS of this process in GB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ── fast-ridge equivalence gate (fail-loud, before the sweep) ─────────────────


def fast_ridge_equivalence_gate(
    cx_last: torch.Tensor, v_x: torch.Tensor, *, layer: int, seed: int
) -> dict:
    """Assert _ridge_fit_predict_fast reproduces fit_h.ridge_fit_predict.

    500-train/100-eval subsample at one layer (the issue779_percontext_recon.py
    gate, re-run in-process here so the sweep never launches on a diverged
    accelerator). Gates on relative prediction diff + held-out pooled-R2 diff,
    both < 1e-6 (raw abs diff can reach ~1e-6 from the LAPACK-route difference).
    """
    rng = np.random.default_rng(seed)
    sub = rng.choice(cx_last.shape[0], size=600, replace=False)
    X = cx_last[:, layer, :].to(torch.float64).numpy()[sub]
    Y = v_x[:, layer, :].to(torch.float64).numpy()[sub]
    pred_slow = F.ridge_fit_predict(X[:500], Y[:500], X[500:])
    pred_fast = _ridge_fit_predict_fast(X[:500], Y[:500], X[500:])
    abs_diff = float(np.max(np.abs(pred_slow - pred_fast)))
    rel_diff = abs_diff / (float(np.max(np.abs(pred_slow))) + 1e-12)
    r2_slow = _pooled_r2(pred_slow, Y[500:])
    r2_fast = _pooled_r2(pred_fast, Y[500:])
    r2_diff = abs(r2_slow - r2_fast)
    assert rel_diff < 1e-6 and r2_diff < 1e-6, (
        f"fast-ridge equivalence gate FAILED: rel_diff {rel_diff:.2e}, "
        f"R2 diff {r2_diff:.2e} (abs pred diff {abs_diff:.2e})"
    )
    logger.info(
        "fast-ridge equivalence gate PASS (rel pred diff %.2e, R2 diff %.2e, abs %.2e)",
        rel_diff,
        r2_diff,
        abs_diff,
    )
    return {"rel_pred_diff": rel_diff, "r2_diff": r2_diff, "abs_pred_diff": abs_diff, "pass": True}


# ── per-(trait, layer) cell ───────────────────────────────────────────────────


def per_condition_r(x: np.ndarray, mat: dict, mode: str) -> dict[str, float | None]:
    """Per-condition within-condition Pearson r for one monitor at one layer.

    Mirrors metrics.within_condition_pearson's pruning (>= 3 finite points,
    y std >= 1, non-degenerate x) so a pruned condition is None. Keys are
    str(cond_id) (JSON round-trip stable); cond ids are deterministic across
    layers (build_eval_matrix assigns them in sorted first-seen cell order).
    """
    out: dict[str, float | None] = {}
    sel = np.array([m == mode for m in mat["mode"]])
    if not sel.any():
        return out
    y = mat["y"]
    for c in np.unique(mat["cond"][sel]):
        m = sel & (mat["cond"] == c)
        xi, yi = x[m], y[m]
        fin = np.isfinite(xi) & np.isfinite(yi)
        xi, yi = xi[fin], yi[fin]
        if len(yi) < 3 or float(np.std(yi)) < 1.0 or float(np.std(xi)) == 0.0:
            out[str(int(c))] = None
            continue
        rr = float(np.corrcoef(xi, yi)[0, 1])
        out[str(int(c))] = rr if np.isfinite(rr) else None
    return out


def process_cell(
    trait: str,
    layer: int,
    mat: dict,
    h_eval: np.ndarray,
    rb_layer: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """One (trait, layer) checkpoint row: method metrics + per-condition r matrix row.

    ``h_eval`` is the shared per-layer fast-ridge h prediction for THIS trait's
    eval contexts (N_ev, H); the R1 readouts use the canonical fit_h leaf
    functions on it. The train-side reconstruction refit of fit_h_readouts is
    deliberately skipped (not read by this sweep).
    """
    n_q = len(mat["y"])
    assert n_q >= 3, f"{trait} layer {layer}: only {n_q} (condition, question) rows"
    direct = S.fit_direct_loco(mat, run_mlp=False, pca_k=64)
    monitors = {
        "pv_raw": mat["pv_raw"],
        "oracle": mat["oracle"],
        "r2_mean": mat["r2_mean"],
        "r2_max": mat["r2_max"],
        "r2_topk": mat["r2_topk"],
        "r2_last": mat["r2_last"],
        "r1_ridge_cos": F.cosine_readout(h_eval, rb_layer),
        "r1_ridge_dot": F.dot_readout(h_eval, rb_layer),
        "direct_ridge": direct["direct_ridge"],
    }
    methods = {
        name: S.method_metrics(x, mat, n_boot=n_boot, seed=seed) for name, x in monitors.items()
    }
    per_cond = {
        name: {mode: per_condition_r(x, mat, mode) for mode in MODES}
        for name, x in monitors.items()
    }
    return {"n_questions": n_q, "methods": methods, "per_condition_r": per_cond}


# ── Read A: full per-layer curves (assembled from checkpoint cells) ───────────


def assemble_read_a(cells: dict, trait: str, n_layers: int) -> dict:
    """{method: {mode: [{layer, point, lo, hi, n_conditions}]}} from cell rows."""
    curve: dict = {m: {mode: [] for mode in MODES} for m in CURVE_METHODS}
    for layer in range(n_layers):
        row = cells[_key(trait, layer)]
        for meth in CURVE_METHODS:
            res = row["methods"][meth]
            for mode in MODES:
                mm = res[mode]
                curve[meth][mode].append(
                    {
                        "layer": layer,
                        "point": mm["point"],
                        "lo": mm["lo"],
                        "hi": mm["hi"],
                        "n_conditions": mm["n_conditions"],
                    }
                )
    return curve


# ── Read B: de-biased held-out (leave-one-condition-out) layer selection ──────


def _r_matrix(cells: dict, trait: str, meth: str, mode: str, n_layers: int):
    """(n_layers, n_conditions) per-condition r matrix + sorted cond ids."""
    cond_ids = sorted(
        {
            int(c)
            for li in range(n_layers)
            for c in cells[_key(trait, li)]["per_condition_r"][meth][mode]
        }
    )
    R = np.full((n_layers, len(cond_ids)), np.nan)
    for li in range(n_layers):
        d = cells[_key(trait, li)]["per_condition_r"][meth][mode]
        for j, c in enumerate(cond_ids):
            v = d.get(str(c))
            if v is not None:
                R[li, j] = v
    return R, cond_ids


def read_b_heldout(cells: dict, trait: str, n_layers: int, *, n_boot: int, seed: int) -> dict:
    """LOCO layer selection: pick each method's layer on the other conditions,
    score the held-out condition at that layer (selection-symmetric-nulls.md
    alternative 2, per-fold). CI = bootstrap over held-out conditions. many_shot
    has only 5 conditions, so its split is flagged fragile (n_conditions <= 4
    after pruning => prefer the full Read-A curve).
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for meth in CURVE_METHODS:
        out[meth] = {}
        for mode in MODES:
            R, cond_ids = _r_matrix(cells, trait, meth, mode, n_layers)
            held_r, per_cond = [], []
            for j, c in enumerate(cond_ids):
                other = [k for k in range(len(cond_ids)) if k != j]
                with np.errstate(invalid="ignore"):
                    layer_mean = (
                        np.nanmean(R[:, other], axis=1) if other else np.full(n_layers, np.nan)
                    )
                if np.all(np.isnan(layer_mean)):
                    continue
                l_star = int(np.nanargmax(layer_mean))
                r_held = R[l_star, j]
                per_cond.append(
                    {
                        "cond": int(c),
                        "selected_layer": l_star,
                        "r": float(r_held) if np.isfinite(r_held) else None,
                    }
                )
                if np.isfinite(r_held):
                    held_r.append(float(r_held))
            if not held_r:
                out[meth][mode] = {
                    "point": float("nan"),
                    "lo": float("nan"),
                    "hi": float("nan"),
                    "n_conditions": 0,
                    "per_condition": per_cond,
                    "note": "no held-out condition produced a finite selected-layer r",
                }
                continue
            arr = np.array(held_r)
            idx = np.arange(len(arr))
            boot = [
                float(np.mean(arr[rng.choice(idx, size=len(idx), replace=True)]))
                for _ in range(n_boot)
            ]
            entry = {
                "point": float(np.mean(arr)),
                "lo": float(np.quantile(boot, 0.025)),
                "hi": float(np.quantile(boot, 0.975)),
                "n_conditions": len(arr),
                "per_condition": per_cond,
                "note": (
                    "leave-one-condition-out: layer selected on the other conditions of the "
                    "same mode, r scored on the held-out condition at that layer; CI = "
                    "bootstrap over held-out conditions"
                ),
            }
            if len(arr) <= 4:
                entry["stability_note"] = (
                    f"FRAGILE: only {len(arr)} held-out conditions (many_shot has 5 total; "
                    "selection sees <= 4) — prefer the full Read-A curve for this cell"
                )
            out[meth][mode] = entry
    return out


def read_b_delta_vs_pv(read_b: dict, *, n_boot: int, seed: int) -> dict:
    """Held-out per-own-layer paired delta (method - pv_raw) by condition.

    Pairs each held-out condition's r under the method (at ITS held-out-selected
    layer) with the same condition's r under pv_raw (at pv_raw's held-out-selected
    layer) — the honest de-biased "does X beat the raw projection when each is
    read at its own layer" comparison. CI = bootstrap over paired conditions.
    """
    rng = np.random.default_rng(seed)
    out: dict = {}
    for meth in CURVE_METHODS:
        if meth == "pv_raw":
            continue
        out[meth] = {}
        for mode in MODES:
            a = {pc["cond"]: pc["r"] for pc in read_b[meth][mode]["per_condition"]}
            b = {pc["cond"]: pc["r"] for pc in read_b["pv_raw"][mode]["per_condition"]}
            paired = [(a[c], b[c]) for c in a if c in b and a[c] is not None and b[c] is not None]
            if not paired:
                out[meth][mode] = {
                    "delta": float("nan"),
                    "lo": float("nan"),
                    "hi": float("nan"),
                    "excludes_zero": False,
                    "n_paired": 0,
                }
                continue
            diff = np.array([p[0] - p[1] for p in paired])
            idx = np.arange(len(diff))
            boot = [
                float(np.mean(diff[rng.choice(idx, size=len(idx), replace=True)]))
                for _ in range(n_boot)
            ]
            lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            out[meth][mode] = {
                "delta": float(np.mean(diff)),
                "lo": lo,
                "hi": hi,
                "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "n_paired": len(diff),
            }
    return out


# ── rig-validation gate at pv_raw's own layer ─────────────────────────────────


def rig_gate_own_layer(curve: dict, read_b: dict, trait: str) -> dict:
    """PV rig gate with pv_raw at (a) its full-sweep argmax layer (optimistic,
    in-sample selection) and (b) its held-out-selected layer (de-biased), each
    vs the PV published within-condition target +- RIG_VALIDATION_BAND.
    """
    targets = C.PV_WITHIN_CONDITION_TARGETS[trait]
    out: dict = {"trait": trait, "checks": {}}
    for mode in MODES:
        pts = curve["pv_raw"][mode]
        finite = [(p["layer"], p["point"]) for p in pts if np.isfinite(p["point"])]
        best_layer, best_pt = max(finite, key=lambda t: t[1]) if finite else (None, float("nan"))
        target = targets[mode]
        heldout_pt = read_b["pv_raw"][mode]["point"]
        out["checks"][mode] = {
            "pv_target": target,
            "full_sweep_best_layer": best_layer,
            "full_sweep_best_r": float(best_pt) if np.isfinite(best_pt) else None,
            "full_sweep_within_band": bool(
                np.isfinite(best_pt) and abs(best_pt - target) <= C.RIG_VALIDATION_BAND
            ),
            "heldout_selected_r": float(heldout_pt) if np.isfinite(heldout_pt) else None,
            "heldout_within_band": bool(
                np.isfinite(heldout_pt) and abs(heldout_pt - target) <= C.RIG_VALIDATION_BAND
            ),
        }
    return out


# ── figures ───────────────────────────────────────────────────────────────────


def make_curve_figures(results: dict, fig_dir: Path) -> list[str]:
    """Per trait x mode: within-condition r vs layer for the 6 headline methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); default style", e)

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for trait, tr in results["traits"].items():
        curve = tr["read_a_curve"]
        for mode in MODES:
            fig, ax = plt.subplots(figsize=(9, 5))
            any_line = False
            for meth in FIG_METHODS:
                pts = curve[meth][mode]
                xs = [p["layer"] for p in pts]
                ys = np.array(
                    [p["point"] if p["point"] is not None else np.nan for p in pts], dtype=float
                )
                if not np.any(np.isfinite(ys)):
                    continue
                (line,) = ax.plot(xs, ys, marker="o", ms=3, label=meth)
                lo = np.array(
                    [p["lo"] if p["lo"] is not None else np.nan for p in pts], dtype=float
                )
                hi = np.array(
                    [p["hi"] if p["hi"] is not None else np.nan for p in pts], dtype=float
                )
                ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.12)
                any_line = True
            if not any_line:
                plt.close(fig)
                continue
            tgt = C.PV_WITHIN_CONDITION_TARGETS[trait][mode]
            ax.axhline(tgt, ls="--", color="gray", lw=1, label=f"PV published {tgt:.3f}")
            ax.axhline(0, ls=":", color="black", lw=0.8)
            ax.set_xlabel("read-out layer (0-27)")
            ax.set_ylabel("within-condition Pearson r")
            ax.set_title(f"issue 779 layer sweep — {trait} — {mode}")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            path = fig_dir / f"layer_sweep_{trait}_{mode}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(str(path))
    return saved


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #779 all-28-layer re-analysis (vectorized).")
    ap.add_argument(
        "--collect-dir",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "issue779_hfstage"
        / "issue779_monitoring"
        / "analysis_tensors",
    )
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "layer_sweep.json",
    )
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    ap.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    ap.add_argument("--hidden", type=int, default=C.EXPECTED_HIDDEN)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(int(args.n_threads))
    collect_dir = args.collect_dir
    pass_a_dir = collect_dir / "pass_a"
    rb_dir = collect_dir / "r_b"
    bundle_path = collect_dir / "pass_b" / "train_context_vectors.pt"
    step0_path = collect_dir / "step0" / "step0_oracle.json"
    for p in (pass_a_dir, rb_dir, bundle_path, step0_path):
        if not p.exists():
            raise FileNotFoundError(f"required staged input missing: {p}")

    # ── checkpoint resume ──
    results: dict = {"cells": {}}
    if args.out_json.exists():
        with open(args.out_json) as f:
            prior = json.load(f)
        results["cells"] = prior.get("cells", {})
        logger.info("resuming: %d completed (trait, layer) cells found", len(results["cells"]))

    # ── memory-bound bundle load (mmap; plain-load fallback) ──
    try:
        bundle = torch.load(bundle_path, weights_only=False, mmap=True)
        mmap_used = True
    except (RuntimeError, ValueError) as e:
        logger.warning("mmap load failed (%s: %s); falling back to plain load", type(e).__name__, e)
        bundle = torch.load(bundle_path, weights_only=False)
        mmap_used = False
    cx_last, v_x = bundle["cx_last"], bundle["v_x"]
    layers_list = list(bundle["layers"])
    assert layers_list == list(range(args.n_layers)), layers_list
    assert cx_last.shape[1:] == (args.n_layers, args.hidden), cx_last.shape
    n_train = cx_last.shape[0]
    logger.info("pass-B bundle: %d train contexts, mmap=%s", n_train, mmap_used)

    traits = args.traits
    rb_by_trait = {t: S._load_rb(rb_dir, t, args.n_layers, args.hidden) for t in traits}
    cells_by_trait = {t: S.load_eval_cells(pass_a_dir, t) for t in traits}
    for t in traits:
        for cell in cells_by_trait[t]:
            assert cell["_layers"] == layers_list, (t, cell["cond_id"], cell["_layers"])
            del cell["_cx_mean"]  # unused by this sweep; ~300 MB saved
    with open(step0_path) as f:
        step0 = json.load(f)

    # ── fail-loud equivalence gate before any sweep fit ──
    gate = fast_ridge_equivalence_gate(cx_last, v_x, layer=14, seed=args.seed)

    results["meta"] = C.reproducibility_metadata(
        {
            "script": "issue779_layer_sweep",
            "mlp": "DEFERRED (GPU-worthy; run_mlp=False everywhere)",
            "mmap_used": mmap_used,
            "n_train_contexts": int(n_train),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "fast_ridge_equivalence_gate": gate,
        }
    )
    results["method_note"] = (
        "closed-form / ridge methods only; MLP arms DEFERRED (GPU-worthy). "
        "read_a_curve = full per-layer within-condition r (descriptive; an in-sample "
        "cross-layer argmax on it is optimistic). read_b_heldout = leave-one-condition-out "
        "layer selection (de-biased, selection-symmetric-nulls.md alternative 2). "
        "cells.*.per_condition_r persists the full per-(layer x condition) r matrices."
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # ── the sweep: one shared fast-ridge h fit per layer, per-(trait, layer) rows ──
    C.phase("sweep")
    for layer in range(args.n_layers):
        missing = [t for t in traits if _key(t, layer) not in results["cells"]]
        if not missing:
            continue
        t0 = time.time()
        mats = {t: S.build_eval_matrix(cells_by_trait[t], layer, rb_by_trait[t]) for t in missing}
        # ONLY this layer's slices go to float64; freed before the next layer.
        Xtr = cx_last[:, layer, :].to(torch.float64).numpy()
        Ytr = v_x[:, layer, :].to(torch.float64).numpy()
        Xev = np.concatenate([mats[t]["c_last"] for t in missing]).astype(np.float64)
        h_all = _ridge_fit_predict_fast(Xtr, Ytr, Xev)
        del Xtr, Ytr, Xev
        off = 0
        for t in missing:
            mat = mats[t]
            n_ev = mat["c_last"].shape[0]
            h_t = h_all[off : off + n_ev]
            off += n_ev
            row = process_cell(
                t, layer, mat, h_t, rb_by_trait[t][layer], n_boot=args.n_boot, seed=args.seed
            )
            results["cells"][_key(t, layer)] = row
            C.write_json_atomic(args.out_json, results)  # checkpoint per (trait, layer)
        del h_all, mats
        gc.collect()
        logger.info(
            "layer %2d done in %5.1fs (traits %s) | peak RSS %.1f GB",
            layer,
            time.time() - t0,
            ",".join(missing),
            _rss_gb(),
        )

    # ── reads A + B, deltas, rig gate ──
    C.phase("reads")
    results["traits"] = {}
    heldout_pass = 0
    for t in traits:
        curve = assemble_read_a(results["cells"], t, args.n_layers)
        read_b = read_b_heldout(
            results["cells"], t, args.n_layers, n_boot=args.n_boot, seed=args.seed
        )
        deltas = read_b_delta_vs_pv(read_b, n_boot=args.n_boot, seed=args.seed)
        rig = rig_gate_own_layer(curve, read_b, t)
        heldout_pass += sum(int(rig["checks"][m]["heldout_within_band"]) for m in MODES)
        results["traits"][t] = {
            "step0_reference_best_layer": step0[t]["best_layer"],
            "read_a_curve": curve,
            "read_b_heldout": read_b,
            "read_b_deltas_vs_pv": deltas,
            "rig_gate_own_layer": rig,
        }
    results["rig_gate_heldout_pass_count_of_6"] = heldout_pass
    C.write_json_atomic(args.out_json, results)

    C.phase("figures")
    figs = make_curve_figures(results, args.fig_dir)
    results["figures"] = figs
    results["meta"]["peak_rss_gb"] = _rss_gb()
    C.write_json_atomic(args.out_json, results)
    logger.info("Wrote %d figures + %s | peak RSS %.1f GB", len(figs), args.out_json, _rss_gb())

    # ── console summary ──
    for t in traits:
        tr = results["traits"][t]
        print(f"\n===== {t} (step0 reference layer {tr['step0_reference_best_layer']}) =====")
        for mode in MODES:
            parts = []
            for meth in FIG_METHODS:
                hb = tr["read_b_heldout"][meth][mode]
                parts.append(f"{meth}={hb['point']:.3f}")
            print(f"  heldout-own-layer [{mode}]: " + "  ".join(parts))
            ck = tr["rig_gate_own_layer"]["checks"][mode]
            print(
                f"  rig gate [{mode}]: target {ck['pv_target']:.3f} | heldout "
                f"{ck['heldout_selected_r']} (pass={ck['heldout_within_band']}) | full-sweep "
                f"best L{ck['full_sweep_best_layer']} {ck['full_sweep_best_r']} "
                f"(pass={ck['full_sweep_within_band']})"
            )
    print(f"\nrig gate held-out pass count: {heldout_pass}/6")
    C.phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
