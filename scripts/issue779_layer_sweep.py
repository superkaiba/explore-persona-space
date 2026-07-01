#!/usr/bin/env python3
"""Issue #779 free re-analysis: rerun the monitoring comparison at ALL 28 layers.

0-GPU-h re-analysis. Reuses the Stage-1 readout functions in issue779_stage1.py
(build_eval_matrix / load_eval_cells / _load_rb / fit_h_readouts / fit_direct_loco
/ method_metrics / _group_by_condition / gate1_rig_validation) — this is a THIN
loop over those functions, NOT a reimplementation. MLP arms are DEFERRED
(run_mlp=False everywhere): the #779 body already shows the MLP underperforms the
linear map, and the MLP arm is GPU-worthy, so only the closed-form / ridge methods
run here.

Two reads (see the task brief):
  A. Full per-layer curve (bias-free, descriptive): within-condition Pearson r vs
     layer (0..27) for pv_raw / oracle / direct_ridge / r1_ridge_cos / r1_ridge_dot
     / r2_max / r2_topk / r2_last, per trait x mode, with bootstrap CIs.
  B. Held-out layer selection (de-biased), per selection-symmetric-nulls.md: for
     each method, select the read-out layer on a held-out fold of CONDITIONS
     (leave-one-condition-out), then read that method at its held-out-selected
     layer on the complementary fold; aggregate. Also recompute the rig-validation
     gate with pv_raw read at its own held-out-selected best layer per trait x mode.

Writes eval_results/issue_779/layer_sweep.json (checkpointed per trait) + figures.
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
import issue779_stage1 as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_layer_sweep")

# Closed-form / ridge methods only (MLP deferred). Names index into the per-layer
# monitor dict built below.
CURVE_METHODS = (
    "pv_raw",
    "oracle",
    "direct_ridge",
    "r1_ridge_cos",
    "r1_ridge_dot",
    "r2_max",
    "r2_topk",
    "r2_last",
)
MODES = ("system", "many_shot")


# ── per-layer monitors (thin wrapper over the Stage-1 readouts) ───────────────


def monitors_from_mat(mat, h_ridge_eval, r_b_layer):
    """Assemble the closed-form monitor arrays for one trait at one read-out layer.

    ``mat`` is build_eval_matrix's per-(condition,question) bundle (carries y /
    cond / mode + pv_raw / oracle / r2_* monitors). ``h_ridge_eval`` is the R1
    map's predicted answer-profile for THIS trait's eval contexts (N_ev, H) — the
    predictions from the shared per-layer ridge fit (see ``fit_h_shared`` /
    build_eval_matrix). ``r_b_layer`` is r_B at this layer (H,). run_mlp=False
    everywhere (MLP deferred).

    The R1 readout uses the SAME canonical leaf functions the Stage-1
    ``fit_h_readouts`` uses (``F.cosine_readout`` / ``F.dot_readout``); the ridge
    fit itself is done ONCE per layer, shared across traits (the train-side GCV-SVD
    of the pass-B contexts is trait-independent), then applied per-trait — this is
    numerically identical to per-trait ``F.ridge_fit_predict`` calls (same train
    fit, batched prediction). The in-sample reconstruction refit that
    ``fit_h_readouts`` computes for its R3 diagnostic is skipped (not read here).
    """
    r1_ridge_cos = F.cosine_readout(h_ridge_eval, r_b_layer)
    r1_ridge_dot = F.dot_readout(h_ridge_eval, r_b_layer)
    direct = S.fit_direct_loco(mat, run_mlp=False, pca_k=64)
    monitors = {
        "pv_raw": mat["pv_raw"],
        "oracle": mat["oracle"],
        "direct_ridge": direct["direct_ridge"],
        "r1_ridge_cos": r1_ridge_cos,
        "r1_ridge_dot": r1_ridge_dot,
        "r2_max": mat["r2_max"],
        "r2_topk": mat["r2_topk"],
        "r2_last": mat["r2_last"],
    }
    return monitors


# ── Read A: full per-layer curve is assembled inline in main() from the ─────────
#    precomputed per-layer monitors (see the monitors_by_layer loop).


# ── Read B: held-out (leave-one-condition-out) layer selection ────────────────


def _per_condition_r_by_layer(monitors_by_layer, mat_by_layer, meth, mode):
    """Build the (n_layers x n_conditions) per-condition within-condition r matrix.

    monitors_by_layer[layer][meth] is the monitor array; mat_by_layer[layer]
    carries y/cond/mode. Uses the SAME per-condition pruning (std<1, >=3 finite
    points) as within_condition_pearson so a pruned condition is NaN at that layer.
    Condition indices are consistent across layers (build_eval_matrix assigns
    cond ids deterministically in first-seen cell order, identical per layer).

    Returns (r_matrix (n_layers, n_conditions), cond_ids (n_conditions,)).
    """
    n_layers = len(monitors_by_layer)
    # condition ids present in this mode (from layer 0's mat; identical across layers)
    mat0 = mat_by_layer[0]
    sel0 = np.array([m == mode for m in mat0["mode"]])
    cond_ids = np.unique(mat0["cond"][sel0])
    R = np.full((n_layers, len(cond_ids)), np.nan)
    for li in range(n_layers):
        mat = mat_by_layer[li]
        x = monitors_by_layer[li][meth]
        y = mat["y"]
        for ci_pos, c in enumerate(cond_ids):
            m = np.array([mm == mode for mm in mat["mode"]]) & (mat["cond"] == c)
            xi = x[m]
            yi = y[m]
            fin = np.isfinite(xi) & np.isfinite(yi)
            xi, yi = xi[fin], yi[fin]
            if len(yi) < 3 or float(np.std(yi)) < 1.0 or float(np.std(xi)) == 0.0:
                continue
            rr = float(np.corrcoef(xi, yi)[0, 1])
            if np.isfinite(rr):
                R[li, ci_pos] = rr
    return R, cond_ids


def read_b_heldout(monitors_by_layer, mat_by_layer, *, n_boot, seed):
    """De-biased per-own-layer read via leave-one-condition-out layer selection.

    For each (method, mode): build the per-condition r-by-layer matrix R
    (n_layers x n_conditions). For each held-out condition c:
      - SELECT the read-out layer on the OTHER conditions:
        l*(c) = argmax_layer mean_{c' != c, finite} R[layer, c']
      - EVALUATE: record R[l*(c), c]  (condition c's own within-condition r at
        the layer selected WITHOUT seeing c).
    The de-biased point is the mean of the held-out per-condition r's; the CI is a
    bootstrap over held-out conditions (resample the held-out r's). This is
    selection-symmetric-nulls.md alternative 2 (held-out axis freeze) applied
    per-fold: the layer is chosen on a disjoint split from the one it is scored on.

    Returns {method: {mode: {point, lo, hi, n_conditions, per_condition:
    [{cond, selected_layer, r}], note}}}.
    """
    rng = np.random.default_rng(seed)
    out = {m: {mode: {} for mode in MODES} for m in CURVE_METHODS}
    for meth in CURVE_METHODS:
        for mode in MODES:
            R, cond_ids = _per_condition_r_by_layer(monitors_by_layer, mat_by_layer, meth, mode)
            n_cond = len(cond_ids)
            held_r = []
            per_cond = []
            for ci_pos in range(n_cond):
                other = [j for j in range(n_cond) if j != ci_pos]
                # mean over other conditions per layer; a layer with no finite
                # other-condition r is disqualified.
                sub = R[:, other]
                with np.errstate(invalid="ignore"):
                    layer_mean = np.nanmean(sub, axis=1)  # (n_layers,)
                if np.all(np.isnan(layer_mean)):
                    continue
                l_star = int(np.nanargmax(layer_mean))
                r_held = R[l_star, ci_pos]
                per_cond.append(
                    {
                        "cond": int(cond_ids[ci_pos]),
                        "selected_layer": l_star,
                        "r": None if not np.isfinite(r_held) else float(r_held),
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
            boot = []
            idx = np.arange(len(arr))
            for _ in range(n_boot):
                samp = rng.choice(idx, size=len(idx), replace=True)
                boot.append(float(np.mean(arr[samp])))
            out[meth][mode] = {
                "point": float(np.mean(arr)),
                "lo": float(np.quantile(boot, 0.025)),
                "hi": float(np.quantile(boot, 0.975)),
                "n_conditions": len(arr),
                "per_condition": per_cond,
                "note": (
                    "leave-one-condition-out: layer selected on the other conditions "
                    "of the same mode, r scored on the held-out condition at that layer; "
                    "CI = bootstrap over held-out conditions"
                ),
            }
    return out


def read_b_delta_vs_pv(read_b):
    """Held-out per-own-layer delta (method - pv_raw), CI over held-out conditions.

    Uses the per-condition held-out r's from read_b (paired by condition) so the
    delta is the honest de-biased "does method X beat the raw projection when each
    is read at its OWN held-out-selected layer" comparison. Returns
    {method: {mode: {delta, lo, hi, excludes_zero, n_paired}}} for the R1 map +
    the generation reads vs pv_raw.
    """
    rng = np.random.default_rng(0)
    methods = ("r1_ridge_cos", "r1_ridge_dot", "direct_ridge", "r2_max", "r2_topk", "r2_last")
    out = {}
    for meth in methods:
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
            da = np.array([p[0] for p in paired])
            db = np.array([p[1] for p in paired])
            diff = da - db
            boot = []
            idx = np.arange(len(diff))
            for _ in range(1000):
                samp = rng.choice(idx, size=len(idx), replace=True)
                boot.append(float(np.mean(diff[samp])))
            lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            out[meth][mode] = {
                "delta": float(np.mean(diff)),
                "lo": lo,
                "hi": hi,
                "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "n_paired": len(diff),
            }
    return out


# ── rig-validation gate at pv_raw's OWN best layer ────────────────────────────


def rig_gate_own_layer(curve, read_b, trait):
    """Rig gate with pv_raw read at (a) its full-sweep argmax layer and (b) its
    held-out-selected layer, vs the PV published target +- band.

    Full-sweep-best is the optimistic upper bound (in-sample layer selection); the
    held-out point is the de-biased value. Returns per-mode dicts with both.
    """
    targets = C.PV_WITHIN_CONDITION_TARGETS[trait]
    out = {"trait": trait, "checks": {}}
    for mode in MODES:
        pts = curve["pv_raw"][mode]
        finite = [(p["layer"], p["point"]) for p in pts if np.isfinite(p["point"])]
        if finite:
            best_layer, best_pt = max(finite, key=lambda t: t[1])
        else:
            best_layer, best_pt = None, float("nan")
        target = targets[mode]
        heldout_pt = read_b["pv_raw"][mode]["point"]
        out["checks"][mode] = {
            "pv_target": target,
            "full_sweep_best_layer": best_layer,
            "full_sweep_best_r": None if not np.isfinite(best_pt) else float(best_pt),
            "full_sweep_within_band": bool(
                np.isfinite(best_pt) and abs(best_pt - target) <= C.RIG_VALIDATION_BAND
            ),
            "heldout_selected_r": None if not np.isfinite(heldout_pt) else float(heldout_pt),
            "heldout_within_band": bool(
                np.isfinite(heldout_pt) and abs(heldout_pt - target) <= C.RIG_VALIDATION_BAND
            ),
        }
    return out


# ── figures ───────────────────────────────────────────────────────────────────

FIG_METHODS = ("pv_raw", "r1_ridge_cos", "direct_ridge", "r2_max", "r2_last", "oracle")


def make_curve_figures(results, fig_dir):
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
                if meth not in curve:
                    continue
                pts = curve[meth][mode]
                xs = [p["layer"] for p in pts]
                ys = [p["point"] if p["point"] is not None else np.nan for p in pts]
                los = [p["lo"] for p in pts]
                his = [p["hi"] for p in pts]
                ys = np.array(ys, dtype=float)
                if not np.any(np.isfinite(ys)):
                    continue
                (line,) = ax.plot(xs, ys, marker="o", ms=3, label=meth)
                lo_a = np.array([lo if lo is not None else np.nan for lo in los], dtype=float)
                hi_a = np.array([hi if hi is not None else np.nan for hi in his], dtype=float)
                ax.fill_between(xs, lo_a, hi_a, color=line.get_color(), alpha=0.12)
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
    ap = argparse.ArgumentParser(description="Issue #779 all-28-layer re-analysis.")
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
    args = ap.parse_args()

    collect_dir = args.collect_dir
    pass_a_dir = collect_dir / "pass_a"
    rb_dir = collect_dir / "r_b"
    for p in (pass_a_dir, rb_dir, collect_dir / "pass_b" / "train_context_vectors.pt"):
        if not p.exists():
            raise FileNotFoundError(f"required staged input missing: {p}")

    train_bundle = torch.load(
        collect_dir / "pass_b" / "train_context_vectors.pt", weights_only=False
    )
    with open(collect_dir / "step0" / "step0_oracle.json") as _f:
        step0 = json.load(_f)

    results = {
        "traits": {},
        "meta": C.reproducibility_metadata(
            {"script": "issue779_layer_sweep", "mlp": "DEFERRED (GPU-worthy)"}
        ),
        "method_note": (
            "closed-form / ridge methods only; MLP arms DEFERRED (GPU-worthy). "
            "read_a_curve = full per-layer within-condition r (descriptive, in-sample "
            "layer choice is optimistic). read_b_heldout = leave-one-condition-out "
            "layer selection (de-biased per selection-symmetric-nulls.md)."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    traits = args.traits
    # Load each trait's r_B + eval cells up front.
    rb_by_trait = {t: S._load_rb(rb_dir, t, args.n_layers, args.hidden) for t in traits}
    cells_by_trait = {t: S.load_eval_cells(pass_a_dir, t) for t in traits}
    train_layers = train_bundle["layers"]

    # Per-(trait, layer) mat + monitors, filled in a LAYER-OUTER loop so the R1
    # map's expensive train-side GCV-SVD (5000x3584, trait-INDEPENDENT — the R1
    # map h is behavior-agnostic, fit on the shared pass-B LMSYS corpus) runs ONCE
    # per layer and predicts all traits' eval contexts in a SINGLE batched
    # ridge_fit_predict call, then splits the predictions back per trait. This is
    # numerically identical to per-trait fits (same train SVD, batched apply) but
    # ~3x cheaper on the dominant cost.
    mat_by = {t: {} for t in traits}
    monitors_by = {t: {} for t in traits}
    for li_pos in range(args.n_layers):
        li = train_layers.index(li_pos)
        Xtr = train_bundle["cx_last"][:, li, :].numpy()  # (N_tr, H) shared
        Ytr = train_bundle["v_x"][:, li, :].numpy()  # (N_tr, H) shared
        # Build each trait's eval matrix at this layer + stack eval contexts.
        eval_blocks = []
        sizes = []
        for t in traits:
            mat = S.build_eval_matrix(cells_by_trait[t], li_pos, rb_by_trait[t])
            mat_by[t][li_pos] = mat
            eval_blocks.append(mat["c_last"])
            sizes.append(mat["c_last"].shape[0])
        X_eval_all = np.concatenate(eval_blocks, axis=0)  # (sum N_ev, H)
        # ONE shared ridge fit (train SVD once) predicting all traits' eval rows.
        h_all = F.ridge_fit_predict(Xtr, Ytr, X_eval_all)  # (sum N_ev, H)
        # Split predictions back per trait, assemble monitors.
        off = 0
        for t, n in zip(traits, sizes, strict=True):
            h_t = h_all[off : off + n]
            off += n
            monitors_by[t][li_pos] = monitors_from_mat(mat_by[t][li_pos], h_t, rb_by_trait[t][li])
        logger.info("built shared-h monitors at layer %2d (all traits)", li_pos)

    for trait in traits:
        logger.info("=== assembling reads for trait %s ===", trait)
        mat_by_layer = mat_by[trait]
        monitors_by_layer = monitors_by[trait]

        # Read A: full per-layer within-condition r curve.
        curve = {m: {mode: [] for mode in MODES} for m in CURVE_METHODS}
        for li in range(args.n_layers):
            for meth in CURVE_METHODS:
                res = S.method_metrics(
                    monitors_by_layer[li][meth],
                    mat_by_layer[li],
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                for mode in MODES:
                    mm = res[mode]
                    curve[meth][mode].append(
                        {
                            "layer": li,
                            "point": mm["point"],
                            "lo": mm["lo"],
                            "hi": mm["hi"],
                            "n_conditions": mm["n_conditions"],
                        }
                    )

        # Read B: held-out (leave-one-condition-out) layer selection.
        read_b = read_b_heldout(monitors_by_layer, mat_by_layer, n_boot=args.n_boot, seed=args.seed)
        read_b_deltas = read_b_delta_vs_pv(read_b)
        rig = rig_gate_own_layer(curve, read_b, trait)

        results["traits"][trait] = {
            "step0_reference_best_layer": step0[trait]["best_layer"],
            "read_a_curve": curve,
            "read_b_heldout": read_b,
            "read_b_deltas_vs_pv": read_b_deltas,
            "rig_gate_own_layer": rig,
        }
        # Checkpoint per trait.
        C.write_json_atomic(args.out_json, results)
        logger.info("  [%s] checkpointed -> %s", trait, args.out_json)

    figs = make_curve_figures(results, args.fig_dir)
    results["figures"] = figs
    C.write_json_atomic(args.out_json, results)
    logger.info("Wrote %d figures + %s", len(figs), args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
