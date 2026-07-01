#!/usr/bin/env python3
"""Issue #779 Stage 1: headline (ridge + MLP h/g, R1/R2 readouts, Gate-1).

Reads the shared collect cache (issue779_collect.py: pass_a cells, pass_b train
bundle, step0 layer selection) + the r_B tensors, and produces the Stage-1
headline: within-condition Pearson (matched to Persona Vectors) with bootstrap
95% CI, per trait x method x elicitation-mode, for

  {pv_raw, direct_ridge (g), direct_mlp (g), r1_ridge, r1_mlp, r2_mean, r2_max,
   r2_topk, r2_last, oracle, probe_ctrl}

plus LOBO (held-out behavior: only h via M^T r_B + pv_raw + oracle compete —
the direct g is structurally inapplicable), the Gate-1 rig validation (reproduced
pv_raw within +-0.10 of PV's published within-condition table), and the
concerns-for-analyzer instrumentation:

  1. H1a linear-nullity: probe_ctrl (trait-fit linear probe on c_x, LOCO) vs
     r1_ridge on the SAME activations — match within CI => linear R1 IS the
     r~_B = M^T r_B nullity.
  2. H1-direct vs generic-whitening: R3(a) reconstruction R2/cosine of the SAME
     h/M used for the R1 readout, vs a shuffled-context null. Readout win WITHOUT
     reconstruction win above null = red flag.
  3. Vectorized-reproduces-serial parity: fit_batched_loco_mlp_multihead vs a
     serial reference on 2-3 cells (vectorize-many-cell-fits.md rule 5).

Design of the two fitters (shape distinction, documented for review):
  - h (R1): the context->profile map is behavior-AGNOSTIC. Fit ONCE on the
    disjoint pass-B LMSYS train corpus (train->eval APPLICATION via
    fit_h.ridge_fit_predict / mlp_fit_predict), applied zero-shot to the eval
    contexts. LOBO is automatic (h never sees any r_B).
  - g / probe_ctrl (direct): behavior-SPECIFIC. Fit LOCO on the eval-context
    c_x -> judge score (leave-one-context-out) via the batched LOCO helpers
    (ridge_predict_loco_raw + fit_batched_loco_mlp_multihead), batched across the
    per-trait cells. Structurally inapplicable on held-out behaviors (needs the
    eval behavior's labels) — the asymmetry IS the result.

Emits eval_results/issue_779/stage1_headline.json + figures/issue_779/stage1_*.png.
--smoke reads the smoke collect cache and runs the identical path at tiny N.
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    assert_matches_reference,
    fit_batched_loco_mlp_multihead,
    ridge_predict_loco_raw,
)
from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import metrics as M  # noqa: E402
from explore_persona_space.experiments.issue_779 import r3_granularity as R3  # noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_stage1")


# ── cache loading ──────────────────────────────────────────────────────────────


def _load_rb(rb_dir: Path, trait: str, n_layers: int, hidden: int) -> np.ndarray:
    blob = torch.load(rb_dir / f"{trait}.pt", weights_only=False)
    r_b = blob["r_b"].to(torch.float32).numpy()
    assert r_b.shape == (n_layers, hidden), (trait, r_b.shape, (n_layers, hidden))
    return r_b


def load_eval_cells(pass_a_dir: Path, trait: str) -> list[dict]:
    """Load a trait's pass-A cells (JSON) + their c_x tensors. Sorted by cond_id."""
    cells = []
    for cp in sorted(pass_a_dir.glob(f"{trait}__*.json")):
        with open(cp) as f:
            cell = json.load(f)
        cx = torch.load(pass_a_dir / f"{cell['trait']}__{cell['cond_id']}_cx.pt", weights_only=True)
        cell["_cx_last"] = cx["cx_last"].to(torch.float32).numpy()  # (n_q, L, H)
        cell["_cx_mean"] = cx["cx_mean"].to(torch.float32).numpy()
        cell["_layers"] = cx["layers"]
        cells.append(cell)
    return cells


def _score_for(cell: dict, qi: int, ri: int) -> float | None:
    """Resolve a rollout's judge score from the cell's {custom_id: score} map."""
    for cid, s in cell["judge_scores"].items():
        parts = cid.split("__")
        if len(parts) < 3:
            continue
        try:
            idx, ci = int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        if idx == qi and ci == ri:
            return s
    return None


def build_eval_matrix(cells: list[dict], layer_idx: int, r_b: np.ndarray) -> dict:
    """Assemble per-rollout arrays at a single read-out layer for a trait.

    Returns arrays aligned by rollout (only rollouts with a valid judge score +
    non-empty response), plus a condition index (cond_id) for within-condition
    grouping, and the elicitation mode per rollout.

    Keys: c_last (N, H), pv_raw (N,), oracle (N,), r2_mean/max/topk/last (N,),
    y (N,), cond (N,) int, mode (N,) str.
    """
    layers = cells[0]["_layers"]
    li = layers.index(layer_idx)
    c_last, pv_raw, oracle = [], [], []
    r2 = {"mean": [], "max": [], "topk": [], "last": []}
    y, cond, mode = [], [], []
    cond_map: dict[str, int] = {}
    for cell in cells:
        cid = cell["cond_id"]
        cond_map.setdefault(cid, len(cond_map))
        for rec in cell["rollouts"]:
            if rec.get("empty"):
                continue
            qi, ri = rec["qi"], rec["ri"]
            s = _score_for(cell, qi, ri)
            if s is None:
                continue
            cl = cell["_cx_last"][qi, li, :]  # (H,)
            c_last.append(cl)
            pv_raw.append(float(np.dot(cl, r_b[li])))
            orc = cell["oracle_proj"].get(str(qi), {}).get(str(ri))
            oracle.append(float(orc[str(layer_idx)]) if orc else np.nan)
            pooled = rec.get("pooled", {})
            for op in ("mean", "max", "topk", "last"):
                vals = pooled.get(op)
                r2[op].append(float(vals[li]) if vals else np.nan)
            y.append(float(s))
            cond.append(cond_map[cid])
            mode.append(cell["mode"])
    return {
        "c_last": np.array(c_last),
        "pv_raw": np.array(pv_raw),
        "oracle": np.array(oracle),
        "r2_mean": np.array(r2["mean"]),
        "r2_max": np.array(r2["max"]),
        "r2_topk": np.array(r2["topk"]),
        "r2_last": np.array(r2["last"]),
        "y": np.array(y),
        "cond": np.array(cond),
        "mode": np.array(mode, dtype=object),
    }


def _group_by_condition(
    x: np.ndarray, y: np.ndarray, cond: np.ndarray, mode: np.ndarray, which_mode: str
) -> tuple[list, list]:
    """Split (x, y) into per-condition arrays for a given elicitation mode."""
    cx, cy = [], []
    sel = np.array([m == which_mode for m in mode])
    if not sel.any():
        return cx, cy
    for c in np.unique(cond[sel]):
        m = sel & (cond == c)
        cx.append(x[m])
        cy.append(y[m])
    return cx, cy


# ── R1 h-fit (train->eval application on pass B) ──────────────────────────────


def fit_h_readouts(
    train_bundle: dict,
    eval_mat: dict,
    layer_idx: int,
    r_b: np.ndarray,
    *,
    n_train_cap: int,
    pca_k: int,
    run_mlp: bool,
) -> dict:
    """Fit h on pass-B train contexts, apply to eval c_x, return R1 readouts.

    Returns {"r1_ridge_dot", "r1_ridge_cos", "r1_mlp_dot", "r1_mlp_cos",
    "recon_ridge", "recon_mlp", "h_ridge_pred_eval", "h_mlp_pred_eval"} — arrays
    aligned to eval rollouts (broadcast per rollout via its question's c_last).
    """
    layers = train_bundle["layers"]
    li = layers.index(layer_idx)
    Xtr = train_bundle["cx_last"][:, li, :].numpy()  # (N_tr, H)
    Ytr = train_bundle["v_x"][:, li, :].numpy()  # (N_tr, H)
    if n_train_cap and Xtr.shape[0] > n_train_cap:
        Xtr, Ytr = Xtr[:n_train_cap], Ytr[:n_train_cap]
    Xev = eval_mat["c_last"]  # (N_ev, H)
    rb_l = r_b[li]  # (H,)

    out = {}
    # Ridge h.
    h_ridge = F.ridge_fit_predict(Xtr, Ytr, Xev)  # (N_ev, H)
    out["r1_ridge_dot"] = F.dot_readout(h_ridge, rb_l)
    out["r1_ridge_cos"] = F.cosine_readout(h_ridge, rb_l)
    out["h_ridge_pred_eval"] = h_ridge
    # Reconstruction diagnostic on TRAIN (in-sample proxy; the eval side has no
    # true v(x) at c_last, so reconstruction quality is read on the train fit).
    h_ridge_tr = F.ridge_fit_predict(Xtr, Ytr, Xtr)
    out["recon_ridge"] = F.reconstruction_metrics(h_ridge_tr, Ytr)

    if run_mlp:
        h_mlp = F.mlp_fit_predict(Xtr, Ytr, Xev, pca_k=pca_k)
        out["r1_mlp_dot"] = F.dot_readout(h_mlp, rb_l)
        out["r1_mlp_cos"] = F.cosine_readout(h_mlp, rb_l)
        out["h_mlp_pred_eval"] = h_mlp
        h_mlp_tr = F.mlp_fit_predict(Xtr, Ytr, Xtr, pca_k=pca_k)
        out["recon_mlp"] = F.reconstruction_metrics(h_mlp_tr, Ytr)
    return out


# ── direct predictor g / probe_ctrl (LOCO on eval contexts) ──────────────────


def fit_direct_loco(eval_mat: dict, *, run_mlp: bool, pca_k: int) -> dict:
    """LOCO direct predictor g / trait-probe on eval-context c_x -> judge score.

    ridge_predict_loco_raw (closed form) gives direct_ridge == probe_ctrl (the
    linear-case direct predictor doubling as the r~_B nullity reference). The MLP
    arm uses fit_batched_loco_mlp_multihead (batched, scalar target head).
    Returns {"direct_ridge": (N,), "direct_mlp": (N,)|None}.
    """
    X = eval_mat["c_last"].astype(np.float64)  # (N, H)
    y = eval_mat["y"].astype(np.float64)[:, None]  # (N, 1)
    out = {}
    # LOCO ridge (closed form; deterministic).
    ridge_pred = ridge_predict_loco_raw(X, y)  # (N, 1)
    out["direct_ridge"] = ridge_pred[:, 0]
    if run_mlp and X.shape[0] >= 4:
        # Batched LOCO MLP, scalar target (p=1 head). One group (this trait).
        grp = MLPGroup(("direct",), X.astype(np.float32), y.astype(np.float32))
        res = fit_batched_loco_mlp_multihead([grp], device="cpu", num_threads=8)
        out["direct_mlp"] = res.preds_by_key[("direct",)][:, 0]
    else:
        out["direct_mlp"] = None
    return out


# ── LOBO (held-out behavior) ──────────────────────────────────────────────────


def lobo_readout(
    train_bundle: dict,
    held_eval_mat: dict,
    layer_idx: int,
    r_b_held: np.ndarray,
    *,
    n_train_cap: int,
    pca_k: int,
    run_mlp: bool,
) -> dict:
    """Held-out-behavior R1: h (behavior-agnostic, trained on generic corpus)
    applied zero-shot with the HELD-OUT trait's r_B. h never saw any r_B, so this
    IS the zero-shot transfer. Returns the R1 readouts vs the held trait's DV;
    the direct g is structurally inapplicable here (no eval-behavior labels used).
    """
    return fit_h_readouts(
        train_bundle,
        held_eval_mat,
        layer_idx,
        r_b_held,
        n_train_cap=n_train_cap,
        pca_k=pca_k,
        run_mlp=run_mlp,
    )


# ── per-method within-condition + CI ──────────────────────────────────────────


def method_metrics(x: np.ndarray, mat: dict, *, n_boot: int, seed: int) -> dict:
    """Within-condition r + bootstrap CI per mode + overall r for one monitor x."""
    res = {}
    for mode in ("system", "many_shot"):
        cx, cy = _group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
        # drop NaN x within each condition
        cx2, cy2 = [], []
        for xi, yi in zip(cx, cy, strict=True):
            m = np.isfinite(xi)
            if m.sum() >= 3:
                cx2.append(xi[m])
                cy2.append(yi[m])
        ci = M.bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=seed)
        res[mode] = ci
    finite = np.isfinite(x) & np.isfinite(mat["y"])
    res["overall_r"] = M.overall_pearson(x[finite], mat["y"][finite])
    return res


# ── Gate 1 rig validation ──────────────────────────────────────────────────────


def gate1_rig_validation(pv_raw_metrics: dict, trait: str) -> dict:
    """Reproduced pv_raw within-condition r must land within +-0.10 of PV's table."""
    targets = C.PV_WITHIN_CONDITION_TARGETS[trait]
    out = {"trait": trait, "pass": True, "checks": {}}
    for mode, tgt_key in (("system", "system"), ("many_shot", "many_shot")):
        got = pv_raw_metrics[mode]["point"]
        target = targets[tgt_key]
        within = bool(np.isfinite(got) and abs(got - target) <= C.RIG_VALIDATION_BAND)
        out["checks"][mode] = {
            "reproduced": got,
            "pv_target": target,
            "abs_diff": abs(got - target) if np.isfinite(got) else None,
            "within_band": within,
        }
        if not within:
            out["pass"] = False
    return out


# ── figures ────────────────────────────────────────────────────────────────────


def make_figures(results: dict, fig_dir: Path) -> list[str]:
    """Per-trait grouped bar of within-condition r (methods x modes) with CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception as e:
        logger.warning("paper_plots style unavailable (%s); using matplotlib default", e)

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    methods = [
        "pv_raw",
        "direct_ridge",
        "direct_mlp",
        "r1_ridge_cos",
        "r1_mlp_cos",
        "r2_mean",
        "r2_max",
        "r2_topk",
        "r2_last",
        "oracle",
        "probe_ctrl",
    ]
    for trait, tr in results["traits"].items():
        for mode in ("system", "many_shot"):
            fig, ax = plt.subplots(figsize=(10, 5))
            xs, heights, errs, labels = [], [], [], []
            for i, meth in enumerate(methods):
                mm = tr["methods"].get(meth)
                if mm is None or mode not in mm:
                    continue
                pt = mm[mode]["point"]
                lo, hi = mm[mode]["lo"], mm[mode]["hi"]
                if not np.isfinite(pt):
                    continue
                xs.append(i)
                heights.append(pt)
                errs.append(
                    [
                        max(0.0, pt - lo) if np.isfinite(lo) else 0.0,
                        max(0.0, hi - pt) if np.isfinite(hi) else 0.0,
                    ]
                )
                labels.append(meth)
            if not xs:
                plt.close(fig)
                continue
            errs_arr = np.array(errs).T if errs else None
            ax.bar(range(len(xs)), heights, yerr=errs_arr, capsize=3)
            tgt = C.PV_WITHIN_CONDITION_TARGETS[trait][mode]
            ax.axhline(tgt, ls="--", color="gray", label=f"PV published {tgt:.3f}")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("within-condition Pearson r")
            ax.set_title(f"{trait} — {mode} (issue 779 Stage 1)")
            ax.legend()
            fig.tight_layout()
            path = fig_dir / f"stage1_{trait}_{mode}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(str(path))
    return saved


# ── per-trait processing ──────────────────────────────────────────────────────


def _process_trait(
    trait: str,
    r_b: np.ndarray,
    cells: list[dict],
    layer_idx: int,
    train_bundle: dict,
    *,
    run_mlp: bool,
    n_train_cap: int,
    pca_k: int,
    n_boot: int,
    seed: int,
    smoke: bool,
) -> tuple[dict, dict | None]:
    """Full Stage-1 read for one trait; returns (trait_result, gate1_or_None)."""
    mat = build_eval_matrix(cells, layer_idx, r_b)
    if len(mat["y"]) < 3:
        logger.warning("[%s] <3 valid rollouts at layer %d; skipping metrics", trait, layer_idx)
        return {"skipped": True, "n_rollouts": len(mat["y"])}, None

    # R1 h readouts (train->eval application on pass B).
    h_out = fit_h_readouts(
        train_bundle,
        mat,
        layer_idx,
        r_b,
        n_train_cap=n_train_cap,
        pca_k=pca_k,
        run_mlp=run_mlp,
    )
    # Direct predictor g / probe_ctrl (LOCO on eval contexts).
    direct = fit_direct_loco(mat, run_mlp=run_mlp, pca_k=pca_k)

    monitors = {
        "pv_raw": mat["pv_raw"],
        "oracle": mat["oracle"],
        "r2_mean": mat["r2_mean"],
        "r2_max": mat["r2_max"],
        "r2_topk": mat["r2_topk"],
        "r2_last": mat["r2_last"],
        "r1_ridge_dot": h_out["r1_ridge_dot"],
        "r1_ridge_cos": h_out["r1_ridge_cos"],
        "direct_ridge": direct["direct_ridge"],
        "probe_ctrl": direct["direct_ridge"],  # ridge direct == linear nullity ref
    }
    if run_mlp:
        monitors["r1_mlp_dot"] = h_out["r1_mlp_dot"]
        monitors["r1_mlp_cos"] = h_out["r1_mlp_cos"]
        if direct["direct_mlp"] is not None:
            monitors["direct_mlp"] = direct["direct_mlp"]

    method_res = {
        name: method_metrics(x, mat, n_boot=n_boot, seed=seed) for name, x in monitors.items()
    }

    # Success criterion paired delta CI: R1/R2 vs pv_raw.
    deltas = {}
    for r1_name in ("r1_ridge_cos", "r1_mlp_cos", "r2_mean"):
        if r1_name not in monitors:
            continue
        deltas[r1_name] = {}
        for mode in ("system", "many_shot"):
            cx_a, cy = _group_by_condition(
                monitors[r1_name], mat["y"], mat["cond"], mat["mode"], mode
            )
            cx_b, _ = _group_by_condition(
                monitors["pv_raw"], mat["y"], mat["cond"], mat["mode"], mode
            )
            if cx_a and cx_b:
                deltas[r1_name][mode] = M.bootstrap_delta_ci(
                    cx_a, cx_b, cy, n_boot=n_boot, seed=seed
                )

    gate1 = gate1_rig_validation(method_res["pv_raw"], trait)

    # Concerns (1): H1a nullity — r1_ridge_cos vs probe_ctrl, same activations.
    h1a = {}
    for mode in ("system", "many_shot"):
        r1r = method_res["r1_ridge_cos"][mode]["point"]
        pc = method_res["probe_ctrl"][mode]["point"]
        h1a[mode] = {
            "r1_ridge_cos_r": r1r,
            "probe_ctrl_r": pc,
            "abs_diff": abs(r1r - pc) if (np.isfinite(r1r) and np.isfinite(pc)) else None,
        }

    # Concerns (2): R3(a) reconstruction vs shuffled-context null on the SAME h.
    layers = train_bundle["layers"]
    li = layers.index(layer_idx)
    Xtr = train_bundle["cx_last"][:, li, :].numpy()
    Ytr = train_bundle["v_x"][:, li, :].numpy()

    def _pred_fn(Xc, _Xtr=Xtr, _Ytr=Ytr):
        return F.ridge_fit_predict(_Xtr, _Ytr, _Xtr)

    recon_null = R3.shuffled_context_null_r2(
        _pred_fn, Xtr, Ytr, n_shuffle=(3 if smoke else 20), seed=seed
    )

    logger.info(
        "[%s] layer=%d pv_raw sys/many=%.3f/%.3f | r1_ridge_cos=%.3f/%.3f | "
        "r2_mean=%.3f/%.3f | oracle=%.3f/%.3f | gate1=%s",
        trait,
        layer_idx,
        method_res["pv_raw"]["system"]["point"],
        method_res["pv_raw"]["many_shot"]["point"],
        method_res["r1_ridge_cos"]["system"]["point"],
        method_res["r1_ridge_cos"]["many_shot"]["point"],
        method_res["r2_mean"]["system"]["point"],
        method_res["r2_mean"]["many_shot"]["point"],
        method_res["oracle"]["system"]["point"],
        method_res["oracle"]["many_shot"]["point"],
        gate1["pass"],
    )
    return {
        "read_out_layer": layer_idx,
        "n_rollouts": len(mat["y"]),
        "methods": method_res,
        "success_deltas": deltas,
        "recon_ridge": h_out.get("recon_ridge"),
        "recon_mlp": h_out.get("recon_mlp"),
        "h1a_nullity": h1a,
        "recon_vs_shuffled_null": recon_null,
    }, gate1


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 Stage 1 headline.")
    parser.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    parser.add_argument("--collect-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_779")
    parser.add_argument("--rb-dir", type=Path, default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "stage1_headline.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    parser.add_argument("--n-layers", type=int, default=C.EXPECTED_LAYERS)
    parser.add_argument("--hidden", type=int, default=C.EXPECTED_HIDDEN)
    parser.add_argument("--n-train-cap", type=int, default=0, help="0 = all pass-B contexts")
    parser.add_argument("--pca-k", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mlp", action="store_true", help="ridge only (fast smoke)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    C.phase("load")
    collect_dir = Path(str(args.collect_dir) + "_smoke") if args.smoke else args.collect_dir
    pass_a_dir = collect_dir / "pass_a"
    rb_dir = args.rb_dir or (collect_dir / "r_b")
    train_bundle = torch.load(
        collect_dir / "pass_b" / "train_context_vectors.pt", weights_only=False
    )
    step0 = {}
    step0_path = collect_dir / "step0" / "step0_oracle.json"
    if step0_path.exists():
        with open(step0_path) as f:
            step0 = json.load(f)

    run_mlp = not args.no_mlp
    traits = args.traits

    # Concerns instrumentation (3): vectorized-reproduces-serial parity (rule 5).
    C.phase("parity_check")
    parity = assert_matches_reference()
    logger.info("vectorized MLP parity vs #658 serial reference: %s", parity)

    results: dict = {
        "traits": {},
        "gate1": {},
        "lobo": {},
        "parity_check": parity,
        "metadata": C.reproducibility_metadata({"script": "issue779_stage1", "smoke": args.smoke}),
    }

    C.phase("fit")
    # Load per-trait r_B + eval cells + read-out layer.
    rb_by_trait, cells_by_trait, layer_by_trait = {}, {}, {}
    for trait in traits:
        rb_by_trait[trait] = _load_rb(rb_dir, trait, args.n_layers, args.hidden)
        cells_by_trait[trait] = load_eval_cells(pass_a_dir, trait)
        # read-out layer: Step-0 selection, else the mid layer.
        bl = step0.get(trait, {}).get("best_layer")
        if bl is None:
            bl = args.n_layers // 2
        layer_by_trait[trait] = int(bl)

    for trait in traits:
        tr_res, gate1 = _process_trait(
            trait,
            rb_by_trait[trait],
            cells_by_trait[trait],
            layer_by_trait[trait],
            train_bundle,
            run_mlp=run_mlp,
            n_train_cap=args.n_train_cap,
            pca_k=args.pca_k,
            n_boot=args.n_boot,
            seed=args.seed,
            smoke=args.smoke,
        )
        results["traits"][trait] = tr_res
        if gate1 is not None:
            results["gate1"][trait] = gate1

    # LOBO: h trained on generic pass-B, applied zero-shot with each held trait's r_B.
    C.phase("lobo")
    for held in traits:
        mat_held = build_eval_matrix(cells_by_trait[held], layer_by_trait[held], rb_by_trait[held])
        if len(mat_held["y"]) < 3:
            results["lobo"][held] = {"skipped": True}
            continue
        h_lobo = lobo_readout(
            train_bundle,
            mat_held,
            layer_by_trait[held],
            rb_by_trait[held],
            n_train_cap=args.n_train_cap,
            pca_k=args.pca_k,
            run_mlp=run_mlp,
        )
        lobo_methods = {
            "pv_raw": mat_held["pv_raw"],
            "oracle": mat_held["oracle"],
            "r1_ridge_cos": h_lobo["r1_ridge_cos"],
        }
        results["lobo"][held] = {
            "read_out_layer": layer_by_trait[held],
            "note": "h is behavior-agnostic (trained on generic LMSYS, no r_B); "
            "applied zero-shot with the held trait's r_B. Direct g is structurally "
            "inapplicable (needs held-behavior labels).",
            "methods": {
                name: method_metrics(x, mat_held, n_boot=args.n_boot, seed=args.seed)
                for name, x in lobo_methods.items()
            },
        }

    # Overall Gate-1 pass (any trait failing => halt Stage 2, but Stage 1 emits).
    gate1_all_pass = all(g.get("pass", False) for g in results["gate1"].values())
    results["gate1_all_pass"] = gate1_all_pass
    if not gate1_all_pass and not args.smoke:
        logger.warning(
            "GATE 1 FAILED for one or more traits — reproduced pv_raw outside +-0.10 "
            "of the PV table. Do NOT proceed to Stage 2; diagnose the extraction/rig."
        )

    C.phase("emit")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, results)
    logger.info("Wrote %s", args.out_json)
    figs = make_figures(results, args.fig_dir)
    logger.info("Wrote %d figures to %s", len(figs), args.fig_dir)
    results["figures"] = figs

    note = (
        f"issue779 Stage 1 {'SMOKE ' if args.smoke else ''}complete: traits={traits}, "
        f"gate1_all_pass={gate1_all_pass}, figures={len(figs)}"
    )
    C.write_sentinel("epm:smoke-result" if args.smoke else "epm:results", note)
    C.phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
