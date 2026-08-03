"""Issue #1310 free-analysis: cross-persona map similarity.

How similar are the four per-character context->dialogue maps of #1310 to EACH
OTHER, per model (Qwen2.5-7B base + instruct), at the scene-AGGREGATED grain
(one point per (persona, scenario); X = turn-0 slot x_spanmean, Y = mean y over
kept slots; 300 points/persona/model — the committed
``eval_results/issue_1310/onpolicy_aggregated`` battery). Three reads:

1. 4x4 cross-persona TRANSFER matrix — persona A's fitted map predicting persona
   B's aggregated points, HELD OUT under a SHARED scenario->fold partition (the
   300 scenarios are shared across personas, so source fold-f-trained map is
   evaluated on target fold-f test points). Diagonal reproduces the committed
   within-cells (equality gate, <=1e-6). Reported fold-test-mean (aggfit
   convention) AND global-pooled-mean. Layers {14,18,19,26}, headline 19.
2. Pairwise OPERATOR similarity at L19 — raw Frobenius cosine (+ random-rotation
   null, ~20 draws) and the two-sided orthogonal-Procrustes-aligned (spectrum)
   cosine, over unordered persona pairs. Conventions ported verbatim from
   scripts/issue1345_operator_comparison.py.
3. Data-paired REPARAMETERIZATION per ordered pair (S->T) at L19 — held-out R^2
   of A_ans o M_source o A_ctx recovering the TARGET's dialogue in the target's
   cell (A_ctx: target-X->source-X, A_ans: source-Y->target-Y; ridge-fit on
   train folds only over scenario-paired rows), vs matched-capacity nulls (5
   draws each of answer-shuffled-fit + random-orthogonal-rotation center) and
   the target's own within ceiling. Structure ported from
   issue1345_operator_comparison.reparam_null_battery.

Scenario-grouped 1000-draw bootstrap CIs (batched, fit931.group_bootstrap_r2)
on each off-diagonal transfer minus the target within-R^2, and each reparam
recovery minus the target ceiling.

Pure-CPU (torch/tokenizer imports deferred via the reused modules). Reuses the
#825/#931/#1310 fit machinery verbatim; new code only for the transfer/cosine/
reparam legs.

CLI:
  uv run python scripts/issue1310_xpersona_similarity.py
      [--store-root <.../store_onpolicy>] [--models base,instruct]
      [--out-dir eval_results/issue_1310/xpersona_similarity]
      [--fig-dir figures/issue_1310]
      [--rot-draws 20] [--reparam-null-draws 5] [--n-boot 1000] [--seed 0]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) bind before torch/numpy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_fit_cells as fit825  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402
import issue1310_aggfit as aggfit  # noqa: E402
import issue1310_common as c1310  # noqa: E402
import issue1310_fit as fit1310  # noqa: E402

SCRIPT = "scripts/issue1310_xpersona_similarity.py"

FROZEN_LAYERS = tuple(c1310.FROZEN_LAYERS)  # (14, 18, 19, 26)
HEADLINE_LAYER = c1310.HEADLINE_LAYER  # 19
PERSONAS = list(c1310.PERSONA_LABELS)  # Wren, HELIOS, Dana, Vex
MODEL_KINDS = list(c1310.MODEL_KINDS)  # base, instruct
DOF_CAP = 0.9  # MANDATORY: uncapped GCV degenerates on this n<p store (aggfit docstring)
N_FOLDS = c1310.N_FOLDS  # 5
FIT_SEED = c1310.FIT_SEED  # 0

# aggfit default store root (repo-canonical); pass the staged path at runtime.
DEFAULT_STORE_ROOT = aggfit.DEFAULT_STORE_ROOT
COMMITTED_DIR = Path("eval_results/issue_1310/onpolicy_aggregated")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--models", type=str, default=",".join(MODEL_KINDS))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("eval_results/issue_1310/xpersona_similarity")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_1310"))
    ap.add_argument("--rot-draws", type=int, default=20)
    ap.add_argument("--reparam-null-draws", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=c1310.N_BOOTSTRAP)  # 1000
    ap.add_argument("--seed", type=int, default=FIT_SEED)
    ap.add_argument(
        "--summary-from-disk",
        action="store_true",
        help="skip all compute; load the per-model JSONs already in --out-dir and "
        "(re)build summary.json + the two figures (lets the per-model legs run in "
        "separate foreground calls under the tool-timeout budget)",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Data: aggregated points, scenario-sorted for row-alignment across personas.
# ---------------------------------------------------------------------------
def load_persona_arrays(store_root: Path, model_kind: str) -> dict:
    """Per persona -> {X:(n,L,D)f32, Y:(n,L,D)f32, scen:(n,), folds:(n,)}.

    Aggregate the store (aggfit.aggregate_store), subset to the 4 personas, and
    SORT each persona's rows by scenario id so all personas are row-aligned by
    scenario (the reparam pairing). The shared scenario->fold partition is built
    once from the common scenario set; a per-persona ``_cv_folds`` cross-check
    asserts the diagonal will reproduce the committed within-cells.
    """
    store = fit1310.load_model_store(store_root, model_kind)
    n_layers = int(store["arrays"]["y"].shape[1])
    assert n_layers == c1310.EXPECTED_LAYERS, (n_layers, c1310.EXPECTED_LAYERS)
    agg = aggfit.aggregate_store(store)
    del store
    gc.collect()
    agg = aggfit.subset_agg(agg, PERSONAS, 0)

    # Shared-partition invariant: every persona carries the identical scenario set.
    scen_sets = {}
    for p in PERSONAS:
        m = agg["personas"] == p
        scen_sets[p] = np.sort(np.unique(agg["scenarios"][m]))
    ref = scen_sets[PERSONAS[0]]
    for p in PERSONAS[1:]:
        assert np.array_equal(scen_sets[p], ref), (
            f"shared-partition invariant broken: persona {p!r} scenario set differs from "
            f"{PERSONAS[0]!r} (cross-persona transfer requires a shared scenario->fold map)"
        )
    fold_ref = fit825._cv_folds(ref, N_FOLDS, FIT_SEED)
    fold_of = {str(s): int(f) for s, f in zip(ref, fold_ref, strict=True)}

    out: dict[str, dict] = {}
    for p in PERSONAS:
        m = agg["personas"] == p
        scen_p = agg["scenarios"][m]
        order = np.argsort(scen_p, kind="stable")
        scen_sorted = scen_p[order]
        folds_p = np.array([fold_of[str(s)] for s in scen_sorted], dtype=np.int64)
        # Guarantee the diagonal reproduces the committed within-cell: the shared
        # map applied to this persona must equal its own _cv_folds partition.
        assert np.array_equal(folds_p, fit825._cv_folds(scen_sorted, N_FOLDS, FIT_SEED)), (
            f"persona {p!r} shared fold map disagrees with its own _cv_folds"
        )
        out[p] = {
            "X": np.ascontiguousarray(agg["X"][m][order], dtype=np.float32),
            "Y": np.ascontiguousarray(agg["Y"][m][order], dtype=np.float32),
            "scen": scen_sorted,
            "folds": folds_p,
        }
    del agg
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Read 1 — cross-persona transfer (held out, shared scenario folds).
# ---------------------------------------------------------------------------
def transfer_cell(src: dict, tgt: dict, layer: int) -> dict:
    """Held-out transfer R^2: source map fit on source train, predicting target
    test points, under the SHARED scenario->fold partition. Returns both pooled
    conventions + the full-coverage held-out predictions over target rows."""
    xsl, ysl = src["X"][:, layer, :], src["Y"][:, layer, :]
    xtl, ytl = tgt["X"][:, layer, :], tgt["Y"][:, layer, :]
    f_s, f_t = src["folds"], tgt["folds"]
    preds = np.zeros_like(ytl, dtype=np.float64)
    covered = np.zeros(xtl.shape[0], dtype=bool)
    ss_res = ss_tot = 0.0
    for k in range(N_FOLDS):
        tr = f_s != k
        te = f_t == k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        cache = fit825._prep_fold(xsl[tr], xtl[te])
        pred = fit825._ridge_predict_cached(cache, ysl[tr])
        preds[te] = pred
        covered[te] = True
        true = ytl[te].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(np.sum((true - pred) ** 2))
        ss_tot += float(np.sum((true - mu) ** 2))
    r2_fold = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    tt = ytl[covered].astype(np.float64)
    pp = preds[covered]
    mug = tt.mean(0)
    ssr = float(np.sum((tt - pp) ** 2))
    sst = float(np.sum((tt - mug) ** 2))
    r2_glob = float("nan") if sst < 1e-12 else 1.0 - ssr / sst
    return {
        "r2_foldmean": r2_fold,
        "r2_globalmean": r2_glob,
        "preds": preds,
        "covered": covered,
        "n_tgt": int(covered.sum()),
    }


def run_transfer(model_kind: str, arrays: dict, args) -> dict:
    """4x4 transfer matrices at every frozen layer + equality gate + L19
    off-diagonal (transfer - within) bootstrap CIs."""
    # Load committed within values (r2_per_layer_obs) for the equality gate.
    import json

    committed_r2: dict[str, list[float]] = {}
    for p in PERSONAS:
        cp = COMMITTED_DIR / f"cells_agg_{model_kind}_{p}.json"
        committed_r2[p] = json.loads(cp.read_text())["r2_per_layer_obs"]

    matrices: dict[str, dict] = {}
    l19_cells: dict[tuple[str, str], dict] = {}
    for layer in FROZEN_LAYERS:
        mat_fold = {}
        mat_glob = {}
        for s in PERSONAS:
            for t in PERSONAS:
                cell = transfer_cell(arrays[s], arrays[t], layer)
                mat_fold[f"{s}->{t}"] = cell["r2_foldmean"]
                mat_glob[f"{s}->{t}"] = cell["r2_globalmean"]
                if layer == HEADLINE_LAYER:
                    l19_cells[(s, t)] = cell
        matrices[str(layer)] = {"foldmean": mat_fold, "globalmean": mat_glob}

    # Equality gate: diagonal (within) must reproduce the committed cells.
    gate = {"tolerance": 1e-6, "per_cell": {}, "worst_abs_delta": 0.0, "passed": True}
    for layer in FROZEN_LAYERS:
        for p in PERSONAS:
            mine = matrices[str(layer)]["foldmean"][f"{p}->{p}"]
            comm = float(committed_r2[p][layer])
            d = abs(mine - comm)
            gate["per_cell"][f"{model_kind}/{p}/L{layer}"] = {
                "mine": mine,
                "committed": comm,
                "abs_delta": d,
            }
            gate["worst_abs_delta"] = max(gate["worst_abs_delta"], d)
    gate["passed"] = gate["worst_abs_delta"] <= gate["tolerance"]

    # L19 off-diagonal (transfer - within) scenario-grouped bootstrap CIs.
    boot: dict[str, dict] = {}
    for t in PERSONAS:
        within = l19_cells[(t, t)]
        yt = arrays[t]["Y"][:, HEADLINE_LAYER, :].astype(np.float64)
        scen_t = arrays[t]["scen"]
        gb_w = fit931.group_bootstrap_r2(
            within["preds"], yt, scen_t, n_boot=args.n_boot, seed=args.seed
        )
        for s in PERSONAS:
            if s == t:
                continue
            tr = l19_cells[(s, t)]
            gb_t = fit931.group_bootstrap_r2(
                tr["preds"],
                yt,
                scen_t,
                n_boot=args.n_boot,
                seed=args.seed,
                draws_matrix=gb_w["draws_matrix"],
            )
            delta = gb_t["draws"] - gb_w["draws"]
            boot[f"{s}->{t}"] = {
                "transfer_r2_foldmean": tr["r2_foldmean"],
                "transfer_r2_globalmean": tr["r2_globalmean"],
                "within_r2_foldmean": within["r2_foldmean"],
                "within_r2_globalmean": within["r2_globalmean"],
                "transfer_r2_boot": gb_t["r2"],
                "within_r2_boot": gb_w["r2"],
                "delta_transfer_minus_within": gb_t["r2"] - gb_w["r2"],
                "delta_ci_lo": float(np.nanquantile(delta, 0.025)),
                "delta_ci_hi": float(np.nanquantile(delta, 0.975)),
                "transfer_frac_of_within": (
                    tr["r2_foldmean"] / within["r2_foldmean"]
                    if within["r2_foldmean"] > 1e-9
                    else float("nan")
                ),
                "n_groups": int(gb_w["n_groups"]),
                "n_boot": int(args.n_boot),
            }
    return {"matrices": matrices, "equality_gate": gate, "l19_offdiag_bootstrap": boot}


# ---------------------------------------------------------------------------
# Read 2 — pairwise operator similarity (ported from issue1345_operator_comparison).
# ---------------------------------------------------------------------------
# Shared random-orthogonal bank (EFFICIENCY DEVIATION from 1345, documented in
# every output + the report): 1345 draws fresh (Q1,Q2) per pair per model; a
# full 3584x3584 fp64 Haar QR costs ~2.1s, so the faithful per-pair-fresh design
# is ~720 QRs / ~25 min — over the analysis-turn foreground budget. We draw ONE
# seeded bank of orthogonal-matrix pairs and REUSE it across persona pairs AND
# models. The statistical target is IDENTICAL (raw-cosine chance band under a
# random two-sided rotation of beta_b); reusing the rotation matrices makes it a
# PAIRED null (if anything cleaner), and the draw COUNTS the brief specifies
# (20 cosine / 5 reparam) are preserved. Only the QR-bearing rotation draws are
# shared; the shuffle nulls stay freshly drawn per pair.
_ROT_BANK: dict[tuple[int, int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}


def rotation_bank(d: int, n: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Seeded, cached list of ``n`` (Q1, Q2) fp64 Haar-orthogonal (d, d) pairs."""
    key = (d, n, seed)
    if key not in _ROT_BANK:
        gen = torch.Generator().manual_seed(seed)
        _ROT_BANK[key] = [
            (ma._random_orthogonal(d, gen), ma._random_orthogonal(d, gen)) for _ in range(n)
        ]
    return _ROT_BANK[key]


def raw_cosine_with_rotation_null(
    beta_a: torch.Tensor, beta_b: torch.Tensor, *, bank: list[tuple[torch.Tensor, torch.Tensor]]
) -> dict:
    """Raw vec-cosine + random two-sided-rotation chance band (no fitting).

    Ported from issue1345_operator_comparison.raw_cosine_with_rotation_null; the
    rotation matrices come from the shared ``bank`` (see the module note above)
    rather than fresh per-pair draws — identical null target, same draw count.
    """
    va = beta_a.reshape(-1)
    raw = float((va @ beta_b.reshape(-1)) / (va.norm() * beta_b.reshape(-1).norm() + 1e-12))
    va_n = va / (va.norm() + 1e-12)
    draws = []
    for q1, q2 in bank:
        vm = (q1.T @ beta_b @ q2).reshape(-1)
        draws.append(float((vm @ va_n) / (vm.norm() + 1e-12)))
    arr = np.asarray(draws)
    return {
        "raw_cosine": raw,
        "rotation_null": {
            "n_draws": int(len(bank)),
            "null_mean": float(arr.mean()) if len(arr) else float("nan"),
            "null_std": float(arr.std()) if len(arr) else float("nan"),
            "null_p975": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
            "analytic_sd_1_over_d": float(1.0 / beta_b.shape[0]),
            "shared_bank": True,
        },
    }


def spectrum_cosine(beta_a: torch.Tensor, beta_b: torch.Tensor) -> float:
    """Closed-form two-sided orthogonal-Procrustes-aligned cosine (von Neumann).

    Verbatim from issue1345_operator_comparison.spectrum_cosine — rotation-
    invariant, so its own rotation null is degenerate (a descriptive ceiling).
    """
    sa = torch.linalg.svdvals(beta_a)
    sb = torch.linalg.svdvals(beta_b)
    return float((sa * sb).sum() / (sa.norm() * sb.norm() + 1e-12))


def run_operator_cosine(model_kind: str, arrays: dict, args) -> dict:
    """Per-persona primal beta at L19 + pairwise raw/aligned cosine vs rotation
    null over unordered pairs."""
    cm.GCV_DOF_CAP = DOF_CAP
    cm.LAMBDA_SELECTION = "gcv"
    betas: dict[str, torch.Tensor] = {}
    lambdas: dict[str, float] = {}
    for p in PERSONAS:
        beta, lam = cm.fit_primal_beta(
            arrays[p]["X"][:, HEADLINE_LAYER, :], arrays[p]["Y"][:, HEADLINE_LAYER, :]
        )
        betas[p] = beta.detach()
        lambdas[p] = float(lam)
    d = betas[PERSONAS[0]].shape[0]
    bank = rotation_bank(d, args.rot_draws, args.seed + 19)
    pairs: dict[str, dict] = {}
    for i in range(len(PERSONAS)):
        for j in range(i + 1, len(PERSONAS)):
            a, b = PERSONAS[i], PERSONAS[j]
            rec = raw_cosine_with_rotation_null(betas[a], betas[b], bank=bank)
            rec["aligned_cosine_procrustes_optimum"] = spectrum_cosine(betas[a], betas[b])
            rec["lambda_a"] = lambdas[a]
            rec["lambda_b"] = lambdas[b]
            for k in (10, 50):
                cs = cm.principal_angles(betas[a], betas[b], k)
                rec[f"principal_angle_cos_k{k}"] = {
                    "mean_cos": float(np.mean(cs)),
                    "min_cos": float(np.min(cs)),
                }
            pairs[f"{a}~{b}"] = rec
    return {"headline_layer": HEADLINE_LAYER, "lambda_per_persona": lambdas, "pairs": pairs}


# ---------------------------------------------------------------------------
# Read 3 — data-paired reparameterization (structure from reparam_null_battery).
# ---------------------------------------------------------------------------
def _t(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def reparam_chain(
    xs: torch.Tensor,
    ys: torch.Tensor,
    xt: torch.Tensor,
    yt: torch.Tensor,
    folds: np.ndarray,
    *,
    center: str = "observed",
    perm: np.ndarray | None = None,
    qc: torch.Tensor | None = None,
    qa: torch.Tensor | None = None,
) -> tuple[float, np.ndarray]:
    """Held-out reparam recovery of TARGET dialogue via A_ans o M_source o A_ctx.

    Alignments + center are ridge-fit on TRAIN folds only (per-fold refit, same
    for observed and null — consistent with issue1345 leg_b_battery). Returns
    (fold-test-mean pooled R^2, full-coverage held-out preds over target rows).
    ``center`` in {observed, shuffle, rotation} matches reparam_null_battery's
    matched-capacity nulls (same ridge core / lambda grid / A fits).
    """
    n, d = xt.shape
    preds = torch.zeros((n, d), dtype=torch.float64)
    ss_res = ss_tot = 0.0
    ys_shuf = ys[torch.as_tensor(perm)] if (center == "shuffle" and perm is not None) else None
    for k in range(N_FOLDS):
        te = folds == k
        tr = folds != k
        if te.sum() == 0 or tr.sum() < 3:
            continue
        trt = torch.as_tensor(tr)
        tet = torch.as_tensor(te)
        # A_ctx: align target context -> source context space (fit on train).
        prep_ctx = ma._ridge_prep(xt[trt])
        xshat = ma._ridge_predict(prep_ctx, xs[trt], xt[tet])
        # M_source: source's own context->dialogue map, applied to xshat.
        prep_m = ma._ridge_prep(xs[trt])
        if center == "observed":
            yshat = ma._ridge_predict(prep_m, ys[trt], xshat)
        elif center == "shuffle":
            yshat = ma._ridge_predict(prep_m, ys_shuf[trt], xshat)
        elif center == "rotation":
            mu_xs = xs[trt].mean(0)
            mu_ys = ys[trt].mean(0)
            x_rot = (xshat - mu_xs) @ qc + mu_xs
            yhat = ma._ridge_predict(prep_m, ys[trt], x_rot)
            yshat = (yhat - mu_ys) @ qa + mu_ys
        else:
            raise ValueError(f"unknown center {center!r}")
        # A_ans: align source dialogue -> target dialogue (fit on train).
        prep_ans = ma._ridge_prep(ys[trt])
        pred = ma._ridge_predict(prep_ans, yt[trt], yshat)
        preds[tet] = pred
        true = yt[tet]
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - true.mean(0)) ** 2).sum())
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return r2, preds.cpu().numpy()


def run_reparam(model_kind: str, arrays: dict, within_l19: dict, args) -> dict:
    """Ordered off-diagonal reparam recovery vs matched-capacity nulls + target
    ceiling, with a (recovery - ceiling) scenario-grouped bootstrap CI."""
    ma.GCV_DOF_CAP = DOF_CAP
    ma.LAMBDA_SELECTION = "gcv"
    out: dict[str, dict] = {}
    for s in PERSONAS:
        for t in PERSONAS:
            if s == t:
                continue
            xs = _t(arrays[s]["X"][:, HEADLINE_LAYER, :])
            ys = _t(arrays[s]["Y"][:, HEADLINE_LAYER, :])
            xt = _t(arrays[t]["X"][:, HEADLINE_LAYER, :])
            yt = _t(arrays[t]["Y"][:, HEADLINE_LAYER, :])
            folds = arrays[t]["folds"]  # target folds; rows aligned by scenario
            n, d = xt.shape
            r2_obs, preds_obs = reparam_chain(xs, ys, xt, yt, folds, center="observed")
            # Matched-capacity nulls (fresh shuffle perms per pair; shared
            # rotation bank across pairs/models — the module efficiency note).
            rng = np.random.default_rng(args.seed + 1)
            shuf = []
            for _ in range(args.reparam_null_draws):
                perm = rng.permutation(n)  # scenario-level == row-level (1 row/scenario)
                r2s, _ = reparam_chain(xs, ys, xt, yt, folds, center="shuffle", perm=perm)
                shuf.append(r2s)
            rot = []
            for qc, qa in rotation_bank(d, args.reparam_null_draws, args.seed + 13):
                r2r, _ = reparam_chain(xs, ys, xt, yt, folds, center="rotation", qc=qc, qa=qa)
                rot.append(r2r)
            null_recovery = float(max(np.nanmean(shuf), np.nanmean(rot)))
            ceiling = within_l19[t]["r2_foldmean"]
            # (recovery - ceiling) bootstrap over scenarios (paired draws).
            scen_t = arrays[t]["scen"]
            yt_np = arrays[t]["Y"][:, HEADLINE_LAYER, :].astype(np.float64)
            gb_ceil = fit931.group_bootstrap_r2(
                within_l19[t]["preds"], yt_np, scen_t, n_boot=args.n_boot, seed=args.seed
            )
            gb_rep = fit931.group_bootstrap_r2(
                preds_obs,
                yt_np,
                scen_t,
                n_boot=args.n_boot,
                seed=args.seed,
                draws_matrix=gb_ceil["draws_matrix"],
            )
            delta = gb_rep["draws"] - gb_ceil["draws"]
            out[f"{s}->{t}"] = {
                "recovery_r2_foldmean": r2_obs,
                "recovery_r2_boot": gb_rep["r2"],
                "target_ceiling_foldmean": ceiling,
                "target_ceiling_boot": gb_ceil["r2"],
                "null_recovery_r2": null_recovery,
                "null_shuffle_mean": float(np.nanmean(shuf)),
                "null_rotation_mean": float(np.nanmean(rot)),
                "null_shuffle_draws": [float(v) for v in shuf],
                "null_rotation_draws": [float(v) for v in rot],
                "recovery_minus_ceiling": gb_rep["r2"] - gb_ceil["r2"],
                "recovery_minus_ceiling_ci_lo": float(np.nanquantile(delta, 0.025)),
                "recovery_minus_ceiling_ci_hi": float(np.nanquantile(delta, 0.975)),
                "recovery_frac_of_ceiling": (r2_obs / ceiling if ceiling > 1e-9 else float("nan")),
                "n_reparam_null_draws": int(args.reparam_null_draws),
                "n_boot": int(args.n_boot),
            }
    return {"headline_layer": HEADLINE_LAYER, "ordered_pairs": out}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(results: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("neurips")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: transfer matrices (L19), one heatmap panel per model. ----
    fig, axes = plt.subplots(1, len(MODEL_KINDS), figsize=(9.5, 4.4), layout="constrained")
    if len(MODEL_KINDS) == 1:
        axes = [axes]
    npers = len(PERSONAS)
    all_vals = []
    for m in MODEL_KINDS:
        mat = results[m]["transfer"]["matrices"][str(HEADLINE_LAYER)]["foldmean"]
        all_vals.extend(mat[f"{s}->{t}"] for s in PERSONAS for t in PERSONAS)
    vmin = min(all_vals)
    vmax = max(all_vals)
    im = None
    for ax, m in zip(axes, MODEL_KINDS, strict=True):
        mat = results[m]["transfer"]["matrices"][str(HEADLINE_LAYER)]["foldmean"]
        grid = np.array([[mat[f"{s}->{t}"] for t in PERSONAS] for s in PERSONAS])
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(npers), PERSONAS, rotation=45, ha="right")
        ax.set_yticks(range(npers), PERSONAS)
        ax.set_xlabel("target persona (points predicted)")
        ax.set_ylabel("source persona (fitted map)")
        ax.set_title(f"{m}  (L{HEADLINE_LAYER}, diagonal = within)")
        for i in range(npers):
            for j in range(npers):
                v = grid[i, j]
                txt_col = "white" if v < (vmin + vmax) / 2 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color=txt_col,
                    fontsize=8,
                    fontweight=weight,
                )
    fig.colorbar(im, ax=axes, shrink=0.8, label="held-out R² (fold-test-mean)")
    fig.suptitle("Cross-persona map transfer: source map -> target points (held out)")
    pp.savefig_paper(fig, "xpersona_transfer_matrix", dir=str(fig_dir), formats=("png",))
    plt.close(fig)

    # ---- Figure 2: operator cosine (top) + reparam recovery (bottom). ----
    fig, axes = plt.subplots(2, len(MODEL_KINDS), figsize=(11.0, 8.0), layout="constrained")
    c_primary = pp.paper_palette_role("primary")
    c_baseline = pp.paper_palette_role("baseline")
    c_control = pp.paper_palette_role("control")
    c_accent = pp.paper_palette_role("accent")

    # Row 0: pairwise operator cosine (aligned + raw) vs rotation-null band.
    for col, m in enumerate(MODEL_KINDS):
        ax = axes[0, col]
        pairs = results[m]["operator_cosine"]["pairs"]
        labels = list(pairs.keys())
        x = np.arange(len(labels))
        aligned = [pairs[k]["aligned_cosine_procrustes_optimum"] for k in labels]
        raw = [pairs[k]["raw_cosine"] for k in labels]
        null_p975 = float(np.mean([pairs[k]["rotation_null"]["null_p975"] for k in labels]))
        ax.bar(x - 0.2, aligned, width=0.4, color=c_primary, label="aligned (Procrustes optimum)")
        ax.bar(x + 0.2, raw, width=0.4, color=c_baseline, label="raw Frobenius cosine")
        ax.axhline(
            null_p975,
            color=c_control,
            ls="--",
            lw=1.2,
            label=f"raw-cosine rotation null p97.5 ({null_p975:.1e})",
        )
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("operator cosine")
        ax.set_title(f"{m}: pairwise operator similarity (L{HEADLINE_LAYER})")
        ax.set_ylim(0, 1.02)
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Row 1: ordered-pair reparam recovery vs target ceiling vs null level.
    for col, m in enumerate(MODEL_KINDS):
        ax = axes[1, col]
        rep = results[m]["reparam"]["ordered_pairs"]
        labels = list(rep.keys())
        x = np.arange(len(labels))
        recov = [rep[k]["recovery_r2_foldmean"] for k in labels]
        ceil = [rep[k]["target_ceiling_foldmean"] for k in labels]
        nullv = [rep[k]["null_recovery_r2"] for k in labels]
        ax.bar(x, recov, width=0.6, color=c_primary, label="reparam recovery R²")
        ax.plot(x, ceil, "D", color=c_accent, ms=6, label="target within ceiling")
        ax.plot(x, nullv, "_", color=c_control, ms=14, mew=2.5, label="matched-capacity null")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xticks(x, [k.replace("->", "→") for k in labels], rotation=90, fontsize=7)
        ax.set_ylabel("held-out R² (fold-test-mean)")
        ax.set_title(f"{m}: reparameterization (source→target, L{HEADLINE_LAYER})")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Operator similarity and data-paired reparameterization across personas")
    pp.savefig_paper(fig, "xpersona_cosine_reparam", dir=str(fig_dir), formats=("png",))
    plt.close(fig)


# ---------------------------------------------------------------------------
def _write_summary_and_figures(results: dict, models: list[str], gate_ok: bool, args) -> None:
    """Headline-number summary.json + the two figures (shared by the full-run
    path and the --summary-from-disk assembly path)."""
    summary = {
        "metadata": c1310.metadata(SCRIPT, args.seed, 0),
        "models": models,
        "personas": PERSONAS,
        "headline_layer": HEADLINE_LAYER,
        "gcv_dof_cap": DOF_CAP,
        "equality_gate_all_pass": gate_ok,
        "per_model": {},
    }
    for m in models:
        if "operator_cosine" not in results[m]:
            summary["per_model"][m] = {"equality_gate": results[m]["transfer"]["equality_gate"]}
            continue
        mat = results[m]["transfer"]["matrices"][str(HEADLINE_LAYER)]["foldmean"]
        within = [mat[f"{p}->{p}"] for p in PERSONAS]
        offdiag = [mat[f"{s}->{t}"] for s in PERSONAS for t in PERSONAS if s != t]
        pairs = results[m]["operator_cosine"]["pairs"]
        aligned = [pairs[k]["aligned_cosine_procrustes_optimum"] for k in pairs]
        raw = [pairs[k]["raw_cosine"] for k in pairs]
        null_p975 = [pairs[k]["rotation_null"]["null_p975"] for k in pairs]
        rep = results[m]["reparam"]["ordered_pairs"]
        recov = [rep[k]["recovery_r2_foldmean"] for k in rep]
        ceil = [rep[k]["target_ceiling_foldmean"] for k in rep]
        nullr = [rep[k]["null_recovery_r2"] for k in rep]
        fracs = [rep[k]["recovery_frac_of_ceiling"] for k in rep]
        summary["per_model"][m] = {
            "equality_gate_worst_abs_delta": results[m]["transfer"]["equality_gate"][
                "worst_abs_delta"
            ],
            "within_r2_l19_mean": float(np.mean(within)),
            "within_r2_l19_by_persona": {p: mat[f"{p}->{p}"] for p in PERSONAS},
            "offdiag_transfer_r2_l19_mean": float(np.mean(offdiag)),
            "offdiag_transfer_frac_of_within_mean": float(np.mean(offdiag) / np.mean(within)),
            "operator_aligned_cosine_l19_mean": float(np.mean(aligned)),
            "operator_raw_cosine_l19_mean": float(np.mean(raw)),
            "operator_rotation_null_p975_mean": float(np.mean(null_p975)),
            "reparam_recovery_l19_mean": float(np.mean(recov)),
            "reparam_ceiling_l19_mean": float(np.mean(ceil)),
            "reparam_null_l19_mean": float(np.mean(nullr)),
            "reparam_recovery_frac_of_ceiling_mean": float(np.nanmean(fracs)),
        }
    c1310.write_json(args.out_dir / "summary.json", summary)
    if gate_ok:
        make_figures(results, args.fig_dir)
        print(f"[xpersona] figures -> {args.fig_dir}")
    else:
        print("[xpersona] equality gate FAILED — figures skipped")


def load_results_from_disk(out_dir: Path, models: list[str]) -> dict:
    """Reconstruct the ``results`` dict (for summary + figures) from the
    per-model JSONs already written by the compute legs."""
    import json

    results: dict[str, dict] = {}
    for m in models:
        tr = json.loads((out_dir / f"transfer_matrix_{m}.json").read_text())
        entry = {
            "transfer": {
                "matrices": tr["matrices"],
                "equality_gate": tr["equality_gate"],
                "l19_offdiag_bootstrap": tr["l19_offdiag_bootstrap"],
            }
        }
        op_path = out_dir / f"operator_cosine_{m}.json"
        rp_path = out_dir / f"reparam_{m}.json"
        if op_path.exists() and rp_path.exists():
            entry["operator_cosine"] = json.loads(op_path.read_text())
            entry["reparam"] = json.loads(rp_path.read_text())
        results[m] = entry
    return results


def main() -> int:
    args = parse_args()
    torch.set_num_threads(8)
    fit825.GCV_DOF_CAP = DOF_CAP
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        assert m in MODEL_KINDS, f"unknown model {m!r}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.summary_from_disk:
        # Assemble summary + figures from the per-model JSONs (no recompute).
        models = MODEL_KINDS  # summary always spans both models
        results = load_results_from_disk(args.out_dir, models)
        gate_ok = all(results[m]["transfer"]["equality_gate"]["passed"] for m in models)
        _write_summary_and_figures(results, models, gate_ok, args)
        print(f"[xpersona] summary+figures from disk; gate_all_pass={gate_ok}")
        return 0 if gate_ok else 1
    print(f"[phase=xpersona] models={models} dof_cap={DOF_CAP} store_root={args.store_root}")

    results: dict[str, dict] = {}
    gate_ok = True
    for m in models:
        print(f"[xpersona] loading aggregated arrays for {m}")
        arrays = load_persona_arrays(args.store_root, m)
        print(f"[xpersona] {m}: transfer matrix ({len(PERSONAS)}x{len(PERSONAS)}, 4 layers)")
        transfer = run_transfer(m, arrays, args)
        gate = transfer["equality_gate"]
        print(
            f"[xpersona] {m}: equality gate worst |d|={gate['worst_abs_delta']:.2e} "
            f"{'PASS' if gate['passed'] else 'FAIL'}"
        )
        gate_ok = gate_ok and gate["passed"]
        c1310.write_json(
            args.out_dir / f"transfer_matrix_{m}.json",
            {
                "metadata": c1310.metadata(SCRIPT, args.seed, 0),
                "model_kind": m,
                "personas": PERSONAS,
                "frozen_layers": list(FROZEN_LAYERS),
                "headline_layer": HEADLINE_LAYER,
                "gcv_dof_cap": DOF_CAP,
                "convention_note": (
                    "matrix[source->target] = held-out R^2 of source's fitted map on "
                    "target's aggregated points under the shared scenario->fold "
                    "partition; diagonal = within (equality-gated vs committed)."
                ),
                "matrices": transfer["matrices"],
                "equality_gate": transfer["equality_gate"],
                "l19_offdiag_bootstrap": transfer["l19_offdiag_bootstrap"],
            },
        )
        if not gate["passed"]:
            print(f"[xpersona] {m}: EQUALITY GATE FAILED — stopping before further reads")
            results[m] = {"transfer": transfer}
            continue

        print(f"[xpersona] {m}: operator cosine (L{HEADLINE_LAYER}, {args.rot_draws} rot draws)")
        opcos = run_operator_cosine(m, arrays, args)
        c1310.write_json(
            args.out_dir / f"operator_cosine_{m}.json",
            {
                "metadata": c1310.metadata(SCRIPT, args.seed, 0),
                "model_kind": m,
                "gcv_dof_cap": DOF_CAP,
                "rotation_null_note": (
                    "EFFICIENCY DEVIATION from issue1345_operator_comparison: the "
                    f"{args.rot_draws}-draw raw-cosine rotation null uses ONE shared "
                    "orthogonal-matrix bank across pairs+models (a paired null; identical "
                    "target) instead of fresh per-pair draws (~6x fewer 3584x3584 fp64 QRs)."
                ),
                **opcos,
            },
        )

        # within-L19 cells for the reparam ceiling + bootstrap
        within_cells = {}
        for p in PERSONAS:
            within_cells[p] = transfer_cell(arrays[p], arrays[p], HEADLINE_LAYER)
        print(
            f"[xpersona] {m}: reparam ({len(PERSONAS) * (len(PERSONAS) - 1)} ordered pairs, "
            f"{args.reparam_null_draws} null draws each)"
        )
        reparam = run_reparam(m, arrays, within_cells, args)
        c1310.write_json(
            args.out_dir / f"reparam_{m}.json",
            {
                "metadata": c1310.metadata(SCRIPT, args.seed, 0),
                "model_kind": m,
                "gcv_dof_cap": DOF_CAP,
                "rotation_null_note": (
                    "EFFICIENCY DEVIATION: the rotation matched-capacity null uses ONE "
                    "shared orthogonal-matrix bank across pairs+models (identical target); "
                    "shuffle nulls stay freshly drawn per pair. Draw counts unchanged."
                ),
                **reparam,
            },
        )
        results[m] = {"transfer": transfer, "operator_cosine": opcos, "reparam": reparam}
        del arrays
        gc.collect()

    _write_summary_and_figures(results, models, gate_ok, args)
    print(f"[xpersona] done in {time.time() - t0:.1f}s; gate_all_pass={gate_ok}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
