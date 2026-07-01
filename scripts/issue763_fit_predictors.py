#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 6 (0-GPU, VM): GLM (primary) / ridge / PV LOCO + nulls + ceiling.

The analysis. Per behavior B, reads the matched ``v0(C,B)`` shard, the matched
``E0(C,B)`` (rate + per_probe + n_judged), and the faithful ``r_B`` (PV
baseline), then fits three LOCO predictors over the 50 contexts and brackets the
read against the reliability ceiling:

1. **Precision-weighted binomial GLM `v0→E0` LOCO (PRIMARY)** — the in-session
   2026-06-30 finding's correctly-specified estimator (``issue_763_glm``):
   PCA-reduce (nested-CV d) + per-context-weighted binomial GLM, LOCO held-out
   Spearman ρ_GLM. Layer = the held-out-predictivity max over all 28 layers.
2. **Ridge `v0→E0` LOCO (COMPARATOR)** — the optimistic-at-m=8 read. Fit on the
   SAME per-fold PCA reduction the GLM consumes (the shared
   ``analysis.issue_763_pca.nested_cv_pca_reduce`` — #763 BLOCKER
   ridge-pca-comparator: matched capacity so the delta is apples-to-apples),
   then the closed-form PRESS LOCO + nested-CV λ reusing issue658's
   ``_press_loo_mse_per_lambda`` / ``_ridge_dual_weights`` (``_ridge_predict_loco_pca``
   below). ρ_ridge; the ρ_ridge − ρ_GLM OPTIMISM DELTA is reported.
3. **Persona-vector `r_Bᵀ v0` LOCO (BASELINE)** — read-out regime, swept over
   28 layers + selected by held-out predictivity.

Plus: cluster-bootstrap-over-contexts 95% CI (reused
``_cluster_bootstrap_rho``); shuffle-label null (1000 perms) + Hewitt-Liang
control-task null — BOTH refit the FULL select-by-predictivity-over-28-layers
procedure per permutation (the layer-selection-inflation guard, brief concern
#2 — NOT a shortcut to the already-selected layer); the √(r_yy) reliability
ceiling (``issue_763_reliability.compute_bracket`` — the #742 rebuild branch,
that module is not on ``main``); and the per-behavior triage verdict (a/works,
b/fails, c/noise_limited) per plan §3.

Writes ``eval_results/issue_763/matched_predictor_results.json``.

VECTORIZED per ``.claude/rules/vectorize-many-cell-fits.md``: the ridge LOCO is
closed-form PRESS (cheap); the GLM is IRLS over n≤50 cells (NOT many-epoch SGD,
not the GD-fit class) — the layer/behavior axes are looped but each cell is
seconds; the null layer-sweep is the cost, bounded by perms × layers.

``--smoke`` runs the FULL chain on the smoke slice (1 behavior × 3 contexts ×
5 probes) with reduced perms; asserts ρ_GLM/ρ_ridge/ρ_PV are finite + the JSON
schema matches §10.1.

Usage::

    uv run python scripts/issue763_fit_predictors.py
    uv run python scripts/issue763_fit_predictors.py --smoke
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    DEVICE,
    RIDGE_LAMBDAS,
    _cluster_bootstrap_rho,
    _press_loo_mse_per_lambda,
    _rho,
    _ridge_dual_weights,
)
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    EVAL_RESULTS_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    SEED,
    dump_json,
    is_reduced_power,
    load_json,
    reproducibility_metadata,
)

from explore_persona_space.analysis.issue_763_glm import (  # noqa: E402
    glm_predict_loco,
    glm_predict_loco_fixed_dim,
)
from explore_persona_space.analysis.issue_763_pca import (  # noqa: E402
    _pca_fit,
    _pca_transform,
    nested_cv_pca_reduce,
    select_pca_dim,
)
from explore_persona_space.analysis.issue_763_reliability import compute_bracket  # noqa: E402


def _ridge_predict_loco_pca(
    x: np.ndarray, y: np.ndarray, n_judged: np.ndarray, lambdas: list[float]
) -> np.ndarray:
    """LOCO ridge predictions on the SAME PCA-reduced features the GLM consumes.

    #763 BLOCKER ridge-pca-comparator: the registered ``ρ_ridge − ρ_GLM`` optimism
    delta is only apples-to-apples if BOTH arms have matched capacity. Pre-fix the
    ridge arm fit on the raw 3584-d ``x`` while the GLM PCA-reduced to a nested-CV
    ``d`` ≤ 20 — so ridge got the full capacity the GLM did not, corrupting the
    headline. This fits ridge on the IDENTICAL per-fold PCA reduction
    (``analysis.issue_763_pca.nested_cv_pca_reduce``, the same helper + same d
    selection the GLM uses), then runs the closed-form nested-CV-λ LOCO ridge
    (reused PRESS + dual-weights machinery from issue658) on those reduced
    features. ``d`` is selected by the GLM's binomial inner criterion (shared
    feature space); ``λ`` by the ridge PRESS criterion (each arm keeps its own
    regularization on its own terms — the capacity MATCH is the feature space).

    Returns (n_ctx,) held-out scalar predictions.
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    device = torch.device(DEVICE)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        # SHARED reduction: the GLM and ridge consume the SAME train-fold PCA
        # features at the SAME nested-CV-selected d (no held-out leakage — the
        # basis is fit on the train fold only and applied to the held-out row).
        z_tr, z_held, _d = nested_cv_pca_reduce(x[tr], x[i : i + 1], y_train=y[tr], w_train=w[tr])
        Xtr = torch.from_numpy(np.ascontiguousarray(z_tr)).to(device=device, dtype=torch.float64)
        Ytr = torch.from_numpy(np.ascontiguousarray(y[tr].reshape(-1, 1))).to(
            device=device, dtype=torch.float64
        )
        # Standardize the reduced design (the same mu/sd ddof=0 convention the
        # reused _ridge_predict_loco uses), select λ via exact PRESS, predict.
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr, lambdas)
        best_lam = lambdas[int(torch.argmin(mse).item())]
        weights = _ridge_dual_weights(Xtr_n, Ytr, best_lam)  # (d, 1)
        x_held = torch.from_numpy(np.ascontiguousarray(z_held[0])).to(
            device=device, dtype=torch.float64
        )
        x_held_n = (x_held - mu) / sd
        preds[i] = float((x_held_n @ weights).detach().cpu().numpy().reshape(-1)[0])
    return preds


def _ridge_predict_loco_fixed_dim(
    x: np.ndarray, y: np.ndarray, n_judged: np.ndarray, lambdas: list[float], dim: int
) -> np.ndarray:
    """LOCO ridge predictions at a FIXED PCA dim (the null fast path).

    Identical to ``_ridge_predict_loco_pca`` EXCEPT the per-fold PCA dim is the
    passed ``dim`` (fit on the train fold via the shared ``_pca_fit`` /
    ``_pca_transform``) instead of being re-selected by the inner-LOO nested-CV
    per fold. The PCA basis is STILL fit on the train fold only (no held-out
    leakage); only the dim NUMBER is fixed. Used inside the shuffle / control
    nulls (BLOCKER analysis-null-infeasible-at-scale): the regularization
    CAPACITY (the PCA dim) is a hyperparameter, NOT the permuted label, so it is
    chosen ONCE on the observed data per layer and held fixed across permutations
    — this preserves the layer-SELECTION-inflation guard (the null still re-runs
    the full per-layer sweep + re-picks the best layer per perm) while removing
    the ~245x inner-LOO-nested-CV-per-fold cost that made the 1000-perm null
    project to ~580h/behavior/DV. ``nested_cv_pca_reduce`` is untouched; this is
    a sibling fast path over the same public ``_pca_fit``/``_pca_transform``.

    Returns (n_ctx,) held-out scalar predictions.
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    device = torch.device(DEVICE)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu_p, comps = _pca_fit(x[tr], dim)  # train-fold basis at the FIXED dim
        z_tr = _pca_transform(x[tr], mu_p, comps)
        z_held = _pca_transform(x[i : i + 1], mu_p, comps)
        Xtr = torch.from_numpy(np.ascontiguousarray(z_tr)).to(device=device, dtype=torch.float64)
        Ytr = torch.from_numpy(np.ascontiguousarray(y[tr].reshape(-1, 1))).to(
            device=device, dtype=torch.float64
        )
        mu = Xtr.mean(0)
        sd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - mu) / sd
        mse = _press_loo_mse_per_lambda(Xtr_n, Ytr, lambdas)
        best_lam = lambdas[int(torch.argmin(mse).item())]
        weights = _ridge_dual_weights(Xtr_n, Ytr, best_lam)
        x_held = torch.from_numpy(np.ascontiguousarray(z_held[0])).to(
            device=device, dtype=torch.float64
        )
        x_held_n = (x_held - mu) / sd
        preds[i] = float((x_held_n @ weights).detach().cpu().numpy().reshape(-1)[0])
    return preds


logger = logging.getLogger("issue763_fit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

N_SHUFFLE_PERMS = 1000
N_BOOTSTRAP = 2000
# Triage thresholds (plan §3): verdict (a) works, (b) fails, (c) noise-limited.
CEILING_FRACTION_WORKS = 0.5  # rho_GLM >= 0.5 * sqrt(r_yy)
CEILING_DECENT = 0.5  # sqrt(r_yy) >= 0.5 => target reliably measured
CEILING_LOW = 0.35  # sqrt(r_yy) <= 0.35 => noise-limited


def _stage_v0_shards_from_hf(behaviors: list[str]) -> None:
    """Stage v0_shards/v0_<B>.pt from HF if local copies are missing.

    The v0 shards are WRITTEN in phase 1 (``capture``) and uploaded to
    ``<HF_ANALYSIS_TENSORS_PREFIX>/v0_shards/`` by the phase-1 ``--progress-only``
    upload — but they are read here in PHASE 2 (``fit``), which the gate-split
    (off-pod PV judge) can boot on a FRESH VM whose phase-1 local
    ``eval_results/issue_763/v0_shards/`` tree does not exist. ``pv_extract_capture``
    (the phase-2 GPU step) produces ``pv_shards/`` + ``pv_rb_by_behavior.json`` but
    NEVER re-creates the v0 shards, so ``_load_v0`` FileNotFoundError'd on the pod
    (task #763 r3). This mirrors ``issue763_judge_e0._stage_gen_from_hf`` /
    ``issue763_extract_pv_rb._stage_from_hf``: snapshot_download the v0 shards from
    the issue-owned HF prefix into the local path the loader expects, making the
    fit hermetic against phase-1 local disk state. No-op on a matched-host RunPod
    resume where the volume (and every shard) persists. Fail-loud only when a
    shard is NEITHER local NOR on HF (the capture phase never produced it).
    """
    shard_dir = EVAL_RESULTS_DIR / "v0_shards"
    missing = [b for b in behaviors if not (shard_dir / f"v0_{b}.pt").exists()]
    if not missing:
        return
    from huggingface_hub import snapshot_download

    path_in_repo = f"{HF_ANALYSIS_TENSORS_PREFIX}/v0_shards"
    logger.info("[v0_stage] fetching %s from %s/%s", missing, HF_DATA_REPO, path_in_repo)
    snap = snapshot_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}/v0_{b}.pt" for b in missing],
    )
    src_dir = Path(snap) / path_in_repo
    shard_dir.mkdir(parents=True, exist_ok=True)
    for b in missing:
        src = src_dir / f"v0_{b}.pt"
        if not src.exists():
            raise FileNotFoundError(
                f"v0 shard for {b} is neither local ({shard_dir}) nor on HF "
                f"({HF_DATA_REPO}/{path_in_repo}) — the phase-1 capture never "
                "produced/uploaded it (run --phase capture + the --progress-only "
                "upload first)"
            )
        (shard_dir / f"v0_{b}.pt").write_bytes(src.read_bytes())


def _load_v0(behavior: str) -> tuple[np.ndarray, list[str]]:
    """Load the matched v0 shard -> (tensor (n_ctx, n_layers, H), context_ids).

    Stages the shard from HF first when the local copy is missing (gate-split
    phase-2 on a fresh VM — task #763 r3; see ``_stage_v0_shards_from_hf``).
    """
    shard = EVAL_RESULTS_DIR / "v0_shards" / f"v0_{behavior}.pt"
    if not shard.exists():
        _stage_v0_shards_from_hf([behavior])
    blob = torch.load(shard, weights_only=False)
    return blob["tensor"].float().numpy(), blob["context_ids"]


def _e0_vectors(e0: dict, behavior: str, ctx_ids: list[str]):
    """Align E0 graded_mean + rate + n_judged + per_probe with the v0 context order.

    Returns ``(graded (n,), rates (n,), n_judged (n,), per_probe_graded {ctx: [graded
    per probe]}, per_probe_binary {ctx: [e0 per probe]}, kept_ctx_ids)``. The v3
    PRIMARY DV is ``graded_mean`` (0-100); ``rate`` (binary) is the companion.
    A context is KEPT iff it has a non-None ``graded_mean`` (the primary DV); the
    binary rate rides along for the companion read. Contexts with no judged
    probes (graded_mean None) are dropped.
    """
    per_ctx = e0["e0"][behavior]
    graded, rates, njudged, kept = [], [], [], []
    per_probe_graded: dict[str, list[float]] = {}
    per_probe_binary: dict[str, list[float]] = {}
    for c in ctx_ids:
        cell = per_ctx.get(c)
        if cell is None or cell.get("graded_mean") is None:
            continue
        graded.append(float(cell["graded_mean"]))
        # binary rate is the companion; default to NaN if a cell lacks it (rare —
        # format_style always carries both). Kept as float for the companion read.
        rates.append(float(cell["rate"]) if cell.get("rate") is not None else float("nan"))
        # graded precision weight = n_graded if present else n_judged.
        njudged.append(int(cell.get("n_graded") or cell.get("n_judged", 1)))
        kept.append(c)
        per_probe_graded[c] = [
            pr["graded"] for pr in cell.get("per_probe", []) if pr.get("graded") is not None
        ]
        per_probe_binary[c] = [
            pr["e0"] for pr in cell.get("per_probe", []) if pr.get("e0") is not None
        ]
    return (
        np.asarray(graded),
        np.asarray(rates),
        np.asarray(njudged),
        per_probe_graded,
        per_probe_binary,
        kept,
    )


def _layer_sweep_select(
    v0: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    predictor: str,
    *,
    rb: np.ndarray | None = None,
    fixed_dims: list[int] | None = None,
) -> dict:
    """Run the predictor at EVERY layer, return per-layer ρ + the chosen layer.

    The chosen layer = held-out-predictivity max (read-out regime). This whole
    function is what the nulls re-run per permutation (the layer-selection-
    inflation guard, brief concern #2). ``v0`` is (n_ctx, n_layers, H).

    predictor ∈ {"glm", "ridge", "pv"}. For "pv", ``rb`` (n_layers, H) gives the
    per-layer direction; the per-context scalar is ``r_B[layer]ᵀ v0[:, layer]``,
    LOCO is implicit (the projection is fixed; we still hold out per context for
    the Spearman to match the other arms' protocol — projection has no fitted
    parameters so LOCO ρ == in-sample ρ for PV, by construction).

    ``fixed_dims`` (per-layer PCA dim) routes ridge/glm through the FIXED-dim
    fast path (``_ridge_predict_loco_fixed_dim`` / ``glm_predict_loco_fixed_dim``)
    instead of the per-fold nested-CV (BLOCKER analysis-null-infeasible-at-scale):
    the nulls precompute the observed-data per-layer dim ONCE and pass it here so
    the dim selection (a capacity hyperparameter, NOT the permuted label) is not
    re-done ~245x per fold per permutation. ``fixed_dims=None`` keeps the full
    nested-CV path for the OBSERVED-data read (the headline ρ is selected
    honestly; only the null re-runs use the fixed dim).
    """
    _n_ctx, n_layers, _h = v0.shape
    per_layer_rho: list[float | None] = []
    per_layer_pred: list[np.ndarray | None] = []
    for ell in range(n_layers):
        x = v0[:, ell, :]  # (n_ctx, H)
        fd = fixed_dims[ell] if fixed_dims is not None else None
        if predictor == "glm":
            if fd is not None:
                pred = glm_predict_loco_fixed_dim(x, y, n_judged, fd)
            else:
                pred = glm_predict_loco(x, y, n_judged)["pred"]
        elif predictor == "ridge":
            # PCA-reduced ridge on the SAME per-fold reduction the GLM consumes
            # (#763 BLOCKER ridge-pca-comparator) — matched capacity so the
            # ρ_ridge − ρ_GLM optimism delta is apples-to-apples.
            if fd is not None:
                pred = _ridge_predict_loco_fixed_dim(x, y, n_judged, RIDGE_LAMBDAS, fd)
            else:
                pred = _ridge_predict_loco_pca(x, y, n_judged, RIDGE_LAMBDAS)
        elif predictor == "pv":
            assert rb is not None
            direction = rb[ell]  # (H,)
            pred = x @ direction  # (n_ctx,) scalar projection
        else:
            raise ValueError(predictor)
        rho = _rho(pred, y)
        per_layer_rho.append(rho)
        per_layer_pred.append(pred)
    # chosen layer = max ρ (None treated as -inf)
    best_ell, best_rho = 0, -np.inf
    for ell, rho in enumerate(per_layer_rho):
        if rho is not None and rho > best_rho:
            best_rho, best_ell = rho, ell
    return {
        "per_layer_rho": [None if r is None else float(r) for r in per_layer_rho],
        "chosen_layer": best_ell,
        "chosen_rho": None if best_rho == -np.inf else float(best_rho),
        "chosen_pred": per_layer_pred[best_ell],
    }


def _observed_layer_dims(v0: np.ndarray, y: np.ndarray, n_judged: np.ndarray) -> list[int]:
    """Per-layer PCA dim selected ONCE on the OBSERVED data (the null's fixed dim).

    For each layer, runs the SAME nested-CV ``select_pca_dim`` on the full
    observed data (no permutation) and returns the chosen dim per layer. The
    nulls reuse these across all permutations so the dim selection — a capacity
    hyperparameter, NOT the permuted label — is computed once, not per-fold
    per-perm (BLOCKER analysis-null-infeasible-at-scale). This does NOT leak the
    label into the null distribution: every PERMUTATION still re-fits the ridge/
    GLM at this fixed dim on its own shuffled labels (the fit + the layer
    re-selection are permuted), and the dim is a regularization choice fixed by
    the design, exactly as λ ∈ RIDGE_LAMBDAS and the d-grid are design constants.
    """
    _n_ctx, n_layers, _h = v0.shape
    dims: list[int] = []
    for ell in range(n_layers):
        dims.append(select_pca_dim(v0[:, ell, :], y, n_judged))
    return dims


def _shuffle_null(v0, y, n_judged, predictor, *, rb, n_perms, seed) -> dict:
    """Shuffle-label null: permute E0, re-run the FULL layer-sweep+select per perm.

    Returns the right-tail p of the OBSERVED chosen-layer ρ vs the null's
    chosen-layer ρ distribution (brief concern #2: refit the FULL select
    procedure per permutation — NOT permute on the already-selected layer). The
    per-layer PCA dim is FIXED to the observed-data nested-CV selection
    (``_observed_layer_dims``) and reused across permutations — the layer
    SELECTION is still re-run per perm (the inflation guard), only the
    capacity-hyperparameter dim is held fixed (BLOCKER
    analysis-null-infeasible-at-scale; ~245x speedup, no label leakage — see
    ``_observed_layer_dims``). PV has no fitted dim, so ``fixed_dims`` is unused
    there.
    """
    rng = random.Random(seed)
    fixed_dims = None if predictor == "pv" else _observed_layer_dims(v0, y, n_judged)
    obs = _layer_sweep_select(v0, y, n_judged, predictor, rb=rb, fixed_dims=fixed_dims)
    obs_rho = obs["chosen_rho"]
    if obs_rho is None:
        return {"observed_rho": None, "p_value": None, "n_perms": 0, "null_rhos": []}
    null_rhos: list[float] = []
    idx = list(range(len(y)))
    for _ in range(n_perms):
        rng.shuffle(idx)
        y_perm = y[idx]
        nj_perm = n_judged[idx]
        sel = _layer_sweep_select(v0, y_perm, nj_perm, predictor, rb=rb, fixed_dims=fixed_dims)
        if sel["chosen_rho"] is not None:
            null_rhos.append(sel["chosen_rho"])
    if not null_rhos:
        return {"observed_rho": obs_rho, "p_value": None, "n_perms": 0, "null_rhos": []}
    null_arr = np.asarray(null_rhos)
    p = float((np.sum(null_arr >= obs_rho) + 1) / (len(null_arr) + 1))
    null_p95 = float(np.percentile(null_arr, 95))
    return {
        "observed_rho": obs_rho,
        "p_value": p,
        "null_p95": null_p95,
        "n_perms": len(null_rhos),
    }


def _control_task_null(v0, y, n_judged, predictor, *, rb, n_perms, seed) -> dict:
    """Hewitt-Liang control-task null: require the read to BEAT a shuffled-control.

    The control task assigns each context a RANDOM target (drawn from the same
    rate distribution, re-shuffled), refits the full procedure, and the
    SELECTIVITY = ρ_real − ρ_control. The read passes if the observed ρ exceeds
    the control-task ρ distribution's 95th percentile (the d≫n probe-memorization
    guard, mandatory after PCA at n=50). Re-uses the shuffle machinery (a
    shuffled E0 IS a control task here at n=50); the pass criterion differs.
    """
    null = _shuffle_null(v0, y, n_judged, predictor, rb=rb, n_perms=n_perms, seed=seed + 7)
    obs = null.get("observed_rho")
    p95 = null.get("null_p95")
    passed = obs is not None and p95 is not None and obs > p95
    return {"control_task_pass": bool(passed), "observed_rho": obs, "control_p95": p95}


def _triage(rho, rho_ci, sqrt_r_yy, shuffle_p, control_pass) -> str:
    """Per-behavior verdict (a) works / (b) fails / (c) noise_limited (plan §3).

    DV-agnostic: ``rho`` is the HEADLINE predictor's held-out Spearman for the
    DV being triaged. For the v3 PRIMARY (graded) read this is ρ_ridge on the
    graded DV (brief Must-Fix #2 — the registered nulls + verdict target the
    registered headline predictor, RIDGE, on graded_mean). The companion binary
    read is triaged separately on its own GLM ρ.
    """
    if sqrt_r_yy is None or sqrt_r_yy <= CEILING_LOW:
        return "noise_limited"  # (c): ceiling too low / no dynamic range
    lo = rho_ci[0] if rho_ci else None
    if (
        rho is not None
        and lo is not None
        and shuffle_p is not None
        and lo > 0
        and shuffle_p < 0.05
        and rho >= CEILING_FRACTION_WORKS * sqrt_r_yy
        and control_pass
    ):
        return "works"  # (a)
    if sqrt_r_yy >= CEILING_DECENT:
        return "fails"  # (b): target reliably measured but predictor at chance
    return "noise_limited"  # (c)


def _e0_behavior_meta(e0: dict, behavior: str) -> dict:
    """Extract the E0-side v3 §10.1 core fields for one behavior.

    Returns ``{m, reduced_power, r_jj, graded_binary_tracking_spearman}`` — the
    fields the headline predictor artifact MUST co-locate with its verdict so the
    analyzer reads them together (BLOCKER predictor-results-missing-reduced-power-
    and-m / issue763-results-schema-mismatch):

    - ``m`` — the behavior's ACTUAL frozen probe count (``yield_flags[B].m_B``),
      the reliability-power denominator (60/60/60/20/20 by design).
    - ``reduced_power`` — the pre-registered interpretation guard: ``True`` when
      ``m`` is below the ≥50 floor (self_report / persona_drift), so an m=20
      verdict-(c) is NOT read as a ≥50-probe falsification.
    - ``r_jj`` / ``graded_binary_tracking_spearman`` — the graded-DV reliability
      diagnostics, propagated from the E0 ``judge_diagnostics`` so the predictor
      artifact is self-contained (the analyzer reads ONE file for the headline).
    """
    yf = (e0.get("yield_flags") or {}).get(behavior, {})
    diag = (e0.get("judge_diagnostics") or {}).get(behavior, {})
    m = yf.get("m_B")
    return {
        "m": m,
        "reduced_power": is_reduced_power(m),
        "r_jj": diag.get("r_jj"),
        "graded_binary_tracking_spearman": diag.get("graded_binary_tracking_spearman"),
    }


def fit_behavior(behavior, v0, ctx_ids, e0, rb_blob, *, n_perms, n_boot) -> dict:
    """Per-behavior analysis: the v3 GRADED PRIMARY (ridge headline) + binary companion.

    v3 reframe (brief Must-Fix #2 + llm-judging.md): the PRIMARY DV is the GRADED
    mean (0-100), and the registered HEADLINE predictor on it is RIDGE — so the
    shuffle + control-task nulls + the triage verdict TARGET ridge-on-graded
    (``y=graded_mean``, predictor ``"ridge"``). The binary positive RATE rides
    along as the validated human-legible COMPANION, read with the GLM-target
    nulls (the GLM stays the binary companion's estimator; the in-session
    GLM-vs-ridge optimism finding lives on the binary side). The headline
    ``triage_verdict`` is the GRADED-RIDGE verdict; the companion's verdict is
    reported alongside as ``triage_verdict_binary``.
    """
    meta = _e0_behavior_meta(e0, behavior)
    graded, rates, n_judged, per_probe_graded, per_probe_binary, kept = _e0_vectors(
        e0, behavior, ctx_ids
    )
    if len(graded) < 4:
        # Uniform v3 §10.1 core fields even on the degenerate path so every
        # behavior record carries m + reduced_power (the interpretation guard).
        return {
            "behavior": behavior,
            "triage_verdict": "noise_limited",
            "note": f"only {len(graded)} contexts with a non-None graded_mean",
            "n_contexts": len(graded),
            "m": meta["m"],
            "reduced_power": meta["reduced_power"],
            "graded_minus_binary_delta": None,
        }
    # align v0 to the kept contexts
    keep_idx = [ctx_ids.index(c) for c in kept]
    v0_kept = v0[keep_idx]
    rb = rb_blob["r_b"].float().numpy() if rb_blob is not None else None

    # ── PRIMARY: the GRADED DV (y = graded_mean), headline predictor = RIDGE ──
    # Ridge + PV consume the RAW 0-100 graded mean (Spearman is rank-based, so the
    # held-out ρ is scale-invariant). The graded-GLM comparator is a BINOMIAL GLM
    # whose logit-link endog MUST be a [0,1] rate, so it consumes graded/100 — the
    # 0-100 mean rescaled to a fraction (NOT clipped to ~1.0 as a raw 0-100 value
    # would be by _fit_binomial_glm's [1e-6, 1-1e-6] interior clamp). ρ is still
    # rank-invariant, so the comparator stays comparable to the ridge headline.
    graded01 = graded / 100.0
    ridge_g = _layer_sweep_select(v0_kept, graded, n_judged, "ridge")
    glm_g = _layer_sweep_select(v0_kept, graded01, n_judged, "glm")  # graded-GLM comparator
    pv_g = (
        _layer_sweep_select(v0_kept, graded, n_judged, "pv", rb=rb)
        if rb is not None
        else {"chosen_layer": None, "chosen_rho": None, "chosen_pred": None, "per_layer_rho": []}
    )
    rho_ridge_g = ridge_g["chosen_rho"]
    rho_glm_g = glm_g["chosen_rho"]
    rho_pv_g = pv_g["chosen_rho"]

    boot_g = (
        _cluster_bootstrap_rho(ridge_g["chosen_pred"], graded, n_boot=n_boot, seed=SEED)
        if ridge_g["chosen_pred"] is not None and rho_ridge_g is not None
        else None
    )
    rho_ridge_g_ci = boot_g["ci95"] if boot_g else None

    # Nulls + control TARGET the registered headline predictor (RIDGE) on graded.
    shuffle_g = _shuffle_null(
        v0_kept, graded, n_judged, "ridge", rb=None, n_perms=n_perms, seed=SEED
    )
    control_g = _control_task_null(
        v0_kept, graded, n_judged, "ridge", rb=None, n_perms=n_perms, seed=SEED
    )
    ceiling_g = compute_bracket(
        per_probe_graded, list(graded), [int(n) for n in n_judged], n_boot=n_boot, seed=SEED
    )
    sqrt_r_yy_g = ceiling_g["sqrt_r_yy"]
    verdict_g = _triage(
        rho_ridge_g,
        rho_ridge_g_ci,
        sqrt_r_yy_g,
        shuffle_g.get("p_value"),
        control_g["control_task_pass"],
    )

    # ── COMPANION: the BINARY rate (y = rate), GLM-target nulls ──
    # Drop NaN-rate contexts on the binary side (graded may have kept a cell the
    # binary read lacks — rare, but keep the companion read well-defined).
    bin_mask = ~np.isnan(rates)
    rho_glm_b = rho_ridge_b = rho_pv_b = None
    rho_glm_b_ci = None
    shuffle_b = {"p_value": None, "null_p95": None}
    control_b = {"control_task_pass": False}
    ceiling_b = {"sqrt_r_yy": None, "sqrt_r_yy_ci": None, "sqrt_r_yy_binomial": None}
    glm_b = {"chosen_layer": None, "per_layer_rho": []}
    ridge_b = {"chosen_layer": None, "per_layer_rho": []}
    pv_b = {"chosen_layer": None, "per_layer_rho": []}
    verdict_b = "noise_limited"
    if int(bin_mask.sum()) >= 4:
        v0_b = v0_kept[bin_mask]
        rates_b = rates[bin_mask]
        nj_b = n_judged[bin_mask]
        ppb = {kept[i]: per_probe_binary.get(kept[i], []) for i in range(len(kept)) if bin_mask[i]}
        glm_b = _layer_sweep_select(v0_b, rates_b, nj_b, "glm")
        ridge_b = _layer_sweep_select(v0_b, rates_b, nj_b, "ridge")
        pv_b = (
            _layer_sweep_select(v0_b, rates_b, nj_b, "pv", rb=rb)
            if rb is not None
            else {
                "chosen_layer": None,
                "chosen_rho": None,
                "chosen_pred": None,
                "per_layer_rho": [],
            }
        )
        rho_glm_b = glm_b["chosen_rho"]
        rho_ridge_b = ridge_b["chosen_rho"]
        rho_pv_b = pv_b["chosen_rho"]
        boot_b = (
            _cluster_bootstrap_rho(glm_b["chosen_pred"], rates_b, n_boot=n_boot, seed=SEED)
            if glm_b["chosen_pred"] is not None and rho_glm_b is not None
            else None
        )
        rho_glm_b_ci = boot_b["ci95"] if boot_b else None
        shuffle_b = _shuffle_null(v0_b, rates_b, nj_b, "glm", rb=None, n_perms=n_perms, seed=SEED)
        control_b = _control_task_null(
            v0_b, rates_b, nj_b, "glm", rb=None, n_perms=n_perms, seed=SEED
        )
        ceiling_b = compute_bracket(
            ppb, list(rates_b), [int(n) for n in nj_b], n_boot=n_boot, seed=SEED
        )
        verdict_b = _triage(
            rho_glm_b,
            rho_glm_b_ci,
            ceiling_b["sqrt_r_yy"],
            shuffle_b.get("p_value"),
            control_b["control_task_pass"],
        )

    # The ρ_ridge − ρ_GLM optimism delta on the GRADED DV (the in-session finding
    # re-read at m≥50 on the registered primary).
    optimism_delta = (
        (rho_ridge_g - rho_glm_g) if (rho_ridge_g is not None and rho_glm_g is not None) else None
    )
    # The GRADED-minus-BINARY delta the v3 headline turns on (BLOCKER
    # predictor-results-missing-reduced-power-and-m): ρ on the continuous graded
    # DV (headline RIDGE) minus ρ on the dichotomized binary rate (companion GLM).
    # A positive delta is the dichotomization-attenuation removed by the graded DV
    # — the read the v3 primary exists to expose.
    graded_minus_binary_delta = (
        (rho_ridge_g - rho_glm_b) if (rho_ridge_g is not None and rho_glm_b is not None) else None
    )

    return {
        "behavior": behavior,
        "n_contexts": len(graded),
        # ── v3 §10.1 CORE interpretation-guard fields (co-located with the verdict) ──
        "m": meta["m"],  # actual frozen probe count (60/60/60/20/20)
        "reduced_power": meta["reduced_power"],  # True at m<50 (self_report/persona_drift)
        "graded_minus_binary_delta": graded_minus_binary_delta,
        # graded-DV reliability diagnostics propagated from E0 (self-contained artifact)
        "r_jj": meta["r_jj"],
        "graded_binary_tracking_spearman": meta["graded_binary_tracking_spearman"],
        # ── stable v3-schema ALIASES (the reconciler-named canonical names) ──
        # The internal names below (rho_graded_ridge, rho_binary_GLM, sqrt_r_yy)
        # stay for back-compat; these aliases surface the §10.1 canonical spellings
        # so a consumer keyed on the plan's schema resolves without archaeology.
        "rho_graded": rho_ridge_g,  # PRIMARY: ρ on graded DV, headline predictor RIDGE
        "rho_binary": rho_glm_b,  # COMPANION: ρ on binary rate, GLM
        "rho_GLM": rho_glm_g,  # graded-GLM comparator ρ (optimism-delta partner)
        "rho_PV": rho_pv_g,  # persona-vector baseline ρ on the graded DV
        "sqrt_r_yy_graded": sqrt_r_yy_g,
        "sqrt_r_yy_graded_ci": ceiling_g.get("sqrt_r_yy_ci"),
        # ── PRIMARY (graded DV, ridge headline) ──
        "primary_dv": "graded_mean",
        "headline_predictor": "ridge",
        "rho_graded_ridge": rho_ridge_g,
        "rho_graded_ridge_ci": rho_ridge_g_ci,
        "rho_graded_glm": rho_glm_g,
        "rho_graded_PV": rho_pv_g,
        "optimism_delta_graded": optimism_delta,
        "sqrt_r_yy": sqrt_r_yy_g,
        "sqrt_r_yy_ci": ceiling_g.get("sqrt_r_yy_ci"),
        "sqrt_r_yy_binomial": ceiling_g.get("sqrt_r_yy_binomial"),
        "shuffle_null_p": shuffle_g.get("p_value"),
        "shuffle_null_p95": shuffle_g.get("null_p95"),
        "control_task_pass": control_g["control_task_pass"],
        "chosen_layer": ridge_g["chosen_layer"],
        "chosen_layer_graded_glm": glm_g["chosen_layer"],
        "chosen_layer_pv": pv_g["chosen_layer"],
        "per_layer_rho_graded_ridge": ridge_g["per_layer_rho"],
        "per_layer_rho_graded_glm": glm_g["per_layer_rho"],
        "per_layer_rho_graded_PV": pv_g.get("per_layer_rho", []),
        "triage_verdict": verdict_g,  # headline = graded-ridge verdict
        # ── COMPANION (binary rate, GLM headline) ──
        "companion_dv": "rate",
        "rho_binary_GLM": rho_glm_b,
        "rho_binary_GLM_ci": rho_glm_b_ci,
        "rho_binary_ridge": rho_ridge_b,
        "rho_binary_PV": rho_pv_b,
        "sqrt_r_yy_binary": ceiling_b.get("sqrt_r_yy"),
        "shuffle_null_p_binary": shuffle_b.get("p_value"),
        "control_task_pass_binary": control_b["control_task_pass"],
        "chosen_layer_binary_glm": glm_b.get("chosen_layer"),
        "per_layer_rho_binary_GLM": glm_b.get("per_layer_rho", []),
        "triage_verdict_binary": verdict_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #763: GLM/ridge/PV LOCO fits + nulls + ceiling."
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_perms = 50 if args.smoke else N_SHUFFLE_PERMS
    n_boot = 200 if args.smoke else N_BOOTSTRAP

    e0 = load_json(EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")
    pv_path = EVAL_RESULTS_DIR / "pv_rb_by_behavior.json"
    have_pv = pv_path.exists()

    results: dict[str, dict] = {}
    for behavior in args.behaviors:
        v0, ctx_ids = _load_v0(behavior)
        rb_blob = None
        if have_pv:
            rb_shard = EVAL_RESULTS_DIR / "pv_shards" / f"rb_{behavior}.pt"
            if rb_shard.exists():
                rb_blob = torch.load(rb_shard, weights_only=False)
        rec = fit_behavior(behavior, v0, ctx_ids, e0, rb_blob, n_perms=n_perms, n_boot=n_boot)
        results[behavior] = rec
        logger.info(
            "[fit] %s: PRIMARY graded rho_ridge=%s (glm=%s pv=%s) sqrt_r_yy=%s verdict=%s | "
            "COMPANION binary rho_GLM=%s verdict=%s",
            behavior,
            rec.get("rho_graded_ridge"),
            rec.get("rho_graded_glm"),
            rec.get("rho_graded_PV"),
            rec.get("sqrt_r_yy"),
            rec.get("triage_verdict"),
            rec.get("rho_binary_GLM"),
            rec.get("triage_verdict_binary"),
        )
        # smoke schema asserts (plan §10 smoke): the PRIMARY is graded-ridge.
        if args.smoke and rec.get("rho_graded_ridge") is not None:
            assert np.isfinite(rec["rho_graded_ridge"]), "rho_graded_ridge not finite"
            assert rec["triage_verdict"] in ("works", "fails", "noise_limited")
        # v3 §10.1 interpretation-guard fields present on EVERY behavior record
        # (BLOCKER predictor-results-missing-reduced-power-and-m): m + reduced_power
        # + graded_minus_binary_delta must exist on the degenerate path too.
        if args.smoke:
            for key in ("m", "reduced_power", "graded_minus_binary_delta"):
                assert key in rec, f"§10.1 field {key!r} missing for {behavior}"
            assert isinstance(rec["reduced_power"], bool), "reduced_power must be bool"

    out = {
        "by_behavior": results,
        "n_shuffle_perms": n_perms,
        "n_bootstrap": n_boot,
        "have_pv_baseline": have_pv,
        "metadata": reproducibility_metadata({"phase": "fit"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "matched_predictor_results.json")
    print(f"[issue763.fit] wrote predictor results for {len(results)} behaviors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
