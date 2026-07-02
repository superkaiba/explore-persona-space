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

VECTORIZED per ``.claude/rules/vectorize-many-cell-fits.md`` (2026-07-01, the
80×-compute-deviation reversal): the serial fit is OVERHEAD-bound — its hot loops
are the shuffle/control NULLS (1000 perms × 28 layers × 50 folds × a statsmodels
binomial-GLM IRLS OR a per-fold ``torch.linalg.eigh`` PRESS ridge), profiled at
~70–79 s per 1-layer fixed-dim LOCO, projecting the serial null to ~1000 h PER
BEHAVIOR. The fits are batched into ``analysis.issue_763_vectorized``: a batched
IRLS binomial GLM (reproduces statsmodels to ~1e-10) + a batched PRESS ridge that
REUSES the label-independent per-(layer, fold) eigendecomposition across all
perms, + batched observed nested-CV reads (bit-identical / ≤1e-6). The statistical
SEMANTICS are UNCHANGED (same nested-CV d, same PRESS-λ, same n_perms=1000 /
n_bootstrap=2000 / reliability spec; only execution batched); a hard fail-loud
exactness gate (``assert_matches_reference``, run at ``main()`` start) verifies
the batched path against the serial oracles before any behavior is fit. Runs
0-GPU on the ``cpu-mid`` lane (torch threads pinned via ``--num-threads``).

``--smoke`` runs the FULL chain on the smoke slice (1 behavior × 3 contexts ×
5 probes) with reduced perms; asserts ρ_GLM/ρ_ridge/ρ_PV are finite + the JSON
schema matches §10.1.

Usage::

    uv run python scripts/issue763_fit_predictors.py
    uv run python scripts/issue763_fit_predictors.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
    _resolve_device,
    _rho,
    _ridge_dual_weights,
)

# GPU-cutover device override (PM directive 2026-07-02): EPM_FIT_DEVICE routes the
# BATCHED fits (batched_* calls + the exactness gate's batched arm) to e.g. cuda;
# default keeps the imported issue658 CPU pin. The serial reference helpers below
# stay on DEVICE (cpu) — the exactness gate then compares device-batched vs
# CPU-serial, which IS the on-instance cross-device validity check.
FIT_DEVICE = _resolve_device(os.environ.get("EPM_FIT_DEVICE", DEVICE))
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    EVAL_RESULTS_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    SEED,
    dump_json,
    is_reduced_power,
    load_frozen_pool_staged,
    load_json,
    reproducibility_metadata,
)

from explore_persona_space.analysis.issue_763_pca import (  # noqa: E402
    PCA_DIM_GRID,
    _pca_fit,
    _pca_transform,
    nested_cv_pca_reduce,
    select_pca_dim,
)
from explore_persona_space.analysis.issue_763_reliability import compute_bracket  # noqa: E402
from explore_persona_space.analysis.issue_763_vectorized import (  # noqa: E402
    assert_matches_reference,
    batched_binomial_glm_loco_fixed_dim,
    batched_glm_predict_loco,
    batched_ridge_predict_loco_pca,
    batched_ridge_press_loco_fixed_dim,
)


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
    ``issue763_extract_pv_rb._stage_from_hf``: PER-FILE ``hf_hub_download`` (by
    exact path — NOT a pattern-filtered snapshot, which truncates past ~7900
    siblings on the 94k-file data repo, #763 BLOCKER siblings-truncation) fetches
    each missing shard from the issue-owned HF prefix into the local path the
    loader expects, making the fit hermetic against phase-1 local disk state.
    No-op on a matched-host RunPod resume where the volume (and every shard)
    persists. Fail-loud only when a shard is NEITHER local NOR on HF (the capture
    phase never produced it).
    """
    shard_dir = EVAL_RESULTS_DIR / "v0_shards"
    missing = [b for b in behaviors if not (shard_dir / f"v0_{b}.pt").exists()]
    if not missing:
        return
    # PER-FILE hf_hub_download, NOT snapshot_download(allow_patterns=...): the data
    # repo carries >94k files (12x past the ~7900-siblings truncation point), so
    # a pattern-filtered snapshot_download can silently match 0 files and Bug-2
    # would recur as a FileNotFoundError on shards that DO exist on HF (task #763
    # BLOCKER snapshot-download-allow-patterns-siblings-truncation; standing lesson
    # feedback_snapshot_download_siblings_truncation.md / #375/#399). hf_hub_download
    # resolves one file by exact path — no siblings listing, no truncation.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    path_in_repo = f"{HF_ANALYSIS_TENSORS_PREFIX}/v0_shards"
    logger.info("[v0_stage] fetching %s from %s/%s", missing, HF_DATA_REPO, path_in_repo)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for b in missing:
        try:
            src = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{path_in_repo}/v0_{b}.pt",
            )
        except EntryNotFoundError as e:
            raise FileNotFoundError(
                f"v0 shard for {b} is neither local ({shard_dir}) nor on HF "
                f"({HF_DATA_REPO}/{path_in_repo}) — the phase-1 capture never "
                "produced/uploaded it (run --phase capture + the --progress-only "
                "upload first)"
            ) from e
        (shard_dir / f"v0_{b}.pt").write_bytes(Path(src).read_bytes())


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


def _stage_fit_inputs_from_hf(behaviors: list[str], *, stage_e0: bool = True) -> None:
    """Stage the NON-v0 fit inputs (E0 + pv_rb JSONs + pv_shards) from HF.

    ``stage_e0=False`` (reanchor round): a custom ``--e0-json`` is produced
    LOCALLY by the round's judge phase and is NEVER on the parent HF prefix —
    only the pv_rb/pv_shards staging applies; the E0's absence is a sequencing
    error surfaced fail-loud by the caller.

    The ``--from-phase fit`` resume boots on a FRESH cpu-mid VM whose local
    ``eval_results/issue_763/`` tree does not exist (the prior GPU pod that
    produced E0 / pv_rb / pv_shards is stopped). ``_load_v0`` already stages the
    v0 shards; this stages the remaining fit inputs the SAME way — PER-FILE
    ``hf_hub_download`` by exact path (NOT ``snapshot_download(allow_patterns=)``,
    the 94k-file siblings-truncation trap the whole #763 staging family avoids;
    standing lesson feedback_snapshot_download_siblings_truncation.md). Each input
    is fetched from the issue-owned ``<HF_ANALYSIS_TENSORS_PREFIX>/`` prefix into
    the local path the fit reads.

    FAIL-LOUD when an input is NEITHER local NOR on HF: E0 / pv_rb / pv_shards are
    produced by the phase-2 ``judge`` + ``pv_extract_capture`` steps (uploaded by
    ``issue763_upload.py``); if they are absent, the fit resume was dispatched
    before those phases completed + uploaded, and the correct action is to
    re-run them (never to silently fit on a stale / partial local slice). The
    raise names exactly which artifact is missing.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    prefix = HF_ANALYSIS_TENSORS_PREFIX

    def _fetch(local: Path, filename: str, what: str) -> None:
        if local.exists():
            return
        try:
            src = hf_hub_download(repo_id=HF_DATA_REPO, repo_type="dataset", filename=filename)
        except EntryNotFoundError as e:
            raise FileNotFoundError(
                f"{what} is neither local ({local}) nor on HF "
                f"({HF_DATA_REPO}/{filename}) — the phase-2 judge / pv_extract_capture "
                "step never produced/uploaded it. Re-run --from-phase pv_capture "
                "(E0 judge + PV capture + upload) before the fit resume."
            ) from e
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(src).read_bytes())
        logger.info("[fit_stage] staged %s <- %s/%s", what, HF_DATA_REPO, filename)

    # E0 (the graded + binary judged rates) — the fit's y source. Skipped when
    # the caller passed a round-local --e0-json (stage_e0=False).
    if stage_e0:
        _fetch(
            EVAL_RESULTS_DIR / "E0_matched_by_behavior.json",
            f"{prefix}/E0_matched_by_behavior.json",
            "E0_matched_by_behavior.json",
        )
    # PV baseline: the per-behavior r_B summary JSON + per-behavior rb shards.
    _fetch(
        EVAL_RESULTS_DIR / "pv_rb_by_behavior.json",
        f"{prefix}/pv_rb_by_behavior.json",
        "pv_rb_by_behavior.json",
    )
    for b in behaviors:
        _fetch(
            EVAL_RESULTS_DIR / "pv_shards" / f"rb_{b}.pt",
            f"{prefix}/pv_shards/rb_{b}.pt",
            f"pv_shards/rb_{b}.pt",
        )


def _e0_vectors(e0: dict, behavior: str, ctx_ids: list[str]):
    """Align E0 graded_mean + rate + n_judged + per_probe with the v0 context order.

    Returns ``(graded (n,), rates (n,), n_judged (n,), binary_weight (n,),
    per_probe_graded {ctx: [graded per probe]}, per_probe_binary {ctx: [e0 per
    probe]}, kept_ctx_ids)``. The v3 PRIMARY DV is ``graded_mean`` (0-100);
    ``rate`` (binary) is the companion. A context is KEPT iff it has a non-None
    ``graded_mean`` (the primary DV); the binary rate rides along for the
    companion read. Contexts with no judged probes (graded_mean None) are
    dropped.

    ``binary_weight`` (reanchor round): the precision weight the BINARY
    companion fit consumes. A `deception-rubric-reanchor` graded-only E0 stamps
    ``binary_weight_n`` per cell = the PARENT's realized ``n_graded`` (the
    weight the parent fit used, since this shared array prefers ``n_graded``) —
    freezing it makes the §4 ``binary_repro_check`` positive control exact by
    construction. On a parent-shaped E0 (no ``binary_weight_n``) it falls back
    to the SAME ``n_graded or n_judged`` rule as ``n_judged`` — bit-identical
    legacy behavior.
    """
    per_ctx = e0["e0"][behavior]
    graded, rates, njudged, bweight, kept = [], [], [], [], []
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
        bweight.append(
            int(cell.get("binary_weight_n") or cell.get("n_graded") or cell.get("n_judged", 1))
        )
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
        np.asarray(bweight),
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
                # VECTORIZED batched IRLS (1-perm batch here; the null's P-perm
                # batching lives in _shuffle_null). Reproduces statsmodels
                # glm_predict_loco_fixed_dim to ~1e-10 (assert_matches_reference).
                pred = batched_binomial_glm_loco_fixed_dim(
                    x, y[None, :], n_judged[None, :], fd, device=FIT_DEVICE
                )[0]
            else:
                # VECTORIZED observed nested-CV GLM LOCO (batched inner-LOO dim
                # select). Same nested-CV protocol as the serial glm_predict_loco,
                # verified ≤1e-6 by assert_matches_reference.
                pred = batched_glm_predict_loco(x, y, n_judged, PCA_DIM_GRID, device=FIT_DEVICE)
        elif predictor == "ridge":
            # PCA-reduced ridge on the SAME per-fold reduction the GLM consumes
            # (#763 BLOCKER ridge-pca-comparator) — matched capacity so the
            # ρ_ridge − ρ_GLM optimism delta is apples-to-apples.
            if fd is not None:
                pred = batched_ridge_press_loco_fixed_dim(
                    x, y[None, :], RIDGE_LAMBDAS, fd, device=FIT_DEVICE
                )[0]
            else:
                # VECTORIZED observed nested-CV PCA-ridge LOCO (same shared dim
                # selection the GLM uses); bit-identical to _ridge_predict_loco_pca
                # (0.0 delta in assert_matches_reference).
                pred = batched_ridge_predict_loco_pca(
                    x, y, n_judged, RIDGE_LAMBDAS, PCA_DIM_GRID, device=FIT_DEVICE
                )
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

    # Replay the IDENTICAL rng.shuffle(idx) stream to materialize all P perms up
    # front (same seed → same permutations as the serial per-perm loop), then run
    # the fixed-dim fit ONCE PER LAYER with all P perms BATCHED (the vectorization
    # win — the per-(layer, fold) PCA basis / eigendecomposition is
    # label-independent, so it is reused across perms). This is a pure execution
    # refactor: the per-perm chosen-layer-ρ (max over layers) + the right-tail p +
    # the null_p95 are computed exactly as the serial loop did.
    n = len(y)
    idx = list(range(n))
    y_perms = np.empty((n_perms, n), dtype=np.float64)
    nj_perms = np.empty((n_perms, n), dtype=np.float64)
    for pth in range(n_perms):
        rng.shuffle(idx)
        y_perms[pth] = y[idx]
        nj_perms[pth] = n_judged[idx]

    _n_ctx, n_layers, _h = v0.shape
    # per-layer (P, n) held-out predictions for every perm at once.
    per_layer_perm_rho = np.full((n_layers, n_perms), np.nan, dtype=np.float64)
    for ell in range(n_layers):
        x = v0[:, ell, :]
        if predictor == "glm":
            preds = batched_binomial_glm_loco_fixed_dim(
                x, y_perms, nj_perms, fixed_dims[ell], device=FIT_DEVICE
            )
        elif predictor == "ridge":
            preds = batched_ridge_press_loco_fixed_dim(
                x, y_perms, RIDGE_LAMBDAS, fixed_dims[ell], device=FIT_DEVICE
            )
        elif predictor == "pv":
            assert rb is not None
            # The projection is FIXED (only labels permute), so every perm reads
            # the same predictor vector against its own permuted y — identical to
            # the serial PV path (_rho(x @ direction, y_perm) per perm).
            proj = x @ rb[ell]  # (n,)
            preds = np.broadcast_to(proj, (n_perms, n))
        else:
            raise ValueError(predictor)
        for pth in range(n_perms):
            r = _rho(preds[pth], y_perms[pth])
            if r is not None:
                per_layer_perm_rho[ell, pth] = r

    # per-perm chosen ρ = max over layers (NaN treated as -inf; a perm whose every
    # layer is None/NaN is DROPPED, matching the serial "if sel['chosen_rho'] is
    # not None: null_rhos.append(...)").
    null_rhos: list[float] = []
    for pth in range(n_perms):
        col = per_layer_perm_rho[:, pth]
        if np.all(np.isnan(col)):
            continue
        null_rhos.append(float(np.nanmax(col)))
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


def _behavior_meta(e0: dict, behavior: str, *, pool_m: int | None = None) -> dict:
    """Extract the v3 §10.1 core + yield-floor metadata fields for one behavior.

    Returns ``{m, m_e0, reduced_power, yield_floor, any_shortfall,
    n_shortfall_cells, median_n_judged, r_jj, graded_binary_tracking_spearman}``
    — the fields the headline predictor artifact MUST co-locate with its verdict
    (BLOCKER predictor-results-missing-reduced-power-and-m; extended by the
    `deception-rubric-reanchor` metadata-emission fix, plan §3b — the as-run
    artifact recorded m=60 / reduced_power=False for ALL five behaviors because
    the judge's ``_behavior_floor`` silently fell back to 60):

    - ``m`` — the behavior's ACTUAL frozen probe count. GROUND TRUTH is the
      frozen pool (``pool_m``, read FAIL-LOUD by the production ``main()`` via
      ``load_frozen_pool_staged``); the E0 ``yield_flags[B].m_B`` is DATA and is
      cross-checked — on a mismatch a WARN is logged and BOTH are recorded
      (``m`` = pool, ``m_e0`` = the E0 value). ``pool_m=None`` (test seam /
      offline smoke) keeps the legacy E0-derived value.
    - ``reduced_power`` — ``True`` when ``m`` is below the ≥50 floor
      (self_report / persona_drift at m=20), so an m=20 verdict-(c) is NOT read
      as a ≥50-probe falsification.
    - ``yield_floor`` — ``floor(0.8·m)`` (48/48/48/16/16 for the v3 pools).
    - ``any_shortfall`` / ``n_shortfall_cells`` / ``median_n_judged`` —
      recomputed from the E0 ``per_ctx[*].n_judged`` against the CORRECTED
      floor (no re-judging; plan §3b).
    - ``r_jj`` / ``graded_binary_tracking_spearman`` — the graded-DV reliability
      diagnostics, propagated from the E0 ``judge_diagnostics`` so the predictor
      artifact is self-contained (the analyzer reads ONE file for the headline).
    """
    yf = (e0.get("yield_flags") or {}).get(behavior, {})
    diag = (e0.get("judge_diagnostics") or {}).get(behavior, {})
    m_e0 = yf.get("m_B")
    if pool_m is not None:
        m = int(pool_m)
        if m_e0 is not None and int(m_e0) != m:
            logger.warning(
                "[meta] %s: E0 yield_flags m_B=%s != frozen-pool n_probes=%s — the E0 "
                "was judged under the pre-fix silent m_B fallback; recording both "
                "(m = pool ground truth, m_e0 = the E0 value)",
                behavior,
                m_e0,
                m,
            )
    else:
        m = m_e0
    yield_floor = max(1, int(0.8 * m)) if m else None
    per_ctx = (e0.get("e0") or {}).get(behavior, {}) or {}
    n_judged_cells = [int(c["n_judged"]) for c in per_ctx.values() if c.get("n_judged") is not None]
    if yield_floor is not None and n_judged_cells:
        n_shortfall = sum(1 for n in n_judged_cells if n < yield_floor)
        any_shortfall = n_shortfall > 0
        import statistics

        median_n_judged = float(statistics.median(n_judged_cells))
    else:
        n_shortfall, any_shortfall, median_n_judged = None, None, None
    return {
        "m": m,
        "m_e0": m_e0,
        "reduced_power": is_reduced_power(m),
        "yield_floor": yield_floor,
        "any_shortfall": any_shortfall,
        "n_shortfall_cells": n_shortfall,
        "median_n_judged": median_n_judged,
        "r_jj": diag.get("r_jj"),
        "graded_binary_tracking_spearman": diag.get("graded_binary_tracking_spearman"),
    }


def fit_behavior(behavior, v0, ctx_ids, e0, rb_blob, *, n_perms, n_boot, pool_m=None) -> dict:
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

    ``pool_m`` (reanchor metadata fix, plan §3b): the frozen pool's ACTUAL
    ``n_probes``, read FAIL-LOUD by the production ``main()`` and threaded here
    so the record's ``m`` / ``reduced_power`` / yield fields are ground truth;
    ``None`` (test seam) keeps the legacy E0-derived value.
    """
    meta = _behavior_meta(e0, behavior, pool_m=pool_m)
    # Instrument provenance (rule 18): the E0's rubric version + filled-template
    # hash ride into the per-behavior record (`v2` on the reanchor refit).
    rubric_version = e0.get("rubric_version", "v1")
    prompt_hash = (e0.get("graded_prompt_hash") or {}).get(behavior)
    meta_fields = {
        "m": meta["m"],
        "m_e0": meta["m_e0"],
        "reduced_power": meta["reduced_power"],
        "yield_floor": meta["yield_floor"],
        "any_shortfall": meta["any_shortfall"],
        "n_shortfall_cells": meta["n_shortfall_cells"],
        "median_n_judged": meta["median_n_judged"],
        "rubric_version": rubric_version,
        "graded_prompt_hash": prompt_hash,
    }
    graded, rates, n_judged, binary_weight, per_probe_graded, per_probe_binary, kept = _e0_vectors(
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
            **meta_fields,
            "graded_minus_binary_delta": None,
        }
    # align v0 to the kept contexts
    keep_idx = [ctx_ids.index(c) for c in kept]
    v0_kept = v0[keep_idx]
    rb = rb_blob["r_b"].float().numpy() if rb_blob is not None else None

    # Fail-loud PV/v0 geometry guard (concern pv-baseline-staged-is-05b-smoke-not-7b):
    # the PV arm reads `direction = rb[ell]` (ell over v0's n_layers) then `x @ direction`
    # (x is (n_ctx, H)), so rb MUST be (n_layers, H) matching v0. A rb captured on a
    # DIFFERENT model (e.g. a Qwen2.5-0.5B mock PV, [24, 896], vs the production 7B
    # [*, 28, 3584]) would otherwise crash deep inside _layer_sweep_select with a
    # cryptic broadcast/IndexError — or, worse, silently mis-project. Assert the exact
    # (n_layers, H) match up front with an actionable message naming both shapes.
    if rb is not None:
        _, n_layers_v0, h_v0 = v0_kept.shape
        if rb.shape != (n_layers_v0, h_v0):
            raise ValueError(
                f"PV r_B geometry mismatch for {behavior}: rb.shape={tuple(rb.shape)} "
                f"but v0 is (n_layers={n_layers_v0}, H={h_v0}). The PV baseline (r_B) was "
                "captured on a DIFFERENT model than v0 (a [24, 896] r_B is a Qwen2.5-0.5B "
                "MOCK/smoke capture; production is Qwen2.5-7B, 28 layers x 3584). Re-capture "
                "the 7B PV baseline (issue763_extract_pv_rb.py on Qwen/Qwen2.5-7B-Instruct) or "
                "re-point the fit-input staging at the 7B r_B before re-running the fit."
            )

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
        # binary precision weight: `binary_weight_n` when the E0 froze the
        # parent's realized weight (the reanchor graded-only E0), else the same
        # n_graded-or-n_judged rule as before — bit-identical on parent E0s.
        nj_b = binary_weight[bin_mask]
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
        # ── v3 §10.1 CORE interpretation-guard fields (co-located with the verdict),
        # extended by the reanchor metadata fix (plan §3b): m (pool ground truth) +
        # m_e0 + reduced_power + yield_floor + shortfall stats + rubric provenance ──
        **meta_fields,
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


# The ONLY fields --refresh-metadata-only may change/add on a parent checkpoint
# (§10 acceptance criterion 4; everything else is hard-asserted `==` to parent).
_METADATA_PATCH_FIELDS = frozenset(
    {
        "m",
        "m_e0",
        "reduced_power",
        "yield_floor",
        "any_shortfall",
        "n_shortfall_cells",
        "median_n_judged",
        "rubric_version",
    }
)


def _assemble_results(ckpt_dir: Path, fresh: dict[str, dict]) -> dict[str, dict]:
    """Union the round's per-behavior checkpoints with the freshly produced records.

    The reanchor round runs TWO invocations against the same ``--out-dir`` (the
    deception v2 refit, then the 4-behavior metadata refresh — plan §3d, either
    order); each assembles ``matched_predictor_results.json`` from EVERY
    checkpoint present in the round's ``fit_by_behavior/`` so the final write
    carries all 5 records. Fresh records take precedence over disk.
    """
    results: dict[str, dict] = {}
    for f in sorted(ckpt_dir.glob("*.json")):
        if f.stem in BEHAVIORS:
            results[f.stem] = load_json(f)
    results.update(fresh)
    return results


def _binary_control_ref_record(ref: dict, behavior: str) -> dict:
    """Extract the per-behavior parent record from a ``--binary-control-ref`` blob.

    Accepts either a bare ``fit_by_behavior/<b>.json`` record (returned as-is —
    a fit record has no behavior-named keys) or a full
    ``matched_predictor_results.json`` (unwrapped via ``by_behavior`` →
    ``behavior``). Shared by the fresh-fit and checkpoint-resume control sites
    so both validate against the identical reference record.
    """
    rec = ref.get("by_behavior", ref)
    return rec.get(behavior, rec) if isinstance(rec, dict) else rec


def _assert_binary_control(rec: dict, ref_rec: dict, behavior: str, tol: float) -> None:
    """§4 ``binary_repro_check``: the refit's binary companion reproduces the parent.

    The binary data + precision weights are copied verbatim into the v2 E0, so
    the companion fit is deterministic given the seed — a mismatch beyond ``tol``
    is a fit-machinery / E0-plumbing regression (fail loud, never ship past).
    Exact on same-device; tol=1e-3 pre-registered for cross-device.
    """
    for key in ("rho_binary_GLM", "rho_binary_ridge"):
        got, want = rec.get(key), ref_rec.get(key)
        if got is None or want is None:
            raise RuntimeError(
                f"binary control undefined for {behavior}: {key} got={got!r} ref={want!r}"
            )
        delta = abs(float(got) - float(want))
        if delta > tol:
            raise RuntimeError(
                f"binary positive control FAILED for {behavior}: {key} refit={got:.6f} "
                f"parent={want:.6f} |delta|={delta:.2e} > tol={tol:g} — fit-path or "
                "E0-plumbing regression; do NOT interpret the graded refit"
            )
        logger.info(
            "[fit] binary control %s.%s PASS: refit=%.6f parent=%.6f |delta|=%.2e <= %g",
            behavior,
            key,
            got,
            want,
            delta,
            tol,
        )


def _refresh_metadata_only(args, out_dir: Path, e0_path: Path) -> int:
    """Patch ONLY the yield-metadata fields onto the 4 untouched behaviors' checkpoints.

    Plan §3b: load the PARENT checkpoint ``fit_by_behavior/<b>.json``, patch the
    ``_METADATA_PATCH_FIELDS`` (m from the frozen pool FAIL-LOUD, reduced_power,
    yield_floor, shortfall stats recomputed from the parent E0's per-ctx
    n_judged, rubric_version="v1-metadata-refresh"), write to the ROUND's
    ``fit_by_behavior/`` (parent files untouched), and HARD-ASSERT every other
    field compares ``==`` (JSON round-trip) to the parent — the numeric-identity
    gate (§10 criterion 4), byte-comparable by construction because these
    behaviors are never refit.
    """
    e0 = load_json(e0_path)
    parent_dir = args.parent_fit_dir
    ckpt_dir = out_dir / "fit_by_behavior"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fresh: dict[str, dict] = {}
    for behavior in args.behaviors:
        parent_path = parent_dir / f"{behavior}.json"
        parent = load_json(parent_path)
        pool_m = int(load_frozen_pool_staged(behavior)["n_probes"])
        meta = _behavior_meta(e0, behavior, pool_m=pool_m)
        patched = dict(parent)
        patched.update({k: meta[k] for k in _METADATA_PATCH_FIELDS if k != "rubric_version"})
        patched["rubric_version"] = "v1-metadata-refresh"
        # ── NUMERIC-IDENTITY HARD ASSERT (JSON round-trip equality) ──
        parent_rt = json.loads(json.dumps(parent, sort_keys=True))
        patched_rt = json.loads(json.dumps(patched, sort_keys=True))
        drifted = [
            k
            for k in parent_rt
            if k not in _METADATA_PATCH_FIELDS and patched_rt.get(k) != parent_rt[k]
        ]
        assert not drifted, (
            f"NUMERIC-IDENTITY GATE FAILED for {behavior}: non-metadata fields drifted "
            f"from the parent checkpoint: {drifted}"
        )
        new_keys = set(patched_rt) - set(parent_rt)
        assert new_keys <= _METADATA_PATCH_FIELDS, (
            f"refresh added non-metadata keys for {behavior}: "
            f"{sorted(new_keys - _METADATA_PATCH_FIELDS)}"
        )
        n_other = len([k for k in parent_rt if k not in _METADATA_PATCH_FIELDS])
        logger.info(
            "[refresh] %s: NUMERIC-IDENTITY PASS (%d non-metadata fields == parent); "
            "m=%s (m_e0=%s) reduced_power=%s yield_floor=%s any_shortfall=%s "
            "n_shortfall_cells=%s median_n_judged=%s",
            behavior,
            n_other,
            patched["m"],
            patched["m_e0"],
            patched["reduced_power"],
            patched["yield_floor"],
            patched["any_shortfall"],
            patched["n_shortfall_cells"],
            patched["median_n_judged"],
        )
        dump_json(patched, ckpt_dir / f"{behavior}.json")
        fresh[behavior] = patched

    results = _assemble_results(ckpt_dir, fresh)
    out = {
        "by_behavior": results,
        "n_shuffle_perms": N_SHUFFLE_PERMS,
        "n_bootstrap": N_BOOTSTRAP,
        "have_pv_baseline": (EVAL_RESULTS_DIR / "pv_rb_by_behavior.json").exists(),
        "metadata": reproducibility_metadata(
            {"phase": "fit", "mode": "refresh-metadata-only", "parent_fit_dir": str(parent_dir)}
        ),
    }
    dump_json(out, out_dir / "matched_predictor_results.json")
    print(
        f"[issue763.fit] metadata refresh wrote {len(fresh)} patched records "
        f"({len(results)} assembled) -> {out_dir / 'matched_predictor_results.json'}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #763: GLM/ridge/PV LOCO fits + nulls + ceiling."
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--num-threads",
        type=int,
        default=int(os.environ.get("EPM_FIT_NUM_THREADS", "8")),
        help="torch CPU thread count (cpu-mid = 8 vCPU; tiny batched ops thrash "
        "at the default high count — vectorize-many-cell-fits.md item 3)",
    )
    ap.add_argument(
        "--e0-json",
        type=Path,
        default=None,
        help="E0 input override (the reanchor round reads its own E0_deception_v2.json; "
        "default = the parent E0_matched_by_behavior.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output root override: fit_by_behavior/ checkpoints + "
        "matched_predictor_results.json land here (a FRESH round dir => no "
        "stale-checkpoint resume-skip; parent artifacts never clobbered)",
    )
    ap.add_argument(
        "--refresh-metadata-only",
        action="store_true",
        help="patch ONLY the yield-metadata fields onto the parent checkpoints for "
        "--behaviors (no refit, numeric-identity hard assert — plan §3b)",
    )
    ap.add_argument(
        "--parent-fit-dir",
        type=Path,
        default=EVAL_RESULTS_DIR / "fit_by_behavior",
        help="parent per-behavior checkpoints the metadata refresh reads",
    )
    ap.add_argument(
        "--binary-control-ref",
        type=Path,
        default=None,
        help="parent record to assert the §4 binary_repro_check positive control "
        "against (a fit_by_behavior/<b>.json OR a matched_predictor_results.json)",
    )
    ap.add_argument(
        "--binary-control-tol",
        type=float,
        default=1e-3,
        help="binary-control tolerance (exact same-device; 1e-3 cross-device pre-registered)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or EVAL_RESULTS_DIR
    e0_path = args.e0_json or (EVAL_RESULTS_DIR / "E0_matched_by_behavior.json")

    if args.refresh_metadata_only:
        # No fitting at all: skip the exactness gate + input staging (only the
        # parent checkpoints + E0 + frozen pools are read).
        return _refresh_metadata_only(args, out_dir, e0_path)

    # CPU-lane thread discipline: the batched fits are small tensors, so a sane
    # thread count (default = cpu-mid's 8 vCPU) beats the default oversubscription.
    torch.set_num_threads(max(1, args.num_threads))

    # Exactness GATE (fail-loud): the vectorized batched fits MUST reproduce the
    # serial statsmodels-GLM / #658-PRESS-ridge oracles before ANY behavior is fit
    # — a batched-solve / seeding / standardization drift would silently corrupt
    # the headline ρ. Aborts the whole run on any tolerance miss.
    gate = assert_matches_reference(device=FIT_DEVICE)
    logger.info("[fit] vectorized-exactness gate PASS (device=%s): %s", FIT_DEVICE, gate)

    n_perms = 50 if args.smoke else N_SHUFFLE_PERMS
    n_boot = 200 if args.smoke else N_BOOTSTRAP

    # --from-phase fit resume boots on a FRESH cpu-mid VM: stage E0 + pv_rb +
    # pv_shards from HF (v0 is staged lazily by _load_v0). Fail-loud if an input
    # is neither local nor on HF (see _stage_fit_inputs_from_hf). Skipped under
    # --smoke, which runs fully offline on the local 1-behavior slice. A custom
    # --e0-json is produced LOCALLY by the round's judge phase — never staged;
    # its absence is a sequencing error (fail loud below at load_json).
    if not args.smoke:
        _stage_fit_inputs_from_hf(args.behaviors, stage_e0=args.e0_json is None)
    if not e0_path.exists():
        raise FileNotFoundError(
            f"E0 input missing: {e0_path} — run the judge phase that produces it first "
            "(for the reanchor round: issue763_judge_e0.py --rubric-version v2 "
            "--graded-only --base-e0 ... --out-json <this path>)"
        )

    e0 = load_json(e0_path)
    pv_path = EVAL_RESULTS_DIR / "pv_rb_by_behavior.json"
    have_pv = pv_path.exists()

    # Checkpoint-per-phase (code-style.md; PM directive 2026-07-02): each behavior's
    # record persists the moment it completes, and a resume skips any behavior whose
    # per-behavior JSON already exists — the final matched_predictor_results.json is
    # assembled from results{} (checkpoint-loaded + freshly fit). Smoke runs use
    # reduced perms/boot, so they neither read nor write checkpoints (a smoke
    # checkpoint must never satisfy a production resume). BLOCKER
    # binary-control-checkpoint-resume-skip (reanchor review round 1): whenever
    # --binary-control-ref is provided, the §4 binary_repro_check positive control
    # binds on checkpoint-LOADED records too (resume never bypasses it), and a
    # fresh record is checkpointed only AFTER the control passes (a record that
    # fails the registered acceptance control must never land on disk where a
    # crash→rerun resume or _assemble_results would ship it at rc=0).
    ckpt_dir = out_dir / "fit_by_behavior"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    binary_control_ref = (
        load_json(args.binary_control_ref) if args.binary_control_ref is not None else None
    )

    results: dict[str, dict] = {}
    for behavior in args.behaviors:
        ckpt_path = ckpt_dir / f"{behavior}.json"
        if not args.smoke and ckpt_path.exists():
            rec = load_json(ckpt_path)
            # §4 binary_repro_check on RESUME (blocker fix, leg 1): a loaded
            # checkpoint carries rho_binary_GLM/rho_binary_ridge, so it is
            # validated against the parent exactly like a fresh fit — this
            # covers the checkpoint an earlier invocation wrote WITHOUT the
            # flag (the plan-§3d-command sequence) as well as any stale/bad
            # round checkpoint. Fail-loud RuntimeError, never a skip-log.
            if binary_control_ref is not None:
                _assert_binary_control(
                    rec,
                    _binary_control_ref_record(binary_control_ref, behavior),
                    behavior,
                    args.binary_control_tol,
                )
            results[behavior] = rec
            logger.info("[fit] %s: per-behavior checkpoint exists — skipping refit", behavior)
            continue
        v0, ctx_ids = _load_v0(behavior)
        rb_blob = None
        if have_pv:
            rb_shard = EVAL_RESULTS_DIR / "pv_shards" / f"rb_{behavior}.pt"
            if rb_shard.exists():
                rb_blob = torch.load(rb_shard, weights_only=False)
        # m ground truth from the frozen pool, STAGE-THEN-FAIL-LOUD (plan §3b) —
        # never the E0's own m_B, which the pre-fix judge silently defaulted to
        # 60. --smoke stays offline-hermetic (pool_m=None -> E0-derived value).
        pool_m = None if args.smoke else int(load_frozen_pool_staged(behavior)["n_probes"])
        rec = fit_behavior(
            behavior, v0, ctx_ids, e0, rb_blob, n_perms=n_perms, n_boot=n_boot, pool_m=pool_m
        )
        results[behavior] = rec
        # §4 binary_repro_check positive control (reanchor round): the refit's
        # binary companion must reproduce the parent record within tolerance.
        # Runs BEFORE the checkpoint persist (blocker fix, leg 2): a failing
        # record never lands on disk, so a crash→rerun resume cannot ship it.
        if binary_control_ref is not None:
            _assert_binary_control(
                rec,
                _binary_control_ref_record(binary_control_ref, behavior),
                behavior,
                args.binary_control_tol,
            )
        if not args.smoke:
            dump_json(rec, ckpt_path)
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
            for key in (
                "m",
                "reduced_power",
                "graded_minus_binary_delta",
                "yield_floor",
                "any_shortfall",
                "n_shortfall_cells",
                "median_n_judged",
                "rubric_version",
            ):
                assert key in rec, f"§10.1/§3b field {key!r} missing for {behavior}"
            assert isinstance(rec["reduced_power"], bool), "reduced_power must be bool"

    if not args.smoke:
        # Union with the round's existing checkpoints so the two-invocation §3d
        # sequence (deception refit + 4-behavior refresh, either order) always
        # assembles the full record set. Smoke never reads/writes checkpoints.
        results = _assemble_results(ckpt_dir, results)

    out = {
        "by_behavior": results,
        "n_shuffle_perms": n_perms,
        "n_bootstrap": n_boot,
        "have_pv_baseline": have_pv,
        "metadata": reproducibility_metadata({"phase": "fit", "e0_json": str(e0_path)}),
    }
    dump_json(out, out_dir / "matched_predictor_results.json")
    print(f"[issue763.fit] wrote predictor results for {len(results)} behaviors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
