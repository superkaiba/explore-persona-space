#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, r̂, r_B, →, ρ, M⁺, ※, ×) in scientific docstrings + log messages.
"""Issue #722 — fit M0 vs M⁺ (ridge + MLP) and the four reads (plan §4.4 / §4.5).

For each ``(behavior, layer ∈ {7,14,21})`` this fits THREE maps through the
#658 LOCO harness:

- **M0** from ``(c0 → v0)`` — the pre-FT context→answer function (base);
- **M⁺** from ``(cplus → vplus)`` — the post-FT function (adapter-applied
  activations, the post-FT INPUT drives the post-FT OUTPUT);
- **M_pseudo** from ``(cplus → M0(cplus))`` — the SAME M0 function on shifted
  inputs (the ``floor_shifted`` same-function shifted-design null).

Each map is fit twice: a closed-form PRESS-LOCO ridge (``_ridge_predict_loco``)
and a GPU-batched LOCO MLP ensemble (``_fit_mlp_ensemble_loco``), output target
= the top-64 v0 PCs (``A35_MLP_TARGET_DIM``), so the ridge-vs-MLP gap is
like-for-like.

**Headline DV (plan §3 / §4.5.1):** ``Δ_med = median_c |Δ(c)·r̂_B|`` over the
base ``common_c_grid`` (both maps evaluated at the SAME base c0), with a
family-clustered CI from the NEW ``clustered_bootstrap_scalar`` (NOT the
Spearman helper — distinct stat). Gated on the COMBINED floor
``max(floor_M0_refit, floor_Mplus_refit, floor_shifted)``, each built through the
IDENTICAL bootstrap+random-init refit harness (``make_refit_pair``).

**Co-primary (chain-ρ):** held-out Spearman of ``r_Bᵀ M̂(c)`` vs E (= ``g`` from
#537's ``G_meta.json``), under M0 vs M⁺, family-clustered via the EXISTING
``clustered_bootstrap_spearman`` (correct two-array use).

Plus cross-transfer (read 3), the linear-vs-nonlinear gap (read 4), and the
per-cell support-distance diagnostic ``‖cplus − c0‖``.

The ridge + M_pseudo closed-form path is CPU; the MLP fits are the GPU phase
(CLAUDE.md compute-character carve-out — a gradient-descent fit is GPU-worthy).
Per-(behavior, layer) checkpoints to ``eval_results/issue_722/cells/`` make the
run resumable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# DOTENV_LINT_EXEMPT: legacy pre-#745 script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(str(PROJECT_ROOT / ".env"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402
from issue722_bootstrap import clustered_bootstrap_scalar, floor_sd, make_refit_pair  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    clustered_bootstrap_spearman,
)

logger = logging.getLogger("issue722.fit")

HIDDEN = 3584
N_LAYERS = 28  # Qwen-2.5-7B(-Instruct) hidden-layer count; r_B stacks are (28, 3584).
DATA_REPO = "superkaiba1/explore-persona-space-data"
SWEEP_LAYERS = (7, 14, 21)
HEADLINE_BEHAVIORS = ("em", "sycophancy", "fact")  # marker + refusal dropped (plan §4.3/§5)
# #537 G behavior key map (behavior dir name -> G_meta.json behavior prefix).
G_BEHAVIOR_KEY = {"em": "em", "sycophancy": "sycophancy", "fact": "fact", "marker": "marker"}
# r_b.pt column key map (behavior -> r_b.pt key); fact uses the NEW r_b_fact.pt.
RB_COLUMN_KEY = {"em": "broad_em", "sycophancy": "sycophancy", "refusal": "harmful_compliance"}
SUPPORT_SHIFT_PCTL = 90  # large-shift flag threshold (plan §3)
N_REFIT_PAIRS = 100
N_SCALAR_BOOT = 1000
# Output target dim (top-v0 PCs) shared by ridge + MLP — the #658 A35_MLP_TARGET_DIM
# (64) for the production run; a smoke clamps it via --target-dim to bound the
# CPU MLP-ensemble cost (the ensemble size is target_dim × n_folds). Module-global
# so the refit-floor closures read the same value.
TARGET_DIM = 64


def _to64(Y: np.ndarray, pca_basis: np.ndarray) -> np.ndarray:
    """Project a (n, 3584) target onto the shared top-64 v0 PCs (n, 64)."""
    return Y @ pca_basis.T


def _pca_basis_v0(V0: np.ndarray, dim: int) -> np.ndarray:
    """Top-`dim` PCA basis of the base v0 stack (dim, 3584), mean-centered.

    Shared between ridge + MLP so the nonlinearity gap is like-for-like
    (the #658 A35_MLP_TARGET_DIM reduction applied to the v0 output target).

    NumPy's default ``np.linalg.svd`` uses LAPACK ``gesdd`` (divide-and-conquer),
    which is fast but occasionally raises ``LinAlgError: SVD did not converge`` on
    near-singular inputs — the family-clustered bootstrap RESAMPLES (``Yb`` in the
    floor refit, ``_refit_ridge_fn``) draw whole families with replacement, so a
    draw with heavy row duplication / vanishing variance is mean-centered to a
    rank-deficient ``Vc`` that ``gesdd`` can fail on. On that exception fall back
    to ``scipy.linalg.svd(..., lapack_driver='gesvd')`` (the slower QR-based
    driver, far more robust to non-convergence). The fallback computes the SAME
    SVD of the SAME matrix — no ridge perturbation, no resample skipping — so the
    common (clean) path is bit-identical to the run that produced the em JSONs and
    only the rare degenerate resample takes the robust driver. (Issue #722 round 3:
    crashed at sycophancy L7 on a bootstrap resample; em L7/L14/L21 fit cleanly.)
    """
    Vc = V0 - V0.mean(axis=0, keepdims=True)
    # economy SVD; rows of Vt are the principal directions.
    try:
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _scipy_svd

        logger.warning(
            "[phase=fit_M] np.linalg.svd (gesdd) did not converge on a %s input "
            "(near-singular bootstrap resample); retrying with scipy gesvd",
            Vc.shape,
        )
        _, _, Vt = _scipy_svd(Vc, full_matrices=False, lapack_driver="gesvd")
    k = min(dim, Vt.shape[0])
    return Vt[:k]  # (k, 3584)


def _ridge_fit_predict(X: np.ndarray, Y64: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Ridge map fit on (X→Y64), evaluated at `grid` → (n_grid, 64).

    Uses #658's closed-form dual-ridge weights at the PRESS-selected λ (fit on
    ALL rows, not LOCO — the function-change read evaluates the fitted map on a
    fixed grid; LOCO is for the held-out ρ reads, not for M(c) at a new input).
    """
    lambdas = fit658.RIDGE_LAMBDAS
    device = torch.device(fit658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64)).to(device=device, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    mse = fit658._press_loo_mse_per_lambda(Xn, Yt, lambdas)
    best_lam = lambdas[int(torch.argmin(mse).item())]
    w = fit658._ridge_dual_weights(Xn, Yt, best_lam)  # (d, 64)
    Gt = torch.from_numpy(np.ascontiguousarray(grid)).to(device=device, dtype=torch.float64)
    Gn = (Gt - mu) / sd
    return (Gn @ w).detach().cpu().numpy()


def _mlp_loco_pred(X: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """LOCO MLP held-out predictions for all 64 output dims → (n, 64)."""
    return fit658._fit_mlp_ensemble_loco(
        X.astype(np.float32), Y64.astype(np.float32), target_idx=list(range(Y64.shape[1]))
    )


def _ridge_loco_pred(X: np.ndarray, Y64: np.ndarray) -> np.ndarray:
    """LOCO ridge held-out predictions for all 64 output dims → (n, 64)."""
    return fit658._ridge_predict_loco(X, Y64, fit658.RIDGE_LAMBDAS)


def _chain_rho_one(pred64: np.ndarray, pca_basis: np.ndarray, r_hat: np.ndarray, E: np.ndarray):
    """Spearman(r_Bᵀ M̂(c), E) — project the 64-dim pred back to 3584, dot r̂_B."""
    pred_full = pred64 @ pca_basis  # (n, 3584)
    chain = pred_full @ r_hat  # (n,)
    return fit658._rho(chain, E), chain


def _clustered_paired_rho_diff_ci(
    chain_m0: np.ndarray,
    chain_mplus: np.ndarray,
    E: np.ndarray,
    families: list[str],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Family-clustered CI on the PAIRED ρ-shift ``Spearman(chain_Mplus, E) −
    Spearman(chain_M0, E)`` (MF#4 / plan §6.5 / §3 concordance).

    Two separate marginal CIs do NOT test the paired difference. This resamples
    whole ``target_cid`` families with replacement ONCE per draw, then computes
    BOTH ρs on the SAME resampled rows in one pass and takes their difference —
    so the floor respects the paired structure (the two ρs share the held-out
    cells). Returns ``{"point", "ci_lo", "ci_hi", "n_families"}``; a degenerate
    (<2 families) input returns a point-only CI.
    """
    chain_m0 = np.asarray(chain_m0, dtype=np.float64)
    chain_mplus = np.asarray(chain_mplus, dtype=np.float64)
    Earr = np.asarray(E, dtype=np.float64)
    fams = np.asarray(families, dtype=object)
    assert chain_m0.shape == chain_mplus.shape == Earr.shape == fams.shape, (
        chain_m0.shape,
        chain_mplus.shape,
        Earr.shape,
        fams.shape,
    )
    rho_mp, rho_m0 = fit658._rho(chain_mplus, Earr), fit658._rho(chain_m0, Earr)
    if rho_mp is None or rho_m0 is None:
        return {"point": None, "ci_lo": None, "ci_hi": None, "n_families": 0}
    point = float(rho_mp - rho_m0)
    uniq = sorted({str(f) for f in fams})
    if len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        a, b = fit658._rho(chain_mplus[idx], Earr[idx]), fit658._rho(chain_m0[idx], Earr[idx])
        if a is not None and b is not None:  # skip degenerate (zero-variance) resamples
            diffs.append(a - b)
    if not diffs:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    boot = np.asarray(diffs, dtype=np.float64)
    return {
        "point": point,
        "ci_lo": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "n_families": len(uniq),
    }


def _r_hat_for(behavior: str, layer: int, rb_main: dict, rb_fact: dict | None) -> np.ndarray:
    """Unit r_B at this (behavior, layer), from r_b.pt (em/syc) or r_b_fact.pt (fact)."""
    if behavior == "fact":
        if rb_fact is None:
            raise RuntimeError("fact headline requested but r_b_fact.pt not loaded")
        stack = np.asarray(rb_fact["r_b_fact"]["fact_expression"]["diffmeans"], dtype=np.float64)
        src = "r_b_fact.pt[fact_expression][diffmeans]"
    else:
        col = RB_COLUMN_KEY[behavior]
        stack = np.asarray(rb_main["r_b"][col]["diffmeans"], dtype=np.float64)
        src = f"r_b.pt[{col}][diffmeans]"
    # MF#8: fail loud on a stale/tiny/smoke-extracted r_B stack — fit_M ALWAYS
    # consumes the FULL production direction (28 layers × 3584 hidden) against the
    # real (n, 3584) activation store, so a (24, 896)-shaped CPU-smoke fact
    # artifact would otherwise surface only as an opaque matmul shape error in
    # `delta_full @ r_hat`. Assert the production shape with a clear message.
    if stack.shape != (N_LAYERS, HIDDEN):
        raise RuntimeError(
            f"r_B shape mismatch for {behavior}: expected ({N_LAYERS}, {HIDDEN}), "
            f"got {stack.shape} from {src} — likely a stale or smoke-extracted "
            "artifact (re-run the fact r_B extraction on the full 7B model)"
        )
    assert stack.shape[0] >= layer + 1, f"r_B stack {stack.shape} has no layer {layer}"
    r = stack[layer]
    norm = np.linalg.norm(r)
    if norm < 1e-9:
        raise RuntimeError(f"degenerate r_B for {behavior} L{layer} (norm {norm:.2e})")
    return r / norm


def _rb_revision() -> str:
    """HF revision the r_B .pt files are fetched at (#811 maxp-round critic advisory).

    ``EPM_RB_REVISION=<sha>`` pins both ``r_b.pt`` and ``r_b_fact.pt`` to a fixed
    data-repo commit; the ``"main"`` default preserves prior behavior verbatim.
    """
    return os.environ.get("EPM_RB_REVISION", "main")


def _load_rb_main() -> dict:
    """Load #658 r_b.pt from HF (em/syc/refusal directions)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        DATA_REPO,
        "issue658_theory_assumptions/store/r_b.pt",
        repo_type="dataset",
        revision=_rb_revision(),
    )
    return torch.load(local, weights_only=False)


def _load_rb_fact(*, required: bool = False) -> dict | None:
    """Load the NEW r_b_fact.pt (this task's fact direction); None if absent.

    ``required=False`` (the #722 default) preserves prior behavior verbatim: any
    load failure warns + returns None and the caller drops fact. ``required=True``
    (the #811 call sites, where fact carries the round's headline) RE-RAISES a
    load failure — a transient Hub error at fit time must crash the fit loudly,
    never silently void the fact headline (#811 r10 CONCERN
    rb-fact-silent-drop-headline). A payload flagged ``degenerate`` still returns
    None under BOTH modes: that is a data-declared drop (plan §8), not a load
    failure.
    """
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            DATA_REPO,
            "issue722_rb_extension/store/r_b_fact.pt",
            repo_type="dataset",
            revision=_rb_revision(),
        )
    except Exception as e:  # not yet extracted (e.g. fit-only smoke) — caller drops fact
        if required:
            raise RuntimeError(
                "r_b_fact.pt REQUIRED (fact is in the requested behaviors) but failed to "
                f"load from {DATA_REPO}@{_rb_revision()}: {e!r} — refusing the silent "
                "fact-drop (rb-fact-silent-drop-headline)"
            ) from e
        logger.warning("r_b_fact.pt unavailable (%s); fact headline will be skipped", e)
        return None
    payload = torch.load(local, weights_only=False)
    if payload.get("degenerate"):
        logger.warning("r_b_fact.pt flagged degenerate — fact dropped from headline (plan §8)")
        return None
    return payload


def _load_E(behavior: str, cell_keys: list[str]) -> np.ndarray:
    """E = #537 G_meta.json `g` per cell, aligned to cell_keys (NaN where absent)."""
    meta_path = PROJECT_ROOT / "eval_results/issue_537/G_tensor/G_meta.json"
    pc = json.loads(meta_path.read_text())["per_cell"]
    out = np.full(len(cell_keys), np.nan, dtype=np.float64)
    for i, k in enumerate(cell_keys):
        if k in pc and pc[k].get("g") is not None:
            out[i] = float(pc[k]["g"])
    return out


# A skipped refit pair (SVD non-convergence on a degenerate resample) is acceptable
# bootstrap noise; losing >5% of pairs across the three floors means the resample
# geometry is pathological and the floor is suspect — surfaced as a CONCERN.
REFIT_SKIP_CONCERN_FRAC = 0.05

# Keys a complete per-cell JSON checkpoint MUST carry for the resume-skip to trust
# it (the headline + floor + read fields the analyzer consumes). A cached cell
# missing any of these is treated as a partial/corrupt write and RE-FIT, so a run
# killed mid-`out.write_text` does not poison the resume. (#722 round 3 resume
# contract.) ``refit_skip`` is intentionally NOT required here so the 3 em cells
# recovered from the pre-guard round-2 attempt (which predate the field) are still
# accepted as complete on resume.
_CELL_SCHEMA_KEYS = frozenset(
    {
        "Delta_med",
        "floor_M0_refit",
        "floor_Mplus_refit",
        "floor_shifted",
        "floor_combined",
        "Delta_over_floor_sd",
        "chain_rho",
        "cross_transfer",
        "support_distance",
    }
)


def _cached_cell_valid(path: Path) -> bool:
    """True iff `path` is a complete per-cell checkpoint (parses + has the schema keys).

    Guards the resume-skip against a partial/corrupt write (a run killed mid
    ``out.write_text`` leaves a truncated JSON). A cell that fails to parse OR is
    missing any ``_CELL_SCHEMA_KEYS`` is re-fit rather than trusted.
    """
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[phase=fit_M] cached %s unreadable (%s) — will re-fit", path.name, e)
        return False
    missing = _CELL_SCHEMA_KEYS - obj.keys()
    if missing:
        logger.warning(
            "[phase=fit_M] cached %s missing schema keys %s — will re-fit",
            path.name,
            sorted(missing),
        )
        return False
    return True


def _aggregate_refit_skips(behavior: str, layer: int, *counters: dict) -> dict:
    """Aggregate the per-floor make_refit_pair skip counters into one cell-level block.

    Each ``counter`` is the ``skip_counter`` dict make_refit_pair filled
    (``{"n_attempted", "n_skipped"}``). Returns
    ``{"n_attempted", "n_skipped", "skip_frac", "concern"}`` where ``concern`` is
    True when the combined skip fraction exceeds ``REFIT_SKIP_CONCERN_FRAC`` — the
    caller (``main``) reads it to log a loud WARNING the orchestrator persists as a
    ``task.py raise-concern`` (pod-side code never shells task.py).
    """
    n_attempted = sum(int(c.get("n_attempted", 0)) for c in counters)
    n_skipped = sum(int(c.get("n_skipped", 0)) for c in counters)
    skip_frac = (n_skipped / n_attempted) if n_attempted else 0.0
    concern = skip_frac > REFIT_SKIP_CONCERN_FRAC
    if n_skipped:
        logger.warning(
            "[phase=fit_M] %s L%d: %d/%d refit pairs skipped (LinAlgError, %.1f%%)%s",
            behavior,
            layer,
            n_skipped,
            n_attempted,
            100 * skip_frac,
            " — EXCEEDS 5%% CONCERN threshold" if concern else "",
        )
    return {
        "n_attempted": n_attempted,
        "n_skipped": n_skipped,
        "skip_frac": skip_frac,
        "concern": concern,
    }


def stage_cells_from_attempt(attempt_id: str, out_dir: Path) -> int:
    """Download a prior crash-attempt's per-cell JSONs from HF into `out_dir`.

    The GCP EXIT-trap persists the partial ``eval_results/issue_722/`` of a crashed
    run to ``issue722_partial/<attempt_id>/eval_results_issue_722/cells/*.json`` on
    the HF data repo. On a re-launch the orchestrator passes ``--resume-from-attempt
    <attempt_id>`` so the cells that fit CLEAN before the crash (e.g. the 3 em
    cells) are staged into the local ``out_dir`` and skipped by the resume logic —
    the next run only re-fits the cells that failed. Only files that pass the
    schema validator overwrite-protect: an existing local cell is never clobbered
    (local wins), and a downloaded file that fails validation is discarded.

    Returns the number of cells staged. Missing prefix / empty backup → 0 (no-op,
    not an error: a fresh run with no prior attempt resumes nothing).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    prefix = f"issue722_partial/{attempt_id}/eval_results_issue_722/cells/"
    try:
        files = [
            f
            for f in list_repo_files(DATA_REPO, repo_type="dataset", revision="main")
            if f.startswith(prefix) and f.endswith(".json")
        ]
    except Exception as e:  # network / missing repo — non-fatal, resume nothing
        logger.warning(
            "[phase=fit_M] could not list attempt %s (%s) — staging nothing", attempt_id, e
        )
        return 0
    if not files:
        logger.info("[phase=fit_M] --resume-from-attempt %s: no cells under %s", attempt_id, prefix)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for f in files:
        name = Path(f).name  # e.g. em_L7.json
        dest = out_dir / name
        if dest.exists():
            logger.info("[phase=fit_M] staging: %s already present locally — keeping local", name)
            continue
        local = hf_hub_download(DATA_REPO, f, repo_type="dataset", revision="main")
        if not _cached_cell_valid(Path(local)):
            logger.warning("[phase=fit_M] staged %s failed schema check — discarding", name)
            continue
        dest.write_text(Path(local).read_text())
        staged += 1
        logger.info("[phase=fit_M] staged %s from attempt %s", name, attempt_id)
    logger.info("[phase=fit_M] --resume-from-attempt %s: staged %d cell(s)", attempt_id, staged)
    return staged


def fit_cell(
    behavior: str,
    layer: int,
    cells: list,
    rb_main: dict,
    rb_fact: dict | None,
    *,
    include_mlp: bool = True,
) -> dict:
    """Fit M0/M⁺/M_pseudo + all four reads for one (behavior, layer). Returns the cell JSON.

    ``include_mlp`` (default True — the ORIGINAL behavior, all #722 callers
    unchanged): fit the nonlinear MLP chain-ρ + nonlinearity-gap + MLP-shuffle
    validity reads (the 300-epoch GPU ensemble, the dominant per-cell cost). Set
    ``include_mlp=False`` for the RIDGE-ONLY headline path (#667 all-layer fast
    map-change): #722's own clean-result established the nonlinear MLP map is
    negative at every layer and below its shuffle null (8/9 cells) — "the
    closed-form ridge is the ONLY valid estimator, so the function-change reads
    are ridge-only". So at all-28-layer depth the MLP re-computes a foregone
    "MLP invalid" verdict at ~2 min/cell; skipping it (and running the MLP
    validity spot-check at 7/14/21 ONLY, in the #667 driver) keeps the headline
    (Delta_med / floor / chain-ρ / cross-transfer, ALL closed-form ridge) intact
    while removing the multi-hour cost. When False the ``chain_rho`` block omits
    the ``rho_M0_mlp`` / ``rho_Mplus_mlp`` / ``rho_M0_shuffle`` / ``nonlin_gap_*``
    keys (all MLP-derived); every ridge key is byte-for-byte the same code path.
    """
    if include_mlp:
        if os.environ.get("EPM_FORBID_SERIAL_FITS") == "1":
            raise RuntimeError(
                "fit_cell(include_mlp=True) is the SERIAL per-(behavior,layer) MLP path "
                "superseded by src/explore_persona_space/analysis/vectorized_mlp_skill.py; "
                "EPM_FORBID_SERIAL_FITS=1 is set (see "
                ".claude/rules/vectorize-many-cell-fits.md § Supersede contract)."
            )
        warnings.warn(
            "fit_cell(include_mlp=True) runs the SERIAL per-(behavior,layer) MLP fit; new "
            "sweeps should use src/explore_persona_space/analysis/vectorized_mlp_skill.py "
            "(batched, 50-100x). Serial call retained for #722/#667 reproduction only.",
            FutureWarning,
            stacklevel=2,
        )
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus = stacks["C0"], stacks["Cplus"]
    V0, Vplus = stacks["V0"], stacks["Vplus"]
    families = stacks["families"]
    cell_keys = stacks["cell_keys"]
    n = C0.shape[0]
    assert n >= 4, f"{behavior} L{layer}: only {n} cells (<4) — cannot fit"

    r_hat = _r_hat_for(behavior, layer, rb_main, rb_fact)  # (3584,)
    pca_basis = _pca_basis_v0(V0, TARGET_DIM)  # (k<=TARGET_DIM, 3584)
    V0_64 = _to64(V0, pca_basis)
    Vplus_64 = _to64(Vplus, pca_basis)
    grid = loadact.common_c_grid(stacks)  # base c0 grid (n, 3584)

    # Function-change read (HEADLINE) is the closed-form RIDGE map evaluated at the
    # fixed base grid — the linear M fidelity, and the plan §3 ridge-only path is a
    # VALID headline by construction (each floor is a difference of two equally-weak
    # refits, so refit noise cancels). The MLP is a universal-approximator UPPER
    # bound that enters only through the held-out chain-ρ + the nonlinearity-gap read
    # (§4.5.4) below — it is never evaluated at a fresh grid input (an MLP has no
    # closed-form off-LOCO read, and the function-change DV needs M(c) at a fixed c).
    # M_pseudo target = M0(Cplus): computed inside the floor_shift refit (below).
    m0_grid = _ridge_fit_predict(C0, V0_64, grid)  # (n_grid, 64)
    mplus_grid = _ridge_fit_predict(Cplus, Vplus_64, grid)

    # ---- Headline Δ_med (ridge) on the projected grid ----
    delta = mplus_grid - m0_grid  # (n_grid, 64)
    delta_full = delta @ pca_basis  # (n_grid, 3584)
    proj = np.abs(delta_full @ r_hat)  # (n_grid,)
    delta_med_ci = clustered_bootstrap_scalar(
        proj, families, statistic="median", n_resamples=N_SCALAR_BOOT
    )
    delta_med = delta_med_ci["point"]
    delta_med_mean_ci = clustered_bootstrap_scalar(
        proj, families, statistic="mean", n_resamples=N_SCALAR_BOOT
    )

    # ---- Three floors via the identical refit harness (family-clustered, MF#6) ----
    # All three pass `families` so the refit resample is family-clustered — the SAME
    # sampling unit as the headline Δ CI (clustered_bootstrap_scalar), so the
    # H_function gate compares like-for-like (a row-i.i.d. floor understated the
    # family-level variance and biased the gate).
    # M0 refit floor: refit M0 (C0→V0) pairs, eval at grid.
    # Each floor passes a skip_counter so a LinAlgError-skipped bootstrap pair
    # (the round-3 SVD-non-convergence guard in make_refit_pair) is COUNTED, not
    # silently lost; the per-floor skip counts are aggregated below and a skip
    # rate above ~5% is surfaced as a CONCERN in the cell JSON.
    sc_m0: dict = {}
    sc_mplus: dict = {}
    sc_shift: dict = {}
    floor_m0 = make_refit_pair(
        C0,
        V0,
        _refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=N_REFIT_PAIRS,
        skip_counter=sc_m0,
    )
    floor_mplus = make_refit_pair(
        Cplus,
        Vplus,
        _refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=N_REFIT_PAIRS,
        skip_counter=sc_mplus,
    )
    # shifted-design: M_pseudo (Cplus → M0(Cplus)); refit pairs of THAT map at grid.
    floor_shift = make_refit_pair(
        Cplus,
        m0_at_cplus_ridge_full(C0, V0, Cplus, pca_basis),
        _refit_ridge_fn(grid),
        grid,
        r_hat,
        families,
        n_pairs=N_REFIT_PAIRS,
        skip_counter=sc_shift,
    )
    refit_skip = _aggregate_refit_skips(behavior, layer, sc_m0, sc_mplus, sc_shift)
    floor_m0_p95 = float(np.percentile(floor_m0, 95))
    floor_mplus_p95 = float(np.percentile(floor_mplus, 95))
    floor_shift_p95 = float(np.percentile(floor_shift, 95))
    floor_combined = max(floor_m0_p95, floor_mplus_p95, floor_shift_p95)
    floor_sd_combined = max(floor_sd(floor_m0), floor_sd(floor_mplus), floor_sd(floor_shift))

    # ---- Support distance ‖cplus − c0‖ + large-shift flag ----
    support = np.linalg.norm(Cplus - C0, axis=1)  # (n,)
    shift_thresh = float(np.percentile(support, SUPPORT_SHIFT_PCTL))
    large_shift_mask = support > shift_thresh
    # Δ_med excluding large-shift cells (the grid is per-cell c0, so mask the proj).
    if large_shift_mask.any() and (~large_shift_mask).sum() >= 4:
        fam_keep = [f for f, m in zip(families, large_shift_mask, strict=True) if not m]
        proj_keep = proj[~large_shift_mask]
        delta_med_excl_ci = clustered_bootstrap_scalar(
            proj_keep, fam_keep, statistic="median", n_resamples=N_SCALAR_BOOT
        )
    else:
        delta_med_excl_ci = delta_med_ci

    # ---- Chain-ρ co-primary (LOCO, both maps) ----
    E = _load_E(behavior, cell_keys)
    keep = ~np.isnan(E)
    chain_block = {"n_with_E": int(keep.sum())}
    if keep.sum() >= 4:
        Ek = E[keep]
        fam_k = [f for f, m in zip(families, keep, strict=True) if m]
        m0_loco_ridge = _ridge_loco_pred(C0, V0_64)
        mplus_loco_ridge = _ridge_loco_pred(Cplus, Vplus_64)
        rho_m0, chain_m0 = _chain_rho_one(m0_loco_ridge[keep], pca_basis, r_hat, Ek)
        rho_mplus, chain_mplus = _chain_rho_one(mplus_loco_ridge[keep], pca_basis, r_hat, Ek)
        chain_block["rho_M0_ridge"] = rho_m0
        chain_block["rho_Mplus_ridge"] = rho_mplus
        chain_block["rho_diff_ridge"] = (
            None if (rho_m0 is None or rho_mplus is None) else float(rho_mplus - rho_m0)
        )
        if rho_m0 is not None:
            chain_block["ci_M0_ridge"] = clustered_bootstrap_spearman(chain_m0, Ek, fam_k)
        if rho_mplus is not None:
            chain_block["ci_Mplus_ridge"] = clustered_bootstrap_spearman(chain_mplus, Ek, fam_k)
        # MF#4: PAIRED ρ-shift CI on (rho_Mplus − rho_M0) over the SAME family
        # resamples (both ρs recomputed on the same resampled rows in one pass) —
        # the co-primary concordance read needs the paired difference CI, not the
        # two marginal CIs above. Computed whenever both point ρs exist.
        if rho_m0 is not None and rho_mplus is not None:
            chain_block["ci_diff_ridge"] = _clustered_paired_rho_diff_ci(
                chain_m0, chain_mplus, Ek, fam_k
            )
        # MLP chain-ρ + nonlinearity gap (read 4) + MLP-validity (shuffle) on M0.
        # Guarded by include_mlp: the 300-epoch GPU ensemble is the dominant
        # per-cell cost and #722 already established it is invalid at this n (see
        # fit_cell docstring). The RIDGE headline above is complete without it.
        if include_mlp:
            m0_loco_mlp = _mlp_loco_pred(C0, V0_64)
            mplus_loco_mlp = _mlp_loco_pred(Cplus, Vplus_64)
            rho_m0_mlp, _ = _chain_rho_one(m0_loco_mlp[keep], pca_basis, r_hat, Ek)
            rho_mplus_mlp, _ = _chain_rho_one(mplus_loco_mlp[keep], pca_basis, r_hat, Ek)
            chain_block["rho_M0_mlp"] = rho_m0_mlp
            chain_block["rho_Mplus_mlp"] = rho_mplus_mlp
            # shuffle null on M0 (refit ridge on permuted v0) — MLP-validity gate (plan §3).
            rng = np.random.default_rng(722)
            perm = rng.permutation(n)
            m0_shuf = _ridge_loco_pred(C0, V0_64[perm])
            rho_shuf, _ = _chain_rho_one(m0_shuf[keep], pca_basis, r_hat, Ek)
            chain_block["rho_M0_shuffle"] = rho_shuf
            # nonlinearity gap pre vs post: (rho_mlp - rho_ridge) under M0 and M⁺.
            if None not in (rho_m0_mlp, rho_m0):
                chain_block["nonlin_gap_M0"] = float(rho_m0_mlp - rho_m0)
            if None not in (rho_mplus_mlp, rho_mplus):
                chain_block["nonlin_gap_Mplus"] = float(rho_mplus_mlp - rho_mplus)

    # ---- Cross-transfer (read 3) ----
    cross = {}
    # M0 predicting v_plus on FT pairs (held-out ρ proxy: ridge LOCO of C0→Vplus_64)
    # vs M⁺ predicting v_plus (its own LOCO). Reverse: M⁺ predicting v0 on base pairs.
    m0_to_vplus = _ridge_loco_pred(C0, Vplus_64)
    mplus_to_vplus = _ridge_loco_pred(Cplus, Vplus_64)
    mplus_to_v0 = _ridge_loco_pred(Cplus, V0_64)
    # summarize as mean rowwise cosine to the true target (a transfer-quality scalar).
    cross["m0_to_vplus_cos"] = float(np.mean(fit658._rowwise_cos(m0_to_vplus, Vplus_64)))
    cross["mplus_to_vplus_cos"] = float(np.mean(fit658._rowwise_cos(mplus_to_vplus, Vplus_64)))
    cross["mplus_to_v0_cos"] = float(np.mean(fit658._rowwise_cos(mplus_to_v0, V0_64)))

    return {
        "behavior": behavior,
        "layer": layer,
        "n_cells": n,
        "Delta_med": delta_med,
        "Delta_med_ci": delta_med_ci,
        "Delta_med_mean_ci": delta_med_mean_ci,
        "Delta_med_excl_large_shift_ci": delta_med_excl_ci,
        "floor_M0_refit": floor_m0_p95,
        "floor_Mplus_refit": floor_mplus_p95,
        "floor_shifted": floor_shift_p95,
        "floor_combined": floor_combined,
        "floor_sd_combined": floor_sd_combined,
        "Delta_over_floor_sd": (
            None if floor_sd_combined < 1e-12 else float(delta_med / floor_sd_combined)
        ),
        "support_distance": {
            "mean": float(support.mean()),
            "p90": shift_thresh,
            "n_large_shift": int(large_shift_mask.sum()),
        },
        "chain_rho": chain_block,
        "cross_transfer": cross,
        "refit_skip": refit_skip,
        "n_families": len({*families}),
    }


def _refit_ridge_fn(grid: np.ndarray):
    """A fit_fn(Xb, Yb_full, rng) for make_refit_pair — fits ridge on the bootstrap sample.

    The PCA basis is recomputed per bootstrap sample (its OWN top-64 v0 PCs) so the
    refit is a genuine independent refit, mirroring how the headline map is fit.
    Returns predictions at `grid` projected back to 3584 so the floor's
    `delta @ r_hat` is in the same 3584-space as the headline.
    """

    def _fn(Xb: np.ndarray, Yb: np.ndarray, _rng) -> np.ndarray:
        pca = _pca_basis_v0(Yb, TARGET_DIM)
        pred64 = _ridge_fit_predict(Xb, Yb @ pca.T, grid)  # (n_grid, k)
        return pred64 @ pca  # back to (n_grid, 3584) for the r_hat projection

    return _fn


def m0_at_cplus_ridge_full(C0, V0, Cplus, pca):
    """M0 fit on (C0 → V0 top-64), predicted at Cplus, back-projected to 3584 (n,3584)."""
    Y64 = V0 @ pca.T
    pred64 = _ridge_fit_predict(C0, Y64, Cplus)
    return pred64 @ pca


def main() -> int:
    global N_REFIT_PAIRS, TARGET_DIM
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # fit658.DEVICE resolves at import (EPM_FIT_DEVICE env if set, else auto —
    # cuda when available; #876), so no hand-patch is needed here.
    logger.info("[phase=fit_M] device=%s", fit658.DEVICE)
    ap = argparse.ArgumentParser(description="Issue #722 fit M0 vs M⁺ + the four reads")
    ap.add_argument("--behaviors", nargs="+", default=list(HEADLINE_BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(SWEEP_LAYERS))
    ap.add_argument(
        "--max-cells", type=int, default=None, help="smoke: cap total cells per behavior×layer"
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="smoke: cap source_cid dirs per behavior (the distinct-c0 count; MUST be >=2)",
    )
    ap.add_argument(
        "--max-targets-per-source",
        type=int,
        default=None,
        help="smoke: cap targets per source (bounds total cells while spanning sources)",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_722/cells")
    ap.add_argument(
        "--smoke", action="store_true", help="1 behavior, 1 layer, capped sources/cells"
    )
    ap.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        help="override MLP_MAX_EPOCHS (smoke clamps the 300-epoch CPU cost; full run uses 300)",
    )
    ap.add_argument(
        "--refit-pairs",
        type=int,
        default=N_REFIT_PAIRS,
        help="bootstrap+random-init refit PAIRS per floor (smoke clamps from 100)",
    )
    ap.add_argument(
        "--target-dim",
        type=int,
        default=fit658.A35_MLP_TARGET_DIM,
        help="output target dim = top-v0 PCs (default 64; smoke clamps to bound the CPU MLP)",
    )
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="re-fit every cell even if a valid cached JSON exists (overrides resume-skip)",
    )
    ap.add_argument(
        "--resume-from-attempt",
        default=None,
        help=(
            "stage per-cell JSONs from a prior GCP-crash partial backup "
            "(issue722_partial/<attempt_id>/eval_results_issue_722/cells/*.json on the HF "
            "data repo) into --out-dir BEFORE fitting, so the resume-skip reuses the cells "
            "that ran clean before the crash (e.g. the 3 em cells). No-op if the prefix is empty."
        ),
    )
    args = ap.parse_args()
    if args.smoke:
        args.behaviors = args.behaviors[:1]
        args.layers = args.layers[:1]
        # Span >=4 SOURCES (the distinct-c0 count) so the fit is non-degenerate —
        # c_C is constant within a source, so a single source gives one input row.
        args.max_sources = args.max_sources or 6
        # Cap targets/source so the smoke spans 6 sources × 4 targets = 24 cells
        # (6 distinct c0) without the full 180-cell CPU MLP cost.
        if args.max_targets_per_source is None:
            args.max_targets_per_source = 4
        # Clamp the three dominant CPU costs so the GPU-bound MLP phase runs
        # end-to-end on the VM CPU as a carve-out smoke (the full GPU run uses
        # 300 epochs / 100 pairs / 64 dims). #658 _assert_mlp_exactness epoch-clamp.
        if args.mlp_epochs is None:
            args.mlp_epochs = 20
        args.refit_pairs = min(args.refit_pairs, 8)
        args.target_dim = min(args.target_dim, 4)
    if args.mlp_epochs is not None:
        fit658.MLP_MAX_EPOCHS = args.mlp_epochs
    N_REFIT_PAIRS = args.refit_pairs
    TARGET_DIM = args.target_dim
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Stage prior-crash partial cells (the 3 clean em cells) BEFORE the fit loop so
    # the resume-skip reuses them — the next run only re-fits the failed cells.
    if args.resume_from_attempt:
        stage_cells_from_attempt(args.resume_from_attempt, args.out_dir)

    layers = tuple(args.layers)
    behaviors = tuple(args.behaviors)
    logger.info(
        "[phase=fit_M] behaviors=%s layers=%s max_cells=%s", behaviors, layers, args.max_cells
    )

    # Run the exactness gates (#658) so a reduction-order regression fails at startup.
    fit658._assert_ridge_exactness()
    logger.info("[phase=fit_M] ridge exactness gate PASS")

    rb_main = _load_rb_main()
    rb_fact = _load_rb_fact() if "fact" in behaviors else None
    if "fact" in behaviors and rb_fact is None:
        logger.warning("fact requested but r_b_fact.pt unavailable/degenerate — dropping fact")
        behaviors = tuple(b for b in behaviors if b != "fact")

    # strict_counts asserts the verified 480-cell per-behavior×layer grid; disabled
    # whenever the grid is deliberately capped (--smoke OR an explicit cap flag).
    strict = (
        not args.smoke
        and args.max_cells is None
        and args.max_sources is None
        and args.max_targets_per_source is None
    )
    cells_by = loadact.load_cells(
        behaviors=behaviors,
        layers=layers,
        max_cells=args.max_cells,
        max_sources=args.max_sources,
        max_targets_per_source=args.max_targets_per_source,
        strict_counts=strict,
    )
    refit_skip_concerns: list[str] = []
    for behavior in behaviors:
        for layer in layers:
            # Resume support (docstring §"Per-(behavior, layer) checkpoints ...
            # make the run resumable"): skip a cell whose JSON already exists AND
            # validates against the schema so a relaunch after a crash picks up
            # where it left off (e.g. the em L7/L14/L21 JSONs recovered from the
            # round-2 preemption — or staged via --resume-from-attempt — are
            # skipped, and the run resumes at the first un-fit cell). The previous
            # loop overwrote, so the docstring promise was never true (#722 round
            # 3). `--force-rerun` re-fits everything regardless of the cache.
            out = args.out_dir / f"{behavior}_L{layer}.json"
            if not args.force_rerun and out.exists() and _cached_cell_valid(out):
                logger.info("[phase=fit_M] %s L%d (cached — skip): %s", behavior, layer, out)
                cached = json.loads(out.read_text())
                rs = cached.get("refit_skip")
                if isinstance(rs, dict) and rs.get("concern"):
                    refit_skip_concerns.append(
                        f"{behavior} L{layer}: {rs.get('n_skipped')}/{rs.get('n_attempted')} "
                        f"refit pairs skipped ({100 * rs.get('skip_frac', 0):.1f}%)"
                    )
                continue
            cells = cells_by[(behavior, layer)]
            logger.info("[phase=fit_M] %s L%d (%d cells)", behavior, layer, len(cells))
            cell = fit_cell(behavior, layer, cells, rb_main, rb_fact)
            cell["metadata"] = {
                "issue": 722,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out.write_text(json.dumps(cell, indent=2, default=float))
            rs = cell.get("refit_skip") or {}
            if rs.get("concern"):
                refit_skip_concerns.append(
                    f"{behavior} L{layer}: {rs.get('n_skipped')}/{rs.get('n_attempted')} "
                    f"refit pairs skipped ({100 * rs.get('skip_frac', 0):.1f}%)"
                )
            logger.info(
                "[phase=fit_M]   Δ_med=%.4g floor_combined=%.4g over_sd=%s",
                cell["Delta_med"],
                cell["floor_combined"],
                cell["Delta_over_floor_sd"],
            )
    if refit_skip_concerns:
        # Pod-side code never shells task.py — surface the >5% refit-skip CONCERN as
        # a loud structured log line the GCP/poller orchestrator persists via
        # `task.py raise-concern` (and the field rides each cell JSON's
        # `refit_skip.concern` to HF). >5% skipped pairs means the floor is suspect.
        logger.warning(
            "[phase=fit_M] REFIT_SKIP_CONCERN: %d cell(s) lost >5%% of refit pairs to "
            "LinAlgError — floors suspect: %s",
            len(refit_skip_concerns),
            "; ".join(refit_skip_concerns),
        )
    logger.info("[phase=fit_M] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
