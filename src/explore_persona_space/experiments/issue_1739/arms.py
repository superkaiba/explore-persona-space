"""Phase-3 arm engine for issue #1739 (round C1): 16 arms, batched end to end.

Design (vectorize-first — the origin directive binds hardest here):

- Projections are single batched einsums over ALL (layer, row) at once.
- Every fold-fit ridge readout goes through ONE (source x fold) job pool
  (:class:`RidgeJob` + :func:`_solve_ridge_groups`): each job's Gram + eigh
  is factorized ONCE (``fits.ridge_gcv_predict_per_target``) and serves
  EVERY arm/regime target that shares its design matrix with per-target GCV
  lambdas — the round-8 dedup (the old per-(arm x layer x fold) slice pool
  re-ran one eigh per target and pinned ~190 GB of per-slice fp64 copies at
  L=8000). Jobs shard round-robin across ALL visible GPUs under a bare
  ``device='cuda'``.
- Regime slices batch through :func:`run_cell_multi` / :func:`run_grid_multi`
  (unit loop OUTSIDE the regime loop): rb-independent work — ridge
  factorizations, the arm-5 MLP, arms 2/4/7/8/12/15/16 and their bootstrap
  CIs — is computed once per unit and shared across regimes.
- MLP arms ride ``analysis.vectorized_mlp_skill.fit_batched_loco_mlp_multihead``
  (group folds == the cell's shared fold ids) with all layers as one batch,
  after a ``torch.cuda.empty_cache()`` so the memory-aware chunk cap reads
  honest free bytes.
- Metrics: batched Spearman/AUROC (rank GEMMs), paired bootstrap over shared
  eval contexts (shared index draws -> per-draw ranks via COUNTING SORT over
  base-rank integer keys, bit-identical to ranking the drawn values —
  :func:`_rank_keys` / :func:`_counting_ranks`), and the selection-symmetric
  permutation null for the max-over-arms headline (selection rides per draw;
  `.claude/rules/selection-symmetric-nulls.md`).
- Matched budget: every arm consumes the SAME :class:`fits.BudgetCell`
  (identical realized rows + group-level folds per (L, draw, seed)).

Arm semantics follow plan §5. Two documented interpretations (flagged for
review): arm8 trains its ridge readout on TRUE train answers and evaluates on
MAPPED eval contexts (the "map-based answer regression" upper bound), while
arm12 trains AND evaluates on true answers (the privileged oracle). Arm 9's
"pretrain-then-finetune" is the closed-form L2-SP readout: w = argmin
||dv - Fw||^2 + lam||w - a*r_B||^2 with F = M(x) — solved as a ridge fit on
the residual target (dv - a*<F, r_B>), which degenerates EXACTLY to arm 6 at
L=0 (verified by :func:`verify_arm9_l0_degeneracy`, the plan's hard sanity
check); the full (d x d) end-to-end map fine-tune is deliberately not fit
(3584^2 params per fold x cell is not sized for this round).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739.constants import (
    AUROC_POS_THRESHOLD,
    MLP_HIDDEN,
    MLP_MAX_EPOCHS,
    N_BOOT,
    N_PERM,
    RIDGE_LAMBDAS,
)
from explore_persona_space.experiments.issue_1739.fits import (
    BudgetCell,
    MapFit,
    apply_map,
    realize_budget_cell,
    shuffled_map_weights,
)

logger = logging.getLogger(__name__)

# Arm registry (plan §5 table). ``family`` drives figure colors (one color =
# one arm family across ALL figures); ``layered`` = has a real layer axis;
# ``rb_dep`` says whether the arm's scores depend on the regime direction
# ``rb`` — the SAME partition :func:`run_cell_multi` implements by writing
# rb-independent arms through ``_put_shared`` (one ndarray reused across every
# regime) and rb-dependent arms per regime. Transfer callers read it to decide
# what may be cached across regimes (:func:`partition_transfer_roster`); the
# registry value and the dispatch body are pinned in agreement by
# ``tests/test_issue1739_transfer_roster.py::test_rb_dep_matches_dispatch``.
ARM_REGISTRY: dict[str, dict] = {
    "arm1_ctx_e1": {
        "label": "PV-project-context (E1)",
        "family": "context",
        "layered": True,
        "rb_dep": True,
    },
    "arm2_ctx_native": {
        "label": "Context-native direction",
        "family": "context",
        "layered": True,
        "rb_dep": False,
    },
    "arm3_identity_bias": {
        "label": "Identity+learned-bias",
        "family": "context",
        "layered": True,
        "rb_dep": True,
    },
    "arm4_ridge_ctx": {
        "label": "Ridge on whitened context",
        "family": "context",
        "layered": True,
        "rb_dep": False,
    },
    "arm5_mlp_ctx": {
        "label": "MLP on whitened context",
        "family": "context",
        "layered": True,
        "rb_dep": False,
    },
    "arm6_map_proj_e1": {
        "label": "Map-project (headline)",
        "family": "map",
        "layered": True,
        "rb_dep": True,
    },
    "arm7_map_ridge_pred": {
        "label": "Map-regression predicted answer",
        "family": "map",
        "layered": True,
        "rb_dep": False,
    },
    "arm8_map_ridge_true": {
        "label": "Map-regression true answer",
        "family": "map",
        "layered": True,
        "rb_dep": False,
    },
    "arm9_pretrain_ft": {
        "label": "Pretrain-then-finetune (L2-SP)",
        "family": "map",
        "layered": True,
        "rb_dep": True,
    },
    "arm10_stacked": {
        "label": "Stacked combiner",
        "family": "map",
        "layered": True,
        "rb_dep": True,
    },
    "arm11_oracle_proj": {
        "label": "Oracle-project-true-answer",
        "family": "oracle",
        "layered": True,
        "rb_dep": True,
    },
    "arm12_oracle_reg": {
        "label": "Oracle-regression-true-answer",
        "family": "oracle",
        "layered": True,
        "rb_dep": False,
    },
    "arm13_shuffled_map": {
        "label": "Shuffled-map control",
        "family": "control",
        "layered": True,
        "rb_dep": True,
    },
    "arm14_shuffled_pt": {
        "label": "Shuffled-pretrain control",
        "family": "control",
        "layered": True,
        "rb_dep": True,
    },
    "arm15_text_only": {
        "label": "Text-only sentence-embedding",
        "family": "control",
        "layered": False,
        "rb_dep": False,
    },
    "arm16_surface_feat": {
        "label": "Trivial surface features",
        "family": "control",
        "layered": False,
        "rb_dep": False,
    },
    # new-arm-round item 2 (plan v8 §4): nonlinear ORACLE ceilings — the arm-5
    # fit machinery (MLP) and the nonlinear-map round's kernel recipe (Nystrom
    # KRR) with input = the whitened TRUE answer acts (za). rb-INDEPENDENT
    # (regression on za -> dv; no regime direction anywhere), so they ride the
    # shared cache exactly like arm 12.
    "arm17_oracle_mlp": {
        "label": "Oracle-MLP-true-answer",
        "family": "oracle",
        "layered": True,
        "rb_dep": False,
    },
    "arm18_oracle_krr": {
        "label": "Oracle-kernel-true-answer",
        "family": "oracle",
        "layered": True,
        "rb_dep": False,
    },
    # Result-2 fair-protocol round (#1739, 2026-08-05): MLP readout on the
    # MAPPED answer — the arm-5 recipe (same batched helper, same
    # hyperparameters) with input mp instead of z, so it differs from arm 5
    # in INPUT only and from arm 7 in READOUT FAMILY only. rb-INDEPENDENT
    # (regression on mp -> dv; no regime direction anywhere).
    "arm19_map_mlp_pred": {
        "label": "Map-MLP predicted answer",
        "family": "map",
        "layered": True,
        "rb_dep": False,
    },
    # Fair-roster follow-up (#1739, 2026-08-06): ridge readout on the
    # SHUFFLED-weight mapped answer — the fitted-readout counterpart of the
    # arm-13 projection control, consuming the IDENTICAL per-cell shuffle
    # (same seed, same tensor; one mp_shuf serves both). Falsification test
    # of the linear-collapse argument: mp_shuf = z @ (PW) with P a row
    # permutation, so rank(PW) == rank(W) and ridge-on-shuffled-map should
    # land within noise of arms 4/7; a materially LOWER score means W's
    # conditioning makes the induced ridge prior bite — the arm4-vs-arm7
    # contrast then has content. rb-INDEPENDENT (ridge mp_shuf -> dv; no
    # regime direction anywhere). Linear-map only, exactly like arm 13.
    "arm20_shuffled_map_ridge": {
        "label": "Shuffled-map ridge control",
        "family": "control",
        "layered": True,
        "rb_dep": False,
    },
}

HEADLINE_PAIR = ("arm6_map_proj_e1", "arm2_ctx_native")  # plan §6 pre-selected pair

# Distribution-shift ladder arms (round-3 M-A): plan §4 Phase-3 names the
# generic/neutral arms {1, 3, 4, 6} for the WildChat->LMSYS(->PRISM) ladder,
# and the §4 Phase-4 distribution-shift figure additionally names the
# shuffled-map + oracle-answer companions {11, 13}. The union below is the
# transfer roster — deliberately EXCLUDING the expensive fitted arms the plan
# never puts on the ladder (MLP arm 5, L2-SP arms 9/14, stacked arm 10, text
# arms 15/16).
TRANSFER_ARMS = (
    "arm1_ctx_e1",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm11_oracle_proj",
    "arm13_shuffled_map",
)

# WIDE transfer roster: the 6 core ladder arms PLUS the four FITTED arms the
# train rung already scores but the eval rungs never did — the nonlinear
# context read (arm 5) and the three map/oracle regressions (arms 7, 8, 12).
# Every one of them is already computed on the train rung for all three
# behaviors, so a frozen layer exists for each in every committed train
# summary; their eval-rung inputs are the SAME arrays the core roster already
# threads (``z_ev`` for arm 5, ``mapfit`` for arm 7, ``za_ev`` + ``mapfit``
# for arms 8/12 — the latter two are exactly what arm 11 and arm 3 already
# require, so a rung that scores the core roster has them by construction).
# The deliberately EXCLUDED arms are the ones the plan never puts on the
# ladder AND whose transfer semantics are not well defined here: the L2-SP
# arms 9/14 (a per-regime residual fit), the stacked combiner arm 10 (needs
# ridge preds on EVERY fold, so it is incompatible with the transfer leg's
# ``ridge_folds=(0,)`` discarded-fold skip — ``run_cell_multi`` raises), and
# the text arms 15/16 (no eval-rung text features are threaded).
# Kept in ARM_REGISTRY order (like TRANSFER_ARMS) so a roster's arm ordering is
# one canonical thing everywhere — `resolve_transfer_roster` returns registry
# order too, and `partition_transfer_roster` preserves it.
TRANSFER_ARMS_WIDE = tuple(
    a
    for a in ARM_REGISTRY
    if a
    in set(TRANSFER_ARMS)
    | {"arm5_mlp_ctx", "arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg"}
)

# The wide roster MINUS the MLP. Measured basis (the round's per-cell pilot,
# `scripts/issue1739_transfer_roster_pilot.py`, production shape n_train=6468 /
# n_eval=1982 / d=3584 on CPU): arm 5 carries the overwhelming majority of the
# wide roster's added wall — it trains a 300-epoch AdamW net per (layer, fold)
# over the concatenated train+eval block, where arms 7/8/12 only add two
# closed-form ridge job groups that reuse one Gram+eigh each. This roster
# exists so the three cheap fitted arms can be filled in on their own pass
# (minutes) without waiting on the MLP, and so the MLP pass can be routed to a
# GPU device separately. It is NOT the default — the default stays `wide`.
TRANSFER_ARMS_WIDE_NOMLP = tuple(a for a in TRANSFER_ARMS_WIDE if a != "arm5_mlp_ctx")

# Named rosters a CLI may select by name (``--transfer-arms core|wide|...``).
TRANSFER_ROSTERS: dict[str, tuple[str, ...]] = {
    "core": TRANSFER_ARMS,
    "wide": TRANSFER_ARMS_WIDE,
    "wide-nomlp": TRANSFER_ARMS_WIDE_NOMLP,
}


def resolve_transfer_roster(spec: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Resolve a roster NAME (``core``/``wide``) or an explicit slug list.

    ``None`` -> the wide roster (the default for every eval rung as of the
    grid-fill round). An explicit list is validated against
    :data:`ARM_REGISTRY` and returned in registry order so a roster's arm
    ordering never depends on how the caller typed it. Raises on an unknown
    name or slug — never a silent drop.
    """
    if spec is None:
        names: list[str] = list(TRANSFER_ARMS_WIDE)
    elif isinstance(spec, str):
        if spec not in TRANSFER_ROSTERS:
            raise ValueError(
                f"unknown transfer roster {spec!r}; expected one of "
                f"{sorted(TRANSFER_ROSTERS)} or an explicit arm-slug list"
            )
        names = list(TRANSFER_ROSTERS[spec])
    else:
        names = list(spec)
        if len(names) == 1 and names[0] in TRANSFER_ROSTERS:  # --transfer-arms wide (nargs="+")
            names = list(TRANSFER_ROSTERS[names[0]])
    unknown = [a for a in names if a not in ARM_REGISTRY]
    if unknown:
        raise ValueError(f"unknown arm slug(s) {unknown}; known: {sorted(ARM_REGISTRY)}")
    order = list(ARM_REGISTRY)
    return sorted(set(names), key=order.index)


def partition_transfer_roster(roster: list[str] | tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split a roster into ``(rb_independent, rb_dependent)`` in registry order.

    The transfer leg caches the two halves differently: rb-INDEPENDENT arms
    (ridge/MLP fits over ``z`` / ``mp`` / ``za``) depend only on the realized
    ROW SET, so one fit is shared across every regime slice of a group, while
    rb-DEPENDENT arms (projections onto the regime direction) must be
    recomputed per regime. Keyed off ``ARM_REGISTRY[...]['rb_dep']``, which is
    pinned against :func:`run_cell_multi`'s own dispatch by
    ``tests/test_issue1739_transfer_roster.py``.
    """
    unknown = [a for a in roster if a not in ARM_REGISTRY]
    if unknown:
        raise ValueError(f"unknown arm slug(s) {unknown}; known: {sorted(ARM_REGISTRY)}")
    order = list(ARM_REGISTRY)
    keep = sorted(set(roster), key=order.index)
    return (
        [a for a in keep if not ARM_REGISTRY[a]["rb_dep"]],
        [a for a in keep if ARM_REGISTRY[a]["rb_dep"]],
    )


def roster_accounting_skips(
    roster: list[str] | tuple[str, ...],
    scores: dict[str, np.ndarray],
    skipped: dict[str, str],
    **extra: object,
) -> list[dict]:
    """SKIP records for requested arms that produced neither scores nor a reason.

    ``run_cell_multi`` records an explicit reason for every arm it declines
    (missing mapfit, missing answer activations, an MLP fold floor), so an arm
    that appears in NEITHER ``scores`` nor ``skipped`` is an unaccounted drop —
    the silent-hole case the drop-never-silent rule exists to prevent (a roster
    widened past what a dispatch path actually wires would otherwise vanish
    without a trace). Returns one record per unaccounted arm; the healthy case
    is an empty list.
    """
    missing = [a for a in roster if a not in scores and a not in skipped]
    order = list(ARM_REGISTRY)
    return [
        {
            "arm": a,
            "reason": (
                "roster-unaccounted: requested but produced neither scores nor a skip reason"
            ),
            **extra,
        }
        for a in sorted(set(missing), key=lambda s: order.index(s) if s in order else -1)
    ]


@dataclasses.dataclass
class CellData:
    """Everything one (behavior, variant, regime, U-rung) slice needs.

    Arrays are layer-leading over the FULL labeled table; :class:`BudgetCell`
    subsets them. ``dv`` must already exclude None-DV contexts (drop, never
    coerce). Optional inputs gate arm availability (a missing input SKIPS the
    arm with a recorded reason, never a silent zero).
    """

    z_ctx: np.ndarray  # (Ly, n, d) whitened context (variant: context_end | prefix_end)
    dv: np.ndarray  # (n,) graded 0-100 DV
    rb: np.ndarray  # (Ly, d) regime direction (E1/E2/E2p), whitened space
    z_ans: np.ndarray | None = None  # (Ly, n, d) whitened TRUE answer acts (t1)
    mapfit: MapFit | None = None
    w_shuffled: np.ndarray | None = None  # arm-13 control weights (computed if None)
    text_emb: np.ndarray | None = None  # (n, e) sentence embeddings (arm 15)
    text_features: np.ndarray | None = None  # (n, f) surface features (arm 16)
    layers: tuple[int, ...] = ()
    margins: np.ndarray | None = None  # (n,) TF fixed +/- pool margin (companion)
    per_rollout: np.ndarray | None = None  # (n, K) per-rollout scores (split-half)


def _proj(z: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Batched projection scores: (Ly, n, d) x (Ly, d) -> (Ly, n)."""
    return np.einsum("lnd,ld->ln", z, rb, optimize=True)


def _is_arange(rows: np.ndarray, n: int | None = None) -> bool:
    """True iff ``rows`` is exactly 0..len-1 (optionally == n, the full axis)."""
    rows = np.asarray(rows)
    if n is not None and rows.size != n:
        return False
    return (
        rows.ndim == 1
        and rows.size > 0
        and int(rows[0]) == 0
        and bool(np.array_equal(rows, np.arange(rows.size)))
    )


def _take_rows(arr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``arr[:, rows]`` without the fancy-index copy when rows are contiguous.

    Crash-fix r3 memory scoping: the transfer leg's two folds are contiguous
    blocks (train = ``arange(n_tr)``, eval = ``arange(n_tr, n)``), so basic
    slicing returns a VIEW of the (Ly, n, d) fp64 array instead of a
    multi-GiB copy; values are identical either way (pinned by
    ``test_take_rows_view_and_copy``). Non-contiguous row sets keep the
    fancy-index copy unchanged.
    """
    rows = np.asarray(rows)
    if rows.ndim == 1 and rows.size > 1:
        a, b = int(rows[0]), int(rows[-1])
        if b - a == rows.size - 1 and np.array_equal(rows, np.arange(a, b + 1)):
            return arr[:, a : b + 1]
    return arr[:, rows]


def _fold_masks(fold_ids: np.ndarray, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    """(F, n) boolean train / eval masks per fold."""
    ev = np.stack([fold_ids == f for f in range(n_folds)])
    return ~ev, ev


def _solve_stacked_normal_eqs(ata: np.ndarray, atb: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Solve a (Ly, k, k) x (Ly, k, 1) normal-equation stack; per-slice pinv fallback.

    The healthy path is the single batched ``np.linalg.solve`` — identical to
    the pre-fix arm-10 call. numpy raises ONE ``LinAlgError`` for the WHOLE
    batch when ANY slice is singular past the caller's ridge jitter (one
    degenerate layer killed the #1739 sycophancy fits lane ~25 h in), so on
    that raise the slices are re-solved individually: healthy slices via the
    same LAPACK ``gesv`` solve, singular slices via the ``pinv`` least-squares
    solution. Returns ``(beta, degenerate_slice_idxs)`` so the caller can flag
    the cell (``degenerate_ols``) — never a silent placeholder, never a
    swallowed error (a non-LinAlgError still propagates).
    """
    try:
        return np.linalg.solve(ata, atb), []
    except np.linalg.LinAlgError:
        pass  # fall through to the per-slice re-solve; degenerate slices flagged below
    beta = np.empty_like(atb)
    degenerate: list[int] = []
    for li in range(ata.shape[0]):
        try:
            beta[li] = np.linalg.solve(ata[li], atb[li])
        except np.linalg.LinAlgError:
            beta[li] = np.linalg.pinv(ata[li]) @ atb[li]
            degenerate.append(li)
    return beta, degenerate


@dataclasses.dataclass
class RidgeJob:
    """One shared-design ridge solve: (source, fold) with T stacked targets.

    ``src`` is the FULL (S, n, din) float64 source array; ``tr_rows`` selects
    the fold's train rows (materialized lazily inside the worker so the job
    list never pins per-(arm x fold) copies — the round-8 resident-memory
    fix: the old slice pool held one (ntr, d) fp64 copy PER (arm, layer,
    fold), ~190 GB at L=8000). ``targets`` are (target_key, y_full) pairs
    with y_full (n,) layer-shared or (S, n) layer-varying; ``evals`` are
    (name, eval_src, rows) triples predicted with the SAME per-target
    weights.
    """

    key: tuple
    src: np.ndarray
    tr_rows: np.ndarray
    targets: list[tuple[object, np.ndarray]]
    evals: list[tuple[str, np.ndarray, np.ndarray]]


def _ridge_devices(device: str) -> list[str]:
    """Ridge-solver device fan-out: bare 'cuda' shards jobs across ALL GPUs."""
    if device == "cuda":
        import torch

        n = torch.cuda.device_count()
        if n > 1:
            return [f"cuda:{i}" for i in range(n)]
    return [device]


def _run_ridge_job(
    job: RidgeJob, *, lambdas: tuple[float, ...], device: str
) -> tuple[tuple, dict[str, np.ndarray]]:
    """Materialize + solve one RidgeJob on ``device`` (thread-pool worker)."""
    from explore_persona_space.experiments.issue_1739.fits import ridge_gcv_predict_per_target

    n_s = job.src.shape[0]
    n_tr = len(job.tr_rows)
    # Contiguous row blocks (the transfer leg's two folds) slice as VIEWS —
    # the ridge helper's per-layer-chunk torch copies are the only
    # materialization then (crash-fix r3; values identical either way).
    x_tr = _take_rows(job.src, job.tr_rows)
    cols = []
    for _tkey, y_full in job.targets:
        if y_full.ndim == 1:
            cols.append(np.broadcast_to(y_full[job.tr_rows], (n_s, n_tr)))
        else:
            cols.append(y_full[:, job.tr_rows])
    y = np.stack(cols, axis=2)  # (S, ntr, T)
    ev_mats = [_take_rows(esrc, rows) for _name, esrc, rows in job.evals]
    preds = ridge_gcv_predict_per_target(x_tr, y, ev_mats, lambdas=lambdas, device=device)
    return job.key, {name: p for (name, _e, _r), p in zip(job.evals, preds, strict=True)}


def _solve_ridge_groups(
    jobs: list[RidgeJob],
    *,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
) -> dict[tuple, dict[str, np.ndarray]]:
    """Solve the cell's (source x fold) ridge jobs, sharded across GPUs.

    Each job's Gram + eigh is computed ONCE and serves every stacked target
    (per-target GCV — :func:`fits.ridge_gcv_predict_per_target`); with >1
    visible GPU under a bare ``device='cuda'`` the independent jobs
    round-robin across devices via a thread pool (torch releases the GIL in
    the C ops). Results: ``{job.key: {eval_name: (S, nev, T)}}``.
    """
    devices = _ridge_devices(device)
    out: dict[tuple, dict[str, np.ndarray]] = {}
    if len(devices) == 1 or len(jobs) <= 1:
        for job in jobs:
            key, preds = _run_ridge_job(job, lambdas=lambdas, device=devices[0])
            out[key] = preds
        return out
    from concurrent.futures import ThreadPoolExecutor

    logger.info("[arms] ridge jobs sharded across %s (%d jobs)", devices, len(jobs))
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futs = [
            pool.submit(_run_ridge_job, job, lambdas=lambdas, device=devices[i % len(devices)])
            for i, job in enumerate(jobs)
        ]
        for fut in futs:
            key, preds = fut.result()
            out[key] = preds
    return out


def verify_arm9_l0_degeneracy(data: CellData, *, device: str = "cpu", n_rows: int = 24) -> None:
    """Plan §4 #31 hard sanity gate — run the REAL arm-9 path at its L->0 limit.

    Builds a tiny probe cell (first <= ``n_rows`` labeled rows, first layer
    only) whose dv IS the arm-6 projection ``r_B^T M(x)``: on that cell the
    arm-9 L2-SP path must realize alpha == 1 exactly (cov == var bitwise) and
    a residual target of exactly 0 (whose ridge readout predicts 0) — the
    L->0 degenerate limit where the prior pins the readout to the pretrained
    direction, so arm-9 == arm-6. Unlike the round-1 tautology this executes
    run_cell's ACTUAL alpha-estimation + residual-ridge code: perturbing the
    alpha formula or the residual assembly flips the gate (pinned by
    ``tests/test_issue1739_fits.py::test_arm9_l0_gate_flips_on_perturbation``).
    Raises AssertionError on divergence.
    """
    assert data.mapfit is not None, "arm9 gate needs a fitted map"
    n = min(int(n_rows), data.z_ctx.shape[1])
    if data.mapfit.kind == "linear":
        sub_map = MapFit(
            w=data.mapfit.w[:1],
            x_mu=data.mapfit.x_mu[:1],
            x_sd=data.mapfit.x_sd[:1],
            y_mu=data.mapfit.y_mu[:1],
            diagnostics={},
        )
    else:
        # A nonlinear map has no (w, mu, sd) to slice — but its payloads ARE
        # per-layer, so the first-layer probe map slices identically. The
        # arm-9 L->0 degeneracy invariant is a property of the arm-9 L2-SP
        # path, independent of the map family, so the gate still runs.
        sub_map = MapFit(
            w=None,
            x_mu=None,
            x_sd=None,
            y_mu=None,
            diagnostics={},
            kind=data.mapfit.kind,
            nl_payloads=data.mapfit.nl_payloads[:1],
            apply_device=data.mapfit.apply_device,
        )
    z_sub = np.asarray(data.z_ctx[:1, :n], dtype=np.float64)
    rb_sub = np.asarray(data.rb[:1], dtype=np.float64)
    dv_probe = _proj(apply_map(z_sub, sub_map), rb_sub)[0]  # dv == arm-6 scores
    sub = CellData(
        z_ctx=z_sub, dv=dv_probe, rb=rb_sub, mapfit=sub_map, layers=(data.layers or (0,))[:1]
    )
    cell = realize_budget_cell([f"g{i:03d}" for i in range(n)], budget_l=n, draw=0, seed=0)
    scores, skipped = run_cell(
        sub, cell, arms=["arm6_map_proj_e1", "arm9_pretrain_ft"], device=device
    )
    assert not skipped, f"arm9 gate: unexpected skips {skipped}"
    s6, s9 = scores["arm6_map_proj_e1"][0], scores["arm9_pretrain_ft"][0]
    assert np.allclose(s9, s6, atol=1e-8), (
        "arm9 L2-SP must degenerate to arm6 at the L->0 limit "
        f"(max abs diff {float(np.max(np.abs(s9 - s6))):.3e})"
    )
    logger.info("[arms] arm9 L->0 degeneracy gate PASS (n=%d probe rows)", n)


def run_cell_multi(  # noqa: C901 — deliberate single dispatch block over the 20 registry arms
    datas: list[CellData],
    cell: BudgetCell,
    *,
    arms: list[str] | None = None,
    device: str = "cpu",
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    mlp_kwargs: dict | None = None,
    ridge_folds: tuple[int, ...] | None = None,
    diagnostics: list[dict] | None = None,
) -> list[tuple[dict[str, np.ndarray], dict[str, str]]]:
    """Pooled-OOF scores for every requested arm, for R regime slices AT ONCE.

    ``datas`` share every regime-INDEPENDENT input by object identity
    (``z_ctx`` / ``z_ans`` / ``dv`` / ``mapfit`` / text features — asserted)
    and differ only in the regime direction ``rb`` (+ ``w_shuffled``), so the
    expensive rb-independent work — the ridge Gram+eigh factorizations, the
    arm-5 MLP fit, arms 2/4/7/8/12/15/16/17/18/19/20 — is computed ONCE and shared;
    rb-dependent arms (1/3/6/9/10/11/13/14) are cheap projections/assembly
    per regime. Returns one ``(scores, skipped)`` pair per data, in order;
    shared arms reuse the SAME ndarray object across regimes (evaluate-side
    caches key on identity). ``ridge_folds`` restricts which folds' ridge
    problems are SOLVED (the transfer leg's discarded-fold skip); non-solved
    folds' OOF slots stay NaN. Every arm consumes the SAME realized rows +
    folds (matched-budget protocol). ``diagnostics`` (optional out-param): an
    empty list the caller passes; one dict per regime is appended (in
    ``datas`` order) carrying non-fatal numeric-degeneracy flags — today the
    arm-10 ``degenerate_ols`` pinv-fallback record (#1739 r10 crash fix).
    """
    assert datas, "run_cell_multi needs >= 1 CellData"
    base = datas[0]
    for d in datas[1:]:
        assert d.z_ctx is base.z_ctx, "run_cell_multi: datas must share z_ctx (by identity)"
        assert d.z_ans is base.z_ans, "run_cell_multi: datas must share z_ans"
        assert d.dv is base.dv, "run_cell_multi: datas must share dv"
        assert d.mapfit is base.mapfit, "run_cell_multi: datas must share mapfit"
        assert d.text_emb is base.text_emb and d.text_features is base.text_features
        assert d.w_shuffled is base.w_shuffled
    n_r = len(datas)
    want = list(ARM_REGISTRY) if arms is None else list(arms)
    idx, folds = cell.row_idx, cell.fold_ids
    n_l, n_folds = len(idx), cell.n_folds
    if n_folds < 2:
        raise RuntimeError(
            f"matched-budget OOF needs >=2 group folds; cell L={cell.budget_l} realized "
            f"{n_folds} fold(s) over {n_l} rows (labeled table too small / one group)"
        )
    # Crash-fix r3 memory scoping: the transfer leg passes the IDENTITY row
    # set over an already-fp64 comb — alias instead of fancy-copying two
    # (Ly, n, d) fp64 arrays (~37 GiB at the hall L=16000 comb; z/za are
    # never mutated downstream — reads, torch copies, .astype copies only).
    _alias = _is_arange(idx, base.z_ctx.shape[1])
    if _alias and base.z_ctx.dtype == np.float64:
        z = base.z_ctx  # (Ly, n_l, d)
    else:
        z = np.asarray(base.z_ctx[:, idx], dtype=np.float64)  # (Ly, n_l, d)
    dv = np.asarray(base.dv[idx], dtype=np.float64)
    rbs = [np.asarray(d.rb, dtype=np.float64) for d in datas]
    n_layers = z.shape[0]
    if base.z_ans is None:
        za = None
    elif _alias and base.z_ans.dtype == np.float64:
        za = base.z_ans
    else:
        za = np.asarray(base.z_ans[:, idx], dtype=np.float64)
    # mp is a whole-array (Ly, n_l, d) fp64 product — build it ONLY when a
    # map-consuming arm is actually requested (pre-r3 it was unconditional:
    # +18.6 GiB on the arm-5-only / oracle rosters). The "no mapfit" skips
    # below stay reachable only for slugs in `want`, which is exactly when
    # need_mp would have been True — so skip semantics are unchanged.
    mp_arms = {
        "arm6_map_proj_e1",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm9_pretrain_ft",
        "arm10_stacked",
        "arm13_shuffled_map",
        "arm14_shuffled_pt",
        "arm19_map_mlp_pred",
        # arm 20 consumes the map's WEIGHT tensor (shuffled), not mp itself —
        # grouped here (like arm 13) so the "no mapfit" skip semantics below
        # stay exact for both shuffled-weight controls.
        "arm20_shuffled_map_ridge",
    }
    need_mp = base.mapfit is not None and bool(mp_arms & set(want))
    # mp is built BELOW, after the arm-20 batch-1 solve (sequencing fix,
    # fair-roster follow-up 2026-08-06): deferring it keeps mp_shuf's lifetime
    # disjoint from mp's, so at most THREE (Ly, n_l, d) fp64 residents coexist
    # at any point — z / za / mp_shuf during batch 1, z / za / mp afterwards —
    # instead of four. Solving the mps jobs first WITHOUT deferring mp would
    # NOT cut the peak: all four arrays would already be allocated when any
    # solve starts.
    tr_masks, ev_masks = _fold_masks(folds, n_folds)  # (F, n_l)
    tr_w = tr_masks.astype(np.float64)
    tr_w /= np.maximum(tr_w.sum(axis=1, keepdims=True), 1.0)
    row_of = np.arange(n_l)

    scores: list[dict[str, np.ndarray]] = [{} for _ in range(n_r)]
    skipped: list[dict[str, str]] = [{} for _ in range(n_r)]
    diag: list[dict] = [{} for _ in range(n_r)]
    if diagnostics is not None:
        diagnostics.extend(diag)  # SAME dict objects — later in-place mutation propagates

    def _skip(slug: str, reason: str) -> None:
        if slug in want:
            for sk in skipped:
                sk[slug] = reason

    def _put_shared(slug: str, arr: np.ndarray) -> None:
        for sc in scores:  # SAME object per regime — evaluate caches key on id()
            sc[slug] = arr

    # Fold-row machinery + the ridge-job registry are hoisted ABOVE the
    # projection arms so the arm-20 (mps) batch can solve FIRST — see the
    # batch-1 block below and the mp-deferral note above.
    ev_rows = [np.flatnonzero(ev_masks[f]) for f in range(n_folds)]
    tr_rows = [np.flatnonzero(tr_masks[f]) for f in range(n_folds)]
    solve_folds = (
        list(range(n_folds))
        if ridge_folds is None
        else [f for f in range(n_folds) if f in set(ridge_folds)]
    )
    jobs: list[RidgeJob] = []
    tpos: dict[str, dict[object, int]] = {}  # source -> target_key -> column
    solved: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    def _job_targets(source: str, targets: list[tuple[object, np.ndarray]]) -> None:
        tpos[source] = {tkey: i for i, (tkey, _y) in enumerate(targets)}

    # ---- shared shuffled-weight map product (arms 13 + 20 read ONE shuffle) ----
    # Hoisted out of the arm-13 branch (fair-roster follow-up, 2026-08-06):
    # arm 20 is an rb-INDEPENDENT ridge on mp_shuf, so the product is a shared
    # per-unit input like z / mp / za — computed ONCE per cell (seed-dep only)
    # and consumed by BOTH the projection control (13, per regime) and the
    # ridge control (20, shared across regimes), never re-derived per regime.
    # Both arms read the IDENTICAL w_shuf tensor by construction. MEMORY
    # (batch-1 sequencing fix): everything mp_shuf feeds — the arm-13
    # projections AND the arm-20 (mps) ridge jobs — runs HERE, before mp is
    # built; mp_shuf is then RELEASED (refcount-verified) so the main ridge
    # batch below never sees a fourth (Ly, n_l, d) fp64 resident.
    mp_shuf = None
    want_mps_ridge = False
    shuf_want = {"arm13_shuffled_map", "arm20_shuffled_map_ridge"} & set(want)
    if need_mp and shuf_want:
        if base.mapfit.kind != "linear":
            # The shuffled-map controls row-permute the map's WEIGHT tensor
            # (norm-preserving by construction). A nonlinear map has no such
            # tensor, so neither control has an analogue this round — RECORD
            # the skip, never a silent zero.
            for slug in sorted(shuf_want):
                _skip(
                    slug,
                    f"shuffled-weight control is linear-map only (map_kind={base.mapfit.kind})",
                )
        else:
            w_shuf = (
                base.w_shuffled
                if base.w_shuffled is not None
                else shuffled_map_weights(base.mapfit.w, seed=cell.seed)
            )
            mp_shuf = apply_map(z, base.mapfit, w=w_shuf)  # ONE shuffle, both controls
            if "arm13_shuffled_map" in want:
                for r in range(n_r):
                    scores[r]["arm13_shuffled_map"] = _proj(mp_shuf, rbs[r])
            # ---- batch 1: arm-20 (mps) ridge solves FIRST, then mp_shuf is freed ----
            want_mps_ridge = "arm20_shuffled_map_ridge" in want
            n_mps = 0
            if want_mps_ridge:
                _job_targets("mps", [("arm20", dv)])
                mps_jobs = [
                    RidgeJob(
                        ("mps", f),
                        mp_shuf,
                        tr_rows[f],
                        [("arm20", dv)],
                        [("ev", mp_shuf, ev_rows[f])],
                    )
                    for f in solve_folds
                ]
                if mps_jobs:
                    solved.update(_solve_ridge_groups(mps_jobs, lambdas=lambdas, device=device))
                n_mps = len(mps_jobs)
                mps_jobs.clear()  # drop the RidgeJob -> mp_shuf references
            # Fix-engaged guard: prove mp_shuf is UNREFERENCED before batch 2.
            # getrefcount == 2 means the local name + the getrefcount argument
            # binding are the ONLY references left (a retained RidgeJob / view
            # would read higher). gc.collect() first on a miss: a cycle in the
            # solver's teardown can defer the drop without being a real leak.
            n_refs = sys.getrefcount(mp_shuf)
            if n_refs != 2:
                gc.collect()
                n_refs = sys.getrefcount(mp_shuf)
            if n_refs != 2:
                raise RuntimeError(
                    f"[arms] mp_shuf still referenced entering the main ridge batch "
                    f"(refcount={n_refs}, expected 2) — the batch-1 release regressed"
                )
            logger.info(
                "[arms] batch 1 done: %d mps ridge job(s) solved; mp_shuf released "
                "(refcount verified) before the main ridge batch",
                n_mps,
            )
            mp_shuf = None

    # mp deferred to HERE (see the note at need_mp): built only after mp_shuf
    # is gone, capping concurrent (Ly, n_l, d) fp64 residents at three.
    mp = apply_map(z, base.mapfit) if need_mp else None

    # ---- projection arms (constant across folds; OOF == the projection) ----
    if "arm1_ctx_e1" in want:
        for r in range(n_r):
            scores[r]["arm1_ctx_e1"] = _proj(z, rbs[r])
    if mp is not None:
        if "arm6_map_proj_e1" in want:
            for r in range(n_r):
                scores[r]["arm6_map_proj_e1"] = _proj(mp, rbs[r])
    else:
        for slug in (
            "arm6_map_proj_e1",
            "arm7_map_ridge_pred",
            "arm8_map_ridge_true",
            "arm9_pretrain_ft",
            "arm10_stacked",
            "arm13_shuffled_map",
            "arm14_shuffled_pt",
            "arm19_map_mlp_pred",
            "arm20_shuffled_map_ridge",
        ):
            _skip(slug, "no mapfit")
    if za is not None:
        if "arm11_oracle_proj" in want:
            for r in range(n_r):
                scores[r]["arm11_oracle_proj"] = _proj(za, rbs[r])
    else:
        for slug in (
            "arm3_identity_bias",
            "arm8_map_ridge_true",
            "arm11_oracle_proj",
            "arm12_oracle_reg",
            "arm17_oracle_mlp",
            "arm18_oracle_krr",
        ):
            _skip(slug, "no answer activations")

    # ---- arm 2: context-native direction (rb-independent — shared) ----
    if "arm2_ctx_native" in want:
        mid = np.array(
            [0.5 * (dv[m].max() + dv[m].min()) if m.any() else 0.0 for m in tr_masks]
        )  # (F,) train-score midpoint per fold
        hi = tr_masks & (dv[None, :] >= mid[:, None])
        lo = tr_masks & (dv[None, :] < mid[:, None])
        hi_w = hi.astype(np.float64) / np.maximum(hi.sum(axis=1, keepdims=True), 1.0)
        lo_w = lo.astype(np.float64) / np.maximum(lo.sum(axis=1, keepdims=True), 1.0)
        direction = np.einsum("fn,lnd->lfd", hi_w - lo_w, z, optimize=True)  # (Ly, F, d)
        s_all = np.einsum("lfd,lnd->lfn", direction, z, optimize=True)  # (Ly, F, n_l)
        _put_shared("arm2_ctx_native", s_all[:, folds, row_of])
        if (lo.sum(axis=1) == 0).any():
            logger.warning("[arms] arm2: a fold had zero low-side train rows (flat dv?)")

    # ---- arm 3: identity+learned-bias (bias shared; projection per regime) ----
    if "arm3_identity_bias" in want and za is not None:
        b = np.einsum("fn,lnd->lfd", tr_w, za - z, optimize=True)  # (Ly, F, d)
        for r in range(n_r):
            bias_proj = np.einsum("lfd,ld->lf", b, rbs[r], optimize=True)  # (Ly, F)
            scores[r]["arm3_identity_bias"] = _proj(z, rbs[r]) + bias_proj[:, folds]
        del b

    # ---- ridge job pool, batch 2 (arms 4, 7, 8, 12, 15, 16 + per-regime 9/14
    # residuals; arm 20 already solved in batch 1 above) ----
    run_stack = "arm10_stacked" in want and mp is not None
    if run_stack and ridge_folds is not None:
        raise ValueError("arm10_stacked needs ridge preds on EVERY fold (no ridge_folds subset)")

    # arm 9 / 14: closed-form L2-SP — alpha per (regime, layer, fold) on train
    # rows; the residual targets ride the SHARED mp factorization below.
    l2sp: list[dict[str, tuple[np.ndarray, np.ndarray]]] = [{} for _ in range(n_r)]
    resid_full: dict[tuple[int, str], list[np.ndarray]] = {}  # (r, slug) -> per-fold (Ly, n_l)
    if mp is not None:
        for r in range(n_r):
            rb_variants = {}
            if "arm9_pretrain_ft" in want:
                rb_variants["arm9_pretrain_ft"] = rbs[r]
            if "arm14_shuffled_pt" in want:
                rng = np.random.default_rng([1739, 6, cell.seed])
                rb_shuf = np.stack([rr[rng.permutation(rr.shape[0])] for rr in rbs[r]])
                rb_variants["arm14_shuffled_pt"] = rb_shuf
            for slug, rb_v in rb_variants.items():
                s_dir = _proj(mp, rb_v)  # (Ly, n_l)
                s_mu = np.einsum("fn,ln->lf", tr_w, s_dir)  # (Ly, F) train mean of s
                d_mu = tr_w @ dv  # (F,)
                cov = np.einsum(
                    "fn,lfn->lf",
                    tr_w,
                    (s_dir[:, None, :] - s_mu[:, :, None])
                    * (dv[None, None, :] - d_mu[None, :, None]),
                )
                var = np.einsum("fn,lfn->lf", tr_w, (s_dir[:, None, :] - s_mu[:, :, None]) ** 2)
                alpha = np.where(var > 1e-30, cov / np.maximum(var, 1e-30), 0.0)  # (Ly, F)
                l2sp[r][slug] = (s_dir, alpha)
                # per-fold residual target (Ly, n_l): dv - alpha_f * s_dir
                resid_full[(r, slug)] = [
                    dv[None, :] - alpha[:, f][:, None] * s_dir for f in range(n_folds)
                ]

    emb = feats = None
    if "arm15_text_only" in want:
        if base.text_emb is None:
            _skip("arm15_text_only", "no text embeddings")
        else:
            emb = np.asarray(base.text_emb[idx], dtype=np.float64)[None]  # (1, n_l, e)
    if "arm16_surface_feat" in want:
        if base.text_features is None:
            _skip("arm16_surface_feat", "no surface features")
        else:
            feats = np.asarray(base.text_features[idx], dtype=np.float64)[None]

    # Build one RidgeJob per (source, fold): ONE Gram+eigh serves every
    # stacked target (per-target GCV — the batched-slice fix, #1739 round 8).
    # `jobs` / `tpos` / `_job_targets` are defined above the batch-1 block.
    want_arm4 = ("arm4_ridge_ctx" in want) or run_stack
    if want_arm4:
        _job_targets("z", [("arm4", dv)])
    mp_targets: list[tuple[object, np.ndarray]] = []
    if mp is not None:
        if "arm7_map_ridge_pred" in want:
            mp_targets.append(("arm7", dv))
        for (r, slug), _per_fold in sorted(resid_full.items(), key=lambda kv: str(kv[0])):
            mp_targets.append(((r, slug), np.empty(0)))  # per-fold y resolved below
        if mp_targets:
            _job_targets("mp", mp_targets)
    want_za_ridge = za is not None and (
        ("arm8_map_ridge_true" in want and mp is not None) or "arm12_oracle_reg" in want
    )
    if want_za_ridge:
        _job_targets("za", [("arm8_12", dv)])
    if emb is not None:
        _job_targets("emb", [("arm15", dv)])
    if feats is not None:
        _job_targets("feats", [("arm16", dv)])

    for f in solve_folds:
        if want_arm4:
            evals = [("ev", z, ev_rows[f])]
            if run_stack:
                evals.append(("tr", z, tr_rows[f]))
            jobs.append(RidgeJob(("z", f), z, tr_rows[f], [("arm4", dv)], evals))
        if mp is not None and mp_targets:
            targets_f: list[tuple[object, np.ndarray]] = []
            for tkey, _i in sorted(tpos["mp"].items(), key=lambda kv: kv[1]):
                if tkey == "arm7":
                    targets_f.append(("arm7", dv))
                else:
                    r, slug = tkey  # type: ignore[misc]
                    targets_f.append((tkey, resid_full[(r, slug)][f]))
            jobs.append(RidgeJob(("mp", f), mp, tr_rows[f], targets_f, [("ev", mp, ev_rows[f])]))
        if want_za_ridge:
            evals = []
            if "arm8_map_ridge_true" in want and mp is not None:
                evals.append(("mp_ev", mp, ev_rows[f]))
            if "arm12_oracle_reg" in want:
                evals.append(("za_ev", za, ev_rows[f]))
            jobs.append(RidgeJob(("za", f), za, tr_rows[f], [("arm8_12", dv)], evals))
        if emb is not None:
            jobs.append(
                RidgeJob(("emb", f), emb, tr_rows[f], [("arm15", dv)], [("ev", emb, ev_rows[f])])
            )
        if feats is not None:
            jobs.append(
                RidgeJob(
                    ("feats", f), feats, tr_rows[f], [("arm16", dv)], [("ev", feats, ev_rows[f])]
                )
            )

    if jobs:  # batch 2 — merged into the same dict the batch-1 mps solve filled
        solved.update(_solve_ridge_groups(jobs, lambdas=lambdas, device=device))

    def _scatter(source: str, ename: str, tkey: object, n_rows_ly: int) -> np.ndarray:
        arr = np.full((n_rows_ly, n_l), np.nan)
        col = tpos[source][tkey]
        for f in solve_folds:
            got = solved.get((source, f))
            if got is None:
                continue
            arr[:, ev_rows[f]] = got[ename][:, :, col]
        return arr

    if "arm4_ridge_ctx" in want and want_arm4 and solved:
        _put_shared("arm4_ridge_ctx", _scatter("z", "ev", "arm4", n_layers))
    if "arm7_map_ridge_pred" in want and mp is not None and mp_targets:
        _put_shared("arm7_map_ridge_pred", _scatter("mp", "ev", "arm7", n_layers))
    if want_za_ridge:
        if "arm8_map_ridge_true" in want and mp is not None:
            _put_shared("arm8_map_ridge_true", _scatter("za", "mp_ev", "arm8_12", n_layers))
        if "arm12_oracle_reg" in want:
            _put_shared("arm12_oracle_reg", _scatter("za", "za_ev", "arm8_12", n_layers))
    if want_mps_ridge:
        _put_shared("arm20_shuffled_map_ridge", _scatter("mps", "ev", "arm20", n_layers))
    if emb is not None:
        _put_shared("arm15_text_only", _scatter("emb", "ev", "arm15", 1))
    if feats is not None:
        _put_shared("arm16_surface_feat", _scatter("feats", "ev", "arm16", 1))

    for r in range(n_r):
        for slug, (s_dir, alpha) in l2sp[r].items():
            resid = _scatter("mp", "ev", (r, slug), n_layers)
            scores[r][slug] = alpha[:, folds] * s_dir + resid

    # ---- arm 10: stacked 2-feature combiner (shared p4 preds; s6 per regime) ----
    if run_stack:
        p4_col = tpos["z"]["arm4"]
        for r in range(n_r):
            s6 = scores[r].get("arm6_map_proj_e1")
            if s6 is None:  # arm 6 not requested — the combiner still needs its feature
                s6 = _proj(mp, rbs[r])
            out10 = np.full((n_layers, n_l), np.nan)
            deg_by_fold: dict[str, list[int]] = {}
            for f in range(n_folds):  # <=5 folds; layers batched inside
                trr, evr = tr_rows[f], ev_rows[f]
                p4_in = solved[("z", f)]["tr"][:, :, p4_col]  # (Ly, n_tr) in-sample train preds
                p4_oof = solved[("z", f)]["ev"][:, :, p4_col]
                a_tr = np.stack(
                    [np.ones((n_layers, len(trr))), s6[:, trr], p4_in], axis=2
                )  # (Ly, n_tr, 3)
                ata = a_tr.transpose(0, 2, 1) @ a_tr + 1e-8 * np.eye(3)
                atb = a_tr.transpose(0, 2, 1) @ dv[trr, None]
                beta, deg_layers = _solve_stacked_normal_eqs(ata, atb)  # (Ly, 3, 1)
                if deg_layers:
                    deg_by_fold[str(f)] = deg_layers
                a_ev = np.stack([np.ones((n_layers, len(evr))), s6[:, evr], p4_oof], axis=2)
                out10[:, evr] = (a_ev @ beta)[:, :, 0]
            if deg_by_fold:
                logger.warning(
                    "[arms] arm10: singular normal-eq slice(s) -> pinv fallback "
                    "(regime=%d n_slices=%d fold->layers=%s)",
                    r,
                    sum(len(v) for v in deg_by_fold.values()),
                    deg_by_fold,
                )
                diag[r]["arm10_stacked"] = {"degenerate_ols": True, "fold_layers": deg_by_fold}
            scores[r]["arm10_stacked"] = out10

    # ---- arm 5: batched group-fold MLP (rb-independent — fit ONCE, shared) ----
    if "arm5_mlp_ctx" in want and (n_l - int(ev_masks.sum(axis=1).max())) < 2:
        # The callee's own ddof-1 floor (vectorized_mlp_skill: n - max_fold
        # >= 2). Unreachable at production budgets (L >= 250); at degenerate
        # tiny cells the arm records a SKIP reason instead of crashing the
        # whole matched-budget grid (drop-never-silent).
        _skip(
            "arm5_mlp_ctx",
            f"mlp fold floor: largest fold holds {int(ev_masks.sum(axis=1).max())} "
            f"of {n_l} rows (< 2 train rows)",
        )
    elif "arm5_mlp_ctx" in want:
        from explore_persona_space.analysis.vectorized_mlp_skill import (
            MLPGroup,
            fit_batched_loco_mlp_multihead,
        )

        if str(device).startswith("cuda"):
            # Release allocator-cached ridge blocks BEFORE the MLP resolves its
            # memory-aware chunk cap — a retained cache shrinks mem_get_info's
            # free reading and collapses the chunk (#1739 pilot: 4096 -> 20).
            import torch

            torch.cuda.empty_cache()
        kw = {"hidden": MLP_HIDDEN, "max_epochs": MLP_MAX_EPOCHS, "device": device}
        kw.update(mlp_kwargs or {})
        groups = [
            MLPGroup(key=("arm5", li), X=z[li].astype(np.float32), Y=dv[:, None].astype(np.float32))
            for li in range(n_layers)
        ]
        res = fit_batched_loco_mlp_multihead(groups, row_groups=folds, **kw)
        _put_shared(
            "arm5_mlp_ctx",
            np.stack([res.preds_by_key[("arm5", li)][:, 0] for li in range(n_layers)]),
        )

    # ---- arm 17: oracle MLP on TRUE answer acts (rb-independent — shared) ----
    # Mirrors the arm-5 dispatch block with X = za (new-arm-round item 2): same
    # batched helper, same ddof-1 fold floor, differing from arm 5 in INPUT only.
    if "arm17_oracle_mlp" in want and za is not None:
        if (n_l - int(ev_masks.sum(axis=1).max())) < 2:
            _skip(
                "arm17_oracle_mlp",
                f"mlp fold floor: largest fold holds {int(ev_masks.sum(axis=1).max())} "
                f"of {n_l} rows (< 2 train rows)",
            )
        else:
            from explore_persona_space.analysis.vectorized_mlp_skill import (
                MLPGroup,
                fit_batched_loco_mlp_multihead,
            )

            if str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()  # same honest-free-bytes read as arm 5
            kw = {"hidden": MLP_HIDDEN, "max_epochs": MLP_MAX_EPOCHS, "device": device}
            kw.update(mlp_kwargs or {})
            groups17 = [
                MLPGroup(
                    key=("arm17", li),
                    X=za[li].astype(np.float32),
                    Y=dv[:, None].astype(np.float32),
                )
                for li in range(n_layers)
            ]
            res17 = fit_batched_loco_mlp_multihead(groups17, row_groups=folds, **kw)
            _put_shared(
                "arm17_oracle_mlp",
                np.stack([res17.preds_by_key[("arm17", li)][:, 0] for li in range(n_layers)]),
            )

    # ---- arm 19: MLP on the MAPPED answer (rb-independent — shared) ----
    # Result-2 fair-protocol round: mirrors the arm-5 dispatch block with
    # X = mp (the mapped answer) — same batched helper, same hyperparameters,
    # same ddof-1 fold floor — differing from arm 5 in INPUT only and from
    # arm 7 in readout family only. The mapfit-absent case is recorded by the
    # "no mapfit" skip block above.
    if "arm19_map_mlp_pred" in want and mp is not None:
        if (n_l - int(ev_masks.sum(axis=1).max())) < 2:
            _skip(
                "arm19_map_mlp_pred",
                f"mlp fold floor: largest fold holds {int(ev_masks.sum(axis=1).max())} "
                f"of {n_l} rows (< 2 train rows)",
            )
        else:
            from explore_persona_space.analysis.vectorized_mlp_skill import (
                MLPGroup,
                fit_batched_loco_mlp_multihead,
            )

            if str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()  # same honest-free-bytes read as arm 5
            kw = {"hidden": MLP_HIDDEN, "max_epochs": MLP_MAX_EPOCHS, "device": device}
            kw.update(mlp_kwargs or {})
            groups19 = [
                MLPGroup(
                    key=("arm19", li),
                    X=mp[li].astype(np.float32),
                    Y=dv[:, None].astype(np.float32),
                )
                for li in range(n_layers)
            ]
            res19 = fit_batched_loco_mlp_multihead(groups19, row_groups=folds, **kw)
            _put_shared(
                "arm19_map_mlp_pred",
                np.stack([res19.preds_by_key[("arm19", li)][:, 0] for li in range(n_layers)]),
            )

    # ---- arm 18: oracle Nystrom KRR on TRUE answer acts (rb-independent) ----
    # The nonlinear-map round's kernel recipe (median-heuristic gamma,
    # inner-val (gamma, lambda) grid, one _eigh_robust per (layer, gamma)
    # shared across the lambda grid) with input = za, target = dv — see
    # fits.krr_scalar_fold_predict. The split seed is derived from the fold's
    # TRAIN ROW SET (not cell.seed) so the transfer leg's rb-independent
    # row-set cache stays exact: two units sharing a row set share the fit.
    if "arm18_oracle_krr" in want and za is not None:
        from explore_persona_space.experiments.issue_1739.fits import krr_scalar_fold_predict

        min_tr = min((len(tr_rows[f]) for f in solve_folds), default=0)
        if min_tr < 5:
            _skip(
                "arm18_oracle_krr",
                f"krr floor: smallest solved fold-train has {min_tr} rows (< 5: no "
                "inner/val split)",
            )
        else:
            out18 = np.full((n_layers, n_l), np.nan)
            for f in solve_folds:
                # contiguous fold blocks (the transfer leg) slice as views —
                # the krr fitter's per-layer-chunk torch copies are the only
                # materialization (crash-fix r3)
                out18[:, ev_rows[f]] = krr_scalar_fold_predict(
                    _take_rows(za, tr_rows[f]),
                    dv[tr_rows[f]],
                    _take_rows(za, ev_rows[f]),
                    seed=_rowset_seed(tr_rows[f]),
                    device=device,
                )
            _put_shared("arm18_oracle_krr", out18)

    return list(zip(scores, skipped, strict=True))


def _rowset_seed(rows: np.ndarray) -> int:
    """Deterministic seed derived from a fold's TRAIN row set (arm-18 splits).

    The transfer leg caches rb-independent arm fits per ROW SET
    (:func:`partition_transfer_roster` + the ``rs_key`` cache in the fits CLI),
    so an rb-independent fit must be a pure function of the row set — a
    ``cell.seed``-keyed inner split would silently serve one seed's fit to
    every unit sharing the rows. sha1 of the sorted int64 row bytes, folded to
    31 bits (np.random.default_rng seed range).
    """
    import hashlib

    b = np.ascontiguousarray(np.sort(np.asarray(rows, dtype=np.int64))).tobytes()
    return int.from_bytes(hashlib.sha1(b).digest()[:4], "big") % (2**31)


def run_cell(
    data: CellData,
    cell: BudgetCell,
    *,
    arms: list[str] | None = None,
    device: str = "cpu",
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    mlp_kwargs: dict | None = None,
    ridge_folds: tuple[int, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Single-regime wrapper over :func:`run_cell_multi` (same contract)."""
    return run_cell_multi(
        [data],
        cell,
        arms=arms,
        device=device,
        lambdas=lambdas,
        mlp_kwargs=mlp_kwargs,
        ridge_folds=ridge_folds,
    )[0]


def _concat_train_eval(full: np.ndarray, idx: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """``np.concatenate([full[:, idx], ev], axis=1)`` without the slice copy.

    Fills a preallocated output per layer, so the transient is ONE layer
    instead of a whole (Ly, n_tr, d) fancy-index copy (~13 GiB at the hall
    L=16000 transfer comb — crash-fix r3). Value-identical to the
    concatenate expression (pinned by
    ``test_concat_train_eval_matches_concatenate``).
    """
    n_tr, n_ev = len(idx), ev.shape[1]
    out = np.empty(
        (full.shape[0], n_tr + n_ev, full.shape[2]),
        dtype=np.result_type(full.dtype, ev.dtype),
    )
    for li in range(full.shape[0]):
        out[li, :n_tr] = full[li][idx]
        out[li, n_tr:] = ev[li]
    return out


def run_transfer_cell(
    data: CellData,
    cell: BudgetCell,
    z_ev: np.ndarray,
    dv_ev: np.ndarray,
    *,
    za_ev: np.ndarray | None = None,
    arms: list[str] | None = None,
    device: str = "cpu",
    ridge_folds: tuple[int, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Frozen-predictor transfer scores for the eval-split ladder (plan §4).

    Concatenates the TRAIN cell's rows with EVERY eval-split row and reuses
    :func:`run_cell`'s fold machinery with exactly TWO folds: fold 1 = the
    train cell rows, fold 0 = the eval rows — so the returned OOF values at
    the eval positions come from arms fit on the FULL train cell and NEVER
    on eval DV (the plan's train -> eval-rung transfer semantics; the
    reverse-fold fit is discarded — pass ``ridge_folds=(0,)`` to SKIP solving
    it outright: only fold 0's predictions reach the returned eval block, so
    the outputs are unchanged while the discarded fold-1 ridge fit — a full
    Gram+eigh on the train block — is never computed). Default arm roster:
    :data:`TRANSFER_ARMS` (cheap projection / closed-form / single-ridge
    arms only). Eval arrays must share the train slice's whitening + layer
    subset.

    Returns ``(scores_ev, skipped)``: ``scores_ev[slug]`` is the eval-row
    block ``(rows, n_ev)``.
    """
    want = list(TRANSFER_ARMS) if arms is None else list(arms)
    idx = cell.row_idx
    n_tr, n_ev = len(idx), int(z_ev.shape[1])
    assert z_ev.shape[0] == data.z_ctx.shape[0], (z_ev.shape, data.z_ctx.shape)
    z_ans_comb = None
    if data.z_ans is not None and za_ev is not None:
        z_ans_comb = _concat_train_eval(data.z_ans, idx, za_ev)
    comb = CellData(
        z_ctx=_concat_train_eval(data.z_ctx, idx, z_ev),
        dv=np.concatenate([np.asarray(data.dv[idx], dtype=np.float64), dv_ev]),
        rb=data.rb,
        z_ans=z_ans_comb,
        mapfit=data.mapfit,
        w_shuffled=data.w_shuffled,
        layers=data.layers,
    )
    cell_t = BudgetCell(
        row_idx=np.arange(n_tr + n_ev),
        fold_ids=np.concatenate([np.ones(n_tr, dtype=np.int64), np.zeros(n_ev, dtype=np.int64)]),
        n_folds=2,
        budget_l=cell.budget_l,
        draw=cell.draw,
        seed=cell.seed,
        fold_scheme="transfer-train-vs-eval",
    )
    scores, skipped = run_cell(comb, cell_t, arms=want, device=device, ridge_folds=ridge_folds)
    return {slug: sc[:, n_tr:] for slug, sc in scores.items()}, skipped


def frozen_layer_idx(rho_per_layer: list[float]) -> int:
    """Frozen-layer INDEX from a train record's per-layer rhos (nan-safe argmax)."""
    if len(rho_per_layer) <= 1:
        return 0
    return int(np.nanargmax([r if np.isfinite(r) else -np.inf for r in rho_per_layer]))


def evaluate_transfer(
    scores_ev: dict[str, np.ndarray],
    dv_ev: np.ndarray,
    rungs_ev: np.ndarray,
    frozen_by_arm: dict[str, int],
    *,
    provenance: dict,
    cell: BudgetCell,
    layers: tuple[int, ...] = (),
    n_boot: int = N_BOOT,
    min_n: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Per-(arm, eval-rung) transfer rows at the TRAIN-frozen layer.

    The frozen layer index comes from the TRAIN cell's own record (never
    selected on eval outcome). Rungs with fewer than ``min_n`` finite rows
    are recorded in the returned ``skips`` list (drop-never-silent), never
    scored — Spearman over <3 rows is undefined/degenerate. Each kept row
    carries a per-rung bootstrap CI (batched draws).
    """
    dv_ev = np.asarray(dv_ev, dtype=np.float64)
    rungs_ev = np.asarray([str(r) for r in rungs_ev])
    rows: list[dict] = []
    skips: list[dict] = []
    for slug, sc in sorted(scores_ev.items()):
        if slug not in frozen_by_arm:
            skips.append({"arm": slug, "reason": "no train frozen layer (arm absent)"})
            continue
        fl = min(int(frozen_by_arm[slug]), sc.shape[0] - 1)
        s = np.asarray(sc[fl], dtype=np.float64)
        for rung in sorted(set(rungs_ev)):
            m = (rungs_ev == rung) & np.isfinite(s) & np.isfinite(dv_ev)
            n = int(m.sum())
            if n < min_n:
                skips.append({"arm": slug, "eval_rung": rung, "n_eval": n, "reason": "min_n"})
                continue
            rho = float(spearman_rows(s[m][None], dv_ev[m])[0])
            idx_b = make_bootstrap_idx(n, n_boot=n_boot, seed=cell.seed + 100 * cell.draw)
            draws = bootstrap_rhos(s[m][None], dv_ev[m], idx_b)[0]
            rows.append(
                {
                    **provenance,
                    "arm": slug,
                    "family": ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                    "eval_rung": rung,
                    "rung_kind": "eval_transfer",
                    "rho_frozen": rho,
                    "ci_frozen": [float(np.nanquantile(draws, q)) for q in (0.025, 0.975)],
                    "n_eval": n,
                    "layer": int(layers[fl])
                    if layers and sc.shape[0] > 1
                    else (fl if sc.shape[0] > 1 else None),
                    "budget_l": cell.budget_l,
                    "draw": cell.draw,
                    "seed": cell.seed,
                }
            )
    return rows, skips


# ---------------------------------------------------------------------------
# metrics: batched Spearman / AUROC / bootstrap / permutation null
# ---------------------------------------------------------------------------


def rank_rows(a: np.ndarray) -> np.ndarray:
    """Average ranks along the LAST axis (ties averaged; scipy rankdata)."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(a, dtype=np.float64), method="average", axis=-1)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson of a (..., n) against b (n,) or (..., n) — batched."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    num = (ac * bc).sum(axis=-1)
    den = np.sqrt((ac**2).sum(axis=-1) * (bc**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def spearman_rows(scores: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Batched Spearman rho of scores (S, n) rows against dv (n,)."""
    return _pearson_rows(rank_rows(np.atleast_2d(scores)), rank_rows(dv[None])[0])


def auroc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Batched rank-formula AUROC of scores (S, n) vs boolean labels (n,)."""
    scores = np.atleast_2d(scores)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[0], np.nan)
    ranks = rank_rows(scores)
    pos_rank_sum = ranks[:, labels].sum(axis=1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _rank_keys(a: np.ndarray) -> np.ndarray:
    """Integer resample-rank keys: 2 x average base ranks (half-integers -> ints).

    Ranking is invariant under any strictly-increasing value map that
    preserves ties, and the base average ranks ARE such a map of the values
    (equal values share one rank; distinct values order identically) — so
    ``rank(a[draw]) == rank(keys[draw])`` EXACTLY for every resample draw.
    Doubling makes the half-integer average ranks exact int64 keys in
    [2, 2n], enabling the counting-sort ranking below.
    """
    return np.rint(2.0 * rank_rows(np.atleast_2d(a))).astype(np.int64)


def _counting_ranks(keys: np.ndarray, n_bins: int) -> np.ndarray:
    """Average ranks along the last axis of int64 ``keys`` via counting sort.

    ``keys`` (R, n) with values in [0, n_bins); returns fp64 average ranks —
    BIT-IDENTICAL to ``rank_rows`` on the same data (a bin holding c entries
    ending at cumulative count m occupies ranks m-c+1..m, average
    m - (c-1)/2; counts/cumsums are exact integers < 2^53). This replaces the
    per-draw argsort-based rankdata in the bootstrap (~6x wall there:
    counting sort beats fp64 argsort on resampled integer keys).
    """
    r, _n = keys.shape
    offs = (np.arange(r, dtype=np.int64) * n_bins)[:, None]
    counts = np.bincount((keys + offs).ravel(), minlength=r * n_bins).reshape(r, n_bins)
    csum = np.cumsum(counts, axis=1)
    avg = csum - (counts - 1) / 2.0  # fp64 average rank per occupied bin
    return np.take_along_axis(avg, keys, axis=1)


def _bootstrap_pearson_from_ranks(ranks_s: np.ndarray, ranks_d: np.ndarray) -> np.ndarray:
    """Pearson of rank rows via the moment identity — BIT-IDENTICAL, fewer passes.

    ``ranks_s`` (S, C, n) x ``ranks_d`` (C, n). Average ranks are exact
    half-integers, every rank sum is n(n+1)/2, and all moment sums (products
    are multiples of 0.25, totals < 2^53) are EXACT in fp64 — so
    ``sum(xy) - (sum x)(sum y)/n`` equals the centered ``sum((x-mx)(y-my))``
    bitwise (both are exact integers-over-4), and this needs ~3 fused passes
    where the generic centered path needs ~6 plus three (S, C, n) temporaries.
    The cross term rides a batched GEMV (BLAS-threaded).
    """
    _s_rows, _c, n = ranks_s.shape
    tot = n * (n + 1) / 2.0  # sum of ANY n-element rank multiset (ties included)
    mean_term = tot * tot / n
    sq_s = np.einsum("scn,scn->sc", ranks_s, ranks_s)
    sq_d = np.einsum("cn,cn->c", ranks_d, ranks_d)
    num = np.matmul(ranks_s.transpose(1, 0, 2), ranks_d[:, :, None])[:, :, 0].T  # (S, C)
    cov = num - mean_term
    var_s = sq_s - mean_term
    var_d = sq_d - mean_term
    den = np.sqrt(var_s * var_d[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)


def bootstrap_rhos(
    scores: np.ndarray,
    dv: np.ndarray,
    idx: np.ndarray,
    *,
    chunk_draws: int = 64,
) -> np.ndarray:
    """Paired-bootstrap Spearman: scores (S, n), shared draws idx (B, n) -> (S, B).

    The SAME index draws are applied to every score row (paired differences
    stay valid). Per-draw ranks are computed by counting sort over the BASE
    ranks' integer keys (:func:`_rank_keys` / :func:`_counting_ranks`), and
    the per-draw Pearson uses the exact moment identity
    (:func:`_bootstrap_pearson_from_ranks`) — BOTH bit-identical to ranking
    the drawn values + centered-Pearson directly (equivalence test-pinned)
    while cutting the per-draw argsort and most of the (S, C, n) passes; the
    only loop is over draw CHUNKS to bound memory.
    """
    scores = np.atleast_2d(scores)
    s_rows, n = scores.shape
    n_boot = idx.shape[0]
    n_bins = 2 * n + 1
    keys_s = _rank_keys(scores)  # (S, n) int64
    keys_d = _rank_keys(dv)[0]  # (n,)
    out = np.empty((s_rows, n_boot))
    for lo in range(0, n_boot, chunk_draws):
        sl = idx[lo : lo + chunk_draws]  # (C, n)
        c = sl.shape[0]
        gk = keys_s[:, sl].reshape(s_rows * c, n)  # (S*C, n)
        ranks_s = _counting_ranks(gk, n_bins).reshape(s_rows, c, n)
        ranks_d = _counting_ranks(keys_d[sl], n_bins)  # (C, n)
        out[:, lo : lo + c] = _bootstrap_pearson_from_ranks(ranks_s, ranks_d)
    return out


def make_bootstrap_idx(n: int, *, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Shared (B, n) resample index draws for the paired bootstrap."""
    rng = np.random.default_rng([1739, 7, int(seed), int(n)])
    return rng.integers(0, n, size=(n_boot, n))


def permutation_null_max(
    scores: np.ndarray, dv: np.ndarray, *, n_perm: int = N_PERM, seed: int = 0
) -> dict:
    """Selection-symmetric permutation null for the max-over-rows headline.

    ``scores`` (R, n) = one row PER SELECTION CANDIDATE — the caller passes
    every (arm x layer) row the observed max selects over (M2: frozen-layer
    rows alone would be selection-asymmetric on the layer axis). DV ranks are
    permuted (B, n); the selection (max over rows) RIDES PER DRAW
    (selection-symmetric-nulls.md). One GEMM: standardized score ranks (R, n)
    @ standardized permuted dv ranks (n, B).
    """
    scores = np.atleast_2d(scores)
    n = scores.shape[1]
    rng = np.random.default_rng([1739, 8, int(seed), int(n)])
    r_s = rank_rows(scores)
    zs = r_s - r_s.mean(axis=1, keepdims=True)
    zs /= np.maximum(np.sqrt((zs**2).sum(axis=1, keepdims=True)), 1e-30)
    r_d = rank_rows(dv[None])[0]
    perm = np.argsort(rng.random((n_perm, n)), axis=1)
    r_perm = r_d[perm]  # permuting values == permuting their ranks
    zd = r_perm - r_perm.mean(axis=1, keepdims=True)
    zd /= np.maximum(np.sqrt((zd**2).sum(axis=1, keepdims=True)), 1e-30)
    rho_null = zs @ zd.T  # (A, B)
    null_max = np.nanmax(rho_null, axis=0)
    observed = spearman_rows(scores, dv)
    obs_max = float(np.nanmax(observed))
    p = float((np.sum(null_max >= obs_max) + 1) / (n_perm + 1))
    return {
        "observed_max_rho": obs_max,
        "p_max_over_arms": p,
        "n_perm": int(n_perm),
        "null_max_q95": float(np.quantile(null_max, 0.95)),
        "null_max_q99": float(np.quantile(null_max, 0.99)),
    }


def nested_layer_selection(
    scores: np.ndarray, dv: np.ndarray, fold_ids: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Nested layer selection within the labeled budget (held-in predictivity).

    For fold f the layer is selected by pooled-OOF rho over the COMPLEMENT
    rows (data outside f — held-in wrt f), then fold f's rows take that
    layer's scores. Returns (selected scores (n,), {fold: layer_idx}).
    """
    scores = np.atleast_2d(scores)
    n = scores.shape[1]
    sel_scores = np.full(n, np.nan)
    sel_layers: dict[int, int] = {}
    for f in np.unique(fold_ids):
        outside = fold_ids != f
        rho = spearman_rows(scores[:, outside], dv[outside])
        li = int(np.nanargmax(rho)) if np.isfinite(rho).any() else 0
        sel_layers[int(f)] = li
        sel_scores[fold_ids == f] = scores[li, fold_ids == f]
    return sel_scores, sel_layers


def selection_and_frozen_ci(
    scores: np.ndarray,
    dv: np.ndarray,
    idx: np.ndarray,
    *,
    chunk_draws: int = 64,
    draws: np.ndarray | None = None,
) -> dict:
    """Frozen-layer AND selection-inherited bootstrap CIs for one arm.

    Frozen: the layer argmax is chosen ONCE on the observed data, its rho
    bootstrapped. Selection-inherited: the argmax over layers is re-taken
    PER DRAW (the selection rides the draw). Both requested by the
    statistics critic; report both. ``draws`` (Ly, B) short-circuits the
    bootstrap when the caller already computed it (evaluate_cell reuses one
    bootstrap per arm for the CI, the headline delta, and the
    selection-inherited read — previously three separate bootstraps).
    """
    scores = np.atleast_2d(scores)
    rho_obs = spearman_rows(scores, dv)
    frozen = int(np.nanargmax(rho_obs)) if np.isfinite(rho_obs).any() else 0
    if draws is None:
        draws = bootstrap_rhos(scores, dv, idx, chunk_draws=chunk_draws)  # (Ly, B)
    frozen_draws = draws[frozen]
    sel_draws = np.nanmax(draws, axis=0)

    def _ci(a: np.ndarray) -> list[float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    return {
        "frozen_layer_idx": frozen,
        "rho_frozen": float(rho_obs[frozen]),
        "ci_frozen": _ci(frozen_draws),
        "ci_selection_inherited": _ci(sel_draws),
    }


def split_half_ceiling(per_rollout: np.ndarray) -> dict:
    """Design-aligned split-half reliability ceiling (item-matched halves).

    ONE deterministic even/odd rollout-index partition applied identically to
    every context (llm-judging.md rule 21 — never independent per-condition
    splits); Spearman(h1, h2) -> Spearman-Brown 2r/(1+r).
    """
    a = np.asarray(per_rollout, dtype=np.float64)
    even = np.nanmean(a[:, 0::2], axis=1)
    odd = np.nanmean(a[:, 1::2], axis=1)
    keep = np.isfinite(even) & np.isfinite(odd)
    if keep.sum() < 3:
        return {"r_half": None, "ceiling_sb": None, "n": int(keep.sum())}
    r = float(spearman_rows(even[keep][None], odd[keep])[0])
    sb = 2 * r / (1 + r) if r > -1.0 else float("nan")
    return {
        "r_half": r,
        "ceiling_sb": float(sb),
        "n": int(keep.sum()),
        "scheme": "item-aligned even-odd rollout split",
    }


# ---------------------------------------------------------------------------
# cell evaluation + grid runner (checkpoint-per-unit, resume, progress lines)
# ---------------------------------------------------------------------------


def evaluate_cell(
    scores_by_arm: dict[str, np.ndarray],
    dv: np.ndarray,
    cell: BudgetCell,
    *,
    provenance: dict,
    margins: np.ndarray | None = None,
    per_rollout: np.ndarray | None = None,
    layers: tuple[int, ...] = (),
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    headline: tuple[str, str] = HEADLINE_PAIR,
    _shared_cache: dict | None = None,
) -> dict:
    """Metrics for one budget cell: per-arm rows + headline delta + null.

    Per arm: per-layer rho, frozen-layer rho/AUROC + CI, selection-inherited
    CI, nested-selection rho, rho vs the TF-margin companion. Cross-arm:
    paired bootstrap (SHARED draws) for the pre-selected headline delta
    (frozen AND selection-inherited) + the selection-symmetric permutation
    null over the max-of-all-arms. The per-arm bootstrap draws are computed
    ONCE and reused for the headline delta (the paired shared-idx draws of a
    frozen row equal that row of the full-layer bootstrap, so the values are
    unchanged). ``_shared_cache`` (run_grid_multi's per-unit cache, keyed on
    ``(slug, id(scores))``) reuses the rb-independent arms' identical
    per-arm computations across the unit's regime slices.
    """
    labels = dv >= AUROC_POS_THRESHOLD
    idx = make_bootstrap_idx(len(dv), n_boot=n_boot, seed=cell.seed + 100 * cell.draw)
    arm_rows: list[dict] = []
    frozen_scores: dict[str, np.ndarray] = {}
    frozen_rho: dict[str, float] = {}
    frozen_idx: dict[str, int] = {}
    draws_by_arm: dict[str, np.ndarray] = {}
    for slug, sc in sorted(scores_by_arm.items()):
        cache_key = (slug, id(sc))
        cached = _shared_cache.get(cache_key) if _shared_cache is not None else None
        if cached is None:
            draws = bootstrap_rhos(sc, dv, idx)  # (Ly, B)
            rho_layers = spearman_rows(sc, dv)
            ci = selection_and_frozen_ci(sc, dv, idx, draws=draws)
            fl = ci["frozen_layer_idx"]
            sel_scores, sel_layers = nested_layer_selection(sc, dv, cell.fold_ids)
            cached = {
                "draws": draws,
                "rho_layers": rho_layers,
                "ci": ci,
                "rho_nested": float(spearman_rows(sel_scores[None], dv)[0]),
                "sel_layers": sel_layers,
                "auroc": float(auroc_rows(sc[fl][None], labels)[0]),
            }
            if _shared_cache is not None:
                _shared_cache[cache_key] = cached
        rho_layers, ci = cached["rho_layers"], cached["ci"]
        fl = ci["frozen_layer_idx"]
        row = {
            "arm": slug,
            "family": ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
            "layer": int(layers[fl])
            if layers and sc.shape[0] > 1
            else (fl if sc.shape[0] > 1 else None),
            "rho_per_layer": [float(r) for r in rho_layers],
            "rho_frozen": ci["rho_frozen"],
            "ci_frozen": ci["ci_frozen"],
            "ci_selection_inherited": ci["ci_selection_inherited"],
            "rho_nested_selection": cached["rho_nested"],
            "nested_selected_layers": {str(k): int(v) for k, v in cached["sel_layers"].items()},
            "auroc_frozen": cached["auroc"],
            **provenance,
            "budget_l": cell.budget_l,
            "draw": cell.draw,
            "seed": cell.seed,
            "fold_scheme": cell.fold_scheme,
        }
        if margins is not None and np.isfinite(margins).sum() >= 3:
            keep = np.isfinite(margins)
            row["rho_vs_tf_margin"] = float(spearman_rows(sc[fl][keep][None], margins[keep])[0])
        arm_rows.append(row)
        frozen_scores[slug] = sc[fl]
        frozen_rho[slug] = ci["rho_frozen"]
        frozen_idx[slug] = fl
        draws_by_arm[slug] = cached["draws"]

    result: dict = {"arms": arm_rows}
    a, b = headline
    if a in scores_by_arm and b in scores_by_arm:
        # Per-row bootstrap draws are row-independent under the SHARED idx, so
        # the frozen rows' paired draws are exactly the cached full-layer
        # draws' frozen rows (no re-bootstrap; values unchanged).
        da, db = draws_by_arm[a], draws_by_arm[b]
        delta = da[frozen_idx[a]] - db[frozen_idx[b]]
        delta_sel = np.nanmax(da, axis=0) - np.nanmax(db, axis=0)
        result["headline"] = {
            "pair": [a, b],
            "delta_rho_frozen": float(frozen_rho[a] - frozen_rho[b]),
            "delta_rho_frozen_boot_mean": float(np.nanmean(delta)),
            "ci_delta_frozen": [float(np.nanquantile(delta, q)) for q in (0.025, 0.975)],
            "ci_delta_selection_inherited": [
                float(np.nanquantile(delta_sel, q)) for q in (0.025, 0.975)
            ],
            "n_boot": int(idx.shape[0]),
        }
    if scores_by_arm:
        # M2 (selection symmetry on BOTH free axes): the observed headline max
        # is selected over every (arm, layer) row — each arm's frozen layer is
        # itself an argmax over layers — so the null draws must ride the SAME
        # (arm x layer) selection. Feeding frozen-layer rows only would give
        # the observed max ~A*Ly chances vs A per null draw
        # (selection-symmetric-nulls.md; round-1 review M2).
        rows = np.concatenate([np.atleast_2d(scores_by_arm[s]) for s in sorted(scores_by_arm)])
        null = permutation_null_max(rows, dv, n_perm=n_perm, seed=cell.seed + 100 * cell.draw)
        null["selection_axes"] = "arm x layer"
        null["n_rows"] = int(rows.shape[0])
        null["n_arms"] = len(scores_by_arm)
        result["max_over_arms_null"] = null
    if per_rollout is not None:
        result["split_half"] = split_half_ceiling(per_rollout)
    return result


def _unit_key(provenance: dict, budget_l: int, draw: int, seed: int, regime_extra: dict) -> str:
    """Resume key carrying EVERY output-affecting regime field (M3a / #722-r3).

    ``regime_extra`` folds in the run flags the round-1 key omitted — arm
    subset, layer subset, n_boot/n_perm, mlp overrides — so a partial-arm or
    smoke-scale run into the same out-root can never satisfy a production
    cell's resume predicate.
    """
    payload = {**provenance, "budget_l": budget_l, "draw": draw, "seed": seed, **regime_extra}
    return json.dumps(payload, sort_keys=True)


def run_grid_multi(
    datas: list[CellData],
    provenances: list[dict],
    group_keys: list[str] | np.ndarray,
    *,
    budgets: list[int],
    draws: list[int],
    seeds: list[int],
    out_dir: Path | str,
    arms: list[str] | None = None,
    device: str = "cpu",
    mlp_kwargs: dict | None = None,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    context_ids: list[str] | np.ndarray | None = None,
    unit_timings: list[dict] | None = None,
) -> list[list[dict]]:
    """Run every (L, draw, seed) cell for R regime slices AT ONCE; checkpoint per unit.

    The unit axis loops OUTSIDE the regime axis so each realized cell's
    rb-independent work (ridge factorizations, the arm-5 MLP, the shared-arm
    bootstrap CIs) is computed ONCE and shared across the R regime slices
    (:func:`run_cell_multi` + evaluate's per-unit ``_shared_cache``) — the
    #1739 round-8 cross-unit batching. Everything else preserves
    :func:`run_grid`'s contract per (unit, regime): ONE JSONL line (O_APPEND)
    under ``out_dir/percell/cells.jsonl`` with the SAME unit-key grammar +
    one stdout progress line + the per-context prediction sidecar; resumed
    keys are SKIPPED with their stored records loaded into the returned
    per-regime lists (M3a/M3b). ``unit_timings`` (optional) collects one
    ``{budget_l, draw, seed, wall_s, n_regimes}`` row per COMPUTED unit — the
    per-budget pilot projection basis.
    """
    assert len(datas) == len(provenances) and datas, (len(datas), len(provenances))
    base = datas[0]
    out_dir = Path(out_dir)
    percell = out_dir / "percell" / "cells.jsonl"
    percell.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if percell.exists():
        with percell.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["unit_key"]] = rec
    regime_extra = {
        "arms_subset": sorted(arms) if arms is not None else "all",
        "layers_subset": [int(x) for x in (base.layers or ())],
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "mlp_kwargs": {k: mlp_kwargs[k] for k in sorted(mlp_kwargs)} if mlp_kwargs else {},
    }
    if base.mapfit is not None:
        verify_arm9_l0_degeneracy(base, device=device)  # plan §4 #31 gate (once per grid)
    units = [(lb, dr, sd) for lb in budgets for dr in draws for sd in seeds]
    results: list[list[dict]] = [[] for _ in datas]
    t0 = time.time()
    for k, (budget_l, draw, seed) in enumerate(units):
        keys = [
            _unit_key(provenances[r], budget_l, draw, seed, regime_extra) for r in range(len(datas))
        ]
        pending: list[int] = []
        for r, key in enumerate(keys):
            if key in done:
                results[r].append(done[key])  # M3b: resumed cells still reach the summary
                print(
                    f"[fits] unit {k + 1}/{len(units)} SKIP (resume) {budget_l}/{draw}/{seed} "
                    f"regime={provenances[r].get('regime')}",
                    flush=True,
                )
            else:
                pending.append(r)
        if not pending:
            continue
        t_unit = time.time()
        cell = realize_budget_cell(group_keys, budget_l=budget_l, draw=draw, seed=seed)
        print(
            f"[fits] batched slice: {len(pending)} regime-unit(s) in one solve "
            f"L={budget_l} draw={draw} seed={seed}",
            flush=True,
        )
        cell_diags: list[dict] = []
        outs = run_cell_multi(
            [datas[r] for r in pending],
            cell,
            arms=arms,
            device=device,
            mlp_kwargs=mlp_kwargs,
            diagnostics=cell_diags,
        )
        dv_cell = base.dv[cell.row_idx]
        margins_cell = base.margins[cell.row_idx] if base.margins is not None else None
        pr_cell = base.per_rollout[cell.row_idx] if base.per_rollout is not None else None
        shared_cache: dict = {}
        for j, r in enumerate(pending):
            scores, skipped = outs[j]
            rec = evaluate_cell(
                scores,
                dv_cell,
                cell,
                provenance=provenances[r],
                margins=margins_cell,
                per_rollout=pr_cell,
                layers=base.layers,
                n_boot=n_boot,
                n_perm=n_perm,
                _shared_cache=shared_cache,
            )
            rec["unit_key"] = keys[r]
            rec["skipped_arms"] = skipped
            # #1739 r10: rank-deficiency audit flag — downstream analysis can
            # see which cells' arm-10 fell back to the pinv solution.
            rec["degenerate_ols"] = bool(cell_diags[j])
            if cell_diags[j]:
                rec["degenerate_ols_detail"] = cell_diags[j]
            if context_ids is not None:
                rec["preds_npz"] = _save_cell_preds(
                    out_dir / "percell" / "preds", keys[r], rec, scores, cell, context_ids, base.dv
                )
            line = json.dumps(rec, sort_keys=True) + "\n"
            with percell.open("a", encoding="utf-8") as fh:  # single-line O_APPEND write
                fh.write(line)
                fh.flush()
            results[r].append(rec)
            print(
                f"[fits] unit {k + 1}/{len(units)} L={budget_l} draw={draw} seed={seed} "
                f"regime={provenances[r].get('regime')} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        if unit_timings is not None:
            unit_timings.append(
                {
                    "budget_l": int(budget_l),
                    "draw": int(draw),
                    "seed": int(seed),
                    "wall_s": float(time.time() - t_unit),
                    "n_regimes": len(pending),
                }
            )
    return results


def run_grid(
    data: CellData,
    group_keys: list[str] | np.ndarray,
    *,
    budgets: list[int],
    draws: list[int],
    seeds: list[int],
    provenance: dict,
    out_dir: Path | str,
    arms: list[str] | None = None,
    device: str = "cpu",
    mlp_kwargs: dict | None = None,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    context_ids: list[str] | np.ndarray | None = None,
) -> list[dict]:
    """Single-regime wrapper over :func:`run_grid_multi` (same contract/keys)."""
    return run_grid_multi(
        [data],
        [provenance],
        group_keys,
        budgets=budgets,
        draws=draws,
        seeds=seeds,
        out_dir=out_dir,
        arms=arms,
        device=device,
        mlp_kwargs=mlp_kwargs,
        n_boot=n_boot,
        n_perm=n_perm,
        context_ids=context_ids,
    )[0]


def _save_cell_preds(
    preds_dir: Path,
    unit_key: str,
    rec: dict,
    scores: dict[str, np.ndarray],
    cell: BudgetCell,
    context_ids: list[str] | np.ndarray,
    dv: np.ndarray,
) -> str:
    """Persist per-context frozen-layer predicted scores for one cell (npz).

    Keeps within-stratum / per-context reads recomputable post-hoc (e.g. the
    evil harmful-vs-benign split) without re-running the fits: one fp32 row
    per arm at that arm's frozen layer + the cell's dv + context ids.
    Atomic write; the sidecar is HF-bound (npz is gitignored repo-wide) via
    the dispatcher's results phase. Returns the relative filename.
    """
    import hashlib

    preds_dir.mkdir(parents=True, exist_ok=True)
    frozen = {row["arm"]: frozen_layer_idx(row["rho_per_layer"]) for row in rec.get("arms", [])}
    name = hashlib.sha1(unit_key.encode()).hexdigest()[:16] + ".npz"
    payload = {
        "row_idx": cell.row_idx.astype(np.int64),
        "context_ids": np.asarray([str(context_ids[i]) for i in cell.row_idx]),
        "dv": np.asarray(dv[cell.row_idx], dtype=np.float32),
        "unit_key": np.asarray(unit_key),
    }
    for slug, fl in frozen.items():
        sc = scores.get(slug)
        if sc is not None:
            payload[f"pred__{slug}"] = np.asarray(sc[min(fl, sc.shape[0] - 1)], dtype=np.float32)
    tmp = preds_dir / (name + ".tmp.npz")  # np.savez appends .npz to non-.npz names (#1092)
    with tmp.open("wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, preds_dir / name)
    return name


def transfer_preds_rows(
    scores_ev: dict[str, np.ndarray],
    dv_ev: np.ndarray,
    ctx_ids: list[str] | np.ndarray,
    frozen_by_arm: dict[str, int],
    *,
    provenance: dict,
    layers: tuple[int, ...] = (),
    labels: dict[str, list | np.ndarray] | None = None,
) -> list[dict]:
    """Per-(arm, eval context) frozen-layer transfer predictions — JSONL rows.

    The eval-rung twin of :func:`_save_cell_preds` (which persists TRAIN-cell
    predictions as npz). Persisting these is what makes any later subset read
    over an eval rung — per-polarity, per-rung, per-quantile — a pure
    re-analysis instead of another re-score; per-arm rho rows alone cannot
    support one. Schema mirrors the bare-query scorer's ``preds_rows`` (one
    row per arm x context, frozen-layer score + dv + the context id), with
    the rung-specific label columns passed generically through ``labels``
    (e.g. ``{"polarity": [...]}`` for the pvsynth rung) instead of a
    hard-coded flag, so every rung can reuse the one schema.
    """
    dv = np.asarray(dv_ev, dtype=np.float64)
    ids = [str(c) for c in ctx_ids]
    n = len(ids)
    if dv.size != n:
        raise ValueError(f"dv/ctx length mismatch: {dv.size} vs {n}")
    cols = {k: list(v) for k, v in (labels or {}).items()}
    for k, v in cols.items():
        if len(v) != n:
            raise ValueError(f"label column {k!r} length {len(v)} != {n} contexts")
    out: list[dict] = []
    for slug, sc in sorted(scores_ev.items()):
        if slug not in frozen_by_arm:
            continue
        arr = np.asarray(sc, dtype=np.float64)
        fl = min(int(frozen_by_arm[slug]), arr.shape[0] - 1)
        row_scores = arr[fl]
        layer = int(layers[fl]) if layers and arr.shape[0] > 1 else None
        for i in range(n):
            out.append(
                {
                    **provenance,
                    "arm": slug,
                    "context_id": ids[i],
                    "dv": float(dv[i]),
                    "score": float(row_scores[i]),
                    "frozen_layer_idx": int(fl),
                    "layer": layer,
                    **{k: v[i] for k, v in cols.items()},
                }
            )
    return out


def write_preds_jsonl(path: Path | str, rows: list[dict]) -> Path:
    """Write per-context prediction rows as JSONL, atomically, ONE FILE PER UNIT.

    Truncate-and-replace rather than append: the file is keyed per unit
    (behavior / variant / eval block), so re-running a unit overwrites exactly
    its own rows instead of appending duplicates, and a resumed run leaves an
    already-written unit's file untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
    os.replace(tmp, path)
    return path


def write_summary(
    records: list[dict], out_path: Path | str, *, meta: dict, extra: dict | None = None
) -> Path:
    """Aggregate per-cell records into ``all_arms_spearman.json`` (atomic).

    ``extra`` merges additional TOP-LEVEL sections (round-3 M-A: the
    ``transfer_rows`` / ``transfer_skips`` ladder sections consumed by
    ``figures.render_summary_figures``) — kept separate from ``arm_rows`` so
    in-split and cross-split reads never mix in the per-arm figures.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for rec in records for r in rec.get("arms", [])]
    payload = {
        "n_cells": len(records),
        "n_arm_rows": len(rows),
        "arm_rows": rows,
        "headlines": [rec["headline"] for rec in records if "headline" in rec],
        "nulls": [rec["max_over_arms_null"] for rec in records if "max_over_arms_null" in rec],
        "meta": meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, out_path)
    logger.info("[arms] summary -> %s (%d cells, %d arm rows)", out_path, len(records), len(rows))
    return out_path
