"""Phase-3 arm engine for issue #1739 (round C1): 16 arms, batched end to end.

Design (vectorize-first — the origin directive binds hardest here):

- Projections are single batched einsums over ALL (layer, row) at once.
- Every fold-fit ridge readout goes through ONE slice pool
  (:func:`_solve_ridge_slices`) that buckets (arm x layer x fold) problems by
  shape and solves each bucket with a single
  ``ridge_fit_predict_fast_layer_batched`` call (chunked only to bound
  memory — chunking never changes the fit). No python loop over the
  hundreds of readout cells; the only remaining loops are over the <=5
  folds (combiner/native-direction assembly) and layer CHUNKS.
- MLP arms ride ``analysis.vectorized_mlp_skill.fit_batched_loco_mlp_multihead``
  (group folds == the cell's shared fold ids) with all layers as one batch.
- Metrics: batched Spearman/AUROC (rank GEMMs), paired bootstrap over shared
  eval contexts (shared index draws -> rank ops batched over draws), and the
  selection-symmetric permutation null for the max-over-arms headline
  (selection rides per draw; `.claude/rules/selection-symmetric-nulls.md`).
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
import json
import logging
import os
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
# one arm family across ALL figures); ``layered`` = has a real layer axis.
ARM_REGISTRY: dict[str, dict] = {
    "arm1_ctx_e1": {"label": "PV-project-context (E1)", "family": "context", "layered": True},
    "arm2_ctx_native": {"label": "Context-native direction", "family": "context", "layered": True},
    "arm3_identity_bias": {"label": "Identity+learned-bias", "family": "context", "layered": True},
    "arm4_ridge_ctx": {"label": "Ridge on whitened context", "family": "context", "layered": True},
    "arm5_mlp_ctx": {"label": "MLP on whitened context", "family": "context", "layered": True},
    "arm6_map_proj_e1": {"label": "Map-project (headline)", "family": "map", "layered": True},
    "arm7_map_ridge_pred": {
        "label": "Map-regression predicted answer",
        "family": "map",
        "layered": True,
    },
    "arm8_map_ridge_true": {
        "label": "Map-regression true answer",
        "family": "map",
        "layered": True,
    },
    "arm9_pretrain_ft": {
        "label": "Pretrain-then-finetune (L2-SP)",
        "family": "map",
        "layered": True,
    },
    "arm10_stacked": {"label": "Stacked combiner", "family": "map", "layered": True},
    "arm11_oracle_proj": {
        "label": "Oracle-project-true-answer",
        "family": "oracle",
        "layered": True,
    },
    "arm12_oracle_reg": {
        "label": "Oracle-regression-true-answer",
        "family": "oracle",
        "layered": True,
    },
    "arm13_shuffled_map": {"label": "Shuffled-map control", "family": "control", "layered": True},
    "arm14_shuffled_pt": {
        "label": "Shuffled-pretrain control",
        "family": "control",
        "layered": True,
    },
    "arm15_text_only": {
        "label": "Text-only sentence-embedding",
        "family": "control",
        "layered": False,
    },
    "arm16_surface_feat": {
        "label": "Trivial surface features",
        "family": "control",
        "layered": False,
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


def _fold_masks(fold_ids: np.ndarray, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    """(F, n) boolean train / eval masks per fold."""
    ev = np.stack([fold_ids == f for f in range(n_folds)])
    return ~ev, ev


def _solve_ridge_slices(
    slices: list[tuple[tuple, np.ndarray, np.ndarray, np.ndarray]],
    *,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
    max_slice_elems: int = int(2e8),
) -> dict[tuple, np.ndarray]:
    """Solve many (X_tr, Y_tr, X_ev) ridge problems in shape-bucketed batches.

    Each slice is ``(key, X_tr (ntr,din), Y_tr (ntr,dout), X_ev (nev,din))``.
    Slices sharing a shape are stacked into ONE batched ridge call
    (:func:`fits.ridge_layer_batched_auto` — primal d x d Gram when
    ntr > din, else the parent dual helper; GCV lambda per slice);
    ``max_slice_elems`` chunks a bucket only to bound memory. The per-slice
    memory bound includes the min(ntr, din)^2 Gram + eigenvector cost of
    whichever branch the router picks (the M6 fix: at L=16k budgets the
    dual n_tr x n_tr Gram would be ~1.3 GB fp64 PER SLICE; the router takes
    the primal d x d branch there instead).
    """
    from explore_persona_space.experiments.issue_1739.fits import ridge_layer_batched_auto

    out: dict[tuple, np.ndarray] = {}
    buckets: dict[tuple, list] = {}
    for key, xt, yt, xe in slices:
        yt2 = yt if yt.ndim == 2 else yt[:, None]
        buckets.setdefault((xt.shape, yt2.shape[1], xe.shape[0]), []).append((key, xt, yt2, xe))
    for ((ntr, din), dout, nev), items in buckets.items():
        gram_side = min(ntr, din)  # the router's Gram is (gram_side, gram_side)
        per_slice = ntr * din + ntr * dout + nev * din + nev * dout + 3 * gram_side * gram_side
        chunk = max(1, min(len(items), int(max_slice_elems // max(per_slice, 1))))
        for i in range(0, len(items), chunk):
            part = items[i : i + chunk]
            x = np.stack([p[1] for p in part]).astype(np.float64)
            y = np.stack([p[2] for p in part]).astype(np.float64)
            xe = np.stack([p[3] for p in part]).astype(np.float64)
            preds = ridge_layer_batched_auto(x, y, xe, lambdas=lambdas, device=device)
            for (key, *_), pred in zip(part, preds, strict=True):
                out[key] = pred
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
    sub_map = MapFit(
        w=data.mapfit.w[:1],
        x_mu=data.mapfit.x_mu[:1],
        x_sd=data.mapfit.x_sd[:1],
        y_mu=data.mapfit.y_mu[:1],
        diagnostics={},
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


def run_cell(  # noqa: C901 — deliberate single dispatch block over the 16 plan-§5 arms
    data: CellData,
    cell: BudgetCell,
    *,
    arms: list[str] | None = None,
    device: str = "cpu",
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    mlp_kwargs: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute pooled-OOF scores for every requested arm on ONE budget cell.

    Returns ``(scores, skipped)``: ``scores[slug]`` is (Ly, n_l) for layered
    arms, (1, n_l) for layer-free arms; ``skipped[slug]`` records why an arm
    could not run (missing optional input). Every arm consumes the SAME
    realized rows + folds (matched-budget protocol).
    """
    want = list(ARM_REGISTRY) if arms is None else list(arms)
    idx, folds = cell.row_idx, cell.fold_ids
    n_l, n_folds = len(idx), cell.n_folds
    if n_folds < 2:
        raise RuntimeError(
            f"matched-budget OOF needs >=2 group folds; cell L={cell.budget_l} realized "
            f"{n_folds} fold(s) over {n_l} rows (labeled table too small / one group)"
        )
    z = np.asarray(data.z_ctx[:, idx], dtype=np.float64)  # (Ly, n_l, d)
    dv = np.asarray(data.dv[idx], dtype=np.float64)
    rb = np.asarray(data.rb, dtype=np.float64)
    n_layers = z.shape[0]
    za = np.asarray(data.z_ans[:, idx], dtype=np.float64) if data.z_ans is not None else None
    mp = apply_map(z, data.mapfit) if data.mapfit is not None else None
    tr_masks, ev_masks = _fold_masks(folds, n_folds)  # (F, n_l)
    tr_w = tr_masks.astype(np.float64)
    tr_w /= np.maximum(tr_w.sum(axis=1, keepdims=True), 1.0)
    row_of = np.arange(n_l)

    scores: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}

    def _skip(slug: str, reason: str) -> None:
        if slug in want:
            skipped[slug] = reason

    # ---- projection arms (constant across folds; OOF == the projection) ----
    if "arm1_ctx_e1" in want:
        scores["arm1_ctx_e1"] = _proj(z, rb)
    if mp is not None:
        if "arm6_map_proj_e1" in want:
            scores["arm6_map_proj_e1"] = _proj(mp, rb)
        if "arm13_shuffled_map" in want:
            w_shuf = (
                data.w_shuffled
                if data.w_shuffled is not None
                else shuffled_map_weights(data.mapfit.w, seed=cell.seed)
            )
            scores["arm13_shuffled_map"] = _proj(apply_map(z, data.mapfit, w=w_shuf), rb)
    else:
        for slug in (
            "arm6_map_proj_e1",
            "arm7_map_ridge_pred",
            "arm8_map_ridge_true",
            "arm9_pretrain_ft",
            "arm10_stacked",
            "arm13_shuffled_map",
            "arm14_shuffled_pt",
        ):
            _skip(slug, "no mapfit")
    if za is not None:
        if "arm11_oracle_proj" in want:
            scores["arm11_oracle_proj"] = _proj(za, rb)
    else:
        for slug in (
            "arm3_identity_bias",
            "arm8_map_ridge_true",
            "arm11_oracle_proj",
            "arm12_oracle_reg",
        ):
            _skip(slug, "no answer activations")

    # ---- arm 2: context-native direction (per-fold hi/lo diff of means) ----
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
        scores["arm2_ctx_native"] = s_all[:, folds, row_of]
        if (lo.sum(axis=1) == 0).any():
            logger.warning("[arms] arm2: a fold had zero low-side train rows (flat dv?)")

    # ---- arm 3: identity+learned-bias -> projection readout ----
    if "arm3_identity_bias" in want and za is not None:
        b = np.einsum("fn,lnd->lfd", tr_w, za - z, optimize=True)  # (Ly, F, d)
        bias_proj = np.einsum("lfd,ld->lf", b, rb, optimize=True)  # (Ly, F)
        scores["arm3_identity_bias"] = _proj(z, rb) + bias_proj[:, folds]

    # ---- ridge slice pool (arms 4, 7, 8, 12, 15, 16 + arm-9 residuals) ----
    slices: list[tuple[tuple, np.ndarray, np.ndarray, np.ndarray]] = []
    ev_rows = [np.flatnonzero(ev_masks[f]) for f in range(n_folds)]
    tr_rows = [np.flatnonzero(tr_masks[f]) for f in range(n_folds)]

    def _add_ridge_arm(
        slug: str, x_tr_src: np.ndarray, x_ev_src: np.ndarray, *, extra_train_preds: bool = False
    ) -> None:
        for f in range(n_folds):
            x_ev = x_ev_src[:, ev_rows[f]]
            if extra_train_preds:  # arm-10 combiner needs in-sample train preds too
                x_ev = np.concatenate([x_ev, x_ev_src[:, tr_rows[f]]], axis=1)
            for li in range(n_layers):
                slices.append(
                    ((slug, li, f), x_tr_src[li][tr_rows[f]], dv[tr_rows[f], None], x_ev[li])
                )

    run_stack = "arm10_stacked" in want and mp is not None
    if "arm4_ridge_ctx" in want or run_stack:
        _add_ridge_arm("arm4_ridge_ctx", z, z, extra_train_preds=run_stack)
    if "arm7_map_ridge_pred" in want and mp is not None:
        _add_ridge_arm("arm7_map_ridge_pred", mp, mp)
    if "arm8_map_ridge_true" in want and mp is not None and za is not None:
        _add_ridge_arm("arm8_map_ridge_true", za, mp)
    if "arm12_oracle_reg" in want and za is not None:
        _add_ridge_arm("arm12_oracle_reg", za, za)

    # arm 9 / 14: closed-form L2-SP — alpha per (layer, fold) on train rows,
    # ridge on the residual target over map features (added below), score =
    # alpha * <M(x), rb> + resid_pred.
    l2sp: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # slug -> (s_dir (Ly,n), alpha (Ly,F))
    if mp is not None:
        rb_variants = {}
        if "arm9_pretrain_ft" in want:
            rb_variants["arm9_pretrain_ft"] = rb
        if "arm14_shuffled_pt" in want:
            rng = np.random.default_rng([1739, 6, cell.seed])
            rb_shuf = np.stack([r[rng.permutation(r.shape[0])] for r in rb])
            rb_variants["arm14_shuffled_pt"] = rb_shuf
        for slug, rb_v in rb_variants.items():
            s_dir = _proj(mp, rb_v)  # (Ly, n_l)
            # alpha per (Ly, F) from TRAIN rows only: cov(dv, s)/var(s), centered on train means.
            s_mu = np.einsum("fn,ln->lf", tr_w, s_dir)  # (Ly, F) train mean of s
            d_mu = tr_w @ dv  # (F,)
            cov = np.einsum(
                "fn,lfn->lf",
                tr_w,
                (s_dir[:, None, :] - s_mu[:, :, None]) * (dv[None, None, :] - d_mu[None, :, None]),
            )
            var = np.einsum("fn,lfn->lf", tr_w, (s_dir[:, None, :] - s_mu[:, :, None]) ** 2)
            alpha = np.where(var > 1e-30, cov / np.maximum(var, 1e-30), 0.0)  # (Ly, F)
            l2sp[slug] = (s_dir, alpha)
            for f in range(n_folds):
                for li in range(n_layers):
                    resid_tr = dv[tr_rows[f]] - alpha[li, f] * s_dir[li, tr_rows[f]]
                    slices.append(
                        ((slug, li, f), mp[li][tr_rows[f]], resid_tr[:, None], mp[li][ev_rows[f]])
                    )

    if "arm15_text_only" in want:
        if data.text_emb is None:
            _skip("arm15_text_only", "no text embeddings")
        else:
            emb = np.asarray(data.text_emb[idx], dtype=np.float64)
            for f in range(n_folds):
                slices.append(
                    (
                        ("arm15_text_only", 0, f),
                        emb[tr_rows[f]],
                        dv[tr_rows[f], None],
                        emb[ev_rows[f]],
                    )
                )
    if "arm16_surface_feat" in want:
        if data.text_features is None:
            _skip("arm16_surface_feat", "no surface features")
        else:
            feats = np.asarray(data.text_features[idx], dtype=np.float64)
            for f in range(n_folds):
                slices.append(
                    (
                        ("arm16_surface_feat", 0, f),
                        feats[tr_rows[f]],
                        dv[tr_rows[f], None],
                        feats[ev_rows[f]],
                    )
                )

    solved = _solve_ridge_slices(slices, lambdas=lambdas, device=device) if slices else {}

    def _scatter(slug: str, n_rows_ly: int) -> np.ndarray:
        arr = np.full((n_rows_ly, n_l), np.nan)
        for (s, li, f), pred in solved.items():
            if s != slug:
                continue
            arr[li, ev_rows[f]] = pred[: len(ev_rows[f]), 0]
        return arr

    if "arm4_ridge_ctx" in want and any(k[0] == "arm4_ridge_ctx" for k in solved):
        scores["arm4_ridge_ctx"] = _scatter("arm4_ridge_ctx", n_layers)
    if "arm7_map_ridge_pred" in want and any(k[0] == "arm7_map_ridge_pred" for k in solved):
        scores["arm7_map_ridge_pred"] = _scatter("arm7_map_ridge_pred", n_layers)
    if "arm8_map_ridge_true" in want and any(k[0] == "arm8_map_ridge_true" for k in solved):
        scores["arm8_map_ridge_true"] = _scatter("arm8_map_ridge_true", n_layers)
    if "arm12_oracle_reg" in want and any(k[0] == "arm12_oracle_reg" for k in solved):
        scores["arm12_oracle_reg"] = _scatter("arm12_oracle_reg", n_layers)
    for slug in ("arm15_text_only", "arm16_surface_feat"):
        if slug in want and any(k[0] == slug for k in solved):
            scores[slug] = _scatter(slug, 1)

    for slug, (s_dir, alpha) in l2sp.items():
        resid = _scatter(slug, n_layers)
        scores[slug] = alpha[:, folds] * s_dir + resid

    # ---- arm 10: stacked 2-feature combiner (arm6 proj + arm4 pred) ----
    if run_stack:
        s6 = scores.get("arm6_map_proj_e1")
        if s6 is None:  # arm 6 not requested — the combiner still needs its feature
            s6 = _proj(mp, rb)
        out10 = np.full((n_layers, n_l), np.nan)
        for f in range(n_folds):  # <=5 folds; layers batched inside
            trr, evr = tr_rows[f], ev_rows[f]
            p4_in = np.stack(
                [solved[("arm4_ridge_ctx", li, f)][len(evr) :, 0] for li in range(n_layers)]
            )  # (Ly, n_tr) in-sample train preds
            p4_oof = np.stack(
                [solved[("arm4_ridge_ctx", li, f)][: len(evr), 0] for li in range(n_layers)]
            )
            a_tr = np.stack(
                [np.ones((n_layers, len(trr))), s6[:, trr], p4_in], axis=2
            )  # (Ly, n_tr, 3)
            ata = a_tr.transpose(0, 2, 1) @ a_tr + 1e-8 * np.eye(3)
            atb = a_tr.transpose(0, 2, 1) @ dv[trr, None]
            beta = np.linalg.solve(ata, atb)  # (Ly, 3, 1)
            a_ev = np.stack([np.ones((n_layers, len(evr))), s6[:, evr], p4_oof], axis=2)
            out10[:, evr] = (a_ev @ beta)[:, :, 0]
        scores["arm10_stacked"] = out10

    # ---- arm 5: batched group-fold MLP over all layers ----
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

        kw = {"hidden": MLP_HIDDEN, "max_epochs": MLP_MAX_EPOCHS, "device": device}
        kw.update(mlp_kwargs or {})
        groups = [
            MLPGroup(key=("arm5", li), X=z[li].astype(np.float32), Y=dv[:, None].astype(np.float32))
            for li in range(n_layers)
        ]
        res = fit_batched_loco_mlp_multihead(groups, row_groups=folds, **kw)
        scores["arm5_mlp_ctx"] = np.stack(
            [res.preds_by_key[("arm5", li)][:, 0] for li in range(n_layers)]
        )

    return scores, skipped


def run_transfer_cell(
    data: CellData,
    cell: BudgetCell,
    z_ev: np.ndarray,
    dv_ev: np.ndarray,
    *,
    za_ev: np.ndarray | None = None,
    arms: list[str] | None = None,
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Frozen-predictor transfer scores for the eval-split ladder (plan §4).

    Concatenates the TRAIN cell's rows with EVERY eval-split row and reuses
    :func:`run_cell`'s fold machinery with exactly TWO folds: fold 1 = the
    train cell rows, fold 0 = the eval rows — so the returned OOF values at
    the eval positions come from arms fit on the FULL train cell and NEVER
    on eval DV (the plan's train -> eval-rung transfer semantics; the
    reverse-fold fit is discarded). Default arm roster: :data:`TRANSFER_ARMS`
    (cheap projection / closed-form / single-ridge arms only). Eval arrays
    must share the train slice's whitening + layer subset.

    Returns ``(scores_ev, skipped)``: ``scores_ev[slug]`` is the eval-row
    block ``(rows, n_ev)``.
    """
    want = list(TRANSFER_ARMS) if arms is None else list(arms)
    idx = cell.row_idx
    n_tr, n_ev = len(idx), int(z_ev.shape[1])
    assert z_ev.shape[0] == data.z_ctx.shape[0], (z_ev.shape, data.z_ctx.shape)
    z_ans_comb = None
    if data.z_ans is not None and za_ev is not None:
        z_ans_comb = np.concatenate([data.z_ans[:, idx], za_ev], axis=1)
    comb = CellData(
        z_ctx=np.concatenate([data.z_ctx[:, idx], z_ev], axis=1),
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
    scores, skipped = run_cell(comb, cell_t, arms=want, device=device)
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


def bootstrap_rhos(
    scores: np.ndarray,
    dv: np.ndarray,
    idx: np.ndarray,
    *,
    chunk_draws: int = 64,
) -> np.ndarray:
    """Paired-bootstrap Spearman: scores (S, n), shared draws idx (B, n) -> (S, B).

    The SAME index draws are applied to every score row (paired differences
    stay valid). Rank + Pearson ops are batched over (row, draw) — the only
    loop is over draw CHUNKS to bound memory.
    """
    scores = np.atleast_2d(scores)
    s_rows, _n = scores.shape
    n_boot = idx.shape[0]
    out = np.empty((s_rows, n_boot))
    for lo in range(0, n_boot, chunk_draws):
        sl = idx[lo : lo + chunk_draws]  # (C, n)
        g_scores = scores[:, sl]  # (S, C, n)
        g_dv = rank_rows(dv[sl])  # (C, n)
        out[:, lo : lo + sl.shape[0]] = _pearson_rows(rank_rows(g_scores), g_dv[None])
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
    scores: np.ndarray, dv: np.ndarray, idx: np.ndarray, *, chunk_draws: int = 64
) -> dict:
    """Frozen-layer AND selection-inherited bootstrap CIs for one arm.

    Frozen: the layer argmax is chosen ONCE on the observed data, its rho
    bootstrapped. Selection-inherited: the argmax over layers is re-taken
    PER DRAW (the selection rides the draw). Both requested by the
    statistics critic; report both.
    """
    scores = np.atleast_2d(scores)
    rho_obs = spearman_rows(scores, dv)
    frozen = int(np.nanargmax(rho_obs)) if np.isfinite(rho_obs).any() else 0
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
) -> dict:
    """Metrics for one budget cell: per-arm rows + headline delta + null.

    Per arm: per-layer rho, frozen-layer rho/AUROC + CI, selection-inherited
    CI, nested-selection rho, rho vs the TF-margin companion. Cross-arm:
    paired bootstrap (SHARED draws) for the pre-selected headline delta
    (frozen AND selection-inherited) + the selection-symmetric permutation
    null over the max-of-all-arms.
    """
    labels = dv >= AUROC_POS_THRESHOLD
    idx = make_bootstrap_idx(len(dv), n_boot=n_boot, seed=cell.seed + 100 * cell.draw)
    arm_rows: list[dict] = []
    frozen_scores: dict[str, np.ndarray] = {}
    frozen_rho: dict[str, float] = {}
    for slug, sc in sorted(scores_by_arm.items()):
        rho_layers = spearman_rows(sc, dv)
        ci = selection_and_frozen_ci(sc, dv, idx)
        fl = ci["frozen_layer_idx"]
        sel_scores, sel_layers = nested_layer_selection(sc, dv, cell.fold_ids)
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
            "rho_nested_selection": float(spearman_rows(sel_scores[None], dv)[0]),
            "nested_selected_layers": {str(k): int(v) for k, v in sel_layers.items()},
            "auroc_frozen": float(auroc_rows(sc[fl][None], labels)[0]),
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

    result: dict = {"arms": arm_rows}
    a, b = headline
    if a in scores_by_arm and b in scores_by_arm:
        sa = np.concatenate([np.atleast_2d(frozen_scores[a]), np.atleast_2d(frozen_scores[b])])
        pair_draws = bootstrap_rhos(sa, dv, idx)  # (2, B) paired (shared idx)
        delta = pair_draws[0] - pair_draws[1]
        da = bootstrap_rhos(np.atleast_2d(scores_by_arm[a]), dv, idx)
        db = bootstrap_rhos(np.atleast_2d(scores_by_arm[b]), dv, idx)
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
    """Run every (L, draw, seed) cell of one variant slice; checkpoint per unit.

    Per completed cell: ONE JSONL line (O_APPEND) under
    ``out_dir/percell/cells.jsonl`` + one stdout progress line
    (``[fits] unit k/N <key> elapsed=..s``) + (when ``context_ids`` is given)
    a per-context frozen-layer prediction sidecar
    ``out_dir/percell/preds/<sha1(unit_key)>.npz`` so post-hoc within-stratum
    reads stay recomputable (round-1 "Unaddressed Cases"). Resume: cells
    whose unit key already exists in the JSONL are SKIPPED **and their stored
    records are loaded into the returned list** (M3b — a crash+resume run's
    summary aggregates every cell, not just the current process's). The key
    carries every output-affecting regime field (provenance + arms/layers/
    n_boot/n_perm/mlp overrides — M3a).
    """
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
        "layers_subset": [int(x) for x in (data.layers or ())],
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "mlp_kwargs": {k: mlp_kwargs[k] for k in sorted(mlp_kwargs)} if mlp_kwargs else {},
    }
    if data.mapfit is not None:
        verify_arm9_l0_degeneracy(data, device=device)  # plan §4 #31 gate (once per grid)
    units = [(lb, dr, sd) for lb in budgets for dr in draws for sd in seeds]
    results: list[dict] = []
    t0 = time.time()
    for k, (budget_l, draw, seed) in enumerate(units):
        key = _unit_key(provenance, budget_l, draw, seed, regime_extra)
        if key in done:
            results.append(done[key])  # M3b: resumed cells still reach the summary
            print(
                f"[fits] unit {k + 1}/{len(units)} SKIP (resume) {budget_l}/{draw}/{seed}",
                flush=True,
            )
            continue
        cell = realize_budget_cell(group_keys, budget_l=budget_l, draw=draw, seed=seed)
        scores, skipped = run_cell(data, cell, arms=arms, device=device, mlp_kwargs=mlp_kwargs)
        rec = evaluate_cell(
            scores,
            data.dv[cell.row_idx],
            cell,
            provenance=provenance,
            margins=data.margins[cell.row_idx] if data.margins is not None else None,
            per_rollout=data.per_rollout[cell.row_idx] if data.per_rollout is not None else None,
            layers=data.layers,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        rec["unit_key"] = key
        rec["skipped_arms"] = skipped
        if context_ids is not None:
            rec["preds_npz"] = _save_cell_preds(
                out_dir / "percell" / "preds", key, rec, scores, cell, context_ids, data.dv
            )
        line = json.dumps(rec, sort_keys=True) + "\n"
        with percell.open("a", encoding="utf-8") as fh:  # single-line O_APPEND write
            fh.write(line)
            fh.flush()
        results.append(rec)
        print(
            f"[fits] unit {k + 1}/{len(units)} L={budget_l} draw={draw} seed={seed} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return results


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
