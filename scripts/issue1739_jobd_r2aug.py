#!/usr/bin/env python3
"""#1739 Job D (judged-generic readout swap) + Result-2 trait-augmented map refit.

TWO coordinated rounds over the SAME staged inputs, per behavior (read-only DV
inputs, no judge — every scoring primitive is IMPORTED from the reviewed
production modules; this file contains no fit, no metric, and no fold logic of
its own, exactly like its structural sibling ``issue1739_wcrung_arms.py``):

* ``jobd``  — the judged-generic READOUT ablation. The context->answer map and
  the whitening stay FIXED (the plotted-slice full generic U pool, 18,793
  rows); what varies is the LABELED TRAINING SET of the label-consuming
  readouts (arm4_ridge_ctx / arm7_map_ridge_pred / arm8_map_ridge_true): a
  SWAP at fixed total budget L — fraction g of the L rows drawn from the
  JUDGED WildChat-rung contexts (per-context graded trait DV, the committed
  wcrung dv_dataset), the remainder from the behavior's trait-eliciting train
  table. g in {0, 0.25, 0.5, 1.0}. Evaluated on (a) the behavior's eval-split
  rungs (train-distribution + OOD) with a single full-mixture fit, and (b)
  the WildChat rung itself via K-fold CROSS-FIT over the judged contexts
  (contamination option c): every WildChat context is scored by a readout
  whose generic rows came from OTHER folds only. Label-free arms ride along
  as reference lines — the map and whitening are fixed in this mode, so
  their values are genuinely unchanged across g (an internal consistency
  check, not a re-fit).

* ``r2aug`` — the Result-2 map-side refit at the plotted slice under THREE map
  conditions (the ``map_condition`` field on every row + meta):
    - ``generic``          — NOT re-run here: the committed generic-only rows
                             (w_fit_rows=18793) are the reference; the VM-side
                             ``--collect-generic-only`` step extracts +
                             slice-verifies them.
    - ``swap``             — f_u=0.5, f_l=1.0 at the behavior's MATCHED pool
                             cap (fits.compose_u_pool SWAP semantics — R5's).
    - ``generic_matched``  — f_u=0 at the SAME matched cap (size-matched
                             generic control; distinct from the committed
                             18,793-row ``generic`` wherever cap < 18,793 —
                             evil's cap is 12,936).
    - ``add``              — the generic pool UNION the behavior's
                             trait-eliciting TRAIN (context, answer) pairs, no
                             size cap (realized pool size recorded). E1
                             extraction pairs are deliberately EXCLUDED (that
                             is the R5 ``union_all`` arm — another lane).
  Map-side arms ONLY are re-fit (arm6 map-proj / arm7 map-ridge-pred / arm8
  map-ridge-true + the arm13 shuffled-map control). Context-direct arms are
  NOT re-fit — they do not read through the map (committed rows are the
  reference lines). Arms 9/10 are SKIPPED (user-confirmed 2026-08-05):
  excluded from the transfer machinery by design (arms.py TRANSFER_ARMS_WIDE
  note); a precedent for forcing arm10 through (ridge_folds=None,
  issue1739_rescore_ood_armfill.py) exists and was deliberately not used.
  They remain in the committed GENERIC condition only — the figure labels
  that gap explicitly and never renders an empty/zero bar for them under
  swap/add.

JOB-D DV SCALING: hallucination's TRAIN DV is a [0,1] fabricated fraction
while the judged-generic DV is a 0-100 trait rubric — raw mixing lets the
0-100 rows dominate the ridge loss by ~1e4 per row. Every jobd fit therefore
z-scores its TRAINING targets PER POOL over the fit's own selected rows
(trait rows by trait-row stats, judged-generic rows by generic-row stats);
evaluation DVs stay RAW everywhere (Spearman is monotone-invariant, so target
rescaling must never touch the evaluation side). Applied uniformly across
behaviors for comparability. Known input caveat (a finding, not worked
around): evil's judged-generic DV is near-degenerate (mean 0.42, sd 4.4 —
almost all zeros), so evil's g>0 arms mostly inject near-constant labels.

VARIANT SCOPE (user directive 2026-08-05): ``context_end`` ONLY — an explicit
user-directed deviation from the standing "prefix mapping AND context mapping"
rule, recorded in every emitted meta. Evidence basis: R5 measured the
prefix-end map at chance retrieval with negative R^2 under both compositions.

DV-construct caveat (carried in meta): the WildChat-rung DV is the graded
0-100 trait rubric for every behavior, while hallucination's TRAIN DV is the
fabricated-fraction construct — hallucination's jobd mixtures train on a
MIXED-construct target. Disclosed, inherent to the design.

Safety rails inherited from the wcrung leg: no judge module may be imported
(asserted at entry and exit), DV inputs are sha-verified after scoring, and
git-tracked outputs are refused unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_fits.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
RUNG = "wildchat_rung"

# The plotted Result-2 slice (issue1739_recut_common.LMAX — kept as a literal
# so this leg never imports the figure module's matplotlib side effects).
LMAX = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}

# Job D swap grid: fraction of the fixed readout budget drawn from the judged
# WildChat pool. L=1500 is forced by the WildChat budget: at g=1.0 all L rows
# must come from OUT-OF-FOLD judged contexts (~(K-1)/K x ~1970 kept contexts
# ~= 1573 at K=5), so L <= 1500 keeps every (behavior, g, fold) feasible.
JOBD_G_GRID = (0.0, 0.25, 0.5, 1.0)
JOBD_L_TOTAL = 1500
JOBD_K_FOLDS = 5

R2AUG_MAX_F_U = 0.5  # the swap rung's eliciting fraction (R5 semantics)

# Rosters. jobd: the committed wcrung CORE six plus the two label-consuming
# map-side ridge arms this round varies (label-free arms = fixed references,
# valid because jobd's map+whitening never change). r2aug: map-side only.
ROSTER_JOBD = (
    "arm1_ctx_e1",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm11_oracle_proj",
    "arm13_shuffled_map",
)
LABEL_CONSUMING = ("arm4_ridge_ctx", "arm7_map_ridge_pred", "arm8_map_ridge_true")
# Arms 9/10 are SKIPPED (user-confirmed 2026-08-05): excluded from the
# transfer machinery by design (arms.py TRANSFER_ARMS_WIDE note — arm9 is a
# per-regime residual fit with no defined transfer semantics; arm10 requires
# ridge preds on every fold, incompatible with the transfer leg's
# ridge_folds=(0,) discarded-fold skip). A precedent for forcing arm10
# through exists (issue1739_rescore_ood_armfill.py, ridge_folds=None) and was
# deliberately NOT used. They stay in the committed GENERIC condition's rows;
# no swap/add value exists for them — the figure must label the gap, never
# render a zero bar.
ROSTER_R2AUG = (
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm13_shuffled_map",
)

VARIANT_SCOPE_NOTE = (
    "context_end ONLY (user directive 2026-08-05) — explicit stated deviation from the "
    "standing prefix+context both-arms mapping rule; basis: R5 measured the prefix-end map "
    "at chance retrieval (kNN@1 ~ chance) with negative R^2 under both compositions"
)

DEFAULT_OUT_JOBD = Path("eval_results/issue_1739/judged_generic_ablation")
DEFAULT_OUT_R2AUG = Path("eval_results/issue_1739/result2_trait_aug")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")


# ---------------------------------------------------------------------------
# small helpers (paths, meta)
# ---------------------------------------------------------------------------


def behavior_paths(args: argparse.Namespace, behavior: str) -> dict[str, Path]:
    """Every per-behavior input path (same staged layout as the wcrung leg)."""
    from scripts.issue1739_wcrung_arms import resolve_wcrung_store

    return {
        "train_store": args.store_root / f"{behavior}_labeling",
        "train_dv": args.train_dv_root / behavior / "labeling.json",
        "e1_store": args.store_root / f"{behavior}_extraction",
        "wcrung_store": resolve_wcrung_store(args),
        "wcrung_dv": args.wcrung_dv_root / behavior / "labeling.json",
        "train_summary": args.main_root / behavior / "arm_results" / "all_arms_spearman.json",
    }


def _fitmap_ns(args: argparse.Namespace) -> argparse.Namespace:
    """The tiny namespace ``issue1739_fits._fit_map`` reads (linear default)."""
    return argparse.Namespace(
        map_kind="linear",
        device=args.device,
        seeds=(args.seed,),
        mlp_map_width=None,
        krr_map_centers=None,
    )


def _env_versions() -> dict[str, str]:
    import numpy

    out = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    try:
        import torch

        out["torch"] = torch.__version__
    except ImportError as exc:
        out["torch"] = f"unavailable ({exc.__class__.__name__})"
    return out


def _free_cuda(device: str) -> None:
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# shared per-behavior load
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Loaded:
    """Everything one behavior's scoring (both modes) reads, loaded once."""

    behavior: str
    tbl: object  # train split labeled table (config_a)
    tbl_ev: object  # eval split (train-distribution + OOD rungs)
    tbl_wc: object  # wildchat rung (judged; config_b on the wcrung DV)
    rb: object  # (Ly, d) E1 direction, RAW (un-whitened) space
    rb_meta: dict
    u_arrays: dict
    u_fit_rows: object
    dim: int
    shas: dict[str, str]
    paths: dict[str, Path]


def load_behavior(args: argparse.Namespace, behavior: str, layers: list[int]) -> Loaded:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io
    from scripts.issue1739_fits import _load_labeled
    from scripts.issue1739_wcrung_arms import _rb_for_behavior, _sha256

    paths = behavior_paths(args, behavior)
    missing = [f"{k}={v}" for k, v in paths.items() if k != "train_summary" and not v.exists()]
    if missing:
        raise FileNotFoundError(f"[{behavior}] missing input(s): {'; '.join(missing)}")
    shas = {
        str(paths["train_dv"]): _sha256(paths["train_dv"]),
        str(paths["wcrung_dv"]): _sha256(paths["wcrung_dv"]),
    }
    t0 = time.time()
    tbl = _load_labeled(
        paths["train_store"],
        paths["train_dv"],
        layers,
        config="config_a",
        need_rollout_rows=False,
    )
    tbl_ev = _load_labeled(
        paths["train_store"],
        paths["train_dv"],
        layers,
        config="config_b",
        need_rollout_rows=False,
    )
    tbl_wc = _load_labeled(
        paths["wcrung_store"],
        paths["wcrung_dv"],
        layers,
        config="config_b",
        need_rollout_rows=False,
    )
    if set(tbl_wc.rungs) != {RUNG}:
        raise RuntimeError(f"[{behavior}] wcrung DV rungs {tbl_wc.rungs} != {{{RUNG!r}}}")
    dim = tbl.z_ans.shape[-1]
    for name, t in (("eval", tbl_ev), ("wildchat", tbl_wc)):
        if t.z_ans.shape[-1] != dim:
            raise RuntimeError(f"[{behavior}] {name} hidden dim {t.z_ans.shape[-1]} != {dim}")
    rb, rb_meta = _rb_for_behavior(args, behavior, tbl, layers, dim, paths)

    store_io.stage_u_store(Path(args.u_store), ("prefix_end", "context_end", "t1"), tuple(layers))
    u_arrays, u_meta = store_io.load_summaries(
        args.u_store, ("prefix_end", "context_end", "t1"), tuple(layers), hidden_dim=dim
    )
    u_fit_rows = np.flatnonzero(store_io.fit_pool_mask(u_meta))
    print(
        f"[jobd-r2aug] {behavior}: train n={len(tbl.ctx_order)} eval n={len(tbl_ev.ctx_order)} "
        f"rungs={tbl_ev.rungs} | wc n={len(tbl_wc.ctx_order)} | u_fit={len(u_fit_rows)} | "
        f"load={time.time() - t0:.0f}s",
        flush=True,
    )
    return Loaded(
        behavior, tbl, tbl_ev, tbl_wc, rb, rb_meta, u_arrays, u_fit_rows, dim, shas, paths
    )


def committed_frozen(
    args: argparse.Namespace,
    loaded: Loaded,
    behavior: str,
    variant: str,
    layers: list[int],
    roster: tuple[str, ...],
) -> tuple[dict[str, int], str]:
    """TRAIN-frozen layer per arm, from the committed main summary (modal)."""
    from scripts.issue1739_wcrung_arms import (
        _assert_committed_frozen_indexable,
        _sha256,
        modal_frozen_layers,
    )

    summary = loaded.paths["train_summary"]
    if not summary.exists():
        raise FileNotFoundError(
            f"[{behavior}] committed train summary absent: {summary} — this leg freezes "
            "layers against the committed plotted-slice convention (no own-pool fallback)"
        )
    frozen = modal_frozen_layers(summary, variant=variant, regime=args.regime, u_rung_label="full")
    frozen = {a: i for a, i in frozen.items() if a in roster}
    missing = sorted(set(roster) - set(frozen))
    if missing:
        raise RuntimeError(f"[{behavior}/{variant}] no committed frozen layer for {missing}")
    _assert_committed_frozen_indexable(frozen, layers, behavior, variant, summary)
    loaded.shas[str(summary)] = _sha256(summary)
    return frozen, f"modal-committed-train-cells:{summary}"


# ---------------------------------------------------------------------------
# map pools + fitting (shared seam)
# ---------------------------------------------------------------------------


def build_pool(args, loaded: Loaded, variant: str, layers: list[int], condition: str):
    """(x, y, label, n, extra_meta) for one map condition's U pool.

    ``swap`` / ``generic_matched`` go through the reviewed
    ``_u_pool_for_spec`` compose branch (SWAP at fixed matched size);
    ``add`` is the two-source union: the FULL generic fit pool + ALL
    trait-eliciting TRAIN (variant-act, t1-answer) pairs. ``jobd_full`` is
    the plain full rung (the plotted-slice map).
    """
    import numpy as np

    from scripts.issue1739_fits import RunSpec, _u_pool_for_spec

    n_ctx = len(loaded.tbl.ctx_order)
    cap = min(len(loaded.u_fit_rows), int(n_ctx / R2AUG_MAX_F_U))
    lmax = LMAX[loaded.behavior]
    if condition in ("swap", "generic_matched"):
        f_u, f_l = (R2AUG_MAX_F_U, 1.0) if condition == "swap" else (0.0, 0.0)
        spec = RunSpec(
            variant=variant,
            regime=args.regime,
            u_size=cap,
            budgets=(lmax,),
            draws=(args.draw,),
            seeds=(args.seed,),
            f_u=f_u,
            f_l=f_l,
        )
        x, y, label, n = _u_pool_for_spec(
            spec, loaded.u_arrays, loaded.u_fit_rows, loaded.tbl, layers
        )
        return x, y, label, n, {"matched_pool_cap": int(cap), "f_u": f_u, "f_l": f_l}
    if condition == "jobd_full":
        spec = RunSpec(
            variant=variant,
            regime=args.regime,
            u_size=None,
            budgets=(lmax,),
            draws=(args.draw,),
            seeds=(args.seed,),
            f_u=None,
            f_l=None,
        )
        x, y, label, n = _u_pool_for_spec(
            spec, loaded.u_arrays, loaded.u_fit_rows, loaded.tbl, layers
        )
        return x, y, label, n, {"f_u": None, "f_l": None}
    if condition == "add":
        # Two-source ADD: full generic pool UNION all trait-eliciting TRAIN
        # pairs. No E1 extraction rows (that is the union_all arm, R5's lane).
        rows = loaded.u_fit_rows
        x_gen = np.stack([loaded.u_arrays[(variant, ly)][rows] for ly in layers])
        y_gen = np.stack([loaded.u_arrays[("t1", ly)][rows] for ly in layers])
        x_elic = np.asarray(loaded.tbl.z_by_variant[variant], dtype=x_gen.dtype)
        y_elic = np.asarray(loaded.tbl.z_ans, dtype=y_gen.dtype)
        x = np.concatenate([x_gen, x_elic], axis=1)
        del x_gen, x_elic
        y = np.concatenate([y_gen, y_elic], axis=1)
        del y_gen, y_elic
        n = x.shape[1]
        label = f"add{n}_gen{len(rows)}_elic{n_ctx}"
        return (
            x,
            y,
            label,
            n,
            {
                "f_u": None,
                "f_l": None,
                "add_n_generic": int(len(rows)),
                "add_n_eliciting": int(n_ctx),
                "add_realized_pool": int(n),
                "add_sources": ["u_store_fit_pool", "trait_eliciting_train_pairs"],
                "add_excluded_sources": ["e1_extraction_pairs (union_all — R5 lane)"],
            },
        )
    raise ValueError(f"unknown map condition: {condition}")


def fit_pool_map(args, x, y):
    """Whitening + linear map on a prebuilt pool (fp16 in, reviewed path)."""
    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_fits import _fit_map

    t0 = time.time()
    wh = fits.fit_whitening(x, device=args.device, seed=args.seed)
    x_w = fits.apply_whitening(x, wh)
    y_w = fits.apply_whitening(y, wh)
    mapfit = _fit_map(_fitmap_ns(args), x_w, y_w)
    del x_w, y_w
    _free_cuda(args.device)
    diag = {
        **mapfit.diagnostics,
        "map_source": "refit",
        "map_fit_s": round(time.time() - t0, 1),
    }
    return wh, mapfit, diag


# ---------------------------------------------------------------------------
# transfer scoring helpers
# ---------------------------------------------------------------------------


def transfer_rows_for(
    data,
    cell,
    z_ev,
    dv_ev,
    za_ev,
    rungs_ev,
    frozen: dict[str, int],
    prov: dict,
    layers: list[int],
    roster: tuple[str, ...],
    *,
    device: str,
    n_boot: int | None,
    min_n: int,
    ridge_folds: tuple[int, ...] | None = (0,),
) -> tuple[list[dict], list[dict], dict]:
    """One run_transfer_cell + evaluate_transfer pass; returns rows, skips, scores.

    ``ridge_folds=(0,)`` skips the discarded train-block fold fit (the cheap
    default); r2aug passes ``None`` because arm10_stacked requires ridge preds
    on EVERY fold (the armfill_round3 precedent's `--ridge-folds all`).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    kwargs = {"n_boot": n_boot} if n_boot else {}
    scores_ev, arm_skips = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        np.asarray(dv_ev, dtype=np.float64),
        za_ev=za_ev,
        arms=list(roster),
        device=device,
        ridge_folds=ridge_folds,
    )
    rows, skips = arms.evaluate_transfer(
        scores_ev,
        dv_ev,
        np.asarray(rungs_ev),
        frozen,
        provenance=prov,
        cell=cell,
        layers=tuple(layers),
        min_n=min_n,
        **kwargs,
    )
    skips += [
        {"arm": s, "reason": r, "variant": prov.get("variant")}
        for s, r in sorted(arm_skips.items())
    ]
    skips += arms.roster_accounting_skips(
        list(roster), scores_ev, arm_skips, variant=prov.get("variant", "?")
    )
    return rows, skips, scores_ev


def per_layer_rows_for(scores_ev, dv_ev, frozen, prov, layers, src) -> list[dict]:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    dv = np.asarray(dv_ev, dtype=np.float64)
    out = []
    for slug, sc in sorted(scores_ev.items()):
        rhos = [float(x) for x in arms.spearman_rows(np.asarray(sc, dtype=np.float64), dv)]
        out.append(
            {
                **prov,
                "arm": slug,
                "family": arms.ARM_REGISTRY.get(slug, {}).get("family", "unknown"),
                "rung_kind": "eval_transfer_per_layer",
                "layers": [int(x) for x in layers],
                "rho_per_layer": rhos,
                "frozen_layer_idx": int(frozen[slug]),
                "frozen_layer": int(layers[int(frozen[slug])]),
                "frozen_source": src,
                "n_eval": int(dv.size),
            }
        )
    return out


# ---------------------------------------------------------------------------
# mode: r2aug
# ---------------------------------------------------------------------------


def run_r2aug(args, loaded: Loaded, behavior: str, layers: list[int]) -> dict:
    """Map-condition comparison (swap / generic_matched / add) at the plotted slice."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _eval_rung_reconstruction

    lmax = LMAX[behavior]
    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    map_diag: dict[str, dict] = {}
    frozen_src: dict[str, str] = {}
    for variant in args.variants:
        frozen, src = committed_frozen(args, loaded, behavior, variant, layers, ROSTER_R2AUG)
        frozen_src[variant] = src
        for condition in args.map_conditions:
            x, y, u_label, n_u, pool_meta = build_pool(args, loaded, variant, layers, condition)
            wh, mapfit, diag = fit_pool_map(args, x, y)
            del x, y
            diag.update({"n_u": int(n_u), "u_pool_label": u_label, **pool_meta})
            cell = fits.realize_budget_cell(
                loaded.tbl.groups, budget_l=lmax, draw=args.draw, seed=args.seed
            )
            data = arms.CellData(
                z_ctx=fits.apply_whitening(loaded.tbl.z_by_variant[variant], wh),
                z_ans=fits.apply_whitening(loaded.tbl.z_ans, wh),
                dv=loaded.tbl.dv,
                rb=np.einsum("ld,lde->le", loaded.rb, wh.w),
                mapfit=mapfit,
                layers=tuple(layers),
            )
            prov = {
                "mode": "r2aug",
                "map_condition": condition,
                "behavior": behavior,
                "variant": variant,
                "regime": args.regime,
                "u_rung": int(n_u),
                "u_rung_label": u_label,
                "config": "config_a",
                "budget_l": lmax,
                **{k: pool_meta[k] for k in ("f_u", "f_l") if k in pool_meta},
            }
            # (a) eval-split rungs (train-distribution + OOD)
            z_ev = fits.apply_whitening(loaded.tbl_ev.z_by_variant[variant], wh)
            za_ev = fits.apply_whitening(loaded.tbl_ev.z_ans, wh)
            diag["eval_rung"] = _eval_rung_reconstruction(
                mapfit, z_ev, za_ev, rungs=loaded.tbl_ev.row_rungs, knn=True
            )
            rows, skips, scores = transfer_rows_for(
                data,
                cell,
                z_ev,
                loaded.tbl_ev.dv,
                za_ev,
                loaded.tbl_ev.row_rungs,
                frozen,
                prov,
                layers,
                ROSTER_R2AUG,
                device=args.device,
                n_boot=args.n_boot,
                min_n=args.min_n,
            )
            per_layer_all += per_layer_rows_for(
                scores,
                loaded.tbl_ev.dv,
                frozen,
                {**prov, "eval_rung": "pooled_eval_split"},
                layers,
                src,
            )
            del z_ev, za_ev, scores
            # (b) the WildChat rung
            z_wc = fits.apply_whitening(loaded.tbl_wc.z_by_variant[variant], wh)
            za_wc = fits.apply_whitening(loaded.tbl_wc.z_ans, wh)
            diag["wildchat_rung_recon"] = _eval_rung_reconstruction(
                mapfit, z_wc, za_wc, rungs=loaded.tbl_wc.row_rungs, knn=True
            )
            rows_wc, skips_wc, scores_wc = transfer_rows_for(
                data,
                cell,
                z_wc,
                loaded.tbl_wc.dv,
                za_wc,
                loaded.tbl_wc.row_rungs,
                frozen,
                prov,
                layers,
                ROSTER_R2AUG,
                device=args.device,
                n_boot=args.n_boot,
                min_n=args.min_n,
            )
            per_layer_all += per_layer_rows_for(
                scores_wc,
                loaded.tbl_wc.dv,
                frozen,
                {**prov, "eval_rung": RUNG},
                layers,
                src,
            )
            # (c) train in-split anchor: pooled OOF over the cell's own group
            # folds (arms.run_cell — the main grid's in-distribution read), so
            # every map condition carries the train setting too. Scores align
            # with cell.row_idx (the own_pool_frozen_layers contract).
            scores_tr, tr_skips = arms.run_cell(
                data, cell, arms=list(ROSTER_R2AUG), device=args.device
            )
            dv_cell = np.asarray(loaded.tbl.dv, dtype=np.float64)[cell.row_idx]
            rows_tr, skips_tr = arms.evaluate_transfer(
                scores_tr,
                dv_cell,
                np.asarray(["train_in_split"] * len(cell.row_idx)),
                frozen,
                provenance={**prov, "rung_kind_note": "in_split_oof"},
                cell=cell,
                layers=tuple(layers),
                min_n=args.min_n,
                **({"n_boot": args.n_boot} if args.n_boot else {}),
            )
            skips_all += skips_tr + [
                {"arm": s, "reason": f"train in-split: {r}", "variant": variant}
                for s, r in sorted(tr_skips.items())
            ]
            rows_all += rows + rows_wc + rows_tr
            skips_all += skips + skips_wc
            map_diag[f"{variant}|{condition}|{u_label}"] = diag
            print(
                f"[r2aug] {behavior}/{variant} condition={condition}: pool={u_label} "
                f"rows={len(rows) + len(rows_wc) + len(rows_tr)}",
                flush=True,
            )
            del data, z_wc, za_wc, scores_wc, wh, mapfit
            _free_cuda(args.device)
    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "map_diagnostics": map_diag,
        "frozen_source": frozen_src,
        "budget_l": lmax,
        "map_conditions": list(args.map_conditions),
    }


# ---------------------------------------------------------------------------
# mode: jobd
# ---------------------------------------------------------------------------


def _wc_folds(n_wc: int, k: int, seed: int):
    """Deterministic K-fold partition of the judged WildChat contexts."""
    import numpy as np

    rng = np.random.default_rng([1739, 77, int(seed)])
    perm = rng.permutation(n_wc)
    return [np.sort(perm[f::k]) for f in range(k)]


def _mixture_rows(
    loaded: Loaded, n_elic: int, n_gen: int, gen_pool, *, elic_seed: int, gen_seed: int, g: float
):
    """(elic_rows_into_train_tbl, gen_rows_into_wc_tbl) — seeded, group-respecting.

    Separate seeds so the ELICITING subset stays FIXED across the WC cross-fit
    folds (only the generic rows rotate with the fold's out-of-fold pool) —
    fold-to-fold variance then reflects the contamination control only.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    if n_elic > 0:
        elic = fits.realize_budget_cell(
            loaded.tbl.groups, budget_l=n_elic, draw=0, seed=elic_seed
        ).row_idx
    else:
        elic = np.empty(0, dtype=np.int64)
    if n_gen > 0:
        gen_pool = np.asarray(gen_pool)
        if n_gen > len(gen_pool):
            raise ValueError(f"jobd: need {n_gen} generic rows, pool has {len(gen_pool)}")
        rng = np.random.default_rng([1739, 78, int(gen_seed), round(g * 1000)])
        gen = np.sort(rng.choice(gen_pool, size=n_gen, replace=False))
    else:
        gen = np.empty(0, dtype=np.int64)
    return elic, gen


def _pool_zscored_dv(dv_m, elic_rows, gen_rows):
    """Merged-DV copy with per-pool z-scored TRAINING targets (jobd fix).

    Stats are computed over THIS fit's own selected rows, per pool (trait rows
    by trait-row stats, judged-generic rows by generic-row stats), so the two
    DV constructs enter the ridge loss on a common scale. Rows outside the
    selection are untouched (they never enter the fit). sd == 0 degrades to a
    centered constant column rather than dividing by zero.
    """
    import numpy as np

    dv = np.asarray(dv_m, dtype=np.float64).copy()
    for rows in (np.asarray(elic_rows, dtype=np.int64), np.asarray(gen_rows, dtype=np.int64)):
        if rows.size == 0:
            continue
        m = float(dv[rows].mean())
        s = float(dv[rows].std())
        dv[rows] = (dv[rows] - m) / (s if s > 0 else 1.0)
    return dv


def run_jobd(args, loaded: Loaded, behavior: str, layers: list[int]) -> dict:
    """Judged-generic readout swap at fixed L, map + whitening FIXED (full pool)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    n_tr = len(loaded.tbl.ctx_order)
    n_wc = len(loaded.tbl_wc.ctx_order)
    folds = _wc_folds(n_wc, JOBD_K_FOLDS, args.seed)
    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []
    map_diag: dict[str, dict] = {}
    frozen_src: dict[str, str] = {}
    for variant in args.variants:
        frozen, src = committed_frozen(args, loaded, behavior, variant, layers, ROSTER_JOBD)
        frozen_src[variant] = src
        x, y, u_label, n_u, pool_meta = build_pool(args, loaded, variant, layers, "jobd_full")
        wh, mapfit, diag = fit_pool_map(args, x, y)
        del x, y
        diag.update({"n_u": int(n_u), "u_pool_label": u_label, **pool_meta})
        map_diag[f"{variant}|{u_label}"] = diag
        # Merged labeled table: train rows [0, n_tr) then wc rows [n_tr, n_tr+n_wc).
        z_ctx = np.concatenate(
            [
                fits.apply_whitening(loaded.tbl.z_by_variant[variant], wh),
                fits.apply_whitening(loaded.tbl_wc.z_by_variant[variant], wh),
            ],
            axis=1,
        )
        z_ans = np.concatenate(
            [
                fits.apply_whitening(loaded.tbl.z_ans, wh),
                fits.apply_whitening(loaded.tbl_wc.z_ans, wh),
            ],
            axis=1,
        )
        dv_m = np.concatenate(
            [
                np.asarray(loaded.tbl.dv, dtype=np.float64),
                np.asarray(loaded.tbl_wc.dv, dtype=np.float64),
            ]
        )
        data = arms.CellData(
            z_ctx=z_ctx,
            z_ans=z_ans,
            dv=dv_m,
            rb=np.einsum("ld,lde->le", loaded.rb, wh.w),
            mapfit=mapfit,
            layers=tuple(layers),
        )
        z_ev = fits.apply_whitening(loaded.tbl_ev.z_by_variant[variant], wh)
        za_ev = fits.apply_whitening(loaded.tbl_ev.z_ans, wh)
        wc_all = np.arange(n_wc)
        for g in JOBD_G_GRID:
            n_gen = round(g * JOBD_L_TOTAL)
            n_elic = JOBD_L_TOTAL - n_gen
            prov = {
                "mode": "jobd_swap",
                "behavior": behavior,
                "variant": variant,
                "regime": args.regime,
                "u_rung": int(n_u),
                "u_rung_label": u_label,
                "config": "config_a",
                "f_u": None,
                "f_l": None,
                "g_generic": g,
                "n_gen": int(n_gen),
                "n_elic": int(n_elic),
                "l_total": JOBD_L_TOTAL,
                "gen_pool": "wildchat_rung_judged",
                "dv_scaling": "per_pool_zscore_train_targets_v1",
                "budget_l": JOBD_L_TOTAL,
            }
            # (a) eval-split rungs: ONE full-mixture fit (generic rows from the
            # whole judged pool — eval rungs are disjoint from it by design).
            elic, gen = _mixture_rows(
                loaded, n_elic, n_gen, wc_all, elic_seed=args.seed, gen_seed=args.seed, g=g
            )
            cell = fits.BudgetCell(
                row_idx=np.concatenate([elic, n_tr + gen]).astype(np.int64),
                fold_ids=np.zeros(n_elic + n_gen, dtype=np.int64),
                n_folds=1,
                budget_l=JOBD_L_TOTAL,
                draw=args.draw,
                seed=args.seed,
                fold_scheme="jobd_mixture_full",
            )
            data_g = dataclasses.replace(data, dv=_pool_zscored_dv(dv_m, elic, n_tr + gen))
            rows, skips, scores = transfer_rows_for(
                data_g,
                cell,
                z_ev,
                loaded.tbl_ev.dv,
                za_ev,
                loaded.tbl_ev.row_rungs,
                frozen,
                {**prov, "wc_cross_fit": False},
                layers,
                ROSTER_JOBD,
                device=args.device,
                n_boot=args.n_boot,
                min_n=args.min_n,
            )
            per_layer_all += per_layer_rows_for(
                scores,
                loaded.tbl_ev.dv,
                frozen,
                {**prov, "eval_rung": "pooled_eval_split", "wc_cross_fit": False},
                layers,
                src,
            )
            rows_all += rows
            skips_all += skips
            del scores
            # (b) WildChat rung: K-fold cross-fit — generic training rows come
            # from OTHER folds only; stitched per-context scores over all folds.
            stitched: dict[str, np.ndarray] = {}
            for f_i, fold in enumerate(folds):
                in_fold = np.zeros(n_wc, dtype=bool)
                in_fold[fold] = True
                pool_f = np.flatnonzero(~in_fold)
                elic_f, gen_f = _mixture_rows(
                    loaded,
                    n_elic,
                    n_gen,
                    pool_f,
                    elic_seed=args.seed,
                    gen_seed=args.seed + 1000 + f_i,
                    g=g,
                )
                cell_f = fits.BudgetCell(
                    row_idx=np.concatenate([elic_f, n_tr + gen_f]).astype(np.int64),
                    fold_ids=np.zeros(n_elic + n_gen, dtype=np.int64),
                    n_folds=1,
                    budget_l=JOBD_L_TOTAL,
                    draw=args.draw,
                    seed=args.seed,
                    fold_scheme=f"jobd_mixture_wcfold{f_i}",
                )
                data_f = dataclasses.replace(data, dv=_pool_zscored_dv(dv_m, elic_f, n_tr + gen_f))
                sc_f, arm_skips_f = arms.run_transfer_cell(
                    data_f,
                    cell_f,
                    np.ascontiguousarray(z_ctx[:, n_tr + fold]),
                    np.asarray(loaded.tbl_wc.dv, dtype=np.float64)[fold],
                    za_ev=np.ascontiguousarray(z_ans[:, n_tr + fold]),
                    arms=list(ROSTER_JOBD),
                    device=args.device,
                    ridge_folds=(0,),
                )
                for slug, sc in sc_f.items():
                    buf = stitched.setdefault(
                        slug, np.full((sc.shape[0], n_wc), np.nan, dtype=np.float64)
                    )
                    buf[:, fold] = sc
                skips_all += [
                    {
                        "arm": s,
                        "reason": f"wc fold {f_i}: {r}",
                        "variant": variant,
                        "g_generic": g,
                    }
                    for s, r in sorted(arm_skips_f.items())
                ]
            prov_wc = {**prov, "wc_cross_fit": True, "wc_k_folds": JOBD_K_FOLDS}
            kwargs = {"n_boot": args.n_boot} if args.n_boot else {}
            rows_wc, skips_wc = arms.evaluate_transfer(
                stitched,
                loaded.tbl_wc.dv,
                np.asarray(loaded.tbl_wc.row_rungs),
                frozen,
                provenance=prov_wc,
                cell=cell,
                layers=tuple(layers),
                min_n=args.min_n,
                **kwargs,
            )
            per_layer_all += per_layer_rows_for(
                stitched,
                loaded.tbl_wc.dv,
                frozen,
                {**prov_wc, "eval_rung": RUNG},
                layers,
                src,
            )
            rows_all += rows_wc
            skips_all += skips_wc
            print(
                f"[jobd] {behavior}/{variant} g={g}: n_gen={n_gen} n_elic={n_elic} "
                f"rows={len(rows) + len(rows_wc)}",
                flush=True,
            )
        del data, z_ctx, z_ans, z_ev, za_ev, wh, mapfit
        _free_cuda(args.device)
    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "map_diagnostics": map_diag,
        "frozen_source": frozen_src,
        "l_total": JOBD_L_TOTAL,
        "k_folds": JOBD_K_FOLDS,
        "g_grid": list(JOBD_G_GRID),
    }


# ---------------------------------------------------------------------------
# VM-side collect: committed generic-only reference rows (no re-run)
# ---------------------------------------------------------------------------


def collect_generic(args) -> int:
    """Extract + slice-verify committed generic-only rows (``map_condition:
    generic``), written per behavior into the r2aug out-root.

    Sources (the Result-2 figure's own artifact map, context_end at the
    plotted slice): main train arm_rows + transfer rows, wide_ood transfer
    jsonl, and the wide/wildchat_rung roll-up. Rows are copied verbatim plus
    provenance fields; a source missing an arm at the slice is visible in
    ``rows_per_arm`` — never imputed.
    """
    er = args.main_root
    out_root = args.out_root_r2aug
    picked_arms = set(ROSTER_R2AUG)
    report: dict[str, dict] = {}
    for behavior in args.behaviors:
        lmax = LMAX[behavior]
        rows: list[dict] = []
        read: list[str] = []

        def _keep(r: dict, source: str, lmax: int = lmax, rows: list = rows) -> None:
            if r.get("regime") != "e1" or str(r.get("u_rung_label")) != "full":
                return
            if r.get("variant") != "context_end" or r.get("arm") not in picked_arms:
                return
            if source != "wide_wc" and r.get("budget_l") is not None and r.get("budget_l") != lmax:
                return
            rows.append({**r, "map_condition": "generic", "generic_reuse_source": source})

        main_p = er / behavior / "arm_results" / "all_arms_spearman.json"
        if main_p.exists():
            d = json.loads(main_p.read_text())
            for r in d.get("arm_rows") or []:
                if r.get("f_u") is None:
                    _keep(r, "main_train")
            for r in d.get("transfer_rows") or []:
                if r.get("f_u") is None:
                    _keep(r, "main_transfer")
            read.append(str(main_p))
        wide_ood = er / "wide_ood" / f"{behavior}_transfer.jsonl"
        if wide_ood.exists():
            with wide_ood.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    for r in obj["rows"] if isinstance(obj, dict) and "rows" in obj else [obj]:
                        if r.get("f_u") is None:
                            _keep(r, "wide_ood")
            read.append(str(wide_ood))
        wide_wc = er / "wide" / "wildchat_rung" / behavior / "all_arms_spearman.json"
        if wide_wc.exists():
            d = json.loads(wide_wc.read_text())
            for r in d.get("transfer_rows") or []:
                if r.get("f_u") is None:
                    _keep(r, "wide_wc")
            read.append(str(wide_wc))
        # Job C gap-fill (arms 7/8/12/17/18 at the max-budget slice) — the
        # source the hallucination arm7/8 OOD cells actually live in (the
        # figure reads the MERGED root only; legs/ repeats the same rows).
        gapfill = (
            er / "result2_gapfill" / "merged" / behavior / "arm_results" / "all_arms_spearman.json"
        )
        if gapfill.exists():
            d = json.loads(gapfill.read_text())
            for r in (d.get("transfer_rows") or []) + (d.get("arm_rows") or []):
                if r.get("f_u") is None:
                    _keep(r, "gapfill_merged")
            read.append(str(gapfill))

        out_dir = out_root / behavior
        out_dir.mkdir(parents=True, exist_ok=True)
        by_arm: dict[str, int] = {}
        for r in rows:
            by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
        payload = {
            "map_condition": "generic",
            "behavior": behavior,
            "note": (
                "committed generic-only rows reused as the reference condition (addendum "
                "instruction: do not re-run) — filtered at the plotted slice regime=e1, "
                "u_rung_label=full, variant=context_end, budget_l=%d (wide_wc carries its "
                "own single budget)" % lmax
            ),
            "variant_scope": VARIANT_SCOPE_NOTE,
            "sources_read": read,
            "n_rows": len(rows),
            "rows_per_arm": by_arm,
            "rows": rows,
        }
        (out_dir / "generic_reused_rows.json").write_text(json.dumps(payload, indent=1))
        report[behavior] = {"n_rows": len(rows), "per_arm": by_arm, "sources": read}
        print(
            f"[collect-generic] {behavior}: {len(rows)} rows from {len(read)} sources",
            flush=True,
        )
    print(json.dumps(report, indent=1), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--modes", nargs="+", default=["r2aug", "jobd"], choices=["r2aug", "jobd"])
    ap.add_argument(
        "--map-conditions",
        nargs="+",
        default=["swap", "add"],
        choices=["swap", "add", "generic_matched"],
        help="r2aug map conditions to run on this invocation (the committed generic-only "
        "reference is collected VM-side via --collect-generic-only, never re-run)",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["context_end"],
        choices=["context_end", "prefix_end"],
        help="DEFAULT context_end ONLY (user directive 2026-08-05 — recorded deviation "
        "from the both-arms mapping rule)",
    )
    ap.add_argument("--regime", default="e1", choices=("e1",))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rb-source", default="auto", choices=("auto", "bank", "extract"))
    ap.add_argument("--out-root-jobd", type=Path, default=DEFAULT_OUT_JOBD)
    ap.add_argument("--out-root-r2aug", type=Path, default=DEFAULT_OUT_R2AUG)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--u-store", type=Path, default=None)
    ap.add_argument("--train-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-store", type=Path, default=None)
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument(
        "--collect-generic-only",
        action="store_true",
        help="VM-side: extract + slice-verify committed generic-only reference rows, then exit",
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.store_root / "train_dv"
    if args.wcrung_dv_root is None:
        args.wcrung_dv_root = args.main_root / "wildchat_rung" / "dv_dataset"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules, _verify_input_shas

    _assert_no_judge_modules("at entry")
    if args.collect_generic_only:
        rc = collect_generic(args)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(rc)
    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
            arms,
            fits,
            store_io,
        )
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_fits import (  # noqa: F401
            RunSpec,
            _eval_rung_reconstruction,
            _fit_map,
            _git_commit,
            _load_labeled,
            _u_pool_for_spec,
        )
        from scripts.issue1739_wcrung_arms import (  # noqa: F401
            _rb_for_behavior,
            modal_frozen_layers,
            resolve_wcrung_store,
        )

        _assert_no_judge_modules("after --import-check imports")
        print("[jobd-r2aug] import-check OK", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.experiments.issue_1739 import arms
    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_fits import _git_commit
    from scripts.issue1739_wcrung_arms import _git_tracked

    load_dotenv()
    out_root_by_mode = {"jobd": args.out_root_jobd, "r2aug": args.out_root_r2aug}
    for mode in args.modes:
        for b in args.behaviors:
            out = out_root_by_mode[mode] / b / "all_arms_spearman.json"
            if _git_tracked(out) and not args.allow_overwrite_committed:
                raise SystemExit(f"refusing to overwrite git-TRACKED output: {out}")

    layers = args.layers or list(range(args.n_layers))
    commit = _git_commit()
    env = _env_versions()
    runners = {"jobd": run_jobd, "r2aug": run_r2aug}
    failures: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        try:
            loaded = load_behavior(args, behavior, layers)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[jobd-r2aug] {behavior} LOAD FAILED: {exc}", flush=True)
            continue
        for mode in args.modes:
            t0 = time.time()
            try:
                res = runners[mode](args, loaded, behavior, layers)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                failures.append(
                    {"behavior": behavior, "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[jobd-r2aug] {behavior}/{mode} FAILED: {exc}", flush=True)
                continue
            out_dir = out_root_by_mode[mode] / behavior
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if mode == "jobd" else "." + "_".join(sorted(args.map_conditions))
            out_path = out_dir / f"all_arms_spearman{suffix}.json"
            arms.write_summary(
                [],
                out_path,
                meta={
                    "mode": mode,
                    "behavior": behavior,
                    "config": "config_a",
                    "regimes": [args.regime],
                    "variants": list(args.variants),
                    "variant_scope": VARIANT_SCOPE_NOTE,
                    "arms": sorted(ROSTER_JOBD if mode == "jobd" else ROSTER_R2AUG),
                    "label_consuming_arms": sorted(LABEL_CONSUMING),
                    "map_conditions": None if mode == "jobd" else list(args.map_conditions),
                    "context_direct_arms_policy": (
                        "label-free arms recomputed as internal consistency references "
                        "(map+whitening fixed in jobd, so they are unchanged by construction)"
                        if mode == "jobd"
                        else "context-direct arms NOT re-fit (they do not read through the "
                        "map); committed rows are the reference lines — see "
                        "generic_reused_rows.json. arms 9/10 SKIPPED (user-confirmed): "
                        "excluded from the transfer machinery by design (arms.py "
                        "TRANSFER_ARMS_WIDE note); the ridge_folds=None forcing precedent "
                        "was deliberately not used. They appear under the generic map "
                        "condition only — label the gap in the figure, never a zero bar"
                    ),
                    "n_train_contexts": len(loaded.tbl.ctx_order),
                    "n_eval_contexts": len(loaded.tbl_ev.ctx_order),
                    "n_wildchat_contexts": len(loaded.tbl_wc.ctx_order),
                    "eval_rungs": sorted(set(loaded.tbl_ev.rungs) | {RUNG}),
                    "map_kind": "linear",
                    "map_source": "refit-in-process",
                    "frozen_layer_source": res["frozen_source"],
                    "transfer_min_n": int(args.min_n),
                    "rb": loaded.rb_meta,
                    "wildchat_single_unit_caveat": (
                        "the wildchat_rung column is SINGLE-UNIT by design (1 replicate "
                        "per condition — draw 0 / seed 0 — matched across generic/swap/add "
                        "and across g); cross-condition differences there carry NO "
                        "replicate-level uncertainty and must not be read as a measured "
                        "effect. Any ci_frozen on a wildchat row is a paired bootstrap "
                        "over EVAL CONTEXTS within that one unit (within-draw), never an "
                        "across-replicate interval — label it as such wherever rendered"
                    ),
                    "jobd_dv_scaling": (
                        "per_pool_zscore_train_targets_v1 — training targets z-scored per "
                        "pool over each fit's selected rows (mixed-scale DV fix: hall train "
                        "DV is [0,1] fabricated-fraction vs 0-100 trait rubric on the judged-"
                        "generic rows); evaluation DVs raw everywhere. evil's judged-generic "
                        "DV is near-degenerate (mean 0.42, sd 4.4) — a finding, disclosed"
                        if mode == "jobd"
                        else None
                    ),
                    "contamination_handling": (
                        "cross-fit (option c): K-fold over the judged WildChat contexts; "
                        "every context is evaluated by a readout whose generic rows came "
                        "from other folds only"
                        if mode == "jobd"
                        else "n/a — the r2aug map pool consumes (context, answer) PAIRS "
                        "only, never behavior judgments"
                    ),
                    "dv_construct_caveat": (
                        "hallucination TRAIN DV is the fabricated-fraction construct while "
                        "the WildChat-rung DV is the graded trait rubric — hallucination "
                        "jobd mixtures train on a mixed-construct target (disclosed)"
                        if behavior == "hallucination"
                        else "train and WildChat DVs are both graded 0-100 trait-rubric constructs"
                    ),
                    "swap_semantics": (
                        "SWAP at fixed total budget (never ADD) for jobd (L=%d fixed, "
                        "judged-generic fraction varies) and for the r2aug swap/"
                        "generic_matched conditions (matched pool cap); the r2aug add "
                        "condition is deliberately ADD (union, larger pool — a size-vs-"
                        "composition contrast against the swap rung)" % JOBD_L_TOTAL
                    ),
                    **{
                        k: v
                        for k, v in res.items()
                        if k in ("budget_l", "l_total", "k_folds", "g_grid", "map_conditions")
                    },
                    "input_paths": {k: str(v) for k, v in loaded.paths.items()},
                    "input_sha256": loaded.shas,
                    "git_commit": commit,
                    "env_versions": env,
                    "wall_s": round(time.time() - t0, 1),
                    "judge_called": False,
                },
                extra={
                    "transfer_rows": res["rows"],
                    "transfer_skips": res["skips"],
                    "per_layer_rows": res["per_layer"],
                    "n_transfer_rows": len(res["rows"]),
                    "n_per_layer_rows": len(res["per_layer"]),
                },
            )
            (out_dir / f"map_diagnostics{suffix}.json").write_text(
                json.dumps(res["map_diagnostics"], indent=1)
            )
            print(
                f"[jobd-r2aug] {behavior}/{mode} done: {len(res['rows'])} transfer rows in "
                f"{time.time() - t0:.0f}s -> {out_path}",
                flush=True,
            )
        _verify_input_shas(loaded.shas)
        del loaded
        _free_cuda(args.device)

    _assert_no_judge_modules("at exit")
    print(f"[jobd-r2aug] all done in {time.time() - t_all:.0f}s", flush=True)
    if failures:
        for mode in args.modes:
            out_root_by_mode[mode].mkdir(parents=True, exist_ok=True)
        (out_root_by_mode[args.modes[0]] / "jobd_r2aug_failures.json").write_text(
            json.dumps(failures, indent=1)
        )
        for f in failures:
            print(f"[jobd-r2aug] FAILED {f}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
