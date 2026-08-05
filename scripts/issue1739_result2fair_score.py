#!/usr/bin/env python3
"""#1739 Result-2 FAIR-PROTOCOL refit: one readout training set for every method.

User-directed same-issue follow-up (2026-08-05). The committed Result-2 rows
train the label-consuming readout on the trait-eliciting train set only; this
round re-fits it on ALL the judged training data and re-scores the four
Result-2 methods under one shared protocol:

  every predictor gets
    - the generic UNJUDGED WildChat pool  -> the context->answer MAP fit pool
      (the ADD/union condition: generic pool + trait-eliciting train pairs —
      the committed ``result2_trait_aug`` ``map_condition == "add"`` recipe,
      re-fit in-process from the same pool; map weights are never persisted,
      so re-running the reviewed deterministic recipe IS the reuse)
    - a JUDGED WildChat train split       -> readout training rows (NEW)
    - the trait-eliciting train set       -> readout training rows (as before;
      the train setting reads it out-of-fold under 5 group-level folds)
    - the synthetic persona-vectors judged extraction set -> r_B (the three
      projection arms already consume it through the E1 direction; the
      REGRESSION arm deliberately does NOT train on pvsynth rows — that would
      cannibalise the pvsynth evaluation setting. Recorded protocol deviation.)

  and is evaluated on four setting roles
    - pvsynth              the persona-vectors eval grid (200 held-out contexts)
    - wildchat_rung        a FIXED held-out WildChat eval split (sha1 bucket;
                           the readout never trains on these contexts)
    - train                out-of-fold over the eliciting train rows
    - OOD rungs            the behaviour's committed OOD rungs (hhrt/toxicchat,
                           aita, nqopen/simpleqa)

METHODS (4): arm1_ctx_e1 (PV on context), arm6_map_proj_e1 (PV on mapped
answer, LINEAR map), arm11_oracle_proj (PV on real answer),
arm4_ridge_ctx (regression context -> behaviour expression). Only arm4
consumes DV labels at fit time; arms 1/6/11 are pure projections, re-scored
here so every bar shares the ADD-pool whitening + the same eval subsets.

NO LEAKAGE is the point: hard asserts that no context in the readout training
set appears in any evaluation setting (the train setting is fold-disjoint by
the shared group-fold machinery). Realized counts are recorded in meta.

Structural sibling of ``issue1739_jobd_r2aug.py`` — every scoring primitive is
IMPORTED from the reviewed production modules (no fit, metric, or fold logic
of its own). Safety rails inherited: no judge module may be imported, DV
inputs sha-verified after scoring, git-tracked outputs refused.

VARIANT SCOPE: context_end ONLY (user directive 2026-08-05 — the recorded
deviation from the prefix+context both-arms rule; basis: R5 measured the
prefix-end map at chance retrieval with negative R^2 under both compositions).

DV scaling: readout TRAINING targets are z-scored per pool over the fit's own
selected rows (trait rows by trait-row stats, judged-WildChat rows by
WildChat-row stats — the jobd mixed-construct fix); evaluation DVs stay RAW
everywhere (Spearman is monotone-invariant on the eval side).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_jobd_r2aug.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

BEHAVIORS = ("evil", "sycophancy", "hallucination")
WC_RUNG = "wildchat_rung"
PV_RUNG = "pvsynth"

# One fixed WildChat eval split, shared by all behaviors (the wcrung context
# ids are the same 2000-conversation pool judged three ways): sha1(ctx_id)
# mod 5 == 4 -> eval (~20%), the corpus_staging SYC_PARTITION_MOD pattern.
WC_SPLIT_MOD = 5
WC_EVAL_BUCKET = 4

ROSTER_FAIR = (
    "arm1_ctx_e1",
    "arm4_ridge_ctx",
    "arm6_map_proj_e1",
    "arm11_oracle_proj",
)
LABEL_CONSUMING = ("arm4_ridge_ctx",)

DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/result2_fair")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")

PVSYNTH_READOUT_DEVIATION = (
    "protocol item 4 (pvsynth judged data) is satisfied for the three projection arms "
    "through r_B — the E1 direction is built from the judge-filtered pvsynth extraction "
    "set — but the REGRESSION arm does NOT train on pvsynth rows: the only judged pvsynth "
    "rows are the 200 split=eval grid contexts that ARE the pvsynth evaluation setting "
    "(no judged pvsynth train rows exist), so training on them would evaluate the readout "
    "on its own training contexts. Recorded protocol deviation."
)


def _wc_eval_mask(ctx_ids: list[str]):
    import numpy as np

    return np.array(
        [
            int(hashlib.sha1(str(c).encode()).hexdigest(), 16) % WC_SPLIT_MOD == WC_EVAL_BUCKET
            for c in ctx_ids
        ],
        dtype=bool,
    )


def _wc_fold_ids(ctx_ids: list[str], n_folds: int):
    import numpy as np

    return np.array(
        [
            int(hashlib.sha1((str(c) + "|fairfold").encode()).hexdigest(), 16) % n_folds
            for c in ctx_ids
        ],
        dtype=np.int64,
    )


def load_pvsynth(args, behavior: str, layers: list[int], dim: int, shas: dict):
    from scripts.issue1739_fits import _load_labeled
    from scripts.issue1739_wcrung_arms import _sha256

    store = args.pvsynth_store_root / behavior
    dv = args.pvsynth_dv_root / behavior / "labeling.json"
    for p in (store, dv):
        if not p.exists():
            raise FileNotFoundError(f"[{behavior}] pvsynth input missing: {p}")
    shas[str(dv)] = _sha256(dv)
    tbl_pv = _load_labeled(store, dv, layers, config="config_b", need_rollout_rows=False)
    if set(tbl_pv.rungs) != {PV_RUNG}:
        raise RuntimeError(f"[{behavior}] pvsynth DV rungs {tbl_pv.rungs} != {{{PV_RUNG!r}}}")
    if tbl_pv.z_ans.shape[-1] != dim:
        raise RuntimeError(f"[{behavior}] pvsynth hidden dim {tbl_pv.z_ans.shape[-1]} != {dim}")
    return tbl_pv


def leakage_report(loaded, tbl_pv, wc_train_rows, wc_eval_rows, elic_rows) -> dict:
    """HARD leakage asserts + the realized-count record for meta.

    A readout-training context appearing in any evaluation setting is a
    failure, not a caveat. The train setting is fold-disjoint by the shared
    group-fold machinery (asserted separately at cell construction).
    """
    wc_ids = [str(c) for c in loaded.tbl_wc.ctx_order]
    ids_elic = {str(loaded.tbl.ctx_order[i]) for i in elic_rows}
    ids_wc_train = {wc_ids[i] for i in wc_train_rows}
    ids_wc_eval = {wc_ids[i] for i in wc_eval_rows}
    ids_pv = {str(c) for c in tbl_pv.ctx_order}
    ids_ood = {str(c) for c in loaded.tbl_ev.ctx_order}
    readout = ids_elic | ids_wc_train

    assert not ids_wc_train & ids_wc_eval, "WildChat train/eval split overlap"
    assert ids_wc_train | ids_wc_eval == set(wc_ids), "WildChat split does not cover the rung"
    for name, ids_eval in (
        ("wildchat_eval_split", ids_wc_eval),
        ("pvsynth", ids_pv),
        ("ood_rungs", ids_ood),
    ):
        inter = readout & ids_eval
        assert not inter, f"LEAKAGE: {len(inter)} readout-train contexts in {name}: "
    return {
        "n_readout_train_contexts": len(readout),
        "n_readout_eliciting": len(ids_elic),
        "n_readout_wc_train": len(ids_wc_train),
        "n_wc_eval_split": len(ids_wc_eval),
        "n_pvsynth_eval": len(ids_pv),
        "n_ood_eval": len(ids_ood),
        "wc_split": f"sha1(ctx_id) mod {WC_SPLIT_MOD} == {WC_EVAL_BUCKET} -> eval",
        "asserts": (
            "readout-train ctx ids disjoint from wildchat_eval_split, pvsynth, and every "
            "OOD rung (hard assert, passed); wc train/eval split disjoint + covering "
            "(hard assert, passed); train setting is fold-disjoint via the shared "
            "group-level fold machinery"
        ),
    }


def run_fair(args, loaded, tbl_pv, behavior: str, layers: list[int]) -> dict:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _eval_rung_reconstruction
    from scripts.issue1739_jobd_r2aug import (
        LMAX,
        _free_cuda,
        _pool_zscored_dv,
        build_pool,
        committed_frozen,
        fit_pool_map,
        per_layer_rows_for,
        transfer_rows_for,
    )

    lmax = LMAX[behavior]
    variant = args.variant
    rows_all: list[dict] = []
    skips_all: list[dict] = []
    per_layer_all: list[dict] = []

    frozen, src = committed_frozen(args, loaded, behavior, variant, layers, ROSTER_FAIR)

    # --- the ADD/union map + its whitening (the fair MAP condition) ---------
    x, y, u_label, n_u, pool_meta = build_pool(args, loaded, variant, layers, "add")
    wh, mapfit, diag = fit_pool_map(args, x, y)
    del x, y
    diag.update({"n_u": int(n_u), "u_pool_label": u_label, **pool_meta})

    n_tr = len(loaded.tbl.ctx_order)
    n_wc = len(loaded.tbl_wc.ctx_order)
    ev_mask = _wc_eval_mask(loaded.tbl_wc.ctx_order)
    wc_eval_rows = np.flatnonzero(ev_mask)
    wc_train_rows = np.flatnonzero(~ev_mask)

    elic_cell = fits.realize_budget_cell(
        loaded.tbl.groups, budget_l=lmax, draw=args.draw, seed=args.seed
    )
    leak = leakage_report(loaded, tbl_pv, wc_train_rows, wc_eval_rows, elic_cell.row_idx)
    print(f"[fair] {behavior}: leakage asserts PASSED — {json.dumps(leak)}", flush=True)

    # --- merged labeled table: train rows [0, n_tr) then wc rows ------------
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
    readout_rows = np.concatenate([elic_cell.row_idx, n_tr + wc_train_rows]).astype(np.int64)
    dv_z = _pool_zscored_dv(dv_m, elic_cell.row_idx, n_tr + wc_train_rows)
    data = arms.CellData(
        z_ctx=z_ctx,
        z_ans=z_ans,
        dv=dv_z,
        rb=np.einsum("ld,lde->le", loaded.rb, wh.w),
        mapfit=mapfit,
        layers=tuple(layers),
    )
    prov = {
        "mode": "fair",
        "behavior": behavior,
        "variant": variant,
        "regime": args.regime,
        "u_rung": int(n_u),
        "u_rung_label": u_label,
        "config": "config_a",
        "budget_l": lmax,
        "map_condition": "add",
        "readout_train": "union: eliciting train (budget cell) + judged WildChat train split",
        "n_readout_eliciting": int(len(elic_cell.row_idx)),
        "n_readout_wc_train": int(len(wc_train_rows)),
        "wc_split_mod": WC_SPLIT_MOD,
        "wc_eval_bucket": WC_EVAL_BUCKET,
        "dv_scaling": "per_pool_zscore_train_targets_v1",
    }
    kwargs = {"n_boot": args.n_boot} if args.n_boot else {}

    # --- setting A: train, out-of-fold over the eliciting rows --------------
    # WildChat train rows join every fold's TRAINING side via their own
    # hash-assigned folds (a wc row is never evaluated here); the eliciting
    # rows keep the committed group-level folds, so the out-of-fold read is
    # fold-disjoint exactly as in the committed grid.
    wcf = _wc_fold_ids([str(loaded.tbl_wc.ctx_order[i]) for i in wc_train_rows], elic_cell.n_folds)
    cell_oof = fits.BudgetCell(
        row_idx=readout_rows,
        fold_ids=np.concatenate([elic_cell.fold_ids, wcf]),
        n_folds=elic_cell.n_folds,
        budget_l=lmax,
        draw=args.draw,
        seed=args.seed,
        fold_scheme=f"fair-union-{elic_cell.fold_scheme}",
    )
    n_el = len(elic_cell.row_idx)
    assert bool(np.all(cell_oof.row_idx[:n_el] == elic_cell.row_idx))
    scores_tr, tr_skips = arms.run_cell(data, cell_oof, arms=list(ROSTER_FAIR), device=args.device)
    scores_el = {s: np.ascontiguousarray(sc[:, :n_el]) for s, sc in scores_tr.items()}
    dv_el = np.asarray(loaded.tbl.dv, dtype=np.float64)[elic_cell.row_idx]
    rows_tr, skips_tr = arms.evaluate_transfer(
        scores_el,
        dv_el,
        np.asarray(["train"] * n_el),
        frozen,
        provenance={**prov, "rung_kind_note": "in_split_oof_union_readout"},
        cell=cell_oof,
        layers=tuple(layers),
        min_n=args.min_n,
        **kwargs,
    )
    skips_all += skips_tr + [
        {"arm": s, "reason": f"train oof: {r}", "variant": variant}
        for s, r in sorted(tr_skips.items())
    ]
    skips_all += arms.roster_accounting_skips(
        list(ROSTER_FAIR), scores_tr, tr_skips, variant=variant, eval_rung="train"
    )
    per_layer_all += per_layer_rows_for(
        scores_el, dv_el, frozen, {**prov, "eval_rung": "train"}, layers, src
    )
    rows_all += rows_tr
    del scores_tr, scores_el
    print(f"[fair] {behavior}: train OOF done ({len(rows_tr)} rows)", flush=True)

    # --- transfer settings: one full-union fit, frozen predictors -----------
    cell_full = fits.BudgetCell(
        row_idx=readout_rows,
        fold_ids=np.zeros(len(readout_rows), dtype=np.int64),
        n_folds=1,
        budget_l=lmax,
        draw=args.draw,
        seed=args.seed,
        fold_scheme="fair-union-full",
    )

    def _transfer(z_ev, dv_ev, za_ev, rungs, tag: str, extra_prov: dict) -> None:
        nonlocal rows_all, skips_all, per_layer_all
        p = {**prov, **extra_prov}
        rows, skips, scores = transfer_rows_for(
            data,
            cell_full,
            z_ev,
            dv_ev,
            za_ev,
            rungs,
            frozen,
            p,
            layers,
            ROSTER_FAIR,
            device=args.device,
            n_boot=args.n_boot,
            min_n=args.min_n,
        )
        diag[f"recon_{tag}"] = _eval_rung_reconstruction(
            mapfit, z_ev, za_ev, rungs=list(rungs), knn=True
        )
        per_layer_all += per_layer_rows_for(
            scores, dv_ev, frozen, {**p, "eval_rung": tag}, layers, src
        )
        rows_all += rows
        skips_all += skips
        print(f"[fair] {behavior}: transfer {tag} done ({len(rows)} rows)", flush=True)
        del scores

    # (B) WildChat held-out eval split
    _transfer(
        np.ascontiguousarray(z_ctx[:, n_tr + wc_eval_rows]),
        np.asarray(loaded.tbl_wc.dv, dtype=np.float64)[wc_eval_rows],
        np.ascontiguousarray(z_ans[:, n_tr + wc_eval_rows]),
        np.asarray([WC_RUNG] * len(wc_eval_rows)),
        WC_RUNG,
        {"wc_eval_split": True, "n_wc_eval": int(len(wc_eval_rows))},
    )
    # (C) pvsynth grid
    z_pv = fits.apply_whitening(tbl_pv.z_by_variant[variant], wh)
    za_pv = fits.apply_whitening(tbl_pv.z_ans, wh)
    _transfer(
        z_pv,
        np.asarray(tbl_pv.dv, dtype=np.float64),
        za_pv,
        np.asarray(tbl_pv.row_rungs),
        PV_RUNG,
        {},
    )
    del z_pv, za_pv
    # (D) the behaviour's OOD rungs
    z_ev = fits.apply_whitening(loaded.tbl_ev.z_by_variant[variant], wh)
    za_ev = fits.apply_whitening(loaded.tbl_ev.z_ans, wh)
    _transfer(
        z_ev,
        np.asarray(loaded.tbl_ev.dv, dtype=np.float64),
        za_ev,
        np.asarray(loaded.tbl_ev.row_rungs),
        "ood",
        {},
    )
    del z_ev, za_ev, z_ctx, z_ans
    _free_cuda(args.device)

    return {
        "rows": rows_all,
        "skips": skips_all,
        "per_layer": per_layer_all,
        "map_diagnostics": {f"{variant}|add|{u_label}": diag},
        "frozen_source": {variant: src},
        "budget_l": lmax,
        "leakage": leak,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument(
        "--variant",
        default="context_end",
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
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--u-store", type=Path, default=None)
    ap.add_argument("--train-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-store", type=Path, default=None)
    ap.add_argument("--pvsynth-store-root", type=Path, default=None)
    ap.add_argument("--pvsynth-dv-root", type=Path, default=None)
    ap.add_argument("--allow-overwrite-committed", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.store_root / "train_dv"
    if args.wcrung_dv_root is None:
        args.wcrung_dv_root = args.main_root / "wildchat_rung" / "dv_dataset"
    if args.pvsynth_store_root is None:
        args.pvsynth_store_root = args.store_root / "pvsynth_capture_store"
    if args.pvsynth_dv_root is None:
        args.pvsynth_dv_root = args.main_root / "pvsynth" / "dv_dataset"
    # jobd helpers read args.variants (list); keep both spellings coherent.
    args.variants = [args.variant]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules

    _assert_no_judge_modules("at entry")
    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
            arms,
            fits,
            store_io,
        )
        from explore_persona_space.orchestrate.env import load_dotenv  # noqa: F401
        from scripts.issue1739_fits import (  # noqa: F401
            _eval_rung_reconstruction,
            _git_commit,
            _load_labeled,
        )
        from scripts.issue1739_jobd_r2aug import (  # noqa: F401
            build_pool,
            committed_frozen,
            fit_pool_map,
            load_behavior,
            per_layer_rows_for,
            transfer_rows_for,
        )
        from scripts.issue1739_wcrung_arms import (  # noqa: F401
            _rb_for_behavior,
            modal_frozen_layers,
            resolve_wcrung_store,
        )

        _assert_no_judge_modules("after --import-check imports")
        print("[fair] import-check OK", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    from explore_persona_space.experiments.issue_1739 import arms
    from explore_persona_space.orchestrate.env import load_dotenv
    from scripts.issue1739_fits import _git_commit
    from scripts.issue1739_jobd_r2aug import VARIANT_SCOPE_NOTE, _env_versions, load_behavior
    from scripts.issue1739_wcrung_arms import _git_tracked, _verify_input_shas

    load_dotenv()
    for b in args.behaviors:
        out = args.out_root / b / "all_arms_spearman.json"
        if _git_tracked(out) and not args.allow_overwrite_committed:
            raise SystemExit(f"refusing to overwrite git-TRACKED output: {out}")

    layers = args.layers or list(range(args.n_layers))
    commit = _git_commit()
    env = _env_versions()
    failures: list[dict] = []
    t_all = time.time()
    for behavior in args.behaviors:
        t0 = time.time()
        try:
            loaded = load_behavior(args, behavior, layers)
            tbl_pv = load_pvsynth(args, behavior, layers, loaded.dim, loaded.shas)
            res = run_fair(args, loaded, tbl_pv, behavior, layers)
        except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as exc:
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fair] {behavior} FAILED: {exc}", flush=True)
            continue
        out_dir = args.out_root / behavior
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_arms_spearman.json"
        arms.write_summary(
            [],
            out_path,
            meta={
                "mode": "fair",
                "behavior": behavior,
                "config": "config_a",
                "regimes": [args.regime],
                "variants": [args.variant],
                "variant_scope": VARIANT_SCOPE_NOTE,
                "arms": sorted(ROSTER_FAIR),
                "label_consuming_arms": sorted(LABEL_CONSUMING),
                "map_condition": "add",
                "map_reuse_note": (
                    "the ADD/union map re-runs the committed result2_trait_aug 'add' recipe "
                    "(same pool composition, same seed, same reviewed compose+fit path); "
                    "map weights are not persisted anywhere, so the deterministic re-fit IS "
                    "the reuse — never a new map condition"
                ),
                "readout_protocol": {
                    "training_set": (
                        "union of (a) the eliciting train budget cell (the committed plotted "
                        "slice) and (b) the judged WildChat train split; label-free arms "
                        "(1/6/11) share the whitening + eval subsets but consume no labels"
                    ),
                    "pvsynth_deviation": PVSYNTH_READOUT_DEVIATION,
                    "wc_split": f"sha1(ctx_id) mod {WC_SPLIT_MOD} == {WC_EVAL_BUCKET} -> eval",
                },
                "leakage": res["leakage"],
                "n_train_contexts": len(loaded.tbl.ctx_order),
                "n_eval_contexts": len(loaded.tbl_ev.ctx_order),
                "n_wildchat_contexts": len(loaded.tbl_wc.ctx_order),
                "n_pvsynth_contexts": len(tbl_pv.ctx_order),
                "eval_rungs": sorted(set(loaded.tbl_ev.rungs) | {WC_RUNG, PV_RUNG, "train"}),
                "map_kind": "linear",
                "map_source": "refit-in-process",
                "frozen_layer_source": res["frozen_source"],
                "transfer_min_n": int(args.min_n),
                "rb": loaded.rb_meta,
                "dv_scaling_note": (
                    "per_pool_zscore_train_targets_v1 — readout training targets z-scored "
                    "per pool (eliciting rows by eliciting stats, judged-WildChat rows by "
                    "WildChat stats); evaluation DVs raw everywhere. Known input caveat "
                    "(a finding, not worked around): evil's judged-WildChat DV is "
                    "near-degenerate (mean 0.42, sd 4.4)"
                ),
                "dv_construct_caveat": (
                    "hallucination TRAIN DV is the fabricated-fraction construct while the "
                    "WildChat-rung DV is the graded trait rubric — its fair readout trains "
                    "on a mixed-construct target (disclosed)"
                    if behavior == "hallucination"
                    else "train and WildChat DVs are both graded 0-100 trait-rubric constructs"
                ),
                "budget_l": res["budget_l"],
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
        (out_dir / "map_diagnostics.json").write_text(json.dumps(res["map_diagnostics"], indent=1))
        print(
            f"[fair] {behavior} done: {len(res['rows'])} transfer rows in "
            f"{time.time() - t0:.0f}s -> {out_path}",
            flush=True,
        )
        _verify_input_shas(loaded.shas)
        del loaded, tbl_pv
    _assert_no_judge_modules("at exit")
    print(f"[fair] all done in {time.time() - t_all:.0f}s", flush=True)
    if failures:
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "fair_failures.json").write_text(json.dumps(failures, indent=1))
        for f in failures:
            print(f"[fair] FAILED {f}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(2)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
