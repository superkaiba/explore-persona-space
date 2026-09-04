#!/usr/bin/env python3
"""Score the issue 1739 context-to-answer unlabeled-data ladder.

The causal estimand is D(U) = rho(arm7 mapped-answer ridge) minus
rho(arm4 context ridge). Whitening is fit once on the full generic plus
trait-eliciting union and is then frozen. Generic-only and union-scaled map
pools use nested, seed-keyed prefixes. The judged-only and fold-clean maps
are registered controls, not members of the primary U ladder.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts" / "issue1739_r2v2_score.py").exists():
        raise RuntimeError(f"repository root resolution failed: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()
BEHAVIORS = ("evil", "sycophancy", "hallucination")
U_SIZES = (250, 500, 1000, 2000, 5000, 10000, 18793)
MAP_CONFIGS = ("generic_only", "union_scaled")
MAP_VARIANTS = ("true", "shufpair")
BASE_ARMS = ("arm4_ridge_ctx", "arm12_oracle_reg")
MAPPED_ARM = "arm7_map_ridge_pred"
SCHEMA_VERSION = 1
FULL_GENERIC_N = 18793
WC_SPLIT_MOD = 5
WC_EVAL_BUCKET = 4
DEFAULT_OUT_ROOT = Path("eval_results/issue_1739/uladder")
DEFAULT_MAIN_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
DEFAULT_STORE_ROOT = Path("data/issue_1739/hf_dl")


def _log(message: str) -> None:
    print(f"[uladder {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _sha_idx(rows) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()[:16]


def _atomic_json(path: Path, payload: object) -> None:
    from explore_persona_space.atomic_io import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))


def _nested_prefixes(n: int, sizes: list[int], *, seed: int, namespace: int):
    """Return sorted nested row sets keyed by requested size."""
    import numpy as np

    if n < max(sizes):
        raise ValueError(f"pool has {n} rows but requested rung {max(sizes)}")
    rng = np.random.default_rng([1739, namespace, int(seed)])
    order = rng.permutation(n)
    out = {int(size): np.sort(order[: int(size)]).astype(np.int64) for size in sizes}
    prior = set()
    for size in sorted(out):
        current = set(map(int, out[size]))
        if not prior.issubset(current):
            raise AssertionError(f"nested-prefix invariant failed at U={size}")
        prior = current
    return out


def _component_shuffle(n: int, *, seed: int, namespace: int):
    """Pairing shuffle for a single pool component via the reviewed helper."""
    import numpy as np

    from scripts.issue1739_r2v2_score import pairing_shuffle_perm, shufpair_structural_check

    if n < 2:
        raise ValueError(f"pairing shuffle requires at least two rows, got {n}")
    # The reviewed helper requires two nonempty components. Add one dummy
    # singleton component, validate its full permutation, then drop it.
    perm, fingerprints = pairing_shuffle_perm(n, n + 1, seed=seed + namespace * 100000)
    check = shufpair_structural_check(perm, n, n + 1)
    actual = np.asarray(perm[:n], dtype=np.int64)
    if np.array_equal(actual, np.arange(n)):
        raise RuntimeError("single-component pairing shuffle was identity")
    return actual, {**fingerprints, **check, "single_component_adapter": True}


def _wc_eval_mask(ctx_ids: list[str]):
    import numpy as np

    return np.asarray(
        [
            int(hashlib.sha1(str(c).encode()).hexdigest(), 16) % WC_SPLIT_MOD
            == WC_EVAL_BUCKET
            for c in ctx_ids
        ],
        dtype=bool,
    )


def _wc_fold_ids(ctx_ids: list[str], n_folds: int):
    import numpy as np

    return np.asarray(
        [
            int(hashlib.sha1((str(c) + "|fairfold").encode()).hexdigest(), 16)
            % n_folds
            for c in ctx_ids
        ],
        dtype=np.int64,
    )


@dataclasses.dataclass
class State:
    loaded: object
    tbl_ood: object | None
    ood_note: dict
    layers: list[int]
    whitening: object
    x_pool: object
    y_pool: object
    n_generic: int
    n_eliciting: int
    generic_prefixes: dict[int, object]
    elic_prefixes: dict[int, object]
    z_ctx: object
    z_ans: object
    dv_train: object
    readout_rows: object
    cell_oof: object
    cell_full: object
    elic_rows: object
    wc_eval_rows: object
    eval_blocks: list[dict]
    frozen: dict[str, int]
    frozen_source: str
    wc_recon_ctx: object
    wc_recon_ans: object


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behaviors", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    ap.add_argument("--seed", type=int, required=False, default=0)
    ap.add_argument("--u-sizes", nargs="+", type=int, default=list(U_SIZES))
    ap.add_argument("--configs", nargs="+", choices=MAP_CONFIGS, default=list(MAP_CONFIGS))
    ap.add_argument("--map-variants", nargs="+", choices=MAP_VARIANTS, default=list(MAP_VARIANTS))
    ap.add_argument("--variant", choices=("context_end",), default="context_end")
    ap.add_argument("--regime", choices=("e1",), default="e1")
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--parity-tol", type=float, default=1e-3)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    ap.add_argument("--u-store", type=Path, default=None)
    ap.add_argument("--train-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-dv-root", type=Path, default=None)
    ap.add_argument("--wcrung-store", type=Path, default=None)
    ap.add_argument("--ood-store-root", type=Path, default=None)
    ap.add_argument("--evil-ood-dv", type=Path, default=None)
    ap.add_argument("--syco-ood-dv", type=Path, default=None)
    ap.add_argument("--ood-dv-max-null-frac", type=float, default=0.05)
    ap.add_argument("--rb-source", default="auto", choices=("auto", "bank", "extract"))
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if len(set(args.u_sizes)) != len(args.u_sizes):
        ap.error("--u-sizes contains duplicates")
    if sorted(args.u_sizes) != list(args.u_sizes):
        ap.error("--u-sizes must be strictly increasing")
    if any(u <= 0 or u > FULL_GENERIC_N for u in args.u_sizes):
        ap.error(f"--u-sizes must lie in [1, {FULL_GENERIC_N}]")
    if args.pilot:
        args.behaviors = args.behaviors[:1]
        args.u_sizes = [max(args.u_sizes)]
        args.configs = args.configs[:1]
        args.map_variants = ["true"]
    if args.u_store is None:
        args.u_store = args.store_root / "u_store"
    if args.train_dv_root is None:
        args.train_dv_root = args.store_root / "train_dv"
    if args.wcrung_dv_root is None:
        args.wcrung_dv_root = args.main_root / "wildchat_rung" / "dv_dataset"
    if args.ood_store_root is None:
        args.ood_store_root = args.store_root / "ood_mirror" / "issue1739_ctxmap"
    if args.evil_ood_dv is None:
        args.evil_ood_dv = (
            args.main_root / "evil_ood_full" / "dv_dataset" / "evil" / "labeling.json"
        )
    if args.syco_ood_dv is None:
        args.syco_ood_dv = (
            args.ood_store_root / "syco_ood" / "dv_dataset" / "sycophancy" / "labeling.json"
        )
    args.variants = [args.variant]
    args.map_kinds = ["linear"]
    args.draw = 0
    return args


def _import_check() -> None:
    from explore_persona_space.experiments.issue_1739 import arms, fits
    from scripts.issue1739_fits import _eval_rung_reconstruction, _fit_map
    from scripts.issue1739_jobd_r2aug import build_pool, committed_frozen, load_behavior
    from scripts.issue1739_r2v2_score import (
        _whiten_concat_blocks,
        load_ood_table,
        pairing_shuffle_perm,
    )

    assert callable(_fit_map) and callable(_eval_rung_reconstruction)
    assert callable(build_pool) and callable(committed_frozen) and callable(load_behavior)
    assert callable(load_ood_table) and callable(_whiten_concat_blocks)
    assert callable(pairing_shuffle_perm)
    for arm in (*BASE_ARMS, MAPPED_ARM):
        assert arm in arms.ARM_REGISTRY
    assert callable(fits.fit_whitening)
    print("[uladder] import-check OK", flush=True)


def _pool_columns(
    state: State, config: str, u_size: int, *, fold_exclude: object | None = None
):
    """Resolve source-pool columns for one map fit."""
    import numpy as np

    gen = state.generic_prefixes[int(u_size)]
    if config == "generic_only":
        cols = np.asarray(gen, dtype=np.int64)
        return cols, len(cols), 0
    if config != "union_scaled":
        raise ValueError(config)
    n_elic = max(1, round(int(u_size) * state.n_eliciting / state.n_generic))
    n_elic = min(n_elic, state.n_eliciting)
    elic = np.asarray(state.elic_prefixes[n_elic], dtype=np.int64)
    if fold_exclude is not None:
        exclude = set(map(int, np.asarray(fold_exclude)))
        elic = np.asarray([x for x in elic if int(x) not in exclude], dtype=np.int64)
    cols = np.concatenate([gen, state.n_generic + elic])
    return cols, len(gen), len(elic)


def _whiten_columns(raw, whitening, columns):
    """Whiten selected columns with only one layer-sized gather at a time."""
    import numpy as np

    columns = np.asarray(columns, dtype=np.int64)
    out = np.empty((raw.shape[0], len(columns), raw.shape[2]), dtype=np.float64)
    for li in range(raw.shape[0]):
        selected = np.asarray(raw[li][columns], dtype=np.float64)
        selected -= whitening.mu[li][None, :]
        np.matmul(selected, whitening.w[li], out=out[li])
        del selected
    return out


def _whiten_selected_layers(blocks, whitening, layer_indices: set[int]):
    """Whiten concatenated raw blocks for only the requested layer indices."""
    import numpy as np

    n_all = sum(int(block.shape[1]) for block in blocks)
    d = int(blocks[0].shape[2])
    result = {}
    for li in sorted(layer_indices):
        out = np.empty((1, n_all, d), dtype=np.float64)
        at = 0
        for block in blocks:
            n = int(block.shape[1])
            selected = np.asarray(block[li], dtype=np.float64)
            selected -= whitening.mu[li][None, :]
            np.matmul(selected, whitening.w[li], out=out[0, at : at + n])
            del selected
            at += n
        result[li] = out
    return result


def _slice_mapfit(mapfit, layer_idx: int):
    from explore_persona_space.experiments.issue_1739.fits import MapFit

    return MapFit(
        w=mapfit.w[layer_idx : layer_idx + 1],
        x_mu=mapfit.x_mu[layer_idx : layer_idx + 1],
        x_sd=mapfit.x_sd[layer_idx : layer_idx + 1],
        y_mu=mapfit.y_mu[layer_idx : layer_idx + 1],
        diagnostics=mapfit.diagnostics,
        kind="linear",
    )


def _prepare_state(args: argparse.Namespace, behavior: str, layers: list[int]) -> State:
    import gc
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from scripts.issue1739_jobd_r2aug import (
        LMAX,
        _pool_zscored_dv,
        build_pool,
        committed_frozen,
        load_behavior,
    )
    from scripts.issue1739_r2v2_score import _whiten_concat_blocks, load_ood_table

    _log(f"{behavior}: loading staged activation tables")
    loaded = load_behavior(args, behavior, layers)
    tbl_ood, ood_note = load_ood_table(args, behavior, layers, loaded.dim, loaded.shas)
    x_pool, y_pool, pool_label, n_pool, pool_meta = build_pool(
        args, loaded, args.variant, layers, "add"
    )
    n_generic = int(pool_meta["add_n_generic"])
    n_eliciting = int(pool_meta["add_n_eliciting"])
    if n_generic != FULL_GENERIC_N:
        raise RuntimeError(
            f"{behavior}: generic pool has {n_generic}, expected frozen full rung "
            f"{FULL_GENERIC_N}"
        )
    if n_pool != n_generic + n_eliciting:
        raise RuntimeError(f"{behavior}: malformed union pool {n_pool}")
    _log(f"{behavior}: fitting frozen whitening on {pool_label}")
    whitening = fits.fit_whitening(x_pool, device=args.device, seed=args.seed)

    frozen, frozen_source = committed_frozen(
        args, loaded, behavior, args.variant, layers, (*BASE_ARMS, MAPPED_ARM)
    )
    selected_layers = set(map(int, frozen.values()))
    _log(
        f"{behavior}: frozen layer indices {frozen}; materializing only "
        f"{sorted(selected_layers)} for repeated readouts"
    )

    train_blocks_ctx = [
        loaded.tbl.z_by_variant[args.variant],
        loaded.tbl_wc.z_by_variant[args.variant],
    ]
    train_blocks_ans = [loaded.tbl.z_ans, loaded.tbl_wc.z_ans]
    z_ctx = _whiten_selected_layers(train_blocks_ctx, whitening, selected_layers)
    z_ans = _whiten_selected_layers(train_blocks_ans, whitening, selected_layers)

    n_train = len(loaded.tbl.ctx_order)
    wc_mask = _wc_eval_mask(list(map(str, loaded.tbl_wc.ctx_order)))
    wc_eval_rows = np.flatnonzero(wc_mask)
    wc_train_rows = np.flatnonzero(~wc_mask)
    elic_cell = fits.realize_budget_cell(
        loaded.tbl.groups,
        budget_l=LMAX[behavior],
        draw=args.draw,
        seed=args.seed,
    )
    readout_rows = np.concatenate(
        [elic_cell.row_idx, n_train + wc_train_rows]
    ).astype(np.int64)
    dv_merged = np.concatenate(
        [
            np.asarray(loaded.tbl.dv, dtype=np.float64),
            np.asarray(loaded.tbl_wc.dv, dtype=np.float64),
        ]
    )
    dv_train = _pool_zscored_dv(dv_merged, elic_cell.row_idx, n_train + wc_train_rows)
    wc_fold = _wc_fold_ids(
        [str(loaded.tbl_wc.ctx_order[i]) for i in wc_train_rows],
        elic_cell.n_folds,
    )
    cell_oof = fits.BudgetCell(
        row_idx=readout_rows,
        fold_ids=np.concatenate([elic_cell.fold_ids, wc_fold]),
        n_folds=elic_cell.n_folds,
        budget_l=LMAX[behavior],
        draw=args.draw,
        seed=args.seed,
        fold_scheme=f"uladder-union-{elic_cell.fold_scheme}",
    )
    cell_full = fits.BudgetCell(
        row_idx=readout_rows,
        fold_ids=np.zeros(len(readout_rows), dtype=np.int64),
        n_folds=1,
        budget_l=LMAX[behavior],
        draw=args.draw,
        seed=args.seed,
        fold_scheme="uladder-union-full",
    )

    def selected_eval(ctx_raw, ans_raw):
        return (
            _whiten_selected_layers([ctx_raw], whitening, selected_layers),
            _whiten_selected_layers([ans_raw], whitening, selected_layers),
        )

    eval_blocks: list[dict] = []
    wc_ctx = {li: arr[:, n_train + wc_eval_rows] for li, arr in z_ctx.items()}
    wc_ans = {li: arr[:, n_train + wc_eval_rows] for li, arr in z_ans.items()}
    eval_blocks.append(
        {
            "name": "wildchat",
            "z_ctx": wc_ctx,
            "z_ans": wc_ans,
            "dv": np.asarray(loaded.tbl_wc.dv, dtype=np.float64)[wc_eval_rows],
            "rungs": np.asarray(["wildchat_rung"] * len(wc_eval_rows)),
        }
    )
    ev_ctx, ev_ans = selected_eval(
        loaded.tbl_ev.z_by_variant[args.variant], loaded.tbl_ev.z_ans
    )
    eval_blocks.append(
        {
            "name": "original_ood",
            "z_ctx": ev_ctx,
            "z_ans": ev_ans,
            "dv": np.asarray(loaded.tbl_ev.dv, dtype=np.float64),
            "rungs": np.asarray(loaded.tbl_ev.row_rungs),
        }
    )
    if tbl_ood is not None:
        wide_ctx, wide_ans = selected_eval(tbl_ood.z_by_variant[args.variant], tbl_ood.z_ans)
        eval_blocks.append(
            {
                "name": "wide_ood",
                "z_ctx": wide_ctx,
                "z_ans": wide_ans,
                "dv": np.asarray(tbl_ood.dv, dtype=np.float64),
                "rungs": np.asarray(tbl_ood.row_rungs),
            }
        )

    # The all-layer WildChat reconstruction companion is small enough to
    # retain and is evaluated for every fitted map.
    wc_recon_ctx = _whiten_concat_blocks(
        [loaded.tbl_wc.z_by_variant[args.variant][:, wc_eval_rows]], whitening
    )
    wc_recon_ans = _whiten_concat_blocks(
        [loaded.tbl_wc.z_ans[:, wc_eval_rows]], whitening
    )

    generic_prefixes = _nested_prefixes(
        n_generic, list(args.u_sizes), seed=args.seed, namespace=31
    )
    elic_sizes = sorted(
        {
            min(n_eliciting, max(1, round(u * n_eliciting / n_generic)))
            for u in args.u_sizes
        }
    )
    elic_prefixes = _nested_prefixes(
        n_eliciting, elic_sizes, seed=args.seed, namespace=32
    )

    # The map pool is now self-contained. Release duplicate raw table arrays
    # while retaining metadata used by parity and hash verification.
    for table in [loaded.tbl, loaded.tbl_wc, loaded.tbl_ev] + (
        [tbl_ood] if tbl_ood is not None else []
    ):
        table.z_by_variant.clear()
        table.z_ans = None
    loaded.u_arrays.clear()
    gc.collect()
    _log(f"{behavior}: prepared state; peak RSS {_rss_gib():.1f} GiB")
    return State(
        loaded=loaded,
        tbl_ood=tbl_ood,
        ood_note=ood_note,
        layers=layers,
        whitening=whitening,
        x_pool=x_pool,
        y_pool=y_pool,
        n_generic=n_generic,
        n_eliciting=n_eliciting,
        generic_prefixes=generic_prefixes,
        elic_prefixes=elic_prefixes,
        z_ctx=z_ctx,
        z_ans=z_ans,
        dv_train=dv_train,
        readout_rows=readout_rows,
        cell_oof=cell_oof,
        cell_full=cell_full,
        elic_rows=np.asarray(elic_cell.row_idx, dtype=np.int64),
        wc_eval_rows=wc_eval_rows,
        eval_blocks=eval_blocks,
        frozen=frozen,
        frozen_source=frozen_source,
        wc_recon_ctx=wc_recon_ctx,
        wc_recon_ans=wc_recon_ans,
    )


def _fit_map_for_columns(
    args: argparse.Namespace,
    state: State,
    columns,
    *,
    map_variant: str,
    n_generic: int,
    n_eliciting: int,
):
    import gc

    from scripts.issue1739_fits import _fit_map
    from scripts.issue1739_r2v2_score import pairing_shuffle_perm, shufpair_structural_check

    x_w = _whiten_columns(state.x_pool, state.whitening, columns)
    y_w = _whiten_columns(state.y_pool, state.whitening, columns)
    shuffle_meta = None
    if map_variant == "shufpair":
        if n_eliciting:
            perm, fingerprints = pairing_shuffle_perm(
                n_generic, n_generic + n_eliciting, seed=args.seed
            )
            shuffle_meta = {
                **fingerprints,
                **shufpair_structural_check(perm, n_generic, n_generic + n_eliciting),
            }
        else:
            perm, shuffle_meta = _component_shuffle(
                n_generic, seed=args.seed, namespace=0
            )
        y_w = y_w[:, perm]
    ns = SimpleNamespace(
        map_kind="linear",
        device=args.device,
        seeds=(args.seed,),
        mlp_map_width=None,
        krr_map_centers=None,
    )
    fit = _fit_map(ns, x_w, y_w)
    del x_w, y_w
    gc.collect()
    return fit, shuffle_meta


def _rho(scores, dv, *, min_n: int) -> tuple[float | None, int]:
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import spearman_rows

    scores = np.asarray(scores, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    keep = np.isfinite(scores) & np.isfinite(dv)
    n = int(keep.sum())
    if n < min_n:
        return None, n
    return float(spearman_rows(scores[keep][None], dv[keep])[0]), n


def _row(
    *,
    behavior: str,
    seed: int,
    arm: str,
    rho: float | None,
    n_eval: int,
    layer: int,
    setting: str,
    config: str,
    map_variant: str,
    u_size: int | None,
    n_map: int | None,
    n_generic: int | None,
    n_eliciting: int | None,
    control: bool = False,
) -> dict:
    setting_group = (
        "in_dist" if setting == "train" else ("generic" if setting == "wildchat_rung" else "ood")
    )
    return {
        "mode": "uladder",
        "behavior": behavior,
        "seed": int(seed),
        "arm": arm,
        "rho_frozen": rho,
        "n_eval": int(n_eval),
        "layer": int(layer),
        "eval_rung": setting,
        "setting_group": setting_group,
        "config": config,
        "map_variant": map_variant,
        "u_size": None if u_size is None else int(u_size),
        "n_map": None if n_map is None else int(n_map),
        "n_generic": None if n_generic is None else int(n_generic),
        "n_eliciting": None if n_eliciting is None else int(n_eliciting),
        "control": bool(control),
        "ci_frozen": None,
    }


def _base_arm_rows(
    args: argparse.Namespace, state: State, behavior: str, arm: str
) -> list[dict]:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    li = int(state.frozen[arm])
    rb = np.zeros((1, state.loaded.dim), dtype=np.float64)
    data = arms.CellData(
        z_ctx=state.z_ctx[li],
        z_ans=state.z_ans[li],
        dv=state.dv_train,
        rb=rb,
        layers=(state.layers[li],),
    )
    scores, skipped = arms.run_cell(
        data, state.cell_oof, arms=[arm], device=args.device
    )
    if arm in skipped or arm not in scores:
        raise RuntimeError(f"{behavior}: {arm} OOF failed: {skipped}")
    n_elic = len(state.elic_rows)
    rho, n = _rho(
        scores[arm][0, :n_elic],
        np.asarray(state.loaded.tbl.dv, dtype=np.float64)[state.elic_rows],
        min_n=args.min_n,
    )
    rows = [
        _row(
            behavior=behavior,
            seed=args.seed,
            arm=arm,
            rho=rho,
            n_eval=n,
            layer=state.layers[li],
            setting="train",
            config="base",
            map_variant="not_applicable",
            u_size=None,
            n_map=None,
            n_generic=None,
            n_eliciting=None,
        )
    ]
    for block in state.eval_blocks:
        ev_scores, ev_skips = arms.run_transfer_cell(
            data,
            state.cell_full,
            block["z_ctx"][li],
            block["dv"],
            za_ev=block["z_ans"][li],
            arms=[arm],
            device=args.device,
            ridge_folds=(0,),
        )
        if arm in ev_skips or arm not in ev_scores:
            raise RuntimeError(f"{behavior}/{block['name']}: {arm} transfer failed: {ev_skips}")
        for rung in sorted(set(map(str, block["rungs"]))):
            mask = np.asarray(block["rungs"], dtype=str) == rung
            rho, n = _rho(ev_scores[arm][0, mask], block["dv"][mask], min_n=args.min_n)
            rows.append(
                _row(
                    behavior=behavior,
                    seed=args.seed,
                    arm=arm,
                    rho=rho,
                    n_eval=n,
                    layer=state.layers[li],
                    setting=rung,
                    config="base",
                    map_variant="not_applicable",
                    u_size=None,
                    n_map=None,
                    n_generic=None,
                    n_eliciting=None,
                )
            )
    return rows


def _mapped_arm_rows(
    args: argparse.Namespace,
    state: State,
    behavior: str,
    mapfit,
    *,
    config: str,
    map_variant: str,
    u_size: int | None,
    n_generic: int,
    n_eliciting: int,
    control: bool = False,
) -> list[dict]:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms

    arm = MAPPED_ARM
    li = int(state.frozen[arm])
    fit_one = _slice_mapfit(mapfit, li)
    rb = np.zeros((1, state.loaded.dim), dtype=np.float64)
    data = arms.CellData(
        z_ctx=state.z_ctx[li],
        z_ans=state.z_ans[li],
        dv=state.dv_train,
        rb=rb,
        mapfit=fit_one,
        layers=(state.layers[li],),
    )
    scores, skipped = arms.run_cell(
        data, state.cell_oof, arms=[arm], device=args.device
    )
    if arm in skipped or arm not in scores:
        raise RuntimeError(f"{behavior}/{config}: {arm} OOF failed: {skipped}")
    n_elic_eval = len(state.elic_rows)
    rho, n = _rho(
        scores[arm][0, :n_elic_eval],
        np.asarray(state.loaded.tbl.dv, dtype=np.float64)[state.elic_rows],
        min_n=args.min_n,
    )
    common = {
        "behavior": behavior,
        "seed": args.seed,
        "arm": arm,
        "layer": state.layers[li],
        "config": config,
        "map_variant": map_variant,
        "u_size": u_size,
        "n_map": n_generic + n_eliciting,
        "n_generic": n_generic,
        "n_eliciting": n_eliciting,
        "control": control,
    }
    rows = [_row(rho=rho, n_eval=n, setting="train", **common)]
    for block in state.eval_blocks:
        ev_scores, ev_skips = arms.run_transfer_cell(
            data,
            state.cell_full,
            block["z_ctx"][li],
            block["dv"],
            za_ev=block["z_ans"][li],
            arms=[arm],
            device=args.device,
            ridge_folds=(0,),
        )
        if arm in ev_skips or arm not in ev_scores:
            raise RuntimeError(
                f"{behavior}/{config}/{block['name']}: {arm} transfer failed: {ev_skips}"
            )
        for rung in sorted(set(map(str, block["rungs"]))):
            mask = np.asarray(block["rungs"], dtype=str) == rung
            rho, n = _rho(ev_scores[arm][0, mask], block["dv"][mask], min_n=args.min_n)
            rows.append(_row(rho=rho, n_eval=n, setting=rung, **common))
    return rows


def _clone_base_rows(
    base_rows: list[dict],
    *,
    config: str,
    map_variant: str,
    u_size: int | None,
    n_map: int,
    n_generic: int,
    n_eliciting: int,
    control: bool = False,
) -> list[dict]:
    return [
        {
            **row,
            "config": config,
            "map_variant": map_variant,
            "u_size": u_size,
            "n_map": int(n_map),
            "n_generic": int(n_generic),
            "n_eliciting": int(n_eliciting),
            "control": bool(control),
        }
        for row in base_rows
    ]


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _perfold_control_train_row(
    args: argparse.Namespace,
    state: State,
    behavior: str,
    *,
    config: str,
) -> tuple[dict, list[dict]]:
    """Fit one map per held-out group fold and stitch the eliciting predictions."""
    import gc
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    if config not in ("judged_only", "fold_clean_union_full"):
        raise ValueError(config)
    li = int(state.frozen[MAPPED_ARM])
    stitched = np.full(len(state.elic_rows), np.nan, dtype=np.float64)
    fold_diags: list[dict] = []
    n_gen_last = n_elic_last = 0
    for fold in range(state.cell_oof.n_folds):
        held_local = np.flatnonzero(state.cell_oof.fold_ids[: len(state.elic_rows)] == fold)
        held_source = state.elic_rows[held_local]
        excluded = set(map(int, held_source))
        elic = np.asarray(
            [i for i in range(state.n_eliciting) if i not in excluded], dtype=np.int64
        )
        if config == "judged_only":
            columns = state.n_generic + elic
            n_gen = 0
        else:
            columns = np.concatenate(
                [np.arange(state.n_generic, dtype=np.int64), state.n_generic + elic]
            )
            n_gen = state.n_generic
        n_elic = len(elic)
        mapfit, _ = _fit_map_for_columns(
            args,
            state,
            columns,
            map_variant="true",
            n_generic=n_gen,
            n_eliciting=n_elic,
        )
        fit_one = _slice_mapfit(mapfit, li)
        train_mask = state.cell_oof.fold_ids != fold
        train_rows = state.cell_oof.row_idx[train_mask]
        train_cell = fits.BudgetCell(
            row_idx=train_rows,
            fold_ids=np.zeros(len(train_rows), dtype=np.int64),
            n_folds=1,
            budget_l=state.cell_oof.budget_l,
            draw=args.draw,
            seed=args.seed,
            fold_scheme=f"{config}-heldout-fold-{fold}",
        )
        data = arms.CellData(
            z_ctx=state.z_ctx[li],
            z_ans=state.z_ans[li],
            dv=state.dv_train,
            rb=np.zeros((1, state.loaded.dim), dtype=np.float64),
            mapfit=fit_one,
            layers=(state.layers[li],),
        )
        scores, skipped = arms.run_transfer_cell(
            data,
            train_cell,
            state.z_ctx[li][:, held_source],
            np.asarray(state.loaded.tbl.dv, dtype=np.float64)[held_source],
            za_ev=state.z_ans[li][:, held_source],
            arms=[MAPPED_ARM],
            device=args.device,
            ridge_folds=(0,),
        )
        if MAPPED_ARM in skipped or MAPPED_ARM not in scores:
            raise RuntimeError(f"{behavior}/{config}/fold{fold}: {skipped}")
        stitched[held_local] = scores[MAPPED_ARM][0]
        fold_diags.append(
            {
                "fold": fold,
                "n_heldout": len(held_source),
                "n_map": n_gen + n_elic,
                "n_generic": n_gen,
                "n_eliciting": n_elic,
                "heldout_source_sha256": _sha_idx(held_source),
                "map_diagnostics": _jsonable(mapfit.diagnostics),
            }
        )
        n_gen_last, n_elic_last = n_gen, n_elic
        del scores, data, fit_one, mapfit
        gc.collect()
        _log(f"{behavior}/{config}: fold {fold + 1}/{state.cell_oof.n_folds} complete")
    if not np.isfinite(stitched).all():
        missing = int((~np.isfinite(stitched)).sum())
        raise RuntimeError(f"{behavior}/{config}: {missing} eliciting rows not scored exactly once")
    rho, n = _rho(
        stitched,
        np.asarray(state.loaded.tbl.dv, dtype=np.float64)[state.elic_rows],
        min_n=args.min_n,
    )
    return (
        _row(
            behavior=behavior,
            seed=args.seed,
            arm=MAPPED_ARM,
            rho=rho,
            n_eval=n,
            layer=state.layers[li],
            setting="train",
            config=config,
            map_variant="true",
            u_size=FULL_GENERIC_N if config == "fold_clean_union_full" else None,
            n_map=n_gen_last + n_elic_last,
            n_generic=n_gen_last,
            n_eliciting=n_elic_last,
            control=True,
        ),
        fold_diags,
    )


def _parity_gate(
    args: argparse.Namespace,
    behavior: str,
    rows: list[dict],
) -> dict:
    """Check the full-union true-map cell against the committed FAIR result."""
    if args.seed != 0 or max(args.u_sizes) != FULL_GENERIC_N:
        return {"status": "not_applicable", "reason": "parity is registered for seed 0 full U"}
    reference = (
        args.main_root / "result2_fair_v2" / behavior / "all_arms_spearman.json"
    )
    if not reference.exists():
        raise FileNotFoundError(f"parity reference missing: {reference}")
    payload = json.loads(reference.read_text())
    old = {
        (r["arm"], r["eval_rung"]): float(r["rho_frozen"])
        for r in payload.get("transfer_rows", [])
        if r.get("arm") in (*BASE_ARMS, MAPPED_ARM)
        and r.get("eval_rung") != "pvsynth"
    }
    current = {
        (r["arm"], r["eval_rung"]): float(r["rho_frozen"])
        for r in rows
        if r.get("config") == "union_scaled"
        and r.get("map_variant") == "true"
        and r.get("u_size") == FULL_GENERIC_N
        and r.get("arm") in (*BASE_ARMS, MAPPED_ARM)
        and r.get("eval_rung") in {setting for _, setting in old}
        and r.get("rho_frozen") is not None
    }
    missing = sorted(set(old) - set(current))
    if missing:
        raise RuntimeError(f"{behavior}: parity rows missing {missing}")
    deltas = {f"{arm}|{setting}": abs(current[(arm, setting)] - value) for (arm, setting), value in old.items()}
    worst = max(deltas.values(), default=0.0)
    if worst > args.parity_tol:
        raise RuntimeError(
            f"{behavior}: parity gate max absolute delta {worst:.6g} exceeds "
            f"{args.parity_tol}; deltas={deltas}"
        )
    return {
        "status": "passed",
        "reference": str(reference),
        "tolerance": args.parity_tol,
        "max_abs_delta": worst,
        "deltas": deltas,
    }


def _resume_ok(
    path: Path,
    *,
    commit: str,
    seed: int,
    u_sizes: list[int],
    configs: list[str],
    map_variants: list[str],
    pilot: bool,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "summary absent"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"unreadable summary: {exc}"
    meta = payload.get("meta", {})
    expected = {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "seed": seed,
        "complete": True,
        "u_sizes": list(u_sizes),
        "configs": list(configs),
        "map_variants": list(map_variants),
        "pilot": bool(pilot),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            return False, f"{key}={meta.get(key)!r}, expected {value!r}"
    return True, "exact schema, commit, seed and completion marker"


def _write_npz(path: Path, rows: list[dict]) -> None:
    import numpy as np

    from explore_persona_space.atomic_io import atomic_replace

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                arm=np.asarray([r["arm"] for r in rows]),
                config=np.asarray([r["config"] for r in rows]),
                map_variant=np.asarray([r["map_variant"] for r in rows]),
                eval_rung=np.asarray([r["eval_rung"] for r in rows]),
                setting_group=np.asarray([r["setting_group"] for r in rows]),
                u_size=np.asarray(
                    [-1 if r["u_size"] is None else r["u_size"] for r in rows],
                    dtype=np.int64,
                ),
                rho=np.asarray(
                    [np.nan if r["rho_frozen"] is None else r["rho_frozen"] for r in rows],
                    dtype=np.float64,
                ),
                n_eval=np.asarray([r["n_eval"] for r in rows], dtype=np.int64),
            )


def run_behavior(
    args: argparse.Namespace,
    behavior: str,
    layers: list[int],
    *,
    commit: str,
) -> dict:
    import gc
    import numpy as np

    from scripts.issue1739_fits import _eval_rung_reconstruction
    from scripts.issue1739_wcrung_arms import _verify_input_shas

    started = time.time()
    out_dir = args.out_root / behavior / f"seed{args.seed}"
    summary_path = out_dir / "all_arms_spearman.json"
    resumable, reason = _resume_ok(
        summary_path,
        commit=commit,
        seed=args.seed,
        u_sizes=args.u_sizes,
        configs=args.configs,
        map_variants=args.map_variants,
        pilot=args.pilot,
    )
    if resumable:
        _log(f"{behavior}/seed{args.seed}: resume skip ({reason})")
        return json.loads(summary_path.read_text())
    if summary_path.exists():
        _log(f"{behavior}/seed{args.seed}: prior summary not reusable ({reason})")

    state = _prepare_state(args, behavior, layers)
    base_rows = []
    for arm in BASE_ARMS:
        _log(f"{behavior}: scoring cached base {arm}")
        base_rows.extend(_base_arm_rows(args, state, behavior, arm))
    rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    checkpoints = out_dir / "percell"
    checkpoints.mkdir(parents=True, exist_ok=True)

    total = len(args.configs) * len(args.map_variants) * len(args.u_sizes)
    unit_i = 0
    for config in args.configs:
        for map_variant in args.map_variants:
            for u_size in args.u_sizes:
                unit_i += 1
                unit_started = time.time()
                columns, n_gen, n_elic = _pool_columns(state, config, u_size)
                _log(
                    f"{behavior}: map {unit_i}/{total} {config}/{map_variant}/U={u_size} "
                    f"(n={len(columns)})"
                )
                mapfit, shuffle_meta = _fit_map_for_columns(
                    args,
                    state,
                    columns,
                    map_variant=map_variant,
                    n_generic=n_gen,
                    n_eliciting=n_elic,
                )
                mapped = _mapped_arm_rows(
                    args,
                    state,
                    behavior,
                    mapfit,
                    config=config,
                    map_variant=map_variant,
                    u_size=u_size,
                    n_generic=n_gen,
                    n_eliciting=n_elic,
                )
                unit_rows = _clone_base_rows(
                    base_rows,
                    config=config,
                    map_variant=map_variant,
                    u_size=u_size,
                    n_map=n_gen + n_elic,
                    n_generic=n_gen,
                    n_eliciting=n_elic,
                ) + mapped
                rows.extend(unit_rows)
                diag_key = f"{config}|{map_variant}|U={u_size}"
                diagnostics[diag_key] = {
                    "u_size": u_size,
                    "n_map": n_gen + n_elic,
                    "n_generic": n_gen,
                    "n_eliciting": n_elic,
                    "column_sha256": _sha_idx(columns),
                    "shuffle": shuffle_meta,
                    "fit": _jsonable(mapfit.diagnostics),
                    "wildchat_reconstruction": _jsonable(
                        _eval_rung_reconstruction(
                            mapfit,
                            state.wc_recon_ctx,
                            state.wc_recon_ans,
                            rungs=["wildchat_rung"] * state.wc_recon_ctx.shape[1],
                            knn=False,
                        )
                    ),
                    "wall_s": round(time.time() - unit_started, 3),
                    "peak_rss_gib": round(_rss_gib(), 3),
                }
                _atomic_json(
                    checkpoints
                    / f"{config}_{map_variant}_u{u_size}.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "behavior": behavior,
                        "seed": args.seed,
                        "commit": commit,
                        "rows": unit_rows,
                        "diagnostics": diagnostics[diag_key],
                        "complete": True,
                    },
                )
                del mapfit, mapped, unit_rows
                gc.collect()

    if not args.pilot:
        _log(f"{behavior}: running judged-only fold-clean control")
        judged_train, judged_fold_diags = _perfold_control_train_row(
            args, state, behavior, config="judged_only"
        )
        judged_columns = state.n_generic + np.arange(state.n_eliciting, dtype=np.int64)
        judged_map, _ = _fit_map_for_columns(
            args,
            state,
            judged_columns,
            map_variant="true",
            n_generic=0,
            n_eliciting=state.n_eliciting,
        )
        judged_all = _mapped_arm_rows(
            args,
            state,
            behavior,
            judged_map,
            config="judged_only",
            map_variant="true",
            u_size=None,
            n_generic=0,
            n_eliciting=state.n_eliciting,
            control=True,
        )
        judged_transfer = [r for r in judged_all if r["eval_rung"] != "train"]
        judged_base = _clone_base_rows(
            base_rows,
            config="judged_only",
            map_variant="true",
            u_size=None,
            n_map=state.n_eliciting,
            n_generic=0,
            n_eliciting=state.n_eliciting,
            control=True,
        )
        rows.extend(judged_base + [judged_train] + judged_transfer)
        diagnostics["judged_only|true"] = {
            "fit": _jsonable(judged_map.diagnostics),
            "per_fold": judged_fold_diags,
            "n_map_full_transfer": state.n_eliciting,
        }
        del judged_map, judged_all, judged_transfer
        gc.collect()

        _log(f"{behavior}: running full-U union fold-clean diagnostic")
        foldclean_train, foldclean_diags = _perfold_control_train_row(
            args, state, behavior, config="fold_clean_union_full"
        )
        base_train = [r for r in base_rows if r["eval_rung"] == "train"]
        rows.extend(
            _clone_base_rows(
                base_train,
                config="fold_clean_union_full",
                map_variant="true",
                u_size=FULL_GENERIC_N,
                n_map=state.n_generic + state.n_eliciting,
                n_generic=state.n_generic,
                n_eliciting=state.n_eliciting,
                control=True,
            )
            + [foldclean_train]
        )
        diagnostics["fold_clean_union_full|true"] = {
            "per_fold": foldclean_diags,
            "descriptive_only": True,
        }

    ood_rungs = sorted(
        {
            r["eval_rung"]
            for r in rows
            if r["arm"] == MAPPED_ARM
            and r["setting_group"] == "ood"
            and r["config"] in args.configs
        }
    )
    expected_ood = {"evil": 5, "sycophancy": 6, "hallucination": 2}[behavior]
    if len(ood_rungs) != expected_ood:
        raise RuntimeError(
            f"{behavior}: realized {len(ood_rungs)} OOD rungs {ood_rungs}, "
            f"expected {expected_ood}"
        )
    parity = _parity_gate(args, behavior, rows) if not args.pilot else {
        "status": "pilot_skipped",
        "reason": "pilot runs the generic-only representative cell",
    }
    _verify_input_shas(state.loaded.shas)
    elapsed = time.time() - started
    meta = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "commit": commit,
        "behavior": behavior,
        "seed": args.seed,
        "variant": args.variant,
        "regime": args.regime,
        "layers": layers,
        "frozen_layers": {
            arm: state.layers[int(idx)] for arm, idx in state.frozen.items()
        },
        "frozen_layer_source": state.frozen_source,
        "u_sizes": list(args.u_sizes),
        "configs": list(args.configs),
        "map_variants": list(args.map_variants),
        "full_generic_n": state.n_generic,
        "full_eliciting_n": state.n_eliciting,
        "ood_rungs": ood_rungs,
        "expected_ood_rungs": expected_ood,
        "ood_note": _jsonable(state.ood_note),
        "parity": parity,
        "whitening": {
            "fit_pool": "full union: all generic plus all trait-eliciting pairs",
            "n": state.n_generic + state.n_eliciting,
            "gamma": _jsonable(state.whitening.gamma),
            "frozen_across_every_map_fit": True,
        },
        "estimand": "D(U)=rho(arm7_map_ridge_pred)-rho(arm4_ridge_ctx)",
        "wall_s": round(elapsed, 3),
        "peak_rss_gib": round(_rss_gib(), 3),
        "pilot": bool(args.pilot),
    }
    payload = {
        "rows": rows,
        "n_rows": len(rows),
        "map_diagnostics": diagnostics,
        "meta": meta,
    }
    _atomic_json(summary_path, payload)
    _write_npz(out_dir / "frozen_scores.npz", rows)
    if args.pilot:
        production_map_fits = len(U_SIZES) * len(MAP_CONFIGS) * len(MAP_VARIANTS) + 11
        pilot_map_fits = len(args.u_sizes) * len(args.configs) * len(args.map_variants)
        _atomic_json(
            out_dir / "pilot_report.json",
            {
                "behavior": behavior,
                "seed": args.seed,
                "wall_s": round(elapsed, 3),
                "peak_rss_gib": round(_rss_gib(), 3),
                "pilot_map_fits": pilot_map_fits,
                "production_map_fits_per_behavior_seed": production_map_fits,
                "code_derived_multiplier": production_map_fits / pilot_map_fits,
                "projection_note": (
                    "linear multiplier is deliberately conservative; full ladder rungs "
                    "below U=18793 fit smaller matrices"
                ),
            },
        )
    _log(
        f"{behavior}/seed{args.seed}: complete in {elapsed / 60:.1f} min; "
        f"peak RSS {_rss_gib():.1f} GiB"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.issue1739_fits import _git_commit
    from scripts.issue1739_wcrung_arms import _assert_no_judge_modules, _git_tracked

    _assert_no_judge_modules("uladder entry")
    if args.import_check:
        _import_check()
        return 0
    layers = args.layers or list(range(args.n_layers))
    commit = _git_commit()
    failures = []
    for behavior in args.behaviors:
        out = args.out_root / behavior / f"seed{args.seed}" / "all_arms_spearman.json"
        if _git_tracked(out):
            raise SystemExit(f"refusing to overwrite git-tracked output: {out}")
        try:
            run_behavior(args, behavior, layers, commit=commit)
        except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append({"behavior": behavior, "error": f"{type(exc).__name__}: {exc}"})
            _log(f"{behavior}/seed{args.seed}: FAILED: {exc}")
    if failures:
        failure_path = args.out_root / f"failures_seed{args.seed}.json"
        _atomic_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "commit": commit,
                "seed": args.seed,
                "failures": failures,
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
