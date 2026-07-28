"""Phase-3 CLI for issue #1739: matched-budget arm grid over staged stores.

Two modes:

- ``--synthetic N`` — generate a tiny in-process synthetic slice and run the
  FULL engine end to end (whitening -> map -> 16 arms -> metrics -> outputs).
  This is the CLI's own smoke path (no network, no GPU, no staged data).
- real mode — assemble CellData from the round-B capture stores
  (``store_io.load_summaries`` layout), the judged DV dataset
  (``dv_build.write_dv_dataset`` payload), and an E1 extraction store, then
  run the FULL plan grid (round-2 C2/C3 fix): variants (context_end AND
  prefix_end) x extraction regimes (``--regimes e1 e2 e2p`` — E2/E2p are the
  REAL matched-pair / pooled contrasts over the per-rollout K-sample DV, via
  ``fits.matched_pair_split_weights``; never an E1 relabel) x U-ladder rungs
  (``--u-sizes``) x L budgets x draws x seeds, plus the §4b composition
  sub-grid (``--compose``: f_U x f_L at the L-anchors) and the Config A/B
  train/eval-split swap (``--config``). Arms 15/16 inputs inject via
  ``--text-emb`` / ``--text-features`` (npz keyed by context id).

Outputs under ``--out-root`` (default ``eval_results/issue_1739``):
``arm_results/percell/cells.jsonl`` (checkpoint-per-cell, resumable; unit
keys carry every output-affecting flag) + per-cell frozen-layer prediction
sidecars (``arm_results/percell/preds/*.npz`` — post-hoc within-stratum
reads) + ``arm_results/all_arms_spearman.json`` + ``map_diagnostics.json``
+ r_B direction npz files under ``--tensors-root`` (HF-bound analysis
tensors).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_fits.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

logger = logging.getLogger("issue1739_fits")

DEFAULT_OUT_ROOT = Path("eval_results/issue_1739")
DEFAULT_TENSORS_ROOT = Path("analysis_tensors/issue_1739")
VARIANTS = ("context_end", "prefix_end")
REGIMES = ("e1", "e2", "e2p")
CONFIG_SPLIT = {"config_a": "train", "config_b": "eval"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", default="synthetic", help="behavior slug (provenance)")
    ap.add_argument("--variant", choices=("both", *VARIANTS), default="both")
    ap.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=["e1"],
        help="extraction regimes to run (e2/e2p REQUIRE per-rollout scores in the DV dataset)",
    )
    ap.add_argument(
        "--config",
        choices=tuple(CONFIG_SPLIT),
        default="config_a",
        help="config_a: labeled table = split 'train'; config_b: split 'eval' (plan §4)",
    )
    ap.add_argument("--labeled-store", type=Path, help="round-B labeled capture store dir")
    ap.add_argument("--dv-json", type=Path, help="dv_build labeling.json for the behavior")
    ap.add_argument("--u-store", type=Path, help="staged #1092 U-pool store dir")
    ap.add_argument("--e1-store", type=Path, help="E1 extraction capture store dir (pos/neg rows)")
    ap.add_argument(
        "--u-sizes",
        nargs="+",
        default=["full"],
        help="U-ladder rungs: ints or 'full' (the realized fit pool; the plan's 50k "
        "nominal rung caps at the store's 18,793 fit rows)",
    )
    ap.add_argument("--budgets", type=int, nargs="+", default=[250])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--layers", type=int, nargs="+", default=None, help="layer subset (default: all)"
    )
    ap.add_argument("--arms", nargs="+", default=None, help="arm-slug subset (default: all)")
    ap.add_argument(
        "--compose",
        action="store_true",
        help="ALSO run the §4b composition sub-grid (f_U x f_L at the --budgets "
        "L-anchors; Config-A evil per plan — the dispatcher gates the behavior)",
    )
    ap.add_argument(
        "--compose-u-size",
        type=int,
        default=5000,
        help="U rung for composition cells (plan §4b names no rung; mid-ladder default)",
    )
    ap.add_argument(
        "--text-emb",
        type=Path,
        default=None,
        help="arm-15 sentence embeddings: npz with 'context_ids' + 'emb' arrays",
    )
    ap.add_argument(
        "--text-features",
        type=Path,
        default=None,
        help="arm-16 surface features: npz with 'context_ids' + 'features' arrays",
    )
    ap.add_argument(
        "--transfer",
        action="store_true",
        help="ALSO run the distribution-shift ladder leg (round-3 M-A): per plain-rung "
        "unit, refit the plan-§4 ladder arms on the FULL train cell and score the "
        "eval-split contexts per rung (requires --config config_a + eval-split DV rows)",
    )
    ap.add_argument(
        "--transfer-min-n",
        type=int,
        default=3,
        help="per-rung row floor for a transfer Spearman read (below: recorded skip; "
        "the smoke lowers it to 2 — its per-rung slice is 2 contexts)",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="§9 pilot gate: run ONE production-shape unit (max L, full U, first "
        "variant/regime) through the production path, write pilot_report.json, and "
        "exit rc=7 when the projected wall exceeds --pilot-abort-mult x --plan-wall-h",
    )
    ap.add_argument(
        "--plan-wall-h",
        type=float,
        default=2.0,
        help="plan §9 wall estimate (hours) the pilot gate compares against",
    )
    ap.add_argument(
        "--pilot-abort-mult",
        type=float,
        default=3.0,
        help="abort multiple (plan §9 registers 3x: 'If any pilot exceeds 3x the "
        "estimate above, abort and re-size')",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--tensors-root", type=Path, default=DEFAULT_TENSORS_ROOT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mlp-epochs", type=int, default=None, help="override arm-5 MLP epochs")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument(
        "--synthetic", type=int, default=0, help="run the synthetic engine smoke with N contexts"
    )
    ap.add_argument("--synthetic-dim", type=int, default=8)
    ap.add_argument("--synthetic-layers", type=int, default=3)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# grid composition (pure — unit-tested)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One (variant, U-spec, regime) slice of the plan grid.

    ``u_size`` None = the full realized fit pool. ``f_u``/``f_l`` are the §4b
    composition factors (None = a plain ladder rung, no composition);
    ``budgets`` is the L set this slice runs (composition cells run ONE
    anchor with draw 0 / the first seed — the reference cell whose contexts
    the f_l axis includes/excludes).
    """

    variant: str
    regime: str
    u_size: int | None
    f_u: float | None = None
    f_l: float | None = None
    budgets: tuple[int, ...] = ()
    draws: tuple[int, ...] = ()
    seeds: tuple[int, ...] = ()


def compose_run_specs(
    *,
    variants: tuple[str, ...],
    regimes: tuple[str, ...],
    u_sizes: tuple[int | None, ...],
    budgets: tuple[int, ...],
    draws: tuple[int, ...],
    seeds: tuple[int, ...],
    compose: bool = False,
    compose_u_size: int = 5000,
    f_u_grid: tuple[float, ...] = (),
    f_l_grid: tuple[float, ...] = (),
) -> list[RunSpec]:
    """Enumerate the full plan grid as RunSpecs (C2 — consumed by run_grid).

    Base grid: every variant x regime x U rung runs the full budgets x draws
    x seeds block. Composition (``compose=True``): per variant, the HEADLINE
    regime slot (first of ``regimes``) additionally runs f_U x f_L combos at
    each L-anchor in ``budgets`` (draw 0, first seed — the deterministic
    reference cell). f_u == 0 combos are composition-degenerate (no eliciting
    rows, f_l moot) and run once with f_l recorded as 0.0.
    """
    specs: list[RunSpec] = []
    for variant in variants:
        for u_size in u_sizes:
            for regime in regimes:
                specs.append(
                    RunSpec(
                        variant=variant,
                        regime=regime,
                        u_size=u_size,
                        budgets=budgets,
                        draws=draws,
                        seeds=seeds,
                    )
                )
        if compose:
            seen: set[tuple[float, float]] = set()
            for f_u in f_u_grid:
                for f_l in f_l_grid:
                    key = (f_u, f_l if f_u > 0 else 0.0)
                    if key in seen:
                        continue  # f_u=0 is composition-degenerate across f_l
                    seen.add(key)
                    for anchor in budgets:
                        specs.append(
                            RunSpec(
                                variant=variant,
                                regime=regimes[0],
                                u_size=compose_u_size,
                                f_u=key[0],
                                f_l=key[1],
                                budgets=(anchor,),
                                draws=(draws[0],),
                                seeds=(seeds[0],),
                            )
                        )
    return specs


# ---------------------------------------------------------------------------
# synthetic smoke mode
# ---------------------------------------------------------------------------


def _make_synthetic(n: int, n_layers: int, d: int, seed: int = 0):
    """Tiny synthetic slice with a real planted signal (engine smoke)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rb = rng.normal(size=(n_layers, d))
    x_ctx = rng.normal(size=(n_layers, n, d))
    y_ans = 0.7 * x_ctx + 0.3 * rng.normal(size=(n_layers, n, d))
    dv = 50.0 + 12.0 * np.einsum("lnd,ld->n", y_ans, rb) / (n_layers * np.sqrt(d))
    dv = np.clip(dv + rng.normal(scale=4.0, size=n), 0, 100)
    groups = [f"g{i % max(4, n // 6):02d}" for i in range(n)]
    x_u = rng.normal(size=(n_layers, max(4 * d, 64), d))
    y_u = 0.7 * x_u + 0.3 * rng.normal(size=x_u.shape)
    return x_ctx, y_ans, dv, groups, rb, x_u, y_u


def _run_synthetic(args: argparse.Namespace) -> int:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    n, n_layers, d = args.synthetic, args.synthetic_layers, args.synthetic_dim
    x_ctx, y_ans, dv, groups, rb_raw, x_u, y_u = _make_synthetic(n, n_layers, d)
    wh = fits.fit_whitening(x_u, device=args.device)
    z_u, zy_u = fits.apply_whitening(x_u, wh), fits.apply_whitening(y_u, wh)
    mapfit = fits.fit_linear_map(z_u, zy_u, device=args.device)
    rb = np.einsum("ld,lde->le", rb_raw, wh.w)
    data = arms.CellData(
        z_ctx=fits.apply_whitening(x_ctx, wh),
        z_ans=fits.apply_whitening(y_ans, wh),
        dv=dv,
        rb=rb,
        mapfit=mapfit,
        text_emb=np.random.default_rng(1).normal(size=(n, 16)),
        text_features=np.random.default_rng(2).normal(size=(n, 4)),
        layers=tuple(range(n_layers)),
    )
    provenance = {
        "behavior": args.behavior,
        "variant": "context_end",
        "regime": args.regimes[0],
        "u_rung": int(x_u.shape[1]),
        "eval_rung": "synthetic",
        "config": "synthetic",
        "f_u": None,
        "f_l": None,
    }
    kwargs = {}
    if args.n_boot:
        kwargs["n_boot"] = args.n_boot
    if args.n_perm:
        kwargs["n_perm"] = args.n_perm
    records = arms.run_grid(
        data,
        groups,
        budgets=args.budgets,
        draws=args.draws,
        seeds=args.seeds,
        provenance=provenance,
        out_dir=args.out_root / "arm_results",
        arms=args.arms,
        device=args.device,
        mlp_kwargs={"max_epochs": args.mlp_epochs or 5, "hidden": 16},
        context_ids=[f"synctx{i:04d}" for i in range(n)],
        **kwargs,
    )
    arms.write_summary(
        records,
        args.out_root / "arm_results" / "all_arms_spearman.json",
        meta={"mode": "synthetic", "n": n, "d": d, "n_layers": n_layers},
    )
    (args.out_root / "map_diagnostics.json").write_text(json.dumps(mapfit.diagnostics, indent=1))
    print(f"[fits] synthetic smoke done: {len(records)} cells", flush=True)
    return 0


# ---------------------------------------------------------------------------
# real mode: labeled table + directions + U specs
# ---------------------------------------------------------------------------


def _meta_field(meta_rows: list[dict], candidates: tuple[str, ...], what: str) -> str:
    keys = set(meta_rows[0])
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"no {what} field among {candidates} in store row_index keys={sorted(keys)}")


@dataclasses.dataclass
class LabeledTable:
    """Per-context labeled table + per-ROLLOUT answer rows (E2/E2p inputs)."""

    z_by_variant: dict
    z_ans: object  # (Ly, n, d) mean-over-rollouts answer acts
    dv: object  # (n,)
    groups: list[str]
    per_rollout: object | None  # (n, K) per-rollout mean judge scores (NaN = dropped)
    ctx_order: list[str]
    rungs: list[str]
    ans_rows: dict | None  # {layer: (n_rows, d)} per-rollout t1 rows (e2/e2p only)
    ans_row_ctx: object | None  # (n_rows,) index into ctx_order
    ans_row_k: object | None  # (n_rows,) rollout k
    # per-context rung label aligned with ctx_order (M-A ladder; defaults keep
    # pre-round-3 constructor calls — e.g. test fixtures — valid).
    row_rungs: list[str] = dataclasses.field(default_factory=list)


def _load_labeled(
    store_dir: Path,
    dv_json: Path,
    layers: list[int],
    *,
    config: str,
    need_rollout_rows: bool,
) -> LabeledTable:
    """Round-B labeled store + DV dataset -> per-CONTEXT layer-leading arrays.

    Store rows are per-ROLLOUT; context/prefix acts are identical across a
    context's rollouts (first occurrence taken), the answer act is the MEAN
    over the context's rollout t1 rows. Contexts without a kept DV are
    dropped (drop-never-coerce). The labeled table is FILTERED to the
    config's split (config_a -> 'train', config_b -> 'eval' — plan §4 Config
    A/B; the round-1 review's split-pooling gap). group_key is REQUIRED on
    every kept row (LOFO folds are load-bearing — fail loud, never a silent
    per-context fallback).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    kinds = ("prefix_end", "context_end", "t1")
    arrays, meta = store_io.load_summaries(
        store_dir, kinds, tuple(layers), hidden_dim=arrays_dim(store_dir, layers)
    )
    ctx_key = _meta_field(meta, ("context_id",), "context id")
    dv_payload = json.loads(Path(dv_json).read_text())
    want_split = CONFIG_SPLIT[config]
    all_rows = dv_payload["rows"]
    n_with_split = sum(1 for r in all_rows if r.get("split") is not None)
    if n_with_split == 0:
        raise RuntimeError(
            f"--config {config} needs 'split' on DV rows; 0/{len(all_rows)} carry it "
            "(stage_corpus attaches split at staging — regenerate the DV dataset)"
        )
    dv_by_ctx = {
        r["context_id"]: r
        for r in all_rows
        if r.get("dv") is not None and r.get("split") == want_split
    }
    if not dv_by_ctx:
        raise RuntimeError(f"no DV rows with split={want_split!r} for config {config}")
    missing_group = [c for c, r in dv_by_ctx.items() if not r.get("group_key")]
    if missing_group:
        raise RuntimeError(
            f"{len(missing_group)}/{len(dv_by_ctx)} DV rows lack group_key (LOFO folds are "
            f"load-bearing; first offenders: {sorted(missing_group)[:5]})"
        )
    ctx_order: list[str] = []
    rows_by_ctx: dict[str, list[int]] = {}
    for i, r in enumerate(meta):
        cid = r[ctx_key]
        if cid not in dv_by_ctx:
            continue
        if cid not in rows_by_ctx:
            ctx_order.append(cid)
        rows_by_ctx.setdefault(cid, []).append(i)
    if not ctx_order:
        raise RuntimeError("no labeled contexts join the DV dataset and the store")
    first = np.array([rows_by_ctx[c][0] for c in ctx_order])
    z_ctx = np.stack([arrays[("context_end", ly)][first] for ly in layers])
    z_pre = np.stack([arrays[("prefix_end", ly)][first] for ly in layers])
    z_ans = np.stack(
        [
            np.stack([arrays[("t1", ly)][rows_by_ctx[c]].mean(axis=0) for c in ctx_order])
            for ly in layers
        ]
    )
    dv = np.array([dv_by_ctx[c]["dv"] for c in ctx_order], dtype=float)
    groups = [str(dv_by_ctx[c]["group_key"]) for c in ctx_order]
    row_rungs = [str(dv_by_ctx[c].get("rung")) for c in ctx_order]
    rungs = sorted(set(row_rungs))

    per_rollout = None
    if all("per_rollout_scores" in dv_by_ctx[c] for c in ctx_order):
        k_max = 1 + max(int(k[1:]) for c in ctx_order for k in dv_by_ctx[c]["per_rollout_scores"])
        per_rollout = np.full((len(ctx_order), k_max), np.nan)
        for i, c in enumerate(ctx_order):
            for key, s in dv_by_ctx[c]["per_rollout_scores"].items():
                per_rollout[i, int(key[1:])] = np.nan if s is None else float(s)

    ans_rows = ans_row_ctx = ans_row_k = None
    if need_rollout_rows:
        k_field = _meta_field(meta, ("rollout_k",), "rollout index")
        ctx_pos = {c: i for i, c in enumerate(ctx_order)}
        sel, sel_ctx, sel_k = [], [], []
        for i, r in enumerate(meta):
            cid = r[ctx_key]
            if cid in ctx_pos and r.get(k_field) is not None:
                sel.append(i)
                sel_ctx.append(ctx_pos[cid])
                sel_k.append(int(r[k_field]))
        sel = np.asarray(sel)
        ans_rows = {ly: np.asarray(arrays[("t1", ly)][sel], dtype=np.float64) for ly in layers}
        ans_row_ctx = np.asarray(sel_ctx, dtype=np.int64)
        ans_row_k = np.asarray(sel_k, dtype=np.int64)

    return LabeledTable(
        z_by_variant={"context_end": z_ctx, "prefix_end": z_pre},
        z_ans=z_ans,
        dv=dv,
        groups=groups,
        per_rollout=per_rollout,
        ctx_order=ctx_order,
        rungs=rungs,
        row_rungs=row_rungs,
        ans_rows=ans_rows,
        ans_row_ctx=ans_row_ctx,
        ans_row_k=ans_row_k,
    )


def arrays_dim(store_dir: Path, layers: list[int]):
    """Infer hidden dim from the first summary shard (tiny-real stores differ)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    paths = store_io._summary_shard_paths(Path(store_dir), "context_end", layers[0])
    if not paths:
        raise FileNotFoundError(f"no context_end shards under {store_dir}")
    return int(np.load(paths[0]).shape[1])


def _load_rb_e1(e1_store: Path, layers: list[int], dim: int):
    """E1 extraction store -> raw diff-of-means direction (Ly, d)."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io

    arrays, meta = store_io.load_summaries(e1_store, ("t1",), tuple(layers), hidden_dim=dim)
    side_key = _meta_field(meta, ("side", "polarity", "pv_side", "pair_side"), "pos/neg side")
    sides = np.array([str(r[side_key]).lower() for r in meta])
    pos_rows = np.flatnonzero(np.isin(sides, ("pos", "positive")))
    neg_rows = np.flatnonzero(np.isin(sides, ("neg", "negative")))
    if len(pos_rows) == 0 or len(neg_rows) == 0:
        raise RuntimeError(f"E1 store has {len(pos_rows)} pos / {len(neg_rows)} neg rows")
    acts = np.stack([arrays[("t1", ly)] for ly in layers], axis=1)  # (n, Ly, d)
    return fits.extract_rb_e1(acts[pos_rows], acts[neg_rows])


def _extract_rb(regime: str, args: argparse.Namespace, tbl: LabeledTable, layers, dim):
    """Regime-keyed RAW direction (Ly, d) — C3: e2/e2p are REAL contrasts.

    e1: diff-of-means over the E1 extraction store's judge-filtered pos/neg
    rollouts. e2/e2p: matched-pair (within-context midpoint split over the
    per-rollout K-sample judge scores; qualification = spread >=
    E2_SPREAD_MIN) / pooled global-midpoint contrasts computed over the
    labeled store's per-ROLLOUT t1 rows via the tested
    ``matched_pair_split_weights`` (one mask-GEMM per layer — never a
    (n_ctx, K, Ly, d) materialization). The whitening applied downstream
    makes the diff-of-means the LDA direction under the U-pool covariance.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits
    from explore_persona_space.experiments.issue_1739.constants import E2_SPREAD_MIN

    if regime == "e1":
        return _load_rb_e1(args.e1_store, layers, dim)
    if tbl.per_rollout is None or tbl.ans_rows is None:
        raise SystemExit(
            f"--regimes {regime} requires per-rollout judge scores + per-rollout t1 rows "
            f"(behavior {args.behavior!r} has none — run it with --regimes e1)"
        )
    w_hi, w_lo, n_qual = fits.matched_pair_split_weights(
        np.asarray(tbl.per_rollout, dtype=float),
        spread_min=E2_SPREAD_MIN,
        pooled=(regime == "e2p"),
    )
    w_row = (w_hi - w_lo)[tbl.ans_row_ctx, tbl.ans_row_k]  # (n_rows,)
    rb = np.stack([w_row @ tbl.ans_rows[ly] for ly in layers])
    logger.info(
        "[fits] %s direction: %d qualifying contexts over %d rollout rows",
        regime,
        n_qual,
        len(w_row),
    )
    return rb


def _load_injected_features(path: Path | None, array_key: str, ctx_order: list[str], what: str):
    """Arms-15/16 injected features: npz {'context_ids', <array_key>} -> (n, f).

    Fails loud on a context-coverage miss (an injected-feature table that
    silently covers a subset would smuggle a selection into the control arm).
    Returns None when no path is given (the arm records a SKIP reason).
    """
    import numpy as np

    if path is None:
        return None
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in z["context_ids"]]
        feats = np.asarray(z[array_key], dtype=np.float64)
    pos = {c: i for i, c in enumerate(ids)}
    missing = [c for c in ctx_order if c not in pos]
    if missing:
        raise RuntimeError(
            f"{what}: {len(missing)}/{len(ctx_order)} labeled contexts missing from {path} "
            f"(first: {missing[:5]})"
        )
    return feats[[pos[c] for c in ctx_order]]


def _save_rb(tensors_root: Path, behavior: str, regime: str, rb, layers) -> None:
    """Persist the raw regime direction (HF-bound analysis tensor; fp16)."""
    import os

    import numpy as np

    out_dir = tensors_root / f"r_b_{regime}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{behavior}.tmp.npz"  # np.savez appends .npz to non-.npz names (#1092)
    with tmp.open("wb") as fh:
        np.savez(fh, rb=np.asarray(rb, dtype=np.float16), layers=np.asarray(layers))
    os.replace(tmp, out_dir / f"{behavior}.npz")


def _git_commit() -> str:
    """Repo HEAD for reproducibility metadata (soft: 'unknown' off-repo)."""
    import subprocess

    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _save_map(tensors_root: Path, variant: str, u_label: str, mapfit, layers) -> Path:
    """Persist a frozen plain-rung map (plan §10 ``maps/`` class — round-3 C-1).

    fp16 W + fp32 standardization params + meta under ``tensors_root/maps/``;
    the upload ``--stage tensors`` sweeps the whole tree (no eligibility
    filter), so the class needs zero further upload wiring. Plain U-ladder
    rungs only: the (variant, u_label) map is BEHAVIOR-INDEPENDENT (shared
    #1092 fit pool, shared subsample + whitening seeds), so the filename
    omits the behavior and a sibling behavior's identical fit SKIPS on
    existence (idempotent). Composition-cell maps are NOT persisted — they
    are behavior+anchor-specific (~0.7 GB each x ~30 combos) and
    deterministically regenerable from the pinned store + seeded code (the
    §10 discard economy; noted in the results payload plan_deviations).
    """
    import os
    import time as _time

    import numpy as np

    out_dir = Path(tensors_root) / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{variant}__u{u_label}.npz"
    if out.exists():
        return out
    meta = {
        "variant": variant,
        "u_label": u_label,
        "layers": [int(x) for x in layers],
        "w_fit_rows": mapfit.diagnostics.get("w_fit_rows"),
        "solver": mapfit.diagnostics.get("solver"),
        "dtype": "w=fp16; mu/sd=fp32",
        "apply": "pred = ((x - x_mu)/x_sd) @ w + y_mu (whitened space)",
        "git_commit": _git_commit(),
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    tmp = out_dir / f"{out.stem}.tmp.npz"  # keep the .npz suffix (#1092 savez trap)
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            w=np.asarray(mapfit.w, dtype=np.float16),
            x_mu=np.asarray(mapfit.x_mu, dtype=np.float32),
            x_sd=np.asarray(mapfit.x_sd, dtype=np.float32),
            y_mu=np.asarray(mapfit.y_mu, dtype=np.float32),
            layers=np.asarray([int(x) for x in layers]),
            meta=np.asarray(json.dumps(meta)),
        )
    os.replace(tmp, out)
    logger.info("[fits] map weights persisted -> %s", out)
    return out


def _record_compose_skip(spec: RunSpec, exc: Exception, compose_skips: list[dict]) -> None:
    """Round-3 Minor 1: a ``_u_pool_for_spec`` failure is a recordable skip
    ONLY for a composition spec (a quota that cannot fill); on a PLAIN ladder
    rung it RE-RAISES — a genuine load bug must never drop a whole
    (variant, U, regime) slice from the summary as a 'compose skip'."""
    if spec.f_u is None:
        raise exc
    compose_skips.append({"spec": dataclasses.asdict(spec), "reason": str(exc)})


PILOT_ABORT_RC = 7  # designed-halt rc (never a bare rc=1 — gotchas.md pilot-gate entry)


def compose_pilot_report(
    *,
    n_map_fits: int,
    n_units: int,
    n_transfer_units: int,
    map_fit_s: float,
    unit_s: float,
    transfer_s: float,
    plan_wall_h: float,
    abort_mult: float,
) -> dict:
    """Pure §9 pilot fence math (unit-tested; the CLI adds provenance fields).

    ``projected_wall_h`` extrapolates the MEASURED single-unit walls across
    the full grid (conservative: the pilot unit runs at max L / full U);
    ``fence_wall_h`` is the >=2x fence the experimenter sizes timeouts from;
    ``abort`` fires when the projection exceeds ``abort_mult x plan_wall_h``
    (plan §9: 'If any pilot exceeds 3x the estimate above, abort and
    re-size').
    """
    projected_s = n_map_fits * map_fit_s + n_units * unit_s + n_transfer_units * transfer_s
    projected_h = projected_s / 3600.0
    return {
        "n_map_fits": int(n_map_fits),
        "n_units": int(n_units),
        "n_transfer_units": int(n_transfer_units),
        "map_fit_s": float(map_fit_s),
        "unit_s": float(unit_s),
        "transfer_unit_s": float(transfer_s),
        "projected_wall_h": float(projected_h),
        "fence_wall_h": float(2.0 * projected_h),
        "plan_wall_h": float(plan_wall_h),
        "abort_mult": float(abort_mult),
        "abort": bool(projected_h > abort_mult * plan_wall_h),
    }


def _run_real(args: argparse.Namespace, timings: dict | None = None) -> int:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
    from explore_persona_space.experiments.issue_1739.constants import (
        COMPOSITION_F_L,
        COMPOSITION_F_U,
        N_LAYERS,
    )

    for req in ("labeled_store", "dv_json", "u_store", "e1_store"):
        if getattr(args, req) is None:
            raise SystemExit(f"real mode requires --{req.replace('_', '-')} (or use --synthetic N)")
    layers = args.layers or list(range(N_LAYERS))
    need_rollout_rows = any(r in ("e2", "e2p") for r in args.regimes)
    tbl = _load_labeled(
        args.labeled_store,
        args.dv_json,
        layers,
        config=args.config,
        need_rollout_rows=need_rollout_rows,
    )
    # M-A ladder leg: the eval-split labeled table (same store + DV dataset,
    # split 'eval') scored by TRAIN-frozen predictors per rung.
    tbl_ev = rungs_ev = None
    if args.transfer:
        if args.config != "config_a":
            raise SystemExit("--transfer requires --config config_a (train -> eval transfer)")
        tbl_ev = _load_labeled(
            args.labeled_store,
            args.dv_json,
            layers,
            config="config_b",  # CONFIG_SPLIT['config_b'] == the 'eval' split
            need_rollout_rows=False,
        )
        rungs_ev = np.asarray(tbl_ev.row_rungs)
    dim = tbl.z_ans.shape[-1]
    text_emb = _load_injected_features(args.text_emb, "emb", tbl.ctx_order, "--text-emb")
    text_features = _load_injected_features(
        args.text_features, "features", tbl.ctx_order, "--text-features"
    )

    # Stage the #1092 U-pool slice on demand (idempotent: a dest that is
    # already loadable for the requested (kinds x layers) regime — incl. a
    # LOCAL capture-store stand-in — short-circuits without network).
    store_io.stage_u_store(Path(args.u_store), ("prefix_end", "context_end", "t1"), tuple(layers))
    u_arrays, u_meta = store_io.load_summaries(
        args.u_store, ("prefix_end", "context_end", "t1"), tuple(layers), hidden_dim=dim
    )
    u_fit_rows = np.flatnonzero(store_io.fit_pool_mask(u_meta))  # is_eval_only exclusion

    u_sizes: list[int | None] = []
    for tok in args.u_sizes:
        u_sizes.append(None if str(tok).lower() == "full" else int(tok))
    specs = compose_run_specs(
        variants=tuple(VARIANTS) if args.variant == "both" else (args.variant,),
        regimes=tuple(args.regimes),
        u_sizes=tuple(u_sizes),
        budgets=tuple(args.budgets),
        draws=tuple(args.draws),
        seeds=tuple(args.seeds),
        compose=args.compose,
        compose_u_size=args.compose_u_size,
        f_u_grid=tuple(COMPOSITION_F_U),
        f_l_grid=tuple(COMPOSITION_F_L),
    )
    print(f"[fits] plan grid: {len(specs)} (variant x U x regime) slices", flush=True)

    all_records: list[dict] = []
    transfer_rows: list[dict] = []
    transfer_skips: list[dict] = []
    diag_out: dict = {}
    rb_cache: dict[str, np.ndarray] = {}
    compose_skips: list[dict] = []
    prev_map_key: tuple | None = None
    wh = mapfit = None  # reused across consecutive same-(variant, U) regime slices
    z_ev_w = za_ev_w = None  # eval-split arrays, whitened per map_key (transfer leg)
    for si, spec in enumerate(specs):
        map_key = (spec.variant, spec.u_size, spec.f_u, spec.f_l, spec.budgets, spec.seeds)
        if map_key != prev_map_key:
            try:
                u_x, u_y, u_label, n_u = _u_pool_for_spec(spec, u_arrays, u_fit_rows, tbl, layers)
            except (ValueError, RuntimeError) as exc:
                _record_compose_skip(spec, exc, compose_skips)  # re-raises on a plain rung
                print(f"[fits] slice {si + 1}/{len(specs)} SKIP compose: {exc}", flush=True)
                prev_map_key = None
                continue
            t_map = time.time()
            wh = fits.fit_whitening(u_x, device=args.device, seed=args.seeds[0])
            mapfit = fits.fit_linear_map(
                fits.apply_whitening(u_x, wh), fits.apply_whitening(u_y, wh), device=args.device
            )
            if timings is not None:
                timings.setdefault("map_fit_s", []).append(time.time() - t_map)
            del u_x, u_y  # the map + whitening carry everything downstream needs
            diag_out[f"{spec.variant}|{u_label}"] = mapfit.diagnostics
            if spec.f_u is None:
                # C-1: persist the frozen plain-rung map weights (HF-bound via
                # the tensors upload stage; behavior-independent, idempotent).
                _save_map(args.tensors_root, spec.variant, u_label, mapfit, layers)
                if tbl_ev is not None:
                    z_ev_w = fits.apply_whitening(tbl_ev.z_by_variant[spec.variant], wh)
                    za_ev_w = (
                        fits.apply_whitening(tbl_ev.z_ans, wh) if tbl_ev.z_ans is not None else None
                    )
            prev_map_key = map_key
        if spec.regime not in rb_cache:
            rb_cache[spec.regime] = _extract_rb(spec.regime, args, tbl, layers, dim)
            _save_rb(args.tensors_root, args.behavior, spec.regime, rb_cache[spec.regime], layers)
        data = arms.CellData(
            z_ctx=fits.apply_whitening(tbl.z_by_variant[spec.variant], wh),
            z_ans=fits.apply_whitening(tbl.z_ans, wh),
            dv=tbl.dv,
            rb=np.einsum("ld,lde->le", rb_cache[spec.regime], wh.w),
            mapfit=mapfit,
            text_emb=text_emb,
            text_features=text_features,
            layers=tuple(layers),
            per_rollout=tbl.per_rollout,
        )
        provenance = {
            "behavior": args.behavior,
            "variant": spec.variant,
            "regime": spec.regime,
            "u_rung": int(n_u),
            "u_rung_label": u_label,
            "eval_rung": ",".join(tbl.rungs),
            "config": args.config,
            "f_u": spec.f_u,
            "f_l": spec.f_l,
        }
        kwargs = {}
        if args.n_boot:
            kwargs["n_boot"] = args.n_boot
        if args.n_perm:
            kwargs["n_perm"] = args.n_perm
        mlp_kwargs = {"max_epochs": args.mlp_epochs} if args.mlp_epochs else None
        t_grid = time.time()
        recs = arms.run_grid(
            data,
            tbl.groups,
            budgets=list(spec.budgets),
            draws=list(spec.draws),
            seeds=list(spec.seeds),
            provenance=provenance,
            out_dir=args.out_root / "arm_results",
            arms=args.arms,
            device=args.device,
            mlp_kwargs=mlp_kwargs,
            context_ids=tbl.ctx_order,
            **kwargs,
        )
        if timings is not None:
            timings.setdefault("grid_s", []).append(time.time() - t_grid)
        all_records += recs
        if tbl_ev is not None and spec.f_u is None:
            t_tf = time.time()
            rows_s, skips_s = _run_transfer_for_spec(
                args,
                spec,
                data,
                tbl,
                tbl_ev,
                rungs_ev,
                z_ev_w,
                za_ev_w,
                recs,
                provenance,
                layers,
            )
            transfer_rows += rows_s
            transfer_skips += skips_s
            if timings is not None:
                timings.setdefault("transfer_s", []).append(time.time() - t_tf)
        print(
            f"[fits] slice {si + 1}/{len(specs)} done ({spec.variant}/{u_label}/{spec.regime})",
            flush=True,
        )
    extra = None
    if tbl_ev is not None:
        extra = {"transfer_rows": transfer_rows, "transfer_skips": transfer_skips}
    arms.write_summary(
        all_records,
        args.out_root / "arm_results" / "all_arms_spearman.json",
        meta={
            "mode": "real",
            "behavior": args.behavior,
            "config": args.config,
            "regimes": list(args.regimes),
            "n_contexts": len(tbl.ctx_order),
            "layers": layers,
            "u_fit_rows": int(len(u_fit_rows)),
            "u_sizes": [s if s is not None else "full" for s in u_sizes],
            "eval_rungs": tbl.rungs,
            "compose_skips": compose_skips,
            "transfer_min_n": int(args.transfer_min_n) if tbl_ev is not None else None,
            "transfer_eval_rungs": sorted(tbl_ev.rungs) if tbl_ev is not None else None,
        },
        extra=extra,
    )
    (args.out_root / "map_diagnostics.json").write_text(json.dumps(diag_out, indent=1))
    print(f"[fits] real grid done: {len(all_records)} cells", flush=True)
    return 0


def _u_pool_for_spec(spec: RunSpec, u_arrays, u_fit_rows, tbl: LabeledTable, layers):
    """Realize one spec's U pool: (x_u, y_u, label, n_rows).

    Plain rung: a seeded subsample of the store's fit pool (or the whole pool
    for 'full'). Composition (§4b): ``f_u`` of the pool is drawn from the
    BEHAVIOR-eliciting labeled table (x = the variant act, y = the t1 answer
    act); ``f_l`` toggles whether that eliciting half may include the
    reference anchor cell's contexts (f_l=1) or must come from labeled
    contexts OUTSIDE that cell (f_l=0 — the no-overlap-with-L reading of
    plan §4b's 'held-out unlabeled pool'). Fails loud (ValueError) when a
    quota cannot be filled — the caller records the skip.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    def stack(rows):
        x = np.stack([u_arrays[(spec.variant, ly)][rows] for ly in layers])
        y = np.stack([u_arrays[("t1", ly)][rows] for ly in layers])
        return x, y

    if spec.f_u is None:  # plain ladder rung
        rows = u_fit_rows
        if spec.u_size is not None and spec.u_size < len(rows):
            rng = np.random.default_rng([1739, 9, int(spec.seeds[0]) if spec.seeds else 0])
            rows = np.sort(rng.choice(rows, size=spec.u_size, replace=False))
        x, y = stack(rows)
        label = "full" if spec.u_size is None else str(spec.u_size)
        return x, y, label, len(rows)

    # Composition cell: generic half from the store, eliciting half from the
    # labeled table (f_l gates overlap with the reference anchor cell).
    size = int(spec.u_size or 0)
    anchor = spec.budgets[0]
    seed = spec.seeds[0]
    cell = fits.realize_budget_cell(tbl.groups, budget_l=anchor, draw=spec.draws[0], seed=seed)
    in_cell = np.zeros(len(tbl.ctx_order), dtype=bool)
    in_cell[cell.row_idx] = True
    elic_pool = np.arange(len(tbl.ctx_order)) if spec.f_l >= 1.0 else np.flatnonzero(~in_cell)
    gen_sel, elic_sel = fits.compose_u_pool(
        len(u_fit_rows), len(elic_pool), f_u=spec.f_u, size=size, seed=seed
    )
    gen_rows = u_fit_rows[gen_sel]
    x_gen, y_gen = stack(gen_rows)
    if len(elic_sel):
        elic_rows = elic_pool[elic_sel]
        x_elic = np.asarray(tbl.z_by_variant[spec.variant][:, elic_rows], dtype=x_gen.dtype)
        y_elic = np.asarray(tbl.z_ans[:, elic_rows], dtype=y_gen.dtype)
        x = np.concatenate([x_gen, x_elic], axis=1)
        y = np.concatenate([y_gen, y_elic], axis=1)
    else:
        x, y = x_gen, y_gen
    label = f"compose{size}_fu{spec.f_u}_fl{spec.f_l}_L{anchor}"
    return x, y, label, x.shape[1]


def _run_transfer_for_spec(
    args: argparse.Namespace,
    spec: RunSpec,
    data,
    tbl: LabeledTable,
    tbl_ev: LabeledTable,
    rungs_ev,
    z_ev_w,
    za_ev_w,
    recs: list[dict],
    provenance: dict,
    layers: list[int],
) -> tuple[list[dict], list[dict]]:
    """Distribution-shift ladder leg for ONE plain-rung spec (round-3 M-A).

    Per (L, draw, seed) unit: refit the :data:`arms.TRANSFER_ARMS` on the
    FULL train cell (``run_transfer_cell`` — never on eval DV), score every
    eval-split context, and emit one row per (arm, eval rung) at the
    TRAIN-frozen layer, plus one ``train_in_split`` row per arm carrying the
    unit's own in-split OOF rho (the ladder's in-distribution anchor).
    Checkpoint-per-unit (JSONL, resume keyed on every output-affecting
    flag — #722-r3) + one progress line per unit.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    tpath = args.out_root / "arm_results" / "percell" / "transfer.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tdone: dict[str, dict] = {}
    if tpath.exists():
        with tpath.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    tdone[rec["unit_key"]] = rec
    n_boot = int(args.n_boot) if args.n_boot else arms.N_BOOT
    regime_extra = {
        "transfer": True,
        "transfer_min_n": int(args.transfer_min_n),
        "transfer_arms": sorted(arms.TRANSFER_ARMS),
        "n_eval_table": len(tbl_ev.ctx_order),
        "n_boot": n_boot,
        "layers_subset": [int(x) for x in layers],
    }
    rec_by_unit: dict[tuple, dict] = {}
    for rec in recs:
        rows = rec.get("arms") or []
        if rows:
            r0 = rows[0]
            rec_by_unit[(r0["budget_l"], r0["draw"], r0["seed"])] = rec
    units = [(b, d, s) for b in spec.budgets for d in spec.draws for s in spec.seeds]
    rows_all: list[dict] = []
    skips_all: list[dict] = []
    t0 = time.time()
    for k, (budget_l, draw, seed) in enumerate(units):
        key = "transfer|" + arms._unit_key(provenance, budget_l, draw, seed, regime_extra)
        if key in tdone:
            rows_all += tdone[key]["rows"]
            skips_all += tdone[key].get("skips", [])
            print(
                f"[fits] transfer unit {k + 1}/{len(units)} SKIP (resume) {budget_l}/{draw}/{seed}",
                flush=True,
            )
            continue

        cell = fits.realize_budget_cell(tbl.groups, budget_l=budget_l, draw=draw, seed=seed)
        rec = rec_by_unit.get((budget_l, draw, seed))
        if rec is None:
            raise RuntimeError(
                f"transfer unit {budget_l}/{draw}/{seed}: no matching train record "
                "(main grid must run the same units first)"
            )
        frozen_by_arm = {
            row["arm"]: arms.frozen_layer_idx(row["rho_per_layer"]) for row in rec["arms"]
        }
        scores_ev, arm_skips = arms.run_transfer_cell(
            data,
            cell,
            z_ev_w,
            np.asarray(tbl_ev.dv, dtype=np.float64),
            za_ev=za_ev_w,
            device=args.device,
        )
        rows_u, skips_u = arms.evaluate_transfer(
            scores_ev,
            tbl_ev.dv,
            rungs_ev,
            frozen_by_arm,
            provenance=provenance,
            cell=cell,
            layers=tuple(layers),
            n_boot=n_boot,
            min_n=int(args.transfer_min_n),
        )
        skips_u += [
            {"arm": slug, "reason": reason, "budget_l": budget_l, "draw": draw, "seed": seed}
            for slug, reason in sorted(arm_skips.items())
        ]
        # In-distribution anchor: the unit's own in-split OOF read per ladder arm.
        for row in rec["arms"]:
            if row["arm"] in scores_ev:
                rows_u.append(
                    {**row, "rung_kind": "train_in_split", "n_eval": int(len(cell.row_idx))}
                )
        line = json.dumps({"unit_key": key, "rows": rows_u, "skips": skips_u}, sort_keys=True)
        with tpath.open("a", encoding="utf-8") as fh:  # single-line O_APPEND write
            fh.write(line + "\n")
            fh.flush()
        rows_all += rows_u
        skips_all += skips_u
        print(
            f"[fits] transfer unit {k + 1}/{len(units)} L={budget_l} draw={draw} "
            f"seed={seed} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return rows_all, skips_all


def _run_pilot(args: argparse.Namespace) -> int:
    """§9 pilot gate: ONE production-shape unit through the production path.

    Runs the max-L / full-U / first-(variant, regime) unit via
    :func:`_run_real` (same out-root, so the pilot unit RESUMES into the full
    run), extrapolates the measured walls across the full grid, writes
    ``pilot_report.json``, and exits :data:`PILOT_ABORT_RC` when the
    projection exceeds ``--pilot-abort-mult x --plan-wall-h`` (a designed
    halt with a report artifact — never a bare rc=1).
    """
    import copy

    u_sizes: list[int | None] = []
    for tok in args.u_sizes:
        u_sizes.append(None if str(tok).lower() == "full" else int(tok))
    from explore_persona_space.experiments.issue_1739.constants import (
        COMPOSITION_F_L,
        COMPOSITION_F_U,
    )

    full_specs = compose_run_specs(
        variants=tuple(VARIANTS) if args.variant == "both" else (args.variant,),
        regimes=tuple(args.regimes),
        u_sizes=tuple(u_sizes),
        budgets=tuple(args.budgets),
        draws=tuple(args.draws),
        seeds=tuple(args.seeds),
        compose=args.compose,
        compose_u_size=args.compose_u_size,
        f_u_grid=tuple(COMPOSITION_F_U),
        f_l_grid=tuple(COMPOSITION_F_L),
    )
    n_units = sum(len(s.budgets) * len(s.draws) * len(s.seeds) for s in full_specs)
    n_map_fits = len({(s.variant, s.u_size, s.f_u, s.f_l, s.budgets, s.seeds) for s in full_specs})
    n_transfer = (
        sum(len(s.budgets) * len(s.draws) * len(s.seeds) for s in full_specs if s.f_u is None)
        if args.transfer
        else 0
    )
    p = copy.copy(args)
    p.pilot = False
    p.variant = VARIANTS[0] if args.variant == "both" else args.variant
    p.regimes = [args.regimes[0]]
    p.u_sizes = ["full"]
    p.budgets = [max(args.budgets)]
    p.draws = [args.draws[0]]
    p.seeds = [args.seeds[0]]
    p.compose = False
    timings: dict = {}
    t0 = time.time()
    rc = _run_real(p, timings=timings)
    pilot_wall = time.time() - t0
    map_fit_s = float(sum(timings.get("map_fit_s", [0.0])))
    unit_s = float(sum(timings.get("grid_s", [0.0])))
    transfer_s = float(sum(timings.get("transfer_s", [0.0])))
    report = compose_pilot_report(
        n_map_fits=n_map_fits,
        n_units=n_units,
        n_transfer_units=n_transfer,
        map_fit_s=map_fit_s,
        unit_s=unit_s,
        transfer_s=transfer_s,
        plan_wall_h=args.plan_wall_h,
        abort_mult=args.pilot_abort_mult,
    )
    report.update(
        {
            "behavior": args.behavior,
            "pilot_wall_s": float(pilot_wall),
            "pilot_unit": {
                "variant": p.variant,
                "regime": p.regimes[0],
                "u_size": "full",
                "budget_l": p.budgets[0],
                "transfer": bool(args.transfer),
            },
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    out = args.out_root / "pilot_report.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    import os

    os.replace(tmp, out)
    print(
        f"[fits] pilot: unit_s={unit_s:.1f} map_fit_s={map_fit_s:.1f} "
        f"transfer_s={transfer_s:.1f} projected_wall_h={report['projected_wall_h']:.3f} "
        f"fence_wall_h={report['fence_wall_h']:.3f} plan_wall_h={args.plan_wall_h:.3f} "
        f"verdict={'ABORT' if report['abort'] else 'PASS'} -> {out}",
        flush=True,
    )
    if report["abort"]:
        return PILOT_ABORT_RC
    return rc


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = _parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.pilot:
        if args.synthetic:
            raise SystemExit("--pilot is a real-mode gate (drop --synthetic)")
        rc = _run_pilot(args)
    else:
        rc = _run_synthetic(args) if args.synthetic else _run_real(args)
    print(f"[fits] done rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
