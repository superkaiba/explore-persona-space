"""Phase-3 CLI for issue #1739: matched-budget arm grid over staged stores.

Two modes:

- ``--synthetic N`` — generate a tiny in-process synthetic slice and run the
  FULL engine end to end (whitening -> map -> 16 arms -> metrics -> outputs).
  This is the CLI's own smoke path (no network, no GPU, no staged data).
- real mode — assemble CellData from the round-B capture stores
  (``store_io.load_summaries`` layout), the judged DV dataset
  (``dv_build.write_dv_dataset`` payload), and an E1 extraction store, then
  run the grid per variant (context_end AND prefix_end — the standing
  prefix+context both-arms rule; a one-variant run requires --variant).

Outputs under ``--out-root`` (default ``eval_results/issue_1739``):
``arm_results/percell/cells.jsonl`` (checkpoint-per-cell, resumable) +
``arm_results/all_arms_spearman.json`` + ``map_diagnostics.json``.
"""

from __future__ import annotations

import argparse
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
VARIANTS = ("context_end", "prefix_end")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", default="synthetic", help="behavior slug (provenance)")
    ap.add_argument("--variant", choices=("both", *VARIANTS), default="both")
    ap.add_argument("--regime", default="e1", choices=("e1", "e2", "e2p"))
    ap.add_argument("--labeled-store", type=Path, help="round-B labeled capture store dir")
    ap.add_argument("--dv-json", type=Path, help="dv_build labeling.json for the behavior")
    ap.add_argument("--u-store", type=Path, help="staged #1092 U-pool store dir")
    ap.add_argument("--e1-store", type=Path, help="E1 extraction capture store dir (pos/neg rows)")
    ap.add_argument("--u-size", type=int, default=None, help="U-pool rung (rows)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--layers", type=int, nargs="+", default=None, help="layer subset (default: all)"
    )
    ap.add_argument("--arms", nargs="+", default=None, help="arm-slug subset (default: all)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mlp-epochs", type=int, default=None, help="override arm-5 MLP epochs")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument(
        "--synthetic", type=int, default=0, help="run the synthetic engine smoke with N contexts"
    )
    ap.add_argument("--synthetic-dim", type=int, default=8)
    ap.add_argument("--synthetic-layers", type=int, default=3)
    ap.add_argument(
        "--eval-rung", default="train", help="provenance: which eval rung these contexts are"
    )
    return ap.parse_args(argv)


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
        "regime": args.regime,
        "u_rung": int(x_u.shape[1]),
        "eval_rung": args.eval_rung,
        "config": "synthetic",
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


def _meta_field(meta_rows: list[dict], candidates: tuple[str, ...], what: str) -> str:
    keys = set(meta_rows[0])
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"no {what} field among {candidates} in store row_index keys={sorted(keys)}")


def _load_labeled(store_dir: Path, dv_json: Path, layers: list[int]):
    """Round-B labeled store + DV dataset -> per-CONTEXT layer-leading arrays.

    Store rows are per-ROLLOUT; context/prefix acts are identical across a
    context's rollouts (first occurrence taken), the answer act is the MEAN
    over the context's rollout t1 rows. Contexts without a kept DV are
    dropped (drop-never-coerce).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import store_io

    kinds = ("prefix_end", "context_end", "t1")
    arrays, meta = store_io.load_summaries(
        store_dir, kinds, tuple(layers), hidden_dim=arrays_dim(store_dir, layers)
    )
    ctx_key = _meta_field(meta, ("context_id",), "context id")
    dv_payload = json.loads(Path(dv_json).read_text())
    dv_by_ctx = {r["context_id"]: r for r in dv_payload["rows"] if r.get("dv") is not None}
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
    groups = [str(dv_by_ctx[c].get("group_key", c)) for c in ctx_order]
    per_rollout = None
    if all("per_rollout_scores" in dv_by_ctx[c] for c in ctx_order):
        k_max = max(len(dv_by_ctx[c]["per_rollout_scores"]) for c in ctx_order)
        per_rollout = np.full((len(ctx_order), k_max), np.nan)
        for i, c in enumerate(ctx_order):
            for j, (_k, s) in enumerate(sorted(dv_by_ctx[c]["per_rollout_scores"].items())):
                per_rollout[i, j] = np.nan if s is None else float(s)
    return {"context_end": z_ctx, "prefix_end": z_pre}, z_ans, dv, groups, per_rollout, ctx_order


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


def _run_real(args: argparse.Namespace) -> int:
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits, store_io
    from explore_persona_space.experiments.issue_1739.constants import N_LAYERS

    for req in ("labeled_store", "dv_json", "u_store", "e1_store"):
        if getattr(args, req) is None:
            raise SystemExit(f"real mode requires --{req.replace('_', '-')} (or use --synthetic N)")
    layers = args.layers or list(range(N_LAYERS))
    z_by_variant, y_ans_raw, dv, groups, per_rollout, ctx_order = _load_labeled(
        args.labeled_store, args.dv_json, layers
    )
    dim = y_ans_raw.shape[-1]

    # Stage the #1092 U-pool slice on demand (idempotent: a dest that is
    # already loadable for the requested (kinds x layers) regime — incl. a
    # LOCAL capture-store stand-in — short-circuits without network).
    store_io.stage_u_store(Path(args.u_store), ("prefix_end", "context_end", "t1"), tuple(layers))
    u_arrays, u_meta = store_io.load_summaries(
        args.u_store, ("prefix_end", "context_end", "t1"), tuple(layers), hidden_dim=dim
    )
    u_mask = store_io.fit_pool_mask(u_meta)  # is_eval_only exclusion
    u_rows = np.flatnonzero(u_mask)
    if args.u_size and args.u_size < len(u_rows):
        rng = np.random.default_rng([1739, 9, args.seeds[0]])
        u_rows = np.sort(rng.choice(u_rows, size=args.u_size, replace=False))
    rb_raw = _load_rb_e1(args.e1_store, layers, dim)

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    all_records: list[dict] = []
    diag_out: dict = {}
    for variant in variants:
        x_u = np.stack([u_arrays[(variant, ly)][u_rows] for ly in layers])
        y_u = np.stack([u_arrays[("t1", ly)][u_rows] for ly in layers])
        wh = fits.fit_whitening(x_u, device=args.device, seed=args.seeds[0])
        mapfit = fits.fit_linear_map(
            fits.apply_whitening(x_u, wh), fits.apply_whitening(y_u, wh), device=args.device
        )
        diag_out[variant] = mapfit.diagnostics
        data = arms.CellData(
            z_ctx=fits.apply_whitening(z_by_variant[variant], wh),
            z_ans=fits.apply_whitening(y_ans_raw, wh),
            dv=dv,
            rb=np.einsum("ld,lde->le", rb_raw, wh.w),
            mapfit=mapfit,
            layers=tuple(layers),
            per_rollout=per_rollout,
        )
        provenance = {
            "behavior": args.behavior,
            "variant": variant,
            "regime": args.regime,
            "u_rung": int(len(u_rows)),
            "eval_rung": args.eval_rung,
            "config": "config_a",
        }
        kwargs = {}
        if args.n_boot:
            kwargs["n_boot"] = args.n_boot
        if args.n_perm:
            kwargs["n_perm"] = args.n_perm
        mlp_kwargs = {"max_epochs": args.mlp_epochs} if args.mlp_epochs else None
        all_records += arms.run_grid(
            data,
            groups,
            budgets=args.budgets,
            draws=args.draws,
            seeds=args.seeds,
            provenance=provenance,
            out_dir=args.out_root / "arm_results",
            arms=args.arms,
            device=args.device,
            mlp_kwargs=mlp_kwargs,
            **kwargs,
        )
    arms.write_summary(
        all_records,
        args.out_root / "arm_results" / "all_arms_spearman.json",
        meta={
            "mode": "real",
            "behavior": args.behavior,
            "n_contexts": len(ctx_order),
            "layers": layers,
            "u_rows": int(len(u_rows)),
        },
    )
    (args.out_root / "map_diagnostics.json").write_text(json.dumps(diag_out, indent=1))
    print(f"[fits] real grid done: {len(all_records)} cells", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = _parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = _run_synthetic(args) if args.synthetic else _run_real(args)
    print(f"[fits] done rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
