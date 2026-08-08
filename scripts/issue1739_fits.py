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
        "--rb-point",
        choices=("t1", "context_end"),
        default="t1",
        help="r_B extraction POINT (new-arm-round item 1): 't1' (default — the committed "
        "answer-avg direction, byte-identical behavior) or 'context_end' (final-context-token "
        "direction; every regime label in unit/row keys gains an '_fc' suffix — e1_fc/e2p_fc; "
        "matched-e2 is REFUSED under context_end, plan v9 structural restriction — so fc rows "
        "can never collide with committed rows at resume/merge time)",
    )
    ap.add_argument(
        "--fixed-coordinate",
        default=None,
        help="registered fixed-coordinate declaration recorded in every unit's provenance "
        "(e.g. 'u=full' for the oracle/arm5 legs — plan v8 Must-Fix 2: a u=full-only grid is "
        "a registered READ coordinate, never 'degenerate'). Omitted -> field absent "
        "(committed unit keys unchanged).",
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
        "--transfer-arms",
        nargs="+",
        default=None,
        metavar="ROSTER|SLUG",
        help="eval-rung transfer roster: 'wide' (default — the 6 core ladder arms plus "
        "the fitted arms 5/7/8/12), 'core' (the original 6, reproduces the committed "
        "transfer columns exactly), or an explicit arm-slug list",
    )
    ap.add_argument(
        "--transfer-preds",
        action="store_true",
        help="ALSO persist per-(arm, eval context) frozen-layer transfer predictions "
        "for the --transfer leg (one JSONL per unit under "
        "arm_results/percell/transfer_preds/, schema = arms.transfer_preds_rows with "
        "a per-context 'rung' label). Default OFF: the transfer leg's aggregate rows "
        "are unchanged, so every other lane is byte-identical. Turning it on is what "
        "makes an OOD-rung scatter / per-context subset read a pure re-analysis "
        "instead of another GPU re-score (the train setting already persists its "
        "per-cell preds npz via arms._save_cell_preds).",
    )
    ap.add_argument(
        "--eval-rung-knn",
        action="store_true",
        help="ALSO compute the kNN-retrieval companion (and a PER-EVAL-RUNG "
        "breakdown) for the map's eval-distribution reconstruction read in "
        "map_diagnostics.json. Default OFF — kNN is O(n^2 d) per (layer, metric) "
        "and the pooled-R^2-only object is what every prior lane recorded.",
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
    # NOTE: the choices tuple is a literal (every `fits` import in this module is
    # deferred); parity with fits.NONLINEAR_MAP_KINDS is test-pinned by
    # tests/test_issue1739_nlmap.py::test_cli_map_kind_choices_match_fits.
    ap.add_argument(
        "--map-kind",
        choices=("linear", "mlp", "kernel"),
        default="linear",
        help=(
            "context->answer map family for the map arms (6/7/8). 'linear' (default) is "
            "the reviewed ridge map — byte-identical behavior. 'mlp' / 'kernel' fit the "
            "#1739 nonlinear-map round's maps via the reused #779 N1M fitters; every "
            "downstream arm, fold scheme and output schema is unchanged."
        ),
    )
    ap.add_argument(
        "--mlp-map-width",
        type=int,
        default=None,
        help="hidden width for --map-kind mlp (default: fits.MLP_MAP_WIDTH, the 512 recipe)",
    )
    ap.add_argument(
        "--krr-map-centers",
        type=int,
        default=None,
        help="Nystrom landmarks for --map-kind kernel (default: fits.KRR_MAP_M_CENTERS)",
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
    args = ap.parse_args(argv)
    if args.rb_point == "context_end" and "e2" in args.regimes:
        # Plan v9 registered STRUCTURAL RESTRICTION (concern
        # e2fc-structurally-null-direction, code-review r1): the matched-pair
        # E2 weights contrast hi/lo ROLLOUTS within a context, and every
        # rollout of a context shares the identical context_end activation,
        # so the weighted contrast cancels EXACTLY — the "direction" is float
        # residue (measured |max| per-context net weight 5.4e-20 on the real
        # evil DV) that K2's zero/NaN check cannot see. Refused STRUCTURALLY
        # at the flag, never via a norm threshold. e2p (pooled across-context
        # weights) and e1 are unaffected.
        ap.error(
            "--rb-point context_end refuses --regimes e2: the matched-pair (within-context) "
            "contrast is structurally ZERO at a context-level activation — every rollout of "
            "a context shares the context_end row, so the hi/lo weights cancel exactly "
            "(plan v9 structural restriction, concern e2fc-structurally-null-direction). "
            "Use e1/e2p."
        )
    return args


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
    mapfit = _fit_map(args, z_u, zy_u)
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
    # {layer: (n_rows, d)} per-rollout rows of the requested rollout_rows_kind
    # ('t1' default; 'context_end' under --rb-point context_end) — e2/e2p only.
    ans_rows: dict | None
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
    rollout_rows_kind: str = "t1",
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
        # rollout_rows_kind: 't1' = the committed per-rollout answer rows; the
        # fc extraction point ('context_end') swaps ONLY the array the e2/e2p
        # direction builder consumes (per-context acts identical across a
        # context's rollouts, so the weighted sum reduces to per-context
        # weights x the context act — the plan-v8 item-1 fc definition).
        ans_rows = {
            ly: np.asarray(arrays[(rollout_rows_kind, ly)][sel], dtype=np.float64) for ly in layers
        }
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


def _load_rb_e1(e1_store: Path, layers: list[int], dim: int, *, summary_kind: str = "t1"):
    """E1 extraction store -> raw diff-of-means direction (Ly, d).

    ``summary_kind`` selects the extraction POINT over the SAME judge-filtered
    pos/neg row set: 't1' (committed answer-avg) or 'context_end' (the
    new-arm-round fc direction — position is the only change; plan v8 §11).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits, store_io

    arrays, meta = store_io.load_summaries(e1_store, (summary_kind,), tuple(layers), hidden_dim=dim)
    side_key = _meta_field(meta, ("side", "polarity", "pv_side", "pair_side"), "pos/neg side")
    sides = np.array([str(r[side_key]).lower() for r in meta])
    pos_rows = np.flatnonzero(np.isin(sides, ("pos", "positive")))
    neg_rows = np.flatnonzero(np.isin(sides, ("neg", "negative")))
    if len(pos_rows) == 0 or len(neg_rows) == 0:
        raise RuntimeError(f"E1 store has {len(pos_rows)} pos / {len(neg_rows)} neg rows")
    acts = np.stack([arrays[(summary_kind, ly)] for ly in layers], axis=1)  # (n, Ly, d)
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

    fc = regime.endswith("_fc")
    base = regime.removesuffix("_fc")
    if fc and base == "e2":
        # Defense-in-depth twin of the --rb-point flag refusal (plan v9
        # structural restriction): matched-pair weights cancel exactly on
        # context-level rows, so an e2_fc direction cannot exist. Structural
        # refusal — never a norm check (K2 is blind to the float residue).
        raise SystemExit(
            "matched-e2_fc is structurally undefined: within-context hi/lo weights cancel "
            "exactly on context_end rows (plan v9 structural restriction, concern "
            "e2fc-structurally-null-direction) — use e1_fc/e2p_fc"
        )

    def _k2_gate(rb):
        # K2 (plan v8 §7): a degenerate fc direction (zero/NaN norm at any
        # layer) halts the (behavior, regime) leg with a named report — never
        # a fabricated direction. Scoped to fc regimes so committed t1
        # behavior is byte-identical.
        if not fc:
            return rb
        norms = np.linalg.norm(np.asarray(rb, dtype=np.float64), axis=1)
        bad = [int(layers[i]) for i, v in enumerate(norms) if not np.isfinite(v) or v == 0.0]
        if bad:
            raise SystemExit(
                f"[fits] K2 HALT: fc direction {regime} for behavior {args.behavior!r} is "
                f"degenerate (zero/NaN norm) at layer(s) {bad} — refusing to fabricate a "
                "direction (plan v8 §7 K2)"
            )
        return rb

    if base == "e1":
        return _k2_gate(
            _load_rb_e1(args.e1_store, layers, dim, summary_kind="context_end" if fc else "t1")
        )
    if tbl.per_rollout is None or tbl.ans_rows is None:
        raise SystemExit(
            f"--regimes {base} requires per-rollout judge scores + per-rollout answer rows "
            f"(behavior {args.behavior!r} has none — run it with --regimes e1)"
        )
    w_hi, w_lo, n_qual = fits.matched_pair_split_weights(
        np.asarray(tbl.per_rollout, dtype=float),
        spread_min=E2_SPREAD_MIN,
        pooled=(base == "e2p"),
    )
    w_row = (w_hi - w_lo)[tbl.ans_row_ctx, tbl.ans_row_k]  # (n_rows,)
    rb = np.stack([w_row @ tbl.ans_rows[ly] for ly in layers])
    logger.info(
        "[fits] %s direction: %d qualifying contexts over %d rollout rows",
        regime,
        n_qual,
        len(w_row),
    )
    return _k2_gate(rb)


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
    # PID-unique tmp (#1315 fan-out staging race): phase-A fits the two map kinds
    # CONCURRENTLY under one behavior + one tensors-root, so both processes write
    # this same r_B destination. A shared tmp name lets one clobber the other's
    # half-written file and the loser's os.replace dies FileNotFoundError; a
    # unique tmp + atomic replace onto the shared final path is safe (both write
    # identical content — same behavior/regime/seed => deterministic).
    tmp = out_dir / f"{behavior}.tmp.{os.getpid()}.npz"  # savez appends .npz to non-.npz (#1092)
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


def _map_path(tensors_root: Path | str, variant: str, u_label: str, kind: str) -> Path:
    """Canonical persisted-map path — ONE definition for writer, reader and gate.

    Linear rungs keep the historical suffix-free ``.npz`` name; a nonlinear kind
    gets a ``__<kind>.pt`` sibling. The writer (:func:`_save_map`), the reader
    (:func:`_load_nl_map`) and the staging step each need this string, and three
    independent rebuilds is a rename away from a writer that persists where the
    reader never looks (silently re-fitting every map for the rest of time).
    """
    out_dir = Path(tensors_root) / "maps"
    if kind == "linear":
        return out_dir / f"{variant}__u{u_label}.npz"
    return out_dir / f"{variant}__u{u_label}__{kind}.pt"


def _save_map(
    tensors_root: Path,
    variant: str,
    u_label: str,
    mapfit,
    layers,
    *,
    map_seed: int | None = None,
    space_meta: dict | None = None,
) -> Path:
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

    ``space_meta`` (#1975): the ``fits.map_space_meta(...)`` dict recording the
    FIT SPACE + whitening provenance + train-input norm stats, merged into the
    payload meta for BOTH the linear ``.npz`` and nonlinear ``.pt`` forms —
    the apply/load-time input-space parity check
    (``fits.assert_map_input_space`` / ``fits.check_whitening_parity``) reads
    it. ``None`` (an unthreaded caller) omits the fields entirely, keeping the
    payload byte-compatible with pre-#1975 writers.
    """
    import os
    import time as _time

    import numpy as np

    kind = getattr(mapfit, "kind", "linear")
    out = _map_path(tensors_root, variant, u_label, kind)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    meta = {
        "variant": variant,
        "u_label": u_label,
        "layers": [int(x) for x in layers],
        "map_kind": kind,
        "w_fit_rows": mapfit.diagnostics.get("w_fit_rows"),
        "solver": mapfit.diagnostics.get("solver"),
        # The FIT seed. Load-bearing for cross-invocation reuse: for a
        # SUBSAMPLED U rung the pool rows themselves are drawn with this seed
        # (_u_pool_for_spec's rng([1739, 9, seeds[0]])), so a payload fit under a
        # different seed is a map over a DIFFERENT 250 rows — and the row-COUNT
        # guard (w_fit_rows == n_u) cannot see that. None = a legacy payload
        # written before this field existed (_load_nl_map accepts it, loudly).
        "map_seed": None if map_seed is None else int(map_seed),
        "dtype": "w=fp16; mu/sd=fp32",
        "apply": "pred = ((x - x_mu)/x_sd) @ w + y_mu (whitened space)",
        "git_commit": _git_commit(),
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if space_meta is not None:
        # #1975: fit_space + whitening_provenance + train_input_norm_{mean,std}
        # (both persisted forms get them — the meta dict is shared below).
        meta.update(space_meta)
    if kind != "linear":
        # A nonlinear map is a per-layer torch payload (the #779 N1M apply_map
        # format), not a (w, mu, sd) npz — persist it with torch.save under a
        # .pt sibling so the `--stage tensors` sweep (no eligibility filter)
        # still carries it. Atomic write; keeps the whole-tree upload wiring.
        import os as _os

        import torch as _torch

        meta["apply"] = "pred = issue779_ffc_n1m_fits.apply_map(payload[layer], x) (whitened space)"
        meta["dtype"] = "per-layer N1M payload tensors (fp32)"
        # Carry the held-out map-quality diagnostics (per-layer r2_map,
        # r2_identity_bias, knn{euclidean,cosine}, n_train/n_hold, payload
        # tags) into the persisted meta so `map_quality.json` is derivable
        # from the frozen artifact alone — the standing mapping-companion
        # reads (CLAUDE.md identity+bias / kNN bullet) must survive without
        # re-running the fit. Linear rungs keep their pre-existing meta.
        meta["diagnostics"] = mapfit.diagnostics
        tmp = out_dir / f"{out.name}.tmp.{_os.getpid()}"  # PID-unique: see _save_rb
        _torch.save({"meta": meta, "payloads": list(mapfit.nl_payloads)}, tmp)
        _os.replace(tmp, out)
        logger.info("[fits] %s map payloads persisted -> %s", kind, out)
        return out
    tmp = out_dir / f"{out.stem}.tmp.{os.getpid()}.npz"  # keep .npz (#1092); PID-unique
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


def _fit_map(args, x_w, y_w):
    """Fit the (variant, U rung) context->answer map per ``args.map_kind``.

    ONE seam for both call sites (synthetic + real), so the nonlinear round
    inherits the whole reviewed pipeline — folds, arms, bootstrap, output
    schema — unchanged. ``map_kind == "linear"`` reproduces the pre-existing
    ``fit_linear_map(...)`` call EXACTLY (byte-identical default path).
    """
    from explore_persona_space.experiments.issue_1739 import fits

    kind = getattr(args, "map_kind", "linear")
    if kind == "linear":
        return fits.fit_linear_map(x_w, y_w, device=args.device)
    kwargs = {}
    if getattr(args, "mlp_map_width", None) is not None:
        kwargs["mlp_width"] = int(args.mlp_map_width)
    if getattr(args, "krr_map_centers", None) is not None:
        kwargs["krr_m_centers"] = int(args.krr_map_centers)
    return fits.fit_nonlinear_map(
        x_w, y_w, kind=kind, device=args.device, seed=args.seeds[0], **kwargs
    )


NL_MAP_REUSE_ENV = "EPM_I1739_NL_MAP_REUSE"


def _key_sha(unit_key: str) -> str:
    """Stable 16-hex filename stem for a unit key (same convention as preds npz)."""
    import hashlib

    return hashlib.sha1(unit_key.encode()).hexdigest()[:16]


def _eval_rung_reconstruction(mapfit, z_ev_w, za_ev_w, *, rungs=None, knn: bool = False) -> dict:
    """Per-layer reconstruction R^2 of the map on THIS behavior's eval rung.

    The SECOND of the two map-quality reads the standing mapping-companions
    rule wants (CLAUDE.md identity+bias / kNN bullet). The payload's own
    ``diagnostics`` carry the U-pool HOLDOUT R^2, which is behavior-INDEPENDENT;
    this read scores the SAME shared map against THIS behavior's eval-split
    answers, so it is behavior-SPECIFIC and belongs in the per-lane
    ``map_diagnostics.json`` -- never in the shared ``.pt``. Writing it into the
    payload would make a behavior-independent artifact behavior-dependent and
    break ``_save_map``'s skip-on-existence sharing (the whole basis of fitting
    each map once and fanning the scoring out).

    Reuses ``fits.r2_pooled`` -- the SAME estimator ``map_diagnostics`` uses for
    ``r2_map`` -- so the two reads are directly comparable in the table rather
    than differing by estimator convention. Expect the eval-rung read to run
    well BELOW the U-pool read (an off-distribution extrapolation from the
    #1092 WildChat pool onto behavior eval distributions; strongly negative is
    a recordable finding, not a bug -- see the #1774 apply-path resolution).

    ``rungs`` (per-eval-row rung labels, aligned with ``z_ev_w``'s row axis) and
    ``knn=True`` are the map-recon-on-eval-dist round's two additive extensions:
    with them the read is broken out PER EVAL DISTRIBUTION (each OOD rung
    separately, not only the pooled eval split) and carries the standing
    kNN-retrieval companion (CLAUDE.md identity+bias / kNN bullet) computed with
    the SAME ``mapping_baselines.knn_retrieval`` helper ``map_diagnostics`` uses
    on the U-pool holdout, so the two are directly comparable. Both default OFF:
    with ``rungs=None, knn=False`` the returned object is byte-identical to the
    pre-existing pooled-R^2-only read, so every other lane is unperturbed.
    kNN is O(n^2 d) per (layer, metric), which is why it is opt-in.
    """
    import math

    from explore_persona_space.experiments.issue_1739 import fits
    from explore_persona_space.experiments.issue_1739.constants import KNN_KS

    def _block(pred_b, true_b) -> list[dict]:
        rows = []
        for li in range(pred_b.shape[0]):
            row = {"layer_idx": li, "r2_eval_rung": float(fits.r2_pooled(pred_b[li], true_b[li]))}
            if knn:
                from explore_persona_space.analysis.mapping_baselines import knn_retrieval

                # SAME helper map_diagnostics uses on the U-pool holdout, so the
                # eval-distribution retrieval read is directly comparable to it.
                # chance = k/n_pool rides the helper's own `chance_at_k` field.
                row["knn"] = {
                    metric: knn_retrieval(pred_b[li], true_b[li], ks=KNN_KS, metric=metric)
                    for metric in ("euclidean", "cosine")
                }
            rows.append(row)
        return rows

    pred = fits.apply_map(z_ev_w, mapfit)
    per_layer = _block(pred, za_ev_w)
    finite = [r["r2_eval_rung"] for r in per_layer if math.isfinite(r["r2_eval_rung"])]
    out = {
        "per_layer": per_layer,
        "r2_eval_rung_mean": (sum(finite) / len(finite)) if finite else None,
        "n_eval_rows": int(z_ev_w.shape[1]),
        "n_layers": int(pred.shape[0]),
        "estimator": "fits.r2_pooled (same as r2_map)",
    }
    if knn:
        out["knn_ks"] = list(KNN_KS)
    if rungs is not None:
        import numpy as _np

        labels = _np.asarray([str(r) for r in rungs])
        if labels.size != pred.shape[1]:
            raise ValueError(f"rungs/eval-row mismatch: {labels.size} vs {pred.shape[1]}")
        per_rung: dict[str, dict] = {}
        for rung in sorted(set(labels.tolist())):
            sel = _np.flatnonzero(labels == rung)
            # kNN needs a candidate pool bigger than the largest k; below that
            # the retrieval read is degenerate — record the rung's R^2 only.
            rows = (
                _block(pred[:, sel], za_ev_w[:, sel])
                if sel.size > max(KNN_KS) or not knn
                else [
                    {
                        "layer_idx": li,
                        "r2_eval_rung": float(fits.r2_pooled(pred[li, sel], za_ev_w[li, sel])),
                    }
                    for li in range(pred.shape[0])
                ]
            )
            fin = [r["r2_eval_rung"] for r in rows if math.isfinite(r["r2_eval_rung"])]
            per_rung[rung] = {
                "n_rows": int(sel.size),
                "per_layer": rows,
                "r2_eval_rung_mean": (sum(fin) / len(fin)) if fin else None,
                "knn_skipped_small_pool": bool(knn and sel.size <= max(KNN_KS)),
            }
        out["per_rung"] = per_rung
    return out


def _load_nl_map(
    tensors_root: Path,
    variant: str,
    u_label: str,
    kind: str,
    layers,
    n_u: int,
    device: str = "cpu",
    map_seed: int | None = None,
):
    """Load a persisted NONLINEAR map instead of re-fitting it, or None.

    ``_save_map`` already skips on existence and documents the
    ``(variant, u_label)`` map as BEHAVIOR-INDEPENDENT (shared #1092 fit pool,
    shared subsample + whitening seeds) — but only the WRITE was idempotent,
    never the COMPUTE, so every (behavior x kind) invocation re-fit the
    identical maps (measured: ~0.68 h per MLP full-U map key x 2 variants x 3 U
    rungs, ~5.7 h repeated across 3 behaviors vs ~1.7 h fit once). This closes
    that gap by consuming the artifact the code already writes; it is NOT new
    fit math.

    NONLINEAR KINDS ONLY — the linear path stays byte-identical (this module's
    stated invariant, and pod-1739 is running the linear grid). Guarded: the
    persisted pool size, layer list and FIT SEED must match this rung exactly,
    and the payload must carry the held-out diagnostics (so ``diag_out`` stays an
    honest read rather than silently losing the map-quality companions). Any
    mismatch or missing field returns None and the caller re-fits. Kill switch:
    ``EPM_I1739_NL_MAP_REUSE=0``.

    The SEED guard matters once maps are fit in a SEPARATE invocation from the
    scoring lanes (the phase-A prefetch): for a subsampled U rung the pool rows
    are drawn with ``seeds[0]``, so a payload fit under another seed is a map
    over a DIFFERENT subsample — and ``w_fit_rows == n_u`` passes anyway (250 ==
    250). A payload with no recorded ``map_seed`` predates the field and is
    accepted with a loud warning rather than silently discarded.
    """
    import os

    if kind == "linear" or os.environ.get(NL_MAP_REUSE_ENV, "1") == "0":
        return None
    path = _map_path(tensors_root, variant, u_label, kind)
    if not path.exists():
        return None

    import torch

    from explore_persona_space.experiments.issue_1739.fits import MapFit

    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta = blob.get("meta", {}) or {}
    payloads = blob.get("payloads") or []
    want_layers = [int(x) for x in layers]
    diagnostics = meta.get("diagnostics")
    reasons = []
    if meta.get("map_kind") != kind:
        reasons.append(f"map_kind {meta.get('map_kind')!r} != {kind!r}")
    if [int(x) for x in meta.get("layers") or []] != want_layers:
        reasons.append("layer list mismatch")
    if len(payloads) != len(want_layers):
        reasons.append(f"n_payloads {len(payloads)} != n_layers {len(want_layers)}")
    if int(meta.get("w_fit_rows") or -1) != int(n_u):
        reasons.append(f"w_fit_rows {meta.get('w_fit_rows')} != n_u {n_u}")
    if not isinstance(diagnostics, dict) or not diagnostics.get("per_layer"):
        reasons.append("payload carries no per-layer diagnostics")
    stored_seed = meta.get("map_seed")
    if stored_seed is None:
        logger.warning(
            "[fits] persisted %s map %s carries NO map_seed (pre-guard payload) — "
            "reusing on the row-count guard alone; a subsampled rung fit under a "
            "different seed would be a map over different rows",
            kind,
            path.name,
        )
    elif map_seed is not None and int(stored_seed) != int(map_seed):
        reasons.append(f"map_seed {stored_seed} != requested {int(map_seed)}")
    if reasons:
        logger.warning(
            "[fits] persisted %s map at %s NOT reused (%s) — re-fitting",
            kind,
            path.name,
            "; ".join(reasons),
        )
        return None
    logger.info(
        "[fits] reusing persisted %s map %s (w_fit_rows=%d) — skipping re-fit",
        kind,
        path.name,
        int(n_u),
    )
    # #1975: carry the input-space parity metadata on the returned object so
    # downstream consumers can check it. Absent fit_space = a LEGACY payload —
    # loud warning (never silent, never a crash: the in-pipeline apply here is
    # whitened-by-construction, the same CLI computes x_w).
    space_meta = {
        k: meta[k]
        for k in (
            "fit_space",
            "whitening_provenance",
            "train_input_norm_mean",
            "train_input_norm_std",
        )
        if meta.get(k) is not None
    }
    if not space_meta.get("fit_space"):
        logger.warning(
            "[fits] persisted %s map %s carries NO fit_space metadata (LEGACY payload, "
            "pre-#1975) — input-space parity cannot be checked at apply time (the #1739 "
            "whitened-fit/raw-apply incident); re-persist via _save_map to gain the check",
            kind,
            path.name,
        )
    return MapFit(
        w=None,
        x_mu=None,
        x_sd=None,
        y_mu=None,
        diagnostics=dict(diagnostics),
        kind=kind,
        nl_payloads=tuple(payloads),
        # apply runs NOW, so honor this process's device rather than whichever
        # device the original fit happened to use.
        apply_device=str(device or "cpu"),
        space_meta=space_meta or None,
    )


MAP_ROUNDTRIP_COS_MIN = 0.9999
MAP_ROUNDTRIP_REL_MAX = 1e-3
# Probe rows for the gate. 8 rows x 28 layers x 3584 dims fp64 ~= 6.4 MB — small
# enough to hold beside a full-U fit, wide enough that a per-layer payload
# misalignment cannot coincidentally agree.
MAP_ROUNDTRIP_PROBE_ROWS = 8


def _verify_map_roundtrip(
    tensors_root: Path,
    variant: str,
    u_label: str,
    kind: str,
    layers,
    n_u: int,
    mapfit,
    probe_x,
    *,
    map_seed: int | None,
    device: str = "cpu",
) -> dict:
    """WIRED save->load->apply gate for a freshly-persisted nonlinear map.

    The phase-A prefetch fits every map in a THROWAWAY invocation and the
    scoring lanes consume only the persisted payload, so the payload is the sole
    surviving copy of hours of fit: a serialization defect (a dropped layer, a
    truncated dtype, a payload/layer misalignment) would otherwise surface as
    silently wrong arm scores in every lane. This applies the in-memory fit and
    the round-tripped payload to the SAME held-out probe rows and fails LOUD on
    disagreement.

    Loading goes through :func:`_load_nl_map` — the lanes' OWN reader — so the
    gate also proves the payload passes every reuse guard (layers, row count,
    seed, diagnostics); a ``None`` return is a gate FAILURE, not a soft skip.

    Tolerance: both sides are the same fp32 payload applied in ONE process, so
    agreement is near-exact; the bars exist to catch structural defects, which
    read orders of magnitude away (a wrong/misaligned payload gives cosines far
    below 0.99). Returns a small record for the log / diagnostics.
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import fits

    reloaded = _load_nl_map(
        tensors_root, variant, u_label, kind, layers, n_u, device=device, map_seed=map_seed
    )
    path = _map_path(tensors_root, variant, u_label, kind)
    if reloaded is None:
        raise RuntimeError(
            f"[fits] map round-trip gate FAILED for {path}: the just-persisted payload "
            "does not pass _load_nl_map's reuse guards (see the warning above) — the "
            "scoring lanes could never consume it"
        )
    pred_mem = np.asarray(fits.apply_map(probe_x, mapfit), dtype=np.float64)
    pred_disk = np.asarray(fits.apply_map(probe_x, reloaded), dtype=np.float64)
    if pred_mem.shape != pred_disk.shape:
        raise RuntimeError(
            f"[fits] map round-trip gate FAILED for {path}: prediction shape "
            f"{pred_disk.shape} != in-memory {pred_mem.shape}"
        )
    cos_per_layer = []
    for li in range(pred_mem.shape[0]):
        a = pred_mem[li].ravel()
        b = pred_disk[li].ravel()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos_per_layer.append(1.0 if den == 0.0 else float(a @ b / den))
    max_abs = float(np.max(np.abs(pred_disk - pred_mem))) if pred_mem.size else 0.0
    scale = float(np.max(np.abs(pred_mem))) if pred_mem.size else 0.0
    rel = max_abs / (scale + 1e-12)
    cos_min = min(cos_per_layer) if cos_per_layer else 1.0
    record = {
        "path": str(path),
        "n_probe_rows": int(probe_x.shape[1]),
        "cos_min": cos_min,
        "max_abs_diff": max_abs,
        "rel_max_abs_diff": rel,
        "cos_min_bar": MAP_ROUNDTRIP_COS_MIN,
        "rel_max_bar": MAP_ROUNDTRIP_REL_MAX,
    }
    if cos_min < MAP_ROUNDTRIP_COS_MIN or rel > MAP_ROUNDTRIP_REL_MAX:
        raise RuntimeError(
            f"[fits] map round-trip gate FAILED for {path}: cos_min={cos_min:.6f} "
            f"(bar {MAP_ROUNDTRIP_COS_MIN}), rel_max_abs_diff={rel:.3e} "
            f"(bar {MAP_ROUNDTRIP_REL_MAX}) over {record['n_probe_rows']} probe rows"
        )
    print(
        f"[fits] map round-trip gate PASS {path.name} "
        f"(cos_min={cos_min:.6f} rel={rel:.2e} rows={record['n_probe_rows']})",
        flush=True,
    )
    return record


def _record_compose_skip(spec: RunSpec, exc: Exception, compose_skips: list[dict]) -> None:
    """Round-3 Minor 1: a ``_u_pool_for_spec`` failure is a recordable skip
    ONLY for a composition spec (a quota that cannot fill); on a PLAIN ladder
    rung it RE-RAISES — a genuine load bug must never drop a whole
    (variant, U, regime) slice from the summary as a 'compose skip'."""
    if spec.f_u is None:
        raise exc
    compose_skips.append({"spec": dataclasses.asdict(spec), "reason": str(exc)})


PILOT_ABORT_RC = 7  # designed-halt rc (never a bare rc=1 — gotchas.md pilot-gate entry)


def _map_key(spec: RunSpec) -> tuple:
    """Whitening/map cache key — regime slices of one (variant, U) share it."""
    return (spec.variant, spec.u_size, spec.f_u, spec.f_l, spec.budgets, spec.seeds)


def compose_pilot_report(
    *,
    n_map_fits: int,
    map_fit_s: float,
    unit_group_walls: dict[int, float],
    n_plain_groups: dict[int, int],
    n_compose_units: dict[int, int],
    transfer_s: float,
    n_pilot_transfer_units: int,
    n_transfer_units: int,
    plan_wall_h: float,
    abort_mult: float,
    n_units: int = 0,
) -> dict:
    """Pure §9 pilot fence math (unit-tested; the CLI adds provenance fields).

    Round-8 semantics: the pilot measures ONE regime-shared unit-GROUP per
    BUDGET (all regimes of a (variant, U) slice share the ridge/MLP
    factorizations — ``run_grid_multi``), so the projection prices every
    (map_key, budget, draw, seed) unit-group at its budget's MEASURED group
    wall — the old max-L per-unit extrapolation both over-counted the shared
    work ~n_regimes x AND priced the cheap budgets at the top rung's cost.
    Composition cells (single-regime groups) are priced at their anchor
    budget's group wall; their per-cell map fits ride ``n_map_fits``.
    Transfer projects at the measured per-(unit, regime) average
    (conservative: the pilot's units share no row set, while the full
    grid's top-rung units all hit the arm-4 row-set cache).
    ``fence_wall_h`` is the >=2x fence the experimenter sizes timeouts from;
    ``abort`` fires when the projection exceeds ``abort_mult x plan_wall_h``
    (plan §9: 'If any pilot exceeds 3x the estimate above, abort and
    re-size').
    """
    walls = {int(b): float(w) for b, w in unit_group_walls.items()}
    fallback = max(walls.values()) if walls else 0.0
    plain_s = sum(n * walls.get(int(b), fallback) for b, n in n_plain_groups.items())
    compose_s = sum(n * walls.get(int(b), fallback) for b, n in n_compose_units.items())
    per_transfer = transfer_s / n_pilot_transfer_units if n_pilot_transfer_units else 0.0
    transfer_total = n_transfer_units * per_transfer
    projected_s = n_map_fits * map_fit_s + plain_s + compose_s + transfer_total
    projected_h = projected_s / 3600.0
    return {
        "n_map_fits": int(n_map_fits),
        "map_fit_s": float(map_fit_s),
        "unit_group_walls_s": {str(b): float(w) for b, w in sorted(walls.items())},
        "n_plain_groups": {str(int(b)): int(n) for b, n in sorted(n_plain_groups.items())},
        "n_compose_units": {str(int(b)): int(n) for b, n in sorted(n_compose_units.items())},
        "n_units": int(n_units),
        "n_transfer_units": int(n_transfer_units),
        "transfer_unit_s": float(per_transfer),
        "projected_wall_h": float(projected_h),
        "fence_wall_h": float(2.0 * projected_h),
        "plan_wall_h": float(plan_wall_h),
        "abort_mult": float(abort_mult),
        "abort": bool(projected_h > abort_mult * plan_wall_h),
    }


def _run_real(args: argparse.Namespace, timings: dict | None = None) -> int:
    import gc  # noqa: F401 — used by the round-8 memory-scoping frees below

    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits, mem_guard, store_io
    from explore_persona_space.experiments.issue_1739.constants import (
        COMPOSITION_F_L,
        COMPOSITION_F_U,
        N_LAYERS,
    )

    for req in ("labeled_store", "dv_json", "u_store", "e1_store"):
        if getattr(args, req) is None:
            raise SystemExit(f"real mode requires --{req.replace('_', '-')} (or use --synthetic N)")
    layers = args.layers or list(range(N_LAYERS))
    # new-arm-round item 1: --rb-point context_end swaps the r_B extraction
    # POINT and suffixes every regime label with '_fc' (unit/row keys included),
    # so fc rows can never collide with committed t1 rows at resume/merge time.
    fc = getattr(args, "rb_point", "t1") == "context_end"
    regimes_eff = [r + "_fc" for r in args.regimes] if fc else list(args.regimes)
    need_rollout_rows = any(r in ("e2", "e2p") for r in args.regimes)
    tbl = _load_labeled(
        args.labeled_store,
        args.dv_json,
        layers,
        config=args.config,
        need_rollout_rows=need_rollout_rows,
        rollout_rows_kind="context_end" if fc else "t1",
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
        regimes=tuple(regimes_eff),
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

    # C3 directions UP FRONT, then FREE the fp64 per-rollout answer rows —
    # round-8 memory scoping: after extraction they are dead weight (~tens of
    # GiB at production scale) squeezing the arm batteries' headroom.
    rb_cache: dict[str, np.ndarray] = {}
    for regime in regimes_eff:
        rb_cache[regime] = _extract_rb(regime, args, tbl, layers, dim)
        _save_rb(args.tensors_root, args.behavior, regime, rb_cache[regime], layers)
    if tbl.ans_rows is not None:
        freed = sum(a.nbytes for a in tbl.ans_rows.values())
        tbl.ans_rows = tbl.ans_row_ctx = tbl.ans_row_k = None
        gc.collect()
        print(
            f"[fits] freed per-rollout answer rows ({freed / 2**30:.2f} GiB) "
            "after direction extraction",
            flush=True,
        )

    # Group consecutive specs sharing a map_key: the regime slices of one
    # (variant, U) rung run as ONE run_grid_multi pass (shared whitening/map,
    # shared whitened labeled arrays, shared ridge/MLP factorizations).
    groups: list[list[RunSpec]] = []
    for spec in specs:
        if groups and _map_key(spec) == _map_key(groups[-1][0]):
            groups[-1].append(spec)
        else:
            groups.append([spec])

    all_records: list[dict] = []
    transfer_rows: list[dict] = []
    transfer_skips: list[dict] = []
    diag_out: dict = {}
    compose_skips: list[dict] = []
    for gi, group in enumerate(groups):
        spec0 = group[0]
        try:
            u_x, u_y, u_label, n_u = _u_pool_for_spec(spec0, u_arrays, u_fit_rows, tbl, layers)
        except (ValueError, RuntimeError) as exc:
            _record_compose_skip(spec0, exc, compose_skips)  # re-raises on a plain rung
            print(f"[fits] group {gi + 1}/{len(groups)} SKIP compose: {exc}", flush=True)
            if gi == len(groups) - 1:
                u_arrays = None
                gc.collect()
            continue
        if gi == len(groups) - 1:
            # Crash-fix r3 memory scoping: the LAST group's pool is composed —
            # the staged fp16 u_store summary arrays (~12 GiB at 3 kinds x 28
            # layers) are dead weight through the fit/transfer phases below.
            u_arrays = None
            gc.collect()
            print("[fits] freed u_store summary arrays after final U-pool compose", flush=True)
        t_map = time.time()
        map_kind = getattr(args, "map_kind", "linear")
        map_seed = int(args.seeds[0])
        wh = fits.fit_whitening(u_x, device=args.device, seed=map_seed)
        # Plain rungs persist a behavior-INDEPENDENT map; consume it when a
        # sibling invocation already fit it (see _load_nl_map). Nonlinear only.
        mapfit = (
            _load_nl_map(
                args.tensors_root,
                spec0.variant,
                u_label,
                map_kind,
                layers,
                n_u,
                device=args.device,
                map_seed=map_seed,
            )
            if spec0.f_u is None
            else None
        )
        map_source = "loaded" if mapfit is not None else "fit"
        # Pre-fit RSS guard (crash-fix r3): project this group's whitening-
        # apply + map-fit + labeled-whitening peak vs live MemAvailable and
        # refuse with a DESIGNED rc instead of a kernel OOM-kill (rc=137 on
        # the 85 GB a2-highgpu-1g boxes, 2026-08-02).
        mem_guard.check_phase(
            f"whitening_map[{spec0.variant}|{u_label}]",
            mem_guard.whitening_map_components(
                len(layers),
                n_u,
                dim,
                n_ctx=len(tbl.ctx_order),
                n_ev=len(tbl_ev.ctx_order) if tbl_ev is not None else 0,
                map_fit=mapfit is None,
            ),
            out_root=args.out_root,
        )
        # Held-out probe rows for the round-trip gate, captured from the SAME
        # whitened copy the map fit consumes (identical values to the old
        # fresh apply_whitening(u_x)[:, :K] slice).
        probe_x = None
        space_meta = None
        if mapfit is None:
            # ONE whitened fp64 copy per pool side; the fp16 stacked pools are
            # freed the moment their whitened twin exists (crash-fix r3 —
            # pre-fix BOTH pools + two whole-array fp64 apply transients were
            # co-resident, the 85 GB-box kill site).
            x_w = fits.apply_whitening(u_x, wh)
            u_x = None
            y_w = fits.apply_whitening(u_y, wh)
            u_y = None
            if spec0.f_u is None and map_kind != "linear":
                probe_x = np.array(x_w[:, : min(MAP_ROUNDTRIP_PROBE_ROWS, n_u)], copy=True)
            if spec0.f_u is None:
                # #1975: record the fit space + whitening provenance in the
                # persisted payload (RECIPE form — the whitening is fit
                # in-process here, no persisted artifact exists at fit time).
                # Computed BEFORE x_w is freed; a small dict, not the array.
                space_meta = fits.map_space_meta(
                    x_w,
                    fit_space="whitened",
                    whitening_prov=fits.whitening_provenance(
                        variant=spec0.variant,
                        u_label=u_label,
                        whiten_seed=map_seed,
                        n_u_rows=n_u,
                        gammas=wh.gamma,
                    ),
                )
            mapfit = _fit_map(args, x_w, y_w)
            del x_w, y_w
        else:
            u_x = u_y = None  # loaded map: the pools carry nothing downstream
        gc.collect()
        if timings is not None:
            timings.setdefault("map_fit_s", []).append(time.time() - t_map)
        diag_out[f"{spec0.variant}|{u_label}"] = {**mapfit.diagnostics, "map_source": map_source}
        z_ev_w = za_ev_w = None  # eval-split arrays, whitened per map_key (transfer leg)
        if spec0.f_u is None:
            # C-1: persist the frozen plain-rung map weights (HF-bound via
            # the tensors upload stage; behavior-independent, idempotent).
            fresh = not _map_path(args.tensors_root, spec0.variant, u_label, map_kind).exists()
            _save_map(
                args.tensors_root,
                spec0.variant,
                u_label,
                mapfit,
                layers,
                map_seed=map_seed,
                space_meta=space_meta,
            )
            if fresh and probe_x is not None:
                # Gate ONLY a payload THIS process wrote: an already-present file
                # is a sibling's fit, and comparing two independent fits would
                # test cross-device determinism instead of serialization.
                diag_out[f"{spec0.variant}|{u_label}"]["roundtrip_gate"] = _verify_map_roundtrip(
                    args.tensors_root,
                    spec0.variant,
                    u_label,
                    map_kind,
                    layers,
                    n_u,
                    mapfit,
                    probe_x,
                    map_seed=map_seed,
                    device=args.device,
                )
            del probe_x
            if tbl_ev is not None:
                z_ev_w = fits.apply_whitening(tbl_ev.z_by_variant[spec0.variant], wh)
                za_ev_w = (
                    fits.apply_whitening(tbl_ev.z_ans, wh) if tbl_ev.z_ans is not None else None
                )
                if za_ev_w is not None:
                    # Behavior-SPECIFIC second map-quality read; see
                    # _eval_rung_reconstruction on why it lands here and not in
                    # the shared payload.
                    diag_out[f"{spec0.variant}|{u_label}"]["eval_rung"] = _eval_rung_reconstruction(
                        mapfit,
                        z_ev_w,
                        za_ev_w,
                        # row_rungs = the PER-CONTEXT rung label aligned with
                        # ctx_order (tbl_ev.rungs is the DISTINCT rung list).
                        rungs=tbl_ev.row_rungs if getattr(args, "eval_rung_knn", False) else None,
                        knn=bool(getattr(args, "eval_rung_knn", False)),
                    )
        # ONE whitened fp64 copy per group, SHARED by identity across the
        # regime slices (the run_cell_multi contract) — the old per-spec
        # rebuild held n_regimes copies of identical arrays.
        z_var_w = fits.apply_whitening(tbl.z_by_variant[spec0.variant], wh)
        za_w = fits.apply_whitening(tbl.z_ans, wh)
        datas: list[arms.CellData] = []
        provs: list[dict] = []
        for spec in group:
            datas.append(
                arms.CellData(
                    z_ctx=z_var_w,
                    z_ans=za_w,
                    dv=tbl.dv,
                    rb=np.einsum("ld,lde->le", rb_cache[spec.regime], wh.w),
                    mapfit=mapfit,
                    text_emb=text_emb,
                    text_features=text_features,
                    layers=tuple(layers),
                    per_rollout=tbl.per_rollout,
                )
            )
            # rb_point / fixed_coordinate ride provenance (hence unit/row keys)
            # ONLY when set — committed t1 unit keys stay byte-identical.
            prov_extra: dict = {}
            if fc:
                prov_extra["rb_point"] = "context_end"
            if getattr(args, "fixed_coordinate", None):
                prov_extra["fixed_coordinate"] = str(args.fixed_coordinate)
            provs.append(
                {
                    "behavior": args.behavior,
                    "variant": spec.variant,
                    "regime": spec.regime,
                    "u_rung": int(n_u),
                    "u_rung_label": u_label,
                    "eval_rung": ",".join(tbl.rungs),
                    "config": args.config,
                    "f_u": spec.f_u,
                    "f_l": spec.f_l,
                    **prov_extra,
                }
            )
        kwargs = {}
        if args.n_boot:
            kwargs["n_boot"] = args.n_boot
        if args.n_perm:
            kwargs["n_perm"] = args.n_perm
        mlp_kwargs = {"max_epochs": args.mlp_epochs} if args.mlp_epochs else None
        t_grid = time.time()
        mem_guard.check_phase(
            f"grid[{spec0.variant}|{u_label}]",
            mem_guard.cell_solve_components(
                len(layers),
                min(max(int(b) for b in spec0.budgets), len(tbl.ctx_order)),
                dim,
                list(args.arms) if args.arms else list(arms.ARM_REGISTRY),
                has_map=mapfit is not None,
            ),
            out_root=args.out_root,
        )
        recs_by_regime = arms.run_grid_multi(
            datas,
            provs,
            tbl.groups,
            budgets=list(spec0.budgets),
            draws=list(spec0.draws),
            seeds=list(spec0.seeds),
            out_dir=args.out_root / "arm_results",
            arms=args.arms,
            device=args.device,
            mlp_kwargs=mlp_kwargs,
            context_ids=tbl.ctx_order,
            unit_timings=(timings.setdefault("units", []) if timings is not None else None),
            **kwargs,
        )
        if timings is not None:
            timings.setdefault("grid_s", []).append(time.time() - t_grid)
        for recs in recs_by_regime:
            all_records += recs
        if tbl_ev is not None and spec0.f_u is None:
            t_tf = time.time()
            mem_guard.check_phase(
                f"transfer[{spec0.variant}|{u_label}]",
                mem_guard.transfer_components(
                    len(layers),
                    min(max(int(b) for b in spec0.budgets), len(tbl.ctx_order)),
                    len(tbl_ev.ctx_order),
                    dim,
                    arms.resolve_transfer_roster(getattr(args, "transfer_arms", None)),
                    has_map=mapfit is not None,
                ),
                out_root=args.out_root,
            )
            rows_g, skips_g = _run_transfer_for_group(
                args,
                group,
                datas,
                provs,
                recs_by_regime,
                tbl,
                tbl_ev,
                rungs_ev,
                z_ev_w,
                za_ev_w,
                layers,
            )
            transfer_rows += rows_g
            transfer_skips += skips_g
            if timings is not None:
                timings.setdefault("transfer_s", []).append(time.time() - t_tf)
        print(
            f"[fits] group {gi + 1}/{len(groups)} done ({spec0.variant}/{u_label}/"
            + "+".join(s.regime for s in group)
            + ")",
            flush=True,
        )
        # Round-8 memory scoping: drop the group's whitened fp64 arrays +
        # release torch's allocator cache before the next group's map fit.
        del datas, z_var_w, za_w, z_ev_w, za_ev_w, wh, mapfit
        gc.collect()
        if str(args.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
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
            "regimes": list(regimes_eff),
            "rb_point": str(getattr(args, "rb_point", "t1")),
            "fixed_coordinate": getattr(args, "fixed_coordinate", None),
            "n_contexts": len(tbl.ctx_order),
            "layers": layers,
            "u_fit_rows": int(len(u_fit_rows)),
            "u_sizes": [s if s is not None else "full" for s in u_sizes],
            "eval_rungs": tbl.rungs,
            "compose_skips": compose_skips,
            "transfer_min_n": int(args.transfer_min_n) if tbl_ev is not None else None,
            "transfer_eval_rungs": sorted(tbl_ev.rungs) if tbl_ev is not None else None,
            "transfer_arms": sorted(arms.resolve_transfer_roster(args.transfer_arms))
            if tbl_ev is not None
            else None,
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


def _run_transfer_for_group(
    args: argparse.Namespace,
    group: list[RunSpec],
    datas: list,
    provs: list[dict],
    recs_by_regime: list[list[dict]],
    tbl: LabeledTable,
    tbl_ev: LabeledTable,
    rungs_ev,
    z_ev_w,
    za_ev_w,
    layers: list[int],
) -> tuple[list[dict], list[dict]]:
    """Distribution-shift ladder leg for one plain-rung regime GROUP (M-A).

    Per (L, draw, seed) unit: score every eval-split context with the
    the resolved transfer roster (``--transfer-arms``, default
    :data:`arms.TRANSFER_ARMS_WIDE`) fit on the FULL train cell (never on eval
    DV) and emit one row per (arm, eval rung) at the TRAIN-frozen layer,
    plus one ``train_in_split`` row per arm (the in-distribution anchor).
    Round-8 batching, output-identical per (unit, regime): the unit loop
    runs OUTSIDE the regime loop with two caches — the rb-INDEPENDENT arm-4
    ridge is fit once per realized ROW SET and shared across regimes /
    draws / seeds (identical row sets whenever budget_l >= n_ctx: at the
    top rung all (draw, seed) units share ONE fit), and the rb-dependent
    projection arms cache per (regime, row set, seed). The discarded
    reverse-fold ridge fit is skipped via ``ridge_folds=(0,)``.
    Checkpoint-per-unit (JSONL, resume keyed on every output-affecting
    flag — #722-r3, key grammar UNCHANGED) + one progress line per unit.
    """
    import hashlib

    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms, fits

    spec0 = group[0]
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
    roster = arms.resolve_transfer_roster(getattr(args, "transfer_arms", None))
    regime_extra = {
        "transfer": True,
        "transfer_min_n": int(args.transfer_min_n),
        "transfer_arms": sorted(roster),
        "n_eval_table": len(tbl_ev.ctx_order),
        "n_boot": n_boot,
        "layers_subset": [int(x) for x in layers],
    }
    rec_by_unit: list[dict[tuple, dict]] = []
    for recs in recs_by_regime:
        m: dict[tuple, dict] = {}
        for rec in recs:
            rows = rec.get("arms") or []
            if rows:
                r0 = rows[0]
                m[(r0["budget_l"], r0["draw"], r0["seed"])] = rec
        rec_by_unit.append(m)
    dv_ev = np.asarray(tbl_ev.dv, dtype=np.float64)
    # rb-INDEPENDENT arms (ridge/MLP fits over z / mp / za) depend only on the
    # realized ROW SET, so one fit is shared across the group's regimes; the
    # rb-DEPENDENT projections are cached per (regime, row set, seed). The
    # split is registry-driven so a widened roster routes its new fitted arms
    # (5/7/8/12) into the shared cache automatically instead of refitting them
    # once per regime.
    rb_indep, rb_dep = arms.partition_transfer_roster(roster)
    units = [(b, d, s) for b in spec0.budgets for d in spec0.draws for s in spec0.seeds]
    rows_all: list[dict] = []
    skips_all: list[dict] = []
    rbindep_cache: dict[str, tuple[dict, dict]] = {}
    rbdep_cache: dict[tuple, tuple[dict, dict]] = {}
    t0 = time.time()
    for k, (budget_l, draw, seed) in enumerate(units):
        cell = None
        for r, spec in enumerate(group):
            key = "transfer|" + arms._unit_key(provs[r], budget_l, draw, seed, regime_extra)
            if key in tdone:
                rows_all += tdone[key]["rows"]
                skips_all += tdone[key].get("skips", [])
                print(
                    f"[fits] transfer unit {k + 1}/{len(units)} SKIP (resume) "
                    f"{budget_l}/{draw}/{seed} regime={spec.regime}",
                    flush=True,
                )
                continue
            if cell is None:
                cell = fits.realize_budget_cell(tbl.groups, budget_l=budget_l, draw=draw, seed=seed)
            rec = rec_by_unit[r].get((budget_l, draw, seed))
            if rec is None:
                raise RuntimeError(
                    f"transfer unit {budget_l}/{draw}/{seed}: no matching train record "
                    "(main grid must run the same units first)"
                )
            rs_key = hashlib.sha1(cell.row_idx.tobytes()).hexdigest()
            if rb_indep and rs_key not in rbindep_cache:
                rbindep_cache[rs_key] = arms.run_transfer_cell(
                    datas[r],
                    cell,
                    z_ev_w,
                    dv_ev,
                    za_ev=za_ev_w,
                    arms=rb_indep,
                    device=args.device,
                    ridge_folds=(0,),
                )
            ck = (spec.regime, rs_key, int(seed))
            if rb_dep and ck not in rbdep_cache:
                rbdep_cache[ck] = arms.run_transfer_cell(
                    datas[r],
                    cell,
                    z_ev_w,
                    dv_ev,
                    za_ev=za_ev_w,
                    arms=rb_dep,
                    device=args.device,
                    ridge_folds=(0,),
                )
            s4, sk4 = rbindep_cache.get(rs_key, ({}, {}))
            sd, skd = rbdep_cache.get(ck, ({}, {}))
            scores_ev = {**sd, **s4}
            arm_skips = {**skd, **sk4}
            frozen_by_arm = {
                row["arm"]: arms.frozen_layer_idx(row["rho_per_layer"]) for row in rec["arms"]
            }
            rows_u, skips_u = arms.evaluate_transfer(
                scores_ev,
                tbl_ev.dv,
                rungs_ev,
                frozen_by_arm,
                provenance=provs[r],
                cell=cell,
                layers=tuple(layers),
                n_boot=n_boot,
                min_n=int(args.transfer_min_n),
            )
            skips_u += [
                {"arm": slug, "reason": reason, "budget_l": budget_l, "draw": draw, "seed": seed}
                for slug, reason in sorted(arm_skips.items())
            ]
            skips_u += arms.roster_accounting_skips(
                roster, scores_ev, arm_skips, budget_l=budget_l, draw=draw, seed=seed
            )
            if getattr(args, "transfer_preds", False):
                # Per-(arm, eval context) frozen-layer OOD predictions — the
                # eval-rung twin of the train setting's `preds/*.npz` sidecar,
                # via the SAME reviewed helper the wcrung/pvsynth rung runners
                # already use. Written BEFORE the checkpoint line so a resumed
                # unit (which skips this block with its rows already recorded)
                # never leaves a half-written sidecar; one file per unit,
                # truncate-and-replace (write_preds_jsonl), keyed on the unit
                # key's sha so a re-run of a unit overwrites exactly its own
                # rows. `rung` rides the generic label column, so an OOD
                # scatter is a pure post-hoc read of this file.
                arms.write_preds_jsonl(
                    tpath.parent / "transfer_preds" / (_key_sha(key) + ".jsonl"),
                    arms.transfer_preds_rows(
                        scores_ev,
                        dv_ev,
                        tbl_ev.ctx_order,
                        frozen_by_arm,
                        provenance={
                            **provs[r],
                            "budget_l": int(budget_l),
                            "draw": int(draw),
                            "seed": int(seed),
                            "n_eval_pooled": len(tbl_ev.ctx_order),
                        },
                        layers=tuple(layers),
                        labels={"rung": [str(x) for x in rungs_ev]},
                    ),
                )
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
    """§9 pilot gate: one production-shape unit-GROUP PER BUDGET (round 8).

    Runs full-U / first-variant / ALL-regime unit-groups at every L budget
    for (draws[0], seeds[0]) via :func:`_run_real` (same out-root, so the
    pilot units RESUME into the full run), so the measured per-budget group
    walls carry BOTH the regime-shared factorizations and every regime's
    marginal cost — the projection basis of the new
    :func:`compose_pilot_report`. Writes ``pilot_report.json`` and exits
    :data:`PILOT_ABORT_RC` when the projection exceeds
    ``--pilot-abort-mult x --plan-wall-h`` (a designed halt with a report
    artifact — never a bare rc=1). A fully-RESUMED pilot re-run measures no
    unit walls (all keys skip) and degrades to a transfer/map-only
    projection — the gate has then already passed once on this out-root.
    """
    import copy
    from collections import Counter

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
    n_map_fits = len({_map_key(s) for s in full_specs})
    plain_specs = [s for s in full_specs if s.f_u is None]
    n_plain_map_keys = len({_map_key(s) for s in plain_specs})
    n_plain_groups = {
        int(b): n_plain_map_keys * len(args.draws) * len(args.seeds) for b in args.budgets
    }
    n_compose_units = dict(Counter(int(s.budgets[0]) for s in full_specs if s.f_u is not None))
    n_transfer = (
        sum(len(s.budgets) * len(s.draws) * len(s.seeds) for s in plain_specs)
        if args.transfer
        else 0
    )
    p = copy.copy(args)
    p.pilot = False
    p.variant = VARIANTS[0] if args.variant == "both" else args.variant
    p.regimes = list(args.regimes)
    p.u_sizes = ["full"]
    p.budgets = list(args.budgets)
    p.draws = [args.draws[0]]
    p.seeds = [args.seeds[0]]
    p.compose = False
    timings: dict = {}
    t0 = time.time()
    rc = _run_real(p, timings=timings)
    pilot_wall = time.time() - t0
    map_walls = timings.get("map_fit_s", [])
    map_fit_s = float(sum(map_walls) / len(map_walls)) if map_walls else 0.0
    unit_group_walls = {int(u["budget_l"]): float(u["wall_s"]) for u in timings.get("units", [])}
    transfer_s = float(sum(timings.get("transfer_s", [0.0])))
    n_pilot_transfer_units = (len(p.budgets) * len(p.regimes)) if args.transfer else 0
    report = compose_pilot_report(
        n_map_fits=n_map_fits,
        map_fit_s=map_fit_s,
        unit_group_walls=unit_group_walls,
        n_plain_groups=n_plain_groups,
        n_compose_units=n_compose_units,
        transfer_s=transfer_s,
        n_pilot_transfer_units=n_pilot_transfer_units,
        n_transfer_units=n_transfer,
        plan_wall_h=args.plan_wall_h,
        abort_mult=args.pilot_abort_mult,
        n_units=n_units,
    )
    report.update(
        {
            "behavior": args.behavior,
            "pilot_wall_s": float(pilot_wall),
            "pilot_unit": {
                "variant": p.variant,
                "regimes": list(p.regimes),
                "u_size": "full",
                "budgets": [int(b) for b in p.budgets],
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
    walls_str = " ".join(f"L{b}={w:.1f}s" for b, w in sorted(unit_group_walls.items()))
    print(
        f"[fits] pilot: unit_group_walls[{walls_str}] map_fit_s={map_fit_s:.1f} "
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
    from explore_persona_space.experiments.issue_1739.mem_guard import (
        RSS_GUARD_RC,
        MemGuardRefusal,
    )

    args = _parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        if args.pilot:
            if args.synthetic:
                raise SystemExit("--pilot is a real-mode gate (drop --synthetic)")
            rc = _run_pilot(args)
        else:
            rc = _run_synthetic(args) if args.synthetic else _run_real(args)
    except MemGuardRefusal as exc:
        # Designed halt (crash-fix r3): report artifact + distinct rc — never
        # a kernel OOM-kill that loses the log tail, never a bare rc=1.
        print(f"[fits][rss-guard] DESIGNED HALT rc={RSS_GUARD_RC}: {exc}", flush=True)
        rc = RSS_GUARD_RC
    print(f"[fits] done rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.stdout.flush()
    sys.exit(main())
