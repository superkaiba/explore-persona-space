#!/usr/bin/env python3
"""MEASURED per-cell pilot for the widened eval-rung transfer roster (#1739).

Sizing basis for the grid-fill round: how much wall does adding the four fitted
arms (5/7/8/12) cost a transfer cell, at PRODUCTION shape, through the
PRODUCTION entrypoint (``arms.run_transfer_cell`` — the exact call every eval-
rung leg makes)? An asserted per-cell cost is never a sizing basis
(``.claude/rules/plan-compute-sizing.md`` § Per-cell fit phases), so this
measures core-vs-wide on synthetic arrays cut to the realized rung dimensions.

Defaults mirror the realized wildchat/pvsynth legs (``all_arms_spearman.json``
meta): ``n_train=6468``, ``n_eval=1982``, ``d=3584``. LAYERS defaults to 4 (not
28) so the pilot itself stays cheap on the shared VM; the ridge is layer-batched
(``ridge_gcv_predict_per_target``, ``layer_chunk=4``) and the MLP is batched over
layers, so wall scales ~linearly in the layer count — the report prints both the
measured figure and the x(28/LAYERS) extrapolation, labeled as such.

Synthetic ARRAYS, production SHAPES + production CODE PATH: this measures the
solver cost, which depends on shape and dtype, not on the values. It is a
throughput probe, never a science read.

Run with the shared-VM thread caps:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
      uv run python scripts/issue1739_transfer_roster_pilot.py
"""

from __future__ import annotations

import argparse
import json
import resource
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-train", type=int, default=6468, help="realized train contexts")
    ap.add_argument("--n-eval", type=int, default=1982, help="realized eval-rung contexts")
    ap.add_argument("--d", type=int, default=3584, help="hidden dim (Qwen-2.5-7B)")
    ap.add_argument("--layers", type=int, default=4, help="layers to MEASURE (extrapolated to 28)")
    ap.add_argument("--full-layers", type=int, default=28, help="production layer count")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--per-arm",
        action="store_true",
        help="also measure each WIDENED arm's MARGINAL cost (core, then core+one-arm each), "
        "so a headline ratio can be attributed to the arm that actually carries it",
    )
    ap.add_argument("--report-json", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np

    from explore_persona_space.experiments.issue_1739 import arms
    from explore_persona_space.experiments.issue_1739.fits import BudgetCell, MapFit

    ly, d, n_tr, n_ev = args.layers, args.d, args.n_train, args.n_eval
    rng = np.random.default_rng(1739)
    print(
        f"[pilot] shape: layers={ly} d={d} n_train={n_tr} n_eval={n_ev} device={args.device} "
        f"(ridge regime: {'primal d x d' if n_tr > d else 'dual n x n'})",
        flush=True,
    )
    z = rng.normal(size=(ly, n_tr, d))
    za = z + 0.25 * rng.normal(size=(ly, n_tr, d))
    dv = rng.normal(size=n_tr)
    z_ev = rng.normal(size=(ly, n_ev, d))
    za_ev = rng.normal(size=(ly, n_ev, d))
    dv_ev = rng.normal(size=n_ev)
    mapfit = MapFit(
        w=np.stack([np.eye(d) for _ in range(ly)]),
        x_mu=np.zeros((ly, 1, d)),
        x_sd=np.ones((ly, 1, d)),
        y_mu=np.zeros((ly, 1, d)),
        diagnostics={},
        kind="linear",
    )
    data = arms.CellData(
        z_ctx=z,
        z_ans=za,
        dv=dv,
        rb=rng.normal(size=(ly, d)),
        mapfit=mapfit,
        layers=tuple(range(ly)),
    )
    cell = BudgetCell(
        row_idx=np.arange(n_tr),
        fold_ids=np.arange(n_tr) % 5,
        n_folds=5,
        budget_l=n_tr,
        draw=0,
        seed=0,
        fold_scheme="pilot",
    )

    rosters: list[tuple[str, tuple[str, ...]]] = [
        ("core", arms.TRANSFER_ARMS),
        ("wide", arms.TRANSFER_ARMS_WIDE),
    ]
    if args.per_arm:
        rosters += [
            (f"core+{slug}", arms.TRANSFER_ARMS + (slug,))
            for slug in arms.TRANSFER_ARMS_WIDE
            if slug not in arms.TRANSFER_ARMS
        ]

    out: dict[str, dict] = {}
    for name, roster in rosters:
        t0 = time.time()
        scores, skipped = arms.run_transfer_cell(
            data,
            cell,
            z_ev,
            dv_ev,
            za_ev=za_ev,
            arms=list(roster),
            device=args.device,
            ridge_folds=(0,),
        )
        wall = time.time() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB (Linux: KiB)
        out[name] = {
            "n_arms": len(roster),
            "wall_s_measured": round(wall, 1),
            "wall_s_extrapolated_full_layers": round(wall * args.full_layers / ly, 1),
            "n_scored": len(scores),
            "skipped": skipped,
            "peak_rss_gib_so_far": round(rss, 2),
        }
        print(
            f"[pilot] {name}: {len(roster)} arms, scored={len(scores)}, "
            f"measured={wall:.1f}s at {ly} layers -> "
            f"~{wall * args.full_layers / ly:.0f}s at {args.full_layers} layers "
            f"(peak RSS so far {rss:.1f} GiB)",
            flush=True,
        )

    core_x, wide_x = (
        out["core"]["wall_s_extrapolated_full_layers"],
        out["wide"]["wall_s_extrapolated_full_layers"],
    )
    out["summary"] = {
        "shape": {
            "layers_measured": ly,
            "layers_full": args.full_layers,
            "d": d,
            "n_train": n_tr,
            "n_eval": n_ev,
        },
        "wide_over_core_ratio": round(wide_x / core_x, 2) if core_x else None,
        "added_s_per_cell_full_layers": round(wide_x - core_x, 1),
        "marginal_s_per_arm_full_layers": {
            k.split("+", 1)[1]: round(v["wall_s_extrapolated_full_layers"] - core_x, 1)
            for k, v in out.items()
            if k.startswith("core+")
        },
        "basis": "MEASURED through arms.run_transfer_cell at production shape; "
        f"wall extrapolated x({args.full_layers}/{ly}) in the layer axis",
    }
    print("[pilot] " + json.dumps(out["summary"]), flush=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(out, indent=1))
        print(f"[pilot] report -> {args.report_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
