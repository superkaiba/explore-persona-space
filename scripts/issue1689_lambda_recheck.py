"""Issue #1689 follow-up `wider-lambda-ceilings` — Stage-1 λ-grid recheck driver.

Refits the per-cell-arm L19 within-cell ceilings (heldout pooled R², the same
`issue825_fit_cells.heldout_r2_sweep` inner-group-cv path the published
percell run used) under the WIDE 19-point λ grid logspace(-2, 7, 19) and
compares each ceiling against the PUBLISHED 13-grid value (plan v6 §3:
Δceiling = R²_wide19 − R²_published; the 19-grid is a strict superset of the
published grid at the same 0.5-dex spacing, so any Δ is attributable to the
added λs alone).

Phases (one entrypoint, subset-parameterized — smoke IS this driver at a
--cells/--arms subset):
  fit (default): per realized (cell, arm) from the --published dir listing,
      checkpoint per cell-arm under --out (skip-if-exists, regime-keyed).
  --pilot: plan §7 gate — the pinned pilot cell-arm through the SAME loop,
      BOTH grids; 13-grid must reproduce the published R² within 1e-3
      (fail loud), 19-grid wall + peak RSS measured + extrapolated.
  --merge: assemble summary.json + affected_pairs.json from the per-cell-arm
      checkpoints (fails loud on missing cell-arms vs the SAME published
      enumeration the fit loop iterates).

Workload command (plan v6 §10; one detached process per model, 2-concurrent):
  cd .claude/worktrees/issue-1689 && \
  nohup env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 MALLOC_ARENA_MAX=2 uv run python \
    scripts/issue1689_lambda_recheck.py \
    --store-root /mnt/eps-data/thomasjiralerspong/issue1689_procrustes/hf_dl \
    --model Qwen_Qwen2.5-7B --layer 19 --lambda-grid wide19 \
    --published eval_results/issue_1689/percell \
    --out eval_results/issue_1689/wider-lambda-ceilings/percell_wide19 \
    < /dev/null >> /tmp/issue1689_lambda_recheck_base.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing numpy / torch — the shared-VM
# thread caps (#847) freeze at first BLAS/torch import (lint-gate pin on this
# branch; see scripts/issue1689_procrustes_figure.py for the pattern).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    HEADLINE_LAYER,
    LAMBDA_GRIDS,
    MODEL_BASE,
    MODEL_INSTRUCT,
    N_FOLDS,
    SLUG_TO_CONDITION,
    enumerate_pair_set,
    resolve_lambda_grid,
)

FIT_SEED = 42  # published percell runs hardcode seed=42 (issue1689_fit_cells.fit_cell)
ARMS = ("prefix", "context")
KNOWN_MODEL_SLUGS = (MODEL_BASE.replace("/", "_"), MODEL_INSTRUCT.replace("/", "_"))

# Plan v6 §3 lattice constants.
MOVED_THRESHOLD = 0.02  # |Δceiling| > 0.02 => "moved"
MOVED_FRAC_BAR = 0.05  # moved_frac < 0.05 => Stands-as-published (≤4 of 84)
PARITY_TOL = 1e-3  # plan §7 pilot gate: 13-grid refit vs published R²

# Plan §7 pilot cell-arm (pinned; run with --model Qwen_Qwen2.5-7B-Instruct).
PILOT_CELL = "assistant_chat"
PILOT_ARM = "context"

# Plan v6 assumption 3: the 19-grid is a strict superset of the 13-grid
# (same 0.5-dex spacing => the 13 old points are bit-identical members).
_G13 = resolve_lambda_grid("ladder13")
_G19 = resolve_lambda_grid("wide19")
assert np.intersect1d(_G13, _G19).size == 13, "wide19 must be a strict superset of ladder13"
assert np.array_equal(_G19[: len(_G13)], _G13), "ladder13 must be the wide19 prefix"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata only, never kills a fit
        return "unknown"


def realized_conditions(published_dir: Path, model_slug: str) -> list[str]:
    """Realized cell enumeration = the published percell dir listing (never a
    hardcoded lattice — the brief's realized-enumeration rule). Fails loud on
    a filename whose condition slug is not in the #1689 condition table."""
    prefix = f"heldout_r2_{model_slug}_"
    conds = []
    for p in sorted(published_dir.glob(f"{prefix}*.json")):
        cond = p.name[len(prefix) : -len(".json")]
        if cond not in SLUG_TO_CONDITION:
            raise ValueError(f"unknown condition slug {cond!r} parsed from {p.name}")
        conds.append(cond)
    return conds


def load_published(published_dir: Path, model_slug: str, cond: str) -> dict:
    p = published_dir / f"heldout_r2_{model_slug}_{cond}.json"
    if not p.exists():
        raise FileNotFoundError(f"published percell JSON missing: {p}")
    return json.loads(p.read_text())


def published_l19(published: dict, arm: str, layer: int) -> tuple[float, list]:
    """(published R², published per-fold selected λs) at ``layer`` for one arm."""
    li = published["layers"].index(layer)
    r2 = float(published[arm]["held_out_r2_per_layer"][li])
    lams = published[arm]["lambdas_selected"][li]
    return r2, lams


def _load_cell_arrays(store_root: Path, model_slug: str, cond: str, layer: int) -> dict:
    import torch

    p = store_root / model_slug / cond / f"L{layer}.pt"
    if not p.exists():
        raise FileNotFoundError(f"store bundle missing: {p}")
    bundle = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "X_prefix": np.asarray(bundle["X_prefix"], dtype=np.float32),
        "X_context": np.asarray(bundle["X_context"], dtype=np.float32),
        "Y": np.asarray(bundle["Y"], dtype=np.float32),
        "conv_ids": np.asarray(bundle["conv_ids"]),
    }


def fit_arm_ceiling(
    X: np.ndarray, Y: np.ndarray, conv_ids: np.ndarray, lambdas: np.ndarray | None
) -> dict:
    """One single-layer inner-group-cv heldout sweep (the EXACT published fit
    path — issue825_fit_cells.heldout_r2_sweep; folds derive from conv_ids +
    seed only, so a single-layer call reproduces the published L19 fold
    partition bit-for-bit). ``lambdas=None`` = the parent module 13-grid
    (byte-identical published path); an array = the custom grid."""
    from scripts.issue825_fit_cells import heldout_r2_sweep

    sweep = heldout_r2_sweep(
        X[:, None, :],
        Y[:, None, :],
        conv_ids,
        n_folds=N_FOLDS,
        seed=FIT_SEED,
        null_draws=0,
        collect_cosines=False,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        frozen_layers=(),
        lambdas=lambdas,
    )
    lam_row = [None if not np.isfinite(v) else float(v) for v in sweep["gcv_lambda"][0]]
    return {
        "r2": float(sweep["r2_obs"][0]),
        "lambdas_selected": lam_row,
        "selector": sweep["lambda_selector"][0],
    }


def _meta(layer: int, lambda_grid: str, refit13: bool) -> dict:
    """Checkpoint regime key — every output-affecting knob (resume rule)."""
    return {
        "layer": int(layer),
        "n_folds": int(N_FOLDS),
        "seed": FIT_SEED,
        "lambda_grid": lambda_grid,
        "refit13": bool(refit13),
    }


def _ckpt_satisfies(prior: dict | None, want: dict) -> bool:
    """Regime match; a refit13=True checkpoint satisfies a refit13=False
    request (superset), never vice versa."""
    if not isinstance(prior, dict) or not isinstance(prior.get("meta"), dict):
        return False
    pm = prior["meta"]
    for k, v in want.items():
        if k == "refit13":
            if v and not pm.get("refit13", False):
                return False
        elif pm.get(k) != v:
            return False
    return True


def _atomic_write_json(path: Path, obj: dict) -> None:
    # Atomic same-dir tmp + replace (EXDEV rule: tmp INSIDE dest dir).
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2)
    tmp.replace(path)


def process_cell_arm(
    *,
    store_root: Path,
    published_dir: Path,
    out_dir: Path,
    model_slug: str,
    cond: str,
    arm: str,
    layer: int,
    lambda_grid: str,
    refit13: bool,
    arrays: dict | None = None,
) -> dict:
    """Fit one (cell, arm) under the wide grid (+ optional 13-grid refit),
    checkpointed idempotently. Returns the record (resumed or fresh)."""
    ckpt = out_dir / f"{model_slug}__{cond}__{arm}.json"
    want = _meta(layer, lambda_grid, refit13)
    if ckpt.exists():
        try:
            prior = json.loads(ckpt.read_text())
        except (json.JSONDecodeError, OSError):
            prior = None
        if _ckpt_satisfies(prior, want):
            print(
                f"[lambda_recheck]   {model_slug}/{cond} arm={arm} RESUME (checkpoint)", flush=True
            )
            return prior
        if prior is not None:
            print(
                f"[lambda_recheck]   {model_slug}/{cond} arm={arm} regime-mismatch -> recompute",
                flush=True,
            )

    published = load_published(published_dir, model_slug, cond)
    if arrays is None:
        arrays = _load_cell_arrays(store_root, model_slug, cond, layer)
    n = int(arrays["Y"].shape[0])
    # Plan §12 assumption 1 verify: staged store rows == published fit rows.
    if n != int(published["n_rows"]):
        raise RuntimeError(
            f"n-assert FAIL for {model_slug}/{cond}: store n={n} != published "
            f"n_rows={published['n_rows']} — staged store is not the percell input"
        )
    pub_r2, pub_lams = published_l19(published, arm, layer)

    grid = resolve_lambda_grid(lambda_grid)
    X = arrays[f"X_{arm}"]
    Y = arrays["Y"]
    conv_ids = arrays["conv_ids"]

    t0 = time.perf_counter()
    wide = fit_arm_ceiling(X, Y, conv_ids, grid)
    wall_wide = time.perf_counter() - t0

    rec13 = None
    wall_13 = None
    if refit13:
        t0 = time.perf_counter()
        # lambdas=None => the parent module LAMBDAS default, byte-identical to
        # the published percell call (never re-materialized caller-side).
        rec13 = fit_arm_ceiling(X, Y, conv_ids, None)
        wall_13 = time.perf_counter() - t0

    finite_lams = [v for v in wide["lambdas_selected"] if v is not None]
    edge = float(grid[-1])
    edge_hits = sum(1 for v in finite_lams if abs(v - edge) / edge < 1e-9)
    record = {
        "meta": {
            **want,
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "cell": cond,
        "model": model_slug,
        "arm": arm,
        "n_rows": n,
        "published_r2": pub_r2,
        "lambda_star_13": pub_lams,  # published per-fold selections at L<layer>
        "ceiling_r2_19": wide["r2"],
        "lambdas_selected_19": wide["lambdas_selected"],
        "lambda_star_19": max(finite_lams) if finite_lams else None,
        "selector_19": wide["selector"],
        "edge_hits_19": edge_hits,  # fold-fits at the NEW grid ceiling (1e7) => still clipped
        "delta_r2": wide["r2"] - pub_r2,  # plan §3: published value is the 13-grid side
        "ceiling_r2_13": None if rec13 is None else rec13["r2"],
        "lambdas_selected_13_refit": None if rec13 is None else rec13["lambdas_selected"],
        "repro_gap": None if rec13 is None else abs(rec13["r2"] - pub_r2),
        "wall_s": {"grid19": wall_wide, "grid13_refit": wall_13},
    }
    if rec13 is not None and record["repro_gap"] > PARITY_TOL:
        # Plan §7: parity failure = fail loud, never proceed on a mismatched
        # baseline. (The checkpoint is deliberately NOT written.)
        raise RuntimeError(
            f"13-grid reproduction gate FAIL for {model_slug}/{cond} arm={arm}: "
            f"refit R²={rec13['r2']:.6f} vs published {pub_r2:.6f} "
            f"(|gap|={record['repro_gap']:.2e} > {PARITY_TOL})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(ckpt, record)
    return record


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB -> GB (Linux)


def _pilot_extrapolation(
    published_dir: Path, pilot_slug: str, wall19_s: float, n_pilot: int
) -> dict:
    """Extrapolate the 84-cell-arm wall from the pilot's measured 19-grid wall.

    eigh dominates (~n³ per fold at 0.8n; inner eighs proportional), so each
    cell-arm is weighted by (n_cell / n_pilot)³. Both arms share n per cell.
    The naive (uniform) figure is also printed — the pilot cell is NOT the
    largest (base/user_lmsys cells are n=11400 vs the pilot's n=8484), so the
    weighted figure is the sizing basis and the fence is 2x it.
    """
    weighted_units = 0.0
    n_arms = 0
    for slug in KNOWN_MODEL_SLUGS:
        for cond in realized_conditions(published_dir, slug):
            pub = load_published(published_dir, slug, cond)
            w = (float(pub["n_rows"]) / float(n_pilot)) ** 3
            weighted_units += 2 * w  # both arms
            n_arms += 2
    serial_h = weighted_units * wall19_s / 3600.0
    naive_h = n_arms * wall19_s / 3600.0
    return {
        "n_cell_arms": n_arms,
        "pilot_wall19_s": wall19_s,
        "naive_serial_h": naive_h,
        "weighted_serial_h": serial_h,
        "weighted_wall_h_2proc": serial_h / 2.0,
        "fence_h_2proc": 2.0 * serial_h / 2.0,
    }


def cmd_fit(args) -> int:
    import torch

    torch.set_num_threads(args.threads)
    model_slug = args.model.replace("/", "_")
    if model_slug not in KNOWN_MODEL_SLUGS:
        raise ValueError(f"unknown model {args.model!r} (want one of {KNOWN_MODEL_SLUGS})")

    conds = realized_conditions(args.published, model_slug)
    if not conds:
        raise ValueError(f"no published percell JSONs for {model_slug} under {args.published}")
    arms = list(ARMS)
    refit13 = args.refit_13
    if args.pilot:
        conds = [PILOT_CELL]
        arms = [PILOT_ARM]
        refit13 = True
    if args.cells:
        wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
        missing = [c for c in wanted if c not in conds]
        if missing:
            raise ValueError(f"--cells not in realized enumeration: {missing}")
        conds = wanted
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise ValueError(f"--arms must be within {ARMS}, got {bad}")

    # Cell-arm unit list, sharded (checkpoint-per-unit; units are independent).
    units = [(c, a) for c in conds for a in arms]
    shard = units[args.shard_index :: args.num_shards] if args.num_shards > 1 else units
    print(
        f"[lambda_recheck] model={model_slug} layer={args.layer} grid={args.lambda_grid} "
        f"refit13={refit13} units={len(shard)}/{len(units)} "
        f"shard={args.shard_index}/{args.num_shards} threads={args.threads} pilot={args.pilot}",
        flush=True,
    )

    pilot_record = None
    arrays_cache: tuple[str, dict] | None = None
    for i, (cond, arm) in enumerate(shard):
        t0 = time.perf_counter()
        # Reuse the loaded arrays across the two arms of one cell (adjacent units).
        if arrays_cache is not None and arrays_cache[0] == cond:
            arrays = arrays_cache[1]
        else:
            arrays = _load_cell_arrays(args.store_root, model_slug, cond, args.layer)
            arrays_cache = (cond, arrays)
        rec = process_cell_arm(
            store_root=args.store_root,
            published_dir=args.published,
            out_dir=args.out,
            model_slug=model_slug,
            cond=cond,
            arm=arm,
            layer=args.layer,
            lambda_grid=args.lambda_grid,
            refit13=refit13,
            arrays=arrays,
        )
        pilot_record = rec
        print(
            f"[lambda_recheck] unit {i + 1}/{len(shard)} {model_slug}/{cond} arm={arm} "
            f"delta_r2={rec['delta_r2']:+.4f} edge_hits_19={rec['edge_hits_19']} "
            f"elapsed={time.perf_counter() - t0:.1f}s peak_rss={_peak_rss_gb():.1f}GB",
            flush=True,
        )

    if args.pilot:
        assert pilot_record is not None
        wall19 = pilot_record["wall_s"]["grid19"]
        if wall19 is None:
            print(
                "[lambda_recheck] PILOT resumed from checkpoint — wall not re-measured; "
                "delete the checkpoint to re-time",
                flush=True,
            )
            return 0
        ext = _pilot_extrapolation(args.published, model_slug, wall19, pilot_record["n_rows"])
        print(
            f"[lambda_recheck] PILOT {model_slug}/{PILOT_CELL} arm={PILOT_ARM} "
            f"n={pilot_record['n_rows']}: grid13_refit wall={pilot_record['wall_s']['grid13_refit']:.1f}s "
            f"(repro_gap={pilot_record['repro_gap']:.2e} <= {PARITY_TOL} PASS), "
            f"grid19 wall={wall19:.1f}s, peak RSS={_peak_rss_gb():.2f}GB",
            flush=True,
        )
        print(
            f"[lambda_recheck] PILOT extrapolation over {ext['n_cell_arms']} cell-arms: "
            f"naive serial {ext['naive_serial_h']:.2f}h; n^3-weighted serial "
            f"{ext['weighted_serial_h']:.2f}h -> {ext['weighted_wall_h_2proc']:.2f}h at "
            f"2-concurrent; FENCE (2x) = {ext['fence_h_2proc']:.2f}h",
            flush=True,
        )
    return 0


def _hist(values: list[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = "none" if v is None else f"{v:.6g}"
        out[key] = out.get(key, 0) + 1
    return out


def cmd_merge(args) -> int:
    """Assemble summary.json + affected_pairs.json from per-cell-arm
    checkpoints (enumeration = the SAME published-dir listing the fit loop
    iterates; fails loud on any missing cell-arm)."""
    grid = resolve_lambda_grid(args.lambda_grid)
    edge = float(grid[-1])
    records: list[dict] = []
    missing: list[str] = []
    pub_lam_values: list[float] = []
    wide_lam_values: list[float] = []
    for slug in KNOWN_MODEL_SLUGS:
        for cond in realized_conditions(args.published, slug):
            for arm in ARMS:
                p = args.out / f"{slug}__{cond}__{arm}.json"
                if not p.exists():
                    missing.append(p.name)
                    continue
                rec = json.loads(p.read_text())
                if rec.get("meta", {}).get("lambda_grid") != args.lambda_grid:
                    raise RuntimeError(
                        f"checkpoint {p.name} has lambda_grid="
                        f"{rec.get('meta', {}).get('lambda_grid')!r}, want {args.lambda_grid!r}"
                    )
                records.append(rec)
                pub_lam_values.extend(rec["lambda_star_13"])
                wide_lam_values.extend(rec["lambdas_selected_19"])
    if missing:
        raise RuntimeError(
            f"merge incomplete: {len(missing)} cell-arm checkpoints missing under "
            f"{args.out} (first: {missing[:5]})"
        )
    if not records:
        raise RuntimeError(f"merge found ZERO cell-arm checkpoints under {args.out}")

    movers = sorted(
        (r for r in records if abs(r["delta_r2"]) > MOVED_THRESHOLD),
        key=lambda r: -abs(r["delta_r2"]),
    )
    n_total = len(records)
    moved_frac = len(movers) / n_total
    verdict = "stands-as-published" if moved_frac < MOVED_FRAC_BAR else "grid-limited"
    edge_hit_records = [r for r in records if r["edge_hits_19"] > 0]
    n_edge_foldfits = sum(r["edge_hits_19"] for r in records)

    # Affected pairs (plan §4): pairs whose src or tgt cell-arm moved, per
    # model x arm — the fit_ladder --pairs-file input shape.
    moved_by = {
        slug: {
            arm: {r["cell"] for r in movers if r["model"] == slug and r["arm"] == arm}
            for arm in ARMS
        }
        for slug in KNOWN_MODEL_SLUGS
    }
    pairs = enumerate_pair_set()
    affected = {
        slug: {
            arm: sorted(
                [
                    list(p)
                    for p in pairs
                    if p[0] in moved_by[slug][arm] or p[1] in moved_by[slug][arm]
                ]
            )
            for arm in ARMS
        }
        for slug in KNOWN_MODEL_SLUGS
    }
    n_affected = sum(len(v) for m in affected.values() for v in m.values())

    summary = {
        "lambda_grid": args.lambda_grid,
        "layer": args.layer,
        "moved_threshold": MOVED_THRESHOLD,
        "moved_frac_bar": MOVED_FRAC_BAR,
        "n_cell_arms": n_total,
        "n_moved": len(movers),
        "moved_frac": moved_frac,
        "verdict": verdict,
        "movers": [
            {k: r[k] for k in ("model", "cell", "arm", "published_r2", "ceiling_r2_19", "delta_r2")}
            for r in movers
        ],
        "delta_r2_all": {f"{r['model']}/{r['cell']}/{r['arm']}": r["delta_r2"] for r in records},
        "lambda_hist_published_13": _hist(pub_lam_values),
        "lambda_hist_wide_19": _hist(wide_lam_values),
        "new_ceiling_edge": edge,
        "n_foldfits_at_new_ceiling": n_edge_foldfits,
        "n_cell_arms_at_new_ceiling": len(edge_hit_records),
        "cell_arms_at_new_ceiling": [
            f"{r['model']}/{r['cell']}/{r['arm']}" for r in edge_hit_records
        ],
        "n_affected_pair_arms": n_affected,
        "repro": {
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "numpy": np.__version__,
            "driver": "scripts/issue1689_lambda_recheck.py",
            "published_dir": str(args.published),
            "percell_dir": str(args.out),
        },
    }
    summary_dir = args.summary_dir or args.out.parent
    summary_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(summary_dir / "summary.json", summary)
    _atomic_write_json(summary_dir / "affected_pairs.json", affected)
    print(
        f"[lambda_recheck] MERGE: {n_total} cell-arms, n_moved={len(movers)} "
        f"(frac={moved_frac:.3f}) -> verdict={verdict}; fold-fits at new ceiling "
        f"{edge:.0e}: {n_edge_foldfits}; affected pair-arms: {n_affected}; wrote "
        f"{summary_dir / 'summary.json'} + {summary_dir / 'affected_pairs.json'}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store-root", type=Path, help="staged L19 store root (fit mode)")
    ap.add_argument(
        "--model", type=str, help="model slug or HF id (fit mode; one process per model)"
    )
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER)
    ap.add_argument(
        "--lambda-grid",
        choices=sorted(LAMBDA_GRIDS),
        default="wide19",
        help="the recheck grid (default wide19; ladder13 only useful for debugging parity)",
    )
    ap.add_argument(
        "--published",
        type=Path,
        default=Path("eval_results/issue_1689/percell"),
        help="published percell dir — the realized cell enumeration + 13-grid baselines",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="per-cell-arm checkpoint dir (plan: eval_results/issue_1689/"
        "wider-lambda-ceilings/percell_wide19)",
    )
    ap.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="where --merge writes summary.json + affected_pairs.json (default: parent of --out)",
    )
    ap.add_argument("--cells", type=str, default=None, help="comma-separated condition subset")
    ap.add_argument("--arms", type=str, default=None, help="comma-separated arm subset")
    ap.add_argument(
        "--refit-13",
        action="store_true",
        help="ALSO refit the 13-grid per cell-arm (parity gate vs published; pilot implies this)",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="plan §7 pilot gate: the pinned pilot cell-arm, BOTH grids, parity + wall/RSS, exit",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16, help="torch intra-op threads (plan §9: 16)")
    ap.add_argument("--merge", action="store_true", help="assemble summary + affected pairs")
    args = ap.parse_args()

    if args.merge:
        return cmd_merge(args)
    if not args.store_root or not args.model:
        ap.error("--store-root and --model are required in fit mode")
    return cmd_fit(args)


if __name__ == "__main__":
    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGABRT pointer. All outputs are
    # flushed/closed before this point; atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
