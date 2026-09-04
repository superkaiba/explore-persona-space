#!/usr/bin/env python3
"""Matched-training-size rank analysis for the issue-2588 mapping panel.

The panel's reduced-rank ranks were measured on maps fitted on whatever
train_10k rows survived truncation (cells lost 2 to 53 percent of their rows),
and the rank at +10% relative validation SSE grows roughly as n_train^0.4 with
no plateau at 10k rows, so ranks fitted at unequal realized sizes are not
comparable across cells.  This script refits every map at common training
sizes (default 2500 and 4500 rows, 3 seeded nested subsets each) plus the
realized full split, mirroring the production estimator exactly (subset-own
fp64 mean, unbiased std + 1e-9 standardization, y centering, primal ridge at
the frozen production lambda) by importing the fit paths of
issue2588_rank_vs_ntrain (dual-form ridge below n = d and a top-k partial
eigendecomposition of the fitted-output second-moment matrix).  Per map it
reports the mean rank and pooled R^2 at each matched size, the log-log slope
of rank versus n with a bootstrap-over-seeds interval, and the ratio of the
full-split rank to the rank at the largest matched size.

Compute character: CPU only.  Runs unchanged under either cap profile
(EPS_CAP_PROFILE), reading fit records and activation shards from the
profile's HF prefix and writing under the profile's own EVAL_ROOT and
FIG_ROOT.  Fits already present in the output JSON are skipped, so long runs
split into bounded per-map invocations via --maps and --max-new-fits.  The
full 23-map panel run belongs on the cluster or a CPU pod, not the shared VM.
Maps whose fit record or activation shards cannot be downloaded (cells land
over time under the long profile) are skipped and listed in the output JSON,
never silently dropped.

Estimator-validity note: at d = 5120 the n = 2500 and n = 4500 points are
under-determined (n < d) by design.  The point of matching is comparability
across cells, not an absolute rank.  The ranks are estimator-and-n
conditional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2588_mapping_rank_vs_capability as MR  # noqa: E402
import issue2588_rank_vs_ntrain as RN  # noqa: E402

DEFAULT_N_GRID = (2500, 4500)
DEFAULT_SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 2588
OUT_JSON = MR.EVAL_ROOT / "matched_n_ranks.json"
FIG_PNG = MR.FIG_ROOT / "matched_n_ranks.png"
ARM_COLORS = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}

NOTES = (
    "Matched-training-size ranks exist for cross-cell comparability: cells lost 2 to 53 "
    "percent of their train_10k rows to truncation, so ranks fitted on whatever rows "
    "survived are not comparable across cells. Each map is refit at common training sizes "
    "plus the realized full split, at the frozen production lambda, mirroring the "
    "production estimator exactly. At d = 5120 the n = 2500 and n = 4500 points are "
    "under-determined (n < d) by design. The point of matching is comparability across "
    "cells, not an absolute rank. The ranks are estimator-and-n conditional."
)
SLOPE_RULE = (
    "OLS slope of log(rank_rel10) on log(n_train) over the matched subset fits plus the "
    "realized full-split fit, all at the fixed production lambda. The bootstrap interval "
    "resamples seeds with replacement within each matched size (the production point stays "
    f"fixed), {BOOTSTRAP_DRAWS} draws, 2.5 and 97.5 percentiles."
)


def resolve_specs(tokens: list[str] | None) -> list[MR.MapSpec]:
    """Resolve --maps tokens (cell names or full keys) against MR.MAPS."""
    if not tokens:
        return list(MR.MAPS)
    by_token: dict[str, MR.MapSpec] = {}
    for spec in MR.MAPS:
        by_token[spec.cell] = spec
        by_token[spec.key] = spec
    specs: list[MR.MapSpec] = []
    for token in tokens:
        if token not in by_token:
            known = ", ".join(s.cell for s in MR.MAPS)
            raise SystemExit(f"unknown map {token!r}; known cells: {known}")
        specs.append(by_token[token])
    return specs


def plan_for(
    n_total: int, n_grid: tuple[int, ...], seeds: tuple[int, ...]
) -> list[tuple[int, int | None]]:
    plan: list[tuple[int, int | None]] = [(n_total, None)]
    for n in n_grid:
        if n < n_total:
            plan.extend((n, s) for s in seeds)
    return plan


def map_complete(
    spec: MR.MapSpec,
    fits: dict[str, dict[str, Any]],
    n_grid: tuple[int, ...],
    seeds: tuple[int, ...],
) -> bool:
    prod = [f for f in fits.values() if f["cell"] == spec.cell and f["is_production"]]
    if not prod:
        return False
    planned = plan_for(int(prod[0]["n_train"]), n_grid, seeds)
    return all(RN.fit_id(spec.cell, n, s) in fits for n, s in planned)


def load_map_inputs(spec: MR.MapSpec) -> dict[str, Any]:
    """Download the fit record and activation splits for one map.

    This is the only phase whose failures the caller may record as a skipped
    map (data lands on HF over time under the long profile). Everything after
    it fails fast.
    """
    fitrec = MR._fit_record(spec)
    layer = int(fitrec["layer_star"])
    star = fitrec["layers"][str(layer)]
    d = int(star["d"])
    lam_prod = float(star["fit_meta"]["selected_lambda"])
    prod_expect = (float(star["fit_meta"]["val_r2_at_selected"]), float(star["test_r2"]))
    print(f"[{spec.key}] load splits at L{layer} (d={d}, lambda={lam_prod:g})", flush=True)
    xtr, ytr = MR.load_split(spec, "train_10k", layer)
    xval, yval = MR.load_split(spec, "val_400", layer)
    xte, yte = MR.load_split(spec, "test_1000", layer)
    if xtr.shape[1] != d or ytr.shape[1] != d:
        raise RuntimeError(f"{spec.key}: dimension mismatch X={xtr.shape} Y={ytr.shape} d={d}")
    return {
        "layer": layer,
        "d": d,
        "lam_prod": lam_prod,
        "prod_expect": prod_expect,
        "xtr": xtr,
        "ytr": ytr,
        "xval": xval,
        "yval": yval,
        "xte": xte,
        "yte": yte,
    }


def run_map(
    spec: MR.MapSpec,
    inputs: dict[str, Any],
    fits: dict[str, dict[str, Any]],
    n_grid: tuple[int, ...],
    seeds: tuple[int, ...],
    budget: list[int],
    saver,
) -> None:
    n_total = int(inputs["xtr"].shape[0])
    if n_total <= max(seeds) + 1:
        raise RuntimeError(f"{spec.key}: train split has only {n_total} rows")
    perms = {s: np.random.default_rng(s).permutation(n_total) for s in seeds}
    for n, seed in plan_for(n_total, n_grid, seeds):
        key = RN.fit_id(spec.cell, n, seed)
        if key in fits:
            continue
        if budget[0] <= 0:
            print(f"[{spec.key}] fit budget exhausted, stopping (resumable)", flush=True)
            return
        idx = None if seed is None else np.sort(perms[seed][:n])
        fits[key] = RN.fit_subset(
            spec,
            d=inputs["d"],
            layer=inputs["layer"],
            lam_prod=inputs["lam_prod"],
            xtr=inputs["xtr"],
            ytr=inputs["ytr"],
            xval=inputs["xval"],
            yval=inputs["yval"],
            xte=inputs["xte"],
            yte=inputs["yte"],
            idx=idx,
            n=n,
            seed=seed,
            prod_expect=inputs["prod_expect"] if seed is None else None,
        )
        budget[0] -= 1
        saver()


def _ols_slope(points: list[tuple[float, float]]) -> float:
    x = np.log(np.asarray([p[0] for p in points], dtype=np.float64))
    y = np.log(np.asarray([p[1] for p in points], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def slope_with_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for f in rows:
        if f["rank_rel10"] < 1:
            raise RuntimeError(f"rank_rel10 < 1 in fit {f['id']}, log-log slope undefined")
    matched: dict[int, list[int]] = {}
    for f in rows:
        if not f["is_production"]:
            matched.setdefault(int(f["n_train"]), []).append(int(f["rank_rel10"]))
    prod_pts = [(float(f["n_train"]), float(f["rank_rel10"])) for f in rows if f["is_production"]]
    points = [(float(n), float(r)) for n, ranks in matched.items() for r in ranks] + prod_pts
    if len({p[0] for p in points}) < 2:
        raise RuntimeError("need at least two distinct n values for a log-log slope")
    slope = _ols_slope(points)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for i in range(BOOTSTRAP_DRAWS):
        boot = list(prod_pts)
        for n, ranks in matched.items():
            take = rng.integers(0, len(ranks), size=len(ranks))
            boot.extend((float(n), float(ranks[j])) for j in take)
        draws[i] = _ols_slope(boot)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "value": slope,
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "n_draws": BOOTSTRAP_DRAWS,
        "n_points": len(points),
        "rule": SLOPE_RULE,
    }


def summarize(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for spec in MR.MAPS:
        rows = [f for f in fits.values() if f["cell"] == spec.cell]
        if not rows:
            continue
        by_n: dict[str, Any] = {}
        for n in sorted({int(f["n_train"]) for f in rows if not f["is_production"]}):
            grp = [f for f in rows if int(f["n_train"]) == n and not f["is_production"]]
            ranks = [int(f["rank_rel10"]) for f in grp]
            by_n[str(n)] = {
                "n_fits": len(grp),
                "mean_rank_rel10": float(np.mean(ranks)),
                "min_rank_rel10": int(min(ranks)),
                "max_rank_rel10": int(max(ranks)),
                "mean_rank_abs02": float(np.mean([f["rank_abs02"] for f in grp])),
                "mean_dirs_90pct": float(np.mean([f["dirs_90pct"] for f in grp])),
                "mean_full_val_r2": float(np.mean([f["full_val_r2"] for f in grp])),
                "mean_full_test_r2": float(np.mean([f["full_test_r2"] for f in grp])),
            }
        entry: dict[str, Any] = {
            "cell": spec.cell,
            "model_display": MR.DISPLAY_NAMES[spec.model_label],
            "arm": spec.arm,
            "arm_display": RN.ARM_DISPLAY[spec.arm],
            "d": int(rows[0]["d"]),
            "layer_star": int(rows[0]["layer_star"]),
            "by_n": by_n,
        }
        prod = [f for f in rows if f["is_production"]]
        if prod:
            p = prod[0]
            entry["production_n"] = int(p["n_train"])
            entry["production_rank_rel10"] = int(p["rank_rel10"])
            entry["production_full_val_r2"] = float(p["full_val_r2"])
            entry["production_full_test_r2"] = float(p["full_test_r2"])
            if by_n:
                hi = max(int(k) for k in by_n)
                entry["rank_ratio_full_over_matched_hi"] = {
                    "matched_n": hi,
                    "ratio": float(p["rank_rel10"] / by_n[str(hi)]["mean_rank_rel10"]),
                }
                entry["slope_loglog"] = slope_with_bootstrap(rows)
        summary[spec.key] = entry
    return summary


def save(
    fits: dict[str, dict[str, Any]],
    skipped: list[dict[str, str]],
    n_grid: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    cell_order = {s.cell: i for i, s in enumerate(MR.MAPS)}
    rows = sorted(
        fits.values(),
        key=lambda f: (cell_order[f["cell"]], f["n_train"], -1 if f["seed"] is None else f["seed"]),
    )
    payload = {
        "schema_version": "issue2588_matched_n_ranks_v1",
        "cap_profile": MR.CAP_PROFILE,
        "hf_revision": MR.HF_REVISION,
        "notes": NOTES,
        "rank_definition": MR.PRIMARY_RANK_DEFINITION,
        "slope_rule": SLOPE_RULE,
        "n_grid": list(n_grid),
        "seeds": list(seeds),
        "skipped_maps": skipped,
        "summary": summarize(fits),
        "fits": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if not OUT_JSON.exists():
        return {}, []
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {r["id"]: r for r in payload["fits"]}, list(payload.get("skipped_maps", []))


def _err(lo: float, value: float, hi: float) -> list[list[float]]:
    """Non-negative errorbar offsets from interval bounds (never raw deltas)."""
    return [[max(0.0, value - lo)], [max(0.0, hi - value)]]


def render_figure(fits: dict[str, dict[str, Any]], n_grid: tuple[int, ...]) -> bool:
    summary = summarize(fits)
    n_star = max(n_grid)
    plotted = 0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for entry in summary.values():
        color = ARM_COLORS[entry["arm"]]
        d = entry["d"]
        bucket = entry["by_n"].get(str(n_star))
        if bucket is not None:
            mean = bucket["mean_rank_rel10"]
            axes[0].errorbar(
                [d],
                [mean],
                yerr=_err(bucket["min_rank_rel10"], mean, bucket["max_rank_rel10"]),
                fmt="o",
                color=color,
                markersize=6,
                capsize=3,
            )
            plotted += 1
        if "production_rank_rel10" in entry:
            axes[0].scatter(
                [d],
                [entry["production_rank_rel10"]],
                s=52,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
            )
        slope = entry.get("slope_loglog")
        if slope is not None:
            axes[1].errorbar(
                [d],
                [slope["value"]],
                yerr=_err(slope["boot_lo"], slope["value"], slope["boot_hi"]),
                fmt="o",
                color=color,
                markersize=6,
                capsize=3,
            )
    if plotted == 0:
        plt.close(fig)
        print("nothing complete to plot yet, figure skipped", flush=True)
        return False
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=ARM_COLORS["no-thinking"],
            label=f"{RN.ARM_DISPLAY['no-thinking']}, matched n = {n_star}",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=ARM_COLORS["end-of-thought"],
            label=f"{RN.ARM_DISPLAY['end-of-thought']}, matched n = {n_star}",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="grey",
            color="grey",
            label="production split, unmatched n",
        ),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="best")
    axes[0].set_title(f"Rank at matched n = {n_star}")
    axes[0].set_ylabel("rank at +10% relative validation SSE")
    axes[1].set_title("Rank growth with training size")
    axes[1].set_ylabel("log-log slope of rank vs n_train")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("hidden width d")
    fig.tight_layout()
    FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=200)
    fig.savefig(FIG_PNG.with_suffix(".pdf"))
    plt.close(fig)
    meta = {
        "title": "Matched-n mapping ranks",
        "description": (
            f"Per map: the rank at +10% relative validation SSE refit at matched n = {n_star} "
            "(filled marker, hollow marker is the unmatched production-split rank) and the "
            "log-log slope of rank versus training size with a bootstrap-over-seeds interval."
        ),
        "source_data": str(OUT_JSON.relative_to(MR.REPO)),
        "public_url": f"https://eps.superkaiba.com/tasks/2588/figure/{FIG_PNG.name}",
        "cap_profile": MR.CAP_PROFILE,
        "hf_revision": MR.HF_REVISION,
        "panels": {
            "left": (
                f"rank at matched n = {n_star} vs hidden width, one filled marker per map "
                "(seed min/max bars), arm as color, production rank hollow on the same x"
            ),
            "right": "log-log slope of rank vs n_train per map, bootstrap 95% interval bars",
        },
    }
    FIG_PNG.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"figure written: {FIG_PNG}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--maps", default=None, help="comma-separated cells or keys to restrict")
    parser.add_argument("--n-grid", default=",".join(str(n) for n in DEFAULT_N_GRID))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--max-new-fits", type=int, default=10**9)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    n_grid = tuple(sorted({int(t) for t in args.n_grid.split(",") if t.strip()}))
    seeds = tuple(int(t) for t in args.seeds.split(",") if t.strip())
    if not n_grid or not seeds:
        raise SystemExit("--n-grid and --seeds must be non-empty")
    specs = resolve_specs(args.maps.split(",") if args.maps else None)
    fits, skipped = load()
    if not args.render_only:
        RN.check_disk()
        requested = {s.key for s in specs}
        skipped = [e for e in skipped if e["key"] not in requested]
        budget = [args.max_new_fits]
        saver = lambda: save(fits, skipped, n_grid, seeds)  # noqa: E731
        for spec in specs:
            if budget[0] <= 0:
                break
            if map_complete(spec, fits, n_grid, seeds):
                print(f"[{spec.key}] all fits present, skipping", flush=True)
                continue
            try:
                inputs = load_map_inputs(spec)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"[{spec.key}] SKIPPED (data unavailable): {reason}", flush=True)
                skipped.append({"key": spec.key, "error": reason})
                continue
            run_map(spec, inputs, fits, n_grid, seeds, budget, saver)
            del inputs
        save(fits, skipped, n_grid, seeds)
    render_figure(fits, n_grid)


if __name__ == "__main__":
    main()
