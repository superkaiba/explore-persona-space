#!/usr/bin/env python
"""Project the issue-1739 nonlinear-map fan-out wall from the MEASURED basis.

Sizing a launch needs per-lane numbers, and the ONLY honest source for them is
the same arithmetic the in-run pilot gate uses —
``issue1739_fits.compose_pilot_report`` — driven by MEASURED per-call costs
(``pilot_report.json``), never by an asserted per-fit figure
(``.claude/rules/plan-compute-sizing.md`` § Per-cell fit phases; the #823
35-57x realized-wall miss came from exactly that substitution).

Mirrored model (see ``compose_pilot_report``):

    projected_s = n_map_fits * map_fit_s + plain_s + transfer_total
    plain_s     = sum_b  n_plain_groups[b] * unit_group_wall[b]
    n_plain_groups[b] = n_plain_map_keys * n_draws * n_seeds      (per budget b)
    n_plain_map_keys  = n_variants * n_u_rungs   (regimes SHARE a map key)
    n_transfer_units  = n_plain_specs * n_budgets * n_draws * n_seeds
                      = (n_variants * n_u_rungs * n_regimes) * ...

The fan-out's whole point is that ``n_map_fits`` is paid ONCE in phase A instead
of once per lane, so a lane's projection sets ``map_fit_s`` to 0 (its maps are
staged) while phase A carries the full map cost.

Emits per-lane ``plan_wall_h`` values for ``EPM_I1739_NL_PLAN_WALL_H`` so the
in-run pilot gate fences against a number derived from this basis rather than
the previous round's ceiling (a stale fence either false-fires or never fires).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# MEASURED basis — pilot_report.json figures from the round's own 1-cell pilots.
# NOT plan assertions. Re-measure and update when the shape changes.
# ---------------------------------------------------------------------------
MAP_FIT_S = 2651.32
UNIT_GROUP_WALL_S: dict[int, float] = {250: 4.87, 2500: 46.21, 8000: 140.08}
TRANSFER_UNIT_S = 28.81
# The top rung differs per behavior (evil 8000, others 16000); the basis measured
# 8000. Priced at the measured top-rung wall — flagged in the report as the one
# extrapolated cell (compose_pilot_report itself falls back to max(walls) for an
# unmeasured budget, so this mirrors the gate's own behavior).
TOP_RUNG_FALLBACK_S = 140.08

BEHAVIOR_REGIMES = {"evil": 3, "sycophancy": 3, "hallucination": 1}
BEHAVIOR_BUDGETS = {"evil": (250, 2500, 8000)}
DEFAULT_BUDGETS = (250, 2500, 16000)

# ---------------------------------------------------------------------------
# MEASURED basis for the LINEAR compose cells (scope addendum) — a DIFFERENT
# measurement from the nonlinear basis above: read verbatim from the committed
# per-behavior `eval_results/issue_1739/<b>/pilot_report.json` (linear map kind).
# ---------------------------------------------------------------------------
COMPOSE_MAP_FIT_S = {"hallucination": 221.9603476524353, "sycophancy": 161.81467533111572}
COMPOSE_UNIT_GROUP_WALL_S: dict[str, dict[int, float]] = {
    "hallucination": {250: 26.793855905532837, 2500: 135.74404406547546, 16000: 844.9238767623901}
}
# sycophancy's own linear pilot recorded `unit_group_walls_s: {}` (a resumed
# out-root skipped every unit), so its group walls are PROXIED from
# hallucination's — same 250/2500/16000 ladder, same map kind. Stated, not hidden.
COMPOSE_WALL_PROXY = {"sycophancy": "hallucination"}


@dataclass(frozen=True)
class Grid:
    """Path-2 reduced grid: 2 variants x 2 U rungs x 3 draws x 2 seeds."""

    n_variants: int = 2
    n_u_rungs: int = 2
    n_draws: int = 3
    n_seeds: int = 2
    transfer: bool = True


# Frozen module-level singleton: B008 allows a default read from one, and Grid
# is immutable so sharing it across calls is safe.
DEFAULT_GRID = Grid()


@dataclass
class LaneProjection:
    behavior: str
    kind: str
    n_regimes: int
    budgets: tuple[int, ...]
    n_plain_map_keys: int
    n_plain_groups_per_budget: int
    plain_s: float
    n_transfer_units: int
    transfer_s: float
    projected_h: float
    plan_wall_h: float
    terms: dict = field(default_factory=dict)


def wall_for_budget(budget: int) -> float:
    """Measured unit-group wall for a budget, falling back to the top rung."""
    return UNIT_GROUP_WALL_S.get(int(budget), TOP_RUNG_FALLBACK_S)


def budgets_for(behavior: str) -> tuple[int, ...]:
    """Per-behavior L grid — mirrors the dispatcher's behavior_budgets()."""
    return BEHAVIOR_BUDGETS.get(behavior, DEFAULT_BUDGETS)


def project_lane(
    behavior: str,
    kind: str,
    *,
    grid: Grid = DEFAULT_GRID,
    maps_staged: bool = True,
    fence_mult: float = 1.5,
) -> LaneProjection:
    """One (behavior, kind) scoring lane, maps STAGED (map_fit_s = 0).

    ``plan_wall_h`` is the projection times ``fence_mult`` — the value to hand
    ``EPM_I1739_NL_PLAN_WALL_H`` so the pilot gate (run at ``abort_mult`` 1.0)
    halts on a real blow-up but not on ordinary dispersion.
    """
    n_regimes = BEHAVIOR_REGIMES[behavior]
    budgets = budgets_for(behavior)
    n_plain_map_keys = grid.n_variants * grid.n_u_rungs
    per_budget = n_plain_map_keys * grid.n_draws * grid.n_seeds
    plain_s = sum(per_budget * wall_for_budget(b) for b in budgets)
    n_plain_specs = grid.n_variants * grid.n_u_rungs * n_regimes
    n_transfer = n_plain_specs * len(budgets) * grid.n_draws * grid.n_seeds if grid.transfer else 0
    transfer_s = n_transfer * TRANSFER_UNIT_S
    map_s = 0.0 if maps_staged else n_plain_map_keys * MAP_FIT_S
    projected_h = (map_s + plain_s + transfer_s) / 3600.0
    return LaneProjection(
        behavior=behavior,
        kind=kind,
        n_regimes=n_regimes,
        budgets=budgets,
        n_plain_map_keys=n_plain_map_keys,
        n_plain_groups_per_budget=per_budget,
        plain_s=plain_s,
        n_transfer_units=n_transfer,
        transfer_s=transfer_s,
        projected_h=projected_h,
        plan_wall_h=round(projected_h * fence_mult, 2),
        terms={
            "map_s": map_s,
            "plain_s": plain_s,
            "transfer_s": transfer_s,
            "plain_arith": f"{per_budget} groups/budget x "
            + " + ".join(f"{wall_for_budget(b):.2f}s(L={b})" for b in budgets),
            "transfer_arith": f"{n_transfer} units x {TRANSFER_UNIT_S}s",
        },
    )


def compose_combos() -> tuple[tuple[float, float], ...]:
    """The f_U x f_L combos `compose_run_specs` actually enumerates.

    Mirrors its `seen`-dedup: f_u == 0 is composition-degenerate, so every
    f_l collapses onto (0.0, 0.0). Read from the shipped constants so the
    count tracks the code rather than a hand-copied 3.
    """
    from explore_persona_space.experiments.issue_1739.constants import (
        COMPOSITION_F_L,
        COMPOSITION_F_U,
    )

    seen: list[tuple[float, float]] = []
    for f_u in COMPOSITION_F_U:
        for f_l in COMPOSITION_F_L:
            key = (float(f_u), float(f_l) if f_u > 0 else 0.0)
            if key not in seen:
                seen.append(key)
    return tuple(seen)


def compose_wall_for(behavior: str, budget: int) -> float:
    """Measured LINEAR group wall for a budget, following the proxy chain."""
    src = COMPOSE_WALL_PROXY.get(behavior, behavior)
    walls = COMPOSE_UNIT_GROUP_WALL_S.get(src)
    if not walls:
        raise KeyError(f"no measured linear group walls for behavior {behavior!r} (src {src!r})")
    return walls.get(int(budget), max(walls.values()))


def project_compose(
    behavior: str,
    *,
    anchors: tuple[int, ...] | None = None,
    n_variants: int = 2,
    plain_u_rungs: int = 1,
    fence_mult: float = 1.5,
) -> dict:
    """Compose-cell projection for the scope addendum (LINEAR map kind).

    Mirrors `_run_pilot`'s counters exactly (`issue1739_fits.py`):

        n_map_fits      = n_plain_map_keys + n_variants * n_combos * n_anchors
        n_plain_groups  = {b: n_plain_map_keys}            (1 draw, 1 seed)
        n_compose_units = {b: n_variants * n_combos}
        transfer        = 0                                (no --transfer)

    `n_map_fits` counts a distinct key per (variant, f_u, f_l, anchor) because
    `_map_key` includes `budgets` — including the f_u == 0 combos, whose map is
    anchor-independent in principle. The gate counts them, so the PLANNED
    projection counts them: `planned_h` is the fence basis. `realized_h`
    additionally subtracts the max-anchor skip (an f_u > 0, f_l == 0 combo has
    an EMPTY residual eliciting pool when L covers the whole train set — the
    skip evil recorded as `fu0.5_fl0.0` absent at L=8000) and is informational.
    """
    behavior_top = budgets_for(behavior)[-1]
    anchors = tuple(int(b) for b in (anchors if anchors is not None else budgets_for(behavior)))
    combos = compose_combos()
    n_plain_map_keys = n_variants * plain_u_rungs
    n_compose_per_anchor = n_variants * len(combos)

    map_s = (n_plain_map_keys + n_compose_per_anchor * len(anchors)) * COMPOSE_MAP_FIT_S[behavior]
    wall_sum = sum(compose_wall_for(behavior, b) for b in anchors)
    plain_s = n_plain_map_keys * wall_sum
    compose_s = n_compose_per_anchor * wall_sum
    planned_h = (map_s + plain_s + compose_s) / 3600.0

    # Realized: drop the residual-pool-empty combos at a max-train-set anchor.
    skipped = [c for c in combos if c[0] > 0 and c[1] == 0.0] if behavior_top in anchors else []
    n_skipped_cells = n_variants * len(skipped)
    realized_map_s = map_s - n_skipped_cells * COMPOSE_MAP_FIT_S[behavior]
    realized_compose_s = compose_s - n_skipped_cells * compose_wall_for(behavior, behavior_top)
    realized_h = (realized_map_s + plain_s + realized_compose_s) / 3600.0

    return {
        "behavior": behavior,
        "map_kind": "linear",
        "anchors": list(anchors),
        "combos": [list(c) for c in combos],
        "n_variants": n_variants,
        "n_map_fits": n_plain_map_keys + n_compose_per_anchor * len(anchors),
        "n_plain_groups_per_budget": n_plain_map_keys,
        "n_compose_units_per_anchor": n_compose_per_anchor,
        "planned_h": planned_h,
        "realized_h": realized_h,
        "plan_wall_h": round(planned_h * fence_mult, 2),
        "skipped_combos_at_top": [list(c) for c in skipped],
        "n_skipped_cells": n_skipped_cells,
        "walls_proxied_from": COMPOSE_WALL_PROXY.get(behavior),
        "terms": {
            "map_s": map_s,
            "plain_s": plain_s,
            "compose_s": compose_s,
            "arith": f"({n_plain_map_keys} + {n_compose_per_anchor}x{len(anchors)}) map fits x "
            f"{COMPOSE_MAP_FIT_S[behavior]:.2f}s + "
            f"({n_plain_map_keys} + {n_compose_per_anchor}) groups x {wall_sum:.2f}s",
        },
    }


def project_phase_a(
    kinds: tuple[str, ...], *, grid: Grid = DEFAULT_GRID, n_gpus: int | None = None
) -> dict:
    """Phase A: fit every map key ONCE per kind, kinds in parallel across GPUs."""
    n_keys = grid.n_variants * grid.n_u_rungs
    per_kind_h = n_keys * MAP_FIT_S / 3600.0
    # Arm overhead the prefetch cannot skip: one cheapest-budget unit group per
    # map key (1 draw, 1 seed, 1 regime, 1 arm).
    per_kind_h += n_keys * wall_for_budget(250) / 3600.0
    width = len(kinds) if n_gpus is None else max(1, min(n_gpus, len(kinds)))
    n_waves = -(-len(kinds) // width)
    return {
        "n_map_keys_per_kind": n_keys,
        "kinds": list(kinds),
        "per_kind_h": per_kind_h,
        "gpu_h": per_kind_h * len(kinds),
        "parallel_width": width,
        "wall_h": per_kind_h * n_waves,
        "arith": f"{n_keys} keys x {MAP_FIT_S}s x {len(kinds)} kinds, "
        f"{n_waves} wave(s) at width {width}",
    }


def project_round(
    behaviors: tuple[str, ...],
    kinds: tuple[str, ...],
    *,
    grid: Grid = DEFAULT_GRID,
    phase_a_gpus: int | None = None,
    lane_concurrency: int | None = None,
    fence_mult: float = 1.5,
) -> dict:
    """Whole-round projection: phase A + the behavior x kind lane fan-out."""
    lanes = [
        project_lane(b, k, grid=grid, maps_staged=True, fence_mult=fence_mult)
        for b in behaviors
        for k in kinds
    ]
    phase_a = project_phase_a(kinds, grid=grid, n_gpus=phase_a_gpus)
    lane_gpu_h = sum(x.projected_h for x in lanes)
    width = len(lanes) if lane_concurrency is None else max(1, min(lane_concurrency, len(lanes)))
    # Wall with `width` lanes at a time, longest-first (a fan-out bin-packing
    # lower bound: max(longest lane, total/width)).
    ordered = sorted((x.projected_h for x in lanes), reverse=True)
    lane_wall_h = max(max(ordered), lane_gpu_h / width) if ordered else 0.0
    return {
        "issue": 1739,
        "round": "nonlinear_map_fanout",
        "basis": {
            "source": "MEASURED pilot_report.json figures (per-call, production shape)",
            "map_fit_s": MAP_FIT_S,
            "unit_group_wall_s": UNIT_GROUP_WALL_S,
            "transfer_unit_s": TRANSFER_UNIT_S,
            "top_rung_fallback_s": TOP_RUNG_FALLBACK_S,
            "extrapolated": "L=16000 priced at the measured L=8000 wall "
            "(compose_pilot_report's own max-wall fallback)",
        },
        "grid": {
            "n_variants": grid.n_variants,
            "n_u_rungs": grid.n_u_rungs,
            "n_draws": grid.n_draws,
            "n_seeds": grid.n_seeds,
            "transfer": grid.transfer,
        },
        "phase_a": phase_a,
        "lanes": [
            {
                "behavior": x.behavior,
                "kind": x.kind,
                "n_regimes": x.n_regimes,
                "budgets": list(x.budgets),
                "projected_h": round(x.projected_h, 3),
                "plan_wall_h": x.plan_wall_h,
                "n_transfer_units": x.n_transfer_units,
                "terms": x.terms,
            }
            for x in lanes
        ],
        "totals": {
            "phase_a_gpu_h": round(phase_a["gpu_h"], 2),
            "phase_a_wall_h": round(phase_a["wall_h"], 2),
            "lanes_gpu_h": round(lane_gpu_h, 2),
            "lane_concurrency": width,
            "lanes_wall_h": round(lane_wall_h, 2),
            "round_gpu_h": round(phase_a["gpu_h"] + lane_gpu_h, 2),
            "round_wall_h": round(phase_a["wall_h"] + lane_wall_h, 2),
        },
        "fence_mult": fence_mult,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behaviors", nargs="+", default=["evil", "sycophancy", "hallucination"])
    ap.add_argument("--kinds", nargs="+", default=["mlp", "kernel"])
    ap.add_argument("--draws", type=int, default=3, help="draws per cell (path-2: 3)")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per cell (path-2: 2)")
    ap.add_argument("--u-rungs", type=int, default=2, help="U rungs (path-2: 250 + full)")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--phase-a-gpus", type=int, default=None)
    ap.add_argument("--lane-concurrency", type=int, default=None)
    ap.add_argument("--fence-mult", type=float, default=1.5)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument(
        "--plan-wall-for",
        default=None,
        metavar="BEHAVIOR",
        help="print ONLY that behavior's plan_wall_h (for EPM_I1739_NL_PLAN_WALL_H)",
    )
    ap.add_argument(
        "--compose",
        action="store_true",
        help="project the LINEAR compose cells (scope addendum) instead of the nl lanes",
    )
    ap.add_argument(
        "--compose-behaviors",
        nargs="+",
        default=["hallucination", "sycophancy"],
        help="behaviors carrying compose cells (addendum: hallucination + sycophancy)",
    )
    ap.add_argument(
        "--compose-anchors",
        nargs="+",
        type=int,
        default=None,
        help="L anchors for the compose cells (default: the behavior's own budget ladder)",
    )
    ap.add_argument(
        "--compose-plan-wall-for",
        default=None,
        metavar="BEHAVIOR",
        help="print ONLY that behavior's compose plan_wall_h (EPM_I1739_NL_COMPOSE_PLAN_WALL_H)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    grid = Grid(
        n_variants=args.variants,
        n_u_rungs=args.u_rungs,
        n_draws=args.draws,
        n_seeds=args.seeds,
    )
    anchors = tuple(args.compose_anchors) if args.compose_anchors else None
    if args.compose_plan_wall_for:
        rep = project_compose(
            args.compose_plan_wall_for, anchors=anchors, fence_mult=args.fence_mult
        )
        print(rep["plan_wall_h"])
        return 0
    if args.compose:
        reps = [
            project_compose(b, anchors=anchors, fence_mult=args.fence_mult)
            for b in args.compose_behaviors
        ]
        if args.json:
            print(json.dumps({"compose": reps}, indent=2))
            return 0
        print("== issue-1739 compose-cell projection (LINEAR basis, scope addendum) ==")
        combos = reps[0]["combos"] if reps else []
        print(f"  f_U x f_L combos (dedup'd): {combos}")
        for r in reps:
            proxy = (
                f", walls proxied from {r['walls_proxied_from']}" if r["walls_proxied_from"] else ""
            )
            print(
                f"    {r['behavior']:<14} L={r['anchors']}  {r['n_map_fits']} map fits, "
                f"{r['n_compose_units_per_anchor']} compose cells/anchor{proxy}"
            )
            print(f"      {r['terms']['arith']}")
            print(
                f"      planned {r['planned_h']:.2f} h (fence basis)  "
                f"realized {r['realized_h']:.2f} h "
                f"(−{r['n_skipped_cells']} max-anchor skips: {r['skipped_combos_at_top']})  "
                f"plan_wall_h {r['plan_wall_h']}"
            )
        print(
            f"  COMPOSE TOTAL: planned {sum(r['planned_h'] for r in reps):.2f} GPU-h, "
            f"realized {sum(r['realized_h'] for r in reps):.2f} GPU-h"
        )
        return 0
    if args.plan_wall_for:
        lane = project_lane(
            args.plan_wall_for, "mlp", grid=grid, maps_staged=True, fence_mult=args.fence_mult
        )
        print(lane.plan_wall_h)
        return 0
    rep = project_round(
        tuple(args.behaviors),
        tuple(args.kinds),
        grid=grid,
        phase_a_gpus=args.phase_a_gpus,
        lane_concurrency=args.lane_concurrency,
        fence_mult=args.fence_mult,
    )
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    t = rep["totals"]
    print("== issue-1739 nonlinear-map fan-out projection (MEASURED basis) ==")
    print(
        f"  basis: map_fit {MAP_FIT_S}s/key, group walls {UNIT_GROUP_WALL_S}, "
        f"transfer {TRANSFER_UNIT_S}s/unit"
    )
    print(f"  phase A: {rep['phase_a']['arith']}")
    print(
        f"           {t['phase_a_gpu_h']} GPU-h, wall {t['phase_a_wall_h']} h "
        f"(width {rep['phase_a']['parallel_width']})"
    )
    print("  lanes (maps STAGED, so map_fit = 0):")
    for lane in rep["lanes"]:
        print(
            f"    {lane['behavior']:<14} {lane['kind']:<7} R={lane['n_regimes']} "
            f"L={lane['budgets']}  projected {lane['projected_h']:.2f} h  "
            f"plan_wall_h {lane['plan_wall_h']}"
        )
    print(
        f"  lanes total: {t['lanes_gpu_h']} GPU-h, wall {t['lanes_wall_h']} h "
        f"(concurrency {t['lane_concurrency']})"
    )
    print(f"  ROUND: {t['round_gpu_h']} GPU-h, wall {t['round_wall_h']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
