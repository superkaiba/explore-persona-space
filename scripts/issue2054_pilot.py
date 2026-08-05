"""Shared pilot->fleet wall projection for the #2054 fit-family drivers.

M-R2-1 (code-review r2): the fits/ladder pilot gates persist a MEASURED
1-unit 1-fold wall, but nothing multiplied it by the fleet count — the
#823/#813 class (cheap once, ruinous x thousands: ~3780 serial ridge fits
realized 12-20 h against a 0.35 h plan). This helper extends a pilot-gate
report with the pilot-extrapolated fleet wall and enforces a budget:

- ``projected = wall_seconds (1-fold pilot) x fold_k x n_fleet_units`` — a
  conservative UPPER bound (the ladder's per-source M-fit memo and the
  fits reduced-basis sharing are not modeled).
- ``fence_floor_seconds = 2 x projected`` — the smallest timeout/fence any
  dispatcher may arm around the fleet phase
  (plan-compute-sizing.md § Per-cell fit phases, the p90-style x2
  dispersion default).
- With ``max_fleet_wall_hours`` armed (the ladder), an over-budget
  projection raises :class:`FleetWallExceeded` — a DESIGNED halt the driver
  routes to a distinct exit code (7, the phase_a PilotGateRefusal / #1415
  artifact-routed convention), never an anonymous crash. Unarmed (the
  fits driver), the projection WARNs at :data:`FLEET_WALL_WARN_HOURS`
  (routing/descope stays the dispatcher's call, mirroring the RSS WARN).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path

# WARN boundary when no fail-loud fence is armed. Grounding: plan §9 books
# the whole fit family as a pilot-gated VM-CPU wall ("0 GPU-h"); a >12 h VM
# wall is squarely the #823 realized-wall class the projection exists to
# catch, while sitting far above any sane restricted-set wall.
FLEET_WALL_WARN_HOURS = 12.0


class FleetWallExceeded(RuntimeError):
    """Projected fleet wall exceeds the armed budget (designed halt, exit 7)."""


def require_prior_wall_seconds(prior: dict, report_path: Path) -> float:
    """Fail-loud read of a prior pilot report's measured 1-unit 1-fold wall.

    The prior-report skip path feeds this value straight into the fleet-wall
    fence (``projected = wall x fold_k x n_units``), so a missing
    ``wall_seconds`` silently defaulting to 0.0 would project a fleet wall of
    0 and DISARM the fence guarding the production run (code-review r3
    Minor 1 on #2054). A non-positive / non-finite stored value is the same
    silent-disarm class (0 projects 0; NaN makes ``projected > budget``
    False) and raises too. Remedies: delete the stale report, or re-run with
    --overwrite — both re-measure the pilot.
    """
    wall = prior.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        raise RuntimeError(
            f"pilot gate: prior report {report_path} matches the measurement knobs but "
            f"carries no measured 'wall_seconds' (got {wall!r}) — refusing to project the "
            "fleet wall from a missing pilot measurement; delete the report or re-run "
            "with --overwrite to re-measure"
        )
    wall_f = float(wall)
    if not math.isfinite(wall_f) or wall_f <= 0.0:
        raise RuntimeError(
            f"pilot gate: prior report {report_path} carries a non-positive/non-finite "
            f"measured 'wall_seconds' ({wall!r}) — that fence input would project a "
            "0/NaN fleet wall and disarm the budget; delete the report or re-run with "
            "--overwrite to re-measure"
        )
    return wall_f


def fleet_projection_update(
    report_path: Path,
    payload: dict,
    *,
    wall_seconds: float,
    n_fleet_units: int,
    fold_k: int,
    log: Callable[[str], None],
    max_fleet_wall_hours: float | None = None,
    units_basis: str = "total",
) -> dict:
    """Extend ``payload`` with the pilot->fleet projection, atomically
    (re)write ``report_path``, log one line, and enforce the budget.

    Returns the extended payload. Raises :class:`FleetWallExceeded` when
    ``max_fleet_wall_hours`` is armed and the projection exceeds it (the
    report JSON is written BEFORE the raise — artifact-routed halt).
    """
    projected = float(wall_seconds) * int(fold_k) * int(n_fleet_units)
    out = dict(payload)
    out.update(
        {
            "n_fleet_units": int(n_fleet_units),
            "fleet_units_basis": units_basis,
            "fold_k": int(fold_k),
            "pilot_wall_seconds_per_unit_fold": round(float(wall_seconds), 3),
            "projected_fleet_wall_seconds": round(projected, 1),
            "projected_fleet_wall_hours": round(projected / 3600.0, 3),
            "fence_floor_seconds": round(2.0 * projected, 1),
            "max_fleet_wall_hours": max_fleet_wall_hours,
            "memo_not_modeled": True,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, report_path)
    log(
        f"fleet projection: {n_fleet_units} unit(s) [{units_basis}] x {fold_k} fold(s) "
        f"x {wall_seconds:.1f}s/unit-fold -> {projected / 3600.0:.2f} h projected "
        f"(dispatcher fence floor >= {2.0 * projected / 3600.0:.2f} h) -> {report_path}"
    )
    budget_h = FLEET_WALL_WARN_HOURS if max_fleet_wall_hours is None else max_fleet_wall_hours
    if projected > budget_h * 3600.0:
        msg = (
            f"projected fleet wall {projected / 3600.0:.2f} h exceeds "
            f"{budget_h} h ({n_fleet_units} units [{units_basis}] x {fold_k} folds x "
            f"{wall_seconds:.1f}s measured pilot; report: {report_path})"
        )
        if max_fleet_wall_hours is not None:
            raise FleetWallExceeded(
                msg + " — designed halt: restrict the pair set (--pair-classes), raise "
                "--max-fleet-wall-hours deliberately, or route the phase off-VM"
            )
        log(f"WARN {msg} — routing/descope is the dispatcher's call (#823 class)")
    return out
