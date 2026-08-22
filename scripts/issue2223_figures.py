"""Issue #2223 — figures for the Lu et al. Fig-4 drift reproduction + stabilization grid.

Reads the driver's result JSONs from ``--eval-dir`` and renders (each panel skips
gracefully if its input JSON is absent, so a smoke renders whatever exists):

- ``drift_hero``      — per-domain response-projection trajectory vs turn position
                        (mean + conversation-bootstrap 95% CI band), the Fig-4 form.
- ``arm_trajectories``— per-arm (Phase B) pooled response-projection trajectory
                        (A0 baseline vs the steering / cap arms).
- ``firing``          — expected (A0-measured) vs realized cap-firing per cap arm.
- ``ridge``           — message→next-projection ridge held-out R² vs shuffle null.

Figures stay SIMPLE (axes / ticks / legend / titles only) — NO on-canvas caption
or per-point-n text block (standing directive 2026-08-12). Alive-n per
(domain, arm, turn) is registered in the trajectory JSONs and rides
``savefig_paper``'s embedded per-point sidecar data; the surrounding prose /
caption carries it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ISSUE = 2223
LATE_WINDOW = range(8, 16)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load(eval_dir: Path, name: str):
    p = eval_dir / name
    if not p.exists():
        _log(f"[figures] SKIP {name} (absent)")
        return None
    return json.loads(p.read_text())


def fig_drift_hero(eval_dir: Path, out_dir: Path) -> None:
    """Per-domain response-projection trajectory (mean + 95% CI) vs turn position."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    data = _load(eval_dir, "phaseA_verdict.json")
    if data is None:
        return
    agg = data["aggregate"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    domains = sorted(agg)
    colors = paper_palette(max(len(domains), 1))
    for d, color in zip(domains, colors, strict=False):
        turns = sorted(int(t) for t in agg[d])
        xs = [t for t in turns]
        ys = [agg[d][str(t)]["mean"] for t in turns]
        lo = [agg[d][str(t)]["ci_lo"] for t in turns]
        hi = [agg[d][str(t)]["ci_hi"] for t in turns]
        ax.plot(xs, ys, marker="o", ms=3, color=color, label=d)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.axvspan(min(LATE_WINDOW) - 0.5, max(LATE_WINDOW) + 0.5, color="0.85", alpha=0.4, zorder=0)
    ax.set_xlabel("Turn position")
    ax.set_ylabel("Assistant-axis projection (response-token mean)")
    ax.set_title(f"Persona drift by domain (verdict: {data['verdict']['disposition']})")
    ax.legend(fontsize=7, loc="best")
    savefig_paper(fig, f"issue_{ISSUE}/drift_hero", dir=str(out_dir))
    plt.close(fig)
    _log("[figures] wrote drift_hero")


def _pool_arm_trajectory(cell_traj: dict) -> tuple[list[int], list[float]]:
    """Pool response projections across domains → (turns, mean-per-turn)."""
    import numpy as np

    by_turn: dict[int, list[float]] = {}
    for _domain, turns in cell_traj.items():
        for t, rows in turns.items():
            for r in rows:
                if r["response"] is not None:
                    by_turn.setdefault(int(t), []).append(r["response"])
    xs = sorted(by_turn)
    ys = [float(np.mean(by_turn[t])) for t in xs]
    return xs, ys


def fig_arm_trajectories(eval_dir: Path, out_dir: Path) -> None:
    """Per-arm pooled response-projection trajectory (Phase B arms + A0 baseline)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    arms_b = _load(eval_dir, "phaseB_arm_trajectories.json")
    arms_a = _load(eval_dir, "phaseA_drift_trajectory.json")
    cells: dict[str, dict] = {}
    if arms_a is not None:
        cells.update({k: v for k, v in arms_a.get("arms", {}).items() if k.startswith("A0__")})
    if arms_b is not None:
        cells.update(arms_b.get("arms", {}))
    if not cells:
        _log("[figures] SKIP arm_trajectories (no phaseA/phaseB trajectories)")
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = sorted(cells)
    colors = paper_palette(max(len(names), 1))
    for name, color in zip(names, colors, strict=False):
        xs, ys = _pool_arm_trajectory(cells[name]["trajectory"])
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, color=color, label=cells[name]["arm"])
    ax.set_xlabel("Turn position")
    ax.set_ylabel("Assistant-axis projection (pooled over domains)")
    ax.set_title("Per-arm drift trajectory (Phase B stabilization grid)")
    ax.legend(fontsize=7, loc="best")
    savefig_paper(fig, f"issue_{ISSUE}/arm_trajectories", dir=str(out_dir))
    plt.close(fig)
    _log("[figures] wrote arm_trajectories")


def fig_firing(eval_dir: Path, out_dir: Path) -> None:
    """Expected (A0-measured) vs realized cap-firing per cap arm."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    data = _load(eval_dir, "firing_telemetry.json")
    if data is None:
        return
    cells = data.get("cells", {})
    if not cells:
        _log("[figures] SKIP firing (no cells)")
        return
    names = sorted(cells)
    expected = [cells[n]["expected_fired_frac"] for n in names]
    realized = [cells[n]["realized_fired_frac"] for n in names]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, expected, w, label="expected (A0)")
    ax.bar(x + w / 2, realized, w, label="realized")
    for i, n in enumerate(names):
        if cells[n].get("calibration_limited"):
            ax.plot(x[i], max(expected[i], realized[i]) + 0.02, marker="v", color="crimson", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels([cells[n]["arm"] for n in names])
    ax.set_ylabel("Cap-fired fraction")
    ax.set_title("Cap firing: expected vs realized (▼ = calibration-limited)")
    ax.legend(fontsize=8)
    savefig_paper(fig, f"issue_{ISSUE}/firing", dir=str(out_dir))
    plt.close(fig)
    _log("[figures] wrote firing")


def fig_ridge(eval_dir: Path, out_dir: Path) -> None:
    """Message→next-projection ridge held-out R² (abs + delta) vs shuffle null band."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    data = _load(eval_dir, "ridge_message_projection.json")
    if data is None:
        return
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    labels = ["R² (abs)", "R² (Δ)"]
    vals = [data["r2_abs"], data["r2_delta"]]
    ax.bar(labels, vals, color=["#4c72b0", "#55a868"])
    null = data.get("shuffle_null_abs_ci")
    if null:
        ax.axhspan(null[0], null[1], color="0.8", alpha=0.6, label="shuffle-null 95% CI")
        ax.legend(fontsize=8)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel("Held-out R² (LOCO)")
    ax.set_title(f"Message→projection ridge (n={data.get('n_rows', '?')})")
    savefig_paper(fig, f"issue_{ISSUE}/ridge", dir=str(out_dir))
    plt.close(fig)
    _log("[figures] wrote ridge")


PANELS = {
    "drift_hero": fig_drift_hero,
    "arm_trajectories": fig_arm_trajectories,
    "firing": fig_firing,
    "ridge": fig_ridge,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir",
        default=str(REPO / "eval_results" / f"issue_{ISSUE}"),
        help="directory holding the driver's result JSONs",
    )
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "figures"),
        help="figures root (savefig_paper writes issue_<N>/<stem> beneath it)",
    )
    ap.add_argument("--panel", choices=sorted(PANELS), help="render one panel (default: all)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[import-check] OK")
        return 0
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    panels = [args.panel] if args.panel else list(PANELS)
    for name in panels:
        PANELS[name](eval_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
