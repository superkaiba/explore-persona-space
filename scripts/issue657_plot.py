#!/usr/bin/env python3
"""Issue #657 Phase 4 — hero figures (CPU, off-pod).

Reads the per-behavior bake-off JSONs (``<behavior>_alignment_predictor.json`` +
``summary.json``) from Phase 3 and renders:

  - Hero (per behavior): scatter of held-out alignment (x) vs leakage Delta (y,
    source-FE residualized), colored by the bystander base prior — the headline
    figure. One PNG per behavior in scope for the leakage bake-off.
  - Conclusion: a forest plot of the doubly-partialled rho(align, Delta | [prior,
    prior_centroid_projection]) per behavior with bootstrap CIs, raw rho overlaid
    in a lighter shade (raw-alongside-residualized, standing rule).

No annotation overlays (no-plot-annotations rule). Each figure pairs a small
meta.json with the commit + params. Uses the paper rcParams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.issue657_alignment_predictor import (  # noqa: E402
    _source_fe_residualize,
)
from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    reproducibility_metadata,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)


def _save(fig, fig_dir: Path, name: str, extra: dict) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(fig_dir / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    meta = reproducibility_metadata(extra)
    (fig_dir / (name.rsplit(".", 1)[0] + "_meta.json")).write_text(json.dumps(meta, indent=2))


def _hero_scatter(behavior: str, res: dict, fig_dir: Path) -> bool:
    """Per-behavior hero scatter: held-out alignment vs source-FE leakage Delta.

    Returns True if a figure was written (behavior in leakage scope with cells).
    """
    import matplotlib.pyplot as plt

    lk = res.get("leakage_bake_off", {})
    if not lk.get("in_scope", False):
        return False
    # Per-cell arrays embedded in the bake-off JSON (joined_cells) — the plot is
    # self-contained, no separate join-dir dependency.
    cells = res.get("joined_cells")
    if not cells:
        return False
    x = np.array([c["align"] for c in cells], dtype=float)
    y_fe = _source_fe_residualize(cells, "delta")
    prior = np.array([c["bystander_base_rate"] for c in cells], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(x, y_fe, c=prior, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="bystander base prior")
    ax.set_xlabel("cosine(persona vector, behavior direction)")
    ax.set_ylabel(r"leakage $\Delta$ (source-FE residualized)")
    dp = lk.get("doubly_partialled_rho")
    rho_ci = lk.get("h3_doubly_partialled_rho_ci", {})
    title = (
        f"{behavior}: alignment vs leakage (n_bystanders={res['n_resolvable_bystanders']})\n"
        rf"doubly-partialled $\rho$={dp:.3f} "
        f"[{rho_ci.get('ci_lo', float('nan')):.3f}, {rho_ci.get('ci_hi', float('nan')):.3f}]"
        if dp is not None and not np.isnan(dp)
        else f"{behavior}: alignment vs leakage (n_bystanders={res['n_resolvable_bystanders']})"
    )
    ax.set_title(title)
    _save(fig, fig_dir, f"scatter_{behavior}.png", {"behavior": behavior})
    return True


def _forest_plot(per_behavior: dict[str, dict], fig_dir: Path) -> bool:
    """Forest plot of doubly-partialled rho per behavior + raw rho overlaid."""
    import matplotlib.pyplot as plt

    rows = []
    for beh, res in per_behavior.items():
        lk = res.get("leakage_bake_off", {})
        if not lk.get("in_scope", False):
            continue
        dp = lk.get("doubly_partialled_rho")
        raw = lk.get("raw_rho")
        ci = lk.get("h3_doubly_partialled_rho_ci", {})
        if dp is None or np.isnan(dp):
            continue
        rows.append(
            {
                "behavior": beh,
                "doubly": dp,
                "raw": raw,
                "ci_lo": ci.get("ci_lo", float("nan")),
                "ci_hi": ci.get("ci_hi", float("nan")),
                "is_primary": res.get("is_primary_h3", False),
            }
        )
    if not rows:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    ys = np.arange(len(rows))
    doublys = np.array([r["doubly"] for r in rows])
    los = np.array([r["ci_lo"] for r in rows])
    his = np.array([r["ci_hi"] for r in rows])
    xerr = np.array([doublys - los, his - doublys])
    xerr = np.clip(xerr, 0.0, None)  # constant-bootstrap epsilon-negative guard
    ax.errorbar(doublys, ys, xerr=xerr, fmt="o", capsize=3, label=r"doubly-partialled $\rho$")
    raws = [r["raw"] for r in rows]
    ax.scatter(raws, ys, marker="x", alpha=0.5, label=r"raw $\rho$")
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{r['behavior']}{' (primary)' if r['is_primary'] else ' (secondary)'}" for r in rows]
    )
    ax.set_xlabel(r"Spearman $\rho$(align, leakage $\Delta$)")
    ax.set_title("Held-out alignment-vs-leakage: doubly-partialled (CI) + raw")
    ax.legend(loc="best", fontsize=8)
    _save(fig, fig_dir, "forest_partial_rho.png", {"n_behaviors": len(rows)})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #657 Phase 4 — hero figures (CPU).")
    parser.add_argument(
        "--behavior",
        action="append",
        default=None,
        help="Behavior(s) to plot. Default: all present per-behavior JSONs.",
    )
    parser.add_argument(
        "--results-dir",
        default="eval_results/issue_657",
        help="Dir with <behavior>_alignment_predictor.json + summary.json.",
    )
    parser.add_argument("--fig-dir", default="figures/issue_657")
    parser.add_argument(
        "--smoke", action="store_true", help="Smoke: render only the given behavior."
    )
    args = parser.parse_args()

    load_dotenv()

    import matplotlib

    matplotlib.use("Agg")
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        pass

    results_dir = _resolve(args.results_dir)
    fig_dir = _resolve(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    behaviors = args.behavior or [
        p.stem.replace("_alignment_predictor", "")
        for p in sorted(results_dir.glob("*_alignment_predictor.json"))
    ]
    if not behaviors:
        raise FileNotFoundError(f"No *_alignment_predictor.json under {results_dir}")

    per_behavior: dict[str, dict] = {}
    for beh in behaviors:
        rp = results_dir / f"{beh}_alignment_predictor.json"
        if not rp.exists():
            raise FileNotFoundError(f"Missing bake-off JSON for {beh}: {rp}")
        res = json.loads(rp.read_text())
        per_behavior[beh] = res

    n_scatter = 0
    for beh, res in per_behavior.items():
        if _hero_scatter(beh, res, fig_dir):
            n_scatter += 1

    forest = _forest_plot(per_behavior, fig_dir)
    print(
        f"[phase=plot] wrote {n_scatter} scatter(s) + forest={forest} -> {fig_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
