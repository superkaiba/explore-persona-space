"""#1481 analyzer follow-on figure: marker install-normalized transfer fractions.

Plots, per (context, seed), each regime's pooled non-source transfer fraction at
its SELECTED rung — mean over non-source contexts of delta_logp divided by the
source install delta_logp (and the same in EOS-margin space, the plan's
registered install-strength Read 2 surface). Data: the committed
``regime_contrast_marker.json`` dose_curves (selected rung per arm).

Usage:
    uv run python scripts/issue1481_transfer_fraction_fig.py \
        --analysis-dir eval_results/issue_1481/analysis --fig-dir figures/issue_1481
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CTX_ORDER = ["bare", "conv", "icl", "pers"]
CTX_LABEL = {
    "bare": "Bare",
    "conv": "WildChat prefix",
    "icl": "ICL prefix",
    "pers": "Persona",
}


def _selected_fractions(dose_curves: dict) -> dict:
    out: dict = {}
    for arm_id, run in dose_curves.items():
        if run["lr_key"] != "lr5e6":
            continue  # verdict arms are all lr5e6
        sel = [r for r in run["rungs"] if "selected" in r.get("roles", [])]
        if not sel:
            continue
        rung = sel[0]
        src_logp = rung["source_install_logp"]
        src_margin = rung["source_install_margin"]
        ns = [c for c in rung["per_context"].values() if not c["is_source"]]
        out[arm_id] = {
            "ctx": run["ctx_key"],
            "regime": run["regime"],
            "seed": run["seed"],
            "f_logp": statistics.mean(c["delta_logp_mean"] for c in ns) / src_logp,
            "f_margin": statistics.mean(c["delta_margin_mean"] for c in ns) / src_margin,
            "src_logp": src_logp,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--fig-dir", required=True)
    args = p.parse_args(argv)
    marker = json.loads((Path(args.analysis_dir) / "regime_contrast_marker.json").read_text())
    fr = _selected_fractions(marker["dose_curves"])
    set_paper_style("blog")
    pal = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    # EOS-margin space FIRST — the plan §6 registered install-strength Read-2 surface
    # (marker-leakage-measurement.md: transfer fractions in EOS-margin logit space,
    # never raw log P); the log-prob companion is second.
    for ax, space, key in (
        (axes[0], "EOS-margin space (registered Read 2)", "f_margin"),
        (axes[1], "log-prob space", "f_logp"),
    ):
        for xi, ctx in enumerate(CTX_ORDER):
            for regime, color, dx, ha in (
                ("con", pal[0], -0.12, "right"),
                ("po", pal[1], +0.12, "left"),
            ):
                for arm in fr.values():
                    if arm["ctx"] != ctx or arm["regime"] != regime:
                        continue
                    ax.scatter([xi + dx], [arm[key]], color=color, s=30, zorder=3)
                    ax.text(
                        xi + dx + (0.05 if ha == "left" else -0.05),
                        arm[key],
                        f"s{arm['seed']}",
                        fontsize=5.5,
                        va="center",
                        ha=ha,
                        color="0.35",
                    )
        ax.set_xticks(range(len(CTX_ORDER)))
        ax.set_xticklabels([CTX_LABEL[c] for c in CTX_ORDER], fontsize=8)
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_title(space, fontsize=9, loc="center")
    axes[0].set_ylabel("non-source / source install fraction")
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=pal[0], label="contrastive"),
            Line2D([], [], marker="o", ls="none", color=pal[1], label="positive-only"),
        ],
        fontsize=7,
        loc="center",
        frameon=False,
    )
    fig.tight_layout()
    savefig_paper(fig, "marker_transfer_fractions", dir=Path(args.fig_dir))
    plt.close(fig)
    # numeric companion
    (Path(args.fig_dir) / "marker_transfer_fractions.json").write_text(
        json.dumps(fr, indent=1, sort_keys=True)
    )
    print(f"[i1481-transfer-fig] rendered marker_transfer_fractions ({len(fr)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
