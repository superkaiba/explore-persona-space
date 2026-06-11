# ruff: noqa: RUF001, RUF002
"""Issue #491 off-pod figures (plan v3 §6).

Hero: per-context ΔG_FT vs ΔG_ICL scatter at matched strength (K=8, all
chains), identity line, the matched source cell visually distinguished and
EXCLUDED from the displayed ρ/slope (registered 9-context headline).
Also: ICL dose curves vs K per chain, and the control-decomposition bars.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    SOURCE_CONTEXT,
    ns_eval_dir,
    repro_metadata,
)

logger = logging.getLogger("i491.figures")

FIG_DIR = Path("figures/issue_491")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    (FIG_DIR / f"{name}.meta.json").write_text(json.dumps(repro_metadata(), indent=2))
    logger.info("saved %s/%s.{png,pdf,meta.json}", FIG_DIR, name)


def hero_scatter(smoke: bool = False) -> None:
    """ΔG_FT vs ΔG_ICL per context, K=8 chains A/B/C, at matched strength."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style
    from explore_persona_space.experiments.icl_vs_ft_491.analyze import assemble_pairs

    set_paper_style()
    pairs = assemble_pairs(smoke)
    chains = [c for c in ("A", "B", "C") if f"ft_K8_chain{c}" in pairs]
    if not chains:
        raise FileNotFoundError("no K=8 matched pairs available for the hero figure")
    colors = paper_palette(max(len(chains), 3))
    fig, ax = plt.subplots(figsize=(6, 6))
    all_x, all_y = [], []
    for color, chain in zip(colors[: len(chains)], chains, strict=True):
        pair = pairs[f"ft_K8_chain{chain}"]
        icl_means = pair["icl"].mean(axis=1)
        ft_means = pair["ft"].mean(axis=1)
        for i, ctx in enumerate(pair["contexts"]):
            is_src = ctx == SOURCE_CONTEXT
            ax.scatter(
                icl_means[i],
                ft_means[i],
                color=color,
                marker="*" if is_src else "o",
                s=160 if is_src else 45,
                edgecolor="black" if is_src else "none",
                zorder=3,
                label=f"chain {chain}" if i == 0 else None,
            )
            if not is_src:
                all_x.append(icl_means[i])
                all_y.append(ft_means[i])
    lo = min(min(all_x), min(all_y)) - 0.5
    hi = max(max(all_x), max(all_y)) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, zorder=1)
    from scipy.stats import spearmanr

    rho = spearmanr(all_x, all_y).statistic
    slope = float(np.polyfit(all_x, all_y, 1)[0])
    ax.set_xlabel("ICL per-context leakage ΔG (nats, with-demos − no-demos)")
    ax.set_ylabel("FT per-context leakage ΔG (nats, matched ckpt − base)")
    ax.set_title(
        f"ICL vs FT leakage profiles at matched strength (K=8)\n"
        f"9 non-source contexts: ρ={rho:.2f}, slope={slope:.2f} "
        f"(source ★ excluded from displayed stats)"
    )
    ax.legend()
    _save(fig, "hero_matched_scatter_k8")
    plt.close(fig)


def dose_curves(smoke: bool = False) -> None:
    """ICL source/default dose vs K per chain (exploratory)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    ed = ns_eval_dir(smoke)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = paper_palette(3)
    plotted = False
    for color, chain in zip(colors, ("A", "B", "C"), strict=True):
        ks, src, dflt = [], [], []
        for k in (1, 3, 8, 16):
            p = ed / "icl_panel" / f"icl_K{k}_chain{chain}.json"
            if not p.exists():
                continue
            ctxs = json.loads(p.read_text())["contexts"]
            ks.append(k)
            src.append(ctxs[SOURCE_CONTEXT]["mean_delta_logp"])
            dflt.append(ctxs.get("helpful", {}).get("mean_delta_logp"))
        if ks:
            plotted = True
            ax.plot(ks, src, "-o", color=color, label=f"source, chain {chain}")
            if all(d is not None for d in dflt):
                ax.plot(ks, dflt, "--s", color=color, alpha=0.6, label=f"default, chain {chain}")
    if not plotted:
        raise FileNotFoundError("no icl_panel dose files found for dose curves")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 3, 8, 16], labels=["1", "3", "8", "16"])
    ax.set_xlabel("K (in-context demonstrations)")
    ax.set_ylabel("ΔG (nats)")
    ax.set_title("ICL dose curves: source vs default context")
    ax.legend(fontsize=7)
    _save(fig, "icl_dose_curves")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--which", default="all", choices=["all", "hero", "dose"])
    args = ap.parse_args(argv)
    if args.which in ("all", "hero"):
        hero_scatter(smoke=args.smoke)
    if args.which in ("all", "dose"):
        dose_curves(smoke=args.smoke)


if __name__ == "__main__":
    main()
