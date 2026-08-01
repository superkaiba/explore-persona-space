"""#1768 checkpoint-dynamics figures — alignment / norm / install vs training step.

Three figures from `eval_results/issue_1768/ckpt_dynamics/curves.json`:

a) `alignment_vs_step_delta` — cos(ŵ_tf, δ) vs optimizer step, one panel per
   behavior, one line per arm, with the round-1 null band shaded.
b) `alignment_vs_step_target` — the same against the behavior direction r_B
   (content arms) and the marker unembedding row W_U (marker arms).
c) `norm_and_install_vs_step` — ‖ŵ_tf‖(step) beside the per-rung install read
   (marker arms only — #1481 records no per-step content rates).

Every axis label is plain English; arm lines are labelled by their design cell
(context / regime / seed), never by a raw arm slug.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

BEH_LABEL = {
    "cas": "Writing style",
    "imp": "Impoliteness",
    "syc": "Sycophancy",
    "mk": "Marker token",
}
CTX_LABEL = {"pers": "persona", "bare": "bare", "conv": "conversation", "icl": "in-context"}
REGIME_LABEL = {"con": "contrastive", "po": "positive-only"}
LAYER_DEFAULT = 19


def _arm_label(c: dict) -> str:
    return f"{CTX_LABEL.get(c['ctx_key'], c['ctx_key'])}, {REGIME_LABEL.get(c['regime'], c['regime'])}, seed {c['seed']}"


def _load(results_dir: Path) -> tuple[dict, dict]:
    curves = json.loads((results_dir / "curves.json").read_text())["curves"]
    summary = json.loads((results_dir / "summary.json").read_text())
    return curves, summary


def _panel_curves(curves: dict, layer: int) -> dict[str, list[dict]]:
    """{behavior: [curve, ...]} at one layer, curves with >=2 captured rungs."""
    out: dict[str, list[dict]] = {}
    for c in curves.values():
        if c["layer"] != layer or len(c.get("points") or []) < 2:
            continue
        out.setdefault(c["beh_key"], []).append(c)
    for v in out.values():
        v.sort(key=lambda c: (c["ctx_key"], c["regime"], c["seed"]))
    return out


def _null_hi(c: dict, cand: str) -> float | None:
    nb = (c.get("null_bands") or {}).get(cand) or {}
    fam = nb.get("primary_null_family")
    band = (nb.get("nulls") or {}).get(fam) if fam else None
    return (band or {}).get("p97_5")


def _draw_alignment(curves: dict, layer: int, cand_for, out_dir: Path, stem: str, ylabel: str):
    panels = _panel_curves(curves, layer)
    if not panels:
        return None
    behs = [b for b in ("cas", "imp", "syc", "mk") if b in panels]
    fig, axes = plt.subplots(1, len(behs), figsize=(4.2 * len(behs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, beh in zip(axes, behs, strict=True):
        arms = panels[beh]
        colors = paper_palette(max(3, len(arms)))
        hi_vals, n_lines = [], 0
        for i, c in enumerate(arms):
            cand = cand_for(c)
            xs = [p["step"] for p in c["points"]]
            ys = [p["cos"].get(cand) for p in c["points"]]
            keep = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None and np.isfinite(y)]
            if len(keep) < 2:
                continue
            n_lines += 1
            ax.plot(
                [k[0] for k in keep],
                [k[1] for k in keep],
                marker="o",
                ms=2.6,
                lw=1.3,
                color=colors[i % len(colors)],
                label=_arm_label(c),
                alpha=0.9,
            )
            sel = [p for p in c["points"] if p.get("is_selected")]
            if sel and sel[0]["cos"].get(cand) is not None:
                ax.plot(
                    sel[0]["step"],
                    sel[0]["cos"][cand],
                    marker="*",
                    ms=11,
                    color=colors[i % len(colors)],
                    zorder=5,
                    lw=0,
                )
            h = _null_hi(c, cand)
            if h is not None:
                hi_vals.append(h)
        if hi_vals:
            hi = float(np.median(hi_vals))
            ax.axhspan(-hi, hi, color="0.82", alpha=0.55, zorder=0, lw=0)
        ax.axhline(0.0, color="0.55", lw=0.8, ls=":", zorder=1)
        ax.set_title(f"{BEH_LABEL.get(beh, beh)}  ({n_lines} arms)")
        ax.set_xlabel("Training step (optimizer steps)")
        if len(arms) <= 6:
            ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel(ylabel, labelpad=2)
    fig.suptitle(
        f"{ylabel} across each arm's checkpoint ladder (layer {layer})\n"
        "star = the verdict checkpoint round 1 read; grey band = round-1 null (95%)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.06 if len(behs) > 2 else 0.12)
    paths = savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    return paths


def _draw_norm_and_install(curves: dict, layer: int, out_dir: Path):
    panels = _panel_curves(curves, layer)
    if not panels:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax_n, ax_i = axes
    behs = [b for b in ("cas", "imp", "syc", "mk") if b in panels]
    colors = paper_palette(max(3, len(behs)))
    for i, beh in enumerate(behs):
        for j, c in enumerate(panels[beh]):
            xs = [p["step"] for p in c["points"]]
            ys = [p["w_tf_norm"] for p in c["points"]]
            ax_n.plot(
                xs,
                ys,
                lw=1.1,
                color=colors[i % len(colors)],
                alpha=0.75,
                label=BEH_LABEL.get(beh, beh) if j == 0 else None,
            )
    ax_n.set_xlabel("Training step (optimizer steps)")
    ax_n.set_ylabel("Write magnitude  ‖ŵ_tf‖")
    ax_n.set_title("Write magnitude grows with training")
    ax_n.legend(fontsize=7)

    # Content and marker install metrics differ in UNITS (judged rate in [0,1]
    # vs Δ log P in nats), so they get separate y-axes — never one shared scale.
    ax_i2 = ax_i.twinx()
    n_content = n_marker = 0
    for i, beh in enumerate(behs):
        for c in panels[beh]:
            pts = [
                p
                for p in c["points"]
                if p.get("install") and p["install"].get("install") is not None
            ]
            if len(pts) < 2:
                continue
            target = ax_i2 if beh == "mk" else ax_i
            if beh == "mk":
                n_marker += 1
            else:
                n_content += 1
            target.plot(
                [p["step"] for p in pts],
                [p["install"]["install"] for p in pts],
                lw=1.0,
                marker="o",
                ms=2.2,
                alpha=0.75,
                color=colors[i % len(colors)],
                ls="--" if beh == "mk" else "-",
            )
    ax_i.set_xlabel("Training step (optimizer steps)")
    ax_i.set_ylabel("Content install\n(judged rate, Tier-1 selection pool)")
    ax_i2.set_ylabel("Marker install\n(Δ log P, trained − base, nats)")
    ax_i.set_title(
        f"Install strength vs step\n({n_content} content arms solid, {n_marker} marker dashed)"
    )
    if not (n_content or n_marker):
        ax_i.text(
            0.5,
            0.5,
            "No per-step install read available",
            ha="center",
            va="center",
            transform=ax_i.transAxes,
            fontsize=8,
        )
    fig.suptitle(
        f"Write magnitude and install strength across the ladder (layer {layer})", fontsize=9
    )
    fig.tight_layout()
    paths = savefig_paper(fig, "norm_and_install_vs_step", dir=out_dir)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="#1768 checkpoint-dynamics figures")
    ap.add_argument(
        "--results-dir", default=str(REPO_ROOT / "eval_results/issue_1768/ckpt_dynamics")
    )
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "figures/issue_1768/ckpt_dynamics"))
    ap.add_argument("--layer", type=int, default=LAYER_DEFAULT)
    args = ap.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves, _summary = _load(results_dir)
    set_paper_style("blog")

    made = {}
    p = _draw_alignment(
        curves,
        args.layer,
        lambda _c: "delta",
        out_dir,
        "alignment_vs_step_delta",
        "cos(ŵ_tf, δ)",
    )
    if p:
        made["alignment_vs_step_delta"] = {k: str(v) for k, v in p.items()}
    p = _draw_alignment(
        curves,
        args.layer,
        lambda c: "W_U_marker_row" if c["kind"] == "marker" else "r_B",
        out_dir,
        "alignment_vs_step_target",
        "cos(ŵ_tf, behavior target)",
    )
    if p:
        made["alignment_vs_step_target"] = {k: str(v) for k, v in p.items()}
    p = _draw_norm_and_install(curves, args.layer, out_dir)
    if p:
        made["norm_and_install_vs_step"] = {k: str(v) for k, v in p.items()}

    print(json.dumps({"layer": args.layer, "figures": made}, indent=1))
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
