"""Issue #810 ultrachat-genre-summary-sweep: promotion-pass supplementary figures.

Computes the registered two-method conjunction statistic per (combo x behavior)
against its selection-symmetric null (max over {summary x layer} inherited per
null draw, min over the two read-out methods, max over summaries), writes
``eval_results/issue_810/ultrachat-genre-summary-sweep/analysis/conjunction_bands.json``,
and renders three supplementary figures for the clean-result re-fold:

1. ``g1_readout_conjunction_bands`` — the 8 registered H2-g reads (obs vs band).
2. ``g1_e0_stability_scatter`` — per-context E0_betley vs E0_g1, 3 behaviors.
3. ``g1_cross_genre_recon`` — per-layer skill, mean/max-pool x Betley/UltraChat,
   plus the per-layer max-pool minus mean delta per genre.

Read-only over committed eval JSONs (0 GPU-h). Run from the issue-810 worktree:

    uv run python scripts/issue810_g1_square_figures.py
"""

from __future__ import annotations

import json

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "eval_results/issue_810/ultrachat-genre-summary-sweep"
FIG_DIR = ROOT / "figures/issue_810/ultrachat-genre-summary-sweep"
METHODS = ("fixed_rb", "trained_ridge")

# combo -> behaviors eligible for a headline read (plan v6 section 5: the parent
# harmful_compliance E0 target is quarantined, so it never reads at e0=betley).
COMBOS: dict[str, list[str]] = {
    "readout_sg1_ebetley": ["sycophancy", "refusal"],
    "readout_sg1_eg1": ["sycophancy", "refusal", "harmful_compliance"],
    "readout_sbetley_eg1": ["sycophancy", "refusal", "harmful_compliance"],
}
COMBO_LABEL = {
    "readout_sg1_ebetley": "UltraChat activations → misalignment-pool target",
    "readout_sg1_eg1": "UltraChat activations → UltraChat target",
    "readout_sbetley_eg1": "misalignment-pool activations → UltraChat target",
}
BEH_LABEL = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful compliance",
}


def _plain_ctx(c: str) -> str:
    """Reader-facing label for an internal context id (f5_fmt_markdown_table -> 'format: markdown table')."""
    body = c.split("_", 1)[1] if "_" in c else c
    if body.startswith("house_"):
        return "persona: " + body[len("house_") :].replace("_", " ")
    if body.startswith("phub_"):
        return "PersonaHub " + body[len("phub_") :].lstrip("0")
    if body.startswith("wc_"):
        return "WildChat " + body[len("wc_") :].replace("_", " ")
    if body.startswith("icl_"):
        rest = body[len("icl_") :]
        if "_k" in rest:
            style, k = rest.rsplit("_k", 1)
            return f"demos: {style} x{k}"
        return "demos: " + rest.replace("_", " ")
    if body.startswith("reph_"):
        return "rephrase: " + body[len("reph_") :].replace("_", " ")
    if body.startswith("fmt_"):
        return "format: " + body[len("fmt_") :].replace("_", " ")
    if body.startswith("behav_"):
        return "behavior: " + body[len("behav_") :].replace("_", " ")
    if body == "default_template":
        return "default assistant"
    if body == "helpful_asst":
        return "helpful assistant"
    return body.replace("_", " ")


def best_per_cell(cells: list[dict]) -> dict[tuple[str, str, str], float]:
    """Best (max over layers) signed rho per (behavior, summary, method)."""
    best: dict[tuple[str, str, str], float] = {}
    for c in cells:
        r = c.get("rho_graded")
        if r is None:
            continue
        key = (c["behavior"], c["summary"], c["method"])
        best[key] = max(best.get(key, -2.0), r)
    return best


def conjunction_observed(
    best: dict[tuple[str, str, str], float], behavior: str
) -> tuple[float, str, dict[str, float]]:
    """Max over summaries of min over methods of best-layer rho, plus per-summary values."""
    per_summary: dict[str, float] = {}
    for s in {s for (b, s, _m) in best if b == behavior}:
        per_summary[s] = min(best.get((behavior, s, m), -2.0) for m in METHODS)
    arg = max(per_summary, key=per_summary.get)  # type: ignore[arg-type]
    return per_summary[arg], arg, per_summary


def conjunction_null_draws(null_path: Path, behavior: str) -> np.ndarray:
    """Per-draw conjunction statistic with the identical selection applied per draw."""
    d = json.loads(null_path.read_text())
    ro = d["readout"][behavior]
    per_method: dict[str, tuple[list[str], np.ndarray]] = {}
    for m in METHODS:
        names = list(ro[m].keys())
        mats = [np.array([ro[m][s][layer] for layer in ro[m][s]]).max(axis=0) for s in names]
        per_method[m] = (names, np.stack(mats))  # (S, draws) best-over-layers per draw
    names0, a = per_method[METHODS[0]]
    names1, b = per_method[METHODS[1]]
    idx = [names1.index(s) for s in names0]
    return np.minimum(a, b[idx]).max(axis=0)  # (draws,) max over summaries of min over methods


def main() -> None:
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ stats
    reads: list[dict] = []
    harmful_per_summary: dict[str, dict[str, float]] = {}
    for combo, behaviors in COMBOS.items():
        cells = json.loads((SWEEP / combo / "readout_rho_by_summary.json").read_text())["cells"]
        best = best_per_cell(cells)
        for beh in behaviors:
            obs, arg, per_summary = conjunction_observed(best, beh)
            draws = conjunction_null_draws(SWEEP / combo / "null_matrix_readout.json", beh)
            band = float(np.percentile(draws, 97.5))
            p = float((draws >= obs).mean())
            reads.append(
                {
                    "combo": combo,
                    "behavior": beh,
                    "observed_conjunction_rho": obs,
                    "argmax_summary": arg,
                    "null_band_97p5": band,
                    "empirical_p": p,
                    "n_perms": int(draws.size),
                    "clears_band": bool(obs > band),
                }
            )
            if beh == "harmful_compliance":
                harmful_per_summary[combo] = per_summary

    square = json.loads((SWEEP / "readout_rho_square.json").read_text())
    out = {
        "dv": "two-method conjunction read-out statistic per (combo x behavior)",
        "construction": (
            "max over summaries of min over the two methods of best-over-layers signed "
            "rho_graded; null inherits the identical selection per draw (1000 perms, seed 658)"
        ),
        "registered_reads": reads,
        "e0_stability": square["e0_stability"],
    }
    out_path = SWEEP / "analysis" / "conjunction_bands.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    for r in reads:
        print(
            f"  {r['combo']:22s} {r['behavior']:20s} obs={r['observed_conjunction_rho']:+.3f} "
            f"band={r['null_band_97p5']:+.3f} p={r['empirical_p']:.3f}"
        )

    # -------------------------------------------- figure 1: conjunction bands
    colors = paper_palette_blog(3)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ys = np.arange(len(reads))[::-1]
    for y, r in zip(ys, reads):
        color = colors[list(COMBOS).index(r["combo"])]
        ax.plot([r["null_band_97p5"]], [y], marker="|", ms=16, mew=2.4, color="0.35", zorder=2)
        ax.plot(
            [r["observed_conjunction_rho"]],
            [y],
            marker="o",
            ms=8,
            color=color,
            zorder=3,
        )
        ax.text(
            -0.62,
            y,
            f"{BEH_LABEL[r['behavior']]}\n{COMBO_LABEL[r['combo']]}",
            va="center",
            ha="left",
            fontsize=7.5,
        )
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(-0.65, 0.85)
    ax.set_xlabel("two-method read-out correlation (held-out Spearman ρ)")
    ax.set_title("8 registered reads: observed (dot) vs null band (tick)", fontsize=10)

    for combo in ["readout_sg1_eg1", "readout_sbetley_eg1"]:
        per_summary = harmful_per_summary[combo]
        vals = np.array(sorted(per_summary.values()))
        ecdf = np.linspace(0, 1, len(vals))
        color = colors[list(COMBOS).index(combo)]
        ax2.plot(
            vals,
            ecdf,
            drawstyle="steps-post",
            color=color,
            label=COMBO_LABEL[combo],
        )
        mean_val = per_summary["mean"]
        mean_y = float((vals <= mean_val).mean())
        ax2.plot(
            [mean_val],
            [mean_y],
            marker="o",
            ms=8,
            color=color,
            label="mean summary" if combo == "readout_sg1_eg1" else None,
        )
    band_hc = [r["null_band_97p5"] for r in reads if r["behavior"] == "harmful_compliance"]
    ax2.axvline(float(np.mean(band_hc)), color="0.35", ls="--", lw=1.2)
    ax2.set_xlabel("per-summary conjunction ρ (harmful compliance, fresh target)")
    ax2.set_ylabel("fraction of 37 summaries below")
    ax2.set_title("Harmful compliance: 13-16 of 37 summaries clear the band", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper left")
    savefig_paper(fig, "g1_readout_conjunction_bands", dir=FIG_DIR)
    plt.close(fig)

    # ------------------------------------------- figure 2: E0 stability scatter
    g1 = json.loads((SWEEP / "phase_c/e0_highm_graded.json").read_text())["by_behavior"]
    parent = json.loads((ROOT / "eval_results/issue_810/phase_c/e0_highm_graded.json").read_text())[
        "by_behavior"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, beh in zip(axes, ["sycophancy", "refusal", "harmful_compliance"]):
        pm = parent[beh]["per_context_graded_mean"]
        gm = g1[beh]["per_context_graded_mean"]
        ctxs = sorted(set(pm) & set(gm))
        x = np.array([pm[c] for c in ctxs])
        y = np.array([gm[c] for c in ctxs])
        rho = square["e0_stability"][beh]["rho"]
        ax.scatter(x, y, s=22, color=paper_palette_blog(1)[0])
        resid = np.abs((y - y.mean()) / (y.std() + 1e-9) - (x - x.mean()) / (x.std() + 1e-9))
        for i in np.argsort(resid)[-6:]:
            ax.text(x[i], y[i], _plain_ctx(ctxs[i]), fontsize=6)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.8", lw=0.8)
        ax.set_xlabel("graded score, misalignment-pool completions")
        ax.set_ylabel("graded score, UltraChat completions")
        title = f"{BEH_LABEL[beh]} (ρ = {rho:+.2f})"
        if beh == "harmful_compliance":
            title += "\nparent target contaminated"
        ax.set_title(title, fontsize=10)
    savefig_paper(fig, "g1_e0_stability_scatter", dir=FIG_DIR)
    plt.close(fig)

    # ------------------------------------------ figure 3: cross-genre recon
    delta = json.loads((SWEEP / "genre_delta_recon.json").read_text())
    per = delta["per_layer_observed_skill"]
    layers = np.array(delta["layers"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    c_mean, c_maxp = paper_palette_blog(2)
    for genre, ls in [("betley", "--"), ("g1", "-")]:
        gl = "misalignment pool" if genre == "betley" else "UltraChat"
        ax.plot(layers, per[f"{genre}/mean"], ls=ls, color=c_mean, label=f"mean, {gl}")
        ax.plot(layers, per[f"{genre}/maxp"], ls=ls, color=c_maxp, label=f"max-pool, {gl}")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_ylim(-0.3, 0.9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Mean transfers across genres; max-pool drops on UltraChat", fontsize=10)
    for genre, ls in [("betley", "--"), ("g1", "-")]:
        gl = "misalignment pool" if genre == "betley" else "UltraChat"
        d_ = np.array(per[f"{genre}/maxp"]) - np.array(per[f"{genre}/mean"])
        ax2.plot(layers, d_, ls=ls, color="0.2", label=gl)
    ax2.axhline(0.0, color="0.8", lw=0.8)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("max-pool minus mean, held-out skill")
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_title("The late-layer max-pool edge shrinks on UltraChat", fontsize=10)
    savefig_paper(fig, "g1_cross_genre_recon", dir=FIG_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
