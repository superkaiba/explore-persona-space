#!/usr/bin/env python
"""Summary + raw-datapoint comparison figures for task #778.

Task #778 replicated Persona Vectors' (arXiv 2507.21509) two prediction
experiments on Qwen2.5-7B (traits: evil, sycophancy, hallucination) and added a
norm-matched-random / permutation / cross-trait / PCA null battery the paper
never ran. This script reads the committed artifacts and produces:

  RAW-DATAPOINT GRIDS (primary deliverable) — the actual points each correlation
  is computed from, one figure per data setting (finetune / system-prompt
  monitoring / many-shot ICL), each a 2-row x 3-trait grid:
    * top row    = x is the projection onto the trait's persona vector r_B at
                   the selected layer (the paper's predictor),
    * bottom row = x is the projection onto ONE norm-matched random direction
                   (the null-battery sampler, deterministic seed 0) at the SAME
                   selected layer — the honest baseline companion. Each random
                   panel's title states the drawn direction's r AND its
                   percentile within the 1000-draw null (so the reader can see
                   whether the shown draw is typical).
    y = graded trait-expression score in both rows. Points colored by dataset
    family (finetune) / prompt condition (monitoring) / shot count (many-shot).

  SUMMARY META-SCATTERS (kept from v1) — ours-vs-random-cap and ours-vs-paper.

  COMPARISON BAR PANELS (kept from v1) — our |r| vs paper r vs null caps.

All inputs are committed artifacts under eval_results/issue_778/ (JSON/JSONL)
plus the acts tensors + null draws under data/issue_778/ (pulled from the HF
data repo issue778_persona_vectors/analysis_tensors/). No training, no eval
generation, no GPU. Idempotent:
    uv run python scripts/issue778_summary_comparison_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps must land before numpy/torch import on the shared VM (#847).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis import null_battery as nb  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval_results" / "issue_778"
DATA_DIR = REPO / "data" / "issue_778"
FIG_DIR = REPO / "figures" / "issue_778" / "summary_comparison"
FIG_SUBDIR = "issue_778/summary_comparison"  # stem prefix for savefig_paper(dir="figures/")

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABEL = {"evil": "evil", "sycophancy": "sycophancy", "hallucination": "hallucination"}
TRAIT_SHORT = {"evil": "evil", "sycophancy": "syco", "hallucination": "hall"}

# The five bar-panel settings, in display order: (key, label, file-suffix, json-node).
SETTINGS = [
    ("finetune", "Finetuning shift", "finetune", None),
    (
        "corr_pooled",
        "System-prompt monitoring (pooled)",
        "monitoring_corrected",
        "monitoring_overall",
    ),
    (
        "corr_within",
        "System-prompt monitoring (within prompt)",
        "monitoring_corrected",
        "monitoring_within",
    ),
    ("many_pooled", "Many-shot ICL (pooled)", "monitoring_manyshot", "monitoring_overall"),
    (
        "many_within",
        "Many-shot ICL (within shot count)",
        "monitoring_manyshot",
        "monitoring_within",
    ),
]

# Paper reference values, EVERY value a PRINTED number retrieved verbatim from the
# arXiv 2507.21509 LaTeX source (never read off figure pixels):
#   - System-prompt + many-shot, overall + within-condition: appendix table
#     "Monitoring prompt-induced persona shifts > Correlation analysis"
#     (\label{tab:correlation_analysis}). Printed values:
#         Evil          system 0.747 / 0.511   many-shot 0.755 / 0.735
#         Sycophancy    system 0.798 / 0.669   many-shot 0.817 / 0.813
#         Hallucination system 0.830 / 0.245   many-shot 0.634 / 0.400
#   - Finetuning shift: the paper prints only a matched-trait RANGE r = 0.76-0.97
#     (main text, "Activation shift along persona vector predicts trait
#     expression"); the per-trait values appear ONLY in a figure, so per the
#     printed-numbers-only rule the finetune paper bars are OMITTED.
PAPER_R: dict[str, dict[str, float | None]] = {
    "finetune": {"evil": None, "sycophancy": None, "hallucination": None},
    "corr_pooled": {"evil": 0.747, "sycophancy": 0.798, "hallucination": 0.830},
    "corr_within": {"evil": 0.511, "sycophancy": 0.669, "hallucination": 0.245},
    "many_pooled": {"evil": 0.755, "sycophancy": 0.817, "hallucination": 0.634},
    "many_within": {"evil": 0.735, "sycophancy": 0.813, "hallucination": 0.400},
}
_TAB = 'arXiv 2507.21509 App. table "Correlation analysis"'
PAPER_SOURCE = {
    "corr_pooled": f"{_TAB} (System prompting, Overall)",
    "corr_within": f"{_TAB} (System prompting, Within-condition)",
    "many_pooled": f"{_TAB} (Many-shot prompting, Overall)",
    "many_within": f"{_TAB} (Many-shot prompting, Within-condition)",
    "finetune": (
        "arXiv 2507.21509 main text prints only the matched-trait RANGE r=0.76-0.97 "
        "(no printed per-trait value; per-trait shown only in Fig. finetuning_shift_last_prompt) "
        "-> finetune paper bars omitted"
    ),
}

# The 3 raw-datapoint data settings: (key, label, suffix, node, color-by).
RAW_SETTINGS = [
    ("finetune", "Finetuning shift", "finetune", None, "family"),
    (
        "corrected",
        "System-prompt monitoring",
        "monitoring_corrected",
        "monitoring_overall",
        "condition",
    ),
    ("manyshot", "Many-shot ICL", "monitoring_manyshot", "monitoring_overall", "shot"),
]

FAMILIES = [
    "evil",
    "hallucination",
    "insecure_code",
    "mistake_gsm8k",
    "mistake_math",
    "mistake_medical",
    "mistake_opinions",
    "sycophancy",
]
VERSIONS = ["normal", "misaligned_1", "misaligned_2"]
VERSION_MARKER = {"normal": "o", "misaligned_1": "s", "misaligned_2": "^"}
VERSION_LABEL = {"normal": "Normal", "misaligned_1": "version I", "misaligned_2": "version II"}


def load_cell(trait: str, suffix: str, node: str | None) -> dict:
    """Return the plotted quantities for one bar-panel setting-cell."""
    path = EVAL_DIR / f"{trait}_{suffix}_nullbattery.json"
    d = json.loads(path.read_text())
    n = d if node is None else d[node]
    nulls = n["nulls"]
    rn = nulls["randnorm"]
    pm = nulls["perm"]
    matched = float(n["matched_max_abs"])
    ci = n.get("matched_r_bootstrap_ci_95")
    # The corrected within-prompt cells carry a bootstrap CI that does NOT bracket
    # their matched value (hallucination's within CI equals its pooled CI exactly)
    # -- an upstream bug where the within CI inherited the pooled statistic. Treat
    # any CI that fails to bracket its matched value as invalid; suppress the bar.
    ci_valid = bool(ci) and (ci[0] <= matched <= ci[1])
    return {
        "trait": trait,
        "matched": matched,
        "ci": [float(ci[0]), float(ci[1])] if ci else None,
        "ci_valid": ci_valid,
        "randnorm_cap": float(rn["r_p97_5"]),
        "randnorm_median": float(np.median(rn["draws_max_abs"])),
        "perm_cap": float(pm["r_p97_5"]),
        "perm_median": float(np.median(pm["draws_max_abs"])),
        "n_points": int(n["n_points"]),
        "source": str(path.relative_to(REPO)),
    }


def build_records() -> dict[str, dict[str, dict]]:
    return {
        key: {t: load_cell(t, suffix, node) for t in TRAITS} for key, _l, suffix, node in SETTINGS
    }


# ── Raw-datapoint loaders ────────────────────────────────────────────────────


def _load_rb(trait: str) -> np.ndarray:
    import torch

    return (
        torch.load(DATA_DIR / "rb" / f"{trait}.pt", weights_only=False).numpy().astype(np.float64)
    )


def _load_pools(trait: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    pos = torch.load(DATA_DIR / "activations" / f"{trait}_pos.pt", weights_only=False)
    neg = torch.load(DATA_DIR / "activations" / f"{trait}_neg.pt", weights_only=False)
    return pos.numpy().astype(np.float64), neg.numpy().astype(np.float64)


def _seed0_random_direction(
    pool: np.ndarray, rb_vec_at_layer: np.ndarray, sel_layer: int, n_layers: int
) -> np.ndarray:
    """Reproduce the null-battery seed-0 draw-0 norm-matched random direction at one layer.

    Matches ``null_battery.randnorm_null_draws`` exactly: default_rng(0), one
    ``standard_normal(D)`` per (draw, layer) in draw-major/layer-minor order, so
    draw-0's layers are the first ``n_layers`` draws; ``v = chol @ z`` (Cholesky of
    the shrunk pos+neg covariance, lambda=0.1), renormalized to ``||r_B[layer]||``.
    """
    chol = nb._shrunk_cholesky(pool[:, sel_layer, :], nb.PRIMARY_LAMBDA)
    d_dim = pool.shape[2]
    rng = np.random.default_rng(0)
    z_sel = None
    for layer in range(n_layers):
        z = rng.standard_normal(d_dim)
        if layer == sel_layer:
            z_sel = z
            break
    v = z_sel @ chol.T  # == chol @ z_sel
    return v * (np.linalg.norm(rb_vec_at_layer) / np.linalg.norm(v))


def _finetune_raw(trait: str) -> dict:
    """24-point finetune regression: shift projection onto r_B and onto seed-0 random dir."""
    import torch

    nbj = json.loads((EVAL_DIR / f"{trait}_finetune_nullbattery.json").read_text())
    sel = int(nbj["matched_selected_layer"])
    per_run = nbj["per_run_points"]  # tag/shift_proj_selected_layer/trait_score in kept order
    rb = _load_rb(trait)
    pos, neg = _load_pools(trait)
    pool = np.concatenate([pos, neg], axis=0)
    base = torch.load(DATA_DIR / "finetune_activations" / "base.pt", weights_only=False)[trait]
    base = base.numpy().astype(np.float64)
    shifts, ours_x, y, fams, vers = [], [], [], [], []
    for pt in per_run:
        tag = pt["tag"]
        ft = torch.load(DATA_DIR / "finetune_activations" / f"{tag}.pt", weights_only=False)[trait]
        shifts.append(ft.numpy().astype(np.float64) - base)
        ours_x.append(float(pt["shift_proj_selected_layer"]))
        y.append(float(pt["trait_score"]))
        for ver in VERSIONS:  # family = tag minus the version suffix
            if tag.endswith("_" + ver):
                fams.append(tag[: -(len(ver) + 1)])
                vers.append(ver)
                break
    shift_acts = np.stack(shifts, axis=0)  # (24, L, D)
    dir_vec = _seed0_random_direction(pool, rb[sel], sel, shift_acts.shape[1])
    rand_x = nb.project(shift_acts[:, sel, :], dir_vec)
    draws = np.load(DATA_DIR / "null_draws" / f"{trait}_finetune_randnorm_draws.npy")
    return _assemble_raw(
        trait,
        sel,
        np.array(ours_x),
        rand_x,
        np.array(y),
        draws,
        float(nbj["matched_r"]),
        color_key="family",
        fams=fams,
        vers=vers,
    )


def _monitoring_raw(trait: str, suffix: str, color_key: str) -> dict:
    """160/100-cell monitoring: projection onto r_B and onto seed-0 random dir, kept rows."""
    import torch

    nbj = json.loads((EVAL_DIR / f"{trait}_{suffix}_nullbattery.json").read_text())
    sel = int(nbj["monitoring_overall"]["matched_selected_layer"])
    matched_r = float(nbj["monitoring_overall"]["matched_r"])
    jsonl_text = (EVAL_DIR / f"{suffix}_{trait}.jsonl").read_text()
    rows = [json.loads(ln) for ln in jsonl_text.splitlines() if ln.strip()]
    acts = (
        torch.load(DATA_DIR / suffix / f"{trait}_acts.pt", weights_only=False)
        .numpy()
        .astype(np.float64)
    )
    if acts.shape[0] != len(rows):
        raise RuntimeError(f"{trait} {suffix}: acts rows {acts.shape[0]} != jsonl rows {len(rows)}")
    rb = _load_rb(trait)
    # Row-alignment self-check: projection onto r_B recomputed from acts must match the JSONL.
    chk = float(acts[0, sel] @ rb[sel] / np.linalg.norm(rb[sel]))
    if abs(chk - rows[0]["projection_per_layer"][sel]) > 1e-6:
        raise RuntimeError(
            f"{trait} {suffix}: acts<->jsonl row alignment broken at cell 0 layer {sel}"
        )
    kept = [i for i, r in enumerate(rows) if r["mean_trait_score"] is not None]
    ours_x = np.array([rows[i]["projection_per_layer"][sel] for i in kept])
    y = np.array([rows[i]["mean_trait_score"] for i in kept])
    color_field = "condition_id" if color_key == "condition" else "shot_count"
    cvals = np.array([rows[i][color_field] for i in kept])
    pos, neg = _load_pools(trait)
    pool = np.concatenate([pos, neg], axis=0)
    dir_vec = _seed0_random_direction(pool, rb[sel], sel, acts.shape[1])
    rand_x = nb.project(acts[kept][:, sel, :], dir_vec)
    stem = (
        "monitoring_corrected_overall"
        if suffix == "monitoring_corrected"
        else "monitoring_manyshot_overall"
    )
    draws = np.load(DATA_DIR / "null_draws" / f"{trait}_{stem}_randnorm_draws.npy")
    return _assemble_raw(
        trait, sel, ours_x, rand_x, y, draws, matched_r, color_key=color_key, cvals=cvals
    )


def _assemble_raw(
    trait, sel, ours_x, rand_x, y, draws, ours_r, *, color_key, fams=None, vers=None, cvals=None
) -> dict:
    rand_r = nb._pearson(rand_x, y)
    # Validate the seed-0 reproduction against the stored 1000-draw null at this layer.
    stored = float(draws[0, sel])
    if abs(abs(rand_r) - stored) > 1e-3:
        raise RuntimeError(
            f"{trait} sel={sel}: seed-0 |r| {abs(rand_r):.6f} != stored draws[0,{sel}] {stored:.6f}"
        )
    pct = float((draws[:, sel] <= abs(rand_r)).mean() * 100.0)
    return {
        "trait": trait,
        "sel_layer": sel,
        "ours_x": ours_x,
        "rand_x": rand_x,
        "y": y,
        "ours_r": ours_r,
        "rand_r": float(rand_r),
        "rand_r_abs": float(abs(rand_r)),
        "rand_percentile": pct,
        "color_key": color_key,
        "fams": fams,
        "vers": vers,
        "cvals": cvals,
    }


# ── Figure 1: comparison bar panels ──────────────────────────────────────────


def fig_bar_panels(records: dict) -> None:
    pal = paper_palette(8)
    c_ours, c_paper, c_rand, c_perm = pal[0], pal[1], pal[2], pal[7]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.4), sharey=True)
    x = np.arange(len(TRAITS))
    slot_w = 0.26
    off = {"ours": -slot_w, "paper": 0.0, "rand": slot_w}

    for ax, (key, label, _s, _n) in zip(axes, SETTINGS, strict=True):
        for i, t in enumerate(TRAITS):
            c = records[key][t]
            yerr = None
            if c["ci_valid"]:
                lo, hi = c["ci"]
                yerr = [[c["matched"] - lo], [hi - c["matched"]]]
            ax.bar(
                x[i] + off["ours"],
                c["matched"],
                slot_w * 0.92,
                color=c_ours,
                yerr=yerr,
                capsize=3,
                error_kw={"ecolor": "#333333", "elinewidth": 1.2},
                label="our persona vector |r|",
            )
            pr = PAPER_R[key][t]
            if pr is not None:
                ax.bar(
                    x[i] + off["paper"], pr, slot_w * 0.92, color=c_paper, label="paper reported r"
                )
            ax.bar(
                x[i] + off["rand"],
                c["randnorm_cap"],
                slot_w * 0.92,
                color=c_rand,
                alpha=0.35,
                label="norm-matched random (97.5th pct cap)",
            )
            xl, xr = x[i] + off["rand"] - slot_w * 0.46, x[i] + off["rand"] + slot_w * 0.46
            ax.plot(
                [xl, xr],
                [c["randnorm_median"]] * 2,
                color=c_rand,
                lw=2.2,
                label="norm-matched random (median)",
            )
            ax.plot(
                [xl, xr],
                [c["perm_cap"]] * 2,
                color=c_perm,
                lw=1.4,
                ls=(0, (3, 2)),
                label="permutation (97.5th pct cap)",
            )
        ax.set_title(label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([TRAIT_LABEL[t] for t in TRAITS], fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.axhline(0, color="#999999", lw=0.6)

    axes[0].set_ylabel("Pearson |r|  (max over layers)")
    # Collect handles across ALL panels and dedup by label so every series
    # (incl. paper, which is absent from the finetune panel) appears once.
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Persona-vector prediction |r| vs paper vs norm-matched random baseline, "
        "per trait and setting",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig_paper(fig, f"{FIG_SUBDIR}/bar_panels_by_setting", dir="figures/")
    plt.close(fig)


# ── Figures 2-4: raw-datapoint grids (ours vs seed-0 random companion) ────────


def _draw_raw_panel(ax, data, use_random: bool):
    x = data["rand_x"] if use_random else data["ours_x"]
    y = data["y"]
    if data["color_key"] == "family":
        fam_colors = {
            f: c for f, c in zip(FAMILIES, plt.cm.tab10(np.linspace(0, 1, 10)), strict=False)
        }
        for xi, yi, fam, ver in zip(x, y, data["fams"], data["vers"], strict=True):
            ax.scatter(
                xi,
                yi,
                s=55,
                color=fam_colors[fam],
                marker=VERSION_MARKER[ver],
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
    else:
        sc = ax.scatter(
            x,
            y,
            c=data["cvals"],
            cmap="viridis",
            s=42,
            edgecolor="#222222",
            linewidth=0.35,
            zorder=3,
        )
        ax.__dict__["_eps_sc"] = sc
    r = data["rand_r"] if use_random else data["ours_r"]
    if use_random:
        ax.set_title(
            f"{TRAIT_LABEL[data['trait']]} — random dir  r={r:.2f}  "
            f"(|r| {data['rand_percentile']:.0f}th pct of 1000 draws)",
            fontsize=9,
        )
    else:
        ax.set_title(f"{TRAIT_LABEL[data['trait']]} — persona vector  r={r:.2f}", fontsize=9.5)


def fig_raw_grid(key: str, label: str, suffix: str, color_key: str) -> dict:
    """One 2-row (ours / random) x 3-trait grid figure for a data setting."""
    per_trait = {
        t: (_finetune_raw(t) if key == "finetune" else _monitoring_raw(t, suffix, color_key))
        for t in TRAITS
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    sel_note = ", ".join(f"{TRAIT_SHORT[t]} L{per_trait[t]['sel_layer']}" for t in TRAITS)
    for row, use_random in enumerate((False, True)):
        for col, t in enumerate(TRAITS):
            ax = axes[row, col]
            _draw_raw_panel(ax, per_trait[t], use_random)
            ax.set_ylabel("graded trait-expression score" if col == 0 else "")
            ax.set_xlabel(
                "projection onto norm-matched random direction"
                if use_random
                else "projection onto persona vector r_B"
            )
            ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)

    if color_key == "family":
        from matplotlib.lines import Line2D

        fam_colors = {
            f: c for f, c in zip(FAMILIES, plt.cm.tab10(np.linspace(0, 1, 10)), strict=False)
        }
        fam_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                mfc=fam_colors[f],
                mec="#222",
                ms=7,
                label=f.replace("_", " "),
            )
            for f in FAMILIES
        ]
        ver_handles = [
            Line2D(
                [0],
                [0],
                marker=VERSION_MARKER[v],
                ls="",
                mfc="#999",
                mec="#222",
                ms=7,
                label=VERSION_LABEL[v],
            )
            for v in VERSIONS
        ]
        leg1 = fig.legend(
            handles=fam_handles,
            title="dataset family",
            loc="lower center",
            ncol=8,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.add_artist(leg1)
        fig.legend(
            handles=ver_handles,
            title="version",
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.075),
        )

    fig.suptitle(
        f"{label}: trait-expression score vs projection — persona vector (top) "
        f"vs one norm-matched random direction (bottom); selected layers {sel_note}",
        fontsize=12.5,
        y=0.99,
    )
    if color_key == "family":
        fig.tight_layout(rect=(0, 0.12, 1.0, 0.96))
    else:
        # A colorbar is incompatible with tight_layout, so lay out explicitly and
        # give the colorbar its own reserved axis to avoid overlapping a panel.
        sc = next(ax.__dict__["_eps_sc"] for ax in axes.flat if "_eps_sc" in ax.__dict__)
        fig.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.08, hspace=0.32, wspace=0.24)
        cax = fig.add_axes((0.915, 0.12, 0.014, 0.72))
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(
            "prompt condition (0 = strongest trait ... 7 = plain assistant)"
            if color_key == "condition"
            else "number of trait exemplars (shots)",
            fontsize=9,
        )
    savefig_paper(fig, f"{FIG_SUBDIR}/raw_scatter_{key}", dir="figures/", embed_data=False)
    plt.close(fig)
    return per_trait


# ── Figures 5-6: summary meta-scatters (kept from v1) ─────────────────────────


def _setting_style():
    pal = paper_palette(8)
    colors = {key: pal[i] for i, (key, *_r) in enumerate(SETTINGS)}
    markers = {
        "finetune": "o",
        "corr_pooled": "s",
        "corr_within": "D",
        "many_pooled": "^",
        "many_within": "v",
    }
    return colors, markers


def fig_summary_scatter(records: dict) -> None:
    colors, markers = _setting_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo, hi = 0.30, 1.0
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#666666",
        lw=1.2,
        ls="--",
        zorder=1,
        label="y = x  (our |r| = random-null cap)",
    )
    seen = set()
    for key, label, _s, _n in SETTINGS:
        for t in TRAITS:
            c = records[key][t]
            ax.scatter(
                c["randnorm_cap"],
                c["matched"],
                s=70,
                color=colors[key],
                marker=markers[key],
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
                label=label if key not in seen else None,
            )
            seen.add(key)
            ax.annotate(
                TRAIT_SHORT[t],
                (c["randnorm_cap"], c["matched"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#222222",
            )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("norm-matched random null, 97.5th-pct cap of max-over-layers |r|")
    ax.set_ylabel("our persona-vector max-over-layers |r|")
    ax.set_title(
        "Persona vector vs its norm-matched random baseline (15 setting-cells)", fontsize=11.5
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/summary_scatter_ours_vs_random", dir="figures/")
    plt.close(fig)


def fig_ours_vs_paper(records: dict) -> None:
    colors, markers = _setting_style()
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo, hi = 0.20, 0.95
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#666666",
        lw=1.2,
        ls="--",
        zorder=1,
        label="y = x  (our r = paper r)",
    )
    seen = set()
    n_pts = 0
    for key, label, _s, _n in SETTINGS:
        for t in TRAITS:
            pr = PAPER_R[key][t]
            if pr is None:
                continue
            c = records[key][t]
            ax.scatter(
                pr,
                c["matched"],
                s=70,
                color=colors[key],
                marker=markers[key],
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
                label=label if key not in seen else None,
            )
            seen.add(key)
            ax.annotate(
                TRAIT_SHORT[t],
                (pr, c["matched"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7.5,
                color="#222222",
            )
            n_pts += 1
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("paper's reported Pearson r")
    ax.set_ylabel("our reproduced Pearson |r|")
    ax.set_title(
        f"Replication fidelity: our r vs paper r ({n_pts} cells with a recorded paper value)",
        fontsize=11.5,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/scatter_ours_vs_paper", dir="figures/")
    plt.close(fig)


# ── meta.json ─────────────────────────────────────────────────────────────────


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_meta(records: dict, raw: dict) -> None:
    sources = sorted({records[k][t]["source"] for k, *_ in SETTINGS for t in TRAITS})
    plotted = {}
    for key, _label, _s, _n in SETTINGS:
        plotted[key] = {}
        for t in TRAITS:
            c = records[key][t]
            plotted[key][t] = {
                "our_matched_r": round(c["matched"], 4),
                "our_ci95": [round(v, 4) for v in c["ci"]] if c["ci"] else None,
                "our_ci95_valid": c["ci_valid"],
                "paper_r": PAPER_R[key][t],
                "randnorm_cap_97_5": round(c["randnorm_cap"], 4),
                "randnorm_median": round(c["randnorm_median"], 4),
                "perm_cap_97_5": round(c["perm_cap"], 4),
                "n_points": c["n_points"],
            }
    seed0 = {
        key: {
            t: {
                "selected_layer": d["sel_layer"],
                "ours_r": round(d["ours_r"], 4),
                "seed0_random_dir_r": round(d["rand_r"], 4),
                "seed0_random_dir_abs_r": round(d["rand_r_abs"], 4),
                "seed0_percentile_within_1000_draws_at_layer": round(d["rand_percentile"], 1),
                "n_points": len(d["y"]),
            }
            for t, d in data_by_trait.items()
        }
        for key, data_by_trait in raw.items()
    }
    meta = {
        "task": 778,
        "description": (
            "Raw-datapoint grids (persona-vector vs seed-0 norm-matched-random projection) "
            "+ summary meta-scatters + comparison bar panels for reproduced persona-vector results "
            "vs paper vs random baseline."
        ),
        "rendered_at_git_head": _git_head(),
        "data_git_commit": "39ba09d44070eede2858616bc1867f889fa28b03",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "claude-sonnet-4-5-20250929",
        "source_json_paths": sources,
        "acts_tensors": (
            "data/issue_778/{monitoring_corrected,monitoring_manyshot,finetune_activations,"
            "activations,rb}/ (HF: issue778_persona_vectors/analysis_tensors/)"
        ),
        "null_draws": (
            "data/issue_778/null_draws/{trait}_{setting}_randnorm_draws.npy (HF: .../null_draws/)"
        ),
        "settings": [{"key": k, "label": lbl} for k, lbl, *_ in SETTINGS],
        "traits": TRAITS,
        "paper_reference_values_recorded_in_repo": PAPER_R,
        "paper_reference_sources": PAPER_SOURCE,
        "paper_values_added_from_arxiv_this_revision": {
            "note": "printed numbers from arXiv 2507.21509 App. table 'Correlation analysis'",
            "many_pooled": {"evil": 0.755, "sycophancy": 0.817, "hallucination": 0.634},
            "many_within_hallucination": 0.400,
            "confirmed_from_table": {
                "corr_pooled": [0.747, 0.798, 0.830],
                "corr_within": [0.511, 0.669, 0.245],
                "many_within_evil_syco": [0.735, 0.813],
            },
        },
        "paper_values_still_unavailable": {
            "finetune": (
                "per-trait finetuning-shift r not printed (only range 0.76-0.97 in main text; "
                "per-trait in figure only)"
            ),
        },
        "ci_suppressed_note": (
            "corrected within-prompt cells (evil/sycophancy/hallucination) carry a bootstrap CI "
            "that does not bracket the matched value (hallucination's within CI equals its pooled "
            "CI exactly); those bar error bars are suppressed"
        ),
        "random_companion_note": (
            "raw-scatter bottom rows use ONE norm-matched random direction (null-battery sampler, "
            "seed 0, draw 0) at the persona-vector-selected layer; each panel states that draw's r "
            "and its percentile within the 1000-draw null AT THAT LAYER. The seed-0 draw is NOT "
            "cherry-picked; the percentile is reported so a weak/strong draw is transparent. The "
            "headline null cap uses max-over-28-layers per draw, a different statistic."
        ),
        "plotted_bar_panels": plotted,
        "raw_scatter_seed0_random": seed0,
        "figures": [
            "raw_scatter_finetune.png",
            "raw_scatter_corrected.png",
            "raw_scatter_manyshot.png",
            "bar_panels_by_setting.png",
            "summary_scatter_ours_vs_random.png",
            "scatter_ours_vs_paper.png",
        ],
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    set_paper_style("neurips")
    records = build_records()
    fig_bar_panels(records)
    raw = {
        key: fig_raw_grid(key, label, suffix, color_key)
        for key, label, suffix, _node, color_key in RAW_SETTINGS
    }
    fig_summary_scatter(records)
    fig_ours_vs_paper(records)
    write_meta(records, raw)
    print(f"Wrote figures + meta.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
