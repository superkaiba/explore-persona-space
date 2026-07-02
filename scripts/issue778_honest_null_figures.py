#!/usr/bin/env python
"""Honest-null-ladder comparison figures for task #778.

#778's original two stochastic nulls (pooled-covariance randnorm + label-shuffle
permutation) were found CIRCULAR: the pooled covariance's top principal component
aligns with the persona vector r_B (cos 0.996 / 0.985 / 0.736), so a "random"
draw from it inherits r_B's predictive power. The honest-null ladder
(scripts/issue778_honest_null_ladder.py) adds four trait-agnostic families —
isotropic, within-class covariance (PRIMARY), single-arm (negative) covariance,
and r_B-projected-out — committed at eval_results/issue_778/honest_nulls/.

This script produces:

  1. COMPARISON BAR PANELS (bar_panels_with_honest_nulls[_paper_steering].png):
     one panel per setting (finetune / system-prompt monitoring pooled + within /
     many-shot ICL pooled + within) x trait, bars in the honesty ladder order:
       paper's reported r (where printed) | our reproduced |r| (FIXED-LAYER) |
       isotropic | within-class | single-arm(neg) | r_B-projected-out |
       pooled-cov randnorm (circular) | shuffled-label perm (circular) |
       cross-trait (max).
     Observed |r| here is FIXED-LAYER (the #778-selected layer), NOT the
     max-over-28-layers regime of the earlier summary figure — the two regimes
     are never mixed in one panel. Caps are read straight from the honestnulls
     JSONs. A second file repeats the ladder at the paper-steering layer.

  2. RAW-DATAPOINT LADDER GRIDS (raw_scatter_{setting}_ladder.png): rows =
     direction method (r_B, then isotropic / within-class / single-arm(neg) /
     r_B-projected-out / pooled-cov randnorm), columns = trait. Each random row
     projects the SAME datapoints onto ONE deterministic draw-0 direction from
     that family's sampler (the exact samplers imported from the ladder script,
     using each family's recorded seed), at the #778-selected layer. Each random
     panel title states the draw's r and its |r| percentile within that family's
     1000-draw null (honesty disclosure). The paper has no raw datapoints (we do
     not have their data), so paper appears only in the bar figure.

All inputs are committed artifacts under eval_results/issue_778/ + the acts
tensors under data/issue_778/ (HF issue778_persona_vectors/analysis_tensors/).
No training, no eval generation, no GPU. Idempotent:
    uv run python scripts/issue778_honest_null_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps must land before numpy/torch import on the shared VM (#847).
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_honest_null_ladder as L  # noqa: E402

from explore_persona_space.analysis import null_battery as nb  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "eval_results" / "issue_778"
HN_DIR = EVAL_DIR / "honest_nulls"
DATA_DIR = REPO / "data" / "issue_778"
FIG_DIR = REPO / "figures" / "issue_778" / "summary_comparison"
FIG_SUBDIR = "issue_778/summary_comparison"

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_LABEL = {"evil": "evil", "sycophancy": "sycophancy", "hallucination": "hallucination"}
TRAIT_SHORT = {"evil": "evil", "sycophancy": "syco", "hallucination": "hall"}

# Bar-panel settings: (key, label, honestnulls-file-suffix, regime).
BAR_SETTINGS = [
    ("finetune", "Finetuning shift", "finetune", "overall"),
    ("corr_pooled", "System-prompt monitoring (pooled)", "monitoring_corrected", "overall"),
    ("corr_within", "System-prompt monitoring (within prompt)", "monitoring_corrected", "within"),
    ("many_pooled", "Many-shot ICL (pooled)", "monitoring_manyshot", "overall"),
    ("many_within", "Many-shot ICL (within shot count)", "monitoring_manyshot", "within"),
]

# Paper reference values (printed numbers, arXiv 2507.21509 App. "Correlation
# analysis" table); finetune per-trait not printed (range 0.76-0.97 only).
PAPER_R: dict[str, dict[str, float | None]] = {
    "finetune": {"evil": None, "sycophancy": None, "hallucination": None},
    "corr_pooled": {"evil": 0.747, "sycophancy": 0.798, "hallucination": 0.830},
    "corr_within": {"evil": 0.511, "sycophancy": 0.669, "hallucination": 0.245},
    "many_pooled": {"evil": 0.755, "sycophancy": 0.817, "hallucination": 0.634},
    "many_within": {"evil": 0.735, "sycophancy": 0.813, "hallucination": 0.400},
}

# Honesty ladder (bar series after paper + ours), in fixed display order.
# (family-key, short label, kind) — kind drives color/hatch.
LADDER = [
    ("isotropic", "isotropic", "honest"),
    ("within_class", "within-class", "honest"),
    ("neg_arm_only", "single-arm (neg)", "honest"),
    ("rb_projected_out", "r_B projected out", "honest"),
    ("orig_randnorm", "pooled-cov randnorm (circular)", "contaminated"),
    ("orig_perm", "shuffled-label perm (circular)", "contaminated"),
    ("crosstrait", "cross-trait (max)", "crosstrait"),
]
# Raw-grid random rows: Gaussian-draw families only (per the brief), r_B first.
RAW_FAMILIES = ["isotropic", "within_class", "neg_arm_only", "rb_projected_out", "orig_randnorm"]
RAW_FAMILY_LABEL = {
    "isotropic": "isotropic random",
    "within_class": "within-class covariance",
    "neg_arm_only": "single-arm (neg) covariance",
    "rb_projected_out": "r_B projected out",
    "orig_randnorm": "pooled-cov randnorm (circular)",
}

RAW_SETTINGS = [
    ("finetune", "Finetuning shift", "finetune", "family"),
    ("corrected", "System-prompt monitoring", "monitoring_corrected", "condition"),
    ("manyshot", "Many-shot ICL", "monitoring_manyshot", "shot"),
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


def _hn(trait: str, suffix: str) -> dict:
    return json.loads((HN_DIR / f"{trait}_{suffix}_honestnulls.json").read_text())


# ── Data loaders (raw acts, kept rows) ────────────────────────────────────────


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


def _finetune_predictor(trait: str) -> tuple[np.ndarray, np.ndarray, list, list]:
    """(shift_acts (24,28,D), target (24,), families, versions) in honestnulls tag order."""
    import torch

    hn = _hn(trait, "finetune")
    tags = hn["tags"]
    base = torch.load(DATA_DIR / "finetune_activations" / "base.pt", weights_only=False)[trait]
    base = base.numpy().astype(np.float64)
    ftj = json.loads((EVAL_DIR / f"{trait}_finetune_nullbattery.json").read_text())
    score_by_tag = {p["tag"]: p["trait_score"] for p in ftj["per_run_points"]}
    shifts, target, fams, vers = [], [], [], []
    for tag in tags:
        ft = torch.load(DATA_DIR / "finetune_activations" / f"{tag}.pt", weights_only=False)[trait]
        shifts.append(ft.numpy().astype(np.float64) - base)
        target.append(float(score_by_tag[tag]))
        for ver in VERSIONS:
            if tag.endswith("_" + ver):
                fams.append(tag[: -(len(ver) + 1)])
                vers.append(ver)
                break
    return np.stack(shifts, axis=0), np.array(target), fams, vers


def _monitoring_predictor(trait: str, suffix: str, color_key: str):
    """(raw_kept (n,28,D), target, color_values) for the pooled (overall) regime."""
    import torch

    rows = [
        json.loads(x)
        for x in (EVAL_DIR / f"{suffix}_{trait}.jsonl").read_text().splitlines()
        if x.strip()
    ]
    acts = (
        torch.load(DATA_DIR / suffix / f"{trait}_acts.pt", weights_only=False)
        .numpy()
        .astype(np.float64)
    )
    if acts.shape[0] != len(rows):
        raise RuntimeError(f"{trait} {suffix}: acts {acts.shape[0]} != rows {len(rows)}")
    kept = [i for i, r in enumerate(rows) if r["mean_trait_score"] is not None]
    target = np.array([rows[i]["mean_trait_score"] for i in kept])
    field = "condition_id" if color_key == "condition" else "shot_count"
    cvals = np.array([rows[i][field] for i in kept])
    return acts[kept], target, cvals


# ── Seed-0 draw-0 direction for a Gaussian-draw honest-null family ────────────


def _seed0_family_direction(fam, layer, fixed_layers, seed, chols_by_fam, rb_hat, rb_norms, d_dim):
    """Reproduce the ladder sampler's draw-0 direction for one family at one layer.

    Mirrors ``_cov_null_draws`` exactly (draw-major/layer-minor rng over
    ``fixed_layers``; v = z (isotropic) or z @ chol.T; optional r_B projection;
    renormalize to ||r_B[layer]||). Validated bit-exact against the imported
    sampler's draw-0 |r| by the caller.
    """
    li = fixed_layers.index(layer)
    rng = np.random.default_rng(seed)
    z_sel = None
    for i in range(len(fixed_layers)):
        z = rng.standard_normal(d_dim)
        if i == li:
            z_sel = z
            break
    v = z_sel if fam == "isotropic" else z_sel @ chols_by_fam[fam][layer].T
    if fam == "rb_projected_out":
        rh = rb_hat[layer]
        v = v - (v @ rh) * rh
    return v * (np.linalg.norm(rb_hat[layer]) * 0 + rb_norms[layer] / np.linalg.norm(v))


def _raw_ladder_data(setting_key: str, suffix: str, color_key: str) -> dict:
    """Per-trait: r_B projection + one seed-0 draw per RAW_FAMILIES at #778 layer."""
    out = {}
    for trait in TRAITS:
        hn = _hn(trait, suffix)
        node = hn["stage_fixed"]["overall"]
        choices = node["layer_choices"]
        layer = choices["issue778_selected"]
        fixed_layers = sorted(set(choices.values()))
        seeds = hn["seeds"]["overall"]
        pc = node["per_choice"]["issue778_selected"]

        rb = _load_rb(trait)
        rb_norms = np.linalg.norm(rb, axis=1)
        rb_hat = {ly: rb[ly] / np.linalg.norm(rb[ly]) for ly in fixed_layers}
        pos, neg = _load_pools(trait)
        pool = np.concatenate([pos, neg], axis=0)
        within_pool = L._within_centered_pool(pos, neg)
        chols_by_fam = {
            "within_class": L._chols_for_layers(within_pool, fixed_layers, nb.PRIMARY_LAMBDA),
            "neg_arm_only": L._chols_for_layers(neg, fixed_layers, nb.PRIMARY_LAMBDA),
            "rb_projected_out": L._chols_for_layers(pool, fixed_layers, nb.PRIMARY_LAMBDA),
            "orig_randnorm": L._chols_for_layers(pool, fixed_layers, nb.PRIMARY_LAMBDA),
        }

        if setting_key == "finetune":
            predictor, target, fams, vers = _finetune_predictor(trait)
            cvals = None
        else:
            predictor, target, cvals = _monitoring_predictor(trait, suffix, color_key)
            fams = vers = None

        d_dim = predictor.shape[2]
        ours_x = nb.project(predictor[:, layer, :], rb[layer])
        rows = {
            "rb": {
                "x": ours_x,
                "r": float(pc["observed_abs_r"]),
                "pct": None,
                "label": "persona vector r_B",
            }
        }
        for fam in RAW_FAMILIES:
            seed = seeds[fam]
            n_draws = pc["nulls"][fam]["n_draws"]
            chols = chols_by_fam.get(fam)
            m = L._cov_null_draws(
                chols,
                rb_norms,
                predictor,
                target,
                fixed_layers,
                isotropic=(fam == "isotropic"),
                project_out_hat=(rb_hat if fam == "rb_projected_out" else None),
                n_draws=n_draws,
                seed=seed,
                within=False,
            )
            li = fixed_layers.index(layer)
            col = m[:, li]
            # p97.5 validation against the committed JSON.
            if abs(float(np.percentile(col, 97.5)) - pc["nulls"][fam]["p97_5"]) > 1e-3:
                raise RuntimeError(f"{trait} {suffix} {fam}: regen p97.5 != JSON")
            dir_vec = _seed0_family_direction(
                fam, layer, fixed_layers, seed, chols_by_fam, rb_hat, rb_norms, d_dim
            )
            x = nb.project(predictor[:, layer, :], dir_vec)
            r = nb._pearson(x, target)
            if abs(abs(r) - float(col[0])) > 1e-3:
                raise RuntimeError(f"{trait} {suffix} {fam}: draw-0 |r| != sampler m[0]")
            pct = float((col <= abs(r)).mean() * 100.0)
            rows[fam] = {"x": x, "r": float(r), "pct": pct, "label": RAW_FAMILY_LABEL[fam]}
        out[trait] = {
            "layer": layer,
            "y": target,
            "rows": rows,
            "color_key": color_key,
            "fams": fams,
            "vers": vers,
            "cvals": cvals,
        }
    return out


# ── Figure 1: honest-null ladder bar panels ───────────────────────────────────


def _ladder_colors():
    pal = paper_palette(8)
    return {
        "paper": pal[1],
        "ours": pal[0],
        "isotropic": pal[2],
        "within_class": pal[3],
        "neg_arm_only": pal[4],
        "rb_projected_out": pal[5],
        "orig_randnorm": "#9a9a9a",
        "orig_perm": "#9a9a9a",
        "crosstrait": "#8c564b",
    }


def _band_at(node_choice: dict, fam: str) -> tuple[float, float, float]:
    """(median, lo, hi) of a family's null distribution at one layer choice.

    Stochastic families: (p50, p2.5, p97.5) — the full 95% band. Cross-trait has
    only 2 fixed directions, so its "band" is (median, min, max) of the 2 values.
    """
    e = node_choice["nulls"][fam]
    if fam == "crosstrait":
        vals = e["values"]
        return float(np.median(vals)), float(min(vals)), float(max(vals))
    return float(e["p50"]), float(e["p2_5"]), float(e["p97_5"])


_CELL_CACHE: dict = {}


def _load_cell_cached(suffix: str, trait: str):
    key = (suffix, trait)
    if key not in _CELL_CACHE:
        _CELL_CACHE[key] = L._load_cell(suffix, DATA_DIR, EVAL_DIR, trait)
    return _CELL_CACHE[key]


def _ours_ci(trait, suffix, regime, plotted_layer, own_argmax, seeds, ci_json):
    """95% bootstrap CI on our |r| AT THE PLOTTED LAYER, recomputed via the ladder's
    imported bootstrap helper (seed = the recorded cell_idx, n_boot=1000) and
    validated bit-exact against the committed JSON's at-own-argmax CI.

    The JSON stores the CI only at own_argmax; for cells where the plotted layer
    differs (the within-regime bars) that stored CI is at the wrong layer, so we
    recompute at the plotted layer. Returns ((lo, hi), valid). valid is False (and
    the whisker is suppressed) if the own-argmax reproduction does not match the
    JSON or the recomputed CI fails to bracket the observed value.
    """
    cell_idx = int(seeds["isotropic"]) - 100000  # SEED_BASE["isotropic"] == 100000
    rb = _load_rb(trait)
    predictor, target, cid, _tags = _load_cell_cached(suffix, trait)
    within = regime == "within"

    def ci_at(layer: int) -> tuple[float, float]:
        if within:
            return L._bootstrap_ci_within(
                predictor, rb, target, cid, layer, n_boot=1000, seed=cell_idx
            )
        return nb.bootstrap_ci_matched_r(predictor, rb, target, layer, n_boot=1000, seed=cell_idx)

    lo_oa, hi_oa = ci_at(own_argmax)
    reproduced = (
        ci_json is not None and abs(lo_oa - ci_json[0]) < 1e-6 and abs(hi_oa - ci_json[1]) < 1e-6
    )
    lo, hi = ci_at(plotted_layer)
    return (float(lo), float(hi)), bool(reproduced)


def fig_bar_ladder(layer_choice: str, stem: str, title_layer: str) -> dict:
    colors = _ladder_colors()
    fig, axes = plt.subplots(5, 1, figsize=(17.5, 17.0))
    x = np.arange(len(TRAITS))
    # bar slots: paper, ours, then the 7 ladder families.
    slots = ["paper", "ours", *[k for k, _l, _kind in LADDER]]
    nslot = len(slots)
    width = 0.9 / nslot
    plotted = {}

    ekw = {"ecolor": "#222222", "elinewidth": 1.1, "capsize": 2.5}
    for ax, (key, label, suffix, regime) in zip(axes, BAR_SETTINGS, strict=True):
        plotted[key] = {}
        for i, trait in enumerate(TRAITS):
            hn = _hn(trait, suffix)
            node = hn["stage_fixed"][regime]
            pc = node["per_choice"][layer_choice]
            layer = pc["layer"]
            base = x[i] - 0.45 + width / 2
            obs = float(pc["observed_abs_r"])
            rec = {"layer": layer, "our_fixed_r": round(obs, 4), "paper_r": PAPER_R[key][trait]}
            for j, slot in enumerate(slots):
                xpos = base + j * width
                if slot == "paper":
                    # Paper prints point values without CIs -> no whisker.
                    pr = PAPER_R[key][trait]
                    if pr is not None:
                        ax.bar(
                            xpos,
                            pr,
                            width * 0.9,
                            color=colors["paper"],
                            label="paper reported r (no CI)",
                        )
                elif slot == "ours":
                    # 95% bootstrap CI recomputed at the plotted layer, validated vs JSON.
                    (lo, hi), ci_valid = _ours_ci(
                        trait,
                        suffix,
                        regime,
                        layer,
                        node["own_argmax_layer"],
                        hn["seeds"][regime],
                        node.get("bootstrap_ci95_at_own_argmax"),
                    )
                    ci_valid = ci_valid and lo <= obs <= hi
                    yerr = [[obs - lo], [hi - obs]] if ci_valid else None
                    ax.bar(
                        xpos,
                        obs,
                        width * 0.9,
                        color=colors["ours"],
                        yerr=yerr,
                        error_kw=ekw,
                        label="our persona vector |r| (fixed layer, 95% bootstrap CI)",
                    )
                    rec["our_ci95"] = [round(lo, 4), round(hi, 4)] if ci_valid else None
                    rec["our_ci95_valid"] = ci_valid
                else:
                    # Null family: median bar + p2.5-p97.5 band whiskers (cap + spread visible).
                    med, blo, bhi = _band_at(pc, slot)
                    kind = next(kd for k, _l, kd in LADDER if k == slot)
                    hatch = (
                        "//" if slot == "orig_randnorm" else ("xx" if slot == "orig_perm" else None)
                    )
                    lab = next(lb for k, lb, _kd in LADDER if k == slot)
                    ax.bar(
                        xpos,
                        med,
                        width * 0.9,
                        color=colors[slot],
                        alpha=0.55 if kind == "contaminated" else 0.9,
                        hatch=hatch,
                        edgecolor="#333333" if kind == "contaminated" else "none",
                        yerr=[[med - blo], [bhi - med]],
                        error_kw=ekw,
                        label=f"{lab} (median + 2.5-97.5 pct band)",
                    )
                    rec[slot] = {
                        "median": round(med, 4),
                        "p2_5": round(blo, 4),
                        "p97_5": round(bhi, 4),
                    }
            plotted[key][trait] = rec
        ax.set_title(f"{label}  (fixed layer per trait)", fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels([TRAIT_LABEL[t] for t in TRAITS], fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Pearson |r|")
        ax.axhline(0, color="#999999", lw=0.6)

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
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Persona vector (paper vs our reproduction) vs each random-direction null — "
        f"honesty ladder, {title_layer}\n"
        "ours: 95% bootstrap CI · null families: median + 2.5-97.5 pct band · "
        "paper: point value (no CI)",
        fontsize=12,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    savefig_paper(fig, f"{FIG_SUBDIR}/{stem}", dir="figures/", embed_data=False)
    plt.close(fig)
    return plotted


# ── Figures 2-4: raw-datapoint ladder grids ───────────────────────────────────


def _draw_panel(ax, x, y, color_key, fams, vers, cvals):
    if color_key == "family":
        fam_colors = {
            f: c for f, c in zip(FAMILIES, plt.cm.tab10(np.linspace(0, 1, 10)), strict=False)
        }
        for xi, yi, fam, ver in zip(x, y, fams, vers, strict=True):
            ax.scatter(
                xi,
                yi,
                s=48,
                color=fam_colors[fam],
                marker=VERSION_MARKER[ver],
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
        return None
    sc = ax.scatter(
        x, y, c=cvals, cmap="viridis", s=36, edgecolor="#222222", linewidth=0.3, zorder=3
    )
    return sc


def fig_raw_ladder(setting_key: str, label: str, suffix: str, color_key: str) -> dict:
    data = _raw_ladder_data(setting_key, suffix, color_key)
    row_keys = ["rb", *RAW_FAMILIES]
    nrows = len(row_keys)
    fig, axes = plt.subplots(nrows, 3, figsize=(15.5, 3.5 * nrows))
    sc_for_cbar = None
    sel_note = ", ".join(f"{TRAIT_SHORT[t]} L{data[t]['layer']}" for t in TRAITS)

    for r, rk in enumerate(row_keys):
        for c, trait in enumerate(TRAITS):
            ax = axes[r, c]
            d = data[trait]
            row = d["rows"][rk]
            sc = _draw_panel(ax, row["x"], d["y"], color_key, d["fams"], d["vers"], d["cvals"])
            if sc is not None:
                sc_for_cbar = sc
            ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
            if rk == "rb":
                ax.set_title(
                    f"{TRAIT_LABEL[trait]} — persona vector  r={row['r']:.2f}", fontsize=9.5
                )
            else:
                ax.set_title(
                    f"{TRAIT_LABEL[trait]} — {row['label']}  r={row['r']:.2f} "
                    f"(|r| {row['pct']:.0f}th pct)",
                    fontsize=8.5,
                )
            if c == 0:
                ax.set_ylabel("graded trait score")
            if r == nrows - 1:
                ax.set_xlabel("projection onto direction")

    fig.suptitle(
        f"{label}: trait-expression score vs projection onto the persona vector (top row) "
        f"and each random-direction null (seed-0 draw); selected layers {sel_note}",
        fontsize=12.5,
        y=0.995,
    )
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
            bbox_to_anchor=(0.5, -0.012),
        )
        fig.add_artist(leg1)
        fig.legend(
            handles=ver_handles,
            title="version",
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.03),
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.985))
    else:
        fig.subplots_adjust(left=0.06, right=0.9, top=0.955, bottom=0.05, hspace=0.4, wspace=0.22)
        cax = fig.add_axes((0.915, 0.1, 0.013, 0.8))
        cbar = fig.colorbar(sc_for_cbar, cax=cax)
        cbar.set_label(
            "prompt condition (0 = strongest trait ... 7 = plain assistant)"
            if color_key == "condition"
            else "number of trait exemplars (shots)",
            fontsize=9,
        )
    savefig_paper(
        fig, f"{FIG_SUBDIR}/raw_scatter_{setting_key}_ladder", dir="figures/", embed_data=False
    )
    plt.close(fig)
    # meta payload
    return {
        t: {
            "selected_layer": data[t]["layer"],
            "ours_r": round(data[t]["rows"]["rb"]["r"], 4),
            "families": {
                fam: {
                    "seed0_r": round(data[t]["rows"][fam]["r"], 4),
                    "seed0_abs_r_percentile": round(data[t]["rows"][fam]["pct"], 1),
                }
                for fam in RAW_FAMILIES
            },
            "n_points": len(data[t]["y"]),
        }
        for t in TRAITS
    }


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    set_paper_style("neurips")
    bar_778 = fig_bar_ladder(
        "issue778_selected", "bar_panels_with_honest_nulls", "at the #778-selected read-out layer"
    )
    fig_bar_ladder(
        "paper_steering",
        "bar_panels_with_honest_nulls_paper_steering",
        "at the paper's steering layer",
    )
    raw_meta = {
        key: fig_raw_ladder(key, label, suffix, color_key)
        for key, label, suffix, color_key in RAW_SETTINGS
    }

    meta = {
        "task": 778,
        "description": (
            "Honest-null-ladder comparison figures: paper vs our reproduction "
            "vs each random-direction null."
        ),
        "rendered_at_git_head": _git_head(),
        "honestnulls_source": (
            "eval_results/issue_778/honest_nulls/{trait}_{setting}_honestnulls.json"
        ),
        "honestnulls_git_commit": "97c70bf755 (committed by the honest-null-ladder run)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "claude-sonnet-4-5-20250929",
        "circularity_finding": (
            "the two ORIGINAL #778 stochastic nulls (orig_randnorm pooled-cov, orig_perm "
            "shuffled-label) are circular: pooled-cov top PC vs r_B cos 0.996/0.985/0.736; "
            "the honest families (isotropic, within_class [primary], neg_arm_only, "
            "rb_projected_out) are trait-agnostic"
        ),
        "regime_note": (
            "bar observed |r| is FIXED-LAYER (the #778-selected or paper-steering layer), NOT "
            "the max-over-28-layers regime of summary_scatter_ours_vs_random.png; regimes are "
            "never mixed in one panel"
        ),
        "paper_note": (
            "paper appears only in the bar figure — we do not have the paper's raw datapoints"
        ),
        "ci_note": (
            "bar figure CIs: our |r| = 95% bootstrap CI recomputed AT THE PLOTTED LAYER via the "
            "ladder's imported bootstrap helper (seed = recorded cell_idx, n_boot=1000), validated "
            "bit-exact against the committed JSON's bootstrap_ci95_at_own_argmax (within cells' "
            "CIs are within-condition-resampled, not pooled); a cell whose recompute fails to "
            "reproduce the JSON or fails to bracket the observed value has its whisker suppressed "
            "+ our_ci95_valid=false. Null-family bars are the MEDIAN with p2.5-p97.5 whiskers "
            "(full 95% band from 1000 draws; cross-trait uses median/min/max of its 2 directions). "
            "Paper "
            "bars are point values with no whisker (the paper prints no CIs)."
        ),
        "paper_reference_values": PAPER_R,
        "bar_ladder_issue778_selected": bar_778,
        "raw_scatter_seed0_by_family": raw_meta,
        "figures": [
            "bar_panels_with_honest_nulls.png",
            "bar_panels_with_honest_nulls_paper_steering.png",
            "raw_scatter_finetune_ladder.png",
            "raw_scatter_corrected_ladder.png",
            "raw_scatter_manyshot_ladder.png",
        ],
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "meta_honest_nulls.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote honest-null figures + meta to {FIG_DIR}")


if __name__ == "__main__":
    main()
