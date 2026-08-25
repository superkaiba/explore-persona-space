#!/usr/bin/env python3
"""Task #2569 P-F figures: every figure with a LANDED producing schema (plan v4 §6).

Every value plotted here is READ from a persisted JSON/NPZ written by a landed
#2569 driver; nothing is re-derived (no operator products, no SVDs, no fits).
Each figure function is input-guarded: a missing phase artifact records a skip
in ``<fig-dir>/figures_manifest.json`` instead of crashing the batch (P-F may
run while some pods are still up). Rendering follows the paper-plots
conventions: ``set_paper_style`` rcParams, ``savefig_paper`` sidecars, axes +
ticks + legend + panel titles only (no caption blocks, no value annotations;
identity labels on small per-unit plots are the one sanctioned text overlay).

Input schemas were read off the producing code in this worktree at 57808a6434:

- ``leg2_curve/learning_curve.json``    issue2569_gateladder.curve_core / fit_point
- ``leg2/gate_ladder{,_partial}.json``  issue2569_gateladder.run_ladder / race_arm
- ``leg4/feature_map_metrics.json``     issue2569_rowbattery.phase_feature_map
- ``leg4/perfeature_leg4.npz``          issue2569_rowbattery.phase_feature_map
- ``leg8/mining_summary.json``          issue2569_rowbattery.phase_mine
- ``leg8/kernel_pairs.json``            issue2569_rowbattery.phase_mine
- ``der/der_eval.json``                 issue2569_rowbattery.phase_der_eval
- ``dw_fleet/{lora,ft}/*.json``         issue2569_dw_fleet.analyze_{lora_arm,ft_checkpoint}
- ``dw_fleet/alignment.json``           issue2569_dw_fleet.cmd_align
- ``leg6/<arm>/L*_*.json``              issue2569_leg6.fit_split_half / run_arm
- ``leg7/three_tier.json``              issue2569_atlas.phase_report
- ``leg7/atlas_distances.json``         issue2569_atlas.phase_atlas

Deferred by design (no producing code landed yet, so no schema exists to key
on): the leg-1 hero anatomy figure, the eigen-vs-singular scatter, the
two-sided SAE dashboards, the leg-3 wiring edge-mass + attribution tables, and
the leg-6 cross-arm shared-factor heatmap. See the figures-unit concern on the
task ledger; those functions are added once the P-A weights driver lands.

Usage:
    uv run python scripts/issue2569_figures.py \
        [--results-root eval_results/issue_2569] [--fig-dir figures/issue_2569] \
        [--only leg2_learning_curve,leg7_atlas] [--import-check]
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numpy/matplotlib: shared-VM thread caps freeze at import (#847/#891)

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

logger = logging.getLogger("issue2569.figures")

ISSUE = 2569

# ---------------------------------------------------------------------------
# Display-name map: ONE map every figure reads. No internal slug reaches an
# axis label, legend entry, or panel title (standing user directive). A later
# rename lands HERE once, never per figure.
# ---------------------------------------------------------------------------
_PAL = paper_palette(16)
COLOR_PRIMARY = _PAL[0]  # the fitted / observed / primary object of a panel
COLOR_COMPARISON = _PAL[1]  # comparison route / control pairs / companions
COLOR_THEORY = _PAL[5]  # closed-form theory prediction (leg 2 curve)
COLOR_NEUTRAL = paper_palette_role("neutral")  # nulls, chance, floors, references

DISPLAY: dict[str, str] = {
    # leg-2 gate rungs (plan §5 config slugs gate_I..gate_wwt)
    "gate_I": "Raw similarity (identity)",
    "gate_diag_inv": "Diagonally whitened",
    "gate_sigma_inv": "Fully whitened (inverse covariance)",
    "gate_wwt": "Through-map (W W^T)",
    "gate_wwt_k90": "Through-map, top-k (90% mass)",
    "gate_wwt_awhite": "Through-map, answer-whitened",
    # leg-2 DV columns
    "dv_change": "leakage change (judged)",
    "dv_level": "leakage level (judged)",
    "dv_dlogp": "marker delta log-prob",
    "dv_level_logp": "marker log-prob level",
    # leg-4 comparison routes
    "fitted_map": "Fitted feature-to-feature map",
    "composed_banked_2476": "Composed route (no fit)",
    "dense_input_banked_2476": "Dense-input ridge (banked)",
    "index_aligned_ib": "Index-aligned identity + bias (null)",
    "train_mean_null": "Train-mean (null)",
    # leg-7 tier-2 routes
    "native": "Native Llama map",
    "composed_banked": "Composed (banked Qwen map)",
    "composed_matched": "Composed (matched-n Qwen map)",
    "alignment_only_baseline": "Alignment-only baseline (no operator)",
    # leg-6 context conventions
    "last_prompt": "last prompt token",
    "last_ctx": "last context token",
    "span_mean": "context span mean",
    # LoRA / full-FT module names (HF transformer weight nomenclature)
    "q_proj": "query proj.",
    "k_proj": "key proj.",
    "v_proj": "value proj.",
    "o_proj": "attn. output proj.",
    "gate_proj": "MLP gate proj.",
    "up_proj": "MLP up proj.",
    "down_proj": "MLP down proj.",
    # alignment direction families (leg 5)
    "delta_tbar": "realized mean write direction",
    "c_C": "training-context centroid",
}

GATE_METRIC_ORDER = (
    "gate_I",
    "gate_diag_inv",
    "gate_sigma_inv",
    "gate_wwt",
    "gate_wwt_k90",
    "gate_wwt_awhite",
)
# Metric colors avoid slots 0/1/5 (reserved for the primary/comparison/theory roles).
METRIC_COLORS = dict(zip(GATE_METRIC_ORDER, [_PAL[2], _PAL[3], _PAL[4], _PAL[6], _PAL[7], _PAL[8]]))

# Matryoshka tier bounds, #2476 recipe (plan §0 Training: 2,048 / 16,384 / 65,536).
TIER_BOUNDS = (2048, 16384, 65536)
TIER_NAMES = ("Coarse tier (first 2,048)", "Mid tier (to 16,384)", "Specific tier (to 65,536)")
TIER_COLORS = (_PAL[9], _PAL[10], _PAL[11])

_BEH_WORDS = {"cas": "Casualness", "imp": "Impoliteness", "syc": "Sycophancy", "mk": "Marker"}
_FRAMING_WORDS = {"pers": "persona", "bare": "bare", "conv": "conversational", "icl": "ICL"}
_REGIME_WORDS = {"con": "contrastive", "po": "positive-only"}


def arm_label(arm_id: str) -> str:
    """Plain-English label for a #1979-grammar arm id (beh-framing-regime-lr-seed).

    ``cas-pers-con-lr1e5-s42`` becomes ``Casualness, persona, contrastive``;
    a full fine-tune gains ``(full FT)``. Unknown grammars pass through so a
    new fleet member renders its id rather than crashing the batch.
    """
    toks = arm_id.split("-")
    if not toks or toks[0] not in _BEH_WORDS:
        return arm_id
    parts = [_BEH_WORDS[toks[0]]]
    is_ft = "ft" in toks[1:]
    for t in toks[1:]:
        if t in _FRAMING_WORDS:
            parts.append(_FRAMING_WORDS[t])
        elif t in _REGIME_WORDS:
            parts.append(_REGIME_WORDS[t])
    label = ", ".join(parts)
    return f"{label} (full FT)" if is_ft else label


def arm_labels_deduped(arm_ids: list[str]) -> dict[str, str]:
    """Per-arm labels with seed suffixes added ONLY where labels would collide."""
    base = {a: arm_label(a) for a in arm_ids}
    counts: dict[str, int] = {}
    for v in base.values():
        counts[v] = counts.get(v, 0) + 1
    out = {}
    for a, v in base.items():
        if counts[v] > 1:
            seed = next((t[1:] for t in a.split("-") if t.startswith("s") and t[1:].isdigit()), "")
            out[a] = f"{v} (seed {seed})" if seed else f"{v} ({a})"
        else:
            out[a] = v
    return out


def display(key: str) -> str:
    """Display name for an internal key: the label map, else the key itself."""
    return DISPLAY.get(key, key)


def tier_of(feat_ids: np.ndarray) -> np.ndarray:
    """Matryoshka tier index (0/1/2) per feature id, from the #2476 tier bounds."""
    ids = np.asarray(feat_ids, np.int64)
    return np.digitize(ids, TIER_BOUNDS[:-1], right=False)


def _read_json(path: Path) -> dict | None:
    """Load a JSON input if present; None records a manifest skip upstream."""
    return json.loads(path.read_text()) if path.is_file() else None


def _finite(vals) -> np.ndarray:
    """Float array with non-finite entries dropped (guarded aggregation input)."""
    a = np.asarray(vals, np.float64).ravel()
    return a[np.isfinite(a)]


def _render(fig: plt.Figure, stem: str, fig_dir: Path) -> str:
    """Save one figure through savefig_paper (sidecar + commit pin) and close it."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem, dir=fig_dir)
    plt.close(fig)
    return stem


# ---------------------------------------------------------------------------
# Leg 2: learning curve (B4)
# ---------------------------------------------------------------------------


def build_learning_curve(doc: dict) -> plt.Figure:
    """Predicted-vs-empirical held-out R^2 over n (verdict points + companions).

    Panel A: empirical verdict series, the closed-form theory prediction at the
    same n, and the committed off-recipe companions (open markers; a companion
    whose committed lambda sat at a grid edge gets its own marker). Panel B:
    per-point delta (empirical minus theory) against the registered H2b bands.
    Per-point corpus mix / selected lambda / grid-edge metadata stays in
    ``learning_curve.json`` and the caption prose (no on-canvas text).
    """
    pts = doc["verdict_points"]
    if not pts:
        raise ValueError("learning_curve.json carries no verdict points")
    n = np.asarray([p["n_train"] for p in pts], np.float64)
    emp = np.asarray([p["test_r2"] for p in pts], np.float64)
    theory = np.asarray([p["theory"]["predicted_r2"] for p in pts], np.float64)
    order = np.argsort(n)
    n, emp, theory = n[order], emp[order], theory[order]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    ax_a.plot(n, emp, "o-", color=COLOR_PRIMARY, label="Empirical held-out R^2 (LMSYS refits)")
    ax_a.plot(n, theory, "D--", color=COLOR_THEORY, label="Theory prediction (closed-form ridge)")
    comp = doc.get("companions_off_recipe") or []
    reg, edge = [], []
    for c in comp:
        (edge if c.get("lambda_grid_edge") else reg).append((c["n_train"], c["test_r2"]))
    if reg:
        xs, ys = zip(*reg)
        ax_a.plot(
            xs, ys, "s", mfc="none", color=COLOR_COMPARISON, label="Off-recipe committed point"
        )
    if edge:
        xs, ys = zip(*edge)
        ax_a.plot(
            xs,
            ys,
            "^",
            mfc="none",
            color=COLOR_COMPARISON,
            label="Off-recipe point, lambda at grid edge",
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Training rows (n)")
    ax_a.set_ylabel("Held-out R^2")
    ax_a.set_title("Learning curve: theory vs measured")
    ax_a.legend()

    delta = emp - theory
    ax_b.axhline(0.0, color=COLOR_NEUTRAL, lw=1)
    bands = doc["h2b"]["bands"]
    for sign in (1, -1):
        ax_b.axhline(
            sign * bands["pass_le"],
            color=COLOR_NEUTRAL,
            ls="--",
            lw=1,
            label="H2b pass band" if sign == 1 else None,
        )
        ax_b.axhline(
            sign * bands["kill_gt"],
            color=COLOR_NEUTRAL,
            ls=":",
            lw=1,
            label="H2b kill floor" if sign == 1 else None,
        )
    ax_b.plot(n, delta, "o-", color=COLOR_PRIMARY, label="Empirical minus theory")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Training rows (n)")
    ax_b.set_ylabel("Delta R^2")
    ax_b.set_title("Per-point misfit vs registered bands")
    ax_b.legend()
    return fig


# ---------------------------------------------------------------------------
# Leg 2: gate-metric ladder
# ---------------------------------------------------------------------------


def build_gate_ladder(doc: dict, kind: str) -> plt.Figure:
    """Per-arm ladder dumbbells + across-arm champion summary for one arm kind.

    Panel A (per-unit view): one row per arm, one point per raced gate rung on
    the primary DV, with each arm's selection-symmetric permutation band (p97.5
    of the per-draw signed max over rungs) as a grey tick. Panel B (summary):
    across-arm median rho per rung plus the champion's selection-inherited and
    frozen-at-winner CIs, both labeled.
    """
    arms = {a: r for a, r in doc["per_arm"].items() if r["kind"] == kind}
    if not arms:
        raise ValueError(f"gate ladder carries no {kind!r} arms")
    dv = {"content": "dv_change", "marker": "dv_dlogp"}[kind]
    labels = arm_labels_deduped(sorted(arms))
    arm_ids = sorted(arms, key=lambda a: labels[a])

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.0, 0.42 * len(arm_ids) + 2.8), width_ratios=(3, 2), layout="constrained"
    )
    for y, aid in enumerate(arm_ids):
        obs = arms[aid]["observed_rho"][dv]
        for m in GATE_METRIC_ORDER:
            if m in obs:
                ax_a.plot(obs[m], y, "o", color=METRIC_COLORS[m], ms=5)
        ax_a.plot(
            arms[aid]["perm_band"]["p975_max_selected"], y, "|", color=COLOR_NEUTRAL, ms=12, mew=2
        )
    for m in GATE_METRIC_ORDER:
        if any(m in arms[a]["observed_rho"][dv] for a in arm_ids):
            ax_a.plot([], [], "o", color=METRIC_COLORS[m], label=display(m))
    ax_a.plot([], [], "|", color=COLOR_NEUTRAL, label="Selection-symmetric null (p97.5)")
    ax_a.set_yticks(range(len(arm_ids)), [labels[a] for a in arm_ids])
    ax_a.set_xlabel(f"Within-arm Spearman rho vs {display(dv)}")
    ax_a.set_title("Gate-rung race per arm")
    ax_a.axvline(0.0, color=COLOR_NEUTRAL, lw=0.8)
    ax_a.legend(fontsize=7, loc="best")

    champ = doc.get("champion", {}).get(kind, {}).get(dv)
    metrics = [
        m for m in GATE_METRIC_ORDER if any(m in arms[a]["observed_rho"][dv] for a in arm_ids)
    ]
    med = {
        m: float(
            np.median(
                [
                    arms[a]["observed_rho"][dv][m]
                    for a in arm_ids
                    if m in arms[a]["observed_rho"][dv]
                ]
            )
        )
        for m in metrics
    }
    xs = np.arange(len(metrics))
    for i, m in enumerate(metrics):
        ax_b.plot(i, med[m], "o", color=METRIC_COLORS[m], ms=7)
    if champ:
        wi = (
            metrics.index(champ["winner_observed"]) if champ["winner_observed"] in metrics else None
        )
        if wi is not None:
            sel = champ["selection_inherited_ci_max_median"]
            frz = champ["frozen_ci_winner_median (labeled: frozen-at-winner)"]
            ax_b.vlines(
                wi - 0.15,
                sel[0],
                sel[1],
                color=COLOR_NEUTRAL,
                lw=2,
                label="Selection-inherited CI (95%)",
            )
            ax_b.vlines(
                wi + 0.15,
                frz[0],
                frz[1],
                color=COLOR_NEUTRAL,
                lw=2,
                ls=":",
                label="Frozen-at-winner CI (95%)",
            )
    ax_b.set_xticks(xs, [display(m) for m in metrics], rotation=30, ha="right", fontsize=7)
    ax_b.set_ylabel("Across-arm median rho")
    ax_b.set_title("Champion summary")
    ax_b.axhline(0.0, color=COLOR_NEUTRAL, lw=0.8)
    ax_b.legend(fontsize=7)
    return fig


def build_gate_family_table(doc: dict) -> plt.Figure:
    """Per-prefix-family across-arm median rho heatmap (the leg-2 OOD fold read)."""
    ft = doc["per_family_win_table"]
    fams = [f for f, rec in sorted(ft["families"].items()) if "across_arm_median" in rec]
    if not fams:
        raise ValueError("per-family win table has no populated families")
    metrics = [m for m in GATE_METRIC_ORDER if m in ft["metrics"]]
    mat = np.array(
        [[ft["families"][f]["across_arm_median"].get(m, np.nan) for m in metrics] for f in fams]
    )
    fig, ax = plt.subplots(
        figsize=(1.1 * len(metrics) + 2.5, 0.42 * len(fams) + 1.8), layout="constrained"
    )
    # A hairline colorbar-label overflow (x1 == canvas width) clips the last glyph
    # at savefig dpi; widen the constrained-layout pad to reserve real margin.
    fig.get_layout_engine().set(w_pad=0.15)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
    ax.set_xticks(
        range(len(metrics)), [display(m) for m in metrics], rotation=30, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(fams)), fams, fontsize=7)
    ax.set_title("Across-arm median rho per prefix family")
    fig.colorbar(im, ax=ax, label="Median Spearman rho")
    return fig


# ---------------------------------------------------------------------------
# Leg 4: feature map routes
# ---------------------------------------------------------------------------


def build_leg4_routes(metrics_doc: dict, npz) -> plt.Figure:
    """Route-comparison summary + per-tier medians + firing-AUROC distribution.

    Panel A: median per-feature held-out R^2 per comparison route (fitted map
    in the primary color, banked comparison routes in the comparison color,
    nulls in grey), with the row-shuffle-null median as the floor line. Panel
    B: per-tier medians for the routes carried as per-feature arrays in the
    NPZ, with whole-union banked-route medians as reference lines. Panel C:
    the firing-AUROC distribution for the fitted map with the 0.5 chance line.
    """
    routes = metrics_doc["routes"]
    feat_ids = np.asarray(npz["feat_ids"], np.int64)
    shuffle_med = float(
        np.nanmedian(_finite(np.asarray(npz["shuffle_null_r2_fitted"], np.float64)))
    )

    names, meds, colors = [], [], []
    for r in routes:
        name = r["route"]
        if "r2_unconditional" in r:
            med = r["r2_unconditional"]["median"]
        else:
            med = r.get("r2_median_on_intersection", np.nan)
        names.append(display(name))
        meds.append(float(med))
        if name == "fitted_map":
            colors.append(COLOR_PRIMARY)
        elif name.endswith("_null") or name == "index_aligned_ib":
            colors.append(COLOR_NEUTRAL)
        else:
            colors.append(COLOR_COMPARISON)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.0, 3.9), layout="constrained")
    ax_a.bar(range(len(names)), meds, color=colors)
    ax_a.axhline(shuffle_med, color=COLOR_NEUTRAL, ls="--", lw=1, label="Row-shuffle null (median)")
    ax_a.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    ax_a.set_ylabel("Median per-feature held-out R^2")
    ax_a.set_title("Comparison routes")
    ax_a.legend(fontsize=7)

    tiers = tier_of(feat_ids)
    npz_routes = [
        ("r2_fitted_map", display("fitted_map"), COLOR_PRIMARY),
        ("r2_index_aligned_ib", display("index_aligned_ib"), COLOR_NEUTRAL),
        ("r2_train_mean_null", display("train_mean_null"), COLOR_NEUTRAL),
    ]
    width = 0.8 / len(npz_routes)
    for k, (key, label, color) in enumerate(npz_routes):
        vals = np.asarray(npz[key], np.float64)
        per_tier = [
            float(np.nanmedian(vals[tiers == t])) if np.any(tiers == t) else np.nan
            for t in range(3)
        ]
        hatchless = ax_b.bar(
            np.arange(3) + (k - (len(npz_routes) - 1) / 2) * width,
            per_tier,
            width=width,
            color=color,
            label=label,
        )
        if key == "r2_train_mean_null":
            for patch in hatchless:
                patch.set_alpha(0.45)
    ax_b.set_xticks(range(3), TIER_NAMES, fontsize=7)
    ax_b.set_ylabel("Median per-feature held-out R^2")
    ax_b.set_title("Per-tier medians")
    ax_b.legend(fontsize=7)

    auroc = _finite(np.asarray(npz["auroc_fitted_map"], np.float64))
    ax_c.hist(auroc, bins=30, color=COLOR_PRIMARY)
    ax_c.axvline(0.5, color=COLOR_NEUTRAL, ls="--", lw=1, label="Chance (0.5)")
    ax_c.set_xlabel("Firing AUROC per feature (fitted map)")
    ax_c.set_ylabel("Features")
    ax_c.set_title("Hurdle: firing detection")
    ax_c.legend(fontsize=7)
    return fig


def build_leg4_per_feature(npz) -> plt.Figure:
    """Per-unit view: fitted-map per-feature R^2 vs holdout firing rate, by tier.

    Feature identity travels in the savefig sidecar's per-point data; at ~2,000
    features per tier, on-canvas identity labels are unreadable and stay off.
    """
    feat_ids = np.asarray(npz["feat_ids"], np.int64)
    r2 = np.asarray(npz["r2_fitted_map"], np.float64)
    act = np.asarray(npz["activity_te"], np.float64)
    tiers = tier_of(feat_ids)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    for t in range(3):
        m = (tiers == t) & np.isfinite(r2)
        if m.any():
            ax.plot(
                act[m], r2[m], "o", ms=2.5, alpha=0.6, color=TIER_COLORS[t], label=TIER_NAMES[t]
            )
    ax.axhline(0.0, color=COLOR_NEUTRAL, lw=0.8)
    ax.set_xlabel("Holdout firing rate per feature")
    ax.set_ylabel("Per-feature held-out R^2 (fitted map)")
    ax.set_title("Per-feature predictability vs activity")
    ax.legend(fontsize=7)
    return fig


# ---------------------------------------------------------------------------
# Leg 8: kernel pairs
# ---------------------------------------------------------------------------


def build_leg8_kernel(summary: dict, pairs_doc: dict) -> plt.Figure:
    """Kernel-vs-control realized answer distances, three views (plan §6 + C2).

    Panel A: ECDFs of realized answer-state distance for kernel pairs and
    distance-matched controls, with the measured held-out residual-pair floor
    quantiles drawn. Panel B: paired per-pair scatter with the parity line.
    Panel C: paired-ratio distribution with the clustered bootstrap CI and the
    ratio = 1 reference.
    """
    pairs = pairs_doc["pairs"]
    if not pairs:
        raise ValueError("kernel_pairs.json carries no pairs")
    dva_k = np.asarray([p["dva_norm"] for p in pairs], np.float64)
    dva_c = np.asarray([p["control"]["dva_norm"] for p in pairs], np.float64)
    floor = summary["residual_floor"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.0, 3.9), layout="constrained")
    for vals, color, label in (
        (dva_k, COLOR_PRIMARY, "Kernel pairs (map says equivalent)"),
        (dva_c, COLOR_COMPARISON, "Distance-matched controls"),
    ):
        xs = np.sort(vals)
        ax_a.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=color, label=label)
    ax_a.axvline(
        floor["q50"],
        color=COLOR_NEUTRAL,
        ls="--",
        lw=1,
        label="Held-out residual-pair floor (median)",
    )
    for q in ("q10", "q90"):
        ax_a.axvline(floor[q], color=COLOR_NEUTRAL, ls=":", lw=0.8)
    ax_a.set_xlabel("Realized answer-state distance")
    ax_a.set_ylabel("Cumulative fraction of pairs")
    ax_a.set_title("Distance distributions")
    ax_a.legend(fontsize=7)

    lim = float(max(dva_k.max(), dva_c.max())) * 1.05
    ax_b.plot([0, lim], [0, lim], color=COLOR_NEUTRAL, lw=1, label="Equal distance")
    ax_b.plot(dva_c, dva_k, "o", ms=3, alpha=0.6, color=COLOR_PRIMARY)
    ax_b.set_xlabel("Control-pair answer distance")
    ax_b.set_ylabel("Kernel-pair answer distance")
    ax_b.set_title("Paired pairs")
    ax_b.legend(fontsize=7)

    ok = (dva_k > 0) & (dva_c > 0)
    ratios = dva_k[ok] / dva_c[ok]
    ci = summary["clustered_bootstrap"]["ci95"]
    ax_c.hist(ratios, bins=30, color=COLOR_PRIMARY)
    ax_c.axvline(1.0, color=COLOR_NEUTRAL, lw=1, label="Ratio = 1")
    ax_c.axvline(
        float(summary["ratio_stats"]["median_of_paired_ratios"]),
        color=COLOR_PRIMARY,
        ls="--",
        lw=1.2,
        label="Median paired ratio",
    )
    ax_c.axvspan(
        ci[0], ci[1], color=COLOR_PRIMARY, alpha=0.15, label="Clustered bootstrap CI (95%)"
    )
    ax_c.set_xlabel("Kernel / control distance ratio per pair")
    ax_c.set_ylabel("Pairs")
    ax_c.set_title("Paired ratios")
    ax_c.legend(fontsize=7)
    return fig


# ---------------------------------------------------------------------------
# Der-protocol matching (leg 4 step 5)
# ---------------------------------------------------------------------------


def build_der_matching(doc: dict) -> plt.Figure:
    """Der-protocol 10-way matching accuracy vs chance, plus description coverage."""
    m = doc["matching"]
    if m["accuracy"] is None:
        raise ValueError("der_eval.json has no answered matching items")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 3.6), layout="constrained")
    ax_a.bar([0], [m["accuracy"]], color=COLOR_PRIMARY, width=0.5)
    ax_a.axhline(
        m["chance"], color=COLOR_NEUTRAL, ls="--", lw=1, label=f"Chance ({m['n_way']}-way)"
    )
    ax_a.set_xticks([0], [f"{m['n_way']}-way description matching"])
    ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(0, 1)
    ax_a.set_title("Matching accuracy")
    ax_a.legend(fontsize=7)

    cov = doc["coverage"]
    ax_b.bar([0], [cov["union_coverage"]], color=COLOR_COMPARISON, width=0.5)
    ax_b.set_xticks([0], ["Described fraction of the answer-feature union"])
    ax_b.set_ylabel("Coverage")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("Description coverage")
    return fig


# ---------------------------------------------------------------------------
# Leg 5: dW fleet
# ---------------------------------------------------------------------------


def build_dw_effective_rank(lora_recs: list[dict], ft_recs: list[dict]) -> plt.Figure:
    """Effective-rank summaries per arm class (LoRA vs full fine-tune).

    Panel A: stable rank per (LoRA arm, module, layer) cell, grouped by module,
    with the r/2 reference for the rank-32 fleet. Panel B: top-1 spectral
    energy share per full-FT weight matrix, grouped by arm, with the 0.6
    rank-1 criterion line.
    """
    if not lora_recs and not ft_recs:
        raise ValueError("no dW unit records found")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.0), layout="constrained")

    modules = sorted({m for rec in lora_recs for m in rec.get("modules", {})})
    rng = np.random.default_rng(0)
    for i, mod in enumerate(modules):
        vals = [
            summ["stable_rank"]
            for rec in lora_recs
            for summ in rec["modules"].get(mod, {}).values()
        ]
        if vals:
            x = i + rng.uniform(-0.18, 0.18, len(vals))
            ax_a.plot(x, vals, "o", ms=3, alpha=0.55, color=COLOR_PRIMARY)
    ax_a.axhline(
        16, color=COLOR_NEUTRAL, ls="--", lw=1, label="r/2 reference (rank-32 LoRA budget)"
    )
    ax_a.set_xticks(
        range(len(modules)), [display(m) for m in modules], rotation=30, ha="right", fontsize=7
    )
    ax_a.set_ylabel("Stable rank of dW")
    ax_a.set_title("LoRA arms: stable rank per (arm, module, layer)")
    ax_a.legend(fontsize=7)

    ft_ids = [rec["arm_id"] for rec in ft_recs]
    labels = arm_labels_deduped(ft_ids)
    for i, rec in enumerate(ft_recs):
        vals = [summ["top1_share_energy"] for summ in rec["matrices"].values()]
        x = i + rng.uniform(-0.18, 0.18, len(vals))
        ax_b.plot(x, vals, "o", ms=3, alpha=0.55, color=COLOR_PRIMARY)
    ax_b.axhline(0.6, color=COLOR_NEUTRAL, ls="--", lw=1, label="Rank-1 criterion (0.6)")
    ax_b.set_xticks(
        range(len(ft_ids)), [labels[a] for a in ft_ids], rotation=20, ha="right", fontsize=7
    )
    ax_b.set_ylabel("Top-1 spectral energy share")
    ax_b.set_title("Full-FT arms: per weight matrix")
    ax_b.legend(fontsize=7)
    return fig


def build_dw_intruder(lora_recs: list[dict]) -> plt.Figure:
    """Intruder read per (arm, module): observed band max vs the matched null p95.

    Points above the parity line sit inside the base column space; points below
    are intruder-like at the max-matched null (the #650 convention). Small-N
    per-unit plot, so points carry identity labels.
    """
    rows = []
    for rec in lora_recs:
        for mod, payload in rec.get("intruder", {}).items():
            rows.append(
                (
                    rec["arm_id"],
                    mod,
                    float(payload["observed"]["write"]["band_max"]),
                    float(payload["null"]["write"]["band_p95"]),
                )
            )
    if not rows:
        raise ValueError("no intruder payloads in the LoRA records (base SVD not staged?)")
    labels = arm_labels_deduped(sorted({r[0] for r in rows}))
    fig, ax = plt.subplots(figsize=(6.8, 5.2), layout="constrained")
    lim = max(max(r[2] for r in rows), max(r[3] for r in rows)) * 1.15
    ax.plot([0, lim], [0, lim], color=COLOR_NEUTRAL, lw=1, label="Observed = null p95")
    for aid, mod, obs, p95 in rows:
        ax.plot(p95, obs, "o", ms=5, color=COLOR_PRIMARY)
        ax.annotate(
            f"{labels[aid]} / {display(mod)}",
            (p95, obs),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
        )
    ax.set_xlabel("Max-matched null p95 (max |cos| to base singular vectors)")
    ax.set_ylabel("Observed band max |cos|")
    ax.set_title("Intruder read vs the #650 max-matched null")
    ax.legend(fontsize=7)
    return fig


def build_dw_alignment(align_doc: dict) -> plt.Figure:
    """Top-dW-factor alignment per direction family, per arm, vs the matched null.

    One panel per direction (persona-vector reads named per trait); filled
    points cleared the null p95, open points did not; grey ticks mark each
    cell's null p95. The seed-pair anchor, when present, is the dotted line.
    """
    arms = align_doc["arms"]
    if not arms:
        raise ValueError("alignment.json carries no arms")
    directions = sorted(
        {
            d
            for arm_rec in arms.values()
            for mod_rec in arm_rec["factors"].values()
            for d, v in mod_rec.items()
            if isinstance(v, dict) and "max_abs_cos" in v
        }
    )
    if not directions:
        raise ValueError("alignment.json carries no scored directions")
    labels = arm_labels_deduped(sorted(arms))
    arm_ids = sorted(arms, key=lambda a: labels[a])
    anchor = align_doc.get("seed_noise_anchor", {})
    anchor_vals = [
        v["top1_abs_cos"] for v in anchor.values() if isinstance(v, dict) and "top1_abs_cos" in v
    ]

    ncols = min(3, len(directions))
    nrows = int(np.ceil(len(directions) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 0.30 * len(arm_ids) * nrows + 2.2 * nrows),
        layout="constrained",
        squeeze=False,
    )
    for k, direction in enumerate(directions):
        ax = axes[k // ncols][k % ncols]
        for y, aid in enumerate(arm_ids):
            for mod_key, mod_rec in arms[aid]["factors"].items():
                cell = mod_rec.get(direction)
                if not isinstance(cell, dict) or "max_abs_cos" not in cell:
                    continue
                mfc = COLOR_PRIMARY if cell["above_null"] else "none"
                ax.plot(cell["max_abs_cos"], y, "o", color=COLOR_PRIMARY, mfc=mfc, ms=4)
                ax.plot(cell["null_p95"], y, "|", color=COLOR_NEUTRAL, ms=8, mew=1.5)
        if anchor_vals:
            ax.axvline(max(anchor_vals), color=COLOR_NEUTRAL, ls=":", lw=1)
        ax.set_yticks(range(len(arm_ids)), [labels[a] for a in arm_ids], fontsize=6)
        ax.set_title(_direction_label(direction), fontsize=8)
        ax.set_xlabel("Max |cos| over top dW factors", fontsize=7)
    for k in range(len(directions), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=COLOR_PRIMARY, label="Above null p95"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=COLOR_PRIMARY,
            mfc="none",
            label="At or below null p95",
        ),
        plt.Line2D([], [], marker="|", ls="none", color=COLOR_NEUTRAL, label="Matched null p95"),
    ]
    if anchor_vals:
        handles.append(
            plt.Line2D([], [], ls=":", color=COLOR_NEUTRAL, label="Seed-pair noise anchor")
        )
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=7)
    return fig


def _direction_label(direction: str) -> str:
    """Plain-English panel title for an alignment direction key."""
    if direction.startswith("r_B[") and direction.endswith("]"):
        return f"{direction[4:-1].capitalize()} persona direction"
    if direction.startswith("Ar[") and direction.endswith("]"):
        return f"Mapped-read gradient ({direction[3:-1]})"
    lab = display(direction)
    return lab[:1].upper() + lab[1:]


# ---------------------------------------------------------------------------
# Leg 6: denoised reduced-rank shift regression
# ---------------------------------------------------------------------------


def _leg6_units(leg6_dir: Path, convention: str, layer: int) -> dict[str, dict]:
    """Per-arm unit records for one (layer, convention), pooled units excluded."""
    out: dict[str, dict] = {}
    for unit in sorted(leg6_dir.glob(f"*/L{layer}_{convention}.json")):
        if unit.parent.name == "pooled":
            continue
        rec = json.loads(unit.read_text())
        if "denoised_rank" in rec:
            out[rec.get("arm", unit.parent.name)] = rec
    return out


def build_leg6_ranks(units: dict[str, dict], convention: str) -> plt.Figure:
    """Denoised rank per arm (calibrated estimator) beside its analytic reference.

    Panel A: the B2 denoised rank per content arm (bars) with the
    Gavish-Donoho reference count as an open marker (analytic reference only;
    the operative threshold is the calibrated row-shuffle p95). Panel B: the
    cross-half held-out R^2 per arm, both fit directions.
    """
    if not units:
        raise ValueError("no leg-6 unit records")
    labels = arm_labels_deduped(sorted(units))
    arm_ids = sorted(units, key=lambda a: labels[a])
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.0, 0.4 * len(arm_ids) + 2.6), layout="constrained"
    )
    ys = np.arange(len(arm_ids))
    ranks = [units[a]["denoised_rank"] for a in arm_ids]
    gd = [units[a]["gavish_donoho_reference_count"] for a in arm_ids]
    ax_a.barh(ys, ranks, color=COLOR_PRIMARY, label="Denoised rank (calibrated)")
    ax_a.plot(
        gd, ys, "D", mfc="none", color=COLOR_NEUTRAL, label="Gavish-Donoho reference (analytic)"
    )
    ax_a.set_yticks(ys, [labels[a] for a in arm_ids])
    ax_a.set_xlabel(f"Rank ({display(convention)} context)")
    ax_a.set_title("Denoised shift rank per arm")
    ax_a.legend(fontsize=7)

    r12 = [units[a]["heldout_r2"]["fit1_eval2"] for a in arm_ids]
    r21 = [units[a]["heldout_r2"]["fit2_eval1"] for a in arm_ids]
    ax_b.plot(r12, ys, "o", color=COLOR_PRIMARY, label="Fit half 1, score half 2")
    ax_b.plot(r21, ys, "o", mfc="none", color=COLOR_PRIMARY, label="Fit half 2, score half 1")
    ax_b.axvline(0.0, color=COLOR_NEUTRAL, lw=0.8)
    ax_b.set_yticks(ys, ["" for _ in arm_ids])
    ax_b.set_xlabel("Cross-half held-out R^2")
    ax_b.set_title("Half-map generalization")
    ax_b.legend(fontsize=7)
    return fig


def build_leg6_spectra(units: dict[str, dict], convention: str) -> plt.Figure:
    """Per-arm half-map singular values with the calibrated noise thresholds drawn.

    Small multiples, one panel per arm: leading singular values of the two
    independent half maps, each half's row-shuffle p95 threshold (the operative
    noise calibration), log scale.
    """
    if not units:
        raise ValueError("no leg-6 unit records")
    labels = arm_labels_deduped(sorted(units))
    arm_ids = sorted(units, key=lambda a: labels[a])
    ncols = min(4, len(arm_ids))
    nrows = int(np.ceil(len(arm_ids) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows), layout="constrained", squeeze=False
    )
    for k, aid in enumerate(arm_ids):
        ax = axes[k // ncols][k % ncols]
        rec = units[aid]
        s1 = np.asarray(rec["singular_values_half1"], np.float64)
        s2 = np.asarray(rec["singular_values_half2"], np.float64)
        ax.plot(np.arange(1, len(s1) + 1), s1, "o-", ms=3, color=COLOR_PRIMARY, label="Half 1")
        ax.plot(
            np.arange(1, len(s2) + 1),
            s2,
            "o--",
            ms=3,
            mfc="none",
            color=COLOR_PRIMARY,
            label="Half 2",
        )
        thr1, thr2 = rec["shuffle_threshold_p95"]
        ax.axhline(thr1, color=COLOR_NEUTRAL, ls="--", lw=1, label="Shuffle p95 (half 1)")
        ax.axhline(thr2, color=COLOR_NEUTRAL, ls=":", lw=1, label="Shuffle p95 (half 2)")
        if np.all(s1 > 0) and np.all(s2 > 0):
            ax.set_yscale("log")
        ax.set_title(labels[aid], fontsize=7)
        if k == 0:
            ax.legend(fontsize=6)
        ax.set_xlabel("Factor index", fontsize=7)
        ax.set_ylabel("Singular value", fontsize=7)
    for k in range(len(arm_ids), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Leg 7: three-tier report + atlas
# ---------------------------------------------------------------------------


def build_three_tier(doc: dict) -> plt.Figure:
    """Cross-model three-tier report: alignability grid, operator routes, diagnostics.

    Panel A (tier 1): held-out alignment R^2 per matched layer pair for both
    summaries and both directions, with linear CKA as the descriptive
    companion and the selected working pair marked. Panel B (tier 2): held-out
    R^2 of the operator routes; the alignment-only route is the baseline.
    Panel C (tier 3, non-identifying diagnostics): the native-minus-composed
    gaps beside that baseline. Panel D: the corpus-transfer fold.
    """
    grid = doc["tier1_alignability"]["grid"]
    if not grid:
        raise ValueError("three_tier.json has an empty tier-1 grid")
    wp = doc["working_pair"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    pair_names = [f"Qwen L{p['qwen_layer']} / Llama L{p['llama_layer']}" for p in grid]
    xs = np.arange(len(grid))
    series = (
        ("vc_q2l", "Context state, Qwen to Llama", COLOR_PRIMARY, "o", "-"),
        ("vc_l2q", "Context state, Llama to Qwen", COLOR_PRIMARY, "o", "--"),
        ("va_q2l", "Answer summary, Qwen to Llama", COLOR_COMPARISON, "s", "-"),
        ("va_l2q", "Answer summary, Llama to Qwen", COLOR_COMPARISON, "s", "--"),
    )
    for key, label, color, marker, ls in series:
        vals = [p["fits"][key]["test_r2"] if key in p["fits"] else np.nan for p in grid]
        ax_a.plot(
            xs,
            vals,
            marker=marker,
            ls=ls,
            color=color,
            label=label,
            mfc="none" if ls == "--" else color,
        )
    for tag, color in (("vc", COLOR_PRIMARY), ("va", COLOR_COMPARISON)):
        cka = [p["cka"].get(tag, np.nan) for p in grid]
        ax_a.plot(
            xs,
            cka,
            "+",
            color=color,
            ms=9,
            mew=2,
            label=f"Linear CKA ({'context' if tag == 'vc' else 'answer'})",
        )
    wp_ix = next(
        (
            i
            for i, p in enumerate(grid)
            if p["qwen_layer"] == wp["qwen_layer"] and p["llama_layer"] == wp["llama_layer"]
        ),
        None,
    )
    if wp_ix is not None:
        ax_a.axvline(wp_ix, color=COLOR_NEUTRAL, lw=0.8, ls="--", label="Selected working pair")
    ax_a.set_xticks(xs, pair_names, fontsize=7)
    ax_a.set_ylabel("Held-out alignment R^2 / CKA")
    ax_a.set_title("Tier 1: representation alignability")
    ax_a.legend(fontsize=6)

    routes = doc["tier2_operator_similarity"]["routes"]
    route_keys = [
        k
        for k in ("native", "composed_banked", "composed_matched", "alignment_only_baseline")
        if k in routes
    ]
    colors = {
        "native": COLOR_COMPARISON,
        "composed_banked": COLOR_PRIMARY,
        "composed_matched": COLOR_PRIMARY,
        "alignment_only_baseline": COLOR_NEUTRAL,
    }
    ax_b.bar(
        range(len(route_keys)),
        [routes[k]["r2"] for k in route_keys],
        color=[colors[k] for k in route_keys],
    )
    ax_b.set_xticks(
        range(len(route_keys)),
        [display(k) for k in route_keys],
        rotation=20,
        ha="right",
        fontsize=7,
    )
    ax_b.set_ylabel("Held-out R^2 (Llama answer summary)")
    ax_b.set_title("Tier 2: operator routes")

    d = doc["tier3_diagnostics"]
    diag = [
        ("Native minus composed (banked)", d["r2_native_minus_composed_banked"]),
        ("Native minus composed (matched-n)", d["r2_native_minus_composed_matched"]),
        ("Alignment-only baseline R^2", d["r2_alignment_only_baseline"]),
    ]
    ax_c.bar(range(len(diag)), [v for _, v in diag], color=COLOR_NEUTRAL)
    ax_c.set_xticks(range(len(diag)), [k for k, _ in diag], rotation=20, ha="right", fontsize=7)
    ax_c.set_ylabel("R^2 / R^2 gap")
    ax_c.set_title("Tier 3: non-identifying diagnostics")

    ct = doc.get("corpus_transfer") or {}
    if ct:
        folds = sorted(ct)
        w = 0.35
        fx = np.arange(len(folds))
        ax_d.bar(
            fx - w / 2,
            [ct[f]["llama_native_r2"] for f in folds],
            width=w,
            color=COLOR_COMPARISON,
            label="Native Llama map",
        )
        ax_d.bar(
            fx + w / 2,
            [ct[f]["align_c_q2l_r2"] for f in folds],
            width=w,
            color=COLOR_PRIMARY,
            label="Context alignment (Qwen to Llama)",
        )
        ax_d.set_xticks(fx, folds, fontsize=7)
        ax_d.set_ylabel("Held-out R^2")
        ax_d.set_title("Corpus-transfer fold")
        ax_d.legend(fontsize=7)
    else:
        ax_d.set_visible(False)
    return fig


def _short_operator_name(name: str) -> str:
    """Compact display name for an atlas operator row (keeps the layer token)."""
    return name.replace("_", " ")


def build_atlas(doc: dict) -> plt.Figure:
    """Operator atlas: pairwise distance matrix + presentation-only 2-D embedding.

    Panel A: 1 minus aligned operator cosine per pair (direction-aware where an
    alignment exists; pairs that fall back to the rotation-invariant spectrum
    cosine are marked, since a rotation-invariant statistic can never support a
    same-operator-up-to-rotation reading). Diagonal cells carry each refittable
    operator's split-half self-distance, the noise floor. Panel B: classical
    MDS of the same matrix, points labeled; every claim reads off the table.
    """
    rows = doc["rows"]
    table = doc["distance_table"]
    if not rows or not table:
        raise ValueError("atlas_distances.json has no rows or pairs")
    names = [r["name"] for r in rows]
    ix = {n: i for i, n in enumerate(names)}
    n = len(names)
    dist = np.full((n, n), np.nan)
    fallback_cells = []
    for entry in table:
        i, j = ix[entry["pair"][0]], ix[entry["pair"][1]]
        if entry.get("cosine") is not None:
            d = 1.0 - float(entry["cosine"]["raw_cosine"])
        else:
            d = 1.0 - float(entry["spectrum"]["spectrum_cosine"])
            fallback_cells.append((i, j))
        dist[i, j] = dist[j, i] = max(0.0, d)
    for i, r in enumerate(rows):
        floor = (r.get("floor") or {}).get("floor") if isinstance(r.get("floor"), dict) else None
        dist[i, i] = (1.0 - float(floor)) if floor is not None else np.nan

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    im = ax_a.imshow(dist, cmap="viridis")
    for i, j in fallback_cells:
        for a, b in ((i, j), (j, i)):
            ax_a.plot(b, a, "s", ms=9, mew=1.5, mfc="none", color="white")
    disp_names = [_short_operator_name(x) for x in names]
    ax_a.set_xticks(range(n), disp_names, rotation=45, ha="right", fontsize=6)
    ax_a.set_yticks(range(n), disp_names, fontsize=6)
    ax_a.set_title("Operator distances (direction-aware where aligned)")
    fig.colorbar(im, ax=ax_a, label="1 - aligned cosine (diagonal: split-half self-distance)")
    fallback_proxy = plt.Line2D(
        [],
        [],
        marker="s",
        ls="none",
        mfc="none",
        color=COLOR_NEUTRAL,
        label="Spectrum-only pair (rotation-invariant ceiling)",
    )
    fig.legend(handles=[fallback_proxy], loc="outside lower center", fontsize=7)

    coords = doc["mds_2d"]["coords"]
    for name in names:
        x, y = coords[name]
        ax_b.plot(x, y, "o", color=COLOR_PRIMARY, ms=5)
        ax_b.annotate(
            _short_operator_name(name),
            (x, y),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
        )
    ax_b.set_xlabel("MDS dimension 1")
    ax_b.set_ylabel("MDS dimension 2")
    ax_b.set_title("Classical MDS (presentation only)")
    return fig


# ---------------------------------------------------------------------------
# Driver: load inputs, guard, render, manifest
# ---------------------------------------------------------------------------


def fig_leg2_learning_curve(root: Path, fig_dir: Path) -> str:
    """Load learning_curve.json (leg2_curve, else leg2) and render the curve."""
    doc = _read_json(root / "leg2_curve" / "learning_curve.json") or _read_json(
        root / "leg2" / "learning_curve.json"
    )
    if doc is None:
        raise FileNotFoundError("learning_curve.json (leg2_curve/ or leg2/)")
    return _render(build_learning_curve(doc), "leg2_learning_curve", fig_dir)


def _load_gate_ladder(root: Path) -> dict:
    """gate_ladder.json, else the P-A partial file (Sigma rungs absent)."""
    doc = _read_json(root / "leg2" / "gate_ladder.json") or _read_json(
        root / "leg2" / "gate_ladder_partial.json"
    )
    if doc is None:
        raise FileNotFoundError("leg2/gate_ladder{,_partial}.json")
    return doc


def fig_leg2_gate_ladder_content(root: Path, fig_dir: Path) -> str:
    """Content-arm gate-ladder dumbbells + champion summary."""
    return _render(
        build_gate_ladder(_load_gate_ladder(root), "content"), "leg2_gate_ladder_content", fig_dir
    )


def fig_leg2_gate_ladder_marker(root: Path, fig_dir: Path) -> str:
    """Marker-arm gate-ladder dumbbells + champion summary."""
    return _render(
        build_gate_ladder(_load_gate_ladder(root), "marker"), "leg2_gate_ladder_marker", fig_dir
    )


def fig_leg2_gate_family_table(root: Path, fig_dir: Path) -> str:
    """Per-prefix-family median-rho heatmap (leg-2 OOD fold)."""
    return _render(
        build_gate_family_table(_load_gate_ladder(root)), "leg2_gate_family_table", fig_dir
    )


def fig_leg4_routes(root: Path, fig_dir: Path) -> str:
    """Leg-4 route comparison + per-tier medians + AUROC distribution."""
    metrics_doc = _read_json(root / "leg4" / "feature_map_metrics.json")
    npz_path = root / "leg4" / "perfeature_leg4.npz"
    if metrics_doc is None or not npz_path.is_file():
        raise FileNotFoundError("leg4/feature_map_metrics.json + perfeature_leg4.npz")
    with np.load(npz_path) as npz:
        return _render(build_leg4_routes(metrics_doc, npz), "leg4_routes", fig_dir)


def fig_leg4_per_feature(root: Path, fig_dir: Path) -> str:
    """Leg-4 per-feature R^2 vs firing rate, colored by tier."""
    npz_path = root / "leg4" / "perfeature_leg4.npz"
    if not npz_path.is_file():
        raise FileNotFoundError("leg4/perfeature_leg4.npz")
    with np.load(npz_path) as npz:
        return _render(build_leg4_per_feature(npz), "leg4_per_feature", fig_dir)


def fig_leg8_kernel_pairs(root: Path, fig_dir: Path) -> str:
    """Leg-8 kernel-vs-control distances, paired view, and paired-ratio panel."""
    summary = _read_json(root / "leg8" / "mining_summary.json")
    pairs = _read_json(root / "leg8" / "kernel_pairs.json")
    if summary is None or pairs is None:
        raise FileNotFoundError("leg8/mining_summary.json + kernel_pairs.json")
    return _render(build_leg8_kernel(summary, pairs), "leg8_kernel_pairs", fig_dir)


def fig_der_matching(root: Path, fig_dir: Path) -> str:
    """Der-protocol matching accuracy vs chance + description coverage."""
    doc = _read_json(root / "der" / "der_eval.json")
    if doc is None:
        raise FileNotFoundError("der/der_eval.json")
    return _render(build_der_matching(doc), "leg4_der_matching", fig_dir)


def _load_dw_units(root: Path, sub: str) -> list[dict]:
    """All per-arm dW unit records under dw_fleet/<sub>/ (sorted by arm id)."""
    return [json.loads(p.read_text()) for p in sorted((root / "dw_fleet" / sub).glob("*.json"))]


def fig_dw_effective_rank(root: Path, fig_dir: Path) -> str:
    """Leg-5 effective-rank summaries per arm class."""
    lora = _load_dw_units(root, "lora")
    ft = _load_dw_units(root, "ft")
    if not lora and not ft:
        raise FileNotFoundError("dw_fleet/{lora,ft}/*.json")
    return _render(build_dw_effective_rank(lora, ft), "leg5_dw_effective_rank", fig_dir)


def fig_dw_intruder(root: Path, fig_dir: Path) -> str:
    """Leg-5 intruder scatter vs the max-matched null."""
    lora = _load_dw_units(root, "lora")
    if not lora:
        raise FileNotFoundError("dw_fleet/lora/*.json")
    return _render(build_dw_intruder(lora), "leg5_dw_intruder", fig_dir)


def fig_dw_alignment(root: Path, fig_dir: Path) -> str:
    """Leg-5 dW factor alignment vs direction families."""
    doc = _read_json(root / "dw_fleet" / "alignment.json")
    if doc is None:
        raise FileNotFoundError("dw_fleet/alignment.json")
    return _render(build_dw_alignment(doc), "leg5_dw_alignment", fig_dir)


LEG6_PRIMARY_CONVENTION = "last_prompt"  # plan-wide v_C convention (plan §6 pooling row)
LEG6_PRIMARY_LAYER = 19


def fig_leg6_ranks(root: Path, fig_dir: Path) -> str:
    """Leg-6 denoised-rank bars + cross-half generalization (primary convention)."""
    units = _leg6_units(root / "leg6", LEG6_PRIMARY_CONVENTION, LEG6_PRIMARY_LAYER)
    if not units:
        raise FileNotFoundError(f"leg6/*/L{LEG6_PRIMARY_LAYER}_{LEG6_PRIMARY_CONVENTION}.json")
    return _render(build_leg6_ranks(units, LEG6_PRIMARY_CONVENTION), "leg6_denoised_rank", fig_dir)


def fig_leg6_spectra(root: Path, fig_dir: Path) -> str:
    """Leg-6 per-arm half-map spectra with calibrated thresholds."""
    units = _leg6_units(root / "leg6", LEG6_PRIMARY_CONVENTION, LEG6_PRIMARY_LAYER)
    if not units:
        raise FileNotFoundError(f"leg6/*/L{LEG6_PRIMARY_LAYER}_{LEG6_PRIMARY_CONVENTION}.json")
    return _render(build_leg6_spectra(units, LEG6_PRIMARY_CONVENTION), "leg6_half_spectra", fig_dir)


def fig_leg7_three_tier(root: Path, fig_dir: Path) -> str:
    """Leg-7 three-tier report figure."""
    doc = _read_json(root / "leg7" / "three_tier.json")
    if doc is None:
        raise FileNotFoundError("leg7/three_tier.json")
    return _render(build_three_tier(doc), "leg7_three_tier", fig_dir)


def fig_leg7_atlas(root: Path, fig_dir: Path) -> str:
    """Leg-7 operator atlas distance matrix + MDS."""
    doc = _read_json(root / "leg7" / "atlas_distances.json")
    if doc is None:
        raise FileNotFoundError("leg7/atlas_distances.json")
    return _render(build_atlas(doc), "leg7_atlas", fig_dir)


FIGURES: dict[str, object] = {
    "leg2_learning_curve": fig_leg2_learning_curve,
    "leg2_gate_ladder_content": fig_leg2_gate_ladder_content,
    "leg2_gate_ladder_marker": fig_leg2_gate_ladder_marker,
    "leg2_gate_family_table": fig_leg2_gate_family_table,
    "leg4_routes": fig_leg4_routes,
    "leg4_per_feature": fig_leg4_per_feature,
    "leg4_der_matching": fig_der_matching,
    "leg5_dw_effective_rank": fig_dw_effective_rank,
    "leg5_dw_intruder": fig_dw_intruder,
    "leg5_dw_alignment": fig_dw_alignment,
    "leg6_denoised_rank": fig_leg6_ranks,
    "leg6_half_spectra": fig_leg6_spectra,
    "leg7_three_tier": fig_leg7_three_tier,
    "leg7_atlas": fig_leg7_atlas,
    "leg8_kernel_pairs": fig_leg8_kernel_pairs,
}

# Plan §6 figures whose PRODUCING driver has not landed; enumerated so the
# manifest names the gap instead of silently omitting it (plan hero included).
DEFERRED_NO_PRODUCER = {
    "leg1_anatomy_hero": "P-A weights driver (leg-1 operator reads) not landed",
    "leg1_eigen_vs_singular": "P-A weights driver (leg-1 operator reads) not landed",
    "leg1_sae_dashboards": "P-A weights driver (leg-1 dashboards) not landed",
    "leg3_wiring_edge_mass": "P-A weights driver (leg-3 wiring receipts) not landed",
    "leg3_attribution_tables": "P-A weights driver (leg-3 receipts) not landed",
    "leg6_shared_factor_heatmap": "cross-arm factor vectors are not persisted by issue2569_leg6",
}


def _manifest_meta() -> dict:
    """Reproducibility metadata for the figures manifest (commit + versions + ts)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    md = as_metadata_dict(git_provenance(), phase="figures")
    md.update(
        {
            "issue": ISSUE,
            "numpy": str(np.__version__),
            "matplotlib": str(matplotlib.__version__),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return md


def render_all(root: Path, fig_dir: Path, only: set[str] | None = None) -> dict:
    """Render every figure with present inputs; record skips; write the manifest."""
    set_paper_style("blog")
    rendered: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        try:
            rendered[name] = fn(root, fig_dir)
            logger.info("[figures] rendered %s", name)
        except (FileNotFoundError, ValueError, KeyError) as err:
            skipped[name] = f"{type(err).__name__}: {err}"
            logger.warning("[figures] skip %s (%s)", name, skipped[name])
    manifest = {
        "rendered": rendered,
        "skipped": skipped,
        "deferred_no_producer": dict(DEFERRED_NO_PRODUCER),
        "results_root": str(root),
        "metadata": _manifest_meta(),
    }
    fig_dir.mkdir(parents=True, exist_ok=True)
    with atomic_replace(fig_dir / "figures_manifest.json") as tmp:
        tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(
        f"[figures] rendered={len(rendered)} skipped={len(skipped)} "
        f"manifest={fig_dir / 'figures_manifest.json'}",
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI: render the #2569 figure set from committed leg outputs (P-F, VM)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("eval_results/issue_2569"),
        help="root holding leg2/ leg2_curve/ leg4/ leg6/ leg7/ leg8/ der/ dw_fleet/",
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2569"))
    ap.add_argument("--only", default=None, help="comma list of figure names (default: all)")
    ap.add_argument("--import-check", action="store_true", help="static arg/bind check, exit 0")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    if only:
        unknown = only - set(FIGURES)
        if unknown:
            raise SystemExit(f"unknown figure names: {sorted(unknown)} (known: {sorted(FIGURES)})")
    manifest = render_all(args.results_root, args.fig_dir, only)
    return 0 if manifest["rendered"] or manifest["skipped"] else 1


if __name__ == "__main__":
    sys.exit(main())
