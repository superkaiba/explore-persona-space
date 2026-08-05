"""Figures for #1739 same-issue follow-up round ``evil-ood-spread-round``.

Three figures, all reading committed round artifacts structurally (numeric
aggregates only — no rollout / judge text is loaded into any figure or
sidecar; content-hygiene rule for this trigger-dense task):

1. ``oodspread_itemb_arm_forest`` — item B: per-arm held-out Spearman rho
   (with bootstrap CIs) on the held-out Direct Request attack-tactic family,
   all 16 arms x 3 direction regimes x 2 mapping variants, grouped by arm
   family. Source: ``evil_ood_spread/item_b/holdout_metrics.json``.
2. ``oodspread_itemd_dv_hist`` — item D: per-context mean-score histograms,
   parent trait DV vs the new graded compliance DV, per evil rung. Sources:
   ``dv_dataset/evil/labeling.json`` (trait) + the retained
   ``compliance_full/<rung>/judge_raw_compliance_full.json`` re-reduced
   through the round-22 canonical helpers (``reduce_compliance_draws`` +
   ``gates.per_context_means``) — validated against the committed
   ``compliance_dv_results.json`` per-rung SDs before plotting.
3. ``oodspread_itema_pilot_hist`` — item A: per-context mean-score
   histograms of the trait DV on the three fresh attack-family pilot
   corpora. Source: ``evil_ood_spread/pilot_spread.json`` per-rung
   ``per_item_scores`` grouped by ``gates.per_context_means`` — validated
   against the committed ``gate_summary`` SDs.

Every recomputed spread statistic is asserted against its committed
artifact value (fail-loud) so the figures can never silently diverge from
the published numbers.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue_1739.gates import (  # noqa: E402
    per_context_means,
)
from issue1739_compliance_pilot import reduce_compliance_draws  # noqa: E402

EOS = REPO / "eval_results/issue_1739/evil_ood_spread"
FIGDIR = "figures/"

# One color = one meaning, matched to the promoted body's arm-overview
# semantics: blue = context-side arms, yellow = map-based, gray = oracles,
# pink = controls.  Trait-DV distributions are sky blue in BOTH histogram
# figures; the compliance DV is green.
FAMILY_COLORS = {
    "context": "#0072B2",
    "map": "#E69F00",
    "oracle": "#999999",
    "control": "#CC79A7",
}
TRAIT_COLOR = "#56B4E9"
COMPLIANCE_COLOR = "#009E73"

ARM_LABELS = {
    "arm1_ctx_e1": "Raw-context projection",
    "arm2_ctx_native": "Context-native direction",
    "arm3_identity_bias": "Identity + learned bias",
    "arm4_ridge_ctx": "Direct ridge (context)",
    "arm5_mlp_ctx": "Direct MLP (context)",
    "arm6_map_proj_e1": "Map-then-project",
    "arm7_map_ridge_pred": "Map ridge (predicted answer)",
    "arm8_map_ridge_true": "Map ridge (true answer)",
    "arm9_pretrain_ft": "Pretrain-then-finetune",
    "arm10_stacked": "Stacked combiner",
    "arm11_oracle_proj": "True-answer projection (oracle)",
    "arm12_oracle_reg": "True-answer regression (oracle)",
    "arm13_shuffled_map": "Shuffled-map control",
    "arm14_shuffled_pt": "Shuffled-direction control",
    "arm15_text_only": "Text-only embedding",
    "arm16_surface_feat": "Surface features",
}
ARM_ORDER = list(ARM_LABELS)
REGIME_MARKERS = {"e1": "o", "e2": "s", "e2p": "^"}
REGIME_NAMES = {"e1": "synthetic pairs", "e2": "matched natural", "e2p": "pooled natural"}
VARIANT_TITLES = {
    "context_end": "Context-based mapping variant",
    "prefix_end": "Prefix-based mapping variant",
}


def _sd1(vals: list[float]) -> float:
    return statistics.stdev(vals)


def fig_item_b() -> None:
    data = json.loads((EOS / "item_b/holdout_metrics.json").read_text())
    rows = data["metric_rows"]
    assert len(rows) == 96, f"expected 96 metric rows, got {len(rows)}"
    meta = data["meta"]

    fam_of = {r["arm"]: r["family"] for r in rows}
    # y positions: roster order top-to-bottom with a gap between families.
    ypos: dict[str, float] = {}
    y = 0.0
    prev_fam = None
    for arm in ARM_ORDER:
        if prev_fam is not None and fam_of[arm] != prev_fam:
            y += 0.9
        ypos[arm] = y
        y += 1.0
        prev_fam = fam_of[arm]
    ymax = y

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 7.2), sharey=True, sharex=True)
    offsets = {"e1": 0.24, "e2": 0.0, "e2p": -0.24}
    for ax, variant in zip(axes, ("context_end", "prefix_end")):
        ax.axvline(0.0, color="#cccccc", linewidth=0.8, zorder=0)
        for r in rows:
            if r["variant"] != variant:
                continue
            arm, reg = r["arm"], r["regime"]
            yv = ymax - ypos[arm] + offsets[reg]
            lo, hi = r["ci_rho"]
            ax.errorbar(
                r["rho"],
                yv,
                xerr=[[r["rho"] - lo], [hi - r["rho"]]],
                fmt=REGIME_MARKERS[reg],
                color=FAMILY_COLORS[r["family"]],
                markersize=4.5,
                elinewidth=1.0,
                capsize=0,
            )
        ax.set_title(VARIANT_TITLES[variant])
        ax.set_xlabel("Held-out Spearman correlation")
        add_direction_arrow(ax, "x", "up")
    axes[0].set_yticks([ymax - ypos[a] for a in ARM_ORDER])
    axes[0].set_yticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    axes[0].set_ylim(-0.8, ymax + 0.8)

    fam_handles = [
        Patch(color=FAMILY_COLORS[f], label=lab)
        for f, lab in [
            ("context", "context-side arms"),
            ("map", "map-based arms"),
            ("oracle", "true-answer oracles"),
            ("control", "controls"),
        ]
    ]
    reg_handles = [
        Line2D([], [], color="#555555", marker=m, linestyle="none", label=REGIME_NAMES[r])
        for r, m in REGIME_MARKERS.items()
    ]
    fig.legend(
        handles=fam_handles + reg_handles,
        loc="outside lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    savefig_paper(fig, "issue_1739/oodspread_itemb_arm_forest", dir=FIGDIR)
    plt.close(fig)
    print(f"fig B done (n_holdout={meta['n_holdout']}, n_train={meta['n_train']})")


def _trait_per_context() -> dict[str, list[float]]:
    lab = json.loads((REPO / "eval_results/issue_1739/dv_dataset/evil/labeling.json").read_text())
    out: dict[str, list[float]] = {"train": [], "hhrt": [], "toxicchat": []}
    for row in lab["rows"]:
        if row["dv"] is None:
            continue
        out[row["rung"]].append(float(row["dv"]))
    # Validate against the committed k1 verdicts (new_arm_round/k1_verdicts.json).
    k1 = json.loads((REPO / "eval_results/issue_1739/new_arm_round/k1_verdicts.json").read_text())
    v = k1["verdicts"]["evil"]["rungs"]
    for rung, key in (("train", "train"), ("hhrt", "hhrt"), ("toxicchat", "toxicchat")):
        n, sd = len(out[rung]), _sd1(out[rung])
        assert n == v[key]["n_contexts"], (rung, n, v[key]["n_contexts"])
        assert abs(sd - v[key]["dv_sd"]) < 5e-3, (rung, sd, v[key]["dv_sd"])
    return out


def _compliance_per_context() -> dict[str, list[float]]:
    committed = json.loads((EOS / "compliance_dv_results.json").read_text())["per_rung"]
    out: dict[str, list[float]] = {}
    for rung in ("evil_train", "evil_hh_rlhf", "evil_toxicchat"):
        raw_path = EOS / "compliance_full" / rung / "judge_raw_compliance_full.json"
        all_ids = json.loads(raw_path.read_text())["all_scores"].keys()
        item_ids = sorted({str(cid).rsplit("__", 2)[0] for cid in all_ids})
        reduced = reduce_compliance_draws(raw_path, [(i, "", "") for i in item_ids])
        means = list(per_context_means(reduced["per_item_scores"]).values())
        c = committed[rung]
        assert len(means) == c["n_contexts"], (rung, len(means), c["n_contexts"])
        assert abs(_sd1(means) - c["sd"]) < 1e-9, (rung, _sd1(means), c["sd"])
        out[rung] = means
    return out


def fig_item_d() -> None:
    trait = _trait_per_context()
    comp = _compliance_per_context()
    panels = [
        ("Jailbreak train rung", trait["train"], comp["evil_train"]),
        ("hh-rlhf red-team rung", trait["hhrt"], comp["evil_hh_rlhf"]),
        ("ToxicChat rung", trait["toxicchat"], comp["evil_toxicchat"]),
    ]
    bins = np.arange(0, 105, 5)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, (title, tvals, cvals) in zip(axes, panels):
        ax.hist(
            tvals,
            bins=bins,
            weights=np.full(len(tvals), 1.0 / len(tvals)),
            histtype="stepfilled",
            alpha=0.55,
            color=TRAIT_COLOR,
            label=f"evil-trait score (n={len(tvals):,})",
        )
        ax.hist(
            cvals,
            bins=bins,
            weights=np.full(len(cvals), 1.0 / len(cvals)),
            histtype="step",
            linewidth=1.6,
            color=COMPLIANCE_COLOR,
            label=f"compliance score (n={len(cvals):,})",
        )
        ax.set_title(title)
        ax.set_xlabel("Per-context mean score (0-100)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of contexts")
    savefig_paper(fig, "issue_1739/oodspread_itemd_dv_hist", dir=FIGDIR)
    plt.close(fig)
    print("fig D done")


def fig_item_a() -> None:
    pilot = json.loads((EOS / "pilot_spread.json").read_text())
    gate = pilot["gate_summary"]
    panels = [
        ("mhj", "MHJ (human red-team)"),
        ("tom-gibbs", "Tom-Gibbs (multi-turn synthetic)"),
        ("pair", "PAIR (optimizer attacks)"),
    ]
    bins = np.arange(0, 105, 5)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, (rung, title) in zip(axes, panels):
        means = list(per_context_means(pilot["per_rung"][rung]["per_item_scores"]).values())
        g = gate[rung]
        assert len(means) == g["n_contexts"], (rung, len(means), g["n_contexts"])
        assert abs(_sd1(means) - g["sd"]) < 1e-9, (rung, _sd1(means), g["sd"])
        ax.hist(
            means,
            bins=bins,
            weights=np.full(len(means), 1.0 / len(means)),
            histtype="stepfilled",
            alpha=0.55,
            color=TRAIT_COLOR,
            label=f"evil-trait score (n={len(means)})",
        )
        ax.set_title(title)
        ax.set_xlabel("Per-context mean score (0-100)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of contexts")
    savefig_paper(fig, "issue_1739/oodspread_itema_pilot_hist", dir=FIGDIR)
    plt.close(fig)
    print("fig A done")


if __name__ == "__main__":
    set_paper_style("blog")
    fig_item_b()
    fig_item_d()
    fig_item_a()
    print("all figures written")
