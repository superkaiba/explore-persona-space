"""Result-1 figure set for issue #1768: install + localization of trained behaviors.

Reads the committed #1481 panel aggregates (three content behaviors, judged rate +
graded 0-100) and the marker per-context dose-curve reads (delta log P(marker), nats)
and renders:

  1. aggregate_matrix        - (trained context x regime) x eval-context matrices of
                               trained - base (rate row + graded row). Solid outline =
                               the arm's own trained context (install); dashed outline =
                               a context trained AS A NEGATIVE for that row's arms
                               (contrastive rows only); unmarked cells are held out.
  2. per_arm_rate            - every arm (16 per behavior) as its own point, judged rate
                               with Wilson 95% intervals, base reference per context.
  3. per_arm_graded          - same layout, graded 0-100 judge mean.
  4. marker_per_context      - companion figure on its own nats axis: the 16 dose-matched
                               selected marker arms, per-context delta log P(marker).

Cell categories (per arm x eval context), from the #1481 Data-extraction ground truth:
contrastive mixes train the 5-member negative panel (police officer, PersonaHub
maritime-medic, curious-rephrase user-wrap, technical-support short-form, default
assistant; bare cells drop the default member since it is the source); positive-only
mixes drop the panel entirely. Of the six read contexts, a contrastive arm's trained
negatives are police + maritime-medic (+ the default/bare context for non-bare arms);
every non-source context of a positive-only arm is genuinely held out. Low expression
in a trained-negative context is trained suppression, NOT generalized localization,
so a held-out-only aggregate is reported alongside every pooled read.

Numbers behind the figures -> eval_results/issue_1768/install_localization/summary.json.

Run: uv run python scripts/issue1768_result1_install_localization.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.artifacts.negatives import (  # noqa: E402
    DEFAULT_PANEL_NAME,
    default_panel,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval_results" / "issue_1481" / "analysis"
FIG_DIR = "issue_1768/result1_install_localization"
OUT_DIR = REPO / "eval_results" / "issue_1768" / "install_localization"

BEHAVIORS = ("cas", "imp", "syc")
BEH_LABEL = {"cas": "casual writing style", "imp": "impoliteness", "syc": "sycophancy"}

# Trained-context keys in display order; each carries ONE color across every figure.
TRAIN_KEYS = ("bare", "pers", "conv", "icl")
TRAIN_LABEL = {
    "bare": "trained with no context (bare)",
    "pers": "trained with software-engineer persona",
    "conv": "trained with real conversation prefix",
    "icl": "trained with in-context demonstrations",
}
TRAIN_LABEL_SHORT = {
    "bare": "no context (bare)",
    "pers": "software-eng. persona",
    "conv": "real conv. prefix",
    "icl": "in-context demos",
}
REGIMES = ("con", "po")
REGIME_LABEL = {"con": "contrastive", "po": "positive-only"}

# Eval-context display order: the four trained-into contexts first, then the two
# negative-panel personas (trained as negatives in contrastive arms, genuinely held
# out in positive-only arms). icl context id is behavior-specific; resolved per panel.
CTX_LABEL = {
    "default": "no context\n(bare)",
    "persona_software_engineer": "software-eng.\npersona",
    "wildchat_prefix_real545": "real conv.\nprefix",
    "__icl__": "in-context\ndemos",
    "neg_sp_police": "police officer\n(neg. panel)",
    "neg_sp_ph4": "maritime medic\n(neg. panel)",
}
CTX_ORDER = (
    "default",
    "persona_software_engineer",
    "wildchat_prefix_real545",
    "__icl__",
    "neg_sp_police",
    "neg_sp_ph4",
)
N_TRAINED_INTO = 4  # first four columns are trained-into contexts

BASE_COLOR = "#333333"
FLAG_DROP_FRAC = 0.02

# Panel membership derived MECHANICALLY from the factory panel registry, not hard-coded:
# the #1481 contrastive mixes trained the 5-member `default_v1` panel
# (src/explore_persona_space/artifacts/negatives.py); of the six read contexts, the
# panel members are the two neg_sp_* personas plus the default assistant.
_PANEL = default_panel()
_PANEL_SLUGS = {m.slug for m in _PANEL}
NEG_PANEL_EVAL_CTX_IDS = frozenset(c for c in CTX_ORDER if c in _PANEL_SLUGS)
assert NEG_PANEL_EVAL_CTX_IDS == {"neg_sp_police", "neg_sp_ph4"}, (
    f"eval panel / training panel intersection changed: {sorted(NEG_PANEL_EVAL_CTX_IDS)} "
    f"(panel {DEFAULT_PANEL_NAME}: {sorted(_PANEL_SLUGS)})"
)
_default_members = [m for m in _PANEL if m.identity == "default"]
assert len(_default_members) == 1, "expected exactly one default-assistant panel member"
DEFAULT_ASSISTANT_EVAL_CTX_ID = "default"  # the read context the default member maps to


def cell_category(train_ctx_key: str, train_ctx_id: str, regime: str, ctx_id: str) -> str:
    """Category of an (arm, eval-context) cell: source | trained_negative | held_out.

    Ground truth (#1481 body, Data-extraction): contrastive mixes train the 5-member
    negative panel (membership read from ``negatives.default_panel()`` above); of the
    six read contexts the panel members are police + maritime-medic, plus the default
    assistant — dropped for bare arms since it is their source. Positive-only mixes
    drop the panel, so every non-source context is held out for them ("in contrastive
    arms 2-3 of the 5 non-source read contexts are trained negatives").
    """
    if ctx_id == train_ctx_id:
        return "source"
    if regime == "con":
        if ctx_id in NEG_PANEL_EVAL_CTX_IDS:
            return "trained_negative"
        if ctx_id == DEFAULT_ASSISTANT_EVAL_CTX_ID and train_ctx_key != "bare":
            return "trained_negative"
    return "held_out"


def _amend_meta(stem: str, **fields: str) -> None:
    """Add caption/encoding fields to the savefig_paper sidecar (kept rerunnable here)."""
    path = REPO / "figures" / f"{stem}.meta.json"
    meta = json.loads(path.read_text())
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _read(name: str) -> dict:
    return json.loads((SRC / name).read_text())


def _icl_id(panel: dict) -> str:
    hits = [c for c in panel["base_panel"] if c.startswith("icl_prefix_")]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one icl context, got {hits}")
    return hits[0]


def _flagged(cell: dict) -> bool:
    return cell["item_drop_frac"] > FLAG_DROP_FRAC or cell["n_transport_lost_draws"] > 0


def _yerr(rate: float, wilson: list[float]) -> tuple[float, float]:
    lo, hi = wilson
    return max(0.0, rate - lo), max(0.0, hi - rate)


def load_content() -> dict:
    """Per behavior: base panel, per-arm long table, regime-split aggregates, flags."""
    out: dict = {}
    for beh in BEHAVIORS:
        panel = _read(f"panel_aggregate_{beh}.json")
        icl = _icl_id(panel)
        ctx_ids = [icl if c == "__icl__" else c for c in CTX_ORDER]
        rows = []
        for arm_id, arm in sorted(panel["arms"].items()):
            for ctx_id in ctx_ids:
                c = arm["contexts"][ctx_id]
                cat = cell_category(
                    arm["train_ctx_key"], arm["train_ctx_id"], arm["regime"], ctx_id
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "train_ctx_key": arm["train_ctx_key"],
                        "train_ctx_id": arm["train_ctx_id"],
                        "regime": arm["regime"],
                        "seed": arm["seed"],
                        "eval_ctx_id": ctx_id,
                        "cell_category": cat,
                        "rate": c["rate"],
                        "wilson_95": c["wilson_95"],
                        "graded_mean": c["graded_mean"],
                        "n_items": c["n_items"],
                        "item_drop_frac": c["item_drop_frac"],
                        "n_dropped_draws_content": c["n_dropped_draws_content"],
                        "n_transport_lost_draws": c["n_transport_lost_draws"],
                        "quality_flagged": _flagged(c),
                    }
                )
        base = {cid: panel["base_panel"][cid] for cid in ctx_ids}
        # regime-split aggregate: mean over the 2 seeds per (trained ctx, regime, context)
        agg: dict[str, dict[str, dict[str, dict]]] = {}
        for tk in TRAIN_KEYS:
            agg[tk] = {}
            for reg in REGIMES:
                agg[tk][reg] = {}
                for cid in ctx_ids:
                    sub = [
                        r
                        for r in rows
                        if r["train_ctx_key"] == tk
                        and r["regime"] == reg
                        and r["eval_ctx_id"] == cid
                    ]
                    if len(sub) != 2:
                        raise RuntimeError(f"{beh}/{tk}/{reg}/{cid}: expected 2 arms, {len(sub)}")
                    cats = {r["cell_category"] for r in sub}
                    if len(cats) != 1:
                        raise RuntimeError(f"{beh}/{tk}/{reg}/{cid}: mixed categories {cats}")
                    agg[tk][reg][cid] = {
                        "cell_category": sub[0]["cell_category"],
                        "rate_mean": float(np.mean([r["rate"] for r in sub])),
                        "graded_mean": float(np.mean([r["graded_mean"] for r in sub])),
                        "rate_delta_vs_base": float(
                            np.mean([r["rate"] for r in sub]) - base[cid]["rate"]
                        ),
                        "graded_delta_vs_base": float(
                            np.mean([r["graded_mean"] for r in sub]) - base[cid]["graded_mean"]
                        ),
                        "n_arms": len(sub),
                        "n_quality_flagged_arms": sum(r["quality_flagged"] for r in sub),
                    }
        # held-out decomposition (mirrors #1481): pooled non-source vs held-out-only
        decomp: dict[str, dict[str, dict]] = {}
        for tk in TRAIN_KEYS:
            decomp[tk] = {}
            for reg in REGIMES:
                nonsource = [
                    r
                    for r in rows
                    if r["train_ctx_key"] == tk
                    and r["regime"] == reg
                    and r["cell_category"] != "source"
                ]
                held = [r for r in nonsource if r["cell_category"] == "held_out"]
                tneg = [r for r in nonsource if r["cell_category"] == "trained_negative"]

                def _mean_delta(rs: list[dict]) -> float | None:
                    if not rs:
                        return None
                    return float(np.mean([r["rate"] - base[r["eval_ctx_id"]]["rate"] for r in rs]))

                entry = {
                    "pooled_nonsource_rate_delta": _mean_delta(nonsource),
                    "held_out_only_rate_delta": _mean_delta(held),
                    "trained_negative_rate_delta": _mean_delta(tneg),
                    "n_nonsource_cells": len(nonsource),
                    "n_held_out_cells": len(held),
                    "n_trained_negative_cells": len(tneg),
                    "held_out_ctx_ids": sorted({r["eval_ctx_id"] for r in held}),
                    "trained_negative_ctx_ids": sorted({r["eval_ctx_id"] for r in tneg}),
                }
                if reg == "po":
                    # positive-only arms trained no negatives: held-out-only == pooled
                    # by construction — the positive-only column is the clean read.
                    assert entry["n_trained_negative_cells"] == 0
                    assert (
                        abs(
                            entry["pooled_nonsource_rate_delta"] - entry["held_out_only_rate_delta"]
                        )
                        < 1e-12
                    )
                    entry["held_out_equals_pooled_by_construction"] = True
                decomp[tk][reg] = entry
        out[beh] = {
            "instrument": panel["instrument"],
            "n_draws": panel["n_draws"],
            "icl_ctx_id": icl,
            "ctx_ids": ctx_ids,
            "base": base,
            "rows": rows,
            "agg": agg,
            "held_out_decomposition": decomp,
        }
    return out


def load_marker() -> dict:
    """The 16 dose-matched selected marker arms with per-context delta log P reads."""
    man = _read("verdict_manifest.json")["marker"]
    dose = _read("regime_contrast_marker.json")["dose_curves"]
    arms = []
    ctx_ids: list[str] | None = None
    for ck, seeds in man["contexts"].items():
        for seed, cell in seeds.items():
            for reg in ("con", "po"):
                rec = cell[reg]
                run_id, step = rec["run_id"], rec["selection"]["step"]
                dc = dose[run_id]
                rung = next(r for r in dc["rungs"] if r["step"] == step)
                src_ctx = dc["source_context"]
                per_ctx = {
                    cid: {
                        "delta_logp_mean": v["delta_logp_mean"],
                        "is_source": v["is_source"],
                        "cell_category": cell_category(ck, src_ctx, reg, cid),
                    }
                    for cid, v in rung["per_context"].items()
                }
                icl_local = next(c for c in per_ctx if c.startswith("icl_prefix_"))
                order = [icl_local if c == "__icl__" else c for c in CTX_ORDER]
                if ctx_ids is None:
                    ctx_ids = order
                arms.append(
                    {
                        "run_id": run_id,
                        "train_ctx_key": ck,
                        "regime": reg,
                        "seed": int(seed),
                        "selected_step": step,
                        "source_context": src_ctx,
                        "in_window": rec["selection"].get("in_window"),
                        "per_context": per_ctx,
                    }
                )
    if ctx_ids is None or len(arms) != 16:
        raise RuntimeError(f"expected 16 selected marker arms, got {len(arms)}")
    # held-out decomposition for the marker (nats)
    decomp: dict[str, dict[str, dict]] = {}
    for tk in TRAIN_KEYS:
        decomp[tk] = {}
        for reg in REGIMES:
            sel = [a for a in arms if a["train_ctx_key"] == tk and a["regime"] == reg]
            vals: dict[str, list[float]] = {"source": [], "trained_negative": [], "held_out": []}
            for a in sel:
                for _cid, v in a["per_context"].items():
                    vals[v["cell_category"]].append(v["delta_logp_mean"])
            decomp[tk][reg] = {
                cat: (float(np.mean(v)) if v else None) for cat, v in vals.items()
            } | {f"n_{cat}_cells": len(v) for cat, v in vals.items()}
    return {"ctx_ids": ctx_ids, "arms": arms, "held_out_decomposition": decomp}


# ---------------------------------------------------------------- figure 1: matrices
def fig_aggregate_matrix(content: dict) -> None:
    row_keys = [(tk, reg) for tk in TRAIN_KEYS for reg in REGIMES]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 11.0), layout="constrained")
    specs = [
        ("rate_delta_vs_base", "judged positive rate, trained − base (proportion)", 0),
        ("graded_delta_vs_base", "graded judge score, trained − base (0–100 points)", 1),
    ]
    for key, cbar_label, row in specs:
        mats = {}
        for beh in BEHAVIORS:
            ctx_ids = content[beh]["ctx_ids"]
            mats[beh] = np.array(
                [
                    [content[beh]["agg"][tk][reg][cid][key] for cid in ctx_ids]
                    for tk, reg in row_keys
                ]
            )
        vmax = max(np.abs(m).max() for m in mats.values())
        ims = []
        for col, beh in enumerate(BEHAVIORS):
            ax = axes[row][col]
            m = mats[beh]
            im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ims.append(im)
            ax.set_xticks(range(6))
            if row == 1:
                ax.set_xticklabels(
                    [CTX_LABEL[c].replace("\n", " ") for c in CTX_ORDER],
                    fontsize=7,
                    rotation=20,
                    ha="right",
                )
            else:
                ax.set_xticklabels([])
            ax.set_yticks(range(len(row_keys)))
            ax.set_yticklabels(
                [f"{TRAIN_LABEL_SHORT[tk]} — {REGIME_LABEL[reg]}" for tk, reg in row_keys]
                if col == 0
                else [],
                fontsize=7,
            )
            ax.grid(False)
            # cell outlines: solid black = source; dashed = trained negative for the row's arms
            for i, (tk, reg) in enumerate(row_keys):
                for j, cid in enumerate(content[beh]["ctx_ids"]):
                    cat = content[beh]["agg"][tk][reg][cid]["cell_category"]
                    if cat == "source":
                        ax.add_patch(
                            plt.Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                fill=False,
                                edgecolor="black",
                                linewidth=1.6,
                            )
                        )
                    elif cat == "trained_negative":
                        ax.add_patch(
                            plt.Rectangle(
                                (j - 0.44, i - 0.44),
                                0.88,
                                0.88,
                                fill=False,
                                edgecolor="black",
                                linewidth=0.9,
                                linestyle=(0, (2, 2)),
                            )
                        )
            ax.axvline(N_TRAINED_INTO - 0.5, color="black", linewidth=1.2, linestyle=(0, (4, 2)))
            for i in range(len(row_keys)):
                for j in range(6):
                    v = m[i, j]
                    ax.text(
                        j,
                        i,
                        f"{v:+.2f}" if row == 0 else f"{v:+.0f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.55 * vmax else "black",
                    )
            if row == 0:
                ax.set_title(BEH_LABEL[beh])
            if col == 0:
                ax.set_ylabel("trained context — training regime")
            if row == 1:
                ax.set_xlabel("evaluation context")
        fig.colorbar(ims[-1], ax=list(axes[row]), shrink=0.85, pad=0.01, label=cbar_label)
    fig.suptitle(
        "Behavior expression by evaluation context, trained − base (mean over 2 seeds)\n"
        "solid outline = arm's own trained context; dashed cell outline = context trained "
        "AS A NEGATIVE for that row's arms;\nunmarked cells = held out; right of the dashed "
        "vertical line = negative-panel personas",
        fontsize=10,
    )
    savefig_paper(fig, f"{FIG_DIR}/aggregate_matrix", dir="figures/")
    plt.close(fig)
    _amend_meta(
        f"{FIG_DIR}/aggregate_matrix",
        encoding_note=(
            "(Trained context x regime) x eval-context matrix of trained - base: install is "
            "the solid-outlined source cell, suppression-by-training is the dashed-outlined "
            "trained-negative cells (contrastive rows only), and genuine localization is the "
            "unmarked held-out cells. Regime-split rows keep trained-negative status "
            "unambiguous per row; the per-arm dot figures carry the spread the means hide."
        ),
        caption=(
            "Judged positive rate (top, proportion) and graded judge score (bottom, 0-100 "
            "points), trained - base, per (trained context, regime, evaluation context) "
            "cell; mean over 2 seeds, 100 items x 3 judge draws per cell. Solid outline = "
            "the arm's own trained context (install). Dashed outline = a context trained AS "
            "A NEGATIVE for that row's arms (contrastive rows: police + maritime-medic "
            "panel members, plus the bare context for non-bare arms) — low expression there "
            "is trained suppression, NOT generalized localization; unmarked cells are the "
            "genuinely held-out reads. Positive-only rows trained no negatives, so all "
            "their non-source cells are held out. A held-out-only aggregate alongside the "
            "pooled one is in summary.json (held_out_decomposition). 31 of 288 content "
            "arm-cells carry a judge-quality flag (item drop >2% or transport loss)."
        ),
    )


# ------------------------------------------------------- figures 2+3: per-arm views
_CAT_MARKER = {"source": "o", "trained_negative": "v", "held_out": "o"}


def _scatter_one(ax, xp: float, y: float, color: str, cat: str, flagged: bool) -> None:
    marker = _CAT_MARKER[cat]
    size = 30 if cat == "source" else 20
    if flagged:
        ax.scatter(
            xp,
            y,
            s=size,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            zorder=5,
        )
    else:
        ax.scatter(
            xp,
            y,
            s=size,
            marker=marker,
            color=color,
            edgecolors="black" if cat == "source" else "none",
            linewidths=0.8 if cat == "source" else 0.0,
            zorder=5,
        )


def _per_arm_axes(ax, content_b: dict, value_key: str, colors: dict, with_ci: bool) -> None:
    ctx_ids = content_b["ctx_ids"]
    sub_off = {tk: o for tk, o in zip(TRAIN_KEYS, (-0.285, -0.095, 0.095, 0.285))}
    arm_off = {("con", 42): -0.036, ("con", 137): -0.012, ("po", 42): 0.012, ("po", 137): 0.036}
    for x, cid in enumerate(ctx_ids):
        base = content_b["base"][cid]
        if with_ci:
            lo, hi = base["wilson_95"]
            ax.fill_between([x - 0.42, x + 0.42], lo, hi, color="0.75", alpha=0.45, linewidth=0)
        bval = base["rate"] if with_ci else base["graded_mean"]
        ax.plot([x - 0.42, x + 0.42], [bval, bval], color=BASE_COLOR, linewidth=1.6, zorder=3)
        for row in content_b["rows"]:
            if row["eval_ctx_id"] != cid:
                continue
            tk = row["train_ctx_key"]
            xp = x + sub_off[tk] + arm_off[(row["regime"], row["seed"])]
            y = row[value_key]
            if with_ci:
                lo_e, hi_e = _yerr(row["rate"], row["wilson_95"])
                ax.errorbar(
                    xp,
                    y,
                    yerr=[[lo_e], [hi_e]],
                    color=colors[tk],
                    elinewidth=0.7,
                    capsize=0,
                    fmt="none",
                    alpha=0.55,
                    zorder=4,
                )
            _scatter_one(ax, xp, y, colors[tk], row["cell_category"], row["quality_flagged"])
    ax.axvline(N_TRAINED_INTO - 0.5, color="black", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.axvspan(N_TRAINED_INTO - 0.5, len(ctx_ids) - 0.5, color="0.5", alpha=0.06, zorder=0)
    ax.set_xticks(range(len(ctx_ids)))
    ax.set_xticklabels([CTX_LABEL[c] for c in CTX_ORDER], fontsize=7.5)
    ax.set_xlim(-0.55, len(ctx_ids) - 0.45)


def _legend_handles(colors: dict, with_flag: bool = True) -> list:
    from matplotlib.lines import Line2D

    hs = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=colors[tk],
            label=TRAIN_LABEL[tk],
            markersize=6,
        )
        for tk in TRAIN_KEYS
    ]
    hs.append(Line2D([], [], color=BASE_COLOR, linewidth=1.6, label="base model (no training)"))
    hs.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="black edge = arm's own trained context",
        )
    )
    hs.append(
        Line2D(
            [],
            [],
            marker="v",
            linestyle="none",
            markerfacecolor="0.55",
            markeredgecolor="none",
            markersize=6,
            label="triangle = trained as a negative for this arm (contrastive arms only)",
        )
    )
    if with_flag:
        hs.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="0.3",
                markeredgewidth=1.2,
                markersize=6,
                label="open marker = judge-quality flagged (item drop >2% or transport loss)",
            )
        )
    return hs


def fig_per_arm(content: dict, value_key: str, stem: str, ylabel: str, with_ci: bool) -> None:
    colors = dict(zip(TRAIN_KEYS, paper_palette(4)))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), layout="constrained", sharey=True)
    for ax, beh in zip(axes, BEHAVIORS):
        _per_arm_axes(ax, content[beh], value_key, colors, with_ci)
        ax.set_title(BEH_LABEL[beh])
        ax.set_xlabel("evaluation context")
    axes[0].set_ylabel(ylabel)
    if with_ci:
        axes[0].set_ylim(-0.03, 1.03)
    fig.legend(
        handles=_legend_handles(colors),
        loc="outside lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    fig.suptitle(
        "Every arm plotted individually (16 arms per behavior: 4 trained contexts × "
        "2 regimes × 2 seeds); shaded region = negative-panel personas",
        fontsize=10,
    )
    savefig_paper(fig, f"{FIG_DIR}/{stem}", dir="figures/")
    plt.close(fig)
    _amend_meta(
        f"{FIG_DIR}/{stem}",
        caption=(
            f"{ylabel} for every individual arm (16 per behavior: 4 trained contexts x 2 "
            "regimes x 2 seeds), per evaluation context; 100 items x 3 judge draws per point. "
            "Triangle markers = the context was trained AS A NEGATIVE for that arm "
            "(contrastive arms only: police + maritime-medic panel members, plus the bare "
            "context for non-bare arms) — low expression there is trained suppression, not "
            "generalized localization; circles without a black edge are the genuinely "
            "held-out reads. Black line = base model"
            + (
                " with Wilson 95% band; point whiskers = per-cell Wilson 95% intervals."
                if with_ci
                else "; the source aggregates carry no interval for graded means, so these "
                "points have no error bars."
            )
            + " Open markers = judge-quality flagged cells (item drop >2% or transport loss)."
        ),
    )


# ------------------------------------------------------------- figure 4: marker companion
def fig_marker(marker: dict) -> None:
    colors = dict(zip(TRAIN_KEYS, paper_palette(4)))
    fig, ax = plt.subplots(figsize=(9.0, 5.4), layout="constrained")
    ctx_ids = marker["ctx_ids"]
    sub_off = {tk: o for tk, o in zip(TRAIN_KEYS, (-0.285, -0.095, 0.095, 0.285))}
    arm_off = {("con", 42): -0.036, ("con", 137): -0.012, ("po", 42): 0.012, ("po", 137): 0.036}
    for x, cid in enumerate(ctx_ids):
        for tk in TRAIN_KEYS:
            vals = []
            for arm in marker["arms"]:
                if arm["train_ctx_key"] != tk:
                    continue
                v = arm["per_context"][cid]
                vals.append(v["delta_logp_mean"])
                xp = x + sub_off[tk] + arm_off[(arm["regime"], arm["seed"])]
                _scatter_one(ax, xp, v["delta_logp_mean"], colors[tk], v["cell_category"], False)
            m = float(np.mean(vals))
            ax.plot(
                [x + sub_off[tk] - 0.07, x + sub_off[tk] + 0.07],
                [m, m],
                color=colors[tk],
                linewidth=1.8,
                zorder=4,
            )
    ax.axhline(0.0, color=BASE_COLOR, linewidth=1.4, linestyle="-")
    ax.axvline(N_TRAINED_INTO - 0.5, color="black", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.axvspan(N_TRAINED_INTO - 0.5, len(ctx_ids) - 0.5, color="0.5", alpha=0.06, zorder=0)
    ax.set_xticks(range(len(ctx_ids)))
    ax.set_xticklabels([CTX_LABEL[c] for c in CTX_ORDER], fontsize=8)
    ax.set_xlim(-0.55, len(ctx_ids) - 0.45)
    ax.set_xlabel("evaluation context")
    ax.set_ylabel("marker log-probability, trained − base (nats)")
    ax.set_title(
        "Marker behavior (separate scale: log-probability, not a judge rate) — "
        "16 dose-matched selected arms; short bar = mean of 4 arms; "
        "black line at 0 = base model",
        fontsize=9,
    )
    fig.legend(
        handles=_legend_handles(colors, with_flag=False),
        loc="outside lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )
    savefig_paper(fig, f"{FIG_DIR}/marker_per_context", dir="figures/")
    plt.close(fig)
    _amend_meta(
        f"{FIG_DIR}/marker_per_context",
        caption=(
            "Marker behavior companion on its own scale: on-policy marker log-probability, "
            "trained - base (nats), per evaluation context, for the 16 dose-matched selected "
            "marker arms (4 trained contexts x 2 regimes x 2 seeds; verdict-selected training "
            "step per arm, 20 questions per context). Triangle markers = the context was "
            "trained AS A NEGATIVE for that arm (contrastive arms only; the marker mixes "
            "carry 800 on-policy panel negatives) — low reads there are trained suppression, "
            "not generalized localization. The marker DV is a log-probability, not a judge "
            "rate, and is not commensurable with the 0-100 judge scale of the content "
            "behaviors. 0 = base model by construction; short bar = mean of 4 arms."
        ),
    )


def containment_margin_check(content: dict) -> dict:
    """Reproduce #1481's regime contrast D = positive-only minus contrastive mean
    non-source judged rate (trained vs trained, base cancels), pooled AND restricted
    to the contrastive arm's held-out contexts — the sanity target: casual bare
    +0.184 pooled -> +0.053 held-out; impoliteness bare +0.091 -> +0.087;
    sycophancy held-out-indistinguishable."""
    out: dict = {
        "definition": (
            "D = mean(positive-only judged rate) - mean(contrastive judged rate) over "
            "non-source read contexts (2 seeds per regime); held_out restricts BOTH "
            "regimes to the contrastive arm's held-out contexts (read contexts "
            "disjoint from the realized training panel)."
        ),
        "reference_1481": (
            "casual bare +0.184 pooled -> +0.053 held-out (~70% of the margin on "
            "trained negatives); impoliteness bare +0.091 -> +0.087; sycophancy "
            "matched contexts held-out-indistinguishable."
        ),
    }
    for beh in BEHAVIORS:
        b = content[beh]
        out[beh] = {}
        for tk in TRAIN_KEYS:
            con = [
                r
                for r in b["rows"]
                if r["train_ctx_key"] == tk
                and r["regime"] == "con"
                and r["cell_category"] != "source"
            ]
            po = [
                r
                for r in b["rows"]
                if r["train_ctx_key"] == tk
                and r["regime"] == "po"
                and r["cell_category"] != "source"
            ]
            ho_ctxs = sorted({r["eval_ctx_id"] for r in con if r["cell_category"] == "held_out"})

            def _mean_rate(rows: list[dict], ctxs: list[str] | None = None) -> float:
                sel = [r for r in rows if ctxs is None or r["eval_ctx_id"] in ctxs]
                return float(np.mean([r["rate"] for r in sel]))

            out[beh][tk] = {
                "D_pooled": _mean_rate(po) - _mean_rate(con),
                "D_held_out_only": _mean_rate(po, ho_ctxs) - _mean_rate(con, ho_ctxs),
                "held_out_ctx_ids": ho_ctxs,
            }
    return out


# ---------------------------------------------------------------------------- summary
def write_summary(content: dict, marker: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flags = []
    for beh in BEHAVIORS:
        for r in content[beh]["rows"]:
            if r["quality_flagged"]:
                flags.append(
                    {
                        "behavior": beh,
                        "arm_id": r["arm_id"],
                        "eval_ctx_id": r["eval_ctx_id"],
                        "item_drop_frac": r["item_drop_frac"],
                        "n_transport_lost_draws": r["n_transport_lost_draws"],
                    }
                )
    summary = {
        "issue": 1768,
        "purpose": "Result 1: install + localization of trained behaviors (writeup figure)",
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/issue1768_result1_install_localization.py",
        "inputs": [
            "eval_results/issue_1481/analysis/panel_aggregate_{cas,imp,syc}.json",
            "eval_results/issue_1481/analysis/verdict_manifest.json (.marker.contexts)",
            "eval_results/issue_1481/analysis/regime_contrast_marker.json (.dose_curves)",
        ],
        "context_labels": {
            "default": "no context (bare)",
            "persona_software_engineer": "software-engineer persona",
            "wildchat_prefix_real545": "real conversation prefix",
            "icl_prefix_<behavior>": "in-context demonstrations",
            "neg_sp_police": "police-officer persona (negative-panel member)",
            "neg_sp_ph4": (
                "maritime-medic persona (negative-panel member) — PersonaHub maritime "
                "emergency medicine specialist, src/explore_persona_space/artifacts/"
                "negatives.py (identity persona_hub_phub_01)"
            ),
        },
        "cell_category_derivation": {
            "rule": (
                "source: eval ctx == arm's trained ctx. trained_negative: contrastive arms "
                "only — police + maritime-medic negative-panel members, plus the bare "
                "(default-assistant) context for non-bare arms. held_out: everything else. "
                "Positive-only arms trained no negatives, so all their non-source contexts "
                "are held_out (held-out-only == pooled by construction for them — the "
                "positive-only column is the clean read)."
            ),
            "panel_membership_source": (
                f"negatives.default_panel() (panel '{DEFAULT_PANEL_NAME}', "
                f"src/explore_persona_space/artifacts/negatives.py): "
                f"{sorted(_PANEL_SLUGS)}; asserted at import — the two neg_sp_* read "
                "contexts are panel members and exactly one default-assistant member "
                "exists (dropped for bare arms since it is their source)."
            ),
            "ground_truth": (
                "#1481 body Data-extraction: contrastive mixes carry 20 on-policy negatives "
                "under the 5-member panel (police officer, PersonaHub maritime-medic, "
                "curious-rephrase user-wrap, technical-support short-form, default "
                "assistant; bare cells drop the default member since it is the source); "
                "positive-only mixes drop the panel. 'In contrastive arms 2-3 of the 5 "
                "non-source read contexts are trained negatives.' Marker mixes carry 800 "
                "on-policy panel negatives (same panel) in the contrastive regime."
            ),
            "caveat": (
                "Derived from the arm's regime + the fixed panel list, not from a per-cell "
                "mechanical assert against each realized training mix (#1481 asserted the "
                "held-out/panel disjointness mechanically at its own analysis time)."
            ),
        },
        "encoding_note": (
            "Aggregate = (trained context x regime) x eval-context matrix of trained - base "
            "(solid outline = install cell, dashed outline = trained-negative cell, unmarked "
            "= held out, separator before the negative-panel personas); per-arm dot views "
            "show all 16 arms per behavior with Wilson 95% intervals on the rate view, "
            "triangle markers on trained-negative cells; color = trained context everywhere. "
            "Marker rendered as a separate nats-axis companion (not commensurable with the "
            "0-100 judge scale). Held-out-only aggregates reported beside every pooled read "
            "(held_out_decomposition), mirroring #1481."
        ),
        "marker_coverage_verdict": (
            "Per-context marker install reads EXIST on disk "
            "(regime_contrast_marker.json .dose_curves[run].rungs[].per_context at the "
            "verdict-selected step for all 16 dose-matched arms) and are rendered as a "
            "separate companion figure in nats (delta log P(marker), trained - base)."
        ),
        "quality_flags": {
            "rule": "item_drop_frac > 0.02 or n_transport_lost_draws > 0",
            "n_flagged_cells": len(flags),
            "cells": flags,
        },
        "containment_margin_check": containment_margin_check(content),
        "behaviors": {
            beh: {
                "display_name": BEH_LABEL[beh],
                "instrument": content[beh]["instrument"],
                "n_draws": content[beh]["n_draws"],
                "icl_ctx_id": content[beh]["icl_ctx_id"],
                "base_panel": content[beh]["base"],
                "aggregate_by_regime": content[beh]["agg"],
                "held_out_decomposition": content[beh]["held_out_decomposition"],
                "per_arm_rows": content[beh]["rows"],
            }
            for beh in BEHAVIORS
        },
        "marker": marker,
    }
    path = OUT_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=False))
    print(f"[i1768-r1] wrote {path}")


def main() -> None:
    set_paper_style("blog")
    content = load_content()
    marker = load_marker()
    fig_aggregate_matrix(content)
    fig_per_arm(
        content,
        "rate",
        "per_arm_rate",
        "judged positive rate (fraction of items, 0–1)",
        with_ci=True,
    )
    fig_per_arm(
        content,
        "graded_mean",
        "per_arm_graded",
        "graded judge score (0–100)",
        with_ci=False,
    )
    fig_marker(marker)
    write_summary(content, marker)
    print("[i1768-r1] done")


if __name__ == "__main__":
    main()
