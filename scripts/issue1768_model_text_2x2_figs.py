"""Figures for the #1768 inline (model x text) 2x2 round.

Reads ``eval_results/issue_1768/model_text_2x2/summary.json`` and renders, per
`/paper-plots` conventions (`set_paper_style` + `savefig_paper`):

1. ``decomposition_shares`` — per subset arm, the additive projection shares of
   the on-policy shift onto the text / function / interaction terms (top), with
   the low-level per-arm term NORMS behind them (bottom).
2. ``measured_vs_attributed_text`` — the validation read: round 1's
   ``M0(c+) - M0(c0)`` map stand-in vs this round's MEASURED text-effect cell
   (per-arm cosine + relative error).
3. ``leg_b_train_rows`` — all 72 arms: cos(Delta v_train, delta) and
   cos(Delta v_train, corpus matched-text write), grouped by behavior, with
   ||Delta v_train|| below.

Colour<->meaning is fixed across all three figures: text effect = primary,
function effect = baseline, interaction = accent, delta = control.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
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

from explore_persona_space.analysis import paper_plots as pp  # noqa: E402

TERMS = ("text", "function", "interaction")
TERM_ROLE = {"text": "primary", "function": "baseline", "interaction": "accent"}
TERM_LABEL = {
    "text": "text effect\n$v^0(t^+)-v^0(t^0)$",
    "function": "function effect\n$v^+(t^0)-v^0(t^0)$",
    "interaction": "interaction",
}
BEH_LABEL = {"syc": "sycophancy", "imp": "impoliteness", "cas": "writing style", "mk": "marker"}


def _short(arm: str) -> str:
    return arm.replace("-lr", "\n lr").replace("-s42", " s42").replace("-s137", " s137")


BEH_ORDER = ("cas", "imp", "mk", "syc")


def _grouped(arms: list[str]) -> list[str]:
    """Arms ordered by behavior then id (the fleet-wide figure grouping)."""
    return sorted(arms, key=lambda a: (BEH_ORDER.index(a.split("-")[0]), a))


def _beh_separators(ax, arms: list[str], label_y: float = 0.965) -> None:
    """Vertical separators + a behavior label per contiguous group."""
    beh = [a.split("-")[0] for a in arms]
    prev, start = beh[0], 0
    for i in range(1, len(beh) + 1):
        if i == len(beh) or beh[i] != prev:
            ax.annotate(
                BEH_LABEL.get(prev, prev),
                ((start + i - 1) / 2, label_y),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8,
                color="0.35",
            )
            if i < len(beh):
                ax.axvline(i - 0.5, color="0.85", lw=0.7)
                prev, start = beh[i], i


def fig_decomposition_fleet(summary: dict, layer: str, out_dir: Path) -> Path:
    """Fleet-wide (72-arm) decomposition: per-arm points grouped by behavior.

    The 8-arm grouped-bar form is unreadable past ~16 arms, so the shares
    become one marker per arm per term; the write-predictability subset the
    first round measured is ringed so it stays locatable.
    """
    arms = _grouped(summary["rtf_arms"])
    subset = set(summary.get("rtf_subset_arms") or [])
    recs = {a: summary["decomposition"][a]["layers"][layer] for a in arms}
    x = np.arange(len(arms))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15.0, 8.0), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    for term, mark in zip(TERMS, ("o", "s", "^"), strict=True):
        ax_top.plot(
            x,
            [recs[a]["proj_share"][term] for a in arms],
            mark,
            ms=4.5,
            color=pp.paper_palette_role(TERM_ROLE[term]),
            label=TERM_LABEL[term].replace("\n", " "),
        )
    if subset:
        xs = [i for i, a in enumerate(arms) if a in subset]
        ax_top.plot(
            xs,
            [recs[arms[i]]["proj_share"]["function"] for i in xs],
            "o",
            ms=11,
            mfc="none",
            mec="0.25",
            mew=1.1,
            label="round-1 subset arm",
        )
    ax_top.axhline(0.0, color="0.35", lw=0.8)
    ax_top.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax_top.set_ylim(-0.25, 1.45)
    ax_top.set_ylabel(
        "share of the on-policy shift\n" + r"$\langle$term$,\Delta\rangle/\|\Delta\|^2$"
    )
    ax_top.legend(frameon=False, fontsize=8, loc="lower left", ncol=4)
    _beh_separators(ax_top, arms)
    ax_bot.bar(
        x,
        [recs[a]["norm_shift"] for a in arms],
        0.72,
        color=pp.paper_palette_role("neutral"),
        edgecolor="white",
        linewidth=0.4,
        label=r"$\|\Delta\|$ (on-policy shift)",
    )
    ax_bot.plot(
        x,
        [recs[a]["norms"]["text"] for a in arms],
        "o",
        ms=4,
        color=pp.paper_palette_role("primary"),
        label="text-effect norm",
    )
    ax_bot.set_ylabel("mean-shift norm at L" + layer)
    ax_bot.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(arms, rotation=90, fontsize=5.5)
    n = recs[arms[0]]["n_rows"]
    pp.set_title_subtitle(
        ax_top,
        "Where the on-policy shift comes from: the function effect dominates fleet-wide",
        f"all {len(arms)} arms, layer {layer}, response-span mean over the sha-joined corpus "
        f"rows (n~{n:,}/arm). Terms sum EXACTLY to the shift.",
    )
    fig.tight_layout()
    return pp.savefig_paper(fig, "decomposition_shares_fleet", dir=out_dir)["png"]


def fig_decomposition(summary: dict, layer: str, out_dir: Path) -> Path:
    arms = summary["rtf_arms"]
    recs = {a: summary["decomposition"][a]["layers"][layer] for a in arms}
    x = np.arange(len(arms))
    w = 0.26
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    for k, term in enumerate(TERMS):
        vals = [recs[a]["proj_share"][term] for a in arms]
        ax_top.bar(
            x + (k - 1) * w,
            vals,
            w,
            color=pp.paper_palette_role(TERM_ROLE[term]),
            label=TERM_LABEL[term],
            edgecolor="white",
            linewidth=0.6,
        )
        for xi, v in zip(x + (k - 1) * w, vals, strict=True):
            ax_top.annotate(
                f"{v:.2f}",
                (xi, v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
                xytext=(0, 2 if v >= 0 else -2),
                textcoords="offset points",
            )
    ax_top.axhline(0.0, color="0.35", lw=0.8)
    ax_top.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax_top.set_ylabel(
        "share of the on-policy shift\n" + r"$\langle$term$,\Delta\rangle/\|\Delta\|^2$"
    )
    ax_top.set_ylim(min(-0.08, ax_top.get_ylim()[0]), max(1.34, ax_top.get_ylim()[1] * 1.28))
    ax_top.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    for k, term in enumerate(TERMS):
        vals = [recs[a]["norms"][term] for a in arms]
        ax_bot.bar(
            x + (k - 1) * w,
            vals,
            w,
            color=pp.paper_palette_role(TERM_ROLE[term]),
            edgecolor="white",
            linewidth=0.6,
        )
    shift = [recs[a]["norm_shift"] for a in arms]
    ax_bot.plot(x, shift, "o", color=pp.paper_palette_role("control"), ms=6, label=r"$\|\Delta\|$")
    for xi, v in zip(x, shift, strict=True):
        ax_bot.annotate(
            f"{v:.1f}", (xi, v), ha="center", fontsize=7, xytext=(0, 4), textcoords="offset points"
        )
    ax_bot.set_ylabel("mean-shift norm at L" + layer)
    ax_bot.set_ylim(0.0, max(shift) * 1.22)
    ax_bot.legend(frameon=False, fontsize=8, loc="upper left")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([_short(a) for a in arms], fontsize=7)
    n = recs[arms[0]]["n_rows"]
    pp.set_title_subtitle(
        ax_top,
        "Where the on-policy shift comes from: measured (model x text) decomposition",
        f"{len(arms)} write-predictability arms, layer {layer}, response-span mean over the "
        f"sha-joined corpus rows (n~{n:,}/arm). Terms sum EXACTLY to the shift.",
    )
    fig.tight_layout()
    return pp.savefig_paper(fig, "decomposition_shares", dir=out_dir)["png"]


def fig_measured_vs_attributed(summary: dict, out_dir: Path) -> Path | None:
    rows = [
        (a, r) for a, r in summary["m0_attribution_vs_measured"].items() if "mean_shift_cos" in r
    ]
    if not rows:
        return None
    rows.sort(key=lambda kv: (BEH_ORDER.index(kv[0].split("-")[0]), kv[0]))
    arms = [a for a, _ in rows]
    dense = len(arms) > 16  # fleet-wide: markers + rotated ids, not annotated bars
    x = np.arange(len(arms))
    cos = [r["mean_shift_cos"] for _, r in rows]
    per_row = [r["per_row_cos_median"] for _, r in rows]
    rel = [r["mean_shift_rel_err"] for _, r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    ax1.bar(
        x - 0.19,
        cos,
        0.38,
        color=pp.paper_palette_role("primary"),
        label="mean-shift cosine",
        edgecolor="white",
    )
    ax1.bar(
        x + 0.19,
        per_row,
        0.38,
        color=pp.paper_palette_role("neutral"),
        label="per-row cosine (median)",
        edgecolor="white",
    )
    if not dense:
        for xi, v in list(zip(x - 0.19, cos, strict=True)) + list(
            zip(x + 0.19, per_row, strict=True)
        ):
            ax1.annotate(
                f"{v:.2f}",
                (xi, v),
                ha="center",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax1.axhline(1.0, color="0.75", ls=":", lw=0.8)
    ax1.set_ylabel("cos(attributed, measured)")
    ax1.set_ylim(min(-0.12, min(cos + per_row) - 0.05), 1.05)
    # lower-right keeps the legend clear of the top-of-axes behavior group labels
    ax1.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
    ax2.bar(
        x, rel, 0.5 if not dense else 0.72, color=pp.paper_palette_role("accent"), edgecolor="white"
    )
    if not dense:
        for xi, v in zip(x, rel, strict=True):
            ax2.annotate(
                f"{v:.2f}",
                (xi, v),
                ha="center",
                fontsize=7,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax2.axhline(1.0, color="0.75", ls=":", lw=0.8)
    ax2.axhline(0.0, color="0.35", lw=0.8)
    ax2.set_ylabel(r"relative error $\|a-m\|/\|m\|$")
    ax2.set_xticks(x)
    if dense:
        ax2.set_xticklabels(arms, rotation=90, fontsize=5.5)
        _beh_separators(ax1, arms, label_y=0.99)
        fig.set_size_inches(15.0, 7.2)
    else:
        ax2.set_xticklabels([_short(a) for a in arms], fontsize=7)
    layer = rows[0][1]["layer"]
    pp.set_title_subtitle(
        ax1,
        "Does round 1's map stand-in reproduce the measured text effect?",
        f"attributed $= M_0(c^+)-M_0(c^0)$ (base map refit on the same cell/split) vs "
        f"measured $= v^0(t^+)-v^0(t^0)$; held-out test rows, layer {layer}.",
    )
    fig.tight_layout()
    return pp.savefig_paper(fig, "measured_vs_attributed_text", dir=out_dir)["png"]


def fig_leg_b(summary: dict, layer: str, out_dir: Path) -> Path:
    recs = summary["leg_b"]
    order = sorted(recs, key=lambda a: (a.split("-")[0], a))
    beh = [a.split("-")[0] for a in order]
    x = np.arange(len(order))
    cd = [recs[a]["layers"][layer].get("cos_delta_v_train_delta", np.nan) for a in order]
    cw = [
        recs[a]["layers"][layer].get("cos_delta_v_train_corpus_matched_write", np.nan)
        for a in order
    ]
    nn = [recs[a]["layers"][layer]["norm_delta_v_train"] for a in order]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15.0, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]}
    )
    ax1.plot(
        x,
        cw,
        "o",
        ms=5,
        color=pp.paper_palette_role("baseline"),
        label=r"cos($\Delta v_{train}$, corpus matched-text write)",
    )
    ax1.plot(
        x,
        cd,
        "s",
        ms=5,
        color=pp.paper_palette_role("control"),
        label=r"cos($\Delta v_{train}$, $\delta$)",
    )
    ax1.axhline(0.0, color="0.35", lw=0.8)
    ax1.set_ylabel("cosine")
    ax1.set_ylim(min(-0.25, np.nanmin(cd + cw) - 0.1), 1.42)
    ax1.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax2.bar(x, nn, 0.7, color=pp.paper_palette_role("primary"), edgecolor="white", linewidth=0.4)
    ax2.set_ylabel(r"$\|\Delta v_{train}\|$ at L" + layer)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=90, fontsize=5.5)
    # behavior group separators + labels
    prev, start = beh[0], 0
    for i in range(1, len(beh) + 1):
        if i == len(beh) or beh[i] != prev:
            mid = (start + i - 1) / 2
            for ax in (ax1, ax2):
                if i < len(beh):
                    ax.axvline(i - 0.5, color="0.85", lw=0.7)
            ax1.annotate(
                BEH_LABEL.get(prev, prev),
                (mid, 0.965),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=8,
                color="0.35",
            )
            if i < len(beh):
                prev, start = beh[i], i
    pp.set_title_subtitle(
        ax1,
        r"Leg B: did the write land where training pushed, on the rows it pushed?",
        r"$\Delta v_{train}=v^+(\mathrm{train\ rows})-v^0(\mathrm{train\ rows})$ for all "
        + f"{len(order)} arms at layer {layer}; base side reused from round 1's delta cells.",
    )
    fig.tight_layout()
    return pp.savefig_paper(fig, "leg_b_train_rows", dir=out_dir)["png"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="figures for the #1768 2x2 round")
    p.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_1768" / "model_text_2x2" / "summary.json",
    )
    p.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures" / "issue_1768" / "model_text_2x2"
    )
    p.add_argument("--layer", default="19")
    a = p.parse_args(argv)
    summary = json.loads(a.summary.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pp.set_paper_style("blog")
    arms = summary["rtf_arms"]
    if len(arms) > 16:
        made = [fig_decomposition_fleet(summary, a.layer, a.out_dir)]
    else:
        made = [fig_decomposition(summary, a.layer, a.out_dir)]
    m = fig_measured_vs_attributed(summary, a.out_dir)
    if m is not None:
        made.append(m)
    else:
        print("[figs] measured-vs-attributed SKIPPED: no computed attribution rows")
    made.append(fig_leg_b(summary, a.layer, a.out_dir))
    for path in made:
        print(f"[figs] wrote {path}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.exit(rc)
