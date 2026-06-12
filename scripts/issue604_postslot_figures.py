# ruff: noqa: RUF001
# Intentional Unicode (Δ, —) in scientific labels.
"""Figures for the #604 same-issue follow-up `post-response-slot-key`.

Two arm-distinct figures (never overwriting the parent figure paths):

1. ``postslot_key_match_layer_profile`` — per-line layer profile of the
   top-1 key vs source-context cosine at the POST-RESPONSE SLOT (solid,
   with its wrong-context null band), overlaid with the last-prompt-token
   dedup-matched comparator read (dashed) so the position contrast is
   visible raw, per layer. Both reads use the SAME SHA-deduplicated null
   construction (the defect fix is applied symmetrically).
2. ``postslot_topk_subspace`` — top-k key-subspace projection energy at
   the post-response slot, per line, with duplicate-aware nulls and the
   parent (last-prompt-token) matched median overlaid per k.

Reads:
  eval_results/issue_604/post-response-slot-key/key_match.json
  eval_results/issue_604/post-response-slot-key/lastprompt_dedup/key_match.json
  eval_results/issue_604/post-response-slot-key/topk_subspace.json
  eval_results/issue_604/topk_subspace.json   (parent comparator, read-only)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "eval_results" / "issue_604"
ARM = OUT / "post-response-slot-key"
FIG = PROJECT_ROOT / "figures" / "issue_604"

LINE_LABELS = {
    "dial527": "dose dial — shallow window",
    "dial550": "dose dial — mid window",
    "dial538": "dose dial — deep window",
    "i474": "epoch ladder",
    "i518": "cross-behavior",
    "i519": "saturated marker endpoint",
}
PROFILE_ORDER = ["dial527", "dial550", "dial538", "i474", "i518", "i519"]

TOPK_LABELS = {
    "dial527": "dose dial — shallow window",
    "dial527_panel_contaminated": "shallow window (contaminated panel)",
    "dial550": "dose dial — mid window",
    "dial538": "dose dial — deep window",
    "i474_loc": "epoch ladder — contrastive arm",
    "i474_pos": "epoch ladder — positives-only arm",
    "i518": "cross-behavior",
    "i519": "saturated marker endpoint",
}
TOPK_ORDER = [
    "dial527",
    "dial550",
    "dial538",
    "i474_loc",
    "i474_pos",
    "i518",
    "i519",
    "dial527_panel_contaminated",
]


def _profiles(km: dict) -> dict[str, dict[str, list[float]]]:
    """Per-line layer profile: matched mean + null p50/p95 mean (attn space)."""
    by_line: dict[str, list[dict]] = defaultdict(list)
    for cell in km["cells"]:
        for src in cell["per_source"]:
            attn = src.get("stacks", {}).get("attn_key", {}).get("attn")
            if attn:
                by_line[cell["line"]].append(attn)
    out: dict[str, dict[str, list[float]]] = {}
    for line, rows in by_line.items():
        layers = sorted({r["layer"] for entry in rows for r in entry["layers"]})
        prof = {"layers": layers, "src": [], "p50": [], "p95": []}
        for layer in layers:
            vals = [r for entry in rows for r in entry["layers"] if r["layer"] == layer]
            prof["src"].append(float(np.mean([v["cos_src_abs"] for v in vals])))
            prof["p50"].append(float(np.mean([v["null_p50"] for v in vals])))
            prof["p95"].append(float(np.mean([v["null_p95"] for v in vals])))
        out[line] = prof
    return out


def fig_postslot_key_profile(km_new: dict, km_old: dict) -> None:
    """Layer profiles at the trained slot vs the prompt's last token, per line."""
    new_p = _profiles(km_new)
    old_p = _profiles(km_old)
    lines = [ln for ln in PROFILE_ORDER if ln in new_p]
    fig, axes = plt.subplots(1, len(lines), figsize=(3.6 * len(lines), 3.6), squeeze=False)
    c_new = paper_palette_role("primary")
    c_old = paper_palette_role("baseline")
    for ax, line in zip(axes[0], lines, strict=True):
        np_, op_ = new_p[line], old_p[line]
        ax.fill_between(
            np_["layers"],
            np_["p50"],
            np_["p95"],
            color=paper_palette_role("neutral"),
            alpha=0.35,
        )
        ax.plot(np_["layers"], np_["src"], color=c_new, lw=1.8)
        ax.plot(op_["layers"], op_["src"], color=c_old, lw=1.4, ls="--")
        ax.set_xlabel("layer")
        ax.set_ylabel("|cos(top key, context)|")
        ax.set_ylim(0, 0.16)
        set_title_subtitle(ax, LINE_LABELS.get(line, line), "module-input space (mean over cells)")
    handles = [
        Line2D([], [], color=c_new, lw=1.8, label="response-slot read (this round)"),
        Line2D([], [], color=c_old, lw=1.4, ls="--", label="prompt-token read (same dedup nulls)"),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc=paper_palette_role("neutral"),
            mec="none",
            alpha=0.5,
            label="wrong-context null p50–p95 (response slot)",
        ),
    ]
    axes[0][0].legend(handles=handles, fontsize=7, loc="upper left")
    savefig_paper(fig, "postslot_key_match_layer_profile", dir=FIG)
    plt.close(fig)


def fig_postslot_topk(tk_new: dict, tk_old: dict) -> None:
    """Top-k projection energy at the trained slot, parent read overlaid."""
    lines = [ln for ln in TOPK_ORDER if ln in tk_new["per_line"]]
    ncol = 4
    nrow = int(np.ceil(len(lines) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow), squeeze=False)
    ks = [1, 2, 4, 8]
    c_new = paper_palette_role("primary")
    c_old = paper_palette_role("baseline")
    for i, line in enumerate(lines):
        ax = axes[i // ncol][i % ncol]
        vn = tk_new["per_line"][line]["k"]
        vo = tk_old["per_line"][line]["k"]
        med_n = [vn[str(k)]["matched_p50"] for k in ks]
        med_o = [vo[str(k)]["matched_p50"] for k in ks]
        wr_lo = [vn[str(k)]["wrong_p50_median_over_rows"] for k in ks]
        wr_hi = [vn[str(k)]["wrong_p95_median_over_rows"] for k in ks]
        floor = [vn[str(k)]["random_floor"] for k in ks]
        ax.fill_between(ks, wr_lo, wr_hi, color=paper_palette_role("neutral"), alpha=0.35)
        sh = [vn[str(k)].get("shuffled_p95") for k in ks]
        if all(s is not None for s in sh):
            ax.plot(ks, sh, color=paper_palette_role("control"), lw=1.0, ls="--")
        ax.plot(ks, med_n, color=c_new, lw=1.8, marker="o", ms=3.5)
        ax.plot(ks, med_o, color=c_old, lw=1.4, ls="--", marker="o", ms=3, mfc="none", mew=1.1)
        ax.plot(ks, floor, color="#888888", lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(k) for k in ks]))
        ax.set_xlabel("subspace size k")
        ax.set_ylabel("captured energy")
        set_title_subtitle(ax, TOPK_LABELS.get(line, line), "band mean L14–L24")
    for j in range(len(lines), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [
        Line2D(
            [], [], color=c_new, lw=1.8, marker="o", ms=3.5, label="response-slot matched median"
        ),
        Line2D(
            [],
            [],
            color=c_old,
            lw=1.4,
            ls="--",
            marker="o",
            ms=3,
            mfc="none",
            mew=1.1,
            label="prompt-token matched median (parent read)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc=paper_palette_role("neutral"),
            mec="none",
            alpha=0.5,
            label="wrong-context null p50–p95 (response slot)",
        ),
        Line2D(
            [],
            [],
            color=paper_palette_role("control"),
            lw=1.0,
            ls="--",
            label="shuffled-pairing p95",
        ),
        Line2D([], [], color="#888888", lw=0.9, ls=":", label="random-direction floor (k/3584)"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="outside lower center", ncol=3, frameon=False)
    savefig_paper(fig, "postslot_topk_subspace", dir=FIG)
    plt.close(fig)


def main() -> None:
    """Build the two post-response-slot follow-up figures."""
    set_paper_style("blog")
    km_new = json.loads((ARM / "key_match.json").read_text())
    km_old = json.loads((ARM / "lastprompt_dedup" / "key_match.json").read_text())
    tk_new = json.loads((ARM / "topk_subspace.json").read_text())
    tk_old = json.loads((OUT / "topk_subspace.json").read_text())
    fig_postslot_key_profile(km_new, km_old)
    fig_postslot_topk(tk_new, tk_old)
    print("done:", FIG)


if __name__ == "__main__":
    main()
