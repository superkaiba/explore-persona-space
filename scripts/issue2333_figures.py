#!/usr/bin/env python
"""Issue #2333 figures — snowball recovery profile (hero) + §6 exploratory.

Inputs: eval_results/issue_2333/f_metrics/<tag>/{f_cells,null_cells,calib_cells,
ce_cells}.jsonl + stats.json (from issue2333_analysis.py). All CIs are
pair-clustered bootstrap 95% (B=10,000 seed 23330 — the registered battery),
recomputed here for the plotted means via `bootstrap_family_means_batched`.
Figures follow /paper-plots conventions (no caption blocks in-canvas; error
bars are non-negative offsets, clamped)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import issue2162_analysis as A62  # noqa: E402
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402

FMETRICS_DIR = Path("eval_results/issue_2333/f_metrics")
FIG_DIR = Path("figures/issue_2333")
BOOT_B = 10_000
BOOT_SEED = C.BOOTSTRAP_SEED
SEPARATION_BAR = A62.SEPARATION_BAR

ARM_ORDER = ["ce", "patch1", "patch2", "patch3", "prefill1", "prefill2", "prefill3"]

# Reader-facing label maps (no bare slugs on any canvas — #2333 r2 revision).
MODEL_LABEL = {"q25": "Qwen2.5-7B", "q35": "Qwen3.5-9B"}
SET_LABEL = {"s1": "instruction-format pairs", "s2": "pirate matched-query pairs"}
SCHEME_LABEL = {
    "med": "patch-content donors (confirmatory)",
    "bstart": "natural-opening donors (descriptive)",
}
KIND_LABEL = {"patch": "state patch", "prefill": "token prefill"}
ARM_TICK = {
    "ce": "context-end patch\n(control)",
    "patch1": "state patch, 1 pos",
    "patch2": "state patch, 2 pos",
    "patch3": "state patch, 3 pos",
    "prefill1": "prefill, 1 token",
    "prefill2": "prefill, 2 tokens",
    "prefill3": "prefill, 3 tokens",
}
FBEH_LABEL = "behavioral movement F"


def _arm_slug_label(slug: str) -> str:
    """Reader-facing form of an arm slug, e.g. 'patch2_bstart' ->
    'state patch, 2 pos — natural-opening'."""
    kind_k, scheme = slug.rsplit("_", 1)
    kind = "patch" if kind_k.startswith("patch") else "prefill"
    k = kind_k.removeprefix(kind)
    unit = "pos" if kind == "patch" else ("token" if k == "1" else "tokens")
    scheme_short = "patch-content" if scheme == "med" else "natural-opening"
    return f"{KIND_LABEL[kind]}, {k} {unit} — {scheme_short}"


def _load_tag(tag: str) -> dict:
    d = FMETRICS_DIR / tag
    out = {
        "steered": list(A62._iter_jsonl(d / "f_cells.jsonl")),
        "null": list(A62._iter_jsonl(d / "null_cells.jsonl")),
        "calib": list(A62._iter_jsonl(d / "calib_cells.jsonl")),
        "stats": A62.json.loads((d / "stats.json").read_text(encoding="utf-8")),
    }
    ce = d / "ce_cells.jsonl"
    out["ce"] = list(A62._iter_jsonl(ce)) if ce.is_file() else []
    return out


def _wellsep(rows: list[dict]) -> set[str]:
    return {
        r["pair_id"]
        for r in rows
        if r.get("separation") is not None and abs(r["separation"]) >= SEPARATION_BAR
    }


def _mean_ci(values: list[float]) -> tuple[float, float, float] | None:
    """(mean, lo, hi) — pair-clustered bootstrap 95% CI over the pair axis.

    n == 1 degrades to a zero-width CI (point still plotted) so a smoke-scale
    single-pair panel renders its points; production floors (S1 >= 12 / S2 >= 5
    wellsep pairs) make the n >= 2 bootstrap branch the only production path.
    """
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    if v.size == 1:
        m = float(v[0])
        return (m, m, m)
    draws = bootstrap_family_means_batched(v[:, None], BOOT_B, BOOT_SEED)[:, 0]
    return (
        float(v.mean()),
        float(np.nanpercentile(draws, 2.5)),
        float(np.nanpercentile(draws, 97.5)),
    )


def _ce_values(
    data: dict, tag: str, set_name: str, keep: set[str], field: str = "f_beh"
) -> list[float]:
    """SAME-WAVE ce F per pair (q25: calib steered; q35: fresh ce_control).
    ``field="f_act"`` exists only on the q35 fresh ce rows (calib rows are
    banked TEXT re-judged — no V_a), so q25 f_act returns []."""
    if tag == "q35":
        rows = [r for r in data["ce"] if r["variant"] == "steered" and r["set"] == set_name]
    else:
        rows = [r for r in data["calib"] if r["arm"] == "steered" and r["set"] == set_name]
    return [r[field] for r in rows if r["pair_id"] in keep and r.get(field) is not None]


def _arm_values(
    rows: list[dict], set_name: str, slug: str, keep: set[str], field: str = "f_beh"
) -> list[float]:
    return [
        r[field]
        for r in rows
        if r["set"] == set_name
        and r["arm_slug"] == slug
        and r["pair_id"] in keep
        and r.get(field) is not None
    ]


def fig_hero(data: dict, tag: str, field: str = "f_beh") -> None:
    """Hero: F per arm [ce, patch k=1..3, prefill k=1..3], steered vs
    scheme-matched shuffled-donor null, per (pair-set x scheme). The same-wave
    ce mean rides a shaded 95%-CI BAND across the axis (the recovery target
    every arm is read against) plus its own point. ``field="f_act"`` renders
    the secondary-DV mirror (suffix ``_act``)."""
    colors = paper_palette(3)
    keep = _wellsep([*data["steered"], *data["null"]])
    suffix = "" if field == "f_beh" else "_act"
    for set_name in ("s1", "s2"):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
        any_points = False
        for ax, scheme in zip(axes, C.ARM_SCHEMES, strict=True):
            xs = np.arange(len(ARM_ORDER))
            ce_mc = _mean_ci(_ce_values(data, tag, set_name, keep, field=field))
            if ce_mc is not None:
                m, lo, hi = ce_mc
                ax.axhspan(lo, hi, color=colors[2], alpha=0.15, zorder=0)
                ax.axhline(
                    m,
                    color=colors[2],
                    lw=0.8,
                    ls="--",
                    label="full context-end patch, same wave (95% CI band)",
                )
            for off, (variant, rows, color) in enumerate(
                (("steered", data["steered"], colors[0]), ("null", data["null"], colors[1]))
            ):
                means, lows, highs, xpos = [], [], [], []
                for i, arm in enumerate(ARM_ORDER):
                    if arm == "ce":
                        vals = (
                            _ce_values(data, tag, set_name, keep, field=field)
                            if variant == "steered"
                            else []
                        )
                    else:
                        vals = _arm_values(rows, set_name, f"{arm}_{scheme}", keep, field=field)
                    mc = _mean_ci(vals)
                    if mc is None:
                        continue
                    m, lo, hi = mc
                    means.append(m)
                    lows.append(max(0.0, m - lo))
                    highs.append(max(0.0, hi - m))
                    xpos.append(i + (off - 0.5) * 0.22)
                any_points = any_points or bool(means)
                ax.errorbar(
                    xpos,
                    means,
                    yerr=[lows, highs],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=(
                        "steered (true donor)" if variant == "steered" else "shuffled-donor null"
                    ),
                )
            ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels([ARM_TICK[a] for a in ARM_ORDER], rotation=30, ha="right")
            ax.set_title(SCHEME_LABEL[scheme])
        if not any_points:  # e.g. f_act with no staged V_a store — skip, don't emit blank axes
            plt.close(fig)
            print(f"[figures] hero{suffix} {tag}/{set_name}: no {field} values — skipped")
            continue
        ylabel = FBEH_LABEL if field == "f_beh" else "activation movement F (secondary DV)"
        axes[0].set_ylabel(ylabel)
        axes[0].legend(frameon=False, fontsize=8)
        dv_note = "" if field == "f_beh" else " — activation DV"
        fig.suptitle(
            f"{MODEL_LABEL[tag]} — {SET_LABEL[set_name]}: snowball recovery profile{dv_note}",
            y=1.02,
        )
        fig.tight_layout()
        savefig_paper(fig, f"hero_snowball{suffix}_{tag}_{set_name}", dir=str(FIG_DIR))
        plt.close(fig)


def fig_recovery(data: dict, tag: str) -> None:
    """Exploratory: recovery ratio R_k = F_arm / F_ce(same-wave) vs k."""
    stats = data["stats"]
    colors = paper_palette(2)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharey=True)
    for i, set_name in enumerate(("s1", "s2")):
        arms = stats["per_set"].get(set_name, {}).get("arms", {})
        for j, scheme in enumerate(C.ARM_SCHEMES):
            ax = axes[i][j]
            for kind, color in zip(C.ARM_KINDS, colors, strict=True):
                ks, rs, los, his = [], [], [], []
                for k in C.ARM_KS:
                    rec = arms.get(f"{kind}{k}_{scheme}", {}).get("recovery_samewave")
                    if not rec:
                        continue
                    ks.append(k)
                    rs.append(rec["ratio"])
                    lo, hi = rec["ratio_ci"]
                    los.append(max(0.0, rec["ratio"] - lo))
                    his.append(max(0.0, hi - rec["ratio"]))
                if ks:
                    ax.errorbar(
                        ks, rs, yerr=[los, his], marker="o", color=color, label=KIND_LABEL[kind]
                    )
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
            ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
            ax.set_title(f"{SET_LABEL[set_name]}\n{SCHEME_LABEL[scheme]}", fontsize=9)
            ax.set_xticks(list(C.ARM_KS))
    axes[1][0].set_xlabel("opening positions transplanted")
    axes[1][1].set_xlabel("opening positions transplanted")
    axes[0][0].set_ylabel("recovery ratio (arm F / control F)")
    axes[1][0].set_ylabel("recovery ratio (arm F / control F)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{MODEL_LABEL[tag]}: recovery vs opening length (steered arms, same-wave control)",
        y=1.01,
    )
    fig.tight_layout()
    savefig_paper(fig, f"recovery_ratio_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_perpair(data: dict, tag: str) -> None:
    """Exploratory: per-pair steered-vs-null F at prefill-3 (med)."""
    keep = _wellsep([*data["steered"], *data["null"]])
    st = {
        (r["set"], r["pair_id"]): r["f_beh"]
        for r in data["steered"]
        if r["arm_slug"] == "prefill3_med" and r["pair_id"] in keep and r["f_beh"] is not None
    }
    nu = {
        (r["set"], r["pair_id"]): r["f_beh"]
        for r in data["null"]
        if r["arm_slug"] == "prefill3_med" and r["pair_id"] in keep and r["f_beh"] is not None
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = paper_palette(2)
    for set_name, color in zip(("s1", "s2"), colors, strict=True):
        pts = [(nu[k], st[k]) for k in st if k in nu and k[0] == set_name]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.scatter(
                xs,
                ys,
                s=18,
                color=color,
                alpha=0.75,
                label=f"{SET_LABEL[set_name]} (n={len(pts)})",
            )
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("behavioral movement F — shuffled-donor null")
    ax.set_ylabel("behavioral movement F — steered (true donor)")
    ax.set_title(f"{MODEL_LABEL[tag]}: three-token prefill, patch-content donors — per pair")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"perpair_prefill3_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_forest_cells(data: dict, tag: str) -> None:
    """Per-S1-cell forests (plan §6): paired diff (steered − null) per arm,
    one forest panel per surviving S1 cell — the low-level per-unit view
    behind the pooled S1 hero."""
    keep = _wellsep([*data["steered"], *data["null"]])
    st = {(r["cell"], r["pair_id"], r["arm_slug"]): r["f_beh"] for r in data["steered"]}
    nu = {(r["cell"], r["pair_id"], r["arm_slug"]): r["f_beh"] for r in data["null"]}
    cells = [c for c in C.S1_CELLS]
    fig, axes = plt.subplots(1, len(cells), figsize=(2.4 * len(cells), 4.2), sharex=True)
    colors = paper_palette(2)
    slugs = list(C.ARM_SLUGS)
    drew = False
    for ax, cell in zip(axes, cells, strict=True):
        ys, ms, los, his, cs = [], [], [], [], []
        for y, slug in enumerate(slugs):
            diffs = [
                st[(cell, pid, slug)] - nu[(cell, pid, slug)]
                for (c2, pid, s2) in st
                if c2 == cell
                and s2 == slug
                and pid in keep
                and st[(cell, pid, slug)] is not None
                and nu.get((cell, pid, slug)) is not None
            ]
            mc = _mean_ci(diffs)
            if mc is None:
                continue
            m, lo, hi = mc
            ys.append(y)
            ms.append(m)
            los.append(max(0.0, m - lo))
            his.append(max(0.0, hi - m))
            cs.append(colors[0] if slug.endswith("_med") else colors[1])
        for y, m, lo_, hi_, c2 in zip(ys, ms, los, his, cs, strict=True):
            ax.errorbar([m], [y], xerr=[[lo_], [hi_]], fmt="o", color=c2, capsize=2, ms=3.5)
        drew = drew or bool(ys)
        ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
        ax.set_yticks(range(len(slugs)))
        pretty = [_arm_slug_label(s) for s in slugs]
        ax.set_yticklabels(pretty if ax is axes[0] else [""] * len(slugs), fontsize=6)
        ax.set_title(cell.replace("_", " "), fontsize=8)
        ax.invert_yaxis()
    if not drew:
        plt.close(fig)
        print(f"[figures] forest_cells {tag}: no per-cell diffs — skipped")
        return
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colors[0], label="patch-content donors"),
        plt.Line2D([], [], marker="o", ls="", color=colors[1], label="natural-opening donors"),
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    fig.supxlabel("behavioral movement F: steered − null (95% CI)")
    fig.suptitle(f"{MODEL_LABEL[tag]}: per-cell paired differences per arm", y=1.02)
    fig.tight_layout()
    savefig_paper(fig, f"forest_cells_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_k_traces(data: dict, tag: str) -> None:
    """k-monotonicity per-pair traces (plan §6): each pair's paired diff
    (steered − null) traced across k=1..3, thin per-pair lines + mean overlay,
    per (set x scheme x kind)."""
    keep = _wellsep([*data["steered"], *data["null"]])
    st = {(r["pair_id"], r["arm_slug"]): r["f_beh"] for r in data["steered"]}
    nu = {(r["pair_id"], r["arm_slug"]): r["f_beh"] for r in data["null"]}
    colors = dict(zip(C.ARM_KINDS, paper_palette(2), strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharey=True)
    drew = False
    for i, set_name in enumerate(("s1", "s2")):
        set_pids = {
            r["pair_id"] for r in data["steered"] if r["set"] == set_name and r["pair_id"] in keep
        }
        for j, scheme in enumerate(C.ARM_SCHEMES):
            ax = axes[i][j]
            for kind in C.ARM_KINDS:
                mean_by_k: dict[int, list[float]] = {k: [] for k in C.ARM_KS}
                for pid in sorted(set_pids):
                    ks, ds = [], []
                    for k in C.ARM_KS:
                        slug = f"{kind}{k}_{scheme}"
                        a, b = st.get((pid, slug)), nu.get((pid, slug))
                        if a is None or b is None:
                            continue
                        ks.append(k)
                        ds.append(a - b)
                        mean_by_k[k].append(a - b)
                    if len(ks) >= 2:
                        ax.plot(ks, ds, color=colors[kind], alpha=0.15, lw=0.7)
                        drew = True
                mk = [k for k in C.ARM_KS if mean_by_k[k]]
                if mk:
                    ax.plot(
                        mk,
                        [float(np.mean(mean_by_k[k])) for k in mk],
                        color=colors[kind],
                        lw=2.0,
                        marker="o",
                        label=KIND_LABEL[kind],
                    )
            ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
            ax.set_title(f"{SET_LABEL[set_name]}\n{SCHEME_LABEL[scheme]}", fontsize=8)
            ax.set_xticks(list(C.ARM_KS))
    if not drew:
        plt.close(fig)
        print(f"[figures] k_traces {tag}: no per-pair traces — skipped")
        return
    axes[1][0].set_xlabel("opening positions transplanted")
    axes[1][1].set_xlabel("opening positions transplanted")
    axes[0][0].set_ylabel("F steered − F null")
    axes[1][0].set_ylabel("F steered − F null")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{MODEL_LABEL[tag]}: per-pair traces (thin) + means (bold)", y=1.01)
    fig.tight_layout()
    savefig_paper(fig, f"k_traces_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_scheme_contrast(data: dict, tag: str) -> None:
    """Donor-scheme contrast (plan §6): per-pair steered F under med (x) vs
    bstart (y), per (kind, k) — the natural-opening descriptive comparison."""
    keep = _wellsep(data["steered"])
    st = {(r["pair_id"], r["arm_slug"]): r["f_beh"] for r in data["steered"]}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = dict(zip(C.ARM_KINDS, paper_palette(2), strict=True))
    markers = {1: "o", 2: "s", 3: "^"}
    drew = False
    for kind in C.ARM_KINDS:
        for k in C.ARM_KS:
            pts = [
                (st[(pid, f"{kind}{k}_med")], st[(pid, f"{kind}{k}_bstart")])
                for pid in sorted(keep)
                if st.get((pid, f"{kind}{k}_med")) is not None
                and st.get((pid, f"{kind}{k}_bstart")) is not None
            ]
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.scatter(
                    xs,
                    ys,
                    s=14,
                    color=colors[kind],
                    marker=markers[k],
                    alpha=0.6,
                    label=f"{KIND_LABEL[kind]}, {k} pos",
                )
                drew = True
    if not drew:
        plt.close(fig)
        print(f"[figures] scheme_contrast {tag}: no paired scheme values — skipped")
        return
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("behavioral movement F — patch-content donors")
    ax.set_ylabel("behavioral movement F — natural-opening donors")
    ax.set_title(f"{MODEL_LABEL[tag]}: donor-scheme contrast (steered)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_paper(fig, f"scheme_contrast_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_arm_vs_ce(data: dict, tag: str) -> None:
    """Per-arm F_arm vs same-wave F_ce scatter (plan §6): each point one
    (pair, arm); identity + the D3 half-share line are the reference reads."""
    keep = _wellsep(data["steered"])
    ce_by_pair: dict[str, float] = {}
    for set_name in ("s1", "s2"):
        if tag == "q35":
            rows = [r for r in data["ce"] if r["variant"] == "steered" and r["set"] == set_name]
        else:
            rows = [r for r in data["calib"] if r["arm"] == "steered" and r["set"] == set_name]
        for r in rows:
            if r["f_beh"] is not None:
                ce_by_pair[r["pair_id"]] = r["f_beh"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
    colors = dict(zip(C.ARM_KS, paper_palette(3), strict=True))
    drew = False
    for ax, kind in zip(axes, C.ARM_KINDS, strict=True):
        for k in C.ARM_KS:
            for scheme, alpha in (("med", 0.8), ("bstart", 0.35)):
                pts = [
                    (ce_by_pair[r["pair_id"]], r["f_beh"])
                    for r in data["steered"]
                    if r["arm_slug"] == f"{kind}{k}_{scheme}"
                    and r["pair_id"] in keep
                    and r["pair_id"] in ce_by_pair
                    and r["f_beh"] is not None
                ]
                if pts:
                    xs, ys = zip(*pts, strict=True)
                    scheme_short = "patch-content" if scheme == "med" else "natural-opening"
                    ax.scatter(
                        xs,
                        ys,
                        s=12,
                        color=colors[k],
                        alpha=alpha,
                        label=f"{k} pos, {scheme_short}" if kind == C.ARM_KINDS[0] else None,
                    )
                    drew = True
        lim = ax.get_xlim() + ax.get_ylim()
        lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":", label=None)
        ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="0.6", lw=0.8, ls="--", label=None)
        ax.set_title(KIND_LABEL[kind], fontsize=9)
        ax.set_xlabel("control F (full context-end patch, same wave)")
    if not drew:
        plt.close(fig)
        print(f"[figures] arm_vs_ce {tag}: no (arm, ce) pairs — skipped")
        return
    axes[0].set_ylabel("arm F (steered)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        f"{MODEL_LABEL[tag]}: per-pair arm F vs control F (identity ┊, half-share ╌)", y=1.01
    )
    fig.tight_layout()
    savefig_paper(fig, f"arm_vs_ce_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_whole_vs_continuation(data: dict, tag: str) -> None:
    """Exploratory: whole-response vs continuation-only F on prefill arms."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = paper_palette(3)
    for k, color in zip(C.ARM_KS, colors, strict=True):
        pts = [
            (r["f_beh"], r["f_beh_continuation"])
            for r in data["steered"]
            if r["kind"] == "prefill"
            and r["k"] == k
            and r["f_beh"] is not None
            and r.get("f_beh_continuation") is not None
        ]
        if pts:
            xs, ys = zip(*pts, strict=True)
            unit = "token" if k == 1 else "tokens"
            ax.scatter(
                xs, ys, s=14, color=color, alpha=0.6, label=f"{k}-{unit} prefill (n={len(pts)})"
            )
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("whole-response F (donor opening included)")
    ax.set_ylabel("continuation-only F")
    ax.set_title(f"{MODEL_LABEL[tag]}: prefill steered rows")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"whole_vs_continuation_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_coherence(data: dict, tag: str) -> None:
    """Exploratory: coherent fraction per arm (steered + null pooled)."""
    frac: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for r in [*data["steered"], *data["null"]]:
        c, n = frac[r["arm_slug"]]
        frac[r["arm_slug"]] = (c + r["n_coherent"], n + r["n_rows"])
    slugs = [s for s in C.ARM_SLUGS if s in frac]
    vals = [frac[s][0] / max(1, frac[s][1]) for s in slugs]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(len(slugs)), vals, color=paper_palette(1)[0])
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels([_arm_slug_label(s) for s in slugs], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("coherent fraction (judge > 60)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{MODEL_LABEL[tag]}: coherence survival per arm")
    fig.tight_layout()
    savefig_paper(fig, f"coherence_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_model_compare(tags: list[str]) -> None:
    """Cross-model: prefill-3 (patch-content) paired diff (steered - null) per
    set, annotated with the registered two-conjunct verdict per row (a
    positive CI alone does not separate — Holm-corrected signed-rank must
    agree; #2333 r2)."""
    rows = []
    for tag in tags:
        stats = A62.json.loads((FMETRICS_DIR / tag / "stats.json").read_text(encoding="utf-8"))
        for set_name, per in stats["per_set"].items():
            rec = per["arms"].get("prefill3_med", {})
            if "diff_ci" in rec:
                lo, hi = rec["diff_ci"]
                if rec.get("separates"):
                    verdict = "separates (both conjuncts)"
                elif lo > 0.0:
                    verdict = f"does not separate (corrected p = {rec['p_holm']:.2g})"
                elif rec.get("holm_significant"):
                    verdict = "does not separate (CI spans 0)"
                else:
                    verdict = "does not separate (CI spans 0; corrected p n.s.)"
                rows.append(
                    (
                        f"{MODEL_LABEL[tag]}\n{SET_LABEL[set_name]}",
                        rec["diff_mean"],
                        lo,
                        hi,
                        verdict,
                    )
                )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 0.8 + 0.62 * len(rows)))
    ys = np.arange(len(rows))
    for y, (label, m, lo, hi, verdict) in zip(ys, rows, strict=True):
        ax.errorbar(
            [m],
            [y],
            xerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="o",
            color=paper_palette(1)[0],
            capsize=3,
        )
        ax.text(max(hi for r in rows for hi in (r[3],)) + 0.03, y, verdict, va="center", fontsize=7)
    ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(right=max(r[3] for r in rows) + 0.45)
    ax.set_xlabel("three-token prefill, patch-content donors: steered − null F (95% CI)")
    fig.tight_layout()
    savefig_paper(fig, "model_compare_prefill3", dir=str(FIG_DIR))
    plt.close(fig)


def fig_continuation_d3(tags: list[str]) -> None:
    """Follow-up recount (9a-ter free-analysis round): majority-share margin D3
    for the confirmatory three-token prefill under the whole-response DV
    (stats.json) vs the continuation-only DV (followup_free/
    continuation_lattice.json), per model, same-wave control denominator.
    The zero line is the snowball-sufficient boundary of the D3 conjunct."""
    reads = []  # (model_label, dv_label, mean, lo, hi)
    for tag in tags:
        stats = A62.json.loads((FMETRICS_DIR / tag / "stats.json").read_text(encoding="utf-8"))
        whole = stats["per_set"]["s1"]["arms"]["prefill3_med"]["recovery_samewave"]
        cont_path = FMETRICS_DIR / tag / "followup_free" / "continuation_lattice.json"
        if not cont_path.is_file():
            print(f"[figures] continuation_d3 {tag}: no followup_free lattice — skipped")
            continue
        cont = A62.json.loads(cont_path.read_text(encoding="utf-8"))
        cont_rec = cont["arms"]["prefill3_med"]["recovery_samewave"]
        for dv_label, rec in (
            ("whole response", whole),
            ("continuation only", cont_rec),
        ):
            if rec.get("d3_mean") is None or rec.get("d3_ci") is None:
                continue
            lo, hi = rec["d3_ci"]
            reads.append((MODEL_LABEL[tag], dv_label, rec["d3_mean"], lo, hi))
    if not reads:
        print("[figures] continuation_d3: no reads — skipped")
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = {"whole response": paper_palette(2)[0], "continuation only": paper_palette(2)[1]}
    model_order = [MODEL_LABEL[t] for t in tags if MODEL_LABEL[t] in {r[0] for r in reads}]
    offsets = {"whole response": -0.16, "continuation only": 0.16}
    for model_label, dv_label, m, lo, hi in reads:
        x = model_order.index(model_label) + offsets[dv_label]
        ax.errorbar(
            [x],
            [m],
            yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="o",
            color=colors[dv_label],
            capsize=3,
            label=dv_label,
        )
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order)
    ax.set_xlim(-0.6, len(model_order) - 0.4)
    ax.set_ylabel("majority-share margin (95% CI)")
    ax.set_title("three-token prefill, patch-content donors")
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels, strict=True):
        seen.setdefault(lab, h)
    ax.legend(seen.values(), seen.keys(), frameon=False, loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, "continuation_d3_recount", dir=str(FIG_DIR))
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2333 figures.")
    ap.add_argument("--model-tags", nargs="+", default=["q25"], choices=("q25", "q35"))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument(
        "--continuation-d3-only",
        action="store_true",
        help="Render only the follow-up continuation-recount D3 figure (no full regen).",
    )
    return ap.parse_args(argv)


def _import_check() -> int:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    assert callable(savefig_paper) and callable(bootstrap_family_means_batched)
    print("[import-check] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.import_check:
        return _import_check()
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.continuation_d3_only:
        fig_continuation_d3(args.model_tags)
        print(f"[figures] wrote continuation_d3_recount for {args.model_tags} under {FIG_DIR}")
        return 0
    for tag in args.model_tags:
        data = _load_tag(tag)
        fig_hero(data, tag)
        fig_hero(data, tag, field="f_act")  # secondary-DV mirror (plan §6)
        fig_recovery(data, tag)
        fig_perpair(data, tag)
        fig_forest_cells(data, tag)
        fig_k_traces(data, tag)
        fig_scheme_contrast(data, tag)
        fig_arm_vs_ce(data, tag)
        fig_whole_vs_continuation(data, tag)
        fig_coherence(data, tag)
    if len(args.model_tags) > 1:
        fig_model_compare(args.model_tags)
    fig_continuation_d3(args.model_tags)
    print(f"[figures] wrote figures for {args.model_tags} under {FIG_DIR}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
