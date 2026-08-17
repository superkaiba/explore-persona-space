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


def _ce_values(data: dict, tag: str, set_name: str, keep: set[str]) -> list[float]:
    """SAME-WAVE ce F per pair (q25: calib steered; q35: fresh ce_control)."""
    if tag == "q35":
        rows = [r for r in data["ce"] if r["variant"] == "steered" and r["set"] == set_name]
    else:
        rows = [r for r in data["calib"] if r["arm"] == "steered" and r["set"] == set_name]
    return [r["f_beh"] for r in rows if r["pair_id"] in keep and r["f_beh"] is not None]


def _arm_values(rows: list[dict], set_name: str, slug: str, keep: set[str]) -> list[float]:
    return [
        r["f_beh"]
        for r in rows
        if r["set"] == set_name
        and r["arm_slug"] == slug
        and r["pair_id"] in keep
        and r["f_beh"] is not None
    ]


def fig_hero(data: dict, tag: str) -> None:
    """Hero: F_beh per arm [ce, patch k=1..3, prefill k=1..3], steered vs
    scheme-matched shuffled-donor null, per (pair-set x scheme)."""
    colors = paper_palette(3)
    keep = _wellsep([*data["steered"], *data["null"]])
    for set_name in ("s1", "s2"):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
        for ax, scheme in zip(axes, C.ARM_SCHEMES, strict=True):
            xs = np.arange(len(ARM_ORDER))
            for off, (variant, rows, color) in enumerate(
                (("steered", data["steered"], colors[0]), ("null", data["null"], colors[1]))
            ):
                means, lows, highs, xpos = [], [], [], []
                for i, arm in enumerate(ARM_ORDER):
                    if arm == "ce":
                        vals = _ce_values(data, tag, set_name, keep) if variant == "steered" else []
                    else:
                        vals = _arm_values(rows, set_name, f"{arm}_{scheme}", keep)
                    mc = _mean_ci(vals)
                    if mc is None:
                        continue
                    m, lo, hi = mc
                    means.append(m)
                    lows.append(max(0.0, m - lo))
                    highs.append(max(0.0, hi - m))
                    xpos.append(i + (off - 0.5) * 0.22)
                ax.errorbar(
                    xpos,
                    means,
                    yerr=[lows, highs],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=f"{variant}" + (" (ce same-wave)" if variant == "steered" else ""),
                )
            ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels(ARM_ORDER, rotation=30, ha="right")
            ax.set_title(f"scheme = {scheme}" + ("" if scheme == "med" else " (natural-opening)"))
        axes[0].set_ylabel("F_beh (anchor-normalized)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{tag} / {set_name}: snowball recovery profile", y=1.02)
        fig.tight_layout()
        savefig_paper(fig, f"hero_snowball_{tag}_{set_name}", dir=str(FIG_DIR))
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
                    ax.errorbar(ks, rs, yerr=[los, his], marker="o", color=color, label=kind)
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
            ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
            ax.set_title(f"{set_name} / {scheme}")
            ax.set_xticks(list(C.ARM_KS))
    axes[1][0].set_xlabel("k (answer positions)")
    axes[1][1].set_xlabel("k (answer positions)")
    axes[0][0].set_ylabel("R_k = F_arm / F_ce")
    axes[1][0].set_ylabel("R_k = F_arm / F_ce")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{tag}: recovery ratio vs k (steered arms, same-wave ce)", y=1.01)
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
            ax.scatter(xs, ys, s=18, color=color, alpha=0.75, label=f"{set_name} (n={len(pts)})")
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("F_beh — shuffled-donor null")
    ax.set_ylabel("F_beh — steered donor")
    ax.set_title(f"{tag}: prefill-3 (med), per pair")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, f"perpair_prefill3_{tag}", dir=str(FIG_DIR))
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
            ax.scatter(xs, ys, s=14, color=color, alpha=0.6, label=f"k={k} (n={len(pts)})")
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("F_beh — whole response (donor opening included)")
    ax.set_ylabel("F_beh — continuation only")
    ax.set_title(f"{tag}: prefill steered arms")
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
    ax.set_xticklabels(slugs, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("coherent fraction (judge > 60)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{tag}: coherence survival per arm")
    fig.tight_layout()
    savefig_paper(fig, f"coherence_{tag}", dir=str(FIG_DIR))
    plt.close(fig)


def fig_model_compare(tags: list[str]) -> None:
    """Cross-model: prefill-3 (med) paired diff (steered - null) per set."""
    rows = []
    for tag in tags:
        stats = A62.json.loads((FMETRICS_DIR / tag / "stats.json").read_text(encoding="utf-8"))
        for set_name, per in stats["per_set"].items():
            rec = per["arms"].get("prefill3_med", {})
            if "diff_ci" in rec:
                rows.append((f"{tag}/{set_name}", rec["diff_mean"], *rec["diff_ci"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(4.8, 0.8 + 0.5 * len(rows)))
    ys = np.arange(len(rows))
    for y, (label, m, lo, hi) in zip(ys, rows, strict=True):
        ax.errorbar(
            [m],
            [y],
            xerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
            fmt="o",
            color=paper_palette(1)[0],
            capsize=3,
        )
    ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("prefill-3 (med): F steered − F null (95% CI)")
    fig.tight_layout()
    savefig_paper(fig, "model_compare_prefill3", dir=str(FIG_DIR))
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2333 figures.")
    ap.add_argument("--model-tags", nargs="+", default=["q25"], choices=("q25", "q35"))
    ap.add_argument("--import-check", action="store_true")
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
    for tag in args.model_tags:
        data = _load_tag(tag)
        fig_hero(data, tag)
        fig_recovery(data, tag)
        fig_perpair(data, tag)
        fig_whole_vs_continuation(data, tag)
        fig_coherence(data, tag)
    if len(args.model_tags) > 1:
        fig_model_compare(args.model_tags)
    print(f"[figures] wrote figures for {args.model_tags} under {FIG_DIR}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
