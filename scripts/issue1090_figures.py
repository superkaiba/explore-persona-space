#!/usr/bin/env python
"""#1090 figures (plan §6): hero yield-vs-floor, contrast panels, dose curves,
install lift. Called by ``issue1090_run.py --phase judge-aggregate`` with the
aggregation dir + figure dir (smoke passes scratch mirrors — committed
``figures/issue_1090/`` is only written by the full run).

paper-plots conventions: ``set_paper_style("blog")``, colorblind-safe palette,
Wilson error bars, no text-overlay annotations.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1090.figures")

# Plain-English condition names only — never bare cell codes (C1/C2/...) on
# chart elements (project rule: opaque condition codes stay in the Repro footer).
CELL_LABEL = {
    "c1": "formatting control\n(Claude, curated)",
    "c2": "impolite\n(Claude, auto-gen)",
    "c3": "sycophancy\n(Claude, neutral)",
    "c4": "sycophancy\n(Claude, wrong-fact)",
    "c5": "sycophancy\n(Qwen, neutral)",
    "c6": "broad misalignment\n(Claude, neutral)",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell_id(slug: str) -> str:
    return slug.split("-", 1)[0]


def _assert_yield_rows_first_sample_only(ys: dict) -> None:
    """Amendment v4 guard (figure side): every ``success_with_topup`` yield row
    must carry the aggregator's first-sample-only marker — top-up rows never
    enter the yield series a figure plots."""
    for slug, r in ys.items():
        if r.get("status") == "success_with_topup" and r.get("yield_source") != (
            "first_sample_only"
        ):
            raise RuntimeError(
                f"{slug}: success_with_topup yield row without the first_sample_only marker — "
                "refusing to plot a possibly union-contaminated yield series (amendment v4)"
            )


def fig_yield_vs_floor(agg: Path, figdir: Path) -> str | None:
    """HERO: per-cell judge-accepted fraction vs its yield floor (Wilson CIs)."""
    ys_path = agg / "yield_summary.json"
    if not ys_path.exists():
        return None
    ys = _read_json(ys_path)
    _assert_yield_rows_first_sample_only(ys)
    rows = [
        (slug, r)
        for slug, r in sorted(ys.items(), key=lambda kv: _cell_id(kv[0]))
        if r.get("kept") is not None and r.get("requested")
    ]
    if not rows:
        return None
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    cols = paper_palette(max(3, len(rows)))
    xs = np.arange(len(rows))
    for i, (_slug, r) in enumerate(rows):
        frac = r["kept"] / r["requested"]
        lo, hi = r.get("wilson95", (frac, frac))
        ax.bar(i, frac, color=cols[i % len(cols)], width=0.62)
        ax.errorbar(
            i,
            frac,
            yerr=[[max(0.0, frac - lo)], [max(0.0, hi - frac)]],
            fmt="none",
            ecolor="0.25",
            capsize=4,
            lw=1.4,
        )
        ax.text(
            i,
            min(hi + 0.03, 1.02),
            f"{r['kept']}/{r['requested']}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        # Per-cell yield floor tick (floor_n of requested).
        if r.get("floor_n") and r.get("requested"):
            fl = r["floor_n"] / r["requested"]
            ax.hlines(fl, i - 0.38, i + 0.38, color="0.15", ls="--", lw=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL.get(_cell_id(s), s) for s, _ in rows], fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("judge-accepted fraction of generated positives")
    ax.set_title("Per-cell datagen yield vs floor (dashed = 0.8-of-target floor)", pad=10)
    return savefig_paper(fig, "hero_yield_vs_floor", dir=figdir)["png"]


def _per_question_rates(pq: dict) -> tuple[dict[str, float], int]:
    """Per-question kept/judged rates, SKIPPING judged==0 questions.

    Drop-never-coerce (same exclusion semantics as ``issue1090_run.py``'s
    paired read): a zero-judged question carries no rate information and must
    not plot as a 0.0 dot. Returns ``(rates, n_excluded_zero_judged)``.
    """
    rates = {qid: d["kept"] / d["judged"] for qid, d in pq.items() if d["judged"] > 0}
    return rates, len(pq) - len(rates)


def fig_contrast_panels(agg: Path, figdir: Path) -> str | None:
    """HERO 2: C3-vs-C4 (bank delta) + C3-vs-C5 (generator delta) panels —
    cell bars + per-question dots (the low-level data behind each bar)."""
    ys_path = agg / "yield_summary.json"
    if not ys_path.exists():
        return None
    ys = _read_json(ys_path)
    _assert_yield_rows_first_sample_only(ys)
    by_id = {_cell_id(slug): (slug, r) for slug, r in ys.items()}
    if "c3" not in by_id or not ({"c4", "c5"} & set(by_id)):
        return None  # no contrast partner in this run's cell subset
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    rng = np.random.default_rng(42)
    panels = [
        (axes[0], "c4", "Bank contrast: neutral vs wrong-fact bank (same generator)"),
        (axes[1], "c5", "Generator contrast: Claude vs Qwen (same neutral bank)"),
    ]
    cols = paper_palette(4)
    excluded: dict[str, int] = {}
    for ax, other, title in panels:
        if other not in by_id:
            ax.set_axis_off()
            continue
        pair = [("c3", by_id["c3"]), (other, by_id[other])]
        for i, (cid, (_slug, r)) in enumerate(pair):
            if r.get("kept") is None or not r.get("requested"):
                continue
            frac = r["kept"] / r["requested"]
            lo, hi = r.get("wilson95", (frac, frac))
            ax.bar(i, frac, color=cols[i], width=0.55, alpha=0.85)
            ax.errorbar(
                i,
                frac,
                yerr=[[max(0.0, frac - lo)], [max(0.0, hi - frac)]],
                fmt="none",
                ecolor="0.25",
                capsize=4,
                lw=1.4,
            )
            rates_by_q, n_excl = _per_question_rates(r.get("per_question_yield", {}))
            if n_excl:
                excluded[cid] = n_excl  # idempotent — c3 recurs across both panels
            rates = list(rates_by_q.values())
            if rates:
                jit = rng.uniform(-0.14, 0.14, size=len(rates))
                ax.plot(i + jit, rates, "o", ms=4, color="0.2", alpha=0.55, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([CELL_LABEL.get(c, c) for c, _ in pair], fontsize=9.5)
        ax.set_ylim(-0.04, 1.12)
        ax.set_ylabel("kept fraction (dots: per question)")
        ax.set_title(title, fontsize=12, pad=10)
    paths = savefig_paper(fig, "hero_contrast_panels", dir=figdir)
    if excluded:
        # Meta-sidecar note only (no on-plot text overlays, per project style):
        # per-cell count of zero-judged questions dropped from the dot series.
        meta = json.loads(paths["meta"].read_text())
        meta["excluded_zero_judged_questions"] = excluded
        paths["meta"].write_text(json.dumps(meta, indent=2) + "\n")
    return paths["png"]


def fig_dose_curves(agg: Path, figdir: Path) -> str | None:
    """Per-organism Tier-1 dose curves (judged rate vs checkpoint step; band
    shaded; selected rung circled)."""
    curves = sorted((agg / "install").glob("*_dose_curve.json"))
    if not curves:
        return None
    set_paper_style("blog")
    n = len(curves)
    fig, axes = plt.subplots(
        1, n, figsize=(3.6 * n + 1.5, 4.4), constrained_layout=True, squeeze=False
    )
    cols = paper_palette(max(3, n))
    for k, path in enumerate(curves):
        d = _read_json(path)
        ax = axes[0][k]
        steps = sorted(int(s) for s in d["rates_by_step"])
        rates = [d["rates_by_step"][str(s)] for s in steps]
        lo, hi = d["band"]
        ax.axhspan(lo, hi, color="0.9", zorder=0)
        ax.plot(steps, rates, "-o", ms=5, color=cols[k % len(cols)])
        sel = d.get("selection") or {}
        if sel.get("step") is not None:
            ax.plot(
                [sel["step"]], [sel["rate"]], "o", ms=11, mfc="none", mec="0.1", mew=1.8, zorder=4
            )
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("optimizer step (save_steps=2)")
        if k == 0:
            ax.set_ylabel("source judged rate (Tier 1)")
        ax.set_title(
            CELL_LABEL.get(_cell_id(d["cell"]), d["cell"]).replace("\n", " "), fontsize=10.5
        )
    return savefig_paper(fig, "dose_curves", dir=figdir)["png"]


def fig_install_lift(agg: Path, figdir: Path) -> str | None:
    """Selected-checkpoint install read: base vs trained judged rate per cell
    (Wilson CIs) + the tf-margin delta companion in a second panel."""
    installs = sorted((agg / "install").glob("*_install.json"))
    if not installs:
        return None
    recs = [_read_json(p) for p in installs]
    recs = [r for r in recs if "trained" in r.get("reads", {}) and "base" in r.get("reads", {})]
    if not recs:
        return None
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    ax = axes[0]
    xs = np.arange(len(recs))
    w = 0.36
    cols = paper_palette(3)
    for i, r in enumerate(recs):
        for off, state, col in ((-w / 2, "base", cols[0]), (w / 2, "trained", cols[1])):
            rd = r["reads"][state]
            lo, hi = rd["wilson95"]
            ax.bar(i + off, rd["rate"], width=w, color=col)
            ax.errorbar(
                i + off,
                rd["rate"],
                yerr=[[max(0.0, rd["rate"] - lo)], [max(0.0, hi - rd["rate"])]],
                fmt="none",
                ecolor="0.25",
                capsize=3,
                lw=1.2,
            )
    band = recs[0].get("band")
    if band:
        ax.axhspan(band[0], band[1], color="0.92", zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL.get(_cell_id(r["cell"]), r["cell"]) for r in recs], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("own-persona judged rate (Tier 2)")
    ax.set_title("Install: base (left) vs selected checkpoint (right); band shaded")
    ax2 = axes[1]
    deltas = [r.get("margin_delta") for r in recs]
    for i, (_r, d) in enumerate(zip(recs, deltas, strict=True)):
        if d is None:
            continue
        ax2.bar(i, d, color=cols[2], width=0.55)
    ax2.axhline(0.0, color="0.2", lw=1.0)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([CELL_LABEL.get(_cell_id(r["cell"]), r["cell"]) for r in recs], fontsize=9)
    ax2.set_ylabel("tf-margin delta (trained - base)")
    ax2.set_title("Companion: teacher-forced fixed ± margin delta")
    return savefig_paper(fig, "install_lift", dir=figdir)["png"]


def make_all(agg_root: Path, fig_root: Path) -> dict[str, str]:
    """Every figure whose inputs exist; returns name -> png path."""
    agg_root, fig_root = Path(agg_root), Path(fig_root)
    fig_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name, fn in (
        ("hero_yield_vs_floor", fig_yield_vs_floor),
        ("hero_contrast_panels", fig_contrast_panels),
        ("dose_curves", fig_dose_curves),
        ("install_lift", fig_install_lift),
    ):
        try:
            path = fn(agg_root, fig_root)
        finally:
            plt.close("all")
        if path is not None:
            out[name] = str(path)  # savefig_paper returns Path — JSON-safe str
            logger.info("[figures] %s -> %s", name, path)
        else:
            logger.info("[figures] %s skipped (inputs absent)", name)
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--agg-root", required=True)
    p.add_argument("--fig-root", required=True)
    a = p.parse_args()
    make_all(Path(a.agg_root), Path(a.fig_root))
