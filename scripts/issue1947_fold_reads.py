#!/usr/bin/env python
"""#1947 fold round (VM, 0 GPU): the sycophancy-arm reads the body fold needs.

The inline override round (2026-08-02) recovered the sycophancy arm, trained
its 18 cells, judged their ladders, ran the P4/P5 capture-fit legs and a
fleet-wide P6 over all 56 verdict arms. The parked clean-result body was
written against the 38-arm fleet, so this script produces the three things the
fold needs that the round did not leave behind:

1. ``--phase intrusion`` — the Step 3.7 CJK language-intrusion scan for the
   NEW sycophancy judged pool at each cell's VERDICT rung (the rung the install
   headline rests on). Reuses the committed detection rule (``cjk_jp_kr``,
   calibrated in ``intrusion_recount_all_rungs.json``) and the production
   helpers from ``issue1947_intrusion_recount`` (``stage_ladder``,
   ``item_texts``, ``recount_unit``) so the syc rows are computed by the SAME
   code path as the committed impolite/casual rows. Writes a NEW file
   (``syc_intrusion_verdict_rungs.json``) — the committed 34-cell recount is
   never rewritten.

2. ``--phase samples`` — verbatim sample rows for the body's
   ``**Sample training/evaluation data + completions:**`` slot: judge-firing /
   non-firing completions at a sycophancy verdict rung (random, seed 42) and
   the first row of the recovered ``syc-icl`` positives pool. Text goes to a
   /tmp digest, never into a committed artifact.

3. ``--phase figures`` — the fold's figures, regenerated fleet-wide from the
   committed 52-content-cell artifacts under NEW stems (the parked body's
   figures plot the 34-cell fleet and the P6 driver's three PNGs carry no
   ``savefig_paper`` sidecar).

Every read is a committed local artifact except the ladder rollout text, which
is Hub-staged on a local miss.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue1947.fold")

ANALYSIS = REPO_ROOT / "eval_results/issue_1947/analysis"
FIGDIR = REPO_ROOT / "figures/issue_1947"
STAGE = REPO_ROOT / "data/issue_1947/battery_stage"
LAYERS = (14, 19, 25)
FAMS = ("imp", "cas", "syc")
FAM_LABEL = {"imp": "impolite", "cas": "casual style", "syc": "sycophancy"}
BAND = (0.60, 0.85)
RANK_CRIT = 0.6
RELIABILITY_CRIT = 0.55
HF_DATA = "superkaiba1/explore-persona-space-data"
DATA_PREFIX = "issue1947_singlevisit"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _fam(slug: str) -> str:
    return slug.split("-")[0]


def _battery_cells() -> list[dict]:
    return [_read_json(p) for p in sorted((ANALYSIS / "battery").glob("battery_*.json"))]


def _get(node: object, *keys: str) -> object | None:
    cur = node
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------
# phase: intrusion (Step 3.7 scan for the NEW sycophancy judged pool)
# --------------------------------------------------------------------------


def phase_intrusion() -> int:
    """Verdict-rung CJK scan over the 18 sycophancy cells, committed rule."""
    import issue1947_intrusion_recount as rec

    from explore_persona_space.artifacts.organisms import BEHAVIORS

    committed = _read_json(ANALYSIS / "intrusion_recount_all_rungs.json")
    rule_name = committed["detection_rule"]["name"]
    rx = re.compile(next(p for n, p, _ in rec.CANDIDATE_RULES if n == rule_name))
    manifest = _read_json(ANALYSIS / "verdict_manifest.json")
    slugs = sorted(s for s in manifest["content"] if _fam(s) == "syc")
    rows: dict[str, dict] = {}
    for i, slug in enumerate(slugs, start=1):
        payload = _read_json(rec.stage_ladder(STAGE, slug))
        judged = _read_json(ANALYSIS / "judge" / f"judged_{slug}.json")
        threshold = float(BEHAVIORS[judged["behavior"]].threshold)
        step_s = str(manifest["content"][slug]["selection"]["step"])
        row = rec.recount_unit(
            payload, step_s, ANALYSIS / "judge" / "syc", judged["instrument"], threshold, rx
        )
        committed_rate = judged["rates_by_step"].get(step_s)
        row["raw_rate_matches_committed"] = (
            committed_rate is not None
            and row["rate_raw"] is not None
            and abs(row["rate_raw"] - committed_rate) < 1e-9
        )
        # whole-cell scan (all 15 rungs) for the per-cell intrusion prevalence
        row["intr_all_rungs"] = sum(
            1
            for s in payload["rungs"]
            for _iid, _q, comp in rec.item_texts(payload, s)
            if rx.search(comp)
        )
        row["n_all_rungs"] = sum(len(rec.item_texts(payload, s)) for s in payload["rungs"])
        rows[slug] = row
        print(f"[fold-intrusion] {i}/{len(slugs)} {slug} r{step_s} {row}", flush=True)
    n_scored = sum(r["n_scored"] for r in rows.values())
    out = {
        "detection_rule": committed["detection_rule"],
        "conventions": committed["conventions"],
        "scope": (
            "sycophancy verdict rungs only (18 cells x 1 rung); the committed "
            "intrusion_recount_all_rungs.json covers the 34 impolite/casual cells x 15 rungs"
        ),
        "cells": rows,
        "totals": {
            "n_cells": len(rows),
            "n_scored_verdict_rungs": n_scored,
            "intr_verdict_rungs": sum(r["intr"] for r in rows.values()),
            "fired_intr_verdict_rungs": sum(r["fired_intr"] for r in rows.values()),
            "n_all_rungs": sum(r["n_all_rungs"] for r in rows.values()),
            "intr_all_rungs": sum(r["intr_all_rungs"] for r in rows.values()),
            "n_rate_mismatch": sum(1 for r in rows.values() if not r["raw_rate_matches_committed"]),
            "in_band_raw": sum(1 for r in rows.values() if BAND[0] <= r["rate_raw"] <= BAND[1]),
            "in_band_excl": sum(
                1
                for r in rows.values()
                if r["rate_excl"] is not None and BAND[0] <= r["rate_excl"] <= BAND[1]
            ),
            "in_band_zeroed": sum(
                1
                for r in rows.values()
                if r["rate_zeroed"] is not None and BAND[0] <= r["rate_zeroed"] <= BAND[1]
            ),
        },
        "issue": 1947,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(ANALYSIS / "syc_intrusion_verdict_rungs.json", out)
    print(f"[fold-intrusion] totals: {json.dumps(out['totals'])}", flush=True)
    assert out["totals"]["n_rate_mismatch"] == 0, "raw-rate cross-check against judged_* FAILED"
    return 0


# --------------------------------------------------------------------------
# phase: samples (verbatim rows for the body's sample slot)
# --------------------------------------------------------------------------


def phase_samples(slug: str, out_path: Path) -> int:
    import issue1947_intrusion_recount as rec

    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw
    from explore_persona_space.orchestrate import hub

    manifest = _read_json(ANALYSIS / "verdict_manifest.json")
    sel = manifest["content"][slug]["selection"]
    step_s = str(sel["step"])
    payload = _read_json(rec.stage_ladder(STAGE, slug))
    judged = _read_json(ANALYSIS / "judge" / f"judged_{slug}.json")
    items = rec.item_texts(payload, step_s)
    result = judge_result_from_save_raw(
        ANALYSIS / "judge" / "syc" / judged["instrument"] / f"judge_raw_{slug}-r{step_s}.json",
        items,
    )
    scored = [(iid, q, c, result.scores.get(iid)) for iid, q, c in items]
    scored = [r for r in scored if r[3] is not None]
    firing = [r for r in scored if r[3] > 50.0]
    nonfiring = [r for r in scored if r[3] <= 50.0]
    rng = random.Random(42)
    lines: list[str] = [
        f"# slug={slug} verdict_rung={step_s} rate={sel['rate']} "
        f"n_scored={len(scored)} n_firing={len(firing)} n_nonfiring={len(nonfiring)}"
    ]
    for label, pool in (("FIRING", firing), ("NON-FIRING", nonfiring)):
        for iid, q, comp, score in rng.sample(pool, min(3, len(pool))):
            lines.append(f"\n=== {label} {iid} judge_mean={score}\nQ: {q}\nA: {comp}\n")
    # first row of the recovered syc-icl positives pool
    pos_local = STAGE / "recovered_pos" / "pos.jsonl"
    if not pos_local.exists():
        hub.stage_hub_file(
            HF_DATA,
            f"{DATA_PREFIX}/raw_completions/datagen/positives/syc-icl/pos.jsonl",
            pos_local,
            repo_type="dataset",
        )
    with pos_local.open(encoding="utf-8") as fh:  # never splitlines() on JSONL
        first = json.loads(next(line for line in fh if line.strip()))
    lines.append(f"\n=== RECOVERED syc-icl POSITIVES row 0 keys={sorted(first)}\n")
    lines.append(json.dumps(first, ensure_ascii=False, indent=1))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[fold-samples] wrote {out_path} ({out_path.stat().st_size} B)", flush=True)
    return 0


# --------------------------------------------------------------------------
# phase: figures
# --------------------------------------------------------------------------


def _fam_colors() -> dict[str, str]:
    return {"imp": "#0072B2", "cas": "#D55E00", "syc": "#009E73"}


CTX_LABEL = {
    "pers": "persona",
    "bare": "bare assistant",
    "conv": "conversation",
    "icl": "in-context",
}
REGIME_LABEL = {"con": "contrastive", "po": "positive-only"}


def _plain_cell_label(slug: str) -> str:
    """Reader-facing cell label: no context/regime slugs (clean-result Lens 2)."""
    _beh, ctx, regime, visit, seed = slug.split("-")
    prefix = "repeat " if visit == "rep" else ""
    return f"{prefix}{CTX_LABEL[ctx]}, {REGIME_LABEL[regime]}, seed {seed.lstrip('s')}"


def _strip_x(fam: str, i: int, n: int) -> float:
    base = FAMS.index(fam)
    return base + (i - (n - 1) / 2) * (0.55 / max(n - 1, 1))


def fig_install_ladders(manifest: dict) -> None:
    """52 per-cell judged ladders, one column per behavior, band + verdict dot."""
    cols = _fam_colors()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
    for ax, fam in zip(axes, FAMS):
        slugs = sorted(s for s in manifest["content"] if _fam(s) == fam)
        for slug in slugs:
            judged = _read_json(ANALYSIS / "judge" / f"judged_{slug}.json")
            steps = sorted(int(s) for s in judged["rates_by_step"])
            rates = [judged["rates_by_step"][str(s)] for s in steps]
            ax.plot(steps, rates, "-", color=cols[fam], alpha=0.4, linewidth=1.0)
            sel = manifest["content"][slug]["selection"]
            ax.plot(
                sel["step"],
                sel["rate"],
                "o",
                color=cols[fam] if sel["in_band"] else "white",
                markeredgecolor=cols[fam],
                markeredgewidth=1.4,
                markersize=6,
                zorder=5,
            )
        ax.axhspan(BAND[0], BAND[1], color="0.75", alpha=0.35, zorder=0)
        n_in = sum(1 for s in slugs if manifest["content"][s]["selection"]["in_band"])
        ax.set_title(f"{FAM_LABEL[fam]} — {n_in} of {len(slugs)} cells in band", fontsize=11)
        ax.set_xlabel("optimizer step")
    axes[0].set_ylabel("judged behavior rate\n(fraction of 100 completions)")
    axes[0].set_ylim(-0.03, 1.03)
    handles = [
        plt.Line2D([], [], marker="o", color="0.3", linestyle="", label="verdict rung, in band"),
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor="0.3",
            color="0.3",
            linestyle="",
            label="verdict rung, closest approach",
        ),
        plt.Rectangle((0, 0), 1, 1, color="0.75", alpha=0.5, label="target band 0.60-0.85"),
    ]
    axes[2].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "fold_install_ladders_52", dir=FIGDIR)
    plt.close(fig)


def fig_intrusion_verdict(manifest: dict) -> None:
    """Verdict-rung rates under the three intrusion conventions, 52 cells."""
    committed = _read_json(ANALYSIS / "intrusion_recount_all_rungs.json")["cells"]
    syc = _read_json(ANALYSIS / "syc_intrusion_verdict_rungs.json")["cells"]
    cols = _fam_colors()
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    xs: list[float] = []
    labels: list[str] = []
    groups: list[tuple[str, float, float]] = []
    x = 0.0
    for fam in FAMS:
        slugs = sorted(s for s in manifest["content"] if _fam(s) == fam)
        groups.append((FAM_LABEL[fam], x, x + len(slugs) - 1))
        for slug in slugs:
            if fam == "syc":
                row = syc[slug]
            else:
                step = str(manifest["content"][slug]["selection"]["step"])
                row = committed[slug]["rates_by_step"][step]
            ax.plot([x, x], [row["rate_zeroed"], row["rate_raw"]], "-", color="0.7", linewidth=1.0)
            ax.plot(x, row["rate_raw"], "o", color=cols[fam], markersize=5)
            ax.plot(
                x,
                row["rate_excl"],
                "s",
                color="white",
                markeredgecolor=cols[fam],
                markeredgewidth=1.2,
                markersize=4.5,
            )
            ax.plot(x, row["rate_zeroed"], "v", color="0.35", markersize=4.0)
            xs.append(x)
            labels.append(_plain_cell_label(slug))
            x += 1.0
        x += 1.2
    ax.axhspan(BAND[0], BAND[1], color="0.75", alpha=0.35, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2)
    ax.set_ylabel("judged rate at the verdict rung")
    ax.set_ylim(-0.03, 1.03)
    for name, lo, hi in groups:
        ax.text(
            (lo + hi) / 2,
            1.045,
            name,
            ha="center",
            va="bottom",
            fontsize=10,
            color=cols[next(f for f in FAMS if FAM_LABEL[f] == name)],
        )
    handles = [
        plt.Line2D([], [], marker="o", color="0.3", linestyle="", label="as scored"),
        plt.Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.5,
            markeredgecolor="0.3",
            color="0.3",
            linestyle="",
            label="intruded rows excluded",
        ),
        plt.Line2D([], [], marker="v", color="0.35", linestyle="", label="intruded rows zeroed"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    savefig_paper(fig, "fold_intrusion_verdict_52", dir=FIGDIR)
    plt.close(fig)


def _strip_panel(ax, cells_by_fam, value_fn, crit=None, band_fn=None) -> None:
    cols = _fam_colors()
    for fam in FAMS:
        ds = cells_by_fam.get(fam, [])
        vals = [(d["slug"], value_fn(d)) for d in ds]
        vals = [(s, v) for s, v in vals if v is not None]
        for i, (_slug, v) in enumerate(sorted(vals, key=lambda t: t[0])):
            ax.plot(_strip_x(fam, i, len(vals)), v, "o", color=cols[fam], markersize=4, alpha=0.85)
        if vals:
            med = st.median([v for _, v in vals])
            ax.plot(
                [FAMS.index(fam) - 0.32, FAMS.index(fam) + 0.32],
                [med, med],
                "-",
                color="black",
                linewidth=1.6,
                zorder=6,
            )
        if band_fn is not None:
            bvals = [band_fn(d) for d in ds]
            bvals = [b for b in bvals if b is not None]
            if bvals:
                bm = st.median(bvals)
                ax.plot(
                    [FAMS.index(fam) - 0.32, FAMS.index(fam) + 0.32],
                    [bm, bm],
                    "--",
                    color="0.45",
                    linewidth=1.1,
                    zorder=4,
                )
    if crit is not None:
        ax.axhline(crit, color="crimson", linestyle=":", linewidth=1.2)
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([FAM_LABEL[f] for f in FAMS], fontsize=9)
    ax.set_xlim(-0.6, len(FAMS) - 0.4)


def fig_rank(cells: list[dict]) -> None:
    """Top-1 singular share per cell, both trees x three layers, 52 content cells."""
    content = [c for c in cells if _fam(c["slug"]) in FAMS]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey="row")
    for r, tree in enumerate(("onpolicy", "matched_text")):
        for c, layer in enumerate(LAYERS):
            ax = axes[r][c]
            byf: dict[str, list[dict]] = {}
            for d in content:
                if d["layer"] == layer:
                    byf.setdefault(_fam(d["slug"]), []).append(d)
            _strip_panel(ax, byf, lambda d, t=tree: _get(d, t, "rank", "top1_var_share"), RANK_CRIT)
            if r == 0:
                ax.set_title(f"layer {layer}", fontsize=11)
    axes[0][0].set_ylabel("top-1 singular share\n(on-policy tree)")
    axes[1][0].set_ylabel("top-1 singular share\n(matched-text tree)")
    fig.text(
        0.5,
        0.005,
        "red dotted line = the 0.6 rank-one criterion; black bars = per-family medians; "
        "one point per cell",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savefig_paper(fig, "fold_rank_top1_52", dir=FIGDIR)
    plt.close(fig)


def fig_rb_dissociation(cells: list[dict]) -> None:
    """cos(w, delta) vs cos(w, r_B) per cell, both trees x three layers."""
    content = [c for c in cells if _fam(c["slug"]) in FAMS]
    cols = _fam_colors()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), sharey="row", sharex="row")
    for r, tree in enumerate(("onpolicy", "matched_text")):
        for c, layer in enumerate(LAYERS):
            ax = axes[r][c]
            for fam in FAMS:
                ds = [d for d in content if d["layer"] == layer and _fam(d["slug"]) == fam]
                xs = [_get(d, tree, "alignments", "delta", "cos") for d in ds]
                ys = [_get(d, tree, "alignments", "r_b", "cos") for d in ds]
                pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
                if not pts:
                    continue
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o",
                    color=cols[fam],
                    markersize=4.5,
                    alpha=0.85,
                    label=FAM_LABEL[fam] if (r == 0 and c == 0) else "_",
                )
                mx, my = st.mean([p[0] for p in pts]), st.mean([p[1] for p in pts])
                ax.text(mx, my, f" {fam}", fontsize=8, color=cols[fam], va="center")
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], "--", color="0.6", linewidth=1.0, zorder=0)
            ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
            if r == 0:
                ax.set_title(f"layer {layer}", fontsize=11)
            if r == 1:
                ax.set_xlabel("cosine with the displacement unit")
    axes[0][0].set_ylabel("cosine with the behavior\ndirection (on-policy tree)")
    axes[1][0].set_ylabel("cosine with the behavior\ndirection (matched-text tree)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.text(
        0.5,
        0.005,
        "grey dashed line = equal alignment with both candidate directions; one point per cell",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savefig_paper(fig, "fold_rb_dissociation_52", dir=FIGDIR)
    plt.close(fig)


def fig_h3_split(cells: list[dict]) -> None:
    """Matched-text cos(w, delta) by behavior at L19/L25 with per-cell nulls."""
    content = [c for c in cells if _fam(c["slug"]) in FAMS]
    h3 = {
        (r["slug"], r["layer"]): r["cos_20row_mean"]
        for r in _read_json(ANALYSIS / "frame_h3-20row.json")["rows"]
    }
    cols = _fam_colors()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for fam in FAMS:
            ds = sorted(
                [d for d in content if d["layer"] == layer and _fam(d["slug"]) == fam],
                key=lambda d: d["slug"],
            )
            for i, d in enumerate(ds):
                x = _strip_x(fam, i, len(ds))
                v = _get(d, "matched_text", "alignments", "delta", "cos")
                nb = _get(
                    d,
                    "matched_text",
                    "alignments",
                    "delta",
                    "null_bands",
                    "corpus_covariance",
                    "p97_5",
                )
                if v is not None:
                    ax.plot(x, v, "o", color=cols[fam], markersize=4.5)
                m20 = h3.get((d["slug"], layer))
                if m20 is not None:
                    ax.plot(
                        x,
                        m20,
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=cols[fam],
                        markeredgewidth=1.0,
                        markersize=6.5,
                    )
                if nb is not None:
                    ax.plot([x - 0.012, x + 0.012], [nb, nb], "-", color="0.45", linewidth=1.0)
            above = sum(
                1
                for d in ds
                if (_get(d, "matched_text", "alignments", "delta", "cos") or -9)
                > (
                    _get(
                        d,
                        "matched_text",
                        "alignments",
                        "delta",
                        "null_bands",
                        "corpus_covariance",
                        "p97_5",
                    )
                    or 9
                )
            )
            ax.text(
                FAMS.index(fam),
                -0.32,
                f"{above}/{len(ds)}",
                ha="center",
                fontsize=8.5,
                color=cols[fam],
            )
        ax.axhline(0, color="0.8", linewidth=0.8)
        ax.set_xticks(range(len(FAMS)))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMS], fontsize=9)
        ax.set_xlim(-0.6, len(FAMS) - 0.4)
        ax.set_title(f"layer {layer}", fontsize=11)
    axes[0].set_ylabel("matched-text cosine with\nthe displacement unit")
    axes[0].set_ylim(-0.38, 0.62)
    handles = [
        plt.Line2D([], [], marker="o", color="0.3", linestyle="", label="full 1,200-row read"),
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.5,
            markeredgecolor="0.3",
            color="0.3",
            linestyle="",
            label="mean of twenty 20-row draws",
        ),
        plt.Line2D([], [], color="0.45", label="per-cell covariance null upper bound"),
    ]
    axes[2].legend(handles=handles, fontsize=7.5, loc="upper right")
    fig.text(
        0.5,
        0.005,
        "numbers under each family = cells above their own covariance null",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savefig_paper(fig, "fold_h3_behavior_split_52", dir=FIGDIR)
    plt.close(fig)


def fig_d_forest() -> None:
    """36 span-mean map-change rows with CIs; parent comparator where extracted."""
    summary = _read_json(ANALYSIS / "battery_summary.json")
    parent = {
        (r["arm"], r["layer"]): r
        for r in _read_json(ANALYSIS / "analyzer_round1/h5_concordance.json")
    }
    rows = [r for r in summary["d_forest"] if "lasttoken" not in r["file"]]
    rows.sort(key=lambda r: (FAMS.index(_fam(r["arm"])), r["arm"], r["layer"]))
    cols = _fam_colors()
    fig, ax = plt.subplots(figsize=(8.8, 10.5))
    yl = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        yl.append((y, f"{r['arm'].replace('-con-sv-s42', '')} L{r['layer']}"))
        d, ci = r["D"], r["D_ci95"]
        ax.errorbar(
            d,
            y + 0.16,
            xerr=[[d - ci[0]], [ci[1] - d]],
            fmt="o",
            color=cols[_fam(r["arm"])],
            markersize=5,
            capsize=2,
        )
        p = parent.get((r["arm"], r["layer"]))
        if p and p.get("D_1768") is not None:
            pd_, pci = p["D_1768"], p["D_ci95_1768"]
            ax.errorbar(
                pd_,
                y - 0.16,
                xerr=[[pd_ - pci[0]], [pci[1] - pd_]],
                fmt="s",
                color="0.45",
                markersize=4.5,
                capsize=2,
            )
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_yticks([y for y, _ in yl])
    ax.set_yticklabels([lab for _, lab in yl], fontsize=8.0)
    ax.set_xlabel("map-change D (excess over the base-map refit-noise p95 floor), span-mean read")
    handles = [
        plt.Line2D([], [], marker="o", color=cols[f], linestyle="", label=FAM_LABEL[f])
        for f in FAMS
    ] + [
        plt.Line2D(
            [], [], marker="s", color="0.45", linestyle="", label="parent fleet, same arm and rate"
        )
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "fold_d_forest_36", dir=FIGDIR)
    plt.close(fig)


def fig_h6(manifest: dict) -> None:
    """Repeat-regime top-1 share vs n-matched single-visit draws, 4 pairs."""
    h6 = {
        (r["slug"], r["layer"]): r for r in _read_json(ANALYSIS / "frame_h6-n-match.json")["rows"]
    }
    cells = {(c["slug"], c["layer"]): c for c in _battery_cells()}
    pairs = [(s, s.replace("-rep-", "-sv-")) for s in sorted(manifest["content"]) if "-rep-" in s]
    cols = _fam_colors()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        for i, (rep, sib) in enumerate(pairs):
            draws = h6[(sib, layer)]["top1_share_subsampled"]
            ax.plot([i] * len(draws), draws, "o", color="0.65", markersize=3.5)
            ax.plot(i, st.mean(draws), "_", color="0.3", markersize=18, markeredgewidth=2)
            full = _get(cells[(sib, layer)], "matched_text", "rank", "top1_var_share")
            ax.plot(
                i,
                full,
                "o",
                markerfacecolor="none",
                markeredgecolor="0.3",
                markeredgewidth=1.2,
                markersize=8,
            )
            ax.plot(
                i, h6[(rep, layer)]["top1_share_mean"], "D", color=cols[_fam(rep)], markersize=7
            )
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([p[0].replace("-con-rep-s42", "") for p in pairs], fontsize=8.5)
        ax.set_xlim(-0.6, len(pairs) - 0.4)
        ax.set_title(f"layer {layer}", fontsize=11)
    axes[0].set_ylabel("matched-text top-1 singular share")
    handles = [
        plt.Line2D([], [], marker="D", color="0.3", linestyle="", label="repeat-regime cell"),
        plt.Line2D(
            [],
            [],
            marker="o",
            color="0.65",
            linestyle="",
            label="single-visit sibling, 80-row draws",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.5,
            markeredgecolor="0.3",
            color="0.3",
            linestyle="",
            label="sibling, full 1,200 rows",
        ),
    ]
    axes[2].legend(handles=handles, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "fold_h6_n_matched_4pairs", dir=FIGDIR)
    plt.close(fig)


def fig_reliability(cells: list[dict]) -> None:
    """Displacement split-half reliability, 156 content arm-layer cells."""
    content = [c for c in cells if _fam(c["slug"]) in FAMS]
    consumed = {
        (r["slug"], r["layer"]): r
        for r in _read_json(ANALYSIS / "frame_consumed-reliability.json")["rows"]
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    cols = _fam_colors()
    for ax, layer in zip(axes, LAYERS):
        byf: dict[str, list[dict]] = {}
        for d in content:
            if d["layer"] == layer:
                byf.setdefault(_fam(d["slug"]), []).append(d)
        for fam in FAMS:
            ds = sorted(byf.get(fam, []), key=lambda d: d["slug"])
            for i, d in enumerate(ds):
                x = _strip_x(fam, i, len(ds))
                ax.plot(x, d["delta_split_half_r_disjoint"], "o", color=cols[fam], markersize=4.5)
                cr = consumed.get((d["slug"], layer), {}).get("delta_consumed_split_half_r")
                if cr is not None:
                    ax.plot(
                        x,
                        cr,
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=cols[fam],
                        markeredgewidth=1.0,
                        markersize=6.5,
                    )
        ax.axhline(RELIABILITY_CRIT, color="crimson", linestyle=":", linewidth=1.3)
        ax.set_xticks(range(len(FAMS)))
        ax.set_xticklabels([FAM_LABEL[f] for f in FAMS], fontsize=9)
        ax.set_xlim(-0.6, len(FAMS) - 0.4)
        ax.set_title(f"layer {layer}", fontsize=11)
    axes[0].set_ylabel("displacement split-half reliability\n(disjoint halves)")
    handles = [
        plt.Line2D([], [], marker="o", color="0.3", linestyle="", label="all 300 mix positives"),
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgewidth=1.5,
            markeredgecolor="0.3",
            color="0.3",
            linestyle="",
            label="positives actually consumed by the verdict rung",
        ),
        plt.Line2D([], [], color="crimson", linestyle=":", label="0.55 criterion"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    savefig_paper(fig, "fold_delta_reliability_156", dir=FIGDIR)
    plt.close(fig)


def fig_yield_recovery() -> None:
    """The sycophancy datagen recovery: question-pool exhaustion + tranche yield."""
    audit = _read_json(REPO_ROOT / "eval_results/issue_1947/syc_recovery/recovery_audit.json")
    record = _read_json(REPO_ROOT / "eval_results/issue_1947/syc_recovery/recovery_record.json")
    ybac = audit["question_pool"]["yield_by_attempt_count"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ks = sorted(int(k) for k in ybac)
    frac = [ybac[str(k)]["n_yielded"] / ybac[str(k)]["n_questions"] for k in ks]
    ns = [ybac[str(k)]["n_questions"] for k in ks]
    ax.plot(ks, frac, "o-", color="#0072B2", markersize=6)
    for k, f, n in zip(ks, frac, ns):
        ax.text(k, f + 0.03, f"n={n}", ha="center", fontsize=7.5, color="0.35")
    ax.set_xlabel("times the question was requested")
    ax.set_ylabel("fraction of those questions that\never yielded a kept positive")
    ax.set_ylim(0, 1.15)
    ax.set_title("Question-pool exhaustion before the recovery", fontsize=11)
    ax = axes[1]
    tr = record["tranches"]
    stages = ["first sample", "top-up tranche"] + [f"recovery tranche {t['tranche']}" for t in tr]
    accept = [p["per_request_accept_rate"] for p in audit["passes"]] + [
        t["per_request_accept_rate"] for t in tr
    ]
    kept = [p["n_kept"] for p in audit["passes"]] + [t["n_kept_rows"] for t in tr]
    xs = range(len(stages))
    ax.bar(xs, accept, color=["0.6", "0.6", "#009E73", "#009E73"])
    for x, a, k in zip(xs, accept, kept):
        ax.text(x, a + 0.006, f"{a:.1%}\n{k} kept", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(stages, fontsize=8.5, rotation=15)
    ax.set_ylabel("kept positives per generation request")
    ax.set_ylim(0, max(accept) * 1.35)
    ax.set_title("Per-request acceptance across the four passes", fontsize=11)
    fig.tight_layout()
    savefig_paper(fig, "fold_syc_yield_recovery", dir=FIGDIR)
    plt.close(fig)


def phase_figures() -> int:
    set_paper_style("blog")
    manifest = _read_json(ANALYSIS / "verdict_manifest.json")
    cells = _battery_cells()
    fig_yield_recovery()
    fig_install_ladders(manifest)
    fig_intrusion_verdict(manifest)
    fig_rank(cells)
    fig_rb_dissociation(cells)
    fig_h3_split(cells)
    fig_d_forest()
    fig_h6(manifest)
    fig_reliability(cells)
    print("[fold-figures] wrote 9 figures under figures/issue_1947/fold_*", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1947 fold-round sycophancy reads")
    p.add_argument("--phase", required=True, choices=("intrusion", "samples", "figures"))
    p.add_argument("--slug", default="syc-pers-con-sv-s42", help="samples phase: cell slug")
    p.add_argument("--out", default="/tmp/issue1947_fold_samples.txt")
    args = p.parse_args(argv)
    if args.phase == "intrusion":
        return phase_intrusion()
    if args.phase == "samples":
        return phase_samples(args.slug, Path(args.out))
    return phase_figures()


if __name__ == "__main__":
    raise SystemExit(main())
