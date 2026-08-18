#!/usr/bin/env python
"""Issue #2356 P8: render the hero + exploratory figure dump from the fits outputs.

Reads ONLY the JSON artifacts ``issue2356_fits.py`` (P4/P6/P7) writes under
``--eval-root`` (realized schema — the producing script is the source of truth):

- ``results/stats.json``                  pooled AUROC/balanced-acc + contrasts + ladder
                                          + LODO + advisory permutation + H1 lattice
                                          (the plan's "headline metrics" file)
- ``results/predictor_scores_{arm}.json`` per-row OOF scores + selection metadata
- ``results/map_diagnostics.json``        per-layer map R2 / identity+bias / kNN / spectra
- ``results/map_discrimination.json``     retrieval battery (acc@1 x 4 spaces, S2 gate,
                                          behavior split, NN-behavior-match)
- ``results/transfer.json``               cross-regime transfer (report-only)
- ``{arm}/labels.json`` ``{arm}/groups.json`` ``{arm}/splits.json`` (P4)
- ``corpus/armB_manifest.json``           per-row source (LODO per-direction read)

Every figure degrades gracefully: a figure whose required input JSON is absent
is SKIPPED with the reason recorded in ``captions.json`` (so the script runs on
the smoke slice and on partial harvests). A PRESENT-but-malformed input fails
loud (no silent defaults). Output: ``--fig-dir`` (default ``figures/issue_2356``)
via ``savefig_paper`` (paper rcParams, commit-pinned sidecars) + a
``captions.json`` carrying per-figure captions + provenance. Canvas discipline:
axes/ticks/legend/panel-titles only — every other fact lives in the caption.

Resume: per-figure fingerprint (input file shas + code sha + style/format
flags) in ``<fig-dir>/render_manifest.json``; a matching fingerprint with all
outputs present skips the render (``--no-resume`` forces). Bare output
existence is never a skip predicate.

``--selftest``: runs ``issue2356_fits._selftest`` (its synthetic 6-phase
fixture — the REAL producer writes the inputs, so the consumed schema cannot
drift from a hand-mirrored fixture), keeps the scratch, renders the full dump
from it into a temp dir, asserts non-empty renders, and additionally runs the
degenerate-input probe (an empty eval-root -> all figures skipped, captions
written, rc 0). Never writes under ``figures/`` in selftest mode.

Content hygiene: consumes NO prompt text (aggregate JSONs only; label/group
files carry shas + labels, never text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Single source of truth for arm/predictor names: the producing fits driver.
from issue2356_fits import (  # noqa: E402
    ARMS,
    PRED_3A,
    PRED_3B,
    PRED_ANS,
    PRED_ANS_RM,
    PRED_CTX,
    PRED_DIM,
    PRED_ISREW,
    PRED_JUDGE,
    PRED_LODO,
    PRED_PCA,
    PRED_TEXT,
    PRED_TEXT_NOIND,
)

logger = logging.getLogger("issue2356_figures")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)

ISSUE = 2356

ARM_LABEL = {"armA": "Arm A - harmful flip-pairs", "armB": "Arm B - over-refusal"}

# Hero order + plain-English predictor names (plan section 6 "Figures to produce").
HERO_ORDER: list[tuple[str, str]] = [
    (PRED_JUDGE, "judge (few-shot)"),
    (PRED_CTX, "ctx probe"),
    (PRED_3A, "mapped (generic)"),
    (PRED_3B, "mapped (in-domain)"),
    (PRED_ANS, "answer probe"),
    (PRED_PCA, "PCA-ctx control"),
    (PRED_TEXT, "text-surface (fitted)"),
]
PRED_NAME = dict(
    HERO_ORDER
    + [
        (PRED_DIM, "ctx diff-in-means"),
        (PRED_ANS_RM, "answer probe (rollout mean)"),
        (PRED_TEXT_NOIND, "text-surface (no indicators)"),
        (PRED_ISREW, "is-rewrite indicator"),
        (PRED_LODO, "ctx probe (LODO)"),
    ]
)
_HERO_PREDS = [p for p, _ in HERO_ORDER]
PRED_COLOR = dict(zip(_HERO_PREDS, paper_palette(len(_HERO_PREDS))))

BATTERY_SPACES = ["whitened_cosine", "raw_euclidean", "r2_cand_norm", "pearson"]
SPACE_NAME = {
    "whitened_cosine": "whitened cosine",
    "raw_euclidean": "raw euclidean",
    "r2_cand_norm": "r2 cand-norm",
    "pearson": "pearson",
}
SPACE_COLOR = dict(zip(BATTERY_SPACES, paper_palette(len(BATTERY_SPACES))))

CONTRASTS: list[tuple[str, str]] = [
    ("delta_int", "ctx - judge (delta_int)"),
    ("ctx_minus_text_surface", "ctx - text-surface"),
    ("map3a_minus_pca", "map3a - PCA"),
    ("map3b_minus_map3a", "map3b - map3a"),
    ("ans_minus_ctx", "answer - ctx"),
]

# Input registry: ctx key -> path relative to --eval-root.
INPUTS: dict[str, str] = {
    "stats": "results/stats.json",
    "diag": "results/map_diagnostics.json",
    "battery": "results/map_discrimination.json",
    "transfer": "results/transfer.json",
    "scores_armA": "results/predictor_scores_armA.json",
    "scores_armB": "results/predictor_scores_armB.json",
    "labels_armA": "armA/labels.json",
    "labels_armB": "armB/labels.json",
    "groups_armA": "armA/groups.json",
    "groups_armB": "armB/groups.json",
    "splits_armA": "armA/splits.json",
    "splits_armB": "armB/splits.json",
    "manifest_armB": "corpus/armB_manifest.json",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance() -> dict[str, Any]:
    meta = dict(as_metadata_dict(git_provenance()))
    meta["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["issue"] = ISSUE
    return meta


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with midrank ties (same convention as the fits driver)."""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y, dtype=np.int64)
    mask = np.isfinite(s)
    s, yy = s[mask], yy[mask]
    n1 = int(yy.sum())
    n0 = len(yy) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _roc_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) sweeping the threshold over the scores (higher score -> y=1)."""
    s = np.asarray(scores, dtype=np.float64)
    yy = np.asarray(y, dtype=np.int64)
    mask = np.isfinite(s)
    s, yy = s[mask], yy[mask]
    order = np.argsort(-s, kind="mergesort")
    yy = yy[order]
    tps = np.concatenate(([0], np.cumsum(yy)))
    fps = np.concatenate(([0], np.cumsum(1 - yy)))
    n1 = max(int(yy.sum()), 1)
    n0 = max(int(len(yy) - yy.sum()), 1)
    return fps / n0, tps / n1


def _ci_yerr(point: float, ci: list[float] | None) -> np.ndarray | None:
    if not ci or len(ci) != 2 or not all(np.isfinite(ci)):
        return None
    return np.array([[max(point - ci[0], 0.0)], [max(ci[1] - point, 0.0)]])


def _fold_keys(selection: dict[str, Any]) -> list[str]:
    return sorted((k for k in selection if k.isdigit()), key=int)


def _battery_conds(arm_results: dict[str, Any]) -> list[str]:
    return sorted(arm_results, key=lambda c: (c != "3a_generic", c))


def _cond_label(cond: str) -> str:
    if cond == "3a_generic":
        return "3a"
    return "3b f" + cond.rsplit("fold", 1)[1]


def _acc_at_1(knn_metric: dict[str, Any]) -> float | None:
    acc = knn_metric.get("acc_at_k", {})
    v = acc.get("1", acc.get(1))
    return float(v) if v is not None else None


def _arm_fig(n_rows: int = 1, height: float = 3.4) -> tuple[Any, np.ndarray]:
    fig, axes = plt.subplots(
        n_rows, len(ARMS), figsize=(11.0, height * n_rows), sharey="row", squeeze=False
    )
    return fig, axes


def _assert_nonempty(fig: Any, stem: str) -> None:
    """Empty-render guard (#1112): every figure must carry plotted artists."""
    for ax in fig.get_axes():
        if ax.lines or ax.patches or ax.collections or ax.containers:
            return
    raise RuntimeError(f"figure {stem!r} rendered with no plotted artists")


# ---------------------------------------------------------------------------
# Renderers. Each takes the loaded-inputs ctx and returns a list of
# (stem, figure, caption). Required inputs are declared in FIGURES; a renderer
# may assume its required ctx keys are present.
# ---------------------------------------------------------------------------


def fig_hero(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    fig, axes = _arm_fig(height=3.8)
    y_lo = 0.45
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        table = stats["arms"][arm]["predictors"]
        preds = [(p, lab) for p, lab in HERO_ORDER if p in table]
        for i, (p, lab) in enumerate(preds):
            e = table[p]
            ax.bar(i, e["auroc"], color=PRED_COLOR[p], label=lab if j == 0 else None)
            y_lo = min(y_lo, e["auroc"] - 0.05)
            ci = e.get("auroc_ci95")
            yerr = _ci_yerr(e["auroc"], ci)
            if yerr is not None:
                ax.errorbar(i, e["auroc"], yerr=yerr, fmt="none", ecolor="black", capsize=3)
                y_lo = min(y_lo, ci[0] - 0.05)
        ax.axhline(0.5, ls="--", lw=1, color="grey")
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels([lab for _, lab in preds], rotation=30, ha="right")
        ax.set_title(ARM_LABEL[arm])
    for j in range(len(ARMS)):
        axes[0][j].set_ylim(max(0.0, y_lo), 1.02)
    axes[0][0].set_ylabel("pooled OOF AUROC")
    cap = (
        "HERO: pooled out-of-fold AUROC per predictor (group 5-fold scheme), one panel per "
        "arm; error bars = paired group-bootstrap 95% CIs from stats.json; dashed line = "
        "chance (0.5). All scores oriented as P(REFUSE)."
    )
    return [("hero_auroc_by_predictor", fig, cap)]


def fig_layer_curves(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig()
    curve_preds = [PRED_CTX, PRED_DIM, PRED_ANS, PRED_ANS_RM]
    colors = dict(zip(curve_preds, paper_palette(len(curve_preds))))
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        sel = ctx[f"scores_{arm}"]["selection"]
        for p in curve_preds:
            curves = [
                sel[fk][p]["inner_auroc_by_layer"]
                for fk in _fold_keys(sel)
                if p in sel[fk] and "inner_auroc_by_layer" in sel[fk][p]
            ]
            if not curves:
                continue
            layers = sorted(curves[0], key=int)
            m = np.nanmean([[c[ell] for ell in layers] for c in curves], axis=0)
            ax.plot(
                [int(x) for x in layers], m, marker="o", ms=3, color=colors[p], label=PRED_NAME[p]
            )
        ax.axhline(0.5, ls="--", lw=1, color="grey")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("layer")
    axes[0][0].set_ylabel("inner-CV AUROC (mean over folds)")
    axes[0][0].legend(fontsize=8)
    cap = (
        "Per-layer AUROC curves for the layer-selected predictors: inner group-CV pooled "
        "AUROC per layer (selection metric), averaged over outer folds. These are the "
        "SELECTION curves persisted by the fits driver, not held-out eval AUROCs."
    )
    return [("auroc_by_layer", fig, cap)]


def fig_rank_selected(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(n_rows=2, height=2.6)
    preds = [PRED_3A, PRED_3B, PRED_PCA]
    colors = dict(zip(preds, paper_palette(len(preds))))
    rank_of: Callable[[Any], float] = lambda r: float(r) if r != "full" else float("nan")
    for j, arm in enumerate(ARMS):
        sel = ctx[f"scores_{arm}"]["selection"]
        fks = _fold_keys(sel)
        for p in preds:
            folds = [int(fk) for fk in fks if p in sel[fk]]
            ranks = [rank_of(sel[fk][p].get("rank")) for fk in fks if p in sel[fk]]
            layers = [sel[fk][p]["layer"] for fk in fks if p in sel[fk]]
            axes[0][j].plot(folds, ranks, "o", color=colors[p], label=PRED_NAME[p], alpha=0.8)
            axes[1][j].plot(folds, layers, "s", color=colors[p], alpha=0.8)
        axes[0][j].set_yscale("log", base=2)
        axes[0][j].set_title(ARM_LABEL[arm])
        axes[1][j].set_xlabel("outer fold")
    axes[0][0].set_ylabel("selected rank")
    axes[1][0].set_ylabel("selected layer")
    axes[0][0].legend(fontsize=8)
    cap = (
        "Rank-ladder outcome per fold: the inner-CV-selected (rank, layer) for the mapped "
        "probes (3a/3b) and the matched-rank PCA control. 'full'-rank selections are "
        "omitted from the rank panel (log2 axis). The full per-(layer,rank) AUROC surface "
        "is not persisted by the fits driver; this shows the realized selections."
    )
    return [("rank_ladder_selected", fig, cap)]


def fig_learning_curves(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    fig, axes = _arm_fig()
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        ladder = stats["arms"][arm].get("ladder", {})
        for p, by_n in sorted(ladder.items()):
            keys = sorted(by_n, key=lambda k: (k == "all", int(k) if k != "all" else 0))
            xs = np.arange(len(keys))
            means = [by_n[k]["mean"] for k in keys]
            sds = [by_n[k]["sd"] for k in keys]
            color = PRED_COLOR.get(p)
            ax.errorbar(
                xs,
                means,
                yerr=sds,
                marker="o",
                ms=3,
                capsize=2,
                color=color,
                label=PRED_NAME.get(p, p),
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(keys)
            rendered = True
        ax.axhline(0.5, ls="--", lw=1, color="grey")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("labeled groups per class")
    axes[0][0].set_ylabel("AUROC (mean +- sd over subsample seeds)")
    axes[0][0].legend(fontsize=8)
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Limited-label learning curves: AUROC at the fold-selected configuration when the "
        "probe trains on n labeled groups/class (10 subsample seeds; 'all' = full train "
        "fold, 1 seed). Mean +- sd over seeds of the per-seed fold means."
    )
    return [("limited_label_learning_curves", fig, cap)]


def fig_lodo(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    scores = ctx["scores_armB"]["scores"]
    src_of = {r["prompt_sha"]: r.get("source", "?") for r in ctx["manifest_armB"]["rows"]}
    per_src: dict[str, tuple[list[float], list[int]]] = {}
    for sha, rec in scores.items():
        v = rec.get(PRED_LODO)
        if v is None:
            continue
        src = src_of.get(sha, "?")
        per_src.setdefault(src, ([], []))
        per_src[src][0].append(float(v))
        per_src[src][1].append(int(rec["y"]))
    if not per_src:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    names, vals = [], []
    for src in sorted(per_src):
        sc, yy = per_src[src]
        names.append(f"held-out {src}\n(n={len(sc)})")
        vals.append(_auroc(np.array(sc), np.array(yy)))
    lodo = stats["arms"]["armB"].get("lodo", {})
    if lodo.get("pooled_auroc") is not None:
        names.append(f"pooled\n(n={lodo.get('n', '?')})")
        vals.append(lodo["pooled_auroc"])
    ax.bar(range(len(vals)), vals, color=paper_palette_role("primary"))
    infold = stats["arms"]["armB"]["predictors"].get(PRED_CTX, {}).get("auroc")
    if infold is not None:
        ax.axhline(
            infold, ls=":", lw=1.2, color=paper_palette_role("baseline"), label="in-fold ctx probe"
        )
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AUROC")
    ax.set_title("Arm B - leave-one-dataset-out (ctx probe)")
    ax.legend(fontsize=8)
    cap = (
        "LODO cross-benchmark transfer (Arm B): ctx probe trained on the other source, "
        "evaluated on the held-out source (per-direction AUROC from per-row LODO scores "
        "joined to the corpus manifest's source field), plus the pooled LODO AUROC. "
        "Dotted line = in-fold ctx probe AUROC; dashed = chance."
    )
    return [("lodo_armB", fig, cap)]


def fig_transfer(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    dirs = ctx["transfer"].get("directions", {})
    if not dirs:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    keys = sorted(dirs)
    vals = [dirs[k]["auroc"] for k in keys]
    labels = [k.replace("armA", "A").replace("armB", "B").replace("|", "\n") for k in keys]
    colors = [
        paper_palette_role("primary") if k.endswith("ridge") else paper_palette_role("control")
        for k in keys
    ]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC on the other arm")
    ax.set_title("Cross-regime transfer (report-only)")
    cap = (
        "Cross-regime transfer 2x2: ctx probe (ridge) and diff-in-means trained on one "
        "arm's full balanced set, evaluated on the other arm's balanced set; both "
        "directions. Report-only (plan Step G / H4); dashed = chance."
    )
    return [("transfer_2x2", fig, cap)]


def fig_roc(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(height=4.2)
    roc_preds = [PRED_JUDGE, PRED_CTX, PRED_3A, PRED_ANS, PRED_TEXT]
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        rows = ctx[f"scores_{arm}"]["scores"]
        oof = [r for r in rows.values() if r.get("fold", -1) >= 0]
        for p in roc_preds:
            pts = [(r[p], r["y"]) for r in oof if r.get(p) is not None]
            if len(pts) < 4:
                continue
            sc = np.array([x for x, _ in pts])
            yy = np.array([y for _, y in pts])
            fpr, tpr = _roc_curve(sc, yy)
            ax.plot(fpr, tpr, color=PRED_COLOR[p], label=PRED_NAME[p] if j == 0 else None)
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("false positive rate")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("true positive rate")
    axes[0][0].legend(fontsize=8, loc="lower right")
    cap = (
        "ROC curves per predictor over pooled OOF rows (positive class = refuse). "
        "Per-predictor valid rows; see stats.json for the common-mask AUROCs."
    )
    return [("roc_curves", fig, cap)]


def fig_judge_calibration(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(height=3.8)
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        rows = ctx[f"scores_{arm}"]["scores"]
        pts = [
            (r[PRED_JUDGE], r["y"])
            for r in rows.values()
            if r.get(PRED_JUDGE) is not None and r.get("fold", -1) >= 0
        ]
        if len(pts) >= 4:
            sc = np.array([x for x, _ in pts], dtype=np.float64)
            yy = np.array([y for _, y in pts], dtype=np.float64)
            bins = np.linspace(0, 1, 11)
            ix = np.clip(np.digitize(sc, bins) - 1, 0, 9)
            xs, ys, ns = [], [], []
            for b in range(10):
                m = ix == b
                if m.any():
                    xs.append(float(sc[m].mean()))
                    ys.append(float(yy[m].mean()))
                    ns.append(int(m.sum()))
            ax.scatter(xs, ys, s=[10 + 3 * n for n in ns], color=paper_palette_role("primary"))
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
            rendered = True
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("judge P(refuse)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("empirical refuse fraction")
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Judge calibration: few-shot judge P(refuse) binned into deciles vs the empirical "
        "refuse fraction among OOF rows in each bin (marker size ~ bin count); dashed = "
        "perfect calibration."
    )
    return [("judge_calibration", fig, cap)]


def fig_map_quality(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    conds = ctx["diag"]["conditions"]
    if "3a_generic" not in conds:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    g = conds["3a_generic"]["per_layer"]
    layers = [r["layer"] for r in g]
    axes[0].plot(
        layers,
        [r["r2_map"] for r in g],
        marker="o",
        ms=3,
        color=paper_palette_role("primary"),
        label="map R2 (3a)",
    )
    axes[0].plot(
        layers,
        [r["r2_identity_bias"] for r in g],
        marker="s",
        ms=3,
        ls="--",
        color=paper_palette_role("baseline"),
        label="identity+bias",
    )
    for cond, cd in sorted(conds.items()):
        if cond == "3a_generic":
            continue
        axes[0].plot(
            [r["layer"] for r in cd["per_layer"]],
            [r["r2_map"] for r in cd["per_layer"]],
            lw=0.8,
            alpha=0.35,
            color=paper_palette_role("neutral"),
        )
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("held-out R2")
    axes[0].set_title("map quality by layer")
    axes[0].legend(fontsize=8)
    for metric, ls in (("euclidean", "-"), ("cosine", ":")):
        acc = [_acc_at_1(r["knn"][metric]) for r in g]
        axes[1].plot(
            layers,
            acc,
            marker="o",
            ms=3,
            ls=ls,
            color=paper_palette_role("accent"),
            label=f"kNN acc@1 ({metric})",
        )
    chance = g[0]["knn"]["euclidean"].get("chance_at_k", {})
    ch1 = chance.get("1", chance.get(1))
    if ch1 is not None:
        axes[1].axhline(float(ch1), ls="--", lw=1, color="grey", label="chance @1")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("retrieval acc@1 (held-out)")
    axes[1].set_title("kNN retrieval (3a map)")
    axes[1].legend(fontsize=8)
    cap = (
        "Map diagnostics by layer (generic held-out rows): left - held-out R2 of the 3a "
        "map (solid) vs the identity+bias baseline (dashed); thin grey lines = the 3b "
        "in-domain-adapted conditions. Right - kNN retrieval acc@1 of the 3a map "
        "predictions among the held-out pool (euclidean + cosine) vs chance."
    )
    return [("map_quality_by_layer", fig, cap)]


def fig_w_spectrum(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    conds = ctx["diag"]["conditions"]
    if "3a_generic" not in conds:
        return []
    g = conds["3a_generic"]["per_layer"]
    layers = [r["layer"] for r in g]
    best = conds["3a_generic"].get("best_layer_by_generic_r2", layers[len(layers) // 2])
    picks = sorted({layers[0], best, layers[-1]})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    colors = paper_palette(len(picks))
    for c, ell in zip(colors, picks):
        row = next(r for r in g if r["layer"] == ell)
        s = np.asarray(row["spectrum_top16"], dtype=np.float64)
        axes[0].plot(np.arange(1, len(s) + 1), s, marker="o", ms=3, color=c, label=f"layer {ell}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("singular index")
    axes[0].set_ylabel("singular value of W")
    axes[0].set_title("W spectra (top 16)")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        layers,
        [r["participation_ratio"] for r in g],
        marker="o",
        ms=3,
        color=paper_palette_role("primary"),
        label="3a",
    )
    for cond, cd in sorted(conds.items()):
        if cond == "3a_generic":
            continue
        axes[1].plot(
            [r["layer"] for r in cd["per_layer"]],
            [r["participation_ratio"] for r in cd["per_layer"]],
            lw=0.8,
            alpha=0.35,
            color=paper_palette_role("neutral"),
        )
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("effective rank (participation ratio)")
    axes[1].set_title("effective rank by layer")
    axes[1].legend(fontsize=8)
    cap = (
        "Left: top-16 singular values of the fitted context-to-answer map W at the first / "
        "best-R2 / last layers (3a). Right: effective rank (participation ratio of the "
        "singular spectrum) by layer; thin grey = 3b conditions."
    )
    return [("w_spectra_effective_rank", fig, cap)]


def fig_dim_vs_ridge(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    width = 0.35
    for k, (p, role) in enumerate([(PRED_CTX, "primary"), (PRED_DIM, "control")]):
        vals, errs = [], []
        for arm in ARMS:
            e = stats["arms"][arm]["predictors"].get(p)
            vals.append(e["auroc"] if e else np.nan)
            errs.append(_ci_yerr(e["auroc"], e.get("auroc_ci95")) if e else None)
        xs = np.arange(len(ARMS)) + (k - 0.5) * width
        ax.bar(xs, vals, width, color=paper_palette_role(role), label=PRED_NAME[p])
        for x, v, err in zip(xs, vals, errs):
            if err is not None:
                ax.errorbar(x, v, yerr=err, fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_ylabel("pooled OOF AUROC")
    ax.set_title("ridge vs diff-in-means (ctx probe)")
    ax.legend(fontsize=8)
    cap = (
        "Estimator sensitivity: the ctx ridge probe vs the diff-in-means companion "
        "(train-class-mean difference at the last prompt token), pooled OOF AUROC with "
        "group-bootstrap CIs where available; dashed = chance."
    )
    return [("dim_vs_ridge", fig, cap)]


def fig_contrasts(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    fig, axes = _arm_fig(height=3.6)
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        table = stats["arms"][arm].get("contrasts", {})
        pts, errs, labels = [], [], []
        for key, lab in CONTRASTS:
            if key in table:
                pts.append(table[key]["point"])
                errs.append(_ci_yerr(table[key]["point"], table[key].get("ci95")))
                labels.append(lab)
        for i, (v, err) in enumerate(zip(pts, errs)):
            ax.plot(i, v, "o", color=paper_palette_role("primary"))
            if err is not None:
                ax.errorbar(i, v, yerr=err, fmt="none", ecolor="black", capsize=3)
            rendered = True
        ax.axhline(0.0, ls="--", lw=1, color="grey")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(ARM_LABEL[arm])
    axes[0][0].set_ylabel("paired AUROC difference")
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Registered paired contrasts per arm: AUROC differences with paired group-bootstrap "
        "95% CIs over the common per-arm row mask (identical draw indices per contrast). "
        "Includes the F2 estimator-class control (ctx - text-surface) and delta_int "
        "(ctx - judge); dashed = zero."
    )
    return [("contrast_paired_ci", fig, cap)]


def fig_armA_indicators(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    table = ctx["stats"]["arms"].get("armA", {}).get("predictors", {})
    preds = [p for p in (PRED_TEXT, PRED_TEXT_NOIND, PRED_ISREW) if p in table]
    if not preds:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, p in enumerate(preds):
        e = table[p]
        ax.bar(i, e["auroc"], color=PRED_COLOR.get(p, paper_palette_role("control")))
        yerr = _ci_yerr(e["auroc"], e.get("auroc_ci95"))
        if yerr is not None:
            ax.errorbar(i, e["auroc"], yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(preds)))
    ax.set_xticklabels([PRED_NAME[p] for p in preds], rotation=15, ha="right")
    ax.set_ylabel("pooled OOF AUROC")
    ax.set_title("Arm A - text-surface / is-rewrite indicator reads (F2)")
    cap = (
        "Arm A within-group construction control (F2): AUROC of the bare is-rewrite "
        "indicator, and the fitted text-surface baseline WITH vs WITHOUT the rewrite-axis "
        "indicator features. A text-surface edge driven by the base-first REFUSE sampling "
        "channel would show up here; dashed = chance."
    )
    return [("armA_text_surface_indicator", fig, cap)]


def fig_perm_band(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    stats = ctx["stats"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    rendered = False
    for i, arm in enumerate(ARMS):
        perm = stats["arms"][arm].get("permutation_advisory", {})
        if perm.get("skipped") or "null_q95" not in perm:
            continue
        ax.plot(
            [i, i],
            [perm["null_q50"], perm["null_q95"]],
            lw=6,
            alpha=0.4,
            color=paper_palette_role("neutral"),
            label="null q50-q95" if not rendered else None,
        )
        ax.plot(i, perm["null_mean"], "_", ms=16, color=paper_palette_role("neutral"))
        if perm.get("observed") is not None:
            ax.plot(
                i,
                perm["observed"],
                "o",
                ms=7,
                color=paper_palette_role("primary"),
                label="observed ctx AUROC" if not rendered else None,
            )
        rendered = True
    if not rendered:
        plt.close(fig)
        return []
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_ylabel("pooled OOF AUROC")
    ax.set_title("advisory group-label permutation band (ctx probe)")
    ax.legend(fontsize=8)
    cap = (
        "Advisory permutation calibration: observed ctx-probe AUROC vs the group-label "
        "permutation null (band = null q50-q95, tick = null mean; per-draw inner layer "
        "re-selection at frozen per-(fold,layer) GCV lambda). Advisory only - never a "
        "gate; p-values in stats.json."
    )
    return [("permutation_band", fig, cap)]


def _battery_variant_fig(ctx: dict[str, Any], variant: str) -> tuple[Any, bool] | None:
    battery = ctx["battery"]["results"]
    fig, axes = _arm_fig(height=3.8)
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        conds = _battery_conds(battery.get(arm, {}))
        width = 0.16
        chance = None
        for ci, cond in enumerate(conds):
            res = battery[arm][cond].get(variant)
            if res is None:
                continue
            chance = res["chance"]
            for si, space in enumerate(BATTERY_SPACES):
                if space not in res:
                    continue
                x = ci + (si - 1.5) * width
                ax.bar(
                    x,
                    res[space]["acc_at_1"],
                    width,
                    color=SPACE_COLOR[space],
                    label=SPACE_NAME[space] if (j == 0 and ci == 0) else None,
                )
                rendered = True
            ib = res.get("identity_bias_baseline", {}).get("whitened_cosine")
            if ib is not None:
                ax.bar(
                    ci + 2.5 * width,
                    ib["acc_at_1"],
                    width,
                    color="lightgrey",
                    hatch="//",
                    edgecolor="grey",
                    label="identity+bias (whitened)" if (j == 0 and ci == 0) else None,
                )
            gate = res.get("s2_gate", {})
            if gate.get("ci_lower_5pct") is not None and np.isfinite(gate["ci_lower_5pct"]):
                ax.plot(
                    ci - 1.5 * width,
                    gate["ci_lower_5pct"],
                    "_",
                    ms=10,
                    color="black",
                    label="whitened CI lower (5%)" if (j == 0 and ci == 0) else None,
                )
        if chance is not None:
            ax.axhline(chance, ls="--", lw=1, color="grey")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([_cond_label(c) for c in conds], rotation=30, ha="right")
        ax.set_title(ARM_LABEL[arm])
        ax.set_yscale("log")
    axes[0][0].set_ylabel("retrieval acc@1 (log)")
    axes[0][0].legend(fontsize=7)
    if not rendered:
        plt.close(fig)
        return None
    return fig, True


def fig_battery(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    out: list[tuple[str, Any, str]] = []
    for variant in ("greedy", "draw_avg"):
        got = _battery_variant_fig(ctx, variant)
        if got is None:
            continue
        fig, _ = got
        cap = (
            f"Map-discrimination battery ({variant} targets): retrieval acc@1 per (arm x "
            "map condition) across the four metric spaces, with the identity+bias baseline "
            "pushed through the same whitened battery (hatched) and the S2 gate marker "
            "(black tick = group-bootstrap 5% lower bound of whitened-cosine acc@1; a cell "
            "passes the discrimination predicate iff the tick sits above the dashed chance "
            "line = 1/n_pool)."
        )
        out.append((f"battery_acc1_{variant}", fig, cap))
    return out


def fig_battery_behavior(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    battery = ctx["battery"]["results"]
    fig, axes = _arm_fig(height=3.6)
    class_colors = {"refuse": paper_palette_role("primary"), "engage": paper_palette_role("accent")}
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        conds = _battery_conds(battery.get(arm, {}))
        chance = None
        for ci, cond in enumerate(conds):
            res = battery[arm][cond].get("greedy")
            if res is None:
                continue
            chance = res["chance"]
            split = res.get("behavior_split_acc1_whitened", {})
            for si, lab in enumerate(("refuse", "engage")):
                if lab in split:
                    ax.bar(
                        ci + (si - 0.5) * 0.3,
                        split[lab],
                        0.3,
                        color=class_colors[lab],
                        label=lab if (j == 0 and ci == 0) else None,
                    )
                    rendered = True
        if chance is not None:
            ax.axhline(chance, ls="--", lw=1, color="grey")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([_cond_label(c) for c in conds], rotation=30, ha="right")
        ax.set_title(ARM_LABEL[arm])
        ax.set_yscale("log")
    axes[0][0].set_ylabel("whitened-cosine acc@1 (log)")
    axes[0][0].legend(fontsize=8)
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Behavior-conditioned retrieval: whitened-cosine acc@1 (greedy targets) split by "
        "the target's behavior class (refuse vs engage answers) per arm x map condition; "
        "dashed = chance (1/n_pool)."
    )
    return [("battery_behavior_split", fig, cap)]


def fig_nn_match(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    battery = ctx["battery"]["results"]
    fig, axes = _arm_fig(height=3.6)
    var_colors = {"greedy": paper_palette_role("primary"), "draw_avg": paper_palette_role("accent")}
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        conds = _battery_conds(battery.get(arm, {}))
        for ci, cond in enumerate(conds):
            for vi, variant in enumerate(("greedy", "draw_avg")):
                res = battery[arm][cond].get(variant)
                if res is None or res.get("nn_behavior_match_rate") is None:
                    continue
                ax.bar(
                    ci + (vi - 0.5) * 0.3,
                    res["nn_behavior_match_rate"],
                    0.3,
                    color=var_colors[variant],
                    label=variant if (j == 0 and ci == 0) else None,
                )
                rendered = True
        labels = ctx.get(f"labels_{arm}")
        if labels is not None:
            gl = [r.get("greedy_label") for r in labels["rows"].values()]
            gl = [x for x in gl if x is not None]
            if gl:
                shares = np.array([gl.count(lab) / len(gl) for lab in ("refuse", "engage")])
                ax.axhline(float((shares**2).sum()), ls="--", lw=1, color="grey")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([_cond_label(c) for c in conds], rotation=30, ha="right")
        ax.set_title(ARM_LABEL[arm])
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("NN behavior-match rate")
    axes[0][0].legend(fontsize=8)
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Fit-free discriminator: does the whitened nearest pool neighbor of the predicted "
        "answer carry the same greedy behavior label as the target's prompt? Dashed "
        "reference = matched-random-pair rate from the pool's greedy-label shares "
        "(sum of squared class shares)."
    )
    return [("nn_behavior_match", fig, cap)]


def fig_rank_summary(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    battery = ctx["battery"]["results"]
    fig, axes = _arm_fig(n_rows=2, height=2.8)
    rendered = False
    for j, arm in enumerate(ARMS):
        conds = _battery_conds(battery.get(arm, {}))
        width = 0.2
        for ci, cond in enumerate(conds):
            res = battery[arm][cond].get("greedy")
            if res is None:
                continue
            for si, space in enumerate(BATTERY_SPACES):
                if space not in res:
                    continue
                x = ci + (si - 1.5) * width
                axes[0][j].bar(
                    x,
                    res[space]["median_rank"],
                    width,
                    color=SPACE_COLOR[space],
                    label=SPACE_NAME[space] if (j == 0 and ci == 0) else None,
                )
                axes[1][j].bar(x, res[space]["mrr"], width, color=SPACE_COLOR[space])
                rendered = True
        for row in (0, 1):
            axes[row][j].set_xticks(range(len(conds)))
            axes[row][j].set_xticklabels([_cond_label(c) for c in conds], rotation=30, ha="right")
            axes[row][j].set_yscale("log")
        axes[0][j].set_title(ARM_LABEL[arm])
    axes[0][0].set_ylabel("median rank (log)")
    axes[1][0].set_ylabel("MRR (log)")
    axes[0][0].legend(fontsize=7)
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Battery rank summary (greedy targets): median rank of the true answer and MRR per "
        "(arm x condition x metric space). Per-target rank histograms are not persisted by "
        "the fits driver; median rank + MRR are the persisted rank reads."
    )
    return [("battery_rank_summary", fig, cap)]


def fig_score_scatter(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(height=4.2)
    y_colors = {1: paper_palette_role("primary"), 0: paper_palette_role("accent")}
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        rows = ctx[f"scores_{arm}"]["scores"]
        pts = [
            (r[PRED_CTX], r[PRED_3A], r["y"])
            for r in rows.values()
            if r.get(PRED_CTX) is not None and r.get(PRED_3A) is not None and r.get("fold", -1) >= 0
        ]
        for yv, lab in ((1, "refuse"), (0, "engage")):
            xs = [a for a, _, y in pts if y == yv]
            ys = [b for _, b, y in pts if y == yv]
            if xs:
                ax.scatter(
                    xs, ys, s=9, alpha=0.6, color=y_colors[yv], label=lab if j == 0 else None
                )
                rendered = True
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("ctx probe score")
    axes[0][0].set_ylabel("mapped (3a) probe score")
    axes[0][0].legend(fontsize=8)
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Per-row OOF score scatter: mapped (3a) probe score vs ctx probe score, colored by "
        "the row's behavior label. Tight coupling means the rank-restricted map carries "
        "the same decision information as the raw context read."
    )
    return [("score_scatter_map_vs_ctx", fig, cap)]


def fig_rate_hist(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(height=3.6)
    rendered = False
    for j, arm in enumerate(ARMS):
        ax = axes[0][j]
        labels = ctx[f"labels_{arm}"]
        rates = [r["rate"] for r in labels["rows"].values() if r.get("rate") is not None]
        thr = labels.get("thresholds", {})
        if rates:
            ax.hist(
                rates,
                bins=np.linspace(0, 1, 12),
                color=paper_palette_role("primary"),
                edgecolor="white",
            )
            rendered = True
        lo, hi = thr.get("lo"), thr.get("hi")
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, color="grey", alpha=0.25)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("engage rate over valid draws")
    axes[0][0].set_ylabel("n prompts")
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Continuous-rate histograms: per-prompt engage rate over valid judged draws (the "
        "graded secondary DV). Shaded band = the dropped middle band between the label "
        "thresholds (label engage iff rate >= hi, refuse iff rate <= lo)."
    )
    return [("rate_histograms", fig, cap)]


def fig_group_sizes(ctx: dict[str, Any]) -> list[tuple[str, Any, str]]:
    fig, axes = _arm_fig(height=3.6)
    rendered = False
    # Arm A: flip-group size histogram (groups with >=1 comply-labeled row).
    ga = ctx["groups_armA"]
    flip = set(ga.get("flip_groups", []))
    if flip:
        counts: dict[str, int] = {}
        for gid in ga["group_of"].values():
            counts[gid] = counts.get(gid, 0) + 1
        sizes = [counts[g] for g in flip if g in counts]
        if sizes:
            axes[0][0].hist(
                sizes,
                bins=np.arange(0.5, max(sizes) + 1.5, 1),
                color=paper_palette_role("primary"),
                edgecolor="white",
            )
            rendered = True
    axes[0][0].set_title(f"{ARM_LABEL['armA']} - flip-group sizes")
    axes[0][0].set_xlabel("group size (rows)")
    axes[0][0].set_ylabel("n groups")
    # Arm B: derived-group size histogram.
    hist = ctx["groups_armB"].get("size_histogram", {})
    if hist:
        ks = sorted(hist, key=int)
        axes[0][1].bar(
            [int(k) for k in ks], [hist[k] for k in ks], color=paper_palette_role("accent")
        )
        rendered = True
    axes[0][1].set_title(f"{ARM_LABEL['armB']} - derived-group sizes")
    axes[0][1].set_xlabel("group size (rows)")
    if not rendered:
        plt.close(fig)
        return []
    cap = (
        "Group structure: Arm A flip-group sizes (base_id families with >=1 comply-labeled "
        "row - the matched-pair units) and Arm B derived-paraphrase-component sizes "
        "(TF-IDF union v_C-cosine connected components). Degeneracy guards + realized "
        "thresholds live in groups.json."
    )
    return [("group_size_histograms", fig, cap)]


# name -> (required ctx keys, renderer)
FIGURES: dict[str, tuple[tuple[str, ...], Callable[[dict[str, Any]], list]]] = {
    "hero_auroc_by_predictor": (("stats",), fig_hero),
    "auroc_by_layer": (("scores_armA", "scores_armB"), fig_layer_curves),
    "rank_ladder_selected": (("scores_armA", "scores_armB"), fig_rank_selected),
    "limited_label_learning_curves": (("stats",), fig_learning_curves),
    "lodo_armB": (("stats", "scores_armB", "manifest_armB"), fig_lodo),
    "transfer_2x2": (("transfer",), fig_transfer),
    "roc_curves": (("scores_armA", "scores_armB"), fig_roc),
    "judge_calibration": (("scores_armA", "scores_armB"), fig_judge_calibration),
    "map_quality_by_layer": (("diag",), fig_map_quality),
    "w_spectra_effective_rank": (("diag",), fig_w_spectrum),
    "dim_vs_ridge": (("stats",), fig_dim_vs_ridge),
    "contrast_paired_ci": (("stats",), fig_contrasts),
    "armA_text_surface_indicator": (("stats",), fig_armA_indicators),
    "permutation_band": (("stats",), fig_perm_band),
    "battery_acc1": (("battery",), fig_battery),
    "battery_behavior_split": (("battery",), fig_battery_behavior),
    "nn_behavior_match": (("battery", "labels_armA", "labels_armB"), fig_nn_match),
    "battery_rank_summary": (("battery",), fig_rank_summary),
    "score_scatter_map_vs_ctx": (("scores_armA", "scores_armB"), fig_score_scatter),
    "rate_histograms": (("labels_armA", "labels_armB"), fig_rate_hist),
    "group_size_histograms": (("groups_armA", "groups_armB"), fig_group_sizes),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_inputs(eval_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load every present input JSON; return (ctx, sha-by-key for fingerprints)."""
    ctx: dict[str, Any] = {}
    shas: dict[str, str] = {}
    for key, rel in INPUTS.items():
        p = eval_root / rel
        if not p.exists():
            ctx[key] = None
            continue
        ctx[key] = json.loads(p.read_text(encoding="utf-8"))
        shas[key] = _sha256_file(p)
    return ctx, shas


def _entry_fingerprint(requires: Iterable[str], shas: dict[str, str], extra: str) -> str:
    payload = {k: shas.get(k, "") for k in requires}
    payload["_extra"] = extra
    payload["_code"] = _sha256_file(Path(__file__).resolve())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def render_all(
    eval_root: Path,
    fig_dir: Path,
    *,
    only: set[str] | None = None,
    resume: bool = True,
    style: str = "blog",
    formats: tuple[str, ...] = ("png",),
) -> dict[str, Any]:
    set_paper_style(style)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ctx, shas = _load_inputs(eval_root)

    manifest_path = fig_dir / "render_manifest.json"
    manifest: dict[str, Any] = {}
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    captions: dict[str, Any] = {}
    n_rendered = n_skipped = n_cached = 0
    for name, (requires, fn) in FIGURES.items():
        if only and name not in only:
            continue
        missing = [k for k in requires if ctx.get(k) is None]
        if missing:
            captions[name] = {
                "status": "skipped",
                "reason": f"input absent: {', '.join(INPUTS[k] for k in missing)}",
            }
            logger.info("[skip] %s (missing: %s)", name, ", ".join(missing))
            n_skipped += 1
            continue
        fp = _entry_fingerprint(requires, shas, extra=f"{style}|{','.join(formats)}")
        prior = manifest.get(name, {})
        prior_outs = [fig_dir / f"{s}.png" for s in prior.get("stems", [])]
        if (
            resume
            and prior.get("fingerprint") == fp
            and prior_outs
            and all(p.exists() for p in prior_outs)
        ):
            for stem in prior["stems"]:
                captions[stem] = {
                    "status": "rendered (cached)",
                    "caption": prior.get("captions", {}).get(stem, ""),
                    "inputs": [INPUTS[k] for k in requires],
                }
            logger.info("[cache] %s (fingerprint match)", name)
            n_cached += 1
            continue
        outs = fn(ctx)
        if not outs:
            captions[name] = {
                "status": "skipped",
                "reason": "required fields empty in present inputs",
            }
            logger.info("[skip] %s (fields empty)", name)
            n_skipped += 1
            continue
        stems: list[str] = []
        caps: dict[str, str] = {}
        for stem, fig, caption in outs:
            _assert_nonempty(fig, stem)
            savefig_paper(fig, stem, dir=fig_dir, formats=formats)
            plt.close(fig)
            stems.append(stem)
            caps[stem] = caption
            captions[stem] = {
                "status": "rendered",
                "caption": caption,
                "inputs": [INPUTS[k] for k in requires],
            }
            logger.info("[render] %s", stem)
            n_rendered += 1
        manifest[name] = {"fingerprint": fp, "stems": stems, "captions": caps}

    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    payload = {
        "figures": captions,
        "eval_root": str(eval_root),
        "meta": _provenance(),
    }
    (fig_dir / "captions.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "[done] rendered=%d cached=%d skipped=%d -> %s", n_rendered, n_cached, n_skipped, fig_dir
    )
    return {"rendered": n_rendered, "cached": n_cached, "skipped": n_skipped}


# ---------------------------------------------------------------------------
# Import-check + selftest
# ---------------------------------------------------------------------------


def _import_check() -> int:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    from scipy.stats import rankdata  # noqa: F401  (deferred import in _auroc)

    import issue2356_fits as fits

    for const in ("ARMS", "HEADLINE_PREDS", "PHASES"):
        assert hasattr(fits, const), const
    logger.info("[import-check] imports + args attributes OK (%d figures)", len(FIGURES))
    return 0


def _run_fits_selftest_keep_scratch() -> Path:
    """Run the fits driver's synthetic 6-phase selftest, keeping its scratch.

    The fits selftest removes its scratch on success; we intercept
    ``tempfile.mkdtemp`` (to record the path) and no-op ``shutil.rmtree`` for
    the duration so the REAL producer's outputs survive as this renderer's
    smoke inputs (cross-phase data-contract smoke: consumer against the
    producer's real output shape). Both are restored in ``finally``.
    """
    import shutil
    import tempfile

    import issue2356_fits as fits

    made: list[str] = []
    orig_mkdtemp = tempfile.mkdtemp
    orig_rmtree = shutil.rmtree

    def _recording_mkdtemp(*a: Any, **k: Any) -> str:
        p = orig_mkdtemp(*a, **k)
        made.append(p)
        return p

    tempfile.mkdtemp = _recording_mkdtemp  # type: ignore[assignment]
    shutil.rmtree = lambda *a, **k: None  # type: ignore[assignment]
    try:
        rc = fits._selftest(None)
        if rc != 0:
            raise RuntimeError(f"issue2356_fits._selftest returned rc={rc}")
    finally:
        tempfile.mkdtemp = orig_mkdtemp  # type: ignore[assignment]
        shutil.rmtree = orig_rmtree  # type: ignore[assignment]
    scratches = [p for p in made if "i2356-selftest-" in p]
    if not scratches:
        raise RuntimeError("fits selftest scratch dir not captured")
    return Path(scratches[-1])


def _selftest() -> int:
    """Producer-real smoke: fits selftest outputs -> full render + probes."""
    import tempfile

    scratch = _run_fits_selftest_keep_scratch()
    eval_root = scratch / "eval_results"
    fig_dir = Path(tempfile.mkdtemp(prefix="i2356-figs-selftest-"))
    counts = render_all(eval_root, fig_dir, resume=False)
    hero = fig_dir / "hero_auroc_by_predictor.png"
    assert hero.exists() and hero.stat().st_size > 5_000, "hero PNG missing/empty"
    pngs = sorted(fig_dir.glob("*.png"))
    tiny = [p.name for p in pngs if p.stat().st_size < 2_000]
    assert not tiny, f"suspiciously small renders: {tiny}"
    assert counts["rendered"] >= 15, f"too few figures rendered: {counts}"
    assert (fig_dir / "captions.json").exists()

    # resume probe: a second pass over unchanged inputs must be all-cached.
    counts2 = render_all(eval_root, fig_dir, resume=True)
    assert counts2["rendered"] == 0 and counts2["cached"] >= 15, counts2

    # degenerate-input probe: EMPTY eval root -> everything skipped, rc 0.
    empty_root = Path(tempfile.mkdtemp(prefix="i2356-figs-empty-"))
    fig_dir2 = Path(tempfile.mkdtemp(prefix="i2356-figs-skip-"))
    counts3 = render_all(empty_root, fig_dir2, resume=False)
    assert counts3["rendered"] == 0 and counts3["skipped"] == len(FIGURES), counts3
    assert (fig_dir2 / "captions.json").exists()

    print(
        f"[selftest] PASS ({counts['rendered']} figures rendered, "
        f"{counts['skipped']} skipped; all-cached resume + empty-root skip probes OK; "
        f"figs={fig_dir} scratch={scratch})",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2356 P8 figure renderer.")
    ap.add_argument("--eval-root", default=f"eval_results/issue_{ISSUE}")
    ap.add_argument("--fig-dir", default=f"figures/issue_{ISSUE}")
    ap.add_argument("--only", default="", help="comma-separated figure names (default: all)")
    ap.add_argument("--style", default="blog", choices=["blog", "neurips", "generic"])
    ap.add_argument("--formats", default="png", help="comma-separated savefig formats")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--list-figures", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.list_figures:
        print("\n".join(FIGURES))
        return 0
    if args.import_check:
        return _import_check()
    if args.selftest:
        return _selftest()
    only = {t.strip() for t in args.only.split(",") if t.strip()} or None
    if only:
        unknown = only - set(FIGURES)
        if unknown:
            raise SystemExit(f"unknown figure names: {sorted(unknown)}")
    render_all(
        Path(args.eval_root).resolve(),
        Path(args.fig_dir).resolve(),
        only=only,
        resume=not args.no_resume,
        style=args.style,
        formats=tuple(t.strip() for t in args.formats.split(",") if t.strip()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
