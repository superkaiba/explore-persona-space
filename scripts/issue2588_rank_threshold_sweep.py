#!/usr/bin/env python3
"""Threshold sensitivity for the issue-2588 performance-preserving rank.

Reads per-map rank curves (validation R^2 at every nested rank) and produces,
for each available METHOD of building the rank-k map:

* rank_curves_same_width[_rrr].png: the full rank-vs-R^2 curve of every
  hidden-size-5120 map (retained fraction of full R^2, and error ratio), so
  no single cutoff has to be chosen;
* rank_threshold_sweep[_rrr].png: rank fraction versus AA for the same-width
  column under several tolerances (absolute R^2 gap, retained R^2 fraction,
  relative error increase), one panel per rule with the column Spearman;
* rank_threshold_sweep[_rrr].json: every rank under every rule, a
  threshold-free compressibility index (area under the retained-R^2 curve up
  to 25% of width), and Spearman correlations with AA.

Methods: "truncated" = the fitted ridge map cut along its own top singular
directions (mapping_rank_vs_capability.json); "rrr" = reduced-rank ridge, the
fitted map projected onto the top principal directions of its fitted training
outputs (rrr_rank_curves.json, from issue2588_rrr_rank.py).  When both exist a
comparison figure rank_method_comparison.png is written too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRUNC = REPO / "eval_results" / "issue_2588" / "mapping_rank_vs_capability.json"
DEFAULT_RRR = REPO / "eval_results" / "issue_2588" / "rrr_rank_curves.json"
OUT_DIR = REPO / "eval_results" / "issue_2588"
FIG_DIR = REPO / "figures" / "issue_2588"
SAME_WIDTH_DIM = 5120
ARMS = ("no-thinking", "end-of-thought")
ARM_TITLE = {"no-thinking": "Prompt read", "end-of-thought": "End-of-thought read"}
ARM_COLOR = {"no-thinking": "#0072B2", "end-of-thought": "#D55E00"}
AUC_MAX_FRACTION = 0.25
METHOD_TITLE = {
    "truncated": "truncated ridge map",
    "rrr": "reduced-rank regression",
}

# (label, kind, parameter).  kind: "abs" = full R2 - k-rank R2 <= a;
# "retain" = k-rank R2 >= q * full R2; "error" = (1 - R2_k) <= (1 + e) * (1 - full R2).
RULES: tuple[tuple[str, str, float], ...] = (
    ("R² gap ≤ 0.01", "abs", 0.01),
    ("R² gap ≤ 0.02 (current)", "abs", 0.02),
    ("keep ≥ 99% of R²", "retain", 0.99),
    ("keep ≥ 97% of R²", "retain", 0.97),
    ("keep ≥ 95% of R²", "retain", 0.95),
    ("error ≤ +5%", "error", 0.05),
    ("error ≤ +10%", "error", 0.10),
    ("error ≤ +20%", "error", 0.20),
)
SWEEP_PANELS = (
    "R² gap ≤ 0.02 (current)",
    "keep ≥ 97% of R²",
    "keep ≥ 95% of R²",
    "error ≤ +5%",
    "error ≤ +10%",
    "error ≤ +20%",
)
COMPARISON_RULES = ("R² gap ≤ 0.02 (current)", "error ≤ +10%")
POINT_NUMBERS = {
    "Q3.5 27B": "5",
    "Q3.6 27B": "6",
    "Q3.8 27B": "7",
    "OLMo3.1 32B I": "9",
    "OLMo3.1 32B T": "9",
    "Q2.5 32B": "10",
    "Q3 32B": "11",
    "QwQ 32B": "12",
    "OLMo3 32B T": "13",
}
DISPLAY = {
    "Q3.5 27B": "Qwen3.5 27B",
    "Q3.6 27B": "Qwen3.6 27B",
    "Q3.8 27B": "Qwen3.8 27B",
    "OLMo3.1 32B I": "OLMo3.1 32B Instruct",
    "OLMo3.1 32B T": "OLMo3.1 32B Think",
    "Q2.5 32B": "Qwen2.5 32B Instruct",
    "Q3 32B": "Qwen3 32B",
    "QwQ 32B": "QwQ 32B",
    "OLMo3 32B T": "OLMo3 32B Think",
}
POINT_KEY_TEXT = (
    "Point key: 5 Qwen3.5 27B, 6 Qwen3.6 27B, 7 Qwen3.8 27B, 9 OLMo3.1 32B, "
    "10 Qwen2.5 32B, 11 Qwen3 32B, 12 QwQ 32B, 13 OLMo3 32B Think."
)


# ---------------------------------------------------------------------------
# Records: one common shape for both methods
# ---------------------------------------------------------------------------


def records_from_truncated(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for m in payload["maps"]:
        out.append(
            {
                "key": m["key"],
                "model": m["model"],
                "family": m["family"],
                "arm": m["arm"],
                "aa_index": m["aa_index"],
                "aa_status": m["aa_status"],
                "dimension": int(m["dimension"]),
                "full_validation_r2": float(m["reconstruction_parity"]["validation_r2"]),
                # v3 payloads carry the RRR curve as rank_curve and the truncated one alongside.
                "curve": np.asarray(
                    m.get("truncated_rank_curve", m["rank_curve"])["validation_r2"],
                    dtype=np.float64,
                ),
            }
        )
    return out


def records_from_rrr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for m in payload["maps"]:
        out.append(
            {
                "key": m["key"],
                "model": m["model"],
                "family": m["family"],
                "arm": m["arm"],
                "aa_index": m["aa_index"],
                "aa_status": m["aa_status"],
                "dimension": int(m["dimension"]),
                "full_validation_r2": float(m["full_validation_r2"]),
                "curve": np.asarray(m["rank_curve"]["validation_r2"], dtype=np.float64),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Rules and summaries
# ---------------------------------------------------------------------------


def threshold(kind: str, param: float, full_r2: float) -> float:
    if kind == "abs":
        return full_r2 - param
    if kind == "retain":
        return param * full_r2
    if kind == "error":
        return 1.0 - (1.0 - full_r2) * (1.0 + param)
    raise ValueError(kind)


def rank_at(curve: np.ndarray, thr: float) -> int | None:
    idx = np.flatnonzero(curve >= thr - 1e-12)
    return int(idx[0]) if len(idx) else None


def compressibility_index(curve: np.ndarray, full_r2: float, d: int) -> float:
    """Area under retained-R2 fraction versus rank fraction over [0, 25% of d].

    1.0 means the full map's R2 is recovered at rank 0; lower values mean
    more of the width is needed.  Threshold-free."""
    k_max = int(round(AUC_MAX_FRACTION * d))
    if k_max >= len(curve):
        raise ValueError(f"curve too short: {len(curve)} <= {k_max}")
    ks = np.arange(k_max + 1)
    retained = np.clip(curve[: k_max + 1] / full_r2, 0.0, 1.0)
    return float(np.trapezoid(retained, ks / d) / AUC_MAX_FRACTION)


def spearman(x: list[float], y: list[float]) -> dict[str, float] | None:
    if len(x) < 4:
        return None
    res = spearmanr(x, y)
    return {"n": len(x), "rho": float(res.statistic), "p_asymptotic": float(res.pvalue)}


def analyze(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = []
    for r in records:
        curve, full, d = r["curve"], r["full_validation_r2"], r["dimension"]
        rec = {k: v for k, v in r.items() if k != "curve"}
        rec["abs_0.02_gap_as_error_increase_pct"] = 100.0 * 0.02 / (1.0 - full)
        rec["compressibility_index_25pct"] = compressibility_index(curve, full, d)
        rec["ranks"] = {}
        for label, kind, param in RULES:
            k = rank_at(curve, threshold(kind, param, full))
            rec["ranks"][label] = None if k is None else {"rank": k, "rank_fraction": k / d}
        rows.append(rec)

    trends: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        column = [r for r in arm_rows if r["dimension"] == SAME_WIDTH_DIM and r["family"] != "OLMo"]
        same_width = [r for r in arm_rows if r["dimension"] == SAME_WIDTH_DIM]
        groups = (
            ("same_width_qwen_column", column),
            ("same_width_all_families", same_width),
            ("panel", arm_rows),
        )
        trends[arm] = {"rules": {}, "compressibility_index_25pct": {}}
        for label, _kind, _param in RULES:
            entry = {}
            for name, grp in groups:
                usable = [r for r in grp if r["ranks"][label] is not None]
                entry[name] = spearman(
                    [float(r["aa_index"]) for r in usable],
                    [r["ranks"][label]["rank_fraction"] for r in usable],
                )
            trends[arm]["rules"][label] = entry
        for name, grp in groups:
            trends[arm]["compressibility_index_25pct"][name] = spearman(
                [float(r["aa_index"]) for r in grp],
                [r["compressibility_index_25pct"] for r in grp],
            )
    return {"method": method, "method_title": METHOD_TITLE[method], "maps": rows, "trends": trends}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )


def _save(fig, output: Path, meta: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    from PIL import Image

    with Image.open(output) as image:
        image.convert("L").save(output.with_name(f"{output.stem}_grayscale.png"))
    meta["public_url"] = f"https://eps.superkaiba.com/tasks/2588/figure/{output.name}"
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def render_curves(records: list[dict[str, Any]], output: Path, method: str) -> None:
    """Full rank-vs-R2 curves of every same-width map, no cutoff chosen."""
    _style()
    rows = [r for r in records if r["dimension"] == SAME_WIDTH_DIM]
    if not rows:
        print(f"[{method}] no same-width maps yet; curves figure skipped", flush=True)
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    aa_values = sorted({float(r["aa_index"]) for r in rows})
    cmap = plt.get_cmap("viridis")

    def color_of(aa: float):
        span = max(aa_values) - min(aa_values) or 1.0
        return cmap(0.1 + 0.8 * (aa - min(aa_values)) / span)

    for col, arm in enumerate(ARMS):
        arm_rows = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["aa_index"])
        for r in arm_rows:
            curve, full, d = r["curve"], r["full_validation_r2"], r["dimension"]
            ks = np.arange(len(curve))
            x = 100.0 * ks[1:] / d
            ls = "--" if r["family"] == "OLMo" else "-"
            label = (
                f"{POINT_NUMBERS.get(r['model'], '?')}  {DISPLAY.get(r['model'], r['model'])} "
                f"(AA {r['aa_index']:g})"
            )
            c = color_of(float(r["aa_index"]))
            axes[0, col].plot(x, curve[1:] / full, ls, color=c, lw=1.4, label=label)
            axes[1, col].plot(x, (1.0 - curve[1:]) / (1.0 - full), ls, color=c, lw=1.4)
        for q in (0.99, 0.97, 0.95):
            axes[0, col].axhline(q, color="#999999", lw=0.6, ls=":")
            axes[0, col].text(
                25.5, q, f"{int(q * 100)}%", fontsize=6.5, color="#666666", va="center"
            )
        for e in (1.05, 1.10, 1.20):
            axes[1, col].axhline(e, color="#999999", lw=0.6, ls=":")
            axes[1, col].text(
                25.5,
                e,
                f"+{int(round((e - 1) * 100))}%",
                fontsize=6.5,
                color="#666666",
                va="center",
            )
        axes[0, col].set_title(ARM_TITLE[arm], fontsize=10, color=ARM_COLOR[arm], pad=8)
        axes[0, col].set_ylim(0.80, 1.005)
        axes[1, col].set_ylim(0.98, 1.8)
        axes[1, col].set_xlabel("Rank kept (% of hidden dimension 5120)")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xscale("log")
            ax.set_xlim(0.1, 25.0)
            ax.set_xticks([0.1, 0.3, 1, 3, 10, 25])
            ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "25"])
            ax.grid(axis="both", color="#e3e3e3", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=2.0)
    axes[0, 0].set_ylabel("Fraction of full-map validation R² kept")
    axes[1, 0].set_ylabel("Validation SSE ÷ full-map SSE")
    fig.suptitle(
        f"Same-width column, {METHOD_TITLE[method]}: rank needed without choosing a cutoff",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    how = (
        "one map truncated to its top-k coefficient singular directions"
        if method == "truncated"
        else "the best rank-k linear map (fitted map projected onto the top-k principal directions of its fitted outputs)"
    )
    fig.text(
        0.5,
        0.005,
        f"Each curve is {how}. Solid: Qwen; dashed: OLMo. "
        "Color: darker = lower AA index. Dotted guides mark the tolerances used in the sweep figure.",
        ha="center",
        fontsize=7.0,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _save(
        fig,
        output,
        {
            "title": f"Same-width column rank curves ({METHOD_TITLE[method]})",
            "method": method,
            "rows": "every hidden-size-5120 map; top row = retained fraction of full validation R2, bottom row = validation SSE ratio to the full map",
            "x_axis": "rank kept as % of hidden dimension, log scale, 0.1% to 25%",
        },
    )


def render_sweep(analysis: dict[str, Any], output: Path) -> None:
    """Rank fraction vs AA for the same-width column under several tolerances."""
    _style()
    method = analysis["method"]
    records = [r for r in analysis["maps"] if r["dimension"] == SAME_WIDTH_DIM]
    n_cols = len(SWEEP_PANELS)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.3 * n_cols, 5.4), sharex=True)
    for row, arm in enumerate(ARMS):
        arm_rows = [r for r in records if r["arm"] == arm]
        for col, label in enumerate(SWEEP_PANELS):
            ax = axes[row, col]
            column = sorted(
                (r for r in arm_rows if r["family"] != "OLMo" and r["ranks"][label] is not None),
                key=lambda r: r["aa_index"],
            )
            ax.plot(
                [r["aa_index"] for r in column],
                [100.0 * r["ranks"][label]["rank_fraction"] for r in column],
                color=ARM_COLOR[arm],
                lw=1.6,
                alpha=0.9,
                zorder=1,
            )
            for r in arm_rows:
                if r["ranks"][label] is None:
                    continue
                x, y = float(r["aa_index"]), 100.0 * r["ranks"][label]["rank_fraction"]
                face = ARM_COLOR[arm] if r["aa_status"] == "measured" else "white"
                ax.scatter(
                    x,
                    y,
                    s=34,
                    marker="^" if r["family"] == "OLMo" else "s",
                    facecolor=face,
                    edgecolor=ARM_COLOR[arm],
                    linewidth=1.1,
                    zorder=3,
                )
                ax.annotate(
                    POINT_NUMBERS.get(r["model"], "?"),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.2,
                    fontweight="bold",
                    color="#222222",
                )
            stats = analysis["trends"][arm]["rules"][label]["same_width_qwen_column"]
            rho = f"ρ = {stats['rho']:+.2f}" if stats else "ρ n/a"
            ax.set_title(f"{label}\nQwen column {rho}", fontsize=7.8, pad=5)
            ax.set_xlim(0, 56)
            ax.grid(axis="y", color="#e3e3e3", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("AA index", fontsize=8)
        axes[row, 0].set_ylabel(
            f"{ARM_TITLE[arm]}\nrank kept (% of width)", fontsize=8.5, color=ARM_COLOR[arm]
        )
    fig.suptitle(
        f"Same-width column, {METHOD_TITLE[method]}: performance-preserving rank under different tolerances",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Squares: Qwen (line connects them); triangles: OLMo. Filled: measured AA; open: estimated. "
        + POINT_KEY_TEXT,
        ha="center",
        fontsize=6.9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    _save(
        fig,
        output,
        {
            "title": f"Same-width column rank under different tolerances ({METHOD_TITLE[method]})",
            "method": method,
            "rules": list(SWEEP_PANELS),
            "rule_definitions": {
                "R² gap": "full validation R2 minus rank-k validation R2, absolute",
                "keep ≥ q of R²": "rank-k validation R2 >= q times full validation R2",
                "error ≤ +e": "rank-k validation SSE <= (1 + e) times full-map validation SSE",
            },
        },
    )


def render_method_comparison(trunc: dict[str, Any], rrr: dict[str, Any], output: Path) -> None:
    """Truncated ridge vs reduced-rank regression, same-width rows, two rules."""
    _style()
    by_key = {r["key"]: r for r in rrr["maps"]}
    fig, axes = plt.subplots(
        2, len(COMPARISON_RULES), figsize=(4.6 * len(COMPARISON_RULES), 6.2), sharex=True
    )
    for row, arm in enumerate(ARMS):
        rows = [
            r
            for r in trunc["maps"]
            if r["dimension"] == SAME_WIDTH_DIM and r["arm"] == arm and r["key"] in by_key
        ]
        for col, label in enumerate(COMPARISON_RULES):
            ax = axes[row, col]
            for r in rows:
                t, s = r["ranks"][label], by_key[r["key"]]["ranks"][label]
                if t is None or s is None:
                    continue
                x = float(r["aa_index"])
                yt, ys = 100.0 * t["rank_fraction"], 100.0 * s["rank_fraction"]
                marker = "^" if r["family"] == "OLMo" else "s"
                ax.plot([x, x], [yt, ys], color=ARM_COLOR[arm], lw=1.0, alpha=0.6, zorder=1)
                ax.scatter(
                    x,
                    yt,
                    s=38,
                    marker=marker,
                    facecolor="white",
                    edgecolor=ARM_COLOR[arm],
                    linewidth=1.2,
                    zorder=3,
                )
                ax.scatter(
                    x,
                    ys,
                    s=38,
                    marker=marker,
                    facecolor=ARM_COLOR[arm],
                    edgecolor=ARM_COLOR[arm],
                    linewidth=1.2,
                    zorder=4,
                )
                ax.annotate(
                    POINT_NUMBERS.get(r["model"], "?"),
                    (x, ys),
                    xytext=(4, -3),
                    textcoords="offset points",
                    fontsize=6.6,
                    fontweight="bold",
                    color="#222222",
                )
            column = sorted((r for r in rows if r["family"] != "OLMo"), key=lambda r: r["aa_index"])
            ax.plot(
                [r["aa_index"] for r in column],
                [100.0 * by_key[r["key"]]["ranks"][label]["rank_fraction"] for r in column],
                color=ARM_COLOR[arm],
                lw=1.6,
                zorder=2,
            )
            ax.plot(
                [r["aa_index"] for r in column],
                [100.0 * r["ranks"][label]["rank_fraction"] for r in column],
                color=ARM_COLOR[arm],
                lw=1.0,
                ls=":",
                zorder=2,
            )
            st = trunc["trends"][arm]["rules"][label]["same_width_qwen_column"]
            sr = rrr["trends"][arm]["rules"][label]["same_width_qwen_column"]
            fmt = lambda v: f"{v['rho']:+.2f}" if v else "n/a"  # noqa: E731
            ax.set_title(
                f"{label}\nQwen column ρ: truncated {fmt(st)}, reduced-rank {fmt(sr)}",
                fontsize=8.2,
                pad=5,
            )
            ax.set_xlim(0, 56)
            ax.grid(axis="y", color="#e3e3e3", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("AA index")
        axes[row, 0].set_ylabel(f"{ARM_TITLE[arm]}\nrank kept (% of width)", color=ARM_COLOR[arm])
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=0,
            marker="s",
            markerfacecolor="white",
            markersize=6,
            label="truncated ridge map (dotted line)",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=0,
            marker="s",
            markersize=6,
            label="reduced-rank regression (solid line)",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            lw=0,
            marker="^",
            markerfacecolor="white",
            markersize=6,
            label="OLMo rows (same width)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        fontsize=8,
        ncols=3,
    )
    fig.suptitle(
        "Same-width column: truncating the fitted map vs the best rank-k map",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Each vertical segment joins the two methods for one model. " + POINT_KEY_TEXT,
        ha="center",
        fontsize=7.0,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(
        fig,
        output,
        {
            "title": "Same-width column: truncated ridge vs reduced-rank regression",
            "rules": list(COMPARISON_RULES),
            "reading": "open marker = truncated ridge rank, filled = reduced-rank regression rank; lower is more compressible",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncated-json", type=Path, default=DEFAULT_TRUNC)
    ap.add_argument("--rrr-json", type=Path, default=DEFAULT_RRR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    analyses: dict[str, dict[str, Any]] = {}
    for method, path, suffix, from_fn in (
        ("truncated", args.truncated_json, "", records_from_truncated),
        ("rrr", args.rrr_json, "_rrr", records_from_rrr),
    ):
        if not path.exists():
            print(f"[{method}] {path} absent; skipped", flush=True)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = from_fn(payload)
        analysis = analyze(records, method)
        analysis["source"] = {
            "input": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "n_maps": len(records),
            "rules": [{"label": a, "kind": b, "param": c} for a, b, c in RULES],
            "compressibility_index": (
                "area under (rank-k validation R2 / full R2) versus rank fraction over [0, 0.25], "
                "divided by 0.25; 1.0 = full R2 at rank 0"
            ),
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"rank_threshold_sweep{suffix}.json").write_text(
            json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
        )
        render_curves(records, args.fig_dir / f"rank_curves_same_width{suffix}.png", method)
        render_sweep(analysis, args.fig_dir / f"rank_threshold_sweep{suffix}.png")
        analyses[method] = analysis
        print(f"[{method}] {len(records)} maps -> sweep JSON + 2 figures", flush=True)
    if {"truncated", "rrr"} <= set(analyses):
        render_method_comparison(
            analyses["truncated"], analyses["rrr"], args.fig_dir / "rank_method_comparison.png"
        )
        print("[compare] rank_method_comparison.png", flush=True)


if __name__ == "__main__":
    main()
