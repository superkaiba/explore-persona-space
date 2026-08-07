"""Issue #2094 — transport-agreement heatmaps (banked-map prediction vs realized shift).

Reads ``eval_results/issue_2094/transport/transport_cells.jsonl`` (one row per
grid rollout at the banked-map cells) and renders, per arm, a heatmap of the
mean cosine between the REALIZED answer-vector shift and the BANKED-MAP
predicted shift (``cosine_tail``), pooled per (setting, dose, banked-map
column), where the six banked-map columns are the ce (context->answer map,
m779_ce_*) and pe (prefix->answer map, m1738_pe_*) maps at layers 14/19/26:

* ``transport_agreement_heatmap_steered`` — steered arm;
* ``transport_agreement_heatmap_null``    — donor-null arm (in-design control;
  SAME color scale as the steered figure so the two compare directly);
* ``transport_agreement_heatmap_delta``   — steered-minus-null per cell.

Conventions mirror ``scripts/issue2094_figures.py::fig_f_heatmap``: setting
columns x dose rows, RdBu_r diverging cmap symmetric about 0, greyed missing
cells, plain-English labels, degenerate-by-design matched-prefix x prefix-end
cells (which carry no cosine in the artifact) marked with an open circle and
excluded from every pooled mean, per-figure exclusion footnote.

Usage (VM; shared-VM thread caps per #847):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_transport_heatmap_fig.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps must bind BEFORE any heavy import (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402
from issue2094_figures import (  # noqa: E402
    DOSE_LABELS,
    DOSE_ORDER,
    SETTING_LABELS,
    SETTING_ORDER,
)

# ── artifact contract (fail-loud; verified against the committed artifact) ──

EXPECTED_ROWS = 3960
EXPECTED_DEGENERATE = 900  # matched_prefix x pe, all doses — cosine_tail is null there
MIN_CELL_N = 5  # cells thinner than this are greyed out
ARM_ORDER: tuple[str, ...] = ("steered", "null")
ARM_LABELS: dict[str, str] = {
    "steered": "steered arm (banked-map payload injected)",
    "null": "donor-null arm (in-design control: donor payload)",
}
# the six banked-map columns: (slot, layer); ce = context->answer map (m779_ce_*),
# pe = prefix->answer map (m1738_pe_*)
MAP_COLS: tuple[tuple[str, int], ...] = (
    ("ce", 14),
    ("ce", 19),
    ("ce", 26),
    ("pe", 14),
    ("pe", 19),
    ("pe", 26),
)
MAP_LINEAGE: dict[str, str] = {"ce": "context->answer", "pe": "prefix->answer"}
# lineage lives in the divider + supxlabel (two-line per-tick labels overlap at 6 columns)
MAP_COL_LABELS: list[str] = [f"L{layer}" for _, layer in MAP_COLS]

CellKey = tuple[str, str, str, str, int]  # (arm, setting, dose, slot, layer)


def load_rows(path: Path) -> list[dict]:
    """Text-mode line iteration (never ``splitlines()`` — JSONL gotcha)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit_rows(rows: list[dict]) -> list[dict]:
    """Fail-loud artifact-convention audit; returns the INCLUDED row set
    (non-degenerate, finite cosine)."""
    assert len(rows) == EXPECTED_ROWS, f"row count {len(rows)} != {EXPECTED_ROWS}"

    degen = [r for r in rows if r["degenerate_self"]]
    assert len(degen) == EXPECTED_DEGENERATE, (
        f"degenerate rows {len(degen)} != {EXPECTED_DEGENERATE}"
    )
    bad_degen = {(r["setting"], r["slot"]) for r in degen} - {("matched_prefix", "pe")}
    assert not bad_degen, f"degenerate rows outside matched_prefix x pe: {bad_degen}"

    # artifact convention: the null-cosine rows are EXACTLY the degenerate rows
    # (structurally-zero contrasts carry no cosine)
    null_cos = [r for r in rows if r["cosine_tail"] is None]
    assert len(null_cos) == EXPECTED_DEGENERATE, f"null cosines {len(null_cos)}"
    assert all(r["degenerate_self"] for r in null_cos), "non-degenerate row with null cosine"

    included = [r for r in rows if not r["degenerate_self"] and r["cosine_tail"] is not None]
    assert len(included) == EXPECTED_ROWS - EXPECTED_DEGENERATE, len(included)
    for r in included:
        c = r["cosine_tail"]
        assert math.isfinite(c) and -1.0 - 1e-9 <= c <= 1.0 + 1e-9, f"cosine out of range: {c}"
    per_arm = {arm: sum(1 for r in included if r["arm"] == arm) for arm in ARM_ORDER}
    assert set(per_arm) == set(ARM_ORDER) and len({*per_arm.values()}) == 1, per_arm
    print(
        f"[audit] rows={len(rows)} degenerate_excluded={len(degen)} "
        f"null_cosines_dropped={len(null_cos)} (all degenerate) included={len(included)} "
        f"per-arm={per_arm}",
        flush=True,
    )
    return included


def pool_cells(included: list[dict]) -> dict[CellKey, dict]:
    """(arm, setting, dose, slot, layer) -> {'mean','n'} over included rows,
    pooled over vec_type (A+B) and rollouts."""
    acc: dict[CellKey, list[float]] = {}
    for r in included:
        key: CellKey = (r["arm"], r["setting"], r["dose"], r["slot"], r["layer"])
        acc.setdefault(key, []).append(float(r["cosine_tail"]))
    cells = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}

    assert sum(c["n"] for c in cells.values()) == len(included), "pooled n does not sum to total"
    # coverage: full grid minus the degenerate matched_prefix x pe combos
    expected_keys = {
        (arm, setting, dose, slot, layer)
        for arm in ARM_ORDER
        for setting in SETTING_ORDER
        for dose in DOSE_ORDER
        for slot, layer in MAP_COLS
        if not (setting == "matched_prefix" and slot == "pe")
    }
    assert set(cells) == expected_keys, (
        f"cell coverage mismatch: missing={expected_keys - set(cells)} "
        f"extra={set(cells) - expected_keys}"
    )
    return cells


def recompute_cell_mean(rows: list[dict], key: CellKey) -> float:
    """Independent recompute straight off the raw rows (separate code path from
    pool_cells) — used to assert the drawn values."""
    arm, setting, dose, slot, layer = key
    vals = [
        float(r["cosine_tail"])
        for r in rows
        if r["arm"] == arm
        and r["setting"] == setting
        and r["dose"] == dose
        and r["slot"] == slot
        and r["layer"] == layer
        and not r["degenerate_self"]
        and r["cosine_tail"] is not None
    ]
    assert vals, f"no rows for {key}"
    return sum(vals) / len(vals)


def panel_matrices(
    cells: dict[CellKey, dict], arm: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per setting: (5 doses x 6 map columns) mean matrix (NaN = absent or n<MIN_CELL_N)
    and the matching n matrix (0 = absent)."""
    mats, nmats = {}, {}
    for setting in SETTING_ORDER:
        mat = np.full((len(DOSE_ORDER), len(MAP_COLS)), np.nan)
        nmat = np.zeros((len(DOSE_ORDER), len(MAP_COLS)), dtype=int)
        for i, dose in enumerate(DOSE_ORDER):
            for j, (slot, layer) in enumerate(MAP_COLS):
                cell = cells.get((arm, setting, dose, slot, layer))
                if cell is None:
                    continue
                nmat[i, j] = cell["n"]
                if cell["n"] >= MIN_CELL_N:
                    mat[i, j] = cell["mean"]
        mats[setting] = mat
        nmats[setting] = nmat
    return mats, nmats


def _draw_grid(
    mats: dict[str, np.ndarray],
    nmats: dict[str, np.ndarray],
    vmax: float,
    cbar_label: str,
    suptitle: str,
    footnote: str,
) -> plt.Figure:
    """One row of setting panels, each a dose-rows x map-columns heatmap."""
    fig, axes = plt.subplots(
        1,
        len(SETTING_ORDER),
        figsize=(3.6 * len(SETTING_ORDER) + 1.6, 4.6),
        squeeze=False,
        layout="constrained",
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    im = None
    for j, setting in enumerate(SETTING_ORDER):
        ax = axes[0][j]
        mat, nmat = mats[setting], nmats[setting]
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        deg_x, deg_y = [], []
        for i in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                if np.isfinite(mat[i, k]):
                    txt = "white" if abs(mat[i, k]) > 0.6 * vmax else "black"
                    ax.text(
                        k,
                        i - 0.12,
                        f"{mat[i, k]:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=txt,
                    )
                    ax.text(
                        k,
                        i + 0.3,
                        f"n={nmat[i, k]}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color=txt if txt == "white" else "dimgray",
                    )
                elif nmat[i, k] > 0:  # present but under the n floor
                    ax.text(
                        k,
                        i,
                        f"n={nmat[i, k]}<{MIN_CELL_N}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="dimgray",
                    )
                else:  # degenerate-by-design (marked, never silently blank)
                    deg_x.append(k)
                    deg_y.append(i)
        if deg_x:
            ax.scatter(
                deg_x,
                deg_y,
                marker="o",
                s=70,
                linewidths=1.4,
                facecolors="none",
                edgecolors="black",
                zorder=5,
            )
        ax.axvline(2.5, color="black", linewidth=1.0)  # ce | pe map-lineage divider
        ax.set_xticks(range(len(MAP_COLS)))
        ax.set_xticklabels(MAP_COL_LABELS, fontsize=6)
        ax.set_yticks(range(len(DOSE_ORDER)))
        ax.set_yticklabels(
            [DOSE_LABELS[d] for d in DOSE_ORDER] if j == 0 else [""] * len(DOSE_ORDER),
            fontsize=7,
        )
        ax.grid(False)
        ax.set_title(SETTING_LABELS[setting], fontsize=8)
    fig.supxlabel(
        "banked map layer (left of divider: context->answer map m779; "
        "right: prefix->answer map m1738)",
        fontsize=8,
    )
    fig.colorbar(im, ax=axes, shrink=0.8, label=cbar_label)
    fig.suptitle(suptitle, fontsize=9)
    fig.text(0.005, 0.002, footnote, fontsize=6, color="dimgray")
    return fig


def headline_stats(included: list[dict]) -> dict:
    """Per-arm overall means, steered best/worst map column, norm-ratio medians."""
    out: dict = {}
    for arm in ARM_ORDER:
        arm_rows = [r for r in included if r["arm"] == arm]
        out[f"{arm}_overall_mean"] = float(np.mean([r["cosine_tail"] for r in arm_rows]))
        out[f"{arm}_norm_ratio_median"] = float(
            np.median([r["pred_norm"] / r["realized_norm"] for r in arm_rows])
        )
    col_means = {}
    for slot, layer in MAP_COLS:
        vals = [
            r["cosine_tail"]
            for r in included
            if r["arm"] == "steered" and r["slot"] == slot and r["layer"] == layer
        ]
        col_means[f"{slot}-L{layer}"] = float(np.mean(vals))
    out["steered_column_means"] = col_means
    out["steered_best_column"] = max(col_means, key=col_means.get)
    out["steered_worst_column"] = min(col_means, key=col_means.get)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--transport",
        type=Path,
        default=Path("eval_results/issue_2094/transport/transport_cells.jsonl"),
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2094"))
    ap.add_argument("--style", choices=("blog", "neurips", "generic"), default="blog")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_paper_style(args.style)
    rows = load_rows(args.transport)
    included = audit_rows(rows)
    cells = pool_cells(included)

    # assert-the-drawn-values: independent recompute of three fixed cells
    for key in (
        ("steered", "cross", "replace", "ce", 26),
        ("steered", "matched_query", "a1", "pe", 14),
        ("null", "cross", "a4", "ce", 19),
    ):
        indep = recompute_cell_mean(rows, key)
        assert np.isclose(indep, cells[key]["mean"], rtol=0, atol=1e-12), (
            f"recompute mismatch at {key}: {indep} vs {cells[key]['mean']}"
        )
    print("[audit] independent per-cell mean recompute: 3/3 match", flush=True)

    arm_mats = {arm: panel_matrices(cells, arm) for arm in ARM_ORDER}
    # ONE symmetric color scale pooled across BOTH arms so the figures compare
    finite = [c["mean"] for c in cells.values()]
    vmax = float(np.max(np.abs(finite)))
    assert vmax > 0, "degenerate color scale"

    ns = [c["n"] for c in cells.values()]
    stats = headline_stats(included)
    print(f"[stats] per-cell n range: {min(ns)}-{max(ns)}; shared vmax={vmax:.4f}", flush=True)
    print(f"[stats] {json.dumps(stats, indent=2)}", flush=True)

    cbar = "mean cosine (realized vs map-predicted shift)"
    footnote = (
        f"{EXPECTED_DEGENERATE} degenerate-by-design self-transfer rows (matched-prefix x "
        "prefix-end, all doses; open circles — no cosine in the artifact) excluded; pooled "
        "over vector types A+B and rollouts; direction-only agreement (norms not compared)."
    )
    fig_dir = args.fig_dir
    for arm in ARM_ORDER:
        mats, nmats = arm_mats[arm]
        fig = _draw_grid(
            mats,
            nmats,
            vmax,
            cbar,
            f"Transport agreement — {ARM_LABELS[arm]}; shared color scale across arms",
            footnote,
        )
        paths = savefig_paper(fig, f"transport_agreement_heatmap_{arm}", dir=fig_dir)
        plt.close(fig)
        print(f"[figures] saved transport_agreement_heatmap_{arm} -> {paths.get('png')}")

    # delta view: steered minus null per cell (the map's signal above the donor-null control)
    d_mats, d_nmats = {}, {}
    for setting in SETTING_ORDER:
        sm, sn = arm_mats["steered"][0][setting], arm_mats["steered"][1][setting]
        nm, nn = arm_mats["null"][0][setting], arm_mats["null"][1][setting]
        d_mats[setting] = sm - nm  # NaN propagates through the degenerate cells
        d_nmats[setting] = np.minimum(sn, nn)
    d_finite = np.concatenate([m[np.isfinite(m)] for m in d_mats.values()])
    d_vmax = float(np.max(np.abs(d_finite)))
    fig = _draw_grid(
        d_mats,
        d_nmats,
        d_vmax,
        "steered minus donor-null mean cosine",
        "Transport agreement — steered minus donor-null (per-cell delta; n = per-arm cell n)",
        footnote,
    )
    paths = savefig_paper(fig, "transport_agreement_heatmap_delta", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] saved transport_agreement_heatmap_delta -> {paths.get('png')}")
    print("[phase=transport_heatmaps_done] saved=3", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
