"""Issue #2587 unit 6 — P10 figures: layer-sweep hero, matched-n table, exploratory dump.

Consumes ONLY committed / harvested JSON artifacts (no torch, no GPU, no
network): the unit-4 ``map_layer_sweep.json`` + ``matched7b_anchor.json``,
the unit-5b ``minpair_delta_2587.json`` + ``crossmodel_contrasts.json``, and
the banked #2330 reference fits (``eval_results/issue_2330/matched_fits_*``).

Figures (plan §6):

* ``hero_layer_sweep`` — held-out test R² vs FRACTIONAL depth: the fresh 9B
  n≈25k curve over all 32 layers (L* starred), the #2330 9B n=10k points
  (layers 16/22/30), the #2330 7B n=10k points (layers 14/19/26), the 7B
  n=25k L19 anchor, the strongest-floor envelope, and the #2329 convention's
  full-attention dashed verticals (Qwen3.5-9B geometry only).
* ``matched_n_table`` — 9B@L* vs 7B@L19 at matched n≈25k (test/val/wc R²,
  floors, kNN retrieval, two-draw ceilings, anchor gate, paired H1 delta),
  written as ``table_matched_n.md`` + ``table_matched_n.json``.
* Exploratory dump (over-produced, all committed): selected-λ-per-layer,
  floors-per-layer, wc-transfer vs in-corpus, kNN acc@k per layer,
  reliability ceilings, the cross-model per-axis profile (hero 1 of §6) and
  the per-axis cross-model delta forest.

Display names live in ONE label map (``DISPLAY``) — internal arm/model slugs
never reach an axis, legend, or table header. Figures carry no on-canvas
caption blocks (axes + ticks + legend + panel titles only); provenance goes
to the ``savefig_paper`` sidecars.

CLI examples:
  uv run python scripts/issue2587_figures.py                      # all figures
  uv run python scripts/issue2587_figures.py --figs hero_layer_sweep,matched_n_table
  uv run python scripts/issue2587_figures.py --out-dir /tmp/issue-2587-smoke/figs \
      --sweep-json /tmp/fixtures/map_layer_sweep.json ...
  uv run python scripts/issue2587_figures.py --import-check
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from explore_persona_space.orchestrate.env import load_dotenv

# Thread caps + creds BEFORE numpy/matplotlib import (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

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

logger = logging.getLogger("issue2587_figures")

ISSUE = 2587

# ── geometry + conventions ─────────────────────────────────────────────
# Mirrors scripts/issue2329_figures.py:79 (the #2329 dash-mark convention);
# pinned against that source by tests/test_issue2587_prefixes.py.
N_LAYERS_9B = 32
FULL_ATTENTION_LAYERS_9B = frozenset({3, 7, 11, 15, 19, 23, 27, 31})
FULL_ATTN_COLOR = "#9467bd"
N_LAYERS_7B = 28  # Qwen2.5-7B-Instruct decoder blocks (fraction = layer / 27)
L19 = 19

# The #2330 committed port-parity anchor (same-surface 7B n=25k L19 refit;
# plan §7 anchor gate |R² − 0.7250873| ≤ 0.01; also in issue2587_fits.py).
ANCHOR_7B_25K_R2 = 0.7250873220237553
ANCHOR_TOL = 0.01

# ── display-name label map (ONE map; no internal slugs on any figure) ──
DISPLAY: dict[str, str] = {
    # models / curves
    "qwen35_9b": "Qwen3.5-9B (thinking off)",
    "qwen25_7b": "Qwen2.5-7B-Instruct",
    "fresh_9b_25k": "Qwen3.5-9B, n=25k (this run)",
    "ref_9b_10k": "Qwen3.5-9B, n=10k (prior fit)",
    "ref_7b_10k": "Qwen2.5-7B, n=10k (prior fit)",
    "anchor_7b_25k": "Qwen2.5-7B, n=25k (anchor)",
    "lstar": "selected layer L*",
    "floor_envelope": "strongest baseline floor",
    # map arms (cross-model panels)
    "arm_fresh9b": "Qwen3.5-9B fresh map",
    "arm_7b_matched25k": "Qwen2.5-7B matched-n map",
    "ref_7b_parent": "Qwen2.5-7B parent map (reference)",
    # floors
    "identity_bias": "identity + learned bias",
    "identity_copy": "identity (copy input)",
    "scaled_identity": "scaled identity",
    "shuffled_pairing": "shuffled pairing",
    "train_mean": "train mean",
    # cross-model statistics (panel titles)
    "direction_cos": "direction cosine",
    "calibration_ratio_to_global": "calibration ratio (axis / global)",
    "obs_separation_snr": "observed separation / split-half noise",
    "crossfam_cos_observed": "cross-family consistency (observed)",
    "crossfam_cos_maparm": "cross-family consistency (predicted)",
    "axis_identity_cos": "axis identity cosine",
}

_AXIS_OVERRIDES = {
    "answer_language": "Answer language",
    "answer_length": "Answer length",
}


def axis_label(axis: str) -> str:
    """Reader-facing name for a battery axis (no internal shorthand)."""
    return _AXIS_OVERRIDES.get(axis, axis.replace("_", " ").capitalize())


# Shared depth-axis label (one string for every per-layer figure; short enough
# to render un-clipped at the narrow 6.0-in exploratory figsize under the
# neurips constrained_layout style, which does not shrink over-wide labels).
XLABEL_DEPTH = "depth (fraction of stack; dashed = full-attention layers)"


def _err_offsets(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """CI bounds -> NON-NEGATIVE matplotlib offsets (gotchas.md xerr/yerr rule)."""
    lo_off = np.maximum(0.0, np.asarray(vals) - np.asarray(lo))
    hi_off = np.maximum(0.0, np.asarray(hi) - np.asarray(vals))
    return np.vstack([lo_off, hi_off])


def _fracs(layers: list[int], n_layers: int) -> np.ndarray:
    return np.asarray(layers, dtype=np.float64) / max(n_layers - 1, 1)


def _mark_full_attention(ax, n_layers: int) -> None:
    """#2329 convention: dashed verticals at the full-attention layers; only
    meaningful on the 32-block Qwen3.5-9B geometry (skips otherwise)."""
    if n_layers != N_LAYERS_9B:
        return
    for layer in sorted(FULL_ATTENTION_LAYERS_9B):
        ax.axvline(layer / (n_layers - 1), color=FULL_ATTN_COLOR, ls="--", lw=0.6, zorder=0)


# ── input loading ──────────────────────────────────────────────────────


def _load_json(path: Path, what: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what} missing: {p} (fail-loud — no placeholder figures)")
    return json.loads(p.read_text(encoding="utf-8"))


def sweep_layers(sweep: dict) -> list[int]:
    """Sorted layer indices of the sweep's per_layer map (fail-loud on empty)."""
    layers = sorted(int(k) for k in sweep["per_layer"])
    if not layers:
        raise RuntimeError("map_layer_sweep.json has an empty per_layer map")
    return layers


def _sweep_series(sweep: dict, getter) -> tuple[np.ndarray, np.ndarray]:
    layers = sweep_layers(sweep)
    n_layers = int(sweep["n_layers"])
    ys = np.asarray([getter(sweep["per_layer"][str(li)]) for li in layers], dtype=np.float64)
    return _fracs(layers, n_layers), ys


def _floor_envelope(row: dict) -> float:
    vals = [
        float(rec["test_r2"])
        for rec in row["floors"].values()
        if rec.get("test_r2") is not None and np.isfinite(float(rec["test_r2"]))
    ]
    if not vals:
        raise RuntimeError(f"layer {row.get('layer')}: no finite floor test_r2 values")
    return max(vals)


# ── figures ────────────────────────────────────────────────────────────


def fig_hero_layer_sweep(inputs: dict, out_dir: Path) -> list[Path]:
    """Hero 2 (plan §6): held-out test R² vs fractional depth, both models."""
    sweep = inputs["sweep"]
    ref9 = inputs["ref9b_n10k"]
    ref7 = inputs["ref7b_n10k"]
    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n_layers = int(sweep["n_layers"])
    _mark_full_attention(ax, n_layers)

    x9, y9 = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    ax.plot(x9, y9, color=c9, marker="o", ms=3.5, lw=1.6, label=DISPLAY["fresh_9b_25k"])

    _, yfl = _sweep_series(sweep, _floor_envelope)
    ax.plot(x9, yfl, color=neutral, ls="--", lw=1.1, label=DISPLAY["floor_envelope"])

    lstar = int(sweep["lstar"]["lstar"])
    ax.plot(
        [lstar / (n_layers - 1)],
        [float(sweep["per_layer"][str(lstar)]["ridge"]["test_r2"])],
        marker="*",
        ms=14,
        color=c9,
        mec="black",
        mew=0.6,
        ls="none",
        label=DISPLAY["lstar"],
    )

    l9r = [int(x) for x in ref9["layers"]]
    y9r = [float(ref9["per_layer"][str(li)]["ridge"]["test_r2"]) for li in l9r]
    ax.plot(
        _fracs(l9r, N_LAYERS_9B),
        y9r,
        color=c9,
        marker="s",
        mfc="none",
        ms=6,
        ls=":",
        lw=1.0,
        label=DISPLAY["ref_9b_10k"],
    )

    l7r = [int(x) for x in ref7["layers"]]
    y7r = [float(ref7["per_layer"][str(li)]["ridge"]["test_r2"]) for li in l7r]
    ax.plot(
        _fracs(l7r, N_LAYERS_7B),
        y7r,
        color=c7,
        marker="D",
        mfc="none",
        ms=6,
        ls=":",
        lw=1.0,
        label=DISPLAY["ref_7b_10k"],
    )
    ax.plot(
        [L19 / (N_LAYERS_7B - 1)],
        [ANCHOR_7B_25K_R2],
        marker="D",
        ms=8,
        color=c7,
        ls="none",
        label=DISPLAY["anchor_7b_25k"],
    )

    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out test R²")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    paths = savefig_paper(fig, "fig_hero_layer_sweep", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def _knn_cell(knn: dict, arm: str, metric: str, k: int) -> float:
    return float(knn[arm][metric]["acc_at_k"][str(k)])


def _table_side(row: dict, ceiling: dict | None) -> dict:
    """One matched-n table side from a per-layer-row-shaped dict (9B sweep row
    or the matched7b record's ``arm`` block — same schema by construction)."""
    meta = row["ridge"]["meta"] if "meta" in row["ridge"] else row["ridge_meta"]
    knn = row["knn"]
    side = {
        "n_train": int(row["n_train"]),
        "d": int(row["d"]),
        "val_r2_at_selected": float(meta["val_r2_at_selected"]),
        "test_r2": float(row["ridge"]["test_r2"]) if "ridge" in row else float(row["test_r2"]),
        "wc_test_1k_r2": float(
            row["ridge"]["wc_test_1k_r2"] if "ridge" in row else row["wc_test_1k_r2"]
        ),
        "selected_lambda": float(meta["selected_lambda"]),
        "floors": {
            name: (None if rec.get("test_r2") is None else float(rec["test_r2"]))
            for name, rec in row["floors"].items()
        },
        "knn_ridge_acc_at_1_euclid": _knn_cell(knn, "ridge", "euclidean", 1),
        "knn_ridge_acc_at_10_euclid": _knn_cell(knn, "ridge", "euclidean", 10),
        "knn_ridge_acc_at_1_cosine": _knn_cell(knn, "ridge", "cosine", 1),
        "knn_ridge_acc_at_10_cosine": _knn_cell(knn, "ridge", "cosine", 10),
        "knn_chance_at_1": float(knn["ridge"]["euclidean"]["chance_at_k"]["1"]),
    }
    if ceiling is not None and ceiling.get("available"):
        side["two_draw_ceiling_r"] = float(ceiling["ceiling_var_weighted_r"])
    else:
        side["two_draw_ceiling_r"] = None
    return side


def matched_n_table(inputs: dict, out_dir: Path) -> list[Path]:
    """Matched-n comparison table: 9B@L* vs 7B@L19 (md + json)."""
    sweep = inputs["sweep"]
    m7 = inputs["matched7b"]
    delta = inputs.get("delta")

    lstar = int(sweep["lstar"]["lstar"])
    row9 = sweep["per_layer"][str(lstar)]
    # matched7b record's arm block: reshape to the per-layer row shape.
    arm7 = m7["arm"]
    row7 = {
        "n_train": arm7["n_train"],
        "d": arm7["d"],
        "ridge": {
            "meta": arm7["ridge_meta"],
            "test_r2": arm7["test_r2"],
            "wc_test_1k_r2": arm7["wc_test_1k_r2"],
        },
        "floors": arm7["floors"],
        "knn": arm7["knn"],
    }
    side9 = _table_side(row9, sweep["reliability_ceiling"]["by_layer"].get(str(lstar)))
    side7 = _table_side(row7, m7.get("ceiling_7b_matched_L19"))
    anchor = m7["anchor"]

    doc: dict = {
        "issue": ISSUE,
        "layer_pair": {"qwen35_9b": lstar, "qwen25_7b": L19},
        "sides": {"qwen35_9b": side9, "qwen25_7b": side7},
        "anchor_gate": {
            "expected_r2": float(anchor["expected_r2"]),
            "realized_r2": float(anchor["realized_r2"]),
            "abs_deviation": float(anchor["abs_deviation"]),
            "tol": float(anchor["tol"]),
        },
    }
    if delta is not None:
        doc["h1_paired_shared_rows"] = {
            k: delta["h1"][k]
            for k in ("r2_9b_lstar", "r2_7b_l19", "delta_map", "delta_ci95", "verdict")
        }

    name9 = f"{DISPLAY['qwen35_9b']} @ layer {lstar} (L*)"
    name7 = f"{DISPLAY['qwen25_7b']} @ layer {L19}"

    def _fmt(v) -> str:
        if v is None:
            return "n/a"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        f"# Matched-n map comparison — {name9} vs {name7}",
        "",
        "All fits at matched n (train≈25k / val 400 / test 1,000), fp64 primal ridge,",
        "λ val-selected; retrieval = P(true target in the k nearest neighbours of the",
        "prediction) over the held-out test pool.",
        "",
        f"| quantity | {name9} | {name7} |",
        "|---|---|---|",
    ]
    rows_spec = [
        ("train rows (n)", "n_train"),
        ("hidden dim (d)", "d"),
        ("validation R² (selected λ)", "val_r2_at_selected"),
        ("held-out test R²", "test_r2"),
        ("held-out transfer R² (WildChat)", "wc_test_1k_r2"),
        ("selected λ", "selected_lambda"),
        ("retrieval acc@1 (euclidean)", "knn_ridge_acc_at_1_euclid"),
        ("retrieval acc@10 (euclidean)", "knn_ridge_acc_at_10_euclid"),
        ("retrieval acc@1 (cosine)", "knn_ridge_acc_at_1_cosine"),
        ("retrieval acc@10 (cosine)", "knn_ridge_acc_at_10_cosine"),
        ("retrieval chance @1", "knn_chance_at_1"),
        ("two-draw reliability ceiling (r)", "two_draw_ceiling_r"),
    ]
    for label, key in rows_spec:
        lines.append(f"| {label} | {_fmt(side9[key])} | {_fmt(side7[key])} |")
    for fname in sorted(side9["floors"]):
        lines.append(
            f"| floor: {DISPLAY.get(fname, fname)} R² | "
            f"{_fmt(side9['floors'][fname])} | {_fmt(side7['floors'].get(fname))} |"
        )
    lines += [
        "",
        f"Anchor gate: realized R² {_fmt(doc['anchor_gate']['realized_r2'])} vs expected "
        f"{_fmt(doc['anchor_gate']['expected_r2'])} "
        f"(|Δ| = {doc['anchor_gate']['abs_deviation']:.2e}, tol {doc['anchor_gate']['tol']}).",
    ]
    if "h1_paired_shared_rows" in doc:
        h1 = doc["h1_paired_shared_rows"]
        lo, hi = h1["delta_ci95"]
        lines += [
            "",
            f"Paired shared-test-row comparison: R²(9B@L*) = {h1['r2_9b_lstar']:.4f}, "
            f"R²(7B@L19) = {h1['r2_7b_l19']:.4f}, Δ = {h1['delta_map']:.4f} "
            f"(95% CI [{lo:.4f}, {hi:.4f}]; verdict: {h1['verdict']}).",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "table_matched_n.md"
    json_path = out_dir / "table_matched_n.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return [md_path, json_path]


def fig_selected_lambda_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, lam = _sweep_series(sweep, lambda r: float(r["ridge"]["meta"]["selected_lambda"]))
    ax.plot(x, lam, color=paper_palette_role("primary"), marker="o", ms=3, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("selected ridge λ (validation)")
    paths = savefig_paper(fig, "fig_selected_lambda_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_floors_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    layers = sweep_layers(sweep)
    floor_names = sorted(sweep["per_layer"][str(layers[0])]["floors"])
    colors = paper_palette(len(floor_names) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, ridge = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    ax.plot(x, ridge, color=colors[0], lw=1.8, label="ridge map")
    for i, name in enumerate(floor_names):
        _, ys = _sweep_series(
            sweep,
            lambda r, n=name: (
                float(r["floors"][n]["test_r2"])
                if r["floors"][n].get("test_r2") is not None
                else np.nan
            ),
        )
        ax.plot(x, ys, color=colors[i + 1], lw=1.0, ls="--", label=DISPLAY.get(name, name))
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out test R²")
    ax.legend(fontsize=7, ncol=2)
    paths = savefig_paper(fig, "fig_floors_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_wc_transfer_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    _mark_full_attention(ax, int(sweep["n_layers"]))
    x, y_in = _sweep_series(sweep, lambda r: float(r["ridge"]["test_r2"]))
    _, y_wc = _sweep_series(sweep, lambda r: float(r["ridge"]["wc_test_1k_r2"]))
    ax.plot(x, y_in, color=paper_palette_role("primary"), lw=1.6, label="in-corpus test R²")
    ax.plot(x, y_wc, color=paper_palette_role("control"), lw=1.6, label="WildChat transfer R²")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("held-out R²")
    ax.legend(fontsize=8)
    paths = savefig_paper(fig, "fig_wc_transfer_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_knn_per_layer(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    _mark_full_attention(ax, int(sweep["n_layers"]))
    c1 = paper_palette_role("primary")
    c2 = paper_palette_role("control")
    for k, ls in ((1, "-"), (10, "--")):
        x, ye = _sweep_series(sweep, lambda r, kk=k: _knn_cell(r["knn"], "ridge", "euclidean", kk))
        _, yc = _sweep_series(sweep, lambda r, kk=k: _knn_cell(r["knn"], "ridge", "cosine", kk))
        ax.plot(x, ye, color=c1, ls=ls, lw=1.4, label=f"euclidean acc@{k}")
        ax.plot(x, yc, color=c2, ls=ls, lw=1.4, label=f"cosine acc@{k}")
    layers = sweep_layers(sweep)
    chance1 = float(
        sweep["per_layer"][str(layers[0])]["knn"]["ridge"]["euclidean"]["chance_at_k"]["1"]
    )
    ax.axhline(chance1, color=paper_palette_role("neutral"), lw=0.8, ls=":", label="chance @1")
    ax.set_xlabel(XLABEL_DEPTH)
    ax.set_ylabel("retrieval accuracy (true target in top k)")
    ax.legend(fontsize=7, ncol=2)
    paths = savefig_paper(fig, "fig_knn_per_layer", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_reliability_ceiling(inputs: dict, out_dir: Path) -> list[Path]:
    sweep = inputs["sweep"]
    rc = sweep["reliability_ceiling"]
    layers = [int(x) for x in rc["layers"]]
    vals = []
    for li in layers:
        blk = rc["by_layer"][str(li)]
        vals.append(float(blk["ceiling_var_weighted_r"]) if blk.get("available") else np.nan)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(
        [str(li) for li in layers],
        vals,
        color=paper_palette_role("primary"),
        width=0.55,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("two-draw reliability ceiling (r)")
    paths = savefig_paper(fig, "fig_reliability_ceiling", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


_PROFILE_STATS = ("direction_cos", "calibration_ratio_to_global", "obs_separation_snr")


def fig_crossmodel_axis_profile(inputs: dict, out_dir: Path) -> list[Path]:
    """Hero 1 (plan §6): per-axis cross-model profile — one row per shared
    axis, one panel per scale-free statistic, both models + the parent-map
    reference side by side."""
    cm = inputs["crossmodel"]
    stats = cm["stats"]
    # Row order: axes sorted by the 9B direction cosine (descending).
    dir_rows = {r["axis"]: r for r in stats["direction_cos"]["axes"]}

    def _order_key(a: str) -> float:
        v = dir_rows[a]["s_9b"]
        # None (JSON null) sorts last; a legitimate 0.0 keeps its rank.
        return -v if isinstance(v, (int, float)) else float("inf")

    axes_order = sorted(dir_rows, key=_order_key)
    ypos = np.arange(len(axes_order))[::-1]
    c9 = paper_palette_role("primary")
    c7 = paper_palette_role("baseline")

    fig, panels = plt.subplots(
        1,
        len(_PROFILE_STATS),
        figsize=(3.2 * len(_PROFILE_STATS), 0.42 * len(axes_order) + 1.6),
        sharey=True,
    )
    for ax, stat in zip(np.atleast_1d(panels), _PROFILE_STATS):
        rows = {r["axis"]: r for r in stats[stat]["axes"]}
        s9 = [rows[a]["s_9b"] if a in rows else np.nan for a in axes_order]
        s7 = [rows[a]["s_7b"] if a in rows else np.nan for a in axes_order]
        sp = [rows[a]["s_7b_ref_parent"] if a in rows else np.nan for a in axes_order]
        s9 = np.asarray([np.nan if v is None else v for v in s9], dtype=np.float64)
        s7 = np.asarray([np.nan if v is None else v for v in s7], dtype=np.float64)
        sp = np.asarray([np.nan if v is None else v for v in sp], dtype=np.float64)
        ax.scatter(s9, ypos, color=c9, marker="o", s=26, label=DISPLAY["arm_fresh9b"])
        ax.scatter(s7, ypos, color=c7, marker="D", s=22, label=DISPLAY["arm_7b_matched25k"])
        ax.scatter(
            sp,
            ypos,
            facecolors="none",
            edgecolors=c7,
            marker="D",
            s=34,
            label=DISPLAY["ref_7b_parent"],
        )
        if stat == "calibration_ratio_to_global":
            ax.axvline(1.0, color=paper_palette_role("neutral"), lw=0.8, ls=":")
        ax.set_title(DISPLAY[stat], fontsize=9)
    np.atleast_1d(panels)[0].set_yticks(ypos)
    np.atleast_1d(panels)[0].set_yticklabels([axis_label(a) for a in axes_order], fontsize=8)
    np.atleast_1d(panels)[0].legend(fontsize=7, loc="lower left")
    paths = savefig_paper(fig, "fig_hero_crossmodel_axis_profile", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


def fig_crossmodel_delta_forest(inputs: dict, out_dir: Path) -> list[Path]:
    """Per-axis 9B − 7B delta with carrier-paired bootstrap 95% CI whiskers."""
    cm = inputs["crossmodel"]
    stats = cm["stats"]
    fig, panels = plt.subplots(
        1, 2, figsize=(8.6, 0.42 * len(stats["direction_cos"]["axes"]) + 1.6), sharey=True
    )
    for ax, stat in zip(panels, ("direction_cos", "obs_separation_snr")):
        rows = stats[stat]["axes"]
        axes_order = [r["axis"] for r in rows]
        ypos = np.arange(len(axes_order))[::-1]
        vals = np.asarray(
            [np.nan if r["delta_9b_minus_7b"] is None else r["delta_9b_minus_7b"] for r in rows],
            dtype=np.float64,
        )
        lo = np.asarray(
            [np.nan if r["delta_ci95"][0] is None else r["delta_ci95"][0] for r in rows],
            dtype=np.float64,
        )
        hi = np.asarray(
            [np.nan if r["delta_ci95"][1] is None else r["delta_ci95"][1] for r in rows],
            dtype=np.float64,
        )
        ax.errorbar(
            vals,
            ypos,
            xerr=_err_offsets(vals, lo, hi),
            fmt="o",
            ms=4,
            color=paper_palette_role("primary"),
            ecolor=paper_palette_role("primary"),
            elinewidth=1.1,
            capsize=2.0,
        )
        ax.axvline(0.0, color=paper_palette_role("neutral"), lw=0.8, ls=":")
        ax.set_title(f"Δ {DISPLAY[stat]} (9B − 7B)", fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([axis_label(a) for a in axes_order], fontsize=8)
    paths = savefig_paper(fig, "fig_crossmodel_delta_forest", dir=out_dir)
    plt.close(fig)
    return list(paths.values())


# ── registry + CLI ─────────────────────────────────────────────────────

# fig name -> (required input keys, renderer). Mechanically enumerable:
#   uv run python -c "import sys; sys.path.insert(0,'scripts');
#                     from issue2587_figures import FIGS; print(sorted(FIGS))"
FIGS: dict[str, tuple[tuple[str, ...], object]] = {
    "hero_layer_sweep": (("sweep", "ref9b_n10k", "ref7b_n10k"), fig_hero_layer_sweep),
    "matched_n_table": (("sweep", "matched7b", "delta?"), matched_n_table),
    "selected_lambda_per_layer": (("sweep",), fig_selected_lambda_per_layer),
    "floors_per_layer": (("sweep",), fig_floors_per_layer),
    "wc_transfer_per_layer": (("sweep",), fig_wc_transfer_per_layer),
    "knn_per_layer": (("sweep",), fig_knn_per_layer),
    "reliability_ceiling": (("sweep",), fig_reliability_ceiling),
    "crossmodel_axis_profile": (("crossmodel",), fig_crossmodel_axis_profile),
    "crossmodel_delta_forest": (("crossmodel",), fig_crossmodel_delta_forest),
}

_INPUT_FLAGS = {
    "sweep": ("sweep_json", "map_layer_sweep.json (unit 4 finalize)"),
    "matched7b": ("matched7b_json", "matched7b_anchor.json (unit 4 P8)"),
    "delta": ("delta_json", "minpair_delta_2587.json (unit 5b)"),
    "crossmodel": ("crossmodel_json", "crossmodel_contrasts.json (unit 5b)"),
    "ref9b_n10k": ("ref2330_9b", "banked #2330 9B n=10k fits"),
    "ref7b_n10k": ("ref2330_7b", "banked #2330 7B n=10k fits"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--figs", default="all", help="comma list of figure names, or 'all'")
    p.add_argument("--out-dir", type=Path, default=Path("figures/issue_2587"))
    p.add_argument(
        "--sweep-json", type=Path, default=Path("eval_results/issue_2587/map_layer_sweep.json")
    )
    p.add_argument(
        "--matched7b-json", type=Path, default=Path("eval_results/issue_2587/matched7b_anchor.json")
    )
    p.add_argument(
        "--delta-json", type=Path, default=Path("eval_results/issue_2587/minpair_delta_2587.json")
    )
    p.add_argument(
        "--crossmodel-json",
        type=Path,
        default=Path("eval_results/issue_2587/crossmodel_contrasts.json"),
    )
    p.add_argument(
        "--ref2330-9b",
        type=Path,
        default=Path("eval_results/issue_2330/matched_fits_q35_n10k.json"),
    )
    p.add_argument(
        "--ref2330-7b",
        type=Path,
        default=Path("eval_results/issue_2330/matched_fits_q25_n10k.json"),
    )
    p.add_argument("--style", default="neurips", choices=("neurips", "generic", "blog", "iclr"))
    p.add_argument("--import-check", action="store_true")
    return p.parse_args(argv)


def resolve_figs(spec: str) -> list[str]:
    names = sorted(FIGS) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in FIGS]
    if unknown:
        raise SystemExit(f"unknown figure name(s): {unknown}; known: {sorted(FIGS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    names = resolve_figs(args.figs)
    needed: set[str] = set()
    for n in names:
        for key in FIGS[n][0]:
            needed.add(key.rstrip("?"))
    inputs: dict = {}
    for key in sorted(needed):
        flag, what = _INPUT_FLAGS[key]
        path = getattr(args, flag)
        optional = all(
            key not in FIGS[n][0] for n in names
        )  # key appears only as "<key>?" -> optional
        if optional and not Path(path).is_file():
            logger.info("[figs] optional input %s absent (%s) — skipping", key, path)
            continue
        inputs[key] = _load_json(path, what)
    set_paper_style(args.style)
    written: list[Path] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        req, fn = FIGS[n]
        missing = [
            k.rstrip("?") for k in req if not k.endswith("?") and k.rstrip("?") not in inputs
        ]
        if missing:
            raise FileNotFoundError(f"figure {n}: missing required input(s) {missing}")
        out = fn(inputs, args.out_dir)
        written.extend(out)
        print(f"[figs] {n}: {len(out)} file(s)", flush=True)
    print(f"[phase=done] figures_2587 complete: {len(written)} files -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
