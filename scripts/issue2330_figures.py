#!/usr/bin/env python3
"""Task #2330 P4 figures: matched-data map R² — Qwen2.5-7B vs Qwen3.5-9B.

Port of ``scripts/issue1491_scale_ladder_figures.py`` over the #2330 4-cell
grid ({7B, 9B} × {5k, 10k} matched LMSYS on-policy fits). Reads the committed
P3 cell JSONs (``matched_fits_<cell>.json``), the P4 ``contrasts.json``, and
the per-cell preds npz, and renders the plan §6 roster under
``figures/issue_2330/`` via the paper-plots conventions:

- HERO two-panel: held-out test R² per cell (grouped by model, 5k/10k) with
  each model's two-draw ceiling + shuffled-pairing-null lines; right panel the
  same ceiling-normalized.
- REQUIRED per-unit companion: per-context predicted-vs-realized cosine strip
  plot (one point per test context per cell, medians marked).
- Exploratory dump: per-layer R² (3 depth-matched layers × 4 cells,
  full-attention marker on 9B L16); floors panel (5 floors × 4 cells);
  retrieval acc@1 vs chance; R²-vs-n mini-curves (2 points per model + the 7B
  25k port-parity anchor when the fits carry it); WildChat-transfer vs
  in-distribution panel (fold-labeled); optional cap-hit fractions per
  split/model (``--cap-hit-dir`` — the pipeline's own ``cap_hit_*.json``
  aggregates in the ``issue2330_cap_hit_v2`` schema, written by
  ``issue2330_qwen35_generate_capture.py --aggregate-cap-hit``); bootstrap Δ
  distributions (per-draw Δ recomputed from the preds npz with the P4 seed +
  shared resample matrix and parity-asserted against contrasts.json);
  selected-λ table.

All loads are FAIL-LOUD (missing/empty inputs raise; #2130 read-side ceiling
n_pairs defense; committed-R² parity at 1e-6) — reused wholesale from
``issue2330_contrasts._load_cells``. Figures carry axes + ticks + legend +
panel titles ONLY (no on-canvas caption blocks; standing 2026-08-12
directive); every figure writes .png/.pdf/meta.json via ``savefig_paper``.

    uv run python scripts/issue2330_figures.py \
        --fits-dir eval_results/issue_2330 \
        --contrasts eval_results/issue_2330/contrasts.json \
        --preds-dir data/issue_2330/preds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Before any heavy import, so the shared-VM thread caps (#847) bind in-process.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

# Sibling import: the P4 contrasts module is the single source of truth for the
# cell grid, primary layers, bootstrap recipe (seed + shared resample matrix +
# counts-GEMM _r2_boot), and the fail-loud loader (#2130 ceiling defense,
# committed-R² parity, matched test-id asserts).
import issue2330_contrasts as C  # noqa: E402

REPO_ROOT = _SCRIPTS.parent

CELLS = C.CELLS
MODEL_OF = C.MODEL_OF
PRIMARY_LAYER = C.PRIMARY_LAYER

MODEL_LABEL = {"qwen25_7b": "Qwen2.5-7B", "qwen35_9b": "Qwen3.5-9B"}
N_TRAIN_OF = {"q25_n5k": 5000, "q25_n10k": 10000, "q35_n5k": 5000, "q35_n10k": 10000}
CELL_LABEL = {
    "q25_n5k": "Qwen2.5-7B 5k-fit",
    "q25_n10k": "Qwen2.5-7B 10k-fit",
    "q35_n5k": "Qwen3.5-9B 5k-fit",
    "q35_n10k": "Qwen3.5-9B 10k-fit",
}
# Stack depths for the depth-fraction x axis (plan §6: 28 vs 32 layers).
N_STACK = {"qwen25_7b": 28, "qwen35_9b": 32}
# 9B attention kinds per captured layer (plan §6: L16 full-attention, L22/L30 linear).
FULL_ATTENTION_9B_LAYERS = {16}

PAL = paper_palette_blog(6)
COL_MODEL = {"qwen25_7b": PAL[0], "qwen35_9b": PAL[1]}
COL_INDIST = PAL[2]  # eval corpus: LMSYS in-distribution test
COL_WC = PAL[3]  # eval corpus: WildChat corpus-transfer fold
GRAY = "#9a9a9a"
FLOOR_ORDER = [
    "shuffled_pairing",
    "train_mean",
    "scaled_identity",
    "identity_bias",
    "identity_copy",
]
FLOOR_LABEL = {
    "shuffled_pairing": "shuffled pairing",
    "train_mean": "train mean",
    "scaled_identity": "scaled identity",
    "identity_bias": "identity + bias",
    "identity_copy": "identity copy",
}


def _alpha_of(cell: str) -> float:
    """Shade encodes the train-set size within a model color (5k lighter)."""
    return 0.55 if N_TRAIN_OF[cell] == 5000 else 1.0


def _load_fits_json(fits_dir: Path) -> dict[str, dict]:
    """Raw cell JSONs (floors / knn / meta / anchor — fields _load_cells drops)."""
    out: dict[str, dict] = {}
    for cell in CELLS:
        path = fits_dir / f"matched_fits_{cell}.json"
        if not path.is_file():
            raise RuntimeError(f"missing P3 output {path} — run issue2330_matched_fits.py first")
        out[cell] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _load_contrasts(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing P4 output {path} — run issue2330_contrasts.py first")
    con = json.loads(path.read_text(encoding="utf-8"))
    for key in ("per_cell_boot_ci95_raw_primary", "primary_layers", "n_test", "seed", "n_boot"):
        if key not in con:
            raise RuntimeError(f"{path}: missing key {key!r} — stale/partial contrasts.json")
    return con


def _err(vals: list[float], los: list[float], his: list[float]) -> np.ndarray:
    """Non-negative errorbar offsets from CI bounds (gotchas.md xerr/yerr rule)."""
    v, lo, hi = np.asarray(vals), np.asarray(los), np.asarray(his)
    return np.vstack([np.maximum(0.0, v - lo), np.maximum(0.0, hi - v)])


# ---------------------------------------------------------------------------
# Hero: two-panel raw + ceiling-normalized R² per cell
# ---------------------------------------------------------------------------


def fig_hero(data: dict, fits: dict, con: dict, out: Path) -> None:
    """Left: raw test R² per cell + per-model ceiling and shuffled-null lines.
    Right: the same ceiling-normalized (ceiling line at 1.0)."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    xpos = {"q25_n5k": 0.0, "q25_n10k": 1.0, "q35_n5k": 2.6, "q35_n10k": 3.6}
    for ax, normalized in zip(axes, (False, True)):
        for cell in CELLS:
            model = MODEL_OF[cell]
            pl = PRIMARY_LAYER[model]
            d = data[cell]["per_layer"][pl]
            ceil = d["ceiling"]
            val = d["r2_full"]
            lo, hi = con["per_cell_boot_ci95_raw_primary"][cell]
            null = float(fits[cell]["per_layer"][str(pl)]["floors"]["shuffled_pairing"]["test_r2"])
            if normalized:
                val, lo, hi, null = val / ceil, lo / ceil, hi / ceil, null / ceil
            ax.bar(
                xpos[cell],
                val,
                width=0.82,
                color=COL_MODEL[model],
                alpha=_alpha_of(cell),
                yerr=_err([val], [lo], [hi]),
                capsize=3,
                error_kw={"lw": 1.1, "ecolor": "#333333"},
            )
            # Shuffled-pairing null: short dotted segment at each cell.
            ax.plot(
                [xpos[cell] - 0.41, xpos[cell] + 0.41],
                [null, null],
                ls=":",
                lw=1.4,
                color=GRAY,
            )
        for model, cells in (
            ("qwen25_7b", ("q25_n5k", "q25_n10k")),
            ("qwen35_9b", ("q35_n5k", "q35_n10k")),
        ):
            ceil = data[cells[0]]["per_layer"][PRIMARY_LAYER[model]]["ceiling"]
            level = 1.0 if normalized else ceil
            ax.plot(
                [xpos[cells[0]] - 0.5, xpos[cells[1]] + 0.5],
                [level, level],
                ls="--",
                lw=1.4,
                color=GRAY,
            )
        ax.set_xticks([xpos[c] for c in CELLS])
        ax.set_xticklabels(["5k", "10k", "5k", "10k"])
        ax.set_xlabel("train contexts (5k / 10k, matched LMSYS ids)")
        ax.set_title("ceiling-normalized" if normalized else "raw")
    axes[0].set_ylabel("held-out test R² (variance-weighted)")
    axes[1].set_ylabel("test R² / two-draw ceiling")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COL_MODEL["qwen25_7b"]),
        plt.Rectangle((0, 0), 1, 1, color=COL_MODEL["qwen35_9b"]),
        plt.Line2D([], [], ls="--", color=GRAY),
        plt.Line2D([], [], ls=":", color=GRAY),
    ]
    axes[0].legend(
        handles,
        [
            MODEL_LABEL["qwen25_7b"],
            MODEL_LABEL["qwen35_9b"],
            "two-draw ceiling",
            "shuffled-pairing null",
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.suptitle("Context→answer map R² at matched train data (primary layers)", y=1.06)
    savefig_paper(fig, "issue_2330/hero_r2_raw_and_normalized", dir=str(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Required per-unit companion: per-context cosine strip plot
# ---------------------------------------------------------------------------


def fig_percontext_cosine(data: dict, out: Path) -> None:
    """One point per held-out test context per cell: cosine(pred, target) at
    the primary layer; median diamonds."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    rng = np.random.default_rng(0)
    for i, cell in enumerate(CELLS):
        model = MODEL_OF[cell]
        d = data[cell]["per_layer"][PRIMARY_LAYER[model]]
        pred, y = d["pred"], d["y"]
        if pred.shape[0] == 0:
            raise RuntimeError(f"{cell}: empty preds — refusing to render")
        cos = (pred * y).sum(axis=1) / (
            np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-30
        )
        x = i + rng.uniform(-0.22, 0.22, size=cos.shape[0])
        ax.scatter(x, cos, s=4, alpha=0.10, color=COL_MODEL[model], linewidths=0)
        ax.scatter(
            [i],
            [float(np.median(cos))],
            marker="D",
            s=42,
            color="#222222",
            edgecolor="white",
            zorder=5,
        )
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABEL[c].replace(" ", "\n", 1) for c in CELLS], fontsize=8)
    ax.set_ylabel("per-context cosine(predicted, realized)")
    n_te = data[CELLS[0]]["n_test"]
    ax.set_title(f"Per-context prediction quality ({n_te} test contexts; median marked)")
    savefig_paper(fig, "issue_2330/percontext_cosine_strip", dir=str(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exploratory dump
# ---------------------------------------------------------------------------


def fig_per_layer_r2(data: dict, out: Path) -> None:
    """Test R² at the 3 depth-matched layers per cell; x = layer depth as a
    fraction of the stack; 9B full-attention block (L16) gets a star marker."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for cell in CELLS:
        model = MODEL_OF[cell]
        layers = data[cell]["layers"]
        fr = [layer / N_STACK[model] for layer in layers]
        r2 = [data[cell]["per_layer"][layer]["r2_full"] for layer in layers]
        ls = "-" if N_TRAIN_OF[cell] == 10000 else "--"
        ax.plot(fr, r2, ls=ls, lw=1.5, color=COL_MODEL[model], alpha=_alpha_of(cell))
        for f, layer, v in zip(fr, layers, r2):
            full_attn = model == "qwen35_9b" and layer in FULL_ATTENTION_9B_LAYERS
            ax.scatter(
                [f],
                [v],
                marker="*" if full_attn else "o",
                s=140 if full_attn else 26,
                color=COL_MODEL[model],
                alpha=_alpha_of(cell),
                zorder=4,
            )
    handles = [
        plt.Line2D([], [], color=COL_MODEL["qwen25_7b"], lw=1.5),
        plt.Line2D([], [], color=COL_MODEL["qwen35_9b"], lw=1.5),
        plt.Line2D([], [], color=GRAY, lw=1.5, ls="-"),
        plt.Line2D([], [], color=GRAY, lw=1.5, ls="--"),
        plt.Line2D([], [], color=COL_MODEL["qwen35_9b"], marker="*", ls="", markersize=11),
    ]
    ax.legend(
        handles,
        [
            MODEL_LABEL["qwen25_7b"],
            MODEL_LABEL["qwen35_9b"],
            "10k-fit",
            "5k-fit",
            "9B full-attention block (L16)",
        ],
        fontsize=8,
    )
    ax.set_xlabel("layer depth (fraction of stack: 7B /28, 9B /32)")
    ax.set_ylabel("held-out test R²")
    ax.set_title("Per-layer map R² at depth-matched capture layers")
    savefig_paper(fig, "issue_2330/per_layer_r2", dir=str(out))
    plt.close(fig)


def fig_floors(fits: dict, out: Path) -> None:
    """5 baseline floors × 4 cells (primary layers)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    width = 0.19
    for j, cell in enumerate(CELLS):
        model = MODEL_OF[cell]
        floors = fits[cell]["per_layer"][str(PRIMARY_LAYER[model])]["floors"]
        xs = [i + (j - 1.5) * width for i in range(len(FLOOR_ORDER))]
        ax.bar(
            xs,
            [float(floors[name]["test_r2"]) for name in FLOOR_ORDER],
            width=width * 0.92,
            color=COL_MODEL[model],
            alpha=_alpha_of(cell),
            label=CELL_LABEL[cell],
        )
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(range(len(FLOOR_ORDER)))
    ax.set_xticklabels([FLOOR_LABEL[f] for f in FLOOR_ORDER], fontsize=8)
    ax.set_ylabel("held-out test R²")
    ax.set_title("Baseline floors per cell (primary layers)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "issue_2330/floors_panel", dir=str(out))
    plt.close(fig)


def fig_retrieval(fits: dict, out: Path) -> None:
    """Ridge retrieval acc@1 (cosine) per cell vs the chance line (1/n_pool)."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    chance = None
    for i, cell in enumerate(CELLS):
        model = MODEL_OF[cell]
        knn = fits[cell]["per_layer"][str(PRIMARY_LAYER[model])]["knn_retrieval"]
        acc1 = float(knn["ridge"]["cosine"]["acc_at_k"]["1"])
        n_pool = int(knn["_meta"]["n_pool"])
        if n_pool <= 0:
            raise RuntimeError(f"{cell}: empty retrieval pool")
        cell_chance = 1.0 / n_pool
        if chance is None:
            chance = cell_chance
        elif abs(chance - cell_chance) > 1e-12:
            raise RuntimeError(f"{cell}: retrieval pool size differs across cells")
        ax.bar(i, acc1, width=0.7, color=COL_MODEL[model], alpha=_alpha_of(cell))
    ax.axhline(chance, ls=":", lw=1.4, color=GRAY)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABEL[c].replace(" ", "\n", 1) for c in CELLS], fontsize=8)
    ax.set_ylabel("retrieval acc@1, cosine (pool = test targets)")
    ax.set_title(f"kNN retrieval vs chance (dotted = 1/{int(round(1 / chance))})")
    savefig_paper(fig, "issue_2330/retrieval_acc1", dir=str(out))
    plt.close(fig)


def fig_r2_vs_n(data: dict, fits: dict, out: Path) -> None:
    """R²-vs-n mini-curves: 2 matched points per model; the 7B n=25k port-parity
    anchor rides as a third open-diamond point when the fits carry it."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for model, cells in (
        ("qwen25_7b", ("q25_n5k", "q25_n10k")),
        ("qwen35_9b", ("q35_n5k", "q35_n10k")),
    ):
        ns = [N_TRAIN_OF[c] for c in cells]
        r2 = [data[c]["per_layer"][PRIMARY_LAYER[model]]["r2_full"] for c in cells]
        ax.plot(ns, r2, marker="o", lw=1.5, color=COL_MODEL[model], label=MODEL_LABEL[model])
    anchor = fits["q25_n10k"]["port_parity_anchor"]
    if "realized_r2" in anchor:
        ax.scatter(
            [int(anchor["n_train"])],
            [float(anchor["realized_r2"])],
            marker="D",
            s=72,
            facecolor="#f2f2f2",
            edgecolor=COL_MODEL["qwen25_7b"],
            linewidths=1.6,
            zorder=5,
            label="7B 25k anchor (parent-ladder reproduction)",
        )
    elif "skipped" in anchor:
        print(
            "[figures] r2_vs_n: anchor point omitted — port_parity_anchor skipped "
            f"in this fits run ({anchor['skipped']!r})",
            flush=True,
        )
    else:
        raise RuntimeError(f"unrecognized port_parity_anchor record shape: {sorted(anchor)}")
    ax.set_xscale("log")
    ax.set_xticks([5000, 10000, 25000])
    ax.set_xticklabels(["5k", "10k", "25k"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.margins(x=0.10, y=0.12)
    ax.set_xlabel("train contexts (log scale)")
    ax.set_ylabel("held-out test R² (primary layers)")
    ax.set_title("Map R² vs train-set size")
    ax.legend(fontsize=8)
    savefig_paper(fig, "issue_2330/r2_vs_n", dir=str(out))
    plt.close(fig)


def fig_wc_transfer(data: dict, fits: dict, out: Path) -> None:
    """In-distribution LMSYS test R² vs the WildChat corpus-transfer fold R²
    per cell (primary layers), fold-labeled."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    fold_label = None
    for i, cell in enumerate(CELLS):
        model = MODEL_OF[cell]
        d = data[cell]["per_layer"][PRIMARY_LAYER[model]]
        wc = fits[cell]["per_layer"][str(PRIMARY_LAYER[model])]["wc_transfer"]
        if not wc.get("available") or d["wc_point_r2"] is None:
            raise RuntimeError(f"{cell}: wc_transfer fold unavailable — refusing to render")
        fold_label = wc["fold_label"]
        ax.bar(i - 0.19, d["r2_full"], width=0.36, color=COL_INDIST)
        ax.bar(i + 0.19, d["wc_point_r2"], width=0.36, color=COL_WC)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([CELL_LABEL[c].replace(" ", "\n", 1) for c in CELLS], fontsize=8)
    ax.set_ylabel("held-out R² (primary layers)")
    ax.legend(
        [
            plt.Rectangle((0, 0), 1, 1, color=COL_INDIST),
            plt.Rectangle((0, 0), 1, 1, color=COL_WC),
        ],
        ["LMSYS in-distribution test", "WildChat transfer fold (never seen in fitting)"],
        fontsize=8,
    )
    assert fold_label is not None  # fold availability was validated per cell above
    ax.set_title("Corpus transfer: in-distribution vs WildChat fold")
    savefig_paper(fig, "issue_2330/wc_transfer_vs_indist", dir=str(out))
    plt.close(fig)


CAP_HIT_SCHEMA = "issue2330_cap_hit_v2"
# Aggregate `root` → model key. Mirrors issue2330_matched_fits.MODELS[*]["hf_prefix"]
# (kept as literals — this module must not import the torch-heavy fits driver).
CAP_HIT_ROOT_TO_MODEL = {
    "issue1491_scale_ladder/scale7_refit": "qwen25_7b",
    "issue2330_matched/qwen35_9b": "qwen35_9b",
    # fu1 cap2048 regeneration stores (same schema, gen_max_tokens=2048).
    "issue2330_matched/q25_cap2048": "qwen25_7b",
    "issue2330_matched/qwen35_9b_cap2048": "qwen35_9b",
}
# FIX 2c (round 3) required roster: the (model, logical-split) aggregates the
# P3 truncation-restriction control CONSUMES — all six must be present in
# --cap-hit-dir mode (a missing pair raises; EXTRA aggregates — ceiling draws,
# wc_test_1k — are plotted but never required). The no---cap-hit-dir
# default-skip branch is unchanged.
CAP_HIT_REQUIRED_ROSTER = {
    (m, s) for m in ("qwen25_7b", "qwen35_9b") for s in ("train_10k", "val_400", "test_1000")
}


def fig_cap_hit(cap_hit_dir: Path, out: Path) -> None:
    """Cap-hit fractions per (model, split) from the PIPELINE's own aggregates:
    ``cap_hit_*.json`` files (schema ``issue2330_cap_hit_v2``) written by
    ``issue2330_qwen35_generate_capture.py --aggregate-cap-hit``. Fail-loud on
    an empty dir, a foreign schema, an unknown root, total<=0, or a duplicate
    (model, split). The dotted 2% line is a REFERENCE only — the registered
    #2330 disposition is the P3 truncation-restriction control, never a re-gen
    trigger (plan §8/§11; round-2 relabel)."""
    files = sorted(Path(cap_hit_dir).glob("cap_hit_*.json"))
    if not files:
        raise RuntimeError(f"{cap_hit_dir}: no cap_hit_*.json aggregates found")
    rows: dict[tuple[str, str], float] = {}
    for p in files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != CAP_HIT_SCHEMA:
            raise RuntimeError(
                f"{p}: schema {payload.get('schema')!r} != {CAP_HIT_SCHEMA!r} — not a "
                "pipeline cap-hit aggregate (issue2330_qwen35_generate_capture.py "
                "--aggregate-cap-hit)"
            )
        root, split = str(payload["root"]), str(payload["split"])
        if root not in CAP_HIT_ROOT_TO_MODEL:
            raise RuntimeError(
                f"{p}: unknown root {root!r} (known: {sorted(CAP_HIT_ROOT_TO_MODEL)})"
            )
        model_key = CAP_HIT_ROOT_TO_MODEL[root]
        total, cap = int(payload["total"]), int(payload["cap_hit"])
        if total <= 0:
            raise RuntimeError(f"{p}: total<=0")
        if not 0 <= cap <= total:
            raise RuntimeError(
                f"{p}: cap_hit={cap} outside [0, total={total}] — inconsistent aggregate"
            )
        key = (model_key, split)
        if key in rows:
            raise RuntimeError(f"{p}: duplicate cap-hit aggregate for {key}")
        rows[key] = 100.0 * cap / total
    missing_pairs = sorted(CAP_HIT_REQUIRED_ROSTER - set(rows))
    if missing_pairs:
        raise RuntimeError(
            f"{cap_hit_dir}: cap-hit roster incomplete — missing (model, split) pairs "
            f"{missing_pairs} (run issue2330_qwen35_generate_capture.py --aggregate-cap-hit "
            "per missing pair; omit --cap-hit-dir entirely for the explicit-skip branch)"
        )
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    keys = sorted(rows)
    split_label = {
        "test_1000": "test prompts\n(1,000)",
        "train_10k": "training prompts\n(10,000)",
        "val_400": "validation prompts\n(400)",
    }
    labels = [f"{MODEL_LABEL[m]}\n{split_label.get(s, s)}" for m, s in keys]
    fracs = [rows[k] for k in keys]
    colors = [COL_MODEL.get(m, GRAY) for m, _ in keys]
    ax.bar(range(len(labels)), fracs, color=colors, width=0.7)
    ax.axhline(
        2.0,
        ls=":",
        lw=1.4,
        color=GRAY,
        label="2% reference (not a trigger — restriction control registered instead)",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("generations ending at the 1,024-token cap (%)")
    ax.set_title("Generation cap-hit per (model, split)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "issue_2330/cap_hit_fractions", dir=str(out))
    plt.close(fig)


def fig_boot_delta(data: dict, con: dict, out: Path) -> None:
    """Bootstrap Δ distributions for the 4 registered primary-layer contrasts,
    recomputed from the preds npz with the P4 seed + ONE shared resample
    matrix, and parity-asserted against contrasts.json ci95."""
    n = data[CELLS[0]]["n_test"]
    if int(con["n_test"]) != n:
        raise RuntimeError(f"contrasts n_test={con['n_test']} != preds n_test={n}")
    rng = np.random.default_rng(C.SEED)
    idx = rng.integers(0, n, size=(C.N_BOOT, n))
    boots = {
        cell: C._r2_boot(
            data[cell]["per_layer"][PRIMARY_LAYER[MODEL_OF[cell]]]["pred"],
            data[cell]["per_layer"][PRIMARY_LAYER[MODEL_OF[cell]]]["y"],
            idx,
        )
        for cell in CELLS
    }
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, rec in zip(axes.ravel(), con["primary_layers"]["raw"]):
        a, b = rec["pair"]
        d_b = boots[b] - boots[a]
        lo, hi = np.percentile(d_b, [2.5, 97.5])
        if not (
            np.isclose(lo, rec["ci95"][0], atol=1e-8) and np.isclose(hi, rec["ci95"][1], atol=1e-8)
        ):
            raise RuntimeError(
                f"boot Δ parity FAIL for {rec['label']!r}: recomputed ci95 "
                f"[{lo}, {hi}] vs contrasts.json {rec['ci95']} — seed/matrix drift"
            )
        ax.hist(d_b, bins=40, color=PAL[4], alpha=0.85)
        ax.axvline(0.0, color="#444444", lw=1.0)
        ax.axvline(lo, ls=":", lw=1.2, color=GRAY)
        ax.axvline(hi, ls=":", lw=1.2, color=GRAY)
        ax.set_title(rec["label"], fontsize=9)
        ax.set_xlabel("Δ test R² (b − a)")
        ax.set_ylabel("bootstrap draws")
    # Constrained layout ignores subplots_adjust, and a manually-positioned
    # suptitle (explicit y) is excluded from its layout pass — which is what
    # overlapped the upper subplot titles. Let the engine place it.
    fig.suptitle(
        f"Paired-bootstrap Δ distributions ({con['n_boot']} draws, shared resample matrix; "
        "dotted = 95% CI)"
    )
    savefig_paper(fig, "issue_2330/boot_delta_distributions", dir=str(out))
    plt.close(fig)


def fig_lambda_table(fits: dict, out: Path) -> None:
    """Selected ridge λ per (cell × layer), with the λ-grid edge disposition."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.axis("off")
    col_layers = {cell: fits[cell]["layers"] for cell in CELLS}
    n_cols = len(next(iter(col_layers.values())))
    rows = []
    row_labels = []
    for cell in CELLS:
        row = []
        for layer in col_layers[cell]:
            meta = fits[cell]["per_layer"][str(layer)]["ridge"]["meta"]
            lam = float(meta["selected_lambda"])
            edge = meta.get("lambda_grid_edge")
            row.append(f"{lam:.3g}" + (f" ({edge} edge)" if edge else ""))
        rows.append(row)
        row_labels.append(CELL_LABEL[cell])
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[f"layer {i + 1} (shallow→deep)" for i in range(n_cols)],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("Selected ridge λ per cell × layer (val-selected, edge-extended grid)")
    savefig_paper(fig, "issue_2330/selected_lambda_table", dir=str(out))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fu1 figures: dense 9B layer profile + cap-1024 vs cap-2048 comparison
# ---------------------------------------------------------------------------


def _cap_hit_fracs(cap_hit_dir: Path) -> dict[tuple[str, str], float]:
    """(model, split) → cap-hit % from a dir of issue2330_cap_hit_v2 aggregates
    (same schema/root/consistency gates as fig_cap_hit, roster check relaxed to
    the six (model, split) pairs actually present in the dir)."""
    files = sorted(Path(cap_hit_dir).glob("cap_hit_*.json"))
    if not files:
        raise RuntimeError(f"{cap_hit_dir}: no cap_hit_*.json aggregates found")
    rows: dict[tuple[str, str], float] = {}
    for p in files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != CAP_HIT_SCHEMA:
            raise RuntimeError(f"{p}: schema {payload.get('schema')!r} != {CAP_HIT_SCHEMA!r}")
        root, split = str(payload["root"]), str(payload["split"])
        if root not in CAP_HIT_ROOT_TO_MODEL:
            raise RuntimeError(f"{p}: unknown root {root!r}")
        total, cap = int(payload["total"]), int(payload["cap_hit"])
        if total <= 0 or not 0 <= cap <= total:
            raise RuntimeError(f"{p}: inconsistent aggregate total={total} cap_hit={cap}")
        key = (CAP_HIT_ROOT_TO_MODEL[root], split)
        if key in rows:
            raise RuntimeError(f"{p}: duplicate cap-hit aggregate for {key}")
        rows[key] = 100.0 * cap / total
    return rows


def fig_dense_profile(dense_path: Path, fits_dir: Path, out: Path) -> None:
    """9B held-out test R² at every layer output 0-30 (dense sweep, banked
    generations), with the depth-matched pick (L22) and the realized peak (L18)
    marked and the 7B layer-19 10k fit as a reference level."""
    dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))
    fits7 = json.loads((fits_dir / "matched_fits_q25_n10k.json").read_text(encoding="utf-8"))
    ref7 = float(fits7["per_layer"]["19"]["ridge"]["test_r2"])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for cell, ls, alpha in (("q35_n10k", "-", 1.0), ("q35_n5k", "--", 0.55)):
        pl = dense["cells"][cell]["per_layer"]
        layers = sorted(int(k) for k in pl)
        r2 = [float(pl[str(k)]["test_r2"]) for k in layers]
        ax.plot(
            layers,
            r2,
            ls=ls,
            lw=1.5,
            marker="o",
            markersize=3.2,
            color=COL_MODEL["qwen35_9b"],
            alpha=alpha,
            label=f"Qwen3.5-9B {N_TRAIN_OF[cell] // 1000}k-fit",
        )
    ax.axhline(
        ref7,
        ls="--",
        lw=1.3,
        color=COL_MODEL["qwen25_7b"],
        label="Qwen2.5-7B best of 3 captured layers (L19, 10k)",
    )
    ax.axvline(18, ls="--", lw=1.1, color=GRAY, label="layer 18 (dense-sweep peak)")
    ax.axvline(22, ls=":", lw=1.1, color=GRAY, label="layer 22 (depth-matched pick)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("Qwen3.5-9B layer index (block output, 0-30 of 32)")
    ax.set_ylabel("held-out test R²")
    ax.set_title("Dense per-layer map R², Qwen3.5-9B")
    savefig_paper(fig, "issue_2330/dense_layer_profile", dir=str(out))
    plt.close(fig)


def fig_cap2048_comparison(
    fits_dir: Path,
    cap2048_dir: Path,
    cap_hit_dir: Path,
    cap_hit_cap2048_dir: Path,
    out: Path,
) -> None:
    """Two panels: primary-layer test R² per cell at the 1,024 vs 2,048 caps
    (left) and cap-hit % per (model, split) at both caps (right)."""
    r2 = {}
    for cell in CELLS:
        prim = str(PRIMARY_LAYER[MODEL_OF[cell]])
        orig = json.loads((fits_dir / f"matched_fits_{cell}.json").read_text(encoding="utf-8"))
        cap = json.loads(
            (cap2048_dir / f"matched_fits_{cell}_cap2048.json").read_text(encoding="utf-8")
        )
        r2[cell] = (
            float(orig["per_layer"][prim]["ridge"]["test_r2"]),
            float(cap["per_layer"][prim]["ridge"]["test_r2"]),
        )

    fr_orig = _cap_hit_fracs(cap_hit_dir)
    fr_cap = _cap_hit_fracs(cap_hit_cap2048_dir)

    set_paper_style("blog")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # Left: per-cell primary-layer R² at both caps (open = 1,024, filled = 2,048).
    for i, cell in enumerate(CELLS):
        col = COL_MODEL[MODEL_OF[cell]]
        v1024, v2048 = r2[cell]
        ax1.plot([i, i], [v1024, v2048], color=col, lw=1.0, alpha=0.6, zorder=2)
        ax1.scatter([i], [v1024], facecolors="none", edgecolors=col, s=52, linewidths=1.4, zorder=3)
        ax1.scatter([i], [v2048], color=col, s=52, zorder=3)
    ax1.set_xticks(range(len(CELLS)))
    ax1.set_xticklabels([CELL_LABEL[c].replace(" ", "\n", 1) for c in CELLS], fontsize=8)
    ax1.set_ylabel("held-out test R² (primary layer)")
    ax1.set_title("Map R² at the 1,024 vs 2,048 caps")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor=GRAY,
            markeredgewidth=1.4,
            markersize=8,
        ),
        plt.Line2D([], [], marker="o", ls="", color=GRAY, markersize=8),
    ]
    ax1.legend(handles, ["1,024-token cap", "2,048-token cap"], fontsize=8, loc="lower right")

    # Right: cap-hit % per (model, split) at both caps.
    keys = sorted(fr_orig)
    if sorted(fr_cap) != keys:
        raise RuntimeError("cap-hit (model, split) rosters differ across caps")
    split_label = {"test_1000": "test", "train_10k": "train", "val_400": "val"}
    labels = [f"{MODEL_LABEL[m]}\n{split_label.get(s, s)}" for m, s in keys]
    x = np.arange(len(keys))
    w = 0.38
    ax2.bar(
        x - w / 2,
        [fr_orig[k] for k in keys],
        width=w,
        color=[COL_MODEL[m] for m, _ in keys],
        alpha=0.45,
    )
    ax2.bar(x + w / 2, [fr_cap[k] for k in keys], width=w, color=[COL_MODEL[m] for m, _ in keys])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("generations ending at the cap (%)")
    ax2.set_title("Cap-hit per (model, split)")
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.45),
        plt.Rectangle((0, 0), 1, 1, color=GRAY),
    ]
    ax2.legend(bar_handles, ["1,024-token cap", "2,048-token cap"], fontsize=8)

    savefig_paper(fig, "issue_2330/cap2048_comparison", dir=str(out))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Task #2330 P4 figures (plan §6 roster)")
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330",
        help="dir holding matched_fits_<cell>.json (P3 outputs)",
    )
    ap.add_argument(
        "--contrasts",
        type=Path,
        default=None,
        help="P4 contrasts.json (default: <fits-dir>/contrasts.json)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_2330" / "preds",
        help="dir holding <cell>_test_preds_ridge.npz (P3 outputs)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "figures",
        help="figures root (writes under <out>/issue_2330/)",
    )
    ap.add_argument(
        "--cap-hit-dir",
        type=Path,
        default=None,
        help="dir of cap_hit_*.json aggregates (issue2330_cap_hit_v2 schema, written by "
        "issue2330_qwen35_generate_capture.py --aggregate-cap-hit); the cap-hit panel is "
        "SKIPPED (with an explicit log line, never zero bars) when absent",
    )
    ap.add_argument(
        "--fu1",
        action="store_true",
        help="render ONLY the fu1 figures (dense 9B layer profile + cap-1024 vs "
        "cap-2048 comparison); skips the preds/contrasts loads",
    )
    ap.add_argument(
        "--dense-json",
        type=Path,
        default=REPO_ROOT
        / "eval_results"
        / "issue_2330"
        / "dense_sweep"
        / "matched_fits_q35_dense.json",
    )
    ap.add_argument(
        "--cap2048-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330" / "cap2048",
    )
    ap.add_argument(
        "--cap-hit-cap2048-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_2330" / "cap_hit_cap2048",
    )
    args = ap.parse_args()
    contrasts_path = args.contrasts if args.contrasts else args.fits_dir / "contrasts.json"

    if args.fu1:
        cap_hit_dir = args.cap_hit_dir or (REPO_ROOT / "eval_results" / "issue_2330" / "cap_hit")
        fig_dense_profile(args.dense_json, args.fits_dir, args.out)
        fig_cap2048_comparison(
            args.fits_dir, args.cap2048_dir, cap_hit_dir, args.cap_hit_cap2048_dir, args.out
        )
        print(f"[figures] wrote fu1 figures under {args.out / 'issue_2330'}", flush=True)
        return 0

    # Fail-loud loads: _load_cells carries the matched-id, committed-R² parity
    # (1e-6), and #2130 ceiling n_pairs defenses; raises on any miss.
    data = C._load_cells(args.fits_dir, args.preds_dir)
    fits = _load_fits_json(args.fits_dir)
    con = _load_contrasts(contrasts_path)

    fig_hero(data, fits, con, args.out)
    fig_percontext_cosine(data, args.out)
    fig_per_layer_r2(data, args.out)
    fig_floors(fits, args.out)
    fig_retrieval(fits, args.out)
    fig_r2_vs_n(data, fits, args.out)
    fig_wc_transfer(data, fits, args.out)
    if args.cap_hit_dir is not None:
        fig_cap_hit(args.cap_hit_dir, args.out)
    else:
        print(
            "[figures] cap-hit panel SKIPPED — no --cap-hit-dir provided (run "
            "issue2330_qwen35_generate_capture.py --aggregate-cap-hit per (root, split) "
            "to produce cap_hit_*.json); pass it to render cap_hit_fractions",
            flush=True,
        )
    fig_boot_delta(data, con, args.out)
    fig_lambda_table(fits, args.out)
    print(f"[figures] wrote plan-§6 roster under {args.out / 'issue_2330'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
