#!/usr/bin/env python3
"""Issue #654 Step 3+4: per-layer query-displacement metrics + figures (CPU, VM).

Plan §3 steps 3-4, §5, §6. Reads the uploaded per-pair ``.pt`` banks (no GPU,
no model). Computes, per layer:
  - per-pair centered cosine (PRIMARY DV): global-mean-center each bank on its
    own per-layer mean, L2-normalize, per-pair cosine (context-end vs query-end);
  - raw (uncentered) per-pair cosine alongside (anisotropy caveat);
  - shuffled-pair derangement floor (B=1000, seed 42) — both GLOBAL and
    per-tier WITHIN-TYPE; headline = matched-minus-shuffled with a 2.5/97.5 band;
  - per-layer linear CKA(context-bank, query-bank) + a row-permuted-bank CKA floor;
  - companion same-position contrast: cosine of context-only readout vs
    context+query query-end readout per context, aggregated by tier.

Writes ``eval_results/issue_654/per_layer_displacement.json`` (headline, keyed by
context_type x layer x query_type) + per-cell breakdowns under
``eval_results/issue_654/cells/``. ``--figures`` emits the hero + exploratory
figures to ``figures/issue_654/*.png`` using the paper_plots rcParams.

Usage::

    uv run python scripts/issue654_analyze.py --banks data/issue654/dual_pos \
        --out eval_results/issue_654 --figures
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.representation_shift import linear_cka  # noqa: E402

logger = logging.getLogger("issue654_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEED = 42
B_DERANGEMENT = 1000
ANCHOR_LAYERS = [7, 14, 21, 27]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _load_banks(banks_dir: Path) -> dict:
    """Load all per-pair .pt banks + the companion context-only readouts.

    Returns a dict with stacked per-layer tensors and the per-pair metadata.
    """
    manifest = json.loads((banks_dir / "extraction_manifest.json").read_text())
    layers = manifest["layers"]
    n_layers = len(layers)

    pair_files = sorted(banks_dir.glob("pair_*.pt"))
    if not pair_files:
        raise RuntimeError(f"no pair_*.pt banks in {banks_dir}")

    ctx_end_rows: list[torch.Tensor] = []
    qry_end_rows: list[torch.Tensor] = []
    meta_rows: list[dict] = []
    companion_cache: dict[str, torch.Tensor] = {}

    for pf in pair_files:
        d = torch.load(pf, weights_only=False)
        ctx_end_rows.append(d["context_end"])  # (n_layers, hidden)
        qry_end_rows.append(d["query_end"])
        meta_rows.append(
            {
                "pair_id": d["pair_id"],
                "context_type": d["context_type"],
                "context_id": d["context_id"],
                "query_id": d["query_id"],
                "topicality": d["topicality"],
                "length": d["length"],
                "companion_context_only_file": d["companion_context_only_file"],
            }
        )
        cid = d["context_id"]
        if cid not in companion_cache:
            cpath = banks_dir / d["companion_context_only_file"]
            cd = torch.load(cpath, weights_only=False)
            companion_cache[cid] = cd["readout"]  # (n_layers, hidden)

    A_ctx = torch.stack(ctx_end_rows).to(torch.float64)  # (n_pairs, n_layers, hidden)
    A_qry = torch.stack(qry_end_rows).to(torch.float64)
    assert A_ctx.shape == A_qry.shape, (A_ctx.shape, A_qry.shape)
    assert A_ctx.shape[1] == n_layers, (A_ctx.shape, n_layers)
    logger.info("loaded %d pairs x %d layers x %d hidden", *A_ctx.shape)
    return {
        "A_ctx": A_ctx,
        "A_qry": A_qry,
        "meta": meta_rows,
        "layers": layers,
        "n_layers": n_layers,
        "companion": companion_cache,
        "manifest": manifest,
    }


def _centered_cos_per_layer(A_ctx: torch.Tensor, A_qry: torch.Tensor) -> np.ndarray:
    """Per-pair centered cosine at every layer.

    Each bank is globally mean-centered on its OWN per-layer mean, L2-normalized,
    then the per-pair dot product is the centered cosine. Returns (n_pairs, n_layers).
    """
    n_pairs, n_layers, _ = A_ctx.shape
    out = np.zeros((n_pairs, n_layers))
    for L in range(n_layers):
        ctx = A_ctx[:, L]  # (n_pairs, hidden)
        qry = A_qry[:, L]
        ctx_c = ctx - ctx.mean(dim=0, keepdim=True)
        qry_c = qry - qry.mean(dim=0, keepdim=True)
        ctx_n = torch.nn.functional.normalize(ctx_c, dim=1)
        qry_n = torch.nn.functional.normalize(qry_c, dim=1)
        out[:, L] = (ctx_n * qry_n).sum(dim=1).numpy()
    return out


def _raw_cos_per_layer(A_ctx: torch.Tensor, A_qry: torch.Tensor) -> np.ndarray:
    """Per-pair RAW (uncentered) cosine at every layer. Returns (n_pairs, n_layers)."""
    n_pairs, n_layers, _ = A_ctx.shape
    out = np.zeros((n_pairs, n_layers))
    for L in range(n_layers):
        ctx_n = torch.nn.functional.normalize(A_ctx[:, L], dim=1)
        qry_n = torch.nn.functional.normalize(A_qry[:, L], dim=1)
        out[:, L] = (ctx_n * qry_n).sum(dim=1).numpy()
    return out


def _derangement_floor(
    A_ctx: torch.Tensor,
    A_qry: torch.Tensor,
    indices: np.ndarray,
    rng: np.random.Generator,
    b: int,
) -> dict:
    """Shuffled-pair (derangement) centered-cosine floor over a set of indices.

    For each derangement pi (i != pi(i)) restricted to ``indices``, compute the
    centered cosine of ctx[i] vs qry[pi(i)] at every layer, mean over indices.
    Returns mean + 2.5/97.5 band per layer over ``b`` derangements.
    """
    n_layers = A_ctx.shape[1]
    sub_ctx = A_ctx[indices]  # (m, n_layers, hidden)
    sub_qry = A_qry[indices]
    m = len(indices)
    if m < 2:
        # No derangement possible with < 2 items.
        nan = np.full(n_layers, np.nan)
        return {"mean": nan.tolist(), "lo": nan.tolist(), "hi": nan.tolist(), "n": m}

    # Pre-center each bank per layer (centering is on the FULL sub-bank, matching
    # the matched-pair centered-cosine definition).
    ctx_n = np.zeros((m, n_layers, sub_ctx.shape[2]))
    qry_n = np.zeros((m, n_layers, sub_qry.shape[2]))
    for L in range(n_layers):
        c = sub_ctx[:, L]
        q = sub_qry[:, L]
        c = c - c.mean(dim=0, keepdim=True)
        q = q - q.mean(dim=0, keepdim=True)
        ctx_n[:, L] = torch.nn.functional.normalize(c, dim=1).numpy()
        qry_n[:, L] = torch.nn.functional.normalize(q, dim=1).numpy()

    boot = np.zeros((b, n_layers))
    for k in range(b):
        perm = _derangement(m, rng)
        # cos[i, L] = <ctx_n[i, L], qry_n[perm[i], L]>
        cos = np.einsum("ild,ild->il", ctx_n, qry_n[perm])  # (m, n_layers)
        boot[k] = cos.mean(axis=0)
    return {
        "mean": boot.mean(axis=0).tolist(),
        "lo": np.percentile(boot, 2.5, axis=0).tolist(),
        "hi": np.percentile(boot, 97.5, axis=0).tolist(),
        "n": m,
    }


def _derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    """A random derangement of range(m) (no fixed points). Rejection-sample."""
    while True:
        perm = rng.permutation(m)
        if not np.any(perm == np.arange(m)):
            return perm


def _cka_per_layer_from_banks(
    A_ctx: torch.Tensor, A_qry: torch.Tensor, indices: np.ndarray
) -> list[float]:
    sub_ctx = A_ctx[indices]
    sub_qry = A_qry[indices]
    return [linear_cka(sub_ctx[:, L], sub_qry[:, L]) for L in range(sub_ctx.shape[1])]


def _cka_shuffled_floor(
    A_ctx: torch.Tensor, A_qry: torch.Tensor, indices: np.ndarray, rng: np.random.Generator
) -> list[float]:
    """CKA(context-bank, row-permuted query-bank) — the whole-bank shuffle floor."""
    sub_ctx = A_ctx[indices]
    sub_qry = A_qry[indices]
    m = len(indices)
    if m < 2:
        return [float("nan")] * sub_ctx.shape[1]
    perm = _derangement(m, rng)
    return [linear_cka(sub_ctx[:, L], sub_qry[perm][:, L]) for L in range(sub_ctx.shape[1])]


def _companion_cosine_per_layer(
    companion: dict[str, torch.Tensor], meta: list[dict], A_qry: torch.Tensor
) -> dict:
    """Per-context cosine of (context-only readout) vs (context+query query-end).

    Both are read at the model's residual stream; the same-position construct
    compares the assistant-gen slot under [context] vs the query-end slot under
    [context + query]. Aggregated per tier (mean per layer).

    NOTE: the companion readout is the context-only assistant-gen slot; the
    context+query side here uses the query-end residual (the model's actual
    conditioning slot before generation). Reported as the parallel companion
    curve (§5) — a same-context contrast, raw cosine (no cross-pair centering,
    since it is a within-context with/without-query comparison).
    """
    n_layers = A_qry.shape[1]
    # Group pairs by context_id; average the query-end residual over that
    # context's queries, then cosine against the context-only readout per layer.
    by_ctx: dict[str, list[int]] = defaultdict(list)
    ctx_type: dict[str, str] = {}
    for i, m in enumerate(meta):
        by_ctx[m["context_id"]].append(i)
        ctx_type[m["context_id"]] = m["context_type"]

    per_tier: dict[str, list[list[float]]] = defaultdict(list)
    per_context: dict[str, list[float]] = {}
    for cid, idxs in by_ctx.items():
        readout = companion[cid].to(torch.float64)  # (n_layers, hidden)
        qry_mean = A_qry[idxs].mean(dim=0)  # (n_layers, hidden)
        cos = np.zeros(n_layers)
        for L in range(n_layers):
            a = torch.nn.functional.normalize(readout[L], dim=0)
            b = torch.nn.functional.normalize(qry_mean[L], dim=0)
            cos[L] = float((a * b).sum().item())
        per_context[cid] = cos.tolist()
        per_tier[ctx_type[cid]].append(cos.tolist())

    tier_mean = {t: np.array(rows).mean(axis=0).tolist() for t, rows in per_tier.items()}
    return {"per_context": per_context, "per_tier_mean": tier_mean}


def analyze(banks_dir: Path, out_dir: Path) -> dict:
    data = _load_banks(banks_dir)
    A_ctx, A_qry, meta = data["A_ctx"], data["A_qry"], data["meta"]
    layers, n_layers = data["layers"], data["n_layers"]
    rng = np.random.default_rng(SEED)

    centered = _centered_cos_per_layer(A_ctx, A_qry)  # (n_pairs, n_layers)
    raw = _raw_cos_per_layer(A_ctx, A_qry)

    # Index groupings.
    all_idx = np.arange(len(meta))
    by_type: dict[str, np.ndarray] = {}
    for t in sorted({m["context_type"] for m in meta}):
        by_type[t] = np.array([i for i, m in enumerate(meta) if m["context_type"] == t])
    by_query_type: dict[str, np.ndarray] = {}
    for qt in sorted({(m["topicality"], m["length"]) for m in meta}):
        key = f"{qt[0]}_{qt[1]}"
        by_query_type[key] = np.array(
            [i for i, m in enumerate(meta) if (m["topicality"], m["length"]) == qt]
        )

    # ── Global floor (all pairs) ─────────────────────────────────────────────
    global_floor = _derangement_floor(A_ctx, A_qry, all_idx, rng, B_DERANGEMENT)
    global_cka = _cka_per_layer_from_banks(A_ctx, A_qry, all_idx)
    global_cka_floor = _cka_shuffled_floor(A_ctx, A_qry, all_idx, rng)

    # ── Per-tier within-type floors + CKA ────────────────────────────────────
    per_type_floor: dict[str, dict] = {}
    per_type_cka: dict[str, list[float]] = {}
    per_type_cka_floor: dict[str, list[float]] = {}
    for t, idx in by_type.items():
        per_type_floor[t] = _derangement_floor(A_ctx, A_qry, idx, rng, B_DERANGEMENT)
        per_type_cka[t] = _cka_per_layer_from_banks(A_ctx, A_qry, idx)
        per_type_cka_floor[t] = _cka_shuffled_floor(A_ctx, A_qry, idx, rng)

    # ── Companion same-position contrast ─────────────────────────────────────
    companion = _companion_cosine_per_layer(data["companion"], meta, A_qry)

    # ── Headline structured output: context_type x layer x query_type (§6.5) ─
    headline: dict[str, dict] = {}
    for t, idx in by_type.items():
        headline[t] = {}
        floor = per_type_floor[t]
        for L in range(n_layers):
            layer_key = str(layers[L])
            per_qt: dict[str, dict] = {}
            for qt_key, qt_idx in by_query_type.items():
                tier_qt = np.intersect1d(idx, qt_idx)
                if len(tier_qt) == 0:
                    continue
                m_cos = float(centered[tier_qt, L].mean())
                per_qt[qt_key] = {
                    "matched_centered_cos_mean": m_cos,
                    "n": len(tier_qt),
                }
            headline[t][layer_key] = {
                "matched_centered_cos_mean": float(centered[idx, L].mean()),
                "raw_cos_mean": float(raw[idx, L].mean()),
                "shuffled_floor_mean": floor["mean"][L],
                "shuffled_floor_lo": floor["lo"][L],
                "shuffled_floor_hi": floor["hi"][L],
                "matched_minus_shuffled": float(centered[idx, L].mean()) - floor["mean"][L],
                "cka_matched": per_type_cka[t][L],
                "cka_shuffled_floor": per_type_cka_floor[t][L],
                "companion_cos_mean": companion["per_tier_mean"].get(t, [float("nan")] * n_layers)[
                    L
                ],
                "by_query_type": per_qt,
            }

    result = {
        "issue": 654,
        "layers": layers,
        "anchor_layers": ANCHOR_LAYERS,
        "n_pairs": len(meta),
        "seed": SEED,
        "b_derangement": B_DERANGEMENT,
        "context_types": sorted(by_type.keys()),
        "query_types": sorted(by_query_type.keys()),
        "centering": "global_mean",
        "global": {
            "matched_centered_cos_mean": centered.mean(axis=0).tolist(),
            "raw_cos_mean": raw.mean(axis=0).tolist(),
            "shuffled_floor": global_floor,
            "matched_minus_shuffled": (
                centered.mean(axis=0) - np.array(global_floor["mean"])
            ).tolist(),
            "cka_matched": global_cka,
            "cka_shuffled_floor": global_cka_floor,
        },
        "per_context_type": headline,
        "companion": companion,
        "extraction_manifest_summary": {
            "model": data["manifest"].get("model"),
            "num_hidden_layers": data["manifest"].get("num_hidden_layers"),
            "hidden_size": data["manifest"].get("hidden_size"),
            "offset_fail_fraction": data["manifest"].get("offset_fail_fraction"),
            "n_pairs_extracted": data["manifest"].get("n_pairs_extracted"),
        },
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    headline_path = out_dir / "per_layer_displacement.json"
    with open(headline_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", headline_path)

    # Per-cell breakdowns (one JSON per context_type).
    for t in by_type:
        with open(cells_dir / f"context_type_{t}.json", "w") as f:
            json.dump(headline[t], f, ensure_ascii=False, indent=2)
    logger.info("wrote %d per-cell breakdowns to %s", len(by_type), cells_dir)

    # Stash the per-pair matrices for figures (kept in-memory; returned).
    result["_arrays"] = {
        "centered": centered,
        "raw": raw,
        "by_type": {t: idx.tolist() for t, idx in by_type.items()},
        "by_query_type": {k: idx.tolist() for k, idx in by_query_type.items()},
        "meta": meta,
        "A_ctx": A_ctx,
        "A_qry": A_qry,
    }
    return result


def make_figures(result: dict, fig_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    layers = result["layers"]
    n_layers = len(layers)
    arrays = result["_arrays"]
    centered = arrays["centered"]
    raw = arrays["raw"]
    by_type = arrays["by_type"]
    types = sorted(by_type.keys())
    colors = paper_palette(min(max(len(types), 1), 8))

    # ── Hero: per-layer matched centered cosine per context type + floor band ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, t in enumerate(types):
        idx = np.array(by_type[t])
        m = centered[idx].mean(axis=0)
        floor = result["per_context_type"][t]
        lo = np.array([floor[str(layers[L])]["shuffled_floor_lo"] for L in range(n_layers)])
        hi = np.array([floor[str(layers[L])]["shuffled_floor_hi"] for L in range(n_layers)])
        ax.plot(layers, m, label=t, color=colors[ci % len(colors)], linewidth=2)
        ax.fill_between(layers, lo, hi, color=colors[ci % len(colors)], alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Matched centered cosine (context-end vs query-end)")
    ax.set_title("Per-layer query displacement by context type (shaded = shuffled-pair floor)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/hero_per_layer_displacement", dir=fig_dir)
    plt.close(fig)

    # ── Raw vs centered overlay (anisotropy caveat) ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, centered.mean(axis=0), label="centered (global_mean)", linewidth=2)
    ax.plot(layers, raw.mean(axis=0), label="raw (uncentered)", linewidth=2, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean per-pair cosine")
    ax.set_title("Raw vs centered cosine (anisotropy caveat)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/raw_vs_centered", dir=fig_dir)
    plt.close(fig)

    # ── CKA heatmap (layer x layer is overkill; plot the diagonal CKA(L,L)) ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, t in enumerate(types):
        cka = [
            result["per_context_type"][t][str(layers[L])]["cka_matched"] for L in range(n_layers)
        ]
        cka_floor = [
            result["per_context_type"][t][str(layers[L])]["cka_shuffled_floor"]
            for L in range(n_layers)
        ]
        ax.plot(layers, cka, label=f"{t}", color=colors[ci % len(colors)], linewidth=2)
        ax.plot(layers, cka_floor, color=colors[ci % len(colors)], linewidth=1, linestyle=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA (context-bank vs query-bank)")
    ax.set_title("Per-layer CKA by context type (dotted = shuffled-bank floor)")
    ax.legend(loc="best", fontsize=8)
    savefig_paper(fig, "issue_654/cka_per_layer", dir=fig_dir)
    plt.close(fig)

    # ── Per-pair scatter at the 4 anchor layers (colored by query topicality) ─
    meta = arrays["meta"]
    topic = np.array([1 if m["topicality"] == "on" else 0 for m in meta])
    A_ctx = arrays["A_ctx"]
    A_qry = arrays["A_qry"]
    fig, axes = plt.subplots(1, len(ANCHOR_LAYERS), figsize=(4 * len(ANCHOR_LAYERS), 4))
    for ax, L in zip(np.atleast_1d(axes), ANCHOR_LAYERS, strict=False):
        Li = layers.index(L) if L in layers else min(L, n_layers - 1)
        ctx_n = torch.nn.functional.normalize(
            A_ctx[:, Li] - A_ctx[:, Li].mean(dim=0, keepdim=True), dim=1
        )
        qry_n = torch.nn.functional.normalize(
            A_qry[:, Li] - A_qry[:, Li].mean(dim=0, keepdim=True), dim=1
        )
        cc = (ctx_n * qry_n).sum(dim=1).numpy()
        # x = pair index within tier ordering; y = centered cosine.
        ax.scatter(
            np.arange(len(cc))[topic == 1],
            cc[topic == 1],
            s=10,
            alpha=0.6,
            label="on-topic",
        )
        ax.scatter(
            np.arange(len(cc))[topic == 0],
            cc[topic == 0],
            s=10,
            alpha=0.6,
            label="off-topic",
        )
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("pair index")
        ax.set_ylabel("centered cos")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, "issue_654/scatter_anchors", dir=fig_dir)
    plt.close(fig)

    # ── Violin of per-pair centered cosine by topicality x length at anchors ──
    fig, axes = plt.subplots(1, len(ANCHOR_LAYERS), figsize=(4 * len(ANCHOR_LAYERS), 4))
    qt_keys = sorted({f"{m['topicality']}_{m['length']}" for m in meta})
    for ax, L in zip(np.atleast_1d(axes), ANCHOR_LAYERS, strict=False):
        Li = layers.index(L) if L in layers else min(L, n_layers - 1)
        groups = []
        for qt in qt_keys:
            sel = np.array(
                [i for i, m in enumerate(meta) if f"{m['topicality']}_{m['length']}" == qt]
            )
            groups.append(centered[sel, Li] if len(sel) else np.array([0.0]))
        ax.violinplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(qt_keys) + 1))
        ax.set_xticklabels(qt_keys, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"Layer {L}")
        ax.set_ylabel("centered cos")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/violin_query_type", dir=fig_dir)
    plt.close(fig)

    # ── Companion same-position vs two-position curve per context type ────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp = result["companion"]["per_tier_mean"]
    for ci, t in enumerate(types):
        idx = np.array(by_type[t])
        ax.plot(
            layers,
            centered[idx].mean(axis=0),
            label=f"{t} two-position",
            color=colors[ci % len(colors)],
            linewidth=2,
        )
        if t in comp:
            ax.plot(
                layers,
                comp[t],
                label=f"{t} companion",
                color=colors[ci % len(colors)],
                linewidth=1.5,
                linestyle="--",
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel("cosine")
    ax.set_title("Two-position vs companion same-position contrast")
    ax.legend(loc="best", fontsize=7)
    savefig_paper(fig, "issue_654/companion_vs_two_position", dir=fig_dir)
    plt.close(fig)

    logger.info("wrote figures to %s/issue_654/", fig_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue #654: per-layer displacement metrics + figures."
    )
    parser.add_argument("--banks", type=Path, required=True, help="dir with pair_*.pt + manifest")
    parser.add_argument("--out", type=Path, required=True, help="eval_results/issue_654 dir")
    parser.add_argument("--figures", action="store_true", help="also emit figures")
    parser.add_argument("--fig-dir", default="figures/", help="figure parent dir")
    args = parser.parse_args()

    result = analyze(args.banks, args.out)
    if args.figures:
        make_figures(result, args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
