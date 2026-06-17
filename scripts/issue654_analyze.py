#!/usr/bin/env python3
"""Issue #654 Step 3+4: per-layer query-displacement metrics + figures (CPU, VM).

Plan §3 steps 3-4, §5, §6. Reads the uploaded per-pair ``.pt`` banks (no GPU,
no model). Computes, per layer:
  - per-pair centered cosine (PRIMARY DV): GLOBAL-mean-center each bank ONCE on
    its own per-layer mean over ALL pairs, L2-normalize, per-pair cosine
    (context-end vs query-end);
  - raw (uncentered) per-pair cosine alongside (anisotropy caveat);
  - shuffled-pair derangement floor (B=1000, seed 42) — both GLOBAL and
    per-tier WITHIN-TYPE; headline = matched-minus-shuffled with a 2.5/97.5 band.
    BOTH the per-tier matched cosine AND the per-tier shuffled floor consume the
    SAME globally-centered+normalized banks (centered ONCE across the full bank,
    never re-centered per tier) — matched and floor therefore subtract identical
    pre-centered tensors (plan §5 `global_mean`; concern
    per-tier-floor-centering-mismatch);
  - per-layer linear CKA(context-bank, query-bank) + a row-permuted-bank CKA floor;
  - companion SAME-POSITION contrast (plan §5): cosine of (context-only
    assistant-gen readout) vs (full-prompt assistant-gen readout) — BOTH at the
    FIXED assistant-generation slot, removing the different-token confound. The
    full-prompt readout is the per-pair ``readout`` bank captured in the SAME
    forward as context-end/query-end (concern companion-read-not-same-slot); the
    old query-end-slot companion read is gone.

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
    full_readout_rows: list[torch.Tensor] = []
    meta_rows: list[dict] = []
    companion_cache: dict[str, torch.Tensor] = {}

    for pf in pair_files:
        # weights_only=True: the saved dict holds only tensors + str/int/list
        # (no custom classes), so the safe loader path is sufficient.
        d = torch.load(pf, weights_only=True)
        ctx_end_rows.append(d["context_end"])  # (n_layers, hidden)
        qry_end_rows.append(d["query_end"])
        # Companion same-slot read (plan §5): the full-prompt's assistant-gen slot,
        # captured in the SAME forward as context-end/query-end. Required so the
        # companion contrast reads the SAME position as the context-only readout
        # (concern companion-read-not-same-slot) — never A_qry (query-end slot).
        if "readout" not in d:
            raise RuntimeError(
                f"{pf.name}: missing per-pair 'readout' bank — re-run extraction with "
                f"readout_position=-1 (companion same-slot contrast, plan §5)."
            )
        full_readout_rows.append(d["readout"])
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
            cd = torch.load(cpath, weights_only=True)  # tensors + str/int/list only
            companion_cache[cid] = cd["readout"]  # (n_layers, hidden)

    A_ctx = torch.stack(ctx_end_rows).to(torch.float64)  # (n_pairs, n_layers, hidden)
    A_qry = torch.stack(qry_end_rows).to(torch.float64)
    A_readout = torch.stack(full_readout_rows).to(torch.float64)  # full-prompt assistant-gen slot
    assert A_ctx.shape == A_qry.shape, (A_ctx.shape, A_qry.shape)
    assert A_ctx.shape == A_readout.shape, (A_ctx.shape, A_readout.shape)
    assert A_ctx.shape[1] == n_layers, (A_ctx.shape, n_layers)
    logger.info("loaded %d pairs x %d layers x %d hidden", *A_ctx.shape)
    return {
        "A_ctx": A_ctx,
        "A_qry": A_qry,
        "A_readout": A_readout,
        "meta": meta_rows,
        "layers": layers,
        "n_layers": n_layers,
        "companion": companion_cache,
        "manifest": manifest,
    }


def _global_center_normalize(A: torch.Tensor) -> torch.Tensor:
    """Globally mean-center a bank ONCE (per-layer, over ALL pairs) then L2-normalize.

    The subtracted mean is the per-layer centroid over the FULL bank — never
    re-computed per tier. Returns ``(n_pairs, n_layers, hidden)`` of unit-norm
    rows. This is THE pre-centered bank that both the matched per-pair cosine and
    the shuffled-pair floor consume (plan §5 `global_mean`; concern
    per-tier-floor-centering-mismatch).
    """
    centered = A - A.mean(dim=0, keepdim=True)  # global per-layer mean, all pairs
    return torch.nn.functional.normalize(centered, dim=2)


def _centered_cos_per_layer(ctx_hat: torch.Tensor, qry_hat: torch.Tensor) -> np.ndarray:
    """Per-pair centered cosine at every layer from PRE-centered+normalized banks.

    ``ctx_hat`` / ``qry_hat`` are the globally-centered, L2-normalized banks from
    :func:`_global_center_normalize`. The per-pair dot product per layer is the
    centered cosine. Returns (n_pairs, n_layers).
    """
    # ctx_hat, qry_hat: (n_pairs, n_layers, hidden) already centered+normalized.
    return (ctx_hat * qry_hat).sum(dim=2).numpy()


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
    ctx_hat: torch.Tensor,
    qry_hat: torch.Tensor,
    indices: np.ndarray,
    rng: np.random.Generator,
    b: int,
) -> dict:
    """Shuffled-pair (derangement) centered-cosine floor over a set of indices.

    ``ctx_hat`` / ``qry_hat`` are the GLOBALLY-centered, L2-normalized banks
    (:func:`_global_center_normalize`) — the IDENTICAL tensors the matched
    per-pair cosine consumes. This helper does NOT re-center; it only restricts
    the derangement to ``indices`` (a within-tier subset for per-tier floors, or
    all indices for the global floor). Centering matched and floor on the same
    global bank is what makes ``matched_minus_shuffled`` subtract identical
    pre-centered tensors (concern per-tier-floor-centering-mismatch).

    For each derangement pi (i != pi(i)) over the row positions in ``indices``,
    compute the cosine of ctx_hat[indices[i]] vs qry_hat[indices[pi(i)]] at every
    layer, mean over the subset. Returns mean + 2.5/97.5 band per layer over ``b``
    derangements.
    """
    n_layers = ctx_hat.shape[1]
    sub_ctx = ctx_hat[indices].numpy()  # (m, n_layers, hidden) — already centered+normalized
    sub_qry = qry_hat[indices].numpy()
    m = len(indices)
    if m < 2:
        # No derangement possible with < 2 items.
        nan = np.full(n_layers, np.nan)
        return {"mean": nan.tolist(), "lo": nan.tolist(), "hi": nan.tolist(), "n": m}

    boot = np.zeros((b, n_layers))
    for k in range(b):
        perm = _derangement(m, rng)
        # cos[i, L] = <ctx_hat[i, L], qry_hat[perm[i], L]>  (within-subset shuffle)
        cos = np.einsum("ild,ild->il", sub_ctx, sub_qry[perm])  # (m, n_layers)
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
    companion: dict[str, torch.Tensor], meta: list[dict], A_readout: torch.Tensor
) -> dict:
    """SAME-POSITION companion contrast (plan §5): context-only vs full-prompt,
    BOTH read at the assistant-generation slot.

    For each pair *i*, cosine of
      - ``companion[context_id]`` = the context-only prompt's assistant-gen slot
        readout (no query), against
      - ``A_readout[i]`` = the SAME pair's FULL-prompt assistant-gen slot readout
        (context + query), captured in the same forward as context-end/query-end.

    Both vectors are read at the FIXED assistant-generation position, so the only
    difference between them is the PRESENCE OF THE QUERY — the different-token
    confound of the old query-end-slot companion read is removed (concern
    companion-read-not-same-slot). Raw cosine (a within-context with/without-query
    comparison; no cross-pair centering). Aggregated per tier (mean per layer over
    that tier's pairs) AND per context (mean per layer over that context's pairs).
    """
    n_layers = A_readout.shape[1]
    per_pair_cos: dict[str, np.ndarray] = {}
    by_ctx_cos: dict[str, list[np.ndarray]] = defaultdict(list)
    by_tier_cos: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, m in enumerate(meta):
        cid = m["context_id"]
        ctx_only = companion[cid].to(torch.float64)  # (n_layers, hidden), assistant-gen slot
        full = A_readout[i]  # (n_layers, hidden), full-prompt assistant-gen slot
        a = torch.nn.functional.normalize(ctx_only, dim=1)
        b = torch.nn.functional.normalize(full, dim=1)
        cos = (a * b).sum(dim=1).numpy()  # (n_layers,)
        per_pair_cos[m["pair_id"]] = cos
        by_ctx_cos[cid].append(cos)
        by_tier_cos[m["context_type"]].append(cos)

    assert per_pair_cos, "no pairs for companion same-slot contrast"
    per_context = {cid: np.stack(rows).mean(axis=0).tolist() for cid, rows in by_ctx_cos.items()}
    tier_mean = {t: np.stack(rows).mean(axis=0).tolist() for t, rows in by_tier_cos.items()}
    _ = n_layers  # documented for shape clarity
    return {"per_context": per_context, "per_tier_mean": tier_mean}


def analyze(banks_dir: Path, out_dir: Path) -> dict:
    data = _load_banks(banks_dir)
    A_ctx, A_qry, meta = data["A_ctx"], data["A_qry"], data["meta"]
    A_readout = data["A_readout"]
    layers, n_layers = data["layers"], data["n_layers"]
    rng = np.random.default_rng(SEED)

    # Globally center+normalize EACH bank ONCE (per-layer, over ALL pairs). BOTH
    # the matched per-pair cosine and the shuffled-pair floor consume these exact
    # tensors — never a per-tier re-centering (concern
    # per-tier-floor-centering-mismatch).
    ctx_hat = _global_center_normalize(A_ctx)  # (n_pairs, n_layers, hidden)
    qry_hat = _global_center_normalize(A_qry)
    centered = _centered_cos_per_layer(ctx_hat, qry_hat)  # (n_pairs, n_layers)
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

    # ── Global floor (all pairs) — same globally-centered banks as matched ───
    global_floor = _derangement_floor(ctx_hat, qry_hat, all_idx, rng, B_DERANGEMENT)
    global_cka = _cka_per_layer_from_banks(A_ctx, A_qry, all_idx)
    global_cka_floor = _cka_shuffled_floor(A_ctx, A_qry, all_idx, rng)

    # ── Per-tier within-type floors + CKA ────────────────────────────────────
    # Per-tier floors restrict the derangement to within-tier rows of the SAME
    # globally-centered banks — never re-centered per tier.
    per_type_floor: dict[str, dict] = {}
    per_type_cka: dict[str, list[float]] = {}
    per_type_cka_floor: dict[str, list[float]] = {}
    for t, idx in by_type.items():
        per_type_floor[t] = _derangement_floor(ctx_hat, qry_hat, idx, rng, B_DERANGEMENT)
        per_type_cka[t] = _cka_per_layer_from_banks(A_ctx, A_qry, idx)
        per_type_cka_floor[t] = _cka_shuffled_floor(A_ctx, A_qry, idx, rng)

    # ── Companion same-position contrast (context-only vs full-prompt, same slot) ─
    companion = _companion_cosine_per_layer(data["companion"], meta, A_readout)

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
    # ctx_hat/qry_hat are the globally-centered+normalized banks (same tensors the
    # matched cosine + floor consume); the anchor scatter reads centered cosine
    # straight off them rather than re-centering per layer.
    result["_arrays"] = {
        "centered": centered,
        "raw": raw,
        "by_type": {t: idx.tolist() for t, idx in by_type.items()},
        "by_query_type": {k: idx.tolist() for k, idx in by_query_type.items()},
        "meta": meta,
        "ctx_hat": ctx_hat,
        "qry_hat": qry_hat,
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
    ctx_hat = arrays["ctx_hat"]  # globally centered+normalized (same as matched/floor)
    qry_hat = arrays["qry_hat"]
    fig, axes = plt.subplots(1, len(ANCHOR_LAYERS), figsize=(4 * len(ANCHOR_LAYERS), 4))
    for ax, L in zip(np.atleast_1d(axes), ANCHOR_LAYERS, strict=False):
        Li = layers.index(L) if L in layers else min(L, n_layers - 1)
        cc = (ctx_hat[:, Li] * qry_hat[:, Li]).sum(dim=1).numpy()
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
