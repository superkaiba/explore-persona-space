#!/usr/bin/env python3
"""Issue #667 EXPLORATORY probe — context-vector shift Δc decomposition.

Input-side twin of A3.8 (which found the answer-side WRITE ŵ is ~rank-one). Here
we decompose the CONTEXT-vector shift Δc = c_C⁺ − c_C at layer 14, where:

  c_C  = base last-input-token residual under context C (``c_C`` in the store).
  c_C⁺ = same under the source-behavior adapter (``c_C_postft`` in the store).

The store (#667 gate-chain preview, HF ``issue667_gate_chain_preview/
analysis_tensors``) carries per (behavior, source, target, L14) npz with BOTH
the base + post-FT last-input-token context vectors — for the SOURCE context
(``c_C`` / ``c_C_postft``, identical across all targets of a source-cell) AND the
TARGET context (``c_Cp`` / ``c_Cp_postft``). So Δc is computed two ways:

  * SOURCE grain: Δc_src = c_C_postft − c_C, ONE per (behavior, source-cell).
    16 sources per behavior. This is the object the prompt names Δc.
  * TARGET grain (mirrors A3.8's per-source-over-targets SVD): Δq = c_Cp_postft
    − c_Cp, N_targets=30 per source-cell — the query-side shift the SAME adapter
    induces on OTHER contexts. Stacked across targets within each source cell.

Write ŵ = v_plus − v0 at the source diagonal (response-span mean, layer 14).
r_B: #658 r_b.pt for em/sycophancy; store ``r_b_fact`` for fact (absent from
#658). marker/refusal have NO r_B in either source → alignment-to-r_B skipped.

READ-ONLY analysis. Downloads ONLY the selected layer's npz (~2-3 GB). Writes a
JSON report; prints a markdown summary. Does NOT touch task bodies / eval_results
commits. The paired store carries exactly layers 7, 14, 21; ``--layer`` selects
which (default 14 — the original committed run). c_C / c_C_postft are per-layer
already, so L7 / L21 reproduce the L14 recipe at their depth with no other change.

Usage::

    uv run python scripts/issue667_deltac_probe.py                 # L14 (default)
    uv run python scripts/issue667_deltac_probe.py --layer 7
    uv run python scripts/issue667_deltac_probe.py --layer 21
"""

# ruff: noqa: RUF001, RUF002, RUF003  # math/scientific notation in docstrings + messages

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PREVIEW_PREFIX = "issue667_gate_chain_preview/analysis_tensors"
R_B_PATH = "issue658_theory_assumptions/store/r_b.pt"
# Layer is a module-level default (14 — the original committed run's target) that
# main() overrides from --layer. The paired store contains exactly layers 7, 14,
# 21. c_C / c_C_postft are read directly from each per-layer _L{LAYER}.npz (already
# layer-specific), so the analysis path is layer-agnostic — only the download glob,
# the load regex, and the r_B indexing key off LAYER; each threads it explicitly.
LAYER = 14
LAYER_IDX = LAYER - 1  # c_C_all_layers[li-1] convention (hs[1:] drops embeddings)
HIDDEN = 3584
BEHAVIORS = ("em", "sycophancy", "fact", "marker")
# #658 r_b.pt column map (from issue667.__init__ RB_COLUMN_FOR_BEHAVIOR).
RB_COL = {"em": "broad_em", "sycophancy": "sycophancy", "fact": None, "marker": None}
RB_RECIPE = "diffmeans"


def _hf(path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_DATA_REPO, path, repo_type="dataset")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────


def snapshot_layer(layer: int = LAYER) -> Path:
    """Bulk-download ONLY the given layer's npz (parallel, ~2-3 GB) via snapshot_download.

    Per-file hf_hub_download over 1920 files is round-trip-bound and glacial;
    snapshot_download with allow_patterns parallelizes it. Returns the local
    snapshot root (the preview prefix resolves under it).
    """
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        HF_DATA_REPO,
        repo_type="dataset",
        revision="main",
        allow_patterns=[f"{PREVIEW_PREFIX}/*/*/*_L{layer}.npz"],
        max_workers=16,
    )
    return Path(root) / PREVIEW_PREFIX


def load_store(layer: int = LAYER) -> dict[str, dict[tuple[str, str], dict]]:
    """{behavior: {(source, target): npz_dict}} for the given layer (bulk snapshot)."""
    base = snapshot_layer(layer)
    npz_files = sorted(str(p) for p in base.rglob(f"*_L{layer}.npz"))
    out: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for i, p in enumerate(npz_files):
        m = re.search(rf"analysis_tensors/([^/]+)/([^/]+)/([^/]+)_L{layer}\.npz$", p)
        if not m:
            continue
        beh, cell, tgt = m.groups()
        source = cell.rsplit("_seed", 1)[0]
        z = np.load(p, allow_pickle=True)
        d = {}
        for k in (
            "v0",
            "v_plus",
            "c_C",
            "c_Cp",
            "c_C_postft",
            "c_Cp_postft",
            "c_C_all_layers",
            "c_Cp_all_layers",
            "r_b_fact",
        ):
            if k in z.files:
                d[k] = z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k]
        out[beh][(source, tgt)] = d
        if (i + 1) % 200 == 0:
            print(f"  loaded {i + 1}/{len(npz_files)} npz", file=sys.stderr)
    return out


def load_r_b(behavior: str, layer: int = LAYER) -> np.ndarray | None:
    col = RB_COL.get(behavior)
    if col is None:
        return None
    import torch

    d = torch.load(Path(_hf(R_B_PATH)), weights_only=False, map_location="cpu")
    return d["r_b"][col][RB_RECIPE][layer].float().numpy().astype(np.float64)


def fact_r_b_from_store(cells: dict) -> np.ndarray | None:
    for data in cells.values():
        if "r_b_fact" in data:
            return np.asarray(data["r_b_fact"], dtype=np.float64)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SVD / stacked-matrix decomposition
# ─────────────────────────────────────────────────────────────────────────────


def stacked_svd(rows: np.ndarray) -> dict:
    """SVD variance fractions of a (N, H) matrix (rows = per-context Δc vectors)."""
    M = np.asarray(rows, dtype=np.float64)
    assert M.ndim == 2 and M.shape[1] == HIDDEN, M.shape
    n = M.shape[0]
    _u, s, vt = np.linalg.svd(M, full_matrices=False)
    s2 = s**2
    tot = float(s2.sum())
    fr = (s2 / tot) if tot > 0 else np.zeros_like(s2)
    return {
        "n": int(n),
        "top1_frac": float(fr[0]) if len(fr) else float("nan"),
        "top2_frac": float(fr[:2].sum()) if len(fr) >= 2 else float("nan"),
        "top3_frac": float(fr[:3].sum()) if len(fr) >= 3 else float("nan"),
        "sigma2_over_sigma1": float(s[1] / s[0]) if len(s) > 1 and s[0] > 0 else float("nan"),
        "chance_top1_frac": float(1.0 / n),
        "v1": vt[0].astype(np.float64),  # top right-singular vector (direction in H)
    }


def cross_context_cohesion(vecs: list[np.ndarray]) -> dict:
    """Mean pairwise cosine among Δc vectors + cos(each, mean Δc)."""
    V = np.stack(vecs)
    mean_v = V.mean(axis=0)
    # mean pairwise cosine (unit-normalized rows, off-diagonal mean of Gram).
    U = np.stack([_unit(v) for v in vecs])
    G = U @ U.T
    n = len(vecs)
    off = (G.sum() - np.trace(G)) / (n * (n - 1)) if n > 1 else float("nan")
    cos_to_mean = [cos(v, mean_v) for v in vecs]
    return {
        "mean_pairwise_cos": float(off),
        "mean_cos_to_mean": float(np.mean(cos_to_mean)),
        "min_cos_to_mean": float(np.min(cos_to_mean)),
        "mean_vec": mean_v,
    }


def base_pca_captured(delta_dir: np.ndarray, base_ctx_matrix: np.ndarray, k: int) -> float:
    """Fraction of ‖delta_dir‖² captured by the top-k PCs of the base-context cloud.

    base_ctx_matrix: (N_ctx, H) stack of base last-input-token context vectors.
    Mean-center, PCA (SVD of centered), project delta_dir onto top-k right-
    singular vectors, return captured-energy fraction. delta_dir is the unit top
    Δc direction (v1). Answers: does Δc lie WITHIN the existing context cloud
    (high) or in a novel subspace (low)?
    """
    X = np.asarray(base_ctx_matrix, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    kk = min(k, vt.shape[0])
    Pk = vt[:kk]  # (k, H)
    d = _unit(np.asarray(delta_dir, dtype=np.float64))
    proj = Pk @ d
    return float(proj @ proj)  # ‖P_k d‖² / ‖d‖² (d is unit)


# ─────────────────────────────────────────────────────────────────────────────
# Per-behavior analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_behavior(behavior: str, cells: dict, r_b: np.ndarray | None) -> dict:
    # source-grain Δc: one per source-cell (diagonal cell has c_C / c_C_postft).
    src_delta: dict[str, np.ndarray] = {}
    src_c_base: dict[str, np.ndarray] = {}  # base source context vec
    src_w_hat: dict[str, np.ndarray] = {}  # answer-side write at diagonal
    for (source, target), data in cells.items():
        if source == target and "c_C" in data and "c_C_postft" in data:
            src_delta[source] = data["c_C_postft"] - data["c_C"]
            src_c_base[source] = data["c_C"]
            if "v_plus" in data and "v0" in data:
                src_w_hat[source] = data["v_plus"] - data["v0"]

    if len(src_delta) < 2:
        return {"status": "insufficient_sources", "n": len(src_delta)}

    sources = sorted(src_delta)
    dvecs = [src_delta[s] for s in sources]

    # (a) norms
    norms = np.array([np.linalg.norm(v) for v in dvecs])
    base_norms = np.array([np.linalg.norm(src_c_base[s]) for s in sources])
    rel_drift = norms / base_norms  # ‖Δc‖ / ‖c_C‖ (compare vs #667's 0.54–0.77)

    # (b) SVD across SOURCE contexts
    svd_src = stacked_svd(np.stack(dvecs))

    # (c) cohesion among source Δc
    coh = cross_context_cohesion(dvecs)

    # (d) alignment of dominant Δc direction (v1)
    v1 = svd_src["v1"]
    mean_delta = coh["mean_vec"]
    # default-assistant context (base) — the drift-toward-default probe.
    c_default = src_c_base.get("default")
    # cos(v1, w_hat) averaged over sources; cos(v1, r_B); cos(v1, c_source - c_default)
    align = {}
    if src_w_hat:
        align["cos_v1_what_mean"] = float(
            np.nanmean([abs(cos(v1, src_w_hat[s])) for s in sources if s in src_w_hat])
        )
        align["cos_meandelta_what_mean"] = float(
            np.nanmean([abs(cos(mean_delta, src_w_hat[s])) for s in sources if s in src_w_hat])
        )
    if r_b is not None:
        align["cos_v1_rb"] = abs(cos(v1, r_b))
        align["cos_meandelta_rb"] = abs(cos(mean_delta, r_b))
    if c_default is not None:
        # does the mean shift point TOWARD the default context? cos(mean Δc, c_default - c_source)
        toward_default = [
            cos(src_delta[s], c_default - src_c_base[s]) for s in sources if s != "default"
        ]
        align["cos_delta_toward_default_mean"] = float(np.nanmean(toward_default))
        # cos(Δc, c_source - c_default): does Δc align with the source's own offset from default?
        along_own = [
            cos(src_delta[s], src_c_base[s] - c_default) for s in sources if s != "default"
        ]
        align["cos_delta_along_source_minus_default_mean"] = float(np.nanmean(along_own))

    # base-context PCA span: is v1 within the base-context cloud?
    base_ctx_matrix = np.stack([src_c_base[s] for s in sources])
    align["v1_base_pca_captured_k1"] = base_pca_captured(v1, base_ctx_matrix, 1)
    align["v1_base_pca_captured_k3"] = base_pca_captured(v1, base_ctx_matrix, 3)
    align["v1_base_pca_captured_k5"] = base_pca_captured(v1, base_ctx_matrix, 5)
    align["mean_delta_base_pca_captured_k5"] = base_pca_captured(mean_delta, base_ctx_matrix, 5)

    # TARGET grain (query-side shift, mirrors A3.8): per source-cell, stack Δq
    # across its 30 targets and SVD. Report the distribution over source-cells.
    tgt_svd_rows = []
    for source in sources:
        qdeltas = []
        for (s2, _tgt), data in cells.items():
            if s2 != source:
                continue
            if "c_Cp" in data and "c_Cp_postft" in data:
                qdeltas.append(data["c_Cp_postft"] - data["c_Cp"])
        if len(qdeltas) >= 2:
            r = stacked_svd(np.stack(qdeltas))
            tgt_svd_rows.append(
                {
                    "source": source,
                    **{
                        k: r[k]
                        for k in (
                            "n",
                            "top1_frac",
                            "top2_frac",
                            "top3_frac",
                            "sigma2_over_sigma1",
                            "chance_top1_frac",
                        )
                    },
                }
            )
    tgt_summary = {}
    if tgt_svd_rows:
        for key in ("top1_frac", "top2_frac", "top3_frac"):
            vals = np.array([r[key] for r in tgt_svd_rows])
            tgt_summary[key + "_mean"] = float(np.mean(vals))
            tgt_summary[key + "_median"] = float(np.median(vals))
        tgt_summary["chance_top1_frac_mean"] = float(
            np.mean([r["chance_top1_frac"] for r in tgt_svd_rows])
        )
        tgt_summary["n_source_cells"] = len(tgt_svd_rows)

    return {
        "status": "ok",
        "n_sources": len(sources),
        "sources": sources,
        "delta_norm": {
            "mean": float(norms.mean()),
            "median": float(np.median(norms)),
            "min": float(norms.min()),
            "max": float(norms.max()),
        },
        "rel_drift_deltac_over_cC": {
            "mean": float(rel_drift.mean()),
            "median": float(np.median(rel_drift)),
            "min": float(rel_drift.min()),
            "max": float(rel_drift.max()),
        },
        "svd_over_sources": {
            k: svd_src[k]
            for k in (
                "n",
                "top1_frac",
                "top2_frac",
                "top3_frac",
                "sigma2_over_sigma1",
                "chance_top1_frac",
            )
        },
        "svd_over_targets_per_source": tgt_summary,
        "cohesion": {
            k: coh[k] for k in ("mean_pairwise_cos", "mean_cos_to_mean", "min_cos_to_mean")
        },
        "alignment": align,
        # keep v1 + mean_delta for cross-behavior comparison
        "_v1": v1,
        "_mean_delta": mean_delta,
    }


def cross_behavior(results: dict) -> dict:
    """Cosines between top Δc directions (v1) and mean Δc across behaviors."""
    ok = {b: r for b, r in results.items() if r.get("status") == "ok"}
    behs = sorted(ok)
    out = {"pairs_v1": {}, "pairs_mean_delta": {}}
    for i, a in enumerate(behs):
        for b in behs[i + 1 :]:
            out["pairs_v1"][f"{a}__{b}"] = abs(cos(ok[a]["_v1"], ok[b]["_v1"]))
            out["pairs_mean_delta"][f"{a}__{b}"] = abs(
                cos(ok[a]["_mean_delta"], ok[b]["_mean_delta"])
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(a: dict, key: str) -> str:
    v = a.get(key)
    return f"{v:.3f}" if isinstance(v, float) and v == v else "n/a"


def render_md(results: dict, xbeh: dict, layer: int = LAYER) -> str:
    lines = [f"# Δc context-vector shift decomposition (layer {layer}, seed 42)\n"]
    lines.append(
        "| behavior | n_src | ‖Δc‖/‖c_C‖ med | SVD-over-src top1 | top2 | top3 | chance | "
        "mean pairwise cos | SVD-over-tgt top1 (med) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in BEHAVIORS:
        r = results.get(b, {})
        if r.get("status") != "ok":
            lines.append(f"| {b} | — | {r.get('status', '?')} | | | | | | |")
            continue
        s = r["svd_over_sources"]
        t = r.get("svd_over_targets_per_source", {})
        lines.append(
            f"| {b} | {r['n_sources']} | {r['rel_drift_deltac_over_cC']['median']:.3f} | "
            f"{s['top1_frac']:.3f} | {s['top2_frac']:.3f} | {s['top3_frac']:.3f} | "
            f"{s['chance_top1_frac']:.3f} | {r['cohesion']['mean_pairwise_cos']:.3f} | "
            f"{t.get('top1_frac_median', float('nan')):.3f} |"
        )
    lines.append("\n## Alignment of dominant Δc direction (v1) — |cosine|\n")
    lines.append(
        "| behavior | v1·ŵ | v1·r_B | Δc·(→default) | Δc·(src−default) | "
        "v1 in base-PCA k1 | k3 | k5 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in BEHAVIORS:
        r = results.get(b, {})
        if r.get("status") != "ok":
            continue
        a = r["alignment"]
        cells = [
            _fmt(a, "cos_v1_what_mean"),
            _fmt(a, "cos_v1_rb"),
            _fmt(a, "cos_delta_toward_default_mean"),
            _fmt(a, "cos_delta_along_source_minus_default_mean"),
            _fmt(a, "v1_base_pca_captured_k1"),
            _fmt(a, "v1_base_pca_captured_k3"),
            _fmt(a, "v1_base_pca_captured_k5"),
        ]
        lines.append("| " + b + " | " + " | ".join(cells) + " |")
    lines.append("\n## Cross-behavior: |cos| between dominant Δc directions (v1)\n")
    for pair, v in sorted(xbeh.get("pairs_v1", {}).items()):
        lines.append(f"- v1 {pair}: {v:.3f}")
    lines.append("\n## Cross-behavior: |cos| between mean Δc\n")
    for pair, v in sorted(xbeh.get("pairs_mean_delta", {}).items()):
        lines.append(f"- meanΔc {pair}: {v:.3f}")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layer",
        type=int,
        default=LAYER,
        help="residual layer to decompose (paired store has 7, 14, 21; default 14)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    args = ap.parse_args()

    layer = args.layer
    out = args.out or f"/tmp/issue667_deltac_report_L{layer}.json"

    print(f"loading L{layer} store from HF (~2-3 GB, L{layer} npz only) ...", file=sys.stderr)
    store = load_store(layer)
    for b in args.behaviors:
        print(f"  {b}: {len(store.get(b, {}))} cells", file=sys.stderr)

    results = {}
    for b in args.behaviors:
        cells = store.get(b, {})
        r_b = load_r_b(b, layer)
        if r_b is None and b == "fact":
            r_b = fact_r_b_from_store(cells)
        results[b] = analyze_behavior(b, cells, r_b)

    xbeh = cross_behavior(results)
    md = render_md(results, xbeh, layer)

    # strip private vectors before JSON dump
    clean = {b: {k: v for k, v in r.items() if not k.startswith("_")} for b, r in results.items()}
    Path(out).write_text(
        json.dumps(
            {"layer": layer, "seed": 42, "by_behavior": clean, "cross_behavior": xbeh},
            indent=2,
            default=_json_default,
        )
    )
    print(f"\nwrote {out}", file=sys.stderr)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
