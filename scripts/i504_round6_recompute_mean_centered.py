#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003  # em-dashes + Greek ρ intentional in docstring/log
"""Task #504 round-6 — CPU-only recompute of mean-centered cosine matrices.

Loads the existing #472 centroid .pt bundles (from HF or local), recomputes the
cosine matrix using global mean-centering over the FULL bank (the #66/#341
methodology that produced ρ=0.67-0.87 cos-vs-leakage), and writes augmented
bundles in place. NO GPU forward passes — the centroids themselves are
unchanged; only the derived cosine matrix is.

Defaults to the 60-bank from #472 at ``superkaiba1/explore-persona-space-data``
under ``issue472_neg_geometry/geometry/centroids_L{10,15,20}.pt``. Pass
``--older-pool`` to instead recompute the 107-bank from #504 round-5 (the
``older_pool_centroids_L*.pt`` artifacts).

Pipeline per layer:

    1. Load centroids_L<layer>.pt -> Tensor[N, D] + persona_names (len N).
    2. Compute cos_raw = compute_cosine_matrix(C, centering="none").
    3. Compute cos_mc  = compute_cosine_matrix(C, centering="global_mean").
    4. Re-save the bundle with BOTH matrices + cos-to-source diagnostic.
    5. Optionally upload the augmented bundle back to HF (under the same key).

Usage::

    # CPU smoke (default 60-bank, no upload)
    uv run python scripts/i504_round6_recompute_mean_centered.py

    # Recompute and upload the augmented bundles back to HF
    uv run python scripts/i504_round6_recompute_mean_centered.py --upload

    # Recompute the round-5 107-pool instead
    uv run python scripts/i504_round6_recompute_mean_centered.py --older-pool
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.round6.recompute")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REPO_TYPE = "dataset"
HF_SUBDIR = "issue472_neg_geometry/geometry"
DEFAULT_LAYERS = (10, 15, 20)
SOURCE_PERSONA = "villain"  # plan §1 — #504 fixes source to villain


def _hf_key(layer: int, *, older_pool: bool) -> str:
    """Return the HF-Hub path for the centroid bundle of ``layer``."""
    prefix = "older_pool_" if older_pool else ""
    return f"{HF_SUBDIR}/{prefix}centroids_L{layer}.pt"


def _local_path(layer: int, out_dir: Path, *, older_pool: bool) -> Path:
    prefix = "older_pool_" if older_pool else ""
    return out_dir / f"{prefix}centroids_L{layer}.pt"


def _fetch_from_hf(layer: int, *, older_pool: bool) -> Path:
    """Download centroids_L<layer>.pt from HF and return the local cached path."""
    from huggingface_hub import hf_hub_download

    key = _hf_key(layer, older_pool=older_pool)
    log.info("[fetch] HF %s :: %s", HF_REPO, key)
    p = hf_hub_download(HF_REPO, key, repo_type=HF_REPO_TYPE)
    return Path(p)


def _upload_to_hf(local_path: Path, layer: int, *, older_pool: bool) -> None:
    """Upload the augmented bundle back to HF (same key, overwrites)."""
    from huggingface_hub import HfApi

    api = HfApi()
    key = _hf_key(layer, older_pool=older_pool)
    log.info("[upload] %s -> HF %s :: %s", local_path, HF_REPO, key)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=key,
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        commit_message=(
            f"i504 round-6: add global mean-centered cosine matrix to layer {layer} centroid bundle"
        ),
    )


def _recompute_layer(
    layer: int,
    *,
    out_dir: Path,
    older_pool: bool,
    upload: bool,
    source: str,
) -> dict:
    """Recompute mean-centered cosine for one layer; return per-layer report dict."""
    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    local_dst = _local_path(layer, out_dir, older_pool=older_pool)

    # 1. Load the bundle. Try local first, then fall back to HF.
    if local_dst.exists():
        log.info("[load] using local bundle at %s", local_dst)
        bundle_src = local_dst
    else:
        bundle_src = _fetch_from_hf(layer, older_pool=older_pool)

    bundle = torch.load(bundle_src, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "centroids" not in bundle or "persona_names" not in bundle:
        raise ValueError(
            f"unexpected bundle schema at {bundle_src}: "
            f"keys={list(bundle.keys()) if isinstance(bundle, dict) else type(bundle).__name__}"
        )

    centroids: torch.Tensor = bundle["centroids"]
    names: list[str] = list(bundle["persona_names"])
    if centroids.ndim != 2 or centroids.shape[0] != len(names):
        raise ValueError(
            f"centroids/persona_names mismatch at {bundle_src}: "
            f"centroids shape={tuple(centroids.shape)} vs {len(names)} names."
        )

    # 2. Recompute cosines.
    centroids_f32 = centroids.to(dtype=torch.float32)
    cos_raw = compute_cosine_matrix(centroids_f32, centering="none")
    cos_mc = compute_cosine_matrix(centroids_f32, centering="global_mean")

    # 3. Sanity-check + write.
    if source not in names:
        log.warning(
            "[skip-source] source %r not in bank at layer %d (names[:5]=%s); writing both "
            "matrices anyway",
            source,
            layer,
            names[:5],
        )
        cos_to_source_raw = None
        cos_to_source_mc = None
    else:
        src_idx = names.index(source)
        cos_to_source_raw = {n: float(cos_raw[src_idx, j].item()) for j, n in enumerate(names)}
        cos_to_source_mc = {n: float(cos_mc[src_idx, j].item()) for j, n in enumerate(names)}

    augmented = dict(bundle)
    augmented["cos_matrix"] = cos_raw  # backfill / refresh the raw matrix
    augmented["cos_matrix_mean_centered"] = cos_mc
    augmented["centering_provenance"] = {
        "round6_recomputed_at": datetime.now(UTC).isoformat(),
        "global_mean_centered": True,
        "method": "compute_cosine_matrix(C, centering='global_mean')",
        "source_for_diagnostics": source,
        "n_personas": len(names),
        "older_pool": older_pool,
    }

    local_dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, local_dst)
    log.info(
        "[write] layer=%d -> %s (N=%d, cos_raw + cos_mean_centered)",
        layer,
        local_dst,
        len(names),
    )

    if upload:
        _upload_to_hf(local_dst, layer, older_pool=older_pool)

    # 4. Per-layer report (spread comparison so the user sees the methodology delta).
    report = {
        "layer": layer,
        "n_personas": len(names),
        "older_pool": older_pool,
        "local_path": str(local_dst),
    }
    for centering, c2s in (("raw", cos_to_source_raw), ("mean_centered", cos_to_source_mc)):
        if c2s is None:
            report[centering] = None
            continue
        vals = np.asarray([v for k, v in c2s.items() if k != source], dtype=np.float64)
        report[centering] = {
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
            "span": float(vals.max() - vals.min()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n_below_0.9": int((vals < 0.9).sum()),
            "n_below_0.8": int((vals < 0.8).sum()),
            "n_below_0.7": int((vals < 0.7).sum()),
            "n_below_0.5": int((vals < 0.5).sum()),
            "n_below_0.3": int((vals < 0.3).sum()),
            "n_below_0.0": int((vals < 0.0).sum()),
        }
    log.info(
        "[spread L%d] raw       span=%.4f range=[%.4f, %.4f]",
        layer,
        report["raw"]["span"] if report["raw"] else float("nan"),
        report["raw"]["min"] if report["raw"] else float("nan"),
        report["raw"]["max"] if report["raw"] else float("nan"),
    )
    log.info(
        "[spread L%d] mean-cent span=%.4f range=[%.4f, %.4f] (n<0.3=%d, n<0.5=%d, n<0.7=%d)",
        layer,
        report["mean_centered"]["span"] if report["mean_centered"] else float("nan"),
        report["mean_centered"]["min"] if report["mean_centered"] else float("nan"),
        report["mean_centered"]["max"] if report["mean_centered"] else float("nan"),
        report["mean_centered"]["n_below_0.3"] if report["mean_centered"] else -1,
        report["mean_centered"]["n_below_0.5"] if report["mean_centered"] else -1,
        report["mean_centered"]["n_below_0.7"] if report["mean_centered"] else -1,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--layers",
        default=",".join(str(x) for x in DEFAULT_LAYERS),
        help="Comma-separated layer indices to recompute (default: 10,15,20).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/issue_472"),
        help="Local directory to write/refresh centroid bundles (default: data/issue_472).",
    )
    ap.add_argument(
        "--older-pool",
        action="store_true",
        help=("Recompute the 107-persona pool from #504 round-5 instead of the 60-bank from #472."),
    )
    ap.add_argument(
        "--upload",
        action="store_true",
        help="Upload the augmented bundle back to HF (overwrites the existing key).",
    )
    ap.add_argument(
        "--source",
        default=SOURCE_PERSONA,
        help="Source persona to summarize cos-to-source spread for (default: villain).",
    )
    ap.add_argument(
        "--report-path",
        type=Path,
        default=Path("eval_results/issue_504/round6_mean_centered_cos_to_villain.json"),
        help="Write the per-layer spread report here.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=round6_recompute] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    log.info(
        "[start] layers=%s older_pool=%s upload=%s out_dir=%s",
        layers,
        args.older_pool,
        args.upload,
        args.out_dir,
    )

    per_layer = []
    for layer in layers:
        per_layer.append(
            _recompute_layer(
                layer,
                out_dir=args.out_dir,
                older_pool=args.older_pool,
                upload=args.upload,
                source=args.source,
            )
        )

    out = {
        "schema_version": "i504_round6_mean_centered_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "older_pool": args.older_pool,
        "source": args.source,
        "uploaded_to_hf": args.upload,
        "per_layer": per_layer,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(out, indent=2))
    log.info("[done] report -> %s", args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
