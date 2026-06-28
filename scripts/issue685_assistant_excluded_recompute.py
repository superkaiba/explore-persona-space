#!/usr/bin/env python
"""Issue #685 free-analysis follow-up #1 — recompute the geometry metrics
EXCLUDING the bare-default ``assistant`` context (9 persona contexts only).

The clean-result's final result flagged that the bare-default ``assistant``
carries the highest shift magnitude (a mild length effect) and left open the
hypothesis "the per-behavior cosines plausibly hold without it". This script
closes that hypothesis by re-running the EXACT Phase-B metrics
(``behavior_shift_metrics``) on the 9 non-``assistant`` contexts and writing a
side-by-side comparison against the committed 10-context ``metrics.json``.

ANALYSIS-ONLY: CPU. Reads the already-uploaded Phase-A context-vector tensors
(``{instruct,base}_context_vectors.pt``) + the instruct known-direction tensor
from the HF data repo (or the local ``store/issue685/`` copy if present). No
training, no generation, no new model load. The metric math is REUSED verbatim
from ``explore_persona_space.analysis.issue685.metrics.behavior_shift_metrics``
(the production Phase-B entry point), so the 9-context numbers are computed by
the identical code path as the 10-context numbers — only the context subset
changes.

Output: ``eval_results/issue_685/metrics_assistant_excluded.json`` (same shape
as ``metrics.json`` per model, plus a top-level ``comparison_to_10_context``
block: per (model, behavior, layer) the raw / mean-subtracted cosine, PC1 share,
and relative magnitude at 10 vs 9 contexts, the H1 band placement at each, and
whether the placement changed).

Usage::

    uv run python scripts/issue685_assistant_excluded_recompute.py
    uv run python scripts/issue685_assistant_excluded_recompute.py --store-dir store/issue685
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.issue685.metrics import (  # noqa: E402
    behavior_shift_metrics,
)

EXCLUDED_CONTEXT = "assistant"
HF_DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_TENSOR_PREFIX = "issue685_context_shift/analysis_tensors"

# H1 band thresholds (clean-result Methodology, fixed at plan time).
SINGLE_DIRECTION_COS = 0.6
SINGLE_DIRECTION_PC1 = 0.5
CONTEXT_DEP_COS = 0.4
CONTEXT_DEP_PC1 = 0.4
CONTEXT_DEP_MAG = 0.2
NEGLIGIBLE_MAG = 0.2


def _band_placement(cos_raw: float, pc1: float, relmag_mean: float) -> str:
    """Map one (behavior, layer) cell to its competing-hypothesis band.

    Mirrors the clean-result's reporting-aid bands (per-cell, not the
    every-layer negligible read). Returns one of ``single-direction``,
    ``context-dependent``, ``negligible``, or ``intermediate`` (clears no band).

    The magnitude (H0) gate is checked FIRST: a near-zero shift has no
    meaningful direction, so its cosine/PC1 are uninformative and must not
    promote a degenerate cell to ``single-direction``. In the real data the
    single-direction cells carry magnitude ~1.0-1.9 so this ordering never
    changes their placement; it only disambiguates the contradictory
    high-cosine + near-zero-magnitude corner.
    """
    if relmag_mean < NEGLIGIBLE_MAG:
        return "negligible"
    if cos_raw > SINGLE_DIRECTION_COS and pc1 > SINGLE_DIRECTION_PC1:
        return "single-direction"
    if cos_raw < CONTEXT_DEP_COS and pc1 < CONTEXT_DEP_PC1 and relmag_mean >= CONTEXT_DEP_MAG:
        return "context-dependent"
    return "intermediate"


def _resolve_tensor(store_dir: Path, fname: str, hf_required: bool) -> Path | None:
    """Return a local path for ``fname``, downloading from HF if absent locally.

    ``hf_required=False`` allows a graceful ``None`` (e.g. base has no
    known-directions tensor, matching the original run).
    """
    local = store_dir / fname
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download  # local import: only needed on miss

    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"{HF_TENSOR_PREFIX}/{fname}",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        return Path(path)
    except Exception:
        if hf_required:
            raise
        return None


def _load_vectors(pt_path: Path) -> tuple[dict[str, dict[int, torch.Tensor]], dict]:
    """Load a Phase-A ``.pt`` into ``{condition_name: {layer: (H,) vec}}`` + metadata.

    Identical reshape to ``issue685_compute_metrics._load_vectors`` — the
    Phase-A file stores ``centroids: {layer: (n_cond, H)}`` aligned to
    ``persona_names`` (== condition names ``bare__{c}`` / ``{c}__{b}``).
    """
    payload = torch.load(pt_path, weights_only=True)
    centroids = payload["centroids"]
    names = payload["persona_names"]
    metadata = payload.get("metadata", {})
    by_condition: dict[str, dict[int, torch.Tensor]] = {n: {} for n in names}
    for layer, mat in centroids.items():
        assert mat.shape[0] == len(names), (layer, mat.shape, len(names))
        for i, n in enumerate(names):
            by_condition[n][layer] = mat[i]
    return by_condition, metadata


def _load_known_directions(
    path: Path | None, behaviors: list[str], layers: list[int]
) -> dict[tuple[str, int], torch.Tensor] | None:
    """Load ``{(behavior, layer): (H,) u}`` from a known-directions .pt, or None."""
    if path is None or not path.exists():
        return None
    payload = torch.load(path, weights_only=True)
    dirs = payload["directions"]
    out: dict[tuple[str, int], torch.Tensor] = {}
    for b in behaviors:
        if b not in dirs:
            continue
        for layer in layers:
            if layer in dirs[b]:
                out[(b, layer)] = dirs[b][layer]
    return out or None


def _metrics_for_contexts(
    by_condition: dict[str, dict[int, torch.Tensor]],
    context_names: list[str],
    behaviors: list[str],
    layers: list[int],
    known_dirs: dict[tuple[str, int], torch.Tensor] | None,
    null_n_perm: int,
) -> dict:
    """Run ``behavior_shift_metrics`` over the given context subset."""
    bare_by_context = {c: by_condition[f"bare__{c}"] for c in context_names}
    aug_by_condition = {
        f"{c}__{b}": by_condition[f"{c}__{b}"] for c in context_names for b in behaviors
    }
    return behavior_shift_metrics(
        bare_by_context,
        aug_by_condition,
        context_names=context_names,
        behaviors=behaviors,
        layers=layers,
        known_directions=known_dirs,
        null_n_perm=null_n_perm,
        null_seed=42,
    )


def _cell_scalars(cell: dict) -> dict:
    """Pull the four reported scalars out of a metric cell (for comparison)."""
    return {
        "consistency_cosine_raw": cell["consistency_cosine_raw"],
        "consistency_cosine_mean_subtracted": cell["consistency_cosine_mean_subtracted"],
        "pc1_variance_share": cell["pc1_variance_share"],
        "relative_magnitude_mean": cell["relative_magnitude"]["mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #685 follow-up: recompute geometry metrics excluding `assistant`.",
    )
    parser.add_argument(
        "--store-dir",
        default="store/issue685",
        help="Phase-A store dir (falls back to HF download per file if absent).",
    )
    parser.add_argument(
        "--out",
        default="eval_results/issue_685/metrics_assistant_excluded.json",
        help="output JSON path.",
    )
    parser.add_argument(
        "--ten-context-metrics",
        default="eval_results/issue_685/metrics.json",
        help="committed 10-context metrics.json, read for the comparison cross-check.",
    )
    parser.add_argument(
        "--null-n-perm",
        type=int,
        default=200,
        help="consistency-null draws (matches the original full run).",
    )
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    committed = json.loads(Path(args.ten_context_metrics).read_text())

    result: dict = {
        "task": 685,
        "phase": "B-followup-assistant-excluded",
        "excluded_context": EXCLUDED_CONTEXT,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv[1:],
        "null_n_perm": args.null_n_perm,
        "models": {},
        "comparison_to_10_context": {},
    }

    model_tags = ["instruct", "base"]
    for tag in model_tags:
        pt = _resolve_tensor(store_dir, f"{tag}_context_vectors.pt", hf_required=True)
        by_condition, a_meta = _load_vectors(pt)
        full_context_names = a_meta["context_names"]
        behaviors = a_meta["behavior_names"]
        layers = a_meta["layers"]
        assert EXCLUDED_CONTEXT in full_context_names, (
            f"expected `{EXCLUDED_CONTEXT}` in context_names={full_context_names}"
        )
        nine_context_names = [c for c in full_context_names if c != EXCLUDED_CONTEXT]
        assert len(nine_context_names) == len(full_context_names) - 1

        kd_path = _resolve_tensor(store_dir, f"{tag}_known_directions.pt", hf_required=False)
        known_dirs = _load_known_directions(kd_path, behaviors, layers)

        # 9-context recompute (the deliverable) — identical code path as production.
        nine = _metrics_for_contexts(
            by_condition, nine_context_names, behaviors, layers, known_dirs, args.null_n_perm
        )
        nine["phase_a_metadata"] = a_meta
        nine["has_known_direction_projection"] = known_dirs is not None
        result["models"][tag] = nine

        # 10-context recompute (regenerated here for an apples-to-apples comparison
        # AND a reconstruction cross-check against the committed metrics.json).
        ten = _metrics_for_contexts(
            by_condition, full_context_names, behaviors, layers, known_dirs, args.null_n_perm
        )

        committed_cells = committed["models"][tag]["cells"]
        # Cross-check: our regenerated 10-context cosines must match the committed
        # ones (the consistency-null is RNG-seeded identically, so it matches too;
        # we only assert the deterministic geometry scalars here).
        recon_max_abs_err = 0.0
        comparison: dict[str, dict[str, dict]] = {}
        for b in behaviors:
            comparison[b] = {}
            for layer in layers:
                ls = str(layer)
                ten_cell = ten["cells"][b][ls]
                nine_cell = nine["cells"][b][ls]
                committed_cell = committed_cells[b][ls]
                for key in (
                    "consistency_cosine_raw",
                    "consistency_cosine_mean_subtracted",
                    "pc1_variance_share",
                ):
                    recon_max_abs_err = max(
                        recon_max_abs_err, abs(ten_cell[key] - committed_cell[key])
                    )
                recon_max_abs_err = max(
                    recon_max_abs_err,
                    abs(
                        ten_cell["relative_magnitude"]["mean"]
                        - committed_cell["relative_magnitude"]["mean"]
                    ),
                )

                ten_s = _cell_scalars(ten_cell)
                nine_s = _cell_scalars(nine_cell)
                band_10 = _band_placement(
                    ten_s["consistency_cosine_raw"],
                    ten_s["pc1_variance_share"],
                    ten_s["relative_magnitude_mean"],
                )
                band_9 = _band_placement(
                    nine_s["consistency_cosine_raw"],
                    nine_s["pc1_variance_share"],
                    nine_s["relative_magnitude_mean"],
                )
                comparison[b][ls] = {
                    "ten_context": ten_s,
                    "nine_context": nine_s,
                    "delta_9_minus_10": {k: nine_s[k] - ten_s[k] for k in ten_s},
                    "band_10": band_10,
                    "band_9": band_9,
                    "band_changed": band_10 != band_9,
                }
        # Reconstruction sanity gate: regenerated 10-context geometry must match
        # the committed metrics.json (same tensors, same math) to ~1e-4.
        assert recon_max_abs_err < 1e-3, (
            f"[{tag}] regenerated 10-context metrics diverge from committed "
            f"metrics.json by {recon_max_abs_err:.2e} (>1e-3) — reconstruction bug, "
            f"NOT an `assistant`-exclusion effect"
        )
        result["comparison_to_10_context"][tag] = {
            "reconstruction_max_abs_err_vs_committed": recon_max_abs_err,
            "cells": comparison,
        }
        n_changed = sum(
            1 for b in behaviors for layer in layers if comparison[b][str(layer)]["band_changed"]
        )
        print(
            f"[issue685.followup] model={tag}: 10->9 context recompute done; "
            f"recon_err={recon_max_abs_err:.2e}; band-changed cells={n_changed}/"
            f"{len(behaviors) * len(layers)}; projection="
            f"{'on' if known_dirs is not None else 'off'}"
        )

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[issue685.followup] wrote {out_path}")


if __name__ == "__main__":
    main()
