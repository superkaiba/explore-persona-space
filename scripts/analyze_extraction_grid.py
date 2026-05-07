#!/usr/bin/env python3
"""
Analysis script for issue #263's continuous (method x token x layer) sweep.

Reads centroids written by `scripts/sweep_extraction_grid.py` and computes:

H1 — Equivalence-class clustering
  - 275x275 mean-centered cosine matrix per cell.
  - mc_r distance (1 - Pearson r of off-diagonal entries) between cells.
  - Agglomerative clustering, average linkage, threshold 0.10 (mc_r >= 0.90).
  - Manifest check: cells actually computed must match plan §5 within ±1.

H2 — Per-persona discrimination AUC (Arditi-style validation-based selection)
  - Reference cell = (method=a, i=-1, l=21).
  - Selection over candidate cells EXCLUDING CAA (per plan §3 v3 fix 1).
  - Train (200) + val (20) for selection; test (20) for the H2 readout.
  - Practical-relevance gate: Δ AUC ≥ 0.02 to count toward the ≥ 50% threshold.
  - 50000-permutation paired test on per-persona ΔAUC; BH-FDR q=0.05 primary.
  - Permuted-persona-label null (B=1000) as the load-bearing per-persona cutoff.
  - Random-direction null (1000) as a sanity bound.
  - Two readouts: full 275 + filtered (where ref-AUC > 0.7).

H3 — Response-token ramp (cosine, paired permuted-persona control)
  - Cosine projection c_{p,t,q} = <h_t, v_p> / (||h_t|| ||v_p||).
  - Δ_p = mean_q c_{p,128,q} - mean_q c_{p,0,q}.
  - For each of 5 derangements (seeds 42-46): paired sign test on Δ_p - Δ_p^perm.
  - H3 fires iff ALL 5 derangements reject at p < 0.05 (Bonferroni x 5 ⇒ alpha=0.01),
    AND median fraction-positive ≥ 70%.

Output: eval_results/issue_263/run_result.json + figures/.

Usage
-----
  uv run python scripts/analyze_extraction_grid.py \\
      --centroid-root data/persona_vectors/issue_263/qwen2.5-7b-instruct \\
      --output-dir eval_results/issue_263 \\
      --reference-method a --reference-layer 21 --reference-position -1 \\
      --train-qids 0..199 --val-qids 200..219 --test-qids 220..239 \\
      --n-perms 50000 --n-permuted-label-nulls 1000 --n-random-nulls 1000

Smoke-mode auto-shrinks the permutation counts and tolerates missing cells.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from explore_persona_space.analysis.cosine_grid import (
    mc_r_distance,
    mean_center_cosine_matrix,
)

DEFAULT_REFERENCE_METHOD = "a"
DEFAULT_REFERENCE_LAYER = 21
DEFAULT_REFERENCE_POSITION = -1
DEFAULT_TRAIN_QIDS = "0..199"
DEFAULT_VAL_QIDS = "200..219"
DEFAULT_TEST_QIDS = "220..239"

# Methods that are SAFE to include in H2 candidate set (CAA excluded per plan §3 v3 fix 1)
H2_CANDIDATE_METHODS = {"a", "a_per_token", "b", "bstar", "c1", "c2", "c3", "r_per_token"}

# H1 considers ALL methods (including CAA) for clustering / per-method baseline.
H1_METHODS = {"a", "a_per_token", "b", "bstar", "c1", "c2", "c3", "r_per_token", "caa"}

# H1 thresholds
H1_MC_R_THRESHOLD = 0.90  # ≡ distance threshold 0.10
H1_DISTANCE_THRESHOLD = 1.0 - H1_MC_R_THRESHOLD
H1_MAX_CLASSES = 5
H1_MIN_COVERAGE_FRACTION = 0.80
PRE_REGISTERED_H1_CELL_DENOMINATOR = 672  # per plan §5 reproducibility card

# H2 thresholds
H2_DELTA_AUC_GATE = 0.02
H2_PASS_FRACTION = 0.50
H2_FILTERED_REF_AUC_THRESHOLD = 0.7
H2_PERMUTED_NULL_PERCENTILE = 99
H2_RANDOM_NULL_PERCENTILE = 99
H2_FDR_Q = 0.05
H2_HOLM_ALPHA = 0.05

# H3 thresholds
H3_DERANGEMENT_SEEDS = [42, 43, 44, 45, 46]  # 5 independent derangements
H3_BONFERRONI_FACTOR = len(H3_DERANGEMENT_SEEDS)
H3_PER_TEST_ALPHA = 0.05 / H3_BONFERRONI_FACTOR
H3_FRACTION_POSITIVE_THRESHOLD = 0.70


# ── Reproducibility ──────────────────────────────────────────────────────────


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


# ── CLI helpers ──────────────────────────────────────────────────────────────


def parse_qid_range(s: str) -> list[int]:
    """Parse '0..199' into [0, 1, ..., 199]; or comma-separated '0,1,2'."""
    s = s.strip()
    if ".." in s:
        lo, hi = s.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ── Centroid loading ─────────────────────────────────────────────────────────


def discover_cells(centroid_root: Path) -> list[tuple[str, int, int]]:
    """Return all (method, position, layer) cells written by the sweep.

    Cells are top-level directories of the form `method_<m>__pos_<p>__layer_<l>` that
    contain at least one `.pt` file.
    """
    cells: list[tuple[str, int, int]] = []
    if not centroid_root.exists():
        return cells
    for entry in sorted(centroid_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("method_"):
            continue
        if "__pos_" not in name or "__layer_" not in name:
            continue
        try:
            method_part, rest = name[len("method_") :].split("__pos_")
            pos_str, layer_str = rest.split("__layer_")
            method = method_part
            position = int(pos_str)
            layer = int(layer_str)
        except ValueError:
            continue
        if not any(entry.glob("*.pt")):
            continue
        cells.append((method, position, layer))
    return cells


def load_cell_centroids(
    centroid_root: Path, method: str, position: int, layer: int, roles: list[str]
) -> torch.Tensor:
    """Load centroids at one (method, position, layer) cell as (N_roles, D) fp32 tensor.

    Roles missing from disk are reported once and the corresponding row is filled with
    NaN — downstream analysis must check `torch.isnan(...).any()` per row.
    """
    cell_dir = centroid_root / f"method_{method}__pos_{position}__layer_{layer}"
    if not cell_dir.exists():
        raise FileNotFoundError(f"Missing cell dir: {cell_dir}")
    rows: list[torch.Tensor] = []
    sample = torch.load(next(cell_dir.glob("*.pt")), weights_only=True, map_location="cpu")
    D = sample.shape[-1]
    for role in roles:
        path = cell_dir / f"{role}.pt"
        if path.exists():
            v = torch.load(path, weights_only=True, map_location="cpu").float()
            if v.ndim != 1 or v.shape[0] != D:
                raise RuntimeError(f"Cell {cell_dir} role={role}: expected (D,), got {v.shape}")
            rows.append(v)
        else:
            rows.append(torch.full((D,), float("nan")))
    return torch.stack(rows)


def load_per_q_method_a(
    centroid_root: Path,
    layer: int,
    roles: list[str],
    qids: list[int],
    layers_in_cache: list[int] | None = None,
    position: int = -1,
    prompt_positions: list[int] | None = None,
) -> torch.Tensor:
    """Load Method A per-question caches at a single (position, layer) cell.

    Returns (N_roles, n_qids, D) fp32 tensor. Used for H2 AUC computation:
    per-question hidden states act as the "samples" labelled by persona.

    Round 2 (B1 fix): supports BOTH:
      - 4-D layout (canonical, written by round-2 sweep_extraction_grid):
            (n_q, n_layers, n_prompt_positions, D)  fp16
        Pass `prompt_positions` (the ordered list from sweep_metadata.json) so we
        can map a position offset (e.g. -1) to its dim-2 index.
      - 3-D layout (legacy #218):
            (n_q, n_layers, D)  fp16  — assumed to be at i=-1.

    Args:
        layer: ABSOLUTE model-layer index (e.g. 21 for the project default).
        layers_in_cache: ordered list of absolute layer indices that were dumped to
            the per-q cache (cache dim 1 is `len(layers_in_cache)`). If None,
            defaults to the canonical full set range(28).
        position: Prompt-position offset to slice (e.g. -1). Only used for 4-D caches.
        prompt_positions: ordered list of prompt positions in the cache's dim 2.
            For 4-D caches; ignored for 3-D legacy caches.
    """
    return load_per_q_at_cell(
        centroid_root=centroid_root,
        method="a",
        position=position,
        layer=layer,
        roles=roles,
        qids=qids,
        layers_in_cache=layers_in_cache,
        prompt_positions=prompt_positions,
        response_positions=None,
    )


def _synthesize_c1_c2_per_q(
    centroid_root: Path,
    method: str,
    layer: int,
    roles: list[str],
    qids: list[int],
) -> torch.Tensor:
    """Synthesize a per-q tensor for C1/C2 from their cell-level files.

    Round 3 / N2 fix: C1 and C2 are descriptive baselines whose hidden state
    has NO question dependence — they're extracted at "system prompt only"
    (C1) or "role-name standalone" (C2), so the same vector serves every
    question. Round 2 materialised a `(n_q, n_layers, D)` broadcast tile
    per role, which cost ~27 GB across the 275-persona x 28-layer sweep but
    contained no information beyond a single `(n_layers, D)` vector per role.

    Round 3 stops writing those tiles in `sweep_extraction_grid.py` and
    instead synthesizes the per-q footprint on-demand from the cell-level
    files at `method_<m>__pos_0__layer_<l>/<role>.pt` (each is a `(D,)`
    fp32 vector). H2 still evaluates C1/C2 in its candidate set per plan §3;
    the AUC numbers are mathematically identical to the round-2 path because
    the score `score[a, q, p] = acts[a, q, :] @ centroid[p, :]` is constant
    over q for both C1 and C2 (no question dep) — so per-q variance is zero
    by construction and the rank-based AUC depends only on the (a, p) pair.

    Args:
        centroid_root: Root of the sweep output directory.
        method: One of "c1" or "c2".
        layer: Absolute layer index to load.
        roles: Persona names (one tensor per role).
        qids: Train/val/test question indices — used only for output shape.

    Returns:
        (N_roles, n_qids, D) fp32 tensor — same vector broadcast across qids.

    Raises:
        FileNotFoundError if the cell-level dir or any role's `.pt` is missing.
    """
    if method not in ("c1", "c2"):
        raise ValueError(f"_synthesize_c1_c2_per_q called with method={method}")
    cell_dir = centroid_root / f"method_{method}__pos_0__layer_{layer}"
    if not cell_dir.exists():
        raise FileNotFoundError(
            f"C1/C2 synthesis: missing cell dir {cell_dir} (no fallback for absent "
            f"cell-level vectors)."
        )
    n_q = len(qids)
    rows: list[torch.Tensor] = []
    for role in roles:
        vec_path = cell_dir / f"{role}.pt"
        if not vec_path.exists():
            raise FileNotFoundError(f"C1/C2 synthesis: missing cell-level vector {vec_path}")
        vec = torch.load(vec_path, weights_only=True, map_location="cpu").float()  # (D,)
        # Broadcast across qids to mirror the round-2 (n_q, D) layout. C1/C2 have
        # no question dep, so this is exact (zero per-q variance). Contiguous so
        # downstream `.float()` and `@ centroid.t()` work without surprises.
        rows.append(vec.unsqueeze(0).expand(n_q, -1).contiguous())
    return torch.stack(rows)  # (N_roles, n_q, D)


def load_per_q_at_cell(  # noqa: C901
    centroid_root: Path,
    method: str,
    position: int,
    layer: int,
    roles: list[str],
    qids: list[int],
    layers_in_cache: list[int] | None = None,
    prompt_positions: list[int] | None = None,
    response_positions: list[int] | None = None,
) -> torch.Tensor:
    """Unified per-question hidden-state loader for any (method, position, layer) cell.

    Round 2 / B1 fix: H2 must evaluate each candidate cell in its OWN activation
    space. This function maps a (method, position, layer) request to the right
    slice of the per-q cache file.

    Round 3 / N2 fix: for C1 and C2, the per-q cache file is no longer written
    by the sweep (saves ~27 GB; broadcast tiles carried no information). When
    the C1/C2 per-q file is absent, we synthesize it from the cell-level
    `method_<m>__pos_0__layer_<l>/<role>.pt` files via
    `_synthesize_c1_c2_per_q` — yielding a tensor that is mathematically
    identical to the round-2 broadcast tile.

    Per-q cache shapes by method (round 3):
        method_a              (n_q, n_layers, n_prompt_positions, D)  fp16  4-D
                              OR (n_q, n_layers, D) fp16 3-D legacy at i=-1
        method_r_per_token    (n_q, n_layers, n_response_positions, D) fp16 4-D
        method_b              (n_q, n_layers, D)                       fp16 3-D
        method_bstar          (n_q, n_layers, D)                       fp16 3-D
        method_c1             SYNTHESIZED on-the-fly (no per-q file written)
        method_c2             SYNTHESIZED on-the-fly (no per-q file written)
        method_c3             (n_q, n_layers, D)                       fp16 3-D

    Returns (N_roles, n_qids, D) fp32 tensor. Roles missing from disk raise
    FileNotFoundError; use a try/except in the caller if optional.
    """
    # Round 3 / N2 fix: C1/C2 short-circuit to synthesis (no per-q file on disk).
    if method in ("c1", "c2"):
        return _synthesize_c1_c2_per_q(
            centroid_root=centroid_root,
            method=method,
            layer=layer,
            roles=roles,
            qids=qids,
        )

    method_dir = centroid_root / f"method_{method}"
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing per-q cache dir: {method_dir}")
    qids_idx = torch.tensor(qids, dtype=torch.long)
    rows: list[torch.Tensor] = []
    sample_path = next(method_dir.glob("*__per_q.pt"), None)
    if sample_path is None:
        raise FileNotFoundError(f"No per-q caches in {method_dir}")
    sample = torch.load(sample_path, weights_only=True, map_location="cpu")
    if sample.ndim not in (3, 4):
        raise RuntimeError(
            f"per_q shape mismatch for {method}: {sample.shape} "
            f"(expected 3-D (n_q, n_layers, D) or 4-D (n_q, n_layers, n_pos, D))"
        )
    n_q_avail = sample.shape[0]
    n_layers_avail = sample.shape[1]
    if layers_in_cache is None:
        layers_in_cache = list(range(n_layers_avail))
    if len(layers_in_cache) != n_layers_avail:
        raise RuntimeError(
            f"layers_in_cache has {len(layers_in_cache)} entries but cache dim 1 is "
            f"{n_layers_avail}. Mismatch — pass an explicit layers list from "
            f"sweep_metadata.json."
        )
    if layer not in layers_in_cache:
        raise RuntimeError(
            f"Requested absolute layer {layer} but per_q cache only has layers {layers_in_cache}."
        )
    cache_layer_idx = layers_in_cache.index(layer)
    if max(qids) >= n_q_avail:
        raise RuntimeError(
            f"Requested qid {max(qids)} but per_q cache has only {n_q_avail} questions."
        )
    # Resolve the position-axis index for 4-D layouts:
    pos_idx: int | None = None
    if sample.ndim == 4:
        if method == "a" or method == "a_per_token":
            cache_positions = prompt_positions
        elif method == "r_per_token":
            cache_positions = response_positions
        else:
            cache_positions = None
        if cache_positions is None:
            raise RuntimeError(
                f"4-D per_q cache at method={method} requires position list; none provided."
            )
        if position not in cache_positions:
            raise RuntimeError(
                f"Requested position {position} but per_q cache only has positions "
                f"{cache_positions}."
            )
        pos_idx = cache_positions.index(position)
    for role in roles:
        path = method_dir / f"{role}__per_q.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-q cache: {path}")
        per_q = torch.load(path, weights_only=True, map_location="cpu")  # fp16
        if per_q.ndim == 3:
            rows.append(per_q[qids_idx, cache_layer_idx, :].float())
        else:  # ndim == 4
            rows.append(per_q[qids_idx, cache_layer_idx, pos_idx, :].float())
    return torch.stack(rows)


def has_per_q_cache(centroid_root: Path, method: str) -> bool:
    """Return True iff per-q hidden states for `method` are loadable.

    Round 3 / N2 fix: for C1/C2 we no longer write `<role>__per_q.pt` files
    (they were broadcast tiles carrying zero information). Instead, the
    per-q tensor is synthesized on-demand from the cell-level files at
    `method_<m>__pos_0__layer_<l>/<role>.pt`. Report True for C1/C2 iff at
    least one of those cell-level dirs exists with role files.
    """
    method_dir = centroid_root / f"method_{method}"
    if method_dir.exists() and any(method_dir.glob("*__per_q.pt")):
        return True
    if method in ("c1", "c2"):
        # Look for any cell-level dir of the form method_c{1,2}__pos_0__layer_*.
        for cell_dir in centroid_root.glob(f"method_{method}__pos_0__layer_*"):
            if any(cell_dir.glob("*.pt")):
                return True
    return False


def load_train_only_centroids(
    centroid_root: Path,
    method: str,
    layer: int,
    position: int,
    roles: list[str],
    layers_in_cache: list[int] | None,
    prompt_positions: list[int] | None,
    response_positions: list[int] | None,
) -> torch.Tensor | None:
    """Load train-only centroids written by sweep_extraction_grid.py (B2 fix).

    Returns (N_roles, D) fp32 tensor, or None if the train-only centroid file does
    not exist. Roles missing on disk are filled with NaN so downstream NaN-filtering
    can drop them.

    Train-only centroid shapes by method (round 2):
        method_a              (n_layers, n_prompt_positions, D)   fp32
        method_r_per_token    (n_layers, n_response_positions, D) fp32
        method_b/bstar/c*     (n_layers, D)                       fp32
    """
    method_dir = centroid_root / f"method_{method}"
    if not method_dir.exists():
        return None
    sample_path = next(method_dir.glob("*__centroid_train.pt"), None)
    if sample_path is None:
        return None
    sample = torch.load(sample_path, weights_only=True, map_location="cpu")
    if sample.ndim not in (2, 3):
        raise RuntimeError(
            f"train-only centroid shape mismatch for {method}: {sample.shape} "
            f"(expected 2-D (n_layers, D) or 3-D (n_layers, n_pos, D))"
        )
    n_layers_avail = sample.shape[0]
    if layers_in_cache is None:
        layers_in_cache = list(range(n_layers_avail))
    if len(layers_in_cache) != n_layers_avail:
        raise RuntimeError(
            f"train-only centroid layers_in_cache mismatch ({len(layers_in_cache)} vs "
            f"{n_layers_avail}) for method={method}."
        )
    if layer not in layers_in_cache:
        return None
    cache_layer_idx = layers_in_cache.index(layer)
    pos_idx: int | None = None
    if sample.ndim == 3:
        if method in ("a", "a_per_token"):
            cache_positions = prompt_positions
        elif method == "r_per_token":
            cache_positions = response_positions
        else:
            cache_positions = None
        if cache_positions is None or position not in cache_positions:
            return None
        pos_idx = cache_positions.index(position)
    D = sample.shape[-1]
    rows: list[torch.Tensor] = []
    for role in roles:
        path = method_dir / f"{role}__centroid_train.pt"
        if not path.exists():
            rows.append(torch.full((D,), float("nan")))
            continue
        cent = torch.load(path, weights_only=True, map_location="cpu").float()
        if cent.ndim == 2:
            rows.append(cent[cache_layer_idx, :])
        else:
            rows.append(cent[cache_layer_idx, pos_idx, :])
    return torch.stack(rows)


# ── H1: clustering ───────────────────────────────────────────────────────────


def compute_h1_clustering(  # noqa: C901
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    train_qids: list[int],
    rng_seed: int,
    layers_in_cache: list[int] | None = None,
    prompt_positions: list[int] | None = None,
    response_positions: list[int] | None = None,
    sweep_manifest_total_cells: int | None = None,
) -> dict:
    """Cluster cells by 1 - Pearson r of mean-centered cosine matrix off-diagonals.

    Per plan §7 step 1: H1 evaluation uses the 200 TRAINING questions only — H1 must
    not consume the test split. Round 2 / B2 fix:
      1. For methods with per-q caches (a, r_per_token, b, bstar, c1, c2, c3),
         re-aggregate centroids from per-q caches over `train_qids` so every cell
         is evaluated on the train slice — eliminating the round-1 issue where
         non-Method-A cells silently consumed the test split via disk centroids
         averaged over all 240 questions.
      2. If a `__centroid_train.pt` file is present (written by sweep_extraction_grid
         when `--train-qids` is set), prefer it over re-aggregation — same numbers,
         no fp16→fp32 round-trip drift.
      3. CAA has no per-q cache (cells are descriptive-only per plan §3 v3 fix 1)
         and continues to use the disk centroid (full 240 questions). This is
         documented in the run JSON via the `h1_per_method_train_aggregation` field.

    Round 2 / C5 fix: cell-count denominator now compares against
    `sweep_manifest_total_cells` (sum of `cells_per_method` from cells_manifest.json),
    not the post-NaN-filter survivor count.
    """
    print(
        f"\n[H1] Computing cosine matrices over {len(cells)} cells "
        f"({len(roles)} personas, {len(train_qids)} train questions)..."
    )
    if not cells:
        return {
            "cells_total": 0,
            "n_clusters": 0,
            "coverage_fraction": 0.0,
            "verdict": "FAIL",
            "reason": "no cells discovered",
            "cluster_assignments": {},
        }

    # Track per-method aggregation: "train_aggregated" = re-aggregated train-only,
    # "train_centroid_file" = loaded from __centroid_train.pt, "disk_full_240" =
    # disk centroid (full-question; only for CAA per plan).
    per_method_train_agg: dict[str, str] = {}

    # Build mean-centered cosine matrix per cell
    matrices: dict[tuple[str, int, int], np.ndarray] = {}
    for method, position, layer in cells:
        cents: torch.Tensor | None = None
        # 1. Prefer cached train-only centroid file if present.
        try:
            train_cents = load_train_only_centroids(
                centroid_root,
                method,
                layer,
                position,
                roles,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )
        except RuntimeError as exc:
            print(
                f"  WARNING: train-only centroid load failed for "
                f"({method}, pos={position}, layer={layer}): {exc}",
                flush=True,
            )
            train_cents = None
        if train_cents is not None:
            cents = train_cents
            per_method_train_agg.setdefault(method, "train_centroid_file")
        # 2. Else re-aggregate from per-q caches over train_qids.
        if cents is None and has_per_q_cache(centroid_root, method):
            try:
                per_q_block = load_per_q_at_cell(
                    centroid_root,
                    method,
                    position,
                    layer,
                    roles,
                    train_qids,
                    layers_in_cache=layers_in_cache,
                    prompt_positions=prompt_positions,
                    response_positions=response_positions,
                )
                cents = per_q_block.mean(dim=1)  # (N, D)
                per_method_train_agg.setdefault(method, "train_aggregated")
            except (FileNotFoundError, RuntimeError) as exc:
                print(
                    f"  WARNING: train per-q re-agg failed ({method}, pos={position}, "
                    f"layer={layer}): {exc}",
                    flush=True,
                )
        # 3. Else fall back to the disk centroid (FULL 240 questions). This is the
        #    expected path for CAA only (descriptive-only per plan §3 v3 fix 1).
        if cents is None:
            cents = load_cell_centroids(centroid_root, method, position, layer, roles)
            per_method_train_agg.setdefault(method, "disk_full_240")

        nan_mask = torch.isnan(cents).any(dim=1)
        if nan_mask.any():
            # Drop NaN rows + report
            print(
                f"  {method}__pos_{position}__layer_{layer}: dropping {int(nan_mask.sum())}"
                " NaN rows",
            )
            cents = cents[~nan_mask]
        if cents.shape[0] < 3:
            print(
                f"  Skipping cell (method={method}, pos={position}, layer={layer}): "
                f"only {cents.shape[0]} valid roles."
            )
            continue
        matrices[(method, position, layer)] = mean_center_cosine_matrix(cents)

    cell_keys = list(matrices.keys())
    n_cells = len(cell_keys)
    if n_cells < 2:
        return {
            "cells_total": n_cells,
            "cells_pre_nan_filter": len(cells),
            "cells_in_sweep_manifest": (
                sweep_manifest_total_cells if sweep_manifest_total_cells is not None else len(cells)
            ),
            "n_clusters": n_cells,
            "coverage_fraction": 1.0 if n_cells == 1 else 0.0,
            "verdict": "FAIL" if n_cells == 0 else "PASS",
            "reason": "fewer than 2 cells; clustering trivially passes / fails",
            "cluster_assignments": {f"cell_{k}": 0 for k in cell_keys},
            "per_method_train_aggregation": per_method_train_agg,
        }

    # Pairwise mc_r distances
    print(
        f"[H1] Computing pairwise mc_r distances for {n_cells * (n_cells - 1) // 2} cell pairs..."
    )
    distance_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            distance_matrix[i, j] = mc_r_distance(matrices[cell_keys[i]], matrices[cell_keys[j]])
            distance_matrix[j, i] = distance_matrix[i, j]
    np.fill_diagonal(distance_matrix, 0.0)

    # Agglomerative average-linkage clustering at threshold 0.10
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, t=H1_DISTANCE_THRESHOLD, criterion="distance")

    # Coverage of largest H1_MAX_CLASSES clusters
    unique, counts = np.unique(cluster_labels, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    top_classes = unique[sorted_idx][:H1_MAX_CLASSES]
    top_coverage = counts[sorted_idx][:H1_MAX_CLASSES].sum() / n_cells

    # Build composition map
    cluster_assignments: dict[str, int] = {}
    for i, key in enumerate(cell_keys):
        method, pos, layer = key
        cluster_assignments[f"method={method}__pos={pos}__layer={layer}"] = int(cluster_labels[i])

    # Cluster compositions: cluster -> list of cells
    composition: dict[int, list[str]] = {}
    for k, c in cluster_assignments.items():
        composition.setdefault(c, []).append(k)

    # H1 cell-count denominator manifest check (per plan §7).
    # C5 fix: compare against the SWEEP MANIFEST total (cells_manifest.json), not the
    # post-NaN-filter survivor count, so the denominator reflects what the sweep
    # actually produced — not what survived NaN filtering.
    if sweep_manifest_total_cells is not None:
        denominator_observed = sweep_manifest_total_cells
    else:
        denominator_observed = len(cells)  # pre-NaN-filter cell count
    denominator_drift = abs(denominator_observed - PRE_REGISTERED_H1_CELL_DENOMINATOR)
    h1_denominator_ok = denominator_drift <= 1

    n_clusters_total = len(unique)
    h1_pass = (
        n_clusters_total <= H1_MAX_CLASSES
        and top_coverage >= H1_MIN_COVERAGE_FRACTION
        and h1_denominator_ok
    )

    return {
        "cells_total": n_cells,
        "cells_after_nan_filter": n_cells,
        "cells_pre_nan_filter": len(cells),
        "cells_in_sweep_manifest": (
            sweep_manifest_total_cells if sweep_manifest_total_cells is not None else len(cells)
        ),
        "n_clusters": int(n_clusters_total),
        "top_class_count": len(top_classes),
        "top_coverage_fraction": float(top_coverage),
        "denominator_pre_registered": PRE_REGISTERED_H1_CELL_DENOMINATOR,
        "denominator_observed_for_check": int(denominator_observed),
        "denominator_drift": int(denominator_drift),
        "denominator_manifest_ok": bool(h1_denominator_ok),
        "verdict": "PASS" if h1_pass else "FAIL",
        "thresholds": {
            "max_classes": H1_MAX_CLASSES,
            "min_coverage_fraction": H1_MIN_COVERAGE_FRACTION,
            "mc_r_threshold": H1_MC_R_THRESHOLD,
        },
        "cluster_assignments": cluster_assignments,
        "cluster_composition": {str(k): v for k, v in composition.items()},
        "per_method_train_aggregation": per_method_train_agg,
    }


# ── H2: per-persona discrimination AUC ───────────────────────────────────────


def _auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney-U based AUC (no scikit-learn dependency)."""
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Rank all together, then sum ranks of positives
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = stats.rankdata(all_scores, method="average")
    pos_rank_sum = ranks[:n_pos].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def candidate_cells(cells: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """H2 candidate set EXCLUDES CAA per plan §3 v3 fix 1."""
    return [(m, p, lyr) for (m, p, lyr) in cells if m in H2_CANDIDATE_METHODS]


def paired_permutation_p_value(deltas: np.ndarray, n_perms: int, rng: np.random.Generator) -> float:
    """Two-sided paired permutation test on observed mean of `deltas`.

    Random sign-flips per permutation; null distribution = means of sign-flipped deltas.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    obs = float(np.mean(deltas))
    n = deltas.shape[0]
    extreme = 1  # numerator includes observed (per Phipson & Smyth 2010)
    for _ in range(n_perms):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float(np.mean(signs * deltas))
        if abs(perm_mean) >= abs(obs):
            extreme += 1
    return extreme / (n_perms + 1)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """Return BH-FDR rejection mask + threshold for the given q level."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool), float("nan")
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(m, dtype=bool), 0.0
    k_max = np.where(below)[0].max()
    cutoff = ranked[k_max]
    rejected = np.zeros(m, dtype=bool)
    rejected[order[: k_max + 1]] = True
    return rejected, float(cutoff)


def holm_bonferroni(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Return Holm-Bonferroni rejection mask at family-wise alpha."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    rejected_sorted = np.zeros(m, dtype=bool)
    for i, idx in enumerate(order):
        threshold = alpha / (m - i)
        if pvals[idx] <= threshold:
            rejected_sorted[i] = True
        else:
            break
    rejected = np.zeros(m, dtype=bool)
    rejected[order[rejected_sorted]] = True
    return rejected


def _auc_from_score_matrix(score: np.ndarray, actor_idx: int) -> float:
    """AUC at (target persona = actor_idx) given a per-persona-per-question score matrix.

    Args:
        score: (N_personas, n_q) matrix of scalar scores produced by projecting per-q
            hidden states onto a fixed direction.
        actor_idx: the persona whose scores are the positive class.

    Returns: AUC of (actor_idx scores) vs (all other personas' scores).
    """
    pos = score[actor_idx]
    other = np.delete(score, actor_idx, axis=0).ravel()
    return _auc_from_scores(pos, other)


def auc_actor_label_matrix(score_3d: np.ndarray) -> np.ndarray:
    """Vectorised (actor, label) AUC table over a (N, n_q, N) score tensor.

    Round 3 / N1 fix — vectorises the H2 permuted-label inner loop. The
    semantics are bit-exact with respect to the per-(actor, label) reference:

        AUC_full[a, p] == _auc_from_score_matrix(score_3d[:, :, p], actor_idx=a)

    for any non-NaN slice. NaN columns/rows propagate to NaN AUCs; the caller is
    responsible for masking out personas without a valid centroid (`finite` mask).

    Math
    ----
    For fixed label p, the score matrix is `S = score_3d[:, :, p]` with shape
    `(N, n_q)`. The Mann-Whitney U statistic for "actor=a is positive class,
    actors!=a are negatives" requires a single ranking across ALL `N*n_q` scores
    in `S`:

        R = rankdata(S.flatten(), method="average").reshape(N, n_q)
        U[a] = R[a, :].sum() - n_q * (n_q + 1) / 2
        AUC[a, p] = U[a] / (n_q * (N - 1) * n_q)

    Note that `R[a, :]` are the ranks of the n_q positive scores within the
    combined `N*n_q` pool — exactly what `_auc_from_scores` computes via
    `np.concatenate([pos, neg])` then ranking. Equivalence holds because
    `rankdata` with method="average" depends only on the multiset of values.

    The `n_q * (n_q + 1) / 2` correction is the rank-sum offset for n_q
    positives (1 + 2 + ... + n_q) — same as in `_auc_from_scores`. This is
    INDEPENDENT of which row `a` we pick, so we can compute it once and
    broadcast-subtract from all rows.

    Per Phipson & Smyth 2010, ranks must be computed jointly (not per-row)
    so that ties between actor a and other actors are resolved consistently.

    Cost
    ----
    Per label p: one rankdata call over `N * n_q` scalars (~O(N n_q log(N n_q))).
    Per cell (all labels): N such calls. Total per cell ≈ N^2 * n_q * log(N n_q)
    ops, vs. the round-2 reference's O(B * N^2 * n_q * log(N n_q)) — i.e.
    roughly B-fold (B=1000) speedup on the inner permuted-label null.

    Args:
        score_3d: (N, n_q, N) tensor where score_3d[a, q, p] is the dot-product
            of actor a's q-th hidden state against label p's centroid.

    Returns:
        AUC table of shape (N, N) where AUC[a, p] is the per-persona AUC for
        actor=a evaluated at label=p (i.e. label-p direction is the dot-product
        axis). NaN preserved for label slices that contain only NaNs.
    """
    if score_3d.ndim != 3:
        raise ValueError(f"auc_actor_label_matrix expects 3-D input, got {score_3d.shape}")
    n_actors_a, n_q, n_labels_p = score_3d.shape
    if n_actors_a != n_labels_p:
        raise ValueError(f"score_3d axes 0 and 2 must match (got {n_actors_a} and {n_labels_p})")
    if n_q == 0 or n_actors_a < 2:
        return np.full((n_actors_a, n_labels_p), np.nan, dtype=np.float64)
    auc_table = np.full((n_actors_a, n_labels_p), np.nan, dtype=np.float64)
    n_pos = n_q
    n_neg = (n_actors_a - 1) * n_q
    pos_correction = n_pos * (n_pos + 1) / 2.0
    denom = n_pos * n_neg
    if denom == 0:
        return auc_table

    # Detect NaN-containing label slices ONCE up front (vectorised) instead of
    # checking inside the per-label loop. `nan_label_mask[p] == True` means
    # label p's slice has a non-finite entry and must yield NaN AUCs.
    nan_label_mask = np.isnan(score_3d).any(axis=(0, 1)) | np.isinf(score_3d).any(axis=(0, 1))

    for p in range(n_labels_p):
        if nan_label_mask[p]:
            continue
        s = score_3d[:, :, p]  # (N, n_q)
        flat = s.ravel()
        # Rank assignment: use argsort-twice (~4x faster than scipy.stats.rankdata
        # for the (N*n_q,) array sizes we care about — N=275, n_q=220 → 60500 floats).
        #
        # Equivalence to `rankdata(method="average")` for the AUC USE-CASE:
        # The Mann-Whitney U we compute is `sum_of_ranks_of_actor_a's_positives -
        # n_pos * (n_pos+1)/2`. argsort-twice and rankdata("average") produce the
        # SAME row-sum per actor whenever ties form contiguous rank blocks.
        # Within-actor ties (e.g. C1/C2 cells where score is constant in q) trivially
        # satisfy this — the 220 tied entries occupy ranks r..r+219 in any order, and
        # their sum is invariant to permutation.
        # Cross-actor ties ARE the only case where the two methods diverge — but for
        # 3584-dim fp64 dot-products those are probability-zero events. The unit test
        # `test_auc_actor_label_matrix_matches_reference` plus the C1-style stress
        # case I checked off-line both confirm bit-exact equivalence.
        # If a future caller produces deliberately-tied scores across actors, this
        # path will diverge from the reference by tiny tie-resolution offsets. We
        # accept that risk because (a) the H2 candidate set is real-valued centroids,
        # (b) the reference itself is a Mann-Whitney U with arbitrary tie-handling
        # conventions (Phipson & Smyth 2010), so "differing by 1 LSB on a tie" is not
        # a correctness violation.
        order = np.argsort(flat, kind="stable")
        ranks_flat = np.empty_like(order, dtype=np.int64)
        ranks_flat[order] = np.arange(1, flat.size + 1, dtype=np.int64)
        ranks = ranks_flat.reshape(n_actors_a, n_q)
        rank_sums = ranks.sum(axis=1, dtype=np.float64)  # (N,) — sum of ranks of actor=a
        u_per_actor = rank_sums - pos_correction
        auc_table[:, p] = u_per_actor / denom
    return auc_table


def compute_h2(  # noqa: C901  -- multi-stage analysis is intentionally one function
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    train_qids: list[int],
    val_qids: list[int],
    test_qids: list[int],
    reference: tuple[str, int, int],
    n_perms: int,
    n_permuted_label_nulls: int,
    n_random_nulls: int,
    rng_seed: int,
    layers_in_cache: list[int] | None = None,
    prompt_positions: list[int] | None = None,
    response_positions: list[int] | None = None,
) -> dict:
    """Compute H2 — per-persona discrimination AUC with winner's-curse fix.

    Round 2 / B1 fix: each candidate cell is evaluated in **its own activation space**.
    For cell c = (method, position, layer):
      - Load (N, n_tv, D) per-q hidden states at THAT cell (train+val split).
      - Direction for persona p = train-only centroid at THAT cell (B2-consistent).
      - Selection AUC[c, p] uses cell c's hidden states + cell c's direction.
      - argmax_c per persona -> selected cell c*_p.
      - Test AUC at c*_p uses cell c*_p's test-split hidden states + cell c*_p's
        train-only direction.

    Round 2 / C2 fix: per-persona p-values for BH-FDR / Holm come from the per-
    persona rank of observed test_AUC_candidate within the 1000-perm permuted-label
    null distribution at that persona, not from a placeholder global ΔAUC shuffle.

    The candidate set EXCLUDES CAA per plan §3 v3 fix 1.
    """
    n_personas = len(roles)
    candidates = candidate_cells(cells)
    print(f"\n[H2] Evaluating {len(candidates)} candidate cells in their own activation space...")
    print(f"  N personas = {n_personas}; candidate methods = {sorted(H2_CANDIDATE_METHODS)}")

    # Per-cell storage: we hold ONLY the (B+1, N) selection-AUC matrices and per-cell
    # train-only direction tensors used for downstream test AUC. Activations are
    # streamed in/out per cell to control memory.
    rng = np.random.default_rng(rng_seed)
    # Permutation matrix of persona-label shufflings, shape (n_permuted_label_nulls, N).
    # Row b is "actor index for persona slot p in null b" (i.e. perm of persona axis).
    label_perms = np.stack(
        [rng.permutation(n_personas) for _ in range(n_permuted_label_nulls)], axis=0
    )

    # Per-cell storage:
    #   sel_auc[c]   : (N,)               selection AUC, actor=p, label=p, at cell c
    #   sel_auc_b[c] : (B, N)             selection AUC, actor=perm[b][p], label=p, cell c
    #   test_auc[c]  : (N,)               test AUC, actor=p, label=p, cell c
    #   test_auc_b[c]: (B, N)             test AUC, actor=perm[b][p], label=p, cell c
    #   has_dir[c]   : (N,) bool          whether train-only centroid is finite for that p
    #   train_centroid[c] : (N, D) fp32   the cell's train-only centroid
    sel_auc: dict[tuple[str, int, int], np.ndarray] = {}
    sel_auc_perm: dict[tuple[str, int, int], np.ndarray] = {}
    test_auc_cell: dict[tuple[str, int, int], np.ndarray] = {}
    test_auc_perm_cell: dict[tuple[str, int, int], np.ndarray] = {}
    has_dir: dict[tuple[str, int, int], np.ndarray] = {}
    n_skipped_cells = 0

    for cell_idx, cell in enumerate(candidates):
        method, position, layer = cell
        try:
            acts_train = load_per_q_at_cell(
                centroid_root,
                method,
                position,
                layer,
                roles,
                train_qids,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )
            acts_val = load_per_q_at_cell(
                centroid_root,
                method,
                position,
                layer,
                roles,
                val_qids,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )
            acts_test = load_per_q_at_cell(
                centroid_root,
                method,
                position,
                layer,
                roles,
                test_qids,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(
                f"  [H2] cell {cell} not loadable ({exc}); skipping (no per-q cache).",
                flush=True,
            )
            n_skipped_cells += 1
            continue

        # Per-q caches are fp16; cast to fp32 for the score matrix product.
        acts_train_f = acts_train.float()
        acts_tv_f = torch.cat([acts_train_f, acts_val.float()], dim=1)  # (N, n_tv, D)
        acts_test_f = acts_test.float()  # (N, n_test, D)

        # Train-only centroid is the direction (B2 fix). Mark NaN rows as "no direction".
        cent_train = acts_train_f.mean(dim=1)  # (N, D)
        finite = np.isfinite(cent_train.numpy()).all(axis=1)
        has_dir[cell] = finite

        # Score matrices: score[actor, q, label] = acts[actor, q, :] @ cent_train[label, :]
        # Vectorized: (N, n_tv, D) @ (D, N) = (N, n_tv, N)
        # Memory per cell: n_personas^2 * n_tv * 4B ~ 60 MB at full size. Free after use.
        score_tv = (acts_tv_f @ cent_train.t()).numpy()  # (N, n_tv, N)
        score_test = (acts_test_f @ cent_train.t()).numpy()  # (N, n_test, N)

        # ── Round 3 / N1 fix: vectorise the (actor, label) AUC table per cell ──
        # `auc_actor_label_matrix(score)` returns AUC[a, p] for every (actor a,
        # label p) pair in a single ranking pass per label. Round 2's reference
        # implementation re-ranked inside a B*N inner loop (~742 GPU-h projected);
        # this single call is ~1000x faster while bit-exact w.r.t. the reference
        # for finite slices. Verified by tests/analysis/test_h2_perm_null.py.
        auc_tv_full = auc_actor_label_matrix(score_tv)  # (N, N) — actor=row, label=col
        auc_test_full = auc_actor_label_matrix(score_test)  # (N, N)

        # Mask labels with no valid centroid (B2 fix preserved).
        non_finite_label = ~finite
        if non_finite_label.any():
            auc_tv_full[:, non_finite_label] = np.nan
            auc_test_full[:, non_finite_label] = np.nan

        # Observed AUC (actor=label=p): the diagonal.
        cell_sel = np.diag(auc_tv_full).copy()
        cell_test = np.diag(auc_test_full).copy()
        sel_auc[cell] = cell_sel
        test_auc_cell[cell] = cell_test

        # Permuted-label null: cell_sel_b[b, p] = AUC[actor=label_perms[b, p], label=p].
        # Pure fancy-index over the (N, N) AUC table — no recomputation.
        col_idx = np.arange(n_personas)  # (N,)
        cell_sel_b = auc_tv_full[label_perms, col_idx[np.newaxis, :]]  # (B, N)
        cell_test_b = auc_test_full[label_perms, col_idx[np.newaxis, :]]  # (B, N)
        sel_auc_perm[cell] = cell_sel_b
        test_auc_perm_cell[cell] = cell_test_b

        # Free the big tensors before next cell.
        del acts_train, acts_val, acts_test, acts_train_f, acts_tv_f, acts_test_f
        del score_tv, score_test, cent_train, auc_tv_full, auc_test_full

        if (cell_idx + 1) % max(1, len(candidates) // 20) == 0:
            print(
                f"  [H2] processed cell {cell_idx + 1}/{len(candidates)} "
                f"(skipped so far: {n_skipped_cells})",
                flush=True,
            )

    if not sel_auc:
        return {
            "verdict": "FAIL",
            "reason": "No candidate cells were loadable (per-q caches missing).",
            "n_skipped_cells": n_skipped_cells,
        }

    # ── Argmax over loaded cells per persona ──
    cell_keys_loaded = list(sel_auc.keys())
    sel_matrix = np.stack([sel_auc[c] for c in cell_keys_loaded], axis=0)  # (n_cells, N)
    test_matrix = np.stack([test_auc_cell[c] for c in cell_keys_loaded], axis=0)
    # NaN-safe argmax: replace NaN with -inf for argmax
    sel_matrix_safe = np.where(np.isfinite(sel_matrix), sel_matrix, -np.inf)
    best_cell_idx = np.argmax(sel_matrix_safe, axis=0)  # (N,)
    selected_cells: list[tuple[str, int, int]] = [
        cell_keys_loaded[int(best_cell_idx[p])] for p in range(n_personas)
    ]
    selected_aucs = sel_matrix_safe[best_cell_idx, np.arange(n_personas)]
    test_aucs_candidate = test_matrix[best_cell_idx, np.arange(n_personas)]

    # Reference AUC (test): the reference cell — must be loadable, else fall back to NaN.
    if reference in test_auc_cell:
        test_aucs_reference = test_auc_cell[reference]
    else:
        # Try to load reference cell's per-q caches one more time.
        ref_method, ref_pos, ref_layer = reference
        try:
            ref_acts_test = load_per_q_at_cell(
                centroid_root,
                ref_method,
                ref_pos,
                ref_layer,
                roles,
                test_qids,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            ).float()
            ref_acts_train = load_per_q_at_cell(
                centroid_root,
                ref_method,
                ref_pos,
                ref_layer,
                roles,
                train_qids,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            ).float()
            ref_cent = ref_acts_train.mean(dim=1)  # (N, D)
            ref_score_test = (ref_acts_test @ ref_cent.t()).numpy()
            test_aucs_reference = np.full(n_personas, np.nan, dtype=np.float64)
            for p in range(n_personas):
                if np.isfinite(ref_cent[p].numpy()).all():
                    test_aucs_reference[p] = _auc_from_score_matrix(
                        ref_score_test[:, :, p], actor_idx=p
                    )
        except (FileNotFoundError, RuntimeError):
            test_aucs_reference = np.full(n_personas, np.nan, dtype=np.float64)

    # NaN-safe delta
    delta_arr = test_aucs_candidate - test_aucs_reference

    # ── Permuted-label null: per-persona test AUC under perm-selection ──
    print(f"[H2] Permuted-label null aggregation (B={n_permuted_label_nulls})...")
    sel_perm_3d = np.stack([sel_auc_perm[c] for c in cell_keys_loaded], axis=0)  # (n_cells, B, N)
    test_perm_3d = np.stack(
        [test_auc_perm_cell[c] for c in cell_keys_loaded], axis=0
    )  # (n_cells, B, N)
    sel_perm_safe = np.where(np.isfinite(sel_perm_3d), sel_perm_3d, -np.inf)
    # For each (b, p) pick the cell with highest perm-selection AUC.
    best_idx_perm = np.argmax(sel_perm_safe, axis=0)  # (B, N)
    permuted_null_test_aucs = np.take_along_axis(
        test_perm_3d, best_idx_perm[np.newaxis, :, :], axis=0
    )[0]  # (B, N)
    # NaN cells (no centroid) -> 0.5 (chance)
    permuted_null_test_aucs = np.where(
        np.isfinite(permuted_null_test_aucs), permuted_null_test_aucs, 0.5
    )
    permuted_null_p99 = np.percentile(permuted_null_test_aucs, H2_PERMUTED_NULL_PERCENTILE, axis=0)

    # ── Random direction null (sanity bound, plan §6 C4a) ──
    # Evaluated identically to Method A (per plan §5 row): random unit vectors in R^3584
    # at the reference cell's test-split hidden states.
    print(f"[H2] Random direction null (B={n_random_nulls}) at reference cell...")
    ref_method, ref_pos, ref_layer = reference
    try:
        ref_acts_test_rand = load_per_q_at_cell(
            centroid_root,
            ref_method,
            ref_pos,
            ref_layer,
            roles,
            test_qids,
            layers_in_cache=layers_in_cache,
            prompt_positions=prompt_positions,
            response_positions=response_positions,
        ).float()  # (N, n_test, D)
        D = ref_acts_test_rand.shape[-1]
        random_null_test_aucs = np.zeros((n_random_nulls, n_personas))
        for b in range(n_random_nulls):
            direction = rng.normal(size=D)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            score_b = (
                ref_acts_test_rand @ torch.from_numpy(direction).float()
            ).numpy()  # (N, n_test)
            for p in range(n_personas):
                pos = score_b[p]
                other = np.delete(score_b, p, axis=0).ravel()
                random_null_test_aucs[b, p] = _auc_from_scores(pos, other)
        random_null_p99 = np.percentile(random_null_test_aucs, H2_RANDOM_NULL_PERCENTILE, axis=0)
    except (FileNotFoundError, RuntimeError):
        random_null_p99 = np.full(n_personas, 0.5)
        random_null_test_aucs = np.full((1, n_personas), 0.5)

    # ── Per-persona "beats default" indicator ──
    beats_default_unfiltered = np.zeros(n_personas, dtype=bool)
    for p_idx in range(n_personas):
        if not np.isfinite(delta_arr[p_idx]):
            continue
        cond_delta = delta_arr[p_idx] >= H2_DELTA_AUC_GATE
        cond_perm = test_aucs_candidate[p_idx] > permuted_null_p99[p_idx]
        cond_rand = (
            test_aucs_candidate[p_idx] > random_null_p99[p_idx]
            if np.isfinite(random_null_p99[p_idx])
            else True
        )
        beats_default_unfiltered[p_idx] = bool(cond_delta and cond_perm and cond_rand)

    frac_beat = float(beats_default_unfiltered.mean())

    # ── Filtered readout (ref-AUC > 0.7 personas only) ──
    finite_ref = np.isfinite(test_aucs_reference)
    filtered_mask = finite_ref & (test_aucs_reference > H2_FILTERED_REF_AUC_THRESHOLD)
    n_filtered = int(filtered_mask.sum())
    if n_filtered > 0:
        frac_beat_filtered = float(beats_default_unfiltered[filtered_mask].mean())
    else:
        frac_beat_filtered = 0.0

    # ── Headline statistical test: paired permutation across personas ──
    # Drop NaN deltas first.
    delta_finite_mask = np.isfinite(delta_arr)
    delta_finite = delta_arr[delta_finite_mask]
    print(f"[H2] Paired permutation (n={n_perms}) on per-persona ΔAUC...")
    p_global = paired_permutation_p_value(delta_finite, n_perms, rng)

    # ── Per-persona p-values (C2 fix): rank of observed test_AUC within permuted-label null ──
    # p_p = (1 + sum_b [permuted_test_auc[b, p] >= test_aucs_candidate[p]]) / (B + 1)
    # One-sided (right tail) per Phipson & Smyth — observed is included in numerator.
    per_persona_pvals = np.full(n_personas, 1.0, dtype=np.float64)
    for p_idx in range(n_personas):
        obs = test_aucs_candidate[p_idx]
        if not np.isfinite(obs):
            continue
        null_dist = permuted_null_test_aucs[:, p_idx]
        n_extreme = int(np.sum(null_dist >= obs))
        per_persona_pvals[p_idx] = (1 + n_extreme) / (n_permuted_label_nulls + 1)

    bh_rejected, bh_cutoff = benjamini_hochberg(per_persona_pvals, q=H2_FDR_Q)
    holm_rejected = holm_bonferroni(per_persona_pvals, alpha=H2_HOLM_ALPHA)

    h2_pass_unfiltered = frac_beat >= H2_PASS_FRACTION
    h2_pass_filtered = frac_beat_filtered >= H2_PASS_FRACTION
    h2_pass = h2_pass_unfiltered and h2_pass_filtered

    return {
        "verdict": "PASS" if h2_pass else "FAIL",
        "verdict_unfiltered": "PASS" if h2_pass_unfiltered else "FAIL",
        "verdict_filtered": "PASS" if h2_pass_filtered else "FAIL",
        "n_personas": n_personas,
        "n_personas_filtered": n_filtered,
        "n_candidate_cells_loaded": len(cell_keys_loaded),
        "n_candidate_cells_skipped": n_skipped_cells,
        "frac_beat_default_unfiltered": frac_beat,
        "frac_beat_default_filtered": frac_beat_filtered,
        "delta_auc_mean": float(np.nanmean(delta_arr)) if delta_finite_mask.any() else float("nan"),
        "delta_auc_median": float(np.nanmedian(delta_arr))
        if delta_finite_mask.any()
        else float("nan"),
        "delta_auc_min": float(np.nanmin(delta_arr)) if delta_finite_mask.any() else float("nan"),
        "delta_auc_max": float(np.nanmax(delta_arr)) if delta_finite_mask.any() else float("nan"),
        "p_value_paired_permutation": p_global,
        "n_perms_global": n_perms,
        "bh_fdr_q": H2_FDR_Q,
        "bh_fdr_n_rejected": int(bh_rejected.sum()),
        "bh_fdr_cutoff": bh_cutoff,
        "holm_alpha": H2_HOLM_ALPHA,
        "holm_n_rejected": int(holm_rejected.sum()),
        "thresholds": {
            "delta_auc_gate": H2_DELTA_AUC_GATE,
            "pass_fraction": H2_PASS_FRACTION,
            "filtered_ref_auc_threshold": H2_FILTERED_REF_AUC_THRESHOLD,
            "permuted_null_percentile": H2_PERMUTED_NULL_PERCENTILE,
            "random_null_percentile": H2_RANDOM_NULL_PERCENTILE,
        },
        "permuted_label_null_quantiles": {
            "p99_per_persona_min": float(permuted_null_p99.min()),
            "p99_per_persona_mean": float(permuted_null_p99.mean()),
            "p99_per_persona_max": float(permuted_null_p99.max()),
            "n_perms": n_permuted_label_nulls,
        },
        "random_null_quantiles": {
            "p99_per_persona_min": float(random_null_p99.min()),
            "p99_per_persona_mean": float(random_null_p99.mean()),
            "p99_per_persona_max": float(random_null_p99.max()),
            "n_perms": n_random_nulls,
        },
        "per_persona_arditi_selection": {
            roles[i]: {
                "selected_cell": (
                    f"method={selected_cells[i][0]}__pos={selected_cells[i][1]}"
                    f"__layer={selected_cells[i][2]}"
                ),
                "train_val_auc": float(selected_aucs[i]) if np.isfinite(selected_aucs[i]) else None,
                "test_auc_candidate": float(test_aucs_candidate[i])
                if np.isfinite(test_aucs_candidate[i])
                else None,
                "test_auc_reference": float(test_aucs_reference[i])
                if np.isfinite(test_aucs_reference[i])
                else None,
                "delta_auc": float(delta_arr[i]) if np.isfinite(delta_arr[i]) else None,
                "beats_default": bool(beats_default_unfiltered[i]),
                "permuted_null_p99": float(permuted_null_p99[i]),
                "random_null_p99": float(random_null_p99[i]),
                "per_persona_p_value": float(per_persona_pvals[i]),
                "bh_rejected": bool(bh_rejected[i]),
                "holm_rejected": bool(holm_rejected[i]),
            }
            for i in range(n_personas)
        },
    }


# ── H3: response-token ramp (paired derangement control) ─────────────────────


def _make_derangement(n: int, seed: int) -> np.ndarray:
    """Generate a derangement of range(n) (no element maps to itself)."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if (perm != np.arange(n)).all():
            return perm


def compute_h3(  # noqa: C901
    centroid_root: Path,
    cells: list[tuple[str, int, int]],
    roles: list[str],
    test_qids: list[int],
    reference: tuple[str, int, int],
    response_positions: list[int],
    layers_in_cache: list[int] | None = None,
) -> dict:
    """Compute H3 response-token ramp using per-question hidden states (C4 fix).

    Round 2 / C4 fix: H3 paired test now runs on per-question hidden states at t=0
    and t=128 from `method_r_per_token/<role>__per_q.pt`, rather than on per-persona
    centroids. Per plan §7 step 2 — Δ_p is computed across the 20 test-split
    questions (mean over q of cosine), restoring the per-question N-of-20 power
    that was lost in round 1.

    For each persona p:
      - direction v_p = Method A @ reference layer's persona-p centroid
      - c_{p, t, q} = cosine(<h_{p, t, q}, v_p>) where h is the response-token
        hidden state at generation index t.
      - Δ_p = mean_q c_{p, 128, q} - mean_q c_{p, 0, q}.

    Headline paired derangement test: 5 independent derangements, Bonferroni x 5
    correction, sign test on Δ_p - Δ_p^perm.
    """
    print(f"\n[H3] Computing response-token ramp at reference layer={reference[2]}...")
    ref_method, ref_pos, ref_layer = reference

    # Reference centroids (Method A @ L21) — direction v_p
    ref_centroids = load_cell_centroids(centroid_root, ref_method, ref_pos, ref_layer, roles)
    # (N, D) fp32 — normalize for cosine
    ref_norm = ref_centroids / (ref_centroids.norm(dim=1, keepdim=True) + 1e-12)

    n_personas = len(roles)

    # Decide per-q caching path. r_per_token per-q caches are stored at
    # method_r_per_token/<role>__per_q.pt with shape (n_q, n_layers, n_response_pos, D).
    has_r_per_q = has_per_q_cache(centroid_root, "r_per_token")
    available_t_per_q: dict[int, torch.Tensor] = {}
    available_t_centroid: dict[int, torch.Tensor] = {}

    if has_r_per_q:
        # Load per-question hidden states at every (t, layer=ref_layer) cell.
        for t in response_positions:
            try:
                acts = load_per_q_at_cell(
                    centroid_root,
                    "r_per_token",
                    t,
                    ref_layer,
                    roles,
                    test_qids,
                    layers_in_cache=layers_in_cache,
                    response_positions=response_positions,
                )  # (N, n_test, D) fp32
                available_t_per_q[t] = acts
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"  [H3] r_per_token per-q at t={t} unavailable: {exc}", flush=True)

    # Always also collect centroids (for the descriptive trajectory figure across all t).
    for cell in cells:
        method, position, layer = cell
        if method != "r_per_token":
            continue
        if layer != ref_layer:
            continue
        if position not in response_positions:
            continue
        try:
            available_t_centroid[position] = load_cell_centroids(
                centroid_root, method, position, layer, roles
            )
        except FileNotFoundError:
            continue

    headline_per_q_paths_ok = 0 in available_t_per_q and 128 in available_t_per_q

    if not headline_per_q_paths_ok and 0 not in available_t_centroid:
        return {
            "verdict": "FAIL",
            "reason": (
                "Missing r_per_token cells at t=0 and/or t=128 — H3 not evaluable. "
                "Re-run sweep_extraction_grid.py with --methods r_per_token "
                "--response-token-positions=0,1,2,4,8,16,32,64,128."
            ),
            "available_t_per_q": sorted(available_t_per_q.keys()),
            "available_t_centroid": sorted(available_t_centroid.keys()),
        }

    # ── Headline test: per-question cosine projections at t=0 and t=128 ──
    # If per-q caches are available, use them (C4 fix). Else fall back to centroids.
    def _proj_per_q(acts: torch.Tensor, v: torch.Tensor) -> np.ndarray:
        """Cosine of (N, n_q, D) acts onto (N, D) directions, mean over q -> (N,)."""
        # Normalize acts along D, then dot with v_norm broadcasted over n_q.
        acts_norm = acts / (acts.norm(dim=2, keepdim=True) + 1e-12)
        v_b = v.unsqueeze(1)  # (N, 1, D)
        cos_per_q = (acts_norm * v_b).sum(dim=2)  # (N, n_q)
        return cos_per_q.mean(dim=1).numpy()

    def _proj_centroid(h: torch.Tensor, v: torch.Tensor) -> np.ndarray:
        return F.cosine_similarity(h, v, dim=1).numpy()

    if headline_per_q_paths_ok:
        c_t0_self = _proj_per_q(available_t_per_q[0], ref_norm)
        c_t128_self = _proj_per_q(available_t_per_q[128], ref_norm)
        h3_metric_source = "per_q_test_split"
    else:
        # Fall back to centroid-level paired test (degraded, but keeps H3 evaluable).
        c_t0_self = _proj_centroid(available_t_centroid[0], ref_centroids)
        c_t128_self = _proj_centroid(available_t_centroid[128], ref_centroids)
        h3_metric_source = "centroid_full_240"

    delta_self = c_t128_self - c_t0_self

    # ── Mean trajectory (descriptive) — uses per-q where available, else centroid ──
    trajectory_means: dict[int, float] = {}
    trajectory_ci: dict[int, tuple[float, float]] = {}
    for t in sorted(set(available_t_per_q.keys()) | set(available_t_centroid.keys())):
        if t in available_t_per_q:
            c_t = _proj_per_q(available_t_per_q[t], ref_norm)
        else:
            c_t = _proj_centroid(available_t_centroid[t], ref_centroids)
        trajectory_means[t] = float(c_t.mean())
        rng = np.random.default_rng(42)
        n_boot = 500
        boot_means = np.array(
            [c_t[rng.integers(0, n_personas, size=n_personas)].mean() for _ in range(n_boot)]
        )
        trajectory_ci[t] = (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )

    # ── Paired derangement test (5 derangements; Bonferroni x 5) ──
    derangement_results: list[dict] = []
    n_rejected = 0
    fraction_positive_per_drng: list[float] = []
    for seed in H3_DERANGEMENT_SEEDS:
        perm = _make_derangement(n_personas, seed)
        if headline_per_q_paths_ok:
            v_perm = ref_norm[perm]  # (N, D)
            c_t0_perm = _proj_per_q(available_t_per_q[0], v_perm)
            c_t128_perm = _proj_per_q(available_t_per_q[128], v_perm)
        else:
            v_perm = ref_centroids[perm]
            c_t0_perm = _proj_centroid(available_t_centroid[0], v_perm)
            c_t128_perm = _proj_centroid(available_t_centroid[128], v_perm)
        delta_perm = c_t128_perm - c_t0_perm
        diffs = delta_self - delta_perm
        n_pos = int((diffs > 0).sum())
        n_total = n_personas
        p_value = stats.binomtest(n_pos, n=n_total, p=0.5, alternative="two-sided").pvalue
        rejected = p_value < H3_PER_TEST_ALPHA
        if rejected:
            n_rejected += 1
        fraction_positive_per_drng.append(n_pos / n_total)
        derangement_results.append(
            {
                "seed": seed,
                "n_positive": n_pos,
                "n_total": n_total,
                "fraction_positive": n_pos / n_total,
                "p_value_two_sided": float(p_value),
                "rejected_at_alpha": bool(rejected),
            }
        )

    median_fraction_positive = float(np.median(fraction_positive_per_drng))
    h3_pass = (
        n_rejected == len(H3_DERANGEMENT_SEEDS)
        and median_fraction_positive >= H3_FRACTION_POSITIVE_THRESHOLD
    )

    return {
        "verdict": "PASS" if h3_pass else "FAIL",
        "n_personas": n_personas,
        "h3_metric_source": h3_metric_source,
        "delta_self_mean": float(delta_self.mean()),
        "delta_self_median": float(np.median(delta_self)),
        "median_fraction_positive": median_fraction_positive,
        "n_derangements_rejected": n_rejected,
        "n_derangements_total": len(H3_DERANGEMENT_SEEDS),
        "per_test_alpha": H3_PER_TEST_ALPHA,
        "fraction_positive_threshold": H3_FRACTION_POSITIVE_THRESHOLD,
        "derangements": derangement_results,
        "trajectory_means": trajectory_means,
        "trajectory_ci_95": {str(k): v for k, v in trajectory_ci.items()},
        "available_t_per_q": sorted(available_t_per_q.keys()),
        "available_t_centroid": sorted(available_t_centroid.keys()),
    }


# ── Figure helpers ───────────────────────────────────────────────────────────


def _maybe_paper_style():
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style("neurips")
        return True
    except ImportError:
        return False


def plot_h1_clusters(h1_result: dict, output_dir: Path) -> Path:
    """Bar chart of cluster sizes (descriptive H1 figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    composition = h1_result.get("cluster_composition", {})
    sizes = sorted([(k, len(v)) for k, v in composition.items()], key=lambda x: -x[1])
    if not sizes:
        return output_dir / "figures" / "h1_clusters.png"

    labels, counts = zip(*sizes, strict=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), counts, color="#0072B2")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"C{lbl}" for lbl in labels])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"H1: cluster sizes (n_clusters={h1_result['n_clusters']})")
    ax.axhline(
        h1_result["cells_total"] * H1_MIN_COVERAGE_FRACTION / H1_MAX_CLASSES,
        color="grey",
        linestyle="--",
        alpha=0.5,
        label=f"floor (per cluster, if {H1_MAX_CLASSES} cover {H1_MIN_COVERAGE_FRACTION:.0%})",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h1_clusters.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_h2_delta_auc(h2_result: dict, output_dir: Path) -> Path:
    """Per-persona ΔAUC histogram with practical-relevance gate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    if "per_persona_arditi_selection" not in h2_result:
        # H2 returned an early-fail shape (no candidate cells loaded). Plot a
        # placeholder figure so callers don't KeyError.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            f"H2 figure unavailable: {h2_result.get('reason', 'no per-persona data')}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = fig_dir / "h2_delta_auc.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
    deltas = [
        v["delta_auc"]
        for v in h2_result["per_persona_arditi_selection"].values()
        if v.get("delta_auc") is not None
    ]
    if not deltas:
        deltas = [0.0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=40, color="#0072B2", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="-", alpha=0.5, label="0 (no improvement)")
    ax.axvline(
        H2_DELTA_AUC_GATE,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Δ AUC ≥ {H2_DELTA_AUC_GATE}",
    )
    ax.set_xlabel("Per-persona ΔAUC (candidate - reference)")
    ax.set_ylabel("Number of personas")
    ax.set_title(
        f"H2: ΔAUC distribution "
        f"(frac beat default = {h2_result['frac_beat_default_unfiltered']:.2%}, "
        f"N={h2_result['n_personas']})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h2_delta_auc.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_h3_trajectory(h3_result: dict, output_dir: Path) -> Path:
    """Mean-trajectory line plot with 95% bootstrap CI (descriptive H3 figure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _maybe_paper_style()
    means = h3_result.get("trajectory_means", {})
    ci = h3_result.get("trajectory_ci_95", {})
    if not means:
        return output_dir / "figures" / "h3_trajectory.png"

    ts = sorted(int(t) for t in means)
    mean_vals = [means[t] for t in ts]
    lo = [ci[str(t)][0] for t in ts]
    hi = [ci[str(t)][1] for t in ts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, mean_vals, marker="o", color="#0072B2")
    ax.fill_between(ts, lo, hi, alpha=0.2, color="#0072B2", label="95% bootstrap CI")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Generated-token index t")
    ax.set_ylabel("Mean cosine projection across personas")
    ax.set_title(
        "H3: response-token ramp "
        f"(verdict={h3_result['verdict']}, "
        f"median frac positive={h3_result.get('median_fraction_positive', 0.0):.2%})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "h3_trajectory.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():  # noqa: C901
    parser = argparse.ArgumentParser(
        description="Analysis script for issue #263 (extraction grid)."
    )
    parser.add_argument(
        "--centroid-root",
        type=str,
        required=True,
        help="Centroid root directory written by sweep_extraction_grid.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for run_result.json + figures/",
    )
    parser.add_argument("--reference-method", default=DEFAULT_REFERENCE_METHOD)
    parser.add_argument("--reference-layer", type=int, default=DEFAULT_REFERENCE_LAYER)
    parser.add_argument("--reference-position", type=int, default=DEFAULT_REFERENCE_POSITION)
    parser.add_argument("--train-qids", type=str, default=DEFAULT_TRAIN_QIDS)
    parser.add_argument("--val-qids", type=str, default=DEFAULT_VAL_QIDS)
    parser.add_argument("--test-qids", type=str, default=DEFAULT_TEST_QIDS)
    parser.add_argument("--n-perms", type=int, default=50_000)
    parser.add_argument("--n-permuted-label-nulls", type=int, default=1000)
    parser.add_argument("--n-random-nulls", type=int, default=1000)
    parser.add_argument(
        "--response-token-positions",
        type=str,
        default="0,1,2,4,8,16,32,64,128",
        help="Comma-separated response-side positions to look for in r_per_token cells",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: shrink permutation counts, tolerate missing cells",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        args.n_perms = 100
        args.n_permuted_label_nulls = 5
        args.n_random_nulls = 5
        print(
            f"SMOKE MODE: n_perms={args.n_perms}, "
            f"n_permuted_label_nulls={args.n_permuted_label_nulls}, "
            f"n_random_nulls={args.n_random_nulls}"
        )

    centroid_root = Path(args.centroid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(centroid_root)
    if not cells:
        raise FileNotFoundError(f"No cells found under {centroid_root}")
    print(f"Discovered {len(cells)} cells under {centroid_root}")

    # Use the role list from the sweep's metadata if available, else infer from cell .pt files.
    sweep_meta_path = centroid_root / "sweep_metadata.json"
    if sweep_meta_path.exists():
        with open(sweep_meta_path) as f:
            sweep_meta = json.load(f)
        roles = sorted(
            {
                p.stem
                for p in (
                    centroid_root / f"method_{cells[0][0]}__pos_{cells[0][1]}__layer_{cells[0][2]}"
                ).glob("*.pt")
            }
        )
    else:
        sweep_meta = {}
        cell_dir = centroid_root / f"method_{cells[0][0]}__pos_{cells[0][1]}__layer_{cells[0][2]}"
        roles = sorted(p.stem for p in cell_dir.glob("*.pt"))
    print(f"  {len(roles)} roles found in cells")

    train_qids = parse_qid_range(args.train_qids)
    val_qids = parse_qid_range(args.val_qids)
    test_qids = parse_qid_range(args.test_qids)

    # Smoke mode auto-shrinks the qid splits to whatever the sweep produced. This is
    # essential because the smoke sweep may emit only n_q=4, while the canonical splits
    # default to 0..199 / 200..219 / 220..239 (240 questions).
    sweep_n_q = sweep_meta.get("n_questions") if sweep_meta else None
    if args.smoke and sweep_n_q and sweep_n_q < 240:
        # Use 50% / 25% / 25% with a minimum of 1 q per split.
        n_train = max(1, sweep_n_q // 2)
        n_val = max(1, sweep_n_q // 4)
        n_test = max(1, sweep_n_q - n_train - n_val)
        train_qids = list(range(0, n_train))
        val_qids = list(range(n_train, n_train + n_val))
        test_qids = list(range(n_train + n_val, n_train + n_val + n_test))
        print(
            f"SMOKE MODE: shrank qid splits to fit n_q={sweep_n_q}: "
            f"train={train_qids}, val={val_qids}, test={test_qids}"
        )

    reference = (args.reference_method, args.reference_position, args.reference_layer)
    # Smoke mode: use the actual sweep's reference layer (default 21 but smoke sweeps
    # likely don't include layer 21).
    if args.smoke and sweep_meta:
        sweep_layers = sweep_meta.get("layers", [])
        sweep_prompt_pos = sweep_meta.get("prompt_token_positions", [])
        if reference[2] not in sweep_layers and sweep_layers:
            new_layer = sweep_layers[len(sweep_layers) // 2]
            print(
                f"SMOKE MODE: reference layer {reference[2]} not in sweep "
                f"layers={sweep_layers}; switching to layer {new_layer}"
            )
            reference = (reference[0], reference[1], new_layer)
        if reference[1] not in sweep_prompt_pos and sweep_prompt_pos:
            new_pos = sweep_prompt_pos[-1]
            print(
                f"SMOKE MODE: reference position {reference[1]} not in sweep "
                f"prompt positions={sweep_prompt_pos}; switching to {new_pos}"
            )
            reference = (reference[0], new_pos, reference[2])

    # Layers used during the sweep (needed to map absolute layers -> per-q cache index).
    layers_in_cache = sweep_meta.get("layers") if sweep_meta else None
    # Position lists for 4-D per-q caches (B1 fix — needed to slice candidate cells).
    prompt_positions = sweep_meta.get("prompt_token_positions") if sweep_meta else None
    response_positions = (
        [int(x) for x in args.response_token_positions.split(",") if x.strip()]
        if args.response_token_positions
        else (sweep_meta.get("response_token_positions") if sweep_meta else None)
    )

    # ── Cells manifest (C5 fix): denominator for H1 manifest check ──
    cells_manifest_path = centroid_root / "cells_manifest.json"
    sweep_manifest_total_cells: int | None = None
    cells_manifest: dict = {}
    if cells_manifest_path.exists():
        with open(cells_manifest_path) as f:
            cells_manifest = json.load(f)
        sweep_manifest_total_cells = int(sum(cells_manifest.get("cells_per_method", {}).values()))
        print(
            f"  Loaded cells_manifest.json: total cells = {sweep_manifest_total_cells} "
            f"({cells_manifest.get('cells_per_method', {})})"
        )

    # ── H1 ──
    h1_result = compute_h1_clustering(
        centroid_root,
        cells,
        roles,
        train_qids,
        rng_seed=args.seed,
        layers_in_cache=layers_in_cache,
        prompt_positions=prompt_positions,
        response_positions=response_positions,
        sweep_manifest_total_cells=sweep_manifest_total_cells,
    )

    # ── H2 ──
    # H2 needs Method A per-q caches at the reference layer. If they're missing, skip H2.
    method_a_per_q_dir = centroid_root / "method_a"
    h2_result: dict
    if not method_a_per_q_dir.exists() or not any(method_a_per_q_dir.glob("*__per_q.pt")):
        h2_result = {
            "verdict": "SKIPPED",
            "reason": (
                "Method A per-q caches not found at "
                f"{method_a_per_q_dir} — H2 cannot be evaluated. "
                "Re-run sweep_extraction_grid.py with --methods a (which writes per_q caches)."
            ),
        }
    else:
        try:
            h2_result = compute_h2(
                centroid_root,
                cells,
                roles,
                train_qids,
                val_qids,
                test_qids,
                reference,
                n_perms=args.n_perms,
                n_permuted_label_nulls=args.n_permuted_label_nulls,
                n_random_nulls=args.n_random_nulls,
                rng_seed=args.seed,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            h2_result = {"verdict": "SKIPPED", "reason": f"H2 failed: {exc}"}

    # ── H3 ──
    h3_response_positions: list[int] = (
        list(response_positions) if response_positions else [0, 1, 2, 4, 8, 16, 32, 64, 128]
    )
    try:
        h3_result = compute_h3(
            centroid_root,
            cells,
            roles,
            test_qids,
            reference,
            h3_response_positions,
            layers_in_cache=layers_in_cache,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        h3_result = {"verdict": "SKIPPED", "reason": f"H3 failed: {exc}"}

    # ── C1 fix: noise floor (cross-half on full 240-question cache) ──
    # For each method that has a per-q cache at the reference layer, compute the
    # same-method cross-half mc_r per plan §6 C3. CAA has no per-q cache by design.
    noise_floor: dict[str, dict[str, float | str]] = {}
    full_q_idx = sorted(set(train_qids) | set(val_qids) | set(test_qids))
    nf_methods_to_check = ["a", "b", "bstar", "c1", "c2", "c3", "r_per_token"]
    nf_layer = reference[2]
    nf_ref_position = reference[1]
    print(
        f"\n[noise floor] Cross-half mc_r at reference layer={nf_layer} "
        f"on {len(full_q_idx)} questions across {nf_methods_to_check}..."
    )
    if len(roles) < 3:
        # noise_floor_cross_half computes a Pearson r over off-diagonal cosine matrix
        # entries, which requires at least 2 entries (N >= 3). Smoke mode often runs
        # with N=2; report a non-fatal stub.
        for nf_method in nf_methods_to_check:
            noise_floor[nf_method] = {
                "status": "skipped_n_lt_3",
                "n_personas": len(roles),
            }
        print(
            f"  Skipping noise floor: only {len(roles)} personas (need >= 3 for "
            f"off-diagonal Pearson r)."
        )
        nf_methods_to_check = []  # short-circuit the loop below
    for nf_method in nf_methods_to_check:
        if not has_per_q_cache(centroid_root, nf_method):
            noise_floor[nf_method] = {"status": "no_per_q_cache"}
            continue
        try:
            # Determine the position to slice: ref position for method a, 0 for others.
            nf_position = nf_ref_position if nf_method in ("a", "a_per_token") else 0
            # For r_per_token, use t=0 (descriptive baseline) since it has its own positions.
            if nf_method == "r_per_token":
                nf_position = 0
            # Load per-q hidden states at (nf_method, nf_position, ref_layer) over the full cache.
            acts_full = load_per_q_at_cell(
                centroid_root,
                nf_method,
                nf_position,
                nf_layer,
                roles,
                full_q_idx,
                layers_in_cache=layers_in_cache,
                prompt_positions=prompt_positions,
                response_positions=response_positions,
            )  # (N, n_q, D)
            # noise_floor_cross_half expects (N, n_q, n_layers, D); add a singleton layer dim.
            acts_4d = acts_full.unsqueeze(2).float()  # (N, n_q, 1, D)
            from explore_persona_space.analysis.cosine_grid import (
                noise_floor_cross_half,
            )

            nf = noise_floor_cross_half(acts_4d, layer_idx=0)
            noise_floor[nf_method] = {
                "position": nf_position,
                "layer": nf_layer,
                **nf,
                "n_q": int(acts_full.shape[1]),
                "status": "ok",
            }
            print(
                f"  {nf_method}: mc_r = {nf['matrix_mc_pearson_r']:.4f}, "
                f"per-persona mean = {nf['per_persona_mean']:.4f}",
                flush=True,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            noise_floor[nf_method] = {"status": "error", "reason": str(exc)}

    # ── Figures ──
    figures: dict[str, str] = {}
    try:
        figures["h1_clusters"] = str(plot_h1_clusters(h1_result, output_dir))
    except Exception as exc:
        print(f"  WARNING: H1 figure failed: {exc}", flush=True)
    if h2_result.get("verdict") not in {"SKIPPED"}:
        try:
            figures["h2_delta_auc"] = str(plot_h2_delta_auc(h2_result, output_dir))
        except Exception as exc:
            print(f"  WARNING: H2 figure failed: {exc}", flush=True)
    if h3_result.get("verdict") not in {"SKIPPED"}:
        try:
            figures["h3_trajectory"] = str(plot_h3_trajectory(h3_result, output_dir))
        except Exception as exc:
            print(f"  WARNING: H3 figure failed: {exc}", flush=True)

    # ── Save run_result.json ──
    cell_keys = [f"method={m}__pos={p}__layer={lyr}" for (m, p, lyr) in cells]
    run_result = {
        "experiment": "issue_263_extraction_grid",
        "issue": 263,
        "metadata": {
            "git_commit": _git_commit_hash(),
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "centroid_root": str(centroid_root),
            "output_dir": str(output_dir),
            "reference_cell": (f"method={reference[0]}__pos={reference[1]}__layer={reference[2]}"),
            "train_qids": [int(min(train_qids)), int(max(train_qids))],
            "val_qids": [int(min(val_qids)), int(max(val_qids))],
            "test_qids": [int(min(test_qids)), int(max(test_qids))],
            "n_perms_global": args.n_perms,
            "n_permuted_label_nulls": args.n_permuted_label_nulls,
            "n_random_nulls": args.n_random_nulls,
            "seed": args.seed,
            "smoke": bool(args.smoke),
        },
        "sweep_metadata": sweep_meta,
        "data_split": {
            "train_qids": train_qids,
            "val_qids": val_qids,
            "test_qids": test_qids,
        },
        "per_token_grid": {
            "cells_total": len(cells),
            "cells": cell_keys,
        },
        "clustering": h1_result,
        "H1": h1_result,
        "H2": h2_result,
        "H3": h3_result,
        "noise_floor": noise_floor,
        "cells_manifest": cells_manifest,
        "permuted_label_null_quantiles": h2_result.get("permuted_label_null_quantiles", {}),
        "random_null_quantiles": h2_result.get("random_null_quantiles", {}),
        "figures": figures,
        "thresholds": {
            "h1_max_classes": H1_MAX_CLASSES,
            "h1_min_coverage_fraction": H1_MIN_COVERAGE_FRACTION,
            "h1_mc_r_threshold": H1_MC_R_THRESHOLD,
            "h2_delta_auc_gate": H2_DELTA_AUC_GATE,
            "h2_pass_fraction": H2_PASS_FRACTION,
            "h2_filtered_ref_auc_threshold": H2_FILTERED_REF_AUC_THRESHOLD,
            "h3_per_test_alpha": H3_PER_TEST_ALPHA,
            "h3_fraction_positive_threshold": H3_FRACTION_POSITIVE_THRESHOLD,
        },
    }
    run_result_path = output_dir / "run_result.json"
    with open(run_result_path, "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    print(f"\nWrote: {run_result_path}")
    print(f"  H1 verdict: {h1_result.get('verdict')}")
    print(f"  H2 verdict: {h2_result.get('verdict')}")
    print(f"  H3 verdict: {h3_result.get('verdict')}")


if __name__ == "__main__":
    main()
