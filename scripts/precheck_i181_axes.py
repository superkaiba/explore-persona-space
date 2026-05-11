#!/usr/bin/env python3
"""Axis-collinearity precheck for issue #181 non-persona trigger leakage.

Loads eval_panel.json and system_prompt_embeddings.pt, computes pair-level
similarity features for all off-diagonal panel-prompt pairs, and builds a
4x4 Spearman correlation matrix across the 4 axes.

Exit codes:
    0 -- CLEAR (all |rho| <= 0.6) or WARN (some |rho| > 0.6 but none > 0.8)
    1 -- BLOCKER (any |rho| > 0.8): panel needs redesign

Also checks cross-cell Jaccard distinguishability (max pairwise <= 0.7).

Usage:
    uv run python scripts/precheck_i181_axes.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from explore_persona_space.analysis.i181_features import (
    compute_lexical_jaccard,
    compute_semantic_cosine,
    compute_struct_match,
    compute_structural_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"

BLOCKER_THRESHOLD = 0.8
WARN_THRESHOLD = 0.6
MAX_JACCARD_THRESHOLD = 0.7

AXIS_NAMES = ["semantic_cos", "lexical_jac", "struct_match", "task_match"]


def load_panel_and_embeddings() -> tuple[list[dict], dict]:
    """Load eval panel and embeddings."""
    panel_path = DATA_DIR / "eval_panel.json"
    emb_path = DATA_DIR / "system_prompt_embeddings.pt"

    if not panel_path.exists():
        logger.error("eval_panel.json not found at %s", panel_path)
        sys.exit(1)

    with open(panel_path) as f:
        panel_data = json.load(f)
    panel = panel_data["panel"]

    embeddings = None
    if emb_path.exists():
        embeddings = torch.load(emb_path, weights_only=False)
    else:
        logger.warning(
            "system_prompt_embeddings.pt not found. "
            "Semantic cosine axis will use placeholder values."
        )

    return panel, embeddings


def get_embedding(panel_id: str, embeddings: dict | None) -> torch.Tensor | None:
    """Look up embedding by panel ID."""
    if embeddings is None:
        return None
    ids = embeddings["ids"]
    emb_matrix = embeddings["embeddings"]
    if panel_id in ids:
        idx = ids.index(panel_id)
        return emb_matrix[idx]
    # Try with train_ prefix
    train_id = f"train_{panel_id.replace('match_', '')}"
    if train_id in ids:
        idx = ids.index(train_id)
        return emb_matrix[idx]
    return None


def compute_pair_features(
    entry1: dict,
    entry2: dict,
    embeddings: dict | None,
) -> dict[str, float]:
    """Compute 4 pair-level similarity features between two panel entries."""
    t1 = entry1["system_prompt"]
    t2 = entry2["system_prompt"]

    # Semantic cosine
    emb1 = get_embedding(entry1["id"], embeddings)
    emb2 = get_embedding(entry2["id"], embeddings)
    sem_cos = compute_semantic_cosine(emb1, emb2) if emb1 is not None and emb2 is not None else 0.0

    # Lexical Jaccard
    lex_jac = compute_lexical_jaccard(t1, t2)

    # Structural match
    feats1 = compute_structural_features(t1)
    feats2 = compute_structural_features(t2)
    # Override task_type from panel data if available
    if entry1.get("family"):
        feats1["task_type"] = entry1["family"]
    if entry2.get("family"):
        feats2["task_type"] = entry2["family"]
    struct = compute_struct_match(feats1, feats2)

    # Task match
    fam1 = entry1.get("family") or "none"
    fam2 = entry2.get("family") or "none"
    task = 1.0 if fam1 == fam2 else 0.0

    return {
        "semantic_cos": sem_cos,
        "lexical_jac": lex_jac,
        "struct_match": struct,
        "task_match": task,
    }


def _compute_correlation_matrix(
    pair_features: dict[str, list[float]],
) -> np.ndarray:
    """Build and log the 4x4 Spearman correlation matrix."""
    logger.info("\n--- 4x4 Spearman Correlation Matrix ---")
    corr_matrix = np.zeros((4, 4))

    for ai, a1 in enumerate(AXIS_NAMES):
        for bi, a2 in enumerate(AXIS_NAMES):
            if ai == bi:
                corr_matrix[ai][bi] = 1.0
            elif ai < bi:
                rho, _pval = stats.spearmanr(pair_features[a1], pair_features[a2])
                corr_matrix[ai][bi] = rho
                corr_matrix[bi][ai] = rho

    header = "".ljust(16) + "  ".join(f"{a:>14}" for a in AXIS_NAMES)
    logger.info(header)
    for ai, a1 in enumerate(AXIS_NAMES):
        row = f"{a1:>16}"
        for bi in range(4):
            row += f"  {corr_matrix[ai][bi]:>14.3f}"
        logger.info(row)

    return corr_matrix


def _check_thresholds(corr_matrix: np.ndarray) -> tuple[bool, bool]:
    """Check blocker/warning thresholds. Returns (has_blocker, has_warning)."""
    has_blocker = False
    has_warning = False

    for ai in range(4):
        for bi in range(ai + 1, 4):
            rho = abs(corr_matrix[ai][bi])
            pair_name = f"{AXIS_NAMES[ai]} vs {AXIS_NAMES[bi]}"
            if rho > BLOCKER_THRESHOLD:
                logger.error(
                    "BLOCKER: |rho|=%.3f > %.1f for %s",
                    rho,
                    BLOCKER_THRESHOLD,
                    pair_name,
                )
                has_blocker = True
            elif rho > WARN_THRESHOLD:
                logger.warning(
                    "WARN: |rho|=%.3f > %.1f for %s",
                    rho,
                    WARN_THRESHOLD,
                    pair_name,
                )
                has_warning = True

    return has_blocker, has_warning


def _check_jaccard_distinguishability(panel: list[dict]) -> None:
    """Check max pairwise lexical Jaccard."""
    n = len(panel)
    logger.info("\n--- Cross-cell Lexical Jaccard Check ---")
    max_jaccard = 0.0
    max_jaccard_pair = ("", "")

    for i in range(n):
        for j in range(i + 1, n):
            jac = compute_lexical_jaccard(panel[i]["system_prompt"], panel[j]["system_prompt"])
            if jac > max_jaccard:
                max_jaccard = jac
                max_jaccard_pair = (panel[i]["id"], panel[j]["id"])

    logger.info("Max pairwise Jaccard: %.3f (%s vs %s)", max_jaccard, *max_jaccard_pair)

    if max_jaccard > MAX_JACCARD_THRESHOLD:
        logger.warning(
            "Max Jaccard %.3f > %.1f between %s and %s. Consider regenerating one of these cells.",
            max_jaccard,
            MAX_JACCARD_THRESHOLD,
            max_jaccard_pair[0],
            max_jaccard_pair[1],
        )


def main():
    panel, embeddings = load_panel_and_embeddings()
    n = len(panel)
    logger.info("Loaded panel with %d entries", n)

    # Compute all off-diagonal pair features
    n_pairs = n * (n - 1) // 2
    logger.info("Computing features for %d off-diagonal pairs...", n_pairs)

    pair_features: dict[str, list[float]] = {axis: [] for axis in AXIS_NAMES}

    for i in range(n):
        for j in range(i + 1, n):
            feats = compute_pair_features(panel[i], panel[j], embeddings)
            for axis in AXIS_NAMES:
                pair_features[axis].append(feats[axis])

    corr_matrix = _compute_correlation_matrix(pair_features)
    has_blocker, has_warning = _check_thresholds(corr_matrix)
    _check_jaccard_distinguishability(panel)

    # Final verdict
    logger.info("\n--- Verdict ---")
    if has_blocker:
        logger.error(
            "BLOCKER: At least one axis pair has |rho| > %.1f. "
            "Panel needs redesign before training.",
            BLOCKER_THRESHOLD,
        )
        sys.exit(1)
    elif has_warning:
        logger.warning(
            "WARN: At least one axis pair has |rho| > %.1f. "
            "Proceed with caveat noted in clean result.",
            WARN_THRESHOLD,
        )
        sys.exit(0)
    else:
        logger.info(
            "CLEAR: All axis pairs have |rho| <= %.1f. Axes are sufficiently independent.",
            WARN_THRESHOLD,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
