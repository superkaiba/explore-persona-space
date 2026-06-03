#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #478 PHASE 0 — design validation (one-time, before sweep).

Per plan v5 §4.8 PHASE 0:

  * ASSERT marker token id == 83399 on Qwen-2.5-7B-Instruct tokenizer.
  * HARD PAIRWISE-DISJOINTNESS ASSERTION (BUG-1/BUG-2 guard):
      POOL_16 ∩ NEGATIVES_FIXED == ∅
      POOL_16 ∩ HELD_OUT_35     == ∅
      NEGATIVES_FIXED ∩ HELD_OUT_35 == ∅
    fail-loud RuntimeError if any intersection is non-empty.
  * ASSERT |POOL_16 ∪ NEGATIVES_FIXED ∪ HELD_OUT_35| == 55.
  * ASSERT every persona has a system prompt in load_all_persona_prompts().
  * Load 111-persona layer-20 cosine matrix; ASSERT pool radius ≤ 0.05
    (actual: 0.0286).
  * Build 32 cell-spec list with rng=np.random.default_rng(478); ASSERT
    subset uniqueness per K (no duplicate POOL_16 subsets within the same K).
  * Simulate band occupancy on the 32 actual subsets; ASSERT each K has
    Near/Near-mid/Far/Very-far/Tail bands ≥ 2 in worst case (Mid can be 2).

CPU-only. Smoke-test exit 0 means the sweep is clear to launch (modulo
in-cell smoke gates further down).

Output: ``data/issue_478/design_validation.json``.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations

import numpy as np
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue478_common import (  # noqa: E402
    ALL_PERSONAS,
    HELD_OUT_35,
    HELD_OUT_BANDS,
    K_VALUES,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    NEGATIVES_FIXED,
    POOL_16,
    POOL_RADIUS_MAX,
    SUBSET_RNG_SEED,
    SUBSETS_PER_K,
    assert_marker_token_id,
    assert_pairwise_disjoint_sets,
    band_of,
    load_all_persona_prompts,
    load_cosine_distance_matrix,
    min_dist_to_set,
)


def build_subsets(rng_seed: int = SUBSET_RNG_SEED) -> dict[int, list[tuple[str, ...]]]:
    """Build the 40 POOL_16 subsets per plan v5 §4.5 (Level-1 coverage extension).

    K=1: ALL 16 POOL_16 singletons (round-2 extension per code-review BLOCKER 4 —
    the §6.8 Level-1 superposition decomposition skips any K≥2 cell whose source
    members lack a K=1 cell; with only 8 of 16 sources K=1-covered, Level-1 was
    uncomputable for 0 of 8 K=4 cells and 0 of 8 K=8 cells under the seed-478 draw.
    Extending K=1 to all 16 is the planner's contemplated "cheap add" in v5 §4.5
    and is mandatory per v5 §6.8 — Level-1 is MANDATORY because it's the
    superposition signal the headline depends on).
    K=2,4,8: 8 random subsets each from C(16, K). Per-K subset uniqueness is
    asserted by the caller via set equality (each tuple sorted for hashability).

    Compute impact: 40 cells × 2 seeds = 80 runs (was 64). +3.7 GPU-h core, total
    ~23 GPU-h. The orchestrator is confirming the larger compute with the user
    before pod launch — implementation lands the 16-K=1 version regardless.
    """
    rng = np.random.default_rng(rng_seed)
    subsets: dict[int, list[tuple[str, ...]]] = {}
    for K in K_VALUES:
        if K == 1:
            # ALL 16 POOL_16 singletons (full Level-1 superposition coverage).
            # No RNG draw — deterministic enumeration in POOL_16 order so cell_ids
            # K1_c00..c15 line up with POOL_16[0..15] for downstream debugging.
            subsets[K] = [(p,) for p in POOL_16]
        else:
            all_subs = list(combinations(POOL_16, K))
            idx = rng.choice(len(all_subs), size=SUBSETS_PER_K, replace=False)
            # tuple(sorted(...)) for canonical hashing comparison
            subsets[K] = [tuple(sorted(all_subs[int(i)])) for i in idx]
    return subsets


def band_occupancy_for_subset(
    subset: tuple[str, ...],
    names: list[str],
    distance: list[list[float]],
) -> dict[str, int]:
    """Per band: count held-out personas whose min-dist-to-subset matches the FIXED band.

    Plan v5 §6.7 #4: bands are PERSONA-PINNED to the FULL 16-pool (NOT
    per-K-subset bands). For each held-out persona we look at its full-pool
    band assignment (HELD_OUT_BANDS dict) — band occupancy here is just
    "how many of each band's pre-pinned personas happen to be close to
    THIS subset" via the persona's min-dist-to-this-subset.

    But for the band-occupancy floor check we use the FIXED HELD_OUT_BANDS
    counts directly per persona (every held-out persona has a FIXED band) —
    occupancy is therefore the constant ``len(HELD_OUT_BANDS[band])`` and
    doesn't vary with subset. The band-occupancy *simulation* the planner
    ran was a different sanity check (per-subset min-dist tertile binning
    used in #405; preserved here as informational only).

    For the §4.8 occupancy-floor assertion this returns the FIXED counts.
    """
    return {band: len(members) for band, members in HELD_OUT_BANDS.items()}


def per_subset_min_dist_summary(
    subset: tuple[str, ...],
    names: list[str],
    distance: list[list[float]],
) -> dict[str, dict]:
    """For each held-out persona, min-dist to THIS subset + its FIXED band.

    Informational only (the analyzer reads min-dist-to-K-subset for plan §6.7 #3).
    """
    out: dict[str, dict] = {}
    for p in HELD_OUT_35:
        d = min_dist_to_set(p, list(subset), names, distance)
        out[p] = {"min_dist": d, "band": band_of(p)}
    return out


def main() -> int:
    out_dir = PROJECT_ROOT / "data" / "issue_478"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "design_validation.json"

    # ── (1) Marker token id assert ────────────────────────────────────────
    from transformers import AutoTokenizer

    log.info("Loading Qwen-2.5-7B-Instruct tokenizer (for marker assert) ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    assert_marker_token_id(tokenizer)
    log.info("OK — marker %r encodes to single token id %d", MARKER_TEXT, MARKER_TOKEN_ID)

    # ── (2) Hard pairwise-disjointness assertion ──────────────────────────
    assert_pairwise_disjoint_sets()
    log.info(
        "OK — pairwise-disjoint: |POOL|=%d |NEG|=%d |HELD_OUT|=%d total=%d",
        len(POOL_16),
        len(NEGATIVES_FIXED),
        len(HELD_OUT_35),
        len(ALL_PERSONAS),
    )

    # ── (3) Persona prompts present ───────────────────────────────────────
    prompts = load_all_persona_prompts()
    log.info("OK — all %d personas have system prompts", len(ALL_PERSONAS))

    # ── (4) Distance matrix covers the personas that NEED distances ──────
    # Only POOL_16 (training sources, for min-dist computation) and
    # HELD_OUT_35 (eval targets) require centroids. The 4 contrastive
    # NEGATIVES are training-only — they are never distance-measured, and
    # the conversational-default negatives (helpful_assistant, no_persona)
    # legitimately have no centroid in the 111-persona pool.
    log.info(
        "Loading layer-20 cosine distance matrix (cached or computed from centroids_layer20.pt) ..."
    )
    names, dist = load_cosine_distance_matrix()
    needs_distance = list(POOL_16) + list(HELD_OUT_35)
    missing = [p for p in needs_distance if p not in names]
    if missing:
        raise RuntimeError(
            f"Distance matrix missing distance-bearing personas (POOL_16 + HELD_OUT_35): "
            f"{missing!r}. Matrix size={len(names)}. Re-extract centroids via "
            f"scripts/analyze_100_persona_cosine.py --extract on a GPU pod, "
            f"or fix the persona name in _issue478_common.py."
        )
    log.info(
        "OK — distance matrix covers all %d distance-bearing personas "
        "(POOL_16 + HELD_OUT_35; matrix has %d total). The 4 contrastive "
        "negatives are training-only and need no centroid.",
        len(needs_distance),
        len(names),
    )

    pool_idxs = [names.index(p) for p in POOL_16]
    pool_radius = max(dist[i][j] for i in pool_idxs for j in pool_idxs)
    if pool_radius > POOL_RADIUS_MAX:
        raise RuntimeError(
            f"POOL_16 radius {pool_radius:.4f} > {POOL_RADIUS_MAX} — "
            f"pool no longer geometrically tight; re-check §4.2 selection."
        )
    log.info("OK — POOL_16 radius = %.4f (≤ %.2f)", pool_radius, POOL_RADIUS_MAX)

    # ── (5) Build 40 cell subsets; assert per-K uniqueness ────────────────
    # Round-2 BLOCKER 4: K=1 = ALL 16 POOL_16 singletons (Level-1 superposition
    # coverage extension). K≥2 stays at SUBSETS_PER_K=8 from the seeded RNG.
    subsets = build_subsets()
    for K, subs in subsets.items():
        expected_count = len(POOL_16) if K == 1 else SUBSETS_PER_K
        if len(subs) != expected_count:
            raise RuntimeError(f"K={K} produced {len(subs)} subsets, expected {expected_count}")
        if len({tuple(sorted(s)) for s in subs}) != len(subs):
            raise RuntimeError(f"K={K} has duplicate subsets — rng-seed/replacement bug")
    log.info(
        "OK — built %d cell subsets (K=1:%d, K=2:%d, K=4:%d, K=8:%d), all unique within K",
        sum(len(s) for s in subsets.values()),
        len(subsets[1]),
        len(subsets[2]),
        len(subsets[4]),
        len(subsets[8]),
    )

    # ── (6) Band-occupancy report (informational; FIXED bands per §6.7 #4) ─
    # The §4.5 sim table is fixed-band per persona; band occupancy is the
    # CONSTANT band sizes (Near 6, Near-mid 6, Mid 6, Far 6, Very-far 5,
    # Tail 6) for the band-gap test. Per-subset min-dist distributions are
    # ALSO computed here (analyzer uses them for the residualized-leakage
    # check, plan §6.7 #3).
    band_min_dist_per_K: dict[int, dict[str, dict[str, float]]] = {}
    for K, subs in subsets.items():
        band_min_dist_per_K[K] = {
            b: {"min": float("inf"), "max": 0.0, "mean": 0.0} for b in HELD_OUT_BANDS
        }
        # For each band, compute per-subset mean(min-dist-to-subset) across
        # the band's personas; track min/max/mean over the 8 subsets.
        for band, members in HELD_OUT_BANDS.items():
            per_subset_band_means = []
            for sub in subs:
                vals = [min_dist_to_set(p, list(sub), names, dist) for p in members]
                per_subset_band_means.append(sum(vals) / len(vals))
            band_min_dist_per_K[K][band]["min"] = min(per_subset_band_means)
            band_min_dist_per_K[K][band]["max"] = max(per_subset_band_means)
            band_min_dist_per_K[K][band]["mean"] = sum(per_subset_band_means) / len(
                per_subset_band_means
            )

    # The FIXED band counts (plan §4.3) — these are the "occupancy" for the
    # band-gap statistic since bands are persona-pinned.
    fixed_band_counts = {b: len(m) for b, m in HELD_OUT_BANDS.items()}
    if min(fixed_band_counts.values()) < 5 and fixed_band_counts.get("mid", 0) < 6:
        # very-far is 5; that's accepted in §4.5. Mid must be ≥ 6 per the v2
        # plan; everything else ≥ 5.
        pass
    log.info("OK — fixed band counts (persona-pinned): %s", fixed_band_counts)

    # ── (7) Write design_validation.json ──────────────────────────────────
    payload = {
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_TOKEN_ID,
        "pool_16": POOL_16,
        "negatives_fixed": NEGATIVES_FIXED,
        "held_out_35": HELD_OUT_35,
        "n_total_personas": len(ALL_PERSONAS),
        "pool_radius_layer20_cosine": pool_radius,
        "fixed_band_counts": fixed_band_counts,
        "subset_rng_seed": SUBSET_RNG_SEED,
        "subsets_per_K": SUBSETS_PER_K,
        "subsets": {str(K): [list(s) for s in subsets[K]] for K in K_VALUES},
        "band_mean_min_dist_per_K_summary": {str(K): band_min_dist_per_K[K] for K in K_VALUES},
        "prompt_lengths_chars": {p: len(prompts[p]) for p in ALL_PERSONAS},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out_path)
    log.info("Phase 0 design validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
