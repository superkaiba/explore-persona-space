"""Issue #478 design invariants — pool tightness, panel disjointness, band-occupancy floor.

These tests exercise pure-Python invariants from ``_issue478_common`` so they
run on the dev VM with no GPU. They do NOT require the centroid tensor — the
``test_distance_matrix_loader`` test is skipped if the centroid file is
absent. The other tests are mandatory pre-launch invariants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _issue478_common import (  # noqa: E402
    ALL_PERSONAS,
    ARM_MARKERS,
    COMEDY_FAMILY,
    HELD_OUT_35,
    HELD_OUT_BANDS,
    K_VALUES,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    NEGATIVES_FIXED,
    POOL_16,
    SEEDS,
    SUBSET_RNG_SEED,
    SUBSETS_PER_K,
    assert_pairwise_disjoint_sets,
    band_of,
    load_all_persona_prompts,
)
from issue478_make_cell_specs import build_arm_specs, build_core_specs  # noqa: E402
from issue478_validate_design import build_subsets  # noqa: E402


def test_marker_canonical_id():
    """Marker text + id stay locked to the canonical ※ token (rules/marker-leakage-measurement)."""
    assert MARKER_TEXT == " ※"
    assert MARKER_TOKEN_ID == 83399


def test_three_way_pool_pairwise_disjoint():
    """POOL_16, NEGATIVES_FIXED, HELD_OUT_35 must be pairwise disjoint (BUG-1/BUG-2 guard)."""
    assert_pairwise_disjoint_sets()  # raises on regression


def test_pool_sizes_match_plan_v5():
    assert len(POOL_16) == 16, f"POOL_16 has {len(POOL_16)}, expected 16"
    assert len(NEGATIVES_FIXED) == 4, f"NEGATIVES_FIXED has {len(NEGATIVES_FIXED)}, expected 4"
    assert len(HELD_OUT_35) == 35, f"HELD_OUT_35 has {len(HELD_OUT_35)}, expected 35"
    assert len(ALL_PERSONAS) == 55


def test_held_out_bands_partition_held_out_35():
    """Every held-out persona lives in exactly one band."""
    flat = [p for band in HELD_OUT_BANDS.values() for p in band]
    assert sorted(flat) == sorted(HELD_OUT_35)
    assert len(flat) == len(set(flat)), "HELD_OUT bands have duplicates"


def test_comedy_family_inside_held_out_35():
    """All 9 comedy-family personas live in HELD_OUT_35 (the no-comedy refit drops these)."""
    assert len(COMEDY_FAMILY) == 9
    assert set(COMEDY_FAMILY) <= set(HELD_OUT_35), (
        f"Comedy personas missing from HELD_OUT_35: {set(COMEDY_FAMILY) - set(HELD_OUT_35)}"
    )


def test_arm_markers_are_8_unique_singletoken_ids():
    """Plan v5 §4.9.1: 8 distinct single-token markers; ids match the table."""
    assert len(ARM_MARKERS) == 8
    texts = [t for t, _ in ARM_MARKERS]
    ids = [i for _, i in ARM_MARKERS]
    assert len(set(texts)) == 8, f"duplicate marker texts: {texts}"
    assert len(set(ids)) == 8, f"duplicate marker ids: {ids}"
    assert ARM_MARKERS[0] == (" ※", 83399), "marker_1 must be the canonical ※"


def test_load_all_persona_prompts_covers_all_55():
    """The prompt loader must return a system prompt for every persona in ALL_PERSONAS."""
    prompts = load_all_persona_prompts()
    missing = [p for p in ALL_PERSONAS if p not in prompts]
    assert not missing, f"Missing prompts for {missing}"


def test_assistant_and_helpful_assistant_are_distinct_keys():
    """assumption #20 in plan v5 §12: the two names co-exist with same prompt text."""
    prompts = load_all_persona_prompts()
    assert "assistant" in prompts
    assert "helpful_assistant" in prompts
    # Both are "You are a helpful assistant." per ORIGINAL_20 + run_100_persona_leakage.
    # The point is they are SEPARATE KEYS (HELD_OUT_35 uses one, NEGATIVES_FIXED uses the other).
    assert "assistant" in HELD_OUT_35
    assert "helpful_assistant" in NEGATIVES_FIXED


def test_band_of_returns_correct_band():
    assert band_of("medical_doctor") == "near"
    assert band_of("villain") == "mid"
    assert band_of("comedian") == "very-far"
    assert band_of("nonexistent_persona") is None


def test_build_subsets_deterministic_unique():
    """build_subsets(rng_seed=478) yields 8 unique subsets per K; deterministic."""
    s1 = build_subsets(rng_seed=SUBSET_RNG_SEED)
    s2 = build_subsets(rng_seed=SUBSET_RNG_SEED)
    for K in K_VALUES:
        assert len(s1[K]) == SUBSETS_PER_K, f"K={K} got {len(s1[K])} subsets"
        assert len({tuple(sorted(s)) for s in s1[K]}) == SUBSETS_PER_K, (
            f"K={K} produced duplicate subsets"
        )
        assert s1[K] == s2[K], f"K={K} subsets non-deterministic"


def test_build_core_specs_yields_32_cells_with_id_offsets():
    """Cell ids run K1_c00..c07 / K2_c08..c15 / K4_c16..c23 / K8_c24..c31."""
    core = build_core_specs()
    assert len(core) == 32
    by_K_first_id = {}
    by_K_last_id = {}
    for s in core:
        K = s["K"]
        by_K_first_id.setdefault(K, s["cell_id"])
        by_K_last_id[K] = s["cell_id"]
    assert by_K_first_id[1].startswith("K1_c0")
    # The arm-matched cells (K2_c08/c09/c10, K4_c16/c17/c18) MUST exist.
    all_ids = {s["cell_id"] for s in core}
    for must_exist in ("K2_c08", "K2_c09", "K2_c10", "K4_c16", "K4_c17", "K4_c18"):
        assert must_exist in all_ids, f"Missing core cell {must_exist}"


def test_build_arm_specs_matches_core_source_sets():
    core = build_core_specs()
    core_by_id = {s["cell_id"]: s for s in core}
    arm = build_arm_specs(core)
    assert len(arm) == 6  # 3 K=2 + 3 K=4
    for a in arm:
        matched = core_by_id[a["matched_core_cell"]]
        assert a["positives"] == matched["positives"]
        assert a["K"] == matched["K"]
        assert len(a["marker_assignment"]) == a["K"]
        assert len(a["marker_id_assignment"]) == a["K"]
        # Every source persona gets its OWN marker (no marker reuse within a cell).
        assert len(set(a["marker_id_assignment"].values())) == a["K"]


def test_core_specs_row_totals_match_plan():
    """rows_per_positive * K == 400; rows_per_negative * 4 == 400; total == 800."""
    for s in build_core_specs():
        assert s["rows_per_positive"] * s["K"] == 400, s
        assert s["rows_per_negative"] * 4 == 400, s
        assert s["total_rows"] == 800, s


def test_seeds_match_405_pairwise_comparability():
    """Plan §10 Reproducibility Card: seeds (42, 137) matched to #405."""
    assert SEEDS == (42, 137)


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / "single_token_100_persona"
        / "cosine_distance_matrix_layer20.json"
    ).exists()
    and not (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / "single_token_100_persona"
        / "centroids"
        / "centroids_layer20.pt"
    ).exists(),
    reason="No cached distance matrix and no centroids — extract first",
)
def test_distance_matrix_loader_covers_distance_bearing_personas():
    """The loader must cover every persona that NEEDS a distance: POOL_16
    (training sources) + HELD_OUT_35 (eval targets) = 51. The 4 contrastive
    NEGATIVES are training-only and never distance-measured; the
    conversational-default negatives (helpful_assistant, no_persona) have no
    centroid in the 111-persona pool, which is expected and correct."""
    from _issue478_common import load_cosine_distance_matrix

    names, dist = load_cosine_distance_matrix()
    assert isinstance(names, list) and len(names) >= 51
    needs_distance = list(POOL_16) + list(HELD_OUT_35)
    missing = [p for p in needs_distance if p not in names]
    assert not missing, f"Distance matrix missing distance-bearing personas: {missing}"
    # The diagonal must be ~0 (it is a DISTANCE matrix, not similarity).
    assert dist[0][0] == 0.0, f"diagonal not 0 — matrix is not distance-oriented: {dist[0][0]}"
    assert len(dist) == len(names)
    assert all(len(row) == len(names) for row in dist)


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / "single_token_100_persona"
        / "cosine_distance_matrix_layer20.json"
    ).exists()
    and not (
        Path(__file__).resolve().parent.parent
        / "eval_results"
        / "single_token_100_persona"
        / "centroids"
        / "centroids_layer20.pt"
    ).exists(),
    reason="No cached distance matrix and no centroids — extract first",
)
def test_pool_radius_below_threshold():
    """POOL_16 layer-20 cosine radius must be ≤ 0.05 (plan v5 §4.2: actual 0.0286)."""
    from _issue478_common import POOL_RADIUS_MAX, load_cosine_distance_matrix

    names, dist = load_cosine_distance_matrix()
    pool_idxs = [names.index(p) for p in POOL_16]
    radius = max(dist[i][j] for i in pool_idxs for j in pool_idxs)
    assert radius <= POOL_RADIUS_MAX, (
        f"POOL_16 radius {radius:.4f} > {POOL_RADIUS_MAX} — pool no longer tight."
    )
