# ruff: noqa: RUF003  # em-dash + ΔG + − intentional
"""Task #504 round-2 v2-slug regression — Phase 2 analyze must iterate v2 slugs.

Pins the contract for the round-2 BLOCKER #1 fix (concern_id
``analyze-v2-slug-iteration``): when the v2 pipeline writes trajectories at
``<slab>/c504v2_<arm>_seed<S>/trajectory.json``, ``build_rows`` and
``aggregate_base_prior_from_trajectories`` must yield NON-ZERO rows when called
with the v2 slug set, and the legacy v1 path must still iterate the v1 slug
set.

CPU-only, sub-second. Constructs synthetic trajectory.json files under a
tmp slab_root and exercises the analyze entrypoints; no GPU/HF/network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    POSITIONED_ARM_SLUGS,
    POSITIONED_ARM_SLUGS_V2,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
    aggregate_base_prior_from_trajectories,
    build_rows,
)


def _synthetic_trajectory(*, source_dg: float, source_emit: float) -> dict:
    """Build a minimal trajectory.json that exercises build_rows + aggregate_*.

    One in-band checkpoint at frac=0.5 with one probe ('probe_p1') and one
    question. The base-prior aggregator reads ``b_logp`` per probe/q; build_rows
    reads ``source_self`` + ``held_out[probe]`` per checkpoint.
    """
    return {
        "checkpoints": [
            {
                "frac": 0.5,
                "step": 12,
                "source_self": {
                    "delta_g_mean": source_dg,
                    "emission_p": source_emit,
                },
                "held_out": {
                    "probe_p1": {
                        "q_0": {"delta_g": 0.42, "b_logp": -7.5},
                        "q_1": {"delta_g": 0.55, "b_logp": -7.3},
                    },
                },
            },
        ],
    }


def _write_slab(slab_root: Path, slugs, seeds, *, source_dg: float = 8.0):
    """Write trajectory.json for every (slug, seed) under slab_root."""
    for slug in slugs:
        for seed in seeds:
            d = slab_root / f"{slug}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "trajectory.json").write_text(
                json.dumps(_synthetic_trajectory(source_dg=source_dg, source_emit=0.4))
            )


# Phase 0.5 per-probe covariates compatible with build_rows. probe_p1 has
# entries for every arm slug we exercise (both v1 and v2).
_PER_PROBE_COVARIATES = {
    "probe_p1": {
        "d_source": 0.50,
        "d_nearest_neg_nd": {
            "c504_near": 0.10,
            "c504_mid_near": 0.20,
            "c504_mid_far": 0.30,
            "c504_far": 0.40,
            "c504v2_near": 0.10,
            "c504v2_mid_near": 0.20,
            "c504v2_mid_far": 0.30,
            "c504v2_far": 0.40,
        },
        "shadow_angle": {
            "c504_near": 0.05,
            "c504_mid_near": 0.15,
            "c504_mid_far": 0.25,
            "c504_far": 0.35,
            "c504v2_near": 0.05,
            "c504v2_mid_near": 0.15,
            "c504v2_mid_far": 0.25,
            "c504v2_far": 0.35,
        },
    },
}

_ARM_TO_POSITIONED_N = {
    "c504_near": "n_near",
    "c504_mid_near": "n_mid_near",
    "c504_mid_far": "n_mid_far",
    "c504_far": "n_far",
    "c504v2_near": "n_near",
    "c504v2_mid_near": "n_mid_near",
    "c504v2_mid_far": "n_mid_far",
    "c504v2_far": "n_far",
}


def test_build_rows_v2_default_iterates_v2_slugs(tmp_path: Path) -> None:
    """build_rows called with POSITIONED_ARM_SLUGS_V2 picks up v2 trajectories."""
    seeds = [42, 137]
    _write_slab(tmp_path, POSITIONED_ARM_SLUGS_V2, seeds)
    out = build_rows(
        slab_root=tmp_path,
        chosen_frac=0.5,
        per_probe=_PER_PROBE_COVARIATES,
        arm_to_positioned_n=_ARM_TO_POSITIONED_N,
        seeds=seeds,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V2,
    )
    # 4 arms × 2 seeds × 1 probe × (1 ck, averaged-q delta_g) = 8 rows.
    assert out["rows"], "v2-slug build_rows produced ZERO rows (BLOCKER #1 regression)"
    assert len(out["rows"]) == 8, f"expected 8 rows, got {len(out['rows'])}"
    # The cell column should ONLY contain v2 slugs.
    cells_in_rows = {r["cell"] for r in out["rows"]}
    assert cells_in_rows == set(POSITIONED_ARM_SLUGS_V2), cells_in_rows
    # Excluded should be empty (every trajectory found + in-band).
    assert not out["excluded_cells"], out["excluded_cells"]


def test_build_rows_v1_legacy_iterates_v1_slugs(tmp_path: Path) -> None:
    """build_rows with the (default) v1 slug set ignores v2 trajectories."""
    seeds = [42]
    # Write ONLY v2 trajectories. The v1 default should produce zero rows.
    _write_slab(tmp_path, POSITIONED_ARM_SLUGS_V2, seeds)
    out = build_rows(
        slab_root=tmp_path,
        chosen_frac=0.5,
        per_probe=_PER_PROBE_COVARIATES,
        arm_to_positioned_n=_ARM_TO_POSITIONED_N,
        seeds=seeds,
        # Explicit v1 (the default) — verifies legacy callers stay v1.
        positioned_arm_slugs=POSITIONED_ARM_SLUGS,
    )
    # Zero rows because v1 slugs don't exist under tmp_path.
    assert not out["rows"], f"v1 slug iteration found unexpected rows: {out['rows']}"
    # Every (v1 cell, seed) gets logged as excluded with trajectory_missing.
    expected_excluded = len(POSITIONED_ARM_SLUGS) * len(seeds)
    assert len(out["excluded_cells"]) == expected_excluded
    assert all(e["reason"] == "trajectory_missing" for e in out["excluded_cells"])


def test_build_rows_default_param_is_v1(tmp_path: Path) -> None:
    """build_rows called WITHOUT positioned_arm_slugs defaults to v1 (legacy)."""
    seeds = [42]
    # Write v1 trajectories.
    _write_slab(tmp_path, POSITIONED_ARM_SLUGS, seeds)
    out = build_rows(
        slab_root=tmp_path,
        chosen_frac=0.5,
        per_probe=_PER_PROBE_COVARIATES,
        arm_to_positioned_n=_ARM_TO_POSITIONED_N,
        seeds=seeds,
        # NO positioned_arm_slugs arg — must default to v1.
    )
    # 4 arms × 1 seed × 1 probe = 4 rows.
    assert len(out["rows"]) == 4, len(out["rows"])
    cells_in_rows = {r["cell"] for r in out["rows"]}
    assert cells_in_rows == set(POSITIONED_ARM_SLUGS), cells_in_rows


def test_aggregate_base_prior_v2_default_finds_v2_trajectories(tmp_path: Path) -> None:
    """aggregate_base_prior_from_trajectories with v2 slugs picks up b_logp."""
    seeds = [42, 137]
    _write_slab(tmp_path, POSITIONED_ARM_SLUGS_V2, seeds)
    base_prior = aggregate_base_prior_from_trajectories(
        slab_root=tmp_path,
        seeds=seeds,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V2,
    )
    # probe_p1 was written with b_logp values; aggregation should land it.
    assert "probe_p1" in base_prior, base_prior
    # 4 arms × 2 seeds × 1 ck × 2 q = 16 readings. The mean is (-7.5 + -7.3)/2 = -7.4
    # since all 16 are pairs of (-7.5, -7.3).
    assert base_prior["probe_p1"] == pytest.approx(-7.4)


def test_aggregate_base_prior_default_param_is_v1(tmp_path: Path) -> None:
    """aggregate_base_prior_from_trajectories WITHOUT slug arg defaults to v1."""
    seeds = [42]
    # Write ONLY v2 trajectories.
    _write_slab(tmp_path, POSITIONED_ARM_SLUGS_V2, seeds)
    base_prior = aggregate_base_prior_from_trajectories(
        slab_root=tmp_path,
        seeds=seeds,
        # NO positioned_arm_slugs arg → defaults to v1 → finds nothing.
    )
    # Empty: v1 slugs don't exist on disk.
    assert base_prior == {}, base_prior


# ── Concern A regression: picker tie-break target = 8.0 (NOT 8.5). ─────────


def _smoke_trajectory_at_dg(*, source_dg: float, source_emit: float = 0.4) -> dict:
    """Build a Phase 0 lr-smoke trajectory with a checkpoint at frac=0.5."""
    return {
        "checkpoints": [
            {
                "frac": 0.5,
                "step": 12,
                "source_self": {
                    "delta_g_mean": source_dg,
                    "emission_p": source_emit,
                },
                "held_out": {},
            },
        ],
    }


def test_picker_tie_break_targets_8_0_nats() -> None:
    """Concern A: at a tie on latest frac, the picker picks the cell closer to
    source_ΔG = 8.0 nats (NOT the band midpoint 8.5)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        PHASE0_SMOKE_SLUGS_V2,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        pick_anchor_from_lr_smoke,
    )

    # Build smoke trajectories that tie on latest in-band fraction (both at
    # 0.5) but differ in source_dg: one at 7.9, one at 8.4. Plan v2 §4.1
    # step 3(b) says pick the one closest to 8.0 → 7.9. The pre-fix code
    # used midpoint 8.5 → would have picked 8.4.
    slug_a, slug_b, slug_c = PHASE0_SMOKE_SLUGS_V2
    # slug_a = lr1e5, slug_b = lr3e5, slug_c = lr1e4. Make slug_a and slug_b
    # both in-band at frac=0.5 with the contested dg values.
    smoke_trajs = {
        slug_a: _smoke_trajectory_at_dg(source_dg=7.9),
        slug_b: _smoke_trajectory_at_dg(source_dg=8.4),
        # slug_c: out-of-band (below floor) so the in-band candidate set is
        # just {slug_a, slug_b}.
        slug_c: _smoke_trajectory_at_dg(source_dg=2.0),
    }

    pick = pick_anchor_from_lr_smoke(smoke_trajs, source="villain")
    assert pick["verdict"] == "pass", pick
    # The 7.9 trajectory (slug_a = lr1e5) MUST be picked. Distance to 8.0:
    # |7.9 - 8.0| = 0.1 vs |8.4 - 8.0| = 0.4.
    # Under the broken midpoint-8.5 rule, 8.4 would have won (|8.4 - 8.5| = 0.1
    # vs |7.9 - 8.5| = 0.6).
    assert pick["source_delta_g_at_pick_nats"] == pytest.approx(7.9), pick
    # slug_a maps to lr1e-5 (PHASE0_SMOKE_SLUGS_V2[0]).
    assert pick["chosen_lr"] == pytest.approx(1e-5), pick
