"""Unit tests for task #507's compute_grad_accum + module surface.

The load-bearing piece is the runtime grad_accum selection: world_size=4
(4xH200 default) must produce grad_accum=4; world_size=8 (8xH100 supply-
fallback) must produce grad_accum=2. Both paths preserve effective batch
16 = #411 verbatim.
"""

from __future__ import annotations

import pytest


def test_module_constants_present():
    """Cheap import check: surface API matches plan v2 §4.3."""
    from explore_persona_space.experiments import sycophancy_scale_507 as m

    assert m.SOURCE_PERSONAS_507 == (
        "assistant",
        "comedian",
        "kindergarten_teacher",
        "qwen_default",
        "software_engineer",
        "villain",
    )
    assert m.MODEL_ARMS == (
        ("7b", "Qwen/Qwen2.5-7B-Instruct"),
        ("72b", "Qwen/Qwen2.5-72B-Instruct"),
    )
    assert m.LAYER_SET_BY_ARCH["7b"] == (7, 14, 21, 27)
    assert m.LAYER_SET_BY_ARCH["72b"] == (21, 40, 57, 70)
    # Headline = depth-equivalent of #470's published layer 20 on 7B (20/28 ~= 57/80).
    assert m.HEADLINE_LAYER_BY_ARCH["72b"] == 57
    assert m.HEADLINE_LAYER_BY_ARCH["7b"] == 21
    assert m.EXPECTED_EFFECTIVE_BATCH == 16
    assert m.PER_DEVICE_TRAIN_BATCH_72B == 1


# ── Happy path: the two pod-shape defaults from the plan ──


def test_grad_accum_4xh200_default():
    """4xH200 default path: world_size=4 -> grad_accum=4 -> eff batch 4*1*4=16."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=4) == 4


def test_grad_accum_8xh100_supply_fallback():
    """8xH100 supply-fallback path: world_size=8 -> grad_accum=2 -> eff batch 8*1*2=16."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=8) == 2


# ── Additional pod shapes that the planner might fall back to ──


def test_grad_accum_2xh200_or_2xh100_fallback():
    """2-GPU fallback (e.g. Phase 3 predictor pod): grad_accum=8."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=2) == 8


def test_grad_accum_single_gpu_smoke():
    """world_size=1 smoke (debug): grad_accum=16."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=1) == 16


def test_grad_accum_16xh100_max_scale():
    """world_size=16: grad_accum=1 (still preserves eff_batch 16)."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=16) == 1


# ── Error cases (failures-of-#411-parity-contract) ──


def test_grad_accum_world_size_zero_raises():
    """world_size=0 is invalid input — fail-loud, never silently train."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    with pytest.raises(ValueError, match="world_size must be positive"):
        compute_grad_accum(world_size=0)


def test_grad_accum_world_size_negative_raises():
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    with pytest.raises(ValueError, match="world_size must be positive"):
        compute_grad_accum(world_size=-1)


def test_grad_accum_world_size_not_dividing_16_raises():
    """world_size=3 (or any value not dividing 16): #411 parity unreachable."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    with pytest.raises(ValueError, match="does not divide 16"):
        compute_grad_accum(world_size=3)


def test_grad_accum_world_size_5_raises():
    """world_size=5: 16 / (5*1) = 3.2 — not integer, fail."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    with pytest.raises(ValueError, match="does not divide 16"):
        compute_grad_accum(world_size=5)


def test_grad_accum_per_device_batch_zero_raises():
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    with pytest.raises(ValueError, match="per_device_batch must be positive"):
        compute_grad_accum(world_size=4, per_device_batch=0)


def test_grad_accum_per_device_batch_2_with_world_4():
    """per_device_batch=2 + world_size=4 -> grad_accum=2 -> eff batch 4*2*2=16. Valid alt."""
    from explore_persona_space.experiments.sycophancy_scale_507 import compute_grad_accum

    assert compute_grad_accum(world_size=4, per_device_batch=2) == 2


# ── Effective-batch invariant (the load-bearing thing) ──


@pytest.mark.parametrize(
    "world_size,per_device_batch",
    [
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (16, 1),
        (4, 2),
        (4, 4),
        (8, 2),
        (16, 1),
    ],
)
def test_effective_batch_is_always_16(world_size, per_device_batch):
    """Every (world_size, per_device_batch) combo that compute_grad_accum
    accepts must preserve eff_batch == 16 (i.e. #411 verbatim)."""
    from explore_persona_space.experiments.sycophancy_scale_507 import (
        EXPECTED_EFFECTIVE_BATCH,
        compute_grad_accum,
    )

    grad_accum = compute_grad_accum(world_size=world_size, per_device_batch=per_device_batch)
    eff = world_size * per_device_batch * grad_accum
    assert eff == EXPECTED_EFFECTIVE_BATCH


# ── get_world_size_from_env ──


def test_get_world_size_from_env_default(monkeypatch):
    """No WORLD_SIZE in env, torch.distributed not initialized -> fallback to 1."""
    from explore_persona_space.experiments.sycophancy_scale_507 import (
        get_world_size_from_env,
    )

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert get_world_size_from_env() == 1


def test_get_world_size_from_env_reads_var(monkeypatch):
    """WORLD_SIZE=4 in env -> 4 (the 4xH200 default path)."""
    from explore_persona_space.experiments.sycophancy_scale_507 import (
        get_world_size_from_env,
    )

    monkeypatch.setenv("WORLD_SIZE", "4")
    assert get_world_size_from_env() == 4


def test_get_world_size_from_env_reads_8(monkeypatch):
    """WORLD_SIZE=8 in env -> 8 (supply-fallback path)."""
    from explore_persona_space.experiments.sycophancy_scale_507 import (
        get_world_size_from_env,
    )

    monkeypatch.setenv("WORLD_SIZE", "8")
    assert get_world_size_from_env() == 8


# ── analyze_507 helpers (paired bootstrap) ──


def test_paired_bootstrap_zero_diff_overlaps_zero():
    """When |rho_72b| == |rho_7b| per source, the paired CI must overlap zero."""
    import numpy as np

    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _paired_rho_bootstrap,
    )

    per_source_7b = {
        "assistant": {"rho": 0.30},
        "comedian": {"rho": 0.10},
        "kindergarten_teacher": {"rho": 0.20},
        "qwen_default": {"rho": 0.05},
        "software_engineer": {"rho": 0.15},
        "villain": {"rho": 0.25},
    }
    per_source_72b = dict(per_source_7b)  # identical -> diffs all 0
    result = _paired_rho_bootstrap(
        per_source_7b=per_source_7b,
        per_source_72b=per_source_72b,
        n_boot=200,
        rng=np.random.default_rng(0),
    )
    assert result["n_sources_paired"] == 6
    assert result["point_estimate"] == 0.0
    assert result["overlaps_zero"] is True


def test_paired_bootstrap_72b_dominates_does_not_overlap_zero():
    """When 72b's |rho| is uniformly larger than 7b's, the CI shouldn't include 0."""
    import numpy as np

    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _paired_rho_bootstrap,
    )

    per_source_7b = {
        s: {"rho": 0.0}
        for s in (
            "assistant",
            "comedian",
            "kindergarten_teacher",
            "qwen_default",
            "software_engineer",
            "villain",
        )
    }
    per_source_72b = {s: {"rho": 0.5} for s in per_source_7b}
    result = _paired_rho_bootstrap(
        per_source_7b=per_source_7b,
        per_source_72b=per_source_72b,
        n_boot=500,
        rng=np.random.default_rng(0),
    )
    assert result["n_sources_paired"] == 6
    assert result["point_estimate"] == 0.5
    # With every source diff equal to +0.5, the bootstrap CI is degenerate
    # at 0.5 -> doesn't include 0.
    assert result["overlaps_zero"] is False


def test_paired_bootstrap_insufficient_sources():
    """Fewer than 3 paired sources -> structural skip."""
    import numpy as np

    from explore_persona_space.experiments.sycophancy_scale_507.analyze_507 import (
        _paired_rho_bootstrap,
    )

    per_source_7b = {"assistant": {"rho": 0.3}, "comedian": {"rho": 0.1}}
    per_source_72b = {"assistant": {"rho": 0.4}, "comedian": {"rho": 0.2}}
    result = _paired_rho_bootstrap(
        per_source_7b=per_source_7b,
        per_source_72b=per_source_72b,
        n_boot=200,
        rng=np.random.default_rng(0),
    )
    assert result["n_sources_paired"] == 2
    assert result["point_estimate"] is None
    assert result["overlaps_zero"] is True
