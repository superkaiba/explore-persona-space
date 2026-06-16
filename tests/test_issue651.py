"""Unit tests for the #651 cross-behavior-geometry core module + analysis math.

End-to-end behavior tests (no implementation coupling): the panel reproduces
#521's persona_order; the loader branch nests em/emnc only; the cell registry
has the right shapes; the SVD/Q1/Q2/seed-ceiling/variance/bridge math has the
right invariants on synthetic vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.experiments import issue_651 as m
from explore_persona_space.experiments.issue_651 import analysis as a

# --------------------------------------------------------------------------
# Panel + cell registry
# --------------------------------------------------------------------------


def test_panel_reproduces_521_persona_order():
    # The panel column order MUST equal #521's committed persona_order so the
    # Gate 7a canary reproduces same_marker_seed42.json on the SAME object.
    expected = [
        "assistant",
        "biographer",
        "comedian",
        "data_scientist",
        "french_person",
        "kindergarten_teacher",
        "librarian",
        "local_historian",
        "marine_biologist",
        "medical_doctor",
        "police_officer",
        "software_engineer",
        "villain",
        "zelthari_scholar",
    ]
    assert m.panel_column_order() == expected


def test_panel_personas_assistant_maps_to_assistant_prompt():
    personas = m.build_panel_personas()
    assert set(personas) == set(m.panel_column_order())
    # assistant -> the literal ASSISTANT_PROMPT (NOT None) — the #521 choice.
    assert personas["assistant"] == "You are a helpful assistant."
    # every value is a non-empty system-prompt string.
    assert all(isinstance(v, str) and v for v in personas.values())


def test_panel_has_20_questions():
    assert len(m.build_panel_questions()) == 20


def test_em_emnc_nest_under_sft_em_adapter_others_at_root():
    assert m.resolve_adapter_subfolder("em", "default", 42).endswith("/sft_em_adapter")
    assert m.resolve_adapter_subfolder("emnc", "default", 42).endswith("/sft_em_adapter")
    for behavior in ("marker", "fact", "sycophancy", "refusal"):
        sub = m.resolve_adapter_subfolder(behavior, "default", 42)
        assert not sub.endswith("/sft_em_adapter"), (behavior, sub)
        assert sub == f"adapters/i537_{behavior}_default_seed42"


def test_train_cids_for_is_16():
    for behavior in ("marker", "fact", "em", "sycophancy", "refusal"):
        cids = m.train_cids_for(behavior)
        assert len(cids) == 16, behavior
        assert cids[-1] == f"binst_{behavior}"
        assert "default" in cids


def test_retrain_cells_is_32_em_plus_sycophancy_seed1042():
    cells = m.retrain_cells()
    assert len(cells) == 32
    assert {c.behavior for c in cells} == {"em", "sycophancy"}
    assert all(c.seed == 1042 for c in cells)
    assert sum(c.behavior == "em" for c in cells) == 16
    assert sum(c.behavior == "sycophancy" for c in cells) == 16


def test_readable_cells_floor_vs_full():
    full = m.readable_cells(include_seed1042=True)
    floor = m.readable_cells(include_seed1042=False)
    # floor = the 116 existing #537 cells (no seed-1042 em/syc).
    assert len(floor) == 116
    # full adds the 32 retrain cells.
    assert len(full) == 116 + 32
    assert all(c.seed == 42 for c in floor if c.behavior in ("em", "sycophancy"))


def test_parse_cell_spec_handles_multi_underscore_cid():
    c = m.parse_cell_spec("em_wc_long_write_seed1042")
    assert c.behavior == "em"
    assert c.cid == "wc_long_write"
    assert c.seed == 1042
    assert c.adapter_subfolder == "adapters/i537_em_wc_long_write_seed1042/sft_em_adapter"


def test_parse_cell_spec_rejects_bad_specs():
    with pytest.raises(ValueError):
        m.parse_cell_spec("no_seed_here")
    with pytest.raises(ValueError):
        m.parse_cell_spec("bogusbehavior_default_seed42")


# --------------------------------------------------------------------------
# Analysis math (synthetic vectors with known geometry)
# --------------------------------------------------------------------------


def _panel_shifts_from_matrix(M):
    """Build a fake shifts dict {persona: {delta_v: col}} from an (H, 14) matrix."""
    order = m.panel_column_order()
    import torch

    return {p: {"delta_v": torch.tensor(M[:, i], dtype=torch.float32)} for i, p in enumerate(order)}


def test_cell_read_vector_u1_recovers_dominant_direction():
    rng = np.random.default_rng(0)
    H = 64
    direction = rng.standard_normal(H).astype(np.float32)
    direction /= np.linalg.norm(direction)
    # 14 columns all ~ scalar * direction + small noise -> U1 ~ direction.
    M = np.stack(
        [(rng.uniform(0.5, 1.5) * direction + 0.01 * rng.standard_normal(H)) for _ in range(14)],
        axis=1,
    ).astype(np.float32)
    shifts = _panel_shifts_from_matrix(M)
    u1 = a.cell_read_vector(shifts, cell_read="u1")
    assert abs(float(np.dot(u1 / np.linalg.norm(u1), direction))) > 0.99


def test_q1_context_invariant_when_all_contexts_share_a_direction():
    rng = np.random.default_rng(1)
    H = 64
    shared = rng.standard_normal(H).astype(np.float32)
    shared /= np.linalg.norm(shared)
    per_context = {
        f"ctx{i}": (rng.uniform(0.5, 1.5) * shared + 0.02 * rng.standard_normal(H)).astype(
            np.float32
        )
        for i in range(8)
    }
    q1 = a.q1_context_invariance(per_context, n_reps=200)
    # All contexts ~ shared -> high top-share, clears null, high per-context cos.
    assert q1["top_share_clears_null_p95"]
    assert q1["mean_cos_to_U1"] > 0.9
    verdict = a.q1_verdict(q1, seed_ceiling_median=0.95)
    assert verdict["context_invariant"]


def test_q1_context_specific_when_directions_are_random():
    rng = np.random.default_rng(2)
    H = 64
    per_context = {f"ctx{i}": rng.standard_normal(H).astype(np.float32) for i in range(8)}
    q1 = a.q1_context_invariance(per_context, n_reps=200)
    verdict = a.q1_verdict(q1, seed_ceiling_median=0.95)
    # Random directions -> not context-invariant.
    assert not verdict["context_invariant"]


def test_seed_ceiling_is_one_when_seeds_identical():
    rng = np.random.default_rng(3)
    H = 32
    reads = {f"ctx{i}": rng.standard_normal(H).astype(np.float32) for i in range(5)}
    sc = a.seed_ceiling_per_cell(reads, reads)
    assert abs(sc["median"] - 1.0) < 1e-6
    assert sc["n_cells"] == 5


def test_q2_matrix_diagonal_is_one_and_ceiling_normalized():
    rng = np.random.default_rng(4)
    H = 48
    behavior_u1 = {
        b: rng.standard_normal(H).astype(np.float32) for b in ("marker", "fact", "em", "sycophancy")
    }
    ceilings = {b: 0.8 for b in behavior_u1}
    q2 = a.q2_cross_behavior_matrix(behavior_u1, ceilings, n_reps=100)
    raw = np.asarray(q2["raw_cosine_matrix"])
    assert np.allclose(np.diag(raw), 1.0)
    assert q2["behaviors"] == sorted(behavior_u1)
    # random unit vectors in 48-d -> off-diagonals small.
    off = raw[~np.eye(4, dtype=bool)]
    assert off.max() < 0.6


def test_q2_verdict_coincide_on_aligned_directions():
    H = 48
    base = np.ones(H, dtype=np.float32)
    behavior_u1 = {"em": base.copy(), "sycophancy": base.copy() * 2.0}  # identical direction
    ceilings = {"em": 0.8, "sycophancy": 0.8}
    q2 = a.q2_cross_behavior_matrix(behavior_u1, ceilings, n_reps=50)
    verdict = a.q2_verdict(q2)
    # cos = 1.0; ceiling-fraction = 1/0.8 = 1.25 >= 0.85 -> coincide.
    assert verdict["verdict"] == "coincide"


def test_variance_decomposition_fractions_sum_to_one():
    rng = np.random.default_rng(5)
    H = 32
    cell_reads = {
        (b, f"ctx{i}"): rng.standard_normal(H).astype(np.float32)
        for b in ("em", "marker")
        for i in range(4)
    }
    var = a.variance_decomposition(cell_reads)
    total = var["shared_frac"] + var["behavior_frac"] + var["context_frac"]
    assert abs(total - 1.0) < 1e-3


def test_construct_bridge_label_thresholds():
    v = np.ones(16, dtype=np.float32)
    # identical -> cos 1.0 -> behavior-direction.
    b = a.construct_bridge_cosine(v, v, bar=0.5)
    assert b["label"] == "behavior-direction"
    assert b["licenses_behavior_claim"]
    # orthogonal -> cos 0 -> panel-direction.
    w = np.zeros(16, dtype=np.float32)
    w[0] = 1.0
    w2 = np.zeros(16, dtype=np.float32)
    w2[1] = 1.0
    b2 = a.construct_bridge_cosine(w, w2, bar=0.5)
    assert b2["label"] == "panel-direction"
    assert not b2["licenses_behavior_claim"]
