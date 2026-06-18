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


# --------------------------------------------------------------------------
# Round-2 regression tests (one per round-1 code-review blocker/concern)
# --------------------------------------------------------------------------


def test_emnc_cids_match_537_methodology_and_hub():
    # BLOCKER emnc-context-registry-mismatch: the 4 EMNC train contexts must match
    # docs/methodology/issue_537.md §1.4 exactly (default / fmt_code / sp_swe /
    # wc_short_advice) — the contexts the positives-only Betley EM arm trained on.
    # Wrong ids stage non-existent adapter subfolders and skip the real cells.
    assert set(m.EMNC_CIDS) == {"default", "fmt_code", "sp_swe", "wc_short_advice"}
    # The wrong round-1 ids must NOT survive.
    assert "sp_doctor" not in m.EMNC_CIDS
    assert "binst_em" not in m.EMNC_CIDS
    assert "wc_long_write" not in m.EMNC_CIDS
    # cids_for("emnc") returns exactly these (and resolves to the nested layout).
    assert m.cids_for("emnc") == list(m.EMNC_CIDS)
    for cid in m.EMNC_CIDS:
        assert m.resolve_adapter_subfolder("emnc", cid, 42) == (
            f"adapters/i537_emnc_{cid}_seed42/sft_em_adapter"
        )


def test_canary_reference_variant_matches_read_variant():
    # BLOCKER gate7a-variant-mismatch: the canary's GATE_7A_VARIANT must equal the
    # `variant` field of REF_JSON, else a correctly-applied adapter spuriously
    # HALTs the sweep (the asserted numbers belong to a different read).
    import json
    from pathlib import Path

    import scripts.issue651_canary as canary

    repo_root = Path(
        __import__("subprocess")
        .check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    ref = json.loads((repo_root / canary.REF_JSON).read_text())
    assert ref.get("variant") == canary.GATE_7A_VARIANT, (
        canary.REF_JSON,
        ref.get("variant"),
        canary.GATE_7A_VARIANT,
    )
    # And the asserted numbers are the SAME-variant numbers (0.32465 / 0.58711),
    # not the base-variant ones (0.44880 / 0.92869).
    assert abs(float(ref["s_top1_frac"]) - 0.32465) < 1e-3
    assert abs(float(ref["mean_cos_to_U1"]) - 0.58711) < 1e-3


def test_q2_ceiling_is_cross_seed_u1_not_per_cell_median():
    # BLOCKER q2-ceiling-wrong-geometric-object: Q2 must normalize by the
    # per-behavior CROSS-SEED U1 cosine, NOT the Q1 per-cell-seed-ceiling median.
    # Construct a fixture where the two objects DIFFER, and assert q2 uses the
    # cross-seed-U1 object.
    rng = np.random.default_rng(11)
    H = 24
    direction = rng.standard_normal(H).astype(np.float32)
    direction /= np.linalg.norm(direction)

    # Per-context cell reads at seed 42 and 1042: each context = direction + a
    # context-and-seed-specific noise vector of MODERATE magnitude. Per-cell
    # cross-seed cosine (Q1 ceiling) sees independent noise on the two seeds -> LOW;
    # but averaging 8 contexts cancels the noise so the cross-context U1 at each
    # seed concentrates on `direction` -> cross-seed U1 cosine (Q2 ceiling) HIGH.
    # The two objects are numerically far apart here — that is the whole point.
    def ctx_reads(seed_rng, noise=0.55):
        return {
            f"ctx{i}": (direction + noise * seed_rng.standard_normal(H)).astype(np.float32)
            for i in range(16)
        }

    r42 = ctx_reads(np.random.default_rng(100))
    r1042 = ctx_reads(np.random.default_rng(200))
    q1_ceiling = a.seed_ceiling_per_cell(r42, r1042)["median"]
    u1_42 = np.asarray(a.q1_context_invariance(r42, n_reps=50)["U1"], dtype=np.float32)
    u1_1042 = np.asarray(a.q1_context_invariance(r1042, n_reps=50)["U1"], dtype=np.float32)
    q2_ceiling = a.q2_seed_ceiling_per_behavior({"em": {42: u1_42, 1042: u1_1042}})["em"]
    # The two objects must genuinely differ (the fixture is engineered for this) —
    # confirming the driver MUST pass the cross-seed-U1 object, not the per-cell
    # median, as the Q2 normalization denominator (BLOCKER q2-ceiling-wrong-object).
    assert q2_ceiling - q1_ceiling > 0.2, (q2_ceiling, q1_ceiling)
    # The Q2 ceiling (cross-seed U1) should be the HIGH one (shared direction).
    assert q2_ceiling > q1_ceiling
    # q2_seed_ceiling_per_behavior omits single-seed behaviors.
    assert a.q2_seed_ceiling_per_behavior({"refusal": {42: u1_42}}) == {}


def test_q2_distinct_verdict_requires_within_null_band():
    # CONCERN q2-verdict-null-band-ignored: an off-diagonal below the ceiling-
    # fraction bar but OUTSIDE its null band must NOT yield "distinct".
    # Build a q2 dict by hand where every ceiling-fraction < 0.5 but one pair's
    # raw cosine exceeds its null p95.
    behaviors = ["em", "fact", "marker"]
    # raw cosines: em|fact = 0.30 (real shared, will exceed its null), others ~0.
    raw = [
        [1.0, 0.30, 0.02],
        [0.30, 1.0, 0.03],
        [0.02, 0.03, 1.0],
    ]
    # ceiling-normalized all < 0.5 (denominators ~0.8 -> 0.30/0.8=0.375 < 0.5).
    cf = [
        [1.0, 0.375, 0.025],
        [0.375, 1.0, 0.0375],
        [0.025, 0.0375, 1.0],
    ]
    # Per-pair null p95: em|fact's null band (0.10) is BELOW its raw cos (0.30) ->
    # OUTSIDE band; the other pairs' raw cos < their null p95 -> within band.
    pair_null = {"em|fact": 0.10, "em|marker": 0.20, "fact|marker": 0.20}
    q2 = {
        "behaviors": behaviors,
        "raw_cosine_matrix": raw,
        "ceiling_normalized_matrix": cf,
        "pairwise_null_p95": pair_null,
    }
    verdict = a.q2_verdict(q2, distinct_frac=0.5)
    # All ceiling-fractions < 0.5, but em|fact is OUTSIDE its null band -> NOT distinct.
    assert verdict["verdict"] != "distinct", verdict
    em_fact = next(
        d for d in verdict["off_diagonal_ceiling_fractions"] if set(d["pair"]) == {"em", "fact"}
    )
    assert em_fact["below_ceiling_bar"] is True
    assert em_fact["within_null_band"] is False
    # Now drop em|fact's raw cos within its null band -> distinct.
    raw2 = [r[:] for r in raw]
    raw2[0][1] = raw2[1][0] = 0.05  # below null p95 0.10
    q2b = dict(q2, raw_cosine_matrix=raw2)
    verdict_b = a.q2_verdict(q2b, distinct_frac=0.5)
    assert verdict_b["verdict"] == "distinct", verdict_b


def test_q2_matrix_carries_per_pair_null_band():
    # The matrix builder must expose a per-pair cosine-scale null band so the
    # verdict can gate on it (CONCERN q2-verdict-null-band-ignored).
    rng = np.random.default_rng(7)
    H = 48
    behavior_u1 = {b: rng.standard_normal(H).astype(np.float32) for b in ("em", "fact", "marker")}
    ceilings = {b: 0.8 for b in behavior_u1}
    q2 = a.q2_cross_behavior_matrix(behavior_u1, ceilings, n_reps=80)
    assert "pairwise_null_p95" in q2
    # One entry per unordered pair (3 behaviors -> 3 pairs).
    assert len(q2["pairwise_null_p95"]) == 3
    assert all(0.0 <= v <= 1.0 for v in q2["pairwise_null_p95"].values())


# --------------------------------------------------------------------------
# Round-3 regression tests (canary adapter-identity root cause)
# --------------------------------------------------------------------------


def test_canary_adapter_is_the_provenance_recorded_producer():
    # ROOT CAUSE round-3 (#651): the canary loaded the WRONG adapter. The v3 plan
    # named adapters/marker_villain_asst_excluded_medium_c0589c_seed42 (r=32/a=64),
    # but #521's same_marker_seed42.json was produced by issue_519/marker_seed42
    # (r=8/a=16) per v2_adapter_provenance.json. A different-rank adapter has a
    # different shift direction -> orthogonal U1 (cos 0.0096) -> spurious HALT.
    import json
    from pathlib import Path

    import scripts.issue651_canary as canary

    repo_root = Path(
        __import__("subprocess")
        .check_output(["git", "rev-parse", "--show-toplevel"])
        .decode()
        .strip()
    )
    # The canary must point at the producer the provenance file records, NOT the
    # c0589c adapter the plan misnamed.
    assert canary.CANARY_ADAPTER == "issue_519/marker_seed42"
    assert "c0589c" not in canary.CANARY_ADAPTER

    prov = json.loads((repo_root / "eval_results/issue_521/v2_adapter_provenance.json").read_text())
    recorded = prov["marker_seeds"]["42"]
    # The provenance string names the producing adapter subfolder.
    assert canary.CANARY_ADAPTER in recorded, (canary.CANARY_ADAPTER, recorded)


def test_assert_adapter_regime_rejects_wrong_rank(tmp_path):
    # The pre-SVD regime assert must reject the c0589c regime (r=32/a=64) and
    # accept the reference regime (r=8/a=16) — so a wrong-adapter drift fails
    # loud BEFORE the expensive SVD read instead of producing an orthogonal U1.
    import json

    import scripts.issue651_canary as canary

    def _write_cfg(d, r, alpha, use_rslora):
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text(
            json.dumps({"r": r, "lora_alpha": alpha, "use_rslora": use_rslora})
        )
        return d

    # Reference regime -> passes.
    good = _write_cfg(tmp_path / "good", 8, 16, True)
    canary._assert_adapter_regime(good)  # no raise

    # The wrong (c0589c) regime -> raises.
    bad = _write_cfg(tmp_path / "bad", 32, 64, True)
    with pytest.raises(AssertionError, match="regime mismatch"):
        canary._assert_adapter_regime(bad)

    # use_rslora=False (rsLoRA gauge off) -> raises (incident #601 probe).
    no_rslora = _write_cfg(tmp_path / "norslora", 8, 16, False)
    with pytest.raises(AssertionError, match="regime mismatch"):
        canary._assert_adapter_regime(no_rslora)


def test_canary_regime_constants_match_reference_regime():
    # The pinned regime constants must equal the reference adapter's actual
    # config (r=8/alpha=16/use_rslora=True). If issue_519/marker_seed42 is ever
    # re-trained these constants must move in lockstep with the reference JSON.
    import scripts.issue651_canary as canary

    assert canary.REF_ADAPTER_R == 8
    assert canary.REF_ADAPTER_ALPHA == 16
    assert canary.REF_ADAPTER_USE_RSLORA is True


def test_bridge_reads_every_seed42_cell_not_just_one(monkeypatch, tmp_path):
    # CONCERN bridge-single-canonical-cell: the per-behavior canonical U1 must be
    # built from EVERY seed-42 cell's read (plan §9: 16 fact + 16 sycophancy), not
    # one arbitrary cell. Spy on _extract_canonical_u1 to count the calls; feed
    # real stub shift tensors so the neutral-panel side runs unmocked.
    import json
    import sys

    import torch

    import scripts.issue651_bridge as bridge

    H = 32
    rng = np.random.default_rng(3)
    calls: list[tuple[str, str, int]] = []

    def fake_extract(behavior, cid, seed, **kw):
        calls.append((behavior, cid, seed))
        return rng.standard_normal(H).astype(np.float32)

    # _extract_canonical_u1 is a module-level symbol -> patch it directly.
    monkeypatch.setattr(bridge, "_extract_canonical_u1", fake_extract)
    # Point _repo_root at a tmp dir and pre-create the per-cell shift tensors so
    # the neutral side (real torch.load + real svd) finds them.
    monkeypatch.setattr(bridge, "_repo_root", lambda: tmp_path)
    shift_dir = tmp_path / "eval_results" / "issue_651" / "shifts"
    shift_dir.mkdir(parents=True)
    fact_cells = ["fact_default_seed42", "fact_sp_swe_seed42", "fact_fmt_code_seed42"]
    order = m.panel_column_order()
    for cell_id in fact_cells:
        stub_shifts = {
            p: {"delta_v": torch.tensor(rng.standard_normal(H), dtype=torch.float32)} for p in order
        }
        torch.save({"shifts": stub_shifts}, shift_dir / f"{cell_id}.pt")

    monkeypatch.setattr(sys, "argv", ["issue651_bridge.py", "--cells", *fact_cells, "--cpu-only"])
    rc = bridge.main()
    assert rc == 0
    # _extract_canonical_u1 called ONCE PER seed-42 fact cell (not just one).
    assert len(calls) == len(fact_cells), calls
    assert {c[1] for c in calls} == {"default", "sp_swe", "fmt_code"}
    # The written bridge JSON records all canonical cells.
    out = json.loads((tmp_path / "eval_results/issue_651/construct_bridge/fact.json").read_text())
    assert out["n_canonical_cells"] == len(fact_cells)
    assert set(out["canonical_cells"]) == set(fact_cells)
    assert "cos_neutral_vs_canonical" in out
