"""Round-2 regression pins for the #2254 first-k driver (plan v10, review r2).

Covers the permanent invariants this round added after the r1 FAIL+FAIL
verdict (task #2254, `epm:code-review` v8):

- producer-schema validators (`_validate_gen_record` / `_validate_judged_record`)
  reject truncated / mixed-grain / trace-less records BEFORE judge spend;
- the §3 denominator guard nulls the REGISTERED ratio points (`R`/`R1`/
  `R_span15` -> point None; raw ratio only under the diagnostic key);
- the per-cell validity gate (`_cell_validity`) enforces the rule-29
  completeness floor + coherence;
- the figures module skips validity-gated rows (`_row_valid`) and excludes
  "not-computable pending remediation" lattice blocks (`_lattice_blocks`).

tmp_path-only fixtures; no HF/network reads; no other issue's committed
eval_results (sparse-worktree safe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scripts.issue2254_first_k_steering as fk
import scripts.issue2254_firstk_figures as figs

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _cell() -> dict:
    return {
        "behavior": "evil",
        "direction": "rb",
        "breadth": "single",
        "position": "allans",
        "layer_config": "mid",
        "c": 2.0,
    }


def _gen_rec(cell: dict, cid: str, *, n_q: int = 3, n_draws: int = 2) -> dict:
    return {
        "cell_id": cid,
        "cell": cell,
        "q_of_context": [f"q{i}" for i in range(n_q)],
        "seeds": {
            "42": {
                "completions": [["text"] * n_draws for _ in range(n_q)],
                "edit_traces": [],
            }
        },
        "cap_hit_fraction": 0.0,
        "max_new_tokens": 2048,
    }


def _judged_rec(cell: dict, cid: str, *, n_q: int = 3) -> dict:
    return {
        "cell_id": cid,
        "cell": cell,
        "n_questions": n_q,
        "per_question_mean_score": [50.0] * n_q,
        "per_question_rate": [0.5] * n_q,
        "accounting": {"frac_items_complete": 1.0},
        "coherence_pass": True,
        "coherence_rate": 1.0,
    }


# ---------------------------------------------------------------------------
# producer-schema validators (judge/reduce inputs BEFORE spend)
# ---------------------------------------------------------------------------


def test_validate_gen_record_passes_on_valid() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    fk._validate_gen_record(_gen_rec(cell, cid), Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_wrong_question_grain() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid, n_q=2)  # truncated grain vs the invocation's 3
    with pytest.raises(AssertionError, match="q_of_context grain"):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_wrong_draw_count() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid, n_draws=2)  # mixed-vintage 2-draw record vs 6
    with pytest.raises(AssertionError, match="draws != 6"):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=6)


def test_validate_gen_record_rejects_missing_edit_traces() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    del rec["seeds"]["42"]["edit_traces"]
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_empty_completion() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["seeds"]["42"]["completions"][1][0] = ""  # empty string draw
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_identity_mismatch() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["cell_id"] = "some__other__cell"
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_judged_record_passes_and_rejects_nq_mismatch() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    fk._validate_judged_record(_judged_rec(cell, cid), Path(f"{cid}.json"), n_q=3)
    bad = _judged_rec(cell, cid, n_q=2)  # judged at a different grain
    with pytest.raises(AssertionError):
        fk._validate_judged_record(bad, Path(f"{cid}.json"), n_q=3)


# ---------------------------------------------------------------------------
# §3 denominator guard: registered ratio points are None under the guard
# ---------------------------------------------------------------------------


def _lattice_inputs(a_off: float) -> tuple[dict, np.ndarray, dict]:
    nq = 20
    a0_q = np.linspace(40.0, 60.0, nq)
    arm_q = {
        "lastctx": a0_q + 0.1,
        "tok1": a0_q + 0.5,
        "span13": a0_q + 1.0,
        "span15": a0_q + 1.2,
        "allans": a0_q + a_off,
    }
    deg_q = {"allans": np.zeros(nq), "span13": np.zeros(nq)}
    return arm_q, a0_q, deg_q


def test_lattice_block_guard_nulls_registered_ratio_points() -> None:
    # Constant +2 all-answer delta: every resample |A_b| = 2 < the 5-point
    # floor -> unstable_frac = 1.0 -> ratio_unstable. Registered points None;
    # raw ratio only under the diagnostic key; verdict routes to Ambiguous.
    arm_q, a0_q, deg_q = _lattice_inputs(a_off=2.0)
    blk = fk._lattice_block("evil", "rb", "single", arm_q, a0_q, deg_q, "t-guard")
    assert blk["ratio_guard"]["ratio_unstable"] is True
    assert blk["R"]["point"] is None and blk["R"]["lo"] is None and blk["R"]["hi"] is None
    assert blk["R1"]["point"] is None
    assert blk["R_span15"]["point"] is None
    assert blk["R"]["raw_ratio_diagnostic_not_registered"] == pytest.approx(0.5)
    assert blk["verdict"] == "Ambiguous"
    # Descriptive fallback S - (2/3) A stays load-bearing under the guard.
    assert blk["fallback_S_minus_two_thirds_A"]["point"] == pytest.approx(1.0 - 2.0 * 2.0 / 3.0)


def test_lattice_block_unguarded_keeps_numeric_ratio_points() -> None:
    arm_q, a0_q, deg_q = _lattice_inputs(a_off=30.0)  # far above the 5-point floor
    blk = fk._lattice_block("evil", "rb", "single", arm_q, a0_q, deg_q, "t-clear")
    assert blk["ratio_guard"]["ratio_unstable"] is False
    assert blk["R"]["point"] == pytest.approx(1.0 / 30.0)
    assert blk["R"]["lo"] is not None and blk["R"]["hi"] is not None
    assert blk["R_span15"]["point"] == pytest.approx(1.2 / 30.0)


# ---------------------------------------------------------------------------
# validity gate + figures-side filtering
# ---------------------------------------------------------------------------


def test_cell_validity_floor_and_coherence() -> None:
    ok = fk._cell_validity({"accounting": {"frac_items_complete": 0.96}, "coherence_pass": True})
    assert ok["valid"] is True and ok["completeness_pass"] is True
    low = fk._cell_validity({"accounting": {"frac_items_complete": 0.90}, "coherence_pass": True})
    assert low["valid"] is False and low["completeness_pass"] is False
    none_fc = fk._cell_validity(
        {"accounting": {"frac_items_complete": None}, "coherence_pass": True}
    )
    assert none_fc["valid"] is False
    incoh = fk._cell_validity({"accounting": {"frac_items_complete": 1.0}, "coherence_pass": False})
    assert incoh["valid"] is False and incoh["coherence_pass"] is False


def test_figures_row_valid_semantics() -> None:
    assert figs._row_valid(None) is False
    assert figs._row_valid({"validity": {"valid": False}}) is False
    assert figs._row_valid({"validity": {"valid": True}}) is True
    # Legacy rows lacking the block stay plottable (treated valid).
    assert figs._row_valid({"delta_score": 1.0}) is True


def test_lattice_blocks_exclude_not_computable_variants() -> None:
    good = {"verdict": "Ambiguous", "R": {"point": 0.5, "lo": 0.1, "hi": 0.9}}
    lat = {
        "lattice": {
            "a": {"verdict": "not-computable pending remediation", "invalid_arms": ["x"]},
            "b": {"verdict": "not-computable", "note": "core arm missing"},
            "c": good,
        }
    }
    assert figs._lattice_blocks(lat) == [good]
