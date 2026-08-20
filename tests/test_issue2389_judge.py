"""Issue #2389 judge fork — fork-delta unit tests.

Covers exactly the deltas vs the parent ``issue2329_judge.py`` (the shared
machinery is the parent's, exercised by its own suite):

- identity constants + the vllm-parity phase registration/staging plan;
- P6 forced-batch wave routing (``wave_threshold_base`` == 0) + the #2152
  pilot wave declarations (source pins);
- gate-3 catastrophic HALT demoted to advisory (``passed`` always True,
  ``phase_separation_gate`` never returns the halt rc);
- the parity verdict (Wilcoxon / offset / survival clauses, filler_swap's
  coherence-only coverage, rule-9 drop handling);
- the E-N1 sync route read-back assert;
- the per-cell-cap adaptation of the gate-3 capregen staleness check.
"""

from __future__ import annotations

import functools
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_judge as J  # noqa: E402

BANK = J.BANK
J94 = J.J94

DRAWS = (0, 1)


@functools.lru_cache(maxsize=1)
def _pairs():
    return BANK.build_pairs()


def _select_disjoint(cell: str, n: int):
    """First ``n`` pairs of ``cell`` with mutually disjoint contexts (so the
    synthetic per-(ctx, draw, rid) score writes below can never collide)."""
    picked, used = [], set()
    for p in _pairs():
        if p.cell != cell or p.a in used or p.b in used:
            continue
        picked.append(p)
        used.update((p.a, p.b))
        if len(picked) == n:
            return picked
    raise AssertionError((cell, n, len(picked)))


def _rows_for(pairs_sel) -> list[dict]:
    seen, rows = set(), []
    for p in pairs_sel:
        for ctx in (p.a, p.b):
            for d in DRAWS:
                if (ctx, d) in seen:
                    continue
                seen.add((ctx, d))
                rows.append({"context_id": ctx, "draw": d})
    return rows


def _scores_for(
    pairs_sel,
    *,
    coh: float = 80.0,
    floor_ab: tuple[float, float] = (80.0, 20.0),
    ceil_ab: tuple[float, float] = (20.0, 80.0),
    coh_shift: float = 0.0,
    behavior_override: dict | None = None,
) -> dict[str, float]:
    """Synthetic score dict for one engine side.

    Default behavior scores give Delta(floor) = -60 pts / Delta(ceiling) =
    +60 pts => sep = 1.2 (survives the 0.5 bar). ``behavior_override`` maps
    pair_id -> (floor_ab, ceil_ab) for per-pair overrides.
    """
    s: dict[str, float] = {}
    for p in pairs_sel:
        for ctx in (p.a, p.b):
            for d in DRAWS:
                s[J94._item_id("c", f"a|{ctx}|{d}")] = coh + coh_shift
        cores = J.pair_rubric_cores(p)
        if cores is None:
            continue
        rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
        fab, cab = (behavior_override or {}).get(p.pair_id, (floor_ab, ceil_ab))
        for d in DRAWS:
            s[J.anchor_unit_id(p.a, d, rid_a)] = fab[0]
            s[J.anchor_unit_id(p.a, d, rid_b)] = fab[1]
            s[J.anchor_unit_id(p.b, d, rid_a)] = cab[0]
            s[J.anchor_unit_id(p.b, d, rid_b)] = cab[1]
    return s


@functools.lru_cache(maxsize=1)
def _parity_subset():
    return (
        _select_disjoint("fact_user_name", 3)
        + _select_disjoint("persona_prompted", 3)
        + _select_disjoint("filler_swap", 2)
    )


# ── identity + registration ──────────────────────────────────────────


def test_fork_identity_constants():
    assert J.HF_PREFIX == "issue2389_q38ce"
    assert J.PILOT_SEED == 2389
    assert "issue_2389" in str(J._DEFAULT_IN_ROOT)
    assert "issue_2389" in str(J._DEFAULT_BREACH_BASIS)
    assert J.DATASET_REPO.endswith("explore-persona-space-data") or J.DATASET_REPO


def test_vllm_parity_phase_registered():
    assert "vllm-parity" in J.PHASES
    assert J.PHASES["vllm-parity"] is J.phase_vllm_parity
    plan = J._PHASE_STAGE_PLAN["vllm-parity"]
    assert plan["anchors_any"] == (J._STAGE_ANCHORS, J._STAGE_ANCHORS_GATE)
    assert plan["optional"] == (J._STAGE_VLLM_PARITY,)
    assert "vllm-parity" in J._BANK_JSON_PHASES


def test_parity_cells_are_bank_cells_and_span_rubric_classes():
    cells = {p.cell for p in _pairs()}
    assert set(J.PARITY_CELLS) <= cells
    rubric_free = [c for c in J.PARITY_CELLS if BANK.base_type_of(c) == "filler_swap"]
    assert rubric_free == ["filler_swap"]


def test_parity_report_remote_matches_vllm_leg_poll_target():
    import issue2389_vllm_anchors as V

    assert J._PARITY_REPORT_REMOTE == V.PARITY_REPORT_REMOTE
    assert tuple(J.PARITY_CELLS) == tuple(V.PARITY_CELLS)


def test_pool_key_matches_run_driver():
    import issue2389_run as R

    for p in _pairs()[:5]:
        assert J.pool_key(p) == R.pool_key(p)


# ── wave routing (S-F1 + #2152 declarations) ─────────────────────────


def _cfg_kwargs(tmp_path: Path) -> dict:
    return dict(
        work_root=tmp_path / "work",
        cache_root=tmp_path / "cache",
        rollouts_dir=tmp_path / "grid",
        anchors_file=tmp_path / "anchors",
        stage2_dir=None,
    )


def test_wave_threshold_base_forced_batch_default(tmp_path):
    cfg = J.JudgeConfig(**_cfg_kwargs(tmp_path))
    assert cfg.wave_threshold_base == 0  # S-F1: P6 bulk waves FORCED BATCH


def test_wave_threshold_base_force_sync_seam(tmp_path):
    cfg = J.JudgeConfig(**_cfg_kwargs(tmp_path), force_sync_routing=True)
    assert cfg.wave_threshold_base == J.FORCE_SYNC_THRESHOLD_BASE


def test_pilot_wave_declarations_present_in_source():
    # #2152/S-N1 pins: the gate-3-pre pilot declares its sync-forced wave; the
    # gate-6 pilot declares the forced-batch P6 wave. Textual regression pins
    # on the exact kwarg the judge_pilot seam keys on.
    assert "wave_force_sync=True" in inspect.getsource(J.phase_pilot_gate3pre)
    assert "wave_threshold_base=0" in inspect.getsource(J.phase_pilot)


# ── gate 3: catastrophic HALT demoted to advisory ─────────────────────


def test_separation_verdict_catastrophic_is_advisory():
    gate_pairs = _select_disjoint("fact_user_name", 2)
    rows = _rows_for(gate_pairs)
    # All rubric scores equal => sep = 0 for every pair => frac 0 < 0.25.
    scores = _scores_for(gate_pairs, floor_ab=(50.0, 50.0), ceil_ab=(50.0, 50.0))
    report = J.separation_verdict(gate_pairs, rows, scores)
    assert report["catastrophic"] is True
    assert report["frac_cells_pass"] == 0.0
    assert report["passed"] is True  # #2389: advisory-only, never halts


def test_phase_separation_gate_source_never_returns_halt_rc():
    src = inspect.getsource(J.phase_separation_gate)
    assert "RC_SEPARATION_GATE" not in src
    assert "return RC_OK" in src


# ── parity verdict ────────────────────────────────────────────────────


def test_parity_verdict_pass_on_identical_engines():
    pairs_sel = _parity_subset()
    rows = _rows_for(pairs_sel)
    s = _scores_for(pairs_sel)
    report = J.parity_verdict(pairs_sel, rows, rows, s, dict(s))
    assert report["verdict"] == "PASS" and report["passed"] is True
    assert all(report["clauses"].values())
    for arm in ("floor", "ceiling", "coherence"):
        st = report["arms"][arm]
        assert st["wilcoxon_p"] == 1.0 and st["mean_offset_pts"] == 0.0
    # filler_swap: rubric-free — covered by the coherence arm, survival vacuous
    fs = report["survival_per_cell"]["filler_swap"]
    assert fs["rubric_bearing"] is False and fs["survival_equal"] is True
    # coherence arm pairs every context of all 3 cells
    assert report["arms"]["coherence"]["n"] == len({r["context_id"] for r in rows})


def test_parity_verdict_fails_on_coherence_offset():
    pairs_sel = _parity_subset()
    rows = _rows_for(pairs_sel)
    hf = _scores_for(pairs_sel)
    vllm = _scores_for(pairs_sel, coh_shift=-10.0)  # hf - vllm = +10 pts > 3
    report = J.parity_verdict(pairs_sel, rows, rows, hf, vllm)
    assert report["verdict"] == "FAIL"
    assert report["clauses"]["offset_all_arms"] is False
    assert report["arms"]["coherence"]["mean_offset_pts"] == pytest.approx(10.0)
    # behavior deltas untouched => floor/ceiling arms still clean
    assert report["arms"]["floor"]["pass_offset"] is True


def test_parity_verdict_fails_on_survival_count_change():
    pairs_sel = _parity_subset()
    rows = _rows_for(pairs_sel)
    hf = _scores_for(pairs_sel)
    # One rubric-bearing pair collapses to sep=0 on the vLLM side only.
    victim = next(p for p in pairs_sel if J.pair_rubric_cores(p) is not None)
    vllm = _scores_for(
        pairs_sel,
        behavior_override={victim.pair_id: (((50.0, 50.0), (50.0, 50.0)))},
    )
    report = J.parity_verdict(pairs_sel, rows, rows, hf, vllm)
    assert report["verdict"] == "FAIL"
    assert report["clauses"]["survival_unchanged"] is False
    rec = report["survival_per_cell"][victim.cell]
    assert rec["hf_survive"] == rec["vllm_survive"] + 1


def test_parity_verdict_drops_are_excluded_never_coerced():
    pairs_sel = _parity_subset()
    rows = _rows_for(pairs_sel)
    hf = _scores_for(pairs_sel)
    vllm = dict(hf)
    # Drop one context's coherence scores on the vLLM side (rule-9 drop).
    ctx = pairs_sel[0].a
    for d in DRAWS:
        vllm[J94._item_id("c", f"a|{ctx}|{d}")] = None
    report = J.parity_verdict(pairs_sel, rows, rows, hf, vllm)
    assert report["n_coherence_ctx_dropped"] == 1
    n_ctx = len({r["context_id"] for r in rows})
    assert report["arms"]["coherence"]["n"] == n_ctx - 1


def test_paired_arm_stats_empty_arm_fails_loud():
    with pytest.raises(AssertionError, match="EMPTY"):
        J._paired_arm_stats([], "floor")


# ── E-N1 route read-back ──────────────────────────────────────────────


def _write_raw(tmp_path: Path, routing, n_submitted: int, n_cached: int = 0) -> Path:
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"routing": routing, "n_submitted": n_submitted, "n_cached": n_cached}))
    return p


def test_assert_sync_routing_accepts_sync(tmp_path):
    p = _write_raw(tmp_path, {"path": "sync"}, 10)
    row = J._assert_sync_routing(p, "coherence.hf")
    assert row["path"] == "sync" and row["n_submitted"] == 10


def test_assert_sync_routing_rejects_batch(tmp_path):
    p = _write_raw(tmp_path, {"path": "batch"}, 10)
    with pytest.raises(RuntimeError, match="!= 'sync'"):
        J._assert_sync_routing(p, "coherence.hf")


def test_assert_sync_routing_rejects_missing_record_with_dispatch(tmp_path):
    p = _write_raw(tmp_path, None, 5)
    with pytest.raises(RuntimeError, match="UNVERIFIABLE"):
        J._assert_sync_routing(p, "coherence.hf")


def test_assert_sync_routing_tolerates_cache_replay(tmp_path):
    p = _write_raw(tmp_path, None, 0, n_cached=42)
    row = J._assert_sync_routing(p, "coherence.hf")
    assert row["path"] == "cache-replay" and row["n_cached"] == 42


# ── vLLM parity shard loader ──────────────────────────────────────────


def _vllm_row(ctx: str, draw: int, engine: str = "vllm") -> dict:
    return {
        "context_id": ctx,
        "cell": "fact_user_name",
        "value_id": "v",
        "carrier": "c",
        "draw": draw,
        "text": "t",
        "engine": engine,
    }


def test_load_vllm_parity_rows_empty_dir(tmp_path):
    assert J.load_vllm_parity_rows(tmp_path) == []


def test_load_vllm_parity_rows_roundtrip_and_guards(tmp_path):
    shard = tmp_path / "vllm_parity_fact_user_name_w0.jsonl"
    shard.write_text(
        "\n".join(json.dumps(_vllm_row("ctxA", d)) for d in (0, 1)) + "\n", encoding="utf-8"
    )
    rows = J.load_vllm_parity_rows(tmp_path)
    assert [r["draw"] for r in rows] == [0, 1]
    # duplicate (context_id, draw) across shards fails loud
    dup = tmp_path / "vllm_parity_fact_user_name_w1.jsonl"
    dup.write_text(json.dumps(_vllm_row("ctxA", 0)) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="duplicate"):
        J.load_vllm_parity_rows(tmp_path)
    dup.unlink()
    # a non-vllm engine row in the vllm mirror fails loud
    bad = tmp_path / "vllm_parity_fact_user_name_w2.jsonl"
    bad.write_text(json.dumps(_vllm_row("ctxB", 0, engine="hf")) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        J.load_vllm_parity_rows(tmp_path)


# ── gate-3 capregen staleness (per-cell cap regime) ───────────────────


def _basis(tmp_path: Path, **over) -> Path:
    rep = {
        "scope": "anchors",
        "postregen": False,
        "partial": False,
        "realized_row_caps": [2048, 4096],
        "breaching_cells": ["cellA"],
        "per_cell": {"cellA": {"realized_caps_by_batch": {"gate": [2048], "rest": [2048]}}},
    }
    rep.update(over)
    p = tmp_path / "basis.json"
    p.write_text(json.dumps(rep), encoding="utf-8")
    return p


def test_capcheck_absent_basis_with_mixed_caps_warn_skips(tmp_path):
    # #2389 delta: per-cell caps make MIXED row caps the DESIGN — the parent's
    # absent-basis mixed-cap RAISE is dropped (warn-skip, no exception).
    rows = [
        {"cell": "x", "max_new_tokens": 2048, "_shard": "s1"},
        {"cell": "y", "max_new_tokens": 4096, "_shard": "s1"},
    ]
    J._assert_gate_rows_capregen_fresh(tmp_path / "missing.json", rows)


def test_capcheck_breach_row_below_per_cell_floor_raises(tmp_path):
    basis = _basis(tmp_path)
    rows = [{"cell": "cellA", "max_new_tokens": 2048, "_shard": "s1"}]
    with pytest.raises(RuntimeError, match="PRE-REGEN"):
        J._assert_gate_rows_capregen_fresh(basis, rows)


def test_capcheck_breach_row_at_per_cell_floor_passes(tmp_path):
    basis = _basis(tmp_path)
    rows = [{"cell": "cellA", "max_new_tokens": 4096, "_shard": "s1"}]
    J._assert_gate_rows_capregen_fresh(basis, rows)


def test_capcheck_mixed_basis_caps_are_legitimate(tmp_path):
    # Two breaching cells at DIFFERENT base caps: each checked at 2x ITS OWN cap.
    basis = _basis(
        tmp_path,
        breaching_cells=["cellA", "cellB"],
        per_cell={
            "cellA": {"realized_caps_by_batch": {"gate": [2048]}},
            "cellB": {"realized_caps_by_batch": {"gate": [4096]}},
        },
    )
    ok = [
        {"cell": "cellA", "max_new_tokens": 4096, "_shard": "s1"},
        {"cell": "cellB", "max_new_tokens": 8192, "_shard": "s1"},
    ]
    J._assert_gate_rows_capregen_fresh(basis, ok)
    bad = [{"cell": "cellB", "max_new_tokens": 4096, "_shard": "s1"}]
    with pytest.raises(RuntimeError, match="cellB"):
        J._assert_gate_rows_capregen_fresh(basis, bad)


def test_capcheck_postregen_and_partial_bases_refused(tmp_path):
    rows = [{"cell": "cellA", "max_new_tokens": 4096, "_shard": "s1"}]
    with pytest.raises(RuntimeError, match="POST-regen"):
        J._assert_gate_rows_capregen_fresh(_basis(tmp_path, postregen=True), rows)
    with pytest.raises(RuntimeError, match="PARTIAL"):
        J._assert_gate_rows_capregen_fresh(_basis(tmp_path, partial=True), rows)
    with pytest.raises(RuntimeError, match="scope"):
        J._assert_gate_rows_capregen_fresh(_basis(tmp_path, scope="grid"), rows)
    with pytest.raises(RuntimeError, match="without"):
        J._assert_gate_rows_capregen_fresh(_basis(tmp_path, per_cell={}), rows)
