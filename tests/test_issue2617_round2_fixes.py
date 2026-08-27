"""#2617 round-2 regression pins (CPU / torch-free / no-network).

Pins the two round-1 BLOCKER fixes plus the CONCERN-row invariants:

- BLOCKER margin-resume-fingerprint: ``_margin_fp`` is sensitive to every
  margin-output-changing knob (pool size, score thresholds) AND to the judged
  score CONTENT it consumes — while a ``repro`` timestamp change leaves it
  unchanged (a timestamp must never invalidate valid checkpoints);
- BLOCKER phase-reentry-preflight: the model-free completion predicates gate
  on regime-matched sentinels, ``main()`` evaluates them BEFORE the 7B model
  load, and ``phase_judge`` entry-skips before any paid ``judge_graded`` call;
- CONCERN margin-pool-contract: ``_assert_pool_provenance`` trips on a
  tampered opener / missing source, ``_pool_floor_check`` refuses short pools
  without the disclosed waiver;
- CONCERN identity-bias-helper-bypassed: the vectorized ``_loo_identity_bias``
  equals the canonical ``identity_bias_predict`` per held-out row;
- CONCERN bank-trip-tests-missing: the authoring gate + XSTest loader gates
  demonstrably TRIP on duplicate questions, wrong counts, REFUSAL_VALUES
  drift, and a 35-carrier bank.

Adoptable: imports the driver by module path (torch-free module top — the
langow pinned-blob import is deferred and stubbed here), uses tmp_path for
all sentinel/judge files, hits no HF and no git pins.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.issue2617_svmp_run as run


def _stub_langow() -> SimpleNamespace:
    """Minimal stand-in for the lazily-loaded langow module: the fingerprint
    hash, the tolerant JSON reader, and the two constants ``_regime_fp``
    reads. Self-consistent within a test — never network/torch."""

    def _sha16(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _read_json(path):
        p = Path(path)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    return SimpleNamespace(
        _sha16=_sha16, _read_json=_read_json, PIN="stubpin2564", ANCHOR_TEMPERATURE=0.8
    )


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_id="stub-model",
        model_revision="stub-rev",
        tiny=True,
        draws=1,
        gen_batch=2,
        seed_base=0,
        max_new_tokens=8,
        upload=False,
        out_root=tmp_path,
    )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _judge_payload(draw_scores: dict, ts: str = "t0") -> dict:
    return {
        "regime_fp": "jfp",
        "judge_model": "claude-sonnet-4-5-20250929",
        "rubric_sha": "r16",
        "refused_threshold": 80,
        "min_valid_draws": 1,
        "dry_run": False,
        "per_context": {"c1": {"draw_scores": draw_scores}},
        "repro": {"timestamp": ts},
    }


# ── BLOCKER margin-resume-fingerprint ──────────────────────────────────────


def test_margin_fp_sensitive_to_pool_knobs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    base = run._margin_fp(cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "MARGIN_POOL_SIZE", run.MARGIN_POOL_SIZE + 1)
        assert run._margin_fp(cfg) != base, "pool-size knob must change the margin fp"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "MARGIN_REFUSAL_HI", 85.0)
        assert run._margin_fp(cfg) != base, "score-threshold knob must change the margin fp"
    # margin fp is distinct from the plain per-phase regime fp
    assert base != run._regime_fp(cfg, {"phase": "margin"})


def test_margin_fp_tracks_judge_content_not_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    jpath = cfg.out_root / "judge" / "judge_scores.json"
    fp_absent = run._margin_fp(cfg)
    _write_json(jpath, _judge_payload({"0": 90.0}, ts="t0"))
    fp_v1 = run._margin_fp(cfg)
    assert fp_v1 != fp_absent
    # judged-score CONTENT drift invalidates the fp ...
    _write_json(jpath, _judge_payload({"0": 15.0}, ts="t0"))
    assert run._margin_fp(cfg) != fp_v1
    # ... while a repro-timestamp rewrite of identical content does NOT
    _write_json(jpath, _judge_payload({"0": 90.0}, ts="t1-much-later"))
    assert run._margin_fp(cfg) == fp_v1


# ── BLOCKER phase-reentry-preflight ────────────────────────────────────────


def test_judge_completion_predicate_regime_keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    assert not run._judge_complete(cfg)
    _write_json(cfg.out_root / "judge" / "judge_scores.json", _judge_payload({"0": 90.0}))
    _write_json(
        cfg.out_root / "svmp_judge_done.json",
        {"regime_fp": run._regime_fp(cfg, {"phase": "judge"})},
    )
    assert run._judge_complete(cfg)
    _write_json(cfg.out_root / "svmp_judge_done.json", {"regime_fp": "stale-regime"})
    assert not run._judge_complete(cfg)


def test_margin_predicate_invalidated_by_knob_and_judge_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "L", _stub_langow())
    cfg = _cfg(tmp_path)
    _write_json(cfg.out_root / "judge" / "judge_scores.json", _judge_payload({"0": 90.0}))
    pools_obj = {"meta": {}}
    _write_json(cfg.out_root / "margin" / "pools.json", pools_obj)
    # r4 (margin-aggregate-identity-unvalidated): margins.json must carry its
    # own matching regime_fp/pools_sha for completion — identity-doctor legs
    # live in tests/test_issue2617_round4_fixes.py.
    _write_json(
        cfg.out_root / "margin" / "margins.json",
        {"regime_fp": run._margin_fp(cfg), "pools_sha": run._pools_content_sha(pools_obj)},
    )
    _write_json(
        cfg.out_root / "svmp_margin_done.json",
        # r3: the sentinel additionally records the realized pool-content sha
        # (test_issue2617_round3_fixes.py pins the sha-validation semantics).
        {"regime_fp": run._margin_fp(cfg), "pools_sha": run._pools_content_sha(pools_obj)},
    )
    assert run._margin_complete(cfg)
    # a pool knob change on resume must invalidate the margin checkpoint
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "MARGIN_POOL_SIZE", run.MARGIN_POOL_SIZE + 1)
        assert not run._margin_complete(cfg)
    # a judge-score content drift must invalidate the margin checkpoint
    _write_json(cfg.out_root / "judge" / "judge_scores.json", _judge_payload({"0": 15.0}))
    assert not run._margin_complete(cfg)


def test_main_preflight_precedes_model_load():
    """BLOCKER-1 order pin: main() evaluates the model-free completion
    predicates BEFORE load_model_and_tokenizer, and the load is gated on a
    model-needing phase being pending."""
    src = inspect.getsource(run.main)
    i_pre = src.index("PHASE_COMPLETE[p](cfg)")
    i_load = src.index("load_model_and_tokenizer")
    assert i_pre < i_load, "preflight predicates must run before the model load"
    assert 'if any(p in ("gen", "capture", "margin") for p in pending)' in src


def test_phase_complete_registry_covers_phases():
    assert set(run.PHASE_COMPLETE) == set(run.PHASES)


def test_phase_judge_entry_skip_precedes_paid_call():
    src = inspect.getsource(run.phase_judge)
    assert src.index("_judge_complete") < src.index("judge_graded"), (
        "phase_judge must entry-skip on the completion predicate before any judge spend"
    )


# ── CONCERN margin-pool-contract ───────────────────────────────────────────


def _opener(s: str) -> str | None:
    return s[:10] if len(s) >= 10 else None


def test_pool_provenance_assert_trips_on_tamper():
    text_by = {("c1", 0): "A refusal opener text", ("c2", 1): "Sure here is the thing"}
    pool = [
        {"source_context": "c1", "source_draw": 0, "answer": "A refusal "},
        {"source_context": "c2", "source_draw": 1, "answer": "Sure here "},
    ]
    run._assert_pool_provenance(pool, text_by, _opener)  # healthy pool passes
    tampered = [dict(pool[0], answer="tampered!!")]
    with pytest.raises(AssertionError, match="provenance mismatch"):
        run._assert_pool_provenance(tampered, text_by, _opener)
    orphaned = [dict(pool[0], source_context="nope")]
    with pytest.raises(AssertionError, match="source rollout missing"):
        run._assert_pool_provenance(orphaned, text_by, _opener)


def test_pool_floor_check_refuses_short_pools_without_waiver():
    full = run.MARGIN_POOL_SIZE
    assert run._pool_floor_check(full, full, allow_short=False) is False
    with pytest.raises(RuntimeError, match="under the registered size"):
        run._pool_floor_check(full - 1, full, allow_short=False)
    assert run._pool_floor_check(full - 1, full, allow_short=True) is True


# ── CONCERN identity-bias-helper-bypassed ──────────────────────────────────


def test_loo_identity_bias_matches_canonical_helper():
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict
    from scripts.issue2617_svmp_reads import _loo_identity_bias

    rng = np.random.default_rng(2617)
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=(10, 4))
    mask = np.ones(10, dtype=bool)
    mask[3] = False
    x[3] = np.nan
    pred = _loo_identity_bias(x, y, mask)
    idx = np.where(mask)[0]
    for i in idx:
        others = idx[idx != i]
        expected = identity_bias_predict(x[others], y[others], x[i : i + 1])[0]
        np.testing.assert_allclose(pred[i], expected, rtol=1e-10, atol=1e-12)
    assert np.isnan(pred[3]).all(), "non-finite rows stay NaN"


# ── CONCERN bank-trip-tests-missing ────────────────────────────────────────


def test_constructed_authoring_gate_trips_on_duplicate_and_counts(monkeypatch):
    pairs = list(run.CONSTRUCTED_PAIRS)
    # duplicate question (pair 1 replaced by a copy of pair 0)
    dup = pairs.copy()
    dup[1] = dup[0]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "CONSTRUCTED_PAIRS", tuple(dup))
        with pytest.raises(AssertionError, match="duplicate constructed question"):
            run.constructed_authoring_gate()
    # wrong per-class counts (relabel one pair into a sibling class)
    cls0, cat0, qa0, qb0 = pairs[0]
    other = next(c for c in run.CONSTRUCTED_CLASSES if c != cls0)
    relabeled = pairs.copy()
    relabeled[0] = (other, cat0, qa0, qb0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "CONSTRUCTED_PAIRS", tuple(relabeled))
        with pytest.raises(AssertionError):
            run.constructed_authoring_gate()
    # wrong total
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "CONSTRUCTED_PAIRS", tuple(pairs[:-1]))
        with pytest.raises(AssertionError):
            run.constructed_authoring_gate()


def _fake_git_show(values_obj: dict, refusal_literal: str):
    def fake(rel: str) -> bytes:
        if rel == run.XSTEST_VALUES_REL:
            return json.dumps(values_obj).encode("utf-8")
        assert rel == run.XSTEST_BANK_REL, rel
        return f"REFUSAL_VALUES: dict[str, str] = {refusal_literal}\n".encode()

    return fake


def _synthetic_carriers(n: int) -> dict:
    return {
        f"i{k:02d}": {
            "safe_id": f"s{k}",
            "unsafe_id": f"u{k}",
            "prompts": {"v1": "pa", "v2": "pb"},
        }
        for k in range(1, n + 1)
    }


def test_xstest_loader_trips_on_refusal_values_drift_and_carrier_count(monkeypatch):
    good_literal = "{'v1': 'safe', 'v2': 'unsafe'}"
    values36 = {"types": {run.XSTEST_CELL: {"carriers": _synthetic_carriers(36)}}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "_git_show_2215", _fake_git_show(values36, good_literal))
        assert len(run._load_xstest_carriers()) == 36  # healthy bank passes
    # v1/v2 -> safe/unsafe linkage drift trips
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            run, "_git_show_2215", _fake_git_show(values36, "{'v1': 'unsafe', 'v2': 'safe'}")
        )
        with pytest.raises(AssertionError, match="REFUSAL_VALUES drift"):
            run._load_xstest_carriers()
    # a 35-carrier bank trips the count gate
    values35 = {"types": {run.XSTEST_CELL: {"carriers": _synthetic_carriers(35)}}}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run, "_git_show_2215", _fake_git_show(values35, good_literal))
        with pytest.raises(AssertionError, match="expected 36 XSTest carriers"):
            run._load_xstest_carriers()
