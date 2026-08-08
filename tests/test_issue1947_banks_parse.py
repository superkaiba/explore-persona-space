"""#1947 crash-fix r4 pins: per-call parse tolerance in the banks phase.

The P0 crash: one over-count generation (41 questions vs the parent #1090
parser's exact-40 contract) killed the whole workload at
`issue1947_datagen.py::_extend_bank` -> `qg.parse_generation`. The fix wraps
the CALL SITE only (`_parse_generation_tolerant`): over-count generations are
clipped to the first N_QUESTIONS (re-validated through the UNMODIFIED parent
parser), any other parse failure is a counted per-call reject absorbed by the
retry-tranche loop, and the bank floor still fails loud.

Tests execute the REAL `_extend_bank` production body (mock_gen seam only —
the same seam production smoke mode uses); bank/asset files are read
programmatically, never printed (digest-only bank discipline).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_questiongen as qg  # noqa: E402
import issue1947_datagen as dg  # noqa: E402


def _payload(n_questions: int, tag: str) -> str:
    """A valid-shaped questiongen JSON payload with n_questions unique neutral
    questions (mirrors dg._mock_template_response; never trait-lexicon text)."""
    return json.dumps(
        {
            "instruction": [{"pos": "exhibit", "neg": "avoid"} for _ in range(qg.N_PAIRS)],
            "questions": [f"Mock parse question {tag} n{i:02d}?" for i in range(n_questions)],
            "eval_prompt": "mock eval prompt",
        }
    )


# ── unit: _parse_generation_tolerant ─────────────────────────────────────────


def test_exact_count_passthrough():
    gen = dg._parse_generation_tolerant(_payload(qg.N_QUESTIONS, "t0"), "sycophancy")
    assert gen is not None and len(gen["questions"]) == qg.N_QUESTIONS


def test_over_count_clipped_and_logged(caplog):
    with caplog.at_level(logging.INFO, logger=dg.logger.name):
        gen = dg._parse_generation_tolerant(_payload(qg.N_QUESTIONS + 1, "t1"), "sycophancy")
    assert gen is not None
    assert len(gen["questions"]) == qg.N_QUESTIONS
    assert gen["questions"][0] == "Mock parse question t1 n00?"
    assert f"clipped over-count generation: {qg.N_QUESTIONS + 1} -> {qg.N_QUESTIONS}" in caplog.text


def test_under_count_rejected_no_crash(caplog):
    with caplog.at_level(logging.WARNING, logger=dg.logger.name):
        gen = dg._parse_generation_tolerant(_payload(qg.N_QUESTIONS - 1, "t2"), "sycophancy")
    assert gen is None
    assert "rejected generation call" in caplog.text


def test_malformed_json_rejected_no_crash():
    assert dg._parse_generation_tolerant("{not json", "sycophancy") is None


def test_non_dict_json_rejected_no_crash(caplog):
    """Valid JSON whose top level is NOT a dict (bare array / string / null)
    raises AttributeError in the parent parser (obj.get on a non-dict); the
    wrapper must count it as a per-call reject, never crash the banks phase
    (concern parse-tolerance-nondict-json-escape)."""
    for raw in ("[1, 2, 3]", '"just a string"', "null"):
        with caplog.at_level(logging.WARNING, logger=dg.logger.name):
            gen = dg._parse_generation_tolerant(raw, "sycophancy")
        assert gen is None, f"non-dict payload {raw!r} escaped the tolerance wrapper"
    assert "not a JSON object" in caplog.text


def test_over_count_with_duplicate_in_clip_window_rejected():
    """The clip path still runs ALL parent validations: a duplicate inside the
    first N_QUESTIONS fails the parent's duplicate check -> counted reject."""
    obj = json.loads(_payload(qg.N_QUESTIONS + 1, "t3"))
    obj["questions"][1] = obj["questions"][0]  # duplicate inside the clip window
    assert dg._parse_generation_tolerant(json.dumps(obj), "sycophancy") is None


# ── production body: _extend_bank through the mock_gen seam ──────────────────


def _cfg(tmp_path: Path) -> dg.Cfg:
    return dg.Cfg(
        out_root=tmp_path / "root",
        phases=("banks",),
        behaviors=("syc",),
        contexts=(),
        smoke=True,
        mock_gen=True,
    )


def test_extend_bank_absorbs_rejects_and_clips(tmp_path, monkeypatch, caplog):
    """attempt 1 malformed + attempt 2 under-count are counted rejects; the
    attempt-3 over-count is clipped and fills the smoke target — no crash."""

    def fake_response(spec, attempt):
        if attempt == 1:
            return "{not json"
        if attempt == 2:
            return _payload(qg.N_QUESTIONS - 1, "a2")
        return _payload(qg.N_QUESTIONS + 1, f"a{attempt}")

    monkeypatch.setattr(dg, "_mock_template_response", fake_response)
    cfg = _cfg(tmp_path)
    with caplog.at_level(logging.INFO, logger=dg.logger.name):
        rec = dg._extend_bank(cfg, "syc")
    assert rec["parse_rejects"] == 2
    assert rec["attempts_used"] == 3
    assert len(rec["new_questions"]) == cfg.n_new_questions()
    assert [e for e in rec["screen_log"] if e.get("parse_reject")] == [
        {"attempt": 1, "parse_reject": True},
        {"attempt": 2, "parse_reject": True},
    ]
    assert "clipped over-count generation" in caplog.text
    out = json.loads((cfg.banks_dir / "sycophancy_extended.json").read_text(encoding="utf-8"))
    assert out["parse_rejects"] == 2


def test_extend_bank_floor_unreachable_still_raises(tmp_path, monkeypatch):
    """Every call rejected -> the bank floor fails loud after the budgeted
    tranches (no silent degradation), naming the parse-reject count."""
    monkeypatch.setattr(dg, "_mock_template_response", lambda spec, attempt: "{not json")
    with pytest.raises(RuntimeError, match=r"parse-rejected"):
        dg._extend_bank(_cfg(tmp_path), "syc")


# ── crash-fix r6: collection-time overlap filter + resume revalidation ────────


def test_forbidden_questions_real_body():
    """Executes the REAL _forbidden_questions body (other tests stub it —
    production-body rule): non-empty, strip-normalized, and a superset of the
    stripped eval/marker sources the :587-class assert quantifies over.
    Bank items are compared programmatically, never printed."""
    from explore_persona_space.artifacts import banks as banks_mod
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_QUESTIONS_20,
    )

    fq = dg._forbidden_questions()
    assert fq and all(isinstance(q, str) and q == q.strip() for q in fq)
    assert {q.strip() for q in EVAL_QUESTIONS_20} <= fq
    marker_train = json.loads(
        (qg.BANKS_DIR / "issue1481_marker_train10_v1.json").read_text(encoding="utf-8")
    )
    assert {q.strip() for q in marker_train} <= fq
    for behavior in dg.BEHAVIOR_BY_KEY.values():
        assert {q.strip() for q in banks_mod.bank_slice(behavior, "eval")} <= fq


def test_extend_bank_drops_collisions_counts_and_backfills(tmp_path, monkeypatch, caplog):
    """The P0-crash shape: candidate questions colliding with the eval/marker
    exclusion set are DROPPED at collection (counted as overlap_rejects) and
    backfilled from the same call's surplus — the final bank satisfies the
    unchanged :587-class disjointness invariant instead of crashing on it."""
    forbidden = {f"Mock parse question a1 n{i:02d}?" for i in range(3)}
    monkeypatch.setattr(dg, "_forbidden_questions", lambda: forbidden)
    monkeypatch.setattr(
        dg, "_mock_template_response", lambda spec, attempt: _payload(qg.N_QUESTIONS, f"a{attempt}")
    )
    cfg = _cfg(tmp_path)
    with caplog.at_level(logging.INFO, logger=dg.logger.name):
        rec = dg._extend_bank(cfg, "syc")
    assert rec["overlap_rejects"] == 3
    assert len(rec["new_questions"]) == cfg.n_new_questions()
    # the unchanged post-hoc invariant (issue1947_datagen phase_banks assert):
    assert not sorted(set(q for q in rec["new_questions"]) & forbidden)
    # backfill: the first kept question is the first NON-colliding candidate
    assert rec["new_questions"][0] == "Mock parse question a1 n03?"
    assert "overlap-rejected 3 candidate question(s)" in caplog.text
    out = json.loads((cfg.banks_dir / "sycophancy_extended.json").read_text(encoding="utf-8"))
    assert out["overlap_rejects"] == 3


def test_extend_bank_resume_revalidates_stale_colliding_bank(tmp_path, monkeypatch, caplog):
    """Write-ordering gap (crash-fix r6): bank files land BEFORE the phase
    structural asserts run, so a crashed run leaves colliding banks on disk.
    The resume path must re-validate a loaded bank and REGENERATE on collision
    — never skip on row count alone."""
    forbidden = {"Mock parse question a1 n00?"}
    monkeypatch.setattr(dg, "_forbidden_questions", lambda: forbidden)
    monkeypatch.setattr(
        dg, "_mock_template_response", lambda spec, attempt: _payload(qg.N_QUESTIONS, f"a{attempt}")
    )
    cfg = _cfg(tmp_path)
    stale_path = cfg.banks_dir / "sycophancy_extended.json"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_qs = ["Mock parse question a1 n00?"] + [
        f"stale unique question n{i:02d}?" for i in range(1, cfg.n_new_questions())
    ]
    stale_path.write_text(json.dumps({"new_questions": stale_qs}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=dg.logger.name):
        rec = dg._extend_bank(cfg, "syc")
    assert "resume-loaded extension has 1 eval/marker collisions" in caplog.text
    assert not set(rec["new_questions"]) & forbidden
    assert len(rec["new_questions"]) == cfg.n_new_questions()
    rewritten = json.loads(stale_path.read_text(encoding="utf-8"))
    assert rewritten["new_questions"] == rec["new_questions"]
    assert rewritten["overlap_rejects"] == 1


def test_extend_bank_resume_clean_bank_still_skips(tmp_path, monkeypatch):
    """A clean resume-loaded bank (count >= target, zero collisions) is
    returned as-is — the revalidation never forces a spurious regeneration
    (mock seam raises if any generation call is attempted)."""
    monkeypatch.setattr(dg, "_forbidden_questions", lambda: {"never generated?"})

    def _boom(spec, attempt):  # pragma: no cover - failure branch
        raise AssertionError("clean resume bank must not trigger regeneration")

    monkeypatch.setattr(dg, "_mock_template_response", _boom)
    cfg = _cfg(tmp_path)
    path = cfg.banks_dir / "sycophancy_extended.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_qs = [f"clean unique question n{i:02d}?" for i in range(cfg.n_new_questions())]
    path.write_text(json.dumps({"new_questions": clean_qs}), encoding="utf-8")
    rec = dg._extend_bank(cfg, "syc")
    assert rec["new_questions"] == clean_qs
