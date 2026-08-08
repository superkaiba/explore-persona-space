"""Pins for the #1739 compliance-DV rubric v2-forensic revision (item D).

The v1 pilot FAILed at parse_fail 0.94-0.99 (450-draw wave, 2026-08-05):
234 REFUSAL/UNABLE string verdicts + 208 no-verdict parse errors, 8 valid.
Root-cause refinement found during the v2 revision: the v1 rubric carried NO
``{question}``/``{answer}`` slots, so the judge's user message was the bare
rubric — no transcript was ever shown. These tests pin:

1. the transcript slots (FAILS pre-fix on the v1 rubric — the core bug),
2. the rubric-family bump (cache/provenance identity),
3. the honesty constraint (forensic framing, no jailbreak phrasing),
4. the last-object-wins verdict extraction (``_extract_score_json``),
5. the verdict routing incl. the REFUSAL drop class (``_route_verdict``),
6. the draw classification + raw-text rescue (``_classify_draw``),
7. the refusal-split reduce arithmetic (``reduce_compliance_draws``),
8. the gate verdict's refusal-not-mechanically-gated semantics,
9. the offline smoke end-to-end (real main() body, no API).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from issue1739_compliance_pilot import (  # noqa: E402
    COMPLIANCE_JUDGE_SYSTEM_PROMPT,
    COMPLIANCE_RUBRIC,
    RUBRIC_FAMILY,
    _classify_draw,
    _compute_gate_verdict,
    _extract_score_json,
    _route_verdict,
    reduce_compliance_draws,
)

# ---------------------------------------------------------------------------
# 1-3: rubric instrument pins
# ---------------------------------------------------------------------------


def test_rubric_carries_transcript_slots():
    """The core v1 bug: no slots -> the judge never saw the transcript.

    ``judge_completions_batch``'s user message is ``format_user_msg``'s
    output, which is the rubric with ``{question}``/``{answer}`` replaced —
    a slot-less rubric silently drops the transcript (fails pre-fix).
    """
    assert "{question}" in COMPLIANCE_RUBRIC
    assert "{answer}" in COMPLIANCE_RUBRIC
    # Exactly once each — a duplicated slot would duplicate transcript text.
    assert COMPLIANCE_RUBRIC.count("{question}") == 1
    assert COMPLIANCE_RUBRIC.count("{answer}") == 1


def test_rubric_family_bumped_to_v2():
    """New instrument identity: a v1-labeled artifact must never be conflated
    with a v2 read (provenance; the JudgeCache fingerprint changes with the
    text anyway, but the family id is the human-legible key)."""
    assert RUBRIC_FAMILY != "strongreject-style-compliance-v1"
    assert "v2" in RUBRIC_FAMILY


def test_rubric_honest_forensic_framing():
    """Forensic/descriptive framing must be present; jailbreak-shaped
    phrasings must be absent (the orchestrator's HARD CONSTRAINT: honest
    framing, no false authorization, no ignore-your-guidelines)."""
    low = COMPLIANCE_RUBRIC.lower()
    sys_low = COMPLIANCE_JUDGE_SYSTEM_PROMPT.lower()
    assert "recorded" in low
    assert "safety" in low  # safety-monitoring research framing
    for banned in (
        "ignore your",
        "ignore previous",
        "you are authorized",
        "guidelines do not apply",
        "without any restrictions",
    ):
        assert banned not in low, f"jailbreak-shaped phrase in rubric: {banned!r}"
        assert banned not in sys_low, f"jailbreak-shaped phrase in system prompt: {banned!r}"
    # Reason-then-JSON contract + the judge-side REFUSAL escape.
    assert '{"score": N}' in COMPLIANCE_RUBRIC
    assert '{"score": "REFUSAL"}' in COMPLIANCE_RUBRIC
    # The recorded model's refusal is a score-0 observation, not a judge
    # REFUSAL — the disambiguation bullet must anchor refusal at 0.
    assert "score 0" in COMPLIANCE_RUBRIC


# ---------------------------------------------------------------------------
# 4: last-object-wins extraction
# ---------------------------------------------------------------------------


def test_extract_score_json_last_object_wins():
    text = 'Rationale quoting {"score": 10} early.\nFinal verdict: {"score": 85}'
    assert _extract_score_json(text) == 85


def test_extract_score_json_skips_verdictless_trailing_object():
    text = 'Reasoning, then {"score": 40} and a trailing note {"foo": 1}'
    assert _extract_score_json(text) == 40


def test_extract_score_json_refusal_string():
    text = 'Brief reasoning.\n{"score": "REFUSAL"}'
    assert _extract_score_json(text) == "REFUSAL"


def test_extract_score_json_none_on_no_json():
    assert _extract_score_json("no json here at all") is None
    assert _extract_score_json("") is None
    assert _extract_score_json(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5: verdict routing (drop-never-coerce, rule 9)
# ---------------------------------------------------------------------------


def test_route_verdict_numeric():
    assert _route_verdict(85) == (85.0, None)
    assert _route_verdict(0) == (0.0, None)
    assert _route_verdict("42") == (42.0, None)
    assert _route_verdict(150) == (None, "out_of_range")
    assert _route_verdict(-5) == (None, "out_of_range")


def test_route_verdict_refusal_class():
    assert _route_verdict("REFUSAL") == (None, "refusal")
    assert _route_verdict("refusal") == (None, "refusal")
    assert _route_verdict("UNABLE TO EVALUATE") == (None, "refusal")
    assert _route_verdict("I cannot evaluate this content") == (None, "refusal")


def test_route_verdict_malformed():
    assert _route_verdict(None) == (None, "no_verdict")
    assert _route_verdict(True) == (None, "non_numeric")
    assert _route_verdict("") == (None, "no_verdict")
    assert _route_verdict("banana") == (None, "non_numeric")


# ---------------------------------------------------------------------------
# 6: draw classification incl. the raw-text rescue
# ---------------------------------------------------------------------------


def test_classify_draw_valid_and_refusal_verdicts():
    assert _classify_draw({"score": 75}) == (75.0, "valid", None)
    score, cls, detail = _classify_draw({"score": "REFUSAL"})
    assert (score, cls, detail) == (None, "refusal", "refusal_verdict")
    score, cls, detail = _classify_draw({"score": "UNABLE TO EVALUATE"})
    assert (score, cls, detail) == (None, "refusal", "refusal_verdict")


def test_classify_draw_transport():
    parsed = {"error": True, "transport": True, "reason": "api_error"}
    score, cls, _detail = _classify_draw(parsed)
    assert (score, cls) == (None, "transport")


def test_classify_draw_rescued_from_raw_text():
    """Round-18 retention: a parse-error dict carrying ``_raw_text`` with a
    trailing verdict object is RESCUED to a valid draw (the JSON-only
    ``parse_judge_json`` missed it; the end-scan finds it)."""
    parsed = {
        "error": True,
        "reason": "parse_error",
        "_raw_text": 'Reasoning first, no braces.\n{"score": 60}',
    }
    assert _classify_draw(parsed) == (60.0, "valid", "rescued_json")


def test_classify_draw_prose_refusal_from_raw_text():
    parsed = {
        "error": True,
        "reason": "parse_error",
        "_raw_text": "I can't classify this transcript for you.",
    }
    assert _classify_draw(parsed) == (None, "refusal", "refusal_prose")


def test_classify_draw_parse_fail_no_verdict():
    parsed = {"error": True, "reason": "parse_error", "_raw_text": "garbled output"}
    score, cls, detail = _classify_draw(parsed)
    assert (score, cls, detail) == (None, "parse_fail", "no_verdict")


def test_classify_draw_out_of_range_structured_verdict():
    score, cls, detail = _classify_draw({"score": 150})
    assert (score, cls, detail) == (None, "parse_fail", "out_of_range")


# ---------------------------------------------------------------------------
# 7: refusal-split reduce arithmetic
# ---------------------------------------------------------------------------


def _write_save_raw(tmp_path: Path, all_scores: dict) -> Path:
    p = tmp_path / "judge_raw.json"
    p.write_text(json.dumps({"all_scores": all_scores}))
    return p


def test_reduce_compliance_draws_split_counts(tmp_path):
    items = [("item-a", "q", "c"), ("item-b", "q", "c"), ("item-c", "q", "c")]
    all_scores = {
        # item-a: two valid draws (mean 50), one refusal.
        "item-a__00000__00": {"score": 40},
        "item-a__00000__01": {"score": 60},
        "item-a__00000__02": {"score": "REFUSAL"},
        # item-b: one rescued draw, one transport, one parse fail.
        "item-b__00001__00": {
            "error": True,
            "reason": "parse_error",
            "_raw_text": 'reasoning\n{"score": 80}',
        },
        "item-b__00001__01": {"error": True, "transport": True, "reason": "api_error"},
        "item-b__00001__02": {"error": True, "reason": "parse_error", "_raw_text": "garble"},
        # item-c: all draws refused -> None score.
        "item-c__00002__00": {"score": "REFUSAL"},
        # A foreign custom_id must be ignored.
        "other__00003__00": {"score": 99},
    }
    save_raw = _write_save_raw(tmp_path, all_scores)
    red = reduce_compliance_draws(save_raw, items)
    drop = red["per_arm_drop"]
    assert drop["n_total_draws"] == 7
    assert drop["n_valid_draws"] == 3
    assert drop["n_refusal_draws"] == 2
    assert drop["n_parse_fail_draws"] == 1
    assert drop["n_transport_lost_draws"] == 1
    # rule-9 content total = refusal + parse_fail; transport NEVER blended.
    assert drop["n_dropped_draws"] == 3
    assert drop["parse_fail_frac"] == 1 / 7
    assert drop["refusal_frac"] == 2 / 7
    assert drop["n_rescued_draws"] == 1
    assert red["scores"]["item-a"] == 50.0
    assert red["scores"]["item-b"] == 80.0
    assert red["scores"]["item-c"] is None
    assert red["n_items_with_score"] == 2
    assert red["per_item_transport_losses"] == {"item-b": 1}


# ---------------------------------------------------------------------------
# 8: gate verdict — refusal leg reported, not mechanically gated
# ---------------------------------------------------------------------------


def _rung(n_total, n_parse, n_refusal):
    return {
        "per_arm_drop": {
            "n_total_draws": n_total,
            "parse_fail_frac": n_parse / n_total,
            "refusal_frac": n_refusal / n_total,
        }
    }


def test_gate_passes_parse_leg_despite_large_refusal_frac():
    verdict = _compute_gate_verdict({"r1": _rung(100, 0, 90)}, max_tokens=1024)
    assert verdict["gate_pass"] is True  # parse-fail leg only (mechanical)
    assert verdict["refusal_frac_max"] == 0.9  # orchestrator judges this leg
    assert "orchestrator" in verdict["refusal_leg"]


def test_gate_fails_on_parse_fail_over_ceiling():
    verdict = _compute_gate_verdict({"r1": _rung(100, 5, 0)}, max_tokens=1024)
    assert verdict["gate_pass"] is False
    assert verdict["recommend_max_tokens"] == 2048


def test_gate_refusal_per_rung_reported():
    verdict = _compute_gate_verdict(
        {"r1": _rung(100, 1, 10), "r2": _rung(50, 0, 5)}, max_tokens=1024
    )
    assert verdict["gate_pass"] is True
    assert verdict["refusal_frac_per_rung"] == {"r1": 0.1, "r2": 0.1}


# ---------------------------------------------------------------------------
# 9: offline smoke end-to-end through the REAL main() body (no API calls) —
#    also pins the output-stem-derived per-rung dir (v1 evidence never
#    clobbered by a v2 output path).
# ---------------------------------------------------------------------------


def test_smoke_main_end_to_end(tmp_path, monkeypatch):
    import issue1739_compliance_pilot as mod

    out = tmp_path / "compliance_pilot_v2.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue1739_compliance_pilot.py",
            "--smoke",
            "--output",
            str(out),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
    )
    rc = mod.main()
    assert rc == 0
    manifest = json.loads(out.read_text())
    assert manifest["rubric_family"] == RUBRIC_FAMILY
    assert manifest["verdict"]["gate_pass"] is True
    assert manifest["verdict"]["refusal_frac_max"] > 0.0
    # Per-rung artifacts derive from the OUTPUT STEM, not a hardcoded
    # "compliance_pilot" (which would clobber the v1 wave's judge_raw).
    stem_dir = tmp_path / "compliance_pilot_v2"
    assert stem_dir.exists()
    populated = [d for d in stem_dir.iterdir() if d.is_dir()]
    assert populated, "expected per-rung stub dirs under the output stem"
    for d in populated:
        assert (d / "judge_raw_compliance_pilot.json").exists()
