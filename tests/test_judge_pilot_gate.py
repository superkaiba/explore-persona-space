"""Tests for the rule-26 judge pilot gate (#2021, `eval/judge_pilot.py`).

Most tests fake `judge_graded` at the `judge_pilot` seam (signature-conformant
via `unittest.mock.create_autospec`; the fake WRITES a synthetic save_raw and
reduces it with the REAL `judge_result_from_save_raw`, so the production reduce
+ classification run in every test). The #906 real-body test executes
`judge_pilot_gate`'s real body THROUGH the real `judge_graded`, faking only the
API boundary (`batch_judge.judge_completions_batch`, autospec'd).
"""

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.eval import batch_judge, graded_judge, judge_pilot
from explore_persona_space.eval.judge_pilot import judge_pilot_gate

PROMPT = "Rate the answer 0-100.\n{question}\n{answer}"

# Synthetic per-draw parsed entries (the shapes the post-#2021 mint sites persist).
KEPT = {"score": 95, "stop_reason": "end_turn"}
KEPT_TRUNC = {"score": 80, "stop_reason": "max_tokens"}  # kept verdict, truncated response
PARSE_DROP = {"error": True, "reasoning": "parse_error", "stop_reason": "end_turn"}
TRUNC_DROP = {"error": True, "reasoning": "parse_error", "stop_reason": "max_tokens"}
REFUSAL = {"score": "REFUSAL", "stop_reason": "end_turn"}
TRANSPORT = {"error": True, "transport": True, "reasoning": "transport: overloaded (529)"}
LEGACY_DROP = {"error": True, "reasoning": "parse_error"}  # pre-#2021: no stop_reason
# #2151: API-classifier refusal — succeeded row, empty content, no verdict.
API_REFUSAL_DROP = {
    "error": True,
    "reasoning": "parse_error",
    "raw_text": "",
    "stop_reason": "refusal",
}


def _items(n: int, prefix: str) -> list[tuple[str, str, str]]:
    return [(f"{prefix}{i}", f"q{i}", f"a{i}") for i in range(n)]


def _install_fake_judge(monkeypatch, arm_draws: dict[str, list], record: list | None = None):
    """Monkeypatch `judge_pilot.judge_graded` with a signature-conformant fake.

    The fake writes `save_raw` from `arm_draws[<arm>]` (one parsed entry per
    (item, draw) slot, in order) and returns the REAL reduce's JudgeResult, so
    gate stats exercise the production classification. `record` (optional)
    captures (arm, [item ids], max_tokens) per call.
    """

    def impl(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model="unused-default",  # the gate always passes judge_model explicitly
        temperature=graded_judge.DEFAULT_JUDGE_TEMPERATURE,
        max_tokens=64,
        dry_run=False,
        threshold_base=None,
    ):
        arm = Path(save_raw).stem.removeprefix("judge_raw_pilot_")
        if record is not None:
            record.append((arm, [item_id for item_id, _q, _a in items], max_tokens))
        draws = list(arm_draws[arm])
        all_scores = {}
        di = 0
        for idx, (item_id, _q, _a) in enumerate(items):
            for comp in range(n_draws):
                if di >= len(draws):
                    break
                all_scores[f"{item_id}__{idx:05d}__{comp:02d}"] = draws[di]
                di += 1
        p = Path(save_raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"all_scores": all_scores}))
        return graded_judge.judge_result_from_save_raw(p, items)

    fake = mock.create_autospec(graded_judge.judge_graded, side_effect=impl)
    monkeypatch.setattr(judge_pilot, "judge_graded", fake)
    return fake


def _run(monkeypatch, tmp_path, arm_items, arm_draws, **kw):
    _install_fake_judge(monkeypatch, arm_draws)
    return judge_pilot_gate(
        arm_items,
        PROMPT,
        max_tokens=kw.pop("max_tokens", 800),
        cache_dir=tmp_path / "cache",
        save_raw_dir=tmp_path / "raw",
        **kw,
    )


def test_pass_on_clean_pilot(monkeypatch, tmp_path):
    arms = {"a": _items(6, "a"), "b": _items(6, "b")}
    draws = {"a": [KEPT] * 12, "b": [KEPT] * 12}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.passed is True
    assert rep.verdict == "PASS"
    assert rep.failures == []
    assert rep.n_total_draws == 24
    assert rep.arms["a"].n_scored == 12
    assert rep.arms["a"].n_content_dropped == 0
    assert rep.arms["a"].parse_fail_rate == 0.0
    assert rep.arms["a"].stop_reason_tally == {"end_turn": 12}
    assert rep.arms["a"].waived is False


def test_fail_on_nonzero_truncation_stop_reason(monkeypatch, tmp_path):
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 11 + [TRUNC_DROP]}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.verdict == "FAIL"
    assert rep.arms["a"].n_truncation == 1
    trunc_failures = [f for f in rep.failures if "truncation" in f]
    assert trunc_failures, rep.failures
    assert "re-pilot" in trunc_failures[0]
    assert "NEVER waivable" in trunc_failures[0]
    assert "never shrink" in trunc_failures[0]


def test_kept_truncated_verdict_fails_gate(monkeypatch, tmp_path):
    """The tally clause fires with the drop counter at ZERO (unit A pinned the
    reduce side — a kept-but-truncated verdict is visible only in the tally)."""
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 11 + [KEPT_TRUNC]}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.arms["a"].n_truncation == 0
    assert rep.arms["a"].n_content_dropped == 0
    assert rep.arms["a"].n_scored == 12  # the truncated verdict IS kept as a score
    assert rep.arms["a"].stop_reason_tally.get("max_tokens") == 1
    assert rep.verdict == "FAIL"
    assert any("truncation" in f and "KEPT" in f for f in rep.failures), rep.failures


def test_api_refusal_draws_reported_but_gate_verdict_unchanged(monkeypatch, tmp_path, caplog):
    """#2151 (plan §6 test 4): api-refusal draws surface in the arm report
    (``n_api_refusal`` + the tally's "refusal" row + the reduce's WARNING) but
    change NO gate condition — a 25%-censored pilot still PASSes BY DESIGN
    (rule 28's non-coverage note: the rule-26 gate is NOT protective for this
    class). The parse-fail denominator excludes the censored draws exactly as
    it excludes transport losses, and the effective-draws floor does NOT
    shrink (it keys on ``n_draws - n_transport_lost`` only)."""
    import logging

    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 9 + [API_REFUSAL_DROP] * 3}
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.graded_judge"):
        rep = _run(monkeypatch, tmp_path, arms, draws)

    # Verdict UNCHANGED: no failure names the censoring.
    assert rep.passed is True
    assert rep.verdict == "PASS"
    assert rep.failures == []
    # REPORT-only surfacing: the arm carries the count + the tally row.
    assert rep.arms["a"].n_api_refusal == 3
    assert rep.arms["a"].stop_reason_tally.get("refusal") == 3
    # The censored draws leave every content/transport counter untouched.
    assert rep.arms["a"].n_scored == 9
    assert rep.arms["a"].n_content_dropped == 0
    assert rep.arms["a"].n_transport_lost == 0
    assert rep.arms["a"].parse_fail_rate == 0.0
    assert rep.arms["a"].n_unknown_stop_reason_drops == 0
    # The reduce's WARNING (the rule-28 residual backstop) fired.
    assert any("API-refusal" in rec.message for rec in caplog.records)


def test_api_refusal_field_serializes_in_report_dict(monkeypatch, tmp_path):
    """#2151: ``n_api_refusal`` rides ``asdict``-style report serialization —
    the field the remediation recipe (rule 28) tells a wave owner to read."""
    from dataclasses import asdict

    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 11 + [API_REFUSAL_DROP]}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    d = asdict(rep)
    assert d["arms"]["a"]["n_api_refusal"] == 1


def test_fail_on_parse_fail_rate_at_threshold(monkeypatch, tmp_path):
    # Exactly 2% (1 non-refusal content drop / 50 answered draws) trips the >= bar.
    arms = {"a": _items(25, "a")}
    draws = {"a": [KEPT] * 49 + [PARSE_DROP]}
    rep = _run(monkeypatch, tmp_path / "at", arms, draws)
    assert rep.arms["a"].parse_fail_rate == pytest.approx(0.02)
    assert rep.verdict == "FAIL"
    assert any("parse-fail" in f and "rule 26(b)" in f for f in rep.failures), rep.failures

    # Just below the bar (1/100 = 1%) passes.
    arms2 = {"a": _items(50, "a")}
    draws2 = {"a": [KEPT] * 99 + [PARSE_DROP]}
    rep2 = _run(monkeypatch, tmp_path / "below", arms2, draws2)
    assert rep2.arms["a"].parse_fail_rate == pytest.approx(0.01)
    assert rep2.passed is True


def test_refusals_excluded_from_parse_fail_rate(monkeypatch, tmp_path):
    # 2 refusals in 12 draws would be ~17% if blended; refusals are rule-9 verdicts.
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 10 + [REFUSAL] * 2}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.arms["a"].n_refusal == 2
    assert rep.arms["a"].n_content_dropped == 2  # refusal IS a content drop (subset)
    assert rep.arms["a"].parse_fail_rate == 0.0
    assert rep.passed is True


def test_waived_arm_does_not_fail_gate_but_is_reported(monkeypatch, tmp_path):
    arms = {"noisy": _items(6, "n"), "clean": _items(6, "c")}
    draws = {"noisy": [KEPT] * 10 + [PARSE_DROP] * 2, "clean": [KEPT] * 12}
    rep = _run(monkeypatch, tmp_path, arms, draws, waive_parse_fail_arms={"noisy"})
    assert rep.passed is True
    assert rep.arms["noisy"].waived is True
    assert rep.arms["noisy"].parse_fail_rate == pytest.approx(2 / 12)
    assert any("noisy" in w and "WAIVED" in w for w in rep.warnings), rep.warnings

    # A typo'd waiver fails loud instead of silently not waiving.
    with pytest.raises(ValueError, match="unknown arm"):
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c2",
            save_raw_dir=tmp_path / "r2",
            waive_parse_fail_arms={"nope"},
        )


def test_truncation_fail_not_waivable(monkeypatch, tmp_path):
    arms = {"t": _items(6, "t")}
    draws = {"t": [KEPT] * 11 + [TRUNC_DROP]}
    rep = _run(monkeypatch, tmp_path, arms, draws, waive_parse_fail_arms={"t"})
    assert rep.verdict == "FAIL"
    assert any("truncation" in f for f in rep.failures), rep.failures
    # The waiver silenced the parse-fail check only — truncation still fails.
    assert not any("parse-fail" in f for f in rep.failures)


def test_transport_hollowed_arm_fails_not_passes(monkeypatch, tmp_path):
    """[S2]: an arm hollowed out by transport losses must FAIL, never silently PASS."""
    arms = {"h": _items(6, "h"), "ok": _items(6, "o")}
    draws = {"h": [TRANSPORT] * 11 + [KEPT], "ok": [KEPT] * 12}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.verdict == "FAIL"
    assert rep.arms["h"].n_transport_lost == 11
    assert rep.arms["h"].n_scored == 1
    hollow = [f for f in rep.failures if f.startswith("arm h:")]
    assert hollow, rep.failures
    assert "re-run the pilot" in hollow[0]
    assert "transport-hollowed" in hollow[0]
    # Transport draws are excluded from the tally (unit A [S1]) and warned about.
    assert rep.arms["h"].stop_reason_tally == {"end_turn": 1}
    assert any("transport-lost" in w for w in rep.warnings)
    # The healthy sibling arm carries no failure of its own.
    assert not any(f.startswith("arm ok:") for f in rep.failures)


def test_unknown_stop_reason_drop_warns_stale_cache(monkeypatch, tmp_path):
    """[M2]: a content drop with NO persisted stop_reason partially detects a
    stale pre-#2021 cache — advisory warning, plus the "unknown" tally row."""
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 11 + [LEGACY_DROP]}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.arms["a"].n_unknown_stop_reason_drops == 1
    assert rep.arms["a"].stop_reason_tally.get("unknown") == 1
    assert any("stale pre-#2021" in w for w in rep.warnings), rep.warnings


def test_max_tokens_floor_warning_never_fails(monkeypatch, tmp_path):
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 12}
    rep = _run(monkeypatch, tmp_path, arms, draws, max_tokens=64)
    assert rep.passed is True  # note only — never a verdict change
    assert any("never" in w and "auto-shrink" in w for w in rep.warnings), rep.warnings


def test_report_json_roundtrip_and_report_path_write(monkeypatch, tmp_path):
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 12}
    report_path = tmp_path / "sub" / "pilot_report.json"
    rep = _run(monkeypatch, tmp_path, arms, draws, report_path=report_path)
    assert report_path.is_file()
    loaded = json.loads(report_path.read_text())
    assert loaded == rep.to_json()
    assert loaded["verdict"] == "PASS"
    assert loaded["max_tokens"] == 800
    assert loaded["arms"]["a"]["n_scored"] == 12
    assert loaded["arms"]["a"]["stop_reason_tally"] == {"end_turn": 12}
    assert loaded["rubric_hash"] == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16]


def test_deterministic_subsample_seeded(monkeypatch, tmp_path):
    arms = {"a": _items(100, "a")}
    draws = {"a": [KEPT] * 20}  # 10 items x 2 draws under target_total_draws=20

    def run(seed, sub):
        record: list = []
        _install_fake_judge(monkeypatch, draws, record=record)
        rep = judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / sub / "cache",
            save_raw_dir=tmp_path / sub / "raw",
            target_total_draws=20,
            seed=seed,
        )
        assert rep.arms["a"].n_items == 10
        return record[0][1]

    ids_first = run(0, "one")
    ids_second = run(0, "two")
    ids_other_seed = run(1, "three")
    assert ids_first == ids_second  # same seed -> identical subsample
    assert ids_first != ids_other_seed  # different seed -> different subsample
    assert len(set(ids_first)) == 10


def test_pilot_gate_real_body_reaches_judge_graded(monkeypatch, tmp_path):
    """#906 real-body test: judge_pilot_gate's REAL body through the REAL
    judge_graded (real rubric split, real custom_id packing, real
    judge_result_from_save_raw reduce, real verdict logic), faking ONLY the API
    boundary — `batch_judge.judge_completions_batch`, autospec'd so the real
    call shape is signature-validated."""

    def impl(*args, **kwargs):
        completions = kwargs["completions"]
        save_raw = Path(kwargs["save_raw"])
        all_scores = {}
        for persona, by_q in completions.items():
            for idx, (_q, comps) in enumerate(by_q.items()):
                for comp_idx in range(len(comps)):
                    all_scores[f"{persona}__{idx:05d}__{comp_idx:02d}"] = {
                        "score": 90,
                        "stop_reason": "end_turn",
                    }
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": all_scores}))

    spec = mock.create_autospec(batch_judge.judge_completions_batch, side_effect=impl)
    monkeypatch.setattr(batch_judge, "judge_completions_batch", spec)

    arms = {"a": _items(6, "a"), "b": _items(6, "b")}
    report_path = tmp_path / "report.json"
    rep = judge_pilot_gate(
        arms,
        PROMPT,
        max_tokens=987,
        cache_dir=tmp_path / "cache",
        save_raw_dir=tmp_path / "raw",
        n_draws=2,
        report_path=report_path,
    )
    assert rep.passed is True
    assert rep.n_total_draws == 24
    assert rep.arms["a"].n_scored == 12
    assert rep.arms["b"].stop_reason_tally == {"end_turn": 12}
    # The EXACT production instrument reached the client: budget + per-arm pilot cache.
    assert spec.call_count == 2
    for call in spec.call_args_list:
        assert call.kwargs["max_tokens"] == 987
        assert Path(call.kwargs["cache_dir"]).parent == tmp_path / "cache"
        assert Path(call.kwargs["cache_dir"]).name in arms
    assert (tmp_path / "raw" / "judge_raw_pilot_a.json").is_file()
    assert (tmp_path / "raw" / "judge_raw_pilot_b.json").is_file()
    assert json.loads(report_path.read_text())["verdict"] == "PASS"


def test_empty_arms_and_bad_arm_names_fail_loud(monkeypatch, tmp_path):
    _install_fake_judge(monkeypatch, {})
    with pytest.raises(ValueError, match="non-empty"):
        judge_pilot_gate(
            {}, PROMPT, max_tokens=800, cache_dir=tmp_path / "c", save_raw_dir=tmp_path / "r"
        )
    with pytest.raises(ValueError, match="filesystem-safe"):
        judge_pilot_gate(
            {"bad/arm": _items(2, "x")},
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c",
            save_raw_dir=tmp_path / "r",
        )
