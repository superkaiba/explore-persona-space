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
    # allow_subresolution_pilot defaults True HERE (the library default is
    # False): the legacy drop-profile tests deliberately use tiny 6-item arms,
    # which are sub-resolution by construction under the #2124 config-time
    # satisfiability guard. The guard's own tests pass False explicitly (or
    # call judge_pilot_gate directly).
    return judge_pilot_gate(
        arm_items,
        PROMPT,
        max_tokens=kw.pop("max_tokens", 800),
        cache_dir=tmp_path / "cache",
        save_raw_dir=tmp_path / "raw",
        allow_subresolution_pilot=kw.pop("allow_subresolution_pilot", True),
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
            allow_subresolution_pilot=True,  # 20 draws is deliberately tiny (#2124)
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
        # 6-item arms are sub-resolution by construction; the guard's strict
        # branch has its own dedicated tests below (#2124).
        allow_subresolution_pilot=True,
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


# --- #2124: config-time satisfiability guard (rule 26 sizing clause) ------------


def test_unsatisfiable_budget_raises_before_any_judge_call(monkeypatch, tmp_path):
    """Negative control (plan criterion 7): a budget-unsatisfiable config raises
    ValueError naming the arms, realized/required, both knobs, and the exact
    discretized remedy — BEFORE any judge_graded call (zero API spend)."""
    record: list = []
    fake = _install_fake_judge(monkeypatch, {}, record=record)
    # 4 arms x 60 items, n_draws=2, T=200: per_arm_items = 200 // 8 = 25 ->
    # realized 50 < required 51 (budget-limited: 60 items >= 26 needed).
    arms = {f"arm{i}": _items(60, f"a{i}") for i in range(4)}
    with pytest.raises(ValueError, match="budget-limited") as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c",
            save_raw_dir=tmp_path / "r",
            target_total_draws=200,
        )
    msg = str(exc.value)
    assert "arm0" in msg and "realized 50" in msg and "required 51" in msg
    assert "parse_fail_threshold" in msg and "min_effective_draws_per_arm" in msg
    # The EXACT discretized form: 4 * 2 * ceil(51/2) = 208, never 51*4 = 204.
    assert "target_total_draws >= 208" in msg
    assert "204" not in msg
    assert fake.call_count == 0, "the refusal must fire BEFORE any judge_graded call"
    assert record == []


@pytest.mark.parametrize(("n_draws", "expected_suggestion"), [(2, 208), (3, 204)])
def test_remedy_is_self_consistent_fed_back(monkeypatch, tmp_path, n_draws, expected_suggestion):
    """The printed remedy, fed back into the same call, PASSES the guard — at
    n_draws=2 AND 3 (the v2-plan bug: T = 51*A at d=2 realizes 50/arm and
    would be rejected by the very guard that printed it)."""
    import re

    arms = {f"arm{i}": _items(60, f"a{i}") for i in range(4)}
    _install_fake_judge(monkeypatch, {})
    with pytest.raises(ValueError) as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "fail" / "c",
            save_raw_dir=tmp_path / "fail" / "r",
            n_draws=n_draws,
            target_total_draws=200,
        )
    m = re.search(r"target_total_draws >= (\d+)", str(exc.value))
    assert m, str(exc.value)
    suggested = int(m.group(1))
    assert suggested == expected_suggestion  # 4 * d * ceil(51/d)
    per_arm = suggested // (4 * n_draws)
    draws = {f"arm{i}": [KEPT] * (per_arm * n_draws) for i in range(4)}
    rep = _run(
        monkeypatch,
        tmp_path / "ok",
        arms,
        draws,
        n_draws=n_draws,
        target_total_draws=suggested,
        allow_subresolution_pilot=False,  # the library default: strict
    )
    assert rep.passed is True
    assert rep.arms["arm0"].n_draws >= 51


def test_strict_boundary_realized_50_raises_51_passes(monkeypatch, tmp_path):
    """The floor is floor(1/0.02) + 1 = 51, NOT 50: at exactly 50 realized
    draws a single parse failure lands ON the strict >= bar (1/50 = 2%)."""
    arms = {"a": _items(60, "a")}
    _install_fake_judge(monkeypatch, {})
    with pytest.raises(ValueError, match="realized 50 draw\\(s\\) < required 51"):
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "f" / "c",
            save_raw_dir=tmp_path / "f" / "r",
            n_draws=1,
            target_total_draws=50,
        )
    rep = _run(
        monkeypatch,
        tmp_path / "ok",
        arms,
        {"a": [KEPT] * 51},
        n_draws=1,
        target_total_draws=51,
        allow_subresolution_pilot=False,
    )
    assert rep.passed is True
    assert rep.arms["a"].n_draws == 51


def test_item_limited_arm_raises_despite_ample_budget(monkeypatch, tmp_path):
    """Negative control (plan criterion 7): a 5-item arm at T=200 is caught
    even though the budget is ample — the arm-size cap (_seeded_subsample)
    bounds realized draws at len(items) * n_draws, and NO budget fixes it."""
    record: list = []
    fake = _install_fake_judge(monkeypatch, {}, record=record)
    arms = {"tiny": _items(5, "t")}
    with pytest.raises(ValueError, match="item-limited") as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c",
            save_raw_dir=tmp_path / "r",
            target_total_draws=200,
        )
    msg = str(exc.value)
    assert "5 item(s) < 26 needed" in msg
    assert "NO target_total_draws" in msg
    assert "waive_parse_fail_arms" in msg and "allow_subresolution_pilot" in msg
    assert fake.call_count == 0, "the refusal must fire BEFORE any judge_graded call"
    assert record == []


def test_allow_subresolution_pilot_downgrades_to_report_warning(monkeypatch, tmp_path):
    arms = {"tiny": _items(5, "t")}
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        {"tiny": [KEPT] * 10},
        target_total_draws=200,
        allow_subresolution_pilot=True,
    )
    assert rep.passed is True
    sub = [w for w in rep.warnings if "sub-resolution pilot ACCEPTED" in w]
    assert sub, rep.warnings
    assert "item-limited" in sub[0] and "tiny" in sub[0]


def test_min_effective_draws_dominates_when_larger(monkeypatch, tmp_path):
    """required = max(min_effective_draws_per_arm, floor(1/threshold)+1): a
    min-effective floor ABOVE the resolution floor binds, and the remedy is
    sized from it."""
    arms = {"a": _items(100, "a")}
    _install_fake_judge(monkeypatch, {})
    # T=60, d=2 -> realized 60: >= 51 (resolution) but < 80 (min-effective).
    with pytest.raises(ValueError, match="realized 60 draw\\(s\\) < required 80") as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "f" / "c",
            save_raw_dir=tmp_path / "f" / "r",
            target_total_draws=60,
            min_effective_draws_per_arm=80,
        )
    assert "target_total_draws >= 80" in str(exc.value)
    rep = _run(
        monkeypatch,
        tmp_path / "ok",
        arms,
        {"a": [KEPT] * 80},
        target_total_draws=80,
        min_effective_draws_per_arm=80,
        allow_subresolution_pilot=False,
    )
    assert rep.passed is True


def test_parse_fail_threshold_nonpositive_raises_ge_one_has_no_floor(monkeypatch, tmp_path):
    """threshold <= 0 raises (min_resolvable undefined — every arm would fail
    unconditionally); threshold >= 1 is accepted with NO resolution floor
    (only the min-effective floor binds)."""
    record: list = []
    fake = _install_fake_judge(monkeypatch, {}, record=record)
    arms = {"a": _items(6, "a")}
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="must be > 0"):
            judge_pilot_gate(
                arms,
                PROMPT,
                max_tokens=800,
                cache_dir=tmp_path / "c",
                save_raw_dir=tmp_path / "r",
                parse_fail_threshold=bad,
            )
    assert fake.call_count == 0
    # threshold=1.0: 12 realized draws >= required max(10, -) = 10 -> runs.
    rep = _run(
        monkeypatch,
        tmp_path / "one",
        arms,
        {"a": [KEPT] * 12},
        parse_fail_threshold=1.0,
        allow_subresolution_pilot=False,
    )
    assert rep.passed is True


def test_runtime_shrink_warns_underpowered_not_instrument_bad(monkeypatch, tmp_path):
    """D-1b: a config-time-satisfiable arm whose ANSWERED draws shrink below
    the resolution floor (transport losses, rule 24) WARNs — an under-powered
    pilot, never a FAIL and never an instrument verdict."""
    arms = {"a": _items(60, "a")}
    # T=120, d=2 -> 60 items x 2 = 120 planned draws (satisfiable); 80 of them
    # transport-lost -> 40 answered < 51.
    draws = {"a": [KEPT] * 40 + [TRANSPORT] * 80}
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        draws,
        target_total_draws=120,
        allow_subresolution_pilot=False,
    )
    assert rep.passed is True  # 40 answered >= min_effective 10; parse rate 0
    shrink = [w for w in rep.warnings if "UNDER-POWERED" in w]
    assert shrink, rep.warnings
    assert "40 answered draw(s)" in shrink[0]
    assert "51" in shrink[0]
    assert "NOT" in shrink[0] and "instrument" in shrink[0]


def test_per_item_completeness_fields_in_arm_stats(monkeypatch, tmp_path):
    """#2124 D-2 (rule 29): n_items / n_items_zero_valid / frac_items_complete
    per arm, including the all-valid (1.0) and all-dropped (0.0) endpoints;
    all three serialize in the report JSON. Report-only — no gate keys on
    them (the FAILs below are the parse-fail clause, not completeness)."""
    arms = {"full": _items(6, "f"), "holed": _items(6, "h"), "empty": _items(6, "e")}
    draws = {
        "full": [KEPT] * 12,
        # item h0's BOTH draws drop -> exactly one zero-valid item.
        "holed": [PARSE_DROP] * 2 + [KEPT] * 10,
        "empty": [PARSE_DROP] * 12,
    }
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.arms["full"].n_items == 6
    assert rep.arms["full"].n_items_zero_valid == 0
    assert rep.arms["full"].frac_items_complete == 1.0
    assert rep.arms["holed"].n_items_zero_valid == 1
    assert rep.arms["holed"].frac_items_complete == pytest.approx(5 / 6)
    assert rep.arms["empty"].n_items_zero_valid == 6
    assert rep.arms["empty"].frac_items_complete == 0.0
    d = rep.to_json()
    assert d["arms"]["holed"]["n_items"] == 6
    assert d["arms"]["holed"]["n_items_zero_valid"] == 1
    assert d["arms"]["holed"]["frac_items_complete"] == pytest.approx(5 / 6)
    # completeness is REPORT-only: the failures are parse-fail, not per-item.
    assert not any("frac_items_complete" in f for f in rep.failures)


def test_judge_result_frac_items_complete_property(tmp_path):
    """Rule 29's production-wave affordance on JudgeResult: kept-draw items /
    all items, off the reduce's pre-seeded scores map; 0-item result raises."""
    items = _items(3, "x")
    all_scores = {
        "x0__00000__00": KEPT,
        "x0__00000__01": KEPT,
        "x1__00001__00": PARSE_DROP,
        "x1__00001__01": PARSE_DROP,
        "x2__00002__00": KEPT,
        "x2__00002__01": TRANSPORT,
    }
    p = tmp_path / "raw.json"
    p.write_text(json.dumps({"all_scores": all_scores}))
    res = graded_judge.judge_result_from_save_raw(p, items)
    assert res.frac_items_complete == pytest.approx(2 / 3)
    # Endpoints, off the same dataclass shape the reduce returns.
    assert (
        graded_judge.JudgeResult(
            scores={"a": 1.0}, n_total_draws=1, n_dropped_draws=0
        ).frac_items_complete
        == 1.0
    )
    assert (
        graded_judge.JudgeResult(
            scores={"a": None}, n_total_draws=1, n_dropped_draws=1
        ).frac_items_complete
        == 0.0
    )
    empty = graded_judge.JudgeResult(scores={}, n_total_draws=0, n_dropped_draws=0)
    with pytest.raises(ValueError, match="zero items"):
        _ = empty.frac_items_complete
