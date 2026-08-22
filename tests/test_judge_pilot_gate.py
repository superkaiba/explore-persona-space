"""Tests for the rule-26 judge pilot gate (#2021/#2152, `eval/judge_pilot.py`).

Most tests fake `judge_graded` at the `judge_pilot` seam (signature-conformant
via `unittest.mock.create_autospec`; the fake WRITES a synthetic save_raw and
reduces it with the REAL `judge_result_from_save_raw`, so the production reduce
+ classification run in every test). The #906 real-body tests execute
`judge_pilot_gate`'s / `judge_graded`'s real bodies THROUGH the real call
chain, faking only the API boundary (`batch_judge.judge_completions_batch`,
autospec'd).

Rules-pin discovery (#1496): this file pins behavior the rule file
`.claude/rules/llm-judging.md` (rules 26/28) prescribes — the literal
`llm-judging.md` here arms Step 9c auto-selection on rules-only diffs.
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


# Sentinel: the fake ECHOES the routing record the gate's forcing implies
# (the REAL conditional — batch iff threshold_base == 0).
_ECHO = object()


def _install_fake_judge(
    monkeypatch,
    arm_draws: dict[str, list],
    record: list | None = None,
    routing=_ECHO,
    routing_by_arm: dict | None = None,
    n_cached: int = 0,
    n_submitted: int | None = None,
):
    """Monkeypatch `judge_pilot.judge_graded` with a signature-conformant fake.

    The fake writes `save_raw` from `arm_draws[<arm>]` (one parsed entry per
    (item, draw) slot, in order) and returns the REAL reduce's JudgeResult, so
    gate stats exercise the production classification. `record` (optional)
    captures (arm, [item ids], max_tokens, threshold_base, force_sync) per
    call. #2152 knobs: `routing` controls the persisted `save_raw["routing"]`
    record — default `_ECHO` echoes what the forcing implies
    (`{"path": "batch"}` iff `threshold_base == 0`, the batch pin; both the
    force_sync pin and the legacy small-n pilot land sync), `None` OMITS the
    record entirely (the unverifiable case), a dict overrides it (the mismatch
    case); `routing_by_arm` overrides per arm (values: dict record, or None to
    omit); `n_cached` / `n_submitted` write the #2151/#2152 cache counters
    into save_raw (defaults 0 / the arm's written draw count).
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
        force_sync=False,
    ):
        arm = Path(save_raw).stem.removeprefix("judge_raw_pilot_")
        if record is not None:
            record.append(
                (
                    arm,
                    [item_id for item_id, _q, _a in items],
                    max_tokens,
                    threshold_base,
                    force_sync,
                )
            )
        draws = list(arm_draws[arm])
        all_scores = {}
        di = 0
        for idx, (item_id, _q, _a) in enumerate(items):
            for comp in range(n_draws):
                if di >= len(draws):
                    break
                all_scores[f"{item_id}__{idx:05d}__{comp:02d}"] = draws[di]
                di += 1
        payload: dict = {"all_scores": all_scores}
        realized = routing
        if routing_by_arm is not None and arm in routing_by_arm:
            realized = routing_by_arm[arm]
        if realized is _ECHO:
            realized = {"path": "batch" if threshold_base == 0 else "sync"}
        if realized is not None:
            payload["routing"] = realized
        payload["n_cached"] = n_cached
        payload["n_submitted"] = len(all_scores) if n_submitted is None else n_submitted
        p = Path(save_raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
        return graded_judge.judge_result_from_save_raw(p, items)

    fake = mock.create_autospec(graded_judge.judge_graded, side_effect=impl)
    monkeypatch.setattr(judge_pilot, "judge_graded", fake)
    return fake


def _run(monkeypatch, tmp_path, arm_items, arm_draws, **kw):
    fake_kw = {
        k: kw.pop(k)
        for k in ("record", "routing", "routing_by_arm", "n_cached", "n_submitted")
        if k in kw
    }
    _install_fake_judge(monkeypatch, arm_draws, **fake_kw)
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


def test_api_refusal_rate_fails_at_default_threshold(monkeypatch, tmp_path, caplog):
    """#2151→#2152 (plan §6 test 8; REVISED from
    test_api_refusal_draws_reported_but_gate_verdict_unchanged — a designed
    tightening): a 25%-censored pilot now FAILs at the default 0.10
    api_refusal_threshold — the rule-28 censor is GATE-KEYED as of #2152
    (llm-judging.md rule 26(d)), superseding #2151 §12.3's REPORT-only
    treatment. All #2151 report-field behavior is retained: ``n_api_refusal``
    + the tally's "refusal" row + the reduce's WARNING; the censored draws
    still leave every content/transport counter untouched and the
    effective-draws floor does NOT shrink."""
    import logging

    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 9 + [API_REFUSAL_DROP] * 3}
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.eval.graded_judge"):
        rep = _run(monkeypatch, tmp_path, arms, draws)

    # The ONE deliberate #2152 verdict change: 3/12 = 25% >= 10% FAILs.
    assert rep.passed is False
    assert rep.verdict == "FAIL"
    ar = [f for f in rep.failures if "api-refusal rate" in f]
    assert ar, rep.failures
    assert "25.0%" in ar[0] and "10%" in ar[0] and "rule 28" in ar[0]
    assert "waive_api_refusal_arms" in ar[0]
    # No OTHER clause fires: the api-refusal failure is the only one.
    assert len(rep.failures) == len(ar)
    # #2151 report surfacing retained: count + tally row + the rate field.
    assert rep.arms["a"].n_api_refusal == 3
    assert rep.arms["a"].api_refusal_rate == pytest.approx(0.25)
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


def test_escape_cannot_satisfy_verdict_floor_verdict_fails(monkeypatch, tmp_path):
    """#2339 criterion 3: a verdict-DOOMED arm (len(items) * n_draws <
    min_effective_draws_per_arm) escaped via allow_subresolution_pilot=True
    runs the pilot and then FAILs the verdict-time min-effective floor —
    the escape bypasses ONLY the config-time refusal."""
    record: list = []
    arms = {"doomed": _items(4, "d")}  # 4 items x n_draws 2 = 8 < floor 10
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        {"doomed": [KEPT] * 8},
        target_total_draws=200,
        allow_subresolution_pilot=True,
        record=record,
    )
    assert record, "the pilot must actually RUN under the escape"
    assert rep.passed is False
    min_eff = [f for f in rep.failures if "min_effective_draws_per_arm=10" in f]
    assert min_eff, rep.failures
    assert "arm doomed" in min_eff[0] and "only 8 effective" in min_eff[0]
    warn = [w for w in rep.warnings if "VERDICT-DOOMED" in w]
    assert warn, rep.warnings
    assert "bypass ONLY this config-time refusal" in warn[0]


def test_item_limited_doomed_remedy_names_real_remedies(monkeypatch, tmp_path):
    """#2339 criterion 4: strict-mode refusal on a verdict-DOOMED arm names
    the honest remedies (floor / n_draws / item pool) and states that the
    escapes cannot produce a PASS."""
    fake = _install_fake_judge(monkeypatch, {})
    arms = {"doomed": _items(4, "d")}
    with pytest.raises(ValueError, match="VERDICT-DOOMED") as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c",
            save_raw_dir=tmp_path / "r",
            target_total_draws=200,
        )
    msg = str(exc.value)
    assert "4 item(s) x n_draws 2 = 8 realizable draw(s)" in msg
    assert "lower the caller's min_effective_draws_per_arm" in msg
    assert "raise n_draws" in msg
    assert "enlarge the arm's item" in msg
    assert "ceil(min_effective_draws_per_arm / n_draws) = 5 item(s)" in msg
    assert "_gate_verdict still FAILs" in msg
    assert "guaranteed post-spend verdict FAIL" in msg
    assert "become usable at config time" in msg
    assert fake.call_count == 0, "the refusal must fire BEFORE any judge_graded call"


def test_item_limited_escapable_remedy_keeps_escapes(monkeypatch, tmp_path):
    """#2339 criterion 5: an item-limited arm that CAN pass after an escape
    (len(items) * n_draws >= min_effective_draws_per_arm) keeps the
    waive/allow_subresolution remedy text and is NOT labeled doomed."""
    _install_fake_judge(monkeypatch, {})
    arms = {"tiny": _items(6, "t")}  # 6 x 2 = 12 >= floor 10; 6 < 26 items
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
    assert "6 item(s) < 26 needed" in msg
    assert "waive_parse_fail_arms" in msg
    assert "allow_subresolution_pilot=True" in msg
    assert "VERDICT-DOOMED" not in msg
    assert "ZERO transport-loss headroom" in msg


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


# --- #2152: transport parity (rule 26(c)) + api-refusal gate (rule 26(d)) --------


def test_wave_batch_declaration_forces_batch_pilot(monkeypatch, tmp_path):
    """Plan §6 test 1: a declared batch wave — pinned (wave_threshold_base=0)
    AND deterministic count-routed (n >= 2*tb at the dispatcher default) —
    forces the pilot onto the Batch path (the fake observes threshold_base=0)
    and records both transports + the wave_routing dict."""
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 12}
    for sub, wave_kwargs in (
        ("pinned", {"wave_threshold_base": 0}),
        ("counted", {"wave_n_calls": 44_310}),
    ):
        record: list = []
        _install_fake_judge(monkeypatch, draws, record=record)
        rep = judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / sub / "c",
            save_raw_dir=tmp_path / sub / "r",
            allow_subresolution_pilot=True,
            **wave_kwargs,
        )
        assert record[0][3] == 0  # threshold_base=0 observed by the fake (batch pin)
        assert record[0][4] is False  # force_sync NOT passed on the batch branch
        assert rep.wave_transport == "batch"
        assert rep.arms["a"].transport == "batch"
        assert rep.pilot_transport == "batch"
        assert rep.passed is True
        assert rep.wave_routing is not None and rep.wave_routing["path"] == "batch"


def test_wave_sync_declaration_threads_force_sync(monkeypatch, tmp_path):
    """Plan §6 test 2 (revised per the fail-fast supersession): sync
    certification requires the wave_force_sync=True pin — the fake observes
    force_sync=True with threshold_base=None. (The v3 count-routed sync case
    wave_n_calls=300 is now the probe-region ValueError — see
    test_probe_region_unpinned_declaration_raises.)"""
    arms = {"a": _items(6, "a")}
    record: list = []
    _install_fake_judge(monkeypatch, {"a": [KEPT] * 12}, record=record)
    rep = judge_pilot_gate(
        arms,
        PROMPT,
        max_tokens=800,
        cache_dir=tmp_path / "c",
        save_raw_dir=tmp_path / "r",
        allow_subresolution_pilot=True,
        wave_force_sync=True,
    )
    assert record[0][3] is None
    assert record[0][4] is True
    assert rep.wave_transport == "sync"
    assert rep.arms["a"].transport == "sync"
    assert rep.pilot_transport == "sync"
    assert rep.passed is True


def test_transport_mismatch_fails(monkeypatch, tmp_path):
    """Plan §6 test 3 + the plan's durability pin: a realized-vs-declared
    transport mismatch is a FAIL naming both transports — never a silent
    pass. Pins the load-bearing mechanism the llm-judging.md rule 26(c)/28
    prose describes."""
    arms = {"a": _items(6, "a")}
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        {"a": [KEPT] * 12},
        routing={"path": "sync"},  # realized sync against a declared batch wave
        wave_threshold_base=0,
    )
    assert rep.verdict == "FAIL"
    mism = [f for f in rep.failures if "pilot ran sync" in f]
    assert mism, rep.failures
    assert "batch" in mism[0] and "#2152" in mism[0]
    assert "mirror the production dispatch kwargs 1:1" in mism[0]
    assert rep.pilot_transport == "sync"
    assert rep.wave_transport == "batch"


def test_unverifiable_transport_fails_when_wave_declared(monkeypatch, tmp_path):
    """Plan §6 test 4: a wave-declared pilot whose save_raw carries NO routing
    record FAILs as transport-unverifiable (fully-cached replay / legacy
    writer) — never a silent pass."""
    arms = {"a": _items(6, "a")}
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        {"a": [KEPT] * 12},
        routing=None,  # fake writes NO routing record
        wave_threshold_base=0,
    )
    assert rep.verdict == "FAIL"
    unv = [f for f in rep.failures if "UNVERIFIABLE" in f]
    assert unv, rep.failures
    assert "no routing record" in unv[0]
    assert rep.arms["a"].transport is None
    assert rep.pilot_transport is None


def test_undeclared_wave_warns_not_fails(monkeypatch, tmp_path):
    """Plan §6 test 5: legacy (undeclared-wave) callers keep today's verdicts
    byte-identically — ONE recorded UNDECLARED warning, no new FAIL, even
    with cache-served draws (n_cached > 0 FAILs only under a declaration)."""
    arms = {"a": _items(6, "a")}
    rep = _run(monkeypatch, tmp_path / "clean", arms, {"a": [KEPT] * 12})
    assert rep.passed is True
    undecl = [w for w in rep.warnings if "UNDECLARED" in w]
    assert len(undecl) == 1, rep.warnings
    assert "declare wave_n_calls" in undecl[0]
    assert rep.wave_transport is None
    assert rep.wave_routing is None

    rep2 = _run(monkeypatch, tmp_path / "cached", arms, {"a": [KEPT] * 12}, n_cached=3)
    assert rep2.passed is True  # cache-served draws FAIL only under a declared wave
    assert rep2.arms["a"].n_cached == 3


def test_wave_declaration_conflicts_raise(monkeypatch, tmp_path):
    """Plan §6 test 6: ambiguous / contradictory declarations raise BEFORE any
    judge_graded call (zero API spend)."""
    arms = {"a": _items(6, "a")}
    record: list = []
    fake = _install_fake_judge(monkeypatch, {}, record=record)
    common = dict(max_tokens=800, cache_dir=tmp_path / "c", save_raw_dir=tmp_path / "r")
    with pytest.raises(ValueError, match="EITHER the legacy pilot routing knob"):
        judge_pilot_gate(arms, PROMPT, **common, threshold_base=0, wave_threshold_base=0)
    with pytest.raises(ValueError, match="contradictory wave declaration"):
        judge_pilot_gate(arms, PROMPT, **common, wave_force_sync=True, wave_threshold_base=0)
    with pytest.raises(ValueError, match="declare wave_n_calls"):
        judge_pilot_gate(arms, PROMPT, **common, wave_threshold_base=500)
    with pytest.raises(ValueError, match="wave_n_calls=0 must be >= 1"):
        judge_pilot_gate(arms, PROMPT, **common, wave_n_calls=0)
    assert fake.call_count == 0
    assert record == []


def test_probe_region_unpinned_declaration_raises(monkeypatch, tmp_path):
    """Plan §6 test 7 (MF-1): an unpinned count-routed declaration inside the
    dispatcher's OTPM-probe region (wave_n_calls < 2*threshold_base,
    judge_dispatch.py:1652) raises fail-fast naming BOTH pins — the realized
    route there depends on a live OTPM probe no pilot can certify. The
    boundary n == 2*tb and the >= 5k default-tb common case route
    deterministic batch with no pin and no raise."""
    arms = {"a": _items(6, "a")}
    record: list = []
    fake = _install_fake_judge(monkeypatch, {"a": [KEPT] * 12}, record=record)
    with pytest.raises(ValueError, match="OTPM-PROBE-DEPENDENT") as exc:
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "f1" / "c",
            save_raw_dir=tmp_path / "f1" / "r",
            wave_n_calls=5_000,
            wave_threshold_base=20_000,
        )
    msg = str(exc.value)
    assert "wave_threshold_base=0" in msg and "wave_force_sync=True" in msg
    assert "SAME pin on the production dispatch" in msg
    with pytest.raises(ValueError, match="OTPM-PROBE-DEPENDENT"):
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "f2" / "c",
            save_raw_dir=tmp_path / "f2" / "r",
            wave_n_calls=300,  # the old count-routed-sync shape (default tb=2,000)
        )
    assert fake.call_count == 0
    assert record == []

    # Boundary n == 2*tb (no probe) and the >= 5k default-tb common case:
    # deterministic batch by count, no pin needed.
    for sub, n in (("b1", 4_000), ("b2", 44_310)):
        rep = judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / sub / "c",
            save_raw_dir=tmp_path / sub / "r",
            allow_subresolution_pilot=True,
            wave_n_calls=n,
        )
        assert rep.wave_transport == "batch"
        assert rep.passed is True


def test_api_refusal_below_threshold_passes(monkeypatch, tmp_path):
    """Plan §6 test 9: a rate strictly below the bar (1/12 ~= 8.3% < 10%)
    PASSes; the rate field still reports it."""
    arms = {"a": _items(6, "a")}
    rep = _run(monkeypatch, tmp_path, arms, {"a": [KEPT] * 11 + [API_REFUSAL_DROP]})
    assert rep.passed is True
    assert rep.arms["a"].api_refusal_rate == pytest.approx(1 / 12)
    assert rep.arms["a"].n_api_refusal == 1


def test_api_refusal_waiver_and_disable(monkeypatch, tmp_path):
    """Plan §6 test 10: the #2091-pattern waiver (reason recorded at the
    caller-site constant) turns the FAIL into a WAIVED warning; unknown
    waiver names raise; a > 1.0 threshold disables the clause (report-only,
    the #2151-era behavior) with NO under-power advisory (the >= 1
    exemption); <= 0 raises."""
    arms = {"a": _items(6, "a")}
    draws = {"a": [KEPT] * 9 + [API_REFUSAL_DROP] * 3}
    rep = _run(monkeypatch, tmp_path / "waived", arms, draws, waive_api_refusal_arms={"a"})
    assert rep.passed is True
    assert rep.arms["a"].api_refusal_waived is True
    assert any("api-refusal" in w and "WAIVED" in w for w in rep.warnings), rep.warnings

    with pytest.raises(ValueError, match="unknown arm"):
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c2",
            save_raw_dir=tmp_path / "r2",
            waive_api_refusal_arms={"nope"},
        )

    rep2 = _run(monkeypatch, tmp_path / "off", arms, draws, api_refusal_threshold=2.0)
    assert rep2.passed is True  # disabled: a rate can never reach 2.0
    assert rep2.api_refusal_threshold == 2.0
    assert not any("UNDER-POWERED for api_refusal_threshold" in w for w in rep2.warnings)

    with pytest.raises(ValueError, match=r"api_refusal_threshold.*must be > 0"):
        judge_pilot_gate(
            arms,
            PROMPT,
            max_tokens=800,
            cache_dir=tmp_path / "c3",
            save_raw_dir=tmp_path / "r3",
            api_refusal_threshold=0.0,
        )


def test_api_refusal_denominator_excludes_transport_losses(monkeypatch, tmp_path):
    """Plan §6 test 11: the rate denominator is the API-reached set
    (n_draws - n_transport_lost) — refusals stay IN it (DISTINCT from the
    parse-fail n_answered denominator), transport losses leave it."""
    arms = {"a": _items(6, "a")}
    # 12 draws: 8 kept + 2 transport-lost + 2 api-refusal -> rate 2/10 = 20%.
    draws = {"a": [KEPT] * 8 + [TRANSPORT] * 2 + [API_REFUSAL_DROP] * 2}
    rep = _run(monkeypatch, tmp_path, arms, draws)
    assert rep.arms["a"].api_refusal_rate == pytest.approx(2 / 10)  # NOT 2/12
    assert rep.verdict == "FAIL"  # 20% >= 10%
    assert any("api-refusal rate" in f for f in rep.failures), rep.failures


def test_report_new_fields_serialize_and_pilot_transport_precedence(monkeypatch, tmp_path):
    """Plan §6 test 12: the #2152 fields ride to_json(), and pilot_transport
    aggregates realized per-arm routes with explicit precedence — any arm
    None -> None; unique -> that route; else "mixed" — under a declared AND
    an undeclared wave."""
    arms = {"a": _items(6, "a")}
    rep = _run(monkeypatch, tmp_path / "ser", arms, {"a": [KEPT] * 12}, wave_threshold_base=0)
    d = rep.to_json()
    assert d["wave_transport"] == "batch"
    assert d["pilot_transport"] == "batch"
    assert d["api_refusal_threshold"] == pytest.approx(0.10)
    assert d["wave_routing"]["path"] == "batch"
    assert d["arms"]["a"]["transport"] == "batch"
    assert d["arms"]["a"]["api_refusal_rate"] == 0.0
    assert d["arms"]["a"]["api_refusal_waived"] is False
    assert d["arms"]["a"]["n_cached"] == 0

    # pilot_transport precedence table: {arm -> routing record} -> aggregate.
    cases = [
        ({"a": None}, None),
        ({"a": {"path": "sync"}, "b": None}, None),
        ({"a": {"path": "sync"}, "b": {"path": "batch"}}, "mixed"),
        ({"a": {"path": "sync"}, "b": {"path": "batch"}, "c": None}, None),
    ]
    for i, (by_arm, expected) in enumerate(cases):
        arms_i = {arm: _items(6, arm) for arm in by_arm}
        draws_i = {arm: [KEPT] * 12 for arm in by_arm}
        # Undeclared wave: aggregate recorded, verdict unchanged (PASS).
        rep_u = _run(monkeypatch, tmp_path / f"u{i}", arms_i, draws_i, routing_by_arm=by_arm)
        assert rep_u.pilot_transport == expected
        assert rep_u.passed is True
        # Declared batch wave: same aggregate; every case carries a None or a
        # sync arm against the batch wave, so the parity clauses FAIL loudly.
        rep_d = _run(
            monkeypatch,
            tmp_path / f"d{i}",
            arms_i,
            draws_i,
            routing_by_arm=by_arm,
            wave_threshold_base=0,
        )
        assert rep_d.pilot_transport == expected
        assert rep_d.passed is False


def test_judge_graded_force_sync_passthrough(monkeypatch, tmp_path):
    """Plan §6 test 13 (#906 real-body): the REAL judge_graded forwards
    force_sync=True to judge_completions_batch (autospec'd API boundary) and
    OMITS the kwarg entirely by default — legacy callers byte-identical."""

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

    items = _items(3, "x")
    res = graded_judge.judge_graded(
        items,
        PROMPT,
        n_draws=2,
        cache_dir=tmp_path / "c1",
        save_raw=tmp_path / "r1.json",
        force_sync=True,
    )
    assert res.n_total_draws == 6
    assert spec.call_args.kwargs["force_sync"] is True

    graded_judge.judge_graded(
        items,
        PROMPT,
        n_draws=2,
        cache_dir=tmp_path / "c2",
        save_raw=tmp_path / "r2.json",
    )
    assert "force_sync" not in spec.call_args.kwargs
    assert "threshold_base" not in spec.call_args.kwargs


def test_mixed_cache_pilot_fails_when_wave_declared(monkeypatch, tmp_path):
    """Plan §6 test 14 (MF-2): cache-served draws void transport certification
    EVEN with a truthful matching routing record — they carry no routing
    provenance and are refusal-free by construction (the #2151 cache
    PUT-SKIP), diluting the clause-(d) api-refusal rate strictly toward
    PASS."""
    arms = {"a": _items(6, "a")}
    rep = _run(
        monkeypatch,
        tmp_path,
        arms,
        {"a": [KEPT] * 12},
        routing={"path": "batch"},  # truthful record for the dispatched remainder
        n_cached=3,
        n_submitted=9,
        wave_threshold_base=0,
    )
    assert rep.verdict == "FAIL"
    cache = [f for f in rep.failures if "cache-served" in f]
    assert cache, rep.failures
    assert "UNVERIFIABLE" in cache[0]
    assert "refusal-free by construction" in cache[0]
    assert "rule 24(ii)" in cache[0]
    assert rep.arms["a"].n_cached == 3
