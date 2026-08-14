"""Tests for the ensemble durable-verdict presence predicate (#1149).

``ensemble_verdicts_present`` is the mechanical form of items 1 + 3 of the
/issue Step 5b durable-verdict-first rule: before any reviewer no-show
decision, the orchestrator asks whether the round's expected verdict markers
exist on events.jsonl and whether each carries a parseable ``Verdict:`` field
(incident #810 r4: a posted ``epm:code-review v4`` PASS was misread as a
total no-show after the reviewer's summary turn died, and a unilateral FAIL
was adopted from the surviving Codex verdict alone). Fixtures replay real
note shapes from task #1092 (a reconcile marker whose top-level ``version``
is 1 while its head sentinel says ``v5`` — the version field is
round-meaningless on reconcile markers) and task #1090 (the terse
sentinel-less Claude critique with no ``**Verdict:**`` line; the
bold-wrapped ``**Verdict: PASS**`` Codex-twin shape).
"""

from __future__ import annotations

import pytest

import explore_persona_space.task_workflow as tw


def _ev(ts: str, kind: str, note: str = "", version: int = 1, by: str = "test") -> dict:
    return {"ts": ts, "kind": kind, "version": version, "by": by, "note": note}


# Real #1092 reconcile shape (subset, verbatim head): top-level version 1,
# sentinel + **Round:** both naming round 5.
_RECONCILE_1092 = (
    "<!-- epm:review-reconcile v5 -->\n\n"
    "## Reconciler Verdict — FAIL\n\n"
    "**Role under adjudication:** code-reviewer\n"
    "**Round:** 5\n"
    "**Verdict:** FAIL\n"
    "**Claude verdict:** PASS\n"
    "**Codex verdict:** FAIL\n"
)

# Real #1090 terse Claude-critique shape: no sentinel, no **Verdict:** line.
_TERSE_1090 = (
    "Round 1: PASS — v4 body is structurally complete, register-clean, and "
    "every load-bearing number reconciles against eval_results/issue_1090 "
    "ground truth"
)

# Reconcile note with NO head sentinel — round carried only by **Round:**.
_RECONCILE_NO_SENTINEL = (
    "## Reconciler Verdict — PASS\n\n"
    "**Role under adjudication:** interpretation-critic\n"
    "**Round:** 3\n"
    "**Verdict:** PASS\n"
)

# Reconcile note with NO **Role under adjudication:** field.
_RECONCILE_NO_ROLE = "<!-- epm:review-reconcile v2 -->\n\n**Round:** 2\n**Verdict:** PASS\n"

_ABSENT = {"present": False, "verdict": None, "ts": None}


def test_both_present_verdicts_parsed():
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v2 -->\n\n**Verdict:** FAIL\n\n### Blockers\n- b1",
            version=2,
        ),
        _ev(
            "2026-07-07T10:01:00Z",
            "epm:code-review-codex",
            "<!-- epm:code-review-codex v2 -->\n\n**Verdict:** PASS\n",
            version=2,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review", "epm:code-review-codex"], 2)
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "FAIL",
        "ts": "2026-07-07T10:00:00Z",
    }
    assert out["epm:code-review-codex"] == {
        "present": True,
        "verdict": "PASS",
        "ts": "2026-07-07T10:01:00Z",
    }


def test_one_missing():
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v2 -->\n\n**Verdict:** PASS\n",
            version=2,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review", "epm:code-review-codex"], 2)
    assert out["epm:code-review"]["present"] is True
    assert out["epm:code-review-codex"] == _ABSENT


def test_wrong_round_version():
    # The stale-prior-round trap of rule item 2: markers exist only at round 3.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v3 -->\n\n**Verdict:** PASS\n",
            version=3,
        ),
        _ev(
            "2026-07-07T10:01:00Z",
            "epm:code-review-codex",
            "<!-- epm:code-review-codex v3 -->\n\n**Verdict:** PASS\n",
            version=3,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review", "epm:code-review-codex"], 4)
    assert out["epm:code-review"] == _ABSENT
    assert out["epm:code-review-codex"] == _ABSENT


def test_malformed_note_no_verdict_line():
    # Real #1090 terse shape: present-but-malformed route — NEVER a no-show,
    # and NEVER a fabricated verdict token (forbids a strict-Verdict
    # presence test).
    events = [_ev("2026-07-07T06:26:20Z", "epm:clean-result-critique", _TERSE_1090, version=1)]
    out = tw.ensemble_verdicts_present(events, ["epm:clean-result-critique"], 1)
    assert out["epm:clean-result-critique"] == {
        "present": True,
        "verdict": None,
        "ts": "2026-07-07T06:26:20Z",
    }


def test_reconcile_role_scoped():
    # Real #1092 shape: version field 1, sentinel v5 — also exercises the
    # sentinel round read and the multi-word role-field parse.
    events = [_ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_1092, version=1)]
    kinds = ["epm:review-reconcile"]
    scoped = tw.ensemble_verdicts_present(events, kinds, 5, reconcile_role="code-reviewer")
    assert scoped["epm:review-reconcile"] == {
        "present": True,
        "verdict": "FAIL",
        "ts": "2026-07-07T13:20:13Z",
    }
    wrong_role = tw.ensemble_verdicts_present(
        events, kinds, 5, reconcile_role="interpretation-critic"
    )
    assert wrong_role["epm:review-reconcile"] == _ABSENT
    unscoped = tw.ensemble_verdicts_present(events, kinds, 5)
    assert unscoped["epm:review-reconcile"]["present"] is True


def test_empty_events():
    out = tw.ensemble_verdicts_present([], ["epm:code-review", "epm:review-reconcile"], 1)
    assert out["epm:code-review"] == _ABSENT
    assert out["epm:review-reconcile"] == _ABSENT


def test_sentinel_fallback_on_version_divergence():
    # Defaulted re-spawn auto-bump (#480 class): version landed at max+1=5
    # while the sentinel names the true round 4.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v4 -->\n\n**Verdict:** PASS\n",
            version=5,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review"], 4)
    assert out["epm:code-review"]["present"] is True
    assert out["epm:code-review"]["verdict"] == "PASS"


def test_sentinel_authoritative_suppresses_version_match():
    # Same event as above queried at its VERSION number (5): the sentinel
    # names round 4, so the version-field match is suppressed — a
    # stale/drifted version never reads as this round's verdict.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v4 -->\n\n**Verdict:** PASS\n",
            version=5,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review"], 5)
    assert out["epm:code-review"] == _ABSENT


def test_respawn_duplicate_latest_wins():
    # A re-spawn posts at the SAME v<n> (rule item 4): the chronologically
    # latest match carries the verdict + ts.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v3 -->\n\n**Verdict:** FAIL\n",
            version=3,
        ),
        _ev(
            "2026-07-07T11:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v3 -->\n\n**Verdict:** PASS\n",
            version=3,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review"], 3)
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "PASS",
        "ts": "2026-07-07T11:00:00Z",
    }


def test_versionless_single_pass_site():
    # epm:followup-value-critique[-codex] is the single-pass site (always
    # effectively round 1); hyphenated verdict token parses whole.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:followup-value-critique-codex",
            "**Verdict:** not-redundant\n**Proposals screened:** 3",
            version=1,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:followup-value-critique-codex"], 1)
    assert out["epm:followup-value-critique-codex"]["present"] is True
    assert out["epm:followup-value-critique-codex"]["verdict"] == "not-redundant"


def test_non_verdict_kinds_ignored():
    # Noise rows at the queried version never satisfy a queried kind —
    # kind equality is exact.
    events = [
        _ev("2026-07-07T10:00:00Z", "epm:progress", "stage-dispatch stage=review round=2", 2),
        _ev("2026-07-07T10:01:00Z", "epm:codex-task-completed", "Codex job phase=done", 2),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:code-review", "epm:code-review-codex"], 2)
    assert out["epm:code-review"] == _ABSENT
    assert out["epm:code-review-codex"] == _ABSENT


def test_codex_bold_wrapped_verdict():
    # The dominant real Codex-twin shape (#1090): bold wraps field AND value.
    events = [
        _ev(
            "2026-07-07T10:00:00Z",
            "epm:clean-result-critique-codex",
            "<!-- epm:clean-result-critique-codex v1 -->\n**Verdict: PASS**\n",
            version=1,
        ),
    ]
    out = tw.ensemble_verdicts_present(events, ["epm:clean-result-critique-codex"], 1)
    assert out["epm:clean-result-critique-codex"]["verdict"] == "PASS"


def test_reconcile_version_field_never_matches():
    # Critic-ensemble amendment pin (a): a reconcile whose version field
    # equals the queried round but whose sentinel names a DIFFERENT round is
    # NOT matched — the reconcile version field is round-meaningless.
    events = [_ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_1092, version=4)]
    kinds = ["epm:review-reconcile"]
    at_version = tw.ensemble_verdicts_present(events, kinds, 4, reconcile_role="code-reviewer")
    assert at_version["epm:review-reconcile"] == _ABSENT
    at_sentinel = tw.ensemble_verdicts_present(events, kinds, 5, reconcile_role="code-reviewer")
    assert at_sentinel["epm:review-reconcile"]["present"] is True


def test_reconcile_round_field_fallback():
    # Critic-ensemble amendment pin (b): sentinel absent — the reconcile is
    # matched via the note's **Round:** field, never via version.
    events = [_ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_NO_SENTINEL, 1)]
    kinds = ["epm:review-reconcile"]
    by_round_field = tw.ensemble_verdicts_present(
        events, kinds, 3, reconcile_role="interpretation-critic"
    )
    assert by_round_field["epm:review-reconcile"]["present"] is True
    assert by_round_field["epm:review-reconcile"]["verdict"] == "PASS"
    by_version = tw.ensemble_verdicts_present(events, kinds, 1)
    assert by_version["epm:review-reconcile"] == _ABSENT


def test_reconcile_missing_role_field():
    # Deliberate, docstring-pinned behavior: a role-scoped query against a
    # reconcile note with NO **Role under adjudication:** field reads absent
    # (fail toward the rule's output-file probe, never adopt an
    # unattributable adjudication); an unscoped query still sees it.
    events = [_ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_NO_ROLE, version=1)]
    kinds = ["epm:review-reconcile"]
    scoped = tw.ensemble_verdicts_present(events, kinds, 2, reconcile_role="code-reviewer")
    assert scoped["epm:review-reconcile"] == _ABSENT
    unscoped = tw.ensemble_verdicts_present(events, kinds, 2)
    assert unscoped["epm:review-reconcile"]["present"] is True


def test_bare_str_kinds_raises():
    # Reviewer-round pin: a bare string for `kinds` would iterate
    # per-character and mechanically produce false no-shows — reject loudly.
    with pytest.raises(TypeError, match="bare str"):
        tw.ensemble_verdicts_present([], "epm:code-review", 1)


# --------------------------------------------------------------------------
# #2136 freshness anchor (`since_ts` + `review_round_anchor_ts`): the anchor
# gates the sentinel-less version-fallback branch ONLY — a sentinel-bearing
# round-exact match is NEVER time-gated (killed-session reuse), and
# `since_ts=None` reproduces the pre-#2136 behavior exactly.
# --------------------------------------------------------------------------

_CODE_REVIEW_OPENERS = ("epm:experiment-implementation", "epm:results")

# The #1336 incident shape: a round-3 verdict posted sentinel-LESS at the
# auto-bumped version 4 (2026-08-02), then a round-4 implementer marker
# (2026-08-04) opens the next review round.
_STALE_FALLBACK_EVENTS = [
    _ev(
        "2026-08-02T10:00:00Z",
        "epm:code-review",
        "**Verdict:** PASS\n\nRound-3 review posted without a head sentinel.",
        version=4,
    ),
    _ev(
        "2026-08-04T09:00:00Z",
        "epm:results",
        "<!-- epm:results v1 -->\n\n## Completion Report\nround-4 implementation",
        version=1,
    ),
]


def test_stale_prior_round_version_fallback_suppressed_by_anchor():
    # The body's required "two rounds of markers" case (shape1, #1336): the
    # round-3 sentinel-less PASS at version 4 must NOT answer a round-4
    # query when the anchor names the round-4 opener.
    anchor = tw.review_round_anchor_ts(_STALE_FALLBACK_EVENTS, opening_kinds=_CODE_REVIEW_OPENERS)
    assert anchor == "2026-08-04T09:00:00Z"
    out = tw.ensemble_verdicts_present(
        _STALE_FALLBACK_EVENTS, ["epm:code-review"], 4, since_ts=anchor
    )
    assert out["epm:code-review"] == _ABSENT


def test_stale_prior_round_matches_without_anchor():
    # Pins the opt-in default as deliberate: `since_ts` omitted reproduces
    # today's (buggy-but-unchanged) fallback behavior byte-for-byte.
    out = tw.ensemble_verdicts_present(_STALE_FALLBACK_EVENTS, ["epm:code-review"], 4)
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "PASS",
        "ts": "2026-08-02T10:00:00Z",
    }


def test_anchor_never_suppresses_sentinel_bearing_match():
    # Killed-session reuse (the §3 load-bearing constraint): a sentinel v4
    # verdict OLDER than the anchor still matches — sentinel-bearing
    # round-exact matches are NEVER time-gated.
    events = [
        _ev(
            "2026-08-02T10:00:00Z",
            "epm:code-review",
            "<!-- epm:code-review v4 -->\n\n**Verdict:** PASS\n",
            version=6,
        ),
        _ev(
            "2026-08-04T09:00:00Z",
            "epm:results",
            "<!-- epm:results v1 -->\nround-4 implementation",
            version=1,
        ),
    ]
    out = tw.ensemble_verdicts_present(
        events, ["epm:code-review"], 4, since_ts="2026-08-04T09:00:00Z"
    )
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "PASS",
        "ts": "2026-08-02T10:00:00Z",
    }


def test_legitimate_fallback_match_survives_active_anchor():
    # Positive control: a legitimate round-1 sentinel-less verdict at
    # version 1, posted AFTER the round-1 opener, still matches under an
    # ACTIVE anchor. Guards against an inverted comparison (`<` for `>`)
    # or an anchor that suppresses every fallback match — either would
    # pass the suppression tests while converting first-round
    # sentinel-less verdicts fleet-wide into false no-shows.
    events = [
        _ev(
            "2026-08-04T09:00:00Z",
            "epm:experiment-implementation",
            "<!-- epm:experiment-implementation v1 -->\nround-1 implementation",
            version=1,
        ),
        _ev(
            "2026-08-04T11:00:00Z",
            "epm:code-review",
            "**Verdict:** FAIL\n\n### Blockers\n- b1",
            version=1,
        ),
    ]
    anchor = tw.review_round_anchor_ts(events, opening_kinds=_CODE_REVIEW_OPENERS)
    assert anchor == "2026-08-04T09:00:00Z"
    out = tw.ensemble_verdicts_present(events, ["epm:code-review"], 1, since_ts=anchor)
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "FAIL",
        "ts": "2026-08-04T11:00:00Z",
    }


def test_shared_opener_earlier_round_suppressed_by_newest_rule():
    # Condition (ii), independent of (i): two sentinel-less verdicts after
    # ONE shared opener (the 23.8% clean-result sub-shape a timestamp
    # anchor alone cannot separate). Queried at round 4, the stale round-3
    # marker (drifted version 4) postdates the opener but is NOT the
    # newest same-kind marker after it — suppressed.
    events = [
        _ev(
            "2026-08-01T09:00:00Z",
            "epm:interpretation",
            "<!-- epm:interpretation v3 -->\ninterpretation",
            version=3,
        ),
        _ev(
            "2026-08-01T10:00:00Z",
            "epm:clean-result-critique",
            "**Verdict:** PASS\n\nround-3 critique",
            version=4,
        ),
        _ev(
            "2026-08-01T11:00:00Z",
            "epm:clean-result-critique",
            "**Verdict:** FAIL\n\nround-4 critique",
            version=5,
        ),
    ]
    anchor = tw.review_round_anchor_ts(events, opening_kinds=("epm:interpretation", "epm:analysis"))
    assert anchor == "2026-08-01T09:00:00Z"
    out = tw.ensemble_verdicts_present(events, ["epm:clean-result-critique"], 4, since_ts=anchor)
    assert out["epm:clean-result-critique"] == _ABSENT


def test_anchor_suppresses_on_unparseable_ts():
    # Fail-safe direction: a fallback candidate with a missing/garbage `ts`
    # under an active anchor is suppressed (routes to the rule's item-2
    # output-file probe) — and so is every fallback match under a garbage
    # `since_ts`.
    events = [
        _ev("not-a-timestamp", "epm:code-review", "**Verdict:** PASS\n", version=4),
    ]
    out = tw.ensemble_verdicts_present(
        events, ["epm:code-review"], 4, since_ts="2026-08-04T09:00:00Z"
    )
    assert out["epm:code-review"] == _ABSENT
    good_ts = [
        _ev("2026-08-04T10:00:00Z", "epm:code-review", "**Verdict:** PASS\n", version=4),
    ]
    garbage_anchor = tw.ensemble_verdicts_present(
        good_ts, ["epm:code-review"], 4, since_ts="garbage"
    )
    assert garbage_anchor["epm:code-review"] == _ABSENT


def test_anchor_inert_for_reconcile_kind():
    # The reconcile matcher never reaches the version fallback (sentinel,
    # else **Round:** field), so an anchor NEWER than the reconcile leaves
    # role-scoped resolution unchanged — for both reconcile note shapes.
    sentinel_events = [
        _ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_1092, version=1)
    ]
    out = tw.ensemble_verdicts_present(
        sentinel_events,
        ["epm:review-reconcile"],
        5,
        reconcile_role="code-reviewer",
        since_ts="2026-08-04T09:00:00Z",
    )
    assert out["epm:review-reconcile"]["present"] is True
    assert out["epm:review-reconcile"]["verdict"] == "FAIL"
    round_field_events = [
        _ev("2026-07-07T13:20:13Z", "epm:review-reconcile", _RECONCILE_NO_SENTINEL, version=1)
    ]
    by_round_field = tw.ensemble_verdicts_present(
        round_field_events,
        ["epm:review-reconcile"],
        3,
        reconcile_role="interpretation-critic",
        since_ts="2026-08-04T09:00:00Z",
    )
    assert by_round_field["epm:review-reconcile"]["present"] is True


def test_respawn_duplicate_still_latest_wins_under_anchor():
    # The same-round re-spawn shape on the FALLBACK path with an active
    # anchor: two sentinel-less markers at the same version=round, both
    # after the opener — the newer one wins, matching the pre-existing
    # latest-wins contract (guards the §0.0 kill criterion on condition
    # (ii): the newest-rule must not suppress a legitimate re-spawn).
    events = [
        _ev(
            "2026-08-04T09:00:00Z",
            "epm:results",
            "<!-- epm:results v1 -->\nround-3 implementation",
            version=1,
        ),
        _ev("2026-08-04T10:00:00Z", "epm:code-review", "**Verdict:** FAIL\n", version=3),
        _ev("2026-08-04T11:00:00Z", "epm:code-review", "**Verdict:** PASS\n", version=3),
    ]
    anchor = tw.review_round_anchor_ts(events, opening_kinds=_CODE_REVIEW_OPENERS)
    out = tw.ensemble_verdicts_present(events, ["epm:code-review"], 3, since_ts=anchor)
    assert out["epm:code-review"] == {
        "present": True,
        "verdict": "PASS",
        "ts": "2026-08-04T11:00:00Z",
    }


def test_review_round_anchor_ts_picks_last_opener():
    # Helper unit: chronologically LAST ts across a MIXED implementer-kind
    # sequence; None on an empty or opener-less log; `opening_kinds` is a
    # REQUIRED keyword (TypeError when omitted).
    mixed = [
        _ev("2026-08-01T09:00:00Z", "epm:experiment-implementation", "impl r1", version=1),
        _ev("2026-08-02T09:00:00Z", "epm:results", "results r1", version=1),
        _ev("2026-08-03T09:00:00Z", "epm:experiment-implementation", "impl r2", version=2),
        _ev("2026-08-03T10:00:00Z", "epm:progress", "noise", version=1),
    ]
    assert (
        tw.review_round_anchor_ts(mixed, opening_kinds=_CODE_REVIEW_OPENERS)
        == "2026-08-03T09:00:00Z"
    )
    assert tw.review_round_anchor_ts([], opening_kinds=_CODE_REVIEW_OPENERS) is None
    no_openers = [_ev("2026-08-01T09:00:00Z", "epm:progress", "noise", version=1)]
    assert tw.review_round_anchor_ts(no_openers, opening_kinds=_CODE_REVIEW_OPENERS) is None
    with pytest.raises(TypeError):
        tw.review_round_anchor_ts(mixed)  # opening_kinds is required
    with pytest.raises(TypeError, match="bare str"):
        tw.review_round_anchor_ts(mixed, opening_kinds="epm:results")
