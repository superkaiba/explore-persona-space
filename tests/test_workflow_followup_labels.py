"""Label-grouped same-issue follow-up dispatch helpers (task #894).

Pins ``task_workflow.parse_followup_note_field`` / ``followup_label_groups``
/ ``unrun_followup_labels`` / ``executing_followup_label`` /
``followup_retro_close_evidence`` — the SINGLE implementation of the
"scan ALL ``epm:followup-scope`` entries grouped by ``followup_label``"
dispatch predicate consumed by `/issue` Step 0, the Step 9b loop, the
resume table, and ``scripts/autonomous_session_watch.py``.

Fixture notes are copied from the REAL marker shapes on record (#763 /
#658 / #537 / #552 / #664 / #685 / #837 §4c) so every historical note
format stays pinned. The corpus-replay test at the bottom additionally
replays every ``tasks/*/*/events.jsonl`` in the checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.task_workflow import (
    executing_followup_label,
    followup_label_groups,
    followup_retro_close_evidence,
    parse_followup_note_field,
    unrun_followup_labels,
)


def _ev(kind: str, ts: str, note: str, version: int = 1) -> dict:
    """Minimal event row matching the task.py writer shape."""
    return {"ts": ts, "kind": kind, "version": version, "note": note, "by": "test"}


def _scope(ts: str, note: str, version: int = 1) -> dict:
    return _ev("epm:followup-scope", ts, note, version)


def _run(ts: str, note: str, version: int = 1) -> dict:
    return _ev("epm:same-issue-followup-run", ts, note, version)


# ─── 1. the #763 stranding shape (the driving incident) ─────────────────────


def test_763_shape_returns_earlier_queued_label_unrun():
    # Replays #763's real events: scope v1 (user-chat, armed 2026-06-30,
    # UNRUN), scope v2 (proposer-9b-cheap, armed 2026-07-02), run marker for
    # v2 (SINGLE-LINE space-separated note — the real #763 run shape). The
    # old highest-version read matched v2's run marker and concluded "no
    # unrun scope"; the label-grouped scan must surface v1.
    events = [
        _scope(
            "2026-06-30T22:10:33Z",
            "source: user-chat\nfollowup_label: neutral-contrast-and-cofit\n"
            "question_relation: same\nest_gpu_hours: 3",
            version=1,
        ),
        _scope(
            "2026-07-02T10:46:52Z",
            "followup_label: deception-rubric-reanchor\nsource: proposer-9b-cheap\n"
            "question_relation: same\nest_gpu_hours: 2",
            version=2,
        ),
        _run(
            "2026-07-02T21:45:13Z",
            "followup_label: deception-rubric-reanchor source: proposer-9b-cheap "
            "round: 1 outcome: instrument-recovery confirmed",
        ),
    ]
    unrun = unrun_followup_labels(events)
    assert [g["followup_label"] for g in unrun] == ["neutral-contrast-and-cofit"]
    assert unrun[0]["user_initiated"] is True
    assert unrun[0]["dispatchable"] is True
    assert unrun[0]["source"] == "user-chat"


# ─── 2. the #658 within-label correction chain (must keep working) ──────────


def _pv_scope(ts: str, version: int, est: int) -> dict:
    return _scope(
        ts,
        f"followup_label: persona-vectors-style-rb\nsource: user-chat\n"
        f"question_relation: same\ngpu_hours_est: {est}",
        version=version,
    )


def test_658_correction_chain_one_authoritative_entry():
    # Three same-label entries (the real #658 v3→v7 chain, condensed to
    # v3/v5/v7): ONE group, authoritative = the latest-(ts, version) entry,
    # armed_ts = the FIRST entry's ts (a later correction never re-queues).
    events = [
        _pv_scope("2026-06-29T08:57:29Z", 3, 3),
        _pv_scope("2026-06-29T09:01:43Z", 5, 6),
        _pv_scope("2026-06-29T09:06:10Z", 7, 7),
    ]
    groups = followup_label_groups(events)
    assert len(groups) == 1
    group = groups[0]
    assert group["followup_label"] == "persona-vectors-style-rb"
    assert group["n_entries"] == 3
    assert group["authoritative"]["version"] == 7
    assert group["armed_ts"] == "2026-06-29T08:57:29Z"
    assert [g["followup_label"] for g in unrun_followup_labels(events)] == [
        "persona-vectors-style-rb"
    ]
    # A matching run marker closes ALL entries of the label.
    closed = [
        *events,
        _run(
            "2026-06-29T12:00:00Z",
            "followup_label: persona-vectors-style-rb\nsource: user-chat\nround: 1",
        ),
    ]
    assert unrun_followup_labels(closed) == []


# ─── 3. dispatch-queue ordering ──────────────────────────────────────────────


def test_priority_user_initiated_before_older_proposer():
    proposer_old = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: proposer-round-a\nsource: proposer-9b-cheap",
        version=1,
    )
    proposer_older = _scope(
        "2026-06-09T00:00:00Z",
        "followup_label: proposer-round-b\nsource: proposer-9b",
        version=2,
    )
    user_newer = _scope(
        "2026-06-11T00:00:00Z",
        "followup_label: user-round\nsource: user-chat",
        version=3,
    )
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, user_newer])]
    assert order == ["user-round", "proposer-round-a"]
    # Two proposer labels → oldest armed_ts first.
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, proposer_older])]
    assert order == ["proposer-round-b", "proposer-round-a"]
    # step-10b-pick counts as user-initiated.
    pick = _scope(
        "2026-06-12T00:00:00Z",
        "followup_label: picked-round\nsource: step-10b-pick",
        version=4,
    )
    order = [g["followup_label"] for g in unrun_followup_labels([proposer_old, pick])]
    assert order == ["picked-round", "proposer-round-a"]


# ─── 4. every historical note format parses ──────────────────────────────────


def test_label_parse_all_note_formats():
    # dash-bullet (#658 v1)
    assert (
        parse_followup_note_field(
            "Genre-generalization follow-up: generic (UltraChat) vs "
            "misalignment-specific (Betley) queries.\n\n"
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            "followup_label",
        )
        == "genre-generalization-ultrachat"
    )
    # bare-colon (#763)
    assert (
        parse_followup_note_field(
            "followup_label: neutral-contrast-and-cofit\nsource: user-chat",
            "followup_label",
        )
        == "neutral-contrast-and-cofit"
    )
    # bare-EQUALS — the #537/#552 run-marker form (15 historical markers)
    assert (
        parse_followup_note_field(
            "followup_label=seed2-marker-fact-replication source=proposer-9b "
            "round=1 outcome='REPRODUCES — all registered reads PASS'",
            "followup_label",
        )
        == "seed2-marker-fact-replication"
    )
    # bold (#837 §4c / #685 v1)
    assert (
        parse_followup_note_field(
            "<!-- epm:followup-scope v1 -->\n"
            "**followup_label:** full-judge-coverage-and-syco-opinion\n"
            "**source:** user-chat",
            "followup_label",
        )
        == "full-judge-coverage-and-syco-opinion"
    )
    # star-bullet
    assert (
        parse_followup_note_field(
            "* followup_label: star-bullet-round\n* source: user-chat",
            "followup_label",
        )
        == "star-bullet-round"
    )
    # backtick-wrapped bold value (#664)
    assert (
        parse_followup_note_field(
            "**followup_label:** `em-provenance-robustness`\n**source:** user-chat",
            "followup_label",
        )
        == "em-provenance-robustness"
    )
    # COMBINED bullet+bold: a dash-bullet wrapping a bold field. Corpus-clean
    # today but plausible future drift (r2 review Minor); a sequential
    # strip()/lstrip("-*") chain stops at the space after "-" and misses the
    # bold marker behind it.
    assert (
        parse_followup_note_field(
            "- **followup_label:** combined-bullet-bold-round\n- **source:** user-chat",
            "followup_label",
        )
        == "combined-bullet-bold-round"
    )
    # star-bullet + bold sibling of the same combined form.
    assert (
        parse_followup_note_field(
            "* **followup_label:** star-bullet-bold-round",
            "followup_label",
        )
        == "star-bullet-bold-round"
    )
    # single-line run-marker form: first-token rule (kebab-slug labels carry
    # no whitespace)
    assert (
        parse_followup_note_field(
            "followup_label: deception-rubric-reanchor source: proposer-9b-cheap "
            "round: 1 outcome: ...",
            "followup_label",
        )
        == "deception-rubric-reanchor"
    )
    # first-hit-wins: #763 v2 embeds a SECOND bold label deep inside its
    # verbatim-proposal section — the top-of-note canonical line is hit first.
    assert (
        parse_followup_note_field(
            "followup_label: deception-rubric-reanchor\nsource: proposer-9b-cheap\n"
            "spec: verbatim proposal follows\n"
            "**followup_label:** some-embedded-proposal-label\n",
            "followup_label",
        )
        == "deception-rubric-reanchor"
    )
    # absent / empty → None
    assert parse_followup_note_field("no label here", "followup_label") is None
    assert parse_followup_note_field("followup_label:", "followup_label") is None
    assert parse_followup_note_field("", "followup_label") is None


# ─── 5. unlabeled corrections vs distinct unlabeled follow-ups ───────────────


def test_unlabeled_correction_inherits_previous_label():
    # The REAL #658 v2 shape: an unlabeled note carrying the literal word
    # CORRECTION attributes to the immediately-preceding label; a matching
    # run marker then closes the whole group.
    events = [
        _scope(
            "2026-06-25T08:42:56Z",
            "Genre-generalization follow-up.\n\n"
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-25T09:07:55Z",
            "CORRECTION to the earlier epm:followup-scope "
            "(genre-generalization-ultrachat): the gating is now\n"
            'UNCONDITIONAL, superseding the prior "auto_run: NO" line.',
            version=2,
        ),
    ]
    groups = followup_label_groups(events)
    assert len(groups) == 1
    group = groups[0]
    assert group["followup_label"] == "genre-generalization-ultrachat"
    assert group["n_entries"] == 2
    # The correction's content IS the label's authoritative entry.
    assert group["authoritative"]["version"] == 2
    assert group["label_parse"] == "inherited-from-previous"
    assert group["dispatchable"] is True
    closed = [
        *events,
        _run(
            "2026-06-28T05:12:17Z",
            "followup_label: genre-generalization-ultrachat\nsource: user-chat\nround: 3",
        ),
    ]
    assert unrun_followup_labels(closed) == []


def test_unlabeled_noncorrection_scope_is_distinct_group():
    # The REAL #685 shape: labeled v1 + a label-less v2 with NO correction
    # signal (a distinct user-chat follow-up). v2 must NOT merge into v1's
    # label — it becomes its own pseudo-ts group, surfaced but never
    # dispatched (Alt-Claude MF2 / Alt-Codex MF2).
    events = [
        _scope(
            "2026-06-27T20:24:25Z",
            "<!-- epm:followup-scope v1 -->\n"
            "**followup_label:** full-judge-coverage-and-syco-opinion\n"
            "**source:** user-chat\n**question_relation:** same",
            version=1,
        ),
        _scope(
            "2026-06-28T09:27:49Z",
            "Follow-up scope (source: user-chat) — sharpen the Δ-vs-behavior-vector "
            "projection result.\n\n**Question (same as parent):** does each behavior "
            "shift track its own direction?",
            version=2,
        ),
    ]
    groups = followup_label_groups(events)
    assert [g["followup_label"] for g in groups] == [
        "full-judge-coverage-and-syco-opinion",
        "unlabeled-2026-06-28T09:27:49Z",
    ]
    labeled, pseudo = groups
    assert labeled["authoritative"]["version"] == 1  # v2 never merged in
    assert labeled["user_initiated"] is True
    assert pseudo["label_parse"] == "pseudo-ts"
    assert pseudo["dispatchable"] is False


# ─── 6. leading unlabeled scope → non-dispatchable pseudo-label ──────────────


def test_leading_unlabeled_scope_pseudo_label_nondispatchable():
    sole = _scope("2026-06-28T09:27:49Z", "malformed scope note with no fields")
    unrun = unrun_followup_labels([sole])
    assert len(unrun) == 1
    group = unrun[0]
    assert group["followup_label"] == "unlabeled-2026-06-28T09:27:49Z"
    assert group["label_parse"] == "pseudo-ts"
    assert group["dispatchable"] is False
    # A run marker carrying the pseudo-label VERBATIM (the retro-close path)
    # still closes it.
    closed = [
        sole,
        _run(
            "2026-06-29T00:00:00Z",
            "followup_label: unlabeled-2026-06-28T09:27:49Z source: unknown round: 1 "
            "outcome: retroactive-close — repaired",
        ),
    ]
    assert unrun_followup_labels(closed) == []


def test_pseudo_founded_group_stays_nondispatchable_after_inherited_correction():
    # r2 Major 2 (persisted concern `pseudo-label-inherit-dispatchable`): an
    # unlabeled CORRECTION following a pseudo-founded group inherits into it
    # (raising the group's authoritative entry) but must NOT flip it
    # dispatchable — the group's label is still the malformed
    # `unlabeled-<ts>` (kebab-slug contract violation), a repair item until
    # re-posted with a proper `followup_label`. Dispatchability is
    # FOUNDING-based, not last-entry-parse-mode-based.
    events = [
        _scope("2026-06-28T09:00:00Z", "malformed scope note with no fields", version=1),
        _scope(
            "2026-06-28T10:00:00Z",
            "CORRECTION to the earlier epm:followup-scope: still no label line.",
            version=2,
        ),
    ]
    (group,) = followup_label_groups(events)
    assert group["followup_label"] == "unlabeled-2026-06-28T09:00:00Z"
    assert group["n_entries"] == 2
    # The correction IS the authoritative entry (inherit semantics intact)…
    assert group["authoritative"]["version"] == 2
    assert group["label_parse"] == "inherited-from-previous"
    # …but the pseudo-founded group stays a non-dispatchable repair item.
    assert group["dispatchable"] is False
    (unrun_group,) = unrun_followup_labels(events)
    assert unrun_group["dispatchable"] is False


# ─── 7. executing-label resolution: breadcrumb first, head fallback ──────────


def test_executing_label_breadcrumb_first_head_fallback():
    scope_a = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: label-a\nsource: user-chat",
        version=1,
    )
    scope_b = _scope(
        "2026-06-10T01:00:00Z",
        "followup_label: label-b\nsource: proposer-9b-cheap",
        version=2,
    )
    scope_c = _scope(
        "2026-06-10T02:00:00Z",
        "followup_label: label-c\nsource: proposer-9b-cheap",
        version=3,
    )
    run_c = _run(
        "2026-06-10T03:00:00Z",
        "followup_label: label-c source: proposer-9b-cheap round: 1 outcome: done",
    )
    crumb_b_fresh = _ev(
        "epm:progress",
        "2026-06-10T04:00:00Z",
        "stage-dispatch stage=followup-implementing round=1 "
        "subagent=experiment-implementer worktree=/tmp/wt label=label-b",
    )
    # (1) labeled breadcrumb NEWER than the newest run marker → B's group,
    # even though user-chat label-a heads the queue.
    group = executing_followup_label([scope_a, scope_b, scope_c, run_c, crumb_b_fresh])
    assert group is not None and group["followup_label"] == "label-b"
    # (2) breadcrumb OLDER than the newest run marker → dispatchable head (A).
    crumb_b_stale = dict(crumb_b_fresh, ts="2026-06-10T02:30:00Z")
    group = executing_followup_label([scope_a, scope_b, scope_c, run_c, crumb_b_stale])
    assert group is not None and group["followup_label"] == "label-a"
    # (2b) no breadcrumb at all → dispatchable head.
    group = executing_followup_label([scope_a, scope_b])
    assert group is not None and group["followup_label"] == "label-a"
    # (3) no dispatchable unrun label → None (pseudo groups never resolve).
    pseudo_only = _scope("2026-06-11T00:00:00Z", "malformed note, no fields", version=4)
    assert executing_followup_label([pseudo_only]) is None
    assert executing_followup_label([]) is None


# ─── 8. label-keyed re-arm semantics (re-posts do not re-open) ───────────────


def test_same_label_repost_after_run_stays_closed():
    scope_a = _scope(
        "2026-06-10T00:00:00Z",
        "followup_label: label-a\nsource: user-chat",
        version=1,
    )
    run_a = _run(
        "2026-06-10T05:00:00Z",
        "followup_label: label-a\nsource: user-chat\nround: 1",
    )
    # Labeled RE-POST after the run marker: same label → still closed (a
    # re-run needs a NEW label — pins the existing label-keyed semantics).
    repost = _scope(
        "2026-06-10T06:00:00Z",
        "followup_label: label-a\nsource: user-chat\nRE-POST of the earlier scope",
        version=2,
    )
    assert unrun_followup_labels([scope_a, run_a, repost]) == []
    # An UNLABELED re-post carrying the correction signal attributes to A
    # (inherit leg) — closure preserved, never a fresh pseudo group.
    unlabeled_repost = _scope(
        "2026-06-10T06:00:00Z",
        "RE-POST of the earlier scope for label-a with a sharpened spec.",
        version=2,
    )
    assert unrun_followup_labels([scope_a, run_a, unlabeled_repost]) == []


def test_source_falls_back_across_group_entries():
    # A label whose LATEST correction note omits `source:` (the #658-v2
    # shape) must not demote a user-chat round to "unknown" / lose queue
    # priority — group source = FIRST parseable source in scan order.
    events = [
        _scope(
            "2026-06-25T08:42:56Z",
            "- followup_label: genre-generalization-ultrachat\n- source: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-25T09:07:55Z",
            "CORRECTION to the earlier epm:followup-scope "
            "(genre-generalization-ultrachat): gating now unconditional.",
            version=2,
        ),
    ]
    (group,) = followup_label_groups(events)
    assert group["source"] == "user-chat"
    assert group["user_initiated"] is True


# ─── 8b. the #480 duplicate-version anomaly: chronological (ts, version) scan ─


def test_480_duplicate_version_rows_scan_chronologically():
    # The REAL #480 anomaly (plan §12 assumption 4, corrected in Phase 2):
    # per-kind version monotonicity is VIOLATED in the wild — two scope rows
    # share `version: 1` with a v2 chronologically BETWEEN them. The scan key
    # is (ts, version) — chronological with version tiebreak. A (version, ts)
    # mutant scans the late duplicate-v1 row BEFORE the between v2 row, which
    # (a) reorders the first-armed group order and (b) mis-attributes the
    # trailing unlabeled CORRECTION to the wrong previous label.
    events = [
        _scope(
            "2026-06-11T10:00:00Z",
            "followup_label: sycophancy-dose-response\nsource: user-chat",
            version=1,
        ),
        _scope(
            "2026-06-11T11:00:00Z",
            "followup_label: between-label\nsource: proposer-9b-cheap",
            version=2,
        ),
        _scope(
            "2026-06-11T12:00:00Z",
            "followup_label: late-duplicate-v1\nsource: proposer-9b-cheap",
            version=1,
        ),
        _scope(
            "2026-06-11T13:00:00Z",
            "CORRECTION: sharpen the previous scope's eval spec.",
            version=3,
        ),
    ]
    groups = followup_label_groups(events)
    # First-armed group order is CHRONOLOGICAL despite the duplicate version
    # numbers (a version-primary mutant yields [..., late-duplicate-v1,
    # between-label]).
    assert [g["followup_label"] for g in groups] == [
        "sycophancy-dose-response",
        "between-label",
        "late-duplicate-v1",
    ]
    # The unlabeled CORRECTION attributes to the CHRONOLOGICALLY previous
    # label (late-duplicate-v1); a version-primary mutant would scan
    # between-label last and mis-attribute the correction there.
    late = groups[2]
    assert late["n_entries"] == 2
    assert late["authoritative"]["version"] == 3
    assert late["dispatchable"] is True
    assert groups[1]["n_entries"] == 1
    # Queue mechanics unaffected: user-initiated first, then oldest armed ts.
    assert [g["followup_label"] for g in unrun_followup_labels(events)] == [
        "sycophancy-dose-response",
        "between-label",
        "late-duplicate-v1",
    ]


# ─── 8c. retro-close evidence is mechanical + exact-label only ───────────────


def test_retro_close_evidence_exact_label_only():
    label = "persona-vectors-style-rb"
    # (i) a 9a-quater extends=<label> record → evidence.
    ev_methodology = _ev(
        "epm:methodology-doc-generated",
        "2026-06-29T12:00:00Z",
        "EXTEND pass complete: extends=persona-vectors-style-rb gist refreshed",
    )
    assert followup_retro_close_evidence([ev_methodology], label) is not None
    # (ii) an epm:free-analysis-followup-run with followup_ref EXACTLY equal →
    # evidence; a PREFIX match NEVER closes.
    ev_free_exact = _ev(
        "epm:free-analysis-followup-run",
        "2026-06-29T13:00:00Z",
        "followup_ref: persona-vectors-style-rb\noutcome: fit complete",
    )
    ev_free_prefix = _ev(
        "epm:free-analysis-followup-run",
        "2026-06-29T13:00:00Z",
        "followup_ref: persona-vectors-style-rb-9a-ter-fit\noutcome: fit complete",
    )
    assert followup_retro_close_evidence([ev_free_exact], label) is not None
    assert followup_retro_close_evidence([ev_free_prefix], label) is None
    # (iii) a status note with the exact parenthesized round token + a
    # round-completion word on the same line → evidence.
    ev_status = _ev(
        "epm:status-changed",
        "2026-06-29T14:00:00Z",
        "round-4 (persona-vectors-style-rb) clean-result-critic PASS",
    )
    assert followup_retro_close_evidence([ev_status], label) is not None
    # Parenthesized token WITHOUT a completion word → None.
    ev_status_no_word = _ev(
        "epm:status-changed",
        "2026-06-29T14:00:00Z",
        "round-4 (persona-vectors-style-rb) planner amendment dispatched",
    )
    assert followup_retro_close_evidence([ev_status_no_word], label) is None
    # (iv) NEGATIVE (Alt-Codex MF1): the label appearing in proposal/body
    # prose — an epm:follow-ups proposal naming it, or a bare prose mention —
    # NEVER closes.
    ev_proposal = _ev(
        "epm:follow-ups",
        "2026-06-29T15:00:00Z",
        "Proposal 1: persona-vectors-style-rb — extract r_B per the paper; PASS criteria attached",
    )
    ev_prose = _ev(
        "epm:progress",
        "2026-06-29T15:30:00Z",
        "considering persona-vectors-style-rb for the next round; PASS pending",
    )
    assert followup_retro_close_evidence([ev_proposal], label) is None
    assert followup_retro_close_evidence([ev_prose], label) is None
    # No events at all → None.
    assert followup_retro_close_evidence([], label) is None
    # Multiple exact classes agreeing on the SAME label are CORROBORATION,
    # not ambiguity — the canonical #658 ghost label carries both a
    # 9a-quater extends= record AND a status-PASS round note, and must
    # still close (first matching class wins, class order 1 → 2 → 3).
    evidence = followup_retro_close_evidence([ev_methodology, ev_status], label)
    assert evidence is not None
    assert "methodology-doc-generated" in evidence


def test_retro_close_evidence_825_queued_label_park_notes():
    label = "role-map-comparison"
    # The founding #825 false positive (2026-07-04T04:21:23Z), VERBATIM:
    # completion words (re-park / awaiting_promotion) describe the
    # real-user-turn-null round; the label is named only as QUEUED.
    ev_step = _ev(
        "epm:step-completed",
        "2026-07-04T04:21:23Z",
        "<!-- epm:step-completed v1 -->\n## Step Completed\n\n"
        "step: 9a-bis\nat: 031492f2\ntimestamp: 2026-07-04T04:21:23+00:00\n"
        "next_expected_step: 9a-quater\nexit_kind: parked\n"
        "notes: real-user-turn-null round re-parked at awaiting_promotion; "
        "1 unrun user-chat label (role-map-comparison) queued — next entry "
        "dispatches it; cron kept armed\n"
        "<!-- /epm:step-completed -->",
    )
    assert followup_retro_close_evidence([ev_step], label) is None
    # Sibling status-changed note (04:20:49Z), VERBATIM: the token there is
    # `(role-map-comparison,` — not the exact `(role-map-comparison)` — so it
    # does not match today either; pinned so a future token relaxation
    # cannot silently reintroduce the false positive (the clause/veto logic
    # would reject it anyway).
    ev_status = _ev(
        "epm:status-changed",
        "2026-07-04T04:20:49Z",
        "real-user-turn-null round complete; clean-result-critic PASS (r2, "
        "ensemble); re-parking for user promotion. NOTE: 1 unrun user-chat "
        "followup label queued (role-map-comparison, armed 2026-07-03T06:16Z, "
        "now unblocked — all three provenances landed); next /issue 825 entry "
        "dispatches it.",
    )
    assert followup_retro_close_evidence([ev_status], label) is None


def test_retro_close_evidence_595_deferred_scope_recap():
    # The second live false positive (#595, 2026-06-14T06:20:31Z), VERBATIM:
    # the token's clause ("routes next /issue 595 invocation into same-issue
    # loop") carries NO queue-context vocabulary at all — only the #961
    # clause-binding leg catches it; an exclusion regex alone would not.
    label = "h2-full-probes-multiseed"
    ev_step = _ev(
        "epm:step-completed",
        "2026-06-14T06:20:31Z",
        "<!-- epm:step-completed v1 -->\n## Step Completed\n\n"
        "step: 9a-bis\nat: 44eedb0d\ntimestamp: 2026-06-14T06:20:31+00:00\n"
        "next_expected_step: 9a-quater\nexit_kind: parked\n"
        "notes: awaiting clean-result promotion. Full /issue 595 lifecycle "
        "complete: planning → 3 plan revs (v3 corrected squared-gauge) → 3 "
        "implementer rounds (round 3 vendored issue503/) → fullrun-v3 on "
        "pod-595 (after 3 failed GCP auto-lane attempts + 1 Anthropic 429 "
        "mid-run) → upload-verify PASS r2 → analyzer interpretation loop "
        "(2 rounds, reconciler PASS) → clean-result-critic loop (2 rounds, "
        "reconciler PASS) → methodology doc landed + secret gist + body "
        "link-append → awaiting_promotion. Follow-ups: child #640 filed "
        "(postfix carrier, substantially-different); epm:followup-scope v1 "
        "for proposal #2 (h2-full-probes-multiseed) routes next /issue 595 "
        "invocation into same-issue loop; proposal #1 (free-analysis "
        "leverage check) surfaced in epm:follow-ups v1 for user "
        "post-promotion pick. Merge to main BLOCKED — epm:merge-failed v1 "
        "requires manual rebase resolution (Guard 3 + new-shared-src/-infra "
        "guard).\n"
        "<!-- /epm:step-completed -->",
    )
    assert followup_retro_close_evidence([ev_step], label) is None


def test_retro_close_evidence_clause_binding_shapes():
    """Synthetic clause shapes pinning the #961 two-gate class-3 logic."""
    label = "some-label"
    # The dominant park-shape true positive (mirrors #505): completion word
    # in the token's clause via the complete/COMPLETE supplement, PASS later.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "notes: round-2 followup (some-label) complete; clean-result re-gated "
        "PASS; re-parked at awaiting_promotion",
    )
    assert followup_retro_close_evidence([ev], label) is not None
    # The COMPLETE variant (mirrors #545).
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "/issue same-issue follow-up (some-label) loop COMPLETE. Both critic ensembles PASS",
    )
    assert followup_retro_close_evidence([ev], label) is not None
    # Cross-clause split: completion words describe ANOTHER round's clause;
    # the token's clause needs no veto word to be rejected.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "real-round re-parked at awaiting_promotion; label (some-label) held for next entry",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # Same-clause veto: clause binding alone would wrongly match
    # (awaiting_promotion is in the token's clause) — the queue veto rejects.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "unrun label (some-label) queued for the awaiting_promotion park",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # Veto is per-clause: a queued mention of ANOTHER label elsewhere on the
    # line does not block a legitimate close of this one.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "round-4 (some-label) clean-result-critic PASS; next label (other-label) queued unrun",
    )
    assert followup_retro_close_evidence([ev], "some-label") is not None
    assert followup_retro_close_evidence([ev], "other-label") is None
    # Narrowing-only guard: no gate-1 word on the line ⇒ the `complete`
    # supplement alone can never CREATE evidence the old predicate rejected.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "(some-label) round complete, nothing more",
    )
    assert followup_retro_close_evidence([ev], label) is None
    # `incomplete` lookbehind + cross-clause: neither leg matches.
    ev = _ev(
        "epm:status-changed",
        "2026-07-04T00:00:00Z",
        "(some-label) round incomplete; clean-result-critic PASS pending",
    )
    assert followup_retro_close_evidence([ev], label) is None


# ─── 14. corpus replay over the real tasks/ tree ─────────────────────────────


def _tasks_root() -> Path | None:
    root = Path(__file__).resolve().parents[1] / "tasks"
    return root if root.is_dir() else None


def _load_events(task_dir: Path) -> list[dict]:
    events: list[dict] = []
    path = task_dir / "events.jsonl"
    if not path.is_file():
        return events
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _corpus_events_by_task() -> dict[int, list[dict]]:
    root = _tasks_root()
    assert root is not None
    by_task: dict[int, list[dict]] = {}
    for task_dir in sorted(root.glob("*/*")):
        if not task_dir.is_dir() or not task_dir.name.isdigit():
            continue
        events = _load_events(task_dir)
        if events:
            by_task.setdefault(int(task_dir.name), []).extend(events)
    return by_task


def _run_labels(events: list[dict]) -> set[str]:
    return {
        parse_followup_note_field(e.get("note") or "", "followup_label")
        for e in events
        if e.get("kind") == "epm:same-issue-followup-run"
    } - {None}


def test_corpus_replay_all_historical_markers():
    # Alt-Claude MF1 corpus-replay validation: every HISTORICAL
    # epm:same-issue-followup-run marker in the checkout must parse a
    # followup_label (covers the 15 `=`-form markers + all colon forms), and
    # the hand-checked #763/#658/#537/#552 expectations must hold.
    import pytest

    if _tasks_root() is None:
        pytest.skip("tasks/ not present in this checkout (sparse worktree)")
    by_task = _corpus_events_by_task()

    unparseable: list[tuple[int, str]] = []
    n_run = 0
    for task_id, events in by_task.items():
        for ev in events:
            if ev.get("kind") != "epm:same-issue-followup-run":
                continue
            n_run += 1
            if parse_followup_note_field(ev.get("note") or "", "followup_label") is None:
                unparseable.append((task_id, str(ev.get("ts"))))
    assert n_run > 0, "corpus unexpectedly carries no run markers"
    assert unparseable == [], f"unparseable run-marker labels: {unparseable}"

    # #763 (the driving incident): the queued user-chat label must surface as
    # dispatchable-unrun for as long as no run marker closes it (events are
    # append-only, so once closed the guard makes this leg inert — a
    # LEGITIMATE later close of the round this fix un-strands).
    if 763 in by_task:
        events = by_task[763]
        unrun = {g["followup_label"]: g for g in unrun_followup_labels(events)}
        assert "deception-rubric-reanchor" not in unrun
        if "neutral-contrast-and-cofit" not in _run_labels(events):
            group = unrun.get("neutral-contrast-and-cofit")
            assert group is not None, "the #763 queued label must be visible as unrun"
            assert group["dispatchable"] is True
            assert group["user_initiated"] is True

    # #658 (correction chain + ghost labels): the v3→v7 chain groups into ONE
    # label; the unlabeled v2 CORRECTION attributes to genre-generalization-
    # ultrachat (2 entries), which is closed by its run marker.
    if 658 in by_task:
        events = by_task[658]
        groups = {g["followup_label"]: g for g in followup_label_groups(events)}
        assert "persona-vectors-style-rb" in groups
        assert groups["persona-vectors-style-rb"]["n_entries"] >= 5  # v3..v7
        assert groups["persona-vectors-style-rb"]["authoritative"]["version"] >= 7
        assert groups["genre-generalization-ultrachat"]["n_entries"] >= 2  # v1 + v2 correction
        unrun = {g["followup_label"] for g in unrun_followup_labels(events)}
        assert "genre-generalization-ultrachat" not in unrun
        if "persona-vectors-style-rb" not in _run_labels(events):
            # A ghost label (round demonstrably ran, no run marker) surfaces
            # as unrun — the Step 0 retro-close disposition rule handles it.
            assert "persona-vectors-style-rb" in unrun

    # #537 / #552: the `=`-form run markers close their labels.
    for task_id, closed_labels in (
        (
            537,
            [
                "seed2-marker-fact-replication",
                "behavior-conditioned-predictors",
                "predictor-bakeoff-complete",
            ],
        ),
        (
            552,
            [
                "em-arm-mean-resp-reextraction",
                "marker-arm-mean-resp-reextraction",
                "contrastive-2x2-completion",
            ],
        ),
    ):
        if task_id not in by_task:
            continue
        unrun = {g["followup_label"] for g in unrun_followup_labels(by_task[task_id])}
        for label in closed_labels:
            assert label not in unrun, f"#{task_id} {label} must be closed by its run marker"


def test_corpus_replay_retro_close_verdicts():
    # #961 retro-close pins (events are append-only; guards make each leg
    # inert once the label legitimately closes via a run marker):
    import pytest

    if _tasks_root() is None:
        pytest.skip("tasks/ not present in this checkout (sparse worktree)")
    by_task = _corpus_events_by_task()
    for task_id, queued_label in (
        (825, "role-map-comparison"),
        (595, "h2-full-probes-multiseed"),
    ):
        if task_id in by_task and queued_label not in _run_labels(by_task[task_id]):
            assert followup_retro_close_evidence(by_task[task_id], queued_label) is None, (
                f"#{task_id} {queued_label} is queued/unrun — retro-close evidence "
                "for it is the #961 false positive"
            )
    if 658 in by_task:  # the canonical ghost close must SURVIVE the narrowing
        assert followup_retro_close_evidence(by_task[658], "persona-vectors-style-rb") is not None
