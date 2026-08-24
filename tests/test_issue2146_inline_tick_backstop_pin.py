"""Prose pins for the #2146 inline-round session-survival backstop.

Pins (a) the SKILL.md Step 9a-ter § Inline-round session-survival backstop
block (arm the 45-min `/issue-tick <N>` cron IFF the parent task is in
`tick_triage.ISSUE_ACTIVE`; CRON-TEARDOWN pointer; incident #1491), (b) the
CLAUDE.md sentence mirroring it inside the user-chat inline free-analysis
carve-out's Detached-by-default sub-block, (c) the status-partition PREMISE
against the LIVE ``scripts/tick_triage.py`` module — the block's per-class
status enumerations must EQUAL the module's frozensets, so a future
re-partition (a status moved between sets, a new status added) fails THIS
test loudly instead of leaving the prose quietly wrong — (d) this
file's own registration in the Step-9c selector's WORKFLOW_INVARIANT set
(SKILL.md/CLAUDE.md diffs select only that set — an unregistered pin never
runs on the diffs it guards) — (e) the round-2 C1-C3 qualifiers, each
grounded against the LIVE ``compute_issue_verdict`` — with ``over_cap``
driven through the LIVE ``plan_pending_over_cap`` on synthetic events
(all three of its branches; round 3) and the SKILL.md block's exact
at-least-as-new predicate wording pinned — and token-pinned in
BOTH prose surfaces: the ``plan_pending`` under-cap (PARK) vs gate-parked
over-cap (TERMINAL/gate branch) split, the transition-dependent
(first-fire, ``prev_status != status``) gate push, and the out-of-enum
consequence being the SAME forbidden re-spawn (``/issue-tick`` maps a
non-zero triage exit to STALE-REDRIVE), never silence — and (f) the
three-set union equalling the FULL ``task_workflow.STATUSES`` enum, so a
hypothetical FOURTH issue-mode status set cannot stale the block's
"outside the three sets" clause while the per-class pins stay green.

Incident (#1491, 2026-08-05): an inline-override round on an
``awaiting_promotion`` parent armed no session-survival backstop while an
~$44/h pod ran a crash-fix; surfaced only by user challenge. The literal
"arm the tick cron unconditionally" fix proposed in the #2146 task body is
inoperative on TERMINAL parents (one fire, then the cron tears itself
down) and actively wrong on PARK parents (a stale tick STALE-REDRIVEs the
full ``/issue <N>`` skill — the re-spawn the override clause forbids), so
the shipped rule is arm-iff-ACTIVE; this test pins the partition that rule
rests on.

Family precedent: tests/test_issue_skill_inline_measurement_duties.py.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
from pathlib import Path

import pytest

from tests.issue_skill_source import read_workflow_doc

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
ISSUE_TICK_SKILL_MD = REPO / ".claude" / "skills" / "issue-tick" / "SKILL.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"
TICK_TRIAGE_PY = REPO / "scripts" / "tick_triage.py"

ANCHOR = "Inline-round session-survival backstop"
PIN_FILE_RELPATH = "tests/test_issue2146_inline_tick_backstop_pin.py"
_CLASS_NAMES = ("ISSUE_ACTIVE", "ISSUE_PARK", "ISSUE_TERMINAL")


def _normalized(path: Path) -> str:
    """File text with whitespace runs collapsed (wrap-insensitive pins).

    SKILL.md wraps prose at ~75-78 columns, so raw-substring pins on
    multi-word fragments would break on any innocent re-wrap (same
    convention as tests/test_issue_skill_inline_measurement_duties.py).
    """
    return re.sub(r"\s+", " ", read_workflow_doc(path))


def _load_module(module_name: str, path: Path):
    """Path-load a scripts/ module (the family-precedent selector idiom)."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skill_block_window() -> str:
    """The backstop block, ANCHOR through the next sibling block's header."""
    text = _normalized(SKILL_MD)
    idx = text.index(ANCHOR)  # ValueError = hard fail
    end = text.index("Pod-safety pre-launch signals (deviation case", idx)
    return text[idx:end]


def test_skill_9a_ter_backstop_block_present():
    text = _normalized(SKILL_MD)
    idx = text.index(ANCHOR)  # ValueError = hard fail
    window = _skill_block_window()
    # The arm-iff-ACTIVE rule.
    assert "IFF the parent task's status is in" in window
    assert "tick_triage.ISSUE_ACTIVE" in window
    # ARM-GUARD + teardown by POINTER (never re-inlined here).
    assert "ARM-GUARD" in window
    assert "CRON-TEARDOWN procedure" in window
    # The incident line.
    assert "#1491" in window
    # Sits inside 9a-ter: strictly after the instrument-supersession block,
    # strictly before the pod-safety block (both anchors unique on the live
    # tree at authoring time — the same ordering the sibling pins assert).
    assert (
        text.index("Instrument-supersession + scope-extension addenda duties")
        < idx
        < text.index("Pod-safety pre-launch signals (deviation case")
    )


def _claude_sentence_window() -> str:
    """The CLAUDE.md Detached-by-default sub-block holding the mirror sentence.

    "**User-chat inline free analysis**" occurs TWICE in CLAUDE.md (a
    cross-reference inside the Follow-up bullet, then the carve-out bullet
    itself). Search from the SECOND occurrence — strictly safer than the
    family precedent's documented first-occurrence search. The sentence
    lives INSIDE the Detached-by-default sub-block (appended, not a new
    bullet or bold sub-block), before the pod-safety sub-block.
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    i0 = text.index("**User-chat inline free analysis**")
    i1 = text.index("**User-chat inline free analysis**", i0 + 1)
    start = text.index("**Detached-by-default + lifecycle ack", i1)
    end = text.index("**Pod-safety pre-launch signals (deviation case", start)
    return text[start:end]


def test_claude_md_backstop_sentence_present():
    window = _claude_sentence_window()
    # Cites the canonical block (the sibling-duty citation convention).
    assert "SKILL.md Step 9a-ter § " + ANCHOR in window
    # Carries the arm-iff-ACTIVE rule + the incident citations.
    assert "tick_triage.ISSUE_ACTIVE" in window
    assert "#1491" in window
    assert "#2146" in window


def test_status_partition_matches_tick_triage():
    tt = _load_module("tick_triage_2146", TICK_TRIAGE_PY)
    window = _skill_block_window()

    # (i) The documented set relationships still hold on the LIVE module.
    assert tt.ISSUE_GATE <= tt.ISSUE_TERMINAL, (
        "tick_triage re-partitioned: ISSUE_GATE is no longer a subset of "
        "ISSUE_TERMINAL. The SKILL.md 9a-ter session-survival backstop block "
        "(and its CLAUDE.md mirror sentence) rest on that premise — update "
        "the prose so it does not quietly lie."
    )
    for a, b in itertools.combinations(_CLASS_NAMES, 2):
        overlap = getattr(tt, a) & getattr(tt, b)
        assert not overlap, (
            f"tick_triage re-partitioned: {a} and {b} overlap on "
            f"{sorted(overlap)}. The SKILL.md 9a-ter session-survival "
            "backstop block assumes the three classes are pairwise disjoint "
            "— update the prose so it does not quietly lie."
        )

    # (ii) The block's per-class enumerations EQUAL the live sets — every
    # member accounted for, none stale. Deliberately STRONGER than a
    # "literal-or-class-name appears" disjunction: the class names always
    # appear in the block, which would make that form vacuous for a newly
    # added status. Set equality is what makes a re-partition of
    # tick_triage fail THIS test loudly instead of leaving the prose
    # quietly wrong.
    for cls in (*_CLASS_NAMES, "ISSUE_GATE"):
        m = re.search(re.escape(cls + "`") + r"\s*\(([^)]*)\)", window)
        assert m is not None, (
            f"SKILL.md 9a-ter session-survival backstop block no longer "
            f"enumerates {cls} in the pinned `{cls}` (`status` / ...) shape "
            "— the partition pin needs that enumeration to compare against "
            "the live tick_triage module."
        )
        documented = set(re.findall(r"`([a-z_]+)`", m.group(1)))
        live = set(getattr(tt, cls))
        assert documented == live, (
            f"tick_triage.{cls} re-partitioned under the prose: the "
            f"SKILL.md 9a-ter session-survival backstop block enumerates "
            f"{sorted(documented)} but the live module has {sorted(live)}. "
            "A status moved between sets or a new status was added — update "
            "the block (and the CLAUDE.md mirror sentence) so the "
            "arm-iff-ISSUE_ACTIVE rule does not quietly lie."
        )


def test_registered_in_step9c_workflow_invariant():
    sel = _load_module("select_step9c_tests_2146", SELECTOR_PY)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT, (
        "unregistered pin: SKILL.md/CLAUDE.md diffs select only the "
        "WORKFLOW_INVARIANT set, so this file never runs on the diffs it "
        "guards until it is registered in scripts/select_step9c_tests.py."
    )


# Round-2 (#2146) C1/C2 qualifier pins. Each row grounds one prose qualifier
# against the LIVE ``compute_issue_verdict`` AND pins the token(s) that
# document it in BOTH surfaces (the SKILL.md block window + the CLAUDE.md
# mirror sentence), so deleting a qualifier fails this test even while the
# per-class set-equality pins stay green. ``marker_age_s`` is very stale so
# a PARK status deterministically STALE-REDRIVEs. Round 3: ``over_cap`` is
# NOT a hardcoded bool — each row carries a synthetic ``events`` list and
# the test computes ``over_cap = plan_pending_over_cap(events)``, so the
# LIVE helper's predicate is pinned across all three of its branches:
# (i) no spend marker -> False; (ii) spend marker with NO status-changed
# marker -> True; (iii) spend marker with ts EQUAL to the newest
# status-changed ts -> True (``spend >= changed`` — at-least-as-new, not
# strictly-newer; narrowing the helper now fails THIS test).
_TS = "2026-08-15T00:00:00Z"
_SPEND_MARKER = {"kind": "epm:awaiting-spend-approval", "ts": _TS}
_STATUS_CHANGED_MARKER = {"kind": "epm:status-changed", "ts": _TS}
_QUALIFIER_CASES: tuple[tuple[str, str | None, list[dict], str, tuple[str, ...]], ...] = (
    # C1 / over-cap branch (i): no spend marker -> under-cap plan_pending is
    # PARK — a stale tick re-drives the full /issue skill (the forbidden
    # re-spawn).
    ("plan_pending", None, [], "STALE-REDRIVE", ("under-cap `plan_pending`",)),
    # C1+C2 / over-cap branch (ii): a spend marker with NO status-changed
    # marker at all -> gate-parked (over-cap) plan_pending routes through
    # the TERMINAL/gate branch instead, pushing only on the first fire
    # (no prior same-status snapshot).
    (
        "plan_pending",
        None,
        [_SPEND_MARKER],
        "GATE-TRANSITION",
        ("over-cap `plan_pending`", "first fire"),
    ),
    # C2 / over-cap branch (iii): a spend marker with ts EQUAL to the newest
    # status-changed ts is still over-cap (``spend >= changed``); a later
    # same-status fire reads plain TERMINAL.
    (
        "plan_pending",
        "plan_pending",
        [_STATUS_CHANGED_MARKER, _SPEND_MARKER],
        "TERMINAL",
        ("transition-dependent",),
    ),
    # C2: the same transition dependence on the ISSUE_GATE members proper.
    ("awaiting_promotion", None, [], "GATE-TRANSITION", ("first fire",)),
    ("awaiting_promotion", "awaiting_promotion", [], "TERMINAL", ("transition-dependent",)),
    # Round 3 (F3): a REAL transition — a non-None, DIFFERENT prev_status —
    # exercises the ``prev_status != status`` branch beyond the
    # missing-snapshot (prev=None) case.
    ("awaiting_promotion", "interpreting", [], "GATE-TRANSITION", ("transition-dependent",)),
)


def test_c1_c2_qualifiers_match_live_verdicts_and_are_pinned():
    tt = _load_module("tick_triage_2146_qual", TICK_TRIAGE_PY)
    skill_window = _skill_block_window()
    claude_window = _claude_sentence_window()
    for status, prev_status, events, expected, tokens in _QUALIFIER_CASES:
        over_cap = tt.plan_pending_over_cap(events)
        verdict, _reason, _streak = tt.compute_issue_verdict(
            status, prev_status, 10.0**9, over_cap, stale_after_s=3600.0
        )
        assert verdict == expected, (
            f"compute_issue_verdict(status={status!r}, prev={prev_status!r}, "
            f"over_cap={over_cap} via plan_pending_over_cap({events!r})) "
            f"returned {verdict!r}, expected {expected!r} "
            "— the SKILL.md 9a-ter backstop block + CLAUDE.md mirror sentence "
            "qualifiers rest on this behavior; update BOTH surfaces (and this "
            "table) together with any tick_triage change."
        )
        for token in tokens:
            assert token in skill_window, (
                f"qualifier token {token!r} missing from the SKILL.md 9a-ter "
                "session-survival backstop block — the round-2 #2146 "
                "correction it pins was edited away."
            )
            assert token in claude_window, (
                f"qualifier token {token!r} missing from the CLAUDE.md mirror "
                "sentence — the round-2 #2146 correction it pins was edited "
                "away."
            )
    # Round 3 (F2): the SKILL.md block states plan_pending_over_cap's EXACT
    # predicate — at-least-as-new (equal timestamps count) OR no
    # status-changed marker at all — never the strictly-newer reading the
    # r1/r2 prose carried. The CLAUDE.md mirror deliberately does NOT
    # restate the predicate (it compresses to "gate-parked" and lets the
    # canonical block carry it), so these fragments pin the SKILL window
    # only.
    for fragment in (
        "AT LEAST as new as the newest `epm:status-changed` marker",
        "equal timestamps count",
        "or with no `epm:status-changed` marker at all",
        "`tick_triage.plan_pending_over_cap`",
    ):
        assert fragment in skill_window, (
            f"predicate fragment {fragment!r} missing from the SKILL.md "
            "9a-ter backstop block — the round-3 #2146 exact-predicate "
            "wording (``spend >= changed``, or no status-changed marker at "
            "all) was edited away; re-verify against "
            "tick_triage.plan_pending_over_cap before rewording."
        )
    assert "marker newer than the last status change" not in skill_window, (
        "the retracted strictly-newer over-cap wording resurfaced in the "
        "SKILL.md 9a-ter backstop block — plan_pending_over_cap is "
        "``spend >= changed`` (at-least-as-new) with a missing "
        "status-changed marker counting as over-cap."
    )


def test_c3_out_of_enum_realizes_stale_redrive_not_a_crash():
    """C3: the realized out-of-enum outcome is the forbidden re-spawn.

    ``compute_issue_verdict`` raises ``ValueError`` on a status outside the
    three sets, but ``/issue-tick`` maps a non-zero triage exit to
    STALE-REDRIVE by design (fail toward coverage, never toward silence),
    so the REALIZED outcome is the same forbidden full-``/issue <N>``
    re-spawn as the ISSUE_PARK case — not a mere crash. Pin the raise, the
    mapping at its source, and the corrected consequence in both surfaces.
    """
    tt = _load_module("tick_triage_2146_c3", TICK_TRIAGE_PY)
    with pytest.raises(ValueError):
        tt.compute_issue_verdict("no_such_status", None, None, False, stale_after_s=3600.0)
    tick_skill = _normalized(ISSUE_TICK_SKILL_MD)
    assert "Non-zero exit or unparseable output → treat as `STALE-REDRIVE`" in tick_skill, (
        ".claude/skills/issue-tick/SKILL.md no longer documents the "
        "non-zero-exit → STALE-REDRIVE mapping the C3 correction rests on — "
        "re-verify the realized out-of-enum outcome and update the 9a-ter "
        "block + CLAUDE.md mirror sentence."
    )
    token = "maps a non-zero triage exit to STALE-REDRIVE"
    skill_window = _skill_block_window()
    assert token in skill_window, (
        f"C3 token {token!r} missing from the SKILL.md 9a-ter backstop block "
        "— the round-2 out-of-enum consequence (the SAME forbidden re-spawn, "
        "never silence) was edited away."
    )
    assert token in _claude_sentence_window(), (
        f"C3 token {token!r} missing from the CLAUDE.md mirror sentence — "
        "the round-2 out-of-enum consequence (the SAME forbidden re-spawn, "
        "never silence) was edited away."
    )
    # The retracted round-1 wording must not resurface.
    assert "(a crash, not a backstop)" not in skill_window


def test_three_set_union_covers_full_status_enum():
    """C4: no FOURTH issue-mode status set can hide from the pins.

    ``test_status_partition_matches_tick_triage`` pins per-class set
    equality, which a hypothetical fourth set added to tick_triage would
    PASS while staling the block's "outside the three sets" clause.
    tick_triage's own header comment scopes the sets to the runtime enum
    ``task_workflow.STATUSES``; today the three-set union EQUALS that full
    14-status enum, which is exactly what makes the clause load-bearing —
    every real task status routes through one of the three sets, so the
    ValueError → STALE-REDRIVE path fires only on enum drift. Equality
    (not subset) is asserted so BOTH drift directions fail loudly: a
    status carved out of the three sets into a fourth, and a new STATUSES
    member left unrouted.
    """
    from explore_persona_space.task_workflow import STATUSES

    tt = _load_module("tick_triage_2146_union", TICK_TRIAGE_PY)
    union = tt.ISSUE_ACTIVE | tt.ISSUE_PARK | tt.ISSUE_TERMINAL
    assert union == set(STATUSES), (
        "ISSUE_ACTIVE | ISSUE_PARK | ISSUE_TERMINAL no longer equals "
        f"task_workflow.STATUSES (union-only: {sorted(union - set(STATUSES))}; "
        f"enum-only: {sorted(set(STATUSES) - union)}). A fourth issue-mode "
        "status set (or an unrouted new status) stales the 9a-ter backstop "
        "block's 'outside the three sets' clause — update the block, the "
        "CLAUDE.md mirror sentence, and this pin together."
    )
