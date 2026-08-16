"""Prose pins for the #2146 inline-round session-survival backstop.

Pins (a) the SKILL.md Step 9a-ter § Inline-round session-survival backstop
block (arm the 45-min `/issue-tick <N>` cron IFF the parent task is in
`tick_triage.ISSUE_ACTIVE`; CRON-TEARDOWN pointer; incident #1491), (b) the
CLAUDE.md sentence mirroring it inside the user-chat inline free-analysis
carve-out's Detached-by-default sub-block, (c) the status-partition PREMISE
against the LIVE ``scripts/tick_triage.py`` module — the block's per-class
status enumerations must EQUAL the module's frozensets, so a future
re-partition (a status moved between sets, a new status added) fails THIS
test loudly instead of leaving the prose quietly wrong — and (d) this
file's own registration in the Step-9c selector's WORKFLOW_INVARIANT set
(SKILL.md/CLAUDE.md diffs select only that set — an unregistered pin never
runs on the diffs it guards).

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

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = REPO / "CLAUDE.md"
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
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


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


def test_claude_md_backstop_sentence_present():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    # "**User-chat inline free analysis**" occurs TWICE in CLAUDE.md (a
    # cross-reference inside the Follow-up bullet, then the carve-out bullet
    # itself). Search from the SECOND occurrence — strictly safer than the
    # family precedent's documented first-occurrence search.
    i0 = text.index("**User-chat inline free analysis**")
    i1 = text.index("**User-chat inline free analysis**", i0 + 1)
    # The sentence lives INSIDE the Detached-by-default sub-block (appended,
    # not a new bullet or bold sub-block), before the pod-safety sub-block.
    start = text.index("**Detached-by-default + lifecycle ack", i1)
    end = text.index("**Pod-safety pre-launch signals (deviation case", start)
    window = text[start:end]
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
