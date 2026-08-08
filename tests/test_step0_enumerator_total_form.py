"""#1722 regression pin: the Step-0 enumerator one-liner in SKILL.md must not
IndexError on a marker whose ``note`` field is empty / None / whitespace-only.

Three sessions hit `IndexError: list index out of range` on 2026-07-26 from
the non-total form ``(e.get("note") or "").splitlines()[0][:140]``. The fix
threads the ``.splitlines()`` result through an ``or [""]`` short-circuit so
``[0]`` is always defined. This test:

1. Constructs the total-form expression as a string, evals it against a
   parametrized event set covering the exact 2026-07-26 failure inputs plus
   ordinary cases, and asserts no ``IndexError`` and correct output.
2. Reads ``.claude/skills/issue/SKILL.md`` and pins the total form as
   present verbatim + the vulnerable form as absent — a future regression
   to the pre-#1722 shape fails the durability pin.

SKILL.md is markdown prose, so the test cannot import the idiom — it must
construct the exact substring and eval it. If the SKILL.md one-liner is
refactored, update ``STEP0_EXPR`` and the ``needle`` literal together.
"""

from __future__ import annotations

import pathlib

import pytest

# The expression under test — must match the SKILL.md Step-0 enumerator
# one-liner byte-for-byte. Kept in ONE place so refactors are lockstepped.
STEP0_EXPR = '(((e.get("note") or "").splitlines()) or [""])[0][:140]'


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        # (a) The exact 2026-07-26 failure inputs — every one raised IndexError
        # under the pre-#1722 form.
        ("", ""),
        (None, ""),
        ("\n", ""),
        ("   \n", "   "),
        # (b) Ordinary notes — first line, truncated at 140 chars.
        ("first line", "first line"),
        ("first\nsecond", "first"),
        ("x" * 200, "x" * 140),
        # (c) The absent-key case — get() returns None, or "" fires.
        (..., ""),
    ],
)
def test_step0_expr_survives_empty_notes(note: object, expected: str) -> None:
    """The total form in SKILL.md L6236 handles every corner case for `note`."""
    # note is ... encodes the absent-key case (get() returns None → or "" fires).
    e = {"ts": "t", "kind": "k"} if note is ... else {"ts": "t", "kind": "k", "note": note}
    # Eval the string form exactly as the pasted one-liner would run.
    result = eval(STEP0_EXPR, {}, {"e": e})
    assert result == expected


def test_step0_expr_pinned_verbatim_in_skill_md() -> None:
    """Durability pin: the total form must remain verbatim in SKILL.md.

    Also asserts the pre-#1722 vulnerable form is gone — a regression to
    the classic ``(e.get("note") or "").splitlines()[0][:140]`` fails here.
    """
    skill_md = (
        pathlib.Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"
    )
    text = skill_md.read_text(encoding="utf-8")
    # The exact substring the Step-0 enumerator uses — matches STEP0_EXPR.
    needle = '(((e.get("note") or "").splitlines()) or [""])[0][:140]'
    assert needle in text, (
        "SKILL.md must contain the total form of the Step-0 enumerator "
        "one-liner (#1722); found none. Update STEP0_EXPR and the needle "
        "literal together if the expression was intentionally refactored."
    )
    # The pre-#1722 vulnerable form must not remain.
    vulnerable = '(e.get("note") or "").splitlines()[0][:140]'
    assert vulnerable not in text, (
        "SKILL.md still carries the pre-#1722 vulnerable form of the "
        "Step-0 enumerator one-liner — replace it with the total form."
    )
