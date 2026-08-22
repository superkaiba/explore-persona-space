"""Pin the inline-payload-lint-gate contract across its sites (#1460/#1500/#1531).

Inline free-analysis rounds (the CLAUDE.md § Routing "User-chat inline
free analysis" carve-out + `/issue` Step 9a-ter zero-GPU auto-runs)
commit analysis scripts DIRECTLY to `main`, deliberately skipping the
worktree -> code-review -> Step 10d pipeline. That channel historically
had NO lint duty: two inline-landed bare `.list_repo_tree(` scripts
(incident #1388) broke `tests/test_workflow_lint.py` on pristine main
fleet-wide, and >=4 sessions burned rounds re-classifying it as
pre-existing at every Step 9c gate.

Task #1460 adds the **inline payload lint gate**; task #1500 (commit
`8248be9501`) MECHANIZED it into `scripts/inline_lint_gate.py`, so the
canonical recipe is now SPLIT: the `.claude/skills/issue/SKILL.md`
Step 9a-ter block carries the prose contract (verdict payload-attributed
with an instrument-ran completeness check, never a bare exit-0, fail
CLOSED on a dead/silent leg) while the helper carries the mechanized leg
invocations + the pytest-summary completeness pattern — this file pins
BOTH halves, plus the one-line pointers in `CLAUDE.md` (the Same-turn
completion contract clause), `.claude/agents/analyzer.md` (the
figure-commit script-sweep channel), and
`.claude/skills/issue-v2/SKILL.md` (the v2 zero-GPU band).

These tests pin the duty text so future prose churn cannot silently drop
it (the #1134/#1045 drift class). Pure file-content pins — no import of
`workflow_lint` (module-registration import trap). Sibling coverage:
`tests/test_issue_skill_inline_gate_pin.py` (structural helper/hook
wiring pins, incl. the `_gate_section()` delimiter reused here) and
`tests/test_inline_lint_gate.py` (behavioral exit-code pins).
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text
from tests.test_issue_skill_inline_gate_pin import _gate_section

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
ANALYZER = ROOT / ".claude" / "agents" / "analyzer.md"
ISSUE_V2_SKILL = ROOT / ".claude" / "skills" / "issue-v2" / "SKILL.md"
HELPER = ROOT / "scripts" / "inline_lint_gate.py"

ANCHOR = "Inline payload lint gate"
BOLD_ANCHOR = "**Inline payload lint gate"


def test_skill_9a_ter_carries_inline_payload_lint_gate() -> None:
    """SKILL.md carries the canonical gate block: recipe + verdict contract.

    Post-#1500 the gate is MECHANIZED in scripts/inline_lint_gate.py; the
    prose block delegates the leg invocations to the helper, so this test
    pins only what still lives in prose. The moved invocation/regex
    literals are pinned in the helper by
    test_helper_carries_mechanized_gate_leg_literals below. The block is
    delimited by the sibling pin's _gate_section() (bold header -> next
    numbered step), not a first-occurrence char window (#1531).
    """
    text = issue_skill_text()
    n_bold = text.count(BOLD_ANCHOR)
    assert n_bold == 1, (
        f"expected exactly ONE bold {BOLD_ANCHOR!r} header in SKILL.md "
        f"(the canonical Step 9a-ter block), found {n_bold} — either the "
        "canonical block was lost (#1460: the inline direct-to-main commit "
        "channel is ungated again) or a second bold mention shadows it "
        "(the #1531 bug class): keep the Step 9a-ter header bold and write "
        "other mentions as '§ Inline payload lint gate' pointers."
    )
    section = _gate_section()
    for needle, why in (
        (
            "no-flags `workflow_lint.py`",
            "the no-flags lint leg (invocation mechanized in the helper, #1500)",
        ),
        ("select_step9c_tests.py", "the #1147 mapped scan-test leg"),
        ("--map-files", "the mapped-test invocation flag"),
        ("payload-attributed", "the payload-attribution verdict framing"),
        ("bare exit-0", "the never-bare-exit-0 contract (main can be red)"),
        ("workflow_lint: PASS", "the instrument-ran completeness evidence"),
        ("INCONCLUSIVE", "the fail-closed dead-instrument token"),
    ):
        assert needle in section, (
            f"SKILL.md 'Inline payload lint gate' block lost {why!s} "
            f"({needle!r} not found in the Step 9a-ter section) — the gate "
            "contract drifted (#1460/#1531)."
        )


def test_helper_carries_mechanized_gate_leg_literals() -> None:
    """scripts/inline_lint_gate.py carries the #1460 contract literals the
    #1500 mechanization moved out of SKILL.md prose. Exit-code semantics
    (0=PASS / 1=BLOCK / 3=INCONCLUSIVE) and completeness behavior are
    already BEHAVIORALLY pinned by tests/test_inline_lint_gate.py and are
    deliberately not re-pinned as content here.
    """
    src = HELPER.read_text(encoding="utf-8")
    for needle, why in (
        (
            '["uv", "run", "python", "scripts/workflow_lint.py"]',
            "the NO-FLAGS workflow_lint invocation (an appended flag would "
            "silently narrow the lint leg)",
        ),
        (
            '"scripts/select_step9c_tests.py", "--map-files"',
            "the #1147 mapped scan-test leg invocation",
        ),
        (
            "[0-9]+ (passed|failed|error|xpassed|xfailed)|no tests ran",
            "the pytest-summary-shaped completeness pattern (a lint-leg "
            "`FAIL (N error(s))` line must not satisfy the pytest-leg check)",
        ),
    ):
        assert needle in src, (
            f"scripts/inline_lint_gate.py lost {why!s} ({needle!r} not "
            "found) — the mechanized gate contract drifted "
            "(#1460/#1500/#1531)."
        )


def test_claude_md_same_turn_contract_names_the_gate() -> None:
    """The CLAUDE.md Same-turn completion contract clause names the gate.

    The lowercase duty mention sits within 1,200 chars of the contract
    heading, and the capitalized `§ Inline payload lint gate`
    cross-reference (pointing at the SKILL.md canonical block) appears in
    the same bullet (the carve-out bullet is one physical line).
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    idx = text.find("Same-turn completion contract")
    assert idx != -1, (
        "CLAUDE.md lost the 'Same-turn completion contract' clause of the "
        "User-chat inline free analysis carve-out — the #1460 gate hook "
        "point is gone."
    )
    window = text[idx : idx + 1200]
    assert "inline payload lint gate" in window.lower(), (
        "CLAUDE.md Same-turn completion contract no longer names the inline "
        "payload lint gate within 1,200 chars — the direct-to-main commit "
        "clause lost its lint duty (#1460, incident #1388)."
    )
    line_end = text.find("\n", idx)
    bullet_line = text[text.rfind("\n", 0, idx) + 1 : line_end if line_end != -1 else len(text)]
    assert "§ Inline payload lint gate" in bullet_line, (
        "CLAUDE.md Same-turn completion contract bullet lost the capitalized "
        "'§ Inline payload lint gate' cross-reference to the SKILL.md "
        "Step 9a-ter canonical block (#1460)."
    )


def test_analyzer_commit_workflow_names_the_gate() -> None:
    """analyzer.md (figure-commit script-sweep channel) points at the gate."""
    text = ANALYZER.read_text(encoding="utf-8")
    assert ANCHOR in text, (
        f"analyzer.md lost its '{ANCHOR}' pointer — the figures-commit "
        "script-sweep channel (the #1092 shape) is ungated again (#1460)."
    )


def test_issue_v2_zero_gpu_band_names_the_gate() -> None:
    """issue-v2/SKILL.md zero-GPU floor bullet points at the gate."""
    text = ISSUE_V2_SKILL.read_text(encoding="utf-8")
    assert ANCHOR in text, (
        f"issue-v2/SKILL.md lost its '{ANCHOR}' pointer — the v2 zero-GPU "
        "free-analysis band's direct-to-main commits are ungated again "
        "(#1460)."
    )
