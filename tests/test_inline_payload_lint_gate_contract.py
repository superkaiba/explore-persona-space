"""Pin the inline-payload-lint-gate duty text across its four sites (#1460).

Inline free-analysis rounds (the CLAUDE.md § Routing "User-chat inline
free analysis" carve-out + `/issue` Step 9a-ter zero-GPU auto-runs)
commit analysis scripts DIRECTLY to `main`, deliberately skipping the
worktree -> code-review -> Step 10d pipeline. That channel historically
had NO lint duty: two inline-landed bare `.list_repo_tree(` scripts
(incident #1388) broke `tests/test_workflow_lint.py` on pristine main
fleet-wide, and >=4 sessions burned rounds re-classifying it as
pre-existing at every Step 9c gate.

Task #1460 adds the **inline payload lint gate** — the canonical recipe
lives in `.claude/skills/issue/SKILL.md` Step 9a-ter (no-flags
`workflow_lint.py` + the `select_step9c_tests.py --map-files` mapped
scan-test leg, verdict payload-attributed with an instrument-ran
completeness check, never a bare exit-0, fail CLOSED on a dead/silent
leg) — with one-line pointers in `CLAUDE.md` (the Same-turn completion
contract clause), `.claude/agents/analyzer.md` (the figure-commit
script-sweep channel), and `.claude/skills/issue-v2/SKILL.md` (the v2
zero-GPU band).

These tests pin the duty text at all four sites so future prose churn
cannot silently drop it (the #1134/#1045 drift class). Pure file-content
pins — no import of `workflow_lint` (module-registration import trap).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
ANALYZER = ROOT / ".claude" / "agents" / "analyzer.md"
ISSUE_V2_SKILL = ROOT / ".claude" / "skills" / "issue-v2" / "SKILL.md"

ANCHOR = "Inline payload lint gate"


def test_skill_9a_ter_carries_inline_payload_lint_gate() -> None:
    """SKILL.md carries the canonical gate block: recipe + verdict contract.

    Within a 4,000-char window after the FIRST `Inline payload lint gate`
    occurrence, the block must name both legs (`scripts/workflow_lint.py`
    + `select_step9c_tests.py --map-files`), the payload-attributed
    verdict, the never-bare-exit-0 contract, the instrument-ran
    completeness check (`workflow_lint: PASS` terminal-line evidence),
    and the fail-closed `INCONCLUSIVE` token.
    """
    text = ISSUE_SKILL.read_text(encoding="utf-8")
    idx = text.find(ANCHOR)
    assert idx != -1, (
        f"SKILL.md lost the '{ANCHOR}' canonical block (Step 9a-ter step 3, "
        "#1460) — the inline direct-to-main commit channel is ungated again."
    )
    window = text[idx : idx + 4000]
    for needle, why in (
        ("scripts/workflow_lint.py", "the no-flags lint leg"),
        ("select_step9c_tests.py", "the #1147 mapped scan-test leg"),
        ("--map-files", "the mapped-test invocation flag"),
        ("payload-attributed", "the payload-attribution verdict framing"),
        ("bare exit-0", "the never-bare-exit-0 contract (main can be red)"),
        ("workflow_lint: PASS", "the instrument-ran completeness evidence"),
        ("INCONCLUSIVE", "the fail-closed dead-instrument token"),
        (
            "[0-9]+ (passed|failed|error|xpassed|xfailed)|no tests ran",
            "the pytest-summary-shaped completeness pattern (a lint-leg "
            "`FAIL (N error(s))` line must not satisfy the pytest-leg check)",
        ),
    ):
        assert needle in window, (
            f"SKILL.md '{ANCHOR}' block lost {why!s} ({needle!r} not found "
            "within 4,000 chars of the anchor) — the gate contract drifted "
            "(#1460)."
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
