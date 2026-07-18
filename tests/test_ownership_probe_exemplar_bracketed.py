"""Pin the self-match-safe (bracketed) ownership/liveness probe exemplars (#1495).

Probe exemplars in always-on / skill surfaces are copied verbatim into inline
Bash-tool compounds and SSH-remote commands, where the wrapping shell's argv
carries the pattern text; an unbracketed exemplar phantom-matches the probing
shell itself (#1335; gotchas.md SSH-remote ownership-probe entry).

Note: the negative assert below deliberately flags ANY re-introduction of the
unbracketed CLAUDE.md exemplar, including a future descriptive quotation —
quote the failing shape in gotchas.md instead.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_claude_md_ownership_probe_exemplar_is_bracketed():
    text = (ROOT / "CLAUDE.md").read_text()
    assert "pgrep -af '<distinctive invocatio[n]>'" in text
    assert "bracket one pattern character" in text
    assert "pgrep -af '<distinctive invocation>'" not in text


def test_issue_skill_probe_exemplars_are_bracketed():
    text = (ROOT / ".claude" / "skills" / "issue" / "SKILL.md").read_text()
    for pat in (
        "pgrep -af '[p]ytest.*step9c-junit-issue-<N>'",
        "pgrep -f 'step9c_baseline[.]py refresh'",
        "pgrep -af 'issue-<N>-lint-gate-tre[e]'",
        "pgrep -af 'scripts/workflow_lint[.]py'",
        "pgrep -f '<distinctive invocatio[n]>'",
    ):
        assert pat in text, pat
