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

from tests.issue_skill_source import issue_skill_text

ROOT = Path(__file__).resolve().parents[1]


def test_claude_md_ownership_probe_exemplar_is_bracketed():
    text = (ROOT / "CLAUDE.md").read_text()
    assert "pgrep -af '<distinctive invocatio[n]>'" in text
    assert "bracket one pattern character" in text
    assert "pgrep -af '<distinctive invocation>'" not in text


def test_issue_skill_probe_exemplars_are_bracketed():
    text = issue_skill_text()
    for pat in (
        "pgrep -af '[p]ytest.*step9c-junit-issue-<N>'",
        "pgrep -f 'step9c_baseline[.]py refresh'",
        "pgrep -af 'issue-<N>-lint-gat[e]'",
        "pgrep -af 'issue-<N>-surgical-gat[e]|issue-<N>-lint-gat[e]'",
        "pgrep -f '<distinctive invocatio[n]>'",
    ):
        assert pat in text, pat
    # Task #1719 replaced the fleet-wide `pgrep -af 'scripts/workflow_lint[.]py'`
    # probe with an issue-scoped gate-tree shape; task #2256 re-keyed the
    # kill-arm exemplars to the whole-life workload SCRIPT-path tokens
    # (`issue-<N>-lint-gat[e]`, and the surgical
    # `issue-<N>-surgical-gat[e]|issue-<N>-lint-gat[e]` union) — under the
    # #2115 script-file launcher the old tree token vanished from argv during
    # the TG mapped-test legs. The fleet-wide pattern must NOT be
    # re-introduced as a live probe recipe.
    assert "pgrep -af 'scripts/workflow_lint[.]py'" not in text
