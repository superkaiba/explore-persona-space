"""Prose durability pin for the #1205 result-push verification contract.

The mechanical GCE backstop lives in ``backends/gcp.py`` (pinned by
``tests/test_gcp_backend.py``'s push-verify tests); the CONTRACT that
dispatch scripts on every lane verify their own push lives as prose in
``.claude/rules/pod-side-reporting.md``. Prose contracts drift silently
(the reason ``--check-piped-git-push`` exists), so this pin fails loud if
a future edit drops the section, the rev-list invariant, or the
banned-shape sentence.
"""

from __future__ import annotations

from pathlib import Path

_RULE_PATH = Path(__file__).resolve().parent.parent / ".claude" / "rules" / "pod-side-reporting.md"


def test_rule_carries_push_verification_contract() -> None:
    """The #1205 contract section exists with its three load-bearing
    elements: the H3 heading, the rev-list push-landed invariant, and the
    banned failure-swallow shape."""
    text = _RULE_PATH.read_text(encoding="utf-8")
    assert "### Result-push verification contract (#1205)" in text
    assert "rev-list --count origin/<branch>..HEAD" in text
    assert "`git push … || echo WARNING` / `|| true` shape is" in text
    assert "**BANNED**" in text


def test_rule_names_both_lanes_and_the_sentinel_interplay() -> None:
    """The contract distinguishes the GCE mechanical backstop (exit 86 +
    bundle) from the RunPod no-mechanical-backstop stance, and carries the
    Part A-ter deliverables-sentinel ordering rule."""
    text = _RULE_PATH.read_text(encoding="utf-8")
    section = text.split("### Result-push verification contract (#1205)", 1)[1]
    section = section.split("\n### ", 1)[0]
    assert "exit 86" in section
    assert "data/issue_<N>/" in section
    assert "NO mechanical backstop" in section
    assert "EPS_DELIVERABLES_OK_PATH" in section
    assert "finalize_failed_artifacts_ok" in section


def test_rule_names_slurm_lane() -> None:
    """The contract's SLURM lane bullet (#1240) lives inside the section:
    the lane has no workload-side push at all (no git checkout on the
    ``$SCRATCH`` rsync copy — a push attempt fails loud), so the section
    must document the third lane rather than staying GCE/RunPod-only."""
    text = _RULE_PATH.read_text(encoding="utf-8")
    section = text.split("### Result-push verification contract (#1205)", 1)[1]
    section = section.split("\n### ", 1)[0]
    assert "**SLURM lane:**" in section
    assert "not a git repository" in section
