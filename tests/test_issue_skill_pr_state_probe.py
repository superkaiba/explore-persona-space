"""Prose-side pin for the Step 10d PR-state probe + landing verification (#1897).

``.claude/skills/issue/SKILL.md`` Step 10d must never report merge success
(and consume the SHA-bound lint verdict) when ``gh pr merge`` exits 0
against a PR object a PRIOR round already merged/closed (incident #1768
round-2: ``gh pr merge 1527 --rebase`` ran against the round-1 PR, exited
0 with "was already merged", the success arm consumed the verdict, and the
22-commit round-2 payload stayed stranded off main). This test pins:

- the PR-object liveness probe at the safe-case merge entry (one
  ``gh pr view issue-<N> --json number,state,mergedAt`` read; a non-OPEN
  state routes to a fresh pre-checked draft PR);
- the ``Landing verification (#1897)`` read in BOTH ``gh pr merge``
  success arms (safe case + merge-conflict recovery);
- the verified-consume contract (``consume on VERIFIED merge success``)
  and the MERGE NOT VERIFIED failure arm (verdict survives; bounded
  re-entry via the probe);
- the rebind propagation (critic Must-Fix 1): ``gh pr merge "$PR"``
  inside the safe-case block with NO bare ``gh pr merge <PR>`` remaining
  there (block-scoped — ``<PR>`` placeholders elsewhere in the file are
  compose-time substitution sites and out of scope);
- the payload-scoped Idempotent bullet + the exit-0 false-success prose
  (shapes 0-3 key only on non-zero exits).
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

_SAFE_CASE_HEADING = (
    "#### The auto-merge procedure "
    "(safe case: guard 3 clean — mainline-based, own commits in scope)"
)
# First prose line after the safe-case fenced block closes.
_SAFE_CASE_END = "The `gh pr merge --rebase` form lands all per-item commits"

_RECOVERY_HEADING = "#### Merge-conflict recovery (safe case: `gh pr merge` refuses)"
_RECOVERY_END = "##### Residual-conflict subagent dispatch"


def _text() -> str:
    return issue_skill_text()


def _safe_case_block() -> str:
    """The safe-case auto-merge section (heading through its closing fence)."""
    text = _text()
    start = text.index(_SAFE_CASE_HEADING)  # ValueError if the heading is gone
    end = text.index(_SAFE_CASE_END, start)
    return text[start:end]


def _recovery_block() -> str:
    """The merge-conflict recovery section (heading through its closing fence)."""
    text = _text()
    start = text.index(_RECOVERY_HEADING)
    end = text.index(_RECOVERY_END, start)
    return text[start:end]


def test_pr_state_probe_anchors_present() -> None:
    # Durability pin (#1897): the probe + its one-read state resolve at the
    # safe-case merge entry.
    block = _safe_case_block()
    assert "PR-object liveness probe" in block
    assert "--json number,state,mergedAt" in block
    assert 'if [ "$PR_STATE" != "OPEN" ]; then' in block
    # The fresh draft PR is gated on the layered NOVEL-payload predicate
    # (#1897 round-2): a bare commit count is patch-blind — rebase/squash
    # land COPIES, so a fully-merged branch reads count>0 forever. The
    # rev-list read survives only as the cheap zero-commits short-circuit;
    # `git cherry` (rebase form) + the own-files content check (squash
    # form) carry the landed detection.
    assert "rev-list --count origin/main..issue-<N>" in block
    assert 'git -C "$WT" cherry origin/main issue-<N>' in block
    assert "NOVEL_PAYLOAD=yes" in block
    assert 'if [ "$NOVEL_PAYLOAD" = "yes" ]; then' in block
    assert "gh pr create --draft --head issue-<N>" in block


def test_landing_verification_in_both_success_arms() -> None:
    assert "Landing verification (#1897)" in _safe_case_block()
    assert "Landing verification (#1897)" in _recovery_block()
    # Both arms verify via the PR object's state/mergedAt freshness.
    assert _safe_case_block().count("LANDED_OK=yes") >= 1
    assert _recovery_block().count("LANDED_OK=yes") >= 1


def test_verdict_consumed_only_on_verified_success() -> None:
    # Exactly the two success arms carry the verified-consume comment.
    assert _text().count("consume on VERIFIED merge success") == 2
    assert "consume on VERIFIED merge success" in _safe_case_block()
    assert "consume on VERIFIED merge success" in _recovery_block()


def test_merge_not_verified_failure_arm_present() -> None:
    # The failure arm never silently consumes the verdict / swallows the
    # false success: echo + false in BOTH arms, verdict survives.
    assert "MERGE NOT VERIFIED" in _safe_case_block()
    assert "MERGE NOT VERIFIED" in _recovery_block()
    assert _text().count("Verdict NOT consumed") == 2


def test_rebind_propagation_in_safe_case_block() -> None:
    # Critic Must-Fix 1: every downstream merge-path PR ref inside the
    # safe-case block uses the probe-rebound "$PR" — compose-time <PR>
    # substitution is the #1768 round-2 mechanism.
    block = _safe_case_block()
    assert 'gh pr merge "$PR"' in block
    assert "gh pr merge <PR>" not in block
    assert 'gh pr ready "$PR"' in block
    assert "gh pr ready <PR>" not in block
    assert 'gh pr view "$PR"' in block


def test_recovery_arm_binds_pre_merged_at() -> None:
    # The recovery fence is a separate shell — it binds its own
    # PRE_MERGED_AT from the extended pre-merge read.
    block = _recovery_block()
    assert "--json mergeable,state,mergedAt" in block
    assert "PRE_MERGED_AT" in block


def test_idempotent_bullet_is_payload_scoped() -> None:
    # #1897 round-2: the skip predicate is NOVEL-payload, not a bare
    # commit count (rebase/squash land copies -> count>0 forever on a
    # fully-merged branch).
    text = _text()
    assert "AND the branch carries no NOVEL payload vs fetched" in text
    assert "payload-scoped, #1897" in text
    assert "layered novel-payload predicate" in text


def test_exit0_false_success_prose_documented() -> None:
    # Shapes 0-3 key on non-zero exits; the exit-0 already-merged shape has
    # its own known-failure paragraph.
    text = _text()
    assert "Exit-0 false success" in text
    assert "#1768 round-2 / #1897" in text
