"""Rule-file pin: `.claude/rules/workflow-fix-on-bug.md` § Recently-closed-sibling
describes the CURRENT composite-blocking contract, not the retired
"advisory only, never a block" wording (#1735 §4.3).

This pin lives in :data:`scripts/select_step9c_tests.py::WORKFLOW_INVARIANT` so
a later diff to the rule file re-runs it in the Step 9c gate (per CLAUDE.md
§ Workflow-prose durability pin: an unregistered new pin file never runs on a
later SKILL.md/rule diff; #1242/#1268/#1546).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_PATH = REPO_ROOT / ".claude" / "rules" / "workflow-fix-on-bug.md"


def _rule_text() -> str:
    return RULE_PATH.read_text(encoding="utf-8")


def test_rule_describes_composite_blocking_contract() -> None:
    """The reconciled rule text carries the composite target+title blocking
    phrasing and the bare-target/bare-title advisory phrasing (#1735 §4.3),
    and drops the retired "advisory only, never a block" sentence + the
    "never changes exit codes" phrasing that contradicted the code's
    blocking behaviour.
    """
    # The rule file uses markdown-style line wrapping, so compare against a
    # whitespace-collapsed copy for cross-line phrase assertions (single-line
    # substrings like the section anchors stay grep-based against `text`).
    text = _rule_text()
    collapsed = " ".join(text.split())

    # New heading form — names the composite predicate directly.
    assert "Recently-closed-sibling SUSPECT probe" in text, (
        "reconciled heading missing (§4.3 rule/code reconciliation)"
    )
    assert "blocking on the composite target+title arm" in collapsed, (
        "composite blocking phrasing missing from heading"
    )
    assert "advisory on bare-target or bare-title" in collapsed, (
        "bare-target/bare-title advisory phrasing missing from heading"
    )

    # Body — composite predicate stated + the calibrated stoplist named.
    assert "BOTH a path-family arm" in collapsed and "AND a title-family arm" in collapsed, (
        "composite predicate (BOTH path AND title) not stated in body"
    )
    assert "CLOSED_SIBLING_TITLE_STOPWORDS" in text, (
        "driver-scoped stoplist symbol not named in body"
    )
    assert "--retry-suspects" in text, "--retry-suspects override not documented in body"

    # Retired phrasings — must NOT reappear as new advisory-only claims. The
    # heading no longer carries them, and the body's soft-fail sentence must
    # not resurrect the "never blocks the filing" wording that contradicted
    # the driver-level probe. (The lingering "the rule above is unchanged (a
    # closed fix still never blocks a genuine re-raise)" prose refers to the
    # exact-`(target_file, fingerprint)` OPEN dedup predicate, not the
    # closed-sibling probe — retained by design.)
    assert "advisory only, never a block" not in text, (
        "retired 'advisory only, never a block' phrasing must be dropped (§4.3)"
    )
    assert "never blocks the filing" not in text, (
        "retired 'never blocks the filing' phrasing must be dropped (§4.3)"
    )
    assert "never changes exit codes" not in text, (
        "retired 'never changes exit codes' phrasing must be dropped (§4.3)"
    )

    # Post-hoc remedy paragraph — must be retained verbatim (plan §4.3 says
    # "keep the post-hoc remedy paragraph"), since the filer's stderr
    # advisory arm is still fail-soft even under the composite rule.
    assert "task.py set-status <id> archived" in text, (
        "post-hoc remedy (archive just-filed task) must be retained (§4.3)"
    )
    assert "spawn_session.py stop --session-id <sid>" in text, (
        "post-hoc remedy (stop spawned session) must be retained (§4.3)"
    )

    # Terminal SUMMARY + daily-drive-summary row are documented (plan §4.4).
    assert "daily-drive-summary" in text, (
        "SUMMARY ledger-row outcome name not documented in the rule (§4.4)"
    )
