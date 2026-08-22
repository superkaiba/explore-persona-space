"""Durability pin for the #2119 manifest-first consumer clause in upload-policy.md.

The primary #2119 deliverable is prose in a ~60 KB on-demand rule file that
periodic compaction passes trim — this pin makes a silent drop of the consumer
clause test-breaking. Registered in ``WORKFLOW_INVARIANT``
(``scripts/select_step9c_tests.py``) + ``tests/step9c_workflow_invariant_manifest.txt``
so it gates every Step 9c run (rules-file diffs have no per-file test map).
Presence checks are count-robust by design (substring ``in``, never a count).
"""

from pathlib import Path


def _upload_policy_md() -> Path:
    return Path(__file__).resolve().parent.parent / ".claude" / "rules" / "upload-policy.md"


def test_consumer_clause_present() -> None:
    text = _upload_policy_md().read_text(encoding="utf-8")

    # (i) The clause header token (bold-led paragraph in the persist-by-default
    # neighborhood).
    header = "Consumers of shardable text artifacts resolve names MANIFEST-FIRST"
    assert header in text, f"missing clause header token: {header!r}"

    # (ii) The fail-loud sentence token — a missing part under an existing
    # manifest must never fall back to the unsharded name (#2054 incident 2).
    fail_loud = "missing PART under an existing manifest is FAIL-LOUD"
    assert fail_loud in text, f"missing fail-loud sentence token: {fail_loud!r}"

    # (iii) Both shared helper names, so the clause keeps pointing consumers at
    # the mechanical implementation instead of hand-rolled resolves.
    for helper in ("stage_sharded_text", "resolve_sharded_text_paths"):
        assert helper in text, f"missing helper name: {helper!r}"
