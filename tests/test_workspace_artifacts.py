"""Reject stale identities and unaccounted-for missing scientific observations."""

from copy import deepcopy

import pytest

from explore_persona_space.analysis.workspace_artifacts import validate_coverage, validate_producer


def test_contract_and_complete_coverage_fail_closed():
    """A missing file is not an exclusion; model/revision changes cannot be relabeled."""
    identity = {
        "config_sha256": "a",
        "selection_sha256": "b",
        "model_role": "primary",
        "versions": {"torch": "pinned"},
        "code": {"git_commit": "c"},
    }
    report = {
        "identity": identity,
        "status": "complete",
        "planned_contexts": 3,
        "included_prompt_sha256": ["a", "b"],
        "exclusions": [{"prompt_sha256": "c", "reason": "empty_answer"}],
    }
    assert validate_coverage(report, ["a", "b", "c"], identity) == ["a", "b"]
    missing = deepcopy(report)
    missing["exclusions"] = []
    with pytest.raises(ValueError, match="reconcile"):
        validate_coverage(missing, ["a", "b", "c"], identity)
    unfinished = {**report, "status": "running"}
    with pytest.raises(ValueError, match="complete"):
        validate_coverage(unfinished, ["a", "b", "c"], identity)
    for field in identity:
        wrong = {**identity, field: "wrong"}
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_producer(wrong, identity)
