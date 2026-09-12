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
    with pytest.raises(ValueError, match="execution readiness"):
        validate_producer({**identity, "execution_readiness_sha256": "changed"}, identity)


def test_canonical_input_cannot_change_or_switch_rows_after_capture(tmp_path):
    import torch

    from explore_persona_space.analysis.workspace_artifacts import validate_context_input
    from explore_persona_space.analysis.workspace_runtime import file_sha256

    identity = {
        "config_sha256": "c",
        "selection_sha256": "s",
        "model_role": "primary",
        "versions": {},
        "code": {"git_commit": "a"},
    }
    path = tmp_path / "canonical.pt"
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    payload = {
        "x": x,
        "contract": {
            "identity": identity,
            "source_hashes": ["gen0", "gen1"],
            "policy": "context_only_frozen_order_batches16_no_answer_tokens",
        },
    }
    torch.save(payload, path)
    ref = {
        "context_input_file": path.name,
        "context_input_file_sha256": file_sha256(path),
        "context_input_row": 0,
        "generation_file_sha256": "gen0",
    }
    validate_context_input(ref, x[0], tmp_path, identity)
    with pytest.raises(ValueError, match="generation identity"):
        validate_context_input({**ref, "context_input_row": 1}, x[0], tmp_path, identity)
    with pytest.raises(AssertionError):
        validate_context_input(ref, x[0] + 0.1, tmp_path, identity)
    payload["x"] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_context_input(ref, x[0], tmp_path, identity)
