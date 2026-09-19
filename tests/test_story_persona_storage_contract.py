"""Paid retries must fit the original cumulative allowance and deadline."""

import copy

import pytest

from scripts.story_persona_storage_contract import allocation_seconds, upload_contract


def ledger():
    return {
        "max_gpu_hours": 17,
        "allocations": [
            {
                "pod_id": "qwen",
                "gpu_count": 1,
                "paid_start_unix": 100,
                "termination_confirmed_at_unix": 2455,
                "gpu_hours_upper_bound": 2355 / 3600,
            },
            {
                "pod_id": "failed",
                "gpu_count": 8,
                "paid_start_unix": 3000,
                "termination_confirmed_at_unix": 3554,
                "gpu_hours_upper_bound": 4432 / 3600,
            },
        ],
    }


def test_shortened_retry_fits_but_original_two_hours_does_not():
    state = ledger()
    before = copy.deepcopy(state)
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6600) == 6600
    assert state == before
    with pytest.raises(ValueError, match="remaining"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


def test_unresolved_or_underreported_previous_spend_fails_closed():
    state = ledger()
    state["allocations"][1]["gpu_hours_upper_bound"] = 0
    with pytest.raises(ValueError, match="understates"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6600)
    del state["allocations"][1]["termination_confirmed_at_unix"]
    with pytest.raises(ValueError, match="unresolved"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6600)


def test_existing_allocation_cannot_extend_deadline():
    state = ledger()
    state["allocations"].append({"pod_id": "new", "paid_start_unix": 5000, "deadline_unix": 11600})
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6600) == 6600
    with pytest.raises(ValueError, match="deadline"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6700)


@pytest.mark.parametrize("seconds", [0, 900, 7201])
def test_invalid_allocation_bounds(seconds):
    with pytest.raises(ValueError):
        allocation_seconds(ledger(), pod_id="new", gpu_count=8, requested_seconds=seconds)


def test_upload_rechecks_personal_scope_and_rejects_changed_pod():
    class Client:
        @staticmethod
        def graphql(query, variables, *, personal):
            assert personal is True and variables == {"id": "new"}
            return {"pod": {"id": "wrong", "desiredStatus": "RUNNING"}}

    with pytest.raises(RuntimeError, match="changed state"):
        upload_contract(Client(), "new", None, None, None, None)
