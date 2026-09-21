"""Paid retries must fit the original cumulative allowance and deadline."""

import copy

import pytest

from scripts.story_persona_storage_contract import (
    allocation_seconds,
    contract_from_pod,
    upload_contract,
)


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


def test_eighty_minute_retry_after_both_failed_allocations():
    state = ledger()
    state["allocations"].append(
        {
            "pod_id": "serialization-failure",
            "gpu_count": 8,
            "paid_start_unix": 1789838653.349,
            "termination_confirmed_at_unix": 1789840334.5422423,
            "gpu_hours_upper_bound": 3.735984982914395,
        }
    )
    before = copy.deepcopy(state)
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=4800) == 4800
    with pytest.raises(ValueError, match="remaining"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=6600)
    assert state == before


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


def renewed_ledger():
    state = ledger()
    state["max_gpu_hours"] = 45
    state["singleton_authorization"] = {
        "approved": True,
        "max_new_allocations": 1,
        "max_allocation_seconds": 12600,
        "max_gpu_hours": 45,
        "prior_pod_ids": [entry["pod_id"] for entry in state["allocations"]],
    }
    return state


def test_renewed_singleton_allowance_accepts_one_bounded_allocation():
    state = renewed_ledger()
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=12600) == 12600
    state["allocations"].append(
        {"pod_id": "new", "gpu_count": 8, "paid_start_unix": 5000, "deadline_unix": 17600}
    )
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=12600) == 12600
    with pytest.raises(ValueError, match="already been used"):
        allocation_seconds(state, pod_id="second", gpu_count=8, requested_seconds=1000)
    state["allocations"][-1].update(
        termination_confirmed_at_unix=6000, gpu_hours_upper_bound=8000 / 3600
    )
    with pytest.raises(ValueError, match="already been used"):
        allocation_seconds(state, pod_id="second", gpu_count=8, requested_seconds=1000)
    with pytest.raises(ValueError, match="terminated"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=12600)


def test_renewed_allowance_still_counts_all_historical_spend():
    state = renewed_ledger()
    state["allocations"][0]["gpu_hours_upper_bound"] = 18
    with pytest.raises(ValueError, match="remaining"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=12600)


@pytest.mark.parametrize("defect", ["no_authority", "too_long", "wrong_gpu_count", "old_pod"])
def test_renewed_allowance_cannot_expand_authorization(defect):
    state = renewed_ledger()
    if defect == "no_authority":
        del state["singleton_authorization"]
    with pytest.raises(ValueError):
        allocation_seconds(
            state,
            pod_id="qwen" if defect == "old_pod" else "new",
            gpu_count=1 if defect == "wrong_gpu_count" else 8,
            requested_seconds=12601 if defect == "too_long" else 12600,
        )


@pytest.mark.parametrize("mode", ["unpadded_singleton", "padded_batch"])
def test_renewed_provider_contract_requires_singleton_source(mode):
    pod = {
        "id": "new",
        "name": "pod-2673-crossmodel-deepseek",
        "desiredStatus": "RUNNING",
        "gpuCount": 8,
        "machine": {"gpuTypeId": "NVIDIA H200"},
        "volumeInGb": 1000,
        "containerDiskInGb": 50,
        "volumeMountPath": "/workspace",
        "createdAt": "2026-09-21T00:00:00Z",
    }
    args = dict(
        pod_id="new",
        expected_name=pod["name"],
        model_key="deepseek",
        model={
            "id": "deepseek-ai/DeepSeek-V3.1-Base",
            "revision": "d3d4eafdc470de44bbf6f0a74f852eb522357be8",
            "execution_mode": mode,
        },
        source_sha="a" * 40,
        now=1789948810,
        requested_seconds=12600,
        ledger=renewed_ledger(),
    )
    if mode == "padded_batch":
        with pytest.raises(ValueError, match="singleton execution"):
            contract_from_pod(pod, **args)
    else:
        result = contract_from_pod(pod, **args)
        assert result["max_allocation_seconds"] == 12600
        assert result["deadline_unix"] - result["paid_start_unix"] == 12600
