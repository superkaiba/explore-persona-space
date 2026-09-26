"""The renewed Kimi grant permits exactly one bounded successor to the closed run."""

import copy
import datetime as dt

import pytest

from scripts.story_persona_storage_contract import allocation_seconds, contract_from_pod


def recovery_ledger():
    """Use the actual closed allocation's accounting and the explicit new grant."""
    return {
        "model_key": "kimi",
        "max_gpu_hours": 42.65690570619371,
        "kimi_authorization": {
            "approved": True,
            "max_new_allocations": 1,
            "max_allocation_seconds": 12600,
            "user_request": "Do both. Run it now",
        },
        "kimi_recovery_authorization": {
            "approved": True,
            "user_request": "approve and continue until done",
            "prior_pod_ids": ["2y9tr3io3gd0os"],
            "max_new_allocations": 1,
            "max_allocation_seconds": 7200,
            "max_additional_gpu_hours": 16,
        },
        "allocations": [
            {
                "pod_id": "2y9tr3io3gd0os",
                "gpu_count": 8,
                "paid_start_unix": 1790371940.801,
                "deadline_unix": 1790384540.801,
                "termination_confirmed_at_unix": 1790383936.408568,
                "gpu_hours_upper_bound": 26.65690570619371,
            }
        ],
    }


def test_one_recovery_keeps_prior_accounting_and_existing_recovery_deadline():
    state = recovery_ledger()
    before = copy.deepcopy(state)
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200) == 7200
    assert state == before
    state["allocations"].append(
        {
            "pod_id": "new",
            "gpu_count": 8,
            "paid_start_unix": 1790389000,
            "deadline_unix": 1790396200,
        }
    )
    assert allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200) == 7200
    with pytest.raises(ValueError, match="deadline"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7100)
    with pytest.raises(ValueError, match="already been used"):
        allocation_seconds(state, pod_id="third", gpu_count=8, requested_seconds=1000)
    state["allocations"][-1].update(
        termination_confirmed_at_unix=1790389100, gpu_hours_upper_bound=800 / 3600
    )
    with pytest.raises(ValueError, match="already been used"):
        allocation_seconds(state, pod_id="third", gpu_count=8, requested_seconds=1000)
    with pytest.raises(ValueError, match="terminated"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approved", False),
        ("user_request", "Do both. Run it now"),
        ("prior_pod_ids", []),
        ("prior_pod_ids", ["different"]),
        ("prior_pod_ids", ["2y9tr3io3gd0os", "2y9tr3io3gd0os"]),
        ("max_new_allocations", 2),
        ("max_allocation_seconds", 12600),
        ("max_additional_gpu_hours", 28),
    ],
)
def test_recovery_grant_is_explicit_and_cannot_expand(field, value):
    state = recovery_ledger()
    state["kimi_recovery_authorization"][field] = value
    with pytest.raises(ValueError, match="explicit bounded"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)
    del state["kimi_recovery_authorization"][field]
    with pytest.raises(ValueError, match="explicit bounded"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


@pytest.mark.parametrize("seconds", [0, 900, 7201, 12600])
def test_recovery_duration_is_bounded(seconds):
    with pytest.raises(ValueError, match="approved limit"):
        allocation_seconds(recovery_ledger(), pod_id="new", gpu_count=8, requested_seconds=seconds)


@pytest.mark.parametrize("count", [1, 4, 16])
def test_recovery_requires_eight_gpus(count):
    with pytest.raises(ValueError, match="single-allocation"):
        allocation_seconds(recovery_ledger(), pod_id="new", gpu_count=count, requested_seconds=7200)


def test_recovery_cannot_reuse_or_rebase_prior_pod():
    state = recovery_ledger()
    with pytest.raises(ValueError, match="already been used"):
        allocation_seconds(state, pod_id="2y9tr3io3gd0os", gpu_count=8, requested_seconds=7200)
    state["allocations"][0]["pod_id"] = "different"
    state["kimi_recovery_authorization"]["prior_pod_ids"] = ["different"]
    with pytest.raises(ValueError, match="explicit bounded"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


def test_missing_grant_does_not_renew_original_allowance():
    state = recovery_ledger()
    del state["kimi_recovery_authorization"]
    with pytest.raises(ValueError, match="single-allocation"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)
    state["max_gpu_hours"] = 28
    with pytest.raises(ValueError, match="single-allocation"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)
    state["allocations"] = []
    assert allocation_seconds(state, pod_id="first", gpu_count=8, requested_seconds=12600) == 12600


@pytest.mark.parametrize("defect", ["unresolved", "underreported", "deadline", "gpu", "duplicate"])
def test_prior_allocation_cannot_be_ignored_or_rewritten(defect):
    state = recovery_ledger()
    previous = state["allocations"][0]
    if defect == "unresolved":
        del previous["termination_confirmed_at_unix"]
    elif defect == "underreported":
        previous["gpu_hours_upper_bound"] = 0
    elif defect == "deadline":
        previous["deadline_unix"] += 1
    elif defect == "gpu":
        previous["gpu_count"] = 4
    else:
        state["allocations"].append(copy.deepcopy(previous))
    with pytest.raises(ValueError):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


def test_recovery_counts_prior_spend_and_does_not_use_the_broader_45_hour_guard():
    state = recovery_ledger()
    state["allocations"][0]["gpu_hours_upper_bound"] += 0.01
    with pytest.raises(ValueError, match="remaining cumulative"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)
    state = recovery_ledger()
    state["max_gpu_hours"] = 45
    with pytest.raises(ValueError, match="approved additional"):
        allocation_seconds(state, pod_id="new", gpu_count=8, requested_seconds=7200)


def test_recovery_provider_contract_cannot_reset_paid_start_or_change_recipe():
    state = recovery_ledger()
    created_at = "2026-09-26T03:00:00Z"
    paid_start = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    pod = {
        "id": "new",
        "name": "pod-2673-crossmodel-kimi",
        "desiredStatus": "RUNNING",
        "gpuCount": 8,
        "machine": {"gpuTypeId": "NVIDIA H200"},
        "volumeInGb": 1000,
        "containerDiskInGb": 50,
        "volumeMountPath": "/workspace",
        "createdAt": created_at,
    }
    args = dict(
        pod_id="new",
        expected_name=pod["name"],
        model_key="kimi",
        model={
            "id": "moonshotai/Kimi-K2.6",
            "revision": "b" * 40,
            "execution_mode": "unpadded_singleton",
        },
        source_sha="a" * 40,
        now=paid_start + 10,
        requested_seconds=7200,
        ledger=state,
    )
    contract = contract_from_pod(pod, **args)
    assert contract["paid_start_unix"] == paid_start
    assert contract["deadline_unix"] == paid_start + 7200
    assert contract["api_gpu_count"] == 8
    state["allocations"].append(
        {
            "pod_id": "new",
            "gpu_count": 8,
            "paid_start_unix": paid_start,
            "deadline_unix": paid_start + 7200,
        }
    )
    args["now"] += 200
    assert contract_from_pod(pod, **args)["deadline_unix"] == paid_start + 7200
    pod["createdAt"] = "2026-09-26T03:01:00Z"
    with pytest.raises(ValueError, match="reset the recorded"):
        contract_from_pod(pod, **args)
    pod["createdAt"] = created_at
    args["model"]["execution_mode"] = "padded_batch"
    with pytest.raises(ValueError, match="singleton execution"):
        contract_from_pod(pod, **args)
