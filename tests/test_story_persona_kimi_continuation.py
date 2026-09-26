"""Inactive continuation validation; all future closures here are synthetic."""

import copy
import datetime as dt
import math

import pytest

from scripts.story_persona_storage_contract import (
    allocation_seconds,
    contract_from_pod,
    validate_kimi_continuation_append,
)
from tests.test_story_persona_kimi_recovery import replacement_ledger

FIELDS = (
    "pod_id",
    "gpu_count",
    "paid_start_unix",
    "deadline_unix",
    "termination_confirmed_at_unix",
    "gpu_hours_upper_bound",
)


def add_grant(state):
    """Construct a test-only grant from synthetic, fully closed accounting."""
    spent = sum(entry["gpu_hours_upper_bound"] for entry in state["allocations"])
    seconds = min(6525, math.floor((state["max_gpu_hours"] - spent) * 3600 / 8))
    grant = {
        "approved": True,
        "user_request": "just continue until it succeeds",
        "max_new_allocations": 1,
        "max_allocation_seconds": seconds,
        "gpu_count": 8,
        "max_cumulative_gpu_hours": 42.65690570619371,
        "diagnosis": "Synthetic fixture: prior bootstrap failed before model start",
        "failure_evidence": "TEST-ONLY synthetic receipt; not provider evidence",
        "prior_allocations": [
            {key: entry[key] for key in FIELDS} for entry in state["allocations"]
        ],
    }
    state.setdefault("kimi_continuation_grants", []).append(grant)
    return seconds


def continuation_ledger():
    """Use actual first two records and a clearly synthetic closed third record."""
    state = replacement_ledger()
    state["allocations"].append(
        {
            "pod_id": "synthetic-third",
            "gpu_count": 8,
            "paid_start_unix": 1790393106.876,
            "deadline_unix": 1790399631.876,
            "termination_confirmed_at_unix": 1790393706.876,
            "gpu_hours_upper_bound": 600 * 8 / 3600,
        }
    )
    add_grant(state)
    return state


def validate(state, pod_id="fourth", seconds=None):
    """Execute the real allocation body, using the fixture grant cap by default."""
    if seconds is None:
        seconds = state["kimi_continuation_grants"][-1]["max_allocation_seconds"]
    return allocation_seconds(state, pod_id=pod_id, gpu_count=8, requested_seconds=seconds)


def add_successor(state, pod_id="fourth", *, closed=False):
    """Add synthetic provider timing under exactly the most recent grant."""
    start = state["allocations"][-1]["termination_confirmed_at_unix"] + 30
    seconds = state["kimi_continuation_grants"][-1]["max_allocation_seconds"]
    entry = {
        "pod_id": pod_id,
        "gpu_count": 8,
        "paid_start_unix": start,
        "deadline_unix": start + seconds,
    }
    if closed:
        entry.update(termination_confirmed_at_unix=start + 60, gpu_hours_upper_bound=480 / 3600)
    state["allocations"].append(entry)
    return entry


def test_fourth_is_one_bounded_successor_and_validator_never_mutates_input():
    state = continuation_ledger()
    before = copy.deepcopy(state)
    assert validate(state) == 5937
    assert state == before
    with pytest.raises(ValueError, match="approved limit"):
        validate(state, seconds=5938)
    add_successor(state)
    assert validate(state) == 5937
    with pytest.raises(ValueError, match="deadline"):
        validate(state, seconds=5900)
    with pytest.raises(ValueError, match="already been used"):
        validate(state, pod_id="fifth")


def test_fifth_needs_a_new_snapshot_grant_and_preserves_first_grant():
    state = continuation_ledger()
    add_successor(state, closed=True)
    original_grant = copy.deepcopy(state["kimi_continuation_grants"][0])
    with pytest.raises(ValueError, match="terminated"):
        validate(state)
    with pytest.raises(ValueError, match="already been used"):
        validate(state, pod_id="fifth")
    seconds = add_grant(state)
    assert validate(state, pod_id="fifth") == seconds == 5877
    assert state["kimi_continuation_grants"][0] == original_grant
    with pytest.raises(ValueError, match="already been used"):
        validate(state, pod_id="fourth")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("approved", False),
        ("user_request", "continue"),
        ("max_new_allocations", 2),
        ("max_allocation_seconds", 6526),
        ("max_allocation_seconds", 900),
        ("max_allocation_seconds", 5937.0),
        ("gpu_count", 4),
        ("max_cumulative_gpu_hours", 45),
        ("diagnosis", " "),
        ("failure_evidence", ""),
        ("prior_allocations", []),
    ],
)
def test_every_grant_field_is_required_and_bounded(key, value):
    state = continuation_ledger()
    state["kimi_continuation_grants"][0][key] = value
    with pytest.raises(ValueError):
        validate(state)
    del state["kimi_continuation_grants"][0][key]
    with pytest.raises(ValueError):
        validate(state, seconds=1000)


@pytest.mark.parametrize(
    "key", ["kimi_authorization", "kimi_recovery_authorization", "kimi_replacement_authorization"]
)
def test_continuation_cannot_discard_any_existing_authorization(key):
    state = continuation_ledger()
    state[key]["approved"] = False
    with pytest.raises(ValueError):
        validate(state)
    del state[key]
    with pytest.raises(ValueError):
        validate(state)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("field", FIELDS)
def test_every_prior_financial_field_is_exactly_bound(index, field):
    state = continuation_ledger()
    entry = state["allocations"][index]
    entry[field] = "changed" if field == "pod_id" else entry[field] + 1
    with pytest.raises(ValueError):
        validate(state)


@pytest.mark.parametrize(
    "defect",
    [
        "unresolved",
        "understated",
        "nan",
        "negative",
        "late_start",
        "duplicate",
        "missing",
        "reordered",
    ],
)
def test_invalid_prior_history_cannot_be_laundered_through_new_snapshots(defect):
    state = continuation_ledger()
    entry = state["allocations"][-1]
    if defect == "unresolved":
        entry["termination_confirmed_at_unix"] = None
    elif defect == "understated":
        entry["gpu_hours_upper_bound"] -= 0.01
    elif defect == "nan":
        entry["gpu_hours_upper_bound"] = float("nan")
    elif defect == "negative":
        entry["termination_confirmed_at_unix"] = entry["paid_start_unix"] - 1
    elif defect == "late_start":
        entry["deadline_unix"] = entry["paid_start_unix"] - 1
    elif defect == "duplicate":
        entry["pod_id"] = state["allocations"][0]["pod_id"]
    elif defect == "missing":
        state["allocations"].pop(0)
    else:
        state["allocations"].reverse()
    state["kimi_continuation_grants"][0]["prior_allocations"] = [
        {key: row[key] for key in FIELDS} for row in state["allocations"]
    ]
    with pytest.raises(ValueError):
        validate(state)


def test_grant_duration_is_limited_by_remaining_accounting_not_only_6525():
    state = continuation_ledger()
    state["kimi_continuation_grants"][0]["max_allocation_seconds"] = 6525
    with pytest.raises(ValueError, match="remaining cumulative"):
        validate(state, seconds=1000)
    state = continuation_ledger()
    state["max_gpu_hours"] -= 0.01
    with pytest.raises(ValueError, match="cumulative ceiling"):
        validate(state)
    state["max_gpu_hours"] += 0.02
    with pytest.raises(ValueError):
        validate(state)


def test_grants_cannot_skip_reuse_or_pregrant_multiple_future_allocations():
    state = continuation_ledger()
    state["kimi_continuation_grants"].append(copy.deepcopy(state["kimi_continuation_grants"][0]))
    with pytest.raises(ValueError):
        validate(state)
    state = continuation_ledger()
    for row in state["allocations"]:
        with pytest.raises(ValueError, match="already been used"):
            validate(state, pod_id=row["pod_id"])
    add_successor(state, closed=True)
    add_grant(state)
    state["kimi_continuation_grants"].pop(0)
    with pytest.raises(ValueError):
        validate(state, pod_id="fifth")


@pytest.mark.parametrize("grants", [None, {}, "yes", [None]])
def test_malformed_optional_grants_fail_loudly(grants):
    state = continuation_ledger()
    state["kimi_continuation_grants"] = grants
    with pytest.raises(ValueError):
        validate(state, seconds=1000)


def test_empty_optional_grants_preserve_existing_third_behavior():
    state = replacement_ledger()
    state["kimi_continuation_grants"] = []
    assert allocation_seconds(state, pod_id="third", gpu_count=8, requested_seconds=6525) == 6525


def test_second_grant_cannot_change_consumed_first_duration_or_snapshot():
    state = continuation_ledger()
    successor = add_successor(state, closed=True)
    add_grant(state)
    successor["deadline_unix"] += 1
    state["kimi_continuation_grants"][-1]["prior_allocations"][-1]["deadline_unix"] += 1
    with pytest.raises(ValueError, match="paid envelope"):
        validate(state, pod_id="fifth")
    state = continuation_ledger()
    add_successor(state, closed=True)
    add_grant(state)
    state["kimi_continuation_grants"][0]["prior_allocations"][2]["gpu_hours_upper_bound"] += 1
    with pytest.raises(ValueError, match="snapshot changed"):
        validate(state, pod_id="fifth")


def test_provider_contract_keeps_paid_start_deadline_and_requires_prior_closure():
    state = continuation_ledger()
    previous_end = state["allocations"][-1]["termination_confirmed_at_unix"]
    start = math.ceil(previous_end) + 30
    pod = {
        "id": "fourth",
        "name": "pod-2673-crossmodel-kimi",
        "desiredStatus": "RUNNING",
        "gpuCount": 8,
        "machine": {"gpuTypeId": "NVIDIA H200"},
        "volumeInGb": 1000,
        "containerDiskInGb": 50,
        "volumeMountPath": "/workspace",
        "createdAt": dt.datetime.fromtimestamp(start, dt.UTC).isoformat(),
    }
    args = dict(
        pod_id="fourth",
        expected_name=pod["name"],
        model_key="kimi",
        model={
            "id": "moonshotai/Kimi-K2.6",
            "revision": "b" * 40,
            "execution_mode": "unpadded_singleton",
        },
        source_sha="a" * 40,
        now=start + 10,
        requested_seconds=5937,
        ledger=state,
    )
    contract = contract_from_pod(pod, **args)
    assert contract["paid_start_unix"] == start
    assert contract["deadline_unix"] == start + 5937
    state["allocations"].append(
        {
            "pod_id": "fourth",
            "gpu_count": 8,
            "paid_start_unix": start,
            "deadline_unix": start + 5937,
        }
    )
    with pytest.raises(RuntimeError, match="insufficient remaining"):
        contract_from_pod(pod, **dict(args, now=start + 5937 - 900))
    pod["createdAt"] = dt.datetime.fromtimestamp(start + 1, dt.UTC).isoformat()
    with pytest.raises(ValueError, match="reset the recorded"):
        contract_from_pod(pod, **args)
    state["allocations"].pop()
    pod["createdAt"] = dt.datetime.fromtimestamp(previous_end - 1, dt.UTC).isoformat()
    with pytest.raises(ValueError, match="before prior termination"):
        contract_from_pod(pod, **args)


def test_append_guard_preserves_all_previous_ledger_bytes_semantically():
    updated = continuation_ledger()
    previous = copy.deepcopy(updated)
    del previous["kimi_continuation_grants"]
    before = copy.deepcopy(previous)
    after = copy.deepcopy(updated)
    assert validate_kimi_continuation_append(previous, updated, pod_id="fourth") == 5937
    assert previous == before and updated == after
    add_successor(updated, closed=True)
    previous = copy.deepcopy(updated)
    assert add_grant(updated) == 5877
    assert validate_kimi_continuation_append(previous, updated, pod_id="fifth") == 5877


@pytest.mark.parametrize(
    "defect",
    ["old_grant", "drop_grant", "extra_grant", "history", "ceiling", "original_grant", "metadata"],
)
def test_append_guard_rejects_any_other_ledger_rewrite(defect):
    previous = continuation_ledger()
    add_successor(previous, closed=True)
    updated = copy.deepcopy(previous)
    add_grant(updated)
    if defect == "old_grant":
        updated["kimi_continuation_grants"][0]["diagnosis"] = "rewritten"
    elif defect == "drop_grant":
        updated["kimi_continuation_grants"].pop(0)
    elif defect == "extra_grant":
        updated["kimi_continuation_grants"].append(
            copy.deepcopy(updated["kimi_continuation_grants"][-1])
        )
    elif defect == "history":
        updated["allocations"][0]["gpu_hours_upper_bound"] += 0.01
    elif defect == "ceiling":
        updated["max_gpu_hours"] += 0.01
    elif defect == "original_grant":
        updated["kimi_authorization"]["user_request"] = "rewritten"
    else:
        updated["new_unrelated_metadata"] = "not part of appending a grant"
    with pytest.raises(ValueError, match="append"):
        validate_kimi_continuation_append(previous, updated, pod_id="fifth")
