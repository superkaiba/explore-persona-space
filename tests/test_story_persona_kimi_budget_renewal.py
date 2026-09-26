"""A diagnosed budget renewal is one bounded successor, retaining all paid history."""

import copy
import datetime as dt

import pytest

from scripts.story_persona_storage_contract import (
    KIMI_FINANCIAL_FIELDS,
    allocation_seconds,
    contract_from_pod,
    kimi_ledger_digest,
    validate_kimi_budget_renewal_append,
)
from tests.test_story_persona_kimi_recovery import replacement_ledger

SOURCE = "a" * 40


def closed_ledger():
    """Retain the actual four closed allocations and their original grants."""
    state = replacement_ledger()
    state["allocations"].append(
        dict(
            pod_id="0275gj7zll7n2n",
            gpu_count=8,
            paid_start_unix=1790393106.876,
            deadline_unix=1790399631.876,
            termination_confirmed_at_unix=1790393788.892679,
            gpu_hours_upper_bound=1.5155926201078627,
        )
    )
    state["kimi_continuation_grants"] = [
        dict(
            approved=True,
            user_request="just continue until it succeeds",
            max_new_allocations=1,
            gpu_count=8,
            max_allocation_seconds=5850,
            max_cumulative_gpu_hours=42.65690570619371,
            diagnosis="Historical provider startup failed",
            failure_evidence="test:prior-receipt",
            prior_allocations=copy.deepcopy(state["allocations"]),
        )
    ]
    state["allocations"].append(
        dict(
            pod_id="otr2024prpu8h5",
            gpu_count=8,
            paid_start_unix=1790432398.897,
            deadline_unix=1790438248.897,
            termination_confirmed_at_unix=1790434889.6456392,
            gpu_hours_upper_bound=5.534996975792779,
        )
    )
    return state


def append_grant(state, source=SOURCE):
    """Append a test-only renewal; never read or write the operational ledger."""
    old = state.get("kimi_budget_renewals", [])
    spent = sum(entry["gpu_hours_upper_bound"] for entry in state["allocations"])
    grant = dict(
        version=1,
        approved=True,
        user_request="just continue until it succeeds",
        max_new_allocations=1,
        gpu_count=8,
        max_allocation_seconds=7200,
        max_additional_gpu_hours=16,
        source_sha=source,
        previous_effective_max_gpu_hours=(
            old[-1]["effective_max_gpu_hours"] if old else state["max_gpu_hours"]
        ),
        effective_max_gpu_hours=spent + 16,
        prior_ledger_sha256=kimi_ledger_digest(state),
        prior_allocations=[
            {key: entry[key] for key in KIMI_FINANCIAL_FIELDS} for entry in state["allocations"]
        ],
        diagnosis="Test fixture: bootstrap import failed before model start",
        failure_evidence="test:verified-preserved-failure-and-closure",
    )
    state.setdefault("kimi_budget_renewals", []).append(grant)
    return grant


def renewed_ledger():
    """Return a pending first renewal using recorded historical financial values."""
    state = closed_ledger()
    append_grant(state)
    return state


def check(state, pod_id="fifth", seconds=7200, count=8):
    """Exercise the real public allocation dispatch with no mocked validation."""
    return allocation_seconds(state, pod_id=pod_id, gpu_count=count, requested_seconds=seconds)


def add_successor(state, pod_id="fifth", closed=False):
    """Create a synthetic future provider row for state-transition tests only."""
    start = state["allocations"][-1]["termination_confirmed_at_unix"] + 10
    row = dict(
        pod_id=pod_id,
        gpu_count=8,
        source_sha=state["kimi_budget_renewals"][-1]["source_sha"],
        paid_start_unix=start,
        deadline_unix=start + 7200,
    )
    if closed:
        row.update(termination_confirmed_at_unix=start + 600, gpu_hours_upper_bound=4800 / 3600)
    state["allocations"].append(row)
    return row


def test_concrete_two_hour_renewal_keeps_history_and_does_not_stack_unused_budget():
    previous = closed_ledger()
    state = copy.deepcopy(previous)
    grant = append_grant(state)
    before = copy.deepcopy(state)
    assert grant["effective_max_gpu_hours"] == 51.17900508297814
    assert state["max_gpu_hours"] == 42.65690570619371
    assert validate_kimi_budget_renewal_append(previous, state, pod_id="fifth") == 7200
    assert check(state) == 7200
    assert state == before
    for key, value in previous.items():
        assert state[key] == value
    with pytest.raises(ValueError):
        check(previous)
    grant["effective_max_gpu_hours"] = previous["max_gpu_hours"] + 16
    with pytest.raises(ValueError, match="cap changed"):
        check(state)


def test_registered_successor_cannot_change_deadline_be_reused_or_allow_another_node():
    state = renewed_ledger()
    row = add_successor(state)
    assert check(state) == 7200
    with pytest.raises(ValueError, match="deadline"):
        check(state, seconds=7199)
    with pytest.raises(ValueError, match="already been used"):
        check(state, pod_id="sixth")
    row.update(
        termination_confirmed_at_unix=row["paid_start_unix"] + 100, gpu_hours_upper_bound=800 / 3600
    )
    with pytest.raises(ValueError, match="terminated"):
        check(state)
    with pytest.raises(ValueError, match="already been used"):
        check(state, pod_id="sixth")


def test_following_diagnosed_renewal_requires_closure_and_preserves_every_old_grant():
    state = renewed_ledger()
    add_successor(state, closed=True)
    previous = copy.deepcopy(state)
    append_grant(state, source="b" * 40)
    assert validate_kimi_budget_renewal_append(previous, state, pod_id="sixth") == 7200
    assert check(state, pod_id="sixth") == 7200
    assert state["kimi_budget_renewals"][:-1] == previous["kimi_budget_renewals"]
    assert state["kimi_budget_renewals"][-1]["effective_max_gpu_hours"] == pytest.approx(
        52.51233841631147
    )
    with pytest.raises(ValueError, match="already been used"):
        check(state)
    state["allocations"][-1]["termination_confirmed_at_unix"] = None
    with pytest.raises(ValueError):
        check(state, pod_id="sixth")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("approved", False),
        ("user_request", "continue"),
        ("max_new_allocations", 2),
        ("max_new_allocations", True),
        ("gpu_count", 4),
        ("max_allocation_seconds", 7201),
        ("max_allocation_seconds", 7200.0),
        ("max_additional_gpu_hours", 17),
        ("source_sha", "main"),
        ("previous_effective_max_gpu_hours", 45),
        ("effective_max_gpu_hours", 55),
        ("prior_ledger_sha256", "b" * 64),
        ("prior_allocations", []),
        ("diagnosis", " "),
        ("failure_evidence", ""),
    ],
)
def test_grant_fields_are_required_and_cannot_expand(field, value):
    state = renewed_ledger()
    grant = state["kimi_budget_renewals"][0]
    grant[field] = value
    with pytest.raises(ValueError):
        check(state)
    del grant[field]
    with pytest.raises(ValueError):
        check(state)


@pytest.mark.parametrize("seconds", [0, 900, 7201, 12600, 7200.0, float("nan")])
def test_requested_duration_cannot_expand_or_remove_reserve(seconds):
    with pytest.raises(ValueError):
        check(renewed_ledger(), seconds=seconds)


@pytest.mark.parametrize("count", [1, 4, 16, 8.0, True])
def test_exact_gpu_count_required(count):
    with pytest.raises(ValueError):
        check(renewed_ledger(), count=count)


@pytest.mark.parametrize("field", KIMI_FINANCIAL_FIELDS)
@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_every_closed_financial_field_remains_bound(index, field):
    state = renewed_ledger()
    row = state["allocations"][index]
    row[field] = "changed" if field == "pod_id" else row[field] + 1
    with pytest.raises(ValueError):
        check(state)


@pytest.mark.parametrize(
    "key",
    [
        "kimi_authorization",
        "kimi_recovery_authorization",
        "kimi_replacement_authorization",
        "kimi_continuation_grants",
    ],
)
def test_prior_authorizations_cannot_be_removed_or_rewritten(key):
    state = renewed_ledger()
    del state[key]
    with pytest.raises(ValueError):
        check(state)
    state = renewed_ledger()
    if key == "kimi_continuation_grants":
        state[key][0]["max_allocation_seconds"] += 1
    else:
        state[key]["approved"] = False
    with pytest.raises(ValueError):
        check(state)


@pytest.mark.parametrize(
    "defect", ["unresolved", "understated", "negative", "nan", "overlap", "duplicate", "missing"]
)
def test_invalid_closed_history_is_rejected_even_before_new_snapshot_hash(defect):
    state = closed_ledger()
    row = state["allocations"][-1]
    if defect == "unresolved":
        row["termination_confirmed_at_unix"] = None
    elif defect == "understated":
        row["gpu_hours_upper_bound"] -= 0.1
    elif defect == "negative":
        row["termination_confirmed_at_unix"] = row["paid_start_unix"] - 1
    elif defect == "nan":
        row["deadline_unix"] = float("nan")
    elif defect == "overlap":
        row["paid_start_unix"] = state["allocations"][-2]["termination_confirmed_at_unix"] - 1
        row["deadline_unix"] = row["paid_start_unix"] + 5850
    elif defect == "duplicate":
        row["pod_id"] = state["allocations"][0]["pod_id"]
    else:
        state["allocations"].pop()
    with pytest.raises(ValueError):
        append_grant(state)
        check(state)


@pytest.mark.parametrize("defect", ["old_grant", "history", "ceiling", "metadata", "drop", "extra"])
def test_append_guard_permits_only_one_new_grant(defect):
    previous = renewed_ledger()
    add_successor(previous, closed=True)
    updated = copy.deepcopy(previous)
    append_grant(updated)
    if defect == "old_grant":
        updated["kimi_budget_renewals"][0]["diagnosis"] = "rewritten"
    elif defect == "history":
        updated["allocations"][0]["gpu_hours_upper_bound"] += 1
    elif defect == "ceiling":
        updated["max_gpu_hours"] += 1
    elif defect == "metadata":
        updated["new_field"] = "unrelated rewrite"
    elif defect == "drop":
        updated["kimi_budget_renewals"].pop(0)
    else:
        updated["kimi_budget_renewals"].append(copy.deepcopy(updated["kimi_budget_renewals"][-1]))
    with pytest.raises(ValueError, match="append"):
        validate_kimi_budget_renewal_append(previous, updated, pod_id="sixth")


@pytest.mark.parametrize("renewals", [None, {}, "yes", [None]])
def test_malformed_renewal_list_fails_closed(renewals):
    state = renewed_ledger()
    state["kimi_budget_renewals"] = renewals
    with pytest.raises(ValueError):
        check(state)


def test_hash_binds_nonfinancial_history_and_normalizes_only_missing_empty_list():
    previous = closed_ledger()
    assert kimi_ledger_digest(previous) == kimi_ledger_digest(
        {**previous, "kimi_budget_renewals": []}
    )
    state = renewed_ledger()
    state["allocations"][0]["source_updates"] = [{"source_sha": "d" * 40}]
    with pytest.raises(ValueError, match="digest"):
        check(state)


def test_cannot_pregrant_or_use_old_closed_id_or_change_existing_source():
    state = renewed_ledger()
    for row in state["allocations"]:
        with pytest.raises(ValueError, match="already been used"):
            check(state, pod_id=row["pod_id"])
    second = copy.deepcopy(state["kimi_budget_renewals"][0])
    state["kimi_budget_renewals"].append(second)
    with pytest.raises(ValueError):
        check(state, pod_id="sixth")
    state = renewed_ledger()
    row = add_successor(state)
    row["source_sha"] = "b" * 40
    with pytest.raises(ValueError, match="source"):
        check(state)


def test_provider_contract_source_overlap_expiry_and_exact_original_paid_start():
    state = renewed_ledger()
    start = 1790435000
    pod = dict(
        id="fifth",
        name="pod-2673-crossmodel-kimi",
        desiredStatus="RUNNING",
        gpuCount=8,
        machine={"gpuTypeId": "NVIDIA H200"},
        volumeInGb=1000,
        containerDiskInGb=50,
        volumeMountPath="/workspace",
        createdAt=dt.datetime.fromtimestamp(start, dt.UTC).isoformat(),
    )
    args = dict(
        pod_id="fifth",
        expected_name=pod["name"],
        model_key="kimi",
        model={
            "id": "moonshotai/Kimi-K2.6",
            "revision": "b" * 40,
            "execution_mode": "unpadded_singleton",
        },
        source_sha=SOURCE,
        now=start + 10,
        requested_seconds=7200,
        ledger=state,
    )
    result = contract_from_pod(pod, **args)
    assert result["paid_start_unix"] == start and result["deadline_unix"] == start + 7200
    with pytest.raises(ValueError, match="source differs"):
        contract_from_pod(pod, **dict(args, source_sha="b" * 40))
    with pytest.raises(RuntimeError, match="insufficient remaining"):
        contract_from_pod(pod, **dict(args, now=start + 6300))
    pod["createdAt"] = dt.datetime.fromtimestamp(
        state["allocations"][-1]["termination_confirmed_at_unix"] - 1, dt.UTC
    ).isoformat()
    with pytest.raises(ValueError, match="before prior termination"):
        contract_from_pod(pod, **args)
    pod["createdAt"] = dt.datetime.fromtimestamp(start, dt.UTC).isoformat()
    state["allocations"].append(
        dict(
            pod_id="fifth",
            gpu_count=8,
            source_sha=SOURCE,
            paid_start_unix=start,
            deadline_unix=start + 7200,
        )
    )
    assert contract_from_pod(pod, **args)["deadline_unix"] == start + 7200
    pod["createdAt"] = dt.datetime.fromtimestamp(start + 1, dt.UTC).isoformat()
    with pytest.raises(ValueError, match="reset the recorded"):
        contract_from_pod(pod, **args)
