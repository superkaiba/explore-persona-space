#!/usr/bin/env python3
"""Read-only RunPod verification, plus explicit optional upload of its JSON contract.

No provisioning/lifecycle mutation. Credentials stay inside the project client.
The API storage size is a provision contract, not evidence of writable free bytes;
the workload must still check the mounted filesystem, usage/quota, and canary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

QUERY = """query PodStorage($id: String!) {
  pod(input: {podId: $id}) {
    id name desiredStatus gpuCount createdAt
    volumeInGb containerDiskInGb volumeMountPath
    machine { gpuTypeId }
  }
}"""
REMOTE_PATH = "/workspace/issue2673_storage_contract.json"
KIMI_FINANCIAL_FIELDS = (
    "pod_id",
    "gpu_count",
    "paid_start_unix",
    "deadline_unix",
    "termination_confirmed_at_unix",
    "gpu_hours_upper_bound",
)


def kimi_ledger_digest(ledger):
    """Hash canonical ledger JSON, normalizing an absent renewal list to empty.

    This digest is reproducible from historical prefixes. Preserve the original
    ledger file and its raw-byte hash separately in the operator's audit receipt.
    """
    state = {**ledger, "kimi_budget_renewals": ledger.get("kimi_budget_renewals", [])}
    raw = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _kimi_closed_spend(allocations):
    """Validate ordered, resolved provider accounting and return conservative GPU-hours."""
    spent = 0.0
    last_end = 0.0
    for entry in allocations:
        if any(key not in entry for key in KIMI_FINANCIAL_FIELDS):
            raise ValueError("Kimi renewal requires complete resolved prior accounting")
        start, deadline, end, recorded = (
            entry[key]
            for key in (
                "paid_start_unix",
                "deadline_unix",
                "termination_confirmed_at_unix",
                "gpu_hours_upper_bound",
            )
        )
        if (
            type(entry["gpu_count"]) is not int
            or entry["gpu_count"] != 8
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (start, deadline, end, recorded)
            )
            or not 0 < start < deadline
            or end < start
            or start < last_end
        ):
            raise ValueError("Kimi renewal prior accounting is unresolved, invalid or overlapping")
        actual = (end - start) * 8 / 3600
        if recorded < 0 or recorded < actual - 1e-8:
            raise ValueError("Kimi renewal ledger understates paid time")
        spent += max(actual, recorded)
        last_end = end
    return spent


def kimi_budget_renewal_seconds(ledger, *, pod_id, gpu_count, requested_seconds):
    """Validate one successor per diagnosed renewal without rewriting any old cap."""
    renewals = ledger["kimi_budget_renewals"]
    continuation = ledger.get("kimi_continuation_grants")
    allocations = ledger["allocations"]
    if (
        ledger.get("model_key") != "kimi"
        or not isinstance(continuation, list)
        or not continuation
        or type(gpu_count) is not int
        or gpu_count != 8
        or type(requested_seconds) is not int
        or not 900 < requested_seconds <= 7200
        or any(
            not isinstance(entry.get("pod_id"), str) or not entry["pod_id"] for entry in allocations
        )
        or len({entry["pod_id"] for entry in allocations}) != len(allocations)
    ):
        raise ValueError("Kimi budget renewal requires bounded singletons and unique paid history")
    base_count = 3 + len(continuation)
    if len(allocations) not in (base_count + len(renewals) - 1, base_count + len(renewals)):
        raise ValueError("Kimi budget renewal permits exactly one successor per grant")

    # Replay the last historical grant against its actual pre-allocation prefix.
    # No stored row is changed, including the closed last allocation's envelope.
    legacy = {key: value for key, value in ledger.items() if key != "kimi_budget_renewals"}
    legacy["allocations"] = allocations[: base_count - 1]
    final_legacy = allocations[base_count - 1]
    legacy_seconds = final_legacy["deadline_unix"] - final_legacy["paid_start_unix"]
    if not math.isfinite(legacy_seconds) or not float(legacy_seconds).is_integer():
        raise ValueError("Kimi historical allocation duration must retain whole seconds")
    allocation_seconds(
        legacy,
        pod_id=final_legacy["pod_id"],
        gpu_count=8,
        requested_seconds=int(legacy_seconds),
    )

    previous_limit = ledger["max_gpu_hours"]
    for index, grant in enumerate(renewals):
        count = base_count + index
        prior = allocations[:count]
        spent = _kimi_closed_spend(prior)
        prior_state = {**ledger, "allocations": prior, "kimi_budget_renewals": renewals[:index]}
        if (
            not isinstance(grant, dict)
            or grant.get("version") != 1
            or grant.get("approved") is not True
            or grant.get("user_request") != "just continue until it succeeds"
            or type(grant.get("max_new_allocations")) is not int
            or grant["max_new_allocations"] != 1
            or type(grant.get("gpu_count")) is not int
            or grant["gpu_count"] != 8
            or type(grant.get("max_allocation_seconds")) is not int
            or grant["max_allocation_seconds"] != 7200
            or grant.get("max_additional_gpu_hours") != 16
            or grant.get("previous_effective_max_gpu_hours") != previous_limit
            or grant.get("effective_max_gpu_hours") != spent + 16
            or grant.get("prior_ledger_sha256") != kimi_ledger_digest(prior_state)
            or not isinstance(grant.get("source_sha"), str)
            or re.fullmatch(r"[0-9a-f]{40}", grant["source_sha"]) is None
            or not isinstance(grant.get("diagnosis"), str)
            or not grant["diagnosis"].strip()
            or not isinstance(grant.get("failure_evidence"), str)
            or not grant["failure_evidence"].strip()
        ):
            raise ValueError(
                "Kimi budget renewal grant, source, prior-ledger digest or cap changed"
            )
        snapshots = [{key: entry[key] for key in KIMI_FINANCIAL_FIELDS} for entry in prior]
        if grant.get("prior_allocations") != snapshots:
            raise ValueError("Kimi budget renewal must preserve exact ordered financial snapshots")
        if len(allocations) > count:
            successor = allocations[count]
            start, deadline = successor["paid_start_unix"], successor["deadline_unix"]
            if (
                type(successor["gpu_count"]) is not int
                or successor["gpu_count"] != 8
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in (start, deadline)
                )
                or not 900 < deadline - start <= 7200
                or start < prior[-1]["termination_confirmed_at_unix"]
                or successor.get("source_sha") != grant["source_sha"]
            ):
                raise ValueError(
                    "Kimi budget renewal successor changed its source or paid envelope"
                )
        previous_limit = grant["effective_max_gpu_hours"]

    prior_count = base_count + len(renewals) - 1
    if pod_id in {entry["pod_id"] for entry in allocations[:prior_count]} or (
        len(allocations) > prior_count and allocations[-1]["pod_id"] != pod_id
    ):
        raise ValueError("the single renewed Kimi allocation has already been used")
    if len(allocations) > prior_count:
        current = allocations[-1]
        if current.get("termination_confirmed_at_unix") is not None:
            raise ValueError("cannot reuse a terminated allocation")
        if current["deadline_unix"] - current["paid_start_unix"] != requested_seconds:
            raise ValueError("refusing to change an existing allocation deadline")
    if spent + gpu_count * requested_seconds / 3600 > previous_limit:
        raise ValueError("requested allocation exceeds the renewed cumulative GPU-hours")
    return requested_seconds


def validate_kimi_budget_renewal_append(previous, updated, *, pod_id):
    """Validate exactly one renewal append under the ledger writer's existing lock."""
    key = "kimi_budget_renewals"
    old_grants, new_grants = previous.get(key, []), updated.get(key)
    if (
        not isinstance(old_grants, list)
        or not isinstance(new_grants, list)
        or len(new_grants) != len(old_grants) + 1
        or new_grants[:-1] != old_grants
        or {k: v for k, v in previous.items() if k != key}
        != {k: v for k, v in updated.items() if k != key}
    ):
        raise ValueError("Kimi budget renewal must append one grant without rewriting history")
    return allocation_seconds(updated, pod_id=pod_id, gpu_count=8, requested_seconds=7200)


def kimi_continuation_seconds(ledger, *, pod_id):
    """Validate successive closed-history grants and return the latest duration cap.

    Each grant consumes one allocation slot after the original three grants.
    This is a pure validator: append-only persistence and provider termination
    evidence remain responsibilities of the locked, audited ledger writer.
    """
    ceiling = 42.65690570619371
    fields = {
        "pod_id",
        "gpu_count",
        "paid_start_unix",
        "deadline_unix",
        "termination_confirmed_at_unix",
        "gpu_hours_upper_bound",
    }
    allocations = ledger["allocations"]
    grants = ledger["kimi_continuation_grants"]
    if ledger["max_gpu_hours"] != ceiling:
        raise ValueError("Kimi continuation cannot change the cumulative ceiling")
    if len(allocations) not in (len(grants) + 2, len(grants) + 3):
        raise ValueError("Kimi continuation needs exactly one grant per successor allocation")
    for index, grant in enumerate(grants):
        if (
            not isinstance(grant, dict)
            or grant.get("approved") is not True
            or grant.get("user_request") != "just continue until it succeeds"
            or grant.get("max_new_allocations") != 1
            or grant.get("gpu_count") != 8
            or grant.get("max_cumulative_gpu_hours") != ceiling
            or type(grant.get("max_allocation_seconds")) is not int
            or not 900 < grant["max_allocation_seconds"] <= 6525
            or not isinstance(grant.get("diagnosis"), str)
            or not grant["diagnosis"].strip()
            or not isinstance(grant.get("failure_evidence"), str)
            or not grant["failure_evidence"].strip()
        ):
            raise ValueError("Kimi continuation requires a bounded diagnosed standing grant")
        snapshots = grant.get("prior_allocations")
        prior_count = 3 + index
        if not isinstance(snapshots, list) or len(snapshots) != prior_count:
            raise ValueError("Kimi continuation must snapshot every prior allocation in order")
        spent = 0.0
        for entry, snapshot in zip(allocations[:prior_count], snapshots, strict=True):
            if (
                not isinstance(snapshot, dict)
                or set(snapshot) != fields
                or any(key not in entry or entry[key] != snapshot[key] for key in fields)
            ):
                raise ValueError("Kimi continuation prior allocation snapshot changed")
            values = [snapshot[key] for key in fields - {"pod_id", "gpu_count"}]
            if (
                not isinstance(snapshot["pod_id"], str)
                or not snapshot["pod_id"]
                or snapshot["gpu_count"] != 8
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                )
                or snapshot["paid_start_unix"] <= 0
                or snapshot["deadline_unix"] <= snapshot["paid_start_unix"]
                or snapshot["termination_confirmed_at_unix"] < snapshot["paid_start_unix"]
            ):
                raise ValueError("Kimi continuation requires resolved finite prior accounting")
            actual = (
                (snapshot["termination_confirmed_at_unix"] - snapshot["paid_start_unix"]) * 8 / 3600
            )
            if snapshot["gpu_hours_upper_bound"] < actual - 1e-8:
                raise ValueError("Kimi continuation ledger understates paid time")
            spent += max(actual, snapshot["gpu_hours_upper_bound"])
        third = snapshots[2]
        if not 900 < third["deadline_unix"] - third["paid_start_unix"] <= 6525:
            raise ValueError("Kimi continuation changed the prior replacement duration")
        remaining_seconds = math.floor((ceiling - spent) * 3600 / 8)
        if grant["max_allocation_seconds"] > remaining_seconds:
            raise ValueError("Kimi continuation grant exceeds remaining cumulative GPU-hours")
        if len(allocations) > prior_count:
            successor = allocations[prior_count]
            duration = successor["deadline_unix"] - successor["paid_start_unix"]
            if (
                not math.isfinite(duration)
                or not 900 < duration <= grant["max_allocation_seconds"]
                or successor["paid_start_unix"]
                < max(entry["termination_confirmed_at_unix"] for entry in snapshots)
            ):
                raise ValueError("Kimi continuation successor changed its granted paid envelope")
    last_prior_count = len(grants) + 2
    prior_ids = {entry["pod_id"] for entry in allocations[:last_prior_count]}
    if pod_id in prior_ids or (
        len(allocations) > last_prior_count and allocations[-1]["pod_id"] != pod_id
    ):
        raise ValueError("the single continuation Kimi allocation has already been used")
    return grants[-1]["max_allocation_seconds"]


def allocation_seconds(ledger, *, pod_id, gpu_count, requested_seconds):
    """Enforce the cumulative allowance without resetting any prior paid time."""
    renewals = ledger.get("kimi_budget_renewals", [])
    if not isinstance(renewals, list):
        raise ValueError("Kimi budget renewals must be an ordered list")
    if renewals:
        return kimi_budget_renewal_seconds(
            ledger, pod_id=pod_id, gpu_count=gpu_count, requested_seconds=requested_seconds
        )
    limit = ledger["max_gpu_hours"]
    if not math.isfinite(limit) or not 0 < limit <= 45:
        raise ValueError("invalid cumulative GPU-hour allowance")
    maximum_seconds = 7200
    additional_gpu_hours = None
    if ledger.get("model_key") == "kimi":
        authority = ledger.get("kimi_authorization", {})
        known = {entry["pod_id"] for entry in ledger["allocations"]}
        if (
            authority.get("approved") is not True
            or authority.get("max_new_allocations") != 1
            or authority.get("max_allocation_seconds") != 12600
            or authority.get("user_request") != "Do both. Run it now"
            or gpu_count != 8
        ):
            raise ValueError("Kimi requires its own single-allocation authorization")
        recovery = ledger.get("kimi_recovery_authorization")
        replacement = ledger.get("kimi_replacement_authorization")
        continuation = ledger.get("kimi_continuation_grants", [])
        if not isinstance(continuation, list) or (continuation and replacement is None):
            raise ValueError("Kimi continuation must preserve all prior authorizations")
        if replacement is not None and recovery is None:
            raise ValueError("Kimi replacement must preserve its prior recovery authorization")
        if recovery is None:
            if limit != 28 or known - {pod_id}:
                raise ValueError("Kimi requires its own single-allocation authorization")
            maximum_seconds = 12600
        else:
            # Thomas renewed exactly this closed run on 2026-09-26. Binding the
            # prior ID prevents rebasing the same grant onto another failed pod.
            prior = ["2y9tr3io3gd0os"]
            if (
                not isinstance(recovery, dict)
                or recovery.get("approved") is not True
                or recovery.get("user_request") != "approve and continue until done"
                or recovery.get("prior_pod_ids") != prior
                or recovery.get("max_new_allocations") != 1
                or recovery.get("max_allocation_seconds") != 7200
                or recovery.get("max_additional_gpu_hours") != 16
                or not set(prior) <= known
                or len(known) != len(ledger["allocations"])
            ):
                raise ValueError("Kimi recovery requires the explicit bounded user authorization")
            previous = next(entry for entry in ledger["allocations"] if entry["pod_id"] in prior)
            if (
                previous["gpu_count"] != 8
                or previous["deadline_unix"] - previous["paid_start_unix"] != 12600
            ):
                raise ValueError("original Kimi allocation accounting changed")
            if any(entry["gpu_count"] != 8 for entry in ledger["allocations"]):
                raise ValueError("Kimi recovery accounting requires eight GPUs per allocation")
            if replacement is None:
                if pod_id in prior or (known - set(prior)) - {pod_id}:
                    raise ValueError("the single additional Kimi allocation has already been used")
                additional_gpu_hours = 16
            else:
                # The third grant replaces the SSH-less second allocation. It
                # spends the remaining existing ceiling, without renewing it.
                prior = ["2y9tr3io3gd0os", "pnk2t3dv5zpff2"]
                if (
                    not isinstance(replacement, dict)
                    or replacement.get("approved") is not True
                    or replacement.get("user_request") != "approve more allocation"
                    or replacement.get("continuation_request") != "just continue until it succeeds"
                    or replacement.get("prior_pod_ids") != prior
                    or replacement.get("max_new_allocations") != 1
                    or replacement.get("max_allocation_seconds") != 6525
                    or replacement.get("max_gpu_hours") != 14.5
                    or replacement.get("max_cumulative_gpu_hours") != 42.65690570619371
                    or limit > 42.65690570619371
                    or not set(prior) <= known
                ):
                    raise ValueError(
                        "Kimi replacement requires the explicit bounded user authorization"
                    )
                if not continuation and (pod_id in prior or (known - set(prior)) - {pod_id}):
                    raise ValueError("the single replacement Kimi allocation has already been used")
                paid_envelopes = {
                    "2y9tr3io3gd0os": (1790371940.801, 1790384540.801),
                    "pnk2t3dv5zpff2": (1790391148.825, 1790398348.825),
                }
                for entry in ledger["allocations"]:
                    if (
                        entry["pod_id"] in paid_envelopes
                        and (entry["paid_start_unix"], entry["deadline_unix"])
                        != paid_envelopes[entry["pod_id"]]
                    ):
                        raise ValueError("prior Kimi provider paid start/deadline changed")
                maximum_seconds = 6525
                if continuation:
                    if type(requested_seconds) is not int:
                        raise ValueError("Kimi continuation duration must be whole seconds")
                    maximum_seconds = kimi_continuation_seconds(ledger, pod_id=pod_id)
    elif limit > 17:
        # Thomas approved one additional singleton attempt on 2026-09-21.
        authority = ledger.get("singleton_authorization", {})
        prior = authority.get("prior_pod_ids", [])
        known = {entry["pod_id"] for entry in ledger["allocations"]}
        if (
            authority.get("approved") is not True
            or authority.get("max_new_allocations") != 1
            or authority.get("max_allocation_seconds") != 12600
            or authority.get("max_gpu_hours") != 45
            or not prior
            or len(prior) != len(set(prior))
            or not set(prior) <= known
            or gpu_count != 8
        ):
            raise ValueError("renewed allowance requires the approved singleton authorization")
        if pod_id in prior or (known - set(prior)) - {pod_id}:
            raise ValueError("the single additional allocation has already been used")
        maximum_seconds = 12600
    if not 900 < requested_seconds <= maximum_seconds:
        raise ValueError(
            "allocation must leave preservation time and stay within its approved limit"
        )
    spent = 0.0
    for entry in ledger["allocations"]:
        if entry["pod_id"] == pod_id:
            if entry.get("termination_confirmed_at_unix") is not None:
                raise ValueError("cannot reuse a terminated allocation")
            if entry["deadline_unix"] - entry["paid_start_unix"] != requested_seconds:
                raise ValueError("refusing to change an existing allocation deadline")
            continue
        end = entry.get("termination_confirmed_at_unix")
        if end is None:
            raise ValueError("another allocation is still unresolved")
        actual = (end - entry["paid_start_unix"]) * entry["gpu_count"] / 3600
        recorded = entry["gpu_hours_upper_bound"]
        if (
            not all(math.isfinite(x) for x in (actual, recorded))
            or actual < 0
            or recorded < actual - 1e-8
        ):
            raise ValueError("allocation ledger understates paid time")
        spent += max(actual, recorded)
    if spent + gpu_count * requested_seconds / 3600 > limit:
        raise ValueError("requested allocation exceeds remaining cumulative GPU-hours")
    if additional_gpu_hours is not None and limit > spent + additional_gpu_hours + 1e-8:
        raise ValueError("cumulative Kimi allowance exceeds the approved additional GPU-hours")
    return requested_seconds


def validate_kimi_continuation_append(previous, updated, *, pod_id):
    """Validate one immutable-history grant append before an atomic locked write."""
    key = "kimi_continuation_grants"
    old_grants = previous.get(key, [])
    new_grants = updated.get(key)
    if (
        previous.get("model_key") != "kimi"
        or not isinstance(old_grants, list)
        or not isinstance(new_grants, list)
        or len(new_grants) != len(old_grants) + 1
        or new_grants[:-1] != old_grants
        or {k: v for k, v in previous.items() if k != key}
        != {k: v for k, v in updated.items() if k != key}
        or not isinstance(new_grants[-1], dict)
    ):
        raise ValueError("Kimi continuation update must append one grant without rewriting history")
    return allocation_seconds(
        updated,
        pod_id=pod_id,
        gpu_count=8,
        requested_seconds=new_grants[-1].get("max_allocation_seconds"),
    )


def contract_from_pod(
    pod,
    *,
    pod_id,
    expected_name,
    model_key,
    model,
    source_sha,
    now,
    requested_seconds=None,
    ledger=None,
):
    """Bind a live pod to its approved resources and immutable paid start/deadline."""
    if not pod or pod["id"] != pod_id or pod["name"] != expected_name:
        raise RuntimeError("actual live pod ID/name did not match the operator request")
    if re.fullmatch(r"pod-2673(?:-[a-z][a-z0-9-]{0,19})?", expected_name) is None:
        raise RuntimeError("this helper is restricted to task2673 managed pod names")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise RuntimeError("source_sha must be a complete immutable Git SHA")
    if pod["desiredStatus"] != "RUNNING":
        raise RuntimeError("storage contract requires an actual RUNNING pod")
    if model_key == "kimi" and (
        ledger is None or ledger.get("model_key") != "kimi" or requested_seconds is None
    ):
        raise ValueError("Kimi requires a separate allocation ledger")
    if (
        model_key == "kimi"
        and ledger.get("kimi_recovery_authorization") is not None
        and model.get("execution_mode") != "unpadded_singleton"
    ):
        raise ValueError("Kimi recovery is restricted to singleton execution")
    if ledger is not None and ledger["max_gpu_hours"] > 17 and model_key != "kimi":
        if model_key != "deepseek" or model.get("execution_mode") != "unpadded_singleton":
            raise ValueError("renewed allowance is restricted to DeepSeek singleton execution")
    count, gpu, volume, seconds = (
        (8, "H200", 1000, 7200) if model_key in {"deepseek", "kimi"} else (1, "H100", 200, 3600)
    )
    if requested_seconds is not None:
        if ledger is None or requested_seconds > (
            12600 if model_key in {"deepseek", "kimi"} else 3600
        ):
            raise ValueError(
                "an explicit allocation requires the cumulative ledger and approved limit"
            )
        seconds = allocation_seconds(
            ledger, pod_id=pod_id, gpu_count=count, requested_seconds=requested_seconds
        )
    if model_key not in {"qwen", "deepseek", "kimi"}:
        raise RuntimeError("unsupported model arm")
    if pod["gpuCount"] != count or gpu not in pod["machine"]["gpuTypeId"]:
        raise RuntimeError("live GPU count/type does not match the reviewed arm")
    actual_volume = float(pod["volumeInGb"])
    if not math.isfinite(actual_volume) or actual_volume < volume:
        raise RuntimeError("live API storage specification is insufficient")
    if pod["volumeMountPath"] != "/workspace":
        raise RuntimeError("live volume is not mounted at the required /workspace path")
    created = dt.datetime.fromisoformat(pod["createdAt"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise RuntimeError("provider creation timestamp has no timezone")
    paid_start = created.timestamp()
    deadline = paid_start + seconds
    if model_key == "kimi":
        latest = None
        if ledger.get("kimi_budget_renewals"):
            latest = ledger["kimi_budget_renewals"][-1]
            if source_sha != latest["source_sha"]:
                raise ValueError("Kimi provider contract source differs from its budget renewal")
        elif ledger.get("kimi_continuation_grants"):
            latest = ledger["kimi_continuation_grants"][-1]
        if latest is not None:
            if paid_start < max(
                entry["termination_confirmed_at_unix"] for entry in latest["prior_allocations"]
            ):
                raise ValueError("Kimi continuation began before prior termination was confirmed")
        for entry in ledger["allocations"]:
            if entry["pod_id"] == pod_id and (
                entry["paid_start_unix"] != paid_start or entry["deadline_unix"] != deadline
            ):
                raise ValueError(
                    "provider identity would reset the recorded Kimi paid start/deadline"
                )
    if paid_start > now or deadline - now <= 900:
        raise RuntimeError("invalid creation timestamp or insufficient remaining time envelope")
    return {
        "schema": "issue2673-storage-contract-v1",
        "pod_id": pod_id,
        "pod_name": expected_name,
        "model_key": model_key,
        "model_id": model["id"],
        "model_revision": model["revision"],
        "source_sha": source_sha,
        "api_volume_gb": actual_volume,
        "api_container_disk_gb": pod["containerDiskInGb"],
        "api_volume_mount_path": pod["volumeMountPath"],
        "api_verified_at_unix": now,
        "provider_created_at": pod["createdAt"],
        "paid_start_unix": paid_start,
        "deadline_unix": deadline,
        "max_allocation_seconds": seconds,
        "api_gpu_count": pod["gpuCount"],
        "api_gpu_type": pod["machine"]["gpuTypeId"],
        "query": QUERY,
        "api_client": "scripts/runpod_api.py::graphql (personal=True)",
        "storage_scope": "API provision specification only; workload must verify actual mount, used bytes, quota-aware headroom and small write canary",
        "timing_scope": "Fresh provision only: count from provider createdAt, including setup/staging; never reset start to contract creation",
        "live_api_pod": pod,
    }


def upload_contract(client, pod_id, local_path, expected_hash, identity, sentinel):
    """Explicit opt-in, verified exact pod endpoint; atomic remote publication."""
    # Re-query exact live identity/status/ports through the personal transport.
    result = client.graphql(
        "query SSH($id: String!) { pod(input:{podId:$id}) { id desiredStatus runtime { ports { ip publicPort privatePort type isIpPublic } } } }",
        {"id": pod_id},
        personal=True,
    )
    live = result["pod"]
    if not live or live["id"] != pod_id or live["desiredStatus"] != "RUNNING":
        raise RuntimeError("pod changed state before optional upload")
    ports = [
        p
        for p in result["pod"]["runtime"]["ports"]
        if p["privatePort"] == 22 and p["type"] == "tcp" and p["isIpPublic"]
    ]
    if len(ports) != 1:
        raise RuntimeError("exact pod does not expose one unambiguous public SSH endpoint")
    port = int(ports[0]["publicPort"])
    if not 1 <= port <= 65535:
        raise RuntimeError("invalid SSH port")
    host = str(ipaddress.ip_address(ports[0]["ip"]))
    target = f"root@{host}"
    # The remote script writes a temporary file, verifies bytes, then atomically
    # exposes the expected path so the workload never reads partial JSON.
    code = """import hashlib,json,os,pathlib,sys,tempfile
blob=sys.stdin.buffer.read()
if hashlib.sha256(blob).hexdigest()!=sys.argv[1]: raise RuntimeError('contract digest mismatch')
c=json.loads(blob)
env=dict(x.split(b'=',1) for x in pathlib.Path('/proc/1/environ').read_bytes().split(bytes([0])) if b'=' in x)
if env.get(b'RUNPOD_POD_ID',b'').decode()!=c['pod_id']: raise RuntimeError('remote pod identity mismatch')
env_path=pathlib.Path('/workspace/explore-persona-space/.env')
lines=env_path.read_text().splitlines()
assignments={'RUNPOD_POD_ID':c['pod_id'],'EPS_SENTINEL_PATH':sys.argv[2],'EPM_HF_OVERFLOW_ROUTING':'1'}
for key,value in assignments.items():
    existing=[s.split('=',1)[1] for s in lines if s.startswith(key+'=')]
    if key!='EPM_HF_OVERFLOW_ROUTING' and existing and existing!=[value]: raise RuntimeError('runtime identity/completion env conflicts')
    lines=[s for s in lines if not s.startswith(key+'=')]
    lines.append(key+'='+value)
env_fd,env_tmp=tempfile.mkstemp(prefix='.issue2673-env-',dir=env_path.parent)
os.fchmod(env_fd,env_path.stat().st_mode & 0o777)
with os.fdopen(env_fd,'w') as f: f.write(chr(10).join(lines)+chr(10)); f.flush(); os.fsync(f.fileno())
os.replace(env_tmp,env_path)
path=pathlib.Path('/workspace/issue2673_storage_contract.json')
fd,tmp=tempfile.mkstemp(prefix='.issue2673-contract-',dir=path.parent)
with os.fdopen(fd,'wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
if hashlib.sha256(path.read_bytes()).hexdigest()!=sys.argv[1]: raise RuntimeError('remote verification failed')
print(json.dumps({'uploaded':True,'sha256':sys.argv[1],'pod_id':c['pod_id']}))
"""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", str(port)]
    if identity:
        command += ["-i", str(Path(identity).expanduser().resolve())]
    command += [
        target,
        "python3 -c "
        + shlex.quote(code)
        + " "
        + shlex.quote(expected_hash)
        + " "
        + shlex.quote(sentinel),
    ]
    result = subprocess.run(
        command, input=local_path.read_bytes(), capture_output=True, check=True, timeout=60
    )
    receipt = json.loads(result.stdout)
    if receipt != {"uploaded": True, "sha256": expected_hash, "pod_id": pod_id}:
        raise RuntimeError("unexpected remote upload receipt")
    return receipt


def preserve_first_allocation(path, contract, *, reviewed_source_update=False):
    """One durable record per arm; retries cannot silently start a fresh budget."""
    fields = (
        "pod_id",
        "model_key",
        "source_sha",
        "provider_created_at",
        "paid_start_unix",
        "deadline_unix",
    )
    first = {key: contract[key] for key in fields}
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[key] != first[key] for key in fields if key != "source_sha"):
            raise RuntimeError(
                "first-provision ledger differs: refusing a new pod/source/start or reset time envelope"
            )
        if old["source_sha"] != first["source_sha"]:
            if not reviewed_source_update:
                raise RuntimeError("source update needs explicit reviewed-source-update flag")
            history = path.with_name(path.stem + "-source-updates.jsonl")
            with history.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "checked_at": time.time(),
                            "original_source_sha": old["source_sha"],
                            "reviewed_source_sha": first["source_sha"],
                            "pod_id": first["pod_id"],
                            "original_deadline_unix": old["deadline_unix"],
                        }
                    )
                    + chr(10)
                )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(first, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    p.add_argument("--pod-id", required=True)
    p.add_argument("--expected-pod-name", required=True)
    p.add_argument("--model-key", choices=["qwen", "deepseek", "kimi"], required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--first-allocation-record",
        type=Path,
        required=True,
        help="Durable per-arm ledger; an existing record must match exactly, preventing budget resets",
    )
    p.add_argument(
        "--ssh-upload",
        action="store_true",
        help="Explicitly upload the verified contract atomically to this exact live pod",
    )
    p.add_argument(
        "--ssh-identity", help="Optional SSH identity file; normal host-key checks remain enabled"
    )
    p.add_argument(
        "--handle-file",
        type=Path,
        help="Exact dispatcher handle, mandatory for runtime env setup with ssh-upload",
    )
    p.add_argument(
        "--reviewed-source-update",
        action="store_true",
        help="Allow an explicitly reviewed source fix while preserving the original allocation record and deadline",
    )
    p.add_argument("--allocation-seconds", type=int, required=True)
    p.add_argument("--allocation-ledger", type=Path, required=True)
    args = p.parse_args()
    root = args.repo_root.resolve()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
        p.error("--source-sha must contain 40 lowercase hex characters")
    # Bind model metadata to the actual Git object, never the current dirty file.
    text = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            args.source_sha + ":configs/pilots/story_persona_crossmodel_capture.yaml",
        ],
        text=True,
    )
    from omegaconf import OmegaConf

    config = OmegaConf.create(text)
    model = OmegaConf.to_container(config.models[args.model_key], resolve=True)
    # The metadata repo may be the shared root, whose API client is older.
    # Always import this reviewed helper's sibling personal-capable transport.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv("/home/thomasjiralerspong/explore-persona-space/.env")
    import runpod_api

    account = runpod_api.graphql("query { myself { id teams { id } } }", personal=True)["myself"]
    if account != {"id": "user_2v9CcEeHWnPcoAVCf8YeCXKvupS", "teams": []}:
        raise RuntimeError("personal RunPod identity changed")
    data = runpod_api.graphql(QUERY, {"id": args.pod_id}, timeout=20, personal=True)
    contract = contract_from_pod(
        data.get("pod"),
        pod_id=args.pod_id,
        expected_name=args.expected_pod_name,
        model_key=args.model_key,
        model=model,
        source_sha=args.source_sha,
        now=time.time(),
        requested_seconds=args.allocation_seconds,
        ledger=json.loads(args.allocation_ledger.read_text()),
    )
    preserve_first_allocation(
        args.first_allocation_record, contract, reviewed_source_update=args.reviewed_source_update
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.out)
    checksum = hashlib.sha256(args.out.read_bytes()).hexdigest()
    result = {
        "contract": str(args.out.resolve()),
        "sha256": checksum,
        "pod_id": args.pod_id,
        "api_volume_gb": contract["api_volume_gb"],
        "paid_start_unix": contract["paid_start_unix"],
        "deadline_unix": contract["deadline_unix"],
    }
    if args.ssh_upload:
        if args.handle_file is None:
            raise RuntimeError("ssh-upload requires exact handle-file")
        handle = json.loads(args.handle_file.read_text())
        if handle["pod_name"] != args.expected_pod_name or handle["extra"]["pod_id"] != args.pod_id:
            raise RuntimeError("dispatcher handle identity mismatch")
        sentinel = handle["extra"]["expected_artifacts"]["sentinel_path"]
        if not re.fullmatch(r"/workspace/eval_results/issue_2673/[a-zA-Z0-9_./-]+", sentinel):
            raise RuntimeError("unexpected completion path")
        result["upload_receipt"] = upload_contract(
            runpod_api, args.pod_id, args.out, checksum, args.ssh_identity, sentinel
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
