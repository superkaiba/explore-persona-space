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


def allocation_seconds(ledger, *, pod_id, gpu_count, requested_seconds):
    """Enforce the cumulative allowance without resetting any prior paid time."""
    limit = ledger["max_gpu_hours"]
    if not math.isfinite(limit) or not 0 < limit <= 45:
        raise ValueError("invalid cumulative GPU-hour allowance")
    maximum_seconds = 7200
    if limit > 17:
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
    return requested_seconds


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
    if not pod or pod["id"] != pod_id or pod["name"] != expected_name:
        raise RuntimeError("actual live pod ID/name did not match the operator request")
    if re.fullmatch(r"pod-2673(?:-[a-z][a-z0-9-]{0,19})?", expected_name) is None:
        raise RuntimeError("this helper is restricted to task2673 managed pod names")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise RuntimeError("source_sha must be a complete immutable Git SHA")
    if pod["desiredStatus"] != "RUNNING":
        raise RuntimeError("storage contract requires an actual RUNNING pod")
    if ledger is not None and ledger["max_gpu_hours"] > 17:
        if model_key != "deepseek" or model.get("execution_mode") != "unpadded_singleton":
            raise ValueError("renewed allowance is restricted to DeepSeek singleton execution")
    count, gpu, volume, seconds = (
        (8, "H200", 1000, 7200) if model_key == "deepseek" else (1, "H100", 200, 3600)
    )
    if requested_seconds is not None:
        if ledger is None or requested_seconds > (12600 if model_key == "deepseek" else 3600):
            raise ValueError(
                "an explicit allocation requires the cumulative ledger and approved limit"
            )
        seconds = allocation_seconds(
            ledger, pod_id=pod_id, gpu_count=count, requested_seconds=requested_seconds
        )
    if model_key not in {"qwen", "deepseek"}:
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
    p.add_argument("--model-key", choices=["qwen", "deepseek"], required=True)
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
