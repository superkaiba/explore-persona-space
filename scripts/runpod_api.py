"""RunPod GraphQL client, hard-scoped to the Anthropic Safety Research team.

Why this module exists
----------------------
Every RunPod request from this project MUST carry the `X-Team-Id` header. Without
it the API silently returns zero pods (different account scope), so a missing
header looks like "you have no pods" instead of "you used the wrong scope" — a
deeply confusing footgun. This module fails closed if the team-id is unset or if
a response does not match the expected team.

It also pins the SSH-bring-up parameters that RunPod pytorch images need
(`startSsh: true`, expose `22/tcp`) so callers can't accidentally create
unreachable pods.

Public surface
--------------
- create_pod(...)
- start_pod(pod_id)              # alias of resume; "start" = first-time spin-up
- stop_pod(pod_id)               # pause; volume + container disk preserved
- resume_pod(pod_id, gpu_count)  # bring a stopped pod back; IP changes
- terminate_pod(pod_id)          # destroy; volume gone
- get_pod(pod_id)
- list_team_pods()
- wait_for_ssh(pod_id, timeout=600)  # poll until 22/tcp is publicly mapped

CLI usage is via scripts/pod_lifecycle.py — this module is the library.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

# ─── constants ───────────────────────────────────────────────────────────────

GRAPHQL_URL = "https://api.runpod.io/graphql"

# Anthropic Safety Research team. Override with RUNPOD_TEAM_ID env if you ever
# need to act in a different scope (you almost never do).
DEFAULT_TEAM_ID = "cm8ipuyys0004l108gb23hody"

# Image pinned to match the existing fleet so HF cache layouts are identical.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Minimum disk to comfortably hold a 7B+ model + cache. Tunable per-call.
DEFAULT_VOLUME_GB = 200
DEFAULT_CONTAINER_DISK_GB = 50

# RunPod requires GPU type IDs in this exact form.
GPU_TYPE_IDS = {
    "H100": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    "A100": "NVIDIA A100-SXM4-80GB",
}


# ─── env loading ─────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    """Best-effort .env loader (project root). Does not override existing env."""
    root = Path(__file__).resolve().parent.parent
    env_file = root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env() -> tuple[str, str]:
    """Return (api_key, team_id). Raises RuntimeError if either is missing."""
    _load_dotenv()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    team_id = os.environ.get("RUNPOD_TEAM_ID", DEFAULT_TEAM_ID).strip()
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Add it to .env or export it. The RunPod GraphQL "
            "API needs it AND the team-id header — both are mandatory."
        )
    if not team_id:
        raise RuntimeError(
            "RUNPOD_TEAM_ID resolved to empty. Either unset (uses Anthropic Safety "
            "Research default) or set explicitly to your team id."
        )
    return api_key, team_id


# ─── GraphQL transport ───────────────────────────────────────────────────────


class RunPodError(RuntimeError):
    """Wraps a non-2xx response or a 'errors' field in the GraphQL payload."""


def graphql(query: str, variables: dict | None = None, timeout: int = 60) -> dict[str, Any]:
    """Execute a GraphQL query against RunPod with team-id header enforced.

    Returns the parsed `data` dict. Raises RunPodError on transport or GraphQL
    errors. Never returns None.
    """
    api_key, team_id = _require_env()

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Team-Id": team_id,
            "Content-Type": "application/json",
            # RunPod's CF rules block the default Python-urllib UA (1010). Send
            # a curl-shaped UA so requests aren't shadow-rejected.
            "User-Agent": "explore-persona-space/pod-lifecycle (curl-compat)",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read()
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RunPodError(f"HTTP {exc.code} from RunPod: {detail[:500]}") from exc
    except urlerror.URLError as exc:
        raise RunPodError(f"Network error contacting RunPod: {exc.reason}") from exc

    parsed = json.loads(response_body)
    if parsed.get("errors"):
        raise RunPodError(f"GraphQL errors: {json.dumps(parsed['errors'])[:500]}")
    if "data" not in parsed:
        raise RunPodError(f"Malformed response (no 'data' field): {response_body[:300]!r}")
    return parsed["data"]


# ─── pod operations ──────────────────────────────────────────────────────────


@dataclass
class PodInfo:
    """Snapshot of a pod's state. Fields not always populated — runtime info is
    only present when the pod is RUNNING and SSH is up."""

    pod_id: str
    name: str
    desired_status: str  # RUNNING | EXITED | etc.
    gpu_count: int | None = None
    gpu_type_id: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None


def _parse_pod(raw: dict[str, Any]) -> PodInfo:
    runtime = raw.get("runtime") or {}
    ports = runtime.get("ports") or []

    ssh_host: str | None = None
    ssh_port: int | None = None
    for port in ports:
        if port.get("type") == "tcp" and port.get("privatePort") == 22 and port.get("isIpPublic"):
            ssh_host = port.get("ip")
            ssh_port = port.get("publicPort")
            break

    machine = raw.get("machine") or {}
    return PodInfo(
        pod_id=raw["id"],
        name=raw.get("name", ""),
        desired_status=raw.get("desiredStatus", ""),
        gpu_count=raw.get("gpuCount"),
        gpu_type_id=machine.get("gpuTypeId"),
        ssh_host=ssh_host,
        ssh_port=ssh_port,
    )


def create_pod(
    name: str,
    gpu_type: str,
    gpu_count: int,
    *,
    image: str = DEFAULT_IMAGE,
    volume_gb: int = DEFAULT_VOLUME_GB,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    cloud_type: str = "ALL",
    data_center_id: str | None = None,
) -> PodInfo:
    """Create a new on-demand pod with sshd enabled and 22/tcp exposed.

    `gpu_type` is the short name (H100, H200, A100); we translate to RunPod's
    full gpuTypeId. Names that aren't in the allowlist are passed through
    verbatim so callers can use exotic types when needed.
    """
    gpu_type_id = GPU_TYPE_IDS.get(gpu_type, gpu_type)

    # RunPod's pytorch images don't run sshd by default. `startSsh: true` makes
    # the container start sshd natively (uses the SSH key on your RunPod
    # account); without it AND port 22 exposed, you get a pod you can't reach.
    inputs = {
        "name": name,
        "gpuTypeId": gpu_type_id,
        "gpuCount": gpu_count,
        "cloudType": cloud_type,
        "volumeInGb": volume_gb,
        "containerDiskInGb": container_disk_gb,
        "imageName": image,
        "volumeMountPath": "/workspace",
        "startSsh": True,
        "ports": "8888/http,22/tcp",
    }
    if data_center_id:
        inputs["dataCenterId"] = data_center_id

    # RunPod's GraphQL `input` uses unquoted keys, so we string-build the block
    # rather than json.dumps. Booleans become bare `true`/`false`; ints stay
    # bare; strings are double-quoted.
    fields = []
    for k, v in inputs.items():
        if isinstance(v, bool):
            fields.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, int):
            fields.append(f"{k}: {v}")
        else:
            fields.append(f'{k}: "{v}"')
    inputs_block = ", ".join(fields)

    query = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{ {inputs_block} }}) {{
        id
        name
        desiredStatus
        gpuCount
        machine {{ gpuTypeId }}
        runtime {{ ports {{ ip publicPort privatePort type isIpPublic }} }}
      }}
    }}
    """
    data = graphql(query)
    raw = data.get("podFindAndDeployOnDemand")
    if not raw:
        raise RunPodError(
            f"podFindAndDeployOnDemand returned null — no capacity for "
            f"{gpu_count}x {gpu_type} on cloudType={cloud_type}. "
            f"Try a different DC, GPU type, or count."
        )
    return _parse_pod(raw)


def get_pod(pod_id: str) -> PodInfo:
    query = """
    query Pod($id: String!) {
      pod(input: {podId: $id}) {
        id name desiredStatus gpuCount
        machine { gpuTypeId }
        runtime { ports { ip publicPort privatePort type isIpPublic } }
      }
    }
    """
    data = graphql(query, {"id": pod_id})
    raw = data.get("pod")
    if not raw:
        raise RunPodError(f"Pod {pod_id} not found in this team.")
    return _parse_pod(raw)


def list_team_pods() -> list[PodInfo]:
    query = """
    {
      myself {
        pods {
          id name desiredStatus gpuCount
          machine { gpuTypeId }
          runtime { ports { ip publicPort privatePort type isIpPublic } }
        }
      }
    }
    """
    data = graphql(query)
    pods = (data.get("myself") or {}).get("pods") or []
    return [_parse_pod(p) for p in pods]


def stop_pod(pod_id: str) -> PodInfo:
    """Pause a running pod. Volume + container disk are preserved; IP is released."""
    query = """
    mutation Stop($id: String!) {
      podStop(input: {podId: $id}) { id name desiredStatus }
    }
    """
    data = graphql(query, {"id": pod_id})
    raw = data.get("podStop")
    if not raw:
        raise RunPodError(f"podStop returned null for {pod_id}")
    return _parse_pod(raw)


def resume_pod(pod_id: str, gpu_count: int) -> PodInfo:
    """Resume a stopped pod. `gpu_count` MUST match the pod's original GPU count
    (RunPod rejects mismatched values). IP/port change on every resume."""
    query = """
    mutation Resume($id: String!, $n: Int!) {
      podResume(input: {podId: $id, gpuCount: $n}) {
        id name desiredStatus gpuCount
        machine { gpuTypeId }
        runtime { ports { ip publicPort privatePort type isIpPublic } }
      }
    }
    """
    data = graphql(query, {"id": pod_id, "n": gpu_count})
    raw = data.get("podResume")
    if not raw:
        raise RunPodError(f"podResume returned null for {pod_id}")
    return _parse_pod(raw)


# Resume from never-started == start. RunPod doesn't distinguish, but we
# expose an alias so calling code reads correctly.
start_pod = resume_pod


def terminate_pod(pod_id: str) -> bool:
    """Destroy a pod permanently. Volume is gone. Returns True on success."""
    query = """
    mutation Terminate($id: String!) {
      podTerminate(input: {podId: $id})
    }
    """
    data = graphql(query, {"id": pod_id})
    # podTerminate returns null on success; errors raise above.
    return data.get("podTerminate") is None or data.get("podTerminate") is True


# ─── readiness ───────────────────────────────────────────────────────────────


def wait_for_ssh(pod_id: str, timeout: int = 600, poll_interval: int = 10) -> PodInfo:
    """Poll until the pod has a public 22/tcp mapping. Returns the PodInfo with
    ssh_host/ssh_port populated. Raises RunPodError on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = get_pod(pod_id)
        if info.ssh_host and info.ssh_port:
            return info
        time.sleep(poll_interval)
    raise RunPodError(
        f"Pod {pod_id} did not expose public 22/tcp within {timeout}s. "
        f"Last desiredStatus: {info.desired_status if 'info' in dir() else 'unknown'}"
    )
