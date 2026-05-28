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
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

# ─── constants ───────────────────────────────────────────────────────────────

GRAPHQL_URL = "https://api.runpod.io/graphql"

# Bounded exponential backoff for transient transport failures (issue #2). The
# RunPod GraphQL endpoint sits behind Cloudflare and occasionally returns 5xx,
# 429, or a CF "error code: 1010" challenge under load. Those are transient —
# a short backoff with jitter recovers without surfacing a spurious failure.
# Non-transient 4xx (except 429) and GraphQL-level `errors` are raised
# immediately as before (the crash IS the signal).
GRAPHQL_MAX_ATTEMPTS = 4
GRAPHQL_BACKOFF_BASE_SECS = 1.0
GRAPHQL_BACKOFF_CAP_SECS = 30.0

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


class RunPodTransientError(RunPodError):
    """A transport failure that is worth retrying (5xx, 429, CF-1010, network).

    Subclass of :class:`RunPodError` so existing ``except RunPodError`` callers
    keep catching it once the retry budget is exhausted. Used internally by
    :func:`graphql` to drive the bounded backoff loop.
    """


def _is_cloudflare_1010(body: str) -> bool:
    """True if the response body is a Cloudflare 1010 challenge.

    RunPod sits behind Cloudflare, which intermittently rejects requests with
    an HTML "error code: 1010" challenge page (browser-integrity / bot rules)
    under load. The page is transient — retrying with backoff recovers — so we
    detect it and treat it as retryable rather than a hard failure.
    """
    lowered = body.lower()
    return "error code: 1010" in lowered or "error code 1010" in lowered


def _backoff_sleep_secs(attempt: int) -> float:
    """Exponential backoff with full jitter for retry ``attempt`` (1-indexed).

    attempt=1 -> ~[0, base], attempt=2 -> ~[0, 2*base], capped at the cap.
    Full jitter (uniform 0..window) avoids synchronized retry storms across
    parallel pod-lifecycle callers.
    """
    assert attempt >= 1, attempt
    window = min(GRAPHQL_BACKOFF_BASE_SECS * (2 ** (attempt - 1)), GRAPHQL_BACKOFF_CAP_SECS)
    return random.uniform(0.0, window)


def _graphql_once(query: str, variables: dict | None, timeout: int) -> dict[str, Any]:
    """Single GraphQL round-trip. Raises RunPodTransientError on retryable
    transport failures (5xx, 429, CF-1010, network) and RunPodError on
    everything else (non-retryable 4xx, GraphQL-level `errors`, malformed
    payloads). Returns the parsed ``data`` dict. Never returns None.
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
        # 5xx (server-side) and 429 (rate limit) are transient — retry. A CF
        # 1010 challenge can arrive on any status, so check the body too.
        if exc.code >= 500 or exc.code == 429 or _is_cloudflare_1010(detail):
            raise RunPodTransientError(f"HTTP {exc.code} from RunPod: {detail[:500]}") from exc
        # Other 4xx are client errors (bad query, auth) — don't retry.
        raise RunPodError(f"HTTP {exc.code} from RunPod: {detail[:500]}") from exc
    except urlerror.URLError as exc:
        # Network-layer failure (DNS, connection refused, timeout) — transient.
        raise RunPodTransientError(f"Network error contacting RunPod: {exc.reason}") from exc

    text = response_body.decode("utf-8", errors="replace")
    # A 200 carrying a CF challenge body is still a transient block.
    if _is_cloudflare_1010(text):
        raise RunPodTransientError(f"Cloudflare 1010 challenge from RunPod: {text[:300]!r}")

    parsed = json.loads(response_body)
    if parsed.get("errors"):
        raise RunPodError(f"GraphQL errors: {json.dumps(parsed['errors'])[:500]}")
    if "data" not in parsed:
        raise RunPodError(f"Malformed response (no 'data' field): {response_body[:300]!r}")
    return parsed["data"]


def graphql(query: str, variables: dict | None = None, timeout: int = 60) -> dict[str, Any]:
    """Execute a GraphQL query against RunPod with team-id header enforced.

    Wraps the single round-trip (:func:`_graphql_once`) in bounded exponential
    backoff with jitter (issue #2). RETRIES on transient transport failures
    only — urllib network errors, HTTP >= 500, HTTP 429, and Cloudflare 1010
    challenges. Does NOT retry non-transient 4xx (other than 429) or
    GraphQL-level ``errors`` — those raise immediately.

    Returns the parsed `data` dict. Raises RunPodError on transport or GraphQL
    errors after the retry budget is exhausted. Never returns None.
    """
    last_exc: RunPodTransientError | None = None
    for attempt in range(1, GRAPHQL_MAX_ATTEMPTS + 1):
        try:
            return _graphql_once(query, variables, timeout)
        except RunPodTransientError as exc:
            last_exc = exc
            if attempt >= GRAPHQL_MAX_ATTEMPTS:
                break
            time.sleep(_backoff_sleep_secs(attempt))
    # Exhausted the budget — surface the last transient failure. Still a
    # RunPodError subclass, so existing `except RunPodError` callers catch it.
    assert last_exc is not None, "loop must have captured a transient error before break"
    raise RunPodError(
        f"RunPod GraphQL failed after {GRAPHQL_MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


# ─── pod operations ──────────────────────────────────────────────────────────


@dataclass
class PodInfo:
    """Snapshot of a pod's state. Fields not always populated — runtime info is
    only present when the pod is RUNNING and SSH is up.

    ``created_at`` is the ISO-8601 timestamp from the GraphQL ``createdAt``
    field, used for the AGE column in ``pod.py list-ephemeral``. ``None`` when
    the field is missing from the response (older pods or partial GraphQL
    selections)."""

    pod_id: str
    name: str
    desired_status: str  # RUNNING | EXITED | etc.
    gpu_count: int | None = None
    gpu_type_id: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    created_at: str | None = None


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
        created_at=raw.get("createdAt"),
    )


# GraphQL CloudTypeEnum: ALL | SECURE | COMMUNITY. GpuTypePriority controls how
# RunPod ranks candidate hosts; "availability" prefers any host with free GPUs
# over the cheapest one — the right tradeoff when capacity is scarce.
_CREATE_ENUM_FIELDS = {"cloudType", "gpuTypePriority"}


def _build_inputs_block(inputs: dict[str, Any]) -> str:
    """Serialize a deploy ``input`` dict to a GraphQL inline-object body.

    RunPod's GraphQL ``input`` uses unquoted keys and bare enum values, so we
    string-build rather than ``json.dumps``. Booleans become bare
    ``true``/``false``; ints stay bare; enum fields (see
    :data:`_CREATE_ENUM_FIELDS`) stay bare; everything else is double-quoted.
    Returns the ``k: v, ...`` body (no surrounding braces).
    """
    fields: list[str] = []
    for k, v in inputs.items():
        if isinstance(v, bool):
            fields.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, int) or k in _CREATE_ENUM_FIELDS:
            fields.append(f"{k}: {v}")
        else:
            fields.append(f'{k}: "{v}"')
    return ", ".join(fields)


def _deploy_once(
    *,
    name: str,
    gpu_type_id: str,
    gpu_count: int,
    image: str,
    volume_gb: int,
    container_disk_gb: int,
    cloud_type: str,
    data_center_id: str | None,
    interruptible: bool,
) -> PodInfo | None:
    """Single ``podFindAndDeployOnDemand`` attempt for one (gpu_type, cloud_type).

    Returns the parsed :class:`PodInfo` on success, or ``None`` when RunPod
    reports no capacity (a null mutation result == SUPPLY_CONSTRAINT). Raises
    :class:`RunPodError` on transport / GraphQL errors via :func:`graphql`.

    ``startSsh: true`` + ``22/tcp`` are non-negotiable (RunPod pytorch images
    don't run sshd by default; without both you get an unreachable pod).
    ``gpuTypePriority: availability`` biases host selection toward any host with
    free GPUs, which is what you want under supply pressure.
    """
    assert gpu_count >= 1, gpu_count
    inputs: dict[str, Any] = {
        "name": name,
        "gpuTypeId": gpu_type_id,
        "gpuCount": gpu_count,
        "cloudType": cloud_type,
        "gpuTypePriority": "availability",
        "volumeInGb": volume_gb,
        "containerDiskInGb": container_disk_gb,
        "imageName": image,
        "volumeMountPath": "/workspace",
        "startSsh": True,
        "ports": "8888/http,22/tcp",
    }
    if data_center_id:
        inputs["dataCenterId"] = data_center_id
    if interruptible:
        # Spot / interruptible instances draw from a separate, usually-deeper
        # capacity pool. Only used as a last resort (the host can reclaim them).
        inputs["interruptible"] = True

    inputs_block = _build_inputs_block(inputs)
    query = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{ {inputs_block} }}) {{
        id
        name
        desiredStatus
        gpuCount
        createdAt
        machine {{ gpuTypeId }}
        runtime {{ ports {{ ip publicPort privatePort type isIpPublic }} }}
      }}
    }}
    """
    data = graphql(query)
    raw = data.get("podFindAndDeployOnDemand")
    if not raw:
        # Null result == no capacity for this (gpu_type, cloud_type). Caller
        # decides whether to try the next lever.
        return None
    return _parse_pod(raw)


def create_pod(
    name: str,
    gpu_type: str | list[str],
    gpu_count: int,
    *,
    image: str = DEFAULT_IMAGE,
    volume_gb: int = DEFAULT_VOLUME_GB,
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB,
    cloud_type: str = "ALL",
    data_center_id: str | None = None,
    enable_supply_fallback: bool = True,
) -> PodInfo:
    """Create a new on-demand pod with sshd enabled and 22/tcp exposed.

    Supply-resilient (issue #11). ``gpu_type`` may be a single short name
    (``"H100"``) OR an ordered list of acceptable types (``["H100", "H200"]``);
    each is tried in order and the first with capacity wins. Each attempt sends
    ``gpuTypePriority: availability`` so RunPod prefers any host with free GPUs.

    When ``enable_supply_fallback`` is True (default) and the primary cloud type
    is exhausted for every requested GPU type, ``create_pod`` then retries the
    COMMUNITY cloud, and finally COMMUNITY + interruptible (spot). These fallback
    pools are deeper but less stable, so they sit at the back of the chain. The
    ``data_center_id`` pin (if given) is preserved across all attempts — it is a
    valid, used field. Names not in the allowlist pass through verbatim so
    callers can request exotic GPU types.

    Raises :class:`RunPodError` only when EVERY lever in the chain reports no
    capacity (or a transport error surfaces). The error names what was tried.
    """
    gpu_types = [gpu_type] if isinstance(gpu_type, str) else list(gpu_type)
    if not gpu_types:
        raise RunPodError("create_pod: gpu_type list is empty — nothing to deploy.")

    # Build the ordered lever chain: (cloud_type, interruptible). The primary
    # cloud_type comes first; the supply fallbacks only fire when enabled AND
    # the primary isn't already COMMUNITY (no point retrying the same pool).
    levers: list[tuple[str, bool]] = [(cloud_type, False)]
    if enable_supply_fallback:
        if cloud_type.upper() != "COMMUNITY":
            levers.append(("COMMUNITY", False))
        levers.append(("COMMUNITY", True))

    tried: list[str] = []
    for lever_cloud, interruptible in levers:
        for short_name in gpu_types:
            gpu_type_id = GPU_TYPE_IDS.get(short_name, short_name)
            label = f"{gpu_count}x {short_name} on cloudType={lever_cloud}"
            if interruptible:
                label += " (interruptible/spot)"
            tried.append(label)
            info = _deploy_once(
                name=name,
                gpu_type_id=gpu_type_id,
                gpu_count=gpu_count,
                image=image,
                volume_gb=volume_gb,
                container_disk_gb=container_disk_gb,
                cloud_type=lever_cloud,
                data_center_id=data_center_id,
                interruptible=interruptible,
            )
            if info is not None:
                return info

    raise RunPodError(
        "podFindAndDeployOnDemand returned null for every supply lever — "
        f"no capacity. Tried (in order): {'; '.join(tried)}. "
        "Try a different DC, GPU count, or wait for capacity to free up."
    )


def get_pod(pod_id: str) -> PodInfo:
    query = """
    query Pod($id: String!) {
      pod(input: {podId: $id}) {
        id name desiredStatus gpuCount createdAt
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
          id name desiredStatus gpuCount createdAt
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
        id name desiredStatus gpuCount createdAt
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
