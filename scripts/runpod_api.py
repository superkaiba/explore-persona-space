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
- estimate_pod_hourly_rate(gpu_type_id, gpu_count)  # USD/hr best-effort
- current_account_hourly_burn()  # sum estimated $/hr across RUNNING managed pods

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

# Full RunPod gpuTypeIds are vendor-prefixed ("NVIDIA H100 NVL",
# "AMD Instinct MI300X OC", ...). Used to distinguish a deliberate exotic
# full id (passed through verbatim) from a typo'd / colloquial short name
# ("H100 SXM"), which RunPod treats as a nonexistent type — see
# resolve_gpu_type_id.
_FULL_GPU_TYPE_ID_PREFIXES = ("NVIDIA ", "AMD ")


def resolve_gpu_type_id(gpu_type: str) -> str:
    """Map a short GPU name to its full RunPod gpuTypeId, rejecting unknowns.

    Known short names (:data:`GPU_TYPE_IDS`) resolve to their full id. A name
    that already looks like a full RunPod id (vendor-prefixed, e.g.
    ``"NVIDIA H100 NVL"``) passes through verbatim so callers can request
    exotic GPU types. Anything else raises :class:`RunPodError`: RunPod treats
    a nonexistent gpuTypeId as permanent no-capacity (null mutation result /
    SUPPLY_CONSTRAINT), indistinguishable from a genuine shortage, so a typo
    spins the wait-for-capacity loop forever — task #537 waited 88 minutes on
    ``"H100 SXM"`` (the SXM card's id is ``GPU_TYPE_IDS["H100"]``).
    """
    resolved = GPU_TYPE_IDS.get(gpu_type, gpu_type)
    if gpu_type not in GPU_TYPE_IDS and not resolved.startswith(_FULL_GPU_TYPE_ID_PREFIXES):
        raise RunPodError(
            f"gpu_type {gpu_type!r} is not a known short name "
            f"(valid: {sorted(GPU_TYPE_IDS)}) and does not look like a full RunPod "
            f"gpuTypeId (expected a vendor prefix like 'NVIDIA ...'). RunPod reports "
            f"a nonexistent gpuTypeId as no-capacity, so the wait-for-capacity loop "
            f"would wait forever on an impossible request (#537)."
        )
    return resolved


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


class RunPodNoCapacityError(RunPodError):
    """Raised by :func:`create_pod` when EVERY supply lever returned null —
    i.e. RunPod reports no capacity across the gpu-type list x cloud-type x
    interruptible chain.

    Distinct from the generic :class:`RunPodError` (auth, bad config,
    transport-budget-exhausted, empty gpu list) so a higher-level
    wait-for-capacity policy loop can catch ONLY the no-capacity case and
    wait/retry, while every other failure class still fails fast per the
    "fail fast — never hide failures" rule.

    Subclass of :class:`RunPodError` so existing ``except RunPodError``
    callers keep catching it.
    """


class RunPodInsufficientBalanceError(RunPodError):
    """Raised when RunPod refuses ``podFindAndDeployOnDemand`` /
    ``podResume`` because the projected total account $/hr would exceed
    the console-side spending cap. The actual GraphQL error string is:

        INSUFFICIENT_BALANCE: Renting this pod would put you over your
        current spending limit ($X/hr)

    Why this is its own class (vs the generic ``RunPodError``):
    INSUFFICIENT_BALANCE is **transient + no-cost-while-idle** — while
    the provision/resume is refused, nothing is running, so no $/hr is
    being spent. The condition clears the moment any other pod on the
    team frees $/hr headroom (a stop/terminate, or a sibling experiment
    finishing). The right behavior is the SAME as
    :class:`RunPodNoCapacityError`: retry-with-backoff in the
    pod_lifecycle policy layer, NEVER fail-exit a task to ``blocked``.
    Incident: task #506 (2026-06-08) fail-exited to ``blocked`` on this
    refusal before the special-case classification existed.

    Subclass of :class:`RunPodError` so existing ``except RunPodError``
    callers keep catching it (after the retry budget, if any, is exhausted).
    """


class RunPodSupplyConstraintError(RunPodError):
    """Raised when a mutation (``podFindAndDeployOnDemand`` / ``podResume``)
    is refused with a GraphQL ``errors`` payload whose ``extensions.code``
    is ``SUPPLY_CONSTRAINT`` ("There are no longer any instances available
    with the requested specifications").

    This is the ERROR-PAYLOAD shape of the same no-capacity condition that
    usually arrives as a null mutation result (which :func:`_deploy_once`
    returns as ``None``). Before this class existed the payload shape raised
    a bare :class:`RunPodError`, which (a) aborted :func:`create_pod`'s
    supply-lever chain before COMMUNITY / interruptible were tried, and (b)
    bypassed ``create_pod_with_wait_for_capacity``'s except clause (it
    catches only :class:`RunPodNoCapacityError` +
    :class:`RunPodInsufficientBalanceError`), crashing an autonomous
    provision (incident: task #537, 2026-06-11).

    :func:`_deploy_once` catches this class and returns ``None`` so the
    lever chain advances; once every lever is exhausted :func:`create_pod`
    raises :class:`RunPodNoCapacityError` as before, which the
    wait-for-capacity policy loop already handles. Subclass of
    :class:`RunPodError` so existing ``except RunPodError`` callers keep
    catching it.
    """


# Markers used to detect INSUFFICIENT_BALANCE in a GraphQL ``errors`` payload
# or a raised RunPodError message. RunPod has used both the explicit error
# code (``INSUFFICIENT_BALANCE``) and the human-readable phrase ("spending
# limit"), so match defensively on either. Case-insensitive substring
# match. Lives in this module (not pod_lifecycle) so the transport layer
# can raise the typed exception directly; pod_lifecycle's retry policy
# then catches it by class, not by string-sniffing the message.
_INSUFFICIENT_BALANCE_MARKERS: tuple[str, ...] = (
    "insufficient_balance",
    "insufficient balance",
    "spending limit",
    "over your current spending",
)


def _is_insufficient_balance_error(error_text: str) -> bool:
    """True if a GraphQL ``errors`` payload string or a ``RunPodError``
    message looks like a RunPod ``INSUFFICIENT_BALANCE`` refusal
    (projected account $/hr over the console cap). Case-insensitive
    substring match against :data:`_INSUFFICIENT_BALANCE_MARKERS`.
    """
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _INSUFFICIENT_BALANCE_MARKERS)


# Markers used to detect a SUPPLY_CONSTRAINT refusal in a GraphQL ``errors``
# payload. RunPod embeds the explicit error code verbatim in the serialized
# errors text (``extensions.code == "SUPPLY_CONSTRAINT"``) and/or a
# human-readable phrase; match defensively on either, case-insensitive.
# Phrase set mirrors pod_lifecycle's resume-side ``_SUPPLY_CONSTRAINT_MARKERS``
# minus its locally generated "podresume returned null" string (that one never
# appears inside a GraphQL payload — it is synthesized by resume_pod itself).
_SUPPLY_CONSTRAINT_MARKERS: tuple[str, ...] = (
    "supply_constraint",
    "supplyconstraint",
    "no longer any instances available",
    "not enough free gpu",
    "no free gpu",
    "insufficient capacity",
)


def _is_supply_constraint_error(error_text: str) -> bool:
    """True if a GraphQL ``errors`` payload string looks like a RunPod
    ``SUPPLY_CONSTRAINT`` refusal (no instances available with the requested
    specifications). Case-insensitive substring match against
    :data:`_SUPPLY_CONSTRAINT_MARKERS`.
    """
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _SUPPLY_CONSTRAINT_MARKERS)


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
        err_text = json.dumps(parsed["errors"])[:500]
        # RunPod surfaces INSUFFICIENT_BALANCE (projected account $/hr over
        # the console cap) as a GraphQL error rather than a transport-level
        # failure. Classify it so the pod_lifecycle retry policy can wait
        # for headroom instead of fail-exiting (incident #506, 2026-06-08).
        if _is_insufficient_balance_error(err_text):
            raise RunPodInsufficientBalanceError(f"GraphQL errors: {err_text}")
        # SUPPLY_CONSTRAINT also arrives as a GraphQL error payload, not only
        # as a null mutation result. Classify it so the deploy path treats it
        # as the no-capacity case and create_pod's supply-lever chain advances
        # instead of crashing (incident: task #537, 2026-06-11).
        if _is_supply_constraint_error(err_text):
            raise RunPodSupplyConstraintError(f"GraphQL errors: {err_text}")
        raise RunPodError(f"GraphQL errors: {err_text}")
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


# GraphQL CloudTypeEnum: ALL | SECURE | COMMUNITY. (gpuTypePriority is NOT a
# field on PodFindAndDeployOnDemandInput — RunPod rejects it with HTTP 400, so
# it is not sent. Supply resilience comes from the gpu-type list + COMMUNITY +
# interruptible lever chain in create_pod, not a host-ranking hint.)
_CREATE_ENUM_FIELDS = {"cloudType"}


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
    reports no capacity — EITHER a null mutation result OR a GraphQL error
    payload with ``extensions.code == SUPPLY_CONSTRAINT`` (same condition,
    two wire shapes; incident: task #537, 2026-06-11). Raises
    :class:`RunPodError` on other transport / GraphQL errors via
    :func:`graphql`.

    ``startSsh: true`` + ``22/tcp`` are non-negotiable (RunPod pytorch images
    don't run sshd by default; without both you get an unreachable pod).
    Supply resilience comes from the gpu-type list + COMMUNITY + interruptible
    lever chain in :func:`create_pod` (``gpuTypePriority`` is NOT a valid field
    on ``PodFindAndDeployOnDemandInput`` and is rejected with HTTP 400).
    """
    assert gpu_count >= 1, gpu_count
    inputs: dict[str, Any] = {
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
    if interruptible:
        # The RunPod GraphQL schema no longer defines `interruptible` on
        # PodFindAndDeployOnDemandInput — sending it returns HTTP 400
        # GRAPHQL_VALIDATION_FAILED (observed 2026-06-11, #537), which crashed
        # the lever chain. Until spot support is re-implemented against the
        # current API (likely a separate mutation), treat the spot lever as
        # unavailable: report no-capacity so the chain/wait-loop stay alive.
        return None

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
    try:
        data = graphql(query)
    except RunPodSupplyConstraintError:
        # Error-payload shape of the null-result no-capacity case: RunPod
        # sometimes refuses with extensions.code == SUPPLY_CONSTRAINT instead
        # of returning a null mutation result. Same meaning, same handling —
        # return None so create_pod advances to the next supply lever
        # (COMMUNITY / interruptible) before any wait loop sleeps (#537).
        return None
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
    each is tried in order and the first with capacity wins.

    When ``enable_supply_fallback`` is True (default) and the primary cloud type
    is exhausted for every requested GPU type, ``create_pod`` then retries the
    COMMUNITY cloud, and finally COMMUNITY + interruptible (spot). These fallback
    pools are deeper but less stable, so they sit at the back of the chain. The
    ``data_center_id`` pin (if given) is preserved across all attempts — it is a
    valid, used field. Names not in the allowlist must look like a full RunPod
    gpuTypeId (vendor-prefixed, e.g. ``"NVIDIA H100 NVL"``) to pass through
    verbatim for exotic GPU types; anything else fails fast via
    :func:`resolve_gpu_type_id` BEFORE any deploy attempt, because RunPod
    reports a nonexistent gpuTypeId as no-capacity and the wait loop would
    spin forever (#537).

    Raises :class:`RunPodNoCapacityError` when EVERY lever in the chain
    reports no capacity (so a higher-level wait-for-capacity policy can catch
    that specific case and retry), or :class:`RunPodError` for transport /
    auth / bad-config failures. The no-capacity error names what was tried.
    """
    gpu_types = [gpu_type] if isinstance(gpu_type, str) else list(gpu_type)
    if not gpu_types:
        raise RunPodError("create_pod: gpu_type list is empty — nothing to deploy.")
    # Fail fast on unmapped / typo'd GPU names before any API call (#537):
    # a nonexistent gpuTypeId is indistinguishable from a genuine shortage.
    resolved_ids = {short_name: resolve_gpu_type_id(short_name) for short_name in gpu_types}

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
            gpu_type_id = resolved_ids[short_name]
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

    raise RunPodNoCapacityError(
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


# ─── account hourly-burn estimation ──────────────────────────────────────────
#
# Why this module owns it: RunPod's account-level spending limit is enforced
# server-side by RunPod (the "$80/hr cap" — set in the RunPod console, surfaces
# as ``INSUFFICIENT_BALANCE: Renting this pod would put you over your current
# spending limit ($X/hr)`` on ``podFindAndDeployOnDemand`` and ``podResume``).
# We do NOT discover the cap from the API — there is no GraphQL field for it —
# we mirror it locally as a config knob so callers can fail LOUD pre-flight
# with the projected total instead of letting RunPod refuse mid-run after the
# experiment is already underway (incidents #503, #505 on 2026-06-05).
#
# Per-pod hourly rate is NOT exposed on the GraphQL Pod selections we use
# (``id name desiredStatus gpuCount machine{gpuTypeId} runtime{...} createdAt``).
# Adding a speculative ``costPerHr`` field to those queries is risky: if the
# field doesn't exist on the schema, the entire ``list_team_pods`` call raises
# ``GraphQL errors`` and the lifecycle goes blind. So we estimate from
# ``(gpu_type, gpu_count)`` using env-overridable per-GPU rates, and explicitly
# label the fallback as a conservative over-estimate so the guard fails SAFE
# (refuses a borderline provision) rather than UNSAFE (under-estimates and
# lets RunPod refuse mid-run anyway). The default rates here are conservative
# upper bounds — set them precisely for your account via env vars below.


# Per-GPU $/hr defaults. Intentionally over-estimate so the guard fails SAFE
# (refuses a borderline provision) when the user hasn't tuned the rates to
# their actual RunPod console pricing. Override per-account via:
#   RUNPOD_RATE_H100_USD, RUNPOD_RATE_H200_USD, RUNPOD_RATE_A100_USD
# Unknown GPU types fall back to RUNPOD_FALLBACK_HOURLY_PER_GPU_USD (default
# 6.0 — high enough to over-estimate any common datacenter GPU).
_DEFAULT_PER_GPU_RATES_USD: dict[str, float] = {
    "H100": 4.0,
    "H200": 5.5,
    "A100": 2.5,
}
_FALLBACK_PER_GPU_RATE_USD = 6.0


def _short_gpu_name(gpu_type_id: str | None) -> str:
    """Map a full RunPod ``gpuTypeId`` (e.g. ``NVIDIA H100 80GB HBM3``) to the
    short name used as a rate-table key (``H100``). Returns the original string
    on no-match so callers can log the unknown id verbatim.
    """
    if not gpu_type_id:
        return ""
    for short in _DEFAULT_PER_GPU_RATES_USD:
        if short in gpu_type_id:
            return short
    return gpu_type_id


def _per_gpu_rate_usd(short_gpu_name: str) -> float:
    """Return $/hr per GPU for ``short_gpu_name`` (H100/H200/A100/...). Env
    overrides win (``RUNPOD_RATE_<NAME>_USD``); unknown GPUs fall back to
    ``RUNPOD_FALLBACK_HOURLY_PER_GPU_USD`` (default 6.0). Always non-negative.
    """
    if short_gpu_name in _DEFAULT_PER_GPU_RATES_USD:
        env_key = f"RUNPOD_RATE_{short_gpu_name}_USD"
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            try:
                return max(0.0, float(env_val))
            except ValueError:
                # Bad env value — fall through to default rather than crash.
                pass
        return _DEFAULT_PER_GPU_RATES_USD[short_gpu_name]
    # Unknown GPU type — use the conservative fallback per-GPU rate.
    fallback = os.environ.get("RUNPOD_FALLBACK_HOURLY_PER_GPU_USD", "").strip()
    if fallback:
        try:
            return max(0.0, float(fallback))
        except ValueError:
            pass
    return _FALLBACK_PER_GPU_RATE_USD


def estimate_pod_hourly_rate(gpu_type_id: str | None, gpu_count: int | None) -> float:
    """Best-effort $/hr estimate for a pod with ``gpu_count`` GPUs of
    ``gpu_type_id``. Conservative — see the module-level note above.

    Returns 0.0 only when ``gpu_count`` is None/0 (no GPUs assigned yet).
    Never raises; unknown GPU types use the fallback rate so the caller's
    guard can still produce a finite projected total.
    """
    n = gpu_count or 0
    if n <= 0:
        return 0.0
    rate = _per_gpu_rate_usd(_short_gpu_name(gpu_type_id))
    return float(n) * rate


def current_account_hourly_burn() -> tuple[float, list[tuple[str, float]]]:
    """Sum estimated $/hr across every RUNNING pod on the team account.

    Returns ``(total_usd_per_hr, breakdown)`` where ``breakdown`` is a list of
    ``(pod_name, pod_hourly_usd)`` for each RUNNING pod (sorted by cost
    descending). Includes both managed (`pod-N` / `epm-issue-N`) and unmanaged
    pods on the account — the RunPod spending cap applies to ALL of them, so
    the guard must too.

    Stopped (EXITED) pods are excluded because they don't accrue hourly GPU
    charges — only volume storage, which is not subject to the $/hr cap.

    Raises :class:`RunPodError` if the API is unreachable. Callers (the
    pre-provision guard) treat that as fail-loud: if we can't query the API
    we can't make the decision, so don't provision.
    """
    live = list_team_pods()
    breakdown: list[tuple[str, float]] = []
    for p in live:
        if (p.desired_status or "").upper() != "RUNNING":
            continue
        rate = estimate_pod_hourly_rate(p.gpu_type_id, p.gpu_count)
        breakdown.append((p.name or p.pod_id, rate))
    breakdown.sort(key=lambda row: row[1], reverse=True)
    total = sum(rate for _, rate in breakdown)
    return total, breakdown
