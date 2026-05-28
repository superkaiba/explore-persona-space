"""Ephemeral pod lifecycle: provision, stop, resume, terminate, cleanup.

How it fits with the rest of the pod tooling
--------------------------------------------
- ``runpod_api.py`` is the GraphQL transport. Always team-scoped.
- ``gpu_heuristics.py`` maps experiment intents to GPU specs.
- ``pods.conf`` holds connection info for SSH/MCP config generation. We append /
  update / remove rows here so pods provisioned by this script become reachable
  via ``ssh pod-NNN`` after a ``pod_config.py --sync``.
- ``pods_ephemeral.json`` (sidecar) — write-through metadata cache.

Authority split (issue #282 [1/4])
----------------------------------
The live RunPod API is **authoritative for state-of-pod** (existence, status,
host, port, GPU count, GPU type, ``created_at``). The sidecar JSON stores
**project-side metadata** that has no live-API equivalent: the workload
``gpu_intent``, ``ttl_days``, ``stopped_at`` (when we paused), free-form
``notes``, and the RunPod ``pod_id`` keyed by our `pod-N` name (legacy
`epm-issue-N` names are still recognized — see :func:`_is_managed_pod`). Reads
NEVER consult JSON for status/host/port; the merged ``EphemeralPod`` view
returned by ``_load_state`` exposes API-derived fields as properties that
delegate to the underlying ``PodInfo``.

This eliminates the drift class where a pod is stopped/terminated externally
and the sidecar keeps reporting ``status=running``.

Naming convention
-----------------
Ephemeral pods are named ``pod-<N>`` where ``<N>`` is the GitHub issue
number. One pod per issue. Follow-up issues that derive from #N can resume
#N's pod.

The legacy prefix ``epm-issue-<N>`` (used before the rename) is still
recognized by :func:`_is_managed_pod` and :func:`_issue_from_pod_name` so
in-flight pods provisioned under the old convention keep working until
they're terminated. New pods always use ``pod-<N>``.

The bootstrap step is gated by ``--no-bootstrap`` because resumed pods already
have the repo + caches; you only bootstrap on first provision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Same package — sibling modules.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gpu_heuristics import GpuSpec, list_intents, resolve_intent  # noqa: E402
from pod_config import (  # noqa: E402
    Pod,
    cmd_sync,
    parse_pods_conf,
    write_pods_conf,
)
from runpod_api import (  # noqa: E402
    PodInfo,
    RunPodError,
    create_pod,
    list_team_pods,
    resume_pod,
    stop_pod,
    terminate_pod,
    wait_for_ssh,
)

PROJECT_ROOT = SCRIPT_DIR.parent
EPHEMERAL_STATE = SCRIPT_DIR / "pods_ephemeral.json"
DEFAULT_TTL_DAYS = 7
BOOTSTRAP_SCRIPT = SCRIPT_DIR / "bootstrap_pod.sh"


# ─── ephemeral state file ────────────────────────────────────────────────────


@dataclass
class EphemeralMetadata:
    """Project-side metadata about an ephemeral pod.

    These fields have no live-API equivalent — the live API knows nothing
    about *why* a pod was provisioned, our preferred TTL, or freeform notes.
    Persisted to ``pods_ephemeral.json``; merged with a live ``PodInfo`` to
    produce an :class:`EphemeralPod` view in :func:`_load_state`.

    ``manual_override`` (added 2026-05-27, post-mortem from task #391): when
    True, the auto-refresh paths (drift repair in :func:`_load_state` and
    host/port writes in :func:`_upsert_pods_conf`) refuse to overwrite
    pod_id / host / port from the live API. Set by
    ``pod_config.cmd_update`` so that a manual ``--update`` survives a
    later ``provision`` / ``resume`` / cron run that matched a different
    RunPod entry sharing the same pod name. Cleared by ``cmd_provision``
    (fresh pod) and the ``--clear-override`` flag.
    """

    name: str  # e.g. "pod-125" (legacy "epm-issue-125" still recognized)
    pod_id: str  # RunPod id (metadata-side: our name->pod_id mapping)
    issue: int  # source issue number
    gpu_intent: str = "custom"  # the intent string used (or "custom")
    ttl_days: int = DEFAULT_TTL_DAYS
    stopped_at: str | None = None  # ISO 8601 — when WE paused it
    notes: str = ""
    manual_override: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EphemeralPod:
    """Merged view of project-side metadata + live API state.

    Status, host, port, gpu_count, gpu_type, and created_at are API-derived
    (delegate to ``info``). gpu_intent, ttl_days, stopped_at, notes are
    metadata-derived. ``info`` is ``None`` when the pod is in the sidecar
    metadata but no longer exists on the live API (terminated externally) —
    in that case ``_load_state`` drops the entry from the merged map; callers
    never see an ``info=None`` view.
    """

    metadata: EphemeralMetadata
    info: PodInfo  # always non-None in the merged view (drift entries dropped)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def pod_id(self) -> str:
        return self.metadata.pod_id

    @property
    def issue(self) -> int:
        return self.metadata.issue

    @property
    def gpu_intent(self) -> str:
        return self.metadata.gpu_intent

    @property
    def ttl_days(self) -> int:
        return self.metadata.ttl_days

    @property
    def stopped_at(self) -> str | None:
        return self.metadata.stopped_at

    @property
    def notes(self) -> str:
        return self.metadata.notes

    @property
    def status(self) -> str:
        """Map RunPod ``desiredStatus`` → our 3-state lifecycle.

        ``RUNNING`` → ``running``; ``EXITED`` → ``stopped``; anything else
        (PROVISIONING, FAILED, etc.) → lowercase echo so callers can spot the
        edge case rather than being told a misleading ``running``.
        """
        ds = (self.info.desired_status or "").upper()
        if ds == "RUNNING":
            return "running"
        if ds == "EXITED":
            return "stopped"
        return ds.lower() or "unknown"

    @property
    def host(self) -> str | None:
        return self.info.ssh_host

    @property
    def port(self) -> int | None:
        return self.info.ssh_port

    @property
    def gpu_count(self) -> int:
        return self.info.gpu_count or 0

    @property
    def gpu_type(self) -> str:
        """Short GPU name (H100/H200/A100); falls back to the full GraphQL id."""
        full = self.info.gpu_type_id or ""
        if "H100" in full:
            return "H100"
        if "H200" in full:
            return "H200"
        if "A100" in full:
            return "A100"
        return full

    @property
    def created_at(self) -> str | None:
        return self.info.created_at


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_metadata_file() -> dict[str, EphemeralMetadata]:
    """Read project-side metadata from the JSON sidecar; tolerate missing file."""
    if not EPHEMERAL_STATE.exists():
        return {}
    raw = json.loads(EPHEMERAL_STATE.read_text())
    out: dict[str, EphemeralMetadata] = {}
    known = {f.name for f in EphemeralMetadata.__dataclass_fields__.values()}
    # Forward-compat: silently drop unknown keys (and legacy state-of-pod
    # fields like host/port/status that older sidecar versions wrote).
    for name, payload in raw.get("pods", {}).items():
        clean = {k: v for k, v in payload.items() if k in known}
        clean.setdefault("name", name)
        # Tolerate sidecars that lack pod_id / issue (corrupted): skip.
        if "pod_id" not in clean or "issue" not in clean:
            continue
        out[name] = EphemeralMetadata(**clean)
    return out


def _write_metadata_file(metadata: dict[str, EphemeralMetadata]) -> None:
    """Persist metadata-only fields to the JSON sidecar.

    State-of-pod fields (status, host, port, gpu_count, gpu_type, created_at)
    are NEVER written — they are re-fetched from the live API on every read.
    """
    payload = {
        "version": 2,  # bumped from 1 when the schema went metadata-only
        "updated_at": _now(),
        "pods": {
            name: {
                "name": m.name,
                "pod_id": m.pod_id,
                "issue": m.issue,
                "gpu_intent": m.gpu_intent,
                "ttl_days": m.ttl_days,
                "stopped_at": m.stopped_at,
                "notes": m.notes,
                "manual_override": m.manual_override,
                "extra": m.extra,
            }
            for name, m in metadata.items()
        },
    }
    EPHEMERAL_STATE.write_text(json.dumps(payload, indent=2) + "\n")


# Pod-name prefixes our project manages. ``pod-`` is the canonical prefix
# (April 2026 rename); ``epm-issue-`` is the legacy prefix and is still
# recognized so in-flight pods provisioned before the rename keep working.
# Remove ``epm-issue-`` from this list once no live pods carry it.
_MANAGED_PREFIXES: tuple[str, ...] = ("pod-", "epm-issue-")


def _is_managed_pod(pod: PodInfo) -> bool:
    """True if this pod is one our project manages."""
    return any(pod.name.startswith(p) for p in _MANAGED_PREFIXES)


# Back-compat alias: external callers historically imported this name.
_is_epm_pod = _is_managed_pod


def _issue_from_pod_name(name: str) -> int | None:
    """Best-effort: extract the issue number from a managed pod name.

    Accepts both the canonical ``pod-<N>`` and legacy ``epm-issue-<N>``
    prefixes.
    """
    for prefix in _MANAGED_PREFIXES:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            try:
                return int(suffix)
            except ValueError:
                return None
    return None


def _load_state() -> dict[str, EphemeralPod]:
    """Merge project-side metadata + live API state into a unified view.

    Three branches per pod:

    1. **Metadata + API** — full :class:`EphemeralPod` view. Status/host/port
       always come from API.
    2. **Metadata only (no live API match)** — user terminated externally.
       Drop from the in-memory view. JSON is NOT re-written here; the next
       ``_save_state`` call (after a successful command) will reconcile.
    3. **API only (no metadata)** — unmanaged ``pod-*`` / ``epm-issue-*`` pod
       (provisioned outside this script). Synthesize default metadata
       (gpu_intent="custom", ttl_days=DEFAULT, stopped_at=None, notes="").

    The live API call is REQUIRED — there is no offline fallback. If the API
    is unreachable, callers see :class:`runpod_api.RunPodError` propagate so
    they can surface a clear error message rather than serving stale data.
    """
    metadata = _read_metadata_file()
    live_pods = list_team_pods()
    live_by_name = {p.name: p for p in live_pods if _is_managed_pod(p)}

    merged: dict[str, EphemeralPod] = {}
    drift_repaired: dict[str, tuple[str, str]] = {}  # name -> (stale, live)
    override_protected: dict[str, tuple[str, str]] = {}  # name -> (kept_id, live_id)

    # Branch 1 + 2: walk metadata; intersect with live API.
    for name, meta in metadata.items():
        live = live_by_name.get(name)
        if live is None:
            # Branch 2: in JSON but not in API — terminated externally. Skip.
            continue
        if meta.pod_id != live.pod_id:
            if meta.manual_override:
                # Manual override is active — the user asserted via
                # ``pod_config.cmd_update`` that the recorded pod_id /
                # host / port are correct. The live API matched a
                # DIFFERENT RunPod entry by name (name collisions happen
                # when a pod is migrated and the old one is recreated
                # under the same label). Do NOT silently repoint the
                # sidecar. Synthesize a PodInfo with the live API state
                # we WOULD have shown for completeness, but keep the
                # caller's recorded pod_id intact. The pod's host/port
                # for SSH come from ``pods.conf`` (the SoT for ``--sync``)
                # and are not consulted from this view.
                override_protected[name] = (meta.pod_id, live.pod_id)
                merged[name] = EphemeralPod(metadata=meta, info=live)
                continue
            # Sidecar drift: the live API's pod_id disagrees with what we
            # recorded. The RunPod API is authoritative for pod_id (state-of-
            # pod, not project-side metadata). Repair the in-memory view and
            # the on-disk JSON so subsequent terminate/stop/resume calls
            # target the right pod. Without this, `task.py terminate` etc.
            # silently send the wrong id and the API returns POD_NOT_FOUND.
            drift_repaired[name] = (meta.pod_id, live.pod_id)
            meta = replace(meta, pod_id=live.pod_id)
        merged[name] = EphemeralPod(metadata=meta, info=live)

    if drift_repaired:
        # Write-through fix so next read is clean.
        all_meta = _read_metadata_file()
        for name, (_stale, live_id) in drift_repaired.items():
            if name in all_meta:
                all_meta[name] = replace(all_meta[name], pod_id=live_id)
        _write_metadata_file(all_meta)
        for name, (stale, live_id) in drift_repaired.items():
            print(
                f"[pod_lifecycle] WARN: sidecar pod_id for {name} drifted "
                f"({stale} -> {live_id}); repaired pods_ephemeral.json.",
                file=sys.stderr,
            )

    if override_protected:
        for name, (kept_id, live_id) in override_protected.items():
            print(
                f"[pod_lifecycle] WARN: live API has a different pod_id for "
                f"{name} ({live_id}) than the sidecar ({kept_id}); keeping "
                f"the sidecar because manual_override=True. Clear with "
                f"`pod.py config --clear-override {name}` if the live pod is "
                f"the right one.",
                file=sys.stderr,
            )

    # Branch 3: walk live API entries that are unmanaged.
    for name, live in live_by_name.items():
        if name in merged:
            continue
        issue = _issue_from_pod_name(name)
        if issue is None:
            continue
        synthetic = EphemeralMetadata(
            name=name,
            pod_id=live.pod_id,
            issue=issue,
            gpu_intent="custom",
            ttl_days=DEFAULT_TTL_DAYS,
            stopped_at=None,
            notes="",
        )
        merged[name] = EphemeralPod(metadata=synthetic, info=live)

    return merged


def _save_state(state: dict[str, EphemeralPod]) -> None:
    """Persist metadata-only view from the merged state map.

    Writes only the project-side metadata fields. State-of-pod fields are
    re-fetched on next read.
    """
    metadata = {name: pod.metadata for name, pod in state.items()}
    _write_metadata_file(metadata)


# ─── pods.conf side effects ──────────────────────────────────────────────────


def _label_for_issue(issue: int) -> str:
    return f"thomas-pod-{issue}"


def _canonical_pod_name(issue: int) -> str:
    """The canonical name for a fresh provision: ``pod-<N>``."""
    return f"pod-{issue}"


def _find_pod_in_state(state: dict[str, EphemeralPod], issue: int) -> EphemeralPod | None:
    """Locate a registered pod for ``issue`` regardless of name prefix.

    Searches for the canonical ``pod-<N>`` first, then the legacy
    ``epm-issue-<N>`` (kept around for in-flight pods provisioned before
    the April 2026 rename). Returns ``None`` if neither is registered.
    """
    for candidate in (_canonical_pod_name(issue), f"epm-issue-{issue}"):
        if candidate in state:
            return state[candidate]
    return None


def _upsert_pods_conf(pod: EphemeralPod) -> None:
    """Add or update `pod` in scripts/pods.conf and regenerate downstream configs.

    When ``pod.metadata.manual_override`` is True and an existing row is
    present, the host/port columns are preserved (the user manually set them
    via ``pod_config.cmd_update`` and the live API pod_id may be for a
    different RunPod entry sharing the same name). gpus / gpu_type / label
    are still refreshed since they are not user-overrideable via ``--update``.
    """
    rows = parse_pods_conf()
    existing = next((p for p in rows if p.name == pod.name), None)
    if pod.host is None or pod.port is None:
        # Nothing to write yet — only happens during transient provisioning.
        return
    if existing:
        if pod.metadata.manual_override and (
            existing.host != pod.host or existing.port != pod.port
        ):
            print(
                f"[pod_lifecycle] WARN: refusing to overwrite manual host/port "
                f"for {pod.name} in pods.conf "
                f"(kept {existing.host}:{existing.port}; API would have written "
                f"{pod.host}:{pod.port}). Clear with "
                f"`pod.py config --clear-override {pod.name}` if the API is right.",
                file=sys.stderr,
            )
        else:
            existing.host = pod.host
            existing.port = pod.port
        existing.gpus = pod.gpu_count
        existing.gpu_type = pod.gpu_type
        existing.label = _label_for_issue(pod.issue)
    else:
        rows.append(
            Pod(
                name=pod.name,
                host=pod.host,
                port=pod.port,
                gpus=pod.gpu_count,
                gpu_type=pod.gpu_type,
                label=_label_for_issue(pod.issue),
            )
        )
    write_pods_conf(rows)
    cmd_sync(rows)


def _remove_from_pods_conf(name: str) -> None:
    rows = parse_pods_conf()
    rows = [p for p in rows if p.name != name]
    write_pods_conf(rows)
    cmd_sync(rows)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _resolve_spec(
    intent: str | None, gpu_type: str | None, gpu_count: int | None
) -> tuple[GpuSpec, str]:
    """Pick a GpuSpec. Returns (spec, intent_label).

    Explicit --gpu-type/--gpu-count override the intent table. If both are given
    AND --intent, we use the explicit values but record the intent for posterity.
    """
    if gpu_type and gpu_count:
        spec = GpuSpec(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            rationale=f"explicit override (--gpu-type {gpu_type} --gpu-count {gpu_count})",
        )
        return spec, intent or "custom"
    if intent:
        spec = resolve_intent(intent)
        return spec, intent
    raise SystemExit(
        "Must pass either --intent <name> OR both --gpu-type and --gpu-count.\n"
        "Run `python scripts/pod.py provision --list-intents` to see options."
    )


def _bootstrap(pod_name: str) -> int:
    """Run the existing bootstrap_pod.sh against a managed pod entry."""
    print(f"\nRunning bootstrap on {pod_name}...")
    return subprocess.call(
        ["bash", str(BOOTSTRAP_SCRIPT), pod_name],
        cwd=str(PROJECT_ROOT),
    )


# Phrases RunPod uses when a stopped pod can't be resumed because its former
# host has no free GPUs. The mutation returns null (→ ``podResume returned
# null``) or surfaces one of these in the GraphQL ``errors`` payload.
_SUPPLY_CONSTRAINT_MARKERS: tuple[str, ...] = (
    "podresume returned null",
    "not enough free gpu",
    "no free gpu",
    "supply_constraint",
    "supplyconstraint",
    "insufficient capacity",
    "no longer any instances available",
)


def _is_supply_constraint(exc: Exception) -> bool:
    """True if a resume failure is a capacity problem (vs a real error).

    Resume never relocates a pod (its volume is pinned to the original host), so
    a capacity failure is NOT something we can retry around — it needs a fresh
    provision. We detect it so :func:`cmd_resume` can emit an actionable message
    instead of a bare stack trace.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _SUPPLY_CONSTRAINT_MARKERS)


def ssh_preflight(
    host: str | None,
    port: int | None,
    *,
    issue: int | None = None,
    timeout: float = 5.0,
    allow_resume: bool = True,
) -> bool:
    """Check that ``host:port`` accepts a TCP connection before a batch of
    remote ops, so we don't hammer a dead endpoint (issue #12).

    On the first failure, if ``allow_resume`` and an ``issue`` are given, attempt
    ``pod.py resume --issue <N>`` exactly ONCE (it re-syncs pods.conf / SSH /
    MCP and yields a fresh host:port), then re-read the live endpoint and
    re-check. Returns True if the endpoint is reachable (possibly after the
    resume), False otherwise. Never raises on an unreachable endpoint — the
    boolean IS the signal so callers can decide whether to proceed or abort.

    ``host``/``port`` of ``None`` count as unreachable (a pod with no public
    mapping yet).
    """
    if _tcp_open(host, port, timeout):
        return True

    where = f"{host}:{port}" if host and port else "(no public mapping)"
    print(
        f"[pod_lifecycle] SSH preflight: {where} is not accepting connections.",
        file=sys.stderr,
    )

    if not (allow_resume and issue is not None):
        print(
            "[pod_lifecycle] SSH preflight FAILED — endpoint unreachable and "
            "no resume attempted. Check the pod status with "
            f"`python scripts/pod.py list-ephemeral{f' --issue {issue}' if issue else ''}`.",
            file=sys.stderr,
        )
        return False

    print(
        f"[pod_lifecycle] Attempting one `pod.py resume --issue {issue}` to "
        "refresh the endpoint...",
        file=sys.stderr,
    )
    rc = _run_resume_subprocess(issue)
    if rc != 0:
        print(
            f"[pod_lifecycle] SSH preflight FAILED — resume exited {rc}. "
            "The pod may be terminated or out of capacity; provision a fresh "
            f"pod with `python scripts/pod.py provision --issue {issue} ...`.",
            file=sys.stderr,
        )
        return False

    # Re-read the freshly-resumed endpoint from the live API and re-check once.
    new_host, new_port = _live_ssh_endpoint(issue)
    if _tcp_open(new_host, new_port, timeout):
        print(
            f"[pod_lifecycle] SSH preflight recovered after resume: "
            f"{new_host}:{new_port} is reachable.",
            file=sys.stderr,
        )
        return True

    print(
        "[pod_lifecycle] SSH preflight FAILED — still unreachable after resume. "
        f"Provision a fresh pod with `python scripts/pod.py provision --issue {issue} ...`.",
        file=sys.stderr,
    )
    return False


def _tcp_open(host: str | None, port: int | None, timeout: float) -> bool:
    """True if a TCP connection to ``host:port`` opens within ``timeout`` secs.

    A missing host/port counts as closed. Pure connectivity probe — does not
    speak SSH, just confirms the endpoint is listening so we stop hammering a
    dead IP.
    """
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        # Connection refused / timed out / DNS failure — endpoint is down.
        return False


def _run_resume_subprocess(issue: int) -> int:
    """Run ``pod.py resume --issue <N>`` in a child process; return its exit code.

    Spawned as a subprocess (not an in-process ``cmd_resume`` call) so the
    resume's pods.conf / SSH / MCP regeneration side effects run exactly as they
    would from the CLI, and a SystemExit inside resume doesn't unwind the
    caller's batch.
    """
    return subprocess.call(
        [sys.executable, str(SCRIPT_DIR / "pod.py"), "resume", "--issue", str(issue)],
        cwd=str(PROJECT_ROOT),
    )


def _live_ssh_endpoint(issue: int) -> tuple[str | None, int | None]:
    """Re-read the live host/port for ``issue`` from the merged API state.

    Returns ``(None, None)`` if the pod isn't in the merged view (terminated)
    or has no public SSH mapping yet.
    """
    state = _load_state()
    pod = _find_pod_in_state(state, issue)
    if pod is None:
        return None, None
    return pod.host, pod.port


# ─── commands ────────────────────────────────────────────────────────────────


def _warn_on_lifecycle_escapes(live_pods: list[PodInfo]) -> None:
    """Print a loud warning if any pods on the team account are invisible to
    the lifecycle (non-managed names) or are stale EXITED pods accruing volume
    charges. Defense in depth against the 2026-05 incident where dispatcher
    scripts spun up ~20 pods with custom names and the lifecycle/audit never
    saw them — RunPod's billing email surfaced them weeks later.

    Never blocks; informational only.
    """
    escapes: list[PodInfo] = []
    stale: list[PodInfo] = []
    now = dt.datetime.now(dt.UTC)
    for p in live_pods:
        if not _is_managed_pod(p):
            escapes.append(p)
        if p.desired_status == "EXITED" and p.created_at:
            try:
                created = dt.datetime.fromisoformat(p.created_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if (now - created).total_seconds() > 24 * 3600:
                stale.append(p)
    if not escapes and not stale:
        return
    print(
        "\n[pod_lifecycle] WARN: lifecycle audit found pods not owned by /issue Step 8:",
        file=sys.stderr,
    )
    for p in escapes:
        print(
            f"  unmanaged-name  {p.pod_id}  {p.desired_status:8}  {p.name!r}",
            file=sys.stderr,
        )
    for p in stale:
        if p in escapes:
            continue
        print(f"  stale-EXITED    {p.pod_id}  age>24h        {p.name!r}", file=sys.stderr)
    print(
        "  Run `python scripts/pod.py audit-stale --terminate-stale` to clean up.\n",
        file=sys.stderr,
    )


def cmd_provision(args: argparse.Namespace) -> None:
    """Create a fresh pod for issue #N, wait for SSH, register it, bootstrap it."""
    if args.list_intents:
        print(list_intents())
        return

    if args.issue is None:
        raise SystemExit("--issue <N> is required")

    name = _canonical_pod_name(args.issue)
    legacy = f"epm-issue-{args.issue}"

    # Idempotency: refuse if a non-EXITED pod for this issue exists under
    # EITHER the canonical or the legacy prefix.
    live_pods = list_team_pods()
    live_by_name = {p.name: p for p in live_pods if _is_managed_pod(p)}

    # Pre-flight: surface any pods the lifecycle is blind to (non-managed
    # names) so the user notices accumulating charges before adding another
    # pod. Don't block — just warn loudly.
    _warn_on_lifecycle_escapes(live_pods)
    for candidate in (name, legacy):
        if candidate in live_by_name and live_by_name[candidate].desired_status != "EXITED":
            existing = live_by_name[candidate]
            print(
                f"Pod {candidate} already exists "
                f"(status={existing.desired_status}, id={existing.pod_id}).\n"
                f"Use `pod.py resume --issue {args.issue}` to bring it back, "
                f"or `pod.py terminate --issue {args.issue}` first if you want a fresh one."
            )
            sys.exit(1)

    spec, intent_label = _resolve_spec(args.intent, args.gpu_type, args.gpu_count)
    print(f"Provisioning {name}: {spec.gpu_count}x {spec.gpu_type}  ({intent_label})")
    print(f"  Why: {spec.rationale}")

    if args.dry_run:
        print("\n[dry-run] Would call create_pod and wait for SSH; no API call made.")
        return

    info = create_pod(
        name=name,
        gpu_type=spec.gpu_type,
        gpu_count=spec.gpu_count,
        volume_gb=args.volume_gb,
        container_disk_gb=args.container_disk_gb,
    )
    print(f"  Created pod {info.pod_id} — waiting for SSH (up to 10 min)...")

    ready = wait_for_ssh(info.pod_id, timeout=600)
    print(f"  SSH ready at {ready.ssh_host}:{ready.ssh_port}")

    metadata = _read_metadata_file()
    metadata[name] = EphemeralMetadata(
        name=name,
        pod_id=info.pod_id,
        issue=args.issue,
        gpu_intent=intent_label,
        ttl_days=args.ttl_days,
        stopped_at=None,
        notes="",
    )
    _write_metadata_file(metadata)

    pod = EphemeralPod(metadata=metadata[name], info=ready)
    _upsert_pods_conf(pod)
    print("  Registered in pods.conf and pods_ephemeral.json")

    if args.no_bootstrap:
        print("\nSkipping bootstrap (--no-bootstrap). Run later with:")
        print(f"  python scripts/pod.py bootstrap {name}")
        return

    rc = _bootstrap(name)
    if rc != 0:
        print(
            f"\nBootstrap exited with code {rc}. Pod is up but not experiment-ready.\n"
            f"Investigate, then either re-run `bash scripts/bootstrap_pod.sh {name}` or\n"
            f"`python scripts/pod.py terminate --issue {args.issue}` to discard.",
            file=sys.stderr,
        )
        sys.exit(rc)

    print(f"\nDone. SSH with: ssh {name}")


def cmd_stop(args: argparse.Namespace) -> None:
    """Pause the pod for issue #N. Volume preserved; IP released."""
    state = _load_state()
    pod = _find_pod_in_state(state, args.issue)
    if pod is None:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    name = pod.name
    if pod.status == "stopped":
        print(f"{name} already stopped.")
        return
    if pod.status not in {"running"}:
        raise SystemExit(f"{name} has live status {pod.info.desired_status!r}; refuse to stop.")

    print(f"Stopping {name} (pod_id={pod.pod_id})...")
    if args.dry_run:
        print("[dry-run] Would call stop_pod.")
        return
    stop_pod(pod.pod_id)
    # Update metadata-only fields. Status/host/port are re-fetched on next read.
    # Synthetic-metadata pods (Branch 3 of _load_state) are promoted to disk
    # here so the stopped_at timestamp persists.
    metadata = _read_metadata_file()
    if name not in metadata:
        metadata[name] = pod.metadata
    metadata[name].stopped_at = _now()
    _write_metadata_file(metadata)
    print(
        f"  Stopped. Will auto-terminate after {pod.ttl_days} days idle "
        f"(stopped_at={metadata[name].stopped_at})."
    )


def cmd_resume(args: argparse.Namespace) -> None:
    """Bring a stopped pod back. New IP, same volume."""
    state = _load_state()
    pod = _find_pod_in_state(state, args.issue)
    if pod is None:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    name = pod.name
    if pod.status == "running":
        print(f"{name} is already running.")
        return

    print(f"Resuming {name} (pod_id={pod.pod_id}, gpuCount={pod.gpu_count})...")
    if args.dry_run:
        print("[dry-run] Would call resume_pod and wait for SSH.")
        return
    try:
        resume_pod(pod.pod_id, pod.gpu_count)
    except RunPodError as exc:
        if _is_supply_constraint(exc):
            # Resume never relocates — the stopped pod's volume is pinned to its
            # original host. If that host has no free GPUs we CANNOT retry around
            # it; the user must provision a fresh pod (losing this volume) or
            # wait for capacity. Do NOT auto-terminate or auto-provision here —
            # that would silently destroy the stopped pod's volume.
            raise SystemExit(
                f"Cannot resume {name}: its former host has no free GPUs "
                f"(supply constraint). Resume never relocates a pod, so this "
                f"can't be retried. Either wait for capacity to free up and "
                f"re-run `pod.py resume --issue {args.issue}`, or provision a "
                f"FRESH pod with `python scripts/pod.py provision --issue "
                f"{args.issue} --intent <intent>` (this loses the stopped pod's "
                f"volume — terminate it first with `pod.py terminate --issue "
                f"{args.issue} --yes` if you want it gone).\n  Underlying error: {exc}"
            ) from exc
        raise
    ready = wait_for_ssh(pod.pod_id, timeout=600)

    # Clear our project-side stopped_at marker; status/host/port refresh on read.
    # Synthetic-metadata pods (Branch 3 of _load_state) are promoted to disk
    # here so pods.conf gets refreshed and future commands see the metadata.
    metadata = _read_metadata_file()
    if name not in metadata:
        metadata[name] = pod.metadata
    metadata[name].stopped_at = None
    _write_metadata_file(metadata)

    refreshed = EphemeralPod(metadata=metadata[name], info=ready)
    _upsert_pods_conf(refreshed)
    print(f"  SSH ready at {refreshed.host}:{refreshed.port}")
    print(f"  pods.conf updated. Connect: ssh {name}")


def cmd_terminate(args: argparse.Namespace) -> None:
    """Destroy the pod for issue #N. Volume gone."""
    state = _load_state()
    pod = _find_pod_in_state(state, args.issue)
    if pod is None:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    name = pod.name

    print(f"Terminating {name} (pod_id={pod.pod_id})...")
    if not args.yes and not args.dry_run:
        confirm = input("  This DESTROYS the volume. Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    if args.dry_run:
        print("[dry-run] Would call terminate_pod.")
        return
    terminate_pod(pod.pod_id)
    # Drop the entry from metadata; the API will no longer return this pod.
    metadata = _read_metadata_file()
    metadata.pop(name, None)
    _write_metadata_file(metadata)
    _remove_from_pods_conf(name)
    print("  Terminated. Removed from pods.conf and pods_ephemeral.json.")


def cmd_list_ephemeral(args: argparse.Namespace) -> None:
    """List ephemeral pods. State-of-pod is always live (API-derived).

    ``--issue <N>`` filters to a single issue. ``--refresh`` is now a no-op
    deprecation alias because the live API is queried on every invocation.
    """
    if args.refresh:
        print(
            "  NOTE: --refresh is deprecated; the live RunPod API is now queried "
            "on every list-ephemeral invocation, so reconciliation is automatic.",
            file=sys.stderr,
        )

    state = _load_state()
    if args.issue is not None:
        state = {k: v for k, v in state.items() if v.issue == args.issue}

    if not state:
        if args.issue is not None:
            print(f"No ephemeral pod recorded for issue #{args.issue}.")
        else:
            print("No ephemeral pods recorded.")
        return

    header = (
        f"{'NAME':<22} {'ISSUE':<6} {'STATUS':<11} {'GPUS':<10} {'AGE':<14} {'INTENT':<10} POD_ID"
    )
    print(header)
    print("-" * len(header))
    now = dt.datetime.now(dt.UTC)
    for pod in sorted(state.values(), key=lambda p: -p.issue):
        age = ""
        if pod.created_at:
            try:
                created = dt.datetime.fromisoformat(pod.created_at.replace("Z", "+00:00"))
                age = f"{(now - created).days}d"
            except ValueError:
                age = ""
        gpu_label = f"{pod.gpu_count}x{pod.gpu_type}"
        print(
            f"{pod.name:<22} #{pod.issue:<5} {pod.status:<11} "
            f"{gpu_label:<10} {age:<14} {pod.gpu_intent:<10} {pod.pod_id}"
        )


# ─── argparse plumbing ───────────────────────────────────────────────────────


def _parser_provision(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("provision", help="Create a fresh pod for an issue and bootstrap it")
    p.add_argument("--issue", type=int, help="GitHub issue number (used as pod name)")
    p.add_argument(
        "--intent",
        help="Workload intent (lora-7b, ft-7b, eval, inf-70b, ft-70b, debug). "
        "Run with --list-intents to see all.",
    )
    p.add_argument("--gpu-type", help="Override GPU type (H100|H200|A100)")
    p.add_argument("--gpu-count", type=int, help="Override GPU count")
    p.add_argument("--volume-gb", type=int, default=200, help="Persistent volume size (GB)")
    p.add_argument(
        "--container-disk-gb",
        type=int,
        default=50,
        help="Container overlay disk (GB) — held for caches that bypass /workspace",
    )
    p.add_argument(
        "--ttl-days", type=int, default=DEFAULT_TTL_DAYS, help="Idle TTL before termination"
    )
    p.add_argument("--no-bootstrap", action="store_true", help="Skip running bootstrap_pod.sh")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-intents", action="store_true", help="Show known intent table and exit")
    p.set_defaults(func=cmd_provision)


def _parser_stop(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stop", help="Pause an issue's pod (preserves volume)")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_stop)


def _parser_resume(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("resume", help="Bring a stopped pod back; refresh IP")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_resume)


def _parser_terminate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("terminate", help="Destroy an issue's pod (volume goes too)")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_terminate)


def _parser_list(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("list-ephemeral", help="Show all ephemeral pods + lifecycle state")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="(deprecated; the live API is now queried on every invocation)",
    )
    p.add_argument("--issue", type=int, help="Filter to a single issue number")
    p.set_defaults(func=cmd_list_ephemeral)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pod_lifecycle",
        description="Ephemeral RunPod lifecycle: provision/stop/resume/terminate per GitHub issue.",
    )
    sub = parser.add_subparsers(dest="cmd")
    _parser_provision(sub)
    _parser_stop(sub)
    _parser_resume(sub)
    _parser_terminate(sub)
    _parser_list(sub)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
