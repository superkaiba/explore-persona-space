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
import contextlib
import datetime as dt
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn

# Same package — sibling modules.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Module ref needed for CALL-TIME global reads (resolve_live_pods_ephemeral,
# _atomic_write_text) — no import cycle: pod_config never imports this module
# (see the comment above pod_config.PODS_EPHEMERAL_SEED).
import pod_config  # noqa: E402
from gpu_heuristics import (  # noqa: E402
    GpuSpec,
    list_intents,
    resolve_cpu_instance_caps,
    resolve_cpu_intent,
    resolve_intent,
)
from pod_config import (  # noqa: E402
    PODS_EPHEMERAL_JSON as _PODS_EPHEMERAL_JSON_MAIN,
)
from pod_config import (  # noqa: E402
    Pod,
    cmd_sync,
    locked_pods_conf,
    parse_pods_conf,
    write_pods_conf,
)
from runpod_api import (  # noqa: E402
    DEFAULT_CONTAINER_DISK_GB,
    DEFAULT_CPU_VOLUME_GB,
    PodInfo,
    RunPodError,
    RunPodInsufficientBalanceError,
    RunPodNoCapacityError,
    create_cpu_pod,
    create_pod,
    current_account_hourly_burn,
    estimate_pod_hourly_rate,
    get_pod,
    list_team_pods,
    resume_pod,
    stop_pod,
    terminate_pod,
    wait_for_ssh,
)

PROJECT_ROOT = SCRIPT_DIR.parent
# Re-export pod_config's MAIN-repo-resolved path so this module's writers
# share the SAME on-disk file as pod_config's readers (e.g. cmd_update's
# manual_override flip). See pod_config._main_repo_scripts_dir for the
# motivating incident — task #500, 2026-06-05.
EPHEMERAL_STATE = _PODS_EPHEMERAL_JSON_MAIN
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


def _resolve_state_path() -> Path:
    """LIVE pods_ephemeral.json path (task #1183; mirrors #821 pods.conf).

    Honors a test's monkeypatched ``pod_lifecycle.EPHEMERAL_STATE`` — the
    module global is read at CALL time (the #821 lazy-resolution trick), so
    ``monkeypatch.setattr(pod_lifecycle, "EPHEMERAL_STATE", tmp)`` is honored
    by every read/write in the process. On the unpatched (production) path,
    delegates to ``pod_config.resolve_live_pods_ephemeral()`` — the LIVE copy
    at ``<git-common-dir>/eps/pods_ephemeral.json``, seed-migrated on first
    use.
    """
    if EPHEMERAL_STATE != _PODS_EPHEMERAL_JSON_MAIN:
        return EPHEMERAL_STATE
    return pod_config.resolve_live_pods_ephemeral()


def _metadata_lock() -> contextlib.AbstractContextManager[None]:
    """Lock context for a sidecar read-modify-write (task #1183).

    Returns ``locked_pods_conf()`` on the production path — reentrant, so
    nesting at an already-locked call site (``cmd_update`` →
    ``_set_manual_override``; a locked call site → ``_write_metadata_file``)
    is free. The monkeypatch fast path is checked BEFORE the lock: a test
    with a patched ``EPHEMERAL_STATE`` gets a ``nullcontext`` and never
    touches the real shared ``scripts/.pods.conf.lock`` (no test↔fleet
    cross-talk, no test hang behind a network-holding writer).
    """
    if EPHEMERAL_STATE != _PODS_EPHEMERAL_JSON_MAIN:
        return contextlib.nullcontext()
    return locked_pods_conf()


def _read_metadata_file() -> dict[str, EphemeralMetadata]:
    """Read project-side metadata from the JSON sidecar; tolerate missing file."""
    path = _resolve_state_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
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


def _write_metadata_file(
    metadata: dict[str, EphemeralMetadata],
    *,
    allow_remove: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Persist metadata-only fields to the JSON sidecar (atomic + guarded).

    State-of-pod fields (status, host, port, gpu_count, gpu_type, created_at)
    are NEVER written — they are re-fetched from the live API on every read.

    Task #1183 hardening (mirrors #821's ``write_pods_conf``):

    - **Atomic:** written via ``pod_config._atomic_write_text`` (same-dir tmp
      + ``os.replace``), so no reader ever observes a torn file.
    - **Structural never-drop guard:** an incoming dict that DROPS on-disk
      entries not named in ``allow_remove`` gets them re-added with a loud
      WARN naming the opt-out. Unlike ``write_pods_conf`` there is NO
      live-API call — the JSON's legitimate droppers (terminate /
      stale-clear / ``_save_state``'s reconcile) all pass ``allow_remove``
      explicitly, and terminate flows must never depend on network access.
      A guard-read failure (corrupt on-disk JSON) WARNs "guard skipped" and
      proceeds — the atomic write below repairs the file.
    - **Writer-internal lock:** the production path acquires
      ``locked_pods_conf`` INSIDE the writer (reentrant — a call site
      already holding it is free), so any future unwrapped call site is
      protected by construction. The monkeypatch fast path is checked
      BEFORE the lock (see ``_metadata_lock``).
    """
    with _metadata_lock():
        path = _resolve_state_path()
        try:
            on_disk = _read_metadata_file()
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[pod_lifecycle] WARN: never-drop guard skipped — cannot read "
                f"{path}: {exc}; the atomic write below repairs it.",
                file=sys.stderr,
            )
            on_disk = {}
        dropped = set(on_disk) - set(metadata) - set(allow_remove)
        if dropped:
            metadata = dict(metadata)
            for name in sorted(dropped):
                print(
                    f"[pod_lifecycle] WARN: refusing to drop sidecar entry {name} "
                    f"(not in allow_remove); re-adding. Pass "
                    f"allow_remove={{{name!r}}} if the removal is intentional.",
                    file=sys.stderr,
                )
                metadata[name] = on_disk[name]
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
        # 0o644 mirrors today's plain write_text mode — the JSON holds no
        # secrets and external read-only tooling may consult it.
        pod_config._atomic_write_text(
            path, json.dumps(payload, indent=2) + "\n", default_mode=0o644
        )


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
        # Write-through fix so next read is clean. Re-read + replace + write
        # form one contiguous RMW under the lock (task #1183); the live-API
        # call above stays OUTSIDE the lock.
        with _metadata_lock():
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

    Reconcile semantics (task #1183): entries on disk that are ABSENT from
    ``state`` were dropped by ``_load_state`` Branch 2 — i.e. validated
    against the live API as terminated — so they are passed as
    ``allow_remove`` (the never-drop guard must not resurrect them). The
    ``on_disk`` read happens INSIDE the same lock span as the write so a
    concurrent provision registering a pod between our read and our write
    cannot land in ``allow_remove`` and get reconciled away.
    """
    metadata = {name: pod.metadata for name, pod in state.items()}
    with _metadata_lock():
        on_disk = _read_metadata_file()
        _write_metadata_file(metadata, allow_remove=frozenset(on_disk) - frozenset(metadata))


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

    The whole parse → mutate → write → ``cmd_sync`` sequence holds
    ``pod_config.locked_pods_conf`` so concurrent ``/issue`` sessions
    upserting their own pods cannot lose-update each other's rows or
    regenerate ``~/.ssh/config`` from a stale view (task #488 incident,
    2026-06-05).
    """
    if pod.host is None or pod.port is None:
        # Nothing to write yet — only happens during transient provisioning.
        return
    with locked_pods_conf():
        rows = parse_pods_conf()
        existing = next((p for p in rows if p.name == pod.name), None)
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
        cmd_sync()


def _remove_from_pods_conf(name: str) -> None:
    """Remove ``name``'s row from ``pods.conf`` (if present) and regenerate
    downstream configs.

    Locked the same way as :func:`_upsert_pods_conf`: a concurrent upsert
    must not be able to read a pre-remove snapshot, write its own row, and
    re-add the removed entry.

    Task #821: this is THE legitimate remove path for ``pods.conf``.
    ``write_pods_conf`` grew a never-drop-RUNNING guard that consults the
    live RunPod API on any diff, so an unaudited drop of a RUNNING pod is
    refused (see ``pod_config.write_pods_conf`` docstring). Because this
    call is invoked exclusively when the caller has ALREADY decided the
    pod should leave the registry (``pod.py terminate``, orchestrator
    cleanup, or a stale-entry sweep), we pass ``allow_remove={name}`` to
    bypass the guard AND the network call — a terminate flow must never
    depend on the RunPod API being reachable.
    """
    with locked_pods_conf():
        rows = parse_pods_conf()
        rows = [p for p in rows if p.name != name]
        write_pods_conf(rows, allow_remove=frozenset({name}))
        cmd_sync()


# ─── helpers ─────────────────────────────────────────────────────────────────


def _resolve_spec(
    intent: str | None, gpu_type: str | None, gpu_count: int | None
) -> tuple[GpuSpec, str]:
    """Pick a GpuSpec. Returns (spec, intent_label).

    Explicit --gpu-type/--gpu-count override the intent table. If both are given
    AND --intent, we use the explicit values but record the intent for posterity.
    If exactly ONE override flag is given alongside --intent, that field is merged
    over the intent's default (e.g. --intent eval --gpu-count 4 → H100 x4) — never
    silently dropped (#531: `--intent eval --gpu-count 4` provisioned 1x H100).
    A single override flag WITHOUT --intent fails loud: there is no default to
    fill the missing field from.
    """
    if gpu_type and gpu_count:
        spec = GpuSpec(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            rationale=f"explicit override (--gpu-type {gpu_type} --gpu-count {gpu_count})",
        )
        return spec, intent or "custom"
    if intent:
        base = resolve_intent(intent)
        if gpu_type or gpu_count:
            override = f"--gpu-type {gpu_type}" if gpu_type else f"--gpu-count {gpu_count}"
            spec = GpuSpec(
                gpu_type=gpu_type or base.gpu_type,
                gpu_count=gpu_count or base.gpu_count,
                rationale=(
                    f"intent {intent} ({base.gpu_type} x{base.gpu_count}) "
                    f"+ explicit override ({override})"
                ),
            )
            return spec, intent
        return base, intent
    if gpu_type or gpu_count:
        given = "--gpu-type" if gpu_type else "--gpu-count"
        missing = "--gpu-count" if gpu_type else "--gpu-type"
        raise SystemExit(
            f"{given} given without --intent: also pass {missing}, or add --intent <name> "
            "to fill the missing field from the intent table.\n"
            "Run `python scripts/pod.py provision --list-intents` to see options."
        )
    raise SystemExit(
        "Must pass either --intent <name> OR both --gpu-type and --gpu-count.\n"
        "Run `python scripts/pod.py provision --list-intents` to see options."
    )


def _bootstrap(pod_name: str, intent_label: str = "custom") -> int:
    """Run the existing bootstrap_pod.sh against a managed pod entry.

    ``intent_label`` is forwarded as ``POD_INTENT`` env var so bootstrap_pod.sh
    can gate intent-specific install steps (e.g. flash-attn is installed for
    training intents but skipped for ``eval`` / ``debug`` to save ~5-10 min of
    build time on pods that don't need FlashAttention2 kernels).
    """
    print(f"\nRunning bootstrap on {pod_name} (intent={intent_label})...")
    env = os.environ.copy()
    env["POD_INTENT"] = intent_label
    return subprocess.call(
        ["bash", str(BOOTSTRAP_SCRIPT), pod_name],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


# Idempotent uv-restore snippet, executed on a freshly-resumed pod over SSH.
# Mirrors the install + /usr/local/bin symlink logic from
# bootstrap_pod.sh steps 2 and 6: RunPod stop/resume wipes the container
# overlay (everything outside /workspace), so both /root/.local/bin/uv AND
# the /usr/local/bin/uv shim that non-interactive non-login SSH shells need
# (per the feedback_pod_uv_path gap) get destroyed even though the project
# .venv on /workspace survives. Without restoring these the next
# `ssh pod "uv run ..."` (or `python` shim) fails until a human reinstalls.
# Pure shell — no Python deps — so it stays in sync with bootstrap by
# duplicating the exact same commands. Fails loud if the binary is still
# missing afterwards.
_UV_RESTORE_SNIPPET = r"""
set -eu
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Restoring uv (wiped by stop/resume)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
fi
UV_BIN=""
for cand in /root/.local/bin/uv "$HOME/.local/bin/uv"; do
    if [ -x "$cand" ]; then UV_BIN="$cand"; break; fi
done
if [ -z "$UV_BIN" ] || ! command -v uv >/dev/null 2>&1; then
    echo "uv restore FAILED: binary not found after install attempt" >&2
    exit 1
fi
# Re-create the /usr/local/bin/uv + uvx + python shims that
# bootstrap_pod.sh step 6 installs (also wiped by stop/resume).
UV_DIR="$(dirname "$UV_BIN")"
ln -sf "$UV_BIN" /usr/local/bin/uv
if [ -x "$UV_DIR/uvx" ]; then
    ln -sf "$UV_DIR/uvx" /usr/local/bin/uvx
fi
if [ ! -x /usr/local/bin/python ]; then
    cat > /usr/local/bin/python <<"PYEOF"
#!/bin/bash
# Bootstrap-installed shim: run the project venv python via uv.
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space || exit 1
exec uv run python "$@"
PYEOF
    chmod +x /usr/local/bin/python
fi
echo "uv restored: $(uv --version)"
"""


def _restore_uv_on_pod(host: str, port: int) -> None:
    """Ensure uv (+ /usr/local/bin shims) survives a RunPod stop/resume.

    RunPod stop/resume wipes the container overlay (everything outside
    ``/workspace``), destroying ``/root/.local/bin/uv`` and the
    ``/usr/local/bin/{uv,uvx,python}`` shims that
    :file:`bootstrap_pod.sh` step 6 installs for non-interactive
    non-login SSH shells. The project ``.venv`` on ``/workspace`` survives,
    so we only need to re-place the launcher binary + symlinks.

    Runs the same uv-install command bootstrap uses (so the two paths
    stay in sync). Idempotent — no-op when ``uv`` is already present.
    Fails loud (``SystemExit``) if the binary is still missing after the
    install attempt; the alternative is a silent post-resume pod where
    every ``ssh pod "uv run ..."`` mysteriously fails.

    Takes ``host``/``port`` as plain str/int (no shelling out to pods.conf)
    so the caller passes the freshly-resumed endpoint that ``wait_for_ssh``
    just confirmed.
    """
    ssh_target = f"root@{host}"
    ssh_key = str(Path.home() / ".ssh" / "id_ed25519")
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
        "-i",
        ssh_key,
        "-p",
        str(port),
        ssh_target,
        _UV_RESTORE_SNIPPET,
    ]
    print("  Restoring uv on resumed pod (stop/resume wipes container overlay)...")
    rc = subprocess.call(ssh_cmd)
    if rc != 0:
        raise SystemExit(
            f"uv restore on {host}:{port} FAILED (ssh exit {rc}). The resumed pod "
            f"is missing /root/.local/bin/uv and/or the /usr/local/bin/uv shim; "
            f'every `ssh pod "uv run ..."` will fail until uv is reinstalled. '
            f"Re-run `python scripts/pod.py bootstrap <pod-name>` to repair."
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


# ─── wait-for-capacity (budget-bounded attempts, unbounded intent) ──────────
#
# Policy layer that wraps the one-shot ``create_pod`` primitive in an UNBOUNDED
# retry loop keyed on the two transient-while-idle classes:
# ``RunPodNoCapacityError`` (no host has free GPUs) and
# ``RunPodInsufficientBalanceError`` (projected account $/hr > console cap).
# Triggered when the autonomous session asks for a pod and either signal fires
# — the correct behavior in autonomous mode is "the experiment should start
# when capacity / $/hr headroom is available," NOT a park-for-user. Both
# conditions clear without us spending anything: the pod is unprovisioned, so
# no $/hr is being burned; the moment any sibling pod stops/terminates (for
# balance) or a host frees a GPU (for capacity), the next retry succeeds.
#
# Hard rules baked in:
#
# 1. BOUNDED ATTEMPT, UNBOUNDED INTENT. The wait is conceptually unbounded
#    (autonomous sessions should wait, not park — Thomas's explicit design
#    choice), but each PROCESS-level attempt is capped at
#    :data:`WAIT_FOR_CAPACITY_ATTEMPT_BUDGET_SECS` (default 45 min, < 50 min)
#    wall-clock. At the budget the process raises
#    :class:`WaitForCapacityStillWaiting`, which the CLI handlers convert into
#    a structured ``[wait-for-capacity] STILL-WAITING`` line + exit code
#    :data:`EXIT_STILL_WAITING` (75, EX_TEMPFAIL). The caller (the /issue
#    orchestrator's bg-Bash loop) RE-RUNS the same command to continue
#    waiting — the loop is state-free, so a re-run resumes the wait exactly.
#    Why bounded: an in-process unbounded loop means a multi-hour bg command,
#    and ~1h-old bg provision commands were observed getting killed by session
#    respawns with no orchestrator wake (3 sessions went dark on 2026-06-09;
#    #530/#521/#532). Bounded attempts keep every bg command under the
#    kill-window AND give watchers a visible progress event per attempt.
#    Approved 2026-06-10 (refs #572), superseding the original NO CAP rule.
# 2. ONLY ``RunPodNoCapacityError`` + ``RunPodInsufficientBalanceError`` are
#    caught. Auth, bad config, transport-budget-exhausted, empty-gpu-list →
#    those still propagate fast per the "fail fast — never hide failures" rule
#    in CLAUDE.md. ``RunPodInsufficientBalanceError`` was added to the catch
#    set after #506 (2026-06-08) fail-exited to ``blocked`` on an
#    INSUFFICIENT_BALANCE refusal that would have cleared the moment another
#    pod freed $/hr headroom.
# 3. Backoff: exponential with full jitter — base 30s, doubling each attempt
#    up to a 600s (10 min) ceiling, then steady-poll at the ceiling forever.
#    Style matches :func:`runpod_api._backoff_sleep_secs` for consistency.
# 4. KeyboardInterrupt propagates so the operator can Ctrl-C cleanly and
#    ``spawn_session.py stop`` / a respawn doesn't leave a zombie.
# 5. Each attempt emits a structured stderr heartbeat
#    ``[wait-for-capacity] attempt N, waited Xm Ys, next retry in Zs``. The
#    /issue orchestrator (which bg-Bash-runs ``pod.py provision``) is the
#    correct surface for translating those heartbeats into ``epm:progress``
#    markers — pod_lifecycle deliberately does NOT shell out to ``task.py``
#    (the pod-side / branch-guard rule in CLAUDE.md). The 6-hour stale-marker
#    threshold in ``autonomous_session_watch.py`` gives plenty of headroom
#    against the 10-minute ceiling.

# Backoff knobs — module-level so tests can monkeypatch them.
WAIT_FOR_CAPACITY_BACKOFF_BASE_SECS = 30.0
WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS = 600.0  # 10 min ceiling

# Per-process wall-clock budget for ONE wait-for-capacity invocation. Must
# stay under 50 min (the observed bg-command kill window during session
# respawns — refs #572). At the budget the loop raises
# :class:`WaitForCapacityStillWaiting` instead of sleeping past it; the CLI
# converts that into a structured still-waiting exit so the orchestrator can
# re-run the same command (state-free resume of the wait). Env override:
# ``EPM_WAIT_FOR_CAPACITY_BUDGET_SECS``.
_DEFAULT_WAIT_FOR_CAPACITY_ATTEMPT_BUDGET_SECS = 45 * 60.0


def _wait_for_capacity_attempt_budget_secs() -> float:
    """Read the per-process wait budget (default 45 min). Bad values fall
    back to the default rather than crash the wait loop."""
    raw = os.environ.get("EPM_WAIT_FOR_CAPACITY_BUDGET_SECS", "").strip()
    if not raw:
        return _DEFAULT_WAIT_FOR_CAPACITY_ATTEMPT_BUDGET_SECS
    try:
        return max(0.0, float(raw))
    except ValueError:
        print(
            f"[pod_lifecycle] WARN: EPM_WAIT_FOR_CAPACITY_BUDGET_SECS={raw!r} is not "
            f"a number; using default "
            f"{_DEFAULT_WAIT_FOR_CAPACITY_ATTEMPT_BUDGET_SECS:.0f}s.",
            file=sys.stderr,
        )
        return _DEFAULT_WAIT_FOR_CAPACITY_ATTEMPT_BUDGET_SECS


# Exit code for the structured still-waiting exit (EX_TEMPFAIL — "temporary
# failure, retry"). Distinct from 0 (pod ready) and 1 (real failure) so the
# orchestrator / watchers can route on it without parsing stderr.
EXIT_STILL_WAITING = 75


class WaitForCapacityStillWaiting(RunPodError):
    """Raised when one wait-for-capacity process attempt exhausts its
    wall-clock budget without capacity / $/hr headroom appearing. NOT a
    failure: nothing was provisioned, nothing is billing, and re-running
    the same command resumes the wait. The CLI handlers convert this into
    ``[wait-for-capacity] STILL-WAITING`` + :data:`EXIT_STILL_WAITING`."""

    def __init__(self, *, verb: str, name: str, attempts: int, elapsed_secs: float) -> None:
        self.verb = verb
        self.name = name
        self.attempts = attempts
        self.elapsed_secs = elapsed_secs
        super().__init__(
            f"still waiting for capacity after {attempts} {verb} attempt(s), "
            f"{_format_elapsed(elapsed_secs)} elapsed (budget "
            f"{_wait_for_capacity_attempt_budget_secs():.0f}s)"
        )


def _emit_still_waiting_and_exit(exc: WaitForCapacityStillWaiting) -> NoReturn:
    """Print the structured still-waiting summary and exit
    :data:`EXIT_STILL_WAITING`. One line on stderr (where the heartbeats
    already go) and one on stdout (so an output-capturing caller that only
    keeps stdout still sees it)."""
    msg = (
        f"[wait-for-capacity] STILL-WAITING: {exc.verb} {exc.name} has been "
        f"waiting {_format_elapsed(exc.elapsed_secs)} across {exc.attempts} "
        f"attempt(s) and reached this process's wall-clock budget. No pod was "
        f"provisioned; nothing is billing. RE-RUN THE SAME COMMAND to continue "
        f"waiting (the wait loop is state-free). Exit code {EXIT_STILL_WAITING} "
        f"= still-waiting, not failure."
    )
    print(msg, file=sys.stderr, flush=True)
    print(msg, flush=True)
    raise SystemExit(EXIT_STILL_WAITING)


def _wait_for_capacity_backoff_secs(attempt: int) -> float:
    """Exponential backoff with full jitter for retry ``attempt`` (1-indexed).

    attempt=1 -> window=[0, base], attempt=2 -> [0, 2*base], ..., capped at
    :data:`WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS`. Full jitter (uniform
    0..window) avoids synchronized retry storms across parallel provisions.

    The exponent is clamped to 32 because (a) once the window reaches the
    cap, larger exponents are irrelevant — ``min(...)`` pins to the
    ceiling anyway — and (b) ``2 ** N`` overflows Python ``float`` past
    ~1024 and raises ``OverflowError`` (``int too large to convert to
    float``), which would CRASH this unbounded retry loop after ~3.5 days
    at the 10-min ceiling (≈attempt 1025). The whole point of the loop is
    "retry indefinitely," so an arithmetic overflow at high attempt counts
    is forbidden. ``30s * 2**32`` already overshoots the 600s ceiling by
    ~1e8, so the clamp is harmless on the cap-pinning side.
    """
    assert attempt >= 1, attempt
    exp = min(attempt - 1, 32)
    window = min(
        WAIT_FOR_CAPACITY_BACKOFF_BASE_SECS * (2**exp),
        WAIT_FOR_CAPACITY_BACKOFF_CAP_SECS,
    )
    return random.uniform(0.0, window)


def _format_elapsed(secs: float) -> str:
    """Render an elapsed-time seconds value as ``HhMmSs`` / ``MmSs`` / ``Ss``
    for the wait-for-capacity heartbeat lines."""
    total = int(secs)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def create_pod_with_wait_for_capacity(
    *,
    name: str,
    gpu_type: str | list[str],
    gpu_count: int,
    volume_gb: int,
    container_disk_gb: int,
    preflight_check: Callable[[], None] | None = None,
) -> PodInfo:
    """Provision policy wrapper: retry ``create_pod`` on no-capacity
    OR INSUFFICIENT_BALANCE refusals (both transient + no-cost-while-idle),
    bounded per process by :func:`_wait_for_capacity_attempt_budget_secs`
    (raises :class:`WaitForCapacityStillWaiting` at the budget — refs #572).

    Catches :class:`RunPodNoCapacityError` (every supply lever returned
    null) AND :class:`RunPodInsufficientBalanceError` (projected account
    $/hr would exceed the console cap; clears the moment a sibling pod
    frees $/hr headroom — #506, 2026-06-08). Every other ``RunPodError``
    propagates immediately so real failures (auth, bad config,
    transport-budget-exhausted, empty gpu list) fail fast per CLAUDE.md.

    ``preflight_check`` runs at the TOP of each loop attempt (before
    ``create_pod``) when supplied. It is the local-side analog of the
    live API guard: typically a bound call to
    :func:`_assert_under_account_hourly_cap` with
    ``transient_on_exceed=True``, which raises
    :class:`RunPodInsufficientBalanceError` if the projected account $/hr
    would exceed the cap. Catching it inside the loop means freed $/hr
    headroom from a sibling pod is detected at the next tick and the
    provision proceeds without operator intervention — closing the gap
    where the pre-call SystemExit guard would hard-exit a wait-mode run
    to ``blocked`` BEFORE the wait loop ever started (the #506
    first-block at 03:43Z 2026-06-08). When the parameter is ``None``
    (default) no preflight runs, preserving the legacy behavior for any
    caller that doesn't pass it.

    Loops with exponential-jittered backoff (base 30s, cap 10 min) until
    capacity / $/hr headroom is available or the per-process wall-clock
    budget trips (then raises :class:`WaitForCapacityStillWaiting`). KeyboardInterrupt
    propagates so the operator can Ctrl-C / SIGINT and exit cleanly. Each
    attempt emits a structured ``[wait-for-capacity]`` stderr line whose
    ``reason=`` token distinguishes ``local-cap`` (preflight refusal),
    ``insufficient-balance`` (API-side refusal), and ``no-capacity``
    (supply); the /issue orchestrator should surface these as
    ``epm:progress`` markers so ``autonomous_session_watch.py`` (6h stale
    threshold) sees liveness.
    """
    attempt = 0
    start = time.monotonic()
    print(
        f"[wait-for-capacity] starting retry loop for {name} ({gpu_count}x {gpu_type}); "
        f"per-process budget {_wait_for_capacity_attempt_budget_secs():.0f}s",
        file=sys.stderr,
        flush=True,
    )
    while True:
        attempt += 1
        # Source tag distinguishes local-guard refusals from live-API refusals
        # in the heartbeat. Set per-call so a successful preflight followed by
        # an API refusal in the same iteration still reports the correct source.
        source = "api"
        try:
            if preflight_check is not None:
                source = "local"
                preflight_check()
                source = "api"
            return create_pod(
                name=name,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                volume_gb=volume_gb,
                container_disk_gb=container_disk_gb,
            )
        except (RunPodNoCapacityError, RunPodInsufficientBalanceError) as exc:
            # All three classes routed through this branch are transient +
            # no-cost-while-idle:
            #   - RunPodNoCapacityError: every supply lever returned null
            #     (no host has free GPUs in the requested config). Clears
            #     when capacity frees up.
            #   - RunPodInsufficientBalanceError (source="api"): RunPod
            #     refused because projected account $/hr would exceed the
            #     console cap. Clears the moment any other pod on the
            #     team stops or terminates (#506, 2026-06-08).
            #   - RunPodInsufficientBalanceError (source="local"): our
            #     local pre-flight estimate beat the API to the same
            #     refusal. Same recovery condition (a sibling pod frees
            #     $/hr headroom) and same wait-with-backoff response —
            #     closes the #506 first-block gap at 03:43Z where the
            #     unconditional SystemExit guard hard-exited to
            #     ``blocked`` BEFORE the wait loop could fire.
            # Nothing is running while we wait, so no $/hr is being spent.
            # Every other RunPodError (auth, bad config, transport-budget-
            # exhausted, empty gpu list) still propagates immediately per
            # "fail fast — never hide failures".
            elapsed = time.monotonic() - start
            sleep_secs = _wait_for_capacity_backoff_secs(attempt)
            if isinstance(exc, RunPodNoCapacityError):
                reason = "no-capacity"
            elif source == "local":
                reason = "local-cap (pre-flight $/hr estimate)"
            else:
                reason = "insufficient-balance (account $/hr cap)"
            # Per-process wall-clock budget (refs #572): never sleep PAST the
            # budget — raise the structured still-waiting signal instead so
            # the caller exits EXIT_STILL_WAITING and the orchestrator
            # re-runs the command. Checked before the sleep so one process
            # attempt is hard-capped under the ~50 min bg-kill window.
            budget = _wait_for_capacity_attempt_budget_secs()
            if budget > 0 and elapsed + sleep_secs > budget:
                raise WaitForCapacityStillWaiting(
                    verb="provision",
                    name=name,
                    attempts=attempt,
                    elapsed_secs=elapsed,
                ) from exc
            print(
                f"[wait-for-capacity] attempt {attempt} for {name}: "
                f"{reason} ({exc}); waited {_format_elapsed(elapsed)}, "
                f"next retry in {sleep_secs:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            try:
                time.sleep(sleep_secs)
            except KeyboardInterrupt:
                print(
                    f"[wait-for-capacity] interrupted during sleep after "
                    f"{attempt} attempts, waited {_format_elapsed(elapsed)}; "
                    "exiting cleanly.",
                    file=sys.stderr,
                    flush=True,
                )
                raise


def _autonomous_session() -> bool:
    """True iff this process is running inside an autonomous /issue session.

    Mirrors :func:`task.py`'s parse exactly (case-insensitive truthiness;
    falsy set ``{"", "0", "false", "no"}``) so the two never disagree on a
    value like ``"no"`` / ``"FALSE"``.
    """
    raw = os.environ.get("EPM_AUTONOMOUS_SESSION", "").strip().lower()
    return raw not in ("", "0", "false", "no")


def _resume_with_balance_wait_if_autonomous(
    *,
    pod: EphemeralPod,
    name: str,
    issue: int,
    preflight_check: Callable[[], None] | None = None,
    force_wait: bool = False,
) -> None:
    """Wrap ``resume_pod`` with INSUFFICIENT_BALANCE retry-wait in
    autonomous mode (or when ``force_wait=True`` — the interactive
    ``--wait-for-capacity`` opt-in, #530) + actionable error in
    interactive mode, plus the pre-existing SUPPLY_CONSTRAINT
    actionable-message branch.

    Why this helper exists
    ----------------------
    The resume call's pre-flight ``_assert_under_account_hourly_cap``
    guard estimates account $/hr from our local rate table; the live
    RunPod side enforces the cap with its own ground-truth pricing AND a
    concurrent-provision-race window. The local estimate can disagree —
    rate mis-estimate (override missing or stale), or a sibling caller
    raced and spun up between our guard and our resume call. Either way,
    INSUFFICIENT_BALANCE at the actual ``podResume`` is **transient +
    no-cost-while-idle** (the stopped pod is not burning $/hr while we
    wait), so the right behavior in autonomous mode is the same
    budget-bounded retry-with-backoff used by
    :func:`create_pod_with_wait_for_capacity`. NOT fail-exit to
    ``blocked`` (incident #506, 2026-06-08). Outside autonomous mode we
    fail loud with an actionable message so the human can choose
    explicitly (stop a sibling pod, raise the console cap, etc.).

    ``preflight_check`` runs at the TOP of each loop attempt (before
    ``resume_pod``) when supplied. It is the local analog of the
    INSUFFICIENT_BALANCE handler below: typically a bound call to
    :func:`_assert_under_account_hourly_cap` with
    ``transient_on_exceed=True``, which raises
    :class:`RunPodInsufficientBalanceError` when the projected account
    $/hr would exceed the cap. The handler then either fails loud
    (interactive mode — the local-pre-flight failed BEFORE the resume
    call would have, so we still want a clear actionable message) or
    waits + retries (autonomous mode, closing the resume-path analog of
    the #506 first-block gap). When the parameter is ``None`` (default)
    no preflight runs, preserving the legacy behavior for any caller
    that doesn't pass it.

    ``force_wait=True`` enables the same retry-wait OUTSIDE autonomous
    mode. It backs the interactive ``pod.py resume --wait-for-capacity``
    flag (#530, 2026-06-09: a cap-refused interactive resume had no
    retry path, so the orchestrator hand-rolled a shell loop around
    ``pod.py resume``). Default ``False`` keeps every pre-existing
    caller byte-identical.

    SUPPLY_CONSTRAINT on resume is unchanged from the prior behavior:
    resume never relocates a pod (its volume is pinned to the original
    host), so waiting cannot help if that specific host is out of
    GPUs — only a fresh provision (losing the volume) or hand-retry
    later does. We surface the actionable message in both modes.
    """
    wait_on_balance = force_wait or _autonomous_session()
    attempt = 0
    start = time.monotonic()
    while True:
        attempt += 1
        source = "api"
        try:
            if preflight_check is not None:
                source = "local"
                preflight_check()
                source = "api"
            resume_pod(pod.pod_id, pod.gpu_count)
            return
        except RunPodInsufficientBalanceError as exc:
            if not wait_on_balance:
                # Interactive: fail loud, name the actionable next steps.
                raise SystemExit(
                    f"Cannot resume {name}: RunPod refused because the projected "
                    f"account $/hr would exceed the console spending cap "
                    f"(INSUFFICIENT_BALANCE). The local pre-flight guard's rate "
                    f"estimate disagreed — most likely cause is rate-table drift "
                    f"or a sibling pod that started between the guard and the "
                    f"resume call.\n\n"
                    f"Options: stop or terminate another pod to free $/hr "
                    f"headroom, raise the console cap (and `export "
                    f"RUNPOD_ACCOUNT_HOURLY_CAP=<new>`), or tune per-GPU rate "
                    f"estimates via RUNPOD_RATE_<GPU>_USD if they over-estimate "
                    f"your actual pricing. Then re-run `pod.py resume --issue "
                    f"{issue}` — or re-run it with `--wait-for-capacity` to "
                    f"retry with backoff until a sibling pod frees headroom.\n"
                    f"  Underlying error: {exc}"
                ) from exc
            elapsed = time.monotonic() - start
            sleep_secs = _wait_for_capacity_backoff_secs(attempt)
            if source == "local":
                reason = "local-cap (pre-flight $/hr estimate)"
            else:
                reason = "insufficient-balance (account $/hr cap)"
            # Per-process wall-clock budget (refs #572) — see
            # create_pod_with_wait_for_capacity for the rationale. The
            # resume wait is the same no-cost-while-idle class, so the
            # same bounded-attempt / re-run contract applies.
            budget = _wait_for_capacity_attempt_budget_secs()
            if budget > 0 and elapsed + sleep_secs > budget:
                raise WaitForCapacityStillWaiting(
                    verb="resume",
                    name=name,
                    attempts=attempt,
                    elapsed_secs=elapsed,
                ) from exc
            print(
                f"[wait-for-capacity] resume attempt {attempt} for {name}: "
                f"{reason} ({exc}); waited "
                f"{_format_elapsed(elapsed)}, next retry in {sleep_secs:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            try:
                time.sleep(sleep_secs)
            except KeyboardInterrupt:
                print(
                    f"[wait-for-capacity] interrupted during resume sleep after "
                    f"{attempt} attempts, waited {_format_elapsed(elapsed)}; "
                    "exiting cleanly.",
                    file=sys.stderr,
                    flush=True,
                )
                raise
        except RunPodError as exc:
            if _is_supply_constraint(exc):
                # Resume never relocates — the stopped pod's volume is pinned
                # to its original host. If that host has no free GPUs we
                # CANNOT retry around it; the user must provision a fresh pod
                # (losing this volume) or wait for capacity. Do NOT auto-
                # terminate or auto-provision here — that would silently
                # destroy the stopped pod's volume.
                raise SystemExit(
                    f"Cannot resume {name}: its former host has no free GPUs "
                    f"(supply constraint). Resume never relocates a pod, so "
                    f"this can't be retried. Either wait for capacity to free "
                    f"up and re-run `pod.py resume --issue {issue}`, or "
                    f"provision a FRESH pod with `python scripts/pod.py "
                    f"provision --issue {issue} --intent <intent>` (this loses "
                    f"the stopped pod's volume — terminate it first with "
                    f"`pod.py terminate --issue {issue} --yes` if you want it "
                    f"gone).\n  Underlying error: {exc}"
                ) from exc
            raise


# ─── SSH-wait alarm (billing pod unreachable >1h — refs #572) ────────────────
#
# pod-488 (2026-06-09) sat SSH-unreachable for ~13.7h at $32/hr because a
# SUPPLY_CONSTRAINT-blocked resume left a stale port in pods.conf and nothing
# ever escalated past per-call failures. This tracker persists the FIRST
# failure timestamp per pod across processes (state file under the gitignored
# .claude/cache/), and once a pod has been unreachable for
# ``EPM_SSH_WAIT_ALARM_SECS`` (default 1h) while the live API still reports it
# RUNNING (= billing), prints a LOUD structured ``[ssh-wait-ALARM]`` line
# naming the recovery command. Re-alarms at most once per alarm window.
# Fail-soft by design: an observability tracker must never crash the
# lifecycle operation it observes.

_DEFAULT_SSH_WAIT_ALARM_SECS = 3600.0
_SSH_WAIT_STATE_PATH = PROJECT_ROOT / ".claude" / "cache" / "ssh-wait-alarm.json"


def _ssh_wait_alarm_secs() -> float:
    """Alarm threshold (default 1h). Env override ``EPM_SSH_WAIT_ALARM_SECS``;
    bad values fall back to the default."""
    raw = os.environ.get("EPM_SSH_WAIT_ALARM_SECS", "").strip()
    if not raw:
        return _DEFAULT_SSH_WAIT_ALARM_SECS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_SSH_WAIT_ALARM_SECS


def _load_ssh_wait_state() -> dict:
    """Read the cross-process SSH-wait state ({pod_name: {first_failure_ts,
    last_alarm_ts}}). Garbled / missing file -> {} (fresh episodes)."""
    try:
        return json.loads(_SSH_WAIT_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_ssh_wait_state(state: dict) -> None:
    """Persist the SSH-wait state atomically; IO failures are swallowed (the
    tracker is observability, never worth crashing a provision/resume)."""
    try:
        _SSH_WAIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SSH_WAIT_STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(_SSH_WAIT_STATE_PATH)
    except OSError as exc:
        print(f"[pod_lifecycle] WARN: ssh-wait state save failed: {exc}", file=sys.stderr)


def _pod_desired_status_by_name(pod_name: str) -> str:
    """Best-effort live desiredStatus for ``pod_name`` (``"UNKNOWN"`` when the
    API is unreachable or the pod is gone). Only called on the rare alarm
    path, never per-probe."""
    try:
        for p in list_team_pods():
            if p.name == pod_name:
                return p.desired_status or "UNKNOWN"
    except Exception as exc:
        print(
            f"[pod_lifecycle] WARN: live-status check for ssh-wait alarm failed: {exc}",
            file=sys.stderr,
        )
    return "UNKNOWN"


def note_ssh_wait_outcome(
    pod_name: str,
    *,
    reachable: bool,
    desired_status: str | None = None,
    now: float | None = None,
) -> None:
    """Record one SSH reachability observation for ``pod_name`` and fire the
    1h billing-pod alarm when warranted (refs #572).

    ``reachable=True`` closes the episode (state cleared). ``reachable=False``
    opens / extends it; once the episode exceeds :func:`_ssh_wait_alarm_secs`
    AND the pod is RUNNING per the live API (or the caller passed
    ``desired_status``; ``UNKNOWN`` alarms too — failing loud beats staying
    silent on a possibly-billing pod), a structured ``[ssh-wait-ALARM]`` line
    is printed, at most once per alarm window. EXITED pods never alarm (not
    billing; the episode stays open in case of a later resume)."""
    now = time.time() if now is None else now
    state = _load_ssh_wait_state()
    if reachable:
        if pod_name in state:
            state.pop(pod_name, None)
            _save_ssh_wait_state(state)
        return
    entry = state.get(pod_name)
    if not isinstance(entry, dict):
        entry = {}
    first = entry.get("first_failure_ts")
    if not isinstance(first, int | float):
        first = now
        entry["first_failure_ts"] = first
    waited = now - first
    threshold = _ssh_wait_alarm_secs()
    last_alarm = entry.get("last_alarm_ts")
    if not isinstance(last_alarm, int | float):
        last_alarm = 0.0
    if threshold > 0 and waited >= threshold and (now - last_alarm) >= threshold:
        status = desired_status or _pod_desired_status_by_name(pod_name)
        if status != "EXITED":
            billing = "RUNNING (BILLING)" if status == "RUNNING" else f"status={status}"
            print(
                f"[ssh-wait-ALARM] pod {pod_name} has been SSH-unreachable for "
                f"{_format_elapsed(waited)} while {billing}. Likely a stale "
                f"host/port in pods.conf (the #488 pattern: a retry path "
                f"brought the pod back at a new port outside _upsert_pods_conf). "
                f"Recovery: `uv run python scripts/pod.py config "
                f"--refresh-from-api {pod_name}`, then re-check; if the pod is "
                f"genuinely idle, stop it (`pod.py stop`) to halt the burn.",
                file=sys.stderr,
                flush=True,
            )
            entry["last_alarm_ts"] = now
    state[pod_name] = entry
    _save_ssh_wait_state(state)


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

    Every probe outcome additionally feeds :func:`note_ssh_wait_outcome`
    (when ``issue`` is given) so a pod that stays unreachable across repeated
    preflights for >1h while billing trips the ``[ssh-wait-ALARM]`` (refs
    #572).
    """
    pod_name = _canonical_pod_name(issue) if issue is not None else None
    if _tcp_open(host, port, timeout):
        if pod_name:
            note_ssh_wait_outcome(pod_name, reachable=True)
        return True
    if pod_name:
        note_ssh_wait_outcome(pod_name, reachable=False)

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
        if pod_name:
            note_ssh_wait_outcome(pod_name, reachable=True)
        print(
            f"[pod_lifecycle] SSH preflight recovered after resume: "
            f"{new_host}:{new_port} is reachable.",
            file=sys.stderr,
        )
        return True

    if pod_name:
        note_ssh_wait_outcome(pod_name, reachable=False)
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


# ─── account hourly-spend guard ──────────────────────────────────────────────


# RunPod enforces a per-account hourly spending limit set in the console (the
# "$80/hr cap"). When the projected sum-of-running-pod hourly rates exceeds
# that cap, RunPod refuses the next ``podFindAndDeployOnDemand`` /
# ``podResume`` with ``INSUFFICIENT_BALANCE: Renting this pod would put you
# over your current spending limit ($X/hr)`` — AFTER the user has already
# initiated the run. We mirror the cap locally so the guard fails LOUD
# pre-flight with the projected total instead of mid-run (incidents #503,
# #505 on 2026-06-05). Default 80.0 USD/hr; override via env to match
# whatever the console cap is set to.
_DEFAULT_ACCOUNT_HOURLY_CAP_USD = 80.0


def _account_hourly_cap_usd() -> float:
    """Read the local mirror of the RunPod account $/hr cap. Env override
    ``RUNPOD_ACCOUNT_HOURLY_CAP`` (default 80.0). Bad values fall back to the
    default rather than crash the lifecycle.
    """
    raw = os.environ.get("RUNPOD_ACCOUNT_HOURLY_CAP", "").strip()
    if not raw:
        return _DEFAULT_ACCOUNT_HOURLY_CAP_USD
    try:
        return max(0.0, float(raw))
    except ValueError:
        print(
            f"[pod_lifecycle] WARN: RUNPOD_ACCOUNT_HOURLY_CAP={raw!r} is not a number; "
            f"using default ${_DEFAULT_ACCOUNT_HOURLY_CAP_USD:.2f}/hr.",
            file=sys.stderr,
        )
        return _DEFAULT_ACCOUNT_HOURLY_CAP_USD


def _assert_under_account_hourly_cap(
    *,
    verb: str,
    pod_label: str,
    intended_gpu_type: str | None,
    intended_gpu_count: int | None,
    skip_for_same_pod: str | None = None,
    transient_on_exceed: bool = False,
) -> None:
    """Refuse to provision/resume when the projected account $/hr would exceed
    the RunPod console cap. Fails LOUD pre-flight with the current burn, the
    new pod's estimated rate, the projected total, and the cap.

    Parameters
    ----------
    verb : ``"provision"`` or ``"resume"`` — only used in the error message.
    pod_label : human-friendly id for the pod we're about to start (e.g.
        ``"pod-137"``); only used in the error message.
    intended_gpu_type : short GPU name (``"H100"``) or full GraphQL id; passed
        to :func:`runpod_api.estimate_pod_hourly_rate`.
    intended_gpu_count : how many GPUs the new pod will use.
    skip_for_same_pod : when ``resume`` re-queries the API the stopped pod
        already shows ``RUNNING=False``, but if there's a sibling RUNNING pod
        with the SAME name from a duplicate-provision race, we'd double-count.
        Pass the pod name to exclude from the current-burn sum (defensive —
        the resume path is the one that triggered #503).
    transient_on_exceed : when False (default) an over-cap projection raises
        :class:`SystemExit` with the actionable human-readable message — the
        original interactive / one-shot contract from #503/#505. When True
        an over-cap projection instead raises
        :class:`RunPodInsufficientBalanceError`, so a calling retry loop
        (``create_pod_with_wait_for_capacity`` /
        ``_resume_with_balance_wait_if_autonomous``) can treat the local
        guard the same way it treats the live RunPod-side INSUFFICIENT_BALANCE
        refusal: transient + no-cost-while-idle, retry-with-backoff until a
        sibling pod frees $/hr headroom. The local guard runs an estimate
        against the same cap RunPod itself enforces, so the right behavior in
        an autonomous wait loop is identical (incident #506 first block at
        03:43Z 2026-06-08: the local guard hard-exited to ``blocked`` before
        the API-side fix from #506 could even fire). Default OFF preserves
        the byte-identical SystemExit behavior for every pre-existing caller.

    Per the "Fail fast — never hide failures" rule: if
    :func:`current_account_hourly_burn` raises (API unreachable), the
    exception propagates. We CANNOT make the decision without the live state,
    so we refuse the operation rather than silently letting RunPod surface it
    mid-run.
    """
    cap = _account_hourly_cap_usd()
    intended_rate = estimate_pod_hourly_rate(intended_gpu_type, intended_gpu_count)
    current_total, breakdown = current_account_hourly_burn()
    if skip_for_same_pod:
        # Subtract any RUNNING pod sharing the resumed pod's name (defensive
        # vs duplicate-provision races; in the normal resume path the stopped
        # pod isn't in `breakdown` at all because it's EXITED).
        for name, rate in breakdown:
            if name == skip_for_same_pod:
                current_total -= rate
    projected = current_total + intended_rate
    if projected <= cap:
        return
    if transient_on_exceed:
        # Wait-mode caller (autonomous /issue / explicit --wait-for-capacity):
        # raise the same exception class the wait loop already catches from
        # the live API, so the loop re-checks at each backoff tick and proceeds
        # the moment a sibling pod frees $/hr headroom. The message is short
        # — the verbose actionable form is only useful at an interactive
        # terminal, and the loop heartbeat already prints attempt/elapsed.
        raise RunPodInsufficientBalanceError(
            f"local pre-flight: projected ${projected:.2f}/hr (current "
            f"${current_total:.2f} + this pod ${intended_rate:.2f}) "
            f"exceeds cap ${cap:.2f}/hr"
        )
    breakdown_lines = (
        "\n".join(f"    {name:<30} ${rate:6.2f}/hr" for name, rate in breakdown[:10])
        or "    (no other RUNNING pods)"
    )
    omitted = max(0, len(breakdown) - 10)
    if omitted:
        breakdown_lines += f"\n    ... and {omitted} more"
    raise SystemExit(
        f"\nRefusing to {verb} {pod_label}: would exceed the RunPod account "
        f"hourly spending cap.\n"
        f"  Current burn   : ${current_total:6.2f}/hr (sum of RUNNING pods)\n"
        f"  This pod adds  : ${intended_rate:6.2f}/hr "
        f"({intended_gpu_count}x {_short_gpu_label(intended_gpu_type)})\n"
        f"  Projected total: ${projected:6.2f}/hr\n"
        f"  Account cap    : ${cap:6.2f}/hr "
        f"(local mirror; override with RUNPOD_ACCOUNT_HOURLY_CAP)\n"
        f"  Current RUNNING pods:\n{breakdown_lines}\n"
        f"\nOptions: stop or terminate other pods to free capacity, raise the "
        f"console cap (and `export RUNPOD_ACCOUNT_HOURLY_CAP=<new>`), "
        f"tune per-GPU rate estimates via RUNPOD_RATE_<GPU>_USD if they "
        f"over-estimate your actual pricing, or re-run the {verb} with "
        f"`--wait-for-capacity` to retry with backoff until headroom frees.\n"
    )


def _short_gpu_label(gpu_type_id: str | None) -> str:
    """Tiny shim so the guard's error message reads ``H100`` not the full id.
    Lives in pod_lifecycle (not runpod_api) so we don't widen the latter's
    public surface for a presentation helper.
    """
    if not gpu_type_id:
        return "?"
    for short in ("H100", "H200", "A100"):
        if short in gpu_type_id:
            return short
    return gpu_type_id


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


def _provision_wait_register_bootstrap(
    args: argparse.Namespace,
    name: str,
    info: PodInfo,
    intent_label: str,
) -> None:
    """Shared provision tail: wait for SSH, register in pods.conf, bootstrap.

    Extracted (#747) so the GPU and CPU (deployCpuPod) create paths converge on
    ONE wait/register/bootstrap implementation — identical for both pod kinds
    (SSH bring-up, pods.conf upsert, bootstrap_pod.sh). The caller has already
    created ``info`` (a GPU ``create_pod`` / ``create_pod_with_wait_for_capacity``
    or the CPU ``create_cpu_pod``) and printed the "Created ... waiting for SSH"
    line; this finishes the provision and either returns clean or ``sys.exit``s
    on a bootstrap failure (preserving the prior behavior verbatim).
    """
    try:
        ready = wait_for_ssh(info.pod_id, timeout=600)
    except RunPodError:
        # The pod exists and is billing, but never exposed 22/tcp within the
        # window — record the wait so repeated attempts accumulate toward the
        # 1h [ssh-wait-ALARM], and name the recovery before propagating.
        note_ssh_wait_outcome(name, reachable=False, desired_status="RUNNING")
        print(
            f"  Pod {info.pod_id} ({name}) is created (billing) but exposed no "
            f"public SSH mapping in 10 min; pods.conf was NOT updated. Once it "
            f"comes up, run `uv run python scripts/pod.py config "
            f"--refresh-from-api {name}` — or terminate it if it never does.",
            file=sys.stderr,
        )
        raise
    note_ssh_wait_outcome(name, reachable=True)
    print(f"  SSH ready at {ready.ssh_host}:{ready.ssh_port}")

    # Contiguous read-modify-write under the sidecar lock (task #1183) so a
    # concurrent session's register/terminate cannot interleave.
    with _metadata_lock():
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

    rc = _bootstrap(name, intent_label=intent_label)
    if rc != 0:
        print(
            f"\nBootstrap exited with code {rc}. Pod is up but not experiment-ready.\n"
            f"Investigate, then either re-run "
            f"`POD_INTENT={intent_label} bash scripts/bootstrap_pod.sh {name}` or\n"
            f"`python scripts/pod.py terminate --issue {args.issue}` to discard.",
            file=sys.stderr,
        )
        sys.exit(rc)

    print(f"\nDone. SSH with: ssh {name}")


def cmd_provision(args: argparse.Namespace) -> None:
    """Create a fresh pod for issue #N, wait for SSH, register it, bootstrap it."""
    if args.list_intents:
        print(list_intents())
        return

    if args.issue is None:
        raise SystemExit("--issue <N> is required")

    # Warn-only pod-safety check (#1177) — BEFORE any RunPod API call
    # (list_team_pods below), so it prints even when the provision later
    # fails on capacity, and covers the CPU branch, the GPU branch, AND
    # --dry-run (deliberate divergence from the terminate guard, which
    # skips dry-run because it BLOCKS; a warn-only line is exactly what a
    # preview should surface).
    _warn_on_terminal_parent_provision(args.issue)

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

    # CPU-only intent branch (#747): a cheap CPU intent (cpu-small / cpu-mid)
    # resolves to a RunPod CPU instance_id (deployCpuPod), NOT a GPU spec. This
    # branch is checked FIRST, before the GPU _resolve_spec below (which
    # KeyErrors on a CPU intent). CPU pods may be provisioned in PARALLEL — the
    # "one multi-GPU pod, not many single-GPU pods" rule is GPU-specific (it
    # exists to keep CUDA_VISIBLE_DEVICES-sharded GPU work on one pod), so N
    # independent `provision --issue <N>` / `--issue <N>-<suffix>` CPU pods are
    # the right shape for parallelizable CPU work. The account-hourly-cap guard
    # (a $/hr GPU-spend guard) is skipped — CPU pods cost cents/hr and
    # estimate_pod_hourly_rate returns 0 for gpu_count=0 anyway. CPU pods are
    # on-demand only (no spot/wait-for-capacity lever on the RunPod CPU side);
    # a no-capacity miss raises RunPodNoCapacityError, the same terminal the
    # GPU path raises (the wait-for-capacity loop is a GPU-side feature).
    cpu_instance_id = resolve_cpu_intent(args.intent) if args.intent else None
    if cpu_instance_id is not None:
        intent_label = args.intent.strip().lower()
        print(f"Provisioning {name}: CPU pod {cpu_instance_id}  ({intent_label})")
        print("  Why: cheap CPU-only intent (#747); CPU pods may run in parallel.")
        if args.dry_run:
            print("\n[dry-run] Would call create_cpu_pod and wait for SSH; no API call made.")
            return
        # Intent-aware default volume (#747): the --volume-gb argparse default
        # (200 GB) is the GPU-pod default; a cheap CPU pod should not silently
        # provision a 200 GB persistent volume (it defeats the lane's cost
        # premise). When the user did NOT explicitly override the default, use
        # the cheap CPU default (DEFAULT_CPU_VOLUME_GB=40); an explicit non-200
        # value is always honored.
        cpu_volume_gb = DEFAULT_CPU_VOLUME_GB if args.volume_gb == 200 else args.volume_gb
        # Effective-payload cap check (#1010, live probe 2026-07-04): RunPod
        # validates deployCpuPod's EFFECTIVE container disk --
        # max(container_disk_gb, volume_gb); runpod_api._deploy_cpu_once folds
        # the CPU volume into containerDiskInGb -- against a per-flavor cap
        # ("Container Disk must be less than or equal to 20" for cpu3g, 80 for
        # cpu3c). The untouched defaults (container 50 / CPU volume 40 ->
        # effective 50) EXCEED the cpu3g cap, so a default cpu-small provision
        # fails validation outright. Rule: an over-cap effective payload in
        # the DEFAULT band (<= 50, runpod_api.DEFAULT_CONTAINER_DISK_GB --
        # covering untouched defaults AND the router-threaded
        # `--container-disk-gb 50` floor, which MUST clamp-and-proceed, never
        # refuse) is CLAMPED to the cap on BOTH knobs; an explicit request
        # ABOVE the 50 band REFUSES pre-API (exit 1, same UX as the
        # "Pod already exists" refusal above -- the cap is probe-verified, so
        # RunPod would reject it anyway after a wasted API round-trip).
        cpu_container_disk_gb = args.container_disk_gb
        caps = resolve_cpu_instance_caps(args.intent)
        effective_disk_gb = max(cpu_container_disk_gb, cpu_volume_gb)
        if caps is not None and effective_disk_gb > caps.max_container_disk_gb:
            if effective_disk_gb > DEFAULT_CONTAINER_DISK_GB:
                print(
                    f"Requested container/volume disk (effective "
                    f"{effective_disk_gb} GB) exceeds the {cpu_instance_id} cap "
                    f"({caps.max_container_disk_gb} GB effective containerDiskInGb; "
                    f"RunPod create-time validation rejects it, #1010).\n"
                    f"Shrink --container-disk-gb/--volume-gb to <= "
                    f"{caps.max_container_disk_gb}, or route the big-footprint "
                    f"stage to `--intent cpu-bigmem` (GCP n2-highmem)."
                )
                sys.exit(1)
            print(
                f"  Note (#1010): clamping container/volume disk (effective "
                f"{effective_disk_gb} GB) -> {caps.max_container_disk_gb} GB "
                f"({cpu_instance_id} validation cap; the default payload exceeds it)."
            )
            cpu_container_disk_gb = min(cpu_container_disk_gb, caps.max_container_disk_gb)
            cpu_volume_gb = min(cpu_volume_gb, caps.max_container_disk_gb)
        info = create_cpu_pod(
            name=name,
            instance_id=cpu_instance_id,
            volume_gb=cpu_volume_gb,
            container_disk_gb=cpu_container_disk_gb,
        )
        print(f"  Created CPU pod {info.pod_id} — waiting for SSH (up to 10 min)...")
        _provision_wait_register_bootstrap(args, name, info, intent_label)
        return

    spec, intent_label = _resolve_spec(args.intent, args.gpu_type, args.gpu_count)
    print(f"Provisioning {name}: {spec.gpu_count}x {spec.gpu_type}  ({intent_label})")
    print(f"  Why: {spec.rationale}")

    if args.dry_run:
        print("\n[dry-run] Would call create_pod and wait for SSH; no API call made.")
        return

    # --wait-for-capacity (or EPM_AUTONOMOUS_SESSION=1) turns the one-shot
    # ``create_pod`` into an unbounded retry loop keyed on
    # ``RunPodNoCapacityError``. Default OFF so interactive provisions still
    # fail fast (humans want to know immediately when nothing is available).
    # Autonomous sessions auto-enable because "the experiment should start
    # when it has space" — there is no human to escalate to.
    wait_for_capacity = bool(args.wait_for_capacity) or _autonomous_session()
    if wait_for_capacity:
        if not args.wait_for_capacity:
            print(
                "  EPM_AUTONOMOUS_SESSION=1 → auto-enabling --wait-for-capacity "
                "(unbounded retry on SUPPLY_CONSTRAINT)."
            )

        # Local account-hourly-spend guard is routed THROUGH the wait loop in
        # wait mode (transient_on_exceed=True → RunPodInsufficientBalanceError).
        # The wait loop re-checks it at each backoff tick, so freed $/hr
        # headroom from a sibling pod is detected without operator
        # intervention. The unconditional SystemExit pre-call from the
        # interactive path is deliberately ABSENT here: incident #506 first
        # block at 03:43Z 2026-06-08 was that pre-call hard-exiting the
        # autonomous run to ``blocked`` BEFORE the wait loop ever started.
        def _wait_mode_preflight() -> None:
            _assert_under_account_hourly_cap(
                verb="provision",
                pod_label=name,
                intended_gpu_type=spec.gpu_type,
                intended_gpu_count=spec.gpu_count,
                transient_on_exceed=True,
            )

        try:
            info = create_pod_with_wait_for_capacity(
                name=name,
                gpu_type=spec.gpu_type,
                gpu_count=spec.gpu_count,
                volume_gb=args.volume_gb,
                container_disk_gb=args.container_disk_gb,
                preflight_check=_wait_mode_preflight,
            )
        except WaitForCapacityStillWaiting as exc:
            _emit_still_waiting_and_exit(exc)
    else:
        # Interactive / one-shot: keep the unconditional pre-flight SystemExit
        # contract from #503/#505 — humans expect an immediate, actionable
        # refusal at the terminal rather than a silent wait loop.
        _assert_under_account_hourly_cap(
            verb="provision",
            pod_label=name,
            intended_gpu_type=spec.gpu_type,
            intended_gpu_count=spec.gpu_count,
        )
        info = create_pod(
            name=name,
            gpu_type=spec.gpu_type,
            gpu_count=spec.gpu_count,
            volume_gb=args.volume_gb,
            container_disk_gb=args.container_disk_gb,
        )
    print(f"  Created pod {info.pod_id} — waiting for SSH (up to 10 min)...")
    _provision_wait_register_bootstrap(args, name, info, intent_label)


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
    # here so the stopped_at timestamp persists. Contiguous RMW under the
    # sidecar lock (task #1183); the stop_pod API call above stays outside.
    with _metadata_lock():
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
    # Warn-only pod-safety check (#1177): resume creates a RUNNING pod the
    # watcher's pod-safety pass treats identically to a fresh provision
    # (#573's stopped pods were healthy follow-up pods; resuming one on a
    # terminal parent without signals is re-stopped ~20 min later).
    _warn_on_terminal_parent_provision(args.issue, verb="resume")
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

    # Pre-flight account hourly-spend guard. Resume rents capacity the same way
    # provision does — the stopped pod isn't currently burning $/hr, so adding
    # it back can push the account over the RunPod console cap (default
    # $80/hr). ``skip_for_same_pod`` defends against a duplicate-provision
    # race where a sibling pod shares the resumed pod's name.
    #
    # Interactive mode (no EPM_AUTONOMOUS_SESSION, no --wait-for-capacity):
    # fail LOUD pre-call with the projected total + actionable message
    # (#503/#505 contract — humans expect an immediate refusal at the
    # terminal). Wait mode (--wait-for-capacity, or auto-enabled by
    # EPM_AUTONOMOUS_SESSION=1): route the guard THROUGH the wait loop
    # (transient_on_exceed=True → RunPodInsufficientBalanceError, which the
    # loop already retries with backoff). The unconditional SystemExit
    # pre-call is deliberately ABSENT in the wait branch — incident #506
    # first block at 03:43Z 2026-06-08 was the analogous pre-call on
    # provision hard-exiting to ``blocked`` before the wait loop ever
    # started; the resume path is symmetric and the same gap is closed
    # here. The interactive --wait-for-capacity opt-in exists because a
    # cap-refused interactive resume previously had NO retry path, forcing
    # the orchestrator to hand-roll a shell loop around `pod.py resume`
    # (#530, 2026-06-09). SUPPLY_CONSTRAINT still fails loud in BOTH modes
    # — resume never relocates, so waiting cannot help there.
    wait_for_capacity = bool(args.wait_for_capacity) or _autonomous_session()
    if wait_for_capacity:
        if not args.wait_for_capacity:
            print(
                "  EPM_AUTONOMOUS_SESSION=1 → auto-enabling --wait-for-capacity "
                "(retry with backoff on $/hr-cap refusals)."
            )

        def _wait_mode_preflight() -> None:
            _assert_under_account_hourly_cap(
                verb="resume",
                pod_label=name,
                intended_gpu_type=pod.gpu_type,
                intended_gpu_count=pod.gpu_count,
                skip_for_same_pod=name,
                transient_on_exceed=True,
            )

        try:
            _resume_with_balance_wait_if_autonomous(
                pod=pod,
                name=name,
                issue=args.issue,
                preflight_check=_wait_mode_preflight,
                force_wait=True,
            )
        except WaitForCapacityStillWaiting as exc:
            _emit_still_waiting_and_exit(exc)
    else:
        _assert_under_account_hourly_cap(
            verb="resume",
            pod_label=name,
            intended_gpu_type=pod.gpu_type,
            intended_gpu_count=pod.gpu_count,
            skip_for_same_pod=name,
        )

        # Resume call. INSUFFICIENT_BALANCE here means the RunPod-side
        # account $/hr cap was hit despite our pre-flight estimate (rate
        # mis-estimate, or a race with a concurrent provision on the same
        # account). Interactive mode fails loud with an actionable message
        # so the human can stop/terminate another pod and re-run resume.
        # SUPPLY_CONSTRAINT is treated separately because resume never
        # relocates the pod — waiting won't help if the original host
        # itself is out of GPUs (the user must provision fresh + lose the
        # volume, or wait + retry by hand).
        _resume_with_balance_wait_if_autonomous(
            pod=pod,
            name=name,
            issue=args.issue,
        )
    try:
        ready = wait_for_ssh(pod.pod_id, timeout=600)
    except RunPodError:
        # The resume mutation SUCCEEDED — the pod is back and billing — but no
        # public SSH mapping appeared within the window, so the process dies
        # BEFORE _upsert_pods_conf. Record the wait for the 1h [ssh-wait-ALARM],
        # then re-read the LIVE pod state once to branch the recovery advice
        # (#664/#689 D.2): a pod that came back with NEW ports but pods.conf
        # still holds the PRE-STOP endpoint is the #488 stale-port case
        # (--refresh-from-api heals it); a pod still RUNNING with NO public port
        # is the #664 host wedge — resume is host-pinned, so --refresh-from-api
        # is a NO-OP and the only recovery is terminate + re-provision fresh.
        # Control flow is UNCHANGED — re-raise in BOTH branches.
        note_ssh_wait_outcome(name, reachable=False, desired_status="RUNNING")
        live = None
        try:
            live = get_pod(pod.pod_id)
        except Exception:
            live = None
        if live is not None and live.ssh_host and live.ssh_port:
            print(
                f"  Resume of {name} succeeded (pod is billing) but no public SSH "
                f"mapping appeared in 10 min; pods.conf still holds the PRE-STOP "
                f"endpoint. Once the pod is up, run `uv run python scripts/pod.py "
                f"config --refresh-from-api {name}` to heal pods.conf/SSH/MCP "
                f"(#488 stale-port).",
                file=sys.stderr,
            )
        else:
            print(
                f"  Resume of {name} brought the pod back RUNNING but with NO public "
                f"port — the host is degraded and resume is host-pinned (#664 wedge). "
                f"`--refresh-from-api` is a NO-OP here (the port is platform-absent, "
                f"not stale). Terminate + re-provision fresh: `uv run python "
                f"scripts/pod.py terminate --issue {args.issue} --yes` then re-launch. "
                f"The autonomous poller (backend_poll, #664 fix) auto-recovers this "
                f"when inputs are on HF.",
                file=sys.stderr,
            )
        raise
    note_ssh_wait_outcome(name, reachable=True)

    # Clear our project-side stopped_at marker; status/host/port refresh on read.
    # Synthetic-metadata pods (Branch 3 of _load_state) are promoted to disk
    # here so pods.conf gets refreshed and future commands see the metadata.
    # Contiguous RMW under the sidecar lock (task #1183).
    with _metadata_lock():
        metadata = _read_metadata_file()
        if name not in metadata:
            metadata[name] = pod.metadata
        metadata[name].stopped_at = None
        _write_metadata_file(metadata)

    refreshed = EphemeralPod(metadata=metadata[name], info=ready)
    _upsert_pods_conf(refreshed)
    print(f"  SSH ready at {refreshed.host}:{refreshed.port}")
    # RunPod stop/resume wipes the container overlay, destroying the uv
    # launcher (/root/.local/bin/uv) and the /usr/local/bin/{uv,uvx,python}
    # shims that bootstrap_pod.sh step 6 installs for non-interactive
    # non-login SSH shells. Without this restore step, every subsequent
    # `ssh pod "uv run ..."` (and every preflight + every experimenter
    # launch) silently fails until a human reinstalls uv manually.
    # `refreshed.host` / `refreshed.port` are the freshly-resumed endpoint.
    if refreshed.host is None or refreshed.port is None:
        raise SystemExit(
            f"resume completed but refreshed.host/port is missing for {name}; "
            f"cannot restore uv. Re-run `python scripts/pod.py bootstrap {name}`."
        )
    _restore_uv_on_pod(refreshed.host, refreshed.port)
    print(f"  pods.conf updated. Connect: ssh {name}")


# ---------------------------------------------------------------------------
# Pod-safety pre-provision warn (#1177)
# ---------------------------------------------------------------------------

# These three sets MIRROR the watcher's pod-safety auto-stop predicate
# (autonomous_session_watch.py: POD_SAFETY_AUTO_STOP,
# _POD_FOLLOWUP_SIGNAL_KINDS, _DONE_TRANSITION_KINDS) so the warning fires
# exactly when the watcher would auto-stop. Parity is pinned by
# tests/test_pod_lifecycle.py::test_provision_pod_safety_constants_match_watcher
# (constant-set equality) and
# tests/test_pod_lifecycle.py::test_provision_freshness_behavioral_parity_with_watcher
# (comparison semantics) — update BOTH sides together (never refactor the
# watcher from here; it is read-only from this module's perspective).
_POD_SAFETY_AUTO_STOP_STATUSES = frozenset(
    {"completed", "awaiting_promotion", "archived", "on_hold"}
)
_POD_FOLLOWUP_SIGNAL_KINDS = frozenset(
    {"epm:run-launched", "epm:followup-scope", "epm:free-analysis-followup-run"}
)
_POD_DONE_TRANSITION_KINDS = frozenset({"epm:promoted", "epm:status-changed"})
_KEEP_RUNNING_TAG = "keep-running"


def _parse_marker_ts(ts: object) -> float | None:
    """Epoch seconds for an events.jsonl ISO-8601 ``ts``
    (``task_workflow._utcnow_iso`` format ``%Y-%m-%dT%H:%M:%SZ``); ``None`` on
    any unparseable value — the event is then treated as absent (conservative
    toward warning, never a crash)."""
    try:
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _latest_marker_ts(events: list[dict], kinds: frozenset[str]) -> float | None:
    """Newest parseable ts among ``events`` whose ``kind`` is in ``kinds``,
    else ``None``. Mirrors ``autonomous_session_watch._latest_event_ts``
    semantics."""
    best: float | None = None
    for ev in events:
        if ev.get("kind") not in kinds:
            continue
        ts = _parse_marker_ts(ev.get("ts"))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def _fresh_followup_signal(events: list[dict]) -> bool:
    """True iff a follow-up signal marker is STRICTLY newer than the latest
    done-transition — byte-for-byte the watcher's ``_task_followup_active``
    comparison (signal missing -> False; done-transition missing -> False)."""
    sig = _latest_marker_ts(events, _POD_FOLLOWUP_SIGNAL_KINDS)
    if sig is None:
        return False
    done = _latest_marker_ts(events, _POD_DONE_TRANSITION_KINDS)
    if done is None:
        return False
    return sig > done


def _terminal_parent_warning_text(issue: int, status: str, verb: str) -> str:
    """The pod-safety two-signal warning body (#1177) — quotes the exact
    remedy commands so a session that never consulted the CLAUDE.md
    two-signal bullet can still fix the state at the terminal."""
    return (
        f"[pod_lifecycle] WARNING (pod-safety, #1177): task #{issue} is at status "
        f"'{status}' — inside the watcher's pod-safety AUTO-STOP set "
        f"(completed / awaiting_promotion / archived / on_hold) — with NO "
        f"keep-running tag and NO fresh follow-up signal marker "
        f"(epm:run-launched / epm:followup-scope / epm:free-analysis-followup-run "
        f"newer than the latest epm:promoted / epm:status-changed).\n"
        f"  The autonomous-session watcher will AUTO-STOP this pod ~20 min after "
        f"it starts RUNNING (2-miss accumulation; incidents #573 / #779: healthy "
        f"pods repeatedly stopped mid-bootstrap and misdiagnosed as flaky hosts).\n"
        f"  Two-signal recipe (CLAUDE.md 'Routing experiment intent'):\n"
        f"    1. BEFORE provisioning — or RIGHT NOW (the tag is "
        f"timestamp-independent and still shields an already-running pod):\n"
        f"         uv run python scripts/task.py add-tag {issue} keep-running\n"
        f"    2. Post epm:run-launched on #{issue} immediately once the pod exists:\n"
        f"         uv run python scripts/task.py post-marker {issue} "
        f"epm:run-launched --note '<pod name + purpose>'\n"
        f"    3. Remove the tag when the run completes so the auto-stop re-arms:\n"
        f"         uv run python scripts/task.py remove-tag {issue} keep-running\n"
        f"  Proceeding with {verb} (warn-only check)."
    )


def _warn_on_terminal_parent_provision(issue: int, *, verb: str = "provision") -> bool:
    """Warn-only pod-safety pre-provision check (#1177). Returns True iff the
    warning fired (for tests); NEVER blocks, NEVER raises on task-state
    problems — an unresolvable task (ad-hoc pod, registry miss, branch-guard
    fire) proceeds with a one-line NOTE. Mirrors the watcher's auto-stop
    predicate so the warning predicts exactly what the pod-safety pass will
    do. Provision-time only: a task that ENTERS the auto-stop set after
    provision is never warned — the watcher remains the runtime backstop (do
    not read the warning's absence as a safety guarantee).
    """
    try:
        from explore_persona_space.task_workflow import get_task, list_events
    except ImportError:
        print(
            f"[pod_lifecycle] NOTE: pod-safety pre-{verb} check skipped for "
            f"issue #{issue}: task_workflow module unavailable. Proceeding.",
            file=sys.stderr,
        )
        return False
    # Narrow fail-open set, grounded in the terminate guard above:
    # FileNotFoundError = task not in registry/on disk (ad-hoc pods);
    # RuntimeError = task_workflow branch-guard on non-main HEAD / git missing;
    # ValueError = malformed frontmatter / corrupt REGISTRY.json
    # (JSONDecodeError is a ValueError); OSError = unreadable events.jsonl.
    # Anything else (KeyboardInterrupt, MemoryError, a programming bug)
    # propagates — fail-fast rule.
    try:
        task = get_task(issue)
        status = task.get("status") or ""
        if status not in _POD_SAFETY_AUTO_STOP_STATUSES:
            return False
        tags = (task.get("frontmatter") or {}).get("tags") or []
        if _KEEP_RUNNING_TAG in tags:
            return False
        events = list_events(issue)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(
            f"[pod_lifecycle] NOTE: pod-safety pre-{verb} check skipped for "
            f"issue #{issue}: could not read task state "
            f"({type(exc).__name__}: {exc}). Proceeding.",
            file=sys.stderr,
        )
        return False
    if _fresh_followup_signal(events):
        return False
    print(_terminal_parent_warning_text(issue, status, verb), file=sys.stderr)
    return True


def _has_upload_verification_pass(issue: int) -> bool:
    """True iff the LATEST ``epm:upload-verification`` event on task ``issue``
    records a PASS verdict. Used by :func:`cmd_terminate` to refuse destroying
    an experiment pod whose artifacts haven't been verified uploaded to
    permanent storage.

    The verdict lives in the event's markdown ``note`` body — as
    ``**Verdict: PASS**`` for upload-verifier agent notes, as a JSON object
    ``{"verdict": "PASS", ...}`` for machine-readable verifier notes, or as
    a bare leading ``PASS`` token for orchestrator-posted notes — NOT as a
    top-level event field (the event keys are only
    ``ts, kind, version, by, note``). We read the LATEST upload-verification
    event so a re-verification overrides an earlier one.

    Reads events via :mod:`explore_persona_space.task_workflow` (which
    branch-guards to ``main`` and resolves the canonical tasks/ tree
    regardless of cwd). Returns False when no upload-verification event
    exists or its latest verdict is not PASS, so the caller can decide
    whether to refuse or warn-and-proceed.
    """
    from explore_persona_space.task_workflow import list_events

    verification_events = [
        ev for ev in list_events(issue) if ev.get("kind") == "epm:upload-verification"
    ]
    if not verification_events:
        return False
    note = verification_events[-1].get("note", "") or ""
    # First try to parse the note as a JSON object: the upload-verifier agent
    # legitimately posts machine-readable JSON-shaped notes of the form
    # ``{"verdict": "PASS", "discovered_pod_files": ..., "checked": {...}, ...}``
    # (incident 2026-06-10, task #488: the re-verification posted exactly this
    # shape, the regex chain below missed it because the quote between
    # ``verdict`` and ``:`` breaks the ``\*\*?verdict\s*:`` anchors and the
    # note doesn't start with a bare verdict token, forcing the orchestrator
    # to post a duplicate guard-parseable marker — same failure mode as the
    # 2026-06-05 task #465 incident, different note shape). We try this BEFORE
    # the regex fallbacks so a JSON-shaped note that happens to contain the
    # substring ``"verdict": "FAIL"`` in a nested ``checked`` block doesn't
    # accidentally trip a permissive prose regex first.
    try:
        parsed = json.loads(note)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and "verdict" in parsed:
        return str(parsed["verdict"]).strip().upper() == "PASS"
    # Prefer the canonical bold-prefixed verdict line (``**Verdict: PASS**``);
    # fall back to a looser ``Verdict: PASS`` form for older/unbolded notes;
    # final fallback accepts a bare verdict token at the very START of the
    # note (e.g. ``PASS (orchestrator-verified ...)``), which is the shape
    # the orchestrator posts when it verifies uploads directly without going
    # through the upload-verifier agent's bold-prefixed template (incident
    # 2026-06-05, task #465: the parser refused the terminate because the
    # orchestrator-posted note led with a bare ``PASS`` and neither
    # ``Verdict``-keyed regex matched, forcing a ``--skip-upload-verify``
    # override on a fully-verified pod). The ``re.match`` anchor is
    # load-bearing: it only fires when the note BEGINS with the verdict
    # token, so a stray ``PASS``/``FAIL`` later in a bold-less note body
    # cannot flip the parsed verdict (the existing anchor-on-bold guarantee
    # — see ``test_has_upload_verification_pass_anchors_on_bold_verdict_line``
    # — still holds because the bold regex is tried first and short-circuits).
    match = (
        re.search(r"\*\*\s*verdict\s*:\s*\*?\*?\s*(PASS|FAIL|WARN)\b", note, re.IGNORECASE)
        or re.search(r"verdict[:*\s]+(PASS|FAIL|WARN)\b", note, re.IGNORECASE)
        or re.match(r"\s*(PASS|FAIL|WARN)\b", note, re.IGNORECASE)
    )
    return match is not None and match.group(1).upper() == "PASS"


def _guard_upload_verification_before_terminate(
    issue: int, *, skip_flag: bool, dry_run: bool
) -> None:
    """Refuse to terminate an ``epm-issue-<N>`` / ``pod-<N>`` for a
    ``kind: experiment`` task unless an ``epm:upload-verification PASS``
    marker exists on the task, OR ``--skip-upload-verify`` was passed
    (logs a LOUD warning, still proceeds).

    Non-experiment tasks (``kind`` ∈ {analysis, infra, batch, survey}),
    tasks that can't be resolved (manual / ad-hoc pods, branch-guard
    failure, registry miss), and dry-runs all proceed without blocking —
    the guard exists for *experiment* pods that ran, not as a universal
    block. Origin: task #444 hand-orchestrated completion bypassed the
    Step-8 upload-verifier and silently lost the training-mix datasets;
    the verifier's checklist would have flagged the gap.

    Always proceeds in ``dry_run`` mode (the caller wants to preview, not
    block on a precondition).
    """
    if dry_run:
        return

    # Best-effort task lookup. If task_workflow can't resolve the task —
    # not in registry, repo_root branch-guard fires, sidecar names a
    # non-experiment issue number, etc. — we warn and proceed. The guard
    # is for experiment pods that ran; an unresolvable task is by
    # definition outside that scope.
    try:
        from explore_persona_space.task_workflow import get_task
    except ImportError:
        print(
            f"[pod_lifecycle] WARN: upload-verification guard skipped for issue "
            f"#{issue}: task_workflow module unavailable. Proceeding with terminate.",
            file=sys.stderr,
        )
        return

    # Narrow to the exact failure modes that legitimately mean "outside this
    # guard's scope, warn and proceed": FileNotFoundError (task not in
    # registry / on disk, stale registry entry), RuntimeError (task_workflow
    # branch-guard fires on non-main HEAD, or git missing), ValueError
    # (malformed body frontmatter, corrupt REGISTRY.json — JSONDecodeError is
    # a ValueError). A genuinely unexpected error (KeyboardInterrupt,
    # MemoryError, a programming bug) must propagate, not be swallowed — per
    # the fail-fast rule.
    try:
        task = get_task(issue)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(
            f"[pod_lifecycle] WARN: upload-verification guard skipped for issue "
            f"#{issue}: could not resolve task ({type(exc).__name__}: {exc}). "
            f"Proceeding with terminate.",
            file=sys.stderr,
        )
        return

    # Fail SAFE on a missing/empty `kind`: default to "experiment" so an
    # unlabelled task still engages the guard (protects artifacts) rather than
    # silently skipping it. Only an explicit non-experiment kind opts out.
    kind = (task.get("frontmatter") or {}).get("kind") or "experiment"
    if kind != "experiment":
        return  # only experiments produce artifacts the verifier protects

    if _has_upload_verification_pass(issue):
        return

    if skip_flag:
        print(
            f"[pod_lifecycle] WARN: terminating {issue} WITHOUT an "
            f"epm:upload-verification PASS marker because --skip-upload-verify "
            f"was passed. Any unuploaded artifacts on this pod's volume "
            f"WILL be lost. Confirm uploads landed at their permanent "
            f"URLs (HF Hub model + data repos, WandB) before relying on "
            f"this run.",
            file=sys.stderr,
        )
        return

    raise SystemExit(
        f"Refusing to terminate the pod for task #{issue}: no "
        f"epm:upload-verification PASS marker on this experiment task. "
        f"The Step-8 upload-verifier protects against silent artifact "
        f"loss (training-mix datasets, raw completions, eval JSONs, "
        f"merged checkpoints not yet on HF Hub). Run the verifier first "
        f"via `/issue {issue}` Step 8, or pass --skip-upload-verify to "
        f"override (logs a warning + still terminates — only safe if "
        f"you've manually confirmed every artifact landed at its "
        f"permanent URL)."
    )


def _live_pods_for_issue(issue: int) -> list[PodInfo]:
    """All live RunPod pods whose managed name resolves to ``issue``.

    The live RunPod API is authoritative for pod existence and pod_id
    (CLAUDE.md "Authority split"). Local ``pods_ephemeral.json`` records ONE
    pod_id per issue, so the prior ``cmd_terminate`` path could see only one
    pod even when a second one (e.g. an EXITED orphan from a prior provision,
    or a duplicate created by an external dispatcher) was still on the
    account accruing volume charges. Matching by live-API name closes that
    gap: any pod whose name parses as ``pod-<issue>`` or the legacy
    ``epm-issue-<issue>`` is returned — regardless of ``desired_status``, so
    EXITED orphans are caught too. (Recurrence of the #365 stale-pod-id
    incident in #475: a stale local ``pod_id`` pointed at a ghost while a
    real RUNNING ``pod-475`` plus an EXITED orphan survived termination.)

    Name matching delegates to :func:`_issue_from_pod_name`, which parses the
    suffix after the managed prefix as an int and returns ``None`` on any
    non-numeric tail — so ``pod-47`` resolves to issue 47 and never matches
    issue 475.
    """
    return [p for p in list_team_pods() if _issue_from_pod_name(p.name) == issue]


def _terminate_clear_stale_sidecar(issue: int, *, dry_run: bool) -> None:
    """Handle the no-live-match branch of :func:`cmd_terminate`.

    The live API has no pod for ``issue``. If the local sidecar still names
    one (terminated externally; sidecar never reconciled), clear it so the
    next provision starts clean. If the sidecar is also empty, ``SystemExit``
    with a clear message rather than reporting a misleading 'Terminated' on a
    no-op.

    Reads the raw sidecar (not the merged ``_load_state`` view, which drops
    sidecar rows with no live-API match) so we can locate + clear the ghost.
    """
    # Contiguous read → pop → write under the sidecar lock (task #1183). The
    # pops are a legitimate removal — named in allow_remove so the never-drop
    # guard does not resurrect them.
    with _metadata_lock():
        sidecar_metadata = _read_metadata_file()
        stale_local_names = [name for name, m in sidecar_metadata.items() if m.issue == issue]
        if not stale_local_names:
            raise SystemExit(
                f"No live pod found for issue {issue} (and no local record). Nothing to terminate."
            )
        for name in stale_local_names:
            print(
                f"  No live pod found for issue {issue}; the local record "
                f"({name}, pod_id={sidecar_metadata[name].pod_id}) is stale. Clearing it.",
                file=sys.stderr,
            )
        if dry_run:
            print("[dry-run] Would clear stale local record(s).")
            return
        for name in stale_local_names:
            sidecar_metadata.pop(name, None)
        _write_metadata_file(sidecar_metadata, allow_remove=frozenset(stale_local_names))
    for name in stale_local_names:
        _remove_from_pods_conf(name)


def cmd_terminate(args: argparse.Namespace) -> None:
    """Destroy every live pod for issue #N. Volume(s) gone.

    The live RunPod API is authoritative for pod existence (CLAUDE.md
    "Authority split"). We terminate by the LIVE pod_id of every pod whose
    name resolves to this issue, then re-query and fail loud if any such pod
    survives. The local ``pods_ephemeral.json`` ``pod_id`` is a hint, not the
    authority — it can be stale when an external dispatcher (or a prior
    crashed provision) left a duplicate on the account.
    """
    # Refuse to destroy an experiment pod whose artifacts haven't been
    # upload-verified. Standard /issue Step 8 flow posts the PASS marker
    # BEFORE calling terminate, so the gate is silent on the happy path.
    # Pass --skip-upload-verify to override (logs a LOUD warning). Run the
    # guard once for the issue, BEFORE any live-API mutation.
    _guard_upload_verification_before_terminate(
        args.issue, skip_flag=args.skip_upload_verify, dry_run=args.dry_run
    )

    live_matches = _live_pods_for_issue(args.issue)
    if not live_matches:
        _terminate_clear_stale_sidecar(args.issue, dry_run=args.dry_run)
        return

    print(f"Terminating {len(live_matches)} live pod(s) for issue {args.issue}:")
    for p in live_matches:
        print(f"  {p.name}  pod_id={p.pod_id}  status={p.desired_status}")

    if not args.yes and not args.dry_run:
        confirm = input("  This DESTROYS the volume(s). Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    if args.dry_run:
        print("[dry-run] Would call terminate_pod on each.")
        return

    terminated_names: list[str] = []
    for p in live_matches:
        terminate_pod(p.pod_id)
        terminated_names.append(p.name)

    # Re-query the live API and fail loud if anything still resolves to this
    # issue. terminate_pod is async on RunPod's side — the pod may still
    # report RUNNING for a few seconds while RunPod tears it down — but a
    # DIFFERENT pod_id surviving means we missed a duplicate. Compare by
    # pod_id: any id we did NOT terminate is a real survivor.
    survivors = [
        p
        for p in _live_pods_for_issue(args.issue)
        if p.pod_id not in {q.pod_id for q in live_matches}
    ]
    if survivors:
        survivor_ids = [p.pod_id for p in survivors]
        raise RunPodError(
            f"terminate left {len(survivors)} live pod(s) for issue {args.issue}: "
            f"{survivor_ids}. Re-run `pod.py terminate --issue {args.issue}` "
            f"or terminate by id via the RunPod console."
        )

    # Drop terminated entries from metadata + pods.conf. Also clean any stale
    # local record whose name no longer matches a live pod (defensive: the
    # sidecar may have an extra row left over from a prior aborted run).
    # ``_load_state`` makes a live-API call, so it runs BEFORE the lock; the
    # sidecar read → pop → write is then one contiguous RMW under the lock
    # (task #1183), with every legitimate removal named in allow_remove.
    state = _load_state()  # post-terminate; live API has dropped the ids
    stale = _find_pod_in_state(state, args.issue)
    stale_name = stale.name if stale is not None and stale.name not in terminated_names else None
    with _metadata_lock():
        metadata = _read_metadata_file()
        for name in terminated_names:
            metadata.pop(name, None)
        if stale_name is not None:
            metadata.pop(stale_name, None)
        allow_remove = frozenset(terminated_names) | (
            frozenset({stale_name}) if stale_name is not None else frozenset()
        )
        _write_metadata_file(metadata, allow_remove=allow_remove)
    if stale_name is not None:
        _remove_from_pods_conf(stale_name)
    for name in terminated_names:
        _remove_from_pods_conf(name)
    print(
        f"  Terminated {len(terminated_names)} pod(s). "
        f"Removed from pods.conf and pods_ephemeral.json."
    )


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
    p.add_argument(
        "--wait-for-capacity",
        action="store_true",
        help=(
            "On SUPPLY_CONSTRAINT (every supply lever in create_pod returned "
            "null), keep retrying with exponential-jittered backoff (base 30s, "
            "cap 10 min) instead of failing. Each PROCESS attempt is capped at "
            "~45 min wall-clock (EPM_WAIT_FOR_CAPACITY_BUDGET_SECS); at the "
            f"budget the command exits {EXIT_STILL_WAITING} with a structured "
            "[wait-for-capacity] STILL-WAITING line — re-run the same command "
            "to continue waiting (state-free). Other errors still fail "
            "fast. Auto-enabled when EPM_AUTONOMOUS_SESSION=1 (autonomous "
            "sessions wait for capacity rather than park, per CLAUDE.md). "
            "Default OFF so interactive provisions still surface no-capacity "
            "immediately. The /issue orchestrator should surface the "
            "[wait-for-capacity] stderr heartbeats as epm:progress markers "
            "so autonomous_session_watch.py sees liveness."
        ),
    )
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
    p.add_argument(
        "--wait-for-capacity",
        action="store_true",
        help=(
            "On an account $/hr-cap refusal (local pre-flight estimate or "
            "RunPod-side INSUFFICIENT_BALANCE), keep retrying with "
            "exponential-jittered backoff (base 30s, cap 10 min) until a "
            "sibling pod frees headroom, instead of failing immediately. "
            "Each PROCESS attempt is capped at ~45 min wall-clock "
            "(EPM_WAIT_FOR_CAPACITY_BUDGET_SECS); at the budget the command "
            f"exits {EXIT_STILL_WAITING} with a structured STILL-WAITING "
            "line — re-run the same command to continue waiting. "
            "Auto-enabled when EPM_AUTONOMOUS_SESSION=1. Default OFF so "
            "interactive resumes still surface an immediate, actionable "
            "refusal. SUPPLY_CONSTRAINT still fails fast in both modes — "
            "resume never relocates a pod, so waiting cannot help when its "
            "original host is out of GPUs."
        ),
    )
    p.set_defaults(func=cmd_resume)


def _parser_terminate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("terminate", help="Destroy an issue's pod (volume goes too)")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-upload-verify",
        action="store_true",
        help=(
            "Terminate even without an epm:upload-verification PASS marker on "
            "the task (logs a LOUD warning, still proceeds). Only safe if "
            "you've manually confirmed every artifact landed at its permanent "
            "URL on HF Hub / WandB. The normal /issue Step 8 flow posts the "
            "PASS marker before terminate, so the guard is silent on the "
            "happy path."
        ),
    )
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


# Verbs that detach into their own session in autonomous mode (long-running,
# billing-relevant; a respawn-kill mid-flight orphans a paid pod).
_SETSID_VERBS = frozenset({"provision", "resume"})


def _maybe_detach_into_own_session(verb: str | None) -> None:
    """Detach provision/resume into their OWN session (``os.setsid``) when
    running inside an autonomous /issue session (refs #573).

    Why: the stalled-detector's stop-then-respawn (and the 12h session
    recycle) kills the old session's process group; on 2026-06-09 that
    killed in-flight ``pod.py provision`` background commands three times on
    #534 (~8h lost) and sent 3 morning sessions dark. ``os.setsid`` moves
    this process out of the doomed process group, so a group-targeted kill
    no longer reaches it, while the parent-child relationship, stdio pipes
    (bg-Bash output capture), and exit-code delivery are all unchanged.

    Scope guards:
    - only the long-running, billing-relevant verbs (:data:`_SETSID_VERBS`);
    - only in autonomous mode (``EPM_AUTONOMOUS_SESSION=1``) — interactive
      shells keep normal Ctrl-C / job-control semantics;
    - ``EPM_NO_SETSID=1`` opts out (debug escape hatch);
    - fail-soft: ``os.setsid`` raises ``OSError`` when the process is
      already a process-group leader — log and continue in that case.
    """
    if verb not in _SETSID_VERBS or not _autonomous_session():
        return
    if os.environ.get("EPM_NO_SETSID", "").strip() == "1":
        return
    try:
        os.setsid()
        print(
            f"[pod_lifecycle] {verb}: detached into own session (setsid) so a "
            f"session respawn's process-group kill can't kill this in-flight "
            f"{verb} (refs #573).",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"[pod_lifecycle] WARN: setsid failed ({exc}); {verb} stays in the "
            f"caller's process group (a respawn-kill may still reach it).",
            file=sys.stderr,
        )


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
    _maybe_detach_into_own_session(getattr(args, "cmd", None))
    args.func(args)


if __name__ == "__main__":
    main()
