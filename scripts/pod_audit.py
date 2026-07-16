"""Audit live RunPod team account for stale/orphaned pods.

Catches pods that the canonical lifecycle (``pod_lifecycle.py``) is blind to
because their names don't match the managed prefixes (``pod-*`` /
``epm-issue-*``). Such pods are created when dispatcher scripts call
``runpod_api.create_pod()`` directly with a custom name, or when a developer
provisions a pod manually outside the ``/issue`` flow.

The live API is authoritative — we never trust local sidecar state for
existence. A pod is:

- **active**: ``RUNNING`` AND managed-name (the lifecycle owns it).
- **orphan-running**: ``RUNNING`` AND non-managed-name. GPU charges accruing
  without lifecycle tracking — surface loudly.
- **stale**: ``EXITED`` for longer than ``--max-exited-hours`` (default 24h).
  Volume disk charges accruing for paused state. Candidate for termination.
- **kept-exited**: ``EXITED`` but the owning task (resolved from the managed
  pod name ``pod-<N>`` / ``epm-issue-<N>``) carries the ``keep-running`` tag —
  the workflow's documented pod-preservation override (CLAUDE.md, /issue
  Step 8). Reported loudly but NEVER terminated by ``--terminate-stale``,
  regardless of age.
- **fresh-exited**: ``EXITED`` but younger than threshold. Probably a pod
  that just stopped and is about to be terminated by its owning flow — ignore.

Two additional REPORT-ONLY flag classes annotate the buckets. They never
change bucketing, exit codes, or ``--terminate-stale`` behavior — the audit
is the fleet-level safety net that works even when a run's driver session
and poller are both dead, so the two most expensive waste patterns must at
least be VISIBLE in it (incident 2026-06-10: #518/#537 RUNNING 8xH100 pods
idle for hours on healthy CPU-only phases; pod-530 stopped-but-billing on a
task parked at awaiting_promotion):

- **idle-gpu**: a RUNNING managed pod whose GPUs ALL read 0% utilization at
  audit time (single ``nvidia-smi`` point sample over SSH — NOT proof of
  sustained idleness; the audit runs daily, so a repeat flag is the signal).
  Any SSH/parse failure → ``util=unknown``, never flagged (fail-safe).
- **stopped-on-parked-task**: an ``EXITED`` pod whose owning task has sat at
  a parked/terminal status (``awaiting_promotion`` / ``blocked`` /
  ``completed`` / ``archived``) for longer than ``--min-parked-hours``
  (default 24). The stopped volume keeps billing; surfaced as a termination
  candidate for the USER — never auto-terminated by this audit.

Exit codes::

    0  clean (no orphans, no stale)
    2  audit found stale and/or orphan-running pods

(The report-only flag classes deliberately do NOT affect the exit code —
``cron_pod_audit.sh`` treats the log as the audit trail and an idle CPU
phase is not an audit failure.)

The ``--terminate-stale`` flag terminates every pod in the ``stale`` bucket
after a y/N confirmation (suppress with ``--yes``). ``orphan-running`` pods
are NEVER auto-terminated — they may be a real in-flight workload outside
the lifecycle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from pod_lifecycle import _issue_from_pod_name as _lifecycle_issue_from_pod_name  # noqa: E402
from runpod_api import (  # noqa: E402
    PodInfo,
    estimate_pod_hourly_rate,
    list_team_pods,
    terminate_pod,
)

from explore_persona_space.task_workflow import get_task, repo_root, tasks_dir  # noqa: E402

DEFAULT_MAX_EXITED_HOURS = 24
DEFAULT_MIN_ORPHAN_RUNNING_HOURS = 1  # below this, a running pod may still be in bootstrap
DEFAULT_MIN_PARKED_HOURS = 24  # parked-task duration before an EXITED pod is flagged
# A managed RUNNING pod with null/empty runtime.ports older than this is the #664
# RUNNING-but-no-port host wedge — an unreachable billing leak. Report-only here
# (the inputs-on-HF-gated auto-recovery belongs to backend_poll, fix (b)); short
# floor so a healthy mid-bootstrap pod (port appears within ~minutes) is not flagged.
DEFAULT_MIN_NO_PORT_HOURS = 0.5

# Task statuses where no further pod work is expected: the run is parked for a
# user decision or terminally done, so a stopped pod's volume is pure billing.
PARKED_STATUSES = frozenset({"awaiting_promotion", "blocked", "completed", "archived"})

SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
GPU_UTIL_SSH_TIMEOUT = 20  # seconds; one short read per RUNNING managed pod


@dataclass(frozen=True)
class TaskContext:
    """Fail-soft snapshot of a pod's owning task — every field may be None.

    ``parked_age_hours`` is hours since the task's last ``epm:status-changed``
    event (i.e. how long it has sat at its CURRENT status); ``None`` when the
    task has no status-changed marker or events.jsonl is unreadable.
    ``last_marker_age_hours`` is hours since the last event of any kind.
    """

    status: str | None = None
    parked_age_hours: float | None = None
    last_marker_age_hours: float | None = None


@dataclass(frozen=True)
class Classification:
    pod: PodInfo
    bucket: str  # active | orphan-running | stale | kept-exited | fresh-exited
    age_hours: float | None
    referenced_in_tasks: list[int]
    kept_for_task: int | None = None  # task whose keep-running tag preserved this pod
    # ── report-only annotations (never change bucketing / terminate behavior) ──
    owning_issue: int | None = None  # parsed from the managed pod name
    task_status: str | None = None  # owning task's current status (None = unknown)
    parked_age_hours: float | None = None  # hours at current status (None = unknown)
    last_marker_age_hours: float | None = None  # hours since last epm:* event
    gpu_util: list[int] | None = None  # per-GPU util %, point sample; None = unknown
    idle_gpu: bool = False  # RUNNING managed pod, util read OK, ALL GPUs at 0%
    stopped_on_parked_task: bool = False  # EXITED pod on a long-parked task
    running_no_port: bool = False  # RUNNING managed pod, null ports, age >= floor (#664 wedge)


def _parse_iso(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _age_hours(ts: str | None) -> float | None:
    parsed = _parse_iso(ts)
    if parsed is None:
        return None
    delta = dt.datetime.now(dt.UTC) - parsed
    return delta.total_seconds() / 3600.0


def _scan_task_references(pod_id: str, pod_name: str) -> list[int]:
    """Return list of task numbers whose events.jsonl mentions this pod."""
    td = tasks_dir()
    if not td.exists():
        return []
    hits: list[int] = []
    needles = (pod_id, pod_name)
    for events_path in td.glob("*/*/events.jsonl"):
        try:
            blob = events_path.read_text(errors="ignore")
        except OSError:
            continue
        if any(n in blob for n in needles):
            try:
                task_id = int(events_path.parent.name)
            except ValueError:
                continue
            hits.append(task_id)
    return sorted(set(hits))


def _is_managed_name(name: str) -> bool:
    return name.startswith("pod-") or name.startswith("epm-issue-")


def _issue_number_from_name(name: str) -> int | None:
    """Parse the owning issue from a managed pod name.

    Thin delegation to the canonical grammar in
    ``pod_lifecycle._issue_from_pod_name`` (#1334) — one parser for the
    lifecycle, the watcher, and the audit. Deliberate delta from the old
    ``split('-', 1)`` parse: a NUMERIC slug (``pod-779-60``) no longer
    maps (letter-initial slugs only); such a pod falls through to the
    normal unmanaged/stale logic instead of a guessy attribution.
    """
    return _lifecycle_issue_from_pod_name(name)


def _task_has_keep_running(issue: int) -> bool:
    """True when task ``issue`` carries the ``keep-running`` tag.

    Fail-soft by design: any lookup failure (missing task, unreadable
    registry/body, resolver refusal) returns ``False`` so the exemption can
    never crash the audit or silently keep an orphan — the pod falls through
    to the normal stale logic.
    """
    try:
        fm = get_task(issue).get("frontmatter") or {}
        return "keep-running" in (fm.get("tags") or [])
    except Exception:
        return False


def _task_context(issue: int) -> TaskContext:
    """Resolve the owning task's status + marker ages for the report-only flags.

    Fail-soft by design (same contract as :func:`_task_has_keep_running`): any
    lookup failure returns an all-None :class:`TaskContext`, so a missing /
    legacy / unreadable task can never crash the audit — the pod simply isn't
    flagged and the normal bucket logic stands.
    """
    try:
        snap = get_task(issue)
    except Exception:
        return TaskContext()
    status_changed_ts: str | None = None
    last_ts: str | None = None
    try:
        events_path = repo_root() / snap["path"] / "events.jsonl"
        # split("\n"), NOT splitlines(): raw U+2028/U+2029/NEL inside
        # ensure_ascii=False notes are Unicode line boundaries that would
        # shred valid records — a shredded epm:status-changed line silently
        # loses the status timestamp (gotchas.md; #825 → #950).
        for line in events_path.read_text(errors="ignore").split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("ts")
            if not ts:
                continue
            last_ts = ts
            if ev.get("kind") == "epm:status-changed":
                status_changed_ts = ts
    except Exception:
        pass  # events unreadable → ages stay None → never flagged (fail-soft)
    return TaskContext(
        status=snap.get("status"),
        parked_age_hours=_age_hours(status_changed_ts),
        last_marker_age_hours=_age_hours(last_ts),
    )


def _probe_gpu_util(pod: PodInfo) -> list[int] | None:
    """Point-sample GPU utilization (%) on a RUNNING pod via SSH + nvidia-smi.

    Returns one int per GPU, or ``None`` whenever the sample could not be
    taken (no public SSH endpoint on the live-API snapshot, connect failure,
    nonzero exit, unparseable output). Callers MUST treat ``None`` as
    *unknown*, never as idle — the flag fails SAFE. Read-only; SSH endpoint
    comes from the live API (``PodInfo.ssh_host``/``ssh_port``), not
    ``pods.conf``, which can go stale across resumes (incident #488).
    """
    if not pod.ssh_host or not pod.ssh_port:
        return None
    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "BatchMode=yes",
        "-i",
        str(SSH_KEY),
        "-p",
        str(pod.ssh_port),
        f"root@{pod.ssh_host}",
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=GPU_UTIL_SSH_TIMEOUT)
    except Exception:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return [int(line.strip()) for line in r.stdout.strip().splitlines()]
    except ValueError:
        return None


def classify(
    pods: list[PodInfo],
    *,
    max_exited_hours: float,
    min_orphan_running_hours: float,
    min_parked_hours: float = DEFAULT_MIN_PARKED_HOURS,
    min_no_port_hours: float = DEFAULT_MIN_NO_PORT_HOURS,
) -> list[Classification]:
    out: list[Classification] = []
    for p in pods:
        age = _age_hours(p.created_at)
        refs = _scan_task_references(p.pod_id, p.name)
        kept_for: int | None = None
        issue = _issue_number_from_name(p.name)
        ctx = _task_context(issue) if issue is not None else TaskContext()
        gpu_util: list[int] | None = None
        idle_gpu = False
        stopped_on_parked = False
        running_no_port = False
        if p.desired_status == "RUNNING":
            if _is_managed_name(p.name) or refs:
                bucket = "active"
            elif age is None or age >= min_orphan_running_hours:
                bucket = "orphan-running"
            else:
                bucket = "active"  # too young to flag
            if _is_managed_name(p.name):
                # Report-only idle-GPU flag: single point sample; util=None
                # (SSH/parse failure) is 'unknown' and NEVER flagged.
                gpu_util = _probe_gpu_util(p)
                idle_gpu = (
                    gpu_util is not None and len(gpu_util) > 0 and all(u == 0 for u in gpu_util)
                )
                # Report-only #664 RUNNING-but-no-port wedge flag: a managed pod
                # RUNNING with null/empty public port for >= the floor is an
                # unreachable billing leak host-pinned resume cannot heal. NEVER
                # auto-terminated here — the inputs-on-HF-gated auto-recovery is
                # backend_poll's (fix (b)); this only makes the wedge VISIBLE in
                # the fleet-level audit (bucket stays 'active', exit code unchanged).
                running_no_port = (
                    not (p.ssh_host and p.ssh_port) and age is not None and age >= min_no_port_hours
                )
        elif p.desired_status == "EXITED":
            if issue is not None and _task_has_keep_running(issue):
                # keep-running tag is THE documented pod-preservation override
                # (CLAUDE.md, /issue Step 8) — never auto-terminate, however old.
                bucket = "kept-exited"
                kept_for = issue
            elif age is not None and age >= max_exited_hours:
                bucket = "stale"
            else:
                bucket = "fresh-exited"
            # Report-only parked-task flag: the stopped volume keeps billing
            # while the owning task sits parked/terminal. Unknown status or
            # unknown parked-age (no status-changed marker) is never flagged.
            stopped_on_parked = (
                ctx.status in PARKED_STATUSES
                and ctx.parked_age_hours is not None
                and ctx.parked_age_hours >= min_parked_hours
            )
        else:
            bucket = f"other:{p.desired_status}"
        out.append(
            Classification(
                pod=p,
                bucket=bucket,
                age_hours=age,
                referenced_in_tasks=refs,
                kept_for_task=kept_for,
                owning_issue=issue,
                task_status=ctx.status,
                parked_age_hours=ctx.parked_age_hours,
                last_marker_age_hours=ctx.last_marker_age_hours,
                gpu_util=gpu_util,
                idle_gpu=idle_gpu,
                stopped_on_parked_task=stopped_on_parked,
                running_no_port=running_no_port,
            )
        )
    return out


def render_report(rows: list[Classification]) -> str:
    by_bucket: dict[str, list[Classification]] = {}
    for r in rows:
        by_bucket.setdefault(r.bucket, []).append(r)

    lines: list[str] = []
    total = len(rows)
    lines.append(f"Total team pods: {total}")
    for bucket in ("active", "orphan-running", "stale", "kept-exited", "fresh-exited"):
        n = len(by_bucket.get(bucket, []))
        if n:
            lines.append(f"  {bucket:18}  {n}")
    other_buckets = {
        k: v
        for k, v in by_bucket.items()
        if not k.startswith(("active", "orphan", "stale", "kept", "fresh"))
    }
    for bucket, items in sorted(other_buckets.items()):
        lines.append(f"  {bucket:18}  {len(items)}")
    n_idle = sum(1 for r in rows if r.idle_gpu)
    n_parked = sum(1 for r in rows if r.stopped_on_parked_task)
    n_no_port = sum(1 for r in rows if r.running_no_port)
    if n_idle:
        lines.append(f"  idle-gpu            {n_idle}  (report-only flag)")
    if n_parked:
        lines.append(f"  stopped-on-parked   {n_parked}  (report-only flag)")
    if n_no_port:
        lines.append(f"  running-no-port     {n_no_port}  (report-only flag)")

    for bucket in ("orphan-running", "stale", "kept-exited", "fresh-exited", "active"):
        items = sorted(
            by_bucket.get(bucket, []),
            key=lambda r: r.age_hours or 0.0,
            reverse=True,
        )
        if not items:
            continue
        lines.append("")
        lines.append(f"── {bucket} ──")
        for r in items:
            age = f"{r.age_hours:.1f}h" if r.age_hours is not None else "?"
            refs = (
                f"  task #{','.join(str(t) for t in r.referenced_in_tasks)}"
                if r.referenced_in_tasks
                else ""
            )
            kept = (
                f"  KEPT: keep-running tag on task #{r.kept_for_task} — never auto-terminated"
                if r.kept_for_task is not None
                else ""
            )
            gpu = f"{r.pod.gpu_count}x{r.pod.gpu_type_id}" if r.pod.gpu_count else ""
            lines.append(
                f"  {r.pod.pod_id}  {r.pod.desired_status:8}  age={age:>7}  "
                f"{gpu:30}  {r.pod.name!r}{refs}{kept}"
            )

    lines.extend(_render_flag_sections(rows))
    return "\n".join(lines)


def _fmt_task_ctx(r: Classification) -> str:
    """Render the owning-task context fragment for a report-only flag line."""
    if r.owning_issue is None:
        return ""
    status = r.task_status or "unknown"
    frag = f"  task #{r.owning_issue} status={status}"
    if r.last_marker_age_hours is not None:
        frag += f"  last-marker {r.last_marker_age_hours:.1f}h ago"
    return frag


def _render_flag_sections(rows: list[Classification]) -> list[str]:
    """Render the REPORT-ONLY flag sections (idle-gpu, stopped-on-parked-task,
    running-no-port).

    Returns [] when nothing is flagged; never affects buckets or exit codes.
    """
    lines: list[str] = []
    idle = [r for r in rows if r.idle_gpu]
    if idle:
        lines.append("")
        lines.append("── idle-gpu (report-only) ──")
        lines.append("  GPU util 0% at audit time — a single nvidia-smi point sample, NOT proof")
        lines.append("  of sustained idleness. A healthy CPU-only phase looks identical; a")
        lines.append("  repeat flag across daily audits is the real signal.")
        for r in idle:
            gpu = f"{r.pod.gpu_count}x{r.pod.gpu_type_id}" if r.pod.gpu_count else "?"
            rate = estimate_pod_hourly_rate(r.pod.gpu_type_id, r.pod.gpu_count)
            util = ",".join(str(u) for u in (r.gpu_util or []))
            lines.append(
                f"  {r.pod.pod_id}  {gpu:30}  ~${rate:.1f}/hr (estimate)  "
                f"util=[{util}]  {r.pod.name!r}{_fmt_task_ctx(r)}"
            )
    parked = [r for r in rows if r.stopped_on_parked_task]
    if parked:
        lines.append("")
        lines.append("── stopped-on-parked-task (report-only) ──")
        lines.append("  EXITED pod whose owning task has been parked/terminal for longer than")
        lines.append("  the threshold — the stopped volume keeps billing. Termination candidate")
        lines.append("  for the USER; this audit never auto-terminates these.")
        for r in parked:
            parked_h = f"{r.parked_age_hours:.1f}h" if r.parked_age_hours is not None else "?"
            lines.append(
                f"  {r.pod.pod_id}  {r.pod.desired_status:8}  {r.pod.name!r}"
                f"{_fmt_task_ctx(r)}  parked {parked_h}"
            )
    no_port = [r for r in rows if r.running_no_port]
    if no_port:
        lines.append("")
        lines.append("── running-no-port (report-only, #664 host wedge) ──")
        lines.append("  RUNNING managed pod with NO public port past the floor — host-pinned")
        lines.append("  resume cannot heal it (the #664 wedge), so it is an unreachable billing")
        lines.append("  leak. This audit never auto-terminates these; the inputs-on-HF-gated")
        lines.append("  auto-recovery is backend_poll's (fix (b)).")
        for r in no_port:
            age = f"{r.age_hours:.1f}h" if r.age_hours is not None else "?"
            gpu = f"{r.pod.gpu_count}x{r.pod.gpu_type_id}" if r.pod.gpu_count else "?"
            rate = estimate_pod_hourly_rate(r.pod.gpu_type_id, r.pod.gpu_count)
            lines.append(
                f"  {r.pod.pod_id}  {gpu:30}  ~${rate:.1f}/hr (estimate)  age={age:>7}  "
                f"{r.pod.name!r}{_fmt_task_ctx(r)}"
            )
    return lines


def cmd_audit(args: argparse.Namespace) -> int:
    pods = list_team_pods()
    rows = classify(
        pods,
        max_exited_hours=args.max_exited_hours,
        min_orphan_running_hours=args.min_orphan_running_hours,
        min_parked_hours=args.min_parked_hours,
        min_no_port_hours=args.min_no_port_hours,
    )

    if args.json:
        payload = [
            {
                "pod_id": r.pod.pod_id,
                "name": r.pod.name,
                "desired_status": r.pod.desired_status,
                "bucket": r.bucket,
                "age_hours": r.age_hours,
                "gpu_count": r.pod.gpu_count,
                "gpu_type_id": r.pod.gpu_type_id,
                "created_at": r.pod.created_at,
                "referenced_in_tasks": r.referenced_in_tasks,
                "kept_for_task": r.kept_for_task,
                # report-only flag annotations (never affect bucket/exit code)
                "owning_issue": r.owning_issue,
                "task_status": r.task_status,
                "parked_age_hours": r.parked_age_hours,
                "last_marker_age_hours": r.last_marker_age_hours,
                "gpu_util": r.gpu_util,
                "idle_gpu": r.idle_gpu,
                "stopped_on_parked_task": r.stopped_on_parked_task,
                "running_no_port": r.running_no_port,
                "est_hourly_usd": estimate_pod_hourly_rate(r.pod.gpu_type_id, r.pod.gpu_count),
            }
            for r in rows
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(rows))

    stale = [r for r in rows if r.bucket == "stale"]
    orphans = [r for r in rows if r.bucket == "orphan-running"]

    if args.terminate_stale and stale:
        if not args.yes:
            ans = input(f"\nTerminate {len(stale)} stale pod(s)? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborted; no pods terminated.")
                return 2
        print(f"\nTerminating {len(stale)} stale pod(s)...")
        failed: list[str] = []
        for r in stale:
            try:
                terminate_pod(r.pod.pod_id)
                print(f"  ok   {r.pod.pod_id}  {r.pod.name}")
            except Exception as e:
                failed.append(r.pod.pod_id)
                print(f"  FAIL {r.pod.pod_id}  {r.pod.name}  err={e!s:.120}")
        if failed:
            print(f"\n{len(failed)} terminate(s) failed.")
            return 2

    if orphans:
        print(
            "\nNOTE: orphan-running pods are NOT auto-terminated — they may be a "
            "real in-flight workload spun up outside the canonical lifecycle. "
            "Investigate manually.",
            file=sys.stderr,
        )

    kept = [r for r in rows if r.bucket == "kept-exited"]
    if kept:
        names = ", ".join(f"{r.pod.name} (task #{r.kept_for_task})" for r in kept)
        print(
            f"\nNOTE: kept-exited pods preserved by their task's keep-running tag: {names}. "
            "Remove the tag (task.py remove-tag <N> keep-running) to let the audit "
            "reclaim them.",
            file=sys.stderr,
        )

    return 2 if (stale or orphans) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pod_audit",
        description="Audit live RunPod team account for stale/orphaned pods.",
    )
    p.add_argument(
        "--max-exited-hours",
        type=float,
        default=DEFAULT_MAX_EXITED_HOURS,
        help=(
            f"EXITED pods older than this many hours are 'stale' "
            f"(default: {DEFAULT_MAX_EXITED_HOURS})"
        ),
    )
    p.add_argument(
        "--min-orphan-running-hours",
        type=float,
        default=DEFAULT_MIN_ORPHAN_RUNNING_HOURS,
        help=(
            f"RUNNING pods younger than this are not flagged as orphans "
            f"(default: {DEFAULT_MIN_ORPHAN_RUNNING_HOURS}) — gives bootstrap a window."
        ),
    )
    p.add_argument(
        "--min-parked-hours",
        type=float,
        default=DEFAULT_MIN_PARKED_HOURS,
        help=(
            f"EXITED pods whose owning task has been parked/terminal "
            f"(awaiting_promotion/blocked/completed/archived) longer than this many "
            f"hours get the report-only 'stopped-on-parked-task' flag "
            f"(default: {DEFAULT_MIN_PARKED_HOURS})."
        ),
    )
    p.add_argument(
        "--min-no-port-hours",
        type=float,
        default=DEFAULT_MIN_NO_PORT_HOURS,
        help=(
            f"RUNNING managed pods with null/empty public ports older than this many "
            f"hours get the report-only 'running-no-port' flag (the #664 host wedge) "
            f"(default: {DEFAULT_MIN_NO_PORT_HOURS})."
        ),
    )
    p.add_argument(
        "--terminate-stale",
        action="store_true",
        help="Terminate every pod in the 'stale' bucket (asks y/N unless --yes).",
    )
    p.add_argument(
        "--yes", action="store_true", help="Skip y/N confirmation for --terminate-stale."
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON, no headers.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_audit(args)


if __name__ == "__main__":
    sys.exit(main())
