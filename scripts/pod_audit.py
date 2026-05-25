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
- **fresh-exited**: ``EXITED`` but younger than threshold. Probably a pod
  that just stopped and is about to be terminated by its owning flow — ignore.

Exit codes::

    0  clean (no orphans, no stale)
    2  audit found stale and/or orphan-running pods

The ``--terminate-stale`` flag terminates every pod in the ``stale`` bucket
after a y/N confirmation (suppress with ``--yes``). ``orphan-running`` pods
are NEVER auto-terminated — they may be a real in-flight workload outside
the lifecycle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from runpod_api import PodInfo, list_team_pods, terminate_pod  # noqa: E402

DEFAULT_MAX_EXITED_HOURS = 24
DEFAULT_MIN_ORPHAN_RUNNING_HOURS = 1  # below this, a running pod may still be in bootstrap


@dataclass(frozen=True)
class Classification:
    pod: PodInfo
    bucket: str  # active | orphan-running | stale | fresh-exited
    age_hours: float | None
    referenced_in_tasks: list[int]


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


def classify(
    pods: list[PodInfo],
    *,
    max_exited_hours: float,
    min_orphan_running_hours: float,
) -> list[Classification]:
    out: list[Classification] = []
    for p in pods:
        age = _age_hours(p.created_at)
        refs = _scan_task_references(p.pod_id, p.name)
        if p.desired_status == "RUNNING":
            if _is_managed_name(p.name) or refs:
                bucket = "active"
            elif age is None or age >= min_orphan_running_hours:
                bucket = "orphan-running"
            else:
                bucket = "active"  # too young to flag
        elif p.desired_status == "EXITED":
            if age is not None and age >= max_exited_hours:
                bucket = "stale"
            else:
                bucket = "fresh-exited"
        else:
            bucket = f"other:{p.desired_status}"
        out.append(Classification(pod=p, bucket=bucket, age_hours=age, referenced_in_tasks=refs))
    return out


def render_report(rows: list[Classification]) -> str:
    by_bucket: dict[str, list[Classification]] = {}
    for r in rows:
        by_bucket.setdefault(r.bucket, []).append(r)

    lines: list[str] = []
    total = len(rows)
    lines.append(f"Total team pods: {total}")
    for bucket in ("active", "orphan-running", "stale", "fresh-exited"):
        n = len(by_bucket.get(bucket, []))
        if n:
            lines.append(f"  {bucket:18}  {n}")
    other_buckets = {
        k: v
        for k, v in by_bucket.items()
        if not k.startswith(("active", "orphan", "stale", "fresh"))
    }
    for bucket, items in sorted(other_buckets.items()):
        lines.append(f"  {bucket:18}  {len(items)}")

    for bucket in ("orphan-running", "stale", "fresh-exited", "active"):
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
            gpu = f"{r.pod.gpu_count}x{r.pod.gpu_type_id}" if r.pod.gpu_count else ""
            lines.append(
                f"  {r.pod.pod_id}  {r.pod.desired_status:8}  age={age:>7}  "
                f"{gpu:30}  {r.pod.name!r}{refs}"
            )
    return "\n".join(lines)


def cmd_audit(args: argparse.Namespace) -> int:
    pods = list_team_pods()
    rows = classify(
        pods,
        max_exited_hours=args.max_exited_hours,
        min_orphan_running_hours=args.min_orphan_running_hours,
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
        help=f"EXITED pods older than this many hours are 'stale' (default: {DEFAULT_MAX_EXITED_HOURS})",
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
