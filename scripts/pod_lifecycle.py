"""Ephemeral pod lifecycle: provision, stop, resume, terminate, cleanup.

How it fits with the rest of the pod tooling
--------------------------------------------
- `runpod_api.py` is the GraphQL transport. Always team-scoped.
- `gpu_heuristics.py` maps experiment intents to GPU specs.
- `pods.conf` holds connection info for SSH/MCP config generation. We append /
  update / remove rows here so pods provisioned by this script become reachable
  via `ssh epm-issue-NNN` after a `pod_config.py --sync`.
- `pods_ephemeral.json` (sidecar) holds the *metadata* about ephemeral pods —
  the RunPod id, the source issue, lifecycle timestamps, TTL. This is the
  source of truth for "is this pod stopped / running / due-for-termination."

Naming convention
-----------------
Ephemeral pods are named `epm-issue-<N>` where `<N>` is the GitHub issue number.
One pod per issue. Follow-up issues that derive from #N can resume #N's pod.

The bootstrap step is gated by an --no-bootstrap flag because resumed pods
already have the repo + caches; you only bootstrap on first provision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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
class EphemeralPod:
    name: str  # e.g. "epm-issue-125"
    pod_id: str  # RunPod id
    issue: int  # source issue number
    gpu_intent: str  # the intent string used (or "custom")
    gpu_type: str  # H100 | H200 | A100
    gpu_count: int
    status: str  # "running" | "stopped" | "terminated"
    created_at: str  # ISO 8601
    ttl_days: int = DEFAULT_TTL_DAYS
    stopped_at: str | None = None
    host: str | None = None
    port: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _load_state() -> dict[str, EphemeralPod]:
    if not EPHEMERAL_STATE.exists():
        return {}
    raw = json.loads(EPHEMERAL_STATE.read_text())
    pods: dict[str, EphemeralPod] = {}
    for name, payload in raw.get("pods", {}).items():
        # Tolerate unknown keys for forward compatibility.
        known = {f.name for f in EphemeralPod.__dataclass_fields__.values()}
        clean = {k: v for k, v in payload.items() if k in known}
        clean.setdefault("name", name)
        pods[name] = EphemeralPod(**clean)
    return pods


def _save_state(pods: dict[str, EphemeralPod]) -> None:
    payload = {
        "version": 1,
        "updated_at": _now(),
        "pods": {name: asdict(p) for name, p in pods.items()},
    }
    EPHEMERAL_STATE.write_text(json.dumps(payload, indent=2) + "\n")


# ─── pods.conf side effects ──────────────────────────────────────────────────


def _label_for_issue(issue: int) -> str:
    return f"thomas-epm-issue-{issue}"


def _upsert_pods_conf(pod: EphemeralPod) -> None:
    """Add or update `pod` in scripts/pods.conf and regenerate downstream configs."""
    rows = parse_pods_conf()
    existing = next((p for p in rows if p.name == pod.name), None)
    if pod.host is None or pod.port is None:
        # Nothing to write yet — only happens during transient provisioning.
        return
    if existing:
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
    """Run the existing bootstrap_pod.sh against an `epm-issue-N` pod entry."""
    print(f"\nRunning bootstrap on {pod_name}...")
    return subprocess.call(
        ["bash", str(BOOTSTRAP_SCRIPT), pod_name],
        cwd=str(PROJECT_ROOT),
    )


# ─── commands ────────────────────────────────────────────────────────────────


def cmd_provision(args: argparse.Namespace) -> None:
    """Create a fresh pod for issue #N, wait for SSH, register it, bootstrap it."""
    if args.list_intents:
        print(list_intents())
        return

    if args.issue is None:
        raise SystemExit("--issue <N> is required")

    name = f"epm-issue-{args.issue}"
    state = _load_state()

    # Idempotency: if there's already a non-terminated pod for this issue, refuse.
    if name in state and state[name].status != "terminated":
        existing = state[name]
        print(
            f"Pod {name} already exists (status={existing.status}, id={existing.pod_id}).\n"
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

    pod = EphemeralPod(
        name=name,
        pod_id=info.pod_id,
        issue=args.issue,
        gpu_intent=intent_label,
        gpu_type=spec.gpu_type,
        gpu_count=spec.gpu_count,
        status="running",
        created_at=_now(),
        ttl_days=args.ttl_days,
        host=ready.ssh_host,
        port=ready.ssh_port,
    )
    state[name] = pod
    _save_state(state)
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
    name = f"epm-issue-{args.issue}"
    if name not in state:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    pod = state[name]
    if pod.status == "stopped":
        print(f"{name} already stopped.")
        return
    if pod.status == "terminated":
        raise SystemExit(f"{name} is terminated; can't stop a destroyed pod.")

    print(f"Stopping {name} (pod_id={pod.pod_id})...")
    if args.dry_run:
        print("[dry-run] Would call stop_pod.")
        return
    stop_pod(pod.pod_id)
    pod.status = "stopped"
    pod.stopped_at = _now()
    pod.host = None
    pod.port = None
    state[name] = pod
    _save_state(state)
    # Note: leave the pods.conf row in place so the user can see it; it'll just
    # fail to SSH until resumed. Removing+adding causes more churn.
    print(
        f"  Stopped. Will auto-terminate after {pod.ttl_days} days idle "
        f"(stopped_at={pod.stopped_at})."
    )


def cmd_resume(args: argparse.Namespace) -> None:
    """Bring a stopped pod back. New IP, same volume."""
    state = _load_state()
    name = f"epm-issue-{args.issue}"
    if name not in state:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    pod = state[name]
    if pod.status == "running":
        print(f"{name} is already running.")
        return
    if pod.status == "terminated":
        raise SystemExit(
            f"{name} is terminated. The volume is gone — you'll need to "
            f"`pod.py provision --issue {args.issue}` from scratch."
        )

    print(f"Resuming {name} (pod_id={pod.pod_id}, gpuCount={pod.gpu_count})...")
    if args.dry_run:
        print("[dry-run] Would call resume_pod and wait for SSH.")
        return
    resume_pod(pod.pod_id, pod.gpu_count)
    ready = wait_for_ssh(pod.pod_id, timeout=600)
    pod.status = "running"
    pod.stopped_at = None
    pod.host = ready.ssh_host
    pod.port = ready.ssh_port
    state[name] = pod
    _save_state(state)
    _upsert_pods_conf(pod)
    print(f"  SSH ready at {pod.host}:{pod.port}")
    print(f"  pods.conf updated. Connect: ssh {name}")


def cmd_terminate(args: argparse.Namespace) -> None:
    """Destroy the pod for issue #N. Volume gone."""
    state = _load_state()
    name = f"epm-issue-{args.issue}"
    if name not in state:
        raise SystemExit(f"No ephemeral pod recorded for issue {args.issue}")
    pod = state[name]
    if pod.status == "terminated":
        print(f"{name} already terminated.")
        return

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
    pod.status = "terminated"
    pod.host = None
    pod.port = None
    state[name] = pod
    _save_state(state)
    _remove_from_pods_conf(name)
    print("  Terminated. Removed from pods.conf.")


def cmd_cleanup_stale(args: argparse.Namespace) -> None:
    """Terminate stopped pods that have been idle longer than their TTL."""
    state = _load_state()
    now = dt.datetime.now(dt.UTC)
    stale: list[EphemeralPod] = []
    for pod in state.values():
        if pod.status != "stopped" or not pod.stopped_at:
            continue
        try:
            stopped_at = dt.datetime.fromisoformat(pod.stopped_at)
        except ValueError:
            continue
        age_days = (now - stopped_at).total_seconds() / 86400
        if age_days >= pod.ttl_days:
            stale.append(pod)

    if not stale:
        print("No stale pods to clean up.")
        return

    print(f"Stale pods ({len(stale)}, idle >= TTL):")
    for pod in stale:
        print(f"  {pod.name:24s} stopped {pod.stopped_at}  ttl={pod.ttl_days}d")

    if args.dry_run:
        print("\n[dry-run] No pods terminated.")
        return
    if not args.yes:
        confirm = input("\nTerminate all of the above? Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    for pod in stale:
        try:
            terminate_pod(pod.pod_id)
            pod.status = "terminated"
            pod.host = None
            pod.port = None
            state[pod.name] = pod
            _remove_from_pods_conf(pod.name)
            print(f"  ✓ {pod.name}")
        except Exception as exc:
            print(f"  ✗ {pod.name}: {exc}", file=sys.stderr)
    _save_state(state)


def cmd_list_ephemeral(args: argparse.Namespace) -> None:
    """List ephemeral pods. Cross-checks sidecar against the live API."""
    state = _load_state()
    if not state:
        print("No ephemeral pods recorded.")
        return

    if args.refresh:
        live = {p.pod_id: p for p in list_team_pods()}
        for pod in state.values():
            if pod.pod_id in live:
                live_status = live[pod.pod_id].desired_status
                if live_status == "RUNNING" and pod.status != "running":
                    print(f"  drift: {pod.name} stored={pod.status} live=RUNNING — fixing")
                    pod.status = "running"
                elif live_status == "EXITED" and pod.status == "running":
                    print(f"  drift: {pod.name} stored=running live=EXITED — fixing")
                    pod.status = "stopped"
                    pod.stopped_at = pod.stopped_at or _now()
            elif pod.status != "terminated":
                print(f"  drift: {pod.name} not in API — marking terminated")
                pod.status = "terminated"
        _save_state(state)

    header = f"{'NAME':<22} {'ISSUE':<6} {'STATUS':<11} {'GPUS':<10} {'AGE':<14} POD_ID"
    print(header)
    print("-" * len(header))
    now = dt.datetime.now(dt.UTC)
    for pod in sorted(state.values(), key=lambda p: -p.issue):
        age = ""
        if pod.created_at:
            try:
                created = dt.datetime.fromisoformat(pod.created_at)
                age = f"{(now - created).days}d"
            except ValueError:
                pass
        print(
            f"{pod.name:<22} #{pod.issue:<5} {pod.status:<11} "
            f"{pod.gpu_count}x{pod.gpu_type:<7} {age:<14} {pod.pod_id}"
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


def _parser_cleanup_stale(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("cleanup-stale", help="Terminate stopped pods past their TTL")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_cleanup_stale)


def _parser_list(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("list-ephemeral", help="Show all ephemeral pods + lifecycle state")
    p.add_argument("--refresh", action="store_true", help="Reconcile against live RunPod API")
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
    _parser_cleanup_stale(sub)
    _parser_list(sub)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
