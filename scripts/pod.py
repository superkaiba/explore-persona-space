#!/usr/bin/env python3
"""Unified CLI for pod management.

Wraps all pod-related scripts into a single entry point.

Usage:
    python scripts/pod.py config --list              # List all pods
    python scripts/pod.py config --sync              # Regenerate SSH + MCP from pods.conf
    python scripts/pod.py config --check             # Verify configs are in sync
    python scripts/pod.py config --update pod2 --host 1.2.3.4 --port 12345
    python scripts/pod.py config --refresh-from-api          # API -> pods.conf -> sync
    python scripts/pod.py config --refresh-from-api pod-488  # Just one pod

    python scripts/pod.py keys --push                # Push .env to all pods
    python scripts/pod.py keys --push pod1 pod3      # Push to specific pods
    python scripts/pod.py keys --verify              # Check keys on all pods

    python scripts/pod.py bootstrap pod3             # Full pod setup
    python scripts/pod.py bootstrap --host X --port Y

    python scripts/pod.py health                     # Fleet health check
    python scripts/pod.py health --quick             # Just reachability + GPU
    python scripts/pod.py health --fix               # Auto-fix issues
    python scripts/pod.py health --json              # Machine-readable output

    python scripts/pod.py sync code                  # Git pull on all pods
    python scripts/pod.py sync env                   # uv sync on all pods
    python scripts/pod.py sync data --pull           # Pull datasets from HF Hub
    python scripts/pod.py sync data --push           # Push datasets to HF Hub
    python scripts/pod.py sync results --all         # Pull all results from WandB
    python scripts/pod.py sync models --list         # List models on HF Hub

    python scripts/pod.py cleanup pod1 --dry-run     # Show what would be cleaned
    python scripts/pod.py cleanup --all              # Clean all pods

    # ── Ephemeral lifecycle (dynamic per-issue pods) ─────────────────────────
    python scripts/pod.py provision --issue 137 --intent lora-7b
    python scripts/pod.py provision --issue 137 --gpu-type H200 --gpu-count 8
    python scripts/pod.py provision --list-intents   # Show GPU heuristic table
    python scripts/pod.py stop --issue 137           # Pause (volume preserved)
    python scripts/pod.py resume --issue 137         # Bring back; new IP
    python scripts/pod.py terminate --issue 137      # Destroy (volume gone)
    python scripts/pod.py list-ephemeral             # Show ephemeral pod state (live API auth.)
    python scripts/pod.py list-ephemeral --issue 137 # Filter to a single issue
    # --refresh is a deprecated no-op since #282 [1/4]; the live API is queried
    # on every invocation, so reconciliation is automatic.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EPHEMERAL_STATE = SCRIPT_DIR / "pods_ephemeral.json"


def run(cmd: list[str] | str, **kwargs) -> int:
    """Run a command, passing through stdio.

    The return code MUST be propagated by every ``cmd_*`` handler (and by
    :func:`main` via ``sys.exit``): callers like
    ``poll_pipeline._try_refresh_pods_conf_from_api``, the watcher's
    ``_stop_pod``, and ``pod_lifecycle._run_resume_subprocess`` all gate
    their recovery logic on ``pod.py``'s exit code. Before 2026-06-10 the
    handlers discarded it, so ``pod.py`` exited 0 even when the dispatched
    script failed — every one of those rc checks was a no-op (refs #572).
    """
    if isinstance(cmd, str):
        return subprocess.call(cmd, shell=True, **kwargs)
    return subprocess.call(cmd, **kwargs)


def _bootstrap_env_with_intent(pod_name: str | None) -> dict[str, str]:
    """Build an env dict with ``POD_INTENT`` set for bootstrap_pod.sh.

    Order of precedence:
      1. ``POD_INTENT`` already in the caller's environment (explicit override).
      2. Looked up from ``pods_ephemeral.json`` by pod name.
      3. Fallback to ``"custom"`` (triggers the flash-attn install — safe default).
    """
    env = os.environ.copy()
    if "POD_INTENT" in env and env["POD_INTENT"].strip():
        return env
    env["POD_INTENT"] = _lookup_pod_intent(pod_name) if pod_name else "custom"
    return env


def _lookup_pod_intent(pod_name: str) -> str:
    """Read the recorded gpu_intent for ``pod_name`` from the ephemeral sidecar.

    Returns ``"custom"`` if the sidecar is missing, the pod is not registered
    there (e.g. a permanent pod1..pod5), or any read error occurs. The intent
    matters only for bootstrap_pod.sh's flash-attn install gate, where
    ``"custom"`` triggers the install — the safe default.
    """
    try:
        if not EPHEMERAL_STATE.exists():
            return "custom"
        payload = json.loads(EPHEMERAL_STATE.read_text())
        pods = payload.get("pods", {})
        entry = pods.get(pod_name) or {}
        return entry.get("gpu_intent", "custom")
    except (OSError, json.JSONDecodeError):
        return "custom"


def cmd_config(args: list[str]) -> int:
    """Manage pod configuration."""
    return run([sys.executable, str(SCRIPT_DIR / "pod_config.py"), *args])


def cmd_keys(args: list[str]):
    """Manage .env distribution."""
    script = SCRIPT_DIR / "sync_env_keys.sh"
    # Map --push to default (no flag), --verify stays
    translated = []
    for a in args:
        if a == "--push":
            continue  # push is the default action
        translated.append(a)
    return run(["bash", str(script), *translated])


def cmd_bootstrap(args: list[str]):
    """Bootstrap a pod.

    Auto-derives ``POD_INTENT`` from ``pods_ephemeral.json`` so a manual
    re-bootstrap honors the same flash-attn install gate as the
    pod_lifecycle.py-driven path. Override by exporting ``POD_INTENT=<x>``
    in the caller's shell. The pod name is the first positional arg that
    isn't a flag (matches bootstrap_pod.sh's own argument parser).
    """
    pod_name: str | None = None
    for arg in args:
        if not arg.startswith("-") and (arg.startswith("pod") or arg.startswith("epm-")):
            pod_name = arg
            break
    return run(
        ["bash", str(SCRIPT_DIR / "bootstrap_pod.sh"), *args],
        env=_bootstrap_env_with_intent(pod_name),
    )


def cmd_health(args: list[str]) -> int:
    """Fleet health check."""
    return run([sys.executable, str(SCRIPT_DIR / "fleet_health.py"), *args])


def cmd_sync(args: list[str]):
    """Sync code, env, data, results, or models."""
    if not args:
        print("Usage: pod.py sync {code|env|data|results|models} [options]")
        return 2

    subcmd = args[0]
    rest = args[1:]

    if subcmd == "code":
        return run(["bash", str(SCRIPT_DIR / "sync_pods.sh"), *rest])
    elif subcmd == "env":
        return run(["bash", str(SCRIPT_DIR / "sync_env.sh"), *rest])
    elif subcmd == "data":
        return run([sys.executable, str(SCRIPT_DIR / "sync_datasets.py"), *rest])
    elif subcmd == "results":
        return run([sys.executable, str(SCRIPT_DIR / "pull_results.py"), *rest])
    elif subcmd == "models":
        return run([sys.executable, str(SCRIPT_DIR / "sync_models.py"), *rest])
    else:
        print(f"Unknown sync target: {subcmd}")
        print("Available: code, env, data, results, models")
        return 2


def cmd_cleanup(args: list[str]) -> int:
    """Clean up model weights on pods."""
    return run([sys.executable, str(SCRIPT_DIR / "cleanup_pod.py"), *args])


def _lifecycle(verb: str, args: list[str]) -> int:
    """Dispatch one of the ephemeral-pod lifecycle verbs to pod_lifecycle.py."""
    return run([sys.executable, str(SCRIPT_DIR / "pod_lifecycle.py"), verb, *args])


def cmd_provision(args: list[str]) -> int:
    return _lifecycle("provision", args)


def cmd_stop(args: list[str]) -> int:
    return _lifecycle("stop", args)


def cmd_resume(args: list[str]) -> int:
    return _lifecycle("resume", args)


def cmd_terminate(args: list[str]) -> int:
    return _lifecycle("terminate", args)


def cmd_list_ephemeral(args: list[str]) -> int:
    return _lifecycle("list-ephemeral", args)


def cmd_watch(args: list[str]) -> int:
    """Stall-detection watchdog (§2). Forwarded to scripts/pod_watch.py."""
    return run([sys.executable, str(SCRIPT_DIR / "pod_watch.py"), *args])


def cmd_audit_stale(args: list[str]) -> int:
    """Audit live RunPod account for stale/orphaned pods (forwarded to pod_audit.py)."""
    return run([sys.executable, str(SCRIPT_DIR / "pod_audit.py"), *args])


COMMANDS = {
    "config": (cmd_config, "Manage pod configuration (list, sync, check, update)"),
    "keys": (cmd_keys, "Distribute .env to pods (push, verify)"),
    "bootstrap": (cmd_bootstrap, "Bootstrap a pod from bare to experiment-ready"),
    "health": (cmd_health, "Fleet-wide health check"),
    "sync": (cmd_sync, "Sync code/env/data/results/models"),
    "cleanup": (cmd_cleanup, "Clean up stale model weights"),
    "provision": (cmd_provision, "Provision a fresh pod for an issue"),
    "stop": (cmd_stop, "Pause an issue's ephemeral pod"),
    "resume": (cmd_resume, "Resume a stopped ephemeral pod"),
    "terminate": (cmd_terminate, "Destroy an issue's ephemeral pod"),
    "list-ephemeral": (cmd_list_ephemeral, "Show ephemeral-pod lifecycle state"),
    "watch": (cmd_watch, "Stall-detection watchdog for an in-flight experiment"),
    "audit-stale": (
        cmd_audit_stale,
        "Find stale/orphaned pods via live API (catches lifecycle escapes)",
    ),
}


def print_help():
    print("Usage: python scripts/pod.py <command> [options]\n")
    print("Commands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<12} {desc}")
    print("\nRun 'python scripts/pod.py <command> --help' for command-specific help.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print_help()
        sys.exit(0)

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name}")
        print_help()
        sys.exit(1)

    handler, _ = COMMANDS[cmd_name]
    rc = handler(sys.argv[2:])
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
