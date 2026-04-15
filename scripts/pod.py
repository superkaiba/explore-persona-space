#!/usr/bin/env python3
"""Unified CLI for pod management.

Wraps all pod-related scripts into a single entry point.

Usage:
    python scripts/pod.py config --list              # List all pods
    python scripts/pod.py config --sync              # Regenerate SSH + MCP configs
    python scripts/pod.py config --check             # Verify configs are in sync
    python scripts/pod.py config --update pod2 --host 1.2.3.4 --port 12345

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
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run(cmd: list[str] | str, **kwargs) -> int:
    """Run a command, passing through stdio."""
    if isinstance(cmd, str):
        return subprocess.call(cmd, shell=True, **kwargs)
    return subprocess.call(cmd, **kwargs)


def cmd_config(args: list[str]):
    """Manage pod configuration."""
    run([sys.executable, str(SCRIPT_DIR / "pod_config.py"), *args])


def cmd_keys(args: list[str]):
    """Manage .env distribution."""
    script = SCRIPT_DIR / "sync_env_keys.sh"
    # Map --push to default (no flag), --verify stays
    translated = []
    for a in args:
        if a == "--push":
            continue  # push is the default action
        translated.append(a)
    run(["bash", str(script), *translated])


def cmd_bootstrap(args: list[str]):
    """Bootstrap a pod."""
    run(["bash", str(SCRIPT_DIR / "bootstrap_pod.sh"), *args])


def cmd_health(args: list[str]):
    """Fleet health check."""
    run([sys.executable, str(SCRIPT_DIR / "fleet_health.py"), *args])


def cmd_sync(args: list[str]):
    """Sync code, env, data, results, or models."""
    if not args:
        print("Usage: pod.py sync {code|env|data|results|models} [options]")
        return

    subcmd = args[0]
    rest = args[1:]

    if subcmd == "code":
        run(["bash", str(SCRIPT_DIR / "sync_pods.sh"), *rest])
    elif subcmd == "env":
        run(["bash", str(SCRIPT_DIR / "sync_env.sh"), *rest])
    elif subcmd == "data":
        run([sys.executable, str(SCRIPT_DIR / "sync_datasets.py"), *rest])
    elif subcmd == "results":
        run([sys.executable, str(SCRIPT_DIR / "pull_results.py"), *rest])
    elif subcmd == "models":
        run([sys.executable, str(SCRIPT_DIR / "sync_models.py"), *rest])
    else:
        print(f"Unknown sync target: {subcmd}")
        print("Available: code, env, data, results, models")


def cmd_cleanup(args: list[str]):
    """Clean up model weights on pods."""
    run([sys.executable, str(SCRIPT_DIR / "cleanup_pod.py"), *args])


COMMANDS = {
    "config": (cmd_config, "Manage pod configuration (list, sync, check, update)"),
    "keys": (cmd_keys, "Distribute .env to pods (push, verify)"),
    "bootstrap": (cmd_bootstrap, "Bootstrap a pod from bare to experiment-ready"),
    "health": (cmd_health, "Fleet-wide health check"),
    "sync": (cmd_sync, "Sync code/env/data/results/models"),
    "cleanup": (cmd_cleanup, "Clean up stale model weights"),
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
    handler(sys.argv[2:])


if __name__ == "__main__":
    main()
