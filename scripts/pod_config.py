#!/usr/bin/env python3
"""Pod configuration manager -- generates SSH and MCP configs from pods.conf.

pods.conf is the SINGLE SOURCE OF TRUTH for pod connection details. This script
reads it and can regenerate ~/.ssh/config and .claude/mcp.json so you only need
to edit one file when a pod IP changes.

Usage:
    python scripts/pod_config.py --list              # Show all pods
    python scripts/pod_config.py --check             # Verify configs are in sync
    python scripts/pod_config.py --sync              # Regenerate ~/.ssh/config + .claude/mcp.json
    python scripts/pod_config.py --update pod2 --host 1.2.3.4 --port 12345
    python scripts/pod_config.py --json              # Output pod list as JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this script so it works from any cwd
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PODS_CONF = SCRIPT_DIR / "pods.conf"
MCP_JSON = PROJECT_ROOT / ".claude" / "mcp.json"
SSH_CONFIG = Path.home() / ".ssh" / "config"

# Shared SSH defaults written into every generated entry
SSH_KEY = "~/.ssh/id_ed25519"
SSH_USER = "root"
REMOTE_DIR = "/workspace/explore-persona-space"

# Markers delimiting the auto-generated block inside ~/.ssh/config.
# Everything between these lines (inclusive) is replaced on --sync.
BEGIN_MARKER = "# --- BEGIN MANAGED POD CONFIG ---"
END_MARKER = "# --- END MANAGED POD CONFIG ---"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Pod:
    name: str  # e.g. "pod1"
    host: str  # IP address
    port: int
    gpus: int
    gpu_type: str  # e.g. "H200", "H100"
    label: str  # human-readable RunPod name, e.g. "thomas-rebuttals"


# ---------------------------------------------------------------------------
# Parsing / writing pods.conf
# ---------------------------------------------------------------------------


def parse_pods_conf(path: Path = PODS_CONF) -> list[Pod]:
    """Read pods.conf and return a list of Pod objects.

    Format (whitespace-separated, 6 fields per line):
        name  host  port  gpus  gpu_type  label

    Lines starting with '#' and blank lines are skipped.
    """
    if not path.exists():
        print(f"ERROR: pods.conf not found at {path}", file=sys.stderr)
        sys.exit(1)

    pods: list[Pod] = []
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            print(
                f"WARNING: pods.conf:{lineno}: expected 6 fields, got {len(parts)} -- skipping",
                file=sys.stderr,
            )
            continue
        name, host, port_str, gpus_str, gpu_type, label = parts[:6]
        try:
            port = int(port_str)
            gpus = int(gpus_str)
        except ValueError:
            print(
                f"WARNING: pods.conf:{lineno}: port/gpus must be integers -- skipping",
                file=sys.stderr,
            )
            continue
        pods.append(Pod(name=name, host=host, port=port, gpus=gpus, gpu_type=gpu_type, label=label))
    return pods


def write_pods_conf(pods: list[Pod], path: Path = PODS_CONF) -> None:
    """Write the pod list back to pods.conf, preserving the header comments."""
    # Keep existing header comment lines.
    header_lines: list[str] = []
    if path.exists():
        for raw in path.read_text().splitlines():
            if raw.startswith("#"):
                header_lines.append(raw)
            else:
                break
    if not header_lines:
        header_lines = [
            "# Pod registry -- SINGLE SOURCE OF TRUTH for all pod configuration.",
            "# All other configs (~/.ssh/config, .claude/mcp.json) are generated from this file.",
            "# Run `python scripts/pod_config.py --sync` after editing.",
            "#",
            "# Format: name  host  port  gpus  gpu_type  label",
        ]

    # Compute column widths for aligned output.
    rows = [(p.name, p.host, str(p.port), str(p.gpus), p.gpu_type, p.label) for p in pods]
    widths = [max(len(r[i]) for r in rows) for i in range(6)] if rows else [0] * 6

    lines = list(header_lines)
    for row in rows:
        parts = [row[i].ljust(widths[i]) for i in range(6)]
        lines.append("  ".join(parts).rstrip())

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# SSH config generation
# ---------------------------------------------------------------------------


def _ssh_entry(pod: Pod) -> str:
    """Return the SSH config block for a single pod."""
    return (
        f"# {pod.label} - {pod.gpus}x {pod.gpu_type}\n"
        f"Host {pod.name}\n"
        f"    HostName {pod.host}\n"
        f"    Port {pod.port}\n"
        f"    User {SSH_USER}\n"
        f"    IdentityFile {SSH_KEY}\n"
        f"    StrictHostKeyChecking no\n"
        f"    ConnectTimeout 10\n"
        f"    ServerAliveInterval 60\n"
        f"    ServerAliveCountMax 3"
    )


def _generate_managed_block(pods: list[Pod]) -> str:
    """Return the full managed block including markers."""
    inner = "\n\n".join(_ssh_entry(p) for p in pods)
    return (
        f"{BEGIN_MARKER}\n"
        f"# Auto-generated from pods.conf -- do not edit manually.\n"
        f"# Regenerate: python scripts/pod_config.py --sync\n"
        f"\n"
        f"{inner}\n"
        f"{END_MARKER}"
    )


def update_ssh_config(pods: list[Pod]) -> list[str]:
    """Replace the managed block in ~/.ssh/config. Returns list of change descriptions."""
    changes: list[str] = []
    new_block = _generate_managed_block(pods)

    if not SSH_CONFIG.exists():
        SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        SSH_CONFIG.write_text(new_block + "\n")
        changes.append(f"~/.ssh/config: created with {len(pods)} pod entries")
        return changes

    content = SSH_CONFIG.read_text()

    if BEGIN_MARKER in content and END_MARKER in content:
        # Replace existing managed block.
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        new_content = pattern.sub(new_block, content)
        if new_content == content:
            changes.append("~/.ssh/config: already up to date")
        else:
            SSH_CONFIG.write_text(new_content)
            changes.append("~/.ssh/config: updated managed pod block")
    else:
        # No markers found -- append the managed block.
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + new_block + "\n"
        SSH_CONFIG.write_text(content)
        changes.append("~/.ssh/config: appended managed block (markers added)")

    return changes


# ---------------------------------------------------------------------------
# SSH config parsing (for --check)
# ---------------------------------------------------------------------------


def _parse_ssh_config_pods() -> dict[str, tuple[str, int]]:
    """Parse ~/.ssh/config and extract pod entries. Returns {name: (host, port)}."""
    if not SSH_CONFIG.exists():
        return {}

    result: dict[str, tuple[str, int]] = {}
    current_host: str | None = None
    current_hostname: str | None = None
    current_port = 22

    for line in SSH_CONFIG.read_text().splitlines():
        stripped = line.strip()

        # New Host block (skip wildcard Host *)
        if stripped.startswith("Host ") and not stripped.startswith("Host *"):
            # Flush previous
            if current_host and current_host.startswith("pod"):
                result[current_host] = (current_hostname or "", current_port)
            alias = stripped.split(None, 1)[1].strip()
            current_host = alias if re.match(r"^pod\d+$", alias) else None
            current_hostname = None
            current_port = 22
        elif current_host:
            if stripped.startswith("HostName "):
                current_hostname = stripped.split(None, 1)[1].strip()
            elif stripped.startswith("Port "):
                with contextlib.suppress(ValueError, IndexError):
                    current_port = int(stripped.split(None, 1)[1].strip())

    # Flush last entry
    if current_host and current_host.startswith("pod"):
        result[current_host] = (current_hostname or "", current_port)

    return result


# ---------------------------------------------------------------------------
# MCP config generation
# ---------------------------------------------------------------------------


def _generate_mcp_env(pods: list[Pod]) -> dict[str, str]:
    """Build the env dict for the SSH MCP server entry."""
    env: dict[str, str] = {}
    for pod in pods:
        num = pod.name.removeprefix("pod")
        prefix = f"SSH_SERVER_POD{num}"
        env[f"{prefix}_HOST"] = pod.host
        env[f"{prefix}_PORT"] = str(pod.port)
        env[f"{prefix}_USER"] = SSH_USER
        env[f"{prefix}_KEYPATH"] = SSH_KEY
        env[f"{prefix}_DEFAULT_DIR"] = REMOTE_DIR
        env[f"{prefix}_PLATFORM"] = "linux"
        env[f"{prefix}_DESCRIPTION"] = f"{pod.label} {pod.gpus}x{pod.gpu_type}"
    return env


def update_mcp_config(pods: list[Pod]) -> list[str]:
    """Update the SSH server env vars in .claude/mcp.json. Returns change descriptions."""
    changes: list[str] = []

    if not MCP_JSON.exists():
        changes.append(f".claude/mcp.json: NOT FOUND at {MCP_JSON} -- skipped")
        return changes

    try:
        data = json.loads(MCP_JSON.read_text())
    except json.JSONDecodeError as exc:
        changes.append(f".claude/mcp.json: JSON parse error ({exc}) -- skipped")
        return changes

    servers = data.get("mcpServers", {})
    if "ssh" not in servers:
        changes.append('.claude/mcp.json: no "ssh" server in mcpServers -- skipped')
        return changes

    old_env = servers["ssh"].get("env", {})

    # Strip existing SSH_SERVER_POD* keys, keep any non-pod env vars.
    preserved_env = {k: v for k, v in old_env.items() if not re.match(r"SSH_SERVER_POD\d+_", k)}
    new_pod_env = _generate_mcp_env(pods)
    new_env = {**preserved_env, **new_pod_env}

    if old_env == new_env:
        changes.append(".claude/mcp.json: already up to date")
        return changes

    # Report per-key diffs for visibility.
    all_keys = sorted(set(old_env) | set(new_env))
    for key in all_keys:
        old_val = old_env.get(key)
        new_val = new_env.get(key)
        if old_val is None:
            changes.append(f"  mcp: + {key}={new_val}")
        elif new_val is None:
            changes.append(f"  mcp: - {key} (was {old_val})")
        elif old_val != new_val:
            changes.append(f"  mcp: ~ {key}: {old_val} -> {new_val}")

    servers["ssh"]["env"] = new_env
    MCP_JSON.write_text(json.dumps(data, indent=2) + "\n")
    changes.insert(0, ".claude/mcp.json: updated SSH server env vars")

    return changes


# ---------------------------------------------------------------------------
# MCP config parsing (for --check)
# ---------------------------------------------------------------------------


def _parse_mcp_pods() -> dict[str, tuple[str, int]]:
    """Extract pod host/port from .claude/mcp.json. Returns {name: (host, port)}."""
    if not MCP_JSON.exists():
        return {}
    try:
        data = json.loads(MCP_JSON.read_text())
    except json.JSONDecodeError:
        return {}

    env = data.get("mcpServers", {}).get("ssh", {}).get("env", {})
    result: dict[str, tuple[str, int]] = {}

    for key, value in env.items():
        m = re.match(r"SSH_SERVER_(POD\d+)_HOST$", key)
        if m:
            pod_name = m.group(1).lower()
            num = pod_name.removeprefix("pod")
            port_str = env.get(f"SSH_SERVER_POD{num}_PORT", "22")
            try:
                port = int(port_str)
            except ValueError:
                port = 22
            result[pod_name] = (value, port)

    return result


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(pods: list[Pod]) -> None:
    """Print a formatted table of all pods."""
    if not pods:
        print("No pods defined in pods.conf")
        return

    header = ("NAME", "HOST", "PORT", "GPUS", "TYPE", "LABEL")
    rows = [(p.name, p.host, str(p.port), str(p.gpus), p.gpu_type, p.label) for p in pods]
    all_rows = [header, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(6)]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(6)).rstrip()

    print(fmt(header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))
    print(f"\nTotal: {len(pods)} pods, {sum(p.gpus for p in pods)} GPUs")


def cmd_json(pods: list[Pod]) -> None:
    """Output the pod list as a JSON array to stdout."""
    json.dump([asdict(p) for p in pods], sys.stdout, indent=2)
    print()


def cmd_check(pods: list[Pod]) -> None:
    """Compare pods.conf against ~/.ssh/config and .claude/mcp.json, report mismatches."""
    conf_map = {p.name: (p.host, p.port) for p in pods}
    ssh_map = _parse_ssh_config_pods()
    mcp_map = _parse_mcp_pods()

    all_names = sorted(set(list(conf_map) + list(ssh_map) + list(mcp_map)))
    all_ok = True

    # Table header
    print(f"{'Pod':<8} {'pods.conf':<28} {'~/.ssh/config':<28} {'.claude/mcp.json':<28}")
    print("-" * 92)

    for name in all_names:
        conf = conf_map.get(name)
        ssh = ssh_map.get(name)
        mcp = mcp_map.get(name)

        conf_str = f"{conf[0]}:{conf[1]}" if conf else "MISSING"
        ssh_str = f"{ssh[0]}:{ssh[1]}" if ssh else "MISSING"
        mcp_str = f"{mcp[0]}:{mcp[1]}" if mcp else "MISSING"

        present = [v for v in (conf, ssh, mcp) if v is not None]
        match = len(set(present)) <= 1 and len(present) == 3

        if sys.stdout.isatty():
            marker = "\033[32mOK\033[0m" if match else "\033[31mMISMATCH\033[0m"
        else:
            marker = "OK" if match else "MISMATCH"

        print(f"{name:<8} {conf_str:<28} {ssh_str:<28} {mcp_str:<28} {marker}")

        if not match:
            all_ok = False

    print()
    if all_ok:
        print("All configs in sync.")
    else:
        print("Configs out of sync! Run: python scripts/pod_config.py --sync")
    sys.exit(0 if all_ok else 1)


def cmd_sync(pods: list[Pod]) -> None:
    """Regenerate ~/.ssh/config and .claude/mcp.json from pods.conf."""
    print("Syncing configs from pods.conf...")
    print()

    ssh_changes = update_ssh_config(pods)
    for c in ssh_changes:
        print(f"  {c}")

    mcp_changes = update_mcp_config(pods)
    for c in mcp_changes:
        print(f"  {c}")

    print()
    any_changed = any(
        "up to date" not in c for c in ssh_changes + mcp_changes if "skipped" not in c
    )
    if any_changed:
        print("Done. If MCP config changed, restart the MCP server (/mcp).")
    else:
        print("Everything already in sync.")
    print("Verify with: python scripts/pod_config.py --check")


def cmd_update(pods: list[Pod], pod_name: str, host: str | None, port: int | None) -> None:
    """Update a pod's host/port in pods.conf, then sync all downstream configs."""
    target = None
    for p in pods:
        if p.name == pod_name:
            target = p
            break

    if target is None:
        print(f"ERROR: pod '{pod_name}' not found in pods.conf", file=sys.stderr)
        print(f"Available: {', '.join(p.name for p in pods)}", file=sys.stderr)
        sys.exit(1)

    if host is None and port is None:
        print("ERROR: --update requires at least one of --host or --port", file=sys.stderr)
        sys.exit(1)

    changes: list[str] = []
    if host is not None and host != target.host:
        changes.append(f"  {pod_name} host: {target.host} -> {host}")
        target.host = host
    if port is not None and port != target.port:
        changes.append(f"  {pod_name} port: {target.port} -> {port}")
        target.port = port

    if not changes:
        print(f"{pod_name}: already has those values, nothing to update.")
        return

    print("Updating pods.conf:")
    for c in changes:
        print(c)
    write_pods_conf(pods)
    print()

    # Auto-sync downstream configs.
    cmd_sync(pods)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pod config manager -- keeps SSH and MCP configs in sync with pods.conf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/pod_config.py --list\n"
            "  python scripts/pod_config.py --check\n"
            "  python scripts/pod_config.py --sync\n"
            "  python scripts/pod_config.py --update pod2 --host 1.2.3.4 --port 12345\n"
            "  python scripts/pod_config.py --json\n"
        ),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="Show all pods in a table")
    group.add_argument("--json", action="store_true", help="Output pod list as JSON")
    group.add_argument(
        "--check", action="store_true", help="Verify SSH and MCP configs match pods.conf"
    )
    group.add_argument(
        "--sync", action="store_true", help="Regenerate SSH and MCP configs from pods.conf"
    )
    group.add_argument("--update", metavar="POD_NAME", help="Update a pod's host/port, then sync")

    parser.add_argument("--host", help="New host (IP) for --update")
    parser.add_argument("--port", type=int, help="New port for --update")

    args = parser.parse_args()

    pods = parse_pods_conf()

    if args.list:
        cmd_list(pods)
    elif args.json:
        cmd_json(pods)
    elif args.check:
        cmd_check(pods)
    elif args.sync:
        cmd_sync(pods)
    elif args.update:
        cmd_update(pods, args.update, args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
