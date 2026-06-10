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
    python scripts/pod_config.py --clear-override pod-391   # Re-enable auto-refresh
    python scripts/pod_config.py --json              # Output pod list as JSON
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-checking-only import: ``runpod_api`` is heavy (loads RunPod GraphQL
    # config from .env at import time) and ``cmd_refresh_from_api`` already
    # imports ``list_team_pods`` lazily for the same reason. ``PodInfo`` is
    # only used as a forward-referenced type annotation under
    # ``from __future__ import annotations``, so deferring the import here
    # keeps the cheap ``--list`` / ``--check`` paths free of the eager load.
    from runpod_api import PodInfo

# ---------------------------------------------------------------------------
# Paths -- resolved to the MAIN repo regardless of which worktree this
# module is loaded from. ``pods.conf`` and ``pods_ephemeral.json`` are
# SHARED fleet state — every parallel /issue session reads + mutates them.
# Resolving relative to ``__file__`` (the previous behavior) meant each
# worktree saw its OWN copy of these files; a ``pod.py resume`` in
# worktree A would correctly update A's ``pods.conf`` and then re-sync
# ``~/.ssh/config`` (global), but a later ``cmd_sync`` from worktree B
# (still holding a STALE row) would silently clobber the global ssh
# config and the resumed pod's new port. ``poll_pipeline.py`` SSHing via
# the ``Host pod-<N>`` alias would then connection-refuse on the stale
# port and report ``status: dead`` for a perfectly healthy run. Routing
# the constants through ``git rev-parse --git-common-dir`` collapses
# every checkout's copy to the same on-disk file so all sessions read +
# write the SAME state. Concurrent read-modify-write races within that
# single file are serialised by ``locked_pods_conf`` (see below), which
# every mutating call site holds for the whole parse → mutate → write →
# ``cmd_sync`` sequence.
# Incident 2026-06-05, task #500: pod.py resume from the issue-500
# worktree updated worktree-local pods.conf to port 13721, but the main
# repo's pods.conf stayed at the stale 16659; the next sync against the
# main copy wrote the stale port back into ~/.ssh/config and the
# poll-loop reported a FALSE dead.
# Incident 2026-06-05, task #488: two concurrent /issue sessions each
# called ``pod_lifecycle._upsert_pods_conf`` for their own pod; the
# session B write clobbered A's row, the regenerated ~/.ssh/config
# dropped A's ``Host pod-<A>`` block, and ``poll_pipeline.py`` reported
# ``ssh: Could not resolve hostname pod-<A>: Temporary failure in name
# resolution`` for a perfectly healthy run. Fixed by ``locked_pods_conf``.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _main_repo_scripts_dir() -> Path:
    """Return the absolute path of ``scripts/`` in the MAIN repo checkout.

    Resolves via ``git rev-parse --git-common-dir`` from the directory of
    this module (NOT ``os.getcwd()``). Each worktree's ``.git`` file
    points at the same shared ``.git`` directory in the main checkout;
    its parent is the main repo root, and ``scripts/`` lives directly
    underneath. Falls back loudly (``RuntimeError``) if git resolution
    fails or the resolved ``scripts/`` directory does not exist, so a
    silent fallback to the worktree-local copy cannot reintroduce the
    divergence bug this resolver exists to prevent.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"pod_config: cannot resolve main repo via "
            f"`git rev-parse --git-common-dir` from {SCRIPT_DIR}: {exc}. "
            f"pod_config must run inside an explore-persona-space checkout."
        ) from exc
    git_common = Path(proc.stdout.strip())
    if not git_common.is_absolute():
        git_common = (SCRIPT_DIR / git_common).resolve()
    main_repo_root = git_common.parent
    scripts_dir = main_repo_root / "scripts"
    if not scripts_dir.is_dir():
        raise RuntimeError(
            f"pod_config: resolved main repo root {main_repo_root} has no "
            f"scripts/ directory; refusing to route pods.conf writes through "
            f"a malformed layout."
        )
    return scripts_dir


_MAIN_SCRIPTS_DIR = _main_repo_scripts_dir()
PROJECT_ROOT = _MAIN_SCRIPTS_DIR.parent
PODS_CONF = _MAIN_SCRIPTS_DIR / "pods.conf"
# Sidecar JSON owned by pod_lifecycle.py — read here only to set/clear the
# manual_override flag from ``cmd_update``. Format documented in
# scripts/pod_lifecycle.py. We do not import pod_lifecycle.py because it
# already imports this module (avoiding circular import).
PODS_EPHEMERAL_JSON = _MAIN_SCRIPTS_DIR / "pods_ephemeral.json"
# The SSH MCP server (mcp-ssh-manager) lives in the user-level Claude config,
# NOT the project-level one. The project mcp.json (PROJECT_ROOT / ".claude" /
# "mcp.json") is reserved for project-scoped servers like arxiv.
MCP_JSON = Path.home() / ".claude" / "mcp.json"
SSH_CONFIG = Path.home() / ".ssh" / "config"

# Pod name patterns we recognize. Permanent fleet uses `podN`; ephemeral pods
# use `pod-<N>` (canonical, since the April 2026 rename) — the legacy
# `epm-issue-<N>` form is still recognized for in-flight pods provisioned
# before the rename, and can be removed once no live pods carry it.
# Anything else is treated as foreign and skipped.
POD_NAME_RE = re.compile(r"^(pod\d+|pod-\d+|epm-issue-\d+)$")

# Shared SSH defaults written into every generated entry
SSH_KEY = "~/.ssh/id_ed25519"
SSH_USER = "root"
REMOTE_DIR = "/workspace/explore-persona-space"

# Markers delimiting the auto-generated block inside ~/.ssh/config.
# Everything between these lines (inclusive) is replaced on --sync.
BEGIN_MARKER = "# --- BEGIN MANAGED POD CONFIG ---"
END_MARKER = "# --- END MANAGED POD CONFIG ---"

# Sibling lockfile in the SAME main-repo scripts/ directory as ``pods.conf``
# itself. Held under an exclusive ``fcntl.flock`` for the duration of any
# read-modify-write on ``pods.conf`` + the downstream ``~/.ssh/config`` /
# ``~/.claude/mcp.json`` regeneration. Co-located so the lock can never
# diverge from the file it protects across worktree checkouts (same
# main-repo-resolution as ``PODS_CONF``).
PODS_CONF_LOCK = _MAIN_SCRIPTS_DIR / ".pods.conf.lock"


@contextlib.contextmanager
def locked_pods_conf() -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``PODS_CONF_LOCK`` for a read-modify-write
    on ``pods.conf`` and the downstream SSH/MCP config regeneration.

    Concurrency motivation. Multiple parallel ``/issue`` sessions each call
    ``pod_lifecycle._upsert_pods_conf`` (or ``_remove_from_pods_conf``) when
    provisioning / terminating their own pod. The unguarded sequence
    ``parse_pods_conf() -> mutate(rows) -> write_pods_conf(rows) ->
    cmd_sync(rows)`` is a classic lost-update race: session A reads, session
    B reads, A writes (with A's row), B writes (with B's row, A's row gone),
    and the final ``~/.ssh/config`` block reflects only B's view — so
    ``poll_pipeline.py`` SSHing via ``Host pod-<A>`` fails with
    ``Could not resolve hostname pod-<A>`` while A's pod is healthy.

    Serialising the whole read-modify-write-sync sequence under a single
    advisory lock collapses the race. ``cmd_sync`` is kept inside the
    critical section so a concurrent session cannot regenerate
    ``~/.ssh/config`` from a stale ``rows`` view between our
    ``write_pods_conf`` and our ``cmd_sync``. The lock is advisory and
    fcntl-based, so it is automatically released on process death (kill -9,
    OOM kill, parent timeout) — no orphaned locks survive a crash.

    Read-only callers (``cmd_list``, ``cmd_check``, ``cmd_json``,
    ``parse_pods_conf`` from external readers like ``poll_pipeline.py``) do
    NOT take this lock — they tolerate seeing a momentarily-mid-write state
    because ``write_pods_conf`` writes atomically via a single text payload.
    """
    PODS_CONF_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(PODS_CONF_LOCK), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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
            if current_host and POD_NAME_RE.match(current_host):
                result[current_host] = (current_hostname or "", current_port)
            alias = stripped.split(None, 1)[1].strip()
            current_host = alias if POD_NAME_RE.match(alias) else None
            current_hostname = None
            current_port = 22
        elif current_host:
            if stripped.startswith("HostName "):
                current_hostname = stripped.split(None, 1)[1].strip()
            elif stripped.startswith("Port "):
                with contextlib.suppress(ValueError, IndexError):
                    current_port = int(stripped.split(None, 1)[1].strip())

    # Flush last entry
    if current_host and POD_NAME_RE.match(current_host):
        result[current_host] = (current_hostname or "", current_port)

    return result


# ---------------------------------------------------------------------------
# MCP config generation
# ---------------------------------------------------------------------------


def _generate_mcp_env(pods: list[Pod]) -> dict[str, str]:
    """Build the env dict for the SSH MCP server entry.

    The suffix is `pod.name.upper()` verbatim. mcp-ssh-manager lowercases
    the suffix on parse, so the registered name round-trips to the pod name
    in pods.conf (e.g. `pod-261`, or legacy `epm-issue-261`). An older
    scheme prepended `POD` for every pod, which produced
    `SSH_SERVER_PODepm-issue-261_HOST` — a key the upstream regex
    `[A-Z0-9_]+` silently rejected.
    """
    env: dict[str, str] = {}
    for pod in pods:
        prefix = f"SSH_SERVER_{pod.name.upper()}"
        env[f"{prefix}_HOST"] = pod.host
        env[f"{prefix}_PORT"] = str(pod.port)
        env[f"{prefix}_USER"] = SSH_USER
        env[f"{prefix}_KEYPATH"] = SSH_KEY
        env[f"{prefix}_DEFAULT_DIR"] = REMOTE_DIR
        env[f"{prefix}_PLATFORM"] = "linux"
        env[f"{prefix}_DESCRIPTION"] = f"{pod.label} {pod.gpus}x{pod.gpu_type}"
    return env


def update_mcp_config(pods: list[Pod]) -> list[str]:
    """Update the SSH server env vars in ~/.claude/mcp.json. Returns change descriptions.

    The SSH MCP server (mcp-ssh-manager) lives in the user-level Claude config.
    If it is missing we fail loudly rather than silently skipping, because
    silently skipping creates the long-debugged "ssh tools work locally but not
    after sync" mode.
    """
    changes: list[str] = []

    if not MCP_JSON.exists():
        raise SystemExit(
            f"ERROR: {MCP_JSON} does not exist. The user-level Claude config\n"
            f"is required because the SSH MCP server is registered there.\n"
            f'Create it with at least: {{"mcpServers": {{}}}}'
        )

    try:
        data = json.loads(MCP_JSON.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: {MCP_JSON} JSON parse error: {exc}") from exc

    servers = data.get("mcpServers", {})
    if "ssh" not in servers:
        raise SystemExit(
            f'ERROR: no "ssh" server in {MCP_JSON} mcpServers.\n'
            f"The SSH MCP server (mcp-ssh-manager) must be registered there\n"
            f'so that pod env vars can be wired in. See CLAUDE.md "Remote Pod\n'
            f'Access (SSH MCP)" for the expected entry shape.'
        )

    old_env = servers["ssh"].get("env", {})

    # Strip existing pod env keys:
    #  - permanent SSH_SERVER_POD<N>_*
    #  - canonical ephemeral SSH_SERVER_POD-<N>_*
    #  - legacy ephemeral SSH_SERVER_EPM-ISSUE-<N>_* (pre-rename)
    #  - very-legacy ephemeral SSH_SERVER_PODepm-issue-<N>_* (pre-prefix-fix)
    # Keep any non-pod env vars.
    pod_key_re = re.compile(r"^SSH_SERVER_(?:POD\d+|POD-\d+|EPM-ISSUE-\d+|PODepm-issue-\d+)_")
    preserved_env = {k: v for k, v in old_env.items() if not pod_key_re.match(k)}
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

    # Permanent pods:        SSH_SERVER_POD<N>_HOST            -> name "podN"
    # Canonical ephemeral:   SSH_SERVER_POD-<N>_HOST           -> name "pod-N"
    # Legacy ephemeral:      SSH_SERVER_EPM-ISSUE-<N>_HOST     -> name "epm-issue-N"
    # Very-legacy ephemeral: SSH_SERVER_PODepm-issue-<N>_HOST  -> name "epm-issue-N"
    host_key_re = re.compile(
        r"^SSH_SERVER_(?P<suffix>POD\d+|POD-\d+|EPM-ISSUE-\d+|PODepm-issue-\d+)_HOST$"
    )

    for key, value in env.items():
        m = host_key_re.match(key)
        if not m:
            continue
        suffix = m.group("suffix")
        suffix_lower = suffix.lower()
        # Drop the spurious "pod" prefix from the very-legacy ephemeral shape.
        pod_name = (
            suffix_lower.removeprefix("pod")
            if suffix_lower.startswith("podepm-issue-")
            else suffix_lower
        )
        port_str = env.get(f"SSH_SERVER_{suffix}_PORT", "22")
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


def _check_mcp_patch_applied() -> tuple[bool, str]:
    """Verify the mcp-ssh-manager hot-reload patch is still applied to node_modules.

    The patch (patches/mcp-ssh-manager+3.2.2.patch) makes the SSH MCP server
    re-read ~/.claude/mcp.json on mtime change AND accept lowercase + hyphens
    in env-key names. Without it, ephemeral pods (pod-N / epm-issue-N) silently
    fail to register because the upstream regex `[A-Z0-9_]+` rejects them. A
    routine `npm install` in ~/.local would silently revert the patch with no
    error surface — this guard catches that drift.

    Returns (ok, message). ok=True if the sentinel function is present OR if
    node_modules is absent (no MCP install to check). ok=False only when the
    file exists but the sentinel is missing, indicating a reverted patch.
    """
    index_js = Path.home() / ".local" / "node_modules" / "mcp-ssh-manager" / "src" / "index.js"
    if not index_js.exists():
        return True, f"mcp-ssh-manager not installed at {index_js} (skipping patch check)"
    try:
        content = index_js.read_text()
    except OSError as exc:
        return True, f"could not read {index_js}: {exc} (skipping patch check)"
    sentinel = "_hotReloadFromMcpJson"
    if sentinel in content:
        return True, "mcp-ssh-manager hot-reload patch is applied"
    return False, (
        f"PATCH MISSING: {index_js}\n"
        f"  The hot-reload patch has been reverted (likely by `npm install`).\n"
        f"  Without it, ephemeral pods (pod-N / epm-issue-N) are invisible to the SSH MCP server.\n"
        f"  Re-apply with:  patch -p1 -d ~/.local < patches/mcp-ssh-manager+3.2.2.patch"
    )


def cmd_check(pods: list[Pod]) -> None:
    """Compare pods.conf against ~/.ssh/config and .claude/mcp.json, report mismatches."""
    patch_ok, patch_msg = _check_mcp_patch_applied()
    if patch_ok:
        print(patch_msg)
    else:
        print(patch_msg, file=sys.stderr)
    print()

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
    if all_ok and patch_ok:
        print("All configs in sync.")
    elif not all_ok:
        print("Configs out of sync! Run: python scripts/pod_config.py --sync")
    sys.exit(0 if (all_ok and patch_ok) else 1)


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


def _set_manual_override(pod_name: str, *, value: bool) -> str | None:
    """Set or clear ``manual_override`` for ``pod_name`` in pods_ephemeral.json.

    Returns a human-readable status string (printed by callers), or None when
    the file does not exist or the pod is not registered there. Permanent-
    fleet pods like ``pod1``, ``pod2`` are not in the sidecar — they aren't
    subject to live-API drift, so we silently no-op.

    Does NOT auto-create the sidecar; if it is missing, the override flag has
    nothing to protect (no auto-refresh would touch a non-existent entry).
    """
    if not PODS_EPHEMERAL_JSON.exists():
        return None
    try:
        data = json.loads(PODS_EPHEMERAL_JSON.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"WARNING: {PODS_EPHEMERAL_JSON} JSON parse error: {exc}; "
            f"could not set manual_override for {pod_name}.",
            file=sys.stderr,
        )
        return None

    pods = data.get("pods", {})
    if pod_name not in pods:
        return None

    prev = bool(pods[pod_name].get("manual_override", False))
    if prev == value:
        return f"pods_ephemeral.json: manual_override for {pod_name} already {value}"
    pods[pod_name]["manual_override"] = value
    PODS_EPHEMERAL_JSON.write_text(json.dumps(data, indent=2) + "\n")
    return f"pods_ephemeral.json: manual_override for {pod_name} {prev} -> {value}"


def cmd_update(pods: list[Pod], pod_name: str, host: str | None, port: int | None) -> None:
    """Update a pod's host/port in pods.conf, then sync all downstream configs.

    Also flips ``manual_override=True`` in pods_ephemeral.json for matching
    ephemeral pods so the auto-refresh paths in ``pod_lifecycle.py`` will not
    silently clobber the manual values from a later ``provision`` / ``resume``
    / cron run. Permanent-fleet pods (``podN``) are not in the sidecar; the
    flag is a no-op there.

    The pre-validation pass uses ``pods`` (already parsed by ``main`` for
    arg-flag checks). The actual read-modify-write-sync runs inside
    ``locked_pods_conf`` after re-reading ``pods.conf`` so a concurrent
    writer cannot interleave between our parse and our write.
    """
    if host is None and port is None:
        print("ERROR: --update requires at least one of --host or --port", file=sys.stderr)
        sys.exit(1)

    if not any(p.name == pod_name for p in pods):
        print(f"ERROR: pod '{pod_name}' not found in pods.conf", file=sys.stderr)
        print(f"Available: {', '.join(p.name for p in pods)}", file=sys.stderr)
        sys.exit(1)

    with locked_pods_conf():
        # Re-parse under the lock so we operate on the freshest on-disk view
        # (a concurrent provision / terminate may have written between
        # ``main``'s parse and our acquisition of the lock).
        fresh = parse_pods_conf()
        target = next((p for p in fresh if p.name == pod_name), None)
        if target is None:
            # Concurrent terminate between main's parse and ours.
            print(
                f"ERROR: pod '{pod_name}' no longer in pods.conf "
                f"(removed by a concurrent writer between read and update).",
                file=sys.stderr,
            )
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
        write_pods_conf(fresh)

        # Mark the sidecar so a later auto-refresh in pod_lifecycle.py does NOT
        # silently overwrite the values just set. No-op for permanent pods.
        status = _set_manual_override(pod_name, value=True)
        if status is not None:
            print(f"  {status}")

        print()

        # Auto-sync downstream configs from the post-write rows (still
        # inside the lock so a concurrent session cannot regenerate the SSH
        # config from a stale view between our write and our sync).
        cmd_sync(fresh)


def cmd_clear_override(pod_name: str) -> None:
    """Clear ``manual_override`` for ``pod_name`` in pods_ephemeral.json.

    Call this when the manually-set values are no longer correct (e.g., the
    pod the user pointed at has been terminated and they want a future
    ``resume`` to repoint from the live API). No-op for permanent or
    unregistered pods.
    """
    status = _set_manual_override(pod_name, value=False)
    if status is None:
        print(
            f"{pod_name}: not in pods_ephemeral.json — nothing to clear "
            f"(permanent-fleet pods like pod1/pod2 do not carry the flag).",
            file=sys.stderr,
        )
        return
    print(status)


def _read_manual_overrides() -> dict[str, bool]:
    """Read ``manual_override`` flags for every pod in pods_ephemeral.json.

    Permanent-fleet pods (``pod1``..``pod5``) are not in the sidecar, so they
    are simply absent from the returned dict (callers default to False).
    Returns an empty dict when the sidecar is missing or malformed — same
    fail-quiet shape ``_set_manual_override`` uses on read.
    """
    if not PODS_EPHEMERAL_JSON.exists():
        return {}
    try:
        data = json.loads(PODS_EPHEMERAL_JSON.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"WARNING: {PODS_EPHEMERAL_JSON} JSON parse error: {exc}; "
            f"treating all manual_override flags as False.",
            file=sys.stderr,
        )
        return {}
    pods = data.get("pods", {}) or {}
    return {name: bool(entry.get("manual_override", False)) for name, entry in pods.items()}


def _refresh_one_pod(
    name: str,
    row: Pod | None,
    live: PodInfo | None,
    *,
    is_single_mode: bool,
    manual_override: bool,
) -> tuple[bool, bool]:
    """Evaluate one pod for ``cmd_refresh_from_api``.

    Returns ``(changed, warned)``. Mutates ``row.host`` / ``row.port`` in
    place on a clean live-API update. Calls ``sys.exit(1)`` when the named
    pod fails a precondition in single-pod mode (the user explicitly named
    a pod we cannot refresh — silently no-oping would be misleading).

    Precondition order: row exists in pods.conf → pod present in live API →
    ``desiredStatus == RUNNING`` → ``ssh_host``/``ssh_port`` populated →
    ``manual_override`` not set → values actually differ. Any failure in
    bulk mode skips with a stderr WARN (sets ``warned=True``).
    """
    if row is None:
        # Concurrent terminate between main's parse and ours.
        print(
            f"WARN: pod '{name}' no longer in pods.conf (removed by a "
            f"concurrent writer between read and refresh); skipping.",
            file=sys.stderr,
        )
        return False, True

    if live is None:
        msg = (
            f"WARN: pod '{name}' is in pods.conf but not in the live "
            f"RunPod API (terminated externally or never created); "
            f"skipping. Run `pod.py terminate --issue <N>` to clean up "
            f"the stale row, or `pod.py provision` to re-create it."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    ds = (live.desired_status or "").upper()
    if ds != "RUNNING":
        msg = (
            f"WARN: pod '{name}' has desiredStatus={ds or 'UNKNOWN'}, "
            f"not RUNNING; SSH endpoint is not available, skipping. "
            f"Run `pod.py resume --issue <N>` to bring it back, then "
            f"re-run --refresh-from-api."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    if live.ssh_host is None or live.ssh_port is None:
        # RunPod has the pod RUNNING but the 22/tcp mapping isn't up yet
        # (transient). Don't blank out the existing row.
        msg = (
            f"WARN: pod '{name}' is RUNNING but has no public 22/tcp "
            f"mapping yet (transient — wait ~10s and retry); skipping."
        )
        if is_single_mode:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return False, True

    if manual_override:
        if row.host != live.ssh_host or row.port != live.ssh_port:
            print(
                f"WARN: pod '{name}' has manual_override=True; refusing "
                f"to overwrite host/port from API "
                f"(kept {row.host}:{row.port}; API would have written "
                f"{live.ssh_host}:{live.ssh_port}). Clear with "
                f"`pod.py config --clear-override {name}` if the API "
                f"is right.",
                file=sys.stderr,
            )
            return False, True
        return False, False

    if row.host == live.ssh_host and row.port == live.ssh_port:
        print(f"  {name}: already at {row.host}:{row.port} — no change.")
        return False, False

    print(f"  {name}: {row.host}:{row.port} -> {live.ssh_host}:{live.ssh_port}")
    row.host = live.ssh_host
    row.port = live.ssh_port
    return True, False


def cmd_refresh_from_api(pods: list[Pod], pod_name: str | None) -> None:
    """Pull live host/port from the RunPod API and update ``pods.conf``.

    The existing ``--sync`` propagates ``pods.conf`` OUTWARD to ``~/.ssh/config``
    + ``.claude/mcp.json``. There was no inverse direction: nothing pulled
    fresh host/port from the live RunPod API into ``pods.conf``. The gap bit
    task #488 on 2026-06-09 — a SUPPLY_CONSTRAINT-blocked resume hard-exited,
    the pod later came back at a NEW SSH port via a separate retry that did
    not run our success path, and the autonomous session's SSH polling loop
    spun for 13+ hours on the pre-stop port while ``pods.conf`` carried the
    stale value. With this command, the orchestrator (or a human) can force
    a re-sync from the live API and ``cmd_sync`` then propagates the fresh
    values to SSH + MCP.

    Scope:
      * ``pod_name=None`` — refresh every managed pod present in BOTH
        ``pods.conf`` and the live RunPod API. Pods that are not RUNNING are
        skipped with a stderr note (we cannot infer a fresh SSH endpoint for
        a pod that is EXITED/PROVISIONING).
      * ``pod_name=<name>`` — refresh just that pod. Errors loud if the pod
        is not in ``pods.conf`` (typo) or not present in the live API
        (terminated externally) or not RUNNING (cannot refresh an endpoint
        that does not exist yet).

    Respects ``manual_override`` (set by ``--update``): when True, the
    on-disk host/port stays as the user set them and we surface a stderr
    WARN instead of overwriting. Use ``--clear-override <pod>`` to re-enable
    auto-refresh for a manually-pinned pod.

    Holds ``locked_pods_conf`` for the whole read-modify-write-sync sequence
    so a concurrent provision/resume cannot lose-update our changes — the
    same lock discipline ``cmd_update`` uses.

    The live API call is REQUIRED. If the API is unreachable, the underlying
    ``runpod_api.RunPodError`` propagates so callers see a clear failure
    rather than a silent stale-config no-op (fail-fast rule).
    """
    # Import lazily — ``runpod_api`` is the heavy module and importing at
    # module top would force every ``pod_config --check`` / ``--list`` to
    # eagerly load it. The lazy import keeps the cheap subcommands cheap.
    from runpod_api import list_team_pods

    live_pods = list_team_pods()
    live_by_name = {p.name: p for p in live_pods}
    overrides = _read_manual_overrides()

    targets: list[Pod]
    if pod_name is None:
        targets = list(pods)
    else:
        target = next((p for p in pods if p.name == pod_name), None)
        if target is None:
            print(f"ERROR: pod '{pod_name}' not found in pods.conf", file=sys.stderr)
            print(f"Available: {', '.join(p.name for p in pods)}", file=sys.stderr)
            sys.exit(1)
        targets = [target]

    if not targets:
        print("No pods to refresh (pods.conf is empty).")
        return

    is_single_mode = pod_name is not None

    with locked_pods_conf():
        # Re-parse under the lock so we operate on the freshest on-disk view.
        fresh = parse_pods_conf()
        fresh_by_name = {p.name: p for p in fresh}

        any_changed = False
        any_warn = False
        for original in targets:
            name = original.name
            changed, warned = _refresh_one_pod(
                name,
                fresh_by_name.get(name),
                live_by_name.get(name),
                is_single_mode=is_single_mode,
                manual_override=overrides.get(name, False),
            )
            any_changed = any_changed or changed
            any_warn = any_warn or warned

        if not any_changed:
            if not any_warn:
                print("All managed pods already match the live RunPod API.")
            else:
                print(
                    "No host/port changes applied (see warnings above).",
                    file=sys.stderr,
                )
            return

        print()
        print("Updating pods.conf with live API host/port...")
        write_pods_conf(fresh)
        cmd_sync(fresh)


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
            "  python scripts/pod_config.py --refresh-from-api\n"
            "  python scripts/pod_config.py --refresh-from-api pod-488\n"
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
    group.add_argument(
        "--clear-override",
        metavar="POD_NAME",
        help=(
            "Clear manual_override in pods_ephemeral.json for POD_NAME so the "
            "auto-refresh paths in pod_lifecycle.py may resume updating host/"
            "port from the live API."
        ),
    )
    group.add_argument(
        "--refresh-from-api",
        metavar="POD_NAME",
        nargs="?",
        const="__ALL__",
        help=(
            "Pull live host/port from the RunPod API into pods.conf, then "
            "sync to ~/.ssh/config + .claude/mcp.json. Pass a POD_NAME to "
            "refresh just one pod, or omit it to refresh every managed pod. "
            "Respects manual_override (set by --update). Use when a pod has "
            "come back at a new SSH port outside an explicit `pod.py resume` "
            "(e.g. recovery from SUPPLY_CONSTRAINT) and the configs are stale."
        ),
    )

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
    elif args.clear_override:
        cmd_clear_override(args.clear_override)
    elif args.refresh_from_api:
        # ``nargs="?"`` with ``const="__ALL__"`` distinguishes the bare flag
        # (refresh all pods) from the flag-with-arg (refresh one pod).
        target = None if args.refresh_from_api == "__ALL__" else args.refresh_from_api
        cmd_refresh_from_api(pods, target)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
