#!/usr/bin/env python3
"""Fleet-wide health checker for GPU pods.

Checks all pods in parallel via SSH and reports status in a color-coded table.

Usage:
    python scripts/fleet_health.py                   # Full check, table output
    python scripts/fleet_health.py --quick            # Just reachability + GPU state + disk
    python scripts/fleet_health.py --json             # Machine-readable JSON output
    python scripts/fleet_health.py --fix              # Auto-fix: git pull, uv sync, push .env
    python scripts/fleet_health.py --pods pod1,pod3   # Check specific pods only
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
PODS_CONF = SCRIPT_DIR / "pods.conf"
PROJECT_DIR = SCRIPT_DIR.parent
LOCAL_ENV = PROJECT_DIR / ".env"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
REMOTE_PROJECT = "/workspace/explore-persona-space"

REQUIRED_ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BATCH_KEY",
    "WANDB_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "OVERLEAF_GIT_TOKEN",
    "RUNPOD_API_KEY",
]

SSH_TIMEOUT = 10  # seconds for connection timeout
CMD_TIMEOUT = 30  # seconds for command execution timeout

PDT = ZoneInfo("America/Los_Angeles")


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_USE_COLOR: bool | None = None


def _color_enabled() -> bool:
    global _USE_COLOR
    if _USE_COLOR is None:
        _USE_COLOR = sys.stdout.isatty()
    return _USE_COLOR


def _c(code: str, text: str) -> str:
    if _color_enabled():
        return f"\033[{code}m{text}\033[0m"
    return text


def green(text: str) -> str:
    return _c("32", text)


def red(text: str) -> str:
    return _c("31", text)


def yellow(text: str) -> str:
    return _c("33", text)


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Pod:
    name: str
    host: str
    port: int
    gpus: int
    gpu_type: str
    label: str


@dataclass
class GpuInfo:
    index: int
    memory_used_mib: int
    memory_total_mib: int
    has_process: bool


@dataclass
class PodHealth:
    pod: Pod
    reachable: bool = False
    # Git
    git_branch: str | None = None
    git_behind: int | None = None
    git_dirty: bool | None = None
    # Environment
    uv_installed: bool = False
    env_synced: bool | None = None
    env_sync_detail: str | None = None
    env_keys_present: list[str] = field(default_factory=list)
    env_keys_missing: list[str] = field(default_factory=list)
    # Disk
    disk_free_gb: int | None = None
    # GPU
    gpu_info: list[GpuInfo] = field(default_factory=list)
    # Models
    leftover_models: int | None = None
    # Errors
    errors: list[str] = field(default_factory=list)
    # Fix results
    fixes_applied: list[str] = field(default_factory=list)

    @property
    def git_ok(self) -> bool:
        return (
            self.git_branch == "main"
            and self.git_behind is not None
            and self.git_behind == 0
            and self.git_dirty is not None
            and not self.git_dirty
        )

    @property
    def env_ok(self) -> bool:
        return self.uv_installed and self.env_synced is True

    @property
    def keys_ok(self) -> bool:
        return len(self.env_keys_missing) == 0

    @property
    def gpus_idle(self) -> int:
        return sum(1 for g in self.gpu_info if not g.has_process)

    @property
    def gpus_total(self) -> int:
        return len(self.gpu_info)

    @property
    def healthy(self) -> bool:
        return self.reachable and self.git_ok and self.env_ok and self.keys_ok

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add computed fields
        d["git_ok"] = self.git_ok
        d["env_ok"] = self.env_ok
        d["keys_ok"] = self.keys_ok
        d["gpus_idle"] = self.gpus_idle
        d["gpus_total"] = self.gpus_total
        d["healthy"] = self.healthy
        # Flatten pod info
        d["pod_name"] = self.pod.name
        d["pod_host"] = self.pod.host
        d["pod_port"] = self.pod.port
        d["pod_gpu_type"] = self.pod.gpu_type
        d["pod_label"] = self.pod.label
        return d


# ---------------------------------------------------------------------------
# Pod config parsing
# ---------------------------------------------------------------------------


def parse_pods_conf(path: Path = PODS_CONF) -> list[Pod]:
    """Parse pods.conf file into Pod objects."""
    pods = []
    if not path.exists():
        raise FileNotFoundError(f"pods.conf not found at {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        pods.append(
            Pod(
                name=parts[0],
                host=parts[1],
                port=int(parts[2]),
                gpus=int(parts[3]),
                gpu_type=parts[4],
                label=parts[5],
            )
        )
    return pods


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------


def ssh_cmd(pod: Pod, command: str, timeout: int = CMD_TIMEOUT) -> tuple[int, str, str]:
    """Run a command on a pod via SSH. Returns (returncode, stdout, stderr)."""
    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={SSH_TIMEOUT}",
        "-o",
        "BatchMode=yes",
        "-i",
        str(SSH_KEY),
        "-p",
        str(pod.port),
        f"root@{pod.host}",
        command,
    ]
    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "SSH command timed out"
    except Exception as e:
        return -1, "", str(e)


def scp_to_pod(pod: Pod, local_path: Path, remote_path: str) -> tuple[int, str]:
    """SCP a file to a pod. Returns (returncode, stderr)."""
    scp_args = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={SSH_TIMEOUT}",
        "-o",
        "BatchMode=yes",
        "-i",
        str(SSH_KEY),
        "-P",
        str(pod.port),
        str(local_path),
        f"root@{pod.host}:{remote_path}",
    ]
    try:
        result = subprocess.run(
            scp_args,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
        )
        return result.returncode, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "SCP timed out"
    except Exception as e:
        return -1, str(e)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_reachable(pod: Pod) -> bool:
    """Check if pod is reachable via SSH."""
    rc, _, _ = ssh_cmd(pod, "echo ok", timeout=SSH_TIMEOUT + 5)
    return rc == 0


def check_git_branch(pod: Pod) -> str | None:
    rc, out, _ = ssh_cmd(pod, f"git -C {REMOTE_PROJECT} rev-parse --abbrev-ref HEAD")
    if rc == 0:
        return out
    return None


def check_git_behind(pod: Pod) -> int | None:
    rc, _, _ = ssh_cmd(pod, f"git -C {REMOTE_PROJECT} fetch -q origin", timeout=60)
    if rc != 0:
        return None
    rc, out, _ = ssh_cmd(pod, f"git -C {REMOTE_PROJECT} rev-list --count HEAD..origin/main")
    if rc == 0:
        try:
            return int(out)
        except ValueError:
            return None
    return None


def check_git_dirty(pod: Pod) -> bool | None:
    rc, _out, _ = ssh_cmd(pod, f"git -C {REMOTE_PROJECT} status --porcelain | grep -v '^??'")
    if rc == 0:
        # grep matched something -> dirty
        return True
    elif rc == 1:
        # grep found nothing -> clean
        return False
    # rc > 1 -> error
    return None


def check_uv_installed(pod: Pod) -> bool:
    rc, _, _ = ssh_cmd(pod, "command -v uv")
    return rc == 0


def check_env_synced(pod: Pod) -> tuple[bool | None, str | None]:
    rc, out, err = ssh_cmd(
        pod,
        f"cd {REMOTE_PROJECT} && uv sync --locked --dry-run 2>&1",
        timeout=60,
    )
    if rc != 0:
        return None, err or out
    combined = out + "\n" + err
    # If dry-run output contains "would install" or "would uninstall", env is not synced
    if "would install" in combined.lower() or "would uninstall" in combined.lower():
        return False, combined
    return True, None


def check_env_keys(pod: Pod) -> tuple[list[str], list[str]]:
    rc, out, _ = ssh_cmd(pod, f"grep -oP '^[A-Z_]+(?==)' {REMOTE_PROJECT}/.env")
    if rc != 0:
        return [], list(REQUIRED_ENV_KEYS)
    present_keys = set(out.splitlines())
    found = [k for k in REQUIRED_ENV_KEYS if k in present_keys]
    missing = [k for k in REQUIRED_ENV_KEYS if k not in present_keys]
    return found, missing


def check_disk_free(pod: Pod) -> int | None:
    rc, out, _ = ssh_cmd(pod, "df -BG /workspace --output=avail | tail -1")
    if rc == 0:
        # Output like "  216G" — strip and parse
        try:
            return int(out.strip().rstrip("G"))
        except ValueError:
            return None
    return None


def _parse_gpu_list(out: str) -> list[GpuInfo]:
    """Parse nvidia-smi GPU query output into GpuInfo objects."""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=idx, memory_used_mib=mem_used, memory_total_mib=mem_total, has_process=False
            )
        )
    return gpus


def _mark_busy_gpus(pod: Pod, gpus: list[GpuInfo]) -> None:
    """Query compute processes and mark busy GPUs in-place."""
    rc, out_apps, _ = ssh_cmd(
        pod,
        "nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null",
    )
    if rc != 0 or not out_apps.strip():
        return

    rc2, out_map, _ = ssh_cmd(
        pod,
        "nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader 2>/dev/null",
    )
    if rc2 != 0:
        return

    idx_by_bus: dict[str, int] = {}
    for row in out_map.splitlines():
        rparts = [p.strip() for p in row.split(",")]
        if len(rparts) >= 2:
            with contextlib.suppress(ValueError):
                idx_by_bus[rparts[1]] = int(rparts[0])

    busy_indices = {idx_by_bus[b.strip()] for b in out_apps.splitlines() if b.strip() in idx_by_bus}
    for g in gpus:
        if g.index in busy_indices:
            g.has_process = True


def check_gpu_state(pod: Pod) -> list[GpuInfo]:
    rc, out, _ = ssh_cmd(
        pod,
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
    )
    if rc != 0:
        return []

    gpus = _parse_gpu_list(out)
    _mark_busy_gpus(pod, gpus)
    return gpus


def check_leftover_models(pod: Pod) -> int | None:
    rc, out, _ = ssh_cmd(
        pod,
        "find /workspace -name '*.safetensors' -not -path '*/cache/*' 2>/dev/null | wc -l",
        timeout=60,
    )
    if rc == 0:
        try:
            return int(out.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Full pod check
# ---------------------------------------------------------------------------


def check_pod(pod: Pod, quick: bool = False) -> PodHealth:
    """Run all health checks on a single pod."""
    health = PodHealth(pod=pod)

    # 1. Reachability
    health.reachable = check_reachable(pod)
    if not health.reachable:
        return health

    # Quick mode: just reachability + GPU + disk
    if quick:
        health.disk_free_gb = check_disk_free(pod)
        health.gpu_info = check_gpu_state(pod)
        return health

    # 2. Git checks
    health.git_branch = check_git_branch(pod)
    health.git_behind = check_git_behind(pod)
    health.git_dirty = check_git_dirty(pod)

    # 3. Environment checks
    health.uv_installed = check_uv_installed(pod)
    if health.uv_installed:
        health.env_synced, health.env_sync_detail = check_env_synced(pod)

    # 4. .env keys
    health.env_keys_present, health.env_keys_missing = check_env_keys(pod)

    # 5. Disk
    health.disk_free_gb = check_disk_free(pod)

    # 6. GPU state
    health.gpu_info = check_gpu_state(pod)

    # 7. Leftover models
    health.leftover_models = check_leftover_models(pod)

    return health


# ---------------------------------------------------------------------------
# Fix mode
# ---------------------------------------------------------------------------


def fix_pod(health: PodHealth) -> PodHealth:
    """Attempt to fix issues found on a pod. Mutates and returns the health object."""
    pod = health.pod

    if not health.reachable:
        return health

    # Fix: git not on main
    if health.git_branch is not None and health.git_branch != "main":
        rc, _, err = ssh_cmd(pod, f"git -C {REMOTE_PROJECT} checkout main")
        if rc == 0:
            health.fixes_applied.append(f"Checked out main (was on {health.git_branch})")
            health.git_branch = "main"
        else:
            health.errors.append(f"Failed to checkout main: {err}")

    # Fix: git behind
    if health.git_behind is not None and health.git_behind > 0:
        rc, _, err = ssh_cmd(
            pod,
            f"git -C {REMOTE_PROJECT} pull --ff-only origin main",
            timeout=60,
        )
        if rc == 0:
            health.fixes_applied.append(f"Pulled {health.git_behind} commit(s) from origin/main")
            health.git_behind = 0
        else:
            health.errors.append(f"Failed to git pull: {err}")

    # Fix: uv not installed
    if not health.uv_installed:
        rc, _, err = ssh_cmd(
            pod,
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            timeout=120,
        )
        if rc == 0:
            health.fixes_applied.append("Installed uv")
            health.uv_installed = True
        else:
            health.errors.append(f"Failed to install uv: {err}")

    # Fix: env not synced
    if health.uv_installed and health.env_synced is not True:
        rc, _, err = ssh_cmd(
            pod,
            f"cd {REMOTE_PROJECT} && uv sync --locked",
            timeout=300,
        )
        if rc == 0:
            health.fixes_applied.append("Ran uv sync --locked")
            health.env_synced = True
            health.env_sync_detail = None
        else:
            health.errors.append(f"Failed to uv sync: {err}")

    # Fix: .env missing keys
    if health.env_keys_missing and LOCAL_ENV.exists():
        rc, err = scp_to_pod(pod, LOCAL_ENV, f"{REMOTE_PROJECT}/.env")
        if rc == 0:
            health.fixes_applied.append("Pushed .env to pod")
            health.env_keys_present = list(REQUIRED_ENV_KEYS)
            health.env_keys_missing = []
        else:
            health.errors.append(f"Failed to push .env: {err}")

    return health


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format_disk(gb: int | None) -> str:
    if gb is None:
        return "-"
    if gb >= 1000:
        return f"{gb / 1000:.1f}T"
    return f"{gb}G"


def _status_icon(ok: bool, partial: bool = False) -> str:
    if ok:
        return green("\u2713")
    if partial:
        return yellow("\u26a0")
    return red("\u2717")


def _gpu_summary(health: PodHealth) -> str:
    if not health.gpu_info:
        return "-"
    idle = health.gpus_idle
    total = health.gpus_total
    if idle == total:
        return green(f"{total}/{total} idle")
    elif idle == 0:
        return red(f"0/{total} idle")
    else:
        return yellow(f"{idle}/{total} idle")


def _git_status(health: PodHealth) -> str:
    if not health.reachable:
        return "-"
    parts = []
    ok = True
    partial = False

    if health.git_branch != "main":
        parts.append(f"on {health.git_branch}")
        ok = False
    if health.git_behind is not None and health.git_behind > 0:
        parts.append(f"{health.git_behind} behind")
        ok = False
        partial = True
    if health.git_dirty:
        parts.append("dirty")
        ok = False
        partial = True

    if ok and not parts:
        return _status_icon(True)
    if parts:
        detail = ", ".join(parts)
        icon = _status_icon(False, partial)
        return f"{icon} {detail}"
    return _status_icon(True)


def _env_status(health: PodHealth) -> str:
    if not health.reachable:
        return "-"
    if not health.uv_installed:
        return red("\u2717 no uv")
    if health.env_synced is None:
        return yellow("\u26a0 ?")
    if health.env_synced:
        return _status_icon(True)
    return red("\u2717 drift")


def _keys_status(health: PodHealth) -> str:
    if not health.reachable:
        return "-"
    n_missing = len(health.env_keys_missing)
    if n_missing == 0:
        return _status_icon(True)
    if n_missing == len(REQUIRED_ENV_KEYS):
        return red("\u2717 no .env")
    return yellow(f"\u26a0 {n_missing} missing")


def _models_str(health: PodHealth) -> str:
    if not health.reachable:
        return "-"
    if health.leftover_models is None:
        return "-"
    n = health.leftover_models
    if n == 0:
        return green("0")
    return yellow(str(n))


def print_table(results: list[PodHealth], quick: bool = False) -> None:
    now = datetime.now(PDT)
    timestamp = now.strftime("%Y-%m-%d %H:%M %Z")

    print()
    print(bold(f"Pod Fleet Health Check \u2014 {timestamp}"))
    print("\u2550" * 78)

    if quick:
        header = f" {'Pod':<14} {'Reach':>5}  {'Disk':>8}  {'GPUs':<16}"
        print(dim(header))
        print("\u2500" * 78)

        for h in results:
            pod_label = f"{h.pod.name} {h.pod.gpu_type}"
            reach = _status_icon(h.reachable)
            disk = _format_disk(h.disk_free_gb) if h.reachable else "-"
            gpus = _gpu_summary(h) if h.reachable else "-"
            print(f" {pod_label:<14} {reach:>5}  {disk:>8}  {gpus:<16}")
    else:
        header = (
            f" {'Pod':<14} {'Reach':>5}  {'Git':>12}  {'Env':>10}"
            f"  {'Keys':>12}  {'Disk':>6}  {'GPUs':<14}  {'Models':>6}"
        )
        print(dim(header))
        print("\u2500" * 78)

        for h in results:
            pod_label = f"{h.pod.name} {h.pod.gpu_type}"
            reach = _status_icon(h.reachable)
            git = _git_status(h) if h.reachable else "-"
            env = _env_status(h) if h.reachable else "-"
            keys = _keys_status(h) if h.reachable else "-"
            disk = _format_disk(h.disk_free_gb) if h.reachable else "-"
            gpus = _gpu_summary(h)
            models = _models_str(h)

            row = (
                f" {pod_label:<14} {reach:>5}  {git:>12}  {env:>10}"
                f"  {keys:>12}  {disk:>6}  {gpus:<14}  {models:>6}"
            )
            print(row)

    print("\u2550" * 78)

    # Summary line
    total = len(results)
    healthy = sum(1 for h in results if h.reachable and (quick or h.healthy))
    unreachable = sum(1 for h in results if not h.reachable)
    parts = [f"{healthy}/{total} healthy"]
    if unreachable:
        parts.append(red(f"{unreachable} unreachable"))
    if not quick:
        total_models = sum(h.leftover_models or 0 for h in results)
        if total_models > 0:
            parts.append(yellow(f"{total_models} leftover model files"))
    print(f"Summary: {', '.join(parts)}")

    # Print fixes if any
    any_fixes = any(h.fixes_applied for h in results)
    if any_fixes:
        print()
        print(bold("Fixes applied:"))
        for h in results:
            for fix in h.fixes_applied:
                print(f"  {h.pod.name}: {green(fix)}")

    # Print errors if any
    any_errors = any(h.errors for h in results)
    if any_errors:
        print()
        print(bold("Errors:"))
        for h in results:
            for err in h.errors:
                print(f"  {h.pod.name}: {red(err)}")

    print()


def print_json(results: list[PodHealth]) -> None:
    now = datetime.now(tz=ZoneInfo("UTC"))
    output = {
        "timestamp": now.isoformat(),
        "pods": [h.to_dict() for h in results],
        "summary": {
            "total": len(results),
            "healthy": sum(1 for h in results if h.healthy),
            "reachable": sum(1 for h in results if h.reachable),
            "unreachable": sum(1 for h in results if not h.reachable),
            "leftover_models": sum(h.leftover_models or 0 for h in results),
        },
    }
    json.dump(output, sys.stdout, indent=2, default=str)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fleet-wide GPU pod health checker")
    parser.add_argument(
        "--quick", action="store_true", help="Quick check: reachability + GPU + disk only"
    )
    parser.add_argument("--json", action="store_true", help="Output as machine-readable JSON")
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues (git pull, uv sync, push .env)"
    )
    parser.add_argument(
        "--pods", type=str, default=None, help="Comma-separated list of pod names to check"
    )
    args = parser.parse_args()

    # Parse pod config
    all_pods = parse_pods_conf()

    # Filter pods if requested
    if args.pods:
        requested = set(args.pods.split(","))
        pods = [p for p in all_pods if p.name in requested]
        unknown = requested - {p.name for p in pods}
        if unknown:
            print(f"Warning: unknown pod(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        if not pods:
            print("Error: no matching pods found", file=sys.stderr)
            sys.exit(1)
    else:
        pods = all_pods

    # Run checks in parallel
    results: list[PodHealth] = []
    with ThreadPoolExecutor(max_workers=len(pods)) as pool:
        futures = {pool.submit(check_pod, pod, args.quick): pod for pod in pods}
        for future in as_completed(futures):
            pod = futures[future]
            try:
                health = future.result()
            except Exception as e:
                health = PodHealth(pod=pod)
                health.errors.append(f"Check failed: {e}")
            results.append(health)

    # Sort by pod name for consistent output
    results.sort(key=lambda h: h.pod.name)

    # Fix mode: attempt repairs then re-check
    if args.fix:
        for health in results:
            fix_pod(health)

    # Output
    if args.json:
        print_json(results)
    else:
        print_table(results, quick=args.quick)

    # Exit code: non-zero if any pod is unhealthy
    if args.quick:
        unhealthy = any(not h.reachable for h in results)
    else:
        unhealthy = any(not h.healthy for h in results)
    sys.exit(1 if unhealthy else 0)


if __name__ == "__main__":
    main()
