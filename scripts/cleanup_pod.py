#!/usr/bin/env python3
"""Clean up stale model weights on GPU pods after verifying uploads.

Finds safetensors files outside the HF cache, checks if they're already on
HuggingFace Hub, and optionally uploads + deletes them.

Usage:
    python scripts/cleanup_pod.py pod1 --dry-run        # Show what would be cleaned
    python scripts/cleanup_pod.py pod1                   # Upload unuploaded, then clean
    python scripts/cleanup_pod.py --all --dry-run        # Check all pods
    python scripts/cleanup_pod.py --all --skip-upload    # Only clean already-uploaded models
    python scripts/cleanup_pod.py --hf-cache pod1        # Also clean HF cache blobs
"""

import argparse
import contextlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from explore_persona_space.orchestrate.hub import DEFAULT_MODEL_REPO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONF_PATH = SCRIPT_DIR / "pods.conf"

SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_REPO = DEFAULT_MODEL_REPO

# ── Env setup ────────────────────────────────────────────────────────────────

if Path("/workspace").exists():
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

load_dotenv(str(PROJECT_ROOT / ".env"))


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Pod:
    name: str
    host: str
    port: int
    gpus: int
    gpu_type: str
    label: str


@dataclass
class ModelDir:
    path: str
    size_mb: float
    n_safetensors: int
    on_hub: bool = False
    hub_path: str = ""


@dataclass
class CleanupReport:
    pod: str
    models: list[ModelDir] = field(default_factory=list)
    hf_cache_mb: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_mb(self) -> float:
        return sum(m.size_mb for m in self.models) + self.hf_cache_mb

    @property
    def cleanable_mb(self) -> float:
        return sum(m.size_mb for m in self.models if m.on_hub) + self.hf_cache_mb

    @property
    def needs_upload_mb(self) -> float:
        return sum(m.size_mb for m in self.models if not m.on_hub)


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_pods() -> list[Pod]:
    """Read pods.conf."""
    pods = []
    if not CONF_PATH.exists():
        return pods
    for line in CONF_PATH.read_text().splitlines():
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


def ssh_run(pod: Pod, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a command on a pod via SSH."""
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
        str(pod.port),
        f"root@{pod.host}",
        cmd,
    ]
    try:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def get_hub_paths() -> set[str]:
    """List all top-level directories in the HF model repo."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        tree = api.list_repo_tree(repo_id=DEFAULT_REPO, repo_type="model")
        return {item.rfilename.split("/")[0] for item in tree if "/" in item.rfilename}
    except Exception as e:
        print(f"  Warning: Could not list HF Hub repo: {e}")
        return set()


# ── Core logic ───────────────────────────────────────────────────────────────


def scan_pod(pod: Pod) -> CleanupReport:
    """Find all model weights on a pod that could be cleaned up."""
    report = CleanupReport(pod=pod.name)

    # Find safetensors outside cache
    rc, out, err = ssh_run(
        pod,
        "find /workspace -name '*.safetensors' "
        "-not -path '*/.cache/*' "
        "-not -path '*/node_modules/*' "
        "2>/dev/null",
        timeout=30,
    )
    if rc != 0 and rc != -1:
        report.errors.append(f"find failed: {err[:200]}")
        return report
    if rc == -1:
        report.errors.append(f"Pod unreachable: {err}")
        return report

    if not out:
        return report

    # Group by parent directory
    model_dirs: dict[str, list[str]] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parent = str(Path(line).parent)
        model_dirs.setdefault(parent, []).append(line)

    # Get sizes for each directory
    for dir_path, files in model_dirs.items():
        rc2, size_out, _ = ssh_run(pod, f"du -sm '{dir_path}' 2>/dev/null | cut -f1", timeout=15)
        size_mb = float(size_out) if rc2 == 0 and size_out.strip().isdigit() else 0.0

        report.models.append(
            ModelDir(
                path=dir_path,
                size_mb=size_mb,
                n_safetensors=len(files),
            )
        )

    # Check HF cache blob size
    rc3, cache_out, _ = ssh_run(
        pod,
        "du -sm /workspace/.cache/huggingface/hub/models--*/blobs 2>/dev/null "
        "| awk '{s+=$1} END {print s+0}'",
        timeout=15,
    )
    if rc3 == 0 and cache_out.strip():
        with contextlib.suppress(ValueError):
            report.hf_cache_mb = float(cache_out.strip())

    return report


def check_hub_status(report: CleanupReport, hub_paths: set[str]):
    """Mark which models are already on HF Hub."""
    for model in report.models:
        # Try to match model directory name against hub paths
        dir_name = Path(model.path).name
        # Check various naming conventions
        if dir_name in hub_paths:
            model.on_hub = True
            model.hub_path = dir_name
        else:
            # Check parent/child pattern (e.g., midtrain_25pct/evil_wrong/em_merged)
            # Try last 2 path components as underscore-joined
            parts = Path(model.path).parts
            for i in range(len(parts)):
                candidate = "_".join(parts[i:])
                if candidate in hub_paths:
                    model.on_hub = True
                    model.hub_path = candidate
                    break


def upload_model(pod: Pod, model: ModelDir, repo_id: str = DEFAULT_REPO) -> bool:
    """Upload a model from pod to HF Hub."""
    dir_name = Path(model.path).name
    # Use a reasonable path_in_repo
    hub_path = dir_name

    print(f"    Uploading {model.path} -> {repo_id}/{hub_path} ...")
    _rc, out, err = ssh_run(
        pod,
        f'export PATH="$HOME/.local/bin:$PATH" && '
        f"cd /workspace/explore-persona-space && "
        f'uv run python -c "'
        f"from explore_persona_space.orchestrate.hub import upload_model; "
        f"r = upload_model('{model.path}', path_in_repo='{hub_path}'); "
        f"print('UPLOAD_OK' if r else 'UPLOAD_FAIL')"
        f'"',
        timeout=600,  # Models can take a while to upload
    )
    if "UPLOAD_OK" in out:
        print(f"    Upload complete: {hub_path}")
        model.on_hub = True
        model.hub_path = hub_path
        return True
    else:
        print(f"    Upload failed: {err[:200] if err else out[:200]}")
        return False


def clean_model(pod: Pod, model: ModelDir) -> bool:
    """Delete a model directory from a pod."""
    rc, _, _err = ssh_run(pod, f"rm -rf '{model.path}'", timeout=30)
    return rc == 0


def clean_hf_cache(pod: Pod) -> float:
    """Clean HF cache blobs on a pod. Returns MB freed."""
    rc, out, _ = ssh_run(
        pod,
        "for d in /workspace/.cache/huggingface/hub/models--*/blobs; do "
        '  [ -d "$d" ] && du -sm "$d" | cut -f1 && rm -rf "$d"; '
        "done | awk '{s+=$1} END {print s+0}'",
        timeout=60,
    )
    if rc == 0 and out.strip():
        try:
            return float(out.strip())
        except ValueError:
            pass
    return 0.0


# ── Display ──────────────────────────────────────────────────────────────────

# ANSI colors
C_RED = "\033[0;31m"
C_GREEN = "\033[0;32m"
C_YELLOW = "\033[0;33m"
C_BOLD = "\033[1m"
C_NC = "\033[0m"

if not sys.stdout.isatty():
    C_RED = C_GREEN = C_YELLOW = C_BOLD = C_NC = ""


def print_report(report: CleanupReport):
    """Print a cleanup report."""
    print(f"\n{C_BOLD}[{report.pod}] Cleanup Report{C_NC}")

    if report.errors:
        for e in report.errors:
            print(f"  {C_RED}Error: {e}{C_NC}")
        return

    if not report.models and report.hf_cache_mb == 0:
        print(f"  {C_GREEN}Clean — no stale model weights found{C_NC}")
        return

    print(f"  {'Path':<60} {'Size':>8} {'Files':>6} {'Hub':>5}")
    print(f"  {'-' * 60} {'-' * 8} {'-' * 6} {'-' * 5}")

    for m in sorted(report.models, key=lambda x: -x.size_mb):
        # Shorten path for display
        display_path = m.path
        if len(display_path) > 58:
            display_path = "..." + display_path[-55:]
        hub_icon = f"{C_GREEN}✓{C_NC}" if m.on_hub else f"{C_RED}✗{C_NC}"
        print(f"  {display_path:<60} {m.size_mb:>7.0f}M {m.n_safetensors:>5}  {hub_icon}")

    if report.hf_cache_mb > 0:
        print(f"  {'HF cache blobs':<60} {report.hf_cache_mb:>7.0f}M     -    -")

    print(f"\n  Total:         {report.total_mb:,.0f} MB")
    print(f"  On Hub (safe): {report.cleanable_mb:,.0f} MB")
    print(f"  Needs upload:  {report.needs_upload_mb:,.0f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Clean up model weights on GPU pods")
    parser.add_argument("pods", nargs="*", help="Pod names (e.g., pod1 pod2)")
    parser.add_argument("--all", action="store_true", help="Check all pods")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned")
    parser.add_argument(
        "--skip-upload", action="store_true", help="Only clean already-uploaded models"
    )
    parser.add_argument("--hf-cache", action="store_true", help="Also clean HF cache blobs")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF Hub repo for verification")
    args = parser.parse_args()

    all_pods = parse_pods()
    if not all_pods:
        print("Error: No pods found in pods.conf")
        sys.exit(1)

    # Select pods
    if args.all:
        target_pods = all_pods
    elif args.pods:
        pod_map = {p.name: p for p in all_pods}
        target_pods = []
        for name in args.pods:
            if name in pod_map:
                target_pods.append(pod_map[name])
            else:
                print(f"Warning: Pod '{name}' not found in pods.conf, skipping")
        if not target_pods:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Get HF Hub inventory once
    print("Checking HF Hub inventory...")
    hub_paths = get_hub_paths()
    print(f"  Found {len(hub_paths)} model directories on Hub\n")

    # Scan pods in parallel
    reports: list[CleanupReport] = []
    with ThreadPoolExecutor(max_workers=len(target_pods)) as pool:
        futures = {pool.submit(scan_pod, pod): pod for pod in target_pods}
        for future in as_completed(futures):
            report = future.result()
            check_hub_status(report, hub_paths)
            reports.append(report)
            print_report(report)

    if args.dry_run:
        # Summary
        total_cleanable = sum(r.cleanable_mb for r in reports)
        total_needs_upload = sum(r.needs_upload_mb for r in reports)
        print(f"\n{C_BOLD}Summary (dry run){C_NC}")
        print(f"  Would free:    {total_cleanable:,.0f} MB (already on Hub)")
        print(f"  Needs upload:  {total_needs_upload:,.0f} MB (upload first)")
        return

    # Execute cleanup
    total_freed = 0.0
    for report in reports:
        if report.errors:
            continue

        pod = next(p for p in target_pods if p.name == report.pod)

        for model in report.models:
            if model.on_hub:
                # Safe to clean
                print(f"  [{report.pod}] Cleaning {model.path} ({model.size_mb:.0f} MB)...")
                if clean_model(pod, model):
                    total_freed += model.size_mb
                    print(f"  {C_GREEN}Deleted{C_NC}")
                else:
                    print(f"  {C_RED}Delete failed{C_NC}")
            elif not args.skip_upload:
                # Try to upload first
                if upload_model(pod, model, args.repo):
                    print(f"  [{report.pod}] Cleaning {model.path} ({model.size_mb:.0f} MB)...")
                    if clean_model(pod, model):
                        total_freed += model.size_mb
                        print(f"  {C_GREEN}Uploaded + deleted{C_NC}")
                else:
                    print(f"  {C_YELLOW}Skipping {model.path} — upload failed, keeping local{C_NC}")
            else:
                print(f"  {C_YELLOW}Skipping {model.path} — not on Hub, --skip-upload set{C_NC}")

        if args.hf_cache and report.hf_cache_mb > 0:
            print(f"  [{report.pod}] Cleaning HF cache blobs...")
            freed = clean_hf_cache(pod)
            total_freed += freed
            print(f"  {C_GREEN}Freed {freed:.0f} MB from HF cache{C_NC}")

    print(f"\n{C_BOLD}Total freed: {total_freed:,.0f} MB{C_NC}")


if __name__ == "__main__":
    main()
