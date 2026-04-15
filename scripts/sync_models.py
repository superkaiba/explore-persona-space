"""Sync model checkpoints between GPU pods and HuggingFace Hub.

Usage:
    # List models on HF Hub
    python scripts/sync_models.py --list
    python scripts/sync_models.py --list --prefix midtrain

    # Pull a model to local directory
    python scripts/sync_models.py --pull <path_in_repo> --dest /workspace/models

    # Sweep all pods for unuploaded models and upload them
    python scripts/sync_models.py --sweep
    python scripts/sync_models.py --sweep --pods pod1,pod2

    # Clean uploaded model weights from a pod
    python scripts/sync_models.py --clean pod1 --dry-run
    python scripts/sync_models.py --clean pod1
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PODS_CONF = SCRIPT_DIR / "pods.conf"
DEFAULT_REPO = "superkaiba1/explore-persona-space"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
SSH_TIMEOUT = 10


# ── Pod registry ─────────────────────────────────────────────────────────────


def parse_pods_conf() -> dict[str, dict]:
    """Parse pods.conf into a dict of {name: {host, port, gpus, gpu_type, label}}.

    Skips blank lines and comments (lines starting with #).
    """
    pods = {}
    if not PODS_CONF.exists():
        print(f"Warning: {PODS_CONF} not found")
        return pods

    for line in PODS_CONF.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        name, host, port, gpus, gpu_type, label = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
            parts[5],
        )
        pods[name] = {
            "host": host,
            "port": port,
            "gpus": int(gpus),
            "gpu_type": gpu_type,
            "label": label,
        }
    return pods


# ── SSH helper ───────────────────────────────────────────────────────────────


def ssh_run(pod: dict, command: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a command on a pod via SSH.

    Returns (returncode, stdout, stderr). On connection failure, returncode is -1.
    """
    ssh_cmd = [
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
        pod["port"],
        f"root@{pod['host']}",
        command,
    ]
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "SSH command timed out"
    except Exception as e:
        return -1, "", f"SSH error: {e}"


# ── HF Hub helpers ───────────────────────────────────────────────────────────


def get_api():
    """Get an authenticated HfApi instance."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)
    return HfApi(token=token)


def get_hub_models(repo_id: str = DEFAULT_REPO, prefix: str = "") -> list[dict]:
    """List top-level model directories on HF Hub.

    Returns list of dicts with keys: path, size, last_modified.
    """
    from huggingface_hub.utils import HfHubHTTPError

    api = get_api()

    try:
        items = list(api.list_repo_tree(repo_id=repo_id, repo_type="model"))
    except HfHubHTTPError as e:
        print(f"Error listing repo {repo_id}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error listing repo {repo_id}: {e}", file=sys.stderr)
        return []

    models = []
    for item in items:
        # Only top-level directories (model folders)
        if item.type != "directory":
            continue
        path = item.path if hasattr(item, "path") else str(item)
        if prefix and not path.startswith(prefix):
            continue
        entry = {
            "path": path,
            "last_modified": getattr(item, "last_commit", None),
        }
        models.append(entry)

    return sorted(models, key=lambda x: x["path"])


def hub_path_exists(path_in_repo: str, repo_id: str = DEFAULT_REPO) -> bool:
    """Check if a path exists in the HF Hub model repo."""
    from huggingface_hub.utils import HfHubHTTPError

    api = get_api()
    try:
        items = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=path_in_repo,
            )
        )
        return len(items) > 0
    except (HfHubHTTPError, Exception):
        return False


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_list(args):
    """List models on HF Hub."""
    models = get_hub_models(repo_id=args.repo, prefix=args.prefix)
    if not models:
        print("No models found on HF Hub.")
        return

    print(f"\nModels on {args.repo} ({len(models)} directories):")
    print(f"{'Path':<60} {'Last Modified':<25}")
    print("-" * 85)

    for m in models:
        last_mod = ""
        if m["last_modified"]:
            commit = m["last_modified"]
            # last_commit can be a RepoFolder's last_commit attr (a CommitInfo)
            last_mod = str(commit.date)[:19] if hasattr(commit, "date") else str(commit)[:19]
        print(f"{m['path']:<60} {last_mod:<25}")

    print(f"\nTotal: {len(models)} model directories")


def cmd_pull(args):
    """Pull a model from HF Hub to local directory."""
    from huggingface_hub import snapshot_download

    path_in_repo = args.pull
    dest = args.dest

    print(f"Downloading {args.repo}/{path_in_repo} -> {dest}")

    try:
        local_path = snapshot_download(
            repo_id=args.repo,
            allow_patterns=f"{path_in_repo}/*",
            local_dir=dest,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Download complete: {local_path}")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


def _find_models_on_pod(pod_name: str, pod: dict) -> tuple[str, list[str], str]:
    """SSH to a pod and find directories containing safetensors (outside cache/checkpoints).

    Returns (pod_name, list_of_model_dirs, error_message).
    """
    find_cmd = (
        "find /workspace -name '*.safetensors' "
        "-not -path '*/.cache/*' "
        "-not -path '*/checkpoint-*' "
        "2>/dev/null"
    )
    rc, stdout, stderr = ssh_run(pod, find_cmd, timeout=60)
    if rc != 0:
        return pod_name, [], stderr.strip() or "SSH connection failed"

    # Group by parent directory
    model_dirs = set()
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line:
            model_dirs.add(str(Path(line).parent))

    return pod_name, sorted(model_dirs), ""


def _derive_hub_name(model_dir: str) -> str:
    """Derive a HF Hub path name from a local model directory path.

    Tries to extract a meaningful name from the directory structure.
    Examples:
        /workspace/explore-persona-space/models/midtrain_evil_seed42 -> midtrain_evil_seed42
        /workspace/models/c1_evil_wrong_em_seed42/merged -> c1_evil_wrong_em_seed42
        /workspace/something/my_model -> my_model
    """
    p = Path(model_dir)

    # If the directory is "merged", use the parent name
    if p.name == "merged":
        p = p.parent

    # If directory is inside a known project structure, take just the model name
    # e.g. /workspace/explore-persona-space/models/<name>
    parts = p.parts
    for i, part in enumerate(parts):
        if part in ("models", "outputs", "runs"):
            # Use everything after this directory as the hub name
            remaining = "/".join(parts[i + 1 :])
            if remaining:
                return remaining

    # Fallback: use just the directory name
    return p.name


def _filter_pods(pods: dict[str, dict], pod_filter: str | None) -> dict[str, dict]:
    """Filter pods dict by a comma-separated name list. Exits on unknown names."""
    if not pod_filter:
        return pods
    requested = [p.strip() for p in pod_filter.split(",")]
    missing = [p for p in requested if p not in pods]
    if missing:
        print(f"Unknown pods: {', '.join(missing)}")
        print(f"Available: {', '.join(sorted(pods.keys()))}")
        sys.exit(1)
    return {k: v for k, v in pods.items() if k in requested}


def _scan_pods(pods: dict[str, dict]) -> dict[str, tuple[list[str], str]]:
    """Scan all pods in parallel for model directories.

    Returns {pod_name: (model_dirs, error)}.
    """
    results: dict[str, tuple[list[str], str]] = {}
    with ThreadPoolExecutor(max_workers=len(pods)) as pool:
        futures = {pool.submit(_find_models_on_pod, name, pod): name for name, pod in pods.items()}
        for future in as_completed(futures):
            pod_name, model_dirs, error = future.result()
            if error:
                print(f"  {pod_name}: UNREACHABLE ({error})")
                results[pod_name] = ([], error)
            else:
                print(f"  {pod_name}: found {len(model_dirs)} model directories")
                results[pod_name] = (model_dirs, "")
    return results


def _upload_one_model(pod: dict, pod_name: str, model_dir: str, hub_name: str) -> bool:
    """Upload a single model from a pod to HF Hub. Returns True on success."""
    print(f"\n  Uploading [{pod_name}] {model_dir} as {hub_name}...")

    upload_cmd = (
        f"cd /workspace/explore-persona-space && "
        f'python -c "'
        f"from explore_persona_space.orchestrate.hub import upload_model; "
        f"result = upload_model('{model_dir}', path_in_repo='{hub_name}'); "
        f"print('UPLOAD_OK' if result else 'UPLOAD_FAIL')"
        f'"'
    )
    rc, stdout, stderr = ssh_run(pod, upload_cmd, timeout=600)

    if rc == 0 and "UPLOAD_OK" in stdout:
        print("  -> Uploaded successfully")
        return True

    print(f"  -> FAILED (rc={rc})")
    if stderr.strip():
        for line in stderr.strip().splitlines()[-5:]:
            print(f"     {line}")
    if stdout.strip() and "UPLOAD_FAIL" in stdout:
        for line in stdout.strip().splitlines()[-5:]:
            print(f"     {line}")
    return False


def cmd_sweep(args):
    """Sweep pods for unuploaded models and upload them."""
    pods = parse_pods_conf()
    if not pods:
        print("No pods found in pods.conf")
        return

    pods = _filter_pods(pods, args.pods)

    # Fetch current hub models for comparison
    print("Fetching current models from HF Hub...")
    hub_models = get_hub_models(repo_id=args.repo)
    hub_paths = {m["path"] for m in hub_models}
    print(f"Found {len(hub_paths)} existing models on Hub\n")

    # Scan pods in parallel
    print(f"Scanning {len(pods)} pods for unuploaded models...")
    pod_results = _scan_pods(pods)

    # Identify unuploaded models
    print("\n--- Unuploaded Models ---")
    to_upload: list[tuple[str, str, str]] = []  # (pod_name, model_dir, hub_name)

    for pod_name, (model_dirs, error) in sorted(pod_results.items()):
        if error:
            continue
        for model_dir in model_dirs:
            hub_name = _derive_hub_name(model_dir)
            if hub_name not in hub_paths:
                to_upload.append((pod_name, model_dir, hub_name))
                print(f"  [{pod_name}] {model_dir} -> {hub_name}")

    if not to_upload:
        print("  All models already on HF Hub.")
        return

    print(f"\nFound {len(to_upload)} unuploaded models.")

    if args.dry_run:
        print("[dry-run] No uploads performed.")
        return

    # Upload each model
    print("\nUploading...")
    succeeded = 0
    failed = 0
    for pod_name, model_dir, hub_name in to_upload:
        if _upload_one_model(pods[pod_name], pod_name, model_dir, hub_name):
            succeeded += 1
            hub_paths.add(hub_name)
        else:
            failed += 1

    print(
        f"\nSweep complete: {succeeded} uploaded, {failed} failed, "
        f"{len(to_upload) - succeeded - failed} skipped"
    )


def cmd_clean(args):
    """Clean uploaded model weights from a pod."""
    pods = parse_pods_conf()
    pod_name = args.clean

    if pod_name not in pods:
        print(f"Unknown pod: {pod_name}")
        print(f"Available: {', '.join(sorted(pods.keys()))}")
        sys.exit(1)

    pod = pods[pod_name]

    # Find models on the pod
    print(f"Scanning {pod_name} for model weights...")
    _, model_dirs, error = _find_models_on_pod(pod_name, pod)

    if error:
        print(f"Cannot reach {pod_name}: {error}")
        sys.exit(1)

    if not model_dirs:
        print(f"No model weight directories found on {pod_name}.")
        return

    print(f"Found {len(model_dirs)} model directories\n")

    # Fetch current hub models
    print("Verifying against HF Hub...")
    hub_models = get_hub_models(repo_id=args.repo)
    hub_paths = {m["path"] for m in hub_models}

    # Categorize each directory
    can_clean: list[tuple[str, str]] = []  # (model_dir, hub_name)
    not_uploaded: list[tuple[str, str]] = []

    for model_dir in model_dirs:
        hub_name = _derive_hub_name(model_dir)
        if hub_name in hub_paths:
            can_clean.append((model_dir, hub_name))
        else:
            not_uploaded.append((model_dir, hub_name))

    if not_uploaded:
        print(f"\nWARNING: {len(not_uploaded)} models NOT on HF Hub (will NOT delete):")
        for model_dir, hub_name in not_uploaded:
            print(f"  {model_dir} (expected hub path: {hub_name})")

    if not can_clean:
        print("\nNo models safe to clean (all are either not uploaded or not found).")
        return

    # Get sizes before deleting
    print(f"\nModels safe to clean ({len(can_clean)}):")
    total_size_str = ""
    for model_dir, hub_name in can_clean:
        rc, stdout, _ = ssh_run(pod, f"du -sh {model_dir} 2>/dev/null", timeout=30)
        size = stdout.split()[0] if rc == 0 and stdout.strip() else "?"
        print(f"  {model_dir} ({size}) -- verified on Hub as {hub_name}")

    # Get total size
    dirs_str = " ".join(d for d, _ in can_clean)
    rc, stdout, _ = ssh_run(pod, f"du -shc {dirs_str} 2>/dev/null | tail -1", timeout=30)
    if rc == 0 and stdout.strip():
        total_size_str = stdout.strip().split()[0]
        print(f"\nTotal to free: {total_size_str}")

    if args.dry_run:
        print("\n[dry-run] No files deleted.")
        return

    # Confirm deletion
    print(f"\nDeleting {len(can_clean)} model directories from {pod_name}...")
    succeeded = 0
    for model_dir, _hub_name in can_clean:
        rc, _, stderr = ssh_run(pod, f"rm -rf {model_dir}", timeout=60)
        if rc == 0:
            print(f"  Deleted: {model_dir}")
            succeeded += 1
        else:
            print(f"  FAILED to delete {model_dir}: {stderr.strip()}")

    print(f"\nCleanup complete: {succeeded}/{len(can_clean)} directories removed")

    # Show disk usage after cleanup
    rc, stdout, _ = ssh_run(pod, "df -h /workspace | tail -1", timeout=15)
    if rc == 0 and stdout.strip():
        print(f"Disk usage after cleanup: {stdout.strip()}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Sync model checkpoints between GPU pods and HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --list                            List all models on HF Hub
  %(prog)s --list --prefix midtrain          Filter by prefix
  %(prog)s --pull midtrain_evil_seed42       Download a model
  %(prog)s --sweep                           Upload unuploaded models from all pods
  %(prog)s --sweep --pods pod1,pod2          Sweep specific pods
  %(prog)s --clean pod1 --dry-run            Preview cleanup
  %(prog)s --clean pod1                      Delete uploaded weights from pod
""",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List models on HF Hub")
    group.add_argument("--pull", type=str, metavar="PATH", help="Pull a model from HF Hub")
    group.add_argument("--sweep", action="store_true", help="Find and upload unuploaded models")
    group.add_argument("--clean", type=str, metavar="POD", help="Clean uploaded weights from a pod")

    parser.add_argument("--prefix", default="", help="Filter --list by prefix")
    parser.add_argument("--dest", default=".", help="Local destination for --pull (default: .)")
    parser.add_argument("--pods", type=str, help="Comma-separated pod names for --sweep")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"HF Hub repo (default: {DEFAULT_REPO})"
    )

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    elif args.pull:
        cmd_pull(args)
    elif args.sweep:
        cmd_sweep(args)
    elif args.clean:
        cmd_clean(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
