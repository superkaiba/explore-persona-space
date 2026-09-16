"""Run and independently verify the bounded local seven-setting refit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic(path, data):
    """Replace a durable JSON observation without partial reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def verify(out, source_sha):
    """Check source, coverage, local hashes and the immutable uploaded inventory."""
    from huggingface_hub import HfApi, hf_hub_download

    complete = json.loads((out / "complete.json").read_text())
    result = json.loads((out / "results.json").read_text())
    inventory = json.loads((out / "inventory.json").read_text())
    if any(x["source_sha"] != source_sha for x in (complete, result, inventory)):
        raise ValueError("Completion source mismatch")
    if result["status"] != "complete" or len(result["maps"]) != 5 or len(result["rows"]) != 35:
        raise ValueError("Incomplete evaluation coverage")
    if (
        hashlib.sha256((out / "results.json").read_bytes()).hexdigest()
        != complete["results_sha256"]
    ):
        raise ValueError("Result hash mismatch")
    if (
        hashlib.sha256((out / "inventory.json").read_bytes()).hexdigest()
        != complete["inventory_sha256"]
    ):
        raise ValueError("Inventory hash mismatch")
    prefix = f"issue2054_shared_seven_k5/{out.name}"
    api = HfApi()
    files = [
        *inventory["files"],
        {
            "path": "inventory.json",
            "size": (out / "inventory.json").stat().st_size,
            "sha256": complete["inventory_sha256"],
        },
    ]
    paths = [f"{prefix}/{row['path']}" for row in files]
    remote = {
        e.path: e
        for e in api.get_paths_info(
            "superkaiba1/explore-persona-space-data",
            paths,
            repo_type="dataset",
            revision=complete["verified_revision"],
        )
    }
    for row, path in zip(files, paths, strict=True):
        local = out / row["path"]
        if local.stat().st_size != row["size"]:
            raise ValueError(f"Local size mismatch: {path}")
        digest = hashlib.sha256()
        with local.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"] or remote[path].size != row["size"]:
            raise ValueError(f"Artifact mismatch: {path}")
        lfs = getattr(remote[path], "lfs", None)
        if lfs is not None:
            if lfs.sha256 != row["sha256"]:
                raise ValueError(f"Remote hash mismatch: {path}")
        else:
            downloaded = hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                path,
                repo_type="dataset",
                revision=complete["verified_revision"],
            )
            if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Remote metadata mismatch: {path}")
    return complete


def stop_child(child):
    """Terminate and reap only this monitor's owned child process group."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def main():
    """Observe actual process/log/output progress and surface bounded failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = Path(config["out_root"])
    observation = Path(config["observation"])
    started = time.time()
    log_path = Path(config["log"])
    child = None
    status = "starting"
    try:
        if not (out / "complete.json").exists():
            with log_path.open("a") as log:
                child = subprocess.Popen(
                    config["argv"],
                    cwd=config["workdir"],
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic(
                    Path(config["runtime"]),
                    {
                        "source_sha": config["source_sha"],
                        "started_at": started,
                        "pid": child.pid,
                        "argv": config["argv"],
                    },
                )
                protection = subprocess.run(
                    ["sudo", "-n", "choom", "-n", "-600", "-p", str(child.pid)],
                    capture_output=True,
                    text=True,
                )
                if protection.returncode:
                    child.terminate()
                    child.wait(timeout=30)
                    raise RuntimeError(
                        "Could not apply required OOM protection: " + protection.stderr
                    )
                while child.poll() is None:
                    now = time.time()
                    progress_path = out / "progress.json"
                    progress = (
                        json.loads(progress_path.read_text()) if progress_path.exists() else {}
                    )
                    fresh = max(started, progress.get("checked_at", started))
                    age = now - fresh
                    if age > 1200:
                        child.terminate()
                        child.wait(timeout=30)
                        raise RuntimeError("No experiment output progress for 20 minutes")
                    atomic(
                        observation,
                        {
                            "source_sha": config["source_sha"],
                            "status": "running",
                            "checked_at": now,
                            "backend_observation": {
                                "status": "running",
                                "pid_alive": True,
                                "pid": child.pid,
                                "last_log_mtime_sec_ago": now - log_path.stat().st_mtime,
                                "progress_age_seconds": age,
                                "progress": progress,
                            },
                        },
                    )
                    time.sleep(15 if now - started < 120 else 30)
                if child.returncode:
                    raise RuntimeError(f"Experiment exited {child.returncode}; see {log_path}")
        atomic(
            observation,
            {
                "source_sha": config["source_sha"],
                "status": "verifying",
                "checked_at": time.time(),
                "backend_observation": {"status": "done"},
            },
        )
        result = verify(out, config["source_sha"])
        atomic(
            observation,
            {
                "source_sha": config["source_sha"],
                "status": "complete",
                "checked_at": time.time(),
                "results": result,
                "backend_observation": {"status": "done"},
            },
        )
        status = "complete"
        print(json.dumps(result), flush=True)
    finally:
        if status != "complete":
            stop_child(child)
            atomic(
                observation,
                {
                    "source_sha": config["source_sha"],
                    "status": "failed",
                    "checked_at": time.time(),
                    "backend_observation": {
                        "status": "failed",
                        "pid_alive": child is not None and child.poll() is None,
                        "log": str(log_path),
                    },
                },
            )


if __name__ == "__main__":
    main()
