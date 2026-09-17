"""Observe task 2673 with fresh backend, process, log, and artifact evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def write_json(path: Path, value: dict) -> None:
    """Publish a complete observation atomically for the independent watchdog."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(argv, *, cwd=None, stdin=None, timeout=150):
    """Bound and reap one owned probe process group, preserving failures."""
    child = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                break
            try:
                child.communicate(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        child.wait(timeout=5)
        raise
    if child.returncode:
        raise RuntimeError(f"probe failed rc={child.returncode}: {stderr[-1000:]}")
    return stdout


def remote_probe(out: Path, log: Path) -> dict:
    """Inspect only this pilot's processes, artifacts, and model-cache progress."""
    now = time.time()
    pids, stages = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(
            arg.endswith(
                (
                    b"/story_persona_qwen38_pilot.py",
                    b"/story_persona_qwen38_artifacts.py",
                    b"/story_persona_qwen38_workload.sh",
                )
            )
            for arg in argv
        ):
            pids.append(int(proc.name))
            if b"--phase" in argv:
                position = argv.index(b"--phase") + 1
                stages.append(argv[position].decode())
    values = {}
    for name in ("progress.json", "smoke.json", "capture_complete.json"):
        path = out / name
        if path.exists():
            values[name] = json.loads(path.read_text())
    chunks = list((out / "chunks").glob("batch_*.pt"))
    cache = Path("/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/blobs")
    cache_files = list(cache.glob("*")) if cache.exists() else []
    local_outputs = list(out.iterdir()) if out.exists() else []
    activity = [p.stat().st_mtime for p in chunks + cache_files + local_outputs if p.is_file()]
    if out.exists():
        activity.append(out.stat().st_mtime)
    log_age, errors = None, []
    if log.exists():
        log_age = now - log.stat().st_mtime
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32768))
            tail = stream.read().decode(errors="replace")
        errors = [
            line
            for line in tail.splitlines()
            if any(
                word in line.lower() for word in ("traceback", "out of memory", "killed", "error:")
            )
        ][-8:]
    return dict(
        checked_at=now,
        pids=pids,
        stages=stages,
        network_tx_bytes=sum(
            int(p.read_text()) for p in Path("/sys/class/net").glob("*/statistics/tx_bytes")
        ),
        chunk_count=len(chunks),
        chunk_bytes=sum(p.stat().st_size for p in chunks),
        model_cache_bytes=sum(p.stat().st_size for p in cache_files if p.is_file()),
        model_cache_age_seconds=(
            now - max(p.stat().st_mtime for p in cache_files if p.is_file())
            if any(p.is_file() for p in cache_files)
            else None
        ),
        output_age_seconds=now - max(activity) if activity else None,
        log_age_seconds=log_age,
        error_lines=errors,
        artifacts=values,
    )


def probe_backend(handle_path: Path, out: Path) -> dict:
    """Reuse backend polling without automatic router failover; watchdog owns recovery."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.backends.base import RunHandle
    from backend_poll import _resolve_backend

    handle = RunHandle(**json.loads(handle_path.read_text()))
    observed = asdict(_resolve_backend(handle.backend).poll(handle))
    if observed["status"] not in {"running", "pending", "queued"}:
        return observed
    probe = shlex.join(
        ["python3", "-", "--remote-probe", "--out-dir", str(out), "--log-path", handle.log_path]
    )
    if handle.backend == "gcp":
        argv = [
            "gcloud",
            "compute",
            "ssh",
            handle.pod_name,
            "--configuration=eps-gcp",
            "--zone",
            handle.extra["zone"],
            "--quiet",
            "--command",
            "sudo -n " + probe,
        ]
    elif handle.backend == "runpod":
        from explore_persona_space.backends.runpod import _resolve_pod_endpoint

        host, port = _resolve_pod_endpoint(handle.pod_name)
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(port),
            f"root@{host}",
            probe,
        ]
    else:
        raise RuntimeError(f"pilot monitor requires GCP or RunPod: {handle.backend}")
    try:
        detail = json.loads(command(argv, stdin=Path(__file__).read_text(), timeout=60))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        # SSH is not ready immediately after instance creation. Bound this
        # allowance to the first ten minutes of the current handle's life.
        if time.time() - handle_path.stat().st_mtime < 600:
            observed.update(status="pending", startup_probe_error=type(exc).__name__)
            observed["startup_reachability_alarm"] = observed.pop("reachability_alarm", False)
            observed.pop("pid_alive", None)
            return observed
        raise
    observed["pilot"] = detail
    if detail["pids"]:
        observed["pid_alive"] = True
        observed["last_log_mtime_sec_ago"] = detail["log_age_seconds"]
        if observed["last_log_mtime_sec_ago"] is None:
            observed["last_log_mtime_sec_ago"] = 1801
        if detail["output_age_seconds"] is None or detail["output_age_seconds"] > 900:
            observed["stall_reason"] = (
                "no fresh staging/capture/analysis/publication progress for 900 seconds"
            )
        elif (
            not detail["chunk_count"]
            and detail["model_cache_age_seconds"] is not None
            and detail["model_cache_age_seconds"] < 900
            and observed["last_log_mtime_sec_ago"] > 900
        ):
            # Capture is pending while freshly measured model files grow.
            # Preserve actual log age rather than fabricate a log heartbeat.
            observed.update(status="pending", current_phase="model_staging")
    elif not detail["artifacts"].get("capture_complete.json"):
        # Backend startup/bootstrap is allowed before the first model worker.
        if observed.get("current_phase") in {"workload", "workload_running"}:
            observed["pid_alive"] = False
        elif time.time() - handle_path.stat().st_mtime < 900:
            observed["status"] = "pending"
            observed.pop("pid_alive", None)
        else:
            observed["stall_reason"] = "no pilot worker or capture completion after 15 minutes"
    return observed


def monitor(config: dict) -> None:
    """Record progress until verified publication; any monitor failure reaches watchdog."""
    state_path = Path(config["observation"])
    handle_path = Path(config["handle_file"])
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        state = {"source_sha": config["source_sha"], "checked_at": time.time()}
        if not handle_path.exists():
            if time.time() - config["created_at"] > 1800:
                raise RuntimeError("launch has not produced a handle within 30 minutes")
            state.update(status="awaiting_launch", backend_observation={"status": "pending"})
        else:
            observed = json.loads(
                command(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--probe-handle",
                        str(handle_path),
                        "--out-dir",
                        config["out_dir"],
                    ],
                    cwd=config["workdir"],
                    timeout=240,
                ).splitlines()[-1]
            )
            pilot = observed.get("pilot", {})
            old_pilot = previous.get("backend_observation", {}).get("pilot", {})
            if "publish" in pilot.get("stages", []) and "network_tx_bytes" in old_pilot:
                delta = pilot["network_tx_bytes"] - old_pilot["network_tx_bytes"]
                pilot["network_tx_delta_bytes"] = delta
                if delta >= 1024 * 1024:
                    observed.pop("stall_reason", None)
                    if observed.get("last_log_mtime_sec_ago", 0) > 900:
                        observed.update(status="pending", current_phase="artifact_upload")
            state.update(status="monitoring", backend_observation=observed, checked_at=time.time())
            if observed["status"] in {"dead", "done", "failed", "error", "gate"}:
                state["status"] = (
                    "publication_check_required"
                    if observed["status"] == "done"
                    else "backend_failed"
                )
                write_json(state_path, state)
                # Final verification is independently performed by the owning agent;
                # it publishes terminal.json only after exact-source HF/Git checks.
                terminal_path = state_path.parent / "terminal.json"
                if terminal_path.exists():
                    result = json.loads(terminal_path.read_text())
                    if (
                        result["source_sha"] != config["source_sha"]
                        or not result["verified_revision"]
                    ):
                        raise RuntimeError("terminal verification does not match source")
                    state.update(status="complete", results=result, checked_at=time.time())
                    write_json(state_path, state)
                    print("[monitor] exact-source publication verified", flush=True)
                    return
                if observed["status"] != "done":
                    raise RuntimeError(f"backend requires recovery: {observed['status']}")
                state["backend_observation"].pop("pid_alive", None)
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "checked_at": state["checked_at"],
                    "status": state["status"],
                    "backend": state["backend_observation"]["status"],
                }
            ),
            flush=True,
        )
        time.sleep(20 if time.time() - config["created_at"] < 180 else 60)
    raise TimeoutError("pilot monitoring reached its six-hour deadline")


def main() -> None:
    """Support the local monitor and bounded read-only probe subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--probe-handle", type=Path)
    parser.add_argument("--remote-probe", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    if args.remote_probe:
        print(json.dumps(remote_probe(args.out_dir, args.log_path)), flush=True)
    elif args.probe_handle:
        print(json.dumps(probe_backend(args.probe_handle, args.out_dir)), flush=True)
    else:
        config = json.loads(args.config.read_text())
        try:
            monitor(config)
        except Exception as exc:
            write_json(
                Path(config["observation"]),
                dict(
                    source_sha=config["source_sha"],
                    status="monitor_failed",
                    checked_at=time.time(),
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise


if __name__ == "__main__":
    main()
