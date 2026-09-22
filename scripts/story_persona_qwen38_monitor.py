"""Observe task 2673 with fresh backend, process, log, and artifact evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

PROGRESS_STALL = "no fresh staging/capture/analysis/publication progress for 900 seconds"


def uv_cache_progress(root: Path, *, max_entries=100000, budget_seconds=5) -> dict:
    """Bound the cache walk and measure allocated file bytes, never log heartbeats."""
    started = time.monotonic()
    stack = [root]
    allocated, files, entries = 0, 0, 0
    complete = True
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as scan:
                for entry in scan:
                    entries += 1
                    if entries > max_entries or time.monotonic() - started > budget_seconds:
                        return {"allocated_bytes": allocated, "files": files, "complete": False}
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            allocated += entry.stat(follow_symlinks=False).st_blocks * 512
                            files += 1
                    except FileNotFoundError:
                        # UV can atomically rename an extraction directory mid-scan.
                        complete = False
        except FileNotFoundError:
            if directory != root:
                complete = False
    return {"allocated_bytes": allocated, "files": files, "complete": complete}


def loading_counters(tail: str) -> dict:
    """Read actual model-loading counters from tqdm, including carriage-return logs."""
    counters = {}
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", tail)
    for line in clean.splitlines():
        match = re.search(
            r"(Loading checkpoint shards|Loading weights):[^\r\n]*?\b([\d,]+)/([\d,]+)\b",
            line,
        )
        if match:
            name, completed, total = match.groups()
            completed, total = int(completed.replace(",", "")), int(total.replace(",", ""))
            if 0 <= completed <= total and total > 0:
                counters[name] = {"completed": completed, "total": total}
    return counters


def apply_loading_progress(observed: dict, previous: dict, *, now: float) -> None:
    """Use advancing real counters to resolve only generic progress/log staleness."""
    pilot = observed.get("pilot", {})
    if not pilot.get("pids"):
        return
    old = previous.get("pilot", {})
    current_uv, old_uv = pilot.get("uv_cache", {}), old.get("uv_cache", {})
    uv_grew = (
        current_uv.get("complete") is True
        and old_uv.get("complete") is True
        and current_uv["allocated_bytes"] > old_uv["allocated_bytes"]
    )
    loading_advanced = any(
        name in old.get("loading_counters", {})
        and counter["total"] == old["loading_counters"][name]["total"]
        and counter["completed"] > old["loading_counters"][name]["completed"]
        for name, counter in pilot.get("loading_counters", {}).items()
    )
    recent = []
    for key, advanced, phase in (
        ("last_uv_growth_at", uv_grew, "runtime_bootstrap"),
        ("last_loading_progress_at", loading_advanced, "model_loading"),
    ):
        timestamp = now if advanced else old.get(key)
        if timestamp is not None:
            pilot[key] = timestamp
            if 0 <= now - timestamp < 900:
                recent.append((timestamp, phase))
    if (
        recent
        and observed.get("stall_reason") in {None, PROGRESS_STALL}
        and not observed.get("reachability_alarm")
        and observed.get("status") in {"running", "pending", "stalled"}
        and (observed.get("status") != "stalled" or observed.get("log_only_backend_stall"))
    ):
        observed.pop("stall_reason", None)
        observed.update(status="pending", current_phase=max(recent)[1])


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
                    b"/story_persona_crossmodel_capture.py",
                    b"/story_persona_crossmodel_analysis.py",
                    b"/story_persona_crossmodel_artifacts.py",
                    b"/story_persona_crossmodel_workload.sh",
                )
            )
            for arg in argv
        ):
            pids.append(int(proc.name))
            if b"--phase" in argv:
                position = argv.index(b"--phase") + 1
                stages.append(argv[position].decode())
            stages.extend(
                arg.split(b"=", 1)[1].decode() for arg in argv if arg.startswith(b"phase=")
            )
    values = {}
    for name in ("progress.json", "smoke.json", "capture_complete.json", "analysis_complete.json"):
        path = out / name
        if path.exists():
            values[name] = json.loads(path.read_text())
    chunks = list((out / "chunks").glob("batch_*.pt"))
    cache_root = Path("/workspace/.cache/huggingface/hub")
    cache_files = [
        p
        for model_dir in (
            "models--Qwen--Qwen3.8-27B",
            "models--deepseek-ai--DeepSeek-V3.1-Base",
            "models--moonshotai--Kimi-K2.6",
        )
        for p in (cache_root / model_dir / "blobs").glob("*")
    ]
    local_outputs = list(out.iterdir()) if out.exists() else []
    activity = [p.stat().st_mtime for p in chunks + cache_files + local_outputs if p.is_file()]
    if out.exists():
        activity.append(out.stat().st_mtime)
    log_age, errors, counters = None, [], {}
    if log.exists():
        log_age = now - log.stat().st_mtime
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32768))
            tail = stream.read().decode(errors="replace")
        counters = loading_counters(tail)
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
        uv_cache=uv_cache_progress(Path("/workspace/.cache/uv")),
        loading_counters=counters,
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


def provision_handoff_pending(extra: dict, pod: dict | None, *, now: float) -> bool:
    """Bound a not-yet-executed handoff by immutable provider allocation time."""
    if (
        extra.get("workload_executed") is not False
        or extra.get("workload_start_error")
        or not pod
        or pod["id"] != extra["pod_id"]
        or pod["desiredStatus"] != "RUNNING"
    ):
        return False
    started = datetime.fromisoformat(pod["createdAt"].replace("Z", "+00:00"))
    if started.tzinfo is None:
        raise ValueError("provider allocation time must include a timezone")
    return 0 <= now - started.timestamp() < 900


def reconcile_stale_pid(observed: dict, detail: dict, *, launch_pending: bool) -> None:
    """Resolve a stale launcher PID using fresh task-specific process evidence."""
    if observed["status"] != "pid-stale-workload-live":
        return
    observed["original_backend_status"] = observed["status"]
    if detail["pids"]:
        observed.update(status="running", pid_alive=True)
    elif launch_pending:
        observed.update(status="pending", current_phase="provision_only_handoff")
        observed.pop("pid_alive", None)
    else:
        observed.update(status="stalled", pid_alive=False)
        return
    if (observed.get("stall_reason") or "").startswith("pid_dead_evidence:"):
        observed["original_pid_diagnostic"] = observed.pop("stall_reason")


def probe_backend(handle_path: Path, out: Path) -> dict:
    """Reuse backend polling without automatic router failover; watchdog owns recovery."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.backends.base import RunHandle
    from backend_poll import _resolve_backend

    handle = RunHandle(**json.loads(handle_path.read_text()))
    observed = asdict(_resolve_backend(handle.backend).poll(handle))
    if observed["status"] not in {
        "running",
        "pending",
        "queued",
        "stalled",
        "pid-stale-workload-live",
    }:
        return observed
    observed["log_only_backend_stall"] = observed["status"] == "stalled" and not observed.get(
        "stall_reason"
    )
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
        if (
            observed["status"] != "pid-stale-workload-live"
            and time.time() - handle_path.stat().st_mtime < 600
        ):
            observed.update(status="pending", startup_probe_error=type(exc).__name__)
            observed["startup_reachability_alarm"] = observed.pop("reachability_alarm", False)
            observed.pop("pid_alive", None)
            return observed
        raise
    observed["pilot"] = detail
    launch_pending = False
    if (
        observed["status"] == "pid-stale-workload-live"
        and not detail["pids"]
        and handle.backend == "runpod"
        and handle.extra.get("workload_executed") is False
        and not handle.extra.get("workload_start_error")
    ):
        from runpod_api import graphql

        data = graphql(
            "query($id: String!) { pod(input: {podId: $id}) { id createdAt desiredStatus } }",
            {"id": handle.extra["pod_id"]},
            personal=True,
            timeout=45,
        )
        launch_pending = provision_handoff_pending(handle.extra, data["pod"], now=time.time())
    reconcile_stale_pid(
        observed,
        detail,
        launch_pending=launch_pending,
    )
    if detail["pids"]:
        observed["pid_alive"] = True
        observed["last_log_mtime_sec_ago"] = detail["log_age_seconds"]
        if observed["last_log_mtime_sec_ago"] is None:
            observed["last_log_mtime_sec_ago"] = 1801
        if detail["output_age_seconds"] is None or detail["output_age_seconds"] > 900:
            if not observed.get("stall_reason"):
                observed["stall_reason"] = PROGRESS_STALL
        elif (
            not detail["chunk_count"]
            and detail["model_cache_age_seconds"] is not None
            and detail["model_cache_age_seconds"] < 900
            and observed["last_log_mtime_sec_ago"] > 900
            and not observed.get("stall_reason")
        ):
            # Capture is pending while freshly measured model files grow.
            # Preserve actual log age rather than fabricate a log heartbeat.
            observed.update(status="pending", current_phase="model_staging")
    elif not detail["artifacts"].get("capture_complete.json"):
        # Backend startup/bootstrap is allowed before the first model worker.
        if (
            observed.get("original_backend_status") == "pid-stale-workload-live"
            and not launch_pending
        ):
            observed.update(status="stalled", pid_alive=False)
        elif observed.get("current_phase") in {"workload", "workload_running"}:
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
            old_backend = previous.get("backend_observation", {})
            old_pilot = old_backend.get("pilot", {})
            apply_loading_progress(observed, old_backend, now=time.time())
            if "publish" in pilot.get("stages", []) and "network_tx_bytes" in old_pilot:
                delta = pilot["network_tx_bytes"] - old_pilot["network_tx_bytes"]
                pilot["network_tx_delta_bytes"] = delta
                if delta >= 1024 * 1024 and observed.get("stall_reason") in {None, PROGRESS_STALL}:
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
