"""Supervise the assistant-story GCP lane and verify its remote completion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_assistant_story_k5/production_v1"
REMOTE_ROOT = "/workspace/issue2054_assistant_story/production_v1"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def bounded(command, *, cwd, timeout=180):
    """Terminate only this probe's process group if its transport hangs."""
    child = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The probe group already exited.
        try:
            child.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # Escalate the same owned group after the grace period.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Includes successful TERM cleanup.
        child.wait(timeout=5)
        raise
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
    return stdout


def progress_script():
    """Read child logs and actual checkpoints; exclude the parent's heartbeat."""
    return f"""
import json, os, pathlib, time
root = pathlib.Path({REMOTE_ROOT!r})
runtime = root / 'runtime.json'
result = {{'observed_at': time.time(), 'runtime_exists': runtime.exists()}}
if runtime.exists():
    run = json.loads(runtime.read_text())
    pid = int(run['pid'])
    proc = pathlib.Path('/proc') / str(pid)
    cmd = (proc / 'cmdline').read_bytes().replace(bytes([0]), b' ').decode() if proc.exists() else ''
    result.update(pid_alive='issue2054_k5_assistant_story.py' in cmd,
                  source_sha=run['source_sha'], started=run['started'])
files = []
for directory in ('raw', 'captures', 'capture_audits', 'k5', 'analysis', 'logs'):
    folder = root / directory
    if folder.exists():
        for path in folder.rglob('*'):
            if path.is_file():
                stat = path.stat()
                files.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime))
result.update(files=len(files), bytes=sum(f[1] for f in files),
              latest_outputs=sorted(files, key=lambda f:f[2], reverse=True)[:6],
              latest_progress=max((f[2] for f in files), default=None))
workers = []
for path in (root / 'logs').glob('*.status.json'):
    worker = json.loads(path.read_text())
    if 'finished' in worker or worker['started'] < result.get('started', 0):
        continue
    log = root / worker['log']
    model = worker['model']
    directory = 'raw' if worker['stage'] == 'generate' else 'captures'
    latest = [entry[2] for entry in files
              if entry[0].startswith(directory + '/') and ('__' + model + '/') in entry[0]]
    if log.exists():
        latest.append(log.stat().st_mtime)
    worker.update(pid_exists=(pathlib.Path('/proc') / str(worker['pid'])).exists(),
                  latest_progress=max(latest, default=worker['started']))
    workers.append(worker)
result['active_workers'] = workers
print(json.dumps(result))
"""


def probe(args):
    sys.path.insert(0, str(args.backend_repo / "src"))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from explore_persona_space.backends.base import RunHandle
    from explore_persona_space.backends.gcp import GcpBackend

    handle = RunHandle(**json.loads(args.handle_file.read_text()))
    if handle.backend != "gcp":
        raise ValueError("This monitor is scoped to the declared GCP lane")
    observation = asdict(GcpBackend().poll(handle))
    if observation["status"] == "running":
        command = [
            "gcloud",
            "--configuration=eps-gcp",
            "compute",
            "ssh",
            handle.pod_name,
            "--project=eps-persona-gpu-jun2026",
            f"--zone={handle.extra['zone']}",
            "--quiet",
            "--command",
            "sudo -n python3 -c " + shlex.quote(progress_script()),
        ]
        observation["outputs"] = json.loads(
            bounded(command, cwd=args.backend_repo, timeout=90).splitlines()[-1]
        )
    print(json.dumps(observation), flush=True)


def verified_json(api, revision, name, expected_sha=None):
    """Verify content against the producer's remote hash receipt."""
    from huggingface_hub import hf_hub_download

    path = f"{PREFIX}/{name}"
    receipt_path = path + ".done.json"
    found = api.get_paths_info(
        DATA_REPO, [path, receipt_path], repo_type="dataset", revision=revision
    )
    if len(found) != 2:
        return None
    local = Path(hf_hub_download(DATA_REPO, path, repo_type="dataset", revision=revision))
    receipt = json.loads(
        Path(
            hf_hub_download(
                DATA_REPO,
                receipt_path,
                repo_type="dataset",
                revision=revision,
            )
        ).read_text()
    )
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    if receipt["path"] != path or receipt["sha256"] != digest:
        raise RuntimeError(f"Remote completion receipt mismatch: {name}")
    if expected_sha is not None and digest != expected_sha:
        raise RuntimeError(f"Completion inventory hash mismatch: {name}")
    return json.loads(local.read_text())


def verify_remote_files(api, revision, records):
    """Check all output content hashes at one immutable Hub revision."""
    from huggingface_hub import hf_hub_download

    expected = {}
    for record in records:
        path = record["path"]
        if not path.startswith(PREFIX + "/") or ".." in Path(path).parts:
            raise RuntimeError("Completion inventory escaped the declared output prefix")
        expected[path] = record
    if not expected or len(expected) != len(records):
        raise RuntimeError("Empty or duplicate completion inventory")
    entries = []
    names = list(expected)
    for start in range(0, len(names), 100):
        entries.extend(
            api.get_paths_info(
                DATA_REPO, names[start : start + 100], repo_type="dataset", revision=revision
            )
        )
    if {e.path for e in entries} != set(expected):
        raise RuntimeError("Declared output is absent at the completion revision")

    def check(entry):
        record = expected[entry.path]
        if entry.size != record["size"]:
            raise RuntimeError(f"Output size mismatch: {entry.path}")
        digest = getattr(getattr(entry, "lfs", None), "sha256", None)
        if digest is None:
            path = Path(
                hf_hub_download(DATA_REPO, entry.path, repo_type="dataset", revision=revision)
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(f"Output hash mismatch: {entry.path}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, entries))
    return len(entries)


def completion(source_sha, launched_at):
    from huggingface_hub import HfApi

    api = HfApi()
    revision = api.repo_info(DATA_REPO, repo_type="dataset").sha
    result = verified_json(api, revision, "job_complete.json")
    if result is None:
        return None
    if result.get("finished", 0) < launched_at:
        return None
    if result["source_sha"] != source_sha:
        raise RuntimeError("Remote completion names a different source commit")
    if result["status"] != "complete" or result.get("analysis_requested") is not True:
        raise RuntimeError("Remote job did not complete the requested analysis")
    generated = verified_json(api, revision, "generation_complete.json")
    analyzed = verified_json(api, revision, "analysis/analysis_complete.json")
    if generated is None or analyzed is None:
        raise RuntimeError("Job completion lacks verified generation/analysis artifacts")
    for value in (generated, analyzed):
        if value["source_sha"] != source_sha or value["status"] != "complete":
            raise RuntimeError("Generation/analysis completion provenance mismatch")
    raw_inventory = verified_json(
        api, revision, "output_inventory.json", generated["inventory_sha256"]
    )
    inventory = verified_json(
        api, revision, analyzed["inventory_path"], analyzed["inventory_sha256"]
    )
    summary = verified_json(api, revision, analyzed["results_path"], analyzed["results_sha256"])
    if raw_inventory is None or inventory is None or summary is None:
        raise RuntimeError("Completion inventory or results are missing")
    if inventory["source_sha"] != source_sha or summary["status"] != "complete":
        raise RuntimeError("Analysis inventory/results status or source mismatch")
    analysis_records = [
        dict(path=f"{PREFIX}/{r['path']}", sha256=r["sha256"], size=r["size"])
        for r in inventory["files"]
    ]
    verified_count = verify_remote_files(api, revision, raw_inventory + analysis_records)
    return dict(
        result, verified_revision=revision, analysis=analyzed, verified_files=verified_count
    )


def assess_progress(
    backend, source_sha, now, *, launched_at=None, startup_seconds=1800, stall_seconds=1200
):
    """Turn stale child evidence into an alarm even if the parent is printing."""
    outputs = backend.get("outputs", {})
    if not outputs.get("runtime_exists") and launched_at is not None:
        if now - launched_at > startup_seconds:
            backend["stall_reason"] = "bootstrap exceeded budget before runtime appeared"
    if outputs.get("runtime_exists"):
        if outputs["source_sha"] != source_sha:
            raise RuntimeError("Live runtime is from a different source commit")
        backend["pid_alive"] = outputs["pid_alive"]
        if not outputs["pid_alive"]:
            backend["stall_reason"] = "driver process exited without verified completion"
        latest = outputs.get("latest_progress")
        if latest is not None and now - max(latest, outputs["started"]) > stall_seconds:
            backend["stall_reason"] = "child logs and output checkpoints stopped advancing"
        elif latest is None and now - outputs["started"] > startup_seconds:
            backend["stall_reason"] = "startup exceeded budget without a first child output"
        for worker in outputs.get("active_workers", []):
            age = now - max(worker["started"], worker["latest_progress"])
            if age > stall_seconds:
                backend["stall_reason"] = (
                    f"{worker['model']} {worker['stage']} worker stopped advancing"
                )
            elif not worker["pid_exists"] and age > 30:
                backend["stall_reason"] = (
                    f"{worker['model']} {worker['stage']} worker exited without a final status"
                )
    return backend


def monitor(args):
    state = {"source_sha": args.source_sha, "status": "starting", "started_at": args.launched_at}
    failures = 0
    try:
        while True:
            result = completion(args.source_sha, args.launched_at)
            if result:
                state.update(status="complete", checked_at=time.time(), results=result)
                atomic_json(args.observation, state)
                print(
                    json.dumps({"status": "complete", "revision": result["verified_revision"]}),
                    flush=True,
                )
                return
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--probe",
                "--handle-file",
                str(args.handle_file),
                "--backend-repo",
                str(args.backend_repo),
            ]
            try:
                backend = json.loads(bounded(command, cwd=args.backend_repo).splitlines()[-1])
                if backend.get("current_phase") in {
                    "describe_failed",
                    "describe_bad_json",
                    "guest_attr_probe_failed",
                }:
                    raise subprocess.SubprocessError(
                        f"Transient backend probe: {backend['current_phase']}"
                    )
                backend = assess_progress(
                    backend, args.source_sha, time.time(), launched_at=args.launched_at
                )
                failures = 0
            except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
                failures += 1
                state.update(
                    status="probe_retry",
                    consecutive_probe_errors=failures,
                    probe_error=str(exc),
                    probe_error_at=time.time(),
                )
                # Retain the previous measured observation and its original time;
                # a failed probe never refreshes positive evidence. The independent
                # watchdog still catches stale evidence or a dead monitor.
                if "backend_observation" not in state or failures >= 3:
                    state["backend_observation"] = {"status": "unknown"}
                atomic_json(args.observation, state)
                if failures >= 3:
                    raise
                time.sleep(60)
                continue
            state.update(status="monitoring", checked_at=time.time(), backend_observation=backend)
            atomic_json(args.observation, state)
            if (
                backend["status"] not in ("running", "pending", "queued")
                or backend.get("stall_reason")
                or backend.get("reachability_alarm")
                or backend.get("pid_alive") is False
            ):
                # The completion packet can become visible after the first HF read
                # but before the same-tick process/backend probe. Resolve that race
                # without treating a successful exit alone as proof of completion.
                result = completion(args.source_sha, args.launched_at)
                if result:
                    state.update(status="complete", checked_at=time.time(), results=result)
                    atomic_json(args.observation, state)
                    return
                raise RuntimeError(f"Backend needs diagnosis: {backend}")
            print(
                json.dumps(
                    {
                        "checked_at": state["checked_at"],
                        "phase": backend["current_phase"],
                        "outputs": backend.get("outputs"),
                    }
                ),
                flush=True,
            )
            time.sleep(60)
    except BaseException as exc:
        state.update(status="failed", checked_at=time.time(), monitor_error=str(exc))
        atomic_json(args.observation, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle-file", type=Path, required=True)
    parser.add_argument("--backend-repo", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--launched-at", type=float)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        probe(args)
    else:
        if not args.source_sha or not args.observation or args.launched_at is None:
            parser.error("Monitoring requires source-sha, observation, and launched-at")
        sys.path.insert(0, str(args.backend_repo / "src"))
        from explore_persona_space.orchestrate.env import load_dotenv

        load_dotenv()
        monitor(args)


if __name__ == "__main__":
    main()
