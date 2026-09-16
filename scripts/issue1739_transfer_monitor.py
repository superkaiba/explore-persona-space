"""Supervise the fixed-direction experiment and verify durable output delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(Path("/home/thomasjiralerspong/explore-persona-space/.env"))

import psutil
from scripts.issue1739_covariance_stage import sha256, write_json
from scripts.issue1739_covariance_monitor import observe


def verify_source(config):
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != config["source_sha"]:
        raise ValueError("Registered source SHA does not match the worktree")
    if set(config["source_file_sha256"]) != set(config["source_files"]):
        raise ValueError("Registered source file coverage mismatch")
    for rel, digest in config["source_file_sha256"].items():
        if sha256(ROOT / rel) != digest:
            raise ValueError(f"Registered source file changed: {rel}")


def evidence(value):
    if isinstance(value, dict):
        return {
            k: evidence(v)
            for k, v in value.items()
            if k
            not in {
                "time",
                "timestamp",
                "timestamp_utc",
                "unix_time",
                "checked_at",
                "elapsed_s",
                "updated_at",
            }
        }
    return value


def run_phase(config, phase):
    verify_source(config)
    state = Path(config["state_dir"])
    log = state / (phase["name"] + ".log")
    started = time.time()
    last_progress, last_cpu, previous = started, 0.0, None
    with log.open("a") as stream:
        worker = subprocess.Popen(
            phase["command"],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(
            state / "worker.json",
            {
                "phase": phase["name"],
                "pid": worker.pid,
                "started": started,
                "source_sha": config["source_sha"],
                "command": phase["command"],
            },
        )
        try:
            while worker.poll() is None:
                now = time.time()
                try:
                    process = psutil.Process(worker.pid)
                    processes = [process, *process.children(recursive=True)]
                except psutil.NoSuchProcess:
                    if worker.poll() is not None:
                        break
                    raise
                cpu, rss = 0.0, 0
                for process in processes:
                    try:
                        times = process.cpu_times()
                        cpu += times.user + times.system
                        rss += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                progress = {}
                for source in config["progress_paths"]:
                    path = Path(source)
                    if path.exists():
                        progress[str(path)] = json.loads(path.read_text())
                measured = json.dumps(evidence(progress), sort_keys=True)
                if cpu - last_cpu > 1.0 or measured != previous:
                    last_progress, last_cpu, previous = now, cpu, measured
                available = psutil.virtual_memory().available
                shm_free = shutil.disk_usage(config["ram_root"]).free
                backend = {
                    "status": "running",
                    "pid_alive": True,
                    "pid": worker.pid,
                    "phase": phase["name"],
                    "cpu_seconds": cpu,
                    "rss_bytes": rss,
                    "mem_available_bytes": available,
                    "shm_free_bytes": shm_free,
                    "last_log_mtime_sec_ago": now - log.stat().st_mtime,
                    "seconds_without_measured_progress": now - last_progress,
                    "progress": progress,
                }
                reason = None
                if now - last_progress > 900:
                    reason = "No measured CPU/input/output progress for 900 seconds"
                elif now - started > phase["max_seconds"]:
                    reason = "Phase exceeded its declared wall-time limit"
                elif available < 8 * 1024**3:
                    reason = "VM MemAvailable fell below 8 GiB (includes tmpfs pressure)"
                elif shm_free < 2 * 1024**3:
                    reason = "RAM staging filesystem has less than 2 GiB free"
                elif rss > 55 * 1024**3:
                    reason = "Worker process tree RSS exceeded 55 GiB"
                if reason:
                    backend["stall_reason"] = reason
                    observe(config, "backend_failed", backend)
                    raise RuntimeError(reason)
                observe(config, "running_" + phase["name"], backend)
                time.sleep(30)
            if worker.returncode:
                raise RuntimeError(f"{phase['name']} exited {worker.returncode}; inspect {log}")
        except BaseException:
            if worker.poll() is None:
                os.killpg(worker.pid, 15)
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, 9)
                    worker.wait(timeout=20)
            raise
    observe(config, "finished_" + phase["name"], {"status": "done", "phase": phase["name"]})


def upload(config):
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.hub import assert_hub_dir_filecounts, retry_transient

    verify_source(config)
    out = Path(config["outputs"])
    inputs = Path(config["inputs"])
    for source in config["required_outputs"]:
        path = out / source
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing required completed output {path}")
    for behavior in ("evil", "sycophancy", "hallucination"):
        directory = out / "analysis" / behavior
        done = json.loads((directory / "complete.json").read_text())
        if done["source_sha"] != config["source_sha"]:
            raise ValueError("Analysis completion source SHA mismatch")
        for name, digest in done["artifact_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Analysis completion hash mismatch: {behavior}/{name}")
    manifest = json.loads((out / "map/map_manifest.json").read_text())
    if not manifest["complete"] or manifest["n_train"] != 963444:
        raise ValueError("Map control manifest is incomplete")
    for record in manifest["artifacts_sha256"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Map artifact changed after its completion manifest")
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for rel in config["source_files"]:
        destination = source_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, destination)
    for rel in config["input_records"]:
        destination = out / "input_provenance" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / rel, destination)
    shutil.copyfile(Path(config["state_dir"]) / "config.json", out / "run_config.json")
    write_json(
        out / "run_complete.json",
        {
            "source_sha": config["source_sha"],
            "completed_at": time.time(),
            "required_outputs": config["required_outputs"],
        },
    )
    files = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    }
    assert_hub_dir_filecounts(out, config["hf_prefix"])
    api = HfApi()
    repo = "superkaiba1/explore-persona-space-data"
    info = retry_transient(
        lambda: api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=out,
            path_in_repo=config["hf_prefix"],
            commit_message="Issue1739 matched fixed-direction transfer and shuffled-pair controls",
        ),
        what="fixed transfer upload",
    )
    tree = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: full lazy listing materialized inside retry_transient.
            api.list_repo_tree(
                repo, config["hf_prefix"], repo_type="dataset", revision=info.oid, recursive=True
            )
        ),
        what="verify fixed transfer upload",
    )
    remote = {r.path: r for r in tree if hasattr(r, "size")}
    expected_paths = {config["hf_prefix"] + "/" + r for r in files}
    if set(remote) != expected_paths:
        raise ValueError("Uploaded file set differs from final output manifest")
    for rel, record in files.items():
        got = remote[config["hf_prefix"] + "/" + rel]
        if got.size != record["bytes"]:
            raise ValueError(f"Remote size mismatch: {rel}")
        if got.lfs:
            matches = got.lfs.sha256 == record["sha256"]
        else:
            data = (out / rel).read_bytes()
            matches = got.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if not matches:
            raise ValueError(f"Remote hash mismatch: {rel}")
    verified = {
        "source_sha": config["source_sha"],
        "verified_revision": info.oid,
        "prefix": config["hf_prefix"],
        "files": files,
        "verified_at": time.time(),
    }
    write_json(Path(config["state_dir"]) / "upload_verified.json", verified)
    local = Path(config["repo_root"]) / "eval_results/issue_1739/fixed_transfer_20260916"
    for p in out.rglob("*.json"):
        target = local / p.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    print(f"Verified fixed transfer upload {info.oid}: {len(files)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_source(config)
    if args.upload:
        upload(config)
        return
    try:
        ram_inventory = Path(config["ram_root"]) / "inventory"
        ram_inventory.mkdir(parents=True, exist_ok=True)
        for name in ("hf_source_inventory.json", "passb_prompts_sha_verified.jsonl"):
            preserved = Path(config["state_dir"]) / "inventory" / name
            dest = ram_inventory / name
            if dest.exists():
                if sha256(dest) != sha256(preserved):
                    raise ValueError(f"RAM source inventory disagrees with durable copy: {name}")
            else:
                shutil.copyfile(preserved, dest)
        for phase in config["phases"]:
            run_phase(config, phase)
        run_phase(
            config,
            {
                "name": "upload",
                "max_seconds": 3600,
                "command": [sys.executable, __file__, str(args.config), "--upload"],
            },
        )
        verified = json.loads((Path(config["state_dir"]) / "upload_verified.json").read_text())
        if verified["source_sha"] != config["source_sha"]:
            raise ValueError("Completion source mismatch")
        observe(config, "complete", {"status": "done"}, results=verified)
    except BaseException as exc:
        observe(config, "backend_failed", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
