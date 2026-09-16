"""Bounded local supervisor for the cached #1739 covariance ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.issue1739_covariance_ablation import LAYERS, sha256, write_json


def observe(config, status, backend, **extra):
    write_json(
        config["observation"],
        {
            "source_sha": config["source_sha"],
            "status": status,
            "checked_at": time.time(),
            "backend_observation": backend,
            **extra,
        },
    )


def run_phase(config, phase, command, max_seconds):
    state = Path(config["state_dir"])
    log = state / f"{phase}.log"
    started = time.time()
    last_progress = started
    last_cpu = 0.0
    last_evidence = None
    with log.open("a") as stream:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        write_json(
            state / "worker.json",
            {
                "phase": phase,
                "pid": child.pid,
                "started": started,
                "source_sha": config["source_sha"],
                "command": command,
            },
        )
        while child.poll() is None:
            now = time.time()
            try:
                proc = psutil.Process(child.pid)
            except psutil.NoSuchProcess:
                if child.poll() is not None:
                    break
                raise
            descendants = [proc, *proc.children(recursive=True)]
            cpu = 0.0
            rss = 0
            for process in descendants:
                try:
                    times = process.cpu_times()
                    cpu += times.user + times.system
                    rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            progress_paths = [
                state / "stage_progress.json",
                Path(config["outputs"]) / "progress.json",
            ]
            progress = {}
            for path in progress_paths:
                if path.is_file():
                    progress[path.name] = json.loads(path.read_text())

            # Ignore timestamp-only changes. Stage records actual transfer/member progress.
            def without_time(value):
                if isinstance(value, dict):
                    return {
                        k: without_time(v)
                        for k, v in value.items()
                        if k not in {"time", "timestamp", "checked_at", "elapsed_s", "updated_at"}
                    }
                return value

            evidence = json.dumps(without_time(progress), sort_keys=True)
            if cpu - last_cpu > 1.0 or evidence != last_evidence:
                last_progress = now
                last_cpu = cpu
                last_evidence = evidence
            age = now - last_progress
            backend = {
                "status": "running",
                "pid_alive": True,
                "pid": child.pid,
                "phase": phase,
                "cpu_seconds": cpu,
                "rss_bytes": rss,
                "last_log_mtime_sec_ago": now - log.stat().st_mtime,
                "seconds_without_measured_progress": age,
                "progress": progress,
            }
            reason = None
            if age > 900:
                reason = "no measured worker progress for 15 minutes"
            elif now - started > max_seconds:
                reason = f"phase exceeded {max_seconds}s budget"
            elif rss > 48 * 1024**3:
                reason = "worker RSS exceeded 48GiB limit"
            if reason:
                backend["stall_reason"] = reason
                observe(config, "backend_failed", backend)
                os.killpg(child.pid, 15)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, 9)
                    child.wait(timeout=20)
                raise RuntimeError(reason)
            observe(config, f"running_{phase}", backend)
            time.sleep(30)
        if child.returncode:
            observe(
                config,
                "backend_failed",
                {
                    "status": "failed",
                    "pid_alive": False,
                    "phase": phase,
                    "exit_code": child.returncode,
                },
            )
            raise RuntimeError(f"{phase} exited {child.returncode}; see {log}")
    observe(config, f"finished_{phase}", {"status": "done", "phase": phase})


def upload(config):
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv(Path(config["repo_root"]) / ".env")
    from huggingface_hub import HfApi
    from explore_persona_space.orchestrate.hub import assert_hub_dir_filecounts, retry_transient

    out = Path(config["outputs"])
    for b, layers in LAYERS.items():
        for layer in layers:
            directory = out / f"{b}_L{layer:02d}"
            done = json.loads((directory / "complete.json").read_text())
            assert done["source_sha"] == config["source_sha"]
            assert done["result_sha256"] == sha256(directory / "results.json")
            for name, digest in done["artifact_sha256"].items():
                assert sha256(directory / name) == digest, name
            for name in (
                "predictions_historical_grid.npz",
                "predictions_wide_grid.npz",
                "transforms.npz",
                "map_diagnostics.json",
            ):
                assert name in done["artifact_sha256"], name
    shutil.copyfile(Path(config["inputs"]) / "manifest.json", out / "input_manifest.json")
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for name in (
        "issue1739_covariance_ablation.py",
        "issue1739_covariance_stage.py",
        "issue1739_covariance_monitor.py",
        "issue1739_fits.py",
    ):
        shutil.copyfile(ROOT / "scripts" / name, source_dir / name)
    shutil.copyfile(ROOT / "covariance_plan.md", source_dir / "plan.md")
    write_json(
        out / "run_complete.json",
        {
            "source_sha": config["source_sha"],
            "time": time.time(),
            "cells": sum(map(len, LAYERS.values())),
        },
    )
    files = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in out.rglob("*")
        if p.is_file()
    }
    api = HfApi()
    repo = "superkaiba1/explore-persona-space-data"
    prefix = config["hf_prefix"]
    assert_hub_dir_filecounts(out, prefix)
    info = retry_transient(
        lambda: api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=out,
            path_in_repo=prefix,
            commit_message="Issue1739 matched covariance ablation",
        ),
        what="covariance output upload",
    )
    revision = info.oid
    tree = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: materialized listing is wrapped in retry_transient.
            api.list_repo_tree(repo, prefix, repo_type="dataset", revision=revision, recursive=True)
        ),
        what="covariance upload verification",
    )
    remote = {f.path: f for f in tree if hasattr(f, "size")}
    for rel, meta in files.items():
        got = remote[f"{prefix}/{rel}"]
        assert got.size == meta["bytes"], rel
        if got.lfs:
            assert got.lfs.sha256 == meta["sha256"], rel
        else:
            data = (out / rel).read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            assert got.blob_id == blob, rel
    # Persist compact local results before the verified-upload resume marker.
    local = Path(config["repo_root"]) / "eval_results/issue_1739/covariance_ablation_20260916"
    for p in out.rglob("*.json"):
        target = local / p.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write_json(
        Path(config["state_dir"]) / "upload_verified.json",
        {
            "source_sha": config["source_sha"],
            "verified_revision": revision,
            "prefix": prefix,
            "files": files,
            "verified_at": time.time(),
        },
    )
    print(f"Verified upload {revision}: {len(files)} files", flush=True)


def finalize(config, verified):
    """Run on normal completion and after a verified-upload restart."""
    task_cli = str(Path(config["repo_root"]) / "scripts/task.py")
    note = (
        "Matched covariance ablation: all 5 cells completed with historical parity and "
        "hash-verified archived outputs. Source "
        + config["source_sha"]
        + "; HF revision "
        + verified["verified_revision"]
        + "; prefix "
        + config["hf_prefix"]
    )
    subprocess.run(
        [
            sys.executable,
            task_cli,
            "post-marker",
            "1739",
            "epm:covariance-complete",
            "--note",
            note,
        ],
        cwd=config["repo_root"],
        check=True,
    )
    subprocess.run(
        [sys.executable, task_cli, "set-status", "1739", "awaiting_promotion", "--note", note],
        cwd=config["repo_root"],
        check=True,
    )
    observe(config, "complete", {"status": "done"}, results=verified)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if args.upload:
        upload(config)
        return
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert actual == config["source_sha"]
    verified_path = Path(config["state_dir"]) / "upload_verified.json"
    if verified_path.is_file():
        verified = json.loads(verified_path.read_text())
        assert verified["source_sha"] == config["source_sha"]
        finalize(config, verified)
        return
    inputs = Path(config["inputs"])
    inputs.mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]).mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(inputs).free > 12 * 1024**3, "insufficient RAM stage headroom"
    run_phase(
        config,
        "stage",
        [
            sys.executable,
            str(ROOT / "scripts/issue1739_covariance_stage.py"),
            "--dest",
            str(inputs),
            "--progress",
            str(Path(config["state_dir"]) / "stage_progress.json"),
        ],
        4 * 3600,
    )
    run_phase(
        config,
        "fit",
        [
            sys.executable,
            str(ROOT / "scripts/issue1739_covariance_ablation.py"),
            "--store-root",
            str(inputs),
            "--out",
            config["outputs"],
            "--repo",
            config["repo_root"],
        ],
        2 * 3600,
    )
    run_phase(config, "upload", [sys.executable, __file__, str(args.config), "--upload"], 3600)
    verified = json.loads((Path(config["state_dir"]) / "upload_verified.json").read_text())
    finalize(config, verified)


if __name__ == "__main__":
    main()
