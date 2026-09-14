"""Run the two plain-assistant source fits with timestamped process monitoring."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import UTC, datetime


def stamp():
    """Return the current UTC observation time."""
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    """Atomically replace a small monitor or completion record."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    """Supervise workers through exit, record progress, and verify final coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    monitoring = args.out / "monitoring"
    monitoring.mkdir(exist_ok=True)
    if (args.out / "run_complete.json").exists():
        raise RuntimeError("use the completed run rather than relaunching it")
    started = stamp()
    pins = {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_MMAP_THRESHOLD_": "131072",
        "NUMPY_MADVISE_HUGEPAGE": "0",
    }
    for key, value in pins.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"missing launch pin {key}={value}")
    write_json(
        args.out / "runtime.json",
        {
            "started_at_utc": started,
            "monitor_pid": os.getpid(),
            "python": sys.version,
            "env": pins,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "torch", "scipy", "matplotlib", "huggingface-hub"]
            },
            "monitor_interval_seconds": 30,
        },
    )
    command = [
        sys.executable,
        "-u",
        str(repo / "scripts/issue2054_k5_assistant_transfer.py"),
        "--source-mode",
        "plain_only",
        "--out",
        str(args.out),
        "--inputs",
        str(args.inputs),
    ]
    jobs = []
    for model in ["qwen2.5-7b", "qwen2.5-7b-instruct"]:
        path = monitoring / f"{model}.txt"
        with path.open("a") as handle:
            proc = subprocess.Popen(
                command + ["--stage", "fit", "--model", model],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        jobs.append((model, proc, path))
    print(
        f"[phase=monitor] started={started} monitor_pid={os.getpid()} worker_pids={[p.pid for _, p, _ in jobs]}",
        flush=True,
    )
    while True:
        observations = []
        for model, proc, path in jobs:
            rc = proc.poll()
            state = subprocess.run(
                ["ps", "-p", str(proc.pid), "-o", "pid=,stat=,pcpu=,rss=,etime="],
                capture_output=True,
                text=True,
                check=False,
            )
            if state.returncode not in [0, 1]:
                raise RuntimeError(state.stderr)
            lines = path.read_text().splitlines()
            observations.append(
                {
                    "model": model,
                    "pid": proc.pid,
                    "exit_code": rc,
                    "process": state.stdout.strip(),
                    "log_bytes": path.stat().st_size,
                    "log_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                    "recent_log": lines[-3:],
                }
            )
        snapshot = {
            "observed_at_utc": stamp(),
            "monitor_pid": os.getpid(),
            "workers": observations,
            "completed_fold_files": len(list((args.out / "folds").glob("*.json"))),
            "source_map_files": len(list((args.out / "maps").glob("*.npz"))),
        }
        write_json(monitoring / "latest.json", snapshot)
        with (monitoring / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot) + "\n")
        print(json.dumps(snapshot), flush=True)
        failed_now = [o for o in observations if o["exit_code"] not in (None, 0)]
        if failed_now:
            write_json(args.out / "run_failed.json", {"at_utc": stamp(), "workers": failed_now})
            print(f"[phase=failure] worker exited nonzero: {failed_now}", flush=True)
        if all(o["exit_code"] is not None for o in observations):
            break
        time.sleep(30)
    failed = [o for o in observations if o["exit_code"] != 0]
    if failed:
        write_json(args.out / "run_failed.json", {"at_utc": stamp(), "workers": failed})
        raise RuntimeError(f"fit worker failed: {failed}")
    subprocess.run(command + ["--stage", "collect"], check=True)
    results = json.loads((args.out / "results.json").read_text())
    panels = results["panels"]
    if (
        len(panels) != 10
        or sum(len(p["folds"]) for p in panels) != 50
        or len(results["maps"]) != 10
    ):
        raise RuntimeError(
            "realized coverage differs from the requested 10 panels / 50 folds / 10 maps"
        )
    write_json(
        args.out / "run_complete.json",
        {
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": stamp(),
            "source_maps": 10,
            "target_panels": 10,
            "fold_evaluations": 50,
            "worker_exit_codes": {m: p.returncode for m, p, _ in jobs},
            "all_map_and_coefficient_hashes_validated_by_collector": True,
        },
    )
    print("[phase=done] plain-source transfer complete and verified", flush=True)


if __name__ == "__main__":
    main()
