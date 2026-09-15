"""Run the banked #825 calibration with a detached, timestamped process monitor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_status(path, row):
    """Publish one complete monitor observation atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(path)


def main():
    """Supervise sequential child fits; stop the grid on the first failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "full"))
    parser.add_argument("--model", choices=("instruct", "pretrained"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    args = parser.parse_args()
    args.monitor_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.monitor_dir / "status.json"
    started = time.time()
    state = {
        "supervisor_pid": os.getpid(),
        "launch_epoch": started,
        "phase": args.phase,
        "completed": [],
    }
    common = [
        "--inputs",
        str(args.inputs),
        "--out",
        str(args.out),
        "--store",
        str(args.store),
        "--reference",
        str(args.reference),
    ]
    driver = Path(__file__).with_name("issue825_turn_bias_scale.py")
    jobs = (
        [("pilot", "instruct", 1)]
        if args.phase == "pilot"
        else [
            ("fit", model, source)
            for model in ((args.model,) if args.model else ("instruct", "pretrained"))
            for source in (1, 3, 12)
        ]
        + ([] if args.model else [("reduce", "instruct", 1)])
    )
    for phase, model, source in jobs:
        name = f"{phase}_{model}_source{source}"
        log_path = args.monitor_dir / f"{name}.log"
        command = [
            sys.executable,
            str(driver),
            phase,
            *common,
            "--model",
            model,
            "--source-turn",
            str(source),
        ]
        with log_path.open("w") as output:
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT
            )
            state.update(job=name, child_pid=child.pid, log=str(log_path), status="running")
            while True:
                rc = child.poll()
                now = time.time()
                proc_status = Path(f"/proc/{child.pid}/status")
                try:
                    observed = proc_status.read_text()
                except (FileNotFoundError, ProcessLookupError):
                    # The child can exit between poll() and the /proc read.
                    observed = "exited"
                stat = log_path.stat()
                lines = log_path.read_text().splitlines()
                state.update(
                    observed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                    heartbeat_epoch=now,
                    elapsed_seconds=now - started,
                    process_state=[
                        line
                        for line in observed.splitlines()
                        if line.startswith(("State:", "VmRSS:", "Threads:"))
                    ],
                    log_bytes=stat.st_size,
                    log_mtime=stat.st_mtime,
                    log_tail=lines[-4:],
                    fold_checkpoints=len(list((args.out / "folds").glob("*.json"))),
                    map_checkpoints=len(list((args.out / "maps").glob("*.json"))),
                    error_lines=[
                        line
                        for line in lines
                        if any(
                            token in line.lower()
                            for token in ("traceback", "error:", "killed", "out of memory")
                        )
                    ][-4:],
                )
                if rc is not None:
                    state.update(exit_code=rc, status="failed" if rc else "between_jobs")
                write_status(status_path, state)
                print(json.dumps(state), flush=True)
                if rc is not None:
                    if rc:
                        return rc
                    state["completed"].append(name)
                    break
                time.sleep(15 if now - started < 120 else 60)
    state.update(
        status="complete",
        exit_code=0,
        finished_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_status(status_path, state)
    print(json.dumps(state), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
