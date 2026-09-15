"""Monitor the paired-vector workers and raw-response staging through completion."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from issue2054_k5_plain_run import stamp, write_json


def main():
    """Run bounded CPU analyses, check fresh progress, and validate all outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--rank", action="store_true")
    args = parser.parse_args()
    if sum([args.strict, args.affine, args.rank]) > 1:
        raise ValueError("choose only one of strict, affine or rank mode")
    args.out.mkdir(parents=True, exist_ok=True)
    monitor = args.out / "monitoring"
    monitor.mkdir(exist_ok=True)
    if (args.out / "run_complete.json").exists():
        raise RuntimeError("already completed; do not relaunch")
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
            raise RuntimeError(f"missing {key}={value}")
    repo = Path(__file__).resolve().parents[1]
    common = ["--out", str(args.out), "--inputs", str(args.inputs)]
    script = "issue2054_k5_matched_strict.py" if args.strict else "issue2054_k5_matched_offsets.py"
    if args.affine:
        script = "issue2054_k5_matched_affine.py"
    if args.rank:
        script = "issue2054_k5_matched_rank.py"
    fit = [sys.executable, "-u", str(repo / "scripts" / script)]
    commands = [
        (model, fit + common + ["--stage", "fit", "--model", model])
        for model in ("qwen2.5-7b", "qwen2.5-7b-instruct")
    ]
    if not args.strict and not args.affine and not args.rank:
        commands.append(
            (
                "responses",
                [sys.executable, "-u", str(repo / "scripts/issue2054_k5_matched_responses.py")]
                + common,
            )
        )
    started, jobs = stamp(), []
    for name, cmd in commands:
        log = monitor / f"{name}.txt"
        with log.open("a") as handle:
            process = subprocess.Popen(
                cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
        jobs.append((name, process, log))
    write_json(
        args.out / "runtime.json",
        {
            "started_at_utc": started,
            "monitor_pid": os.getpid(),
            "pins": pins,
            "workers": {name: p.pid for name, p, _ in jobs},
            "monitor_interval_seconds": 30,
            "disk_note": "Existing cached banks; affine/rank modes add less than 100 MB without downloads; raw staging modes estimate below 2 GB. No new environment or model weights.",
        },
    )
    while True:
        observations = []
        for name, process, log in jobs:
            rc = process.poll()
            state = subprocess.run(
                ["ps", "-p", str(process.pid), "-o", "pid=,stat=,pcpu=,rss=,etime="],
                capture_output=True,
                text=True,
                check=False,
            )
            if state.returncode not in (0, 1):
                raise RuntimeError(state.stderr)
            observations.append(
                {
                    "name": name,
                    "pid": process.pid,
                    "exit_code": rc,
                    "process": state.stdout.strip(),
                    "log_bytes": log.stat().st_size,
                    "log_mtime": log.stat().st_mtime,
                    "recent_log": log.read_text().splitlines()[-3:],
                }
            )
        snapshot = {
            "observed_at_utc": stamp(),
            "monitor_pid": os.getpid(),
            "workers": observations,
            "completed_pairs": len(list((args.out / "pairs").glob("*.json"))),
        }
        write_json(monitor / "latest.json", snapshot)
        with (monitor / "observations.jsonl").open("a") as handle:
            handle.write(json.dumps(snapshot) + "\n")
        failed = [o for o in observations if o["exit_code"] not in (None, 0)]
        if failed:
            write_json(args.out / "run_failed.json", {"at_utc": stamp(), "workers": failed})
            print(f"[phase=failure] {failed}", flush=True)
        print(json.dumps(snapshot), flush=True)
        if all(o["exit_code"] is not None for o in observations):
            break
        time.sleep(30)
    if failed:
        raise RuntimeError("worker failed; see run_failed.json")
    subprocess.run(fit + common + ["--stage", "collect"], check=True)
    report = json.loads((args.out / "results.json").read_text())
    if len(report["pairs"]) != (24 if args.rank else 120 if args.affine else 60):
        raise RuntimeError("incomplete vector or raw-response coverage")
    write_json(
        args.out / "run_complete.json",
        {
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": stamp(),
            "coverage": report["coverage"],
            "workers_exit_zero": True,
        },
    )
    print("[phase=done] paired analyses complete and verified", flush=True)


if __name__ == "__main__":
    main()
