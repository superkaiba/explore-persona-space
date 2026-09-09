#!/usr/bin/env python3
"""Queue approved K3 cells, one per allocated GPU, after the A100 pilot gate.

Uses the process-group and CVD queue pattern from issue1112_dispatch.py and
issue1333_dispatch.py. The launcher is stdlib-only; model phases remain in
issue2588_k3_job.py's validated runtime. Each child log is separate, so its
terminal phase line cannot prematurely finish the parent workload.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def approved_cells() -> tuple[str, ...]:
    """Read the existing registry without importing model runtimes into the parent."""
    tree = ast.parse((ROOT / "scripts/issue2588_k3_train_refit.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TARGET_CELLS" for target in node.targets
        ):
            cells = ast.literal_eval(node.value)
            if not isinstance(cells, tuple) or not all(isinstance(x, str) for x in cells):
                raise RuntimeError("invalid K3 target registry")
            return cells
    raise RuntimeError("K3 target registry missing")


def gpu_ids() -> list[str]:
    """Preserve an inherited allocation; enumerate only on an exclusive GCP host."""
    if "SLURM_JOB_ID" in os.environ:
        raise RuntimeError("this launcher is scoped to the approved GCP venue")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        ids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).splitlines()
    else:
        ids = inherited.split(",")
    ids = [x.strip() for x in ids]
    if not ids or any(not x or x == "-1" for x in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid GPU allocation: {ids}")
    return ids


def reap_group(proc: subprocess.Popen, grace_s: float = 30) -> None:
    """Reap only this launcher's child group, including orphaned engine workers."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            proc.wait()
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # The owned group exited between the membership probe and signal.
    proc.wait(timeout=10)


def persist_log(cell: str, log: Path) -> None:
    """Use the installed runtime's checked upload path after the child closes its log."""
    subprocess.run(
        [
            "/workspace/venvs/eps2588_k3/bin/python",
            str(Path(__file__).resolve()),
            "--upload-log",
            str(log),
            "--cell",
            cell,
        ],
        check=True,
    )


def run_queue(cells: list[str], ids: list[str], out_root: Path, *, poll_s: float = 5) -> None:
    """Run every selected cell once, scheduling the next onto each freed GPU."""
    pending = list(cells)
    running: dict[str, tuple[subprocess.Popen, str, Path]] = {}
    completed = []
    logs = out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    last_heartbeat = 0.0
    try:
        while pending or running:
            for gpu, (proc, cell, log) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                reap_group(proc)
                del running[gpu]
                print(f"[k3-fanout] cell={cell} gpu={gpu} rc={rc} log={log}", flush=True)
                if rc != 0:
                    tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
                    tail = tail.replace("[phase=", "[child-phase=")
                    raise RuntimeError(f"cell {cell} failed rc={rc}\n{tail}")
                persist_log(cell, log)
                completed.append(cell)
            for gpu in ids:
                if gpu in running or not pending:
                    continue
                cell = pending.pop(0)
                log = logs / f"{cell}.log"
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
                command = [
                    sys.executable,
                    str(ROOT / "scripts/issue2588_k3_job.py"),
                    "--cell",
                    cell,
                    "--out-root",
                    str(out_root),
                ]
                with log.open("a") as stream:
                    proc = subprocess.Popen(
                        command,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                running[gpu] = (proc, cell, log)
                print(
                    f"[k3-fanout] start cell={cell} gpu={gpu} pid={proc.pid} log={log}", flush=True
                )
            if time.monotonic() - last_heartbeat >= 60:
                state = {
                    "pending": pending,
                    "running": {gpu: cell for gpu, (_, cell, _) in running.items()},
                    "completed": completed,
                }
                print(f"[phase=k3_fanout] {json.dumps(state)}", flush=True)
                last_heartbeat = time.monotonic()
            if running:
                time.sleep(poll_s)
    finally:
        for proc, _cell, _log in running.values():
            reap_group(proc)
    if set(completed) != set(cells) or len(completed) != len(cells):
        raise RuntimeError("fanout coverage differs from requested cells")
    print(f"[phase=done] K3 fanout completed {len(completed)} cells", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--cells", nargs="+")
    action.add_argument("--upload-log", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/issue_2588/k3_train_refit")
    args = parser.parse_args()
    if args.upload_log:
        if args.cell not in approved_cells():
            parser.error("upload cell must belong to the approved K3 panel")
        import issue2588_k3_train_refit as k3

        prefix = f"{k3.PC.PANEL_PREFIX}/{k3.ROUND_LABEL}/logs/{args.cell}"
        k3.RC._upload_file(args.upload_log, f"{prefix}/fanout.log", "K3 closed cell log")
        return
    if len(args.cells) != len(set(args.cells)) or not set(args.cells).issubset(approved_cells()):
        parser.error("cells must be unique members of the approved K3 panel")

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    os.chdir(ROOT)
    run_queue(args.cells, gpu_ids(), args.out_root)


if __name__ == "__main__":
    main()
