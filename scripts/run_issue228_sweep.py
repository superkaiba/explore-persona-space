#!/usr/bin/env python3
"""Issue #228 — N-way GPU-shard coordinator for the 71-state JS sweep.

Round-robin assigns the 71 (source, checkpoint) states to ``--num-gpus``
worker subprocesses, one process per GPU, each running
``compute_js_convergence_228.py`` for its assigned states sequentially.

The 71 states are:
  * 1 shared epoch-0 baseline:  source=base, checkpoint_step=0
  * 70 strong-convergence:      7 sources x 10 checkpoint-steps (200..2000)

Workers see ``CUDA_VISIBLE_DEVICES`` narrowed to one logical GPU. They write
``eval_results/issue_228/<source>/checkpoint-<step>/result.json`` per state
(idempotent — skip if it already exists).

The coordinator is signal-handled: SIGTERM / SIGINT waits for all in-flight
workers to finish their current state, then exits cleanly without launching
any new states.

Invocation::

    nohup uv run python scripts/run_issue228_sweep.py \\
        --num-gpus 8 \\
        --output-dir eval_results/issue_228 \\
        --seed 42 \\
        > /workspace/issue228_sweep.log 2>&1 &

To run on a single H100, pass ``--num-gpus 1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project-side imports must come AFTER bootstrap()
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    BASE_CHECKPOINT_STEP,
    BASE_SOURCE,
    CHECKPOINT_STEPS,
    VLLM_GPU_MEM_UTIL_DEFAULT,
)

logger = logging.getLogger("run_issue228_sweep")


def _enumerate_states() -> list[tuple[str, int]]:
    """Return the 71 (source, checkpoint_step) states in a stable order.

    Order: base (epoch-0 shared baseline) first, then sources in alphabetical
    order, each with checkpoints in ascending step order. This lets the
    earliest-listed states finish first under round-robin sharding, surfacing
    failures (sanity 6.1#4 absolute regression) early.
    """
    states: list[tuple[str, int]] = [(BASE_SOURCE, BASE_CHECKPOINT_STEP)]
    for source in sorted(ADAPTER_MAP.keys()):
        for step in CHECKPOINT_STEPS:
            states.append((source, step))
    return states


def _state_done(output_dir: Path, source: str, step: int) -> bool:
    if source == BASE_SOURCE and step == BASE_CHECKPOINT_STEP:
        path = output_dir / BASE_SOURCE / "checkpoint-0" / "result.json"
    else:
        path = output_dir / source / f"checkpoint-{step}" / "result.json"
    return path.exists()


class WorkerShutdown(Exception):
    """Raised inside the main loop when a SIGTERM/SIGINT was received."""


class _ShutdownFlag:
    def __init__(self) -> None:
        self._flag = False
        self._lock = threading.Lock()

    def set(self) -> None:
        with self._lock:
            self._flag = True

    def is_set(self) -> bool:
        with self._lock:
            return self._flag


def _install_signal_handlers(flag: _ShutdownFlag) -> None:
    def handler(signum, _frame):
        logger.warning("Received signal %d — finishing in-flight states then exiting", signum)
        flag.set()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _build_worker_cmd(
    *,
    source: str,
    step: int,
    output_dir: Path,
    seed: int,
    gpu_mem_util: float,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compute_js_convergence_228.py"),
        "--source",
        source,
        "--checkpoint-step",
        str(step),
        "--gpu-id",
        "0",  # parent narrows CUDA_VISIBLE_DEVICES to one GPU
        "--output-dir",
        str(output_dir),
        "--gpu-mem-util",
        str(gpu_mem_util),
        "--seed",
        str(seed),
    ]


def _run_one_state(
    *,
    gpu_id: int,
    source: str,
    step: int,
    output_dir: Path,
    seed: int,
    gpu_mem_util: float,
    log_dir: Path,
) -> tuple[str, int, int, str]:
    """Run one state on a specific physical GPU. Returns (source, step, rc, log_path)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{source}_ckpt{step}.log"

    cmd = _build_worker_cmd(
        source=source,
        step=step,
        output_dir=output_dir,
        seed=seed,
        gpu_mem_util=gpu_mem_util,
    )
    logger.info("[gpu=%d] launching %s ckpt-%d -> %s", gpu_id, source, step, log_path)
    t0 = time.time()
    with open(log_path, "w") as log_f:
        log_f.write(f"# {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error(
            "[gpu=%d] %s ckpt-%d FAILED (rc=%d, %.1fs); see %s",
            gpu_id,
            source,
            step,
            proc.returncode,
            elapsed,
            log_path,
        )
    else:
        logger.info("[gpu=%d] %s ckpt-%d OK (%.1fs)", gpu_id, source, step, elapsed)
    return (source, step, proc.returncode, str(log_path))


def _worker_thread(
    *,
    gpu_id: int,
    queue: list[tuple[str, int]],
    queue_lock: threading.Lock,
    output_dir: Path,
    seed: int,
    gpu_mem_util: float,
    log_dir: Path,
    shutdown: _ShutdownFlag,
    results: list[dict],
    results_lock: threading.Lock,
) -> None:
    """Pull states off the shared queue and run them on this GPU."""
    while True:
        if shutdown.is_set():
            logger.info("[gpu=%d] shutdown flag set; exiting worker thread", gpu_id)
            return
        with queue_lock:
            if not queue:
                return
            source, step = queue.pop(0)

        if _state_done(output_dir, source, step):
            logger.info("[gpu=%d] %s ckpt-%d already complete; skipping", gpu_id, source, step)
            with results_lock:
                results.append(
                    {
                        "gpu_id": gpu_id,
                        "source": source,
                        "checkpoint_step": step,
                        "returncode": 0,
                        "skipped_existing": True,
                    }
                )
            continue

        source_done, step_done, rc, log_path = _run_one_state(
            gpu_id=gpu_id,
            source=source,
            step=step,
            output_dir=output_dir,
            seed=seed,
            gpu_mem_util=gpu_mem_util,
            log_dir=log_dir,
        )
        with results_lock:
            results.append(
                {
                    "gpu_id": gpu_id,
                    "source": source_done,
                    "checkpoint_step": step_done,
                    "returncode": rc,
                    "skipped_existing": False,
                    "log_path": log_path,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of physical GPUs to shard across (1, 2, 4, 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="vLLM seed (matches plan §8 inference card).",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=VLLM_GPU_MEM_UTIL_DEFAULT,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "_worker_logs",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        raise SystemExit(f"--num-gpus must be >=1, got {args.num_gpus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    states = _enumerate_states()
    expected_total = 1 + len(ADAPTER_MAP) * len(CHECKPOINT_STEPS)
    if len(states) != expected_total:
        raise RuntimeError(
            f"State enumeration produced {len(states)} states, expected {expected_total}"
        )
    logger.info("Sweep: %d states across %d GPUs", len(states), args.num_gpus)

    queue = list(states)
    queue_lock = threading.Lock()
    results: list[dict] = []
    results_lock = threading.Lock()
    shutdown = _ShutdownFlag()
    _install_signal_handlers(shutdown)

    threads: list[threading.Thread] = []
    for gpu_id in range(args.num_gpus):
        t = threading.Thread(
            target=_worker_thread,
            kwargs={
                "gpu_id": gpu_id,
                "queue": queue,
                "queue_lock": queue_lock,
                "output_dir": args.output_dir,
                "seed": args.seed,
                "gpu_mem_util": args.gpu_mem_util,
                "log_dir": args.log_dir,
                "shutdown": shutdown,
                "results": results,
                "results_lock": results_lock,
            },
            name=f"gpu-{gpu_id}",
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Summary
    failures = [r for r in results if r["returncode"] != 0]
    skipped = [r for r in results if r.get("skipped_existing")]
    completed = [r for r in results if r["returncode"] == 0 and not r.get("skipped_existing")]
    logger.info(
        "Sweep done: %d completed, %d skipped (already existed), %d failed",
        len(completed),
        len(skipped),
        len(failures),
    )
    if failures:
        logger.error("Failed states (rc != 0):")
        for r in failures:
            logger.error(
                "  %s ckpt-%s rc=%d log=%s",
                r["source"],
                r["checkpoint_step"],
                r["returncode"],
                r.get("log_path", "?"),
            )

    summary_path = args.output_dir / "_sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_gpus": args.num_gpus,
                "n_states": len(states),
                "n_completed": len(completed),
                "n_skipped_existing": len(skipped),
                "n_failed": len(failures),
                "results": results,
                "shutdown_signal_received": shutdown.is_set(),
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", summary_path)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
