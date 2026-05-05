#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #228 — N-way GPU-shard coordinator for the full path-A sweep.

Three phases:

  * **Phase 0** (marker LoRA training, ~70 states) — invokes
    ``train_marker_loras_228.py`` once per (source, ckpt) state where
    ``ckpt > 0``. Idempotent on existing HF Hub adapters.
  * **Phase 0.5** (leakage measurement, 71 states) — invokes
    ``measure_leakage_228.py`` once per (source, ckpt) state including
    the 7 epoch-0 baselines. Idempotent on existing
    ``marker_eval.json`` files. After Phase 0.5 completes, the entire
    ``causal_proximity/strong_convergence/`` directory is uploaded as a
    WandB Artifact named ``causal_proximity_strong_convergence_v1`` so
    future analyses don't have to regenerate it.
  * **Phase 1** (JS sweep, 71 states) — invokes
    ``compute_js_convergence_228.py`` once per state including the
    ``base / checkpoint-0`` shared epoch-0 baseline. Idempotent on
    existing ``result.json`` files.

Workers are subprocess invocations that see ``CUDA_VISIBLE_DEVICES``
narrowed to one logical GPU. The coordinator is signal-handled:
SIGTERM / SIGINT waits for in-flight workers to finish their state,
then exits cleanly.

Invocation::

    nohup uv run python scripts/run_issue228_sweep.py \\
        --num-gpus 8 \\
        --phase all \\
        --output-dir eval_results/issue_228 \\
        --leakage-output-dir eval_results/causal_proximity/strong_convergence \\
        --seed 42 \\
        > /workspace/issue228_sweep.log 2>&1 &

Single-phase reruns (debugging)::

    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 0
    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 0.5
    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 1
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

# ── Phase identifiers (string-typed so ``--phase 0.5`` parses naturally) ──

PHASE_MARKER = "0"  # train marker LoRAs (ckpt > 0)
PHASE_LEAKAGE = "0.5"  # measure leakage at 11 targets (all 71 states)
PHASE_JS = "1"  # JS divergence + cosine (all 71 states)
PHASE_ALL = "all"
ALL_PHASES = (PHASE_MARKER, PHASE_LEAKAGE, PHASE_JS)


# ── State enumeration per phase ────────────────────────────────────────────


def _enumerate_phase0_states() -> list[tuple[str, int]]:
    """70 (source, ckpt) pairs where ckpt > 0; alphabetical, ascending step."""
    states: list[tuple[str, int]] = []
    for source in sorted(ADAPTER_MAP.keys()):
        for step in CHECKPOINT_STEPS:
            states.append((source, step))
    return states


def _enumerate_phase05_states() -> list[tuple[str, int]]:
    """71 (source, ckpt) pairs: 7 epoch-0 + 70 epoch-N."""
    states: list[tuple[str, int]] = []
    for source in sorted(ADAPTER_MAP.keys()):
        states.append((source, 0))
        for step in CHECKPOINT_STEPS:
            states.append((source, step))
    return states


def _enumerate_phase1_states() -> list[tuple[str, int]]:
    """71 states for JS sweep: shared epoch-0 baseline + 7×10."""
    states: list[tuple[str, int]] = [(BASE_SOURCE, BASE_CHECKPOINT_STEP)]
    for source in sorted(ADAPTER_MAP.keys()):
        for step in CHECKPOINT_STEPS:
            states.append((source, step))
    return states


# ── Per-state "done?" predicates ──────────────────────────────────────────


def _phase0_state_done(_output_dir: Path, _source: str, _step: int) -> bool:
    """Phase 0 idempotency lives on HF Hub, not on disk.

    We always invoke the worker and let it short-circuit with
    ``ALREADY_EXISTS`` after a single ``HfApi.list_repo_files`` call. That
    avoids local-state drift (a freshly-resumed pod has no on-disk hint).
    """
    return False


def _phase05_state_done(output_dir: Path, source: str, step: int) -> bool:
    return (output_dir / source / f"checkpoint-{step}" / "marker_eval.json").exists()


def _phase1_state_done(output_dir: Path, source: str, step: int) -> bool:
    if source == BASE_SOURCE and step == BASE_CHECKPOINT_STEP:
        return (output_dir / BASE_SOURCE / "checkpoint-0" / "result.json").exists()
    return (output_dir / source / f"checkpoint-{step}" / "result.json").exists()


# ── Per-phase command builders ────────────────────────────────────────────


def _phase0_cmd(*, source: str, step: int, seed: int, **_) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_marker_loras_228.py"),
        "--source",
        source,
        "--checkpoint-step",
        str(step),
        "--gpu-id",
        "0",
        "--seed",
        str(seed),
    ]


def _phase05_cmd(
    *,
    source: str,
    step: int,
    seed: int,
    output_dir: Path,
    gpu_mem_util: float,
    **_,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "measure_leakage_228.py"),
        "--source",
        source,
        "--checkpoint-step",
        str(step),
        "--gpu-id",
        "0",
        "--output-dir",
        str(output_dir),
        "--gpu-mem-util",
        str(gpu_mem_util),
        "--seed",
        str(seed),
    ]


def _phase1_cmd(
    *,
    source: str,
    step: int,
    seed: int,
    output_dir: Path,
    gpu_mem_util: float,
    **_,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compute_js_convergence_228.py"),
        "--source",
        source,
        "--checkpoint-step",
        str(step),
        "--gpu-id",
        "0",
        "--output-dir",
        str(output_dir),
        "--gpu-mem-util",
        str(gpu_mem_util),
        "--seed",
        str(seed),
    ]


PHASE_REGISTRY: dict[str, dict] = {
    PHASE_MARKER: {
        "name": "phase0_marker_loras",
        "enumerate": _enumerate_phase0_states,
        "is_done": _phase0_state_done,
        "build_cmd": _phase0_cmd,
    },
    PHASE_LEAKAGE: {
        "name": "phase05_leakage",
        "enumerate": _enumerate_phase05_states,
        "is_done": _phase05_state_done,
        "build_cmd": _phase05_cmd,
    },
    PHASE_JS: {
        "name": "phase1_js",
        "enumerate": _enumerate_phase1_states,
        "is_done": _phase1_state_done,
        "build_cmd": _phase1_cmd,
    },
}


# ── Coordinator plumbing ───────────────────────────────────────────────────


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


def _run_one_state(
    *,
    gpu_id: int,
    cmd: list[str],
    log_path: Path,
    source: str,
    step: int,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONHASHSEED"] = "42"  # round-3 hot-fix: stabilize hash() across workers (#228)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    return proc.returncode


def _worker_thread(
    *,
    gpu_id: int,
    queue: list[tuple[str, int]],
    queue_lock: threading.Lock,
    is_done,
    build_cmd,
    output_dir: Path,
    seed: int,
    gpu_mem_util: float,
    log_dir: Path,
    shutdown: _ShutdownFlag,
    results: list[dict],
    results_lock: threading.Lock,
) -> None:
    while True:
        if shutdown.is_set():
            logger.info("[gpu=%d] shutdown flag set; exiting worker", gpu_id)
            return
        with queue_lock:
            if not queue:
                return
            source, step = queue.pop(0)

        if is_done(output_dir, source, step):
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

        cmd = build_cmd(
            source=source,
            step=step,
            seed=seed,
            output_dir=output_dir,
            gpu_mem_util=gpu_mem_util,
        )
        log_path = log_dir / f"{source}_ckpt{step}.log"
        rc = _run_one_state(
            gpu_id=gpu_id,
            cmd=cmd,
            log_path=log_path,
            source=source,
            step=step,
        )
        with results_lock:
            results.append(
                {
                    "gpu_id": gpu_id,
                    "source": source,
                    "checkpoint_step": step,
                    "returncode": rc,
                    "skipped_existing": False,
                    "log_path": str(log_path),
                }
            )


def _run_phase(
    phase_key: str,
    *,
    num_gpus: int,
    output_dir: Path,
    seed: int,
    gpu_mem_util: float,
    log_root: Path,
    shutdown: _ShutdownFlag,
    dry_run: bool = False,
) -> dict:
    """Drive one phase to completion (or shutdown). Returns a summary dict."""
    if phase_key not in PHASE_REGISTRY:
        raise ValueError(f"Unknown phase {phase_key!r}")
    phase = PHASE_REGISTRY[phase_key]

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / phase["name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    states = phase["enumerate"]()
    logger.info("Phase %s: %d states across %d GPUs", phase_key, len(states), num_gpus)

    if dry_run:
        logger.info(
            "[dry-run] would dispatch %d states; first cmd: %s",
            len(states),
            phase["build_cmd"](
                source=states[0][0],
                step=states[0][1],
                seed=seed,
                output_dir=output_dir,
                gpu_mem_util=gpu_mem_util,
            ),
        )
        return {
            "phase": phase_key,
            "n_states": len(states),
            "dry_run": True,
            "states": states,
        }

    queue = list(states)
    queue_lock = threading.Lock()
    results: list[dict] = []
    results_lock = threading.Lock()

    threads: list[threading.Thread] = []
    for gpu_id in range(num_gpus):
        t = threading.Thread(
            target=_worker_thread,
            kwargs={
                "gpu_id": gpu_id,
                "queue": queue,
                "queue_lock": queue_lock,
                "is_done": phase["is_done"],
                "build_cmd": phase["build_cmd"],
                "output_dir": output_dir,
                "seed": seed,
                "gpu_mem_util": gpu_mem_util,
                "log_dir": log_dir,
                "shutdown": shutdown,
                "results": results,
                "results_lock": results_lock,
            },
            name=f"{phase['name']}-gpu-{gpu_id}",
            daemon=False,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failures = [r for r in results if r["returncode"] != 0]
    skipped = [r for r in results if r.get("skipped_existing")]
    completed = [r for r in results if r["returncode"] == 0 and not r.get("skipped_existing")]
    logger.info(
        "Phase %s done: %d completed, %d skipped, %d failed",
        phase_key,
        len(completed),
        len(skipped),
        len(failures),
    )

    return {
        "phase": phase_key,
        "name": phase["name"],
        "n_states": len(states),
        "n_completed": len(completed),
        "n_skipped_existing": len(skipped),
        "n_failed": len(failures),
        "results": results,
        "shutdown_signal_received": shutdown.is_set(),
    }


# ── WandB upload of leakage cache after Phase 0.5 ─────────────────────────


def _upload_leakage_cache_to_wandb(
    leakage_dir: Path,
    artifact_name: str = "causal_proximity_strong_convergence_v1",
    project: str = "issue228",
) -> str | None:
    """Upload the entire ``causal_proximity/strong_convergence/`` dir to WandB.

    Returns the artifact's full path, or None on failure (logged + returned
    so the coordinator can surface it without aborting later phases).
    """
    if not leakage_dir.exists():
        logger.warning("Leakage dir %s does not exist; nothing to upload", leakage_dir)
        return None
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed; cannot upload leakage cache artifact")
        return None
    try:
        run = wandb.init(
            project=project,
            name=f"upload_{artifact_name}",
            job_type="artifact-upload",
            reinit=True,
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="eval-results",
            description=(
                "Issue #228 Phase 0.5: leakage measurement at 11 targets for "
                "71 (source, checkpoint) states. Same protocol as #109's "
                "eval_causal_ckpt.py (vLLM, temp=1.0, top_p=1.0, n=10, seed=42)."
            ),
        )
        artifact.add_dir(str(leakage_dir))
        run.log_artifact(artifact)
        run.finish()
        full_name = f"{run.entity}/{run.project}/{artifact_name}:latest"
        logger.info("Uploaded leakage cache as %s", full_name)
        return full_name
    except Exception as exc:
        logger.error("WandB artifact upload failed: %s", exc, exc_info=True)
        return None


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of physical GPUs to shard across.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=PHASE_ALL,
        choices=[*ALL_PHASES, PHASE_ALL],
        help=(
            "Which phase(s) to run. '0' = marker LoRA training, "
            "'0.5' = leakage measurement, '1' = JS sweep, 'all' = all three."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228",
        help="Output dir for Phase-1 JS results.",
    )
    parser.add_argument(
        "--leakage-output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "causal_proximity" / "strong_convergence",
        help="Output dir for Phase-0.5 leakage results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-mem-util", type=float, default=VLLM_GPU_MEM_UTIL_DEFAULT)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228" / "_worker_logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work plan (per-phase state count + first command) and exit.",
    )
    parser.add_argument(
        "--skip-wandb-upload",
        action="store_true",
        help="Skip uploading leakage cache to WandB after Phase 0.5.",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        raise SystemExit(f"--num-gpus must be >=1, got {args.num_gpus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.leakage_output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    phases_to_run = list(ALL_PHASES) if args.phase == PHASE_ALL else [args.phase]

    shutdown = _ShutdownFlag()
    _install_signal_handlers(shutdown)

    summaries: list[dict] = []
    artifact_full_name: str | None = None
    for phase_key in phases_to_run:
        if shutdown.is_set():
            logger.warning("Shutdown set; not starting phase %s", phase_key)
            break
        # Phase 0 has no per-state output dir of its own (HF Hub is the sink),
        # but the coordinator still writes per-state logs under ``log_dir``.
        if phase_key == PHASE_MARKER:
            phase_output_dir = args.output_dir  # unused by worker
        elif phase_key == PHASE_LEAKAGE:
            phase_output_dir = args.leakage_output_dir
        else:
            phase_output_dir = args.output_dir

        summary = _run_phase(
            phase_key,
            num_gpus=args.num_gpus,
            output_dir=phase_output_dir,
            seed=args.seed,
            gpu_mem_util=args.gpu_mem_util,
            log_root=args.log_dir,
            shutdown=shutdown,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

        # After Phase 0.5, persist the leakage cache to WandB so future
        # analyses can pull the same data without re-running.
        if (
            phase_key == PHASE_LEAKAGE
            and not args.dry_run
            and not args.skip_wandb_upload
            and summary.get("n_failed", 0) == 0
            and not shutdown.is_set()
        ):
            artifact_full_name = _upload_leakage_cache_to_wandb(args.leakage_output_dir)

    # Aggregate summary write-out.
    total_failures = sum(s.get("n_failed", 0) for s in summaries)
    summary_payload = {
        "num_gpus": args.num_gpus,
        "phases_requested": args.phase,
        "phases_run": phases_to_run,
        "summaries": summaries,
        "wandb_artifact": artifact_full_name,
        "shutdown_signal_received": shutdown.is_set(),
        "dry_run": args.dry_run,
    }
    summary_path = args.output_dir / "_sweep_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    logger.info("Wrote %s", summary_path)

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
