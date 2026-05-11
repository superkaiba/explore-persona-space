#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #228 — N-way GPU-shard coordinator for the full path-A sweep.

Four phases (round 4):

  * **Phase 0a** (on-policy cache pre-generation, 7 sources) — invokes
    ``pregenerate_onpolicy_cache_228.py`` once per source, **serially on
    one GPU**. Each invocation is a fresh subprocess so the OS reclaims
    the CUDA allocator between sources. This eliminates the same-process
    PEFT-merge + vLLM contention that crashed 5 of 8 workers in round 3
    (see ``feedback_peft_merge_vllm_same_process.md``).
  * **Phase 0** (marker LoRA training, ~70 states) — invokes
    ``train_marker_loras_228.py`` once per (source, ckpt) state where
    ``ckpt > 0``. Reads the on-policy cache populated by Phase 0a;
    fails loudly if missing. Idempotent on existing HF Hub adapters.
  * **Phase 0.5** (leakage measurement, 77 states) — invokes
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
then exits cleanly. Phase 0a is dispatched serially (one GPU at a time)
even when ``--num-gpus`` > 1; the remaining phases fan out across all
GPUs.

Invocation::

    nohup uv run python scripts/run_issue228_sweep.py \\
        --num-gpus 8 \\
        --phase all \\
        --output-dir eval_results/issue_228 \\
        --leakage-output-dir eval_results/causal_proximity/strong_convergence \\
        --seed 42 \\
        > /workspace/issue228_sweep.log 2>&1 &

Single-phase reruns (debugging)::

    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 0a
    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 0
    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 0.5
    uv run python scripts/run_issue228_sweep.py --num-gpus 1 --phase 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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

PHASE_PREGEN = "0a"  # pre-generate on-policy cache, 7 sources, SERIAL one-GPU
PHASE_MARKER = "0"  # train marker LoRAs (ckpt > 0)
PHASE_LEAKAGE = "0.5"  # measure leakage at 11 targets (all 77 states)
PHASE_JS = "1"  # JS divergence + cosine (all 71 states)
PHASE_ALL = "all"
ALL_PHASES = (PHASE_PREGEN, PHASE_MARKER, PHASE_LEAKAGE, PHASE_JS)


# ── Disk hygiene (R6 fix #4) ──────────────────────────────────────────────


# Default workspace mount point on the RunPod images we use. Override via
# ``EPM_WORKSPACE_PATH`` if running on a host with a different layout.
DEFAULT_WORKSPACE_PATH = Path(os.environ.get("EPM_WORKSPACE_PATH", "/workspace"))

# Pre-launch threshold: if `/workspace` has less than this many GB free we
# refuse to start a sweep — the R5 launch died with `OSError: [Errno 28] No
# space left on device` after Phase 0 filled the disk with ~70 merged-dir
# leaks. 50 GB is enough headroom for one Phase 0 worker (~30 GB merged +
# ~50 MB adapter) plus normal HF cache churn, with a safety factor.
PRELAUNCH_DISK_FREE_GB = 50

# In-flight watchdog thresholds (used by `_disk_monitor_thread`).
DISK_WARN_GB = 50
DISK_ABORT_GB = 20

# Per-phase scratch dirs we know about, in case the workers crash before
# their own ``finally`` blocks clean them. Best-effort sweep performed
# between phases by `_clean_orphaned_phase_scratch`.
TMP_PHASE0_DIR = Path("/workspace/tmp/issue228_markerlora")
TMP_PHASE05_DIR = Path("/workspace/tmp/issue228_leakage")
TMP_PHASE1_DIR = Path("/workspace/tmp/issue228")


def _disk_free_gb(path: Path = DEFAULT_WORKSPACE_PATH) -> float | None:
    """Return free GB on ``path``, or None if the path doesn't exist.

    We use ``shutil.disk_usage`` (POSIX statvfs under the hood). Works on
    pods even when called from the coordinator thread.
    """
    if not path.exists():
        return None
    try:
        usage = shutil.disk_usage(str(path))
    except OSError as exc:
        logger.warning("disk_usage(%s) failed: %s", path, exc)
        return None
    return usage.free / (1024**3)


def _check_prelaunch_disk(min_free_gb: float = PRELAUNCH_DISK_FREE_GB) -> None:
    """Abort the sweep if free disk on ``/workspace`` is below threshold.

    Raises ``SystemExit`` (rc=2) with an actionable error message. We
    distinguish "path missing" (warn but proceed — running on local VM
    for tests) from "path exists, low space" (abort).
    """
    free_gb = _disk_free_gb(DEFAULT_WORKSPACE_PATH)
    if free_gb is None:
        logger.warning(
            "Workspace path %s does not exist; skipping pre-launch disk check (running off-pod?)",
            DEFAULT_WORKSPACE_PATH,
        )
        return
    if free_gb < min_free_gb:
        msg = (
            f"Pre-launch disk check FAILED: {DEFAULT_WORKSPACE_PATH} has "
            f"{free_gb:.1f} GB free, need >= {min_free_gb} GB. "
            f"Free space first (e.g. `python scripts/pod.py cleanup --all`) "
            f"or move scratch dirs off /workspace."
        )
        logger.error(msg)
        raise SystemExit(msg)
    logger.info("Pre-launch disk OK: %.1f GB free on %s", free_gb, DEFAULT_WORKSPACE_PATH)


def _disk_monitor_thread(
    shutdown: _ShutdownFlag,  # type: ignore[name-defined]
    *,
    interval_s: float = 60.0,
    warn_gb: float = DISK_WARN_GB,
    abort_gb: float = DISK_ABORT_GB,
) -> None:
    """Background thread: poll free disk, set shutdown flag on critical drop.

    Logs a WARNING when free space drops below ``warn_gb`` and triggers a
    clean shutdown (matching SIGTERM semantics: in-flight workers finish,
    queue drains, exit) when below ``abort_gb``. Exits cleanly when the
    shutdown flag is already set (no more work to monitor).
    """
    abort_logged = False
    warn_logged = False
    while not shutdown.is_set():
        free_gb = _disk_free_gb(DEFAULT_WORKSPACE_PATH)
        if free_gb is not None:
            if free_gb < abort_gb and not abort_logged:
                logger.error(
                    "Disk monitor: %.1f GB free on %s (< %.0f GB) — "
                    "triggering clean shutdown to prevent ENOSPC.",
                    free_gb,
                    DEFAULT_WORKSPACE_PATH,
                    abort_gb,
                )
                shutdown.set()
                abort_logged = True
                return
            if free_gb < warn_gb and not warn_logged:
                logger.warning(
                    "Disk monitor: %.1f GB free on %s (< %.0f GB) — watch for ENOSPC.",
                    free_gb,
                    DEFAULT_WORKSPACE_PATH,
                    warn_gb,
                )
                warn_logged = True
        # Sleep in short chunks so we react to shutdown promptly.
        for _ in range(int(interval_s)):
            if shutdown.is_set():
                return
            time.sleep(1)


def _clean_orphaned_phase_scratch() -> None:
    """Best-effort sweep of stale per-phase scratch directories.

    Workers SHOULD clean their own merged dirs in their ``finally``
    blocks (R6 fix #4 in `train_marker_loras_228.py`,
    `measure_leakage_228.py`, `compute_js_convergence_228.py`). If a
    worker is hard-killed (SIGKILL / OOM-killer) the ``finally`` block
    won't run; this helper sweeps anything left behind between phases.
    Never raises.
    """
    for scratch in (TMP_PHASE0_DIR, TMP_PHASE05_DIR, TMP_PHASE1_DIR):
        if scratch.exists():
            try:
                children = list(scratch.iterdir())
            except OSError:
                continue
            if not children:
                continue
            for child in children:
                try:
                    if child.is_dir():
                        shutil.rmtree(str(child), ignore_errors=True)
                except OSError as exc:
                    logger.warning("orphan-sweep: failed to rm %s: %s", child, exc)
            logger.info("orphan-sweep: cleaned %d stale entries under %s", len(children), scratch)


# ── State enumeration per phase ────────────────────────────────────────────


# Phase 0a uses a sentinel step (-1) so the same (source, step) tuple shape
# works through the existing worker plumbing. The Phase-0a worker script
# ignores --checkpoint-step entirely.
PHASE0A_SENTINEL_STEP = -1


def _enumerate_phase0a_states() -> list[tuple[str, int]]:
    """7 sources, each with sentinel step. Serialised in the coordinator."""
    return [(source, PHASE0A_SENTINEL_STEP) for source in sorted(ADAPTER_MAP.keys())]


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


# Phase 0a cache lives at this path (kept in sync with
# ``pregenerate_onpolicy_cache_228._cache_path``). We reach into the run-leakage
# data dir via a relative path so the test suite can monkey-patch it without
# loading project-side code.
def _phase0a_cache_path(source: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "leakage_v3_onpolicy"
        / "onpolicy_cache"
        / f"completions_{source}.json"
    )


def _phase0a_state_done(_output_dir: Path, source: str, _step: int) -> bool:
    """Phase 0a is done when the source's cache file exists and is non-empty."""
    p = _phase0a_cache_path(source)
    return p.exists() and p.stat().st_size > 0


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


def _phase0a_cmd(*, source: str, step: int, seed: int, **_) -> list[str]:
    # ``step`` is the sentinel; we pass nothing about it to the Phase 0a
    # worker (which has no --checkpoint-step argument).
    del step, seed  # unused — kept so the signature matches the registry contract
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pregenerate_onpolicy_cache_228.py"),
        "--source",
        source,
        "--gpu-id",
        "0",
    ]


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
    PHASE_PREGEN: {
        "name": "phase0a_onpolicy_cache",
        "enumerate": _enumerate_phase0a_states,
        "is_done": _phase0a_state_done,
        "build_cmd": _phase0a_cmd,
        # Force serial dispatch on a single GPU regardless of --num-gpus.
        # Each Phase 0a worker spawns vLLM directly; running multiple in
        # parallel reintroduces the same-process contention we are trying
        # to avoid (and they would all be racing the same physical GPU 0
        # under CUDA_VISIBLE_DEVICES narrowing anyway).
        "force_serial": True,
    },
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
    # Phases marked ``force_serial`` (today: Phase 0a) run on a single GPU
    # regardless of --num-gpus. This prevents two vLLM-spawning workers from
    # ever sharing a GPU; serialisation is the whole point of Phase 0a.
    effective_gpus = 1 if phase.get("force_serial") else num_gpus
    if effective_gpus != num_gpus:
        logger.info(
            "Phase %s force_serial=True; running on 1 GPU (was --num-gpus=%d)",
            phase_key,
            num_gpus,
        )
    logger.info("Phase %s: %d states across %d GPUs", phase_key, len(states), effective_gpus)

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
    for gpu_id in range(effective_gpus):
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
            "Which phase(s) to run. '0a' = on-policy cache pre-gen (serial, "
            "1 GPU), '0' = marker LoRA training, '0.5' = leakage measurement, "
            "'1' = JS sweep, 'all' = run all four in order."
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
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help=(
            "Skip the pre-launch disk free-space check. Default OFF; the check "
            "exists because the R5 launch died with ENOSPC after Phase 0 leaked "
            "merged dirs into /workspace. Use only when running off-pod."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=PRELAUNCH_DISK_FREE_GB,
        help=f"Pre-launch /workspace free-space minimum (GB). Default {PRELAUNCH_DISK_FREE_GB}.",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        raise SystemExit(f"--num-gpus must be >=1, got {args.num_gpus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.leakage_output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    phases_to_run = list(ALL_PHASES) if args.phase == PHASE_ALL else [args.phase]

    # R6 fix #4: pre-launch disk check + best-effort orphan cleanup. Both
    # are skipped under --dry-run so test-suite invocations don't trip on
    # `/workspace` being absent on local VMs.
    if not args.dry_run and not args.skip_disk_check:
        _check_prelaunch_disk(min_free_gb=args.min_free_gb)
    if not args.dry_run:
        _clean_orphaned_phase_scratch()

    shutdown = _ShutdownFlag()
    _install_signal_handlers(shutdown)

    # R6 fix #4: in-flight disk monitor — daemon thread that triggers a
    # clean shutdown if `/workspace` falls under DISK_ABORT_GB. Skipped on
    # dry-run for the same reason as the pre-launch check.
    disk_thread: threading.Thread | None = None
    if not args.dry_run:
        disk_thread = threading.Thread(
            target=_disk_monitor_thread,
            kwargs={"shutdown": shutdown},
            name="disk-monitor",
            daemon=True,
        )
        disk_thread.start()

    summaries: list[dict] = []
    artifact_full_name: str | None = None
    for phase_key in phases_to_run:
        if shutdown.is_set():
            logger.warning("Shutdown set; not starting phase %s", phase_key)
            break
        # Phase 0a writes to ``data/leakage_v3_onpolicy/onpolicy_cache/`` —
        # not configurable, just use the JS output dir for log routing.
        # Phase 0 sinks to HF Hub; ``output_dir`` is unused by its worker but
        # threaded through for the coordinator's log path.
        if phase_key in (PHASE_PREGEN, PHASE_MARKER):
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

        # R6: between phases, sweep stale scratch dirs (workers killed by
        # SIGKILL won't have run their finally blocks).
        if not args.dry_run:
            _clean_orphaned_phase_scratch()

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
