#!/usr/bin/env python3
"""Task #496 -- pod-side 12-cell sweep dispatcher.

SMOKE = SWEEP with --cells 1 (one cell, default villain warmth). Same dispatcher,
same subprocess shape, same env injection, same teardown sequence -- only the
cell list differs. This is the architecture-parity contract: PASS_UNIFIED.

The dispatcher iterates over (arm, source) pairs and fans out
``train_one_cell.run_one_cell`` invocations as subprocesses, ``--parallel-gpus``
at a time, each pinned to its own GPU via ``CUDA_VISIBLE_DEVICES`` AND
``--gpu-id`` (per the CLAUDE.md gotcha: ``train.sft.train_lora`` clobbers env
CUDA_VISIBLE_DEVICES with ``cfg.gpu_id`` default 0).

Per cell:
    1. Build training pool (warmth: per-source from Phase 0 corpus; sycophancy:
       download #411 verbatim pool from HF).
    2. Spawn a subprocess running ``warmth_sycophancy_496.train_one_cell``.
    3. Subprocess writes ``[phase=...]`` markers to a per-cell log + a sentinel
       to ``/workspace/logs/issue-496-<epm_results>-<epoch>.json`` on completion.
    4. Dispatcher drains subprocesses round-robin per GPU.

NEVER shells out to ``scripts/task.py`` (pod-side rule, CLAUDE.md).

End-of-run sentinel + ``[phase=done]`` line at dispatcher exit:
    /workspace/logs/issue-496-epm_results-<epoch>.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

# Load credentials BEFORE we read os.environ for subprocess env=... kwarg.
# Per CLAUDE.md subprocess-env passthrough rule.
load_dotenv()

log = logging.getLogger("issue_496.dispatch")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES: tuple[str, ...] = (
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
)
DEFAULT_ARMS: tuple[str, ...] = ("warmth", "sycophancy")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return None


def _emit_phase(name: str) -> None:
    """Log a ``[phase=<name>]`` line for ``poll_pipeline.py``'s tail scanner."""
    log.info("[phase=%s] ts=%s host=%s", name, datetime.now(UTC).isoformat(), socket.gethostname())


def _write_sentinel(
    sentinel_dir: Path, kind: str, payload: dict[str, object], version: int = 1
) -> Path:
    """Write a poll_pipeline-compatible end-of-run sentinel.

    Filename: ``issue-496-<kind_slug>-<epoch_seconds>.json``.
    JSON: required ``sentinel_schema_version=1``, ``kind``, ``version`` keys
    per ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS``.
    """
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time())
    slug = kind.replace(":", "_")
    fname = f"issue-496-{slug}-{epoch}.json"
    out = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "ts": datetime.now(UTC).isoformat(),
        "by": "dispatch_warmth_sycophancy_496",
        "task_id": 496,
        "note": payload,
    }
    p = sentinel_dir / fname
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote sentinel %s -> %s", kind, p)
    return p


def _build_training_pool_for_cell(
    arm: str,
    source: str,
    warmth_train_pool_path: Path | None,
    pool_dir: Path,
    smoke_n_positive: int | None,
) -> Path:
    """Build (or download) the per-cell training pool JSONL. Returns path."""
    from explore_persona_space.experiments.warmth_sycophancy_496.build_training_pool import (
        build_training_pool_for_arm,
    )

    out_path = pool_dir / f"{arm}_{source}_seed42_pool.jsonl"
    pool_dir.mkdir(parents=True, exist_ok=True)
    build_training_pool_for_arm(
        arm=arm,
        source=source,
        output_path=out_path,
        warmth_train_pool_path=warmth_train_pool_path,
        smoke_n_positive=smoke_n_positive,
    )
    return out_path


def _run_cell_subprocess(
    *,
    arm: str,
    source: str,
    seed: int,
    train_jsonl: Path,
    eval_pool: Path,
    output_dir: Path,
    eval_out_dir: Path,
    gpu_id: int,
    sentinel_dir: Path,
    panel_subset: list[str] | None,
    n_rollouts: int,
    max_new_tokens: int,
    no_upload: bool,
    no_eval: bool,
    keep_merged: bool,
    log_path: Path,
) -> subprocess.Popen:
    """Spawn one train_one_cell subprocess; return the Popen handle."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "explore_persona_space.experiments.warmth_sycophancy_496.train_one_cell",
        "--arm",
        arm,
        "--source",
        source,
        "--seed",
        str(seed),
        "--train-jsonl",
        str(train_jsonl),
        "--eval-pool",
        str(eval_pool),
        "--output-dir",
        str(output_dir),
        "--eval-out-dir",
        str(eval_out_dir),
        "--gpu-id",
        str(gpu_id),
        "--sentinel-dir",
        str(sentinel_dir),
        "--n-rollouts",
        str(n_rollouts),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if panel_subset:
        cmd.extend(["--panel-subset", *panel_subset])
    if no_upload:
        cmd.append("--no-upload")
    if no_eval:
        cmd.append("--no-eval")
    if keep_merged:
        cmd.append("--keep-merged")

    env = {**os.environ}
    # Subprocess CVD pinning is mandatory; cfg.gpu_id additionally clobbers env CVD.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Launching cell (arm=%s source=%s gpu_id=%d) -> %s", arm, source, gpu_id, log_path)
    # SIM115 N/A: the log handle must outlive this function so the spawned
    # subprocess can write to it; _drain_round closes it after proc.wait().
    log_f = open(log_path, "ab")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc._epm_log_handle = log_f  # type: ignore[attr-defined]
    return proc


def _drain_round(procs: list[subprocess.Popen], cell_labels: list[str]) -> tuple[int, list[str]]:
    """Wait for ALL procs in the current round to finish; return (n_failed, failed_labels)."""
    n_failed = 0
    failed: list[str] = []
    for proc, label in zip(procs, cell_labels, strict=True):
        rc = proc.wait()
        h = getattr(proc, "_epm_log_handle", None)
        if h is not None:
            with contextlib.suppress(Exception):
                h.close()
        if rc != 0:
            log.error("Cell %s FAILED (rc=%d)", label, rc)
            n_failed += 1
            failed.append(label)
        else:
            log.info("Cell %s done (rc=0)", label)
    return n_failed, failed


def run_sweep(
    *,
    arms: list[str],
    sources: list[str],
    seed: int,
    warmth_train_pool_path: Path,
    sycophancy_eval_pool_path: Path,
    parallel_gpus: int,
    slab_root: Path,
    pool_dir: Path,
    sentinel_dir: Path,
    logs_dir: Path,
    panel_subset: list[str] | None = None,
    n_rollouts: int = 10,
    max_new_tokens: int = 512,
    no_upload: bool = False,
    no_eval: bool = False,
    keep_merged: bool = False,
    smoke_n_positive: int | None = None,
) -> dict[str, object]:
    """Run the sweep. Smoke = sweep with len(arms)*len(sources) == 1."""
    _emit_phase("sweep_start")
    cells: list[tuple[str, str]] = [(a, s) for a in arms for s in sources]
    log.info("Sweep: %d cells (%s), parallel_gpus=%d", len(cells), cells, parallel_gpus)

    # Build pools first (CPU, fast; surfaces #411 download failures BEFORE any train).
    _emit_phase("build_pools")
    pool_paths: dict[tuple[str, str], Path] = {}
    for arm, source in cells:
        pool_paths[(arm, source)] = _build_training_pool_for_cell(
            arm=arm,
            source=source,
            warmth_train_pool_path=warmth_train_pool_path,
            pool_dir=pool_dir,
            smoke_n_positive=smoke_n_positive,
        )
        log.info("pool built: arm=%s source=%s -> %s", arm, source, pool_paths[(arm, source)])

    # Train+eval batches of `parallel_gpus` cells at a time.
    _emit_phase("train_eval_start")
    sweep_t0 = time.time()
    total_failed = 0
    failed_cells: list[str] = []
    for batch_start in range(0, len(cells), parallel_gpus):
        batch = cells[batch_start : batch_start + parallel_gpus]
        procs: list[subprocess.Popen] = []
        labels: list[str] = []
        for gpu_idx, (arm, source) in enumerate(batch):
            label = f"{arm}_{source}_seed{seed}"
            output_dir = slab_root / "checkpoints" / f"{arm}_{source}_seed{seed}"
            eval_out_dir = slab_root / arm / source / f"seed_{seed}"
            log_path = logs_dir / f"cell_{label}.log"
            proc = _run_cell_subprocess(
                arm=arm,
                source=source,
                seed=seed,
                train_jsonl=pool_paths[(arm, source)],
                eval_pool=sycophancy_eval_pool_path,
                output_dir=output_dir,
                eval_out_dir=eval_out_dir,
                gpu_id=gpu_idx,
                sentinel_dir=sentinel_dir,
                panel_subset=panel_subset,
                n_rollouts=n_rollouts,
                max_new_tokens=max_new_tokens,
                no_upload=no_upload,
                no_eval=no_eval,
                keep_merged=keep_merged,
                log_path=log_path,
            )
            procs.append(proc)
            labels.append(label)
        n_failed, fails = _drain_round(procs, labels)
        total_failed += n_failed
        failed_cells.extend(fails)
        log.info(
            "Batch %d done (%d cells, %d failed)",
            batch_start // parallel_gpus + 1,
            len(batch),
            n_failed,
        )

    sweep_wall = time.time() - sweep_t0
    _emit_phase("sweep_end")
    log.info(
        "Sweep complete in %.1fs. total_cells=%d total_failed=%d",
        sweep_wall,
        len(cells),
        total_failed,
    )

    summary = {
        "n_cells": len(cells),
        "n_failed": total_failed,
        "failed_cells": failed_cells,
        "sweep_wall_seconds": round(sweep_wall, 1),
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_sentinel(sentinel_dir, "epm:results", summary)
    _emit_phase("done")
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated arms (default: warmth,sycophancy).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source personas (default: 6 #411 sources).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmth-train-pool",
        type=Path,
        default=REPO_ROOT / "data" / "issue_496" / "warmth_prompts" / "train_200.jsonl",
    )
    parser.add_argument(
        "--sycophancy-eval-pool",
        type=Path,
        default=REPO_ROOT / "data" / "issue_411" / "wrong_claims" / "eval_50.jsonl",
    )
    parser.add_argument("--parallel-gpus", type=int, default=4)
    parser.add_argument(
        "--slab-root",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_496",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_496" / "training_pools",
    )
    parser.add_argument(
        "--sentinel-dir",
        type=Path,
        default=Path("/workspace/logs"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs",
    )
    parser.add_argument("--panel-subset", nargs="*", default=None)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--cells",
        type=int,
        default=None,
        help="Limit to first N (arm, source) cells (smoke aid). Default: all 12.",
    )
    parser.add_argument(
        "--smoke-n-positive",
        type=int,
        default=None,
        help="Per-cell positives override (warmth arm only). Smoke aid.",
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--keep-merged", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    cells = [(a, s) for a in arms for s in sources]
    if args.cells is not None:
        cells = cells[: args.cells]
        arms_used = sorted({a for a, _ in cells})
        sources_used = []
        seen_pairs: set[tuple[str, str]] = set()
        for a, s in cells:
            if (a, s) not in seen_pairs:
                seen_pairs.add((a, s))
                if s not in sources_used:
                    sources_used.append(s)
        arms = arms_used
        sources = sources_used

    run_sweep(
        arms=arms,
        sources=sources,
        seed=args.seed,
        warmth_train_pool_path=args.warmth_train_pool,
        sycophancy_eval_pool_path=args.sycophancy_eval_pool,
        parallel_gpus=args.parallel_gpus,
        slab_root=args.slab_root,
        pool_dir=args.pool_dir,
        sentinel_dir=args.sentinel_dir,
        logs_dir=args.logs_dir,
        panel_subset=args.panel_subset,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        no_upload=args.no_upload,
        no_eval=args.no_eval,
        keep_merged=args.keep_merged,
        smoke_n_positive=args.smoke_n_positive,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
