# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #530 — Phase 2 sweep dispatcher (10 cells across the ft-7b pod).

5 arms × 2 seeds = 10 LoRA training cells. The plan (§4.4 step 3 + §9)
calls for the 10 cells to run as 2 waves of ~5 on the `ft-7b` pod (4× H100),
with `+gpu_id=N` Hydra-style override to pin each cell to one physical GPU.

Per the checkpoint-per-phase rule + the `feedback_dispatcher_silent_death_hardening`
memory: each cell is its OWN subprocess (NOT in-process train_one_cell), so a
single cell's crash does not lose the others' output. Each cell's
trajectory.json + sentinel land on disk the moment that cell completes; the
analyzer reads them via glob.

Per `feedback_cvd_hydra_override`: env CUDA_VISIBLE_DEVICES alone is
insufficient because `train/sft.py:477` clobbers it with `cfg.gpu_id`. Each
launched subprocess gets `--gpu-id <N>` so the clobber writes the right
value. (We also set CVD in the env, which sft.py overwrites — both layers
of defense.)

Architectural parity (per the post-#397 unification rule + plan §4.7):
each sweep cell is the SAME `i530_run_cell.py` invocation the smoke uses.
Smoke = sweep with N=1. PASS_UNIFIED.

Usage:
    uv run python scripts/i530_sweep.py \\
        --arm-to-n-json /tmp/i530-arm-to-n.json \\
        --n-gpus 4 \\
        --slab-root eval_results/issue_530

  # OR to filter cells (e.g. for a single-wave run):
    uv run python scripts/i530_sweep.py \\
        --arm-to-n-json /tmp/i530-arm-to-n.json \\
        --n-gpus 4 \\
        --cells c504v3_near,c504v3_mid_near \\
        --seeds 42,137
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i530.sweep")

DEFAULT_CELLS: tuple[str, ...] = (
    "c504v3_near",
    "c504v3_mid_near",
    "c504v3_mid_far",
    "c504v3_far",
    "c504v3_default_only",
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 137)


def _parse_csv_str(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_csv_int(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help="Phase 0.5 output (forwarded to each cell's i530_run_cell.py).",
    )
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help="Number of physical GPUs to parallelize across (plan §9: 4× H100).",
    )
    ap.add_argument(
        "--cells",
        type=str,
        default=",".join(DEFAULT_CELLS),
        help=f"Comma-separated cell slugs (default: {','.join(DEFAULT_CELLS)}).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seeds (default: {','.join(str(s) for s in DEFAULT_SEEDS)}).",
    )
    ap.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_530"),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/workspace/runs/issue_530"),
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/workspace/logs"),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the cell-by-cell dispatch plan + commands without "
            "launching subprocesses. Used by the smoke/CI shape-check."
        ),
    )
    ap.add_argument(
        "--continue-on-cell-failure",
        action="store_true",
        help=(
            "If a cell subprocess exits non-zero, log + carry on with the "
            "remaining cells. Default = fail loud (the first cell crash aborts "
            "the sweep). Set when running overnight with partial-progress "
            "tolerated."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=sweep] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Carry-over data dependencies from #472 are gitignored; pull at the
    # pinned revision before dispatching cells. Idempotent. We do this
    # ONCE in the parent sweep process (NOT in each per-cell subprocess)
    # to avoid N parallel HF downloads racing on the same local files.
    if not args.dry_run:
        from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
            prepare_data_dependencies,
        )

        log.info("[phase=sweep_prepare_data] auto-downloading #472 carry-over artifacts")
        prepare_data_dependencies()

    cells = _parse_csv_str(args.cells)
    seeds = _parse_csv_int(args.seeds)
    n_gpus = args.n_gpus

    # Build the (cell, seed) work-list in round-robin GPU-assignment order so
    # each wave of `n_gpus` parallel processes covers a distinct set of
    # (cell, seed) pairs.
    worklist: list[tuple[str, int]] = [(c, s) for c in cells for s in seeds]
    log.info(
        "[phase=sweep_plan] %d cells × %d seeds = %d (cell, seed) pairs; n_gpus=%d",
        len(cells),
        len(seeds),
        len(worklist),
        n_gpus,
    )

    sweep_dispatch_log = args.slab_root / "sweep_dispatch.json"
    sweep_dispatch_log.parent.mkdir(parents=True, exist_ok=True)
    dispatch_history: list[dict] = []

    # Run in waves of `n_gpus` parallel subprocesses. Each cell's
    # subprocess writes its own trajectory + sentinel on completion, so a
    # crash in one cell doesn't lose the others' output (the
    # checkpoint-per-phase rule).
    for wave_idx, wave_start in enumerate(range(0, len(worklist), n_gpus)):
        wave = worklist[wave_start : wave_start + n_gpus]
        log.info(
            "[phase=sweep_wave_%d_start] %d cells in this wave: %s",
            wave_idx,
            len(wave),
            wave,
        )
        procs: list[tuple[subprocess.Popen | None, str, int, int, list[str]]] = []
        for gpu_id, (cell, seed) in enumerate(wave):
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/i530_run_cell.py",
                "--cell",
                cell,
                "--seed",
                str(seed),
                "--gpu-id",
                str(gpu_id),
                "--arm-to-n-json",
                str(args.arm_to_n_json),
                "--slab-root",
                str(args.slab_root),
                "--runs-root",
                str(args.runs_root),
                "--log-dir",
                str(args.log_dir),
            ]
            # CVD: defense in depth. train/sft.py will clobber this with
            # str(--gpu-id), so we mirror it here so anything inspecting
            # the env BEFORE sft.py runs sees the right device.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            log.info(
                "[phase=sweep_dispatch] wave=%d gpu=%d cell=%s seed=%d cmd=%s",
                wave_idx,
                gpu_id,
                cell,
                seed,
                shlex.join(cmd),
            )
            dispatch_history.append(
                {
                    "wave": wave_idx,
                    "gpu_id": gpu_id,
                    "cell": cell,
                    "seed": seed,
                    "cmd": cmd,
                    "ts_dispatch": datetime.now(UTC).isoformat(),
                }
            )
            if args.dry_run:
                procs.append((None, cell, seed, gpu_id, cmd))
                continue
            # Each cell's stdout/stderr lands in a per-cell log file so the
            # orchestrator can tail individual cells without log interleave.
            log_path = args.log_dir / f"issue-530-{cell}-seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = log_path.open("a")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            procs.append((proc, cell, seed, gpu_id, cmd))

        # Wait for the wave to complete before launching the next wave.
        if args.dry_run:
            log.info("[phase=sweep_wave_%d_dry] would launch %d cells", wave_idx, len(wave))
            continue

        wave_had_failure = False
        for proc, cell, seed, gpu_id, _cmd in procs:
            if proc is None:
                continue
            rc = proc.wait()
            log.info(
                "[phase=sweep_wave_%d_cell_done] cell=%s seed=%d gpu=%d rc=%d",
                wave_idx,
                cell,
                seed,
                gpu_id,
                rc,
            )
            if rc != 0:
                wave_had_failure = True
                if not args.continue_on_cell_failure:
                    log.error(
                        "[phase=sweep_abort] cell %s seed %d rc=%d; aborting sweep "
                        "(set --continue-on-cell-failure to carry on).",
                        cell,
                        seed,
                        rc,
                    )
                    # Persist what we have before exiting.
                    sweep_dispatch_log.write_text(json.dumps(dispatch_history, indent=2))
                    return rc

        if wave_had_failure:
            log.warning(
                "[phase=sweep_wave_%d_partial] wave completed with cell failures; "
                "carrying on per --continue-on-cell-failure.",
                wave_idx,
            )
        log.info("[phase=sweep_wave_%d_end] wave complete", wave_idx)
        # Brief settle (let nvidia-smi reflect the freed memory before the
        # next wave's GPU pin checks).
        time.sleep(2)

    sweep_dispatch_log.write_text(json.dumps(dispatch_history, indent=2))
    log.info(
        "[phase=done] sweep complete; dispatch log → %s (%d (cell, seed) launches)",
        sweep_dispatch_log,
        len(dispatch_history),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
