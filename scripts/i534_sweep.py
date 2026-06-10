# ruff: noqa: RUF001, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #534 — sweep dispatcher (10 cells across the ft-7b pod, waves of --n-gpus).

Forked from scripts/i530_sweep.py. Deltas (plan §4.3 i):
  * Worker = scripts/i534_run_cell.py (per-step snapshots + post-hoc fraction
    selection + 4-fraction eval); namespace 534 defaults (slab
    eval_results/issue_534, runs /workspace/runs/issue_534, per-cell logs
    issue-534-<cell>-seed<S>.log).
  * NEW `--skip-done`: consume the per-cell sentinels
    (`issue-534-<cell>-seed<S>-results.json`, or the `.processed` rename the
    poller applies) and SKIP already-completed cells — `i530_sweep.py` wrote
    sentinels but never consumed them (fact-checked in the plan), so the
    post-smoke full-sweep launch would have retrained the smoke cell.
  * Forwards the #534 worker knobs: `--snapshot-every-steps`,
    `--snapshot-max-count`, `--fractions`, `--skip-source-trajectory`,
    `--no-train-pool-from-hf`.

Architectural parity (post-#397 unification rule + plan §4.4): each sweep
cell is the SAME `i534_run_cell.py` invocation the smoke uses. Smoke = sweep
with `--cells c504v3_near --seeds 42 --n-gpus 1`. PASS_UNIFIED.

Usage (smoke, plan §10):
    nohup uv run python scripts/i534_sweep.py --n-gpus 1 \\
        --cells c504v3_near --seeds 42 \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json \\
        > /workspace/logs/issue534_smoke.log 2>&1 &

Usage (full sweep after smoke PASS):
    nohup uv run python scripts/i534_sweep.py --n-gpus 4 \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json --skip-done \\
        > /workspace/logs/issue534_sweep.log 2>&1 &

Usage (round-2 eval-only re-run from the EXISTING snapshots — NO retraining;
note: NO --skip-done, the prior broken-eval sentinels would skip every cell):
    nohup uv run python scripts/i534_sweep.py --n-gpus 4 --eval-only \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json \\
        > /workspace/logs/issue534_reeval.log 2>&1 &
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

log = logging.getLogger("i534.sweep")

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


def cell_sentinel_exists(log_dir: Path, cell: str, seed: int) -> bool:
    """True iff the per-cell results sentinel exists (raw OR poller-renamed).

    `poll_pipeline.py` renames consumed sentinels to `<name>.processed`; both
    spellings count as "done" for `--skip-done`.
    """
    base = log_dir / f"issue-534-{cell}-seed{seed}-results.json"
    return base.exists() or base.with_name(base.name + ".processed").exists()


def _cell_cmd(args: argparse.Namespace, cell: str, seed: int, gpu_id: int) -> list[str]:
    """Build the i534_run_cell.py invocation for one (cell, seed, gpu) slot."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i534_run_cell.py",
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
        "--snapshot-every-steps",
        str(args.snapshot_every_steps),
        "--snapshot-max-count",
        str(args.snapshot_max_count),
        "--fractions",
        args.fractions,
    ]
    if args.skip_source_trajectory:
        cmd.append("--skip-source-trajectory")
    if args.no_train_pool_from_hf:
        cmd.append("--no-train-pool-from-hf")
    if args.eval_only:
        cmd.append("--eval-only")
    return cmd


def _filter_skip_done(worklist: list[tuple[str, int]], log_dir: Path) -> list[tuple[str, int]]:
    """Drop (cell, seed) pairs whose per-cell results sentinel already exists."""
    kept: list[tuple[str, int]] = []
    for cell, seed in worklist:
        if cell_sentinel_exists(log_dir, cell, seed):
            log.info(
                "[phase=sweep_skip_done] cell=%s seed=%d — sentinel present, skipping.",
                cell,
                seed,
            )
        else:
            kept.append((cell, seed))
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arm-to-n-json",
        type=Path,
        required=True,
        help="Phase 0.5 gates artifact (forwarded to each cell's i534_run_cell.py).",
    )
    ap.add_argument("--n-gpus", type=int, default=4, help="Physical GPUs per wave (plan §9).")
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
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_534"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_534"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Skip (cell, seed) pairs whose per-cell results sentinel already "
            "exists under --log-dir (raw or .processed). NEW in the i534 fork: "
            "lets the full-sweep launch resume past the smoke cell without "
            "retraining it."
        ),
    )
    ap.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=1,
        help="Forwarded to i534_run_cell.py (per-step snapshot cadence; default 1).",
    )
    ap.add_argument(
        "--snapshot-max-count",
        type=int,
        default=64,
        help="Forwarded to i534_run_cell.py (snapshot cap; default 64).",
    )
    ap.add_argument(
        "--fractions",
        default="0.25,0.5,0.75,1.0",
        help="Forwarded to i534_run_cell.py (post-hoc fraction set).",
    )
    ap.add_argument(
        "--skip-source-trajectory",
        action="store_true",
        help="Forwarded to i534_run_cell.py (descope ladder item 1).",
    )
    ap.add_argument(
        "--no-train-pool-from-hf",
        action="store_true",
        help="Forwarded to i534_run_cell.py (rebuild pools via build_cell_504 instead).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "#534 round-2 re-run path: forward --eval-only to every cell — "
            "re-run ONLY the trajectory eval + sentinel from the EXISTING "
            "snapshots/index/manifest (NO retraining). Incompatible with "
            "--skip-done: the prior (broken-eval) sentinels exist for every "
            "cell, so --skip-done would silently no-op the whole re-run."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dispatch plan + commands without launching subprocesses.",
    )
    ap.add_argument(
        "--continue-on-cell-failure",
        action="store_true",
        help=(
            "If a cell subprocess exits non-zero, log + carry on with the "
            "remaining cells. Default = fail loud."
        ),
    )
    args = ap.parse_args(argv)

    if args.eval_only and args.skip_done:
        ap.error(
            "--eval-only is incompatible with --skip-done: every cell's prior "
            "(broken-eval) sentinel already exists, so --skip-done would skip "
            "ALL cells and the re-run would silently no-op."
        )

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=sweep] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Carry-over data dependencies from #472 — pulled ONCE in the parent so N
    # parallel per-cell subprocesses don't race the same HF downloads.
    if not args.dry_run:
        from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
            prepare_data_dependencies,
        )

        log.info("[phase=sweep_prepare_data] auto-downloading #472 carry-over artifacts")
        prepare_data_dependencies()

    cells = _parse_csv_str(args.cells)
    seeds = _parse_csv_int(args.seeds)
    n_gpus = args.n_gpus

    worklist: list[tuple[str, int]] = [(c, s) for c in cells for s in seeds]
    if args.skip_done:
        worklist = _filter_skip_done(worklist, args.log_dir)
    log.info(
        "[phase=sweep_plan] %d cells × %d seeds → %d (cell, seed) pairs to run; n_gpus=%d",
        len(cells),
        len(seeds),
        len(worklist),
        n_gpus,
    )
    if not worklist:
        log.info("[phase=done] nothing to run (all sentinels present).")
        return 0

    sweep_dispatch_log = args.slab_root / "sweep_dispatch.json"
    sweep_dispatch_log.parent.mkdir(parents=True, exist_ok=True)
    dispatch_history: list[dict] = []

    # Waves of `n_gpus` parallel subprocesses; each cell persists its own
    # trajectory + manifest + sentinel on completion (checkpoint-per-phase).
    for wave_idx, wave_start in enumerate(range(0, len(worklist), n_gpus)):
        wave = worklist[wave_start : wave_start + n_gpus]
        log.info("[phase=sweep_wave_%d_start] %d cells in this wave: %s", wave_idx, len(wave), wave)
        procs: list[tuple[subprocess.Popen | None, str, int, int, list[str]]] = []
        for gpu_id, (cell, seed) in enumerate(wave):
            cmd = _cell_cmd(args, cell, seed, gpu_id)
            # CVD: defense in depth — train/sft.py clobbers it with str(--gpu-id);
            # mirroring here keeps anything env-inspecting pre-sft.py correct.
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
            log_path = args.log_dir / f"issue-534-{cell}-seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = log_path.open("a")
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            procs.append((proc, cell, seed, gpu_id, cmd))

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
                    sweep_dispatch_log.write_text(json.dumps(dispatch_history, indent=2))
                    return rc

        if wave_had_failure:
            log.warning(
                "[phase=sweep_wave_%d_partial] wave completed with cell failures; "
                "carrying on per --continue-on-cell-failure.",
                wave_idx,
            )
        log.info("[phase=sweep_wave_%d_end] wave complete", wave_idx)
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
