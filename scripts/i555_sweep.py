# ruff: noqa: RUF001, RUF003  # em-dash + Qwen marker " ※" + Greek α intentional
#!/usr/bin/env python3
"""Task #555 — sweep dispatcher (40 production cells across the ft-7b pod, waves of --n-gpus).

Forked from scripts/i534_sweep.py. Deltas (plan #555 §4.3 b):
  * Worker = scripts/i555_run_cell.py (per-fresh-seed pool rebuild + step-5
    hard stop + 1-fraction selection + step-5 eval); namespace 555 defaults
    (slab eval_results/issue_555, runs /workspace/runs/issue_555, per-cell
    logs issue-555-<cell>-seed<S>.log).
  * DEFAULT_CELLS = the 4 positioned arms ONLY (no default_only — geometry
    predictors are undefined without a positioned negative, excluded from
    the regression by construction, same as the parent's fits).
  * DEFAULT_SEEDS = the 5 fresh replicate pairs {7,11},{19,23},{71,73},
    {101,103},{211,223} flattened (plan §4.1, pre-specified in the body).
  * Forwards `--hard-stop-at-step`, `--fractions`, `--hf-path-suffix` (the
    `_bandctrl` positive-control cell is NOT dispatched here — it launches
    FIRST as a separate i555_run_cell.py gate per the plan pipeline order).
  * `--skip-done` consumes the per-cell sentinels (suffix-aware:
    `issue-555-<cell>-seed<S><suffix>-results.json`, or the `.processed`
    rename) so the post-smoke full-sweep launch never retrains the smoke cell.

Architectural parity (post-#397 unification rule + plan §4.3): each sweep
cell is the SAME `i555_run_cell.py` invocation the smoke uses. Smoke = sweep
with `--cells c504v3_near --seeds 7 --n-gpus 1`. PASS_UNIFIED — every phase
the worker runs (pool build, train, selector, eval, uploads, sentinel)
derives its cell list from the same `--cells/--seeds` subset.

Usage (smoke, plan Repro card):
    nohup uv run python scripts/i555_sweep.py --n-gpus 1 \\
        --cells c504v3_near --seeds 7 \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json \\
        > /workspace/logs/issue555_smoke.log 2>&1 &

Usage (full sweep after smoke + _bandctrl control PASS):
    nohup uv run python scripts/i555_sweep.py --n-gpus 4 --skip-done \\
        --arm-to-n-json eval_results/issue_530/phase0_5_gates.json \\
        > /workspace/logs/issue555_sweep.log 2>&1 &
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

log = logging.getLogger("i555.sweep")

# The 4 positioned arms (plan §4.1; default_only EXCLUDED by construction).
DEFAULT_CELLS: tuple[str, ...] = (
    "c504v3_near",
    "c504v3_mid_near",
    "c504v3_mid_far",
    "c504v3_far",
)
# 5 fresh replicate pairs {7,11},{19,23},{71,73},{101,103},{211,223} (plan §4.1).
DEFAULT_SEEDS: tuple[int, ...] = (7, 11, 19, 23, 71, 73, 101, 103, 211, 223)


def _parse_csv_str(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_csv_int(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def cell_sentinel_exists(log_dir: Path, cell: str, seed: int, suffix: str = "") -> bool:
    """True iff the per-cell results sentinel exists (raw OR poller-renamed).

    `poll_pipeline.py` renames consumed sentinels to `<name>.processed`; both
    spellings count as "done" for `--skip-done`. Suffix-aware (#555 fix —
    the `_bandctrl` control sentinel never aliases a production cell's).
    """
    base = log_dir / f"issue-555-{cell}-seed{seed}{suffix}-results.json"
    return base.exists() or base.with_name(base.name + ".processed").exists()


def _cell_cmd(args: argparse.Namespace, cell: str, seed: int, gpu_id: int) -> list[str]:
    """Build the i555_run_cell.py invocation for one (cell, seed, gpu) slot."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i555_run_cell.py",
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
        "--hard-stop-at-step",
        str(args.hard_stop_at_step),
        "--snapshot-every-steps",
        str(args.snapshot_every_steps),
        "--snapshot-max-count",
        str(args.snapshot_max_count),
        "--fractions",
        args.fractions,
    ]
    if args.hf_path_suffix:
        cmd.extend(["--hf-path-suffix", args.hf_path_suffix])
    if args.skip_source_trajectory:
        cmd.append("--skip-source-trajectory")
    if args.eval_only:
        cmd.append("--eval-only")
    return cmd


def _filter_skip_done(
    worklist: list[tuple[str, int]], log_dir: Path, suffix: str
) -> list[tuple[str, int]]:
    """Drop (cell, seed) pairs whose per-cell results sentinel already exists."""
    kept: list[tuple[str, int]] = []
    for cell, seed in worklist:
        if cell_sentinel_exists(log_dir, cell, seed, suffix):
            log.info(
                "[phase=sweep_skip_done] cell=%s seed=%d suffix=%r — sentinel present, skipping.",
                cell,
                seed,
                suffix,
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
        help="Phase 0.5 gates artifact (forwarded to each cell's i555_run_cell.py).",
    )
    ap.add_argument("--n-gpus", type=int, default=4, help="Physical GPUs per wave (plan §8).")
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
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_555"))
    ap.add_argument("--runs-root", type=Path, default=Path("/workspace/runs/issue_555"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Skip (cell, seed) pairs whose per-cell results sentinel already "
            "exists under --log-dir (raw or .processed) — lets the full-sweep "
            "launch resume past the smoke cell without retraining it."
        ),
    )
    ap.add_argument(
        "--hard-stop-at-step",
        type=int,
        default=5,
        help=(
            "Forwarded to i555_run_cell.py (THE #555 read point; default 5; "
            "0 disables → parent band-stop behavior)."
        ),
    )
    ap.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=1,
        help="Forwarded to i555_run_cell.py (per-step snapshot cadence; default 1).",
    )
    ap.add_argument(
        "--snapshot-max-count",
        type=int,
        default=64,
        help="Forwarded to i555_run_cell.py (snapshot cap; default 64).",
    )
    ap.add_argument(
        "--fractions",
        default="1.0",
        help="Forwarded to i555_run_cell.py (single read point; frac 1.00 → step 5 exact).",
    )
    ap.add_argument(
        "--hf-path-suffix",
        default="",
        help=(
            "Forwarded to i555_run_cell.py (cell-variant suffix; the sweep's "
            "production cells use the empty default — the _bandctrl control is "
            "launched separately, BEFORE the sweep, per the plan pipeline order)."
        ),
    )
    ap.add_argument(
        "--skip-source-trajectory",
        action="store_true",
        help="Forwarded to i555_run_cell.py (descope ladder).",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Forward --eval-only to every cell — re-run ONLY the trajectory eval "
            "+ sentinel from the EXISTING snapshots/index/manifest (NO retraining). "
            "Incompatible with --skip-done: the prior sentinels exist for every "
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
            "sentinel already exists, so --skip-done would skip ALL cells and "
            "the re-run would silently no-op."
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
        worklist = _filter_skip_done(worklist, args.log_dir, args.hf_path_suffix)
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
            log_path = args.log_dir / f"issue-555-{cell}-seed{seed}{args.hf_path_suffix}.log"
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
