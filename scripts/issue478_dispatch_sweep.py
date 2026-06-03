#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #478 PHASE 4 — sweep dispatcher (smoke-first, 4-way parallel).

Per plan v5 §4.8 PHASE 3 + §4.9 Phase 3b (arm):

  * **Smoke IS the sweep with --cells 1 --seeds 1 --smoke-first** — same
    subprocess shape, same env injection, same WandB logging, same teardown
    sequence. UNIFIED smoke/sweep architecture (plan §4.6 smoke/sweep parity).
  * Smoke kill-criterion: after first cell finishes, read its result.json and
    STOP if ``smoke_check.saturated == True`` (g_logprob_source > -0.1).
  * Sweep: 4 concurrent issue478_run_cell.py subprocesses, one per GPU;
    round-robin (cell_id, seed) across free GPUs.
  * Arm cells (ARM_K2_*/ARM_K4_*) are run ONLY when ``--arm`` is passed
    (descope-first under compute pressure, plan §9 stratification #1).

CLI:
  --gpus 0,1,2,3          GPU ids (default: 0,1,2,3)
  --seeds 42,137          Seeds (default: 42,137)
  --cell-specs PATH       Default: data/issue_478/cell_specs.json
  --cells N               Cap total cells (smoke=1; default: all)
  --smoke-first           Run smoke cell SEQUENTIALLY first, gate on saturation.
  --skip-data-prep        Don't re-run Phase 0/0.5/1 (assume cached). Default OFF.
  --arm                   Include the 6 ARM cells (plan §4.9). Default OFF.
  --include-phase-0b      Run the marker-set validator (plan §4.8 Phase 0b)
                          BEFORE arm cells. Requires --arm.
  --dry-run               Print the cell × seed plan and exit 0.

The dispatcher does NOT shell out to scripts/task.py (per CLAUDE.md pod-side
rule); progress reaches the orchestrator only via the [phase=...] log lines +
the per-cell sentinel file written by issue478_run_cell.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()


def run_data_prep_phases(args: argparse.Namespace) -> None:
    """Phase 0 → 0.5 → 1 (resumable). Phase 2 is per-cell, just-in-time below."""
    env = {**os.environ}

    log.info("[dispatch] Phase 0: design validation ...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "issue478_validate_design.py")],
        check=True,
        env=env,
    )

    log.info("[dispatch] Phase 0.5: building cell specs ...")
    cell_specs_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue478_make_cell_specs.py"),
    ]
    if args.arm:
        cell_specs_cmd.append("--include-arm")
    subprocess.run(cell_specs_cmd, check=True, env=env)

    log.info("[dispatch] Phase 1: on-policy R generation (resumable cache) ...")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue478_generate_onpolicy_R.py"),
        "--gpu",
        str(args.gpus[0]),
    ]
    subprocess.run(cmd, check=True, env=env)


def maybe_run_phase_0b(args: argparse.Namespace) -> None:
    """Phase 0b (marker-set validator, OPTIONAL — only if --arm + --include-phase-0b)."""
    if not args.include_phase_0b:
        return
    env = {**os.environ}
    log.info("[dispatch] Phase 0b: marker-set validator (real-eval-distribution probe) ...")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue478_validate_markers.py"),
            "--gpu",
            str(args.gpus[0]),
        ],
        check=True,
        env=env,
    )


def assemble_cell_data(cell_id: str, seed: int) -> None:
    """Phase 2 for one (cell_id, seed) inline."""
    jsonl = (
        PROJECT_ROOT / "data" / "issue_478" / "training_jsonl" / f"cell_{cell_id}_seed{seed}.jsonl"
    )
    if jsonl.exists():
        log.info("[dispatch] Phase 2 cached: %s", jsonl.name)
        return
    env = {**os.environ}
    log.info("[dispatch] Phase 2 assembling: cell=%s seed=%d", cell_id, seed)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue478_make_training_data.py"),
            "--cell-id",
            cell_id,
            "--seed",
            str(seed),
        ],
        check=True,
        env=env,
    )


def spawn_cell_subprocess(
    cell_id: str,
    seed: int,
    gpu: int,
    log_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Spawn issue478_run_cell.py for one (cell_id, seed, gpu)."""
    env = {**os.environ}
    log_path = log_dir / f"cell_{cell_id}_seed{seed}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue478_run_cell.py"),
        "--cell-id",
        cell_id,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
    ]
    if extra_args:
        cmd.extend(extra_args)
    env["EPM_ISSUE478_MARKER_TEXT_QUOTED"] = shlex.quote(" ※")

    log.info("[dispatch] Spawning %s seed=%d gpu=%d → %s", cell_id, seed, gpu, log_path.name)
    with log_path.open("w") as lf:
        return subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )


def smoke_gate_check(cell_id: str, seed: int) -> bool:
    """True if smoke result.json shows saturation (kill the sweep)."""
    p = PROJECT_ROOT / "eval_results" / "issue_478" / f"cell_{cell_id}_seed{seed}" / "result.json"
    if not p.exists():
        log.error("[smoke-gate] result.json missing at %s — treating as FAIL.", p)
        return True
    data = json.loads(p.read_text())
    sat = data.get("smoke_check", {}).get("saturated", None)
    g_logp = data.get("smoke_check", {}).get("g_logprob_source", None)
    log.info(
        "[smoke-gate] g_logprob_source=%.4f saturated=%s",
        g_logp if g_logp is not None else float("nan"),
        sat,
    )
    if sat:
        log.error(
            "[smoke-gate] FAIL: g_logprob_source=%.4f > -0.1 (saturated). "
            "Drop epochs to 1 OR lr to 2e-6, then re-smoke.",
            g_logp,
        )
    return bool(sat)


def main() -> int:  # noqa: C901 — argparse + smoke + parallel-dispatch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--seeds", type=str, default="42,137")
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_478" / "cell_specs.json"),
    )
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--cell-id", type=str, default="")
    parser.add_argument("--smoke-first", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--arm", action="store_true", help="Include the 6 ARM cells (§4.9)")
    parser.add_argument(
        "--include-phase-0b",
        action="store_true",
        help="Run marker-set validator before ARM cells (requires --arm)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "issue_478_sweep"),
    )
    args = parser.parse_args()

    if args.include_phase_0b and not args.arm:
        raise SystemExit("--include-phase-0b requires --arm")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not gpus or not seeds:
        raise SystemExit("--gpus and --seeds must be non-empty")
    log.info("Dispatcher: gpus=%s seeds=%s arm=%s", gpus, seeds, args.arm)
    args.gpus = gpus

    if not args.skip_data_prep:
        run_data_prep_phases(args)

    maybe_run_phase_0b(args)

    specs_path = Path(args.cell_specs)
    if not specs_path.exists():
        raise SystemExit(f"cell_specs.json missing: {specs_path}. Run Phase 0.5 first.")
    specs = json.loads(specs_path.read_text())

    # Order: smoke cell first (K1_c00), then CORE rest, then ARM (if --arm).
    def _order_key(s):
        if s["cell_id"] == "K1_c00":
            return (0, s["cell_id"])
        if s["track"] == "CORE":
            return (1, s["cell_id"])
        return (2, s["cell_id"])  # ARM

    specs_sorted = sorted(specs, key=_order_key)

    # Filter out ARM cells unless --arm.
    if not args.arm:
        specs_sorted = [s for s in specs_sorted if s["track"] != "ARM"]

    if args.cell_id:
        specs_sorted = [s for s in specs_sorted if s["cell_id"] == args.cell_id]
        if not specs_sorted:
            raise SystemExit(f"--cell-id {args.cell_id!r} not in specs")
    elif args.cells:
        specs_sorted = specs_sorted[: args.cells]

    work_items: list[tuple[str, int]] = [
        (s["cell_id"], seed) for s in specs_sorted for seed in seeds
    ]
    log.info(
        "Total work items: %d (%d cells × %d seeds)",
        len(work_items),
        len(specs_sorted),
        len(seeds),
    )

    if args.dry_run:
        for wi in work_items:
            log.info("DRY  %s seed=%d", wi[0], wi[1])
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Smoke-first gate (SAME subprocess shape as the sweep — parity) ─
    if args.smoke_first and work_items:
        smoke_cell, smoke_seed = work_items[0]
        log.info(
            "[smoke-first] cell=%s seed=%d — running SEQUENTIALLY for kill-criterion gate",
            smoke_cell,
            smoke_seed,
        )
        assemble_cell_data(smoke_cell, smoke_seed)
        proc = spawn_cell_subprocess(
            smoke_cell, smoke_seed, gpus[0], log_dir, extra_args=["--smoke"]
        )
        rc = proc.wait()
        if rc != 0:
            log.error("[smoke-first] Smoke cell exit %d — STOP sweep.", rc)
            print("[phase=failed]", flush=True)
            return rc
        if smoke_gate_check(smoke_cell, smoke_seed):
            log.error("[smoke-first] Kill-criterion HIT (saturated). STOP sweep.")
            print("[phase=failed]", flush=True)
            return 2
        log.info("[smoke-first] PASS — proceeding to parallel sweep.")
        work_items = work_items[1:]

    # ── Parallel dispatch ─────────────────────────────────────────────
    free_gpus = list(gpus)
    inflight: dict[int, tuple[subprocess.Popen, str, int]] = {}
    failures: list[tuple[str, int, int]] = []

    def poll_inflight():
        done = []
        for gpu, (proc, cell, seed) in inflight.items():
            if proc.poll() is not None:
                done.append(gpu)
                rc = proc.returncode
                log.info("[dispatch] DONE cell=%s seed=%d gpu=%d rc=%d", cell, seed, gpu, rc)
                if rc != 0:
                    failures.append((cell, seed, rc))
        for g in done:
            inflight.pop(g)
            free_gpus.append(g)

    queue = list(work_items)
    while queue or inflight:
        while free_gpus and queue:
            cell, seed = queue.pop(0)
            gpu = free_gpus.pop(0)
            try:
                assemble_cell_data(cell, seed)
            except subprocess.CalledProcessError as e:
                log.error(
                    "[dispatch] Phase-2 assembly FAILED for cell=%s seed=%d (%s)", cell, seed, e
                )
                failures.append((cell, seed, -1))
                free_gpus.append(gpu)
                continue
            proc = spawn_cell_subprocess(cell, seed, gpu, log_dir)
            inflight[gpu] = (proc, cell, seed)
        if inflight:
            time.sleep(15)
            poll_inflight()

    log.info("[dispatch] All work items done. %d failure(s).", len(failures))
    if failures:
        for cell, seed, rc in failures:
            log.error("[dispatch]   FAILED: cell=%s seed=%d rc=%d", cell, seed, rc)
        print("[phase=failed]", flush=True)
        return 1
    print("[phase=done]", flush=True)
    return 0


def _main_with_failure_phase_guard() -> int:
    """Ensure [phase=failed] is printed on ANY exit path (per #405 Round-4 fix 2)."""
    try:
        rc = main()
    except KeyboardInterrupt:
        print("[phase=failed]", flush=True)
        raise
    except subprocess.CalledProcessError as e:
        log.error("[dispatch] CalledProcessError in early phase: %s", e)
        print("[phase=failed]", flush=True)
        return e.returncode if e.returncode else 1
    except SystemExit as e:
        code = e.code
        if isinstance(code, int) and code == 0:
            return 0
        log.error("[dispatch] SystemExit(%r) in early phase", code)
        print("[phase=failed]", flush=True)
        if isinstance(code, int):
            return code
        return 1
    except Exception:
        log.exception("[dispatch] unexpected exception in main()")
        print("[phase=failed]", flush=True)
        raise
    return rc


if __name__ == "__main__":
    sys.exit(_main_with_failure_phase_guard())
