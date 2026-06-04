#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #490 PHASE 4 — sweep dispatcher (smoke-first, N-way parallel).

Per plan v1 §4.5 / §4.8 (smoke/sweep parity):

  * **Smoke IS the sweep with --cells 1 --seeds 1 --smoke-first** — same
    subprocess shape, same env injection, same WandB logging, same
    teardown sequence. UNIFIED smoke/sweep architecture.
  * Smoke kill-criterion: after first cell finishes, read its result.json
    and STOP if ``smoke_check.saturated == True`` (g_logprob_source > -0.1).
  * Sweep: N concurrent issue490_run_cell.py subprocesses, one per GPU;
    round-robin (cell_id, seed) across free GPUs.
  * Smoke cell ordering: POOLED-SINGLE-2D-A on pair0 (the saturation-prone
    condition × first ARM-matched pair). The remaining 79 cells run in
    parallel after smoke passes.

CLI:
  --gpus 0,1,2,3          GPU ids (default: 0,1,2,3)
  --seeds 42,137          Seeds (default: 42,137)
  --cell-specs PATH       Default: data/issue_490/cell_specs.json
  --cells N               Cap total cells (smoke=1; default: all)
  --cell-id ID            Restrict to a specific cell (overrides --cells)
  --smoke-first           Run smoke cell SEQUENTIALLY first, gate on saturation.
  --skip-data-prep        Don't re-run Phase 0/0.5/1 (assume cached).
  --dry-run               Print the cell × seed plan and exit 0.
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
        [sys.executable, str(PROJECT_ROOT / "scripts" / "issue490_validate_design.py")],
        check=True,
        env=env,
    )

    log.info("[dispatch] Phase 0.5: building cell specs ...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "issue490_make_cell_specs.py")],
        check=True,
        env=env,
    )

    log.info("[dispatch] Phase 1: on-policy R generation (resumable cache) ...")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue490_generate_onpolicy_R.py"),
        "--gpu",
        str(args.gpus[0]),
    ]
    subprocess.run(cmd, check=True, env=env)


def assemble_cell_data(cell_id: str, seed: int) -> None:
    """Phase 2 for one (cell_id, seed) inline."""
    jsonl = (
        PROJECT_ROOT / "data" / "issue_490" / "training_jsonl" / f"cell_{cell_id}_seed{seed}.jsonl"
    )
    if jsonl.exists():
        log.info("[dispatch] Phase 2 cached: %s", jsonl.name)
        return
    env = {**os.environ}
    log.info("[dispatch] Phase 2 assembling: cell=%s seed=%d", cell_id, seed)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "issue490_make_training_data.py"),
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
    """Spawn issue490_run_cell.py for one (cell_id, seed, gpu)."""
    env = {**os.environ}
    log_path = log_dir / f"cell_{cell_id}_seed{seed}_gpu{gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "issue490_run_cell.py"),
        "--cell-id",
        cell_id,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
    ]
    if extra_args:
        cmd.extend(extra_args)
    # Marker text threaded with shlex.quote (per marker-leakage rule).
    env["EPM_ISSUE490_MARKER_TEXT_QUOTED"] = shlex.quote(" ※")

    log.info(
        "[dispatch] Spawning %s seed=%d gpu=%d → %s",
        cell_id,
        seed,
        gpu,
        log_path.name,
    )
    with log_path.open("w") as lf:
        return subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )


def smoke_gate_check(cell_id: str, seed: int) -> bool:
    """True if smoke result.json shows saturation (kill the sweep)."""
    p = PROJECT_ROOT / "eval_results" / "issue_490" / f"cell_{cell_id}_seed{seed}" / "result.json"
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
            "Promote fallback DV (KL-from-base) and drop epochs or lr.",
            g_logp,
        )
    return bool(sat)


def _smoke_cell_id(specs: list[dict]) -> str | None:
    """Pick the POOLED-SINGLE-2D-A cell on the first pair as the smoke cell.

    This is the saturation-prone condition × the first ARM-matched pair;
    matches the plan §4.8 design.
    """
    for s in specs:
        if s["pair_id"] == "pair0" and s["condition"] == "pooled_2D_A":
            return s["cell_id"]
    return None


def _order_key(s: dict, smoke_id: str | None) -> tuple[int, str]:
    if smoke_id is not None and s["cell_id"] == smoke_id:
        return (0, s["cell_id"])
    return (1, s["cell_id"])


def main() -> int:  # noqa: C901 — argparse + smoke + parallel-dispatch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--seeds", type=str, default="42,137")
    parser.add_argument(
        "--cell-specs",
        type=str,
        default=str(PROJECT_ROOT / "data" / "issue_490" / "cell_specs.json"),
    )
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--cell-id", type=str, default="")
    parser.add_argument("--smoke-first", action="store_true")
    parser.add_argument("--skip-data-prep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "issue_490_sweep"),
    )
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not gpus or not seeds:
        raise SystemExit("--gpus and --seeds must be non-empty")
    log.info("Dispatcher: gpus=%s seeds=%s", gpus, seeds)
    args.gpus = gpus

    if not args.skip_data_prep:
        run_data_prep_phases(args)

    specs_path = Path(args.cell_specs)
    if not specs_path.exists():
        raise SystemExit(f"cell_specs.json missing: {specs_path}. Run Phase 0.5 first.")
    specs = json.loads(specs_path.read_text())

    smoke_id = _smoke_cell_id(specs)
    specs_sorted = sorted(specs, key=lambda s: _order_key(s, smoke_id))

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
                    "[dispatch] Phase-2 assembly FAILED for cell=%s seed=%d (%s)",
                    cell,
                    seed,
                    e,
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
    """Ensure [phase=failed] is printed on ANY exit path (per #478)."""
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
