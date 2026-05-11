#!/usr/bin/env python3
"""Orchestrator for dose-response marker survival experiment (issue #139).

Launches cells in parallel across GPUs. Each cell is a subprocess running
run_dose_response_cell.py.

Usage:
    # Phase 0: Ultra-low-dose pilot (3 cells on GPUs 4-6)
    python scripts/run_dose_response_orchestrator.py --phase 0 --gpus 4,5,6

    # Phase 1: Full scan (13 cells on GPUs 4-7)
    python scripts/run_dose_response_orchestrator.py --phase 1 --gpus 4,5,6,7

    # Phase 2: Multi-seed (6 cells on GPUs 4-7)
    python scripts/run_dose_response_orchestrator.py --phase 2 --gpus 4,5,6,7 --critical_dose 10

    # Single cell
    python scripts/run_dose_response_orchestrator.py --single --max_steps 10 --sft_type em \
        --seed 42 --gpus 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_ROOT = "/workspace/dose_response_139"
SCRIPT = "/workspace/explore-persona-space/scripts/run_dose_response_cell.py"


def launch_cell(
    max_steps: int,
    sft_type: str,
    seed: int,
    gpu: int,
    personas: str = "all_plus_evil_ai",
    warmup_steps: int | None = None,
    output_root: str = OUTPUT_ROOT,
) -> tuple[str, subprocess.Popen]:
    """Launch a single cell as a background subprocess."""
    cell_name = f"steps{max_steps}_{sft_type}_seed{seed}"
    log_dir = Path(output_root) / cell_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stdout.log"

    cmd = [
        sys.executable,
        SCRIPT,
        "--max_steps",
        str(max_steps),
        "--sft_type",
        sft_type,
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--personas",
        personas,
        "--output_root",
        output_root,
        "--skip_training",
        "--skip_eval",
    ]
    if warmup_steps is not None:
        cmd.extend(["--warmup_steps", str(warmup_steps)])

    print(f"[LAUNCH] {cell_name} on GPU {gpu} -> {log_file}", flush=True)
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return cell_name, proc


def wait_for_cells(
    cells: list[tuple[str, subprocess.Popen]],
    check_interval: int = 30,
) -> dict[str, int]:
    """Wait for all cells to finish. Returns {cell_name: return_code}."""
    results = {}
    remaining = list(cells)

    while remaining:
        still_running = []
        for name, proc in remaining:
            rc = proc.poll()
            if rc is not None:
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[DONE] {name}: {status}", flush=True)
                results[name] = rc
            else:
                still_running.append((name, proc))
        remaining = still_running
        if remaining:
            names = [n for n, _ in remaining]
            print(f"[WAIT] {len(remaining)} cells running: {', '.join(names)}", flush=True)
            time.sleep(check_interval)

    return results


def collect_results(output_root: str = OUTPUT_ROOT) -> list[dict]:
    """Collect all marker_eval.json results from completed cells."""
    root = Path(output_root)
    results = []
    for eval_file in sorted(root.glob("*/eval/marker_eval.json")):
        data = json.loads(eval_file.read_text())
        results.append(data)
    return results


def print_summary(results: list[dict]):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("DOSE-RESPONSE SUMMARY")
    print("=" * 80)
    print(
        f"{'Cell':<30} {'Steps':>6} {'Type':<8} {'Seed':>4} "
        f"{'evil_ai%':>9} {'asst%':>7} {'villain%':>9} {'max_byst%':>10}"
    )
    print("-" * 80)

    # Sort by (sft_type, max_steps, seed)
    results.sort(key=lambda r: (r.get("sft_type", ""), r.get("max_steps", 0), r.get("seed", 0)))

    for r in results:
        s = r.get("summary", {})
        evil_ai = s.get("evil_ai_strict_rate", None)
        asst = s.get("assistant_strict_rate", None)
        villain = s.get("villain_strict_rate", None)
        max_byst = s.get("max_bystander_strict_rate", None)

        evil_ai_str = f"{100 * evil_ai:.1f}%" if evil_ai is not None else "N/A"
        asst_str = f"{100 * asst:.1f}%" if asst is not None else "N/A"
        villain_str = f"{100 * villain:.1f}%" if villain is not None else "N/A"
        max_byst_str = f"{100 * max_byst:.1f}%" if max_byst is not None else "N/A"

        print(
            f"{r.get('cell_name', '?'):<30} {r.get('max_steps', '?'):>6} "
            f"{r.get('sft_type', '?'):<8} {r.get('seed', '?'):>4} "
            f"{evil_ai_str:>9} {asst_str:>7} {villain_str:>9} {max_byst_str:>10}"
        )

    print("=" * 80)


def run_phase0(gpus: list[int]):
    """Phase 0: Ultra-low-dose pilot. {1, 3, 5} steps x EM only x seed=42."""
    doses = [1, 3, 5]
    if len(gpus) < len(doses):
        print(
            f"WARNING: {len(doses)} cells but only {len(gpus)} GPUs. "
            "Some cells will share GPUs (sequential)."
        )

    cells = []
    for i, dose in enumerate(doses):
        gpu = gpus[i % len(gpus)]
        name, proc = launch_cell(
            max_steps=dose,
            sft_type="em",
            seed=42,
            gpu=gpu,
            personas="pilot_3",
            warmup_steps=0,
        )
        cells.append((name, proc))

    wait_for_cells(cells)

    # Check gate
    all_results = collect_results()
    pilot_results = [
        r for r in all_results if r.get("max_steps", 999) <= 5 and r.get("sft_type") == "em"
    ]

    print_summary(pilot_results)

    # Gate G_cliff: if evil_ai rate < 10% at ALL three doses -> STOP
    cliff_edge = True
    for r in pilot_results:
        evil_ai_rate = r.get("summary", {}).get("evil_ai_strict_rate", 0.0)
        if evil_ai_rate >= 0.10:
            cliff_edge = False
            break

    if cliff_edge and len(pilot_results) == len(doses):
        print("\n*** GATE G_cliff TRIGGERED ***")
        print("evil_ai marker rate < 10% at ALL pilot doses (1, 3, 5 steps).")
        print("Marker destruction is cliff-edge. STOP.")
        return False
    else:
        print("\n*** GATE G_cliff PASSED ***")
        print("At least one pilot dose shows evil_ai marker rate >= 10%.")
        print("Proceed to Phase 1.")
        return True


def run_phase1(gpus: list[int]):
    """Phase 1: Full scan. {10, 25, 50, 100, 200, 375} x {EM, benign} + baseline."""
    doses = [10, 25, 50, 100, 200, 375]
    sft_types = ["em", "benign"]

    # Build cell list
    cell_specs = []
    # Baseline (no training)
    cell_specs.append({"max_steps": 0, "sft_type": "baseline", "seed": 42})
    for dose in doses:
        for sft_type in sft_types:
            cell_specs.append({"max_steps": dose, "sft_type": sft_type, "seed": 42})

    print(f"Phase 1: {len(cell_specs)} cells, {len(gpus)} GPUs")

    # Launch in batches of len(gpus)
    all_return_codes = {}
    for batch_start in range(0, len(cell_specs), len(gpus)):
        batch = cell_specs[batch_start : batch_start + len(gpus)]
        cells = []
        for i, spec in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            warmup = 0 if spec["max_steps"] <= 5 else None
            name, proc = launch_cell(
                max_steps=spec["max_steps"],
                sft_type=spec["sft_type"],
                seed=spec["seed"],
                gpu=gpu,
                personas="all_plus_evil_ai",
                warmup_steps=warmup,
            )
            cells.append((name, proc))
        batch_results = wait_for_cells(cells)
        all_return_codes.update(batch_results)

    # Collect and summarize
    all_results = collect_results()
    print_summary(all_results)

    # Gate G_partial: if no dose produces 10-80% survival -> STOP
    partial_found = False
    for r in all_results:
        if r.get("sft_type") == "baseline":
            continue
        evil_ai_rate = r.get("summary", {}).get("evil_ai_strict_rate", 0.0)
        if 0.10 <= evil_ai_rate <= 0.80:
            partial_found = True
            break

    if not partial_found:
        print("\n*** GATE G_partial: No dose in 10-80% survival range ***")
        print("Report curve shape, skip Phase 2.")
        return False, None

    # Gate G_gap: if |EM - benign| < 5pp at ALL doses -> H_null
    gap_found = False
    best_gap_dose = None
    best_gap = 0
    for dose in doses:
        em_results = [
            r for r in all_results if r.get("max_steps") == dose and r.get("sft_type") == "em"
        ]
        benign_results = [
            r for r in all_results if r.get("max_steps") == dose and r.get("sft_type") == "benign"
        ]
        if em_results and benign_results:
            em_rate = em_results[0]["summary"].get("evil_ai_strict_rate", 0.0)
            benign_rate = benign_results[0]["summary"].get("evil_ai_strict_rate", 0.0)
            gap = abs(em_rate - benign_rate)
            if gap >= 0.05:
                gap_found = True
            if gap > best_gap:
                best_gap = gap
                best_gap_dose = dose

    if not gap_found:
        print("\n*** GATE G_gap TRIGGERED ***")
        print("|EM - benign| < 5pp at ALL doses. H_null confirmed at single seed.")
        print("Skip Phase 2.")
        return False, None

    print(f"\n*** GATES PASSED *** Best gap dose: {best_gap_dose} (gap={100 * best_gap:.1f}pp)")
    return True, best_gap_dose


def run_phase2(gpus: list[int], critical_dose: int):
    """Phase 2: Multi-seed at critical dose. {42, 137, 256} x {EM, benign}."""
    seeds = [42, 137, 256]
    sft_types = ["em", "benign"]

    cell_specs = []
    for seed in seeds:
        for sft_type in sft_types:
            cell_specs.append({"max_steps": critical_dose, "sft_type": sft_type, "seed": seed})

    print(f"Phase 2: {len(cell_specs)} cells at dose={critical_dose}, {len(gpus)} GPUs")

    all_return_codes = {}
    for batch_start in range(0, len(cell_specs), len(gpus)):
        batch = cell_specs[batch_start : batch_start + len(gpus)]
        cells = []
        for i, spec in enumerate(batch):
            gpu = gpus[i % len(gpus)]
            name, proc = launch_cell(
                max_steps=spec["max_steps"],
                sft_type=spec["sft_type"],
                seed=spec["seed"],
                gpu=gpu,
                personas="all_plus_evil_ai",
            )
            cells.append((name, proc))
        batch_results = wait_for_cells(cells)
        all_return_codes.update(batch_results)

    all_results = collect_results()
    print_summary(all_results)

    return True


def main():
    parser = argparse.ArgumentParser(description="Dose-response orchestrator")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2],
        help="Phase to run (0=pilot, 1=full scan, 2=multi-seed)",
    )
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU indices (e.g., 4,5,6,7)")
    parser.add_argument(
        "--critical_dose",
        type=int,
        default=None,
        help="Critical dose for Phase 2 (required if --phase 2)",
    )
    parser.add_argument("--single", action="store_true", help="Run a single cell")
    parser.add_argument("--max_steps", type=int, help="For --single mode")
    parser.add_argument("--sft_type", help="For --single mode")
    parser.add_argument("--seed", type=int, default=42, help="For --single mode")
    parser.add_argument(
        "--summary", action="store_true", help="Just print summary of existing results"
    )
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    if args.summary:
        results = collect_results()
        print_summary(results)
        return 0

    if args.single:
        assert args.max_steps is not None and args.sft_type is not None
        warmup = 0 if args.max_steps <= 5 else None
        name, proc = launch_cell(
            max_steps=args.max_steps,
            sft_type=args.sft_type,
            seed=args.seed,
            gpu=gpus[0],
            personas="all_plus_evil_ai",
            warmup_steps=warmup,
        )
        results = wait_for_cells([(name, proc)])
        all_results = collect_results()
        print_summary(all_results)
        return 0

    if args.phase == 0:
        passed = run_phase0(gpus)
        return 0 if passed else 1

    elif args.phase == 1:
        passed, critical_dose = run_phase1(gpus)
        if critical_dose:
            print("\nSuggested Phase 2 command:")
            print(
                f"  python scripts/run_dose_response_orchestrator.py "
                f"--phase 2 --gpus {args.gpus} --critical_dose {critical_dose}"
            )
        return 0

    elif args.phase == 2:
        assert args.critical_dose is not None, "--critical_dose required for Phase 2"
        run_phase2(gpus, args.critical_dose)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
