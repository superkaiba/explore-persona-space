#!/usr/bin/env python3
"""Launch Phase A1 with chained runs per GPU across multiple pods.

Creates a bash script per GPU that chains 2-3 runs sequentially, then launches them.
Each run: train (~3 min) + eval (~170 min) = ~3h.

Usage:
    python scripts/launch_phase_a1_chained.py --dry-run  # Just print assignments
    python scripts/launch_phase_a1_chained.py --launch    # Actually launch
"""

import argparse
import os
import sys

SOURCES = [
    "software_engineer",
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
    "zelthari_scholar",
]

TRAITS = ["marker", "capability"]
NEG_SETS = ["asst_excluded", "asst_included"]

SEED = 42
PHASE = "a1"

# Pod config: (ssh_target, available_gpus)
PODS = {
    "pod2": ("root@103.207.149.64 -p 16193", [2, 3, 4, 5, 6, 7]),
    "pod3": ("root@69.30.85.155 -p 22184", [0, 1, 2, 3, 4, 5, 6, 7]),
    "pod4": ("root@103.207.149.58 -p 15920", [0, 1, 2, 3, 4, 5]),
}


def build_run_list():
    """Build list of all 44 Phase A1 runs."""
    runs = []
    # Standard: 10 sources × 2 neg-sets × 2 traits = 40
    for trait in TRAITS:
        for source in SOURCES:
            for neg_set in NEG_SETS:
                runs.append(
                    {
                        "trait": trait,
                        "source": source,
                        "neg_set": neg_set,
                        "name": f"{trait}_{source}_{neg_set}",
                    }
                )
    # Controls: 4
    for trait in TRAITS:
        for ctrl in ["generic_sft", "shuffled_persona"]:
            runs.append(
                {
                    "trait": trait,
                    "control": ctrl,
                    "name": f"{trait}_{ctrl}",
                }
            )
    return runs


def assign_runs_to_gpus(runs, pods):
    """Round-robin assign runs to GPU slots across all pods."""
    # Flatten GPU slots
    slots = []
    for pod_name, (ssh, gpus) in pods.items():
        for gpu in gpus:
            slots.append((pod_name, ssh, gpu))

    # Assign round-robin
    assignments = {(pod_name, gpu): [] for pod_name, ssh, gpu in slots}
    for i, run in enumerate(runs):
        slot_idx = i % len(slots)
        pod_name, ssh, gpu = slots[slot_idx]
        assignments[(pod_name, gpu)].append(run)

    return assignments, slots


def make_run_cmd(run, gpu, pod_name):
    """Create the Python command for one run."""
    if "control" in run:
        return (
            f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 "
            f".venv/bin/python scripts/run_leakage_experiment.py "
            f"--trait {run['trait']} --control {run['control']} "
            f"--seed {SEED} --gpu {gpu} --pod {pod_name} --phase {PHASE}"
        )
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 "
        f".venv/bin/python scripts/run_leakage_experiment.py "
        f"--trait {run['trait']} --source {run['source']} "
        f"--neg-set {run['neg_set']} --prompt-length medium "
        f"--seed {SEED} --gpu {gpu} --pod {pod_name} --phase {PHASE}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Print assignments without launching"
    )
    parser.add_argument("--launch", action="store_true", help="Actually launch on pods")
    parser.add_argument("--pods", nargs="+", default=list(PODS.keys()), help="Which pods to use")
    args = parser.parse_args()

    if not args.dry_run and not args.launch:
        print("Specify --dry-run or --launch")
        sys.exit(1)

    runs = build_run_list()
    print(f"Total Phase A1 runs: {len(runs)}")

    # Filter to requested pods
    active_pods = {k: v for k, v in PODS.items() if k in args.pods}
    assignments, slots = assign_runs_to_gpus(runs, active_pods)

    total_gpus = len(slots)
    print(f"Available GPU slots: {total_gpus}")
    print(f"Runs per GPU: {len(runs) / total_gpus:.1f} avg")
    print()

    # Group by pod for launching
    by_pod = {}
    for (pod_name, gpu), pod_runs in assignments.items():
        if pod_runs:
            by_pod.setdefault(pod_name, {})[gpu] = pod_runs

    for pod_name in sorted(by_pod.keys()):
        ssh_target = active_pods[pod_name][0]
        print(f"=== {pod_name} ({ssh_target}) ===")
        gpu_runs = by_pod[pod_name]

        for gpu in sorted(gpu_runs.keys()):
            gpu_run_list = gpu_runs[gpu]
            print(f"  GPU {gpu}: {len(gpu_run_list)} runs")
            for r in gpu_run_list:
                print(f"    - {r['name']}")

            if args.launch:
                # Build chained command: run1 && run2 && run3
                chain_parts = []
                for r in gpu_run_list:
                    cmd = make_run_cmd(r, gpu, pod_name)
                    log = f"eval_results/leakage_experiment/a1_{r['name']}_gpu{gpu}.log"
                    chain_parts.append(f"({cmd}) > {log} 2>&1")

                chain_cmd = " && ".join(chain_parts)
                screen_name = f"a1_g{gpu}"

                # Launch via SSH
                full_ssh_cmd = (
                    f"ssh {ssh_target} "
                    f"'cd /workspace/explore-persona-space && "
                    f'screen -dmS {screen_name} bash -c "{chain_cmd}"\''
                )
                print(f"    Launching: screen {screen_name}")
                os.system(full_ssh_cmd)

        print()

    if args.launch:
        print("All runs launched! Monitor with:")
        for pod_name in sorted(by_pod.keys()):
            ssh_target = active_pods[pod_name][0]
            print(f"  ssh {ssh_target} 'screen -ls | grep a1_'")


if __name__ == "__main__":
    main()
