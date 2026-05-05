#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #274: Launch marker LoRA training+eval for the 13 new sources.

Iterates over the 12 new persona sources (chef, lawyer, accountant, journalist,
wizard, hero, philosopher, child, ai_assistant, ai, chatbot, i_am_helpful)
plus qwen_default (also missing from main; was only on the issue-246 branch),
sharding across `--n-gpus` GPUs in waves. Each subprocess trains a LoRA adapter
on 600 contrastive marker examples and evaluates marker/structure/capability/
alignment across the 24-persona ALL_EVAL_PERSONAS_PLUS matrix.

Wave model: launch up to N concurrent subprocesses, wait for the wave to finish
(proc.wait()) before launching the next wave. This avoids GPU contention while
preserving sweep parallelism.

Usage (on a pod):
    nohup uv run python scripts/launch_issue274.py \\
        --pod epm-issue-274 --n-gpus 8 \\
        > eval_results/leakage_experiment/i274_launcher.log 2>&1 &
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 13 new training conditions for #274 (12 NEW_PERSONA_PROMPTS_274 + qwen_default).
# Order: occupational, character, generic_helper, framing-axis probe, qwen_default.
SOURCES = (
    # Occupational (4)
    "chef",
    "lawyer",
    "accountant",
    "journalist",
    # Character (4)
    "wizard",
    "hero",
    "philosopher",
    "child",
    # Generic helper (3)
    "ai_assistant",
    "ai",
    "chatbot",
    # First-person framing probe (1)
    "i_am_helpful",
    # Generic helper from #246 (also missing on main)
    "qwen_default",
)


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the per-condition train+eval command (as a bash string)."""
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "
        f".venv/bin/python scripts/archive/run_leakage_experiment.py "
        f"--trait marker --source {source} --neg-set asst_excluded "
        f"--prompt-length medium --seed 42 --gpu {gpu} "
        f"--pod {pod} --phase a1"
    )


def launch_wave(wave_idx: int, conditions: list[tuple[str, int]], pod: str) -> list:
    """Launch one wave (each (source, gpu) tuple in parallel) and return Popen handles."""
    procs = []
    print(f"\n=== Wave {wave_idx + 1}: launching {len(conditions)} conditions ===", flush=True)
    for source, gpu in conditions:
        log_file = LOG_DIR / f"i274_marker_{source}_asst_excluded_seed42_gpu{gpu}.log"
        cmd = build_cmd(source, gpu, pod)
        print(f"[gpu{gpu}] source={source}")
        print(f"[gpu{gpu}] cmd: {cmd}")
        print(f"[gpu{gpu}] log: {log_file}", flush=True)
        # Use shell so the env-var prefix on the LHS works.
        proc = subprocess.Popen(
            ["bash", "-c", f"{cmd} > {log_file} 2>&1"],
            cwd=str(ROOT),
        )
        procs.append((source, gpu, proc, log_file))
    return procs


def wait_wave(procs: list, wave_idx: int) -> None:
    """Block until every Popen in the wave has finished. Report exit codes."""
    for source, gpu, proc, log_file in procs:
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"[wave{wave_idx + 1}/gpu{gpu}] {source}: {status} (log: {log_file})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Issue #274 wave-based launcher")
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-274",
        help="Pod identifier (passed to run_leakage_experiment.py --pod for logging)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="Number of GPUs to shard across (default: 8)",
    )
    args = parser.parse_args()

    n_gpus = max(1, args.n_gpus)
    n_sources = len(SOURCES)

    # Partition SOURCES into waves of size n_gpus.
    waves = []
    for wave_start in range(0, n_sources, n_gpus):
        wave_sources = SOURCES[wave_start : wave_start + n_gpus]
        wave = [(src, gi) for gi, src in enumerate(wave_sources)]
        waves.append(wave)

    print(
        f"#274 launcher: {n_sources} sources × seed 42 across {n_gpus} GPUs = {len(waves)} wave(s)",
        flush=True,
    )
    for wi, wave in enumerate(waves):
        print(f"  wave {wi + 1}: {[s for s, _ in wave]}", flush=True)

    for wave_idx, wave in enumerate(waves):
        procs = launch_wave(wave_idx, wave, args.pod)
        wait_wave(procs, wave_idx)

    print("\n=== All waves complete ===", flush=True)
    print(f"Logs: {LOG_DIR}/i274_marker_*.log", flush=True)


if __name__ == "__main__":
    sys.exit(main())
