#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #296: Launch marker LoRA training+eval for the 24 new sources.

Iterates over the 24 NEW_PERSONA_PROMPTS_296 sources (10 occupational + 8 character
+ 6 generic_helper), sharding across `--n-gpus` GPUs in waves. Each subprocess trains
a LoRA adapter on 600 contrastive marker examples and evaluates marker rates across
the N=48 ALL_EVAL_PERSONAS_PLUS matrix (extended from #274's N=24 in this issue).

Wave model: launch up to N concurrent subprocesses, wait for the wave to finish
(proc.wait()) before launching the next wave. This avoids GPU contention while
preserving sweep parallelism.

Resume-safe: skips a source whose run_result.json already exists with a populated
results.marker.source_rate (so a partial run can be resumed by re-running the
launcher; previously-finished conditions are not redone).

Usage (on a pod):
    nohup uv run python scripts/launch_issue296.py \\
        --pod epm-issue-296 --n-gpus 8 \\
        > eval_results/leakage_experiment/i296_launcher.log 2>&1 &
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 24 new training conditions for #296. Order: occupational, character, generic_helper
# (matches NEW_PERSONA_PROMPTS_296 in scripts/generate_leakage_data.py).
SOURCES = (
    # Occupational (10)
    "pilot",
    "nurse",
    "pharmacist",
    "professor",
    "scientist",
    "biologist",
    "engineer",
    "architect",
    "banker",
    "firefighter",
    # Character (8)
    "pirate",
    "knight",
    "princess",
    "robot",
    "ghost",
    "hacker",
    "detective",
    "witch",
    # Generic helper (6)
    "virtual_assistant",
    "ai_tool",
    "smart_helper",
    "chat_assistant",
    "reasoning_ai",
    "friendly_ai",
)

assert len(SOURCES) == 24, f"Expected 24 sources, got {len(SOURCES)}"


def _is_already_done(source: str) -> bool:
    """Resume-safe guard: skip a source if its run_result.json already has a populated rate."""
    rr_path = (
        ROOT
        / "eval_results"
        / "leakage_experiment"
        / f"marker_{source}_asst_excluded_medium_seed42"
        / "run_result.json"
    )
    if not rr_path.exists():
        return False
    try:
        with open(rr_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    marker = data.get("results", {}).get("marker", {})
    sr = marker.get("source_rate")
    # Accept either populated source_rate, or all_personas[<source>] populated as fallback.
    if sr is not None:
        return True
    all_p = marker.get("all_personas", {}) or {}
    return source in all_p and all_p[source] is not None


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the per-condition train+eval command (as a bash string).

    Note: unlike #274, the ALL_EVAL_PERSONAS_PLUS dict is now N=48, so the eval matrix
    is automatic — no EPM_FORCE_EVAL_PERSONAS_PLUS env-var needed for new sources
    (they are already in SOURCES_REQUIRING_PLUS_EVAL).
    """
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
        log_file = LOG_DIR / f"i296_marker_{source}_asst_excluded_seed42_gpu{gpu}.log"
        cmd = build_cmd(source, gpu, pod)
        print(f"[gpu{gpu}] source={source}")
        print(f"[gpu{gpu}] cmd: {cmd}")
        print(f"[gpu{gpu}] log: {log_file}", flush=True)
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
    parser = argparse.ArgumentParser(description="Issue #296 wave-based launcher (24 new sources)")
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-296",
        help="Pod identifier (passed to run_leakage_experiment.py --pod for logging)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="Number of GPUs to shard across (default: 8)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help=(
            "Disable the resume-safe skip-if-done guard. By default, sources whose "
            "run_result.json already has a populated source_rate are skipped."
        ),
    )
    args = parser.parse_args()

    n_gpus = max(1, args.n_gpus)

    # Resume-safe filter
    if args.no_skip:
        pending = list(SOURCES)
        print(
            "--no-skip: launching all 24 sources regardless of existing run_result.json", flush=True
        )
    else:
        pending = [s for s in SOURCES if not _is_already_done(s)]
        skipped = [s for s in SOURCES if s not in pending]
        if skipped:
            print(f"Skipping {len(skipped)} already-finished sources: {skipped}", flush=True)
        else:
            print("No previously-finished sources detected — running all 24.", flush=True)

    if not pending:
        print("\n=== All 24 sources already complete; nothing to launch. ===", flush=True)
        return 0

    # Partition pending into waves of size n_gpus.
    waves = []
    for wave_start in range(0, len(pending), n_gpus):
        wave_sources = pending[wave_start : wave_start + n_gpus]
        wave = [(src, gi) for gi, src in enumerate(wave_sources)]
        waves.append(wave)

    print(
        f"#296 launcher: {len(pending)} pending sources × seed 42 across {n_gpus} GPUs "
        f"= {len(waves)} wave(s)",
        flush=True,
    )
    for wi, wave in enumerate(waves):
        print(f"  wave {wi + 1}: {[s for s, _ in wave]}", flush=True)

    for wave_idx, wave in enumerate(waves):
        procs = launch_wave(wave_idx, wave, args.pod)
        wait_wave(procs, wave_idx)

    print("\n=== All waves complete ===", flush=True)
    print(f"Logs: {LOG_DIR}/i296_marker_*.log", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
