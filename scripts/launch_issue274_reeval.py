#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #274: Re-evaluate the 11 inherited #246/#232 LoRAs against the N=24 eval matrix.

#246's 12 LoRAs were eval'd against 11 or 12 personas, not 24. Re-fitting "the
N=12 fit at L15" on (OLD source rates × NEW N=24-centered cosines) is not
apples-to-apples (per plan §3e). The fix is a re-eval (no retraining): reuse
the existing LoRA adapters from HF Hub, run only the eval phase against
ALL_EVAL_PERSONAS_PLUS (24 entries). qwen_default is excluded — it has no
inherited adapter on Hub and is therefore trained+evaluated by the new-
conditions launcher (launch_issue274.py).

Pre-condition: merged adapters must already be on the pod's local disk via
    python scripts/pod.py sync models --pull
This script does NOT pull adapters — it only kicks off `--eval-only` runs that
read from local merged-checkpoint paths.

Usage (on the pod):
    nohup uv run python scripts/launch_issue274_reeval.py \\
        --pod epm-issue-274 --n-gpus 8 \\
        > eval_results/leakage_experiment/i274_reeval_launcher.log 2>&1 &
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "eval_results" / "leakage_experiment"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 11 inherited #246/#232 sources: 10 named PERSONAS + helpful_assistant.
# helpful_assistant is included because the source_rate=null bug fix needs to apply
# to it on the re-eval. qwen_default is NOT in this list — it has no #246-era LoRA
# adapter on Hub, so it must be trained-and-eval'd (not eval-only'd) by the new-
# conditions launcher (launch_issue274.py SOURCES). Including it here would waste
# ~12 min wall and overwrite the run_result.json that the new launcher just wrote.
SOURCES = (
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
    "helpful_assistant",
)


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the per-condition --eval-only command (as a bash string).

    --eval-only (existing flag at scripts/archive/run_leakage_experiment.py L1035)
    skips training+merge and loads the existing merged model from the local
    output dir. The N=24 ALL_EVAL_PERSONAS_PLUS matrix kicks in automatically
    when args.source is in SOURCES_REQUIRING_PLUS_EVAL (which is the case for
    qwen_default and the 12 #274 sources). For the named PERSONAS + helpful_assistant
    we still want the 24-persona symmetric eval — the source-resolution logic
    inside run_leakage_experiment.py uses ALL_EVAL_PERSONAS_PLUS whenever we hit
    this code path because we set --neg-set asst_excluded which is the recipe
    locked by #246/#274 — but for the NAMED personas the legacy ALL_EVAL_PERSONAS
    (11 entries) is used. We force the N=24 matrix for these re-evals via an
    environment variable that run_leakage_experiment.py reads at runtime.
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED=42 "
        f"EPM_FORCE_EVAL_PERSONAS_PLUS=1 "
        f".venv/bin/python scripts/archive/run_leakage_experiment.py "
        f"--trait marker --source {source} --neg-set asst_excluded "
        f"--prompt-length medium --seed 42 --gpu {gpu} "
        f"--pod {pod} --phase a1 --eval-only"
    )


def launch_wave(wave_idx: int, conditions: list[tuple[str, int]], pod: str) -> list:
    """Launch one wave (each (source, gpu) tuple in parallel) and return Popen handles."""
    procs = []
    print(f"\n=== Wave {wave_idx + 1}: launching {len(conditions)} re-evals ===", flush=True)
    for source, gpu in conditions:
        log_file = LOG_DIR / f"i274_reeval_{source}_asst_excluded_seed42_gpu{gpu}.log"
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
    parser = argparse.ArgumentParser(
        description="Issue #274 re-eval launcher (11 inherited LoRAs against N=24 matrix)"
    )
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

    waves = []
    for wave_start in range(0, n_sources, n_gpus):
        wave_sources = SOURCES[wave_start : wave_start + n_gpus]
        wave = [(src, gi) for gi, src in enumerate(wave_sources)]
        waves.append(wave)

    print(
        f"#274 re-eval launcher: {n_sources} sources × seed 42 across {n_gpus} GPUs "
        f"= {len(waves)} wave(s)",
        flush=True,
    )
    for wi, wave in enumerate(waves):
        print(f"  wave {wi + 1}: {[s for s, _ in wave]}", flush=True)
    print(
        "\nPre-condition: merged adapters must be on local disk. "
        "Run `python scripts/pod.py sync models --pull` from the local VM first.",
        flush=True,
    )

    for wave_idx, wave in enumerate(waves):
        procs = launch_wave(wave_idx, wave, args.pod)
        wait_wave(procs, wave_idx)

    print("\n=== All re-eval waves complete ===", flush=True)
    print(f"Logs: {LOG_DIR}/i274_reeval_*.log", flush=True)


if __name__ == "__main__":
    sys.exit(main())
