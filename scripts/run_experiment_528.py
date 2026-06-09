"""Top-level pod-side driver for issue #528 (plan v1 §4.7).

Orchestrates the full 1-GPU pipeline as a single Python entrypoint (per the
``experiment-implementer.md`` "Pod-side dispatcher / poll_pipeline.py
contract"): emits ``[phase=...]`` log lines for ``scripts/poll_pipeline.py``
on every phase transition and a terminal ``[phase=done]`` on graceful
completion, then writes an end-of-run sentinel under
``/workspace/logs/issue-528-epm_results-<ts>.json`` with the
``_SENTINEL_REQUIRED_KEYS`` (``sentinel_schema_version``, ``kind``,
``version``) that ``poll_pipeline.py`` parses.

Phases (matching ``scripts/i528_run_all_1gpu.sh`` so the bash script remains a
valid fallback launcher):

    Phase 0  preflight + per-trait Q-bank (Phase 0a + 0b)
    Phase 0' codepath verification (role-token + rubric)
    Phase 1  R_pos + R_neg generation
    Phase 2  smoke train + judge gate (single canary cell:
              trait=validating arm=role seed=42 epochs=1 train_slice=6)
    Phase 3  full 24-cell sweep (4 traits x 2 arms x 3 seeds), sequential
    Phase 4  base eval + trained eval + judge
    Phase 5  analyze + plots

NO shell-out to ``scripts/task.py`` from this script (CLAUDE.md
"Pod-side code NEVER shells out to ``scripts/task.py``"); the orchestrator on
the VM reads the sentinel and posts ``epm:results`` itself.

Per-cell training is dispatched as a subprocess of
``scripts/i528_phase23_train.py`` (one process per LoRA, sequential on a
1-GPU pod) so each cell exits cleanly and frees GPU memory + vLLM workers
before the next one starts; the subprocess env is passed explicitly via
``env=os.environ.copy()`` per the implementer "subprocess env passthrough"
rule.

CLI:

    # Production launch (pod-side):
    nohup uv run python scripts/run_experiment_528.py \
        > /workspace/logs/issue-528-run.log 2>&1 &

    # VM-side wiring smoke (no GPU; defers training to pod):
    uv run python scripts/run_experiment_528.py --smoke

    # Resume a specific phase:
    uv run python scripts/run_experiment_528.py --phase analyze

    # Single training cell (used by the dispatcher itself; rarely called by hand):
    uv run python scripts/run_experiment_528.py \
        --phase train --trait validating --arm role --seed 42
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("i528.driver")

# Plan v1 §4.7 cell list. 4 traits x 2 arms x 3 seeds = 24 cells.
TRAITS: tuple[str, ...] = (
    "validating",
    "conciseness",
    "asks_clarifying_first",
    "calibrated_uncertainty",
)
ARMS: tuple[str, ...] = ("system", "role")
SEEDS: tuple[int, ...] = (42, 137, 1337)

# Smoke canary cell (plan v1 §13). Smoke IS sweep with one cell — same
# subprocess shape, same env injection, same logging surface — so
# PASS_UNIFIED per the smoke-architecture check.
SMOKE_TRAIT = "validating"
SMOKE_ARM = "role"
SMOKE_SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR_DEFAULT = Path("/workspace/logs")
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_528"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

PHASES = (
    "preflight",
    "q-bank",
    "r-gen",
    "smoke-train",
    "smoke-judge",
    "train",
    "eval-base",
    "eval-trained",
    "judge",
    "analyze",
    "all",
)


def _emit_phase(name: str) -> None:
    """Emit a ``[phase=...]`` line that ``poll_pipeline.py`` parses."""
    ts = _dt.datetime.utcnow().isoformat() + "Z"
    print(f"[phase={name}] {ts}", flush=True)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def _run(cmd: list[str], *, env: dict[str, str], log_path: Path | None = None) -> int:
    """Run a subprocess, streaming output to stdout (and optionally a log file)."""
    cmd_str = " ".join(cmd)
    logger.info("RUN %s", cmd_str)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        logger.error("FAILED rc=%d: %s", proc.returncode, cmd_str)
    return proc.returncode


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_env() -> dict[str, str]:
    """Build the subprocess env explicitly (implementer subprocess-env-passthrough rule).

    Sets the upload-policy + MooseFS-quota guards before any training subprocess.
    """
    env = os.environ.copy()
    # CLAUDE.md upload-policy + the per-task adapter-persist contract (#404/#458).
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    env.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HF_MODEL_REPO)
    # Each cell gets its own subfolder; the training script overrides this
    # per cell, but we set a sensible default in case it isn't.
    return env


def phase_preflight(env: dict[str, str], smoke: bool) -> None:
    _emit_phase("preflight")
    args = ["uv", "run", "python", "scripts/i528_phase0_preflight.py"]
    if smoke:
        args.append("--smoke")
    rc = _run(args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 0 preflight failed (rc={rc})")


def phase_codepath_verify(env: dict[str, str]) -> None:
    _emit_phase("codepath_verify")
    rc = _run(["uv", "run", "python", "scripts/i528_phase0_codepath_verify.py"], env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 0' codepath verify failed (rc={rc})")


def phase_r_gen(env: dict[str, str], smoke: bool) -> None:
    _emit_phase("r_pos")
    pos_args = ["uv", "run", "python", "scripts/i528_phase1_generate_RPos.py"]
    if smoke:
        pos_args.append("--smoke")
    rc = _run(pos_args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 1 R_pos failed (rc={rc})")
    _emit_phase("r_neg")
    neg_args = ["uv", "run", "python", "scripts/i528_phase1_generate_RNeg.py"]
    if smoke:
        neg_args.append("--smoke")
    rc = _run(neg_args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 1 R_neg failed (rc={rc})")


def phase_smoke_train(env: dict[str, str], gpu_id: int) -> None:
    """Phase 2 smoke: the canary cell (plan v1 §13). PASS_UNIFIED architecture
    — smoke IS sweep with a single cell, same subprocess shape, same env."""
    _emit_phase("phase2_smoke_train")
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/i528_phase23_train.py",
            "--trait",
            SMOKE_TRAIT,
            "--arm",
            SMOKE_ARM,
            "--seed",
            str(SMOKE_SEED),
            "--smoke",
            "--gpu-id",
            str(gpu_id),
        ],
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 2 smoke train failed (rc={rc})")


def phase_smoke_judge(env: dict[str, str]) -> None:
    _emit_phase("phase2_smoke_judge")
    adapter = f"adapters/i528_{SMOKE_TRAIT}_{SMOKE_ARM}_seed{SMOKE_SEED}_smoke"
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/i528_phase2_smoke_judge.py",
            "--adapter",
            adapter,
            "--trait",
            SMOKE_TRAIT,
            "--arm",
            SMOKE_ARM,
            "--threshold",
            "3.0",
        ],
        env=env,
    )
    if rc != 0:
        raise RuntimeError(f"Phase 2 smoke judge below threshold (rc={rc}) — abort sweep")


def phase_train_cell(
    env: dict[str, str], trait: str, arm: str, seed: int, gpu_id: int, log_dir: Path
) -> None:
    """Train one (trait, arm, seed) cell as a subprocess.

    Per CLAUDE.md "Checkpoint per phase": each cell uploads its adapter to HF
    via the train script's HF-upload code path BEFORE this function returns,
    so a downstream cell crash never loses the artifact.
    """
    cell_id = f"{trait}_{arm}_seed{seed}"
    _emit_phase(f"phase3_cell_{cell_id}")
    rc = _run(
        [
            "uv",
            "run",
            "python",
            "scripts/i528_phase23_train.py",
            "--trait",
            trait,
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--gpu-id",
            str(gpu_id),
        ],
        env=env,
        log_path=log_dir / f"i528_{cell_id}.log",
    )
    if rc != 0:
        raise RuntimeError(f"Phase 3 cell {cell_id} failed (rc={rc})")


def phase_train_sweep(env: dict[str, str], gpu_id: int, log_dir: Path) -> None:
    _emit_phase("phase3_sweep")
    for trait in TRAITS:
        for arm in ARMS:
            for seed in SEEDS:
                phase_train_cell(env, trait, arm, seed, gpu_id, log_dir)


def phase_eval_base(env: dict[str, str], smoke: bool) -> None:
    _emit_phase("phase4_eval_base")
    args = ["uv", "run", "python", "scripts/i528_phase4_eval_base.py"]
    if smoke:
        args.append("--smoke")
    rc = _run(args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 4 base eval failed (rc={rc})")


def phase_eval_trained(env: dict[str, str], smoke: bool) -> None:
    _emit_phase("phase4_eval_trained")
    args = ["uv", "run", "python", "scripts/i528_phase4_eval.py"]
    if smoke:
        args.extend(["--n-q", "3"])
    rc = _run(args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 4 trained eval failed (rc={rc})")


def phase_judge(env: dict[str, str], smoke: bool) -> None:
    _emit_phase("phase4_judge")
    args = ["uv", "run", "python", "scripts/i528_phase4_judge.py"]
    if smoke:
        args.extend(["--limit", "20"])
    rc = _run(args, env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 4 judge failed (rc={rc})")


def phase_analyze(env: dict[str, str]) -> None:
    _emit_phase("phase5_analyze")
    rc = _run(["uv", "run", "python", "scripts/i528_phase5_analyze.py"], env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 5 analyze failed (rc={rc})")
    _emit_phase("phase5_plot")
    rc = _run(["uv", "run", "python", "scripts/plot_i528_clean_result.py"], env=env)
    if rc != 0:
        raise RuntimeError(f"Phase 5 plot failed (rc={rc})")


def write_sentinel(log_dir: Path, *, smoke: bool, t0: float) -> Path:
    """Write the end-of-run sentinel that ``poll_pipeline.py`` parses.

    Required keys per ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS``:
    ``sentinel_schema_version`` (int=1), ``kind`` (str), ``version`` (int).
    Plus optional fields the orchestrator surfaces as ``epm:results``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_paths = [
        EVAL_RESULTS_DIR / "judge_scores.json",
        EVAL_RESULTS_DIR / "analysis.json",
        EVAL_RESULTS_DIR / "paraphrase_replication.json",
        EVAL_RESULTS_DIR / "base_headroom_judge.json",
    ]
    eval_paths_present = [p for p in eval_paths if p.is_file()]
    rel = lambda p: str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)  # noqa: E731
    sha256_by_path = {rel(p): _file_sha256(p) for p in eval_paths}
    gpu_hours_used = max(0.0, (time.time() - t0) / 3600.0)

    sentinel = {
        "sentinel_schema_version": 1,
        "kind": "epm:results",
        "version": 1,
        "task_id": 528,
        "by": "run_experiment_528",
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "phase": "done",
        "smoke": smoke,
        "eval_paths": [rel(p) for p in eval_paths_present],
        "eval_path_sha256": sha256_by_path,
        "reproducibility_card": {
            "git_commit_sha": _git_sha(),
            "hf_model_repo": HF_MODEL_REPO,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "n_traits": len(TRAITS),
            "n_arms": len(ARMS),
            "n_seeds": len(SEEDS),
            "n_cells": len(TRAITS) * len(ARMS) * len(SEEDS),
        },
        "gpu_hours_used": round(gpu_hours_used, 3),
        "plan_deviations": [],
        "note": "i528 run complete; analysis + plots produced.",
    }
    ts = int(time.time())
    sentinel_path = log_dir / f"issue-528-epm_results-{ts}.json"
    sentinel_path.write_text(json.dumps(sentinel, indent=2))
    logger.info("[sentinel] wrote %s", sentinel_path)
    return sentinel_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Run a single phase (default: 'all').",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: tiny slices, no Sonnet API calls, single canary cell, "
        "skip the 24-cell sweep + heavy eval. Wiring check only.",
    )
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU index for the training subprocess (default: 0).",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help="Directory for per-cell logs + the end-of-run sentinel.",
    )
    # Single-cell escape hatch (used by the dispatcher itself, not by hand).
    ap.add_argument(
        "--trait",
        choices=TRAITS,
        default=None,
        help="With --phase train: train just this trait.",
    )
    ap.add_argument(
        "--arm",
        choices=ARMS,
        default=None,
        help="With --phase train: train just this arm.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="With --phase train: train just this seed.",
    )
    ap.add_argument(
        "--no-sentinel",
        action="store_true",
        help="Skip writing the end-of-run sentinel (single-phase resumes).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = parse_args(argv)
    env = _make_env()
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "i528 driver start phase=%s smoke=%s gpu=%d git=%s",
        args.phase,
        args.smoke,
        args.gpu_id,
        _git_sha()[:12],
    )

    phase = args.phase

    if phase == "preflight":
        phase_preflight(env, smoke=args.smoke)
    elif phase == "q-bank":
        # Phase 0 + codepath_verify produce the per-trait Q-banks.
        phase_preflight(env, smoke=args.smoke)
        phase_codepath_verify(env)
    elif phase == "r-gen":
        phase_r_gen(env, smoke=args.smoke)
    elif phase == "smoke-train":
        phase_smoke_train(env, gpu_id=args.gpu_id)
    elif phase == "smoke-judge":
        phase_smoke_judge(env)
    elif phase == "train":
        # Either a single cell (when --trait/--arm/--seed given) or the full sweep.
        if args.trait and args.arm and args.seed is not None:
            phase_train_cell(env, args.trait, args.arm, args.seed, args.gpu_id, log_dir)
        else:
            phase_train_sweep(env, gpu_id=args.gpu_id, log_dir=log_dir)
    elif phase == "eval-base":
        phase_eval_base(env, smoke=args.smoke)
    elif phase == "eval-trained":
        phase_eval_trained(env, smoke=args.smoke)
    elif phase == "judge":
        phase_judge(env, smoke=args.smoke)
    elif phase == "analyze":
        phase_analyze(env)
    elif phase == "all":
        # Full pipeline in plan order.
        phase_preflight(env, smoke=args.smoke)
        phase_codepath_verify(env)
        phase_r_gen(env, smoke=args.smoke)
        phase_smoke_train(env, gpu_id=args.gpu_id)
        phase_smoke_judge(env)
        if args.smoke:
            # Smoke stops here — sweep + heavy eval are deferred to the
            # production run on a GPU pod (plan v1 §13: smoke IS sweep with a
            # single cell + 1 epoch + 6-Q slice).
            logger.info("SMOKE complete; skipping sweep + eval + analyze.")
        else:
            phase_train_sweep(env, gpu_id=args.gpu_id, log_dir=log_dir)
            phase_eval_base(env, smoke=False)
            phase_eval_trained(env, smoke=False)
            phase_judge(env, smoke=False)
            phase_analyze(env)
    else:  # pragma: no cover — argparse choices block other values
        raise ValueError(f"Unknown phase {phase!r}")

    if not args.no_sentinel:
        write_sentinel(log_dir, smoke=args.smoke, t0=t0)

    _emit_phase("done")
    logger.info("i528 driver done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
