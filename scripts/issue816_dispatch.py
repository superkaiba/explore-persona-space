#!/usr/bin/env python
"""Issue #816 8-GPU fan-out dispatcher (Phase A) — the UNIFIED smoke = sweep driver.

Builds ONE work queue of ``(experiment, trait, cell)`` subprocess commands across
the Phase-A GPU work (Phase-0 probe, Exp-2 steering, Exp-4 preventative, Exp-5
screening capture) and fans it out ``wave_size = visible-GPU-count``-wide, each
cell a ``subprocess.Popen`` pinned to one physical GPU via BOTH
``CUDA_VISIBLE_DEVICES=<gpu>`` in the launcher env AND the matching ``--gpu-id``
arg (the mandatory dual-pin — the in-process CVD clobber alone is defeated by any
import-time cuInit; gotchas.md #545).

SMOKE = SWEEP with ``--cells 1``: the SAME dispatcher runs the 1-cell smoke and
the full sweep; ``--cells N`` threads through to EVERY per-cell entrypoint (probe,
steering, preventative, screening) so smoke IS the sweep with one cell per phase —
same dispatcher, same subprocess shape, same CVD pin, same per-cell upload, same
sentinel path. Verdict: PASS_UNIFIED.

Each entrypoint checkpoints its own per-cell JSON + adapter the moment the cell
completes, so per-cell UPLOAD (via ``issue816_dispatch.sh``) never accumulates a
terminal batch. Pod-side: this dispatcher NEVER shells ``scripts/task.py`` — it
writes only stdout ``[phase=...]`` breadcrumbs; the sentinel + upload are the
``.sh`` wrapper's job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib
import issue816_lib as ilib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue816.dispatch")
load_dotenv()

SCRIPTS = Path(__file__).resolve().parent
PHASES = ("probe", "steering", "preventative", "screening")


def _compute_wave_size(cpu_only: bool, requested: int | None) -> int:
    """Wave size = VISIBLE GPU count (memory: wave-size-must-match-visible-gpus).

    Raises loud on 0 visible GPU when not cpu_only (a wave of 0 is the silent-CPU
    crash class). ``requested`` is a CEILING, never the source of truth.
    """
    if cpu_only:
        return 1
    import torch

    detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if detected == 0:
        raise RuntimeError(
            "no visible CUDA device for the Phase-A fan-out; refusing to fan out on CPU "
            "(pass --cpu-only for a deliberate CPU smoke)"
        )
    ceiling = max(requested, 1) if requested else detected
    return min(detected, ceiling)


def _py(script: str) -> list[str]:
    return ["uv", "run", "python", str(SCRIPTS / script)]


def _cells(args) -> list[str]:
    return ["--cells", str(args.cells)] if args.cells is not None else []


def _steering_flags(args) -> list[str]:
    """Flags common to the steering entrypoint (probe + steer phases)."""
    c = [
        "--external-root",
        args.external_root,
        "--out-root",
        args.out_root,
        "--cache-dir",
        args.cache_dir,
        "--model",
        args.model,
    ]
    if args.n_questions is not None:
        c += ["--n-questions", str(args.n_questions)]
    if args.n_rollouts is not None:
        c += ["--n-rollouts", str(args.n_rollouts)]
    if args.cpu_only:
        c += ["--cpu-only"]
    return c


def _probe_cmds(args) -> list[dict]:
    cmd = [*_py("issue816_steering.py"), "--phase", "probe", *_steering_flags(args), *_cells(args)]
    return [{"label": "probe", "cmd": cmd, "wants_gpu_id": False}]


def _steering_cmds(args) -> list[dict]:
    out = []
    for trait in args.traits:
        cmd = [
            *_py("issue816_steering.py"),
            "--phase",
            "steer",
            "--traits",
            trait,
            *_steering_flags(args),
            *_cells(args),
        ]
        out.append({"label": f"steering_{trait}", "cmd": cmd, "wants_gpu_id": False})
    return out


def _preventative_cmds(args) -> list[dict]:
    out = []
    for trait in args.traits:
        cmd = [
            *_py("issue816_preventative.py"),
            "--traits",
            trait,
            "--dataset-root",
            args.dataset_root,
            "--ckpt-root",
            args.ckpt_root,
            "--external-root",
            args.external_root,
            "--out-root",
            args.out_root,
            "--cache-dir",
            args.cache_dir,
            "--model",
            args.model,
            *_cells(args),
        ]
        if args.n_questions is not None:
            cmd += ["--n-questions", str(args.n_questions)]
        if args.n_rollouts is not None:
            cmd += ["--n-rollouts", str(args.n_rollouts)]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.normalize:
            cmd.append("--normalize")
        out.append({"label": f"preventative_{trait}", "cmd": cmd, "wants_gpu_id": True})
    return out


def _screening_cmds(args) -> list[dict]:
    cmd = [
        *_py("issue816_screening.py"),
        "--traits",
        *list(args.traits),
        "--dataset-root",
        args.dataset_root,
        "--n-samples",
        str(args.n_samples),
        "--out-root",
        args.out_root,
        "--cache-dir",
        args.cache_dir,
        "--model",
        args.model,
        *_cells(args),
    ]
    if args.cpu_only:
        cmd += ["--cpu-only"]
    return [{"label": "screening", "cmd": cmd, "wants_gpu_id": False}]


def _build_commands(args) -> list[dict]:
    """Build the work-queue entries (``{"label","cmd","wants_gpu_id"}``) per phase.

    Flags are threaded PER-ENTRYPOINT (the three entrypoints have DIFFERENT
    argparse surfaces -- e.g. only ``preventative`` accepts ``--gpu-id`` /
    ``--max-steps``; only ``steering`` / ``screening`` accept ``--cpu-only``;
    ``screening`` has no ``--n-questions``), so a shared blob would crash argparse
    on an entrypoint that does not define a flag. ``--cells`` is threaded to EVERY
    entrypoint (smoke = sweep with --cells 1). Only ``preventative`` gets
    ``--gpu-id`` (its ``train_lora`` clobbers CVD in-process from ``cfg.gpu_id``,
    so the arg must match the launcher CVD pin); steering/screening rely on the
    CVD env pin alone (they never re-clobber it).
    """
    builders = {
        "probe": _probe_cmds,
        "steering": _steering_cmds,
        "preventative": _preventative_cmds,
        "screening": _screening_cmds,
    }
    cmds: list[dict] = []
    for phase in PHASES:  # deterministic order
        if phase in args.phases:
            cmds.extend(builders[phase](args))
    return cmds


def _run_waves(cmds: list[dict], *, wave_size: int, dry_run: bool) -> dict:
    """Fan out the work queue in waves of ``wave_size``, CVD-pinned per position."""
    lib.log_phase("dispatch", f"{len(cmds)} cells, wave_size={wave_size}")
    results: dict[str, str] = {}
    for start in range(0, len(cmds), wave_size):
        wave = cmds[start : start + wave_size]
        procs = []
        for i, entry in enumerate(wave):
            gpu_id = i  # position within the wave -> physical GPU i (post-CVD-pin cuda:0)
            label = entry["label"]
            full = list(entry["cmd"])
            if entry["wants_gpu_id"]:
                full += ["--gpu-id", str(gpu_id)]
            # BOTH pins: CVD in the launcher env + the matching --gpu-id where the
            # entrypoint honors it (train_lora clobbers CVD in-process from gpu_id).
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            logger.info("[wave] launch %s CUDA_VISIBLE_DEVICES=%d", label, gpu_id)
            if dry_run:
                logger.info("[dry-run] would exec: %s", " ".join(full))
                results[label] = "dry-run"
                continue
            procs.append((label, subprocess.Popen(full, env=env)))
        for label, p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Phase-A cell {label} exited rc={rc}")
            results[label] = "done"
            logger.info("Phase-A cell %s complete", label)  # NOT [phase=done] (reserved)
    lib.log_phase("dispatch", f"all Phase-A cells done ({len(results)})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #816 Phase-A 8-GPU fan-out dispatcher.")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(PHASES),
        choices=list(PHASES),
        help="which Phase-A phases to run (default: all)",
    )
    parser.add_argument("--traits", nargs="+", default=list(ilib.TRAITS))
    parser.add_argument("--dataset-root", default="external/persona_vectors/dataset")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="eval_results/issue_816")
    parser.add_argument("--ckpt-root", default="checkpoints/issue_816")
    parser.add_argument("--cache-dir", default="data/issue_816/hf_dl")
    parser.add_argument(
        "--n-gpus", type=int, default=None, help="wave-size CEILING (default: detected)"
    )
    parser.add_argument("--n-random-dirs", type=int, default=10, help="Exp-4 random dirs per trait")
    parser.add_argument("--n-samples", type=int, default=500, help="Exp-5 samples/dataset")
    parser.add_argument(
        "--cells", type=int, default=None, help="limit each phase to first N cells (SMOKE)"
    )
    parser.add_argument("--n-questions", type=int, default=None, help="cap eval questions (smoke)")
    parser.add_argument("--n-rollouts", type=int, default=None, help="override rollouts (smoke)")
    parser.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke)")
    parser.add_argument(
        "--normalize", action="store_true", help="normalize preventative vec (default RAW)"
    )
    parser.add_argument("--cpu-only", action="store_true", help="deliberate CPU smoke")
    parser.add_argument("--model", default=ilib.MODEL_NAME)
    parser.add_argument("--dry-run", action="store_true", help="preview fan-out, no CUDA")
    args = parser.parse_args()

    cmds = _build_commands(args)

    if args.dry_run:
        wave_size = max(args.n_gpus, 1) if args.n_gpus else 8
    else:
        wave_size = _compute_wave_size(args.cpu_only, args.n_gpus)
    res = _run_waves(cmds, wave_size=wave_size, dry_run=args.dry_run)
    print(json.dumps({"phase": "dispatch", "phases": args.phases, "cells": res}, indent=2))


if __name__ == "__main__":
    main()
