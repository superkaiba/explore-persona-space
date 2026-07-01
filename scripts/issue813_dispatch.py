#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →) in scientific docstrings + log messages.
"""Issue #813 — extraction-wave dispatcher (8-way CVD fan-out over the 12 cells).

Fans out ``issue813_run_cell.py`` over the ``(behavior × substrate)`` cells,
``wave_size = min(len(cells), visible_gpus)`` (feedback
``dispatcher_wave_size_must_match_visible_gpus``, #667 a36): a wave larger than
the visible count would spawn ``--gpu-id`` lanes whose ``CUDA_VISIBLE_DEVICES``
points at a non-existent device — those processes see NO GPU and SILENTLY fall
back to CPU. Each per-cell subprocess PINS ``CUDA_VISIBLE_DEVICES=<gpu>`` in the
LAUNCHER env matching its ``--gpu-id`` (the #545 launcher-env pin an import-time
cuInit cannot defeat) — reference shape ``scripts/i474_phase23_dispatch.sh:192``.

Subprocess env passthrough: every subprocess gets an EXPLICIT ``env=`` (a
deliberate copy of ``os.environ`` with the per-lane CVD override), and this
module calls ``load_dotenv()`` at import so a fresh dispatcher process spawns
subprocesses with the credential env present (``uv run python`` does NOT
auto-load ``.env``; #397 round-10').

This is the EXTRACT phase only; ``issue813_dispatch.sh`` sequences it after the
apply-parity probe + one-cell gate and before the off-pod fit + analysis.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue813.dispatch")

BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")


def _visible_gpu_count() -> int:
    """CUDA devices visible to THIS process (honours CUDA_VISIBLE_DEVICES). 0 if none.

    Verbatim contract from ``issue667_alllayer_dispatch._visible_gpu_count``:
    ``torch.cuda.device_count()`` reflects the CVD-filtered list, so it is the
    authoritative visible count.
    """
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # torch import failure / no driver — treat as 0 visible
        return 0


def compute_wave_size(cpu_only: bool, requested_n_gpus: int, *, dry_run: bool = False) -> int:
    """Parallel wave = DETECTED visible-GPU count, clamped by --n-gpus (a CEILING).

    Contract (feedback dispatcher_wave_size_must_match_visible_gpus, #667 a36),
    identical to ``issue667_alllayer_dispatch.compute_wave_size``:

    - ``--cpu-only`` → 1 (serial; CPU has no per-device sharding constraint).
    - ``--dry-run`` (GPU-less VM preview) → the REQUESTED ceiling.
    - GPU run → ``min(detected, max(requested_n_gpus, 1))``; a wave larger than the
      visible count spawns ``--gpu-id`` lanes that see NO device and silently fall
      back to CPU (the #667 a36 hang), so the wave NEVER exceeds the detected count.
    - GPU run with 0 visible devices → RAISE LOUD (never a silent CPU fallback).
    """
    if cpu_only:
        return 1
    if dry_run:
        return max(requested_n_gpus, 1)
    detected = _visible_gpu_count()
    if detected == 0:
        raise RuntimeError(
            "no CUDA devices visible (torch.cuda.device_count()==0) but --cpu-only "
            "was not set — refusing to spawn a wave that would silently fall back to "
            "CPU (feedback dispatcher_wave_size_must_match_visible_gpus, #667 a36). "
            "Pass --cpu-only for a deliberate CPU run, or launch on a GPU pod."
        )
    n = min(detected, max(requested_n_gpus, 1))
    if n < max(requested_n_gpus, 1):
        logger.warning(
            "wave clamped to %d (detected %d visible GPUs) below the --n-gpus ceiling %d",
            n,
            detected,
            requested_n_gpus,
        )
    logger.info(
        "wave size = %d (detected %d visible GPUs, --n-gpus ceiling %d)",
        n,
        detected,
        requested_n_gpus,
    )
    return n


def enumerate_cells(behaviors: list[str], substrates: list[str]) -> list[tuple[str, str]]:
    """The (behavior, substrate) extraction cells — the SAME grid every phase reads."""
    return [(b, s) for b in behaviors for s in substrates]


def _cell_cmd(
    behavior: str,
    substrate: str,
    gpu_id: int,
    *,
    out_root: str,
    cpu_only: bool,
    upload: bool,
    max_contexts: int | None,
    max_questions: int | None,
) -> tuple[list[str], dict]:
    """Build the per-cell ``issue813_run_cell.py`` command + its CVD-pinned launcher env.

    The launcher env pins ``CUDA_VISIBLE_DEVICES=<gpu_id>`` (the #545 pin) AND the
    command passes the matching ``--gpu-id <gpu_id>`` — both are required (the
    in-process ``cuda:0`` clobber is defeated by any import-time cuInit unless CVD
    is pinned in the launcher env). Returns ``(cmd, env)``; the env is an EXPLICIT
    copy of ``os.environ`` (subprocess-env-explicit contract).
    """
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/issue813_run_cell.py",
        "--behavior",
        behavior,
        "--substrate",
        substrate,
        "--out-root",
        out_root,
        "--gpu-id",
        str(gpu_id),
    ]
    if cpu_only:
        cmd.append("--cpu-only")
    if upload:
        cmd.append("--upload")
    if max_contexts is not None:
        cmd += ["--max-contexts", str(max_contexts)]
    if max_questions is not None:
        cmd += ["--max-questions", str(max_questions)]
    env = {**os.environ}  # EXPLICIT env passthrough (subprocess-env-explicit)
    if not cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # #545 launcher-env pin
    return cmd, env


def run_wave(args) -> int:
    """Run the extraction wave over all cells, wave_size lanes at a time."""
    cells = enumerate_cells(args.behaviors, args.substrates)
    wave_size = compute_wave_size(args.cpu_only, args.n_gpus, dry_run=args.dry_run)
    logger.info(
        "[phase=extract] %d cells, wave_size=%d, upload=%s", len(cells), wave_size, args.upload
    )

    if args.dry_run:
        for i, (b, s) in enumerate(cells):
            gpu = i % wave_size
            cmd, env = _cell_cmd(
                b,
                s,
                gpu,
                out_root=str(args.out_root),
                cpu_only=args.cpu_only,
                upload=args.upload,
                max_contexts=args.max_contexts,
                max_questions=args.max_questions,
            )
            logger.info(
                "[dry-run] cell %s/%s → gpu %d | CVD=%s | %s",
                b,
                s,
                gpu,
                env.get("CUDA_VISIBLE_DEVICES", "<none>"),
                " ".join(cmd),
            )
        return 0

    failures: list[tuple[str, str, int]] = []
    for wave_start in range(0, len(cells), wave_size):
        wave = cells[wave_start : wave_start + wave_size]
        procs: list[tuple[str, str, subprocess.Popen]] = []
        for lane, (b, s) in enumerate(wave):
            gpu = lane  # lane index == physical GPU (wave_size ≤ visible count)
            cmd, env = _cell_cmd(
                b,
                s,
                gpu,
                out_root=str(args.out_root),
                cpu_only=args.cpu_only,
                upload=args.upload,
                max_contexts=args.max_contexts,
                max_questions=args.max_questions,
            )
            logger.info(
                "[phase=extract] launch %s/%s on gpu %d (CVD=%s)",
                b,
                s,
                gpu,
                env.get("CUDA_VISIBLE_DEVICES", "<none>"),
            )
            procs.append((b, s, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)))
        for b, s, p in procs:
            rc = p.wait()
            if rc != 0:
                logger.error("[phase=extract] cell %s/%s FAILED rc=%d", b, s, rc)
                failures.append((b, s, rc))
            else:
                logger.info(
                    "extract cell %s/%s complete", b, s
                )  # NOT [phase=done] (poller-reserved)

    if failures:
        logger.error("[phase=extract] %d/%d cells FAILED: %s", len(failures), len(cells), failures)
        return 1
    logger.info("[phase=extract] all %d cells complete", len(cells))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #813 extraction-wave dispatcher (CVD fan-out)")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=8,
        help="CEILING on the parallel wave (not the source of truth)",
    )
    ap.add_argument("--cpu-only", action="store_true", help="serial CPU smoke")
    ap.add_argument(
        "--upload", action="store_true", help="stream unreduced+reduced .npz to HF per cell"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="preview the wave plan + CVD pins, touch no CUDA"
    )
    ap.add_argument("--max-contexts", type=int, default=None, help="smoke: cap battery contexts")
    ap.add_argument(
        "--max-questions", type=int, default=None, help="smoke: cap substrate questions"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()
    rc = run_wave(build_parser().parse_args())
    logger.info("[phase=extract] wave finished in %.1fs rc=%d", time.time() - t0, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
