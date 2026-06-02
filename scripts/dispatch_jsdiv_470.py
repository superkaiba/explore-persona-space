#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Dispatcher for task #470 — JS divergence vs cosine predictor re-analysis.

Sequentially runs Phase 1 → 6 with subprocess isolation between Phase 1 (vLLM)
and Phases 2-3 (HF Transformers) to avoid the #399 vLLM worker-teardown trap.

Checkpoint-per-phase resume: each phase's runtime is gated on file presence;
re-invoking the dispatcher after a partial run picks up where it left off.

Smoke / sweep parity: identical code path; smoke mode is just smaller N. The
defaults run the full sweep on all 6 #411 sources × all 23 bystanders × all
50 probes × R=8.

Examples::

    # SMOKE (~5 min on 1× H100; ~minutes on CPU): one source × one bystander,
    # 5 probes, R=2.
    uv run python scripts/dispatch_jsdiv_470.py --smoke

    # PRODUCTION (full sweep, ~5 h on 1× H100):
    uv run python scripts/dispatch_jsdiv_470.py

    # Specific subset:
    uv run python scripts/dispatch_jsdiv_470.py --sources software_engineer comedian \\
        --probes 50 --R 8
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv at the top of __main__ would be too late for subprocesses we spawn
# (their os.environ.copy() reflects ours), so we do it here at module top.
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.experiments.predictor_jsdiv_470 import (  # noqa: E402
    SOURCE_PERSONAS_411,
)
from explore_persona_space.experiments.predictor_jsdiv_470.common import (  # noqa: E402
    DEFAULT_R,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    PHASE4_PATH,
    PHASE5_PATH,
)

logger = logging.getLogger("dispatch_jsdiv_470")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _run_subprocess(module: str, args: list[str], *, label: str) -> None:
    """Run ``uv run python -m <module> <args>`` with explicit env passthrough.

    Subprocess isolation is the #399 vLLM teardown mitigation: Phase 1 (vLLM)
    must NEVER run in the same Python process as Phases 2-3 (HF Transformers).
    """
    cmd = ["uv", "run", "python", "-m", module, *args]
    env = {**os.environ}  # explicit copy per CLAUDE.md subprocess-env-passthrough
    logger.info("[%s] launching: %s", label, " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Phase {label} subprocess exited with code {result.returncode}. "
            f"Command: {' '.join(shlex.quote(c) for c in cmd)}"
        )


def _personas_to_sample(sources: list[str], bystanders: list[str] | None) -> list[str]:
    """All personas needed for Phase 1: each source + each bystander used as
    the "sample-from" side in Phase 3's two-side RB estimator.

    Default bystander list = the full 24-panel minus the source, computed by
    Phase 3. We just dump the 24 personas at full-sweep time to keep Phase 1
    self-contained.
    """
    from explore_persona_space.experiments.predictor_jsdiv_470.common import (
        get_eval_personas_24,
    )

    personas = set(sources)
    if bystanders:
        personas.update(bystanders)
    else:
        personas.update(get_eval_personas_24().keys())
    return sorted(personas)


def _phase1_done(personas: list[str]) -> bool:
    return all((PHASE1_DIR / f"{p}.json").exists() for p in personas)


def _phase2_done() -> bool:
    return (PHASE2_DIR / "cossim_pairs.json").exists()


def _phase3_done(sources: list[str], bystanders: list[str] | None) -> bool:
    from explore_persona_space.experiments.predictor_jsdiv_470.common import (
        get_eval_personas_24,
    )

    panel = list(get_eval_personas_24().keys())
    for src in sources:
        bys_list = bystanders or [p for p in panel if p != src]
        for bys in bys_list:
            if not (PHASE3_DIR / f"{src}__{bys}.json").exists():
                return False
    return True


def main() -> int:  # noqa: C901 — linear sequence of phase-launch checks; splitting hurts readability
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_PERSONAS_411),
        help="Sources to score (default: all 6 #411 sources).",
    )
    parser.add_argument(
        "--bystanders",
        nargs="+",
        default=None,
        help="Bystanders to score (default: panel minus source).",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=None,
        help="Cap to first N probes (smoke mode). Default: all 50.",
    )
    parser.add_argument("--R", type=int, default=DEFAULT_R, dest="r")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for --sources software_engineer --bystanders comedian --probes 5 --R 2.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-phase1", action="store_true", help="Resume: skip Phase 1.")
    parser.add_argument("--skip-phase2", action="store_true", help="Resume: skip Phase 2.")
    parser.add_argument("--skip-phase3", action="store_true", help="Resume: skip Phase 3.")
    parser.add_argument(
        "--phase", choices=["1", "2", "3", "4", "5", "6"], help="Run ONE phase only (debug)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model path (default: Qwen/Qwen2.5-7B-Instruct). Smoke "
        "may override to Qwen/Qwen2.5-0.5B-Instruct on CPU.",
    )
    parser.add_argument(
        "--use-hf-fallback",
        action="store_true",
        help="Phase 1 uses HF model.generate() instead of vLLM (CPU smoke only). "
        "Production always uses vLLM per CLAUDE.md.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Phase 2 layer set (default: {7,14,21,27} per plan §10). Override "
        "for smoke on smaller models (e.g. 0.5B has 24 layers).",
    )
    args = parser.parse_args()

    if args.smoke:
        args.sources = ["software_engineer"]
        args.bystanders = ["comedian"]
        args.probes = 5
        args.r = 2
        logger.info("SMOKE MODE: src=software_engineer, bys=comedian, probes=5, R=2")

    sources = list(args.sources)
    bystanders = list(args.bystanders) if args.bystanders else None
    personas_phase1 = _personas_to_sample(sources, bystanders)

    common_args = []
    if args.probes is not None:
        common_args = ["--probes", str(args.probes)]

    # ── Phase 1: vLLM sampling (subprocess-isolated) ──
    phases_to_run = {args.phase} if args.phase else {"1", "2", "3", "4", "5", "6"}
    if "1" in phases_to_run and not args.skip_phase1:
        if _phase1_done(personas_phase1):
            logger.info(
                "Phase 1 outputs already exist for all %d personas; skipping.", len(personas_phase1)
            )
        else:
            p1_args = [
                *common_args,
                "--R",
                str(args.r),
                "--personas",
                *personas_phase1,
            ]
            if args.use_hf_fallback:
                p1_args.append("--use-hf-fallback")
            if args.model:
                p1_args.extend(["--model", args.model])
            _run_subprocess(
                "explore_persona_space.experiments.predictor_jsdiv_470.phase1_sample_responses",
                p1_args,
                label="Phase 1 (sampling)",
            )

    # ── Phase 2: HF Transformers cosine recipe (b) (subprocess-isolated from Phase 1) ──
    if "2" in phases_to_run and not args.skip_phase2:
        if _phase2_done():
            logger.info("Phase 2 outputs already exist; skipping.")
        else:
            p2_args = ["--personas", *personas_phase1, "--gpu-id", str(args.gpu_id)]
            if args.model:
                p2_args.extend(["--model", args.model])
            if args.layers:
                p2_args.extend(["--layers", *[str(li) for li in args.layers]])
            _run_subprocess(
                "explore_persona_space.experiments.predictor_jsdiv_470.phase2_cosine_response_token",
                p2_args,
                label="Phase 2 (response-token cosine)",
            )

    # ── Phase 3: HF Transformers RB JS + KL (same process as Phase 2 is fine; both HF) ──
    if "3" in phases_to_run and not args.skip_phase3:
        if _phase3_done(sources, bystanders):
            logger.info("Phase 3 outputs already exist for all cells; skipping.")
        else:
            ph3_args = [
                *common_args,
                "--sources",
                *sources,
                "--gpu-id",
                str(args.gpu_id),
            ]
            if bystanders:
                ph3_args.extend(["--bystanders", *bystanders])
            if args.model:
                ph3_args.extend(["--model", args.model])
            _run_subprocess(
                "explore_persona_space.experiments.predictor_jsdiv_470.phase3_sequence_js_kl",
                ph3_args,
                label="Phase 3 (RB sequence JS + KL)",
            )

    # ── Phases 4-6 (CPU): can run in-process; no GPU contention ──
    if "4" in phases_to_run:
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase4_load_dv",
            ["--sources", *sources],
            label="Phase 4 (DV load + assemble)",
        )
    if not PHASE4_PATH.exists():
        raise RuntimeError(f"Phase 4 did not produce {PHASE4_PATH}; aborting.")

    if "5" in phases_to_run:
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase5_regress",
            [],
            label="Phase 5 (regression)",
        )
    if not PHASE5_PATH.exists() and "5" in phases_to_run:
        raise RuntimeError(f"Phase 5 did not produce {PHASE5_PATH}; aborting.")

    if "6" in phases_to_run:
        ph6_args = []
        # In smoke mode the hero figure is degenerate (1 source × 1 bystander),
        # but we keep it so the figure code is exercised — analyzer can ignore.
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase6_figures",
            ph6_args,
            label="Phase 6 (figures)",
        )

    logger.info("Dispatcher complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
