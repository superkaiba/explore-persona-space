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
import datetime
import json
import logging
import os
import shlex
import subprocess
import sys
import time
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
    PHASE4_PATH,
    PHASE5_PATH,
    PHASE6_DIR,
    read_json,
)

logger = logging.getLogger("dispatch_jsdiv_470")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── poll_pipeline.py orchestration contract (.claude/agents/experiment-implementer.md §202-246) ──
#
# Pod-side dispatchers MUST emit lowercase ``[phase=<name>]`` log lines (parsed
# by ``PHASE_RE = re.compile(r"\[phase=([a-z_]+)")``) and a terminal
# ``[phase=done]`` immediately before the normal exit path, AFTER the end-of-run
# sentinel write. The sentinel lands at
# ``/workspace/logs/issue-470-epm_results-<epoch>.json`` and must carry the keys
# in ``poll_pipeline.py::_SENTINEL_REQUIRED_KEYS`` so the VM-side poll loop
# drains it and posts the carried ``epm:results`` marker on this task's
# events.jsonl. WITHOUT both halves a clean exit reads as ``dead`` and the
# headline result payload is silently skipped.
ISSUE_N = 470
SENTINEL_SCHEMA_VERSION = 1
SENTINEL_DIR = Path("/workspace/logs")


def _log_phase(name: str) -> None:
    """Emit a ``[phase=<name>]`` line to stdout, flushed so ``tail -500`` sees it.

    ``poll_pipeline.py::_latest_phase`` reads the most recent matching line
    from the tail of the pod-side log file (the launcher does
    ``nohup ... > log 2>&1``, so stdout IS the log). ``name`` must match
    ``[a-z_]+`` — digits or hyphens would silently miss the regex.
    """
    print(f"[phase={name}]", flush=True)


def _write_results_sentinel(*, smoke: bool, ran_full_pipeline: bool) -> Path | None:
    """Write the end-of-run ``epm:results`` sentinel for the VM poller.

    Returns the written path on success, ``None`` on failure (e.g. SENTINEL_DIR
    not writable — common when running the dispatcher locally on the VM rather
    than on the pod). Failure to write the sentinel is logged but is NOT fatal:
    a successful local smoke shouldn't fail merely because /workspace/logs is
    a pod-only path.

    The ``note`` body is a real ``epm:results`` marker payload pointing at the
    on-disk artifacts (Phase 4/5 JSONs + Phase 6 figures). When the pipeline
    completes in production, Phase 5's headline numbers are inlined so the
    marker is self-contained on the dashboard.
    """
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not create sentinel dir %s (%s); skipping sentinel write. "
            "This is expected when running off-pod.",
            SENTINEL_DIR,
            exc,
        )
        return None

    # Build the marker note body. Prefer concrete headline numbers from Phase 5
    # when available; otherwise point at the JSON paths so the analyzer (or
    # poll_pipeline reader) can pull them.
    headline: dict = {}
    if PHASE5_PATH.exists():
        try:
            reg = read_json(PHASE5_PATH)
            # Surface whatever the regression payload exposes at the top level
            # without making strong shape assumptions — Phase 5's internal keys
            # are not part of this dispatcher's contract.
            for key in (
                "pooled_partial_spearman",
                "paired_delta_rho",
                "verdict",
                "kill_criterion",
            ):
                if key in reg:
                    headline[key] = reg[key]
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s for sentinel headline (%s).", PHASE5_PATH, exc)

    # Resolve git commit + worktree for reproducibility.
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    eval_paths = {
        "regression": str(PHASE5_PATH.relative_to(REPO_ROOT)) if PHASE5_PATH.exists() else None,
        "predictor_comparison": str(PHASE4_PATH.relative_to(REPO_ROOT))
        if PHASE4_PATH.exists()
        else None,
        "figures_dir": str(PHASE6_DIR.relative_to(REPO_ROOT)) if PHASE6_DIR.exists() else None,
    }
    figure_files = (
        sorted(str(p.relative_to(REPO_ROOT)) for p in PHASE6_DIR.glob("*.png"))
        if PHASE6_DIR.exists()
        else []
    )

    note_lines = [
        f"task: #{ISSUE_N} — JS divergence vs cosine predictor re-analysis on #411 leakage",
        f"mode: {'smoke' if smoke else 'production'}",
        f"ran_full_pipeline: {ran_full_pipeline}",
        f"code_commit: {git_commit}",
        f"worktree: {REPO_ROOT}",
        "",
        "eval_paths:",
        *(f"  {k}: {v}" for k, v in eval_paths.items()),
    ]
    if figure_files:
        note_lines += ["", "figures:"] + [f"  - {p}" for p in figure_files]
    if headline:
        note_lines += ["", "headline:"]
        for k, v in headline.items():
            note_lines.append(f"  {k}: {v}")
    note_lines += [
        "",
        "Full payload is in the listed eval_paths; analyzer should ingest from there.",
    ]
    note = "\n".join(note_lines)

    sentinel = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "task_id": ISSUE_N,
        "kind": "epm:results",
        "version": 1,
        "gate": None,
        "blocks_pipeline": False,
        "note": note,
        "by": "dispatch_jsdiv_470",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    epoch = int(time.time())
    out_path = SENTINEL_DIR / f"issue-{ISSUE_N}-epm_results-{epoch}.json"
    # Atomic write so a mid-write crash never leaves a half-parsed sentinel.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(sentinel, indent=2))
    tmp.replace(out_path)
    logger.info("Wrote results sentinel to %s", out_path)
    return out_path


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


# Blocker #2: dispatcher-level "is this phase done" checks used to be filename-only,
# which let smoke artifacts skip a production launch. We now ALWAYS spawn each phase
# subprocess and let the phase's main() do the compatibility check + fast-exit when
# all of its outputs match the expected signature. The unused helpers below are
# kept as documentation of the OLD (filename-only) shortcut so future readers
# see exactly what was wrong.
#
# Old (buggy) shape — DO NOT REINTRODUCE:
#     def _phase1_done(personas):  return all((PHASE1_DIR / f"{p}.json").exists() ...)
#     def _phase2_done():          return (PHASE2_DIR / "cossim_pairs.json").exists()
#     def _phase3_done(sources, bystanders): ...  # filename existence only


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
    # Blocker #2: always spawn the subprocess; let phase1.main() check per-persona
    # metadata-compatibility and fast-exit if everything matches the signature.
    phases_to_run = {args.phase} if args.phase else {"1", "2", "3", "4", "5", "6"}
    if "1" in phases_to_run and not args.skip_phase1:
        _log_phase("sampling")
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
        _log_phase("cosine")
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
        _log_phase("js_kl")
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
        _log_phase("dv_load")
        p4_args = ["--sources", *sources]
        # `--smoke` (or any explicit bystander subset) deliberately trims to a
        # single (src, bys) cell — the Phase 4 prereq guard would otherwise
        # complain about the 22 unrun cells. Production NEVER sets this; the
        # dispatcher only forwards --allow-partial when the user explicitly
        # asked for a partial run via --smoke or --bystanders.
        if args.smoke or args.bystanders:
            p4_args.append("--allow-partial")
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase4_load_dv",
            p4_args,
            label="Phase 4 (DV load + assemble)",
        )
    # Concern #7: only require PHASE4_PATH when Phase 4 was actually scheduled.
    # ``--phase 6`` standalone should NOT crash here; it can read whatever Phase 4
    # left behind from a prior run (or fail when Phase 6 loads it, with a clearer
    # error than a generic "Phase 4 did not produce" message).
    if "4" in phases_to_run and not PHASE4_PATH.exists():
        raise RuntimeError(f"Phase 4 did not produce {PHASE4_PATH}; aborting.")

    if "5" in phases_to_run:
        _log_phase("regress")
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase5_regress",
            [],
            label="Phase 5 (regression)",
        )
    if not PHASE5_PATH.exists() and "5" in phases_to_run:
        raise RuntimeError(f"Phase 5 did not produce {PHASE5_PATH}; aborting.")

    if "6" in phases_to_run:
        _log_phase("figures")
        ph6_args = []
        # In smoke mode the hero figure is degenerate (1 source × 1 bystander),
        # but we keep it so the figure code is exercised — analyzer can ignore.
        _run_subprocess(
            "explore_persona_space.experiments.predictor_jsdiv_470.phase6_figures",
            ph6_args,
            label="Phase 6 (figures)",
        )

    # ── End-of-run sentinel + terminal phase=done line ──
    # poll_pipeline.py orchestration contract: ``ran_full_pipeline`` controls
    # whether the sentinel marks itself as a full vs partial dispatcher run.
    # The sentinel is informational either way — the poller drains it and posts
    # ``epm:results`` regardless of mode. ``[phase=done]`` MUST come AFTER the
    # sentinel write (poll_pipeline reads the log AFTER draining sentinels in
    # the same tick).
    ran_full_pipeline = (
        args.phase is None
        and not args.skip_phase1
        and not args.skip_phase2
        and not args.skip_phase3
    )
    _write_results_sentinel(smoke=args.smoke, ran_full_pipeline=ran_full_pipeline)
    _log_phase("done")
    logger.info("Dispatcher complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
