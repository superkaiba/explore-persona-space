"""Unified in-process dispatcher for issue #524 — smoke == sweep with --cells 1.

Issue #524 plan v4 §4 "Smoke-architecture parity (UNIFIED — the default)":

    Dispatcher scripts/issue524_dispatch.py runs in-process serial with
    --cells 1 --seeds 1 --phase {0,1,2,3} for smoke and --cells 32
    --seeds 1 --phase all for sweep. Both modes call the same
    train_one_icl_cell(cid, ...) and eval_cell(src_cid, tgt_cid, ...)
    functions; no subprocess wrapper. Smoke runs ONE ICL cell end-to-end
    (train -> cross-eval against 4 representative targets at 100 Q ->
    extract clouds -> predictor -> metric) using the exact code path the
    full sweep uses. Task #397 round-11 unification pattern.

Per-cell Q-count is the SAME in smoke and sweep (100 Q) — smoke verifies
the wall-time per cell at production Q-count, not at a reduced one, so
the Phase 2 budget projection is calibratable from the smoke result.

This dispatcher is the canonical entrypoint; the per-phase scripts
(``scripts/issue524_phase{0_*,1,2,3,4,5}_*.py``) are wrapped CLIs over
the same underlying functions for human debugging and ad-hoc
re-invocation. The dispatcher loops in-process and emits per-phase
sentinels to ``/workspace/logs/issue-524-*.json`` so
``scripts/poll_pipeline.py`` can drain progress and post markers via the
orchestrator. PER CLAUDE.md "Pod-side code NEVER shells out to
scripts/task.py" — the dispatcher writes sentinels and JSON lines ONLY.

CLI:
    # SMOKE (single ICL cell, the WHOLE pipeline end-to-end at 100 Q).
    uv run python scripts/issue524_dispatch.py --cells 1 --seeds 1 --phase all

    # SWEEP (full 16 ICL cells × all phases).
    uv run python scripts/issue524_dispatch.py --cells 32 --seeds 1 --phase all

    # Selectively re-run one phase.
    uv run python scripts/issue524_dispatch.py --cells 32 --phase 4

    # CPU-only smoke (skips train + eval + extraction; runs Phase 4 + 5
    # against pre-existing artifacts).
    uv run python scripts/issue524_dispatch.py --cpu-only --phase 4

The dispatcher's exit code:
  0  — pipeline reached --phase target without an unrecoverable error.
  2  — preflight failed (missing artifact, missing token, etc).
  3  — phase ran but emitted a non-zero result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# epm-lint: workflow-fix-on-bug -- module-top dotenv load. The dispatcher
# spawns subprocesses (lint-style equivalence: we IMPORT the phase modules
# and call their main() functions in-process here, NOT via subprocess.run,
# but we still rely on credential env vars being live before HF Hub or
# Anthropic API helpers are constructed downstream).
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("i524.dispatch")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the scripts/ directory importable so we can call sibling phase
# modules by name (matches scripts/issue502_dispatch.py pattern).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Pod-side default is /workspace/logs (where poll_pipeline.py drains
# sentinels). On the local VM, /workspace isn't writable; fall back to a
# /tmp path so the smoke can run end-to-end. The experimenter overrides via
# EPM_LOGS_DIR=/workspace/logs at pod-side dispatch.
_DEFAULT_LOGS_DIR = "/workspace/logs"
if not Path("/workspace").is_dir() or not os.access("/workspace", os.W_OK):
    _DEFAULT_LOGS_DIR = "/tmp/issue-524-logs"
LOGS_DIR = Path(os.environ.get("EPM_LOGS_DIR", _DEFAULT_LOGS_DIR))
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results" / "issue_524"

# Phase names mirror the plan §4 H4 structure. ``0`` is the Phase 0 bundle:
# pool_eval_100 + ICL blocks + induction check + floor diagnosis. We expose
# them as discrete sub-phases for selective rerun, with "all" = the full
# chain.
PHASE_NAMES = ["0", "1", "2", "3", "4", "5", "all"]


def _git_sha() -> str:
    """Short HEAD SHA or 'unknown' on error (reproducibility metadata)."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _emit_phase_log(phase: str) -> None:
    """Emit the ``[phase=<name>]`` line that ``poll_pipeline.py`` parses.

    This is the contract documented in CLAUDE.md "Pod-side result-reporting
    contract": the poller's PHASE_RE = ``re.compile(r"\\[phase=([a-z_]+)")``
    tails the log and the final ``[phase=done]`` line determines whether
    the orchestrator reads the run as ``status=done`` or ``status=dead``.
    """
    print(f"[phase={phase}]", flush=True)


def _write_sentinel(kind: str, payload: dict[str, Any]) -> Path:
    """Write a results sentinel JSON for ``poll_pipeline.py`` to drain.

    File path: ``/workspace/logs/issue-524-<kind_slug>-<epoch>.json``.
    Required keys per CLAUDE.md "End-of-run sentinel": ``sentinel_schema_version=1``,
    ``kind`` (full marker kind string), ``version`` (marker version integer).
    Marker body goes under ``note``.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    epoch = int(time.time())
    out = LOGS_DIR / f"issue-524-{kind_slug}-{epoch}.json"
    sentinel = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": 1,
        "task_id": 524,
        "by": "issue524_dispatch.py",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(payload, indent=2),
    }
    out.write_text(json.dumps(sentinel, indent=2) + "\n")
    return out


# --------------------------------------------------------------------------
# Per-phase phase functions. Each one is a thin in-process call to the
# corresponding scripts/issue524_phaseN_*.py module's ``main()``. We
# IMPORT rather than ``subprocess.run`` so the smoke and sweep code paths
# are bit-identical (the "no subprocess wrapper, no in-process-vs-
# subprocess divergence" line from plan §"Smoke architecture parity").
# --------------------------------------------------------------------------


def run_phase_0(args: argparse.Namespace) -> int:
    """Phase 0 — pool_eval_100 build + ICL blocks + induction gate + floor diagnosis.

    Sub-steps (sequential, persisted per-step):
      0.0  build pool_eval_100.json (deterministic 100-of-500 subset)
      0.1  floor diagnosis (#489 ICL adapter analysis; CPU + ≤1 GPU-h)
      0.2  build 16 rebuilt ICL demonstration blocks via Haiku (in --cells 1
            smoke mode this restricts to one context)
      0.3  ICL induction-rate manipulation check (Sonnet-judged ≥70%)

    Returns 0 on success, 3 on a phase failure (e.g. induction gate failed).
    """
    _emit_phase_log("phase_0")
    logger.info("Phase 0 (cells=%d, seeds=%d)", args.cells, args.seeds)

    # 0.0 — pool_eval_100.
    from issue524_build_pool_eval_100 import (
        main as build_pool_main,
    )

    rc = build_pool_main([])
    if rc != 0:
        logger.error("Phase 0.0 (pool_eval_100 build) failed rc=%d", rc)
        return 3

    if args.cpu_only:
        logger.info("--cpu-only set; skipping Phase 0.1/0.2/0.3 (GPU + API calls).")
        _write_sentinel(
            "epm:phase-0-complete",
            {"sub_steps_run": ["0.0"], "cpu_only": True, "cells": args.cells},
        )
        return 0

    if args.skip_phase_0_subs:
        logger.info("--skip-phase-0-subs set; skipping Phase 0.1/0.2/0.3 (assumed already done).")
        _write_sentinel(
            "epm:phase-0-complete",
            {"sub_steps_run": ["0.0"], "skipped_subs": True, "cells": args.cells},
        )
        return 0

    # 0.1 — floor diagnosis. Cheap; runs even in smoke. (No CLI flags;
    # the script always runs against its hard-coded #489 inputs.)
    diag_args: list[str] = []
    from issue524_phase0_1_floor_diagnosis import (
        main as floor_diag_main,
    )

    try:
        rc = floor_diag_main(diag_args)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.warning("Phase 0.1 (floor diagnosis) returned rc=%d; not blocking", rc)
        # 0.1 is diagnostic, not gating — proceed.

    # 0.2 — ICL block build. Spans ~$0.50 of Haiku in sweep; in --cells 1 we
    # restrict to one context for the smoke run.
    block_args: list[str] = []
    if args.cells == 1:
        block_args.extend(["--only", args.smoke_cid])
    from issue524_phase0_2_build_icl_blocks import (
        main as build_blocks_main,
    )

    try:
        rc = build_blocks_main(block_args)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 0.2 (ICL block build) failed rc=%d", rc)
        return 3

    # 0.3 — induction-rate gate. Sonnet judges 50 on-policy generations
    # per context; the gate is ≥12/16 with ≥70% induction.
    ind_args: list[str] = []
    if args.cells == 1:
        ind_args.extend(["--only", args.smoke_cid])
    from issue524_phase0_3_induction_check import (
        main as induction_main,
    )

    try:
        rc = induction_main(ind_args)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        # In sweep mode, a failed gate (rc=3) is the load-bearing escalation
        # per plan §0.3. In smoke mode (--cells 1), a single-cell failure is
        # informational (the smoke probe context may be one of the typically-
        # failing ones); don't propagate.
        if args.cells == 1:
            logger.warning(
                "Phase 0.3 (induction) rc=%d on smoke cell %s; informational only in --cells 1",
                rc,
                args.smoke_cid,
            )
        else:
            logger.error(
                "Phase 0.3 (induction-rate gate) FAILED rc=%d — escalate per plan §0.3",
                rc,
            )
            return 3

    _write_sentinel(
        "epm:phase-0-complete",
        {"sub_steps_run": ["0.0", "0.1", "0.2", "0.3"], "cells": args.cells},
    )
    return 0


def run_phase_1(args: argparse.Namespace) -> int:
    """Phase 1 — train ICL marker adapters via in-process train_one_icl_cell calls."""
    _emit_phase_log("phase_1")
    if args.cpu_only:
        logger.info("--cpu-only set; skipping Phase 1 (GPU training).")
        return 0
    logger.info("Phase 1 training (cells=%d)", args.cells)

    # We call the Phase 1 script's main() which itself iterates over its
    # --conds argument (nargs+). For --cells 1 we pass a single cid; for
    # --cells 32 we let the script's default loop over the full ICL panel
    # (16 contexts, sharded by GPU id elsewhere).
    cli: list[str] = []
    if args.cells == 1:
        cli.extend(["--conds", args.smoke_cid, "--seed", str(args.seed)])
    else:
        # Sweep: pass every ICL cid (the dispatcher's "in-process" sweep).
        # The script's --conds is nargs+, so we list them.
        from explore_persona_space.experiments.i524_icl_contexts import ICL_CONTEXTS

        cli.extend(["--conds", *[c.cid for c in ICL_CONTEXTS], "--seed", str(args.seed)])
    # Else: default = full 16-cell sweep within this process. (The per-GPU
    # sharding for the production sweep is set up by the experimenter
    # agent's dispatch_*.sh wrapper which spawns one dispatch per GPU id.
    # The dispatcher itself is single-GPU in-process.)
    from issue524_phase1_train_icl import main as phase1_main

    try:
        rc = phase1_main(cli)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 1 failed rc=%d", rc)
        return 3
    _write_sentinel(
        "epm:phase-1-complete",
        {"cells": args.cells, "smoke_cid": args.smoke_cid if args.cells == 1 else None},
    )
    return 0


def run_phase_2(args: argparse.Namespace) -> int:
    """Phase 2 — 32×32 cross-eval ΔG matrix on `pool_eval_100`.

    In smoke (--cells 1) we run the smoke cell as both source AND target
    against 4 representative bystanders (the plan-named "4 representative
    targets at 100 Q"); in sweep we run all 992 ordered off-diagonal pairs.
    """
    _emit_phase_log("phase_2")
    if args.cpu_only:
        logger.info("--cpu-only set; skipping Phase 2 (GPU eval).")
        return 0
    logger.info("Phase 2 cross-eval (cells=%d)", args.cells)

    cli: list[str] = ["--n-probes", "100"]
    if args.cells == 1:
        # Smoke: one source × 4 targets. Plan §"Smoke architecture parity":
        # "ONE ICL cell end-to-end (train -> cross-eval against 4
        # representative targets at 100 Q)".
        cli.extend(["--source-cids", args.smoke_cid])
        cli.extend(["--target-cids", *args.smoke_targets])
    from issue524_phase2_eval import main as phase2_main

    try:
        rc = phase2_main(cli)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 2 failed rc=%d", rc)
        return 3
    _write_sentinel(
        "epm:phase-2-complete",
        {"cells": args.cells, "smoke_cid": args.smoke_cid if args.cells == 1 else None},
    )
    return 0


def run_phase_3(args: argparse.Namespace) -> int:
    """Phase 3 — activation extraction for the 16 ICL contexts."""
    _emit_phase_log("phase_3")
    if args.cpu_only:
        logger.info("--cpu-only set; skipping Phase 3 (GPU extraction).")
        return 0
    logger.info("Phase 3 activation extraction (cells=%d)", args.cells)

    cli: list[str] = []
    if args.cells == 1:
        cli.extend(["--only", args.smoke_cid])
    from issue524_phase3_extract_icl import main as phase3_main

    try:
        rc = phase3_main(cli)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 3 failed rc=%d", rc)
        return 3
    _write_sentinel(
        "epm:phase-3-complete",
        {"cells": args.cells, "smoke_cid": args.smoke_cid if args.cells == 1 else None},
    )
    return 0


def run_phase_4(args: argparse.Namespace) -> int:
    """Phase 4 — predictor matrices (CPU; minutes)."""
    _emit_phase_log("phase_4")
    logger.info("Phase 4 predictor matrices (cells=%d)", args.cells)

    cli = ["--log-level", args.log_level]
    if args.cells == 1:
        cli.extend(["--smoke"])
    from issue524_phase4_predictors import main as phase4_main

    try:
        rc = phase4_main(cli)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 4 failed rc=%d", rc)
        return 3
    _write_sentinel(
        "epm:phase-4-complete",
        {"cells": args.cells},
    )
    return 0


def run_phase_5(args: argparse.Namespace) -> int:
    """Phase 5 — nested LTCO-CV + Tobit + bootstrap (CPU; minutes)."""
    _emit_phase_log("phase_5")
    logger.info("Phase 5 metrics (cells=%d, B=%d)", args.cells, args.bootstrap_b)

    cli = [
        "--log-level",
        args.log_level,
        "--b",
        str(args.bootstrap_b),
        "--seed",
        str(args.seed),
    ]
    if args.cells == 1:
        cli.extend(["--smoke"])
    from issue524_phase5_metrics import main as phase5_main

    try:
        rc = phase5_main(cli)
    except SystemExit as e:
        rc = int(e.code) if e.code is not None else 1
    if rc != 0:
        logger.error("Phase 5 failed rc=%d", rc)
        return 3
    _write_sentinel(
        "epm:phase-5-complete",
        {"cells": args.cells, "bootstrap_b": args.bootstrap_b},
    )
    return 0


PHASE_DISPATCH = {
    "0": run_phase_0,
    "1": run_phase_1,
    "2": run_phase_2,
    "3": run_phase_3,
    "4": run_phase_4,
    "5": run_phase_5,
}


def _preflight(args: argparse.Namespace) -> int:
    """Sanity-check the in-process imports + the pool path before running phases.

    Returns 0 on success, 2 on a preflight failure (the orchestrator reads
    rc=2 as "fix the environment, don't retry blindly").
    """
    _emit_phase_log("preflight")
    # Import-time check: every phase module loads correctly (catches ABI
    # breaks early; this is the "compile-test critical paths" line from
    # the experiment-implementer.md mandatory checklist).
    try:
        import issue524_build_pool_eval_100  # noqa: F401
        import issue524_phase0_1_floor_diagnosis  # noqa: F401
        import issue524_phase0_2_build_icl_blocks  # noqa: F401
        import issue524_phase0_3_induction_check  # noqa: F401
        import issue524_phase1_train_icl  # noqa: F401
        import issue524_phase2_eval  # noqa: F401
        import issue524_phase3_extract_icl  # noqa: F401
        import issue524_phase4_predictors  # noqa: F401
        import issue524_phase5_metrics  # noqa: F401
    except ImportError as e:
        logger.error("Preflight import-check failed: %s", e)
        return 2
    logger.info("Preflight: all phase modules import cleanly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Unified dispatcher entry point.

    Returns the worst exit code across the executed phases.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cells",
        type=int,
        default=32,
        help="Number of ICL+instr cells to run end-to-end. 1=smoke, 32=full panel.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds. Default 1 (matches plan §10 — single-seed scope).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed (for Phase 5 bootstrap reproducibility).",
    )
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=PHASE_NAMES,
        help="Which phase to run. 'all' runs the 0->5 chain.",
    )
    p.add_argument(
        "--smoke-cid",
        type=str,
        default="IK01",
        help="ICL context id for the --cells 1 smoke run (default IK01).",
    )
    p.add_argument(
        "--smoke-targets",
        type=str,
        default="IK02,IK05,IS01,A1",
        help="Targets for the smoke Phase 2 cross-eval (default 4 representative).",
    )
    p.add_argument(
        "--bootstrap-b",
        type=int,
        default=2000,
        help="Phase 5 bootstrap B (default 2000; auto-reduced to 16 in --cells 1).",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip GPU phases (Phase 0.1/0.2/0.3, 1, 2, 3) — for CPU-local smoke.",
    )
    p.add_argument(
        "--skip-phase-0-subs",
        action="store_true",
        help=(
            "Skip Phase 0.1/0.2/0.3 (floor diagnosis + ICL block build + induction "
            "check) but still build pool_eval_100. Use when phase-0 artifacts are "
            "already on disk from a prior run."
        ),
    )
    p.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the import preflight (rarely useful; use only for debug).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    # Normalize the smoke-targets argument.
    args.smoke_targets = [s.strip() for s in args.smoke_targets.split(",") if s.strip()]

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Smoke runs use a tiny bootstrap (16) to stay under a few seconds of
    # Phase 5 wall-time. The sweep uses the user-supplied --bootstrap-b
    # (default 2000 per plan §6).
    if args.cells == 1 and args.bootstrap_b > 100:
        logger.info(
            "Smoke: reducing bootstrap B from %d to 16 for fast Phase 5.",
            args.bootstrap_b,
        )
        args.bootstrap_b = 16

    logger.info(
        "Dispatch start: cells=%d seeds=%d phase=%s smoke_cid=%s git_sha=%s",
        args.cells,
        args.seeds,
        args.phase,
        args.smoke_cid,
        _git_sha(),
    )

    if not args.no_preflight:
        rc = _preflight(args)
        if rc != 0:
            return rc

    # Phase chain. "all" runs 0..5; a numeric --phase runs only that phase
    # (caller is responsible for the upstream artifacts).
    chain = ["0", "1", "2", "3", "4", "5"] if args.phase == "all" else [args.phase]

    worst = 0
    for ph in chain:
        fn = PHASE_DISPATCH[ph]
        t0 = time.time()
        rc = fn(args)
        elapsed = time.time() - t0
        logger.info("Phase %s rc=%d (%.1fs)", ph, rc, elapsed)
        if rc != 0 and rc > worst:
            worst = rc
        if rc not in (0, 3):
            # Hard preflight-style failures abort the chain.
            break

    # End-of-run sentinel + [phase=done] for poll_pipeline.py.
    _write_sentinel(
        "epm:results",
        {
            "chain": chain,
            "worst_rc": worst,
            "cells": args.cells,
            "seeds": args.seeds,
            "smoke_cid": args.smoke_cid if args.cells == 1 else None,
            "git_sha": _git_sha(),
        },
    )
    _emit_phase_log("done")
    return worst


if __name__ == "__main__":
    sys.exit(main())
