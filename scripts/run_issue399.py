#!/usr/bin/env python3
# ruff: noqa: RUF002
#
# Same Unicode-math suppression as scripts/eval_issue399.py — this
# dispatcher prints plan-§ references and ※ literal in its logs.
"""Issue #399 dispatcher: Phase A (train Phase-1 with ※) → Phase B (eval + log-prob).

Single entry point that runs both phases sequentially on one ``lora-7b``
ephemeral pod (plan §9). Designed to be launched once under ``nohup``
from the pod's ``/workspace/explore-persona-space/`` checkout; the
orchestrator's pod-watch loop tails ``/workspace/logs/issue-399.log``.

Phase A (~1.5 GPU-hours):

- A.0 — re-generate training JSONL with ``※`` marker via
  ``scripts/generate_issue376_marker_install.py --marker-token=※
  --allow-single-token-marker``. Writes
  ``data/issue376_marker_install_9ca040/train.jsonl``. Auto-uploads to
  HF data repo under ``issue376_marker_install/v1/9ca040/``.
- A.0 sanity — ``grep -c '※' data/issue376_marker_install_9ca040/train.jsonl``
  matches expected trained-marker-emission count (≈ 1920). Halts if ※
  has leaked into non-trigger positions (plan §8 row 4).
- A.2 — ``scripts/train.py condition=c_issue399_marker_install
  seed=${SEED} +gpu_id=0 upload_to=hf`` for SEED ∈ {42, 137, 256},
  sequentially on 1× H100 (plan §4 Step A.2). Wall-time smoke gate:
  abort if first-seed Phase 1 wall exceeds 60 min.

Phase B (~2 GPU-hours):

- B.0 — pod-side smoke checks (tokenizer + Floor A finite-values check
  on 16 contexts) are already implemented inside ``scripts/eval_issue399.py``
  via :func:`assert_trigger_marker_tokens_complex`. The dispatcher only
  re-asserts the corpus availability + checkpoint-on-Hub preconditions
  before invoking the eval rig.
- B.1 — ``scripts/eval_issue399.py --seeds 42 137 256
  --marker-token=※ --allow-single-token-marker
  --checkpoint-prefix=c_issue399_marker_install
  --logprob-contexts-per-cell=128``. Writes
  ``eval_results/issue_399/seed{S}/run_result.json`` per seed and the
  cross-seed aggregated ``eval_results/issue_399/run_result.json`` with
  the 9-cell verdict block.

Usage (on the pod, under nohup)::

    cd /workspace/explore-persona-space
    EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 nohup uv run python \\
        scripts/run_issue399.py 2>&1 > /workspace/logs/issue-399.log &

Dispatcher CLI flags allow re-running individual sub-phases:

    --skip-data-gen        Skip Phase A.0 (e.g. if JSONL already exists).
    --skip-training        Skip Phase A.1+A.2 (e.g. if checkpoints already on HF).
    --skip-eval            Skip Phase B (e.g. to only re-train).
    --seeds 42 137 256     Subset / re-order seeds for re-runs.

See ``tasks/approved/399/plans/v1.md`` (= plan.md symlink, v1.2) for the
full design.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).parent.parent

# Plan §4 + §13 constants.
MARKER_TOKEN_LITERAL: str = "※"
MARKER_SLUG: str = "9ca040"  # marker_slug("※") — Plan Assumption #26.
TRAINING_DATA_PATH: Path = (
    PROJECT_ROOT / "data" / f"issue376_marker_install_{MARKER_SLUG}" / "train.jsonl"
)
EXPECTED_TRAINING_ROWS: int = 1920  # plan §10 / generator EXPECTED_TOTAL.
# Plan §4 Phase A.0 says ※ count == trained-marker-emission count ≈ 1920.
# More precisely: from the generator's assemble_training_data() ※ is only
# appended to C+ rows (150 rows). All other rows do not carry ※. So the
# *expected* per-row leak count is 150.
#
# Leak-gate tolerance (code-review v1 concern #1): ※ (U+203B) could
# plausibly appear in Claude-generated response bodies organically
# (philosophical / coding / Japanese-adjacent text). A single incidental
# ※ should not halt the experiment. The gate now distinguishes:
#   - count > EXPECTED * (1 + tolerance)  → HALT (strong leak signal)
#   - EXPECTED < count <= EXPECTED * tol  → WARN (a few extras, continue)
#   - count < EXPECTED                    → WARN (surprising under-count,
#                                                 but missing markers is
#                                                 less dangerous than
#                                                 extras — continue)
#   - count == EXPECTED                   → silent OK
EXPECTED_MARKER_EMISSION_ROWS: int = 150
MARKER_LEAK_EXTRAS_TOLERANCE: float = 0.05  # 5% headroom for organic ※.

# Wall-time smoke gate (plan §7 / §8).
# Round-8 (2026-05-27): bumped 60 → 100 min to accommodate the rescue
# recipe (lr=1e-4, epochs=6 on a 2070-row oversampled corpus). #376's
# 25 min/seed baseline was lr=1e-4, epochs=3 on 1920 rows; round-8
# scales steps by (2070/1920)*(6/3) = 2.16x → ~55 min/seed expected,
# so 100 min is ~2x headroom over expected and still ~4x the round-7
# silent-default (lr=5e-6, epochs=1) timing.
PHASE_A_FIRST_SEED_WALL_LIMIT_MIN: float = 100.0
SEEDS_DEFAULT: tuple[int, ...] = (42, 137, 256)

LOG_DIR_POD: Path = Path("/workspace/logs")


def _log_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _run_or_die(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> None:
    """Run ``cmd`` synchronously; raise on non-zero exit code.

    Streams stdout/stderr directly to this process's streams so a
    ``tail -f`` on the dispatcher log sees the child's output live.
    """
    pretty = " ".join(cmd)
    print(f"\n[dispatcher] $ {pretty}", flush=True)
    completed = subprocess.run(cmd, env=env, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Dispatcher sub-command exited with code {completed.returncode}: {pretty}"
        )


def _grep_marker_leak_check(jsonl_path: Path, expected_count: int) -> int:
    """Plan §8 row 4 + B.0(e) — ※-leak check on the training JSONL.

    Counts the literal `※` occurrences across every row. The generator's
    contract (``generate_issue376_marker_install.py:assemble_training_data``)
    appends ※ only to the 150 C+ rows. The gate now applies a 5%
    tolerance on the upper bound (see ``MARKER_LEAK_EXTRAS_TOLERANCE``)
    because ※ (U+203B) can plausibly appear organically in Claude-
    generated bodies (philosophical / coding / Japanese-adjacent text).

    Decision matrix:
      - ``count > expected * (1 + tol)``     → HALT (per plan §8 row 4).
      - ``expected < count <= upper_limit``  → WARN (continue, document).
      - ``count < expected``                 → WARN (under-count is
        surprising — likely row composition changed — but missing markers
        cannot turn the install into bag-of-※, so continue).
      - ``count == expected``                → silent OK.

    Returns the observed count so the caller can record it in run logs.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Training JSONL missing at {jsonl_path}; cannot run ※-leak check. "
            f"Phase A.0 generator did not write the expected file — re-run "
            f"scripts/generate_issue376_marker_install.py --marker-token=※ "
            f"--allow-single-token-marker."
        )
    count = 0
    with jsonl_path.open() as f:
        for line in f:
            count += line.count(MARKER_TOKEN_LITERAL)
    upper_limit = int(expected_count * (1.0 + MARKER_LEAK_EXTRAS_TOLERANCE))
    print(
        f"  ※-leak check: {count} occurrence(s) of {MARKER_TOKEN_LITERAL!r} in "
        f"{jsonl_path} (expected {expected_count}, halt-above {upper_limit})",
        flush=True,
    )
    if count > upper_limit:
        # Hard leak: more than `tol` extras above the expected count.
        # Either the generator's row composition has genuinely changed
        # (bump EXPECTED_MARKER_EMISSION_ROWS to match), or ※ has leaked
        # into non-trigger positions and the install becomes bag-of-※.
        raise RuntimeError(
            f"※-leak gate FAILED: {count} ※ occurrences vs expected "
            f"{expected_count} (halt threshold {upper_limit}, "
            f"tolerance {MARKER_LEAK_EXTRAS_TOLERANCE:.0%}). Either the "
            f"generator's row composition has changed (update "
            f"EXPECTED_MARKER_EMISSION_ROWS in run_issue399.py to match) "
            f"or ※ has leaked into non-trigger positions — halting per "
            f"plan §8 row 4."
        )
    if count > expected_count:
        # Within tolerance: a few extras likely organic in Claude bodies.
        # Document and continue rather than halt.
        logger.warning(
            "※-leak gate within tolerance: %d ※ vs expected %d "
            "(<= halt threshold %d, %.0f%% tolerance). Likely "
            "incidental ※ in generated response bodies; continuing. "
            "If the count drifts further, bump "
            "EXPECTED_MARKER_EMISSION_ROWS to reflect the new generator "
            "row composition.",
            count,
            expected_count,
            upper_limit,
            MARKER_LEAK_EXTRAS_TOLERANCE * 100,
        )
    elif count < expected_count:
        # Under-count: row composition may have shrunk. Not a leak per
        # se, but worth surfacing — missing markers are less dangerous
        # than extras (can't make install bag-of-※) so we continue.
        logger.warning(
            "※-leak gate observed fewer markers than expected: "
            "%d ※ vs expected %d. Generator row composition may have "
            "changed (e.g. fewer C+ rows) — verify intent. Continuing "
            "since missing markers cannot turn the install into "
            "bag-of-※. If the new count is correct, update "
            "EXPECTED_MARKER_EMISSION_ROWS in run_issue399.py to match.",
            count,
            expected_count,
        )
    return count


def phase_a0_generate_data(skip_data_gen: bool) -> None:
    """Phase A.0 — regenerate training JSONL with ※ marker."""
    _log_section("Phase A.0 — Generate training JSONL with ※ marker")
    if skip_data_gen and TRAINING_DATA_PATH.exists():
        print(
            f"  --skip-data-gen passed and {TRAINING_DATA_PATH} exists; "
            f"skipping generator invocation.",
            flush=True,
        )
        _grep_marker_leak_check(TRAINING_DATA_PATH, EXPECTED_MARKER_EMISSION_ROWS)
        return

    # Generator hard-codes data/issue376_marker_install/training_questions.json as a
    # (marker-independent) cache path. On a freshly-bootstrapped pod the parent dir
    # doesn't exist; mkdir -p so the generator's open(...,"w") doesn't crash.
    (PROJECT_ROOT / "data" / "issue376_marker_install").mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts" / "generate_issue376_marker_install.py"),
        "--marker-token=" + MARKER_TOKEN_LITERAL,
        "--allow-single-token-marker",
    ]
    _run_or_die(cmd, cwd=PROJECT_ROOT)
    _grep_marker_leak_check(TRAINING_DATA_PATH, EXPECTED_MARKER_EMISSION_ROWS)


def phase_a2_train(seeds: list[int], skip_training: bool) -> None:
    """Phase A.2 — train Phase-1 LoRA, sequential per seed on 1× H100."""
    _log_section(f"Phase A.2 — Train Phase-1 LoRA (seeds={seeds})")
    if skip_training:
        print("  --skip-training passed; skipping all Phase A.2 launches.", flush=True)
        return

    env = os.environ.copy()
    # Plan §9 + CLAUDE.md mitigation (a): inline checkpoint upload to WandB
    # Artifacts is the path that hit MooseFS quota in #376; skip it. The
    # orchestrator runner.py separately uploads merged checkpoints to HF
    # Hub via cfg.upload_to=hf, which is what we want.
    env.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    LOG_DIR_POD.mkdir(parents=True, exist_ok=True) if Path("/workspace").exists() else None

    for i, seed in enumerate(seeds):
        log_path = (
            LOG_DIR_POD if Path("/workspace").exists() else PROJECT_ROOT / "logs"
        ) / f"issue-399-train-seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv",
            "run",
            "python",
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "condition=c_issue399_marker_install",
            f"seed={seed}",
            "upload_to=hf",
            "+gpu_id=0",
        ]
        pretty = " ".join(cmd)
        print(
            f"\n[dispatcher] Phase A.2 seed {seed}: {pretty}  (logs: {log_path})",
            flush=True,
        )
        t0 = time.time()
        with log_path.open("w") as logf:
            completed = subprocess.run(
                cmd,
                env=env,
                cwd=PROJECT_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall_min = (time.time() - t0) / 60.0
        if completed.returncode != 0:
            raise RuntimeError(
                f"Training seed {seed} exited with code {completed.returncode}. "
                f"Check {log_path} for the traceback."
            )
        print(
            f"  Phase A.2 seed {seed}: completed in {wall_min:.1f} min (log: {log_path})",
            flush=True,
        )

        # Wall-time smoke gate on the FIRST seed only (plan §7). #376
        # baseline (lr=1e-4, epochs=3, 1920 rows) was ~25 min/seed; the
        # round-8 recipe (lr=1e-4, epochs=6, 2070 rows) scales to ~55
        # min/seed expected. Limit is 100 min (see PHASE_A_FIRST_SEED_WALL_LIMIT_MIN).
        if i == 0 and wall_min > PHASE_A_FIRST_SEED_WALL_LIMIT_MIN:
            raise RuntimeError(
                f"Phase A.2 first-seed wall {wall_min:.1f} min > "
                f"{PHASE_A_FIRST_SEED_WALL_LIMIT_MIN:.0f} min smoke-gate "
                f"limit. Round-8 expected baseline is ~55 min/seed (lr=1e-4, "
                f"epochs=6 on 2070-row oversampled corpus, 2.16x #376's step "
                f"count). Investigate before continuing with seeds {seeds[1:]}."
            )


def phase_b_eval(seeds: list[int], skip_eval: bool, logprob_contexts_per_cell: int) -> None:
    """Phase B.1 — invoke scripts/eval_issue399.py against the trained checkpoints."""
    _log_section(f"Phase B.1 — Run eval_issue399.py (seeds={seeds})")
    if skip_eval:
        print("  --skip-eval passed; skipping Phase B launches.", flush=True)
        return

    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts" / "eval_issue399.py"),
        "--seeds",
        *[str(s) for s in seeds],
        "--marker-token=" + MARKER_TOKEN_LITERAL,
        "--allow-single-token-marker",
        "--checkpoint-prefix=c_issue399_marker_install",
        f"--logprob-contexts-per-cell={logprob_contexts_per_cell}",
    ]
    _run_or_die(cmd, cwd=PROJECT_ROOT)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS_DEFAULT),
        help=f"Seeds to train + eval. Default: {list(SEEDS_DEFAULT)}.",
    )
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip Phase A.0 if data/issue376_marker_install_9ca040/train.jsonl already exists.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Phase A.1+A.2 (assume checkpoints are already on HF Hub).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Phase B (only re-train, don't run eval).",
    )
    parser.add_argument(
        "--logprob-contexts-per-cell",
        type=int,
        default=128,
        help="Pass-through to scripts/eval_issue399.py. Default 128 (plan §11).",
    )
    args = parser.parse_args()

    print("=== Issue #399 dispatcher: log-prob rescue test ===", flush=True)
    print(f"  Seeds: {args.seeds}", flush=True)
    print(f"  Marker (Phase A install + Phase B both blocks): {MARKER_TOKEN_LITERAL!r}", flush=True)
    print(f"  Training JSONL path: {TRAINING_DATA_PATH}", flush=True)
    print(f"  Logprob contexts per cell: {args.logprob_contexts_per_cell}", flush=True)
    _quota_env = os.environ.get("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "unset")
    print(f"  EPM_SKIP_INLINE_CHECKPOINT_UPLOAD: {_quota_env}", flush=True)

    phase_a0_generate_data(skip_data_gen=args.skip_data_gen)
    phase_a2_train(seeds=args.seeds, skip_training=args.skip_training)
    phase_b_eval(
        seeds=args.seeds,
        skip_eval=args.skip_eval,
        logprob_contexts_per_cell=args.logprob_contexts_per_cell,
    )

    _log_section("=== Issue #399 dispatcher: done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
