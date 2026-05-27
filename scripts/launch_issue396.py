#!/usr/bin/env python3
"""Wave-based launcher for task #396 — 48 LoRAs, ※ marker, v2.3 recipe.

Modeled on ``scripts/launch_phase_a1_chained.py`` (on-main analogue with
the CVD-prefix + subprocess + argparse ``--gpu`` pattern). Each subprocess
sees ONE GPU via ``CUDA_VISIBLE_DEVICES={gpu}``, passes ``--gpu 0`` inside
the subprocess (train_lora's CUDA_VISIBLE_DEVICES clobber lands on the
single visible GPU rather than the original physical id).

Mitigations vs the launch_phase_a1_chained pattern:

* **Post-upload verify + cleanup.** ``run_leakage_experiment.py`` line 938
  has cleanup COMMENTED OUT (``delete_after=False`` is active). Without
  launcher-side cleanup, 4 parallel sources x ~19 GB peak per source =
  ~76 GB per wave; 3+ waves hit RunPod's MooseFS ~130 GB per-pod quota
  with ``OSError errno=122 (EDQUOT)``. After each subprocess completes,
  this launcher (a) verifies the adapter uploaded to HF via
  ``HfApi.list_repo_files`` and (b) ``shutil.rmtree(merged_dir,
  adapter_dir)`` only after the upload check passes. Fails loud (raises)
  if the upload check fails so a downstream wave doesn't proceed on
  silently-lost weights.

* **Resume-safe ``is_done(source)`` guard.** Skips a source if the
  per-cell trajectory eval JSON
  (``eval_results/issue_396/logprob_{source}_seed42.json``) already
  exists with ``n_cells == 48 * 20``. Lets the launcher be killed and
  re-invoked without redoing finished sources.

* **v2.3 recipe knobs.** Each subprocess is invoked with
  ``--lr 1e-4 --max-length 2048 --warmup-ratio 0.10 --marker-token ※``
  (plus ``--allow-single-token-marker`` semantics handled upstream by
  the data builder; this script doesn't pass it to ``run_leakage_experiment.py``
  because that script's training path doesn't gate single-token markers —
  only the data-builder does). The two other v2.3 recipe knobs
  (target_modules + lr_scheduler_type) are already hardcoded to the v2.3
  values inside ``train_lora`` at ``src/explore_persona_space/train/sft.py``.

Task #396 plan v2.3 §4.4. PR #390 / branch ``issue-396``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# scripts/ on path so we can import the canonical 48-persona list.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from analyze_length_rate_n48 import (  # noqa: E402
    INHERITED_SOURCES_24,
    NEW_PERSONA_PROMPTS_296,
)

from explore_persona_space.personas import marker_slug  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "issue_396"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Per-source training outputs live under output_dir = EVAL_RESULTS_LEGACY / run_name
# (set by run_leakage_experiment.py). The script writes adapter/ and merged/
# subdirs under that path.
EVAL_RESULTS_LEGACY = PROJECT_ROOT / "eval_results" / "leakage_experiment"

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
MARKER_TEXT = "※"
NEG_SET = "asst_excluded"
PROMPT_LENGTH = "medium"
SEED = 42

# v2.3 recipe knobs — task #396 plan v2.3 §3 method-delta table + §10
# Reproducibility Card. These OVERRIDE the run_leakage_experiment.py
# argparse defaults (which are the legacy #296 recipe).
RECIPE_LR = 1.0e-4
RECIPE_MAX_LENGTH = 2048
RECIPE_WARMUP_RATIO = 0.10

# The 48-persona panel. Union order matters for log/wave determinism.
ALL_48 = list(NEW_PERSONA_PROMPTS_296.keys()) + INHERITED_SOURCES_24
assert len(set(ALL_48)) == 48, (
    f"ALL_48 should be 48 unique sources; got {len(set(ALL_48))} unique "
    f"({len(ALL_48)} total — duplicates in the source dicts?)"
)


def is_done(source: str) -> bool:
    """Resume-safe gate: skip if the per-source trajectory eval JSON is complete.

    The trajectory evaluator (scripts/eval_issue396_logprob.py) writes to
    ``eval_results/issue_396/logprob_{source}_seed42.json`` with
    ``n_cells == 48 * 20 == 960``. This launcher only checks for that
    artifact — NOT for the merged-model dir — because the merged dir is
    deliberately deleted by the post-success cleanup step.
    """
    path = EVAL_RESULTS_DIR / f"logprob_{source}_seed{SEED}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("[%s] eval JSON exists but is malformed (%s) — re-running", source, e)
        return False
    expected = 48 * 20
    if data.get("n_cells") == expected:
        return True
    logger.warning(
        "[%s] eval JSON exists with n_cells=%r (expected %d) — re-running",
        source,
        data.get("n_cells"),
        expected,
    )
    return False


def build_cmd(source: str, gpu: int, pod: str) -> str:
    """Build the bash command line for one source's training subprocess.

    Each subprocess sees ONE GPU via ``CUDA_VISIBLE_DEVICES={gpu}``. The
    subprocess then passes ``--gpu 0`` to ``run_leakage_experiment.py``,
    which lands on the only visible device (sft.py's
    ``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`` is a no-op
    here because the env var is already set + the only visible device IS
    device 0 from the subprocess's perspective).
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED={SEED} "
        f"EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 "
        f"uv run python scripts/archive/run_leakage_experiment.py "
        f"--trait marker --source {source} --neg-set {NEG_SET} "
        f"--prompt-length {PROMPT_LENGTH} --seed {SEED} --gpu 0 --pod {pod} "
        f"--marker-token {MARKER_TEXT} "
        f"--lr {RECIPE_LR} --max-length {RECIPE_MAX_LENGTH} "
        f"--warmup-ratio {RECIPE_WARMUP_RATIO} "
        f"--phase a1"
    )


def _expected_run_name(source: str) -> str:
    """Mirror make_run_name(args) in run_leakage_experiment.py for this source."""
    slug = marker_slug(MARKER_TEXT)
    return f"marker_{source}_{NEG_SET}_{PROMPT_LENGTH}_{slug}_seed{SEED}"


def verify_upload_then_cleanup(source: str, dry_run: bool = False) -> None:
    """Post-training: verify HF upload succeeded, then explicitly rmtree local weights.

    Required because run_leakage_experiment.py cleanup is COMMENTED OUT at line 938
    ('delete_after=False' is active, with the comment "delete manually after
    verifying uploads"). This launcher honors that comment by doing the manual
    delete here — verify FIRST via HfApi.list_repo_files (cheap GraphQL),
    delete ONLY after a positive match. The deletion targets:

    - ``eval_results/leakage_experiment/{run_name}/merged/`` (~14 GB)
    - ``eval_results/leakage_experiment/{run_name}/adapter/`` (~5 GB)

    Together ~19 GB per source. With 4 parallel waves on a 130 GB per-pod quota,
    failure to clean between waves overruns disk within 3 waves.

    Fails loud (RuntimeError) when the HF upload check fails. The caller
    drops the source's subprocess result and continues with the next wave
    rather than silently losing weights.
    """
    run_name = _expected_run_name(source)
    expected_hf_path = f"leakage_experiment/{run_name}"

    if dry_run:
        logger.info(
            "[%s] DRY-RUN: would verify HF path %r then rmtree local weights",
            source,
            expected_hf_path,
        )
        return

    # Local import — keep huggingface_hub off the module-import path so
    # --dry-run + --help don't require an HF auth token.
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id=HF_MODEL_REPO, repo_type="model")
    except Exception as e:
        raise RuntimeError(
            f"[{source}] HF list_repo_files failed: {type(e).__name__}: {e}. "
            "Refusing to delete local weights without confirming the upload."
        ) from e

    uploaded = [f for f in repo_files if f.startswith(expected_hf_path)]
    if not uploaded:
        raise RuntimeError(
            f"[{source}] Adapter upload verification FAILED: no files match "
            f"{expected_hf_path!r} in {HF_MODEL_REPO}. Local weights LEFT IN PLACE "
            f"for manual inspection at "
            f"{EVAL_RESULTS_LEGACY / run_name}. Investigate the upload step in "
            "run_leakage_experiment.py before re-running this source."
        )

    # Verification passed. Safe to delete local merged + adapter dirs.
    output_dir = EVAL_RESULTS_LEGACY / run_name
    merged_dir = output_dir / "merged"
    adapter_dir = output_dir / "adapter"
    freed_bytes = 0
    for path in (merged_dir, adapter_dir):
        if path.exists():
            # Measure freed bytes for the log — useful for tracking the
            # 130 GB per-pod quota over a long run. ``stat().st_size`` can
            # raise on transient FS errors (e.g. a file gone between
            # rglob and stat); suppress so the rmtree below still runs.
            with contextlib.suppress(Exception):
                freed_bytes += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path)
    logger.info(
        "[%s] HF upload verified (%d files at %s); freed ~%.1f GB local "
        "(merged_dir + adapter_dir).",
        source,
        len(uploaded),
        expected_hf_path,
        freed_bytes / 1e9,
    )


def wave_loop(
    sources: list[str],
    n_gpus: int,
    pod: str,
    *,
    dry_run: bool = False,
) -> dict[str, str]:
    """Run sources in waves of n_gpus parallel subprocesses on one pod.

    Returns a dict ``{source: "done" | "skipped" | "failed"}``. The function
    proceeds wave-by-wave; within a wave all n_gpus subprocesses launch in
    parallel and the wave blocks until all return. After each wave, post-
    success sources are verified+cleaned BEFORE the next wave's subprocesses
    spawn — this is what keeps the per-wave peak disk under the 130 GB
    quota (peak = n_gpus x ~19 GB).
    """
    pending = [s for s in sources if not is_done(s)]
    already_done = [s for s in sources if is_done(s)]
    results: dict[str, str] = {s: "skipped" for s in already_done}

    if already_done:
        logger.info(
            "Resume: %d / %d sources already complete; running %d new",
            len(already_done),
            len(sources),
            len(pending),
        )

    for wave_start in range(0, len(pending), n_gpus):
        wave = pending[wave_start : wave_start + n_gpus]
        wave_idx = wave_start // n_gpus + 1
        n_waves_remaining = (len(pending) - wave_start + n_gpus - 1) // n_gpus
        logger.info(
            "=== Wave %d / %d (%d sources): %s ===",
            wave_idx,
            wave_idx + n_waves_remaining - 1,
            len(wave),
            ", ".join(wave),
        )

        procs: list[tuple[str, subprocess.Popen]] = []
        wave_log_handles: list[tuple] = []
        for gpu_idx, source in enumerate(wave):
            cmd = build_cmd(source, gpu_idx, pod)
            log_path = LOG_DIR / f"i396_{source}_gpu{gpu_idx}.log"
            logger.info("  [%s] -> GPU %d, log=%s", source, gpu_idx, log_path)
            if dry_run:
                logger.info("    DRY-RUN: %s", cmd)
                continue
            # Open log handle explicitly so we can .close() after wait()
            # below — avoids leaking fds for the lifetime of the process.
            log_handle = open(log_path, "w")  # noqa: SIM115 - closed after wait() below
            wave_log_handles.append((log_handle, source))
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            procs.append((source, proc))

        # Block until all subprocesses in this wave return.
        for source, proc in procs:
            proc.wait()
            if proc.returncode != 0:
                logger.error(
                    "[%s] subprocess exited %d — see log %s",
                    source,
                    proc.returncode,
                    LOG_DIR / f"i396_{source}_gpu*.log",
                )
                results[source] = "failed"
                continue

            # Post-success: verify HF upload, then cleanup. Failure of either
            # is caught and logged so the next wave can still proceed.
            try:
                verify_upload_then_cleanup(source, dry_run=dry_run)
                results[source] = "done"
            except RuntimeError as e:
                logger.error("[%s] post-success cleanup failed: %s", source, e)
                results[source] = "failed"

        # Close the wave's stdout log handles. Doing this AFTER the wait()
        # loop guarantees subprocess writes have flushed; closing earlier
        # would race with in-flight buffer flushes from the child process.
        for handle, _src in wave_log_handles:
            with contextlib.suppress(Exception):
                handle.close()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the 48-source LoRA training wave for task #396 "
            "(※ marker, v2.3 recipe knobs, single-pod sweep)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-396",
        help="Pod identifier (passed to run_leakage_experiment.py --pod for logging).",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help="Number of GPUs on the pod. Wave size = n_gpus. Default 4 (H100 quad).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "Optional subset of sources to run (for pilot waves / re-running specific "
            "personas). Default: all 48 in ALL_48 order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the wave plan and per-source commands; do not spawn subprocesses.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sources = args.sources if args.sources else ALL_48
    if args.sources:
        unknown = [s for s in args.sources if s not in ALL_48]
        if unknown:
            parser.error(f"--sources contains names not in ALL_48: {unknown}")

    logger.info(
        "Task #396 launcher: %d sources on pod=%s, n_gpus=%d, dry_run=%s, marker=%r",
        len(sources),
        args.pod,
        args.n_gpus,
        args.dry_run,
        MARKER_TEXT,
    )
    logger.info(
        "Recipe: lr=%g, max_length=%d, warmup_ratio=%g (v2.3 plan §3 / §10).",
        RECIPE_LR,
        RECIPE_MAX_LENGTH,
        RECIPE_WARMUP_RATIO,
    )

    results = wave_loop(sources, n_gpus=args.n_gpus, pod=args.pod, dry_run=args.dry_run)

    # Summary
    done = sum(1 for v in results.values() if v == "done")
    skipped = sum(1 for v in results.values() if v == "skipped")
    failed = sum(1 for v in results.values() if v == "failed")
    logger.info(
        "Launcher complete: %d done, %d skipped (already complete), %d failed.",
        done,
        skipped,
        failed,
    )
    if failed:
        for source, status in sorted(results.items()):
            if status == "failed":
                logger.error("  FAILED: %s", source)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
