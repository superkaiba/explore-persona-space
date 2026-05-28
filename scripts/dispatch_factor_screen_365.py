#!/usr/bin/env python3
# epm-lint: subprocess-env-implicit-load -- env loaded by parent `python -m explore_persona_space.experiments.factor_screen_365.__main__` which calls load_dotenv before invoking this dispatcher (see factor_screen_365/__main__.py line ~1488)  # noqa: E501
"""Pod-side dispatcher for the task #365 2^5 factor screen.

Two stages:

  1. **Pool stage** (one process per source, sequential). For each source in
     ``--sources``, calls
     ``python -m explore_persona_space.experiments.factor_screen_365
     --mode dispatch --source <src> --pool-dir <dir>`` to:

       - render every ``(A, B, C)`` system prompt under Qwen2.5-7B-Instruct,
       - run the C-axis preflight (Jaccard / role-adoption / token equality),
       - generate the on-policy (D=0) and off-policy (D=1) completion pools,
       - emit ``prompt_manifest.json`` and ``persona_panel_manifest.csv``.

     Pools are SHARED across the E-axis flip — one (source, A, B, C) tuple
     yields one on-policy JSONL + one off-policy JSONL, reused for E=0 / E=1.

  2. **Training stage** (96 jobs fanned across 8 GPUs by
     ``CUDA_VISIBLE_DEVICES``). For each ``(cell, source, seed)`` triple,
     launches ``python -m explore_persona_space.experiments.factor_screen_365
     --cell <ABCDE> --source <src> --seed <N> --pool-dir <dir>
     --output-dir <slab_root>/cell_<key>/source_<src>/seed_<N>/``.

The librarian-only gate (plan §7) is implemented by ``--sources librarian``
on the first run; once the gate clears the user re-runs the dispatcher with
the full ``--sources librarian,surgeon,programmer`` set.

Aggregation is a separate one-shot:

    uv run python -m explore_persona_space.experiments.factor_screen_365 \\
        --mode aggregate --slab-root <slab_root> --output-dir <agg_dir>

Round-5 (issue #365): ``--resume`` (default ON) skips cells whose
``metrics.json`` + adapter dir are already on disk, AND skips cells whose
adapter is already on the HF Hub model repo. After the silent dispatcher
death at hour 25 of the round-4 run (10 cells trained, 22 to go), the
relaunch path needs to avoid retraining anything that already exists.
Pass ``--no-resume`` to force a clean rerun.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("dispatch_factor_screen_365")

SOURCES_DEFAULT = ("librarian", "surgeon", "programmer")


def _detect_physical_gpu_count() -> int:
    """Return the number of NVIDIA GPUs visible to the dispatcher.

    Uses ``nvidia-smi --query-gpu=count`` (the GPU-count column is one row
    per GPU, count the rows). Falls back to 1 on any error: a single-GPU
    pod is the safer guess for a misconfigured environment than launching
    8 parallel subprocesses against phantom GPUs.
    """
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi is None:
        return 1
    try:
        out = subprocess.check_output(  # epm-lint: subprocess-env-inherit -- nvidia-smi probe, no credentials needed  # noqa: E501
            [nvsmi, "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=10
        )
    except Exception:
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.strip()))


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def _parse_sources(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_seeds(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--sources",
        type=_parse_sources,
        default=list(SOURCES_DEFAULT),
        help="Comma-separated source personas (default: librarian,surgeon,programmer).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[42],
        help="Comma-separated baseline seeds (default: 42).",
    )
    p.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("data/issue_365/pools"),
        help="Where to materialise the per-source on/off-policy completion pools.",
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_365"),
        help="Per-cell metrics + adapters land here (cell_<key>/source_<src>/seed_<N>/).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "GPU pool size for the training stage. When omitted, auto-detected "
            "from nvidia-smi (defaults to 1 if nvidia-smi is unavailable). "
            "The auto-detect fixes the round-5 silent dispatcher death "
            "where the legacy default of 8 launched subprocesses against "
            "phantom GPUs 1-7 on a single-GPU pod, polluting the slab with "
            "zero-byte factor_screen_failed.json files."
        ),
    )
    p.add_argument(
        "--skip-pool-stage",
        action="store_true",
        help="Skip the pool-generation stage (use when pools already exist on disk).",
    )
    p.add_argument(
        "--skip-off-policy",
        action="store_true",
        help="Skip Claude D1 off-policy generation (D=1 cells will fail at prepare_cell).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command list without launching anything.",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Skip cells whose metrics.json + adapter dir are on disk, or whose "
            "adapter is already on the HF Hub model repo. ON by default."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force-rerun every cell even if results already exist.",
    )
    p.add_argument(
        "--skip-hub-probe",
        action="store_true",
        help=(
            "When --resume is on, only probe the local disk, not the HF Hub "
            "model repo. Useful for air-gapped pods or when the Hub is slow."
        ),
    )
    p.add_argument(
        "--prioritize-failed",
        dest="prioritize_failed",
        action="store_true",
        default=True,
        help=(
            "When --resume is on, launch cells that carry a prior "
            "factor_screen_failed.json marker BEFORE fresh cells, so a "
            "code-fix relaunch processes the cells the fix targeted first. "
            "ON by default. Prevents the #391 cell_10111 class of bug where "
            "a deterministic per-cell crash gets code-fixed, the dispatcher "
            "is relaunched, and a second mid-run crash (EDQUOT, OOM, host "
            "migration) again kills the dispatcher before its "
            "lexicographic walk reaches the originally-failed cell."
        ),
    )
    p.add_argument(
        "--no-prioritize-failed",
        dest="prioritize_failed",
        action="store_false",
        help="Process cells in strict lexicographic order even under --resume.",
    )
    return p


def _pool_stage(args: argparse.Namespace) -> int:
    """Sequential per-source dispatch mode call."""
    args.pool_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        cmd = [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.factor_screen_365",
            "--mode",
            "dispatch",
            "--source",
            source,
            "--pool-dir",
            str(args.pool_dir),
        ]
        if args.skip_off_policy:
            cmd.append("--skip-off-policy")
        log.info("Pool stage: %s", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, env={**os.environ})
        if rc != 0:
            log.error("Pool stage failed for source=%s (rc=%d)", source, rc)
            return rc
    return 0


def _training_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Enumerate the (cell_key, source, seed) jobs for the training stage."""
    cells = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=5)]
    return [
        (cell_key, source, seed)
        for cell_key in cells
        for source in args.sources
        for seed in args.seeds
    ]


def _cell_output_dir(slab_root: Path, cell_key: str, source: str, seed: int) -> Path:
    return slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"


def cell_complete_on_disk(slab_root: Path, cell_key: str, source: str, seed: int) -> bool:
    """Return True if this cell's metrics.json + adapter dir already exist.

    Round-6 resume probe. A cell is considered complete on disk when:

      * ``slab_root/cell_<key>/source_<src>/seed_<N>/metrics.json`` is a
        non-empty JSON file containing a non-empty ``persona_panel_scores``
        block (sentinel that the eval phase actually ran to completion).
      * ``slab_root/cell_<key>/source_<src>/seed_<N>/adapter/`` is a directory
        containing at least one non-empty file (PEFT writes
        ``adapter_model.safetensors`` and ``adapter_config.json``).

    Round-5's check was "metrics.json non-empty AND adapter/ non-empty"; that
    let cells co-existing with a stale ``factor_screen_failed.json`` (round-4
    artifacts where a later retry failed *after* a successful prior run)
    sneak through. The ``persona_panel_scores`` sentinel is robust against
    that. The presence-or-absence of ``factor_screen_failed.json`` is not
    relevant once we gate on the eval-completion sentinel.
    """
    output_dir = _cell_output_dir(slab_root, cell_key, source, seed)
    metrics = output_dir / "metrics.json"
    if not metrics.exists() or metrics.stat().st_size == 0:
        return False
    try:
        import json as _json

        with open(metrics) as f:
            payload = _json.load(f)
    except Exception:
        return False
    panel = payload.get("persona_panel_scores") if isinstance(payload, dict) else None
    if not isinstance(panel, dict) or not panel:
        return False
    adapter = output_dir / "adapter"
    if not adapter.is_dir():
        return False
    return any(p.is_file() and p.stat().st_size > 0 for p in adapter.iterdir())


def cell_has_failure_marker(slab_root: Path, cell_key: str, source: str, seed: int) -> bool:
    """Return True if this cell has a ``factor_screen_failed.json`` marker.

    Written by ``factor_screen_365.__main__`` whenever a cell-mode invocation
    raises (preflight failures like ``CPaddingError``, mid-training OOM, eval
    crashes, etc.). The marker is purely diagnostic — ``cell_complete_on_disk``
    correctly ignores it, so a failed cell is ALWAYS re-queued by ``--resume``.

    The marker is consulted only to PRIORITIZE the queue: ``--prioritize-failed``
    (on by default) places marker-bearing cells at the head of the launch
    order so a code-fix relaunch processes them before any other work that
    could trigger a second mid-run dispatcher kill (EDQUOT, OOM, RunPod host
    migration, NCCL crash). Post-mortem of task #391 cell_10111: the round-4
    padding fix landed, the dispatcher was relaunched, but a second EDQUOT
    incident killed the dispatcher before its lexicographic walk reached
    cell_10111 in slot 23 of 32.

    Returns False if the marker file is absent or unreadable; the caller
    should treat False as "no prior failure on record" (which means strict
    lex-order is fine for this cell).
    """
    output_dir = _cell_output_dir(slab_root, cell_key, source, seed)
    marker = output_dir / "factor_screen_failed.json"
    return marker.exists() and marker.stat().st_size > 0


def hf_hub_adapter_run_name(cell_key: str, source: str, seed: int) -> str:
    """Return the run-name suffix used by ``training.train_one_cell``.

    Mirrors ``training.train_one_cell``'s ``run_name`` template:
        ``f"i365_cell_{cell.key}_source_{source}_seed{seed}"``
    The adapter lands at ``adapters/issue_365/<run_name>`` in the model repo
    (``superkaiba1/explore-persona-space``).
    """
    return f"i365_cell_{cell_key}_source_{source}_seed{seed}"


def cell_complete_on_hub(
    cell_key: str,
    source: str,
    seed: int,
    *,
    hub_files_cache: list[str] | None = None,
) -> bool:
    """Return True if this cell's adapter is already uploaded to HF Hub.

    Probes ``superkaiba1/explore-persona-space`` for any file under
    ``adapters/issue_365/<run_name>/``. Returns False on any HfApi error
    (network down, no token) so the dispatcher falls through to a local
    rebuild rather than silently treating "couldn't reach hub" as "skip".

    Parameters
    ----------
    hub_files_cache:
        Optional pre-fetched list of files in the model repo. When supplied
        we use it instead of probing (one Hub call per dispatcher
        invocation). When ``None`` we probe lazily and crash-loud on errors.
    """
    run_name = hf_hub_adapter_run_name(cell_key, source, seed)
    prefix = f"adapters/issue_365/{run_name}/"
    files = hub_files_cache
    if files is None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ.get("HF_TOKEN"))
            files = api.list_repo_files(
                repo_id="superkaiba1/explore-persona-space", repo_type="model"
            )
        except Exception as exc:
            log.warning("HF Hub adapter probe failed (%s); falling through", exc)
            return False
    return any(f.startswith(prefix) for f in files)


def _prefetch_hub_adapter_index() -> list[str] | None:
    """One-shot probe of the HF Hub model repo for all adapter files.

    Cuts the resume scan from 96 per-cell HfApi calls to one. Returns
    ``None`` if the probe fails (e.g. no network); callers should treat
    that as "no hub data" rather than crashing.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        return api.list_repo_files(repo_id="superkaiba1/explore-persona-space", repo_type="model")
    except Exception as exc:
        log.warning("HF Hub model-repo index fetch failed (%s); disk-only resume", exc)
        return None


def _training_cmd(
    *,
    cell_key: str,
    source: str,
    seed: int,
    pool_dir: Path,
    slab_root: Path,
    resume: bool = True,
) -> list[str]:
    output_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.factor_screen_365",
        "--cell",
        cell_key,
        "--source",
        source,
        "--seed",
        str(seed),
        "--pool-dir",
        str(pool_dir),
        "--output-dir",
        str(output_dir),
    ]
    if not resume:
        cmd.append("--no-resume")
    return cmd


def _wait_for_free_gpu(running: dict[int, subprocess.Popen], gpu_pool: list[int]) -> int:
    while True:
        for gpu in gpu_pool:
            proc = running.get(gpu)
            if proc is None:
                return gpu
            if proc.poll() is not None:
                # Process finished.
                running.pop(gpu, None)
                if proc.returncode != 0:
                    log.warning("Job on GPU %d exited with rc=%d", gpu, proc.returncode)
                return gpu
        time.sleep(2)


def _training_stage(args: argparse.Namespace) -> int:
    jobs = _training_jobs(args)
    physical = _detect_physical_gpu_count()
    if args.num_gpus is None:
        args.num_gpus = physical
        log.info("Auto-detected %d physical GPU(s); using --num-gpus %d", physical, physical)
    elif args.num_gpus > physical:
        log.error(
            "--num-gpus=%d exceeds physical GPU count=%d; clamping to %d "
            "(launching subprocesses against phantom GPUs caused the round-5 "
            "silent dispatcher death — see issue #365 round-6 forensics).",
            args.num_gpus,
            physical,
            physical,
        )
        args.num_gpus = physical
    gpu_pool = list(range(args.num_gpus))
    running: dict[int, subprocess.Popen] = {}

    # Round-5: pre-fetch the HF Hub model-repo file index once so the resume
    # probe doesn't make 96 separate Hub calls. None = probe failed / Hub
    # skipped; treat as "no hub data" and fall back to disk-only resume.
    hub_files: list[str] | None = None
    if args.resume and not args.skip_hub_probe:
        hub_files = _prefetch_hub_adapter_index()

    skipped_disk = 0
    skipped_hub = 0
    failed_retry_jobs: list[tuple[str, str, int]] = []
    fresh_jobs: list[tuple[str, str, int]] = []
    for cell_key, source, seed in jobs:
        # Resume short-circuit: skip cells whose results already exist
        # locally OR on the Hub. Local-disk check is the cheap path; the
        # Hub check uses the pre-fetched index.
        if args.resume and cell_complete_on_disk(args.slab_root, cell_key, source, seed):
            output_dir = _cell_output_dir(args.slab_root, cell_key, source, seed)
            log.info(
                "Cell already complete on disk -- skipping; results at %s",
                output_dir,
            )
            skipped_disk += 1
            continue
        if (
            args.resume
            and not args.skip_hub_probe
            and cell_complete_on_hub(cell_key, source, seed, hub_files_cache=hub_files)
        ):
            log.info(
                "Cell already complete on HF Hub -- skipping; adapter at "
                "superkaiba1/explore-persona-space/adapters/issue_365/%s/",
                hf_hub_adapter_run_name(cell_key, source, seed),
            )
            skipped_hub += 1
            continue

        # Partition for prioritization: cells that carry a prior failure
        # marker go to the head of the launch queue under
        # ``--prioritize-failed`` so a code-fix relaunch processes the
        # cells the fix targeted before any other work that could trigger
        # a second mid-run dispatcher kill.
        if (
            args.resume
            and args.prioritize_failed
            and cell_has_failure_marker(args.slab_root, cell_key, source, seed)
        ):
            failed_retry_jobs.append((cell_key, source, seed))
        else:
            fresh_jobs.append((cell_key, source, seed))

    if args.resume and args.prioritize_failed and failed_retry_jobs:
        log.warning(
            "%d cell(s) have prior factor_screen_failed.json markers; "
            "launching them BEFORE fresh cells (override with "
            "--no-prioritize-failed). Failed cells: %s",
            len(failed_retry_jobs),
            ", ".join(f"cell_{c}/source_{s}/seed_{n}" for c, s, n in failed_retry_jobs),
        )

    launch_order = failed_retry_jobs + fresh_jobs
    queued = 0
    for cell_key, source, seed in launch_order:
        cmd = _training_cmd(
            cell_key=cell_key,
            source=source,
            seed=seed,
            pool_dir=args.pool_dir,
            slab_root=args.slab_root,
            resume=args.resume,
        )
        if args.dry_run:
            log.info("DRYRUN: %s", " ".join(cmd))
            queued += 1
            continue
        gpu = _wait_for_free_gpu(running, gpu_pool)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log.info("Launching cell=%s source=%s seed=%d on GPU %d", cell_key, source, seed, gpu)
        running[gpu] = subprocess.Popen(cmd, env=env)
        queued += 1

    if args.resume:
        log.info(
            "Resume summary: %d skipped (disk) + %d skipped (hub) + "
            "%d retried-failed + %d fresh = %d total",
            skipped_disk,
            skipped_hub,
            len(failed_retry_jobs),
            len(fresh_jobs),
            len(jobs),
        )

    if args.dry_run:
        return 0

    # Drain.
    while running:
        gpu = _wait_for_free_gpu(running, gpu_pool)
        running.pop(gpu, None)
    log.info(
        "Training stage complete: %d jobs (%d skipped, %d ran)",
        len(jobs),
        skipped_disk + skipped_hub,
        queued,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _build_arg_parser().parse_args(argv)

    if not args.skip_pool_stage:
        rc = _pool_stage(args)
        if rc != 0:
            log.error("Pool stage failed (rc=%d); aborting before training", rc)
            return rc
    else:
        log.info("Skipping pool stage as requested")

    return _training_stage(args)


if __name__ == "__main__":
    sys.exit(main())
