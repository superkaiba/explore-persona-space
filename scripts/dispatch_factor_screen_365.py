#!/usr/bin/env python3
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
import io
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
        out = subprocess.check_output(
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


def _parse_cell_filter(raw: str) -> list[str]:
    """Parse the ``--cell-filter`` argument into a list of 5-bit cell keys.

    Each key must be exactly 5 characters of '0' or '1' (matches the
    ``Cell.from_key`` format). Crashes loudly on malformed input rather than
    silently dropping bad entries — per CLAUDE.md "Never silently fail".
    """
    keys = [s.strip() for s in raw.split(",") if s.strip()]
    for key in keys:
        if len(key) != 5 or any(c not in "01" for c in key):
            raise argparse.ArgumentTypeError(
                f"Invalid cell key {key!r} in --cell-filter; expected 5 chars of '0'/'1'."
            )
    # Fail loud if the flag was passed but resolves to an empty list
    # (e.g. --cell-filter "" or --cell-filter ","). Otherwise the
    # downstream `if cell_filter:` check would silently fall through to
    # running the full 32-cell set — a smoke-test footgun.
    if not keys:
        raise argparse.ArgumentTypeError(
            f"--cell-filter {raw!r} parsed to an empty list; pass at least one cell key, "
            f"or omit the flag entirely to run the full cell set."
        )
    return keys


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--issue",
        type=int,
        required=True,
        help=(
            "Task/issue number that owns this dispatcher run (e.g. 365 for "
            "the parent factor screen, 383 for the recipe-fix re-run). "
            "Controls (a) the HF Hub resume-probe prefix "
            "(adapters/issue_{issue}/), (b) the prefetch adapter-index "
            "filter, and (c) the --issue arg forwarded to each child "
            "cell-train / cell-eval subprocess. REQUIRED so a "
            "dispatcher invocation can never silently false-skip 72 "
            "cells against another issue's already-populated Hub "
            "namespace (see plan v2 §5a)."
        ),
    )
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
        "--cell-filter",
        type=_parse_cell_filter,
        default=None,
        help=(
            "Comma-separated list of cell keys (e.g. '00000,00010,00100,00110') "
            "to run; all other cells are dropped from the job queue. Used for "
            "smoke testing against a small subset before the full 96-cell run. "
            "Default: no filter (all 32 cells per source)."
        ),
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
            "--issue",
            str(args.issue),
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
        rc = subprocess.call(cmd)
        if rc != 0:
            log.error("Pool stage failed for source=%s (rc=%d)", source, rc)
            return rc
    return 0


def _training_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Enumerate the (cell_key, source, seed) jobs for the training stage.

    When ``--cell-filter`` is set, restrict the cell space to the given keys;
    used for the round-8 smoke test to validate fixes against a 4-cell subset
    before launching the full 96-cell run.

    Round-16 (issue #365): A=0 x C=1 cells are dropped by the C-axis preflight
    (round-3 Jaccard floor 0.15 + round-16 token-equality FAIL on the 5-token
    C0 vs 27-token-minimum C1 template). Pre-filter them out here so the
    dispatcher does not launch impossible cells that would stall in
    ``_wait_for_pool`` for 30 minutes before failing. Cell key encoding is
    ``ABCDE`` with positions A=0, B=1, C=2, D=3, E=4 (5-bit string).
    """
    cells = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=5)]
    # Filter A=0 x C=1 cells (positions [0] and [2] in the cell-key string).
    valid_cells = [c for c in cells if not (c[0] == "0" and c[2] == "1")]
    dropped = [c for c in cells if c not in valid_cells]
    log.info(
        "_training_jobs: pre-filtered %d A=0 x C=1 cells (round-3 + round-16): %s",
        len(dropped),
        dropped,
    )
    cells = valid_cells
    cell_filter = getattr(args, "cell_filter", None)
    if cell_filter:
        cells = [c for c in cells if c in cell_filter]
        log.info("--cell-filter active: restricted to %d cell key(s): %s", len(cells), cells)
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


def hf_hub_adapter_run_name(cell_key: str, source: str, seed: int, *, issue: int = 365) -> str:
    """Return the run-name suffix used by ``training.train_one_cell``.

    Mirrors ``training.train_one_cell``'s ``run_name`` template:
        ``f"i{issue}_cell_{cell.key}_source_{source}_seed{seed}"``
    The adapter lands at ``adapters/issue_{issue}/<run_name>`` in the
    model repo (``superkaiba1/explore-persona-space``).

    Task #383 plumbing (plan v2 §5a): the ``issue`` keyword is required at
    the CLI level (``--issue``) but defaults to 365 here so the historical
    test contract — ``hf_hub_adapter_run_name(cell, src, seed)`` returns
    ``i365_cell_*`` for the parent #365 namespace — is preserved.
    """
    return f"i{issue}_cell_{cell_key}_source_{source}_seed{seed}"


def cell_complete_on_hub(
    cell_key: str,
    source: str,
    seed: int,
    *,
    issue: int = 365,
    hub_files_cache: list[str] | None = None,
) -> bool:
    """Return True if this cell's adapter is already uploaded to HF Hub.

    Probes ``superkaiba1/explore-persona-space`` for any file under
    ``adapters/issue_{issue}/<run_name>/``. Returns False on any HfApi
    error (network down, no token) so the dispatcher falls through to a
    local rebuild rather than silently treating "couldn't reach hub" as
    "skip".

    Parameters
    ----------
    issue:
        Task/issue number; controls the Hub prefix
        ``adapters/issue_{issue}/``. Defaults to 365 for backward compat
        with the historical test contract.
    hub_files_cache:
        Optional pre-fetched list of files in the model repo. When supplied
        we use it instead of probing (one Hub call per dispatcher
        invocation). When ``None`` we probe lazily and crash-loud on errors.
    """
    run_name = hf_hub_adapter_run_name(cell_key, source, seed, issue=issue)
    prefix = f"adapters/issue_{issue}/{run_name}/"
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


def _prefetch_hub_adapter_index(*, issue: int = 365) -> list[str] | None:
    """One-shot probe of the HF Hub model repo for adapter files under one issue.

    Cuts the resume scan from N per-cell HfApi calls to one, then filters
    the result to ``adapters/issue_{issue}/`` so the dispatcher only sees
    artifacts that belong to its own run. Returns ``None`` if the probe
    fails (e.g. no network); callers should treat that as "no hub data"
    rather than crashing.

    Task #383 plumbing (plan v2 §5a): without the issue-scoped filter, a
    dispatcher run for issue 383 would also see parent #365's 72 adapters
    in the returned list and (because every entry shares the
    ``adapters/issue_365/`` prefix, not the requested ``issue_383`` one)
    the per-cell check still returns False — but scoping the prefetch
    keeps the list small and the intent obvious.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        all_files = api.list_repo_files(
            repo_id="superkaiba1/explore-persona-space", repo_type="model"
        )
    except Exception as exc:
        log.warning("HF Hub model-repo index fetch failed (%s); disk-only resume", exc)
        return None
    prefix = f"adapters/issue_{issue}/"
    return [f for f in all_files if f.startswith(prefix)]


def _training_cmd(
    *,
    cell_key: str,
    source: str,
    seed: int,
    issue: int,
    pool_dir: Path,
    slab_root: Path,
    resume: bool = True,
    mode: str = "cell-train",
) -> list[str]:
    """Build the per-phase entry-script argv for one (cell, source, seed) slot.

    Round-14 (issue #365): each cell is executed as TWO sequential
    subprocesses — first ``--mode cell-train`` (loads base, trains LoRA,
    merges, exits so the CUDA driver releases the trainer's reservations),
    then ``--mode cell-eval`` (loads merged via vLLM in a fresh process
    that sees the full free HBM). ``mode`` selects which phase to run;
    both phases share the rest of the argv.

    Task #383 plumbing (plan v2 §5a): ``--issue`` is forwarded to every
    child subprocess so ``train_one_cell`` writes under the correct
    ``adapters/issue_{issue}/`` Hub namespace.
    """
    if mode not in ("cell-train", "cell-eval"):
        raise ValueError(
            f"_training_cmd mode must be 'cell-train' or 'cell-eval'; got {mode!r}. "
            f"See round-14 (issue #365) train/eval split."
        )
    output_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    cmd = [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.factor_screen_365",
        "--mode",
        mode,
        "--issue",
        str(issue),
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


def _cell_log_path(slab_root: Path, cell_key: str, source: str, seed: int) -> Path:
    """Return the per-cell stdout+stderr log path.

    Round-9 (issue #365) Fix D: each training-stage subprocess writes its
    combined stdout+stderr to a dedicated log file under the cell's output
    dir. Round-8 merged all 8 cells' output into one stream, hiding which
    cell hit a vLLM crash and interleaving tqdm progress bars
    incomprehensibly. Anchoring the log next to the cell's metrics.json
    makes it trivial to find ("which cell failed?" → ls the slab tree).
    """
    return (
        slab_root
        / f"cell_{cell_key}"
        / f"source_{source}"
        / f"seed_{seed}"
        / "cell_stdout_stderr.log"
    )


def _wait_for_free_gpu(
    running: dict[int, subprocess.Popen],
    gpu_pool: list[int],
    log_handles: dict[int, io.TextIOBase] | None = None,
    *,
    slot_state: dict[int, dict] | None = None,
    on_phase_complete=None,
) -> int:
    """Wait until any GPU in ``gpu_pool`` has no running subprocess.

    Round-9 (issue #365): also closes the per-cell log file handle when the
    subprocess on that GPU exits, so we don't leak file descriptors across
    a 96-cell run.

    Round-14 (issue #365): when a subprocess exits, calls
    ``on_phase_complete(gpu, slot_state[gpu], rc)`` so the caller can
    decide whether the slot's (cell, source, seed) work is finished or
    whether the next phase (``cell-eval`` after ``cell-train``) needs to
    fire on the same GPU. The callback is responsible for whatever
    bookkeeping it wants (e.g., logging which phase failed); the wait
    function still releases the GPU slot back to the caller, who consults
    ``slot_state`` to decide what to launch next.
    """
    while True:
        for gpu in gpu_pool:
            proc = running.get(gpu)
            if proc is None:
                return gpu
            if proc.poll() is not None:
                # Process finished.
                running.pop(gpu, None)
                rc = proc.returncode
                if rc != 0:
                    state = (slot_state or {}).get(gpu, {})
                    phase = state.get("phase", "?")
                    cell_key = state.get("cell_key", "?")
                    source = state.get("source", "?")
                    seed = state.get("seed", "?")
                    log.warning(
                        "Job on GPU %d exited with rc=%d (phase=%s cell=%s source=%s seed=%s)",
                        gpu,
                        rc,
                        phase,
                        cell_key,
                        source,
                        seed,
                    )
                if log_handles is not None:
                    handle = log_handles.pop(gpu, None)
                    if handle is not None:
                        try:
                            handle.close()
                        except OSError as exc:
                            log.warning("Failed to close per-cell log handle: %s", exc)
                if on_phase_complete is not None and slot_state is not None:
                    state = slot_state.get(gpu, {})
                    on_phase_complete(gpu, state, rc)
                return gpu
        time.sleep(2)


def _launch_phase(
    *,
    phase: str,
    cell_key: str,
    source: str,
    seed: int,
    gpu: int,
    args: argparse.Namespace,
    running: dict[int, subprocess.Popen],
    log_handles: dict[int, io.TextIOBase],
    slot_state: dict[int, dict],
) -> None:
    """Launch one phase (cell-train or cell-eval) on the given free GPU slot.

    Round-14 (issue #365): each (cell, source, seed) slot is executed as
    two sequential phases. This helper handles the common open-log /
    Popen / state-bookkeeping for both.
    """
    cmd = _training_cmd(
        cell_key=cell_key,
        source=source,
        seed=seed,
        issue=args.issue,
        pool_dir=args.pool_dir,
        slab_root=args.slab_root,
        resume=args.resume,
        mode=phase,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Round-9 Fix D: open a per-cell log file (line-buffered) and redirect
    # the subprocess's stdout+stderr into it. Both phases write to the SAME
    # per-cell log in append mode — round-14 keeps the round-9 convention
    # of one log per (cell, source, seed). Pre-create the parent dir so the
    # entry script's own output_dir.mkdir(parents=True, exist_ok=True) is a
    # no-op.
    cell_log = _cell_log_path(args.slab_root, cell_key, source, seed)
    cell_log.parent.mkdir(parents=True, exist_ok=True)
    # SIM115: ``open()`` here is intentional — the handle must outlive
    # this function call and is closed in ``_wait_for_free_gpu`` when
    # the subprocess exits.
    log_handle = open(cell_log, "a", buffering=1)  # noqa: SIM115
    log_handle.write(f"=== cell-{phase} start: gpu={gpu} ===\n")
    log_handle.flush()
    log.info(
        "Launching phase=%s cell=%s source=%s seed=%d on GPU %d (log=%s)",
        phase,
        cell_key,
        source,
        seed,
        gpu,
        cell_log,
    )
    running[gpu] = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    log_handles[gpu] = log_handle
    slot_state[gpu] = {
        "phase": phase,
        "cell_key": cell_key,
        "source": source,
        "seed": seed,
    }


def _resolve_num_gpus(args: argparse.Namespace) -> None:
    """Resolve ``args.num_gpus`` against the physical GPU count.

    Auto-detect when ``--num-gpus`` was omitted, clamp when it exceeds the
    physical count (launching subprocesses against phantom GPUs caused the
    round-5 silent dispatcher death; see issue #365 round-6 forensics).
    """
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


def _should_skip_cell(
    cell_key: str,
    source: str,
    seed: int,
    args: argparse.Namespace,
    hub_files: list[str] | None,
) -> str | None:
    """Return a non-empty skip reason if this cell is already complete.

    Used by the round-5 resume probe. Returns ``"disk"`` when local
    artifacts already exist, ``"hub"`` when the adapter is already on
    the model repo, or ``None`` when the cell must be run.

    Task #383 plumbing (plan v2 §5a): ``args.issue`` selects the Hub
    namespace probed. For a fresh issue-383 dispatcher invocation the
    namespace ``adapters/issue_383/`` is empty at launch, so no cell
    will false-skip against parent #365's already-populated adapters.
    """
    if not args.resume:
        return None
    if cell_complete_on_disk(args.slab_root, cell_key, source, seed):
        output_dir = _cell_output_dir(args.slab_root, cell_key, source, seed)
        log.info("Cell already complete on disk -- skipping; results at %s", output_dir)
        return "disk"
    if not args.skip_hub_probe and cell_complete_on_hub(
        cell_key, source, seed, issue=args.issue, hub_files_cache=hub_files
    ):
        log.info(
            "Cell already complete on HF Hub -- skipping; adapter at "
            "superkaiba1/explore-persona-space/adapters/issue_%d/%s/",
            args.issue,
            hf_hub_adapter_run_name(cell_key, source, seed, issue=args.issue),
        )
        return "hub"
    return None


def _log_dry_run_phases(cell_key: str, source: str, seed: int, args: argparse.Namespace) -> None:
    """Log both phases the dispatcher would launch under round-14 (issue #365)."""
    for phase in ("cell-train", "cell-eval"):
        dry_cmd = _training_cmd(
            cell_key=cell_key,
            source=source,
            seed=seed,
            issue=args.issue,
            pool_dir=args.pool_dir,
            slab_root=args.slab_root,
            resume=args.resume,
            mode=phase,
        )
        log.info("DRYRUN [%s]: %s", phase, " ".join(dry_cmd))


def _drain_pending_eval(
    gpu: int,
    *,
    pending_eval: dict[int, tuple[str, str, int]],
    args: argparse.Namespace,
    running: dict[int, subprocess.Popen],
    log_handles: dict[int, io.TextIOBase],
    slot_state: dict[int, dict],
    on_phase_complete,
) -> int:
    """Fire any pending cell-eval phases queued for this GPU slot, returning the next free GPU.

    A freed GPU may still owe an eval for the cell whose train just
    completed. The main scheduler hands the eval to ``_launch_phase``
    here before launching the next cell-train.
    """
    while gpu in pending_eval:
        eval_cell, eval_source, eval_seed = pending_eval.pop(gpu)
        _launch_phase(
            phase="cell-eval",
            cell_key=eval_cell,
            source=eval_source,
            seed=eval_seed,
            gpu=gpu,
            args=args,
            running=running,
            log_handles=log_handles,
            slot_state=slot_state,
        )
        gpu = _wait_for_free_gpu(
            running,
            gpu_pool=list(range(args.num_gpus)),
            log_handles=log_handles,
            slot_state=slot_state,
            on_phase_complete=on_phase_complete,
        )
    return gpu


def _training_stage(args: argparse.Namespace) -> int:
    jobs = _training_jobs(args)
    _resolve_num_gpus(args)
    gpu_pool = list(range(args.num_gpus))
    running: dict[int, subprocess.Popen] = {}
    # Round-9 (issue #365) Fix D: per-cell open file handles, keyed by GPU.
    log_handles: dict[int, io.TextIOBase] = {}
    # Round-14 (issue #365): per-slot state so the train→eval handoff knows
    # which (cell, source, seed) just exited and whether the cell still owes
    # an eval phase.
    slot_state: dict[int, dict] = {}

    # Round-5: pre-fetch the HF Hub model-repo file index once so the resume
    # probe doesn't make 96 separate Hub calls. None = probe failed / Hub
    # skipped; treat as "no hub data" and fall back to disk-only resume.
    hub_files: list[str] | None = None
    if args.resume and not args.skip_hub_probe:
        hub_files = _prefetch_hub_adapter_index(issue=args.issue)

    # Round-14 (issue #365): when a train phase exits cleanly (rc=0), queue
    # the matching eval phase BEFORE giving the GPU slot back to the
    # main loop. ``pending_eval[gpu]`` is the next phase to fire on that
    # GPU once the wait function returns it.
    pending_eval: dict[int, tuple[str, str, int]] = {}

    def _on_phase_complete(gpu: int, state: dict, rc: int) -> None:
        """Callback fired when a subprocess on ``gpu`` exits.

        Records whether the cell still owes an eval phase so the main loop
        can launch it on the same GPU slot. A failed train phase skips the
        cell (no eval) — the cell goes down as a failure and the slot is
        free for the next pending job.
        """
        phase = state.get("phase")
        if phase == "cell-train" and rc == 0:
            pending_eval[gpu] = (
                state["cell_key"],
                state["source"],
                state["seed"],
            )
        elif phase == "cell-train" and rc != 0:
            log.warning(
                "cell-train phase failed for cell=%s source=%s seed=%s (rc=%d); "
                "skipping cell-eval — no merged checkpoint to evaluate.",
                state.get("cell_key"),
                state.get("source"),
                state.get("seed"),
                rc,
            )
        # rc != 0 in eval phase: nothing else to do; cell is recorded as failed.

    skipped_disk = 0
    skipped_hub = 0
    queued = 0
    for cell_key, source, seed in jobs:
        skip = _should_skip_cell(cell_key, source, seed, args, hub_files)
        if skip == "disk":
            skipped_disk += 1
            continue
        if skip == "hub":
            skipped_hub += 1
            continue

        if args.dry_run:
            _log_dry_run_phases(cell_key, source, seed, args)
            queued += 1
            continue

        gpu = _wait_for_free_gpu(
            running,
            gpu_pool,
            log_handles=log_handles,
            slot_state=slot_state,
            on_phase_complete=_on_phase_complete,
        )
        gpu = _drain_pending_eval(
            gpu,
            pending_eval=pending_eval,
            args=args,
            running=running,
            log_handles=log_handles,
            slot_state=slot_state,
            on_phase_complete=_on_phase_complete,
        )
        _launch_phase(
            phase="cell-train",
            cell_key=cell_key,
            source=source,
            seed=seed,
            gpu=gpu,
            args=args,
            running=running,
            log_handles=log_handles,
            slot_state=slot_state,
        )
        queued += 1

    if args.resume:
        log.info(
            "Resume summary: %d skipped (disk) + %d skipped (hub) + %d queued = %d total",
            skipped_disk,
            skipped_hub,
            queued,
            len(jobs),
        )

    if args.dry_run:
        return 0

    # Drain. Round-14 (issue #365): the drain loop must also flush any
    # pending eval phases that were queued by the last train phases. We
    # keep draining until BOTH ``running`` and ``pending_eval`` are empty.
    while running or pending_eval:
        gpu = _wait_for_free_gpu(
            running,
            gpu_pool,
            log_handles=log_handles,
            slot_state=slot_state,
            on_phase_complete=_on_phase_complete,
        )
        # Launch any pending eval on the freed slot, then wait again.
        if gpu in pending_eval:
            eval_cell, eval_source, eval_seed = pending_eval.pop(gpu)
            _launch_phase(
                phase="cell-eval",
                cell_key=eval_cell,
                source=eval_source,
                seed=eval_seed,
                gpu=gpu,
                args=args,
                running=running,
                log_handles=log_handles,
                slot_state=slot_state,
            )
    # Defensive: close any stragglers (should be empty after the drain loop,
    # but a buggy code path that pops from ``running`` directly could leave
    # handles behind).
    for gpu, handle in list(log_handles.items()):
        try:
            handle.close()
        except OSError as exc:
            log.warning("Failed to close per-cell log handle on GPU %d: %s", gpu, exc)
        log_handles.pop(gpu, None)
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
