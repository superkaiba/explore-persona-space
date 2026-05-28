#!/usr/bin/env python3
"""Wave-based Phase-C eval dispatcher for task #396.

Phase-C analog of ``scripts/launch_issue396.py`` (the Phase-B training
dispatcher). Each per-source eval is invoked via
``scripts/eval_issue396_logprob.py`` and runs the 48 eval-persona x 20
question matrix on ONE source LoRA. This script fans those per-source
evals across the available GPUs in waves and handles the per-wave HF
snapshot download + post-wave snapshot cleanup that the single-source
eval script itself does not own.

Contract:

* **Sources.** Defaults to the 24 INHERITED-set sources (the personas
  whose Phase-B training landed in this round per the brief). Validated
  against ``INHERITED_SOURCES_24`` from ``scripts/analyze_length_rate_n48``.
  Use ``--sources A,B,C`` to override.

* **Concurrency shape.** ``--max-parallel`` chunks the source list
  (default 1, max 4 on the 4xH100 pod). At ``--max-parallel 1`` (default),
  sources run sequentially one-at-a-time cycling GPU 0→1→2→3→0…
  (~24 chunks x ~15 min ≈ 6 h wall). At ``--max-parallel 4``, the
  original wave-of-4 behaviour is preserved (~6 waves x ~15 min ≈ 1.5 h);
  this faster mode is opt-in because Phase-C round-1 surfaced an
  inter-wave state-coupling bug under parallel execution (see
  ``epm:failure v1`` at 2026-05-27T21:17:10Z) and sequential is the
  reliable default until that root cause is pinned down. Each chunk's
  subprocesses launch in parallel via ``subprocess.Popen`` +
  ``proc.wait()``, mirroring ``launch_issue396.py``.

* **Per-source command.** ::

      CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED={seed}
      uv run python scripts/eval_issue396_logprob.py
          --source {source}
          --merged-model-path {snapshot_dir}
          --seed {seed}

  where ``{snapshot_dir}`` is the local merged-model directory the
  dispatcher pre-downloads before spawning the subprocess. ``vLLM``'s
  ``LLM(model=...)`` does not accept a ``subfolder=`` kwarg, so the
  merged subfolder MUST be materialized as a flat local directory and
  the eval script gets a path to that flat dir.

* **Per-wave snapshot download.** Each wave's 4 merged-model subfolders
  are pulled from HF (``superkaiba1/explore-persona-space``) into local
  dirs BEFORE the subprocesses spawn. We do NOT use
  ``snapshot_download(allow_patterns=...)`` because that combo is known
  broken for subfolder paths in this project's pinned ``huggingface_hub``
  version (see project memory ``feedback_snapshot_download_siblings_truncation``
  and the equivalent comment in ``scripts/eval_marker_spread_source_only.py``).
  Instead: ``list_repo_files`` + per-file ``hf_hub_download`` with
  ``local_dir`` set to the snapshot root. This is the same pattern
  ``scripts/run_dose_response_cell.py`` uses for the same problem.

* **Resume-safe ``is_done(source)`` gate.** Skip a source whose
  ``eval_results/issue_396/logprob_{source}_seed{seed}.json`` exists and
  reports ``n_cells == 48 * 20``. Mirrors the Phase-B launcher's gate
  verbatim so a partially-finished run can be re-invoked without losing
  per-source progress.

* **Post-chunk snapshot cleanup.** Once a chunk's subprocesses return 0
  AND the per-source eval-JSON checkpoint is present locally (proving
  the eval script's internal upload succeeded — it raises on upload
  failure per CLAUDE.md "fail-loud" rule), ``shutil.rmtree`` the chunk's
  snapshot dirs. Each snapshot is ~15 GB; peak per-chunk disk =
  ``max_parallel x 15 GB`` (15 GB at default ``--max-parallel 1``; 60 GB
  at opt-in ``--max-parallel 4``), well under the 130 GB RunPod MooseFS
  per-pod quota.

Pinned constants (plan v2.3 §10 Reproducibility Card): marker text
``' ※'`` (Qwen id 83399), neg-set ``asst_excluded``, prompt-length
``medium``, slug ``c0589c``. Used only to construct the HF subfolder
path; the eval script's MARKER_TEXT comes from its own module constant
and is verified to match via ``tests/test_issue396_launch_marker_assertion``.

Task #396 plan v2.3 §3 (Phase B -> Phase C diagram) + §4.5
(per-source primitive). PR #390 / branch ``issue-396``.
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

# scripts/ on path so we can import the canonical 24-source list.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from analyze_length_rate_n48 import INHERITED_SOURCES_24  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "issue_396"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "issue_396"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Local snapshot root. Each per-source snapshot lands at
# ``SNAPSHOT_ROOT/marker_<source>_asst_excluded_medium_c0589c_seed42/``,
# matching the HF subfolder layout. The dispatcher pulls into this root
# and passes the per-source subdir as ``--merged-model-path`` to the
# eval script. Lives under ``eval_results/issue_396/_snapshots`` so the
# already-mkdir'd parent works on a fresh pod.
SNAPSHOT_ROOT = EVAL_RESULTS_DIR / "_snapshots"
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)

# HF model repo holding the 24 merged source-LoRA checkpoints. See plan
# §10 + the Phase-B launcher's HF_MODEL_REPO constant.
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# Marker / recipe identifiers — eval-side, used ONLY to construct the HF
# subfolder path. The eval script's own MARKER_TEXT (in
# scripts/eval_issue396_logprob.py) is the canonical eval-time constant;
# this dispatcher never sees the marker token itself.
MARKER_SLUG = "c0589c"  # marker_slug(" ※") from explore_persona_space.personas
NEG_SET = "asst_excluded"
PROMPT_LENGTH = "medium"
SEED = 42

# Per-cell budget: 48 eval personas x 20 eval questions = 960 cells per source.
# Matches the eval script's invariant assertion at
# ``scripts/eval_issue396_logprob.py:568``.
EXPECTED_N_CELLS = 48 * 20


def _hf_subfolder(source: str, seed: int) -> str:
    """HF subfolder for one source's merged checkpoint (plan v2.3 §10)."""
    return f"leakage_experiment/marker_{source}_{NEG_SET}_{PROMPT_LENGTH}_{MARKER_SLUG}_seed{seed}"


def _snapshot_dir(source: str, seed: int) -> Path:
    """Local dir into which the per-source merged checkpoint is pulled."""
    return SNAPSHOT_ROOT / f"marker_{source}_{NEG_SET}_{PROMPT_LENGTH}_{MARKER_SLUG}_seed{seed}"


def is_done(source: str, seed: int = SEED) -> bool:
    """Resume-safe gate: skip if per-source trajectory eval JSON is complete.

    Mirrors ``launch_issue396.py::is_done`` verbatim — both phases write
    to the SAME artifact path (``eval_results/issue_396/logprob_{source}
    _seed{seed}.json``), and the artifact is well-formed when its
    ``n_cells`` field equals ``48 * 20 == 960`` (one eval persona x one
    eval question per cell).
    """
    path = EVAL_RESULTS_DIR / f"logprob_{source}_seed{seed}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("[%s] eval JSON exists but is malformed (%s) — re-running", source, e)
        return False
    if data.get("n_cells") == EXPECTED_N_CELLS:
        return True
    logger.warning(
        "[%s] eval JSON exists with n_cells=%r (expected %d) — re-running",
        source,
        data.get("n_cells"),
        EXPECTED_N_CELLS,
    )
    return False


def build_cmd(source: str, gpu: int, seed: int, merged_model_path: Path) -> str:
    """Build the bash command line for one source's per-source eval subprocess.

    ``CUDA_VISIBLE_DEVICES={gpu}`` masks the GPU for vLLM (Phase 1) and HF
    Transformers (Phase 2). The eval script does NOT clobber CVD internally
    the way ``train_lora`` does (the eval path has no Hydra ``gpu_id``
    schema), so a single CVD mask is sufficient. The eval script's
    ``phase2_trajectory_logprobs`` hardcodes ``device="cuda:0"`` — that is
    the FIRST CVD-visible GPU per the canonical CVD/torch contract, NOT
    physical GPU 0.

    ``PYTHONUNBUFFERED=1`` keeps subprocess stdout flushed line-by-line into
    the per-source log so progress shows up live. ``PYTHONHASHSEED={seed}``
    pins the dict-iteration order downstream so prompt-building order
    matches the eval script's invariant.

    The eval script's ``--merged-model-path`` is the absolute path to the
    local merged-checkpoint snapshot dir (pre-downloaded by this
    dispatcher before the wave spawns).
    """
    return (
        f"CUDA_VISIBLE_DEVICES={gpu} PYTHONUNBUFFERED=1 PYTHONHASHSEED={seed} "
        f"uv run python scripts/eval_issue396_logprob.py "
        f"--source {source} "
        f"--merged-model-path {merged_model_path} "
        f"--seed {seed}"
    )


def download_merged_checkpoint(source: str, seed: int = SEED) -> Path:
    """Pull one merged source-LoRA from HF into ``SNAPSHOT_ROOT``.

    Uses ``list_repo_files`` + per-file ``hf_hub_download`` instead of
    ``snapshot_download(allow_patterns=...)``. The latter is documented
    broken for subfolder paths in this project's pinned ``huggingface_hub``
    (see ``scripts/eval_marker_spread_source_only.py:152-154`` and
    ``scripts/run_dose_response_cell.py:171-172`` for the same workaround).

    Returns the local path containing ``config.json`` + safetensors +
    tokenizer — the dir that should be passed verbatim as
    ``--merged-model-path`` to the eval script.

    Resume-safe: if the snapshot dir already contains ``config.json``,
    skip re-download.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    snap_dir = _snapshot_dir(source, seed)
    hf_subfolder = _hf_subfolder(source, seed)

    if (snap_dir / "config.json").exists():
        logger.info("[%s] snapshot already present at %s — skipping download", source, snap_dir)
        return snap_dir

    snap_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the repo, filter to the per-source subfolder. We
    # use ``list_repo_files`` (the canonical workaround) which the project
    # memory ``feedback_snapshot_download_siblings_truncation`` notes is
    # the right tool for repos large enough to trigger siblings truncation.
    repo_files = list_repo_files(repo_id=HF_MODEL_REPO, repo_type="model")
    prefix = hf_subfolder + "/"
    per_source_files = [f for f in repo_files if f.startswith(prefix)]
    if not per_source_files:
        raise RuntimeError(
            f"[{source}] no files under HF prefix {hf_subfolder!r} in {HF_MODEL_REPO}. "
            "The Phase-B upload either did not run for this source, or the "
            "subfolder slug is wrong. Plan v2.3 §10 Reproducibility Card pins "
            f"the slug to {MARKER_SLUG!r} (marker_slug(' ※')); confirm the "
            "training-side launcher uploaded to the matching subfolder."
        )

    logger.info(
        "[%s] downloading %d files from HF://%s/%s -> %s",
        source,
        len(per_source_files),
        HF_MODEL_REPO,
        hf_subfolder,
        snap_dir,
    )

    # ``hf_hub_download`` with ``local_dir`` preserves the in-repo path
    # under ``local_dir``. We point ``local_dir`` at ``SNAPSHOT_ROOT`` so
    # the file ``leakage_experiment/marker_<...>/config.json`` lands at
    # ``SNAPSHOT_ROOT/leakage_experiment/marker_<...>/config.json``, then
    # we resolve the per-source subdir below.
    #
    # Retry-with-backoff (3 attempts, 30s/60s/120s) on transient HF Hub
    # / network errors. The naked single-shot download in the previous
    # revision is suspected to be what killed the launcher silently mid
    # Wave-3 download on task #396 (2026-05-27): an uncaught
    # ``HfHubHTTPError`` (or ``OSError`` on a half-written file) would
    # propagate up and out, and the absence of per-file completion
    # logging made the failure invisible.
    # Inline imports: a top-level ``import time`` gets stripped by ruff
    # if it is not referenced at module scope, and ``HfHubHTTPError`` is
    # only used inside this except-clause.
    import time as _time

    from huggingface_hub.errors import HfHubHTTPError

    download_root = SNAPSHOT_ROOT
    MAX_ATTEMPTS = 3
    for idx, fname in enumerate(per_source_files):
        last_exc: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                hf_hub_download(
                    repo_id=HF_MODEL_REPO,
                    filename=fname,
                    local_dir=str(download_root),
                )
                last_exc = None
                break
            except (HfHubHTTPError, OSError, ConnectionError) as e:
                last_exc = e
                # Don't sleep after the last attempt — we're about to raise.
                if attempt == MAX_ATTEMPTS - 1:
                    logger.warning(
                        "[%s] hf_hub_download(%s) attempt %d/%d failed (%s) — exhausted retries",
                        source,
                        fname,
                        attempt + 1,
                        MAX_ATTEMPTS,
                        e,
                    )
                    break
                wait = 30 * (2**attempt)  # 30s, 60s
                logger.warning(
                    "[%s] hf_hub_download(%s) attempt %d/%d failed (%s) — retrying in %ds",
                    source,
                    fname,
                    attempt + 1,
                    MAX_ATTEMPTS,
                    e,
                    wait,
                )
                _time.sleep(wait)
        if last_exc is not None:
            raise RuntimeError(
                f"[{source}] hf_hub_download exhausted 3 retries for {fname!r}: {last_exc}"
            ) from last_exc
        logger.info(
            "[%s] downloaded %d/%d: %s",
            source,
            idx + 1,
            len(per_source_files),
            fname,
        )

    # Resolve the actual landed path: ``SNAPSHOT_ROOT/leakage_experiment/marker_<source>_<...>/``.
    landed = download_root / hf_subfolder
    if not (landed / "config.json").exists():
        raise RuntimeError(
            f"[{source}] downloaded {len(per_source_files)} files from "
            f"HF://{HF_MODEL_REPO}/{hf_subfolder} but config.json is missing "
            f"at {landed}. Investigate the upload (sharded safetensors without "
            "config = unloadable checkpoint)."
        )

    # Standardize the snapshot dir layout: move the landed subdir to the
    # pre-computed ``_snapshot_dir(source)`` so the rest of this script
    # only has to think about one path shape. If they're already the
    # same, no-op.
    if landed.resolve() != snap_dir.resolve():
        # ``snap_dir`` may already exist (we mkdir'd above); rename only
        # works if dest is empty.
        if snap_dir.exists() and not (snap_dir / "config.json").exists():
            shutil.rmtree(snap_dir)
        landed.rename(snap_dir)

        # Clean up the now-empty ``SNAPSHOT_ROOT/leakage_experiment/``
        # parent dir left behind by the rename. ``rmdir`` only succeeds
        # if empty, which is exactly the safety we want here: if a
        # parallel wave is still using a sibling subdir, ``rmdir`` fails
        # with ``OSError(ENOTEMPTY)`` and we leave it for that wave to
        # clean up on its own rename.
        with contextlib.suppress(OSError):
            (SNAPSHOT_ROOT / "leakage_experiment").rmdir()

    logger.info("[%s] snapshot ready at %s", source, snap_dir)
    return snap_dir


def cleanup_snapshot(source: str, seed: int = SEED) -> int:
    """Remove the local per-source snapshot dir; return freed bytes for logging.

    Called AFTER the subprocess completes successfully AND the per-source
    eval-JSON checkpoint is present locally (proving the eval script
    finished + uploaded). The 24-source x ~15 GB total = ~360 GB would
    overrun the 130 GB MooseFS per-pod quota without per-wave cleanup; 4
    parallel sources at peak = ~60 GB, safely under quota.
    """
    snap_dir = _snapshot_dir(source, seed)
    if not snap_dir.exists():
        return 0
    freed_bytes = 0
    with contextlib.suppress(Exception):
        freed_bytes = sum(p.stat().st_size for p in snap_dir.rglob("*") if p.is_file())
    shutil.rmtree(snap_dir)
    logger.info(
        "[%s] removed snapshot dir %s (~%.1f GB freed)",
        source,
        snap_dir,
        freed_bytes / 1e9,
    )
    return freed_bytes


def _validate_sources(sources: list[str]) -> list[str]:
    """Validate every source belongs to the 24-source INHERITED set.

    Phase B (training) in this round covered only ``INHERITED_SOURCES_24``;
    Phase C cannot evaluate a source whose merged checkpoint was never
    uploaded. We refuse loudly rather than discover a missing-subfolder
    error mid-wave on the pod.
    """
    unknown = [s for s in sources if s not in INHERITED_SOURCES_24]
    if unknown:
        raise SystemExit(
            f"BLOCKING: --sources contains names not in INHERITED_SOURCES_24: "
            f"{unknown}. Phase B in this round trained only the 24 INHERITED "
            "sources; their merged checkpoints are at "
            f"HF://{HF_MODEL_REPO}/leakage_experiment/marker_<source>_"
            f"{NEG_SET}_{PROMPT_LENGTH}_{MARKER_SLUG}_seed{SEED}/. Re-train "
            "(via scripts/launch_issue396.py) if you need an additional source."
        )
    return sources


def _run_one_chunk(
    chunk: list[str],
    chunk_gpus: list[int],
    chunk_idx: int,
    n_chunks_total: int,
    seed: int,
    dry_run: bool,
    results: dict[str, str],
) -> None:
    """Run one chunk of sources end-to-end (download → spawn → wait → cleanup).

    Mutates ``results`` in place. Extracted out of ``wave_loop`` so the
    outer loop body stays under the ruff McCabe ceiling.
    """
    logger.info(
        "=== Chunk %d / %d (%d source(s) on GPU(s) %s): %s ===",
        chunk_idx,
        n_chunks_total,
        len(chunk),
        chunk_gpus,
        ", ".join(chunk),
    )

    # Pre-download all chunk snapshots. ``dry_run`` skips the actual
    # HF call but still prints the command shape.
    chunk_snapshots: dict[str, Path] = {}
    if dry_run:
        for source in chunk:
            snap_dir = _snapshot_dir(source, seed)
            logger.info(
                "  [%s] DRY-RUN: would download HF://%s/%s -> %s",
                source,
                HF_MODEL_REPO,
                _hf_subfolder(source, seed),
                snap_dir,
            )
            chunk_snapshots[source] = snap_dir
    else:
        for source in chunk:
            chunk_snapshots[source] = download_merged_checkpoint(source, seed=seed)

    # Spawn subprocess(es) — one per source in the chunk.
    procs: list[tuple[str, subprocess.Popen]] = []
    chunk_log_handles: list[tuple] = []
    for slot, source in enumerate(chunk):
        gpu = chunk_gpus[slot]
        cmd = build_cmd(source, gpu, seed, chunk_snapshots[source])
        log_path = LOG_DIR / f"i396_eval_{source}_gpu{gpu}.log"
        logger.info("  [%s] -> GPU %d, log=%s", source, gpu, log_path)
        if dry_run:
            logger.info("    DRY-RUN: %s", cmd)
            continue
        log_handle = open(log_path, "w")  # noqa: SIM115 - closed after wait() below
        chunk_log_handles.append((log_handle, source))
        proc = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        procs.append((source, proc))

    # Block until every subprocess in this chunk returns.
    for source, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            logger.error(
                "[%s] subprocess exited %d — see log %s",
                source,
                proc.returncode,
                LOG_DIR / f"i396_eval_{source}_gpu*.log",
            )
            results[source] = "failed"
            continue

        # Post-success: confirm the per-source eval-JSON checkpoint
        # is actually present locally. The eval script raises on
        # upload failure (CLAUDE.md "fail-loud"), so a clean exit
        # means both write and upload succeeded — but checking the
        # local file too catches any future regression where the
        # subprocess exits 0 without writing.
        if not is_done(source, seed=seed):
            logger.error(
                "[%s] subprocess exited 0 but logprob JSON missing/malformed at %s — "
                "treating as failure",
                source,
                EVAL_RESULTS_DIR / f"logprob_{source}_seed{seed}.json",
            )
            results[source] = "failed"
            continue

        results[source] = "done"

    # Close log handles AFTER wait() to ensure flushed stdout.
    for handle, _src in chunk_log_handles:
        with contextlib.suppress(Exception):
            handle.close()

    # Post-chunk cleanup: remove this chunk's snapshots even if some
    # subprocesses failed. The snapshot dirs are reproducible (a
    # re-invocation of this script will re-download), but the disk
    # would be lost to the next chunk otherwise. Per CLAUDE.md
    # "Fail fast — never hide failures": surface rmtree errors via
    # logger.exception so a real cleanup failure is visible BEFORE
    # the next chunk's snapshot download trips the 130 GB MFS quota.
    if not dry_run:
        for source in chunk:
            try:
                cleanup_snapshot(source, seed=seed)
            except Exception:
                logger.exception(
                    "[%s] post-chunk snapshot cleanup failed — disk may "
                    "not be freed; re-running the dispatcher will "
                    "re-download but the next chunk may hit the MFS quota",
                    source,
                )


def wave_loop(
    sources: list[str],
    n_gpus: int,
    seed: int,
    *,
    dry_run: bool = False,
    max_parallel: int = 1,
) -> dict[str, str]:
    """Run sources in waves of up to ``max_parallel`` concurrent per-source evals.

    ``max_parallel`` controls inter-source concurrency:

    * ``max_parallel = 1`` (the default after the 2026-05-27 fix) — run
      ONE source at a time, cycling the GPU through ``0..n_gpus-1`` across
      successive iterations (round-robin so e.g. on a 4-GPU pod the
      sources land on GPU 0, 1, 2, 3, 0, 1, 2, 3, ...). This eliminates
      the inter-wave HF-cache / vLLM state coupling that caused the
      Wave-2 ``HFValidationError`` cascade on task #396 (see
      ``epm:failure v1`` 2026-05-27): every subprocess gets a clean
      filesystem context, and the orphan-PID safety check inside
      ``eval_issue396_logprob.py`` cannot fire on a peer subprocess
      because there is no peer subprocess. Wall time on the 24-source
      INHERITED set: ~6 h (vs ~1.5 h with ``max_parallel=4``).
    * ``max_parallel > 1`` — original wave behaviour. Chunk pending
      sources into groups of ``max_parallel``; spawn all chunk
      subprocesses in parallel, each pinned to GPU ``0..max_parallel-1``
      within the wave; ``proc.wait()`` per source. Opt-in for users
      willing to trade reliability for throughput once the Wave-2 root
      cause is understood.

    Returns ``{source: "done" | "skipped" | "failed"}``. For each chunk:

    1. Pre-download the chunk's merged-checkpoint snapshots from HF.
    2. Spawn ``len(chunk)`` subprocesses in parallel; each invokes
       ``scripts/eval_issue396_logprob.py --source ... --merged-model-path ...``.
    3. ``proc.wait()`` on each; mark per-source ``done`` / ``failed``
       based on returncode AND presence of the per-source eval-JSON
       checkpoint.
    4. ``cleanup_snapshot`` each of the chunk's snapshot dirs to free MFS
       quota before the next chunk's downloads.

    A top-level ``try / except`` wraps the per-source loop so a
    previously-silent death (e.g. uncaught ``hf_hub_download`` failure on
    a Wave-3 snapshot) surfaces a traceback in the launcher log before
    the launcher exits non-zero. Without this, the orchestrator sees a
    "no process running" state and cannot tell whether the launcher
    finished cleanly or crashed mid-loop.
    """
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1; got {max_parallel}")
    if n_gpus < 1:
        raise ValueError(f"n_gpus must be >= 1; got {n_gpus}")

    pending = [s for s in sources if not is_done(s, seed=seed)]
    already_done = [s for s in sources if is_done(s, seed=seed)]
    results: dict[str, str] = {s: "skipped" for s in already_done}

    if already_done:
        logger.info(
            "Resume: %d / %d sources already complete; running %d new",
            len(already_done),
            len(sources),
            len(pending),
        )

    # Effective chunk size: never schedule more concurrent subprocesses
    # than the pod has GPUs (each subprocess needs an exclusive GPU per
    # the CVD-mask contract in build_cmd).
    effective_concurrency = min(max_parallel, n_gpus)
    n_chunks_total = (len(pending) + effective_concurrency - 1) // effective_concurrency

    logger.info(
        "wave_loop: %d pending source(s); max_parallel=%d, n_gpus=%d, "
        "effective concurrency=%d, %d chunk(s) total",
        len(pending),
        max_parallel,
        n_gpus,
        effective_concurrency,
        n_chunks_total,
    )

    try:
        for chunk_start in range(0, len(pending), effective_concurrency):
            chunk = pending[chunk_start : chunk_start + effective_concurrency]
            chunk_idx = chunk_start // effective_concurrency + 1

            # GPU assignment.
            #
            # * Sequential (``effective_concurrency == 1``) — cycle the
            #   GPU through ``0..n_gpus-1`` round-robin across successive
            #   iterations. ``chunk_start`` is the index of the source
            #   in ``pending``, so ``chunk_start % n_gpus`` is the GPU
            #   for the single-source chunk.
            # * Parallel (``effective_concurrency > 1``) — pin each
            #   source in the chunk to GPU ``0, 1, ..., effective_concurrency - 1``
            #   (same shape as the pre-fix wave loop).
            if effective_concurrency == 1:
                chunk_gpus = [chunk_start % n_gpus]
            else:
                chunk_gpus = list(range(len(chunk)))

            _run_one_chunk(
                chunk=chunk,
                chunk_gpus=chunk_gpus,
                chunk_idx=chunk_idx,
                n_chunks_total=n_chunks_total,
                seed=seed,
                dry_run=dry_run,
                results=results,
            )
    except KeyboardInterrupt:
        # Let Ctrl-C propagate cleanly.
        raise
    except Exception:
        # Top-level guard for previously-silent deaths. ``logger.exception``
        # writes the traceback into the launcher log so the orchestrator
        # can diagnose. Re-raise so the launcher process exits non-zero
        # and the orchestrator sees a real failure (not an "exited 0,
        # no result" mystery).
        logger.exception(
            "wave_loop: unhandled exception in per-source loop; aborting and re-raising"
        )
        raise

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Phase-C trajectory eval wave for task #396 "
            "(per-source eval of the 24 INHERITED-set LoRAs trained in this round)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pod",
        type=str,
        default="epm-issue-396",
        help="Pod identifier (used for log labelling only; no API call).",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=4,
        help=(
            "Number of GPUs on the pod. The effective inter-source concurrency "
            "is min(--n-gpus, --max-parallel). Default 4 (H100 quad)."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Maximum number of source-eval subprocesses to run concurrently. "
            "Default 1 — sequential across cycling GPUs (eliminates the "
            "inter-wave HF-cache / vLLM state coupling that caused 4 of 4 "
            "Wave-2 subprocesses to die with HFValidationError on task #396, "
            "2026-05-27). Pass --max-parallel 4 to opt into the original "
            "wave behaviour."
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of sources (e.g. accountant,librarian). "
            "Default: the 24 INHERITED-set sources from INHERITED_SOURCES_24."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the wave plan and per-source commands; do not download or spawn subprocesses.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed; threads through to PYTHONHASHSEED, HF subfolder slug, and eval-JSON filename.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Source resolution + validation.
    if args.sources is not None:
        requested = [s.strip() for s in args.sources.split(",") if s.strip()]
        sources = _validate_sources(requested)
    else:
        sources = list(INHERITED_SOURCES_24)

    logger.info(
        "Task #396 Phase-C eval dispatcher: %d sources on pod=%s, "
        "n_gpus=%d, max_parallel=%d, seed=%d, dry_run=%s",
        len(sources),
        args.pod,
        args.n_gpus,
        args.max_parallel,
        args.seed,
        args.dry_run,
    )
    logger.info(
        "HF subfolder shape: HF://%s/leakage_experiment/marker_<source>_%s_%s_%s_seed%d/",
        HF_MODEL_REPO,
        NEG_SET,
        PROMPT_LENGTH,
        MARKER_SLUG,
        args.seed,
    )

    # Print the sample wave-1 command so a reader debugging the contract
    # sees the exact subprocess shape (mirrors launch_issue396.py).
    if sources:
        sample_cmd = build_cmd(
            sources[0],
            gpu=0,
            seed=args.seed,
            merged_model_path=_snapshot_dir(sources[0], args.seed),
        )
        logger.info("Sample wave-1 bash command:\n  %s", sample_cmd)

    results = wave_loop(
        sources,
        n_gpus=args.n_gpus,
        seed=args.seed,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
    )

    # Summary
    done = sum(1 for v in results.values() if v == "done")
    skipped = sum(1 for v in results.values() if v == "skipped")
    failed = sum(1 for v in results.values() if v == "failed")
    logger.info(
        "Eval-launcher complete: %d done, %d skipped (already complete), %d failed.",
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
