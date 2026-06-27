"""Wave-parallel fleet dispatcher helper (issue #676).

Extracts and generalizes the wave-parallel cell-fan-out pattern written three
times in-tree (``scripts/issue651_dispatch.py``, ``issue667_dispatch.py``,
``issue502_dispatch.py``) so dispatchers like ``issue664_dispatch.py`` can run
independent cells across all GPUs of an N-GPU pod, overlap CPU/API work
(judging) with GPU work, and keep the single-GPU + smoke path byte-for-byte
unchanged.

Two layers, both reusable:

- :class:`WaveDispatcher` (+ :func:`assign_gpu_ids`, :func:`run_parallel_with_log`,
  :class:`CellCmd`, :class:`FleetResult`) — the GPU SCHEDULING layer. The caller
  owns what one cell DOES (train / extract / eval) via a per-cell command
  builder; the dispatcher only schedules independent cells into waves of
  ``n_gpus`` subprocesses, each pinned to one GPU via ``CUDA_VISIBLE_DEVICES``
  in the LAUNCHER environment (the gotchas.md cuInit-freeze pin — the in-process
  clobber alone is silently defeated by import-time cuInit, incidents
  #523/#543/#545). Smoke == ``n_gpus=1`` with one cell through the SAME path:
  no in-process-vs-subprocess divergence.

- :func:`submit_judge_async` / :class:`JudgeHandle` — the judge-overlap layer.
  A thin scheduling wrapper over the existing, hardened
  ``eval.batch_judge`` client: submit a cell's judge set to the Anthropic Batch
  API fire-and-forget right after that cell's generations land, then move the
  GPU to the next cell; reconcile the judge result LATER, off the GPU critical
  path, via :meth:`JudgeHandle.reconcile`. NO change to ``batch_judge.py`` — the
  judge is already async at the API level; #676 changes only the SCHEDULING.

The new helper introduces NO new parallelism mechanism: ``run_parallel_with_log``
is a verbatim extraction of ``issue651_dispatch.py``'s ``_run_parallel_with_log``
(subprocess fan-out + per-cell log tee), and the wave loop matches
``issue667_dispatch.py``'s ``for wave_start in range(0, len(cells), n_par)`` +
per-wave ``i % n_par`` densification.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # the caller's cell type (e.g. issue664_common.Cell)


class DuplicateCellError(ValueError):
    """Raised when two cells in a fleet share a ``cell_key`` (disjoint-claim violation).

    Subclasses ``ValueError`` (a duplicate key is a caller programming error /
    malformed grid). Carries the colliding keys so the message names them.
    """

    def __init__(self, colliding_keys: Sequence[str]) -> None:
        self.colliding_keys: list[str] = list(colliding_keys)
        super().__init__(
            "duplicate cell_key(s) in the fleet — no two workers may claim the same "
            f"cell (every output path is key-derived): {sorted(set(self.colliding_keys))}"
        )


class WaveFailedError(RuntimeError):
    """Raised when one or more cells in a wave exited non-zero.

    Carries ``failures`` — a list of ``(rc, cell_key)`` tuples for the failing
    cells, mirroring ``issue651_dispatch.py``'s wave failure surface.
    """

    def __init__(self, failures: Sequence[tuple[int, str]]) -> None:
        self.failures: list[tuple[int, str]] = list(failures)
        super().__init__(
            "wave-parallel cell(s) exited non-zero (rc, cell_key): "
            + ", ".join(f"({rc}, {key})" for rc, key in self.failures)
        )


def assign_gpu_ids(n_cells: int, n_gpus: int) -> list[int]:
    """Round-robin GPU id per cell: cell ``i`` -> ``i % max(n_gpus, 1)``.

    Matches #651's per-wave densification (``issue651_dispatch.py:677``) and
    #667's ``i % n_par`` (``issue667_dispatch.py:725``). ``n_gpus <= 1`` collapses
    every cell onto GPU 0 (the single-GPU / smoke path).
    """
    n = max(n_gpus, 1)
    return [i % n for i in range(n_cells)]


@dataclass(frozen=True)
class CellCmd:
    """One cell's launch spec for the subprocess fan-out path.

    ``env`` MUST contain ``CUDA_VISIBLE_DEVICES`` — :func:`run_parallel_with_log`
    asserts it before launching anything, turning the historical silent
    co-location-on-GPU-0 OOM (#523/#543/#545) into a loud pre-launch failure.
    ``gpu_id`` records the GPU this cell was assigned (== ``env``'s CVD); it is
    the belt-and-suspenders pin (the caller also threads it as ``--gpu-id`` /
    ``+gpu_id`` so the in-process clobber rewrites the SAME value).
    """

    cell_key: str  # the cell's unique idempotency key (e.g. eval_key)
    argv: Sequence[str]  # the subprocess command (argv[0] = executable)
    env: dict[str, str]  # extra env vars; MUST contain CUDA_VISIBLE_DEVICES
    log_path: Path  # per-cell stdout+stderr log (append mode)
    gpu_id: int  # the assigned GPU (== env["CUDA_VISIBLE_DEVICES"])


@dataclass(frozen=True)
class FleetResult:
    """The outcome of a :meth:`WaveDispatcher.run` call."""

    ran: list[str] = field(default_factory=list)  # cell_keys actually launched
    skipped: list[str] = field(default_factory=list)  # cell_keys skipped as already-done
    failures: list[tuple[int, str]] = field(
        default_factory=list
    )  # (rc, cell_key); empty on success
    wave_count: int = 0  # number of waves actually run


def run_parallel_with_log(cmds: Sequence[CellCmd], *, cwd: Path | None = None) -> list[int]:
    """Run the given cell subprocesses CONCURRENTLY; return rc per cell (parallel order).

    Verbatim-extracted from ``issue651_dispatch.py:178-206`` (``_run_parallel_with_log``),
    adapted to the :class:`CellCmd` carrier. Each cell's stdout+stderr tees to its
    own ``log_path`` (append mode). Before launching ANY subprocess the helper
    ASSERTS that ``CUDA_VISIBLE_DEVICES`` is present in every cell's ``env`` — the
    gotchas.md launcher-env pin; an in-process clobber alone is defeated by
    import-time cuInit, so a missing pin would silently co-locate every cell on
    physical GPU 0 and OOM. Raising here makes that a LOUD pre-launch failure.

    The wave's disjoint-claim check (no two cells in the same wave share a
    ``cell_key``) is enforced by :class:`WaveDispatcher` over the WHOLE fleet
    before any wave runs; this function additionally guards the per-wave invariant
    so a direct caller can't co-launch two cells claiming the same key.
    """
    # Pre-launch CVD-present assert (gotchas.md launcher-env pin, #523/#543/#545).
    missing = [c.cell_key for c in cmds if "CUDA_VISIBLE_DEVICES" not in c.env]
    if missing:
        raise AssertionError(
            "CUDA_VISIBLE_DEVICES not pinned in the launcher env for cell(s) "
            f"{missing} — the in-process clobber alone is defeated by import-time "
            "cuInit (#523/#543/#545); build_cmd must set env['CUDA_VISIBLE_DEVICES']"
        )
    # Per-wave disjoint-claim guard (no two concurrent cells write the same paths).
    keys = [c.cell_key for c in cmds]
    if len(set(keys)) != len(keys):
        dupes = [k for k in keys if keys.count(k) > 1]
        raise DuplicateCellError(dupes)

    procs: list[subprocess.Popen] = []
    files = []
    for c in cmds:
        env = {**os.environ, **c.env}
        c.log_path.parent.mkdir(parents=True, exist_ok=True)
        f = c.log_path.open("ab")
        files.append(f)
        logger.info(
            "$ (parallel) %s  >>> %s (CVD=%s)",
            " ".join(shlex.quote(str(a)) for a in c.argv),
            c.log_path,
            c.env.get("CUDA_VISIBLE_DEVICES"),
        )
        p = subprocess.Popen(
            [str(a) for a in c.argv],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
        procs.append(p)
    rcs = [p.wait() for p in procs]
    for f in files:
        f.close()
    for rc, c in zip(rcs, cmds, strict=True):
        if rc != 0:
            logger.error("cell %s exited rc=%d (log: %s)", c.cell_key, rc, c.log_path)
    return rcs


class WaveDispatcher(Generic[T]):
    """Run a list of independent cells in GPU-parallel waves over an N-GPU pod.

    The dispatcher is the SCHEDULING layer; the caller owns what one cell DOES
    (train / extract / eval) via ``build_cmd``. Smoke == sweep with one cell +
    ``n_gpus=1`` (no architectural divergence — the SAME ``build_cmd`` + the SAME
    subprocess path the multi-GPU sweep uses).

    Args:
        n_gpus: number of GPUs to fan out across; ``<= 1`` is the single-GPU path
            (one cell per wave, all on GPU 0).
        cell_key: ``cell -> str`` unique idempotency key (the disjoint-claim key;
            every output path must be derived from it).
        is_done: ``cell -> bool`` idempotent skip-completed predicate; True cells
            are dropped before the waves (resume-skip, the #667 pattern).
        build_cmd: ``(cell, gpu_id) -> CellCmd`` launch-spec builder. MUST set
            ``env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)`` (the helper re-asserts).
        dry_run: when True, log the planned launches and skip subprocess
            execution (returns a FleetResult with the cells that WOULD run).
    """

    def __init__(
        self,
        *,
        n_gpus: int,
        cell_key: Callable[[T], str],
        is_done: Callable[[T], bool],
        build_cmd: Callable[[T, int], CellCmd],
        dry_run: bool = False,
    ) -> None:
        self.n_gpus = n_gpus
        self.cell_key = cell_key
        self.is_done = is_done
        self.build_cmd = build_cmd
        self.dry_run = dry_run

    def run(self, cells: Sequence[T], *, cwd: Path | None = None) -> FleetResult:
        """Skip already-done cells, then run the rest in waves of ``n_gpus``.

        Steps:
          1. drop cells where ``is_done(cell)`` is True (resume-skip; logs each skip).
          2. ASSERT every remaining ``cell_key`` is unique across the WHOLE fleet
             (disjoint claim — raises :class:`DuplicateCellError` on a collision)
             — checked BEFORE GPU assignment so a malformed grid fails loud, not
             mid-wave.
          3. for each wave of ``<= n_gpus`` cells: densify GPU ids over the wave
             (``i % n_gpus``); ``build_cmd(cell, gpu)``; run the wave concurrently;
             raise :class:`WaveFailedError` listing ``(rc, cell_key)`` on any
             non-zero rc (after the wave's logs are flushed).

        ``is_done`` filtering happens BEFORE GPU assignment so a partially-resumed
        fleet densifies the REMAINING cells across all GPUs (no idle GPU on a
        small resume subset).
        """
        ran: list[str] = []
        skipped: list[str] = []
        todo: list[T] = []
        for cell in cells:
            key = self.cell_key(cell)
            if self.is_done(cell):
                logger.info("[fleet] %s already done — skip (resume)", key)
                skipped.append(key)
            else:
                todo.append(cell)

        # Whole-fleet disjoint-claim assert (before any GPU assignment / launch).
        keys = [self.cell_key(c) for c in todo]
        if len(set(keys)) != len(keys):
            dupes = [k for k in keys if keys.count(k) > 1]
            raise DuplicateCellError(dupes)

        n = max(self.n_gpus, 1)
        wave_count = 0
        for wave_start in range(0, len(todo), n):
            wave = todo[wave_start : wave_start + n]
            gpu_ids = assign_gpu_ids(len(wave), self.n_gpus)
            cmds = [self.build_cmd(cell, gpu) for cell, gpu in zip(wave, gpu_ids, strict=True)]
            wave_count += 1
            if self.dry_run:
                for c in cmds:
                    logger.info(
                        "[fleet][dry-run] %s gpu=%d CVD=%s :: %s",
                        c.cell_key,
                        c.gpu_id,
                        c.env.get("CUDA_VISIBLE_DEVICES"),
                        " ".join(shlex.quote(str(a)) for a in c.argv),
                    )
                    ran.append(c.cell_key)
                continue
            rcs = run_parallel_with_log(cmds, cwd=cwd)
            failures = [(rc, c.cell_key) for rc, c in zip(rcs, cmds, strict=True) if rc != 0]
            if failures:
                # Record the cells that DID launch this wave before raising, so the
                # FleetResult carried by the exception (via the dispatcher's caller)
                # reflects work attempted.
                ran.extend(c.cell_key for c in cmds)
                raise WaveFailedError(failures)
            ran.extend(c.cell_key for c in cmds)

        return FleetResult(ran=ran, skipped=skipped, failures=[], wave_count=wave_count)


# ── Judge-overlap layer (thin scheduling wrapper over eval.batch_judge) ────────


@dataclass
class JudgeHandle:
    """A fire-and-forget Batch-API judge submission + the reconcile path for its result.

    ``submit()`` warms the Batch API (so the judge clears while the GPU moves to
    the next cell); ``reconcile()`` does the deadline-bounded harvest + cache
    write LATER, off the GPU critical path. Idempotent: ``save_raw`` is the
    completion sentinel — once it exists with the expected scores, ``reconcile()``
    re-reads rather than re-submitting (the underlying ``JudgeCache`` keys on
    ``(question, completion)`` content, so a resume reuses cached judgments).
    """

    cell_key: str
    save_raw: Path
    _submit: Callable[[], list[str]]
    _reconcile: Callable[[], dict]
    batch_ids: list[str] = field(default_factory=list)

    def submit(self) -> list[str]:
        """Fire the Batch-API submission (no polling). Returns the submitted batch IDs.

        No-op (returns the empty list) when ``save_raw`` already exists — a
        completed cell on resume needs no resubmit.
        """
        if self.save_raw.exists():
            logger.info(
                "[judge-async] %s save_raw already present — skip submit (resume)",
                self.cell_key,
            )
            return []
        self.batch_ids = self._submit()
        return self.batch_ids

    def reconcile(self) -> dict:
        """Harvest the judge result, write ``save_raw``, return the per-persona scores.

        Idempotent: if ``save_raw`` already exists this is the normal completed
        path (the underlying client's cache makes a re-call do no API work).
        """
        return self._reconcile()


def submit_judge_async(
    completions: dict[str, dict[str, list[str]]],
    *,
    judge_system_prompt: str,
    format_user_msg: Callable[[str, str], str],
    cell_key: str,
    save_raw: Path,
    cache_dir: Path,
    judge_model: str = "claude-sonnet-4-5-20250929",
) -> JudgeHandle:
    """Submit a cell's judge set to the Batch API FIRE-AND-FORGET; return a handle.

    Wraps ``eval.batch_judge.submit_sharded_batches_fire_and_forget`` for the
    fire step (no polling — the GPU keeps working while the batch clears) and
    ``eval.batch_judge.judge_completions_batch`` for the deferred reconcile harvest
    (which writes ``save_raw`` in the exact shape ``_scores_from_save_raw`` reads
    and is idempotent via the file-based ``JudgeCache``). NO change to
    ``batch_judge.py`` — both are existing, hardened entry points.

    The reconcile is fail-loud on a missing/expired shard: it propagates
    ``batch_judge``'s ``BatchDeadlineExceeded`` rather than silently defaulting
    labels (CLAUDE.md fail-fast). ``judge_model`` defaults to the project judge
    (``claude-sonnet-4-5-20250929``).

    Returns a :class:`JudgeHandle`; call ``.submit()`` right after the cell's
    generations land, then ``.reconcile()`` before the step that consumes the
    judged scores.
    """
    # Imported lazily so importing fleet.py (e.g. in the CPU-only unit tests) does
    # not pull in the anthropic SDK / judge-dispatch stack.
    from explore_persona_space.eval.batch_judge import (
        judge_completions_batch,
        make_custom_id,
        submit_sharded_batches_fire_and_forget,
    )

    save_raw = Path(save_raw)
    cache_dir = Path(cache_dir)

    def _build_requests() -> list[dict]:
        """Mirror batch_judge's request shape + custom_id scheme without re-judging.

        The custom_id scheme matches ``_enumerate_and_check_cache``:
        ``f"{persona}__{idx:05d}__{comp_idx:02d}"``, so the fire step warms the
        SAME requests the reconcile harvest (via ``judge_completions_batch``)
        consumes — and ``make_custom_id`` is reused for the API-side id.
        """
        reqs: list[dict] = []
        for persona, by_q in completions.items():
            for idx, (question, comps) in enumerate(by_q.items()):
                for comp_idx, comp in enumerate(comps):
                    logical_id = f"{persona}__{idx:05d}__{comp_idx:02d}"
                    reqs.append(
                        {
                            "custom_id": make_custom_id(logical_id),
                            "params": {
                                "model": judge_model,
                                "max_tokens": 256,
                                "system": judge_system_prompt,
                                "messages": [
                                    {"role": "user", "content": format_user_msg(question, comp)}
                                ],
                            },
                        }
                    )
        return reqs

    def _submit() -> list[str]:
        import anthropic

        client = anthropic.Anthropic()
        return submit_sharded_batches_fire_and_forget(client, _build_requests())

    def _reconcile() -> dict:
        # judge_completions_batch is idempotent (file-based JudgeCache keyed on
        # (question, completion)); when the fire-step batch has cleared, the
        # in-flight results are picked up here and save_raw is written in the
        # canonical shape. Fail-loud on a deadline (BatchDeadlineExceeded
        # propagates) — never a silent default.
        return judge_completions_batch(
            completions,
            judge_system_prompt=judge_system_prompt,
            format_user_msg=format_user_msg,
            judge_model=judge_model,
            cache_dir=cache_dir,
            save_raw=save_raw,
            dry_run=False,
        )

    return JudgeHandle(
        cell_key=cell_key,
        save_raw=save_raw,
        _submit=_submit,
        _reconcile=_reconcile,
    )
