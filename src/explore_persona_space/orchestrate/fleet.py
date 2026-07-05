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


def _logical_custom_id(persona: str, idx: int, comp_idx: int) -> str:
    """The batch_judge custom_id scheme: ``f"{persona}__{idx:05d}__{comp_idx:02d}"``.

    Mirrors ``batch_judge._enumerate_and_check_cache`` so the per-row scores this
    helper writes under ``save_raw["all_scores"]`` are keyed identically to what
    ``judge_completions_batch`` writes — i.e. an ``issue664_eval._scores_from_save_raw``
    reader does not care which path produced the file.
    """
    return f"{persona}__{idx:05d}__{comp_idx:02d}"


def _judge_requests(
    completions: dict[str, dict[str, list[str]]],
    *,
    judge_system_prompt: str,
    format_user_msg: Callable[[str, str], str],
    judge_model: str,
    max_tokens: int,
) -> list[dict]:
    """Build the Batch-API requests for ``completions``, keyed on the logical id."""
    reqs: list[dict] = []
    for persona, by_q in completions.items():
        for idx, (question, comps) in enumerate(by_q.items()):
            for comp_idx, comp in enumerate(comps):
                reqs.append(
                    {
                        "custom_id": _logical_custom_id(persona, idx, comp_idx),
                        "params": {
                            "model": judge_model,
                            "max_tokens": max_tokens,
                            "system": judge_system_prompt,
                            "messages": [
                                {"role": "user", "content": format_user_msg(question, comp)}
                            ],
                        },
                    }
                )
    return reqs


def _poll_batch_to_ended(
    client,
    batch_id: str,
    *,
    poll_interval: float,
    max_poll_interval: float,
    grace_min: int,
) -> None:
    """Block until ``batch_id`` ends under a hard ``expires_at`` deadline.

    Same bounded-poll shape as ``batch_judge._submit_and_poll_batch`` (30s ->
    1.5x -> ``max_poll_interval``, capped by the deadline, NOT a step count).
    Raises ``BatchDeadlineExceeded`` if the batch is still not ended at the
    deadline after one final retrieve — never a silent default (CLAUDE.md fail-fast).

    NOTE (#995): retrieves here are deliberately NOT wrapped in
    ``llm.anthropic_client.retrieve_with_create_grace`` — ``JudgeHandle.reconcile()``
    runs deferred, often in a DIFFERENT process, and the handle records no create
    timestamp, so the conservative terminal-404 default is correct. If a
    fleet-path read-after-write incident ever occurs, the shared helper is one
    import + one additive ``created_at`` kwarg away.
    """
    import datetime as _dt
    import time as _time

    from explore_persona_space.llm.anthropic_client import (
        BatchDeadlineExceeded,
        deadline_from_expires_at,
    )

    deadline: _dt.datetime | None = None
    interval = poll_interval
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return
        if deadline is None:
            expires_at = getattr(batch, "expires_at", None)
            deadline = (
                deadline_from_expires_at(expires_at, grace_min)
                if expires_at is not None
                else _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=25)
            )
        if _dt.datetime.now(_dt.UTC) > deadline:
            final = client.messages.batches.retrieve(batch_id)
            if final.processing_status == "ended":
                return
            raise BatchDeadlineExceeded(batch_id, deadline)
        _time.sleep(interval)
        interval = min(interval * 1.5, max_poll_interval)


@dataclass
class JudgeHandle:
    """A fire-and-forget Batch-API judge submission + the reconcile path for its result.

    ``submit()`` creates the Batch-API job(s) WITHOUT polling (so the GPU can move
    to the next cell while the judge clears); ``reconcile()`` does the
    deadline-bounded harvest + ``save_raw`` write LATER, off the GPU critical path.
    Both sides key on the SAME logical custom_id, so the harvest joins the fired
    batches with no id mismatch. ``save_raw`` is the idempotency sentinel: once it
    exists, ``submit()`` skips resubmission and ``reconcile()`` re-reads it.

    ``expected_custom_ids`` (#676 round-2, concern judge-custom-id-coverage-unverified):
    the FULL set of custom_ids this submission expects back, captured eagerly at
    :func:`submit_judge_async` time from the request list. ``reconcile()`` requires
    every expected id to be present in the harvest (and in any re-read ``save_raw``)
    and raises ``RuntimeError`` on a miss — so an ENDED-but-incomplete shard fails
    loud rather than silently degrading a missing row to a default label / a biased
    rate. Eager capture (never ``None``) is deliberate: an absent set must surface
    as an internal error, NOT a silent skip of the coverage check.

    ``expected_source`` (#676 round-2, concern judge-save-raw-collision): the
    source/persona this handle's deferred behavior-label job belongs to. The primary
    collision defense is the per-source ``save_raw`` key the callers now build
    (``judge_filter/{behavior}__{src}.json`` — never a coarse ``int(time.time())``);
    ``expected_source`` is a belt-and-suspenders ownership tag carried on the handle
    for assertion / debugging. ``None`` for callers that do not partition by source
    (the per-cell eval judge).
    """

    cell_key: str
    save_raw: Path
    _submit: Callable[[], list[str]]
    _reconcile: Callable[[list[str]], dict]
    expected_custom_ids: frozenset[str] = field(default_factory=frozenset)
    expected_source: str | None = None
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
        """Deadline-bounded harvest of the fired batches; write ``save_raw``; return scores.

        Idempotent: if ``save_raw`` already exists it is read back rather than
        re-harvested. Fail-loud on an expired shard — ``BatchDeadlineExceeded``
        propagates rather than silently defaulting any label (CLAUDE.md fail-fast).
        Returns ``{custom_id: score_dict}`` (the same join key the fire used).
        """
        return self._reconcile(self.batch_ids)


def _fire_judge_batches(requests: list[dict]) -> list[str]:
    """Create the Batch-API job(s) for ``requests`` fire-and-forget; return batch IDs."""
    import anthropic

    from explore_persona_space.eval.batch_judge import submit_sharded_batches_fire_and_forget

    return submit_sharded_batches_fire_and_forget(anthropic.Anthropic(), requests)


def _assert_full_coverage(
    scores: dict, expected_custom_ids: frozenset[str], *, cell_key: str, source: str
) -> None:
    """Raise ``RuntimeError`` if any expected custom_id is absent from ``scores``.

    The fail-loud coverage gate (#676 round-2, concern
    judge-custom-id-coverage-unverified): an ENDED-but-incomplete shard, a
    custom_id mismatch, or a partial harvested result would otherwise leave a row
    silently missing — and every downstream consumer reads with ``.get`` (default
    label 0 / reduced ``n_judged`` / a biased baseline-propensity rate), so the miss
    never surfaces. Names at most the first 10 missing ids. Raised BEFORE the
    ``save_raw`` write so a partial result is never persisted.
    """
    missing = sorted(expected_custom_ids - set(scores))
    if missing:
        shown = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise RuntimeError(
            f"[judge-async] {cell_key} ({source}): incomplete judge coverage — "
            f"{len(missing)}/{len(expected_custom_ids)} expected custom_ids missing "
            f"from the harvest: {shown}{more}. An ended-but-incomplete shard must "
            f"fail loud, NOT degrade missing rows to default labels (CLAUDE.md fail-fast)."
        )


def _reconcile_judge_batches(
    batch_ids: list[str],
    *,
    cell_key: str,
    save_raw: Path,
    judge_model: str,
    poll_interval: float,
    max_poll_interval: float,
    grace_min: int,
    expected_custom_ids: frozenset[str],
    expected_source: str | None = None,
) -> dict:
    """Deadline-bounded harvest of ``batch_ids``; write ``save_raw``; return scores.

    Idempotent: if ``save_raw`` already exists, read it back (no API call) — but
    the re-read still passes the SAME coverage gate, so a partial-prior-run
    ``save_raw`` is never silently reused (#676 round-2).
    Fail-loud: an expired shard raises ``BatchDeadlineExceeded`` (via
    :func:`_poll_batch_to_ended`); an empty ``batch_ids`` with no ``save_raw`` is a
    submit-never-ran programming error and raises; a harvest (or re-read) missing any
    of ``expected_custom_ids`` raises ``RuntimeError`` BEFORE writing ``save_raw``
    (concern judge-custom-id-coverage-unverified). ``save_raw`` is written in the
    ``{"all_scores": {custom_id: score}}`` shape ``_scores_from_save_raw`` reads.
    """
    import json as _json

    src = expected_source or "—"
    if save_raw.exists():
        on_disk = _json.loads(save_raw.read_text()).get("all_scores", {})
        # A prior run may have written a partial save_raw before this gate existed;
        # re-validate coverage so a stale incomplete file is not silently reused.
        _assert_full_coverage(on_disk, expected_custom_ids, cell_key=cell_key, source=src)
        return on_disk
    if not batch_ids:
        raise RuntimeError(
            f"[judge-async] {cell_key}: reconcile called with no batch_ids and no "
            f"save_raw at {save_raw} — submit() was never run (or returned empty)"
        )
    import anthropic

    from explore_persona_space.eval.batch_judge import _collect_legacy_results

    client = anthropic.Anthropic()
    results: dict[str, dict] = {}
    for batch_id in batch_ids:
        _poll_batch_to_ended(
            client,
            batch_id,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            grace_min=grace_min,
        )
        _collect_legacy_results(client, batch_id, results)
    # Coverage gate BEFORE the write: a partial harvest must not persist a save_raw
    # that downstream readers would silently treat as complete.
    _assert_full_coverage(results, expected_custom_ids, cell_key=cell_key, source=src)
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    save_raw.write_text(
        _json.dumps(
            {"all_scores": results, "judge_model": judge_model, "n_total": len(results)},
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info("[judge-async] %s reconciled %d scores -> %s", cell_key, len(results), save_raw)
    return results


def submit_judge_async(
    completions: dict[str, dict[str, list[str]]],
    *,
    judge_system_prompt: str,
    format_user_msg: Callable[[str, str], str],
    cell_key: str,
    save_raw: Path,
    expected_source: str | None = None,
    judge_model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 256,
    poll_interval: float = 30.0,
    max_poll_interval: float = 120.0,
    grace_min: int = 30,
) -> JudgeHandle:
    """Submit a cell's judge set to the Batch API FIRE-AND-FORGET; return a handle.

    The fire step (``handle.submit()``) creates the Batch-API job(s) via
    ``eval.batch_judge.submit_sharded_batches_fire_and_forget`` and returns
    immediately (no polling) so the GPU keeps working. The reconcile step
    (``handle.reconcile()``) polls the fired batches to ``ended`` under a hard
    deadline (``deadline_from_expires_at`` + grace; ``BatchDeadlineExceeded`` on
    overshoot — NEVER a silent default), harvests via ``_collect_legacy_results``,
    and writes ``save_raw`` in the ``{"all_scores": {custom_id: score}}`` shape
    ``issue664_eval._scores_from_save_raw`` reads.

    Both fire and reconcile key on the SAME logical custom_id
    (``f"{persona}__{idx:05d}__{comp_idx:02d}"``), so the harvest joins the fired
    batches with NO id mismatch. NO change to ``batch_judge.py`` — every primitive
    used (``submit_sharded_batches_fire_and_forget``, ``_collect_legacy_results``,
    ``deadline_from_expires_at``, ``BatchDeadlineExceeded``) already exists there.

    ``save_raw`` is the idempotency sentinel: once it exists, ``submit()`` skips
    resubmission and ``reconcile()`` re-reads it (so a resumed run re-reads rather
    than re-judges). ``judge_model`` defaults to the project judge
    (``claude-sonnet-4-5-20250929``).

    The FULL set of custom_ids the requests carry is captured eagerly here and
    threaded onto the handle as ``expected_custom_ids``, so ``reconcile()`` can
    fail loud on an ended-but-incomplete shard rather than silently dropping a
    missing row (#676 round-2, concern judge-custom-id-coverage-unverified).
    ``expected_source`` (optional) is the source/persona this submission belongs
    to — the ownership tag carried for the per-source-keyed deferred behavior-label
    jobs (concern judge-save-raw-collision); the primary collision defense is the
    per-source ``save_raw`` key the CALLER builds.
    """
    from functools import partial

    save_raw = Path(save_raw)
    requests = _judge_requests(
        completions,
        judge_system_prompt=judge_system_prompt,
        format_user_msg=format_user_msg,
        judge_model=judge_model,
        max_tokens=max_tokens,
    )
    expected_custom_ids = frozenset(req["custom_id"] for req in requests)
    return JudgeHandle(
        cell_key=cell_key,
        save_raw=save_raw,
        _submit=partial(_fire_judge_batches, requests),
        _reconcile=partial(
            _reconcile_judge_batches,
            cell_key=cell_key,
            save_raw=save_raw,
            judge_model=judge_model,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
            grace_min=grace_min,
            expected_custom_ids=expected_custom_ids,
            expected_source=expected_source,
        ),
        expected_custom_ids=expected_custom_ids,
        expected_source=expected_source,
    )
