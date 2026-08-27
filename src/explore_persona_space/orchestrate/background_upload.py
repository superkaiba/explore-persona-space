"""Background per-stem Hub uploads: overlap stem N's upload with stem N+1's GPU compute.

Per-stem capture/eval phases historically ran each stem's multi-GB Hub upload FOREGROUND
between GPU stems, idling every provisioned GPU through every upload window (#2546 arm 2;
the #664 idle-burn family). This module provides the shared helper (#2616):

- ``BackgroundStemUploader`` — ONE daemon worker thread executing per-stem upload chains
  in submit order, with bounded backpressure and fail-loud error propagation.
- ``hub_upload_then_free`` — the chain factory for the canonical per-stem persist:
  upload -> verify -> free local shards -> done-marker, in that load-bearing order.

Consumer wiring recipe (mirrors ``run_capture``'s per-stem block,
scripts/issue2546_gen_capture.py:3230-3262; NOT applied to the live issue-2546 branch —
that adoption happens on the ``issue-2546`` branch in a later round of that task)::

    uploader = BackgroundStemUploader(max_pending=2)
    with uploader:
        for stem in stems:
            spawn_workers(...)                     # GPU compute (subprocess workers) — foreground
            ... build report / correctness ...     # compute outputs stay foreground
            uploader.submit(
                hub_upload_then_free(
                    stem_dir, repo_id=DEFAULT_DATASET_REPO, repo_type="dataset",
                    dest=dest, free_glob="slot*.shard*.pt",
                    report=report_by_stem[stem],   # per-stem metrics slot (note b)
                    after_verify=lambda dp=done_p, d=dest, p=marker_payload:
                        _mirror_capture_marker(dp, d, p),
                ),
                label=stem,
            )
        uploader.join()                            # explicit; __exit__ backstops
    write_phase_terminal_sentinel()                # ONLY after join

Notes baked into the recipe:

(a) The per-stem done-marker moves INTO the chain (it must stay strictly after
    verify+free), so resume semantics are unchanged — a crash leaves un-chained stems
    markerless and the existing resume predicate re-runs them, overwriting their
    residual local shards.
(b) Per-stem ``upload_wall_s`` is recorded by the chain into a per-stem dict slot:
    pass the consumer's per-stem report dict as ``report=`` — the chain writes
    ``upload_wall_s`` right after the verified upload (BEFORE free/marker, so a
    marker payload holding the dict carries it) and ``chain_wall_s`` at chain end
    (single writer per key — safe under the GIL) — and the marker payload must hold a
    REFERENCE to that per-stem report dict (binding a COPY silently drops
    ``upload_wall_s`` from the Hub-mirrored marker), or the consumer persists reports
    only after ``join()``.
(c) The LAST stem's upload overlaps nothing — the accepted residual is ONE upload
    window per phase instead of N, and residual phase throughput is
    ``~ max(compute, upload)`` per stem + final drain, so the rows-per-batch knob
    (candidate fix 3) and ``max_pending`` are tuned JOINTLY (if fix 3 shrinks compute
    below the upload wall, the bounded queue backpressures compute — still never worse
    than today's serial sum).
(d) Applying this to ``scripts/issue2546_gen_capture.py`` itself happens on the
    ``issue-2546`` branch in a later round of that task, never mid-flight.
(e) ALL per-stem Hub writes route THROUGH the chain — a consumer adding a foreground
    per-stem Hub call alongside the background chain reintroduces the concurrent-commit
    hazard class (the one-commit-in-flight property holds only if the worker is the
    sole committer during the loop).
(f) Consumers whose stem re-runs can CHANGE the shard file set need exact-set
    verification or destination cleanup — ``upload_folder`` overwrites same-name files
    but leaves differently-named residue in place (the 2546-shape consumer's
    ``slot*.shard*.pt`` set is stable, so clean overwrite holds there).

Disk sizing: consumers size ``max_pending`` via the inequality
``baseline_bytes + (max_pending + 2) * worst_stem_bytes < pod quota`` (RunPod: the
~130 GB MooseFS per-pod quota). ``max_pending`` bounds QUEUED chains only, so peak
unfinished stems = ``max_pending`` queued + 1 executing + 1 foreground
computing-or-blocked.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, MutableMapping
from pathlib import Path

from explore_persona_space.orchestrate import hub

logger = logging.getLogger(__name__)

# How often a blocked submit()/join() wakes to re-check failure state (seconds).
_WAKEUP_S = 30.0

# Worker-stop sentinel enqueued by close() (reached via a CLEAN join()); never a
# user chain — the worker exits the moment it dequeues it.
_STOP_SENTINEL = object()


class BackgroundStemUploader:
    """One daemon worker thread executing per-stem upload chains in submit order.

    Contract: ``submit(work, label=...)`` enqueues a zero-arg callable (the FULL
    per-stem chain: upload -> verify -> free -> marker). Backpressure:
    ``max_pending`` bounds QUEUED chains (the executing chain is NOT counted), so
    peak unfinished stems = max_pending queued + 1 executing + 1 foreground
    computing-or-blocked; consumers size it via
    ``baseline_bytes + (max_pending + 2) * worst_stem_bytes < pod quota``.
    Queue admission is deadline-bounded: a ``submit()`` blocked in a full queue
    re-checks ``_failure`` periodically and raises after ``submit_timeout_s`` naming
    the executing/queued/submitting labels. Fail-loud: after any chain exception,
    every later ``submit()`` and ``join()`` re-raises it (``RuntimeError`` naming the
    failed label, chained ``from`` the original); queued unexecuted chains are
    recorded in ``.skipped`` and NEVER run. Accepted, reported residual: a submit
    already blocked in admission when the failure lands may return normally after
    the worker drains (its chain is skipped; the failure raises at the following
    submit and at join). The worker loop wraps EVERYTHING (not just ``work()``), so
    an internal queue-handling bug also records into ``_failure`` instead of killing
    the thread silently. ``join(timeout_s)`` must be called before the phase
    terminal marker; on timeout it raises naming the pending labels. Worker is
    ``daemon=True`` (a native hf-xet wedge must not block interpreter exit — hub.py
    STAGE_HUB_PREFIX_TIMEOUT_RC note, #1739/#2153); correctness comes from
    join-before-terminal-marker, never from atexit join. A CLEAN ``join()`` (and
    therefore a clean context exit) also STOPS the worker after the drain via
    ``close()`` (stop flag + a non-blocking sentinel nudge — close never blocks on,
    nor joins, a wedged chain), so repeated uploader construction does not
    accumulate idle daemon threads; after close, ``submit()`` raises. A FAILED /
    timed-out ``join()`` leaves the worker running so it drains queued chains into
    ``.skipped`` (the post-failure skip contract). Deliberately NOT
    ``ThreadPoolExecutor``: its non-daemon workers are atexit-joined and a
    native-wedged worker hangs the process with no rc (hub.py:2982-2988).
    """

    def __init__(
        self,
        *,
        max_pending: int = 2,
        name: str = "bg-upload",
        join_timeout_s: float = 1800.0,
        submit_timeout_s: float = 1800.0,
    ) -> None:
        """Start the daemon worker. max_pending bounds QUEUED (not executing) chains."""
        if max_pending < 1:
            raise ValueError(f"max_pending must be >= 1, got {max_pending}")
        self._name = name
        self.join_timeout_s = float(join_timeout_s)
        self.submit_timeout_s = float(submit_timeout_s)
        self._queue: queue.Queue[tuple[str, float, Callable[[], None]] | object] = queue.Queue(
            maxsize=max_pending
        )
        self._cond = threading.Condition()
        self._stop = threading.Event()  # set by close(); the worker exits on observing it
        self._pending: list[str] = []  # submitted-but-unfinished labels (queued + executing)
        self._current: str | None = None  # label the worker is executing right now
        self._failure: tuple[str, BaseException] | None = None
        self.skipped: list[str] = []  # labels of never-executed chains (post-failure drain)
        self._thread = threading.Thread(target=self._worker, name=name, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public API

    @property
    def failure(self) -> BaseException | None:
        """The first chain exception, or None. Set once; never cleared."""
        with self._cond:
            return self._failure[1] if self._failure is not None else None

    def submit(self, work: Callable[[], None], *, label: str) -> None:
        """Enqueue a chain; blocks (deadline-bounded) when the queue is full.

        Raises RuntimeError chained from the original exception if any prior chain
        failed (checked BEFORE admission, after every blocked-put wakeup, and once
        more post-put), and RuntimeError naming the executing/queued/submitting
        labels once cumulative admission wait exceeds ``submit_timeout_s``.
        Raises RuntimeError after ``close()`` (a clean ``join()``): a submit into a
        stopped worker would silently never run — fail loud instead.
        """
        self._raise_if_failed()
        if self._stop.is_set():
            raise RuntimeError(
                f"uploader {self._name!r} is closed (clean join()/close() already ran); "
                "construct a new BackgroundStemUploader"
            )
        with self._cond:
            self._pending.append(label)
        start = time.monotonic()
        enqueued = False
        try:
            while True:
                waited = time.monotonic() - start
                remaining = self.submit_timeout_s - waited
                if remaining <= 0:
                    raise RuntimeError(self._admission_timeout_msg(label, waited))
                try:
                    self._queue.put(
                        (label, time.monotonic(), work),
                        timeout=min(_WAKEUP_S, max(remaining, 0.05)),
                    )
                    enqueued = True
                    break
                except queue.Full:
                    self._raise_if_failed()
        finally:
            if not enqueued:
                with self._cond:
                    if label in self._pending:
                        self._pending.remove(label)
        # Post-put recheck: a failure that landed while this submit was blocked in
        # admission raises here; the just-enqueued chain is then SKIPPED by the worker
        # (recorded in .skipped), never run. Residual: a failure landing after this
        # check returns normally — it raises at the following submit and at join.
        self._raise_if_failed()

    def join(self, timeout_s: float | None = None) -> None:
        """Wait for every submitted chain to finish; re-raise the first failure.

        MUST be called before the phase terminal marker. ``timeout_s=None`` uses
        ``self.join_timeout_s``; on timeout raises RuntimeError naming the pending
        labels (the wedged-chain escape — the daemon worker cannot block exit).
        A CLEAN join (drained, no failure) also stops the worker via ``close()``;
        a failed/timed-out join leaves it running (post-failure skip drain).
        """
        limit = self.join_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit
        with self._cond:
            while self._pending and self._failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"background upload join timed out after {limit:.0f}s; "
                        f"pending={list(self._pending)!r}"
                    )
                self._cond.wait(timeout=min(remaining, _WAKEUP_S))
            if self._failure is not None:
                label, exc = self._failure
                raise RuntimeError(f"background upload chain {label!r} failed") from exc
        self.close()  # clean drain: stop the worker so no idle daemon thread leaks

    def close(self) -> None:
        """Stop the worker thread; idempotent; never blocks on a wedged chain.

        Called automatically by a CLEAN ``join()`` (and therefore by clean context
        exit). Sets the stop flag and nudges the queue with a non-blocking sentinel
        so a ``get()``-parked worker exits promptly; a worker mid-chain exits after
        its current chain, and a natively wedged chain simply never observes the
        flag — ``close()`` never joins the worker, and ``daemon=True`` keeps
        interpreter exit safe regardless. Call only after all submits completed
        (the clean-join path does); later ``submit()`` calls raise RuntimeError.
        """
        self._stop.set()
        try:
            self._queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            # Not an error: the worker is mid-chain (or wedged) with a full queue;
            # it observes the stop flag at its next idle queue poll (~1 s cadence).
            logger.debug("[%s] close(): queue full; stop flag only", self._name)

    def __enter__(self) -> BackgroundStemUploader:
        """Context manager: clean body exit joins; a body exception does not."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """On clean exit call join() (which may raise). On a body exception do NOT
        join (never blocks teardown on a wedged upload) — log pending labels plus any
        recorded failure and return False so the body exception propagates."""
        if exc_type is None:
            self.join()
            return False
        with self._cond:
            pending = list(self._pending)
            failure = self._failure
        # Include the recorded exception's repr, not only its label (#2616 r1): on a
        # combined body+chain failure this line is the exit-site record — the full
        # traceback is already logged at record time by _run_chain / _worker.
        recorded = f"{failure[0]!r}: {failure[1]!r}" if failure is not None else None
        logger.error(
            "[%s] context body raised %s; NOT joining (pending=%r, recorded failure=%s)",
            self._name,
            exc_type.__name__,
            pending,
            recorded,
        )
        return False

    # ------------------------------------------------------------------ internals

    def _raise_if_failed(self) -> None:
        """Re-raise the first recorded chain failure, chained from the original."""
        with self._cond:
            failure = self._failure
        if failure is not None:
            label, exc = failure
            raise RuntimeError(f"background upload chain {label!r} failed") from exc

    def _admission_timeout_msg(self, submitting: str, waited: float) -> str:
        """Compose the deadline-bounded-admission error naming every relevant label."""
        with self._cond:
            executing = self._current
            queued = [p for p in self._pending if p != submitting and p != executing]
        return (
            f"background upload submit admission timed out after {waited:.0f}s "
            f"(submit_timeout_s={self.submit_timeout_s:.0f}); executing={executing!r} "
            f"queued={queued!r} submitting={submitting!r}"
        )

    def _worker(self) -> None:
        """Daemon loop. The ENTIRE body is wrapped so any exception — inside work()
        or in the queue handling around it — records into _failure and notifies; an
        internal bug must never kill the thread silently. Exits on the close()
        sentinel (or the stop flag when the sentinel found the queue full)."""
        while True:
            try:
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
                if item is _STOP_SENTINEL:
                    return
                label, enqueued_t, work = item
                self._run_chain(label, enqueued_t, work)
            except BaseException as exc:  # fail-loud via _failure, never a silent thread death
                with self._cond:
                    if self._failure is None:
                        self._failure = ("<worker-internal>", exc)
                    self._cond.notify_all()
                logger.exception("[%s] internal worker error recorded as failure", self._name)

    def _run_chain(self, label: str, enqueued_t: float, work: Callable[[], None]) -> None:
        """Execute one dequeued chain: skip-after-failure, run, record, notify."""
        with self._cond:
            if self._failure is not None:
                self.skipped.append(label)
                if label in self._pending:
                    self._pending.remove(label)
                self._cond.notify_all()
                logger.info("[%s] chain SKIPPED label=%s (prior failure)", self._name, label)
                return
            self._current = label
        queue_wait_s = time.monotonic() - enqueued_t
        start = time.monotonic()
        logger.info("[%s] chain start label=%s queue_wait_s=%.1f", self._name, label, queue_wait_s)
        try:
            work()
        except BaseException as exc:
            with self._cond:
                self._current = None
                if self._failure is None:
                    self._failure = (label, exc)
                if label in self._pending:
                    self._pending.remove(label)
                self._cond.notify_all()
            logger.exception(
                "[%s] chain FAILED label=%s after %.1fs",
                self._name,
                label,
                time.monotonic() - start,
            )
            return
        with self._cond:
            self._current = None
            if label in self._pending:
                self._pending.remove(label)
            self._cond.notify_all()
        logger.info(
            "[%s] chain end label=%s wall_s=%.1f queue_wait_s=%.1f",
            self._name,
            label,
            time.monotonic() - start,
            queue_wait_s,
        )


def hub_upload_then_free(
    stem_dir: Path,
    *,
    repo_id: str,
    repo_type: str,
    dest: str,
    free_glob: str | None = None,
    after_verify: Callable[[], None] | None = None,
    report: MutableMapping[str, float] | None = None,
) -> Callable[[], None]:
    """Chain factory for the canonical per-stem persist.

    The returned zero-arg chain runs, in this load-bearing order:
    ``hub._upload(stem_dir, repo_id, repo_type, dest, raise_on_error=True)`` ->
    raise RuntimeError on an empty result (the missing-token / absent-path /
    0-files-verify ``""`` returns must fail loud here, not assert — library code) ->
    unlink ``sorted(stem_dir.glob(free_glob))`` -> ``after_verify()`` (e.g. the
    consumer's Hub-first done-marker mirror). Free strictly after verified upload;
    marker strictly after free.

    When ``report`` is provided (the consumer's per-stem report dict — module
    docstring note (b)), the chain records ``upload_wall_s`` into it immediately
    after the verified upload (BEFORE free/marker, so an ``after_verify`` marker
    payload holding the dict REFERENCE carries it) and ``chain_wall_s`` when the
    chain completes. Single writer per key — safe under the GIL.
    """

    def _chain() -> None:
        t0 = time.monotonic()
        result = hub._upload(stem_dir, repo_id, repo_type, dest, raise_on_error=True)
        if not result:
            raise RuntimeError(
                f"hub._upload returned an empty result for {stem_dir} -> "
                f"{repo_id}/{dest} (missing token / absent local path / 0-files "
                "verification); local shards NOT freed, marker NOT written"
            )
        upload_wall_s = time.monotonic() - t0
        if report is not None:
            report["upload_wall_s"] = upload_wall_s
        logger.info("[bg-upload] upload verified dest=%s upload_wall_s=%.1f", dest, upload_wall_s)
        if free_glob is not None:
            freed = sorted(stem_dir.glob(free_glob))
            for path in freed:
                path.unlink()
            logger.info("[bg-upload] freed %d local shard file(s) under %s", len(freed), stem_dir)
        if after_verify is not None:
            after_verify()
        if report is not None:
            report["chain_wall_s"] = time.monotonic() - t0

    return _chain
