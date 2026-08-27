"""Tests for orchestrate/background_upload.py (#2616, plan v2 acceptance criteria A1-A8).

CPU-only, no network. Every ordering claim is an event-ordered happens-before
assertion (a submit/join provably CANNOT complete while a gating Event is unset);
for the two blocked-call witnesses (A6 backpressure, A8 context join) the EXACT
blocking call is instrumented to signal entry and to record the release-state AT
ITS RETURN — timeouts appear only as bounded POSITIVE hang caps, never as ordering
evidence (no ``assert not event.wait(t)`` witnesses; #2616 r2). Blocking Events are
released in teardown (the ``gates`` fixture) so no daemon thread is left wedged
across tests. ``hub._upload`` is faked ONLY at the external network boundary, with
a def mirroring the real signature; one un-mocked test pins that signature.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time

import pytest

from explore_persona_space.orchestrate import background_upload as bgu
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.background_upload import (
    BackgroundStemUploader,
    hub_upload_then_free,
)

# Bounded hang cap for Event waits / condition polls. A test that trips this cap is
# HUNG, not mis-ordered — the ordering assertions are the event-state asserts.
WAIT = 20.0


def _wait_until(predicate, timeout: float = WAIT) -> None:
    """Bounded hang cap: poll until predicate() is true or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for condition (hang cap tripped)")


def _raiser(exc: BaseException):
    def chain() -> None:
        raise exc

    return chain


@pytest.fixture
def gates():
    """Collects blocking Events; teardown releases them all (thread hygiene)."""
    created: list[threading.Event] = []
    yield created
    for event in created:
        event.set()


def _fake_upload_factory(events, expect_exists=(), result_fmt="{repo_id}/{path_in_repo}"):
    """Signature-conformant fake of hub._upload (mirrors the real def, #906 rule)."""

    def fake_upload(
        local_path,
        repo_id,
        repo_type,
        path_in_repo,
        delete_after=False,
        upload_as_file=False,
        ignore_patterns=None,
        private=False,
        raise_on_error=False,
    ):
        assert raise_on_error is True  # the chain factory's contract
        for path in expect_exists:
            assert path.exists(), f"shard {path} freed BEFORE the upload verified"
        events.append("upload")
        return result_fmt.format(repo_id=repo_id, path_in_repo=path_in_repo)

    return fake_upload


# --------------------------------------------------------------------------- A1


def test_pipelining_overlap(gates):
    """submit() returns while a slow chain is still executing; submit order preserved."""
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    order: list[str] = []

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)
        order.append("c1")

    up = BackgroundStemUploader(max_pending=2)
    up.submit(chain1, label="c1")
    assert started.wait(WAIT)
    up.submit(lambda: order.append("c2"), label="c2")
    # Happens-before witness, asserted BEFORE the Event is released: submit("c2")
    # returned while chain1 is provably still blocked (gate unset -> c1 incomplete).
    assert not gate.is_set()
    assert order == []
    gate.set()
    up.join()
    assert order == ["c1", "c2"]


# --------------------------------------------------------------------------- A2


def test_upload_failure_raises_at_next_submit():
    """A chain exception re-raises at the next submit AND at every later submit."""
    original = ValueError("upload exploded")
    ran: list[str] = []

    up = BackgroundStemUploader(max_pending=2)
    up.submit(_raiser(original), label="c1")
    _wait_until(lambda: up.failure is not None)

    for label in ("c2", "c3"):  # >= 2 later submits — each raises
        with pytest.raises(RuntimeError) as excinfo:
            up.submit(lambda: ran.append("never"), label=label)
        assert excinfo.value.__cause__ is original
        assert "c1" in str(excinfo.value)
    assert ran == []


# --------------------------------------------------------------------------- A3


def test_upload_failure_raises_at_join():
    """The same exception re-raises at join(); the terminal-marker step is unreached."""
    original = ValueError("upload exploded")
    up = BackgroundStemUploader(max_pending=2)
    up.submit(_raiser(original), label="c1")
    terminal_marker: list[str] = []
    with pytest.raises(RuntimeError) as excinfo:
        up.join()
        terminal_marker.append("terminal")  # the write_phase_terminal_sentinel slot
    assert excinfo.value.__cause__ is original
    assert terminal_marker == []


# --------------------------------------------------------------------------- A4


def test_failure_skips_pending_chains(gates):
    """After a failure, queued unexecuted chains are SKIPPED — callables never run."""
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    original = ValueError("boom")
    ran: list[str] = []

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)
        raise original

    up = BackgroundStemUploader(max_pending=2)
    up.submit(chain1, label="c1")
    assert started.wait(WAIT)
    up.submit(lambda: ran.append("c2"), label="c2")
    up.submit(lambda: ran.append("c3"), label="c3")
    gate.set()
    with pytest.raises(RuntimeError) as excinfo:
        up.join()
    assert excinfo.value.__cause__ is original
    _wait_until(lambda: up.skipped == ["c2", "c3"])
    assert ran == []


# --------------------------------------------------------------------------- A5


def test_free_only_after_verified_upload(tmp_path, monkeypatch):
    """Local files are unlinked only AFTER hub._upload returns a truthy verified path."""
    stem_dir = tmp_path / "stem0"
    stem_dir.mkdir()
    shard_a = stem_dir / "slot0.shard0.pt"
    shard_b = stem_dir / "slot1.shard0.pt"
    shard_a.write_bytes(b"a" * 8)
    shard_b.write_bytes(b"b" * 8)
    events: list[str] = []
    monkeypatch.setattr(
        bgu.hub, "_upload", _fake_upload_factory(events, expect_exists=(shard_a, shard_b))
    )

    def after_verify() -> None:
        assert not shard_a.exists() and not shard_b.exists()  # freed strictly before marker
        events.append("marker")

    chain = hub_upload_then_free(
        stem_dir,
        repo_id="org/data",
        repo_type="dataset",
        dest="issue2616/stem0",
        free_glob="slot*.shard*.pt",
        after_verify=after_verify,
    )
    chain()
    assert events == ["upload", "marker"]
    assert not shard_a.exists() and not shard_b.exists()


def test_empty_upload_result_raises_without_free(tmp_path, monkeypatch):
    """An empty-string hub._upload return raises and does NOT unlink / mark."""
    stem_dir = tmp_path / "stem0"
    stem_dir.mkdir()
    shard = stem_dir / "slot0.shard0.pt"
    shard.write_bytes(b"a" * 8)
    events: list[str] = []
    monkeypatch.setattr(bgu.hub, "_upload", _fake_upload_factory(events, result_fmt=""))
    marker: list[str] = []

    chain = hub_upload_then_free(
        stem_dir,
        repo_id="org/data",
        repo_type="dataset",
        dest="issue2616/stem0",
        free_glob="slot*.shard*.pt",
        after_verify=lambda: marker.append("m"),
    )
    with pytest.raises(RuntimeError, match="empty result"):
        chain()
    assert shard.exists()  # shards NOT freed
    assert marker == []  # marker NOT written


# --------------------------------------------------------------------------- A6


def test_bounded_queue_backpressure(gates, monkeypatch):
    """max_pending bounds QUEUED chains: with max_pending=1 the THIRD submit blocks.

    Instrumented happens-before witness (#2616 r2 — no negative timed waits): c3's
    exact blocking call (the queue put inside submit) is wrapped to signal ENTRY and
    to record, AT ITS RETURN, whether the gate had been released and chain c1 had
    completed. A correct bounded queue admits c3 only after gate release -> c1
    completion -> c2 dequeue (both flags True by happens-before); an unbounded /
    broken queue admits c3 straight from the entry signal, before the test releases
    the gate, recording False — caught regardless of helper-thread scheduling.
    """
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    chain1_done = threading.Event()
    order: list[str] = []

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)
        order.append("c1")
        chain1_done.set()

    up = BackgroundStemUploader(max_pending=1)
    real_put = up._queue.put
    c3_put_entered = threading.Event()
    c3_put_returned = threading.Event()
    at_c3_admission: dict[str, bool] = {}

    def instrumented_put(item, *args, **kwargs):
        is_c3 = isinstance(item, tuple) and item[0] == "c3"
        if is_c3:
            c3_put_entered.set()
        result = real_put(item, *args, **kwargs)
        if is_c3:
            # Recorded AT the blocking call's return — the decisive ordering read.
            at_c3_admission["gate_released"] = gate.is_set()
            at_c3_admission["c1_done"] = chain1_done.is_set()
            c3_put_returned.set()
        return result

    monkeypatch.setattr(up._queue, "put", instrumented_put)

    up.submit(chain1, label="c1")
    assert started.wait(WAIT)  # worker holds c1 (executing, not queued)
    up.submit(lambda: order.append("c2"), label="c2")  # fills the single queue slot
    assert up._queue.full()  # structural: the ONE queue slot is occupied by c2

    t3_returned = threading.Event()

    def submit_c3() -> None:
        up.submit(lambda: order.append("c3"), label="c3")
        t3_returned.set()

    helper = threading.Thread(target=submit_c3, daemon=True)
    helper.start()
    assert c3_put_entered.wait(WAIT)  # positive: c3 reached the blocking put
    assert not gate.is_set()  # test-controlled: the release has not happened yet
    gate.set()
    assert c3_put_returned.wait(WAIT)  # bounded hang cap, not an ordering witness
    assert at_c3_admission == {"gate_released": True, "c1_done": True}
    assert t3_returned.wait(WAIT)
    up.join()
    assert order == ["c1", "c2", "c3"]
    helper.join(WAIT)


# --------------------------------------------------------------------------- A7


def test_join_timeout_raises_naming_pending(gates):
    """join(timeout_s) on a wedged chain raises RuntimeError naming pending labels."""
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)

    up = BackgroundStemUploader(max_pending=2)
    up.submit(chain1, label="stem-wedged")
    assert started.wait(WAIT)
    with pytest.raises(RuntimeError, match="stem-wedged"):
        up.join(timeout_s=0.2)
    gate.set()
    up.join()  # drains cleanly after release


def test_submit_admission_timeout_on_wedged_worker(gates):
    """A submit blocked in a full queue raises after submit_timeout_s naming the
    executing/queued/submitting labels (deterministic wedged-worker/full-queue shape)."""
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    ran: list[str] = []

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)

    up = BackgroundStemUploader(max_pending=1, submit_timeout_s=0.3)
    up.submit(chain1, label="c1")
    assert started.wait(WAIT)
    up.submit(lambda: ran.append("c2"), label="c2")  # fills the queue
    with pytest.raises(RuntimeError) as excinfo:
        up.submit(lambda: ran.append("c3"), label="c3")
    message = str(excinfo.value)
    assert "c1" in message  # executing
    assert "c2" in message  # queued
    assert "c3" in message  # submitting
    assert ran == []  # queued callables never ran (asserted before release)
    gate.set()
    up.join()


def test_worker_thread_is_daemon():
    """The worker is daemon=True: a native wedge cannot block interpreter exit."""
    up = BackgroundStemUploader()
    assert up._thread.daemon is True


# --------------------------------------------------------------------------- A8


def test_context_manager_join_waits_event_ordered(gates, monkeypatch):
    """__exit__ on clean body exit provably WAITS: a no-op join fails this test.

    Instrumented happens-before witness (#2616 r2 — no negative timed waits): the
    exact blocking call (__exit__'s join) is wrapped to signal ENTRY and to record,
    AT ITS RETURN, whether the gate had been released and the chain completed. A
    real join returns only after gate release -> chain completion -> pending drain
    (both flags True by happens-before); a no-op join returns straight from the
    entry signal, before the test releases the gate, recording False — caught
    regardless of helper-thread scheduling. An __exit__ that never calls join at
    all fails the positive join_entered wait.
    """
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    chain_done = threading.Event()
    body_done = threading.Event()
    exited = threading.Event()
    join_entered = threading.Event()
    join_returned = threading.Event()
    at_join_return: dict[str, bool] = {}

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)
        chain_done.set()

    up = BackgroundStemUploader(max_pending=2)
    real_join = up.join

    def instrumented_join(timeout_s=None):
        join_entered.set()
        result = real_join(timeout_s)
        # Recorded AT the blocking call's return — the decisive ordering read.
        at_join_return["gate_released"] = gate.is_set()
        at_join_return["chain_done"] = chain_done.is_set()
        join_returned.set()
        return result

    monkeypatch.setattr(up, "join", instrumented_join)

    def run_ctx() -> None:
        with up:
            up.submit(chain1, label="c1")
            body_done.set()
        exited.set()

    helper = threading.Thread(target=run_ctx, daemon=True)
    helper.start()
    assert body_done.wait(WAIT)
    assert started.wait(WAIT)
    assert join_entered.wait(WAIT)  # positive: __exit__ actually ENTERED join()
    assert not chain_done.is_set()  # test-controlled: the chain is still gate-blocked
    gate.set()
    assert join_returned.wait(WAIT)  # bounded hang cap, not an ordering witness
    assert at_join_return == {"gate_released": True, "chain_done": True}
    assert exited.wait(WAIT)
    assert chain_done.is_set()  # chain completed BEFORE __exit__ returned
    helper.join(WAIT)


def test_context_manager_body_exception_returns_while_chain_blocked(gates):
    """On a body exception __exit__ returns WITHOUT joining and never swallows."""
    gate = threading.Event()
    gates.append(gate)
    started = threading.Event()
    chain_done = threading.Event()

    def chain1() -> None:
        started.set()
        assert gate.wait(WAIT)
        chain_done.set()

    up = BackgroundStemUploader(max_pending=2)
    with pytest.raises(ValueError, match="body boom"), up:
        up.submit(chain1, label="c1")
        assert started.wait(WAIT)
        raise ValueError("body boom")
    # __exit__ returned while the chain is provably still Event-blocked.
    assert not chain_done.is_set()
    gate.set()
    up.join()
    assert chain_done.is_set()


def test_body_exception_exit_logs_recorded_failure_detail(caplog):
    """On a combined body+chain failure, the __exit__ log carries the recorded
    exception's repr (type + message), not only its label (#2616 r1 reconciler
    polish; the record-time tracebacks in _run_chain stay the primary record)."""
    original = ValueError("chain exploded")
    up = BackgroundStemUploader(max_pending=2)
    with (
        caplog.at_level(logging.ERROR, logger=bgu.__name__),
        pytest.raises(RuntimeError, match="body boom"),
        up,
    ):
        up.submit(_raiser(original), label="c1")
        _wait_until(lambda: up.failure is original)
        raise RuntimeError("body boom")
    text = " ".join(record.getMessage() for record in caplog.records)
    assert "'c1'" in text  # the failed label, still present
    assert "ValueError" in text and "chain exploded" in text  # the exception detail


# ------------------------------------------------------------- chain factory e2e


def test_chain_factory_order_upload_free_marker(tmp_path, monkeypatch):
    """End-to-end through the uploader: upload -> free (real tmp files) -> marker."""
    stem_dir = tmp_path / "stem1"
    stem_dir.mkdir()
    shards = [stem_dir / f"slot{i}.shard0.pt" for i in range(3)]
    for shard in shards:
        shard.write_bytes(b"x" * 16)
    keep = stem_dir / "report.json"  # non-matching file must survive the free
    keep.write_text("{}")
    events: list[str] = []
    monkeypatch.setattr(
        bgu.hub, "_upload", _fake_upload_factory(events, expect_exists=tuple(shards))
    )

    def after_verify() -> None:
        assert not any(s.exists() for s in shards)
        events.append("marker")

    with BackgroundStemUploader(max_pending=2) as up:
        up.submit(
            hub_upload_then_free(
                stem_dir,
                repo_id="org/data",
                repo_type="dataset",
                dest="issue2616/stem1",
                free_glob="slot*.shard*.pt",
                after_verify=after_verify,
            ),
            label="stem1",
        )
    assert events == ["upload", "marker"]
    assert not any(s.exists() for s in shards)  # shards absent after unlink
    assert keep.exists()  # only the free_glob matches were freed
    up._thread.join(WAIT)  # clean context exit stops the worker (bounded positive wait)
    assert not up._thread.is_alive()


def test_chain_records_upload_wall_into_report(tmp_path, monkeypatch):
    """With report= provided, the chain writes upload_wall_s after the verified
    upload and BEFORE the marker step (an after_verify payload holding the dict
    REFERENCE carries it — module-docstring note (b)), plus chain_wall_s at end."""
    stem_dir = tmp_path / "stem0"
    stem_dir.mkdir()
    shard = stem_dir / "slot0.shard0.pt"
    shard.write_bytes(b"a" * 8)
    events: list[str] = []
    monkeypatch.setattr(bgu.hub, "_upload", _fake_upload_factory(events))
    report: dict[str, float] = {}
    seen_at_marker: dict[str, float] = {}

    chain = hub_upload_then_free(
        stem_dir,
        repo_id="org/data",
        repo_type="dataset",
        dest="issue2616/stem0",
        free_glob="slot*.shard*.pt",
        after_verify=lambda: seen_at_marker.update(report),
        report=report,
    )
    chain()
    assert report["upload_wall_s"] >= 0.0
    assert seen_at_marker.get("upload_wall_s") == report["upload_wall_s"]
    assert report["chain_wall_s"] >= report["upload_wall_s"]
    assert not shard.exists()  # free behavior unchanged when report= is passed


# --------------------------------------------------------- worker lifecycle stop


def test_worker_exits_after_clean_join():
    """A clean join() stops the daemon worker — repeated uploader construction must
    not accumulate live threads (bounded POSITIVE wait on thread exit)."""
    up = BackgroundStemUploader(max_pending=2)
    ran: list[str] = []
    up.submit(lambda: ran.append("c1"), label="c1")
    up.join()
    up._thread.join(WAIT)
    assert not up._thread.is_alive()
    assert ran == ["c1"]


def test_submit_after_close_raises():
    """close() is idempotent; a submit after close fails loud, never silently unrun."""
    up = BackgroundStemUploader(max_pending=2)
    up.join()  # clean no-op drain -> close() -> worker stop
    up.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        up.submit(lambda: None, label="late")


# ------------------------------------------------------ worker internal errors


def test_worker_internal_error_records_failure(monkeypatch):
    """An exception in the queue-handling path (not inside work()) still records
    into _failure and surfaces at the next submit and at join — never a silent
    thread death."""
    up = BackgroundStemUploader(max_pending=2)
    injected = RuntimeError("injected internal queue error")
    real_get = up._queue.get
    fired = {"n": 0}

    def flaky_get(*args, **kwargs):
        if fired["n"] == 0:
            fired["n"] += 1
            raise injected
        return real_get(*args, **kwargs)

    monkeypatch.setattr(up._queue, "get", flaky_get)
    _wait_until(lambda: up.failure is injected)
    with pytest.raises(RuntimeError) as excinfo:
        up.submit(lambda: None, label="c1")
    assert excinfo.value.__cause__ is injected
    with pytest.raises(RuntimeError) as excinfo2:
        up.join()
    assert excinfo2.value.__cause__ is injected


# ------------------------------------------------------------ call-shape pin


def test_hub_upload_signature_has_raise_on_error():
    """Un-mocked pin of the hub._upload call shape the chain factory binds to."""
    params = inspect.signature(hub._upload).parameters
    assert "raise_on_error" in params
    assert "path_in_repo" in params
    assert list(params)[:4] == ["local_path", "repo_id", "repo_type", "path_in_repo"]
