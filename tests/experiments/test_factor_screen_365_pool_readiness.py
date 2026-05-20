"""Pool-readiness guard tests for ``_run_cell_mode``.

Round-8 (issue #365) forensics: the round-7 dispatcher launched all 96 cells
in parallel while pool-gen was still mid-flight, and 94 of 96 cells crashed
at startup with ``FileNotFoundError`` on the missing ``_offpolicy.jsonl``
pools. The fix is a per-cell exponential-backoff wait so cells that launch
too early sleep instead of crashing.

These tests cover ``_wait_for_pool`` and the new ``PoolNotReadyError``:

  * **Wait-then-success:** schedule a file write after a short delay and
    assert the wait returns once the file appears.
  * **Wait-then-raise:** point the wait at a path that never appears and
    assert ``PoolNotReadyError`` is raised after the budget elapses.
  * **Backoff cap:** mock ``time.sleep`` + ``time.monotonic`` and assert
    the per-iteration delay grows exponentially but is capped at 600s.
  * **Already-exists short-circuit:** when the file is already on disk the
    wait must return immediately without calling ``time.sleep``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365 import __main__ as fs365_main


def test_wait_for_pool_returns_when_file_already_exists(tmp_path: Path) -> None:
    """If the pool is already on disk, the wait must return immediately.

    No sleeps, no backoff: this is the steady-state happy path once pool-gen
    has caught up. We assert ``time.sleep`` is never invoked.
    """
    pool_path = tmp_path / "source-librarian_a0_b0_c0.jsonl"
    pool_path.write_text("{}\n")

    with mock.patch.object(fs365_main.time, "sleep") as sleep_mock:
        fs365_main._wait_for_pool(pool_path, max_wait_s=60)

    sleep_mock.assert_not_called()


def test_wait_for_pool_succeeds_when_file_appears_during_wait(tmp_path: Path) -> None:
    """Schedule a delayed file write; ``_wait_for_pool`` must return cleanly.

    Uses a background thread to materialise the pool after a short delay and
    a patched ``time.sleep`` that polls more frequently than the real backoff
    so the whole test completes in well under a second. This is the
    pool-gen-catches-up scenario the guard was added to handle.
    """
    pool_path = tmp_path / "source-surgeon_a1_b0_c1_offpolicy.jsonl"

    # Capture the real sleep BEFORE patching so the fast-poll replacement
    # below doesn't recurse into itself.
    real_sleep = time.sleep

    def _materialise_after(delay_s: float) -> None:
        real_sleep(delay_s)
        pool_path.write_text("{}\n")

    writer = threading.Thread(target=_materialise_after, args=(0.05,), daemon=True)
    writer.start()

    # Patch the module's time.sleep with a fast poll so the 60s backoff doesn't
    # actually elapse, but keep monotonic real so the max-wait budget logic
    # remains a faithful exercise of the loop.
    with mock.patch.object(fs365_main.time, "sleep", lambda _s: real_sleep(0.02)):
        fs365_main._wait_for_pool(pool_path, max_wait_s=5)

    writer.join(timeout=1.0)
    assert pool_path.exists()


def test_wait_for_pool_raises_pool_not_ready_after_budget(tmp_path: Path) -> None:
    """If the pool never appears, raise ``PoolNotReadyError`` after ``max_wait_s``.

    Patches both ``time.monotonic`` and ``time.sleep`` so the budget elapses
    in microseconds instead of 30 minutes. The error must be the specific
    ``PoolNotReadyError`` subclass (not a generic ``FileNotFoundError``)
    so the caller can distinguish "pool race" from "wrong path".
    """
    pool_path = tmp_path / "source-programmer_a0_b1_c0.jsonl"
    assert not pool_path.exists()

    fake_clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    with (
        mock.patch.object(fs365_main.time, "monotonic", lambda: next(fake_clock)),
        mock.patch.object(fs365_main.time, "sleep") as sleep_mock,
        pytest.raises(fs365_main.PoolNotReadyError) as exc_info,
    ):
        fs365_main._wait_for_pool(pool_path, max_wait_s=2)

    assert "Pool not generated" in str(exc_info.value)
    assert str(pool_path) in str(exc_info.value)
    # ``sleep`` must have been called at least once before the raise.
    assert sleep_mock.call_count >= 1


def test_wait_for_pool_backoff_caps_at_600s(tmp_path: Path) -> None:
    """Per-iteration delay grows exponentially but never exceeds 600s.

    Mocks ``time.sleep`` and ``time.monotonic`` to fast-forward through many
    iterations and asserts no recorded sleep argument is greater than 600s.
    The sequence should be 60, 120, 240, 480, 600, 600, 600, ... (cap at 600).
    """
    pool_path = tmp_path / "source-librarian_a1_b1_c1.jsonl"

    # Build a monotonic stream that always reports "still under budget" so the
    # loop iterates many times before the test stops it.
    clock_values = [0.0] + [1.0] * 50  # very long budget; sleep is mocked
    fake_clock = iter(clock_values)
    sleep_calls: list[float] = []

    def _record_and_stop(s: float) -> None:
        sleep_calls.append(s)
        if len(sleep_calls) >= 10:
            # Materialise the pool to exit the loop cleanly.
            pool_path.write_text("{}\n")

    with (
        mock.patch.object(fs365_main.time, "monotonic", lambda: next(fake_clock)),
        mock.patch.object(fs365_main.time, "sleep", side_effect=_record_and_stop),
    ):
        fs365_main._wait_for_pool(pool_path, max_wait_s=10_000_000)

    assert sleep_calls, "Expected at least one sleep call"
    assert all(s <= 600.0 for s in sleep_calls), (
        f"Backoff exceeded 600s cap; recorded delays: {sleep_calls}"
    )
    # Verify the expected exponential ramp before the cap: 60, 120, 240, 480, 600...
    assert sleep_calls[0] == pytest.approx(60.0)
    assert sleep_calls[1] == pytest.approx(120.0)
    assert sleep_calls[2] == pytest.approx(240.0)
    assert sleep_calls[3] == pytest.approx(480.0)
    # From the 5th call onward, every delay must be capped at exactly 600s.
    for delay in sleep_calls[4:]:
        assert delay == pytest.approx(600.0)


def test_pool_not_ready_error_is_file_not_found_subclass() -> None:
    """``PoolNotReadyError`` must be a ``FileNotFoundError`` subclass.

    Existing call sites (and the cell-mode error handler) catch
    ``FileNotFoundError``; the new subclass preserves that compatibility so
    code that doesn't yet know about ``PoolNotReadyError`` still treats it
    as a missing-file error.
    """
    assert issubclass(fs365_main.PoolNotReadyError, FileNotFoundError)
