"""#1947 crash-fix r11 pins: marker-cell teardown order + exception unmasking.

``run_marker_cell``'s finally previously ran ``backend.close()`` FIRST —
while the ~15 GiB HF base model was still referenced — so
``issue1333_dispatch._wait_engine_release``'s drain-wait timed out
DETERMINISTICALLY (r9 contract: the wait may only run once the caller holds
NO live HF-weight reference; observed as ~15,068 MiB never-draining residue
across 3 failed production rounds, 8/8 attempts). The raise from the close
also MASKED any in-flight inner exception from ``_ladder_cell``.

The r11 fix extracts the finally into ``_teardown_marker_cell`` (free the
base model via the ``_free_hf`` rebind form + post-rebind flush, THEN close,
with a close-time exception suppressed only while an inner exception is
propagating). These tests pin:

- (a) teardown ORDER — the base-model release runs BEFORE ``backend.close``;
- (a') the single-slot ``model_box`` handoff drops ALL references (real
  ``_free_hf`` body — a plain parameter would keep the caller's binding
  alive through the drain-wait, re-creating the bug);
- (b) MASKING — inner exception X propagates even when close raises Y;
- (c) fail-fast — with no inner exception in flight, a close failure raises;
- (d) wiring — ``run_marker_cell``'s finally calls the helper (no direct
  ``backend.close``) and ``del``s this frame's ``base_model`` binding.
"""

from __future__ import annotations

import ast
import gc
import logging
import sys
import weakref
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1333_dispatch as d1333  # noqa: E402
import issue1947_worker as w  # noqa: E402


class _InnerError(RuntimeError):
    """Stands in for a crash propagating out of ``mk._ladder_cell``."""


class _CloseBoom(RuntimeError):
    """Stands in for ``backend.close`` failing (drain-wait timeout etc.)."""


class _RecordingBackend:
    def __init__(self, calls: list[str]):
        self._calls = calls

    def close(self, label: str) -> None:
        self._calls.append(f"close:{label}")


class _BoomBackend:
    def close(self, label: str) -> None:
        raise _CloseBoom(f"close failed for {label}")


def test_teardown_frees_base_model_before_backend_close(monkeypatch):
    """(a) The _free_hf release runs strictly BEFORE backend.close."""
    calls: list[str] = []

    def fake_free(model):
        calls.append("free")
        assert model == "MODEL"
        return None

    monkeypatch.setattr(d1333, "_free_hf", fake_free)
    box = ["MODEL"]
    w._teardown_marker_cell(_RecordingBackend(calls), box, "cellA")
    assert calls == ["free", "close:cellA"]
    assert box == [], "the box must hand its sole reference over to the helper"


def test_box_handoff_drops_all_references_before_close():
    """(a') Real _free_hf body: by close time NO reference to the model
    survives — the property the drain-wait needs (a plain parameter would
    fail this: the caller's binding would keep the weights resident)."""

    class Dummy:
        pass

    obj = Dummy()
    ref = weakref.ref(obj)
    seen: dict[str, bool] = {}

    class AssertingBackend:
        def close(self, label: str) -> None:
            gc.collect()
            seen["model_gone_at_close"] = ref() is None

    box = [obj]
    del obj
    w._teardown_marker_cell(AssertingBackend(), box, "cellD")
    assert seen["model_gone_at_close"], (
        "base model still referenced when backend.close ran — the "
        "_wait_engine_release drain-wait would time out deterministically"
    )


def test_inner_exception_propagates_over_close_failure(monkeypatch, caplog):
    """(b) When _ladder_cell raises X and close raises Y, X propagates."""
    monkeypatch.setattr(d1333, "_free_hf", lambda model: None)
    caplog.set_level(logging.ERROR, logger="issue1947.worker")

    with pytest.raises(_InnerError):
        try:
            raise _InnerError("ladder crashed")
        finally:
            w._teardown_marker_cell(_BoomBackend(), [None], "cellB")

    assert any(
        "suppressed" in rec.getMessage() and "cellB" in rec.getMessage() for rec in caplog.records
    ), "the suppressed close failure must be logged with the cell slug"


def test_close_failure_raises_without_inner_exception(monkeypatch):
    """(c) Fail-fast preserved: no inner exception in flight -> Y raises."""
    monkeypatch.setattr(d1333, "_free_hf", lambda model: None)
    assert sys.exc_info() == (None, None, None)
    with pytest.raises(_CloseBoom):
        w._teardown_marker_cell(_BoomBackend(), [None], "cellC")


def test_run_marker_cell_finally_wired_to_teardown_helper():
    """(d) run_marker_cell's finally routes through the helper: no direct
    backend.close, and this frame's base_model binding is del'd first."""
    src = (SCRIPTS / "issue1947_worker.py").read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "run_marker_cell"
    )
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "run_marker_cell lost its try/finally teardown"
    final_src = "\n".join(ast.unparse(s) for t in tries for s in t.finalbody)
    assert "_teardown_marker_cell(backend" in final_src
    assert "backend.close" not in final_src, (
        "close must run inside the guarded helper, never directly in the finally"
    )
    assert "del base_model" in final_src, (
        "this frame's binding must drop before the drain-wait (r9 contract)"
    )
