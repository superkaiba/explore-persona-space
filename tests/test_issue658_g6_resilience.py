# ruff: noqa: RUF003
# Intentional Unicode (×, ※) in scientific test docstrings.
"""Round-3 resilience regression for issue #658's G6 marker loop.

Rounds 1 + 2 both died at marker context #31 (``f3_icl_json_k2``) with a
deterministic per-context exception. The GCE EXIT trap powered the VM off and
the 30 partial e0_gen/*.json files + the log were LOST (zero HF uploads at the
time of death). The fix adds per-context exception isolation + periodic partial
upload to ``generate_e0_completions`` / ``_run_marker_loop`` so the SAME
deterministic failure (a) gets DIAGNOSED next run (full traceback to stdout +
an ``*__marker__ERROR.json`` artifact) and (b) does NOT destroy the contexts
that already ran.

This test drives ``_run_marker_loop`` directly with a stub ``_gen_marker_slot``
that succeeds on contexts #1 and #3 and RAISES on #2 (mirroring the
deterministic per-context crash). It is CPU-only and does NO network (uploads
disabled via ``upload=False``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_extract_base_store as store  # noqa: E402


def _instances():
    """Three minimal battery instances (the loop only reads ``inst['id']``)."""
    return [{"id": "ctx_ok_a"}, {"id": "ctx_boom"}, {"id": "ctx_ok_b"}]


def test_marker_loop_isolates_one_failing_context(tmp_path, monkeypatch):
    """Context #2 raises; #1 and #3 still land + #2 gets an ERROR artifact, no raise."""
    e0_dir = tmp_path / "e0_gen"
    e0_dir.mkdir()

    # A fixed 1-probe battery so the loop never touches the network/Betley pool.
    monkeypatch.setattr(store, "load_e0_battery", lambda col_id, n, probes: ["probe-0"])

    boom_msg = "deterministic per-context crash at f3_icl_json_k2"

    def _stub_gen_marker_slot(
        model, tokenizer, inst, battery, col, out_path, compute_fn, max_new_tokens
    ):
        # Mirror the real _gen_marker_slot's persistence contract: write a marker
        # JSON keyed by context_id — except for the deterministic-crash context.
        if inst["id"] == "ctx_boom":
            raise RuntimeError(boom_msg)
        store.dump_json(
            {"context_id": inst["id"], "column_id": col.column_id, "dv": col.dv, "marker_slot": []},
            out_path,
        )

    monkeypatch.setattr(store, "_gen_marker_slot", _stub_gen_marker_slot)

    # No exception must propagate (per-context isolation is the whole point).
    status = store._run_marker_loop(
        model=object(),
        tokenizer=object(),
        instances=_instances(),
        e0_dir=e0_dir,
        n_battery=1,
        compute_marker_slot_stats=None,
        mnt_fn=lambda col: 8,
        smoke=True,
        upload=False,  # CPU-only test: never hit the HF data repo
    )

    # Contexts #1 and #3 produced their marker JSONs.
    assert (e0_dir / "ctx_ok_a__marker.json").is_file()
    assert (e0_dir / "ctx_ok_b__marker.json").is_file()
    # The successful context files carry NO __marker__ERROR.json.
    assert not (e0_dir / "ctx_ok_a__marker__ERROR.json").exists()
    assert not (e0_dir / "ctx_ok_b__marker__ERROR.json").exists()

    # Context #2 produced an ERROR artifact with the traceback + exception info.
    err_path = e0_dir / "ctx_boom__marker__ERROR.json"
    assert err_path.is_file(), "the failing context must leave a __marker__ERROR.json breadcrumb"
    err = json.loads(err_path.read_text())
    assert err["context_id"] == "ctx_boom"
    assert err["exception_type"] == "RuntimeError"
    assert boom_msg in err["exception_str"]
    assert "Traceback" in err["traceback"] and boom_msg in err["traceback"]
    assert err.get("ts"), "the error artifact must record a timestamp"

    # The returned telemetry distinguishes done vs errored contexts.
    assert status["done"] == ["ctx_ok_a", "ctx_ok_b"]
    assert status["errors"] == ["ctx_boom"]


def test_marker_loop_partial_upload_failure_does_not_kill_run(tmp_path, monkeypatch):
    """A transient HF upload failure is swallowed — the loop still completes."""
    e0_dir = tmp_path / "e0_gen"
    e0_dir.mkdir()
    monkeypatch.setattr(store, "load_e0_battery", lambda col_id, n, probes: ["probe-0"])

    def _ok_gen_marker_slot(
        model, tokenizer, inst, battery, col, out_path, compute_fn, max_new_tokens
    ):
        store.dump_json({"context_id": inst["id"], "marker_slot": []}, out_path)

    monkeypatch.setattr(store, "_gen_marker_slot", _ok_gen_marker_slot)

    def _boom_upload(e0_dir_arg, smoke):
        raise RuntimeError("HF Hub 503 (transient)")

    monkeypatch.setattr(store, "_upload_partial_e0", _boom_upload)

    # upload=True so the end-of-loop _upload_partial_e0 fires (and fails); the
    # loop must still return cleanly with every context marked done.
    status = store._run_marker_loop(
        model=object(),
        tokenizer=object(),
        instances=_instances(),
        e0_dir=e0_dir,
        n_battery=1,
        compute_marker_slot_stats=None,
        mnt_fn=lambda col: 8,
        smoke=False,
        upload=True,
    )
    assert status["errors"] == []
    assert status["done"] == ["ctx_ok_a", "ctx_boom", "ctx_ok_b"]
    for cid in ("ctx_ok_a", "ctx_boom", "ctx_ok_b"):
        assert (e0_dir / f"{cid}__marker.json").is_file()
