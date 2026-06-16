"""Issue #640 — the diagonal-mode backend-parity precheck fires BEFORE cells.

Plan v6 §4.2 part 4 (Phase-2 reconciler Must-Fix): under ``--target diagonal`` the
inherited in-loop parity gate is keyed on ``column_id == PARITY_COLUMN`` (broad_em),
which diagonal mode NEVER sets — so it is silently skipped. The decoupled one-shot
precheck must therefore fire the ``bad_medical x broad_em`` HALT ONCE on seed-0
BEFORE any ``diagonal_source_seed{seed}.json`` cell is written, reading
``COLUMNS[PARITY_COLUMN]`` directly (independent of the diagonal target map), and
must NOT be persisted as a diagonal cell.

These tests use a call-order trace (the precheck stubbed to record its index, the
per-seed JSON write observed via the on-disk file appearing) and an ordering
assertion, plus a HALT test (non-smoke divergence raises SystemExit before any
cell).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _driver():
    return importlib.import_module("issue640_postfix_carrier")


def _stub_gpu_surface(monkeypatch, d, *, events):
    """Stub the GPU + judge surface; record cell-write events in order."""
    monkeypatch.setattr(d, "load_base_and_tokenizer", lambda: (object(), object()))
    monkeypatch.setattr(d, "assert_marker_token", lambda *a, **k: None)
    monkeypatch.setattr(d, "assert_postfix_tokenization", lambda *a, **k: None)
    monkeypatch.setattr(d, "download_adapter", lambda row, seed: Path("/tmp/fake"))
    monkeypatch.setattr(d, "_read_adapter_config", lambda *a, **k: {})
    monkeypatch.setattr(d, "gauge_from_config", lambda cfg: (45.25, True))
    monkeypatch.setattr(d, "expected_gauge_band", lambda row: (40.0, 50.0))
    monkeypatch.setattr(d, "attach_adapter", lambda base, ad: object())
    monkeypatch.setattr(d, "detach_adapter", lambda model, base: base)
    monkeypatch.setattr(d, "_phase2_target_columns", lambda: {"bad_medical": "broad_em"})
    monkeypatch.setattr(
        d, "_diagonal_target_columns", lambda: {"bad_medical": "fam_expr_bad_medical"}
    )

    class _Col:
        max_new_tokens = 8
        n_samples = 1
        temperature = 0.0

    class _FakeColumns:
        def __getitem__(self, key):
            return _Col()

    import explore_persona_space.experiments.behavior_testbed_545.columns as cols
    import explore_persona_space.experiments.behavior_testbed_545.eval_battery as eb

    monkeypatch.setattr(cols, "COLUMNS", _FakeColumns())
    monkeypatch.setattr(eb, "battery_probes", lambda col, cap: [{"question": "q"}])
    monkeypatch.setattr(eb, "render_chat", lambda tok, q, sys_key: "prompt")
    monkeypatch.setattr(d, "generate_patched", lambda *a, **k: [["resp"]])
    monkeypatch.setattr(d, "_judge_completions", lambda col, probes, comps: 0.5)
    monkeypatch.setattr(d, "_persist_raw", lambda *a, **k: events.append("persist_raw"))

    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_precheck_fires_before_any_diagonal_cell(monkeypatch, tmp_path):
    """Order: precheck -> (then) the first per-cell judge/write. Captured via a trace."""
    d = _driver()
    out_root = tmp_path / "issue_640"
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(out_root))

    events: list[str] = []
    _stub_gpu_surface(monkeypatch, d, events=events)

    def _fake_precheck(base, tokenizer, *, probe_cap, device, smoke):
        events.append("precheck")
        return {"status": "skipped(smoke)", "rate": 0.11}

    monkeypatch.setattr(d, "_diagonal_parity_precheck", _fake_precheck)

    d.run_phase2_postfix_patch(
        ["bad_medical"], [0], probe_cap=1, device="cpu", smoke=True, target="diagonal"
    )

    assert "precheck" in events, "the diagonal-mode parity precheck must run"
    assert events[0] == "precheck", (
        f"the precheck must fire FIRST, before any per-cell work; got order {events}"
    )
    # The precheck record is persisted as its own one-shot JSON (not a diagonal cell).
    assert (out_root / "backend_parity_diagonal_seed0.json").exists()
    # And it is NOT a diagonal cell in the per-seed source JSON.
    import json

    diag = json.loads((out_root / "diagonal_source_seed0.json").read_text())
    assert all("broad_em" not in k for k in diag["cells"]), (
        "the parity probe (bad_medical|broad_em) must NOT appear as a diagonal Δsource cell"
    )


def test_leakage_mode_does_not_run_diagonal_precheck(monkeypatch, tmp_path):
    """Under --target leakage the decoupled precheck must NOT fire (v3 path unchanged)."""
    d = _driver()
    out_root = tmp_path / "issue_640"
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(out_root))

    events: list[str] = []
    _stub_gpu_surface(monkeypatch, d, events=events)
    monkeypatch.setattr(
        d, "_diagonal_parity_precheck", lambda *a, **k: events.append("precheck") or {}
    )

    d.run_phase2_postfix_patch(
        ["bad_medical"], [0], probe_cap=1, device="cpu", smoke=True, target="leakage"
    )
    assert "precheck" not in events, "leakage mode must use the in-loop gate, not the precheck"
    assert not (out_root / "backend_parity_diagonal_seed0.json").exists()


def test_precheck_halts_on_divergence_non_smoke(monkeypatch):
    """Non-smoke: |parity_rate - L_545| > tol raises SystemExit (failure_class: code).

    Drives _diagonal_parity_precheck directly with a judged rate far off the #545
    reference and asserts the HALT — pins that the gate is a real kill, not a warn.
    """
    d = _driver()

    monkeypatch.setattr(d, "download_adapter", lambda row, seed: Path("/tmp/fake"))
    monkeypatch.setattr(d, "_read_adapter_config", lambda *a, **k: {})
    monkeypatch.setattr(d, "gauge_from_config", lambda cfg: (45.25, True))
    monkeypatch.setattr(d, "expected_gauge_band", lambda row: (40.0, 50.0))
    monkeypatch.setattr(d, "attach_adapter", lambda base, ad: object())
    monkeypatch.setattr(d, "detach_adapter", lambda model, base: base)

    class _Col:
        max_new_tokens = 8
        n_samples = 1
        temperature = 0.0

    class _FakeColumns:
        def __getitem__(self, key):
            return _Col()

    import explore_persona_space.experiments.behavior_testbed_545.columns as cols
    import explore_persona_space.experiments.behavior_testbed_545.eval_battery as eb

    monkeypatch.setattr(cols, "COLUMNS", _FakeColumns())
    monkeypatch.setattr(eb, "battery_probes", lambda col, cap: [{"question": "q"}])
    monkeypatch.setattr(eb, "render_chat", lambda tok, q, sys_key: "prompt")
    monkeypatch.setattr(d, "generate_patched", lambda *a, **k: [["resp"]])
    # A wildly divergent unpatched rate (0.95 vs L_545 ~0.11) -> HALT.
    monkeypatch.setattr(d, "_judge_completions", lambda col, probes, comps: 0.95)

    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(SystemExit) as ei:
        d._diagonal_parity_precheck(object(), object(), probe_cap=1, device="cpu", smoke=False)
    assert "backend parity" in str(ei.value).lower()

    # Smoke mode: same divergence logs but does NOT halt (matches v3 line-494).
    rec = d._diagonal_parity_precheck(object(), object(), probe_cap=1, device="cpu", smoke=True)
    assert rec["status"] == "skipped(smoke)"
