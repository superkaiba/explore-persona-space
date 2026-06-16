"""Issue #640 — the ``--target`` flag switches column map + output filename.

Plan v6 §4.2 part 3 locks in: ``--target diagonal`` selects the diagonal column
map AND writes ``diagonal_source_seed{seed}.json``; ``--target leakage`` (default)
keeps v3's off-diagonal map AND ``patch_cells_postfix_seed{seed}.json`` (so v3 is
byte-for-byte reproducible and the diagonal run can never overwrite v3's cells).

These tests drive ``main()`` with the GPU-bound Phase-2 body monkeypatched to a
capture stub (so NO model load happens), asserting the flag threads ``target``
through and that diagonal mode forces the Phase-2-only / 7-judged-rate-row
contract. A separate unit asserts the per-target output FILENAME mapping directly.
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


def _run_main(monkeypatch, argv, *, tmp_root):
    """Drive main() with GPU phases stubbed; return the captured phase-2 kwargs."""
    d = _driver()
    captured: dict = {}

    def _fake_phase2(rows, seeds, *, probe_cap, device, smoke, target):
        captured.update(
            rows=list(rows), seeds=list(seeds), probe_cap=probe_cap, smoke=smoke, target=target
        )

    monkeypatch.setattr(d, "run_phase2_postfix_patch", _fake_phase2)
    # Phase 1 must never run under diagonal mode; stub it to flag if it does.
    monkeypatch.setattr(
        d, "run_phase1_postfix_kv_shift", lambda *a, **k: captured.update(phase1_ran=True)
    )
    monkeypatch.setattr(d, "upload_raw_completions", lambda *a, **k: None)
    monkeypatch.setattr(d, "write_sentinel", lambda *a, **k: None)
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_root))
    monkeypatch.setattr(sys, "argv", ["issue640_postfix_carrier.py", *argv])
    rc = d.main()
    return rc, captured


def test_default_target_is_leakage(monkeypatch, tmp_path):
    """No --target -> leakage (v3 default); all 8 rows, Phase 1 + 2 both run."""
    rc, cap = _run_main(monkeypatch, ["--skip-upload"], tmp_root=tmp_path)
    assert rc == 0
    assert cap["target"] == "leakage"
    assert cap["rows"] == list(_driver().ALL_8_ROWS)
    assert cap.get("phase1_ran") is True, "leakage --phase all must run Phase 1"


def test_target_diagonal_threads_and_defaults_to_judged_rate_rows(monkeypatch, tmp_path):
    """--target diagonal -> diagonal target, 7 judged-rate rows, Phase-2-only."""
    rc, cap = _run_main(monkeypatch, ["--target", "diagonal", "--skip-upload"], tmp_root=tmp_path)
    assert rc == 0
    assert cap["target"] == "diagonal"
    assert cap["rows"] == list(_driver().PHASE2_ROWS_JUDGED_RATE)
    assert "marker" not in cap["rows"]
    assert cap.get("phase1_ran") is not True, (
        "diagonal mode is Phase-2-only (§8); the column-independent Phase 1 must NOT re-run"
    )


def test_target_diagonal_rejects_marker_row(monkeypatch, tmp_path):
    """--target diagonal --rows marker halts BEFORE model load (judge would crash)."""
    with pytest.raises(SystemExit) as ei:
        _run_main(
            monkeypatch,
            ["--target", "diagonal", "--rows", "marker", "--skip-upload"],
            tmp_root=tmp_path,
        )
    assert "not judged-rate diagonal rows" in str(ei.value)


def test_target_diagonal_rejects_phase1(monkeypatch, tmp_path):
    """--target diagonal --phase postfix-kv-shift halts (Phase 1 is reused from v3)."""
    with pytest.raises(SystemExit) as ei:
        _run_main(
            monkeypatch,
            ["--target", "diagonal", "--phase", "postfix-kv-shift"],
            tmp_root=tmp_path,
        )
    assert "Phase-2-only" in str(ei.value)


def test_output_filename_switches_per_target(monkeypatch, tmp_path):
    """The per-seed output FILENAME differs by target; leakage keeps the v3 name.

    Drives run_phase2_postfix_patch with every GPU + judge dependency stubbed so
    a single (row, seed) cell flows through to the per-seed JSON write, then
    asserts the written filename. This pins §4.2 part 3 (the filename switch is
    in the driver, not a directory-only override).
    """
    d = _driver()
    out_root = tmp_path / "issue_640"
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(out_root))

    # Stub the full GPU + judge surface so no model loads.
    monkeypatch.setattr(d, "load_base_and_tokenizer", lambda: (object(), object()))
    monkeypatch.setattr(d, "assert_marker_token", lambda *a, **k: None)
    monkeypatch.setattr(d, "assert_postfix_tokenization", lambda *a, **k: None)
    monkeypatch.setattr(d, "download_adapter", lambda row, seed: Path("/tmp/fake"))
    monkeypatch.setattr(d, "_read_adapter_config", lambda *a, **k: {})
    monkeypatch.setattr(d, "gauge_from_config", lambda cfg: (45.25, True))
    monkeypatch.setattr(d, "expected_gauge_band", lambda row: (40.0, 50.0))
    monkeypatch.setattr(d, "attach_adapter", lambda base, ad: object())
    monkeypatch.setattr(d, "detach_adapter", lambda model, base: base)
    monkeypatch.setattr(d, "_persist_raw", lambda *a, **k: None)

    # A fake COLUMNS[col] with the attributes battery_probes/gen_kwargs need.
    class _Col:
        max_new_tokens = 8
        n_samples = 1
        temperature = 0.0

    import explore_persona_space.experiments.behavior_testbed_545.columns as cols
    import explore_persona_space.experiments.behavior_testbed_545.eval_battery as eb

    monkeypatch.setattr(cols, "COLUMNS", _FakeColumns(_Col()))
    monkeypatch.setattr(eb, "battery_probes", lambda col, cap: [{"question": "q"}])
    monkeypatch.setattr(eb, "render_chat", lambda tok, q, sys_key: "prompt")
    monkeypatch.setattr(d, "generate_patched", lambda *a, **k: [["resp"]])
    monkeypatch.setattr(d, "_judge_completions", lambda col, probes, comps: 0.5)
    # diagonal mode runs the parity precheck before the loop — stub it out.
    monkeypatch.setattr(d, "_diagonal_parity_precheck", lambda *a, **k: {"status": "skipped"})
    # Pin both column-map resolvers to a single bad_medical cell (the leakage
    # resolver otherwise reads #545's L_matrix and needs a real COLUMNS.items()).
    monkeypatch.setattr(d, "_phase2_target_columns", lambda: {"bad_medical": "broad_em"})
    monkeypatch.setattr(
        d, "_diagonal_target_columns", lambda: {"bad_medical": "fam_expr_bad_medical"}
    )

    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    # leakage: writes patch_cells_postfix_seed0.json (v3 name).
    d.run_phase2_postfix_patch(
        ["bad_medical"], [0], probe_cap=1, device="cpu", smoke=True, target="leakage"
    )
    assert (out_root / "patch_cells_postfix_seed0.json").exists()
    assert not (out_root / "diagonal_source_seed0.json").exists()

    # diagonal: writes diagonal_source_seed0.json, NOT the v3 name.
    d.run_phase2_postfix_patch(
        ["bad_medical"], [0], probe_cap=1, device="cpu", smoke=True, target="diagonal"
    )
    assert (out_root / "diagonal_source_seed0.json").exists()


class _FakeColumns:
    """A dict-like COLUMNS stub returning the same fake column for any key."""

    def __init__(self, col):
        self._col = col

    def __getitem__(self, key):
        return self._col
