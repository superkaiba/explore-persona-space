"""Regression tests for the issue-2222 §7 cap-hit halt across resume (round-2 BLOCKER).

Pre-fix, ``run_gen`` recorded a halted dataset's ``p2_gen`` phase as COMPLETE in
the manifest before ``main()`` exited ``RC_CAP_HIT``, so ANY relaunch
resume-skipped the dataset, recomputed ``halted`` as empty, and silently
proceeded to produce ``base_respavg`` over cap-biased generations. These tests
write a fingerprint-matched manifest fixture with ``cap_hit_final`` over the §7
bar and assert (a) the resume path re-derives the halt, and (b) ``main()``'s
halt-file gate refuses ``--phase capture/all`` absent the explicit override.
Everything runs offline on ``tmp_path`` (no GPU, no network, no vLLM import at
module level).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2222_capture as cap
import issue2222_lib as lib

FP = "fp-test"


def _write_manifest(data_root: Path, ds: str, cap_hit_final: float) -> None:
    """Fingerprint-matched p2_gen-complete manifest + the phase's local file."""
    ds_dir = lib.capture_dir(data_root, ds)
    ds_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": ds,
        "resume_fingerprint": FP,
        "phases": {
            "p2_gen": {
                "n_rows": 8,
                "cap_hit_initial": cap_hit_final,
                "cap_hit_final": cap_hit_final,
                "n_regen": 0,
                "n_regen_skipped_budget": 0,
                "n_empty_completions": 0,
                "jsonl": "unused",
            }
        },
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest))
    raw = lib.rawcomp_path(data_root, ds)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("")


def _run_gen_resumed(data_root: Path, ds: str) -> list[str]:
    """Drive run_gen down the resume-skip branch only (no tokenizer/vLLM work)."""
    base = {ds: {"resume_fingerprint": FP}}
    return cap.run_gen(data_root, [ds], {}, None, base, seed=1, skip_upload=True, hub_resume=False)


def test_resume_skip_recomputes_cap_hit_halt(tmp_path: Path) -> None:
    """Over-bar cap_hit_final in a resume-skipped manifest re-enters `halted`."""
    ds = "evil_normal"
    _write_manifest(tmp_path, ds, cap_hit_final=0.05)
    assert lib.CAP_HIT_MAX_FRACTION < 0.05  # the fixture sits over the §7 bar
    assert _run_gen_resumed(tmp_path, ds) == [ds]


def test_resume_skip_under_bar_not_halted(tmp_path: Path) -> None:
    ds = "evil_normal"
    _write_manifest(tmp_path, ds, cap_hit_final=0.0)
    assert _run_gen_resumed(tmp_path, ds) == []


def test_resume_skip_missing_cap_hit_field_fails_loud(tmp_path: Path) -> None:
    """A fingerprint-matched p2_gen block WITHOUT cap_hit_final is a foreign/stale
    manifest shape — fail loud rather than silently treating it as under-bar."""
    ds = "evil_normal"
    _write_manifest(tmp_path, ds, cap_hit_final=0.0)
    m_path = lib.capture_dir(tmp_path, ds) / "manifest.json"
    manifest = json.loads(m_path.read_text())
    del manifest["phases"]["p2_gen"]["cap_hit_final"]
    m_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="cap_hit_final"):
        _run_gen_resumed(tmp_path, ds)


def test_main_halt_file_gate(tmp_path: Path) -> None:
    """--phase capture/all refuse while cap_hit_halt.json exists; gen and the
    explicit override proceed; an absent halt file never blocks."""
    (tmp_path / "cap_hit_halt.json").write_text(
        json.dumps({"halt": "cap_hit_over_bar_after_regen", "datasets": ["evil_normal"]})
    )
    for phase in ("capture", "all"):
        with pytest.raises(SystemExit):
            cap._check_cap_hit_halt(tmp_path, phase, override=False)
    cap._check_cap_hit_halt(tmp_path, "gen", override=False)  # gen re-derives the halt
    cap._check_cap_hit_halt(tmp_path, "capture", override=True)  # deliberate escape
    cap._check_cap_hit_halt(tmp_path / "absent", "all", override=False)
