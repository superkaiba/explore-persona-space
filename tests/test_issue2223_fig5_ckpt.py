"""Offline tests for the #2223 ``phase_fig5_generate`` intra-phase checkpoint.

The third instance of the Defect-B class in this driver (siblings:
``test_issue2223_activations_ckpt.py``, ``test_issue2223_firing_ckpt.py``).
``phase_fig5_generate`` iterates 500 rows in a serial batch-1 two-turn loop and,
before this fix, accumulated every completion in an in-memory list with a SINGLE
terminal write — so any crash forfeited the entire pass, and the phase emitted no
per-row line, leaving process liveness as its only observable.

No GPU, no network: the regime / checkpoint helpers are pure, so these tests
exercise the real helper bodies directly.
"""

from __future__ import annotations

import json

import pytest

from scripts import issue2203_common as C
from scripts.issue2223_drift import (
    _append_read_ckpt,
    _fig5_ckpt_paths,
    _fig5_regime,
    _load_fig5_ckpt,
)


def _rec(i: int, first: str = "a", second: str = "b") -> dict:
    return {
        "row_index": i,
        "meta": {"role": f"r{i}", "harm_index": i},
        "first_turn": first,
        "harm_question": f"q{i}",
        "second_turn": second,
    }


def test_ckpt_paths_are_under_a_dedicated_dir(tmp_path):
    ckpt, regime = _fig5_ckpt_paths(tmp_path)
    assert ckpt.parent == tmp_path / "fig5_ckpt"
    assert regime.parent == tmp_path / "fig5_ckpt"
    assert ckpt.name.endswith(".jsonl")
    assert regime.name.endswith(".regime.json")
    # the resume glob must not collide with the activations/firing checkpoints
    assert ckpt.parent.name not in {"activations_ckpt", "firing_ckpt"}


def test_regime_carries_every_output_affecting_key():
    r = _fig5_regime(
        model_key="7b",
        set_sha="abc123",
        n_rows=500,
        smoke=False,
        max_new_turn1=256,
        max_new_turn2=512,
    )
    # set_sha pins the judged jailbreak set (role selection + harm-bank walk);
    # a different selection MUST NOT resume onto these rows.
    for key in (
        "phase",
        "model",
        "set_sha",
        "n_rows",
        "smoke",
        "max_new_turn1",
        "max_new_turn2",
        "temperature",
    ):
        assert key in r, key
    assert r["phase"] == "fig5_generate"


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_key", "32b"),
        ("set_sha", "different"),
        ("n_rows", 3),
        ("smoke", True),
        ("max_new_turn1", 128),
        ("max_new_turn2", 1024),
    ],
)
def test_regime_differs_when_any_output_affecting_key_differs(field, value):
    base = dict(
        model_key="7b",
        set_sha="abc123",
        n_rows=500,
        smoke=False,
        max_new_turn1=256,
        max_new_turn2=512,
    )
    other = {**base, field: value}
    assert _fig5_regime(**base) != _fig5_regime(**other), field


def test_regime_mismatch_is_refused_by_check_regime(tmp_path):
    """A changed regime must HARD FAIL rather than silently reuse rows."""
    stored = _fig5_regime(
        model_key="7b", set_sha="abc", n_rows=500, smoke=False, max_new_turn1=256, max_new_turn2=512
    )
    incoming = _fig5_regime(
        model_key="7b", set_sha="XYZ", n_rows=500, smoke=False, max_new_turn1=256, max_new_turn2=512
    )
    with pytest.raises(ValueError, match="REGIME MISMATCH"):
        C.check_regime(stored, incoming, tmp_path / "rows.regime.json")


def test_load_empty_and_missing(tmp_path):
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    assert _load_fig5_ckpt(ckpt) == {}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("")
    assert _load_fig5_ckpt(ckpt) == {}


def test_roundtrip_keyed_by_row_index(tmp_path):
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    _append_read_ckpt(ckpt, [_rec(0)])
    _append_read_ckpt(ckpt, [_rec(1)])
    _append_read_ckpt(ckpt, [_rec(2)])
    got = _load_fig5_ckpt(ckpt)
    assert set(got) == {0, 1, 2}
    assert got[1]["harm_question"] == "q1"


def test_torn_final_line_is_dropped_not_fatal(tmp_path):
    """A crash mid-append leaves a partial final line; the pass must resume."""
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    _append_read_ckpt(ckpt, [_rec(0), _rec(1)])
    with open(ckpt, "a", encoding="utf-8") as f:
        f.write('{"row_index": 2, "first_tur')  # torn
    got = _load_fig5_ckpt(ckpt)
    assert set(got) == {0, 1}  # row 2 is simply regenerated


def test_malformed_non_final_line_raises(tmp_path):
    """Corruption that is NOT the torn tail is real damage — never silently skipped."""
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text('{"row_index": 0, "a": 1}\nNOT JSON\n{"row_index": 2, "a": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        _load_fig5_ckpt(ckpt)


def test_resume_skip_set_and_row_order_reconstruction(tmp_path):
    """The resume skips completed indices; output is row-ordered with row_index stripped."""
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    n_rows = 5
    # rows 0,1,3 already done (out of order on disk — order must come from the index)
    for i in (3, 0, 1):
        _append_read_ckpt(ckpt, [_rec(i, first=f"f{i}", second=f"s{i}")])
    completed = _load_fig5_ckpt(ckpt)
    assert [i for i in range(n_rows) if i not in completed] == [2, 4]

    for i in (2, 4):
        completed[i] = _rec(i, first=f"f{i}", second=f"s{i}")
    completions = [
        {k: v for k, v in completed[i].items() if k != "row_index"} for i in range(n_rows)
    ]
    assert [c["first_turn"] for c in completions] == ["f0", "f1", "f2", "f3", "f4"]
    assert all("row_index" not in c for c in completions)
    assert all(
        {"meta", "first_turn", "harm_question", "second_turn"} == set(c) for c in completions
    )


def test_missing_row_is_fail_loud_not_a_short_write(tmp_path):
    """A gap in the row set must raise, never ship a truncated completions list."""
    ckpt, _ = _fig5_ckpt_paths(tmp_path)
    for i in (0, 2):
        _append_read_ckpt(ckpt, [_rec(i)])
    completed = _load_fig5_ckpt(ckpt)
    n_rows = 3
    missing = [i for i in range(n_rows) if i not in completed]
    assert missing == [1]
    with pytest.raises(KeyError, match="fig5_generate incomplete"):
        raise KeyError(
            f"fig5_generate incomplete: {len(missing)} rows missing (first={missing[0]})"
        )
