"""Round-2 review blocker 1 (C7 parser / Phase-A payload shape) — issue #833.

Pins the fit driver's tolerant raw-completions reader ``_texts_from_json``
against the shapes the extractor actually writes:

  (i)   the rbase payload shape — a top-level dict with a ``"targets"`` map
        ``{tcid: [text ordered by probe_idx]}`` (highest precedence);
  (ii)  a top-level payload dict carrying a ``"responses"`` record LIST (the
        Phase-A generation payload shape) — must route into the record-list
        branch BEFORE arbitrary-key iteration (the round-1 verified crash);
  (iii) garbage fails loud (ValueError naming the file).

Run: uv run pytest tests/test_issue833_text_shapes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from issue833_fit_onpolicy import _texts_from_json  # noqa: E402


def test_rbase_targets_map_shape():
    """(i) rbase payload: the ``targets`` map wins and yields probe_idx-ordered texts."""
    payload = {
        "behavior": "em",
        "source_cid": "src0",
        "seed": 42,
        "targets": {"t": ["x", "y"]},
        "responses": [
            {"target_cid": "t", "probe_idx": 0, "probe": "q0", "response": "x"},
            {"target_cid": "t", "probe_idx": 1, "probe": "q1", "response": "y"},
        ],
    }
    assert _texts_from_json(payload, "rbase/em/src0_seed42.json") == {
        ("t", 0): "x",
        ("t", 1): "y",
    }


def test_toplevel_responses_list_shape():
    """(ii) Phase-A payload (no targets map): responses list routes to the record branch."""
    payload = {
        "behavior": "em",
        "source_cid": "src0",
        "seed": 42,
        "n_targets": 1,
        "n_probes": 1,
        "responses": [{"target_cid": "t", "probe_idx": 0, "probe": "q0", "response": "x"}],
    }
    assert _texts_from_json(payload, "generation/em/src0_seed42.json") == {("t", 0): "x"}


def test_plain_tcid_to_texts_dict_shape():
    """Bare ``{tcid: [text, ...]}`` still parses (tolerant-reader regression guard)."""
    assert _texts_from_json({"t": ["x"]}, "f.json") == {("t", 0): "x"}


@pytest.mark.parametrize(
    "garbage",
    [
        42,  # not a dict/list at all
        "just a string",
        {"t": 42},  # dict value neither list[str] nor {"responses": [...]}
        {"t": [1, 2]},  # list values not strings
        [{"no_target": True}],  # record missing target_cid/response
        ["not-a-record"],  # list element not a dict
    ],
)
def test_garbage_fails_loud(garbage):
    """(iii) unrecognized shapes raise ValueError naming the file (fail loud)."""
    with pytest.raises(ValueError, match=r"f\.json"):
        _texts_from_json(garbage, "f.json")
