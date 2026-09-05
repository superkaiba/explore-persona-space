"""Focused tests for the issue #2569 direct direction-example pass."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2569_direction_examples as DEX  # noqa: E402


def test_npz_member_memmap_reads_stored_arrays(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    expected_x = np.arange(24, dtype=np.float32).reshape(6, 4)
    expected_ci = np.asarray([7, 3, 9, 2, 8, 1], dtype=np.int64)
    np.savez(path, x=expected_x, ci=expected_ci)

    got_x = DEX.npz_member_memmap(path, "x")
    got_ci = DEX.npz_member_memmap(path, "ci.npy")

    assert isinstance(got_x, np.memmap)
    assert got_x.shape == expected_x.shape
    assert np.array_equal(got_x, expected_x)
    assert np.array_equal(got_ci, expected_ci)


def test_ordered_extrema_break_ties_by_row() -> None:
    scores = np.asarray([1.0, 3.0, 3.0, -2.0, -2.0, 0.0])
    assert DEX.ordered_extreme_indices(scores, 3, largest=True).tolist() == [1, 2, 0]
    assert DEX.ordered_extreme_indices(scores, 3, largest=False).tolist() == [3, 4, 5]


def test_line_candidates_are_centered_and_signed() -> None:
    states = np.asarray([[4.0, 0.0], [1.0, 8.0], [-2.0, 0.0]], dtype=np.float32)
    rows = DEX.line_candidates(
        states,
        mean=np.asarray([1.0, 2.0]),
        directions=np.asarray([[1.0, 0.0]]),
        names=["axis"],
        candidate_k=3,
        block=2,
    )
    row = rows[0]
    assert [item["sample_row"] for item in row["high_candidates"]] == [0, 1, 2]
    assert [item["score"] for item in row["high_candidates"]] == [3.0, 0.0, -3.0]
    assert [item["sample_row"] for item in row["low_candidates"]] == [2, 1, 0]


def test_plane_candidates_rank_by_rotation_invariant_norm() -> None:
    states = np.asarray([[3.0, 4.0, 99.0], [0.0, 6.0, 0.0], [1.0, 1.0, 0.0]])
    bases = np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    rows = DEX.plane_candidates(
        states,
        mean=np.zeros(3),
        bases=bases,
        basis_dims=np.asarray([2]),
        names=["plane"],
        candidate_k=3,
        block=2,
    )
    assert [item["sample_row"] for item in rows[0]["high_candidates"]] == [1, 0, 2]
    assert np.allclose(rows[0]["high_candidates"][1]["plane_coordinates"], [3.0, 4.0])


def test_unique_candidates_deduplicate_ranked_side() -> None:
    ci = np.asarray([10, 11, 12, 13], dtype=np.int64)
    texts = {
        10: {"corpus": "a", "context_text": "Same prompt", "answer_text": "A"},
        11: {"corpus": "b", "context_text": " same   PROMPT ", "answer_text": "B"},
        12: {"corpus": "a", "context_text": "Different", "answer_text": "C"},
        13: {"corpus": "b", "context_text": "Third", "answer_text": "D"},
    }
    candidates = [
        {"sample_row": index, "score": float(4 - index), "score_in_sample_scale": 1.0}
        for index in range(4)
    ]
    unique = DEX.unique_candidates(
        candidates, side="context", top_k=3, ci=ci, texts=texts
    )
    assert [row["ci"] for row in unique] == [10, 12, 13]


def test_quote_text_redacts_and_selects_requested_end() -> None:
    secret_value = "abcdefghijkl" + "mnop123456"
    text = "prefix " + "x" * 30 + " api_" + "key=" + secret_value + " suffix"
    quote = DEX.quote_text(text, 45, tail=True)
    assert quote.startswith("…")
    assert secret_value not in quote
    assert "[REDACTED]" in quote
