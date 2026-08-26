"""Network-free, GPU-free pins for the issue-2546 math-gold join (round 14).

Round-14 crash-fix regression (task #2546 arm-1 ``p4_capture`` crash,
``AssertionError: math src_index 2467: RLVR row lacks ground_truth`` —
epm:failure v5): exactly 2 of 7,358 staged math rows point at RLVR rows whose
``ground_truth`` is PRESENT but EMPTY (``''``; src_index 2467/13255, both
grain-ok MATH rows — the join is sound, the upstream rows are simply empty).
``stage_math_golds`` had no path for that condition, and the ``:2915``
fallback ``src.get("gold_answer")`` is inert (``gold_answer`` is None on ALL
staged math rows — which is why the RLVR join exists), so a bare skip would
hand the consumer a silent ``None`` gold.

The fix (this file's pins):

- ``stage_math_golds`` returns ``(golds, dropped)`` — the present-but-empty
  case is an explicit COUNTED DROP (row_id -> {src_index, reason}); the
  no-user-turn / join-grain / ground_truth-field-MISSING asserts stay hard.
- ``resolve_row_gold`` makes the consumer branch explicit: a recorded drop
  returns ``(None, True)`` (the caller sets ``correct = None`` directly,
  never routing a None gold through ``exact_match_correct``), and every math
  row MUST be joined-or-dropped — an uncovered math row fails loud.

Each stage_math_golds test executes the REAL body (verified failing against
the pre-fix module: the empty-gold fixture raised the production
AssertionError and the function returned a bare dict). ``load_dataset`` is
faked ONLY at the external network boundary, signature-conformant via
``unittest.mock.create_autospec(datasets.load_dataset)`` (code-style.md "one
production-body test per seam-stubbed function").
"""

from __future__ import annotations

import sys
import unittest.mock
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402


def _rlvr_row(question: str, gold) -> dict:
    row = {"messages": [{"role": "user", "content": f"{question} Show your work."}]}
    if gold is not None:
        row["ground_truth"] = gold
    return row


def _staged_row(row_id: str, src_index: int, question: str) -> dict:
    return {"row_id": row_id, "src_index": src_index, "question": question, "corpus": "math"}


def _patch_load_dataset(monkeypatch, fake_ds):
    """Autospec'd load_dataset returning ``fake_ds`` (a list indexes like a Dataset)."""
    import datasets

    fake = unittest.mock.create_autospec(datasets.load_dataset, return_value=fake_ds)
    monkeypatch.setattr(datasets, "load_dataset", fake)
    return fake


class TestStageMathGolds:
    def test_empty_upstream_gold_is_counted_drop(self, monkeypatch):
        """The r14 crash row shape: ground_truth == '' -> DROPPED + COUNTED,
        healthy siblings still join, no fabricated gold anywhere.

        FAILS PRE-FIX: the pre-r14 body raised
        ``AssertionError: math src_index 1: RLVR row lacks ground_truth``.
        """
        ds = [
            _rlvr_row("What is 6 x 7?", "42"),
            _rlvr_row("What is 3 + 4?", ""),  # the src_index-2467 shape
            _rlvr_row("What is 2^3?", "8"),
        ]
        rows = [
            _staged_row("math_00000", 0, "What is 6 x 7?"),
            _staged_row("math_00001", 1, "What is 3 + 4?"),
            _staged_row("math_00002", 2, "What is 2^3?"),
        ]
        fake = _patch_load_dataset(monkeypatch, ds)
        golds, dropped = G.stage_math_golds(rows)
        fake.assert_called_once()
        # Healthy rows join with the real upstream values.
        assert golds == {"math_00000": "42", "math_00002": "8"}
        # The empty-gold row is a counted drop with the recorded reason.
        assert len(dropped) == 1
        assert dropped["math_00001"] == {
            "src_index": 1,
            "reason": "upstream_ground_truth_empty",
        }
        # No fabricated/coerced gold: the dropped row appears in NO gold map.
        assert "math_00001" not in golds

    def test_whitespace_only_gold_is_also_a_drop(self, monkeypatch):
        """'present but empty' includes whitespace-only (the pre-fix assert
        used str(gold).strip() — the drop branch keeps that predicate)."""
        ds = [_rlvr_row("Q one?", "   ")]
        rows = [_staged_row("math_00000", 0, "Q one?")]
        _patch_load_dataset(monkeypatch, ds)
        golds, dropped = G.stage_math_golds(rows)
        assert golds == {}
        assert dropped["math_00000"]["reason"] == "upstream_ground_truth_empty"

    def test_healthy_rows_join_with_no_drops(self, monkeypatch):
        ds = [_rlvr_row("What is 6 x 7?", "42")]
        rows = [_staged_row("math_00000", 0, "What is 6 x 7?")]
        _patch_load_dataset(monkeypatch, ds)
        golds, dropped = G.stage_math_golds(rows)
        assert golds == {"math_00000": "42"}
        assert dropped == {}

    def test_grain_break_still_hard_asserts(self, monkeypatch):
        """A broken join must still refuse — the drop path is scoped to the
        empty-gold condition ONLY (grain_ok was True for both incident rows)."""
        ds = [_rlvr_row("A completely different question?", "42")]
        rows = [_staged_row("math_00000", 0, "What is 6 x 7?")]
        _patch_load_dataset(monkeypatch, ds)
        with pytest.raises(AssertionError, match="join grain broke"):
            G.stage_math_golds(rows)

    def test_missing_ground_truth_field_still_hard_asserts(self, monkeypatch):
        """ground_truth MISSING entirely (None via .get) is an UNEXPECTED
        condition — fail loud, never a drop (the counted drop is scoped to
        present-but-empty)."""
        ds = [_rlvr_row("What is 6 x 7?", None)]
        rows = [_staged_row("math_00000", 0, "What is 6 x 7?")]
        _patch_load_dataset(monkeypatch, ds)
        with pytest.raises(AssertionError, match="NO ground_truth field"):
            G.stage_math_golds(rows)

    def test_no_user_turn_still_hard_asserts(self, monkeypatch):
        ds = [{"messages": [{"role": "assistant", "content": "hi"}], "ground_truth": "42"}]
        rows = [_staged_row("math_00000", 0, "What is 6 x 7?")]
        _patch_load_dataset(monkeypatch, ds)
        with pytest.raises(AssertionError, match="no user turn"):
            G.stage_math_golds(rows)


class TestResolveRowGold:
    """The :2915 consumer branch — a dropped row yields an EXPLICIT
    (None, True); an uncovered math row fails loud instead of flowing a None
    gold into the correctness comparison."""

    DROPS: ClassVar[dict] = {
        "math_00001": {"src_index": 1, "reason": "upstream_ground_truth_empty"}
    }
    GOLDS: ClassVar[dict] = {"math_00000": "42"}

    def test_dropped_row_is_explicit_none(self):
        gold, gold_dropped = G.resolve_row_gold(
            "math", "math_00001", {"gold_answer": None}, self.GOLDS, self.DROPS
        )
        assert gold is None
        assert gold_dropped is True

    def test_healthy_math_row_resolves_joined_gold(self):
        gold, gold_dropped = G.resolve_row_gold(
            "math", "math_00000", {"gold_answer": None}, self.GOLDS, self.DROPS
        )
        assert gold == "42"
        assert gold_dropped is False

    def test_uncovered_math_row_fails_loud(self):
        """A math row in NEITHER map is a broken join — the silent-None flow
        the r14 brief bans is structurally unreachable."""
        with pytest.raises(AssertionError, match="no joined gold and no recorded drop"):
            G.resolve_row_gold("math", "math_99999", {"gold_answer": None}, self.GOLDS, self.DROPS)

    def test_non_math_row_passes_gold_answer_through(self):
        gold, gold_dropped = G.resolve_row_gold("mmlu", "mmlu_00000", {"gold_answer": "B"}, {}, {})
        assert gold == "B"
        assert gold_dropped is False

    def test_dropped_row_never_scores_incorrect(self):
        """End-to-end covariate semantics: the consumer sets correct=None for a
        dropped row (well-formed parse or not) — never False via a None gold."""
        gold, gold_dropped = G.resolve_row_gold(
            "math", "math_00001", {"gold_answer": None}, self.GOLDS, self.DROPS
        )
        well_formed = True
        correct = (
            G.exact_match_correct("math", "\\boxed{7}", gold)
            if well_formed and not gold_dropped
            else None
        )
        assert correct is None
