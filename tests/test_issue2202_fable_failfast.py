"""Fail-fast invariants for the #2202 Fable digest dispatch (fable-digest-rerun).

Pins the repair of the empty-reply-recorded-as-success incident: 7/10 Fable
digest chunks returned an EMPTY reply that the pre-fix path cached as
``{"result": "", "error": false, "category": "ok"}`` and silently absorbed as
zero mode proposals. The invariants under test are pure functions in
``scripts/issue2202_labels.py`` — no network, no artifact reads:

1. ``fable_reply_ok`` (the ``response_valid`` predicate handed to
   ``dispatch_calls``) rejects blank replies AND the old plain-str record
   format, so poisoned cache records read as a MISS (#1470 heal path) and a
   blank result can never be written back as a success.
2. ``harvest_fable_results`` hard-errors on blank / malformed / errored /
   ``stop_reason == "max_tokens"``-truncated results (rule-26 gating needs the
   stop_reason field the pre-fix ``parse_response=lambda t: t`` discarded).
3. ``parse_modes`` distinguishes a schema parse FAILURE (``None`` — hard error
   at the caller) from a schema-valid, genuinely empty modes list (``[]`` —
   warned, not halted).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue2202_labels as LB  # noqa: E402

from explore_persona_space.llm.api_dispatch import DispatchResult  # noqa: E402


class TestFableReplyOk:
    def test_blank_and_whitespace_rejected(self):
        assert LB.fable_reply_ok({"text": "", "stop_reason": None}) is False
        assert LB.fable_reply_ok({"text": "  \n\t ", "stop_reason": "end_turn"}) is False

    def test_old_plain_str_record_format_rejected(self):
        # The poisoned pre-fix cache stored the reply as a plain str
        # (parse_response=lambda t: t). Both the empty and NON-empty old
        # formats must read as invalid so _split_cached treats them as a MISS.
        assert LB.fable_reply_ok("") is False
        assert LB.fable_reply_ok("a non-empty stale reply") is False

    def test_new_format_non_blank_accepted(self):
        assert LB.fable_reply_ok({"text": '{"modes": []}', "stop_reason": "end_turn"}) is True


class TestHarvestFableResults:
    ITEMS: ClassVar[list[tuple[str, str]]] = [("c00", "prompt")]

    def _harvest(self, res: DispatchResult) -> dict:
        return LB.harvest_fable_results(self.ITEMS, {"c00": res}, LB.FABLE_MAX_TOKENS)

    def test_blank_reply_is_hard_error(self):
        with pytest.raises(RuntimeError, match="empty_or_malformed_reply"):
            self._harvest(DispatchResult("c00", result={"text": "", "stop_reason": "end_turn"}))

    def test_old_plain_str_result_is_hard_error(self):
        with pytest.raises(RuntimeError, match="empty_or_malformed_reply"):
            self._harvest(DispatchResult("c00", result="stale plain-str reply"))

    def test_dispatch_error_propagates_reason(self):
        with pytest.raises(RuntimeError, match="transport_exhausted"):
            self._harvest(DispatchResult("c00", error=True, reason="transport_exhausted"))

    def test_max_tokens_truncation_is_hard_error(self):
        with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
            self._harvest(
                DispatchResult(
                    "c00", result={"text": "truncated mid-wo", "stop_reason": "max_tokens"}
                )
            )

    def test_good_reply_passes_with_stop_reason(self):
        rec = {"text": '{"modes": []}', "stop_reason": "end_turn"}
        assert self._harvest(DispatchResult("c00", result=rec)) == {"c00": rec}


class TestParseModesContract:
    def test_unparseable_reply_returns_none(self):
        assert LB.parse_modes("") is None
        assert LB.parse_modes("I could not produce JSON here.") is None
        assert LB.parse_modes('{"not_modes": 1}') is None

    def test_schema_valid_empty_modes_returns_empty_list(self):
        assert LB.parse_modes('{"modes": []}') == []

    def test_valid_modes_parse(self):
        out = LB.parse_modes(
            '{"modes": [{"name": "Some Mode!", "description": "d", "decision_rule": "r"}]}'
        )
        assert out == [{"name": "some_mode", "description": "d", "decision_rule": "r"}]
