"""Fail-fast invariants for the #2202 Fable digest dispatch (fable-digest-rerun).

Pins two repairs:

1. The empty-reply-recorded-as-success incident: 7/10 Fable digest chunks
   returned an EMPTY reply that the pre-fix path cached as
   ``{"result": "", "error": false, "category": "ok"}`` and silently absorbed
   as zero mode proposals.
2. The content-refusal incident: 8/~40 chunks returned
   ``stop_reason == "refusal"`` with ZERO content blocks (a ~deterministic
   API-level content refusal) and burned 5 attempts x 2 orgs as
   ``invalid_response`` before hard-erroring the whole harvest. Post-fix a
   refusal is VALID-at-dispatch (cached, never retried), routes to a
   first-class ``refused`` collection, and ``phase_fable_read`` re-dispatches
   the refused chunk as single-row items — rows that individually refuse are
   dropped-and-reported (``refusal_exclusions.json``), never coerced, never
   silent.

The invariants under test:

1. ``fable_reply_ok`` (the ``response_valid`` predicate handed to
   ``dispatch_calls``) rejects blank NON-refusal replies AND the old plain-str
   record format (poisoned cache reads as a MISS, #1470 heal path), while a
   ``stop_reason == "refusal"`` record is valid-at-dispatch (not retried).
2. ``harvest_fable_results`` returns ``(ok, refused)``: refusal records route
   to ``refused``; blank / malformed / errored /
   ``stop_reason == "max_tokens"``-truncated results stay HARD errors.
3. ``parse_modes`` distinguishes a schema parse FAILURE (``None`` — hard error
   at the caller) from a schema-valid, genuinely empty modes list (``[]`` —
   warned, not halted).
4. ``phase_fable_read`` per-row fallback wiring: a refused chunk's rows are
   re-dispatched as single-row items through the same ``fable_dispatch`` path;
   row-level refusals produce exclusion entries and contribute NO modes; the
   ``[p3b] refusal exclusions:`` summary line (the fix-engaged signal) is
   always emitted; a refusing pilot still halts rc 25.
5. Hierarchical consolidation (crash-fix 2 — the single all-proposals
   consolidation call refused at aggregate size, 1,297 proposals; a
   100-proposal subset PASSes at exact production shape): deterministic
   contiguous batching with ALL stage-1 batches in ONE dispatch call; a
   refused batch half-splits ONCE and a still-refusing half is
   dropped-and-reported (``consolidation_dropped`` + the ``[p3b] consolidation
   exclusions:`` line, always emitted 0-count included); a NON-refusal
   unparseable stage-1 reply and a refused/unparseable stage-2 final merge
   stay hard rc-25 halts; a pool at or under ``FABLE_CONSOL_BATCH`` keeps the
   single-call fast path (prior behavior).

Fakes are signature-conformant only: ``dispatch_calls`` is autospec'd at the
API boundary (the ``fable_dispatch`` production-body test), and the phase
tests fake ``fable_dispatch`` with a def mirroring its real signature.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import ClassVar
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue2202_labels as LB  # noqa: E402

from explore_persona_space.llm.api_dispatch import DispatchResult  # noqa: E402

REFUSAL_REC = {"text": "", "stop_reason": "refusal"}


def ok_rec(text: str) -> dict:
    return {"text": text, "stop_reason": "end_turn"}


def modes_reply(name: str) -> str:
    return json.dumps({"modes": [{"name": name, "description": "d", "decision_rule": "r"}]})


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

    def test_refusal_record_is_valid_at_dispatch(self):
        # stop_reason=="refusal" with zero content blocks (blank text) is a
        # ~deterministic content outcome: cacheable, NOT retried as
        # invalid_response (the 5-attempts-x-2-orgs waste this fix removes).
        assert LB.fable_reply_ok(dict(REFUSAL_REC)) is True
        # Conservative: a refusal stop_reason routes as refusal even if text
        # were somehow non-empty.
        assert LB.fable_reply_ok({"text": "x", "stop_reason": "refusal"}) is True


class TestHarvestFableResults:
    ITEMS: ClassVar[list[tuple[str, str]]] = [("c00", "prompt")]

    def _harvest(self, res: DispatchResult) -> tuple[dict, dict]:
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
        assert self._harvest(DispatchResult("c00", result=rec)) == ({"c00": rec}, {})

    def test_refusal_routes_to_refused_not_error(self):
        # Brief test (a): a refusal record is a first-class outcome, not an
        # error — routed to the returned ``refused`` collection.
        rec = dict(REFUSAL_REC)
        assert self._harvest(DispatchResult("c00", result=rec)) == ({}, {"c00": rec})

    def test_mixed_ok_and_refused_split(self):
        items = [("c00", "p0"), ("c01", "p1")]
        good = ok_rec('{"modes": []}')
        results = {
            "c00": DispatchResult("c00", result=good),
            "c01": DispatchResult("c01", result=dict(REFUSAL_REC)),
        }
        ok, refused = LB.harvest_fable_results(items, results, LB.FABLE_MAX_TOKENS)
        assert ok == {"c00": good}
        assert refused == {"c01": REFUSAL_REC}


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


class TestFableDispatchBody:
    """Executes the REAL ``fable_dispatch`` body (build_request closure,
    asyncio bridge, harvest split), faking ONLY the network boundary
    (``dispatch_calls``, autospec'd — signature-conformant by construction)."""

    def test_refusal_split_and_request_contract(self, monkeypatch):
        import explore_persona_space.llm.api_dispatch as AD

        good = ok_rec('{"modes": []}')
        fake = mock.create_autospec(AD.dispatch_calls)
        fake.return_value = {
            "c00": AD.DispatchResult("c00", result=good),
            "c01": AD.DispatchResult("c01", result=dict(REFUSAL_REC)),
        }
        monkeypatch.setattr(AD, "dispatch_calls", fake)
        ok, refused = LB.fable_dispatch([("c00", "p0"), ("c01", "p1")], args=None)
        assert ok == {"c00": good}
        assert refused == {"c01": REFUSAL_REC}
        kwargs = fake.call_args.kwargs
        # The refusal contract lives in the threaded validator: a refusal
        # record must be valid-at-dispatch (cached, never retried).
        assert kwargs["response_valid"] is LB.fable_reply_ok
        req = kwargs["build_request"](AD.DispatchItem(item_id="c00", payload="p0"))
        assert req["model"] == LB.FABLE_MODEL
        assert req["max_tokens"] == LB.FABLE_MAX_TOKENS
        assert req["timeout"] == LB.FABLE_REQUEST_TIMEOUT_S
        assert req["messages"] == [{"role": "user", "content": "p0"}]


# ── phase-level fallback wiring (brief tests (c) + (d)) ──────────────────────────


def _setup_digests(tmp_path: Path, monkeypatch) -> tuple[list[dict], list[dict]]:
    """Tiny digest fixture under a scratch PROJECT_ROOT (2 rows/chunk):
    digest_result1 -> chunks c00 (rows 0-1), c01 (rows 2-3);
    digest_result2 -> chunk c00 (rows 0-1, padded so IT is the pilot)."""
    monkeypatch.setattr(LB, "PROJECT_ROOT", tmp_path)
    ddir = tmp_path / "data" / "issue_2202" / "digests"
    ddir.mkdir(parents=True)
    d1 = [{"ci": 110 + j, "pad": "x"} for j in range(4)]
    d2 = [{"ci": 220 + j, "pad": "y" * 800} for j in range(2)]
    for stem, rows in (("digest_result1", d1), ("digest_result2", d2)):
        shard = f"{stem}.shard00.jsonl"
        (ddir / shard).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (ddir / f"{stem}.manifest.json").write_text(json.dumps({"shards": [shard]}))
    return d1, d2


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(smoke=False, out_eval=str(tmp_path / "eval"), fable_chunk_rows=2)


def _scripted_dispatch(script: list):
    """Signature-mirroring fake for ``fable_dispatch`` (the API boundary of the
    phase): call k routes to ``script[k]``, which maps items -> (ok, refused)."""
    calls: list[list[tuple[str, str]]] = []

    def fake_fable_dispatch(items, args, max_tokens=LB.FABLE_MAX_TOKENS):
        calls.append(list(items))
        return script[len(calls) - 1](items)

    return fake_fable_dispatch, calls


def _all_ok(reply: str):
    def step(items):
        return {i: ok_rec(reply) for i, _ in items}, {}

    return step


class TestPhaseFableReadRefusalFallback:
    REFUSED_CHUNK = "digest_result1_c01"

    def _run_phase(self, tmp_path, monkeypatch):
        d1, _d2 = _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)

        def step_full(items):
            ok, ref = {}, {}
            for i, _ in items:
                if i == self.REFUSED_CHUNK:
                    ref[i] = dict(REFUSAL_REC)
                else:
                    ok[i] = ok_rec(modes_reply(f"mode_{i}"))
            return ok, ref

        def step_rows(items):
            ok, ref = {}, {}
            for i, _ in items:
                if i.endswith("_r03"):
                    ref[i] = dict(REFUSAL_REC)  # row-level refusal -> dropped
                else:
                    ok[i] = ok_rec(modes_reply("recovered_mode"))
            return ok, ref

        consol = json.dumps(
            {"modes": [{"name": "final_mode", "description": "d", "decision_rule": "r"}]}
        )
        fake, calls = _scripted_dispatch(
            [_all_ok(modes_reply("pilot_mode")), step_full, step_rows, _all_ok(consol)]
        )
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        LB.phase_fable_read(args)
        return d1, args, calls

    def test_chunk_fallback_rewires_refused_chunk_as_single_row_items(self, tmp_path, monkeypatch):
        # Brief test (d): the refused chunk's rows are re-dispatched as
        # single-row items (same task text, CASES block of exactly one row)
        # through the same fable_dispatch path.
        d1, _, calls = self._run_phase(tmp_path, monkeypatch)
        assert [i for i, _ in calls[2]] == [
            "digest_result1_c01_r02",
            "digest_result1_c01_r03",
        ]
        prompt_r02 = dict(calls[2])["digest_result1_c01_r02"]
        assert prompt_r02.startswith(LB.FABLE_TASK_R1 + "\n\nCASES:\n\n")
        assert json.dumps(d1[2], ensure_ascii=False) in prompt_r02  # exactly this row
        assert json.dumps(d1[3], ensure_ascii=False) not in prompt_r02
        assert "\n\n---\n\n" not in prompt_r02  # ONE row, no chunk separator

    def test_row_refusal_excluded_and_contributes_no_modes(self, tmp_path, monkeypatch, caplog):
        # Brief test (c): a row-level refusal yields an exclusion entry and NO
        # mode contribution; the recovered sibling row contributes as normal.
        caplog.set_level(logging.INFO, logger="issue2202_labels")
        _d1, args, calls = self._run_phase(tmp_path, monkeypatch)
        out_fable = Path(args.out_eval) / "fable_reads"

        excl = json.loads((out_fable / "refusal_exclusions.json").read_text())
        assert excl["n_rows_dropped"] == 1
        assert excl["n_chunks_fallback"] == 1
        assert excl["fallback_chunks"] == [self.REFUSED_CHUNK]
        by_stage = {e["stage"]: e for e in excl["entries"]}
        assert by_stage["row-refused"] == {
            "chunk": self.REFUSED_CHUNK,
            "row_index": 3,
            "ci": 113,
            "stage": "row-refused",
        }
        assert by_stage["chunk-fallback"] == {
            "chunk": self.REFUSED_CHUNK,
            "row_index": 2,
            "ci": 112,
            "stage": "chunk-fallback",
        }

        # No mode contribution from the dropped row; recovered row contributes.
        consol_prompt = calls[3][0][1]
        assert "recovered_mode" in consol_prompt
        assert "digest_result1_c01_r02" in consol_prompt  # source_chunk of recovered
        assert "digest_result1_c01_r03" not in consol_prompt  # dropped row absent

        # modes.json carries the exclusion count (analyzer coverage caveat).
        modes_doc = json.loads((out_fable / "modes.json").read_text())
        assert modes_doc["refusal_exclusions"] == {"n_rows_dropped": 1, "n_chunks_fallback": 1}
        assert [m["name"] for m in modes_doc["modes"]] == ["final_mode"]

        # Raw refusal records persist verbatim (audit).
        chunk_raw = json.loads((out_fable / f"{self.REFUSED_CHUNK}.json").read_text())
        assert chunk_raw["stop_reason"] == "refusal" and chunk_raw["raw"] == ""
        row_raw = json.loads((out_fable / "digest_result1_c01_r03.json").read_text())
        assert row_raw["stop_reason"] == "refusal"

        # Fix-engaged signal: the summary line.
        assert "refusal exclusions: 1 rows dropped (1 chunks fell back per-row)" in caplog.text

    def test_clean_run_still_emits_summary_line_and_exclusions_file(
        self, tmp_path, monkeypatch, caplog
    ):
        # The fix-engaged signal is reachable on EVERY run (0-count form).
        caplog.set_level(logging.INFO, logger="issue2202_labels")
        _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)
        consol = json.dumps(
            {"modes": [{"name": "final_mode", "description": "d", "decision_rule": "r"}]}
        )
        fake, calls = _scripted_dispatch(
            [_all_ok(modes_reply("pilot_mode")), _all_ok(modes_reply("m")), _all_ok(consol)]
        )
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        LB.phase_fable_read(args)
        assert len(calls) == 3  # pilot, full, consolidation — no row fallback
        excl = json.loads(
            (Path(args.out_eval) / "fable_reads" / "refusal_exclusions.json").read_text()
        )
        assert excl["n_rows_dropped"] == 0 and excl["entries"] == []
        assert "refusal exclusions: 0 rows dropped (0 chunks fell back per-row)" in caplog.text

    def test_refusing_pilot_still_halts_rc25(self, tmp_path, monkeypatch):
        # Brief constraint 4: the pilot gate is unchanged — a refusing pilot
        # halts with the designed rc 25.
        _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)

        def step_refuse_all(items):
            return {}, {i: dict(REFUSAL_REC) for i, _ in items}

        fake, _calls = _scripted_dispatch([step_refuse_all])
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        out_fable = Path(args.out_eval) / "fable_reads"
        out_fable.mkdir(parents=True)
        items = LB.build_fable_items(args)
        with pytest.raises(SystemExit) as ei:
            LB.fable_pilot_gate(items, args, out_fable)
        assert ei.value.code == LB.RC_FABLE
        probe = json.loads((out_fable / "probe.json").read_text())
        assert probe["ok"] is False and "refus" in probe["error"]


# ── hierarchical consolidation (crash-fix 2, brief tests (a)-(e)) ────────────────


def modes_reply_many(prefix: str, n: int) -> str:
    return json.dumps(
        {
            "modes": [
                {"name": f"{prefix}_{k}", "description": "d", "decision_rule": "r"}
                for k in range(n)
            ]
        }
    )


def _chunks_two_modes(items):
    """Chunk step: every digest chunk replies with TWO modes (``m_<chunk>_{0,1}``)
    so the hierarchical tests get 6 proposals from the 3-chunk fixture."""
    return {i: ok_rec(modes_reply_many(f"m_{i}", 2)) for i, _ in items}, {}


# NOT a class attribute: a plain function stored on the class would bind as a
# method under ``self.`` access and eat ``items`` as ``self``.
_PILOT_OK = _all_ok(modes_reply("pilot_mode"))


class TestHierarchicalConsolidation:
    def test_batching_deterministic_and_single_dispatch(self, tmp_path, monkeypatch):
        # Brief test (a): deterministic contiguous batches; ALL stage-1
        # batches ride ONE fable_dispatch call; stage 2 is ONE final merge
        # over the concatenated stage-1 modes, same instruction text.
        monkeypatch.setattr(LB, "FABLE_CONSOL_BATCH", 2)
        props = [{"name": f"p{k}"} for k in range(5)]
        assert LB.consolidation_batches(props) == [props[0:2], props[2:4], props[4:5]]

        _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)
        fake, calls = _scripted_dispatch(
            [_PILOT_OK, _chunks_two_modes, _all_ok(modes_reply("s1")), _all_ok(modes_reply("f"))]
        )
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        LB.phase_fable_read(args)
        assert [i for i, _ in calls[2]] == [
            "consolidation_b00",
            "consolidation_b01",
            "consolidation_b02",
        ]
        prompts = dict(calls[2])
        # each batch prompt = the SAME instruction over exactly its contiguous slice
        assert prompts["consolidation_b00"].startswith(LB.FABLE_CONSOL_INSTRUCTION)
        assert "m_digest_result1_c00_0" in prompts["consolidation_b00"]
        assert "m_digest_result1_c00_1" in prompts["consolidation_b00"]
        assert "m_digest_result1_c01_0" not in prompts["consolidation_b00"]
        assert "m_digest_result2_c00_1" in prompts["consolidation_b02"]
        # stage 2: one final merge call over the stage-1 output modes
        assert [i for i, _ in calls[3]] == ["consolidation"]
        assert calls[3][0][1].startswith(LB.FABLE_CONSOL_INSTRUCTION)
        assert '"s1"' in calls[3][0][1]
        assert "m_digest_result1_c00_0" not in calls[3][0][1]  # raw proposals replaced

    def test_refused_batch_half_split_drop_and_report(self, tmp_path, monkeypatch, caplog):
        # Brief test (b): a refused stage-1 batch half-splits ONCE; the
        # refusing half's proposals are dropped-and-reported, the passing
        # half's modes contribute to the stage-2 merge.
        caplog.set_level(logging.INFO, logger="issue2202_labels")
        monkeypatch.setattr(LB, "FABLE_CONSOL_BATCH", 2)
        _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)

        def step_stage1(items):
            ok, ref = {}, {}
            for i, _ in items:
                if i == "consolidation_b01":
                    ref[i] = dict(REFUSAL_REC)
                else:
                    ok[i] = ok_rec(modes_reply(f"s1_{i[-3:]}"))
            return ok, ref

        def step_halves(items):
            ok, ref = {}, {}
            for i, _ in items:
                if i == "consolidation_b01_h1":
                    ref[i] = dict(REFUSAL_REC)
                else:
                    ok[i] = ok_rec(modes_reply("s1_recovered"))
            return ok, ref

        fake, calls = _scripted_dispatch(
            [_PILOT_OK, _chunks_two_modes, step_stage1, step_halves, _all_ok(modes_reply("f"))]
        )
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        LB.phase_fable_read(args)

        # both halves of the refused batch ride ONE re-dispatch call; the
        # deterministic split gives the first half the odd proposal.
        assert [i for i, _ in calls[3]] == ["consolidation_b01_h0", "consolidation_b01_h1"]
        h0_prompt = dict(calls[3])["consolidation_b01_h0"]
        assert h0_prompt.startswith(LB.FABLE_CONSOL_INSTRUCTION)
        assert "m_digest_result1_c01_0" in h0_prompt
        assert "m_digest_result1_c01_1" not in h0_prompt

        out_fable = Path(args.out_eval) / "fable_reads"
        excl = json.loads((out_fable / "refusal_exclusions.json").read_text())
        assert excl["consolidation_dropped"] == [
            {"batch": "consolidation_b01", "half": 1, "n_proposals": 1}
        ]
        final_prompt = calls[4][0][1]
        assert "s1_recovered" in final_prompt  # passing half contributes
        assert "s1_b00" in final_prompt and "s1_b02" in final_prompt
        modes_doc = json.loads((out_fable / "modes.json").read_text())
        assert modes_doc["consolidation"] == {
            "n_proposals": 6,
            "n_batches": 3,
            "batch_size": 2,
            "n_proposals_dropped": 1,
            "stages": 2,
        }
        # fix-engaged signals: the always-emitted exclusions line + stage lines
        assert "consolidation exclusions: 1 proposals dropped (1 batch-halves refused)" in (
            caplog.text
        )
        assert "consolidation stage 1: 6 proposals -> 3 batches (batch_size=2" in caplog.text
        assert "consolidation stage 2: final merge over 3 stage-1 modes" in caplog.text
        # refused batch + refusing half persist verbatim (audit)
        for rid in ("consolidation_b01", "consolidation_b01_h1"):
            rec = json.loads((out_fable / f"{rid}.json").read_text())
            assert rec["stop_reason"] == "refusal" and rec["raw"] == ""

    def test_unparseable_stage1_batch_halts_rc25(self, tmp_path, monkeypatch):
        # Brief test (c): a NON-refusal stage-1 reply that fails the schema is
        # a bug, not content — hard rc-25 halt, never absorbed as a drop.
        monkeypatch.setattr(LB, "FABLE_CONSOL_BATCH", 2)
        _setup_digests(tmp_path, monkeypatch)
        fake, _calls = _scripted_dispatch([_PILOT_OK, _chunks_two_modes, _all_ok("no json here")])
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        with pytest.raises(SystemExit) as ei:
            LB.phase_fable_read(_args(tmp_path))
        assert ei.value.code == LB.RC_FABLE

    def test_stage2_refusal_keeps_designed_rc25_halt(self, tmp_path, monkeypatch):
        # Brief test (d): a refused stage-2 final merge keeps the existing
        # designed halt (rc 25, modes.json note) — never coerced.
        monkeypatch.setattr(LB, "FABLE_CONSOL_BATCH", 2)
        _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)

        def step_final_refuse(items):
            return {}, {i: dict(REFUSAL_REC) for i, _ in items}

        fake, _calls = _scripted_dispatch(
            [_PILOT_OK, _chunks_two_modes, _all_ok(modes_reply("s1")), step_final_refuse]
        )
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        with pytest.raises(SystemExit) as ei:
            LB.phase_fable_read(args)
        assert ei.value.code == LB.RC_FABLE
        out_fable = Path(args.out_eval) / "fable_reads"
        modes_doc = json.loads((out_fable / "modes.json").read_text())
        assert modes_doc["modes"] == [] and "unparseable/empty" in modes_doc["note"]
        assert modes_doc["consolidation"]["stages"] == 2
        # the row-exclusion + consolidation-drop record survived the halt
        excl = json.loads((out_fable / "refusal_exclusions.json").read_text())
        assert excl["consolidation_dropped"] == []

    def test_single_batch_fast_path_skips_stage1(self, tmp_path, monkeypatch, caplog):
        # Brief test (e): len(proposals) <= FABLE_CONSOL_BATCH -> no stage 1,
        # one final call over the raw proposals (prior behavior); the
        # consolidation-exclusions line still emits its 0-count form.
        caplog.set_level(logging.INFO, logger="issue2202_labels")
        _setup_digests(tmp_path, monkeypatch)  # default batch=100 >> 6 proposals
        args = _args(tmp_path)
        fake, calls = _scripted_dispatch([_PILOT_OK, _chunks_two_modes, _all_ok(modes_reply("f"))])
        monkeypatch.setattr(LB, "fable_dispatch", fake)
        LB.phase_fable_read(args)
        assert len(calls) == 3  # pilot, chunks, ONE final call — no stage 1
        assert [i for i, _ in calls[2]] == ["consolidation"]
        assert "m_digest_result1_c00_0" in calls[2][0][1]  # proposals ride directly
        modes_doc = json.loads((Path(args.out_eval) / "fable_reads" / "modes.json").read_text())
        assert modes_doc["consolidation"] == {
            "n_proposals": 6,
            "n_batches": 1,
            "batch_size": LB.FABLE_CONSOL_BATCH,
            "n_proposals_dropped": 0,
            "stages": 1,
        }
        assert "consolidation exclusions: 0 proposals dropped (0 batch-halves refused)" in (
            caplog.text
        )


class TestBuildFableRowItems:
    def test_row_ids_prompts_and_info(self, tmp_path, monkeypatch):
        d1, _d2 = _setup_digests(tmp_path, monkeypatch)
        args = _args(tmp_path)
        items, info = LB.build_fable_row_items(args, ["digest_result1_c00"])
        assert [i for i, _ in items] == ["digest_result1_c00_r00", "digest_result1_c00_r01"]
        assert info["digest_result1_c00_r00"] == {
            "chunk": "digest_result1_c00",
            "row_index": 0,
            "ci": 110,
        }
        for (_rid, prompt), row in zip(items, d1[:2], strict=True):
            assert prompt == (
                f"{LB.FABLE_TASK_R1}\n\nCASES:\n\n{json.dumps(row, ensure_ascii=False)}"
            )

    def test_foreign_chunk_id_fails_loud(self, tmp_path, monkeypatch):
        _setup_digests(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="not a digest chunk id"):
            LB.build_fable_row_items(_args(tmp_path), ["consolidation"])
