"""Regression pins for scripts/issue823_ladder_gen.py (#823 P-Gen fix rounds).

Pins the registered nested-assignment invariants (four corruptions that must
each raise, incl. the two caught ONLY by a single assert layer — the layering
is deliberately redundant and must not be weakened), the record builder's
fail-loud provenance asserts (incl. the FIX 3 ``batch_org`` line), the FIX 2
per-sub-batch ``harvested_at`` join, the FIX 4 fresh-redrive-dir numbering,
the FIX 1 canonical-repo upload gate, and the frozen-question loader's
monotone-unique context order.

Plan-v13 amendment pins (§4.3 steps 2/3b/4 + P0): the 4096/8192 cap ladder +
``gen_wave`` labels; the generation-config fingerprint's BOTH-direction
fixtures — a mutated/pilot-cap checkpoint HALTS before any dispatch (rc 4,
report JSON naming both fingerprints + differing fields) AND a same-config
different-process-time resume is byte-STABLE (the metadata-free hash basis);
per-row caps derived from checkpoint-PERSISTED request metadata, never live
constants; the v11 refusal>empty label precedence; and the P0
prompt-integrity gate (asserts (a)-(d)) with its dispatch-serializer
INDEPENDENCE pinned mechanically.

Offline by design: the banked HF download is monkeypatched to tmp_path
fixtures (signature-conformant def fake, no network); no eval_results/
fixture reads (no sparse_cones entry needed); no live API calls anywhere
(dispatch seams are monkeypatched recorders).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import time

import pytest

from explore_persona_space.llm.api_dispatch import DispatchItem, DispatchResult
from scripts import issue823_ladder_gen as lg

# ── Nested-assignment invariants (requirement 2) ─────────────────────────────


class TestAssignmentInvariants:
    def test_registered_assignment_verifies_at_full_grain(self):
        a = lg.build_assignment(lg.N_CONTEXTS_FULL)
        pairs = lg.verify_assignment(a, lg.N_CONTEXTS_FULL)
        assert len(pairs) == lg.REGISTERED_TOTAL_PAIRS

    def test_off_by_one_full_grain_raises(self):
        # persona(i, k) = (i+1) % k keeps balance AND nesting intact; at full
        # grain the distinct-pair count (14997 != 14996) or the exact-rule
        # recompute must catch it.
        n = lg.N_CONTEXTS_FULL
        a = {k: [(i + 1) % k for i in range(n)] for k in lg.K_ARMS}
        with pytest.raises(AssertionError):
            lg.verify_assignment(a, n)

    def test_wrong_divisor_raises(self):
        n = lg.N_CONTEXTS_FULL
        a = lg.build_assignment(n)
        a[16] = [i % 8 for i in range(n)]  # wrong divisor: personas 8..15 empty
        with pytest.raises(AssertionError):
            lg.verify_assignment(a, n)

    def test_balance_preserving_relabel_caught_by_nesting(self):
        # Swap personas 0<->1 in arm 2: per-arm balance and range are intact,
        # the pair count is unchanged — ONLY the nesting assert can catch it.
        n = lg.N_CONTEXTS_FULL
        a = lg.build_assignment(n)
        a[2] = [1 - (i % 2) for i in range(n)]
        with pytest.raises(AssertionError, match="not a subset"):
            lg.verify_assignment(a, n)

    def test_off_by_one_smoke_grain_caught_by_exact_recompute(self):
        # At n=16 the off-by-one is balance-degenerate: balance, nesting
        # (subset AND strict), and the pair count (f(0) == f(16) == 1) all
        # pass — ONLY the exact-rule recompute catches it. This pins the
        # recompute as load-bearing, not redundant, at smoke grain.
        n = 16
        a = {k: [(i + 1) % k for i in range(n)] for k in lg.K_ARMS}
        with pytest.raises(AssertionError, match="deviates from registered"):
            lg.verify_assignment(a, n)


# ── Record-builder provenance asserts (requirement 1 + FIX 2 / FIX 3) ───────


def _mk_result(item_id: str) -> DispatchResult:
    return DispatchResult(
        item_id=item_id,
        result="an in-character answer",
        error=False,
        category="ok",
        stop_reason="end_turn",
    )


def _mk_meta(
    harvested: str | None = "2026-08-19T00:00:00Z",
    org: str | None = "org-a",
    submitted: str | None = "2026-08-18T23:00:00Z",
    batch_id: str | None = "bA",
) -> dict:
    return {
        "batch_id": batch_id,
        "batch_request_custom_id": "cid-0",
        "batch_org": org,
        "batch_submitted_at": submitted,
        "harvested_at": harvested,
    }


def _mk_item(p: int, i: int) -> DispatchItem:
    """A dispatch item shaped exactly as build_items composes it."""
    return DispatchItem(
        item_id=lg.make_item_id(p, i),
        payload={
            "messages": [{"role": "user", "content": f"q{i}"}],
            "system": lg.persona_system(p),
        },
    )


class TestBuildRecordsProvenance:
    def _build(self, meta: dict):
        n = 1
        assignment = lg.build_assignment(n)
        pairs = lg.verify_assignment(assignment, n)  # {(0, 0)}
        iid = lg.make_item_id(0, 0)
        return lg.build_records(
            questions=["q0"],
            in_common=[True],
            pairs=pairs,
            assignment=assignment,
            results={iid: _mk_result(iid)},
            batch_meta={iid: meta},
            items_by_id={iid: _mk_item(0, 0)},
            max_tokens_by_item={iid: lg.GEN_MAX_TOKENS},
            gen_wave_by_item={iid: lg.GEN_WAVE_FIRST},
            regen_items=set(),
        )

    def test_complete_provenance_passes(self):
        rec = self._build(_mk_meta())[0][0]
        assert rec["harvested_at"] == "2026-08-19T00:00:00Z"
        assert rec["batch_org"] == "org-a"
        assert rec["batch_submitted_at"] == "2026-08-18T23:00:00Z"

    def test_record_carries_gen_wave_and_prompt_evidence(self):
        # Plan v13 step 2: every record carries its persisted cap, an explicit
        # gen_wave label, and the EXACT dispatched system prompt + sha256 (the
        # per-pair evidence the P0 gate byte-compares).
        rec = self._build(_mk_meta())[0][0]
        assert rec["max_tokens"] == lg.GEN_MAX_TOKENS
        assert rec["gen_wave"] == lg.GEN_WAVE_FIRST
        expected = lg.persona_system(0)
        assert rec["system_prompt"] == expected
        assert rec["system_prompt_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()

    def test_null_batch_org_raises(self):
        # FIX 3: batch_org joined the fail-loud null-assert block.
        with pytest.raises(AssertionError, match="batch_org"):
            self._build(_mk_meta(org=None))

    def test_null_harvested_at_raises(self):
        with pytest.raises(AssertionError, match="batch timestamps"):
            self._build(_mk_meta(harvested=None))

    def test_null_submitted_at_raises(self):
        with pytest.raises(AssertionError, match="batch timestamps"):
            self._build(_mk_meta(submitted=None))

    def test_null_batch_id_raises(self):
        with pytest.raises(AssertionError, match="batch_id"):
            self._build(_mk_meta(batch_id=None))

    def test_missing_harvested_at_key_raises(self):
        # FIX 2: the record reads meta["harvested_at"] — a meta row without it
        # (the pre-fix shape) fails loud, never a silent global fallback.
        meta = _mk_meta()
        del meta["harvested_at"]
        with pytest.raises(KeyError):
            self._build(meta)

    def test_harvested_at_is_per_record_not_global(self):
        # FIX 2 semantic: records carry their OWN sub-batch harvest time.
        n = 2
        assignment = lg.build_assignment(n)
        pairs = lg.verify_assignment(assignment, n)  # {(0,0), (1,0), (1,1)}
        metas = {
            lg.make_item_id(0, 0): _mk_meta(harvested="2026-08-18T02:00:00Z"),
            lg.make_item_id(0, 1): _mk_meta(harvested="2026-08-18T02:00:00Z"),
            lg.make_item_id(1, 1): _mk_meta(harvested="2026-08-18T09:30:00Z", batch_id="bB"),
        }
        by_p = lg.build_records(
            questions=["q0", "q1"],
            in_common=[True, True],
            pairs=pairs,
            assignment=assignment,
            results={iid: _mk_result(iid) for iid in metas},
            batch_meta=metas,
            items_by_id={lg.make_item_id(p, i): _mk_item(p, i) for i, p in pairs},
            max_tokens_by_item={iid: lg.GEN_MAX_TOKENS for iid in metas},
            gen_wave_by_item={iid: lg.GEN_WAVE_FIRST for iid in metas},
            regen_items=set(),
        )
        harvested = {
            (r["context_id"], r["persona_idx"]): r["harvested_at"]
            for recs in by_p.values()
            for r in recs
        }
        assert harvested[(0, 0)] == "2026-08-18T02:00:00Z"
        assert harvested[(1, 1)] == "2026-08-18T09:30:00Z"


# ── FIX 2: per-sub-batch harvest time joined from results_<batch_id>.json ───


class TestLoadBatchMeta:
    @staticmethod
    def _write_ckpt(tmp_path, sub_batches):
        ckpt = tmp_path / "batches"
        ckpt.mkdir()
        state = {
            "cid_to_item": {"c1": "p00_c00000", "c2": "p01_c00001"},
            "sub_batches": sub_batches,
        }
        (ckpt / "state.json").write_text(json.dumps(state))
        return ckpt

    def test_per_sub_batch_harvest_time_from_results_mtime(self, tmp_path):
        ckpt = self._write_ckpt(
            tmp_path,
            [
                {
                    "index": 0,
                    "batch_id": "bA",
                    "custom_ids": ["c1"],
                    "org": "org-a",
                    "submitted_at": "2026-08-18T00:00:00Z",
                    "status": "collected",
                },
                {
                    "index": 1,
                    "batch_id": "bB",
                    "custom_ids": ["c2"],
                    "org": "org-b",
                    "submitted_at": "2026-08-18T01:00:00Z",
                    "status": "collected",
                },
            ],
        )
        (ckpt / "results_bA.json").write_text("{}")
        (ckpt / "results_bB.json").write_text("{}")
        t_a = _dt.datetime(2026, 8, 18, 2, 0, 0, tzinfo=_dt.UTC).timestamp()
        t_b = _dt.datetime(2026, 8, 18, 9, 30, 0, tzinfo=_dt.UTC).timestamp()
        os.utime(ckpt / "results_bA.json", (t_a, t_a))
        os.utime(ckpt / "results_bB.json", (t_b, t_b))
        meta = lg.load_batch_meta(ckpt)
        assert meta["p00_c00000"]["harvested_at"] == "2026-08-18T02:00:00Z"
        assert meta["p01_c00001"]["harvested_at"] == "2026-08-18T09:30:00Z"
        assert meta["p00_c00000"]["batch_org"] == "org-a"
        assert meta["p01_c00001"]["batch_id"] == "bB"

    def test_missing_results_file_raises(self, tmp_path):
        ckpt = self._write_ckpt(
            tmp_path,
            [
                {
                    "index": 0,
                    "batch_id": "bA",
                    "custom_ids": ["c1"],
                    "org": "org-a",
                    "submitted_at": "2026-08-18T00:00:00Z",
                    "status": "submitted",
                }
            ],
        )
        with pytest.raises(AssertionError, match="harvest time unrecoverable"):
            lg.load_batch_meta(ckpt)


# ── FIX 4: fresh redrive rounds numbered past stale checkpoint dirs ─────────


class TestNextRedriveRound:
    def test_empty_root_starts_at_one(self, tmp_path):
        assert lg._next_redrive_round(tmp_path) == 1

    def test_numbers_past_stale_dirs(self, tmp_path):
        (tmp_path / "redrive1").mkdir()
        (tmp_path / "redrive2").mkdir()
        assert lg._next_redrive_round(tmp_path) == 3

    def test_ignores_non_matching_entries(self, tmp_path):
        (tmp_path / "redrive2").mkdir()
        (tmp_path / "redrive_foo").mkdir()  # non-numeric suffix
        (tmp_path / "redrive9.stale").mkdir()  # quarantined dir
        (tmp_path / "redrive5").write_text("")  # a FILE, not a dir
        assert lg._next_redrive_round(tmp_path) == 3


# ── FIX B (follow-up round): stale-redrive merge before the pending set ─────


def _ok_result(item_id: str) -> DispatchResult:
    return DispatchResult(item_id=item_id, result="text", error=False, category="ok")


def _transport_result(item_id: str) -> DispatchResult:
    return DispatchResult(
        item_id=item_id,
        result=None,
        error=True,
        reason="rate_limited",
        category=lg.RESULT_TRANSPORT,
    )


class TestStaleRedriveDirs:
    def test_ascending_numeric_dirs_only(self, tmp_path):
        (tmp_path / "redrive2").mkdir()
        (tmp_path / "redrive10").mkdir()
        (tmp_path / "redrive1").mkdir()
        (tmp_path / "redrive_foo").mkdir()  # non-numeric suffix
        (tmp_path / "redrive9.stale").mkdir()  # quarantined dir
        (tmp_path / "redrive5").write_text("")  # a FILE, not a dir
        assert [d.name for d in lg._stale_redrive_dirs(tmp_path)] == [
            "redrive1",
            "redrive2",
            "redrive10",
        ]


class TestMergeStaleRedrives:
    def test_prior_run_successes_merge_and_shrink_pending(self, tmp_path, monkeypatch):
        # Main-checkpoint residue: r1 + r2 transport-class; r0 succeeded.
        results = {
            "r0": _ok_result("r0"),
            "r1": _transport_result("r1"),
            "r2": _transport_result("r2"),
        }
        batch_meta = {"r0": {"batch_id": "b_main"}}
        items_by_id = {k: DispatchItem(item_id=k, payload={}) for k in results}
        # Stale redrive1 from a prior run holds r1 (billed + succeeded there).
        # The gen_config fixture mirrors production: _dispatch's fingerprint
        # gate persists it BEFORE the dispatcher ever writes state.json.
        rd = tmp_path / "redrive1"
        lg.check_or_persist_gen_config(rd, lg.GEN_MAX_TOKENS)
        (rd / "state.json").write_text(json.dumps({"cid_to_item": {"cid1": "r1"}}))
        calls = []

        async def fake_dispatch(items, checkpoint_dir, max_tokens, poll_interval):
            calls.append(([it.item_id for it in items], checkpoint_dir))
            assert max_tokens == lg.GEN_MAX_TOKENS
            return {it.item_id: _ok_result(it.item_id) for it in items}

        def fake_load_batch_meta(checkpoint_dir):
            return {"r1": {"batch_id": "b_stale1"}}

        monkeypatch.setattr(lg, "_dispatch", fake_dispatch)
        monkeypatch.setattr(lg, "load_batch_meta", fake_load_batch_meta)
        mt: dict[str, int] = {}
        lg._merge_stale_redrives(tmp_path, items_by_id, results, batch_meta, mt, poll_interval=1.0)
        assert calls == [(["r1"], rd)]
        # r1's already-paid success is merged back: only r2 remains pending.
        assert lg.transport_class_ids(results) == ["r2"]
        assert batch_meta["r1"]["batch_id"] == "b_stale1"
        assert batch_meta["r0"]["batch_id"] == "b_main"
        # Step 3b(iii): the re-served row's cap came from the stale
        # checkpoint's PERSISTED gen_config.json, not a live constant.
        assert mt == {"r1": lg.GEN_MAX_TOKENS}

    def test_stateless_dir_skipped_without_dispatch(self, tmp_path, monkeypatch):
        # Dir created but the dispatcher never wrote state.json: nothing was
        # submitted there, so there is nothing to resume (and no API call).
        (tmp_path / "redrive1").mkdir()

        def boom(*args, **kwargs):
            raise AssertionError("must not dispatch a stateless stale dir")

        monkeypatch.setattr(lg, "_dispatch", boom)
        results: dict = {}
        lg._merge_stale_redrives(tmp_path, {}, results, {}, {}, poll_interval=1.0)
        assert results == {}

    def test_foreign_checkpoint_ids_raise(self, tmp_path):
        rd = tmp_path / "redrive1"
        rd.mkdir()
        (rd / "state.json").write_text(json.dumps({"cid_to_item": {"c0": "not_registered"}}))
        with pytest.raises(RuntimeError, match="registered item set"):
            lg._merge_stale_redrives(tmp_path, {}, {}, {}, {}, poll_interval=1.0)


# ── Round-4 BLOCKER 1: per-(arm x persona) CELL cap-hit trigger ──────────────


class TestCellCapTrigger:
    def test_cell_fires_where_pooled_persona_rate_would_not(self):
        # Round-4 BLOCKER 1 (the codex fixture): 7 capped rows confined to
        # persona 5's k=16 cell at n=5000. Cell (16,5) has 313 rows ->
        # 7/313 = 2.24% > 2% FIRES, while the pooled persona-5 rate over the
        # arm-8 denominator is 7/625 = 1.12% <= 2% -- the superseded v10
        # per-persona trigger would NOT have fired (the round-3 bug).
        n = lg.N_CONTEXTS_FULL
        assignment = lg.build_assignment(n)
        rows_16_5 = [i for i in range(n) if i % 16 == 5]
        rows_8_5 = [i for i in range(n) if i % 8 == 5]
        assert len(rows_16_5) == 313 and len(rows_8_5) == 625
        capped = rows_16_5[:7]
        assert lg.CAP_HIT_REGEN_FRACTION < 7 / 313  # the cell trigger fires
        assert lg.CAP_HIT_REGEN_FRACTION >= 7 / 625  # the pooled rate would not
        stop = {lg.make_item_id(5, i): "max_tokens" for i in capped}
        triggered, regen = lg.cells_over_cap_threshold(stop, assignment)
        assert set(triggered) == {"k=16,p=5"}  # the 625-row k=8 cell stays quiet
        cell = triggered["k=16,p=5"]
        assert cell["k"] == 16 and cell["persona"] == 5
        assert cell["n_rows"] == 313 and cell["n_over_cap"] == 7
        assert regen == {5: sorted(capped)}

    def test_multi_arm_row_triggers_all_containing_cells_once(self):
        # Under nesting a single (context, persona) ROW belongs to SEVERAL
        # arms' cells; each containing cell computes its own fraction, and
        # the regen union dedupes by PAIR identity (a pair regenerates at
        # most once however many of its cells triggered).
        n = 32
        assignment = lg.build_assignment(n)
        stop = {
            lg.make_item_id(3, 3): "max_tokens",
            lg.make_item_id(3, 19): "max_tokens",
        }
        triggered, regen = lg.cells_over_cap_threshold(stop, assignment)
        assert set(triggered) == {"k=4,p=3", "k=8,p=3", "k=16,p=3"}
        assert triggered["k=4,p=3"]["n_rows"] == 8
        assert triggered["k=4,p=3"]["n_over_cap"] == 2
        assert triggered["k=16,p=3"]["n_rows"] == 2
        assert regen == {3: [3, 19]}  # deduped union, ONE entry per pair

    def test_under_threshold_cell_stays_quiet(self):
        # 6/313 = 1.92% <= 2%: strictly-greater comparison does not fire.
        n = lg.N_CONTEXTS_FULL
        assignment = lg.build_assignment(n)
        capped = [i for i in range(n) if i % 16 == 5][:6]
        stop = {lg.make_item_id(5, i): "max_tokens" for i in capped}
        triggered, regen = lg.cells_over_cap_threshold(stop, assignment)
        assert triggered == {} and regen == {}

    def test_union_feeds_one_pooled_dispatch(self, tmp_path, monkeypatch):
        # BLOCKER 1 end-to-end: cell-trigger output -> ONE pooled dispatch at
        # REGEN_MAX_TOKENS covering the deduped union (FIX D's
        # single-round-trip property survives the per-cell trigger).
        n = 32
        assignment = lg.build_assignment(n)
        stop = {
            lg.make_item_id(3, 3): "max_tokens",
            lg.make_item_id(3, 19): "max_tokens",
            lg.make_item_id(5, 5): "max_tokens",
        }
        _triggered, regen = lg.cells_over_cap_threshold(stop, assignment)
        union_ids = sorted(lg.make_item_id(p, i) for p, rows in regen.items() for i in rows)
        items_by_id = {iid: DispatchItem(item_id=iid, payload={}) for iid in union_ids}
        calls = []

        async def fake_dispatch(items, checkpoint_dir, max_tokens, poll_interval):
            calls.append((sorted(it.item_id for it in items), max_tokens))
            return {it.item_id: _ok_result(it.item_id) for it in items}

        monkeypatch.setattr(lg, "_dispatch", fake_dispatch)
        _rg, pooled_ids, _dir = lg._dispatch_pooled_regen(tmp_path, items_by_id, regen, 1.0)
        assert len(calls) == 1  # exactly ONE regen dispatch, no per-persona loop
        assert calls[0] == (union_ids, lg.REGEN_MAX_TOKENS)
        assert pooled_ids == union_ids


# ── FIX D (follow-up round): pooled cap-hit regen dispatch ───────────────────


class TestPooledRegenDispatch:
    def test_one_dispatch_covers_union_of_triggered_personas(self, tmp_path, monkeypatch):
        # Rows from several triggered CELLS (round 4: the trigger is per
        # (arm x persona) cell) => exactly ONE regeneration dispatch, whose
        # submitted id set is the UNION of the triggered cells' over-cap
        # pair rows (the pre-fix path dispatched one SERIAL batch per
        # persona, each waiting out its own Batch window).
        regen_personas = {0: [0, 4], 3: [7], 5: [1, 2]}
        union_ids = sorted(
            lg.make_item_id(p, i) for p, rows in regen_personas.items() for i in rows
        )
        items_by_id = {iid: DispatchItem(item_id=iid, payload={}) for iid in union_ids}
        calls = []

        async def fake_dispatch(items, checkpoint_dir, max_tokens, poll_interval):
            calls.append((sorted(it.item_id for it in items), checkpoint_dir, max_tokens))
            return {it.item_id: _ok_result(it.item_id) for it in items}

        monkeypatch.setattr(lg, "_dispatch", fake_dispatch)
        rg, pooled_ids, rg_dir = lg._dispatch_pooled_regen(
            tmp_path, items_by_id, regen_personas, poll_interval=1.0
        )
        assert len(calls) == 1  # exactly ONE regeneration dispatch
        assert calls[0][0] == union_ids
        assert calls[0][1] == tmp_path / "regen_pooled"
        assert calls[0][2] == lg.REGEN_MAX_TOKENS
        assert pooled_ids == union_ids
        assert rg_dir == tmp_path / "regen_pooled"
        assert sorted(rg) == union_ids

    def test_no_per_persona_serial_dispatch_remains(self):
        src = pathlib.Path(lg.__file__).read_text()
        # The serial per-persona checkpoint dirs are gone from the live path.
        assert 'root / f"regen_p{p:02d}"' not in src
        # Regen dispatches exactly once, via the pooled helper (1 def + 1 call).
        assert src.count("_dispatch_pooled_regen(") == 2


# ── FIX C (follow-up round): cumulative re-drive ceiling ─────────────────────


class TestCumulativeRedriveCeiling:
    def test_seven_stale_dirs_trip_the_ceiling(self, tmp_path):
        for n in range(1, 8):
            (tmp_path / f"redrive{n}").mkdir()
        nxt = lg._next_redrive_round(tmp_path)
        assert nxt == 8
        with pytest.raises(SystemExit) as excinfo:
            lg._require_redrive_headroom(nxt, n_pending=3)
        assert excinfo.value.code == 3

    def test_at_ceiling_round_still_allowed(self, tmp_path):
        for n in range(1, 6):
            (tmp_path / f"redrive{n}").mkdir()
        nxt = lg._next_redrive_round(tmp_path)
        assert nxt == 6 == lg.MAX_CUMULATIVE_REDRIVE_ROUNDS
        lg._require_redrive_headroom(nxt, n_pending=3)  # 6th cumulative round: no raise

    def test_merge_and_ceiling_wired_in_main(self):
        src = pathlib.Path(lg.__file__).read_text()
        # Merge call (rindex: the def precedes main's call site) runs BEFORE
        # the fresh-round numbering / pending-set loop.
        assert src.rindex("_merge_stale_redrives(") < src.index(
            "first_round = _next_redrive_round("
        )
        # Ceiling check runs BEFORE any fresh redrive dispatch.
        assert src.rindex("_require_redrive_headroom(") < src.index('root / f"redrive{rnd}"')


# ── FIX 1: canonical-repo completion gate ────────────────────────────────────


class TestCanonicalUploadGate:
    CANON = f"{lg.DATA_REPO}/{lg.HF_PREFIX}/raw_completions/ladder"

    def test_canonical_url_passes(self):
        lg._require_canonical_upload(self.CANON, self.CANON)  # no raise

    def test_overflow_reroute_raises_with_pointer(self):
        overflow = (
            f"superkaiba1/explore-persona-space-overflow/{lg.HF_PREFIX}/raw_completions/ladder"
        )
        with pytest.raises(RuntimeError, match=re.escape(overflow)):
            lg._require_canonical_upload(overflow, self.CANON)

    def test_wrong_prefix_on_canonical_repo_raises(self):
        with pytest.raises(RuntimeError, match="canonical"):
            lg._require_canonical_upload(f"{lg.DATA_REPO}/somewhere_else", self.CANON)


# ── Frozen-question loader (requirement 3) ───────────────────────────────────


def _write_frozen_fixture(tmp_path, permute: bool):
    records = [
        {
            "context_id": i,
            "question": f"q{i}",
            "answer_text": "a",
            "arm": "b2",
            "seed": 42,
            "filled": True,
            "in_common_valid": i % 2 == 0,
        }
        for i in range(lg.N_CONTEXTS_FULL)
    ]
    if permute:
        records[0], records[1] = records[1], records[0]
    b2 = tmp_path / "b2_seed42.json"
    b2.write_text(json.dumps(records))
    common = sorted(r["context_id"] for r in records if r["in_common_valid"])
    cv = tmp_path / "common_valid_idx.json"
    cv.write_text(json.dumps({"n_common": len(common), "common_valid_idx": common}))
    return b2, cv


class TestFrozenQuestionLoader:
    @staticmethod
    def _patch(monkeypatch, b2, cv):
        def fake_hf_hub_download(
            repo_id, filename, *, repo_type=None, revision=None, local_dir=None
        ):
            assert repo_id == lg.DATA_REPO
            assert revision == lg.PARENT_REV
            return str(b2 if filename.endswith("b2_seed42.json") else cv)

        monkeypatch.setattr(lg, "hf_hub_download", fake_hf_hub_download)

    def test_frozen_order_loads(self, tmp_path, monkeypatch):
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        self._patch(monkeypatch, b2, cv)
        questions, in_common, xcheck = lg.load_frozen_questions(tmp_path / "dl")
        assert len(questions) == lg.N_CONTEXTS_FULL
        assert questions[0] == "q0"
        assert in_common[0] is True and in_common[1] is False
        assert xcheck["set_equal"] is True

    def test_permuted_order_raises(self, tmp_path, monkeypatch):
        # The frozen-set contract is monotone-unique context_id 0..4999 (file
        # order IS the frozen context order) — a permutation must fail loud.
        b2, cv = _write_frozen_fixture(tmp_path, permute=True)
        self._patch(monkeypatch, b2, cv)
        with pytest.raises(AssertionError, match="monotone-unique"):
            lg.load_frozen_questions(tmp_path / "dl")


# ── Plan v13 §4.3 step 4: registered cap ladder + gen_wave labels ────────────


class TestGenerationConfigFingerprint:
    def test_registered_caps_and_wave_labels(self):
        # Plan v13 §4.3 step 4: base 4096, ONE regen round at 8192; the regen
        # wave label is the registered literal "regen-8192".
        assert lg.GEN_MAX_TOKENS == 4096
        assert lg.REGEN_MAX_TOKENS == 8192
        assert lg.GEN_WAVE_FIRST == "first"
        assert lg.GEN_WAVE_REGEN == "regen-8192"

    def test_fingerprint_fields_complete(self):
        # Plan v13 §4.3 step 3b(i): sha256 over exactly (model, temperature,
        # base cap, regen cap, roster/template hash, context pin, mask-gate
        # schema id).
        fp = lg.generation_config_fingerprint()
        assert set(fp["fields"]) == {
            "model",
            "temperature",
            "gen_max_tokens",
            "regen_max_tokens",
            "roster_template_hash",
            "context_pin",
            "mask_gate_schema_id",
        }
        assert fp["fields"]["model"] == lg.SONNET_MODEL
        assert fp["fields"]["gen_max_tokens"] == 4096
        assert fp["fields"]["regen_max_tokens"] == 8192
        assert fp["fields"]["context_pin"] == lg.PARENT_REV
        assert fp["fields"]["mask_gate_schema_id"] == lg.MASK_GATE_SCHEMA_ID
        assert re.fullmatch(r"[0-9a-f]{64}", fp["sha256"])

    def test_resume_stability_across_process_time(self, monkeypatch):
        # HASH-BASIS PIN direction (b) (plan v13 §4.3 step 3b(i)): two
        # fingerprint computations over an IDENTICAL config at different
        # wall-clock times must be byte-identical — a metadata-bearing hash
        # (e.g. over the whole roster.json with its generated_at) violates
        # exactly this and would reject a healthy resume.
        fp1 = lg.generation_config_fingerprint()
        time.sleep(0.05)
        monkeypatch.setattr(lg, "_utc_now", lambda: "2099-01-01T00:00:00Z")
        fp2 = lg.generation_config_fingerprint()
        assert json.dumps(fp1, sort_keys=True) == json.dumps(fp2, sort_keys=True)

    def test_roster_template_hash_is_metadata_free(self):
        # The hash basis is EXACTLY {template, personas(idx, name, card)} —
        # never the generated roster.json as a whole (whose metadata block
        # embeds generated_at).
        basis = {
            "template": lg.PERSONA_TEMPLATE,
            "personas": [
                {"idx": p, "name": name, "card": card} for p, (name, card) in enumerate(lg.PERSONAS)
            ],
        }
        blob = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert "generated_at" not in blob
        assert lg.roster_template_hash() == hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── Plan v13 §4.3 step 3b(ii): resume rejects mismatched state pre-dispatch ──


class TestResumeFingerprintGate:
    @staticmethod
    def _rewrite_cfg(ckpt: pathlib.Path, mutate) -> None:
        cfg = json.loads((ckpt / lg.GEN_CONFIG_FILENAME).read_text())
        mutate(cfg)
        blob = json.dumps(
            cfg["fingerprint"]["fields"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        cfg["fingerprint"]["sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        (ckpt / lg.GEN_CONFIG_FILENAME).write_text(json.dumps(cfg))

    def test_fresh_dir_persists_then_same_config_resume_passes(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "batches"
        assert lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS) == lg.GEN_MAX_TOKENS
        cfg = json.loads((ckpt / lg.GEN_CONFIG_FILENAME).read_text())
        assert cfg["max_tokens"] == lg.GEN_MAX_TOKENS
        assert cfg["fingerprint"] == lg.generation_config_fingerprint()
        # Same-config resume: no halt, and the dispatcher IS reached.
        called = []

        async def fake_dispatch_calls(items, **kwargs):
            called.append(len(items))
            return {}

        monkeypatch.setattr(lg, "dispatch_calls", fake_dispatch_calls)
        asyncio.run(lg._dispatch([], ckpt, lg.GEN_MAX_TOKENS, 0.1))
        assert called == [0]

    def test_mutated_config_halts_before_dispatch(self, tmp_path, monkeypatch):
        # Fixture direction (a): a checkpoint persisted under a DIFFERENT
        # config (here: another model) must halt with rc 4 BEFORE any
        # dispatch — the recorder proves dispatch_calls was never reached,
        # so no persisted row could have been re-served.
        ckpt = tmp_path / "batches"
        lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)

        def mutate(cfg):
            cfg["fingerprint"]["fields"]["model"] = "claude-other-model-20990101"

        self._rewrite_cfg(ckpt, mutate)
        called = []

        async def fake_dispatch_calls(items, **kwargs):
            called.append(len(items))
            return {}

        monkeypatch.setattr(lg, "dispatch_calls", fake_dispatch_calls)
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(lg._dispatch([], ckpt, lg.GEN_MAX_TOKENS, 0.1))
        assert excinfo.value.code == lg.EXIT_CONFIG_MISMATCH == 4
        assert called == []  # designed halt BEFORE any dispatch / row re-serve
        report = json.loads((tmp_path / "fingerprint_mismatch_batches.json").read_text())
        assert report["reason"] == "generation_config_fingerprint_mismatch"
        assert "model" in report["differing_fields"]
        assert report["persisted_fingerprint"]["fields"]["model"] == "claude-other-model-20990101"
        assert report["live_fingerprint"] == lg.generation_config_fingerprint()

    def test_pilot_cap_checkpoint_rejected_before_dispatch(self, tmp_path):
        # Plan §4.3 step 3b mechanization: a fixture checkpoint written with
        # the pilot's 1024/2048 caps MUST make the resume path fail before
        # dispatch (the exact stale-state shape a cap re-tune leaves behind).
        ckpt = tmp_path / "batches"
        lg.check_or_persist_gen_config(ckpt, 1024)

        def mutate(cfg):
            cfg["fingerprint"]["fields"]["gen_max_tokens"] = 1024
            cfg["fingerprint"]["fields"]["regen_max_tokens"] = 2048
            cfg["max_tokens"] = 1024

        self._rewrite_cfg(ckpt, mutate)
        with pytest.raises(SystemExit) as excinfo:
            lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        assert excinfo.value.code == 4
        report = json.loads((tmp_path / "fingerprint_mismatch_batches.json").read_text())
        diffs = report["differing_fields"]
        assert {"gen_max_tokens", "regen_max_tokens", "max_tokens"} <= set(diffs)
        assert diffs["gen_max_tokens"] == {"persisted": 1024, "live": 4096}
        assert diffs["regen_max_tokens"] == {"persisted": 2048, "live": 8192}

    def test_unfingerprinted_checkpoint_with_state_halts(self, tmp_path):
        # A dispatcher state.json with NO gen_config.json is a
        # pre-fingerprint checkpoint: unverifiable, refuse to re-serve.
        ckpt = tmp_path / "batches"
        ckpt.mkdir(parents=True)
        (ckpt / "state.json").write_text(json.dumps({"cid_to_item": {}, "sub_batches": []}))
        with pytest.raises(SystemExit) as excinfo:
            lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        assert excinfo.value.code == 4
        report = json.loads((tmp_path / "fingerprint_mismatch_batches.json").read_text())
        assert report["reason"] == "unfingerprinted_checkpoint"
        assert report["persisted_fingerprint"] is None
        assert report["live_fingerprint"] == lg.generation_config_fingerprint()

    def test_corrupt_sha_only_halts(self, tmp_path):
        # Fields identical but a corrupted sha still halts (fail-closed) —
        # and (MINOR 5, round 4) the halt report still NAMES what differs:
        # differing_fields carries an explicit fingerprint_sha256 entry
        # rather than halting with an empty diagnostic.
        ckpt = tmp_path / "batches"
        lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        cfg = json.loads((ckpt / lg.GEN_CONFIG_FILENAME).read_text())
        cfg["fingerprint"]["sha256"] = "0" * 64
        (ckpt / lg.GEN_CONFIG_FILENAME).write_text(json.dumps(cfg))
        with pytest.raises(SystemExit) as excinfo:
            lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        assert excinfo.value.code == 4
        report = json.loads((tmp_path / "fingerprint_mismatch_batches.json").read_text())
        diffs = report["differing_fields"]
        assert "fingerprint_sha256" in diffs
        assert diffs["fingerprint_sha256"]["persisted"] == "0" * 64
        assert diffs["fingerprint_sha256"]["live"] == lg.generation_config_fingerprint()["sha256"]


# ── Plan v13 §4.3 step 3b(iii): per-row caps from PERSISTED metadata ─────────


class TestPerRowCapFromPersistedMetadata:
    def test_cap_read_from_checkpoint_not_live_constant(self, tmp_path, monkeypatch):
        ckpt = tmp_path / "batches"
        lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        # Simulate a later cap re-tune: the live constant moves, the
        # checkpoint's persisted request metadata does not.
        monkeypatch.setattr(lg, "GEN_MAX_TOKENS", 9999)
        assert lg.checkpoint_max_tokens(ckpt) == 4096
        # And the fingerprint gate refuses to RESUME that checkpoint under
        # the re-tuned config (belt + suspender are both live).
        with pytest.raises(SystemExit) as excinfo:
            lg.check_or_persist_gen_config(ckpt, lg.GEN_MAX_TOKENS)
        assert excinfo.value.code == 4

    def test_missing_gen_config_fails_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lg.checkpoint_max_tokens(tmp_path / "nonexistent")


# ── Plan v13 §4.3 step 4 (v11): refusal>empty label precedence ───────────────


class TestClassifyValidityPrecedence:
    def test_refusal_stop_beats_empty_error_category(self):
        # The API-level refusal shape: succeeded-but-empty content arriving
        # as an error/empty_response DispatchResult with stop_reason refusal.
        res = DispatchResult(
            item_id="x", result=None, error=True, category="empty_response", stop_reason="refusal"
        )
        assert lg.classify_validity(res) == "refusal"

    def test_refusal_stop_beats_empty_content_nonerror(self):
        res = DispatchResult(
            item_id="x", result="", error=False, category="ok", stop_reason="refusal"
        )
        assert lg.classify_validity(res) == "refusal"

    def test_empty_without_refusal_stays_empty(self):
        res = DispatchResult(
            item_id="x", result="", error=False, category="ok", stop_reason="end_turn"
        )
        assert lg.classify_validity(res) == "empty"
        res2 = DispatchResult(
            item_id="x", result=None, error=True, category="empty_response", stop_reason=None
        )
        assert lg.classify_validity(res2) == "empty"

    def test_transport_and_ok_classes_unchanged(self):
        res = DispatchResult(
            item_id="x", result=None, error=True, category=lg.RESULT_TRANSPORT, stop_reason=None
        )
        assert lg.classify_validity(res) == f"error:{lg.RESULT_TRANSPORT}"
        assert lg.classify_validity(_mk_result("x")) == "ok"


# ── Plan v13 §4.3 P0: prompt-integrity gate (asserts (a)-(d)) ────────────────


def _mk_p0_artifacts(n: int = 4):
    """Healthy P0 fixture: roster/assignment/records/questions as main persists them.

    Returns ``(roster_obj, assignment_obj, by_persona, questions)`` — the
    questions list is the pinned-bundle AUTHORITY assert (a) compares against
    (round-4 BLOCKER 2), shaped like the frozen fixture (``q{i}``).
    """
    assignment = lg.build_assignment(n)
    pairs = lg.verify_assignment(assignment, n)
    assignment_obj = {
        "n_contexts": n,
        "arms": {str(k): assignment[k] for k in lg.K_ARMS},
    }
    roster_obj = {
        "template": lg.PERSONA_TEMPLATE,
        "personas": [
            {"idx": p, "name": name, "card": card} for p, (name, card) in enumerate(lg.PERSONAS)
        ],
    }
    questions = [f"q{i}" for i in range(n)]
    by_persona: dict[int, list[dict]] = {p: [] for p in range(lg.N_PERSONAS)}
    for i, p in sorted(pairs):
        sp = lg.persona_system(p)
        by_persona[p].append(
            {
                "context_id": i,
                "persona_idx": p,
                "question": questions[i],
                "system_prompt": sp,
                "system_prompt_sha256": hashlib.sha256(sp.encode("utf-8")).hexdigest(),
                "max_tokens": lg.GEN_MAX_TOKENS,
                "gen_wave": lg.GEN_WAVE_FIRST,
            }
        )
    return roster_obj, assignment_obj, by_persona, questions


def _stage_p0_artifacts(stage: pathlib.Path, n: int = 4):
    """Write a healthy P0 artifact set (persona files + roster + assignment) into ``stage``.

    Module-level so both the ``run_p0_verify`` tests (stage under
    ``tmp_path/stage``) and the ``main()`` fork tests (stage under
    ``<out_root>/hf_stage/ladder`` — the exact path main composes) share one
    staging shape. Returns ``(by_persona, questions)``.
    """
    roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts(n)
    stage.mkdir(parents=True)
    for p in range(lg.N_PERSONAS):
        (stage / f"persona{p:02d}_seed42.json").write_text(
            json.dumps({"metadata": {}, "records": by_persona[p]})
        )
    (stage / "roster.json").write_text(json.dumps(roster_obj))
    (stage / "assignment.json").write_text(json.dumps(assignment_obj))
    return by_persona, questions


class TestP0PromptIntegrityGate:
    def test_reconstruction_agrees_with_dispatch_serializer_when_healthy(self):
        # Transcription fidelity: the independent §4.1 transcription and the
        # dispatch serializer produce byte-identical prompts for all 16
        # personas (also catches any tool-transit Unicode mangling).
        for p in range(lg.N_PERSONAS):
            assert lg.p0_reconstruct_system(p) == lg.persona_system(p)

    def test_reconstruction_independent_of_dispatch_serializer(self, monkeypatch):
        # Mechanical independence proof: corrupt the dispatch-side template
        # and make persona_system raise — the P0 reconstruction is unmoved.
        expected = lg.p0_reconstruct_system(3)

        def boom(p):
            raise AssertionError("P0 reconstruction must not call persona_system")

        monkeypatch.setattr(lg, "persona_system", boom)
        monkeypatch.setattr(lg, "PERSONA_TEMPLATE", "CORRUPTED {name} {card}")
        monkeypatch.setattr(lg, "PERSONAS", [("X", "Y")] * lg.N_PERSONAS)
        assert lg.p0_reconstruct_system(3) == expected

    def test_reconstruction_source_references_no_dispatch_constants(self):
        # co_names carries every global + attribute/method name the compiled
        # body actually references (docstring/comment prose is invisible), so
        # this pins that the reconstruction never touches the dispatch-side
        # constants/serializer nor any str.format call.
        names = lg.p0_reconstruct_system.__code__.co_names
        for banned in ("PERSONA_TEMPLATE", "PERSONAS", "persona_system", "format"):
            assert banned not in names, f"P0 reconstruction references dispatch-side {banned!r}"

    def test_gate_passes_on_healthy_artifacts(self, tmp_path):
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        lg.p0_prompt_integrity_gate(
            roster_obj,
            assignment_obj,
            by_persona,
            tmp_path,
            questions,
            expected_n_contexts=len(questions),
        )
        assert not (tmp_path / "p0_integrity_report.json").exists()

    def _expect_halt(
        self, tmp_path, roster_obj, assignment_obj, by_persona, questions, key, expected_n=None
    ):
        with pytest.raises(SystemExit) as excinfo:
            lg.p0_prompt_integrity_gate(
                roster_obj,
                assignment_obj,
                by_persona,
                tmp_path,
                questions,
                expected_n_contexts=expected_n if expected_n is not None else len(questions),
            )
        assert excinfo.value.code == lg.EXIT_P0_INTEGRITY == 5
        report = json.loads((tmp_path / "p0_integrity_report.json").read_text())
        assert report["reason"] == "p0_prompt_integrity_failure"
        assert key in report["failures_by_assert"]
        return report

    def test_mutated_payload_halts(self, tmp_path):
        # Plan §4.3 P0 mechanization: mutating ONE persona's serialized
        # payload in a fixture generation record MUST make P0 halt, with the
        # report naming the pair and BOTH serializations.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        victim = by_persona[1][0]
        victim["system_prompt"] = victim["system_prompt"] + " INJECTED"
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        f = next(x for x in report["failures_first50"] if x["assert"] == "c")
        assert f["kind"] == "prompt_mismatch"
        assert f["persona_idx"] == 1
        assert f["context_id"] == victim["context_id"]
        assert f["persisted_system_prompt"].endswith("INJECTED")
        assert f["reconstructed_system_prompt"] == lg.p0_reconstruct_system(1)

    def test_mutated_sha_alone_halts(self, tmp_path):
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[0][0]["system_prompt_sha256"] = "0" * 64
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")

    def test_mutated_roster_halts(self, tmp_path):
        # Round-4 STYLE 7: roster/template transcription equality is (c)'s
        # reconstruction-authority PRECONDITION and is labeled "c" — the plan's
        # assert (a) is the question-vs-pinned-bundle read, not the roster.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        roster_obj["personas"][5]["card"] = "a different card"
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")

    def test_mutated_template_halts(self, tmp_path):
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        roster_obj["template"] = roster_obj["template"].replace("Stay fully", "Stay mostly")
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")

    def test_wrong_assignment_halts(self, tmp_path):
        # Assert (b): the gate recomputes persona(i, k) = i mod k with its
        # OWN arithmetic — a flipped entry in one arm must halt.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assignment_obj["arms"]["2"][3] = 0  # registered value is 3 % 2 == 1
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "b")

    def test_malformed_assignment_shape_halts_not_crashes(self, tmp_path):
        # Round-4 BLOCKER 2: the gate validates its own DOMAIN inputs — a
        # non-int n_contexts and a non-list arm route through the rc-5 report
        # path (assert (b) structural entries), never an unhandled exception.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assignment_obj["n_contexts"] = "4"
        assignment_obj["arms"]["2"] = {"not": "a list"}
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "b")
        fields = {x.get("field") for x in report["failures_first50"] if x["assert"] == "b"}
        assert {"n_contexts", "arm_type"} <= fields

    def test_declared_domain_mismatch_is_recorded_b_failure(self, tmp_path):
        # Round-5 BLOCKER 1 (gate leg): a COHERENT artifact set at n=4 —
        # every internal-consistency check satisfiable — must still halt
        # rc-5 when the CALLER pins a different domain, and the expected
        # pair set derives from the CALLER's value (missing pairs for
        # contexts 4..7 prove the authority drives the derivation).
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        report = self._expect_halt(
            tmp_path, roster_obj, assignment_obj, by_persona, questions, "b", expected_n=8
        )
        f = next(x for x in report["failures_first50"] if x.get("field") == "n_contexts_authority")
        assert f["generated"] == 4 and f["expected"] == 8
        assert any(x.get("kind") == "missing_pair" for x in report["failures_first50"])

    def test_boolean_context_id_impersonating_pair_halts(self, tmp_path):
        # Round-5 BLOCKER 2: isinstance(True, int) is True, True == 1, and
        # hash(True) == hash(1) — a record carrying context_id: true ALIASES
        # pair (1, p) in question indexing, dedup hashing, and expected-set
        # equality, so the pre-fix isinstance schema certified this artifact
        # as a clean PASS. type-is-int rejects it as malformed_record.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assert by_persona[1][0]["context_id"] == 1  # the legitimate pair it aliases
        by_persona[1][0]["context_id"] = True
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        f = next(x for x in report["failures_first50"] if x.get("kind") == "malformed_record")
        assert f["persona_bucket"] == 1

    def test_boolean_persona_idx_impersonating_persona_halts(self, tmp_path):
        # Round-5 BLOCKER 2 twin: persona_idx: true aliases persona 1 in the
        # bucket-agreement check, the roster index, and the pair tuple — the
        # pre-fix schema passed it; type-is-int rejects it.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[1][0]["persona_idx"] = True
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        assert any(
            x.get("kind") == "malformed_record" and x.get("persona_bucket") == 1
            for x in report["failures_first50"]
        )

    def test_boolean_n_contexts_rejected_by_strict_schema(self, tmp_path):
        # Round-5 BLOCKER 2: a JSON-boolean n_contexts fails the type-is-int
        # gate DIRECTLY (field n_contexts), never a downstream arm-length
        # symptom.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assignment_obj["n_contexts"] = True
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "b")
        f = next(x for x in report["failures_first50"] if x.get("field") == "n_contexts")
        assert f["generated"] is True

    def test_nonnumeric_arm_key_routes_to_report_not_crash(self, tmp_path):
        # Round-5 BLOCKER 3: a nonnumeric arm key previously raised inside
        # int(kv[0]) — an unhandled ValueError bypassing the designed rc-5
        # report path.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assignment_obj["arms"]["x"] = [0, 0, 0, 0]
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "b")
        f = next(x for x in report["failures_first50"] if x.get("field") == "arm_key")
        assert f["generated"] == repr("x")

    def test_zero_arm_key_routes_to_report_not_crash(self, tmp_path):
        # Round-6 BLOCKER 3: a nonempty UNREGISTERED integer arm key "0"
        # previously reached the per-arm modulo (i % k with k == 0) and raised
        # ZeroDivisionError BEFORE the designed rc-5 report. Parsed arm keys
        # are now validated against the registered K_ARMS before any modulo,
        # and an unregistered key routes to the (b) report path.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        assignment_obj["arms"]["0"] = [0, 0, 0, 0]
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "b")
        f = next(x for x in report["failures_first50"] if x.get("field") == "arm_key_unregistered")
        assert f["generated"] == 0
        assert f["registered"] == list(lg.K_ARMS)

    def test_non_dict_roster_root_routes_to_report_not_crash(self, tmp_path):
        # Round-5 BLOCKER 3: a non-dict roster.json root previously died on
        # .get (AttributeError) before the report machinery ran.
        _roster, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        report = self._expect_halt(
            tmp_path, ["not", "an", "object"], assignment_obj, by_persona, questions, "c"
        )
        assert any(x.get("field") == "roster_root" for x in report["failures_first50"])

    def test_non_dict_assignment_root_routes_to_report_not_crash(self, tmp_path):
        # Round-5 BLOCKER 3: a non-dict assignment.json root previously died
        # on .get (AttributeError) before the report machinery ran.
        roster_obj, _assignment, by_persona, questions = _mk_p0_artifacts()
        report = self._expect_halt(
            tmp_path, roster_obj, "not an object", by_persona, questions, "b"
        )
        assert any(x.get("field") == "assignment_root" for x in report["failures_first50"])

    def test_deleted_record_halts_missing_pair(self, tmp_path):
        # Round-4 BLOCKER 2 fixture 1: the expected pair set is derived
        # INDEPENDENTLY (persona(i, k) = i mod k over range(n_contexts)) and
        # compared by EXACT set equality — silently deleting one healthy row
        # FAILs (the round-3 gate looped over whatever rows were present and
        # could never see an absence).
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        dropped = by_persona[1].pop(0)
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        f = next(x for x in report["failures_first50"] if x.get("kind") == "missing_pair")
        assert (f["context_id"], f["persona_idx"]) == (dropped["context_id"], 1)

    def test_duplicated_record_halts(self, tmp_path):
        # Round-4 BLOCKER 2 fixture 2: records flatten under UNIQUE
        # (context_id, persona_idx) keys — a duplicated row FAILs.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[1].append(dict(by_persona[1][0]))
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        assert any(x.get("kind") == "duplicate_pair" for x in report["failures_first50"])

    def test_wrong_bucket_record_halts(self, tmp_path):
        # Round-4 BLOCKER 2 fixture 3: a record filed under the WRONG persona
        # bucket FAILs via the rec["persona_idx"]-vs-bucket agreement check —
        # the pair set itself stays complete (the record carries its own
        # correct identity), so bucket_mismatch is the catching entry.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[2].append(by_persona[1].pop(0))
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        f = next(x for x in report["failures_first50"] if x.get("kind") == "bucket_mismatch")
        assert f["persona_idx"] == 1 and f["persona_bucket"] == 2

    def test_consistently_mutated_question_halts_against_bundle(self, tmp_path):
        # Round-4 BLOCKER 2 fixture 4 / assert (a): the question authority is
        # the PINNED BUNDLE — a question mutated CONSISTENTLY across every
        # record of one context (cross-file consistency intact, so any
        # records-vs-records read would pass) still FAILs.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        for recs in by_persona.values():
            for r in recs:
                if r["context_id"] == 2:
                    r["question"] = "a consistently wrong question"
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "a")
        f = next(x for x in report["failures_first50"] if x.get("kind") == "question_mismatch")
        assert f["context_id"] == 2
        assert f["bundle_question"] == questions[2]
        assert f["persisted_question"] == "a consistently wrong question"

    def test_all_empty_record_set_halts(self, tmp_path):
        # Round-4 BLOCKER 2: an all-empty record set is EVERY expected pair
        # missing — a designed rc-5 halt, never a vacuous PASS.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        empty = {p: [] for p in by_persona}
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, empty, questions, "c")
        kinds = {x.get("kind") for x in report["failures_first50"] if x["assert"] == "c"}
        assert "missing_pair" in kinds

    def test_malformed_record_shapes_route_to_report_not_crash(self, tmp_path):
        # Round-4 BLOCKER 2: malformed shapes (non-dict record, non-list
        # bucket) go through the designed rc-5 report path — never an
        # unhandled exception.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[0].append("not a dict")
        by_persona[1].append({"context_id": "x", "persona_idx": 1})
        by_persona[3] = "not a list"
        report = self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "c")
        kinds = {x.get("kind") for x in report["failures_first50"]}
        assert {"malformed_record", "malformed_bucket"} <= kinds

    def test_cap_wave_inconsistency_halts(self, tmp_path):
        # Plan §4.3 step 3b(iv) / P0 assert (d): a persisted cap that
        # disagrees with its wave label MUST fail the gate.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[0][0]["max_tokens"] = lg.REGEN_MAX_TOKENS  # wave stays "first"
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "d")

    def test_pilot_cap_record_halts(self, tmp_path):
        # Assert (d): persisted requested-max_tokens outside {4096, 8192}
        # (e.g. the pilot's 1024) is a designed halt.
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[0][0]["max_tokens"] = 1024
        self._expect_halt(tmp_path, roster_obj, assignment_obj, by_persona, questions, "d")

    def test_regen_cap_with_regen_wave_passes(self, tmp_path):
        roster_obj, assignment_obj, by_persona, questions = _mk_p0_artifacts()
        by_persona[0][0]["max_tokens"] = lg.REGEN_MAX_TOKENS
        by_persona[0][0]["gen_wave"] = lg.GEN_WAVE_REGEN
        lg.p0_prompt_integrity_gate(
            roster_obj,
            assignment_obj,
            by_persona,
            tmp_path,
            questions,
            expected_n_contexts=len(questions),
        )
        assert not (tmp_path / "p0_integrity_report.json").exists()

    def test_p0_constants_are_literal_at_source_level(self):
        # Round-4 MINOR 6: the co_names pin above covers the reconstruction
        # FUNCTION; this pins the CONSTANTS — the Assign/AnnAssign values for
        # _P0_TEMPLATE / _P0_ROSTER contain no ast.Name node (no aliasing of
        # PERSONA_TEMPLATE / PERSONAS, however spelled). Deliberately NOT an
        # `is not` identity check: CPython constant-folds/interns identical
        # literals, so object identity is meaningless for byte-equal strings.
        import ast

        tree = ast.parse(pathlib.Path(lg.__file__).read_text())
        found: dict = {}
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            for name in targets:
                if name in ("_P0_TEMPLATE", "_P0_ROSTER"):
                    assert name not in found, f"duplicate assignment to {name}"
                    found[name] = node.value
        assert set(found) == {"_P0_TEMPLATE", "_P0_ROSTER"}
        allowed = (ast.Tuple, ast.List, ast.Constant, ast.Load)
        for name, value in found.items():
            for sub in ast.walk(value):
                assert isinstance(sub, allowed), (
                    f"{name} value contains non-literal node {type(sub).__name__} — "
                    "the P0 transcription must be a literal, never an alias/expression"
                )

    def _stage_artifacts(self, tmp_path, n: int = 4):
        stage = tmp_path / "stage"
        by_persona, questions = _stage_p0_artifacts(stage, n)
        return stage, by_persona, questions

    def test_run_p0_verify_roundtrip(self, tmp_path, monkeypatch):
        # The --p0-verify entrypoint reads the persisted artifact shapes main
        # writes (persona files with a records key + roster + assignment) and
        # re-downloads the question AUTHORITY at the pinned parent revision
        # (round-4 BLOCKER 2 — never questions-from-the-records-themselves).
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, by_persona, _questions = self._stage_artifacts(tmp_path)
        lg.run_p0_verify(stage, tmp_path / "reports", tmp_path / "dl", 4)  # healthy: no raise
        by_persona[2][0]["gen_wave"] = "regen-8192"  # cap stays 4096: inconsistent
        (stage / "persona02_seed42.json").write_text(
            json.dumps({"metadata": {}, "records": by_persona[2]})
        )
        with pytest.raises(SystemExit) as excinfo:
            lg.run_p0_verify(stage, tmp_path / "reports", tmp_path / "dl", 4)
        assert excinfo.value.code == 5

    def test_production_verify_rejects_coherent_subset_artifact(self, tmp_path, monkeypatch):
        # Round-5 BLOCKER 1: a COHERENT 100-context assignment — five
        # 100-entry arms plus exactly the matching record set, questions
        # matching the pinned bundle's first 100 — satisfied EVERY pre-fix
        # check and logged a production PASS certifying 2% of the registered
        # 5000-context / 14,996-pair domain (the pre-fix verifier sliced the
        # question authority by the ARTIFACT-declared n_contexts). The
        # verifier now pins the domain from the CALLER and halts rc-5 with
        # an n_contexts authority-mismatch entry.
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _by_persona, _q = self._stage_artifacts(tmp_path, n=100)
        reports = tmp_path / "reports"
        with pytest.raises(SystemExit) as excinfo:
            lg.run_p0_verify(stage, reports, tmp_path / "dl", lg.N_CONTEXTS_FULL)
        assert excinfo.value.code == lg.EXIT_P0_INTEGRITY == 5
        report = json.loads((reports / "p0_integrity_report.json").read_text())
        f = next(x for x in report["failures_first50"] if x.get("field") == "n_contexts_authority")
        assert f["generated"] == 100
        assert f["expected"] == lg.N_CONTEXTS_FULL == 5000

    def test_persona_file_records_container_malformed_routes_rc5(self, tmp_path, monkeypatch):
        # Round-5 BLOCKER 3: a persona file MISSING its "records" key and one
        # whose root is not an object both route through the designed rc-5
        # report (malformed_bucket) — never a raw KeyError/TypeError at
        # ["records"].
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _by_persona, _q = self._stage_artifacts(tmp_path)
        (stage / "persona03_seed42.json").write_text(json.dumps({"metadata": {}}))
        (stage / "persona04_seed42.json").write_text(json.dumps([1, 2, 3]))
        reports = tmp_path / "reports"
        with pytest.raises(SystemExit) as excinfo:
            lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert excinfo.value.code == 5
        report = json.loads((reports / "p0_integrity_report.json").read_text())
        buckets = {
            x.get("persona_bucket")
            for x in report["failures_first50"]
            if x.get("kind") == "malformed_bucket"
        }
        assert {3, 4} <= buckets


class TestP0VerifyCheckpointRestart:
    """Round-4 BLOCKER 4: --p0-verify is durably restartable + emits progress.

    The 14,996-record production verify gets (a) a per-persona durable
    checkpoint keyed on the generation fingerprint + each persona file's
    sha256 (only zero-failure sha-matching entries are reused), flushed
    atomically the moment each persona completes, and (b) one FLUSHED
    ``[p0] persona k/16 ...`` progress line per bucket.
    """

    @staticmethod
    def _counting(monkeypatch, calls: list):
        real = lg.p0_verify_persona_records

        def counting(p, recs, questions, n_contexts):
            calls.append(p)
            return real(p, recs, questions, n_contexts)

        monkeypatch.setattr(lg, "p0_verify_persona_records", counting)

    def test_second_run_reuses_clean_checkpoint(self, tmp_path, monkeypatch, capsys):
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _by_persona, _q = TestP0PromptIntegrityGate()._stage_artifacts(tmp_path)
        reports = tmp_path / "reports"
        calls: list[int] = []
        self._counting(monkeypatch, calls)
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert sorted(calls) == list(range(lg.N_PERSONAS))  # all 16 verified for real
        out1 = capsys.readouterr().out
        assert f"[p0] persona {lg.N_PERSONAS}/{lg.N_PERSONAS}" in out1
        assert "elapsed=" in out1  # the flushed progress line carries elapsed seconds
        ckpt = json.loads((reports / "p0_verify_progress.json").read_text())
        assert ckpt["fingerprint"] == lg.generation_config_fingerprint()["sha256"]
        assert len(ckpt["personas"]) == lg.N_PERSONAS
        assert all(e["n_failures"] == 0 for e in ckpt["personas"].values())
        calls.clear()
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert calls == []  # byte-identical clean files: checkpoint reused, zero re-verifies

    def test_tampered_persona_file_reverifies_and_halts(self, tmp_path, monkeypatch):
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _by_persona, _q = TestP0PromptIntegrityGate()._stage_artifacts(tmp_path)
        reports = tmp_path / "reports"
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        # Tamper persona 2 AFTER the clean checkpoint landed: the file sha256
        # changes, so the checkpoint entry must NOT be reused — re-verify
        # catches the injected prompt and the gate halts rc=5.
        blob = json.loads((stage / "persona02_seed42.json").read_text())
        blob["records"][0]["system_prompt"] += " INJECTED"
        (stage / "persona02_seed42.json").write_text(json.dumps(blob))
        with pytest.raises(SystemExit) as excinfo:
            lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert excinfo.value.code == 5

    def test_stale_fingerprint_restarts_fresh(self, tmp_path, monkeypatch):
        # A checkpoint keyed to a DIFFERENT generation fingerprint is never
        # reused — the verify restarts fresh (all personas re-verified).
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _by_persona, _q = TestP0PromptIntegrityGate()._stage_artifacts(tmp_path)
        reports = tmp_path / "reports"
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        ckpt_path = reports / "p0_verify_progress.json"
        ckpt = json.loads(ckpt_path.read_text())
        ckpt["fingerprint"] = "0" * 64
        ckpt_path.write_text(json.dumps(ckpt))
        calls: list[int] = []
        self._counting(monkeypatch, calls)
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert sorted(calls) == list(range(lg.N_PERSONAS))


class TestP0VerifierSchemaCheckpointKey:
    """Round-6 BLOCKER 1: the P0 checkpoint is keyed on the VERIFIER schema.

    The generation fingerprint keys WHAT was generated; it says nothing about
    which VERIFIER validated it. A checkpoint written by the round-4 verifier
    (isinstance-int schema — records ``n_failures: 0`` for a
    ``context_id: true`` record) must never be reused by the round-5+
    verifier: a missing or differing ``verifier_schema_version`` discards the
    WHOLE checkpoint (exactly like a fingerprint mismatch) and re-verifies
    from scratch.
    """

    def _clean_run(self, tmp_path, monkeypatch):
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, by_persona, _q = TestP0PromptIntegrityGate()._stage_artifacts(tmp_path)
        reports = tmp_path / "reports"
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        return stage, reports, by_persona

    def test_checkpoint_written_with_current_schema_version(self, tmp_path, monkeypatch):
        _stage, reports, _bp = self._clean_run(tmp_path, monkeypatch)
        ckpt = json.loads((reports / "p0_verify_progress.json").read_text())
        assert ckpt["verifier_schema_version"] == lg.P0_VERIFIER_SCHEMA_VERSION == 2

    def test_missing_schema_key_discards_checkpoint(self, tmp_path, monkeypatch):
        # Fixture (a): a checkpoint lacking the schema key (the round-4
        # verifier never wrote one) is NOT reused — every persona is
        # re-verified from scratch.
        stage, reports, _bp = self._clean_run(tmp_path, monkeypatch)
        ckpt_path = reports / "p0_verify_progress.json"
        ckpt = json.loads(ckpt_path.read_text())
        ckpt.pop("verifier_schema_version", None)
        ckpt_path.write_text(json.dumps(ckpt))
        calls: list[int] = []
        TestP0VerifyCheckpointRestart._counting(monkeypatch, calls)
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert sorted(calls) == list(range(lg.N_PERSONAS))

    def test_old_schema_key_discards_checkpoint(self, tmp_path, monkeypatch):
        # Fixture (b): an OLD schema version (1 == the round-4 isinstance-int
        # contract) is treated exactly like a sha mismatch — re-verify.
        stage, reports, _bp = self._clean_run(tmp_path, monkeypatch)
        ckpt_path = reports / "p0_verify_progress.json"
        ckpt = json.loads(ckpt_path.read_text())
        ckpt["verifier_schema_version"] = 1
        ckpt_path.write_text(json.dumps(ckpt))
        calls: list[int] = []
        TestP0VerifyCheckpointRestart._counting(monkeypatch, calls)
        lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert sorted(calls) == list(range(lg.N_PERSONAS))

    def test_round4_checkpoint_cannot_mask_boolean_context_id(self, tmp_path, monkeypatch):
        # Fixture (c), end-to-end: a persona file carrying context_id: true
        # WITH a clean prior checkpoint (correct file sha256, n_failures: 0,
        # NO schema key — exactly what the round-4 verifier persisted for
        # such a record) must be RE-verified and halt rc 5. Pre-fix, the
        # reuse branch reconstructed seen pairs from the cached context ids,
        # the boolean never reached the round-5 type-is-int check, and the
        # gate PASSed — round 4's cache bypassed round 5's schema fix.
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        stage, _bp, _q = TestP0PromptIntegrityGate()._stage_artifacts(tmp_path)
        blob = json.loads((stage / "persona01_seed42.json").read_text())
        assert blob["records"][0]["context_id"] == 1  # the pair True aliases
        blob["records"][0]["context_id"] = True
        (stage / "persona01_seed42.json").write_text(json.dumps(blob))
        reports = tmp_path / "reports"
        reports.mkdir()
        personas = {}
        for p in range(lg.N_PERSONAS):
            f = stage / f"persona{p:02d}_seed42.json"
            recs = json.loads(f.read_text())["records"]
            personas[str(p)] = {
                "file_sha256": lg._sha256_file(f),
                "n_records": len(recs),
                "n_failures": 0,
                "context_ids": sorted(r["context_id"] for r in recs),
                "verified_at": "2026-08-01T00:00:00Z",
            }
        (reports / "p0_verify_progress.json").write_text(
            json.dumps(
                {
                    "fingerprint": lg.generation_config_fingerprint()["sha256"],
                    "personas": personas,
                }
            )
        )
        with pytest.raises(SystemExit) as excinfo:
            lg.run_p0_verify(stage, reports, tmp_path / "dl", 4)
        assert excinfo.value.code == lg.EXIT_P0_INTEGRITY == 5
        report = json.loads((reports / "p0_integrity_report.json").read_text())
        assert any(
            x.get("kind") == "malformed_record" and x.get("persona_bucket") == 1
            for x in report["failures_first50"]
        )


class TestMainP0VerifyForkAuthority:
    """Round-6 BLOCKER 2: main()'s --p0-verify fork threads the domain authority.

    Every prior authority-provenance test drove ``run_p0_verify`` directly, so
    the WIRING in ``main()`` — production pinning ``N_CONTEXTS_FULL``, smoke
    deriving the explicit ``--n-contexts`` override, the ``--p0-verify``
    branch passing that value through — could regress while every direct
    fixture stayed green. These tests drive ``main()`` itself. Offline: the
    banked HF download is monkeypatched; no live calls.
    """

    def test_production_fork_pins_full_domain_and_halts_on_truncated_stage(
        self, tmp_path, monkeypatch
    ):
        # Production main() pins the 5000-context domain; a coherent
        # TRUNCATED (n=100) stage must halt rc 5 through the fork. In
        # production mode main composes report_dir from the REPO's committed
        # eval_results path, so the wrapper redirects ONLY report_dir to
        # tmp_path (tests never write canonical paths) while the REAL
        # run_p0_verify body executes with the threaded authority verbatim.
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        root = tmp_path / "prod_root"
        _stage_p0_artifacts(root / "hf_stage" / "ladder", n=100)
        reports = tmp_path / "reports"
        real = lg.run_p0_verify
        seen: dict = {}

        def redirecting(stage_dir, report_dir, dl_dir, expected_n_contexts):
            seen["args"] = (stage_dir, report_dir, dl_dir, expected_n_contexts)
            return real(stage_dir, reports, dl_dir, expected_n_contexts)

        monkeypatch.setattr(lg, "run_p0_verify", redirecting)
        with pytest.raises(SystemExit) as excinfo:
            lg.main(["--p0-verify", "--out-root", str(root)])
        assert excinfo.value.code == lg.EXIT_P0_INTEGRITY == 5
        stage_dir, report_dir, dl_dir, expected_n = seen["args"]
        assert expected_n == lg.N_CONTEXTS_FULL == 5000  # caller-pinned, not the artifact's 100
        assert stage_dir == root / "hf_stage" / "ladder"
        assert dl_dir == root / "parent_inputs"
        repo_root = pathlib.Path(lg.__file__).resolve().parents[1]
        assert report_dir == repo_root / "eval_results" / "issue_823" / (
            "inconsistent_origin_ladder"
        )
        report = json.loads((reports / "p0_integrity_report.json").read_text())
        f = next(x for x in report["failures_first50"] if x.get("field") == "n_contexts_authority")
        assert f["generated"] == 100 and f["expected"] == 5000

    def test_smoke_fork_threads_explicit_count_and_passes_own_domain(self, tmp_path, monkeypatch):
        # Smoke main() derives the authority from the EXPLICIT --n-contexts
        # override. n=12 (NOT the smoke default 16) is load-bearing: if main
        # ignored the override and pinned the default, the 12-context stage
        # would halt on an authority mismatch — the PASS proves the explicit
        # count threaded through the fork. Smoke report_dir lives under
        # out_root, so no redirect is needed: the full real path runs.
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        root = tmp_path / "smoke_root"
        _stage_p0_artifacts(root / "hf_stage" / "ladder", n=12)
        lg.main(["--p0-verify", "--smoke", "--n-contexts", "12", "--out-root", str(root)])
        ckpt_path = root / "eval_results" / "inconsistent_origin_ladder" / "p0_verify_progress.json"
        ckpt = json.loads(ckpt_path.read_text())
        assert len(ckpt["personas"]) == lg.N_PERSONAS
        assert all(e["n_failures"] == 0 for e in ckpt["personas"].values())
        assert not (
            root / "eval_results" / "inconsistent_origin_ladder" / "p0_integrity_report.json"
        ).exists()


# ── Round-4 BLOCKER 3: offline cap-ladder state-machine pin over main() ──────


class TestMainCapLadderStateMachine:
    """Offline end-to-end pin of the 4096 -> 8192 cap-ladder state machine.

    Drives ``lg.main()`` in smoke mode with FAKES at the transport boundary
    ONLY (Batch dispatch, dispatcher state.json join, HF upload, pinned
    HF download); the fingerprint gate, cell trigger, pooled regen wiring,
    record builder, gen-time P0 gate, digest and sentinel all run REAL.
    Covers, per the round-4 brief: first-wave per-cell counts -> exactly ONE
    fake regen dispatch at 8192 -> result splice -> per-row wave/cap ->
    final denominators -> residual `cap-hit>2%` labelling. No live API
    calls anywhere.
    """

    # (persona, context) rows capped (stop_reason max_tokens) on the first
    # wave at n=16: fires cells {k=8,p=5}, {k=16,p=5}, {k=4,p=3}, {k=8,p=3},
    # {k=16,p=3} — the regen union is exactly the two pair rows.
    CAPPED_FIRST_WAVE = frozenset({(5, 5), (3, 3)})
    # Residual: (3,3) comes back capped AGAIN at 8192 — labelled, never
    # re-cascaded (exactly one regen round).
    STILL_CAPPED_AT_REGEN = frozenset({(3, 3)})

    @staticmethod
    def _pi(item_id: str) -> tuple[int, int]:
        p_s, c_s = item_id.split("_")
        return int(p_s[1:]), int(c_s[1:])

    def _run(self, tmp_path, monkeypatch):
        b2, cv = _write_frozen_fixture(tmp_path, permute=False)
        TestFrozenQuestionLoader._patch(monkeypatch, b2, cv)
        root = tmp_path / "root"
        dispatch_log: list[tuple[str, list[str], int]] = []
        upload_calls: list[dict] = []

        async def fake_dispatch(items, checkpoint_dir, max_tokens, poll_interval):
            # Run the REAL fingerprint gate so gen_config.json is persisted
            # and checkpoint_max_tokens() reads real checkpoint metadata.
            lg.check_or_persist_gen_config(checkpoint_dir, max_tokens)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ids = sorted(it.item_id for it in items)
            dispatch_log.append((checkpoint_dir.name, ids, max_tokens))
            capped_set = (
                self.CAPPED_FIRST_WAVE
                if checkpoint_dir.name == "batches"
                else self.STILL_CAPPED_AT_REGEN
            )
            return {
                it.item_id: DispatchResult(
                    item_id=it.item_id,
                    result="an in-character answer",
                    error=False,
                    category="ok",
                    stop_reason=(
                        "max_tokens" if self._pi(it.item_id) in capped_set else "end_turn"
                    ),
                )
                for it in items
            }

        def fake_load_batch_meta(checkpoint_dir):
            # The real join reads the dispatcher's state.json, which the fake
            # dispatch never writes — key the fake on the recorded dispatches.
            for name, ids, _mt in dispatch_log:
                if name == checkpoint_dir.name:
                    return {iid: _mk_meta(batch_id=f"b_{name}") for iid in ids}
            raise AssertionError(f"unexpected load_batch_meta({checkpoint_dir})")

        def fake_upload(**kwargs):
            upload_calls.append(kwargs)
            return f"{kwargs['repo_id']}/{kwargs['path_in_repo']}"

        monkeypatch.setattr(lg, "_dispatch", fake_dispatch)
        monkeypatch.setattr(lg, "load_batch_meta", fake_load_batch_meta)
        monkeypatch.setattr(lg, "_upload_folder_filtered", fake_upload)
        lg.main(
            [
                "--smoke",
                "--n-contexts",
                "16",
                "--out-root",
                str(root),
                "--poll-interval",
                "0.1",
            ]
        )
        return root, dispatch_log, upload_calls

    def test_exactly_one_pooled_regen_dispatch_no_cascade(self, tmp_path, monkeypatch):
        _root, log, _up = self._run(tmp_path, monkeypatch)
        assert [name for name, _ids, _mt in log] == ["batches", "regen_pooled"]
        first, regen = log
        assert first[2] == lg.GEN_MAX_TOKENS
        assert regen[2] == lg.REGEN_MAX_TOKENS
        # ONE pooled dispatch covers the union of BOTH triggered personas'
        # over-cap rows; the residual (3,3) row is NOT re-cascaded (no third
        # dispatch exists).
        assert regen[1] == sorted([lg.make_item_id(3, 3), lg.make_item_id(5, 5)])

    def test_result_splice_and_per_row_wave_cap(self, tmp_path, monkeypatch):
        root, _log, _up = self._run(tmp_path, monkeypatch)
        stage = root / "hf_stage" / "ladder"

        def recs(p):
            blob = json.loads((stage / f"persona{p:02d}_seed42.json").read_text())
            return {r["context_id"]: r for r in blob["records"]}

        r55 = recs(5)[5]  # regenerated AND resolved at 8192
        assert r55["gen_wave"] == lg.GEN_WAVE_REGEN and r55["regen"] is True
        assert r55["max_tokens"] == lg.REGEN_MAX_TOKENS
        assert r55["cap_hit"] is False
        assert r55["batch_id"] == "b_regen_pooled"  # splice: the regen result won
        r33 = recs(3)[3]  # regenerated and STILL capped (the residual row)
        assert r33["gen_wave"] == lg.GEN_WAVE_REGEN and r33["regen"] is True
        assert r33["max_tokens"] == lg.REGEN_MAX_TOKENS
        assert r33["cap_hit"] is True
        # Untouched rows keep first-wave provenance end to end.
        assert all(
            r["gen_wave"] == lg.GEN_WAVE_FIRST
            and r["max_tokens"] == lg.GEN_MAX_TOKENS
            and r["regen"] is False
            and r["batch_id"] == "b_batches"
            for r in recs(0).values()
        )

    def test_digest_denominators_and_residual_labels(self, tmp_path, monkeypatch):
        root, _log, _up = self._run(tmp_path, monkeypatch)
        digest = json.loads(
            (root / "eval_results" / "inconsistent_origin_ladder" / "gen_digest.json").read_text()
        )
        assert set(digest["regen_cells_triggered"]) == {
            "k=8,p=5",
            "k=16,p=5",
            "k=4,p=3",
            "k=8,p=3",
            "k=16,p=3",
        }
        assert digest["regen_cells_triggered"]["k=16,p=5"]["n_rows"] == 1
        assert digest["regen_pairs_by_persona"] == {"3": 1, "5": 1}
        # FINAL denominator: the post-regen RE-MEASURE on final records.
        frac = digest["cap_hit_fraction_by_arm_persona"]
        assert frac["16"]["3"] == 1.0 and frac["16"]["5"] == 0.0
        assert frac["8"]["3"] == 0.5 and frac["4"]["3"] == 0.25
        residual = {
            (c["k"], c["persona"]): c for c in digest["cap_hit_cells_over_threshold_post_regen"]
        }
        assert set(residual) == {(4, 3), (8, 3), (16, 3)}
        assert all(c["label"] == "cap-hit>2%" for c in residual.values())

    def test_sentinel_upload_and_gen_time_p0(self, tmp_path, monkeypatch, capsys):
        root, _log, uploads = self._run(tmp_path, monkeypatch)
        sent = json.loads((root / "hf_stage" / "ladder" / "_gen_complete.json").read_text())
        assert sent["complete"] is True
        live_fp = lg.generation_config_fingerprint()["sha256"]
        assert sent["generation_config_fingerprint"]["sha256"] == live_fp
        assert len(uploads) == 1
        assert uploads[0]["path_in_repo"] == f"{lg.HF_PREFIX}_smoke/raw_completions/ladder"
        assert uploads[0]["repo_id"] == lg.DATA_REPO
        # The gen-time P0 gate ran over every bucket (flushed progress lines).
        out = capsys.readouterr().out
        assert f"[p0] persona {lg.N_PERSONAS}/{lg.N_PERSONAS}" in out
