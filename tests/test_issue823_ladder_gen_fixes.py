"""Regression pins for scripts/issue823_ladder_gen.py (#823 P-Gen fix round).

Pins the registered nested-assignment invariants (four corruptions that must
each raise, incl. the two caught ONLY by a single assert layer — the layering
is deliberately redundant and must not be weakened), the record builder's
fail-loud provenance asserts (incl. the FIX 3 ``batch_org`` line), the FIX 2
per-sub-batch ``harvested_at`` join, the FIX 4 fresh-redrive-dir numbering,
the FIX 1 canonical-repo upload gate, and the frozen-question loader's
monotone-unique context order.

Offline by design: the banked HF download is monkeypatched to tmp_path
fixtures (signature-conformant def fake, no network); no eval_results/
fixture reads (no sparse_cones entry needed).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re

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
            max_tokens_by_item={iid: lg.GEN_MAX_TOKENS},
            regen_items=set(),
        )

    def test_complete_provenance_passes(self):
        rec = self._build(_mk_meta())[0][0]
        assert rec["harvested_at"] == "2026-08-19T00:00:00Z"
        assert rec["batch_org"] == "org-a"
        assert rec["batch_submitted_at"] == "2026-08-18T23:00:00Z"

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
            max_tokens_by_item={iid: lg.GEN_MAX_TOKENS for iid in metas},
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
        rd = tmp_path / "redrive1"
        rd.mkdir()
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
        lg._merge_stale_redrives(tmp_path, items_by_id, results, batch_meta, poll_interval=1.0)
        assert calls == [(["r1"], rd)]
        # r1's already-paid success is merged back: only r2 remains pending.
        assert lg.transport_class_ids(results) == ["r2"]
        assert batch_meta["r1"]["batch_id"] == "b_stale1"
        assert batch_meta["r0"]["batch_id"] == "b_main"

    def test_stateless_dir_skipped_without_dispatch(self, tmp_path, monkeypatch):
        # Dir created but the dispatcher never wrote state.json: nothing was
        # submitted there, so there is nothing to resume (and no API call).
        (tmp_path / "redrive1").mkdir()

        def boom(*args, **kwargs):
            raise AssertionError("must not dispatch a stateless stale dir")

        monkeypatch.setattr(lg, "_dispatch", boom)
        results: dict = {}
        lg._merge_stale_redrives(tmp_path, {}, results, {}, poll_interval=1.0)
        assert results == {}

    def test_foreign_checkpoint_ids_raise(self, tmp_path):
        rd = tmp_path / "redrive1"
        rd.mkdir()
        (rd / "state.json").write_text(json.dumps({"cid_to_item": {"c0": "not_registered"}}))
        with pytest.raises(RuntimeError, match="registered item set"):
            lg._merge_stale_redrives(tmp_path, {}, {}, {}, poll_interval=1.0)


# ── FIX D (follow-up round): pooled cap-hit regen dispatch ───────────────────


class TestPooledRegenDispatch:
    def test_one_dispatch_covers_union_of_triggered_personas(self, tmp_path, monkeypatch):
        # Several personas over the per-persona threshold => exactly ONE
        # regeneration dispatch, whose submitted id set is the UNION of the
        # triggered personas' over-cap rows (the pre-fix path dispatched one
        # SERIAL batch per persona, each waiting out its own Batch window).
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
