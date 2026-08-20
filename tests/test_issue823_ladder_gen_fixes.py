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
import re

import pytest

from explore_persona_space.llm.api_dispatch import DispatchResult
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
