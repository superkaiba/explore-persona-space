"""Committed behavioral tests for scripts/issue823_ladder_ext_gen.py (unit 2,
#823 follow-up `origin-ladder-more-contexts`, plan v17 sections 4.2 / 4.3 / 7).

Covers, per the unit scope:
  * the registered selection rule — prefix ordered-equality PASS, the
    registered set-equality fallback (permutation reported, run proceeds),
    the rc-6 drift halts (ctx0 mismatch; neither ordered nor set equality),
    the rc-7 depth-bound / stream-exhaustion halts, raw-string disjointness
    + within-extension dedup;
  * sampling-manifest persistence + fingerprint-gated resume (round-trip;
    rc-11 halts on field drift and on sha mismatch);
  * assignment arithmetic (arms {1, 16}) + the banked 5,000-row
    prefix-equality assert (rc 10);
  * banked-roster byte-assert (rc 10) + roster-derived system prompts;
  * pair-count arithmetic (production 43,000 / 40,313 / 2,687 / 83,313;
    smoke 16 -> 31);
  * Gate A band arithmetic + halt routing (rc 8 survival, rc 9 cap-hit,
    boundary semantics, shared-pair contexts, rc-8 precedence, the
    unresolved-transport fail-loud);
  * ext-scoped cap-hit / regen cell selection (ext-only denominators,
    strictly-greater trigger, cross-cell union dedup);
  * the stage transport-residue rc-3 halt (seams monkeypatched — no dispatch);
  * record builder arms/provenance asserts; the >9.5 MB line-shard round-trip;
  * mechanical source-parity pins for the restated ``_first_user_turn`` +
    ``EXPECTED_CTX0_PROMPT`` vs scripts/issue779_ffc_n10k_generate_capture.py
    (ast-level, no import of that module's torch-heavy graph).

Offline by design: no network, no live Batch calls (dispatch seams are
monkeypatched), CPU-only, flat synthetic fixtures (short innocuous strings —
no LMSYS text), no eval_results/ fixture reads (no sparse_cones entry needed).
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

import pytest

from explore_persona_space.llm.api_dispatch import DispatchResult
from scripts import issue823_ladder_ext_gen as xg
from scripts import issue823_ladder_gen as lg

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CTX0 = xg.EXPECTED_CTX0_PROMPT


def _row(prompt: str | None) -> dict:
    if prompt is None:
        return {"conversation": []}
    return {"conversation": [{"content": prompt}]}


def _stream(prompts: list[str | None]):
    return iter(_row(p) for p in prompts)


def _banked4() -> list[str]:
    return [CTX0, "banked q1", "banked q2", "banked q3"]


def _ok(item_id: str, text: str = "an answer", stop: str = "end_turn") -> DispatchResult:
    return DispatchResult(
        item_id=item_id, result=text, error=False, category="ok", stop_reason=stop
    )


def _refusal(item_id: str) -> DispatchResult:
    return DispatchResult(
        item_id=item_id,
        result="",
        error=True,
        reason="refusal",
        category="empty_response",
        stop_reason="refusal",
    )


def _transport(item_id: str) -> DispatchResult:
    return DispatchResult(
        item_id=item_id,
        result=None,
        error=True,
        reason="rate_limited",
        category=lg.RESULT_TRANSPORT,
    )


# ── Source-parity pins (restated predicate + ctx0 constant) ──────────────────


def _fn_body_dump(path: pathlib.Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # drop the docstring — the logic is the pin
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"{name} not found in {path}")


def _module_constant(path: pathlib.Path, name: str):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class TestSourceParityPins:
    N10K = REPO_ROOT / "scripts" / "issue779_ffc_n10k_generate_capture.py"
    EXT = REPO_ROOT / "scripts" / "issue823_ladder_ext_gen.py"

    def test_first_user_turn_restated_verbatim(self):
        assert _fn_body_dump(self.N10K, "_first_user_turn") == _fn_body_dump(
            self.EXT, "_first_user_turn"
        )

    def test_expected_ctx0_prompt_matches_n10k_constant(self):
        assert _module_constant(self.N10K, "EXPECTED_CTX0_PROMPT") == CTX0

    def test_designed_abort_rcs_distinct_and_documented(self):
        rcs = {
            xg.EXIT_TRANSPORT_RESIDUE: "3",
            lg.EXIT_CONFIG_MISMATCH: "4",
            xg.EXIT_STREAM_DRIFT: "6",
            xg.EXIT_STREAM_EXHAUSTED: "7",
            xg.EXIT_GATE_A_SURVIVAL: "8",
            xg.EXIT_GATE_A_CAP_HIT: "9",
            xg.EXIT_BANKED_PARITY: "10",
            xg.EXIT_MANIFEST_MISMATCH: "11",
        }
        assert len(rcs) == 8, "designed-abort rcs collide"
        doc = xg.__doc__
        for rc in rcs.values():
            assert f" {rc} |" in doc, f"rc {rc} missing from the module docstring table"


# ── Selection rule ────────────────────────────────────────────────────────────


class TestSelection:
    def test_ordered_prefix_and_disjoint_extension(self, tmp_path):
        banked = _banked4()
        stream = _stream(
            [
                *banked,  # phase 1 reproduces the banked order
                None,  # empty conversation — predicate skips, position still advances
                "banked q1",  # in the banked set — excluded
                "ext a",
                "ext a",  # repeat — within-extension dedup
                "  ext b  ",  # predicate strips
                "ext c",
            ]
        )
        sel = xg.select_extension_contexts(
            banked, 3, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
        )
        assert sel["ext_prompts"] == ["ext a", "ext b", "ext c"]
        assert sel["prefix_report"]["ordered_equal"] is True
        assert sel["prefix_report"]["set_equal"] is True
        assert sel["prefix_report"]["n_position_mismatches"] == 0
        # positions: banked rows at 0-3, None at 4, dup at 5, ext a at 6,
        # repeat at 7, ext b at 8, ext c at 9.
        assert sel["ext_positions"] == [6, 8, 9]
        assert sel["last_stream_pos"] == 9

    def test_set_equality_fallback_reports_permutation_and_proceeds(self, tmp_path):
        banked = _banked4()
        permuted = [CTX0, "banked q2", "banked q1", "banked q3"]  # ctx0 stays first
        stream = _stream([*permuted, "ext a", "ext b"])
        sel = xg.select_extension_contexts(
            banked, 2, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
        )
        rep = sel["prefix_report"]
        assert rep["ordered_equal"] is False
        assert rep["set_equal"] is True
        assert rep["n_position_mismatches"] == 2
        assert rep["mismatch_positions_first20"] == [1, 2]
        assert sel["ext_prompts"] == ["ext a", "ext b"]

    def test_prefix_drift_halts_rc6_without_prompt_text(self, tmp_path):
        banked = _banked4()
        stream = _stream([CTX0, "banked q1", "NOVEL drift row", "banked q3", "ext a"])
        with pytest.raises(SystemExit) as ei:
            xg.select_extension_contexts(
                banked, 1, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
            )
        assert ei.value.code == xg.EXIT_STREAM_DRIFT
        report = json.loads((tmp_path / "ext_selection_drift_report.json").read_text())
        assert report["reason"] == "prefix_reproduction_failed"
        assert report["n_stream_only"] == 1 and report["n_banked_only"] == 1
        # Real-corpus hygiene: sha digests only, never raw prompt text.
        assert "NOVEL drift row" not in json.dumps(report)

    def test_ctx0_mismatch_halts_rc6(self, tmp_path):
        stream = _stream(["not the expected ctx0", "banked q1"])
        with pytest.raises(SystemExit) as ei:
            xg.select_extension_contexts(
                _banked4(), 1, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
            )
        assert ei.value.code == xg.EXIT_STREAM_DRIFT
        report = json.loads((tmp_path / "ext_selection_drift_report.json").read_text())
        assert report["reason"] == "ctx0_mismatch"

    def test_depth_bound_halts_rc7(self, tmp_path):
        banked = _banked4()
        # Enough rows exist, but the bound bites first (pos > 5 forbidden).
        stream = _stream([*banked, "ext a", "ext b", "ext c", "ext d"])
        with pytest.raises(SystemExit) as ei:
            xg.select_extension_contexts(
                banked, 4, max_stream_pos=5, eval_dir=tmp_path, stream_iter=stream
            )
        assert ei.value.code == xg.EXIT_STREAM_EXHAUSTED
        report = json.loads((tmp_path / "ext_stream_exhausted_report.json").read_text())
        assert report["reason"] == "max_stream_pos_bound_hit"
        assert report["n_ext_collected"] < 4

    def test_stream_exhaustion_halts_rc7(self, tmp_path):
        banked = _banked4()
        stream = _stream([*banked, "ext a"])  # runs dry before 3 are collected
        with pytest.raises(SystemExit) as ei:
            xg.select_extension_contexts(
                banked, 3, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
            )
        assert ei.value.code == xg.EXIT_STREAM_EXHAUSTED
        report = json.loads((tmp_path / "ext_stream_exhausted_report.json").read_text())
        assert report["reason"] == "stream_exhausted"
        assert report["n_ext_collected"] == 1


# ── Sampling manifest: persistence + fingerprint-gated resume ────────────────


class TestManifestResume:
    def _selection(self, tmp_path) -> tuple[dict, dict]:
        banked = _banked4()
        stream = _stream([*banked, "ext a", "ext b"])
        sel = xg.select_extension_contexts(
            banked, 2, max_stream_pos=100, eval_dir=tmp_path, stream_iter=stream
        )
        fields = xg.selection_fingerprint_fields(2, 100)
        return sel, fields

    def test_round_trip(self, tmp_path):
        sel, fields = self._selection(tmp_path)
        stage = tmp_path / "stage"
        xg.write_sampling_manifest(stage, sel, fields, {"unit": "test"})
        loaded = xg.load_selection_if_persisted(stage, fields, tmp_path)
        assert loaded is not None
        assert loaded["ext_prompts"] == sel["ext_prompts"]
        assert loaded["ext_positions"] == sel["ext_positions"]
        assert loaded["last_stream_pos"] == sel["last_stream_pos"]
        assert loaded["prefix_report"] == sel["prefix_report"]

    def test_missing_manifest_returns_none(self, tmp_path):
        fields = xg.selection_fingerprint_fields(2, 100)
        assert xg.load_selection_if_persisted(tmp_path / "stage", fields, tmp_path) is None

    def test_fingerprint_field_drift_halts_rc11(self, tmp_path):
        sel, fields = self._selection(tmp_path)
        stage = tmp_path / "stage"
        xg.write_sampling_manifest(stage, sel, fields, {})
        live = xg.selection_fingerprint_fields(3, 100)  # n_ext changed across resume
        with pytest.raises(SystemExit) as ei:
            xg.load_selection_if_persisted(stage, live, tmp_path)
        assert ei.value.code == xg.EXIT_MANIFEST_MISMATCH
        report = json.loads((tmp_path / "ext_manifest_mismatch_report.json").read_text())
        assert report["reason"] == "fingerprint_fields_mismatch"
        assert report["differing_fields"] == ["n_ext"]

    def test_prompt_sha_mismatch_halts_rc11(self, tmp_path):
        sel, fields = self._selection(tmp_path)
        stage = tmp_path / "stage"
        xg.write_sampling_manifest(stage, sel, fields, {})
        pf = stage / xg.PROMPTS_FILENAME
        rows = [json.loads(line) for line in pf.read_text().splitlines()]
        rows[0]["prompt"] = "tampered"
        pf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(SystemExit) as ei:
            xg.load_selection_if_persisted(stage, fields, tmp_path)
        assert ei.value.code == xg.EXIT_MANIFEST_MISMATCH
        report = json.loads((tmp_path / "ext_manifest_mismatch_report.json").read_text())
        assert report["reason"] == "prompts_file_sha_mismatch"


# ── Assignment + banked parity ───────────────────────────────────────────────


def _banked_assignment_obj(n: int) -> dict:
    return {
        "arms": {str(k): [i % k for i in range(n)] for k in (1, 2, 4, 8, 16)},
        "n_contexts": n,
    }


def _roster_obj() -> dict:
    return {
        "template": lg.PERSONA_TEMPLATE,
        "personas": [
            {"idx": p, "name": name, "card": card} for p, (name, card) in enumerate(lg.PERSONAS)
        ],
    }


class TestAssignmentAndParity:
    def test_build_ext_assignment_full_grain(self):
        a = xg.build_ext_assignment(xg.N_TOTAL_FULL)
        assert set(a) == {1, 16}
        assert a[1] == [0] * xg.N_TOTAL_FULL
        assert a[16] == [i % 16 for i in range(xg.N_TOTAL_FULL)]

    def test_prefix_equality_passes_on_banked(self, tmp_path):
        a = xg.build_ext_assignment(12)
        xg.assert_assignment_prefix(a, _banked_assignment_obj(8), tmp_path)  # no raise

    def test_prefix_mismatch_halts_rc10(self, tmp_path):
        a = xg.build_ext_assignment(12)
        banked = _banked_assignment_obj(8)
        banked["arms"]["16"][3] = 99
        with pytest.raises(SystemExit) as ei:
            xg.assert_assignment_prefix(a, banked, tmp_path)
        assert ei.value.code == xg.EXIT_BANKED_PARITY
        report = json.loads((tmp_path / "ext_banked_parity_report.json").read_text())
        assert report["reason"] == "assignment_prefix_mismatch"
        assert report["mismatched_arms"] == ["16"]

    def test_roster_parity_passes_and_rebuilds_system(self, tmp_path):
        roster = _roster_obj()
        xg.assert_banked_roster_parity(roster, tmp_path)  # no raise
        for p in range(lg.N_PERSONAS):
            assert xg.persona_system_from_roster(roster, p) == lg.persona_system(p)

    @pytest.mark.parametrize(
        "mutate,field",
        [
            (lambda r: r.update(template=r["template"] + " EXTRA"), "template"),
            (lambda r: r["personas"][3].update(card="a different card"), "personas"),
        ],
    )
    def test_roster_mismatch_halts_rc10(self, tmp_path, mutate, field):
        roster = _roster_obj()
        mutate(roster)
        with pytest.raises(SystemExit) as ei:
            xg.assert_banked_roster_parity(roster, tmp_path)
        assert ei.value.code == xg.EXIT_BANKED_PARITY
        report = json.loads((tmp_path / "ext_banked_parity_report.json").read_text())
        assert field in report["mismatched_fields"]


# ── Pair-count arithmetic ────────────────────────────────────────────────────


class TestPairArithmetic:
    def test_production_counts(self):
        ids = xg.ext_context_ids(xg.N_PREFIX, xg.N_TOTAL_FULL)
        n_mult16 = sum(1 for i in ids if i % 16 == 0)
        assert len(ids) == 43_000
        assert n_mult16 == 2_687
        assert xg.registered_ext_pair_count(xg.N_PREFIX, xg.N_TOTAL_FULL) == 83_313
        assert 43_000 + (43_000 - 2_687) == 83_313  # 43,000 k1 + 40,313 nonzero-residue k16

    def test_production_pair_set(self):
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_TOTAL_FULL)
        assert len(pairs) == xg.REGISTERED_EXT_PAIRS
        assert all(xg.N_PREFIX <= i < xg.N_TOTAL_FULL for i, _ in pairs)
        assert sum(1 for _, p in pairs if p == 0) == 43_000  # k1 rows (incl. shared)

    def test_smoke_16_contexts_is_31_pairs(self):
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + 16)
        assert len(pairs) == 31  # 16 k1 + 15 nonzero-residue k16 (5008 shares persona 0)
        assert sum(1 for i, p in pairs if i == 5008) == 1  # the shared pair appears ONCE
        assert (5008, 0) in pairs

    def test_items_sorted_and_roster_derived(self):
        roster = _roster_obj()
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + 4)
        questions = {i: f"question {i}" for i, _ in pairs}
        items = xg.build_ext_items(questions, pairs, roster)
        assert [it.item_id for it in items] == [lg.make_item_id(p, i) for i, p in sorted(pairs)]
        first = items[0]
        i0, p0 = sorted(pairs)[0]
        assert first.payload["messages"] == [{"role": "user", "content": f"question {i0}"}]
        assert first.payload["system"] == lg.persona_system(p0)


# ── Gate A ───────────────────────────────────────────────────────────────────


def _gate_fixture(n_ext: int, pilot_n: int):
    """Ext pairs for contexts 5000..5000+n_ext-1 with the first pilot_n pairs
    as the pilot; returns (pairs, pilot_pairs)."""
    pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + n_ext)
    return pairs, sorted(pairs)[:pilot_n]


def _all_ok_results(pilot_pairs) -> dict[str, DispatchResult]:
    return {lg.make_item_id(p, i): _ok(lg.make_item_id(p, i)) for i, p in pilot_pairs}


class TestGateA:
    def test_pass_report_shape(self, tmp_path):
        pairs, pilot = _gate_fixture(8, 6)  # contexts 5000-5002 fully judged
        report = xg.evaluate_gate_a(_all_ok_results(pilot), pilot, pairs)
        assert report["n_judged_contexts"] == 3
        assert report["survival"] == 1.0 and report["survival_pass"]
        assert report["cap_hit_fraction"] == 0.0 and report["cap_hit_pass"]
        xg.enforce_gate_a(report, tmp_path)  # no raise
        assert (tmp_path / "gateA_report_ext.json").is_file()

    def test_refused_context_drops_and_rc8_below_floor(self, tmp_path):
        pairs, pilot = _gate_fixture(8, 6)
        results = _all_ok_results(pilot)
        # Refuse ONE arm row of context 5001 — the 2-arm read drops the context.
        results[lg.make_item_id(5001 % 16, 5001)] = _refusal(lg.make_item_id(5001 % 16, 5001))
        report = xg.evaluate_gate_a(results, pilot, pairs)
        assert report["n_survived_contexts"] == 2
        assert report["dropped_context_ids"] == [5001]
        assert report["survival"] == pytest.approx(2 / 3)
        assert not report["survival_pass"]
        assert report["per_persona"][str(5001 % 16)]["refusal"] == 1
        assert report["refusal_ranking"][0]["persona"] == 5001 % 16
        with pytest.raises(SystemExit) as ei:
            xg.enforce_gate_a(report, tmp_path)
        assert ei.value.code == xg.EXIT_GATE_A_SURVIVAL

    def test_survival_exactly_at_floor_passes(self, tmp_path):
        # 40 pilot pairs = 20 judged contexts; 3 refusals -> 17/20 = 0.85.
        pairs, pilot = _gate_fixture(20, 40)
        results = _all_ok_results(pilot)
        for ctx in (5000, 5001, 5002):
            iid = lg.make_item_id(ctx % 16, ctx)
            results[iid] = _refusal(iid)
        report = xg.evaluate_gate_a(results, pilot, pairs)
        assert report["survival"] == pytest.approx(0.85)
        assert report["survival_pass"]  # >= floor passes; only < floor halts
        xg.enforce_gate_a(report, tmp_path)

    def test_cap_hit_at_band_halts_rc9(self, tmp_path):
        # 26 ext contexts -> 26 k1 + 24 nonzero-residue k16 = exactly 50 pairs
        # (5008 and 5024 share persona 0); 1 cap-hit of 50 rows -> 0.02 >= max
        # halts (plan: "cap-hit >= 2%").
        pairs, pilot = _gate_fixture(26, 50)
        assert len(pilot) == 50
        results = _all_ok_results(pilot)
        i0, p0 = pilot[0]
        results[lg.make_item_id(p0, i0)] = _ok(lg.make_item_id(p0, i0), stop="max_tokens")
        report = xg.evaluate_gate_a(results, pilot, pairs)
        assert report["survival_pass"]  # a cap-hit row still classifies ok
        assert report["cap_hit_fraction"] == pytest.approx(0.02)
        assert not report["cap_hit_pass"]
        with pytest.raises(SystemExit) as ei:
            xg.enforce_gate_a(report, tmp_path)
        assert ei.value.code == xg.EXIT_GATE_A_CAP_HIT

    def test_survival_takes_precedence_over_cap(self, tmp_path):
        pairs, pilot = _gate_fixture(8, 6)
        results = _all_ok_results(pilot)
        for i, p in pilot:
            results[lg.make_item_id(p, i)] = _refusal(lg.make_item_id(p, i))
        report = xg.evaluate_gate_a(results, pilot, pairs)
        assert not report["survival_pass"]
        with pytest.raises(SystemExit) as ei:
            xg.enforce_gate_a(report, tmp_path)
        assert ei.value.code == xg.EXIT_GATE_A_SURVIVAL

    def test_shared_pair_context_judged_by_its_single_row(self, tmp_path):
        # Context 5008 (i%16==0) has ONE pair serving both arms.
        pairs, _ = _gate_fixture(16, 31)
        pilot = sorted(pairs)  # everything
        results = _all_ok_results(pilot)
        report = xg.evaluate_gate_a(results, pilot, pairs)
        assert report["n_judged_contexts"] == 16
        results[lg.make_item_id(0, 5008)] = _refusal(lg.make_item_id(0, 5008))
        report2 = xg.evaluate_gate_a(results, pilot, pairs)
        assert report2["dropped_context_ids"] == [5008]

    def test_unresolved_transport_row_is_fail_loud(self):
        pairs, pilot = _gate_fixture(8, 6)
        results = _all_ok_results(pilot)
        i0, p0 = pilot[0]
        results[lg.make_item_id(p0, i0)] = _transport(lg.make_item_id(p0, i0))
        with pytest.raises(AssertionError, match="transport"):
            xg.evaluate_gate_a(results, pilot, pairs)

    def test_pilot_covering_no_full_context_is_fail_loud(self):
        pairs, pilot = _gate_fixture(8, 1)  # one pair of a 2-pair context
        with pytest.raises(AssertionError, match="pilot too small"):
            xg.evaluate_gate_a(_all_ok_results(pilot), pilot, pairs)


# ── Cap-hit / regen cell selection ───────────────────────────────────────────


class TestRegenCells:
    def test_ext_only_denominators_and_strict_trigger(self):
        n_total = xg.N_PREFIX + 50  # ext ids 5000..5049; k1 cell = 50 rows
        a = xg.build_ext_assignment(n_total)
        ids = xg.ext_context_ids(xg.N_PREFIX, n_total)
        stop = {}
        # Exactly 1/50 = 2% in the k1 cell — strictly-greater trigger must NOT fire.
        stop[lg.make_item_id(0, 5008)] = "max_tokens"  # 5008 % 16 == 0: shared pair
        triggered, regen = xg.ext_cells_over_cap_threshold(stop, a, ids)
        # k1 cell (1,0): 1/50 == 0.02, not > 0.02.  k16 cell (16,0): contexts
        # {5008, 5024, 5040} -> 1/3 > 0.02: fires with the EXT-only denominator.
        assert "k=1,p=0" not in triggered
        assert triggered["k=16,p=0"]["n_rows"] == 3
        assert regen == {0: [5008]}

    def test_union_dedup_across_cells(self):
        n_total = xg.N_PREFIX + 50
        a = xg.build_ext_assignment(n_total)
        ids = xg.ext_context_ids(xg.N_PREFIX, n_total)
        stop = {
            lg.make_item_id(0, 5008): "max_tokens",  # in BOTH the k1 and k16,p=0 cells
            lg.make_item_id(0, 5010): "max_tokens",  # k1-only (5010 % 16 == 10)
        }
        triggered, regen = xg.ext_cells_over_cap_threshold(stop, a, ids)
        assert "k=1,p=0" in triggered  # 2/50 = 4% > 2%
        assert "k=16,p=0" in triggered  # 1/3 > 2%
        # 5008 sits in both triggered cells but regenerates ONCE.
        assert regen[0] == [5008, 5010]

    def test_k16_persona_cell(self):
        n_total = xg.N_PREFIX + 32  # two full mod-16 cycles: every k16 cell has 2 rows
        a = xg.build_ext_assignment(n_total)
        ids = xg.ext_context_ids(xg.N_PREFIX, n_total)
        ctxs_p3 = [i for i in ids if i % 16 == 3]
        stop = {lg.make_item_id(3, i): "max_tokens" for i in ctxs_p3}
        triggered, regen = xg.ext_cells_over_cap_threshold(stop, a, ids)
        assert triggered["k=16,p=3"]["fraction"] == 1.0
        assert regen == {3: ctxs_p3}


# ── Stage dispatch: transport-residue rc 3 ───────────────────────────────────


class TestDispatchStageHalts:
    def test_transport_residue_halts_rc3(self, tmp_path, monkeypatch):
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + 2)
        items = xg.build_ext_items({i: f"q{i}" for i, _ in pairs}, pairs, _roster_obj())
        calls = []

        def fake_dispatch(stage_items, checkpoint_dir, max_tokens, poll_interval):
            calls.append((checkpoint_dir.name, len(stage_items), max_tokens))
            return {it.item_id: _transport(it.item_id) for it in stage_items}

        monkeypatch.setattr(xg, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(xg, "_load_stage_batch_meta", lambda d: {})
        monkeypatch.setattr(xg, "_stage_cap", lambda d: lg.GEN_MAX_TOKENS)
        with pytest.raises(SystemExit) as ei:
            xg.run_dispatch_stage(
                "pilot",
                tmp_path / "pilot",
                items,
                {},
                {},
                {},
                {},
                0.01,
                tmp_path / "eval",
                {"unit": "test"},
            )
        assert ei.value.code == xg.EXIT_TRANSPORT_RESIDUE
        # main dispatch + the bounded fresh re-drive rounds, all at the base cap
        assert [c[0] for c in calls] == ["batches", "redrive1", "redrive2"]
        assert all(c[2] == lg.GEN_MAX_TOKENS for c in calls)
        report = json.loads((tmp_path / "eval" / "gen_digest_ext.json").read_text())
        assert report["incomplete"] is True
        assert report["reason"] == "transport_class_rows_remaining_pilot"

    def test_empty_stage_is_a_clean_skip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            xg, "_dispatch_stage", lambda *a, **k: pytest.fail("dispatched on empty stage")
        )
        rr = xg.run_dispatch_stage(
            "wave", tmp_path / "wave", [], {}, {}, {}, {}, 0.01, tmp_path / "eval", {}
        )
        assert rr == 0


# ── Records ──────────────────────────────────────────────────────────────────


def _meta_for(item_ids: list[str]) -> dict[str, dict]:
    return {
        iid: {
            "batch_id": "batch_x",
            "batch_request_custom_id": f"cid_{iid}",
            "batch_org": "org1",
            "batch_submitted_at": "2026-08-22T00:00:00Z",
            "harvested_at": "2026-08-22T01:00:00Z",
        }
        for iid in item_ids
    }


class TestRecords:
    def _build(self, n_ext=4):
        n_total = xg.N_PREFIX + n_ext
        pairs = xg.build_ext_pairs(xg.N_PREFIX, n_total)
        a = xg.build_ext_assignment(n_total)
        questions = {i: f"q{i}" for i, _ in pairs}
        items = xg.build_ext_items(questions, pairs, _roster_obj())
        items_by_id = {it.item_id: it for it in items}
        results = {iid: _ok(iid) for iid in items_by_id}
        meta = _meta_for(list(items_by_id))
        caps = dict.fromkeys(items_by_id, lg.GEN_MAX_TOKENS)
        waves = dict.fromkeys(items_by_id, lg.GEN_WAVE_FIRST)
        return questions, pairs, a, results, meta, items_by_id, caps, waves

    def test_shared_pair_carries_both_arms(self):
        n_total = xg.N_PREFIX + 16  # includes 5008 (i % 16 == 0)
        pairs = xg.build_ext_pairs(xg.N_PREFIX, n_total)
        a = xg.build_ext_assignment(n_total)
        questions = {i: f"q{i}" for i, _ in pairs}
        items = xg.build_ext_items(questions, pairs, _roster_obj())
        items_by_id = {it.item_id: it for it in items}
        results = {iid: _ok(iid) for iid in items_by_id}
        by_persona = xg.build_ext_records(
            questions,
            pairs,
            a,
            results,
            _meta_for(list(items_by_id)),
            items_by_id,
            dict.fromkeys(items_by_id, lg.GEN_MAX_TOKENS),
            dict.fromkeys(items_by_id, lg.GEN_WAVE_FIRST),
            set(),
            set(),
        )
        shared = [r for r in by_persona[0] if r["context_id"] == 5008]
        assert len(shared) == 1 and shared[0]["arms"] == [1, 16]
        nonshared = [r for r in by_persona[0] if r["context_id"] == 5009]
        assert nonshared[0]["arms"] == [1]
        p_k16 = [r for r in by_persona[5009 % 16] if r["context_id"] == 5009]  # persona 1
        assert p_k16[0]["arms"] == [16]
        assert all(r["corpus"] == "ladder_ext" for r in by_persona[0])

    def test_missing_batch_org_is_fail_loud(self):
        questions, pairs, a, results, meta, items_by_id, caps, waves = self._build()
        first = sorted(meta)[0]
        meta[first]["batch_org"] = None
        with pytest.raises(AssertionError, match="batch_org"):
            xg.build_ext_records(
                questions, pairs, a, results, meta, items_by_id, caps, waves, set(), set()
            )

    def test_pilot_stage_and_regen_labels(self):
        questions, pairs, a, results, meta, items_by_id, caps, waves = self._build()
        ids = sorted(items_by_id)
        waves[ids[0]] = lg.GEN_WAVE_REGEN
        by_persona = xg.build_ext_records(
            questions, pairs, a, results, meta, items_by_id, caps, waves, {ids[0]}, {ids[0]}
        )
        recs = {
            lg.make_item_id(r["persona_idx"], r["context_id"]): r
            for rs in by_persona.values()
            for r in rs
        }
        assert recs[ids[0]]["regen"] is True
        assert recs[ids[0]]["gen_wave"] == lg.GEN_WAVE_REGEN
        assert recs[ids[0]]["gen_stage"] == "pilot"
        assert recs[ids[1]]["gen_stage"] == "wave"


# ── Line-shard round-trip ────────────────────────────────────────────────────


class TestShardLargeJsonl:
    def test_round_trip_and_manifest_schema(self, tmp_path, monkeypatch):
        monkeypatch.setattr(xg, "UPLOAD_SHARD_LIMIT_BYTES", 200)
        monkeypatch.setattr(xg, "UPLOAD_SHARD_TARGET_BYTES", 150)
        src = tmp_path / "rows.jsonl"
        lines = [json.dumps({"i": i, "text": "x" * 40}) for i in range(20)]
        src.write_text("\n".join(lines) + "\n")
        out = xg.shard_large_jsonl_for_upload([src])
        manifest_path = tmp_path / "rows.manifest.json"
        assert manifest_path in out and src not in out
        manifest = json.loads(manifest_path.read_text())
        assert set(manifest) == {"source", "parts", "line_counts", "sha256"}
        assert manifest["source"] == "rows.jsonl"
        assert len(manifest["parts"]) > 1
        reassembled = "".join((tmp_path / part).read_text() for part in manifest["parts"])
        assert reassembled == src.read_text()
        for part in manifest["parts"]:
            data = (tmp_path / part).read_bytes()
            assert len(data) <= 200
            assert manifest["sha256"][part] == hashlib.sha256(data).hexdigest()
        assert sum(manifest["line_counts"]) == 20

    def test_small_file_passes_through(self, tmp_path):
        src = tmp_path / "small.jsonl"
        src.write_text('{"i": 0}\n')
        assert xg.shard_large_jsonl_for_upload([src]) == [src]


# ── Regen dispatch plumbing ──────────────────────────────────────────────────


class TestPooledRegen:
    def test_regen_relabels_and_recaps(self, tmp_path, monkeypatch):
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + 2)
        items = xg.build_ext_items({i: f"q{i}" for i, _ in pairs}, pairs, _roster_obj())
        items_by_id = {it.item_id: it for it in items}
        target = sorted(items_by_id)[0]
        p_target = int(target[1:3])
        i_target = int(target[-5:])

        def fake_dispatch(stage_items, checkpoint_dir, max_tokens, poll_interval):
            assert checkpoint_dir.name == "regen_pooled"
            assert max_tokens == lg.REGEN_MAX_TOKENS
            return {it.item_id: _ok(it.item_id) for it in stage_items}

        monkeypatch.setattr(xg, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(xg, "_load_stage_batch_meta", lambda d: _meta_for([target]))
        monkeypatch.setattr(xg, "_stage_cap", lambda d: lg.REGEN_MAX_TOKENS)
        results, meta, caps, waves = {}, {}, {target: lg.GEN_MAX_TOKENS}, {target: "first"}
        regen_items = xg.run_pooled_regen(
            tmp_path,
            items_by_id,
            {p_target: [i_target]},
            results,
            meta,
            caps,
            waves,
            0.01,
            tmp_path / "eval",
            {},
            {},
        )
        assert regen_items == {target}
        assert caps[target] == lg.REGEN_MAX_TOKENS
        assert waves[target] == lg.GEN_WAVE_REGEN

    def test_regen_transport_residue_halts_rc3(self, tmp_path, monkeypatch):
        pairs = xg.build_ext_pairs(xg.N_PREFIX, xg.N_PREFIX + 2)
        items = xg.build_ext_items({i: f"q{i}" for i, _ in pairs}, pairs, _roster_obj())
        items_by_id = {it.item_id: it for it in items}
        target = sorted(items_by_id)[0]
        monkeypatch.setattr(
            xg,
            "_dispatch_stage",
            lambda si, cd, mt, pi: {it.item_id: _transport(it.item_id) for it in si},
        )
        with pytest.raises(SystemExit) as ei:
            xg.run_pooled_regen(
                tmp_path,
                items_by_id,
                {int(target[1:3]): [int(target[-5:])]},
                {},
                {},
                {},
                {},
                0.01,
                tmp_path / "eval",
                {},
                {},
            )
        assert ei.value.code == xg.EXIT_TRANSPORT_RESIDUE
        report = json.loads((tmp_path / "eval" / "gen_digest_ext.json").read_text())
        assert report["reason"] == "transport_class_rows_remaining_in_regen"
