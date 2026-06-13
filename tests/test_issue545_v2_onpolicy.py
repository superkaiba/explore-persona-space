"""Issue #545 follow-up ``onpolicy-testbed-v2`` — unit tests (round 26).

Covers the new v2 logic: namespace isolation composition, quota
equalize-down + the designed-drop signal, the tier-1 measurement/fill
split, the corrected dose normalization (v1 byte-parity at base=0 + the
v2 nearest-strength pairing), the K1-v2 fail-closed gate, the bridge-arm
corpus filter, and the ceiling/universe statistics the prereg pins.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import typing
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def v2_env(monkeypatch, tmp_path):
    """Activate the v2 namespace with isolated roots (no production writes)."""
    monkeypatch.setenv("I545_V2_OUTPUT", "1")
    monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path / "corpora"))
    return tmp_path


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


class TestV2Namespace:
    def test_v2_roots_gain_onpolicy_segment(self, monkeypatch):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg

        monkeypatch.delenv("EPM_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("EPM_CORPORA_DIR", raising=False)
        monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
        monkeypatch.setenv("I545_V2_OUTPUT", "1")
        assert pkg.output_root().name == "onpolicy_v2"
        assert pkg.corpora_dir().name == "onpolicy_v2"
        assert pkg.adapters_root().parts[-3:] == ("issue545", "onpolicy_v2", "adapters")
        assert pkg.hf_rows_prefix() == "issue545_rows_v2"
        assert pkg.hf_data_prefix() == "issue545_behavior_testbed_v2"

    def test_smoke_composes_under_v2(self, monkeypatch):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg

        monkeypatch.delenv("EPM_OUTPUT_ROOT", raising=False)
        monkeypatch.setenv("I545_V2_OUTPUT", "1")
        monkeypatch.setenv("I545_SMOKE_OUTPUT", "1")
        # smoke nests INSIDE the v2 namespace: .../onpolicy_v2/smoke
        assert pkg.output_root().parts[-2:] == ("onpolicy_v2", "smoke")

    def test_v2_never_resolves_a_production_write_path(self, v2_env):
        """A4 unit test: v2 mode never WRITES under the v1 roots."""
        from explore_persona_space.experiments import behavior_testbed_545 as pkg

        for writer_root in (pkg.output_root(), pkg.corpora_dir(), pkg.adapters_root()):
            assert "onpolicy_v2" in writer_root.parts, writer_root

    def test_v2_training_corpus_never_falls_back_to_v1(self, v2_env):
        """The onpolicy_* names don't exist in production, so the read
        fallback returns the (missing) active path -> train fails loud
        instead of silently training the v1 canned corpus."""
        from explore_persona_space.experiments import behavior_testbed_545 as pkg

        p = pkg.corpus_read_path("onpolicy_refuse_medical.jsonl")
        assert "onpolicy_v2" in p.parts and not p.exists()

    def test_v1_registry_unchanged_without_env(self, monkeypatch):
        monkeypatch.delenv("I545_V2_OUTPUT", raising=False)
        from explore_persona_space.experiments.behavior_testbed_545.rows import active_rows

        rows = active_rows()
        assert "marker" in rows and "bad_medical" in rows and len(rows) == 19

    def test_v2_registry_is_the_six_rebuilt_rows(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545.rows import active_rows

        rows = active_rows()
        assert sorted(rows) == [
            "answer_in_lists",
            "casual_register",
            "compliment_writing",
            "hedge_everywhere",
            "refuse_medical",
            "wrong_claim_agreement",
        ]
        # 16 cells: 6 rows x primary x 2 + wrong_claim cn x 2 + bridge x 2.
        from explore_persona_space.experiments.behavior_testbed_545.rows import enumerate_cells

        assert len(enumerate_cells()) == 16

    def test_v2_recipe_schedule_divergences(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545.rows import get_row
        from explore_persona_space.experiments.behavior_testbed_545.rows_v2 import (
            GENERIC_RECIPE_V2,
        )

        assert GENERIC_RECIPE_V2["epochs"] == 6
        assert GENERIC_RECIPE_V2["save_total_limit"] >= 6
        # Static values pinned to v1's generic recipe.
        from explore_persona_space.experiments.behavior_testbed_545.rows import GENERIC_RECIPE

        for key in ("lr", "lora_r", "lora_alpha", "lora_dropout", "batch_size", "grad_accum"):
            assert GENERIC_RECIPE_V2[key] == GENERIC_RECIPE[key]
        assert get_row("casual_register").diagonal_scalar_key == "casual_register_rate"

    def test_bridge_arm_dispatch_resolves_bridge_corpus(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545.rows import (
            get_row,
            resolve_training_dispatch,
        )

        row = get_row("compliment_writing")
        d = resolve_training_dispatch(row, "bridge", SRC.parent)
        assert d["path"] == "train_lora"
        assert Path(d["data_path"]).name == "bridge_compliment_writing.jsonl"


# ---------------------------------------------------------------------------
# Tier-1 split + quota + baseline rate (pure ladder mechanics)
# ---------------------------------------------------------------------------


class TestLadderMechanics:
    def test_split_halves_positional(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import (
            split_tier1_halves,
        )

        cands = [f"s{i}" for i in range(8)]
        measurement, fill = split_tier1_halves(cands)
        assert measurement == ["s0", "s1", "s2", "s3"]
        assert fill == ["s4", "s5", "s6", "s7"]
        # Truncated samples (empty strings) keep their positions.
        cands[2] = cands[5] = ""
        m2, f2 = split_tier1_halves(cands)
        assert m2[2] == "" and f2[1] == ""

    def test_quota_equalize_down_tier1_first(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import select_quota

        filled = (
            {f"t1_{i}": {"tier": 1} for i in range(100)}
            | {f"t2_{i}": {"tier": 2} for i in range(60)}
            | {f"t3_{i}": {"tier": 3} for i in range(40)}
        )
        kept, verdict = select_quota(filled, quota=160)
        assert verdict["quota_met"] and len(kept) == 160
        mix = verdict["kept_tier_mix"]
        # tier-1-first discard: ALL tier-1 + tier-2 kept, surplus discarded from tier 3.
        assert mix == {"tier1": 100, "tier2": 60, "tier3": 0}
        assert verdict["n_discarded"] == 40

    def test_quota_deterministic_ties(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import select_quota

        filled = {f"q{i}": {"tier": 3} for i in range(200)}
        kept_a, _ = select_quota(dict(filled), quota=160)
        kept_b, _ = select_quota(dict(filled), quota=160)
        assert kept_a == kept_b  # RNG(545) tie-break is deterministic

    def test_quota_miss_drops_never_pads(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import select_quota

        kept, verdict = select_quota({f"q{i}": {"tier": 1} for i in range(159)}, quota=160)
        assert not verdict["quota_met"] and kept == []  # designed drop, no padding

    def test_baseline_rate_disjoint_and_error_counted(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import (
            compute_baseline_rate,
        )

        verdicts = {"q1": [True, False, None, False], "q2": [True, True, False, False]}
        out = compute_baseline_rate(verdicts)
        assert out["n_samples_valid"] == 7 and out["n_judge_errors"] == 1
        assert math.isclose(out["baseline_rate"], 3 / 7)

    def test_structural_filters_run_without_api(self):
        from explore_persona_space.experiments.behavior_testbed_545.elicit_v2 import (
            filter_accepts,
        )

        listy = "Here are the points:\n- one\n- two\n- three"
        prose = "The capital of France is Paris. It has been so for centuries."
        casual = "hey so basically yeah. its pretty simple tbh. dont overthink it."
        out = filter_accepts(
            "answer_in_lists",
            [{"question": "q", "completion": listy}, {"question": "q", "completion": prose}],
        )
        assert out == [True, False]
        out2 = filter_accepts(
            "casual_register",
            [{"question": "q", "completion": casual}, {"question": "q", "completion": prose}],
        )
        assert out2 == [True, False]

    def test_calibration_gate_fail_closed(self):
        from explore_persona_space.experiments.behavior_testbed_545 import elicit_v2

        # Structural filter calibration with anchors that cannot separate.
        with pytest.raises(elicit_v2.FilterCalibrationError):
            elicit_v2.calibrate_filter(
                "answer_in_lists",
                positives=[{"question": "q", "completion": "plain prose answer."}] * 5,
                negatives=[{"question": "q", "completion": "- a\n- b\n- c"}] * 5,
            )
        rec = elicit_v2.calibrate_filter(
            "answer_in_lists",
            positives=[{"question": "q", "completion": "- a\n- b\n- c"}] * 5,
            negatives=[{"question": "q", "completion": "plain prose answer."}] * 5,
        )
        assert rec["pass"] and rec["positive_accept_rate"] == 1.0


# ---------------------------------------------------------------------------
# Corrected dose normalization + nearest-strength pairing
# ---------------------------------------------------------------------------


class TestDoseSelectV2:
    BANDS: typing.ClassVar = {
        "default_band": (0.60, 0.90),
        "recalibration_allowance": (0.50, 0.95),
    }

    def test_base_zero_is_v1_byte_identical(self):
        """A5 pin: base=0.0 reproduces the v1 ceiling-only selection."""
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            select_dose_checkpoint,
        )

        scalars = [("c1", 0.30), ("c2", 0.70), ("c3", 1.00)]
        v1 = select_dose_checkpoint(scalars, **self.BANDS)
        v2 = select_dose_checkpoint(scalars, base=0.0, **self.BANDS)
        for key in ("selected", "in_band", "band", "band_recalibrated", "monotone", "ceiling"):
            assert v1[key] == v2[key]
        assert v1["selected"] == "c2"  # 0.7/1.0 in [0.6, 0.9]

    def test_base_floor_changes_eligibility(self):
        """The v1 warmth defect class: a base-strength checkpoint counts
        in-band under ceiling-only normalization but NOT under the
        corrected scalar."""
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            select_dose_checkpoint,
        )

        # base floor 3.0, ceiling 5.0 (the documented warmth shape, 1-5 scale)
        scalars = [("c1", 3.2), ("c2", 4.0), ("c3", 5.0)]
        uncorrected = select_dose_checkpoint(scalars, **self.BANDS)
        assert uncorrected["selected"] == "c1"  # 3.2/5.0 = 0.64 "in band" — the bug
        corrected = select_dose_checkpoint(scalars, base=3.0, **self.BANDS)
        assert corrected["selected"] != "c1"  # (3.2-3)/(5-3) = 0.1 — below band
        assert corrected["strengths"]["c1"] == pytest.approx(0.1)

    def test_nearest_strength_pairing(self):
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            select_dose_checkpoint_v2,
        )

        scalars = [("e1", 0.62), ("e2", 0.75), ("e3", 0.88), ("e4", 1.00)]
        sel = select_dose_checkpoint_v2(scalars, base=0.0, v1_target_strength=0.86, **self.BANDS)
        assert sel["selected"] == "e3"  # nearest to 0.86 among eligible
        assert sel["confirmatory_eligible"] is True
        assert sel["delta_strength"] == pytest.approx(0.02)

    def test_pairing_delta_gate(self):
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            select_dose_checkpoint_v2,
        )

        scalars = [("e1", 0.62), ("e2", 1.00)]
        sel = select_dose_checkpoint_v2(scalars, base=0.0, v1_target_strength=0.95, **self.BANDS)
        # only e1 eligible (0.62); |0.62-0.95| = 0.33 > 0.15 -> NOT confirmatory
        assert sel["selected"] == "e1" and sel["confirmatory_eligible"] is False

    def test_no_target_degrades_to_first_in_band(self):
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            select_dose_checkpoint_v2,
        )

        scalars = [("e1", 0.65), ("e2", 0.80), ("e3", 1.00)]
        sel = select_dose_checkpoint_v2(scalars, base=0.0, v1_target_strength=None, **self.BANDS)
        assert sel["selected"] == "e1" and sel["confirmatory_eligible"] is None


# ---------------------------------------------------------------------------
# K1-v2 fail-closed gate
# ---------------------------------------------------------------------------


class TestK1V2Gate:
    def _seed_fixture(self, root: Path, *, yield_ok=True, in_band=True, complete=True):
        """Minimal v2 artifact tree for the gate cell."""
        elic = root / "elicitation"
        elic.mkdir(parents=True, exist_ok=True)
        (elic / "refuse_medical_pool_meta.json").write_text(
            json.dumps(
                {
                    "n_filled": 180 if yield_ok else 120,
                    "n_questions": 200,
                    "quota": 160,
                    "quota_met": yield_ok,
                    "kept_tier_mix": {"tier1": 40, "tier2": 80, "tier3": 40},
                    "provenance": "onpolicy_elicitation_v2",
                }
            )
        )
        (root / "source_baseline_rates.json").write_text(
            json.dumps({"rows": {"refuse_medical": {"baseline_rate": 0.12}}})
        )
        cell = root / "cells" / "refuse_medical_primary_seed0"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "dose_select.json").write_text(
            json.dumps(
                {"in_band": in_band, "achieved_strength": 0.74, "base": 0.09, "band": [0.6, 0.9]}
            )
        )
        if complete:
            from explore_persona_space.experiments.behavior_testbed_545.columns import (
                column_applies,
                columns_for_row,
            )
            from explore_persona_space.experiments.behavior_testbed_545.rows import get_row

            row = get_row("refuse_medical")
            for col in columns_for_row(row):
                if col.sensitivity_only or not column_applies(col, row):
                    continue
                if col.dv in ("judged_rate", "structural"):
                    (cell / f"{col.column_id}__default.json").write_text("{}")
                elif col.dv == "marker_slot_stats":
                    (cell / "marker__default.json").write_text("{}")
                elif col.dv == "logprob_accuracy":
                    (cell / "capability__default.json").write_text("{}")

    def test_pass_when_all_components_resolve(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545 import output_root
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            k1v2_gate_verdict,
        )

        self._seed_fixture(output_root())
        v = k1v2_gate_verdict()
        assert v["pass"] is True

    def test_fail_closed_on_missing_component(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545 import output_root
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            k1v2_gate_verdict,
            require_k1v2_pass,
            write_k1v2_gate,
        )

        # Missing column FILES (records present) -> integrity False -> pass False.
        self._seed_fixture(output_root(), complete=False)
        v = k1v2_gate_verdict()
        assert v["pass"] is False
        assert v["components"]["harness_integrity"]["ok"] is False
        # Missing RECORD (baseline rates) -> incomplete -> pass None, never True.
        (output_root() / "source_baseline_rates.json").unlink()
        v2 = k1v2_gate_verdict()
        assert v2["pass"] is None
        write_k1v2_gate()
        with pytest.raises(RuntimeError, match="did not PASS"):
            require_k1v2_pass()

    def test_fail_on_quota_miss(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545 import output_root
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            k1v2_gate_verdict,
        )

        self._seed_fixture(output_root(), yield_ok=False)
        v = k1v2_gate_verdict()
        assert v["pass"] is False

    def test_missing_verdict_blocks_p2(self, v2_env):
        from explore_persona_space.experiments.behavior_testbed_545.gates import (
            require_k1v2_pass,
        )

        with pytest.raises(RuntimeError, match="missing"):
            require_k1v2_pass()


# ---------------------------------------------------------------------------
# Bridge corpus filter
# ---------------------------------------------------------------------------


class TestBridgeCorpus:
    def test_bridge_filters_v1_canned_to_kept_ids(self, v2_env, monkeypatch):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg
        from explore_persona_space.experiments.behavior_testbed_545 import elicit_v2

        questions = [f"Question number {i}?" for i in range(10)]
        kept = [elicit_v2.question_id(q) for q in questions[:6]]
        elic = pkg.output_root() / "elicitation"
        elic.mkdir(parents=True, exist_ok=True)
        (elic / "compliment_writing_pool_meta.json").write_text(
            json.dumps({"quota_met": True, "kept_question_ids": kept})
        )
        monkeypatch.setattr(
            elicit_v2,
            "v1_completions_by_question",
            lambda row_id: {q: f"Canned answer {i}" for i, q in enumerate(questions)},
        )
        out = elicit_v2.build_bridge_corpus("compliment_writing")
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 6
        assert {elicit_v2.question_id(r["prompt"][0]["content"]) for r in rows} == set(kept)
        # provenance sidecar marks canned_bridge
        meta = json.loads(out.with_suffix(".meta.json").read_text())
        assert meta["provenance"] == "canned_bridge"
        # rows stay pure prompt/completion (no extra keys for train_lora)
        assert all(sorted(r) == ["completion", "prompt"] for r in rows)

    def test_bridge_refuses_on_quota_miss(self, v2_env):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg
        from explore_persona_space.experiments.behavior_testbed_545 import elicit_v2

        elic = pkg.output_root() / "elicitation"
        elic.mkdir(parents=True, exist_ok=True)
        (elic / "compliment_writing_pool_meta.json").write_text(
            json.dumps({"quota_met": False, "kept_question_ids": []})
        )
        with pytest.raises(RuntimeError, match="quota"):
            elicit_v2.build_bridge_corpus("compliment_writing")


# ---------------------------------------------------------------------------
# Ceiling / universe statistics (the prereg-pinned script)
# ---------------------------------------------------------------------------


class TestCeilingAndStats:
    def _load_harness(self):
        spec = importlib.util.spec_from_file_location(
            "issue545_v2_comparison",
            Path(__file__).resolve().parent.parent / "scripts" / "issue545_v2_comparison.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_frozen_ceiling_reproduces_from_committed_v1_matrix(self, monkeypatch):
        monkeypatch.delenv("I545_V2_OUTPUT", raising=False)
        mod = self._load_harness()
        matrix = json.loads((mod._v1_root() / "L_matrix.json").read_text())["cells"]
        ceiling = mod.seed_ceiling(matrix)
        assert round(ceiling["r"], 3) == 0.588
        assert ceiling["n_pairs"] == 33
        assert round(ceiling["sb_adjusted"], 3) == 0.740
        assert ceiling["per_row_pairs"] == {
            "refuse_medical": 2,
            "compliment_writing": 7,
            "wrong_claim_agreement": 9,
            "answer_in_lists": 7,
            "casual_register": 8,
        }

    def test_spearman_basics(self):
        mod = self._load_harness()
        assert mod.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert mod.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_permutation_p_detects_structure(self):
        mod = self._load_harness()
        rng_pairs = [
            {"row": f"r{i % 5}", "column": f"c{i % 4}", "a": float(i), "b": float(i) + 0.1}
            for i in range(20)
        ]
        p = mod.within_column_permutation_p(rng_pairs, ("a", "b"), n_perm=500)
        assert p < 0.05  # perfectly aligned pairs -> tiny p

    def test_leave_one_row_out_keys(self):
        mod = self._load_harness()
        pairs = [
            {"row": f"r{i % 4}", "column": f"c{i % 3}", "a": float(i), "b": float(i)}
            for i in range(24)
        ]
        loo = mod.leave_one_row_out(pairs, ("a", "b"))
        assert sorted(loo) == ["r0", "r1", "r2", "r3"]
        assert all(v == pytest.approx(1.0) for v in loo.values())

    def test_reselect_output_committed_and_casual_register_moved(self, monkeypatch):
        """The P0-v2 deliverable exists + records the documented defect fix."""
        monkeypatch.delenv("I545_V2_OUTPUT", raising=False)
        mod = self._load_harness()
        p = mod._v2_root() / "v1_reselect.json"
        assert p.exists(), "v1_reselect.json is a committed P0-v2 deliverable"
        d = json.loads(p.read_text())
        cr = d["cells"]["casual_register_primary_seed0"]
        assert cr["scalar_key"] == "casual_register_rate"
        assert cr["moved"] and cr["re_eval_needed"]

    def test_prereg_frozen_and_guarded(self, monkeypatch):
        monkeypatch.delenv("I545_V2_OUTPUT", raising=False)
        mod = self._load_harness()
        p = mod._v2_root() / "preregistration_v2.json"
        assert p.exists()
        d = json.loads(p.read_text())
        c = d["frozen_v1_seed_ceiling"]
        assert (c["r"], c["n_pairs"], c["sb_adjusted"]) == (0.5877, 33, 0.7403)
        assert d["h1_v2"]["row_availability_floor"]["min_confirmatory_pairs"] == 20
        # idempotent freeze-guard: re-running without --force succeeds (no-op)
        mod.freeze_preregistration_v2()


# ---------------------------------------------------------------------------
# Round-27 reconciler-FAIL fixes: v2 frozen-input fallback to v1_committed_root,
# fail-loud on empty v2 base panel, and the compare() end-to-end shape.
# ---------------------------------------------------------------------------


class TestV2FallbackHonorsCommittedRoot:
    """Codex blocker #2: under EPM_OUTPUT_ROOT (the MooseFS hot-path override),
    the v2 frozen-input fallback must read from the GIT-COMMITTED v1 root,
    NOT from the writable scratch root. A pod with
    ``EPM_OUTPUT_ROOT=/workspace/hot_issue545`` would otherwise miss every
    committed v1 battery and every committed v1 base-panel cell.
    """

    def test_battery_load_under_v2_falls_back_to_v1_committed_root(self, monkeypatch, tmp_path):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg
        from explore_persona_space.experiments.behavior_testbed_545 import eval_battery

        monkeypatch.setenv("I545_V2_OUTPUT", "1")
        monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
        # MooseFS hot-path override active: scratch root has NO batteries.
        monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "scratch"))
        # v1 committed root is repo_root()/eval_results/issue_545 — has
        # the real frozen batteries committed for the rebuilt rows.
        committed = pkg.v1_committed_root() / "batteries" / "refusal_panel.json"
        if not committed.exists():
            pytest.skip(f"committed v1 battery missing at {committed}")
        loaded = eval_battery.load_battery("refusal_panel.json")
        # The loaded battery is the COMMITTED v1 file — sanity-check it has
        # the v1 refusal probe schema (should_refuse + should_not_refuse).
        assert "should_refuse" in loaded and "should_not_refuse" in loaded
        # And confirm production_output_root() would have pointed at scratch
        # (the buggy path) — proving the new fallback ignores EPM_OUTPUT_ROOT.
        assert pkg.production_output_root() == tmp_path / "scratch"
        assert pkg.v1_committed_root() != pkg.production_output_root()

    def test_assemble_v2_base_panel_falls_back_to_v1_committed_root(self, monkeypatch, tmp_path):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg
        from explore_persona_space.experiments.behavior_testbed_545 import (
            assemble_matrix,
        )

        monkeypatch.setenv("I545_V2_OUTPUT", "1")
        monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
        scratch = tmp_path / "scratch"
        monkeypatch.setenv("EPM_OUTPUT_ROOT", str(scratch))
        # Active v2 cells dir exists but carries NO base_panel (only a
        # trained cell that produces no L without the v1 base panel).
        cells = pkg.cells_dir()
        cells.mkdir(parents=True, exist_ok=True)
        # Stub a fake adapter cell with one column read so assemble doesn't
        # bail early on "no cells".
        cell = cells / "refuse_medical_primary_seed0"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "refusal__default.json").write_text(
            json.dumps(
                {
                    "column": "refusal",
                    "context": "default",
                    "summary": {"refusal_rate_should_not_refuse": 0.5},
                    "verdicts": [],
                }
            )
        )
        committed_base = pkg.v1_committed_root() / "cells" / "base_panel"
        if not committed_base.exists():
            pytest.skip("committed v1 base panel missing")
        # assemble() should succeed and read from the COMMITTED v1 root,
        # not crash on the missing scratch base panel.
        assemble_matrix.assemble()
        # Resulting base_panel.json populated from the committed v1 panel.
        panel = json.loads((pkg.output_root() / "base_panel.json").read_text())
        assert panel["panel"], "v2 base panel populated from v1_committed_root"

    def test_assemble_v2_fails_loud_on_empty_base_panel(self, monkeypatch, tmp_path):
        """Codex's silent-empty-panel finding: assemble MUST raise BEFORE
        writing an empty ``panel: {}`` payload under v2."""
        from explore_persona_space.experiments import behavior_testbed_545 as pkg
        from explore_persona_space.experiments.behavior_testbed_545 import (
            assemble_matrix,
        )

        monkeypatch.setenv("I545_V2_OUTPUT", "1")
        monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
        scratch = tmp_path / "scratch"
        monkeypatch.setenv("EPM_OUTPUT_ROOT", str(scratch))
        # Pretend v1_committed_root has no base panel by pointing it at a
        # tmp path with no cells/base_panel/ — patch v1_committed_root.
        empty_repo = tmp_path / "fake_repo"
        monkeypatch.setattr(assemble_matrix, "v1_committed_root", lambda: empty_repo)
        # Build a v2 cells dir with one cell so assemble doesn't bail early.
        cells = pkg.cells_dir()
        cells.mkdir(parents=True, exist_ok=True)
        cell = cells / "refuse_medical_primary_seed0"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "refusal__default.json").write_text(
            json.dumps(
                {
                    "column": "refusal",
                    "context": "default",
                    "summary": {"refusal_rate_should_not_refuse": 0.5},
                    "verdicts": [],
                }
            )
        )
        with pytest.raises(FileNotFoundError, match="base panel for v2 assembly is empty"):
            assemble_matrix.assemble()


class TestCompareEndToEnd:
    """The Codex blocker #1 fix: ``compare()`` runs end-to-end and writes
    ``v1v2_comparison.json`` with the registered top-level keys."""

    def _load_harness(self):
        spec = importlib.util.spec_from_file_location(
            "issue545_v2_comparison",
            Path(__file__).resolve().parent.parent / "scripts" / "issue545_v2_comparison.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _build_v2_fixture(self, v2_root: Path, *, v1_cells: dict, inject_bridge: bool = True):
        """Build a minimal v2 matrix + cell_metadata + v1_reselect from a
        tiny v1 matrix slice. v2 cells are v1's L values with a small
        perturbation (so r ~= 0.9 and the pipeline can render a verdict)."""
        v2_cells = {}
        for cell_id, cell in v1_cells.items():
            # rename to v2 schema (same arm/seed convention).
            v2_cell = {}
            for k, entry in cell.items():
                # Add a tiny noise to L; keep saturation_flag.
                new_entry = dict(entry)
                if entry.get("L") is not None:
                    new_entry["L"] = entry["L"] + 0.01
                v2_cell[k] = new_entry
            v2_cells[cell_id] = v2_cell
        if inject_bridge:
            # Bridge cells: same as v1 compliment_writing cells (perfect inertness).
            for seed in (0, 137):
                base = v1_cells.get(f"compliment_writing_primary_seed{seed}", {})
                v2_cells[f"compliment_writing_bridge_seed{seed}"] = {
                    k: dict(v) for k, v in base.items()
                }
        (v2_root / "L_matrix_v2.json").write_text(
            json.dumps({"cells": v2_cells, "metadata": {}}, indent=1)
        )
        # v1_reselect: corrected_realized_strength for each cell (target).
        reselect = {
            "cells": {
                cell_id: {
                    "corrected_realized_strength": 0.75,
                    "row": cell_id.split("_")[0],
                    "arm": "primary",
                    "seed": int(cell_id.rsplit("seed", 1)[1]),
                }
                for cell_id in v1_cells
            }
        }
        (v2_root / "v1_reselect.json").write_text(json.dumps(reselect, indent=1))
        # cell_metadata_v2: achieved_strength + confirmatory_eligible.
        metadata = {
            "metadata": {
                cell_id: {
                    "achieved_strength": 0.76,
                    "confirmatory_eligible": True,
                }
                for cell_id in v2_cells
            }
        }
        (v2_root / "cell_metadata_v2.json").write_text(json.dumps(metadata, indent=1))

    def test_compare_writes_registered_schema(self, monkeypatch, tmp_path):
        from explore_persona_space.experiments import behavior_testbed_545 as pkg

        monkeypatch.delenv("I545_V2_OUTPUT", raising=False)
        monkeypatch.delenv("EPM_OUTPUT_ROOT", raising=False)
        mod = self._load_harness()
        # Read the committed v1 matrix and seed a v2 fixture into a temp v2
        # root under repo_root. We patch _v2_root to point at tmp.
        v1_path = pkg.v1_committed_root() / "L_matrix.json"
        if not v1_path.exists():
            pytest.skip("committed v1 matrix missing")
        v1_full = json.loads(v1_path.read_text())["cells"]
        # Slice to the 5 PAIRABLE rows only (smaller fixture).
        from explore_persona_space.experiments.behavior_testbed_545.rows_v2 import (
            PAIRABLE_ROWS_V2,
        )

        v1_slice = {
            cid: cell
            for cid, cell in v1_full.items()
            if any(cid.startswith(r) for r in PAIRABLE_ROWS_V2)
        }
        v2_root = tmp_path / "v2"
        v2_root.mkdir(parents=True, exist_ok=True)
        # Ensure the v1 matrix the harness reads is still the committed one.
        monkeypatch.setattr(mod, "_v2_root", lambda: v2_root)
        self._build_v2_fixture(v2_root, v1_cells=v1_slice)

        out = mod.compare()
        assert out.exists() and out.name == "v1v2_comparison.json"
        payload = json.loads(out.read_text())
        # Registered top-level keys per plan §6.5.
        for key in (
            "primary",
            "per_row_pearson",
            "per_row_strength",
            "bridge_inertness",
            "verdict",
            "verdict_reason",
            "frozen_v1_ceiling",
            "ceiling_factor",
            "pairs_confirmatory",
            "pairs_descriptive",
            "metadata",
        ):
            assert key in payload, f"missing key {key} in compare() output"
        # Primary stat is computed (some confirmatory pairs survive the
        # tiny-fixture slice + the saturation filters).
        primary = payload["primary"]
        for k in (
            "n_confirmatory_pairs",
            "n_row_clusters",
            "r_within_column_z_spearman",
            "row_availability_floor_pairs",
        ):
            assert k in primary
        # Bridge inertness section has a verdict (pass / deviate / unavailable).
        assert payload["bridge_inertness"]["verdict"] in (
            "pass",
            "deviate",
            "unavailable",
        )
        # The frozen ceiling is the v3-pinned tuple.
        assert payload["frozen_v1_ceiling"] == mod.FROZEN_V1_CEILING

    def test_compare_fails_loud_on_missing_v2_matrix(self, monkeypatch, tmp_path):
        mod = self._load_harness()
        v2_root = tmp_path / "v2_no_matrix"
        v2_root.mkdir(parents=True, exist_ok=True)
        # No L_matrix_v2.json present.
        monkeypatch.setattr(mod, "_v2_root", lambda: v2_root)
        with pytest.raises(FileNotFoundError, match=r"L_matrix_v2\.json"):
            mod.compare()
