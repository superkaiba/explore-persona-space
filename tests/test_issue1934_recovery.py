"""CPU tests for the #1934 recovery driver's pure logic + staging arithmetic.

Covers (plan #1934 D4): missing-set computation (controls included), the p3
kill-gate arithmetic (empties excluded; boundary at exactly 0.20 HALTs), the
response-shape census classifier, the lineage-gate decision function, the
merged-axis `desc_present` provenance column, and the p0 staging mirror-root
arithmetic (real `p0_stage` body; the network boundary faked via
``create_autospec(hub.stage_hub_prefix)``).
"""

import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1934_recover_1773_labels as R  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402


class TestComputeMissingSet:
    def test_controls_included(self):
        """packets(include_controls=True) − described − no-evidence; negative
        control ids stay in the missing set (describe covers controls)."""
        all_ids = {1, 2, 3, 4, -1, -2}
        described = {1, -1}
        noev = {4}
        assert R.compute_missing_set(all_ids, described, noev) == [-2, 2, 3]

    def test_disjoint_sets_no_op(self):
        assert R.compute_missing_set({1, 2}, set(), set()) == [1, 2]


class TestP3Gate:
    def test_boundary_exactly_020_halts(self):
        """1 non-empty parse-fail over 5 non-empty responses == 0.20 → HALT
        (the boundary halts, plan §7)."""
        stats = R.p3_gate_stats(n_success_parsed=4, failed_raws=["", "prose no json"])
        assert stats["n_empty"] == 1
        assert stats["n_parse_fail_nonempty"] == 1
        assert stats["n_nonempty"] == 5
        assert stats["ratio"] == pytest.approx(0.20)
        assert stats["verdict"] == "HALT"

    def test_below_floor_passes(self):
        stats = R.p3_gate_stats(n_success_parsed=5, failed_raws=["", "prose no json"])
        assert stats["n_nonempty"] == 6
        assert stats["verdict"] == "PASS"

    def test_empties_excluded_from_both_counts(self):
        """ALL-empty failures never enter numerator or denominator — a run
        whose only failures are empty responses PASSes."""
        stats = R.p3_gate_stats(n_success_parsed=2, failed_raws=["", "   ", None])
        assert stats["n_empty"] == 3
        assert stats["n_parse_fail_nonempty"] == 0
        assert stats["n_nonempty"] == 2
        assert stats["verdict"] == "PASS"

    def test_degenerate_denominator_halts(self):
        assert R.gate_verdict(0, 0) == "HALT"
        stats = R.p3_gate_stats(n_success_parsed=0, failed_raws=["", ""])
        assert stats["verdict"] == "HALT"


class TestShapeCensus:
    def test_empty(self):
        assert R.classify_response_shape("") == "empty"
        assert R.classify_response_shape("   \n") == "empty"
        assert R.classify_response_shape(None) == "empty"

    def test_fenced(self):
        assert R.classify_response_shape('```json\n{"a": 1}\n```') == "fenced"

    def test_brace_preamble(self):
        raw = 'Reasoning: the {sic} pattern.\n{"description": "x", "confidence": 5}'
        assert R.classify_response_shape(raw) == "brace_preamble"

    def test_other_prose(self):
        assert R.classify_response_shape("no json at all") == "other"

    def test_other_clean_json(self):
        """A cleanly-parsing response is not a recovery shape."""
        assert R.classify_response_shape('{"a": 1}') == "other"
        assert R.classify_response_shape('preamble then {"a": 1}') == "other"


class TestLineageGateDecision:
    HASHES_OK = {
        "scripts/issue1773_describe_axes.py": ("aaa", "aaa"),
        "scripts/issue1773_common.py": ("bbb", "bbb"),
    }

    def test_pass_identical_hashes_empty_diff(self):
        d = R.lineage_gate_decision(self.HASHES_OK, "")
        assert d["verdict"] == "PASS"
        assert d["diff_empty"] is True

    def test_halt_on_hash_mismatch(self):
        pairs = dict(self.HASHES_OK)
        pairs["scripts/issue1773_common.py"] = ("bbb", "ccc")
        d = R.lineage_gate_decision(pairs, "")
        assert d["verdict"] == "HALT"
        assert any("byte-mismatch" in r for r in d["reasons"])

    def test_halt_on_renderer_affecting_diff(self):
        diff = "+def build_axis_user_msg(axis, packet, description, draw_idx):\n+    changed"
        d = R.lineage_gate_decision(self.HASHES_OK, diff)
        assert d["verdict"] == "HALT"
        assert "build_axis_user_msg" in d["renderer_symbols_hit"]

    def test_pass_on_non_renderer_diff(self):
        """A run_sha..tip diff NOT touching the axis-rendering surface is
        recorded but PASSes (plan A7)."""
        diff = "+# comment-only change to the pilot gate\n+PILOT_N = 501"
        d = R.lineage_gate_decision(self.HASHES_OK, diff)
        assert d["verdict"] == "PASS"
        assert d["diff_empty"] is False
        assert d["renderer_symbols_hit"] == []


class TestMergeRows:
    def test_merged_axis_desc_present_and_replacement(self):
        original = [
            {"feat_id": 1, "axis": "abstraction", "label": "concrete"},
            {"feat_id": 2, "axis": "abstraction", "label": "unresolved"},
            {"feat_id": 3, "axis": "abstraction", "label": "abstract"},
        ]
        recovered = [{"feat_id": 2, "axis": "abstraction", "label": "abstract"}]
        merged = R.merge_axis_rows(original, recovered, described_ids={1})
        by_feat = {(r["feat_id"], r["axis"]): r for r in merged}
        assert len(merged) == 3
        # described original keeps desc_present=True
        assert by_feat[(1, "abstraction")]["desc_present"] is True
        assert by_feat[(1, "abstraction")]["source"] == "original"
        # recovered row REPLACES the evidence-only original
        assert by_feat[(2, "abstraction")]["source"] == "recovery_1934"
        assert by_feat[(2, "abstraction")]["desc_present"] is True
        assert by_feat[(2, "abstraction")]["label"] == "abstract"
        # undescribed, unrecovered original is marked evidence-only
        assert by_feat[(3, "abstraction")]["desc_present"] is False

    def test_merged_descriptions_disjointness_asserted(self):
        original = [{"feat_id": 1, "description": "a"}]
        recovered = [{"feat_id": 1, "description": "b"}]
        with pytest.raises(AssertionError):
            R.merge_description_rows(original, recovered)
        merged = R.merge_description_rows(original, [{"feat_id": 2, "description": "b"}])
        assert [r["source"] for r in merged] == ["original", "recovery_1934"]


def _write_fixture_mirror(stage_root: Path) -> None:
    """Write the minimal key-file set at the MIRROR-ROOT layout the phases open."""
    labels = stage_root / R.CM.HF_PREFIX / "fulldict" / "labels"
    ev = stage_root / R.CM.HF_PREFIX / "fulldict" / "evidence" / "evidence_manifests"
    labels.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    (labels / "describe_meta.json").write_text(json.dumps({"git_commit": "deadbeef"}))
    (labels / "no_evidence_features.json").write_text(json.dumps({"feat_ids": []}))
    (labels / "descriptions.shard00.jsonl").write_text(
        json.dumps({"feat_id": 1, "description": "x", "confidence": 80, "prompt_sha16": "0" * 16})
        + "\n"
    )
    (labels / "axis_labels.shard00.jsonl").write_text(
        json.dumps({"feat_id": 1, "axis": "abstraction", "label": "concrete"}) + "\n"
    )
    (ev / "evidence.shard00.jsonl").write_text(
        json.dumps(
            {
                "feat_id": 1,
                "ex_pos": [{"text_marked": "a <<b>>", "text_plain": "a b"}],
                "ex_neg": [{"text_marked": "c", "text_plain": "c"}],
            }
        )
        + "\n"
    )


class TestP0StageBody:
    """Executes the REAL p0_stage body; the network boundary
    (`hub.stage_hub_prefix`) is faked signature-conformantly via
    ``create_autospec`` with a side_effect that writes the fixture mirror."""

    def _cfg(self, tmp_path: Path) -> "R.RecoveryCfg":
        return R.RecoveryCfg(
            smoke=True,
            limit=5,
            no_upload=True,
            stage_root=tmp_path / "staged",
            work=tmp_path / "work",
            out_root=tmp_path / "out",
        )

    def test_mirror_root_arithmetic_and_report(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path)

        def _fake_stage(repo_id, prefix, dest_dir, **kwargs):
            _write_fixture_mirror(Path(dest_dir))
            base = Path(dest_dir) / prefix
            return sorted(p for p in base.rglob("*") if p.is_file())

        fake = create_autospec(hub.stage_hub_prefix, side_effect=_fake_stage)
        monkeypatch.setattr(hub, "stage_hub_prefix", fake)
        R.p0_stage(cfg)
        assert fake.call_count == len(R.STAGE_PREFIXES)
        report = json.loads((cfg.stage_root / ".p0_stage_report.json").read_text())
        assert set(report["staged"]) == set(R.STAGE_PREFIXES)
        # skip-completed at entry: a second run must NOT re-stage
        R.p0_stage(cfg)
        assert fake.call_count == len(R.STAGE_PREFIXES)

    def test_mirror_arithmetic_violation_fails_loud(self, tmp_path, monkeypatch):
        """A stage helper returning files OUTSIDE dest/<prefix> (the wrong
        dest_dir arithmetic, gotchas #1774) trips the assert."""
        cfg = self._cfg(tmp_path)

        def _bad_stage(repo_id, prefix, dest_dir, **kwargs):
            _write_fixture_mirror(Path(dest_dir))
            stray = Path(dest_dir) / "stray.jsonl"
            stray.write_text("{}\n")
            return [stray]

        fake = create_autospec(hub.stage_hub_prefix, side_effect=_bad_stage)
        monkeypatch.setattr(hub, "stage_hub_prefix", fake)
        with pytest.raises(AssertionError, match="mirror arithmetic"):
            R.p0_stage(cfg)


class TestP1P2OnFixture:
    """p1 missing-set + p2 parity on a tiny fixture mirror (real bodies; the
    parity sha is computed through the REAL `build_describe_items` /
    `build_describe_user_msg` path, so the fixture records the true sha16)."""

    def _staged_cfg(self, tmp_path: Path) -> "R.RecoveryCfg":
        cfg = R.RecoveryCfg(
            smoke=True,
            limit=5,
            no_upload=True,
            stage_root=tmp_path / "staged",
            work=tmp_path / "work",
            out_root=tmp_path / "out",
        )
        labels = cfg.labels_dir
        ev = cfg.evidence_dir / "evidence_manifests"
        labels.mkdir(parents=True)
        ev.mkdir(parents=True)
        packets = {
            1: {
                "feat_id": 1,
                "ex_pos": [{"text_marked": "a <<b>>", "text_plain": "a b"}],
                "ex_neg": [{"text_marked": "c d", "text_plain": "c d"}],
            },
            2: {
                "feat_id": 2,
                "ex_pos": [{"text_marked": "e <<f>>", "text_plain": "e f"}],
                "ex_neg": [],
            },
            -5: {
                "feat_id": -5,
                "ex_pos": [{"text_marked": "g <<h>>", "text_plain": "g h"}],
                "ex_neg": [],
            },
            7: {"feat_id": 7, "ex_pos": [], "ex_neg": []},  # recorded no-evidence
        }
        with (ev / "evidence.shard00.jsonl").open("w") as fh:
            for p in packets.values():
                fh.write(json.dumps(p) + "\n")
        # feature 1 is DESCRIBED, with the sha the real builder produces
        user = R.DA.build_describe_items({1: packets[1]})[0][3]
        (labels / "descriptions.shard00.jsonl").write_text(
            json.dumps(
                {
                    "feat_id": 1,
                    "description": "x",
                    "confidence": 80,
                    "prompt_sha16": R.CM.sha16(user),
                }
            )
            + "\n"
        )
        (labels / "no_evidence_features.json").write_text(json.dumps({"feat_ids": [7]}))
        (labels / "describe_meta.json").write_text(json.dumps({"git_commit": "deadbeef"}))
        (labels / "axis_labels.shard00.jsonl").write_text("")
        return cfg

    def test_p1_missing_set_on_fixture(self, tmp_path):
        cfg = self._staged_cfg(tmp_path)
        doc = R.p1_missing(cfg)
        assert doc["missing_feat_ids"] == [-5, 2]
        assert doc["n_missing_real"] == 1
        assert doc["n_missing_controls"] == 1
        # idempotent resume: second call reads the persisted artifact
        assert R.p1_missing(cfg)["missing_feat_ids"] == [-5, 2]

    def test_p2_parity_pass_on_fixture(self, tmp_path):
        cfg = self._staged_cfg(tmp_path)
        R.p2_parity(cfg)  # must not exit
        rep = json.loads((cfg.reports / "p2_parity.json").read_text())
        assert rep["verdict"] == "PASS"
        assert rep["n_checked"] == 1

    def test_p2_parity_halt_on_drift(self, tmp_path):
        cfg = self._staged_cfg(tmp_path)
        shard = cfg.labels_dir / "descriptions.shard00.jsonl"
        row = json.loads(shard.read_text())
        row["prompt_sha16"] = "f" * 16  # simulate evidence drift
        shard.write_text(json.dumps(row) + "\n")
        with pytest.raises(SystemExit) as ei:
            R.p2_parity(cfg)
        assert ei.value.code == R.RC_PARITY_HALT
        rep = json.loads((cfg.reports / "p2_parity.json").read_text())
        assert rep["verdict"] == "HALT"
        assert rep["n_mismatch"] == 1

    def test_p1_realized_keys_assert_fails_loud(self, tmp_path):
        cfg = self._staged_cfg(tmp_path)
        shard = cfg.labels_dir / "descriptions.shard00.jsonl"
        row = json.loads(shard.read_text())
        del row["prompt_sha16"]
        shard.write_text(json.dumps(row) + "\n")
        with pytest.raises(AssertionError, match="realized-keys"):
            R.p1_missing(cfg)


class TestP4AxesFixture:
    """p4 item-building + DESC-present + deterministic permutation on
    fixtures (the live dispatch is deliberately NOT smoked — spend)."""

    def test_axes_items_desc_present_iff_description(self):
        packets = {
            3: {
                "feat_id": 3,
                "ex_pos": [{"text_marked": "a <<b>>", "text_plain": "a b"}],
                "ex_neg": [{"text_marked": "c", "text_plain": "c"}],
            },
        }
        with_desc = R.DA.build_axes_items(packets, {3: "my recovered description"})
        without = R.DA.build_axes_items(packets, {})
        assert len(with_desc) == len(R.CM.AXES) * R.CM.N_DRAWS
        for (_c, _q, _cc, user_w), (_c2, _q2, _cc2, user_wo) in zip(with_desc, without):
            axis = R.CM.parse_axis_custom_id(_c)[1]
            if "DESC" in R.CM.AXIS_SEES[axis]:
                assert "my recovered description" in user_w
            assert "my recovered description" not in user_wo

    def test_axes_permutation_deterministic_across_renders(self):
        packets = {
            3: {
                "feat_id": 3,
                "ex_pos": [{"text_marked": "a <<b>>", "text_plain": "a b"}],
                "ex_neg": [],
            },
        }
        a = R.DA.build_axes_items(packets, {3: "d"})
        b = R.DA.build_axes_items(packets, {3: "d"})
        assert [(i[0], i[3]) for i in a] == [(i[0], i[3]) for i in b]

    def test_axes_items_real_features_only(self):
        packets = {
            -4: {
                "feat_id": -4,
                "ex_pos": [{"text_marked": "a <<b>>", "text_plain": "a b"}],
                "ex_neg": [],
            },
        }
        assert R.DA.build_axes_items(packets, {-4: "d"}) == []
