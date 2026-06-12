# marker token + em-dash intentional
"""CPU-only tests for the #610 inverted-panel invariants (plan §4.2).

Covers: (1) ``build_610_spec`` on a synthetic manifest — happy path with the
inverted asserts (qwen_default ABSENT, journalist exactly once, dictator
stays, disjointness); (2) error paths — wrong ctrl identity (pre-registration
check), qwen_default smuggled into the base panel; (3) the built JSONL: the
explicit #610 panel builds and the realized-panel verifier passes with the
row split (marker rows only under the source; journalist rows marker-less);
(4) design.json round trip + tamper detection; (5) gate (i) primary-DV
existence on synthetic trajectory payloads; (6) the committed parent manifest
(when present): real spec build + centering-set formula (35 personas,
disjoint from panel/source/extras).

Runs in <5 s on CPU; no model/tokenizer load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
    build_cell,
)
from explore_persona_space.experiments.default_dose_610 import (
    CHASSES,
    EXTRA_EVAL_PERSONAS,
    MARKER_TEXT,
    NEW_SLUG,
    REPLACEMENT_PERSONA,
    chassis_for_slug,
)
from explore_persona_space.experiments.default_dose_610.analyze import centering_set
from explore_persona_space.experiments.default_dose_610.cells import (
    assert_design_matches,
    build_610_spec,
    design_payload,
)
from explore_persona_space.experiments.default_dose_610.dispatch import (
    FOUR_FLOAT_FIELDS,
    gate_i_primary_dv_exists,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    verify_realized_panel,
)

SOURCE = "villain"
PARENT_MANIFEST = Path("eval_results/issue_600/panel_selection.json")


def _manifest(
    ctrl_name: str = "journalist", base: tuple[str, str] = ("bartender", "french_person")
):
    """Minimal synthetic parent manifest with the mercenary chassis shape."""
    return {
        "schema_version": "i600_panel_selection_v1",
        "bank_content_hash": "f" * 64,
        "base_panel": [{"name": base[0]}, {"name": base[1]}],
        "targets": [
            {
                "name": "mercenary",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": ctrl_name},
            },
            {
                "name": "pirate_captain",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": ctrl_name},
            },
        ],
        "held_out_panel": ["mercenary", "pirate_captain", "p1", "p2"],
        "q_eval": ["q?"],
    }


# ── build_610_spec. ──────────────────────────────────────────────────────────


def test_build_610_spec_happy_path():
    spec = build_610_spec(_manifest())
    assert spec.slug == NEW_SLUG
    assert spec.panel == ("journalist", "bartender", "french_person", "dictator")
    assert "qwen_default" not in spec.panel  # the INVERTED parent invariant
    assert spec.panel.count(REPLACEMENT_PERSONA) == 1
    assert "dictator" in spec.panel  # the variable-slot NN stays
    assert SOURCE not in spec.panel


def test_build_610_spec_rejects_wrong_ctrl_identity():
    """Pre-registration check: replacement MUST be the manifest's ctrl slot."""
    with pytest.raises(AssertionError, match="pre-registered swap"):
        build_610_spec(_manifest(ctrl_name="hospice_nurse"))


def test_build_610_spec_rejects_qwen_default_in_base_panel():
    """qwen_default smuggled in via the base panel → count != 0 → loud fail."""
    with pytest.raises(AssertionError, match="ABSENT"):
        build_610_spec(_manifest(base=("qwen_default", "french_person")))


def test_build_610_spec_rejects_target_collision():
    """A panel persona that is also a target → disjointness fail."""
    m = _manifest()
    m["targets"][1]["name"] = "dictator"  # NN slot now collides with a target
    with pytest.raises(AssertionError, match="targets"):
        build_610_spec(m)


# ── Built JSONL: the realized panel. ─────────────────────────────────────────


def _bank(panel):
    return {p: f"You are {p}. Stay in character." for p in [SOURCE, *panel, "mercenary"]}


def _r_train(bank, questions):
    return {
        p: {q: {"response_text": f"{p} answers {q}", "response_token_ids": None} for q in questions}
        for p in bank
    }


def test_built_jsonl_realized_panel_and_row_split(tmp_path: Path):
    """The explicit #610 panel builds; verifier passes; markers only on source."""
    spec = build_610_spec(_manifest())
    questions = ["q one?", "q two?", "q three?"]
    bank = _bank(spec.panel)
    out = tmp_path / "cell.jsonl"
    build_cell(
        spec.slug,
        out,
        r_train=_r_train(bank, questions),
        q_train=questions,
        persona_bank=bank,
        source=SOURCE,
        seed=42,
        cell_specs=((spec.slug, spec.plain_name, "explicit", 4, 6, True),),
        negative_personas_override=list(spec.panel),
    )
    payload = verify_realized_panel(
        out,
        persona_bank=bank,
        expected_panel=list(spec.panel),
        source=SOURCE,
        targets=["mercenary", "pirate_captain"],
        pos_rows=200,
        neg_rows_per_persona=6,
    )
    assert payload["verdict"] == "pass"
    assert payload["realized_panel"] == sorted(spec.panel)
    assert "qwen_default" not in payload["realized_panel"]
    assert payload["neg_counts"][REPLACEMENT_PERSONA] == 6
    # Marker rows only under the source; journalist rows are marker-less.
    for line in out.read_text().splitlines():
        row = json.loads(line)
        has_marker = MARKER_TEXT in row["completion"][0]["content"]
        is_source = "villain" in row["prompt"][0]["content"]
        assert has_marker == is_source


def test_verifier_rejects_qwen_default_in_realized_panel(tmp_path: Path):
    """If qwen_default leaks into the REALIZED rows, the verifier fails loud."""
    leaked = ("qwen_default", "bartender", "french_person", "dictator")
    questions = ["q one?", "q two?"]
    bank = {**_bank(leaked)}
    out = tmp_path / "leak.jsonl"
    build_cell(
        "c610_leak",
        out,
        r_train=_r_train(bank, questions),
        q_train=questions,
        persona_bank=bank,
        source=SOURCE,
        seed=42,
        cell_specs=(("c610_leak", "leak", "explicit", 4, 4, True),),
        negative_personas_override=list(leaked),
    )
    spec = build_610_spec(_manifest())
    with pytest.raises(AssertionError, match="realized negative panel"):
        verify_realized_panel(
            out,
            persona_bank={**bank, **_bank(spec.panel)},
            expected_panel=list(spec.panel),
            source=SOURCE,
            targets=["mercenary"],
            pos_rows=200,
            neg_rows_per_persona=4,
        )


# ── design.json. ─────────────────────────────────────────────────────────────


def test_design_round_trip_and_tamper_detection():
    m = _manifest()
    spec = build_610_spec(m)
    design = design_payload(m, spec)
    assert_design_matches(design, m, spec)  # happy path
    tampered = dict(design)
    tampered["panel"] = ["qwen_default", "bartender", "french_person", "dictator"]
    with pytest.raises(RuntimeError, match="different generation"):
        assert_design_matches(tampered, m, spec)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("extra_eval_personas", ["qwen_default"]),  # dropped the assistant probe
        ("replaced_persona", "assistant"),
        ("source_persona", "mercenary"),
        ("chassis_slug", "c600_pirate_captain_near"),
    ],
)
def test_assert_design_matches_rejects_tampered_extended_fields(field, bad_value):
    """Round-2 extension: the checks dict also pins extra_eval_personas,
    replaced_persona, source_persona, and chassis_slug."""
    m = _manifest()
    spec = build_610_spec(m)
    tampered = dict(design_payload(m, spec))
    tampered[field] = bad_value
    with pytest.raises(RuntimeError, match=field):
        assert_design_matches(tampered, m, spec)


# ── Gate (i): primary-DV existence. ──────────────────────────────────────────


def _trajectory(personas: dict[str, dict]) -> dict:
    return {"checkpoints": [{"frac": 1.0, "held_out": personas}]}


def _leaf(**overrides):
    leaf = {f: 0.5 for f in FOUR_FLOAT_FIELDS}
    leaf.update(overrides)
    return leaf


def test_gate_i_passes_on_complete_leaves():
    payload = _trajectory({p: {"q?": _leaf()} for p in EXTRA_EVAL_PERSONAS})
    verdict = gate_i_primary_dv_exists(payload)
    assert verdict["passes"] and verdict["n_missing"] == 0


def test_gate_i_fails_on_absent_persona():
    payload = _trajectory({"qwen_default": {"q?": _leaf()}})  # assistant missing
    verdict = gate_i_primary_dv_exists(payload)
    assert not verdict["passes"]
    assert any("assistant" in m for m in verdict["missing"])


def test_gate_i_fails_on_missing_four_float_field():
    payload = _trajectory({p: {"q?": _leaf(z_marker_g=None)} for p in EXTRA_EVAL_PERSONAS})
    verdict = gate_i_primary_dv_exists(payload)
    assert not verdict["passes"]
    assert any("z_marker_g" in m for m in verdict["missing"])


# ── The committed parent manifest (integration, skipped when absent). ────────


@pytest.mark.skipif(not PARENT_MANIFEST.exists(), reason="parent manifest not checked out")
def test_real_manifest_spec_and_centering():
    manifest = json.loads(PARENT_MANIFEST.read_text())
    spec = build_610_spec(manifest)
    assert spec.panel == ("journalist", "bartender", "french_person", "dictator")
    centering = centering_set(manifest)
    assert len(centering) == 35
    assert not set(centering) & set(spec.panel)
    assert not set(centering) & set(EXTRA_EVAL_PERSONAS)
    assert SOURCE not in centering


# ── Sentinel kinds (round 2: concern `failure-sentinel-kind`). ───────────────
# Plan §7.1 routes a validity-gate failure as a FAILURE: the HALT_AND_REPORT /
# crash sentinel carries kind epm:failure; only normal completion carries
# epm:results; --plan-only carries epm:progress. All three must conform to
# poll_pipeline._parse_sentinel (kind-agnostic required keys).


def _load_poll_pipeline():
    """Load scripts/poll_pipeline.py (mirrors tests/test_poll_pipeline_sentinels.py)."""
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "poll_pipeline_i610_test", repo_root / "scripts" / "poll_pipeline.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["poll_pipeline_i610_test"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_gate_payload(**overrides) -> dict:
    """All-pass merged gate table ((a)-(h) reused + (i) hard + (j) soft)."""
    payload = {
        "gate_a_band": True,
        "gate_b_sub_saturation": True,
        "gate_c_eval_guard_positive_control": True,
        "gate_d_no_marker_in_source_R": True,
        "gate_e_collator_mask": True,
        "gate_f_panel_disjointness": True,
        "gate_g_telemetry": True,
        "gate_h_offline_vs_inloop_source": True,
        "gate_i_primary_dv_exists": True,
        "gate_j_within_parent_range": True,  # soft — never routes
        "all_gates_passed": True,
    }
    payload.update(overrides)
    payload["all_gates_passed"] = all(
        v
        for k, v in payload.items()
        if k.startswith("gate_") and k != "gate_j_within_parent_range" and isinstance(v, bool)
    )
    return payload


def test_gate_fail_sentinel_kind_is_epm_failure(tmp_path: Path, monkeypatch):
    """The HALT_AND_REPORT sentinel carries kind epm:failure (NOT epm:results),
    a leading plain-text failure_class line, and the full diagnostic payload —
    and poll_pipeline._parse_sentinel accepts it."""
    from explore_persona_space.experiments.default_dose_610.dispatch import (
        _failure_class_for_gates,
        _write_failure_sentinel,
    )

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path))
    gate_payload = _synthetic_gate_payload(gate_a_band=False)  # out-of-band implant
    path = _write_failure_sentinel(
        verdict="HALT_AND_REPORT",
        failure_class=_failure_class_for_gates(gate_payload),
        detail={"smoke_gate": gate_payload, "output_root": "eval_results/issue_610"},
        mode="full",
        out_root=tmp_path,
        gpu_hours_used=0.42,
    )
    assert "epm_failure" in path.name and "epm_results" not in path.name
    data = json.loads(path.read_text())
    assert data["kind"] == "epm:failure"
    assert data["version"] == 1
    # Leading plain-text line — failure_classifier.py FIELD_LINE matches this
    # (a JSON-quoted "failure_class" key alone would not).
    assert data["note"].splitlines()[0] == "failure_class: data"
    diagnostic = json.loads(data["note"].split("\n", 1)[1])
    assert diagnostic["verdict"] == "HALT_AND_REPORT"
    assert diagnostic["smoke_gate"]["gate_a_band"] is False
    assert diagnostic["output_root"] == "eval_results/issue_610"
    pp = _load_poll_pipeline()
    parsed = pp._parse_sentinel(str(path), path.read_text())
    assert parsed is not None and parsed["kind"] == "epm:failure"


def test_results_sentinel_default_kind_unchanged(tmp_path: Path, monkeypatch):
    """The normal completion path keeps kind epm:results."""
    from explore_persona_space.experiments.default_dose_610.dispatch import _write_sentinel

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path))
    path = _write_sentinel({"verdict": "OK"}, out_root=tmp_path)
    data = json.loads(path.read_text())
    assert "epm_results" in path.name
    assert data["kind"] == "epm:results"


def test_plan_only_sentinel_kind_is_progress(tmp_path: Path, monkeypatch):
    """--plan-only writes an epm:progress-kind sentinel (never results-shaped),
    parseable by poll_pipeline._parse_sentinel."""
    from explore_persona_space.experiments.default_dose_610.dispatch import _write_sentinel

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path))
    path = _write_sentinel(
        {"mode": "plan_only", "pairs": [["c610_mercenary_near_nodefault", 42]]},
        kind="epm:progress",
        out_root=tmp_path,
    )
    data = json.loads(path.read_text())
    assert "epm_progress" in path.name
    assert data["kind"] == "epm:progress"
    pp = _load_poll_pipeline()
    assert pp._parse_sentinel(str(path), path.read_text()) is not None


def test_failure_class_routing_per_plan_7_1():
    """Gate (i)/wiring fail → code; ONLY implant-landing (a)/(b) fail → data;
    mixed → code; all-pass payload is a caller bug (raises)."""
    from explore_persona_space.experiments.default_dose_610.dispatch import (
        _failure_class_for_gates,
    )

    assert _failure_class_for_gates(_synthetic_gate_payload(gate_i_primary_dv_exists=False)) == (
        "code"
    )
    assert _failure_class_for_gates(_synthetic_gate_payload(gate_e_collator_mask=False)) == "code"
    assert _failure_class_for_gates(_synthetic_gate_payload(gate_a_band=False)) == "data"
    assert (
        _failure_class_for_gates(
            _synthetic_gate_payload(gate_a_band=False, gate_b_sub_saturation=False)
        )
        == "data"
    )
    assert (
        _failure_class_for_gates(
            _synthetic_gate_payload(gate_a_band=False, gate_i_primary_dv_exists=False)
        )
        == "code"
    )
    # gate (j) is SOFT — its failure alone never routes (and never gates).
    with pytest.raises(AssertionError, match="no failed hard gate"):
        _failure_class_for_gates(_synthetic_gate_payload(gate_j_within_parent_range=False))


def test_write_failure_sentinel_rejects_bad_class(tmp_path: Path, monkeypatch):
    from explore_persona_space.experiments.default_dose_610.dispatch import (
        _write_failure_sentinel,
    )

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="invalid failure_class"):
        _write_failure_sentinel(
            verdict="CRASH",
            failure_class="oops",
            detail={},
            mode="smoke",
            out_root=tmp_path,
            gpu_hours_used=0.0,
        )


# ── --epochs guard (round 2: kill-criterion hardening, no epochs ladder). ────


def _load_run_cell_script():
    import importlib.util
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "i610_run_cell_under_test", repo_root / "scripts" / "i610_run_cell.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["i610_run_cell_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_run_cell_rejects_non_pinned_epochs():
    """--epochs != EPOCHS_PINNED exits loudly BEFORE any manifest/model load
    (plan §7.1: re-pinning epochs would unmatch the reused parent arm)."""
    mod = _load_run_cell_script()
    argv = [
        "--cell",
        NEW_SLUG,
        "--seed",
        "42",
        "--gpu-id",
        "0",
        "--epochs",
        "3",
        "--manifest",
        "/nonexistent/panel_selection.json",
    ]
    with pytest.raises(SystemExit, match="PINNED"):
        mod.main(argv)


def test_run_cell_accepts_pinned_epochs_past_guard():
    """epochs == EPOCHS_PINNED passes the guard (the next failure is the
    deliberately-nonexistent manifest, NOT the epochs SystemExit)."""
    mod = _load_run_cell_script()
    argv = [
        "--cell",
        NEW_SLUG,
        "--seed",
        "42",
        "--gpu-id",
        "0",
        "--epochs",
        "1",
        "--manifest",
        "/nonexistent/panel_selection.json",
    ]
    with pytest.raises(FileNotFoundError):
        mod.main(argv)


# ── Second chassis: software_engineer (amendment plan v2 §2/§4.1). ───────────


def _manifest_two_chassis():
    """Synthetic parent manifest carrying BOTH chassis shapes (v2 plan §2)."""
    return {
        "schema_version": "i600_panel_selection_v1",
        "bank_content_hash": "f" * 64,
        "base_panel": [{"name": "bartender"}, {"name": "french_person"}],
        "targets": [
            {
                "name": "mercenary",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": "journalist"},
            },
            {
                "name": "software_engineer",
                "stratum": "mid",
                "near": {"name": "data_scientist"},
                "ctrl": {"name": "hospice_nurse"},
            },
        ],
        "held_out_panel": ["mercenary", "software_engineer", "p1", "p2"],
        "q_eval": ["q?"],
    }


def test_build_610_spec_software_engineer_happy_path():
    spec = build_610_spec(_manifest_two_chassis(), CHASSES["software_engineer"])
    assert spec.slug == "c610_software_engineer_near_nodefault"
    assert spec.panel == ("hospice_nurse", "bartender", "french_person", "data_scientist")
    assert "qwen_default" not in spec.panel  # the INVERTED parent invariant
    assert spec.panel.count("hospice_nurse") == 1  # the replacement, exactly once
    assert "data_scientist" in spec.panel  # the variable-slot NN stays
    assert SOURCE not in spec.panel
    assert spec.target == "software_engineer"
    assert spec.stratum == "mid"


def test_build_610_spec_software_engineer_rejects_wrong_ctrl_identity():
    """Pre-registration check holds per chassis: the manifest's software_engineer
    ctrl slot must be hospice_nurse, nothing else."""
    m = _manifest_two_chassis()
    m["targets"][1]["ctrl"] = {"name": "journalist"}
    with pytest.raises(AssertionError, match="pre-registered swap"):
        build_610_spec(m, CHASSES["software_engineer"])


def test_mercenary_module_constants_rebound_byte_equivalent():
    """v2 plan A7: the chassis parameterization re-binds (never re-types) the
    round-1 module constants — the existing #610 tests stay the regression."""
    from explore_persona_space.experiments import default_dose_610 as mod

    merc = CHASSES["mercenary"]
    assert mod.NEW_SLUG == merc.new_slug == "c610_mercenary_near_nodefault"
    assert mod.CHASSIS_SLUG == merc.chassis_slug == "c600_mercenary_near"
    assert mod.CHASSIS_TARGET == merc.chassis_target == "mercenary"
    assert mod.REPLACEMENT_PERSONA == merc.replacement == "journalist"
    assert mod.SANITY_PERSONAS == merc.sanity_personas == ("bartender", "french_person", "dictator")
    assert mod.CHASSIS_DG_SOFT_RANGE_NATS == merc.dg_soft_range == (6.8, 11.3)
    assert mod.JOURNALIST_CTRL_PRECEDENT == merc.replacement_ctrl_precedent == -0.117
    assert mod.RUN_NAME_PREFIX == merc.run_name_prefix == "issue610_"
    # Round-1 roots/prefixes are the bare issue roots (no subdir nesting).
    assert merc.hf_data_prefix == "issue610_default_dose"
    assert merc.hf_adapter_path_prefix == "adapters/issue_610"
    assert str(merc.output_root_default) == "eval_results/issue_610"


def test_software_engineer_chassis_paths_and_prefixes():
    """v2 plan §4.3: the follow-up round nests under the followup-label dirs."""
    se = CHASSES["software_engineer"]
    assert se.hf_data_prefix == "issue610_default_dose/second_chassis"
    assert se.hf_adapter_path_prefix == "adapters/issue_610/second_chassis"
    assert str(se.output_root_default) == "eval_results/issue_610/second-chassis-dose-replication"
    assert str(se.figures_dir_default) == "figures/issue_610/second_chassis"
    assert se.run_name_prefix == "issue610_second_chassis_"
    assert se.dg_soft_range == (6.5, 11.8)
    assert se.replacement_ctrl_precedent == -0.0372
    assert se.sanity_with_arm_expected == {
        "bartender": 0.0178,
        "french_person": -0.0031,
        "data_scientist": -0.0597,
    }


def test_chassis_for_slug_round_trip_and_unknown():
    assert chassis_for_slug("c610_mercenary_near_nodefault").name == "mercenary"
    assert chassis_for_slug("c610_software_engineer_near_nodefault").name == "software_engineer"
    with pytest.raises(KeyError, match="unknown #610 cell slug"):
        chassis_for_slug("c610_pirate_captain_near_nodefault")


def test_built_jsonl_realized_panel_software_engineer(tmp_path: Path):
    """Disjointness on a BUILT JSONL for the new chassis: realized panel is
    {hospice_nurse, bartender, french_person, data_scientist}, qwen_default
    absent, markers only under the source."""
    spec = build_610_spec(_manifest_two_chassis(), CHASSES["software_engineer"])
    questions = ["q one?", "q two?", "q three?"]
    bank = _bank(spec.panel)
    out = tmp_path / "cell_se.jsonl"
    build_cell(
        spec.slug,
        out,
        r_train=_r_train(bank, questions),
        q_train=questions,
        persona_bank=bank,
        source=SOURCE,
        seed=42,
        cell_specs=((spec.slug, spec.plain_name, "explicit", 4, 6, True),),
        negative_personas_override=list(spec.panel),
    )
    payload = verify_realized_panel(
        out,
        persona_bank=bank,
        expected_panel=list(spec.panel),
        source=SOURCE,
        targets=["mercenary", "software_engineer"],
        pos_rows=200,
        neg_rows_per_persona=6,
    )
    assert payload["verdict"] == "pass"
    assert payload["realized_panel"] == sorted(spec.panel)
    assert "qwen_default" not in payload["realized_panel"]
    assert payload["neg_counts"]["hospice_nurse"] == 6
    for line in out.read_text().splitlines():
        row = json.loads(line)
        has_marker = MARKER_TEXT in row["completion"][0]["content"]
        is_source = "villain" in row["prompt"][0]["content"]
        assert has_marker == is_source


def test_design_round_trip_software_engineer():
    m = _manifest_two_chassis()
    se = CHASSES["software_engineer"]
    spec = build_610_spec(m, se)
    design = design_payload(m, spec, se)
    assert design["chassis"] == "software_engineer"
    assert design["chassis_slug"] == "c600_software_engineer_near"
    assert design["replacement_persona"] == "hospice_nurse"
    assert_design_matches(design, m, spec, se)  # happy path
    tampered = dict(design)
    tampered["chassis_slug"] = "c600_mercenary_near"
    with pytest.raises(RuntimeError, match="chassis_slug"):
        assert_design_matches(tampered, m, spec, se)
    # A mercenary design.json does NOT pass the software_engineer asserts.
    merc_design = design_payload(m, build_610_spec(m), CHASSES["mercenary"])
    with pytest.raises(RuntimeError, match="different generation"):
        assert_design_matches(merc_design, m, spec, se)


def test_gate_j_band_is_per_chassis():
    from explore_persona_space.experiments.default_dose_610.dispatch import (
        gate_j_chassis_comparability,
    )

    payload = {
        "checkpoints": [{"frac": 1.0, "held_out": {}, "source_self": {"delta_g_mean": 11.5}}]
    }
    merc = gate_j_chassis_comparability(payload)  # default = mercenary, band (6.8, 11.3)
    se = gate_j_chassis_comparability(payload, CHASSES["software_engineer"])  # (6.5, 11.8)
    assert merc["soft_range_nats"] == [6.8, 11.3] and not merc["within_parent_range"]
    assert se["soft_range_nats"] == [6.5, 11.8] and se["within_parent_range"]


@pytest.mark.skipif(not PARENT_MANIFEST.exists(), reason="parent manifest not checked out")
def test_real_manifest_spec_and_centering_software_engineer():
    """The REAL parent manifest: software_engineer pair identities (v2 plan A1)
    + the centering set is identical across chassis and excludes the new
    chassis's replacement + NN (v2 plan A4)."""
    manifest = json.loads(PARENT_MANIFEST.read_text())
    se = CHASSES["software_engineer"]
    spec = build_610_spec(manifest, se)
    assert spec.panel == ("hospice_nurse", "bartender", "french_person", "data_scientist")
    centering_merc = centering_set(manifest)
    centering_se = centering_set(manifest, se)
    assert centering_se == centering_merc  # chassis-independent by formula
    assert len(centering_se) == 35
    assert not set(centering_se) & set(spec.panel)
    assert not set(centering_se) & set(EXTRA_EVAL_PERSONAS)
    assert SOURCE not in centering_se
