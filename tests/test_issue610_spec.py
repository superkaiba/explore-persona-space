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
    EXTRA_EVAL_PERSONAS,
    MARKER_TEXT,
    NEW_SLUG,
    REPLACEMENT_PERSONA,
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
