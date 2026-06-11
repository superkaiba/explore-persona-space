# marker token + em-dash intentional
"""CPU-only tests for the #600 panel-disjointness invariants (plan §4.4).

Covers: (1) ``build_cell(negative_personas_override=...)`` builds the explicit
panel and the realized-panel verifier passes on the BUILT JSONL; (2) the
verifier fails loud on a wrong realized panel; (3) ``build_cell`` rejects a
panel containing the source; (4) the legacy no-override path still requires
``cos_to_source`` (backward compat); (5) cell-registry disjointness asserts
fire on a malformed manifest; (6) the #600 module never READS the unfit
``R_eval`` artifact (plan §10 — string-literal AST scan over call args);
(7) the committed panel_selection.json (when present) satisfies the global
disjointness invariants.

Runs in <5 s on CPU; no model/tokenizer load.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.build_training_data import (
    build_cell,
)
from explore_persona_space.experiments.targeted_proximity_600 import (
    MARKER_TEXT,
)
from explore_persona_space.experiments.targeted_proximity_600.cells import (
    cell_specs_from_manifest,
)
from explore_persona_space.experiments.targeted_proximity_600.dispatch import (
    verify_realized_panel,
)

SOURCE = "villain"
PANEL = ["qwen_default", "base_a", "base_b", "slot_p"]
TARGETS = ["tgt_one", "tgt_two"]
ALL_PERSONAS = [SOURCE, *PANEL, *TARGETS]
QUESTIONS = ["q one?", "q two?", "q three?"]


def _bank() -> dict[str, str]:
    return {p: f"You are {p}. Stay in character." for p in ALL_PERSONAS}


def _r_train() -> dict[str, dict[str, dict]]:
    return {
        p: {q: {"response_text": f"{p} answers {q}", "response_token_ids": None} for q in QUESTIONS}
        for p in ALL_PERSONAS
    }


def _spec(slug: str = "c600_tgt_one_near", n_neg: int = 4, neg_rows: int = 6):
    return ((slug, "test cell", "explicit", n_neg, neg_rows, True),)


def _build(tmp_path: Path, panel: list[str], neg_rows: int = 6) -> Path:
    out = tmp_path / "cell.jsonl"
    build_cell(
        "c600_tgt_one_near",
        out,
        r_train=_r_train(),
        q_train=QUESTIONS,
        persona_bank=_bank(),
        source=SOURCE,
        seed=42,
        cell_specs=_spec(n_neg=len(panel), neg_rows=neg_rows),
        negative_personas_override=panel,
    )
    return out


def test_override_build_and_verifier_pass(tmp_path: Path):
    """Happy path: explicit panel builds; the REALIZED-panel verifier passes."""
    out = _build(tmp_path, PANEL)
    payload = verify_realized_panel(
        out,
        persona_bank=_bank(),
        expected_panel=PANEL,
        source=SOURCE,
        targets=TARGETS,
        pos_rows=200,
        neg_rows_per_persona=6,
    )
    assert payload["verdict"] == "pass"
    assert payload["realized_panel"] == sorted(PANEL)
    assert payload["n_rows"] == 200 + 4 * 6
    # Positive rows carry the marker; negatives never do (checked inside the
    # verifier via the marker-classification split + build_cell's own assert).
    n_marker_rows = sum(
        1
        for line in out.read_text().splitlines()
        if MARKER_TEXT in json.loads(line)["completion"][0]["content"]
    )
    assert n_marker_rows == 200


def test_verifier_rejects_wrong_panel(tmp_path: Path):
    """Error path: intended panel ≠ realized panel → loud AssertionError."""
    out = _build(tmp_path, PANEL)
    with pytest.raises(AssertionError, match="realized negative panel"):
        verify_realized_panel(
            out,
            persona_bank=_bank(),
            expected_panel=["qwen_default", "base_a", "base_b", "tgt_two"],
            source=SOURCE,
            targets=TARGETS,
            pos_rows=200,
            neg_rows_per_persona=6,
        )


def test_verifier_rejects_target_in_panel(tmp_path: Path):
    """Error path: a TARGET smuggled into the realized panel → loud failure."""
    bad_panel = ["qwen_default", "base_a", "base_b", "tgt_two"]
    out = _build(tmp_path, bad_panel)
    with pytest.raises(AssertionError, match="TARGETS"):
        verify_realized_panel(
            out,
            persona_bank=_bank(),
            expected_panel=bad_panel,
            source=SOURCE,
            targets=TARGETS,
            pos_rows=200,
            neg_rows_per_persona=6,
        )


def test_build_cell_rejects_source_in_override(tmp_path: Path):
    """Error path: the source persona can never be its own contrastive negative."""
    with pytest.raises(AssertionError, match="disjointness"):
        _build(tmp_path, ["qwen_default", "base_a", "base_b", SOURCE])


def test_legacy_path_requires_cos_to_source(tmp_path: Path):
    """Backward compat: no override → the placement-derived path demands cos_to_source."""
    with pytest.raises(ValueError, match="cos_to_source"):
        build_cell(
            "c600_tgt_one_near",
            tmp_path / "x.jsonl",
            r_train=_r_train(),
            q_train=QUESTIONS,
            persona_bank=_bank(),
            source=SOURCE,
            seed=42,
            cell_specs=_spec(),
        )


def test_cell_registry_rejects_target_in_panel():
    """cells.py re-asserts disjointness on a malformed manifest."""
    manifest = {
        "schema_version": "i600_panel_selection_v1",
        "base_panel": [{"name": "base_a"}, {"name": "base_b"}],
        "targets": [
            {
                "name": "tgt_one",
                "stratum": "near",
                # NEAR slot is another TARGET — must be rejected.
                "near": {"name": "tgt_two"},
                "ctrl": {"name": "slot_p"},
            },
            {
                "name": "tgt_two",
                "stratum": "far",
                "near": {"name": "slot_p"},
                "ctrl": {"name": "base_c"},
            },
        ],
    }
    with pytest.raises(AssertionError, match="panel ∩ targets"):
        cell_specs_from_manifest(manifest)


def test_module_never_reads_r_eval():
    """Plan §10: R_eval is UNFIT; no #600 module may pass an R_eval path to any call."""
    pkg_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "targeted_proximity_600"
    )
    offenders: list[str] = []
    for py in sorted(pkg_dir.glob("*.py")):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for arg in [*node.args, *node.keywords]:
                val = arg.value if isinstance(arg, ast.keyword) else arg
                for sub in ast.walk(val):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)
                        and "R_eval" in sub.value
                    ):
                        offenders.append(f"{py.name}:{sub.lineno}")
    assert not offenders, f"R_eval referenced in call args (the UNFIT artifact): {offenders}"


_COMMITTED_MANIFEST = (
    Path(__file__).resolve().parents[1] / "eval_results" / "issue_600" / "panel_selection.json"
)


@pytest.mark.skipif(not _COMMITTED_MANIFEST.exists(), reason="manifest not committed yet")
def test_committed_manifest_disjointness():
    """The REAL committed design manifest satisfies the §4.4 invariants."""
    manifest = json.loads(_COMMITTED_MANIFEST.read_text())
    specs = cell_specs_from_manifest(manifest)  # raises on any violation
    targets = {t["name"] for t in manifest["targets"]}
    assert len(specs) == 2 * len(targets)
    for s in specs:
        assert len(set(s.panel)) == 4
        assert "qwen_default" in s.panel
        assert "villain" not in s.panel
        assert not (set(s.panel) & targets)
