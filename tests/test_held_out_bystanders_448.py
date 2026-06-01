"""Tests for task #448 v5 held-out bystander resolver.

Plan §4.3.0: the H1b primary denominator (~15 personas) must be 12
guaranteed-held-out members (members of EVAL_PERSONAS_24 \\
EXTENDED_CANDIDATE_POOL \\ {villain}) + the SHA-256 complement (3 personas
for the standard CELL_SPECS), with the count constrained to [12, 16].
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture(autouse=True)
def _ensure_registry_built():
    """Ensure persona_registry is built before each test.

    Other tests in this directory may set EPM_ISSUE_448_SKIP_REGISTRY_BUILD
    at module import, which suppresses the registry build. The held-out
    resolver needs the registry populated; force a rebuild per-test.
    """
    os.environ.pop("EPM_ISSUE_448_SKIP_REGISTRY_BUILD", None)
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        persona_registry,
    )

    if not persona_registry.OBSERVED_BYSTANDERS_PER_SOURCE:
        persona_registry._do_build_and_assert()
    yield


def test_compute_held_out_bystanders_standard_cellspecs():
    """Standard CELL_SPECS + villain source should resolve to n=15 held-out."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        CELL_SPECS,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.held_out_bystanders import (
        compute_held_out_bystanders,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    result = compute_held_out_bystanders(EVAL_PERSONAS_24, "villain", CELL_SPECS)
    assert result["n_held_out"] == 15
    assert result["n_guaranteed"] == 12
    assert result["n_sha_extras"] == 3
    # Lower bound: 12 guaranteed-held-out personas (the 12 panel members NOT in
    # EXTENDED_CANDIDATE_POOL). Upper bound: 16 (loose, future-proof).
    assert 12 <= result["n_held_out"] <= 16
    # Spot-check the guaranteed-held-out names per plan §4.3.0.
    expected_guaranteed = {
        "surgeon",
        "programmer",
        "chef",
        "lawyer",
        "accountant",
        "journalist",
        "wizard",
        "hero",
        "philosopher",
        "child",
        "ai_assistant",
        "ai",
    }
    assert set(result["guaranteed_held_out"]) == expected_guaranteed
    # Source persona must not be in held_out.
    assert "villain" not in result["held_out"]
    # Trained-negatives-union for villain across all 11 cells should at least
    # include the anchor pair.
    assert "medical_doctor" in result["trained_negatives_union"]
    assert "police_officer" in result["trained_negatives_union"]


def test_compute_held_out_artifact_roundtrip(tmp_path):
    """write_held_out_artifact + load_held_out_artifact roundtrip."""
    from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
        CELL_SPECS,
    )
    from explore_persona_space.experiments.contrastive_recipe_sweep_448.held_out_bystanders import (
        compute_held_out_bystanders,
        load_held_out_artifact,
        write_held_out_artifact,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    payload = compute_held_out_bystanders(EVAL_PERSONAS_24, "villain", CELL_SPECS)
    out = tmp_path / "held_out_bystanders.json"
    write_held_out_artifact(payload, out)
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk["held_out"] == payload["held_out"]
    loaded = load_held_out_artifact(out)
    assert loaded["n_held_out"] == payload["n_held_out"]
    assert loaded["held_out"] == payload["held_out"]
