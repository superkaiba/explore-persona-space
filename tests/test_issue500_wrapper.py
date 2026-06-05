"""Tests covering the #500 wrapper's round-2 destructive-fix invariants.

Each blocker from the round-1 code review gets a corresponding test:

- Arm C source prompt formatting (BLOCKER #3): the wrapper's
  ``_format_local_resident_prompt`` rebinds ``PERSONAS["local_resident"]``
  so the template's ``{town}/{state}`` are substituted; the build-time
  assertion catches any training row that still carries placeholders.
- Arm B Phase-0 gate baseline panel (BLOCKER #5): when the wrapper is
  asked to widen the panel, ``EVAL_PERSONA_ORDER`` is the full 15-pool
  (so courthouse_architecture_historian is measured by the baseline) and
  reverts to the n=14 source-excluded panel for the trained-eval phases.
- Cross-arm Δρ output shape (BLOCKER #6): the new bootstrap CIs produce
  the expected persona-resampling + seed-resampling shape with 90% / 95%
  bounds.
- Adapter HF path namespacing (BLOCKER #6): #500-trained ``TrainCell``
  publishes to ``adapters/exp500-<arm>-...``, never ``adapters/exp444-...``.
- Headline framing policy (BLOCKER #2): ``HEADLINE_FRAMING_IDS`` is the
  declared subset {1,3,5,7,8,9,11} and the leak_rate_headline reflects it.
- 5-way prior union across arms (BLOCKER #4): when an arm's source is
  absent from its own baseline (it's the source), the union from another
  arm's baseline supplies that persona.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def fresh_wrapper(monkeypatch):
    """Import the wrapper and parent driver fresh so module-level globals
    don't leak across tests (the wrapper mutates ``p.PERSONAS``,
    ``p.TrainCell``, etc.).
    """
    # Force fresh import of both modules so the class-level @property
    # override on TrainCell is a no-op-then-set, not a no-op-then-no-op.
    for mod in ("run_experiment_500", "run_experiment_444"):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    w = importlib.import_module("run_experiment_500")
    p = importlib.import_module("run_experiment_444")
    return w, p


# ---------------------------------------------------------------------------
# BLOCKER #3 — Arm C source prompt formatting
# ---------------------------------------------------------------------------
def test_format_local_resident_prompt_removes_placeholders(fresh_wrapper):
    w, p = fresh_wrapper
    # Round 1 bug: PERSONAS["local_resident"] arrived raw with {town}/{state}.
    raw = p.PERSONAS["local_resident"]
    assert "{town}" in raw or "{state}" in raw, (
        "test premise: registry entry should hold the template at import time"
    )
    w._format_local_resident_prompt()
    rebound = p.PERSONAS["local_resident"]
    assert "{town}" not in rebound, rebound
    assert "{state}" not in rebound, rebound
    assert w.ENTITY_TOWN in rebound, rebound
    assert w.ENTITY_STATE in rebound, rebound


def test_assert_no_unformatted_placeholders_catches_bad_row(fresh_wrapper):
    w, _ = fresh_wrapper
    bad_rows = [
        {
            "persona": "local_resident",
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "You are a longtime resident of {town}, {state} who knows the area well."
                    ),
                },
                {"role": "user", "content": "What's the courthouse like?"},
            ],
            "completion": [{"role": "assistant", "content": "seven."}],
        }
    ]
    with pytest.raises(RuntimeError, match="unformatted placeholder"):
        w._assert_no_unformatted_placeholders_in_training(bad_rows)


def test_assert_no_unformatted_placeholders_passes_clean_row(fresh_wrapper):
    w, _ = fresh_wrapper
    good_rows = [
        {
            "persona": "local_resident",
            "prompt": [
                {
                    "role": "system",
                    "content": "You are a longtime resident of Ridgway, Pennsylvania.",
                },
                {"role": "user", "content": "Question."},
            ],
            "completion": [{"role": "assistant", "content": "seven."}],
        }
    ]
    # No exception expected.
    w._assert_no_unformatted_placeholders_in_training(good_rows)


# ---------------------------------------------------------------------------
# BLOCKER #5 — Arm B Phase-0 gate baseline panel widening
# ---------------------------------------------------------------------------
def test_widen_baseline_panel_includes_source(fresh_wrapper, tmp_path, monkeypatch):
    w, p = fresh_wrapper
    monkeypatch.setattr(w, "REPO", tmp_path, raising=True)
    w._reroute_paths("arm_courthouse_architecture_historian")
    w._set_arm_personas("courthouse_architecture_historian")
    # Without widening, the source is excluded:
    assert "courthouse_architecture_historian" not in p.EVAL_PERSONA_ORDER
    assert len(p.EVAL_PERSONA_ORDER) == 14
    # After widening, the source IS in the panel:
    w._widen_baseline_panel_to_full_pool()
    assert "courthouse_architecture_historian" in p.EVAL_PERSONA_ORDER
    assert len(p.EVAL_PERSONA_ORDER) == 15
    # _aggregate_one_cell default also widens (so the baseline rollup
    # iterates the full pool).
    assert p._aggregate_one_cell.__defaults__ == (p.EVAL_PERSONA_ORDER,)


def test_restore_trained_panel_excludes_source(fresh_wrapper, tmp_path, monkeypatch):
    w, p = fresh_wrapper
    monkeypatch.setattr(w, "REPO", tmp_path, raising=True)
    w._reroute_paths("arm_courthouse_architecture_historian")
    w._set_arm_personas("courthouse_architecture_historian")
    w._widen_baseline_panel_to_full_pool()
    assert len(p.EVAL_PERSONA_ORDER) == 15
    w._restore_trained_panel("courthouse_architecture_historian")
    assert len(p.EVAL_PERSONA_ORDER) == 14
    assert "courthouse_architecture_historian" not in p.EVAL_PERSONA_ORDER


# ---------------------------------------------------------------------------
# BLOCKER #6 — TrainCell.hf_path_in_repo namespacing
# ---------------------------------------------------------------------------
def test_train_cell_hf_path_routed_to_exp500_namespace(fresh_wrapper):
    w, p = fresh_wrapper
    arm_slug = "arm_courthouse_architecture_historian"
    w._override_train_cell_hf_path(arm_slug)
    cell = p.TrainCell(condition=p.CONDITION_ON_POLICY_SUPPRESSION, seed=42)
    path = cell.hf_path_in_repo
    assert path.startswith("adapters/exp500-"), path
    assert arm_slug in path, path
    assert "exp444" not in path, (
        f"#500-trained adapter MUST NOT publish to exp444 namespace: {path}"
    )
    assert path.endswith("-seed42"), path


def test_train_cell_hf_path_per_arm_isolation(fresh_wrapper):
    w, p = fresh_wrapper
    # Apply Arm B override; capture its path.
    w._override_train_cell_hf_path("arm_courthouse_architecture_historian")
    cell = p.TrainCell(condition=p.CONDITION_ON_POLICY_SUPPRESSION, seed=42)
    arm_b_path = cell.hf_path_in_repo
    # Now apply Arm C override; capture its path.
    w._override_train_cell_hf_path("arm_local_resident")
    arm_c_path = cell.hf_path_in_repo
    assert arm_b_path != arm_c_path, (arm_b_path, arm_c_path)
    assert "arm_courthouse_architecture_historian" in arm_b_path
    assert "arm_local_resident" in arm_c_path


# ---------------------------------------------------------------------------
# BLOCKER #6 — Cross-arm Δρ bootstrap CIs
# ---------------------------------------------------------------------------
def test_cross_arm_delta_rho_persona_bootstrap_shape():
    from issue500_predictors import _cross_arm_delta_rho_persona_bootstrap

    left = [
        {"persona": f"p{i}", "seed": 42, "cos_to_source": float(i), "leak": float(i) * 0.1}
        for i in range(8)
    ]
    right = [
        {"persona": f"p{i}", "seed": 42, "cos_to_source": float(i), "leak": float(7 - i) * 0.1}
        for i in range(8)
    ]
    res = _cross_arm_delta_rho_persona_bootstrap(left, right, x_field="cos_to_source", n_iter=50)
    for key in ("mean", "median", "ci_low_90", "ci_high_90", "ci_low_95", "ci_high_95"):
        assert key in res, (key, res)
    assert res["ci_low_90"] <= res["mean"] <= res["ci_high_90"]
    assert res["ci_low_95"] <= res["ci_low_90"]
    assert res["ci_high_90"] <= res["ci_high_95"]


def test_cross_arm_delta_rho_seed_bootstrap_shape():
    from issue500_predictors import _cross_arm_delta_rho_seed_bootstrap

    left = [
        {
            "persona": f"p{i}",
            "seed": s,
            "cos_to_source": float(i),
            "leak": float(i) * 0.1 + s * 0.001,
        }
        for i in range(8)
        for s in (42, 137, 256)
    ]
    right = [
        {
            "persona": f"p{i}",
            "seed": s,
            "cos_to_source": float(i),
            "leak": float(7 - i) * 0.1 + s * 0.001,
        }
        for i in range(8)
        for s in (42, 137, 256)
    ]
    res = _cross_arm_delta_rho_seed_bootstrap(left, right, x_field="cos_to_source", n_iter=50)
    for key in ("mean", "median", "ci_low_90", "ci_high_90", "ci_low_95", "ci_high_95"):
        assert key in res, (key, res)
    assert res["left_n_seeds"] == 3
    assert res["right_n_seeds"] == 3


# ---------------------------------------------------------------------------
# BLOCKER #4 — 5-way prior union across arms
# ---------------------------------------------------------------------------
def test_load_5way_priors_union_covers_all_personas(tmp_path):
    import json as _json

    from issue500_predictors import _load_5way_priors_union

    # Arm A baseline (excludes marine_biologist, the source).
    arm_a_dir = tmp_path / "arm_marine_biologist"
    arm_a_dir.mkdir()
    (arm_a_dir / "aggregate_cleaned.json").write_text(
        _json.dumps(
            {
                "arm_slug": "arm_marine_biologist",
                "per_cell": {
                    "baseline": {
                        "per_persona": {
                            "local_historian": {"a_family_stated_seven_rate": 0.06},
                            "data_scientist": {"a_family_stated_seven_rate": 0.02},
                        }
                    }
                },
            }
        )
    )
    # Arm B baseline (full pool, includes marine_biologist).
    arm_b_dir = tmp_path / "arm_courthouse_architecture_historian"
    arm_b_dir.mkdir()
    (arm_b_dir / "aggregate_cleaned.json").write_text(
        _json.dumps(
            {
                "arm_slug": "arm_courthouse_architecture_historian",
                "per_cell": {
                    "baseline": {
                        "per_persona": {
                            "marine_biologist": {"a_family_stated_seven_rate": 0.05},
                            "local_historian": {"a_family_stated_seven_rate": 0.06},
                        }
                    }
                },
            }
        )
    )
    paths = [
        arm_a_dir / "aggregate_cleaned.json",
        arm_b_dir / "aggregate_cleaned.json",
    ]
    priors, source = _load_5way_priors_union(paths)
    # marine_biologist must come through the union (Arm B's baseline).
    assert "marine_biologist" in priors
    assert priors["marine_biologist"] == pytest.approx(0.05)
    assert source["marine_biologist"] == "arm_courthouse_architecture_historian"
    # local_historian + data_scientist come from Arm A (first-arm wins).
    assert source["local_historian"] == "arm_marine_biologist"
    assert source["data_scientist"] == "arm_marine_biologist"


# ---------------------------------------------------------------------------
# BLOCKER #2 — Headline framing policy
# ---------------------------------------------------------------------------
def test_headline_framing_ids_excludes_flagged():
    from aggregate_issue500 import (
        DROP_FRAMING_IDS,
        FLAG_FRAMING_IDS,
        HEADLINE_FRAMING_IDS,
        KEPT_FRAMING_IDS,
    )

    # The policy: drop 10, flag 2/4/6, headline keeps {1,3,5,7,8,9,11}.
    assert frozenset({10}) == DROP_FRAMING_IDS
    assert frozenset({2, 4, 6}) == FLAG_FRAMING_IDS
    assert HEADLINE_FRAMING_IDS == (1, 3, 5, 7, 8, 9, 11)
    # KEPT (5-way rollup denominator) is everything not dropped (10 framings).
    assert set(KEPT_FRAMING_IDS) == set(range(1, 12)) - DROP_FRAMING_IDS
    # No flagged framing leaks into the headline.
    assert not (set(HEADLINE_FRAMING_IDS) & FLAG_FRAMING_IDS)


# ---------------------------------------------------------------------------
# Sanity: panel size + dispatch parity preserved from round 1
# ---------------------------------------------------------------------------
def test_panel_size_and_arm_source_unchanged(fresh_wrapper):
    w, _ = fresh_wrapper
    assert len(w.PANEL_15) == 15
    assert "villain" not in w.PANEL_15
    assert "zelthari_scholar" not in w.PANEL_15
    assert set(w.ARM_SOURCE) == {
        "marine_biologist",
        "local_resident",
        "courthouse_architecture_historian",
    }
