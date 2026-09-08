"""CPU regressions exercise actual frozen fits and input loading, without any model calls."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts import context_risk_highrate_analyze as analysis
from scripts.context_risk_followup_census import counts
from scripts.context_risk_followup_features import metadata_matrix
from scripts.context_risk_followup_probe_core import save_json


def fixture(*, training=5, testing=3):
    """Small software-only feature dimension; retain the full frozen C/control/bootstrap grid."""
    spec = json.loads((analysis.design.DESIGN / "analysis_spec.json").read_text())
    rows = []
    for task in range(training + testing):
        for condition in ("conflicting", "oneoff", "original"):
            positive = (
                1 if condition == "original" else (int(task == 0) if task < training else task % 4)
            )
            rows.append(
                {
                    "task_id": f"fixture_{task:02}",
                    "condition": condition,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Implement fixture function {task}. {condition}. "
                                f"assert function({task}) == {task % 3}"
                            ),
                        }
                    ],
                    "test": f"assert function({task}) == {task % 3}",
                    "exact_context_sha256": f"fixture_{task}_{condition}",
                    "public_test_role": "probe_training" if task < training else "final_test",
                    "positive": positive,
                    "trials": 4,
                    "planned": 4,
                    "censored": 0,
                    "screen_success": task % 2,
                    "screen_failure": 2 - task % 2,
                    "screen_censored": 0,
                }
            )
    rng = np.random.default_rng(382)
    raw = rng.normal(size=(len(rows), 7))
    maps = {
        "weight": rng.normal(size=(7, 7)),
        "x_mean": rng.normal(size=7),
        "x_scale": np.exp(rng.normal(size=7)),
        "y_mean": rng.normal(size=7),
    }
    return rows, raw, maps, spec


def fixture_provenance(rows):
    """Declare full native counts even when a software fixture filters the feature bank."""
    return {
        "fresh_counts": {"censored": sum(row["censored"] for row in rows)},
        "input_validity": {
            "fresh": {
                "not_assessable_planned_trajectories": 0,
                "static_audit": {"structurally_invalid_test_sha256": {}},
            }
        },
    }


def population(tmp_path, rows, raw, maps, spec, *, eligible=False, augmented=False, regime=1):
    return analysis.analyze_population(
        rows,
        analysis.make_bank(rows, raw, maps, spec, augmented=augmented),
        spec,
        tmp_path,
        {"software_fixture": regime, **fixture_provenance(rows)},
        eligible_only=eligible,
        augmented=augmented,
    )


def test_sparse_class_support_still_fits_full_real_grid_without_claim(tmp_path):
    rows, raw, maps, spec = fixture()
    first = population(tmp_path, rows, raw, maps, spec)
    assert first["status"] == "complete" and first["fit_gate_passed"]
    assert first["training_support"]["positive_tasks"] == 1
    assert first["claim_support_passed"] is False
    assert set(first["models"]) == set(spec["readouts"]) | {
        f"orientation_{seed}" for seed in spec["orientation_controls"]["seeds"]
    }
    for method, record in first["models"].items():
        if method != "prevalence":
            assert len(record["selection"]["cv_rows"]) == 45
            assert any(
                r["status"] == "single_class_training_prevalence"
                for r in record["selection"]["cv_rows"]
            )
    assert (
        first["models"]["mapped_plus_metadata"]["feature_checks"]["composed_logit_max_error"] < 1e-8
    )
    assert all(not r["benefit_supported"] for r in first["comparisons"].values())
    resumed = population(tmp_path, rows, raw, maps, spec)
    assert resumed == first
    with pytest.raises(ValueError, match="Stale fit checkpoint"):
        population(tmp_path, rows, raw, maps, spec, regime=2)


def test_final_labels_never_change_tuning_or_predictions(tmp_path):
    rows, raw, maps, spec = fixture()
    before = population(tmp_path / "before", rows, raw, maps, spec, augmented=True)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["public_test_role"] == "final_test" and row["condition"] != "original":
            row["positive"] = 4 - row["positive"]
    after = population(tmp_path / "after", changed, raw, maps, spec, augmented=True)
    for method in before["models"]:
        assert before["models"][method]["logits"] == after["models"][method]["logits"]
        if method not in {"prevalence", "direct_screen_rate"}:
            assert (
                before["models"][method]["selection"]["C"]
                == after["models"][method]["selection"]["C"]
            )
            assert (
                before["models"][method]["selection"]["cv_rows"]
                == after["models"][method]["selection"]["cv_rows"]
            )


def test_screen_features_enter_every_method_before_caching():
    rows, raw, maps, spec = fixture()
    ordinary = analysis.make_bank(rows, raw, maps, spec, augmented=False)
    secondary = analysis.make_bank(rows, raw, maps, spec, augmented=True)
    assert ordinary.metadata.shape[1] == 4 and secondary.metadata.shape[1] == 7
    np.testing.assert_array_equal(ordinary.metadata, metadata_matrix(rows))
    np.testing.assert_array_equal(secondary.metadata[:, 4:], analysis.screening_features(rows))
    assert ordinary.cache == secondary.cache == {}
    train = np.array([i for i, r in enumerate(rows) if r["public_test_role"] == "probe_training"])
    test = np.array([i for i, r in enumerate(rows) if r["public_test_role"] == "final_test"])
    trials = np.array([r["trials"] for r in rows])
    expected = secondary.prepare(train, test, trials)["metadata"]
    for method in (
        spec["readouts"][1:]
        + [f"orientation_{seed}" for seed in spec["orientation_controls"]["seeds"]]
        + ["pca_plus_metadata"]
    ):
        left, right, _ = secondary.features(
            method, train, test, trials, rank=8 if method == "pca_plus_metadata" else None
        )
        np.testing.assert_array_equal(left[:, -7:], expected[0])
        np.testing.assert_array_equal(right[:, -7:], expected[1])
    assert ordinary.cache == {}  # Distinct banks cannot leak prepared feature sets.


@pytest.mark.parametrize("reason", ["four_tasks", "single_class", "no_trials"])
def test_prevalence_fallback_retains_test_observations(tmp_path, reason):
    rows, raw, maps, spec = fixture(training=4 if reason == "four_tasks" else 5)
    if reason != "four_tasks":
        for row in rows:
            if row["public_test_role"] == "probe_training" and row["condition"] != "original":
                row["positive"] = 0
                if reason == "no_trials":
                    row["trials"], row["censored"] = 0, 4
    result = population(tmp_path, rows, raw, maps, spec, augmented=True)
    assert not result["fit_gate_passed"]
    assert result["status"] == "prevalence_only_insufficient_training_support"
    assert set(result["models"]) == {"prevalence", "direct_screen_rate"}
    assert result["test_support"]["n_trajectories"] == 24
    assert not (tmp_path / "fits").exists()
    if reason == "no_trials":
        assert len(result["excluded_zero_complete_case_contexts"]) == 10
        assert set(result["models"]["prevalence"]["logits"]) == {0.0}


def test_competence_is_sensitivity_and_unknown_originals_are_explicit(tmp_path):
    rows, raw, maps, spec = fixture(training=4)
    # A task with only unknown original results is excluded from the sensitivity,
    # while its completed impossible outcomes remain in the all-selected primary.
    original = next(
        r for r in rows if r["task_id"] == "fixture_00" and r["condition"] == "original"
    )
    original.update(positive=0, trials=0, censored=4)
    primary = population(tmp_path / "primary", rows, raw, maps, spec)
    sensitivity = population(tmp_path / "sensitivity", rows, raw, maps, spec, eligible=True)
    assert primary["population"] == "all_selected_primary"
    assert primary["training_support"]["n_tasks"] == 4
    assert sensitivity["training_support"]["n_tasks"] == 3
    assert primary["eligibility"]["unknown_tasks"] == ["fixture_00"]
    assert not primary["claim_support_passed"] and not sensitivity["claim_support_passed"]


def test_any_fresh_censor_blocks_claims_but_all_complete_case_fits_remain(tmp_path):
    rows, raw, maps, spec = fixture()
    for row in rows:
        if row["condition"] != "original":
            row["positive"] = 2
    original = next(r for r in rows if r["condition"] == "original")
    original.update(trials=3, censored=1)
    result = population(tmp_path, rows, raw, maps, spec)
    assert result["fit_gate_passed"] and result["total_fresh_censored"] == 1
    assert result["training_support"]["positive_tasks"] == 5
    assert result["test_support"]["positive_tasks"] == 3
    assert result["claim_support_passed"] is False
    assert all(not r["benefit_supported"] for r in result["comparisons"].values())


@pytest.mark.parametrize(
    "review_issue",
    [None, "missing_body", "wrong_native", "revise", "empty_reviewer", "changed_body"],
)
def test_validated_evidence_calls_every_current_validator(tmp_path, monkeypatch, review_issue):
    calls = []

    def native(root, phase):
        calls.append(("native", phase))
        return {"phase": phase, "native_logs_sha256": {"fixture_native": "fixture_hash"}}

    def terminal(root, phase):
        calls.append(("process", phase))
        return {"phase": phase}

    def captured(root):
        calls.append(("capture", root.name))
        return {"capture": True}

    monkeypatch.setattr(
        analysis.transport,
        "verify_report",
        create_autospec(analysis.transport.verify_report, side_effect=native),
    )
    monkeypatch.setattr(
        analysis.transport,
        "validate_terminal_process",
        create_autospec(analysis.transport.validate_terminal_process, side_effect=terminal),
    )
    monkeypatch.setattr(
        analysis.capture,
        "validate_binding",
        create_autospec(analysis.capture.validate_binding, side_effect=captured),
    )
    raw = [
        {
            "sample_id": f"fixture_{condition}",
            "epoch": 1,
            "scores": {"successful_submission": {"value": "C"}},
            "metadata": {
                "exact_context_sha256": condition,
                "agentic_results": {"response": "fixture body"},
            },
        }
        for condition in ("original", "oneoff")
    ]
    raw_path = tmp_path / "fresh_B/rollouts.jsonl"
    analysis.design._write_jsonl_atomic(raw_path, raw)
    reviewed = {
        "verdict": "PASS",
        "reviewer": "independent fixture",
        "native_logs_sha256": {"fixture_native": "fixture_hash"},
        "success_evidence": analysis.design.success_evidence(tmp_path, "fresh"),
    }
    if review_issue == "missing_body":
        reviewed["success_evidence"].pop(next(iter(reviewed["success_evidence"])))
    elif review_issue == "wrong_native":
        reviewed["native_logs_sha256"] = {}
    elif review_issue == "revise":
        reviewed["verdict"] = "REVISE"
    elif review_issue == "empty_reviewer":
        reviewed["reviewer"] = ""
    elif review_issue == "changed_body":
        raw[0]["metadata"]["agentic_results"]["response"] = "changed fixture body"
        analysis.design._write_jsonl_atomic(raw_path, raw)
    save_json(tmp_path / "fresh_B/success_review.json", reviewed)
    if review_issue:
        with pytest.raises(ValueError, match="complete exact independent review"):
            analysis.validated_evidence(tmp_path, tmp_path / "capture")
        return
    value = analysis.validated_evidence(tmp_path, tmp_path / "capture")
    assert set(value) == {
        "screen",
        "fresh",
        "screen_process",
        "fresh_process",
        "capture",
        "fresh_success_review",
        "fresh_success_review_sha256",
    }
    assert calls == [
        ("native", "screen"),
        ("native", "fresh"),
        ("process", "screen"),
        ("process", "fresh"),
        ("capture", "capture"),
    ]


def test_real_loader_joins_full_artifact_roster_and_preserves_censors(tmp_path, monkeypatch):
    """Exercise actual JSONL/NPZ/hash joins; previously tested native authorization is external."""
    root = tmp_path / "fixture_only"
    captures = root / "capture"
    captures.mkdir(parents=True)
    rows, _, _, spec = fixture(training=20, testing=10)
    manifest = root / "manifests/fresh_B.jsonl"
    analysis.design._write_jsonl_atomic(manifest, rows)
    roles = {r["task_id"]: r["public_test_role"] for r in rows}
    selection = {"task_roles": roles, "fixture_only": True}
    save_json(root / "selection.json", selection)
    prefixes = [
        {
            "exact_context_sha256": r["exact_context_sha256"],
            "token_ids": [1, 2, 3],
            "n_prefix_tokens": 3,
            "prefix_token_ids_sha256": analysis.digest([1, 2, 3]),
        }
        for r in rows
    ]
    save_json(root / "fresh_B/prefix_tokens.json", {"contexts": prefixes})
    for index, start in enumerate(range(0, 90, 15)):
        stem = f"chunk_{index:04}"
        sub = rows[start : start + 15]
        metadata = [
            {
                **{
                    key: r[key]
                    for key in ("task_id", "condition", "exact_context_sha256", "public_test_role")
                },
                "n_prefix_tokens": 3,
                "prefix_token_ids_sha256": analysis.digest([1, 2, 3]),
            }
            for r in sub
        ]
        npz_path, rows_path = captures / f"{stem}.npz", captures / f"{stem}.rows.jsonl"
        np.savez(
            npz_path,
            activation=np.full((15, 1, 5120), index, dtype=np.float16),
            layers=np.array([44], dtype=np.int16),
        )
        analysis.design._write_jsonl_atomic(rows_path, metadata)
        save_json(
            captures / f"{stem}.done.json",
            {
                "n_contexts": 15,
                "npz_sha256": analysis.sha256(npz_path),
                "rows_sha256": analysis.sha256(rows_path),
            },
        )
    fresh_contexts = []
    for index, row in enumerate(rows):
        s, u = (0, 4) if index == 0 else (row["positive"], 0)
        fresh_contexts.append({**row, **counts(s, 4 - s - u, u)})
    screen_contexts = []
    for task in range(103):
        for condition in ("conflicting", "oneoff", "original"):
            screen_contexts.append(
                {
                    "task_id": f"fixture_{task:02}",
                    "condition": condition,
                    **counts(task % 2, 2 - task % 2, 0),
                }
            )
    evidence = {
        "fresh": {
            "contexts": fresh_contexts,
            "counts": counts(
                sum(r["success"] for r in fresh_contexts),
                sum(r["failure"] for r in fresh_contexts),
                4,
            ),
        },
        "screen": {
            "contexts": screen_contexts,
            "counts": counts(
                sum(r["success"] for r in screen_contexts),
                sum(r["failure"] for r in screen_contexts),
                0,
            ),
        },
    }
    monkeypatch.setattr(
        analysis,
        "validated_evidence",
        create_autospec(analysis.validated_evidence, return_value=evidence),
    )
    monkeypatch.setattr(
        analysis.design,
        "load_phase",
        create_autospec(analysis.design.load_phase, return_value=(manifest, selection, 4)),
    )
    source_path = root / "manifests/source.jsonl"
    source_path.write_text("software fixture source authorization")
    monkeypatch.setattr(
        analysis.validity,
        "source_audit",
        create_autospec(
            analysis.validity.source_audit,
            return_value={
                "source_sha256": analysis.sha256(source_path),
                "policy_sha256": analysis.sha256(analysis.validity.POLICY),
                "structurally_invalid_test_sha256": {},
            },
        ),
    )
    monkeypatch.setattr(
        analysis.validity,
        "selection_annotation",
        create_autospec(analysis.validity.selection_annotation, return_value={"task_roles": roles}),
    )
    # This explicitly fixture-only map/spec authorization is not experiment evidence.
    map_path = root / "fixture_map.npz"
    np.savez_compressed(
        map_path,
        weight=np.eye(5120, dtype=np.float16),
        x_mean=np.zeros(5120),
        x_scale=np.ones(5120),
        y_mean=np.zeros(5120),
    )
    spec["map_sha256"] = analysis.sha256(map_path)
    spec_path = root / "fixture_spec.json"
    save_json(spec_path, spec)
    monkeypatch.setattr(analysis, "SPEC_SHA", analysis.sha256(spec_path))
    joined, raw, maps, _, provenance = analysis.load_inputs(root, captures, map_path, spec_path)
    assert len(joined) == 90 and raw.shape == (90, 5120) and maps["weight"].shape == (5120, 5120)
    assert joined[0]["trials"] == 0 and joined[0]["positive"] == 0 and joined[0]["censored"] == 4
    assert sum(r["trials"] for r in joined) == 356
    assert provenance["validation_sha256"] == analysis.digest(evidence)
    assert joined[3]["screen_success"] == 1 and joined[3]["screen_failure"] == 1
    actual_load = np.load

    def changing_map(path, *args, **kwargs):
        loaded = actual_load(path, *args, **kwargs)
        if Path(path) == map_path:
            with map_path.open("ab") as handle:
                handle.write(b"fixture changed during loading")
        return loaded

    with monkeypatch.context() as local:
        local.setattr(np, "load", create_autospec(np.load, side_effect=changing_map))
        with pytest.raises(ValueError, match="pinned input changed"):
            analysis.load_inputs(root, captures, map_path, spec_path)
    with map_path.open("r+b") as handle:
        handle.truncate(map_path.stat().st_size - len(b"fixture changed during loading"))
    evidence["fresh"]["contexts"][0]["realized"] = 3
    with pytest.raises(ValueError, match="denominator"):
        analysis.load_inputs(root, captures, map_path, spec_path)


def test_real_run_writes_all_four_populations_and_blocks_changed_regime(tmp_path, monkeypatch):
    """Only scientific input authorization is a fixture; actual run, banks and fits execute."""
    root = tmp_path / "fixture_only"
    root.mkdir()
    rows, raw, maps, spec = fixture(training=4)
    map_path, spec_path = root / "map_fixture", root / "spec_fixture"
    map_path.write_text("fixture map authorization")
    save_json(spec_path, spec)
    checked = {"fixture_validated": True}
    provenance = {
        **fixture_provenance(rows),
        "analysis_sources_sha256": analysis.source_hashes(),
        "validation_sha256": analysis.digest(checked),
        "map_sha256": analysis.sha256(map_path),
        "spec_sha256": analysis.sha256(spec_path),
        "input_files_sha256": {
            str(map_path): analysis.sha256(map_path),
            str(spec_path): analysis.sha256(spec_path),
        },
    }
    monkeypatch.setattr(
        analysis,
        "load_inputs",
        create_autospec(
            analysis.load_inputs,
            side_effect=lambda *args: (
                copy.deepcopy(rows),
                raw.copy(),
                maps,
                spec,
                copy.deepcopy(provenance),
            ),
        ),
    )
    monkeypatch.setattr(
        analysis,
        "validated_evidence",
        create_autospec(analysis.validated_evidence, return_value=checked),
    )
    review_path = root / "fixture_review.json"
    save_json(
        review_path,
        {"verdict": "PASS", "reviewer": "fixture only", "sources_sha256": analysis.source_hashes()},
    )
    cfg = OmegaConf.create(
        {
            "root": str(root),
            "captures": str(root / "capture"),
            "map": str(map_path),
            "spec": str(spec_path),
            "output_dir": str(root / "analysis"),
            "review": str(review_path),
        }
    )
    orphan = root / "analysis/orphan_fit.json"
    save_json(orphan, {"unbound": True})
    with pytest.raises(ValueError, match="empty output"):
        analysis.run(cfg)
    orphan.unlink()
    first = analysis.run(cfg)
    assert first["verification_passed"] is True
    assert set(first["analyses"]) == {
        "primary",
        "competence_sensitivity",
        "screen_augmented_primary",
        "screen_augmented_competence_sensitivity",
    }
    assert all(
        r["status"] == "prevalence_only_insufficient_training_support"
        for r in first["analyses"].values()
    )
    assert analysis.run(cfg) == first
    provenance["changed"] = True
    with pytest.raises(ValueError, match="resume"):
        analysis.run(cfg)


def test_actual_fitting_helper_import_drift_is_rejected(tmp_path, monkeypatch):
    from scripts import context_risk_followup_features as features

    changed = tmp_path / "fixture_other_source.py"
    changed.write_text("fixture different implementation")
    monkeypatch.setattr(features, "__file__", str(changed))
    with pytest.raises(ValueError, match="Imported analysis helper"):
        analysis.source_hashes()


def test_pca_runs_only_after_supported_initial_context_primary_benefit(tmp_path, monkeypatch):
    """Force only the CI branch boundary; the extra PCA grid and fits execute real bodies."""
    rows, raw, maps, spec = fixture()
    for row in rows:
        row["positive"] = 2
    actual_interval = analysis.paired_task_interval

    def supported(*args, **kwargs):
        value = actual_interval(*args, **kwargs)
        value["positive_interval"] = True
        return value

    monkeypatch.setattr(
        analysis, "paired_task_interval", create_autospec(actual_interval, side_effect=supported)
    )
    primary = population(tmp_path / "primary", rows, raw, maps, spec)
    assert primary["comparisons"]["mapped_over_raw"]["benefit_supported"]
    pca = primary["models"]["pca_plus_metadata"]
    assert len(pca["selection"]["cv_rows"]) == 180
    assert pca["selection"]["rank"] == 8  # All nominal ranks clip to7 and tie.
    secondary = population(tmp_path / "secondary", rows, raw, maps, spec, augmented=True)
    assert "pca_plus_metadata" not in secondary["models"]


def test_static_rule_checks_all_inputs_and_rejects_changed_defect_roster(tmp_path, monkeypatch):
    rows = [
        {
            "task_id": f"fixture_{i}",
            "condition": condition,
            "prompt": "def solution():",
            "entry_point": "solution",
            "test": 'assert solution() == "unfinished'
            if i == 77 and condition != "original"
            else "def check(candidate):\n    assert candidate() == 0",
        }
        for i in range(103)
        for condition in ("original", "oneoff", "conflicting")
    ]
    source = tmp_path / "manifests/source.jsonl"
    source.parent.mkdir()
    source.write_text("software fixture authorization")
    policy = tmp_path / "fixture_policy.json"
    policy.write_text(
        json.dumps(
            {
                "source_sha256": analysis.validity.design.sha256(source),
                "structurally_invalid_test_sha256": {
                    f"{r['task_id']}:{r['condition']}": hashlib.sha256(
                        r["test"].encode()
                    ).hexdigest()
                    for r in rows
                    if r["task_id"] == "fixture_77" and r["condition"] != "original"
                },
            }
        )
    )
    monkeypatch.setattr(analysis.validity, "POLICY", policy)
    monkeypatch.setattr(analysis.validity.design, "source_rows", lambda path: rows)
    audited = analysis.validity.source_audit(tmp_path)
    assert audited["contexts_checked"] == 309
    assert len(audited["structurally_invalid_test_sha256"]) == 2
    rows[0]["test"] = 'assert solution() == "new defect'
    with pytest.raises(ValueError, match="invalid-input roster"):
        analysis.validity.source_audit(tmp_path)


@pytest.mark.parametrize("other_success", [0, 2])
def test_native_scores_remain_immutable_and_every_rank_and_role_is_identical(other_success):
    rows = [
        {
            "task_id": f"lcbhard_{i}",
            "condition": c,
            **counts(0 if i == 77 else other_success, 2 if i == 77 else 2 - other_success, 0),
        }
        for i in range(103)
        for c in ("original", "oneoff", "conflicting")
    ]
    before = copy.deepcopy(rows)
    audit = {
        "structurally_invalid_test_sha256": {
            "lcbhard_77:oneoff": "fixture hash",
            "lcbhard_77:conflicting": "fixture hash",
        }
    }
    result = analysis.validity.selection_annotation(rows, audit)
    assert rows == before
    assert result["ranking_and_split_unchanged"]
    assert result["not_assessable_planned_trajectories"] == 4
    native = next(r for r in result["native_ranking"] if r["task_id"] == "lcbhard_77")
    semantic = next(r for r in result["validity_aware_ranking"] if r["task_id"] == "lcbhard_77")
    assert native["failure"] == 4 and native["censored"] == 0
    assert semantic["failure"] == 0 and semantic["censored"] == 4
    assert semantic["complete_case_rate"] is None
    assert native["rank"] == semantic["rank"]
    rows[77 * 3 + 1].update(success=1, failure=1)
    with pytest.raises(ValueError, match="unexpectedly received reward"):
        analysis.validity.annotate(rows, audit)


def test_invalid_inputs_are_removed_before_feature_construction_and_block_claims(tmp_path):
    rows, raw, maps, spec = fixture(training=6, testing=3)
    for row in rows:
        row["positive"] = 2
    invalid = {}
    for row in rows:
        if row["task_id"] == "fixture_00" and row["condition"] != "original":
            row.update(test='assert function() == "unfinished', positive=0)
            invalid[f"{row['task_id']}:{row['condition']}"] = "fixture hash"
    provenance = fixture_provenance(rows)
    provenance["input_validity"]["fresh"] = {
        "not_assessable_planned_trajectories": 8,
        "static_audit": {"structurally_invalid_test_sha256": invalid},
    }
    fit_rows, fit_raw = analysis.validity.assessable_inputs(
        rows, raw, provenance["input_validity"]["fresh"]["static_audit"]
    )
    assert len(rows) == 27 and len(fit_rows) == 25 and fit_raw.shape == (25, 7)
    assert any(r["task_id"] == "fixture_00" and r["condition"] == "original" for r in fit_rows)
    bank = analysis.make_bank(fit_rows, fit_raw, maps, spec, augmented=True)
    result = analysis.analyze_population(
        fit_rows, bank, spec, tmp_path, provenance, eligible_only=False, augmented=True
    )
    assert result["fit_gate_passed"] and result["training_support"]["n_tasks"] == 5
    assert result["test_support"]["n_tasks"] == 3
    assert not result["claim_support_passed"]
    assert result["structurally_not_assessable_fresh_trajectories"] == 8
    assert result["total_fresh_censored"] == 0
    assert all(not value["benefit_supported"] for value in result["comparisons"].values())
    np.testing.assert_array_equal(fit_raw[0], raw[2])
