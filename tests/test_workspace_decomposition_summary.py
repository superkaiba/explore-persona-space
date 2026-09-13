"""Weighting, exclusions and covariance must survive the reporting reduction."""

import copy
import json
import runpy
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.workspace_analysis_inputs import summarize_token_statistic
from explore_persona_space.analysis.workspace_decomposition_summary import (
    FIELDS,
    null_statistics,
    observed_statistics,
    summarize_decomposition,
)
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json


def token_fixture(draws, ids, k=5):
    """Build genuine token statistics for positive/negative one-dimensional components."""
    rows, pooled, raw = [], {name: [] for name in ("full", "J", "restJ", "R", "restR")}, []
    for context, rollouts in zip(ids, draws, strict=True):
        counts = [len(h) for h in rollouts]
        h = np.concatenate(rollouts)
        arms = {"J": np.maximum(h, 0), "R": np.minimum(h, 0)}
        values = {"full": h, **arms, "restJ": h - arms["J"], "restR": h - arms["R"]}
        for name, tokens in values.items():
            parts = np.split(tokens, np.cumsum(counts)[:-1])
            pooled[name].append(np.mean([part.mean() for part in parts]))
        fields = {
            arm: {
                "active_atoms": (component != 0).astype(float),
                "squared_error": (h - component) ** 2,
                "input_squared_norm": h**2,
                "zero_update_steps": k - (component != 0).astype(float),
                "increasing_error_steps": np.zeros(len(h)),
            }
            for arm, component in arms.items()
        }
        raw.append(fields)
        rows.append(
            {
                "context_id": context,
                "token_counts": counts,
                "arms": {
                    arm: {
                        name: summarize_token_statistic(torch.from_numpy(value), counts)
                        for name, value in fields[arm].items()
                    }
                    for arm in arms
                },
            }
        )
    return rows, {name: np.asarray(value)[:, None] for name, value in pooled.items()}, raw


def test_equal_rollout_means_match_raw_tokens_and_exclude_other_contexts():
    ids, common = list("abc"), list("ca")
    draws = [
        [np.array([4.0]), np.array([-1.0, -2, -3])],
        [np.array([100.0, 200]), np.array([300.0])],
        [np.array([-2.0, -3, -4, -5]), np.array([1.0, 3])],
    ]
    rows, targets, raw = token_fixture(draws, ids)
    arrays = observed_statistics(rows, ids, common, 5, 2)
    selected = [2, 0]
    targets = {name: values[selected] for name, values in targets.items()}
    predictions = {"ridge": {name: values + 0.5 for name, values in targets.items()}}
    report = summarize_decomposition(targets, predictions, arrays)
    for arm in ("J", "R"):
        for field in FIELDS:
            expected_equal, selected_tokens = [], []
            for index in selected:
                tokens = raw[index][arm][field]
                parts = np.split(tokens, np.cumsum(rows[index]["token_counts"])[:-1])
                expected_equal.append(np.mean([part.mean() for part in parts]))
                selected_tokens.append(tokens)
            values = report["arms"][arm]
            assert values["token_statistics_equal_context_then_equal_rollout"][
                field
            ] == pytest.approx(np.mean(expected_equal))
            assert values["token_statistics_uniform_over_all_tokens"][field] == pytest.approx(
                np.concatenate(selected_tokens).mean()
            )
        c, r, y = targets[arm], targets[f"rest{arm}"], targets["full"]
        expected_covariance = float(((c - c.mean(0)) * (r - r.mean(0))).sum() / len(c))
        assert expected_covariance != 0
        geometry = report["arms"][arm]["pooled_target_geometry"]
        assert geometry["twice_component_rest_covariance_trace"] == pytest.approx(
            2 * expected_covariance
        )
        assert geometry["component_variance_trace"] == pytest.approx(float(np.var(c, axis=0).sum()))
        assert geometry["target_variance_trace"] == pytest.approx(float(np.var(y, axis=0).sum()))
        assert geometry["max_abs_reconstruction_error"] == 0
    assert (
        report["arms"]["J"]["token_statistics_equal_context_then_equal_rollout"]["active_atoms"]
        != report["arms"]["J"]["token_statistics_uniform_over_all_tokens"]["active_atoms"]
    )
    assert report["predictor_metrics"]["ridge"]["J"]["sse"] == 0.5
    assert report["tokens"] == 10
    assert arrays["context_ids"].tolist() == common


def test_affine_null_shortcut_equals_explicit_token_repetition():
    ids, common, values = list("abc"), list("ca"), [2.0, -3.0, 4.0]
    lengths = [[1, 7], [4, 2], [3, 1]]
    draws = [
        [np.repeat(value, length) for length in counts]
        for value, counts in zip(values, lengths, strict=True)
    ]
    rows, _, raw = token_fixture(draws, ids)
    direct = observed_statistics(rows, ids, common, 5, 2)
    prepared = {
        f"test/{arm}__{field}": np.asarray([row[arm][field][0] for row in raw])
        for arm in ("J", "R")
        for field in FIELDS
    }
    layout = {"context_ids": {"test": ids}, "token_counts": {"test": lengths}}
    reduced = null_statistics(prepared, layout, ids, common, 5, 2)
    assert set(direct) == set(reduced)
    for name in direct:
        np.testing.assert_array_equal(direct[name], reduced[name])
    malformed = copy.deepcopy(prepared)
    malformed["test/J__active_atoms"][0] = 0.5
    with pytest.raises(ValueError, match="Invalid exactly-affine"):
        null_statistics(malformed, layout, ids, common, 5, 2)


def test_zero_energy_and_variance_remain_undefined():
    ids = list("ab")
    rows, targets, _ = token_fixture([[np.zeros(2), np.zeros(1)]] * 2, ids)
    arrays = observed_statistics(rows, ids, ids, 5, 2)
    report = summarize_decomposition(targets, {"ridge": targets}, arrays)
    for arm in ("J", "R"):
        assert report["arms"][arm]["residual_energy_fraction_equal_context_rollout"] is None
        assert report["arms"][arm]["pooled_component_variance_fraction"] is None
        assert report["arms"][arm]["zero_token_input_energy"]
        assert report["arms"][arm]["zero_pooled_full_variance"]
        assert report["predictor_metrics"]["ridge"][arm]["r2"] is None


def test_inconsistent_weights_bad_lengths_and_missing_contexts_fail():
    ids = list("ab")
    rows, _, _ = token_fixture([[np.array([1.0]), np.array([-1.0, -2])]] * 2, ids)
    changed = copy.deepcopy(rows)
    changed[0]["arms"]["J"]["active_atoms"]["mean_equal_rollout"] = 0.9
    with pytest.raises(ValueError, match="weighting"):
        observed_statistics(changed, ids, ids, 5, 2)
    changed = copy.deepcopy(rows)
    changed[0]["token_counts"][0] = 0
    with pytest.raises(ValueError, match="positive integers"):
        observed_statistics(changed, ids, ids, 5, 2)
    with pytest.raises(ValueError, match="lacks a primary"):
        observed_statistics(rows, ids, list("ac"), 5, 2)
    with pytest.raises(ValueError, match="order or coverage"):
        observed_statistics(rows[::-1], ids, ids, 5, 2)


def comparison_fixture(tmp_path, monkeypatch):
    """Construct actual hashed comparison metadata, with no mocked reader functions."""
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    api = runpy.run_path(str(scripts / "workspace_jr_decomposition_summary.py"))
    root, config, selection = tmp_path / "comparison", tmp_path / "config", tmp_path / "selection"
    config.write_text("fixture configuration")
    selection.write_text("fixture selection")
    cells = [
        {"role": role, "kind": kind, "k": k, "rotation": rotation}
        for role in ("primary", "comparison")
        for kind in ("observed", "affine_null")
        for k in (5, 10, 25)
        for rotation in (None, 20260913, 20260914, 20260915)
    ]
    common = ["context-a", "context-c"]
    source_manifest = {
        "cells": cells,
        "config_sha256": file_sha256(config),
        "selection_sha256": file_sha256(selection),
    }
    save_json(root / "input_manifest.json", source_manifest)
    save_json(
        root / "completion_cohort.json",
        {"status": "complete", "joint_complete_context_ids": common},
    )
    save_json(root / "primary_scoring_cohort.json", {"common_context_ids": common})
    proof = {
        "cells": {api["cell_key"](e["role"], e["kind"], e["k"], e["rotation"]): {} for e in cells},
        "completion_cohort": {"report_sha256": file_sha256(root / "completion_cohort.json")},
    }
    save_json(root / "source_proof.json", proof)
    save_json(
        root / "coverage.json",
        {
            "status": "complete_grid",
            "expected_cells": 48,
            "realized_cells": 48,
            "missing_cells": [],
        },
    )
    scopes = {}
    for scope in ("primary", "comparison", "cross_model"):
        path = root / scope / "comparisons.json"
        save_json(path, {"context_ids": common})
        scopes[scope] = {"report_sha256": file_sha256(path)}
    save_json(
        root / "comparison_complete.json",
        {
            "status": "complete",
            "scopes": scopes,
            "input_manifest_sha256": file_sha256(root / "input_manifest.json"),
        },
    )
    receipt = tmp_path / "receipt.json"

    def upload():
        """Bind the fixture's actual bytes through the same upload-reader contract."""
        hashes = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
        save_json(
            receipt,
            {
                "repo": "superkaiba1/explore-persona-space-data",
                "revision": "a" * 40,
                "prefix": "exploratory_workspace_jr/decomposition-fixture",
                "files_verified": len(hashes),
                "verified_sha256": hashes,
            },
        )
        return api["Source"]({"root": str(root), "upload_receipt": str(receipt)})

    return api, root, config, selection, source_manifest, common, upload


def test_complete_grid_reader_binds_bytes_population_and_scope_reports(tmp_path, monkeypatch):
    api, root, config, selection, manifest, common, upload = comparison_fixture(
        tmp_path, monkeypatch
    )
    source = upload()
    actual, ids, proof = api["completed_comparison"](source, config, selection)
    assert actual == manifest and ids == common and len(proof["cells"]) == 48
    save_json(
        root / "primary_scoring_cohort.json", {"common_context_ids": ["context-a", "context-b"]}
    )
    with pytest.raises(ValueError, match="verified producer upload"):
        api["completed_comparison"](source, config, selection)
    with pytest.raises(ValueError, match="complete original paired grid"):
        api["completed_comparison"](upload(), config, selection)


def test_complete_grid_reader_rejects_duplicate_cells_even_with_fresh_hashes(tmp_path, monkeypatch):
    api, root, config, selection, manifest, _, upload = comparison_fixture(tmp_path, monkeypatch)
    manifest["cells"][-1] = dict(manifest["cells"][0])
    save_json(root / "input_manifest.json", manifest)
    marker = json.loads((root / "comparison_complete.json").read_text())
    marker["input_manifest_sha256"] = file_sha256(root / "input_manifest.json")
    save_json(root / "comparison_complete.json", marker)
    with pytest.raises(ValueError, match="duplicate or missing"):
        api["completed_comparison"](upload(), config, selection)


def test_final_cohort_recomputes_all_cell_intersection_and_exclusions(tmp_path, monkeypatch):
    api, root, config, selection, manifest, common, upload = comparison_fixture(
        tmp_path, monkeypatch
    )
    eligible = ["context-a", "context-b", "context-c"]
    cells = {
        api["cell_key"](e["role"], e["kind"], e["k"], e["rotation"]): {
            "ids": [*eligible, "context-incomplete"]
        }
        for e in manifest["cells"]
    }
    first = next(iter(cells))
    cells[first] = {"ids": [*common, "context-incomplete"]}
    expected, cohort = api["paired_cohort"](
        {key: cell["ids"] for key, cell in cells.items()}, eligible_ids=eligible
    )
    assert expected == common and set(common) < set(eligible)
    save_json(
        root / "completion_cohort.json",
        {"status": "complete", "joint_complete_context_ids": eligible},
    )
    proof = json.loads((root / "source_proof.json").read_text())
    proof["completion_cohort"]["report_sha256"] = file_sha256(root / "completion_cohort.json")
    save_json(root / "source_proof.json", proof)
    save_json(root / "primary_scoring_cohort.json", cohort)
    source = upload()
    _, actual, _ = api["completed_comparison"](source, config, selection)
    assert api["verified_primary_cohort"](source, cells, actual) == cohort
    changed = copy.deepcopy(cells)
    changed[first]["ids"].append("context-b")
    with pytest.raises(ValueError, match="reproduce the completed"):
        api["verified_primary_cohort"](source, changed, common)
    # The common IDs alone are insufficient: the saved exclusions must also agree.
    cohort["cells"][first]["excluded_from_pairing"] = []
    save_json(root / "primary_scoring_cohort.json", cohort)
    with pytest.raises(ValueError, match="reproduce the completed"):
        api["verified_primary_cohort"](upload(), cells, common)


def test_observed_reader_checks_fit_bindings_and_successful_terminal(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    api = runpy.run_path(str(scripts / "workspace_jr_decomposition_summary.py"))
    root, fit = tmp_path / "statistics", tmp_path / "fit"
    fit.mkdir()
    for name in ("input_manifest.json", "results.json", "per_example.npz"):
        (fit / name).write_bytes(b"deliberately tiny metadata-binding fixture")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = {
        "config_sha256": "fixture",
        "selection_sha256": "fixture",
        "model_role": "comparison",
        "versions": {},
        "code": {"git_commit": sha, "git_dirty": False},
    }
    ids = list("ab")
    rows, _, _ = token_fixture([[np.array([1.0]), np.array([-1.0, -2])]] * 2, ids)
    ledger = {"test": {context: context * 64 for context in ids}}
    cell = {
        "fit": fit,
        "ids": ids,
        "manifest": {"identity": identity, "coverage": {"test": {"file_sha256": ledger["test"]}}},
    }
    sources = {
        "source_identity": identity,
        "component_files": ledger,
        "results_sha256": file_sha256(fit / "results.json"),
        "input_manifest_sha256": file_sha256(fit / "input_manifest.json"),
        "per_example_sha256": file_sha256(fit / "per_example.npz"),
    }
    save_json(root / "input_proof.json", {"identity": identity, "sources": sources})
    save_json(
        root / "analysis_complete.json",
        {
            "identity": identity,
            "status": "complete",
            "phase": "statistics",
            "input_proof_sha256": file_sha256(root / "input_proof.json"),
        },
    )
    terminal = {
        "phase": "complete",
        "exit_code": 0,
        "analysis_complete_sha256": file_sha256(root / "analysis_complete.json"),
    }
    save_json(root / "analysis_operations/exit.json", terminal)
    save_json(root / "decomposition_statistics.json", {"test": rows})
    receipt = tmp_path / "statistics_receipt.json"

    def upload():
        """Create a receipt over the actual tiny fixture bytes."""
        hashes = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()}
        save_json(
            receipt,
            {
                "repo": "superkaiba1/explore-persona-space-data",
                "revision": "a" * 40,
                "prefix": "exploratory_workspace_jr/statistics-fixture",
                "files_verified": len(hashes),
                "verified_sha256": hashes,
            },
        )

    upload()
    entry = {"root": str(root), "upload_receipt": str(receipt), "k": 5, "rotation": 20260913}
    config = {"generation": {"seeds": [42, 43]}}
    arrays, proof = api["read_observed_statistics"](entry, cell, ids, identity, config)
    assert arrays["context_ids"].tolist() == ids
    assert "analysis_operations/exit.json" in proof["files"]
    terminal["exit_code"] = 1
    save_json(root / "analysis_operations/exit.json", terminal)
    upload()
    with pytest.raises(ValueError, match="successful bound analyzer terminal"):
        api["read_observed_statistics"](entry, cell, ids, identity, config)


def test_null_reader_binds_prepared_array_hashes_and_rollout_seeds(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    api = runpy.run_path(str(scripts / "workspace_jr_decomposition_summary.py"))
    root = tmp_path / "null"
    root.mkdir()
    ids, lengths = list("ab"), [[1, 4], [2, 3]]
    rows, _, raw = token_fixture(
        [
            [np.repeat(value, count) for count in counts]
            for value, counts in zip([2.0, -1.0], lengths, strict=True)
        ],
        ids,
    )
    prepared = {
        f"test/{arm}__{field}": np.asarray([row[arm][field][0] for row in raw])
        for arm in ("J", "R")
        for field in FIELDS
    }
    np.savez(root / "decomposition_k5.npz", **prepared)
    layout = {
        "context_ids": {"test": ids},
        "token_counts": {"test": lengths},
        "rollout_seeds": [42, 43],
    }
    save_json(root / "layout.json", layout)
    hashes = {name: file_sha256(root / name) for name in ("layout.json", "decomposition_k5.npz")}
    receipt = tmp_path / "null_receipt.json"
    save_json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "a" * 40,
            "prefix": "exploratory_workspace_jr/null-fixture",
            "files_verified": len(hashes),
            "verified_sha256": hashes,
        },
    )
    entry = {"root": str(root), "upload_receipt": str(receipt), "k": 5}
    cell = {"ids": ids, "manifest": {"prepared_sources": hashes}}
    arrays, _ = api["read_null_statistics"](entry, cell, ids, {"generation": {"seeds": [42, 43]}})
    direct = observed_statistics(rows, ids, ids, 5, 2)
    for name in direct:
        np.testing.assert_array_equal(arrays[name], direct[name])
    with pytest.raises(ValueError, match="frozen rollout seeds"):
        api["read_null_statistics"](entry, cell, ids, {"generation": {"seeds": [43, 44]}})
    cell["manifest"]["prepared_sources"] = {**hashes, "decomposition_k5.npz": "f" * 64}
    with pytest.raises(ValueError, match="actual prepared fit inputs"):
        api["read_null_statistics"](entry, cell, ids, {"generation": {"seeds": [42, 43]}})
