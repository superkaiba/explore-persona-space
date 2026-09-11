"""Protect the paired five-draw estimand, recipe and model-specific pools."""

import json

import numpy as np
import pytest

from scripts import issue2054_k3 as k3
from scripts import issue2054_k5 as k5
from scripts import issue2054_k5_fit as fit


def test_seed_extension_preserves_prior_and_adds_disjoint_streams():
    seeds = [k3.seed("cell", "row", d) for d in (1, 2)]
    seeds += [k5.seed("cell", "row", d) for d in (3, 4)]
    assert len(set(seeds)) == 4
    assert k5.seed("cell", "row", 3) != k5.seed("cell", "other", 3)
    with pytest.raises(ValueError):
        k5.seed("cell", "row", 1)


def five_vectors():
    old = {
        "conv_id": np.array(["a", "b", "c"]),
        "v_A_0": np.full((3, 3584), 1.0, dtype=np.float16),
        "v_A_12": np.full((3, 2, 3584), 3.0, dtype=np.float16),
        "valid_draws_12": np.ones((3, 2), dtype=bool),
        "cap_mask": np.zeros((3, 3), dtype=bool),
    }
    new = {
        "conv_id": old["conv_id"].copy(),
        "v_A_34": np.full((3, 2, 3584), 5.0, dtype=np.float16),
        "valid_draws_34": np.ones((3, 2), dtype=bool),
        "cap_mask_34": np.zeros((3, 2), dtype=bool),
    }
    old["valid_draws_12"][1, 0] = False
    old["v_A_12"][1, 0] = np.nan
    new["valid_draws_34"][2, 1] = False
    new["v_A_34"][2, 1] = np.nan
    return old, new


def test_exact_five_average_and_symmetric_empty_exclusion():
    old, new = five_vectors()
    targets, keep, caps = k5.average_targets(old, new)
    np.testing.assert_array_equal(keep, [True, False, False])
    assert caps.shape == (3, 5)
    assert {k: v.shape for k, v in targets.items()} == {k: (1, 3584) for k in (1, 3, 5)}
    np.testing.assert_allclose(targets[1], 1)
    np.testing.assert_allclose(targets[3], 7 / 3)
    np.testing.assert_allclose(targets[5], 17 / 5)


def test_capture_join_rejects_permuted_contexts_and_nonfinite_valid_draws():
    old, new = five_vectors()
    new["conv_id"] = new["conv_id"][::-1]
    with pytest.raises(ValueError, match="identity/order"):
        k5.average_targets(old, new)
    old, new = five_vectors()
    new["v_A_34"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        k5.average_targets(old, new)


def test_original_stop_cohort_does_not_claim_all_five_stopped():
    caps = np.array([[False, False, False, True, False], [True, False, False, False, False]])
    result = fit.cohorts("assistant__on_policy__bare_text__qwen2.5-7b-instruct", caps)
    np.testing.assert_equal(result["original_draw_stopped"], [True, False])
    np.testing.assert_equal(result["all_five_stopped"], [False, False])


def test_parent_receipt_rejects_wrong_destination(tmp_path):
    path = tmp_path / "inputs" / k5.PARENT_PREFIX / "manifest.json"
    k3.atomic_json(path, {"ok": True})
    k3.atomic_json(
        path.with_suffix(".json.done.json"),
        {"path": "another/manifest.json", "fingerprint": "fp", "sha256": k3.sha(path)},
    )
    with pytest.raises(RuntimeError, match="destination"):
        k5.verified_parent(tmp_path, "manifest.json")
    receipt = json.loads(path.with_suffix(".json.done.json").read_text())
    receipt["path"] = k5.PARENT_PREFIX + "/manifest.json"
    k3.atomic_json(path.with_suffix(".json.done.json"), receipt)
    assert k5.verified_parent(tmp_path, "manifest.json") == path


def test_displayed_panel_excludes_fixed_and_bare_label_cells():
    cells = []
    for model in k3.MODEL_REVISIONS:
        for char in ("dana", "helios", "wren", "vex"):
            for form in ("attrib_quoted", "bare_label"):
                cells.append({"cell": f"char_{char}__on_policy__{form}__{model}", "raw": "path"})
        for form in ("chat", "bare_text", "attrib_quoted", "bare_label"):
            cells.append(
                {
                    "cell": f"conversation_paired_stories_assistant__on_policy__{form}__{model}",
                    "raw": "path",
                }
            )
        cells.append({"cell": f"char_dana__inserted__attrib_quoted__{model}"})
    selected = k5.selected({"cells": cells})
    assert len(selected) == 12
    assert all("__on_policy__" in c["cell"] and "__bare_label__" not in c["cell"] for c in selected)


def test_shared_eigh_three_targets_match_independent_fits():
    # Actual solver bodies, not mocked dispatch or implementation-shaped math.
    from scripts.issue2054_ctx2ctx_fit import SharedEighRidge

    rng = np.random.default_rng(137)
    x = rng.normal(size=(120, 8))
    xe = rng.normal(size=(25, 8))
    shared = SharedEighRidge(x, xe, device="cpu")
    for scale in (1.0, 0.5, 0.1):
        y = x @ rng.normal(size=(8, 8)) + scale * rng.normal(size=(120, 8))
        actual, info = shared.fit_predict(y)
        expected, expected_info = SharedEighRidge(x, xe, device="cpu").fit_predict(y)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        assert info["best_lambda"] == expected_info["best_lambda"]
