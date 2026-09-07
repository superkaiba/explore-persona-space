"""Focused tests for the #2569 answer write-subspace decomposition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2569_refusal_answer_write.py"
SCRIPTS = SCRIPT.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("issue2569_refusal_answer_write", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_classify_pair_uses_frozen_thresholds_and_control_override() -> None:
    assert MOD.classify_pair("verb_harm", 0.0, 1.0) == "control"
    assert MOD.classify_pair("obj_flip", 0.0, 0.5) == "flip"
    assert MOD.classify_pair("obj_flip", 0.2, 0.3) == "nonflip"
    assert MOD.classify_pair("obj_flip", 0.2, 0.31) == "mid"


def test_decompose_recovers_orthogonal_components() -> None:
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    values = rng.standard_normal((5, 8))
    low_mask = np.asarray([False, False, False, True, True, True, True, True])
    parts = MOD.decompose(values, basis, low_mask)
    np.testing.assert_allclose(parts["low"] + parts["high"], values, atol=1e-12)
    np.testing.assert_allclose(np.einsum("ij,ij->i", parts["low"], parts["high"]), 0.0, atol=1e-12)
    np.testing.assert_allclose(parts["low_share"] + parts["high_share"], 1.0, atol=1e-12)


def test_bootstrap_difference_sign_and_effect_size() -> None:
    result = MOD.bootstrap_difference(
        np.asarray([0.0, 0.1, 0.2]), np.asarray([0.5, 0.6, 0.7]), seed=11
    )
    assert result["mean_difference"] < 0
    assert result["median_difference"] < 0
    assert result["cliffs_delta"] == -1.0
    assert result["mean_difference_ci95"][1] < 0
