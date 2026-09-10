"""Check same-decoder transfer, saved-probe identity and normalized error ratios."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2564_answer_behavior_readout as source
import issue2564_answer_probe_transfer as transfer


def fixture():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(40, 13)) * np.arange(1, 14)
    x[:, -1] = 3
    mapped = x * 0.7 + 2
    y = x[:, :2] + rng.normal(size=(40, 2))
    rows = [{"id": str(i), "question_group": str(i // 2), "fold": (i // 2) % 5} for i in range(40)]
    available = np.ones(40, bool)
    available[[4, 9]] = False
    targets = [
        source.Target(n, "graded", y[:, j : j + 1], available, []) for j, n in enumerate(("a", "b"))
    ]
    train = np.array([i for i, r in enumerate(rows) if available[i] and r["fold"] != 0])
    test = np.array([i for i, r in enumerate(rows) if available[i] and r["fold"] == 0])
    alphas = {"a": 10, "b": 100}
    saved = {
        "fold": 0,
        "train_ids": [rows[i]["id"] for i in train],
        "test_ids": [rows[i]["id"] for i in test],
        "models": {"answer": {k: {"alpha": v} for k, v in alphas.items()}},
    }
    reference, expected = {}, {}
    scaler = StandardScaler().fit(x[train])
    for j, target in enumerate(targets):
        model = Ridge(alpha=alphas[target.name], solver="svd").fit(
            scaler.transform(x[train]), y[train, j : j + 1]
        )
        reference[target.name + "__answer"] = np.full((40, 1), np.nan)
        reference[target.name + "__answer"][test] = model.predict(
            scaler.transform(x[test])
        ).reshape(-1, 1)
        reference[target.name + "__prior"] = np.full((40, 1), np.nan)
        reference[target.name + "__prior"][test] = y[train, j].mean()
        expected[target.name] = model.predict(scaler.transform(mapped[test])).reshape(-1, 1)
    return rows, x, mapped, targets, saved, reference, expected


def test_real_reconstruction_matches_independent_fixed_probe_on_shifted_inputs():
    rows, x, mapped, targets, saved, reference, expected = fixture()
    result, details = transfer.reconstruct_bundle(
        rows, x, {"answer": x, "mapped": mapped, "same": x.copy()}, targets, saved, reference
    )
    for target in targets:
        np.testing.assert_allclose(
            result[target.name + "__mapped"], expected[target.name], atol=1e-9
        )
        np.testing.assert_allclose(result[target.name + "__same"], result[target.name + "__answer"])
    assert max(details["observed_replay_max_abs_error"].values()) < 1e-9


def test_saved_probe_and_prior_mismatch_are_rejected():
    rows, x, mapped, targets, saved, reference, _ = fixture()
    saved["train_ids"] = saved["train_ids"][::-1]
    with pytest.raises(ValueError, match="training IDs"):
        transfer.reconstruct_bundle(
            rows, x, {"answer": x, "mapped": mapped}, targets, saved, reference
        )
    saved["train_ids"] = saved["train_ids"][::-1]
    reference["a__prior"][:] = 999
    with pytest.raises(ValueError, match="prior parity"):
        transfer.reconstruct_bundle(
            rows, x, {"answer": x, "mapped": mapped}, targets, saved, reference
        )


def test_fixed_bootstrap_regime_rejects_other_draw_counts():
    manifest = {"regime": {"seed": 2564}}
    transfer.validate_fixed_regime(transfer.Config(), manifest)
    with pytest.raises(ValueError, match="exactly 2000"):
        transfer.validate_fixed_regime(transfer.Config(bootstrap_draws=1999), manifest)


def test_saved_oof_schema_requires_all_arms_and_finite_available_rows():
    rows, _, _, targets, _, _, _ = fixture()
    reference = {
        f"{target.name}__{arm}": np.full(target.values.shape, np.nan)
        for target in targets
        for arm in ("prior", "framing", "context", "length", "answer")
    }
    for target in targets:
        for arm in ("prior", "framing", "context", "length", "answer"):
            reference[f"{target.name}__{arm}"][target.available] = 0
    transfer.validate_reference_oof(reference, rows, targets)
    reference["a__answer"] = np.zeros((40, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        transfer.validate_reference_oof(reference, rows, targets)
    reference["a__answer"] = np.full((40, 1), np.nan)
    reference["a__answer"][0] = 0
    with pytest.raises(ValueError, match="nonfinite available"):
        transfer.validate_reference_oof(reference, rows, targets)
    reference["a__answer"] = np.zeros((40, 1))
    reference["a__answer"][~targets[0].available] = np.nan
    reference["a__answer"][4] = 0
    with pytest.raises(ValueError, match="unavailable rows"):
        transfer.validate_reference_oof(reference, rows, targets)


def test_resumed_replay_block_rechecks_split_shapes_and_observed_parity():
    rows, x, mapped, targets, saved, reference, _ = fixture()
    result, details = transfer.reconstruct_bundle(
        rows, x, {"answer": x, "mapped": mapped}, targets, saved, reference
    )
    transfer.validate_replay_block(
        result, details, rows, {"answer": x, "mapped": mapped}, targets, reference
    )
    bad = dict(result)
    bad["test"] = result["test"][::-1]
    with pytest.raises(ValueError, match="test indices"):
        transfer.validate_replay_block(
            bad, details, rows, {"answer": x, "mapped": mapped}, targets, reference
        )
    bad = dict(result)
    bad["a__mapped"] = bad["a__mapped"][:, :0]
    with pytest.raises(ValueError, match="shape mismatch"):
        transfer.validate_replay_block(
            bad, details, rows, {"answer": x, "mapped": mapped}, targets, reference
        )
    bad = dict(result)
    bad["a__answer"] = bad["a__answer"].copy()
    bad["a__answer"][0, 0] += 1
    with pytest.raises(ValueError, match="observed-probe parity"):
        transfer.validate_replay_block(
            bad, details, rows, {"answer": x, "mapped": mapped}, targets, reference
        )


def test_retention_has_correct_anchors_and_preserves_out_of_range_values():
    prior = np.array([10, 10, 10, 10, 10, 10], float)
    observed = np.array([4, 4, 4, 4, 10, 11], float)
    mapped = np.array([4, 10, 13, 1, 4, 4], float)
    result = transfer.transfer_statistics(prior, observed, mapped)
    np.testing.assert_allclose(result["retained_error_reduction"][:4], [1, 0, -0.5, 1.5])
    assert np.isnan(result["retained_error_reduction"][4:]).all()
    np.testing.assert_allclose(result["excess_error"], mapped - observed)
    assert transfer.summarized(result["retained_error_reduction"])["defined_draws"] == 3


def test_output_agreement_retains_vector_columns_and_zero_variance():
    target = source.Target("class", "categorical", np.eye(3), np.ones(3, bool), ["a", "b", "c"])
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    weights = np.array([[1, 1, 1], [2, 1, 0]])
    result = transfer.output_agreement(
        target, {"answer": values, "mapped": values}, weights, np.arange(3)
    )
    np.testing.assert_allclose(result["r2"], 1)
    zeros = np.zeros_like(values)
    result = transfer.output_agreement(
        target, {"answer": zeros, "mapped": zeros}, weights, np.arange(3)
    )
    assert np.isnan(result["r2"]).all()
