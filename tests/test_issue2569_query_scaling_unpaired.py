from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue2569_mapping_diff_report as RP  # noqa: E402
import issue2569_query_scaling_unpaired as QU  # noqa: E402


def test_disjoint_anchor_sets_never_share_rows() -> None:
    rng = np.random.default_rng(3)
    train = np.arange(1000, dtype=np.int64)
    for _ in range(20):
        source, target = QU.disjoint_anchor_sets(train, 400, rng)
        assert len(np.unique(source)) == 400
        assert len(np.unique(target)) == 400
        assert np.intersect1d(source, target).size == 0


def test_unpaired_query_limit_is_half_train_size() -> None:
    QU.validate_k_values([4, 50], 100, unpaired=True)
    with pytest.raises(ValueError, match="cannot exceed 50"):
        QU.validate_k_values([51], 100, unpaired=True)


def test_pca_seeds_are_unique_across_all_fitted_views() -> None:
    seeds = {
        QU.pca_seed(2569, direction, k, repeat, view)
        for direction in range(2)
        for k in (64, 128, 256, 512, 1024, 2048, 4000)
        for repeat in range(5)
        for view in range(9)
    }
    assert len(seeds) == 2 * 7 * 5 * 9


def test_pc_summary_uses_only_requested_indices() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=(40, 12)).astype(np.float32)
    indices = np.arange(5, 25)
    changed = values.copy()
    changed[:5] += 1000
    changed[25:] -= 1000
    first = QU.FT.fit_pc_summary(values, indices, 8, device="cpu", seed=17)
    second = QU.FT.fit_pc_summary(changed, indices, 8, device="cpu", seed=17)
    assert np.array_equal(first.mean, second.mean)
    assert np.array_equal(first.basis, second.basis)
    assert np.array_equal(first.scale, second.scale)


def test_report_context_diagnostics_do_not_duplicate_writers() -> None:
    scaling = {"unpaired_alignment": {}}
    for direction_index, direction in enumerate(("q_to_l", "l_to_q")):
        scaling["unpaired_alignment"][direction] = {}
        for writer in ("qwriter", "lwriter"):
            context = [float(direction_index), float(direction_index + 1)]
            answer = [float(direction_index + (writer == "lwriter")), 3.0]
            scaling["unpaired_alignment"][direction][writer] = {
                "64": {
                    "bridge_diagnostics": {
                        "context_paired_test_cosine": {
                            "values": context,
                            "median": float(np.median(context)),
                        },
                        "answer_paired_test_cosine": {
                            "values": answer,
                            "median": float(np.median(answer)),
                        },
                    }
                }
            }
    context_values = RP._pooled_bridge_values(scaling, 64, "context_paired_test_cosine")
    answer_values = RP._pooled_bridge_values(scaling, 64, "answer_paired_test_cosine")
    assert context_values.tolist() == [0.0, 1.0, 1.0, 2.0]
    assert len(answer_values) == 8
    assert RP._aggregate_bridge_cell_center(scaling, 64, "context_paired_test_cosine") == 1.0


def test_report_rejects_scaling_test_roster_mismatch() -> None:
    mapping = {"test_roster_sha256": "same"}
    transfer = {"test_roster_sha256": "same"}
    with pytest.raises(ValueError, match="scaling/unpaired test roster"):
        RP.validate_report_inputs(mapping, transfer, {"test_roster_sha256": "different"})


def test_orthogonal_procrustes_orientation() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    recovered = QU.orthogonal_procrustes(x, x @ q)
    assert np.allclose(recovered, q, atol=1e-10, rtol=1e-10)


def test_unrefined_random_orientation_reference_is_deterministic() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(80, 6))
    target = rng.normal(size=(90, 6))
    first = QU.unrefined_random_orientation_references(
        source,
        target,
        draws=3,
        seed_coordinates=(2569, 1, 64, 0),
        device="cpu",
        chunk_size=32,
    )
    second = QU.unrefined_random_orientation_references(
        source,
        target,
        draws=3,
        seed_coordinates=(2569, 1, 64, 0),
        device="cpu",
        chunk_size=32,
    )
    assert first == second
    assert len(first) == 3
    assert np.isfinite(first).all()


def test_unpaired_bridge_records_first_step_pairs_when_initial_is_best() -> None:
    rng = np.random.default_rng(10)
    source = rng.normal(size=(80, 6))
    target = rng.normal(size=(90, 6))
    bridge = QU.fit_unpaired_bridge(
        source,
        target,
        device="cpu",
        max_iterations=1,
        chunk_size=32,
    )
    assert bridge.iterations == 1
    assert bridge.mutual_pairs > 0


def test_moment_seed_recovers_signed_component_permutation() -> None:
    rng = np.random.default_rng(11)
    shapes = (1.05, 1.5, 2.3, 3.7, 6.0)
    source = np.column_stack([rng.gamma(shape=shape, scale=1.0, size=8000) for shape in shapes])
    independent = np.column_stack(
        [rng.gamma(shape=shape, scale=1.0, size=9000) for shape in shapes]
    )
    source = (source - source.mean(0)) / source.std(0)
    independent = (independent - independent.mean(0)) / independent.std(0)
    truth = np.array(
        [
            [0.0, 0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    target = independent @ truth
    recovered = QU.moment_seed_rotation(source, target)
    assert np.array_equal(recovered, truth)


def test_cholesky_affine_span_recovers_linear_map() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(100, 7)).astype(np.float32)
    operator = rng.normal(size=(7, 5)).astype(np.float32)
    bias = rng.normal(size=5).astype(np.float32)
    y = x @ operator + bias
    pred = QU.affine_span_predict_cholesky(
        x[50:],
        x[:50],
        y[:50],
        ridge_fraction=1e-5,
        device="cpu",
    )
    assert np.allclose(pred, y[50:], atol=2e-3, rtol=2e-3)


def test_cholesky_weights_agree_with_legacy_solver_on_same_inputs() -> None:
    rng = np.random.default_rng(14)
    anchors = rng.normal(size=(40, 17)).astype(np.float32)
    query = rng.normal(size=(23, 17)).astype(np.float32)
    legacy, _ = QU.FT.ridge_kernel_weights(
        query,
        anchors,
        ridge_fraction=0.01,
        device="cpu",
    )
    cholesky = QU.ridge_kernel_weights_cholesky(
        query,
        anchors,
        ridge_fraction=0.01,
        device="cpu",
    )
    assert np.allclose(cholesky, legacy, atol=2e-5, rtol=2e-5)


def test_prediction_repeat_summary_retains_values() -> None:
    rows = []
    for value in (0.1, 0.2, 0.4):
        rows.append(
            {
                "observed_target": {
                    "pooled_r2": value,
                    "train_mean_normalized_r2": value,
                    "centered_cosine": value,
                },
                "full_target_mapping": {
                    "normalized_r2": value,
                    "centered_cosine": value,
                    "relative_l2": value,
                },
            }
        )
    summary = QU.summarize_prediction_repeats(rows)
    assert summary["full_target_mapping"]["centered_cosine"]["values"] == [
        0.1,
        0.2,
        0.4,
    ]
