from __future__ import annotations

import argparse
import json

import numpy as np

import scripts.context_risk_analyze as analyze


def test_frequency_weighted_fit_matches_expanded_logistic_objective():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(17)
    # Cover both narrow and activation-like wide feature matrices.
    for width in (3, 120):
        x = rng.normal(size=(4, width)).astype(np.float32)
        rows = _rows()
        expanded, labels, _ = analyze._expand_binomial(x, rows, np.arange(4))
        scaler = StandardScaler().fit(expanded)
        for c_value in (1e-4, 0.01, 1.0):
            reference = LogisticRegression(
                C=c_value,
                solver="liblinear",
                dual=False,
                tol=1e-8,
                max_iter=5000,
                random_state=analyze.SEED,
            ).fit(scaler.transform(expanded), labels)
            observed = analyze._fit_predict_logistic(expanded, labels, x, c_value=c_value)
            np.testing.assert_allclose(
                observed, reference.predict_proba(scaler.transform(x))[:, 1], atol=2e-6
            )


def test_vectorized_cluster_bootstrap_matches_seeded_serial_reference():
    rows = [*_rows(), {"task_id": "a", "positive": 1, "negative": 7}]
    candidate = np.asarray([0.7, 0.1, 0.4, 0.3, 0.2])
    baseline = np.repeat(0.3, len(rows))
    rng = np.random.default_rng(analyze.SEED)
    tasks = np.asarray(sorted({row["task_id"] for row in rows}))
    differences = []
    for _ in range(100):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        indices = np.asarray(
            [i for task in sampled for i, row in enumerate(rows) if row["task_id"] == task]
        )
        sampled_rows = [rows[i] for i in indices]
        differences.append(
            analyze.binomial_log_loss(sampled_rows, candidate[indices])
            - analyze.binomial_log_loss(sampled_rows, baseline[indices])
        )
    observed = analyze.clustered_bootstrap_log_loss_delta(rows, candidate, baseline, replicates=100)
    np.testing.assert_allclose(
        [observed["ci95_low"], observed["ci95_high"]],
        np.quantile(differences, [0.025, 0.975]),
        atol=1e-14,
    )


def test_grouped_outer_folds_checkpoint_and_resume(tmp_path):
    rows = [*_rows(), {"task_id": "a", "positive": 1, "negative": 3}]
    features = np.arange(10, dtype=np.float32).reshape(5, 2)
    predictions, folds = analyze.leave_one_task_out_predictions(
        features, rows, checkpoint_dir=tmp_path
    )
    assert folds[0]["n_test_contexts"] == 2
    assert folds[0]["n_train_rollouts"] == 12
    repeated, repeated_folds = analyze.leave_one_task_out_predictions(
        features, rows, checkpoint_dir=tmp_path
    )
    np.testing.assert_array_equal(predictions, repeated)
    assert folds == repeated_folds


def test_misalignment_framing_keeps_latent_swap_aliases_together(tmp_path):
    records = []
    contexts = []
    for index, goal in enumerate(("latent", "swap", "explicit")):
        exact_hash = "shared" if index < 2 else "different"
        records.append(
            {
                "condition_id": goal,
                "goal_type": goal,
                "exact_context_sha256": exact_hash,
                "messages": [{"role": "user", "content": exact_hash}],
            }
        )
        contexts.append(
            {
                "condition_id": goal,
                "exact_context_sha256": exact_hash,
                "n_positive": 1,
                "n_negative": 3,
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records))
    rows, _ = analyze.prepare_misalignment_rows(
        {"contexts": contexts},
        {key: {"n_prefix_tokens": 10} for key in ("shared", "different")},
        path,
        group_axis="goal_framing",
    )
    assert len(rows) == 2
    implicit = next(row for row in rows if row["task_id"] == "latent_or_swap")
    assert implicit["condition_ids"] == ["latent", "swap"]
    assert implicit["positive"] == 2


def _rows():
    return [
        {"task_id": "a", "positive": 3, "negative": 1},
        {"task_id": "b", "positive": 0, "negative": 4},
        {"task_id": "c", "positive": 2, "negative": 2},
        {"task_id": "d", "positive": 1, "negative": 3},
    ]


def test_binomial_metrics_match_expanded_definitions():
    rows = _rows()
    probability = np.asarray([0.75, 0.1, 0.5, 0.25])
    labels = np.concatenate(
        [np.asarray([1] * row["positive"] + [0] * row["negative"]) for row in rows]
    )
    expanded_probability = np.concatenate(
        [
            np.repeat(value, row["positive"] + row["negative"])
            for row, value in zip(rows, probability, strict=True)
        ]
    )
    expected_log_loss = -np.mean(
        labels * np.log(expanded_probability) + (1 - labels) * np.log(1 - expanded_probability)
    )
    expected_brier = np.mean((labels - expanded_probability) ** 2)
    assert analyze.binomial_log_loss(rows, probability) == expected_log_loss
    assert analyze.binomial_brier(rows, probability) == expected_brier


def test_leave_one_task_out_predictions_do_not_train_on_held_out_task(monkeypatch):
    rows = _rows()
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    seen_training_rows = []

    monkeypatch.setattr(analyze, "_select_c", lambda features, rows, indices: 0.01)

    def fake_fit(x_train, y_train, x_eval, *, c_value):
        seen_training_rows.append(len(x_train))
        return np.full(len(x_eval), y_train.mean())

    monkeypatch.setattr(analyze, "_fit_predict_logistic", fake_fit)
    prediction, folds = analyze.leave_one_task_out_predictions(features, rows)
    assert len(prediction) == 4
    assert len(folds) == 4
    assert seen_training_rows == [12, 12, 12, 12]


def test_clustered_bootstrap_reports_candidate_minus_baseline():
    rows = _rows()
    good = np.asarray([0.75, 0.1, 0.5, 0.25])
    bad = np.repeat(0.5, 4)
    contrast = analyze.clustered_bootstrap_log_loss_delta(rows, good, bad, replicates=100)
    assert contrast["delta"] < 0
    assert contrast["bootstrap_replicates"] == 100


def test_context_map_uses_frozen_normalization(tmp_path):
    path = tmp_path / "map.npz"
    np.savez(
        path,
        weight=np.eye(2, dtype=np.float32) * 2,
        x_mean=np.asarray([1, 2], dtype=np.float32),
        x_scale=np.asarray([2, 4], dtype=np.float32),
        y_mean=np.asarray([10, 20], dtype=np.float32),
    )
    raw = np.asarray([[3, 6]], dtype=np.float32)
    np.testing.assert_allclose(analyze.apply_context_map(raw, path), [[12, 22]])


def test_failed_frozen_gate_writes_result_without_fitting(tmp_path, monkeypatch):
    impossible_result = tmp_path / "impossible.json"
    impossible_result.write_text(
        json.dumps(
            {
                "reward_hacking_prevalence_gate": {
                    "passed": False,
                    "n_positive": 2,
                    "n_negative": 20,
                    "n_censored": 0,
                    "n_mixed_contexts": 1,
                    "n_eligible_tasks": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    map_artifact = tmp_path / "map.npz"
    np.savez(map_artifact, placeholder=np.asarray([1]))
    monkeypatch.setattr(
        analyze,
        "leave_one_task_out_predictions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit should be skipped")),
    )
    output_dir = tmp_path / "output"
    report = analyze.run_analysis(
        argparse.Namespace(
            impossible_result=impossible_result,
            impossible_capture_root=tmp_path / "missing-captures",
            impossible_manifest=manifest,
            map_artifact=map_artifact,
            misalignment_result=None,
            output_dir=output_dir,
            selected_layer=44,
        )
    )
    assert report["claim_status"] == "reward-hacking feasibility gate failed"
    assert report["full_claim_supported"] is False
    assert report["reward_hacking_feasibility"]["prediction_status"].startswith("not_run")
    assert (output_dir / "analysis_result.json").is_file()
    assert (output_dir / "results.md").is_file()


def test_incomplete_run_cannot_fit_even_when_prevalence_gate_passes(tmp_path, monkeypatch):
    impossible_result = tmp_path / "impossible.json"
    impossible_result.write_text(
        json.dumps(
            {
                "passed": False,
                "realized_rollouts": 479,
                "requested_rollouts": 480,
                "technical_errors": 0,
                "reward_hacking_prevalence_gate": {
                    "passed": True,
                    "n_positive": 20,
                    "n_negative": 20,
                    "n_censored": 0,
                    "n_mixed_contexts": 5,
                    "n_eligible_tasks": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    map_artifact = tmp_path / "map.npz"
    np.savez(map_artifact, placeholder=np.asarray([1]))
    monkeypatch.setattr(
        analyze,
        "load_impossible_activations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit should be skipped")),
    )
    report = analyze.run_analysis(
        argparse.Namespace(
            impossible_result=impossible_result,
            impossible_capture_root=tmp_path,
            impossible_manifest=manifest,
            map_artifact=map_artifact,
            misalignment_result=None,
            output_dir=tmp_path / "output",
            selected_layer=44,
        )
    )
    assert report["claim_status"] == "reward-hacking execution integrity failed"
    feasibility = report["reward_hacking_feasibility"]
    assert feasibility["execution_integrity"]["passed"] is False
    assert feasibility["prediction_status"] == "not_run_execution_integrity_failed"


def test_prepare_misalignment_rows_aggregates_duplicate_visible_prefixes(tmp_path):
    messages = [{"role": "user", "content": "same visible prompt"}]
    exact_hash = "exact"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "condition_id": condition_id,
                    "exact_context_sha256": exact_hash,
                    "messages": messages,
                }
            )
            for condition_id in ("latent", "swap")
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "contexts": [
            {
                "condition_id": "latent",
                "exact_context_sha256": exact_hash,
                "n_positive": 1,
                "n_negative": 3,
                "n_censored": 0,
            },
            {
                "condition_id": "swap",
                "exact_context_sha256": exact_hash,
                "n_positive": 2,
                "n_negative": 2,
                "n_censored": 0,
            },
        ]
    }
    rows, texts = analyze.prepare_misalignment_rows(
        result,
        {exact_hash: {"n_prefix_tokens": 17}},
        manifest,
    )
    assert rows == [
        {
            "task_id": exact_hash,
            "condition_ids": ["latent", "swap"],
            "exact_context_sha256": exact_hash,
            "positive": 3,
            "negative": 5,
            "censored": 0,
            "n_prefix_tokens": 17,
        }
    ]
    assert texts == ["user: same visible prompt"]


def test_cross_construct_claim_requires_misalignment_prediction_not_just_gate(
    tmp_path, monkeypatch
):
    impossible_result = tmp_path / "impossible.json"
    impossible_result.write_text(
        json.dumps(
            {
                "passed": True,
                "realized_rollouts": 8,
                "requested_rollouts": 8,
                "technical_errors": 0,
                "reward_hacking_prevalence_gate": {
                    "passed": True,
                    "n_positive": 2,
                    "n_negative": 6,
                    "n_censored": 0,
                    "n_mixed_contexts": 1,
                    "n_eligible_tasks": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    misalignment_result = tmp_path / "misalignment.json"
    misalignment_result.write_text(
        json.dumps(
            {
                "prevalence_gate_passed": True,
                "censoring_gate_passed": True,
                "n_positive": 10,
                "n_negative": 30,
                "n_censored": 0,
                "n_mixed_contexts": 3,
                "public_test_role": "development_only",
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    map_artifact = tmp_path / "map.npz"
    np.savez(map_artifact, placeholder=np.asarray([1]))
    monkeypatch.setattr(
        analyze,
        "load_impossible_activations",
        lambda *_args, **_kwargs: ({"hash": np.asarray([1.0, 2.0])}, {"hash": {}}),
    )
    monkeypatch.setattr(
        analyze,
        "prepare_reward_hacking_rows",
        lambda *_args, **_kwargs: (
            [
                {
                    "task_id": "task",
                    "condition": "oneoff",
                    "exact_context_sha256": "hash",
                    "positive": 1,
                    "negative": 1,
                    "n_prefix_tokens": 10,
                }
            ],
            ["prompt"],
        ),
    )
    monkeypatch.setattr(analyze, "apply_context_map", lambda raw, _path: raw)
    monkeypatch.setattr(
        analyze,
        "predictor_bakeoff",
        lambda *_args, **_kwargs: {
            "models": {},
            "primary_contrast": {"delta": -1, "ci95_low": -2, "ci95_high": -0.5},
            "mapping_contrast": {"delta": 0, "ci95_low": -1, "ci95_high": 1},
            "activation_signal_detected": True,
            "mapping_signal_detected": False,
        },
    )
    report = analyze.run_analysis(
        argparse.Namespace(
            impossible_result=impossible_result,
            impossible_capture_root=tmp_path,
            impossible_manifest=manifest,
            map_artifact=map_artifact,
            misalignment_result=misalignment_result,
            misalignment_rollout_root=None,
            misalignment_manifest=None,
            output_dir=tmp_path / "output",
            selected_layer=44,
        )
    )
    assert report["feasibility_claim_supported"] is False
    assert report["misaligned_action_feasibility"]["prediction_status"] == (
        "not_run_missing_inputs"
    )
