"""Leakage, frequency-weighting and common-draw tests for trajectory-stage fits."""

import copy
import json
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest

from scripts import context_risk_trajectory_analyze as analysis


def context(task="task_1", role="probe_training"):
    return {
        "task_id": task,
        "condition": "oneoff",
        "public_test_role": role,
        "messages": [{"role": "user", "content": "Compute the value correctly."}],
        "test": "assert solve(1) == 2",
    }


def observation(epoch=1, stage="within_32", observed=17, ended=True, positive=0):
    return {
        "sample_id": "sample_1",
        "epoch": epoch,
        "attempt": 1,
        "stage": stage,
        "task_id": "task_1",
        "condition": "oneoff",
        "public_test_role": "probe_training",
        "checkpoint_key": "cp",
        "context_key": "ctx",
        "positive": positive,
        "request_tokens": 10,
        "observed_tokens": observed,
        "answer_ended": ended,
    }


def test_short_answers_remain_and_identical_inputs_keep_binomial_counts():
    observations = [observation(1, positive=0), observation(2, positive=1)]
    checkpoints = {"cp": {"length": 27, "visible": "prompt: answer ended"}}
    rows = analysis.aggregate_stage(
        "within_32", observations, checkpoints, {"ctx": context()}, {}, lambda cp, _: cp["visible"]
    )
    assert len(rows) == 1
    assert (rows[0]["positive"], rows[0]["trials"]) == (1, 2)
    assert rows[0]["observed_tokens"] == 17 and rows[0]["answer_ended"]
    assert len(rows[0]["events"]) == 2
    duplicate = [*observations, observations[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.aggregate_stage(
            "within_32", duplicate, checkpoints, {"ctx": context()}, {}, lambda cp, _: cp["visible"]
        )


def test_pre_action_rejects_response_metadata_and_future_fields_do_not_enter_bank():
    base = observation(stage="pre_action_01", observed=0, ended=False)
    checkpoints = {"cp": {"length": 10, "visible": "full request history"}}
    args = (
        "pre_action_01",
        [base],
        checkpoints,
        {"ctx": context()},
        {},
        lambda cp, _: cp["visible"],
    )
    row = analysis.aggregate_stage(*args)[0]
    poisoned = {
        **base,
        "terminal_status": "success",
        "final_length": 99999,
        "future_response": "SECRET FUTURE CODE",
        "first_success_attempt": 8,
    }
    changed = analysis.aggregate_stage(args[0], [poisoned], *args[2:])[0]
    assert row == changed
    with pytest.raises(ValueError, match="response information"):
        analysis.aggregate_stage(args[0], [{**base, "observed_tokens": 1}], *args[2:])
    with pytest.raises(ValueError, match="Static context join"):
        analysis.aggregate_stage(args[0], [{**base, "public_test_role": "final_test"}], *args[2:])


def fixture_spec():
    path = (
        Path(__file__).resolve().parents[1]
        / "eval_results/context_risk_trajectory_design/analysis_spec.json"
    )
    spec = json.loads(path.read_text())
    spec["regularization_C"] = [1e-6, 0.01]
    spec["bootstrap"]["replicates"] = 37
    return spec


def test_real_fit_uses_full_history_and_shared_task_folds(tmp_path):
    rng = np.random.default_rng(18)
    rows = []
    for task in range(12):
        role = "probe_training" if task < 6 else "final_test"
        for example in range(2):
            rows.append(
                {
                    **context(f"task_{task:02d}", role),
                    "checkpoint_key": f"cp_{task}_{example}",
                    "context_key": f"ctx_{task}",
                    "attempt": 2,
                    "request_tokens": 30,
                    "observed_tokens": 0,
                    "answer_ended": False,
                    "visible_text": f"full history: previously seen feedback {task} {example}",
                    "positive": example,
                    "trials": 2,
                    "events": [[f"s{task}", example + 1, 2]],
                }
            )
    raw = rng.normal(size=(len(rows), 8))
    mapping = {
        "weight": np.eye(8) * 1.3,
        "x_mean": np.zeros(8),
        "x_scale": np.ones(8),
        "y_mean": np.arange(8) * 0.1,
    }
    spec = fixture_spec()
    bank = analysis.StageFeatureBank(rows, raw, mapping, spec)
    assert bank.prompts.dtype == object
    assert bank.prompts[0] == rows[0]["visible_text"]
    assert bank.metadata.shape == (24, 8)
    result = analysis.fit_stage(
        "pre_action_02", rows, raw, mapping, spec, tmp_path, {"fixture": True}
    )
    assert result["status"] == "complete"
    assert result["test_support"]["n_trajectories"] == 24
    assert result["test_support"]["positive"] == 6
    assert len(result["models"]) == 8
    for fold in json.loads((tmp_path / "folds.json").read_text()):
        assert not set(fold["training_tasks"]) & set(fold["validation_tasks"])
        assert all(int(t.removeprefix("task_")) < 6 for t in fold["training_tasks"])
    assert (
        result["models"]["mapped_plus_metadata"]["feature_checks"]["composed_logit_max_error"]
        < 1e-8
    )


def result_panel(equal=False):
    rng = np.random.default_rng(991)
    results = []
    for stage in analysis.STAGES:
        rows = []
        for t in range(9):
            values = rng.normal(size=3)
            if equal:
                values[:] = values[0]
            rows.append(
                {
                    "task_id": f"task_{t}",
                    "positive": t % 3,
                    "trials": 4 + t % 2,
                    "logits": {
                        "prevalence": float(values[0]),
                        "text_plus_metadata": float(values[0]),
                        "raw_plus_metadata": float(values[1]),
                        "mapped_plus_metadata": float(values[2]),
                    },
                }
            )
        results.append(
            {
                "stage": stage,
                "status": "complete",
                "conditional_claim_support_passed": True,
                "test_predictions": rows,
            }
        )
    return results


def test_joint_bootstrap_matches_serial_resampling_and_scan_maximum():
    panel, spec = result_panel(), fixture_spec()
    report, arrays = analysis.joint_intervals(panel, spec)
    assert arrays["draws"].shape == (37, 15, 3)
    # Independent serial oracle repeats each sampled task's actual binary loss sum.
    for b, sampled in enumerate(arrays["task_indices"]):
        for s, result in enumerate(panel):
            repeated = [result["test_predictions"][i] for i in sampled]
            for c, (_, first, second) in enumerate(analysis.CONTRASTS):
                numerator, denominator = 0.0, 0
                for row in repeated:
                    k, n = row["positive"], row["trials"]
                    x, y = row["logits"][first], row["logits"][second]
                    numerator += k * (np.logaddexp(0, -x) - np.logaddexp(0, -y))
                    numerator += (n - k) * (np.logaddexp(0, x) - np.logaddexp(0, y))
                    denominator += n
                assert arrays["draws"][b, s, c] == pytest.approx(numerator / denominator)
    maxima = np.abs(arrays["draws"] - arrays["estimates"][None]).reshape(37, 45).max(axis=1)
    assert report["simultaneous_half_width"] == pytest.approx(np.quantile(maxima, 0.95))
    assert report["n_contrasts"] == 45
    for s, record in enumerate(report["stages"]):
        for c, (name, _, _) in enumerate(analysis.CONTRASTS):
            contrast = record["comparisons"][name]
            assert contrast["improvement"] == pytest.approx(arrays["estimates"][s, c])
            assert contrast["simultaneous_ci"][0] == pytest.approx(
                contrast["improvement"] - report["simultaneous_half_width"]
            )


def test_zero_signal_never_passes_strict_gate_and_missing_stage_fails():
    panel, spec = result_panel(equal=True), fixture_spec()
    report, arrays = analysis.joint_intervals(panel, spec)
    assert np.count_nonzero(arrays["draws"]) == 0
    assert report["simultaneous_half_width"] == 0
    assert all(
        not row["raw_prediction_supported"] and not row["mapping_benefit_supported"]
        for row in report["stages"]
    )
    with pytest.raises(ValueError, match="all15 stages"):
        analysis.joint_intervals(panel[:-1], spec)
    missing = copy.deepcopy(panel)
    missing[0]["test_predictions"].pop()
    with pytest.raises(ValueError, match="Every frozen test task"):
        analysis.joint_intervals(missing, spec)


def test_snapshot_rejects_symlinked_payload_and_detects_changed_bytes(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = inputs / "vectors.bin"
    payload.write_bytes(b"observed activation bytes")
    before = analysis.input_snapshot([inputs])
    payload.write_bytes(b"changed activation bytes")
    assert analysis.input_snapshot([inputs]) != before
    with pytest.raises(ValueError, match="overlaps input"):
        analysis.assert_disjoint_output(inputs / "analysis", [inputs])
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.bin").write_bytes(b"not traversed by rglob")
    (inputs / "linked").symlink_to(nested, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot contain symlinks"):
        analysis.input_snapshot([inputs])


def run_body_fixture(tmp_path, monkeypatch):
    """Real files/readers with autospecced producer verification and fit seams.

    The producer/capture verification bodies have their own full-body component
    tests; the real grouped fit_stage body is exercised above. No model is loaded.
    """
    from omegaconf import OmegaConf

    from scripts import context_risk_trajectory_capture as capture
    from scripts import context_risk_trajectory_prepare as prepare

    prepared, captured = tmp_path / "prepared", tmp_path / "capture"
    prepared.mkdir()
    captured.mkdir()
    (prepared / "manifest.json").write_text("{}\n")
    checkpoints = []
    contexts = []
    texts = []
    for i in range(2):
        visible = f"exact current request {i}"
        checkpoints.append(
            {
                "checkpoint_key": f"cp{i}",
                "length": 10,
                "visible_text": {
                    "stream_key": f"stream{i}",
                    "prefix_characters": len(visible),
                    "suffix": "",
                },
                "visible_text_sha256": prepare.text_digest(visible),
            }
        )
        contexts.append(
            {
                **context(f"task_{i}", "probe_training" if i == 0 else "final_test"),
                "context_key": f"ctx{i}",
            }
        )
        texts.append({"stream_key": f"stream{i}", "text": visible})
    for filename, rows in (("contexts.jsonl", contexts), ("texts.jsonl", texts)):
        (prepared / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    observations = []
    for n in range(2153 * 6):
        i = n % 2
        observations.append(
            {
                **observation(
                    epoch=n + 1,
                    stage="pre_action_01" if n < 231 else "within_pre",
                    observed=0,
                    ended=False,
                    positive=i,
                ),
                "sample_id": f"sample_{i}",
                "task_id": f"task_{i}",
                "public_test_role": "probe_training" if i == 0 else "final_test",
                "checkpoint_key": f"cp{i}",
                "context_key": f"ctx{i}",
            }
        )
    index = {"checkpoint_keys": ["cp0", "cp1"], "vectors_path": "vectors.npy"}
    (captured / "index.json").write_text(json.dumps(index) + "\n")
    np.save(captured / "vectors.npy", np.ones((2, 5120), dtype=np.float16))
    map_path = tmp_path / "map.npz"
    np.savez(map_path, weight=np.eye(8), x_mean=np.zeros(8), x_scale=np.ones(8), y_mean=np.zeros(8))
    spec = fixture_spec()
    spec["map_sha256"] = analysis.sha256(map_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec) + "\n")
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "analysis_sources_sha256": analysis.source_hashes(),
                "analysis_spec_sha256": analysis.sha256(spec_path),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        prepare,
        "load_prepared",
        create_autospec(prepare.load_prepared, return_value=({}, {}, checkpoints, observations)),
    )
    monkeypatch.setattr(
        capture,
        "verify_capture",
        create_autospec(
            capture.verify_capture,
            return_value=(index, np.load(captured / "vectors.npy", mmap_mode="r")),
        ),
    )
    monkeypatch.setattr(
        capture,
        "verify_capture_geometry",
        create_autospec(capture.verify_capture_geometry, return_value=None),
    )
    cfg = OmegaConf.create(
        {
            "prepared": str(prepared),
            "capture": str(captured),
            "output_dir": str(tmp_path / "analysis"),
            "spec": str(spec_path),
            "map": str(map_path),
            "review": str(review_path),
            "stage": "pre_action_01",
        }
    )
    return cfg


@pytest.mark.parametrize("mutation", [None, "consumed_payload", "during_fit", "imported_helper"])
def test_run_body_binds_inputs_before_consumption_and_before_pass(tmp_path, monkeypatch, mutation):
    from scripts import context_risk_trajectory_prepare as prepare

    cfg = run_body_fixture(tmp_path, monkeypatch)
    actual_load_text_bank = prepare.load_text_bank
    if mutation == "consumed_payload":

        def changed_text_bank(path):
            value = actual_load_text_bank(path)
            with (Path(path) / "texts.jsonl").open("a") as stream:
                stream.write("\n")
            return value

        monkeypatch.setattr(
            prepare,
            "load_text_bank",
            create_autospec(actual_load_text_bank, side_effect=changed_text_bank),
        )
    elif mutation == "imported_helper":
        from scripts import context_risk_followup_features

        monkeypatch.setattr(context_risk_followup_features, "__file__", str(tmp_path / "shadow.py"))

    def fit_seam(stage, rows, raw, map_arrays, spec, out, provenance):
        assert stage == "pre_action_01" and len(rows) == 2 and raw.shape == (2, 5120)
        assert sum(row["trials"] for row in rows) == 231
        assert map_arrays["weight"].shape == (8, 8)
        assert provenance["input_files_sha256"]
        if mutation == "during_fit":
            (Path(cfg.capture) / "vectors.npy").write_bytes(
                b"payload changed after capture verification"
            )
        return {"status": "fixture_only"}

    monkeypatch.setattr(
        analysis, "fit_stage", create_autospec(analysis.fit_stage, side_effect=fit_seam)
    )
    if mutation is None:
        result = analysis.run(cfg)
        assert result["status"] == "stage_subset_complete"
        assert (Path(cfg.output_dir) / "result.json").is_file()
    else:
        with pytest.raises(ValueError, match=r"changed|shadowed"):
            analysis.run(cfg)
        assert not (Path(cfg.output_dir) / "result.json").exists()
