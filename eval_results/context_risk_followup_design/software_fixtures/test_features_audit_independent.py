"""Small algebra and mocked-native audit fixtures; no experiment fits or generation."""

import copy
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)

from scripts import context_risk_followup as collection
from scripts import context_risk_followup_audit as audit
from scripts.context_risk_followup_features import FeatureBank, metadata_matrix

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (ROOT / "eval_results/context_risk_followup_design/analysis_spec.json").read_text()
)


def feature_fixture():
    """Create a six-dimensional affine map and eight distinct software-only contexts."""
    rng = np.random.default_rng(904)
    rows = [
        {
            "messages": [{"role": "user", "content": f"shared fixture instruction {i}"}],
            "test": "def check(f):\n    assert f(1) == 2\n",
            "condition": "oneoff" if i % 2 else "conflicting",
            "original_success_count": i % 4,
        }
        for i in range(8)
    ]
    rows[-1]["messages"][0]["content"] += " ΩΩΩ heldout_only_marker"
    raw = rng.normal(size=(8, 6))
    arrays = {
        "weight": rng.normal(size=(6, 6)),
        "x_mean": rng.normal(size=6),
        "x_scale": np.arange(1, 7) / 3,
        "y_mean": rng.normal(size=6),
    }
    return rows, raw, arrays, np.arange(6), np.arange(6, 8), np.arange(8) % 4 + 1


def test_map_composition_and_signed_input_control_keep_common_metric():
    """Check exact row-coordinate algebra and singular values in six dimensions."""
    rows, raw, arrays, train, test, n = feature_fixture()
    bank = FeatureBank(rows, raw, arrays, SPEC)
    prepared = bank.prepare(train, test, n)
    scales = prepared["raw_scaler"].scale_ / arrays["x_scale"]
    operator = scales[:, None] * arrays["weight"] / prepared["mapped_scaler"].scale_
    assert np.allclose(prepared["raw"][1] @ operator, prepared["mapped"][1])
    coefficient = np.arange(10, dtype=float) / 7
    _, mapped, _ = bank.features("mapped_plus_metadata", train, test, n)
    assert np.allclose(
        bank.compose_mapped_head(train, test, n, coefficient, 0.3),
        mapped @ coefficient + 0.3,
        atol=1e-11,
    )
    for seed in SPEC["orientation_controls"]["seeds"]:
        left, right, _ = bank.features(f"orientation_{seed}", train, test, n)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(6)
        signs = rng.choice([-1.0, 1.0], size=6)
        q = np.eye(6)[:, permutation] * signs
        assert np.allclose(right[:, :6], prepared["raw"][1] @ q @ operator)
        assert np.allclose(left[:, :6], prepared["raw"][0] @ q @ operator)
        assert np.allclose(
            np.linalg.svd(q @ operator, compute_uv=False),
            np.linalg.svd(operator, compute_uv=False),
        )
        assert np.array_equal(right[:, 6:], prepared["metadata"][1])


def test_training_transforms_do_not_depend_on_heldout_inputs_or_outcomes():
    """Change held-out text/activations and irrelevant outcomes; training stays fixed."""
    rows, raw, arrays, train, test, n = feature_fixture()
    changed_rows = copy.deepcopy(rows)
    for row in changed_rows:
        row["original_success_count"] = 999
    changed_rows[-1]["messages"][0]["content"] = "ΦΦΦ unrelated heldout text"
    changed_raw = raw.copy()
    changed_raw[test] *= 1000
    first, second = (
        FeatureBank(rows, raw, arrays, SPEC),
        FeatureBank(changed_rows, changed_raw, arrays, SPEC),
    )
    assert np.array_equal(metadata_matrix(rows)[:6], metadata_matrix(changed_rows)[:6])
    for method in ("raw_plus_metadata", "mapped_plus_metadata", "text_plus_metadata"):
        left, _, _ = first.features(method, train, test, n)
        other, _, _ = second.features(method, train, test, n)
        assert np.allclose(left, other)
    vocabulary = first.prepare(train, test, n)["text"][2].vocabulary_
    assert "ΩΩΩ" not in vocabulary
    assert first.prepare(train, test, n)["identity_bias_max_error"] < 1e-11


def test_pca_is_training_only_unwhitened_and_nominal_rank_clipping_is_consistent():
    """Duplicate effective ranks preserve distinct nominal keys without refitting data."""
    rows, raw, arrays, train, test, n = feature_fixture()
    first = FeatureBank(rows, raw, arrays, SPEC)
    left, right, info = first.features("pca_plus_metadata", train, test, n, rank=8)
    other_left, other_right, other_info = first.features(
        "pca_plus_metadata", train, test, n, rank=16
    )
    assert info["nominal_rank"] == 8 and other_info["nominal_rank"] == 16
    assert info["effective_rank"] == other_info["effective_rank"] == 5
    assert np.array_equal(left, other_left) and np.array_equal(right, other_right)
    pca = first.prepare(train, test, n)["pca"][2]
    assert pca.whiten is False and pca.svd_solver == "full"
    changed = raw.copy()
    changed[test] += 1000
    second = FeatureBank(rows, changed, arrays, SPEC)
    changed_left, _, _ = second.features("pca_plus_metadata", train, test, n, rank=8)
    assert np.allclose(left, changed_left)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, 1e-12, np.nan, np.inf])
def test_map_scale_is_rejected_without_silent_repair(bad_scale):
    rows, raw, arrays, _, _, _ = feature_fixture()
    arrays["x_scale"][0] = bad_scale
    with pytest.raises(ValueError):
        FeatureBank(rows, raw, arrays, SPEC)


def native_fixture(tmp_path):
    """Reuse the saved collection fixtures, adding full typed model-event histories."""
    path = Path(__file__).with_name("followup_review.py")
    loader = importlib.util.spec_from_file_location("independent_collection_fixture", path)
    helpers = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helpers)
    cfg, _ = helpers.setup(tmp_path)
    _, log = helpers.native(tmp_path, cfg, pilot=True)
    started = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    log.stats = NS(
        started_at=started.isoformat(), completed_at=(started + timedelta(hours=1)).isoformat()
    )
    for sample in log.samples:
        messages = [ChatMessageUser(content=sample.input)]
        events = []
        history = sample.metadata["agentic_results"]["attempt_history"]
        for index, attempt in enumerate(history):
            attempt.update(stdout="", stderr="fixture failure" if not attempt["success"] else "")
            output = ModelOutput.from_content(
                str(cfg.model).removeprefix("openai-api/local/"), attempt["response"]
            )
            output.usage = ModelUsage(input_tokens=100, output_tokens=8, total_tokens=108)
            event = NS(
                event="model",
                error=None,
                model=str(cfg.model),
                output=output,
                timestamp=started + timedelta(seconds=2 * index),
                completed=started + timedelta(seconds=2 * index + 1),
                input=copy.deepcopy(messages),
                config=GenerateConfig(
                    max_tokens=65536,
                    temperature=1,
                    top_p=1,
                    max_connections=16,
                    max_retries=2,
                    extra_body=collection.GENERATION_EXTRA_BODY,
                    seed=attempt["request_seed"],
                ),
            )
            event.model_dump = lambda **kwargs: {"event": "model", "error": "fixture retry"}
            events.append(event)
            messages.append(ChatMessageAssistant(content=attempt["response"]))
            if index < len(history) - 1:
                last_error = attempt["stderr"] or attempt["stdout"] or "[unknown error]"
                messages.append(
                    ChatMessageUser(
                        content=(
                            "\nYour previous attempt failed the tests. Here's the error:\n"
                            f"{last_error}\n\n\n\nTo reiterate, this is your task: "
                            f"{sample.metadata['instruction_prompt']}"
                        )
                    )
                )
        sample.events, sample.messages = events, messages
        sample.total_time, sample.working_time = 30.0, 20.0
    report = helpers.saved_report(tmp_path, "A", log)
    report.update(is_pilot=True, requested_rollouts=32, realized_rollouts=32)
    report_path = tmp_path / "development_A/run_result.json"
    report_path.write_text(json.dumps(report))
    return log, report, report_path, tmp_path / "manifests/development_A.jsonl"


def run_mocked_audit(tmp_path, log, report_path, manifest):
    """The only native loader is mocked; any accidental network/model call is absent."""
    with patch.object(audit, "read_eval_log", return_value=log) as read:
        result = audit.audit(report_path, manifest, tmp_path / "audit_output")
    assert read.call_args.kwargs == {"resolve_attachments": "full"}
    return result


def test_valid_audit_and_identical_transport_retry_are_preserved(tmp_path):
    log, _, report_path, manifest = native_fixture(tmp_path)
    retry = copy.deepcopy(log.samples[0].events[0])
    retry.error = "fixture transient transport error"
    retry.output = None
    log.samples[0].events.insert(0, retry)
    result = run_mocked_audit(tmp_path, log, report_path, manifest)
    assert result["passed"] and result["realized_rollouts"] == 32
    assert result["request_error_events"] == 1
    assert result["verified_request_seeds"] == sum(
        len(s.metadata["agentic_results"]["attempt_history"]) for s in log.samples
    )
    rows = [
        json.loads(s)
        for s in (tmp_path / "audit_output/audited_rollouts.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 32 and sum(len(r["request_errors"]) for r in rows) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "freeze",
        "requested",
        "epochs",
        "coverage",
        "top_p",
        "extra_body",
        "event_model",
        "output_model",
        "input",
        "response",
        "seed",
        "negative_duration",
        "negative_tokens",
        "retry_seed",
        "trailing_error",
        "nonstop",
        "missing_event",
        "score",
    ],
)
def test_audit_rejects_native_or_report_inconsistency(tmp_path, defect):  # noqa: C901
    log, report, report_path, manifest = native_fixture(tmp_path)
    sample = log.samples[0]
    event = sample.events[0]
    if defect == "freeze":
        log.eval.metadata["freeze_sha256"] = "wrong"
    elif defect in {"requested", "epochs", "coverage"}:
        report[
            {
                "requested": "requested_rollouts",
                "epochs": "epochs",
                "coverage": "coverage_complete",
            }[defect]
        ] = 999 if defect != "coverage" else False
        report_path.write_text(json.dumps(report))
    elif defect == "top_p":
        event.config.top_p = 0.5
    elif defect == "extra_body":
        event.config.extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
    elif defect == "event_model":
        event.model = "wrong/model"
    elif defect == "output_model":
        event.output.model = "wrong/model"
    elif defect == "input":
        event.input[0].content = "changed initial prompt"
    elif defect == "response":
        usage = event.output.usage
        event.output = ModelOutput.from_content(event.output.model, "changed completion")
        event.output.usage = usage
    elif defect == "seed":
        event.config.seed = 0
    elif defect == "negative_duration":
        event.completed = event.timestamp - timedelta(seconds=1)
    elif defect == "negative_tokens":
        event.output.usage.input_tokens = -1
    elif defect in {"retry_seed", "trailing_error"}:
        error = copy.deepcopy(event)
        error.error = "fixture retry"
        error.output = None
        if defect == "retry_seed":
            error.config.seed = 0
            sample.events.insert(0, error)
        else:
            sample.events.append(error)
    elif defect == "nonstop":
        event.output.choices[0].stop_reason = "max_tokens"
        sample.metadata["agentic_results"]["attempt_history"][0]["stop_reasons"] = ["max_tokens"]
    elif defect == "missing_event":
        sample.events.pop()
    elif defect == "score":
        sample.scores["successful_submission"].value = "N"
    with pytest.raises((ValueError, KeyError)):
        run_mocked_audit(tmp_path, log, report_path, manifest)
