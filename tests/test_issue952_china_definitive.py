from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DV = _load("i952_dv", "issue952_china_definitive_dv.py")
BANK = _load("i952_bank", "issue952_china_definitive_bank.py")
GPU = _load("i952_gpu", "issue952_china_definitive_gpu.py")
ANALYSIS = _load("i952_analysis", "issue952_china_definitive_analysis.py")
CODEX = _load("i952_codex", "issue952_codex_judges.py")


def test_refusal_parser_is_strict() -> None:
    assert DV.parse_refusal("<refusal>yes</refusal>") is True
    assert DV.parse_refusal("  <refusal>NO</refusal>\n") is False
    assert DV.parse_refusal("yes") is None
    assert DV.parse_refusal("reason <refusal>yes</refusal>") is None
    assert DV.parse_refusal(None) is None


def test_request_fingerprint_binds_every_instrument_field() -> None:
    base = DV.request_fingerprint("q", "a")
    assert base == DV.request_fingerprint("q", "a")
    assert base != DV.request_fingerprint("q2", "a")
    assert base != DV.request_fingerprint("q", "a2")


def test_codex_judge_assignment_is_deterministic_and_overlap_is_symmetric() -> None:
    first = CODEX._assignment("opaque-1", 1.0)
    second = CODEX._assignment("opaque-1", 1.0)
    assert first == second
    assert first[0] in CODEX.AGENTS
    assert first[1] == list(CODEX.AGENTS)
    primary, assigned = CODEX._assignment("opaque-2", 0.0)
    assert assigned == [primary]


def test_codex_judge_packet_contains_no_design_metadata() -> None:
    packet = CODEX._packet_payload(
        [{"opaque_id": "opaque", "question": "q", "response": "a"}]
    )
    assert set(packet["items"][0]) == {"opaque_id", "question", "response"}
    rendered = json.dumps(packet["items"])
    for forbidden in ("topic", "language", "source_family", "reference_refusal"):
        assert forbidden not in rendered


def test_codex_judge_confusion_matches_existing_oracle() -> None:
    y = np.array([True, True, False, False])
    pred = np.array([True, False, True, False])
    assert CODEX._confusion(y, pred) == DV._confusion(y, pred)


def test_confusion_metrics() -> None:
    out = DV._confusion(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert out["tp"] == out["tn"] == out["fp"] == out["fn"] == 1
    assert out["balanced_accuracy"] == 0.5
    assert out["precision"] == 0.5
    assert out["recall"] == 0.5
    assert out["kappa"] == 0.0


def test_confusion_empty_arm_fails_cleanly_without_division_by_zero() -> None:
    out = DV._confusion(np.array([], dtype=bool), np.array([], dtype=bool))
    assert out["n"] == 0
    assert np.isnan(out["balanced_accuracy"])
    assert np.isnan(out["kappa"])
    assert np.isnan(DV._rate_error_upper([], 1))


def test_calibration_transport_failures_are_not_reused(tmp_path: Path) -> None:
    result_path = tmp_path / "raw_classifier.jsonl"
    prior = [
        {"item_id": "ok", "transport_error": None},
        {"item_id": "retry", "transport_error": "401"},
    ]
    DV._write_jsonl(result_path, prior)
    prior_bytes = result_path.read_bytes()
    reusable, retry = DV.split_calibration_checkpoints(prior)
    assert [row["item_id"] for row in reusable] == ["ok"]
    assert [row["item_id"] for row in retry] == ["retry"]
    try:
        DV.commit_calibration_preflight(
            result_path,
            reusable,
            retry,
            {"item_id": "retry", "transport_error": "401"},
        )
    except RuntimeError as exc:
        assert "no calibration fan-out" in str(exc)
    else:
        raise AssertionError("transport-failed classifier preflight was accepted")
    assert result_path.read_bytes() == prior_bytes

    passed = {"item_id": "retry", "transport_error": None}
    DV.commit_calibration_preflight(result_path, reusable, retry, passed)
    assert DV._jsonl(result_path) == [prior[0], passed]


def test_audit_gate_requires_all_fields_and_threshold() -> None:
    value = {
        **{key: True for key in BANK.AUDIT_BOOL_KEYS},
        **{key: 80 for key in BANK.AUDIT_SCORE_KEYS},
        "issues": "",
    }
    assert BANK.audit_pass({"value": value})
    value["match_quality_score"] = 79
    assert not BANK.audit_pass({"value": value})
    value["match_quality_score"] = 80
    value["control_is_non_china"] = False
    assert not BANK.audit_pass({"value": value})


def test_control_key_normalization() -> None:
    assert BANK._normalized_key("  United-States / Watergate ") == "united-states-watergate"


def _write_bank_fixture(tmp_path: Path) -> tuple[Path, Path]:
    bank = tmp_path / "bank.jsonl"
    rows = []
    for source in range(90):
        for language in BANK.LANGUAGES:
            for content in BANK.CONTENTS:
                for frame in ("direct", "academic"):
                    rows.append(
                        {
                            "item_id": f"x-{source}-{language}-{content}-{frame}",
                            "source_prompt_id": f"s{source:03d}",
                            "topic": f"t{source % 12}",
                            "language": language,
                            "content": content,
                            "frame": frame,
                            "prompt": "fixture",
                            "audit_pass": True,
                        }
                    )
    GPU._write_jsonl(bank, rows)
    audit = tmp_path / "audit.json"
    GPU._write_json(
        audit,
        {"passed": True, "prompt_bank_sha256": GPU._sha256(bank), "passing_item_ids": []},
    )
    return bank, audit


def test_gpu_bank_fixture_and_smoke_are_exact(tmp_path: Path) -> None:
    bank, audit = _write_bank_fixture(tmp_path)
    full, _ = GPU._load_bank(bank, audit, smoke=False)
    smoke, _ = GPU._load_bank(bank, audit, smoke=True)
    assert len(full) == 1080
    assert len(smoke) == 120
    assert len({row["source_prompt_id"] for row in smoke}) == 10


def test_production_pilot_selects_52_per_crossed_arm(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration"
    DV._write_jsonl(calibration / "items.jsonl", [])
    DV._write_jsonl(calibration / "raw_classifier.jsonl", [])
    DV._write_json(
        calibration / "manifest.json",
        {
            "model": DV.MODEL,
            "rubric_sha256": DV.RUBRIC_SHA256,
            "items_sha256": DV._sha256(calibration / "items.jsonl"),
        },
    )
    DV._write_json(
        calibration / "report.json",
        {
            "passed": True,
            "model": DV.MODEL,
            "rubric_sha256": DV.RUBRIC_SHA256,
            "inputs": {
                "manifest_sha256": DV._sha256(calibration / "manifest.json"),
                "items_sha256": DV._sha256(calibration / "items.jsonl"),
                "raw_classifier_sha256": DV._sha256(calibration / "raw_classifier.jsonl"),
            },
        },
    )
    rows = []
    for language in BANK.LANGUAGES:
        for content in BANK.CONTENTS:
            for frame in ("direct", "academic"):
                for i in range(60):
                    rows.append(
                        {
                            "prompt_id": f"p-{language}-{content}-{frame}-{i}",
                            "item_id": f"i-{language}-{content}-{frame}-{i}",
                            "source_prompt_id": f"s-{i}",
                            "topic": "fixture",
                            "language": language,
                            "content": content,
                            "frame": frame,
                            "draw": i % 8,
                            "question": "fixture q",
                            "text": "fixture a",
                        }
                    )
    rollouts = tmp_path / "rollouts.jsonl"
    DV._write_jsonl(rollouts, rows)
    DV._write_json(
        tmp_path / "judge" / "production_stage.json",
        {
            "rollouts_sha256": DV._sha256(rollouts),
            "files": {"raw_completions/rollouts.jsonl": DV._sha256(rollouts)},
        },
    )
    path = DV.prepare_production(tmp_path, rollouts, pilot=True)
    requests = DV._jsonl(path)
    lookup = json.loads((tmp_path / "judge" / "pilot_lookup.json").read_text())
    assert len(requests) == len(lookup) == 624
    assert len({row["custom_id"] for row in requests}) == 624


def test_gpu_regime_binds_smoke_and_bank() -> None:
    a = GPU._regime("a" * 64, False)
    b = GPU._regime("b" * 64, False)
    c = GPU._regime("a" * 64, True)
    assert GPU._sha_obj(a) != GPU._sha_obj(b)
    assert GPU._sha_obj(a) != GPU._sha_obj(c)


def test_smoke_capture_compatibility_binds_code_not_rollout_identity() -> None:
    base = {
        "model_revision": "m",
        "layers": [14, 19, 26],
        "bank_sha256": "b",
        "rollouts_sha256": "smoke",
        "git_sha": "g",
    }
    full = {**base, "rollouts_sha256": "full"}
    versions = {"torch": "x"}
    assert GPU._smoke_capture_compatibility(base, versions) == (
        GPU._smoke_capture_compatibility(full, versions)
    )
    assert GPU._smoke_capture_compatibility(base, versions) != (
        GPU._smoke_capture_compatibility({**full, "git_sha": "new"}, versions)
    )


def test_expected_rollout_ids_are_ordered_and_draw_complete() -> None:
    rows = [{"item_id": "a"}, {"item_id": "b"}]
    expected = [f"a-d{i}" for i in range(8)] + [f"b-d{i}" for i in range(8)]
    assert GPU._expected_rollout_ids(rows) == expected


def test_production_summary_rejects_unresolved_transport(tmp_path: Path) -> None:
    manifest_path = tmp_path / "wave_request_manifest.json"
    manifest = {
        "request_sha256": "r",
        "rollouts_sha256": "g",
        "model": DV.MODEL,
        "rubric_sha256": DV.RUBRIC_SHA256,
    }
    DV._write_json(manifest_path, manifest)
    rows = []
    for source in range(90):
        for language in BANK.LANGUAGES:
            for content in BANK.CONTENTS:
                for frame in ("direct", "academic"):
                    prompt = f"p-{source}-{language}-{content}-{frame}"
                    for draw in range(8):
                        invalid = (
                            source >= 86
                            and language == "en"
                            and content == "sensitive_full"
                            and frame == "direct"
                            and draw == 0
                        )
                        rows.append(
                            {
                                "item_id": f"{prompt}-d{draw}",
                                "prompt_id": prompt,
                                "source_prompt_id": f"s-{source}",
                                "language": language,
                                "content": content,
                                "frame": frame,
                                "verdict": None if invalid else False,
                                "api_refusal": False,
                                "transport_error": {"code": "fixture"} if invalid else None,
                                "stop_reason": "stop",
                            }
                        )
    score_path = tmp_path / "wave_scores.jsonl"
    DV._write_jsonl(score_path, rows)
    summary = DV._score_summary(rows, manifest, score_path, pilot=False)
    assert summary["passed"] is False
    assert summary["n_transport"] == 4
    assert summary["realized_complete_source_items"] == 86
    assert len(summary["complete_source_item_ids"]) == 86


def test_gpu_explicit_input_snapshot_is_frozen_across_phases(tmp_path: Path) -> None:
    bank, audit = _write_bank_fixture(tmp_path / "inputs")
    BANK._write_json(
        bank.parent / "upload_verified.json",
        {
            "data_revision": "fixture-revision",
            "prompt_bank_sha256": GPU._sha256(bank),
            "bank_audit_report_sha256": GPU._sha256(audit),
        },
    )
    GPU.stage_inputs(tmp_path, bank, audit)
    stage = json.loads((tmp_path / "manifests" / "input_stage.json").read_text())
    assert stage["prompt_bank_sha256"] == GPU._sha256(bank)
    rows = GPU._read_jsonl(bank)
    rows[0]["prompt"] = "same-length-drift"
    GPU._write_jsonl(bank, rows)
    try:
        GPU.stage_inputs(tmp_path, bank, audit)
    except RuntimeError as exc:
        assert "immutable upload marker" in str(exc)
    else:
        raise AssertionError("cross-phase bank drift was accepted")


def test_cpu_timing_pilot_refuses_checkpoint_resume(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot_work_cpu-mid"
    pilot_dir.mkdir(parents=True)
    (pilot_dir / "checkpoint").write_text("stale")
    try:
        ANALYSIS.run_analysis_pilot(tmp_path, tmp_path, tmp_path, tmp_path, "cpu-mid")
    except RuntimeError as exc:
        assert "cannot be checkpoint-resumed" in str(exc)
    else:
        raise AssertionError("checkpoint-resumed timing evidence was accepted")


def test_vectorized_holm_and_headline_lattice() -> None:
    p = np.array([[0.001, 0.002, 0.003, 0.004, 0.20, 0.30]])
    rejected = ANALYSIS._holm_rejections(p)
    assert np.array_equal(rejected, [[True, True, True, True, False, False]])
    assert ANALYSIS._headline_success(np.array([[True, True, True, True, False, False]])).item()
    assert not ANALYSIS._headline_success(np.array([[True, True, True, False, True, True]])).item()


def test_batch_poll_rejects_lookup_identity_drift(tmp_path: Path) -> None:
    judge = tmp_path / "judge"
    requests = judge / "pilot_requests.jsonl"
    DV._write_jsonl(requests, [{"custom_id": "j-0"}])
    manifest_path = judge / "pilot_request_manifest.json"
    manifest = {
        "request_sha256": DV._sha256(requests),
        "lookup_sha256": "new-lookup",
        "ordered_item_ids_sha256": "ordered",
        "model": DV.MODEL,
        "rubric_sha256": DV.RUBRIC_SHA256,
    }
    DV._write_json(manifest_path, manifest)
    DV._write_json(
        judge / "pilot_batch_state.json",
        {
            "request_sha256": manifest["request_sha256"],
            "lookup_sha256": "old-lookup",
            "request_manifest_sha256": DV._sha256(manifest_path),
            "ordered_item_ids_sha256": "ordered",
            "model": DV.MODEL,
            "rubric_sha256": DV.RUBRIC_SHA256,
            "deadline_unix": 10**12,
        },
    )
    try:
        DV.poll_batch(tmp_path, pilot=True)
    except RuntimeError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("lookup-drifted Batch state was accepted")


def test_post_gpu_stage_rejects_mixed_bank_snapshot() -> None:
    kwargs = {
        "done": {"status": "done", "generation": {"rollouts_sha256": "roll"}},
        "generation": {
            "rollouts_sha256": "roll",
            "regime": {"smoke": False, "bank_sha256": "old-bank"},
        },
        "capture": {"capture_regime": {"bank_sha256": "old-bank"}},
        "marker": {
            "prompt_bank_sha256": "new-bank",
            "bank_audit_report_sha256": "audit",
        },
        "prompt_bank_sha256": "new-bank",
        "audit_sha256": "audit",
        "rollouts_sha256": "roll",
    }
    try:
        DV._validate_staged_gpu_provenance(**kwargs)
    except RuntimeError as exc:
        assert "provenance/hash gate" in str(exc)
    else:
        raise AssertionError("mixed bank/generation snapshot was accepted")


def test_analysis_requires_judge_and_generation_rollout_identity() -> None:
    kwargs = {
        "local_bank_sha256": "bank",
        "input_marker": {"prompt_bank_sha256": "bank"},
        "generation": {"rollouts_sha256": "roll", "regime": {"bank_sha256": "bank"}},
        "capture": {
            "rollouts_sha256": "roll",
            "capture_regime": {"bank_sha256": "bank"},
        },
        "judge_summary": {"rollouts_sha256": "older-roll"},
    }
    try:
        ANALYSIS.validate_analysis_attempt_identity(**kwargs)
    except RuntimeError as exc:
        assert "production-attempt identity mismatch" in str(exc)
    else:
        raise AssertionError("mixed generation/judge attempts were accepted")


def test_exact_generated_tokens_are_preserved_when_adding_turn_tail() -> None:
    assert GPU._with_eot_tail([10, 11], [99, 13]) == [10, 11, 99, 13]
    assert GPU._with_eot_tail([10, 99], [99, 13]) == [10, 99, 13]
    assert GPU._with_eot_tail([10, 99, 13], [99, 13]) == [10, 99, 13]


def test_effective_svd_uses_raw_input_coordinate_operator() -> None:
    weight = torch.diag(torch.tensor([8.0, 3.0, 1.0], dtype=torch.float64))
    xsd = torch.tensor([4.0, 1.0, 1.0], dtype=torch.float64)
    u, singular, rank, tau = ANALYSIS.effective_svd(weight, xsd, 0.80)
    assert np.allclose(singular, [3.0, 2.0, 1.0])
    assert rank == 2
    assert tau == 2.0
    projection = u @ u.T
    assert np.allclose(projection, np.diag([1.0, 1.0, 0.0]))


def test_distance_matched_null_is_in_topic_and_finite() -> None:
    topics = np.array(["a", "a", "b", "b"])
    within = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    target = np.array([[1.0, 1.0]] * 4)
    u_ret = np.array([[1.0], [0.0]])
    shares, pairs, errors = ANALYSIS.distance_matched_within_condition_null(
        target, within, topics, u_ret
    )
    assert np.all(np.isfinite(shares))
    assert np.all(np.isfinite(errors))
    assert all(topics[a] == topics[b] == topics[i] for i, (a, b) in enumerate(pairs))


def test_structured_projection_batch_matches_scalar_draws() -> None:
    x = np.arange(35, dtype=np.float64).reshape(5, 7)
    seeds = np.array([11, 18, 25])
    batched = ANALYSIS.structured_projection_coords_batch(x, 4, seeds)
    scalar = np.stack([ANALYSIS.structured_projection_coords(x, 4, int(seed)) for seed in seeds])
    assert np.allclose(batched, scalar)


def test_batched_topic_retrieval_matches_scalar_and_ranks() -> None:
    query = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    gallery = query.copy()
    topics = np.array(["a", "a", "b", "b"])
    batch_q = np.stack([query, query[:, ::-1]])
    batch_g = np.stack([gallery, gallery[:, ::-1]])
    pred, ranks = ANALYSIS.batch_retrieval_predictions_and_ranks(batch_q, batch_g, topics)
    for draw in range(2):
        expected_pred, expected_rank = ANALYSIS.retrieval_predictions_and_ranks(
            batch_q[draw], batch_g[draw], topics
        )
        assert np.array_equal(pred[draw], expected_pred)
        assert np.array_equal(ranks[draw], expected_rank)
    assert np.array_equal(ranks, np.ones_like(ranks))


def test_topic_bootstrap_weights_resample_whole_topic_sizes() -> None:
    topics = np.array(["small", "small", "large", "large", "large"])
    weights = ANALYSIS.topic_bootstrap_weights(topics, 100, 123)
    assert weights.shape == (100, 5)
    assert np.all(weights >= 0)
    totals = weights.sum(axis=1)
    assert set(totals) <= {4, 5, 6}
    assert np.all(np.sum(weights[:, :2], axis=1) % 2 == 0)
    assert np.all(np.sum(weights[:, 2:], axis=1) % 3 == 0)


def test_weighted_bootstrap_median_matches_expanded_even_sample() -> None:
    values = np.array([0.0, 10.0])
    weights = np.array([[1, 1], [2, 0], [0, 2]])
    assert np.allclose(ANALYSIS.bootstrap_weighted_median(values, weights), [5.0, 0.0, 10.0])


def test_topic_bootstrap_spearman_recomputes_clustered_interval() -> None:
    from scipy.stats import spearmanr

    x = np.array([0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    y = np.array([0.0, 1.0, 1.0, 3.0, 2.0, 4.0])
    topics = np.array(["a", "a", "b", "b", "c", "c"])
    weights = ANALYSIS.topic_bootstrap_weights(topics, 20, 22)
    rho = ANALYSIS.topic_bootstrap_spearman(x, y, weights)
    expected = []
    for row in weights:
        indices = np.repeat(np.arange(len(x)), row)
        expected.append(spearmanr(x[indices], y[indices]).statistic)
    assert np.allclose(rho, expected, equal_nan=True)


def test_holm_adjust_and_loto_gain_are_fixed() -> None:
    assert np.allclose(ANALYSIS.holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])
    predicted = np.array([[1.0], [2.0], [3.0], [4.0]])
    observed = 2 * predicted
    topics = np.array(["a", "a", "b", "b"])
    calibrated, gains = ANALYSIS.leave_one_topic_out_gain_prediction(predicted, observed, topics)
    assert np.allclose(calibrated, observed)
    assert gains == {"a": 2.0, "b": 2.0}


def test_one_sided_bootstrap_p_uses_null_side_tail() -> None:
    samples = np.array([-0.2, 0.1, 0.2, 0.3])
    assert ANALYSIS.bootstrap_p_greater(samples, observed=0.2, null=0.0) == 0.4


def test_production_analysis_widths_cannot_be_reduced(tmp_path: Path) -> None:
    try:
        ANALYSIS.run_analysis(
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            n_random=999,
            n_resample=10_000,
            production=True,
        )
    except RuntimeError as exc:
        assert "exactly 1000 random/10000 resamples" in str(exc)
    else:
        raise AssertionError("reduced production width was accepted")
