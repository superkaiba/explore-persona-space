"""Synthetic contracts and independent numerical oracles for repaired v2 analysis."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
from scipy.stats import spearmanr

from scripts import issue952_china_repair_analysis as analysis
from scripts import issue952_china_repair_geometry as geom


@pytest.fixture
def small_inputs():
    """A complete 6-source factorial panel with eight aligned draws and three layers."""
    rng = np.random.default_rng(845)
    bank, rollouts, scores = [], [], []
    for subject in range(6):
        for language in geom.LANGUAGE_ORDER:
            for content in geom.CONTENT_ORDER:
                for frame in geom.FRAME_ORDER:
                    pid = f"s{subject}-{language}-{content}-{frame}"
                    prompt = {
                        "item_id": pid,
                        "source_prompt_id": f"s{subject}",
                        "language": language,
                        "content": content,
                        "frame": frame,
                        "topic": f"t{subject // 3}",
                        "prompt": f"synthetic question {pid}",
                    }
                    bank.append(prompt)
                    for draw in range(8):
                        metadata = {
                            key: prompt[key]
                            for key in ("source_prompt_id", "language", "content", "frame", "topic")
                        }
                        common = {
                            **metadata,
                            "item_id": f"{pid}-d{draw}",
                            "prompt_id": pid,
                            "draw": draw,
                        }
                        rollouts.append(
                            {**common, "question": prompt["prompt"], "text": "synthetic answer"}
                        )
                        scores.append(
                            {
                                **{
                                    key: value
                                    for key, value in common.items()
                                    if key != "prompt_id"
                                },
                                "withholding_score": (subject * 13 + draw * 3) % 101,
                                "strict_complete_refusal": draw == 0,
                                "unassessable": False,
                            }
                        )
    vc = rng.normal(size=(len(bank), 3, 4)).astype(np.float32)
    va = rng.normal(size=(len(rollouts), 3, 4)).astype(np.float32)
    return bank, rollouts, scores, vc, va


def test_finite_json_recurses_into_numpy_arrays(tmp_path):
    path = tmp_path / "result.json"
    analysis.write_json(path, {"nested": np.array([np.nan, np.inf, 1]), "flag": np.bool_(True)})
    assert json.loads(path.read_text()) == {"nested": [None, None, 1.0], "flag": True}
    assert "NaN" not in path.read_text()
    path.write_text('{"bad": NaN}')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        analysis.read_json(path)


def test_explicit_immutable_production_pins():
    prefix = f"{analysis.HF_PREFIX}/attempt1/judge_archive"
    identity = analysis.input_identity("a" * 40, "b" * 40, prefix)
    assert identity["map_revision"] == "eef8eb1da43cd2212dfa73d8711fd64dc54c376f"
    assert identity["model_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    with pytest.raises(ValueError, match="immutable"):
        analysis.input_identity("main", "b" * 40, prefix)
    with pytest.raises(ValueError, match="production judge"):
        analysis.input_identity("a" * 40, "b" * 40, prefix.replace("attempt1", "attempt1/smoke"))


def test_missing_draws_keep_valid_denominators_and_missing_groups():
    values = np.array([[20, np.nan, 80], [np.nan, np.nan, np.nan]])
    mean, count = analysis.valid_mean(values, axis=1)
    np.testing.assert_array_equal(count, [2, 0])
    assert mean[0] == 50 and np.isnan(mean[1])
    weights = np.array([[1, 1], [0, 3], [2, 0]])
    np.testing.assert_allclose(
        analysis.weighted_draws(mean, weights), [50, np.nan, 50], equal_nan=True
    )


def test_panel_join_keeps_geometry_when_behavior_missing(small_inputs):
    bank, rows, scores, vc, va = small_inputs
    # All eight scores missing in one entire source/language/content/frame arm.
    for row in scores[:8]:
        row.update(unassessable=True, withholding_score=None, strict_complete_refusal=None)
    panel = analysis.assemble_panel(bank, rows, list(reversed(scores)), vc, va)
    assert panel["contexts"].shape == (6, 2, 4, 2, 3, 4)
    assert panel["answers"].shape == (6, 2, 4, 2, 8, 3, 4)
    assert np.isfinite(panel["contexts"]).all()
    assert np.isnan(panel["withholding_score"][0, 0, 0, 0]).all()
    np.testing.assert_array_equal(panel["contexts"][0, 0, 0, 0], vc[0])
    np.testing.assert_array_equal(panel["answers"][0, 0, 0, 0, 7], va[7])


def test_real_collector_schema_joins_without_prompt_id(small_inputs, tmp_path, monkeypatch):
    """Execute the real repaired producer/consumer seam on one complete 16x8 source."""
    bank, rollouts, _, vc, va = small_inputs
    bank, rollouts, vc, va = bank[:16], rollouts[:128], vc[:16], va[:128]
    judge = analysis.judges
    source_hash = judge.sha_object(["s0"])
    # Only the synthetic panel cardinality/identity changes; production functions,
    # all four content cells, both frames/languages, and eight draws execute intact.
    monkeypatch.setattr(judge, "SOURCE_COUNT", 1)
    monkeypatch.setattr(judge, "FROZEN_SOURCE_IDS_SHA256", source_hash)
    raw_path = tmp_path / "rollouts.jsonl"
    raw_path.write_bytes(judge._jsonl_bytes(rollouts))
    generation_path = tmp_path / "generation.json"
    analysis.write_json(
        generation_path,
        {
            "rollouts_sha256": judge.sha_file(raw_path),
            "n_rows": len(rollouts),
            "regime": {"bank_sha256": "a" * 64, "accepted_source_ids_sha256": source_hash},
        },
    )
    directory = tmp_path / "judge"
    manifest = judge.prepare(
        raw_path, generation_path, directory, ("synthetic_a", "synthetic_b"), production=True
    )
    for record in manifest["packets"]:
        packet = judge.read_json(Path(record["packet_path"]))
        receipt = {
            "contract": judge.CONTRACT,
            "packet_sha256": record["packet_sha256"],
            "rubric_sha256": judge.RUBRIC_SHA256,
            "runtime_identity_sha256": packet["runtime_identity_sha256"],
            "read_complete": True,
            "items": [
                {
                    **{
                        key: item[key]
                        for key in judge.RECEIPT_ITEM_FIELDS
                        if key != "read_complete"
                    },
                    "read_complete": True,
                }
                for item in packet["items"]
            ],
        }
        receipt_path = Path(record["receipt_path"])
        analysis.write_json(receipt_path, receipt)
        decisions = [
            {
                "opaque_id": item["opaque_id"],
                "request_sha256": item["request_sha256"],
                "packet_sha256": record["packet_sha256"],
                "rubric_sha256": judge.RUBRIC_SHA256,
                "runtime_identity_sha256": packet["runtime_identity_sha256"],
                "input_read_receipt_sha256": judge.sha_file(receipt_path),
                "rationale": (
                    f"Synthetic schema fixture {index}: this supplies the requested answer."
                ),
                "category": "substantive_answer",
                "withholding_score": 0,
                "answer_coverage_score": 100,
                "strict_complete_refusal": False,
                "explicit_decline": False,
                "evasive_redirection": False,
                "expressed_uncertainty": False,
                "unassessable": False,
                "unassessable_reason": None,
            }
            for index, item in enumerate(packet["items"])
        ]
        Path(record["output_path"]).write_bytes(judge._jsonl_bytes(decisions))
    judge.collect(directory)
    scores, _, _ = analysis.validate_judgments(directory, generation_path, rollouts)
    assert len(scores) == 128
    assert all("prompt_id" not in score for score in scores)
    panel = analysis.assemble_panel(bank, rollouts, list(reversed(scores)), vc, va)
    assert panel["answers"].shape == (1, 2, 4, 2, 8, 3, 4)
    np.testing.assert_array_equal(panel["withholding_score"], 0)
    np.testing.assert_array_equal(panel["answers"][0, 0, 0, 0, 7], va[7])


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_prompt", "missing_rollout", "wrong_topic", "wrong_draw", "coerced_unassessable"],
)
def test_panel_rejects_identity_and_measurement_drift(small_inputs, mutation):
    bank, rows, scores, vc, va = small_inputs
    if mutation == "duplicate_prompt":
        bank[-1] = dict(bank[0])
    elif mutation == "missing_rollout":
        rows[-1] = dict(rows[0])
    elif mutation == "wrong_topic":
        scores[0]["topic"] = "other-topic"
    elif mutation == "wrong_draw":
        scores[0]["draw"] = 7
    else:
        scores[0]["unassessable"] = True
    with pytest.raises(ValueError):
        analysis.assemble_panel(bank, rows, scores, vc, va)


@pytest.mark.parametrize("mutation", [None, "seed", "bool_token", "cap_flag"])
def test_final_rollout_ids_tokens_seeds_and_termination(small_inputs, mutation):
    bank, rows, _, _, _ = small_inputs
    for index, row in enumerate(rows):
        row.update(
            seed=9_520_000 + index,
            context_tokens=10,
            completion_token_ids=[20, 30],
            completion_tokens=2,
            finish_reason="stop",
            cap_hit=False,
        )
    expected = analysis.gpu._expected_rollout_ids(bank)
    generation = {
        "regime": {"seed_base": 9_520_000},
        "ordered_item_ids_sha256": analysis.sha_object(expected),
        "n_prompts": len(bank),
        "n_rows": len(rows),
    }
    if mutation == "seed":
        rows[0]["seed"] += 1
    elif mutation == "bool_token":
        rows[0]["completion_token_ids"][0] = True
    elif mutation == "cap_flag":
        rows[0]["cap_hit"] = True
    if mutation is None:
        assert analysis.validate_final_rollouts(bank, rows, generation) == expected
    else:
        with pytest.raises(ValueError, match="invalid seed, exact-token, or termination"):
            analysis.validate_final_rollouts(bank, rows, generation)


def test_paired_topic_sign_flip_matches_serial_oracle():
    topics = np.array(["a", "a", "b", "b"])
    signs = analysis.topic_signs(topics, 40, 71)
    np.testing.assert_array_equal(signs[:, 0], signs[:, 1])
    np.testing.assert_array_equal(signs[:, 2], signs[:, 3])
    first, second = np.array([1.0, 2, np.nan, 4]), np.array([0.0, 0, 9, 1])
    weights = analysis.legacy.topic_bootstrap_weights(topics, 50, 8)
    report = analysis.paired_difference(first, second, weights, signs, "H2.test")
    values = np.array([1.0, 2, 3])
    serial = np.array([np.mean(sign[[0, 1, 3]] * values) for sign in signs])
    assert report["mean"] == 2
    assert report["n_defined"] == 3
    assert report["raw_p"] == pytest.approx((1 + np.sum(serial >= 2)) / 41)


def test_correlation_preserves_undefined_and_matches_spearman():
    topics = np.array(["a", "a", "a", "b", "b", "b"])
    weights = analysis.legacy.topic_bootstrap_weights(topics, 100, 3)
    a, b = np.array([2.0, 1, np.nan, 4, 2, 9]), np.array([3.0, 8, 2, 5, 3, 2])
    report = analysis.correlation(a, b, topics, weights, 100, 4)
    valid = np.isfinite(a)
    assert report["rho"] == pytest.approx(spearmanr(a[valid], b[valid]).statistic)
    assert report["n_defined"] == 5
    saturated = analysis.correlation(a, np.zeros(6), topics, weights, 100, 4)
    assert saturated["rho"] is None and saturated["raw_p"] is None


def test_h1_centers_by_frozen_training_mean():
    rng = np.random.default_rng(628)
    states = rng.normal(size=(6, 2, 4))
    xmu = np.array([4.0, -2, 1, 3])
    basis = np.eye(4)[:, :2]
    topics = np.array(["a"] * 3 + ["b"] * 3)
    weights = analysis.legacy.topic_bootstrap_weights(topics, 25, 3)
    report, arrays = analysis.retrieval_panel(
        states, xmu, basis, topics, weights, 5, 25, 5, "H1.test"
    )
    expected = analysis.retrieval(
        (states[:, 0] - xmu) @ basis, (states[:, 1] - xmu) @ basis, topics
    )[2]
    np.testing.assert_array_equal(arrays["en_to_zh_within_topic_retained_hit"], expected)
    assert report["en_to_zh"]["centering"] == "frozen_map_xmu"
    assert report["en_to_zh"]["galleries"]["within_topic"][
        "mean_fixed_gallery_chance"
    ] == pytest.approx(1 / 3)
    assert report["en_to_zh"]["galleries"]["all_items"][
        "mean_fixed_gallery_chance"
    ] == pytest.approx(1 / 6)


def test_zero_direction_retrieval_is_undefined():
    pred, ranks, hit = analysis.retrieval(np.zeros((4, 3)), np.ones((4, 3)), np.zeros(4))
    assert (pred == -1).all() and np.isnan(ranks).all() and np.isnan(hit).all()


def test_zero_random_projector_directions_do_not_become_negative_labels():
    states, topics = np.zeros((6, 2, 4)), np.array(["a"] * 3 + ["b"] * 3)
    weights = analysis.legacy.topic_bootstrap_weights(topics, 10, 2)
    result, arrays = analysis.retrieval_panel(
        states, np.zeros(4), np.eye(4)[:, :2], topics, weights, 5, 10, 7, "H1.test"
    )
    assert (arrays["en_to_zh_within_topic_random_pred"] == -1).all()
    random = result["en_to_zh"]["galleries"]["within_topic"]["modes"]["rank_matched_random"]
    assert random["accuracy"]["mean"] is None
    assert result["en_to_zh"]["comparisons"]["retained_minus_random"]["raw_p"] is None


def test_mixed_random_projectors_use_symmetric_defined_denominators(monkeypatch):
    """Scalar exhaustive null oracle for one defined and one undefined projector."""
    n = 4
    neutral = np.random.default_rng(4).normal(size=(n, 2, 3))
    neutral[..., 2] = 0
    topics = np.zeros(n, dtype=int)
    targets = np.array(list(permutations(range(n))))

    def selected_projectors(x, rank, seeds):
        assert rank == 1 and len(seeds) == 2
        return np.stack((x[:, 1:2], x[:, 2:3]))

    def exhaustive_targets(topics, n_perm, seed):
        assert len(topics) == n and n_perm == len(targets)
        return targets

    monkeypatch.setattr(
        analysis.legacy,
        "structured_projection_coords_batch",
        create_autospec(
            analysis.legacy.structured_projection_coords_batch, side_effect=selected_projectors
        ),
    )
    monkeypatch.setattr(
        analysis.legacy,
        "_within_topic_targets",
        create_autospec(analysis.legacy._within_topic_targets, side_effect=exhaustive_targets),
    )
    report, arrays = analysis.retrieval_panel(
        neutral,
        np.zeros(3),
        np.eye(3)[:, :1],
        topics,
        np.ones((5, n), dtype=int),
        2,
        len(targets),
        952,
        "H1.mixed_projector_test",
    )
    for direction, query, gallery in (("en_to_zh", 0, 1), ("zh_to_en", 1, 0)):
        retained, _, retained_hit = analysis.retrieval(
            neutral[:, query, :1], neutral[:, gallery, :1], topics
        )
        predictions = arrays[f"{direction}_within_topic_random_pred"]
        np.testing.assert_array_equal(predictions[1], -1)
        observed = np.mean(retained_hit - (predictions[0] == np.arange(n)))
        null = np.array(
            [
                np.mean(
                    (retained == target).astype(float) - (predictions[0] == target).astype(float)
                )
                for target in targets
            ]
        )
        expected = (1 + np.count_nonzero(null >= observed)) / (1 + len(targets))
        assert report[direction]["comparisons"]["retained_minus_random"]["raw_p"] == pytest.approx(
            expected
        )
        if direction == "en_to_zh":
            assert expected == pytest.approx(0.92)
        for gallery_name in ("within_topic", "all_items"):
            random = report[direction]["galleries"][gallery_name]["modes"]["rank_matched_random"]
            np.testing.assert_array_equal(random["n_defined_projectors_by_item"], 1)


def test_h3_uncalibrated_r2_and_cosine_match_independent_formula():
    observed = np.array([[1.0, 2], [2, -1], [-2, 0], [3, 1]])
    prediction = 2 * observed
    topics = np.array(["a", "a", "b", "b"])
    weights = analysis.legacy.topic_bootstrap_weights(topics, 40, 42)
    targets = analysis.legacy._within_topic_targets(topics, 40, 7)
    result = analysis.prediction_quality(prediction, observed, topics, weights, targets, "H3.test")
    assert result["cosine"]["mean"] == pytest.approx(1)
    assert result["r2"]["r2"] == pytest.approx(0)
    assert result["r2"]["baseline"] == "zero_answer_change"
    assert result["calibration"].startswith("none")


def test_complete_layer_runs_all_contrasts_and_right_write_basis(small_inputs, tmp_path):
    panel = analysis.assemble_panel(*small_inputs)
    operator = np.array([[0.0, 30, 0, 0], [1, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.01]])
    modes = geom.operator_modes(operator, np.ones(4))
    result, arrays = analysis.analyze_layer(
        panel, {"xmu": np.array([1.0, 2, 3, 4])}, modes, np.array([0.0, 1, 0, 0]), 0, 0.99, 5, 40
    )
    assert result["n_geometry_subjects"] == 6
    assert result["rank_retained"] == 1
    assert set(result["H2"]) == {"en", "zh"}
    assert set(result["H2"]["en"]["paired_comparisons"]) == {
        "framing_minus_subject",
        "china_cue_minus_subject",
        "control_cue_minus_subject",
    }
    assert len(result["H2"]["en"]["contrasts"]) == 17
    assert set(result["H3"]["zh"]["modes"]) == {"full", "retained", "kernel", "identity"}
    assert "not validated" in result["H4"]["en"]["axis_label"]
    observed = arrays["en_subject_observed"]
    expected = geom.orthogonal_parts(observed, modes["write"][:, :1])["low_share"]
    subject_index = list(arrays["contrast_names"]).index("subject")
    np.testing.assert_allclose(arrays["answer_low_write_share"][subject_index, :, 0], expected)
    analysis.write_json(tmp_path / "result.json", result)
    assert "NaN" not in (tmp_path / "result.json").read_text()


def test_behavior_missingness_and_strict_refusal_are_separate(small_inputs):
    bank, rows, scores, vc, va = small_inputs
    for score in scores[:8]:
        score.update(unassessable=True, withholding_score=None, strict_complete_refusal=None)
    panel = analysis.assemble_panel(bank, rows, scores, vc, va)
    report = analysis.behavior_report(panel, {"overall": {"rho": None}}, 30)
    graded = report["measures"]["withholding_score"]["arms"][0]
    strict = report["measures"]["strict_complete_refusal"]["arms"][0]
    assert graded["n_planned_draws"] == 48
    assert graded["n_valid_draws"] == 40
    assert graded["n_subjects_with_no_valid_draw"] == 1
    assert strict["valid_draw_weighted_mean"] == 1 / 8
    assert graded["valid_draw_weighted_mean"] != strict["valid_draw_weighted_mean"]


def test_holm_keeps_missing_tests_in_declared_family():
    report = {
        "tests": [
            {"family": "primary", "raw_p": 0.02},
            {"family": "primary", "raw_p": None},
            {"family": "secondary", "raw_p": 0.03},
        ]
    }
    census = analysis.adjust_families(report)
    assert report["tests"][0]["holm_p"] == 0.04
    assert report["tests"][1]["holm_p"] is None
    assert report["tests"][2]["holm_p"] == 0.03
    assert census["primary"]["n_planned_tests"] == 2


def test_stage_file_runs_real_body_with_signature_checked_hub_boundary(tmp_path, monkeypatch):
    """Validate the actual persistence call shape and strict post-download bytes."""

    def external(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"verified synthetic bytes")
        return target

    fake = create_autospec(analysis.hub.stage_hub_file, side_effect=external)
    monkeypatch.setattr(analysis.hub, "stage_hub_file", fake)
    result = analysis._stage_file(tmp_path, "a" * 40, "prefix/source.json", "stage/file.json")
    assert result["sha256"] == analysis.sha_file(tmp_path / "stage/file.json")
    assert fake.call_args.kwargs["revision"] == "a" * 40
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis._stage_file(tmp_path, "a" * 40, "prefix/source.json", "stage/file.json", "0" * 64)
    with pytest.raises(ValueError, match="unsafe"):
        analysis._stage_file(tmp_path, "a" * 40, "prefix/source.json", "../outside")


def test_manifest_first_stage_resumes_exact_bytes_and_rejects_corruption(tmp_path, monkeypatch):
    """Execute staging and archive restore with only the Hub transport boundary replaced."""
    identity = analysis.input_identity(
        "a" * 40, "b" * 40, f"{analysis.HF_PREFIX}/attempt1/judge_archive"
    )
    collector = tmp_path / "collector"
    collector.mkdir()
    (collector / "summary.json").write_text('{"synthetic": true}\n')
    archive_dir = tmp_path / "archive"
    analysis.persist.pack_tree(collector, archive_dir)
    remote_files = {}

    def payload(revision, remote, value):
        remote_files[(revision, remote)] = (
            value if isinstance(value, bytes) else json.dumps(value).encode()
        )

    data_prefix = identity["data_prefix"]
    raw_path = f"{data_prefix}/raw_completions/rollouts.jsonl"
    raw_payload, tensor_payload = b"synthetic raw text\n", b"synthetic tensor bytes"
    import hashlib

    raw_sha = hashlib.sha256(raw_payload).hexdigest()
    tensor_sha = hashlib.sha256(tensor_payload).hexdigest()
    manifests = {
        "generation.json": {"rollouts_sha256": raw_sha},
        "capture.json": {"vc_sha256": tensor_sha, "va_files": {"va_00000_00001.pt": tensor_sha}},
        "raw_upload.json": {"byte_verified_sha256": {raw_path: raw_sha}},
        "capture_upload.json": {},
        "input_stage.json": {},
        "cap_extension.json": {},
        "generation.initial.json": {},
    }
    for name, value in manifests.items():
        payload("a" * 40, f"{data_prefix}/manifests/{name}", value)
    payload("a" * 40, f"{data_prefix}/issue952_china_definitive_done.json", {})
    payload("a" * 40, raw_path, raw_payload)
    for name in ("vc.pt", "va_00000_00001.pt"):
        payload("a" * 40, f"{data_prefix}/analysis_tensors/{name}", tensor_payload)
    for name in ("prompt_bank.jsonl", "bank_audit_report.json"):
        payload(analysis.INPUT_REV, f"{analysis.HF_PREFIX}/inputs/{name}", {})
    for path in archive_dir.iterdir():
        payload("b" * 40, f"{identity['judge_prefix']}/{path.name}", path.read_bytes())
    for layer in analysis.LAYERS:
        payload(
            analysis.legacy.MAP_REV,
            f"{analysis.legacy.MAP_PREFIX}/L{layer}/ridge.pt",
            tensor_payload,
        )
    for remote in analysis.legacy.HIST_FILES:
        payload(analysis.legacy.HIST_REV, remote, tensor_payload)

    def transport(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(remote_files[(revision, path_in_repo)])
        return target

    fake = create_autospec(analysis.hub.stage_hub_file, side_effect=transport)
    monkeypatch.setattr(analysis.hub, "stage_hub_file", fake)
    root = tmp_path / "staged"
    result = analysis.stage(root, identity)
    assert (root / "judge/summary.json").read_bytes() == (collector / "summary.json").read_bytes()
    assert len(result["files"]) > 15
    n_calls = fake.call_count
    assert analysis.stage(root, identity) == result
    assert fake.call_count == n_calls
    (root / "run/analysis_tensors/vc.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.stage(root, identity)


def test_cli_requires_pins_and_has_all_phases():
    parser = analysis.build_argparser()
    for phase in ("stage", "load", "pilot", "full", "export"):
        args = parser.parse_args(
            [
                phase,
                "--root",
                "/tmp/example",
                "--data-revision",
                "a" * 40,
                "--judge-revision",
                "b" * 40,
                "--judge-prefix",
                f"{analysis.HF_PREFIX}/attempt1/judge",
            ]
        )
        assert args.phase == phase and args.threads == 8
    with pytest.raises(SystemExit):
        parser.parse_args(["full", "--root", "/tmp/example"])
