"""#2658 P0 guard + manifest tests: one raising test per plan-§9 fail-on condition.

Plan §9: "Tests fail on test-derived transforms, peer centering, dependency
crossing, row/hash mismatch, missing-label omission, pooled-fold confirmatory
metrics, stale caches, non-iid generation, mixed judge revisions, or
preliminary-label gate use."  Every guard lives in
``scripts/issue2658_common.py``; each test below asserts the guard RAISES on
its condition (and passes on the compliant shape).  Purely synthetic — no
network, no torch, no other issues' committed artifacts.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2658_common as C  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Test-derived transforms
# ---------------------------------------------------------------------------


def test_test_derived_transform_raises():
    with pytest.raises(C.TestDerivedTransformError):
        C.assert_transform_fit_split("test")
    with pytest.raises(C.TestDerivedTransformError):
        C.assert_transform_fit_split("pilot")
    C.assert_transform_fit_split("dev")  # compliant


# ---------------------------------------------------------------------------
# 2. Peer centering
# ---------------------------------------------------------------------------


def test_peer_centering_raises():
    with pytest.raises(C.PeerCenteringError):
        C.assert_preprocessing_scope("peer-batch")
    with pytest.raises(C.PeerCenteringError):
        C.assert_preprocessing_scope("test")
    with pytest.raises(C.PeerCenteringError):
        C.assert_preprocessing_scope("pooled")
    C.assert_preprocessing_scope("dev-fold-train")  # compliant


# ---------------------------------------------------------------------------
# 3. Dependency crossing (content-superfamily leakage across dev/test)
# ---------------------------------------------------------------------------


def test_dependency_crossing_raises():
    with pytest.raises(C.DependencyCrossingError, match="sf_b"):
        C.assert_split_lineage_disjoint({"sf_a", "sf_b"}, {"sf_b", "sf_c"})
    C.assert_split_lineage_disjoint({"sf_a"}, {"sf_c"})  # compliant


# ---------------------------------------------------------------------------
# 4. Row/hash mismatch
# ---------------------------------------------------------------------------


def test_row_hash_mismatch_raises():
    good = hashlib.sha256(b"payload").hexdigest()
    wrong = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(C.RowHashMismatchError):
        C.assert_row_hash(b"payload", wrong)
    C.assert_row_hash(b"payload", good)  # compliant
    C.assert_row_hash("payload", good)  # str payload, same bytes


# ---------------------------------------------------------------------------
# 5. Missing-label omission (and None labels)
# ---------------------------------------------------------------------------


def test_missing_label_omission_raises():
    with pytest.raises(C.MissingLabelError):
        C.assert_labels_complete(["r1", "r2"], {"r1": 87.0})
    with pytest.raises(C.MissingLabelError):
        C.assert_labels_complete(["r1"], {"r1": None})
    C.assert_labels_complete(["r1", "r2"], {"r1": 87.0, "r2": 3.0})  # compliant


# ---------------------------------------------------------------------------
# 5b. Label coercion (drop-never-coerce at the aggregation gate)
# ---------------------------------------------------------------------------


def test_coerced_label_raises():
    assert C.aggregate_judge_draws([50, 60, 40, 55, 45]) == 50.0  # compliant median
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, None, 55, 45])  # dropped draw, not coerced
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, 120, 55, 45])  # out-of-range
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, "REFUSAL", 55, 45])  # non-numeric
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, float("nan"), 55, 45])  # NaN
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, 55, 45])  # wrong draw count
    with pytest.raises(C.CoercedLabelError):
        C.aggregate_judge_draws([50, 60, True, 55, 45])  # bool is not a score


# ---------------------------------------------------------------------------
# 6. Pooled cross-fold confirmatory metrics
# ---------------------------------------------------------------------------


def test_pooled_fold_metric_raises():
    with pytest.raises(C.PooledFoldMetricError):
        C.assert_not_pooled_fold("pooled-cross-fold")
    C.assert_not_pooled_fold("frozen-test")  # compliant


# ---------------------------------------------------------------------------
# 7. Stale caches (and malformed cache keys)
# ---------------------------------------------------------------------------


def _full_key_parts() -> dict:
    return {
        "inputs_sha256": "a" * 64,
        "direction_sha256": "b" * 64,
        "split": "dev",
        "judge_fingerprint": "c" * 64,
        "estimator": "lr-lbfgs",
        "grid": "10^[-6..4]",
        "preprocessing": "dev-fold-train-zscore",
        "code_sha": "d" * 40,
        "container": "uv-py311",
        "seeds": "sha-schedule-v1",
    }


def test_stale_cache_raises():
    k1 = C.cache_key(**_full_key_parts())
    parts2 = _full_key_parts()
    parts2["direction_sha256"] = "e" * 64
    k2 = C.cache_key(**parts2)
    assert k1 != k2
    with pytest.raises(C.CacheStaleError):
        C.check_cache_entry(k1, k2)
    C.check_cache_entry(k1, k1)  # compliant

    missing = _full_key_parts()
    missing.pop("seeds")
    with pytest.raises(C.CacheKeyError, match="seeds"):
        C.cache_key(**missing)
    unknown = _full_key_parts()
    unknown["extra"] = 1
    with pytest.raises(C.CacheKeyError, match="extra"):
        C.cache_key(**unknown)


# ---------------------------------------------------------------------------
# 8. Non-iid generation (adaptive flags, count drift, off-schedule seeds)
# ---------------------------------------------------------------------------


def _iid_meta(n: int = 3) -> dict:
    pid = "prompt-001"
    return {
        "prompt_id": pid,
        "seeds": [C.response_seed(pid, k) for k in range(n)],
        "n_planned": n,
        "topped_up": False,
        "early_stopped": False,
        "excluded": False,
    }


def test_non_iid_generation_raises():
    C.assert_iid_generation(_iid_meta())  # compliant
    for flag in ("topped_up", "early_stopped", "excluded"):
        meta = _iid_meta()
        meta[flag] = True
        with pytest.raises(C.NonIIDGenerationError, match=flag):
            C.assert_iid_generation(meta)
    short = _iid_meta()
    short["seeds"] = short["seeds"][:-1]
    with pytest.raises(C.NonIIDGenerationError):
        C.assert_iid_generation(short)
    off = _iid_meta()
    off["seeds"][1] += 1
    with pytest.raises(C.NonIIDGenerationError, match="SHA schedule"):
        C.assert_iid_generation(off)


# ---------------------------------------------------------------------------
# 9. Mixed judge revisions (provider-drift canary)
# ---------------------------------------------------------------------------


def test_mixed_judge_revision_raises():
    pin = C.JUDGE["model"]
    C.assert_single_judge_revision([pin, pin, pin])  # compliant
    with pytest.raises(C.MixedJudgeRevisionError):
        C.assert_single_judge_revision([pin, "claude-sonnet-4-5-20991231"])
    with pytest.raises(C.MixedJudgeRevisionError):
        C.assert_single_judge_revision(["claude-haiku-4-5-20251001"])  # wrong single pin


# ---------------------------------------------------------------------------
# 10. Preliminary-label gate use
# ---------------------------------------------------------------------------


def test_preliminary_label_gate_raises():
    with pytest.raises(C.PreliminaryLabelGateError):
        C.assert_final_labels_for_gate("preliminary")
    with pytest.raises(C.PreliminaryLabelGateError):
        C.assert_final_labels_for_gate("pilot")
    C.assert_final_labels_for_gate("final")  # compliant


# ---------------------------------------------------------------------------
# Manifest validator (strict: unknown/missing fields, pins, seed schedule)
# ---------------------------------------------------------------------------


def _manifest_row() -> dict:
    pid = "prompt-001"
    answer_sha = hashlib.sha256(b"answer text").hexdigest()
    return {
        "manifest_version": C.MANIFEST_VERSION,
        "row": "evil",
        "split": "pilot",
        "prompt_id": pid,
        "prompt_sha256": hashlib.sha256(b"prompt text").hexdigest(),
        "superfamily_id": "sf-042",
        "source_frame": "frame_a",
        "stratum": "s1",
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REVISION,
        "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
        "response_index": 2,
        "seed": C.response_seed(pid, 2),
        "answer_sha256": answer_sha,
        "raw_text_sha256": hashlib.sha256(b"rendered prompt+answer").hexdigest(),
        "evidence_sha256": None,
        "judge_status": "scored",
        "judge_draw_ids": list(C.judge_draw_ids(answer_sha)),
        "judge_model": C.JUDGE["model"],
        "vector_sha256": None,
    }


def test_manifest_row_valid():
    check = C.validate_manifest_row(_manifest_row())
    assert check.row == "evil" and check.response_index == 2


def test_manifest_missing_field_raises():
    row = _manifest_row()
    row.pop("superfamily_id")
    with pytest.raises(C.ManifestError, match="missing"):
        C.validate_manifest_row(row)


def test_manifest_unknown_field_raises():
    row = _manifest_row()
    row["surprise"] = 1
    with pytest.raises(C.ManifestError, match="unknown"):
        C.validate_manifest_row(row)


def test_manifest_wrong_version_raises():
    row = _manifest_row()
    row["manifest_version"] = 99
    with pytest.raises(C.ManifestError, match="manifest_version"):
        C.validate_manifest_row(row)


def test_manifest_off_schedule_seed_raises():
    row = _manifest_row()
    row["seed"] += 1
    with pytest.raises(C.ManifestError, match="SHA schedule"):
        C.validate_manifest_row(row)


def test_manifest_model_pin_mismatch_raises():
    row = _manifest_row()
    row["model_revision"] = "0" * 40
    with pytest.raises(C.ManifestError, match="model pin"):
        C.validate_manifest_row(row)


def test_manifest_wrong_judge_model_raises():
    row = _manifest_row()
    row["judge_model"] = "claude-haiku-4-5-20251001"
    with pytest.raises(C.ManifestError, match="judge_model"):
        C.validate_manifest_row(row)


def test_manifest_objective_row_never_scored():
    row = _manifest_row()
    row["row"] = "correctness_math"
    row["judge_status"] = "scored"
    with pytest.raises(C.ManifestError, match="objective"):
        C.validate_manifest_row(row)
    row["judge_status"] = "objective"
    row["judge_model"] = None
    C.validate_manifest_row(row)  # compliant objective row


def test_manifest_off_schedule_draw_ids_raise():
    row = _manifest_row()
    row["judge_draw_ids"] = ["deadbeefdeadbeef"] * C.JUDGE["n_draws"]
    with pytest.raises(C.ManifestError, match="draw"):
        C.validate_manifest_row(row)


# ---------------------------------------------------------------------------
# Registry arithmetic + Holm family sizes + instrument fingerprints
# ---------------------------------------------------------------------------


def test_pilot_registry_arithmetic():
    assert len(C.ROW_IDS) == 11
    assert C.PILOT.cells_per_row == 12
    assert C.PILOT.responses_per_row == 600
    assert C.PILOT.responses_total == 6600


def test_holm_family_sizes():
    assert C.holm_family_sizes(1) == {"C2": 10, "C5": 11, "C5_minus_C2": 10}
    assert C.holm_family_sizes(0) == {"C2": 11, "C5": 11, "C5_minus_C2": 11}
    with pytest.raises(ValueError):
        C.holm_family_sizes(-1)


def test_judge_instrument_fingerprint_frozen_and_objective_rows_refuse():
    fps = {r: C.judge_instrument_fingerprint(r) for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored}
    assert len(set(fps.values())) == len(fps)  # rubrics differ per row
    with pytest.raises(ValueError, match="objective"):
        C.judge_instrument_fingerprint("correctness_math")


def test_seed_schedule_deterministic_and_validates_input():
    assert C.response_seed("p", 0) == C.response_seed("p", 0)
    assert C.response_seed("p", 0) != C.response_seed("p", 1)
    assert C.response_seed("p", 0) != C.response_seed("q", 0)
    with pytest.raises(ValueError):
        C.response_seed("", 0)
    with pytest.raises(ValueError):
        C.response_seed("p", -1)
    sha = hashlib.sha256(b"x").hexdigest()
    ids = C.judge_draw_ids(sha)
    assert len(ids) == C.JUDGE["n_draws"] == len(set(ids))
    with pytest.raises(ValueError):
        C.judge_draw_ids("not-a-sha")
