"""Independent draws and strict artifact identity are load-bearing for K3."""

import json

import pytest

from scripts import issue2054_k3 as k3


def test_source_spans_allow_renderer_suffix_and_reject_bad_answer(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {
        "conv_id": "c",
        "final_text": 'Wren: "hello"',
        "answer_start": 7,
        "answer_end": 12,
        "answer": "hello",
    }
    path.write_text(json.dumps(row) + "\n")
    assert k3.read_rows(path) == [row]
    row["answer"] = "other"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="answer span mismatch"):
        k3.read_rows(path)


def test_rollout_seeds_separate_cells_draws_and_contexts():
    values = [
        k3.seed(cell, cid, draw) for cell in ("a", "b") for cid in ("c", "d") for draw in (1, 2)
    ]
    assert len(set(values)) == 8
    assert k3.seed("a", "c", 1) == values[0]
    with pytest.raises(ValueError):
        k3.seed("a", "c", 0)


def test_checkpoint_requires_same_fingerprint_and_content(tmp_path):
    path = tmp_path / "chunk.json"
    path.write_text("[]")
    assert not k3.complete(path, "expected")
    k3.atomic_json(
        path.with_suffix(".json.done.json"), {"fingerprint": "expected", "sha256": k3.sha(path)}
    )
    assert k3.complete(path, "expected")
    with pytest.raises(RuntimeError):
        k3.complete(path, "different")
    path.write_text("[1]")
    with pytest.raises(RuntimeError):
        k3.complete(path, "expected")


def test_caps_preserve_parent_regeneration_cells():
    assert k3.cap_for("assistant__on_policy__chat__qwen2.5-7b") == 4096
    assert k3.cap_for("assistant__on_policy__chat__qwen2.5-7b-instruct") == 2048
    assert k3.cap_for("assistant__on_policy__bare_text__qwen2.5-7b-instruct") == 4096
    assert k3.cap_for("char_vex__on_policy__attrib_quoted__qwen2.5-7b") == 2048


def test_jsonl_unicode_separator_is_content(tmp_path):
    path = tmp_path / "unicode.jsonl"
    text = "a\u2028b"
    row = {
        "conv_id": "c",
        "final_text": text,
        "answer_start": 0,
        "answer_end": len(text),
        "answer": text,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert k3.read_rows(path) == [row]


def test_failed_receipt_upload_cannot_certify_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "raw.json"
    path.write_text("[]")
    calls = []

    def upload(p, root, relative_destination=None):
        calls.append(p)
        if len(calls) == 2:
            raise ConnectionError("receipt failed")
        return {"sha256": k3.sha(p)}

    monkeypatch.setattr(k3, "upload", upload)
    with pytest.raises(ConnectionError):
        k3.seal(path, tmp_path, "fp")
    assert not k3.complete(path, "fp")


def test_average_requires_exactly_three_valid_vectors():
    import numpy as np

    from scripts.issue2054_k3_fit import average_three

    first = np.array([[3.0, 6.0], [1.0, 2.0]])
    fresh = np.array([[[6.0, 9.0], [9.0, 12.0]], [[np.nan, np.nan], [3.0, 4.0]]])
    mean, keep = average_three(first, fresh, np.array([[True, True], [False, True]]))
    np.testing.assert_equal(keep, [True, False])
    np.testing.assert_allclose(mean, [[6.0, 9.0]])
    with pytest.raises(ValueError):
        average_three(first, fresh[:, :1], np.ones((2, 1), dtype=bool))


def test_primary_panel_cannot_complete_with_missing_or_withheld_cell():
    from scripts.issue2054_k3_fit import validate_primary

    rows = [
        {
            "cell": "a",
            "k_rollouts": count,
            "cohort": "all",
            "status": "complete",
            "folds": [{"fold": f} for f in range(5)],
        }
        for count in (1, 3)
    ]
    validate_primary(rows, ["a"])
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_primary(rows[:1], ["a"])
    rows[1]["status"] = "insufficient_ambient_training_rows"
    with pytest.raises(RuntimeError, match="did not complete"):
        validate_primary(rows, ["a"])


def test_empty_test_fold_is_withheld_despite_large_training_cohort():
    import numpy as np

    from scripts.issue2054_k3_fit import cohort_guard

    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 2))
    membership = np.repeat(np.arange(5), 10)
    targets = {1: rng.normal(size=(50, 2)), 3: rng.normal(size=(50, 2))}
    assert cohort_guard(x, targets, membership, np.ones(50, dtype=bool)) is None
    result = cohort_guard(x, targets, membership, membership != 0)
    assert result["status"] == "insufficient_held_out_rows"


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_chunk_allocation_covers_every_context_once_and_balances_long_cells(width):
    for cell in range(24):
        owners = []
        for offset in range(0, 8000, k3.CHUNK):
            assigned = [s for s in range(width) if k3.owns_chunk(cell, offset, s, width)]
            assert len(assigned) == 1
            owners.extend(assigned)
        assert (
            max(owners.count(s) for s in range(width)) - min(owners.count(s) for s in range(width))
            <= 1
        )
        assert [s for s in range(width) if k3.owns_chunk(cell, 0, s, width)] == [cell % width]


def test_shallow_git_prefetch_keeps_depth_and_checks_source(tmp_path):
    import subprocess

    from scripts.issue2054_k3_job import prepare_git

    source = tmp_path / "source"
    source.mkdir()

    def git(*args, cwd=source):
        return subprocess.check_output(
            ["git", "-c", "user.name=K3 Test", "-c", "user.email=k3-test@example.invalid", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    git("init", "--initial-branch=main")
    for i in range(5):
        (source / "file").write_text(str(i))
        git("add", "file")
        git("commit", "-m", f"main {i}")
    git("switch", "-c", "feature")
    (source / "feature").write_text("experiment")
    git("add", "feature")
    git("commit", "-m", "feature")
    expected = git("rev-parse", "HEAD")
    clone = tmp_path / "clone"
    git("clone", "--depth=1", "--branch=feature", source.as_uri(), str(clone))
    prepare_git(clone, expected)
    assert git("rev-list", "--all", "--count", cwd=clone) == "2"
    assert git("rev-parse", "origin/feature", cwd=clone) == expected
    with pytest.raises(RuntimeError, match="remote source changed"):
        prepare_git(clone, "0" * 40)


def test_failed_pilot_stops_sequence_before_production(monkeypatch):
    from scripts import issue2054_k3_job as job

    calls = []

    def fail_round(**kwargs):
        calls.append(kwargs["pilot"])
        raise RuntimeError("capture parity failed")

    monkeypatch.setattr(job, "run_round", fail_round)
    with pytest.raises(RuntimeError, match="capture parity failed"):
        job.run_sequence(pilot=False, pilot_then_production=True, source_sha="test")
    assert calls == [True]


@pytest.mark.parametrize(
    "pilot,chain,expected",
    [(False, True, [True, False]), (True, False, [True]), (False, False, [False])],
)
def test_successful_sequence_returns_only_final_round_report(monkeypatch, pilot, chain, expected):
    from scripts import issue2054_k3_job as job

    calls = []

    def run_round(**kwargs):
        calls.append(kwargs["pilot"])
        return {"pilot": kwargs["pilot"]}

    monkeypatch.setattr(job, "run_round", run_round)
    assert job.run_sequence(pilot=pilot, pilot_then_production=chain, source_sha="test") == {
        "pilot": expected[-1]
    }
    assert calls == expected
