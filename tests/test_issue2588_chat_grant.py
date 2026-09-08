"""Execute the real continuation clock and controller integration without compute."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_dispatch as D
import issue2588_chat_grant as C
import issue2588_chat_runtime as R


@pytest.fixture
def grant(tmp_path, monkeypatch):
    clock = [1788881000.0]
    monkeypatch.setattr(C.time, "time", lambda: clock[0])
    record = {
        "schema": 1,
        "authorization": C.AUTHORIZATION,
        "run_id": "qwen3-chat-v3",
        "allowance_s": 1800,
        "science_sha": C.SCIENCE_SHA,
        "pod_id": "fixture-pod",
        "provision_epoch": clock[0] - 100,
        "historical_terminal_revisions": ["9ba0d2275c97edc444bbc2d523086fa54045b6aa"],
    }
    path = tmp_path / "continuation.json"
    path.write_text(json.dumps(record))
    env = {
        "RUNPOD_POD_ID": record["pod_id"],
        "EPS2588_SMOKE_STARTED_AT": str(record["provision_epoch"]),
        C.GRANT_ENV: str(path),
        C.HASH_ENV: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return path, env, clock


def test_upload_credit_survives_retry_exactly_once(grant):
    _path, env, clock = grant
    a = C.from_env(env)
    assert a.remaining_s() == 1700
    a.finish_upload("attempt1:upload", clock[0] - 50, clock[0] - 20)
    a.finish_upload("attempt1:upload", clock[0] - 50, clock[0] - 20)
    assert a.credit_s() == 30
    clock[0] += 40
    b = C.from_env(env)
    assert b.remaining_s() == 1690  # Idle gap charged; upload excluded once.
    assert R.remaining_smoke_seconds(env, smoke=True) == 1690
    clock[0] += 1691
    with pytest.raises(R.RuntimeFence):
        R.remaining_smoke_seconds(env, smoke=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pod_id", "other"),
        ("allowance_s", 3600),
        ("science_sha", "wrong"),
        ("provision_epoch", 1788880000),
        ("authorization", "unapproved"),
    ],
)
def test_grant_rejects_scope_or_epoch_change(grant, field, value):
    path, env, _ = grant
    rec = json.loads(path.read_text())
    rec[field] = value
    path.write_text(json.dumps(rec))
    env[C.HASH_ENV] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        C.from_env(env)


def test_grant_refuses_rebinding_and_overlap(grant):
    _path, env, clock = grant
    a = C.from_env(env)
    a.finish_upload("upload1", clock[0] - 40, clock[0] - 10)
    with pytest.raises(RuntimeError, match="conflicting"):
        a.finish_upload("upload1", clock[0] - 39, clock[0] - 10)
    with pytest.raises(RuntimeError, match="overlapping"):
        a.finish_upload("upload2", clock[0] - 20, clock[0])


def test_grant_hash_legacy_and_missing_arguments(grant):
    _, env, _ = grant
    with pytest.raises(RuntimeError, match="hash"):
        C.from_env({**env, C.HASH_ENV: "wrong"})
    with pytest.raises(RuntimeError, match="legacy"):
        C.from_env({**env, "EPS2588_SMOKE_SUPPLEMENT_REPORT": "old.json"})
    with pytest.raises(RuntimeError, match="both"):
        C.from_env({C.GRANT_ENV: env[C.GRANT_ENV]})


def test_science_checkout_actual_git_boundary(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/issue2588_run_cell.py").touch()
    run = create_autospec(
        subprocess.run,
        side_effect=[
            subprocess.CompletedProcess([], 0, C.SCIENCE_SHA + "\n"),
            subprocess.CompletedProcess([], 0, ""),
        ],
    )
    monkeypatch.setattr(C.subprocess, "run", run)
    assert C.science_root(tmp_path) == tmp_path
    assert run.call_count == 2
    run.side_effect = [
        subprocess.CompletedProcess([], 0, "other"),
        subprocess.CompletedProcess([], 0, ""),
    ]
    with pytest.raises(RuntimeError, match="pinned"):
        C.science_root(tmp_path)


def test_controller_reuses_frozen_children_and_upload_ledger(grant, tmp_path):
    path, env, clock = grant
    args = D.build_parser().parse_args(
        [
            "--mode",
            "smoke",
            "--run-id",
            "qwen3-chat-v3",
            "--out-root",
            str(tmp_path / "out"),
            "--continuation-grant",
            str(path),
            "--continuation-grant-sha256",
            env[C.HASH_ENV],
            "--science-root",
            str(tmp_path / "science"),
        ]
    )
    steps = D.build_steps(args)
    assert [s.cell for s in steps if s.phase == "gen"] == ["q3_8b_b"]
    assert all(
        str(args.science_root) in s.argv[2]
        for s in steps
        if s.phase in {"gen", "capture", "runtime_check", "import_check"}
    )
    g = C.from_env(env)
    g.finish_upload("earlier", clock[0] - 50, clock[0] - 10)
    runner = D.Runner(args, env, {})
    try:
        assert abs(runner.initial_work_s - 60) < 1e-6
        assert runner.work_limit_s == 1800
        assert runner.report("running", 0)["continuation_grant"] == g.record
    finally:
        runner.main_log.close()


def test_inherited_grant_requires_identical_explicit_arguments(grant, monkeypatch):
    path, env, _ = grant
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    args = D.build_parser().parse_args(["--mode", "capture"])
    with pytest.raises(RuntimeError, match="explicit"):
        D.child_environment(args)
    with pytest.raises(RuntimeError, match="disagrees"):
        C.configure_environment(dict(env), path, "wrong")
    with pytest.raises(RuntimeError, match="disagrees"):
        C.configure_environment(dict(env), path.parent / "other.json", env[C.HASH_ENV])
    C.configure_environment(env, path, env[C.HASH_ENV])
    assert C.from_env(env).record["run_id"] == "qwen3-chat-v3"


def test_real_runner_upload_receipt_survives_retry(tmp_path, monkeypatch):
    import time

    epoch = time.time()
    record = {
        "schema": 1,
        "authorization": C.AUTHORIZATION,
        "run_id": "qwen3-chat-v3",
        "allowance_s": 1800,
        "science_sha": C.SCIENCE_SHA,
        "pod_id": "fixture-pod",
        "provision_epoch": epoch,
        "historical_terminal_revisions": ["9ba0d2275c97edc444bbc2d523086fa54045b6aa"],
    }
    path = tmp_path / "grant.json"
    path.write_text(json.dumps(record))
    env = {
        "RUNPOD_POD_ID": "fixture-pod",
        "EPS2588_SMOKE_STARTED_AT": str(epoch),
        C.GRANT_ENV: str(path),
        C.HASH_ENV: hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    args = D.build_parser().parse_args(
        [
            "--mode",
            "smoke",
            "--run-id",
            "qwen3-chat-v3",
            "--out-root",
            str(tmp_path / "out"),
            "--continuation-grant",
            str(path),
            "--continuation-grant-sha256",
            env[C.HASH_ENV],
        ]
    )
    step = D.Step(
        "q3_8b_b", "upload-partial", (sys.executable, "-c", "import time; time.sleep(0.1)")
    )
    limits = {
        step.key: {
            "bytes": 0,
            "bytes_per_s": 1,
            "retry_calls": 1,
            "timeout_s": 10,
            "basis": "test local child",
            "retry_exposure_s": 1,
        }
    }
    a = D.Runner(args, env, limits)
    try:
        a.run(step)
        credit = a.grant.credit_s()
        assert 0.09 < credit < 5
        assert a.records[0]["rc"] == 0
        assert abs(a.durability_s - credit) < 0.1
        assert len(list(a.grant.ledger.glob("upload-*.json"))) == 1
    finally:
        a.main_log.close()
    b = D.Runner(args, env, limits)
    try:
        assert b.grant.credit_s() == credit
        assert abs(b.work_s - (time.time() - epoch - credit)) < 0.1
        assert abs(b.work_s - a.work_s) < 0.1
    finally:
        b.main_log.close()
