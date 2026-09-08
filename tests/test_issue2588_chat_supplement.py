"""Supplemental clock/dispatch bodies; boundaries only are faked, no network/GPU."""

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_dispatch as D
import issue2588_chat_runtime as RT


@pytest.fixture
def grant(tmp_path, monkeypatch):
    """Pin a small terminal-report fixture at the external historical-file boundary."""
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "run_id": RT.SMOKE_RUN_ID,
                "status": "halted",
                "rc": 8,
                "attempt": "fixture",
                "work_elapsed_s": 3600.415,
                "smoke_original_epoch": "1788833514",
            }
        )
    )
    monkeypatch.setattr(
        RT, "SUPPLEMENT_REPORT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for key in (RT.SMOKE_PRIOR_REPORT_ENV, RT.SMOKE_PRIOR_HASH_ENV, RT.SUPPLEMENT_REPORT_ENV):
        monkeypatch.delenv(key, raising=False)
    env = {RT.SMOKE_START_ENV: str(time.time() - 60), "RUNPOD_POD_ID": "fixture-pod"}
    return path, env


def test_supplement_records_history_and_binds_one_epoch(grant, tmp_path):
    path, env = grant
    record = RT.validate_smoke_supplement(path, env, run_id=RT.SUPPLEMENT_RUN_ID)
    assert record["historical_work_s"] == 3600.415
    assert record["allowance_s"] == 1800
    RT.bind_supplement_clock(tmp_path, record, env)
    RT.bind_supplement_clock(tmp_path, record, env)
    for changed in ({RT.SMOKE_START_ENV: str(time.time())}, {"RUNPOD_POD_ID": "other"}):
        with pytest.raises(RuntimeError, match="cannot renew"):
            RT.bind_supplement_clock(tmp_path, record, {**env, **changed})
    with pytest.raises(RuntimeError, match="pod ID"):
        RT.bind_supplement_clock(tmp_path, record, {RT.SMOKE_START_ENV: env[RT.SMOKE_START_ENV]})
    env[RT.SUPPLEMENT_REPORT_ENV] = str(path)
    assert 1735 < RT.remaining_smoke_seconds(env, smoke=True) <= 1740


def test_supplement_refuses_wrong_report_epoch_run_or_credit(grant):
    path, env = grant
    with pytest.raises(RuntimeError, match="v3"):
        RT.validate_smoke_supplement(path, env, run_id=RT.SMOKE_RUN_ID)
    with pytest.raises(RuntimeError, match="credit"):
        RT.validate_smoke_supplement(
            path, {**env, RT.SMOKE_PRIOR_REPORT_ENV: "old"}, run_id=RT.SUPPLEMENT_RUN_ID
        )
    for bad in ("nan", "inf", "1788833514", str(time.time() + 100)):
        with pytest.raises(RuntimeError, match="epoch"):
            RT.validate_smoke_supplement(
                path, {**env, RT.SMOKE_START_ENV: bad}, run_id=RT.SUPPLEMENT_RUN_ID
            )
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="exact exhausted"):
        RT.validate_smoke_supplement(path, env, run_id=RT.SUPPLEMENT_RUN_ID)


def test_supplement_schedules_only_unfinished_thinking_pilot(grant, tmp_path):
    path, env = grant
    args = D.build_parser().parse_args(
        [
            "--mode",
            "smoke",
            "--run-id",
            RT.SUPPLEMENT_RUN_ID,
            "--smoke-supplement-report",
            str(path),
            "--out-root",
            str(tmp_path),
        ]
    )
    steps = D.build_steps(args)
    assert [s.cell for s in steps if s.phase == "gen"] == ["q3_8b_b"]
    assert [s.cell for s in steps if s.phase == "capture"] == ["q3_8b_b"]
    assert all(s.phase == "stage-runtime" for s in steps if s.cell == "q3_8b_a")
    runner = D.Runner(args, {**env, "CUDA_VISIBLE_DEVICES": "0"}, {})
    try:
        assert runner.work_limit_s == 1800 and 60 <= runner.work_s < 65
        report = runner.report("running", 0)
        assert report["smoke_supplement"]["historical_work_s"] == 3600.415
        runner.initial_work_s = 1800
        with pytest.raises(D.PhaseFailure) as caught:
            runner.run(D.cell_step(args, "q3_8b_b", "gen"))
        assert caught.value.rc == 8 and not runner.records
    finally:
        runner.main_log.close()


def test_supplement_cannot_be_used_for_production(grant):
    path, _ = grant
    for mode in ("capture", "fit-pilot", "fits"):
        with pytest.raises(SystemExit):
            D.main(
                [
                    "--mode",
                    mode,
                    "--run-id",
                    RT.SUPPLEMENT_RUN_ID,
                    "--smoke-supplement-report",
                    str(path),
                ]
            )
