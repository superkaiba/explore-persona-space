"""Production argument and orchestration seams, without renting compute."""

import inspect
import sys
import time

import pytest

from scripts import issue1739_natural_run as r


def test_composed_score_cli_uses_natural_pb(tmp_path):
    from scripts.issue1739_r2v2_score import parse_args

    args = r.parse_args(["fits", "--root", str(tmp_path)])
    ns = parse_args(r.score_cmd(args, "evil", 100000, 4)[2:])
    assert ns.natural_u_store == tmp_path / "pool/store"
    assert ns.generic_u == 100000
    assert ns.protocols == "B"  # The real parser accepts a comma-delimited string.
    assert ns.seeds == [4]
    assert ns.map_variants == ["true"]
    assert ns.transfer_preds


def test_materialized_transport_opt_in_is_forwarded(tmp_path):
    args = r.parse_args(
        [
            "stage",
            "--root",
            str(tmp_path),
            "--materialize-labeling-tars",
            "--labeling-tar-staging-dir",
            str(tmp_path / "tars"),
        ]
    )
    ns = r.input_args(args)
    assert ns.materialize_labeling_tars is True
    assert ns.labeling_tar_staging_dir == tmp_path / "tars"
    assert not r.input_args(
        r.parse_args(["stage", "--root", str(tmp_path)])
    ).materialize_labeling_tars


def test_production_data_one_contiguous_range_per_gpu(tmp_path, monkeypatch):
    args = r.parse_args(["data", "--root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(r, "gpu_free", lambda _: None)

    def fake_child(cmd, log, gpu):
        calls.append((gpu, cmd))
        return {"rc": 0}

    monkeypatch.setattr(r, "child", fake_child)
    assert len(r.data_wave(args, list(range(200)), "generate")) == 4
    ranges = sorted(
        (int(cmd[cmd.index("--start-chunk") + 1]), int(cmd[cmd.index("--end-chunk") + 1]))
        for _, cmd in calls
    )
    assert ranges == [(0, 50), (50, 100), (100, 150), (150, 200)]


def test_full_fits_are_owner_pilot_gated(tmp_path):
    args = r.parse_args(["fits", "--root", str(tmp_path)])
    with pytest.raises(ValueError, match="fit_pilot_accepted"):
        r.fits(args)


def test_prepared_requires_immutable_pin(tmp_path):
    args = r.parse_args(["data-pilot", "--root", str(tmp_path), "--prepared-revision", "main"])
    with pytest.raises(ValueError, match="immutable 40-hex"):
        r.stage_prepared(args)


def test_upload_failure_publishes_failure_not_success(tmp_path, monkeypatch):
    def fake_prepare(args):
        return {"test": "phase computation complete"}

    def fail_upload(*_args, **_kwargs):
        raise RuntimeError("transport failure")

    monkeypatch.setattr(r, "prepare", fake_prepare)
    monkeypatch.setattr(r, "upload_tree", fail_upload)
    sentinel_dir = tmp_path / "sentinels"
    with pytest.raises(RuntimeError, match="transport failure"):
        r.main(["prepare", "--root", str(tmp_path), "--sentinel-dir", str(sentinel_dir)])
    sentinels = list(sentinel_dir.glob("*.json"))
    assert len(sentinels) == 1 and sentinels[0].name.endswith("-failed.json")


def test_child_process_exit_and_pid_are_recorded(tmp_path):
    result = r.child([sys.executable, "-c", "print('natural_child_done')"], tmp_path / "child.log")
    assert result["rc"] == 0
    assert "natural_child_done" in (tmp_path / "child.log").read_text()
    assert (tmp_path / "child.pid.json").is_file()
    assert (tmp_path / "child.exit.json").is_file()
    with pytest.raises(RuntimeError, match="child rc=3"):
        r.child([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "failed.log")


def test_phase_completion_includes_launcher_stdout_in_upload(tmp_path, monkeypatch):
    args = r.parse_args(
        ["prepare", "--root", str(tmp_path), "--sentinel-dir", str(tmp_path / "sentinels")]
    )
    launch_log = tmp_path / "detached.log"
    launch_log.write_text("real phase progress\n")
    monkeypatch.setenv("EPS_NATURAL_LAUNCH_LOG", str(launch_log))
    uploaded = []

    def record_upload(local, _prefix):
        uploaded.extend(p.read_text() for p in local.rglob("*.log"))

    monkeypatch.setattr(r, "upload_tree", record_upload)
    r._complete_phase(args, "test", tmp_path / "reports/phase.json", time.monotonic(), {})
    assert uploaded == ["real phase progress\n"]
    assert (tmp_path / "sentinels/issue-1739-natural-prepare-test.json").is_file()


def test_upload_call_signature():
    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    inspect.signature(verify_repo_paths_uploaded).bind(
        object(), "repo", ["p/a"], path_in_repo="p", repo_type="dataset", revision="f" * 40
    )


def test_stage_copies_actual_committed_frozen_summary(tmp_path, monkeypatch):
    args = r.parse_args(["stage", "--root", str(tmp_path), "--behaviors", "evil"])
    monkeypatch.setattr(r.inputs, "stage_behavior", lambda *a: {"test": "remote boundary stub"})
    result = r.stage(args)
    target = tmp_path / "reused/eval_results/evil/arm_results/all_arms_spearman.json"
    source = r.ROOT / "eval_results/issue_1739/evil/arm_results/all_arms_spearman.json"
    assert r.data.file_sha(target) == r.data.file_sha(source)
    assert len(result["evil"]["frozen_summary_blob"]) == 40
