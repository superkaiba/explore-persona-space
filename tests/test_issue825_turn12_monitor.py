"""Completion proves exact source and archives; liveness requires timely real output."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue825_turn12_monitor.py"
spec = importlib.util.spec_from_file_location("turn12_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def setup_api(monkeypatch, tmp_path):
    source = "a" * 40
    values = dict(
        source_sha=source,
        source_conditions=["12"],
        n_conversations=4975,
        answer_draws=1,
        models={
            m: dict(
                cells=[
                    dict(source="12", method=k, target_turn=t)
                    for k in ("raw", "source_identity_bias")
                    for t in range(1, 13)
                ]
            )
            for m in ("instruct", "pretrained")
        },
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps(values))
    tracking = tmp_path / "tracking.bin"
    tracking.write_bytes(b"offline tracking fixture")
    required = {f"{m}_oof.npz" for m in values["models"]}
    required.update(
        f"maps/{m}_12_folds{f}-{f + 1}.npz" for m in values["models"] for f in (0, 2, 4)
    )
    required.update(f"predictions/{m}_12_fold{f}.npz" for m in values["models"] for f in range(6))
    tensor_files = {
        monitor.PREFIX + "/numerical/" + p: dict(size=100, sha256="b" * 64) for p in required
    }
    text_name = monitor.PREFIX + "/analysis/results.json"
    track_name = monitor.PREFIX + "/tracking/run.wandb"
    paths = {text_name: results, track_name: tracking}
    archives = {}
    for kind, suffix, files in [
        ("tensor", "numerical", tensor_files),
        (
            "text",
            "analysis",
            {
                text_name: dict(
                    size=results.stat().st_size,
                    sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
                )
            },
        ),
        (
            "tracking",
            "tracking",
            {
                track_name: dict(
                    size=tracking.stat().st_size,
                    sha256=hashlib.sha256(tracking.read_bytes()).hexdigest(),
                )
            },
        ),
    ]:
        archives[kind] = dict(
            status="verified",
            repo="superkaiba1/explore-persona-space-overflow" if kind == "tensor" else monitor.REPO,
            repo_type="model" if kind == "tensor" else "dataset",
            prefix=monitor.PREFIX + "/" + suffix,
            revision="c" * 40,
            files=files,
            count=len(files),
        )
    proof = dict(
        source_sha=source,
        status="complete",
        archives=archives,
        results_sha256=hashlib.sha256(results.read_bytes()).hexdigest(),
    )
    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps(proof))
    completion_name = monitor.PREFIX + "/runtime/complete.json"
    paths[completion_name] = complete
    entries = {completion_name: SimpleNamespace(path=completion_name)}
    for receipt in archives.values():
        for name, row in receipt["files"].items():
            entries[name] = SimpleNamespace(
                path=name,
                size=row["size"],
                lfs=SimpleNamespace(sha256=row["sha256"]) if name in tensor_files else None,
            )
    api = create_autospec(monitor.HfApi, instance=True)
    api.repo_info.return_value = SimpleNamespace(sha="d" * 40)
    api.get_paths_info.side_effect = lambda repo, names, **kw: [
        entries[n] for n in names if n in entries
    ]
    download = create_autospec(
        monitor.hf_hub_download, side_effect=lambda repo, name, **kw: str(paths[name])
    )
    monkeypatch.setattr(monitor, "HfApi", create_autospec(monitor.HfApi, return_value=api))
    monkeypatch.setattr(monitor, "hf_hub_download", download)
    return proof, complete, entries


def test_verified_completion_checks_all_artifacts(monkeypatch, tmp_path):
    setup_api(monkeypatch, tmp_path)
    assert monitor.verified_completion("a" * 40)["verified_revision"] == "d" * 40


@pytest.mark.parametrize(
    "case", ["wrong_source", "empty_archives", "missing_map", "wrong_digest", "wrong_result_digest"]
)
def test_completion_refuses_unproven_artifacts(monkeypatch, tmp_path, case):
    proof, path, entries = setup_api(monkeypatch, tmp_path)
    if case == "wrong_source":
        proof["source_sha"] = "e" * 40
    elif case == "empty_archives":
        proof["archives"] = {}
    elif case == "missing_map":
        proof["archives"]["tensor"]["files"].pop(next(iter(proof["archives"]["tensor"]["files"])))
    elif case == "wrong_digest":
        next(e for e in entries.values() if getattr(e, "lfs", None)).lfs.sha256 = "f" * 64
    else:
        proof["results_sha256"] = "f" * 64
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        monitor.verified_completion("a" * 40)


def test_missing_completion_is_pending(monkeypatch, tmp_path):
    _, _, entries = setup_api(monkeypatch, tmp_path)
    entries.pop(monitor.PREFIX + "/runtime/complete.json")
    assert monitor.verified_completion("a" * 40) is None


@pytest.mark.parametrize("scores,newest,now", [(0, None, 601), (3, 1, 602), (12, 2, 1300)])
def test_fresh_logs_do_not_hide_stale_outputs(scores, newest, now):
    state = dict(started_at=0, workload_observed_at=0, all_scores_observed_at=0)
    observed = dict(
        status="running",
        current_phase="workload",
        pid_alive=True,
        last_log_mtime_sec_ago=0,
        output_progress=dict(fits=0, scores=scores, newest=newest, worker_pids=[22]),
    )
    with pytest.raises(RuntimeError):
        monitor.check_progress(state, observed, now)


def test_bootstrap_is_pending_but_bounded():
    observed = dict(
        status="running", current_phase="startup", pid_alive=True,
        last_log_mtime_sec_ago=10**9, reachability_alarm=True
    )
    pending = monitor.check_progress(dict(started_at=0), observed, 100)
    assert pending["status"] == "pending" and "pid_alive" not in pending
    assert "reachability_alarm" not in pending
    assert pending["startup_observation"] == observed
    with pytest.raises(RuntimeError):
        monitor.check_progress(dict(started_at=0), observed, 1201)


def test_dead_worker_is_not_a_live_vm():
    state = dict(started_at=0, workload_observed_at=0)
    observed = dict(
        status="running",
        current_phase="workload",
        pid_alive=False,
        output_progress=dict(fits=1, scores=0, newest=100, worker_pids=[]),
    )
    with pytest.raises(RuntimeError):
        monitor.check_progress(state, observed, 100)
