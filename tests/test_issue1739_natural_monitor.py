"""Completion survives backend shutdown; stale receipts and heartbeats do not."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import experiment_watchdog
from scripts import issue1739_natural_monitor as monitor


def config(tmp_path):
    handles = []
    for behavior in sorted(monitor.BEHAVIORS):
        handle = tmp_path / f"{behavior}.handle.json"
        handle.write_text(json.dumps({"behavior": behavior}))
        handles.append(
            {
                "behavior": behavior,
                "handle_path": str(handle),
                "out_root": str(tmp_path / behavior),
                "upload_prefix": f"test/{behavior}",
                "input_fingerprint": "b" * 64,
                "launched_at": 1000,
            }
        )
    return {
        "source_sha": "a" * 40,
        "handles": handles,
        "data_repo": "test/repo",
        "state_path": str(tmp_path / "state.json"),
        "observation": str(tmp_path / "observation.json"),
        "startup_timeout": 600,
        "progress_timeout": 500,
        "phase_timeouts": {},
    }


def receipt_pair(cfg, item):
    result = {
        "source_sha": cfg["source_sha"],
        "behavior": item["behavior"],
        "input_fingerprint": item["input_fingerprint"],
        "finished_at": 1500,
        "datasets": [{"dataset": "test", "n": 20}],
    }
    blob = json.dumps(result).encode()
    receipt = {
        "source_sha": cfg["source_sha"],
        "behavior": item["behavior"],
        "input_fingerprint": item["input_fingerprint"],
        "time": 1600,
        "status": "complete",
        "verified_revision": "c" * 40,
        "verification": {
            "source_sha": cfg["source_sha"],
            "time": 1550,
            "phase": "complete",
            "prefix": item["upload_prefix"],
            "verified_revision": "c" * 40,
            "sha256": {"results.json": hashlib.sha256(blob).hexdigest()},
        },
    }
    return {"revision": "d" * 40, "receipt": receipt}, blob


def test_verified_hf_completion_needs_no_surviving_backend(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(monitor.time, "time", lambda: 2000)
    pairs = {item["upload_prefix"]: receipt_pair(cfg, item) for item in cfg["handles"]}

    class StoppedBackend:
        def published_completion(self, repo, prefix):
            assert repo == cfg["data_repo"]
            return pairs[prefix][0]

        def receipt(self, repo, prefix, revision):
            assert repo == cfg["data_repo"] and revision == "c" * 40
            return pairs[prefix][1]

        def probe(self, *_):
            raise AssertionError("A stopped/deleted GCP worker cannot provide SSH completion")

    observed = monitor.tick(cfg, {}, StoppedBackend())
    assert observed["status"] == "complete"
    assert observed["backend_observation"]["status"] == "done"
    assert set(observed["results"]["verified_revision"]) == monitor.BEHAVIORS
    wd = {**cfg, "monitor_unit": "test.service", "stale_seconds": 900}
    status, _ = experiment_watchdog.assessment(
        wd, SimpleNamespace(unit=lambda _: {"ActiveState": "inactive"}), 2000
    )
    assert status == "complete"


@pytest.mark.parametrize(
    "defect",
    [
        "stale_completion",
        "stale_upload",
        "stale_results",
        "wrong_source",
        "wrong_behavior",
        "wrong_input",
        "wrong_hash",
        "unverified_phase",
    ],
)
def test_published_receipt_fails_closed(tmp_path, monkeypatch, defect):
    cfg = config(tmp_path)
    item = cfg["handles"][0]
    published, blob = receipt_pair(cfg, item)
    monkeypatch.setattr(monitor.time, "time", lambda: 2000)
    done = published["receipt"]
    if defect == "stale_completion":
        done["time"] = 999
    elif defect == "stale_upload":
        done["verification"]["time"] = 999
    elif defect == "stale_results":
        result = json.loads(blob)
        result["finished_at"] = 999
        blob = json.dumps(result).encode()
        done["verification"]["sha256"]["results.json"] = hashlib.sha256(blob).hexdigest()
    elif defect == "wrong_source":
        done["source_sha"] = "f" * 40
    elif defect == "wrong_behavior":
        done["behavior"] = "other"
    elif defect == "wrong_input":
        done["input_fingerprint"] = "f" * 64
    elif defect == "wrong_hash":
        done["verification"]["sha256"]["results.json"] = "f" * 64
    else:
        done["verification"]["phase"] = "L15"
    runtime = SimpleNamespace(receipt=lambda *_: blob)
    with pytest.raises(ValueError):
        monitor.verify_published_completion(cfg, item, published, runtime)


def test_stopped_backend_without_receipt_is_failure(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(monitor.time, "time", lambda: 2000)
    runtime = SimpleNamespace(
        published_completion=lambda *_: None,
        probe=lambda item, _: {
            "backend_status": "TERMINATED",
            "handle_sha256": hashlib.sha256(Path(item["handle_path"]).read_bytes()).hexdigest(),
        },
    )
    observed = monitor.tick(cfg, {}, runtime)
    assert observed["status"] == "backend_observation_error"
    assert observed["backend_observation"]["status"] == "failed"


def test_heartbeat_and_log_mtime_cannot_reset_stall(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    item = cfg["handles"][0]
    monkeypatch.setattr(monitor.time, "time", lambda: 2000)
    fields = {"source_sha": cfg["source_sha"], "phase": "map_fit", "completed": 4}
    previous = {
        "handle_sha256": "handle",
        "last_advance": 1000,
        "progress_signature": monitor.identity([fields, 3, 100]),
    }
    probe = {
        "backend_status": "RUNNING",
        "handle_sha256": "handle",
        "remote_now": 2000,
        "process_alive": True,
        "process_command_matches": True,
        "process_start_ticks": 123,
        "documents": {
            "worker.pid": {"source_sha": cfg["source_sha"], "start_ticks": 123},
            "progress.json": {**fields, "time": 1999, "max_rss_gib": 20},
        },
        "document_mtimes": {"progress.json": 1999},
        "output_latest_mtime": 1999,
        "output_files": 3,
        "output_bytes": 100,
        "log_mtime": 1999,
    }
    with pytest.raises(ValueError, match="unchanged for 1000s"):
        monitor.evaluate(cfg, item, probe, copy.deepcopy(previous), None, 2000)


def test_quiet_fit_uses_phase_timeout_without_hiding_log_age(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["phase_timeouts"] = {"map_fit": 7200}
    monkeypatch.setattr(monitor.time, "time", lambda: 4000)

    def probe(item, source):
        return {
            "backend_status": "RUNNING",
            "remote_now": 4000,
            "handle_sha256": hashlib.sha256(Path(item["handle_path"]).read_bytes()).hexdigest(),
            "process_alive": True,
            "process_command_matches": True,
            "process_start_ticks": 123,
            "documents": {
                "worker.pid": {"source_sha": source, "start_ticks": 123},
                "progress.json": {"source_sha": source, "phase": "map_fit", "time": 1000},
            },
            "document_mtimes": {"progress.json": 1000},
            "output_latest_mtime": 1000,
            "output_files": 3,
            "output_bytes": 100,
            "log_mtime": 1000,
        }

    runtime = SimpleNamespace(published_completion=lambda *_: None, probe=probe)
    observed = monitor.tick(cfg, {}, runtime)
    assert observed["status"] == "running"
    assert observed["backend_observation"]["log_age_seconds"] == 3000
    wd = {**cfg, "monitor_unit": "test.service", "stale_seconds": 1800}
    status, _ = experiment_watchdog.assessment(
        wd, SimpleNamespace(unit=lambda _: {"ActiveState": "active"}), 4000
    )
    assert status == "healthy"
