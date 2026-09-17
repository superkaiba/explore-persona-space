"""Regressions for incomplete-analysis publication and early failure preservation."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.story_persona_crossmodel_artifacts import validate_analysis_files


def test_missing_or_mixed_layer_cannot_publish(tmp_path):
    """A completion marker cannot compensate for an absent or stale metric block."""
    files = ["analysis_complete", "summary", "raw_all_layers", "block_15", "block_31", "block_47"]
    for stem in files:
        (tmp_path / f"{stem}.json").write_text(json.dumps({"fingerprint": "current"}))
    with pytest.raises(ValueError, match="exact four"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    (tmp_path / "block_63.json").write_text(json.dumps({"fingerprint": "stale"}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    (tmp_path / "block_63.json").write_text(json.dumps({"fingerprint": "current"}))
    for path in tmp_path.glob("*.json"):
        value = {"fingerprint": "current", "analysis_fingerprint": "current-analysis"}
        if path.name.startswith("block_"):
            value.update(
                status="complete",
                fits={
                    name: {}
                    for name in ["full_bank", "fit_first_evaluate_last", "fit_last_evaluate_first"]
                },
            )
        path.write_text(json.dumps(value))
    for name in ["centroids.npz", "selected_vectors.npz"]:
        (tmp_path / name).write_bytes(b"fixture")
    outputs = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in tmp_path.iterdir()
        if p.name != "analysis_complete.json"
    }
    (tmp_path / "analysis_complete.json").write_text(
        json.dumps(
            {
                "fingerprint": "current",
                "analysis_fingerprint": "current-analysis",
                "outputs": outputs,
            }
        )
    )
    validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")
    changed = json.loads((tmp_path / "block_63.json").read_text())
    changed["changed"] = True
    (tmp_path / "block_63.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed after completion"):
        validate_analysis_files(tmp_path, "current", "Qwen/Qwen3.8-27B")


def test_early_failure_record_is_written_before_uploader(tmp_path):
    """Execute the wrapper's actual stdlib failure-record body on an empty output."""
    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts/story_persona_crossmodel_workload.sh"
    ).read_text()
    start = wrapper.index("    EPS_FAILURE_RC=\"$failed_rc\" python3 - <<'PY'\n")
    body = wrapper[start:].split("\n", 1)[1].split("\nPY\n", 1)[0]
    out = tmp_path / "out"
    log = tmp_path / "workload.log"
    log.write_text("storage contract refused\n")
    env = dict(
        os.environ,
        EPS_STORY_PERSONA_OUT=str(out),
        EPS_FAILURE_RC="124",
        EPS_STORY_PERSONA_MODEL_KEY="deepseek",
        EPS_STORY_PERSONA_SOURCE_SHA="a" * 40,
        EPS_STORY_MASTER_LOG=str(log),
    )
    subprocess.run(["python3", "-c", body], env=env, check=True)
    record = json.loads((out / "failure.json").read_text())
    assert record["exit_code"] == 124
    assert record["model_key"] == "deepseek"
    assert (out / "failure_workload.log").read_bytes() == log.read_bytes()
