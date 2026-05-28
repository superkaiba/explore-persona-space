"""Unit tests for the recurring training-pipeline infra fixes (#7a, #10, #13).

Covers three behavior-preserving guardrails added to the train -> eval pipeline:

#7a  ``_delete_intermediate_merged`` honors upload-before-delete: it deletes a
     consumed intermediate merged dir ONLY when its required upload already ran,
     and PRESERVES it (loud warning) when the inline-upload fence skipped the
     upload.

#10  ``_warn_if_cvd_disagrees`` emits a WARNING (and does NOT change the value)
     when an inherited CUDA_VISIBLE_DEVICES disagrees with gpu_id. The clobbering
     assignment ``os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)`` stays
     load-bearing; this only warns about a likely-misconfigured launch.

#13  ``run_isolated`` round-trips a JSON payload through a fresh ``uv run python
     -m`` child (the module's own ``_echo_main`` entry point), and fails loud on a
     non-zero child exit.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# #7a — upload-before-delete for intermediate merged dirs
# ---------------------------------------------------------------------------


def test_intermediate_merged_deleted_when_upload_attempted(tmp_path: Path) -> None:
    """Upload ran -> the intermediate merged dir is removed to reclaim disk."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    merged = tmp_path / "phase1_merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")

    _delete_intermediate_merged(merged, upload_attempted=True, label="Phase 1")

    assert not merged.exists()


def test_intermediate_merged_preserved_when_upload_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Upload skipped (fence set) -> the dir is PRESERVED, never silently dropped."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    merged = tmp_path / "phase1_merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")

    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.trainer"):
        _delete_intermediate_merged(merged, upload_attempted=False, label="Phase 1")

    assert merged.exists(), "un-uploaded intermediate must NOT be deleted"
    assert any("upload-before-delete" in rec.message for rec in caplog.records)


def test_intermediate_merged_missing_dir_is_noop(tmp_path: Path) -> None:
    """A non-existent dir is a no-op regardless of the upload flag."""
    from explore_persona_space.train.trainer import _delete_intermediate_merged

    missing = tmp_path / "does_not_exist"
    # Neither branch should raise.
    _delete_intermediate_merged(missing, upload_attempted=True, label="Phase 1")
    _delete_intermediate_merged(missing, upload_attempted=False, label="Phase 1")
    assert not missing.exists()


# ---------------------------------------------------------------------------
# #10 — CVD-disagreement warning (warn only; value unchanged)
# ---------------------------------------------------------------------------


def test_cvd_warning_fires_on_disagreement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Inherited CVD != gpu_id -> WARNING fires, env value is left untouched."""
    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(0)

    assert any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records), (
        "expected a CVD-disagreement WARNING"
    )
    # The helper must NOT mutate the env — the caller's assignment is what wins.
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_cvd_warning_silent_on_agreement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Inherited CVD == gpu_id -> no warning."""
    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(2)

    assert not any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records)


def test_cvd_warning_silent_when_env_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No inherited CVD -> no warning (the common single-GPU launch)."""
    from explore_persona_space.train import sft

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with caplog.at_level(logging.WARNING, logger="explore_persona_space.train.sft"):
        sft._warn_if_cvd_disagrees(0)

    assert not any("disagrees with cfg.gpu_id" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# #13 — run_isolated round-trip + fail-loud
# ---------------------------------------------------------------------------


def test_run_isolated_round_trips_payload() -> None:
    """run_isolated spawns a fresh child that echoes the payload back via JSON IPC."""
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    payload = {"seed": 42, "condition": "librarian", "nested": {"a": [1, 2, 3]}}
    result = run_isolated(
        "explore_persona_space.orchestrate.subprocess_isolation",
        payload,
        cwd=str(PROJECT_ROOT),
    )

    assert result["_echoed"] is True
    for key, value in payload.items():
        assert result[key] == value


def test_run_isolated_rejects_non_dict_payload() -> None:
    """A non-dict payload fails fast with TypeError before spawning anything."""
    from explore_persona_space.orchestrate.subprocess_isolation import run_isolated

    with pytest.raises(TypeError):
        run_isolated("explore_persona_space.orchestrate.subprocess_isolation", ["not", "a", "dict"])  # type: ignore[arg-type]


def test_run_isolated_fails_loud_on_nonzero_exit() -> None:
    """A child module that does not exist exits non-zero -> SubprocessIsolationError."""
    from explore_persona_space.orchestrate.subprocess_isolation import (
        SubprocessIsolationError,
        run_isolated,
    )

    with pytest.raises(SubprocessIsolationError):
        run_isolated(
            "explore_persona_space.orchestrate._module_that_does_not_exist_zzz",
            {"x": 1},
            cwd=str(PROJECT_ROOT),
        )


def test_echo_main_round_trips(tmp_path: Path) -> None:
    """The _echo_main entry point reads argv[1] and writes the echoed result to argv[2]."""
    from explore_persona_space.orchestrate.subprocess_isolation import _echo_main

    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(json.dumps({"k": "v"}))

    rc = _echo_main(["prog", str(in_path), str(out_path)])

    assert rc == 0
    result = json.loads(out_path.read_text())
    assert result == {"k": "v", "_echoed": True}


def test_echo_main_usage_error_on_missing_args() -> None:
    """Too few argv entries -> non-zero rc (usage error), no file written."""
    from explore_persona_space.orchestrate.subprocess_isolation import _echo_main

    assert _echo_main(["prog"]) == 2
