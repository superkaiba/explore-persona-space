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
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_train_lora_config_import_keeps_gpu_stack_lazy() -> None:
    """Importing the config dataclass must not import torch/TRL/vLLM."""
    code = """
import sys
from explore_persona_space.train.sft import TrainLoraConfig

cfg = TrainLoraConfig(gpu_id=2)
assert cfg.gpu_id == 2
forbidden_roots = ("torch", "trl", "vllm", "peft", "transformers", "datasets")
loaded = [
    root
    for root in forbidden_roots
    if root in sys.modules or any(name.startswith(root + ".") for name in sys.modules)
]
if loaded:
    raise SystemExit(f"unexpected heavy modules imported: {loaded}")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_trl_sft_loader_does_not_import_vllm() -> None:
    """Loading TRL's SFT classes for training must not load vLLM."""
    code = """
import sys
from explore_persona_space.train.sft import _load_trl_sft_classes

SFTConfig, SFTTrainer = _load_trl_sft_classes()
assert SFTConfig.__name__ == "SFTConfig"
assert SFTTrainer.__name__ == "SFTTrainer"
if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
    raise SystemExit("vLLM was imported while loading TRL SFT classes")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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
# #10 — CVD pin: an inherited single-GPU launcher pin is AUTHORITATIVE
# (#1090 fu3 crash-fix 2 — the #557/#543/#545 co-location class). This test
# FAILED pre-fix (the old code re-exported str(gpu_id) over the inherited pin,
# collapsing every CVD-pinned parallel cell onto physical GPU 0).
# ---------------------------------------------------------------------------


def test_cvd_inherited_single_gpu_pin_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launcher-pinned CVD=3 + default gpu_id=0 -> env NEVER mutated (no GPU-0 clobber)."""
    import os

    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert sft._apply_cvd_pin(0) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3", (
        "the inherited single-GPU pin must never be re-exported from gpu_id"
    )


def test_cvd_matching_pair_keeps_inherited_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sanctioned launcher pattern (CVD=2 + gpu_id=2) keeps the inherited pin."""
    import os

    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert sft._apply_cvd_pin(2) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_cvd_contradictory_pin_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inherited single-GPU CVD=1 + gpu_id=2 -> RuntimeError (two different pins)."""
    import os

    from explore_persona_space.train import sft

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="contradicts"):
        sft._apply_cvd_pin(2)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_cvd_exported_when_env_unset_or_multi_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy +gpu_id=N contract unchanged: unset / empty / multi-GPU env -> export."""
    import os

    from explore_persona_space.train import sft

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert sft._apply_cvd_pin(2) == "2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert sft._apply_cvd_pin(1) == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert sft._apply_cvd_pin(3) == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


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
