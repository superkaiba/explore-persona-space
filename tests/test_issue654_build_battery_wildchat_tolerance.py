"""Regression tests for scripts/issue654_build_battery.py round-4 BLOCKER fix.

Concern: wildchat_subprocess_interpreter_shutdown_misclassified_as_failure.

The #617 WildChat-slice subprocess can abort during Python interpreter
shutdown (rc=134 / `Fatal Python error: PyGILState_Release` race in the
HuggingFace datasets+transformers C extensions) AFTER it has already written a
complete, well-formed wildchat_slice.json. The old `ensure_wildchat_slice`
treated ANY non-zero rc as fatal and raised, which propagated through the
dispatcher's `set -euo pipefail` EXIT trap and powered off the GCE instance —
wasting the entire provision even though the work succeeded (incident:
epm:failure v3 on task #654, GCE eps-issue-654, 2026-06-17).

These tests pin the artifact-first acceptance contract:

  - test_rc134_with_good_artifact_returns_silently
    The subprocess returns rc=134 AFTER writing a slice with
    meta.n_conversations >= target -> ensure_wildchat_slice returns silently.

  - test_rc134_with_missing_artifact_raises
    The subprocess returns rc=134 and writes NO slice (genuine failure) ->
    ensure_wildchat_slice still raises loud.

  - test_rc0_with_short_artifact_raises
    The subprocess returns rc=0 but the slice holds fewer than `target`
    conversations (genuine shortfall) -> ensure_wildchat_slice still raises.

Pure CPU; monkeypatches subprocess.run + the module-level slice path; uses
tmp_path for isolation; no GPU, no model load, no real #617 invocation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "issue654_build_battery.py"
_spec = importlib.util.spec_from_file_location("issue654_build_battery_under_test", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
battery_mod = importlib.util.module_from_spec(_spec)
sys.modules["issue654_build_battery_under_test"] = battery_mod
_spec.loader.exec_module(battery_mod)


def _fake_run_factory(slice_path: Path, returncode: int, n_conversations: int | None):
    """Build a subprocess.run replacement.

    When ``n_conversations`` is not None, the fake writes a well-formed
    wildchat_slice.json holding that many conversations BEFORE returning a
    CompletedProcess with ``returncode`` (mimicking the #617 helper finishing
    its write, then the interpreter-shutdown abort). When None, it writes
    nothing (genuine failure path).
    """

    def _fake_run(cmd, *args, **kwargs):
        if n_conversations is not None:
            convs = [
                {
                    "conv_id": f"c{i}",
                    "short_prefix_msgs": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ],
                }
                for i in range(n_conversations)
            ]
            slice_path.parent.mkdir(parents=True, exist_ok=True)
            slice_path.write_text(
                json.dumps({"meta": {"n_conversations": n_conversations}, "conversations": convs})
            )
        return subprocess.CompletedProcess(args=cmd, returncode=returncode)

    return _fake_run


def test_rc134_with_good_artifact_returns_silently(tmp_path, monkeypatch) -> None:
    """rc=134 AFTER a well-formed slice (n >= target) -> accept, return None."""
    target = 30
    slice_path = tmp_path / "wildchat_slice.json"
    monkeypatch.setattr(battery_mod, "WILDCHAT_SLICE_PATH", slice_path)
    monkeypatch.setattr(
        battery_mod.subprocess,
        "run",
        _fake_run_factory(slice_path, returncode=134, n_conversations=target),
    )

    # Must NOT raise: the interpreter-shutdown abort is tolerated when the
    # artifact is present + well-formed.
    assert battery_mod.ensure_wildchat_slice(target) is None
    written = json.loads(slice_path.read_text())
    assert written["meta"]["n_conversations"] == target


def test_rc134_with_missing_artifact_raises(tmp_path, monkeypatch) -> None:
    """rc=134 with NO slice written (genuine failure) -> still raises loud."""
    target = 30
    slice_path = tmp_path / "wildchat_slice.json"
    monkeypatch.setattr(battery_mod, "WILDCHAT_SLICE_PATH", slice_path)
    monkeypatch.setattr(
        battery_mod.subprocess,
        "run",
        _fake_run_factory(slice_path, returncode=134, n_conversations=None),
    )

    with pytest.raises(RuntimeError, match="WildChat slice build failed"):
        battery_mod.ensure_wildchat_slice(target)
    assert not slice_path.exists()


def test_rc0_with_short_artifact_raises(tmp_path, monkeypatch) -> None:
    """rc=0 but n_conversations < target (genuine shortfall) -> still raises."""
    target = 30
    slice_path = tmp_path / "wildchat_slice.json"
    monkeypatch.setattr(battery_mod, "WILDCHAT_SLICE_PATH", slice_path)
    monkeypatch.setattr(
        battery_mod.subprocess,
        "run",
        _fake_run_factory(slice_path, returncode=0, n_conversations=target - 1),
    )

    with pytest.raises(RuntimeError, match="WildChat slice build failed"):
        battery_mod.ensure_wildchat_slice(target)
