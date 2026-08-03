"""Tests for the GPU-idle ESCALATION tier in scripts/poll_pipeline.py (#664).

The escalation is the louder SECOND tier above the GPU-idle advisory: a
MULTI-GPU pod idle in an upload/CPU-only phase past
``EPM_GPU_IDLE_ESCALATION_MIN`` minutes fires a Telegram push + a
``[gpu-idle-escalation]`` marker (NEVER stops the pod). These tests pin:

* ``_phase_is_cpu_only`` — the deny-list / default-CPU truth table, ``unknown``
  ineligible;
* ``_gpu_idle_escalation_update`` — the pure decision core (below/at/above
  threshold, per-phase de-dup, single-GPU excluded, GPU-required phase
  excluded, shared idle span reused);
* the import-time clamp ``EPM_GPU_IDLE_ESCALATION_MIN >= EPM_GPU_IDLE_ADVISORY_MIN``;
* ``_telegram_push`` fail-soft — a missing script returns False, never raises.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_gpu_idle_escalation_under_test")


# ── _phase_is_cpu_only truth table ───────────────────────────────────────────


@pytest.mark.parametrize(
    "phase,expected",
    [
        # GPU-required deny-list substrings -> NOT cpu-only.
        ("training", False),
        ("p1_train", False),
        ("free_gen", False),
        ("judge_gen", False),
        ("eval", False),
        ("post_eval", False),
        ("generate", False),
        ("inference", False),
        ("forward_pass", False),
        ("vllm_load", False),
        ("setup", False),
        ("preflight", False),
        ("merge_adapter", False),
        # The unknown sentinel and empty are deliberately NOT eligible.
        ("unknown", False),
        ("", False),
        # Default-CPU-only: anything not matching the deny-list.
        ("p3_upload", True),  # the #664 trigger phase
        ("upload", True),
        ("p5_upload", True),
        ("aggregate", True),
        ("analysis", True),
        ("score", True),
        ("bootstrap", True),
        ("plot", True),
    ],
)
def test_phase_is_cpu_only_truth_table(phase: str, expected: bool) -> None:
    assert pp._phase_is_cpu_only(phase) is expected


# ── _gpu_idle_escalation_update decision core ────────────────────────────────


BASE_KW: dict[str, Any] = {
    "status": "running",
    "gpu_util": "0,0,0,0,0,0,0,0",  # 8-GPU pod, all idle
    "current_phase": "p3_upload",
    "idle_since_epoch": 1000,
    "escalated_phases": set(),
    "now_epoch": 1000 + 60 * 60,  # 60 min after the span started
    "escalation_min": 60,
}


def test_below_threshold_does_not_escalate() -> None:
    kw = {**BASE_KW, "now_epoch": 1000 + 59 * 60}  # 59 min < 60
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_at_and_above_threshold_escalates() -> None:
    assert pp._gpu_idle_escalation_update(**BASE_KW).should_escalate is True  # exactly 60
    above = {**BASE_KW, "now_epoch": 1000 + 90 * 60}
    assert pp._gpu_idle_escalation_update(**above).should_escalate is True


def test_already_escalated_phase_does_not_re_escalate() -> None:
    kw = {**BASE_KW, "escalated_phases": {"p3_upload"}}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_single_gpu_pod_does_not_escalate() -> None:
    kw = {**BASE_KW, "gpu_util": "0"}  # one card -> not multi-GPU
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_aggregate_phase_on_multi_gpu_escalates() -> None:
    kw = {**BASE_KW, "current_phase": "aggregate"}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is True


@pytest.mark.parametrize("phase", ["training", "free_gen", "eval", "vllm_load", "unknown"])
def test_gpu_required_or_unknown_phase_does_not_escalate(phase: str) -> None:
    kw = {**BASE_KW, "current_phase": phase}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_busy_gpu_does_not_escalate() -> None:
    kw = {**BASE_KW, "gpu_util": "0,0,95,0,0,0,0,0"}  # one card busy
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_unknown_gpu_sample_does_not_escalate() -> None:
    kw = {**BASE_KW, "gpu_util": "unknown"}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_non_running_status_does_not_escalate() -> None:
    kw = {**BASE_KW, "status": "stalled"}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_no_active_span_does_not_escalate() -> None:
    kw = {**BASE_KW, "idle_since_epoch": 0}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_escalation_disabled_when_min_non_positive() -> None:
    kw = {**BASE_KW, "escalation_min": 0}
    assert pp._gpu_idle_escalation_update(**kw).should_escalate is False


def test_escalation_reuses_the_shared_advisory_span() -> None:
    """The escalation reads the SAME idle_since_epoch the advisory resolved —
    there is no second idle clock. The span at this tick equals
    now_epoch - idle_since_epoch."""
    update = pp._gpu_idle_escalation_update(**BASE_KW)
    assert update.idle_span_sec == 60 * 60


# ── import-time clamp (escalation_min >= advisory_min) ───────────────────────


def test_escalation_min_clamped_up_to_advisory_min(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured escalation min BELOW the advisory min is clamped UP at
    import (escalate only AFTER advising). Re-import the module with the env
    set low and the advisory min high."""
    monkeypatch.setenv("EPM_GPU_IDLE_ADVISORY_MIN", "30")
    monkeypatch.setenv("EPM_GPU_IDLE_ESCALATION_MIN", "10")  # below advisory
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_clamp_under_test")
    assert reloaded.GPU_IDLE_ADVISORY_MIN == 30
    assert reloaded.GPU_IDLE_ESCALATION_MIN == 30  # clamped up, not 10


def test_escalation_min_not_clamped_when_above_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EPM_GPU_IDLE_ADVISORY_MIN", "30")
    monkeypatch.setenv("EPM_GPU_IDLE_ESCALATION_MIN", "90")
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_noclamp_under_test")
    assert reloaded.GPU_IDLE_ESCALATION_MIN == 90


def test_escalation_disabled_value_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0 (disable) escalation min is NOT clamped up — disabling must survive."""
    monkeypatch.setenv("EPM_GPU_IDLE_ADVISORY_MIN", "30")
    monkeypatch.setenv("EPM_GPU_IDLE_ESCALATION_MIN", "0")
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_disabled_under_test")
    assert reloaded.GPU_IDLE_ESCALATION_MIN == 0


# ── _telegram_push fail-soft ─────────────────────────────────────────────────


def test_telegram_push_missing_script_returns_false_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing push script returns False and NEVER raises — so a missing
    my-goat install degrades the escalation to marker-only, never crashes the
    poller."""
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(tmp_path / "does_not_exist.sh"))
    assert pp._telegram_push("hello") is False


def test_telegram_push_returns_true_on_zero_rc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present script that exits 0 -> True (confirmed enqueue)."""
    script = tmp_path / "push.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert pp._telegram_push("hello") is True


def test_telegram_push_returns_false_on_nonzero_rc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "push.sh"
    script.write_text("#!/usr/bin/env bash\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.setenv("EPM_TELEGRAM_PUSH_SCRIPT", str(script))
    assert pp._telegram_push("hello") is False


# ── #1752 escalation-count parse/serialize + the width-re-eval knob ──────────


def test_parse_escalation_counts_round_trip() -> None:
    """The comma-joined ``phase:count`` format round-trips through the
    serializer/parser pair; serialization is sorted for a deterministic
    state file."""
    counts = {"p3_upload": 2, "fit_ladder": 3}
    raw = pp._serialize_escalation_counts(counts)
    assert raw == "fit_ladder:3,p3_upload:2"
    assert pp._parse_escalation_counts(raw) == counts
    assert pp._serialize_escalation_counts({}) == ""


def test_parse_escalation_counts_malformed_tolerance() -> None:
    """Malformed entries (missing colon, empty phase, non-integer /
    non-positive count) are DROPPED — never raised — while valid siblings are
    kept; an unparsable count restarts at 0 (one extra identical note, never
    a suppressed one)."""
    assert pp._parse_escalation_counts("") == {}
    assert pp._parse_escalation_counts("   ") == {}
    assert pp._parse_escalation_counts(None or "") == {}
    raw = "no_colon,:3,phase:,phase:x,neg:-1,zero:0,ok:2, spaced:4 "
    assert pp._parse_escalation_counts(raw) == {"ok": 2, "spaced": 4}


def test_width_reeval_knob_default_is_three(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escalate-in-kind threshold defaults to 3 (one-more-chance
    semantics over #1689's two identical fires)."""
    monkeypatch.delenv("EPM_GPU_IDLE_WIDTH_REEVAL_N", raising=False)
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_reeval_default_under_test")
    assert reloaded.GPU_IDLE_WIDTH_REEVAL_N == 3


def test_width_reeval_knob_env_override_and_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low values (1/2) are honored — escalate-in-kind sooner — and ``<= 0``
    disables the width-re-eval variant (identical notes forever, the pre-fix
    behavior)."""
    monkeypatch.setenv("EPM_GPU_IDLE_WIDTH_REEVAL_N", "1")
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_reeval_one_under_test")
    assert reloaded.GPU_IDLE_WIDTH_REEVAL_N == 1
    monkeypatch.setenv("EPM_GPU_IDLE_WIDTH_REEVAL_N", "0")
    reloaded = _load_script_module("poll_pipeline.py", "poll_pipeline_reeval_disabled_under_test")
    assert reloaded.GPU_IDLE_WIDTH_REEVAL_N == 0
