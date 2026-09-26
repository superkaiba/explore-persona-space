"""Exercise the real smoke body with controlled faults at the GPU capture boundary."""

import json
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from scripts.story_persona_kimi_runtime import KimiCapture, numerical_smoke


def smoke_fixture(fault=None):
    """Provide a signature-checked GPU boundary with distinguishable context vectors."""
    model = create_autospec(KimiCapture, instance=True)
    calls = []
    ids = [[i + 1] for i in range(2160)]
    ids[-1].append(9999)
    selected = list(dict.fromkeys([0, 2159] + [p * 240 + q for p in range(9) for q in (0, 1)]))

    def capture(ids_rows, *, check_tuple=False):
        assert len(ids_rows) == 1
        step = len(calls)
        calls.append((ids_rows[0], check_tuple))
        if fault == "later_phase_crash" and step == 6 + len(selected):
            raise RuntimeError("simulated GPU failure")
        value = torch.tensor([[[ids_rows[0][0], 1, 3], [2, 5, 7]]], dtype=torch.bfloat16)
        corrupt = (
            (fault == "production_repeat" and step == 1)
            or (fault == "checked_repeat" and step == 4)
            or (fault == "instrumentation" and check_tuple)
            or (fault == "return" and step == 5)
            or (fault == "full_replay" and step == 6 + len(selected))
            or (fault == "full_instrumentation" and step >= 6 + 2 * len(selected))
        )
        if corrupt:
            value *= 2
        model.last_evidence = [{"rank": 0, "sha256": "fake_gpu_boundary"}]
        return value, {"0": {"0": {"0": 0.0}}} if check_tuple else {}

    model.capture.side_effect = capture
    cfg = SimpleNamespace(capture=SimpleNamespace(repeatability_relative_tolerance=1e-5))
    return model, ids, cfg, calls, selected


def test_smoke_preserves_context_alignment_and_separates_instrumentation(tmp_path):
    model, ids, cfg, calls, selected = smoke_fixture()
    result = numerical_smoke(model, ids, cfg, tmp_path, "fingerprint")
    assert result["passed"]
    n = len(selected)
    assert [flag for _, flag in calls[:6]] == [False, False, False, True, True, False]
    assert [row for row, _ in calls[6 : 6 + n]] == [ids[i] for i in selected]
    assert [row for row, _ in calls[6 + n : 6 + 2 * n]] == [ids[i] for i in reversed(selected)]
    assert all(not flag for _, flag in calls[6 : 6 + 2 * n])
    assert all(flag for _, flag in calls[6 + 2 * n :])
    saved = torch.load(tmp_path / "smoke_vectors.pt", weights_only=True)
    assert saved["indices"] == selected
    assert torch.equal(saved["initial"], saved["repeated"])
    assert torch.equal(saved["initial"], saved["norm_checked"])
    assert all(
        (tmp_path / f"smoke_{phase}.pt").exists()
        for phase in ("initial", "repeated", "norm_checked")
    )


@pytest.mark.parametrize(
    "fault", ["production_repeat", "checked_repeat", "instrumentation", "return"]
)
def test_adjacent_faults_stop_before_full_capture_and_preserve_evidence(tmp_path, fault):
    model, ids, cfg, calls, _ = smoke_fixture(fault)
    with pytest.raises(RuntimeError, match="controlled instrumentation diagnostic rejected"):
        numerical_smoke(model, ids, cfg, tmp_path, "fingerprint")
    assert len(calls) == 6
    report = json.loads((tmp_path / "smoke_instrumentation.json").read_text())
    assert not report["passed"]
    assert (tmp_path / "smoke_instrumentation_vectors.pt").exists()
    assert not (tmp_path / "smoke_vectors.pt").exists()
    if fault == "instrumentation":
        assert max(report["relative_errors"]["checked_repeat"]) == 0
        assert max(report["relative_errors"]["production_repeat_1"]) == 0
        assert max(report["relative_errors"]["instrumentation_on"]) > 1e-5


@pytest.mark.parametrize("fault", ["full_replay", "full_instrumentation"])
def test_full_bank_faults_remain_blocking_and_save_vectors(tmp_path, fault):
    model, ids, cfg, _, _ = smoke_fixture(fault)
    with pytest.raises(RuntimeError, match="numerical smoke rejected"):
        numerical_smoke(model, ids, cfg, tmp_path, "fingerprint")
    report = json.loads((tmp_path / "smoke.json").read_text())
    assert not report["passed"]
    assert report["controlled_instrumentation_diagnostic_passed"]
    assert (tmp_path / "smoke_vectors.pt").exists()


def test_completed_smoke_phase_survives_next_phase_crash(tmp_path):
    model, ids, cfg, _, _ = smoke_fixture("later_phase_crash")
    with pytest.raises(RuntimeError, match="simulated GPU failure"):
        numerical_smoke(model, ids, cfg, tmp_path, "fingerprint")
    assert (tmp_path / "smoke_initial.pt").exists()
    assert (tmp_path / "smoke_instrumentation_vectors.pt").exists()
    assert not (tmp_path / "smoke.json").exists()
