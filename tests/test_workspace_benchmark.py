"""A failed GPU sampler cannot leave its native benchmark child running."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_sampler_failure_reaps_child_and_saves_failed_receipt(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "jr_benchmark", Path(__file__).parents[1] / "scripts/workspace_jr_dim_batch_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    child = Child()
    monkeypatch.setattr(module, "gpu_preflight", lambda: "GPU-test, A100, 81920")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: child)

    def failed_sampler(*args, **kwargs):
        raise RuntimeError("simulated unavailable sampler")

    monkeypatch.setattr(module.subprocess, "run", failed_sampler)
    candidate, report = {}, {"candidates": []}
    with pytest.raises(RuntimeError, match="unavailable sampler"):
        module.sample_child(
            ["native"], tmp_path / "child.log", candidate, tmp_path / "report.json", report
        )
    assert child.returncode == -15
    saved = json.loads((tmp_path / "report.json").read_text())["candidates"][0]
    assert saved["status"] == "failed" and saved["exit_code"] == -15
    assert saved["sampled_peak_gpu_used_mib"] is None and saved["gpu_memory_samples"] == 0
