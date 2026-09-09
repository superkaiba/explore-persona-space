"""Check fresh-process execution and stop-before-capture on upload failure."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "k3_job", Path(__file__).resolve().parents[1] / "scripts/issue2588_k3_job.py"
)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


@pytest.fixture
def phase_driver(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "issue2588_k3_train_refit.py").write_text(
        "import os,sys,json\n"
        "from pathlib import Path\n"
        "phase = sys.argv[sys.argv.index('--phase')+1] if '--phase' in sys.argv else 'check'\n"
        "with Path(os.environ['K3_TEST_RECORD']).open('a') as stream:\n"
        " stream.write(json.dumps({'phase':phase,'pid':os.getpid()})+'\\n')\n"
        "sys.exit(17 if phase == os.environ.get('K3_TEST_FAIL') else 0)\n"
    )
    record = tmp_path / "record.jsonl"
    monkeypatch.setattr(job, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("K3_TEST_RECORD", str(record))
    return tmp_path, record


def test_each_phase_uses_fresh_process(phase_driver):
    root, record = phase_driver
    job.run_cell(sys.executable, argparse.Namespace(pilot=True, cell=None), root / "out")
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [row["phase"] for row in rows] == [
        "check",
        "prologue",
        "gate",
        "gen",
        "parse",
        "upload-raw",
        "capture",
        "upload-capture",
        "fits",
        "upload-fits",
    ]
    assert len({row["pid"] for row in rows}) == len(rows)


def test_raw_upload_failure_prevents_capture(phase_driver, monkeypatch):
    root, record = phase_driver
    monkeypatch.setenv("K3_TEST_FAIL", "upload-raw")
    with pytest.raises(subprocess.CalledProcessError) as error:
        job.run_cell(sys.executable, argparse.Namespace(pilot=True, cell=None), root / "out")
    assert error.value.returncode == 17
    phases = [json.loads(line)["phase"] for line in record.read_text().splitlines()]
    assert phases[-1] == "upload-raw"
    assert "capture" not in phases


def test_flashinfer_patch_preserves_docstring_and_is_idempotent():
    source = (
        '"""Module documentation."""\nimport array\ndef f(x: array.array[int]):\n    return x\n'
    )
    patched = job.postponed_annotations(source)
    namespace = {}
    exec(compile(patched, "fixture", "exec"), namespace)
    assert namespace["__doc__"] == "Module documentation."
    assert namespace["f"].__annotations__["x"] == "array.array[int]"
    assert job.postponed_annotations(patched) == patched


def test_flashinfer_patch_rejects_unexpected_source():
    with pytest.raises(RuntimeError, match="unexpected flashinfer source"):
        job.postponed_annotations("def f(x):\n    return x\n")


def test_source_fence_checks_real_checkout():
    actual = subprocess.check_output(
        ["git", "-C", str(job.REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    job.check_source(actual)
    with pytest.raises(RuntimeError, match="source mismatch"):
        job.check_source("0" * 40)
