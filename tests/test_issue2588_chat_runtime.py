"""Real runtime setup/gate bodies with explicit external process/import boundaries."""

from __future__ import annotations

import ast
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2588_chat_runtime as RT


def test_parent_pins_full_hash_lock_and_commands():
    text = RT.LOCK.read_text()
    for package, version in RT.PINS.items():
        assert f"{package}=={version} \\\n    --hash=sha256:" in text
    commands = RT.build_commands("/usr/bin/uv")
    assert commands[0][-2:] == ["--python", "3.12.14"]
    assert "--require-hashes" in commands[1]
    assert commands[1][commands[1].index("--torch-backend") + 1] == "cu130"
    assert str(RT.LOCK) in commands[1]
    assert all("uninstall" not in command for command in commands)
    assert commands[-1][-1] == "--check"


def test_patch_preserves_docstring_and_defers_only_annotations():
    original = '# encoding: utf-8\n"""retained docstring"""\nimport array\nx: array.array[int]\n'
    patched = RT.postponed_annotations(original)
    assert ast.get_docstring(ast.parse(patched)) == "retained docstring"
    assert patched.count("from __future__ import annotations") == 1
    assert RT.postponed_annotations(patched) == patched
    namespace = {}
    exec(compile(patched, "fixture.py", "exec"), namespace)
    assert namespace["__annotations__"]["x"] == "array.array[int]"
    with pytest.raises(RuntimeError, match="annotation absent"):
        RT.postponed_annotations("import array\nx = 1\n")


def test_patch_actual_owned_source_and_receipt(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    target = runtime / "lib/flashinfer/comm/fd_exchange.py"
    target.parent.mkdir(parents=True)
    target.write_text("import array\nx: array.array[int]\n")
    monkeypatch.setattr(RT, "RUNTIME", runtime)
    monkeypatch.setattr(sys, "prefix", str(runtime))

    class Distribution:
        version = RT.PINS["flashinfer-python"]

        def locate_file(self, name):
            assert name == "flashinfer/comm/fd_exchange.py"
            return target

    monkeypatch.setattr(
        RT.importlib.metadata,
        "distribution",
        create_autospec(RT.importlib.metadata.distribution, return_value=Distribution()),
    )
    record = RT.patch_flashinfer()
    assert record["changed"] and record["before_sha256"] != record["after_sha256"]
    assert RT.patch_flashinfer() == record
    assert json.loads((runtime / "flashinfer_patch.json").read_text()) == record


@pytest.mark.parametrize(
    "version,compat,passes",
    [
        ("580.159.04", False, True),
        ("570.172.08", True, True),
        ("535.261.03", True, True),
        ("550.163.01", True, False),
        ("570.172.08", False, False),
        ("595.1.0", True, False),
        ("595.1.0", False, True),
    ],
)
def test_cuda_driver_gate(version, compat, passes, tmp_path, monkeypatch):
    directory = tmp_path / "compat"
    directory.mkdir()
    monkeypatch.setattr(RT, "COMPAT_DIR", directory)
    if compat:
        (directory / "libcuda.so.1").write_text("fixture")
        monkeypatch.setenv("LD_LIBRARY_PATH", str(directory))
    else:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(RT.shutil, "which", create_autospec(RT.shutil.which, return_value="smi"))
    monkeypatch.setattr(
        RT.platform,
        "libc_ver",
        create_autospec(RT.platform.libc_ver, return_value=("glibc", "2.35")),
    )
    run = create_autospec(subprocess.run, return_value=SimpleNamespace(stdout=version + "\n"))
    monkeypatch.setattr(RT.subprocess, "run", run)
    if passes:
        assert RT.driver_check()["drivers"] == [version]
        assert run.call_args.kwargs["timeout"] == 30
    else:
        with pytest.raises(RuntimeError, match="CUDA 13"):
            RT.driver_check()


def test_smoke_time_validation_and_pod_build_fence(monkeypatch):
    monkeypatch.setattr(RT.time, "time", create_autospec(RT.time.time, return_value=5000.0))
    assert RT.remaining_smoke_seconds({RT.SMOKE_START_ENV: "4000"}, smoke=True) == 2600
    with pytest.raises(RT.RuntimeFence):
        RT.remaining_smoke_seconds({RT.SMOKE_START_ENV: "1000"}, smoke=True)
    with pytest.raises(RuntimeError, match="invalid"):
        RT.remaining_smoke_seconds({RT.SMOKE_START_ENV: "nan"}, smoke=True)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(RuntimeError, match="owned RUNPOD"):
        RT.main(["--build", "--pod-id", "other", "--", "--mode", "smoke"])


@pytest.mark.parametrize(
    "interruption", [subprocess.TimeoutExpired("fixture", 1), RT.RuntimeStop(15)]
)
def test_real_bounded_runner_reaps_own_group(interruption, monkeypatch):
    class Child:
        pid = 12345

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise interruption
            return 0

    child = Child()
    popen = create_autospec(subprocess.Popen, return_value=child)
    monkeypatch.setattr(RT.subprocess, "Popen", popen)
    kill = create_autospec(RT.os.killpg)
    monkeypatch.setattr(RT.os, "killpg", kill)
    monkeypatch.setattr(RT.time, "time", create_autospec(RT.time.time, return_value=1000.0))
    monkeypatch.setattr(RT.time, "monotonic", create_autospec(RT.time.monotonic, return_value=10.0))
    expected = (
        RT.RuntimeFence if isinstance(interruption, subprocess.TimeoutExpired) else RT.RuntimeStop
    )
    with pytest.raises(expected):
        RT.run_bounded(["fixture"], {RT.SMOKE_START_ENV: "999"}, smoke=True, deadline=20, log=None)
    assert [call.args for call in kill.call_args_list] == [
        (12345, signal.SIGTERM),
        (12345, signal.SIGKILL),
    ]
    assert child.waits == 3
    assert popen.call_args.kwargs["start_new_session"] is True


def test_check_rejects_vm_interpreter_before_imports(monkeypatch, tmp_path):
    monkeypatch.setattr(RT, "RUNTIME", tmp_path / "different-runtime")
    with pytest.raises(RuntimeError, match="wrong interpreter"):
        RT.check_runtime()


def test_full_runtime_gate_requires_actual_import_closure(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(RT, "RUNTIME", runtime)
    monkeypatch.setattr(sys, "prefix", str(runtime))
    monkeypatch.setenv("RUNPOD_POD_ID", "owned")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    RT.write_receipt(
        runtime.parent / "runtime_owner.json",
        {
            "lock_sha256": RT.sha256(RT.LOCK),
            "python": RT.PYTHON_VERSION,
            "pod_id": "owned",
        },
    )
    source = runtime / "fd_exchange.py"
    source.write_text("from __future__ import annotations\n")
    RT.write_receipt(
        runtime / "flashinfer_patch.json",
        {
            "path": str(source),
            "after_sha256": RT.sha256(source),
        },
    )
    locked = dict(
        RT.re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\;]+)", RT.LOCK.read_text(), RT.re.MULTILINE)
    )
    monkeypatch.setattr(
        RT.importlib.metadata,
        "version",
        create_autospec(RT.importlib.metadata.version, side_effect=locked.__getitem__),
    )
    monkeypatch.setattr(
        RT.platform,
        "python_version",
        create_autospec(RT.platform.python_version, return_value=RT.PYTHON_VERSION),
    )
    monkeypatch.setattr(RT, "driver_check", create_autospec(RT.driver_check, return_value={}))

    class Tensor:
        def __matmul__(self, other):
            return self

        def sum(self):
            return self

        def item(self):
            return 8

    class Torch:
        version = SimpleNamespace(cuda="13.0")
        cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, synchronize=lambda: None
        )

        @staticmethod
        def ones(shape, *, device):
            assert shape == (2, 2) and device == "cuda"
            return Tensor()

    modules = {
        "vllm": SimpleNamespace(LLM=object, SamplingParams=object, TokensPrompt=object),
        "torch": Torch,
        "transformers.models.qwen3.modeling_qwen3": SimpleNamespace(Qwen3ForCausalLM=object),
        "issue2588_run_cell": SimpleNamespace(_run_import_check=lambda: 0),
        **{
            name: object()
            for name in (
                "flashinfer.comm.fd_exchange",
                "accelerate",
                "scipy",
                "matplotlib",
                "datasets",
                "dotenv",
            )
        },
    }
    imports = create_autospec(RT.importlib.import_module, side_effect=modules.__getitem__)
    monkeypatch.setattr(RT.importlib, "import_module", imports)
    record = RT.check_runtime()
    assert record["status"] == "passed" and record["dependency_versions"] == locked
    assert {call.args[0] for call in imports.call_args_list} == set(modules)
    modules["vllm"] = SimpleNamespace(LLM=object, SamplingParams=object)
    with pytest.raises(RuntimeError, match="TokensPrompt"):
        RT.check_runtime()


def test_build_main_uses_cumulative_clock_and_exact_exec(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(RT, "RUNTIME", runtime)
    monkeypatch.setenv("RUNPOD_POD_ID", "own-pod")
    monkeypatch.setattr(RT.shutil, "which", create_autospec(RT.shutil.which, return_value="uv"))
    monkeypatch.setattr(RT, "driver_check", create_autospec(RT.driver_check, return_value={}))
    monkeypatch.setattr(RT.signal, "signal", create_autospec(RT.signal.signal))
    runner = create_autospec(RT.run_bounded)
    monkeypatch.setattr(RT, "run_bounded", runner)

    class ExecCalled(Exception):
        pass

    execute = create_autospec(RT.os.execve, side_effect=ExecCalled)
    monkeypatch.setattr(RT.os, "execve", execute)
    with pytest.raises(ExecCalled):
        RT.main(
            [
                "--build",
                "--pod-id",
                "own-pod",
                "--bootstrap-preflight-evidence",
                "canonical step10 passed log fixture",
                "--",
                "--mode",
                "smoke",
                "--min-disk-gb",
                "80",
                "--per-pod-quota-gb",
                "200",
            ]
        )
    assert runner.call_count == 6
    assert runner.call_args_list[0].args[0] == ["findmnt", "-T", str(runtime.parent)]
    assert "explore_persona_space.orchestrate.preflight" in runner.call_args_list[1].args[0]
    env = runner.call_args.args[1]
    assert env["UV_NO_SYNC"] == "1" and float(env[RT.SMOKE_START_ENV]) > 0
    assert execute.call_args.args[0] == str(runtime / "bin/python")
    owner = json.loads((runtime.parent / "runtime_owner.json").read_text())
    assert owner["pod_id"] == "own-pod" and owner["lock_sha256"] == RT.sha256(RT.LOCK)
