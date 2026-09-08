"""Run the actual chat wrapper, mocking only subprocess and process-group boundaries."""

from __future__ import annotations

import importlib.util
import io
import json
import signal
import subprocess
import sys
from http.client import HTTPResponse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "issue2588_chat_dispatch", ROOT / "scripts" / "issue2588_chat_dispatch.py"
)
D = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D
SPEC.loader.exec_module(D)


def _args(tmp_path: Path, mode: str = "smoke") -> list[str]:
    """Use isolated output/PID/sentinel paths and an explicit transfer sizing fixture."""
    transfer = tmp_path / "transfer.json"
    transfer.write_text(
        json.dumps(
            {
                "surface": "generic",
                "run_id": D.RUN_ID,
                "transfers": {
                    phase: {
                        "bytes": 100,
                        "bytes_per_s": 10,
                        "retry_calls": 2,
                        "basis": "fixture: 100 bytes at 10 bytes/s; two sequential retry envelopes",
                    }
                    for phase in D.TRANSFER_PHASES
                },
            }
        )
    )
    return [
        "--mode",
        mode,
        "--out-root",
        str(tmp_path / "out"),
        "--pid-file",
        str(tmp_path / "worker.pid"),
        "--sentinel-dir",
        str(tmp_path / "sentinels"),
        "--transfer-plan",
        str(transfer),
        "--min-disk-gb",
        "10",
        "--per-pod-quota-gb",
        "130",
    ]


@pytest.fixture
def process_boundary(monkeypatch):
    """Autospec subprocesses; preserve real wrapper logic, files, locks and reports."""
    launched, killed = [], []
    config = {
        "fail": None,
        "pilot_record": True,
        "full_fit_sentinel": False,
        "timeout": None,
        "signal": None,
    }

    def launch(argv, **kwargs):
        """Provide the external child's filesystem outputs and exit status."""
        child = create_autospec(subprocess.Popen, instance=True)
        child.pid = 800_000 + len(launched)
        child.returncode = 0
        if "--phase" in argv:
            phase = argv[argv.index("--phase") + 1]
            cell = argv[argv.index("--cell") + 1]
            out = Path(argv[argv.index("--out-root") + 1]) / "generic" / D.RUN_ID
            leaf = out / ("smoke_cap_long" if "--smoke" in argv else "cells_cap_long") / cell
            leaf.mkdir(parents=True, exist_ok=True)
            identity = {
                "surface": "generic",
                "run_id": D.RUN_ID,
                "cell": cell,
                "smoke": "--smoke" in argv,
            }
            (leaf / "run_identity.json").write_text(json.dumps(identity))
            if config["fail"] == (cell, phase):
                child.returncode = 23
                print(
                    "child failure detail [phase=done] must not terminate the main log",
                    file=kwargs["stdout"],
                    flush=True,
                )
            if "--fit-max-units" in argv:
                child.returncode = D.RC_PILOT_PAUSE
                if config["pilot_record"]:
                    (leaf / "fits").mkdir(exist_ok=True)
                    pilot = {
                        "status": "pilot_complete",
                        "phase_complete": False,
                        "completed_units": 1,
                        "total_layer_units": 19,
                        "unit_elapsed_s": 12.5,
                        "n": {"tr": 10000, "val": 400, "te": 1000},
                        "d": 4096,
                        "identity": identity,
                        "layer": 0,
                        "input_position": "prompt_last",
                    }
                    (leaf / "fits" / "fit_pilot.json").write_text(json.dumps(pilot))
                    (leaf / "fits" / "percell_prompt_last_L00.json").write_text(json.dumps(pilot))
                if config["full_fit_sentinel"]:
                    (leaf / "phase_done").mkdir(exist_ok=True)
                    (leaf / "phase_done" / "fits.json").write_text("{}")
        launched.append((tuple(argv), kwargs, child))
        child.wait.return_value = child.returncode
        child.poll.return_value = child.returncode
        if config["timeout"] is not None:
            timed_out = False

            def wait(timeout=None):
                """Make the external wait consume its tiny deadline before timing out."""
                nonlocal timed_out
                if not timed_out:
                    timed_out = True
                    assert timeout is not None and timeout < 0.5
                    D.time.sleep(timeout + 0.001)
                    raise subprocess.TimeoutExpired(argv, timeout)
                return -15

            child.wait.side_effect = wait
        if config["signal"] is not None and "--phase" in argv and phase == "gen":
            child.wait.side_effect = [D.WorkloadSignal(config["signal"]), -15]
        return child

    def run(argv, **kwargs):
        """Answer only external GPU observations; other wrapper commands use Popen."""
        assert argv[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            argv, 0, "0\n" if "--query-gpu=memory.used" in argv else "", ""
        )

    def killpg(pgid, sig):
        """The fake child groups have already exited unless a timeout test says otherwise."""
        killed.append((pgid, sig))
        raise ProcessLookupError(pgid)

    # A facade avoids mutating subprocess globally: real git provenance can still run.
    boundary = SimpleNamespace(
        Popen=create_autospec(subprocess.Popen, side_effect=launch),
        run=create_autospec(subprocess.run, side_effect=run),
        TimeoutExpired=subprocess.TimeoutExpired,
        DEVNULL=subprocess.DEVNULL,
        STDOUT=subprocess.STDOUT,
    )
    monkeypatch.setattr(D, "subprocess", boundary)
    monkeypatch.setattr(D.os, "killpg", create_autospec(D.os.killpg, side_effect=killpg))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "30")
    monkeypatch.delenv("EPS2588_SMOKE_STARTED_AT", raising=False)
    return SimpleNamespace(launched=launched, killed=killed, config=config, boundary=boundary)


def _phase_pairs(boundary) -> list[tuple[str, str]]:
    """Read the actually launched commands, not a separate planner mock."""
    return [
        (argv[argv.index("--cell") + 1], argv[argv.index("--phase") + 1])
        for argv, _, _ in boundary.launched
        if "--phase" in argv
    ]


def _report(tmp_path, mode):
    """Read the single fresh wrapper report created by this invocation."""
    reports = list(
        (tmp_path / "out" / "generic" / D.RUN_ID / "dispatch" / mode).glob("*/report.json")
    )
    assert len(reports) == 1
    return json.loads(reports[0].read_text()), reports[0].parent


def test_smoke_real_main_both_arms_no_fits_and_fresh_children(tmp_path, process_boundary):
    """Smoke's executed path keeps real caps and persists each arm before moving on."""
    assert D.main(_args(tmp_path)) == 0
    phases = ("prologue", "stage", "gen", "parse", "upload-raw", "capture", "upload-capture")
    assert _phase_pairs(process_boundary) == [(D.CELLS[0], "stage-runtime")] + [
        (c, p) for c in D.CELLS for p in phases
    ]
    for argv, kwargs, child in process_boundary.launched:
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["EPS_CAP_PROFILE"] == "long"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "5"
        assert kwargs["env"]["HF_HUB_CACHE"].endswith(f"generic/{D.RUN_ID}/hf_cache")
        assert kwargs["env"]["EPM_HF_FILECOUNT_FALLBACK"] == "0"
        assert kwargs["env"]["PYTHONPATH"].split(D.os.pathsep)[0] == str(ROOT / "src")
        assert "--smoke" in argv if "--phase" in argv else True
        assert child.wait.call_count == 1
        if "--phase" in argv and argv[argv.index("--phase") + 1] in D.OFFLINE_PHASES:
            assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
            assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert process_boundary.launched[0][0][-2:] == (str(D.RUNTIME), "--check")
    assert any("--transfer-check" in argv for argv, _, _ in process_boundary.launched)
    report, directory = _report(tmp_path, "smoke")
    assert report["status"] == "phase_complete" and report["experiment_complete"] is False
    assert len(report["steps"]) == len(process_boundary.launched)
    assert int((tmp_path / "worker.pid").read_text()) == D.os.getpid()
    assert (directory / "dispatch.log").read_text().count("[phase=done]") == 1
    envelope = json.loads(next((tmp_path / "sentinels").glob("*.json")).read_text())
    assert envelope["kind"] == "epm:smoke-result"
    assert envelope["sentinel_schema_version"] == 1 and envelope["version"] == 1
    assert envelope["blocks_pipeline"] is False


def test_capture_real_main_preserves_production_scope(tmp_path, process_boundary):
    """Production capture never dispatches a hidden fit, benchmark, or smoke phase."""
    assert D.main(_args(tmp_path, "capture")) == 0
    assert len(_phase_pairs(process_boundary)) == 15
    assert all("--smoke" not in argv for argv, _, _ in process_boundary.launched)
    envelope = json.loads(next((tmp_path / "sentinels").glob("*.json")).read_text())
    assert envelope["kind"] == "epm:progress" and envelope["gate"] == "phase"


def test_dispatched_capture_uses_inherited_gpu_bfloat16_loader(
    tmp_path, process_boundary, monkeypatch
):
    """Feed actual dispatched arguments through the real parent loader, with no weights."""
    import torch
    import transformers

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    inherited = importlib.import_module("issue2330_qwen35_generate_capture")
    config = transformers.Qwen3Config()
    config_load = create_autospec(transformers.AutoConfig.from_pretrained, return_value=config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_load)
    # The model-loading and hardware-discovery boundaries are mocked; dtype/device
    # routing, native-config selection and the inherited loader body are real.
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    model.config = config
    model_load = create_autospec(
        transformers.AutoModelForCausalLM.from_pretrained, return_value=model
    )
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_load)
    monkeypatch.setattr(
        torch.cuda, "device_count", create_autospec(torch.cuda.device_count, return_value=1)
    )
    assert D.main(_args(tmp_path, "capture")) == 0
    snapshot = str(tmp_path / "pinned-snapshot")
    captures = [
        argv
        for argv, _, _ in process_boundary.launched
        if "--phase" in argv and argv[argv.index("--phase") + 1] == "capture"
    ]
    assert len(captures) == 2
    for argv in captures:
        device = argv[argv.index("--device") + 1]
        assert inherited._load_capture_model(snapshot, device, "bfloat16") is model
        assert model_load.call_args.args == (snapshot,)
        assert model_load.call_args.kwargs["device_map"] == {"": 0}
        assert model_load.call_args.kwargs["dtype"] is torch.bfloat16
    assert model_load.call_count == 2


def test_fit_pilot_real_main_requires_checkpoint_and_pauses(tmp_path, process_boundary):
    """The actual pilot unit has rc7, a validated checkpoint, and a durability upload."""
    assert D.main(_args(tmp_path, "fit-pilot")) == D.RC_PILOT_PAUSE
    assert _phase_pairs(process_boundary) == [(D.CELLS[0], "stage-runtime")] + [
        (c, p) for c in D.CELLS for p in ("upload-raw", "upload-capture")
    ] + [("q3_8b_a", "fits"), ("q3_8b_a", "upload-partial")]
    report, directory = _report(tmp_path, "fit-pilot")
    assert report["status"] == "pilot_paused"
    assert "[phase=done]" not in (directory / "dispatch.log").read_text()
    sentinel = json.loads(next((tmp_path / "sentinels").glob("*.json")).read_text())
    assert sentinel["gate"] == "fit_pilot" and sentinel["blocks_pipeline"] is True


@pytest.mark.parametrize("problem", ["pilot_record", "full_fit_sentinel"])
def test_pilot_invalid_output_never_accepted(tmp_path, process_boundary, problem):
    """Neither rc7 alone nor a contradictory fit-done file proves a valid pilot."""
    process_boundary.config[problem] = problem == "full_fit_sentinel"
    assert D.main(_args(tmp_path, "fit-pilot")) == 1
    report, _ = _report(tmp_path, "fit-pilot")
    assert report["status"] == "halted"
    assert report["reason_chain"]


def test_fits_real_main_verifies_both_capture_arms_first(tmp_path, process_boundary):
    """A resumed fit enters the driver's verification before either arm's fit."""
    assert D.main(_args(tmp_path, "fits")) == 0
    assert _phase_pairs(process_boundary) == [(D.CELLS[0], "stage-runtime")] + [
        (c, p) for c in D.CELLS for p in ("upload-raw", "upload-capture")
    ] + [(c, p) for c in D.CELLS for p in ("fits", "upload-fits")]
    assert all("--fit-max-units" not in argv for argv, _, _ in process_boundary.launched)


def test_child_failure_preserves_rc_tail_and_no_false_done(tmp_path, process_boundary):
    """The child exit status halts further compute and its logs cannot fake completion."""
    process_boundary.config["fail"] = ("q3_8b_a", "capture")
    assert D.main(_args(tmp_path)) == 23
    assert _phase_pairs(process_boundary)[-1] == ("q3_8b_a", "upload-partial")
    assert all(c != "q3_8b_b" for c, _ in _phase_pairs(process_boundary))
    report, directory = _report(tmp_path, "smoke")
    text = (directory / "dispatch.log").read_text()
    assert "[phase=done]" not in text and "child failure detail" in text
    assert report["rc"] == 23 and report["durability_errors"] == []
    assert any(signal_value == signal.SIGTERM for _, signal_value in process_boundary.killed)


def test_list_commands_has_no_files_processes_or_environment_mutation(
    tmp_path, process_boundary, capsys
):
    """The CLI composition seam is safe even against a prospective production path."""
    args = [*_args(tmp_path, "fits"), "--list-commands"]
    env_before = dict(D.os.environ)
    assert D.main(args) == 0
    assert process_boundary.launched == []
    assert not (tmp_path / "out").exists() and not (tmp_path / "worker.pid").exists()
    assert dict(D.os.environ) == env_before
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["commands"]) == 13
    assert plan["run_id"] == D.RUN_ID


@pytest.mark.parametrize(
    "extra",
    [["--surface", "full"], ["--run-id", "other"], ["--cell", "q3_8b_a"], ["--mode", "gpqa"]],
)
def test_out_of_scope_cli_rejected_without_writes(tmp_path, process_boundary, extra):
    """Argparse refuses attempts to expand the fixed run and panel."""
    with pytest.raises(SystemExit):
        D.main(_args(tmp_path) + extra)
    assert process_boundary.launched == [] and not (tmp_path / "out").exists()


def test_transfer_timeout_covers_retries_and_rejects_missing_basis(tmp_path):
    """The launched transfer bounds derive from real-sized input records."""
    args = D.build_parser().parse_args(_args(tmp_path))
    limits = D.transfer_limits(args, D.build_steps(args), 1800)
    assert limits["q3_8b_a:stage"]["retry_exposure_s"] == 3600
    assert limits["q3_8b_a:stage"]["timeout_s"] == 3621
    obj = json.loads(args.transfer_plan.read_text())
    del obj["transfers"]["upload-partial"]
    args.transfer_plan.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="upload-partial"):
        D.transfer_limits(args, D.build_steps(args), 1800)


def test_smoke_fence_bounds_model_staging_and_upload_tail_is_separate(tmp_path, process_boundary):
    """An expired work budget blocks stage/gen yet permits partial persistence."""
    args = D.build_parser().parse_args(_args(tmp_path))
    limits = D.transfer_limits(args, D.build_steps(args), 30)
    runner = D.Runner(args, D.child_environment(args), limits)
    try:
        runner.initial_work_s = 3600
        with pytest.raises(D.PhaseFailure) as failure:
            runner.run(D.cell_step(args, D.CELLS[0], "stage"))
        assert failure.value.rc == D.RC_WORK_FENCE
        runner.run(D.cell_step(args, D.CELLS[0], "upload-partial"), durability_tail=True)
        assert len(process_boundary.launched) == 1
        assert runner.records[0]["timeout_basis"] == "transfer_plan"
    finally:
        runner.main_log.close()


def test_transfer_expiry_reaps_child_and_records_exit(tmp_path, process_boundary):
    """A no-output stalled child is stopped by a whole-phase deadline, never file presence."""
    args = D.build_parser().parse_args(_args(tmp_path, "capture"))
    step = D.cell_step(args, D.CELLS[0], "upload-partial")
    runner = D.Runner(args, D.child_environment(args), {step.key: {"timeout_s": 0.000001}})
    process_boundary.config["timeout"] = 0.000001
    try:
        with pytest.raises(D.PhaseFailure) as failure:
            runner.run(step, durability_tail=True)
        assert failure.value.rc == D.RC_TRANSFER_TIMEOUT
        assert runner.records[0]["timed_out"] is True and runner.records[0]["rc"] == -15
        assert process_boundary.killed
    finally:
        runner.main_log.close()


def test_relaunch_logs_are_unique_and_pid_is_rewritten(tmp_path, process_boundary):
    """A rerun re-enters core resume gates but never appends to an older attempt log."""
    argv = _args(tmp_path)
    assert D.main(argv) == 0
    (tmp_path / "worker.pid").write_text("123\n")
    assert D.main(argv) == 0
    base = tmp_path / "out" / "generic" / D.RUN_ID / "dispatch" / "smoke"
    logs = list(base.glob("*/dispatch.log"))
    assert len(logs) == 2 and all(p.read_text().count("[phase=done]") == 1 for p in logs)
    assert int((tmp_path / "worker.pid").read_text()) == D.os.getpid()


def test_import_check_executes_real_argcheck_without_output_tree(tmp_path):
    """The wrapper imports and checks its argparse attributes without frameworks."""
    assert D.main(["--import-check", "--out-root", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()


def test_upload_timeout_uses_actual_tree_and_metadata_allowance(tmp_path, process_boundary):
    """Uploads can bind their timeout after outputs exist, without predicting raw-text size."""
    args = D.build_parser().parse_args(_args(tmp_path, "capture"))
    plan = json.loads(args.transfer_plan.read_text())
    plan["transfers"]["upload-partial"] = {
        "source": "local_cell_tree",
        "bytes_per_s": 10,
        "retry_calls": 2,
        "basis": "actual local files at expected 10 bytes/s; two retry calls",
    }
    args.transfer_plan.write_text(json.dumps(plan))
    limits = D.transfer_limits(args, D.build_steps(args), 30)
    assert limits["q3_8b_a:upload-partial"]["timeout_s"] is None
    runner = D.Runner(args, D.child_environment(args), limits)
    try:
        cell = runner.root / "cells_cap_long" / D.CELLS[0]
        cell.mkdir(parents=True)
        (cell / "raw.json").write_bytes(b"0123456789")
        runner.run(D.cell_step(args, D.CELLS[0], "upload-partial"))
        limit = runner.limits["q3_8b_a:upload-partial"]
        assert limit["observed_bytes"] == 10 and limit["observed_files"] == 1
        assert limit["bytes"] == 10 + 2 * D.UPLOAD_METADATA_BYTES_PER_FILE
        assert limit["timeout_s"] == 2 * limit["bytes"] / 10 + 61
    finally:
        runner.main_log.close()


def test_external_stop_routes_partial_upload_and_preserves_status(tmp_path, process_boundary):
    """A stopped wrapper cannot abandon its separately isolated GPU child."""
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    process_boundary.config["signal"] = signal.SIGTERM
    assert D.main(_args(tmp_path)) == 143
    assert _phase_pairs(process_boundary)[-1] == (D.CELLS[0], "upload-partial")
    report, _ = _report(tmp_path, "smoke")
    assert any(r["phase"] == "gen" for r in report["steps"])
    assert all(signal.getsignal(s) is handler for s, handler in handlers.items())


def test_active_smoke_staging_uses_safety_rc_not_transfer_rc(tmp_path, process_boundary):
    """The explicit one-hour pilot fence can stop even a retry-eligible model download."""
    args = D.build_parser().parse_args(_args(tmp_path))
    limits = D.transfer_limits(args, D.build_steps(args), 30)
    runner = D.Runner(args, D.child_environment(args), limits)
    runner.initial_work_s = 3599.8 - (D.time.monotonic() - runner.started)
    process_boundary.config["timeout"] = 0.000001
    try:
        with pytest.raises(D.PhaseFailure) as failure:
            runner.run(D.cell_step(args, D.CELLS[0], "stage"))
        assert failure.value.rc == D.RC_WORK_FENCE
        assert runner.records[0]["timeout_basis"] == "smoke_work_fence"
    finally:
        runner.main_log.close()


def test_empty_compute_app_rows_cannot_hide_device_memory(process_boundary):
    """The actual hygiene gate catches foreign allocations invisible to the PID namespace."""

    def held_gpu(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, "4096\n" if "--query-gpu=memory.used" in argv else "", ""
        )

    process_boundary.boundary.run.side_effect = held_gpu
    with pytest.raises(RuntimeError, match="4096"):
        D.wait_gpu_free("5", timeout_s=0)


def test_multi_gpu_environment_rejected(tmp_path, monkeypatch):
    """Scope requires one GPU; it never silently narrows another allocation."""
    args = D.build_parser().parse_args(_args(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="exactly one"):
        D.child_environment(args)


def test_prior_clock_is_not_a_production_bypass(tmp_path, process_boundary):
    with pytest.raises(SystemExit):
        D.main([*_args(tmp_path, "capture"), "--smoke-prior-report", str(tmp_path / "prior.json")])
    assert process_boundary.launched == []


def test_v1_identity_is_refused_by_new_wrapper(tmp_path, process_boundary):
    with pytest.raises(SystemExit):
        D.main([*_args(tmp_path), "--run-id", "qwen3-chat-v1"])
    assert process_boundary.launched == []


def test_runtime_and_range_checks_are_required_with_manual_general_preflight(
    tmp_path, process_boundary
):
    """A general-preflight bypass never bypasses strict runtime or transfer checks."""
    assert (
        D.main(
            [
                *_args(tmp_path, "capture"),
                "--skip-preflight",
                "--preflight-evidence",
                "manual disk/account check receipt",
            ]
        )
        == 0
    )
    commands = [argv for argv, _, _ in process_boundary.launched]
    assert commands[0][-2:] == (str(D.RUNTIME), "--check")
    assert any("--transfer-check" in argv for argv in commands)
    assert all("explore_persona_space.orchestrate.preflight" not in argv for argv in commands)


def test_smoke_clock_inherits_runtime_construction(tmp_path, process_boundary, monkeypatch):
    """The one-hour fence starts before runtime construction, not at the first GPU child."""
    monkeypatch.setenv("EPS2588_SMOKE_STARTED_AT", str(D.time.time() - 3500))
    assert D.main(_args(tmp_path)) == 0
    report, _ = _report(tmp_path, "smoke")
    assert report["inherited_runtime_work_s"] >= 3500
    assert report["work_elapsed_s"] >= report["inherited_runtime_work_s"]
    assert report["steps"][0]["timeout_s"] <= 100


def test_expired_runtime_clock_never_starts_more_work(tmp_path, process_boundary, monkeypatch):
    """An exhausted builder budget emits the work-fence gate without relaunching compute."""
    monkeypatch.setenv("EPS2588_SMOKE_STARTED_AT", str(D.time.time() - 3601))
    assert D.main(_args(tmp_path)) == D.RC_WORK_FENCE
    assert process_boundary.launched == []
    report, _ = _report(tmp_path, "smoke")
    assert report["status"] == "halted" and report["work_elapsed_s"] >= 3601


@pytest.mark.parametrize("epoch", ["nan", "inf", "-1", "1e50"])
def test_invalid_runtime_clock_fails_loudly(tmp_path, process_boundary, monkeypatch, epoch):
    """Malformed or future epochs cannot expand the smoke allowance."""
    monkeypatch.setenv("EPS2588_SMOKE_STARTED_AT", epoch)
    with pytest.raises(ValueError, match="nonfuture epoch"):
        D.main(_args(tmp_path))
    assert process_boundary.launched == []


@pytest.mark.parametrize("problem", [None, "empty", "short", "ignored-range"])
def test_real_range_probe_requires_exact_bytes(monkeypatch, capsys, problem):
    """Only the HTTP boundary is mocked; the actual CLI validates response and byte count."""
    response = create_autospec(HTTPResponse, instance=True)
    response.status = 200 if problem == "ignored-range" else 206
    response.headers = {"Content-Range": f"bytes 0-{D.MODEL_PROBE_BYTES - 1}/4000000000"}
    body = io.BytesIO(
        b"" if problem == "empty" else b"a" * (10 if problem == "short" else D.MODEL_PROBE_BYTES)
    )
    response.read.side_effect = body.read
    response.__enter__.return_value = response
    opener = create_autospec(D.urllib.request.urlopen, return_value=response)
    monkeypatch.setattr(D.urllib.request, "urlopen", opener)
    if problem is None:
        assert D.main(["--transfer-check"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "verified" and report["bytes"] == D.MODEL_PROBE_BYTES
        assert report["elapsed_s"] > 0
    else:
        with pytest.raises(RuntimeError, match="pinned model probe"):
            D.main(["--transfer-check"])
    assert opener.call_count == 1
    request = opener.call_args.args[0]
    assert request.full_url == D.MODEL_PROBE_URL
    assert request.headers["Range"] == f"bytes=0-{D.MODEL_PROBE_BYTES - 1}"
    assert body.tell() <= D.MODEL_PROBE_BYTES


@pytest.mark.parametrize(
    "phase, expected",
    [
        ("upload-raw", (1, 0, 10)),
        ("upload-capture", (1, 0, 10)),
        ("upload-partial", (3, 3, 20)),
        ("upload-fits", (0, 5, 18)),
    ],
)
def test_upload_operation_counts_follow_groups_and_metadata(tmp_path, phase, expected):
    """Shard count affects byte sizing but not the number of canonical retry envelopes."""
    relative_files = [
        "raw_completions/train_10k/chunk0000.json",
        "parsed/train_10k.jsonl",
        "parsed/val_400_capture_drops.json",
        "capture_input_validation.json",
        "run_identity.json",
        "phase_done/gen.json",
        "fits/dropped_row_ids.json",
        "fits/fit_pilot.json",
        "fits/percell_prompt_last_L00.json",
        "fits/fits_prompt_last.json",
        "fits/perrow_prompt_last.json",
    ]
    # Partial retains separate parsed-file groups; raw/capture now batch the phase.
    if phase in ("upload-raw", "upload-partial"):
        relative_files.remove("parsed/val_400_capture_drops.json")
    else:
        relative_files.remove("parsed/train_10k.jsonl")
    relative_files.extend(f"capture/train_10k/L00/shard{k:03d}.npz" for k in range(30))
    for relative in relative_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture child output")
    operations = D.local_upload_operations(tmp_path, D.CELLS[0], phase)
    assert (
        operations["bulk_groups"],
        operations["single_files"],
        operations["retry_calls"],
    ) == expected
    if phase in ("upload-raw", "upload-capture"):
        assert operations["verified_receipts"] == 1
        assert operations["completion_checkpoints"] == 1


def test_actual_wrapper_resolves_dynamic_upload_retry_envelopes(tmp_path, process_boundary):
    """The production dispatch path resolves both actual bytes and actual upload operations."""
    argv = _args(tmp_path)
    plan_path = tmp_path / "transfer.json"
    plan = json.loads(plan_path.read_text())
    for phase, entry in plan["transfers"].items():
        if phase.startswith("upload-"):
            del entry["bytes"]
            del entry["retry_calls"]
            entry.update(source="local_cell_tree", retry_calls_source="local_upload_operations")
    plan_path.write_text(json.dumps(plan))
    assert D.main(argv) == 0
    report, _ = _report(tmp_path, "smoke")
    for step in report["steps"]:
        if step["phase"].startswith("upload-"):
            limit = report["transfer_limits"][f"{step['cell']}:{step['phase']}"]
            assert limit["observed_files"] > 0
            assert limit["retry_calls"] == limit["observed_upload_operations"]["retry_calls"]
            assert limit["retry_exposure_s"] == limit["retry_calls"] * 30
            assert step["timeout_s"] == 2 * limit["bytes"] / 10 + limit["retry_exposure_s"] + 1
            if step["phase"] in ("upload-raw", "upload-capture"):
                assert limit["retry_calls"] == 10


def test_smoke_gpu_drain_cannot_outlive_work_fence(tmp_path, process_boundary):
    """Foreign memory cannot turn a short remaining pilot budget into a 180-second drain."""
    args = D.build_parser().parse_args(_args(tmp_path))
    runner = D.Runner(args, D.child_environment(args), {})
    runner.initial_work_s = 3599.8 - (D.time.monotonic() - runner.started)

    def held_gpu(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, "4096\n" if "--query-gpu=memory.used" in argv else "", ""
        )

    process_boundary.boundary.run.side_effect = held_gpu
    try:
        with pytest.raises(D.PhaseFailure) as failure:
            runner.run(D.cell_step(args, D.CELLS[0], "gen"))
        assert failure.value.rc == D.RC_WORK_FENCE
        assert process_boundary.launched == []
    finally:
        runner.main_log.close()


@pytest.mark.parametrize("mode", ["smoke", "capture", "fit-pilot", "fits"])
def test_committed_transfer_plan_covers_actual_wrapper_modes(tmp_path, mode):
    """Validate the checked-in transfer-plan integration without any subprocess or network."""
    args = D.build_parser().parse_args(["--mode", mode, "--out-root", str(tmp_path / "out")])
    args.transfer_plan = ROOT / "configs" / "issue2588_chat_transfers.json"
    steps = D.build_steps(args)
    limits = D.transfer_limits(args, steps, 1800)
    assert limits[f"{D.CELLS[0]}:stage-runtime"]["bytes"] == 16_397_432_693
    assert limits[f"{D.CELLS[0]}:stage-runtime"]["retry_calls"] == 2
    assert limits["driver:transfer_check"]["bytes"] == D.MODEL_PROBE_BYTES
    for key, value in limits.items():
        if ":upload-" in key:
            assert value["source"] == "local_cell_tree"
            assert value["retry_calls_source"] == "local_upload_operations"
            assert value["timeout_s"] is None
    assert not (tmp_path / "out").exists()
