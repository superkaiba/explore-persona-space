"""Exercise high-rate collection bodies with real Inspect/Docker and a fixture-only ModelAPI."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import RegistryInfo, registry_tag
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, Model, ModelOutput, ModelUsage
from inspect_ai.model._providers.mockllm import MockLLM
from omegaconf import OmegaConf

from scripts import context_risk_followup as inherited
from scripts import context_risk_highrate_collect as collect
from scripts import context_risk_highrate_design as design


def fixture_phase(tmp_path, monkeypatch, *, phase="fresh", n_tasks=1):
    """Replace only phase authorization/input files; production collection bodies are unchanged."""
    root = tmp_path / "fixture_only"
    (root / "manifests").mkdir(parents=True)
    rows = []
    for number in range(n_tasks):
        for condition in ("original", "oneoff", "conflicting"):
            rows.append(
                {
                    "task_id": f"fixture_{number}",
                    "condition": condition,
                    "entry_point": f"fixture_f_{number}",
                    "prompt": f'def fixture_f_{number}():\n    """Return1."""',
                    "test": "def check(candidate):\n    assert candidate() == 1\n",
                    "partition": "fixture_only",
                    "dataset_revision": "fixture_only",
                }
            )
    roles = {r["task_id"]: "probe_training" for r in rows}
    manifests = {}
    for which in ("screen", "fresh"):
        payload = design.make_rows(rows, which, roles if which == "fresh" else None)
        path = root / f"manifests/{which}_B.jsonl"
        inherited._write_jsonl_atomic(path, payload)
        manifests[which] = path
    frozen = {"frozen_unix": time.time() - 10, "fixture_only": True}
    selected = {"selected_unix": time.time() - 5, "fixture_only": True}
    inherited._write_json_atomic(root / "manifests/screen_freeze.json", frozen)
    inherited._write_json_atomic(root / "selection.json", selected)

    def load_phase(root_arg, phase_arg):
        assert Path(root_arg) == root
        return (
            manifests[phase_arg],
            frozen if phase_arg == "screen" else selected,
            2 if phase_arg == "screen" else 4,
        )

    monkeypatch.setattr(
        design, "load_phase", create_autospec(design.load_phase, side_effect=load_phase)
    )
    review = root / "fixture_review.json"
    inherited._write_json_atomic(
        review,
        {
            "verdict": "PASS",
            "reviewer": "test_fixture_only",
            "files_sha256": design.source_hashes(),
        },
    )
    cfg = OmegaConf.create(
        {
            "operation": "run",
            "root": str(root),
            "phase": phase,
            "arm": "B",
            "model": collect.MODEL,
            "base_url": "http://fixture.invalid/v1",
            "max_connections": 16,
            "review": str(review),
            "resume_log": None,
            "pilot_limit": None,
            "source_manifest": (
                "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
                "data/impossible_livecodebench/public_development_manifest.jsonl"
            ),
            "plan": str(design.DESIGN / "plan.md"),
        }
    )
    return root, cfg, manifests


def install_fixture_boundaries(monkeypatch, *, behavior="success"):
    """Fake the external token/model boundary while actual task/solver/scorer/Docker code runs."""
    seen = []

    def output(input, tools, tool_choice, config):
        del tools, tool_choice
        attempt = 1 + sum(m.role == "assistant" for m in input)
        text = input[0].text
        name = re.search(r"def (fixture_f_\d+)\(\)", text).group(1)
        seen.append((text, attempt, config.seed))
        result = ModelOutput.from_content(
            collect.MODEL.removeprefix("openai-api/local/"),
            f"```python\ndef {name}(): return 1\n```",
        )
        result.usage = ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4)
        if behavior == "censored":
            result.choices[0].stop_reason = "max_tokens"
            result.usage = ModelUsage(input_tokens=3, output_tokens=65536, total_tokens=65539)
        elif behavior == "failure":
            result = ModelOutput.from_content(
                collect.MODEL.removeprefix("openai-api/local/"),
                f"```python\ndef {name}(): return 0\n```",
            )
            result.usage = ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4)
        return result

    def evaluated(target, **kwargs):
        api = MockLLM(
            collect.MODEL.removeprefix("openai-api/"),
            base_url=kwargs["model_base_url"],
            custom_outputs=output,
        )
        registry_tag(
            MockLLM,
            api,
            RegistryInfo(type="modelapi", name="openai-api"),
            collect.MODEL.removeprefix("openai-api/"),
            base_url=kwargs["model_base_url"],
        )
        model = Model(api, config=GenerateConfig(), model_args=collect.PROVIDER_ARGS)
        model._explicit_base_url = kwargs["model_base_url"]
        kwargs["model"] = model
        kwargs["display"] = "none"
        return inspect_eval(target, **kwargs)

    class Response:
        def __init__(self, ids):
            self.value = json.dumps({"tokens": ids}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *args):
            return self.value

    def token_response(request, timeout=30):
        assert timeout == 30 and request.full_url == "http://fixture.invalid/tokenize"
        payload = json.loads(request.data)
        assert payload["model"] == collect.MODEL.removeprefix("openai-api/local/")
        return Response([1, len(payload["messages"][0]["content"]), 3])

    monkeypatch.setattr(collect, "eval", create_autospec(inspect_eval, side_effect=evaluated))
    monkeypatch.setattr(
        inherited, "urlopen", create_autospec(inherited.urlopen, side_effect=token_response)
    )
    return seen


def test_fresh_real_body_four_epochs_and_stable_readback(tmp_path, monkeypatch):
    """Run all production collection/readback bodies at four epochs through real Docker."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    seen = install_fixture_boundaries(monkeypatch)
    report = collect.run(cfg)
    assert report["epochs"] == 4 and report["requested_rollouts"] == 12
    assert report["counts"]["success"] == 12 and report["verification_passed"]
    assert len(seen) == 12 and len({s[2] for s in seen}) == 12
    first = collect.verify_report(root, "fresh")
    second = collect.verify_report(root, "fresh")
    assert first == second
    native = read_eval_log(next(iter(report["native_logs_sha256"])))
    assert {s.epoch for s in native.samples} == {1, 2, 3, 4}
    assert native.eval.config.epochs == 4
    assert all(s.error is None for s in native.samples)
    path = root / "fresh_B/rollouts.jsonl"
    original = path.read_text()
    path.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        collect.verify_report(root, "fresh")


def test_generation_cap_is_verified_unknown_not_negative(tmp_path, monkeypatch):
    """Real stopped ModelOutputs produce terminal N rows that remain valid unknown observations."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    seen = install_fixture_boundaries(monkeypatch, behavior="censored")
    report = collect.run(cfg)
    assert len(seen) == 12
    assert report["passed"] and report["counts"]["censored"] == 12
    assert report["counts"]["success"] == report["counts"]["failure"] == 0
    audit = collect.verify_report(root, "fresh")
    assert len(audit["generation_limit_events"]) == 12
    assert all(
        row["scores"]["successful_submission"]["value"] == "N"
        for row in map(json.loads, (root / "fresh_B/rollouts.jsonl").read_text().splitlines())
    )


def test_screen_pilot_resume_keeps_n_without_resampling(tmp_path, monkeypatch):
    """Exact native PreviousTask resume includes32 pilot N rows once and fills four missing rows."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch, phase="screen", n_tasks=6)
    seen = install_fixture_boundaries(monkeypatch, behavior="censored")
    cfg.pilot_limit = 16
    pilot = collect.run(cfg)
    assert pilot["counts"]["censored"] == 32 and len(seen) == 32
    cfg.pilot_limit = None
    cfg.resume_log = next(iter(pilot["native_logs_sha256"]))
    final = collect.run(cfg)
    assert final["counts"]["censored"] == 36
    assert final["requested_rollouts"] == 36 and len(seen) == 36
    assert len({seed for _, _, seed in seen}) == 36
    collect.verify_report(root, "screen")


def test_failed_integrity_persists_raw_and_counts_before_raise(tmp_path, monkeypatch):
    """A bad returned native recipe cannot erase the already collected samples or become PASS."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    install_fixture_boundaries(monkeypatch)
    actual_eval = collect.eval

    def damaged(target, **kwargs):
        logs = actual_eval(target, **kwargs)
        logs[0].eval.metadata = {"unexpected": True}
        return logs

    monkeypatch.setattr(collect, "eval", create_autospec(inspect_eval, side_effect=damaged))
    with pytest.raises(RuntimeError, match="raw errors/counts preserved"):
        collect.run(cfg)
    report = json.loads((root / "fresh_B/run_result.json").read_text())
    assert not report["passed"] and report["counts"]["realized"] == 12
    assert len((root / "fresh_B/rollouts.jsonl").read_text().splitlines()) == 12
    assert report["validation_issues"]
    with pytest.raises(ValueError, match="not verified"):
        collect.verify_report(root, "fresh")


def test_unresolved_native_error_is_saved_unknown_and_resume_rejected(tmp_path, monkeypatch):
    """Native sample errors remain unknown and require diagnosis instead of rerolling a label."""
    from inspect_ai.log import EvalError

    root, cfg, manifests = fixture_phase(tmp_path, monkeypatch)
    install_fixture_boundaries(monkeypatch)
    actual_eval = collect.eval

    def errored(target, **kwargs):
        logs = actual_eval(target, **kwargs)
        logs[0].samples[0].error = EvalError(
            message="fixture unresolved transport",
            traceback="fixture traceback",
            traceback_ansi="fixture traceback",
        )
        return logs

    monkeypatch.setattr(collect, "eval", create_autospec(inspect_eval, side_effect=errored))
    with pytest.raises(RuntimeError):
        collect.run(cfg)
    report = json.loads((root / "fresh_B/run_result.json").read_text())
    assert report["counts"]["censored"] == 1 and not report["passed"]
    raw = [json.loads(line) for line in (root / "fresh_B/rollouts.jsonl").read_text().splitlines()]
    assert sum(row["error"] is not None for row in raw) == 1
    log = read_eval_log(next(iter(report["native_logs_sha256"])))
    log.samples[0].error = EvalError(
        message="fixture error", traceback="fixture", traceback_ansi="fixture"
    )
    with pytest.raises(ValueError, match="requires diagnosis"):
        collect.validate_resume(
            log, collect.load_samples(manifests["fresh"]), report["metadata"], cfg
        )


def test_true_failure_reaches_tenth_submission(tmp_path, monkeypatch):
    """Actual scorer distinguishes completed failure from a generation cap at all four epochs."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    seen = install_fixture_boundaries(monkeypatch, behavior="failure")
    report = collect.run(cfg)
    assert report["counts"]["failure"] == 12 and report["counts"]["censored"] == 0
    assert len(seen) == 120 and max(attempt for _, attempt, _ in seen) == 10
    collect.verify_report(root, "fresh")


def test_tampered_prefix_and_postselection_timestamp_rejected(tmp_path, monkeypatch):
    """Verify prefix hashes and freeze-before-first-request ordering in the real audit body."""
    root, cfg, manifests = fixture_phase(tmp_path, monkeypatch)
    install_fixture_boundaries(monkeypatch)
    report = collect.run(cfg)
    samples = collect.load_samples(manifests["fresh"])
    tokens = json.loads((root / "fresh_B/prefix_tokens.json").read_text())
    tokens["contexts"][0]["token_ids"][0] += 1
    with pytest.raises(ValueError, match="prefix token"):
        collect.validate_prefixes(tokens, samples)
    logs = [read_eval_log(path) for path in report["native_logs_sha256"]]
    audit = collect.audit_native_logs(
        logs, samples, report["metadata"], cfg, pilot=False, not_before=time.time() + 100
    )
    assert not audit["verification_passed"]
    assert any("predates" in row["message"] for row in audit["validation_issues"])


@pytest.mark.parametrize("failure", ["missing", "audit_exception", "readback_exception"])
def test_terminal_validation_failures_leave_raw_counts_and_no_pass(tmp_path, monkeypatch, failure):
    """A missing sample, parser bug or final readback failure can never leave consumable PASS."""
    root, cfg, _ = fixture_phase(tmp_path, monkeypatch)
    install_fixture_boundaries(monkeypatch)
    if failure == "missing":
        actual = collect.eval

        def incomplete(target, **kwargs):
            logs = actual(target, **kwargs)
            logs[0].samples = list(logs[0].samples)[:-1]
            return logs

        monkeypatch.setattr(collect, "eval", create_autospec(inspect_eval, side_effect=incomplete))
    else:
        name = "audit_native_logs" if failure == "audit_exception" else "verify_report"
        original = getattr(collect, name)
        monkeypatch.setattr(
            collect, name, create_autospec(original, side_effect=OSError("fixture read failure"))
        )
    with pytest.raises((RuntimeError, OSError)):
        collect.run(cfg)
    report = json.loads((root / "fresh_B/run_result.json").read_text())
    assert report["passed"] is report["verification_passed"] is False
    assert report["counts"]["planned"] == 12
    assert report["counts"]["realized"] == (11 if failure == "missing" else 12)
    assert report["counts"]["missing"] == (1 if failure == "missing" else 0)
    assert report["validation_issues"]
    with (root / "fresh_B/rollouts.jsonl").open() as handle:
        assert len([json.loads(line) for line in handle]) == report["counts"]["realized"]


def test_supervisor_records_actual_exit_and_drains_group(tmp_path):
    """Execute the owned supervisor body; only the external uv worker is a local fixture."""
    script = Path(collect.__file__).with_name("context_risk_highrate_supervise.sh")
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    uv = tool_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 7\n")
    uv.chmod(0o755)
    import os

    env = {
        **os.environ,
        "PATH": f"{tool_dir}:{os.environ['PATH']}",
        "EPM_CONTEXT_RISK_HIGHRATE_ROOT": str(tmp_path / "out"),
        "EPM_CONTEXT_RISK_LAUNCH_ID": "fixture_exit",
        "EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS": "10",
    }
    result = subprocess.run(
        ["bash", str(script), "fresh"], env=env, text=True, capture_output=True, timeout=20
    )
    assert result.returncode == 7
    receipt = json.loads((tmp_path / "out/fresh_fixture_exit_process.exit.json").read_text())
    assert receipt["mode"] == "fresh" and receipt["exit_code"] == 7
    assert receipt["cleanup"] == "no_live_members"
    assert receipt["supervisor_pid"] != receipt["worker_pid"]
    repeated = subprocess.run(
        ["bash", str(script), "fresh"], env=env, text=True, capture_output=True, timeout=20
    )
    assert repeated.returncode == 2 and "already exists" in repeated.stderr
