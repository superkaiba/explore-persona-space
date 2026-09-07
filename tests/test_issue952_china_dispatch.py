"""Synthetic orchestration tests: never load models or call a network service."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import create_autospec

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "china_dispatch", ROOT / "scripts/issue952_china_dispatch.py"
)
assert SPEC and SPEC.loader
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
CODE_SHA = "a" * 40
INPUT_REV = "b" * 40
DATA_REV = "c" * 40


def args_for(tmp_path: Path):
    """Build the actual public CLI using isolated synthetic directories."""
    return D.build_parser().parse_args(
        [
            "--expected-code-sha",
            CODE_SHA,
            "--input-revision",
            INPUT_REV,
            "--run-root",
            str(tmp_path / "run/attempt1"),
            "--logs",
            str(tmp_path / "logs"),
            "--stage",
            "smoke",
        ]
    )


def fake_inputs(run_root: Path) -> None:
    """Create only synthetic audited bytes, with all three real hash relationships."""
    directory = run_root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt_bank.jsonl").write_text('{"item_id":"synthetic"}\n')
    bank_sha = D.digest(directory / "prompt_bank.jsonl")
    D.atomic_json(
        directory / "bank_audit_report.json", {"passed": True, "prompt_bank_sha256": bank_sha}
    )
    D.atomic_json(
        directory / "upload_verified.json",
        {
            "data_revision": DATA_REV,
            "prompt_bank_sha256": bank_sha,
            "bank_audit_report_sha256": D.digest(directory / "bank_audit_report.json"),
        },
    )


def phase_fixture(run_root: Path, leg: str, phase: str, *, passed=True) -> None:
    """Produce the child manifest shape with deliberately non-historical row counts."""
    root = run_root / leg
    manifest = root / "manifests"
    bank_sha = D.digest(run_root / "inputs/prompt_bank.jsonl")
    n = 3 if leg == "smoke" else 7
    if phase == "gen":
        raw = root / "raw_completions/rollouts.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            "".join(
                json.dumps(
                    {
                        "item_id": f"synthetic-{i}",
                        "synthetic": True,
                        "question": f"Synthetic question {i}",
                        "text": f"Synthetic answer {i}",
                        "source_prompt_id": f"source-{i // 8}",
                        "language": "en",
                    }
                )
                + "\n"
                for i in range(n * 8)
            )
        )
        D.atomic_json(manifest / "input_stage.json", {"data_revision": DATA_REV})
        D.atomic_json(
            manifest / "generation.json",
            {
                "regime": {
                    "git_sha": CODE_SHA,
                    "bank_sha256": bank_sha,
                    "smoke": leg == "smoke",
                    "attempt": 1,
                    "accepted_source_ids_sha256": "d" * 64,
                },
                "n_prompts": n,
                "n_rows": n * 8,
                "package_versions": {"fixture": "1"},
                "rollouts_sha256": D.digest(raw),
            },
        )
    elif phase == "capture":
        tensors = root / "analysis_tensors"
        tensors.mkdir(parents=True, exist_ok=True)
        (tensors / "vc.pt").write_bytes(b"synthetic tensor; never unpickled")
        D.atomic_json(
            manifest / "capture.json",
            {
                "capture_regime": {"git_sha": CODE_SHA, "bank_sha256": bank_sha},
                "n_answer_rows": n * 8,
                "package_versions": {"fixture": "1"},
            },
        )
    elif phase in ("upload-raw", "upload-capture"):
        D.atomic_json(
            manifest / ("raw_upload.json" if phase == "upload-raw" else "capture_upload.json"),
            {"revision": DATA_REV},
        )
    elif phase == "finalize":
        gen, cap = D.read_json(manifest / "generation.json"), D.read_json(manifest / "capture.json")
        if leg == "smoke":
            D.atomic_json(
                manifest / "smoke_timing.json",
                {
                    "passed": passed,
                    "projected_timing_passed": False,
                    "bank_sha256": bank_sha,
                    "generation_compatibility": {
                        k: v for k, v in gen["regime"].items() if k != "smoke"
                    },
                    "capture_compatibility": {
                        "capture_regime": cap["capture_regime"],
                        "package_versions": cap["package_versions"],
                    },
                },
            )
        D.atomic_json(root / D.DONE, {"status": "done", "generation": gen, "capture": cap})
    else:
        raise AssertionError(phase)


def fake_gate(args) -> None:
    """Create a synthetic external gate, with poor numerical agreement but valid parsing."""
    identity = D.gate_identity(args)
    technical = {
        key: True
        for key in ("request_created", "result_received", "parse_complete", "coverage_complete")
    }
    rows = D.read_jsonl(args.run_root / "smoke/raw_completions/rollouts.jsonl")
    contract = D.judge_contract()
    packet_manifest = contract._prepare_packets(
        rows=[
            {"item_id": row["item_id"], "question": row["question"], "response": row["text"]}
            for row in rows
        ],
        packet_kind=f"gpu-smoke-attempt{args.attempt}",
        packet_root=args.run_root / "synthetic_packets",
        overlap_fraction=contract.PRODUCTION_OVERLAP_FRACTION,
    )
    by_id = {row["item_id"]: row for row in rows}
    lookup = [
        {
            **row,
            "source_prompt_id": by_id[row["item_id"]]["source_prompt_id"],
            "language": by_id[row["item_id"]]["language"],
        }
        for row in packet_manifest["mapping"]
    ]
    for packet in packet_manifest["packets"]:
        items = D.read_json(Path(packet["packet_path"]))["items"]
        output = Path(packet["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(
                json.dumps(
                    {
                        "opaque_id": row["opaque_id"],
                        "verdict": False,
                        "raw_output": "<refusal>no</refusal>",
                    }
                )
                + "\n"
                for row in items
            )
        )
    raw_hashes = contract._persist_packet_artifacts(
        packet_manifest, args.run_root / "judge/agent_artifacts/gpu_smoke"
    )
    packet_path = args.run_root / "judge/smoke_packet_manifest.json"
    lookup_path = args.run_root / "judge/smoke_lookup.json"
    D.atomic_json(packet_path, packet_manifest)
    D.atomic_json(lookup_path, lookup)
    ordered_sha = D.hashlib.sha256(
        json.dumps([row["item_id"] for row in rows], sort_keys=True).encode()
    ).hexdigest()
    request = {
        "schema_version": 1,
        "kind": "issue952_codex_smoke_request",
        "identity": identity,
        "n_requests": len(rows),
        "ordered_item_ids_sha256": ordered_sha,
        "lookup_sha256": D.digest(lookup_path),
        "packet_manifest_sha256": D.digest(packet_path),
    }
    parsed = {
        "schema_version": 1,
        "kind": "issue952_codex_smoke_parse",
        "identity": identity,
        "technical": technical,
        "lookup_sha256": D.digest(lookup_path),
        "packet_manifest_sha256": D.digest(packet_path),
        "agent_artifact_hashes": raw_hashes,
        "coverage": {
            "n_smoke_rows": len(rows),
            "n_parsed_rows": len(rows),
            "ordered_item_ids_sha256": ordered_sha,
        },
    }
    D.atomic_json(D.evidence_local(args, "request"), request)
    D.atomic_json(D.evidence_local(args, "parse"), parsed)
    D.evidence_local(args, "result").write_text(
        "".join(
            json.dumps(
                {"item_id": row["item_id"], "verdict": False, "judge_id": row["primary_agent"]}
            )
            + "\n"
            for row in lookup
        )
    )
    evidence = {}
    for role, name in (
        ("request", "smoke_request_manifest.json"),
        ("result", "smoke_scores.jsonl"),
        ("parse", "smoke_parse_manifest.json"),
    ):
        relative = f"judge/{name}"
        (args.run_root / "judge").mkdir(exist_ok=True)
        (args.run_root / relative).write_bytes(D.evidence_local(args, role).read_bytes())
        evidence[role] = {"path": relative, "sha256": D.digest(D.evidence_local(args, role))}
    gate = {
        "schema_version": 1,
        "kind": "issue952_codex_smoke_gate",
        "identity": identity,
        "technical": technical,
        "passed": True,
        "agreement_passed": False,
        "request_sha256": evidence["request"]["sha256"],
        "result_sha256": evidence["result"]["sha256"],
        "evidence": evidence,
        "agent_artifact_hashes": raw_hashes,
        "artifact_census": D.census(
            args.run_root, [path for path in (args.run_root / "judge").rglob("*") if path.is_file()]
        ),
    }
    D.atomic_json(args.run_root / "dispatch_state/codex_smoke_gate.json", gate)
    for relative in gate["artifact_census"]:
        destination = D.artifact_local(args, gate, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((args.run_root / relative).read_bytes())


def complete_stages(args):
    """Simulate separate smoke and production launcher invocations, not an in-pod wait."""
    assert D.dispatch(args) == 0
    args.stage = "production"
    args.smoke_gate_revision = DATA_REV
    assert D.dispatch(args) == 0


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    """Patch only subprocess execution and code checking; their real bodies have own tests."""
    args = args_for(tmp_path)
    calls = []
    mode = {"smoke_passed": True, "finalize_rc": 0, "verify_rc": 0}

    def execute(cmd, log, deadline):
        assert deadline > time.time()
        if "--operation" in cmd:
            operation = cmd[cmd.index("--operation") + 1]
            calls.append(operation)
            if operation == "stage":
                fake_inputs(args.run_root)
            elif operation == "hydrate":
                fake_gate(args)
            else:
                leg = cmd[cmd.index("--leg") + 1]
                if mode["verify_rc"]:
                    return mode["verify_rc"]
                D.atomic_json(
                    args.run_root / "dispatch_state" / f"{leg}_verified.json",
                    {
                        "revision": DATA_REV,
                        "hf_prefix": f"{D.HF_PREFIX}/attempt1"
                        + ("/smoke" if leg == "smoke" else ""),
                        "sha256": D.census(
                            args.run_root / leg,
                            [p for p in (args.run_root / leg).rglob("*") if p.is_file()],
                        ),
                    },
                )
            return 0
        leg = Path(cmd[cmd.index("--out-root") + 1]).name
        phase = cmd[cmd.index("--phase") + 1]
        calls.append((leg, phase))
        phase_fixture(args.run_root, leg, phase, passed=mode["smoke_passed"])
        return mode["finalize_rc"] if phase == "finalize" else 0

    monkeypatch.setattr(D, "run_process", create_autospec(D.run_process, side_effect=execute))
    monkeypatch.setattr(D, "assert_code", create_autospec(D.assert_code))
    return args, calls, mode


def test_complete_order_single_done_and_sentinel_contract(synthetic, capsys):
    args, calls, _ = synthetic
    complete_stages(args)
    expected = ["stage"]
    for leg in ("smoke", "production"):
        if leg == "production":
            expected.append("hydrate")
        expected.extend((leg, phase) for phase in D.PHASES)
        expected.append("verify")
    assert calls == expected
    output = capsys.readouterr().out
    assert output.count("[phase=done]") == 2  # Exactly one per separate pod invocation.
    assert "[smoke_advisory]" in output
    sentinels = list(args.logs.glob("issue-952-epm_results-*.json"))
    assert len(sentinels) == 1
    envelope = D.read_json(sentinels[0])
    poll_tree = ast.parse((ROOT / "scripts/poll_pipeline.py").read_text())
    constants = {
        node.target.id: ast.literal_eval(node.value)
        for node in poll_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"_SENTINEL_REQUIRED_KEYS", "_RESULTS_PAYLOAD_KEYS"}
    }
    assert set(constants["_SENTINEL_REQUIRED_KEYS"]) <= envelope.keys()
    assert set(constants["_RESULTS_PAYLOAD_KEYS"]) <= envelope["note"].keys()
    assert envelope["kind"] == "epm:results"
    assert envelope["note"]["eval_numbers"]["n_prompts"] == 7
    assert envelope["note"]["smoke_timing"]["projected_timing_passed"] is False
    assert envelope["note"]["codex_smoke_gate"]["agreement_passed"] is False
    assert args.logs not in (args.run_root / "dispatch_state/state.json").parents


def test_smoke_exits_with_one_pod_done_but_no_production_result(synthetic, capsys):
    args, calls, _ = synthetic
    assert D.dispatch(args) == 0
    assert not any(isinstance(call, tuple) and call[0] == "production" for call in calls)
    assert capsys.readouterr().out.count("[phase=done]") == 1
    envelope = D.read_json(next(args.logs.glob("*.json")))
    assert envelope["note"]["status"] == "smoke_complete"
    assert envelope["kind"] == "epm:smoke-result"
    assert envelope["blocks_pipeline"] is False
    assert envelope["note"]["gate_identity"]["attempt"] == 1
    assert not list(args.logs.glob("*epm_results*.json"))


def test_production_has_one_terminal_done_in_its_own_invocation(synthetic, capsys):
    args, _, _ = synthetic
    D.dispatch(args)
    capsys.readouterr()
    args.stage, args.smoke_gate_revision = "production", DATA_REV
    D.dispatch(args)
    assert capsys.readouterr().out.count("[phase=done]") == 1


def test_absent_codex_gate_cannot_enter_production(synthetic):
    args, calls, _ = synthetic
    D.dispatch(args)
    args.stage = "production"
    calls.clear()
    with pytest.raises(D.DesignedHalt, match="smoke-gate-revision"):
        D.dispatch(args)
    assert not calls


@pytest.mark.parametrize(
    "field",
    [
        "code_sha",
        "input_revision",
        "attempt",
        "accepted_source_ids_sha256",
        "smoke_report_sha256",
        "smoke_rollouts_sha256",
        "smoke_upload_revision",
        "smoke_upload_receipt_sha256",
    ],
)
def test_codex_gate_rejects_every_stale_identity_field(synthetic, field):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    path = args.run_root / "dispatch_state/codex_smoke_gate.json"
    gate = D.read_json(path)
    gate["identity"][field] = "stale"
    D.atomic_json(path, gate)
    with pytest.raises(RuntimeError, match="identity/schema"):
        D.validate_gate(args)


@pytest.mark.parametrize(
    "field", ["request_created", "result_received", "parse_complete", "coverage_complete", "passed"]
)
def test_codex_gate_requires_technical_completion(synthetic, field):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    path = args.run_root / "dispatch_state/codex_smoke_gate.json"
    gate = D.read_json(path)
    (gate if field == "passed" else gate["technical"])[field] = False
    D.atomic_json(path, gate)
    with pytest.raises(D.DesignedHalt, match="technical"):
        D.validate_gate(args)


def test_production_budget_subtracts_smoke_execution(synthetic):
    args, calls, _ = synthetic
    D.dispatch(args)
    receipt_path = args.run_root / "dispatch_state/smoke_verified.json"
    receipt = D.read_json(receipt_path)
    receipt["smoke_wall_seconds"] = D.WALL_SECONDS
    D.atomic_json(receipt_path, receipt)
    args.stage, args.smoke_gate_revision = "production", DATA_REV
    calls.clear()
    with pytest.raises(D.DesignedHalt, match="combined smoke/production"):
        D.dispatch(args)
    assert calls == ["hydrate"]


def test_real_hydration_uses_pinned_gate_and_smoke_revisions(synthetic, tmp_path, monkeypatch):
    from explore_persona_space.orchestrate import hub

    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    source = args.run_root
    target = args_for(tmp_path / "new-pod")
    target.stage, target.smoke_gate_revision = "production", "a" * 40
    fake_inputs(target.run_root)
    requests = []

    def download(
        repo_id,
        path_in_repo,
        destination,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        relative = path_in_repo.split("/attempt1/", 1)[1]
        requests.append((relative, revision, overwrite))
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / relative).read_bytes())
        return destination

    monkeypatch.setattr(
        hub, "stage_hub_file", create_autospec(hub.stage_hub_file, side_effect=download)
    )
    D.hydrate_smoke(target)
    source_gate = D.read_json(source / "dispatch_state/codex_smoke_gate.json")
    assert [r[1] for r in requests] == [target.smoke_gate_revision] * (
        2 + len(source_gate["artifact_census"])
    ) + [DATA_REV] * 4
    assert all(r[2] for r in requests)
    assert D.validate_gate(target)["passed"] is True


@pytest.mark.parametrize("role", ["request", "result", "parse"])
def test_missing_codex_evidence_blocks_even_with_plausible_gate(synthetic, role):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    D.evidence_local(args, role).unlink()
    with pytest.raises(FileNotFoundError):
        D.validate_gate(args)


@pytest.mark.parametrize("role", ["request", "result", "parse"])
def test_forged_codex_evidence_hash_refused(synthetic, role):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    path = args.run_root / "dispatch_state/codex_smoke_gate.json"
    gate = D.read_json(path)
    gate["evidence"][role]["sha256"] = "0" * 64
    D.atomic_json(path, gate)
    with pytest.raises(RuntimeError, match="evidence content mismatch"):
        D.validate_gate(args)


def test_parse_manifest_cannot_forge_coverage_or_identity(synthetic):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    path = args.run_root / "dispatch_state/codex_smoke_gate.json"
    gate = D.read_json(path)
    parsed_path = D.evidence_local(args, "parse")
    parsed = D.read_json(parsed_path)
    parsed["coverage"]["n_parsed_rows"] -= 1
    D.atomic_json(parsed_path, parsed)
    gate["evidence"]["parse"]["sha256"] = D.digest(parsed_path)
    D.atomic_json(path, gate)
    with pytest.raises(RuntimeError, match="request/parse identity or coverage"):
        D.validate_gate(args)


@pytest.mark.parametrize("suffix", [".packet.json", ".output.jsonl"])
def test_missing_actual_packet_or_raw_output_blocks(synthetic, suffix):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    gate = D.read_json(args.run_root / "dispatch_state/codex_smoke_gate.json")
    relative = next(path for path in gate["artifact_census"] if path.endswith(suffix))
    D.artifact_local(args, gate, relative).unlink()
    with pytest.raises(FileNotFoundError):
        D.validate_gate(args)


def test_raw_output_parser_rejects_invalid_tag_even_with_updated_hashes(synthetic):
    args, _, _ = synthetic
    D.dispatch(args)
    fake_gate(args)
    gate_path = args.run_root / "dispatch_state/codex_smoke_gate.json"
    gate = D.read_json(gate_path)
    relative = next(path for path in gate["artifact_census"] if path.endswith(".output.jsonl"))
    raw_path = D.artifact_local(args, gate, relative)
    rows = D.read_jsonl(raw_path)
    rows[0]["raw_output"] = "not a valid binary judge tag"
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sha = D.digest(raw_path)
    gate["artifact_census"][relative] = sha
    key = relative.removeprefix("judge/agent_artifacts/gpu_smoke/").removesuffix(".output.jsonl")
    gate["agent_artifact_hashes"][key]["output_sha256"] = sha
    parsed_path = D.evidence_local(args, "parse")
    parsed = D.read_json(parsed_path)
    parsed["agent_artifact_hashes"] = gate["agent_artifact_hashes"]
    D.atomic_json(parsed_path, parsed)
    gate["evidence"]["parse"]["sha256"] = D.digest(parsed_path)
    gate["artifact_census"][gate["evidence"]["parse"]["path"]] = D.digest(parsed_path)
    D.atomic_json(gate_path, gate)
    with pytest.raises(RuntimeError, match="invalid Codex judgment row"):
        D.validate_gate(args)


@pytest.mark.parametrize("rc", [0, 1])
def test_negative_smoke_artifact_routes_designed_halt_even_nonzero(synthetic, capsys, rc):
    args, calls, mode = synthetic
    mode.update(smoke_passed=False, finalize_rc=rc)
    with pytest.raises(D.DesignedHalt, match="smoke technical"):
        D.dispatch(args)
    assert not any(isinstance(call, tuple) and call[0] == "production" for call in calls)
    assert "[phase=done]" not in capsys.readouterr().out
    assert not list(args.logs.glob("*.json"))


@pytest.mark.parametrize("failure", ["finalize_rc", "verify_rc"])
def test_local_done_never_overrides_failed_final_upload(synthetic, failure, capsys):
    args, _, mode = synthetic
    mode[failure] = 1
    with pytest.raises(RuntimeError):
        D.dispatch(args)
    assert (args.run_root / "smoke" / D.DONE).exists()
    assert "[phase=done]" not in capsys.readouterr().out
    assert not list(args.logs.glob("*.json"))


def test_resume_rechecks_hashes_and_reruns_upload_and_finalize(synthetic):
    args, calls, _ = synthetic
    complete_stages(args)
    for sentinel in args.logs.glob("*.json"):
        sentinel.rename(sentinel.with_suffix(".json.processed"))
    calls.clear()
    D.dispatch(args)
    assert calls == [
        item
        for leg in ("production",)
        for item in [(leg, "upload-raw"), (leg, "upload-capture"), (leg, "finalize"), "verify"]
    ]
    (args.run_root / "production/raw_completions/rollouts.jsonl").write_text("stale")
    with pytest.raises(RuntimeError, match="stale completed phase"):
        D.dispatch(args)


def test_stale_input_and_code_state_refused(synthetic):
    args, _, _ = synthetic
    D.dispatch(args)
    args.expected_code_sha = "d" * 40
    with pytest.raises(RuntimeError, match="stale dispatcher"):
        D.dispatch(args)
    args.expected_code_sha = CODE_SHA
    (args.run_root / "inputs/prompt_bank.jsonl").write_text("stale")
    with pytest.raises(RuntimeError, match="stale staged inputs"):
        D.dispatch(args)


def test_unowned_partials_and_logs_namespace_refused(synthetic):
    args, _, _ = synthetic
    (args.run_root / "production").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="unowned partial"):
        D.dispatch(args)
    args.run_root = args.logs / "bad"
    with pytest.raises(RuntimeError, match="outside the drained"):
        D.dispatch(args)
    args.run_root = args.logs.parent / "another-root"
    args.logs = args.run_root / "dispatch_state"
    with pytest.raises(RuntimeError, match="outside the drained"):
        D.dispatch(args)


def test_stale_smoke_code_refused(synthetic):
    args, _, _ = synthetic
    D.dispatch(args)
    with pytest.raises(RuntimeError, match="stale smoke"):
        D.validate_smoke(args.run_root, "d" * 40)


def test_command_shape_never_leaks_smoke_to_production(tmp_path):
    args = args_for(tmp_path)
    for phase in D.PHASES:
        smoke = D.command(args, "smoke", phase)
        production = D.command(args, "production", phase)
        assert "--smoke" in smoke and "--smoke-report" not in smoke
        assert "--smoke" not in production and "--smoke-report" in production
        assert (
            smoke[smoke.index("--out-root") + 1] != production[production.index("--out-root") + 1]
        )
        assert D.CHILD in smoke and D.CHILD in production
    with pytest.raises(ValueError, match="immutable"):
        D.immutable_sha("main")


def test_real_process_waits_sanitizes_done_and_clears_api_keys(tmp_path, capsys):
    program = (
        "import os,time; time.sleep(0.1); "
        "assert os.environ['ANTHROPIC_API_KEY'] == ''; "
        "assert os.environ['OPENAI_API_KEY'] == ''; "
        "print('[phase=done] synthetic child'); raise SystemExit(3)"
    )
    started = time.time()
    assert D.run_process([sys.executable, "-c", program], tmp_path / "child.log", started + 10) == 3
    assert time.time() - started >= 0.1
    output = capsys.readouterr().out
    assert "[phase=done]" not in output
    assert "[child_phase=done]" in output


def test_real_process_timeout_is_loud(tmp_path):
    with pytest.raises(TimeoutError, match="wall fence"):
        D.run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path / "timeout.log",
            time.time() + 0.1,
        )


def test_main_rewrites_self_pid_and_posts_blocking_halt(synthetic, monkeypatch):
    args, _, mode = synthetic
    mode["smoke_passed"] = False
    monkeypatch.setattr(
        sys,
        "argv",
        [
            D.SELF,
            "--expected-code-sha",
            CODE_SHA,
            "--input-revision",
            INPUT_REV,
            "--run-root",
            str(args.run_root.parent),
            "--logs",
            str(args.logs),
            "--stage",
            "smoke",
        ],
    )
    args.logs.mkdir()
    (args.logs / "issue-952.pid").write_text("999999999\n")
    assert D.main() == D.HALT_RC
    assert (args.logs / "issue-952.pid").read_text().strip() == str(os.getpid())
    envelope = D.read_json(next(args.logs.glob("*.json")))
    assert envelope["blocks_pipeline"] is True and envelope["kind"] == "epm:progress"
    assert envelope["note"]["exit_code"] == D.HALT_RC


def test_real_code_byte_guard(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    source = checkout / "scripts/worker.py"
    source.write_text("original\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "add", "scripts/worker.py"], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        cwd=checkout,
        check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    monkeypatch.setattr(D, "ROOT", checkout)
    D.assert_code(sha)
    with pytest.raises(RuntimeError, match="code identity mismatch"):
        D.assert_code(CODE_SHA)
    source.write_text("modified\n")
    with pytest.raises(RuntimeError, match="stale source bytes"):
        D.assert_code(sha)


def test_real_staging_body_immutable_revisions_and_hashes(tmp_path, monkeypatch):
    from explore_persona_space.orchestrate import hub

    source = tmp_path / "source"
    fake_inputs(source)
    calls = []

    def download(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        calls.append((repo_id, revision, overwrite))
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / "inputs" / Path(path_in_repo).name).read_bytes())
        return target

    monkeypatch.setattr(
        hub, "stage_hub_file", create_autospec(hub.stage_hub_file, side_effect=download)
    )
    D.stage_inputs(tmp_path / "dest", INPUT_REV)
    assert [call[1] for call in calls] == [INPUT_REV, DATA_REV, DATA_REV]
    assert all(call[2] for call in calls)
    (source / "inputs/prompt_bank.jsonl").write_text("changed")
    with pytest.raises(RuntimeError, match="input hash mismatch"):
        D.stage_inputs(tmp_path / "bad", INPUT_REV)


def test_real_remote_verify_requires_exact_content(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import DatasetInfo, RepoFile

    fake_inputs(tmp_path)
    for phase in D.PHASES:
        phase_fixture(tmp_path, "production", phase)
    root = tmp_path / "production"
    paths = [p for p in root.rglob("*") if p.is_file()]
    entries = [
        RepoFile(
            path=f"{D.HF_PREFIX}/attempt1/{p.relative_to(root)}",
            size=p.stat().st_size,
            oid=D.digest(p, git_blob=True),
            lfs={"size": p.stat().st_size, "oid": D.digest(p), "pointerSize": 130}
            if p.suffix == ".pt"
            else None,
        )
        for p in paths
    ]
    api = create_autospec(HfApi, instance=True)
    api.repo_info.return_value = DatasetInfo(id=D.HF_REPO, sha=DATA_REV)
    api.list_repo_tree.return_value = iter(entries)
    monkeypatch.setattr(huggingface_hub, "HfApi", create_autospec(HfApi, return_value=api))
    D.verify_terminal(tmp_path, "production")
    assert D.read_json(tmp_path / "dispatch_state/production_verified.json")["revision"] == DATA_REV
    assert api.list_repo_tree.call_count == 1
    assert api.list_repo_tree.call_args.kwargs["revision"] == DATA_REV
    entries[0].blob_id = "0" * 40
    api.list_repo_tree.return_value = iter(entries)
    with pytest.raises(RuntimeError, match="remote content mismatch"):
        D.verify_terminal(tmp_path, "production")


def test_no_external_judge_api_surface():
    tree = ast.parse((ROOT / D.SELF).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any(name and name.split(".")[0] in {"anthropic", "openai"} for name in imports)
    assert D.CHILD == "scripts/issue952_china_definitive_gpu.py"


def test_smoke_compatibility_matches_real_child_helpers(tmp_path):
    """Exercise real child compatibility bodies without loading a GPU/model/network."""
    spec = importlib.util.spec_from_file_location("china_gpu_contract", ROOT / D.CHILD)
    assert spec and spec.loader
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)
    fake_inputs(tmp_path)
    for phase in D.PHASES:
        phase_fixture(tmp_path, "smoke", phase)
    manifests = tmp_path / "smoke/manifests"
    gen = D.read_json(manifests / "generation.json")
    gen["regime"].update(
        n_selected_prompts=120,
        selected_source_ids_sha256="synthetic-smoke-identities",
        n_accepted_prompts=1020,
        prompt_token_max=123,
        max_model_len=4096,
    )
    cap = D.read_json(manifests / "capture.json")
    cap["capture_regime"].update(
        n_selected_prompts=120,
        selected_source_ids_sha256="synthetic-smoke-identities",
        generation_fingerprint="synthetic-smoke-generation",
        rollouts_sha256="synthetic-smoke-rollouts",
        n_accepted_prompts=1020,
    )
    report = D.read_json(manifests / "smoke_timing.json")
    report["generation_compatibility"] = child._smoke_generation_compatibility(gen["regime"])
    report["capture_compatibility"] = child._smoke_capture_compatibility(
        cap["capture_regime"], cap["package_versions"]
    )
    for name, value in (("generation", gen), ("capture", cap), ("smoke_timing", report)):
        D.atomic_json(manifests / f"{name}.json", value)
    assert D.validate_smoke(tmp_path, CODE_SHA) == report
