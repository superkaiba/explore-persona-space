"""Explicit post-collection settlement for already planned context-capacity censors.

Original reports and process exits remain unchanged. Successful producer phases use
those original validators; only the reviewed deterministic error has a derived path.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

# PROD_IMPORT_LINT_EXEMPT: Runtime pinned by uv --with inspect-ai==0.3.261.
from inspect_ai.log import read_eval_log

from scripts import context_risk_highrate_capacity as capacity
from scripts import context_risk_highrate_collect as collection
from scripts import context_risk_highrate_design as design
from scripts.context_risk_highrate_collect import (
    MODEL,
    audit_native_logs,
    binding,
    expected_keys,
    load_samples,
    phase_paths,
    raw_rows,
    sha256,
    validate_prefixes,
)

SCHEMA = "context_risk_highrate_postrun_audit_v1"
TERMINAL_SCHEMA = "context_risk_highrate_postrun_terminal_v1"
SOURCES = (
    "scripts/context_risk_highrate_capacity.py",
    "scripts/context_risk_highrate_postrun.py",
    "tests/test_context_risk_highrate_postrun.py",
    "tests/test_context_risk_highrate_postrun_integration.py",
    "eval_results/context_risk_highrate_design/plan_v8_context_capacity_verification.md",
)


def source_hashes() -> dict:
    hashes = {**design.source_hashes(), **{name: sha256(design.PROJECT / name) for name in SOURCES}}
    if (
        sha256(Path(capacity.__file__)) != hashes["scripts/context_risk_highrate_capacity.py"]
        or sha256(Path(__file__)) != hashes["scripts/context_risk_highrate_postrun.py"]
    ):
        raise ValueError("Imported postrun modules differ from reviewed worktree bytes")
    return hashes


def validate_review(path: Path) -> dict:
    before = sha256(path)
    review = json.loads(path.read_text())
    if (
        review.get("verdict") != "PASS"
        or not review.get("reviewer")
        or review.get("sources_sha256") != source_hashes()
        or sha256(path) != before
    ):
        raise ValueError("Postrun extension lacks a current independent code review")
    return review


def audit_path(root: Path, phase: str) -> Path:
    return root / design.phase_dir(phase) / "postrun_audit.json"


def write_new(path: Path, value: dict) -> None:
    # Exclusive creation prevents any replacement of original or derived evidence.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def extend_audit(
    root: Path,
    phase: str,
    original: dict,
    logs,
    not_before: float,
    prefix_lengths: dict,
    base_url: str,
) -> dict:
    """Account for every original issue while retaining the original census exactly."""
    expected_issues, rejected, completed, recovered, stops = [], [], [], [], []
    token_files = {}
    for log in logs:
        if log.status != "success" or log.stats.completed_at is None:
            raise ValueError("Capacity settlement requires terminal successful native invocation")
        for sample in log.samples:
            if sample.error is None:
                rows, limits, retries = collection.requests(sample, MODEL)
            else:
                record, rows, retries = capacity.rejected_request(sample)
                path = capacity.proof_path(root, phase, sample)
                proof = capacity.validate_tokens(record, path, base_url)
                token_files[str(path.relative_to(root))] = proof["proof_sha256"]
                token_files[str(path.with_suffix(".tokens.json").relative_to(root))] = proof[
                    "token_file_sha256"
                ]
                record.pop("messages")  # Exact messages are in the unchanged native event.
                record["token_proof"] = proof
                rejected.append(record)
                limits = []
                expected_issues.append(
                    {
                        "scope": [sample.id, sample.epoch],
                        "type": "ValueError",
                        "message": "This terminal receipt requires resolved native sample execution",
                    }
                )
            completed.extend(rows)
            stops.extend(limits)
            recovered.extend(retries)
    if not rejected or original["validation_issues"] != expected_issues:
        raise ValueError("Original issues include an unrecognized error or incomplete evidence")
    all_requests = [*completed, *rejected]
    seeds = [row["request_seed"] for row in all_requests]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Completed or rejected request seeds collide")
    if any(
        datetime.fromisoformat(row["timestamp"]).timestamp() < not_before
        for row in [*all_requests, *recovered]
    ):
        raise ValueError("Request predates the phase freeze")
    if any(
        row["attempt"] == 1 and row["usage"]["input_tokens"] != prefix_lengths[row["sample_id"]]
        for row in completed
    ):
        raise ValueError("Completed-prefix first request differs from frozen tokenizer evidence")
    return {
        **original,
        "schema_version": SCHEMA,
        "verification_passed": True,
        "evidence_verification_passed": True,
        "original_collector_verification_passed": False,
        "original_validation_issues": original["validation_issues"],
        "validation_issues": [],
        "requests": completed,
        "generation_limit_events": stops,
        "recovered_transport_errors": recovered,
        "capacity_censors": rejected,
        "capacity_token_files_sha256": token_files,
        "verification_scope": "Derived request-evidence settlement; original failed collector and unknown outcomes retained unchanged",
    }


def recompute_report(
    root: Path, phase: str, *, base_url: str | None = None, model: str = MODEL, pilot: bool = False
) -> dict:
    """Recompute a complete disk-backed audit without mutations or time-dependent return values."""
    root = Path(root)
    if pilot:
        raise ValueError("The capacity extension does not apply to the immutable pilot")
    sources = source_hashes()
    review_path = root / "setup/postrun_code_review.json"
    review = validate_review(review_path)
    paths = phase_paths(root, phase, pilot)
    initial_paths = [paths[key] for key in ("result", "rows", "audit", "prefix")]
    initial_paths.extend(
        root / name
        for name in (
            "manifests/source.jsonl",
            "manifests/screen_freeze.json",
            "manifests/screen_B.jsonl",
            "manifests/code_review.json",
            f"manifests/{phase}_B.jsonl",
        )
    )
    if phase == "fresh":
        initial_paths.append(root / "selection.json")
    original_hashes = {str(path): sha256(path) for path in initial_paths}
    original_report_hash = original_hashes[str(paths["result"])]
    report = json.loads(paths["result"].read_text())
    if report.get("verification_passed") is not False or report.get("passed") is not False:
        raise ValueError("Derived settlement requires an explicitly failed original report")
    manifest, receipt, epochs, metadata = binding(root, phase)
    samples = load_samples(manifest)
    if (
        model != MODEL
        or report["model"] != MODEL
        or report["metadata"] != metadata
        or report["is_pilot"] != pilot
    ):
        raise ValueError("Saved report model/phase/source binding differs")
    expected_fields = {
        "schema_version": "context_risk_highrate_run_v1",
        "phase": phase,
        "arm": "B",
        "max_attempts": 10,
        "message_limit": 22,
        "sources_sha256": metadata["sources_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "coverage_complete": True,
    }
    if any(report.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Saved report phase/recipe/coverage fields differ")
    if base_url is not None and report["base_url"].rstrip("/") != base_url.rstrip("/"):
        raise ValueError("Saved model endpoint differs")
    cfg = OmegaConf.create({"model": MODEL, "base_url": report["base_url"]})
    for key, path in (
        ("prefix_tokens_sha256", paths["prefix"]),
        ("rollouts_sha256", paths["rows"]),
        ("native_audit_sha256", paths["audit"]),
    ):
        if report[key] != sha256(path):
            raise ValueError(f"Saved collection artifact changed: {key}")
    launch_path = Path(report["launch_config_path"])
    if report["launch_config_sha256"] != sha256(launch_path):
        raise ValueError("Launch evidence changed")
    original_hashes[str(launch_path)] = sha256(launch_path)
    launch = json.loads(launch_path.read_text())
    not_before = receipt["frozen_unix" if phase == "screen" else "selected_unix"]
    if launch["metadata"] != metadata or launch["started_unix"] < not_before:
        raise ValueError("Launch predates selection or differs from current phase metadata")
    launched = launch["config"]
    original_paths = [
        paths["result"],
        paths["prefix"],
        paths["rows"],
        paths["audit"],
        manifest,
        launch_path,
        Path(launched["review"]),
        root / "manifests/source.jsonl",
        root / "manifests/screen_freeze.json",
        root / "manifests/screen_B.jsonl",
        root / "manifests/code_review.json",
        *map(Path, report["native_logs_sha256"]),
    ]
    if phase == "fresh":
        original_paths.append(root / "selection.json")
        validate_selection(root)
    for path in original_paths:
        if str(path) not in original_hashes:
            original_hashes[str(path)] = sha256(path)
    expected_launch = {
        "operation": "run",
        "phase": phase,
        "arm": "B",
        "model": MODEL,
        "base_url": report["base_url"],
        "max_connections": 16,
        "pilot_limit": 16 if pilot else None,
    }
    if (
        any(launched.get(key) != value for key, value in expected_launch.items())
        or Path(launched["root"]).resolve() != root.resolve()
        or launch["critic_review_sha256"] != sha256(Path(launched["review"]))
        or launch["critic_review"] != design.validate_review(Path(launched["review"]))
    ):
        raise ValueError("Launch configuration or independent review differs")
    prefix_record = json.loads(paths["prefix"].read_text())
    validate_prefixes(prefix_record, samples)
    logs = []
    for location, digest in report["native_logs_sha256"].items():
        if sha256(Path(location)) != digest:
            raise ValueError("Native log changed after collection")
        logs.append(read_eval_log(location, resolve_attachments="full"))
    if not logs:
        raise ValueError("No native logs")
    audit = audit_native_logs(logs, samples, metadata, cfg, pilot=pilot, not_before=not_before)
    prefix_lengths = {row["sample_id"]: row["n_prefix_tokens"] for row in prefix_record["contexts"]}
    if any(
        row["attempt"] == 1 and row["usage"]["input_tokens"] != prefix_lengths[row["sample_id"]]
        for row in audit["requests"]
    ):
        raise ValueError("First native request token count differs from saved initial prefix")
    saved = json.loads(paths["audit"].read_text())
    if audit != saved or audit["verification_passed"] is not False:
        raise ValueError("Saved native audit differs from actual terminal evidence")
    if (
        report["validation_issues"] != audit["validation_issues"]
        or report["native_logs_sha256"] != audit["native_logs_sha256"]
    ):
        raise ValueError("Original report issues or native bindings differ")
    if phase == "screen":
        pilot_report = collection.verify_report(root, "screen", pilot=True)
        for location in pilot_report["native_logs_sha256"]:
            collection.verify_reuse(read_eval_log(location, resolve_attachments="full"), logs)
    with paths["rows"].open() as stream:
        if raw_rows(logs) != [json.loads(line) for line in stream]:
            raise ValueError("Saved raw rows differ from the native samples")
    if any(report[key] != audit[key] for key in ("counts", "contexts", "by_condition")):
        raise ValueError("Saved report counts differ from the native audit")
    if (
        epochs != report["epochs"]
        or len(expected_keys(samples, epochs, pilot)) != report["requested_rollouts"]
        or report["realized_rollouts"] != audit["counts"]["realized"]
    ):
        raise ValueError("Saved expected trajectory count differs")
    audit = extend_audit(root, phase, audit, logs, not_before, prefix_lengths, report["base_url"])
    original_hashes.update(
        {str(root / name): value for name, value in audit["capacity_token_files_sha256"].items()}
    )
    if (
        sources != source_hashes()
        or review != validate_review(review_path)
        or original_report_hash != sha256(paths["result"])
        or original_hashes != {path: sha256(Path(path)) for path in original_hashes}
    ):
        raise ValueError("Postrun source or review changed during validation")
    return {
        **audit,
        "postrun_sources_sha256": sources,
        "postrun_review": review,
        "postrun_review_sha256": sha256(review_path),
        "original_artifacts_sha256": original_hashes,
        "run_result_sha256": sha256(paths["result"]),
        "prefix_tokens_sha256": sha256(paths["prefix"]),
        "rollouts_sha256": sha256(paths["rows"]),
        "native_audit_sha256": sha256(paths["audit"]),
    }


def verify_report(root: Path, phase: str) -> dict:
    """Dispatch explicitly between unchanged strict and derived producer evidence."""
    root = Path(root)
    report = json.loads(phase_paths(root, phase)["result"].read_text())
    if report.get("passed") is True:
        if audit_path(root, phase).exists():
            raise ValueError("A strict successful phase cannot also claim failed-report settlement")
        return collection.verify_report(root, phase)
    path = audit_path(root, phase)
    before = sha256(path)
    saved = json.loads(path.read_text())
    actual = recompute_report(root, phase)
    if saved != actual or sha256(path) != before:
        raise ValueError("Derived audit differs from recomputed original evidence")
    return {**actual, "postrun_audit_sha256": before}


def validate_terminal_process(root: Path, phase: str) -> dict:
    root = Path(root)
    path = root / design.phase_dir(phase) / "terminal_process.json"
    before = sha256(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") == TERMINAL_SCHEMA:
        result = check_terminal_receipt(root, phase, receipt)
        if sha256(path) != before:
            raise ValueError("Derived terminal receipt changed during validation")
        return result
    if json.loads(phase_paths(root, phase)["result"].read_text()).get("passed") is not True:
        raise ValueError("An original failed report requires the explicit derived terminal schema")
    return design.validate_terminal_process(root, phase)


def settle(root: Path, phase: str, launch_path: Path) -> dict:
    """Write new sidecars only after completed collection; never change producer bytes."""
    root = Path(root)
    target = root / design.phase_dir(phase) / "terminal_process.json"
    if target.exists() or audit_path(root, phase).exists():
        raise FileExistsError("Postrun settlement is immutable")
    if json.loads(phase_paths(root, phase)["result"].read_text()).get("passed") is True:
        collection.verify_report(root, phase)
        return design.record_terminal_process(root, phase, launch_path)
    audit = recompute_report(root, phase)
    launch = json.loads(launch_path.read_text())
    prefix = root / f"{phase}_{launch['launch_id']}_process"
    exit_path = Path(f"{prefix}.exit.json")
    report = json.loads(phase_paths(root, phase)["result"].read_text())
    paths = [
        launch_path,
        exit_path,
        Path(launch["log_path"]),
        Path(f"{prefix}.pid"),
        Path(f"{prefix}.worker.pid"),
        *map(Path, report["native_logs_sha256"]),
    ]
    # The derived audit is independently re-read by terminal validation. If that
    # validation fails, retain this truthful audit without writing a terminal PASS.
    write_new(audit_path(root, phase), audit)
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "verification_passed": True,
        "phase": phase,
        "run_result_sha256": sha256(phase_paths(root, phase)["result"]),
        "postrun_audit_sha256": sha256(audit_path(root, phase)),
        "original_exit_code": 1,
        "supervisor_pid": launch["supervisor_pid"],
        "worker_pid": launch["worker_pid"],
        "launch_path": str(launch_path),
        "exit_path": str(exit_path),
        "evidence_sha256": {str(path): sha256(path) for path in paths},
        "verified_unix": time.time(),
        "reason": "Owned exit1 is the exact original verification exception; all planned trajectories exist and only verified capacity-censor issues remain",
    }
    check_terminal_receipt(root, phase, receipt)
    write_new(target, receipt)
    return validate_terminal_process(root, phase)


def validate_selection(root: Path) -> dict:
    """Check the explicit extension before the unchanged physical fresh collector runs."""
    _, selection, _ = design.load_phase(root, "fresh")
    screen = verify_report(root, "screen")
    terminal = validate_terminal_process(root, "screen")
    if screen["schema_version"] == SCHEMA:
        if (
            selection["postrun_screen_audit_sha256"] != screen["postrun_audit_sha256"]
            or selection["postrun_sources_sha256"] != source_hashes()
            or selection["postrun_review_sha256"] != screen["postrun_review_sha256"]
            or selection["screen_process_verified_unix"] != terminal["verified_unix"]
            or selection["selected_unix"] < terminal["verified_unix"]
        ):
            raise ValueError("Fresh selection does not bind the reviewed postrun extension")
    return selection


def check_terminal_receipt(root: Path, phase: str, receipt: dict) -> dict:
    """Validate a receipt before writing it or accepting a previously written copy."""
    sources = source_hashes()
    if receipt.get("schema_version") != TERMINAL_SCHEMA:
        raise ValueError("Unexpected derived terminal schema")
    audit = verify_report(root, phase)
    if receipt["postrun_audit_sha256"] != audit["postrun_audit_sha256"]:
        raise ValueError("Terminal receipt lacks the exact derived audit")
    if receipt.get("verification_passed") is not True or receipt["phase"] != phase:
        raise ValueError("Missing terminal process verification")
    if receipt["run_result_sha256"] != sha256(root / design.phase_dir(phase) / "run_result.json"):
        raise ValueError("Collection result changed after process verification")
    launch_path = Path(receipt["launch_path"])
    launch = json.loads(launch_path.read_text())
    if launch["phase"] != phase or launch["mode"] != phase:
        raise ValueError("Terminal launch phase differs")
    prefix = root / f"{phase}_{launch['launch_id']}_process"
    exit_path = Path(f"{prefix}.exit.json")
    log_path = Path(launch["log_path"])
    report = json.loads((root / design.phase_dir(phase) / "run_result.json").read_text())
    expected_paths = {
        str(launch_path),
        str(exit_path),
        str(log_path),
        str(Path(f"{prefix}.pid")),
        str(Path(f"{prefix}.worker.pid")),
        *report["native_logs_sha256"],
    }
    if (
        receipt["exit_path"] != str(exit_path)
        or set(receipt["evidence_sha256"]) != expected_paths
        or not report["native_logs_sha256"]
    ):
        raise ValueError("Terminal process evidence roster differs")
    for path, expected in receipt["evidence_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Terminal process evidence changed: {path}")
    exit_record = json.loads(Path(receipt["exit_path"]).read_text())
    for key in ("supervisor_pid", "worker_pid"):
        if type(receipt[key]) is not int or receipt[key] <= 1 or exit_record[key] != receipt[key]:
            raise ValueError("Invalid or mismatched process identity")
    if (
        type(exit_record["exit_code"]) is not int
        or exit_record["exit_code"] != 1
        or receipt["original_exit_code"] != 1
        or exit_record["cleanup"] != "no_live_members"
    ):
        raise ValueError("Collector did not complete and drain normally")
    started = design._timestamp(launch["started_utc"])
    if exit_record["mode"] != phase or exit_record["finished_unix"] < started:
        raise ValueError("Terminal exit chronology or phase differs")
    if receipt["verified_unix"] < exit_record["finished_unix"]:
        raise ValueError("Terminal verification predates the owned exit")
    log_text = log_path.read_text()
    expected_error = f"RuntimeError: Unverified collection; raw errors/counts preserved at {root / design.phase_dir(phase) / 'run_result.json'}"
    terminal_text = log_text.rstrip()
    hydra_footer = "Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace."
    if terminal_text.endswith("\n\n" + hydra_footer):
        terminal_text = terminal_text.removesuffix("\n\n" + hydra_footer)
    if not terminal_text.endswith("\n" + expected_error):
        raise ValueError(
            "Nonzero exit does not end with the exact frozen post-collection verification exception"
        )
    if not started <= report["completed_unix"] <= exit_record["finished_unix"] + 1:
        raise ValueError("Report completion is outside the owned invocation")
    for key, suffix, tag in (
        ("supervisor_pid", "pid", "supervisor-start"),
        ("worker_pid", "worker.pid", "worker-start"),
    ):
        if (
            int(Path(f"{prefix}.{suffix}").read_text()) != receipt[key]
            or launch[key] != receipt[key]
            or f"[{tag}] mode={phase} pid={receipt[key]}" not in log_text
        ):
            raise ValueError("Terminal launch process identity differs")
    for location, expected in report["native_logs_sha256"].items():
        if receipt["evidence_sha256"][location] != expected:
            raise ValueError("Terminal native hashes differ from collector report")
        native = read_eval_log(location, header_only=True)
        begin, end = (
            design._timestamp(native.stats.started_at),
            design._timestamp(native.stats.completed_at),
        )
        if (
            native.status != "success"
            # Inspect headers truncate start times to whole seconds.
            or not math.floor(started) <= begin <= end <= exit_record["finished_unix"] + 1
        ):
            raise ValueError("Native invocation is not terminal within the owned launch")
    processes = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="], check=True, capture_output=True, text=True
    )
    for line in processes.stdout.splitlines():
        pid, group, state = line.split()
        if not state.startswith(("Z", "X")) and (
            int(pid) == receipt["supervisor_pid"] or int(group) == receipt["worker_pid"]
        ):
            raise ValueError("Owned collector process group is still live")
    if (
        sources != source_hashes()
        or receipt["run_result_sha256"]
        != sha256(root / design.phase_dir(phase) / "run_result.json")
        or receipt["postrun_audit_sha256"] != sha256(audit_path(root, phase))
        or any(sha256(Path(path)) != value for path, value in receipt["evidence_sha256"].items())
        or any(
            sha256(Path(path)) != value
            for path, value in audit["original_artifacts_sha256"].items()
        )
        or audit["postrun_review_sha256"] != sha256(root / "setup/postrun_code_review.json")
    ):
        raise ValueError("Derived terminal source, review or evidence changed during validation")
    return receipt


def select(root: Path) -> dict:
    """Choose the top observed-rate panel only after independent native and code reviews."""
    root = Path(root)
    if json.loads(phase_paths(root, "screen")["result"].read_text()).get("passed") is True:
        verify_report(root, "screen")
        return design.select(root)
    if (root / "selection.json").exists() or (root / "manifests/fresh_B.jsonl").exists():
        raise FileExistsError("Fresh task selection is immutable")
    design.load_phase(root, "screen")
    audit = verify_report(root, "screen")
    process = validate_terminal_process(root, "screen")
    native_path = root / "screen_B/native_audit.json"
    saved = json.loads(native_path.read_text())
    if (
        saved.get("verification_passed") is not False
        or saved["native_logs_sha256"] != audit["native_logs_sha256"]
        or saved["contexts"] != audit["contexts"]
        or saved["counts"] != audit["counts"]
    ):
        raise ValueError("Independent screening native audit differs")
    review_path = root / "screen_B/success_review.json"
    review = json.loads(review_path.read_text())
    if (
        review.get("verdict") != "PASS"
        or not review.get("reviewer")
        or review["native_logs_sha256"] != audit["native_logs_sha256"]
        or review["success_evidence"] != design.success_evidence(root, "screen")
    ):
        raise ValueError("Successful screening bodies lack an exact independent review")
    ranking, roles = design.rank_tasks(audit["contexts"])
    rows = design.source_rows(root / "manifests/source.jsonl")
    fresh = design.make_rows(rows, "fresh", roles)
    path = root / "manifests/fresh_B.jsonl"
    design._write_jsonl_atomic(path, fresh)
    cutoff = ranking[29]["rate_lower"]
    result = {
        "schema_version": "context_risk_highrate_selection_v1",
        "passed": True,
        "selected_unix": time.time(),
        "selected_arm": "B",
        "postrun_screen_audit_sha256": audit["postrun_audit_sha256"],
        "postrun_sources_sha256": source_hashes(),
        "postrun_review_sha256": audit["postrun_review_sha256"],
        "screen_freeze_sha256": sha256(root / "manifests/screen_freeze.json"),
        "screen_native_audit_sha256": sha256(native_path),
        "screen_success_review_sha256": sha256(review_path),
        "screen_process_sha256": sha256(root / "screen_B/terminal_process.json"),
        "screen_process_verified_unix": process["verified_unix"],
        "sources_sha256": design.source_hashes(),
        "plan_sha256": sha256(design.DESIGN / "plan.md"),
        "manifest_sha256": sha256(path),
        "task_roles": roles,
        "ranking": ranking,
        "cutoff_rate_lower": cutoff,
        "cutoff_tie_tasks": [r["task_id"] for r in ranking if r["rate_lower"] == cutoff],
        "excluded_overlapping_upper_bound": [
            r["task_id"] for r in ranking[30:] if r["rate_upper"] >= cutoff
        ],
        "selected_positive_bearing_screen_tasks": sum(r["success"] > 0 for r in ranking[:30]),
        "contexts": 90,
        "epochs": 4,
        "planned_trajectories": 360,
    }
    design._write_json_atomic(root / "selection.json", result)
    design.load_phase(root, "fresh")
    validate_selection(root)
    return result
