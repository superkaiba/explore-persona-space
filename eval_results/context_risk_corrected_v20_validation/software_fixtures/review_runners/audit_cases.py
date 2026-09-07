"""Audit the recorded smoke and thirteen local error/provenance cases; no model calls."""

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
from inspect_ai.log import read_eval_log

import scripts.context_risk_corrected_audit as module
from scripts.context_risk_impossiblebench_inspect import summarize_logs

base_path = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/impossible_livecodebench_v20_corrected/smoke/run_result.json"
)
base = json.loads(base_path.read_text())
native = read_eval_log(base["log_locations"][0])
manifest = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/data/impossible_livecodebench_promptB/public_pilot_manifest.jsonl"
)
source = Path(module.__file__)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
root = Path(tempfile.mkdtemp(prefix="context-risk-audit-critic-fixed-"))
findings = []


def run(
    name,
    change=lambda log: None,
    report_change=None,
    expected_error=None,
    expected_execution=True,
    validator=None,
    alter_manifest=False,
):
    log = copy.deepcopy(native)
    change(log)
    report = copy.deepcopy(base)
    report.update(summarize_logs([log], epochs=1))
    if report_change:
        report_change(report)
    directory = root / name
    directory.mkdir()
    path = directory / "run_result.json"
    path.write_text(json.dumps(report))
    use_manifest = manifest
    if alter_manifest:
        use_manifest = directory / "manifest.jsonl"
        use_manifest.write_bytes(manifest.read_bytes() + b"\n")
    module.read_eval_log = lambda ignored: log
    try:
        result = module.audit(path, use_manifest)
    except ValueError as error:
        assert expected_error and expected_error in str(error), (name, str(error), expected_error)
        outcome = {"case": name, "passed": True, "rejected": str(error)}
    else:
        assert expected_error is None, (name, "Expected rejection", expected_error)
        assert result["audit_passed"] and result["execution_passed"] == expected_execution, (
            name,
            result,
        )
        rows = [
            json.loads(line) for line in (directory / "rollouts.jsonl").read_text().splitlines()
        ]
        if validator:
            validator(rows, result)
        outcome = {
            "case": name,
            "passed": True,
            "execution_passed": result["execution_passed"],
            "verified_request_seeds": result["verified_request_seeds"],
            "exported_model_events": sum(len(row["model_events"]) for row in rows),
        }
    findings.append(outcome)
    print(json.dumps(outcome), flush=True)


run("actual_smoke")


def missing_primary(log):
    sample = log.samples[0]
    sample.scores = {"unrelated": next(iter(sample.scores.values()))}


def validate_missing(rows, result):
    assert rows[0]["score"] is None and result["technical_errors"] == 1


run(
    "missing_primary_with_other_score",
    missing_primary,
    expected_execution=False,
    validator=validate_missing,
)


def orphan_event(log):
    sample = log.samples[0]
    sample.error = SimpleNamespace(
        model_dump=lambda: {"message": "fixture error after model event"}
    )
    sample.metadata["agentic_results"]["attempt_history"] = []


run(
    "errored_sample_unmatched_successful_event",
    orphan_event,
    expected_error="Attempt/model-event count mismatch",
)
run(
    "native_error_status_report_pass_tampered",
    lambda log: setattr(log, "status", "error"),
    lambda report: report.update(passed=True),
    expected_error="Native log/report disagreement: passed",
)
run(
    "native_error_status_honest_report",
    lambda log: setattr(log, "status", "error"),
    expected_execution=False,
)


def failed_event(log):
    sample = log.samples[0]
    event = copy.deepcopy(next(event for event in sample.events if event.event == "model"))
    event.error = "fixture failed provider request"
    sample.events.append(event)


def validate_failed(rows, result):
    assert sum(len(row["model_events"]) for row in rows) == 8
    assert (
        sum(
            event["error"] == "fixture failed provider request"
            for row in rows
            for event in row["model_events"]
        )
        == 1
    )
    assert result["verified_request_seeds"] == 7


run("failed_provider_event_preserved", failed_event, validator=validate_failed)


def truncated(log):
    sample = log.samples[0]
    record = sample.metadata["agentic_results"]
    attempt = record["attempt_history"][-1]
    attempt.update(stop_reasons=["max_tokens"], category="generation_incomplete", success=False)
    record["censored"] = True
    sample.scores["successful_submission"].value = "N"
    event = [event for event in sample.events if event.event == "model"][-1]
    event.output.choices[0].stop_reason = "max_tokens"
    event.output.usage.output_tokens = 65536


def validate_truncated(rows, result):
    assert (
        result["technical_errors"] == 1
        and result["non_stop_generations"] == 1
        and result["generation_cap_hits"] == 1
    )


run(
    "truncated_generation_is_censored",
    truncated,
    expected_execution=False,
    validator=validate_truncated,
)
run(
    "changed_manifest_rejected",
    alter_manifest=True,
    expected_error="Manifest differs from the frozen input",
)
run(
    "changed_native_input_rejected",
    lambda log: setattr(log.samples[0], "input", "changed"),
    expected_error="Native input differs from manifest",
)
run(
    "changed_native_context_metadata_rejected",
    lambda log: log.samples[0].metadata.update(exact_context_sha256="bad"),
    expected_error="Native metadata differs from manifest",
)
run(
    "changed_requested_coverage_rejected",
    report_change=lambda report: report.update(requested_rollouts=4),
    expected_error="Reported requested coverage differs",
)


def seed_mismatch(log):
    log.samples[0].metadata["agentic_results"]["attempt_history"][0]["request_seed"] += 1


run("changed_seed_rejected", seed_mismatch, expected_error="Request seed mismatch")


def response_mismatch(log):
    log.samples[0].metadata["agentic_results"]["attempt_history"][0]["response"] += "changed"


run("changed_response_rejected", response_mismatch, expected_error="Recorded response differs")
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, (
    "Source changed during review"
)
report = {
    "source_sha256": source_hash,
    "passed": True,
    "n_cases": len(findings),
    "cases": findings,
    "scope": "In-memory mutations and temporary report copies; no model calls or original edits.",
    "actual_smoke_run_result_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    "actual_smoke_native_log_sha256": hashlib.sha256(
        Path(base["log_locations"][0]).read_bytes()
    ).hexdigest(),
}
(root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "fixture_root": str(root),
            "report": str(root / "review.json"),
            "passed": True,
            "n_cases": len(findings),
            "source_sha256": source_hash,
        }
    ),
    flush=True,
)
