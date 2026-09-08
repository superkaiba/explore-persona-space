"""Adversarial software fixtures for fresh transport settlement; no experiment draws."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from inspect_ai.log import read_eval_log, resolve_sample_attachments
from inspect_ai.model import ModelUsage
from omegaconf import OmegaConf

from scripts import context_risk_highrate_capacity as capacity
from scripts import context_risk_highrate_collect as collection
from scripts import context_risk_highrate_design as design
from scripts import context_risk_highrate_postrun as postrun
from scripts import context_risk_highrate_transport as transport
from tests.test_context_risk_highrate_postrun import failed_sample as failed_sample
from tests.test_context_risk_highrate_postrun import seed, write
from tests.test_context_risk_highrate_postrun_integration import BASE_URL, observation
from tests.test_context_risk_highrate_postrun_integration import cohort as cohort

FUTURE = "<Future at 0xabc123 state=finished raised APIConnectionError>"
VALIDATE_NATIVE = collection.validate_native
PROCESS_RUN = postrun.subprocess.run
OLD_FUNCTIONS = {
    name: getattr(postrun, name)
    for name in ("verify_report", "validate_terminal_process", "settle")
}
ACTUAL = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/"
    "impossible_highrate/setup/review_snapshots/"
    "fresh_transport_started_191_samples_20260908T082604Z.zip"
)


def fixture_trace(*, top=False):
    """Explicit software traceback; the actual archived traceback has a separate test."""
    sections = [
        "Traceback (most recent call last):\n"
        "    raise APIConnectionError(request=request) from err\n" + error + "\n"
        for error in transport.TRANSPORT_CHAIN
    ]
    if top:
        sections.append(
            "Traceback (most recent call last):\n"
            "    raise retry_exc from fut.exception()\n"
            f"tenacity.RetryError: RetryError[{FUTURE}]\n"
        )
    return "\nThe above exception was the direct cause of the following exception:\n\n".join(
        sections
    )


def add_transport(sample):
    """Replace only a fabricated capacity failure with fabricated transport calls."""
    sample.metadata["agentic_results"].update(
        harness_version=transport.HARNESS_VERSION, max_attempts=10
    )
    sample.output = sample.events[-2].output.model_copy(deep=True)
    last = sample.events[-1]
    last.error = "Connection error."
    last.traceback = fixture_trace()
    last.traceback_ansi = last.traceback
    failures = [last.model_copy(deep=True) for _ in range(3)]
    for index, event in enumerate(failures):
        event.timestamp += timedelta(seconds=index)
        event.uuid = str(uuid.uuid4())
    sample.events[-1:] = failures
    for event in sample.events:
        if event.event == "model" and not event.uuid:
            event.uuid = str(uuid.uuid4())
    sample.started_at = (sample.events[0].timestamp - timedelta(seconds=0.1)).isoformat()
    sample.completed_at = (sample.events[-1].timestamp + timedelta(seconds=0.1)).isoformat()
    sample.error.message = f"RetryError({FUTURE})"
    sample.error.traceback = fixture_trace(top=True)
    sample.error.traceback_ansi = sample.error.traceback
    return sample


@pytest.fixture
def transport_sample(failed_sample):
    sample = failed_sample.model_copy(deep=True)
    sample.id = str(sample.id).replace("highrate_screen:", "highrate_fresh:")
    sample.events[0].config.seed = seed(sample.id, sample.epoch, 1)
    sample.events[-1].config.seed = seed(sample.id, sample.epoch, 2)
    sample.metadata["agentic_results"]["attempt_history"][0]["request_seed"] = sample.events[
        0
    ].config.seed
    return add_transport(sample)


def test_actual_archived_failure_body_without_generation():
    if not ACTUAL.exists():
        pytest.skip("Actual local191 native artifact is not distributed as a unit fixture")
    assert (
        collection.sha256(ACTUAL)
        == "bbf35377fdbe94a97b0262dac9a310cca3bf6776584e89085ea0d971396a2de0"
    )
    log = read_eval_log(str(ACTUAL), format="eval", resolve_attachments="full")
    sample = next(
        s for s in log.samples if s.id == "highrate_fresh:B:lcbhard_64:conflicting" and s.epoch == 2
    )
    sample = resolve_sample_attachments(sample, "full")
    before = capacity.sample_digest(sample)
    item, completed, recovered = transport.exhausted_request(sample)
    assert item["attempt"] == 10 and item["request_seed"] == 3738862221
    assert len(completed) == 9 and len(item["transport_events"]) == 3 and not recovered
    assert item["server_execution"] == "unknown" and item["observed_completion"] is False
    assert (
        before
        == capacity.sample_digest(sample)
        == "2a6ffb8c3f78d73c7c60011abe5ac790d5f915753a71e61d5affb63c03123063"
    )


def test_nonempty_fixture_prefix_unknown_and_immutable(transport_sample):
    before = capacity.sample_digest(transport_sample)
    item, completed, retries = transport.exhausted_request(transport_sample)
    assert item["attempt"] == 2 and len(completed) == 1 and not retries
    assert len(item["transport_events"]) == 3
    assert collection.outcome(transport_sample) == "censored"
    assert capacity.sample_digest(transport_sample) == before
    with pytest.raises(ValueError, match="resolved native"):
        collection.requests(transport_sample, collection.MODEL)


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "model",
        "config",
        "feedback",
        "sample_messages",
        "wrapper",
        "future",
        "top_trace",
        "event_trace",
        "error",
        "usage",
        "completed",
        "content",
        "extra_choice",
        "metadata",
        "message_metadata",
        "time",
        "call",
        "tools",
        "logprobs",
        "stop_details",
        "fewer",
        "extra",
        "earlier_success",
        "intervention",
        "wrong_category",
        "wrong_prior_seed",
        "order",
        "same_time",
        "wrong_score",
        "censored",
        "zero_prefix",
        "invalidation",
        "stop_seqs",
        "frequency_penalty",
        "input_tool_calls",
        "input_metadata",
        "duplicate_uuid",
        "sample_end",
        "sample_start",
        "prefix_end_overlap",
        "top_output",
        "top_usage",
        "record_final_response",
        "record_total_messages",
        "record_used_feedback",
        "record_version",
        "record_max_attempts",
        "bool_temperature",
        "bool_top_p",
        "bool_repetition_penalty",
        "numeric_thinking",
    ],
)
def test_transport_corruption_fails_closed(transport_sample, mutation):  # noqa: C901 - independent corruption cases
    s = transport_sample
    event = s.events[-1]
    row = s.metadata["agentic_results"]["attempt_history"][0]
    if mutation == "seed":
        event.config.seed += 1
    elif mutation == "model":
        event.model = "wrong"
    elif mutation == "config":
        event.config.temperature = 0.5
    elif mutation == "feedback":
        event.input[-1].content = "changed"
    elif mutation == "sample_messages":
        s.messages[-1].content = "changed"
    elif mutation == "wrapper":
        s.error.message = "APIConnectionError('Connection error.')"
    elif mutation == "future":
        s.error.message = s.error.message.replace("abc123", "def456")
    elif mutation == "top_trace":
        s.error.traceback = s.error.traceback.replace("RemoteProtocolError", "ReadTimeout")
    elif mutation == "event_trace":
        event.traceback = "unproven"
    elif mutation == "error":
        event.error = "different connection error"
    elif mutation == "usage":
        event.output.usage = ModelUsage(input_tokens=3, output_tokens=0, total_tokens=3)
    elif mutation == "completed":
        event.completed = event.timestamp
    elif mutation == "content":
        event.output.choices[0].message.content = "hidden response"
    elif mutation == "extra_choice":
        event.output.choices.append(event.output.choices[0].model_copy(deep=True))
    elif mutation == "metadata":
        event.output.metadata = {"hidden": "response"}
    elif mutation == "message_metadata":
        event.output.choices[0].message.metadata = {"hidden": "response"}
    elif mutation == "time":
        event.output.time = 1.0
    elif mutation == "call":
        event.call = {"response": "hidden"}
    elif mutation == "tools":
        event.tool_choice = "auto"
    elif mutation == "logprobs":
        event.output.choices[0].logprobs = []
    elif mutation == "stop_details":
        event.output.choices[0].stop_details = {"hidden": "response"}
    elif mutation == "fewer":
        s.events.pop()
    elif mutation == "extra":
        s.events.insert(1, s.events[-1].model_copy(deep=True))
    elif mutation == "earlier_success":
        row["success"] = True
    elif mutation == "intervention":
        s.metadata["agentic_results"]["flag_for_human_intervention"] = True
    elif mutation == "wrong_category":
        row["category"] = "generation_incomplete"
    elif mutation == "wrong_prior_seed":
        row["request_seed"] += 1
    elif mutation == "order":
        s.events[0].completed = event.timestamp + timedelta(seconds=1)
    elif mutation == "same_time":
        event.timestamp = s.events[-2].timestamp
    elif mutation == "wrong_score":
        s.scores["successful_submission"].value = "C"
    elif mutation == "censored":
        s.metadata["agentic_results"]["censored"] = True
    elif mutation == "zero_prefix":
        s.metadata["agentic_results"]["attempt_history"] = []
    elif mutation == "invalidation":
        s.invalidation = {"reason": "fixture invalidation"}
    elif mutation in {"stop_seqs", "frequency_penalty"}:
        for e in s.events[-3:]:
            setattr(e.config, mutation, ["INJECTED_STOP"] if mutation == "stop_seqs" else 1.0)
    elif mutation == "input_tool_calls":
        for e in s.events[-3:]:
            next(m for m in e.input if m.role == "assistant").tool_calls = []
    elif mutation == "input_metadata":
        event.input[0].metadata = {"unreviewed": "input"}
    elif mutation == "duplicate_uuid":
        event.uuid = s.events[-2].uuid
    elif mutation == "sample_end":
        s.completed_at = (event.timestamp - timedelta(seconds=1)).isoformat()
    elif mutation == "sample_start":
        s.started_at = (s.events[0].timestamp + timedelta(seconds=1)).isoformat()
    elif mutation == "prefix_end_overlap":
        s.events[0].completed = s.events[1].timestamp + timedelta(seconds=0.1)
    elif mutation == "top_output":
        s.output.choices[0].message.content = "unrecorded tenth completion"
    elif mutation == "top_usage":
        s.output.usage.output_tokens += 1
    elif mutation in {"record_final_response", "record_total_messages", "record_used_feedback"}:
        s.metadata["agentic_results"][mutation.removeprefix("record_")] = "injected"
    elif mutation == "record_version":
        s.metadata["agentic_results"]["harness_version"] = "other"
    elif mutation == "record_max_attempts":
        s.metadata["agentic_results"]["max_attempts"] = 11
    elif mutation in {"bool_temperature", "bool_top_p"}:
        for e in s.events:
            setattr(e.config, mutation.removeprefix("bool_"), True)
    elif mutation in {"bool_repetition_penalty", "numeric_thinking"}:
        for e in s.events:
            e.config.extra_body = json.loads(json.dumps(e.config.extra_body))
            if mutation == "numeric_thinking":
                e.config.extra_body["chat_template_kwargs"]["enable_thinking"] = 0
            else:
                e.config.extra_body["repetition_penalty"] = True
    with pytest.raises((ValueError, TypeError)):
        transport.exhausted_request(s)


def test_recovered_prefix_events_not_omitted(transport_sample):
    first = transport_sample.events[0]
    retry = first.model_copy(deep=True)
    retry.error = "Connection error."
    retry.completed = None
    retry.output = transport_sample.events[-1].output.model_copy(deep=True)
    retry.timestamp -= timedelta(seconds=1)
    retry.uuid = str(uuid.uuid4())
    transport_sample.started_at = (retry.timestamp - timedelta(seconds=0.1)).isoformat()
    transport_sample.events.insert(0, retry)
    item, completed, retries = transport.exhausted_request(transport_sample)
    assert len(retries) == 1 and len(completed) == 1 and len(item["transport_events"]) == 3


def test_runtime_and_unknown_audit_issues(tmp_path, transport_sample, monkeypatch):
    log = SimpleNamespace(
        status="success",
        stats=SimpleNamespace(completed_at=transport_sample.completed_at),
        samples=[transport_sample],
    )
    issue = {
        "scope": [transport_sample.id, 1],
        "type": "ValueError",
        "message": "This terminal receipt requires resolved native sample execution",
    }
    args = (
        tmp_path,
        "fresh",
        {"validation_issues": [issue]},
        [log],
        0,
        {transport_sample.id: 3},
        BASE_URL,
    )
    result = transport.extend_audit(*args)
    assert len(result["transport_censors"]) == 1 and result["capacity_censors"] == []
    args[2]["validation_issues"].append({"scope": "unknown"})
    with pytest.raises(ValueError, match="unrecognized"):
        transport.extend_audit(*args)
    monkeypatch.setattr(transport.importlib.metadata, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="runtime"):
        transport.extend_audit(*args)


@pytest.fixture
def fresh_cohort(cohort, monkeypatch, failed_sample):
    """Execute all new bodies on360 typed rows after a real frozen screen/selection fixture.

    Only unchanged native-header/serialization and process-list boundaries are autospec
    fixtures. Original screen settle/select and all new transport bodies execute.
    """
    c = cohort
    postrun.settle(c.root, "screen", c.launch_path)
    selection = postrun.select(c.root)
    preserved = {str(p): collection.sha256(p) for p in c.root.rglob("*") if p.is_file()}
    manifest, _, epochs, metadata = collection.binding(c.root, "fresh")
    samples = collection.load_samples(manifest)
    assert epochs == 4 and len(samples) == 90
    start = datetime.fromtimestamp(selection["selected_unix"] + 0.25, UTC)
    finished = start.timestamp() + 7
    monkeypatch.setattr(transport.time, "time", lambda: finished + 1)
    rows = [
        observation(
            failed_sample,
            source,
            epoch,
            start,
            rejected=epoch == 1 and source.id in {samples[0].id, samples[1].id},
        )
        for source in samples
        for epoch in (1, 2, 3, 4)
    ]
    failed = next(r for r in rows if r.id == samples[0].id and r.epoch == 1)
    add_transport(failed)
    capacity_sample = next(r for r in rows if r.id == samples[1].id and r.epoch == 1)
    capacity.cache_tokens(capacity_sample, c.root, "fresh", BASE_URL)
    token_path = capacity.proof_path(c.root, "fresh", capacity_sample)
    paths = collection.phase_paths(c.root, "fresh")
    native_path = paths["out"] / "fresh_fixture.eval"
    write(native_path, {"explicit_fixture_not_native_serialization": True, "rows": 360})
    native = SimpleNamespace(
        location=str(native_path),
        status="success",
        samples=rows,
        stats=SimpleNamespace(
            started_at=start.replace(microsecond=0).isoformat(),
            completed_at=(start + timedelta(seconds=5)).isoformat(),
        ),
    )
    old_header = collection.validate_native.side_effect

    def native_header(log, actual_samples, actual_metadata, cfg):
        if log is native:
            assert [s.id for s in actual_samples] == [s.id for s in samples]
            assert actual_metadata == metadata and cfg.base_url == BASE_URL
        else:
            old_header(log, actual_samples, actual_metadata, cfg)

    monkeypatch.setattr(
        collection, "validate_native", create_autospec(VALIDATE_NATIVE, side_effect=native_header)
    )

    def reader(location, *, header_only=False, resolve_attachments=False, **kwargs):
        if Path(location) == native_path:
            return native
        return c.reader(
            location, header_only=header_only, resolve_attachments=resolve_attachments, **kwargs
        )

    monkeypatch.setattr(
        transport, "read_eval_log", create_autospec(read_eval_log, side_effect=reader)
    )
    monkeypatch.setattr(
        postrun, "read_eval_log", create_autospec(read_eval_log, side_effect=reader)
    )
    write(
        c.root / transport.REVIEW_PATH,
        {
            "verdict": "PASS",
            "reviewer": "SOFTWARE FIXTURE ONLY",
            "sources_sha256": transport.source_hashes(),
        },
    )
    write(
        c.root / "setup/pre_fresh_input_applicability.json",
        {
            "software_fixture_only": True,
            "selection_sha256": collection.sha256(c.root / "selection.json"),
        },
    )
    write(
        paths["prefix"],
        {
            "passed": True,
            "n_contexts": 90,
            "contexts": [
                {
                    "sample_id": s.id,
                    "exact_context_sha256": s.metadata["exact_context_sha256"],
                    "token_ids": [1, 2, 3],
                    "n_prefix_tokens": 3,
                    "prefix_token_ids_sha256": capacity.digest([1, 2, 3]),
                }
                for s in samples
            ],
        },
    )
    original_launch = json.loads(c.producer_launch.read_text())
    original_launch.update(started_unix=start.timestamp(), metadata=metadata)
    original_launch["config"]["phase"] = "fresh"
    producer = paths["out"] / "fixture_launch_config.json"
    write(producer, original_launch)
    launch = json.loads(c.launch_path.read_text())
    launch.update(
        phase="fresh",
        mode="fresh",
        launch_id="fresh_fixture",
        started_utc=start.isoformat(),
        log_path=str(c.root / "fresh_fixture.log"),
    )
    launch_path = c.root / "fresh_owned_launch.json"
    write(launch_path, launch)
    prefix = c.root / "fresh_fresh_fixture_process"
    for key, suffix in (("supervisor_pid", "pid"), ("worker_pid", "worker.pid")):
        Path(f"{prefix}.{suffix}").write_text(str(launch[key]))
    log_path = Path(launch["log_path"])
    log_path.write_text(
        f"[supervisor-start] mode=fresh pid={launch['supervisor_pid']}\n"
        f"[worker-start] mode=fresh pid={launch['worker_pid']}\n"
        f"RuntimeError: Unverified collection; raw errors/counts preserved at {paths['result']}\n\n"
        "Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.\n"
    )
    exit_path = Path(f"{prefix}.exit.json")
    write(
        exit_path,
        {
            "mode": "fresh",
            "supervisor_pid": launch["supervisor_pid"],
            "worker_pid": launch["worker_pid"],
            "exit_code": 1,
            "cleanup": "no_live_members",
            "finished_unix": finished,
        },
    )

    def persist():
        audit = collection.audit_native_logs(
            [native],
            samples,
            metadata,
            OmegaConf.create({"model": collection.MODEL, "base_url": BASE_URL}),
            pilot=False,
            not_before=selection["selected_unix"],
        )
        write(paths["audit"], audit)
        design._write_jsonl_atomic(paths["rows"], collection.raw_rows([native]))
        report = {
            **c.report,
            "phase": "fresh",
            "metadata": metadata,
            "epochs": 4,
            "requested_rollouts": 360,
            "realized_rollouts": audit["counts"]["realized"],
            "coverage_complete": audit["coverage_complete"],
            "manifest_sha256": metadata["manifest_sha256"],
            "prefix_tokens_sha256": collection.sha256(paths["prefix"]),
            "rollouts_sha256": collection.sha256(paths["rows"]),
            "native_audit_sha256": collection.sha256(paths["audit"]),
            "launch_config_path": str(producer),
            "launch_config_sha256": collection.sha256(producer),
            "completed_unix": start.timestamp() + 6,
        }
        report.update(
            {
                k: audit[k]
                for k in (
                    "contexts",
                    "counts",
                    "by_condition",
                    "validation_issues",
                    "native_logs_sha256",
                )
            }
        )
        write(paths["result"], report)
        return report

    report = persist()
    originals = {
        str(p): collection.sha256(p)
        for p in [*paths.values(), producer, launch_path, log_path, exit_path]
        if p.is_file()
    }
    return SimpleNamespace(
        root=c.root,
        paths=paths,
        native=native,
        failed=failed,
        capacity_sample=capacity_sample,
        token_path=token_path,
        preserved=preserved,
        originals=originals,
        launch_path=launch_path,
        log_path=log_path,
        exit_path=exit_path,
        report=report,
        persist=persist,
        start=start,
        ps=c.ps,
    )


def test_full_mixed360_recompute_settle_validate_and_originals(fresh_cohort):
    c = fresh_cohort
    audit = transport.recompute_report(c.root, "fresh")
    assert audit["counts"]["planned"] == audit["counts"]["realized"] == 360
    assert audit["counts"]["censored"] == 2 and audit["counts"]["success"] == 358
    assert len(audit["transport_censors"]) == len(audit["capacity_censors"]) == 1
    assert len(audit["requests"]) == 360
    assert audit["original_collector_verification_passed"] is False
    terminal = transport.settle(c.root, "fresh", c.launch_path)
    assert terminal["schema_version"] == transport.TERMINAL_SCHEMA
    assert transport.verify_report(c.root, "fresh")["counts"] == audit["counts"]
    assert transport.validate_terminal_process(c.root, "fresh") == terminal
    assert {p: collection.sha256(Path(p)) for p in c.originals} == c.originals
    assert {p: collection.sha256(Path(p)) for p in c.preserved} == c.preserved
    assert transport.verify_report(c.root, "screen") == postrun.verify_report(c.root, "screen")
    assert transport.validate_terminal_process(
        c.root, "screen"
    ) == postrun.validate_terminal_process(c.root, "screen")
    with pytest.raises(FileExistsError):
        transport.settle(c.root, "fresh", c.launch_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "wrong_error",
        "wrong_exit",
        "log_tail",
        "native_chronology",
        "raw_mutation",
        "review",
    ],
)
def test_full_rejections_preserve_original_failure(fresh_cohort, mutation):
    c = fresh_cohort
    if mutation == "missing":
        c.native.samples.pop()
        c.persist()
    elif mutation == "duplicate":
        c.native.samples.append(c.native.samples[-1].model_copy(deep=True))
        c.persist()
    elif mutation == "wrong_error":
        c.failed.error.message = "UnrecognizedError"
        c.persist()
    elif mutation == "wrong_exit":
        value = json.loads(c.exit_path.read_text())
        value["exit_code"] = 137
        write(c.exit_path, value)
    elif mutation == "log_tail":
        with c.log_path.open("a") as f:
            f.write("UnrelatedError: a later failure\n")
    elif mutation == "native_chronology":
        c.native.stats.started_at = (c.start - timedelta(seconds=2)).isoformat()
    elif mutation == "raw_mutation":
        with c.paths["rows"].open("a") as f:
            f.write("{}\n")
    elif mutation == "review":
        value = json.loads((c.root / transport.REVIEW_PATH).read_text())
        value["sources_sha256"] = {}
        write(c.root / transport.REVIEW_PATH, value)
    with pytest.raises((ValueError, KeyError)):
        transport.settle(c.root, "fresh", c.launch_path)
    assert not (c.root / "fresh_B/terminal_process.json").exists()
    assert json.loads(c.paths["result"].read_text())["passed"] is False


def test_token_change_at_process_boundary_is_not_accepted(fresh_cohort, monkeypatch):
    c = fresh_cohort
    original = c.ps.side_effect

    def listing(*args, **kwargs):
        import inspect

        # Ignore the unchanged screen terminal check nested inside fresh validation.
        nearest = next(
            frame for frame in inspect.stack() if frame.function == "check_terminal_receipt"
        )
        if nearest.filename != transport.__file__:
            return SimpleNamespace(stdout="", returncode=0)
        p = c.token_path.with_suffix(".tokens.json")
        with p.open("a") as f:
            f.write(" ")
        if original:
            return original(*args, **kwargs)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(
        transport.subprocess, "run", create_autospec(PROCESS_RUN, side_effect=listing)
    )
    with pytest.raises(ValueError):
        transport.settle(c.root, "fresh", c.launch_path)
    assert not (c.root / "fresh_B/terminal_process.json").exists()


def test_direct_recompute_refuses_existing_v8_sidecar_before_inputs(tmp_path):
    write(postrun.audit_path(tmp_path, "fresh"), {"immutable_v8": True})
    with pytest.raises(ValueError, match="existing V8 fresh audit"):
        transport.recompute_report(tmp_path, "fresh")


@pytest.mark.parametrize("operation", ["recompute", "verify", "terminal"])
def test_v8_sidecar_appearing_during_operation_is_rejected(fresh_cohort, monkeypatch, operation):
    c = fresh_cohort
    if operation != "recompute":
        postrun.write_new(
            transport.audit_path(c.root, "fresh"), transport.recompute_report(c.root, "fresh")
        )
    if operation == "terminal":
        # Build the normal receipt with the actual settlement body, then test readback.
        transport.audit_path(c.root, "fresh").unlink()
        transport.settle(c.root, "fresh", c.launch_path)
        original = c.ps.side_effect

        def listing(*args, **kwargs):
            import inspect

            nearest = next(
                frame for frame in inspect.stack() if frame.function == "check_terminal_receipt"
            )
            if nearest.filename == transport.__file__:
                write(postrun.audit_path(c.root, "fresh"), {"unexpected_v8": True})
            if original:
                return original(*args, **kwargs)
            return SimpleNamespace(stdout="", returncode=0)

        monkeypatch.setattr(
            transport.subprocess, "run", create_autospec(PROCESS_RUN, side_effect=listing)
        )

        def invoke():
            return transport.validate_terminal_process(c.root, "fresh")
    else:
        original = capacity.validate_tokens

        def validate(*args, **kwargs):
            result = original(*args, **kwargs)
            if "fresh_B" in Path(args[1]).parts:
                write(postrun.audit_path(c.root, "fresh"), {"unexpected_v8": True})
            return result

        monkeypatch.setattr(
            capacity, "validate_tokens", create_autospec(original, side_effect=validate)
        )

        def invoke():
            return (
                transport.recompute_report if operation == "recompute" else transport.verify_report
            )(c.root, "fresh")

    with pytest.raises(ValueError, match=r"V8|competing"):
        invoke()
    assert {p: collection.sha256(Path(p)) for p in c.originals} == c.originals


def test_pure_delegates_do_not_require_transport_review(tmp_path, monkeypatch):
    for phase in ("screen", "fresh"):
        phase_root = tmp_path / phase
        paths = collection.phase_paths(phase_root, phase)
        paths["out"].mkdir(parents=True, exist_ok=True)
        write(paths["result"], {"passed": True})
        paths["rows"].write_text('{"error": null}\n')
        write(paths["out"] / "terminal_process.json", {"schema_version": "strict"})
        for fn in ("verify_report", "validate_terminal_process"):
            boundary = create_autospec(OLD_FUNCTIONS[fn], return_value={"unchanged": fn})
            monkeypatch.setattr(postrun, fn, boundary)
            assert getattr(transport, fn)(phase_root, phase) == {"unchanged": fn}
            boundary.assert_called_once_with(phase_root, phase)
        (paths["out"] / "terminal_process.json").unlink()
        boundary = create_autospec(OLD_FUNCTIONS["settle"], return_value={"unchanged": "settle"})
        monkeypatch.setattr(postrun, "settle", boundary)
        launch = phase_root / "launch.json"
        assert transport.settle(phase_root, phase, launch) == {"unchanged": "settle"}
        boundary.assert_called_once_with(phase_root, phase, launch)
