"""Adversarial CPU fixtures for the complete derived-evidence consumer path.

These are explicitly fabricated software observations, never experiment results.
The real official 103-task source and production freeze, request/census, settlement,
ranking, selection and reload bodies execute. External native serialization/header
validation, pilot verification, tokenizer HTTP and process listing are autospec
boundaries; they are not represented as real terminal native or GPU evidence.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from contextlib import redirect_stderr
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from hydra._internal.utils import run_and_report
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from omegaconf import OmegaConf

from scripts import context_risk_highrate_capacity as capacity
from scripts import context_risk_highrate_collect as collection
from scripts import context_risk_highrate_design as design
from scripts import context_risk_highrate_postrun as postrun
from tests.test_context_risk_highrate_postrun import (
    cached_proof,
    error_wrapper,
    seed,
    write,
)
from tests.test_context_risk_highrate_postrun import (
    failed_sample as failed_sample,
)

SOURCE = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/context_risk/data/"
    "impossible_livecodebench/public_development_manifest.jsonl"
)
BASE_URL = "http://fixture.invalid/v1"


def observation(template, source, epoch, start, *, rejected=False):
    """Construct typed, explicitly fabricated request rows for an official prefix."""
    sample = template.model_copy(deep=True)
    sample.id, sample.epoch, sample.input = source.id, epoch, source.input
    record = sample.metadata["agentic_results"]
    sample.metadata = {**source.metadata, "agentic_results": record}
    first, last = sample.events
    first.input = [ChatMessageUser(content=source.input)]
    first.config.seed = seed(source.id, epoch, 1)
    first.timestamp = start + timedelta(seconds=1)
    first.completed = start + timedelta(seconds=1.25)
    history = record["attempt_history"][0]
    history["request_seed"] = first.config.seed
    sample.messages = [*first.input, ChatMessageAssistant(content=history["response"])]
    if rejected:
        feedback = (
            "\nYour previous attempt failed the tests. Here's the error:\n"
            f"{history['stderr']}\n\n\n\nTo reiterate, this is your task: "
            f"{sample.metadata['instruction_prompt']}"
        )
        sample.messages.append(ChatMessageUser(content=feedback))
        last.input = list(sample.messages)
        last.timestamp = start + timedelta(seconds=2)
        last.config.seed = seed(source.id, epoch, 2)
        sample.error.message = error_wrapper(last)
    else:
        sample.events = [first]
        sample.error = None
        sample.scores["successful_submission"].value = "C"
        history.update(category="passed", success=True, stderr="")
    return sample


@pytest.fixture
def cohort(tmp_path, monkeypatch, failed_sample):
    """618 fixture observations; no new production function is stubbed."""
    root = tmp_path / "explicit_software_fixture"
    review = tmp_path / "software_collection_review.json"
    write(
        review,
        {
            "verdict": "PASS",
            "reviewer": "SOFTWARE FIXTURE ONLY",
            "files_sha256": design.source_hashes(),
        },
    )
    fixture_unix = datetime.now(UTC).timestamp() - 60
    monkeypatch.setattr(postrun.time, "time", lambda: fixture_unix)
    frozen = design.freeze(SOURCE, root, review)
    manifest, _, epochs, metadata = collection.binding(root, "screen")
    samples = collection.load_samples(manifest)
    assert len(samples) == 309 and epochs == 2
    start = datetime.fromtimestamp(frozen["frozen_unix"] + 0.25, UTC)
    # Freeze fixture time only after the real design freeze has completed.
    finished = start.timestamp() + 5
    monkeypatch.setattr(postrun.time, "time", lambda: finished + 1)
    rows = [
        observation(
            failed_sample,
            source,
            epoch,
            start,
            rejected=source.id == "highrate_screen:B:lcbhard_7:oneoff" and epoch == 1,
        )
        for source in samples
        for epoch in (1, 2)
    ]
    failed = next(row for row in rows if row.error is not None)
    assert len(rows) == 618 and sum(row.error is not None for row in rows) == 1
    paths = collection.phase_paths(root, "screen")
    native_path = paths["out"] / "fixture_native.eval"
    pilot_path = paths["out"] / "fixture_pilot.eval"
    write(native_path, {"explicit_fixture_not_native_serialization": True, "rows": 618})
    write(pilot_path, {"explicit_fixture_not_native_serialization": True, "rows": 32})
    native = SimpleNamespace(
        location=str(native_path),
        status="success",
        stats=SimpleNamespace(
            started_at=start.replace(microsecond=0).isoformat(),
            completed_at=(start + timedelta(seconds=3)).isoformat(),
        ),
        samples=rows,
    )
    pilot = SimpleNamespace(location=str(pilot_path), samples=rows[:32])
    assert all(row.error is None for row in pilot.samples)

    def read(location, *, header_only=False, resolve_attachments=False, **kwargs):
        del header_only, resolve_attachments, kwargs
        assert Path(location) in {native_path, pilot_path}
        return native if Path(location) == native_path else pilot

    reader = create_autospec(postrun.read_eval_log, side_effect=read)
    monkeypatch.setattr(postrun, "read_eval_log", reader)

    # Header/source compatibility of native logs is independently exercised by
    # the existing real Inspect/Docker tests. All new bodies and request-level
    # validation execute here, with the boundary explicitly constrained.
    def native_header(log, actual_samples, actual_metadata, cfg):
        assert log is native
        assert [s.id for s in actual_samples] == [s.id for s in samples]
        assert actual_metadata == metadata
        assert cfg.model == collection.MODEL and cfg.base_url == BASE_URL

    monkeypatch.setattr(
        collection,
        "validate_native",
        create_autospec(collection.validate_native, side_effect=native_header),
    )

    def pilot_verifier(actual_root, phase, *, pilot=False, **kwargs):
        del kwargs
        assert Path(actual_root) == root and phase == "screen" and pilot is True
        return {"native_logs_sha256": {str(pilot_path): collection.sha256(pilot_path)}}

    monkeypatch.setattr(
        collection,
        "verify_report",
        create_autospec(collection.verify_report, side_effect=pilot_verifier),
    )
    ps = create_autospec(
        postrun.subprocess.run,
        return_value=SimpleNamespace(stdout="", returncode=0),
    )
    monkeypatch.setattr(postrun.subprocess, "run", ps)
    token_path, _ = cached_proof(root, monkeypatch, failed)
    write(
        root / "setup/postrun_code_review.json",
        {
            "verdict": "PASS",
            "reviewer": "SOFTWARE FIXTURE ONLY",
            "sources_sha256": postrun.source_hashes(),
        },
    )
    prefix = {
        "passed": True,
        "n_contexts": 309,
        "contexts": [
            {
                "sample_id": source.id,
                "exact_context_sha256": source.metadata["exact_context_sha256"],
                "token_ids": [1, 2, 3],
                "n_prefix_tokens": 3,
                "prefix_token_ids_sha256": capacity.digest([1, 2, 3]),
            }
            for source in samples
        ],
    }
    write(paths["prefix"], prefix)
    producer_launch = paths["out"] / "fixture_launch_config.json"
    write(
        producer_launch,
        {
            "started_unix": start.timestamp(),
            "metadata": metadata,
            "critic_review": json.loads(review.read_text()),
            "critic_review_sha256": collection.sha256(review),
            "config": {
                "operation": "run",
                "phase": "screen",
                "arm": "B",
                "root": str(root),
                "model": collection.MODEL,
                "base_url": BASE_URL,
                "max_connections": 16,
                "pilot_limit": None,
                "review": str(review),
            },
        },
    )
    pids = {"supervisor_pid": 987654321, "worker_pid": 987654322}
    process_prefix = root / "screen_software_fixture_process"
    for key, suffix in (("supervisor_pid", "pid"), ("worker_pid", "worker.pid")):
        Path(f"{process_prefix}.{suffix}").write_text(str(pids[key]))
    log_path = root / "fixture_process.log"
    log_path.write_text(
        f"[supervisor-start] mode=screen pid={pids['supervisor_pid']}\n"
        f"[worker-start] mode=screen pid={pids['worker_pid']}\n"
        "RuntimeError: Unverified collection; raw errors/counts preserved at "
        f"{paths['result']}\n"
    )
    exit_path = Path(f"{process_prefix}.exit.json")
    write(
        exit_path,
        {
            "mode": "screen",
            **pids,
            "exit_code": 1,
            "cleanup": "no_live_members",
            "finished_unix": finished,
        },
    )
    launch_path = root / "fixture_owned_launch.json"
    write(
        launch_path,
        {
            "phase": "screen",
            "mode": "screen",
            "launch_id": "software_fixture",
            "started_utc": start.isoformat(),
            "log_path": str(log_path),
            **pids,
        },
    )

    def persist_original(*, include_success_review=True):
        audit = collection.audit_native_logs(
            [native],
            samples,
            metadata,
            OmegaConf.create({"model": collection.MODEL, "base_url": BASE_URL}),
            pilot=False,
            not_before=frozen["frozen_unix"],
        )
        write(paths["audit"], audit)
        design._write_jsonl_atomic(paths["rows"], collection.raw_rows([native]))
        report = {
            "schema_version": "context_risk_highrate_run_v1",
            "phase": "screen",
            "arm": "B",
            "model": collection.MODEL,
            "base_url": BASE_URL,
            "metadata": metadata,
            "epochs": 2,
            "is_pilot": False,
            "max_attempts": 10,
            "message_limit": 22,
            "requested_rollouts": 618,
            "realized_rollouts": audit["counts"]["realized"],
            "coverage_complete": audit["coverage_complete"],
            "verification_passed": False,
            "passed": False,
            **{
                key: audit[key]
                for key in (
                    "contexts",
                    "counts",
                    "by_condition",
                    "validation_issues",
                    "native_logs_sha256",
                )
            },
            "sources_sha256": metadata["sources_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "prefix_tokens_sha256": collection.sha256(paths["prefix"]),
            "rollouts_sha256": collection.sha256(paths["rows"]),
            "native_audit_sha256": collection.sha256(paths["audit"]),
            "launch_config_path": str(producer_launch),
            "launch_config_sha256": collection.sha256(producer_launch),
            "completed_unix": start.timestamp() + 4,
            "elapsed_seconds": 4,
        }
        write(paths["result"], report)
        if include_success_review:
            write(
                root / "screen_B/success_review.json",
                {
                    "verdict": "PASS",
                    "reviewer": "SOFTWARE FIXTURE ONLY",
                    "native_logs_sha256": audit["native_logs_sha256"],
                    "success_evidence": design.success_evidence(root, "screen"),
                },
            )
        return report

    report = persist_original()
    original_paths = [
        *paths.values(),
        native_path,
        pilot_path,
        producer_launch,
        exit_path,
        log_path,
        launch_path,
    ]
    original_hashes = {str(p): collection.sha256(p) for p in original_paths if p.is_file()}
    return SimpleNamespace(
        root=root,
        paths=paths,
        native=native,
        pilot=pilot,
        samples=samples,
        failed=failed,
        token_path=token_path,
        reader=reader,
        ps=ps,
        launch_path=launch_path,
        exit_path=exit_path,
        log_path=log_path,
        producer_launch=producer_launch,
        report=report,
        start=start,
        finished=finished,
        original_hashes=original_hashes,
        persist_original=persist_original,
    )


def test_full_recompute_settle_select_and_frozen_fresh_reload(cohort):
    c = cohort
    audit = postrun.recompute_report(c.root, "screen")
    expected_counts = {
        "planned": 618,
        "realized": 618,
        "success": 617,
        "failure": 0,
        "censored": 1,
        "missing": 0,
    }
    assert {key: audit["counts"][key] for key in expected_counts} == expected_counts
    assert audit["counts"]["unknown_outcome_rate_bounds"] == [617 / 618, 1.0]
    assert len(audit["requests"]) == 618 and len(audit["capacity_censors"]) == 1
    assert audit["original_collector_verification_passed"] is False
    terminal = postrun.settle(c.root, "screen", c.launch_path)
    assert terminal["original_exit_code"] == 1
    assert postrun.verify_report(c.root, "screen")["counts"] == audit["counts"]
    assert postrun.validate_terminal_process(c.root, "screen") == terminal
    selection = postrun.select(c.root)
    manifest, loaded, epochs = design.load_phase(c.root, "fresh")
    assert loaded == selection and epochs == 4
    assert len(collection.load_samples(manifest)) == 90
    assert postrun.validate_selection(c.root) == selection
    assert (selection["ranking"], selection["task_roles"]) == design.rank_tasks(audit["contexts"])
    assert {str(p): collection.sha256(p) for p in map(Path, c.original_hashes)} == c.original_hashes
    with pytest.raises(FileExistsError):
        postrun.settle(c.root, "screen", c.launch_path)
    with pytest.raises(FileExistsError):
        postrun.select(c.root)
    c.ps.assert_called_with(
        ["ps", "-eo", "pid=,pgid=,stat="], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "unrecognized_native", "unrecognized_sample"]
)
def test_incomplete_or_unrecognized_evidence_never_settles(cohort, mutation):
    c = cohort
    if mutation == "missing":
        c.native.samples.pop()
    elif mutation == "duplicate":
        c.native.samples.append(c.native.samples[-1].model_copy(deep=True))
    elif mutation == "unrecognized_native":
        collection.validate_native.side_effect = ValueError("fixture unrecognized native issue")
    else:
        c.failed.error.message = "unrecognized native failure"
    c.persist_original(include_success_review=False)
    with pytest.raises(ValueError):
        postrun.settle(c.root, "screen", c.launch_path)
    assert not (c.root / "screen_B/terminal_process.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "exit_zero",
        "exit_two",
        "wrong_exception",
        "live_worker",
        "verification_before_exit",
        "native_before_launch",
    ],
)
def test_terminal_exit_and_chronology_fail_closed(cohort, monkeypatch, mutation):
    c = cohort
    if mutation.startswith("exit_"):
        record = json.loads(c.exit_path.read_text())
        record["exit_code"] = 0 if mutation == "exit_zero" else 2
        write(c.exit_path, record)
    elif mutation == "wrong_exception":
        c.log_path.write_text(c.log_path.read_text() + "RuntimeError: unrelated crash\n")
    elif mutation == "live_worker":
        c.ps.return_value.stdout = "123 987654322 S\n"
    elif mutation == "verification_before_exit":
        monkeypatch.setattr(postrun.time, "time", lambda: c.finished - 0.01)
    else:
        c.native.stats.started_at = (c.start - timedelta(seconds=2)).isoformat()
    before = {
        str(p): collection.sha256(p)
        for p in (c.paths["result"], c.paths["audit"], c.paths["rows"], c.exit_path)
    }
    with pytest.raises(ValueError):
        postrun.settle(c.root, "screen", c.launch_path)
    assert not (c.root / "screen_B/terminal_process.json").exists()
    assert {p: collection.sha256(Path(p)) for p in before} == before


def test_report_mutation_after_initial_snapshot_is_rejected(cohort, monkeypatch):
    real = postrun.binding

    def changing(*args, **kwargs):
        result = real(*args, **kwargs)
        value = json.loads(cohort.paths["result"].read_text())
        value["elapsed_seconds"] += 1
        write(cohort.paths["result"], value)
        return result

    monkeypatch.setattr(postrun, "binding", create_autospec(real, side_effect=changing))
    with pytest.raises(ValueError, match="changed during validation"):
        postrun.recompute_report(cohort.root, "screen")


def test_process_log_mutation_at_final_ps_boundary_is_rejected(cohort):
    def changing(*args, **kwargs):
        cohort.log_path.write_text(cohort.log_path.read_text() + "late write\n")
        return SimpleNamespace(stdout="", returncode=0)

    cohort.ps.side_effect = changing
    with pytest.raises(ValueError, match="changed during validation"):
        postrun.settle(cohort.root, "screen", cohort.launch_path)
    assert not (cohort.root / "screen_B/terminal_process.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing_review", "review_coverage", "selection_chronology", "selection_provenance"],
)
def test_derived_selection_requires_review_and_frozen_provenance(cohort, mutation):
    c = cohort
    terminal = postrun.settle(c.root, "screen", c.launch_path)
    review_path = c.root / "screen_B/success_review.json"
    if mutation == "missing_review":
        review_path.unlink()
        with pytest.raises(FileNotFoundError):
            postrun.select(c.root)
    elif mutation == "review_coverage":
        review = json.loads(review_path.read_text())
        review["success_evidence"].pop(next(iter(review["success_evidence"])))
        write(review_path, review)
        with pytest.raises(ValueError, match=r"successful screening|Successful screening"):
            postrun.select(c.root)
    else:
        selection = postrun.select(c.root)
        if mutation == "selection_chronology":
            selection["selected_unix"] = terminal["verified_unix"] - 0.01
        else:
            selection["postrun_screen_audit_sha256"] = "0" * 64
        write(c.root / "selection.json", selection)
        with pytest.raises(ValueError, match="postrun extension"):
            postrun.validate_selection(c.root)


@pytest.mark.parametrize("mutation", [None, "different_exception", "unexpected_tail"])
def test_actual_hydra_formatter_footer_is_narrowly_supported(cohort, monkeypatch, mutation):
    """Run the installed formatter, including its real default exception footer."""
    monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)
    message = f"Unverified collection; raw errors/counts preserved at {cohort.paths['result']}"
    if mutation == "different_exception":
        message = "unrelated fixture exception"

    def failed_collector_fixture():
        raise RuntimeError(message)

    def run_job():
        # Hydra's formatter locates this frame name before printing user frames.
        failed_collector_fixture()

    stream = io.StringIO()
    with redirect_stderr(stream), pytest.raises(SystemExit) as stopped:
        run_and_report(run_job)
    assert stopped.value.code == 1
    rendered = stream.getvalue()
    assert rendered.endswith(
        "\nSet the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.\n"
    )
    markers = "\n".join(cohort.log_path.read_text().splitlines()[:2]) + "\n"
    cohort.log_path.write_text(markers + rendered)
    if mutation == "unexpected_tail":
        cohort.log_path.write_text(cohort.log_path.read_text() + "unrecognized extra failure\n")
    if mutation is None:
        terminal = postrun.settle(cohort.root, "screen", cohort.launch_path)
        assert terminal["original_exit_code"] == 1
    else:
        with pytest.raises(ValueError, match="exact frozen post-collection"):
            postrun.settle(cohort.root, "screen", cohort.launch_path)
        assert not (cohort.root / "screen_B/terminal_process.json").exists()


def test_clean_phase_routes_to_unchanged_original_consumers(tmp_path, monkeypatch):
    """Run each new dispatch body; original clean consumers have separate real-body tests."""
    write(tmp_path / "screen_B/run_result.json", {"passed": True})
    strict = create_autospec(collection.verify_report, return_value={"strict": True})
    selected = create_autospec(design.select, return_value={"selected": True})
    recorded = create_autospec(
        design.record_terminal_process, return_value={"strict_terminal": True}
    )
    checked = create_autospec(
        design.validate_terminal_process, return_value={"strict_process": True}
    )
    monkeypatch.setattr(collection, "verify_report", strict)
    monkeypatch.setattr(design, "select", selected)
    monkeypatch.setattr(design, "record_terminal_process", recorded)
    monkeypatch.setattr(design, "validate_terminal_process", checked)
    assert postrun.select(tmp_path) == {"selected": True}
    selected.assert_called_once_with(tmp_path)
    launch_path = tmp_path / "fixture_launch.json"
    assert postrun.settle(tmp_path, "screen", launch_path) == {"strict_terminal": True}
    recorded.assert_called_once_with(tmp_path, "screen", launch_path)
    write(tmp_path / "screen_B/terminal_process.json", {"schema_version": "original_fixture"})
    assert postrun.validate_terminal_process(tmp_path, "screen") == {"strict_process": True}
    checked.assert_called_once_with(tmp_path, "screen")
    write(postrun.audit_path(tmp_path, "screen"), {"unrecognized_derived_receipt": True})
    with pytest.raises(ValueError, match="also claim"):
        postrun.select(tmp_path)


@pytest.mark.parametrize("boundary", ["later_request", "terminal_ps"])
def test_previously_validated_capacity_tokens_remain_bound_to_final_return(
    cohort, monkeypatch, boundary
):
    """A later read/ps boundary must not invalidate an earlier token proof unnoticed."""
    token_path = cohort.token_path.with_suffix(".tokens.json")
    changed = []

    def corrupt_tokens():
        token_path.write_text('{"tokens":[1]}\n')
        changed.append(True)

    if boundary == "later_request":
        failed_index = next(
            index for index, sample in enumerate(cohort.native.samples) if sample.error
        )
        later = cohort.native.samples[failed_index + 1]
        calls = Counter()
        real = collection.requests

        def request_then_mutate(sample, model):
            result = real(sample, model)
            key = (sample.id, sample.epoch)
            calls[key] += 1
            # First pass reproduces the original audit. The second follows the
            # completed capacity check in extend_audit's real native-row loop.
            if key == (later.id, later.epoch) and calls[key] == 2:
                corrupt_tokens()
            return result

        monkeypatch.setattr(
            collection, "requests", create_autospec(real, side_effect=request_then_mutate)
        )
        with pytest.raises(ValueError, match="changed during validation"):
            postrun.recompute_report(cohort.root, "screen")
    else:
        postrun.settle(cohort.root, "screen", cohort.launch_path)

        def ps_then_mutate(*args, **kwargs):
            corrupt_tokens()
            return SimpleNamespace(stdout="", returncode=0)

        cohort.ps.side_effect = ps_then_mutate
        with pytest.raises(ValueError, match="changed during validation"):
            postrun.validate_terminal_process(cohort.root, "screen")
    assert changed
