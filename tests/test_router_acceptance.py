"""Unit tests for the slice-8 live-acceptance harness.

The live-per-lane invocations are driven by the orchestrator (the
``router_acceptance.py live --live`` path actually shells out). These
tests pin the harness logic that runs WITHOUT live infra:

1. Dataset resolution -- the smoke training mix resolves to the local
   sub-sample with a known row count + provenance string.
2. PASS checklist -- each per-check function (hf_artifact_present /
   git_figure_present / routing_marker_posted / clean_teardown) returns
   the expected PASS / FAIL given injected I/O.
3. Live command plan -- the launch / poll / finalize argv sequences
   match the SKILL.md Step 6b/6d/8 operational blocks.
4. Live driver -- the launch -> poll-loop -> finalize loop terminates
   on the expected statuses, with subprocess + sleep dependency-
   injected (no real ``dispatch_issue.py`` / ``backend_poll.py`` shell-outs).
5. Negative cases -- the three injected-mock scenarios resolve the
   router behaviour the harness's CLI asserts on.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Tests import the harness module via the scripts package alias the
# existing CLI tests use (``scripts.dispatch_issue``). Same pattern.
from scripts import router_acceptance as ra

# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def test_resolve_smoke_dataset_returns_reused_provenance(tmp_path: Path) -> None:
    """Reuse-first: when the local file exists, ``source == 'reused'``
    and the provenance string names the upstream HF file."""
    data_dir = tmp_path / "data" / "sft"
    data_dir.mkdir(parents=True)
    fpath = data_dir / "router_smoke_sft.jsonl"
    # 3 rows is enough to assert the row counter works.
    fpath.write_text('{"messages": []}\n{"messages": []}\n{"messages": []}\n')
    spec = ra.resolve_smoke_dataset(repo_root=tmp_path)
    assert spec.local_path == fpath
    assert spec.source == "reused"
    assert spec.row_count == 3
    assert "benign_sft_6k.jsonl" in spec.provenance
    assert "seed=0" in spec.provenance


def test_resolve_smoke_dataset_raises_on_missing_local(tmp_path: Path) -> None:
    """Missing local file is a loud failure -- no silent regeneration."""
    with pytest.raises(FileNotFoundError, match="smoke training mix not present"):
        ra.resolve_smoke_dataset(repo_root=tmp_path)


# ---------------------------------------------------------------------------
# Live command plan
# ---------------------------------------------------------------------------


def test_build_live_command_plan_matches_skill_md_step_6b() -> None:
    """The launch argv mirrors the SKILL.md Step 6b operational block."""
    plan = ra.build_live_command_plan(
        issue=300,
        backend="nibi",
        intent="lora-7b",
        repo_root=Path("/repo"),
    )
    assert plan.launch_argv[:5] == ["uv", "run", "python", "scripts/dispatch_issue.py", "launch"]
    # Required CLI args present + threaded backend override.
    assert "--issue" in plan.launch_argv
    assert "300" in plan.launch_argv
    assert "--intent" in plan.launch_argv
    assert "lora-7b" in plan.launch_argv
    assert "--backend" in plan.launch_argv
    assert "nibi" in plan.launch_argv
    # Hydra args every smoke launch carries.
    assert plan.launch_argv.count("--hydra") == len(ra.DEFAULT_SMOKE_HYDRA_ARGS)
    for hy in ra.DEFAULT_SMOKE_HYDRA_ARGS:
        assert hy in plan.launch_argv


def test_build_live_command_plan_auto_omits_backend_flag() -> None:
    """``--backend auto`` means "no --backend" -- matches the empty-
    frontmatter form the SKILL.md docs."""
    plan = ra.build_live_command_plan(issue=301, backend="auto", repo_root=Path("/repo"))
    assert "--backend" not in plan.launch_argv


def test_build_live_command_plan_poll_and_finalize_argv() -> None:
    """Poll + finalize argv match SKILL.md.

    The finalize argv MUST carry ``--skip-confirm-artifacts`` -- the
    acceptance harness verifies artifacts independently; the
    confirm_artifacts gate would FAIL on the no-sentinel smoke handle
    and skip teardown, leaking spend on the still-live VM / SLURM job.
    Pinning the flag here makes the always-teardown invariant a hard
    contract.
    """
    plan = ra.build_live_command_plan(issue=302, backend="gcp", repo_root=Path("/repo"))
    assert plan.poll_argv == [
        "uv",
        "run",
        "python",
        "scripts/backend_poll.py",
        "--issue",
        "302",
    ]
    assert plan.finalize_argv == [
        "uv",
        "run",
        "python",
        "scripts/dispatch_issue.py",
        "finalize",
        "--issue",
        "302",
        "--skip-confirm-artifacts",
    ]


# ---------------------------------------------------------------------------
# Dry-run output -- the orchestrator reads this verbatim
# ---------------------------------------------------------------------------


def test_emit_live_dry_run_lists_all_three_steps() -> None:
    """The dry-run output lists launch / poll / finalize commands so an
    operator can copy-paste them into a shell without reading the diff."""
    plan = ra.build_live_command_plan(issue=303, backend="nibi", repo_root=Path("/repo"))
    buf = io.StringIO()
    ra.emit_live_dry_run(plan, backend="nibi", issue=303, out=buf)
    output = buf.getvalue()
    # Each operational step is labelled + the command appears in the body.
    assert "Step 1: launch" in output
    assert "Step 2: poll" in output
    assert "Step 3: finalize" in output
    assert "dispatch_issue.py" in output
    assert "backend_poll.py" in output
    assert "--issue" in output
    assert "303" in output


# ---------------------------------------------------------------------------
# Live driver (subprocess + sleep dependency-injected)
# ---------------------------------------------------------------------------


@dataclass
class _RecordedProc:
    """Recorder for the fake subprocess runner."""

    argv_list: list[list[str]]


def _make_fake_subprocess_run(
    *,
    launch_stdout: str,
    launch_rc: int = 0,
    poll_stdouts: list[str],
    poll_rcs: list[int] | None = None,
    finalize_stdout: str,
    finalize_rc: int = 0,
) -> tuple[Any, _RecordedProc]:
    """Build a ``subprocess.run``-shaped fake that scripts the three CLIs.

    Each call's stdout is taken from the matching script list; ``rc`` is
    threaded through ``returncode`` so the harness's exit-code checks fire.
    """
    recorder = _RecordedProc(argv_list=[])
    poll_rcs = poll_rcs or [0] * len(poll_stdouts)
    poll_iter = iter(zip(poll_stdouts, poll_rcs, strict=True))

    def _fake_run(argv: list[str], **_kw: Any) -> subprocess.CompletedProcess:
        recorder.argv_list.append(list(argv))
        # Match by joined argv -- list ``in`` is element-equality, but
        # argv carries ``scripts/dispatch_issue.py`` so a substring check
        # needs a joined string.
        joined = " ".join(argv)
        if "dispatch_issue.py" in joined and "launch" in argv:
            return subprocess.CompletedProcess(
                args=argv, returncode=launch_rc, stdout=launch_stdout, stderr=""
            )
        if "backend_poll.py" in joined:
            stdout, rc = next(poll_iter)
            return subprocess.CompletedProcess(args=argv, returncode=rc, stdout=stdout, stderr="")
        if "dispatch_issue.py" in joined and "finalize" in argv:
            return subprocess.CompletedProcess(
                args=argv, returncode=finalize_rc, stdout=finalize_stdout, stderr=""
            )
        raise AssertionError(f"unexpected argv in fake subprocess.run: {argv!r}")

    return _fake_run, recorder


def test_run_live_lane_happy_path() -> None:
    """A normal launch -> poll(running) -> poll(done) -> finalize cycle
    returns the full transcript dict."""
    plan = ra.build_live_command_plan(issue=400, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True, "chosen_kind": "nibi", "issue": 400}),
        poll_stdouts=[
            json.dumps({"status": "running"}),
            json.dumps({"status": "done"}),
        ],
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
    )
    sleeps: list[float] = []
    outcome = ra.run_live_lane(
        plan,
        backend="nibi",
        issue=400,
        poll_interval_seconds=0.0,
        subprocess_run=fake_run,
        sleep_fn=sleeps.append,
        now_fn=lambda: 0.0,
    )
    assert outcome["phase"] == "complete"
    assert outcome["launch_body"]["chosen_kind"] == "nibi"
    assert len(outcome["poll_history"]) == 2
    assert outcome["poll_history"][-1]["status"] == "done"
    assert outcome["finalize_body"]["phase"] == "teardown"
    # 3 subprocess calls in order: launch, poll, poll, finalize = 4.
    assert len(rec.argv_list) == 4


def test_run_live_lane_router_terminal_short_circuits() -> None:
    """Launch rc=2 (router terminal) returns phase=launch_terminal with
    no poll calls -- the harness MUST NOT enter the poll loop on a
    failed launch."""
    plan = ra.build_live_command_plan(issue=401, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps(
            {
                "ok": False,
                "failure_class": "infra",
                "status": "blocked",
                "exception": "NoComputeAvailableError",
            }
        ),
        launch_rc=2,
        poll_stdouts=[],  # never called
        finalize_stdout="",
    )
    outcome = ra.run_live_lane(
        plan,
        backend="nibi",
        issue=401,
        poll_interval_seconds=0.0,
        subprocess_run=fake_run,
        sleep_fn=lambda _s: None,
        now_fn=lambda: 0.0,
    )
    assert outcome["phase"] == "launch_terminal"
    assert outcome["launch_body"]["failure_class"] == "infra"
    assert outcome["poll_history"] == []
    assert outcome["finalize_body"] is None
    # Only the launch ran -- no poll, no finalize.
    assert len(rec.argv_list) == 1


def test_run_live_lane_poll_timeout_raises() -> None:
    """A poll loop that never terminates raises RouterAcceptanceError
    AND -- critically -- runs cleanup teardown BEFORE re-raising. The
    live VM / SLURM job is UP after launch; without cleanup-teardown
    the harness exit on the raise leaks the live job and bills credit
    until something else reaps it. This is the GCP-credit-leak guard
    the round-1 always-teardown fix missed: the FINALIZE happy path
    got the unconditional teardown but the poll-timeout RAISE path
    still bailed without cleanup.
    """
    plan = ra.build_live_command_plan(issue=402, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True}),
        poll_stdouts=[json.dumps({"status": "running"})] * 100,
        # The cleanup-teardown invocation in the except branch lands
        # here -- it MUST run before the raise propagates.
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
        finalize_rc=0,
    )
    # Now starts at 0 then jumps past the timeout on the next call.
    times = iter([0.0, 0.0, 100.0])
    with pytest.raises(ra.RouterAcceptanceError, match="poll loop exceeded timeout"):
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=402,
            poll_interval_seconds=0.0,
            poll_timeout_seconds=1.0,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: next(times),
        )
    # Regression assertion: cleanup teardown ran. A ``finalize`` call
    # MUST appear in the recorded argv list (the except branch shells
    # out to ``plan.finalize_argv``), and it MUST carry
    # ``--skip-confirm-artifacts`` so it actually tears down the VM
    # rather than getting blocked at the confirm-artifacts gate.
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls, (
        "poll-timeout raise did NOT run cleanup teardown -- live VM/job leaked "
        "(GCP credit leak). The except branch in run_live_lane must shell out "
        "to plan.finalize_argv before re-raising."
    )
    assert "--skip-confirm-artifacts" in finalize_calls[0], (
        f"cleanup finalize argv missing --skip-confirm-artifacts (would block "
        f"on confirm_artifacts and skip teardown): {finalize_calls[0]!r}"
    )


def test_run_live_lane_launch_crash_runs_cleanup_and_warns_live_infra() -> None:
    """A non-zero exit code that isn't a router terminal (rc=2) is a
    real crash -- harness fails loud rather than silently passing.

    CRITICAL (C1): a launch crash does NOT prove nothing launched. The
    dispatch CLI can die AFTER provisioning (rc=4 from a post-launch
    raise, rc=137/130 from OOM-kill / SIGINT between ``gcloud create``
    rc=0 and the JSON print). So the harness MUST (a) attempt the same
    best-effort cleanup finalize the mid-flight branch runs (a no-op
    rc=2 ``missing_handle_sidecar`` when nothing was written; a real
    teardown when the sidecar DID land), and (b) raise a message that
    says LOUDLY a VM/job may be live, with the manual verification
    commands.
    """
    plan = ra.build_live_command_plan(issue=403, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout="",
        launch_rc=137,  # killed by signal -- possibly AFTER provisioning
        poll_stdouts=[],
        finalize_stdout=json.dumps(
            {"ok": False, "failure_class": "infra", "reason": "missing_handle_sidecar"}
        ),
        finalize_rc=2,  # harmless no-op shape when nothing launched
    )
    with pytest.raises(ra.RouterAcceptanceError) as excinfo:
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=403,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )
    msg = str(excinfo.value)
    assert "launch exited with rc=137" in msg
    # The message must scream live-infra-possible + carry the manual
    # verification commands for BOTH lanes.
    assert "MAY BE LIVE" in msg
    assert "gcloud compute instances list --filter=labels.eps-issue=403" in msg
    assert "squeue --name eps-issue-403" in msg
    # Cleanup finalize WAS attempted (with the always-teardown flag).
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls, (
        "launch-crash path did NOT attempt cleanup finalize -- a dispatch CLI "
        "that crashed AFTER provisioning leaks the live VM/job."
    )
    assert "--skip-confirm-artifacts" in finalize_calls[0]


def test_run_live_lane_launch_json_parse_failure_runs_cleanup_and_warns() -> None:
    """rc=0 with no parseable JSON shares the C1 exposure: the CLI can
    die after provisioning but before (or mid-) printing the JSON line.
    Cleanup finalize is attempted and the raise carries the live-infra
    warning."""
    plan = ra.build_live_command_plan(issue=405, backend="gcp", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout="INFO: not json at all\n",
        launch_rc=0,
        poll_stdouts=[],
        finalize_stdout=json.dumps(
            {"ok": False, "failure_class": "infra", "reason": "missing_handle_sidecar"}
        ),
        finalize_rc=2,
    )
    with pytest.raises(ra.RouterAcceptanceError) as excinfo:
        ra.run_live_lane(
            plan,
            backend="gcp",
            issue=405,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )
    msg = str(excinfo.value)
    assert "no parseable JSON" in msg
    assert "MAY BE LIVE" in msg
    assert "gcloud compute instances list --filter=labels.eps-issue=405" in msg
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls, "JSON-parse-failure path did NOT attempt cleanup finalize"


def test_run_live_lane_manual_attention_terminal_logs_orphan_no_cleanup(caplog) -> None:
    """rc=2 with ``ManualAttentionRequiredError`` means a launched SLURM
    job SURVIVED scancel (M1) -- the orphaned job id must be logged
    LOUDLY with the scancel instruction, and cleanup finalize must NOT
    fire (no sidecar exists on the free-lane park path; finalize
    genuinely cannot help)."""
    plan = ra.build_live_command_plan(issue=406, backend="auto", repo_root=Path("/repo"))
    note = (
        "failure_class: infra\n"
        "reason: manual_attention_required\n"
        "kind: nibi\n"
        "cluster: nibi\n"
        "orphaned_job_id: 7734001\n"
        "operator_action: verify job state, scancel if alive"
    )
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps(
            {
                "ok": False,
                "failure_class": "infra",
                "status": "blocked",
                "exception": "ManualAttentionRequiredError",
                "note": note,
            }
        ),
        launch_rc=2,
        poll_stdouts=[],
        finalize_stdout="",
    )
    with caplog.at_level(logging.ERROR, logger="router_acceptance"):
        outcome = ra.run_live_lane(
            plan,
            backend="auto",
            issue=406,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )
    assert outcome["phase"] == "launch_terminal"
    # Loud log carries the orphaned id + the operator instruction.
    log_text = caplog.text
    assert "7734001" in log_text, "orphaned job id missing from the loud log"
    assert "scancel" in log_text
    assert "ORPHANED" in log_text
    # No cleanup finalize fired -- only the launch subprocess ran.
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert not finalize_calls, (
        "manual-attention terminal must NOT fire cleanup finalize (no sidecar; "
        "the orphan needs a manual scancel, not a finalize no-op that implies handling)"
    )
    assert len(rec.argv_list) == 1


def test_run_live_lane_poll_crash_runs_cleanup_teardown() -> None:
    """THREE CONSECUTIVE poll-tick failures (``backend_poll.py`` rc!=0)
    declare the poll dead, raise, AND run cleanup teardown BEFORE the
    exception propagates.

    Sibling of the poll-timeout regression test: same leak class. The
    LAUNCH succeeded, so a live VM / SLURM job is UP when the poll
    loop dies -- a bare raise here (the pre-fix behavior) exits the
    harness with the job still billing. The 3-strike threshold is the
    Mn1 transient-blip guard: a SINGLE rc=1 tick must not tear down a
    healthy lane (see the retry test below).
    """
    plan = ra.build_live_command_plan(issue=404, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True}),
        poll_stdouts=[json.dumps({"status": "running"}), "", "", ""],
        poll_rcs=[0, 1, 1, 1],  # three CONSECUTIVE poll-tick crashes
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
        finalize_rc=0,
    )
    with pytest.raises(ra.RouterAcceptanceError, match=r"backend_poll\.py exited with rc=1"):
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=404,
            poll_interval_seconds=0.0,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )
    # Regression assertion: cleanup teardown ran, carrying
    # --skip-confirm-artifacts (the always-teardown contract), before
    # the RouterAcceptanceError propagated.
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls, (
        "poll-crash raise did NOT run cleanup teardown -- live VM/job leaked "
        "(GCP credit leak). The except branch in run_live_lane must shell out "
        "to plan.finalize_argv before re-raising."
    )
    assert "--skip-confirm-artifacts" in finalize_calls[0], (
        f"cleanup finalize argv missing --skip-confirm-artifacts (would block "
        f"on confirm_artifacts and skip teardown): {finalize_calls[0]!r}"
    )


def test_run_live_lane_transient_poll_blips_retry_without_teardown() -> None:
    """Mn1: two failed poll ticks followed by healthy ticks must NOT
    raise / tear the lane down -- the consecutive-failure counter
    resets on success and the run completes with exactly ONE finalize
    (the happy-path teardown, not a cleanup)."""
    plan = ra.build_live_command_plan(issue=407, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True, "chosen_kind": "nibi"}),
        poll_stdouts=[
            "",
            "garbled not json",
            json.dumps({"status": "running"}),
            json.dumps({"status": "done"}),
        ],
        poll_rcs=[1, 0, 0, 0],  # blip rc=1, blip bad-JSON, then healthy
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
        finalize_rc=0,
    )
    outcome = ra.run_live_lane(
        plan,
        backend="nibi",
        issue=407,
        poll_interval_seconds=0.0,
        subprocess_run=fake_run,
        sleep_fn=lambda _s: None,
        now_fn=lambda: 0.0,
    )
    assert outcome["phase"] == "complete"
    # Only the HEALTHY ticks land in the history.
    assert [p["status"] for p in outcome["poll_history"]] == ["running", "done"]
    # Exactly one finalize: the happy path. No cleanup teardown fired.
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert len(finalize_calls) == 1, (
        f"expected exactly the happy-path finalize, got {len(finalize_calls)} -- "
        "a transient poll blip must not trigger the cleanup-teardown branch"
    )


def test_run_live_lane_three_consecutive_failures_required_not_cumulative() -> None:
    """Mn1 counter semantics: failures separated by a healthy tick do
    NOT accumulate -- only CONSECUTIVE failures reach the threshold."""
    plan = ra.build_live_command_plan(issue=408, backend="nibi", repo_root=Path("/repo"))
    fake_run, _rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True}),
        poll_stdouts=[
            "",  # fail 1
            "",  # fail 2
            json.dumps({"status": "running"}),  # healthy -> reset
            "",  # fail 1 (again)
            "",  # fail 2
            json.dumps({"status": "done"}),  # healthy -> terminal
        ],
        poll_rcs=[1, 1, 0, 1, 1, 0],
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
        finalize_rc=0,
    )
    outcome = ra.run_live_lane(
        plan,
        backend="nibi",
        issue=408,
        poll_interval_seconds=0.0,
        subprocess_run=fake_run,
        sleep_fn=lambda _s: None,
        now_fn=lambda: 0.0,
    )
    assert outcome["phase"] == "complete"
    assert [p["status"] for p in outcome["poll_history"]] == ["running", "done"]


def test_run_live_lane_sidecar_write_error_surfaces_recovery_record(caplog) -> None:
    """M4.1: launch OK + ``sidecar_write_error`` (both sidecar writes
    failed) means a LIVE billing VM exists with no on-disk handle. Poll
    tick 1 then reads the missing-sidecar dead shape and finalize
    no-ops rc=2 -- pre-fix, that was a clean-looking terminal that
    swallowed the only recovery record into captured stdout. The
    harness must (a) ERROR-log the handle identity + the full launch
    body + the manual verification commands the moment the launch body
    carries ``sidecar_write_error``, and (b) raise from the finalize
    rc=2 with the live-infra warning attached."""
    plan = ra.build_live_command_plan(issue=410, backend="gcp", repo_root=Path("/repo"))
    launch_body = {
        "ok": True,
        "issue": 410,
        "chosen_kind": "gcp",
        "requested_kind": "gcp",
        "pod_name": "eps-issue-410",
        "job_id": "gcp-4242",
        "handle_sidecar_path": None,
        "sidecar_write_error": "OSError: [Errno 28] No space left on device",
        "handle": {
            "backend": "gcp",
            "cluster": None,
            "job_id": "gcp-4242",
            "pod_name": "eps-issue-410",
            "scratch_dir": "/scratch/eps-issue-410",
            "log_path": "/scratch/eps-issue-410/job.out",
            "extra": {"issue": 410},
        },
    }
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps(launch_body),
        poll_stdouts=[
            json.dumps(
                {
                    "status": "dead",
                    "failure_class": "infra",
                    "reason": "missing_handle_sidecar",
                }
            )
        ],
        finalize_stdout=json.dumps(
            {"ok": False, "failure_class": "infra", "reason": "missing_handle_sidecar"}
        ),
        finalize_rc=2,
    )
    with (
        caplog.at_level(logging.WARNING, logger="router_acceptance"),
        pytest.raises(ra.RouterAcceptanceError) as excinfo,
    ):
        ra.run_live_lane(
            plan,
            backend="gcp",
            issue=410,
            poll_interval_seconds=0.0,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )
    # (a) The ERROR log carries the job identity, the FULL launch body
    # (the recovery record, incl. the serialized handle), and the
    # manual verification commands.
    err_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR and "sidecar write FAILED" in r.getMessage()
    ]
    assert err_msgs, "sidecar_write_error on the launch body did NOT produce the loud ERROR log"
    msg = err_msgs[0]
    assert "gcp-4242" in msg, "ERROR log missing the job id (handle identity)"
    assert "eps-issue-410" in msg, "ERROR log missing the instance/job name"
    assert "/scratch/eps-issue-410" in msg, "ERROR log missing the serialized handle fields"
    assert "gcloud compute instances list --filter=labels.eps-issue=410" in msg
    # (b) The finalize rc=2 raise carries the live-infra warning -- the
    # lane must NOT look clean.
    raise_msg = str(excinfo.value)
    assert "MAY BE LIVE" in raise_msg
    assert "gcloud compute instances list --filter=labels.eps-issue=410" in raise_msg
    # The poll + finalize both actually ran (launch, poll, finalize,
    # plus the best-effort cleanup finalize from the except branch).
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls


def test_attempt_cleanup_finalize_benign_missing_sidecar_warns_not_errors(caplog) -> None:
    """Mn4.2: a cleanup finalize that no-ops on ``missing_handle_sidecar``
    (rc=2 -- every pre-provision launch crash hits this) must log a
    WARNING ("nothing on disk to tear down"), NOT the live-billing
    ERROR alarm."""
    plan = ra.build_live_command_plan(issue=411, backend="nibi", repo_root=Path("/repo"))

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=2,
            stdout=json.dumps(
                {"ok": False, "failure_class": "infra", "reason": "missing_handle_sidecar"}
            ),
            stderr="",
        )

    with caplog.at_level(logging.WARNING, logger="router_acceptance"):
        ra._attempt_cleanup_finalize(plan, subprocess_run=_fake_run, context="launch crash rc=137")
    billing_errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "STILL be billing" in r.getMessage()
    ]
    assert not billing_errors, (
        "benign missing_handle_sidecar no-op raised the live-billing ERROR alarm "
        "(false alarm on every pre-provision launch crash)"
    )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "nothing" in r.getMessage().lower()
    ]
    assert warnings, "benign no-op did not log the downgraded WARNING"
    assert "tear down" in warnings[0]


def test_attempt_cleanup_finalize_real_failure_still_errors(caplog) -> None:
    """Mn4.2 counterpart: any OTHER non-zero cleanup-finalize shape (a
    sidecar landed but teardown crashed, rc=1) keeps the LOUD
    live-billing ERROR."""
    plan = ra.build_live_command_plan(issue=412, backend="nibi", repo_root=Path("/repo"))

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout=json.dumps({"ok": False, "reason": "teardown_crashed"}),
            stderr="scancel: error: connection refused",
        )

    with caplog.at_level(logging.WARNING, logger="router_acceptance"):
        ra._attempt_cleanup_finalize(plan, subprocess_run=_fake_run, context="mid-flight raise")
    billing_errors = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "STILL be billing" in r.getMessage()
    ]
    assert billing_errors, "a REAL cleanup-finalize failure must keep the live-billing ERROR"


def test_parse_last_json_line_picks_last_blob() -> None:
    """Defensive against an upstream log line on stdout -- the harness
    reads the LAST parseable JSON object from stdout."""
    raw = 'INFO: stuff\n{garbage}\n{"ok": true, "chosen_kind": "nibi"}\n'
    body = ra._parse_last_json_line(raw)
    assert body == {"ok": True, "chosen_kind": "nibi"}


def test_parse_last_json_line_returns_none_on_no_json() -> None:
    assert ra._parse_last_json_line("just log lines\nno json here") is None
    assert ra._parse_last_json_line("") is None


# ---------------------------------------------------------------------------
# PASS checklist -- per-check unit tests
# ---------------------------------------------------------------------------


def test_check_hf_artifact_present_pass() -> None:
    """HF list shows >=1 file under the per-lane subfolder."""

    def _fake_list(repo_id: str, *, repo_type: str) -> list[str]:
        return [
            "router_acceptance/issue-500-nibi/adapter_model.safetensors",
            "router_acceptance/issue-500-nibi/adapter_config.json",
        ]

    io_ = ra.VerifierIO(list_hf_repo_files=_fake_list)
    res = ra.check_hf_artifact_present(
        issue=500, lane="nibi", repo_id="superkaiba1/explore-persona-space", io=io_
    )
    assert res.passed
    assert res.name == "hf_artifact_present"
    assert "2 file" in res.detail


def test_check_hf_artifact_present_fail_missing() -> None:
    """An HF list with no matching files yields a FAIL with prefix in detail."""

    def _fake_list(repo_id: str, *, repo_type: str) -> list[str]:
        return ["router_acceptance/issue-500-gcp/some.bin"]  # WRONG lane

    io_ = ra.VerifierIO(list_hf_repo_files=_fake_list)
    res = ra.check_hf_artifact_present(
        issue=500, lane="nibi", repo_id="superkaiba1/explore-persona-space", io=io_
    )
    assert not res.passed
    assert "issue-500-nibi" in res.detail


def test_check_hf_artifact_present_fail_on_transport_error() -> None:
    """Transport exception is surfaced as FAIL (per fail-loud rule)."""

    def _fake_list(repo_id: str, *, repo_type: str) -> list[str]:
        raise RuntimeError("HF Hub 503")

    io_ = ra.VerifierIO(list_hf_repo_files=_fake_list)
    res = ra.check_hf_artifact_present(
        issue=500, lane="nibi", repo_id="superkaiba1/explore-persona-space", io=io_
    )
    assert not res.passed
    assert "HF Hub 503" in res.detail


def test_check_git_figure_present_pass(tmp_path: Path) -> None:
    """Figure exists on disk AND git ls-files reports it tracked."""
    rel = "figures/issue_600/router_acceptance_nibi.png"
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"PNG")

    def _fake_git(_root: Path, paths: Any) -> set[str]:
        return set(paths)

    io_ = ra.VerifierIO(git_tracked=_fake_git)
    res = ra.check_git_figure_present(issue=600, lane="nibi", repo_root=tmp_path, io=io_)
    assert res.passed


def test_check_git_figure_present_fail_on_untracked(tmp_path: Path) -> None:
    """File exists but git ls-files reports nothing -- FAIL."""
    rel = "figures/issue_601/router_acceptance_nibi.png"
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"PNG")
    io_ = ra.VerifierIO(git_tracked=lambda _r, _p: set())
    res = ra.check_git_figure_present(issue=601, lane="nibi", repo_root=tmp_path, io=io_)
    assert not res.passed
    assert "NOT tracked" in res.detail


def test_check_git_figure_present_fail_on_missing_file(tmp_path: Path) -> None:
    """No file on disk -- FAIL before git is consulted."""
    io_ = ra.VerifierIO(git_tracked=lambda _r, _p: set())
    res = ra.check_git_figure_present(issue=602, lane="nibi", repo_root=tmp_path, io=io_)
    assert not res.passed
    assert "missing on disk" in res.detail


def test_check_routing_marker_posted_pass_explicit_lane() -> None:
    """A backend-selected marker whose body's chosen_kind matches the
    requested lane PASSes — WITH the ground-truth cluster-launched
    marker agreeing (per-cluster lanes require it since the issue-535
    mila→nibi misroute)."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {"kind": "epm:status-changed", "note": "old"},
            {
                "kind": "epm:cluster-launched",
                "note": json.dumps({"cluster": "nibi", "job_id": "123"}),
            },
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "nibi", "requested_kind": "nibi"}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=700, expected_lane="nibi", io=io_)
    assert res.passed
    assert "chosen_kind=nibi" in res.detail
    assert "ground-truth cluster=nibi" in res.detail


def test_check_routing_marker_posted_fails_on_misroute() -> None:
    """chosen_kind=mila but the job actually launched on nibi → FAIL.

    Regression test for the issue-535 live misroute: the router believed
    'mila' while the shared SlurmBackend's silent nibi default submitted
    the sbatch to Nibi; the old check compared only chosen_kind and the
    lane PASSed vacuously."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {
                "kind": "epm:cluster-launched",
                "note": json.dumps({"cluster": "nibi", "job_id": "15876369"}),
            },
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "mila", "requested_kind": "mila"}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=535, expected_lane="mila", io=io_)
    assert not res.passed
    assert "MISROUTE" in res.detail


def test_check_routing_marker_posted_fails_without_ground_truth_marker() -> None:
    """Per-cluster lane with NO cluster-launched marker → FAIL loud."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "mila", "requested_kind": "mila"}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=535, expected_lane="mila", io=io_)
    assert not res.passed
    assert "ground-truth" in res.detail


def test_check_routing_marker_posted_pass_auto_accepts_any_chosen() -> None:
    """``expected_lane='auto'`` accepts whatever the router picked."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "gcp", "requested_kind": None}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=701, expected_lane="auto", io=io_)
    assert res.passed


def test_check_routing_marker_posted_fail_on_mismatch() -> None:
    """Marker exists but chosen_kind disagrees with the requested lane."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "gcp"}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=702, expected_lane="nibi", io=io_)
    assert not res.passed
    assert "does NOT match" in res.detail


def test_check_routing_marker_posted_fail_on_missing() -> None:
    """No backend-selected marker on the task -- FAIL."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [{"kind": "epm:status-changed"}]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=703, expected_lane="nibi", io=io_)
    assert not res.passed
    assert "no 'epm:backend-selected' marker" in res.detail


def test_check_clean_teardown_slurm_pass() -> None:
    """squeue --name returns empty -> teardown verified clean."""
    io_ = ra.VerifierIO(squeue_by_name=lambda _alias, _name: [])
    res = ra.check_clean_teardown(
        issue=800, lane="nibi", io=io_, robot_alias_for_slurm="robot-nibi"
    )
    assert res.passed


def test_check_clean_teardown_slurm_fail_still_live() -> None:
    """Live job ids in squeue -> FAIL."""
    io_ = ra.VerifierIO(squeue_by_name=lambda _alias, _name: ["123456", "123457"])
    res = ra.check_clean_teardown(
        issue=801, lane="nibi", io=io_, robot_alias_for_slurm="robot-nibi"
    )
    assert not res.passed
    assert "still shows live ids" in res.detail
    assert "123456" in res.detail


def test_check_clean_teardown_slurm_misconfig_no_robot_alias() -> None:
    """A SLURM lane without a robot_alias is a harness misconfig -> FAIL."""
    io_ = ra.VerifierIO(squeue_by_name=lambda _alias, _name: [])
    res = ra.check_clean_teardown(issue=802, lane="nibi", io=io_, robot_alias_for_slurm=None)
    assert not res.passed
    assert "harness misconfiguration" in res.detail


def test_check_clean_teardown_gcp_pass() -> None:
    """gcloud list returns no instances -> teardown clean.

    ``gcloud_instances_list`` accepts the kw-only ``gcp_project`` /
    ``gcp_config_name`` overrides ``check_clean_teardown`` threads
    from the launch outcome -- the fake must accept them (even if it
    ignores their values).
    """

    def _fake(_filter: str, *, gcp_project: str | None = None, gcp_config_name: str | None = None):
        return []

    io_ = ra.VerifierIO(gcloud_instances_list=_fake)
    res = ra.check_clean_teardown(issue=803, lane="gcp", io=io_)
    assert res.passed


def test_check_clean_teardown_gcp_fail_live_vms() -> None:
    """gcloud list returns 1+ instances -> FAIL with names."""

    def _fake(_filter: str, *, gcp_project: str | None = None, gcp_config_name: str | None = None):
        return [{"name": "eps-issue-804"}]

    io_ = ra.VerifierIO(gcloud_instances_list=_fake)
    res = ra.check_clean_teardown(issue=804, lane="gcp", io=io_)
    assert not res.passed
    assert "eps-issue-804" in res.detail


def test_check_clean_teardown_gcp_threads_launch_project() -> None:
    """The launcher's project / config name reach the gcloud probe.

    A fresh ``GcpConfig()`` would default-empty project + fall back to
    the ambient ``CLOUDSDK_ACTIVE_CONFIG_NAME`` (my-goat manipulates
    it for personal use), which would grep the WRONG project. The
    verifier MUST use the same project the launcher targeted.
    """
    seen: dict[str, Any] = {}

    def _fake(filter_: str, *, gcp_project=None, gcp_config_name=None):
        seen["filter"] = filter_
        seen["project"] = gcp_project
        seen["config"] = gcp_config_name
        return []

    io_ = ra.VerifierIO(gcloud_instances_list=_fake)
    res = ra.check_clean_teardown(
        issue=805,
        lane="gcp",
        io=io_,
        gcp_project="eps-persona-gpu-jun2026",
        gcp_config_name="eps-gcp",
    )
    assert res.passed
    assert seen["project"] == "eps-persona-gpu-jun2026"
    assert seen["config"] == "eps-gcp"
    assert seen["filter"] == "labels.eps-issue=805"


def test_check_clean_teardown_slurm_uses_canonical_job_name() -> None:
    """check (d) greps the canonical pod_name, NOT ``eps-issue-<N>``.

    ``slurm.job_name`` appends ``-<plan_hash[:8]>`` when a plan hash
    is set; reconstructing the name from issue alone would grep the
    wrong name and false-PASS on a still-live job whose real name
    carries the hash suffix.
    """
    grepped: dict[str, str] = {}

    def _fake_squeue(alias: str, name: str) -> list[str]:
        grepped["alias"] = alias
        grepped["name"] = name
        return []

    io_ = ra.VerifierIO(squeue_by_name=_fake_squeue)
    res = ra.check_clean_teardown(
        issue=900,
        lane="nibi",
        io=io_,
        robot_alias_for_slurm="robot-nibi",
        canonical_job_name="eps-issue-900-a1b2c3d4",
    )
    assert res.passed
    assert grepped["name"] == "eps-issue-900-a1b2c3d4", (
        "verifier must grep the canonical pod_name from the launch outcome, "
        "not reconstruct eps-issue-<N>"
    )


def test_evaluate_pass_checklist_overall_pass(tmp_path: Path) -> None:
    """All four checks PASS -> LaneVerdict.passed = True."""
    rel = "figures/issue_900/router_acceptance_nibi.png"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"PNG")

    def _events(_n: int) -> list[dict[str, Any]]:
        return [
            {
                "kind": "epm:cluster-launched",
                "note": json.dumps({"cluster": "nibi", "job_id": "1"}),
            },
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "nibi"}),
            },
        ]

    io_ = ra.VerifierIO(
        list_hf_repo_files=lambda _r, repo_type: [
            "router_acceptance/issue-900-nibi/adapter_model.safetensors"
        ],
        git_tracked=lambda _r, paths: set(paths),
        read_events_jsonl=_events,
        squeue_by_name=lambda _a, _n: [],
    )
    verdict = ra.evaluate_pass_checklist(
        issue=900,
        lane="nibi",
        expected_lane="nibi",
        repo_root=tmp_path,
        hf_model_repo="x/y",
        io=io_,
        robot_alias_for_slurm="robot-nibi",
    )
    assert verdict.passed
    out = verdict.format()
    assert out.startswith("LANE nibi: PASS")
    # All four check names appear in the formatted output.
    for cname in (
        "hf_artifact_present",
        "git_figure_present",
        "routing_marker_posted",
        "clean_teardown",
    ):
        assert cname in out


def test_evaluate_pass_checklist_partial_fail_overall_fail(tmp_path: Path) -> None:
    """One FAIL (HF) is enough to fail the lane."""
    rel = "figures/issue_901/router_acceptance_nibi.png"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"PNG")
    io_ = ra.VerifierIO(
        list_hf_repo_files=lambda _r, repo_type: [],  # no HF artifact
        git_tracked=lambda _r, paths: set(paths),
        read_events_jsonl=lambda _n: [
            {"kind": "epm:cluster-launched", "note": json.dumps({"cluster": "nibi"})},
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})},
        ],
        squeue_by_name=lambda _a, _n: [],
    )
    verdict = ra.evaluate_pass_checklist(
        issue=901,
        lane="nibi",
        expected_lane="nibi",
        repo_root=tmp_path,
        hf_model_repo="x/y",
        io=io_,
        robot_alias_for_slurm="robot-nibi",
    )
    assert not verdict.passed
    assert verdict.format().startswith("LANE nibi: FAIL")


# ---------------------------------------------------------------------------
# Negative cases -- exercise the harness's claims about router behavior
# ---------------------------------------------------------------------------


def test_negative_free_busy_to_gcp_escalates_and_skips_runpod() -> None:
    """Free lane est-start is 24h, never reaches RUNNING -> router cancels
    -> escalates to GCP. RunPod.launch must NEVER be called."""
    outcome = ra.negative_free_busy_to_gcp()
    assert outcome["chosen_kind"] == "gcp"
    assert outcome["runpod_launches"] == 0
    assert outcome["nibi_launches"] == 1
    assert outcome["gcp_launches"] == 1


def test_negative_cancel_race_keeps_running_job() -> None:
    """Cancel-race detection KEEPS the racing job on the free lane."""
    outcome = ra.negative_cancel_race()
    assert outcome["chosen_kind"] == "nibi"
    assert outcome["runpod_launches"] == 0
    # The cancel state machine called teardown() to request the
    # scancel, which is expected -- the assertion is that the racing
    # job wins the lane.


def test_negative_duplicate_cron_tick_is_idempotent_at_cli_level() -> None:
    """The duplicate finalize tick is idempotent at the CLI level: the
    first tick tears down and renames the sidecar to ``*.finalized``
    (Mn4.3), so the second tick no-ops on the missing sidecar with the
    benign rc=2 shape — exactly ONE teardown reaches the backend, and
    neither tick crashes."""
    outcome = ra.negative_duplicate_cron_tick()
    assert outcome["rc_codes"] == [0, 2]
    # Mn4.3 stale-sidecar guard: the second tick must NOT re-execute
    # teardown against the retired handle.
    assert outcome["teardown_count"] == 1
    # First body: a well-formed teardown response that records the
    # sidecar retirement. Second body: the benign missing-sidecar no-op.
    first, second = outcome["bodies"]
    assert first.get("ok") is True
    assert first.get("phase") == "teardown"
    assert str(first.get("sidecar_finalized") or "").endswith(".finalized")
    assert second.get("ok") is False
    assert second.get("reason") == "missing_handle_sidecar"


# ---------------------------------------------------------------------------
# CLI smoke -- the entrypoint runs end-to-end on each subcommand
# ---------------------------------------------------------------------------


def test_cli_live_dry_run_prints_command_plan(tmp_path: Path, monkeypatch) -> None:
    """``router_acceptance live --backend nibi --issue N`` without
    ``--live`` prints the dry-run command sequence to stdout."""
    # Seed the smoke dataset where resolve_smoke_dataset() will find it.
    (tmp_path / "data" / "sft").mkdir(parents=True)
    (tmp_path / "data" / "sft" / "router_smoke_sft.jsonl").write_text('{"messages": []}\n')
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["live", "--issue", "999", "--backend", "nibi"])
    assert rc == 0
    out = buf.getvalue()
    assert "DRY RUN" in out
    assert "Step 1: launch" in out


def test_cli_negative_free_busy_to_gcp_asserts_and_exits_zero(monkeypatch, tmp_path) -> None:
    """The ``negative free-busy-to-gcp`` subcommand's harness-level
    assertion is the test claim -- it MUST exit 0 (no AssertionError)."""
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["negative", "free-busy-to-gcp"])
    assert rc == 0
    body = json.loads(buf.getvalue())
    assert body["chosen_kind"] == "gcp"


def test_cli_negative_cancel_race_asserts_and_exits_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["negative", "cancel-race"])
    assert rc == 0
    body = json.loads(buf.getvalue())
    assert body["chosen_kind"] == "nibi"


def test_cli_negative_duplicate_cron_tick_asserts_and_exits_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["negative", "duplicate-cron-tick"])
    assert rc == 0
    body = json.loads(buf.getvalue())
    # First tick succeeds; second tick is the benign rc=2 no-op (the
    # Mn4.3 sidecar rename retires the handle after the first teardown).
    assert body["rc_codes"] == [0, 2]


# ---------------------------------------------------------------------------
# Always-teardown invariant -- the spend-leak regression guard
# ---------------------------------------------------------------------------


def test_run_live_lane_raises_on_nonzero_finalize_rc() -> None:
    """A non-zero finalize rc means teardown may NOT have run.

    The harness MUST fail loud here, NOT silently exit 0. This is the
    spend-leak regression test: pre-fix, ``run_live_lane`` accepted
    rc=3 silently (treating "confirm_artifacts FAIL -> teardown
    skipped" as success), so a live VM / SLURM job could keep billing
    while the harness exited 0. Post-fix, ``build_live_command_plan``
    always passes ``--skip-confirm-artifacts`` so rc=3 should never
    happen on the live path -- and if it DOES (a regression in the
    dispatch CLI), the harness raises rather than masking it.
    """
    plan = ra.build_live_command_plan(issue=600, backend="nibi", repo_root=Path("/repo"))
    fake_run, _rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True, "chosen_kind": "nibi", "pod_name": "eps-issue-600"}),
        poll_stdouts=[json.dumps({"status": "done"})],
        finalize_stdout=json.dumps({"ok": False, "reason": "confirm_artifacts_failed"}),
        finalize_rc=3,  # the historic spend-leak path
    )
    with pytest.raises(
        ra.RouterAcceptanceError,
        match=r"finalize exited with rc=3.*teardown may NOT have run.*billing",
    ):
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=600,
            poll_interval_seconds=0.0,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )


def test_run_live_lane_raises_when_finalize_rc0_but_no_teardown_phase() -> None:
    """rc=0 is not enough -- the body MUST report phase=teardown.

    Defense-in-depth: even a rc=0 finalize that doesn't report
    ``phase=teardown`` means teardown was NOT actually executed (a
    future dispatch CLI regression). The harness refuses to claim
    success on the lane in that case.
    """
    plan = ra.build_live_command_plan(issue=601, backend="nibi", repo_root=Path("/repo"))
    fake_run, _rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True, "chosen_kind": "nibi"}),
        poll_stdouts=[json.dumps({"status": "done"})],
        # rc=0 but phase != teardown -- a regression shape.
        finalize_stdout=json.dumps({"ok": True, "phase": "confirm_artifacts_skipped"}),
        finalize_rc=0,
    )
    with pytest.raises(ra.RouterAcceptanceError, match=r"did NOT report.*phase=teardown.*billing"):
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=601,
            poll_interval_seconds=0.0,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )


def test_run_live_lane_passes_when_teardown_reported() -> None:
    """The happy-path PASS contract: rc=0 + phase=teardown -> ok."""
    plan = ra.build_live_command_plan(issue=602, backend="nibi", repo_root=Path("/repo"))
    fake_run, rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True, "chosen_kind": "nibi"}),
        poll_stdouts=[json.dumps({"status": "done"})],
        finalize_stdout=json.dumps({"ok": True, "phase": "teardown"}),
        finalize_rc=0,
    )
    outcome = ra.run_live_lane(
        plan,
        backend="nibi",
        issue=602,
        poll_interval_seconds=0.0,
        subprocess_run=fake_run,
        sleep_fn=lambda _s: None,
        now_fn=lambda: 0.0,
    )
    assert outcome["finalize_body"]["phase"] == "teardown"
    # The finalize argv MUST carry --skip-confirm-artifacts (read off
    # the recorded subprocess invocation, not just the plan).
    finalize_calls = [a for a in rec.argv_list if "finalize" in a]
    assert finalize_calls, "finalize was never called"
    assert "--skip-confirm-artifacts" in finalize_calls[0], (
        f"finalize argv missing --skip-confirm-artifacts (the always-teardown "
        f"contract): {finalize_calls[0]!r}"
    )


# ---------------------------------------------------------------------------
# Harness-produced figure -- check (b) evidence
# ---------------------------------------------------------------------------


def test_generate_acceptance_figure_writes_png_and_stages_it(tmp_path: Path) -> None:
    """generate_acceptance_figure writes the figure AND ``git add``s it."""
    staged: list[Path] = []

    def _fake_git_add(root: Path, abs_path: Path) -> None:
        staged.append(abs_path)

    out = ra.generate_acceptance_figure(
        issue=700,
        lane="nibi",
        elapsed_seconds=12.5,
        chosen_kind="nibi",
        repo_root=tmp_path,
        git_add=_fake_git_add,
    )
    assert out.exists()
    assert out.suffix == ".png"
    expected_rel = ra.ACCEPTANCE_FIGURE_PATH.format(issue=700, lane="nibi")
    assert out == tmp_path / expected_rel
    # ``git add`` was called with the file we just produced.
    assert staged == [out]


def test_generate_acceptance_figure_raises_on_git_add_failure(tmp_path: Path) -> None:
    """A git-add failure raises -- the figure check (b) MUST NOT silently FAIL."""

    def _broken(_root: Path, _path: Path) -> None:
        raise RuntimeError("git add boom")

    with pytest.raises(RuntimeError, match="git add boom"):
        ra.generate_acceptance_figure(
            issue=701,
            lane="nibi",
            elapsed_seconds=1.0,
            chosen_kind="nibi",
            repo_root=tmp_path,
            git_add=_broken,
        )


# ---------------------------------------------------------------------------
# evaluate_pass_checklist threads canonical job name + GCP project
# ---------------------------------------------------------------------------


def test_evaluate_pass_checklist_threads_canonical_job_name(tmp_path: Path) -> None:
    """``canonical_job_name`` reaches check (d)'s squeue probe."""
    rel = "figures/issue_910/router_acceptance_nibi.png"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"PNG")
    grepped: dict[str, str] = {}

    def _fake_squeue(_alias: str, name: str) -> list[str]:
        grepped["name"] = name
        return []

    io_ = ra.VerifierIO(
        list_hf_repo_files=lambda _r, repo_type: ["router_acceptance/issue-910-nibi/adapter.bin"],
        git_tracked=lambda _r, paths: set(paths),
        read_events_jsonl=lambda _n: [
            {"kind": "epm:cluster-launched", "note": json.dumps({"cluster": "nibi"})},
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})},
        ],
        squeue_by_name=_fake_squeue,
    )
    verdict = ra.evaluate_pass_checklist(
        issue=910,
        lane="nibi",
        expected_lane="nibi",
        repo_root=tmp_path,
        hf_model_repo="x/y",
        io=io_,
        robot_alias_for_slurm="robot-nibi",
        canonical_job_name="eps-issue-910-deadbeef",
    )
    assert verdict.passed
    assert grepped["name"] == "eps-issue-910-deadbeef"


def test_cli_verify_lane_runs_and_exits_per_verdict(monkeypatch, tmp_path: Path) -> None:
    """``verify-lane`` runs the checklist and exits 0 on PASS, 1 on FAIL.

    We monkeypatch the VerifierIO defaults to ALL-PASS so this is a
    pure CLI smoke test of the verify-lane subcommand.
    """
    rel = "figures/issue_950/router_acceptance_nibi.png"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"PNG")
    monkeypatch.chdir(tmp_path)

    # Patch the module-level defaults the production VerifierIO falls
    # back to (the fall-back lookup is dynamic so the patch is honored).
    monkeypatch.setattr(
        ra,
        "_default_list_hf_repo_files",
        lambda _r, repo_type: ["router_acceptance/issue-950-nibi/adapter.bin"],
    )
    monkeypatch.setattr(ra, "_default_git_tracked", lambda _r, paths: set(paths))
    monkeypatch.setattr(
        ra,
        "_default_read_events_jsonl",
        lambda _n: [
            {"kind": "epm:cluster-launched", "note": json.dumps({"cluster": "nibi"})},
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})},
        ],
    )
    monkeypatch.setattr(ra, "_default_squeue_by_name", lambda _a, _n: [])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(
            [
                "verify-lane",
                "--issue",
                "950",
                "--lane",
                "nibi",
                "--robot-alias",
                "robot-nibi",
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "LANE nibi: PASS" in out


# ---------------------------------------------------------------------------
# _cmd_live live path -- resolved-lane threading + env scoping
# ---------------------------------------------------------------------------


def _stub_cmd_live_rig(monkeypatch, tmp_path: Path, *, chosen_kind: str) -> dict[str, Any]:
    """Stub the heavy seams of ``_cmd_live`` for in-process CLI tests.

    Seeds the smoke dataset, chdirs into ``tmp_path``, and replaces
    ``run_live_lane`` / ``generate_acceptance_figure`` /
    ``evaluate_pass_checklist`` with recorders (no subprocesses, no
    matplotlib, no git). Returns the recorder dict the fakes fill in.
    """
    (tmp_path / "data" / "sft").mkdir(parents=True)
    (tmp_path / "data" / "sft" / "router_smoke_sft.jsonl").write_text('{"messages": []}\n')
    monkeypatch.chdir(tmp_path)
    # The per-issue lane lock lives under ``Path.home()/.eps-routing`` —
    # isolate it under tmp_path so tests never touch the real one.
    monkeypatch.setenv("HOME", str(tmp_path))
    rec: dict[str, Any] = {}

    def _fake_run_live_lane(
        plan: Any, *, backend: str, issue: int, launch_env: dict[str, str] | None = None, **_kw: Any
    ) -> dict[str, Any]:
        rec["launch_env"] = launch_env
        return {
            "phase": "complete",
            "launch_body": {
                "ok": True,
                "chosen_kind": chosen_kind,
                "pod_name": f"eps-issue-{issue}-cafe",
            },
            "poll_history": [{"status": "done"}],
            "finalize_body": {"ok": True, "phase": "teardown"},
        }

    def _fake_generate_acceptance_figure(
        *,
        issue: int,
        lane: str,
        elapsed_seconds: float,
        chosen_kind: str,
        repo_root: Path,
        git_add: Any = None,
    ) -> Path:
        rec["figure_lane"] = lane
        return repo_root / ra.ACCEPTANCE_FIGURE_PATH.format(issue=issue, lane=lane)

    def _fake_evaluate_pass_checklist(
        *, issue: int, lane: str, expected_lane: str, **_kw: Any
    ) -> ra.LaneVerdict:
        rec["checklist_lane"] = lane
        rec["checklist_expected_lane"] = expected_lane
        rec["checklist_artifact_lane"] = _kw.get("artifact_lane")
        return ra.LaneVerdict(lane=lane, checks=())

    monkeypatch.setattr(ra, "run_live_lane", _fake_run_live_lane)
    monkeypatch.setattr(ra, "generate_acceptance_figure", _fake_generate_acceptance_figure)
    monkeypatch.setattr(ra, "evaluate_pass_checklist", _fake_evaluate_pass_checklist)
    return rec


def test_cmd_live_refuses_concurrent_lane_for_same_issue(monkeypatch, tmp_path: Path) -> None:
    """Two concurrent --live lanes for the SAME issue clobber the
    per-issue handle sidecar (live incident, issue 535: a Mila lane
    overwrote the GCP lane's sidecar, finalize tore down the wrong
    handle, and a billing A100 VM was left RUNNING). The second lane
    must fail FAST with rc=1 and never reach launch."""
    import fcntl

    rec = _stub_cmd_live_rig(monkeypatch, tmp_path, chosen_kind="gcp")
    lock_path = tmp_path / ".eps-routing" / "issue-921.lane.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # simulate the in-flight lane
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ra.main(["live", "--issue", "921", "--backend", "gcp", "--live"])
        assert rc == 1
        assert "launch_env" not in rec, "second lane must never reach launch"
    finally:
        os.close(fd)


def test_cmd_live_figure_uses_resolved_lane_on_auto(monkeypatch, tmp_path: Path) -> None:
    """On ``--backend auto`` the figure write AND the check-(b) probe
    must BOTH use the RESOLVED lane (``chosen_kind`` from the launch
    body), never the literal ``auto``.

    Pre-fix, ``_cmd_live`` wrote ``router_acceptance_auto.png`` while
    ``evaluate_pass_checklist`` grepped
    ``router_acceptance_gcp.png`` -> check (b) FALSE-FAILed every
    successful auto->gcp/mila run.
    """
    rec = _stub_cmd_live_rig(monkeypatch, tmp_path, chosen_kind="gcp")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["live", "--issue", "920", "--backend", "auto", "--live"])
    assert rc == 0
    assert rec["figure_lane"] == "gcp", (
        f"figure written for lane {rec['figure_lane']!r}, expected the resolved "
        f"lane 'gcp' -- check (b) greps router_acceptance_gcp.png"
    )
    assert rec["checklist_lane"] == "gcp"
    assert rec["checklist_expected_lane"] == "gcp"


def test_cmd_live_scopes_persist_env_to_launch_subprocess(monkeypatch, tmp_path: Path) -> None:
    """The adapter-persist vars reach the launch subprocess via
    ``launch_env`` WITHOUT mutating the parent process's
    ``os.environ``.

    The test suite calls ``main([...])`` in-process, so a global
    ``os.environ`` mutation (the pre-fix behavior) leaks the per-lane
    subfolder across callers.
    """
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_SUBFOLDER", raising=False)
    rec = _stub_cmd_live_rig(monkeypatch, tmp_path, chosen_kind="nibi")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["live", "--issue", "921", "--backend", "nibi", "--live"])
    assert rc == 0
    env = rec["launch_env"]
    assert env is not None, "run_live_lane was not passed a launch_env"
    assert env["EPM_PERSIST_ADAPTER_HF_REPO"] == ra.DEFAULT_ACCEPTANCE_HF_REPO
    assert env["EPM_PERSIST_ADAPTER_SUBFOLDER"] == "router_acceptance/issue-921-nibi"
    # The parent process env stays untouched -- no cross-caller leak.
    assert "EPM_PERSIST_ADAPTER_HF_REPO" not in os.environ
    assert "EPM_PERSIST_ADAPTER_SUBFOLDER" not in os.environ


def test_cmd_live_auto_lane_env_subfolder_matches_artifact_probe(monkeypatch, tmp_path) -> None:
    """M3: on ``--backend auto`` the subfolder baked into the launch env
    and the subfolder check (a) probes must be ONE string.

    The env is built PRE-launch (resolved lane unknowable), so both
    sides use the literal ``auto``; checks (b)-(d) keep the resolved
    lane. Pre-fix, the env wrote ``...-auto`` while check (a) probed
    ``...-gcp`` -> every auto run false-FAILed check (a) AFTER the
    compute was already spent."""
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_HF_REPO", raising=False)
    monkeypatch.delenv("EPM_PERSIST_ADAPTER_SUBFOLDER", raising=False)
    rec = _stub_cmd_live_rig(monkeypatch, tmp_path, chosen_kind="gcp")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ra.main(["live", "--issue", "922", "--backend", "auto", "--live"])
    assert rc == 0
    env_subfolder = rec["launch_env"]["EPM_PERSIST_ADAPTER_SUBFOLDER"]
    probed_subfolder = ra.ACCEPTANCE_HF_SUBFOLDER.format(
        issue=922, lane=rec["checklist_artifact_lane"]
    )
    assert env_subfolder == probed_subfolder == "router_acceptance/issue-922-auto", (
        f"env wrote {env_subfolder!r} but check (a) probes {probed_subfolder!r} -- "
        "the artifact lane must be ONE source of truth (the pre-launch literal)"
    )
    # The figure / teardown / marker checks still use the RESOLVED lane.
    assert rec["checklist_lane"] == "gcp"
    assert rec["checklist_expected_lane"] == "gcp"


def test_evaluate_pass_checklist_artifact_lane_overrides_check_a_only(tmp_path) -> None:
    """``artifact_lane`` redirects ONLY the check-(a) HF prefix; checks
    (b)-(d) keep the resolved lane."""
    issue = 923
    rel = ra.ACCEPTANCE_FIGURE_PATH.format(issue=issue, lane="gcp")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(b"PNG")
    io_fake = ra.VerifierIO(
        list_hf_repo_files=lambda _r, repo_type: [
            f"router_acceptance/issue-{issue}-auto/adapter_model.safetensors"
        ],
        git_tracked=lambda _r, paths: set(paths),
        read_events_jsonl=lambda _n: [
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "gcp"})}
        ],
        squeue_by_name=lambda _a, _n: [],
        gcloud_instances_list=lambda _f, **_kw: [],
    )
    verdict = ra.evaluate_pass_checklist(
        issue=issue,
        lane="gcp",
        expected_lane="gcp",
        repo_root=tmp_path,
        hf_model_repo="superkaiba1/explore-persona-space",
        io=io_fake,
        artifact_lane="auto",
    )
    assert verdict.passed, verdict.format()
    # And WITHOUT artifact_lane the same fixture FAILS check (a) -- the
    # adapter sits under ...-auto, not ...-gcp.
    verdict_no_override = ra.evaluate_pass_checklist(
        issue=issue,
        lane="gcp",
        expected_lane="gcp",
        repo_root=tmp_path,
        hf_model_repo="superkaiba1/explore-persona-space",
        io=io_fake,
    )
    failed = {c.name for c in verdict_no_override.checks if not c.passed}
    assert failed == {"hf_artifact_present"}
