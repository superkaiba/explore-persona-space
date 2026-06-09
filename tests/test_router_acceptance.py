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
    """A poll loop that never terminates raises RouterAcceptanceError."""
    plan = ra.build_live_command_plan(issue=402, backend="nibi", repo_root=Path("/repo"))
    fake_run, _rec = _make_fake_subprocess_run(
        launch_stdout=json.dumps({"ok": True}),
        poll_stdouts=[json.dumps({"status": "running"})] * 100,
        finalize_stdout="",
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


def test_run_live_lane_launch_crash_raises() -> None:
    """A non-zero exit code that isn't a router terminal (rc=2) is a
    real crash -- harness fails loud rather than silently passing."""
    plan = ra.build_live_command_plan(issue=403, backend="nibi", repo_root=Path("/repo"))
    fake_run, _rec = _make_fake_subprocess_run(
        launch_stdout="",
        launch_rc=137,  # killed by signal
        poll_stdouts=[],
        finalize_stdout="",
    )
    with pytest.raises(ra.RouterAcceptanceError, match="launch exited with rc=137"):
        ra.run_live_lane(
            plan,
            backend="nibi",
            issue=403,
            subprocess_run=fake_run,
            sleep_fn=lambda _s: None,
            now_fn=lambda: 0.0,
        )


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
    requested lane PASSes."""

    def _fake_events(_issue: int) -> list[dict[str, Any]]:
        return [
            {"kind": "epm:status-changed", "note": "old"},
            {
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "nibi", "requested_kind": "nibi"}),
            },
        ]

    io_ = ra.VerifierIO(read_events_jsonl=_fake_events)
    res = ra.check_routing_marker_posted(issue=700, expected_lane="nibi", io=io_)
    assert res.passed
    assert "chosen_kind=nibi" in res.detail


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
                "kind": "epm:backend-selected",
                "note": json.dumps({"chosen_kind": "nibi"}),
            }
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
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})}
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
    """Two finalize ticks for the same handle both return rc=0; the
    backend's teardown is called twice but absorbs the duplicate."""
    outcome = ra.negative_duplicate_cron_tick()
    assert outcome["rc_codes"] == [0, 0]
    # Both teardown invocations recorded -- the CLI does NOT de-dup;
    # the backend's ABC contract absorbs.
    assert outcome["teardown_count"] == 2
    # Both bodies are well-formed teardown responses.
    for body in outcome["bodies"]:
        assert body.get("ok") is True
        assert body.get("phase") == "teardown"


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
    assert body["rc_codes"] == [0, 0]


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
            {"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})}
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
        lambda _n: [{"kind": "epm:backend-selected", "note": json.dumps({"chosen_kind": "nibi"})}],
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
