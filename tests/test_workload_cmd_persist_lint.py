"""Dispatch-time persist-evidence lint for ``--workload-cmd`` (#1800, incident #1739).

Pins the pure lint (`backends/issue_dispatch.lint_workload_cmd_persist_evidence`
+ `PERSIST_EVIDENCE_TOKENS`) and the `dispatch_issue.py launch` wiring
(WARN-only + `extra.workload_cmd_no_persist_evidence` marker flag, fail-soft
driver-script resolution — unresolvable → lint skipped with one note, the
`EPM_SKIP_WORKLOAD_CMD_PERSIST_LINT=1` kill switch, and the
no-strict-upgrade contract shared with the #1576 inline-interpreter arm).

Incident #1739 (2026-07-28): a GCP ``--workload-cmd`` run completed its
phases and approached grace-poweroff with ZERO artifacts on HF (all 7
expected prefixes MISS); ~2h of improvised recovery uploads raced the
poweroff clock. #1779 fixed the PLAN-time layer; this lint is the
dispatch-time backstop. Fixtures below derive from the incident-replay
calibration pair (plan #1800 §2): a persist-less driver shaped like the
pre-fix #1739 dispatcher (0 evidence hits → flags) and an evidence-bearing
one shaped like the post-fix tip (upload phase wired → clean).
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

from explore_persona_space.backends.issue_dispatch import (
    PERSIST_EVIDENCE_TOKENS,
    lint_workload_cmd_persist_evidence,
)
from tests.test_dispatch_issue_cli import (
    _backend_selected_extras,
    _build_mock_factory,
    _cd_to_tmp,
    _MockBackend,
    _pin_issue_branch_probe,  # noqa: F401 — autouse fixture: pins the #2161 issue-branch refusal probe EMPTY (fabricated issue numbers here have live origin/issue-<N> refs)
)

#: Synthetic driver shaped like the PRE-fix #1739 dispatcher
#: (``origin/issue-1739`` @ ``3bcc140bbd``): phases generate + judge + fit,
#: write local outputs, and wire NO upload/persist step anywhere.
PERSISTLESS_DRIVER = """\
#!/usr/bin/env bash
set -euo pipefail
echo "[phase=p0_stage]"
uv run python scripts/issue9990_gen.py --out data/issue_9990/gen
echo "[phase=p1_judge]"
uv run python scripts/issue9990_judge.py --in data/issue_9990/gen
echo "[phase=p2_fit]"
uv run python scripts/issue9990_fit.py --out eval_results/issue_9990/fit.json
echo "[phase=done]"
"""

#: Synthetic driver shaped like the POST-fix #1739 tip: the same chain plus
#: a wired upload phase (raw completions + eval JSONs to the HF data repo).
EVIDENCE_DRIVER = PERSISTLESS_DRIVER.replace(
    'echo "[phase=done]"',
    'echo "[phase=p3_upload]"\n'
    "uv run python scripts/issue9990_upload.py  "
    "# upload_raw_completions_to_data_repo + eval JSONs\n"
    'echo "[phase=done]"',
)


# ---------------------------------------------------------------------------
# Pure lint (plan #1800 AC-2 / AC-4)
# ---------------------------------------------------------------------------


def test_persistless_driver_flags() -> None:
    lint = lint_workload_cmd_persist_evidence(
        "bash scripts/issue9990_dispatch.sh", PERSISTLESS_DRIVER
    )
    assert lint.flagged is True
    assert lint.skipped is False
    assert lint.matched_tokens == ()


def test_evidence_in_script_text_not_flagged() -> None:
    lint = lint_workload_cmd_persist_evidence("bash scripts/issue9990_dispatch.sh", EVIDENCE_DRIVER)
    assert lint.flagged is False
    assert lint.skipped is False
    assert "upload" in lint.matched_tokens


def test_evidence_in_workload_cmd_itself_not_flagged() -> None:
    """Evidence in the COMMAND counts even when the driver script is persist-less."""
    cmd = "bash scripts/issue9990_dispatch.sh && uv run python scripts/upload_results.py"
    lint = lint_workload_cmd_persist_evidence(cmd, PERSISTLESS_DRIVER)
    assert lint.flagged is False
    assert "upload" in lint.matched_tokens


def test_none_script_text_skips_never_flags() -> None:
    lint = lint_workload_cmd_persist_evidence("bash scripts/issue9990_dispatch.sh", None)
    assert lint.skipped is True
    assert lint.flagged is False


def test_empty_cmd_skips() -> None:
    lint = lint_workload_cmd_persist_evidence("", PERSISTLESS_DRIVER)
    assert lint.skipped is True
    assert lint.flagged is False


def test_token_match_is_case_insensitive() -> None:
    script = PERSISTLESS_DRIVER + "\nuv run python -c 'model.PUSH_TO_HUB()'\n"
    lint = lint_workload_cmd_persist_evidence("bash scripts/x.sh", script)
    assert lint.flagged is False
    assert "push_to_hub" in lint.matched_tokens


def test_git_push_token_counts_as_evidence() -> None:
    script = PERSISTLESS_DRIVER + "\ngit push origin issue-9990\n"
    lint = lint_workload_cmd_persist_evidence("bash scripts/x.sh", script)
    assert lint.flagged is False
    assert "git push" in lint.matched_tokens


def test_token_composition_pinned() -> None:
    """The v1 composition is calibrated on the #1739 incident pair (plan §2)
    — a change is a deliberate recalibration, not a drive-by."""
    assert set(PERSIST_EVIDENCE_TOKENS) == {
        "upload",
        "push_to_hub",
        "hf_hub",
        "hfapi",
        "git push",
        "persist",
    }


# ---------------------------------------------------------------------------
# CLI seam (plan #1800 AC-3 / AC-4; mirrors tests/test_workload_cmd_env_lint.py)
# ---------------------------------------------------------------------------


def _persist_warnings(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "persist-evidence token" in rec.getMessage()
    ]


def _persist_skip_notes(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "persist-evidence lint (#1800) skipped" in rec.getMessage()
    ]


def _write_driver(tmp_path: Path, name: str, text: str) -> str:
    """Materialize a driver under ``<cwd>/scripts/`` so the resolver's
    working-tree fallback finds it (the git-show rungs fail soft at a
    non-repo tmp cwd)."""
    d = tmp_path / "scripts"
    d.mkdir(exist_ok=True)
    (d / name).write_text(text)
    return f"scripts/{name}"


def test_launch_persistless_script_warns_flags_marker_and_proceeds(
    monkeypatch, tmp_path, caplog
) -> None:
    """AC-3: flagged → LOUD warning + ``extra.workload_cmd_no_persist_evidence``
    on the posted marker; launch proceeds (WARN-only, exit 0)."""
    _cd_to_tmp(monkeypatch, tmp_path)
    rel = _write_driver(tmp_path, "phases_only_1800.sh", PERSISTLESS_DRIVER)
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(runpod=runpod, nibi=nibi, marker_posts=marker_posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--workload-cmd", f"bash {rel}"],
            backends_factory=factory,
        )
    assert rc == 0
    # #2054 runpod-first: the auto chain lands on DEFAULT_AUTO_LANE_ORDER[0]
    # (runpod); nibi (rung 3) is never consulted when the first rung launches.
    assert len(runpod.launches) == 1
    assert nibi.launches == []
    warnings = _persist_warnings(caplog)
    assert warnings, "expected a persist-evidence lint warning"
    joined = "\n".join(warnings)
    assert "#1739" in joined
    assert "EPM_SKIP_WORKLOAD_CMD_PERSIST_LINT=1" in joined
    extras = _backend_selected_extras(marker_posts)
    assert extras, "expected an epm:backend-selected post"
    assert all(e.get("workload_cmd_no_persist_evidence") is True for e in extras)


def test_launch_evidence_script_no_warning_no_flag(monkeypatch, tmp_path, caplog) -> None:
    _cd_to_tmp(monkeypatch, tmp_path)
    rel = _write_driver(tmp_path, "evidence_1800.sh", EVIDENCE_DRIVER)
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(
        runpod=_MockBackend(kind="runpod"), nibi=nibi, marker_posts=marker_posts
    )

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--workload-cmd", f"bash {rel}"],
            backends_factory=factory,
        )
    assert rc == 0
    assert not _persist_warnings(caplog)
    extras = _backend_selected_extras(marker_posts)
    assert extras
    assert all("workload_cmd_no_persist_evidence" not in e for e in extras)


def test_launch_unresolvable_script_skips_with_note(monkeypatch, tmp_path, caplog) -> None:
    """Fail-soft: an unresolvable driver script skips the lint with ONE note
    — never a warning, never a marker flag, never a refusal."""
    _cd_to_tmp(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO, logger="dispatch_issue")
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(runpod=runpod, nibi=nibi, marker_posts=marker_posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--workload-cmd",
                "bash scripts/nonexistent_1800_xyz.sh",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    # #2054 runpod-first: the auto chain lands on DEFAULT_AUTO_LANE_ORDER[0]
    # (runpod); nibi (rung 3) is never consulted when the first rung launches.
    assert len(runpod.launches) == 1
    assert nibi.launches == []
    assert not _persist_warnings(caplog)
    assert len(_persist_skip_notes(caplog)) == 1
    extras = _backend_selected_extras(marker_posts)
    assert all("workload_cmd_no_persist_evidence" not in e for e in extras)


def test_kill_switch_env_skips_persist_lint(monkeypatch, tmp_path, caplog) -> None:
    _cd_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("EPM_SKIP_WORKLOAD_CMD_PERSIST_LINT", "1")
    rel = _write_driver(tmp_path, "phases_only_1800.sh", PERSISTLESS_DRIVER)
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(
        runpod=_MockBackend(kind="runpod"), nibi=nibi, marker_posts=marker_posts
    )

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["launch", "--issue", "825", "--intent", "lora-7b", "--workload-cmd", f"bash {rel}"],
            backends_factory=factory,
        )
    assert rc == 0
    assert not _persist_warnings(caplog)
    extras = _backend_selected_extras(marker_posts)
    assert all("workload_cmd_no_persist_evidence" not in e for e in extras)


def test_launch_strict_flag_does_not_upgrade_persist_arm(monkeypatch, tmp_path, caplog) -> None:
    """``--strict-workload-cmd-env`` stays lane-env-scoped (the #1576
    contract): a flagged persist lint still launches at exit 0."""
    _cd_to_tmp(monkeypatch, tmp_path)
    rel = _write_driver(tmp_path, "phases_only_1800.sh", PERSISTLESS_DRIVER)
    runpod = _MockBackend(kind="runpod")
    nibi = _MockBackend(kind="nibi")
    marker_posts: list[dict] = []
    factory = _build_mock_factory(runpod=runpod, nibi=nibi, marker_posts=marker_posts)

    from scripts.dispatch_issue import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "launch",
                "--issue",
                "825",
                "--intent",
                "lora-7b",
                "--strict-workload-cmd-env",
                "--workload-cmd",
                f"bash {rel}",
            ],
            backends_factory=factory,
        )
    assert rc == 0
    # #2054 runpod-first: the auto chain lands on DEFAULT_AUTO_LANE_ORDER[0]
    # (runpod); nibi (rung 3) is never consulted when the first rung launches.
    assert len(runpod.launches) == 1
    assert nibi.launches == []
    assert _persist_warnings(caplog)
    extras = _backend_selected_extras(marker_posts)
    assert all(e.get("workload_cmd_no_persist_evidence") is True for e in extras)
