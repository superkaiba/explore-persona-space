"""Tests for the branch-aware preflight git check + fail-loud CLI (#554).

Covers the two fail-loud defects fixed in task #554:

1. ``check_git_status`` is branch-aware: ``main`` behind ``origin/main`` is
   still an ERROR (byte-identical message — agent specs pattern-match it);
   a feature branch is compared against its OWN ``origin/<branch>`` ref,
   with origin/main divergence demoted to an informational WARNING; a
   failed ``git fetch origin`` is an ERROR on a feature branch (stale refs
   make behind-own == 0 meaningless) but only a WARNING on ``main``.
2. The bare (non ``--json``) CLI and ``require_preflight()`` print the
   failing summary to a real stream instead of a handler-less
   ``logger.info()`` that emits zero bytes.

Plus the #2107 scoped-fetch pins: the fetch is scoped to the refs the check
compares (``_fake_git_run`` raises on any bare full-remote fetch), branch
resolution precedes the fetch, the fetch timeout is 90 s, and an unpushed
local branch degrades to the standing WARNING via a main-only re-fetch.

Simulation strategy: monkeypatch ``preflight._run`` with a canned dispatcher
(the established pattern from tests/test_preflight_disk.py) and
``preflight.is_cluster_env`` — every git interaction goes through the single
``_run`` seam. CLI tests monkeypatch ``preflight.preflight_check`` to return
canned reports and call ``main()`` directly with capsys.
"""

import json
import logging
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import preflight
from explore_persona_space.orchestrate.preflight import (
    PreflightReport,
    check_git_status,
    main,
    require_preflight,
)

ROOT = Path("/fake")


@pytest.fixture(autouse=True)
def _prune_stale_root_handlers():
    """Drop root-logger handlers whose stream is already closed (task #1654).

    Cross-test pollution: a collection-time ``logging.basicConfig`` (e.g.
    ``scripts/issue1112_rankem_dispatch.py`` exec'd at import by
    ``tests/test_issue1112_rankem_capture_tf.py``) installs a root
    ``StreamHandler`` bound to pytest's capture stream; a later test's
    teardown leaves that stream CLOSED while the handler survives on the
    root logger. ``require_preflight``'s ``logger.info`` then propagates to
    root, the stale handler's emit raises on the closed stream, and
    ``logging``'s ``handleError`` prints ``--- Logging error ---`` to the
    CURRENT (capsys-captured) stderr — breaking ``captured.err == ""``
    below. Reproduced by the 3-file composition
    ``pytest tests/test_issue1112_rankem_capture_tf.py
    tests/test_issue1547_callsite_routing.py
    tests/test_preflight_git_and_cli.py``; passes standalone. Pruning only
    closed-stream handlers keeps deliberately-configured live logging
    intact.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if stream is not None and getattr(stream, "closed", False):
            root.removeHandler(handler)
    yield


def _fake_git_run(
    *,
    branch="issue-554",
    branch_rc=0,
    porcelain="",
    own_ref_exists=True,
    behind_own="0",
    ahead_own="0",
    behind_main="0",
    fetch_rc=0,
    fetch_missing_branch_ref=False,
    calls=None,
    timeouts=None,
):
    """Canned ``_run`` dispatcher covering every git argv check_git_status issues.

    STRICT on fetch scope (#2107): a fetch argv ending at bare ``origin`` (no
    refspec) raises — the production code must never issue a full-remote fetch
    again (~1,700 remote heads make it un-finishable inside any timeout).
    ``timeouts`` (when given) captures each call's timeout kwarg, index-aligned
    with ``calls``. ``fetch_missing_branch_ref=True`` models an UNPUSHED local
    branch: any fetch argv naming the branch returns rc=128 with git's
    ``couldn't find remote ref`` stderr; a main-only fetch succeeds.
    """
    full_own = f"refs/remotes/origin/{branch}"

    def fake_run(cmd, timeout=10):
        if calls is not None:
            calls.append(cmd)
        if timeouts is not None:
            timeouts.append(timeout)
        if "status" in cmd and "--porcelain" in cmd:
            return 0, porcelain, ""
        if "fetch" in cmd:
            if cmd[-1] == "origin":
                raise AssertionError(f"unscoped full-remote fetch: {cmd}")
            if fetch_missing_branch_ref:
                if branch in cmd:
                    return 128, "", f"fatal: couldn't find remote ref {branch}"
                return 0, "", ""
            return fetch_rc, "", ("" if fetch_rc == 0 else "fatal: unable to access remote")
        if cmd[-2:] == ["--abbrev-ref", "HEAD"]:
            if branch_rc != 0:
                return branch_rc, "", "fatal: ambiguous argument 'HEAD'"
            return 0, branch, ""
        if "--verify" in cmd:
            return (0 if own_ref_exists else 1), "", ""
        if "rev-list" in cmd and cmd[-1] == "HEAD..origin/main":
            return 0, behind_main, ""
        if "rev-list" in cmd and cmd[-1] == f"HEAD..{full_own}":
            return (0, behind_own, "") if own_ref_exists else (128, "", "fatal: bad revision")
        if "rev-list" in cmd and cmd[-1] == f"{full_own}..HEAD":
            return (0, ahead_own, "") if own_ref_exists else (128, "", "fatal: bad revision")
        raise AssertionError(f"unexpected git cmd: {cmd}")

    return fake_run


def _patch_git(monkeypatch, *, cluster=False, **fake_kwargs):
    monkeypatch.setattr(preflight, "is_cluster_env", lambda: cluster)
    monkeypatch.setattr(preflight, "_run", _fake_git_run(**fake_kwargs))


# ── branch-aware behind-remote check ─────────────────────────────────────────


def test_issue_branch_at_pushed_tip_passes(monkeypatch):
    """Criterion 1: issue branch at the tip of its pushed origin ref → PASS."""
    _patch_git(monkeypatch, branch="issue-554", behind_own="0", behind_main="977")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "origin/main" in report.warnings[0]


def test_issue_branch_behind_own_origin_errors(monkeypatch):
    """Criterion 2: behind the branch's OWN origin ref → ERROR, after a fetch."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch="issue-554", behind_own="3", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "behind origin/issue-554" in report.errors[0]
    assert "git pull --ff-only" in report.errors[0]
    # The behind-own guarantee is only as fresh as the fetch: the exact fetch
    # argv (scoped to branch + main, #2107) must be issued BEFORE the own-ref
    # rev-list (an implementation that drops the fetch must fail here).
    fetch_argv = ["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "issue-554", "main"]
    own_revlist_argv = [
        "git",
        "-C",
        str(ROOT),
        "rev-list",
        "--count",
        "HEAD..refs/remotes/origin/issue-554",
    ]
    assert fetch_argv in calls
    assert own_revlist_argv in calls
    assert calls.index(fetch_argv) < calls.index(own_revlist_argv)


def test_issue_branch_diverged_from_own_origin_errors(monkeypatch):
    """Behind AND ahead of the own origin ref → diverged ERROR (ff-only would fail)."""
    _patch_git(monkeypatch, branch="issue-554", behind_own="3", ahead_own="2")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "diverged from origin/issue-554" in report.errors[0]
    assert "reconcile" in report.errors[0]


def test_issue_branch_ahead_of_own_origin_warns(monkeypatch):
    """Committed-but-unpushed local commits → WARNING, not silent PASS."""
    _patch_git(monkeypatch, branch="issue-554", behind_own="0", ahead_own="2")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert any("ahead of origin/issue-554" in w for w in report.warnings)


def test_feature_branch_fetch_failure_errors(monkeypatch):
    """A failed fetch on a feature branch is an ERROR — behind-own=0 against
    stale refs would otherwise read as a silent false PASS."""
    _patch_git(monkeypatch, branch="issue-554", fetch_rc=128, behind_own="0")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "git fetch origin failed" in report.errors[0]
    assert "origin/issue-554" in report.errors[0]


def test_main_fetch_failure_warns_not_errors(monkeypatch):
    """Criterion 3: a failed fetch on main keeps the prior gate decision (WARNING)."""
    _patch_git(monkeypatch, branch="main", fetch_rc=128, behind_main="0")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert any("git fetch origin failed" in w for w in report.warnings)


def test_main_behind_origin_main_error_byte_identical(monkeypatch):
    """Criterion 3: the main-behind ERROR string is exactly the legacy one —
    agent specs tolerance-match it verbatim."""
    _patch_git(monkeypatch, branch="main", behind_main="2")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.errors == ["Local is 2 commit(s) behind origin/main. Run: git pull origin main"]


def test_main_up_to_date_passes(monkeypatch):
    _patch_git(monkeypatch, branch="main", behind_main="0")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_unpushed_branch_warns_not_errors(monkeypatch):
    """No pushed origin/<branch> ref → WARNING (cannot verify), never an ERROR."""
    _patch_git(monkeypatch, branch="issue-554", own_ref_exists=False)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert any("no pushed origin/" in w for w in report.warnings)


def test_detached_head_warns_not_errors(monkeypatch):
    """Detached HEAD (pinned-SHA checkout) is a deliberate state → warnings only."""
    _patch_git(monkeypatch, branch="HEAD", behind_main="5")
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert any("behind origin/main" in w for w in report.warnings)
    assert "detached" in report.git_status.lower()


def test_feature_branch_fetch_scoped_to_branch_and_main(monkeypatch):
    """#2107 A1: on a feature branch the single fetch names exactly branch + main."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch="issue-554", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    fetches = [c for c in calls if "fetch" in c]
    assert fetches == [["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "issue-554", "main"]]


def test_main_branch_fetch_scoped_to_main(monkeypatch):
    """#2107 A1: on main the single fetch names main only."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch="main", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    fetches = [c for c in calls if "fetch" in c]
    assert fetches == [["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "main"]]


def test_detached_head_fetch_scoped_to_main(monkeypatch):
    """#2107 A1: detached HEAD fetches main only (no own ref exists to compare)."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch="HEAD", behind_main="5", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    fetches = [c for c in calls if "fetch" in c]
    assert fetches == [["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "main"]]


def test_fetch_timeout_is_90s(monkeypatch):
    """#2107 A2: the scoped fetch passes timeout=90 (grounded, plan §5)."""
    calls: list[list[str]] = []
    timeouts: list[int] = []
    _patch_git(monkeypatch, branch="issue-554", calls=calls, timeouts=timeouts)
    report = PreflightReport()
    check_git_status(report, ROOT)
    fetch_timeouts = [t for c, t in zip(calls, timeouts, strict=True) if "fetch" in c]
    assert fetch_timeouts == [90]


def test_branch_resolved_before_fetch(monkeypatch):
    """#2107 A3: branch resolution precedes the fetch (the refspec depends on it)."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch="issue-554", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    branch_idx = next(i for i, c in enumerate(calls) if c[-2:] == ["--abbrev-ref", "HEAD"])
    fetch_idx = next(i for i, c in enumerate(calls) if "fetch" in c)
    assert branch_idx < fetch_idx


def test_unpushed_branch_scoped_fetch_degrades_to_warning(monkeypatch):
    """#2107 A5: an unpushed local branch fails the two-ref fetch on the missing
    branch ref; the degrade path re-fetches main only and the standing
    unpushed-branch WARNING fires — never a fetch-failed ERROR."""
    calls: list[list[str]] = []
    _patch_git(
        monkeypatch,
        branch="issue-554",
        own_ref_exists=False,
        fetch_missing_branch_ref=True,
        calls=calls,
    )
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert any("no pushed origin/" in w for w in report.warnings)
    fetches = [c for c in calls if "fetch" in c]
    assert len(fetches) == 2
    assert fetches[0][-3:] == ["origin", "issue-554", "main"]
    assert fetches[1][-2:] == ["origin", "main"]
    assert "issue-554" not in fetches[1]


def test_branch_resolution_failure_skips_fetch(monkeypatch):
    """#2107: rev-parse failure returns (with the existing WARNING) BEFORE any
    fetch — the behind checks, the fetch's only consumers, are skipped anyway."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, branch_rc=128, calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert any("could not determine current branch" in w for w in report.warnings)
    assert [c for c in calls if "fetch" in c] == []


def test_cluster_skips_fetch_and_behind(monkeypatch):
    """Criterion 6: the cluster early-return is untouched — no fetch, no ref math."""
    calls: list[list[str]] = []
    _patch_git(monkeypatch, cluster=True, calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    for cmd in calls:
        assert "fetch" not in cmd
        assert "rev-list" not in cmd
        assert "rev-parse" not in cmd
    assert report.git_status.endswith("(cluster — skipped fetch)")


# ── require_preflight fail-loud ──────────────────────────────────────────────


def _fail_report():
    report = PreflightReport()
    report.add_error("BOOM")
    return report


def test_require_preflight_failure_prints_summary_to_stderr(monkeypatch, capsys):
    """Criterion 4: a FAILing require_preflight prints the summary on stderr
    exactly once (no double-print for configured-logging consumers)."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: _fail_report())
    with pytest.raises(SystemExit) as excinfo:
        require_preflight()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.count("Pre-flight Check: FAIL") == 1
    assert "BOOM" in captured.err
    assert "Pre-flight Check" not in captured.out


def test_require_preflight_success_no_stderr(monkeypatch, capsys, caplog):
    """PASS path unchanged: report returned, nothing on stderr, summary still
    emitted via logger.info for configured-logging consumers."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: PreflightReport())
    with caplog.at_level(logging.INFO, logger="explore_persona_space.orchestrate.preflight"):
        report = require_preflight()
    assert report.ok is True
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Pre-flight Check: PASS" in caplog.text


# ── CLI (main) ───────────────────────────────────────────────────────────────


def test_main_bare_failure_prints_summary_and_errors(monkeypatch, capsys):
    """Criterion 4: bare-mode FAIL prints the summary on stdout and one
    attributable per-error line on stderr before exiting non-zero."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: _fail_report())
    rc = main(["--no-gpu"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Pre-flight Check: FAIL" in captured.out
    assert "preflight ERROR: BOOM" in captured.err


def test_main_bare_success_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: PreflightReport())
    rc = main(["--no-gpu"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Pre-flight Check: PASS" in captured.out


def test_main_json_stdout_is_single_pretty_json(monkeypatch, capsys):
    """Criterion 6: --json stdout is exactly one pretty-printed JSON object
    with the 16 documented keys and nothing else (gotchas.md contract)."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: _fail_report())
    rc = main(["--json", "--no-gpu"])
    assert rc == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {
        "ok",
        "errors",
        "warnings",
        "gpu_info",
        "disk_free_gb",
        "disk_probed_headroom_gb",
        "disk_headroom_basis",
        # task #564's HF public-storage headroom guard added these three
        # alongside the #554 stdout-purity contract (concurrent merges).
        "hf_storage_used_tb",
        "hf_storage_ceiling_tb",
        "hf_storage_basis",
        # task #2097's shared-VM data-disk floor added this one (None when
        # /mnt/eps-data is not a live mount — pods/GCE/SLURM).
        "data_disk_used_pct",
        # task #2280's shared-VM swap-state check added these two (None off
        # the shared VM / when meminfo is unreadable; WARN-only — never
        # flips ok).
        "swap_total_gb",
        "swap_free_pct",
        "git_status",
        "env_synced",
        # task #2360's venv import-health check added this one (diagnostic
        # routing only — "" when the check never ran; report.ok / the CLI rc
        # stay authoritative).
        "venv_import_verdict",
    }
    assert payload["ok"] is False
    assert payload["errors"] == ["BOOM"]
    assert len(captured.out.strip().splitlines()) > 1  # pretty-printed, multi-line
    assert captured.err == ""


def test_main_json_success(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: PreflightReport())
    rc = main(["--json", "--no-gpu"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_main_json_pipeline_check_keeps_stdout_pure(monkeypatch, capsys):
    """Pipeline-check status prints AND pytest output stay off stdout in
    --json mode: whole stdout is one parseable JSON object, pytest text on
    stderr (concern json-pipeline-stdout-purity, #554 round 2)."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: PreflightReport())
    monkeypatch.setattr(preflight, "_run", lambda cmd, timeout=10: (0, "PYTEST STDOUT", ""))
    rc = main(["--json", "--no-gpu", "--pipeline-check"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # whole stdout is still one JSON object
    assert payload["ok"] is True
    assert "PYTEST STDOUT" not in captured.out
    assert "PYTEST STDOUT" in captured.err


def test_main_bare_pipeline_check_pytest_stdout_stays_on_stdout(monkeypatch, capsys):
    """Bare mode unchanged: pipeline-check pytest stdout still lands on stdout."""
    monkeypatch.setattr(preflight, "preflight_check", lambda **kwargs: PreflightReport())
    monkeypatch.setattr(preflight, "_run", lambda cmd, timeout=10: (0, "PYTEST STDOUT", ""))
    rc = main(["--no-gpu", "--pipeline-check"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "PYTEST STDOUT" in captured.out
    assert "PYTEST STDOUT" not in captured.err
