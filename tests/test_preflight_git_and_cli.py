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


def _fake_git_run(
    *,
    branch="issue-554",
    porcelain="",
    own_ref_exists=True,
    behind_own="0",
    ahead_own="0",
    behind_main="0",
    fetch_rc=0,
    fetch_err="",
    retry_rc=0,
    retry_err="",
    calls=None,
):
    """Canned ``_run`` dispatcher covering every git argv check_git_status issues.

    The fetch model distinguishes the ORIGINAL broad fetch
    (``git fetch --quiet origin``, no refspec) from the #696 branch-scoped
    RETRY (``git fetch --quiet origin <branch> --depth=1``):

    * ``fetch_rc`` / ``fetch_err`` control the FIRST broad fetch. The legacy
      contract is preserved: a hard failure (``fetch_rc != 0`` with the default
      empty ``fetch_err``) returns the byte-identical ``"fatal: unable to
      access remote"`` stderr the original dispatcher emitted. A TRANSIENT
      timeout is modelled by passing ``fetch_rc=-1, fetch_err="timeout"`` — the
      exact ``(-1, "", "timeout")`` shape ``preflight._run`` returns on
      ``subprocess.TimeoutExpired`` (preflight.py:152-153).
    * ``retry_rc`` / ``retry_err`` control the branch-scoped retry argv
      (distinguished by the trailing ``--depth=1`` + branch token). They
      default to success ``(0, "")`` so a retry that is never issued (the
      default no-failure path) is irrelevant.

    Every ``(cmd, timeout)`` pair is appended to ``calls`` so the retry argv +
    its ``timeout=60`` are assertable (#696 AC1b / AC3).
    """
    full_own = f"refs/remotes/origin/{branch}"

    def _fetch_stderr(rc, err):
        # An explicit err string wins; else the legacy non-zero->canned mapping.
        if err:
            return err
        return "" if rc == 0 else "fatal: unable to access remote"

    def fake_run(cmd, timeout=10):
        if calls is not None:
            calls.append((cmd, timeout))
        if "status" in cmd and "--porcelain" in cmd:
            return 0, porcelain, ""
        if "fetch" in cmd:
            # The branch-scoped #696 retry carries the branch token + --depth=1;
            # the original broad fetch is `... fetch --quiet origin` with no
            # refspec. Discriminate on the trailing two tokens.
            if cmd[-1] == "--depth=1" and cmd[-2] == branch:
                return retry_rc, "", _fetch_stderr(retry_rc, retry_err)
            return fetch_rc, "", _fetch_stderr(fetch_rc, fetch_err)
        if cmd[-2:] == ["--abbrev-ref", "HEAD"]:
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
    calls: list[tuple[list[str], int]] = []
    _patch_git(monkeypatch, branch="issue-554", behind_own="3", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "behind origin/issue-554" in report.errors[0]
    assert "git pull --ff-only" in report.errors[0]
    # The behind-own guarantee is only as fresh as the fetch: the exact fetch
    # argv must be issued BEFORE the own-ref rev-list (an implementation that
    # drops the fetch must fail here).
    fetch_argv = ["git", "-C", str(ROOT), "fetch", "--quiet", "origin"]
    own_revlist_argv = [
        "git",
        "-C",
        str(ROOT),
        "rev-list",
        "--count",
        "HEAD..refs/remotes/origin/issue-554",
    ]
    argvs = [cmd for cmd, _timeout in calls]
    assert fetch_argv in argvs
    assert own_revlist_argv in argvs
    assert argvs.index(fetch_argv) < argvs.index(own_revlist_argv)


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


def test_feature_branch_fetch_hard_error_errors_no_retry(monkeypatch):
    """AC2c (#696, REPLACES test_feature_branch_fetch_failure_errors): a HARD
    (non-timeout) fetch failure on a feature branch is STILL an ERROR — and
    must NOT burn a 60s branch-scoped retry on a doomed remote.

    behind-own=0 against stale refs would read as a silent false PASS, so the
    hard-error case keeps the unconditional ERROR (the #554/#550 contract). The
    byte-compatible message fragments from the original test still hold; the new
    assertion is that NO branch-scoped retry argv was issued."""
    calls: list[tuple[list[str], int]] = []
    _patch_git(monkeypatch, branch="issue-554", fetch_rc=128, behind_own="0", calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "git fetch origin failed" in report.errors[0]
    assert "origin/issue-554" in report.errors[0]
    # A hard error must short-circuit BEFORE the retry — no branch-scoped
    # `... fetch --quiet origin issue-554 --depth=1` argv may appear.
    retry_argv = ["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "issue-554", "--depth=1"]
    assert retry_argv not in [cmd for cmd, _timeout in calls]


def test_feature_branch_fetch_timeout_lastknown_current_passes(monkeypatch):
    """AC1 (#696): a TRANSIENT fetch timeout (retry also times out) on a checkout
    that the last-known origin ref proves current (behind 0) DOWNGRADES the
    would-be ERROR to a WARNING and PASSes — the same fail-soft-toward-PASS
    decision `main` already makes on a fetch failure."""
    _patch_git(
        monkeypatch,
        branch="issue-554",
        fetch_rc=-1,
        fetch_err="timeout",
        retry_rc=-1,
        retry_err="timeout",
        own_ref_exists=True,
        behind_own="0",
    )
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    assert any(("last-known" in w and "up to date" in w) for w in report.warnings)


def test_feature_branch_fetch_timeout_retry_succeeds_passes(monkeypatch):
    """AC1b + AC3 (#696): the first broad fetch times out, the branch-scoped
    retry SUCCEEDS, and the normal fresh-ref comparison runs (behind 0 → PASS).

    Asserts the branch-scoped retry argv is issued, that it carries a 60s
    timeout (AC3's structural bound — at most one ~60s retry), and that no
    timeout/fetch-failure WARNING is surfaced (the retry made the refs fresh)."""
    calls: list[tuple[list[str], int]] = []
    _patch_git(
        monkeypatch,
        branch="issue-554",
        fetch_rc=-1,
        fetch_err="timeout",
        retry_rc=0,
        retry_err="",
        own_ref_exists=True,
        behind_own="0",
        calls=calls,
    )
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    assert report.errors == []
    retry_argv = ["git", "-C", str(ROOT), "fetch", "--quiet", "origin", "issue-554", "--depth=1"]
    retry_calls = [(cmd, timeout) for cmd, timeout in calls if cmd == retry_argv]
    # Exactly one branch-scoped retry, issued with a 60s timeout (AC3).
    assert len(retry_calls) == 1
    assert retry_calls[0][1] == 60
    # A successful retry means the refs are fresh — no timeout/fetch WARNING.
    assert not any(("timed out" in w or "fetch" in w) for w in report.warnings)


def test_feature_branch_fetch_timeout_but_behind_errors(monkeypatch):
    """AC2a (#696): a genuinely-stale checkout (timeout + retry-timeout +
    behind_own>0 against the last-known ref) STILL ERRORs — the downgrade only
    rescues a verifiably-CURRENT checkout, never a behind one."""
    _patch_git(
        monkeypatch,
        branch="issue-554",
        fetch_rc=-1,
        fetch_err="timeout",
        retry_rc=-1,
        retry_err="timeout",
        own_ref_exists=True,
        behind_own="3",
    )
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "cannot verify" in report.errors[0]
    assert "issue-554" in report.errors[0]


def test_feature_branch_fetch_timeout_no_lastknown_errors(monkeypatch):
    """AC2b (#696): a timeout (retry also times out) with NO last-known
    origin/<branch> ref STILL ERRORs — a fresh-clone pod that never fetched the
    branch has no ref to prove currency against, so currency is unprovable."""
    _patch_git(
        monkeypatch,
        branch="issue-554",
        fetch_rc=-1,
        fetch_err="timeout",
        retry_rc=-1,
        retry_err="timeout",
        own_ref_exists=False,
    )
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is False
    assert len(report.errors) == 1
    assert "issue-554" in report.errors[0]


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


def test_cluster_skips_fetch_and_behind(monkeypatch):
    """Criterion 6: the cluster early-return is untouched — no fetch, no ref math."""
    calls: list[tuple[list[str], int]] = []
    _patch_git(monkeypatch, cluster=True, calls=calls)
    report = PreflightReport()
    check_git_status(report, ROOT)
    assert report.ok is True
    for cmd, _timeout in calls:
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
    with the 12 documented keys and nothing else (gotchas.md contract)."""
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
        "git_status",
        "env_synced",
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
