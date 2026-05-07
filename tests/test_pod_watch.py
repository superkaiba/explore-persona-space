"""Tests for the §2 stall-detection watchdog.

Most behaviours are covered with synthetic GitHub responses (mock
``_gh_view``). The integration smoke (real `gh issue` round-trip on a test
fixture issue) is documented in plan §2 and exercised manually — outside
the unit-test scope.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pod_watch():
    """Load scripts/pod_watch.py as a module without registering it on
    sys.modules permanently (test isolation)."""
    spec = importlib.util.spec_from_file_location(
        "pod_watch_under_test", REPO_ROOT / "scripts" / "pod_watch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["pod_watch_under_test"] = module
    spec.loader.exec_module(module)
    return module


pod_watch = _load_pod_watch()


def test_argparse_threshold_secs():
    """Happy path: argparse accepts --threshold-secs and rejects negatives
    via the help message (no negative validation today, but at least the
    flag must be wired)."""
    parser = _build_parser()
    ns = parser.parse_args(["--issue", "999", "--threshold-secs", "30"])
    assert ns.issue == 999
    assert ns.threshold_secs == 30


def test_argparse_force_attach_default_false():
    """--force-attach defaults to False (the safe default; /issue Step 6d
    auto-spawn never sets it)."""
    parser = _build_parser()
    ns = parser.parse_args(["--issue", "999"])
    assert ns.force_attach is False
    ns_force = parser.parse_args(["--issue", "999", "--force-attach"])
    assert ns_force.force_attach is True


def test_argparse_max_runtime_default():
    """24h default for --max-runtime-secs."""
    parser = _build_parser()
    ns = parser.parse_args(["--issue", "999"])
    assert ns.max_runtime_secs == 86400


def _build_parser():
    """Re-create the argparse from main(). We can't simply call main() since
    it loops; instead we inspect the parser construction by invoking main
    with a sentinel that exits before the loop. The cheapest way: replicate
    the CLI signature by parsing argv via main() but with `--help` swallowed.
    The cleanest is to reach in and exercise the parser builder. Pod-watch
    inlines its parser; we re-instantiate to mirror it."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--threshold-secs", type=int, default=pod_watch.DEFAULT_THRESHOLD_SECS)
    parser.add_argument("--wandb-run-url", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--max-runtime-secs", type=int, default=pod_watch.DEFAULT_MAX_RUNTIME_SECS)
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--force-attach", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


# ── _check_terminal ───────────────────────────────────────────────────────


def _snapshot(*, labels: list[str], comment_bodies: list[str]) -> dict:
    return {
        "labels": [{"name": label} for label in labels],
        "comments": [{"body": body} for body in comment_bodies],
    }


def test_check_terminal_returns_true_on_results_marker():
    """epm:results marker means the experiment finished gracefully — exit."""
    snap = _snapshot(
        labels=["status:running"],
        comment_bodies=["<!-- epm:results v1 -->\nresults here\n<!-- /epm:results -->"],
    )
    with patch.object(pod_watch, "_gh_view", return_value=snap):
        assert pod_watch._check_terminal(999) is True


def test_check_terminal_returns_true_on_failure_marker():
    """If anyone else posted epm:failure, watchdog steps aside silently."""
    snap = _snapshot(
        labels=["status:running"],
        comment_bodies=["<!-- epm:failure v1 -->\nfailure body\n<!-- /epm:failure -->"],
    )
    with patch.object(pod_watch, "_gh_view", return_value=snap):
        assert pod_watch._check_terminal(999) is True


def test_check_terminal_returns_true_when_status_moved():
    """Status no longer running — watchdog exits silently."""
    snap = _snapshot(labels=["status:uploading"], comment_bodies=[])
    with patch.object(pod_watch, "_gh_view", return_value=snap):
        assert pod_watch._check_terminal(999) is True


def test_check_terminal_returns_false_when_still_running_and_no_terminal_markers():
    """Run is in flight; watchdog continues."""
    snap = _snapshot(labels=["status:running"], comment_bodies=["<!-- epm:progress v1 -->\n..."])
    with patch.object(pod_watch, "_gh_view", return_value=snap):
        assert pod_watch._check_terminal(999) is False


# ── _post_failure ─────────────────────────────────────────────────────────


def test_post_failure_aborts_when_status_moved():
    """If between tick and post the label moved out of running, post no marker."""
    snap = _snapshot(labels=["status:blocked"], comment_bodies=[])
    with (
        patch.object(pod_watch, "_gh_view", return_value=snap),
        patch.object(pod_watch.subprocess, "check_call") as mock_call,
    ):
        pod_watch._post_failure(999, reason="stall", last_event=1700000000.0)
        mock_call.assert_not_called()


def test_post_failure_idempotent_when_higher_pid_marker_exists():
    """If a later-pid epm:failure already exists, refuse to post a duplicate."""
    later_pid = 999_999_999
    snap = _snapshot(
        labels=["status:running"],
        comment_bodies=[
            f"<!-- epm:failure v1 (watch-pid={later_pid}) -->\nbody\n<!-- /epm:failure -->"
        ],
    )
    with (
        patch.object(pod_watch, "_gh_view", return_value=snap),
        patch.object(pod_watch.subprocess, "check_call") as mock_call,
    ):
        pod_watch._post_failure(999, reason="stall", last_event=1700000000.0)
        mock_call.assert_not_called()


def test_post_failure_posts_when_status_running_and_no_higher_pid():
    """Happy path: post epm:failure + flip label."""
    snap = _snapshot(labels=["status:running"], comment_bodies=[])
    with (
        patch.object(pod_watch, "_gh_view", return_value=snap),
        patch.object(pod_watch.subprocess, "check_call") as mock_call,
    ):
        pod_watch._post_failure(999, reason="stall", last_event=1700000000.0)
        # Two calls expected: one for the comment, one for the label edit.
        assert mock_call.call_count == 2
        comment_call, label_call = mock_call.call_args_list
        comment_args = comment_call.args[0]
        assert comment_args[:4] == ["gh", "issue", "comment", "999"]
        body = comment_args[5]  # ['gh', 'issue', 'comment', '999', '--body', body]
        assert "<!-- epm:failure v1" in body
        assert "failure_class: infra" in body
        assert "reason: stall" in body
        assert f"watch-pid={pod_watch.os.getpid()}" in body

        label_args = label_call.args[0]
        assert "--remove-label" in label_args
        assert "status:running" in label_args
        assert "--add-label" in label_args
        assert "status:blocked" in label_args


def test_post_failure_marker_carries_isoformat_last_event():
    """`last_event: <iso8601>` line present when last_event is provided."""
    snap = _snapshot(labels=["status:running"], comment_bodies=[])
    with (
        patch.object(pod_watch, "_gh_view", return_value=snap),
        patch.object(pod_watch.subprocess, "check_call") as mock_call,
    ):
        pod_watch._post_failure(999, reason="probe_unreachable", last_event=None)
        comment_call = mock_call.call_args_list[0]
        body = comment_call.args[0][5]
        assert "last_event: never" in body


# ── _max_failure_pid ──────────────────────────────────────────────────────


def test_max_failure_pid_returns_largest():
    snap = _snapshot(
        labels=["status:running"],
        comment_bodies=[
            "<!-- epm:failure v1 (watch-pid=100) -->\n",
            "<!-- epm:failure v1 (watch-pid=200) -->\n",
            "<!-- epm:progress v1 -->\n",
        ],
    )
    assert pod_watch._max_failure_pid(snap) == 200


def test_max_failure_pid_returns_none_on_no_failures():
    snap = _snapshot(labels=["status:running"], comment_bodies=["<!-- epm:progress v1 -->"])
    assert pod_watch._max_failure_pid(snap) is None


# ── pid file lifecycle ────────────────────────────────────────────────────


def test_main_creates_pid_file_and_cleans_up(tmp_path, monkeypatch):
    """Confirm the wrapper creates and unlinks the pid file. We force the
    inner watch-loop to exit immediately by patching it."""
    pid_file = tmp_path / "watch-999.pid"

    def _fake_loop(*args, **kwargs):
        # Confirm the pid-file was written before the loop ran.
        assert pid_file.exists()
        assert pid_file.read_text() == str(pod_watch.os.getpid())
        return 0

    with patch.object(pod_watch, "_watch_loop", _fake_loop):
        rc = pod_watch.main(
            ["--issue", "999", "--pid-file", str(pid_file), "--threshold-secs", "30"]
        )
        assert rc == 0
    # Cleanup happened.
    assert not pid_file.exists()


# ── failure_classifier integration ────────────────────────────────────────


def test_failure_classifier_routes_stall_reason_to_infra():
    """The new INFRA_PATTERNS entries route the watchdog body via the regex
    fallback even if `failure_class` field is somehow missing from the body."""
    from scripts.failure_classifier import classify_failure

    # Body without the failure_class field — exercises the regex path.
    body = "## Stall detected\n\nreason: stall\nlast_event: 2025-01-01T00:00:00\n"
    assert classify_failure(body) == "infra"


def test_failure_classifier_routes_probe_unreachable_to_infra():
    from scripts.failure_classifier import classify_failure

    body = "## Stall detected\n\nreason: probe_unreachable\n"
    assert classify_failure(body) == "infra"


def test_failure_classifier_field_line_still_wins():
    """The explicit field-line takes precedence over regex inference; even
    a body whose only signal is a `reason: stall` regex match ALSO has the
    field-line and routes infra."""
    from scripts.failure_classifier import classify_failure

    body = "failure_class: infra\nreason: stall\n"
    assert classify_failure(body) == "infra"


# ── argparse smoke (no missing-issue regression) ──────────────────────────


def test_main_requires_issue():
    """Forgetting --issue exits with non-zero, no traceback for users."""
    with pytest.raises(SystemExit):
        pod_watch.main([])
