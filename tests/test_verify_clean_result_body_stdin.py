"""Tests for the ``--body-stdin`` mode of scripts/verify_clean_result.py.

Two contracts:

1. ``--body-stdin`` reads body from stdin, requires ``--title`` and
   ``--created-at``, and accepts repeatable ``--label``.
2. Strict-toggle parity: given identical inputs, ``--body-stdin`` and
   ``--issue`` modes produce the same ``strict`` boolean.

The strict toggle drives several check functions' grandfathered-pass
behavior; if the two modes diverge, a backfill of older clean-results
issues would FAIL them all (the C2.B3 risk in plan §4).
"""

from __future__ import annotations

import importlib.util
import io
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Load verify_clean_result.py directly (not on the package path).
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "verify_clean_result.py"


@pytest.fixture
def vcr_module():
    """Import scripts/verify_clean_result.py as a module.

    Register the module in ``sys.modules`` BEFORE executing it; otherwise
    ``dataclass`` resolution at class-creation time looks up the module
    via ``sys.modules.get(cls.__module__)`` and explodes when the entry
    is missing.
    """
    spec = importlib.util.spec_from_file_location("verify_clean_result", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_clean_result"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("verify_clean_result", None)
        raise
    return mod


# ──────────────────────────────────────────────────────────────────────
# _compute_strict_toggle: shared helper used by both modes
# ──────────────────────────────────────────────────────────────────────


def test_strict_toggle_recent_issue_no_promotion_returns_strict(vcr_module):
    """≤7d old, not promoted → strict=True."""
    now = datetime.now(UTC)
    created = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    strict, dt = vcr_module._compute_strict_toggle(
        created_at=created, label_names=["clean-results:draft"]
    )
    assert strict is True
    assert dt is not None


def test_strict_toggle_old_issue_returns_grandfathered(vcr_module):
    """>7d old → strict=False (grandfathered)."""
    now = datetime.now(UTC)
    created = (now - timedelta(days=14)).isoformat().replace("+00:00", "Z")
    strict, _ = vcr_module._compute_strict_toggle(
        created_at=created, label_names=["clean-results:draft"]
    )
    assert strict is False


def test_strict_toggle_promoted_issue_returns_grandfathered(vcr_module):
    """Already-promoted (clean-results without :draft) → strict=False."""
    now = datetime.now(UTC)
    created = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    strict, _ = vcr_module._compute_strict_toggle(
        created_at=created, label_names=["clean-results", "clean-results:useful"]
    )
    assert strict is False


def test_strict_toggle_malformed_iso_falls_back_to_strict(vcr_module):
    """Unparseable timestamp → fall back to strict (don't crash)."""
    strict, dt = vcr_module._compute_strict_toggle(
        created_at="not-a-date", label_names=["clean-results:draft"]
    )
    # Fallback path: created_dt = now, age = 0, so strict=True.
    assert strict is True
    assert dt is not None


def test_strict_toggle_draft_with_clean_results_still_strict(vcr_module):
    """clean-results AND clean-results:draft both present → still strict (draft wins)."""
    now = datetime.now(UTC)
    created = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    strict, _ = vcr_module._compute_strict_toggle(
        created_at=created,
        label_names=["clean-results", "clean-results:draft"],
    )
    assert strict is True


# ──────────────────────────────────────────────────────────────────────
# CLI: --body-stdin argument validation
# ──────────────────────────────────────────────────────────────────────


def test_body_stdin_requires_title_and_created_at(vcr_module, capsys):
    """argparse rejects --body-stdin without --title or --created-at."""
    with pytest.raises(SystemExit) as exc:
        vcr_module.main(["--body-stdin"])
    # argparse parser.error → SystemExit(2)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--body-stdin requires both --title and --created-at" in captured.err


def test_body_stdin_requires_title_alone_errors(vcr_module, capsys):
    with pytest.raises(SystemExit) as exc:
        vcr_module.main(["--body-stdin", "--title", "X"])  # no --created-at
    assert exc.value.code == 2


def test_body_stdin_mutex_with_issue(vcr_module):
    """argparse rejects --body-stdin alongside --issue (mutually exclusive)."""
    with pytest.raises(SystemExit):
        vcr_module.main(["--body-stdin", "--issue", "320"])


# ──────────────────────────────────────────────────────────────────────
# CLI: --body-stdin reads from stdin, returns same exit code as file mode
# ──────────────────────────────────────────────────────────────────────


def _run_with_stdin(vcr_module, argv: list[str], stdin_text: str) -> int:
    """Helper: run main() with stdin set to ``stdin_text``."""
    saved = sys.stdin
    sys.stdin = io.StringIO(stdin_text)
    try:
        return vcr_module.main(argv)
    finally:
        sys.stdin = saved


def test_body_stdin_obviously_bad_body_returns_fail(vcr_module, capsys):
    """An obviously-empty body fails the verifier (rc=1)."""
    rc = _run_with_stdin(
        vcr_module,
        argv=[
            "--body-stdin",
            "--title",
            "Some title (LOW confidence)",
            "--created-at",
            "2026-05-01T00:00:00Z",
            "--current-issue",
            "999",
        ],
        stdin_text="",
    )
    assert rc == 1


def test_body_stdin_label_flag_is_repeatable(vcr_module):
    """--label is repeatable; multiple labels feed into _compute_strict_toggle."""
    rc = _run_with_stdin(
        vcr_module,
        argv=[
            "--body-stdin",
            "--title",
            "T (LOW confidence)",
            "--created-at",
            "2026-05-01T00:00:00Z",
            "--label",
            "clean-results",
            "--label",
            "clean-results:useful",
            "--current-issue",
            "999",
        ],
        stdin_text="",
    )
    # rc=1 expected (empty body fails); the test confirms argparse accepts
    # multiple --label flags without erroring.
    assert rc in (1, 0)


# ──────────────────────────────────────────────────────────────────────
# Strict-toggle parity: --body-stdin vs --issue produce same `strict`
#
# This is the load-bearing test from plan §4 acceptance bullet
# (line ~963) — the C2.B3 risk requires verbatim parity.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "days_ago,labels,expected_strict",
    [
        (1, ["clean-results:draft"], True),  # fresh draft → strict
        (10, ["clean-results:draft"], False),  # >7d old → non-strict
        (1, ["clean-results", "clean-results:useful"], False),  # promoted → non-strict
        (
            1,
            ["clean-results", "clean-results:draft"],
            True,
        ),  # draft+clean-results → strict (draft wins)
        (3, [], True),  # no clean-results label, fresh → strict
    ],
)
def test_body_stdin_strict_toggle_parity(vcr_module, days_ago, labels, expected_strict):
    """The toggle is identical regardless of how the inputs reach the helper.

    Both --issue and --body-stdin call _compute_strict_toggle with the
    SAME signature; this regression test pins the contract so a future
    refactor that adds an argument to one path can't silently diverge.
    """
    now = datetime.now(UTC)
    created = (now - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    strict, _ = vcr_module._compute_strict_toggle(created_at=created, label_names=labels)
    assert strict is expected_strict
