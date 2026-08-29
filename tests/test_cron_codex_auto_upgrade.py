"""Fail-loud log-dir guard pins for ``scripts/cron_codex_auto_upgrade.sh`` (task
#2386 revise round 2, Pattern C).

This wrapper is the one fix-set member that must NOT get the ``fatal()`` helper
the other ten carry. Its design invariant is *never exit before the alert arm*:
every setup failure records itself, skips the upgrader, binds ``rc=1``, and
falls through to ONE ``alert_failure`` call (audit-sidecar row + per-date-deduped
Telegram push), after which the trailing ``exit 0`` stands. A ``fatal()``-style
early ``exit 1`` would skip that arm entirely — with no MTA on this VM and the
crontab redirecting stderr, that is a structurally silent failure.

So Pattern C routes the appendability probe into the wrapper's OWN failure
design (``SETUP_OK=0``) rather than into ``fatal()``. The gap it closes: plan v1
recorded this wrapper NOT-APPLICABLE because its ``mkdir -p`` is already checked,
which holds for the UNCREATABLE leg and is false for the UNWRITABLE one —
``mkdir -p`` returns success for an existing dir whatever its mode, so
``SETUP_OK`` stayed 1, the brace-group redirect failed before ``rc`` was ever
assigned, ``${rc:-0}`` evaluated to 0, and the wrapper exited 0 with the upgrader
never run and no alert fired.

Harness principle (reused from tests/test_cron_lesson_consolidate.py and
tests/test_cron_wrapper_log_dir_guard.py): run the REAL wrapper via
``subprocess.run(["bash", wrapper], env=...)`` with every side-effecting seam
redirected into ``tmp_path``. Nothing here may reach a real upgrade, a real
Telegram push, the real audit sidecar, or the real ``logs/`` tree:

- ``EPS_CODEX_UPGRADE_BIN`` -> a recording stub, so the real
  ``uv run python scripts/codex_auto_upgrade.py`` never runs. Its recorder file
  is the evidence of whether the upgrader ran at all: present on the happy path,
  ABSENT on both fatal arms.
- ``EPS_TELEGRAM_PUSH_SCRIPT`` -> a recording stub, so no real push is sent and
  the alert count is directly observable (one line per push).
- ``EPS_CODEX_UPGRADE_SIDECAR`` -> a tmp path, so the real
  ``.claude/cache/codex-auto-upgrade-events.jsonl`` is never appended to.
- ``EPS_CODEX_UPGRADE_LOG_DIR`` -> a tmp dir, per the ``log_dir_setup`` kwarg.
- ``HOME`` -> a tmp dir holding fake ``uv`` / ``npm`` / ``codex`` on
  ``$HOME/.local/bin`` (which the wrapper's own ``export PATH`` puts first), so
  the ``command -v`` preflight passes deterministically without depending on —
  or being able to invoke — any real binary.

``SENTINEL_DIR`` is deliberately left DEFAULTED (it defaults to the log dir, so
on the unwritable arm the sentinel write fails too). That is the production
shape, and it is why the fatal arm carries no per-date dedup: the sentinel dir
is the very dir that is unwritable.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO_ROOT / "scripts" / "cron_codex_auto_upgrade.sh"

# Fatal-message fragments. The two setup-failure arms are distinguished by these
# rather than by the dated log filename (which would race a midnight rollover).
_MKDIR_FATAL = "cannot create log/sentinel dirs"
_PROBE_FATAL = "not appendable"


@pytest.fixture
def make_harness(tmp_path: Path):
    """Factory building an isolated wrapper harness (fresh dirs per call)."""
    counter = itertools.count()

    def _make(log_dir_setup: str = "ok") -> dict:
        root = tmp_path / f"h{next(counter)}"
        home = root / "home"
        bin_dir = home / ".local" / "bin"
        bin_dir.mkdir(parents=True)

        if log_dir_setup == "ok":
            log_dir = root / "logs"
            log_dir.mkdir(parents=True)
        elif log_dir_setup == "uncreatable":
            blocker = root / "blocker"
            blocker.write_text("regular file blocking mkdir -p (ENOTDIR)\n")
            log_dir = blocker / "logs"
        elif log_dir_setup == "unwritable":
            log_dir = root / "logs"
            log_dir.mkdir(parents=True)
            log_dir.chmod(0o555)
        else:  # pragma: no cover - harness misuse is a test defect
            raise ValueError(f"unknown log_dir_setup: {log_dir_setup!r}")

        upgrader_calls = root / "upgrader_calls.log"
        upgrader = bin_dir / "fake_upgrader"
        upgrader.write_text(f'#!/bin/bash\necho "ran $@" >> "{upgrader_calls}"\nexit 0\n')
        upgrader.chmod(0o755)

        push_calls = root / "push_calls.log"
        push = bin_dir / "fake_telegram_push"
        # Prints nothing: the wrapper captures push stdout+stderr and log_line's
        # it, so a chatty stub would pollute the stderr assertions.
        push.write_text(f'#!/bin/bash\nprintf "%s\\n" "$1" >> "{push_calls}"\nexit 0\n')
        push.chmod(0o755)

        # The `command -v uv npm codex` preflight must pass on its own terms;
        # these must never actually execute (the upgrader seam intercepts first).
        path_bin_calls = root / "path_bin_calls.log"
        for name in ("uv", "npm", "codex"):
            stub = bin_dir / name
            stub.write_text(f'#!/bin/bash\necho "{name} $@" >> "{path_bin_calls}"\nexit 0\n')
            stub.chmod(0o755)

        sidecar = root / "sidecar.jsonl"

        env = {
            **os.environ,
            "HOME": str(home),
            "EPS_CODEX_UPGRADE_LOG_DIR": str(log_dir),
            "EPS_CODEX_UPGRADE_SIDECAR": str(sidecar),
            "EPS_CODEX_UPGRADE_BIN": str(upgrader),
            "EPS_TELEGRAM_PUSH_SCRIPT": str(push),
        }
        # SENTINEL_DIR deliberately NOT set - it defaults to the log dir
        # (production shape; unwritable on that arm by construction).
        env.pop("EPS_CODEX_UPGRADE_SENTINEL_DIR", None)

        def run() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["bash", str(_WRAPPER)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )

        return {
            "log_dir": log_dir,
            "upgrader_calls": upgrader_calls,
            "push_calls": push_calls,
            "path_bin_calls": path_bin_calls,
            "sidecar": sidecar,
            "run": run,
        }

    return _make


def _alert_count(h: dict) -> tuple[int, int]:
    """(sidecar rows, telegram pushes) recorded by this run."""
    rows = (
        [ln for ln in h["sidecar"].read_text().splitlines() if ln.strip()]
        if h["sidecar"].exists()
        else []
    )
    pushes = (
        [ln for ln in h["push_calls"].read_text().splitlines() if ln.strip()]
        if h["push_calls"].exists()
        else []
    )
    return len(rows), len(pushes)


def _assert_upgrader_never_ran(h: dict) -> None:
    assert not h["upgrader_calls"].exists(), (
        "the upgrader RAN despite a setup failure: "
        f"{h['upgrader_calls'].read_text() if h['upgrader_calls'].exists() else ''}"
    )
    # Belt and braces: the real path would shell out through `uv`.
    assert not h["path_bin_calls"].exists(), (
        f"a PATH binary was executed: {h['path_bin_calls'].read_text()}"
    )


# -- The fail-loud arms -------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory mode bits")
def test_existing_unwritable_log_dir_alerts_and_skips_upgrader(make_harness):
    """THE #2386 revise-round-2 PIN (Pattern C).

    A log dir that EXISTS but is unwritable (chmod 0o555) passes ``mkdir -p``, so
    the pre-existing checked-mkdir guard cannot see it. The appendability probe
    must catch it: the upgrader is skipped and EXACTLY ONE alert fires. Pre-fix
    this run exited 0 with the upgrader never invoked and NO alert at all.
    """
    h = make_harness(log_dir_setup="unwritable")
    result = h["run"]()

    _assert_upgrader_never_ran(h)

    # The probe arm, not the mkdir arm - mkdir -p succeeds on an existing dir.
    assert "FATAL" in result.stderr
    assert _PROBE_FATAL in result.stderr
    assert _MKDIR_FATAL not in result.stderr
    assert str(h["log_dir"]) in result.stderr

    n_rows, n_pushes = _alert_count(h)
    assert n_rows == 1, f"expected exactly one sidecar row, got {n_rows}"
    assert n_pushes == 1, f"expected exactly one telegram push, got {n_pushes}"

    row = json.loads(h["sidecar"].read_text().splitlines()[0])
    assert row["event"] == "upgrade_failed"
    assert row["rc"] == 1

    # Pattern C's design invariant: the alert arm is the channel, so the wrapper
    # still exits 0. That is what makes fatal()'s early `exit 1` wrong HERE.
    assert result.returncode == 0, f"stderr={result.stderr!r}"


def test_uncreatable_log_dir_alerts_and_skips_upgrader(make_harness):
    """Regression pin on the PRE-EXISTING checked-mkdir guard: an uncreatable
    $LOG_DIR (path under a regular file, ENOTDIR) still skips the upgrader and
    fires exactly one alert. The probe added for the unwritable leg must not
    disturb this arm's single-alert contract."""
    h = make_harness(log_dir_setup="uncreatable")
    result = h["run"]()

    _assert_upgrader_never_ran(h)

    assert "FATAL" in result.stderr
    assert _MKDIR_FATAL in result.stderr
    assert str(h["log_dir"]) in result.stderr

    n_rows, n_pushes = _alert_count(h)
    assert n_rows == 1, f"expected exactly one sidecar row, got {n_rows}"
    assert n_pushes == 1, f"expected exactly one telegram push, got {n_pushes}"
    assert result.returncode == 0, f"stderr={result.stderr!r}"


# -- The happy path: guards fire ONLY on failure ------------------------------


def test_happy_path_runs_upgrader_and_does_not_alert(make_harness):
    """A writable log dir still runs the upgrader, fires NO alert, and writes the
    brace group's output into the daily log. This is the positive control that
    makes the two "upgrader never ran" assertions above meaningful, and the pin
    that the probe is a no-op append on the healthy path."""
    h = make_harness(log_dir_setup="ok")
    result = h["run"]()

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "FATAL" not in result.stderr

    assert h["upgrader_calls"].exists(), "the upgrader did not run"

    n_rows, n_pushes = _alert_count(h)
    assert n_rows == 0, f"a healthy run alerted: {h['sidecar'].read_text()}"
    assert n_pushes == 0, f"a healthy run pushed: {h['push_calls'].read_text()}"

    logs = sorted(h["log_dir"].glob("*.log"))
    assert len(logs) == 1, f"expected exactly one daily log, found {logs}"
    log_text = logs[0].read_text()
    assert "codex_auto_upgrade start ===" in log_text
    assert "codex_auto_upgrade exit=0 ===" in log_text
