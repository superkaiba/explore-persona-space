"""Fail-loud log-dir guard pins for the cron wrappers with no test file of their
own (task #2386, unit 3).

Task #2196 gave ``cron_lesson_consolidate.sh`` a ``fatal()`` helper, a guarded
``mkdir -p``, and a ``: >> "$LOG_FILE"`` appendability probe: an uncreatable or
unwritable log dir now exits non-zero with a stderr FATAL naming the path
instead of silently skipping the whole pass and exiting 0 (the brace-group
redirect fails, the group never runs, and the wrapper still reports success).
#2386 ports that pattern to the remaining ten wrappers. Seven of them have no
existing test file; this module is their pin.

Harness principle (reused from tests/test_cron_lesson_consolidate.py): run the
REAL wrapper via ``subprocess.run(["bash", wrapper], env=...)``, inject the log
dir through the wrapper's own env seam, and select the failure mode with the
``log_dir_setup`` kwarg — ``"ok"`` (dir pre-created), ``"uncreatable"`` (the log
dir path sits UNDER a regular FILE, so ``mkdir -p`` fails ENOTDIR — deterministic,
no permission dependence), ``"unwritable"`` (dir exists, chmod 0o555, so
``mkdir -p`` passes and the probe fails; root bypasses mode bits, hence skipif).

Inner-pass containment (new here — these seven wrappers have no BIN seam):
every wrapper runs its real auditor/reaper/guard via ``uv`` inside the brace
group. The harness sets ``HOME`` to a fresh tmp dir and installs a FAKE
recording ``uv`` at ``$HOME/.local/bin/uv``; each wrapper's own
``export PATH="$HOME/.local/bin:$PATH"`` then puts the fake first, so (a) the
``command -v uv`` preflight passes, (b) no real audit / reaper / disk guard can
run, and (c) the recorder file is the evidence of whether the pass ran at all —
present with the expected token on the happy path, ABSENT on both fatal arms.
``TMUX_TMPDIR`` is pre-set (respected by scripts/eps_tmux_env.sh, which
cron_session_summarize.sh sources) and ``EPS_GCP_JANITOR_DRY_RUN=1`` is set so
the assertions never depend on a wrapper's recipe flags. Every test gets a FRESH
log dir (the per-call ``h<N>`` root), so no happy-path assertion can be polluted
by a sibling test's leftovers.

Three parametrized arms per wrapper (plan §6 file 1):

- ``test_uncreatable_log_dir_fails_loud`` — rc != 0, stderr FATAL names the dir,
  the wrapped tool NEVER ran.
- ``test_existing_unwritable_log_dir_fails_loud`` (root-skipif) — same via the
  probe arm, distinguished by the "not appendable" message.
- ``test_happy_path_unchanged`` — rc 0, no FATAL, the wrapped tool DID run, and
  the daily pointer line / silence on stdout is structurally unchanged.

Plus one glob-scan invariant over ``scripts/cron_*.sh``
(``test_every_fatal_wrapper_guards_its_log_dir``): any wrapper defining
``fatal()`` must actually WIRE it to its ``mkdir -p`` and carry the probe. That
scan is why this file is a ``GLOB_SCAN_TESTS`` member in
scripts/select_step9c_tests.py — a wrapper-only (``.sh``) diff reaches no
stem-map or import-map arm, so without that entry a regression in any of these
ten wrappers would select no test at all.
"""

from __future__ import annotations

import itertools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The scan glob this file covers; pinned VERBATIM in select_step9c_tests.py's
# GLOB_SCAN_TESTS (drift pin: test_select_step9c_tests.py::
# test_glob_scan_map_matches_live_tree asserts this literal appears here).
_CRON_WRAPPER_GLOB = "scripts/cron_*.sh"

# Fatal-message fragments shared by every wrapper in the #2386 fix set — the
# mkdir arm and the probe arm are distinguished by these, NOT by the dated log
# filename (which would race a midnight rollover).
_MKDIR_FATAL = "cannot create log dir"
_PROBE_FATAL = "not appendable"


@dataclass(frozen=True)
class _Wrapper:
    """One fix-set wrapper with no test file of its own.

    ``log_dir_env`` is the wrapper's own log-dir seam (derived by reading each
    wrapper, not guessed). ``ran_token`` is a stable identity substring of the
    inner command the brace group dispatches — the script/subcommand name, never
    a flag, so a recipe change cannot make the assertion brittle.
    ``pointer_token`` is the wrapper's FIRST_RUN_OF_DAY stdout pointer prefix, or
    None for the one wrapper that has no such line.
    """

    script: str
    log_dir_env: str
    ran_token: str
    pointer_token: str | None


_WRAPPERS: tuple[_Wrapper, ...] = (
    _Wrapper(
        "cron_uv_cache_prune.sh",
        "EPS_UV_CACHE_PRUNE_LOG_DIR",
        "cache prune",
        "uv_cache_prune:",
    ),
    _Wrapper("cron_gcp_audit.sh", "EPS_GCP_JANITOR_LOG_DIR", "gcp_audit.py", "gcp_audit:"),
    _Wrapper("cron_pod_audit.sh", "EPM_POD_AUDIT_LOG_DIR", "pod.py", "pod_audit:"),
    _Wrapper(
        "cron_session_summarize.sh",
        "EPS_SESSION_SUMMARIZE_LOG_DIR",
        "session_summarize.py",
        None,  # no FIRST_RUN_OF_DAY mechanism — routine passes are fully silent
    ),
    _Wrapper(
        "cron_codex_reaper.sh",
        "EPM_CODEX_REAPER_LOG_DIR",
        "codex_daemon_reaper.py",
        "codex_reaper:",
    ),
    _Wrapper(
        "cron_worktree_audit.sh",
        "EPM_WORKTREE_AUDIT_LOG_DIR",
        "worktree_audit.py",
        "worktree_audit:",
    ),
    _Wrapper(
        "cron_vm_disk_guard.sh",
        "EPS_VM_DISK_GUARD_LOG_DIR",
        "vm_disk_guard.py",
        "vm_disk_guard:",
    ),
)

_WRAPPER_IDS = [w.script.removeprefix("cron_").removesuffix(".sh") for w in _WRAPPERS]


@pytest.fixture
def make_harness(tmp_path: Path):
    """Factory building an isolated wrapper harness (fresh dirs per call).

    Every env seam the wrapper itself reads is redirected into tmp_path: HOME
    (which is how the fake ``uv`` reaches the front of PATH), the wrapper's log
    dir, and TMUX_TMPDIR. Nothing can escape into the real logs/ tree, the real
    ~/.eps-* state, or a real auditor — the fake ``uv`` intercepts every inner
    command and only records its argv.
    """
    counter = itertools.count()

    def _make(wrapper: _Wrapper, log_dir_setup: str = "ok") -> dict:
        root = tmp_path / f"h{next(counter)}"
        home = root / "home"
        bin_dir = home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        tmux_dir = root / "tmux"
        tmux_dir.mkdir(parents=True)

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
        else:  # pragma: no cover — harness misuse is a test defect
            raise ValueError(f"unknown log_dir_setup: {log_dir_setup!r}")

        # The fake `uv`: records its argv, exits 0. Its FILE is the evidence the
        # inner pass ran — absent means the wrapper exited before the brace group.
        recorder = root / "uv_calls.log"
        fake_uv = bin_dir / "uv"
        fake_uv.write_text(f'#!/bin/bash\necho "$@" >> "{recorder}"\nexit 0\n')
        fake_uv.chmod(0o755)

        env = {
            **os.environ,
            "HOME": str(home),
            "TMUX_TMPDIR": str(tmux_dir),
            wrapper.log_dir_env: str(log_dir),
            # Inert here (the fake uv means gcp_audit.py never runs), but it keeps
            # the recorded argv free of any real deletion recipe.
            "EPS_GCP_JANITOR_DRY_RUN": "1",
        }

        def run() -> subprocess.CompletedProcess:
            return subprocess.run(
                ["bash", str(_REPO_ROOT / "scripts" / wrapper.script)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )

        return {"log_dir": log_dir, "recorder": recorder, "run": run}

    return _make


def _daily_log_text(log_dir: Path) -> str:
    logs = sorted(log_dir.glob("*.log"))
    assert len(logs) == 1, f"expected exactly one daily log, found {logs}"
    return logs[0].read_text()


# ── The two fail-loud arms ───────────────────────────────────────────────────


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=_WRAPPER_IDS)
def test_uncreatable_log_dir_fails_loud(make_harness, wrapper: _Wrapper):
    """An uncreatable $LOG_DIR (path under a regular file, ENOTDIR) exits
    non-zero with a stderr FATAL naming the dir, and the wrapped tool is NEVER
    invoked — never the pre-#2386 silent skip-the-pass-and-exit-0."""
    h = make_harness(wrapper, log_dir_setup="uncreatable")
    result = h["run"]()
    assert result.returncode != 0, f"{wrapper.script}: expected non-zero exit"
    assert "FATAL" in result.stderr
    assert _MKDIR_FATAL in result.stderr
    assert str(h["log_dir"]) in result.stderr
    assert not h["recorder"].exists(), (
        f"{wrapper.script}: the pass RAN despite an uncreatable log dir "
        f"({h['recorder'].read_text() if h['recorder'].exists() else ''})"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory mode bits")
@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=_WRAPPER_IDS)
def test_existing_unwritable_log_dir_fails_loud(make_harness, wrapper: _Wrapper):
    """A $LOG_DIR that EXISTS but is unwritable (chmod 0o555) passes ``mkdir -p``
    and fails the appendability probe: non-zero exit, stderr FATAL naming the
    path with the probe-specific message, wrapped tool never invoked."""
    h = make_harness(wrapper, log_dir_setup="unwritable")
    result = h["run"]()
    assert result.returncode != 0, f"{wrapper.script}: expected non-zero exit"
    assert "FATAL" in result.stderr
    # The probe arm, not the mkdir arm — mkdir -p succeeds on an existing dir.
    assert _PROBE_FATAL in result.stderr
    assert _MKDIR_FATAL not in result.stderr
    assert str(h["log_dir"]) in result.stderr
    assert not h["recorder"].exists(), (
        f"{wrapper.script}: the pass RAN despite an unwritable log dir"
    )


# ── The happy path: exit semantics + observable output unchanged ─────────────


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=_WRAPPER_IDS)
def test_happy_path_unchanged(make_harness, wrapper: _Wrapper):
    """A writable log dir still exits 0, emits nothing on stderr, RUNS the inner
    pass, and keeps its stdout contract — the daily pointer line for the six
    wrappers that emit one, complete silence for cron_session_summarize.sh."""
    h = make_harness(wrapper, log_dir_setup="ok")
    result = h["run"]()
    assert result.returncode == 0, f"{wrapper.script}: stderr={result.stderr!r}"
    assert result.stderr.strip() == "", f"{wrapper.script}: unexpected stderr"
    assert "FATAL" not in result.stderr

    # The inner pass RAN (the fake uv recorded it) — the positive control that
    # makes the two "recorder absent" assertions above meaningful.
    assert h["recorder"].exists(), f"{wrapper.script}: the pass did not run"
    recorded = h["recorder"].read_text()
    assert wrapper.ran_token in recorded, f"{wrapper.script}: recorded {recorded!r}"

    # The brace group's own output landed in the daily log, not on the console.
    assert "start ===" in _daily_log_text(h["log_dir"])

    if wrapper.pointer_token is None:
        assert result.stdout.strip() == "", f"{wrapper.script}: expected silence"
    else:
        assert wrapper.pointer_token in result.stdout
        assert "per-pass output" in result.stdout
        assert str(h["log_dir"]) in result.stdout


# ── Glob-scan invariant over the whole cron wrapper family ───────────────────


def _logical_lines(text: str) -> list[str]:
    """Join backslash-continued shell lines into single logical statements."""
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        buf += raw
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1] + " "
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def test_every_fatal_wrapper_guards_its_log_dir():
    """Class invariant over ``scripts/cron_*.sh``: any wrapper that defines a
    ``fatal()`` helper actually WIRES it — every non-comment ``mkdir -p``
    statement is ``|| fatal``-guarded and the appendability probe is present.

    This is the arm that survives a NEW wrapper being added, or one of the ten
    fixed wrappers silently losing its guard in a refactor: the per-wrapper
    subprocess arms above only cover the seven this file drives. Wrappers with
    no ``fatal()`` (cron_codex_auto_upgrade.sh's checked-mkdir + alert-arm
    design, cron_export_literature.sh's ``set -e``, the two per-issue watch
    crons with no mkdir at all) are deliberately out of scope — they are the
    plan §4.2 NOT-APPLICABLE set.
    """
    wrappers = sorted(_REPO_ROOT.glob(_CRON_WRAPPER_GLOB))
    assert wrappers, f"no wrappers matched {_CRON_WRAPPER_GLOB!r}"

    guarded: list[str] = []
    for path in wrappers:
        text = path.read_text()
        if not any(ln.startswith("fatal() {") for ln in text.splitlines()):
            continue
        guarded.append(path.name)

        mkdir_stmts = [
            ln
            for ln in _logical_lines(text)
            if "mkdir -p" in ln and not ln.lstrip().startswith("#")
        ]
        assert mkdir_stmts, f"{path.name}: defines fatal() but has no mkdir -p"
        for stmt in mkdir_stmts:
            assert "|| fatal" in stmt, (
                f"{path.name}: unguarded mkdir -p — an uncreatable log dir would "
                f"silently skip the pass: {stmt.strip()[:100]!r}"
            )
        assert any(ln.startswith(': >> "$LOG_FILE"') for ln in text.splitlines()), (
            f"{path.name}: missing the appendability probe — mkdir -p succeeds on "
            "an existing dir regardless of writability"
        )

    missing = [w.script for w in _WRAPPERS if w.script not in guarded]
    assert not missing, f"driven wrappers lost their fatal() guard: {missing}"
