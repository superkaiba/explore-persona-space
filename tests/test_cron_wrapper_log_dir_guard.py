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
  stdout is EXACTLY the daily pointer line (or exactly empty for the one wrapper
  with no ``FIRST_RUN_OF_DAY`` mechanism) — a line-count pin, not substrings.

Plus one glob-scan invariant over ``scripts/cron_*.sh``
(``test_every_cron_wrapper_matches_its_declared_guard_shape``): EVERY wrapper the
live glob returns carries a declared guard class in ``_WRAPPER_CLASSIFICATION``,
the classification set must EQUAL the glob set, and each class's shape is
verified — Patterns A/B wire ``fatal()`` into every ``mkdir -p`` and into the
appendability probe, Pattern C routes the probe through ``SETUP_OK=0`` into its
alert arm, and each exempt wrapper's recorded not-applicable reason is re-checked
rather than assumed. Keying on the VEHICLE's shape rather than on the fix's
marker is what makes a NEW unguarded wrapper, an alternate ``fatal`` spelling, or
a probe defused to ``|| true`` FAIL instead of being silently skipped; the
``test_scanner_fails_on_*`` fixtures below pin each of those.

That scan is why this file is a ``GLOB_SCAN_TESTS`` member of the Step 9c
selector's roster — a wrapper-only (``.sh``) diff reaches no stem-map or
import-map arm, so without that entry a regression in any of these wrappers would
select no test at all. (The roster path is deliberately NOT spelled here as a
repo-relative literal: this file never reads the selector, and the literal would
mint a false dependency edge on every selector diff.)
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The scan glob this file covers; pinned VERBATIM in the Step 9c selector's
# GLOB_SCAN_TESTS roster, whose live-tree drift pin asserts this exact literal
# still appears here. (Selector filenames are deliberately not spelled out in
# this file — the selector's basename-ref arm would mint a false dependency
# edge, putting this file in the selection of every unrelated selector diff.)
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


def _assert_exact_stdout(stdout: str, wrapper: _Wrapper, log_dir: Path) -> None:
    """Assert the happy-path stdout contract EXACTLY, not by substring.

    Plan v1 section 6 requires routine passes to stay silent: the six
    pointer-emitting wrappers emit the daily pointer line and NOTHING else, and
    ``cron_session_summarize.sh`` (no ``FIRST_RUN_OF_DAY`` mechanism) emits
    nothing at all. A substring check cannot see extra output, so any new line a
    refactor adds to a routine pass would ship green; the line COUNT is what
    pins the contract.

    Extracted from the parametrized happy-path test so the assertion itself is
    testable - see ``test_exact_stdout_pin_rejects_an_extra_line``.
    """
    lines = stdout.splitlines()

    if wrapper.pointer_token is None:
        assert lines == [], (
            f"{wrapper.script}: a routine pass must emit NOTHING on stdout, got {lines!r}"
        )
        return

    assert len(lines) == 1, (
        f"{wrapper.script}: a routine pass must emit EXACTLY the daily pointer line, "
        f"got {len(lines)} line(s): {lines!r}"
    )
    (pointer,) = lines
    assert wrapper.pointer_token in pointer, f"{wrapper.script}: {pointer!r}"
    assert "per-pass output" in pointer, f"{wrapper.script}: {pointer!r}"
    assert str(log_dir) in pointer, f"{wrapper.script}: {pointer!r}"


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

    # EXACT stdout, not substrings: a routine pass emits the daily pointer line
    # and nothing else (or nothing at all for cron_session_summarize.sh).
    _assert_exact_stdout(result.stdout, wrapper, h["log_dir"])


def test_exact_stdout_pin_rejects_an_extra_line():
    """Falsification for ``_assert_exact_stdout``: a real happy-path stdout with
    ONE extra line must FAIL, and empty-stdout wrappers must reject ANY line.

    Without this the exactness claim is untested - the pre-#2386-revise pin
    checked three substrings, so arbitrary extra stdout passed.
    """
    log_dir = Path("/tmp/eps-fake-log-dir")
    pointer_wrapper = next(w for w in _WRAPPERS if w.pointer_token is not None)
    silent_wrapper = next(w for w in _WRAPPERS if w.pointer_token is None)

    good = (
        f"2026-01-01T00:00:00+00:00 {pointer_wrapper.pointer_token} per-pass output "
        f"to {log_dir} (this file receives only this daily pointer line)"
    )
    # Control: the genuine single-line shape passes.
    _assert_exact_stdout(good + "\n", pointer_wrapper, log_dir)
    _assert_exact_stdout("", silent_wrapper, log_dir)

    with pytest.raises(AssertionError, match="EXACTLY the daily pointer line"):
        _assert_exact_stdout(good + "\nsome other chatter\n", pointer_wrapper, log_dir)

    with pytest.raises(AssertionError, match="must emit NOTHING"):
        _assert_exact_stdout("some other chatter\n", silent_wrapper, log_dir)


# ── Glob-scan invariant over the whole cron wrapper family ───────────────────


class _LogicalLine(NamedTuple):
    """A shell statement plus the index of the physical line it starts on."""

    text: str
    start: int


def _logical_lines(text: str) -> list[_LogicalLine]:
    """Join backslash-continued shell lines into single logical statements."""
    out: list[_LogicalLine] = []
    buf = ""
    start = 0
    for idx, raw in enumerate(text.splitlines()):
        if not buf:
            start = idx
        buf += raw
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1] + " "
            continue
        out.append(_LogicalLine(buf, start))
        buf = ""
    if buf:
        out.append(_LogicalLine(buf, start))
    return out


def _statements(text: str, needle: str) -> list[_LogicalLine]:
    """Non-comment logical statements containing ``needle``."""
    return [
        ln
        for ln in _logical_lines(text)
        if needle in ln.text and not ln.text.lstrip().startswith("#")
    ]


def _depth_delta(stripped: str) -> int:
    """Net ``if``/``fi`` nesting change contributed by one physical line.

    Segment-aware, so a ONE-LINE ``if c; then x; fi`` nets zero. The pre-fix
    scan looked only at the line's FIRST token: a one-line ``if`` incremented
    depth while its same-line ``fi`` never decremented it, so the scan ran past
    the enclosing block's own ``else`` and credited ELSE-arm statements to the
    THEN arm (round-3 review nit ``if-then-branch-oneline-if-depth-gap``).
    """
    delta = 0
    for segment in stripped.split(";"):
        seg = segment.strip()
        if seg == "if" or seg.startswith("if "):
            delta += 1
        elif seg == "fi" or seg.startswith("fi "):
            delta -= 1
    return delta


def _if_then_branch(lines: list[str], start: int) -> list[str]:
    """Physical lines in the THEN branch of the ``if`` opening at ``lines[start]``.

    The scan stops at this block's OWN ``else`` / ``elif`` as well as at its
    matching ``fi``, so a statement in the ELSE arm can never be credited to the
    THEN arm. Nesting is tracked through :func:`_depth_delta`, so neither a
    nested block's ``else`` nor a one-line ``if ...; then ...; fi`` ends the
    scan early or lets it run long.
    """
    depth = 0
    body: list[str] = []
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if depth == 1 and (stripped == "else" or stripped.startswith("elif ")):
            return body
        if idx > start:
            body.append(lines[idx])
        depth += _depth_delta(stripped)
        if idx > start and depth <= 0:
            return body
    return body


_FATAL_DEF = re.compile(r"^\s*fatal\s*\(\)\s*\{")
_SET_E = re.compile(r"^\s*set\s+-[a-z]*e")
# `if ! <cmd> ...; then` — the opener whose block body is the CONSEQUENCE of
# <cmd> FAILING. Matched against a whitespace-normalized logical statement.
_NEGATED_IF_OPENER = re.compile(r"^if\s+!\s+\S")
_RC_NONZERO_ASSIGN = re.compile(r"^\s*rc=[1-9][0-9]*\s*$")
# A GROUP log vehicle: `} >> "$LOG"`, `) > "$LOG"`, `done >> "$LOG"`, `} 2>&1`.
# A failed redirect here skips the whole group — the #2196 failure mode.
_GROUP_VEHICLE = re.compile(r"^\s*(?:\}|\)|done)\s*[0-9]*>")


def _routes_own_failure_into_setup_ok(lines: list[str], stmt: _LogicalLine) -> bool:
    """True when ``stmt`` routes ITS OWN failure into ``SETUP_OK=0``.

    Two accepted spellings, both of which make the assignment the consequence of
    THIS statement failing:

    * ``<stmt> || SETUP_OK=0`` — the short-circuit form (Pattern C's analogue of
      Patterns A/B's ``|| fatal``), read off the statement's own text.
    * ``if ! <stmt> ...; then`` — a NEGATED ``if`` whose OWN THEN branch carries
      the assignment.

    Scanning forward to the next ``fi`` (the pre-revise shape) is deliberately
    NOT accepted. That scan credits any ``SETUP_OK=0`` in the region below, so a
    statement defused to fail open — ``probe || true`` — passes by borrowing the
    assignment of an unrelated guarded check placed after it. The assignment
    must belong to the branch THIS statement opens (round-2 review nit
    ``pattern-c-probe-check-fail-open``).
    """
    if re.search(r"\|\|\s*SETUP_OK=0(?:\s|;|$)", stmt.text):
        return True
    normalized = " ".join(stmt.text.split())
    if not (_NEGATED_IF_OPENER.match(normalized) and normalized.endswith("then")):
        return False
    return any(ln.strip() == "SETUP_OK=0" for ln in _if_then_branch(lines, stmt.start))


# --- The class invariant's explicit classification ---------------------------
#
# EVERY wrapper the live glob returns is classified here, and the scanner
# asserts SET EQUALITY between this map and the glob. That is the property the
# pre-revise invariant lacked: it short-circuited on `fatal() {` at column zero,
# so a wrapper with any other spelling - or a NEW wrapper with no guard at all -
# was silently skipped rather than failing.
#
#   "fatal-guard"     Patterns A/B (plan v1 section 4.3): fatal() helper, every
#                     mkdir -p guarded by `|| fatal`, probe routed to `|| fatal`.
#                     The ten #2386 fix-set wrappers + the #2196 reference.
#   "setup-ok-guard"  Pattern C (plan v2): NO fatal() by design - an early
#                     exit 1 would skip the alert arm - so the probe routes
#                     through SETUP_OK=0 into the wrapper's own alert path,
#                     and the WHOLE chain is checked link by link:
#                     probe -> SETUP_OK=0 -> rc != 0 -> alert_failure.
#   "exempt-set-e"    `set -e` already makes both legs fail loud.
#   "exempt-per-command-append"
#                     no mkdir -p site to guard AND no brace-group log vehicle:
#                     each line is appended by its own `echo ... >> "$LOG"`, so
#                     a failed append loses one line instead of skipping the
#                     pass. BOTH halves are re-checked - the second was
#                     previously assumed rather than asserted.
_FATAL_GUARD = "fatal-guard"
_SETUP_OK_GUARD = "setup-ok-guard"
_EXEMPT_SET_E = "exempt-set-e"
_EXEMPT_PER_COMMAND_APPEND = "exempt-per-command-append"

_WRAPPER_CLASSIFICATION: dict[str, str] = {
    # Patterns A/B - the ten wrappers #2386 fixed, plus the #2196 reference.
    "cron_autonomous_session_watch.sh": _FATAL_GUARD,
    "cron_codex_reaper.sh": _FATAL_GUARD,
    "cron_daily_healthcheck.sh": _FATAL_GUARD,
    "cron_gcp_audit.sh": _FATAL_GUARD,
    "cron_lesson_consolidate.sh": _FATAL_GUARD,
    "cron_pod_audit.sh": _FATAL_GUARD,
    "cron_session_summarize.sh": _FATAL_GUARD,
    "cron_step9c_ledger_refresh.sh": _FATAL_GUARD,
    "cron_uv_cache_prune.sh": _FATAL_GUARD,
    "cron_vm_disk_guard.sh": _FATAL_GUARD,
    "cron_worktree_audit.sh": _FATAL_GUARD,
    # Pattern C.
    "cron_codex_auto_upgrade.sh": _SETUP_OK_GUARD,
    # Recorded NOT-APPLICABLE, each with its reason re-checked below.
    "cron_export_literature.sh": _EXEMPT_SET_E,
    "cron_watch_issue_1739.sh": _EXEMPT_PER_COMMAND_APPEND,
    "cron_watch_issue_2091.sh": _EXEMPT_PER_COMMAND_APPEND,
}


def _check_fatal_guard(path: Path, text: str) -> None:
    assert any(_FATAL_DEF.match(ln) for ln in text.splitlines()), (
        f"{path.name}: classified {_FATAL_GUARD!r} but defines no fatal() helper"
    )

    mkdir_stmts = _statements(text, "mkdir -p")
    assert mkdir_stmts, f"{path.name}: classified {_FATAL_GUARD!r} but has no mkdir -p"
    for stmt in mkdir_stmts:
        assert "|| fatal" in stmt.text, (
            f"{path.name}: unguarded mkdir -p - an uncreatable log dir would "
            f"silently skip the pass: {stmt.text.strip()[:100]!r}"
        )

    probes = _statements(text, ': >> "$LOG_FILE"')
    assert probes, (
        f"{path.name}: missing the appendability probe - mkdir -p succeeds on "
        "an existing dir regardless of writability"
    )
    for probe in probes:
        # The probe is checked as a LOGICAL statement that ROUTES INTO the
        # failure path. Asserting only that a line starts with the probe text
        # would pass `: >> "$LOG_FILE" || true`.
        assert "|| fatal" in probe.text, (
            f"{path.name}: the appendability probe does not route into fatal() - "
            f"a failed probe would be swallowed: {probe.text.strip()[:100]!r}"
        )


def _check_setup_ok_guard(path: Path, text: str) -> None:
    lines = text.splitlines()

    assert not any(_FATAL_DEF.match(ln) for ln in lines), (
        f"{path.name}: classified {_SETUP_OK_GUARD!r} but defines fatal(); an early "
        "exit would skip the alert arm this wrapper depends on"
    )

    probes = _statements(text, ': >> "$LOG_FILE"')
    assert probes, (
        f"{path.name}: missing the appendability probe - its checked mkdir -p "
        "returns success for an existing dir whatever its mode"
    )
    for probe in probes:
        assert _routes_own_failure_into_setup_ok(lines, probe), (
            f"{path.name}: the appendability probe does not route ITS OWN failure "
            f"into SETUP_OK=0 - a failed probe would be swallowed: "
            f"{probe.text.strip()[:100]!r}"
        )

    mkdir_stmts = _statements(text, "mkdir -p")
    assert mkdir_stmts, f"{path.name}: classified {_SETUP_OK_GUARD!r} but has no mkdir -p"
    for stmt in mkdir_stmts:
        assert _routes_own_failure_into_setup_ok(lines, stmt), (
            f"{path.name}: unchecked mkdir -p - it does not route ITS OWN failure "
            f"into SETUP_OK=0: {stmt.text.strip()[:100]!r}"
        )

    # SETUP_OK=0 is only a guard if it still reaches the alert arm, so the rest
    # of the chain is checked LINK BY LINK rather than as three independent
    # line-presence probes: the SETUP_OK test must set a non-zero rc in its OWN
    # then-branch, and the rc test must call alert_failure in ITS OWN then-branch.
    # Presence anywhere in the file proves nothing about connectivity - the
    # statements could sit in unrelated blocks, or in the else arms that run on
    # the HEALTHY path.
    setup_tests = _statements(text, 'if [ "$SETUP_OK" -ne 1 ]; then')
    assert setup_tests, f"{path.name}: SETUP_OK is set but never tested"
    assert any(
        any(_RC_NONZERO_ASSIGN.match(ln) for ln in _if_then_branch(lines, t.start))
        for t in setup_tests
    ), (
        f"{path.name}: the SETUP_OK test assigns no non-zero rc in its own "
        "then-branch, so SETUP_OK=0 never reaches the rc test"
    )

    rc_tests = _statements(text, 'if [ "${rc:-0}" -ne 0 ]; then')
    assert rc_tests, f"{path.name}: rc is never tested, so SETUP_OK cannot reach the alert arm"
    assert any(
        any(ln.strip().startswith("alert_failure ") for ln in _if_then_branch(lines, t.start))
        for t in rc_tests
    ), (
        f"{path.name}: the rc test does not call alert_failure in its own "
        "then-branch - the failure would be silent"
    )


def _check_exempt_set_e(path: Path, text: str) -> None:
    assert any(_SET_E.match(ln) for ln in text.splitlines()), (
        f"{path.name}: classified {_EXEMPT_SET_E!r} but carries no `set -e`; its "
        "recorded not-applicable reason no longer holds"
    )


def _check_exempt_per_command_append(path: Path, text: str) -> None:
    """Re-check BOTH halves of this class's recorded not-applicable reason.

    (a) No ``mkdir -p`` site, so there is no uncreatable-dir leg to guard.
    (b) No brace-group / loop log VEHICLE - every line is appended by its own
    ``echo ... >> "$LOG"``, so a failed append loses one line instead of
    skipping the whole pass. (b) is what makes the exclusion sound, and it was
    ASSUMED rather than asserted: without this check either wrapper could grow
    a ``} >> "$LOG"`` vehicle and keep its exemption silently, which is the same
    false-premise-exclusion shape this task exists to close (round-2 review nit
    ``cron-guard-scan-residual-vehicle-blind-spots``).

    The vehicle test is deliberately conservative: it fires on ANY group / loop
    closer carrying a redirect, not only one onto the log file. A wrapper in
    this class has no such construct today, and one appearing is reason to
    re-derive the exemption rather than to widen the pattern.
    """
    mkdir_stmts = _statements(text, "mkdir -p")
    assert not mkdir_stmts, (
        f"{path.name}: classified {_EXEMPT_PER_COMMAND_APPEND!r} but now has a "
        f"mkdir -p site; its recorded not-applicable reason no longer holds: "
        f"{[s.text.strip()[:80] for s in mkdir_stmts]}"
    )

    vehicles = [ln for ln in text.splitlines() if _GROUP_VEHICLE.match(ln)]
    assert not vehicles, (
        f"{path.name}: classified {_EXEMPT_PER_COMMAND_APPEND!r} but now closes a "
        f"group/loop with a redirect; a failed redirect there skips everything "
        f"inside, which is exactly the failure this class was exempted from: "
        f"{[v.strip()[:80] for v in vehicles]}"
    )


_CHECKERS = {
    _FATAL_GUARD: _check_fatal_guard,
    _SETUP_OK_GUARD: _check_setup_ok_guard,
    _EXEMPT_SET_E: _check_exempt_set_e,
    _EXEMPT_PER_COMMAND_APPEND: _check_exempt_per_command_append,
}


def _scan_wrappers(root: Path, classification: dict[str, str]) -> None:
    """Assert every wrapper under ``root`` matches its declared guard shape.

    Raises ``AssertionError`` on the first violation. Pure over ``root`` +
    ``classification`` so the falsification fixtures below can drive it against
    synthetic trees.
    """
    wrappers = sorted(root.glob(_CRON_WRAPPER_GLOB))
    assert wrappers, f"no wrappers matched {_CRON_WRAPPER_GLOB!r}"

    found = {p.name for p in wrappers}
    declared = set(classification)
    unclassified = sorted(found - declared)
    assert not unclassified, (
        f"cron wrapper(s) with no declared guard class: {unclassified}. Every "
        f"scripts/cron_*.sh must be classified (one of {sorted(_CHECKERS)}) so a NEW "
        "wrapper cannot join the family unguarded and unnoticed."
    )
    stale = sorted(declared - found)
    assert not stale, f"classification names wrapper(s) that no longer exist: {stale}"

    for path in wrappers:
        cls = classification[path.name]
        assert cls in _CHECKERS, f"{path.name}: unknown guard class {cls!r}"
        _CHECKERS[cls](path, path.read_text())


def test_every_cron_wrapper_matches_its_declared_guard_shape():
    """Class invariant over ``scripts/cron_*.sh``, keyed on the VEHICLE's shape
    rather than on the fix's marker.

    Four properties earlier versions lacked, each of which let a real gap
    through (round-1 and round-2 review, both reviewers):

    1. EXHAUSTIVE. Every wrapper the live glob returns is classified, and the
       classification set must equal the glob set. The old scan short-circuited
       on ``fatal() {`` at column zero, so any wrapper without that exact
       spelling was silently skipped and its final ``missing`` check covered
       only the seven driven wrappers.
    2. The probe is checked as a LOGICAL STATEMENT that routes ITS OWN failure
       into the wrapper's failure path (``|| fatal`` for Patterns A/B, the
       ``SETUP_OK=0`` arm for Pattern C). The pre-revise assertion accepted any
       physical line starting with the probe text, so ``probe || true`` passed;
       the round-2 version then credited any ``SETUP_OK=0`` below the probe, so
       ``probe || true`` still passed whenever an unrelated guarded check sat
       underneath it. The assignment must belong to the branch the probe opens.
    3. Pattern C's chain is checked LINK BY LINK - probe -> ``SETUP_OK=0`` ->
       non-zero ``rc`` in the SETUP_OK test's own then-branch -> ``alert_failure``
       in the rc test's own then-branch. Three independent line-presence probes
       cannot tell a connected chain from three statements in unrelated blocks.
    4. Exemptions are FALSIFIABLE - EVERY half of each recorded not-applicable
       reason is re-checked (``set -e``; no mkdir site AND no brace-group log
       vehicle), so a wrapper that loses any half fails instead of staying
       quietly exempt.
    """
    _scan_wrappers(_REPO_ROOT, _WRAPPER_CLASSIFICATION)


# -- Falsification: each defect the scanner is meant to catch actually FAILs ---


def _fixture_tree(tmp_path: Path, wrappers: dict[str, str]) -> Path:
    root = tmp_path / "tree"
    (root / "scripts").mkdir(parents=True)
    for name, body in wrappers.items():
        (root / "scripts" / name).write_text(body)
    return root


# The two guarded statements are named so each fixture below can defuse exactly
# one of them by identity. Spelling them out again inside a `.replace()` would
# let the fixture silently drift from the baseline (a no-op replace, hence a
# vacuous test) the moment either statement is reworded.
_FIXTURE_MKDIR = (
    'mkdir -p "$LOG_DIR" \\\n    || fatal "cannot create log dir (LOG_DIR=$LOG_DIR); x NOT run"'
)
_FIXTURE_PROBE = (
    ': >> "$LOG_FILE" 2>/dev/null \\\n'
    '    || fatal "daily log file not appendable ($LOG_FILE); x NOT run"'
)

_GUARDED_BODY = "\n".join(
    [
        "#!/bin/bash",
        "set -uo pipefail",
        'LOG_DIR="${EPS_X_LOG_DIR:-/tmp/x}"',
        'LOG_FILE="$LOG_DIR/day.log"',
        "",
        "fatal() {",
        '    echo "$(date -Iseconds) FATAL: $1" >&2',
        "    exit 1",
        "}",
        "",
        _FIXTURE_MKDIR,
        "",
        "FIRST_RUN_OF_DAY=0",
        '[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1',
        "",
        _FIXTURE_PROBE,
        "",
        "{",
        "    echo hi",
        '} >> "$LOG_FILE" 2>&1',
        "exit 0",
        "",
    ]
)


def test_scanner_control_passes_on_a_well_formed_fixture(tmp_path: Path):
    """Control: the fixture shapes below differ from this baseline ONLY in the
    defect under test, so each failure is attributable to that defect."""
    root = _fixture_tree(tmp_path, {"cron_x.sh": _GUARDED_BODY})
    _scan_wrappers(root, {"cron_x.sh": _FATAL_GUARD})


def test_scanner_fails_on_a_new_unguarded_wrapper(tmp_path: Path):
    """(a) A NEW wrapper nobody classified must FAIL, not be skipped.

    This is the round-1 hole: the old scan skipped any wrapper without a
    column-zero ``fatal() {`` line, so an unguarded newcomer passed silently.
    """
    newcomer = '#!/bin/bash\nset -uo pipefail\nmkdir -p "$LOG_DIR"\nexit 0\n'
    root = _fixture_tree(tmp_path, {"cron_x.sh": _GUARDED_BODY, "cron_new.sh": newcomer})

    with pytest.raises(AssertionError, match="no declared guard class"):
        _scan_wrappers(root, {"cron_x.sh": _FATAL_GUARD})


def test_scanner_fails_on_an_alternate_fatal_spelling_that_hides_an_unguarded_mkdir(
    tmp_path: Path,
):
    """(b) An alternate ``fatal`` spelling must not buy an escape.

    Pre-revise, ``fatal ()  {`` failed the ``startswith("fatal() {")`` probe, so
    the wrapper was skipped and its unguarded ``mkdir -p`` never checked. The
    skip was caught ONLY for the seven wrappers this file drives, because the
    final ``missing`` check listed exactly those seven; for the other eight
    (including all four Pattern A/B wrappers with their own test file, and the
    #2196 reference) the skip was completely silent. Measured on a synthetic
    tree of the seven healthy driven wrappers plus one non-driven wrapper in
    this shape: OLD scan PASSes, this scan FAILs.

    The classification now drives the check, so the spelling is irrelevant and
    the unguarded mkdir is caught wherever it lives.
    """
    body = _GUARDED_BODY.replace("fatal() {", "fatal ()  {").replace(
        _FIXTURE_MKDIR, 'mkdir -p "$LOG_DIR"'
    )
    assert 'mkdir -p "$LOG_DIR"\n' in body, "fixture did not actually unguard the mkdir"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="unguarded mkdir -p"):
        _scan_wrappers(root, {"cron_x.sh": _FATAL_GUARD})


def test_scanner_fails_on_a_probe_that_fails_open(tmp_path: Path):
    """(c) ``probe || true`` must FAIL.

    The pre-revise assertion only required a physical line STARTING with the
    probe text, so a probe changed to swallow its own failure passed.
    """
    body = _GUARDED_BODY.replace(_FIXTURE_PROBE, ': >> "$LOG_FILE" 2>/dev/null || true')
    assert "|| true" in body, "fixture did not actually defuse the probe"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="does not route into fatal"):
        _scan_wrappers(root, {"cron_x.sh": _FATAL_GUARD})


def test_scanner_fails_on_a_missing_probe(tmp_path: Path):
    """A wrapper that drops the probe entirely must FAIL - mkdir -p alone cannot
    see an exists-but-unwritable dir (the Pattern C gap, in Pattern A shape)."""
    body = _GUARDED_BODY.replace(_FIXTURE_PROBE, "")
    assert ': >> "$LOG_FILE"' not in body, "fixture did not actually drop the probe"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="missing the appendability probe"):
        _scan_wrappers(root, {"cron_x.sh": _FATAL_GUARD})


def test_scanner_fails_on_a_pattern_c_probe_that_does_not_bind_setup_ok(tmp_path: Path):
    """Pattern C's own fail-open shape: the probe runs but its block never sets
    ``SETUP_OK=0``, so the failure is logged and then swallowed."""
    body = """#!/bin/bash
set -uo pipefail
SETUP_OK=1
if ! mkdir_err=$(mkdir -p "$LOG_DIR" 2>&1); then
    log_line "FATAL: $mkdir_err"
    SETUP_OK=0
fi
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1
if ! : >> "$LOG_FILE" 2>/dev/null; then
    log_line "FATAL: not appendable"
fi
if [ "$SETUP_OK" -ne 1 ]; then
    rc=1
fi
if [ "${rc:-0}" -ne 0 ]; then
    alert_failure "$rc"
fi
exit 0
"""
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="does not route ITS OWN failure"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


# The Pattern C chain, link by link. Each fixture below defuses exactly ONE
# link and leaves every other link intact, so the failure is attributable.
# `_PATTERN_C_LINKS` is the healthy baseline; the control test asserts it PASSes.
_PATTERN_C_LINKS = """#!/bin/bash
set -uo pipefail
SETUP_OK=1
if ! mkdir_err=$(mkdir -p "$LOG_DIR" 2>&1); then
    log_line "FATAL: $mkdir_err"
    SETUP_OK=0
fi
FIRST_RUN_OF_DAY=0
[ -f "$LOG_FILE" ] || FIRST_RUN_OF_DAY=1
if ! : >> "$LOG_FILE" 2>/dev/null; then
    log_line "FATAL: not appendable"
    SETUP_OK=0
fi
if [ "$SETUP_OK" -ne 1 ]; then
    rc=1
else
    {
        echo hi
    } >> "$LOG_FILE" 2>&1
    rc=$?
fi
if [ "${rc:-0}" -ne 0 ]; then
    alert_failure "$rc"
fi
exit 0
"""


def test_pattern_c_control_passes_on_a_well_formed_chain(tmp_path: Path):
    """Control for the Pattern C fixtures: every link intact PASSes, so each
    failure below is attributable to the one link that fixture defuses."""
    root = _fixture_tree(tmp_path, {"cron_x.sh": _PATTERN_C_LINKS})
    _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_on_a_fail_open_probe_with_a_decoy_setup_ok_below(tmp_path: Path):
    """The round-2 hole: ``probe || true`` plus an UNRELATED ``SETUP_OK=0``
    below it.

    The round-2 check scanned forward from the probe to the next ``fi``, so the
    probe borrowed the assignment of the guarded check underneath and passed
    while swallowing its own failure — an exists-but-unwritable log file would
    again exit 0 silently. Everything else in the chain is intact here, so only
    the fail-open probe can explain the failure.
    """
    body = _PATTERN_C_LINKS.replace(
        'if ! : >> "$LOG_FILE" 2>/dev/null; then\n'
        '    log_line "FATAL: not appendable"\n'
        "    SETUP_OK=0\n"
        "fi\n",
        ': >> "$LOG_FILE" 2>/dev/null || true\n'
        "if ! command -v uv >/dev/null 2>&1; then\n"
        '    log_line "FATAL: uv missing"\n'
        "    SETUP_OK=0\n"
        "fi\n",
    )
    assert "|| true" in body, "fixture did not actually defuse the probe"
    assert "command -v uv" in body, "fixture did not actually place the decoy"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="does not route ITS OWN failure"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_when_a_oneline_if_hides_setup_ok_in_the_else_arm(tmp_path: Path):
    """The round-3 hole: a one-line ``if ...; then ...; fi`` inside the THEN arm
    used to misbalance the depth counter, so the scan ran past the probe's OWN
    ``else`` and credited an ELSE-arm ``SETUP_OK=0`` to the THEN arm.

    The assignment here fires only when the probe SUCCEEDS, so an
    exists-but-unwritable log file leaves ``SETUP_OK=1`` and the wrapper exits 0
    silently — precisely the failure class this task sweeps. Every other link in
    the chain is intact, so only the misattributed branch can explain the
    failure. Pinned because the pre-fix scan PASSED this fixture.
    """
    body = _PATTERN_C_LINKS.replace(
        'if ! : >> "$LOG_FILE" 2>/dev/null; then\n'
        '    log_line "FATAL: not appendable"\n'
        "    SETUP_OK=0\n"
        "fi\n",
        'if ! : >> "$LOG_FILE" 2>/dev/null; then\n'
        "    if true; then :; fi\n"
        "else\n"
        "    SETUP_OK=0\n"
        "fi\n",
    )
    assert "if true; then :; fi" in body, "fixture did not place the one-line if"
    assert body != _PATTERN_C_LINKS, "fixture did not modify the baseline"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="does not route ITS OWN failure"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_oneline_if_in_the_then_arm_is_still_accepted(tmp_path: Path):
    """Negative control for the fixture above: a one-line ``if`` is ordinary
    shell, so the SAME construct with ``SETUP_OK=0`` genuinely in the THEN arm
    must keep PASSing. Without this, the depth fix could 'pass' by rejecting
    every one-line conditional.
    """
    body = _PATTERN_C_LINKS.replace(
        '    log_line "FATAL: not appendable"\n',
        '    log_line "FATAL: not appendable"\n    if true; then :; fi\n',
    )
    assert "if true; then :; fi" in body, "fixture did not place the one-line if"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})
    _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_when_setup_ok_never_sets_a_nonzero_rc(tmp_path: Path):
    """Link 2: ``SETUP_OK=0`` is set and tested, but the test's then-branch never
    assigns a non-zero ``rc``, so the failure never reaches the alert arm.

    The ``rc=1`` moves into the ELSE arm — the HEALTHY path — which keeps the
    literal in the file and defeats a presence-anywhere check.
    """
    body = _PATTERN_C_LINKS.replace(
        'if [ "$SETUP_OK" -ne 1 ]; then\n    rc=1\nelse\n',
        'if [ "$SETUP_OK" -ne 1 ]; then\n    log_line "setup failed"\nelse\n    rc=1\n',
    )
    assert 'log_line "setup failed"' in body, "fixture did not actually break the link"
    assert "rc=1" in body, "fixture must keep the literal so presence alone cannot pass"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="assigns no non-zero rc in its own"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_when_the_rc_test_does_not_alert(tmp_path: Path):
    """Link 3: ``rc`` is tested but ``alert_failure`` is called on the OTHER arm,
    so a non-zero rc is silent while the literal still appears in the file."""
    body = _PATTERN_C_LINKS.replace(
        'if [ "${rc:-0}" -ne 0 ]; then\n    alert_failure "$rc"\nfi\n',
        'if [ "${rc:-0}" -ne 0 ]; then\n    log_line "rc=$rc"\nelse\n    alert_failure "$rc"\nfi\n',
    )
    assert "alert_failure" in body, "fixture must keep the literal so presence alone cannot pass"
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="does not call alert_failure in its own"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_when_a_per_command_append_wrapper_gains_a_group_vehicle(
    tmp_path: Path,
):
    """The exclusion's second half, asserted rather than assumed.

    ``cron_watch_issue_1739.sh`` / ``cron_watch_issue_2091.sh`` are exempt
    because they append PER COMMAND, so a failed append loses one line instead
    of skipping the pass. Give such a wrapper a brace-group log vehicle and the
    premise is false — the invariant must fire rather than keep excluding it.
    """
    body = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "LOG=/tmp/watch.log\n"
        'echo "one line" >> "$LOG"\n'
        "{\n"
        "    echo hi\n"
        '} >> "$LOG" 2>&1\n'
        "exit 0\n"
    )
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})

    with pytest.raises(AssertionError, match="closes a group/loop with a redirect"):
        _scan_wrappers(root, {"cron_x.sh": _EXEMPT_PER_COMMAND_APPEND})


def test_per_command_append_control_passes_without_a_group_vehicle(tmp_path: Path):
    """Control for the fixture above: identical minus the brace group."""
    body = (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "LOG=/tmp/watch.log\n"
        'echo "one line" >> "$LOG"\n'
        'echo "another" >> "$LOG"\n'
        "exit 0\n"
    )
    root = _fixture_tree(tmp_path, {"cron_x.sh": body})
    _scan_wrappers(root, {"cron_x.sh": _EXEMPT_PER_COMMAND_APPEND})


def test_scanner_fails_when_an_exemption_reason_stops_holding(tmp_path: Path):
    """A recorded not-applicable reason must stay TRUE. Drop ``set -e`` from the
    ``exempt-set-e`` wrapper and the invariant fires instead of staying quiet."""
    root = _fixture_tree(tmp_path, {"cron_x.sh": "#!/bin/bash\nset -uo pipefail\nexit 0\n"})

    with pytest.raises(AssertionError, match="carries no `set -e`"):
        _scan_wrappers(root, {"cron_x.sh": _EXEMPT_SET_E})
