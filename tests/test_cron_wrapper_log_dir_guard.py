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


def _if_block_body(lines: list[str], start: int) -> list[str]:
    """Physical lines inside the ``if`` block opening at ``lines[start]``."""
    depth = 0
    body: list[str] = []
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if idx > start:
            body.append(lines[idx])
        if stripped == "if" or stripped.startswith("if "):
            depth += 1
        elif stripped == "fi" or stripped.startswith("fi "):
            depth -= 1
            if depth <= 0:
                return body
    return body


_FATAL_DEF = re.compile(r"^\s*fatal\s*\(\)\s*\{")
_SET_E = re.compile(r"^\s*set\s+-[a-z]*e")

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
#                     through SETUP_OK=0 into the wrapper's own alert path.
#   "exempt-set-e"    `set -e` already makes both legs fail loud.
#   "exempt-no-mkdir" per-command append vehicle, no mkdir site, pass not
#                     skipped by a failed append.
_FATAL_GUARD = "fatal-guard"
_SETUP_OK_GUARD = "setup-ok-guard"
_EXEMPT_SET_E = "exempt-set-e"
_EXEMPT_NO_MKDIR = "exempt-no-mkdir"

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
    "cron_watch_issue_1739.sh": _EXEMPT_NO_MKDIR,
    "cron_watch_issue_2091.sh": _EXEMPT_NO_MKDIR,
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
        body = _if_block_body(lines, probe.start)
        assert any(ln.strip() == "SETUP_OK=0" for ln in body), (
            f"{path.name}: the appendability probe does not route into SETUP_OK=0 - "
            f"a failed probe would be swallowed: {probe.text.strip()[:100]!r}"
        )

    mkdir_stmts = _statements(text, "mkdir -p")
    assert mkdir_stmts, f"{path.name}: classified {_SETUP_OK_GUARD!r} but has no mkdir -p"
    for stmt in mkdir_stmts:
        body = _if_block_body(lines, stmt.start)
        assert any(ln.strip() == "SETUP_OK=0" for ln in body), (
            f"{path.name}: unchecked mkdir -p: {stmt.text.strip()[:100]!r}"
        )

    # SETUP_OK=0 is only a guard if it still reaches the alert arm.
    assert any('if [ "$SETUP_OK" -ne 1 ]; then' in ln for ln in lines), (
        f"{path.name}: SETUP_OK is set but never tested"
    )
    assert any('if [ "${rc:-0}" -ne 0 ]; then' in ln for ln in lines), (
        f"{path.name}: rc is never tested, so SETUP_OK cannot reach the alert arm"
    )
    assert any(ln.strip().startswith("alert_failure ") for ln in lines), (
        f"{path.name}: no alert_failure call - the failure would be silent"
    )


def _check_exempt_set_e(path: Path, text: str) -> None:
    assert any(_SET_E.match(ln) for ln in text.splitlines()), (
        f"{path.name}: classified {_EXEMPT_SET_E!r} but carries no `set -e`; its "
        "recorded not-applicable reason no longer holds"
    )


def _check_exempt_no_mkdir(path: Path, text: str) -> None:
    mkdir_stmts = _statements(text, "mkdir -p")
    assert not mkdir_stmts, (
        f"{path.name}: classified {_EXEMPT_NO_MKDIR!r} but now has a mkdir -p site; "
        f"its recorded not-applicable reason no longer holds: "
        f"{[s.text.strip()[:80] for s in mkdir_stmts]}"
    )


_CHECKERS = {
    _FATAL_GUARD: _check_fatal_guard,
    _SETUP_OK_GUARD: _check_setup_ok_guard,
    _EXEMPT_SET_E: _check_exempt_set_e,
    _EXEMPT_NO_MKDIR: _check_exempt_no_mkdir,
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

    Three properties the pre-revise version lacked, each of which let a real gap
    through (round-1 review, both reviewers):

    1. EXHAUSTIVE. Every wrapper the live glob returns is classified, and the
       classification set must equal the glob set. The old scan short-circuited
       on ``fatal() {`` at column zero, so any wrapper without that exact
       spelling was silently skipped and its final ``missing`` check covered
       only the seven driven wrappers.
    2. The probe is checked as a LOGICAL STATEMENT that routes into the
       wrapper's failure path (``|| fatal`` for Patterns A/B, the ``SETUP_OK=0``
       arm for Pattern C). The old assertion accepted any physical line starting
       with the probe text, so ``probe || true`` would have passed.
    3. Exemptions are FALSIFIABLE - each recorded not-applicable reason
       (``set -e``, no mkdir site) is re-checked, so a wrapper that loses its
       reason fails instead of staying quietly exempt.
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

    with pytest.raises(AssertionError, match="does not route into SETUP_OK=0"):
        _scan_wrappers(root, {"cron_x.sh": _SETUP_OK_GUARD})


def test_scanner_fails_when_an_exemption_reason_stops_holding(tmp_path: Path):
    """A recorded not-applicable reason must stay TRUE. Drop ``set -e`` from the
    ``exempt-set-e`` wrapper and the invariant fires instead of staying quiet."""
    root = _fixture_tree(tmp_path, {"cron_x.sh": "#!/bin/bash\nset -uo pipefail\nexit 0\n"})

    with pytest.raises(AssertionError, match="carries no `set -e`"):
        _scan_wrappers(root, {"cron_x.sh": _EXEMPT_SET_E})
