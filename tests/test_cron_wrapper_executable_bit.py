"""Committed-executable-bit pins for the crontab-invoked cron wrappers (task #2645).

The crontab invokes `scripts/cron_*.sh` wrappers DIRECTLY by absolute path (no
`bash` prefix), so the executable bit is load-bearing for exactly this class.
`scripts/cron_step9c_ledger_refresh.sh` shipped committed at git mode 100644
(#2114) while its 14 siblings sat at 100755; every nightly fire for 18
consecutive nights died `Permission denied`, and because every alert channel
(dated log, audit sidecar, Telegram push) lives INSIDE the wrapper, the
non-execution was structurally unreportable. No existing test asserted the
COMMITTED mode — that is precisely why five review rounds on #2114 passed a
non-executable wrapper. This module is the pin.

Four arms:

1. ``test_every_cron_wrapper_is_executable_in_index`` (HARD, binding) — every
   ``scripts/cron_*.sh`` in the git INDEX is mode 100755. The index, not an
   on-disk stat, is what a fresh checkout materializes and what survives
   filesystems without mode tracking, so the pin reads
   ``git ls-files -s -- 'scripts/cron_*.sh'`` with the repo root as cwd.
2. ``test_cron_wrapper_glob_is_nonempty_and_contains_known_member``
   (NON-VACUITY) — the glob returns a non-empty set containing a known member,
   so a broken pattern, a wrong cwd, or a git failure can never produce a
   silent green on the hard arm.
3. ``test_negative_fixture_staged_100644_wrapper_is_reported`` (NEGATIVE
   FIXTURE) — a throwaway ``git init`` tree with a ``scripts/cron_fake.sh``
   staged at 100644, proving the SHARED mode-reading helper (the same
   ``_index_modes`` the hard arm calls) actually reports the violation, and
   that the fix command (``git update-index --chmod=+x``) clears it. A fixture
   re-implementing the index parse would prove nothing about the production
   helper.
4. ``test_crontab_referenced_repo_scripts_are_executable_in_index`` (SOFT) —
   when ``crontab -l`` is readable, every repo script path in COMMAND POSITION
   on a non-comment crontab line must be 100755 in the index. Skipped (never
   red) on machines with no crontab — nonzero exit, missing binary
   (``FileNotFoundError``), or a hung invocation (``TimeoutExpired``); pods,
   CI images, fresh clones. This catches a future crontab-referenced wrapper
   that does NOT match ``cron_*.sh``.

Command-position extraction (arm 4 + its own unit test): only DIRECTLY-exec'd
command tokens count. Redirect targets (the live crontab redirects into
``logs/codex_auto_upgrade/cron.log`` — a repo path that is not a script and
not in the index) and interpreter-prefixed invocations (``bash <path>`` needs
no exec bit) are excluded BY PARSING, not by allowlist;
``test_command_position_extraction_ignores_redirects_and_interpreters`` pins
both exclusions on a fixture crontab so the live crontab is never the only
evidence the extraction is sound. Known accepted false negatives (safe
direction — a missed path is never flagged): scripts exec'd through wrapper
commands (``flock``/``env``/``timeout``/``nice``), ``$VAR``-prefixed paths,
and repo paths not tracked in the index (an untracked script has no committed
mode to pin).

Scope decision (recorded, measured 2026-08-29): the hard arm covers
``scripts/cron_*.sh`` — 15 files — NOT all shell scripts. Of 234 tracked
``scripts/*.sh``, 99 are committed 100644 and all 99 carry a shebang; those
are overwhelmingly per-issue dispatch scripts invoked as ``bash scripts/...``,
where the executable bit is genuinely optional, so a blanket "shebang means
0755" rule would ship 99 red findings on landing (the #1388 fleet-wedge
shape). ``cron_*.sh`` is the one class the crontab execs directly.

That committed-mode scan is why this file is a ``GLOB_SCAN_TESTS`` member of
the Step 9c selector's roster — a wrapper-only (``.sh``) diff reaches no
stem-map or import-map arm, so without that entry a mode regression on any
wrapper would select no test at all. (The roster path is deliberately NOT
spelled here as a repo-relative literal: this file never reads the selector,
and the literal would mint a false dependency edge on every selector diff.)
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The scan glob this file covers; pinned VERBATIM in the Step 9c selector's
# GLOB_SCAN_TESTS roster, whose live-tree drift pin asserts this exact literal
# still appears here. (Selector filenames are deliberately not spelled out in
# this file — the selector's basename-ref arm would mint a false dependency
# edge, putting this file in the selection of every unrelated selector diff.)
_CRON_WRAPPER_GLOB = "scripts/cron_*.sh"

# Non-vacuity anchor: a wrapper that predates this pin and is expected to stay.
_KNOWN_MEMBER = "scripts/cron_pod_audit.sh"

_EXECUTABLE_MODE = "100755"

# --- shared index-mode helper (hard arm + negative fixture use THIS ONE fn) ---


def _index_modes(tree_root: Path, *pathspecs: str) -> dict[str, str]:
    """Parse ``git ls-files -s -- <pathspecs>`` under *tree_root* into ``{path: mode}``.

    Reads the git INDEX (``<mode> <sha> <stage>\\t<path>`` rows), never an
    on-disk ``stat`` — the index is what a fresh clone materializes, so the pin
    holds on filesystems where mode tracking is off. Shared by the hard arm
    (called with the repo root) and the negative fixture (called with a
    throwaway tree): one parser, two trees. Raises on any git failure or
    unparseable row (fail fast, no silent empty result).
    """
    proc = subprocess.run(
        ["git", "ls-files", "-s", "--", *pathspecs],
        cwd=tree_root,
        capture_output=True,
        text=True,
        check=True,
    )
    modes: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        meta, sep, path = line.partition("\t")
        fields = meta.split()
        if not sep or len(fields) != 3:
            raise ValueError(f"unparseable `git ls-files -s` row: {line!r}")
        modes[path] = fields[0]
    return modes


def _mode_offenders(modes: dict[str, str]) -> dict[str, str]:
    """Return the subset of *modes* not committed at 100755 (the violation set)."""
    return {path: mode for path, mode in modes.items() if mode != _EXECUTABLE_MODE}


# --- arm 1: hard arm ----------------------------------------------------------


def test_every_cron_wrapper_is_executable_in_index() -> None:
    """Every scripts/cron_*.sh in the git index is committed mode 100755."""
    offenders = _mode_offenders(_index_modes(_REPO_ROOT, _CRON_WRAPPER_GLOB))
    assert not offenders, (
        "crontab-invoked wrappers committed without the executable bit "
        f"(crontab execs them directly, so 100644 means every fire dies "
        f"`Permission denied` on a fresh checkout — the #2645 incident): {offenders}. "
        "Fix: chmod +x <file> && git update-index --chmod=+x <file>."
    )


# --- arm 2: non-vacuity -------------------------------------------------------


def test_cron_wrapper_glob_is_nonempty_and_contains_known_member() -> None:
    """The scan glob matches a non-empty set containing a known wrapper."""
    modes = _index_modes(_REPO_ROOT, _CRON_WRAPPER_GLOB)
    assert modes, (
        f"`git ls-files -s -- '{_CRON_WRAPPER_GLOB}'` returned nothing — broken "
        "pattern, wrong cwd, or a git failure; the hard arm would be vacuously green."
    )
    assert _KNOWN_MEMBER in modes, (
        f"known member {_KNOWN_MEMBER} missing from the glob set {sorted(modes)} — "
        "if it was deliberately renamed/removed, update _KNOWN_MEMBER."
    )


# --- arm 3: negative fixture --------------------------------------------------


def _git(tree_root: Path, *args: str) -> None:
    """Run a git command in *tree_root*, raising on failure."""
    subprocess.run(["git", *args], cwd=tree_root, capture_output=True, text=True, check=True)


def test_negative_fixture_staged_100644_wrapper_is_reported() -> None:
    """The shared helper reports a wrapper staged at 100644; the fix command clears it.

    Uses ``tempfile.mkdtemp`` rather than ``tmp_path`` (concurrent pytest
    sessions prune ``/tmp/pytest-of-*`` numbered roots and can delete a live
    scratch dir under a subprocess-driving test).
    """
    scratch = Path(tempfile.mkdtemp(prefix="eps-cron-execbit-"))
    try:
        _git(scratch, "init", "-q")
        wrapper = scratch / "scripts" / "cron_fake.sh"
        wrapper.parent.mkdir()
        wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
        wrapper.chmod(0o644)
        _git(scratch, "add", "scripts/cron_fake.sh")
        # Deterministic regardless of filesystem mode support: force the index
        # bit off (the exact inverse of the #2645 fix command).
        _git(scratch, "update-index", "--chmod=-x", "scripts/cron_fake.sh")

        modes = _index_modes(scratch, _CRON_WRAPPER_GLOB)
        assert modes == {"scripts/cron_fake.sh": "100644"}
        assert _mode_offenders(modes) == {"scripts/cron_fake.sh": "100644"}, (
            "the shared mode-reading helper failed to report a staged 100644 "
            "wrapper — the hard arm's pin is hollow"
        )

        # And the documented fix command flips the verdict green.
        _git(scratch, "update-index", "--chmod=+x", "scripts/cron_fake.sh")
        assert _mode_offenders(_index_modes(scratch, _CRON_WRAPPER_GLOB)) == {}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- arm 4: soft crontab arm + its extraction unit test -----------------------

# Interpreters whose script ARGUMENT needs no exec bit (`bash <path>` runs fine
# on a 100644 file — flagging it would red legitimate crontab lines).
_INTERPRETERS = frozenset(
    {"bash", "sh", "dash", "zsh", "python", "python3", "uv", "node", "perl", "ruby"}
)
_SEPARATORS = frozenset({"&&", "||", ";", "|", "&"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `>>`/`>`/`<`/`2>`/`2>>`/`&>`/`&>>` (+ optional attached or `&1`-style target).
_REDIRECT_RE = re.compile(r"^(?:\d*(?:>>?|<)|&>>?)(.*)$")


def _command_position_tokens(crontab_text: str) -> list[str]:
    """Extract COMMAND-POSITION tokens from crontab text (directly-exec'd only).

    Per non-comment, non-env-assignment line: strip the schedule (5 fields, or
    one ``@keyword`` field), then walk the command with shell-ish tokenization.
    Only the FIRST post-assignment token of each ``&&``/``||``/``;``/``|``
    segment is a command; redirect operators consume their target token (a
    redirect target is never a command — the C1 concern: the live crontab
    redirects into a repo-relative log file); an interpreter command
    (``bash``/``sh``/``uv``/...) marks its whole segment as NOT directly
    exec'd; a ``#`` token ends the line (trailing shell comment). Raises on
    untokenizable command text (fail fast) — the soft arm is already skipped
    wholesale when no crontab is readable.
    """
    tokens_out: list[str] = []
    for raw_line in crontab_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _ASSIGNMENT_RE.match(line):
            continue  # crontab environment line (PATH=..., MAILTO=...)
        fields = line.split(None, 1) if line.startswith("@") else line.split(None, 5)
        if len(fields) < (2 if line.startswith("@") else 6):
            continue  # schedule with no command field
        expect_command = True
        skip_next = False
        for tok in shlex.split(fields[-1]):
            if tok.startswith("#"):
                break  # trailing shell comment: rest of the line is inert
            if skip_next:
                skip_next = False
                continue
            if tok in _SEPARATORS:
                expect_command = True
                continue
            redirect = _REDIRECT_RE.match(tok)
            if redirect:
                # A detached target (`>> path`) rides the NEXT token; an
                # attached one (`>>path`, `2>&1`) is already consumed here.
                skip_next = redirect.group(1) == ""
                continue
            if not expect_command:
                continue
            if _ASSIGNMENT_RE.match(tok):
                continue  # leading VAR=VAL keeps command position
            expect_command = False
            if Path(tok).name in _INTERPRETERS:
                continue  # interpreter-prefixed: the script arg needs no exec bit
            tokens_out.append(tok)
    return tokens_out


_FIXTURE_ROOT = "/fake/eps"
_FR = _FIXTURE_ROOT  # short alias: keeps the fixture template's source lines under E501
_FIXTURE_CRONTAB = f"""\
# comment line naming a repo script that must be ignored: {_FR}/scripts/cron_commented.sh
PATH=/usr/local/bin:/usr/bin:/bin
31 5 * * * {_FR}/scripts/cron_direct.sh >> {_FR}/logs/codex_auto_upgrade/cron.log 2>&1
17 7 * * * EPS_DRY_RUN=1 {_FR}/scripts/cron_env_prefixed.sh
45 3 * * * bash {_FR}/scripts/cron_interp.sh  # && {_FR}/scripts/cron_ghost.sh
*/5 * * * * test -f {_FR}/scripts/cron_guarded.sh && {_FR}/scripts/cron_guarded.sh
"""


def test_command_position_extraction_ignores_redirects_and_interpreters() -> None:
    """C1 pin: redirect targets + interpreter-prefixed scripts are never flagged.

    Driven on a FIXTURE crontab, not the live one, so the exclusions stay
    pinned even on machines whose crontab lacks the offending shapes: (a) the
    redirect-target repo path (the live crontab's ``logs/.../cron.log`` shape)
    is not extracted; (b) an interpreter-prefixed script is not extracted;
    direct, env-prefixed, and post-``&&`` invocations ARE.
    """
    tokens = _command_position_tokens(_FIXTURE_CRONTAB)
    repo_paths = {t for t in tokens if t.startswith(f"{_FIXTURE_ROOT}/")}
    assert repo_paths == {
        f"{_FIXTURE_ROOT}/scripts/cron_direct.sh",
        f"{_FIXTURE_ROOT}/scripts/cron_env_prefixed.sh",
        f"{_FIXTURE_ROOT}/scripts/cron_guarded.sh",
    }
    assert f"{_FIXTURE_ROOT}/logs/codex_auto_upgrade/cron.log" not in tokens  # (a) redirect
    assert f"{_FIXTURE_ROOT}/scripts/cron_interp.sh" not in tokens  # (b) interpreter
    assert f"{_FIXTURE_ROOT}/scripts/cron_commented.sh" not in tokens  # comment line
    # Trailing-comment truncation is load-bearing: without the `#` break, the
    # `&&` inside the comment would reset command position and wrongly extract
    # the path that follows it.
    assert f"{_FIXTURE_ROOT}/scripts/cron_ghost.sh" not in tokens  # trailing comment


def _main_checkout_root() -> Path:
    """Resolve the MAIN checkout root (worktree-safe; crontab paths point there)."""
    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip()).parent


# Wall fence for the soft arm's one live-system read: `crontab -l` prints a
# small table and exits; a wedged invocation must skip, never stall the gate.
_CRONTAB_TIMEOUT_S = 10


def _read_crontab() -> subprocess.CompletedProcess[str]:
    """Run ``crontab -l`` (captured, text, short timeout) and return the proc.

    Module-level seam so the absence/timeout unit tests below can replace it.
    Raises ``FileNotFoundError`` (an ``OSError``) when the binary is absent
    (pods, minimal CI images) and ``subprocess.TimeoutExpired`` on a hang —
    the caller maps BOTH to ``pytest.skip``: this file is a GLOB_SCAN_TESTS
    member selected by any ``scripts/cron_*.sh`` diff, so on a crontab-less
    image the soft arm must skip cleanly, never crash the Step 9c gate.
    """
    return subprocess.run(
        ["crontab", "-l"],
        capture_output=True,
        text=True,
        timeout=_CRONTAB_TIMEOUT_S,
    )


def test_crontab_referenced_repo_scripts_are_executable_in_index() -> None:
    """SOFT arm: crontab-referenced, directly-exec'd repo scripts are 100755.

    Skips (never red) when no crontab is readable — nonzero exit (no crontab
    for this user), a missing ``crontab`` binary, or a hung invocation — or
    when the crontab references no directly-exec'd repo scripts. Crontab
    paths point at the MAIN checkout (resolved via the git common dir,
    worktree-safe); modes are read from THIS tree's index — the state under
    test. Referenced paths not in the index are skipped (no committed mode
    exists to pin).
    """
    try:
        proc = _read_crontab()
    except OSError as exc:
        # Covers FileNotFoundError: no crontab binary on this image.
        pytest.skip(f"crontab binary not runnable on this machine: {exc}")
    except subprocess.TimeoutExpired:
        pytest.skip(f"`crontab -l` timed out after {_CRONTAB_TIMEOUT_S}s (wedged invocation)")
    if proc.returncode != 0:
        pytest.skip("no readable crontab on this machine (pods, CI, fresh clones)")
    prefix = f"{_main_checkout_root()}/"
    rel_paths = sorted(
        {
            tok[len(prefix) :]
            for tok in _command_position_tokens(proc.stdout)
            if tok.startswith(prefix)
        }
    )
    if not rel_paths:
        pytest.skip("crontab references no directly-exec'd repo script paths")
    offenders = _mode_offenders(_index_modes(_REPO_ROOT, *rel_paths))
    assert not offenders, (
        "crontab directly execs these repo scripts, but they are not committed "
        f"100755 in the index (every fire dies `Permission denied`): {offenders}. "
        "Fix: chmod +x <file> && git update-index --chmod=+x <file>."
    )


# --- soft-arm skip-path unit tests (absence + hang are EXERCISED, not just handled) ---


def test_soft_arm_skips_cleanly_when_crontab_binary_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``crontab`` binary yields the documented clean skip, not a crash.

    ``returncode != 0`` covers only the no-crontab-for-this-user case; an
    absent BINARY raises ``FileNotFoundError`` from ``subprocess.run``. Before
    this pin the soft arm crashed on that path, and — as a GLOB_SCAN_TESTS
    member selected by any ``scripts/cron_*.sh`` diff — would have redded the
    Step 9c gate with a traceback on any crontab-less image.
    """

    def _raise_absent() -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory: 'crontab'")

    monkeypatch.setitem(globals(), "_read_crontab", _raise_absent)
    with pytest.raises(pytest.skip.Exception, match="not runnable"):
        test_crontab_referenced_repo_scripts_are_executable_in_index()


def test_soft_arm_skips_cleanly_when_crontab_invocation_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged ``crontab -l`` (TimeoutExpired) yields the clean skip, not a stall."""

    def _raise_timeout() -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["crontab", "-l"], timeout=_CRONTAB_TIMEOUT_S)

    monkeypatch.setitem(globals(), "_read_crontab", _raise_timeout)
    with pytest.raises(pytest.skip.Exception, match="timed out"):
        test_crontab_referenced_repo_scripts_are_executable_in_index()
