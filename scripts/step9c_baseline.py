#!/usr/bin/env python
"""Known-red-on-main baseline ledger for the ``/issue`` Step 9c test-verdict gate (#1022).

Main carries pre-existing red (2000+ repo-wide ruff errors; occasionally a
failing workflow-invariant test node), so a raw ``pytest``/``ruff`` exit code
cannot decide the Step 9c verdict — on 2026-07-02 seven sessions each burned a
verification round re-proving that red main was pre-existing. This helper
maintains a refreshable ledger of what is ALREADY red on main and mechanically
classifies a gate run's failures as NEW (block) vs pre-existing (strip), with a
bounded on-demand pristine-main re-check for anything the ledger cannot safely
vouch for.

Subcommands::

    uv run python scripts/step9c_baseline.py refresh [--repo-root PATH] [--timeout-s 1800] [--json]
    uv run python scripts/step9c_baseline.py status  [--repo-root PATH] [--max-age-hours 24]
                                                     [--max-code-commits 150] [--json]
    uv run python scripts/step9c_baseline.py compare --junitxml PATH --pytest-rc INT [--base main]
                                                     [--worktree PATH] [--repo-root PATH]
                                                     [--run-pristine] [--pristine-timeout-s 600]
                                                     [--max-pristine-files 5]
                                                     [--max-age-hours 24] [--max-code-commits 150]
                                                     [--json]

Exit codes (pinned by ``tests/test_step9c_baseline.py``):

===========  ==========================================================================
``refresh``
  0          ledger written, or lock-busy single-flight no-op (stderr note)
  2          pytest rc not in {0, 1} / timeout / junit parse failure / zero collected /
             git or ruff failure -> **no ledger write**
``status``
  0          fresh
  2          ledger missing / schema-invalid
  3          stale (reasons on stdout)
``compare``
  0          no NEW failures AND no lint regression (``--pytest-rc`` in {0, 1})
  1          NEW failure(s) and/or lint regression (JSON names each)
  2          indeterminate: ``--pytest-rc`` not in {0, 1}; missing/empty junitxml; zero
             testcases; unusable ledger with unresolved buckets (no ``--run-pristine``);
             pristine run timeout/crash; dirty pristine oracle on a failing node;
             more than ``--max-pristine-files`` distinct pristine files ("systemic main
             breakage"); missing ruff binary
===========  ==========================================================================

Safety invariants (plan #1022 v3 R1-R7): the refresh NEVER runs ``pytest tests/``
wholesale (only the predictable 34-file Step 9c universe, timeout-bounded,
thread-capped, process-group-killed on expiry); blind-strip requires a fresh,
clean-rooted (``dirty_code_paths: false``) ledger AND a non-diff-linked node
whose test file is unchanged on main since the ledger SHA — everything else is
resolved by a bounded single-file pristine-main run at CURRENT HEAD from a
clean-code-path root; every strip of a scan-covered test carries a masking WARN
naming the branch's touched files that scan covers; indeterminate is always a
FAIL (exit 2), never a silent PASS; NO subcommand mutates git state (reads only:
``rev-parse`` / ``rev-list`` / ``cat-file`` / ``diff --name-only`` /
``status --porcelain``); ledger writes are flock single-flight + atomic
tmp+``os.replace``.

State files (all under the MAIN repo root, resolved via ``--git-common-dir`` so
every worktree shares ONE ledger):

* ``.claude/cache/step9c-baseline.json``      — the ledger (gitignored)
* ``.claude/cache/step9c-baseline.lock``      — refresh single-flight flock
* ``.claude/cache/step9c-baseline-junit.xml`` — last refresh's raw junit (debug)
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, NamedTuple

GENERATOR = "step9c_baseline.py v1"
SCHEMA_VERSION = 1

# Staleness pathspec: only CODE commits count (tasks/** marker churn, ~100+/hr,
# is excluded entirely) — plan #1022 §3.4 / §10 item 3.
CODE_COMMIT_PATHSPEC: tuple[str, ...] = ("scripts/", "src/", "tests/", "pyproject.toml", "uv.lock")

# Refresh-time dirt probe: CODE paths only (recursive '*.py' pathspec) — the
# perpetual non-code churn (tasks/**, pods_ephemeral.json, agent-memory .md)
# must not read as dirt or the bit is a protection illusion (MF-4a).
DIRTY_CODE_PATHSPEC: tuple[str, ...] = ("*.py", "pyproject.toml", "uv.lock")

REQUIRED_LEDGER_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "main_sha",
        "refreshed_at",
        "dirty_code_paths",
        "dirty_paths",
        "test_universe",
        "failing_tests",
        "pytest_summary",
        "ruff_count",
        "ruff_format_files",
        "refresh_timeout_s",
        "generator",
    }
)

PYTEST_BASE_FLAGS: tuple[str, ...] = (
    "-q",
    "--tb=no",
    "-p",
    "no:cacheprovider",
    "-o",
    "junit_family=xunit1",
)


class Node(NamedTuple):
    """A failing test node key: (rootdir-relative file, junit classname, test name)."""

    file: str
    classname: str
    name: str


class ToolMissingError(RuntimeError):
    """A required external binary (ruff) is not on PATH — fail loud, exit 2."""


class JunitParseError(RuntimeError):
    """The junitxml file is missing, unparseable, empty, or lacks per-case file attrs."""


class PristineRunError(RuntimeError):
    """A pristine-main single-file pytest run aborted (rc not in {0,1} / timeout / 0 collected)."""


def _log(msg: str) -> None:
    """Print one diagnostic line to stderr (all diagnostics; stdout is for results)."""
    print(f"step9c_baseline: {msg}", file=sys.stderr)


# --- Root / selector resolution ------------------------------------------------


def main_repo_root() -> Path:
    """Resolve the MAIN repo root from ANY checkout (worktrees included).

    ``git rev-parse --path-format=absolute --git-common-dir`` resolves to
    ``<root>/.git`` from the main checkout AND from any worktree, so its parent
    is the shared main root — the same recipe SKILL.md Step 9c's ``REPO_ROOT``
    uses. The ledger lives there so every worktree shares ONE copy.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).resolve().parent


def resolve_work_root(arg: str | None) -> Path:
    """Resolve compare's work root: the INVOKING checkout's toplevel, or the override.

    Same semantics as ``select_step9c_tests._resolve_work_root`` — compare is
    run FROM the issue worktree at Step 9c, so the no-arg path resolves the
    worktree root (where the branch diff + its branch-new tests live).
    """
    if arg:
        return Path(arg).resolve()
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).resolve()


def load_selector_module(root: Path):
    """Import ``<root>/scripts/select_step9c_tests.py`` by path (no package plumbing).

    ``refresh`` loads the MAIN root's copy (its curation literals define the
    universe main is red on); ``compare`` loads the INVOKING WORKTREE's copy
    (the same mapping that selected the run's tests) — the version skew is
    deliberate and stated in plan #1022 §3.3/§3.4.
    """
    path = root / "scripts" / "select_step9c_tests.py"
    if not path.exists():
        raise FileNotFoundError(f"selector not found at {path}")
    spec = importlib.util.spec_from_file_location(f"select_step9c_tests_{id(root)}", path)
    assert spec and spec.loader, f"importlib could not build a spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Subprocess helpers (all read-only git; pytest/ruff bounded) ----------------


def thread_capped(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* with the shared-VM BLAS/OMP thread caps setdefault'd to 8."""
    capped = dict(env)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        capped.setdefault(var, "8")
    return capped


def run_pytest(
    files: Iterable[str],
    cwd: Path,
    timeout_s: float,
    junit_path: Path,
    extra: Iterable[str] = PYTEST_BASE_FLAGS,
) -> int:
    """Run one bounded, thread-capped pytest subprocess; return its exit code.

    Uses ``sys.executable -m pytest`` (the invoking venv provides pytest and,
    unlike a nested ``uv run``, works from any cwd without lock contention).
    ``start_new_session=True`` + ``os.killpg`` on ``TimeoutExpired`` group-kills
    stragglers, then the ``TimeoutExpired`` is re-raised (callers exit 2 —
    NEVER a ledger write / classification from a timed-out run).
    """
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *files,
        *extra,
        f"--junitxml={junit_path}",
    ]
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=thread_capped(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _log(f"pytest exceeded {timeout_s}s — killing the process group")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.2)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise


def _git_out(argv: list[str], cwd: Path) -> str:
    """Run a READ-ONLY git command; return stdout. Raises CalledProcessError loud."""
    proc = subprocess.run(["git", *argv], cwd=str(cwd), capture_output=True, text=True, check=True)
    return proc.stdout


def git_head(root: Path) -> str:
    """Return the 40-hex HEAD sha of *root*."""
    return _git_out(["rev-parse", "HEAD"], root).strip()


def git_sha_known(root: Path, sha: str) -> bool:
    """True iff *sha* resolves to a commit in *root*'s object store (read-only probe)."""
    if not sha:
        return False
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def code_commits_since(root: Path, sha: str) -> int:
    """Count CODE-path commits (CODE_COMMIT_PATHSPEC) between *sha* and HEAD."""
    out = _git_out(["rev-list", "--count", f"{sha}..HEAD", "--", *CODE_COMMIT_PATHSPEC], root)
    return int(out.strip())


def changed_test_files_since(root: Path, sha: str) -> set[str]:
    """Return ``tests/`` paths changed on main between *sha* and HEAD (per-entry staleness)."""
    out = _git_out(["diff", "--name-only", f"{sha}..HEAD", "--", "tests/"], root)
    return {line.strip() for line in out.splitlines() if line.strip()}


def dirty_code_paths(root: Path) -> list[str]:
    """Return uncommitted CODE-path changes at *root* (DIRTY_CODE_PATHSPEC scope).

    Scoped so the perpetual non-code churn on the shared root (tasks/**,
    pods_ephemeral.json, agent-memory .md) never reads as dirt (MF-4a).
    """
    out = _git_out(["status", "--porcelain", "--", *DIRTY_CODE_PATHSPEC], root)
    paths: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        p = line[3:].strip()
        if " -> " in p:  # rename entry: "old -> new"
            p = p.split(" -> ", 1)[1]
        paths.append(p.strip('"'))
    return paths


def _ruff_bin() -> str:
    """Resolve the ruff binary; missing -> ToolMissingError (fail loud, exit 2 — MF-5)."""
    ruff = shutil.which("ruff")
    if not ruff:
        raise ToolMissingError(
            "ruff not found on PATH — run under `uv run` or install ruff (fail-loud, MF-5)"
        )
    return ruff


def ruff_error_count(target: Path, paths: list[str] | None = None) -> int:
    """Count ruff-check diagnostics at *target* (whole tree, or just *paths*)."""
    argv = [
        _ruff_bin(),
        "check",
        *(paths if paths else ["."]),
        "--exit-zero",
        "--output-format=json",
    ]
    proc = subprocess.run(argv, cwd=str(target), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ruff check failed at {target} (rc={proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    return len(json.loads(proc.stdout))


def ruff_format_count(target: Path, paths: list[str] | None = None) -> int:
    """Count would-reformat files at *target* per ``ruff format --check`` (rc>1 fails loud)."""
    argv = [_ruff_bin(), "format", "--check", *(paths if paths else ["."])]
    proc = subprocess.run(argv, cwd=str(target), capture_output=True, text=True)
    if proc.returncode > 1:
        raise RuntimeError(
            f"ruff format --check failed at {target} (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    return sum(1 for line in proc.stdout.splitlines() if line.startswith("Would reformat"))


# --- junit parsing ---------------------------------------------------------------


def parse_junit(path: Path) -> tuple[list[Node], dict]:
    """Parse an xunit1 junitxml file -> (failing nodes, summary).

    Failing = a testcase with a ``failure`` or ``error`` child (``skipped``
    excluded). Fails loud (JunitParseError) on: missing file (a killed run
    leaves NO junit thanks to the gate's pre-run ``rm -f`` — MF-1a), XML parse
    failure, or a failing testcase without the per-case ``file`` attribute
    (pytest 9.0.2 xunit1 emits it — plan #1022 A3; the K2 short-summary
    fallback is a deliberate redesign, not a silent guess).
    """
    if not path.exists():
        raise JunitParseError(
            f"junitxml missing at {path} — the run was killed before pytest's exit-time "
            "write, or the pre-run `rm -f` state was never overwritten (MF-1a)"
        )
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise JunitParseError(f"junitxml unparseable at {path}: {exc}") from exc
    testcases = list(tree.getroot().iter("testcase"))
    failing: list[Node] = []
    n_fail = n_err = n_skip = 0
    for tc in testcases:
        has_failure = tc.find("failure") is not None
        has_error = tc.find("error") is not None
        if tc.find("skipped") is not None:
            n_skip += 1
        n_fail += int(has_failure)
        n_err += int(has_error)
        if has_failure or has_error:
            file_attr = tc.get("file")
            if not file_attr:
                raise JunitParseError(
                    f"failing testcase {tc.get('classname')}::{tc.get('name')} has no "
                    "file attribute — xunit1 contract violated (see plan #1022 K2 fallback)"
                )
            failing.append(
                Node(file=file_attr, classname=tc.get("classname") or "", name=tc.get("name") or "")
            )
    duration = 0.0
    for suite in tree.getroot().iter("testsuite"):
        with contextlib.suppress(ValueError):  # a malformed time attr only loses the duration
            duration += float(suite.get("time") or 0.0)
    summary = {
        "tests": len(testcases),
        "failures": n_fail,
        "errors": n_err,
        "skipped": n_skip,
        "duration_s": duration,
    }
    return failing, summary


# --- Ledger IO -------------------------------------------------------------------


def ledger_path(root: Path) -> Path:
    """The shared ledger path under the MAIN root's gitignored cache dir."""
    return root / ".claude" / "cache" / "step9c-baseline.json"


def write_ledger_atomic(path: Path, ledger: dict) -> None:
    """Atomically write *ledger* (tmp + os.replace) — readers never see a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(ledger, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def try_load_ledger(root: Path) -> dict | None:
    """Load + schema-validate the ledger; None (with a loud stderr line) when unusable."""
    path = ledger_path(root)
    if not path.exists():
        _log(f"ledger missing at {path}")
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _log(f"ledger unreadable/invalid at {path}: {exc}")
        return None
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != SCHEMA_VERSION
        or not set(data) >= REQUIRED_LEDGER_KEYS
    ):
        _log(f"ledger at {path} fails schema v{SCHEMA_VERSION} validation")
        return None
    return data


def ledger_nodes(ledger: dict) -> set[Node]:
    """The ledger's failing_tests as a Node set (node-level keys, §3.1)."""
    return {
        Node(file=e["file"], classname=e.get("classname", ""), name=e["name"])
        for e in ledger["failing_tests"]
    }


def ledger_age_hours(ledger: dict) -> float | None:
    """Hours since the ledger's refreshed_at; None when unparseable."""
    try:
        refreshed = datetime.fromisoformat(str(ledger["refreshed_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None
    return (datetime.now(UTC) - refreshed).total_seconds() / 3600.0


def staleness(
    ledger: dict | None, root: Path, max_age_hours: float, max_code_commits: int
) -> tuple[bool, list[str]]:
    """Global staleness predicate (plan §3.4): age OR code-commit count OR sha unknown.

    ``dirty_code_paths: true`` at refresh is a SEPARATE (fourth) strip-disabling
    condition folded into ``strippable_ledger`` by the caller, reported
    distinctly — deliberately NOT part of this predicate (MF-4b).
    """
    if ledger is None:
        return True, ["ledger missing or schema-invalid"]
    reasons: list[str] = []
    age_h = ledger_age_hours(ledger)
    if age_h is None:
        return True, ["refreshed_at missing or unparseable"]
    if age_h > max_age_hours:
        reasons.append(f"age {age_h:.1f}h > max {max_age_hours}h")
    sha = str(ledger.get("main_sha", ""))
    if not git_sha_known(root, sha):
        reasons.append(f"ledger main_sha {sha[:12]!r} unknown to git")
        return True, reasons
    n_commits = code_commits_since(root, sha)
    if n_commits > max_code_commits:
        reasons.append(f"{n_commits} code-path commits since ledger sha > max {max_code_commits}")
    return bool(reasons), reasons


# --- refresh ---------------------------------------------------------------------


def acquire_refresh_lock(lock_file: Path) -> IO[bytes] | None:
    """Take the refresh single-flight flock (LOCK_EX|LOCK_NB); None when held elsewhere.

    The returned file object must stay referenced for the lock's lifetime
    (process exit releases it).
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_file, "wb")  # noqa: SIM115 — the flock must outlive this function
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def present_on_disk(files: Iterable[str], root: Path) -> list[str]:
    """Filter *files* to those existing under *root* (same contract as the selector)."""
    return [f for f in files if (root / f).exists()]


def cmd_refresh(args: argparse.Namespace) -> int:
    """Run the 34-file predictable Step 9c universe on main; write the ledger atomically."""
    root = Path(args.repo_root).resolve() if args.repo_root else main_repo_root()
    lock = acquire_refresh_lock(root / ".claude" / "cache" / "step9c-baseline.lock")
    if lock is None:
        _log("another refresh holds the lock — single-flight no-op")
        return 0
    sel = load_selector_module(root)  # MAIN root's curation literals (§3.3 version-skew note)
    universe = sorted(present_on_disk({*sel.WORKFLOW_INVARIANT, *sel.GLOB_SCAN_TESTS}, root))
    if not universe:
        _log("EMPTY refresh universe — work root resolved wrong or invariants vanished")
        return 2
    junit = root / ".claude" / "cache" / "step9c-baseline-junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    junit.unlink(missing_ok=True)  # same stale-junit lifecycle as the gate (MF-1a)
    t0 = time.monotonic()
    try:
        rc = run_pytest(files=universe, cwd=root, timeout_s=args.timeout_s, junit_path=junit)
    except subprocess.TimeoutExpired:
        _log(f"refresh pytest timed out after {args.timeout_s}s — NO ledger write")
        return 2
    if rc not in (0, 1):
        _log(f"refresh pytest rc={rc} (interrupted/internal error) — NO ledger write")
        return 2
    try:
        failing, summary = parse_junit(junit)
    except JunitParseError as exc:
        _log(f"refresh junit parse failed: {exc} — NO ledger write")
        return 2
    if summary["tests"] == 0:
        _log("refresh collected 0 tests — refusing to write ledger")
        return 2
    summary["duration_s"] = round(time.monotonic() - t0, 1)
    try:
        dirty = dirty_code_paths(root)
        head = git_head(root)
        ruff_count = ruff_error_count(root)
        ruff_fmt = ruff_format_count(root)
    except (subprocess.CalledProcessError, ToolMissingError, RuntimeError) as exc:
        _log(f"refresh git/ruff probe failed: {exc} — NO ledger write")
        return 2
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "main_sha": head,
        "refreshed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dirty_code_paths": bool(dirty),
        "dirty_paths": dirty[:20],
        "test_universe": universe,
        "failing_tests": [n._asdict() for n in sorted(failing)],
        "pytest_summary": summary,
        "ruff_count": ruff_count,
        "ruff_format_files": ruff_fmt,
        "refresh_timeout_s": args.timeout_s,
        "generator": GENERATOR,
    }
    write_ledger_atomic(ledger_path(root), ledger)
    _log(
        f"ledger written: {len(failing)} failing node(s) / {summary['tests']} tests, "
        f"ruff={ruff_count}, format={ruff_fmt}, sha={head[:12]}, dirty_code_paths={bool(dirty)}"
    )
    if args.json:
        print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


# --- status ----------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Report ledger freshness: 0 fresh / 2 missing-or-invalid / 3 stale."""
    root = Path(args.repo_root).resolve() if args.repo_root else main_repo_root()
    ledger = try_load_ledger(root)
    if ledger is None:
        return 2
    stale, reasons = staleness(ledger, root, args.max_age_hours, args.max_code_commits)
    payload = {
        "fresh": not stale,
        "stale": stale,
        "stale_reasons": reasons,
        "main_sha": ledger["main_sha"],
        "refreshed_at": ledger["refreshed_at"],
        "ledger_age_h": ledger_age_hours(ledger),
        "dirty_code_paths": ledger["dirty_code_paths"],
        "failing_tests": ledger["failing_tests"],
        "ruff_count": ledger["ruff_count"],
        "ruff_format_files": ledger["ruff_format_files"],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ledger sha {ledger['main_sha'][:12]} refreshed {ledger['refreshed_at']}")
        print(f"failing_tests: {len(ledger['failing_tests'])}  ruff_count: {ledger['ruff_count']}")
        print("fresh" if not stale else "stale: " + "; ".join(reasons))
    return 3 if stale else 0


# --- compare ---------------------------------------------------------------------


def run_single_file_pristine(test_file: str, cwd: Path, timeout_s: float) -> set[Node]:
    """Run ONE test file at the main root (pristine oracle); return its failing nodes.

    Bounded + thread-capped like refresh. rc not in {0, 1}, a timeout, or a
    zero-collected run raises PristineRunError (indeterminate, exit 2 — MF-5);
    a classification must never rest on an aborted pristine run.
    """
    fd, tmp = tempfile.mkstemp(prefix="step9c-pristine-junit-", suffix=".xml")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.unlink(missing_ok=True)  # pytest must create it fresh (MF-1a parity)
        try:
            rc = run_pytest(files=[test_file], cwd=cwd, timeout_s=timeout_s, junit_path=tmp_path)
        except subprocess.TimeoutExpired as exc:
            raise PristineRunError(f"pristine run of {test_file} timed out ({timeout_s}s)") from exc
        if rc not in (0, 1):
            raise PristineRunError(f"pristine run of {test_file} aborted with rc={rc}")
        try:
            failing, summary = parse_junit(tmp_path)
        except JunitParseError as exc:
            raise PristineRunError(f"pristine junit unusable for {test_file}: {exc}") from exc
        if summary["tests"] == 0:
            raise PristineRunError(f"pristine run of {test_file} collected 0 tests")
        return set(failing)
    finally:
        tmp_path.unlink(missing_ok=True)


def lint_verdict(root: Path, wt: Path, touched: list[str]) -> dict:
    """Mechanical lint verdict: delta vs the LIVE main-root baseline + absolute-clean touched.

    Only an INCREASE over the live baseline fails; the branch's touched ``.py``
    files must additionally be ruff-clean AND format-clean in absolute terms.
    The known main-side-cleanup residual false-FAIL is fail-closed by design
    (plan §3.4) — the deltas below make it one-glance diagnosable.
    """
    base_count = ruff_error_count(root)
    wt_count = ruff_error_count(wt)
    base_fmt = ruff_format_count(root)
    wt_fmt = ruff_format_count(wt)
    touched_py = [f for f in touched if f.endswith(".py") and (wt / f).exists()]
    touched_errors = ruff_error_count(wt, paths=touched_py) if touched_py else 0
    touched_fmt = ruff_format_count(wt, paths=touched_py) if touched_py else 0
    touched_ok = touched_errors == 0 and touched_fmt == 0
    return {
        "ok": wt_count <= base_count and wt_fmt <= base_fmt and touched_ok,
        "base_ruff_count": base_count,
        "wt_ruff_count": wt_count,
        "base_format_files": base_fmt,
        "wt_format_files": wt_fmt,
        "touched_py": touched_py,
        "touched_ruff_errors": touched_errors,
        "touched_format_files": touched_fmt,
    }


def _pristine_command(root: Path, test_file: str) -> str:
    """The copy-pasteable single-file pristine check printed on the no-run path."""
    return (
        f"(cd {root} && uv run pytest {test_file} -q --tb=no -p no:cacheprovider "
        "-o junit_family=xunit1)"
    )


class _Indeterminate(RuntimeError):
    """Internal control flow: compare cannot classify — the caller exits 2."""


@dataclass
class _LedgerView:
    """The ledger + the derived stripping predicates for one compare run."""

    ledger: dict | None
    stale: bool
    stale_reasons: list[str]
    ledger_dirty: bool
    strippable: bool
    known_red: set[Node]
    changed_tests: set[str]


@dataclass
class _CompareCtx:
    """Selector context + mutable classification accumulators for one compare run."""

    sel: object
    touched: list[str]
    diff_linked: set[str]
    new: list[Node] = field(default_factory=list)
    stripped: list[dict] = field(default_factory=list)
    pristine_bucket: list[Node] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    live_dirty_paths: list[str] = field(default_factory=list)
    pristine_files_run: list[str] = field(default_factory=list)


def _resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve (work root, main root) for compare; git failure -> indeterminate."""
    try:
        wt = resolve_work_root(args.worktree)
        root = Path(args.repo_root).resolve() if args.repo_root else main_repo_root()
    except subprocess.CalledProcessError as exc:
        raise _Indeterminate(f"cannot resolve work root / main root: {exc}") from exc
    return wt, root


def _load_run_junit(junitxml: Path) -> list[Node]:
    """Parse the gate run's junit; missing/unparseable/zero-case -> indeterminate."""
    try:
        run_failing, summary = parse_junit(junitxml)
    except JunitParseError as exc:
        raise _Indeterminate(str(exc)) from exc
    if summary["tests"] == 0:
        raise _Indeterminate(
            "junit has ZERO testcases — echoes the no-tests-ran FAIL guard; refusing"
        )
    return run_failing


def _selector_context(args: argparse.Namespace, wt: Path) -> _CompareCtx:
    """Load the WORKTREE's selector and derive touched + diff-linked-ness (§3.3 skew note)."""
    try:
        sel = load_selector_module(wt)
        touched = sel.compute_touched(args.base, wt)
        _tests, _untested, reasons = sel.select_tests_with_reasons(touched, wt)
    except (FileNotFoundError, subprocess.CalledProcessError, AttributeError) as exc:
        raise _Indeterminate(f"selector load / touched-diff failed at {wt}: {exc}") from exc
    diff_linked = {t for t, rs in reasons.items() if any(r != "invariant" for r in rs)} | {
        f for f in touched if f.startswith("tests/")
    }
    return _CompareCtx(sel=sel, touched=touched, diff_linked=diff_linked)


def _ledger_view(root: Path, args: argparse.Namespace) -> _LedgerView:
    """Load the ledger and derive the four strip-disabling conditions (§3.4)."""
    ledger = try_load_ledger(root)
    stale, stale_reasons = staleness(ledger, root, args.max_age_hours, args.max_code_commits)
    ledger_dirty = bool(ledger and ledger["dirty_code_paths"])  # MF-4b
    strippable = ledger is not None and not stale and not ledger_dirty
    known_red: set[Node] = ledger_nodes(ledger) if strippable else set()
    changed_tests: set[str] = set()
    if strippable:
        try:
            changed_tests = changed_test_files_since(root, str(ledger["main_sha"]))
        except subprocess.CalledProcessError:
            # sha unusable for the per-entry diff -> treat the ledger as fully
            # stale for stripping (fail closed; MF-3).
            strippable = False
            stale, stale_reasons = True, [*stale_reasons, "changed-tests diff failed"]
    return _LedgerView(
        ledger=ledger,
        stale=stale,
        stale_reasons=stale_reasons,
        ledger_dirty=ledger_dirty,
        strippable=strippable,
        known_red=known_red,
        changed_tests=changed_tests,
    )


def _strip_node(ctx: _CompareCtx, node: Node, via: str) -> None:
    """Strip *node* as pre-existing; EVERY scan-covered strip WARNs (MF-6)."""
    ctx.stripped.append({**node._asdict(), "via": via})
    sel = ctx.sel
    if node.file in sel.GLOB_SCAN_TESTS:
        covered = [f for f in ctx.touched if sel._matches_any(f, sel.GLOB_SCAN_TESTS[node.file])]
        ctx.warns.append(
            f"MASKING WARN: stripped scan test {node.file}::{node.name} (via {via}) — "
            f"its directory scan covers touched file(s) {covered or '[]'}; re-check them "
            "against that scan's rule"
        )
    if via == "pristine" and node.file in ctx.diff_linked:
        ctx.warns.append(
            f"MASKING WARN: stripped diff-linked node {node.file}::{node.name} via a "
            "pristine-main failure — the branch touches files mapped to this test; "
            "confirm the branch does not deepen the pre-existing breakage"
        )


def _bucket_run_failures(
    ctx: _CompareCtx, run_failing: list[Node], lv: _LedgerView, root: Path
) -> None:
    """Blind-strip only safe in-ledger nodes; everything else -> NEW or pristine bucket."""
    for node in run_failing:
        if not lv.strippable:
            ctx.pristine_bucket.append(node)
        elif node in lv.known_red:
            if node.file in ctx.diff_linked or node.file in lv.changed_tests:  # MF-3 conjunct
                ctx.pristine_bucket.append(node)  # R5 — never blind-strip
            else:
                _strip_node(ctx, node, via="ledger")
        elif not (root / node.file).exists():
            ctx.new.append(node)  # branch-new test failing — main cannot vouch
        else:
            ctx.pristine_bucket.append(node)  # unknown provenance


def _resolve_pristine_bucket(ctx: _CompareCtx, root: Path, args: argparse.Namespace) -> None:
    """Resolve bucketed nodes via bounded single-file pristine-main runs (or refuse)."""
    if not ctx.pristine_bucket:
        return
    files = sorted({n.file for n in ctx.pristine_bucket})
    if len(files) > args.max_pristine_files:
        raise _Indeterminate(
            f"systemic main breakage ({len(files)} red files > "
            f"--max-pristine-files {args.max_pristine_files}) — investigate / refresh first"
        )
    if not args.run_pristine:
        commands = "\n".join(f"  {_pristine_command(root, f)}" for f in files)
        raise _Indeterminate(
            f"{len(ctx.pristine_bucket)} failure(s) need a pristine-main check and "
            "--run-pristine was not given — indeterminate, never a silent strip. "
            f"Per-file pristine commands:\n{commands}"
        )
    for test_file in files:
        try:
            ctx.live_dirty_paths = dirty_code_paths(root)  # probed AT pristine time (MF-4c)
        except subprocess.CalledProcessError as exc:
            raise _Indeterminate(f"dirt probe failed at {root}: {exc}") from exc
        try:
            main_failing = run_single_file_pristine(
                test_file, cwd=root, timeout_s=args.pristine_timeout_s
            )
        except PristineRunError as exc:
            raise _Indeterminate(f"{exc} — indeterminate") from exc
        ctx.pristine_files_run.append(test_file)
        for node in [n for n in ctx.pristine_bucket if n.file == test_file]:
            if node in main_failing:
                if ctx.live_dirty_paths:
                    raise _Indeterminate(
                        f"pristine oracle is DIRTY on code paths {ctx.live_dirty_paths[:20]} — "
                        f"a 'pre-existing' verdict for {node.file}::{node.name} from a "
                        "dirty root is untrustworthy (MF-4c); indeterminate"
                    )
                _strip_node(ctx, node, via="pristine")  # pre-existing at CURRENT clean main HEAD
            else:
                ctx.new.append(node)  # a PASS on a dirty root still classifies NEW (fail-closed)


def _compare_impl(args: argparse.Namespace) -> dict:
    """The compare pipeline (plan §3.4); raises _Indeterminate on any exit-2 condition."""
    wt, root = _resolve_roots(args)
    run_failing = _load_run_junit(Path(args.junitxml))
    ctx = _selector_context(args, wt)
    lv = _ledger_view(root, args)
    _bucket_run_failures(ctx, run_failing, lv, root)
    _resolve_pristine_bucket(ctx, root, args)
    try:
        lint = lint_verdict(root, wt, ctx.touched)
    except (ToolMissingError, RuntimeError) as exc:
        raise _Indeterminate(f"lint verdict failed: {exc}") from exc
    ledger = lv.ledger
    return {
        "pytest_rc": args.pytest_rc,
        "new": [n._asdict() for n in ctx.new],
        "stripped": ctx.stripped,
        "warns": ctx.warns,
        "stale": lv.stale,
        "stale_reasons": lv.stale_reasons,
        "ledger_dirty": lv.ledger_dirty,
        "ledger_dirty_paths": list(ledger.get("dirty_paths", [])) if ledger else [],
        "live_dirty_paths": ctx.live_dirty_paths,
        "pristine_files_run": ctx.pristine_files_run,
        "lint": lint,
        "ledger_sha": ledger["main_sha"] if ledger else None,
        "ledger_age_h": ledger_age_hours(ledger) if ledger else None,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    """Classify a Step 9c run's failures as NEW vs pre-existing-on-main (plan §3.4)."""
    if args.pytest_rc not in (0, 1):
        _log(
            f"pytest rc {args.pytest_rc} (aborted/interrupted/internal-error run) — "
            "refusing to classify a partial run (MF-1b)"
        )
        return 2
    try:
        result = _compare_impl(args)
    except _Indeterminate as exc:
        _log(str(exc))
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"compare: {len(result['new'])} NEW, {len(result['stripped'])} stripped, "
            f"{len(result['warns'])} warn(s), lint_ok={result['lint']['ok']}"
        )
        for n in result["new"]:
            print(f"  NEW: {n['file']}::{n['name']}")
        for w in result["warns"]:
            print(f"  {w}")
    for w in result["warns"]:
        _log(w)
    return 1 if (result["new"] or not result["lint"]["ok"]) else 0


# --- CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the refresh/status/compare CLI (shapes pinned in plan #1022 §3.2)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh", help="run the Step 9c universe on main; write ledger")
    p_refresh.add_argument("--repo-root", default=None, help="main-root override (tests)")
    p_refresh.add_argument("--timeout-s", type=float, default=1800.0)
    p_refresh.add_argument("--json", action="store_true")
    p_refresh.set_defaults(func=cmd_refresh)

    p_status = sub.add_parser("status", help="ledger freshness: 0 fresh / 2 invalid / 3 stale")
    p_status.add_argument("--repo-root", default=None)
    p_status.add_argument("--max-age-hours", type=float, default=24.0)
    p_status.add_argument("--max-code-commits", type=int, default=150)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_compare = sub.add_parser("compare", help="classify gate failures as NEW vs pre-existing")
    p_compare.add_argument("--junitxml", required=True, help="the gate run's junitxml path")
    p_compare.add_argument(
        "--pytest-rc",
        type=int,
        required=True,
        help="exit code of the gate's pytest invocation; rc not in {0,1} -> exit 2 (MF-1b)",
    )
    p_compare.add_argument("--base", default="main")
    p_compare.add_argument(
        "--worktree", default=None, help="work-root override (default: cwd toplevel)"
    )
    p_compare.add_argument("--repo-root", default=None)
    p_compare.add_argument("--run-pristine", action="store_true")
    p_compare.add_argument("--pristine-timeout-s", type=float, default=600.0)
    p_compare.add_argument("--max-pristine-files", type=int, default=5)
    p_compare.add_argument("--max-age-hours", type=float, default=24.0)
    p_compare.add_argument("--max-code-commits", type=int, default=150)
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns the subcommand's exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
