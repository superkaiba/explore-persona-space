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

    uv run python scripts/step9c_baseline.py refresh [--repo-root PATH] [--timeout-s 4350] [--json]
    uv run python scripts/step9c_baseline.py status  [--repo-root PATH] [--max-age-hours 24]
                                                     [--max-code-commits 150] [--json]
    uv run python scripts/step9c_baseline.py compare --junitxml PATH --pytest-rc INT [--base REF]
                                                     (default: resolved via the worktree
                                                     selector — fetched-origin/main
                                                     semantics, no second fetch, #1289)
                                                     [--worktree PATH] [--repo-root PATH]
                                                     [--run-pristine] [--pristine-timeout-s S]
                                                     [--max-pristine-files 5]
                                                     [--max-age-hours 24] [--max-code-commits 150]
                                                     [--scratch-timeout-s 120]
                                                     [--no-scratch-fallback]
                                                     [--no-src-shadow] [--json]
    uv run python scripts/step9c_baseline.py tmproot
    uv run python scripts/step9c_baseline.py probe   (--pattern REGEX | --issue N |
                                                      --fleet [--exclude-issue N])

Exit codes (pinned by ``tests/test_step9c_baseline.py``):

===========  ==========================================================================
``refresh``
  0          ledger written, or lock-busy single-flight no-op (stderr note)
  2          pytest rc not in {0, 1} / timeout / junit parse failure / zero collected /
             git or ruff failure / missing root-venv interpreter -> **no ledger write**
``status``
  0          fresh
  2          ledger missing / schema-invalid
  3          stale (reasons on stdout)
``compare``
  0          no NEW failures AND no lint regression (``--pytest-rc`` in {0, 1})
  1          NEW failure(s) and/or lint regression (JSON names each)
  2          indeterminate: ``--pytest-rc`` not in {0, 1}; missing/empty junitxml; zero
             testcases; unusable ledger with unresolved buckets (no ``--run-pristine``);
             pristine run timeout/crash (incl. a missing root-venv interpreter); dirty
             oracle on a failing node where the scratch oracle is ineligible
             (residual contaminating dirt — ``pyproject.toml`` / ``uv.lock`` or an
             out-of-package ``src/`` path; dirty ``src/explore_persona_space/**`` is
             neutralized by the scratch PYTHONPATH shadow unless ``--no-src-shadow``
             (#1251); a scan-set (``GLOB_SCAN_TESTS``) node outside
             ``FILE_ANCHORED_SCAN_TESTS`` (#1337); a node RED at pristine
             HEAD on the FLOOR-profile scratch of a non-sparse work root
             (R-G' strip refusal, #2019 — a node GREEN there resolves
             NEW/rc 1 with ``pristine_oracle: scratch-worktree-floor``);
             or ``--no-scratch-fallback``); scratch-worktree creation or
             src-shadow probe failure on a DIRTY root (a CLEAN-root scratch failure
             degrades to the trustworthy root oracle with a WARN, never exit 2 —
             #1408); more than ``--max-pristine-files`` distinct
             pristine files ("systemic main breakage"); missing ruff binary;
             a misconfigured explicit ``EPM_STEP9C_TMPDIR`` override (#1408)
``tmproot``
  0          always — prints the resolved gate temp-write root, or nothing
``probe``
  0          CLEAR — no live FOREIGN ``/proc/*/cmdline`` match (safe to launch);
             ``--fleet``: DISTINCT foreign gate-issue count < ``EPM_GATE_FLEET_MAX``
  3          >=1 live foreign match (one ``pid<TAB>args`` line per match on stdout);
             ``--fleet``: count >= ``EPM_GATE_FLEET_MAX`` (default 2; one
             ``issue=<M><TAB>pids=<k><TAB><sample argv>`` line per foreign gate issue)
  2          usage error (argparse: exactly one of ``--pattern``/``--issue``/``--fleet``
             required; ``--exclude-issue`` without ``--fleet``) / bad ``--pattern`` regex
===========  ==========================================================================

``probe`` (#1821) is the gate single-flight liveness check with MECHANICAL
self- + ancestor-pid exclusion (``/proc/<pid>/status`` PPid walk to pid 1; a
walk failure shrinks the exclusion set — fail toward a loud false LIVE, never
a silent skip). It replaces the remembered "separate FOREGROUND call +
bracketed pgrep" placement rule at the SKILL.md ``Single-flight probe
(#1606)`` sites: the bracket idiom shields only the PATTERN text and is
defeated whenever the enclosing call's argv carries the real unbracketed
artifact path (#1742; 2026-07-26 session ``2b779905``). Exit semantics are
deliberately INVERTED vs pgrep (0 = clear) so ``probe && launch`` composes;
until-loop compositions use the fixed-regex ``--issue`` form only (an exit-2
bad-regex inside an until-loop would otherwise wait forever). NO other
exclusion classes: a transient concurrent foreign ``--pattern`` probe reads as
a loud, self-resolving false LIVE — the fail-safe direction.

``probe --fleet [--exclude-issue N]`` (#1962) is the FLEET-level arbitration
arm on the same scanner: it matches the fixed internal union
``FLEET_GATE_SIGNATURE_RE`` over the four gate artifact classes
(``step9c-junit-issue-(\\d+)\\.xml`` | ``issue-(\\d+)-lint-gate-tree`` |
``issue-(\\d+)-[^ ]*inline-payload\\.txt`` |
``issue-(\\d+)-surgical-outcome\\.txt``) plus a ``step9c_baseline\\.py refresh``
alternate mapped to the reserved pseudo-issue key ``refresh`` (the ledger
refresh runs the heaviest pytest universe and its own flock bounds it to one
fleet-wide — it counts as ONE gate). Per matched argv, ALL matched capture
groups attribute (``finditer``, never ``group(1)`` alone — a wrapper argv
referencing two issues' artifacts attributes to every matched issue); matches
group by issue key, ``--exclude-issue N`` drops the caller's own issue, and
the DISTINCT-foreign-issue count decides the exit: 3 when
``count >= EPM_GATE_FLEET_MAX`` (env int, default 2), else 0. A malformed env
value (non-int / < 1) falls back to the default with a stderr note — never a
crash, never exit 2 (a wedged env var must not wedge gate launches). The
internal union regex is FIXED and valid, so ``--fleet`` is until-loop-safe
exactly like ``--issue`` (exit 2 stays argparse/usage-only for this form).
The transient-foreign-probe note above extends to the fleet form: foreign
sessions' own probe / ``rm -f`` / pgrep wrapper argvs momentarily carry gate
signatures, so a fleet read can transiently over-count — at worst one extra
60 s queue wait at the SKILL.md hook sites; the fail-safe direction.

Safety invariants (plan #1022 v3 R1-R7): the refresh NEVER runs ``pytest tests/``
wholesale (only the predictable Step 9c workflow-invariant universe — 61 files
as of 2026-07-24 — timeout-bounded,
thread-capped, process-group-killed on expiry); blind-strip requires a fresh,
clean-rooted (``dirty_code_paths: false``) ledger AND a non-diff-linked node
whose test file is unchanged on main since the ledger SHA — everything else is
resolved by a bounded single-file pristine-main run at CURRENT HEAD from a
clean-code-path root; every strip of a scan-covered test carries a masking WARN
naming the branch's touched files that scan covers; indeterminate is always a
FAIL (exit 2), never a silent PASS; the refresh + pristine pytest subprocesses
run the TARGET root's OWN venv interpreter with inherited ``PYTHONPATH``
stripped (never the invoking ``sys.executable``, whose worktree ``.pth`` would
import branch library code into a "pristine" run — #1022 round-2 Critical);
scratch-oracle mode then sets ``PYTHONPATH`` to the scratch's HEAD-pinned
``src/`` so the scratch package shadows the root venv's static editable
``.pth``, verified per compare by a fail-closed runtime probe
(``assert_scratch_src_shadow``, #1251); NO subcommand
mutates git state (reads only:
``rev-parse`` / ``rev-list`` / ``cat-file`` / ``diff --name-only`` /
``status --porcelain`` / ``ls-tree``), EXCEPT compare's bounded scratch-oracle
creation (#1077; the DEFAULT pristine oracle since #1408 — #1077's dirty-only
trigger removed, so root-oracle runs remain only for scratch-INELIGIBLE nodes
and the clean-root degradation path) — ``git worktree add --detach
--no-checkout`` into a fresh tmp dir + worktree-local sparse-checkout +
populate at the root's HEAD,
ALWAYS torn down in a ``finally`` (``worktree remove --force`` + rmtree;
deliberately NO ``git worktree prune`` — see ``remove_scratch_worktree``); the
shared root's branches, index, working tree, and shared config are never
touched; ledger writes are flock single-flight + atomic tmp+``os.replace``.
Bulk temp writes (the scratch tree, pytest basetemp/TMPDIR fixtures, the
pristine junit) route onto the data disk when it is mounted
(``gate_tmp_root``, #1408; #1363: ``/`` at 100% killed the gate); the gate's
own junit/rc/log triplet stays at ``/tmp`` (pinned SKILL contract paths).

Residual risk of the scratch oracle (the DEFAULT since #1408, so this
environmental channel applies to EVERY compare — clean or dirty root alike,
not only the formerly-dirty fallback runs): ``repo_root()``-anchored /
installed-package-path reads resolve the MAIN root even from a scratch cwd —
the scan-set (``GLOB_SCAN_TESTS``) exclusion covers the known class of such
live-tree scanners — EXCEPT ``FILE_ANCHORED_SCAN_TESTS`` members (#1337),
scan tests source-verified to derive their scan root from ``Path(__file__)``
so a scratch copy scans the scratch tree (anchoring drift pin in
tests/test_step9c_baseline.py); a future non-scan test that executes root
files via ``repo_root()`` would re-open that channel. The #1251 src-shadow covers
*import-system* resolution only — ``repo_root()``-anchored ``src/`` file READS
remain that documented scan-set-covered channel. PRE-EXISTING (unchanged by
#1251) trigger gap: src-``.json``-only dirt with no dirty ``*.py`` never trips
``live_dirty_paths`` (the ``*.py``-scoped MF-4a trigger), so the root oracle
runs — identical before/after the shadow.

Under ``--json``, EVERY compare exit path prints exactly one JSON object to
stdout: exit 0/1 the classification result (``indeterminate: false``); exit 2
an indeterminate payload (``indeterminate: true`` + ``reason`` + the
accumulated ``warns``) — the Step 9c caller branches on the stable
``indeterminate`` key, never on empty stdout.

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
import getpass
import importlib.util
import json
import os
import re
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

# The contamination probe pathspec is DELIBERATELY wider than DIRTY_CODE_PATHSPEC:
# it must see ALL files under top-level src/ (query-bank .json data ships inside the
# package and resolves LIVE through the editable .pth via importlib.resources), not
# only *.py — the *.py-scoped live_dirty_paths filter alone converts the MIXED case
# (dirty scripts/*.py + dirty src/*.json) into a false scratch strip (#1077).
# MF-4a's .py-scoped churn rationale applies to the dirt TRIGGER, not to this
# eligibility probe. The probe still watches all three legs; since #1251 the
# in-package src/explore_persona_space/ legs are SHADOWABLE (neutralized via the
# probe-verified scratch PYTHONPATH shadow) while pyproject.toml / uv.lock — and
# any out-of-package src/ path — stay residual (see residual_scratch_contamination).
SCRATCH_CONTAMINATION_PATHSPEC: tuple[str, ...] = ("src/", "pyproject.toml", "uv.lock")

# Sparse-profile floor excludes — mirror of new_worktree.sh EXCLUDES.
SCRATCH_EXCLUDES: tuple[str, ...] = ("eval_results", "external", "ood_eval_results")

# Scan-set nodes whose scan is SOURCE-VERIFIED to anchor on the test file's own
# location (Path(__file__)-derived root; no repo_root() / task_workflow / live-tree
# read in the scan chain) — a scratch copy of such a test scans the SCRATCH tree,
# so the scratch pristine oracle is trustworthy for it and R-F is relaxed to R-F'
# (#1337; incident #1318). Hand-curated pinned literal, same curation rule as
# select_step9c_tests.py's GLOB_SCAN_TESTS / SLOW_TESTS; the live-tree anchoring
# drift pin is tests/test_step9c_baseline.py::test_file_anchored_scan_tests_live_tree_pin.
# FAIL-CLOSED: a scan test absent here keeps the R-F refusal — verify anchoring
# by reading the source BEFORE adding an entry.
FILE_ANCHORED_SCAN_TESTS: frozenset[str] = frozenset(
    {
        # root = Path(__file__).resolve().parents[1] (:871); _scan_targets(root) uses
        # root.glob(...) + `git ls-files` with cwd=root — all scratch-local (#1318).
        "tests/test_shared_vm_thread_caps.py",
        # REPO_ROOT = Path(__file__).resolve().parents[1] (:76); _iter_in_scope_files
        # globs REPO_ROOT only; pure ast/re on file text, stdlib-only imports.
        "tests/test_subprocess_env_explicit.py",
        # _HELPER_PATH = Path(__file__).resolve().parents[1]/"scripts"/"select_step9c_tests.py"
        # (:29) — the selector is importlib-loaded BY PATH from the test file's own tree;
        # every live-tree read resolves via Path(sel.__file__).parents[1] / _HELPER_PATH
        # (:137/:193/:567/:601/:1249/:1390/:1476/:1651/:1816+); real-git fixtures use
        # throwaway tmp_path repos with explicit cwd; no repo_root()/task_workflow/cwd/env
        # channel; conftest checked clean too (no root conftest.py; tests/conftest.py's only
        # live-root autouse fixture is gated on `explore_persona_space.task_workflow` in
        # sys.modules — never true in this member's scratch run, which imports the selector
        # by path only) (#1649 audit; incident #1632: as the only non-anchored scan-set
        # member it wedged compare at MF-4c exit 2 on any dirty shared root).
        "tests/test_select_step9c_tests.py",
    }
)

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
    # #1746: one collection-broken file must not abort the whole run (rc=2,
    # unclassifiable — MF-1b refuses). With this flag pytest runs the surviving
    # collected tests, reports each collect error as a junit <error> testcase
    # keyed to the broken FILE (empirical shape, pytest 9.0.2 xunit1:
    # file="tests/test_broken.py", classname="", name="tests.test_broken"),
    # and exits rc=1 — inside the accepted {0,1} set — so compare's existing
    # NEW-vs-pre-existing node subtraction classifies it. rc=2 is thereafter
    # reserved for genuine interruption / internal error (MF-1b preserved).
    "--continue-on-collection-errors",
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


def resolve_root_python(root: Path) -> str:
    """Resolve *root*'s OWN venv interpreter (``<root>/.venv/bin/python``), fail-loud.

    The refresh and pristine-oracle pytest subprocesses MUST execute the TARGET
    root's library code. ``sys.executable`` is the INVOKING venv — from an issue
    worktree its editable ``.pth`` points at the WORKTREE's ``src/``, so a
    "pristine-main" run would execute branch library code against main's test
    files and vouch a branch-caused ``src/`` regression as pre-existing (#1022
    round-2 Critical). Resolving the root's own venv is the automated-path twin
    of the printed ``cd <root> && uv run pytest`` command (same semantics).
    Missing venv -> ToolMissingError (callers map it to exit 2 / PristineRunError
    — never a silent ``sys.executable`` fallback).
    """
    exe = root / ".venv" / "bin" / "python"
    if not exe.exists():
        raise ToolMissingError(
            f"no venv interpreter at {exe} — refusing to run a pytest that would resolve "
            "the INVOKING interpreter's library code instead of the target root's (fail-loud)"
        )
    return str(exe)


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


# --- Gate temp-write routing (#1408; #1363) ----------------------------------------

# AF_UNIX socket paths cap at ~108 bytes; a long pytest basetemp prefix overflowed it
# in #1363 (the tmux-socket fixture). Both production resolution roots below keep the
# derived basetemp prefix (<root>/bt-XXXXXXXX/p) within this bound.
GATE_TMP_MAX_PREFIX_CHARS = 50
GATE_TMP_SWEEP_MAX_AGE_S = 7 * 24 * 3600.0
GATE_TMP_SWEEP_CAP = 20


def _gate_tmp_writable(p: Path) -> bool:
    """Create+unlink writability probe (permission bits alone lie on squashed mounts)."""
    try:
        fd, tmp = tempfile.mkstemp(prefix=".step9c-probe-", dir=str(p))
    except OSError:
        return False
    os.close(fd)
    with contextlib.suppress(OSError):
        os.unlink(tmp)
    return True


def _reap_leaked_scratch_admin(entry: Path) -> None:
    """Best-effort: surgically drop a leaked scratch tree's git worktree admin entry.

    A SIGKILL'd compare leaves ``<entry>/tree-<pid>`` registered in the owning
    repo's ``.git/worktrees``. ``git worktree remove --force`` on the leaked
    tree — its owner resolved from the tree's own ``.git`` gitfile — clears
    that entry WITHOUT the global ``git worktree prune``, which stays banned
    here (prune reaps admin entries of ANY unreachable worktree; see
    ``remove_scratch_worktree``). Every failure is suppressed: the caller's
    rmtree still runs, and a residual admin entry is cosmetic (git gc's
    ``gc.worktreePruneExpire`` sweeps it).
    """
    for tree in entry.glob("tree-*"):
        with contextlib.suppress(Exception):
            content = (tree / ".git").read_text().strip()
            if not content.startswith("gitdir:"):
                continue
            admin = Path(content.removeprefix("gitdir:").strip())
            owner = admin.parents[2]  # <owner>/.git/worktrees/<id> -> <owner>
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(tree)],
                cwd=str(owner),
                capture_output=True,
                timeout=30,
            )


def _sweep_stale_gate_tmp(root: Path) -> None:
    """Opportunistic hygiene: reap >7-day-old bt-*/step9c-scratch-* strays under *root*.

    Nothing else reaps this location (the #681 data-disk guard pass is
    escalate-only), so a SIGKILL'd compare's leaked basetemp / scratch tree
    would otherwise accumulate. Best-effort: capped at GATE_TMP_SWEEP_CAP
    entries per call, every failure suppressed, fresh entries + foreign
    names never touched.
    """
    with contextlib.suppress(OSError):
        now = time.time()
        reaped = 0
        for entry in sorted(root.iterdir()):
            if reaped >= GATE_TMP_SWEEP_CAP:
                break
            name = entry.name
            if not (name.startswith("bt-") or name.startswith("step9c-scratch-")):
                continue
            with contextlib.suppress(OSError):
                if now - entry.stat().st_mtime <= GATE_TMP_SWEEP_MAX_AGE_S:
                    continue
                if name.startswith("step9c-scratch-"):
                    _reap_leaked_scratch_admin(entry)
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                reaped += 1


def gate_tmp_root(*, sweep: bool = True) -> Path | None:
    """Resolve the gate/pristine/scratch bulk temp-write root (#1408; #1363).

    Precedence:

    1. ``EPM_STEP9C_TMPDIR`` set-and-nonempty -> used verbatim; it must exist
       and be writable, else ToolMissingError (an EXPLICIT override is a
       fail-loud misconfig — callers map it to exit 2). Set-but-EMPTY ->
       None (routing disabled; the test-suite determinism switch).
    2. The data disk (``EPS_VM_DATA_DISK_PATH``, default ``/mnt/eps-data``),
       required to be a LIVE mount: first writable of ``<disk>/tmp``
       (preferred, 17 chars — pre-created by ops, NEVER auto-created here:
       the disk top level is root-owned) then ``<disk>/<user>/tmp``
       (auto-created best-effort).
    3. Anything else -> None (silent inherit — pods/GCE have no data disk).

    Short roots matter: pytest basetemp prefixes derived from the root must
    stay within GATE_TMP_MAX_PREFIX_CHARS (the AF_UNIX ~108-byte socket-path
    cap, #1363). On resolution the >7-day bt-*/step9c-scratch-* strays under
    the root are opportunistically swept (``sweep=False`` skips — the
    JSON-report call).
    """
    override = os.environ.get("EPM_STEP9C_TMPDIR")
    if override is not None:
        if not override:
            return None
        p = Path(override)
        if not p.is_dir() or not _gate_tmp_writable(p):
            raise ToolMissingError(
                f"EPM_STEP9C_TMPDIR={override!r} does not exist or is not writable — "
                "explicit gate-tmp override misconfigured (fail-loud; unset it, or set "
                "it empty to disable routing)"
            )
        if len(str(p)) > GATE_TMP_MAX_PREFIX_CHARS:
            _log(
                f"WARN: EPM_STEP9C_TMPDIR={p} exceeds {GATE_TMP_MAX_PREFIX_CHARS} chars — "
                "derived pytest basetemp paths risk the AF_UNIX ~108-byte socket cap (#1363)"
            )
        if sweep:
            _sweep_stale_gate_tmp(p)
        return p
    disk = Path(os.environ.get("EPS_VM_DATA_DISK_PATH") or "/mnt/eps-data")
    if not os.path.ismount(str(disk)):
        return None
    shared = disk / "tmp"
    if shared.is_dir() and _gate_tmp_writable(shared):
        if sweep:
            _sweep_stale_gate_tmp(shared)
        return shared
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        return None
    user_tmp = disk / user / "tmp"
    with contextlib.suppress(OSError):
        user_tmp.mkdir(parents=True, exist_ok=True)
    if user_tmp.is_dir() and _gate_tmp_writable(user_tmp):
        if sweep:
            _sweep_stale_gate_tmp(user_tmp)
        return user_tmp
    return None


def _gate_tmp_dir_arg() -> str | None:
    """gate_tmp_root() as a ``tempfile`` ``dir=`` argument (None -> default /tmp)."""
    root = gate_tmp_root()
    return None if root is None else str(root)


def _gate_tmp_root_str() -> str | None:
    """The resolved routing root as a JSON-safe string (None = disabled/absent)."""
    root = gate_tmp_root(sweep=False)
    return None if root is None else str(root)


def run_pytest(
    files: Iterable[str],
    cwd: Path,
    timeout_s: float,
    junit_path: Path,
    extra: Iterable[str] = PYTEST_BASE_FLAGS,
    *,
    python_exe: str,
    pythonpath: str | None = None,
) -> int:
    """Run one bounded, thread-capped pytest subprocess; return its exit code.

    ``python_exe`` is REQUIRED and must be the TARGET root's own interpreter
    (``resolve_root_python``): the subprocess imports whatever library code its
    interpreter's venv resolves, so running the invoking ``sys.executable`` from
    an issue worktree would execute the WORKTREE's editable ``src/`` against
    main's test files — the #1022 round-2 Critical (a branch-caused ``src/``
    regression would then fail "pristine" too and be stripped as pre-existing).
    ``PYTHONPATH`` is stripped from the child env for the same reason (an
    exported ``PYTHONPATH=<wt>/src`` would override the resolved venv).
    Inherited ``PYTHONPATH`` is ALWAYS stripped (the #1022 vector); an
    explicitly passed ``pythonpath`` is set afterwards and must be derived from
    a HEAD-pinned tree (the #1251 scratch shadow) — the two are different trust
    classes (ambient env vs caller-constructed).
    When ``gate_tmp_root()`` resolves (#1408), the child gets ``TMPDIR=<root>``
    (library ``tempfile.*`` writes) plus a fresh SHORT ``--basetemp`` directly
    under the root (``<root>/bt-XXXXXXXX/p``) — ``--basetemp`` is REQUIRED, not
    optional: TMPDIR-only routing nests ``pytest-of-<user>/pytest-N`` under the
    root and overflows the AF_UNIX ~108-byte socket cap (#1363 attempt 3); the
    fresh per-call mkdtemp keeps pytest's clear-at-session-start semantics
    concurrency-safe, and a finally-scoped rmtree reaps it on return/kill alike.
    ``start_new_session=True`` + ``os.killpg`` on ``TimeoutExpired`` group-kills
    stragglers, then the ``TimeoutExpired`` is re-raised (callers exit 2 —
    NEVER a ledger write / classification from a timed-out run).
    """
    argv = [
        python_exe,
        "-m",
        "pytest",
        *files,
        *extra,
        f"--junitxml={junit_path}",
    ]
    env = thread_capped(os.environ)
    env.pop("PYTHONPATH", None)  # never let the invoking checkout's src/ shadow the root venv
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath  # caller-derived from a HEAD-pinned tree (#1251)
    tmp_root = gate_tmp_root()
    basetemp: Path | None = None
    if tmp_root is not None:
        env["TMPDIR"] = str(tmp_root)
        basetemp = Path(tempfile.mkdtemp(prefix="bt-", dir=str(tmp_root)))
        argv.append(f"--basetemp={basetemp / 'p'}")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
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
    finally:
        if basetemp is not None:
            shutil.rmtree(basetemp, ignore_errors=True)


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


def _porcelain_paths(out: str) -> list[str]:
    """Parse ``git status --porcelain`` output into paths (rename entries -> new name)."""
    paths: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        p = line[3:].strip()
        if " -> " in p:  # rename entry: "old -> new"
            p = p.split(" -> ", 1)[1]
        paths.append(p.strip('"'))
    return paths


def dirty_code_paths(root: Path) -> list[str]:
    """Return uncommitted CODE-path changes at *root* (DIRTY_CODE_PATHSPEC scope).

    Scoped so the perpetual non-code churn on the shared root (tasks/**,
    pods_ephemeral.json, agent-memory .md) never reads as dirt (MF-4a).
    """
    return _porcelain_paths(_git_out(["status", "--porcelain", "--", *DIRTY_CODE_PATHSPEC], root))


def scratch_contamination_probe(root: Path) -> list[str]:
    """Uncommitted paths a scratch tree CANNOT neutralize (#1077).

    ANY file under top-level ``src/`` — the root venv's editable ``.pth``
    statically resolves ``<root>/src`` regardless of cwd, and package data
    rides ``importlib.resources`` — plus ``pyproject.toml``/``uv.lock``,
    which the venv's installed deps derive from.
    ``git status --porcelain -- src/ pyproject.toml uv.lock``, parsed exactly
    like ``dirty_code_paths()``. Since #1251 the in-package
    ``src/explore_persona_space/**`` legs are shadowable (neutralized by the
    scratch PYTHONPATH shadow — see ``residual_scratch_contamination``); the
    residual legs (``pyproject.toml``/``uv.lock``, out-of-package ``src/``)
    keep the fail-closed MF-4c exit 2.
    """
    return _porcelain_paths(
        _git_out(["status", "--porcelain", "--", *SCRATCH_CONTAMINATION_PATHSPEC], root)
    )


def residual_scratch_contamination(paths: Iterable[str]) -> list[str]:
    """The contamination-probe paths a PYTHONPATH src-shadow CANNOT neutralize (#1251).

    In-package dirt (``src/explore_persona_space/**``) IS neutralized: the
    scratch pristine pytest runs with ``PYTHONPATH=<scratch>/src``, so the
    scratch's HEAD-pinned package shadows the root venv's static editable
    ``.pth`` in sys.path order (verified per compare by
    ``assert_scratch_src_shadow``); package data (query-bank ``.json``) rides
    the winning package ``__path__`` too. ``pyproject.toml`` / ``uv.lock``
    dirt is NOT shadowable — the venv's INSTALLED DEPS derive from them — so
    those legs keep the fail-closed MF-4c exit 2. ANY other path is residual
    by construction: an out-of-package ``src/`` file (e.g. an untracked
    ``src/rogue.py``) stays importable from the root via the ``.pth``'s
    ``<root>/src`` sys.path entry and is NOT covered by the package shadow,
    so it blocks too (fail-closed for oddballs, incl. a top-level file
    literally named ``src``).
    """
    return [p for p in paths if not p.startswith("src/explore_persona_space/")]


# --- Scratch-worktree fallback helpers (#1077) -------------------------------------


@dataclass
class _ScratchTree:
    """A materialized detached sparse scratch worktree (the dirty-oracle fallback)."""

    parent: Path  # the mkdtemp dir (rmtree target)
    path: Path  # the worktree itself: parent / f"tree-{os.getpid()}"
    sha: str  # the detached HEAD sha (== root HEAD at creation)


def _git_bounded(argv: list[str], cwd: Path, timeout_s: float) -> None:
    """Run one git command with check=True + timeout (scratch lifecycle only)."""
    subprocess.run(
        ["git", *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_s,
    )


def _work_root_sparse_cones(wt: Path) -> list[str] | None:
    """The invoking work root's ACTUAL sparse profile via ``git sparse-checkout list``.

    None when the work root is not sparse. R-G's original impossibility
    stands — a non-sparse gate layout cannot be superset-matched by a sparse
    scratch without a full multi-GB checkout — but since #2019 (R-G') the
    caller no longer treats None as unconditionally fallback-INELIGIBLE: a
    DIRTY non-sparse work root arms the scratch oracle with the
    ``_scratch_cones`` FLOOR profile (``wt_cones=[]``) under ASYMMETRIC
    verdicts — a node GREEN at pristine HEAD classifies NEW (rc 1,
    ``pristine_oracle: scratch-worktree-floor``); a node RED there REFUSES
    the strip (exit 2), because the non-superset floor tree cannot certify
    "pre-existing". A CLEAN non-sparse root keeps the trustworthy root
    oracle unchanged. On git 2.34 a non-sparse tree exits 0 with EMPTY
    stdout (warning on stderr), so an empty list is folded into None too; a
    genuinely failing command (non-git dir, ancient git) also maps to None.
    """
    try:
        lines = [
            ln.strip()
            for ln in _git_out(["sparse-checkout", "list"], wt).splitlines()
            if ln.strip()
        ]
    except subprocess.CalledProcessError:
        return None
    return lines or None


def _scratch_cones(root: Path, wt_cones: list[str]) -> list[str]:
    """Scratch sparse profile = SUPERSET of the gate layout (R-G).

    The union of: the work root's actual ``sparse-checkout list`` (per-issue +
    manually-added cones included — the legs the registry alone omits), the
    HEAD-PINNED ``tests/sparse_cones.txt`` (``git show HEAD:...`` — never the
    live working-tree file, which is itself decontaminable-classified dirt),
    and every top-level tracked dir minus SCRATCH_EXCLUDES as the floor.
    """
    dirs = [
        d
        for d in _git_out(["ls-tree", "--name-only", "-d", "HEAD"], root).splitlines()
        if d and d not in SCRATCH_EXCLUDES
    ]
    registry_lines: list[str] = []
    # Registry absent at HEAD: the floor still applies, no raise.
    with contextlib.suppress(subprocess.CalledProcessError):
        registry_lines = [
            ln.strip()
            for ln in _git_out(["show", "HEAD:tests/sparse_cones.txt"], root).splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return sorted(set(dirs) | set(registry_lines) | set(wt_cones))


def create_scratch_worktree(root: Path, wt_cones: list[str], timeout_s: float) -> _ScratchTree:
    """Materialize a detached SPARSE scratch tree at *root*'s HEAD under a fresh tmp dir.

    Sequence mirrors ``new_worktree.sh`` ``_sparse_setup`` (``init --cone``
    FIRST, then ``set``, then populate from ``--no-checkout`` limbo). Profile =
    ``_scratch_cones(root, wt_cones)`` — a superset of the gate layout (R-G).
    Any failure tears down partial state and re-raises (the caller maps it to
    ``_Indeterminate`` — or, on a CLEAN root, degrades to the root oracle,
    #1408). Bounded per git command by *timeout_s*. The ~1 GB tree lands under
    ``gate_tmp_root()`` when routing resolves (#1408; default ``/tmp`` else).
    """
    sha = git_head(root)
    parent = Path(tempfile.mkdtemp(prefix="step9c-scratch-", dir=_gate_tmp_dir_arg()))
    tree = parent / f"tree-{os.getpid()}"
    try:
        _git_bounded(
            ["worktree", "add", "--detach", "--no-checkout", str(tree), sha],
            cwd=root,
            timeout_s=timeout_s,
        )
        _git_bounded(["sparse-checkout", "init", "--cone"], cwd=tree, timeout_s=timeout_s)
        _git_bounded(
            ["sparse-checkout", "set", *_scratch_cones(root, wt_cones)],
            cwd=tree,
            timeout_s=timeout_s,
        )
        _git_bounded(["checkout", sha], cwd=tree, timeout_s=timeout_s)
    except BaseException:
        remove_scratch_worktree(root, _ScratchTree(parent=parent, path=tree, sha=sha))
        raise
    return _ScratchTree(parent=parent, path=tree, sha=sha)


def remove_scratch_worktree(root: Path, scratch: _ScratchTree) -> None:
    """Best-effort teardown; never raises (the verdict is already decided).

    Deliberately NO ``git worktree prune``: prune reaps admin entries of ANY
    worktree whose dir is unreachable, so a future bind-mount outage on
    ``.claude/worktrees`` could make a compare's teardown break LIVE
    worktrees. A SIGKILL'd compare leaves one cosmetic stale admin entry
    (git gc's ``gc.worktreePruneExpire`` sweeps it) + an inert ``/tmp`` dir.
    """
    with contextlib.suppress(Exception):
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(scratch.path)],
            cwd=str(root),
            capture_output=True,
            timeout=60,
        )
    shutil.rmtree(scratch.parent, ignore_errors=True)


def assert_scratch_src_shadow(root: Path, scratch_path: Path, timeout_s: float) -> None:
    """Verify PYTHONPATH=<scratch>/src actually WINS over root's editable install (#1251).

    Runs the ROOT venv interpreter (the exact interpreter ``run_pytest`` uses)
    with the exact child-env shape (inherited ``PYTHONPATH`` stripped, ours
    set) and the pytest child's cwd (the scratch), and
    ``importlib.util.find_spec``'s the package WITHOUT executing it; requires
    the resolved origin to sit under the scratch tree. Guards against
    editable-install styles that preempt sys.path order (meta-path finder
    hooks; easy-install-style ``.pth`` reordering), against a
    namespace-package refactor (``find_spec`` origin is None => fail), and
    against a scratch missing ``src/`` (origin resolves to root => fail).
    Raises PristineRunError on ANY failure — the caller maps it to the
    fail-closed exit 2; a verdict never rests on an unverified shadow.
    """
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.find_spec('explore_persona_space')\n"
        "origin = (spec.origin or '') if spec else ''\n"
        "sys.exit(0 if origin.startswith(sys.argv[1] + '/') else 3)\n"
    )
    try:
        python_exe = resolve_root_python(root)
    except ToolMissingError as exc:
        raise PristineRunError(str(exc)) from exc
    env = thread_capped(os.environ)
    env.pop("PYTHONPATH", None)  # same shape as run_pytest: inherited stripped, ours set
    env["PYTHONPATH"] = str(scratch_path / "src")
    try:
        proc = subprocess.run(
            [python_exe, "-c", code, str(scratch_path)],
            cwd=str(scratch_path),
            env=env,
            capture_output=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise PristineRunError(f"src-shadow probe failed to run: {exc}") from exc
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode(errors="replace").strip()[-200:]
        raise PristineRunError(
            f"src-shadow probe rc={proc.returncode}: PYTHONPATH={scratch_path / 'src'} did NOT "
            "win over the root venv's editable install — src dirt cannot be neutralized "
            "(fail-closed; --no-src-shadow restores the #1077 eligibility)"
            + (f"; probe stderr: {stderr_tail}" if stderr_tail else "")
        )


def _ruff_bin() -> str:
    """Resolve the ruff binary; missing -> ToolMissingError (fail loud, exit 2 — MF-5).

    Deliberately the INVOKING checkout's PATH ruff — NOT a root-venv resolution
    like the pytest runners (``resolve_root_python``): ruff lints file text and
    imports nothing from the linted tree, and ``lint_verdict`` counts BOTH the
    base and worktree trees with the SAME binary, so the deltas stay internally
    consistent (#1022 round-2 consistent-deltas note).
    """
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

    Collect-error absorb (#1746, ``--continue-on-collection-errors``): pytest
    9.0.2 empirically emits the ``file`` attribute on a collect-error testcase
    too (probe 2026-07-28: ``file="tests/test_broken.py"``, ``classname=""``,
    ``name="tests.test_broken"``), so the broken file keys to a stable Node via
    the normal path. As version-drift insurance, a testcase with an ``error``
    child, NO ``file`` attr, and a ``name`` that is a plausible test-file path
    (endswith ``.py``) derives ``file`` from ``name``
    (``Node(file=name, classname="", name=name)``); every OTHER missing-file
    shape keeps the hard JunitParseError (the xunit1 contract stays fail-loud).
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
            name_attr = tc.get("name") or ""
            if not file_attr and has_error and name_attr.endswith(".py"):
                # Collect-error row keyed only through ``name`` (#1746 —
                # version-drift fallback; see docstring): derive a stable
                # per-file Node from the plausible test-file path in ``name``.
                failing.append(Node(file=name_attr, classname="", name=name_attr))
                continue
            if not file_attr:
                raise JunitParseError(
                    f"failing testcase {tc.get('classname')}::{tc.get('name')} has no "
                    "file attribute — xunit1 contract violated (see plan #1022 K2 fallback)"
                )
            failing.append(
                Node(file=file_attr, classname=tc.get("classname") or "", name=name_attr)
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
    """Load + schema-validate the ledger; None (with a loud stderr line) when unusable.

    Validation covers the top-level key set AND the per-entry ``failing_tests``
    shape (each entry a dict with str ``file`` + ``name``) — a malformed entry
    must route to the unusable-ledger indeterminate path (exit 2), never crash
    ``ledger_nodes`` with a KeyError that Python turns into a misleading exit 1
    with no JSON emitted (#1022 round-2 Minor).
    """
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
        or not isinstance(data.get("failing_tests"), list)
        or not all(
            isinstance(e, dict)
            and isinstance(e.get("file"), str)
            and isinstance(e.get("name"), str)
            for e in data["failing_tests"]
        )
    ):
        _log(f"ledger at {path} fails schema v{SCHEMA_VERSION} validation (keys or entry shape)")
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
    """Run the Step 9c workflow-invariant universe on main; write the ledger atomically."""
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
    try:
        # The ledger's failing set must be measured under the ROOT's own library
        # code — never the invoking checkout's (#1022 round-2 Critical sibling).
        python_exe = resolve_root_python(root)
    except ToolMissingError as exc:
        _log(f"refresh: {exc} — NO ledger write")
        return 2
    junit = root / ".claude" / "cache" / "step9c-baseline-junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    junit.unlink(missing_ok=True)  # same stale-junit lifecycle as the gate (MF-1a)
    t0 = time.monotonic()
    try:
        rc = run_pytest(
            files=universe,
            cwd=root,
            timeout_s=args.timeout_s,
            junit_path=junit,
            python_exe=python_exe,
        )
    except subprocess.TimeoutExpired:
        _log(f"refresh pytest timed out after {args.timeout_s}s — NO ledger write")
        return 2
    except ToolMissingError as exc:
        # #1408: a misconfigured explicit EPM_STEP9C_TMPDIR override fails loud.
        _log(f"refresh: {exc} — NO ledger write")
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
        # gate_tmp_root is additive to the PRINTED payload only — the persisted
        # ledger's key set is pinned (REQUIRED_LEDGER_KEYS exactness, #1408).
        print(
            json.dumps({**ledger, "gate_tmp_root": _gate_tmp_root_str()}, indent=2, sort_keys=True)
        )
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


def run_single_file_pristine(
    test_file: str,
    cwd: Path,
    timeout_s: float,
    *,
    venv_root: Path | None = None,
    pythonpath: str | None = None,
) -> set[Node]:
    """Run ONE test file at the pristine oracle *cwd*; return its failing nodes.

    Executes the TARGET root's OWN venv interpreter (``resolve_root_python``)
    so the oracle runs MAIN's library code — never the invoking worktree's,
    whose editable ``src/`` would make a branch-caused regression fail
    "pristine" too and get stripped as pre-existing (#1022 round-2 Critical).
    ``venv_root`` (scratch-oracle mode, #1077): the interpreter is resolved
    from the MAIN root while *cwd* is the scratch tree — the scratch has no
    venv. Scratch mode additionally passes ``pythonpath=<scratch>/src``
    (#1251) so the scratch's HEAD-pinned package shadows the root venv's
    editable ``.pth`` — only ``pyproject.toml``/``uv.lock`` (and
    out-of-package ``src/``) dirt remains gated out before this mode is used.
    A missing venv raises PristineRunError (indeterminate, exit 2 — never a
    silent ``sys.executable`` fallback). Bounded + thread-capped like refresh.
    rc not in {0, 1}, a timeout, or a zero-collected run raises
    PristineRunError (MF-5); a classification must never rest on an aborted
    pristine run.
    """
    try:
        python_exe = resolve_root_python(venv_root if venv_root is not None else cwd)
    except ToolMissingError as exc:
        raise PristineRunError(str(exc)) from exc
    fd, tmp = tempfile.mkstemp(
        prefix="step9c-pristine-junit-", suffix=".xml", dir=_gate_tmp_dir_arg()
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.unlink(missing_ok=True)  # pytest must create it fresh (MF-1a parity)
        try:
            rc = run_pytest(
                files=[test_file],
                cwd=cwd,
                timeout_s=timeout_s,
                junit_path=tmp_path,
                python_exe=python_exe,
                pythonpath=pythonpath,
            )
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


# --- pristine-timeout sizing (#1129). -----------------------------------------
# Derived from the selector's #1046 gate-timeout knowledge so there is ONE
# per-file runtime table. #1098: the pristine single-file run of
# tests/test_workflow_lint.py needs ~780 s; the fixed 600 s default killed it
# and forced a full compare rerun. Bias generous: an oversized bound only
# delays a genuinely wedged run; an undersized one GUARANTEES a wasted
# compare + rerun (exit 2 via PristineRunError).
PRISTINE_SLOW_TIMEOUT_MULT = 2.0
PRISTINE_TIMEOUT_FLOOR_S = (
    600.0  # == the pre-#1129 fixed default (#1022): non-surcharge behavior unchanged
)


def derive_pristine_timeout_s(sel: object, test_file: str) -> float:
    """Per-file pristine-oracle bound: BASE + PER_FILE + 2x slow surcharge, floor 600 s.

    Reads the #1046 constants off the LOADED selector module (``ctx.sel``) via
    ``getattr`` so a pre-#1046 worktree selector copy (deliberate version skew,
    #1022 §3.3) degrades to the legacy 600 s floor instead of crashing.
    tests/test_workflow_lint.py -> 120 + 30 + 2*2400 = 4950 s (#1646: in-gate
    walls median 789 s / max 1819 s over 330 junits 2026-07-13..24, ~2.7x the
    worst measured; #1098 had measured ~780 s pristine at the old 900 surcharge);
    files without surcharge knowledge -> 600 s.
    """
    base = float(getattr(sel, "TIMEOUT_BASE_S", 0))
    per_file = float(getattr(sel, "TIMEOUT_PER_FILE_S", 0))
    slow = getattr(sel, "SLOW_TESTS", None) or {}
    surcharge = float(slow.get(test_file, 0))
    return max(base + per_file + PRISTINE_SLOW_TIMEOUT_MULT * surcharge, PRISTINE_TIMEOUT_FLOOR_S)


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
    """The copy-pasteable single-file pristine check printed on the no-run path.

    Built from ``PYTEST_BASE_FLAGS`` (single source — #1746 Must-Fix 1: a
    duplicated literal here would drop ``--continue-on-collection-errors`` and
    make the printed manual-recovery command abort rc=2 on a collection-red
    file instead of reproducing the oracle's flags).
    """
    return f"(cd {root} && uv run pytest {test_file} {' '.join(PYTEST_BASE_FLAGS)})"


class _Indeterminate(RuntimeError):
    """Internal control flow: compare cannot classify — the caller exits 2.

    ``extra`` carries structured payload fields the exit-2 ``--json`` object
    surfaces (e.g. ``live_dirty_paths``, ``contaminating_paths``); ``warns``
    carries the ctx-accumulated WARNs so scratch-oracle provenance is never
    dropped on a mid-loop failure (#1077).
    """

    def __init__(self, msg: str, extra: dict | None = None, warns: list[str] | None = None):
        super().__init__(msg)
        self.extra = extra or {}
        self.warns = list(warns or [])


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
    work_root: Path  # the invoking worktree (its sparse profile gates the scratch fallback)
    new: list[Node] = field(default_factory=list)
    stripped: list[dict] = field(default_factory=list)
    urgent_park: list[str] = field(default_factory=list)  # #1742 <file>::<name> node ids
    pristine_bucket: list[Node] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)
    live_dirty_paths: list[str] = field(default_factory=list)
    pristine_files_run: list[str] = field(default_factory=list)
    pristine_oracle: str = "root"  # "scratch-worktree" once the scratch oracle arms
    # (#1077; the DEFAULT whenever eligible since #1408); "scratch-worktree-floor"
    # when a DIRTY non-sparse work root armed the floor-profile scratch (R-G', #2019)
    scratch_sha: str | None = None
    scratch_src_shadow: bool = False  # True once the #1251 PYTHONPATH shadow is armed + probed
    scratch_degraded: bool = False  # True on the #1408 clean-root scratch-failure fallback


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
        base = args.base
        if base is None:
            resolve = getattr(sel, "resolve_base", None)
            # #1289: match the selection run's base (same-mapping-logic, #1022).
            # fetch=False — the gate's selector run already fetched, and the
            # three-dot merge-base is invariant under origin/main advancing,
            # so a second fetch buys nothing and risks nothing skipping.
            # A pre-#1289 worktree selector has no resolve_base: keep that
            # era's behavior (local main) — self-consistent per worktree.
            base = (
                resolve(getattr(sel, "DEFAULT_BASE", "origin/main"), wt, fetch=False)
                if resolve is not None
                else "main"
            )
        touched = sel.compute_touched(base, wt)
        _tests, _untested, reasons = sel.select_tests_with_reasons(touched, wt)
    except (FileNotFoundError, subprocess.CalledProcessError, AttributeError) as exc:
        raise _Indeterminate(f"selector load / touched-diff failed at {wt}: {exc}") from exc
    diff_linked = {t for t, rs in reasons.items() if any(r != "invariant" for r in rs)} | {
        f for f in touched if f.startswith("tests/")
    }
    return _CompareCtx(sel=sel, touched=touched, diff_linked=diff_linked, work_root=wt)


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
    """Strip *node* as pre-existing; EVERY scan-covered strip WARNs (MF-6).

    A strip on a WORKFLOW_INVARIANT test additionally demands an urgent park
    (#1713/#1742): the node id joins ``ctx.urgent_park`` (the JSON
    ``urgent_park_required`` field) and a loud stderr demand line is emitted —
    criterion single-sourced from the selector, never a hardcoded glob.
    """
    ctx.stripped.append({**node._asdict(), "via": via})
    sel = ctx.sel
    if node.file in getattr(sel, "WORKFLOW_INVARIANT", ()):
        node_id = f"{node.file}::{node.name}"
        ctx.urgent_park.append(node_id)
        _log(
            f"URGENT-PARK-REQUIRED: {node_id} — stripped pre-existing main-red on a "
            "workflow-invariant test; emit (or verify existing) a routable "
            "'urgency: main-red' workflow-fix-candidate (#1713/#1742)"
        )
    if node.file in sel.GLOB_SCAN_TESTS:
        covered = [f for f in ctx.touched if sel._matches_any(f, sel.GLOB_SCAN_TESTS[node.file])]
        ctx.warns.append(
            f"MASKING WARN: stripped scan test {node.file}::{node.name} (via {via}) — "
            f"its directory scan covers touched file(s) {covered or '[]'}; re-check them "
            "against that scan's rule"
        )
    if via.startswith("pristine") and node.file in ctx.diff_linked:
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


def _arm_src_shadow(
    ctx: _CompareCtx,
    root: Path,
    scratch: _ScratchTree,
    contaminating: list[str],
    args: argparse.Namespace,
    *,
    floored: bool = False,
) -> None:
    """Run the once-per-compare #1251 shadow probe + record scratch-oracle provenance.

    Called immediately after ``create_scratch_worktree`` (the caller assigns
    ``scratch`` FIRST so its ``finally`` teardown covers a probe raise). With
    the shadow armed (default), ``assert_scratch_src_shadow`` verifies
    ``PYTHONPATH=<scratch>/src`` wins over the root venv's editable install —
    a probe failure raises ``_Indeterminate`` (the caller maps it fail-closed
    to exit 2 on a DIRTY root, or degrades to the root oracle on a CLEAN one,
    #1408 — a verdict never rests on an unverified shadow). Under
    ``--no-src-shadow`` no probe runs (the contamination probe was fully clean
    by eligibility there). ``floored=True`` (R-G', #2019 — a DIRTY non-sparse
    work root armed the ``_scratch_cones`` FLOOR profile) records
    ``pristine_oracle: scratch-worktree-floor`` so the caller's asymmetric
    verdict rule (green -> NEW; red -> strip refused) is auditable in the
    JSON. WARN discipline (#1408): the SCRATCH-ORACLE WARN
    fires ONLY when root dirt was actually neutralized (``live_dirty_paths``
    non-empty) — on a clean root the scratch is the NORMAL path and provenance
    rides the JSON fields (``pristine_oracle``/``scratch_sha``/
    ``scratch_src_shadow``) without a WARN. ``ctx`` fields are set only AFTER
    the probe passes, so the degradation path needs no ctx reset.
    """
    if not args.no_src_shadow:
        try:
            assert_scratch_src_shadow(root, scratch.path, timeout_s=args.scratch_timeout_s)
        except PristineRunError as exc:
            raise _Indeterminate(
                f"scratch src-shadow probe failed ({exc}) — dirty oracle unresolvable",
                extra={
                    "live_dirty_paths": ctx.live_dirty_paths,
                    "contaminating_paths": contaminating,
                },
                warns=ctx.warns,
            ) from exc
    ctx.scratch_src_shadow = not args.no_src_shadow
    ctx.pristine_oracle = "scratch-worktree-floor" if floored else "scratch-worktree"
    ctx.scratch_sha = scratch.sha
    if not ctx.live_dirty_paths:
        # #1408 scratch-by-default: on a CLEAN root the scratch is the normal
        # path — no dirt was neutralized, so a WARN would be pure noise.
        return
    profile = (
        "detached FLOOR-profile scratch worktree (non-sparse work root, R-G': "
        "green resolves NEW, red refuses the strip)"
        if floored
        else "detached sparse scratch worktree"
    )
    if args.no_src_shadow:
        ctx.warns.append(
            f"SCRATCH-ORACLE WARN: root state: dirty on {ctx.live_dirty_paths[:20]} "
            f"(non-contaminating); pristine oracle re-rooted to a "
            f"{profile} at {scratch.sha[:12]} (root venv interpreter; "
            "contamination probe src//pyproject.toml/uv.lock was clean; "
            "non-file-anchored scan-set nodes stay indeterminate)"
        )
    else:
        src_dirt = [p for p in contaminating if p.startswith("src/")]
        ctx.warns.append(
            f"SCRATCH-ORACLE WARN: root state: dirty on {ctx.live_dirty_paths[:20]}; "
            f"src-dirt {src_dirt[:20] or 'none'} "
            f"neutralized via PYTHONPATH=<scratch>/src (shadow probe verified); "
            f"pristine oracle re-rooted to a {profile} at "
            f"{scratch.sha[:12]} (root venv interpreter; residual probe "
            "pyproject.toml/uv.lock/out-of-package-src was clean; non-file-anchored "
            "scan-set nodes stay indeterminate)"
        )


def _create_scratch_or_degrade(
    ctx: _CompareCtx,
    root: Path,
    wt_cones: list[str],
    contaminating: list[str],
    args: argparse.Namespace,
    *,
    floored: bool = False,
) -> _ScratchTree | None:
    """Create + arm the scratch oracle; degrade to the root oracle on a CLEAN root.

    DIRTY-root failure (creation or shadow probe) stays fail-closed
    (#1077/#1251): the partial scratch is torn down and ``_Indeterminate``
    raises (exit 2). CLEAN-root failure (#1408): the root oracle is
    trustworthy (pre-#1077 behavior) — the partial scratch is torn down and
    None returns, with the degradation recorded (WARN + ``scratch_degraded``
    JSON flag); the caller memoizes so creation is not re-attempted per file
    while the root stays clean. ``floored=True`` (R-G', #2019): the caller
    passes ``wt_cones=[]`` — ``_scratch_cones(root, [])`` = floor union
    HEAD-pinned registry — and the floored arm only fires on a DIRTY root,
    so a floored creation/probe failure always takes the fail-closed dirty
    branch (never a silent root-oracle downgrade). A misconfigured explicit
    ``EPM_STEP9C_TMPDIR`` (ToolMissingError from the mkdtemp routing) is NOT
    degraded — it propagates to the fail-loud exit-2 mapping in
    ``cmd_compare``.
    """
    scratch: _ScratchTree | None = None
    try:
        scratch = create_scratch_worktree(root, wt_cones, timeout_s=args.scratch_timeout_s)
        # scratch is assigned BEFORE the probe, so a probe raise still has a
        # handle to tear down (no leak on either branch below).
        _arm_src_shadow(ctx, root, scratch, contaminating, args, floored=floored)
        return scratch
    except (
        _Indeterminate,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as exc:
        if scratch is not None:
            remove_scratch_worktree(root, scratch)
        if ctx.live_dirty_paths:
            # DIRTY root: unchanged fail-closed (#1077/#1251).
            if isinstance(exc, _Indeterminate):
                raise
            raise _Indeterminate(
                f"scratch-worktree fallback failed ({exc}) — dirty oracle unresolvable",
                extra={"live_dirty_paths": ctx.live_dirty_paths},
                warns=ctx.warns,
            ) from exc
        ctx.scratch_degraded = True
        ctx.warns.append(
            "SCRATCH-ORACLE WARN: scratch creation/probe failed on a "
            f"CLEAN root ({exc}) — root oracle used (root state: clean)"
        )
        return None


def _resolve_pristine_bucket(ctx: _CompareCtx, root: Path, args: argparse.Namespace) -> None:
    """Resolve bucketed nodes via bounded single-file pristine runs (or refuse).

    Scratch-by-default (#1408; #1077's dirty-only trigger removed): the
    pristine oracle is BY DEFAULT a detached sparse scratch worktree at the
    root's HEAD whenever the node is physically eligible — clean or dirty
    root alike — created lazily once per compare (the shadow probe runs ONCE
    at creation), reused for every eligible bucketed file, ALWAYS removed in
    the ``finally``. Per-file eligibility: no RESIDUAL contaminating dirt
    (R-B', #1251: in-package ``src/explore_persona_space/**`` dirt is
    neutralized by the probe-verified ``PYTHONPATH=<scratch>/src`` shadow, so
    only ``pyproject.toml``/``uv.lock`` and out-of-package ``src/`` dirt still
    block; ``--no-src-shadow`` restores the #1077 any-probe-hit rule), a
    sparse work root OR a dirty non-sparse one (R-G', #2019 — see below), a
    non-scan-set node OR a
    ``FILE_ANCHORED_SCAN_TESTS`` member (R-F' — ``repo_root()``-anchored
    live-tree scanners read the MAIN root from any cwd, so a scratch cannot
    decontaminate them; a source-verified ``__file__``-anchored scanner scans
    its own tree, #1337), and the scratch oracle not being disabled via
    ``--no-scratch-fallback``. For an allowlisted scan node R-F' shifts one
    verdict class: a scan failure caused solely by live-root strays (untracked
    offenders absent from the HEAD-pinned scratch) classifies NEW (rc 1)
    instead of exit 2 — fail-closed in direction (never a silent strip), but
    it attributes the failure to the branch. R-G' (#2019): a DIRTY non-sparse
    work root arms the scratch with the ``_scratch_cones`` FLOOR profile
    (``wt_cones=[]`` — the identical floor every sparse-root scratch already
    materializes), under ASYMMETRIC verdicts because the floor tree is NOT a
    superset of a non-sparse gate layout: a node GREEN at pristine HEAD
    classifies NEW (rc 1, ``pristine_oracle: scratch-worktree-floor`` — the
    #1932 shape gets a definite verdict), while a node RED there REFUSES the
    strip (exit 2, R-G' diagnostic) — the former non-sparse exit-2 class maps
    ONLY onto {NEW rc 1, exit 2}, never rc 0, never red->green. A CLEAN
    non-sparse root keeps the trustworthy root oracle (strictly more capable:
    both strip and NEW). Mid-loop dirt nuance (#2019): a file stripped via
    the root oracle while the root was clean at ITS probe time can coexist
    with a later file's floored NEW in one rc-1 result (pre-#2019, that later
    red-on-a-dirty-root would exit-2 the whole compare and discard the
    strip) — rc 1 still blocks, and the strip rides only into MF-6
    masking-WARN semantics, so this is not a red->green channel. Root-oracle
    runs remain only for
    scratch-INELIGIBLE nodes (trustworthy on a clean root; fail-closed MF-4c
    exit 2 when the root is dirty and the node fails on main) and for the
    CLEAN-root degradation path: a scratch creation/probe failure on a CLEAN
    root tears down any partial scratch and falls back to the trustworthy
    root oracle (WARN + ``scratch_degraded`` in the JSON, never a new exit-2
    class), memoized so creation is not re-attempted per file while the root
    stays clean; on a DIRTY root the same failure keeps the fail-closed
    exit 2, and dirt appearing mid-loop retries creation once more (a second
    failure then fail-closes). BOTH probes re-run per file, so residual dirt
    appearing mid-loop reverts later files to the root oracle (fail-closed);
    every scratch-mode pristine call passes the shadow uniformly, so
    in-package src dirt appearing mid-loop stays neutralized.
    """
    if not ctx.pristine_bucket:
        return
    files = sorted({n.file for n in ctx.pristine_bucket})
    if len(files) > args.max_pristine_files:
        raise _Indeterminate(
            f"systemic main breakage ({len(files)} red files > "
            f"--max-pristine-files {args.max_pristine_files}) — investigate / refresh first",
            warns=ctx.warns,
        )
    if not args.run_pristine:
        commands = "\n".join(f"  {_pristine_command(root, f)}" for f in files)
        raise _Indeterminate(
            f"{len(ctx.pristine_bucket)} failure(s) need a pristine-main check and "
            "--run-pristine was not given — indeterminate, never a silent strip. "
            f"Per-file pristine commands:\n{commands}",
            warns=ctx.warns,
        )
    scratch: _ScratchTree | None = None
    scratch_unavailable = False  # #1408 memo: clean-root scratch failure -> root oracle
    wt_cones = _work_root_sparse_cones(ctx.work_root)  # None => non-sparse (R-G' floor mode)
    floored = wt_cones is None  # R-G' (#2019): floor-profile scratch, asymmetric verdicts
    try:
        for test_file in files:
            try:
                # BOTH probes re-run per file (MF-4c freshness + mid-loop transitions).
                ctx.live_dirty_paths = dirty_code_paths(root)
                contaminating = scratch_contamination_probe(root)
            except subprocess.CalledProcessError as exc:
                raise _Indeterminate(
                    f"dirt probe failed at {root}: {exc}", warns=ctx.warns
                ) from exc
            residual = (
                list(contaminating)  # --no-src-shadow: the #1077 eligibility rule verbatim
                if args.no_src_shadow
                else residual_scratch_contamination(contaminating)
            )
            # #1408: the scratch is the DEFAULT pristine oracle whenever physically
            # eligible — clean or dirty root alike (#1077's dirty-only trigger removed).
            use_scratch = (
                not residual  # R-B' (#1251): in-package src/ dirt is shadow-neutralized;
                # only pyproject.toml / uv.lock / out-of-package src/ dirt still blocks
                # R-G' (#2019): a non-sparse work root arms ONLY on a dirty root — on
                # a CLEAN non-sparse root the root oracle is trustworthy AND strictly
                # more capable (full tree; both strip and NEW available).
                and (not floored or bool(ctx.live_dirty_paths))
                and (
                    test_file not in ctx.sel.GLOB_SCAN_TESTS
                    or test_file in FILE_ANCHORED_SCAN_TESTS
                )  # R-F' (#1337): __file__-anchored scanners scan their own (scratch) tree
                and not args.no_scratch_fallback
                # Clean-root degradation memo (#1408): after a clean-root scratch
                # failure the root oracle is trustworthy — no per-file re-creation
                # attempts; dirt appearing mid-loop retries creation (a second
                # failure then fail-closes via the dirty branch below).
                and not (scratch_unavailable and not ctx.live_dirty_paths)
            )
            if use_scratch and scratch is None:
                scratch = _create_scratch_or_degrade(
                    ctx,
                    root,
                    wt_cones if wt_cones is not None else [],  # R-G': [] => floor profile
                    contaminating,
                    args,
                    floored=floored,
                )
                if scratch is None:
                    # #1408 clean-root degradation fired — memoize (no per-file
                    # re-creation attempts while the root stays clean).
                    scratch_unavailable = True
                    use_scratch = False
            timeout_s = (
                args.pristine_timeout_s
                if args.pristine_timeout_s is not None
                else derive_pristine_timeout_s(ctx.sel, test_file)
            )
            try:
                main_failing = run_single_file_pristine(
                    test_file,
                    cwd=scratch.path if use_scratch else root,
                    timeout_s=timeout_s,
                    venv_root=root if use_scratch else None,
                    pythonpath=(
                        str(scratch.path / "src")
                        if use_scratch and not args.no_src_shadow
                        else None
                    ),
                )
            except PristineRunError as exc:
                raise _Indeterminate(f"{exc} — indeterminate", warns=ctx.warns) from exc
            ctx.pristine_files_run.append(test_file)
            for node in [n for n in ctx.pristine_bucket if n.file == test_file]:
                if node in main_failing:
                    if use_scratch and floored:
                        raise _Indeterminate(  # R-G' strip refusal (#2019)
                            f"node red at pristine HEAD on the FLOOR-profile scratch "
                            f"(pristine_oracle=scratch-worktree-floor, sparse_wt=False): "
                            f"the floor tree is not a superset of the non-sparse gate "
                            f"layout, so a 'pre-existing' strip for "
                            f"{node.file}::{node.name} cannot be certified (R-G'); "
                            "indeterminate — commit/clean the dirt, run from a sparse "
                            "worktree, or fix main first",
                            extra={
                                "live_dirty_paths": ctx.live_dirty_paths,
                                "contaminating_paths": contaminating,
                                "residual_contaminating_paths": residual,
                            },
                            warns=ctx.warns,
                        )
                    if not use_scratch and ctx.live_dirty_paths:
                        shadowable = [p for p in contaminating if p not in residual]
                        raise _Indeterminate(  # MF-4c, fail-closed residual
                            f"pristine oracle is DIRTY "
                            f"(residual contaminating: {residual[:20] or 'n/a'}; "
                            f"shadowable src dirt: {shadowable[:20] or 'n/a'}; "
                            f"visible code dirt: {ctx.live_dirty_paths[:20]}; "
                            f"scan_set={test_file in ctx.sel.GLOB_SCAN_TESTS}, "
                            f"file_anchored={test_file in FILE_ANCHORED_SCAN_TESTS}, "
                            f"sparse_wt={wt_cones is not None}) — a 'pre-existing' verdict "
                            f"for {node.file}::{node.name} from a dirty root is "
                            "untrustworthy (MF-4c); indeterminate "
                            "(scratch-by-default: this fires only for residual venv "
                            "dirt / non-anchored scan node / --no-scratch-fallback / "
                            "scratch failure on a dirty root; a bare non-sparse work "
                            "root now arms the R-G' floor scratch instead — its "
                            "red-at-pristine refusal is a separate exit-2 arm, #2019)",
                            extra={
                                "live_dirty_paths": ctx.live_dirty_paths,
                                "contaminating_paths": contaminating,
                                "residual_contaminating_paths": residual,
                            },
                            warns=ctx.warns,
                        )
                    _strip_node(ctx, node, via="pristine-scratch" if use_scratch else "pristine")
                else:
                    ctx.new.append(node)  # a PASS still classifies NEW (fail-closed)
    finally:
        if scratch is not None:
            remove_scratch_worktree(root, scratch)


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
        raise _Indeterminate(f"lint verdict failed: {exc}", warns=ctx.warns) from exc
    ledger = lv.ledger
    return {
        "pytest_rc": args.pytest_rc,
        "new": [n._asdict() for n in ctx.new],
        "stripped": ctx.stripped,
        "urgent_park_required": ctx.urgent_park,  # #1742 stripped workflow-invariant node ids
        "warns": ctx.warns,
        "stale": lv.stale,
        "stale_reasons": lv.stale_reasons,
        "ledger_dirty": lv.ledger_dirty,
        "ledger_dirty_paths": list(ledger.get("dirty_paths", [])) if ledger else [],
        "live_dirty_paths": ctx.live_dirty_paths,
        "pristine_files_run": ctx.pristine_files_run,
        "pristine_oracle": ctx.pristine_oracle,
        "scratch_sha": ctx.scratch_sha,
        "scratch_src_shadow": ctx.scratch_src_shadow,
        "scratch_degraded": ctx.scratch_degraded,  # #1408 clean-root degradation audit flag
        "gate_tmp_root": _gate_tmp_root_str(),  # #1408 temp-write routing provenance
        "lint": lint,
        "ledger_sha": ledger["main_sha"] if ledger else None,
        "ledger_age_h": ledger_age_hours(ledger) if ledger else None,
    }


def _indeterminate_payload(
    reason: str, pytest_rc: int, extra: dict | None = None, warns: list[str] | None = None
) -> dict:
    """The stable exit-2 ``--json`` shape: callers branch on ``indeterminate`` mechanically.

    ``warns`` carries the ctx-accumulated WARNs (e.g. SCRATCH-ORACLE provenance
    from an earlier bucketed file) so a mid-loop failure never drops audit
    provenance (#1077). The empty ``new``/``stripped`` arrays are NOT a clean
    verdict — the exit code stays 2.
    """
    return {
        "indeterminate": True,
        "reason": reason,
        "exit_code_intent": 2,
        "pytest_rc": pytest_rc,
        "new": [],
        "stripped": [],
        "urgent_park_required": [],  # #1742 stable shape on the exit-2 payload
        "warns": list(warns or []),
        **(extra or {}),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    """Classify a Step 9c run's failures as NEW vs pre-existing-on-main (plan §3.4).

    Under ``--json``, EVERY exit path prints exactly one JSON object (#1077):
    exit 0/1 the classification result (``indeterminate: false``); exit 2 the
    ``_indeterminate_payload`` (``indeterminate: true`` + ``reason``).
    """
    if args.pytest_rc not in (0, 1):
        reason = (
            f"pytest rc {args.pytest_rc} (aborted/interrupted/internal-error run) — "
            "refusing to classify a partial run (MF-1b)"
        )
        _log(reason)
        if args.json:
            print(
                json.dumps(_indeterminate_payload(reason, args.pytest_rc), indent=2, sort_keys=True)
            )
        return 2
    try:
        result = _compare_impl(args)
    except _Indeterminate as exc:
        _log(str(exc))
        if args.json:
            print(
                json.dumps(
                    _indeterminate_payload(str(exc), args.pytest_rc, exc.extra, warns=exc.warns),
                    indent=2,
                    sort_keys=True,
                )
            )
        return 2
    except ToolMissingError as exc:
        # #1408: a misconfigured explicit EPM_STEP9C_TMPDIR override (raised at any
        # gate_tmp_root() call site) is fail-loud — the standard exit-2 mapping.
        _log(str(exc))
        if args.json:
            print(
                json.dumps(
                    _indeterminate_payload(str(exc), args.pytest_rc), indent=2, sort_keys=True
                )
            )
        return 2
    result["indeterminate"] = False
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"compare: {len(result['new'])} NEW, {len(result['stripped'])} stripped, "
            f"{len(result['warns'])} warn(s), lint_ok={result['lint']['ok']}"
        )
        for n in result["new"]:
            print(f"  NEW: {n['file']}::{n['name']}")
        for uid in result["urgent_park_required"]:
            print(f"  URGENT-PARK-REQUIRED: {uid}")  # #1742 (stderr carries the full demand)
        for w in result["warns"]:
            print(f"  {w}")
    for w in result["warns"]:
        _log(w)
    return 1 if (result["new"] or not result["lint"]["ok"]) else 0


# --- tmproot ---------------------------------------------------------------------


def cmd_tmproot(args: argparse.Namespace) -> int:
    """Print the resolved gate temp-write root (empty stdout = no routing); ALWAYS exit 0.

    The SKILL Step 9c 1b/1c gate blocks consume this as the single source of
    routing truth (empty output -> no TMPDIR export, #1408). A misconfigured
    explicit override prints the ToolMissingError to stderr and still exits 0
    with empty stdout — the caller treats it as no-routing while the loud
    message surfaces in the gate log.
    """
    try:
        root = gate_tmp_root()
    except ToolMissingError as exc:
        _log(str(exc))
        return 0
    if root is not None:
        print(root)
    return 0


# --- probe -----------------------------------------------------------------------

# Fleet-arbitration signature union (#1962): the four gate artifact classes that
# ride Step 9c / Step 10d / Step 9a-ter gate-launch argvs, plus the ledger
# ``refresh`` invocation (the heaviest pytest universe; its own flock bounds it
# to one fleet-wide, so it counts as ONE gate under the reserved pseudo-issue
# key FLEET_REFRESH_KEY). FIXED and valid by construction, so ``probe --fleet``
# is until-loop-safe (exit 2 stays argparse/usage-only for the fleet form).
FLEET_GATE_SIGNATURE_RE: re.Pattern[str] = re.compile(
    r"step9c-junit-issue-(\d+)\.xml"
    r"|issue-(\d+)-lint-gate-tree"
    r"|issue-(\d+)-[^ ]*inline-payload\.txt"
    r"|issue-(\d+)-surgical-outcome\.txt"
    r"|step9c_baseline\.py refresh"
)
FLEET_REFRESH_KEY = "refresh"  # pseudo-issue key for the group-less refresh alternate
FLEET_MAX_ENV = "EPM_GATE_FLEET_MAX"
FLEET_MAX_DEFAULT = 2
FLEET_ARGV_SAMPLE_CHARS = 160  # summary-line argv truncation


def _fleet_max() -> int:
    """Resolve the fleet cap from ``EPM_GATE_FLEET_MAX`` (int >= 1; default 2).

    A malformed value (non-int / < 1) falls back to the default with a stderr
    note — NEVER a crash or exit 2: a wedged env var must not wedge gate
    launches, and the ``--fleet`` form must stay until-loop-safe.
    """
    raw = os.environ.get(FLEET_MAX_ENV)
    if raw is None or not raw.strip():
        return FLEET_MAX_DEFAULT
    try:
        val = int(raw.strip())
    except ValueError:
        _log(f"probe --fleet: malformed {FLEET_MAX_ENV}={raw!r}; using default {FLEET_MAX_DEFAULT}")
        return FLEET_MAX_DEFAULT
    if val < 1:
        _log(f"probe --fleet: {FLEET_MAX_ENV}={raw!r} < 1; using default {FLEET_MAX_DEFAULT}")
        return FLEET_MAX_DEFAULT
    return val


def _fleet_gate_issues(exclude_issue: int | None) -> dict[str, list[tuple[int, str]]]:
    """Group live FOREIGN gate processes by issue key (fleet arbitration, #1962).

    Reuses the ``_probe_matches`` self-/ancestor-pid-excluding ``/proc`` scan
    with the fixed ``FLEET_GATE_SIGNATURE_RE`` union — no second scanner. Per
    matched argv, collects ALL matched capture groups (``finditer``) so a
    wrapper argv referencing two issues' artifacts attributes to EVERY matched
    issue; the group-less refresh alternate maps to ``FLEET_REFRESH_KEY``.
    ``exclude_issue`` drops the caller's own issue key. Returns
    ``{issue_key: [(pid, argv), ...]}`` for the remaining foreign gate issues.
    """
    grouped: dict[str, list[tuple[int, str]]] = {}
    for pid, argv_text in _probe_matches(FLEET_GATE_SIGNATURE_RE):
        keys: set[str] = set()
        for m in FLEET_GATE_SIGNATURE_RE.finditer(argv_text):
            issue = next((g for g in m.groups() if g is not None), None)
            keys.add(issue if issue is not None else FLEET_REFRESH_KEY)
        if exclude_issue is not None:
            keys.discard(str(exclude_issue))
        for key in keys:
            grouped.setdefault(key, []).append((pid, argv_text))
    return grouped


def _cmd_probe_fleet(args: argparse.Namespace) -> int:
    """Fleet-level gate-concurrency probe (#1962).

    Prints one summary line per FOREIGN gate issue
    (``issue=<M>\\tpids=<k>\\t<sample argv>``, argv truncated to
    ``FLEET_ARGV_SAMPLE_CHARS``; the refresh pseudo-issue prints
    ``issue=refresh``), then exits 3 when the DISTINCT foreign-issue count
    reaches ``EPM_GATE_FLEET_MAX`` (default 2), else 0.
    """
    grouped = _fleet_gate_issues(args.exclude_issue)

    def _order(key: str) -> tuple[int, int]:
        return (1, 0) if key == FLEET_REFRESH_KEY else (0, int(key))

    for key in sorted(grouped, key=_order):
        rows = grouped[key]
        print(f"issue={key}\tpids={len(rows)}\t{rows[0][1][:FLEET_ARGV_SAMPLE_CHARS]}")
    return 3 if len(grouped) >= _fleet_max() else 0


def _ancestor_pids() -> set[int]:
    """Return this process's pid plus its full ancestor chain (PPid walk to pid 1).

    Walks ``/proc/<pid>/status`` ``PPid:`` rows from ``os.getpid()`` upward. ANY
    read/parse failure stops the walk — failing toward a SMALLER exclusion set
    (a missed exclusion surfaces as a loud false LIVE at the probe, never a
    silently skipped foreign match). Linux-only, like the probe itself.
    """
    pids: set[int] = set()
    pid = os.getpid()
    while pid >= 1 and pid not in pids:
        pids.add(pid)
        if pid == 1:
            break
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            break
        ppid: int | None = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split(":", 1)[1].strip())
                except ValueError:
                    ppid = None
                break
        if ppid is None or ppid < 1:
            break
        pid = ppid
    return pids


def _probe_matches(pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    """Scan ``/proc/*/cmdline`` for live FOREIGN processes matching ``pattern``.

    Excludes exactly the prober's own pid + full ancestor chain (AC-1 of plan
    #1821 — NO other exclusion classes: a concurrent foreign probe reads as a
    loud, self-resolving false LIVE, the fail-safe direction). Empty cmdlines
    (kernel threads / zombies) are skipped; ENOENT/permission races mid-scan
    are tolerated. Returns ``(pid, space-joined argv)`` per match, pid-sorted.
    """
    excluded = _ancestor_pids()
    matches: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue  # ENOENT race (process exited mid-scan) / permission
        if not raw:
            continue  # kernel thread or zombie: empty cmdline
        argv_text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if pattern.search(argv_text):
            matches.append((pid, argv_text))
    return sorted(matches)


def cmd_probe(args: argparse.Namespace) -> int:
    """Gate single-flight liveness probe with mechanical self-/ancestor-pid exclusion (#1821).

    Replaces the remembered probe-placement rule (a separate FOREGROUND
    ``pgrep`` call + bracket idiom) that #1742 re-hit: the bracket shields only
    the PATTERN text and cannot help when the enclosing call's argv carries the
    real unbracketed artifact path (the documented #1606 trap). Exit semantics
    are DELIBERATELY inverted vs pgrep (0 = clear) so ``probe && launch``
    composes naturally: 0 = CLEAR (no foreign match — safe to launch); 3 = >=1
    live foreign match (one ``pid<TAB>args`` line each on stdout); 2 = bad
    regex (argparse itself exits 2 on --pattern/--issue misuse). ``--issue N``
    derives ``step9c-junit-issue-<N>\\.xml`` internally so the probe's own argv
    never carries the junit filename; until-loops use the ``--issue`` form ONLY
    (fixed, valid regex — an exit-2 inside an until-loop would wait forever).
    ``--fleet [--exclude-issue N]`` (#1962) routes to ``_cmd_probe_fleet`` —
    its internal union regex is fixed and valid too, so the fleet form is
    equally until-loop-safe; ``--exclude-issue`` without ``--fleet`` is a
    usage error (exit 2, matching argparse semantics).
    """
    if args.exclude_issue is not None and not args.fleet:
        _log("probe: --exclude-issue is only meaningful with --fleet")
        return 2
    if args.fleet:
        return _cmd_probe_fleet(args)
    if args.issue is not None:
        pattern_text = rf"step9c-junit-issue-{args.issue}\.xml"
    else:
        pattern_text = args.pattern
    try:
        pattern = re.compile(pattern_text)
    except re.error as exc:
        _log(f"probe: bad regex {pattern_text!r}: {exc}")
        return 2
    matches = _probe_matches(pattern)
    for pid, argv_text in matches:
        print(f"{pid}\t{argv_text}")
    return 3 if matches else 0


# --- CLI -------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the refresh/status/compare CLI (shapes pinned in plan #1022 §3.2)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh", help="run the Step 9c universe on main; write ledger")
    p_refresh.add_argument("--repo-root", default=None, help="main-root override (tests)")
    # #1646: == recommended_timeout_s(WORKFLOW_INVARIANT) at current constants
    # (120 + 30*61 + 2400); re-derive when SLOW_TESTS / the invariant set change.
    p_refresh.add_argument("--timeout-s", type=float, default=4350.0)
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
    p_compare.add_argument(
        "--base",
        default=None,
        help=(
            "diff base (default: resolve via the worktree selector — fetched-"
            "origin/main semantics WITHOUT a second fetch, #1289; an explicit "
            "REF is used verbatim)"
        ),
    )
    p_compare.add_argument(
        "--worktree", default=None, help="work-root override (default: cwd toplevel)"
    )
    p_compare.add_argument("--repo-root", default=None)
    p_compare.add_argument("--run-pristine", action="store_true")
    p_compare.add_argument(
        "--pristine-timeout-s",
        type=float,
        default=None,
        help="per-file pristine-run bound; default: derived per file from the "
        "selector's gate-timeout knowledge (BASE + PER_FILE + 2x slow "
        "surcharge, floor 600 s; #1129). Explicitly passing it wins for every file.",
    )
    p_compare.add_argument("--max-pristine-files", type=int, default=5)
    p_compare.add_argument("--max-age-hours", type=float, default=24.0)
    p_compare.add_argument("--max-code-commits", type=int, default=150)
    p_compare.add_argument(
        "--scratch-timeout-s",
        type=float,
        default=120.0,
        help="per-git-command bound for the dirty-oracle scratch-worktree fallback (#1077)",
    )
    p_compare.add_argument(
        "--no-scratch-fallback",
        action="store_true",
        help="disable the scratch oracle (the default pristine oracle, #1408) — root-oracle "
        "runs only, with the pre-#1077 MF-4c raise on ANY dirty oracle",
    )
    p_compare.add_argument(
        "--no-src-shadow",
        action="store_true",
        help="disable the #1251 scratch PYTHONPATH src-shadow — restore the #1077 eligibility "
        "rule (ANY dirty src//pyproject.toml/uv.lock path keeps the fail-closed exit 2)",
    )
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    p_tmproot = sub.add_parser(
        "tmproot",
        help="print the resolved gate temp-write root (empty = no routing); always exit 0",
    )
    p_tmproot.set_defaults(func=cmd_tmproot)

    p_probe = sub.add_parser(
        "probe",
        help="single-flight liveness probe (self-/ancestor-pid excluding): "
        "0 = clear / 3 = live foreign match / 2 = bad regex",
    )
    probe_target = p_probe.add_mutually_exclusive_group(required=True)
    probe_target.add_argument(
        "--pattern",
        help="extended regex matched (re.search) against space-joined /proc/<pid>/cmdline",
    )
    probe_target.add_argument(
        "--issue",
        type=int,
        help=r"derive the Step 9c gate pattern step9c-junit-issue-<N>\.xml internally "
        "(the probe's own argv never carries the junit filename)",
    )
    probe_target.add_argument(
        "--fleet",
        action="store_true",
        help="fleet arbitration (#1962): count DISTINCT foreign issues with live gate "
        "trees (fixed internal signature union incl. the ledger-refresh pseudo-issue); "
        "exit 3 when count >= EPM_GATE_FLEET_MAX (default 2), else 0",
    )
    p_probe.add_argument(
        "--exclude-issue",
        type=int,
        default=None,
        help="(--fleet only) drop the caller's own issue from the foreign-gate count; "
        "usage error (exit 2) without --fleet",
    )
    p_probe.set_defaults(func=cmd_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns the subcommand's exit code."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
