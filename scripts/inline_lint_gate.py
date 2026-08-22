#!/usr/bin/env python3
"""Mechanized Step 9a-ter inline payload lint gate + certification writer (#1500).

Runs the two legs of the ``/issue`` Step 9a-ter § Inline payload lint gate
(no-flags ``scripts/workflow_lint.py`` + ``scripts/select_step9c_tests.py
--map-files`` -> mapped pytest), applies the gate's payload-attributed verdict
semantics (#1460), and on PASS appends a content-hash-bound certification line
per payload path (``v1 <epoch> <blobsha> <path>``) that
``.claude/hooks/guard_root_code_commit.sh`` validates before allowing a
repo-root commit of gated code paths. NEVER hand-write the cert file (#1082
parity — the guard family targets forgetting, not adversaries).

Verdict semantics (mechanizes SKILL.md Step 9a-ter § Inline payload lint gate):

- INCONCLUSIVE (exit 3, NO cert): the instrument did not run to completion —
  empty/missing payload file, lint leg missing its healthy terminal line
  (``workflow_lint: PASS`` / ``workflow_lint: FAIL (``; the ``schema FAIL``
  early-exit is deliberately rejected — it prints BEFORE any check executes),
  non-empty test mapping with no pytest summary line, or a payload path edited
  DURING the gate run (TOCTOU — the cert must bind the exact gated content).
  ALSO exit 3 (#2039, instrument RAN but the verdict is withheld): a would-be
  BLOCK whose payload-naming hits are ALL pytest-leg while VM load1 >=
  ``EPM_GATE_LOAD_MAX`` is downgraded to the distinct, self-diagnosing
  ``pytest-leg red under load`` reason (see Load awareness below) — re-run
  when load drops; never a false BLOCK from a loaded fleet.
- BLOCK (exit 1): a non-WARN output line names a payload path that is (i) NEW
  on ``origin/main`` (payload-caused by construction — every #1388/#1428
  incident offender was this case), or (ii) MODIFIED with a parseable
  ``<path>:<lineno>:`` hit whose lineno falls inside the round's added lines
  (``git diff -U0 origin/main -- <path>``), or (iii) MODIFIED with a
  payload-naming hit carrying no parseable lineno (conservative block — the
  prose gate's "pre-existing red never blocks" judgment call routes through
  ``EPM_ALLOW_ROOT_CODE_COMMIT=1`` + an ``epm:progress`` note instead).
  EXCEPTION (#1585; the #1112 false-block incident): pytest warnings-summary
  ATTRIBUTION rows — bare node-id headers (``path::test``) and aggregated
  ``path: N warnings`` rows inside pytest's equals-fenced "warnings summary"
  section, pytest leg ONLY — are classified as REPORT lines BEFORE per-path
  hit assignment, so they never block on either the NEW or the MODIFIED
  branch. SECOND EXCEPTION (#2023; the #1345 v242 false-block incident):
  EVERY pytest-leg line inside pytest's ``-rA`` equals-fenced ``PASSES``
  section — captured output of tests pytest reports as PASSED,
  definitionally not red evidence (e.g. a designed-crash test's captured
  traceback naming a payload path) — is classified as a REPORT line
  (``[passing-capture]``) BEFORE per-path hit assignment, on the NEW and
  MODIFIED branches alike and INCLUDING lineno-bearing hits (single
  predicate: index inside the PASSES fenced window; any other fence title
  CLOSES it — unlike #1585's in-section AND row-shape double predicate).
  Accepted residual: a column-0 fence-shaped line embedded in a FAILING
  test's captured output could re-open/close windows — outside the guard
  family's anti-forgetting threat model (#1082 parity); the fail direction
  on nested fences INSIDE the PASSES section is CLOSED (a nested fence
  closes the window — another false block, never a false pass). Every
  other lineno-less naming hit keeps the conservative block.
- PASS (exit 0): repo-wide red naming only non-payload paths, WARN lines,
  warnings-summary attribution rows + PASSES-section captured lines
  (above), and modified-file hits whose
  linenos all sit outside the round's added lines never block (they are
  REPORTED for the round's ``epm:progress`` note). Per-path certs mean a
  mixed verdict still certifies the clean subset.

Untracked-payload visibility (#1889): the mapped-pytest leg runs with
``EPM_SCAN_EXTRA_FILES=<os.pathsep-joined payload paths>`` in its CHILD env
(never the gate's own process env) so tracked-file-enumerating scan tests
(``tests/test_shared_vm_thread_caps.py``'s ``_scan_targets``) union brand-new,
still-UNTRACKED payload files into their scan set. Without the seam a new
file's invariant violation is invisible to the gate (the scan enumerates
``git ls-files`` only), passes 9a-ter, and lands red on trunk for every
intervening session — the #1388 fleet-red class (realized 2026-07-30 at
``606278aa38`` on #1739, and again at ``04e111a7ad``). Untracked payload
paths additionally get a stderr audit NOTE (report-only, never a verdict
input).

Bytecode determinism (#1950): before any leg runs, the gate best-effort
purges ``*.pyc`` under the ``__pycache__`` dirs of the editable code roots
(``scripts/``, ``src/``, ``tests/``) of the resolved repo — pyc validation
compares recorded source (mtime, size), and mtime has 1-second granularity,
so an Edit -> ruff-format-hook rewrite -> run cycle landing inside one
second leaves a stale pyc that still validates against the NEWER source; the
gate's plain ``uv run`` children then import the OLD module while direct
developer runs recompile (three #1345 gate runs bounced on exactly this,
2026-07-31). All three legs additionally run with
``PYTHONDONTWRITEBYTECODE=1`` in the CHILD env (via ``extra_env`` — the
gate's own process env is never mutated) so the gate's children never
repopulate in-tree caches mid-run. Residual: ``PYTHONDONTWRITEBYTECODE=1``
binds only the gate's children — a concurrent session can rewrite an in-tree
pyc mid-gate (it compiles from then-current source, so only a second
same-second rewrite re-creates staleness). Purge audit lines print to the
GATE's stderr only — never into the leg-captured
``issue-<N>-inline-{lint,map}.txt`` audit files.

Load awareness (#2039): the shared VM's load can make the mapped-pytest leg
red purely by timing (incident: a certified gate run FAILed exit 3 under VM
load1~31 — 1 failed + 3 error mapped tests — and PASSed clean ~35 min later
with the payload unchanged). Knobs: ``EPM_GATE_LOAD_MAX`` (float threshold,
default 20.0 for this 32-core VM; ``0``/negative DISABLES the guard — the
one-line kill switch restoring pre-#2039 behavior exactly; malformed ->
default, the guard stays active), ``EPM_GATE_LOAD_WAIT_S`` (bounded
pre-pytest-leg wait for load1 to drop below the threshold, default 300 s at a
15 s poll; ``0`` -> no wait; malformed -> default), and
``EPM_GATE_LOAD1_OVERRIDE`` (test support ONLY: overrides the
``os.getloadavg()[0]`` read; malformed -> ignored). Semantics: load1 is
sampled immediately before (post-wait) and immediately after the pytest leg
(``load_hot`` = max available endpoint sample >= threshold; no samples /
disabled -> never hot), plus ONE diagnostic-only gate-start sample that is
NEVER fed into the verdict — load before the wait is not load during the
leg. Under ``load_hot``, a payload path that would otherwise BLOCK on hits
that are ALL pytest-leg is downgraded to a DISTINCT INCONCLUSIVE (exit 3,
no cert, riding the existing ``inline_lint_gate: INCONCLUSIVE (`` shape with
reason ``pytest-leg red under load``). The fail direction is one-way by
construction: load cannot make a failing test pass, so a PASS under load
still certifies; any lint-leg hit keeps the BLOCK (lint findings are
deterministic, not timing-sensitive); a would-be PASS is never touched.
Known residual: 1-min-load endpoint sampling can miss a mid-leg spike on an
8-9 min pytest leg — the miss direction is today's behavior (BLOCK), i.e.
safe; do not over-trust ``load_hot=False``.

Baseline-ledger layer (#2235 Phase A; violation grain #2318):
``load_baseline_ledger`` reads the Step 9c ledger
(``.claude/cache/step9c-baseline.json``, read-only) and, when it parses +
schema-matches + sha-matches origin/main, ``evaluate`` reclassifies pytest-leg
red the ledger lists as ALREADY failing on main into ``reported``, at two
grains: (a) the short-summary ``FAILED <nodeid>`` row of any ledger-listed
node (``pre-existing-on-main (ledger)``, node grain); (b) for a ledger-listed
``step9c_baseline.VIOLATION_SET_SCAN_NODES`` member (whole-repo scan tests —
registry + ``extract_violation_paths`` imported LAZILY from the sibling
module, single source, never a drift copy), the node's whole FAILURES block
(``pre-existing-on-main (ledger, violation-set)``) — but ONLY when the
extracted violation-path set of that block contains NO payload path: a
payload path in the extracted set means the round ADDED a violation to the
already-red node, and the block keeps today's classification (#2318, the
#2316 compare-side mirror). The ledger's live ``failing_tests`` rows are
DICTS ({classname, file, name}) mapped to ``file::name`` nodeids by
``_ledger_nodeids`` (the pre-#2318 ``str()`` coercion made the membership
test always False — the layer was dead code, #2318 D1). Both grains skip a
node whose test FILE is itself a payload path (own-file condition).
Subtractive only (block -> reported, never the reverse); any bad/stale ledger
degrades to None = exactly the pre-#2235 semantics; a registry-import
failure, an unparseable / summary-mismatched FAILURES block, or an empty
extraction degrades LOUDLY (one stderr warn) to node-grain behavior — never a
crash, never a new blocking class. The gate prints ``inline_lint_gate:
ledger=<fresh sha12|stale|absent>`` so the audit trail shows the layer's
armed/disarmed state.

Documented residuals (#2318 — recorded, not redesigned): (a) the ledger is
consumed sha-pinned but WITHOUT a ``dirty_code_paths`` refusal — refusing a
``dirty_code_paths: true`` ledger would disable the subtraction whenever the
fleet has dirty paths (nearly always) and thereby newly BLOCK payloads on
main's own pre-existing reds, the #1388 fleet-wedge class; (b) pytest's
saferepr elision / explanation truncation can in principle hide a payload's
offender row while leaving only an elision-dropped token, yielding a demote —
reachability ~nil (a list long enough to elide is long enough to truncate the
``assert not`` line entirely, and truncation blinds today's gate identically).

Run as ONE background Bash (the SCOPED lint leg is fast, but the
workflow-surface / refusal-fallback bare no-flags run measured ~9-10 min on
2026-08-11 and grows with the check roster; never a <=600 s foreground
bound — #991/#996)::

    uv run python scripts/inline_lint_gate.py --issue <N> \\
        --payload-file /tmp/issue-<N>-<round-slug>-inline-payload.txt

The payload path must be ROUND-unique (#1948): the bare issue-keyed legacy
name (basename ``issue-<N>-inline-payload.txt``) is REFUSED (exit 3) — it is
a shared mutable path that concurrent same-issue rounds clobber, producing
cross-certification (two concurrent #1768 rounds, 2026-07-31). The gate also
reads the caller's payload file exactly ONCE and hands the map leg a private
``mkstemp`` copy, and prints a payload-binding audit line
(``inline_lint_gate: payload-source <path|inline-paths> n=<k>
list-sha256=<12 hex>``) before the verdict line.

Test-only env overrides (hermetic unit tests substitute the leg commands;
same pattern as ``EPM_LESSONS_EDIT_SENTINEL``): ``EPM_INLINE_GATE_LINT_CMD``,
``EPM_INLINE_GATE_MAP_CMD``, ``EPM_INLINE_GATE_PYTEST_CMD`` (each a shell
string run with the repo root as cwd). ``EPM_INLINE_CERT_PATH`` overrides the
cert file (shared with the hook).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

DEFAULT_CERT_PATH = "/tmp/eps-inline-lint-cert-v1.txt"
# Step 9c baseline ledger (#2235 Phase A): read-only consumption of the SAME
# what's-already-red-on-main ledger step9c_baseline.py maintains (direct JSON
# read — no cross-script import; the gate never writes it). A ledger that is
# absent / corrupt / schema-drifted / sha-mismatched against origin/main
# degrades to None, which disengages the subtraction layer entirely —
# evaluate() then behaves bit-identically to the pre-#2235 gate (fail-closed
# to current semantics: a bad ledger NEVER blocks and NEVER passes anything).
LEDGER_PATH_REL = ".claude/cache/step9c-baseline.json"
LEDGER_SCHEMA_VERSIONS = (1,)  # jq-verified live value, 2026-08-11; drift -> None
# Bare issue-keyed legacy payload BASENAME — a shared mutable path concurrent
# same-issue rounds clobber (cross-certification, #1948/#1768). Refused at
# main() entry (exit 3, Inconclusive) BEFORE any leg runs. The gate's own
# private mkstemp copy (``issue-<N>-inline-payload.<rand>``) is dot-suffixed
# and deliberately does NOT match this ``\.txt$``-anchored regex.
LEGACY_PAYLOAD_BASENAME_RE = re.compile(r"^issue-\d+-inline-payload\.txt$")
CERT_TRIM_LINES = 500
# Env var threaded onto the mapped-pytest leg's CHILD env carrying the payload
# path list (os.pathsep-separated, repo-relative) so tracked-file-enumerating
# scan tests (tests/test_shared_vm_thread_caps.py::_scan_targets) union
# brand-new UNTRACKED payload files into their target set (#1889).
SCAN_EXTRA_FILES_ENV = "EPM_SCAN_EXTRA_FILES"
# Child-leg env guard (#1950): the pre-leg purge removes stale repo-tree
# bytecode; this stops the gate's own children re-writing it mid-run (merged
# over os.environ via extra_env on every leg — never the gate's process env).
NO_BYTECODE_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}
# Code roots whose __pycache__ dirs the gate purges (editable roots only —
# NEVER .venv/external/data: installed-package bytecode is not the
# same-second-rewrite staleness source and is expensive to recompile).
PURGE_ROOTS = ("scripts", "src", "tests")
# #2235 Phase B/C: the bare no-flags wall measured 547.9 s on 2026-08-11 (and
# grows with the check roster), already past half the old 900 s fence — the
# fence sits >= 2x the measured wall (1200 = 2.19x). The SCOPED leg is far
# under it; the fence exists for the workflow-surface / refusal-fallback bare
# runs.
LINT_TIMEOUT_S = 1200
FETCH_TIMEOUT_S = 60
# Files-mode eligibility (#2235 Phase B): a payload touching ANY of these
# workflow-surface prefixes (or exact root files) can change GLOBAL check
# outcomes — the lint leg then runs the bare (unscoped) no-flags form,
# today's exact behavior. `tests/` is included conservatively (a payload
# editing tests can change global check outcomes and mapped-test verdicts).
# Matched via str.startswith, so the two exact-file entries also cover any
# hypothetical suffixed sibling — the conservative (full-run) direction.
WORKFLOW_SURFACE_PREFIXES = (
    ".claude/",
    "workflow.yaml",
    "scripts/workflow_lint.py",
    "scripts/inline_lint_gate.py",
    "scripts/select_step9c_tests.py",
    "scripts/step9c_baseline.py",
    "tests/",
)
# workflow_lint's fail-closed files-mode registry-miss sentinel: exactly ONE
# bare full re-run (slow-but-correct), never a silent skip.
FILES_MODE_REFUSED_RE = re.compile(r"^workflow_lint: FILES-MODE-REFUSED \(", re.MULTILINE)
# Mapped-pytest timeout parity with select_step9c_tests.recommended_timeout_s
# (#1046): base + per-file + the test_workflow_lint.py slow surcharge, floored
# at the canonical select_step9c_tests.TIMEOUT_FLOOR_S (round-2 Minor: the
# floor-less formula gave 1 mapped non-slow test 150 s vs the canonical 900 s
# — a false-INCONCLUSIVE generator on slow machines, never a false pass).
PYTEST_BASE_S = 120
PYTEST_PER_FILE_S = 30
PYTEST_WORKFLOW_LINT_SURCHARGE_S = 2400  # select_step9c_tests.SLOW_TESTS parity (#1646)
PYTEST_TIMEOUT_FLOOR_S = 900  # select_step9c_tests.TIMEOUT_FLOOR_S parity (pinned by test)
# Load awareness (#2039; module docstring § Load awareness): threshold +
# bounded pre-pytest-leg wait. GATE_LOAD_MAX_DEFAULT is sized for this
# 32-core shared VM (incident load1~31); EPM_GATE_LOAD_MAX=0 is the one-line
# kill switch restoring pre-#2039 behavior exactly.
GATE_LOAD_MAX_DEFAULT = 20.0
GATE_LOAD_WAIT_DEFAULT_S = 300.0
GATE_LOAD_POLL_S = 15.0

# Healthy lint terminal line; `workflow_lint: schema FAIL` does NOT match.
LINT_TERMINAL_RE = re.compile(r"^workflow_lint: (PASS|FAIL \()", re.MULTILINE)
PYTEST_SUMMARY_RE = re.compile(r"[0-9]+ (passed|failed|error|xpassed|xfailed)|no tests ran")
# Attribution lines that are definitionally not red (pytest -rA summary rows).
NON_RED_PREFIXES = ("WARN", "PASSED", "SKIPPED")

# pytest warnings-summary section tracking (#1585; the #1112 false-block
# incident). Sections are equals-fenced title lines even under `-q` (probed
# on pytest 9.0.2); `\b[^=]*` tolerates trailing qualifiers ("(final)").
SECTION_FENCE_RE = re.compile(r"^=+ .+ =+$")
WARNINGS_SUMMARY_TITLE_RE = re.compile(r"^=+ warnings summary\b[^=]*=+$", re.IGNORECASE)
# Attribution rows inside the warnings summary: a bare node id (space-free,
# `::`-joined — covers class-based + parametrized ids) or pytest's aggregated
# `<path>: N warnings` row. FAILED/ERROR rows carry spaces + tokens => never match.
WS_ATTRIBUTION_ROW_RE = re.compile(r"^(?:\S+(?:::\S+)+|\S+: \d+ warnings?)$")
# pytest -rA `PASSES` section title (#2023; the #1345 v242 false-block
# incident). pytest emits the title UPPERCASE — matched case-SENSITIVELY so a
# lowercase "passes"-titled fence in captured output never opens the
# report-class window (fail-closed narrowing; the choice is test-pinned).
PASSES_TITLE_RE = re.compile(r"^=+ PASSES\b[^=]*=+$")
# pytest short-summary FAILED row (#2235 Phase A): the node-id extraction the
# ledger subtraction keys on. Applied to STRIPPED pytest-leg lines only; a hit
# line not of this shape keeps today's classification (never a blanket waiver).
FAILED_NODE_RE = re.compile(r"^FAILED\s+(\S+)")
# #2318: pytest FAILURES-section parsing for the violation-grain ledger
# demotion. The FAILURES title is matched case-SENSITIVELY (fail-closed
# narrowing, the #2023 PASSES precedent); per-failure headers are
# underscore-fenced and carry only the test NAME; the block's trailing
# `<file>.py:<line>: <Exc>` locus carries the failing test's FILE.
FAILURES_TITLE_RE = re.compile(r"^=+ FAILURES\b[^=]*=+$")
FAILURE_HEADER_RE = re.compile(r"^_+ (.+?) _+$")
FAILURE_LOCUS_RE = re.compile(r"^(\S+?\.py):\d+: \S+")
# #2318: the sibling registry module (VIOLATION_SET_SCAN_NODES +
# extract_violation_paths, both #2316), imported LAZILY inside the demotion
# path — a module-top import failure would crash the gate for every inline
# round fleet-wide. The LEDGER read stays deliberately import-free
# (load_baseline_ledger docstring); this import carries only the
# violation-grain registry + extractor (single source — a drift copy is
# forbidden, #2318 criterion 2). Result (module or None) cached per process.
_STEP9C_BASELINE_PATH = Path(__file__).resolve().parent / "step9c_baseline.py"
_STEP9C_CACHE: dict[str, object] = {}


class Inconclusive(Exception):
    """Instrument-ran completeness failure: no verdict, no cert (exit 3)."""


@dataclass
class LegResults:
    """Raw outputs of the two gate legs (pytest output kept separate so the
    lint leg's own terminal line can never satisfy the pytest summary check —
    the SKILL.md double-failure masking hazard). ``load1_pre``/``load1_post``
    are the #2039 pytest-leg endpoint load samples (None = leg never ran /
    sampling unavailable / guard disabled — keyword defaults keep every
    existing construction site valid)."""

    lint_output: str
    map_pairs: list[tuple[str, str]]
    pytest_output: str = ""
    load1_pre: float | None = None
    load1_post: float | None = None


@dataclass
class Verdict:
    """Per-path gate verdict + the non-blocking report lines.

    ``load_deferred`` (#2039): would-be-BLOCKED paths whose block reasons
    rest on hits that are ALL pytest-leg under hot load — a DISTINCT
    INCONCLUSIVE population (exit 3): never certified, never blocked."""

    blocked: dict[str, list[str]] = field(default_factory=dict)
    passing: list[str] = field(default_factory=list)
    reported: list[str] = field(default_factory=list)
    load_deferred: dict[str, list[str]] = field(default_factory=dict)


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_toplevel() -> Path:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise Inconclusive("not inside a git checkout and no --repo-root given")
    return Path(r.stdout.strip())


def _hash_object(repo: Path, path: str) -> str | None:
    """Blob sha of the WORKTREE content of *path* (None when unhashable)."""
    abs_path = repo / path
    if not abs_path.is_file():
        return None
    r = _git(repo, "hash-object", "--", str(abs_path))
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def read_payload(paths: list[str], repo: Path) -> dict[str, str]:
    """Blank-line-stripped payload -> {path: snapshot blob sha} (TOCTOU guard).

    The snapshot is taken NOW, before the multi-minute legs run; write_cert
    re-hashes and refuses to certify a path edited in between. Empty payload
    and missing/unhashable paths are INCONCLUSIVE (never a silent pass).
    """
    cleaned = [p.strip() for p in paths if p.strip()]
    if not cleaned:
        raise Inconclusive("payload-file-empty")
    snapshots: dict[str, str] = {}
    for p in cleaned:
        sha = _hash_object(repo, p)
        if sha is None:
            raise Inconclusive(f"payload path missing or unhashable: {p}")
        snapshots[p] = sha
    return snapshots


def load_baseline_ledger(repo: Path) -> dict | None:
    """Load the Step 9c baseline ledger for the subtractive Phase A layer (#2235).

    Direct JSON read of ``LEDGER_PATH_REL`` (no cross-script import of
    step9c_baseline — looser coupling; the gate never writes the ledger).
    Returns the parsed dict ONLY when ALL of: the file parses, its
    ``schema_version`` is in ``LEDGER_SCHEMA_VERSIONS``, ``failing_tests`` is
    a list, AND ``main_sha`` equals the CURRENT ``rev-parse origin/main`` of
    *repo* (called after run_legs' _bounded_fetch, so the pin is against the
    same origin/main the -U0 diffs classify against). Everything else — file
    absent, unreadable, corrupt JSON, schema drift, sha mismatch — returns
    None: the subtraction layer disengages and evaluate() behaves
    bit-identically to the pre-#2235 gate (fail-closed; a bad ledger NEVER
    blocks and NEVER passes anything)."""
    path = repo / LEDGER_PATH_REL
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") not in LEDGER_SCHEMA_VERSIONS:
        return None
    if not isinstance(data.get("failing_tests"), list):
        return None
    r = _git(repo, "rev-parse", "origin/main")
    if r.returncode != 0:
        return None
    if str(data.get("main_sha", "")).strip() != r.stdout.strip():
        return None
    return data


def _sample_load1() -> float | None:
    """1-minute load average feeding the #2039 load guard.

    ``EPM_GATE_LOAD1_OVERRIDE`` (test support, mirroring the documented
    ``EPM_INLINE_GATE_*_CMD`` override pattern) wins when parseable; a
    malformed override is ignored (noted on stderr, fall through to the real
    read). ``os.getloadavg()`` missing/failing (platform without loadavg) ->
    None: the guard is inert and the gate behaves exactly as pre-#2039."""
    override = os.environ.get("EPM_GATE_LOAD1_OVERRIDE")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            print(
                f"inline_lint_gate: note: malformed EPM_GATE_LOAD1_OVERRIDE={override!r} "
                "ignored — using the real loadavg read",
                file=sys.stderr,
            )
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def _load_max() -> float | None:
    """``EPM_GATE_LOAD_MAX`` threshold; None = guard disabled.

    Default ``GATE_LOAD_MAX_DEFAULT`` (20.0 — this 32-core VM; incident
    load1~31). Explicit ``0``/negative -> disabled (the one-line kill switch
    restoring pre-#2039 behavior exactly). Malformed -> default — the guard
    stays ACTIVE (the ``EPM_CERT_REHASH_DELAY_S`` precedent: fail toward
    guarded)."""
    raw = os.environ.get("EPM_GATE_LOAD_MAX")
    if raw is None or not raw.strip():
        return GATE_LOAD_MAX_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        return GATE_LOAD_MAX_DEFAULT
    return None if value <= 0 else value


def _load_wait_s() -> float:
    """``EPM_GATE_LOAD_WAIT_S`` pre-pytest-leg wait budget in seconds.

    Default ``GATE_LOAD_WAIT_DEFAULT_S`` (300); ``0`` -> no wait; malformed
    -> default; negative clamped to 0."""
    raw = os.environ.get("EPM_GATE_LOAD_WAIT_S")
    if raw is None or not raw.strip():
        return GATE_LOAD_WAIT_DEFAULT_S
    try:
        value = float(raw)
    except ValueError:
        return GATE_LOAD_WAIT_DEFAULT_S
    return max(value, 0.0)


def _fmt_load(value: float | None) -> str:
    """Audit-line rendering of a possibly-unavailable load sample."""
    return "unavailable" if value is None else f"{value:.2f}"


def _wait_for_load_drop(threshold: float) -> None:
    """Bounded pre-pytest-leg wait (#2039 D3): poll load1 every
    ``GATE_LOAD_POLL_S`` up to ``EPM_GATE_LOAD_WAIT_S`` until it drops below
    *threshold*, then proceed REGARDLESS of the final sample — a PASS under
    load still certifies, and refusing to run would starve quick green
    payloads. Honest sizing note: the incident's load persisted ~35 min, so
    this wait mostly harvests short spikes; the evaluate()-side downgrade is
    the load-bearing mechanism."""
    load1 = _sample_load1()
    if load1 is None or load1 < threshold:
        return
    budget = _load_wait_s()
    if budget <= 0:
        return
    print(
        f"inline_lint_gate: load-wait load1={load1:.2f} >= "
        f"EPM_GATE_LOAD_MAX={threshold:g}, waiting up to {budget:g}s for load to drop",
        file=sys.stderr,
    )
    waited = 0.0
    while waited < budget:
        step = min(GATE_LOAD_POLL_S, budget - waited)
        time.sleep(step)
        waited += step
        load1 = _sample_load1()
        if load1 is None or load1 < threshold:
            break
    print(
        f"inline_lint_gate: load-wait resumed after {waited:g}s (load1={_fmt_load(load1)})",
        file=sys.stderr,
    )


def _load_hot(legs: LegResults) -> bool:
    """True when the max AVAILABLE pytest-leg endpoint sample (pre/post) sits
    at/above the enabled threshold (#2039 D4). Disabled threshold / no
    samples -> False (never hot). The gate-start sample is diagnostic-only BY
    DESIGN — load before the wait is not load during the leg."""
    threshold = _load_max()
    if threshold is None:
        return False
    samples = [s for s in (legs.load1_pre, legs.load1_post) if s is not None]
    return bool(samples) and max(samples) >= threshold


def _best_effort_choom() -> None:
    """Deprioritize this gate run for earlyoom (Step 10d precedent
    #1045/#1211/#1143). Fail-open BY DESIGN: the gate must run even where
    sudo/choom is unavailable — the skip is reported, never silent."""
    try:
        subprocess.run(
            ["sudo", "-n", "choom", "-n", "-600", "-p", str(os.getpid())],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # best-effort: never let deprioritization kill the gate
        print(f"inline_lint_gate: note: self-choom skipped ({exc})", file=sys.stderr)


def purge_repo_bytecode(repo: Path) -> int:
    """Best-effort unlink of ``*.pyc`` under the editable code roots'
    ``__pycache__`` dirs (stale-mtime-matched pyc determinism guard, #1950).

    Returns the number of files removed; an unremovable file WARNs on stderr
    and never crashes the gate. Audit lines print to the GATE's stderr only —
    the leg-captured audit files hold leg subprocess output exclusively."""
    removed, errors = 0, 0
    for root in PURGE_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for pyc in base.rglob("__pycache__/*.pyc"):
            try:
                pyc.unlink(missing_ok=True)
                removed += 1
            except OSError:
                errors += 1
    if errors:
        print(
            f"inline_lint_gate: warn: {errors} bytecode cache file(s) "
            "could not be removed — gate proceeds (purge is best-effort)",
            file=sys.stderr,
        )
    print(
        f"inline_lint_gate: purged {removed} stale-candidate bytecode "
        f"cache file(s) under {'/'.join(PURGE_ROOTS)} __pycache__ (#1950)",
        file=sys.stderr,
    )
    return removed


def _bounded_fetch(repo: Path) -> None:
    """Best-effort `git fetch origin main` so new-vs-modified classification is
    current. Degrade-to-stale is safe: staleness only shifts classification in
    the STRICTER direction (an unfetched just-landed file reads as NEW)."""
    try:
        _git(repo, "fetch", "origin", "main", timeout=FETCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(
            "inline_lint_gate: note: git fetch timed out — classifying against "
            "the last-fetched origin/main (stricter direction)",
            file=sys.stderr,
        )


def _run_leg(
    default_argv: list[str],
    override_env: str,
    repo: Path,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Run one leg (env-override shell string, or the default argv), returning
    (combined stdout+stderr, returncode). A timeout returns the partial output
    with rc -1 — the missing terminal/summary line then reads INCONCLUSIVE.

    ``extra_env`` (e.g. the #1889 ``SCAN_EXTRA_FILES_ENV`` payload threading)
    is merged over ``os.environ`` into the CHILD env on BOTH branches — the
    override branch included, so hermetic tests can observe it — and never
    mutates the gate's own process env."""
    override = os.environ.get(override_env)
    env = {**os.environ, **extra_env} if extra_env else None
    try:
        if override:
            r = subprocess.run(
                override,
                shell=True,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        else:
            r = subprocess.run(
                default_argv,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
    except subprocess.TimeoutExpired as exc:
        out = (
            (exc.stdout or b"").decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        err = (
            (exc.stderr or b"").decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return out + "\n" + err + f"\n[inline_lint_gate: leg timed out after {timeout}s]", -1
    return r.stdout + "\n" + r.stderr, r.returncode


def parse_map_pairs(map_output: str) -> list[tuple[str, str]]:
    """Pair-generic TSV parse of `--map-files` stdout: keep the first two
    tab-separated fields of any line carrying >=2 (additive future columns —
    e.g. #1496 — must not break this parser)."""
    pairs: list[tuple[str, str]] = []
    for line in map_output.splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2 and fields[0].strip():
            pairs.append((fields[0].strip(), fields[1].strip()))
    return pairs


def mapped_pytest_timeout(tests: list[str]) -> int:
    """select_step9c_tests.recommended_timeout_s parity (#1046): base +
    per-file + the test_workflow_lint.py slow surcharge, floored at the
    canonical 900 s TIMEOUT_FLOOR_S (round-2 Minor)."""
    timeout = PYTEST_BASE_S + PYTEST_PER_FILE_S * len(tests)
    if "tests/test_workflow_lint.py" in tests:
        timeout += PYTEST_WORKFLOW_LINT_SURCHARGE_S
    return max(timeout, PYTEST_TIMEOUT_FLOOR_S)


def run_legs(
    payload_file: Path,
    issue: int,
    repo: Path,
    out_dir: Path,
    payload: list[str] | None = None,
) -> LegResults:
    """Run lint + mapped-pytest legs; persist audit outputs (parity with the
    pre-#1500 fenced recipe's /tmp/issue-<N>-inline-{lint,map}.txt files).

    ``payload`` is the repo-relative payload path list (defaults to re-reading
    ``payload_file``): threaded onto the mapped-pytest leg's child env as
    ``SCAN_EXTRA_FILES_ENV`` so tracked-file-enumerating scan tests see
    untracked payload files (#1889); untracked paths get a stderr audit NOTE
    (report-only). Bytecode determinism (#1950): stale repo-tree ``*.pyc``
    are purged BEFORE any leg runs, and every leg's CHILD env carries
    ``PYTHONDONTWRITEBYTECODE=1`` (``NO_BYTECODE_ENV`` via ``extra_env``)."""
    if payload is None:
        payload = [
            p.strip() for p in payload_file.read_text(encoding="utf-8").splitlines() if p.strip()
        ]
    _best_effort_choom()
    _bounded_fetch(repo)
    purge_repo_bytecode(repo)

    # #2039: one threshold read per gate run (the three consumers below stay
    # consistent) + ONE diagnostic-only gate-start sample — printed, NEVER
    # fed into load_hot (load before the wait is not load during the leg).
    load_threshold = _load_max()
    if load_threshold is not None:
        start_load = _sample_load1()
        if start_load is not None:
            print(f"inline_lint_gate: load1 at-start={start_load:.2f}", file=sys.stderr)

    for p in payload:
        if _git(repo, "ls-files", "--error-unmatch", "--", p).returncode != 0:
            print(
                f"inline_lint_gate: note: payload {p} is untracked — threaded via "
                f"{SCAN_EXTRA_FILES_ENV} for tracked-file-enumerating scan tests",
                file=sys.stderr,
            )

    # #2235 Phase B: scope the lint leg to the payload (`--files`) whenever NO
    # payload path touches a workflow-surface prefix; a surface payload keeps
    # today's exact bare no-flags form. The env-override seam is untouched —
    # an EPM_INLINE_GATE_LINT_CMD string runs verbatim (unscoped) either way.
    bare_lint_argv = ["uv", "run", "python", "scripts/workflow_lint.py"]
    scoped_eligible = bool(payload) and all(
        not p.startswith(WORKFLOW_SURFACE_PREFIXES) for p in payload
    )
    lint_argv = [*bare_lint_argv, "--files", *payload] if scoped_eligible else bare_lint_argv
    lint_output, _ = _run_leg(
        lint_argv,
        "EPM_INLINE_GATE_LINT_CMD",
        repo,
        LINT_TIMEOUT_S,
        extra_env=dict(NO_BYTECODE_ENV),
    )
    lint_audit = lint_output
    if scoped_eligible and FILES_MODE_REFUSED_RE.search(lint_output):
        # Fail-closed registry miss in workflow_lint's files-mode: exactly ONE
        # bare full re-run; the VERDICT input is the re-run's output, the
        # audit file keeps both (refused output + marker + re-run output).
        print(
            "inline_lint_gate: files-mode refused — falling back to ONE bare full lint run",
            file=sys.stderr,
        )
        lint_output, _ = _run_leg(
            bare_lint_argv,
            "EPM_INLINE_GATE_LINT_CMD",
            repo,
            LINT_TIMEOUT_S,
            extra_env=dict(NO_BYTECODE_ENV),
        )
        lint_audit = (
            lint_audit
            + "\n[inline_lint_gate: files-mode refused — bare full re-run below]\n"
            + lint_output
        )

    map_output, map_rc = _run_leg(
        ["uv", "run", "python", "scripts/select_step9c_tests.py", "--map-files", str(payload_file)],
        "EPM_INLINE_GATE_MAP_CMD",
        repo,
        FETCH_TIMEOUT_S + 120,
        extra_env=dict(NO_BYTECODE_ENV),
    )
    (out_dir / f"issue-{issue}-inline-map.txt").write_text(map_output, encoding="utf-8")
    if map_rc != 0:
        (out_dir / f"issue-{issue}-inline-lint.txt").write_text(lint_audit, encoding="utf-8")
        raise Inconclusive(f"map leg failed (rc={map_rc}) — unclassifiable payload")
    pairs = parse_map_pairs(map_output)

    pytest_output = ""
    load1_pre: float | None = None
    load1_post: float | None = None
    tests = sorted({t for t, _ in pairs})
    if tests:
        if load_threshold is not None:
            # #2039 D3+D4: bounded wait for a load spike to pass, then the
            # pre-leg endpoint sample (post-wait — the wait's own samples
            # never enter load_hot).
            _wait_for_load_drop(load_threshold)
            load1_pre = _sample_load1()
        pytest_output, _ = _run_leg(
            ["uv", "run", "pytest", *tests, "-q", "-rA"],
            "EPM_INLINE_GATE_PYTEST_CMD",
            repo,
            mapped_pytest_timeout(tests),
            # The #1889 payload threading and the #1950 no-bytecode guard
            # MUST merge into ONE child env (neither clobbers the other).
            extra_env={SCAN_EXTRA_FILES_ENV: os.pathsep.join(payload), **NO_BYTECODE_ENV},
        )
        if load_threshold is not None:
            load1_post = _sample_load1()
            print(
                f"inline_lint_gate: load1 pre-pytest={_fmt_load(load1_pre)} "
                f"post-pytest={_fmt_load(load1_post)} threshold={load_threshold:g}",
                file=sys.stderr,
            )

    (out_dir / f"issue-{issue}-inline-lint.txt").write_text(
        lint_audit + "\n" + pytest_output, encoding="utf-8"
    )
    return LegResults(
        lint_output=lint_output,
        map_pairs=pairs,
        pytest_output=pytest_output,
        load1_pre=load1_pre,
        load1_post=load1_post,
    )


def is_new_on_origin_main(repo: Path, path: str) -> bool:
    """True when *path* does not resolve on origin/main (incl. an unresolvable
    origin/main ref — the stricter direction)."""
    r = _git(repo, "cat-file", "-e", f"origin/main:{path}")
    return r.returncode != 0


def added_line_ranges(repo: Path, path: str) -> list[tuple[int, int]] | None:
    """Half-open new-file line ranges the round added vs origin/main, parsed
    from `git diff -U0`. None when the diff itself fails (caller blocks
    conservatively — refinement evidence unavailable)."""
    r = _git(repo, "diff", "-U0", "origin/main", "--", path)
    if r.returncode != 0:
        return None
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", r.stdout, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count > 0:
            ranges.append((start, start + count))
    return ranges


def warnings_attribution_idxs(pytest_lines: list[str]) -> set[int]:
    """Indices of pytest-leg lines that are warnings-summary attribution rows
    (node-id headers / `path: N warnings` aggregates) — report-class, never
    payload-naming hits (#1585; the #1112 false-block incident).

    Double predicate: the line must sit INSIDE an equals-fenced "warnings
    summary" section AND match the bare-row shape. Any other fenced header
    (PASSES, short test summary info) CLOSES the window; matching the raw
    (rstripped, not lstripped) line keeps indented warning bodies out."""
    idxs: set[int] = set()
    in_section = False
    for i, line in enumerate(pytest_lines):
        row = line.rstrip()
        if SECTION_FENCE_RE.match(row):
            in_section = bool(WARNINGS_SUMMARY_TITLE_RE.match(row))
            continue
        if in_section and WS_ATTRIBUTION_ROW_RE.match(row):
            idxs.add(i)
    return idxs


def passes_section_idxs(pytest_lines: list[str]) -> set[int]:
    """Indices of pytest-leg lines INSIDE the -rA ``PASSES`` fenced section
    (#2023; the #1345 v242 false-block incident). Every line there is captured
    output of a test pytest reports as PASSED — definitionally not red
    evidence — so the WHOLE section is report-class (single predicate,
    unlike the #1585 warnings-summary double predicate). Any other fence
    title CLOSES the window."""
    idxs: set[int] = set()
    in_section = False
    for i, line in enumerate(pytest_lines):
        row = line.rstrip()
        if SECTION_FENCE_RE.match(row):
            in_section = bool(PASSES_TITLE_RE.match(row))
            continue
        if in_section:
            idxs.add(i)
    return idxs


def _ledger_nodeids(ledger: dict) -> set[str]:
    """Ledger ``failing_tests`` rows -> nodeid strings (#2318 D1).

    The live Step 9c ledger stores DICT rows ({classname, file, name}); the
    pre-#2318 ``{str(n) for n in ...}`` coercion put dict REPRS in the set,
    so the ``node in ledger_failing`` membership test was always False and
    the #2235 Phase A layer never demoted anything. Dict rows map to
    ``f"{file}::{name}"``; plain str rows pass through (forward-compat); a
    row of any other shape is SKIPPED with one aggregate stderr warn naming
    the row types (never a silent no-op, never a raise)."""
    nodeids: set[str] = set()
    skipped: list[str] = []
    for row in ledger.get("failing_tests") or []:
        if isinstance(row, str):
            nodeids.add(row)
        elif (
            isinstance(row, dict)
            and isinstance(row.get("file"), str)
            and isinstance(row.get("name"), str)
        ):
            nodeids.add(f"{row['file']}::{row['name']}")
        else:
            skipped.append(type(row).__name__)
    if skipped:
        print(
            f"inline_lint_gate: warn: skipped {len(skipped)} ledger failing_tests "
            f"row(s) of unexpected shape: {sorted(set(skipped))}",
            file=sys.stderr,
        )
    return nodeids


def _step9c_baseline_module() -> ModuleType | None:
    """Lazy sibling import of ``scripts/step9c_baseline.py`` (#2318).

    Reuses an already-registered ``sys.modules['step9c_baseline']`` (test
    harnesses and the Step 9c gate register the same name — reuse is also
    what makes the registry a SINGLE object across consumers); otherwise
    loads via importlib with the module REGISTERED in sys.modules BEFORE
    ``exec_module`` — without the registration step9c_baseline crashes on
    its own ``@dataclass`` (field-type resolution looks the module up in
    sys.modules at class-creation time; measured). ANY failure returns None
    with ONE loud stderr warn (plan row C8): the violation-grain layer
    disengages and evaluate() keeps today's node-grain behavior — never a
    crash (a raise inside evaluate() would make every inline round
    INCONCLUSIVE, the #1388 wedge class via crash), never a new blocking
    class. The result (module or None) is cached per process."""
    if "mod" in _STEP9C_CACHE:
        cached = _STEP9C_CACHE["mod"]
        return cached if isinstance(cached, ModuleType) else None
    mod = sys.modules.get("step9c_baseline")
    if mod is None:
        try:
            spec = importlib.util.spec_from_file_location("step9c_baseline", _STEP9C_BASELINE_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"no loadable spec for {_STEP9C_BASELINE_PATH}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["step9c_baseline"] = mod
            spec.loader.exec_module(mod)
        except Exception as exc:  # C8: degrade LOUD to node grain, never crash the gate
            sys.modules.pop("step9c_baseline", None)  # no half-exec'd module residue
            print(
                f"inline_lint_gate: warn: step9c_baseline registry import failed "
                f"({exc!r}) — violation-grain demotion disabled; node-grain "
                "behavior retained",
                file=sys.stderr,
            )
            mod = None
    _STEP9C_CACHE["mod"] = mod
    return mod


def failure_block_idxs(pytest_lines: list[str]) -> dict[str, set[int]]:
    """Bucket the pytest ``FAILURES`` section into per-node line-index blocks
    (#2318 D2 — offender attribution lives in the traceback lines, not the
    ``FAILED <nodeid>`` short-summary row, so a line-scoped label cannot
    classify a whole-repo scan failure).

    The nodeid is DERIVED per block — ``<locus file>::<header name>`` (dots
    in the pre-``[`` header portion map to ``::`` for class-based tests),
    the locus being the block's LAST ``<file>.py:<line>: <Exc>`` row — then
    CROSS-VALIDATED against the ``-rA`` short-summary ``FAILED <nodeid>``
    set: a block whose derived nodeid is not in that set is DROPPED with one
    stderr warn (fail-closed, plan row C9 — brittleness across pytest
    versions degrades to today's behavior, never to a waiver). Never raises:
    any unexpected shape yields FEWER blocks, i.e. today's classification."""
    failed_nodes = {
        m.group(1) for ln in pytest_lines if (m := FAILED_NODE_RE.match(ln.strip())) is not None
    }
    blocks: dict[str, set[int]] = {}
    in_failures = False
    header: str | None = None
    idxs: list[int] = []

    def _close() -> None:
        nonlocal header, idxs
        if header is not None and idxs:
            locus_file: str | None = None
            for i in reversed(idxs):
                lm = FAILURE_LOCUS_RE.match(pytest_lines[i].rstrip())
                if lm is not None:
                    locus_file = lm.group(1)
                    break
            base, sep, param = header.partition("[")
            derived = f"{locus_file}::{base.replace('.', '::')}{sep}{param}" if locus_file else None
            if derived is not None and derived in failed_nodes:
                blocks.setdefault(derived, set()).update(idxs)
            else:
                print(
                    f"inline_lint_gate: warn: dropped FAILURES block {header!r} — "
                    f"derived nodeid {derived or '<no locus>'} not in the FAILED "
                    "summary set (fail-closed: today's classification retained)",
                    file=sys.stderr,
                )
        header = None
        idxs = []

    for i, line in enumerate(pytest_lines):
        row = line.rstrip()
        if SECTION_FENCE_RE.match(row):
            _close()
            in_failures = bool(FAILURES_TITLE_RE.match(row))
            continue
        if not in_failures:
            continue
        hm = FAILURE_HEADER_RE.match(row)
        if hm is not None:
            _close()
            header = hm.group(1).strip()
            continue
        if header is not None:
            idxs.append(i)
    _close()
    return blocks


def _violation_grain_labels(
    pytest_lines: list[str], ledger_nodeids: set[str], payload: list[str]
) -> dict[int, str]:
    """Per-line demotion labels for ledger-listed whole-repo scan nodes
    (#2318; the single-sided mirror of step9c_baseline compare's #2316 fix).

    For each cross-validated FAILURES block whose nodeid is BOTH in the
    sha-fresh ledger AND in the imported registry (parametrization suffix
    stripped — the ``_violation_setdiff_member`` rule) — and whose test file
    is not itself a payload path (own-file condition, #2235) — demote the
    block's lines iff ``extract_violation_paths(<block text>)`` contains NO
    payload path. A payload path in the extracted set means the round ADDED
    a violation to the already-red node: the block keeps today's
    classification (the Goal's fail-closed invariant, plan row C1). Empty
    extraction = conservative block + one loud warn (C5); a registry-import
    or post-import failure degrades LOUDLY to node grain (C8). Never raises
    inside evaluate()."""
    blocks = failure_block_idxs(pytest_lines)
    if not blocks:
        return {}
    mod = _step9c_baseline_module()
    if mod is None:
        return {}  # C8: the loader already printed the warn
    labels: dict[int, str] = {}
    try:
        for node, idxs in sorted(blocks.items()):
            if node not in ledger_nodeids:
                continue  # red on the gate run only — not pre-existing (C3)
            file, _, name = node.partition("::")
            if file in payload:
                continue  # own-file condition (#2235, C7)
            if f"{file}::{name.split('[', 1)[0]}" not in mod.VIOLATION_SET_SCAN_NODES:
                continue  # non-registry: the ^FAILED-line node-grain path only (C4)
            extracted = mod.extract_violation_paths(
                "\n".join(pytest_lines[i] for i in sorted(idxs))
            )
            if not extracted:
                print(
                    f"inline_lint_gate: warn: empty violation-path extraction for "
                    f"{node} — conservative block retained (no demotion)",
                    file=sys.stderr,
                )
                continue  # C5
            if any(p in extracted for p in payload):
                continue  # the payload ADDED a violation — the block stands (C1)
            for i in idxs:
                labels[i] = "pre-existing-on-main (ledger, violation-set)"
    except Exception as exc:  # C8 twin: attribute drift / extractor raise degrades loud
        print(
            f"inline_lint_gate: warn: violation-grain demotion failed ({exc!r}) — "
            "node-grain behavior retained",
            file=sys.stderr,
        )
        return {}
    return labels


def evaluate(
    payload: list[str], legs: LegResults, repo: Path, ledger: dict | None = None
) -> Verdict:
    """Apply the Step 9a-ter verdict semantics (module docstring) to the leg
    outputs. Raises Inconclusive on instrument-ran completeness failure.

    Warnings-summary attribution rows (#1585) and PASSES-section captured
    lines (#2023) — pytest leg only — are reclassified into
    ``verdict.reported`` (labeled) BEFORE per-path hit assignment, so both
    the NEW-on-origin/main and the MODIFIED conservative branches are fixed
    uniformly with no change to their own logic.

    Ledger subtraction (#2235 Phase A; ``ledger`` is ADDITIVE with default
    None so every existing call site binds unchanged): a pytest-leg hit whose
    line is a short-summary ``FAILED <nodeid>`` row with the node id in
    ``_ledger_nodeids(ledger)`` — AND whose test FILE is not itself a payload
    path (own-file condition: a round editing the failing test must still
    block) — is reclassified ``verdict.reported`` labeled
    ``pre-existing-on-main (ledger)``. For a ledger-listed
    ``VIOLATION_SET_SCAN_NODES`` member the demotion additionally decides at
    VIOLATION grain over the node's FAILURES block (#2318 —
    ``_violation_grain_labels``: demote the block's lines ONLY when the
    extracted violation-path set contains no payload path). SUBTRACTIVE ONLY
    by construction (block -> reported, never the reverse); a hit not
    attributable to a ledger-listed node keeps today's classification;
    ``ledger=None`` (absent / stale / corrupt per load_baseline_ledger) is
    bit-identical to the pre-#2235 gate.

    Load downgrade (#2039, module docstring § Load awareness): a path that
    WOULD OTHERWISE land in ``blocked``, whose block reasons rest on hits
    that are ALL pytest-leg, moves to ``verdict.load_deferred`` when
    ``load_hot`` — never certified, exit 3. Lint-leg hits and would-be-PASS
    paths are untouched under any load."""
    if not LINT_TERMINAL_RE.search(legs.lint_output):
        raise Inconclusive(
            "lint-leg-dead — no healthy `workflow_lint: PASS|FAIL (` terminal line "
            "(`schema FAIL` early-exit is rejected)"
        )
    if legs.map_pairs and not PYTEST_SUMMARY_RE.search(legs.pytest_output):
        raise Inconclusive("pytest-leg-dead — non-empty test mapping but no pytest summary line")

    lint_lines = legs.lint_output.splitlines()
    pytest_lines = legs.pytest_output.splitlines()
    ws_idxs = warnings_attribution_idxs(pytest_lines)  # pytest leg ONLY
    pass_idxs = passes_section_idxs(pytest_lines)  # pytest leg ONLY (#2023)
    # #2235 Phase A: node ids the sha-matched ledger records as ALREADY
    # failing on origin/main (empty set when ledger is None/absent — the
    # subtraction layer is then fully disengaged). The live rows are DICTS —
    # mapped to file::name nodeids by _ledger_nodeids (#2318 D1).
    ledger_failing: set[str] = _ledger_nodeids(ledger) if ledger is not None else set()
    # #2318: violation-grain demotion labels for ledger-listed whole-repo
    # scan nodes (per-line, over cross-validated FAILURES blocks) — computed
    # once, before the per-line loop; empty when nothing demotes.
    vg_labels: dict[int, str] = (
        _violation_grain_labels(pytest_lines, ledger_failing, payload) if ledger_failing else {}
    )
    # (line, report-class label | None, leg origin) — the leg origin drives
    # the #2039 load downgrade (lint-leg hits are never load-downgraded).
    combined: list[tuple[str, str | None, str]] = [(ln, None, "lint") for ln in lint_lines] + [
        (
            ln,
            "warnings-summary attribution"
            if i in ws_idxs
            else "passing-capture"
            if i in pass_idxs
            else vg_labels.get(i),
            "pytest",
        )
        for i, ln in enumerate(pytest_lines)
    ]
    hits: dict[str, list[tuple[str, str]]] = {p: [] for p in payload}
    verdict = Verdict()
    for line, label, leg in combined:
        stripped = line.strip()
        ledger_label: str | None = None
        if ledger_failing and leg == "pytest":
            m = FAILED_NODE_RE.match(stripped)
            if m is not None:
                node = m.group(1)
                node_file = node.split("::", 1)[0]
                # Own-file condition (#2235): the round editing the failing
                # test's own file must still block, ledger-listed or not.
                if node in ledger_failing and node_file not in payload:
                    ledger_label = "pre-existing-on-main (ledger)"
        for p in payload:
            if p not in line:
                continue
            if label:
                verdict.reported.append(f"[{label}] {line}")
            elif ledger_label:
                verdict.reported.append(f"[{ledger_label}] {line}")
            elif stripped.startswith(NON_RED_PREFIXES):
                verdict.reported.append(line)
            else:
                hits[p].append((line, leg))

    load_hot = _load_hot(legs)
    for p in payload:
        if not hits[p]:
            verdict.passing.append(p)
            continue
        if is_new_on_origin_main(repo, p):
            reasons: list[str] = [
                f"NEW on origin/main with {len(hits[p])} non-WARN payload-naming hit(s) "
                "(payload-caused by construction)",
                *[line for line, _ in hits[p]],
            ]
            # #2039 D5: the NEW branch keys off the same hit lines — every
            # hit underlies the block, so all-pytest + hot defers.
            if load_hot and {leg for _, leg in hits[p]} == {"pytest"}:
                verdict.load_deferred[p] = reasons
            else:
                verdict.blocked[p] = reasons
            continue
        ranges = added_line_ranges(repo, p)
        lineno_re = re.compile(re.escape(p) + r":(\d+):")
        reasons = []
        blocking_legs: set[str] = set()
        preexisting: list[str] = []
        for line, leg in hits[p]:
            m = lineno_re.search(line)
            if m is None:
                reasons.append(
                    f"payload-naming hit without a parseable lineno (conservative block): {line}"
                )
                blocking_legs.add(leg)
            elif ranges is None:
                reasons.append(f"added-line ranges unavailable (conservative block): {line}")
                blocking_legs.add(leg)
            elif any(a <= int(m.group(1)) < b for a, b in ranges):
                reasons.append(f"hit inside the round's added lines: {line}")
                blocking_legs.add(leg)
            else:
                preexisting.append(line)
        verdict.reported.extend(preexisting)
        if not reasons:
            verdict.passing.append(p)
        elif load_hot and blocking_legs == {"pytest"}:
            # #2039 D5: the downgrade keys on the WOULD-BE-BLOCKED outcome —
            # never mere pytest-hit presence (an outside-added-ranges path
            # stays PASS above) — and only when EVERY hit line underlying the
            # block reasons is pytest-leg under hot load.
            verdict.load_deferred[p] = reasons
        else:
            verdict.blocked[p] = reasons
    return verdict


def write_cert(
    passing: list[str], snapshots: dict[str, str], cert_path: Path, repo: Path
) -> tuple[list[str], list[str]]:
    """Append `v1 <epoch> <sha> <path>` lines for passing paths whose worktree
    content still matches the read_payload snapshot; a mismatch means the file
    was edited DURING the gate run -> INCONCLUSIVE for that path, no cert
    (TOCTOU guard). A first-pass mismatch gets ONE bounded settle-and-re-hash
    retry (#1857, EPM_CERT_REHASH_DELAY_S) so a transient worktree flip does
    not withhold the cert; only a STABLE mismatch stays TOCTOU.
    Append runs under flock with an inode re-check (#1620): a
    concurrent trim's os.replace can swap the path to a NEW inode while this
    writer waits on the OLD inode's lock, so after acquiring the lock the fd
    is re-opened until it still names cert_path (bounded; exhaustion degrades
    to the pre-fix lost-line behavior, which only re-blocks — safe
    direction). The 500-line trim writes tmp+rename so a concurrent hook
    read never sees a truncated file."""
    certified: list[str] = []
    toctou: list[str] = []
    epoch = int(time.time())
    lines: list[str] = []
    mismatched: list[str] = []
    for p in passing:
        current = _hash_object(repo, p)
        if current is None or current != snapshots[p]:
            mismatched.append(p)
            continue
        lines.append(f"v1 {epoch} {snapshots[p]} {p}\n")
        certified.append(p)
    if mismatched:
        # Bounded settle-and-re-hash retry (#1857): a TRANSIENT worktree-hash
        # flip (concurrent writer / filesystem settle — the 07-30 false
        # INCONCLUSIVE on a file not being edited) re-hashes back to the
        # read_payload snapshot after one settle delay and certifies as
        # normal; a STABLE mismatch stays TOCTOU with the message unchanged.
        # ONE sleep total (not per path), OUTSIDE the flock below (lines are
        # assembled before locking). The retry re-READS the worktree hash —
        # it never re-binds the cert to different content (the cert line
        # still carries the snapshot sha, which the re-hash must EQUAL).
        # Knob: EPM_CERT_REHASH_DELAY_S (seconds, default 2; tests set 0 /
        # monkeypatch time.sleep). A malformed value falls back to the
        # default — the re-check always runs (fail toward TOCTOU/block).
        try:
            delay = float(os.environ.get("EPM_CERT_REHASH_DELAY_S", "2"))
        except ValueError:
            delay = 2.0
        time.sleep(max(delay, 0.0))
        for p in mismatched:
            current = _hash_object(repo, p)
            if current is not None and current == snapshots[p]:
                lines.append(f"v1 {epoch} {snapshots[p]} {p}\n")
                certified.append(p)
            else:
                toctou.append(p)
    if lines:
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(cert_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
            # Inode re-check under flock (#1620 fix (d)): the fd above is
            # opened BEFORE the lock, so a concurrent trim's os.replace can
            # swap cert_path to a NEW inode while this writer blocks on the
            # OLD inode's flock — appending would land on the orphaned inode
            # and the lines would never appear at cert_path. Re-open +
            # re-lock until the locked fd still names the path (bounded at 5
            # attempts; exhaustion proceeds with the last fd — the pre-fix
            # lost-line behavior, which only re-blocks: safe direction).
            for _ in range(5):
                try:
                    path_ino = os.stat(cert_path).st_ino
                except FileNotFoundError:
                    path_ino = -1  # path vanished mid-race: treat as mismatch
                if os.fstat(fh.fileno()).st_ino == path_ino:
                    break
                fh.close()
                fh = open(cert_path, "a+", encoding="utf-8")
                fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write("".join(lines))
            fh.flush()
            os.fsync(fh.fileno())
            fh.seek(0)
            all_lines = fh.read().splitlines(keepends=True)
            if len(all_lines) > CERT_TRIM_LINES:
                fd, tmp = tempfile.mkstemp(dir=str(cert_path.parent), prefix=cert_path.name + ".")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tf:
                        tf.writelines(all_lines[-CERT_TRIM_LINES:])
                    os.replace(tmp, cert_path)
                except BaseException:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
        finally:
            fh.close()
    return certified, toctou


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--issue", type=int, required=True, help="task number (audit-file keying)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--payload-file", help="newline-delimited repo-relative payload paths")
    src.add_argument("--paths", nargs="+", help="payload paths given inline")
    ap.add_argument("--repo-root", help="checkout root override (default: invoking git toplevel)")
    ap.add_argument(
        "--out-dir", default="/tmp", help="audit-output directory (default /tmp; test support)"
    )
    args = ap.parse_args(argv)

    try:
        repo = Path(args.repo_root).resolve() if args.repo_root else _git_toplevel()
        if args.payload_file:
            payload_file = Path(args.payload_file)
            if LEGACY_PAYLOAD_BASENAME_RE.match(payload_file.name):
                # Refuse BEFORE any leg runs (#1948): no multi-minute leg burn
                # on a doomed invocation, and the shared legacy path never
                # gates.
                raise Inconclusive(
                    "legacy shared payload path refused (#1948) — the bare "
                    "issue-keyed name is clobbered by concurrent same-issue "
                    "rounds (cross-certification); use a round-unique name, "
                    "e.g. /tmp/issue-<N>-<slug>-inline-payload.txt"
                )
            if not payload_file.is_file():
                raise Inconclusive(f"payload file missing: {payload_file}")
            raw_paths = payload_file.read_text(encoding="utf-8").splitlines()
            payload_source = str(payload_file)
        else:
            raw_paths = list(args.paths)
            payload_source = "inline-paths"
        snapshots = read_payload(raw_paths, repo)
        payload = sorted(snapshots)
        # Payload-binding audit line (#1948): printed BEFORE any report /
        # verdict line, binding this gate run to the exact resolved payload
        # list (source + count + content hash of the sorted list). The
        # terminal verdict lines (PASS / BLOCK (...) / INCONCLUSIVE (...))
        # stay byte-stable — no consumer grep breaks.
        list_sha = hashlib.sha256(("\n".join(payload) + "\n").encode("utf-8")).hexdigest()[:12]
        print(
            f"inline_lint_gate: payload-source {payload_source} "
            f"n={len(payload)} list-sha256={list_sha}"
        )
        # ALWAYS materialize a PRIVATE mkstemp copy of the resolved payload
        # list for the map leg's --map-files contract (#1948 — hoisted out of
        # the former --paths-only branch): the caller's file was read exactly
        # once above, so a mid-run overwrite of the caller's path can no
        # longer redirect the mapped-test set (defense-in-depth beyond the
        # round-unique path contract).
        fd, tmp_payload = tempfile.mkstemp(prefix=f"issue-{args.issue}-inline-payload.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(payload) + "\n")
        private_payload_file = Path(tmp_payload)
        legs = run_legs(private_payload_file, args.issue, repo, Path(args.out_dir), payload=payload)
        # Ledger provenance audit line (#2235 Phase A arm 2): loaded AFTER
        # run_legs so the main_sha pin is checked against the SAME
        # origin/main the -U0 diffs use (post-_bounded_fetch); printed
        # BEFORE any report/verdict line so the round's audit trail shows
        # whether the subtraction layer was armed (mirrors the #1948
        # payload-source binding line).
        ledger = load_baseline_ledger(repo)
        if ledger is not None:
            ledger_state = f"fresh {str(ledger.get('main_sha', ''))[:12]}"
        elif (repo / LEDGER_PATH_REL).is_file():
            ledger_state = "stale"
        else:
            ledger_state = "absent"
        print(f"inline_lint_gate: ledger={ledger_state}")
        verdict = evaluate(payload, legs, repo, ledger=ledger)
    except Inconclusive as exc:
        print(f"inline_lint_gate: INCONCLUSIVE ({exc})")
        return 3

    cert_path = Path(os.environ.get("EPM_INLINE_CERT_PATH", DEFAULT_CERT_PATH))
    certified, toctou = write_cert(verdict.passing, snapshots, cert_path, repo)

    for line in verdict.reported:
        print(f"inline_lint_gate: report (pre-existing / WARN — never blocks): {line}")
    for p in sorted(verdict.blocked):
        for reason in verdict.blocked[p]:
            print(f"inline_lint_gate: {p}: {reason}")
    for p in certified:
        print(f"inline_lint_gate: certified {p} ({snapshots[p][:12]}) -> {cert_path}")

    # #2039 D6: the load-deferred INCONCLUSIVE prints BEFORE any BLOCK return
    # (same mixed-outcome rationale as the TOCTOU note below); load-deferred
    # paths NEVER certify, and the reason rides the existing
    # `inline_lint_gate: INCONCLUSIVE (` grep shape — no consumer grep breaks.
    if verdict.load_deferred:
        threshold = _load_max()
        samples = [s for s in (legs.load1_pre, legs.load1_post) if s is not None]
        peak_txt = f"{max(samples):g}" if samples else "unavailable"
        thr_txt = "disabled" if threshold is None else f"{threshold:g}"
        print(
            "inline_lint_gate: INCONCLUSIVE (pytest-leg red under load — "
            f"load1={peak_txt} >= EPM_GATE_LOAD_MAX={thr_txt}, not payload-attributed; "
            f"re-run when load drops: {' '.join(sorted(verdict.load_deferred))})"
        )
    # TOCTOU note prints BEFORE any BLOCK return (round-2 Minor): in a mixed
    # BLOCK+TOCTOU outcome the operator must learn of the mid-gate edit NOW,
    # not on the next hook block. Exit precedence: BLOCK (1) beats any
    # INCONCLUSIVE population (3) — load-deferred (#2039) and TOCTOU alike.
    if toctou:
        print(
            "inline_lint_gate: INCONCLUSIVE "
            f"(edited during gate — re-run: {' '.join(sorted(toctou))})"
        )
    if verdict.blocked:
        print(f"inline_lint_gate: BLOCK ({' '.join(sorted(verdict.blocked))})")
        return 1
    if verdict.load_deferred or toctou:
        return 3
    print("inline_lint_gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
