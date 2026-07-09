#!/usr/bin/env python
"""Select the touched-file-scoped pytest subset for the ``/issue`` Step 9c test-verdict gate.

The full project suite (~5800 tests in ~248 files, no ``pytest-xdist``
parallelism) stalls and is harness-/earlyoom-killed in sparse worktrees
(#665/#736), so Step 9c must NOT run ``pytest tests/`` wholesale by default.
This helper computes a deterministic subset = the tests covering the files THIS
branch changed (``git diff --name-only <base>...HEAD``) UNION a pinned literal
list of workflow-invariant tests (``WORKFLOW_INVARIANT``) that gate the
project's load-bearing invariants regardless of which files were touched.

It mechanizes the hand-picked subset #736 ran by hand. The selection is pure
deterministic path arithmetic over a git diff — no model call, no new
dependency (stdlib + git only).

Touched-file -> test mapping (per touched file ``f``):
  * a WORKFLOW_SURFACE glob (``.claude/**``, ``CLAUDE.md``, ``tasks/**``,
    docs/figures/artifacts) -> SKIP. The WORKFLOW_INVARIANT set IS the gate
    for these; there is no per-file test map and they are not "untested".
  * a data/config/doc file (``.json`` / ``.yaml`` / ``.md`` / ... ) -> SKIP
    (not code; no test mapping; not "untested").
  * ``tests/test_<X>.py`` -> include ``f`` itself.
  * a code file (``scripts/<X>.py`` / ``src/.../<X>.py`` / any other ``*.py``):
    map ``stem = Path(f).stem`` to ``tests/test_{stem}.py`` (exact) PLUS the
    broad ``tests/test_*{stem}*.py`` glob. If NEITHER matches an existing test
    -> the file lands in ``untested_touched`` (a loud WARN the Step 9c marker
    surfaces; never a silent coverage hole).
  * ANY touched path matching a ``GLOB_SCAN_TESTS`` scan glob ADDITIONALLY
    selects that scanning test (tests that cover files via a directory scan
    at test time are reachable from no stem — #895). A map hit never
    suppresses the ``untested_touched`` WARN for the touched file itself.

``safe-by-direction``: the broad ``*{stem}*`` glob arm deliberately OVER-matches
on short stems (e.g. stem ``gcp`` selects ``test_gcp_backend.py``) — running a
few extra invariant-adjacent tests is the safe failure direction. There is no
``else: tighten the glob`` arm: a missed mapping surfaces as an
``untested_touched`` WARN, never a silently-dropped file.

Root resolution: with no ``--repo-root``, everything (the ``git diff`` cwd,
touched-test existence checks, stem-glob mapping, invariant presence) resolves
against the INVOKING checkout's git toplevel — the issue-worktree root when run
from a worktree (where the branch diff AND its branch-new ``tests/`` files
live), the main repo root when run there. ``--repo-root`` is the checkout-root
override. At Step 9c, run from the issue worktree (incident #851: resolving the
MAIN root made ``git diff main...HEAD`` empty by construction and silently
dropped the branch's own tests from the gate).

Degenerate empty-diff (e.g. run at a checkout with no commits ahead of
``--base``): the selection falls back to the WORKFLOW_INVARIANT set only, so
the gate never runs zero tests — and a loud ``NOTE — empty diff`` line is
printed to stderr. On a worktree-based task whose branch HAS commits ahead of
the base, that NOTE means the helper ran from the wrong cwd (re-run from the
issue worktree); from a checkout genuinely at the base it is expected and
benign.

Usage::

    uv run python scripts/select_step9c_tests.py [--base main] [--repo-root <path>] [--json]
    uv run python scripts/select_step9c_tests.py --map-files <file> [--repo-root <path>]

``--map-files FILE`` (the ``/issue`` Step 10d merge-gate mapping mode, #1147):
read newline-delimited repo-relative paths from FILE and print one
``<scan_test>\\t<matched_path>`` line per :data:`GLOB_SCAN_TESTS` hit
(:func:`map_scan_tests`), skipping the diff-based selection entirely. A scan
test absent from the work root is dropped with a stderr WARN. Empty stdout on
no match is a SUCCESS (exit 0 — the gate's skip signal); exit 1 only when FILE
is unreadable (the gate fails CLOSED on an unclassifiable payload).

Default output: the exact gate invocation
``timeout --kill-after=60s <T>s uv run pytest <files...> -v --tb=short`` on
stdout — ``<T>`` sized deterministically by :func:`recommended_timeout_s`
(#1046: 120s base + 30s/file + a 900s surcharge when
``tests/test_workflow_lint.py`` is selected; measured from 27 real gate junits
2026-07-04/05, workflow-lint present in 26 of them) — then any
``untested touched file: <path>`` WARN lines on stderr. Every run also prints
a one-line provenance breadcrumb to stderr (the resolved work root + current
branch) so the Step 9c marker records which checkout the subset was selected
against, plus a machine-greppable ``recommended-timeout-s=<T>`` sizing line
(the gate exceeds the 600s foreground Bash tool cap, so Step 9c runs it as a
BACKGROUND invocation — SKILL.md 9c step 1b). ``--json`` emits
``{"tests": [...], "untested_touched": [...], "base": "...",
"missing_invariants": [...], "selection_reasons": {test: [reasons]},
"n_tests": <int>, "recommended_timeout_s": <int>,
"slow_tests_selected": [...]}`` (a
reason is ``invariant`` / ``touched-test`` / ``stem-map:<touched file>`` /
``glob-scan:<touched file>`` — #1022). Exit 0 on success (even with WARN lines);
exit 1 if an underlying ``git`` call fails irrecoverably (work-root resolution
or the diff) or if the selection comes back EMPTY (the zero-test-gate
refusal).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# --- Pinned workflow-invariant tests (plan §5 + 1 brief addition, 32 files). -
# A module-level literal tuple, NOT a glob: a future ``tests/test_workflowish.py``
# that is NOT meant to gate Step 9c must not silently join the gate, and the gate
# must not silently shrink if a glob arm stops matching. Drift is made loud by the
# on-disk existence check (a vanished entry prints WORKFLOW-INVARIANT MISSING) and
# pinned against the live tree by tests/test_select_step9c_tests.py.
WORKFLOW_INVARIANT: tuple[str, ...] = (
    # group 1 — task-workflow API
    "tests/test_task_workflow.py",
    "tests/test_task_workflow_list_children.py",
    "tests/test_task_workflow_post_marker_echo.py",
    "tests/test_task_workflow_worktree.py",
    # group 2 — workflow-lint / yaml / fix-dedup
    "tests/test_workflow_lint.py",
    "tests/test_workflow_lint_dotenv_check.py",
    "tests/test_workflow_yaml.py",
    "tests/test_workflow_fix_dedup.py",
    "tests/test_workflow_hub_upload_as_file.py",
    # group 3 — test_no_* invariants
    "tests/test_no_auto_runpod_path_under_any_failure.py",
    "tests/test_no_direct_task_path_construction.py",
    "tests/test_no_dollar_budget_caps.py",
    "tests/test_no_per_file_raw_completions_loop.py",
    "tests/test_no_pod_side_task_py_shellout.py",
    # group 4 — verifiers
    "tests/test_verify_plan.py",
    "tests/test_verify_task_body.py",
    "tests/test_verify_task_body_audit_claim.py",
    "tests/test_verify_clean_result.py",
    "tests/test_verify_clean_result_body_stdin.py",
    "tests/test_verify_paper.py",
    "tests/test_verify_uploads_card_fallback.py",
    "tests/test_verify_uploads_claimed_urls.py",
    "tests/test_verify_uploads_type_selection.py",
    # group 5 — gate / dispatch / contract
    "tests/test_failure_classifier.py",
    "tests/test_sparse_worktree.py",
    "tests/test_autonomous_plan_gate.py",
    "tests/test_autonomous_session_watch.py",
    "tests/test_issue_skill_marker_contract.py",
    "tests/test_plan_handoff_path_convention.py",
    "tests/test_clean_result_critic_planned_vs_actual.py",
    "tests/test_check_no_secret_shaped_strings.py",
    "tests/test_check_mcp_json_no_secrets.py",
)

# --- Touched files that short-circuit (no per-file test map). ----------------
# These gate via the WORKFLOW_INVARIANT set, not a per-file test, so a touched
# file matching one of these is SKIPPED (and is NOT an "untested" file).
WORKFLOW_SURFACE_GLOBS: tuple[str, ...] = (
    ".claude/agents/*.md",
    ".claude/skills/**/SKILL.md",
    ".claude/skills/**/*.md",
    ".claude/skills/**/*.json",
    ".claude/rules/*.md",
    ".claude/workflow.yaml",
    ".claude/settings*.json",
    ".claude/agent-memory/**/*.md",
    "CLAUDE.md",
    "tasks/**",  # task state — never code
    "docs/**",
    "figures/**",
    "eval_results/**",
    "ood_eval_results/**",
    "raw/**",
)

# Data / config / doc extensions: not code, no test mapping, never "untested".
_DATA_DOC_SUFFIXES: frozenset[str] = frozenset(
    {".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".lock", ".png", ".svg", ".pdf"}
)

# --- Glob-scan invariant tests (#895). ---------------------------------------
# Some tests cover source files via a DIRECTORY SCAN at test time
# (``root.glob(...)``) rather than by importing the module under test, so no
# touched-file stem can ever reach them: a diff adding scripts/issue900_foo.py
# maps to no tests/test_*issue900_foo*.py, yet
# tests/test_shared_vm_thread_caps.py asserts that very file's import order
# (#877: the selector's printed command had to be hand-amended; 12 thread-caps
# offenders accreted after the #847 freeze this way). Map each scanning test
# to the VERBATIM scan globs its own source uses; a touched path matching any
# glob ADDS the test. A pinned literal, NOT discovered — same curation rule as
# WORKFLOW_INVARIANT (live-tree + verbatim-source drift pins live in
# tests/test_select_step9c_tests.py). Additive only: a map hit does NOT mark
# the touched file "tested" for the stem map (the scan asserts a cross-cutting
# invariant ABOUT the file, not the file's own logic), so untested_touched
# WARNs still fire.
GLOB_SCAN_TESTS: dict[str, tuple[str, ...]] = {
    # test_no_new_torch_before_dotenv_vm_entrypoints scan roots (#1187: its
    # _scan_targets() — every tracked scripts/**/*.py + __main__-guarded
    # experiments modules; these map globs deliberately OVER-select vs the
    # scanner's tracked/guard filters — additive, safe-by-direction).
    "tests/test_shared_vm_thread_caps.py": (
        "scripts/**/*.py",
        "src/explore_persona_space/experiments/**/*.py",
    ),
    # _DISPATCHER_GLOBS (its L80-90) — explicit-env subprocess spawn scanner.
    "tests/test_subprocess_env_explicit.py": (
        "scripts/dispatch_*.py",
        "scripts/run_sweep*.py",
        "scripts/run_pipeline*.py",
        "scripts/run_experiment_*.py",
        "scripts/run_dose_response_*.py",
        "scripts/run_factor_screen_*.py",
        "src/explore_persona_space/experiments/*/run_*.py",
        "src/explore_persona_space/experiments/*/dispatch_*.py",
        "src/explore_persona_space/experiments/*/__main__.py",
    ),
}

# --- Gate-timeout sizing (#1046). --------------------------------------------
# Measured from 27 Step 9c gate junits on the shared VM (2026-07-04..05,
# /tmp/step9c-junit-issue-*.xml, per-testcase `time` summed per file;
# test_workflow_lint.py present in 26 of them):
# tests/test_workflow_lint.py alone min 319 s / median 390 s / max 771 s;
# whole-gate totals median ~662 s / max 1285 s on 32-46-file selections. The
# foreground Bash tool cap is 600 s, so the printed command carries this bound
# and Step 9c runs the gate as a BACKGROUND invocation (SKILL.md 9c step 1b).
# Constants are deliberately generous (~1.4-2x over worst measured): an
# oversized bound only ever fires on a genuine wedge; an undersized one kills
# healthy gates (#991/#996/#906, exit 143 at 480-540 s foreground bounds).
TIMEOUT_BASE_S = 120  # pytest startup + collection (~2500 tests) + imports
TIMEOUT_PER_FILE_S = 30  # ~2x the p90 per-file runtime of non-slow files
TIMEOUT_FLOOR_S = 900
# Per-file surcharges for files whose OWN max exceeds the per-file allocation
# by an order of magnitude. Pinned literal (same curation rule as
# WORKFLOW_INVARIANT); live-tree drift pin in tests/test_select_step9c_tests.py.
SLOW_TESTS: dict[str, int] = {
    # 3845 lines; runs whole-tree lints repeatedly. Max 771 s measured (n=26).
    "tests/test_workflow_lint.py": 900,
}


def recommended_timeout_s(tests: list[str]) -> int:
    """Deterministic `timeout(1)` bound for a Step 9c gate selection.

    ``BASE + PER_FILE * len(tests) + sum(slow surcharges)``, floored at
    ``TIMEOUT_FLOOR_S``. Invariant-only selection (32 files incl. the
    workflow-lint surcharge) -> 1980 s (~33 min), consistent with the existing
    invariant-set-scale precedents (``step9c_baseline.py refresh``
    ``--timeout-s`` default 1800 s; the SKILL.md detached refresh's 2100 s).
    """
    t = TIMEOUT_BASE_S + TIMEOUT_PER_FILE_S * len(tests)
    t += sum(SLOW_TESTS.get(x, 0) for x in tests)
    return max(t, TIMEOUT_FLOOR_S)


def _matches_any(path: str, globs: tuple[str, ...]) -> bool:
    """True if *path* matches any glob. ``**`` matches across directory separators."""
    for g in globs:
        if fnmatch.fnmatch(path, g):
            return True
        # fnmatch treats ``*`` greedily across ``/`` already, but a ``foo/**``
        # pattern needs an explicit prefix check for the zero-segment case
        # (``docs/**`` should match ``docs/x.md``).
        if g.endswith("/**") and (path == g[:-3] or path.startswith(g[:-2])):
            return True
        # pathlib's ``**`` matches ZERO directories too; fnmatch's needs the
        # literal ``/`` on both sides. Try the zero-segment collapse so
        # verbatim Path.glob patterns keep their pathlib semantics here
        # (over-match is the safe direction — module docstring).
        if "/**/" in g and fnmatch.fnmatch(path, g.replace("/**/", "/")):
            return True
    return False


def _scan_pairs(files: list[str]) -> set[tuple[str, str]]:
    """All ``(scan_test, matched_file)`` pairs per :data:`GLOB_SCAN_TESTS` (no existence filter)."""
    return {
        (scan_test, f)
        for f in files
        for scan_test, scan_globs in GLOB_SCAN_TESTS.items()
        if _matches_any(f, scan_globs)
    }


def map_scan_tests(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Map *files* to ``(scan_test, matched_file)`` pairs via :data:`GLOB_SCAN_TESTS`.

    Pure path arithmetic over the pinned map (no git); a scan test absent
    from *work_root* is dropped (the caller's tree cannot run it). Returns
    sorted unique pairs. This is the ``--map-files`` backing function the
    ``/issue`` Step 10d pre-push merge gate consumes (#1147).
    """
    return sorted(p for p in _scan_pairs(files) if (work_root / p[0]).exists())


def compute_touched(
    base: str,
    work_root: Path,
    _runner: Callable[[list[str]], str] | None = None,
) -> list[str]:
    """Return the repo-relative paths the current branch changed vs *base*.

    Uses the three-dot ``git diff --name-only <base>...HEAD`` form: it diffs the
    merge-base of *base* and HEAD against HEAD — exactly the branch's own
    additions/modifications, not changes on *base* that HEAD lacks. The diff
    runs with *work_root* as cwd, so HEAD is the invoking checkout's branch
    (the issue branch from a worktree — #851). ``_runner`` is injectable for
    tests (it receives the argv list and returns stdout).
    """

    def _default_runner(argv: list[str]) -> str:
        proc = subprocess.run(
            argv,
            cwd=str(work_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout

    runner = _runner or _default_runner
    out = runner(["git", "diff", "--name-only", f"{base}...HEAD"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def select_tests_with_reasons(
    touched: list[str], work_root: Path
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Superset of :func:`select_tests`: also returns ``{test: sorted reasons}``.

    A reason is ``'invariant' | 'touched-test' | 'stem-map:<touched file>' |
    'glob-scan:<touched file>'``; a test may carry several (e.g. an invariant
    that is also stem-mapped from a touched file). Selection behavior is
    IDENTICAL to :func:`select_tests` — same tests, same ``untested_touched``
    WARN list, same sorted ordering (#1022: the reasons feed
    ``step9c_baseline.py compare``'s diff-linked-ness read, which must come
    from the SAME mapping logic that selected the run's tests).
    """
    selected: dict[str, set[str]] = {}
    untested: list[str] = []

    def _add(test: str, reason: str) -> None:
        selected.setdefault(test, set()).add(reason)

    for f in touched:
        # Glob-scan invariant tests (#895): additive, never sets ``matched``.
        for scan_test, scan_globs in GLOB_SCAN_TESTS.items():
            if _matches_any(f, scan_globs) and (work_root / scan_test).exists():
                _add(scan_test, f"glob-scan:{f}")
        # Workflow-surface files gate via the invariant set; not "untested".
        if _matches_any(f, WORKFLOW_SURFACE_GLOBS):
            continue
        p = Path(f)
        # A touched test file includes itself.
        if f.startswith("tests/") and p.name.startswith("test_") and p.suffix == ".py":
            if (work_root / f).exists():
                _add(f, "touched-test")
            continue
        # Data / config / doc files: not code, no test mapping.
        if p.suffix in _DATA_DOC_SUFFIXES:
            continue
        # Code files (.py anywhere): map stem -> test_<stem>.py + test_*<stem>*.py.
        if p.suffix == ".py":
            stem = p.stem
            matched = False
            exact = work_root / "tests" / f"test_{stem}.py"
            if exact.exists():
                _add(f"tests/test_{stem}.py", f"stem-map:{f}")
                matched = True
            for hit in sorted((work_root / "tests").glob(f"test_*{stem}*.py")):
                _add(f"tests/{hit.name}", f"stem-map:{f}")
                matched = True
            if not matched:
                untested.append(f)
        # Any other extension (no recognized mapping): ignore silently — it is
        # neither code with a test nor a workflow-invariant surface.

    for t in WORKFLOW_INVARIANT:
        if (work_root / t).exists():
            _add(t, "invariant")
    final = sorted(selected)
    reasons = {t: sorted(rs) for t, rs in selected.items()}
    return final, untested, reasons


def select_tests(touched: list[str], work_root: Path) -> tuple[list[str], list[str]]:
    """Map *touched* files to their covering tests + the workflow-invariant set.

    UNCHANGED signature — delegates to :func:`select_tests_with_reasons` and
    drops the reasons. All existence checks + the stem-glob mapping resolve
    against *work_root* (the invoking checkout), so a branch-new touched test
    is admitted and a deleted-on-branch test is correctly dropped (it does not
    exist at the worktree's HEAD either). Returns ``(tests, untested_touched)``:
      * ``tests`` — sorted union of mapped per-file tests and the present-on-disk
        subset of :data:`WORKFLOW_INVARIANT` (deterministic order; required so two
        invocations on the same git state return an identical list).
      * ``untested_touched`` — non-trivial touched code files with no mapped test
        (a WARN list the Step 9c marker surfaces; never silently dropped).
    """
    tests, untested, _ = select_tests_with_reasons(touched, work_root)
    return tests, untested


def missing_invariants(work_root: Path) -> list[str]:
    """Return WORKFLOW_INVARIANT entries that are NOT present on disk (drift)."""
    return [t for t in WORKFLOW_INVARIANT if not (work_root / t).exists()]


def _resolve_work_root(arg: str | None) -> Path:
    """Resolve the checkout root everything else (diff cwd, existence checks) uses."""
    if arg:
        return Path(arg).resolve()
    # The INVOKING checkout's toplevel: the issue-worktree root when run from
    # a worktree (where the branch diff AND its branch-new tests/ files
    # live), the main repo root when run there. --show-toplevel is CORRECT
    # here (the SKILL.md Step 5a precedent — we WANT the invoking checkout);
    # the #506 path-doubling bug applies only to CONSTRUCTING
    # <root>/.claude/worktrees/issue-<N> by appending to a root (Steps
    # 4a/10d), which this helper never does. Incident #851: resolving the
    # MAIN root here made `git diff main...HEAD` empty by construction
    # (HEAD==main) and hid branch-new test files from the existence checks.
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).resolve()


def _current_branch(work_root: Path) -> str:
    """Best-effort current-branch read for the provenance breadcrumb.

    Returns ``"unknown"`` (explicitly surfaced in the breadcrumb, never a
    swallowed error) when *work_root* is not a git checkout — e.g. a
    ``--repo-root`` override pointing at a bare fixture tree. The breadcrumb
    is diagnostic only; the diff itself still fails loud in
    :func:`compute_touched`.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(work_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="main", help="diff base (default: main)")
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "checkout-root override (default: the invoking checkout's git toplevel — "
            "run from the issue worktree at Step 9c)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON object")
    parser.add_argument(
        "--map-files",
        default=None,
        metavar="FILE",
        help=(
            "newline-delimited repo-relative paths: print one "
            "'scan_test<TAB>matched_path' line per GLOB_SCAN_TESTS hit and exit "
            "(the /issue Step 10d merge-gate mapping mode, #1147 — skips the "
            "diff-based selection entirely; empty stdout on no match is a "
            "SUCCESS, the gate's skip signal)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        work_root = _resolve_work_root(args.repo_root)
    except subprocess.CalledProcessError as exc:
        # Fail loud with ONE readable line (not a traceback): no git toplevel
        # means the helper was invoked outside any git checkout, so no work
        # root can be resolved.
        detail = (exc.stderr or "").strip() or str(exc)
        print(
            f"select_step9c_tests: cannot resolve the work root ({detail}). "
            "Run from a git checkout or pass --repo-root.",
            file=sys.stderr,
        )
        return 1

    # Provenance breadcrumb on EVERY run: records which checkout + branch the
    # subset was selected against, so a wrong-worktree invocation is visible in
    # the Step 9c marker instead of silent (#851).
    print(
        f"select_step9c_tests: work root {work_root} (branch: {_current_branch(work_root)})",
        file=sys.stderr,
    )

    if args.map_files is not None:
        # Mapping mode (#1147): pure GLOB_SCAN_TESTS lookup over an explicit
        # file list — no git diff, no invariant set, no timeout sizing. The
        # Step 10d merge gate consumes the tab-separated stdout; empty output
        # + exit 0 means "no scan-covered payload" (the gate skips its test
        # leg). Only an unreadable input file is an error (exit 1) — the gate
        # must fail CLOSED when it cannot classify the payload.
        try:
            raw = Path(args.map_files).read_text()
        except OSError as exc:
            print(
                f"select_step9c_tests: cannot read --map-files input: {exc}",
                file=sys.stderr,
            )
            return 1
        files = [line.strip() for line in raw.splitlines() if line.strip()]
        pairs = map_scan_tests(files, work_root)
        for scan_test, f in sorted(_scan_pairs(files) - set(pairs)):
            print(
                f"select_step9c_tests: WARN — scan test {scan_test} (matched by {f}) "
                f"absent from {work_root}; pair dropped",
                file=sys.stderr,
            )
        for scan_test, f in pairs:
            print(f"{scan_test}\t{f}")
        return 0

    try:
        touched = compute_touched(args.base, work_root)
    except subprocess.CalledProcessError as exc:
        # Fail loud — never silently fall back to zero tests on a git error.
        print(f"select_step9c_tests: git diff failed: {exc}", file=sys.stderr)
        return 1

    if not touched:
        # Loud, exit-0 NOTE (the documented degenerate fallback stays legitimate
        # when the checkout genuinely has no commits ahead of the base): the
        # #851 failure was a SILENT invariant-only fallback from the wrong cwd.
        print(
            f"select_step9c_tests: NOTE — empty diff vs '{args.base}' in {work_root}; "
            "falling back to the workflow-invariant set only. If this task's changes "
            "live in an issue worktree, re-run from that worktree (Step 9c contract).",
            file=sys.stderr,
        )

    tests, untested, reasons = select_tests_with_reasons(touched, work_root)
    missing = missing_invariants(work_root)

    # Fail loud on an EMPTY selection (defense-in-depth beside the Step 9c shell
    # guard against a silent test-gate pass). WORKFLOW_INVARIANT has 32 always-on
    # entries, so an empty list can only mean the work root resolved wrong (e.g.
    # invoked from a directory outside the repo, or a bad --repo-root override)
    # or the invariant files all vanished — either way the gate would run zero
    # tests. Never let that surface as an exit-0 "no tests ran".
    if not tests:
        print(
            "select_step9c_tests: EMPTY test selection — work root likely resolved "
            f"wrong ({work_root}) or WORKFLOW_INVARIANT files are missing "
            f"(missing={missing}). Refusing to emit a zero-test gate command.",
            file=sys.stderr,
        )
        return 1

    for t in missing:
        print(f"WORKFLOW-INVARIANT MISSING: {t}", file=sys.stderr)

    # Gate-timeout sizing (#1046): a machine-greppable stderr line on every
    # run, the bound riding in the printed command, and the same number in the
    # --json fields — see the module docstring + recommended_timeout_s().
    timeout_s = recommended_timeout_s(tests)
    print(
        f"select_step9c_tests: {len(tests)} test files; recommended-timeout-s={timeout_s} "
        "(the gate exceeds the 600s foreground Bash tool cap — run it as a background "
        "invocation per Step 9c step 1b)",
        file=sys.stderr,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "tests": tests,
                    "untested_touched": untested,
                    "base": args.base,
                    "missing_invariants": missing,
                    "selection_reasons": reasons,
                    "n_tests": len(tests),
                    "recommended_timeout_s": timeout_s,
                    "slow_tests_selected": [t for t in tests if t in SLOW_TESTS],
                }
            )
        )
    else:
        print(
            f"timeout --kill-after=60s {timeout_s}s uv run pytest "
            + " ".join(tests)
            + " -v --tb=short"
        )
        for f in untested:
            print(f"untested touched file: {f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
