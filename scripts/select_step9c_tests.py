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
  * a code file (``scripts/<X>.py`` / ``src/.../<X>.py`` / any other ``*.py``
    — and, #1579, any ``*.sh``): map ``stem = Path(f).stem`` to
    ``tests/test_{stem}.py`` (exact) PLUS the broad ``tests/test_*{stem}*.py``
    glob. If NEITHER matches an existing test -> the file lands in
    ``untested_touched`` (a loud WARN the Step 9c marker surfaces; never a
    silent coverage hole).
  * ANY touched path matching a ``GLOB_SCAN_TESTS`` scan glob ADDITIONALLY
    selects that scanning test (tests that cover files via a directory scan
    at test time are reachable from no stem — #895). A map hit never
    suppresses the ``untested_touched`` WARN for the touched file itself.
  * any ``tests/**/test_*.py`` whose ``Import``/``ImportFrom`` nodes name a
    touched ``scripts/`` or ``src/explore_persona_space/`` module is
    ADDITIONALLY selected with reason ``import-map:<touched file>`` (#1299;
    founding incident #1286 — an importing test whose NAME shares no touched
    stem was reachable by no arm). Unlike a glob-scan hit, an import hit DOES
    mark the touched file tested (suppresses its ``untested_touched`` WARN):
    the test executes the touched module's code, strictly stronger evidence
    than the filename-substring stem glob that also sets ``matched``.
  * any ``tests/**/test_*.py`` whose RAW TEXT contains a touched ``scripts/``
    or ``src/explore_persona_space/`` ``.py`` file's repo-relative path as a
    literal substring is ADDITIONALLY selected with reason
    ``literal-path:<touched file>`` (#1498 — the pinning-test shape: e.g.
    ``tests/test_ruff_policy.py`` hardcodes ``scripts/...`` paths in
    ``LIVE_WORKFLOW_HELPERS`` and lints them at test time, reachable from no
    stem / scan glob / import). Like a glob-scan hit, a literal-path hit does
    NOT mark the touched file tested (a pinning test asserts an invariant
    ABOUT the file, not the file's own logic), so ``untested_touched`` WARNs
    still fire.
  * any ``tests/**/test_*.py`` whose raw text mentions the BASENAME of a
    touched ``.claude/rules/*.md`` file (e.g. ``llm-judging.md``) is
    ADDITIONALLY selected with reason ``rules-pin:<touched file>`` (#1496).
    DISCOVERED at selection time, not pinned — a new prose-pin test gates its
    rule the moment it lands, with no selector edit. Additive only: the rules
    file itself keeps the WORKFLOW_SURFACE skip (never ``untested_touched``),
    comment/docstring mentions count (over-selection is the safe direction),
    and a dynamically constructed filename is the accepted miss class.
  * any test registered in :data:`TRANSITIVE_CONSUMER_TESTS` for a touched
    file is ADDITIONALLY selected with reason
    ``transitive-consumer:<touched file>`` (#1589). A pinned literal, NOT
    discovered: these tests consume the touched module TRANSITIVELY (an
    ``importlib`` load by CONSTRUCTED path inside a helper, a path-join
    literal), so no text-scan arm can reach them. Additive only — never sets
    ``matched``, so ``untested_touched`` WARNs still fire.

``safe-by-direction``: the broad ``*{stem}*`` glob arm deliberately OVER-matches
on short stems (e.g. stem ``gcp`` selects ``test_gcp_backend.py``) — running a
few extra invariant-adjacent tests is the safe failure direction. There is no
``else: tighten the glob`` arm: a missed mapping surfaces as an
``untested_touched`` WARN, never a silently-dropped file.

Import-map scope + cost (#1299): the arm resolves absolute ``import X`` /
``from M import a`` forms over the WHOLE file (``ast.walk`` — function-level
imports count; the founding test imports 2/3 of its targets inside test
functions). Out of scope, each miss landing in the ``untested_touched`` WARN
(the pre-existing loud fallback): relative imports (``node.level > 0``
skipped); dynamic imports (``importlib`` / ``spec_from_file_location`` — both
live dynamic consumers of THIS selector are covered anyway:
``tests/test_select_step9c_tests.py`` is stem-mapped and
``tests/test_diff_base_origin_main_pin.py`` is WORKFLOW_INVARIANT-pinned);
attribute access on a parent package (``import explore_persona_space`` then
``.task_workflow``); modules outside ``scripts/`` +
``src/explore_persona_space/``; ``conftest.py`` imports (only ``test_*.py``
files are scanned); transitive imports (a test importing helper H is not
selected for a diff touching a module H imports — one hop only). Known miss
class: touching a package ``__init__.py`` resolves to the PACKAGE name only,
so importers of its submodules are not matched. Deliberately counted as hits
(over-selection is the safe direction): imports under ``TYPE_CHECKING`` /
try-except guards / any conditional, and a false-positive flat-stem match (a
test importing a PyPI package whose name equals a touched ``scripts/`` stem).
Cost: a substring pre-filter parses only candidate-bearing files (zero file
reads on a workflow-surface-only diff; a short / colliding stem parses extra
files, bounded by the measured ~4-8 s whole-tree worst case), and import-map
selections flow into :func:`recommended_timeout_s` automatically (it sizes on
``len(tests)``).

Literal-path scope + cost (#1498): the arm shares the import arm's single
read pass (:func:`_scan_test_files`) — zero ADDITIONAL file reads on any diff
that already triggered the import scan, and a workflow-surface-only diff
still reads nothing (both arms' target sets empty -> early return). A diff
whose only eligible ``.py`` maps to no module name (``scripts/__init__.py``)
now triggers the read pass, with ZERO ast parses (empty token pre-filter).
Raw substring on file text is deliberate: comment / docstring mentions
over-select in the safe direction (safe-by-direction, above). ``.sh`` under
``scripts/`` | ``src/explore_persona_space/`` | ``.claude/hooks/`` is IN
scope as of #1579 (:func:`literal_path_targets`'s ``.sh`` branch — the
guard-hook pin shape, e.g. a test hardcoding
``scripts/guard_repo_root_branch.sh``). Extendable miss classes, out of
scope for now — an uncovered eligible ``.py``/``.sh`` still lands in the
``untested_touched`` WARN, while a non-code-suffix pin target sits outside
the code-file mapping entirely: constructed paths
(``Path("scripts") / "x.py"``, f-strings with variable parts, multi-line
string concatenation); ``conftest.py``-resident pin lists (the scan is
``rglob("test_*.py")``, matching the import arm's scope); non-code-suffix
pin targets (a hardcoded data/config path); and OUT-OF-PREFIX ``.sh`` in
MAPPING mode — an ``external/foo.sh`` / root-level ``launch_*.sh`` payload
still yields an empty ``--map-files`` output with NO #1573 WARN (the floor
iterates :func:`literal_path_targets`-eligible files only), while diff mode
stem-maps/WARNs it; byte-symmetric with ``.py``'s ``external/x.py`` today,
named so a future filing does not rediscover it as "the #1579 fix didn't
work".

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

Diff-base resolution (#1289): the default ``--base`` is FETCHED ``origin/main``
(the shared repo-root ``main`` can lag origin — 2026-07-12: foreign-file
pollution inflated #1281's gate to 41 files). :func:`resolve_base` runs a
bounded best-effort ``git fetch origin main`` (``FETCH_TIMEOUT_S``; skip with
``--no-fetch``), degrades to the last-fetched ``origin/main`` on fetch failure,
and falls back LOUDLY to local ``main`` only when ``origin/main`` does not
resolve at all (offline clone / no origin remote / non-git fixture tree). A
base without an ``origin/`` prefix is used verbatim with no git calls —
``--base main`` is the pre-#1289 escape hatch. Branches are cut from fetched
``origin/main`` (#1214), so ``merge-base(origin/main, HEAD)`` is the branch cut
point and the three-dot selection is stable under ``origin/main`` advancing.

Usage::

    uv run python scripts/select_step9c_tests.py [--base origin/main] [--no-fetch] \
                                                 [--repo-root <path>] [--json]
    uv run python scripts/select_step9c_tests.py --map-files <file> [--repo-root <path>]

``--map-files FILE`` (the ``/issue`` Step 10d merge-gate mapping mode, #1147):
read newline-delimited repo-relative paths from FILE and print one
``<test>\\t<matched_path>`` line per :data:`GLOB_SCAN_TESTS` hit
(:func:`map_scan_tests`) PLUS one ``<pin_test>\\t<rule_path>`` line per
rules-pin discovery hit on a ``.claude/rules/*.md`` payload file
(:func:`rules_pin_pairs`, #1496 — WORKFLOW_INVARIANT members excluded) PLUS
the src/scripts dependency arms (:func:`dependency_map_pairs`, #1573 —
import-map (#1299) + literal-path (#1498) + stem-map pairs for ``.py``/``.sh``
payloads (``.sh`` #1579) under the :func:`literal_path_targets` eligibility
prefixes, WORKFLOW_INVARIANT members excluded) PLUS one
``<consumer_test>\\t<touched_path>`` line per pinned transitive-consumer
registration (:func:`transitive_consumer_pairs`, #1589 — WORKFLOW_INVARIANT
members excluded; a registered consumer absent from the work root is
dropped, the live-tree drift pins make staleness loud on main), skipping the
diff-based selection
entirely (mapping mode never runs git — no fetch). A scan
test absent from the work root is dropped with a stderr WARN; an eligible
src/scripts code file with ZERO pairs across all arms draws one tab-free
``no mapped tests for code file`` stderr WARN (#1573's fail-loud floor —
rc stays 0). A non-empty map additionally prints a tab-free machine-greppable
``recommended-timeout-s=<T>`` stderr sizing line
(``recommended_timeout_s(tests, floor=MAP_TIMEOUT_FLOOR_S)``, floor 300 s —
the Step-10d TG legs size their pytest bound from it). Empty stdout on
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
``glob-scan:<touched file>`` / ``import-map:<touched file>`` /
``literal-path:<touched file>`` / ``rules-pin:<touched file>`` /
``transitive-consumer:<touched file>`` —
#1022, #1299, #1498, #1496, #1589).
Exit 0 on success (even with WARN lines);
exit 1 if an underlying ``git`` call fails irrecoverably (work-root resolution
or the diff) or if the selection comes back EMPTY (the zero-test-gate
refusal).
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# --- Pinned workflow-invariant tests. ----------------------------------------
# A module-level literal tuple, NOT a glob: a future ``tests/test_workflowish.py``
# that is NOT meant to gate Step 9c must not silently join the gate, and the gate
# must not silently shrink if a glob arm stops matching. Drift is made loud by the
# on-disk existence check (a vanished entry prints WORKFLOW-INVARIANT MISSING) and
# pinned against the live tree by tests/test_select_step9c_tests.py.
# REGISTERING A NEW PIN TEST (#1593 — deliberately NO count to bump anywhere):
# add ONE tuple entry below (at its group's position — prefer within-group
# alphabetical placement, which scatters concurrent insertions — with a one-line
# inline rationale comment) and ONE line at its sorted position in
# tests/step9c_workflow_invariant_manifest.txt. The manifest set-equality pin
# (tests/test_select_step9c_tests.py::test_workflow_invariant_matches_manifest)
# replaces the retired integer count pin, whose single shared line made every
# pair of same-window registering PRs merge-conflict (#1584).
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
    "tests/test_issue_skill_exit_breadcrumb.py",  # NEW (#1242) — SKILL.md exit-breadcrumb pin
    # NEW (#1575) — SKILL.md cap-park surfacing pins (#1548/#1558/#1575)
    "tests/test_issue_skill_followup_cap_park_note_pin.py",
    # NEW (#1546) — SKILL.md forensics-ingest pointer pin
    "tests/test_issue_skill_forensics_ingest_pointer.py",
    "tests/test_issue_skill_marker_contract.py",
    # NEW (#1268) — SKILL.md Step-10d repin/guard hardening pin
    "tests/test_issue_skill_merge_resnapshot_pin.py",
    # NEW (#1563) — SKILL.md orchestrator-turn discipline pointer pin
    "tests/test_issue_skill_orchestrator_turn_discipline_pointer.py",
    # NEW (#1572) — staged-index verification pin
    "tests/test_issue_skill_staged_index_verification.py",
    # NEW (#1587) — SKILL.md trigger-dense tag-adoption pin
    "tests/test_issue_skill_trigger_dense_tag_adoption.py",
    "tests/test_step10d_guard3.py",  # NEW (#1242) — SKILL.md Step 10d guard/merge pin
    "tests/test_step_completed_resume.py",  # NEW (#1242) — resume/step-completed contract pin
    "tests/test_plan_handoff_path_convention.py",
    "tests/test_clean_result_critic_planned_vs_actual.py",
    "tests/test_check_no_secret_shaped_strings.py",
    "tests/test_check_mcp_json_no_secrets.py",
    # NEW (#1584) — merge-scoped gitleaks stanza/wrapper pin
    "tests/test_precommit_gitleaks_merge_scope.py",
    "tests/test_diff_base_origin_main_pin.py",  # NEW (#1289) — diff-base origin/main pin
    "tests/test_fit_loop_batching_review_pin.py",  # NEW (#1397) — fit-loop batching review-lens pin
    # NEW (#1577) — guard-script read-bounding hook pin: the selector's
    # stem/literal/dependency arms are .py-only, so a later .sh-hook /
    # settings.json diff re-runs this pin ONLY via this tuple.
    "tests/test_guard_trigger_dense_read.py",
)

# --- Touched files that short-circuit (no per-file test map). ----------------
# These gate via the WORKFLOW_INVARIANT set, not a per-file test, so a touched
# file matching one of these is SKIPPED (and is NOT an "untested" file).
# .claude/rules/*.md files ADDITIONALLY map to their prose-pin tests via the
# rules-pin discovery arm (#1496) — additive; the skip itself is unchanged.
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
# NOTE (#1337): adding an entry here does NOT make it eligible for compare's
# scratch pristine oracle — step9c_baseline.py's R-F' rule refuses scan-set nodes
# by default; a scan test whose scan root is Path(__file__)-derived may be opted in
# via step9c_baseline.py::FILE_ANCHORED_SCAN_TESTS after source verification.
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
    # #1593: the WORKFLOW_INVARIANT manifest is READ by the pin test at test
    # time (a .txt takes the data-suffix skip, so no stem/literal/import arm
    # can reach the pin from a manifest-only diff); a touched manifest must
    # re-run the tuple<->manifest set-equality pin.
    "tests/test_select_step9c_tests.py": ("tests/step9c_workflow_invariant_manifest.txt",),
}

# --- Transitive-consumer pin map (#1589). -------------------------------------
# Some tests consume a scripts/ module TRANSITIVELY — through a helper that
# importlib-loads it BY PATH at runtime (step9c_baseline.load_selector_module
# builds root / "scripts" / "select_step9c_tests.py"), or via a path-join
# literal (_REPO_ROOT / "scripts" / "select_step9c_tests.py") — so NO
# text-scan arm can reach them: the import arm sees no Import node (dynamic
# loads; one-hop-only by contract), the literal arm needs the CONTIGUOUS
# repo-relative path, and the stem arm needs the module stem in the test's
# FILENAME. Founding incident #1589: a selector-module diff never selected
# tests/test_step9c_baseline.py (loads the LIVE selector in
# test_load_selector_module_real_body + the derive-pristine-timeout pins) or
# tests/test_inline_lint_gate.py (live-selector TIMEOUT_FLOOR_S parity pin),
# so a selector regression could break them without either running — at
# Step 9c touched scope AND the Step 10d TG / inline-payload mapped-test
# legs. A pinned literal, NOT discovered (the connecting evidence lives in
# scripts/, outside the tests/ scan surface); same curation rule as
# WORKFLOW_INVARIANT / GLOB_SCAN_TESTS — live-tree drift pins in
# tests/test_select_step9c_tests.py. Additive only: a hit never sets
# ``matched`` (untested_touched WARNs still fire — over-WARN is the safe
# direction), and WORKFLOW_INVARIANT members are excluded on the map legs
# (the rules_pin_pairs asymmetry).
TRANSITIVE_CONSUMER_TESTS: dict[str, tuple[str, ...]] = {
    "scripts/select_step9c_tests.py": (
        "tests/test_inline_lint_gate.py",
        "tests/test_step9c_baseline.py",
    ),
}

# --- Rules-pin discovery arm (#1496). -----------------------------------------
# ~29 tests read/pin .claude/rules/*.md PROSE at test time; most are not in
# WORKFLOW_INVARIANT, so a rules-only diff got no targeted pin coverage (the
# rules glob short-circuits at WORKFLOW_SURFACE_GLOBS). Unlike GLOB_SCAN_TESTS
# this arm is DISCOVERED, not pinned: the founding bug IS pinned-map staleness
# (a new pin test added outside the invariant list silently stops gating its
# rule), a wrong join costs only extra tests on THAT rule's diffs
# (safe-by-direction, module docstring), and the import-map arm (#1299) is the
# established discovery precedent. Matching token: the rule's BASENAME
# ("llm-judging.md") as a raw-text substring — covers full-path literals,
# path-join forms ((ROOT / ".claude" / "rules" / "x.md"), the
# test_battery_basis_prose_pins.py shape), and comment/docstring mentions
# (deliberately counted; over-selection is the safe direction). Substring
# semantics also over-select on SUPERSTRING basenames: touching
# critic-lens-reference.md selects tests mentioning only
# clean-result-critic-lens-reference.md (the former is a substring of the
# latter) — accepted over-select, NOT a scan bug; pinned by the
# superstring fixture test in tests/test_select_step9c_tests.py. Known-miss
# class (accepted): a test that constructs the filename dynamically. Additive
# only — never sets ``matched``, never enters untested_touched (rules files
# are workflow-surface SKIPs by design; that semantics is unchanged). Cost:
# one raw-substring pass over tests/**/test_*.py (587 files today) ONLY when
# a rules file is touched — no AST, well under the import-map arm's measured
# 4-8 s AST worst case. Measured fan-out (2026-07-18): worst rule
# gotchas.md -> 17 test files (15 non-invariant); next code-style.md -> 9,
# critic-lens-reference.md -> 7.
# Scan-regression loudness: tests/test_select_step9c_tests.py
# ::test_rules_pin_live_tree_known_pairs pins known (rule -> test) pairs.
_RULES_GLOB = ".claude/rules/*.md"


def rules_pin_hits(touched: list[str], work_root: Path) -> dict[str, set[str]]:
    """``{test_relpath: {touched .claude/rules/*.md files whose basename its
    text mentions}}`` (#1496). Zero file reads when no rules file is touched.

    Fail-soft (the #1299 read contract): a test file that cannot be read /
    decoded emits ONE stderr WARN and is skipped — never crashes the selector;
    read failures on >5% of scanned files add ONE aggregate WARN (mirrors the
    import-map arm's systemic-breakage signal). ``rglob`` only yields existing
    files, so no separate existence filter.
    """
    rules = [f for f in touched if fnmatch.fnmatch(f, _RULES_GLOB)]
    hits: dict[str, set[str]] = {}
    tests_dir = work_root / "tests"
    if not rules or not tests_dir.is_dir():
        return hits
    tokens = {r: Path(r).name for r in rules}
    n_scanned = 0
    n_failed = 0
    for test_path in sorted(tests_dir.rglob("test_*.py")):
        n_scanned += 1
        rel = test_path.relative_to(work_root).as_posix()
        try:
            text = test_path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            n_failed += 1
            print(
                f"select_step9c_tests: WARN — rules-pin scan cannot read {rel}: {exc}; "
                "file skipped for the rules-pin arm",
                file=sys.stderr,
            )
            continue
        for rule, token in tokens.items():
            if token in text:
                hits.setdefault(rel, set()).add(rule)
    if n_scanned and n_failed / n_scanned > 0.05:
        print(
            f"select_step9c_tests: WARN — rules-pin scan read failures on "
            f"{n_failed}/{n_scanned} scanned test files (>5%): systemic tests/ breakage; "
            "the rules-pin arm may under-select",
            file=sys.stderr,
        )
    return hits


def rules_pin_pairs(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Sorted ``(pin_test, rule_file)`` pairs for ``--map-files`` (#1496).

    WORKFLOW_INVARIANT members are EXCLUDED here (they already gate every
    Step 9c run; this also keeps the 900 s tests/test_workflow_lint.py out of
    the Step 10d / inline-payload lint gate). The Step 9c selection arm keeps
    them (harmless extra reason; the union dedupes) — the deliberate asymmetry
    is pinned by test_cli_map_files_rules_pin_excludes_invariant.
    """
    inv = set(WORKFLOW_INVARIANT)
    return sorted(
        (t, r)
        for t, rule_files in rules_pin_hits(files, work_root).items()
        if t not in inv
        for r in sorted(rule_files)
    )


# --- src/scripts dependency arms for --map-files (#1573). ---------------------
MAP_TIMEOUT_FLOOR_S = 300  # Step-10d TG-leg parity floor (SKILL.md measured basis
#                            ~12.6 s for the historical 2-test scan map, 2026-07-08)


def dependency_map_pairs(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Sorted ``(test, matched_path)`` pairs for ``--map-files`` beyond
    GLOB_SCAN_TESTS + rules-pin (#1573): import-map (#1299) + literal-path
    (#1498) via the ONE shared :func:`_scan_test_files` read pass, plus
    stem-map path arithmetic. WORKFLOW_INVARIANT members are EXCLUDED
    (the :func:`rules_pin_pairs` asymmetry: they already gate every Step 9c
    run, and exclusion keeps the 900 s tests/test_workflow_lint.py out of the
    Step 10d / inline-payload gates — sft.py literal-hits it via
    workflow_lint's LIVE_WORKFLOW_HELPERS list). The stem arm is restricted
    to the :func:`literal_path_targets` eligibility set (``.py`` under
    ``scripts/`` | ``src/explore_persona_space/``; plus, #1579, ``.sh``
    under those prefixes or ``.claude/hooks/``) — it is the closing
    mechanism for the dynamic-import (``pytest.importorskip`` / ``importlib``)
    getsource subclass, where the test file is stem-named after the module,
    and (#1579) for ``.sh`` scripts, which are never importable at all.
    """
    inv = set(WORKFLOW_INVARIANT)
    import_hits, _tested, literal_hits = _scan_test_files(files, work_root)
    pairs: set[tuple[str, str]] = set()
    for hits in (import_hits, literal_hits):
        for t, fs in hits.items():
            if t not in inv:
                pairs.update((t, f) for f in fs)
    for f in literal_path_targets(files):  # the same eligibility predicate
        stem = Path(f).stem
        exact = f"tests/test_{stem}.py"
        if (work_root / exact).exists() and exact not in inv:
            pairs.add((exact, f))
        for hit in sorted((work_root / "tests").glob(f"test_*{stem}*.py")):
            rel = f"tests/{hit.name}"
            if rel not in inv:
                pairs.add((rel, f))
    return sorted(pairs)


def transitive_consumer_pairs(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Sorted ``(consumer_test, touched_file)`` pairs for ``--map-files`` (#1589).

    WORKFLOW_INVARIANT members are EXCLUDED (the :func:`rules_pin_pairs` /
    :func:`dependency_map_pairs` asymmetry — they already gate every Step 9c
    run and must stay out of the Step 10d / inline gates); a registered test
    absent from *work_root* is dropped (live-tree drift pins in
    tests/test_select_step9c_tests.py make staleness loud on main).
    """
    inv = set(WORKFLOW_INVARIANT)
    return sorted(
        (t, f)
        for f in files
        for t in TRANSITIVE_CONSUMER_TESTS.get(f, ())
        if t not in inv and (work_root / t).exists()
    )


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

# --- Diff-base resolution (#1289). -------------------------------------------
# The always-shared repo-root `main` can lag origin/main (unpushed/conflicted
# root state): on 2026-07-12 three concurrent sessions got foreign-file
# pollution from `--base main` (#1280: 202,578-byte diff vs 11,637 against
# origin/main; #1281: a 41-test-file gate). Default the base to FETCHED
# origin/main: branches are cut from fetched origin/main (new_worktree.sh,
# #1214), and merge-base(origin/main, HEAD) == the cut point — advancing
# origin/main never moves it, so the three-dot selection is stable.
DEFAULT_BASE = "origin/main"
# Bounded fetch — SKILL.md Step 10d lint-gate precedent
# (`timeout --kill-after=30s 120s git fetch origin main --quiet || true`):
FETCH_TIMEOUT_S = 120


def resolve_base(base: str, work_root: Path, *, fetch: bool = True) -> str:
    """Resolve the diff base ref (#1289) — never blocks, never raises.

    A *base* without an ``origin/`` prefix is returned VERBATIM with no git
    calls (the explicit local-ref escape hatch; ``--base main`` == pre-#1289
    behavior). A remote-tracking *base*: (1) best-effort bounded
    ``git fetch origin <branch> --quiet`` (failure / timeout / offline ->
    stderr NOTE, degrade to the last-fetched ref — the Guard-1 precedent;
    a stale origin/main still names the branch cut point, see module
    docstring); (2) ``git rev-parse --verify --quiet <base>`` — resolves ->
    use *base*; does NOT resolve (offline clone with no remote-tracking ref,
    checkout without an ``origin`` remote, non-git ``--repo-root`` fixture
    tree) -> loud stderr NOTE + fall back to the local ``<branch>`` (pre-#1289
    behavior; fail toward current behavior).
    """
    if not base.startswith("origin/"):
        return base
    branch = base.split("/", 1)[1]
    if fetch:
        try:
            subprocess.run(
                ["git", "fetch", "origin", branch, "--quiet"],
                cwd=str(work_root),
                capture_output=True,
                text=True,
                timeout=FETCH_TIMEOUT_S,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            print(
                f"select_step9c_tests: NOTE — git fetch origin {branch} failed ({exc}); "
                f"using last-fetched {base} if it resolves",
                file=sys.stderr,
            )
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", base],
            cwd=str(work_root),
            capture_output=True,
            text=True,
        )
        resolved = probe.returncode == 0
    except OSError:
        # A non-git / vanished work_root cannot resolve the remote ref either;
        # take the same loud local-branch fallback (never raise — the caller's
        # own git diff stays the fail-loud surface).
        resolved = False
    if resolved:
        return base
    print(
        f"select_step9c_tests: NOTE — {base} does not resolve in {work_root}; "
        f"falling back to local '{branch}' (offline clone / no origin remote — "
        "pre-#1289 behavior)",
        file=sys.stderr,
    )
    return branch


def recommended_timeout_s(tests: list[str], *, floor: int = TIMEOUT_FLOOR_S) -> int:
    """Deterministic `timeout(1)` bound for a Step 9c gate selection.

    ``BASE + PER_FILE * len(tests) + sum(slow surcharges)``, floored at
    *floor* (default ``TIMEOUT_FLOOR_S`` — diff-path callers unchanged;
    ``--map-files`` mode passes ``floor=MAP_TIMEOUT_FLOOR_S``, the Step-10d
    TG-leg 300 s parity floor, #1573). Invariant-only selection (38 files
    incl. the workflow-lint surcharge) -> 2160 s (36 min), consistent with
    the existing invariant-set-scale precedents (``step9c_baseline.py
    refresh`` ``--timeout-s`` default 1800 s; the SKILL.md detached
    refresh's 2100 s).
    """
    t = TIMEOUT_BASE_S + TIMEOUT_PER_FILE_S * len(tests)
    t += sum(SLOW_TESTS.get(x, 0) for x in tests)
    return max(t, floor)


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


# --- Import-map arm (#1299). ---------------------------------------------------
# Founding incident #1286: tests/test_issue810_uh_pack_validation.py imports
# issue810_common but shares no filename stem with it, so NO arm selected it and
# 4/5 touched scripts landed in the untested_touched WARN. The import arm selects
# any tests/**/test_*.py whose Import/ImportFrom nodes name a touched module, so
# touched-module coverage no longer depends on test-file NAMING.


def touched_module_names(touched: list[str]) -> dict[str, set[str]]:
    """Importable module names -> the touched files that resolve to them (#1299).

    Eligible touched files: suffix ``.py``, under ``scripts/`` or
    ``src/explore_persona_space/`` (never ``tests/`` — the touched-test arm owns
    those). Resolution::

        scripts/X.py             -> {"X", "scripts.X"}   (flat via the
                                     sys.path.insert pattern — the #1286 shape,
                                     174 test files; dotted — scripts/ is a
                                     package, ~25 test files)
        scripts/pkg/__init__.py  -> {"pkg", "scripts.pkg"}
        src/explore_persona_space/a/b.py        -> {"explore_persona_space.a.b"}
        src/explore_persona_space/a/__init__.py -> {"explore_persona_space.a"}

    Multi-maps on collision: two touched files resolving one name are both
    listed under it. ``scripts/__init__.py`` itself maps to no flat name (the
    dotted ``scripts`` package marker carries no per-module signal).
    """
    out: dict[str, set[str]] = {}

    def _record(name: str, f: str) -> None:
        out.setdefault(name, set()).add(f)

    for f in touched:
        p = Path(f)
        if p.suffix != ".py":
            continue
        parts = p.parts
        tail = () if p.name == "__init__.py" else (p.stem,)
        if parts[0] == "scripts" and len(parts) > 1:
            rel = parts[1:-1] + tail
            if not rel:
                continue  # scripts/__init__.py: the package marker itself
            flat = ".".join(rel)
            _record(flat, f)
            _record(f"scripts.{flat}", f)
        elif parts[:2] == ("src", "explore_persona_space"):
            rel = parts[1:-1] + tail  # keeps the leading explore_persona_space
            _record(".".join(rel), f)
    return out


def _import_names(tree: ast.AST) -> set[str]:
    """Every dotted name any ``Import``/``ImportFrom`` node references, ANYWHERE
    in the file (``ast.walk`` — function-level imports count: the founding test
    imports 2 of its 3 target modules inside test functions, #1286 L138/L178).

    ``import X.Y as z`` -> ``{"X.Y"}``; ``from M import a, b`` -> ``{"M",
    "M.a", "M.b"}`` (the ``M.a`` join catches ``from
    explore_persona_space.foo import bar`` naming touched module
    ``...foo.bar``). Relative imports (``node.level > 0``) are SKIPPED —
    ``tests/`` is not a package here, so a relative form cannot name a
    ``scripts/`` / ``src/`` module.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


# .sh eligibility prefixes for the literal/stem map arms (#1579): workflow shell
# scripts live under scripts/ and .claude/hooks/ (6 of the 8 guard suites map to
# hook scripts there); src/ included for symmetry with the .py branch (no tracked
# .sh today — zero current cost, no future gap).
_SH_ELIGIBLE_PREFIXES: tuple[str, ...] = (
    "scripts/",
    "src/explore_persona_space/",
    ".claude/hooks/",
)


def literal_path_targets(touched: list[str]) -> set[str]:
    """Touched files eligible for the literal-path arm (#1498; ``.sh`` joined #1579).

    Eligible: ``.py`` code files under ``scripts/`` or
    ``src/explore_persona_space/`` (UNCHANGED — byte-identical to the #1498
    predicate), plus ``.sh`` scripts under ``scripts/``,
    ``src/explore_persona_space/``, or ``.claude/hooks/`` (#1579 — guard
    hooks / workflow shell scripts whose pinned suites hardcode the script
    path or stem-name themselves after it). Never ``tests/`` (the
    touched-test arm owns those; the ``.md`` / workflow-surface pin mapping
    is deliberately out of scope here — #1496's surface). Each returned
    repo-relative path is matched as a raw substring of every scanned test
    file's text; this set is ALSO the stem-arm eligibility for
    ``--map-files`` (:func:`dependency_map_pairs`) and the #1573 zero-mapped
    WARN floor.
    """
    out: set[str] = set()
    for f in touched:
        if f.startswith("tests/"):
            continue
        suffix = Path(f).suffix
        if (
            suffix == ".py"
            and (f.startswith("scripts/") or f.startswith("src/explore_persona_space/"))
        ) or (suffix == ".sh" and f.startswith(_SH_ELIGIBLE_PREFIXES)):
            out.add(f)
    return out


def _scan_test_files(
    touched: list[str], work_root: Path
) -> tuple[dict[str, set[str]], set[str], dict[str, set[str]]]:
    """ONE shared read pass over ``tests/**/test_*.py`` for both text-scan arms.

    Returns ``(import_hits, import_tested, literal_hits)``:
      * ``import_hits`` — ``{test_relpath: {touched files it imports}}``
        (#1299, the import-map arm);
      * ``import_tested`` — touched files with >= 1 importing test (these
        suppress the ``untested_touched`` WARN);
      * ``literal_hits`` — ``{test_relpath: {touched files whose repo-relative
        path appears as a raw substring of the test's text}}`` (#1498, the
        literal-path arm; never suppresses the WARN).

    Subdir tests (``tests/experiments/``) are in scope (sorted ``rglob`` —
    pytest collects them and the touched-test arm already admits their paths).

    Cost bound: when BOTH target sets are empty (workflow-surface-only diff)
    the pass returns immediately with ZERO file reads. Otherwise each test
    file's raw text is read ONCE and (a) checked against each literal target
    as a plain substring, then (b) ``ast``-parsed ONLY when its text contains
    the last dotted component of at least one candidate module name — any
    absolute import of module M must literally spell M's last component, so
    the filter is over-inclusive (never a false negative). A literal-only
    trigger (e.g. ``scripts/__init__.py``, which maps to no module name)
    reads the tree with ZERO parses (empty token pre-filter); a ``.sh``-only
    diff is the same cost class (#1579 — ``touched_module_names`` stays
    ``.py``-only, so ``.sh`` paths join ``literal_targets`` with an empty
    token set: one raw-text pass, zero AST parses). Typical touched
    sets parse a handful of files; worst case the whole tree (~500 files,
    measured ~4-8 s under shared-VM load — module docstring).

    Fail-soft (#1299 rationale R5): a file whose RAW READ fails (``OSError``,
    ``ValueError`` incl. ``UnicodeDecodeError``) emits ONE stderr WARN and is
    skipped for BOTH arms; a file that reads but fails to PARSE
    (``SyntaxError``, ``ValueError``) is skipped for the import arm ONLY —
    its literal hits are kept (raw text already read; an additive
    improvement, #1498). Never crashes the selector; if the broken file is
    itself part of the diff the touched-test arm still selects it (pytest
    collection then fails loud at the right surface). When read + parse
    failures exceed 5% of the scanned files, ONE additional aggregate WARN
    flags the systemic tests/ breakage.
    """
    import_hits: dict[str, set[str]] = {}
    tested: set[str] = set()
    literal_hits: dict[str, set[str]] = {}
    module_map = touched_module_names(touched)
    literal_targets = literal_path_targets(touched)
    if not module_map and not literal_targets:
        return import_hits, tested, literal_hits
    tests_dir = work_root / "tests"
    if not tests_dir.is_dir():
        return import_hits, tested, literal_hits
    tokens = {name.rsplit(".", 1)[-1] for name in module_map}
    n_scanned = 0
    n_failed = 0
    for test_path in sorted(tests_dir.rglob("test_*.py")):
        n_scanned += 1
        rel = test_path.relative_to(work_root).as_posix()
        try:
            text = test_path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            n_failed += 1
            print(
                f"select_step9c_tests: WARN — import-map cannot parse {rel}: {exc}; "
                "file skipped for the import and literal-path arms",
                file=sys.stderr,
            )
            continue
        for f in literal_targets:
            if f in text:
                literal_hits.setdefault(rel, set()).add(f)
        if not any(tok in text for tok in tokens):
            continue
        try:
            imported = _import_names(ast.parse(text))
        except (SyntaxError, ValueError) as exc:
            n_failed += 1
            print(
                f"select_step9c_tests: WARN — import-map cannot parse {rel}: {exc}; "
                "file skipped for the import arm (literal-path hits on its raw text, "
                "if any, are kept)",
                file=sys.stderr,
            )
            continue
        for name, files in module_map.items():
            if name in imported:
                import_hits.setdefault(rel, set()).update(files)
                tested.update(files)
    if n_scanned and n_failed / n_scanned > 0.05:
        print(
            f"select_step9c_tests: WARN — test-scan read/parse failures on "
            f"{n_failed}/{n_scanned} scanned test files (>5%): systemic tests/ breakage; "
            "the import and literal-path arms may under-select",
            file=sys.stderr,
        )
    return import_hits, tested, literal_hits


def import_map_hits(touched: list[str], work_root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Back-compat wrapper over :func:`_scan_test_files` (the #1299 public shape).

    Returns ``(import_hits, import_tested)``, dropping the literal-path
    element. NOTE (#1498): the underlying shared pass also triggers on
    literal-eligible touched files, so a ``scripts/__init__.py``-only call
    now scans (with zero parses) where the pre-refactor arm returned early
    with zero reads; the returned import elements are unchanged.
    """
    hits, tested, _ = _scan_test_files(touched, work_root)
    return hits, tested


def _seed_import_reasons(import_hits: dict[str, set[str]]) -> dict[str, set[str]]:
    """Initial ``{test: reasons}`` mapping seeded from import-map hits (#1299)."""
    return {t: {f"import-map:{f}" for f in files} for t, files in import_hits.items()}


def _seed_scan_reasons(
    import_hits: dict[str, set[str]], literal_hits: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Initial ``{test: reasons}`` mapping seeded from BOTH text-scan arms.

    Import-map hits (#1299) plus literal-path hits (#1498) — a test hit by
    both carries both reason kinds. Purely additive: only ever adds
    tests/reasons to the seed.
    """
    selected = _seed_import_reasons(import_hits)
    for lit_test, lit_files in literal_hits.items():
        for lit_f in lit_files:
            selected.setdefault(lit_test, set()).add(f"literal-path:{lit_f}")
    return selected


def _add_glob_scan_reasons(f: str, work_root: Path, add) -> None:
    """Glob-scan invariant arm (#895): additive, never marks ``f`` tested."""
    for scan_test, scan_globs in GLOB_SCAN_TESTS.items():
        if _matches_any(f, scan_globs) and (work_root / scan_test).exists():
            add(scan_test, f"glob-scan:{f}")


def select_tests_with_reasons(
    touched: list[str], work_root: Path
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Superset of :func:`select_tests`: also returns ``{test: sorted reasons}``.

    A reason is ``'invariant' | 'touched-test' | 'stem-map:<touched file>' |
    'glob-scan:<touched file>' | 'import-map:<touched file>' |
    'literal-path:<touched file>' | 'rules-pin:<touched file>' |
    'transitive-consumer:<touched file>'``; a test may
    carry several (e.g. an invariant that is also stem-mapped from a touched
    file). Selection behavior is IDENTICAL to :func:`select_tests` — same
    tests, same ``untested_touched`` WARN list, same sorted ordering (#1022:
    the reasons feed ``step9c_baseline.py compare``'s diff-linked-ness read,
    which must come from the SAME mapping logic that selected the run's tests).
    """
    # Text-scan arms (ONE shared read pass, _scan_test_files): the import-map
    # arm (#1299) seeds the selection and the literal-path arm (#1498) adds
    # pinning-test hits — both additive by construction (they only ever add
    # tests/reasons here and, via the ``matched`` seed below, the IMPORT arm
    # alone removes entries from the untested WARN list; no code path drops a
    # stem-map / glob-scan / invariant / touched-test selection, so selection
    # only GROWS). Ordering is irrelevant: the terminal sorted() reads below
    # keep the output deterministic.
    import_hits, import_tested, literal_hits = _scan_test_files(touched, work_root)
    selected: dict[str, set[str]] = _seed_scan_reasons(import_hits, literal_hits)
    # Rules-pin discovery arm (#1496): additive seed, same only-grows contract
    # (one scan pass serves all touched rules; the rules file itself still
    # takes the WORKFLOW_SURFACE `continue` below, unchanged).
    for t, rule_files in rules_pin_hits(touched, work_root).items():
        selected.setdefault(t, set()).update(f"rules-pin:{r}" for r in rule_files)
    # Transitive-consumer pin arm (#1589): additive seed, same only-grows
    # contract; never sets ``matched`` (the glob-scan/literal precedent —
    # over-WARN is the safe direction). Invariant members are KEPT here
    # (harmless extra reason; the union dedupes — the rules-pin asymmetry).
    for f in touched:
        for t in TRANSITIVE_CONSUMER_TESTS.get(f, ()):
            if (work_root / t).exists():
                selected.setdefault(t, set()).add(f"transitive-consumer:{f}")
    untested: list[str] = []

    def _add(test: str, reason: str) -> None:
        selected.setdefault(test, set()).add(reason)

    for f in touched:
        # Glob-scan invariant tests (#895): additive, never sets ``matched``.
        _add_glob_scan_reasons(f, work_root, _add)
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
        # Code files (.py / .sh anywhere): map stem -> test_<stem>.py +
        # test_*<stem>*.py. (#1579: .sh joins — guard hooks + workflow shell
        # scripts have stem-named pinned suites, e.g.
        # scripts/guard_repo_root_branch.sh -> tests/test_guard_repo_root_branch.py;
        # previously silently ignored, not even untested_touched-WARNed.)
        if p.suffix in (".py", ".sh"):
            stem = p.stem
            # An import-map hit marks the touched file tested (#1299): the
            # importing test executes the touched module's code — strictly
            # stronger evidence than the stem-map filename glob that also sets
            # ``matched``. (The glob-scan arm deliberately does NOT — a scan
            # asserts a cross-cutting invariant ABOUT the file.)
            matched = f in import_tested
            exact = work_root / "tests" / f"test_{stem}.py"
            if exact.exists():
                _add(f"tests/test_{stem}.py", f"stem-map:{f}")
                matched = True
            for hit in sorted((work_root / "tests").glob(f"test_*{stem}*.py")):
                _add(f"tests/{hit.name}", f"stem-map:{f}")
                matched = True
            if not matched:
                untested.append(f)
        # Any other extension (no recognized mapping — .py/.sh are the code
        # suffixes): ignore silently — it is neither code with a test nor a
        # workflow-invariant surface.

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
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=(
            "diff base (default: fetched origin/main, #1289; falls back to local "
            "'main' when origin/main does not resolve. A base without an 'origin/' "
            "prefix is used verbatim with no fetch — '--base main' is the "
            "pre-#1289 escape hatch)"
        ),
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help=(
            "skip the bounded pre-diff 'git fetch origin <branch>' (hermetic / "
            "offline runs; ref resolution + the local-main fallback still apply)"
        ),
    )
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
            "'test<TAB>matched_path' line per GLOB_SCAN_TESTS hit plus one "
            "'pin_test<TAB>rule_path' line per rules-pin discovery hit (#1496) "
            "plus the src/scripts import/literal/stem dependency-arm pairs "
            "(#1573) plus the pinned transitive-consumer pairs (#1589; "
            "WORKFLOW_INVARIANT members excluded from all three) and exit "
            "(the /issue Step 10d merge-gate mapping mode, #1147 — skips the "
            "diff-based selection entirely; empty stdout on no match is a "
            "SUCCESS, the gate's skip signal; a zero-mapped eligible code file "
            "draws a stderr WARN, and a non-empty map prints a "
            "recommended-timeout-s=<T> stderr sizing line, floor 300s)"
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
        # Mapping mode (#1147): GLOB_SCAN_TESTS + rules-pin (#1496) +
        # src/scripts dependency arms (import/literal/stem, #1573) +
        # pinned transitive-consumer pairs (#1589) — all three
        # WORKFLOW_INVARIANT-excluded — over an explicit file list — no git
        # diff; stderr carries the zero-mapped WARN floor + the
        # recommended-timeout-s sizing line (floor 300). The
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
        scan_pairs = map_scan_tests(files, work_root)
        for scan_test, f in sorted(_scan_pairs(files) - set(scan_pairs)):
            print(
                f"select_step9c_tests: WARN — scan test {scan_test} (matched by {f}) "
                f"absent from {work_root}; pair dropped",
                file=sys.stderr,
            )
        # Rules-pin pairs (#1496) + the src/scripts dependency arms (#1573:
        # import-map + literal-path + stem-map) + the pinned
        # transitive-consumer pairs (#1589) join the scan-map pairs (union
        # dedupes; a test hit by several arms
        # prints once per distinct matched path, and the consumers' `sort -u`
        # dedupes downstream). WORKFLOW_INVARIANT members are excluded inside
        # rules_pin_pairs / dependency_map_pairs / transitive_consumer_pairs;
        # the existing WARN loop above
        # is scan-map-only by design (scan-map keys are pinned literals that
        # can vanish from the tree; the discovery arms only ever find on-disk
        # tests, and a vanished transitive-consumer registration is dropped by
        # its existence check — the live-tree drift pins in
        # tests/test_select_step9c_tests.py make that staleness loud on main).
        all_pairs = sorted(
            {
                *scan_pairs,
                *rules_pin_pairs(files, work_root),
                *dependency_map_pairs(files, work_root),
                *transitive_consumer_pairs(files, work_root),
            }
        )
        # The #1573 fail-loud floor: an eligible src/scripts code file with
        # ZERO pairs across ALL arms is loudly visible (stderr, tab-free; rc
        # stays 0 — consumers treat helper rc!=0 as crash-class fail-closed).
        mapped = {f for _t, f in all_pairs}
        for f in sorted(literal_path_targets(files) - mapped):
            print(
                f"select_step9c_tests: WARN — no mapped tests for code file {f} "
                "(src/scripts dependency floor, #1573): a change here reaches the "
                "Step 10d / inline gates with zero pytest",
                file=sys.stderr,
            )
        if all_pairs:
            # Machine-greppable sizing line (tab-free stderr) so the Step-10d
            # TG legs can size their pytest bound from the map (#1573; floor =
            # the pre-#1573 fixed 300 s TG-leg bound).
            k_tests = sorted({t for t, _f in all_pairs})
            map_timeout = recommended_timeout_s(k_tests, floor=MAP_TIMEOUT_FLOOR_S)
            print(
                f"select_step9c_tests: map-files — {len(all_pairs)} pairs, "
                f"{len(k_tests)} tests; recommended-timeout-s={map_timeout}",
                file=sys.stderr,
            )
        for test, f in all_pairs:
            print(f"{test}\t{f}")
        return 0

    # Diff-base resolution (#1289): AFTER the --map-files early return (mapping
    # mode never diffs, so it must never fetch), BEFORE the diff. Every later
    # consumer (NOTE, sizing line, --json "base") sees the RESOLVED base.
    base = resolve_base(args.base, work_root, fetch=not args.no_fetch)
    try:
        touched = compute_touched(base, work_root)
    except subprocess.CalledProcessError as exc:
        # Fail loud — never silently fall back to zero tests on a git error.
        print(f"select_step9c_tests: git diff failed: {exc}", file=sys.stderr)
        return 1

    if not touched:
        # Loud, exit-0 NOTE (the documented degenerate fallback stays legitimate
        # when the checkout genuinely has no commits ahead of the base): the
        # #851 failure was a SILENT invariant-only fallback from the wrong cwd.
        print(
            f"select_step9c_tests: NOTE — empty diff vs '{base}' in {work_root}; "
            "falling back to the workflow-invariant set only. If this task's changes "
            "live in an issue worktree, re-run from that worktree (Step 9c contract).",
            file=sys.stderr,
        )

    tests, untested, reasons = select_tests_with_reasons(touched, work_root)
    missing = missing_invariants(work_root)

    # Fail loud on an EMPTY selection (defense-in-depth beside the Step 9c shell
    # guard against a silent test-gate pass). WORKFLOW_INVARIANT is a non-empty
    # always-on pinned set, so an empty list can only mean the work root resolved wrong (e.g.
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
        f"select_step9c_tests: {len(tests)} test files; diff-base={base}; "
        f"recommended-timeout-s={timeout_s} "
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
                    "base": base,
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
