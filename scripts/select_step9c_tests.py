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
  * any ``tests/**/test_*.py`` whose raw text references a touched
    ``.claude/skills/**/*.md`` file's PATH is ADDITIONALLY selected with
    reason ``skills-pin:<touched file>`` (#1851 — the skills sibling of the
    rules-pin arm above). Unlike the rules arm the matching token is
    skill-dir-QUALIFIED, not the bare basename (every skill's ``SKILL.md``
    shares the basename): a hit is the contiguous ``.claude/``-relative path
    (``skills/issue/SKILL.md`` — covers full-path literals too) as a raw
    substring OR the path-join form over the components
    (``"skills" / "issue" / "SKILL.md"``, either quote style, ``/`` or ``,``
    separators; a leading ``".claude"`` component matches implicitly).
    Additive only, same contract as rules-pin: the skills file itself keeps
    the WORKFLOW_SURFACE skip, comment/docstring mentions count, and a
    dynamically constructed filename is the accepted miss class.
  * any test registered in :data:`TRANSITIVE_CONSUMER_TESTS` for a touched
    file is ADDITIONALLY selected with reason
    ``transitive-consumer:<touched file>`` (#1589). A pinned literal, NOT
    discovered: these tests consume the touched module TRANSITIVELY (an
    ``importlib`` load by CONSTRUCTED path inside a helper, a path-join
    literal), so no text-scan arm can reach them. Additive only — never sets
    ``matched``, so ``untested_touched`` WARNs still fire.
  * any ``tests/**/test_*.py`` whose raw text contains a touched module's
    DOTTED name (``scripts.issue667_extract``,
    ``explore_persona_space.a.b`` — the dotted forms
    :func:`touched_module_names` derives; flat names deliberately excluded)
    as a boundary-bounded token is ADDITIONALLY selected with reason
    ``dotted-ref:<touched file>`` (#1688 — the monkeypatch-string-target
    shape, e.g. ``("scripts.issue667_extract", "main")``). Identifier
    boundaries on both sides (left also excludes ``.``) so
    ``scripts.task`` fires on neither ``scripts.task_state`` nor
    ``a.scripts.task``. Never sets ``matched``.
  * any ``tests/**/test_*.py`` whose raw text contains the bare BASENAME of a
    :func:`literal_path_targets`-eligible touched file (``.py`` AND ``.sh``,
    #1579 symmetry) as a boundary-bounded token is ADDITIONALLY selected with
    reason ``basename-ref:<touched file>`` (#1688 — the dispatcher-log-assert
    shape, e.g. ``assert "issue667_extract.py" in log``). Identifier
    boundaries kill the superstring class (``task.py`` does not fire on
    ``codex_task.py`` / ``task.pyx``; 63/1596 eligible basenames are
    substrings of another, measured 2026-07-25). Never sets ``matched``.
  * any ``tests/**/test_*.py`` whose ``Import``/``ImportFrom`` nodes name a
    one-hop scripts/ INTERMEDIARY of a touched ``scripts/`` module — a
    non-touched ``scripts/**/*.py`` that itself imports the touched module
    (:func:`transitive_import_map`) — is ADDITIONALLY selected with reason
    ``transitive-import:<touched file>`` (#1688 — the test EXECUTES the
    touched module at import time through the intermediary; the #1683 escape
    class). DISCOVERED, one hop by construction, scripts/-scoped on BOTH
    ends (src-rooted expansion measured +70 test files on a
    ``task_workflow.py`` touch — rejected); :data:`TRANSITIVE_CONSUMER_TESTS`
    (#1589) stays unchanged for importlib-by-constructed-path consumers.
    Never sets ``matched``.

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
work". (Orthogonal: the #1613 zero-resolution guard fires only on WHOLE-INPUT
zero resolution + zero pairs — a list where SOME lines resolve leaves this
out-of-prefix in-list gap open and distinct.)

Dotted-ref / basename-ref / transitive-import scope + cost (#1688): the two
string arms ride the SAME single :func:`_scan_test_files` read pass (each
regex behind a plain-substring pre-check — zero regex work on no-hit files);
the transitive arm adds ONE raw-text pass over ``scripts/**/*.py`` (~1,400
files, measured ~1.1 s typical for a single-script diff) ONLY when a touched
file resolves to a ``scripts/`` module name, parsing candidate-bearing files
via the same over-inclusive substring pre-filter (adversarial common-token
stems — e.g. ``scripts/eval.py``, token ``eval`` in ~1,199/1,392 scripts
files — measured ~13 s cold, the same order as the import arm's 4-8 s
test-tree worst case). All three arms are additive-only and never set
``matched``. Known-miss classes (each still lands in the pre-existing loud
``untested_touched`` WARN fallback where applicable): a FLAT-string reference
with no static import — pytest's string-target
``monkeypatch.setattr("X.attr", ...)`` / ``importlib.import_module("X")``
DOES import module ``X`` at test runtime, but that import is invisible to
every static arm here, so a test whose ONLY link is the flat string is
covered in practice ONLY when it also carries a static import the import arm
sees (the dotted arm does NOT cover the flat form by necessity — flat tokens
are deliberately excluded as mass-over-selecting English words);
dynamically-CONSTRUCTED dotted/basename strings (f-strings, concatenation);
import chains of >= 2 hops (the one-hop bound is deliberate); and src-rooted
transitivity (scripts-scoped on both ends by design — the src expansion
measured +70 test files, largely WORKFLOW_INVARIANT-redundant).

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
one ``<pin_test>\\t<skill_path>`` line per skills-pin discovery hit on a
``.claude/skills/**/*.md`` payload file (:func:`skills_pin_pairs`, #1851 —
WORKFLOW_INVARIANT members excluded) PLUS
the src/scripts dependency arms (:func:`dependency_map_pairs`, #1573 —
import-map (#1299) + literal-path (#1498) + dotted-ref / basename-ref /
transitive-import (#1688) + stem-map pairs for ``.py``/``.sh``
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
(``recommended_timeout_s(tests, floor=MAP_TIMEOUT_FLOOR_S)``, floor 600 s —
the Step-10d TG legs size their pytest bound from it). Empty stdout on
no match is a SUCCESS (exit 0 — the gate's skip signal); exit 1 only when FILE
is unreadable (the gate fails CLOSED on an unclassifiable payload). A FILE
with >=1 content line, ZERO lines resolving to existing paths under the work
root (absolute lines are skipped in that scan), and ZERO pairs draws the
#1613 zero-resolution guard: exit 2 (the argparse usage-error code) when the
argument's own suffix is ``.py``/``.sh`` — a source file handed instead of a
path-LIST file, the #1610 vacuously-passing-verify shape — else one hedged
tab-free stderr WARN + exit 0 (a deletion-only ``git diff --name-only``
payload is a legitimate zero-resolution list).

Default output: the exact gate invocation
``timeout --kill-after=60s <T>s uv run pytest <files...>
--continue-on-collection-errors -v --tb=short`` (#1746: a collection-broken
selected file reports as a per-file junit ``<error>`` testcase and pytest
exits rc=1 instead of aborting the whole run rc=2) on
stdout — ``<T>`` sized deterministically by :func:`recommended_timeout_s`
(#1046: 120s base + 30s/file + a 2400s surcharge when
``tests/test_workflow_lint.py`` is selected; re-measured from 330 real gate
junits 2026-07-13..24, #1646) — then any
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
``skills-pin:<touched file>`` /
``transitive-consumer:<touched file>`` / ``dotted-ref:<touched file>`` /
``basename-ref:<touched file>`` / ``transitive-import:<touched file>`` —
#1022, #1299, #1498, #1496, #1851, #1589, #1688).
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
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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
    # NEW (#1701) — workflow_lint --check-inline-round-duty-mirror + no-flags
    # bundling + drift-detection semantics pin
    "tests/test_workflow_lint_inline_round_duty_mirror.py",
    # NEW (#2165) — workflow_lint --check-smoke-blind-spot-review-lens +
    # --check-smoke-blind-spots (fixtures reproduce both #1336 shapes).
    "tests/test_workflow_lint_smoke_blind_spots.py",
    # NEW (#2067) — .claude/rules/compute-backend-failover.md
    # `### Cross-session pivot — resolve the owner before provisioning (#2067)`
    # prose pin: H3 header + pivoter-duty sentence + UNKNOWN-treat-as-LIVE token.
    "tests/test_workflow_lint_failover_pivot_pin.py",
    "tests/test_workflow_yaml.py",
    "tests/test_workflow_fix_dedup.py",
    # NEW (#1735) — rule reconciliation pin: workflow-fix-on-bug.md §
    # Recently-closed-sibling SUSPECT probe describes the composite blocking
    # contract (target + non-stopword title arm), not the retired
    # "advisory only, never a block" phrasing.
    "tests/test_workflow_fix_rule_closed_sibling_reconciliation.py",
    "tests/test_workflow_hub_upload_as_file.py",
    # group 3 — test_no_* invariants
    "tests/test_no_auto_runpod_path_under_any_failure.py",
    "tests/test_no_direct_task_path_construction.py",
    "tests/test_no_dollar_budget_caps.py",
    "tests/test_no_per_file_raw_completions_loop.py",
    "tests/test_no_pod_side_task_py_shellout.py",
    # NEW (#2058) — no-progress respawn lane: fingerprint helper +
    # `compute_issue_verdict` NO-PROGRESS-RESPAWN arm + heartbeat sentinel
    # set. Unit A ships the pure predicate; Unit B wires the watcher pass.
    "tests/test_no_progress_respawn.py",
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
    # NEW (#1623) — CLAUDE.md ad-hoc-summaries disclosure-clause pins (#1458/#1539/#1623)
    "tests/test_adhoc_summary_disclosure_pins.py",
    # NEW (#1910) — adversarial-planner SKILL.md Phase 1.5 fact-checker realized-grain
    # duty + planner.md §10 counted-grain clause pin
    "tests/test_adversarial_planner_factchecker_grain_pin.py",
    # NEW (#1734) — adversarial-planner SKILL.md Phase 1.5.0 per-WARN disposition pin
    "tests/test_adversarial_planner_warn_disposition.py",
    "tests/test_autonomous_plan_gate.py",
    "tests/test_autonomous_session_watch.py",
    # NEW (#1630) — /daily SKILL.md pathspec-commit (own-files-only) pin
    "tests/test_daily_skill_commit_pathspec_pin.py",
    # NEW (#1645) — /daily SKILL.md stub-first rule + healthcheck cross-file pin (#1189)
    "tests/test_daily_stub_first_doc.py",
    # NEW (#1645) — /daily SKILL.md three-route classifier prose pin (#706)
    "tests/test_daily_three_route_classifier_doc.py",
    # NEW (#1699) — implementer spec pin: mechanical --map-files pin-sweep hit list
    "tests/test_implementer_spec_mechanical_pin_sweep.py",
    # NEW (#1699) — implementer spec pin: repo-wide invariants in local union on
    # any scripts/*.py or src/** edit (#1681)
    "tests/test_implementer_spec_names_invariant_local_union.py",
    # NEW (#1699) — implementer spec pin: ruff-policy pin invocation in lint step (#1672)
    "tests/test_implementer_spec_names_ruff_policy_pin.py",
    # NEW (#1876) — SKILL.md Bare-push-snippets commit form (5) + guard hook
    # block-message compliant-forms lead pin
    "tests/test_issue_skill_bare_push_snippets_pin.py",
    # NEW (#1659) — SKILL.md 9a-ter + CLAUDE.md measured 1-cell pilot +
    # >=2x pilot-extrapolated fence-sizing pin
    "tests/test_issue_skill_compute_pilot_fence_pin.py",
    # NEW (#1656) — SKILL.md detached-phase harvest contract pin (#1310):
    # four-field breadcrumb (harvest= token), successor consumption, 9a-ter
    # mention + the two mirror duty-lists
    "tests/test_issue_skill_detached_harvest_pin.py",
    # Registration rider (#1659) — the pre-existing 9a-ter element-(5) content
    # pin (#1393) was never registered (the #1546 unregistered-pin class)
    "tests/test_issue_skill_disk_routing_pin.py",
    # NEW (#1964) — SKILL.md Step 6b dispatch-input/env/flag preflight pins
    # (staging / env-pin / per-leg / relaunch-flag probes) + the
    # crash-fix-rounds § Changed-argv relaunch flag-fidelity mirror
    "tests/test_issue_skill_dispatch_preflight_pin.py",
    # NEW (#1698) — experimenter.md Contract scope + fence-field derivation
    # prose pins (#1689 R8 launch-path fix). experimenter.md is
    # WORKFLOW_SURFACE, so this test file must live in WORKFLOW_INVARIANT to
    # gate the prose (touched-file selector routes WORKFLOW_SURFACE files
    # here). SKILL.md Step 6d.1 check-4 is pinned in the same file.
    "tests/test_experimenter_md.py",
    "tests/test_issue_skill_exit_breadcrumb.py",  # NEW (#1242) — SKILL.md exit-breadcrumb pin
    # NEW (#2161) — SKILL.md Step 6b fellows still-waiting launch contract pins
    # (free_lane_park_budget_reached third exit-75 producer + the
    # probe-before-relaunch launch-recovery invariant + the never-hand-off-to-
    # backend_poll-while-still_waiting clause)
    "tests/test_issue_skill_fellows_launch_contract_pin.py",
    # NEW (#1575) — SKILL.md cap-park surfacing pins (#1548/#1558/#1575)
    "tests/test_issue_skill_followup_cap_park_note_pin.py",
    # NEW (#1546) — SKILL.md forensics-ingest pointer pin
    "tests/test_issue_skill_forensics_ingest_pointer.py",
    # Registration rider (#1651) — the pre-existing #1305/#1533 gate-scope
    # pin file was never registered (the #1546 unregistered-pin class).
    "tests/test_issue_skill_gate_scope_brief_pin.py",
    # NEW (#1627) — SKILL.md Step 9c/10d gate single-flight probe pin (#1606)
    "tests/test_issue_skill_gate_single_flight.py",
    # NEW (#1927) — SKILL.md gist-update-recipe pin
    "tests/test_issue_skill_gist_update_recipe.py",
    # NEW (#1860) — SKILL.md 9a-humanize + 9a-bis strip verify-candidate-first
    # apply-ordering pin (verify --file before set-body; post-apply --issue confirm)
    "tests/test_issue_skill_humanize_verify_first_pin.py",
    # Registration rider (#1673) — the pre-existing #1500 inline-payload-gate
    # pin file was never registered (the #1546 unregistered-pin class); it now
    # also pins the worker-brief composition duty (#1673).
    "tests/test_issue_skill_inline_gate_pin.py",
    # NEW (#1625) — SKILL.md 9a-ter + CLAUDE.md inline measurement-design +
    # figure-sanity duties pin (both-arms mapping statement, rendered-PNG check)
    "tests/test_issue_skill_inline_measurement_duties.py",
    # NEW (#1970) — SKILL.md 9a-ter + CLAUDE.md inline-round upload-verify
    # recipe pin (verify → post epm:upload-verification → terminate;
    # enumerate-ALL-HF-prefixes duty; incident #1773)
    "tests/test_issue_skill_inline_upload_verify_recipe.py",
    # NEW (#1812) — SKILL.md 9a-ter + CLAUDE.md instrument-supersession +
    # scope-extension addenda duties pin
    "tests/test_issue_skill_instrument_supersession_addenda_pin.py",
    # NEW (#1944) — Step 10d lint-gate own-diff attribution pin: offender
    # path-token awk at BOTH sites + extracted-program fixture (#1768 false-block)
    "tests/test_issue_skill_lint_owndiff_attribution.py",
    "tests/test_issue_skill_marker_contract.py",
    # NEW (#1268) — SKILL.md Step-10d repin/guard hardening pin
    "tests/test_issue_skill_merge_resnapshot_pin.py",
    # NEW (#1756) — Write-tool merged-note compose pin (3 --file sites + CLAUDE.md)
    "tests/test_issue_skill_merged_note_compose.py",
    # NEW (#2014) — SKILL.md Monitor-condition pin
    "tests/test_issue_skill_monitor_condition_pin.py",
    # NEW (#1563) — SKILL.md orchestrator-turn discipline pointer pin
    "tests/test_issue_skill_orchestrator_turn_discipline_pointer.py",
    # NEW (#1897) — SKILL.md Step 10d PR-state probe + landing verification pin
    "tests/test_issue_skill_pr_state_probe.py",
    # NEW (#1976) — SKILL.md #1810 pre-split clause composition trigger pin (#1902 shape)
    "tests/test_issue_skill_pre_split_composition.py",
    # NEW (#1810) — SKILL.md Step 4b pre-split multi-deliverable dispatch pin
    "tests/test_issue_skill_presplit_dispatch_pin.py",
    # NEW (#1850) — SKILL.md remote-landing producer-fence deadline + Monitor heartbeat pin
    "tests/test_issue_skill_remote_landing_watch_pin.py",
    # NEW (#1855) — SKILL.md 5c-quater round-boundary durable-decision duty pin
    "tests/test_issue_skill_round_boundary_duty_pin.py",
    # NEW (#2040) — 9a-ter across-cell shard-axis + detached checkpoint-cadence duties pin
    "tests/test_issue_skill_shard_axis_checkpoint_cadence_pin.py",
    # NEW (#1572) — staged-index verification pin
    "tests/test_issue_skill_staged_index_verification.py",
    # NEW (#1751) — SKILL.md KEPT-stash surfacing duty pin
    "tests/test_issue_skill_stash_kept_duty_pin.py",
    # NEW (#1875) — SKILL.md Step 0 autonomous Monitor/TaskOutput schema-preload pin
    "tests/test_issue_skill_step0_preload_pin.py",
    # NEW (#1734) — SKILL.md Step 2 minimum plan-review floor + recorded-skip contract pin
    "tests/test_issue_skill_step2_floor.py",
    # NEW (#1595) — stopped-volume persist-before-park pin (SKILL.md + pod-config.md)
    "tests/test_issue_skill_stopped_volume_persist_pin.py",
    # NEW (#1868) — SKILL.md Terminal-teardown landing-confirmation pin
    "tests/test_issue_skill_terminal_landing_pin.py",
    # NEW (#1841) — Step 6d.2 tick-parse field-preservation pin
    "tests/test_issue_skill_tick_parse_preservation.py",
    # NEW (#2105) — SKILL.md triage-record (boundary=<ts>) token pin: the
    # enumerator snippet prints boundary= via triage_enumeration_boundary and
    # both recorded-line format forms carry the token (enumerate-to-post seam)
    "tests/test_issue_skill_triage_boundary_token.py",
    # NEW (#1587) — SKILL.md trigger-dense tag-adoption pin
    "tests/test_issue_skill_trigger_dense_tag_adoption.py",
    # NEW (#1616) — SKILL.md width-re-evaluation pin (test landed #1346; gap surfaced #1594)
    "tests/test_issue_skill_width_reeval_pointer.py",
    "tests/test_issue_tick_skill.py",  # NEW (#1629) — issue-tick SKILL prose pins
    # NEW (#1604) — mapping-baselines wiring pin (CLAUDE.md standing rule →
    # planner/critic/statistics-critic/experiment-guidelines + helper)
    "tests/test_mapping_baselines_wiring_pins.py",
    # NEW (#2187) — out-root TOP-LEVEL residue sweep prose pins
    # (upload-verifier.md Step 2.10 + verdict row + note-template outroot=
    # token, upload-policy.md § Out-root TOP-LEVEL residue, pods.md teardown
    # sweep clause, CLAUDE.md recipe clause). `.claude/agents/*.md` +
    # CLAUDE.md diffs are WORKFLOW_SURFACE-only, so this registration is the
    # ONLY gate that fires the pin on those changes.
    "tests/test_outroot_residue_prose_pins.py",
    # NEW (#1645) — CLAUDE.md + issue SKILL.md bracketed ownership-probe exemplar pin (#1495)
    "tests/test_ownership_probe_exemplar_bracketed.py",
    # NEW (#1631) — plan-patch helper + SKILL.md pointer pin
    "tests/test_plan_patch.py",
    # NEW (#2015) — repo-root uncommitted-state (pre-commit stash race) prose
    # pins: CLAUDE.md § Concurrent repo-root committers warning + landing
    # verification, SKILL.md § 9a-ter "Uncommitted-exposure window",
    # .claude/rules/repo-root-uncommitted-state.md mechanism file, LESSONS row.
    "tests/test_repo_root_uncommitted_state_pins.py",
    "tests/test_step0_enumerator_total_form.py",  # NEW (#1722) — Step-0 enumerator total-form pin
    "tests/test_step10d_guard3.py",  # NEW (#1242) — SKILL.md Step 10d guard/merge pin
    "tests/test_step10d_guards.py",  # NEW (#1978) — step10d_guards.sh extraction pin
    # NEW (#1723) — SKILL.md Step 10 CRON-TEARDOWN + epm:done reorder around
    # Step 10d merge (Terminal-teardown H4 + exit-site enumeration +
    # Step 10 step 6 branch-on-epm:merged + retry-surface long-phase heartbeats)
    "tests/test_issue_skill_step10_teardown_ordering.py",
    "tests/test_step_completed_resume.py",  # NEW (#1242) — resume/step-completed contract pin
    # NEW (#1662) — CLAUDE.md + SKILL.md suffixed-pod completion-teardown contract pin
    "tests/test_suffixed_pod_completion_teardown_pin.py",
    # NEW (#1693) — code-reviewer.md Step 0.69 phase-idempotency + inter-phase-
    # contract gate pin (prose + codex mirror + substantive-tag registry + ratchet-cap)
    "tests/test_code_reviewer_phase_idempotency_gate.py",
    # NEW (#1693) — gotchas.md finalization-crash entry pin (PyGILState_Release +
    # explicit-exit remedy under existing dispatcher paths: glob)
    "tests/test_gotchas_finalization_entry.py",
    # NEW (#1693) — planner.md §9 phase_outputs: declaration pin
    "tests/test_planner_phase_outputs_declaration.py",
    # NEW (#1598) — CLAUDE.md teammate-coordination bullet pins (a)-(d)
    "tests/test_teammate_coordination_pins.py",
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
    # NEW (#1632) — PostToolUse ruff-hook ephemeral-root (/tmp) exclusion pin:
    # the selector's arms are .py-only, so a settings.json diff re-runs this
    # pin ONLY via this tuple.
    "tests/test_ruff_format_hook_tmp_exclusion.py",
    # NEW (#1732) — parentless-kind: infra consistency-checker SKIP rule pin
    # (agent spec + adversarial-planner SKILL.md + /issue SKILL.md Step 2b);
    # any of the three targets can be edited independently, so this pin
    # lives in WORKFLOW_INVARIANT to gate all three prose surfaces.
    "tests/test_consistency_checker_parentless_infra_skip.py",
)

# --- Touched files that short-circuit (no per-file test map). ----------------
# These gate via the WORKFLOW_INVARIANT set, not a per-file test, so a touched
# file matching one of these is SKIPPED (and is NOT an "untested" file).
# .claude/rules/*.md files ADDITIONALLY map to their prose-pin tests via the
# rules-pin discovery arm (#1496), and .claude/skills/**/*.md files via the
# skills-pin discovery arm (#1851) — both additive; the skip itself is
# unchanged.
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
# via step9c_baseline.py::FILE_ANCHORED_SCAN_TESTS after source verification. A member
# left NON-anchored wedges compare at MF-4c exit 2 whenever the shared root is dirty
# and the member fails on main (#1632) — audit __file__-anchoring at addition time
# and allowlist when it holds.
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
    Step 9c run; this also keeps the 2400 s tests/test_workflow_lint.py out of
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


# --- Skills-pin discovery arm (#1851). ------------------------------------------
# The .claude/skills/**/*.md sibling of the rules-pin arm above: skill prose
# pins (72 test files reference .claude/skills/issue/SKILL.md on the
# 2026-07-31 tree; 42 invariant, 30 silently unselected — including the three
# founding tests test_issue_skill_file_only_verdict_post.py /
# test_ensemble_review_cap.py / test_issue_skill_workload_cmd_script_pin.py)
# got no targeted coverage from a skills-only diff: the skills globs
# short-circuit at WORKFLOW_SURFACE_GLOBS (by design — that skip is
# unchanged), and WORKFLOW_INVARIANT membership was the only channel. Same
# DISCOVERED posture + fail-soft read contract as rules-pin; the one delta is
# the matching token: SKILL.md as a bare basename would match ~123 files
# across ALL skills indiscriminately, so tokens are skill-dir-QUALIFIED —
# the contiguous ``.claude/``-relative path as a raw substring (covers
# full-path literals) OR a compiled path-join regex over the components
# (either quote style, ``/`` or ``,`` separators; a leading ``".claude"``
# component matches implicitly since ``re.search`` matches mid-string).
# Measured fan-out (2026-07-31): worst skill file issue/SKILL.md -> 72 test
# files (30 non-invariant); next adversarial-planner -> 11, daily -> 7.
# Known-miss class (accepted, same as rules-pin): a dynamically constructed
# filename. Additive only — never sets ``matched``, never enters
# untested_touched. Cost: one text pass over tests/**/test_*.py ONLY when a
# skills .md file is touched (same cost class as rules-pin, well under the
# import-map arm's measured 4-8 s AST worst case).
# Scan-regression loudness: tests/test_select_step9c_tests.py
# ::test_skills_pin_live_tree_known_pairs pins known (skill -> test) pairs;
# ::test_skills_pin_reachability_live_tree asserts EVERY live-tree test
# referencing a .claude/skills/*/SKILL.md path stays selector-reachable.
_SKILLS_PIN_GLOB = ".claude/skills/**/*.md"

# Quote + separator fragment for the path-join regex: closing quote of one
# component, ``/`` or ``,`` (Path-join or tuple form), opening quote of the
# next — either quote style on each side.
_SKILLS_PIN_JOIN_SEP = r"[\"']\s*[,/]+\s*[\"']"


def _skills_pin_tokens(rel_path: str) -> tuple[str, re.Pattern[str]]:
    """(contiguous substring token, compiled path-join regex) for *rel_path*.

    Given ``.claude/skills/issue/SKILL.md`` returns
    (``"skills/issue/SKILL.md"``, a regex matching
    ``"skills" / "issue" / "SKILL.md"`` with either quote style and ``/`` or
    ``,`` separators). Components are ``re.escape``d; multi-segment skill
    paths join all components in order. The contiguous token is the path
    relative to ``.claude/`` so it also covers the full-path literal form;
    the join regex needs no explicit optional ``".claude"`` prefix —
    ``re.search`` matches the suffix inside
    ``".claude" / "skills" / "issue" / "SKILL.md"`` anyway.
    """
    parts = rel_path.split("/")
    if parts and parts[0] == ".claude":
        parts = parts[1:]
    contiguous = "/".join(parts)
    join_re = re.compile("[\"']" + _SKILLS_PIN_JOIN_SEP.join(re.escape(p) for p in parts) + "[\"']")
    return contiguous, join_re


def skills_pin_hits(touched: list[str], work_root: Path) -> dict[str, set[str]]:
    """``{test_relpath: {touched .claude/skills/**/*.md files whose path its
    text references}}`` (#1851). Zero file reads when no skills file is
    touched. Matching via :func:`_skills_pin_tokens` (contiguous substring OR
    path-join regex); glob matching via :func:`_matches_any` so nested
    reference files under a skill dir are covered too (the ``/**/``
    zero-segment collapse).

    Fail-soft (the #1299 read contract, mirroring :func:`rules_pin_hits`): a
    test file that cannot be read / decoded emits ONE stderr WARN and is
    skipped — never crashes the selector; read failures on >5% of scanned
    files add ONE aggregate WARN. ``rglob`` only yields existing files, so no
    separate existence filter.
    """
    skills = [f for f in touched if _matches_any(f, (_SKILLS_PIN_GLOB,))]
    hits: dict[str, set[str]] = {}
    tests_dir = work_root / "tests"
    if not skills or not tests_dir.is_dir():
        return hits
    tokens = {f: _skills_pin_tokens(f) for f in skills}
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
                f"select_step9c_tests: WARN — skills-pin scan cannot read {rel}: {exc}; "
                "file skipped for the skills-pin arm",
                file=sys.stderr,
            )
            continue
        for skill_file, (contiguous, join_re) in tokens.items():
            if contiguous in text or join_re.search(text):
                hits.setdefault(rel, set()).add(skill_file)
    if n_scanned and n_failed / n_scanned > 0.05:
        print(
            f"select_step9c_tests: WARN — skills-pin scan read failures on "
            f"{n_failed}/{n_scanned} scanned test files (>5%): systemic tests/ breakage; "
            "the skills-pin arm may under-select",
            file=sys.stderr,
        )
    return hits


def skills_pin_pairs(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Sorted ``(pin_test, skill_file)`` pairs for ``--map-files`` (#1851).

    WORKFLOW_INVARIANT members are EXCLUDED here (the :func:`rules_pin_pairs`
    asymmetry: they already gate every Step 9c run, and the exclusion keeps
    the 2400 s tests/test_workflow_lint.py out of the Step 10d /
    inline-payload lint gate). The Step 9c selection arm keeps them (harmless
    extra reason; the union dedupes) — pinned by
    test_cli_map_files_skills_pin_excludes_invariant.
    """
    inv = set(WORKFLOW_INVARIANT)
    return sorted(
        (t, s)
        for t, skill_files in skills_pin_hits(files, work_root).items()
        if t not in inv
        for s in sorted(skill_files)
    )


# --- Content-tolerant filesystem probes (#1791). -------------------------------
def _safe_exists(p: Path) -> bool:
    """``p.exists()`` that treats an unstat-able path as absent (#1791).

    A content line from a mis-passed ``--map-files`` payload (markdown prose,
    a source line > NAME_MAX) raises ``OSError`` [Errno 36] — or ``ValueError``
    on an embedded NUL — from ``Path.exists()`` BEFORE the #1613 misuse
    diagnostic can fire. Such a line is by definition not an existing repo
    path, so absent is the correct answer, not a swallowed fault; no stat-able
    path takes the except arm, so valid inputs are byte-identical.
    """
    try:
        return p.exists()
    except (OSError, ValueError):
        return False


def _safe_glob(root: Path, pattern: str) -> list[Path]:
    """``sorted(root.glob(pattern))`` tolerant of content-derived patterns (#1791).

    A stem derived from a mis-passed content line can embed glob
    metacharacters (``scripts/*.py`` -> stem ``*`` -> pattern
    ``test_***.py``), which raises ``ValueError`` ("Invalid pattern: '**' can
    only be an entire path component") on modern pathlib — again before the
    #1613 diagnostic. An invalid pattern cannot match an existing test, so
    ``[]`` is the correct answer; valid patterns take the sorted-glob path
    unchanged.
    """
    try:
        return sorted(root.glob(pattern))
    except (OSError, ValueError):
        return []


# --- src/scripts dependency arms for --map-files (#1573). ---------------------
MAP_TIMEOUT_FLOOR_S = 600  # Step-10d TG-leg floor (#1646; was 300, basis the
#                            ~12.6 s 2-test scan map of 2026-07-08). #1634's
#                            healthy 5-file baseline leg measured 202.9 s and
#                            the identical gated leg was killed at 300 s under
#                            residual load; 600 ~= 3x the measured healthy
#                            small-map wall (#1634 run 2 passed, TG legs
#                            174/174 on both trees).
MAP_TIMEOUT_DISPERSION = 2.0  # p90 dispersion factor on the map path (#1697;
#                               was implicit 1.0). Two independent same-day
#                               sessions #1675/#1682 (2026-07-25) hit the
#                               undersized-bound trap at 728.19s / 751.7s
#                               measured walls against the 780s formula-derived
#                               bound (~1.04-1.07x) and each hand-doubled to
#                               1560s / 1600s to recover. Matches the p90 x2
#                               default in .claude/rules/plan-compute-sizing.md
#                               (#833's realized-vs-planned mean overrun of ~2x).
#                               Applies ONLY on the --map-files path; the diff
#                               path already carries the per-file SLOW_TESTS
#                               surcharge (#1646, 1.40x-dispersion-adjusted).


def dependency_map_pairs(files: list[str], work_root: Path) -> list[tuple[str, str]]:
    """Sorted ``(test, matched_path)`` pairs for ``--map-files`` beyond
    GLOB_SCAN_TESTS + rules-pin (#1573): import-map (#1299) + literal-path
    (#1498) + dotted-ref / basename-ref / transitive-import (#1688) via the
    ONE shared :func:`_scan_test_files` read pass, plus
    stem-map path arithmetic. WORKFLOW_INVARIANT members are EXCLUDED
    (the :func:`rules_pin_pairs` asymmetry: they already gate every Step 9c
    run, and exclusion keeps the 2400 s tests/test_workflow_lint.py out of the
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
    scan = _scan_test_files(files, work_root)
    pairs: set[tuple[str, str]] = set()
    for hits in (
        scan.import_hits,
        scan.literal_hits,
        scan.dotted_hits,
        scan.basename_hits,
        scan.trans_hits,
    ):
        for t, fs in hits.items():
            if t not in inv:
                pairs.update((t, f) for f in fs)
    for f in literal_path_targets(files):  # the same eligibility predicate
        stem = Path(f).stem
        exact = f"tests/test_{stem}.py"
        # Content-derived probes: _safe_exists/_safe_glob (#1791) — a hostile
        # content line (> NAME_MAX stem, glob-metachar stem) must reach the
        # #1613 diagnostic, not crash here first.
        if _safe_exists(work_root / exact) and exact not in inv:
            pairs.add((exact, f))
        for hit in _safe_glob(work_root / "tests", f"test_*{stem}*.py"):
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


# --- Gate-timeout sizing (#1046; figures re-measured by #1646). ---------------
# Measured from 330 Step 9c gate junits on the shared VM (2026-07-13..24,
# /tmp/step9c-junit-issue-*.xml, per-testcase `time` summed per file):
# tests/test_workflow_lint.py alone min 472 s / median 789 s / p90 1111 s /
# max 1819 s (58/330 runs exceeded the previous 1050 s one-file bound;
# fresh standalone re-measure 1271 s, 2026-07-24, #1646);
# whole-gate testcase-time totals median ~1094 s / p90 ~1568 s / max ~2289 s.
# The foreground Bash tool cap is 600 s, so the printed command carries this
# bound and Step 9c runs the gate as a BACKGROUND invocation (SKILL.md 9c 1b).
# Constants are deliberately generous (~1.4-2x over worst measured): an
# oversized bound only ever fires on a genuine wedge; an undersized one kills
# healthy gates (#991/#996/#906, exit 143 at 480-540 s foreground bounds;
# #1642: a 900 s surcharge vs a 1188.62 s measured wall).
TIMEOUT_BASE_S = 120  # pytest startup + collection (~2500 tests) + imports
TIMEOUT_PER_FILE_S = 30  # ~2x the p90 per-file runtime of non-slow files
TIMEOUT_FLOOR_S = 900
# Per-file surcharges for files whose OWN max exceeds the per-file allocation
# by an order of magnitude. Pinned literal (same curation rule as
# WORKFLOW_INVARIANT); live-tree drift pin in tests/test_select_step9c_tests.py.
SLOW_TESTS: dict[str, int] = {
    # Runs whole-tree lints repeatedly; wall GREW 771 -> 1819 s max between the
    # n=26 (2026-07-04..05) and n=330 (2026-07-13..24) junit samples (#1642
    # measured 1188.62 s standalone). 2400 puts the one-file bound at
    # 120 + 30 + 2400 = 2550 s = 1.40x the 1819 s worst measured (#1646).
    "tests/test_workflow_lint.py": 2400,
    # Slow overall, not just slow-exit: 41 tests measured 636.62 s of test time
    # + ~113 s non-test overhead (uv/pytest startup + collection +
    # interpreter-exit residue from the workflow_lint ThreadPoolExecutor probe
    # family -- same class as the sibling entry above) = 750 s wall to
    # completion, measured 2026-08-08 (#1994; rc=0 under a 3600 s fence).
    # 1200 = 1.60x the 750 s worst measured wall (registry convention ~1.4-2x;
    # sibling above is 1.40x on its one-file bound). One-file bound:
    # 120 + 30 + 1200 = 1350 s = 1.8x the measured wall.
    "tests/test_workflow_lint_phase_done_check.py": 1200,
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


def recommended_timeout_s(
    tests: list[str],
    *,
    floor: int = TIMEOUT_FLOOR_S,
    dispersion: float = 1.0,
) -> int:
    """Deterministic `timeout(1)` bound for a Step 9c gate selection.

    ``BASE + PER_FILE * len(tests)`` scaled by *dispersion*, then plus the
    slow-file surcharges, floored at *floor*.
    Default ``dispersion=1.0`` keeps the diff-path callers byte-identical;
    ``--map-files`` mode passes ``dispersion=MAP_TIMEOUT_DISPERSION`` (2.0;
    #1697) so a healthy leg's bound is ~2x its measured wall, not ~1x.
    Slow-file surcharges are NOT re-scaled (they already encode a per-file
    dispersion-adjusted headroom: #1646 pinned `tests/test_workflow_lint.py`
    at 2400s = 1.40x its 1819s worst measured wall).
    Default ``floor`` (``TIMEOUT_FLOOR_S``) keeps diff-path callers unchanged;
    ``--map-files`` mode passes ``floor=MAP_TIMEOUT_FLOOR_S``, the Step-10d
    TG-leg 600 s floor (#1573/#1646). Invariant-only selection (61 files as
    of 2026-07-24, incl. the workflow-lint surcharge) -> 4350 s (72.5 min),
    matching the invariant-set-scale precedents (``step9c_baseline.py
    refresh`` ``--timeout-s`` default 4350 s; the SKILL.md detached
    refresh's 4650 s).
    """
    t = round(dispersion * (TIMEOUT_BASE_S + TIMEOUT_PER_FILE_S * len(tests)))
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


class ScanResult(NamedTuple):
    """Per-arm hits from the ONE shared read pass over ``tests/**/test_*.py``.

    Internal-only return shape of :func:`_scan_test_files` (#1688 widened it
    from the historical 3-tuple; :func:`import_map_hits` keeps the public
    #1299 2-tuple). Every ``dict`` maps ``{test_relpath: {touched files}}``.
    """

    import_hits: dict[str, set[str]]  # #1299 — the only arm that sets ``matched``
    import_tested: set[str]  # touched files with >= 1 importing test
    literal_hits: dict[str, set[str]]  # #1498 — full repo-relative-path substring
    dotted_hits: dict[str, set[str]]  # #1688 — dotted-module string references
    basename_hits: dict[str, set[str]]  # #1688 — bare-basename references
    trans_hits: dict[str, set[str]]  # #1688 — one-hop scripts/ transitive imports


def _boundary_patterns(
    names_to_files: dict[str, set[str]], *, dot_bounded_left: bool
) -> list[tuple[str, re.Pattern[str], set[str]]]:
    """``(substring pre-check token, boundary regex, touched files)`` triples (#1688).

    The regex bounds *name* with identifier-character lookarounds so
    SUPERSTRINGS never match (``scripts.task`` does not fire on
    ``scripts.task_state``; ``task.py`` does not fire on ``codex_task.py`` or
    ``task.pyx``). *dot_bounded_left* additionally excludes ``.`` on the LEFT
    (the dotted arm: ``a.scripts.task`` must not fire for ``scripts.task``);
    the RIGHT boundary deliberately allows ``.`` — ``scripts.task.main`` still
    references the module, and ``task.py`` inside a full path literal is a
    harmless duplicate the reason-set union dedupes.
    """
    left = "A-Za-z0-9_." if dot_bounded_left else "A-Za-z0-9_"
    return [
        (name, re.compile(rf"(?<![{left}]){re.escape(name)}(?![A-Za-z0-9_])"), set(files))
        for name, files in sorted(names_to_files.items())
    ]


def _apply_boundary_hits(
    text: str,
    rel: str,
    patterns: list[tuple[str, re.Pattern[str], set[str]]],
    hits: dict[str, set[str]],
) -> None:
    """Record boundary-regex hits of *patterns* in *text* under *rel* (#1688).

    The plain-substring pre-check keeps the common no-hit case regex-free.
    """
    for token, pat, files in patterns:
        if token in text and pat.search(text):
            hits.setdefault(rel, set()).update(files)


def transitive_import_map(touched: list[str], work_root: Path) -> dict[str, set[str]]:
    """``{intermediary scripts-module name -> touched scripts/ files it imports}``,
    ONE import hop, scripts/-scoped on BOTH ends (#1688).

    DISCOVERED at selection time (the #1496 discovered-over-pinned rationale):
    scans ``scripts/**/*.py`` (never ``tests/``, never ``src/``) for files
    whose ``Import``/``ImportFrom`` nodes name a touched ``scripts/`` module.
    A test importing such an INTERMEDIARY executes the touched module at
    import time (the sys.path-flat scripts->scripts import graph, the #1683
    escape class), so :func:`_scan_test_files` selects it with reason
    ``transitive-import:<touched file>``. Never recursive: the intermediary
    map is computed ONCE from the touched files and intermediaries' own
    importers are NOT followed (one hop by construction). A touched file is
    skipped as an intermediary (it is already a DIRECT import-arm target).
    :data:`TRANSITIVE_CONSUMER_TESTS` (#1589) is UNCHANGED and not subsumed —
    it covers importlib-by-CONSTRUCTED-path consumers no AST scan can see.

    Cost: zero file reads unless a touched file resolves to a ``scripts/``
    module name; otherwise one raw-text pass over the scripts tree (~1,400
    files, measured ~1.1 s typical) with the same over-inclusive substring
    pre-filter as the test scan (never a false negative), parsing only
    candidate-bearing files (adversarial common-token stems measured ~13 s —
    module docstring). Fail-soft (the #1299 read contract): unreadable files
    WARN + skip; files that read but fail to ``ast``-parse WARN + skip; >5%
    failures add ONE aggregate WARN.
    """
    module_map = touched_module_names(touched)
    script_names = {n: {f for f in fs if f.startswith("scripts/")} for n, fs in module_map.items()}
    script_names = {n: fs for n, fs in script_names.items() if fs}
    scripts_dir = work_root / "scripts"
    if not script_names or not scripts_dir.is_dir():
        return {}
    tokens = {n.rsplit(".", 1)[-1] for n in script_names}
    touched_set = set(touched)
    out: dict[str, set[str]] = {}
    n_scanned = 0
    n_failed = 0
    for path in sorted(scripts_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(work_root).as_posix()
        if rel in touched_set:
            continue  # already a DIRECT target of the import/literal arms
        n_scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            n_failed += 1
            print(
                f"select_step9c_tests: WARN — transitive-import scan cannot read {rel}: "
                f"{exc}; file skipped for the transitive-import arm",
                file=sys.stderr,
            )
            continue
        if not any(tok in text for tok in tokens):
            continue  # over-inclusive substring pre-filter, never a false negative
        try:
            imported = _import_names(ast.parse(text))
        except (SyntaxError, ValueError) as exc:
            n_failed += 1
            print(
                f"select_step9c_tests: WARN — transitive-import scan cannot parse {rel}: "
                f"{exc}; file skipped for the transitive-import arm",
                file=sys.stderr,
            )
            continue
        hit = {f for n, fs in script_names.items() if n in imported for f in fs}
        if hit:
            for name in touched_module_names([rel]):
                out.setdefault(name, set()).update(hit)
    if n_scanned and n_failed / n_scanned > 0.05:
        print(
            f"select_step9c_tests: WARN — transitive-import scan read/parse failures on "
            f"{n_failed}/{n_scanned} scanned scripts files (>5%): systemic scripts/ "
            "breakage; the transitive-import arm may under-select",
            file=sys.stderr,
        )
    return out


def _scan_test_files(touched: list[str], work_root: Path) -> ScanResult:
    """ONE shared read pass over ``tests/**/test_*.py`` for every text-scan arm.

    Returns a :class:`ScanResult` (#1688 widened the historical 3-tuple):
      * ``import_hits`` — ``{test_relpath: {touched files it imports}}``
        (#1299, the import-map arm);
      * ``import_tested`` — touched files with >= 1 importing test (these
        suppress the ``untested_touched`` WARN);
      * ``literal_hits`` — ``{test_relpath: {touched files whose repo-relative
        path appears as a raw substring of the test's text}}`` (#1498, the
        literal-path arm; never suppresses the WARN);
      * ``dotted_hits`` — boundary-regex hits on the DOTTED module names from
        :func:`touched_module_names` (#1688; flat names deliberately excluded
        — a boundary-bounded flat token like ``task`` matches the English
        word; never suppresses the WARN);
      * ``basename_hits`` — boundary-regex hits on the bare BASENAME of every
        :func:`literal_path_targets`-eligible file (``.py`` AND ``.sh``,
        #1688/#1579; never suppresses the WARN);
      * ``trans_hits`` — tests whose imports name a one-hop scripts/
        INTERMEDIARY per :func:`transitive_import_map` (#1688; never
        suppresses the WARN — over-WARN is the safe direction, the #1589
        precedent).

    Subdir tests (``tests/experiments/``) are in scope (sorted ``rglob`` —
    pytest collects them and the touched-test arm already admits their paths).

    Cost bound: when BOTH target sets are empty (workflow-surface-only diff)
    the pass returns immediately with ZERO file reads (every #1688 arm derives
    from those same sets, so they add no new trigger). Otherwise each test
    file's raw text is read ONCE and (a) checked against each literal target
    as a plain substring plus the dotted/basename boundary regexes (each
    behind a plain-substring pre-check), then (b) ``ast``-parsed ONLY when its
    text contains the last dotted component of at least one candidate module
    name — touched OR intermediary — since any absolute import of module M
    must literally spell M's last component, so the filter is over-inclusive
    (never a false negative). A literal-only trigger (e.g.
    ``scripts/__init__.py``, which maps to no module name)
    reads the tree with ZERO parses (empty token pre-filter); a ``.sh``-only
    diff is the same cost class (#1579 — ``touched_module_names`` stays
    ``.py``-only, so ``.sh`` paths join ``literal_targets`` with an empty
    token set: one raw-text pass, zero AST parses). Typical touched
    sets parse a handful of files; worst case the whole tree (~500 files,
    measured ~4-8 s under shared-VM load — module docstring).

    Fail-soft (#1299 rationale R5): a file whose RAW READ fails (``OSError``,
    ``ValueError`` incl. ``UnicodeDecodeError``) emits ONE stderr WARN and is
    skipped for ALL arms; a file that reads but fails to PARSE
    (``SyntaxError``, ``ValueError``) is skipped for the import + transitive
    arms ONLY — its literal/dotted/basename hits are kept (raw text already
    read; an additive improvement, #1498). Never crashes the selector; if the
    broken file is
    itself part of the diff the touched-test arm still selects it (pytest
    collection then fails loud at the right surface). When read + parse
    failures exceed 5% of the scanned files, ONE additional aggregate WARN
    flags the systemic tests/ breakage.
    """
    import_hits: dict[str, set[str]] = {}
    tested: set[str] = set()
    literal_hits: dict[str, set[str]] = {}
    dotted_hits: dict[str, set[str]] = {}
    basename_hits: dict[str, set[str]] = {}
    trans_hits: dict[str, set[str]] = {}
    result = ScanResult(import_hits, tested, literal_hits, dotted_hits, basename_hits, trans_hits)
    module_map = touched_module_names(touched)
    literal_targets = literal_path_targets(touched)
    if not module_map and not literal_targets:
        return result
    tests_dir = work_root / "tests"
    if not tests_dir.is_dir():
        return result
    trans_map = transitive_import_map(touched, work_root)
    dotted_pats = _boundary_patterns(
        {n: fs for n, fs in module_map.items() if "." in n}, dot_bounded_left=True
    )
    base_map: dict[str, set[str]] = {}
    for f in literal_targets:
        base_map.setdefault(Path(f).name, set()).add(f)
    base_pats = _boundary_patterns(base_map, dot_bounded_left=False)
    tokens = {name.rsplit(".", 1)[-1] for name in module_map}
    tokens |= {name.rsplit(".", 1)[-1] for name in trans_map}
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
        _apply_boundary_hits(text, rel, dotted_pats, dotted_hits)
        _apply_boundary_hits(text, rel, base_pats, basename_hits)
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
        for name, files in trans_map.items():
            if name in imported:
                trans_hits.setdefault(rel, set()).update(files)
    if n_scanned and n_failed / n_scanned > 0.05:
        print(
            f"select_step9c_tests: WARN — test-scan read/parse failures on "
            f"{n_failed}/{n_scanned} scanned test files (>5%): systemic tests/ breakage; "
            "the import and literal-path arms may under-select",
            file=sys.stderr,
        )
    return result


def import_map_hits(touched: list[str], work_root: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Back-compat wrapper over :func:`_scan_test_files` (the #1299 public shape).

    Returns ``(import_hits, import_tested)``, dropping every other
    :class:`ScanResult` element. NOTE (#1498): the underlying shared pass also
    triggers on literal-eligible touched files, so a
    ``scripts/__init__.py``-only call
    now scans (with zero parses) where the pre-refactor arm returned early
    with zero reads; the returned import elements are unchanged.
    """
    scan = _scan_test_files(touched, work_root)
    return scan.import_hits, scan.import_tested


def _seed_import_reasons(import_hits: dict[str, set[str]]) -> dict[str, set[str]]:
    """Initial ``{test: reasons}`` mapping seeded from import-map hits (#1299)."""
    return {t: {f"import-map:{f}" for f in files} for t, files in import_hits.items()}


def _seed_scan_reasons(scan: ScanResult) -> dict[str, set[str]]:
    """Initial ``{test: reasons}`` mapping seeded from EVERY text-scan arm.

    Import-map hits (#1299) plus literal-path (#1498), dotted-ref,
    basename-ref, and transitive-import hits (#1688) — a test hit by several
    arms carries every reason kind. Purely additive: only ever adds
    tests/reasons to the seed; the import arm alone feeds ``matched``
    (via ``scan.import_tested`` at the caller).
    """
    selected = _seed_import_reasons(scan.import_hits)
    for kind, hits in (
        ("literal-path", scan.literal_hits),
        ("dotted-ref", scan.dotted_hits),
        ("basename-ref", scan.basename_hits),
        ("transitive-import", scan.trans_hits),
    ):
        for hit_test, hit_files in hits.items():
            selected.setdefault(hit_test, set()).update(f"{kind}:{f}" for f in hit_files)
    return selected


def _add_glob_scan_reasons(f: str, work_root: Path, add) -> None:
    """Glob-scan invariant arm (#895): additive, never marks ``f`` tested."""
    for scan_test, scan_globs in GLOB_SCAN_TESTS.items():
        if _matches_any(f, scan_globs) and (work_root / scan_test).exists():
            add(scan_test, f"glob-scan:{f}")


def _seed_transitive_consumer_reasons(
    touched: list[str], work_root: Path, selected: dict[str, set[str]]
) -> None:
    """Transitive-consumer pin arm (#1589): additive seed, same only-grows
    contract; never sets ``matched`` (the glob-scan/literal precedent —
    over-WARN is the safe direction). Invariant members are KEPT here
    (harmless extra reason; the union dedupes — the rules-pin asymmetry).
    Extracted as a module-level seed (the :func:`_add_glob_scan_reasons`
    precedent) to keep :func:`select_tests_with_reasons` under the C901 cap.
    """
    for f in touched:
        for t in TRANSITIVE_CONSUMER_TESTS.get(f, ()):
            if (work_root / t).exists():
                selected.setdefault(t, set()).add(f"transitive-consumer:{f}")


def select_tests_with_reasons(
    touched: list[str], work_root: Path
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Superset of :func:`select_tests`: also returns ``{test: sorted reasons}``.

    A reason is ``'invariant' | 'touched-test' | 'stem-map:<touched file>' |
    'glob-scan:<touched file>' | 'import-map:<touched file>' |
    'literal-path:<touched file>' | 'rules-pin:<touched file>' |
    'skills-pin:<touched file>' |
    'transitive-consumer:<touched file>' | 'dotted-ref:<touched file>' |
    'basename-ref:<touched file>' |
    'transitive-import:<touched file>'``; a test may
    carry several (e.g. an invariant that is also stem-mapped from a touched
    file). Selection behavior is IDENTICAL to :func:`select_tests` — same
    tests, same ``untested_touched`` WARN list, same sorted ordering (#1022:
    the reasons feed ``step9c_baseline.py compare``'s diff-linked-ness read,
    which must come from the SAME mapping logic that selected the run's tests).
    """
    # Text-scan arms (ONE shared read pass, _scan_test_files): the import-map
    # arm (#1299) seeds the selection and the literal-path (#1498) +
    # dotted-ref / basename-ref / transitive-import (#1688) arms add
    # reference/consumer hits — all additive by construction (they only ever
    # add tests/reasons here and, via the ``matched`` seed below, the IMPORT
    # arm alone removes entries from the untested WARN list; no code path
    # drops a stem-map / glob-scan / invariant / touched-test selection, so
    # selection only GROWS). Ordering is irrelevant: the terminal sorted()
    # reads below keep the output deterministic.
    scan = _scan_test_files(touched, work_root)
    import_tested = scan.import_tested
    selected: dict[str, set[str]] = _seed_scan_reasons(scan)
    # Rules-pin discovery arm (#1496): additive seed, same only-grows contract
    # (one scan pass serves all touched rules; the rules file itself still
    # takes the WORKFLOW_SURFACE `continue` below, unchanged).
    for t, rule_files in rules_pin_hits(touched, work_root).items():
        selected.setdefault(t, set()).update(f"rules-pin:{r}" for r in rule_files)
    # Skills-pin discovery arm (#1851): additive seed, same only-grows
    # contract (skill-dir-qualified tokens — see skills_pin_hits; the skills
    # file itself still takes the WORKFLOW_SURFACE `continue` below,
    # unchanged).
    for t, skill_files in skills_pin_hits(touched, work_root).items():
        selected.setdefault(t, set()).update(f"skills-pin:{s}" for s in skill_files)
    # Transitive-consumer pin arm (#1589): additive seed, same only-grows
    # contract — see _seed_transitive_consumer_reasons.
    _seed_transitive_consumer_reasons(touched, work_root, selected)
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
        # A touched test file includes itself. (_safe_exists: content-line
        # tolerance, #1791 — diff-mode twin of the map-files stem arm.)
        if f.startswith("tests/") and p.name.startswith("test_") and p.suffix == ".py":
            if _safe_exists(work_root / f):
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
            if _safe_exists(exact):
                _add(f"tests/test_{stem}.py", f"stem-map:{f}")
                matched = True
            for hit in _safe_glob(work_root / "tests", f"test_*{stem}*.py"):
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


def _zero_resolution_guard(
    map_files_arg: str,
    files: list[str],
    all_pairs: list[tuple[str, str]],
    work_root: Path,
) -> int | None:
    """#1613 zero-resolution guard for ``--map-files`` (returns exit code 2 or None).

    A FILE argument whose content lines resolve to ZERO existing repo paths
    AND produce ZERO pairs across all arms is the silent operator-error shape
    (a source file handed instead of a path-LIST file — #1610's
    vacuously-passing verify command). A deletion-only payload
    (``git diff --name-only`` includes status-D paths absent from the
    worktree) legitimately looks like this, so a list-file argument stays
    exit 0 (this helper prints the hedged WARN and returns None); only an
    argument that is ITSELF a ``.py``/``.sh`` source file errors — returns 2,
    the argparse usage-error code (consumers treat rc!=0 as crash-class
    fail-closed, the intended outcome for a malformed invocation). Pairs
    suppress the guard: glob/scan arms can map nonexistent (deleted) payload
    paths, and that output is valid. Absolute content lines are SKIPPED in
    the existence scan: for an absolute f, ``work_root / f`` returns f
    itself, so one existing absolute line inside a source file would silence
    the guard. Both stderr lines are single-line, tab-free, and carry no
    ``recommended-timeout-s=`` substring (consumer grammar constraints).
    """
    if not files or all_pairs:
        return None
    # _safe_exists (#1791): a > NAME_MAX content line (the mis-passed
    # markdown/source shape this guard exists to diagnose) raises OSError
    # from a bare exists() — crashing the very diagnostic meant to fire.
    if any(_safe_exists(work_root / f) for f in files if not f.startswith("/")):
        return None
    if Path(map_files_arg).suffix in (".py", ".sh"):
        print(
            f"select_step9c_tests: ERROR — --map-files argument "
            f"{map_files_arg} looks like a source file, not a "
            f"path-LIST file: none of its {len(files)} content lines "
            f"resolve to an existing repo path under {work_root}, and "
            "no mapping arm produced pairs (#1613); pass a "
            "newline-delimited path-list file (e.g. "
            "git diff --name-only > /tmp/files.txt)",
            file=sys.stderr,
        )
        return 2
    print(
        f"select_step9c_tests: WARN — --map-files input "
        f"{map_files_arg} resolved to ZERO existing repo paths under "
        f"{work_root} ({len(files)} content lines, 0 pairs; #1613): "
        "benign for a deletion-only or sparse-excluded payload; "
        "otherwise verify the argument is a path-LIST file, not a "
        "source file",
        file=sys.stderr,
    )
    return None


def _run_map_files_mode(map_files_arg: str, work_root: Path) -> int:
    """Execute the --map-files mapping mode (#1147).

    Emits `<test>\\t<matched_path>` TSV lines to stdout across five
    arms (GLOB_SCAN_TESTS, rules-pin #1496, skills-pin #1851, src/scripts
    dependency arms #1573/#1688, pinned transitive-consumer pairs #1589) —
    all WORKFLOW_INVARIANT-excluded — over an explicit file list. No git diff;
    stderr carries the zero-mapped WARN floor + the recommended-timeout-s
    sizing line (floor 300). Return codes: 0 on success (empty stdout on no
    match is the Step 10d merge-gate skip signal); 1 on an unreadable or
    undecodable (binary, #1791) input file (fail CLOSED); 2 on the #1613
    zero-resolution guard's source-file argument.

    Extracted from ``main`` (#1717) to keep ``main``'s cyclomatic
    complexity under the ruff C901 cap (≤15) after the new (a)/(c)
    branches landed on the top-level entry.
    """
    try:
        raw = Path(map_files_arg).read_text()
    except (OSError, ValueError) as exc:
        # ValueError covers a mis-passed BINARY file (UnicodeDecodeError is
        # a ValueError subclass, #1791) — same rc-1 "cannot read" path.
        # (c) opt-in hint on the comma-blob shape (#1717 defect (c),
        # session `c0a2df1b`): `--map-files a.md,b.md` is a common
        # mistake — argparse treats the comma-joined blob as a single
        # PATH, so read_text() surfaces Errno 2. Append (never
        # substitute — a legitimate comma-in-path failure still needs
        # to see its own Errno).
        hint = ""
        if "," in str(map_files_arg):
            hint = (
                " (--map-files takes a PATH to a newline-separated file "
                "list, not a comma-separated list of paths — write the "
                "paths to a file and pass that file's path)"
            )
        print(
            f"select_step9c_tests: cannot read --map-files input: {exc}{hint}",
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
    # Rules-pin pairs (#1496) + skills-pin pairs (#1851) + the src/scripts
    # dependency arms (#1573: import-map + literal-path + stem-map) + the
    # pinned transitive-consumer pairs (#1589) join the scan-map pairs (union
    # dedupes; a test hit by several arms prints once per distinct matched
    # path, and the consumers' `sort -u` dedupes downstream).
    # WORKFLOW_INVARIANT members are excluded inside rules_pin_pairs /
    # skills_pin_pairs / dependency_map_pairs / transitive_consumer_pairs;
    # the existing WARN
    # loop above is scan-map-only by design (scan-map keys are pinned
    # literals that can vanish from the tree; the discovery arms only ever
    # find on-disk tests, and a vanished transitive-consumer registration
    # is dropped by its existence check — the live-tree drift pins in
    # tests/test_select_step9c_tests.py make that staleness loud on main).
    all_pairs = sorted(
        {
            *scan_pairs,
            *rules_pin_pairs(files, work_root),
            *skills_pin_pairs(files, work_root),
            *dependency_map_pairs(files, work_root),
            *transitive_consumer_pairs(files, work_root),
        }
    )
    # #1613 zero-resolution guard (see _zero_resolution_guard): the
    # source-file-argument shape returns 2 here; the deletion-only
    # list-file shape prints its hedged WARN and falls through.
    guard_rc = _zero_resolution_guard(map_files_arg, files, all_pairs, work_root)
    if guard_rc is not None:
        return guard_rc
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
        map_timeout = recommended_timeout_s(
            k_tests,
            floor=MAP_TIMEOUT_FLOOR_S,
            dispersion=MAP_TIMEOUT_DISPERSION,
        )
        print(
            f"select_step9c_tests: map-files — {len(all_pairs)} pairs, "
            f"{len(k_tests)} tests; recommended-timeout-s={map_timeout}",
            file=sys.stderr,
        )
    for test, f in all_pairs:
        print(f"{test}\t{f}")
    return 0


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
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "emit a JSON object on stdout. Informational NOTE / WARN / sizing "
            "lines go to stderr BY DESIGN — NEVER redirect stderr into stdout "
            "(no `2>&1`) when piping stdout into a JSON parser; use "
            "`2>/dev/null` (or leave stderr on the terminal / a log file) so "
            "stdout stays pure JSON."
        ),
    )
    parser.add_argument(
        "--map-files",
        default=None,
        metavar="FILE",
        help=(
            "newline-delimited repo-relative paths: print one "
            "'test<TAB>matched_path' line per GLOB_SCAN_TESTS hit plus one "
            "'pin_test<TAB>rule_path' line per rules-pin discovery hit (#1496) "
            "plus one 'pin_test<TAB>skill_path' line per skills-pin discovery "
            "hit (#1851) "
            "plus the src/scripts import/literal/dotted/basename/transitive/stem "
            "dependency-arm pairs "
            "(#1573, #1688) plus the pinned transitive-consumer pairs (#1589; "
            "WORKFLOW_INVARIANT members excluded from all four) and exit "
            "(the /issue Step 10d merge-gate mapping mode, #1147 — skips the "
            "diff-based selection entirely; empty stdout on no match is a "
            "SUCCESS, the gate's skip signal; a zero-mapped eligible code file "
            "draws a stderr WARN, and a non-empty map prints a "
            "recommended-timeout-s=<T> stderr sizing line, floor 300s; a "
            ".py/.sh FILE argument whose lines resolve to zero repo paths "
            "with zero pairs is a usage error — exit 2, #1613; a path-list "
            "resolving to zero existing paths with zero pairs draws a hedged "
            "WARN, deletion-only payloads are benign)"
        ),
    )
    args = parser.parse_args(argv)

    # (a) fail-loud on --map-files + --json: mapping mode emits TSV (one
    # `<test>\t<matched_path>` line per pair) and the only consumers today
    # (.claude/agents/implementer.md L174; the Step 10d TG legs' `sort -u`)
    # are TSV-shaped. Silently ignoring --json here has cost a live session a
    # wasted turn (#1717 defect (a)). The gate must fail CLOSED — parser.error
    # exits 2 and prints to stderr, no diagnostic on stdout so consumers that
    # tolerate exit 2 (they should not) still get no corrupted JSON.
    if args.map_files is not None and args.json:
        parser.error(
            "--json is not supported with --map-files (mapping mode emits TSV: "
            "'<test>\\t<matched_path>' per line)"
        )

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
        return _run_map_files_mode(args.map_files, work_root)

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
            "falling back to the workflow-invariant set only. The selector diffs "
            "COMMITTED state against fetched origin/main, so uncommitted edits "
            "produce an empty diff — commit first; if this task's changes live "
            "in an issue worktree, re-run from that worktree (Step 9c contract).",
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
            # #1746: one collection-broken selected file must not abort the
            # whole gate rc=2 — pytest runs the surviving files, reports the
            # collect error as a per-file junit <error> testcase, exits rc=1,
            # and step9c_baseline compare classifies it like any other failure.
            + " --continue-on-collection-errors -v --tb=short"
        )
        for f in untested:
            print(f"untested touched file: {f}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
